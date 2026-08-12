"""Citation-existence index for the law_v1 curation pipeline.

Gate role: a synthesized answer may only cite authorities that either (a)
appear in the citation index built from a real case corpus, or (b) were
already present in the grounding context handed to the generator. Anything
else is an invented authority and kills the example - reject, never repair.
novel_citations() is that gate's primitive.

Normalization is what makes the check work at all: the same judgment is
written "2023 INSC 45" and "2023 INSC 0045", "(2008) 1 SCC 1" and
"(2008)  1  SCC  1" in the wild, so every citation on BOTH sides of every
comparison passes through normalize() before anything is compared.

Index file format: sorted, newline-delimited, utf-8 text (LF). Deliberately
not JSON or sqlite - it diffs, greps, streams line-by-line, and survives a
hand edit.

Build:  python -m tuned.data.citations --build --config configs/data_law_v1.yaml
        [--source kanoongpt] [--out PATH]
"""

import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

INDEX_FILENAME = "citation_index.txt"

# Columns read off the KanoonGPT rows. headnote_text is COPYRIGHTED editorial
# matter (SCC/AIR headnotes) - it is never read, never indexed, never stored.
_CITATION_COLUMNS = ("neutral_citation", "law_report_citation")
_FORBIDDEN_COLUMNS = ("headnote_text",)


# --------------------------------------------------------------------------
# Patterns.
#
# Every numeric field is fenced with (?<!\d) / (?!\d) rather than \b so a
# citation can never be carved out of the middle of a longer number:
# "2023 INSC 4512" yields 4512 (not 45), and "12023 INSC 45" yields nothing.
# --------------------------------------------------------------------------

# Reporter citations come with or without parentheses around the year -
# "(1974) 2 SCR 348" and "1974 2 SCR 348" are the same case. The bare branch
# demands whitespace after the year so "20081 SCC 1" cannot be read as
# year 2008 + volume 1; the parenthesised branch does not need it.
_YEAR = r"(?:\(\s*(?P<year>\d{4})\s*\)\s*|(?<!\d)(?P<year_bare>\d{4})\s+)"

CITATION_PATTERNS: dict[str, re.Pattern] = {
    # Supreme Court neutral citation: 2023 INSC 45
    "insc": re.compile(r"(?<!\d)(?P<year>\d{4})\s+INSC\s+(?P<num>\d{1,6})(?!\d)", re.IGNORECASE),
    # High Court neutral citation: 2023:DHC:2720, 2023:DHC:2720-DB, 2024:KER:12345
    "hc_neutral": re.compile(
        r"(?<![\w:])(?P<year>\d{4}):(?P<court>[A-Za-z]{2,10}):(?P<num>\d{1,7})"
        r"(?:[-:](?P<suffix>[A-Za-z0-9]{1,10}))?(?![\w:])",
        re.IGNORECASE,
    ),
    # Supreme Court Cases: (2008) 1 SCC 1
    "scc": re.compile(_YEAR + r"(?P<vol>\d{1,3})\s*SCC\s*(?P<page>\d{1,5})(?!\d)", re.IGNORECASE),
    # All India Reporter: AIR 1973 SC 1461
    "air": re.compile(
        r"\bAIR\s+(?P<year>\d{4})\s+(?P<court>[A-Za-z][A-Za-z.&]{0,7})\s+(?P<page>\d{1,5})(?!\d)",
        re.IGNORECASE,
    ),
    # Supreme Court Reports: (1974) 2 SCR 348
    "scr": re.compile(_YEAR + r"(?P<vol>\d{1,3})\s*SCR\s*(?P<page>\d{1,5})(?!\d)", re.IGNORECASE),
}


def _num(raw: str) -> str:
    """Strip leading zeros without ever emptying the string: 0045 -> 45, 0 -> 0."""
    return str(int(raw))


def _year_of(m: re.Match) -> str:
    return m.group("year") or m.group("year_bare")


def _canon_insc(m: re.Match) -> str:
    return f"{m.group('year')} INSC {_num(m.group('num'))}"


def _canon_hc(m: re.Match) -> str:
    parts = [m.group("year"), m.group("court").upper(), _num(m.group("num"))]
    if m.group("suffix"):
        # "-DB" and ":DB" are the same bench marker; canonical form uses ":".
        parts.append(m.group("suffix").upper())
    return ":".join(parts)


def _canon_scc(m: re.Match) -> str:
    return f"({_year_of(m)}) {_num(m.group('vol'))} SCC {_num(m.group('page'))}"


def _canon_air(m: re.Match) -> str:
    court = m.group("court").replace(".", "").upper()
    return f"AIR {m.group('year')} {court} {_num(m.group('page'))}"


def _canon_scr(m: re.Match) -> str:
    return f"({_year_of(m)}) {_num(m.group('vol'))} SCR {_num(m.group('page'))}"


_CANON = {
    "insc": _canon_insc,
    "hc_neutral": _canon_hc,
    "scc": _canon_scc,
    "air": _canon_air,
    "scr": _canon_scr,
}


def normalize(raw: str) -> str:
    """Canonical index form for a single citation string.

    Known reporters are rewritten to one exact spelling (leading zeros
    stripped, reporter/court tokens upper-cased, SCC/SCR years always
    parenthesised). Anything unrecognised keeps its text but gets
    whitespace-collapsed and upper-cased, so it is still usable as an opaque
    index key - an unknown-format citation must never silently become a
    DIFFERENT key on the two sides of a comparison.
    """
    s = " ".join((raw or "").split())
    if not s:
        return ""
    for key, pattern in CITATION_PATTERNS.items():
        m = pattern.fullmatch(s)
        if m is not None:
            return _CANON[key](m)
    return s.upper()


def extract_citations(text: str) -> list[str]:
    """Every citation in `text`, normalized, in order of first appearance,
    deduped. Overlapping matches (one pattern eating another's span) are
    resolved leftmost-longest so a span is only ever counted once."""
    if not text:
        return []
    spans: list[tuple[int, int, int, str, re.Match]] = []
    seq = 0
    for key, pattern in CITATION_PATTERNS.items():
        for m in pattern.finditer(text):
            spans.append((m.start(), -(m.end() - m.start()), seq, key, m))
            seq += 1
    spans.sort(key=lambda t: (t[0], t[1], t[2]))

    out: list[str] = []
    seen: set[str] = set()
    last_end = -1
    for start, _neg_len, _seq, key, m in spans:
        if start < last_end:
            continue
        last_end = m.end()
        value = _CANON[key](m)
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


class CitationIndex:
    """In-memory set of canonical citation keys, backed by a sorted text file."""

    def __init__(self, entries: set[str], path: Path | None = None):
        self._entries = entries
        self.path = path

    @classmethod
    def build(cls, citations: Iterable[str], out_path: str | Path) -> "CitationIndex":
        entries = {key for key in (normalize(c) for c in citations if c) if key}
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        # newline="\n": the index is a portable artifact, never CRLF even when
        # it is built on Windows.
        with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
            for key in sorted(entries):
                f.write(key + "\n")
        os.replace(tmp_path, out_path)
        return cls(entries, out_path)

    @classmethod
    def load(cls, path: str | Path) -> "CitationIndex":
        """Lines are canonical by construction (build() normalized them), so
        load does NOT re-normalize - on a 17M-row index that would cost
        minutes for no gain. contains() normalizes the QUERY instead, which
        is the side that arrives raw from model output."""
        path = Path(path)
        entries = set()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.add(line)
        return cls(entries, path)

    def contains(self, raw_or_normalized: str) -> bool:
        key = normalize(raw_or_normalized)
        return bool(key) and key in self._entries

    def __contains__(self, raw_or_normalized: str) -> bool:
        return self.contains(raw_or_normalized)

    def __len__(self) -> int:
        return len(self._entries)


def novel_citations(text: str, allowed_context: str, index: CitationIndex) -> list[str]:
    """THE GATE PRIMITIVE. Citations in `text` that are neither carried by the
    grounding context nor present in the index - i.e. authorities the model
    produced from nowhere. A non-empty return means: reject the example."""
    allowed = set(extract_citations(allowed_context or ""))
    return [c for c in extract_citations(text or "") if c not in allowed and not index.contains(c)]


# --------------------------------------------------------------------------
# Corpus ingestion (CLI only - no network at import time).
# --------------------------------------------------------------------------

def citations_from_row(row: dict) -> list[str]:
    """Citation strings carried by one corpus row.

    Reads ONLY _CITATION_COLUMNS. headnote_text is copyrighted editorial
    matter and is never touched, here or anywhere else. A column value that
    parses as one or more known formats contributes those; a value that
    parses as none contributes its opaque normalized form (a real citation in
    a reporter we do not model yet is still worth having in the index).
    """
    out: list[str] = []
    for column in _CITATION_COLUMNS:
        value = row.get(column)
        if not isinstance(value, str) or not value.strip():
            continue
        found = extract_citations(value)
        out.extend(found if found else [normalize(value)])
    return out


def _stream_kanoongpt() -> Iterator[str]:
    # Lazy import: `import tuned.data.citations` must never touch the network
    # or drag in datasets/pyarrow (same discipline as replay.py / smoke.py).
    from datasets import load_dataset

    ds = load_dataset("KanoonGPT/indian-case-laws", split="train", streaming=True)
    for row in ds:
        yield from citations_from_row(row)


_SOURCES = {"kanoongpt": _stream_kanoongpt}


if __name__ == "__main__":
    import argparse
    import sys

    from tuned.data.config import load_build_config
    from tuned.data.paths import build_paths

    p = argparse.ArgumentParser(description="Build the citation-existence index.")
    p.add_argument("--build", action="store_true", help="build the index (the only mode)")
    p.add_argument("--config", default="configs/data_law_v1.yaml")
    p.add_argument("--source", default="kanoongpt", choices=sorted(_SOURCES))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if not args.build:
        p.error("nothing to do: pass --build")

    cfg = load_build_config(args.config)
    out = Path(args.out) if args.out else build_paths(cfg.build.workdir).ensure().corpus_dir / INDEX_FILENAME

    # Memory note: the whole key set is held in RAM to dedupe before the sorted
    # write. ~17M rows of KanoonGPT land in the low GBs - run this on the build
    # box, never inside a training job.
    index = CitationIndex.build(_SOURCES[args.source](), out)
    print(f"indexed {len(index)} distinct citations -> {out}")

    # Same reasoning as replay.py: abandoned streaming iterators leave
    # non-daemon datasets/hf-xet threads that wedge interpreter shutdown after
    # all output is written. Skip shutdown entirely.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
