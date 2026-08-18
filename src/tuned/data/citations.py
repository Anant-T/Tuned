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

# Reporter citations come with or without parentheses around the year AND
# around the volume - "(1974) 2 SCR 348", "1974 2 SCR 348" and
# "1974 (2) SCR 348" are one case - and reporter tokens are written dotted or
# undotted ("S.C.C." == "SCC"). Recall here is a SAFETY property: a citation
# this module fails to extract is a citation the existence gate never checks,
# so an unmatched fabrication passes silently. Every whitespace run is bounded
# rather than \s* to keep matching linear on degenerate input - but generously,
# because citations wrap across indented lines ("(2008) 1 SCC\n        77").
_G = r"\s{0,12}"
_G1 = r"\s{1,12}"
_YEAR = r"(?:\(\s{0,2}(?P<year>\d{4})\s{0,2}\)" + _G + r"|(?<!\d)(?P<year_bare>\d{4})" + _G1 + r")"
_VOL = r"(?:\(\s{0,2}(?P<vol>\d{1,3})\s{0,2}\)|(?P<vol_bare>\d{1,3}))" + _G


def _dotted(letters: str, *, guard: bool = True) -> str:
    """"SCC" -> a pattern matching SCC, S.C.C., S. C. C. The guard stops the
    token running into a longer word (SCCX)."""
    return r"\.?\s{0,2}".join(letters) + r"\.?" + (r"(?![A-Za-z])" if guard else "")


# AIR court names. A multi-word court is only accepted when it is a name we
# recognise ("SUPREME COURT", "<name> HIGH COURT") - a free-form multi-word
# token turned "reported in AIR 1973 at page 1461" into the phantom key
# "AIR 1973 AT PAGE 1461", which then failed the existence gate and rejected a
# perfectly good example.
_AIR_COURT = (
    r"(?P<court>"
    r"Supreme" + _G1 + r"Court(?:" + _G1 + r"of" + _G1 + r"India)?"
    r"|[A-Za-z][A-Za-z.&]{0,14}" + _G1 + r"High" + _G1 + r"Court"
    r"|[A-Za-z][A-Za-z.&]{0,11}"
    r")"
)

CITATION_PATTERNS: dict[str, re.Pattern] = {
    # Supreme Court neutral citation: 2023 INSC 45
    "insc": re.compile(
        r"(?<!\d)(?P<year>\d{4})" + _G1 + r"INSC" + _G1 + r"(?P<num>\d{1,6})(?!\d)", re.IGNORECASE
    ),
    # High Court neutral citation: 2023:DHC:2720, 2023:DHC:2720-DB, 2024:KER:12345
    "hc_neutral": re.compile(
        r"(?<![\w:])(?P<year>\d{4}):(?P<court>[A-Za-z]{2,10}):(?P<num>\d{1,7})"
        r"(?:[-:](?P<suffix>[A-Za-z0-9]{1,10}))?(?![\w:])",
        re.IGNORECASE,
    ),
    # SCC OnLine: 2019 SCC OnLine SC 4321, (2019) SCC Online Del 12
    # Must precede "scc": it starts the same way but has no volume.
    "scc_online": re.compile(
        _YEAR + _dotted("SCC", guard=False) + _G + r"On\s{0,2}-?\s{0,2}Line" + _G
        + r"(?P<court>[A-Za-z]{2,12})" + _G + r"(?P<num>\d{1,6})(?!\d)",
        re.IGNORECASE,
    ),
    # Supreme Court Cases: (2008) 1 SCC 1, 2008 (1) SCC 77, (2008) 1 S.C.C. 55
    "scc": re.compile(
        _YEAR + _VOL + _dotted("SCC") + _G + r"(?P<page>\d{1,5})(?!\d)", re.IGNORECASE
    ),
    # All India Reporter: AIR 1973 SC 1461, AIR 2019 SUPREME COURT 9999.
    # (?-i:AIR) - the reporter token is case-SENSITIVE even though the rest of
    # the pattern is not, so "clean air 2019 standards mandate 45 units" is not
    # a citation.
    "air": re.compile(
        r"\b(?-i:AIR)" + _G1 + r"(?P<year>\d{4})" + _G1 + _AIR_COURT
        + _G1 + r"(?P<page>\d{1,5})(?!\d)",
        re.IGNORECASE,
    ),
    # Supreme Court Reports: (1974) 2 SCR 348, 1974 (2) S.C.R. 348
    "scr": re.compile(
        _YEAR + _VOL + _dotted("SCR") + _G + r"(?P<page>\d{1,5})(?!\d)", re.IGNORECASE
    ),
    # Criminal Law Journal: 1980 Cri LJ 1440, 1980 CriLJ 1440, 1999 Cr.L.J. 12
    "crilj": re.compile(
        _YEAR + r"(?:" + _VOL + r")?" + r"Cri?" + r"\.?\s{0,2}L\.?\s{0,2}J\.?(?![A-Za-z])"
        + _G + r"(?P<page>\d{1,6})(?!\d)",
        re.IGNORECASE,
    ),
}

# Long-form AIR court names that are the SAME court as the abbreviation the
# index is built from - without this "AIR 2019 SUPREME COURT 9999" and
# "AIR 2019 SC 9999" would be two different keys. Everything else keeps the
# court name as written (upper-cased, de-dotted, whitespace-collapsed).
_AIR_COURT_ALIASES = {"SUPREME COURT": "SC", "SUPREME COURT OF INDIA": "SC"}


def _num(raw: str) -> str:
    """Strip leading zeros without ever emptying the string: 0045 -> 45, 0 -> 0."""
    return str(int(raw))


def _year_of(m: re.Match) -> str:
    return m.group("year") or m.group("year_bare")


def _vol_of(m: re.Match) -> str | None:
    return m.group("vol") or m.group("vol_bare")


def _canon_insc(m: re.Match) -> str:
    return f"{m.group('year')} INSC {_num(m.group('num'))}"


def _canon_hc(m: re.Match) -> str:
    parts = [m.group("year"), m.group("court").upper(), _num(m.group("num"))]
    if m.group("suffix"):
        # "-DB" and ":DB" are the same bench marker; canonical form uses ":".
        parts.append(m.group("suffix").upper())
    return ":".join(parts)


def _canon_scc(m: re.Match) -> str:
    return f"({_year_of(m)}) {_num(_vol_of(m))} SCC {_num(m.group('page'))}"


def _canon_scc_online(m: re.Match) -> str:
    return f"{_year_of(m)} SCC ONLINE {m.group('court').upper()} {_num(m.group('num'))}"


def _canon_air(m: re.Match) -> str:
    court = " ".join(m.group("court").replace(".", "").upper().split())
    return f"AIR {m.group('year')} {_AIR_COURT_ALIASES.get(court, court)} {_num(m.group('page'))}"


def _canon_scr(m: re.Match) -> str:
    return f"({_year_of(m)}) {_num(_vol_of(m))} SCR {_num(m.group('page'))}"


def _canon_crilj(m: re.Match) -> str:
    # "Cri LJ" and "CrLJ" are the same reporter (Criminal Law Journal); the
    # volume is optional and only appears in the key when the citation has one.
    vol = _vol_of(m)
    volume = f"{_num(vol)} " if vol else ""
    return f"{_year_of(m)} {volume}CRI LJ {_num(m.group('page'))}"


_CANON = {
    "insc": _canon_insc,
    "hc_neutral": _canon_hc,
    "scc_online": _canon_scc_online,
    "scc": _canon_scc,
    "air": _canon_air,
    "scr": _canon_scr,
    "crilj": _canon_crilj,
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


def _known_spans(text: str) -> list[tuple[int, int, str, re.Match]]:
    """Non-overlapping (start, end, key, match) for every known format, in
    text order. Overlaps between patterns are resolved leftmost-longest so a
    span is only ever counted once."""
    spans: list[tuple[int, int, int, str, re.Match]] = []
    seq = 0
    for key, pattern in CITATION_PATTERNS.items():
        for m in pattern.finditer(text):
            spans.append((m.start(), -(m.end() - m.start()), seq, key, m))
            seq += 1
    spans.sort(key=lambda t: (t[0], t[1], t[2]))

    kept: list[tuple[int, int, str, re.Match]] = []
    last_end = -1
    for start, _neg_len, _seq, key, m in spans:
        if start < last_end:
            continue
        last_end = m.end()
        kept.append((start, m.end(), key, m))
    return kept


def extract_citations(text: str) -> list[str]:
    """Every citation in `text`, normalized, in order of first appearance,
    deduped."""
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for _start, _end, key, m in _known_spans(text):
        value = _CANON[key](m)
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


# A citation-SHAPED string: year, optional volume, a reporter-looking run of
# capitalised tokens, then a page. Deliberately NOT re.IGNORECASE - every
# reporter word must start with a capital, which is what keeps ordinary prose
# ("in 2023 the court awarded 5 crore") out.
_SUSPECT_RE = re.compile(
    r"(?<!\d)(?:\(\s{0,2}\d{4}\s{0,2}\)|\d{4})" + _G
    + r"(?:(?:\(\s{0,2}\d{1,3}\s{0,2}\)|\d{1,3})" + _G + r")?"
    r"(?P<reporter>[A-Z][A-Za-z.&]{0,11}(?:\s{1,2}[A-Z][A-Za-z.&]{0,11}){0,3})" + _G
    + r"(?P<page>\d{1,6})(?!\d)"
)
_TWO_CAPS_RE = re.compile(r"[A-Z]{2}")


def suspect_citations(text: str) -> list[str]:
    """Citation-SHAPED strings that no known pattern matched - unmodelled
    reporters (KLT, MhLJ, Bom CR, ...) and invented ones.

    This is the SECOND channel of the existence gate and it exists because
    silence is dangerous: a string extract_citations() cannot parse is a
    string the index is never asked about, so a fabrication in an unknown
    format would otherwise pass the hard gate untouched. The gate layer is
    expected to reject-on-unknown (optionally diffing against
    suspect_citations(context) first, since a suspect that came in with the
    grounding passage is not the model's invention).

    Returns OPAQUE keys - normalize()'s unknown-format path, i.e. whitespace-
    collapsed and upper-cased source text. They are only reliably comparable
    to OTHER suspect keys: never pass one to CitationIndex.contains(), because
    the index holds canonical reporter forms and an opaque key will miss
    against it for spelling reasons rather than existence reasons. Compare
    suspects against suspect_citations(context), nothing else.

    Order-preserving and deduped.
    """
    if not text:
        return []
    taken = [(start, end) for start, end, _key, _m in _known_spans(text)]
    out: list[str] = []
    seen: set[str] = set()
    for m in _SUSPECT_RE.finditer(text):
        if any(m.start() < end and start < m.end() for start, end in taken):
            continue
        reporter = m.group("reporter").replace(".", "").replace(" ", "")
        # A reporter abbreviation always carries an all-caps run somewhere
        # (LJ, SCC, CR, KLT, MhLJ); "Delhi High Court" style prose does not.
        if not _TWO_CAPS_RE.search(reporter):
            continue
        value = normalize(m.group(0))
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


_SUSPECT_KEY_TOKEN_RE = re.compile(r"\d+|[A-Z]+")


def suspect_key(value: str) -> str:
    """The form two suspect citations must share to be THE SAME citation.

    suspect_citations() returns normalize()'s opaque form - whitespace
    COLLAPSED and upper-cased - and the gate layer diffs the output's suspects
    against the grounding's as raw strings. That diff is blind to the two ways
    an unmodelled reporter is legitimately re-typed, and both cost a PERMANENT
    reject in the pilot (citations is in gates.PERMANENT_GATES, so the seed is
    burned without the attempts being spent):

        grounding  '2015(4) KLT 163(LB)'   output  '2015 (4) KLT 163'
        grounding  '[2006 (7) SCALE 28 ]'  output  '(2006) 7 SCALE 28'

    Both are the same case. The first pair differs by ONE SPACE before a
    bracket; the second by which of the year and the volume is parenthesised -
    a re-rendering into standard form, i.e. a teacher doing its job. Neither
    is a fabrication and neither survived a raw string comparison.

    So the key keeps the TOKENS and discards the punctuation between them:
    upper-case, drop the full stops that separate reporter letters ('A.C.' and
    'AC' are one reporter, which _SUSPECT_RE's own reporter test already
    assumes when it strips them), then take the maximal runs of digits and
    letters in order and join them with a single space.

        '2015 (4) KLT 163'    -> '2015 4 KLT 163'
        '2015(4) KLT 163'     -> '2015 4 KLT 163'
        '(2006) 7 SCALE 28'   -> '2006 7 SCALE 28'
        '2006 (7) SCALE 28'   -> '2006 7 SCALE 28'

    JOINED, NOT CONCATENATED, and the separator is load-bearing: '2015 (4) KLT
    163' and '2015 (41) KLT 63' both concatenate to the same digits either side
    of the reporter, and a key that dropped the boundary would fold two
    different cases into one - on a permanent gate, in the direction that lets
    a fabrication through. With the boundary kept they stay '2015 4 KLT 163'
    and '2015 41 KLT 63'.

    THIS DOES NOT WEAKEN THE ABSENT CASE. The third pilot reject, '(1955) I LLJ
    688' attached to Shivnandan Sharma v. Punjab National Bank, was a citation
    from the teacher's memory: its grounding carried NO suspects at all, so
    there is nothing for any key to match it against and it is still rejected.
    Folding only ever equates two strings that are both present.

    Comparable to other suspect keys and to nothing else - never pass one to
    CitationIndex.contains(), for the reason suspect_citations() gives.
    """
    return " ".join(_SUSPECT_KEY_TOKEN_RE.findall((value or "").upper().replace(".", "")))


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
