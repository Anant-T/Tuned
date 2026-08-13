"""Choose which of the acquired judgments enter the corpus.

Input is the SC metadata parquet acquire.py landed (read through the
store's artifact index, so selection sees exactly what is on disk); output
is corpus/selection.jsonl - one projected row per judgment worth
extracting - plus a run_event carrying the counts.

THE SIGNALS, in the order the research ranked them
--------------------------------------------------
1. `citation` is not null. The S.C.R. is the court's OWN
   reportable-decisions publication, so a citation is a significance
   judgement made by the court rather than inferred by us. Primary signal,
   and the most overlooked one.
2. Coram >= 5. A Constitution Bench is a landmark by construction, and the
   bench size is derivable from the judge fields without reading a word of
   the judgment.
3. Membership of `opennyaiorg/InJudgements_dataset`, a pre-computed
   most-cited list stratified over 8 case types. GATED: when it is absent
   the selection still runs, and says DEGRADED rather than looking
   complete.

Case type is a STRATUM, never a filter: it decides what a capped run keeps
from each bucket, and nothing is dropped for being the wrong kind of case.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
* The upstream outcome column is corrupt - unresolved repo issue #29, 32
  documented contradictions - so it is never read. Its name does not
  appear anywhere in this file, which makes a grep a proof rather than an
  argument, and there is a test that greps. Operative outcomes come from
  the judgment's own conclusion, later in the pipeline.
* The line-1 significance flag from the court-released PDF is not a signal
  here. P0 raw-searched 70 sampled documents and found it in 2: this
  bucket ships the S.C.R. typeset reprint, which does not carry it.
  Selecting on it would silently select nothing.
* Citation in-degree. Not computable from the metadata layer (no
  citation-graph field), and graph building was costed at 3-5 days for the
  ~10% of the value the three signals above do not already capture.

Regional-language judgments are handled here, as a metadata question
(`available_languages`), rather than downstream as an extraction problem.

Build:  python -m tuned.data.select --config configs/data_law_v1.yaml
        [--years 2010-2025] [--limit N] [--out PATH] [--no-landmarks]
"""

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from tuned.data.acquire import DEV_YEARS, HF_SOURCES, SC_BUCKET, SC_SOURCE_ID, parse_year
from tuned.data.seeds import classify_case_type

SELECTION_FILENAME = "selection.jsonl"

CITATION = "citation"
CORAM = "coram"
LANDMARK = "landmark"

# Ranked, primary first. The order is the order the signals come back in,
# and the weights make it an ordering rather than a claim: the court's own
# filter outranks the other two together, so a capped or interrupted run
# keeps reported judgments before it keeps anything else.
SIGNAL_ORDER = (CITATION, CORAM, LANDMARK)
SIGNAL_WEIGHTS = {CITATION: 4, CORAM: 2, LANDMARK: 1}

CONSTITUTION_BENCH = 5

SELECTION_FIELDS = (
    "case_id",
    "title",
    "citation",
    "year",
    "court",
    "coram",
    "case_type",
    "signals",
    "priority",
    "scr_prefix",
    "pdf_key",
    "source_id",
)

# Candidate column names, tried in order. The SC parquet schema is NOT
# verified offline (18 fields, named in the research but never enumerated
# against a real file), so every read is a small ordered list rather than
# one guess - and select_corpus reports per-field coverage so the FIRST
# real run says which names actually matched instead of quietly producing
# a corpus with no Constitution Benches in it.
_CITATION_FIELDS = ("citation", "law_report_citation", "neutral_citation")
_JUDGE_FIELDS = ("judge", "author_judge", "coram_members", "bench")
_TITLE_FIELDS = ("title", "case_title", "case_id", "diary_number")
_CASE_ID_FIELDS = ("case_id", "docket_number", "diary_number", "title")
_LANGUAGE_FIELDS = ("available_languages", "language_codes", "language")
_YEAR_FIELDS = ("year", "decision_date", "date", "judgment_date", "decision_year")
_COURT_FIELDS = ("court", "court_name", "court_name_normalized")
_PDF_FIELDS = ("pdf_key", "pdf_link", "pdf_url", "raw_file_path", "file_path", "pdf_path")

# What field_coverage reports on: the columns a wrong name would quietly
# disable a signal or a filter through.
COVERAGE_FIELDS = ("citation", "judge", "author_judge", "title", "case_id", "available_languages")

ENGLISH_CODES = frozenset({"en", "eng", "english"})

# "absent" as scraped exports spell it. A literal "NA" read as a citation
# would select the entire corpus.
_NULLISH = frozenset({"", "-", "--", "na", "n/a", "nil", "none", "null", "nan"})

_JUDGE_SPLIT = re.compile(r"[,;\n\r|/]+|\band\b", re.IGNORECASE)
_HONORIFICS = re.compile(
    r"\b(hon'?ble|honou?rable|justice|chief\s+justice(?:\s+of\s+india)?|cji|"
    r"mr|mrs|ms|dr|smt|shri|sri|j)\b\.?",
    re.IGNORECASE,
)
_TITLE_NOISE = re.compile(r"\b(and|ors?|anr|another|others|etc|through|rep)\b")
_VERSUS = re.compile(r"\b(?:vs?|versus)\b\.?", re.IGNORECASE)
_YEAR_IN_TEXT = re.compile(r"(?<!\d)(1[89]\d{2}|20\d{2})(?!\d)")
# "[2020] 7 S.C.R. 941" -> volume 7, page 941 of the 2020 reports, which is
# also how the english PDFs are named ({year}_{vol}_{start}_{end}_EN.pdf).
# The year bracket is [square] by S.C.R. convention and (round) by several
# other reporters', so it is not what distinguishes them - the reporter
# name is, and matching the bracket instead would quietly accept an SCC
# citation as an S.C.R. page reference.
_SCR = re.compile(
    r"[\[(]?(\d{4})[\])]?\s+(\d{1,3})\s+S\.?\s*C\.?\s*R\.?\s+(\d{1,4})", re.IGNORECASE
)


@dataclass(frozen=True)
class Decision:
    selected: bool
    reason: str | None
    signals: tuple[str, ...]
    stratum: str
    priority: int
    year: int | None
    coram: int


# --------------------------------------------------------------------------
# Reading one metadata row.
# --------------------------------------------------------------------------

def _clean(value) -> str:
    """A field as text, with the several spellings of "absent" collapsed to ""."""
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return "" if text.lower() in _NULLISH else text


def _first(row, fields) -> str:
    for field in fields:
        text = _clean(row.get(field))
        if text:
            return text
    return ""


def citation_of(row) -> str | None:
    return _first(row, _CITATION_FIELDS) or None


def title_of(row) -> str | None:
    return _first(row, _TITLE_FIELDS) or None


def case_id_of(row) -> str | None:
    return _first(row, _CASE_ID_FIELDS) or None


def court_of(row) -> str | None:
    return _first(row, _COURT_FIELDS) or None


def _judge_key(name: str) -> str:
    """Identity key for one judge: letters only, no honorifics, no spacing.

    "Hon'ble Mr. Justice A.K. Sikri", "A K SIKRI" and "A.K.Sikri" are one
    judge. Dropping the spacing as well as the punctuation is what makes
    the initials agree, and the author field agreeing with the bench string
    is the difference between a 4-judge bench and a Constitution Bench.
    """
    return re.sub(r"[^a-z]", "", _HONORIFICS.sub(" ", name or "").lower())


def judges_of(row) -> tuple[str, ...]:
    """The bench, as written, deduped on identity across the judge fields."""
    parts: list[str] = []
    for field in _JUDGE_FIELDS:
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            parts.extend(str(item) for item in value)
        else:
            parts.extend(_JUDGE_SPLIT.split(str(value)))
    seen: set[str] = set()
    names: list[str] = []
    for part in parts:
        key = _judge_key(part)
        if not key or key in seen:
            continue
        seen.add(key)
        names.append(" ".join(part.split()))
    return tuple(names)


def coram_size(row) -> int:
    return len(judges_of(row))


def english_available(row) -> bool | None:
    """True / False / None when the row does not say.

    None is NOT False on purpose: the PDF side is already partitioned into
    english/ and regional/ prefixes, and an absent (or renamed) column must
    not empty the corpus.
    """
    for field in _LANGUAGE_FIELDS:
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            items = [str(item) for item in value]
        else:
            items = re.split(r"[,;|\s]+", str(value))
        codes = {item.strip().lower() for item in items if item.strip()}
        if not codes:
            continue
        return bool(codes & ENGLISH_CODES)
    return None


def year_of(row) -> int | None:
    """The year the judgment sits in - from the partition the file came out
    of when the reader supplied it, else from whichever date field exists."""
    for field in _YEAR_FIELDS:
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        match = _YEAR_IN_TEXT.search(_clean(value))
        if match:
            return int(match.group(1))
    return None


def case_type_of(row) -> str:
    """Stratum for this judgment - the same funnel seeds.py puts its own
    rows through, so the corpus and the seed table share one vocabulary."""
    return classify_case_type(f"{title_of(row) or ''} {case_id_of(row) or ''}")


def scr_prefix(citation: str | None) -> str | None:
    """The english PDF filename prefix a S.C.R. citation addresses.

    P0 found the english objects named {year}_{volume}_{startpage}_
    {endpage}_EN.pdf and identified that as S.C.R. pagination, so
    "[2020] 7 S.C.R. 941" addresses 2020_7_941_*. Emitted as a HINT for
    extraction, never as a filter: a citation in another reporter simply
    has no prefix.
    """
    match = _SCR.search(citation or "")
    if not match:
        return None
    year, volume, page = match.groups()
    return f"{int(year)}_{int(volume)}_{int(page)}_"


def pdf_key_of(row) -> str | None:
    """The bucket key of this judgment's PDF, if a column carries one.

    Never a filter (see the field-name note above): a run whose metadata
    schema does not carry the link still selects, and extraction falls back
    to matching scr_prefix against the keys acquire indexed.
    """
    for field in _PDF_FIELDS:
        text = _first(row, (field,))
        if not text:
            continue
        key = text.split("?", 1)[0]
        for marker in (f"{SC_BUCKET}/", "amazonaws.com/"):
            if marker in key:
                key = key.split(marker, 1)[1]
        key = key.lstrip("/")
        if key.lower().endswith(".pdf"):
            return key
    return None


# --------------------------------------------------------------------------
# The landmark list (gated).
# --------------------------------------------------------------------------

def landmark_key(title: str | None) -> str:
    """Join key for a case name.

    The most-cited list and the bucket metadata share no identifier - one
    is keyed on IndianKanoon doc urls, the other on the court's own case
    ids - so the join is by normalised title: case folded, "vs./versus"
    reduced to " v ", punctuation dropped, and the "& Ors"/"and Another"
    tail removed, since that half of a case name is written whichever way
    the reporter felt like.
    """
    text = _VERSUS.sub(" v ", (title or "").lower())
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = _TITLE_NOISE.sub(" ", text)
    return " ".join(text.split())


def landmark_keys(rows: Iterable[dict]) -> frozenset[str]:
    """Normalised titles from InJudgements rows (its column is `Titles`)."""
    keys = set()
    for raw in rows:
        key = landmark_key(_first(raw, ("Titles", "title", "case_title")))
        if key:
            keys.add(key)
    return frozenset(keys)


# --------------------------------------------------------------------------
# The decision.
# --------------------------------------------------------------------------

def select_judgment(row, *, landmarks: frozenset[str] | None = None, years=DEV_YEARS) -> Decision:
    """Does this judgment enter the corpus, and on which signals?

    Hard filters first (in scope, in English), then significance: at least
    one of the three signals must fire. `landmarks=None` means the gated
    list was unavailable, which weakens the third signal to nothing - the
    caller reports that as degraded rather than pretending otherwise.
    """
    stratum = case_type_of(row)
    coram = coram_size(row)
    year = year_of(row)

    def no(reason: str) -> Decision:
        return Decision(False, reason, (), stratum, 0, year, coram)

    if year is None:
        return no("no_year")
    if year not in years:
        return no("out_of_scope_year")
    if english_available(row) is False:
        return no("not_english")

    signals: list[str] = []
    if citation_of(row):
        signals.append(CITATION)
    if coram >= CONSTITUTION_BENCH:
        signals.append(CORAM)
    if landmarks is not None and landmark_key(title_of(row)) in landmarks:
        signals.append(LANDMARK)
    if not signals:
        return no("no_significance_signal")

    return Decision(
        True,
        None,
        tuple(signals),
        stratum,
        sum(SIGNAL_WEIGHTS[signal] for signal in signals),
        year,
        coram,
    )


def selection_row(row, decision: Decision) -> dict:
    """Project one selected judgment onto SELECTION_FIELDS.

    An EXPLICIT projection, not a copy of the row: the metadata carries
    columns that must not travel downstream at all (see the module
    docstring), and a `{**row}` would defeat the grep that proves it.
    """
    citation = citation_of(row)
    return {
        "case_id": case_id_of(row),
        "title": title_of(row),
        "citation": citation,
        "year": decision.year,
        "court": court_of(row),
        "coram": decision.coram,
        "case_type": decision.stratum,
        "signals": list(decision.signals),
        "priority": decision.priority,
        "scr_prefix": scr_prefix(citation),
        "pdf_key": pdf_key_of(row),
        "source_id": SC_SOURCE_ID,
    }


def stratified_take(rows: Sequence[dict], n: int) -> list[dict]:
    """Round-robin across case types, strongest signals first within each.

    A plain "first n" would hand the whole cap to whichever stratum the
    parquet happens to list first; the strata exist so that a capped run is
    still a spread of civil/criminal/constitutional/commercial matters.
    """
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row["case_type"], []).append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: -row["priority"])  # stable: input order breaks ties
    taken: list[dict] = []
    order = sorted(buckets)
    while len(taken) < n and any(buckets.values()):
        for name in order:
            if not buckets[name]:
                continue
            taken.append(buckets[name].pop(0))
            if len(taken) >= n:
                break
    return taken


def select_corpus(
    rows: Iterable[dict],
    *,
    landmarks: frozenset[str] | None = None,
    years=DEV_YEARS,
    limit: int | None = None,
) -> tuple[list[dict], dict]:
    """Run the selection over metadata rows; returns (selection, stats)."""
    coverage = dict.fromkeys(COVERAGE_FIELDS, 0)
    rejects: dict[str, int] = {}
    chosen: list[dict] = []
    total = 0

    for row in rows:
        total += 1
        for field in COVERAGE_FIELDS:
            if _clean(row.get(field)):
                coverage[field] += 1
        decision = select_judgment(row, landmarks=landmarks, years=years)
        if not decision.selected:
            rejects[decision.reason] = rejects.get(decision.reason, 0) + 1
            continue
        chosen.append(selection_row(row, decision))

    matched = len(chosen)
    if limit is not None and limit < matched:
        chosen = stratified_take(chosen, limit)

    by_stratum: dict[str, int] = {}
    by_signal: dict[str, int] = {}
    for row in chosen:
        by_stratum[row["case_type"]] = by_stratum.get(row["case_type"], 0) + 1
        for signal in row["signals"]:
            by_signal[signal] = by_signal.get(signal, 0) + 1

    return chosen, {
        "total": total,
        "matched": matched,
        "selected": len(chosen),
        "rejects": rejects,
        "by_stratum": by_stratum,
        "by_signal": by_signal,
        "field_coverage": coverage,
        # None means the gated list was not available at all, which is a
        # different (and weaker) run from one that had it and matched
        # nothing - and the difference is what the match rate diagnoses.
        "landmarks": None if landmarks is None else len(landmarks),
        "degraded": landmarks is None,
        "landmark_matches": by_signal.get(LANDMARK, 0),
    }


# --------------------------------------------------------------------------
# Reading what acquire indexed.
# --------------------------------------------------------------------------

def metadata_rows(store, years=DEV_YEARS) -> Iterator[dict]:
    """Every metadata row acquire.py landed, for those year partitions.

    Driven off the artifact index rather than a directory walk, so a row
    can only be read out of a file the store says is complete - and the
    year comes across from the key, which is where the partition lives.
    """
    import pyarrow.parquet as pq

    for key, artifact in sorted(store.artifact_index(SC_SOURCE_ID).items()):
        if not key.lower().endswith(".parquet"):
            continue
        year = parse_year(key)
        if year is not None and year not in years:
            continue
        for record in pq.read_table(artifact["local_path"]).to_pylist():
            if year is not None:
                record.setdefault("year", year)
            yield record


def landmark_set(store) -> frozenset[str] | None:
    """Landmark titles from the local InJudgements snapshot, or None if it
    was never acquired (gated - see acquire.py)."""
    import pyarrow.parquet as pq

    source = HF_SOURCES["injudgements"]
    rows: list[dict] = []
    found = False
    for key, artifact in sorted(store.artifact_index(source.source_id).items()):
        if not key.lower().endswith(".parquet"):
            continue
        path = artifact["local_path"]
        # Title columns ONLY: this dataset carries whole judgments, and
        # reading them to build a set of names would cost gigabytes.
        columns = [c for c in ("Titles", "title", "case_title") if c in pq.read_schema(path).names]
        if not columns:
            continue
        found = True
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    if not found:
        return None
    return landmark_keys(rows)


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

_UNSET = object()


def main(argv: Sequence[str] | None = None, *, rows=None, landmarks=_UNSET) -> int:
    import argparse

    from tuned.data.acquire import parse_years
    from tuned.data.config import load_build_config
    from tuned.data.jsonl import write_jsonl
    from tuned.data.paths import build_paths
    from tuned.data.store import Store

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data_law_v1.yaml")
    parser.add_argument("--years", default=None, help=f"default {DEV_YEARS[0]}-{DEV_YEARS[-1]}")
    parser.add_argument("--limit", type=int, default=None, help="cap, taken across strata")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--no-landmarks",
        action="store_true",
        help="ignore the InJudgements list even if it is on disk (degraded run)",
    )
    args = parser.parse_args(argv)

    years = parse_years(args.years) if args.years else DEV_YEARS
    cfg = load_build_config(args.config)
    paths = build_paths(cfg.build.workdir).ensure()
    out_path = Path(args.out) if args.out else paths.corpus_dir / SELECTION_FILENAME
    store = Store.open(paths.state_db)
    try:
        if landmarks is _UNSET:
            landmarks = None if args.no_landmarks else landmark_set(store)
        source = rows if rows is not None else metadata_rows(store, years)
        chosen, stats = select_corpus(source, landmarks=landmarks, years=years, limit=args.limit)
        written = write_jsonl(out_path, chosen)
        store.log_event("corpus_selection", {**stats, "out_path": str(out_path), "years": list(years)})

        print(f"metadata rows {stats['total']}  selected {stats['selected']} of {stats['matched']} matched")
        for signal in SIGNAL_ORDER:
            print(f"  signal {signal:<12}{stats['by_signal'].get(signal, 0):>8}")
        for stratum, count in sorted(stats["by_stratum"].items()):
            print(f"  stratum {stratum:<11}{count:>8}")
        for reason, count in sorted(stats["rejects"].items()):
            print(f"  reject[{reason}]: {count}")
        print("  field coverage: " + ", ".join(f"{k}={v}" for k, v in stats["field_coverage"].items()))
        if stats["degraded"]:
            print(
                f"  DEGRADED: selected without the landmark list. Grant access to "
                f"{HF_SOURCES['injudgements'].repo_id} "
                f"({HF_SOURCES['injudgements'].url}), re-run `python -m tuned.data.acquire "
                f"--kind hf --hf-source injudgements`, then re-run this command."
            )
        elif stats["landmarks"] and not stats["landmark_matches"]:
            print(
                f"  WARNING: {stats['landmarks']} landmark titles loaded and NONE matched - "
                f"the title join is the thing to check before trusting this run."
            )
        print(f"wrote {written} rows -> {out_path}")
        if stats["total"] and not stats["selected"]:
            # Every judgment failing all three signals is not a plausible
            # corpus, it is a column name that did not match. Exiting 0 here
            # would hand extraction an empty selection and look successful.
            print(
                "  NOTHING SELECTED from a non-empty metadata read - check the "
                "field coverage above against the parquet's real column names"
            )
            return 1
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    import os
    import sys

    exit_code = main()
    # Same reasoning as replay.py/seeds.py: pyarrow/hf-xet can leave
    # non-daemon threads that wedge interpreter shutdown after all output
    # is written. Skip shutdown entirely.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
