"""Hold out the eval split at CASE level, so no two rows about one case
can land on opposite sides of the train/eval boundary.

Input is `out/deduped.jsonl` - dedupe.py's output, verified against the digest
dedupe.py recorded for it. Output is two JSONL files, `out/split_train.jsonl`
and `out/split_eval.jsonl`, plus a manifest that carries dedupe's own manifest
(and through it decontamination's) forward whole, so stats.py can say from the
chain alone that the rows it grades were screened, deduped and split.

THE PLAN SAYS "CNR-LEVEL ASSIGNMENT BEFORE ROW BUILD"
-----------------------------------------------------
Honoured as: assignment keys on the CASE, never on the row. Rows are grouped
into assignment atoms first and a whole atom goes to one side, so sibling rows
cannot separate no matter what order they arrive in or how the target is
filled. "Before row build" was the plan's way of saying the same thing at a
stage where the rows did not exist yet; the rows exist by the time this module
runs, and grouping them is what makes the property hold rather than a hope.

The atom is `dedupe.case_id_of` - IMPORTED, not reimplemented, because split
buckets and per-case cap buckets have to be the same buckets. A row with no
case identifier is its own atom, keyed by its content.

Then the disjointness is ASSERTED ANYWAY, on the finished sides, and a
straddle is a refusal rather than a log line. By construction it cannot happen;
the assert is there because "by construction" is a claim about code that can be
edited, and the failure it guards is invisible in the output (an eval score
quietly inflated by a case the model trained on).

WHERE THE ASSERT STOPS, stated rather than implied. It compares the atom each
side used, which is the STRONGEST identifier each row carries. One real case
reachable under two different strongest identifiers - a row with a CNR and a
citation, and a sibling with the citation only - is two atoms to dedupe's cap
as well as to this pass, and neither can see that they are one case. Widening
the assert to every identifier is not the remedy: `identifiers_from_text` also
extracts the authorities a passage CITES, so nearly every landmark citation
appears on both sides of any honest split and the assert would fire on every
run. The manifest counts `cross_side_identifiers` instead, so the size of the
hole is a number rather than a worry.

CHRONOLOGICALLY LATER PREFERRED, AND WHAT ACTUALLY CARRIES A DATE
-----------------------------------------------------------------
Measured against the code that builds the rows, not assumed:

  * `_prov` CARRIES NO DATE FIELD TODAY. seeds.py writes `decision_date: None`
    in all three of its converters (predex, tathyanyaya, injudgements), and
    `decontaminate.generated_rows` - which is what turns an accepted generation
    into a row - does not copy `decision_date` into `_prov` at all. replay.py
    and curated.py write `_prov` blocks of exactly
    {source, license, native_id, reasoning}. So a date channel that read
    `_prov` alone would be inert on every row this build can currently produce,
    and would have looked like a working preference.
  * THE CASE IDENTIFIER CARRIES A YEAR, and it is the channel that works. A
    `cnr:` key is 4 letters + 2 digits + 6 digits + a 4-DIGIT YEAR
    (`ESCR010004512020` -> 2020). A `cit:` key is citations.normalize's
    canonical form, and every one of the seven canonical forms puts a 4-digit
    year in it - `2023 INSC 45`, `2023 DELHI:45`, `(2020) 7 SCC 1`,
    `AIR 1960 SC 30`, `(2020) 7 SCR 941`, `1998 3 CRI LJ 45`,
    `2023 SCC ONLINE DEL 45`. A `title:` key carries none.

So the ladder is: an explicit `_prov` date field first (it is the only one that
can be more precise than a year, and it is where a future seed builder should
put one), then the year in the case identifier. The manifest records how many
eval cases came from each channel, which is the instrument that says when the
`_prov` channel starts carrying anything.

THE DATE IS THE CASE'S OWN, NOT ITS CITATION GRAPH'S. Only the identifier the
row is BUCKETED under is read, never the other identifiers on the row: a 2020
judgment discussing a 1960 authority carries `cit:AIR 1960 SC 30` in its
identifier set, and a channel that read every identifier would date the case to
1960 and put it in train as "old". A case whose bucket is itself a cited
authority (possible when identifiers come from the text - see dedupe's
`--no-case-id-from-text`) is dated as that authority; the flag decontamination
recorded travels in the manifest so a reader can tell.

DETERMINISM: CONTENT-KEYED, NOTHING ELSE
-----------------------------------------
No RNG, no clock, no iteration order. Dated atoms fill the eval side
newest-first; whatever is left to reach the fraction comes from the date-less
atoms in order of sha256 over the atom's key, which is derived from the row's
content. The same input bytes produce the same assignment forever, and shuffling
the input file's lines produces the same assignment too - the atoms are built by
grouping, ordered by date and hash, and the target is filled by walking that
order, so nothing in the decision can see what line a row arrived on. Pinned in
both directions by the shuffle test.

Build:  python -m tuned.data.split --config configs/data_law_v1.yaml
        [--in PATH] [--out-train PATH] [--out-eval PATH]
"""

import hashlib
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from tuned.data.decontaminate import Item, row_prov, stream_items
from tuned.data.dedupe import MANIFEST_FILENAME as DEDUPE_MANIFEST_FILENAME
from tuned.data.dedupe import OUT_FILENAME as DEDUPE_OUT_FILENAME
from tuned.data.dedupe import case_id_of

TRAIN_FILENAME = "split_train.jsonl"
EVAL_FILENAME = "split_eval.jsonl"
MANIFEST_FILENAME = "split.json"

# 1  the first version. Case-level assignment over dedupe.case_id_of atoms,
#    newest-first from the case identifier's year with a content-keyed hash
#    filling the remainder, disjointness and drop-nothing asserted as refusals,
#    and dedupe's manifest carried forward against its output digest.
SPLIT_VERSION = 1

# The `_prov` fields that could carry a date, strongest first: a full date
# beats a bare year, and an explicit decision date beats an unqualified one.
# NONE OF THESE IS WRITTEN BY ANYTHING IN THE BUILD TODAY (see the module
# docstring) - they are the shape a seed builder should fill, and the manifest
# counts how often the channel fires so that stays a measurement.
PROV_DATE_FIELDS = ("decision_date", "judgment_date", "decided_on", "date",
                    "decision_year", "year")

DATE_FROM_PROV = "prov"
DATE_FROM_CITATION = "citation"
DATE_FROM_CNR = "cnr"
DATE_FROM_NONE = "none"
DATE_CHANNELS = (DATE_FROM_PROV, DATE_FROM_CITATION, DATE_FROM_CNR, DATE_FROM_NONE)

# A four-digit run that could be a year of an Indian law report. The floor is
# not decoration: `(2005) 7 SCC 510` normalises with a 3-digit page and
# `1234 OF 2019` style docket text yields a leading run that is not a year, so
# the first run IN RANGE is taken rather than the first run.
_YEAR_MIN, _YEAR_MAX = 1800, 2199
_FOUR_DIGITS = re.compile(r"(?<!\d)(\d{4})(?!\d)")
# CNR: 16 characters, and the last four are the year of institution.
_CNR_KEY = re.compile(r"^[A-Za-z]{4}\d{12}$")

_ISO_DAY = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_ISO_MONTH = re.compile(r"^(\d{4})-(\d{2})$")
_BARE_YEAR = re.compile(r"^(\d{4})$")


class SplitRefusal(RuntimeError):
    """Nothing was written and no output carries a split stamp."""


class StraddlingCase(SplitRefusal):
    """One case identifier reached both sides. The eval score would be a lie."""


class RowsLost(SplitRefusal):
    """Rows in != train + eval. A split is a partition or it is a filter."""


class DegenerateSplit(SplitRefusal):
    """One side came out empty when the fraction asked for rows on both."""


# --------------------------------------------------------------------------
# Dates: what a row can actually be dated by.
# --------------------------------------------------------------------------

def date_key(value) -> str | None:
    """A sortable YYYY-MM-DD key for whatever a date field holds, or None.

    A bare year becomes `YYYY-00-00` so it compares against full dates in the
    same year without being mistaken for one - and it sorts EARLIEST in its
    year, which is the conservative direction: a case known only by its year is
    less likely to be pulled into the newest-first eval side than one with a
    real date, rather than more.
    """
    if value is None:
        return None
    text = str(value).strip()[:10]
    for pattern, filler in ((_ISO_DAY, ""), (_ISO_MONTH, "-00"), (_BARE_YEAR, "-00-00")):
        match = pattern.match(text)
        if match is None:
            continue
        year = int(match.group(1))
        if not (_YEAR_MIN <= year <= _YEAR_MAX):
            return None
        return text + filler
    return None


def prov_date(prov) -> str | None:
    """The first usable date in `_prov`, in PROV_DATE_FIELDS order."""
    for field in PROV_DATE_FIELDS:
        key = date_key((prov or {}).get(field))
        if key is not None:
            return key
    return None


def year_in(text: str) -> str | None:
    """The first four-digit run in `text` that could be a year, or None."""
    for match in _FOUR_DIGITS.finditer(text or ""):
        if _YEAR_MIN <= int(match.group(1)) <= _YEAR_MAX:
            return match.group(1)
    return None


def case_id_date(case_id: str | None) -> tuple[str | None, str]:
    """(date key, channel) from the identifier the row is bucketed under.

    `cnr:` is positional - the LAST four characters of the 16-character key are
    the year, and reading the first four-digit run instead would take the
    6-digit case number's leading digits. `cit:` is scanned, because the year
    sits first in six of the seven canonical forms and second in `AIR 1960 SC
    30`. `title:` carries no year at all.
    """
    if not case_id:
        return None, DATE_FROM_NONE
    namespace, _, value = case_id.partition(":")
    if namespace == "cnr":
        if _CNR_KEY.match(value):
            key = date_key(value[-4:])
            if key is not None:
                return key, DATE_FROM_CNR
        return None, DATE_FROM_NONE
    if namespace == "cit":
        year = year_in(value)
        if year is not None:
            return f"{year}-00-00", DATE_FROM_CITATION
    return None, DATE_FROM_NONE


def item_date(item: Item, case_id: str | None) -> tuple[str | None, str]:
    """(date key, channel) for one row: `_prov` first, then its case id."""
    key = prov_date(row_prov(item.row))
    if key is not None:
        return key, DATE_FROM_PROV
    return case_id_date(case_id)


# --------------------------------------------------------------------------
# The assignment atoms.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Unit:
    """One case (or one case-less row), and every row that belongs to it.

    `indexes` are positions in the input sequence, so a unit can put its rows
    back on the side it was assigned to without carrying the rows themselves,
    and the drop-nothing check is a counting argument over the same indexes.
    """

    key: str
    case_id: str | None
    date: str | None
    channel: str
    indexes: tuple[int, ...]

    @property
    def rows(self) -> int:
        return len(self.indexes)

    @property
    def order_hash(self) -> str:
        """sha256 of the atom's key - the content-keyed order for date-less
        atoms. The KEY, not the row bytes: every row of one case must share an
        order position, and the key is what they share."""
        return hashlib.sha256(self.key.encode("utf-8")).hexdigest()


def units_of(items: Sequence[Item]) -> list[Unit]:
    """Group rows into assignment atoms - one per case, one per case-less row.

    A case-less row is keyed by its own content id rather than by its position,
    so shuffling the input cannot move it. Two case-less rows with the SAME
    content id become one atom, which is harmless (they go to one side, which
    is what the case rule wants anyway) and is what keeps the key content-bound.

    A case's date is the NEWEST any of its rows names. A case is as recent as
    the most recent thing known about it; taking the oldest would let one
    under-annotated row drag a whole case out of the newest-first band.
    """
    order: list[str] = []
    grouped: dict[str, list[int]] = {}
    facts: dict[str, tuple[str | None, str | None, str]] = {}
    for index, item in enumerate(items):
        case_id = case_id_of(item)
        key = case_id if case_id is not None else f"row:{item.key}"
        date, channel = item_date(item, case_id)
        if key not in grouped:
            order.append(key)
            grouped[key] = []
            facts[key] = (case_id, date, channel)
        else:
            known_case, known_date, _ = facts[key]
            if date is not None and (known_date is None or date > known_date):
                facts[key] = (known_case, date, channel)
        grouped[key].append(index)
    return [
        Unit(key=key, case_id=facts[key][0], date=facts[key][1], channel=facts[key][2],
             indexes=tuple(grouped[key]))
        for key in order
    ]


def eval_target(total_rows: int, fraction: float) -> int:
    """How many rows the eval side is aiming for.

    Half-up rather than Python's round(), which is half-to-even: at 45 rows and
    0.10 the two disagree (4 against 5), and a split target that rounds one way
    at 45 rows and the other way at 55 is a rule nobody can state.
    """
    return math.floor(total_rows * fraction + 0.5)


def ordered_units(units: Sequence[Unit]) -> tuple[list[Unit], list[Unit]]:
    """(dated atoms newest-first, date-less atoms by content hash).

    Two sorts rather than one composite key: the primary is DESCENDING and the
    tie-break ASCENDING, and Python's sort is stable, so ordering by the
    tie-break and then re-sorting by date reversed gives exactly that without
    inventing a negation for a string. Ties break on the content hash rather
    than on the key so that a court's identifier prefix cannot decide which of
    two same-day cases is held out.
    """
    dated = sorted((u for u in units if u.date is not None),
                   key=lambda u: (u.order_hash, u.key))
    dated.sort(key=lambda u: u.date, reverse=True)
    dateless = sorted((u for u in units if u.date is None),
                      key=lambda u: (u.order_hash, u.key))
    return dated, dateless


def assign_units(items: Sequence[Item], *, fraction: float) -> tuple[list[int], list[int], dict]:
    """(train indexes, eval indexes, stats) - whole atoms, newest first.

    The walk stops as soon as the eval side has reached the target, so it can
    overshoot by at most one atom's worth of rows minus one (at most two rows,
    with dedupe's cap of three in force). Overshooting is preferred to
    undershooting: the fraction is a floor on how much is held out, and a rule
    that stopped short would silently shrink the eval set whenever the newest
    case happened to be large.
    """
    units = units_of(items)
    total_rows = sum(u.rows for u in units)
    target = eval_target(total_rows, fraction)
    dated, dateless = ordered_units(units)

    chosen: list[Unit] = []
    rows = 0
    for unit in [*dated, *dateless]:
        if rows >= target:
            break
        chosen.append(unit)
        rows += unit.rows

    picked = {u.key for u in chosen}
    eval_indexes = sorted(i for u in chosen for i in u.indexes)
    train_indexes = sorted(i for u in units if u.key not in picked for i in u.indexes)
    date_assigned = [u for u in chosen if u.date is not None]
    stats = {
        "rows": total_rows,
        "units": len(units),
        "cases": sum(1 for u in units if u.case_id is not None),
        "caseless_rows": sum(u.rows for u in units if u.case_id is None),
        "eval_target_rows": target,
        "eval_rows": len(eval_indexes),
        "train_rows": len(train_indexes),
        "eval_units": len(chosen),
        "date_assigned_units": len(date_assigned),
        "hash_assigned_units": len(chosen) - len(date_assigned),
        # The oldest date that still made the eval side - everything newer than
        # this is held out, which is the sentence the temporal split is for.
        # None when no dated atom was taken at all.
        "date_boundary": min((u.date for u in date_assigned), default=None),
        "dated_units": sum(1 for u in units if u.date is not None),
        "by_date_channel": {
            channel: sum(1 for u in units if u.channel == channel)
            for channel in DATE_CHANNELS
        },
    }
    stats["eval_fraction_achieved"] = (
        round(len(eval_indexes) / total_rows, 6) if total_rows else 0.0
    )
    return train_indexes, eval_indexes, stats


# --------------------------------------------------------------------------
# The invariants. Each one is a refusal.
# --------------------------------------------------------------------------

def assert_disjoint(train: Sequence[Item], evaluation: Sequence[Item]) -> None:
    """Refuse if any case identifier reached both sides."""
    train_cases = {c for c in (case_id_of(i) for i in train) if c is not None}
    eval_cases = {c for c in (case_id_of(i) for i in evaluation) if c is not None}
    straddling = sorted(train_cases & eval_cases)
    if straddling:
        raise StraddlingCase(
            f"{len(straddling)} case identifier(s) reached BOTH sides of the split: "
            f"{', '.join(straddling[:10])}"
            f"{'...' if len(straddling) > 10 else ''}. Every eval row about a case the "
            f"model trained on inflates the eval score by an amount nothing downstream can "
            f"measure, so nothing is written."
        )


def assert_nothing_dropped(total: int, train: Sequence, evaluation: Sequence) -> None:
    """Refuse unless the two sides are a partition of the input."""
    if len(train) + len(evaluation) != total:
        raise RowsLost(
            f"read {total} rows and wrote {len(train)} + {len(evaluation)} = "
            f"{len(train) + len(evaluation)}. A split is a partition; this one is a "
            f"filter, and the rows it lost are lost silently."
        )


def assert_both_sides_populated(stats: dict) -> None:
    """Refuse an empty side when the fraction asked for rows on both."""
    if stats["rows"] == 0:
        raise DegenerateSplit(
            "the input holds no rows. Exiting 0 here would report an empty corpus as split."
        )
    if stats["eval_target_rows"] > 0 and stats["eval_rows"] == 0:
        raise DegenerateSplit(
            f"the eval side came out EMPTY against a target of {stats['eval_target_rows']} "
            f"rows. There is no held-out set, and a build that ships one anyway has no "
            f"honest number to report."
        )
    if stats["train_rows"] == 0:
        raise DegenerateSplit(
            f"the TRAIN side came out empty - all {stats['rows']} rows are one atom's "
            f"worth, or the fraction is 1. There is nothing to train on."
        )


def cross_side_identifiers(train: Sequence[Item], evaluation: Sequence[Item]) -> int:
    """How many `cnr:` identifiers appear on both sides without being an atom.

    NOT a gate - an instrument. A CNR is a case's own number and never a
    citation graph edge, so a CNR on both sides means one real case that the
    atom rule split under two different strongest identifiers. Counted so the
    size of the assert's known blind spot is a number. (`cit:` is excluded
    deliberately: the authorities a passage cites are extracted too, so a
    landmark citation appears on both sides of every honest split.)
    """
    def cnrs(items):
        return {i for item in items for i in item.identifiers if i.startswith("cnr:")}

    return len(cnrs(train) & cnrs(evaluation))


def split_items(items: Sequence[Item], *, fraction: float, assign=assign_units):
    """(train items, eval items, stats), or a refusal.

    `assign` is a seam so a test can drive the invariants with an assigner that
    breaks them - a gate whose failure branch nothing can reach is a gate
    nobody has checked.
    """
    train_indexes, eval_indexes, stats = assign(items, fraction=fraction)
    train = [items[i] for i in train_indexes]
    evaluation = [items[i] for i in eval_indexes]
    # The counting check first: it is the one that catches an assigner that
    # dropped or duplicated an index, and the two below read cleaner once the
    # sides are known to be a partition.
    assert_nothing_dropped(len(items), train, evaluation)
    if sorted([*train_indexes, *eval_indexes]) != list(range(len(items))):
        raise RowsLost(
            "the two sides hold the right NUMBER of rows and not the right rows: some "
            "index is on both sides or on neither, which a count alone cannot see."
        )
    assert_disjoint(train, evaluation)
    assert_both_sides_populated(stats)
    stats["cross_side_identifiers"] = cross_side_identifiers(train, evaluation)
    return train, evaluation, stats


# --------------------------------------------------------------------------
# The chain of custody, generalised for the whole assembly tail.
# --------------------------------------------------------------------------

CUSTODY_VERIFIED = "verified"
CUSTODY_NO_MANIFEST = "no_manifest"
CUSTODY_MISMATCH = "content_mismatch"
CUSTODY_NO_DIGEST = "no_output_digest"
CUSTODY_UNREADABLE = "unreadable_manifest"


def manifest_digests(manifest) -> set[str]:
    """Every output digest a stage manifest records, in either shape.

    decontaminate.py writes a single `output` record; dedupe.py and everything
    in this tail write an `outputs` list, because a stage that writes two files
    cannot be described by one record. Both are read here so one function
    serves the whole chain.
    """
    found: set[str] = set()
    if not isinstance(manifest, dict):
        return found
    records = list(manifest.get("outputs") or [])
    single = manifest.get("output")
    if isinstance(single, dict):
        records.append(single)
    for record in records:
        if isinstance(record, dict) and record.get("sha256"):
            found.add(str(record["sha256"]))
    return found


def read_manifest(path: Path) -> dict | None:
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def custody_of(inputs: Sequence[Path], *, manifest_filename: str) -> tuple[dict | None, dict]:
    """(the upstream manifest to carry forward, the custody record).

    dedupe.custody_of's rule, generalised to many inputs and any upstream
    stage: the manifest is looked for beside the first input, and every input's
    bytes must be one of the outputs that manifest recorded. Matching is BY
    DIGEST, never by path - the same bytes under another name are still the
    rows the upstream pass wrote, and a file at the expected path with other
    bytes is not.

    UNLIKE dedupe.py, a failure here is a REFUSAL rather than a banner. dedupe
    can honestly say "these rows may never have been screened" and still ship a
    deduplicated file; this tail cannot, because the artifact it produces is
    the dataset, and a dataset whose provenance is a directory layout is a
    dataset nobody can write a card for.
    """
    from tuned.data.acquire import sha256_file

    paths = [Path(p) for p in inputs]
    record = {
        "status": CUSTODY_NO_MANIFEST,
        "manifest": None,
        "inputs": [str(p) for p in paths],
        "input_sha256": {},
        "manifest_outputs": [],
    }
    if not paths:
        return None, record
    manifest_path = paths[0].parent / manifest_filename
    record["manifest"] = str(manifest_path)
    if not manifest_path.exists():
        return None, record
    upstream = read_manifest(manifest_path)
    if upstream is None:
        record["status"] = CUSTODY_UNREADABLE
        return None, record
    digests = manifest_digests(upstream)
    recorded = list(upstream.get("outputs") or [])
    if isinstance(upstream.get("output"), dict):
        recorded.append(upstream["output"])
    record["manifest_outputs"] = [
        {k: r.get(k) for k in ("path", "rows", "sha256")}
        for r in recorded
        if isinstance(r, dict)
    ]
    if not digests:
        record["status"] = CUSTODY_NO_DIGEST
        return None, record
    record["input_sha256"] = {str(p): sha256_file(p) for p in paths}
    if not set(record["input_sha256"].values()) <= digests:
        record["status"] = CUSTODY_MISMATCH
        return None, record
    record["status"] = CUSTODY_VERIFIED
    return upstream, record


CUSTODY_BANNERS = {
    CUSTODY_NO_MANIFEST: (
        "NO UPSTREAM MANIFEST beside this input, so nothing here can say these rows were"
        " deduplicated or screened at all"
    ),
    CUSTODY_UNREADABLE: (
        "THE UPSTREAM MANIFEST BESIDE THIS INPUT CANNOT BE READ as JSON - it is truncated"
        " or was written by something else"
    ),
    CUSTODY_NO_DIGEST: (
        "THE UPSTREAM MANIFEST BESIDE THIS INPUT RECORDS NO OUTPUT DIGEST (it predates"
        " dedupe_version 4), so nothing here can tell whether it describes these rows."
        " Re-run the upstream stage"
    ),
    CUSTODY_MISMATCH: (
        "THE UPSTREAM MANIFEST BESIDE THIS INPUT DESCRIBES DIFFERENT ROWS - its output"
        " digests do not match the bytes read here, so these rows were NOT the ones it"
        " reports on"
    ),
}


def custody_refusal(record: dict, *, stage: str, remedy: str) -> str:
    return (
        f"{stage} REFUSES TO RUN: {CUSTODY_BANNERS[record['status']]}.\n"
        f"  manifest looked for: {record['manifest']}\n"
        f"  remedy: {remedy}\n"
        f"  nothing was written; no output carries a {stage} stamp."
    )


def output_records(pairs: Iterable[tuple[Path, int]]) -> list[dict]:
    """What this pass wrote, identified by content - decontaminate's shape."""
    from tuned.data.acquire import sha256_file

    return [
        {"path": str(path), "rows": rows, "sha256": sha256_file(path)}
        for path, rows in pairs
    ]


def manifest_of(stats: dict, *, inputs: Sequence[str], outputs: Sequence[dict],
                fraction: float, upstream: dict | None, custody: dict) -> dict:
    from tuned.data.store import utcnow

    return {
        "stage": "split",
        "split_version": SPLIT_VERSION,
        "at": utcnow(),
        "inputs": list(inputs),
        "outputs": list(outputs),
        "eval_fraction": fraction,
        "counts": {
            key: stats[key]
            for key in ("rows", "units", "cases", "caseless_rows", "train_rows", "eval_rows",
                        "eval_target_rows", "eval_units", "dated_units")
        },
        "assignment": {
            "date_assigned_units": stats["date_assigned_units"],
            "hash_assigned_units": stats["hash_assigned_units"],
            "date_boundary": stats["date_boundary"],
            "by_date_channel": stats["by_date_channel"],
            "eval_fraction_achieved": stats["eval_fraction_achieved"],
            # The assert's known blind spot, as a number. See the docstring.
            "cross_side_identifiers": stats["cross_side_identifiers"],
        },
        # The chain, carried whole rather than summarised: stats.py has to be
        # able to reach decontamination's own record through this.
        "dedupe": upstream,
        "dedupe_check": custody,
    }


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from tuned.data.config import load_build_config
    from tuned.data.decontaminate import write_manifest
    from tuned.data.jsonl import write_jsonl
    from tuned.data.paths import build_paths
    from tuned.data.store import Store

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data_law_v1.yaml")
    parser.add_argument("--in", dest="input", default=None,
                        help=f"default out/{DEDUPE_OUT_FILENAME} (dedupe.py's output)")
    parser.add_argument("--out-train", default=None, help=f"default out/{TRAIN_FILENAME}")
    parser.add_argument("--out-eval", default=None, help=f"default out/{EVAL_FILENAME}")
    args = parser.parse_args(argv)

    cfg = load_build_config(args.config)
    paths = build_paths(cfg.build.workdir).ensure()
    in_path = Path(args.input) if args.input else paths.out_dir / DEDUPE_OUT_FILENAME
    train_path = Path(args.out_train) if args.out_train else paths.out_dir / TRAIN_FILENAME
    eval_path = Path(args.out_eval) if args.out_eval else paths.out_dir / EVAL_FILENAME
    # The eval fraction is build.held_out_frac, not a knob of this module's
    # own: a second copy of 10% is a fence that can disagree with the fencing.
    fraction = cfg.build.held_out_frac

    if not in_path.exists():
        print(
            f"no such input: {in_path}\n"
            f"  run: python -m tuned.data.dedupe --config {args.config}"
        )
        return 2

    upstream, custody = custody_of([in_path], manifest_filename=DEDUPE_MANIFEST_FILENAME)
    if upstream is None:
        print(custody_refusal(
            custody, stage="split",
            remedy=f"python -m tuned.data.dedupe --config {args.config}",
        ))
        return 2

    # The case-identifier channel dedupe recorded, carried into the printout:
    # this pass buckets by the same identifiers it did, so which channel fed
    # them decides which rows can straddle in the first place.
    ids_from_text = (upstream.get("thresholds") or {}).get("case_ids_from_text")
    items = list(stream_items([in_path], ids_from_text=bool(ids_from_text)))

    try:
        train, evaluation, stats = split_items(items, fraction=fraction)
    except SplitRefusal as exc:
        print(f"split REFUSES TO RUN: {exc}")
        return 2

    train_rows = write_jsonl(train_path, [i.row for i in train])
    eval_rows = write_jsonl(eval_path, [i.row for i in evaluation])
    manifest = manifest_of(
        stats,
        inputs=[str(in_path)],
        outputs=output_records([(train_path, train_rows), (eval_path, eval_rows)]),
        fraction=fraction,
        upstream=upstream,
        custody=custody,
    )
    write_manifest(train_path.parent / MANIFEST_FILENAME, manifest)
    store = Store.open(paths.state_db)
    try:
        store.log_event("split", manifest)
    finally:
        store.close()

    print(f"read {stats['rows']} rows from {in_path}")
    print(
        f"  {stats['units']} assignment atoms: {stats['cases']} cases and "
        f"{stats['caseless_rows']} rows carrying no case identifier"
    )
    print(
        f"  eval target {stats['eval_target_rows']} rows "
        f"({fraction:.0%} of {stats['rows']}), achieved "
        f"{stats['eval_fraction_achieved']:.4f} over {stats['eval_units']} atoms"
    )
    print(
        f"    {stats['date_assigned_units']} dated (boundary {stats['date_boundary']}), "
        f"{stats['hash_assigned_units']} by content hash"
    )
    for channel in DATE_CHANNELS:
        print(f"    date[{channel}]: {stats['by_date_channel'][channel]}")
    if stats["by_date_channel"][DATE_FROM_PROV] == 0:
        # Stated rather than left to be inferred from a zero: no row builder in
        # this repo writes a date into _prov, so this line is the expected one
        # and its ABSENCE is the news.
        print(
            "  no row carried an explicit _prov date - the newest-first preference ran "
            "entirely off case identifiers, which is what this build currently produces"
        )
    if stats["cross_side_identifiers"]:
        print(
            f"  {stats['cross_side_identifiers']} CNR(s) appear on both sides without "
            f"being one atom - one real case reached under two different strongest "
            f"identifiers. Not a straddle by the atom rule, and not invisible either"
        )
    print(f"wrote {train_rows} rows -> {train_path}")
    print(f"      {eval_rows} rows -> {eval_path}")
    print(f"      manifest -> {train_path.parent / MANIFEST_FILENAME}")
    return 0


if __name__ == "__main__":
    import os
    import sys

    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
