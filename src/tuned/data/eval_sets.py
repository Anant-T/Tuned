"""The eval corpora this build decontaminates against.

Lifted verbatim out of decontaminate.py on 2026-08-30. It is a suffix-
dispatching reader, a split-selection policy and nine status codes with
remedies, and it shares nothing with the n-gram screen except the tokeniser
- so it was 611 lines a reader had to page through to reach the rules that
decide whether a row ships.

The dependency runs one way on purpose: this module imports decontaminate's
primitives, decontaminate imports this one only inside the functions that
need it (manifest_of, main). Reversing that, or importing both ways at
module scope, is a circular import.
"""

import csv
import json
import math
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tuned.data.acquire import HF_SOURCES, rebase_under_corpus
from tuned.data.decontaminate import (
    SHORT_MIN_TOKENS,
    identifiers_from_fields,
    identifiers_from_text,
    tokens,
    window_for,
)


# --------------------------------------------------------------------------
# The eval corpora.
# --------------------------------------------------------------------------

# When the per-config+split row counts below were read off the Hub's
# datasets-server. An edit to one of them is a decision, not a typo-fix.
EVAL_COUNTS_VERIFIED_AT = "2026-08-14"


@dataclass(frozen=True)
class EvalPart:
    """One config+split of an eval set, and the rows the Hub reported for it.

    `config` is descriptive - the label the datasets-server gives that view of
    the repo - because selection matches on the SPLIT name in a file's path,
    which is the only thing a snapshot on disk actually carries. `rows` is None
    where the count has not been read, and a None anywhere in a set's parts
    turns that set's floor and shortfall instrument off rather than
    denominating them against a guess.
    """

    config: str
    split: str
    rows: int | None = None


@dataclass(frozen=True)
class EvalSet:
    """One eval corpus: acquire.py owns WHERE it comes from, this owns WHY.

    repo_id/license/source_id are read off acquire.HF_SOURCES rather than
    repeated here - two spellings of a repo id is how the acquire command
    this module prints stops matching the source id it then looks up.

    `include_splits`/`exclude_splits` are the EVAL SURFACE: which splits of the
    repo this dataset is screened against. They are matched as NAME COMPONENTS
    of an object key, so `test_specific-00000-of-00003.parquet` is selected by
    "test" and `train_all-...` is excluded by "train".
    """

    key: str
    why: str
    include_splits: tuple[str, ...] = ()
    exclude_splits: tuple[str, ...] = ()
    parts: tuple[EvalPart, ...] = ()
    # Why this surface and not the whole repo. Printed and recorded, because a
    # subset that is not named narrows the guarantee silently.
    selection_note: str = ""

    @property
    def expect_rows(self) -> int | None:
        """Rows expected FOR THE SPLITS THIS SET IS SCREENED AGAINST.

        Not the whole repo: `rows` is counted after the split filter, so a
        whole-repo expectation makes a correct, complete download read short on
        every run - and the SHORT line the CLI prints below is what an
        operator reads to decide whether a config or a shard is missing.
        """
        counts = [part.rows for part in self.parts]
        return sum(counts) if counts and None not in counts else None

    @property
    def source(self):
        return HF_SOURCES[self.key]

    @property
    def repo_id(self) -> str:
        return self.source.repo_id

    @property
    def license(self) -> str:
        return self.source.license

    @property
    def source_id(self) -> str:
        return self.source.source_id

    @property
    def url(self) -> str:
        return self.source.url


# The three eval corpora the charter is judged against. Their repo ids live in
# acquire.HF_SOURCES and were checked against the Hub on 2026-08-14 - one of
# the three was wrong at that check (opennyaiorg/aibe -> aibe_dataset), which
# is exactly what a wrong id does here: `not_acquired`, a REFUSAL with the id
# printed, never a quiet skip.
#
# THE ROW COUNTS ARE PER CONFIG AND SPLIT, and that is the whole point of the
# shape: `rows` is counted AFTER the split filter, so an expectation that
# describes the whole repo makes a correct, complete download print a
# shortfall on every run - which is the SHORT line the CLI prints per set.
EVAL_SETS = {
    "bbl": EvalSet(
        key="bbl",
        why="the project's headline forgetting guard (24,365 questions)",
        # TWO CONFIGS, both split `test`, and the expectation encodes both: an
        # English-only download reads 17,047 against 24,365 and says so loudly,
        # which is precisely the "one config landed" failure a whole-repo
        # number cannot express. There is no train split to exclude.
        include_splits=("test",),
        parts=(
            EvalPart("english", "test", 17_047),
            EvalPart("hindi", "test", 7_318),
        ),
        selection_note="both configs, split test - the whole set is the eval surface",
    ),
    "iltur": EvalSet(
        key="iltur",
        why="8 Indian legal tasks; CJPE shares appellate pools with PredEx",
        # THE EVAL SURFACE IS THE TEST-TYPE SPLITS OF EVERY CONFIG, and the
        # arithmetic is in the module docstring under WHAT IL-TUR IS SCREENED
        # AGAINST. Short version: the repo is ~488,000 rows across 8 configs
        # (bail alone is 353,698), the index costs ~220 bytes per distinct
        # gram and an item of L tokens contributes about L of them, so the
        # whole repo is tens of gigabytes of index - it does not fit, and a
        # screen that OOMs on the operator's machine screens nothing.
        #
        # Matched on the SPLIT NAME, never on a config name: the config names
        # were not verified and a guessed one would silently exclude a config.
        # `train_all` and `fold_N` are training material; `test_specific` and
        # `expert` are what the tasks are scored on.
        include_splits=("test", "expert"),
        exclude_splits=("train", "fold", "dev", "val", "validation"),
        # No verified counts: the datasets-server was read for the repo total
        # and for `bail`, not per split. The floor and the shortfall line are
        # therefore OFF for this set and say so, rather than being denominated
        # against a number nobody has read. First-run item.
        selection_note=(
            "test-type splits of every config (test/expert); train_all, fold_N, dev and "
            "val are EXCLUDED - see the module docstring for the memory arithmetic"
        ),
    ),
    "aibe": EvalSet(
        key="aibe",
        why="the bar-exam MCQ set Aalap measured against",
        # ONE SPLIT, named `train`, and it IS the eval set - so there is no
        # filter at all here and the complete shape must read `ok` with a
        # shortfall of zero. Filtering for a `test` split would empty it.
        parts=(EvalPart("default", "train", 1_157),),
        selection_note="single split `train`, which is the whole set",
    ),
}

EVAL_OK = "ok"
EVAL_NOT_ACQUIRED = "not_acquired"
EVAL_NO_FILES = "no_files"
EVAL_NO_TEXT_COLUMN = "no_text_column"
EVAL_EMPTY = "empty"
EVAL_UNREADABLE = "unreadable"
# The files are there and readable in principle, but the library that reads
# them is not installed. A different remedy from every other rung, and the one
# EVERY operator hits on the first real run: HF snapshots are usually parquet.
EVAL_NO_READER = "no_reader"
# Loaded, and NOTHING in it can match anything. An eval set the screen
# compared against nothing is not a screened eval set, whatever the row count
# says, and "cannot reach an eval set must not report clean" is not satisfied
# by reaching it in name only.
EVAL_UNMATCHABLE = "unmatchable"
# Loaded, and far smaller than the set is documented to be - a fragment, a
# single shard, or a repo id that resolved to something else.
EVAL_TOO_FEW = "too_few_rows"
# The set has a SPLIT FILTER and no object on disk names a screened split.
#
# This used to be a silent fallback to "read everything", justified as
# "over-screening is safe". It is not safe here and the arithmetic says why:
# IL-TUR is ~488,000 rows across 8 heterogeneous configs and the eval surface
# chosen for it is a few thousand of them, so reading everything is ~20 GB of
# index on a machine that does not have it - the OOM this module's own
# docstring rejects, arrived at from the other side. Worse, the fallback
# OVERWROTE `record["excluded"] = []`, so the manifest read "selected all,
# excluded nothing" while ~488,000 training rows became the eval surface, and
# its designated tell - `surplus` - is structurally 0 for the one set with
# fallback risk, because IL-TUR has no verified row count.
#
# So a filtered set whose filter selects nothing is a REFUSAL, the same regime
# as every other "loaded but useless" state, and the remedy names the layout it
# saw against the layout it expected. A set with NO filter (aibe, whose single
# `train` split IS the set) never reaches this: reading everything is its
# normal path, not a fallback.
EVAL_NO_SPLIT = "no_screened_split"

# The floor is a HUNDREDTH of the expected count, not a half. The counts are
# verified now (EVAL_COUNTS_VERIFIED_AT), but a shard that has not finished
# downloading is a legitimate mid-pull state and the FILE LAYOUT that decides
# which of them are read is not verified at all - so a floor that refuses a
# CORRECT download would push the operator
# straight to `--allow-missing-eval bbl`, which is the exact outcome this
# module exists to prevent - so the refusal is sized to catch a set that is
# obviously not the set, and the SHORTFALL against expect_rows is printed and
# recorded at every level below it, where an operator can act on it without
# being tempted to waive anything.
EVAL_MIN_SHARE = 0.01

# Every candidate column an eval row might carry its question in, tried in
# order. Same discipline as select.py's schema handling: the real column names
# cannot be checked offline, so each read is a short ordered list and the
# WINNER is reported - "no candidate matched" and "the fallback matched" must
# never be the same number on the one instrument built to tell them apart.
_EVAL_TEXT_FIELDS = (
    "question", "query", "prompt", "input", "text", "instruction",
    "question_text", "stem", "context", "passage", "document",
)
# Read after the question and appended to it: the options and the reference
# answer are part of the item a row could have memorised.
_EVAL_EXTRA_FIELDS = ("options", "choices", "answer", "correct_answer", "target", "output")

_READABLE_SUFFIXES = (".jsonl", ".ndjson", ".json", ".csv", ".tsv", ".parquet")


@dataclass(frozen=True)
class EvalItem:
    set_key: str
    item_id: str
    text: str
    identifiers: frozenset[str]


@dataclass
class EvalCorpus:
    spec: EvalSet
    status: str
    items: list[EvalItem] = field(default_factory=list)
    files: int = 0
    rows: int = 0
    text_field: str | None = None
    detail: str = ""
    allowed_missing: bool = False
    # Items nothing in this module can match (under SHORT_MIN_TOKENS). Counted
    # HERE and not only in the index, because a set that is entirely
    # unmatchable has to be refused before anything is written.
    unmatchable: int = 0
    # Which objects were read and which were left out, with the reason. An
    # undocumented subset narrows the guarantee silently, so the selection is
    # recorded rather than implied by a row count.
    selection: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == EVAL_OK

    @property
    def shortfall(self) -> int:
        """How many rows short of the expectation FOR THE SPLITS SELECTED.

        `rows` counts what the split filter kept, so the expectation it is
        compared against has to describe the same population - otherwise a
        correct, complete download reads short on every run and the banner
        that says so is noise.
        """
        expect = self.spec.expect_rows
        return max(0, expect - self.rows) if expect else 0

    @property
    def surplus(self) -> int:
        """Rows ABOVE the expectation for the splits this set is screened against.

        The tell that the filter selected MORE than the eval surface: a shard
        counted twice, a config that is not in `parts`, or an object whose name
        happens to carry a screened split. It used to be described as the
        fallback's tell, which it never was and now could not be - the fallback
        is a refusal (EVAL_NO_SPLIT) and IL-TUR, the only set with fallback
        risk, has no verified count for this to be denominated against at all.
        It fires for BBL and aibe, whose counts are verified, and it is OFF and
        says so where they are not.
        """
        expect = self.spec.expect_rows
        return max(0, self.rows - expect) if expect else 0


def read_rows(path: Path) -> Iterator[dict]:
    """Rows out of one snapshot file, dispatched on suffix.

    jsonl/json/csv/tsv are read here in pure Python so the load path is
    exercised offline. Parquet - what HF snapshots usually ship - is behind a
    lazy pyarrow import, and it HAS executed: the test round-trips a real
    pyarrow file where the [build] extra is present and keeps the ImportError
    path (EVAL_NO_READER, a refusal with `pip install -e .[build]` as the
    remedy) where it is not.
    """
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        yield record
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else payload.get("data", [])
        yield from (r for r in records if isinstance(r, dict))
    elif suffix in (".csv", ".tsv"):
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ",")
    elif suffix == ".parquet":
        import pyarrow.parquet as pq

        yield from pq.read_table(path).to_pylist()


def eval_item_texts(record: dict) -> tuple[list[str], str | None]:
    """The screenable texts of one eval row, and the column its question came from.

    The question and the options/answer are SEPARATE items, never one
    concatenated blob, because containment divides by the eval item's own
    length: a 29-token question with 20 tokens of options appended scores
    17/37 = 0.46 for a row that quotes the whole question verbatim - under the
    threshold, and a leak the screen would have reported clean. Measured on
    the fixture in test_a_question_is_screened_separately_from_its_options
    (0.89 concatenated vs 1.0 apart, and worse for shorter questions).

    Sub-floor leftovers (an answer key of "b") are not returned: a 1-token
    string is not evidence of leakage, and treating it as an unscreenable
    item would bury the count of questions that genuinely are.
    """
    question, winner = "", None
    for name in _EVAL_TEXT_FIELDS:
        value = record.get(name)
        if value:
            question, winner = str(value), name
            break
    if winner is None:
        return [], None
    extras = []
    for name in _EVAL_EXTRA_FIELDS:
        value = record.get(name)
        if value:
            extras.append(" ".join(str(v) for v in value) if isinstance(value, list) else str(value))
    texts = [question]
    rest = "\n".join(extras)
    if len(tokens(rest)) >= SHORT_MIN_TOKENS:
        texts.append(rest)
    return texts, winner


_NAME_PART = re.compile(r"[^a-z0-9]+")


def _name_parts(key: str) -> set[str]:
    """The alphanumeric runs of an object key - `data/test-00000.parquet` ->
    {data, test, 00000, parquet}, and `latest.parquet` -> {latest, parquet}."""
    return set(_NAME_PART.split(key.lower()))


def select_split_files(spec: EvalSet, paths: Sequence[tuple[str, Path]]):
    """The files of this set that are its eval surface, and the record of it.

    A NAME COMPONENT, not a substring: `latest.parquet` contains "test", and a
    bare substring test would select it and silently screen against the wrong
    file.

    EXCLUDE BEATS INCLUDE, so `train_test-00000.parquet` is excluded: a name
    carrying both is ambiguous and the safe reading of an ambiguous name is the
    one that does not put a training split into the eval surface.

    A filter that selects nothing is NOT a fallback to reading everything - see
    EVAL_NO_SPLIT for the arithmetic. `no_screened_split` says so and the
    exclusion record is left intact, because "selected all, excluded nothing"
    is exactly the sentence the manifest must never be able to write about a
    layout nobody recognised.
    """
    include, exclude = set(spec.include_splits), set(spec.exclude_splits)
    record = {
        "include_splits": list(spec.include_splits),
        "exclude_splits": list(spec.exclude_splits),
        "note": spec.selection_note,
        "selected": [],
        "excluded": [],
        "no_screened_split": False,
    }
    if not include and not exclude:
        # No filter at all: the whole repo IS the eval surface (aibe). Reading
        # everything here is the normal path and not a fallback from anything.
        record["selected"] = [key for key, _ in paths]
        return list(paths), record
    selected = []
    for key, path in paths:
        parts = _name_parts(key)
        if exclude & parts:
            record["excluded"].append({"key": key, "why": sorted(exclude & parts)[0]})
        elif include and not (include & parts):
            record["excluded"].append({"key": key, "why": "no screened split in the name"})
        else:
            selected.append((key, path))
            record["selected"].append(key)
    if not selected:
        record["no_screened_split"] = True
        return [], record
    return selected, record


def eval_corpus(store, spec: EvalSet, *, reader=read_rows, corpus_dir=None) -> EvalCorpus:
    """Load one eval set from what acquire.py landed, or say exactly why not.

    Read through the store's artifact index rather than a directory walk, for
    select.py's reason: an item can only come out of a file the store says is
    complete. Every way this comes back short is a DIFFERENT status, because
    they send the operator to different places - `not_acquired` is an acquire
    run (or an access grant), `no_text_column` is a column name in a file
    already on disk.
    """
    index = store.artifact_index(spec.source_id)
    if not index:
        return EvalCorpus(spec, EVAL_NOT_ACQUIRED, detail="nothing indexed under this source id")
    # The recorded local_path is absolute and acquisition-time; re-root it
    # under THIS checkout's corpus dir when it no longer resolves (see
    # acquire.rebase_under_corpus - the worktree it was indexed under is
    # gone, and a CI runner's root differs again).
    paths = [
        (
            key,
            Path(row["local_path"]) if corpus_dir is None
            else rebase_under_corpus(row["local_path"], key, corpus_dir),
        )
        for key, row in sorted(index.items())
        if Path(key).suffix.lower() in _READABLE_SUFFIXES
    ]
    if not paths:
        return EvalCorpus(
            spec, EVAL_NO_FILES, detail=f"{len(index)} objects, none of {_READABLE_SUFFIXES}"
        )
    seen = sorted({part for key, _ in paths for part in _name_parts(key)})
    paths, selection = select_split_files(spec, paths)
    if selection["no_screened_split"]:
        # BEFORE a single row is read. The banner this replaced printed after
        # eval_corpus had materialised every row and both indexes were built -
        # i.e. after the ~20 GB the docstring says does not fit.
        return EvalCorpus(
            spec, EVAL_NO_SPLIT, files=0, selection=selection,
            detail=(
                f"{len(selection['excluded'])} objects, none naming any of "
                f"{list(spec.include_splits)} as a path component. The names carry "
                f"{seen[:12]}"
            ),
        )

    items: list[EvalItem] = []
    rows = 0
    winners: dict[str, int] = {}
    for key, path in paths:
        try:
            records = list(reader(path))
        except ImportError as exc:
            # NOT "unreadable". The file is fine and the repo id is fine; the
            # reader is missing, and sending an operator to re-download a
            # correct snapshot - or worse, to doubt a repo id that is the one
            # genuinely uncertain thing here - is the wrong instruction. HF
            # snapshots are usually parquet, so this is what the first real
            # run hits, on all three sets at once.
            return EvalCorpus(
                spec, EVAL_NO_READER, files=len(paths), selection=selection,
                detail=f"{key}: {exc}",
            )
        except Exception as exc:  # a corrupt or unreadable snapshot file
            return EvalCorpus(
                spec, EVAL_UNREADABLE, files=len(paths), selection=selection,
                detail=f"{key}: {type(exc).__name__}: {exc}",
            )
        for i, record in enumerate(records):
            rows += 1
            texts, winner = eval_item_texts(record)
            if winner is None or not any(t.strip() for t in texts):
                continue
            winners[winner] = winners.get(winner, 0) + 1
            identifiers = identifiers_from_fields(record) | identifiers_from_text(
                "\n".join(texts)
            )
            for part, text in enumerate(texts):
                items.append(
                    EvalItem(
                        spec.key, f"{key}#{i}" + ("" if part == 0 else f"/{part}"),
                        text, frozenset(identifiers),
                    )
                )
    if rows and not items:
        return EvalCorpus(
            spec, EVAL_NO_TEXT_COLUMN, files=len(paths), rows=rows, selection=selection,
            detail=f"{rows} rows, none carrying any of {_EVAL_TEXT_FIELDS}",
        )
    if not items:
        return EvalCorpus(spec, EVAL_EMPTY, files=len(paths), rows=rows,
                          selection=selection, detail="0 rows")
    corpus = EvalCorpus(
        spec, EVAL_OK, items=items, files=len(paths), rows=rows, selection=selection,
        text_field=max(winners, key=winners.get),
        unmatchable=sum(1 for item in items if not window_for(len(tokens(item.text)))),
    )
    floor = math.ceil((spec.expect_rows or 0) * EVAL_MIN_SHARE)
    # ROWS, not items: a BBL row with options produces two items, so an item
    # floor compares one population against another and moves with a column
    # list rather than with the download. The counts in EvalPart are rows.
    if floor and rows < floor:
        corpus.status = EVAL_TOO_FEW
        corpus.detail = (
            f"{rows} rows, and the splits this set is screened against hold "
            f"{spec.expect_rows} (verified {EVAL_COUNTS_VERIFIED_AT}). "
            f"Under {floor} ({EVAL_MIN_SHARE:.0%}) this is a fragment, a single shard, or "
            f"a different dataset"
        )
    elif corpus.unmatchable == len(items):
        # The one thing that must not be got wrong, one layer in from a
        # missing set: the screen ran against a corpus none of whose items it
        # can match, and the run would otherwise be stamped decontaminated.
        corpus.status = EVAL_UNMATCHABLE
        corpus.detail = (
            f"all {len(items)} items are under {SHORT_MIN_TOKENS} tokens, so NOTHING here "
            f"can match any of them - this set was compared against nothing"
        )
    return corpus


def eval_corpora(store, *, allow_missing: Iterable[str] = (), reader=read_rows,
                 keys: Iterable[str] | None = None, corpus_dir=None) -> dict[str, EvalCorpus]:
    allowed = set(allow_missing)
    out: dict[str, EvalCorpus] = {}
    for key in (sorted(EVAL_SETS) if keys is None else list(keys)):
        corpus = eval_corpus(store, EVAL_SETS[key], reader=reader, corpus_dir=corpus_dir)
        corpus.allowed_missing = not corpus.ok and key in allowed
        out[key] = corpus
    return out


def _acquire_remedy(key: str, spec: EvalSet) -> str:
    return (
        f"python -m tuned.data.acquire --kind hf --hf-source {key}\n"
        f"               (all three eval sets report gated=auto, so accept the terms\n"
        f"                at {spec.url} and set HF_TOKEN;\n"
        f"                the repo id itself was verified against the Hub 2026-08-14)"
    )


def _remedy(key: str, corpus: EvalCorpus) -> str:
    """What to DO about this status. They send the operator to different
    places, which is the whole reason each way of coming back short is its own
    status rather than one warning."""
    spec = corpus.spec
    if corpus.status == EVAL_NO_READER:
        return (
            "pip install -e .[build]\n"
            "               (the snapshot is fine and so is the repo id - the reader for\n"
            "                these files is not installed. HF snapshots are usually\n"
            "                parquet, so this is the first-run case, not a corrupt file)"
        )
    if corpus.status == EVAL_UNREADABLE:
        return (
            f"the file itself is corrupt - delete it and re-run\n"
            f"               python -m tuned.data.acquire --kind hf --hf-source {key}"
        )
    if corpus.status == EVAL_NO_TEXT_COLUMN:
        return (
            "add this set's real question column to _EVAL_TEXT_FIELDS in this module\n"
            "               (the file is on disk and readable; nothing in it was\n"
            "                recognised as a question)"
        )
    if corpus.status == EVAL_UNMATCHABLE:
        return (
            "read the column that was chosen - an item under 5 tokens is an answer\n"
            "               key or a label, not a question, so _EVAL_TEXT_FIELDS is\n"
            "               probably matching the wrong column"
        )
    if corpus.status == EVAL_NO_SPLIT:
        return (
            f"check the layout under {spec.url} against this set's split filter\n"
            f"               (expected a path component from {list(spec.include_splits)};\n"
            f"                the layout it actually saw is printed above.\n"
            f"                Fix include_splits in EVAL_SETS for the real layout - this is\n"
            f"                NOT read-everything-instead: {spec.key} is ~488,000 rows across\n"
            f"                8 configs and the whole repo is tens of GB of index, so an\n"
            f"                unrecognised layout means the eval surface is unknown, not wide)"
        )
    if corpus.status == EVAL_TOO_FEW:
        return (
            f"check the configs and the shard count under {spec.url}\n"
            f"               ({spec.expect_rows} is the sum of "
            f"{[(p.config, p.split, p.rows) for p in spec.parts]}, read off the\n"
            f"                datasets-server on {EVAL_COUNTS_VERIFIED_AT}; if a config is\n"
            f"                simply missing from the snapshot, acquire it rather than\n"
            f"                editing EvalSet.parts. Do NOT waive the set to get past this:\n"
            f"                a waived BBL is the failure this module exists to prevent)"
        )
    return _acquire_remedy(key, spec)


def refusals(corpora: dict[str, EvalCorpus]) -> list[str]:
    """One actionable refusal per eval set that could not be read and was not
    explicitly waived. Empty means the pass may run."""
    out = []
    for key in sorted(corpora):
        corpus = corpora[key]
        if corpus.ok or corpus.allowed_missing:
            continue
        spec = corpus.spec
        out.append(
            f"eval set {key!r} ({spec.repo_id}) is {corpus.status}: {corpus.detail}.\n"
            f"    it guards: {spec.why}\n"
            f"    fix it:    {_remedy(key, corpus)}\n"
            f"    override:  --allow-missing-eval {key}  (recorded in the manifest)"
        )
    return out


REFUSAL_HEADER = (
    "REFUSING TO DECONTAMINATE: an eval set this dataset is measured against cannot be read,\n"
    "and a pass that cannot reach an eval set must not report clean. A leak into training\n"
    "inflates the headline number in the flattering direction and nothing downstream can\n"
    "detect it."
)


