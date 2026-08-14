"""Drop every candidate row that overlaps an eval set - and REFUSE to run blind.

Input is the assembly stream (the JSONL streams replay.py/curated.py wrote,
plus the accepted generations in the store); output is
`out/decontaminated.jsonl`, a drop log with a machine-readable reason per
dropped row, and a manifest that says what this pass could and could not see.

THE ONE THING THIS MODULE EXISTS TO PREVENT
-------------------------------------------
The project's headline number is a blind pairwise judge run plus
BhashaBench-Legal as a forgetting guard. A leaked eval question does not make
that number noisy - it makes it wrong in the FLATTERING direction, and
nothing downstream can detect it. So the failure to design against is not an
imprecise matcher; it is

    THE MATCHER NEVER RAN AND THE RUN STILL EXITED 0.

Hence the rule that outranks every other decision here: **an eval set that is
absent, unreadable or empty is a REFUSAL, not a warning.** Nothing is
written - no output file, no "decontaminated" stamp - and the CLI exits 2
naming the set and the command that fixes it. An operator who genuinely wants
to proceed says so per set (`--allow-missing-eval bbl`, the fleet's
`--allow-pool-gaps` idiom), and that override is written INTO THE MANIFEST,
not merely printed, so stats.py and the dataset card carry it forward. A
dataset built without BBL screening must be able to say so about itself years
later.

THE ASYMMETRY THAT DECIDES EVERY JUDGEMENT CALL HERE
----------------------------------------------------
A false negative (an eval row survives into training) is invisible,
permanent, and inflates the metric the project is judged by. A false positive
costs one row out of ~18,000. These are not comparable, so every borderline
call in this module goes to DROPPING - and every drop carries a
machine-readable reason so the yield price stays auditable rather than
mysterious. (dedupe.py is NOT in this regime; see its docstring.)

THE LEVELS, and the case each one alone must carry
--------------------------------------------------
1. TEXT CONTAINMENT - the share of an eval item's 13-grams that appear in the
   row is >= CONTAINMENT (0.5). Catches the eval question quoted, verbatim or
   near-verbatim, inside a much longer training row. Applies to items of 38
   tokens or more; below that the 13-gram window cannot deliver "near".
2. NARROW-WINDOW CONTAINMENT - the same rule at a SMALLER window, for items
   between 13 and 37 tokens. See THE WINDOW IS LENGTH-AWARE below: this is
   the band BhashaBench-Legal lives in, and at a fixed 13 the rule there is an
   exact-match rule wearing a near-match label.
3. SHORT-ITEM CONTAINMENT - an eval item shorter than 13 tokens has no
   13-grams at all. It is matched WHOLE, and nothing less than the whole of it
   counts. Items below SHORT_MIN_TOKENS (5) are matchable by nothing here and
   are COUNTED as `unmatchable`, per set, because a 3-token question would
   otherwise match half the corpus - and a set whose items are ALL unmatchable
   is a refusal, not a clean run.
4. CASE-IDENTIFIER OVERLAP - CNR, neutral/reporter citation, or normalised
   case title shared between a row and an eval item. This is the level that
   catches SAME JUDGMENT, DIFFERENT QUESTION, which no n-gram method can ever
   see (PredEx and IL-TUR CJPE draw on the same appellate pools), and it is
   the cheapest and most certain of the four.

THE WINDOW IS LENGTH-AWARE, and the measurement that forced it
---------------------------------------------------------------
One substituted token destroys every gram covering it - up to n of them. An
item of L tokens has G = L - n + 1 grams, so after one central edit its
containment in a row carrying the rest is (G - n)/G, which reaches 0.5 only
when L >= 3n - 1. At n = 13 that is L >= 38 tokens. Measured:

    item tokens  |  13     25     29     35     38     50     150
    C after 1 edit  0.000  0.000  0.235  0.435  0.500  0.658  0.906

So at a fixed 13 the rule is EXACT MATCH below ~38 tokens, whatever the
threshold says - and BBL is 24,365 MCQ questions, most of them shorter than
that. The window is therefore the largest one that survives a single central
edit at this item's length:

    window_for(L) = min(13, (L + 1) // 3)        for L >= 13
                  = L      (whole sequence)      for 5 <= L < 13
                  = nothing                      below 5

which puts C after one edit at or above 0.5 at EVERY length from 13 up, and
leaves items of 38 tokens and more exactly where they were.

WHAT IT COSTS, because it is a real cost and run one has to read it. The
weakest evidence the rule will accept is one contiguous run of (L + n - 1)/2
tokens shared with the row. At a fixed 13 that run is (L + 12)/2; length-aware
it is (2L - 1)/3, i.e. two thirds of the item at every length:

    item tokens      13    16    20    25    30    38+
    run needed @13   12.5  14.5  16.0  18.5  21.0  unchanged
    run needed now    8.3  10.3  13.0  16.0  19.0  unchanged

So the exposure widens by at most ~4.5 tokens, and only for items under 38.
The new false-positive class is a SHORT eval question two thirds of which is a
statutory phrase a training row also quotes. That is the same trade the
statute exception is calibrated on, one length band down, and it is why the
narrow level is counted separately: if `narrow` drops dominate run one, the
calibrated relaxation is to raise the divisor (3 -> 2.5 costs the one-edit
guarantee below ~20 tokens and buys two tokens of evidence back). Do not
change it before reading that count.

WHAT IS STILL NOT DELIVERED, stated rather than implied:

* Below 38 tokens the tolerance is EXACTLY ONE substituted token and it is a
  CLIFF, not a slope. The window is chosen so an item's gram count lands on
  2*window, which is what makes one central edit score exactly 0.5 - and two
  edits then destroy 2*window >= all of them, so containment goes straight to
  ZERO with no band in between. Above 38 the window stops shrinking and the
  slope returns (a 100-token item scores 0.85 after one edit and 0.26 after
  five). Measured in
  test_the_narrow_band_tolerates_exactly_one_edit_and_no_more.
* A 5% word-level paraphrase measures C ~ 0.33 at 400 tokens and above, ~0.26
  at 100 and ~0.07 at 40 - i.e. it is never caught by any exact rule at this
  threshold, and SHORTER items are worse, not better. That is what the
  semantic layer is for, and the semantic layer is optional and unverified.

WHERE THIS DEPARTS FROM THE PLAN, and the measurement that forced it
--------------------------------------------------------------------
The plan (and the brief) specify "13-gram Jaccard >= 0.8". Jaccard cannot
express the leak this module is for. A 30-token eval question sitting
verbatim inside a 2,500-token training row scores

    Jaccard = |A n B| / |A u B| ~ 18 / 2,500 ~ 0.007

- three orders of magnitude below 0.8 - while its containment in the row is
  1.0. (Measured on the fixture in
  test_the_jaccard_rule_the_brief_asks_for_cannot_see_a_verbatim_leak.) A Jaccard threshold only fires when the two texts are
  the SAME LENGTH and nearly identical, i.e. when the training row IS the
  eval item. That case is real but rare; the common one is quotation inside a
  longer row, and Jaccard is blind to it.

So the rule is CONTAINMENT, at 0.5, and Jaccard is still computed and
recorded on every drop as a diagnostic. Jaccard is NOT kept as a second rule
because it would be a branch with no case of its own: J <= C always
(|A n B|/|A u B| <= |A n B|/|B|), so {J >= 0.8} is a strict subset of
{C >= 0.5} and the branch could never fire alone. Pinned by
test_the_jaccard_rule_the_brief_asks_for_is_subsumed_by_containment.

Containment at 0.5 (rather than "one shared 13-gram", the GPT-3/LLaMA rule)
is what keeps the spec's standing exception: a shared STATUTE QUOTATION alone
is not contamination - statutes are the domain being taught. One quoted
provision inside a 200-token eval item is ~10% of its grams and survives;
the question itself reproduced is 100% and does not.

THE CANDIDATE STEP CONTRIBUTES ZERO FALSE NEGATIVES
---------------------------------------------------
Candidates come from an inverted index over eval-item 13-gram hashes: any row
with containment > 0 necessarily shares at least one 13-gram with the item,
so the index is EXACT and the expensive containment/Jaccard arithmetic runs
only on candidates. MinHashLSH is deliberately NOT used here - LSH is an
approximation whose error direction is MISSED PAIRS, and missed pairs are the
catastrophic direction. The exactness is asserted against brute force in
test_the_gram_index_finds_every_pair_brute_force_finds, not merely claimed.

Gram hashes are a rolling polynomial over crc32 token hashes, NOT Python's
`hash()`: `hash()` of a str is salted per process, so an index built with it
would give a different candidate set (and therefore a different dataset) on
every run.

WHAT THIS MODULE CANNOT SEE (read before trusting a green run)
--------------------------------------------------------------
* A PARAPHRASED eval question. n-grams are defeated by rewriting; the
  semantic layer (semhash) is the intended answer and is optional - its
  status is recorded in the manifest either way, and `--require-semantic`
  turns its absence into a refusal.
* An eval item under 5 tokens (counted PER SET - and a set whose items
  are all under it is refused, because a screen that compared against
  nothing is not a screen).
* A row whose case identity nothing records. `case_identifier_coverage` in
  the manifest is that instrument: if it reads 0, the case-identifier level
  did not run, and the run says so loudly rather than reporting clean. NOTHING
  in the pipeline
  populates seed.cnr or seed.neutral_citation today (grep-verified), so on
  the first real build this level rests entirely on identifiers found in the
  row text and on `_prov`.

Build:  python -m tuned.data.decontaminate --config configs/data_law_v1.yaml
        [--in PATH]... [--out PATH] [--allow-missing-eval KEY]...
        [--no-generated] [--no-case-id-from-text] [--require-semantic]
"""

import csv
import hashlib
import json
import math
import os
import re
import zlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tuned.data.acquire import HF_SOURCES
from tuned.data.citations import extract_citations
from tuned.data.select import landmark_key

OUT_FILENAME = "decontaminated.jsonl"
DROPS_FILENAME = "decontamination_drops.jsonl"
MANIFEST_FILENAME = "decontamination.json"

# Bump when a rule changes: the manifest records it, so a dataset card can say
# which decontamination rules produced the corpus it describes.
#
# 2: the manifest gained an `output` block (path, rows, sha256) and dedupe.py
#    binds its chain of custody to that DIGEST rather than to the directory
#    the manifest happens to sit in.
DECON_VERSION = 2

# The n-gram window AT ITS WIDEST. 13 tokens is long enough that ordinary
# legal phrasing does not collide by accident; window_for narrows it for items
# too short to tolerate an edit at 13.
NGRAM = 13
# Share of an eval item's grams that must appear in the row. See the docstring
# for why this is containment and not Jaccard, and why it is not "one shared
# gram".
CONTAINMENT = 0.5
# An eval item shorter than NGRAM is matched whole; below this it is matchable
# by nothing (a 3-token question would drop half the corpus) and is counted.
SHORT_MIN_TOKENS = 5
# A case title shorter than this is not a usable identifier ("state v kumar").
TITLE_MIN_TOKENS = 4

LEVEL_TEXT = "text"
LEVEL_NARROW = "narrow"
LEVEL_SHORT = "short"
LEVEL_CASE_ID = "case_id"
LEVEL_SEMANTIC = "semantic"
# The order they are reported in: cheapest and most certain first. The
# semantic layer is not in NGRAM_LEVELS because it is not part of the exact
# screen - it only ever adds drops on top of it.
NGRAM_LEVELS = (LEVEL_CASE_ID, LEVEL_TEXT, LEVEL_NARROW, LEVEL_SHORT)
LEVELS = NGRAM_LEVELS + (LEVEL_SEMANTIC,)


class DecontaminationError(RuntimeError):
    """The pass cannot run as asked. Actionable by construction: the message
    says what to do, not what raised."""


# --------------------------------------------------------------------------
# Text primitives. dedupe.py imports these - ONE definition of what a row's
# text is, so the two passes cannot disagree about it.
# --------------------------------------------------------------------------

# `[^\W_]+` and NOT `[a-z0-9]+`: an ASCII-only class drops every Devanagari
# word, so a Hindi BhashaBench question leaked verbatim into a row would
# compare as a handful of digits, match nothing, and the row would pass this
# screen by being unreadable rather than by being clean. BhashaBench-Legal
# ships Hindi, so that is a leak this module would have reported as healthy.
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_MASK = (1 << 64) - 1
_BASE = 1_000_003


def tokens(text: str) -> tuple[str, ...]:
    """Case-folded word tokens IN ANY SCRIPT - the unit both passes compare in.

    Punctuation, markup and whitespace are dropped, so a row that differs
    from an eval item only in typesetting still matches it. Digits are KEPT:
    section numbers and years are most of what distinguishes one legal
    question from another.
    """
    return tuple(_TOKEN.findall((text or "").lower()))


def gram_hashes(toks: Sequence[str], n: int = NGRAM) -> frozenset[int]:
    """Stable 64-bit hashes of every n-token window.

    STABLE is the load-bearing word. Python's `hash()` for str is salted per
    process (PYTHONHASHSEED), so an index keyed on it selects a different
    candidate set - and therefore a different dataset - on every run.
    crc32 is deterministic across runs, machines and Python versions; the
    rolling polynomial over it is what keeps this one multiply per token
    rather than n.

    A 64-bit collision can only ever ADD to a posting count, never remove
    from one, so the error direction is the safe one - it can cost a row, not
    hide a leak. It is NOT rejected downstream, though: the posting count IS
    the intersection size the arithmetic divides, so a collision inflates
    containment directly rather than proposing a candidate something later
    throws out. (The comment here used to claim the latter, which would have
    been a stronger guarantee than the code gives.)
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if len(toks) < n:
        return frozenset()
    hs = [zlib.crc32(t.encode("utf-8")) for t in toks]
    power = pow(_BASE, n, 1 << 64)
    window = 0
    for value in hs[:n]:
        window = (window * _BASE + value) & _MASK
    out = [window]
    for i in range(n, len(hs)):
        window = (window * _BASE - hs[i - n] * power + hs[i]) & _MASK
        out.append(window)
    return frozenset(out)


def window_for(length: int, *, n: int = NGRAM, short_min: int = SHORT_MIN_TOKENS) -> int:
    """The n-gram window an eval item of `length` tokens is screened at.

    0 means NOTHING here can match it (the item is under the floor).

    THE DERIVATION, not a tuning knob. One substituted token destroys every
    gram covering it - n of them, centrally. An item of L tokens has
    G = L - n + 1 grams, so a row carrying the rest of it holds (G - n)/G of
    them, which is >= CONTAINMENT (0.5) exactly when G >= 2n, i.e. when

        L >= 3n - 1.

    Turned round, the widest window that still tolerates one edit at length L
    is (L + 1) // 3. Above 3*NGRAM - 1 = 38 tokens that bound exceeds NGRAM
    and the cap takes over, so long items are screened exactly as before; the
    cap is what keeps a 1,200-token IL-TUR judgment from being gramed at 400,
    where three edits anywhere would take its containment to zero.

    Below NGRAM an item has no grams at all, so it is matched WHOLE - the
    window is its own length and nothing short of all of it counts. Below
    `short_min` it is matchable by nothing: `unmatchable`, counted per set,
    and a set that is entirely unmatchable is a refusal.
    """
    if length < short_min:
        return 0
    if length < n:
        return length
    return min(n, (length + 1) // 3)


def level_for(length: int, *, n: int = NGRAM, short_min: int = SHORT_MIN_TOKENS) -> str | None:
    """Which text level screens an item of this length, or None for nothing.

    Three bands, and each carries a case the others cannot:
    `text` a 150-token judgment summary quoted at 55% of its grams;
    `narrow` a 20-token MCQ question with ONE word changed (containment 0.000
    at a fixed 13, and not a whole-sequence match either);
    `short` a 9-token question reproduced entire.
    """
    window = window_for(length, n=n, short_min=short_min)
    if not window:
        return None
    if window >= n:
        return LEVEL_TEXT
    if length < n:
        return LEVEL_SHORT
    return LEVEL_NARROW


def jaccard_from(shared: int, n_a: int, n_b: int) -> float:
    """|a n b| / |a u b| from the three counts.

    The counted form is the definition because the index already knows how
    many grams a candidate shares (it counted the postings), and re-deriving
    it with a set intersection would be both slower and a second answer to
    the same question.
    """
    if not n_a or not n_b:
        return 0.0
    return shared / (n_a + n_b - shared)


def containment_from(shared: int, n_part: int) -> float:
    """Share of `part` that the other side also holds."""
    return (shared / n_part) if n_part else 0.0


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    """Set form of jaccard_from - two texts too short to gram are not
    'identical', so the empty case reads 0.0 rather than raising."""
    return jaccard_from(len(a & b), len(a), len(b))


def containment(part: frozenset[int], whole: frozenset[int]) -> float:
    """Share of `part` that `whole` also holds.

    Asymmetric on purpose: `part` is the eval item and `whole` is the
    training row, and the question is "how much of the eval item is in this
    row", not "how similar are they".
    """
    return containment_from(len(part & whole), len(part))


# --------------------------------------------------------------------------
# The row contract.
#
# An assembly row is {"messages": [...], "_prov": {...}} - replay.py and
# curated.py's shape, and the shape store rows are lifted into below. The row
# dict is never mutated: everything this pass derives lives on the Item beside
# it, so the bytes written out are the bytes read in.
# --------------------------------------------------------------------------

def row_messages(row) -> list[dict]:
    messages = row.get("messages")
    return [m for m in messages if isinstance(m, dict)] if isinstance(messages, list) else []


def row_prompt(row) -> str:
    """Everything shown to the model - the question and its grounding."""
    return "\n".join(
        str(m.get("content") or "") for m in row_messages(row) if m.get("role") != "assistant"
    )


def row_answer(row) -> str:
    return "\n".join(
        str(m.get("content") or "") for m in row_messages(row) if m.get("role") == "assistant"
    )


def row_prov(row) -> dict:
    prov = row.get("_prov")
    return prov if isinstance(prov, dict) else {}


def row_form(row) -> str:
    """The QUESTION FORM this row is an instance of.

    Two rows built on the same judgment in two different forms are two
    examples, not one duplicate - which is why dedupe.py's prompt rule and
    the per-case cap both read this. Falls back to the stream/source so rows
    that carry no task identity are still grouped by something real.
    """
    prov = row_prov(row)
    for key in ("form", "task_type", "prompt_id", "stream", "source"):
        value = prov.get(key)
        if value:
            return str(value)
    return ""


# CNR: 16 characters - 4-letter state/court code, 2-digit establishment,
# 6-digit case number, 4-digit year (ESCR010004512020). Written with or
# without separators; fenced on both sides so it cannot be carved out of a
# longer alphanumeric run.
_CNR = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{4}[-]?\d{2}[-]?\d{6}[-]?\d{4})(?![A-Za-z0-9])")

_ID_FIELDS = {
    "cnr": ("cnr", "cnr_number", "case_number_cnr"),
    "cit": ("neutral_citation", "citation", "law_report_citation", "case_citation"),
    "title": ("case_name", "case_title", "title", "name", "native_id"),
}


def cnr_keys(text: str) -> list[str]:
    """Normalised CNRs in `text` (separators dropped, upper-cased)."""
    return [m.group(1).replace("-", "").upper() for m in _CNR.finditer(text or "")]


def title_key(text: str) -> str | None:
    """A case name as a join key, or None if it is too generic to be one.

    select.landmark_key is the normalisation - one definition of "these two
    strings name the same case" for the whole build. The length floor is what
    keeps 'state v kumar' from joining half the criminal docket.
    """
    key = landmark_key(text or "")
    return key if len(key.split()) >= TITLE_MIN_TOKENS else None


def identifiers_from_fields(fields) -> set[str]:
    """Namespaced identifiers from explicit metadata (never from prose)."""
    out: set[str] = set()
    for namespace, names in _ID_FIELDS.items():
        for name in names:
            value = fields.get(name)
            if not value:
                continue
            text = str(value)
            if namespace == "cnr":
                out.update(f"cnr:{key}" for key in cnr_keys(text))
            elif namespace == "cit":
                out.update(f"cit:{key}" for key in extract_citations(text))
            else:
                key = title_key(text)
                if key:
                    out.add(f"title:{key}")
    return out


def identifiers_from_text(text: str) -> set[str]:
    """Identifiers a passage NAMES: CNRs and citations, never titles.

    Titles are not extractable from prose without inventing a parser, and a
    guessed one would generate identifiers no other side can match. The
    citations found here are the authorities the passage cites AS WELL AS its
    own - which is deliberate on the eval side and the reason the row side of
    this channel is separately counted and switchable (see
    `case_ids_from_text`): if one landmark citation turns out to account for
    a large share of the drops, that is the citation graph, not
    contamination, and the manifest's `top_identifiers` is where it shows up.
    """
    out = {f"cnr:{key}" for key in cnr_keys(text)}
    out.update(f"cit:{key}" for key in extract_citations(text or ""))
    return out


@dataclass(frozen=True)
class Item:
    """One candidate row plus everything derived from it, computed once.

    `row` is untouched: the output file is the input rows minus the drops,
    byte for byte.
    """

    row: dict
    origin: str
    key: str
    prompt: str
    answer: str
    form: str
    identifiers: frozenset[str]

    @property
    def text(self) -> str:
        return f"{self.prompt}\n{self.answer}"


def item_key(prompt: str, answer: str) -> str:
    """Stable content id for a row - the same bytes always get the same key."""
    payload = f"{prompt}\n\x00\n{answer}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def item_of(row: dict, origin: str, *, ids_from_text: bool = True) -> Item:
    prompt, answer = row_prompt(row), row_answer(row)
    prov = row_prov(row)
    identifiers = identifiers_from_fields(prov)
    if ids_from_text:
        # The PROMPT only. An identifier in the answer is an authority the
        # model cited, not the case the row is about.
        identifiers |= identifiers_from_text(prompt)
    return Item(
        row=row,
        origin=origin,
        key=item_key(prompt, answer),
        prompt=prompt,
        answer=answer,
        form=row_form(row),
        identifiers=frozenset(identifiers),
    )


# --------------------------------------------------------------------------
# The eval corpora.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalSet:
    """One eval corpus: acquire.py owns WHERE it comes from, this owns WHY.

    repo_id/license/source_id are read off acquire.HF_SOURCES rather than
    repeated here - two spellings of a repo id is how the acquire command
    this module prints stops matching the source id it then looks up.
    """

    key: str
    why: str
    split: str | None = None
    # How many rows this set is DOCUMENTED to hold, or None where nobody has
    # written the number down. Never checked against the Hub - see
    # EVAL_MIN_SHARE for what is and is not done with it.
    expect_rows: int | None = None

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
# printed, never a quiet skip. The ROW COUNTS below are still unverified.
EVAL_SETS = {
    "bbl": EvalSet(
        key="bbl",
        why="the project's headline forgetting guard (24,365 questions)",
        # Test split only: the train split is not what the model is measured
        # on, and screening against it would cost yield for nothing.
        split="test",
        expect_rows=24_365,
    ),
    "iltur": EvalSet(
        key="iltur",
        why="8 Indian legal tasks; CJPE shares appellate pools with PredEx",
    ),
    "aibe": EvalSet(
        key="aibe",
        why="the bar-exam MCQ set Aalap measured against",
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

# The floor is a HUNDREDTH of the documented count, not a half, and the reason
# is that the documented count comes from the same unverified source as the
# repo ids. A floor that refuses a CORRECT download would push the operator
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

    @property
    def ok(self) -> bool:
        return self.status == EVAL_OK

    @property
    def shortfall(self) -> int:
        """How many rows short of the documented count, or 0."""
        expect = self.spec.expect_rows
        return max(0, expect - self.rows) if expect else 0


def read_rows(path: Path) -> Iterator[dict]:
    """Rows out of one snapshot file, dispatched on suffix.

    jsonl/json/csv/tsv are read here in pure Python so the load path is
    exercised offline. Parquet - what HF snapshots usually ship - is behind a
    lazy pyarrow import and HAS NEVER EXECUTED in this worktree (pyarrow is in
    the [build] extra and is not installed); see the module residuals.
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


def eval_corpus(store, spec: EvalSet, *, reader=read_rows) -> EvalCorpus:
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
    paths = [
        (key, Path(row["local_path"]))
        for key, row in sorted(index.items())
        if Path(key).suffix.lower() in _READABLE_SUFFIXES
    ]
    if spec.split:
        # Prefer the named split when the layout names it, but never filter
        # everything away: an unmatched split leaves the whole set in, which
        # over-screens (safe) rather than screening nothing (not safe).
        #
        # A NAME COMPONENT, not a substring: `latest.parquet` contains "test",
        # and a bare substring test would select it and silently screen
        # against the wrong file. Over-screening is safe; screening the wrong
        # file is not.
        split_paths = [p for p in paths if spec.split in _name_parts(p[0])]
        paths = split_paths or paths
    if not paths:
        return EvalCorpus(
            spec, EVAL_NO_FILES, detail=f"{len(index)} objects, none of {_READABLE_SUFFIXES}"
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
                spec, EVAL_NO_READER, files=len(paths),
                detail=f"{key}: {exc}",
            )
        except Exception as exc:  # a corrupt or unreadable snapshot file
            return EvalCorpus(
                spec, EVAL_UNREADABLE, files=len(paths),
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
            spec, EVAL_NO_TEXT_COLUMN, files=len(paths), rows=rows,
            detail=f"{rows} rows, none carrying any of {_EVAL_TEXT_FIELDS}",
        )
    if not items:
        return EvalCorpus(spec, EVAL_EMPTY, files=len(paths), rows=rows, detail="0 rows")
    corpus = EvalCorpus(
        spec, EVAL_OK, items=items, files=len(paths), rows=rows,
        text_field=max(winners, key=winners.get),
        unmatchable=sum(1 for item in items if not window_for(len(tokens(item.text)))),
    )
    floor = math.ceil((spec.expect_rows or 0) * EVAL_MIN_SHARE)
    if floor and rows < floor:
        corpus.status = EVAL_TOO_FEW
        corpus.detail = (
            f"{rows} rows, and this set is documented to hold {spec.expect_rows}. "
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
                 keys: Iterable[str] | None = None) -> dict[str, EvalCorpus]:
    allowed = set(allow_missing)
    out: dict[str, EvalCorpus] = {}
    for key in (sorted(EVAL_SETS) if keys is None else list(keys)):
        corpus = eval_corpus(store, EVAL_SETS[key], reader=reader)
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
    if corpus.status == EVAL_TOO_FEW:
        return (
            f"check the split and the shard count under {spec.url}\n"
            f"               (if {spec.expect_rows} is simply the wrong number, fix\n"
            f"                EvalSet.expect_rows - it has never been checked against the\n"
            f"                Hub. Do NOT waive the set to get past this: a waived BBL is\n"
            f"                the failure this module exists to prevent)"
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


# --------------------------------------------------------------------------
# The index: exact candidate generation over the eval side.
# --------------------------------------------------------------------------

class EvalIndex:
    """Inverted index over eval-item n-grams, ONE TABLE PER WINDOW.

    EXACT: containment > 0 requires at least one shared gram AT THAT ITEM'S
    WINDOW, so every pair the arithmetic could drop is generated here. Nothing
    approximate (LSH, embeddings) stands between an eval item and its
    candidate row - the error direction of an approximation is missed pairs,
    which is the one direction this module may not have.

    The window is a property of the ITEM (see window_for), so the index is
    keyed by it and a row is gramed once per window the index actually holds.
    A corpus of BBL-length questions occupies a handful of windows; the walk
    costs one pass over the row per window and nothing per item.
    """

    def __init__(self, items: Iterable[EvalItem], *, n: int = NGRAM,
                 short_min: int = SHORT_MIN_TOKENS):
        self.n = n
        self.short_min = short_min
        self.items: list[EvalItem] = []
        # COUNTS, not the sets. The posting walk already yields the exact
        # intersection size, so the only thing the arithmetic still needs from
        # an eval item is how many grams it had - and holding ~4,000 grams per
        # IL-TUR judgment as a Python set costs ~60 bytes each, which is the
        # difference between an index that fits and one that does not.
        self.gram_counts: list[int] = []
        self.windows: list[int] = []
        self.token_counts: list[int] = []
        # window -> {gram -> item indexes}
        self.by_gram: dict[int, dict[int, list[int]]] = {}
        self.unmatchable: list[str] = []
        self.by_identifier: dict[str, list[int]] = {}
        for ix, item in enumerate(items):
            toks = tokens(item.text)
            self.items.append(item)
            self.token_counts.append(len(toks))
            window = window_for(len(toks), n=n, short_min=short_min)
            self.windows.append(window)
            grams = gram_hashes(toks, window) if window else frozenset()
            self.gram_counts.append(len(grams))
            if window:
                table = self.by_gram.setdefault(window, {})
                for gram in grams:
                    table.setdefault(gram, []).append(ix)
            else:
                # Too short for any rule here. NOT silently ignored: it is
                # a hole in the screen and the manifest counts it, per set.
                self.unmatchable.append(item.item_id)
            for identifier in item.identifiers:
                self.by_identifier.setdefault(identifier, []).append(ix)

    def __len__(self) -> int:
        return len(self.items)

    def query(self, toks: Sequence[str]) -> dict[int, frozenset[int]]:
        """The row's grams at every window this index holds.

        Built once per row and handed to both `candidates` and the Jaccard
        diagnostic, so a row is never gramed twice at the same window.
        """
        return {window: gram_hashes(toks, window) for window in self.by_gram}

    def candidates(self, query: dict[int, frozenset[int]]) -> dict[int, int]:
        """Every eval item sharing >=1 gram with the row -> how many it shares.

        The count IS the intersection size (grams are a set on both sides), so
        containment and Jaccard both fall out of the posting walk and no
        candidate is ever re-intersected. That matters at the scale this runs
        at: one boilerplate gram shared with 5,000 eval items would otherwise
        cost 5,000 set intersections per row.
        """
        found: dict[int, int] = {}
        for window, grams in query.items():
            table = self.by_gram.get(window)
            if not table:
                continue
            for gram in grams:
                for ix in table.get(gram, ()):
                    found[ix] = found.get(ix, 0) + 1
        return found

    def level_of(self, ix: int) -> str | None:
        return level_for(self.token_counts[ix], n=self.n, short_min=self.short_min)

    def length_report(self) -> dict[str, dict]:
        """PER SET: how many items each level screens, and the token spread.

        A pooled count cannot tell an operator WHICH set is blind, and the
        window calibration above is an argument until this table is read
        against a real download - which is why it is in the manifest and on
        the first-run checklist rather than in a comment. `unmatchable` is the
        hole: items nothing here can match.
        """
        out: dict[str, dict] = {}
        for ix, item in enumerate(self.items):
            entry = out.setdefault(
                item.set_key,
                {LEVEL_TEXT: 0, LEVEL_NARROW: 0, LEVEL_SHORT: 0, "unmatchable": 0, "_t": []},
            )
            entry[self.level_of(ix) or "unmatchable"] += 1
            entry["_t"].append(self.token_counts[ix])
        for entry in out.values():
            spread = sorted(entry.pop("_t"))
            entry["min_tokens"] = spread[0]
            entry["median_tokens"] = spread[len(spread) // 2]
            entry["max_tokens"] = spread[-1]
        return dict(sorted(out.items()))

    def identifier_candidates(self, identifiers: Iterable[str]) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        for identifier in sorted(identifiers):
            for ix in self.by_identifier.get(identifier, ()):
                out.append((identifier, ix))
        return out


# --------------------------------------------------------------------------
# The decision.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Hit:
    level: str
    eval_set: str
    item_id: str
    detail: dict


def hits_for(item: Item, index: EvalIndex, *, threshold: float = CONTAINMENT) -> list[Hit]:
    """Every level that fires on this row, in LEVELS order.

    All levels are evaluated rather than short-circuited: the first is the
    drop reason, and the rest are what says whether a level is carrying cases
    of its own on real data or is dead weight the others subsume.
    """
    found: list[Hit] = []
    toks = tokens(item.text)
    query = index.query(toks)

    for identifier, ix in index.identifier_candidates(item.identifiers):
        hit = index.items[ix]
        found.append(Hit(LEVEL_CASE_ID, hit.set_key, hit.item_id, {"identifier": identifier}))
        break

    # The best hit PER LEVEL, not one overall: the levels are a union and the
    # counts are the instrument that says whether each branch is carrying
    # cases of its own on real data. One pooled "best" would hide a level that
    # never fires behind one that always does.
    best: dict[str, tuple[float, int, int]] = {}
    for ix, shared in sorted(index.candidates(query).items()):
        share = containment_from(shared, index.gram_counts[ix])
        if share < threshold:
            continue
        level = index.level_of(ix)
        if level is not None and (level not in best or share > best[level][0]):
            best[level] = (share, ix, shared)
    for level in (LEVEL_TEXT, LEVEL_NARROW, LEVEL_SHORT):
        if level not in best:
            continue
        share, ix, shared = best[level]
        hit = index.items[ix]
        window = index.windows[ix]
        found.append(
            Hit(
                level, hit.set_key, hit.item_id,
                # Jaccard rides along as a diagnostic: it is the number the
                # plan asked for as a rule, and recording it is how the first
                # real run shows how far below 0.8 a genuine leak scores. The
                # window and the item's length ride along because the level a
                # drop is filed under is a fact about the ITEM's length, and
                # the first run has to be able to read the two together.
                {"containment": round(share, 4),
                 "jaccard": round(
                     jaccard_from(shared, index.gram_counts[ix], len(query.get(window, ()))), 4
                 ),
                 "window": window,
                 "eval_tokens": index.token_counts[ix]},
            )
        )

    return sorted(found, key=lambda h: NGRAM_LEVELS.index(h.level))


def decontaminate_items(
    items: Iterable[Item],
    index: EvalIndex,
    *,
    threshold: float = CONTAINMENT,
    semantic=None,
) -> tuple[list[Item], list[dict], dict]:
    """Split candidate rows into (kept, drop records, stats).

    Order is the input's and is never re-sorted: two runs over the same input
    produce the same output bytes, which is what stops the train/test boundary
    downstream from moving between runs.
    """
    kept: list[Item] = []
    drops: list[dict] = []
    stats = {
        "total": 0,
        "kept": 0,
        "dropped": 0,
        "empty_text": 0,
        "with_identifier": 0,
        "by_reason": {},
        "by_level": dict.fromkeys(LEVELS, 0),
        "by_eval_set": {},
        "identifier_drops": {},
    }
    for item in items:
        stats["total"] += 1
        if item.identifiers:
            stats["with_identifier"] += 1
        found = hits_for(item, index, threshold=threshold)
        if not tokens(item.text):
            # A row that tokenises to NOTHING can match nothing, so it would
            # leave this pass looking screened while never having been
            # compared to anything. Counted, and printed by the CLI.
            stats["empty_text"] += 1
        if semantic is not None:
            found = found + list(semantic(item))
        if not found:
            kept.append(item)
            continue
        first = found[0]
        reason = f"{first.level}:{first.eval_set}"
        stats["dropped"] += 1
        stats["by_reason"][reason] = stats["by_reason"].get(reason, 0) + 1
        for hit in found:
            stats["by_level"][hit.level] = stats["by_level"].get(hit.level, 0) + 1
            stats["by_eval_set"][hit.eval_set] = stats["by_eval_set"].get(hit.eval_set, 0) + 1
            if hit.level == LEVEL_CASE_ID:
                key = hit.detail["identifier"]
                stats["identifier_drops"][key] = stats["identifier_drops"].get(key, 0) + 1
        drops.append(
            {
                "key": item.key,
                "origin": item.origin,
                "reason": reason,
                "form": item.form,
                "hits": [
                    {"level": h.level, "eval_set": h.eval_set, "item_id": h.item_id, **h.detail}
                    for h in found
                ],
            }
        )
    stats["kept"] = len(kept)
    return kept, drops, stats


# --------------------------------------------------------------------------
# The semantic layer (optional, and recorded either way).
# --------------------------------------------------------------------------

SEMANTIC_UNAVAILABLE = "semhash-not-installed"
# The seam answered in a shape this code cannot read, or answered wrongly on a
# pair whose answer is not in doubt. NOT "ran": see semantic_control.
SEMANTIC_UNUSABLE = "semhash-control-failed"
SEMANTIC_NO_ITEMS = "no-eval-items-to-compare"
SEMANTIC_RAN = "ran"

# The negative half of the control. Nothing in an Indian-law eval set is
# semantically near this, so a seam that flags it flags everything - which is
# the drift direction that would empty the corpus rather than pass it.
SEMANTIC_CONTROL_NEGATIVE = (
    "the sourdough starter doubled overnight so I shaped the loaf and baked it at "
    "two hundred and thirty degrees with steam for the first fifteen minutes"
)


class SemanticSeamError(RuntimeError):
    """semhash answered in a shape or a direction this code cannot use.

    Raised rather than defaulted around. A permissive `getattr(result,
    "selected", <something>)` on an API THIS PROJECT HAS NEVER EXECUTED is
    exactly the assertion the brief forbids: under a result object that names
    its survivors anything else, the default decides the answer and the run
    records a semantic layer that never compared anything.
    """


def semhash_available() -> bool:
    try:  # pragma: no cover - depends on the environment, not on logic
        import semhash  # noqa: F401
    except ImportError:
        return False
    return True


def selected_records(result) -> list:
    """`result.selected`, or a named error naming what came back instead.

    UNVERIFIED AGAINST A REAL INSTALL (semhash is in the [build] extra and is
    not installed here), which is precisely why there is no default: the
    attribute name is the one thing about this API that could be wrong, and a
    default turns being wrong about it into a silent clean bill of health.
    """
    try:
        selected = result.selected
    except AttributeError as exc:
        raise SemanticSeamError(
            f"semhash returned a {type(result).__name__} with no `.selected` "
            f"(it carries {sorted(vars(result))[:8] if hasattr(result, '__dict__') else 'no __dict__'}). "
            f"The API this module is written against is documented as "
            f"`SemHash.from_records(...).deduplicate(...).selected`; if the installed "
            f"version renames it, fix selected_records - do NOT default it, because a "
            f"default here reads as 'nothing was semantically similar'."
        ) from exc
    if selected is None:
        return []
    return list(selected)


class SemanticFilter:
    """Flags rows semantically close to an eval item.

    UNVERIFIED AGAINST A REAL INSTALL: written against the documented
    `SemHash.from_records(...).deduplicate(records=...)` shape, so the FIRST
    thing a machine with the extra must do is watch `semantic_control` pass.
    Everything the no-false-negative guarantee rests on (the n-gram levels and
    the case-identifier level) is pure Python above and is exercised offline;
    this layer only ever ADDS drops.
    """

    def __init__(self, eval_texts: Sequence[str], *, threshold: float = 0.9):
        from semhash import SemHash  # local import: absence is a status, not a crash

        self.threshold = threshold
        self.index = SemHash.from_records(records=list(eval_texts))

    def matches(self, text: str) -> bool:
        """Is this text a semantic duplicate of something in the eval index?"""
        result = self.index.deduplicate(records=[text], threshold=self.threshold)
        return not selected_records(result)

    def __call__(self, item: Item):
        if self.matches(item.text):
            return (Hit(LEVEL_SEMANTIC, "*", "semhash", {"threshold": self.threshold}),)
        return ()


def semantic_control(seam: SemanticFilter, positive: str) -> None:
    """Raise unless the seam is OBSERVED working, in both directions.

    `semantic: "ran"` in the manifest has to mean "this layer compared rows to
    eval items and the comparison worked", not "the call did not raise". The
    two ways it can be wrong point in opposite directions and both are silent:
    a result object whose survivors are named something else makes every row
    look like a duplicate (drops everything), and the reverse default makes
    every row look clean (drops nothing, which is this module's catastrophic
    direction). So the control pins both ends against answers that are not in
    doubt - an eval item is a duplicate of itself, and a bread recipe is not.
    """
    if not seam.matches(positive):
        raise SemanticSeamError(
            "the semantic layer did not flag an eval item against its own index. "
            "A layer that cannot recognise an exact copy cannot recognise a paraphrase, "
            "and recording it as having run would put a screen in the manifest that "
            "never screened anything."
        )
    if seam.matches(SEMANTIC_CONTROL_NEGATIVE):
        raise SemanticSeamError(
            "the semantic layer flagged text with nothing to do with Indian law, so it "
            "flags everything - every row would be dropped as contaminated."
        )


# --------------------------------------------------------------------------
# Loading the candidate rows.
# --------------------------------------------------------------------------

def stream_items(paths: Iterable[Path], *, ids_from_text: bool = True) -> Iterator[Item]:
    """Rows out of the assembly stream files, in file then line order."""
    from tuned.data.jsonl import read_jsonl

    for path in paths:
        for i, row in enumerate(read_jsonl(path)):
            yield item_of(row, f"{Path(path).name}#{i}", ids_from_text=ids_from_text)


def stream_paths(streams_dir: Path) -> list[Path]:
    """Every stream file, in sorted order.

    Sorted, not glob order: the input order is the survivor order downstream
    (dedupe keeps the first row of a duplicate cluster), so a filesystem's
    idea of directory order would decide which rows ship.
    """
    return sorted(p for p in Path(streams_dir).glob("*.jsonl") if p.is_file())


def generated_rows(store, cfg=None, *, state: str = "accepted") -> Iterator[dict]:
    """The accepted generations, as assembly rows.

    THE PROMPT IS THE GROUNDING, NOT THE RENDERED TEMPLATE. Every generated
    row is rendered from the same handful of prompt templates, so a text
    comparison over the rendered prompt would find every pair of rows ~80%
    identical - the template, not the content. What distinguishes these rows
    is the seed's grounding text and the model's answer, so that is what this
    pass compares. The rendered row is assemble.py's business and is built
    later, after this decision.

    A row the seed no longer backs still comes through (with an empty
    grounding) rather than being skipped: skipping it would let a row into the
    dataset that this pass never screened.
    """
    think_open = getattr(cfg, "think_open", "<think>")
    think_close = getattr(cfg, "think_close", "</think>")
    for gen in store.latest_generations(state):
        seed = store.get_seed(gen["seed_id"]) or {}
        think, answer = gen.get("think") or "", gen.get("answer") or ""
        content = f"{think_open}\n{think}\n{think_close}\n\n{answer}" if think else answer
        scores = [
            value
            for judgement in store.judgements_for(gen["gen_id"])
            for value in (judgement["grounding"], judgement["validity"], judgement["coverage"])
            if value is not None
        ]
        yield {
            "messages": [
                {"role": "user", "content": seed.get("text") or ""},
                {"role": "assistant", "content": content},
            ],
            "_prov": {
                "source": gen.get("stream"),
                "native_id": seed.get("native_id"),
                "cnr": seed.get("cnr"),
                "neutral_citation": seed.get("neutral_citation"),
                "task_type": gen.get("task_type"),
                "prompt_id": gen.get("prompt_id"),
                "seed_id": gen.get("seed_id"),
                "gen_id": gen.get("gen_id"),
                "score": round(sum(scores) / len(scores), 3) if scores else None,
                "reasoning": bool(think),
            },
        }


def store_items(store, cfg=None, *, state: str = "accepted",
                ids_from_text: bool = True) -> Iterator[Item]:
    for row in generated_rows(store, cfg, state=state):
        yield item_of(row, f"store:{row['_prov']['gen_id']}", ids_from_text=ids_from_text)


# --------------------------------------------------------------------------
# The manifest: what this pass could and could not see.
# --------------------------------------------------------------------------

def output_record(path: Path, rows: int) -> dict:
    """What this pass wrote, identified by CONTENT.

    The chain of custody downstream cannot be a directory: dedupe.py used to
    read the manifest beside its input and inherit whatever it found there, so
    an --in from anywhere else adopted a manifest describing OTHER ROWS and
    shipped never-screened rows under a decontamination stamp. A digest of the
    bytes actually written is the only claim that cannot be inherited by a
    file that was not screened.
    """
    from tuned.data.acquire import sha256_file

    return {"path": str(path), "rows": rows, "sha256": sha256_file(path)}


def manifest_of(stats: dict, corpora: dict[str, EvalCorpus], index: EvalIndex, *,
                inputs: Sequence[str], semantic: str, semantic_detail: str = "",
                output: dict | None = None, generations: dict | None = None,
                threshold: float = CONTAINMENT, ids_from_text: bool = True,
                top: int = 20) -> dict:
    """The record that has to outlive this run.

    Every waived eval set, every hole in the screen and every threshold is in
    here, because "was this dataset screened against BBL?" has to be
    answerable from the dataset itself years later - a printed warning is not
    an answer.
    """
    from tuned.data.store import utcnow

    coverage = stats["with_identifier"] / stats["total"] if stats["total"] else 0.0
    lengths = index.length_report()
    return {
        "stage": "decontaminate",
        "decon_version": DECON_VERSION,
        "at": utcnow(),
        "inputs": list(inputs),
        # The rows this pass produced, by content. dedupe.py verifies its own
        # input against this and refuses to inherit the record otherwise.
        "output": output,
        "thresholds": {
            "ngram": index.n,
            "containment": threshold,
            "short_min_tokens": index.short_min,
            # Below this many tokens an item is screened at a NARROWER window
            # than `ngram` - see window_for. The number an operator reading a
            # `narrow` drop count needs beside it.
            "full_ngram_from_tokens": 3 * index.n - 1,
            "title_min_tokens": TITLE_MIN_TOKENS,
            "case_ids_from_text": ids_from_text,
        },
        "counts": {k: stats[k] for k in ("total", "kept", "dropped", "empty_text")},
        "by_reason": dict(sorted(stats["by_reason"].items())),
        "by_level": dict(sorted(stats["by_level"].items())),
        "by_eval_set": dict(sorted(stats["by_eval_set"].items())),
        "eval_sets": {
            key: {
                "repo_id": corpus.spec.repo_id,
                "license": corpus.spec.license,
                "split": corpus.spec.split,
                "status": corpus.status,
                "allowed_missing": corpus.allowed_missing,
                "files": corpus.files,
                "rows": corpus.rows,
                "expect_rows": corpus.spec.expect_rows,
                "row_shortfall": corpus.shortfall,
                "items": len(corpus.items),
                "text_field": corpus.text_field,
                "detail": corpus.detail,
                # The token-length histogram, by the level that screens each
                # band. Read this against the window calibration on run one.
                "item_tokens": lengths.get(key),
            }
            for key, corpus in sorted(corpora.items())
        },
        # The holes, named. An eval item too short for any rule is not
        # screened against, and a corpus whose rows carry no case identifier
        # never met the case-identifier level.
        # Whether the accepted generations were screened at all. A run that
        # only looked at the stream files must not be indistinguishable, years
        # later, from one that looked at everything - and neither must a run
        # that asked for them and got none back.
        "generations_screened": bool((generations or {}).get("screened")),
        "generations": generations or {"screened": False, "state": None, "read": 0},
        # PER SET. A pooled number cannot tell the operator which set is
        # blind, and the refusal above needs the per-set count anyway.
        "unmatchable_eval_items": {
            key: entry["unmatchable"] for key, entry in sorted(lengths.items())
        },
        "unmatchable_eval_items_total": len(index.unmatchable),
        "case_identifier_coverage": round(coverage, 4),
        "case_identifier_level_inert": not stats["with_identifier"] or not index.by_identifier,
        "eval_identifiers": len(index.by_identifier),
        # "ran" here means OBSERVED WORKING - semantic_control passed on a pair
        # whose answer is not in doubt - not merely "called". See
        # semantic_control for the two silent directions it pins.
        "semantic": semantic,
        "semantic_detail": semantic_detail,
        "top_identifiers": sorted(
            stats["identifier_drops"].items(), key=lambda kv: (-kv[1], kv[0])
        )[:top],
    }


def write_manifest(path: Path, manifest: dict) -> None:
    """Durably, same rule as everything else here: written to a sibling .tmp
    and renamed, so the manifest is either absent or whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None, *, reader=read_rows) -> int:
    import argparse

    from tuned.data.config import load_build_config
    from tuned.data.jsonl import write_jsonl
    from tuned.data.paths import build_paths
    from tuned.data.store import TASK_STATES, Store

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data_law_v1.yaml")
    parser.add_argument("--in", dest="inputs", action="append", default=None,
                        help="stream JSONL to screen (repeatable; default every *.jsonl "
                             "in the streams dir)")
    parser.add_argument("--out", default=None, help=f"default out/{OUT_FILENAME}")
    parser.add_argument("--allow-missing-eval", action="append", default=[], metavar="KEY",
                        choices=sorted(EVAL_SETS),
                        help="run without this eval set - RECORDED IN THE MANIFEST")
    parser.add_argument("--no-generated", action="store_true",
                        help="screen the stream files only, not the accepted generations")
    parser.add_argument("--no-case-id-from-text", action="store_true",
                        help="take a row's case identifiers from _prov only, not from the "
                             "citations its grounding names")
    parser.add_argument("--require-semantic", action="store_true",
                        help="refuse to run when semhash is unavailable")
    parser.add_argument("--state", default="accepted", choices=TASK_STATES,
                        help="task state to read (default accepted). Validated: a typo'd "
                             "state used to read zero generations and still record them "
                             "as screened, and --state '' used to disable the filter "
                             "entirely and ship REJECTED generations")
    args = parser.parse_args(argv)

    cfg = load_build_config(args.config)
    paths = build_paths(cfg.build.workdir).ensure()
    out_path = Path(args.out) if args.out else paths.out_dir / OUT_FILENAME
    ids_from_text = not args.no_case_id_from_text

    store = Store.open(paths.state_db)
    try:
        corpora = eval_corpora(store, allow_missing=args.allow_missing_eval, reader=reader)
        blocked = refusals(corpora)
        if blocked:
            print(REFUSAL_HEADER)
            for line in blocked:
                print(f"  {line}")
            print("  nothing was written; no output carries a decontaminated stamp.")
            return 2

        semantic_fn, semantic_status, semantic_detail = None, SEMANTIC_UNAVAILABLE, ""
        texts = [i.text for c in corpora.values() for i in c.items]
        if semhash_available() and texts:
            try:
                seam = SemanticFilter(texts)
                semantic_control(seam, texts[0])
            except Exception as exc:
                # NOT recorded as "ran". The seam is installed and answered;
                # it answered in a way that means nothing was compared.
                #
                # Broad on purpose, and only around CONSTRUCTION plus the
                # control: semhash builds an embedding index and fetches a
                # model, so on a machine with the extra installed and no
                # network this raises something that is not a SemanticSeamError
                # - and the module's contract is that this layer's absence is a
                # STATUS, not a crash that kills a decontamination run the
                # refusal ladder has already cleared. Once the control passes,
                # the shape is confirmed and the per-row path is not wrapped.
                semantic_status = SEMANTIC_UNUSABLE
                semantic_detail = (
                    str(exc) if isinstance(exc, SemanticSeamError)
                    else f"{type(exc).__name__}: {exc}"
                )
            else:
                semantic_fn, semantic_status = seam, SEMANTIC_RAN
        elif semhash_available():
            semantic_status = SEMANTIC_NO_ITEMS
        if args.require_semantic and semantic_status != SEMANTIC_RAN:
            print(
                f"REFUSING TO DECONTAMINATE: --require-semantic was passed and the semantic "
                f"layer is {semantic_status}.\n"
                f"  {semantic_detail or 'run: pip install -e .[build]'}"
            )
            return 2

        inputs = (
            [Path(p) for p in args.inputs] if args.inputs else stream_paths(paths.streams_dir)
        )
        missing = [str(p) for p in inputs if not Path(p).exists()]
        if missing:
            print(f"no such input: {', '.join(missing)}")
            return 2

        items = list(stream_items(inputs, ids_from_text=ids_from_text))
        input_names = [str(p) for p in inputs]
        generations = {"screened": False, "state": None, "read": 0}
        if not args.no_generated:
            generated = list(store_items(store, cfg, state=args.state, ids_from_text=ids_from_text))
            print(f"read {len(items)} stream rows and {len(generated)} {args.state} generations")
            items += generated
            input_names.append(f"store:{args.state}")
            # What ACTUALLY happened, not what was asked for: `true` used to
            # mean only that --no-generated was absent, so a run that read
            # nothing at all from the store recorded the generations as
            # screened.
            generations = {
                "screened": bool(generated), "state": args.state, "read": len(generated),
            }
            if not generated:
                print(
                    f"    NO GENERATIONS IN STATE {args.state!r} - the store side of this run "
                    f"screened NOTHING, and the manifest records generations_screened: false. "
                    f"If tasks have been accepted, check --state."
                )
        else:
            print(f"read {len(items)} stream rows (generations NOT screened: --no-generated)")

        index = EvalIndex([item for c in corpora.values() for item in c.items])
        manifest_bands = index.length_report()
        for key in sorted(corpora):
            corpus = corpora[key]
            state = corpus.status if corpus.ok else f"{corpus.status} (WAIVED)"
            print(f"  eval {key:<6} {len(corpus.items):>7} items  {state}"
                  f"{'  <- ' + corpus.text_field if corpus.text_field else ''}")
            bands = manifest_bands.get(key)
            if bands:
                # The token-length histogram, printed where the operator is
                # already looking: which level each set is actually screened
                # by is what decides whether the window calibration holds.
                print(
                    f"    tokens: median {bands['median_tokens']}"
                    f" (min {bands['min_tokens']}, max {bands['max_tokens']})"
                    f"  screened by: {LEVEL_TEXT} {bands[LEVEL_TEXT]},"
                    f" {LEVEL_NARROW} {bands[LEVEL_NARROW]},"
                    f" {LEVEL_SHORT} {bands[LEVEL_SHORT]},"
                    f" unmatchable {bands['unmatchable']}"
                )
            if corpus.shortfall:
                # Not a refusal below the floor, but never silent either: the
                # documented count is the only handle on "this is a fragment
                # of the set" that does not require the network.
                print(
                    f"    {corpus.rows} rows, {corpus.spec.expect_rows} documented"
                    f" - {corpus.shortfall} SHORT. Check the split and the shard count"
                    f" before trusting this screen"
                )
            if corpus.ok and len(corpus.items) < corpus.rows:
                # Some rows of a set that IS loaded carried none of the
                # candidate column names. Screening against 10% of BBL while
                # the status reads `ok` is the shape this module exists to
                # refuse, so the shortfall is printed next to the status.
                print(
                    f"    {corpus.rows - len(corpus.items)} of {corpus.rows} rows carried none"
                    f" of {_EVAL_TEXT_FIELDS[:4]}... - THOSE QUESTIONS WERE NOT SCREENED"
                    f" AGAINST. Add the real column name to _EVAL_TEXT_FIELDS."
                )

        kept, drops, stats = decontaminate_items(items, index, semantic=semantic_fn)
        # Written FIRST, so the manifest can carry the digest of the bytes
        # that actually landed rather than of the bytes this pass intended.
        written = write_jsonl(out_path, [item.row for item in kept])
        manifest = manifest_of(
            stats, corpora, index, inputs=input_names, semantic=semantic_status,
            semantic_detail=semantic_detail, ids_from_text=ids_from_text,
            output=output_record(out_path, written), generations=generations,
        )
        write_jsonl(out_path.parent / DROPS_FILENAME, drops)
        write_manifest(out_path.parent / MANIFEST_FILENAME, manifest)
        store.log_event("decontamination", manifest)

        print(f"  screened {stats['total']}  kept {stats['kept']}  dropped {stats['dropped']}")
        for reason, count in sorted(stats["by_reason"].items()):
            print(f"    drop[{reason}]: {count}")
        print(
            f"  case identifiers: {stats['with_identifier']}/{stats['total']} rows carried one"
            f" ({manifest['case_identifier_coverage']:.1%}), eval side "
            f"{manifest['eval_identifiers']}"
        )
        if manifest["case_identifier_level_inert"]:
            # The Task-11 lesson: an instrument that reads healthy in exactly
            # the case it exists for. A level with nothing to match on did not
            # run, and must not be mistaken for a level that found nothing.
            print(
                "    THE CASE-IDENTIFIER LEVEL DID NOT RUN - one side carried no identifiers"
                " at all. That is the level that catches SAME JUDGMENT, DIFFERENT QUESTION,"
                " which no n-gram method can see. Nothing in the pipeline populates seed.cnr"
                " or seed.neutral_citation today; populate them, or accept that this run was"
                " screened on text alone (the manifest says which)."
            )
        for key, count in sorted(manifest["unmatchable_eval_items"].items()):
            if count:
                print(
                    f"    {key}: {count} eval items are under {SHORT_MIN_TOKENS} tokens and "
                    f"NOTHING here can match them"
                )
        if stats["empty_text"]:
            print(
                f"    {stats['empty_text']} candidate rows carried NO USABLE TEXT - they can "
                f"match nothing, so they pass this screen by being unreadable, not by being clean"
            )
        if not manifest["generations_screened"]:
            print(
                "    THE ACCEPTED GENERATIONS WERE NOT SCREENED (--no-generated). They are the "
                "rows built FROM the seed pools the eval sets draw on, so this run has not "
                "screened the corpus - recorded in the manifest as generations_screened: false"
            )
        if manifest["top_identifiers"]:
            print("  identifiers that cost the most rows (read this before trusting the yield):")
            for identifier, count in manifest["top_identifiers"][:5]:
                print(f"    {identifier:<48} {count}")
        if semantic_status != SEMANTIC_RAN:
            print(
                f"  semantic layer did NOT run ({semantic_status}) - paraphrased eval questions"
                f" are not screened. Recorded in the manifest."
            )
            if semantic_detail:
                print(f"    {semantic_detail}")
        for key in sorted(corpora):
            if corpora[key].allowed_missing:
                print(f"  WAIVED: {key} was not screened against ({corpora[key].status}) - "
                      f"this is recorded in {MANIFEST_FILENAME} and travels to the dataset card")
        print(f"wrote {written} rows -> {out_path}")
        print(f"      {len(drops)} drops -> {out_path.parent / DROPS_FILENAME}")
        print(f"      manifest -> {out_path.parent / MANIFEST_FILENAME}")
        if stats["total"] and not written:
            print(
                "  EVERYTHING WAS DROPPED from a non-empty input - that is a matcher fault, "
                "not a corpus: check the eval item counts above against the drop reasons."
            )
            return 1
        if not stats["total"]:
            print(
                "  NOTHING READ: no candidate rows at all. Exiting 0 here would stamp an "
                "empty dataset decontaminated - check --in and whether any task is accepted."
            )
            return 1
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    import os
    import sys

    exit_code = main()
    # Same reasoning as select.py/extract.py: pyarrow/hf-xet can leave
    # non-daemon threads that wedge interpreter shutdown after all output is
    # written. Skip shutdown entirely.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
