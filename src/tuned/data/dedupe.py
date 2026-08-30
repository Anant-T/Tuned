"""Remove near-duplicates within the accepted set, and cap one case at 3 rows.

Input is `out/decontaminated.jsonl` - decontaminate.py's output, and THAT
ORDER IS A CORRECTNESS PROPERTY, not a convention (see below). Output is
`out/deduped.jsonl`, a drop log carrying a machine-readable reason per dropped
row, and a manifest that carries decontamination's own manifest forward so the
chain of custody survives to stats.py and the dataset card.

THIS MODULE IS NOT IN DECONTAMINATION'S REGIME
----------------------------------------------
There, a false negative is invisible, permanent and inflates the headline
number, so everything borderline goes to dropping. Here both directions
merely cost quality: an over-eager drop costs a training example, a missed
duplicate costs a little repetition. Nothing in this file should be argued
from decontaminate.py's asymmetry, and nothing here refuses a run.

THE ORDER: DECONTAMINATE FIRST, THEN DEDUPE
-------------------------------------------
If dedupe ran first, a duplicate cluster whose kept representative is
contaminated and whose twin is clean would lose BOTH: the twin dies as a
duplicate, then the representative dies as contamination, and a usable row is
gone. Run the other way round the contaminated row is already out and the
clean twin is the representative. Pinned - in both directions - by
test_the_clean_twin_of_a_contaminated_row_ships and
test_running_dedupe_first_loses_the_clean_twin_as_well.

THE STACK
---------
1. EXACT - sha256 of the (prompt, answer) pair AS WRITTEN, with a NUL between
   the two halves so that a row whose prompt ends where another's answer
   begins is not the same key. NOTHING IS NORMALISED here (it used to say
   "normalised", which is a stronger claim than item_key makes): this rule is
   byte identity, and rules 2-3 are what catch everything softer. First
   occurrence in input order wins.
2. NEAR-PROMPT - Jaccard >= 0.85 over 5-grams of the prompt, WITHIN ONE
   QUESTION FORM (see the deviation below).
3. NEAR-ROW - Jaccard >= 0.90 over 5-grams of prompt+answer, across forms.
4. PER-CASE CAP - at most 3 rows per case identifier.

Every near-duplicate decision is an EXACT Jaccard computed on a candidate
pair; the candidate step never decides anything by itself. Which is why:

WHY THERE IS NO MinHashLSH HERE, and what replaces it
------------------------------------------------------
The plan specifies datasketch MinHashLSH (128 perm) as the candidate
generator. What is built instead is a PREFIX-FILTER inverted index, which is
EXACT: order grams by a fixed global key, index only the first
|A| - ceil(t*|A|) + 1 grams of each row, and any pair with Jaccard >= t must
collide there. (Proof, and the reason it is not a heuristic: if the prefixes
are disjoint and A's last prefix gram precedes B's, every shared gram must lie
in A's suffix, so |A n B| <= ceil(t|A|) - 1 < t|A| <= t|A u B| <= |A n B| -
a contradiction. Asserted against brute force in
test_the_prefix_index_finds_every_pair_brute_force_finds.)

That is a strict improvement on LSH for this corpus rather than a
substitution of convenience: LSH's error direction is missed pairs, exactness
costs nothing at 18k rows (measured: see the benchmark in the task report),
and it removes a dependency nothing here imports - datasketch has been taken
out of the [build] extra to match.

WHERE THIS DEPARTS FROM THE PLAN'S THRESHOLDS, and the measurement for it
--------------------------------------------------------------------------
"J >= 0.85 on prompts" is unsafe applied across question forms. A generated
row's prompt is the seed's GROUNDING TEXT plus a task instruction, and the
wave planner deliberately builds up to PER_SEED_CAP (4) tasks on one seed in
different forms. Those four prompts share ~2,500 tokens of grounding and
differ by an instruction, so their pairwise Jaccard is ~0.99 and a bare
prompt rule deletes three of every four - the exact diversity the per-case
cap exists to preserve. Measured on the four-forms-one-seed fixture:
without the form guard 3 of 4 rows are dropped, with it 0.
So rule 2 requires SAME FORM, and rule 3 (whole row, 0.90) is what catches a
genuine duplicate that changed its label.

THE PER-CASE CAP, and which three survive
------------------------------------------
The cap exists so one judgment cannot dominate the dataset, and so that
split.py's CNR-level assignment is not deciding the fate of fifty rows at a
time. WHICH three is a quality decision and it is made this way:

    DISTINCT QUESTION FORMS FIRST, then judge score, then a stable key.

Three rows on one case in three forms teach three things; three rows in one
form teach one thing three times, and the model has already seen this case
twice by then. So the first pass takes the best row of each form the group
has (forms visited by their best row), and only when the forms run out does
the second pass fill the remaining slots with the next-best rows. Ties break
on the row's content key, never on dict or set order.

A ROW WITH NO CASE IDENTIFIER IS NOT CAPPED. Grouping them under one None
bucket would cap the whole replay stream - thousands of general-reasoning
rows that share no case at all - at three rows. They are counted in the
manifest as `uncapped_rows` instead, which is also the instrument that says
how much of the corpus the cap could not reach.

DETERMINISM
-----------
Two runs over the same input produce byte-identical output WITH THE SEMANTIC
LAYER OFF, and that qualification is not a formality: everything the EXACT
stack decides is input order or a content hash, nothing iterates a set or a
dict of hashes to make a decision, and the gram hashes are crc32-based rather
than `hash()` (see decontaminate.gram_hashes) - but semhash searches an
approximate-nearest-neighbour index whose determinism this project has not
measured. The determinism test runs the whole pass twice under different
PYTHONHASHSEEDs and compares bytes, with the layer off and with a fake; the
REAL semhash path is a first-run item. If it turns out to be unstable, the
manifest's `semantic` field is what says which runs it touched.

Build:  python -m tuned.data.dedupe --config data/configs/data_law_v1.yaml
        [--in PATH] [--out PATH] [--no-cap]
"""

import array
import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from tuned.data.decontaminate import (
    MANIFEST_FILENAME as DECON_MANIFEST_FILENAME,
)
from tuned.data.decontaminate import (
    OUT_FILENAME as DECON_OUT_FILENAME,
)
from tuned.data.decontaminate import (
    SEMANTIC_CONTROL_ITEM,
    SEMANTIC_CONTROL_NEGATIVE,
    SEMANTIC_NO_MODEL,
    SEMANTIC_RAN,
    SEMANTIC_UNAVAILABLE,
    SEMANTIC_UNUSABLE,
    Item,
    SemanticModelError,
    SemanticSeamError,
    gram_hashes,
    jaccard_from,
    output_record,
    provenance_text,
    row_prov,
    selected_records,
    semhash_available,
    semhash_index,
    stream_items,
    tokens,
    write_manifest,
)

OUT_FILENAME = "deduped.jsonl"
DROPS_FILENAME = "dedupe_drops.jsonl"
MANIFEST_FILENAME = "dedupe.json"

# 2: the manifest gained `decontamination_check` - the chain of custody is
#    bound to the upstream output's DIGEST, not to the directory it sits in.
# 3: the semantic self-dedupe is scoped to one question form (unguarded, it
#    collapsed four tasks built on one seed into one at every threshold), its
#    threshold is recorded, and its drops name the form. The emitted set moves
#    wherever that layer runs.
# 4: the manifest gained `outputs` - a digest of the rows THIS pass wrote,
#    the same record decontaminate.py has carried since decon_version 2. Every
#    version-3 manifest asserted a chain of custody it could not itself be
#    checked against: split.py verifies its input against the digest dedupe
#    recorded, and there was none, so the chain ended here. Version-3
#    manifests are not stale in their COUNTS, only in what a downstream pass
#    can prove about them - split.py refuses one by name rather than
#    inheriting it.
DEDUPE_VERSION = 4

# Shorter window than decontamination's 13: this is a similarity question
# between two rows of comparable length, not a "did this text appear inside
# that one" question, and 5-grams are what the plan calibrated the thresholds
# against.
NGRAM = 5
PROMPT_JACCARD = 0.85
ROW_JACCARD = 0.90
# At most this many rows per case identifier.
CNR_CAP = 3
# The semantic self-dedupe threshold. NOT decontaminate's 0.8: this comparison
# is row against row - texts of comparable length, which is the regime cosine
# similarity is well behaved in - rather than a short eval question against a
# long row. Measured against the installed library (cache-only) on a rewritten
# 60-word judgment summary and a genuinely different row of the same corpus:
#
#   threshold           0.6  0.7  0.75  0.8  0.85  0.9  0.95
#   reworded row        [1]  [1]  [1]   [1]  [1]   [1]  []
#   two different rows  []   []   []    []   []    []   []
#
# The rewording is collapsed everywhere up to 0.9 and lost at 0.95, and no
# threshold in the range confuses two different rows, so 0.9 is the tightest
# point that still catches what this layer is for.
SEMANTIC_THRESHOLD = 0.9

REASON_EXACT = "exact"
REASON_NEAR_PROMPT = "near_prompt"
REASON_NEAR_ROW = "near_row"
REASON_CAP = "case_cap"
REASON_SEMANTIC = "semantic"
REASONS = (REASON_EXACT, REASON_NEAR_PROMPT, REASON_NEAR_ROW, REASON_SEMANTIC, REASON_CAP)


# --------------------------------------------------------------------------
# The exact candidate generator.
# --------------------------------------------------------------------------

def prefix_length(size: int, threshold: float) -> int:
    """How many of a row's grams must be indexed for exactness at `threshold`.

    |A| - ceil(t*|A|) + 1, the standard prefix-filter bound. At t=0.9 that is
    ~10% of the grams, which is where the memory saving comes from - and it is
    a BOUND, not a tuning knob: one gram fewer and a genuine pair can be
    missed, which is what test_the_prefix_bound_is_tight_in_both_directions
    pins.
    """
    if size <= 0:
        return 0
    return max(1, size - math.ceil(threshold * size) + 1)


class PrefixIndex:
    """Exact candidate generator for Jaccard >= threshold over gram sets.

    Grams are ordered by a fixed global key (their hash value), which is
    deterministic across runs because the hashes are crc32-derived. Any global
    order works for correctness; a stable one is what makes the RESULT stable.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.by_gram: dict[int, list[int]] = {}
        # Packed 8-byte-per-gram storage rather than a list of frozensets.
        # Exactness needs every indexed row's WHOLE gram set (the prefix is a
        # candidate filter, never the answer), and at ~1,800 grams a row a
        # Python set costs ~60 bytes each: measured 294 MB at 2,000 rows,
        # which extrapolates past 5 GB at the 18,000-row target. The array
        # holds the same grams in 8 bytes each and the verification counts
        # them against the query set without rebuilding one.
        self.grams: list[array.array] = []

    def candidates(self, grams: frozenset[int]) -> dict[int, int]:
        """Indexed rows sharing >=1 prefix gram with this one -> shared count.

        The count is over the PREFIX only, so it is a lower bound on the true
        intersection and is never used as the decision - `verified` recomputes
        the exact Jaccard on the full sets.
        """
        found: dict[int, int] = {}
        for gram in self._prefix(grams):
            for ix in self.by_gram.get(gram, ()):
                found[ix] = found.get(ix, 0) + 1
        return found

    def verified(self, grams: frozenset[int]) -> tuple[int, float] | None:
        """The best already-indexed row at or above the threshold, or None.

        EXACT Jaccard on every candidate: the index only proposes. Without
        this step the thresholds would be decoration - which is the whole
        reason the plan's LSH numbers (128 perm, J=0.85/0.9) mean anything.
        """
        best: tuple[int, float] | None = None
        for ix in sorted(self.candidates(grams)):
            stored = self.grams[ix]
            shared = sum(1 for gram in stored if gram in grams)
            score = jaccard_from(shared, len(stored), len(grams))
            if score >= self.threshold and (best is None or score > best[1]):
                best = (ix, score)
        return best

    def add(self, grams: frozenset[int]) -> int:
        ix = len(self.grams)
        ordered = sorted(grams)
        self.grams.append(array.array("Q", ordered))
        for gram in ordered[: prefix_length(len(ordered), self.threshold)]:
            self.by_gram.setdefault(gram, []).append(ix)
        return ix

    def _prefix(self, grams: frozenset[int]) -> list[int]:
        return sorted(grams)[: prefix_length(len(grams), self.threshold)]


# --------------------------------------------------------------------------
# The rows this pass reasons about.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """An Item with the two sort keys this pass needs.

    Deliberately does NOT carry its gram sets. Holding them for every row at
    once is what put a 2,000-row benchmark at 318 MB (extrapolating past 5 GB
    at the 18,000-row target); they are built inside the loop, used, and
    dropped, and only a SURVIVOR's grams stay - packed, in the index.
    """

    item: Item
    score: float | None
    case_id: str | None

    @property
    def key(self) -> str:
        return self.item.key

    def prompt_grams(self, n: int = NGRAM) -> frozenset[int]:
        return gram_hashes(tokens(self.item.prompt), n)

    def row_grams(self, n: int = NGRAM) -> frozenset[int]:
        return gram_hashes(tokens(self.item.text), n)


def case_id_of(item: Item) -> str | None:
    """The one identifier a row is capped under, or None.

    A row can carry several (a CNR and a citation and a title); the cap needs
    ONE bucket per case, so the strongest available is chosen in a fixed
    order - cnr, then citation, then title - and ties inside a namespace break
    on the sorted value. A row with no identifier gets None and is never
    capped (see the module docstring).
    """
    for namespace in ("cnr", "cit", "title"):
        matching = sorted(i for i in item.identifiers if i.startswith(f"{namespace}:"))
        if matching:
            return matching[0]
    return None


def score_of(item: Item) -> float | None:
    """The row's judge score if anything recorded one.

    `None` is NOT zero: a stream row nobody judged must not lose a cap slot to
    a generated row that scraped a 3, so unscored rows sort together and break
    on the stable key.
    """
    value = row_prov(item.row).get("score")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def candidate_of(item: Item) -> Candidate:
    return Candidate(item=item, score=score_of(item), case_id=case_id_of(item))


# --------------------------------------------------------------------------
# The near-duplicate pass.
# --------------------------------------------------------------------------

def dedupe_candidates(
    candidates: Sequence[Candidate],
    *,
    prompt_threshold: float = PROMPT_JACCARD,
    row_threshold: float = ROW_JACCARD,
) -> tuple[list[Candidate], list[dict], dict]:
    """Keep the first row of every duplicate cluster; returns (kept, drops, stats).

    FIRST IN INPUT ORDER wins, deliberately: the survivor of a cluster is then
    a function of the input file alone, and the input order is itself
    deterministic (sorted stream files, then gen_id order). Choosing the
    "best" member instead would make the survivor depend on a score that many
    rows do not have, and dedupe is not the stage that ranks quality - the
    per-case cap is.
    """
    # No "semantic" counter here: this function does not run that layer, and
    # the count it used to hold was incremented in dedupe_items and read by
    # nobody. by_reason is the one answer to "how many did each rule drop".
    stats = {
        "total": len(candidates),
        "kept": 0,
        "exact": 0,
        "near_prompt": 0,
        "near_row": 0,
    }
    drops: list[dict] = []
    kept: list[Candidate] = []
    seen_exact: dict[str, str] = {}
    # One prompt index PER FORM: rule 2 only compares rows of the same
    # question form, and keeping them apart is also what stops the index from
    # proposing thousands of same-grounding candidates it would then reject.
    prompt_indexes: dict[str, PrefixIndex] = {}
    prompt_keys: dict[str, list[str]] = {}
    row_index = PrefixIndex(row_threshold)
    row_keys: list[str] = []

    for candidate in candidates:
        item = candidate.item
        twin = seen_exact.get(item.key)
        if twin is not None:
            stats["exact"] += 1
            drops.append(_drop(candidate, REASON_EXACT, twin, {"sha256_16": item.key}))
            continue

        form = item.form
        index = prompt_indexes.setdefault(form, PrefixIndex(prompt_threshold))
        keys = prompt_keys.setdefault(form, [])
        prompt_grams = candidate.prompt_grams()
        hit = index.verified(prompt_grams) if prompt_grams else None
        if hit is not None:
            ix, score = hit
            stats["near_prompt"] += 1
            drops.append(
                _drop(candidate, REASON_NEAR_PROMPT, keys[ix],
                      {"jaccard": round(score, 4), "form": form})
            )
            continue

        row_grams = candidate.row_grams()
        hit = row_index.verified(row_grams) if row_grams else None
        if hit is not None:
            ix, score = hit
            stats["near_row"] += 1
            drops.append(
                _drop(candidate, REASON_NEAR_ROW, row_keys[ix], {"jaccard": round(score, 4)})
            )
            continue

        seen_exact[item.key] = item.key
        index.add(prompt_grams)
        keys.append(item.key)
        row_index.add(row_grams)
        row_keys.append(item.key)
        kept.append(candidate)

    stats["kept"] = len(kept)
    return kept, drops, stats


def _drop(candidate: Candidate, reason: str, twin: str | None, detail: dict) -> dict:
    return {
        "key": candidate.key,
        "origin": candidate.item.origin,
        "reason": reason,
        "duplicate_of": twin,
        "case_id": candidate.case_id,
        # Here rather than in `detail` so EVERY reason carries it - the exact
        # rule passes no form at all. See decontaminate's drop record for why
        # `form` cannot stand in for it.
        "source": (candidate.item.row.get("_prov") or {}).get("source") or "",
        **detail,
    }


# --------------------------------------------------------------------------
# The per-case cap.
# --------------------------------------------------------------------------

def cap_survivors(group: Sequence[Candidate], cap: int = CNR_CAP) -> list[Candidate]:
    """Which `cap` rows of one case survive - forms first, then score.

    Deterministic end to end: forms are visited in the order of their best
    row, "best" is (score desc, content key asc), and an unscored row sorts
    after a scored one at the same form rather than beating it by accident.
    """
    # No "small group" fast path. It would be equivalent in WHICH rows
    # survive (the ranking below returns every member of a group at or under
    # the cap) but not in the ORDER this function documents, so it is a branch
    # that can only ever disagree with the contract - and a mutation of its
    # boundary is unkillable precisely because the disagreement is invisible
    # to the one caller. Sorting three rows costs nothing.
    ranked = sorted(group, key=_rank)
    by_form: dict[str, list[Candidate]] = {}
    for candidate in ranked:
        by_form.setdefault(candidate.item.form, []).append(candidate)
    # Forms in the order their strongest row appears, so the first slot goes
    # to the best row overall and the second to the best row of a DIFFERENT
    # form even when the same form holds the two best rows.
    order = sorted(by_form, key=lambda form: _rank(by_form[form][0]))
    taken: list[Candidate] = []
    for form in order:
        if len(taken) >= cap:
            break
        taken.append(by_form[form][0])
    if len(taken) < cap:
        chosen = {c.key for c in taken}
        for candidate in ranked:
            if len(taken) >= cap:
                break
            if candidate.key not in chosen:
                taken.append(candidate)
                chosen.add(candidate.key)
    # Returned in the order they were CHOSEN (forms first, then fill), which
    # is what a caller reading "which three survived" wants to see. apply_cap
    # re-establishes input order for the file it writes.
    return taken


def _rank(candidate: Candidate) -> tuple:
    # -score first (higher is better), unscored last, then the content key.
    return (0, -candidate.score, candidate.key) if candidate.score is not None else (
        1, 0.0, candidate.key
    )


def apply_cap(
    candidates: Sequence[Candidate], *, cap: int = CNR_CAP
) -> tuple[list[Candidate], list[dict], dict]:
    """Drop everything past `cap` rows for any one case identifier."""
    groups: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        if candidate.case_id is not None:
            groups.setdefault(candidate.case_id, []).append(candidate)
    keep: set[str] = set()
    for case_id, group in groups.items():
        keep.update(c.key for c in cap_survivors(group, cap))
    kept: list[Candidate] = []
    drops: list[dict] = []
    stats = {
        "capped": 0,
        "uncapped_rows": sum(1 for c in candidates if c.case_id is None),
        "cases": len(groups),
        "cases_over_cap": sum(1 for g in groups.values() if len(g) > cap),
    }
    for candidate in candidates:
        if candidate.case_id is None or candidate.key in keep:
            kept.append(candidate)
            continue
        stats["capped"] += 1
        drops.append(
            _drop(candidate, REASON_CAP, None,
                  {"cap": cap, "case_rows": len(groups[candidate.case_id])})
        )
    return kept, drops, stats


# --------------------------------------------------------------------------
# The whole pass.
# --------------------------------------------------------------------------

def flagged_indexes(texts: Sequence[str], *, threshold: float = SEMANTIC_THRESHOLD) -> list[int]:
    """Which of `texts` semhash's self-deduplication did NOT keep.

    Written against `SemHash.from_records(...).self_deduplicate()`, and the
    survivors are read through decontaminate.selected_records, which RAISES
    rather than defaulting when that shape is not what came back. The default
    this replaced (`getattr(result, "selected", texts)`) read as "semhash kept
    everything" and so recorded a clean corpus indistinguishable from a
    screened one. Index CONSTRUCTION goes through decontaminate.semhash_index,
    so a model that cannot be fetched is its own status rather than the same
    one as a drifted API.

    Survivors are matched back BY COUNT, not by set membership: two candidates
    can carry the same text (item_key hashes the prompt and the answer
    separately, so "a\\nb" + "c" and "a" + "b\\nc" are different rows with the
    same joined text), and a set makes those two rows unflaggable no matter
    what semhash says about them.

    AND THEY ARE READ THROUGH THE SHARED HELPER, not coerced with `str`.
    semhash returns survivors in two shapes - plain strings, and single-key
    mappings under the column name it gave the records - and `str({"text":
    ...})` is `"{'text': ...}"`, which matches no row's text at all. Under the
    dict shape every `budget[text]` read 0 and EVERY row flagged as a duplicate
    of itself: the corpus-emptying direction, arrived at silently, in the
    function whose docstring cites the fail-loud rule as its protection. A
    survivor this helper cannot read is a shape neither seam knows, and it
    raises rather than being counted as a row nothing matches.
    """
    result = semhash_index(texts).self_deduplicate(threshold=threshold)
    kept = []
    for record in selected_records(result):
        text = provenance_text(record)
        if text is None:
            raise SemanticSeamError(
                f"semhash returned a survivor this build cannot read as text "
                f"({type(record).__name__}). Every survivor it does not recognise would "
                f"count as a row nothing matches, i.e. as a DUPLICATE, so the corpus would "
                f"empty itself quietly. Check the installed semhash version against "
                f"selected_records and provenance_text."
            )
        kept.append(text)
    budget = Counter(kept)
    flagged = []
    for ix, text in enumerate(texts):
        if budget[text] > 0:
            budget[text] -= 1
        else:
            flagged.append(ix)
    return flagged


def semantic_self_dedupe(candidates: Sequence[Candidate], *, threshold: float = SEMANTIC_THRESHOLD):
    """semhash self-deduplication over the surviving rows -> {key: twin key}.

    WITHIN ONE QUESTION FORM, for rule 2's reason and by the same measurement.
    A generated row's text is the seed's grounding plus an instruction, and the
    wave planner builds up to PER_SEED_CAP tasks on one seed in different
    forms; measured against the installed library on the four-forms-one-seed
    fixture, an unguarded semantic self-dedupe flags 3 of the 4 AT EVERY
    THRESHOLD FROM 0.8 TO 0.99 - the grounding is 95% of the text and the
    instruction cannot move a whole-text embedding far enough to matter. So an
    unguarded layer deletes the wave planner's design the moment semhash is
    installed, which is the same fault the plan's bare `J >= 0.85 on prompts`
    had, one measure over. Cross-form duplicates are rule 3's job.
    """
    by_form: dict[str, list[int]] = {}
    for ix, candidate in enumerate(candidates):
        by_form.setdefault(candidate.item.form, []).append(ix)
    flagged: dict[str, None] = {}
    for form in sorted(by_form):
        group = by_form[form]
        if len(group) < 2:
            # Nothing to compare against, and semhash refuses an empty or
            # single-record set anyway.
            continue
        texts = [candidates[ix].item.text for ix in group]
        for hit in flagged_indexes(texts, threshold=threshold):
            flagged[candidates[group[hit]].key] = None
    return flagged


# A NEAR-DUPLICATE pair - reworded, not byte-identical - plus a stranger. Two
# identical strings are collapsed by a seam with no semantic power at all, and
# rule 1 has already removed every byte-identical pair before this layer sees
# the corpus, so a control on identical text certifies the seam on a case it
# can never meet. This pair is one clause reordering apart: measured, its
# 5-gram row Jaccard is 0.714, so RULE 3 DOES NOT CATCH IT - it is exactly the
# kind of duplicate this layer is added for. Measured against the installed
# library it collapses to [1] at every threshold from 0.8 to 0.97, and two
# genuinely different rows of this corpus collapse at none of them.
SEMANTIC_CONTROL_TEXTS = (
    f"{SEMANTIC_CONTROL_ITEM} and his appeal against that conviction was dismissed by the "
    f"high court",
    "his appeal against that conviction was dismissed by the high court after the appellant "
    "was convicted under section 302 of the penal code and sentenced to imprisonment for "
    "life by the court of sessions",
    SEMANTIC_CONTROL_NEGATIVE,
)


def semantic_control(*, threshold: float = SEMANTIC_THRESHOLD) -> None:
    """Raise unless the seam is OBSERVED collapsing a near-duplicate and only it.

    `semantic: "ran"` has to mean "this layer ran and worked", not "the import
    succeeded". The failure that made this necessary was silent in the
    permissive direction - a drifted result shape flagged NOTHING and the
    manifest and the dataset card both said `ran` over an unscreened corpus.

    TWO CHECKS, NOT ONE, because they fail for different reasons and a single
    `flagged != [1]` let the count stand in for the identity: a seam that flags
    exactly one WRONG record - keeps both members of the duplicate pair and
    drops the stranger - answers a count test correctly while being exactly
    backwards.
    """
    flagged = flagged_indexes(list(SEMANTIC_CONTROL_TEXTS), threshold=threshold)
    if len(flagged) != 1:
        raise SemanticSeamError(
            f"semhash self-deduplication flagged {flagged} of a control of one reworded "
            f"duplicate pair and one unrelated record; exactly one record must go. A seam "
            f"that flags nothing here would record `semantic: ran` over rows it never "
            f"compared, and one that flags everything would delete the corpus."
        )
    if flagged != [1]:
        raise SemanticSeamError(
            f"semhash self-deduplication flagged record {flagged[0]} of the control rather "
            f"than record 1: it kept both members of the near-duplicate pair and dropped "
            f"something else. The count is right and the answer is backwards, which is the "
            f"reading a count-only check cannot tell from a working seam."
        )


def dedupe_items(
    items: Iterable[Item],
    *,
    prompt_threshold: float = PROMPT_JACCARD,
    row_threshold: float = ROW_JACCARD,
    cap: int | None = CNR_CAP,
    semantic=None,
) -> tuple[list[Item], list[dict], dict]:
    candidates = [candidate_of(item) for item in items]
    kept, drops, stats = dedupe_candidates(
        candidates, prompt_threshold=prompt_threshold, row_threshold=row_threshold,
    )
    if semantic is not None:
        # A BATCH pass over the survivors, not a per-row hook: semhash's own
        # API is batch self-deduplication, and a streaming double would have
        # been a shape the real library does not have.
        flagged = semantic(kept)
        if flagged:
            survivors = []
            for candidate in kept:
                if candidate.key in flagged:
                    drops.append(
                        _drop(candidate, REASON_SEMANTIC, flagged[candidate.key],
                              {"form": candidate.item.form})
                    )
                else:
                    survivors.append(candidate)
            kept = survivors
    if cap is None:
        # NULL, not 0. `uncapped_rows: 0, cases: 0, cases_over_cap: 0` is
        # byte-identical to "the cap ran and reached every row", which is the
        # opposite of what --no-cap did.
        stats.update(capped=None, uncapped_rows=None, cases=None, cases_over_cap=None, cap=None)
    else:
        kept, cap_drops, cap_stats = apply_cap(kept, cap=cap)
        drops += cap_drops
        stats.update(cap_stats)
        stats["cap"] = cap
    stats["kept"] = len(kept)
    stats["dropped"] = len(drops)
    stats["by_reason"] = {
        reason: sum(1 for d in drops if d["reason"] == reason)
        for reason in REASONS
        if any(d["reason"] == reason for d in drops)
    }
    return [c.item for c in kept], drops, stats


def read_decontamination_manifest(path: Path) -> dict | None:
    """decontaminate.py's manifest, if the input came with one.

    Carried forward rather than summarised: a waived eval set has to reach the
    dataset card, and this is the stage between it and the card.
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


CUSTODY_VERIFIED = "verified"
CUSTODY_NO_MANIFEST = "no_manifest"
CUSTODY_MISMATCH = "content_mismatch"
CUSTODY_NO_DIGEST = "no_output_digest"
CUSTODY_SEVERAL_INPUTS = "several_inputs"

CUSTODY_BANNERS = {
    CUSTODY_NO_MANIFEST: (
        "NO DECONTAMINATION MANIFEST beside this input - these rows may never have been"
        " screened against the eval sets, and the manifest records that as"
        " decontamination: null"
    ),
    CUSTODY_MISMATCH: (
        "THE DECONTAMINATION MANIFEST BESIDE THIS INPUT DESCRIBES DIFFERENT ROWS - its"
        " output digest does not match the bytes read here, so these rows were NOT the"
        " ones screened. The record is NOT inherited: decontamination: null"
    ),
    CUSTODY_NO_DIGEST: (
        "THE DECONTAMINATION MANIFEST BESIDE THIS INPUT CARRIES NO OUTPUT DIGEST (it"
        " predates decon_version 2), so nothing here can tell whether it describes these"
        " rows. Not inherited: decontamination: null. Re-run decontaminate.py"
    ),
    CUSTODY_SEVERAL_INPUTS: (
        "SEVERAL INPUTS were read, so at most one of them can be the decontamination"
        " pass's output and no manifest describes the whole. Not inherited:"
        " decontamination: null"
    ),
}


def custody_of(inputs: Sequence[Path]) -> tuple[dict | None, dict]:
    """(the upstream manifest to carry forward, the custody record).

    BOUND TO CONTENT, NOT TO A DIRECTORY. The manifest is still looked for
    beside the input - the question it answers is "were THESE rows screened" -
    but finding one is no longer enough to inherit it: the digest
    decontaminate.py recorded for its own output has to match the bytes read
    here. Without that check, four never-screened rows shipped under a
    manifest asserting a five-row decontamination pass, exit 0, with the
    counts visibly disagreeing and nothing looking at them; and every
    `decontaminate --out elsewhere/x.jsonl` left a stale pair in out/ for a
    later bare `dedupe` to adopt.

    A path that differs is NOT a failure - the same bytes under another name
    are still the screened rows - so only the digest decides. Every non-verified
    outcome is its own status, because they send the operator to different
    places, and the record travels into the manifest so `decontamination: null`
    always says WHY.
    """
    from tuned.data.acquire import sha256_file

    paths = [Path(p) for p in inputs]
    record = {
        "status": CUSTODY_NO_MANIFEST,
        "manifest": None,
        "input": str(paths[0]) if paths else None,
        "input_sha256": None,
        "manifest_output": None,
    }
    if len(paths) != 1:
        record["status"] = CUSTODY_SEVERAL_INPUTS
        record["input"] = [str(p) for p in paths]
        return None, record
    manifest_path = paths[0].parent / DECON_MANIFEST_FILENAME
    upstream = read_decontamination_manifest(manifest_path)
    if upstream is None:
        return None, record
    record["manifest"] = str(manifest_path)
    output = upstream.get("output") or {}
    record["manifest_output"] = {k: output.get(k) for k in ("path", "rows", "sha256")}
    if not output.get("sha256"):
        record["status"] = CUSTODY_NO_DIGEST
        return None, record
    record["input_sha256"] = sha256_file(paths[0])
    if record["input_sha256"] != output["sha256"]:
        record["status"] = CUSTODY_MISMATCH
        return None, record
    record["status"] = CUSTODY_VERIFIED
    return upstream, record


def ids_from_text_for(upstream: dict | None, *, override: bool = False) -> bool:
    """Whether a row's case identifiers may come from the citations it names.

    `--no-case-id-from-text` is the lever that answers the case-identifier
    level's false-positive risk (one landmark citation matching an eval item
    can cost hundreds of rows). It was passed to decontaminate.py and dropped
    here - stream_items always asked for identifiers from text - so the screen
    and the per-case cap disagreed about which case a row belonged to, and the
    lever only half worked.

    The upstream manifest is the answer when it is verified, because a row's
    identity must not depend on which command last looked at it; the flag
    overrides it, and False is the default the flag turns off.
    """
    if override:
        return False
    recorded = (upstream or {}).get("thresholds", {}).get("case_ids_from_text")
    return True if recorded is None else bool(recorded)


def manifest_of(stats: dict, *, inputs: Sequence[str], upstream: dict | None,
                semantic: str, thresholds: dict, semantic_detail: str = "",
                custody: dict | None = None, outputs: Sequence[dict] = ()) -> dict:
    from tuned.data.store import utcnow

    upstream_summary = None
    if upstream is not None:
        upstream_summary = {
            "at": upstream.get("at"),
            "decon_version": upstream.get("decon_version"),
            "counts": upstream.get("counts"),
            "eval_sets": {
                key: {
                    "status": value.get("status"),
                    "allowed_missing": value.get("allowed_missing"),
                    "items": value.get("items"),
                }
                for key, value in (upstream.get("eval_sets") or {}).items()
            },
            "semantic": upstream.get("semantic"),
        }
    return {
        "stage": "dedupe",
        "dedupe_version": DEDUPE_VERSION,
        "at": utcnow(),
        "inputs": list(inputs),
        # What this pass WROTE, by content. decontaminate.py has recorded this
        # since decon_version 2 and it is what custody_of above verifies its
        # own input against; dedupe's manifest carried none, so the chain of
        # custody it asserts upstream could not be extended downstream at all.
        # A LIST rather than a single record because split.py writes two files
        # and stats.py reads two, and one shape for the whole tail is one
        # reader (`manifest_digests`) rather than three.
        "outputs": list(outputs),
        "thresholds": thresholds,
        "counts": {
            key: stats[key]
            for key in ("total", "kept", "dropped", "uncapped_rows", "cases", "cases_over_cap")
        },
        "by_reason": dict(sorted(stats["by_reason"].items())),
        # "ran" means the seam was OBSERVED collapsing a control duplicate, not
        # that semhash imported. See semantic_control.
        "semantic": semantic,
        "semantic_detail": semantic_detail,
        # The chain of custody. `null` here means this input was NOT the
        # decontamination pass's output, which is the one thing a reader of
        # the dataset card must be able to tell - and `decontamination_check`
        # says WHICH way it failed, so a null is never mistaken for a missing
        # field and a mismatch is never mistaken for a missing manifest.
        "decontamination": upstream_summary,
        "decontamination_check": custody,
    }


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from tuned.data.config import load_build_config
    from tuned.data.jsonl import write_jsonl
    from tuned.data.paths import build_paths
    from tuned.data.store import Store

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="data/configs/data_law_v1.yaml")
    parser.add_argument("--in", dest="inputs", action="append", default=None,
                        help=f"default out/{DECON_OUT_FILENAME} (decontaminate.py's output - "
                             f"running this on un-decontaminated rows loses clean twins)")
    parser.add_argument("--out", default=None, help=f"default out/{OUT_FILENAME}")
    parser.add_argument("--no-cap", action="store_true",
                        help=f"skip the {CNR_CAP}-rows-per-case cap")
    parser.add_argument("--no-case-id-from-text", action="store_true",
                        help="take a row's case identifiers from _prov only, not from the "
                             "citations its grounding names. DEFAULTS TO WHAT "
                             "decontaminate.py RECORDED for the same rows - the cap and the "
                             "screen must not disagree about what case a row is about")
    args = parser.parse_args(argv)

    cfg = load_build_config(args.config)
    paths = build_paths(cfg.build.workdir).ensure()
    default_in = paths.out_dir / DECON_OUT_FILENAME
    inputs = [Path(p) for p in args.inputs] if args.inputs else [default_in]
    out_path = Path(args.out) if args.out else paths.out_dir / OUT_FILENAME

    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        print(
            f"no such input: {', '.join(missing)}\n"
            f"  run: python -m tuned.data.decontaminate --config {args.config}\n"
            f"  (decontamination runs FIRST: dedupe on un-screened rows can keep a "
            f"contaminated representative and drop its clean twin, losing both)"
        )
        return 2

    # Beside the INPUT and MATCHING ITS BYTES: the question the manifest
    # answers is "were THESE rows screened", and a manifest found next to an
    # --in from somewhere else describes different rows.
    upstream, custody = custody_of(inputs)
    # INHERITED FROM THE SCREEN, overridable but never silently disagreeing:
    # `--no-case-id-from-text` is the approved remedy for the case-identifier
    # level's false-positive risk, and a lever that moves one of the two
    # passes is not a remedy. When custody could not be verified there is
    # nothing to inherit and the default stands.
    ids_from_text = ids_from_text_for(upstream, override=args.no_case_id_from_text)
    items = list(stream_items(inputs, ids_from_text=ids_from_text))
    semantic, semantic_status, semantic_detail = None, SEMANTIC_UNAVAILABLE, ""
    if semhash_available():
        try:
            semantic_control()
        except SemanticModelError as exc:
            # Its own rung, for decontaminate.py's reason: an index that could
            # not be BUILT is a machine without the model, and the remedy is
            # network or a warm cache rather than a hunt for API drift.
            semantic_status, semantic_detail = SEMANTIC_NO_MODEL, str(exc)
        except Exception as exc:
            # Installed, called, and wrong. NOT "ran": the whole point of that
            # word in the manifest is that it distinguishes a layer that
            # compared rows from a layer that was merely invoked. Broad for
            # decontaminate.py's reason - semhash may raise anything at all
            # from inside a model it loaded lazily, and nothing in this module
            # refuses a run.
            semantic_status = SEMANTIC_UNUSABLE
            semantic_detail = (
                str(exc) if isinstance(exc, SemanticSeamError)
                else f"{type(exc).__name__}: {exc}"
            )
        else:
            semantic, semantic_status = semantic_self_dedupe, SEMANTIC_RAN
    kept, drops, stats = dedupe_items(
        items, cap=None if args.no_cap else CNR_CAP, semantic=semantic
    )
    # THE ROWS BEFORE THE MANIFEST: the manifest records a digest of what was
    # written, so it cannot be built before the bytes exist. A manifest written
    # first would have to name a file it had not seen.
    written = write_jsonl(out_path, [item.row for item in kept])
    write_jsonl(out_path.parent / DROPS_FILENAME, drops)
    manifest = manifest_of(
        stats,
        inputs=[str(p) for p in inputs],
        outputs=[output_record(out_path, written)],
        upstream=upstream,
        semantic=semantic_status,
        semantic_detail=semantic_detail,
        custody=custody,
        thresholds={
            "ngram": NGRAM,
            "prompt_jaccard": PROMPT_JACCARD,
            "row_jaccard": ROW_JACCARD,
            "cap": None if args.no_cap else CNR_CAP,
            "case_ids_from_text": ids_from_text,
            "semantic": SEMANTIC_THRESHOLD,
        },
    )

    write_manifest(out_path.parent / MANIFEST_FILENAME, manifest)
    store = Store.open(paths.state_db)
    try:
        store.log_event("dedupe", manifest)
    finally:
        store.close()

    print(f"read {stats['total']} rows from {', '.join(str(p) for p in inputs)}")
    for reason, count in sorted(stats["by_reason"].items()):
        print(f"    drop[{reason}]: {count}")
    if stats["cap"] is None:
        print("  THE PER-CASE CAP DID NOT RUN (--no-cap): one judgment may hold any number of "
              "rows, and the manifest records the cap counts as null rather than as zero")
    else:
        print(
            f"  {stats['cases']} cases carried an identifier, {stats['cases_over_cap']} over the"
            f" cap; {stats['uncapped_rows']} rows carry NO case identifier and were never capped"
        )
    if semantic_status != SEMANTIC_RAN:
        print(f"  semantic self-dedupe did NOT run ({semantic_status})")
        if semantic_detail:
            print(f"    {semantic_detail}")
    if upstream is None:
        # The order is a correctness property, so an input that did not come
        # through decontamination is stated, not assumed away - and WHICH way
        # the custody check failed is stated too, because "there is no
        # manifest here" and "the manifest here is about other rows" send the
        # operator to different places.
        print(f"  {CUSTODY_BANNERS[custody['status']]}")
    else:
        # The timestamp is printed because a verified digest says these rows
        # ARE the screened rows, not that they were screened recently: a
        # refusal leaves the previous success's output and manifest in place,
        # self-consistent, and this is what makes that visible.
        print(
            f"  decontamination verified: {upstream.get('at')} "
            f"(decon_version {upstream.get('decon_version')}, "
            f"{(upstream.get('counts') or {}).get('kept')} rows screened)"
        )
        waived = sorted(
            key for key, value in (upstream.get("eval_sets") or {}).items()
            if value.get("allowed_missing")
        )
        if waived:
            print(f"  carried forward: eval sets WAIVED upstream: {', '.join(waived)}")
    print(f"wrote {written} rows -> {out_path}")
    print(f"      {len(drops)} drops -> {out_path.parent / DROPS_FILENAME}")
    print(f"      manifest -> {out_path.parent / MANIFEST_FILENAME}")
    if stats["total"] and not written:
        print("  EVERYTHING WAS DROPPED from a non-empty input - that is a rule fault, "
              "not a corpus.")
        return 1
    if not stats["total"]:
        print("  NOTHING READ: the input is empty. Exiting 0 here would report an empty "
              "dataset as deduplicated.")
        return 1
    return 0


if __name__ == "__main__":
    import os
    import sys

    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
