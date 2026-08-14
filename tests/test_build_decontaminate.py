import json
import math
import os
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from pipeline_fakes import open_store, paths_for, temp_config

from tuned.data.decontaminate import (
    CONTAINMENT,
    DECON_VERSION,
    EVAL_COUNTS_VERIFIED_AT,
    EVAL_EMPTY,
    EVAL_MIN_SHARE,
    EVAL_NO_FILES,
    EVAL_NO_READER,
    EVAL_NO_TEXT_COLUMN,
    EVAL_NOT_ACQUIRED,
    EVAL_OK,
    EVAL_SETS,
    EVAL_TOO_FEW,
    EVAL_UNMATCHABLE,
    EVAL_UNREADABLE,
    LEVEL_CASE_ID,
    LEVEL_NARROW,
    LEVEL_SHORT,
    LEVEL_TEXT,
    NGRAM,
    SEMANTIC_NO_ITEMS,
    SEMANTIC_NO_MODEL,
    SEMANTIC_PROBE_STRIDE,
    SEMANTIC_PROBE_WORDS,
    SEMANTIC_THRESHOLD,
    SEMANTIC_UNAVAILABLE,
    SEMANTIC_UNUSABLE,
    SHORT_MIN_TOKENS,
    TITLE_MIN_TOKENS,
    EvalIndex,
    EvalItem,
    EvalPart,
    SemanticSeamError,
    containment,
    decontaminate_items,
    eval_corpora,
    eval_corpus,
    gram_hashes,
    hits_for,
    identifiers_from_fields,
    identifiers_from_text,
    item_of,
    jaccard,
    level_for,
    manifest_of,
    probe_texts,
    refusals,
    row_form,
    selected_records,
    store_items,
    title_key,
    tokens,
    window_for,
)
from tuned.data.decontaminate import main as decon_main
from tuned.data.jsonl import write_jsonl
from tuned.data.store import Store

# --------------------------------------------------------------------------
# Fixture material. Legal-sounding prose from a fixed vocabulary, so a text's
# length and its overlap with another are both under the test's control.
# --------------------------------------------------------------------------

_VOCAB = (
    "appellant respondent conviction sentence tribunal petitioner acquittal remand statute "
    "evidence witness testimony jurisdiction limitation compensation negligence contract "
    "arbitration injunction possession partition succession maintenance custody bail charge "
    "framing revision appeal writ mandamus certiorari quashing prosecution accused magistrate "
    "sessions chargesheet investigation deposition cross examination affidavit pleadings decree"
).split()


def prose(seed: int, n: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(_VOCAB) + str(rng.randrange(10_000)) for _ in range(n))


def row(prompt: str, answer: str = "answer text here", **prov) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "_prov": {"source": "test", "license": "Apache-2.0", **prov},
    }


def items(*rows) -> list:
    return [item_of(r, f"fixture#{i}") for i, r in enumerate(rows)]


def index_of(*texts, set_key="bbl", identifiers=()) -> EvalIndex:
    return EvalIndex(
        EvalItem(set_key, f"{set_key}#{i}", text, frozenset(identifiers))
        for i, text in enumerate(texts)
    )


def eval_snapshot(store, root: Path, key: str, records, *, name="data/test-0.jsonl"):
    """One eval set on disk and in the artifact index, acquire.py's shape."""
    spec = EVAL_SETS[key]
    store.upsert_source(spec.source_id, spec.license, url=spec.url)
    path = Path(root) / key / name
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, records)
    store.record_artifact(
        spec.source_id, name, local_path=path, size_bytes=path.stat().st_size, sha256="0" * 64
    )
    return path


def eval_filler(spec, have: int) -> list[dict]:
    """Enough extra questions to clear this set's documented-row floor.

    Production refuses a set holding under EVAL_MIN_SHARE of the rows it is
    documented to hold (244 for BBL), because a set that small is a fragment
    or a different dataset. Fixtures are three rows long, so they say so here
    rather than each test carrying 244 lines - and the floor itself is tested
    on both sides in test_a_set_far_smaller_than_it_is_documented_to_be_is_a_
    refusal, which is where a mutation of it dies.
    """
    floor = math.ceil((spec.expect_rows or 0) * EVAL_MIN_SHARE)
    return [{"question": prose(700_000 + i, 20)} for i in range(max(0, floor - have))]


def all_eval_snapshots(store, root: Path, records_by_key=None):
    records_by_key = records_by_key or {}
    for key, spec in EVAL_SETS.items():
        records = records_by_key.get(key, [{"question": prose(900 + ord(key[0]), 40)}])
        eval_snapshot(store, root, key, list(records) + eval_filler(spec, len(records)))


# --------------------------------------------------------------------------
# A fake `semhash`. The real one is in the [build] extra and is NOT installed,
# so both seams are written against its documented API and neither has ever
# executed - which is exactly why the SHAPE of its answer is faked here in
# every direction it could drift, including the one that used to read as a
# clean corpus.
# --------------------------------------------------------------------------

def _near(text: str, others, threshold: float) -> bool:
    """Stand-in for semantic similarity: token-set Jaccard.

    Not a claim about how semhash decides - a way to give the fake a decision
    that responds to its input, so a seam that ignores its input is
    distinguishable from one that reads it.
    """
    a = set(tokens(text))
    for other in others:
        b = set(tokens(other))
        if a and b and len(a & b) / len(a | b) >= threshold:
            return True
    return False


@pytest.fixture(autouse=True)
def _the_semantic_layer_is_opt_in(monkeypatch):
    """No test here runs the REAL semhash unless it says so.

    The [build] extra may or may not be installed on any given machine, and a
    suite whose drop counts depend on that is not a suite - with semhash
    present the seam runs inside every CLI test, fetches a model, and flags
    rows the exact stack did not. `sys.modules[name] = None` is the documented
    way to make `import semhash` raise ImportError, which is precisely the
    state semhash_available() exists to report; install_fake_semhash overrides
    this entry for the tests that pin the seam itself.
    """
    monkeypatch.setitem(sys.modules, "semhash", None)


def install_fake_semhash(monkeypatch, *, attr="selected", answer="real"):
    """Put a `semhash` module in sys.modules.

    `attr` is what the result object names its survivors - the documented name
    is `selected`, and a build that renames it is the drift both seams have to
    fail loudly on rather than default around. `answer` is "real" (decide by
    similarity), "keep-everything" (nothing is ever a duplicate),
    "keep-nothing" (everything is), "flag-stranger" (self-dedupe keeps a
    duplicate pair and drops the unrelated record instead - the right COUNT
    and the wrong record), "exact-only" (a seam with NO semantic power at all:
    it collapses byte-identical records and nothing else) or "no-model"
    (construction raises the way an installed-but-air-gapped semhash does).

    `deduplicate` compares the query records against the INDEX ONLY and never
    against each other. That is not a simplification, it is what the installed
    library does - verified by passing it two byte-identical query records and
    getting both back - and it matters here because a row's probe windows
    overlap: a fake that also deduplicated within the query would flag every
    long row whatever the eval side held.
    """
    import sys
    import types

    class Result:
        def __init__(self, kept):
            setattr(self, attr, list(kept))

    class SemHash:
        def __init__(self, records):
            if answer == "no-model":
                raise OSError(
                    "We couldn't connect to 'https://huggingface.co' to load the model"
                )
            self.records = list(records)

        @classmethod
        def from_records(cls, records):
            return cls(records)

        def deduplicate(self, records, threshold=0.9):
            """Query records that are duplicates OF THE INDEX are removed."""
            if answer == "keep-everything":
                return Result(list(records))
            if answer == "keep-nothing":
                return Result([])
            if answer == "exact-only":
                return Result([r for r in records if r not in self.records])
            return Result(
                [r for r in records if not _near(r, self.records, threshold)]
            )

        def self_deduplicate(self, threshold=0.9):
            if answer == "keep-everything":
                return Result(list(self.records))
            if answer == "keep-nothing":
                return Result([])
            if answer == "exact-only":
                kept = []
                for record in self.records:
                    if record not in kept:
                        kept.append(record)
                return Result(kept)
            if answer == "flag-stranger":
                # The right count and the wrong record: both members of the
                # duplicate pair survive and the unrelated one is dropped.
                return Result(list(self.records[:-1]))
            kept, seen = [], []
            for record in self.records:
                if not _near(record, seen, threshold):
                    kept.append(record)
                    seen.append(record)
            return Result(kept)

    module = types.ModuleType("semhash")
    module.SemHash = SemHash
    monkeypatch.setitem(sys.modules, "semhash", module)
    return module


def shuffled(text: str, seed: int) -> str:
    """The same words in a different order - token-set identical, so the fake
    semantic layer calls it a duplicate while the n-gram levels cannot."""
    words = text.split()
    random.Random(seed).shuffle(words)
    return " ".join(words)


# --------------------------------------------------------------------------
# The primitives.
# --------------------------------------------------------------------------

def test_tokens_drop_typesetting_and_keep_the_numbers_that_distinguish_questions():
    assert tokens("Section 302, I.P.C. -- **held**: NO.") == (
        "section", "302", "i", "p", "c", "held", "no",
    )
    # Same words, different punctuation and case: one token stream, so a
    # re-typeset eval question still matches.
    assert tokens("Section 302 IPC held no") == tokens("section 302; (IPC) HELD -- No!")


def test_gram_hashes_walk_the_same_windows_a_naive_implementation_would():
    import zlib

    from tuned.data.decontaminate import _BASE, _MASK

    toks = tokens(prose(3, 40))
    naive = set()
    hs = [zlib.crc32(t.encode()) for t in toks]
    for i in range(len(hs) - NGRAM + 1):
        window = 0
        for value in hs[i : i + NGRAM]:
            window = (window * _BASE + value) & _MASK
        naive.add(window)
    assert gram_hashes(toks, NGRAM) == naive
    assert len(naive) == len(toks) - NGRAM + 1


def test_the_window_boundary_is_exactly_the_ngram_constant():
    exact = tuple(f"w{i}" for i in range(NGRAM))
    assert len(gram_hashes(exact, NGRAM)) == 1
    assert gram_hashes(exact[:-1], NGRAM) == frozenset()


def test_gram_hashes_do_not_move_with_the_process_hash_seed(tmp_path):
    """The determinism hazard at its source: an index keyed on Python's own
    str hash selects a different candidate set - and so a different dataset -
    on every run."""
    script = (
        "from tuned.data.decontaminate import gram_hashes, tokens;"
        "print(sorted(gram_hashes(tokens('the appellant was convicted under section 302 of "
        "the penal code and sentenced to life'), 5))[:4])"
    )
    outs = []
    for seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.append(
            subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True
            ).stdout
        )
    assert outs[0] == outs[1] == outs[2]
    assert outs[0].strip() != "[]"


def test_containment_is_asymmetric_and_jaccard_is_not():
    small, large = frozenset({1, 2}), frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
    assert containment(small, large) == 1.0
    assert containment(large, small) == 0.2
    assert jaccard(small, large) == jaccard(large, small) == 0.2


# --------------------------------------------------------------------------
# The premise the brief hands over, measured.
# --------------------------------------------------------------------------

def test_the_jaccard_rule_the_brief_asks_for_cannot_see_a_verbatim_leak():
    """13-gram Jaccard >= 0.8 is blind to the leak this module exists for.

    An eval question sitting VERBATIM inside a long training row scores
    ~0.005 Jaccard - three orders of magnitude under 0.8 - because Jaccard
    divides by the union, which the row's own length dominates. Containment
    reads 1.0 on the same pair. This is the measurement the module's rule is
    built on, so it is asserted rather than described.
    """
    question = "which of the following best describes the doctrine of " + prose(1, 31)
    # Long enough to be screened at the full window, so this test is about
    # Jaccard against containment and not about window_for.
    assert level_for(len(tokens(question))) == LEVEL_TEXT
    long_row = prose(2, 1200) + " " + question + " " + prose(3, 1200)
    q_grams = gram_hashes(tokens(question), NGRAM)
    r_grams = gram_hashes(tokens(long_row), NGRAM)

    assert containment(q_grams, r_grams) == 1.0
    assert jaccard(q_grams, r_grams) < 0.02
    # ... and the rule the plan specifies would therefore have kept it.
    assert jaccard(q_grams, r_grams) < 0.8

    kept, drops, _ = decontaminate_items(items(row(long_row)), index_of(question))
    assert not kept
    assert drops[0]["reason"] == f"{LEVEL_TEXT}:bbl"
    assert drops[0]["hits"][0]["containment"] == 1.0
    assert drops[0]["hits"][0]["jaccard"] < 0.8


def test_a_jaccard_rule_is_a_branch_with_no_case_of_its_own():
    """J <= C always, so {J >= 0.8} is a strict subset of {C >= 0.5}: a second
    Jaccard branch could never fire alone, which is why there is not one."""
    rng = random.Random(5)
    subsumed = 0
    for _ in range(400):
        a = frozenset(rng.sample(range(400), rng.randint(10, 60)))
        # Half the pairs are near-copies of `a`, so the J >= 0.8 region the
        # claim is about is actually populated; the rest are unrelated.
        if rng.random() < 0.5:
            keep = rng.sample(sorted(a), max(1, int(len(a) * rng.uniform(0.75, 1.0))))
            b = frozenset(keep) | frozenset(rng.sample(range(400, 800), rng.randint(0, 4)))
        else:
            b = frozenset(rng.sample(range(400), rng.randint(10, 60)))
        assert jaccard(a, b) <= containment(a, b) + 1e-12
        if jaccard(a, b) >= 0.8:
            assert containment(a, b) >= CONTAINMENT
            subsumed += 1
    # The premise of the claim: the sample really did contain such pairs.
    assert subsumed > 20


# --------------------------------------------------------------------------
# The candidate step contributes zero false negatives.
# --------------------------------------------------------------------------

def test_the_gram_index_finds_every_pair_brute_force_finds():
    """The exactness property, asserted against brute force rather than
    argued in a comment. Any row with containment > 0 shares a gram, so the
    index proposes it; and the shared COUNT the walk produces is the true
    intersection size, which is what the arithmetic then divides."""
    rng = random.Random(11)
    eval_texts = [prose(rng.randrange(10_000), rng.randint(20, 60)) for _ in range(40)]
    index = index_of(*eval_texts)
    # The premise the multi-window index has to survive: these items do NOT
    # all sit at one window, so a walk that read a single table would miss
    # whole bands of the corpus.
    assert len(set(index.windows)) > 1
    checked = 0
    for _ in range(40):
        base = rng.choice(eval_texts)
        overlap = " ".join(base.split()[: rng.randint(0, 40)])
        row_text = prose(rng.randrange(10_000), 30) + " " + overlap
        query = index.query(tokens(row_text))
        proposed = index.candidates(query)
        for ix, text in enumerate(eval_texts):
            window = index.windows[ix]
            true_shared = len(gram_hashes(tokens(text), window) & query[window])
            if true_shared:
                assert ix in proposed, "candidate step lost a pair with a real overlap"
                assert proposed[ix] == true_shared
                checked += 1
    assert checked > 20, "the fixture never produced an overlapping pair to check"


# --------------------------------------------------------------------------
# The text levels: containment at the full window and at the narrow one,
# both sides of the threshold. (The module numbers four levels and these
# comments used to number themselves separately, which drifted apart.)
# --------------------------------------------------------------------------

def _row_sharing(question_words, share: int, filler: int = 60):
    """A row carrying `share` of an eval question's words, plus filler."""
    return " ".join(question_words[:share]) + " " + prose(77, filler)


def test_the_containment_threshold_decides_in_both_directions():
    """The module's most important constant, pinned to +/-0.02 in BOTH
    directions.

    The old fixture sat at 0.3981 and 0.7500 while its comment claimed to
    straddle closely: 0.55, 0.60, 0.65 and 0.70 all survived, and LOOSENING is
    the dangerous direction - it is the one that lets a leak through.
    """
    words = prose(21, 120).split()
    question = " ".join(words)
    q_grams = gram_hashes(tokens(question), NGRAM)

    heavy = _row_sharing(words, 68)   # 56 of 108 grams
    light = _row_sharing(words, 64)   # 52 of 108 grams
    exact = _row_sharing(words, 66)   # 54 of 108 grams - the constant itself
    heavy_share = containment(q_grams, gram_hashes(tokens(heavy), NGRAM))
    light_share = containment(q_grams, gram_hashes(tokens(light), NGRAM))
    exact_share = containment(q_grams, gram_hashes(tokens(exact), NGRAM))
    assert light_share < CONTAINMENT < heavy_share
    assert 0.46 < light_share and heavy_share < 0.54
    assert exact_share == CONTAINMENT == 0.5

    index = index_of(question)
    assert decontaminate_items(items(row(heavy)), index)[0] == []
    assert len(decontaminate_items(items(row(light)), index)[0]) == 1
    # The boundary is INCLUSIVE, and it is reachable: 54/108 lands on it
    # exactly, so `>= threshold` is not interchangeable with `> threshold`.
    assert decontaminate_items(items(row(exact)), index)[0] == []


def _edited(words, k, seed=7):
    """`words` with `k` substitutions spread evenly through the sequence."""
    rng = random.Random(seed)
    out = list(words)
    step = len(words) / (k + 1)
    for i in range(k):
        out[int(step * (i + 1))] = "xreplacedx" + str(rng.randrange(10_000))
    return out


def _containment_after_edits(length: int, edits: int, window: int, seed: int = 0) -> float:
    words = prose(length * 3 + 1 + seed, length).split()
    left = gram_hashes(tokens(" ".join(words)), window)
    right = gram_hashes(tokens(" ".join(_edited(words, edits))), window)
    return containment(left, right)


def test_one_substituted_token_still_scores_above_the_threshold_at_every_length():
    """The property the length-aware window exists to deliver, measured over
    the whole range rather than asserted at one point.

    Below 38 tokens a FIXED 13-gram window puts a one-word edit at zero - it
    is an exact-match rule wearing a near-match label - and BBL is 24,365 MCQ
    questions, most of them shorter than that.
    """
    for length in range(NGRAM, 220):
        window = window_for(length)
        score = _containment_after_edits(length, 1, window)
        assert score >= CONTAINMENT, (length, window, score)

    # ... and the fixed window it replaces does not, which is the measurement
    # that forced this. These are the numbers in the module docstring.
    fixed = {length: _containment_after_edits(length, 1, NGRAM)
             for length in (13, 20, 25, 29, 35, 37, 38, 50)}
    assert [round(fixed[k], 3) for k in (13, 20, 25)] == [0.0, 0.0, 0.0]
    assert 0.2 < fixed[29] < 0.3 and 0.4 < fixed[35] < 0.45
    # The crossover: at 38 tokens the two windows ARE the same window, so a
    # rule that started the full window one token earlier or later would be
    # claiming an edit tolerance it does not have.
    assert fixed[37] < CONTAINMENT <= fixed[38]
    assert window_for(38) == NGRAM and window_for(37) < NGRAM


def test_the_narrow_level_carries_a_case_neither_of_the_others_can():
    """A 20-token MCQ question with ONE word changed, leaked into a long row.

    Invisible to the full window (containment 0.000, so `text` cannot see it),
    and not a whole-sequence match (so `short` cannot either) - and its own
    length is what puts it out of both.
    """
    words = prose(401, 20).split()
    question = " ".join(words)
    leaked_form = " ".join(_edited(words, 1))
    assert level_for(len(tokens(question))) == LEVEL_NARROW
    assert containment(
        gram_hashes(tokens(question), NGRAM), gram_hashes(tokens(leaked_form), NGRAM)
    ) == 0.0

    leaked = prose(402, 300) + " " + leaked_form + " " + prose(403, 300)
    kept, drops, _ = decontaminate_items(items(row(leaked)), index_of(question))
    assert not kept
    assert drops[0]["reason"] == f"{LEVEL_NARROW}:bbl"
    assert drops[0]["hits"][0]["window"] == 7
    assert drops[0]["hits"][0]["eval_tokens"] == 20


def test_the_narrow_band_tolerates_exactly_one_edit_and_no_more():
    """Its control, and the limitation stated at the strength it actually has.

    The window is chosen so that G = L - w + 1 lands on 2w, which is what makes
    one EVENLY SPREAD edit score exactly CONTAINMENT and two of them score
    zero. That is the worst case, and this docstring used to generalise it into
    a cliff - "two edits destroy 2w >= G grams and containment goes to ZERO,
    with no band in between". Swept over every placement instead, the module is
    MORE sensitive than that: 63 of the 190 two-edit placements at L = 20 still
    drop, and 140 of the 1,140 three-edit ones. The survivors are edits near an
    END of the item, where a token is covered by fewer windows.

    Above 38 tokens the window stops shrinking and the slope is gentle
    everywhere: a 100-token item still scores 0.85 after one edit and 0.26
    after five.
    """
    import itertools

    words = prose(401, 20).split()
    question = " ".join(words)
    window = window_for(20)
    scores = [
        containment(gram_hashes(tokens(question), window),
                    gram_hashes(tokens(" ".join(_edited(words, k))), window))
        for k in (1, 2, 3)
    ]
    assert scores == [CONTAINMENT, 0.0, 0.0], "evenly spread: the worst case"

    # ... and the whole placement sweep, which is what the prose now claims.
    item_grams = gram_hashes(tokens(question), window)

    def edited_at(positions):
        out = list(words)
        for p in positions:
            out[p] = f"xreplacedx{p}"
        return containment(item_grams, gram_hashes(tokens(" ".join(out)), window))

    swept = {
        k: [edited_at(c) for c in itertools.combinations(range(20), k)] for k in (1, 2, 3)
    }
    dropped = {k: sum(1 for s in v if s >= CONTAINMENT) for k, v in swept.items()}
    assert dropped == {1: 20, 2: 63, 3: 140}
    assert (round(min(swept[2]), 3), round(max(swept[2]), 3)) == (0.0, 0.857)
    # The survivors are the ones near an end: two edits at the item's head
    # destroy far fewer grams than two in the middle.
    assert edited_at((0, 1)) > CONTAINMENT > edited_at((9, 10))

    leaked = prose(404, 300) + " " + " ".join(_edited(words, 2)) + " " + prose(405, 300)
    kept, drops, _ = decontaminate_items(items(row(leaked)), index_of(question))
    assert len(kept) == 1 and not drops

    long_words = prose(406, 100).split()
    long_question = " ".join(long_words)
    slope = [
        round(containment(gram_hashes(tokens(long_question), NGRAM),
                          gram_hashes(tokens(" ".join(_edited(long_words, k))), NGRAM)), 2)
        for k in (1, 5)
    ]
    assert slope == [0.85, 0.26], "above the cap the rule degrades gradually again"


def test_the_cost_table_in_the_docstring_is_what_the_rule_actually_charges():
    """The weakest evidence the rule accepts, at each length, measured on both
    windows - because the docstring's table is what a reader deciding whether
    to move the divisor works from, and it was quoting a closed form that is
    wrong by up to a token in both directions."""
    def shortest_run(length, window):
        item = tokens(prose(430, length))[:length]
        return next(
            (r for r in range(1, length + 1)
             if containment(gram_hashes(item, window), gram_hashes(item[:r], window))
             >= CONTAINMENT),
            None,
        )

    lengths = (13, 16, 20, 25, 30, 38, 50)
    assert [shortest_run(L, NGRAM) for L in lengths] == [13, 14, 16, 19, 21, 25, 31]
    assert [shortest_run(L, window_for(L)) for L in lengths] == [8, 10, 13, 16, 20, 25, 31]
    # The band edges, and the sharpest discontinuity in the design: 12 -> 13,
    # where an item stops being matched WHOLE and starts needing 62% of itself.
    assert shortest_run(12, window_for(12)) == 12
    assert shortest_run(13, window_for(13)) == 8
    assert all(shortest_run(L, window_for(L)) >= 11 for L in range(17, 25))


def test_the_narrow_bands_false_positive_class_is_question_boilerplate():
    """Named from measurement, because the docstring named statutory quotation
    and that is not what collides down here. An MCQ stem is mostly stem: two of
    these three drop against rows that merely share ordinary legal phrasing,
    where the fixed 13-gram window scores them 0.000."""
    stems = [
        "under which section of the indian penal code is criminal breach of trust punishable",
        "which of the following is not an essential ingredient of the offence of theft",
        "what is the limitation period prescribed for filing an appeal against a decree",
    ]
    rows = [
        "the question before this court is under which section of the indian penal code "
        "criminal breach of trust by a public servant is punishable and what the sentence is",
        "counsel argued which of the following is not an essential ingredient of the offence "
        "and the court held that dishonest intention is",
        "the limitation period prescribed for filing an appeal against a decree of the civil "
        "court is thirty days from the date of the decree",
    ]
    narrow_drops, fixed_drops = 0, 0
    for stem in stems:
        item = tokens(stem)
        assert NGRAM <= len(item) <= 16, "the band this class lives in"
        for text in rows:
            row_toks = tokens(text)
            window = window_for(len(item))
            narrow_drops += (
                containment(gram_hashes(item, window), gram_hashes(row_toks, window))
                >= CONTAINMENT
            )
            fixed_drops += (
                containment(gram_hashes(item, NGRAM), gram_hashes(row_toks, NGRAM))
                >= CONTAINMENT
            )
    assert narrow_drops == 2 and fixed_drops == 0


def test_capping_the_window_at_the_constant_is_what_keeps_long_items_screenable():
    """The cap's own case. A 150-token IL-TUR-style item gramed at (L+1)//3 =
    50 would need fifty consecutive untouched tokens per gram, so three edits
    take it to zero; at 13 the same item still scores 0.72."""
    words = prose(411, 150).split()
    question = " ".join(words)
    quoted = " ".join(_edited(words, 3))
    assert window_for(150) == NGRAM
    assert containment(gram_hashes(tokens(question), 50), gram_hashes(tokens(quoted), 50)) == 0.0
    assert containment(gram_hashes(tokens(question), NGRAM),
                       gram_hashes(tokens(quoted), NGRAM)) > CONTAINMENT

    leaked = prose(412, 400) + " " + quoted + " " + prose(413, 400)
    kept, drops, _ = decontaminate_items(items(row(leaked)), index_of(question))
    assert not kept and drops[0]["reason"] == f"{LEVEL_TEXT}:bbl"


def test_a_row_that_leaks_two_lengths_of_item_is_counted_against_both_levels():
    """The levels are a union and by_level is the instrument that says whether
    each branch is carrying cases of its own on real data - the first-run
    check is literally "if `short` and `case_id` are both 0 the union
    collapsed to one rule". A single best hit across all levels reports the
    strongest one and files the row under that level alone, so the instrument
    under-counts exactly the branch it exists to watch. Found by mutation: the
    per-level structure survived the whole suite as one pooled best.
    """
    long_question = prose(431, 150)
    short_question = "what is the punishment for criminal breach of trust"
    assert level_for(len(tokens(long_question))) == LEVEL_TEXT
    assert level_for(len(tokens(short_question))) == LEVEL_SHORT

    words = long_question.split()
    leaked = " ".join(words[:100]) + " " + prose(432, 200) + " " + short_question
    index = index_of(long_question, short_question)
    _, drops, stats = decontaminate_items(items(row(leaked)), index)
    assert len(drops) == 1
    levels = [hit["level"] for hit in drops[0]["hits"]]
    assert levels == [LEVEL_TEXT, LEVEL_SHORT]
    assert stats["by_level"][LEVEL_TEXT] == stats["by_level"][LEVEL_SHORT] == 1
    # The short item scores higher, so a pooled best would have reported IT
    # and filed the row under `short`.
    containments = {h["level"]: h["containment"] for h in drops[0]["hits"]}
    assert containments[LEVEL_SHORT] > containments[LEVEL_TEXT] >= CONTAINMENT
    assert drops[0]["reason"] == f"{LEVEL_TEXT}:bbl"


def test_a_short_question_that_shares_a_statutory_phrase_has_a_boundary_too():
    """The statute exception, one length band down - the false-positive price
    the narrow window is paid for with, pinned on BOTH sides.

    At 20 tokens the rule tolerates a shared run of 12 tokens and refuses 13,
    i.e. it asks for two thirds of the question rather than the four fifths a
    fixed 13-gram window asks for. That gap is the whole cost of the change.
    """
    words = prose(421, 20).split()
    question = " ".join(words)
    index = index_of(question)
    for shared, expect_drop in ((12, False), (13, True)):
        quoted = " ".join(words[:shared])
        row_text = prose(422, 200) + " " + quoted + " " + prose(423, 200)
        kept, drops, _ = decontaminate_items(items(row(row_text)), index)
        assert bool(drops) is expect_drop, (shared, drops)


def test_a_statute_quotation_both_sides_quote_is_not_contamination():
    """The spec's standing exception: statutes are the domain being taught, so
    a shared provision must not empty the corpus. This is the reason the rule
    is containment >= 0.5 and not 'one shared 13-gram'."""
    statute = (
        "whoever commits theft shall be punished with imprisonment of either description for a "
        "term which may extend to three years or with fine or with both"
    )
    question = statute + " " + prose(31, 120)
    row_text = statute + " " + prose(32, 400)
    q_grams = gram_hashes(tokens(question), NGRAM)
    r_grams = gram_hashes(tokens(row_text), NGRAM)
    # The premise: they DO share grams, so this is the case a one-gram rule
    # would drop.
    assert len(q_grams & r_grams) >= 10
    assert containment(q_grams, r_grams) < CONTAINMENT
    # The number itself, because the length-aware window must not move it: at
    # 146 tokens this item is screened at the FULL window, exactly as before.
    assert window_for(len(tokens(question))) == NGRAM
    assert round(containment(q_grams, r_grams), 4) == 0.1045

    kept, drops, _ = decontaminate_items(items(row(row_text)), index_of(question))
    assert len(kept) == 1 and not drops


# --------------------------------------------------------------------------
# The short-item rule, and the case only it can carry.
# --------------------------------------------------------------------------

def test_a_short_eval_question_is_invisible_to_the_ngram_level_and_the_short_rule_carries_it():
    question = "what is the punishment for criminal breach of trust"
    assert len(tokens(question)) < NGRAM
    # The premise, stated as a fact about the fixture: the full window has
    # NOTHING to match on here, so a drop can only come from the short rule.
    assert gram_hashes(tokens(question), NGRAM) == frozenset()

    leaked = prose(41, 300) + " " + question + " " + prose(42, 300)
    index = index_of(question)
    assert index.windows == [len(tokens(question))]
    assert NGRAM not in index.by_gram

    kept, drops, _ = decontaminate_items(items(row(leaked)), index)
    assert not kept
    assert drops[0]["reason"] == f"{LEVEL_SHORT}:bbl"
    assert drops[0]["hits"][0]["eval_tokens"] == len(tokens(question))


def test_a_row_that_does_not_carry_the_short_question_survives_it():
    question = "what is the punishment for criminal breach of trust"
    other = prose(43, 300) + " what is the punishment for wrongful restraint " + prose(44, 300)
    kept, drops, _ = decontaminate_items(items(row(other)), index_of(question))
    assert len(kept) == 1 and not drops


def test_an_eval_item_at_exactly_the_floor_is_still_matched():
    """The floor is inclusive, and it is a real boundary: one token either
    side of it decides between a screened question and a hole in the screen."""
    question = "criminal breach of trust punishment"
    assert len(tokens(question)) == SHORT_MIN_TOKENS == 5
    index = index_of(question)
    assert index.unmatchable == []

    leaked = prose(45, 200) + " " + question + " " + prose(46, 200)
    kept, drops, _ = decontaminate_items(items(row(leaked)), index)
    assert not kept and drops[0]["reason"] == f"{LEVEL_SHORT}:bbl"


def test_the_window_length_decides_which_level_carries_an_item():
    """A behavioural pin on 13, from both sides: a 13-token question is
    n-grammable (at a narrowed window) and a 12-token one is not, so the
    constant decides which rule an eval item is screened by - and the fixture
    states the two lengths as literals rather than reading them off the
    constant under test."""
    thirteen = "which of the following is the correct measure of damages in this case"
    twelve = "which of the following is the correct measure of damages in case"
    assert (len(tokens(thirteen)), len(tokens(twelve))) == (13, 12)

    for question, expected in ((thirteen, LEVEL_NARROW), (twelve, LEVEL_SHORT)):
        leaked = prose(47, 150) + " " + question + " " + prose(48, 150)
        _, drops, _ = decontaminate_items(items(row(leaked)), index_of(question))
        assert drops[0]["reason"] == f"{expected}:bbl", question


def test_short_items_of_different_lengths_are_each_screened():
    """The short index is keyed BY LENGTH, so it is a set of rules and not
    one: a walk that stopped at the first length would screen against the
    5-token questions and silently skip the 9-token ones."""
    five = "criminal breach of trust punishment"
    nine = "what is the punishment for criminal breach of trust"
    index = index_of(five, nine)
    assert sorted(index.by_gram) == [5, 9], "the premise: two different short lengths"

    for question in (five, nine):
        leaked = prose(49, 150) + " " + question + " " + prose(50, 150)
        kept, drops, _ = decontaminate_items(items(row(leaked)), index)
        assert not kept, question
        assert drops[0]["reason"] == f"{LEVEL_SHORT}:bbl"


def test_the_identifiers_that_cost_the_most_rows_are_listed_first():
    """The yield-price instrument: one landmark citation matching an eval item
    can drop hundreds of rows, and this list is how that shows up. Listed
    least-first it would point the operator at the cheapest identifier."""
    index = EvalIndex([
        EvalItem("iltur", "iltur#0", prose(52, 40), frozenset({"cit:2020 INSC 484"})),
        EvalItem("iltur", "iltur#1", prose(53, 40), frozenset({"cit:2019 INSC 12"})),
    ])
    rows = [row(prose(54 + i, 30), citation="2020 INSC 484") for i in range(3)]
    rows.append(row(prose(60, 30), citation="2019 INSC 12"))
    _, _, stats = decontaminate_items(items(*rows), index)
    manifest = manifest_of(stats, {}, index, inputs=[], semantic="x")
    assert manifest["top_identifiers"] == [("cit:2020 INSC 484", 3), ("cit:2019 INSC 12", 1)]


def test_one_row_is_one_case_identifier_hit_however_many_pairs_matched():
    """The `break` had no case. A row sharing two identifiers with three eval
    items each is ONE contaminated row, and a Hit per (identifier, item) pair
    inflates `by_level["case_id"]` - which first-run check #4 reads to decide
    whether this branch carries cases of its own - by the size of the citation
    graph rather than by the number of rows."""
    index = EvalIndex([
        EvalItem("iltur", f"iltur#{i}", prose(560 + i, 40),
                 frozenset({"cit:2020 INSC 484", "cnr:DLHC010001232020"}))
        for i in range(3)
    ])
    leaky = row(prose(563, 30), citation="2020 INSC 484", cnr="DLHC010001232020")
    _, drops, stats = decontaminate_items(items(leaky), index)
    assert stats["by_level"][LEVEL_CASE_ID] == 1
    assert [h["level"] for h in drops[0]["hits"]] == [LEVEL_CASE_ID]
    assert stats["dropped"] == 1


def test_a_drop_blames_every_identifier_that_matched_and_not_just_the_first():
    """`identifier_drops` fed `top_identifiers`, which is first-run check #3
    and the instrument the --no-case-id-from-text decision is made on. It
    blamed the SORTED-FIRST matching identifier only - and "cit:" sorts before
    "cnr:", so a row matched on both its own case number and a landmark
    citation always named the citation and never the case. That lever removes
    a whole channel, so the operator has to see every identifier that would
    have cost the row, redundant ones included."""
    index = EvalIndex([
        EvalItem("iltur", "iltur#0", prose(570, 40),
                 frozenset({"cit:2020 INSC 484", "cnr:DLHC010001232020"})),
    ])
    both = row(prose(571, 30), citation="2020 INSC 484", cnr="DLHC010001232020")
    only_citation = row(prose(572, 30), citation="2020 INSC 484")
    _, drops, stats = decontaminate_items(items(both, only_citation), index)
    assert stats["identifier_drops"] == {"cit:2020 INSC 484": 2, "cnr:DLHC010001232020": 1}
    # The drop record carries both, and still files the row under one reason.
    assert drops[0]["hits"][0]["identifiers"] == ["cit:2020 INSC 484", "cnr:DLHC010001232020"]
    assert drops[0]["reason"] == f"{LEVEL_CASE_ID}:iltur"
    # It is a count of ROW-MATCHES, so it sums to at least the drop count.
    manifest = manifest_of(stats, {}, index, inputs=[], semantic="x")
    assert sum(count for _, count in manifest["top_identifiers"]) >= stats["dropped"]


def test_an_eval_item_under_the_floor_is_counted_not_silently_ignored():
    """A 3-token question would match half the corpus, so nothing here can use
    it - and that hole is COUNTED, because an unusable eval item that nobody
    counts is a screen reporting clean on a question it never checked."""
    tiny = "define estoppel"
    assert len(tokens(tiny)) < SHORT_MIN_TOKENS
    index = index_of(tiny)
    assert index.unmatchable == ["bbl#0"]

    row_text = prose(51, 100) + " define estoppel " + prose(52, 100)
    kept, _, stats = decontaminate_items(items(row(row_text)), index)
    assert len(kept) == 1
    manifest = manifest_of(stats, {}, index, inputs=[], semantic="x")
    assert manifest["unmatchable_eval_items"] == {"bbl": 1}
    assert manifest["unmatchable_eval_items_total"] == 1


# --------------------------------------------------------------------------
# The case-identifier level - the one no n-gram method can replace.
# --------------------------------------------------------------------------

def test_same_judgment_different_question_is_caught_only_by_the_case_identifier_level():
    """The union's third branch, with the case only it can carry: two texts
    about ONE judgment that share no wording at all."""
    eval_text = "in ESCR010004512020 what relief did the court finally grant " + prose(61, 60)
    row_text = "summarise the reasoning of the bench in " + prose(62, 400)
    index = EvalIndex([EvalItem("iltur", "iltur#0", eval_text, frozenset({"cnr:ESCR010004512020"}))])

    # Premise: the text levels have nothing to say about this pair, at ANY
    # window the index holds.
    assert index.candidates(index.query(tokens(row_text))) == {}

    item = item_of(row(row_text, cnr="ESCR01-000451-2020"), "fixture#0")
    assert "cnr:ESCR010004512020" in item.identifiers
    found = hits_for(item, index)
    assert [h.level for h in found] == [LEVEL_CASE_ID]
    assert found[0].detail["identifier"] == "cnr:ESCR010004512020"


def test_each_identifier_namespace_carries_a_case_the_others_cannot():
    cnr = identifiers_from_fields({"cnr": "DLHC010001232020"})
    citation = identifiers_from_fields({"neutral_citation": "2020 INSC 484"})
    title = identifiers_from_fields(
        {"case_title": "Government of India & Ors versus ISRO Drivers Association"}
    )
    assert cnr == {"cnr:DLHC010001232020"}
    assert citation == {"cit:2020 INSC 484"}
    assert title == {"title:government of india v isro drivers association"}
    # Three disjoint namespaces: a row carrying only one of them is reachable
    # only through that one.
    assert not (cnr & citation) and not (citation & title) and not (cnr & title)


def test_a_citation_is_normalised_before_it_is_compared():
    """'2020 INSC 0484' and '2020 INSC 484' are one judgment - the citation
    index's normalisation is what makes the join work, and it is reused here
    rather than re-implemented."""
    left = identifiers_from_fields({"citation": "2020 INSC 0484"})
    right = identifiers_from_text("as held in 2020 INSC 484 the appeal was allowed")
    assert left & right == {"cit:2020 INSC 484"}


def test_a_title_too_generic_to_join_on_is_not_an_identifier():
    """The floor pinned on both sides, and measured with the function the RULE
    reads - title_key counts words of landmark_key(...), not tokens(...), and
    a premise stated in the wrong units can be true of a fixture the rule sees
    differently."""
    from tuned.data.select import landmark_key

    three = "State v Kumar"
    four = "Kesavananda Bharati v Kerala"
    assert len(landmark_key(three).split()) == TITLE_MIN_TOKENS - 1 == 3
    assert len(landmark_key(four).split()) == TITLE_MIN_TOKENS == 4
    assert title_key(three) is None
    assert title_key(four) is not None


def test_an_identifier_in_the_answer_is_not_what_the_row_is_about():
    """A citation the model cited is an authority, not the case the row is
    built on - taking identifiers from the answer would drop every row that
    mentions a landmark."""
    item = item_of(row("a question about " + prose(71, 40), answer="see 2020 INSC 484"), "x#0")
    assert not any(i.startswith("cit:") for i in item.identifiers)
    from_prompt = item_of(row("consider 2020 INSC 484 and " + prose(71, 40)), "x#1")
    assert "cit:2020 INSC 484" in from_prompt.identifiers


def test_identifiers_from_text_can_be_switched_off_leaving_the_prov_channel():
    text = "consider 2020 INSC 484 in the light of " + prose(72, 40)
    with_text = item_of(row(text, cnr="DLHC010001232020"), "x#0", ids_from_text=True)
    without = item_of(row(text, cnr="DLHC010001232020"), "x#0", ids_from_text=False)
    assert "cit:2020 INSC 484" in with_text.identifiers
    assert without.identifiers == frozenset({"cnr:DLHC010001232020"})


def test_the_drop_reason_names_the_level_and_the_eval_set_that_caught_it():
    question = prose(81, 120)
    index = index_of(question, set_key="aibe")
    kept, drops, stats = decontaminate_items(items(row(question)), index)
    assert not kept
    assert drops[0]["reason"] == f"{LEVEL_TEXT}:aibe"
    assert stats["by_eval_set"] == {"aibe": 1}
    assert stats["by_reason"] == {f"{LEVEL_TEXT}:aibe": 1}


# --------------------------------------------------------------------------
# The instruments that must not read healthy when they are blind.
# --------------------------------------------------------------------------

def test_the_case_identifier_level_reports_when_it_did_not_run():
    """Task 11's lesson applied here: a level with nothing to match on must
    not be indistinguishable from a level that found nothing."""
    index = index_of(prose(91, 40))  # eval side carries no identifiers
    kept, _, stats = decontaminate_items(items(row(prose(92, 40))), index)
    assert len(kept) == 1
    manifest = manifest_of(stats, {}, index, inputs=[], semantic="x")
    assert manifest["case_identifier_level_inert"] is True
    assert manifest["case_identifier_coverage"] == 0.0

    # ... and when both sides do carry identifiers, it is not inert.
    live_index = EvalIndex([EvalItem("bbl", "bbl#0", prose(93, 40), frozenset({"cit:X"}))])
    _, _, live = decontaminate_items(items(row(prose(94, 40), citation="2020 INSC 484")), live_index)
    live_manifest = manifest_of(live, {}, live_index, inputs=[], semantic="x")
    assert live_manifest["case_identifier_level_inert"] is False
    assert live_manifest["case_identifier_coverage"] == 1.0


def test_a_row_with_no_text_is_counted_rather_than_passing_the_screen():
    empty = {"messages": [{"role": "user", "content": ""}], "_prov": {}}
    punctuation = {"messages": [{"role": "user", "content": "--- *** ... ,,, "}], "_prov": {}}
    kept, _, stats = decontaminate_items(items(empty, punctuation), index_of(prose(95, 40)))
    assert len(kept) == 2  # they are not dropped ...
    assert stats["empty_text"] == 2  # ... but they are not silently screened either


def test_a_devanagari_question_is_screened_and_not_reduced_to_its_digits():
    """BhashaBench-Legal ships Hindi. An ASCII-only tokeniser turns a Hindi
    question into a couple of digits, so a verbatim leak of it would match
    nothing and the row would pass by being unreadable - a leak this module
    would have reported as clean."""
    question = (
        "भारतीय दंड संहिता की धारा 302 के अंतर्गत हत्या के लिए निर्धारित दंड "
        "क्या है और अपवाद किन परिस्थितियों में लागू होता है"
    )
    assert len(tokens(question)) >= NGRAM, "the premise: the tokeniser reads this script"
    leaked = prose(96, 200) + " " + question + " " + prose(97, 200)
    kept, drops, stats = decontaminate_items(items(row(leaked)), index_of(question))
    assert not kept and drops[0]["hits"][0]["containment"] == 1.0
    assert stats["empty_text"] == 0


# A 16-word Hindi question. Every claim below is measured on it.
_HINDI_QUESTION = (
    "भारतीय दंड संहिता की धारा तीन सौ दो के अंतर्गत हत्या के लिए दंड क्या है"
)
# The English equivalent, same number of words, as the control: whatever the
# tokeniser does to Hindi it must already be doing to this.
_ENGLISH_EQUIVALENT = (
    "under which section of the indian penal code is the punishment for murder "
    "prescribed and what is it"
)


def test_a_hindi_word_is_one_token_and_not_one_token_per_matra():
    """Devanagari vowel signs and the virama are categories Mn/Mc, which `\\w`
    does not match, so `[^\\W_]+` split every Hindi word at every matra:
    `भारतीय` came out as 3 tokens and `दंड` as 2. Nothing about that is
    cosmetic - the token count decides which LEVEL screens an item."""
    assert tokens("भारतीय दंड संहिता की धारा") == (
        "भारतीय", "दंड", "संहिता", "की", "धारा",
    )
    assert len(tokens(_HINDI_QUESTION)) == len(_HINDI_QUESTION.split()) == 16
    # The English side is untouched by the change.
    assert tokens("Section 302, I.P.C. -- **held**: NO.") == (
        "section", "302", "i", "p", "c", "held", "no",
    )
    # A combining mark only ever joins the word BEFORE it; it cannot start one.
    assert tokens("ा भारतीय") == ("भारतीय",)


def test_one_edited_hindi_word_still_drops_at_every_position():
    """The one-edit guarantee is denominated in TOKENS with zero margin, so an
    inflated Hindi token count spent it two or three times over on a single
    edited word. Measured under the old class: 3 of these 16 positions
    survived (a leak reported clean); on the English equivalent, 0 of 18."""
    words = _HINDI_QUESTION.split()
    survived = []
    for i in range(len(words)):
        edited = [*words[:i], "बदलाव", *words[i + 1 :]]
        leaked = prose(400, 150) + " " + " ".join(edited) + " " + prose(401, 150)
        kept, _, _ = decontaminate_items(items(row(leaked)), index_of(_HINDI_QUESTION))
        if kept:
            survived.append(i)
    assert survived == [], f"one edited Hindi word survived at word positions {survived}"

    english = _ENGLISH_EQUIVALENT.split()
    survived_en = []
    for i in range(len(english)):
        edited = [*english[:i], "changed", *english[i + 1 :]]
        leaked = prose(402, 150) + " " + " ".join(edited) + " " + prose(403, 150)
        kept, _, _ = decontaminate_items(items(row(leaked)), index_of(_ENGLISH_EQUIVALENT))
        if kept:
            survived_en.append(i)
    assert survived_en == []


def test_a_two_word_hindi_phrase_is_as_unmatchable_as_its_english_twin():
    """`अपील खारिज` read as FIVE tokens under the old class, cleared the
    5-token floor and became matchable - so a stock phrase two words long
    could drop rows, while English "appeal dismissed" (2 tokens) correctly
    could not. The floor has to mean the same thing in both scripts."""
    assert window_for(len(tokens("अपील खारिज"))) == 0
    assert window_for(len(tokens("appeal dismissed"))) == 0
    index = index_of("अपील खारिज", "appeal dismissed")
    assert len(index.unmatchable) == 2
    kept, _, _ = decontaminate_items(
        items(row("अपील खारिज की गई " + prose(404, 60) + " appeal dismissed")), index
    )
    assert len(kept) == 1


def test_the_length_histogram_reads_the_same_band_for_a_question_in_either_script():
    """The instrument has to move with the rule: an inflated Hindi token count
    filed a 16-word question under `text` (29 tokens) while its English twin
    read `narrow`, so the table the window calibration is decided from was
    ~2x wrong on 7,318 of BBL's 24,365 questions."""
    index = EvalIndex(
        [
            EvalItem("bbl", "hi#0", _HINDI_QUESTION, frozenset()),
            EvalItem("bbl", "en#0", _ENGLISH_EQUIVALENT, frozenset()),
        ]
    )
    bands = index.length_report()["bbl"]
    assert bands[LEVEL_NARROW] == 2 and bands[LEVEL_TEXT] == 0
    assert bands["min_tokens"] == 16 and bands["max_tokens"] == 18


def test_the_cli_says_when_the_generations_were_not_screened(tmp_path, capsys):
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    write_jsonl(paths.streams_dir / "replay.jsonl", [row(prose(98, 40))])
    store = open_store(tmp_path, n_seeds=1, db_path=paths.state_db)
    all_eval_snapshots(store, tmp_path / "hf")
    _accept_generation(store, answer="an accepted answer")
    store.close()

    assert decon_main(["--config", cfg, "--no-generated"]) == 0
    out = capsys.readouterr().out
    manifest = json.loads((paths.out_dir / "decontamination.json").read_text(encoding="utf-8"))
    assert manifest["generations_screened"] is False
    assert manifest["generations"] == {"screened": False, "state": None, "read": 0}
    assert "THE ACCEPTED GENERATIONS WERE NOT SCREENED" in out

    assert decon_main(["--config", cfg]) == 0
    out = capsys.readouterr().out
    manifest = json.loads((paths.out_dir / "decontamination.json").read_text(encoding="utf-8"))
    assert manifest["generations_screened"] is True
    assert manifest["generations"] == {"screened": True, "state": "accepted", "read": 1}
    assert "THE ACCEPTED GENERATIONS WERE NOT SCREENED" not in out


def test_asking_for_a_state_the_store_holds_none_of_is_not_a_screened_run(tmp_path, capsys):
    """`true` used to mean only that --no-generated was absent, so a run that
    read NOTHING from the store recorded its generations as screened - and a
    typo'd --state was exactly that run."""
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    write_jsonl(paths.streams_dir / "replay.jsonl", [row(prose(98, 40))])
    store = open_store(tmp_path, n_seeds=1, db_path=paths.state_db)
    all_eval_snapshots(store, tmp_path / "hf")
    _accept_generation(store, answer="an accepted answer")
    store.close()

    assert decon_main(["--config", cfg, "--state", "rejected"]) == 0
    out = capsys.readouterr().out
    manifest = json.loads((paths.out_dir / "decontamination.json").read_text(encoding="utf-8"))
    assert manifest["generations_screened"] is False
    assert manifest["generations"] == {"screened": False, "state": "rejected", "read": 0}
    assert "NO GENERATIONS IN STATE 'rejected'" in out


@pytest.mark.parametrize("state", ["", "acceptedd", "ACCEPTED"])
def test_a_state_that_is_not_a_task_state_is_refused_by_the_parser(tmp_path, state):
    """`--state ''` shipped a REJECTED generation: the store's filter was
    keyed on truthiness, so an empty string disabled it and the pass read
    EVERY state. A typo read zero rows and reported them screened. Neither is
    reachable now, in the parser and in the store."""
    cfg = temp_config(tmp_path)
    paths_for(tmp_path)
    with pytest.raises(SystemExit) as exc:
        decon_main(["--config", cfg, "--state", state])
    assert exc.value.code == 2


def test_every_real_task_state_is_still_accepted_by_the_parser(tmp_path):
    """The other direction of the same guard, which had no case: a `choices`
    list narrowed to one literal state refuses `--state rejected` and reads as
    the validation working. The parser's list has to be the store's."""
    from tuned.data.store import TASK_STATES

    cfg = temp_config(tmp_path)
    paths_for(tmp_path)
    assert len(TASK_STATES) > 1
    for state in TASK_STATES:
        # It gets past the parser and fails later on the missing eval sets,
        # which is a refusal (2) with the eval banner rather than a usage error.
        assert decon_main(["--config", cfg, "--state", state, "--no-generated"]) == 2


def test_eval_rows_that_carry_no_question_column_are_reported_even_when_the_set_loads(
    tmp_path, capsys
):
    """`ok` on a set 90% of whose questions were never read is the same
    failure as a missing set, one layer down."""
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    write_jsonl(paths.streams_dir / "replay.jsonl", [row(prose(99, 40))])
    store = Store.open(paths.state_db)
    # iltur rather than bbl: it carries no documented row count, so the two
    # numbers this test is about are the ones it writes and not a floor's
    # padding.
    all_eval_snapshots(
        store, tmp_path / "hf",
        {"iltur": [{"question": prose(100, 40)}] + [{"mystery_column": "x"}] * 9},
    )
    store.close()

    assert decon_main(["--config", cfg, "--no-generated"]) == 0
    out = capsys.readouterr().out
    assert "9 of 10 rows carried none of" in out
    manifest = json.loads((paths.out_dir / "decontamination.json").read_text(encoding="utf-8"))
    assert (manifest["eval_sets"]["iltur"]["rows"],
            manifest["eval_sets"]["iltur"]["items"]) == (10, 1)


# --------------------------------------------------------------------------
# The refusal. The one thing that must not be got wrong.
# --------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    store = open_store(tmp_path, n_seeds=0)
    yield store
    store.close()


def test_an_eval_set_that_was_never_acquired_is_a_refusal(store, tmp_path):
    corpora = eval_corpora(store)
    assert {c.status for c in corpora.values()} == {EVAL_NOT_ACQUIRED}
    blocked = refusals(corpora)
    assert len(blocked) == len(EVAL_SETS)
    # Each refusal names ITS OWN set - one message repeated with the wrong
    # repo id would send the operator to acquire a dataset they already have.
    for text, key in zip(blocked, sorted(EVAL_SETS), strict=True):
        assert EVAL_SETS[key].repo_id in text
        assert f"--allow-missing-eval {key}" in text
        assert f"acquire --kind hf --hf-source {key}" in text
        for other in sorted(set(EVAL_SETS) - {key}):
            assert EVAL_SETS[other].repo_id not in text


@pytest.mark.parametrize(
    "records,files,expected",
    [
        ([{"question": "what is the punishment for theft under the code"}], "jsonl", EVAL_OK),
        ([{"unexpected_column": "a question nobody can find"}], "jsonl", EVAL_NO_TEXT_COLUMN),
        ([], "none", EVAL_NO_FILES),
        (None, "broken", EVAL_UNREADABLE),
    ],
)
def test_each_way_an_eval_set_comes_back_short_has_its_own_status(
    store, tmp_path, records, files, expected
):
    """They send the operator to different places, so they are different
    statuses - select.py's not_acquired/no_title_column lesson."""
    # expect_rows cleared: the documented-row floor is a rung of its own and
    # is tested on both sides in its own test, so it is not what decides here.
    spec = replace(EVAL_SETS["bbl"], parts=())
    store.upsert_source(spec.source_id, spec.license)
    if files == "jsonl":
        eval_snapshot(store, tmp_path, "bbl", records)
    elif files == "none":
        # An object is indexed, but nothing this pass can read.
        path = tmp_path / "bbl" / "README.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not data", encoding="utf-8")
        store.record_artifact(
            spec.source_id, "README.md", local_path=path, size_bytes=8, sha256="0" * 64
        )
    else:
        path = tmp_path / "bbl" / "data.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json at all\n", encoding="utf-8")
        store.record_artifact(
            spec.source_id, "data.jsonl", local_path=path, size_bytes=8, sha256="0" * 64
        )
    corpus = eval_corpus(store, spec)
    assert corpus.status == expected
    assert (corpus.status == EVAL_OK) == (not refusals({"bbl": corpus}))


def test_an_eval_set_that_loads_but_can_match_nothing_is_a_refusal(store, tmp_path, capsys):
    """Reproduced before it was fixed: a BBL snapshot of one 3-token question
    loaded `ok`, screened against NOTHING, and the run exited 0 with the
    output stamped decontaminated. The refusal ladder had rungs for absent,
    no-files, no-column, unreadable and zero-rows, and none for LOADED AND
    USELESS - which defeats the one requirement the module has, because
    reaching an eval set in name only is not reaching it."""
    spec = replace(EVAL_SETS["bbl"], parts=())
    eval_snapshot(store, tmp_path, "bbl", [{"question": "define estoppel"},
                                           {"question": "what is res judicata"}])
    corpus = eval_corpus(store, spec)
    assert corpus.status == EVAL_UNMATCHABLE
    assert corpus.unmatchable == len(corpus.items) == 2
    blocked = refusals({"bbl": corpus})
    assert len(blocked) == 1
    assert "NOTHING here can match any of them" in blocked[0]
    # ... and the remedy is the column, not another download.
    assert "_EVAL_TEXT_FIELDS" in blocked[0]

    # The control: ONE matchable item among them (the same file rewritten, so
    # the only difference is that one question) and the set is usable again,
    # with the hole counted rather than refused.
    eval_snapshot(store, tmp_path, "bbl", [{"question": "define estoppel"},
                                           {"question": prose(500, 40)}])
    live = eval_corpus(store, spec)
    assert live.status == EVAL_OK
    assert (live.unmatchable, len(live.items)) == (1, 2)
    assert not refusals({"bbl": live})


def test_a_set_far_smaller_than_it_is_documented_to_be_is_a_refusal(store, tmp_path):
    """The three repo ids are admitted guesses, so 'wrong but resolvable id'
    is a live first-run failure - and a fragment of the right dataset is the
    same shape. Both sides of the floor, which is a HUNDREDTH of the
    documented count for the reason written beside EVAL_MIN_SHARE."""
    spec = EVAL_SETS["bbl"]
    floor = math.ceil(spec.expect_rows * EVAL_MIN_SHARE)
    assert (spec.expect_rows, floor) == (24_365, 244)

    eval_snapshot(store, tmp_path, "bbl", [{"question": prose(510 + i, 40)}
                                           for i in range(floor - 1)])
    short = eval_corpus(store, spec)
    assert short.status == EVAL_TOO_FEW
    assert short.rows == floor - 1
    blocked = refusals({"bbl": short})
    assert "24365" in blocked[0].replace(",", "")
    # The trap this rung sets for itself: waiving BBL to get past a number
    # nobody has checked is the failure the module exists to prevent.
    assert "Do NOT waive" in blocked[0]

    # The same file rewritten with ONE more row - exactly at the floor, which
    # is what makes this a boundary and not a gap.
    eval_snapshot(store, tmp_path, "bbl", [{"question": prose(510 + i, 40)}
                                           for i in range(floor)])
    at_the_floor = eval_corpus(store, spec)
    assert (at_the_floor.rows, at_the_floor.status) == (floor, EVAL_OK)


def test_a_set_that_is_short_of_its_documented_count_says_so_without_refusing(store, tmp_path):
    """Between the floor and the documented count there is no refusal - the
    number has never been checked against the Hub - but there is never silence
    either."""
    spec = EVAL_SETS["bbl"]
    eval_snapshot(store, tmp_path, "bbl",
                  [{"question": prose(700 + i, 40)} for i in range(300)])
    corpus = eval_corpus(store, spec)
    assert corpus.status == EVAL_OK
    assert corpus.shortfall == 24_365 - 300
    assert not refusals({"bbl": corpus})


def test_the_shortfall_line_and_the_manifest_carry_the_verified_expectation(tmp_path, capsys):
    """First-run check #1 IS this line, so the numbers in it are load-bearing
    and had no test. It has to name the rows read, the rows expected for the
    splits selected, and the date the expectation was verified - and the
    manifest has to carry the same three so the dataset card can."""
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    write_jsonl(paths.streams_dir / "s.jsonl", [row(prose(920, 60))])
    store = Store.open(paths.state_db)
    all_eval_snapshots(store, tmp_path / "hf")
    store.close()
    assert decon_main(["--config", cfg, "--no-generated"]) == 0
    out = capsys.readouterr().out
    manifest = json.loads((paths.out_dir / "decontamination.json").read_text(encoding="utf-8"))

    bbl = manifest["eval_sets"]["bbl"]
    assert bbl["expect_rows"] == 24_365
    assert bbl["expect_verified_at"] == EVAL_COUNTS_VERIFIED_AT
    assert bbl["row_shortfall"] == 24_365 - bbl["rows"] > 0
    assert bbl["row_surplus"] == 0
    assert bbl["selection"]["include_splits"] == ["test"]
    assert [p["rows"] for p in bbl["expect_parts"]] == [17_047, 7_318]
    assert f"{bbl['rows']} rows, 24365 expected across 2 config/split" in out
    assert f"{bbl['row_shortfall']} SHORT" in out
    assert EVAL_COUNTS_VERIFIED_AT in out

    # ... and the set with no verified count says the instrument is off rather
    # than printing a shortfall of zero, which reads identically to "complete".
    assert manifest["eval_sets"]["iltur"]["expect_rows"] is None
    assert manifest["eval_sets"]["iltur"]["expect_verified_at"] is None
    assert "the floor and the shortfall line are OFF here" in out


def test_the_floor_counts_ROWS_and_not_the_items_they_produce(store, tmp_path):
    """The distinction the whole floor argument rests on, and it had no test:
    a BBL row with options produces TWO items, so a floor read off `len(items)`
    is satisfied by half the download. Under an item floor these 122 rows -
    exactly half of BBL's 244 - would clear it, because they carry 244 items
    between them."""
    spec = EVAL_SETS["bbl"]
    floor = math.ceil(spec.expect_rows * EVAL_MIN_SHARE)
    half = [
        {"question": prose(880 + i, 40), "options": [prose(881 + i, 20)]}
        for i in range(floor // 2)
    ]
    eval_snapshot(store, tmp_path, "bbl", half)
    corpus = eval_corpus(store, spec)
    assert corpus.rows == floor // 2 < floor
    assert len(corpus.items) == floor >= floor, "the items alone would clear the floor"
    assert corpus.status == EVAL_TOO_FEW


def test_bbls_expectation_encodes_BOTH_CONFIGS_so_one_of_them_reads_short(store, tmp_path):
    """BBL is two configs of one split - English 17,047 and Hindi 7,318,
    verified against the datasets-server. An English-only download is the live
    version of "the row count and the expectation describe different
    populations": it is a complete config and an incomplete set, and it has to
    say so rather than pass silently or be refused."""
    spec = EVAL_SETS["bbl"]
    assert [(p.config, p.split, p.rows) for p in spec.parts] == [
        ("english", "test", 17_047), ("hindi", "test", 7_318),
    ]
    assert spec.expect_rows == 24_365
    eval_snapshot(store, tmp_path, "bbl", [{"question": prose(890 + i, 40)} for i in range(300)],
                  name="english/test-00000-of-00002.jsonl")
    corpus = eval_corpus(store, spec)
    assert corpus.status == EVAL_OK  # a config that IS complete is not a refusal
    assert corpus.shortfall == 24_365 - 300
    assert corpus.surplus == 0


def test_aibes_single_train_split_is_the_complete_shape_and_reads_no_shortfall(store, tmp_path):
    """The other end of the same fix. aibe has ONE split, named `train`, and
    it IS the eval set (1,157 rows, verified) - so a filter for a `test` split
    would empty it and a whole-repo expectation would be right by accident.
    The complete download must read `ok` with a shortfall of exactly zero."""
    spec = EVAL_SETS["aibe"]
    assert spec.expect_rows == 1_157 and not spec.include_splits
    eval_snapshot(store, tmp_path, "aibe",
                  [{"question": prose(900 + i, 40)} for i in range(1_157)],
                  name="data/train-00000-of-00001.jsonl")
    corpus = eval_corpus(store, spec)
    assert (corpus.status, corpus.rows, corpus.shortfall, corpus.surplus) == (EVAL_OK, 1_157, 0, 0)
    assert corpus.selection["selected"] == ["data/train-00000-of-00001.jsonl"]


def test_iltur_is_screened_against_its_test_splits_and_the_manifest_names_which(store, tmp_path):
    """~488,000 rows across 8 heterogeneous configs, `bail` alone 353,698. At
    the measured ~220 bytes per distinct gram that whole repo is tens of GB of
    index, and a screen that OOMs on the operator's machine screens nothing -
    so the eval surface is the TEST-TYPE splits of every config. A subset that
    is not named narrows the guarantee silently, so it is named here, matched
    on the SPLIT (config names were never verified) and recorded."""
    spec = EVAL_SETS["iltur"]
    for name in ("bail/train_all-00000-of-00004.jsonl", "cjpe/fold_1-00000-of-00001.jsonl",
                 "rr/dev-00000-of-00001.jsonl"):
        eval_snapshot(store, tmp_path, "iltur", [{"question": prose(910, 40)}], name=name)
    for name in ("bail/test_specific-00000-of-00001.jsonl", "lsi/expert-00000-of-00001.jsonl"):
        eval_snapshot(store, tmp_path, "iltur", [{"question": prose(911, 40)}], name=name)
    corpus = eval_corpus(store, spec)
    assert corpus.status == EVAL_OK
    assert corpus.files == 2 and corpus.rows == 2
    assert corpus.selection["selected"] == [
        "bail/test_specific-00000-of-00001.jsonl", "lsi/expert-00000-of-00001.jsonl",
    ]
    assert {e["key"] for e in corpus.selection["excluded"]} == {
        "bail/train_all-00000-of-00004.jsonl", "cjpe/fold_1-00000-of-00001.jsonl",
        "rr/dev-00000-of-00001.jsonl",
    }
    assert {e["why"] for e in corpus.selection["excluded"]} == {"train", "fold", "dev"}
    assert not corpus.selection["fallback_all_files"]
    # No verified per-split counts for this set, so the floor and the shortfall
    # instrument are OFF rather than denominated against a number nobody read.
    assert spec.expect_rows is None and corpus.shortfall == 0


def test_a_layout_that_names_no_split_reads_everything_and_says_so(store, tmp_path):
    """Over-screening is safe; screening nothing is the failure this module
    exists to prevent. But the fallback also means the row count is being
    compared against the wrong expectation, so it is recorded and printed."""
    spec = EVAL_SETS["iltur"]
    eval_snapshot(store, tmp_path, "iltur", [{"question": prose(912, 40)}], name="rows.jsonl")
    corpus = eval_corpus(store, spec)
    assert corpus.files == 1 and corpus.selection["fallback_all_files"] is True
    assert corpus.selection["excluded"] == []


def test_an_unverified_part_turns_the_expectation_off_rather_than_guessing():
    """A None anywhere in a set's parts means the sum would be a guess, and a
    floor denominated against a guess refuses correct downloads."""
    spec = replace(EVAL_SETS["bbl"], parts=(EvalPart("english", "test", 17_047),
                                            EvalPart("hindi", "test", None)))
    assert spec.expect_rows is None
    assert replace(spec, parts=()).expect_rows is None


def test_an_eval_set_that_holds_no_rows_at_all_is_a_refusal(store, tmp_path):
    """EVAL_EMPTY, which was the one status in the ladder with no test."""
    spec = replace(EVAL_SETS["bbl"], parts=())
    eval_snapshot(store, tmp_path, "bbl", [])
    corpus = eval_corpus(store, spec)
    assert corpus.status == EVAL_EMPTY
    assert corpus.files == 1 and corpus.rows == 0
    assert len(refusals({"bbl": corpus})) == 1


def test_a_missing_reader_is_not_a_corrupt_file_and_not_a_wrong_repo_id(store, tmp_path):
    """The parquet case, i.e. what EVERY operator hits on the first real run
    on all three sets at once: HF snapshots are usually parquet and pyarrow is
    in the [build] extra. It refused correctly before, but told the operator
    to re-download and to doubt the repo id - which is the one genuinely
    uncertain thing here, so the misdirection pointed straight at it."""
    spec = replace(EVAL_SETS["bbl"], parts=())
    path = tmp_path / "bbl" / "data" / "test-0.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"PAR1")
    store.upsert_source(spec.source_id, spec.license, url=spec.url)
    store.record_artifact(
        spec.source_id, "data/test-0.parquet", local_path=path, size_bytes=4, sha256="0" * 64
    )

    def reader(path):
        raise ImportError("No module named 'pyarrow'")

    corpus = eval_corpus(store, spec, reader=reader)
    assert corpus.status == EVAL_NO_READER
    assert corpus.files == 1, "the premise: the .parquet file WAS selected to be read"
    blocked = refusals({"bbl": corpus})
    assert "pip install -e .[build]" in blocked[0]
    assert "re-run" not in blocked[0] and "repo id is wrong" not in blocked[0]

    # ... and a file that is genuinely corrupt still says so, with the other
    # remedy, so this is a fork in the ladder and not a rename of it.
    def corrupt(path):
        raise ValueError("Expecting value: line 1 column 1")

    assert eval_corpus(store, spec, reader=corrupt).status == EVAL_UNREADABLE
    assert "corrupt" in refusals({"bbl": eval_corpus(store, spec, reader=corrupt)})[0]


def test_the_cli_refuses_an_unmatchable_set_and_writes_nothing(tmp_path, capsys):
    """End to end, because the reproduction was end to end: TINY-BBL exit 0,
    bbl status ok, unmatchable 1."""
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    write_jsonl(paths.streams_dir / "replay.jsonl", [row(prose(520, 40))])
    store = Store.open(paths.state_db)
    all_eval_snapshots(store, tmp_path / "hf")
    # Replace bbl entirely with questions nothing can match, at a row count
    # that clears the documented-row floor so THIS rung is what fires.
    store.conn.execute("DELETE FROM artifact WHERE source_id = ?", (EVAL_SETS["bbl"].source_id,))
    eval_snapshot(store, tmp_path / "hf", "bbl",
                  [{"question": "define estoppel"} for _ in range(300)])
    store.close()

    assert decon_main(["--config", cfg, "--no-generated"]) == 2
    out = capsys.readouterr().out
    assert "REFUSING TO DECONTAMINATE" in out
    assert EVAL_UNMATCHABLE in out
    assert not (paths.out_dir / "decontaminated.jsonl").exists()
    assert not (paths.out_dir / "decontamination.json").exists()


def test_a_question_is_screened_separately_from_its_options(store, tmp_path):
    """Containment divides by the EVAL ITEM's length, so appending the options
    to the question makes a verbatim leak of the question score lower than a
    leak of nothing much. The two are separate items for that reason."""
    question = "which of the following best describes the doctrine of " + prose(200, 20)
    options = ["a) " + prose(201, 12), "b) " + prose(202, 12), "c) " + prose(203, 12)]
    eval_snapshot(store, tmp_path, "bbl", [{"question": question, "options": options}])
    corpus = eval_corpus(store, EVAL_SETS["bbl"])
    assert [i.item_id for i in corpus.items] == ["data/test-0.jsonl#0", "data/test-0.jsonl#0/1"]

    leaked = prose(204, 400) + " " + question + " " + prose(205, 400)
    index = EvalIndex(corpus.items)
    kept, drops, _ = decontaminate_items(items(row(leaked)), index)
    assert not kept
    assert drops[0]["hits"][0]["containment"] == 1.0

    # The measurement that earns the split: concatenated, the SAME leak scores
    # under the threshold and this row would have shipped.
    concatenated = gram_hashes(tokens(question + "\n" + "\n".join(options)), NGRAM)
    assert containment(concatenated, gram_hashes(tokens(leaked), NGRAM)) < CONTAINMENT


def test_an_answer_key_too_short_to_screen_does_not_become_an_item(store, tmp_path):
    eval_snapshot(store, tmp_path, "bbl", [{"question": prose(206, 40), "answer": "b"}])
    corpus = eval_corpus(store, EVAL_SETS["bbl"])
    assert len(corpus.items) == 1
    assert EvalIndex(corpus.items).unmatchable == []


def test_read_rows_dispatches_on_the_suffix(tmp_path):
    """The snapshot layouts this has to survive. jsonl/json/csv/tsv are read
    in pure Python and exercised here; .parquet - what HF snapshots usually
    ship - is behind a lazy pyarrow import which DOES now execute where the
    [build] extra is installed (round-tripped in the parquet test below), and
    where it is not, the ImportError is EVAL_NO_READER: its own rung, whose
    remedy is `pip install -e .[build]` rather than EVAL_UNREADABLE's "the file
    is corrupt, re-download it". Either way a refusal, never an empty set."""
    from tuned.data.decontaminate import _READABLE_SUFFIXES, read_rows

    (tmp_path / "a.jsonl").write_text('{"question": "one"}\n\n{"question": "two"}\n',
                                      encoding="utf-8")
    (tmp_path / "b.json").write_text('{"data": [{"question": "three"}]}', encoding="utf-8")
    (tmp_path / "c.json").write_text('[{"question": "four"}]', encoding="utf-8")
    (tmp_path / "d.csv").write_text("question,answer\nfive,x\n", encoding="utf-8")
    (tmp_path / "e.tsv").write_text("question\tanswer\nsix\tx\n", encoding="utf-8")
    (tmp_path / "f.md").write_text("not data", encoding="utf-8")

    seen = [
        r["question"]
        for name in ("a.jsonl", "b.json", "c.json", "d.csv", "e.tsv", "f.md")
        for r in read_rows(tmp_path / name)
    ]
    assert seen == ["one", "two", "three", "four", "five", "six"]

    # The parquet branch, EXECUTED rather than named, either way round.
    # (`assert ".parquet" in _READABLE_SUFFIXES` said nothing: the tuple two
    # lines above it is the only thing that could have made it false.)
    #
    # Without the [build] extra the branch raises ImportError, which
    # eval_corpus turns into EVAL_NO_READER - a refusal with the right
    # remedy, never an empty set. With it, the branch actually reads, and this
    # is the only place in the suite that proves it: HF snapshots are usually
    # parquet, so it is what the first real run depends on.
    assert ".parquet" in _READABLE_SUFFIXES
    try:
        import pyarrow
        import pyarrow.parquet as pq
    except ImportError:
        (tmp_path / "g.parquet").write_bytes(b"PAR1")
        with pytest.raises(ImportError):
            list(read_rows(tmp_path / "g.parquet"))
    else:
        pq.write_table(
            pyarrow.table({"question": ["seven", "eight"], "answer": ["a", "b"]}),
            tmp_path / "g.parquet",
        )
        rows = list(read_rows(tmp_path / "g.parquet"))
        assert [r["question"] for r in rows] == ["seven", "eight"]
        assert rows[0]["answer"] == "a", "rows come back as dicts, which eval_item_texts reads"


def test_the_split_filter_matches_a_name_component_and_not_a_substring(store, tmp_path):
    """`latest.parquet` contains "test". A bare substring filter selects it,
    drops the real split, and screens against the wrong file - which
    over-screens nothing and under-screens everything."""
    spec = replace(EVAL_SETS["bbl"], parts=())
    eval_snapshot(store, tmp_path, "bbl", [{"question": "the real split " + prose(6, 30)}],
                  name="data/test-00000-of-00001.jsonl")
    eval_snapshot(store, tmp_path, "bbl", [{"question": "a rolling export " + prose(7, 30)}],
                  name="data/latest.jsonl")
    corpus = eval_corpus(store, spec)
    assert corpus.files == 1
    assert corpus.items[0].text.startswith("the real split")


def test_the_split_filter_prefers_the_named_split_but_never_empties_the_set(store, tmp_path):
    spec = replace(EVAL_SETS["bbl"], parts=())
    eval_snapshot(store, tmp_path, "bbl", [{"question": "train only " + prose(1, 30)}],
                  name="data/train-0.jsonl")
    eval_snapshot(store, tmp_path, "bbl", [{"question": "test only " + prose(2, 30)}],
                  name="data/test-0.jsonl")
    corpus = eval_corpus(store, spec)
    assert corpus.files == 1 and len(corpus.items) == 1
    assert corpus.items[0].text.startswith("test only")

    # A layout that does not name the split at all keeps EVERYTHING, which
    # over-screens (safe) rather than screening nothing (not safe).
    other = open_store(tmp_path / "other", n_seeds=0, db_path=tmp_path / "other.sqlite3")
    eval_snapshot(other, tmp_path / "o", "bbl", [{"question": prose(3, 30)}], name="rows.jsonl")
    assert eval_corpus(other, spec).files == 1
    other.close()


def test_the_cli_refuses_and_writes_nothing_when_an_eval_set_is_missing(tmp_path, capsys):
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    write_jsonl(paths.streams_dir / "replay.jsonl", [row(prose(101, 40))])
    store = Store.open(paths.state_db)
    all_eval_snapshots(store, tmp_path / "hf")
    # ... except one.
    store.conn.execute("DELETE FROM artifact WHERE source_id = ?", (EVAL_SETS["aibe"].source_id,))
    store.close()

    assert decon_main(["--config", cfg]) == 2
    out = capsys.readouterr().out
    assert "REFUSING TO DECONTAMINATE" in out
    assert "opennyaiorg/aibe_dataset" in out
    assert not (paths.out_dir / "decontaminated.jsonl").exists()
    assert not (paths.out_dir / "decontamination.json").exists()


def test_the_override_is_per_set_and_lands_in_the_manifest(tmp_path, capsys):
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    write_jsonl(paths.streams_dir / "replay.jsonl", [row(prose(102, 40))])
    store = Store.open(paths.state_db)
    all_eval_snapshots(store, tmp_path / "hf")
    for key in ("aibe", "iltur"):
        store.conn.execute(
            "DELETE FROM artifact WHERE source_id = ?", (EVAL_SETS[key].source_id,)
        )
    store.close()

    # Waiving ONE of the two missing sets still refuses.
    assert decon_main(["--config", cfg, "--allow-missing-eval", "aibe"]) == 2
    assert "Exploration-Lab/IL-TUR" in capsys.readouterr().out

    assert decon_main(
        ["--config", cfg, "--allow-missing-eval", "aibe", "--allow-missing-eval", "iltur"]
    ) == 0
    out = capsys.readouterr().out
    assert "WAIVED: aibe" in out and "WAIVED: iltur" in out
    manifest = json.loads((paths.out_dir / "decontamination.json").read_text(encoding="utf-8"))
    assert manifest["eval_sets"]["aibe"]["allowed_missing"] is True
    assert manifest["eval_sets"]["aibe"]["status"] == EVAL_NOT_ACQUIRED
    assert manifest["eval_sets"]["bbl"]["allowed_missing"] is False
    assert manifest["eval_sets"]["bbl"]["status"] == EVAL_OK
    # The waiver has to survive to the dataset card, which reads the store.
    event = json.loads(
        Store.open(paths.state_db).events("decontamination")[0]["detail_json"]
    )
    assert event["eval_sets"]["iltur"]["allowed_missing"] is True


# --------------------------------------------------------------------------
# Reading the candidate rows.
# --------------------------------------------------------------------------

def _accept_generation(store, *, seed_id="seed000", task_id="t1", think="", answer="A",
                       task_type="irac_analysis", scores=None):
    store.create_tasks([{
        "task_id": task_id, "seed_id": seed_id, "stream": "synthesis",
        "task_type": task_type, "prompt_id": f"{task_type}_v1", "prompt_sha": "abc",
        "sample_ix": 0,
    }])
    gen_id = store.record_generation({
        "task_id": task_id, "attempt": 1, "provider": "p", "model": "m",
        "raw_path": "raw.jsonl", "raw_offset": 0, "think": think, "answer": answer,
    })
    if scores:
        store.record_judgement(gen_id, "a", {
            "grounding": scores[0], "validity": scores[1], "coverage": scores[2]
        })
    store.set_task_state(task_id, "accepted")
    return gen_id


def test_a_generated_rows_text_is_its_grounding_and_not_the_prompt_template(tmp_path):
    """Every generated row renders from the same handful of templates, so a
    comparison over the rendered prompt would find every pair ~80% identical -
    the template, not the content."""
    store = open_store(tmp_path, n_seeds=1)
    _accept_generation(store, seed_id="seed000", think="my reasoning", answer="my answer",
                       scores=(5, 4, 3))
    loaded = list(store_items(store))
    store.close()
    assert len(loaded) == 1
    seed_text = "circumstantial evidence"
    assert seed_text in loaded[0].prompt
    assert "my answer" in loaded[0].answer and "my reasoning" in loaded[0].answer
    assert loaded[0].row["_prov"]["score"] == 4.0
    assert loaded[0].form == "irac_analysis"


def test_the_generations_reach_the_assembly_pass_in_gen_id_order(tmp_path):
    """The store's ORDER BY only decides which rows ship if the assembly pass
    preserves it - dedupe keeps the FIRST row of a cluster and the first three
    of a case, and this is the stage that fixes what "first" means."""
    store = open_store(tmp_path, n_seeds=3)
    for i in (2, 1, 0):
        _accept_generation(store, seed_id=f"seed{i:03d}", task_id=f"t{i}", answer=f"answer {i}")
    loaded = list(store_items(store))
    store.close()
    gen_ids = [item.row["_prov"]["gen_id"] for item in loaded]
    assert gen_ids == sorted(gen_ids)
    assert [item.row["_prov"]["seed_id"] for item in loaded] == ["seed002", "seed001", "seed000"]


def test_the_cli_screens_the_generations_by_default_and_says_so_when_it_does_not(tmp_path, capsys):
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    write_jsonl(paths.streams_dir / "replay.jsonl", [row(prose(103, 40))])
    store = open_store(tmp_path, n_seeds=1, db_path=paths.state_db)
    all_eval_snapshots(store, tmp_path / "hf")
    _accept_generation(store, answer="an accepted answer")
    store.close()

    assert decon_main(["--config", cfg]) == 0
    out = capsys.readouterr().out
    assert "read 1 stream rows and 1 accepted generations" in out
    assert len(list((paths.out_dir / "decontaminated.jsonl").read_text().splitlines())) == 2

    assert decon_main(["--config", cfg, "--no-generated"]) == 0
    out = capsys.readouterr().out
    assert "generations NOT screened" in out
    assert len(list((paths.out_dir / "decontaminated.jsonl").read_text().splitlines())) == 1


def test_an_accepted_generation_that_leaks_an_eval_question_is_dropped(tmp_path):
    """The reason the store side is read at all: the rows most likely to
    carry eval text are the ones the fleet generated from seeds."""
    store = open_store(tmp_path, n_seeds=1)
    question = "which of the following best describes " + prose(104, 30)
    _accept_generation(store, answer=question)
    loaded = list(store_items(store))
    store.close()
    kept, drops, _ = decontaminate_items(loaded, index_of(question))
    assert not kept
    assert drops[0]["reason"] == f"{level_for(len(tokens(question)))}:bbl"


def test_nothing_read_exits_1(tmp_path, capsys):
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    store = Store.open(paths.state_db)
    all_eval_snapshots(store, tmp_path / "hf")
    store.close()
    assert decon_main(["--config", cfg]) == 1
    assert "NOTHING READ" in capsys.readouterr().out


def test_the_cli_writes_the_rows_the_drops_and_the_manifest(tmp_path, capsys):
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    question = prose(105, 120)
    write_jsonl(
        paths.streams_dir / "curated.jsonl",
        [row(question), row(prose(106, 200))],
    )
    store = Store.open(paths.state_db)
    all_eval_snapshots(store, tmp_path / "hf", {"bbl": [{"question": question}]})
    store.close()

    assert decon_main(["--config", cfg, "--no-generated"]) == 0
    out = capsys.readouterr().out
    kept = [json.loads(line) for line in (paths.out_dir / "decontaminated.jsonl").read_text().splitlines()]
    drops = [json.loads(line) for line in (paths.out_dir / "decontamination_drops.jsonl").read_text().splitlines()]
    manifest = json.loads((paths.out_dir / "decontamination.json").read_text(encoding="utf-8"))
    assert len(kept) == 1 and len(drops) == 1
    # The row written out is the row read in, untouched.
    assert kept[0] == row(prose(106, 200))
    assert drops[0]["reason"] == f"{LEVEL_TEXT}:bbl"
    assert manifest["counts"] == {"total": 2, "kept": 1, "dropped": 1, "empty_text": 0}
    assert manifest["thresholds"]["containment"] == CONTAINMENT
    assert manifest["thresholds"]["ngram"] == NGRAM
    # The version travels with the shape: a manifest that changed what it says
    # while still claiming the old number cannot be read years later.
    assert manifest["decon_version"] == DECON_VERSION == 3
    assert "split" not in manifest["eval_sets"]["bbl"], "replaced by `selection`"
    assert "screened 2  kept 1  dropped 1" in out


# --------------------------------------------------------------------------
# Determinism.
# --------------------------------------------------------------------------

def _cli_bytes(tmp_path, seed: str, module: str, args) -> dict:
    """Run a CLI in a subprocess under a chosen PYTHONHASHSEED; return the
    bytes of everything it wrote.

    The subprocess is out of monkeypatch's reach, so the semantic layer is
    switched off the only way a separate interpreter can be told: a `semhash`
    module on the path that refuses to import. Without it these runs compare
    the bytes of a pass that ran a model, on machines that happen to have the
    [build] extra and not on others.
    """
    stub = tmp_path / "no_semhash"
    stub.mkdir(exist_ok=True)
    (stub / "semhash.py").write_text(
        'raise ImportError("semhash is switched off for this test")\n', encoding="utf-8"
    )
    env = {
        **os.environ,
        "PYTHONHASHSEED": seed,
        # PREPENDED, never replaced: the mutation sandbox puts its own src on
        # PYTHONPATH and a subprocess that loses it tests the installed copy.
        "PYTHONPATH": os.pathsep.join(
            [str(stub), *([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])]
        ),
    }
    result = subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True, text=True, env=env, cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return {
        path.name: path.read_bytes()
        for path in sorted((tmp_path / "build" / "out").glob("*"))
        if path.is_file()
    }


def test_two_runs_under_different_hash_seeds_are_byte_identical(tmp_path):
    """Run it twice for real rather than reading the source for sorted():
    survivor selection and index iteration are exactly where PYTHONHASHSEED
    leaks in, and downstream a shifting survivor set shifts split.py's
    train/test boundary."""
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    question = prose(111, 120)
    rows = [row(question)] + [row(prose(200 + i, 60), cnr=f"DLHC01000{i:03d}2020") for i in range(12)]
    write_jsonl(paths.streams_dir / "a.jsonl", rows)
    write_jsonl(paths.streams_dir / "b.jsonl", rows[::-1])
    store = Store.open(paths.state_db)
    all_eval_snapshots(store, tmp_path / "hf", {"bbl": [{"question": question}]})
    store.close()

    first = _cli_bytes(tmp_path, "0", "tuned.data.decontaminate", ["--config", cfg, "--no-generated"])
    second = _cli_bytes(tmp_path, "1", "tuned.data.decontaminate", ["--config", cfg, "--no-generated"])
    assert set(first) == {"decontaminated.jsonl", "decontamination_drops.jsonl", "decontamination.json"}
    assert first["decontaminated.jsonl"] == second["decontaminated.jsonl"]
    assert first["decontamination_drops.jsonl"] == second["decontamination_drops.jsonl"]
    # The manifest carries a timestamp; everything else in it must match.
    left = json.loads(first["decontamination.json"])
    right = json.loads(second["decontamination.json"])
    left.pop("at"), right.pop("at")
    assert left == right
    assert left["counts"]["dropped"] >= 1, "a run that drops nothing cannot show determinism"


# --------------------------------------------------------------------------
# The semantic seam. It has never executed against the real library, so what
# is pinned here is the SHAPE of the answer it is written against - and both
# ways a different shape could be read as "nothing was contaminated".
# --------------------------------------------------------------------------

def _semantic_run(tmp_path, monkeypatch, *, attr="selected", answer="real", extra=()):
    """A CLI run whose eval side holds one question and whose input holds a
    row that is a WORD-ORDER shuffle of it: invisible to every n-gram level,
    a duplicate to the semantic one."""
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    question = prose(301, 40)
    # A second, unrelated row: without one, a working seam empties the input
    # and the CLI's own "EVERYTHING WAS DROPPED" backstop fires instead.
    write_jsonl(
        paths.streams_dir / "replay.jsonl",
        [row(shuffled(question, 302)), row(prose(303, 60))],
    )
    store = Store.open(paths.state_db)
    all_eval_snapshots(store, tmp_path / "hf", {"bbl": [{"question": question}]})
    store.close()
    install_fake_semhash(monkeypatch, attr=attr, answer=answer)
    code = decon_main(["--config", cfg, "--no-generated", *extra])
    manifest_path = paths.out_dir / "decontamination.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    )
    return code, manifest, paths


def test_the_semantic_layer_catches_the_paraphrase_the_ngram_levels_cannot(tmp_path, monkeypatch):
    """The premise first: a word-order shuffle shares NO 13-gram with its
    source, so this drop can only have come from the semantic layer."""
    question = prose(301, 40)
    paraphrase = shuffled(question, 302)
    assert containment(
        gram_hashes(tokens(question), NGRAM), gram_hashes(tokens(paraphrase), NGRAM)
    ) == 0.0

    code, manifest, paths = _semantic_run(tmp_path, monkeypatch)
    assert code == 0
    assert manifest["semantic"] == "ran"
    assert manifest["counts"]["dropped"] == 1
    drops = [
        json.loads(line)
        for line in (paths.out_dir / "decontamination_drops.jsonl").read_text().splitlines()
    ]
    assert drops[0]["reason"] == "semantic:*"


def test_a_semhash_that_names_its_survivors_something_else_never_reads_as_screened(
    tmp_path, monkeypatch, capsys
):
    """The seam's own failure, in the direction that used to be invisible: a
    result object carrying `.deduplicated` instead of `.selected`. The old
    `getattr(result, "selected", None)` read that as 'not a duplicate' for
    every row, so the pass flagged nothing and the manifest - and the dataset
    card behind it - said the corpus had been semantically screened."""
    code, manifest, _ = _semantic_run(tmp_path, monkeypatch, attr="deduplicated")
    out = capsys.readouterr().out
    assert code == 0
    assert manifest["semantic"] == SEMANTIC_UNUSABLE
    assert manifest["semantic"] != "ran"
    assert ".selected" in manifest["semantic_detail"]
    assert "semantic layer did NOT run" in out
    # ... and it did not quietly drop things either.
    assert manifest["counts"]["dropped"] == 0


def test_a_semantic_seam_that_flags_everything_is_caught_before_it_empties_the_corpus(
    tmp_path, monkeypatch
):
    """The opposite drift, which fails loudly by ruining the yield instead of
    the screen. The control's negative half is what separates the two."""
    code, manifest, _ = _semantic_run(tmp_path, monkeypatch, answer="keep-nothing")
    assert code == 0
    assert manifest["semantic"] == SEMANTIC_UNUSABLE
    assert "nothing to do with Indian law" in manifest["semantic_detail"]
    assert manifest["counts"]["dropped"] == 0


def test_a_semantic_seam_that_flags_nothing_is_caught_too(tmp_path, monkeypatch):
    code, manifest, _ = _semantic_run(tmp_path, monkeypatch, answer="keep-everything")
    assert code == 0
    assert manifest["semantic"] == SEMANTIC_UNUSABLE
    assert "did not find a REWORDED copy of its control item" in manifest["semantic_detail"]


def test_require_semantic_refuses_when_the_layer_is_installed_but_not_working(
    tmp_path, monkeypatch, capsys
):
    """An operator who asked for the semantic layer gets a refusal, not a
    layer that was called and did nothing."""
    code, manifest, _ = _semantic_run(
        tmp_path, monkeypatch, attr="deduplicated", extra=("--require-semantic",)
    )
    assert code == 2
    assert manifest is None  # nothing was written
    assert "REFUSING TO DECONTAMINATE" in capsys.readouterr().out


def test_the_item_length_histogram_reaches_the_manifest_and_the_screen(tmp_path, capsys):
    """I2(d)'s whole point: this table is what turns the window calibration
    from an argument into a decision on run one. It was printed and recorded
    and NOTHING read either, so a histogram of zeros - or one that counted a
    band twice - passed the suite."""
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    write_jsonl(paths.streams_dir / "s.jsonl", [row(prose(940, 60))])
    store = Store.open(paths.state_db)
    # One item per band, plus one under the floor that nothing can match.
    # iltur, because it carries no verified row count and so no floor filler:
    # the histogram under test would otherwise be the filler's.
    all_eval_snapshots(store, tmp_path / "hf", {"iltur": [
        {"question": prose(941, 60)},   # text
        {"question": prose(942, 20)},   # narrow
        {"question": prose(943, 8)},    # short
        {"question": "far too short"},  # unmatchable (3 tokens)
    ]})
    store.close()
    assert decon_main(["--config", cfg, "--no-generated"]) == 0
    out = capsys.readouterr().out
    manifest = json.loads((paths.out_dir / "decontamination.json").read_text(encoding="utf-8"))

    bands = manifest["eval_sets"]["iltur"]["item_tokens"]
    assert bands[LEVEL_TEXT] == 1 and bands[LEVEL_NARROW] == 1
    assert bands[LEVEL_SHORT] == 1 and bands["unmatchable"] == 1
    assert (bands["min_tokens"], bands["max_tokens"]) == (3, 60)
    assert bands["median_tokens"] == 20
    assert sum(bands[k] for k in (LEVEL_TEXT, LEVEL_NARROW, LEVEL_SHORT, "unmatchable")) == 4
    # ... and the same numbers are on the operator's screen, under that set.
    assert (f"tokens: median {bands['median_tokens']} (min {bands['min_tokens']},"
            f" max {bands['max_tokens']})") in out
    assert (f"screened by: {LEVEL_TEXT} 1, {LEVEL_NARROW} 1, {LEVEL_SHORT} 1,"
            f" unmatchable 1") in out
    assert manifest["unmatchable_eval_items"]["iltur"] == 1
    # The band boundary the histogram is read against, in the same manifest.
    assert manifest["thresholds"]["full_ngram_from_tokens"] == 3 * NGRAM - 1 == 38
    assert window_for(38) == NGRAM and window_for(37) < NGRAM


def test_a_row_with_no_usable_text_reaches_the_manifest_counts_and_the_screen(tmp_path, capsys):
    """`empty_text` counts rows that pass by being unreadable rather than by
    being clean. Forcing it to 0 in the manifest survived the suite, so the
    hole could be counted internally and reported as none."""
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    write_jsonl(paths.streams_dir / "s.jsonl", [
        row(prose(950, 60)),
        {"messages": [{"role": "user", "content": "--- *** ,,,"}], "_prov": {}},
    ])
    store = Store.open(paths.state_db)
    all_eval_snapshots(store, tmp_path / "hf")
    store.close()
    assert decon_main(["--config", cfg, "--no-generated"]) == 0
    out = capsys.readouterr().out
    manifest = json.loads((paths.out_dir / "decontamination.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["empty_text"] == 1
    assert manifest["counts"]["total"] == 2 and manifest["counts"]["kept"] == 2
    assert "1 candidate rows carried NO USABLE TEXT" in out


def test_the_content_key_separates_the_prompt_from_the_answer():
    """`item_key` hashes prompt and answer with a NUL between them. Join them
    with a newline instead and ("a\\nb", "c") and ("a", "b\\nc") become the same
    row - two different rows that dedupe would then treat as one, silently."""
    from tuned.data.decontaminate import item_key

    assert item_key("a\nb", "c") != item_key("a", "b\nc")
    assert item_key("a", "b") == item_key("a", "b")
    assert "\x00" not in item_key("a", "b")  # the separator is in the payload, not the key


def test_this_module_cannot_reach_the_network():
    """The guard on the guard. The semantic layer is opt-in here through a
    fixture, and deleting that fixture left the suite green on an air-gapped
    machine while making 38 outbound HTTP attempts on a networked one - a
    suite whose meaning depends on which box it runs on. conftest.py refuses
    the socket layer for this module, and this is the assertion that says so
    rather than assuming it."""
    import socket

    with pytest.raises(Exception) as exc:  # noqa: B017 - the type is conftest's
        socket.create_connection(("huggingface.co", 443), timeout=1)
    assert "hermetic" in str(exc.value)


def test_a_machine_without_semhash_records_that_and_never_reads_as_screened(tmp_path, capsys):
    """The state this project was in until the extras landed, and the state
    every fresh clone is in. `semhash-not-installed` appeared in NO test, so
    initialising the status to `ran` instead survived the whole suite - and the
    manifest, the dataset card behind it and the operator's screen would all
    have said a paraphrase screen ran over a corpus it never touched."""
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    write_jsonl(paths.streams_dir / "s.jsonl", [row(prose(510, 60))])
    store = Store.open(paths.state_db)
    all_eval_snapshots(store, tmp_path / "hf")
    store.close()
    # No install_fake_semhash here: the autouse fixture leaves `import semhash`
    # raising ImportError, which is the un-installed machine exactly.
    assert decon_main(["--config", cfg, "--no-generated"]) == 0
    manifest = json.loads((paths.out_dir / "decontamination.json").read_text(encoding="utf-8"))
    assert manifest["semantic"] == SEMANTIC_UNAVAILABLE == "semhash-not-installed"
    assert manifest["semantic"] != "ran"
    assert "semantic layer did NOT run (semhash-not-installed)" in capsys.readouterr().out


def test_a_run_that_waived_every_eval_set_has_nothing_to_compare_and_says_so(
    tmp_path, monkeypatch, capsys
):
    """`no-eval-items-to-compare` also appeared in no test, and the branch is
    unreachable while the opt-in fixture is on - so `ran` here survived too. A
    run that waived every set must not record a semantic screen."""
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    write_jsonl(paths.streams_dir / "s.jsonl", [row(prose(511, 60))])
    Store.open(paths.state_db).close()  # nothing acquired: every set is missing
    install_fake_semhash(monkeypatch)
    waivers = [arg for key in sorted(EVAL_SETS) for arg in ("--allow-missing-eval", key)]
    assert decon_main(["--config", cfg, "--no-generated", *waivers]) == 0
    manifest = json.loads((paths.out_dir / "decontamination.json").read_text(encoding="utf-8"))
    assert manifest["semantic"] == SEMANTIC_NO_ITEMS == "no-eval-items-to-compare"
    assert manifest["semantic"] != "ran"
    assert "semantic layer did NOT run (no-eval-items-to-compare)" in capsys.readouterr().out


def test_a_model_that_cannot_be_fetched_is_not_a_drifted_api(tmp_path, monkeypatch, capsys):
    """Two failures with opposite remedies used to share one status. semhash
    downloads an embedding model on first use, so an installed-but-air-gapped
    machine - the state this module was WRITTEN on - recorded
    `semhash-control-failed`, whose documented reading is "the API drifted, fix
    selected_records". The instruction was false on that machine.

    This is also what makes the broad `except Exception` load-bearing: narrow
    it back to SemanticSeamError and this OSError kills a decontamination run
    the refusal ladder has already cleared."""
    code, manifest, _ = _semantic_run(tmp_path, monkeypatch, answer="no-model")
    out = capsys.readouterr().out
    assert code == 0
    assert manifest["semantic"] == SEMANTIC_NO_MODEL == "semhash-model-unavailable"
    assert manifest["semantic"] not in ("ran", SEMANTIC_UNUSABLE)
    assert "pre-warm the HuggingFace cache" in manifest["semantic_detail"]
    assert "NOT API drift" in manifest["semantic_detail"]
    assert "semantic layer did NOT run (semhash-model-unavailable)" in out
    # The exact stack still ran: this layer's absence is a status, not a crash.
    assert manifest["counts"]["total"] == 2


def test_a_seam_that_only_recognises_an_exact_copy_fails_the_control(tmp_path, monkeypatch):
    """The control's whole purpose. The one it replaced fed the seam an exact
    copy of an eval item, which is recognised by a seam with no semantic power
    whatsoever - measured against the real library, that control passed at
    every threshold from 0.3 to 0.95, including the 0.9 it shipped at, where
    the seam caught 0 of 2 paraphrases of an eval question."""
    code, manifest, _ = _semantic_run(tmp_path, monkeypatch, answer="exact-only")
    assert code == 0
    assert manifest["semantic"] == SEMANTIC_UNUSABLE
    assert "REWORDED copy" in manifest["semantic_detail"]


def test_a_seam_that_reads_the_whole_row_and_not_its_windows_fails_the_control(monkeypatch):
    """The other half of the same fix, and the reason the control's positive is
    a row rather than a sentence: measured against the installed library, an
    eval question quoted VERBATIM inside a 288-word row is invisible to a
    whole-row comparison at every threshold from 0.5 up, and at 0.4 and below a
    CLEAN row of the same length is flagged too. A whole-row seam has no
    operating point at all, and the control has to fail for it."""
    install_fake_semhash(monkeypatch)
    import tuned.data.decontaminate as decon

    monkeypatch.setattr(decon, "probe_texts", lambda text, **kw: [text] if text else [])
    with pytest.raises(SemanticSeamError) as exc:
        decon.semantic_control()
    assert "REWORDED copy" in str(exc.value)


def test_the_probe_windows_cover_every_position_of_the_row():
    """The geometry the control depends on: no word of the row sits outside
    every window, and the whole row is a probe in its own right because an
    IL-TUR judgment item is only comparable to the whole of one."""
    words = [f"w{i}" for i in range(95)]
    probes = probe_texts(" ".join(words))
    assert probes[0] == " ".join(words), "the whole row is always a probe"
    windows = probes[1:]
    assert all(len(w.split()) == SEMANTIC_PROBE_WORDS for w in windows)
    covered = set()
    for i, window in enumerate(windows):
        covered.update(range(i * SEMANTIC_PROBE_STRIDE,
                             i * SEMANTIC_PROBE_STRIDE + SEMANTIC_PROBE_WORDS))
    assert set(range(len(words))) <= covered
    # A row no longer than one window is not chopped up.
    assert probe_texts(" ".join(words[:SEMANTIC_PROBE_WORDS])) == [
        " ".join(words[:SEMANTIC_PROBE_WORDS])
    ]
    assert probe_texts("") == []


def test_the_semantic_operating_point_reaches_the_manifest(tmp_path, monkeypatch):
    """`semantic: ran` at one threshold is not the same screen as `ran` at
    another, and the dataset card has to be able to say which."""
    _, manifest, _ = _semantic_run(tmp_path, monkeypatch)
    assert manifest["thresholds"]["semantic"] == SEMANTIC_THRESHOLD == 0.8
    assert manifest["thresholds"]["semantic_probe_words"] == SEMANTIC_PROBE_WORDS
    assert manifest["thresholds"]["semantic_probe_stride"] == SEMANTIC_PROBE_STRIDE


def test_selected_records_raises_instead_of_defaulting():
    """The seam-level statement of the same fact, so a future caller cannot
    reintroduce the default without deleting this."""
    class Drifted:
        deduplicated = ["a"]

    with pytest.raises(SemanticSeamError) as exc:
        selected_records(Drifted())
    assert ".selected" in str(exc.value)
    assert selected_records(type("R", (), {"selected": ["a", "b"]})()) == ["a", "b"]
    assert selected_records(type("R", (), {"selected": None})()) == []


def test_row_form_falls_back_through_the_identity_fields():
    assert row_form(row("x", form="a", task_type="b")) == "a"
    assert row_form(row("x", task_type="b")) == "b"
    assert row_form(row("x")) == "test"  # the stream/source
    assert row_form({"messages": []}) == ""
