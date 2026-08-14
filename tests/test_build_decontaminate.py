import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest
from pipeline_fakes import open_store, paths_for, temp_config

from tuned.data.decontaminate import (
    CONTAINMENT,
    EVAL_NO_FILES,
    EVAL_NO_TEXT_COLUMN,
    EVAL_NOT_ACQUIRED,
    EVAL_OK,
    EVAL_SETS,
    EVAL_UNREADABLE,
    LEVEL_CASE_ID,
    LEVEL_NARROW,
    LEVEL_SHORT,
    LEVEL_TEXT,
    NGRAM,
    SEMANTIC_UNUSABLE,
    SHORT_MIN_TOKENS,
    TITLE_MIN_TOKENS,
    EvalIndex,
    EvalItem,
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


def all_eval_snapshots(store, root: Path, records_by_key=None):
    records_by_key = records_by_key or {}
    for key in EVAL_SETS:
        eval_snapshot(store, root, key, records_by_key.get(key, [{"question": prose(900 + ord(key[0]), 40)}]))


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


def install_fake_semhash(monkeypatch, *, attr="selected", answer="real"):
    """Put a `semhash` module in sys.modules.

    `attr` is what the result object names its survivors - the documented name
    is `selected`, and a build that renames it is the drift both seams have to
    fail loudly on rather than default around. `answer` is "real" (decide by
    similarity), "keep-everything" (nothing is ever a duplicate) or
    "keep-nothing" (everything is).
    """
    import sys
    import types

    class Result:
        def __init__(self, kept):
            setattr(self, attr, list(kept))

    class SemHash:
        def __init__(self, records):
            self.records = list(records)

        @classmethod
        def from_records(cls, records):
            return cls(records)

        def _kept(self, records, against, threshold):
            if answer == "keep-everything":
                return list(records)
            if answer == "keep-nothing":
                return []
            kept = []
            seen = list(against)
            for record in records:
                if not _near(record, seen, threshold):
                    kept.append(record)
                    seen.append(record)
            return kept

        def deduplicate(self, records, threshold=0.9):
            return Result(self._kept(records, self.records, threshold))

        def self_deduplicate(self, threshold=0.9):
            return Result(self._kept(self.records, [], threshold))

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
# Level 1: text containment, both sides of the threshold.
# --------------------------------------------------------------------------

def _row_sharing(question_words, share: int, filler: int = 60):
    """A row carrying `share` of an eval question's words, plus filler."""
    return " ".join(question_words[:share]) + " " + prose(77, filler)


def test_the_containment_threshold_decides_in_both_directions():
    words = prose(21, 120).split()
    question = " ".join(words)
    q_grams = gram_hashes(tokens(question), NGRAM)

    heavy = _row_sharing(words, 93)
    light = _row_sharing(words, 55)
    heavy_share = containment(q_grams, gram_hashes(tokens(heavy), NGRAM))
    light_share = containment(q_grams, gram_hashes(tokens(light), NGRAM))
    # The fixture STRADDLES the constant closely on both sides - it is neither
    # "identical vs unrelated" nor so far from the line that loosening the
    # threshold would go unnoticed.
    assert light_share < CONTAINMENT < heavy_share
    assert 0.35 < light_share < 0.45 and 0.6 < heavy_share < 0.85

    index = index_of(question)
    assert decontaminate_items(items(row(heavy)), index)[0] == []
    assert len(decontaminate_items(items(row(light)), index)[0]) == 1


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


def test_the_narrow_level_is_a_near_match_rule_and_not_a_loose_one():
    """Its control: a rule that fires on everything is not a rule. The same
    20-token question with SIX of its words changed is not a leak of it."""
    words = prose(401, 20).split()
    question = " ".join(words)
    paraphrase = " ".join(_edited(words, 6))
    leaked = prose(404, 300) + " " + paraphrase + " " + prose(405, 300)
    kept, drops, _ = decontaminate_items(items(row(leaked)), index_of(question))
    assert len(kept) == 1 and not drops


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
# Level 2: the short-item rule, and the case only it can carry.
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
# Level 3: case identifiers - the level no n-gram method can replace.
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
    assert title_key("State v Kumar") is None
    assert len(tokens("State v Kumar")) < TITLE_MIN_TOKENS + 1
    assert title_key("Kesavananda Bharati v State of Kerala") is not None


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


def test_the_cli_says_when_the_generations_were_not_screened(tmp_path, capsys):
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    write_jsonl(paths.streams_dir / "replay.jsonl", [row(prose(98, 40))])
    store = Store.open(paths.state_db)
    all_eval_snapshots(store, tmp_path / "hf")
    store.close()

    assert decon_main(["--config", cfg, "--no-generated"]) == 0
    out = capsys.readouterr().out
    manifest = json.loads((paths.out_dir / "decontamination.json").read_text(encoding="utf-8"))
    assert manifest["generations_screened"] is False
    assert "THE ACCEPTED GENERATIONS WERE NOT SCREENED" in out

    assert decon_main(["--config", cfg]) == 0
    out = capsys.readouterr().out
    manifest = json.loads((paths.out_dir / "decontamination.json").read_text(encoding="utf-8"))
    assert manifest["generations_screened"] is True
    assert "THE ACCEPTED GENERATIONS WERE NOT SCREENED" not in out


def test_eval_rows_that_carry_no_question_column_are_reported_even_when_the_set_loads(
    tmp_path, capsys
):
    """`ok` on a set 90% of whose questions were never read is the same
    failure as a missing set, one layer down."""
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    write_jsonl(paths.streams_dir / "replay.jsonl", [row(prose(99, 40))])
    store = Store.open(paths.state_db)
    all_eval_snapshots(
        store, tmp_path / "hf",
        {"bbl": [{"question": prose(100, 40)}] + [{"mystery_column": "x"}] * 9},
    )
    store.close()

    assert decon_main(["--config", cfg, "--no-generated"]) == 0
    out = capsys.readouterr().out
    assert "9 of 10 rows carried none of" in out
    manifest = json.loads((paths.out_dir / "decontamination.json").read_text(encoding="utf-8"))
    assert (manifest["eval_sets"]["bbl"]["rows"], manifest["eval_sets"]["bbl"]["items"]) == (10, 1)


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
    spec = EVAL_SETS["bbl"]
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
    ship - is behind a lazy pyarrow import that has never executed in this
    worktree, and a reader that raises is caught as EVAL_UNREADABLE (a
    refusal), never as an empty set."""
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
    assert ".parquet" in _READABLE_SUFFIXES


def test_the_split_filter_prefers_the_named_split_but_never_empties_the_set(store, tmp_path):
    spec = EVAL_SETS["bbl"]
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
    assert "opennyaiorg/aibe" in out
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
    assert "screened 2  kept 1  dropped 1" in out


# --------------------------------------------------------------------------
# Determinism.
# --------------------------------------------------------------------------

def _cli_bytes(tmp_path, seed: str, module: str, args) -> dict:
    """Run a CLI in a subprocess under a chosen PYTHONHASHSEED; return the
    bytes of everything it wrote."""
    env = {**os.environ, "PYTHONHASHSEED": seed}
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
    assert "did not flag an eval item against its own index" in manifest["semantic_detail"]


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
