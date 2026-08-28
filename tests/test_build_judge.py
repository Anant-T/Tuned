import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from pipeline_fakes import (
    LONG_SEED_TEXT,
    NARROW_GENERATOR_CONTEXT,
    TRANSITION_META,
    FakeRouter,
    StealsTheLease,
    build_cfg,
    cfg_with_context,
    cfg_with_fourth_judge_family,
    cfg_with_gpt_oss_as_sole_generator,
    cfg_with_two_generator_families,
    cfg_without_the_free_tiebreak,
    cfg_without_the_paid_judges,
    cfg_without_the_promoted_judge,
    chat_response,
    judge_reply,
    open_store,
    paths_for,
    temp_config,
)

from tuned.data.generate import build_prompt, judge_messages, judge_needed_tokens, run_workers
from tuned.data.jsonl import read_at
from tuned.data.providers import ProviderError
from tuned.data import judge as judge_mod
from tuned.data.judge import (
    BORDERLINE,
    FAIL,
    JUDGE_MAX_TOKENS,
    JUDGE_STATE_FROM,
    JUDGE_STATE_TO,
    MAX_JUDGE_ATTEMPTS,
    PASS,
    JudgeParseError,
    JudgeScores,
    JudgeStats,
    SlotOutcome,
    _outcome_from_row,
    decide,
    failing_rationale,
    generation_family,
    judge_task,
    parse_judge_reply,
    run_judges,
    split_reply_think,
    thresholds_active,
    undersized_families,
)
from tuned.data.judge import main as judge_main
from tuned.data.tasks import plan_wave, reopen_tasks


@pytest.fixture
def cfg():
    return build_cfg()


@pytest.fixture
def paths(tmp_path):
    return paths_for(tmp_path)


def judged_store(tmp_path, paths, cfg, n=1, generator_script=None):
    """A store whose n tasks have been generated and are waiting in 'judging'.

    The generator is pinned to cerebras/gpt-oss-120b, because this whole
    suite is written against "the generator was gpt-oss" (family separation,
    model ids, event fields below all assume it).

    Used to get that pin for free by excluding bai's key from the shipped
    two-ref routing.generator list, falling over to cerebras. As of
    2026-08-28 bai/deepseek-v4-flash is the SOLE shipped generator ref
    (operator directive: deepseek is the sole generator, cerebras spends
    only on judging) - excluding bai now leaves nothing to fall over to.
    cfg_with_gpt_oss_as_sole_generator builds its OWN generator list pointing
    at cerebras/gpt-oss-120b instead (still valid in cfg.providers, just no
    longer routed to in production), the same way every other
    synthetic-routing fixture in this module does (see
    cfg_with_two_generator_families) rather than relying on the shipped
    config to carry a ref this suite needs for its own reasons.
    """
    gen_cfg = cfg_with_gpt_oss_as_sole_generator(cfg)
    store = open_store(tmp_path, n_seeds=n)
    plan_wave(store, gen_cfg, "synthesis", n, task_type_mix={"irac_analysis": 1.0})
    asyncio.run(
        run_workers(
            store, gen_cfg, FakeRouter(gen_cfg, generator_script), paths=paths,
            streams=["synthesis"], n_workers=n, max_batches=1,
        )
    )
    assert store.task_counts() == {"judging": n}
    return store


def run_judge(store, cfg, router, paths, **kwargs):
    kwargs.setdefault("streams", ["synthesis"])
    kwargs.setdefault("n_workers", 4)
    kwargs.setdefault("max_batches", 1)
    return asyncio.run(run_judges(store, cfg, router, paths=paths, **kwargs))


def only_task(store):
    return dict(store.conn.execute("SELECT * FROM task LIMIT 1").fetchone())


def scores(g=5, v=5, c=5, rationale="fine"):
    return JudgeScores(grounding=g, validity=v, coverage=c, rationale=rationale)


# Exact model ids the shipped FakeRouter lands on for slots a / b / tiebreak.
QWEN = "qwen/qwen3.6-27b"
GEMMA = "gemma-4-31b"
MISTRAL = "mistral-large-latest"


def _activate_rules(store, *specs):
    """Write active judge_threshold rows. Each spec is (model, rule, threshold)."""
    store.record_judge_thresholds(
        [
            {
                "calib_id": f"task4-{i}-{model}",
                "model": model,
                "rule": rule,
                "threshold": threshold,
            }
            for i, (model, rule, threshold) in enumerate(specs)
        ]
    )


# --------------------------------------------------------------------------
# The parser (contract 3).
# --------------------------------------------------------------------------

def test_parser_accepts_the_short_axis_names():
    parsed = parse_judge_reply('{"grounding": 4, "validity": 3, "coverage": 5, "rationale": "x"}')
    assert (parsed.grounding, parsed.validity, parsed.coverage) == (4, 3, 5)
    assert parsed.rationale == "x"


def test_parser_accepts_the_rubric_axis_names():
    parsed = parse_judge_reply(
        '{"grounding_faithfulness": 2, "reasoning_validity": 4, "issue_coverage": 3}'
    )
    assert (parsed.grounding, parsed.validity, parsed.coverage) == (2, 4, 3)


def test_parser_accepts_mixed_aliases_and_odd_casing():
    parsed = parse_judge_reply(
        '{"Grounding_Faithfulness": 5, "validity": 4, "Issue_Coverage": 4}'
    )
    assert (parsed.grounding, parsed.validity, parsed.coverage) == (5, 4, 4)


def test_parser_accepts_a_json_fence():
    parsed = parse_judge_reply(
        'Sure.\n```json\n{"grounding": 3, "validity": 3, "coverage": 3}\n```'
    )
    assert parsed.min_axis == 3


def test_parser_accepts_json_buried_in_prose():
    parsed = parse_judge_reply(
        "I have read the work carefully.\n"
        '{"grounding": 5, "validity": 4, "coverage": 4, "rationale": "solid"}\n'
        "Let me know if you need more."
    )
    assert parsed.verdict == PASS


def test_parser_takes_the_last_complete_object():
    text = (
        'The contract is {"grounding": 4, "validity": 2, "coverage": 3, "rationale": "example"}. '
        'My verdict: {"grounding": 5, "validity": 5, "coverage": 5, "rationale": "mine"}'
    )
    assert parse_judge_reply(text).rationale == "mine"


def test_parser_survives_braces_inside_the_rationale():
    parsed = parse_judge_reply(
        '{"grounding": 4, "validity": 4, "coverage": 4, "rationale": "uses {curly} braces"}'
    )
    assert parsed.rationale == "uses {curly} braces"


def test_parser_coerces_string_and_float_scores():
    parsed = parse_judge_reply('{"grounding": "4", "validity": 4.0, "coverage": "3/5"}')
    assert (parsed.grounding, parsed.validity, parsed.coverage) == (4, 4, 3)


@pytest.mark.parametrize(
    "text",
    [
        "I cannot evaluate this work.",
        '{"grounding": 4, "validity": 4}',
        '{"grounding": 0, "validity": 4, "coverage": 4}',
        '{"grounding": 9, "validity": 4, "coverage": 4}',
        '{"grounding": true, "validity": 4, "coverage": 4}',
        '{"grounding": "good", "validity": 4, "coverage": 4}',
        "",
    ],
)
def test_parser_rejects_unscorable_replies(text):
    with pytest.raises(JudgeParseError):
        parse_judge_reply(text)


# --------------------------------------------------------------------------
# The parser, on a judge that thinks out loud (contract 3, 2026-08-18).
# --------------------------------------------------------------------------
# groq/qwen/qwen3.6-27b inlines <think>...</think> and reasons about the rubric
# inside it before answering. Three shapes have to come apart, and the middle
# one is the whole reason this exists.

THINK_PREAMBLE = (
    "<think>\nHere's a thinking process:\n\n1. **Analyze User Input:**\n"
    "   - **Role:** Strict evaluator of legal reasoning. Score 1-5 on three axes.\n"
    "   - The schema wants something like "
    '{"grounding": 1, "validity": 1, "coverage": 1, "rationale": "..."} '
    "which I should not fill in yet.\n"
    "2. Weigh the trace. The chain has a gap at its centre.\n"
)


def test_parser_scores_the_object_after_a_closed_think_block():
    """The shipped qwen shape once it finishes: reason, close, then answer."""
    parsed = parse_judge_reply(
        THINK_PREAMBLE + "</think>\n\n"
        '{"grounding": 4, "validity": 3, "coverage": 5, "rationale": "gap at the centre"}'
    )
    assert (parsed.grounding, parsed.validity, parsed.coverage) == (4, 3, 5)
    assert parsed.rationale == "gap at the centre"


def test_parser_refuses_an_object_that_never_leaves_the_think_block():
    """THE ONE THIS EXISTS FOR. The preamble above carries a complete, valid,
    all-1s object - the model quoting the schema at itself - and the reply then
    closes with no verdict at all.

    Scoring it would be a judgement nobody made, written into the judgement
    table at min_axis 1 and read later by P5 calibration and gold labelling as
    a real verdict. It is also the exact failure this codebase keeps finding in
    its own instruments: the machinery reporting a result in the case it exists
    to catch. The fixture has to be able to EXPRESS the misreading, so the
    object inside the block is well-formed and would parse on its own - which
    the assertion below checks, so this test cannot pass by accident."""
    inside_only = THINK_PREAMBLE + "</think>\n\nI will stop here."
    with pytest.raises(JudgeParseError) as excinfo:
        parse_judge_reply(inside_only)
    assert "after the reasoning block" in str(excinfo.value)
    # The object really is scorable when it is not sealed inside the block.
    assert parse_judge_reply(
        '{"grounding": 1, "validity": 1, "coverage": 1, "rationale": "..."}'
    ).min_axis == 1


def test_parser_refuses_a_think_block_that_never_closes():
    """The shape 7 of 7 slot-B replies actually had on 2026-08-18: the model
    spent its whole 1,024-token budget inside <think> and was cut mid-word. A
    verdict guessed out of half a thought is not a verdict, so this is a parse
    error - which costs one retried slot and never a score."""
    with pytest.raises(JudgeParseError) as excinfo:
        parse_judge_reply(THINK_PREAMBLE + "3. The delay is unexplained on these pap")
    assert "unclosed <think>" in str(excinfo.value)


def test_parser_refuses_a_reply_that_reopens_a_block_it_never_closes():
    """Truncation after a verdict is still truncation. One test covers both
    ways a reply can stop mid-thought, so a reply that answered and then
    started thinking again cannot be scored on the answer it had abandoned."""
    with pytest.raises(JudgeParseError):
        parse_judge_reply(
            "<think>done</think>\n"
            '{"grounding": 4, "validity": 4, "coverage": 4}\n'
            "<think>actually, on reflection"
        )


def test_parser_still_takes_a_plain_reply_with_no_think_at_all():
    """Slot A must not regress. mistral-small-latest returns a fenced object
    and no tags whatever, and it is the model that produced every judgement in
    the store; a think-aware parser that needed a think block would have taken
    the working half of the pool down with the broken one."""
    parsed = parse_judge_reply(
        '```json\n{\n  "grounding": 5,\n  "validity": 3,\n  "coverage": 4,\n'
        '  "rationale": "compresses the step from the error to the restoration."\n}\n```'
    )
    assert (parsed.grounding, parsed.validity, parsed.coverage) == (5, 3, 4)
    assert parsed.verdict == BORDERLINE


def test_parser_refuses_a_draft_object_sealed_in_an_UPPERCASE_think_block():
    """The reviewer's measured escape, and the reason this parser matches the
    tags case-insensitively while gates._tag_positions does not.

    An unrecognised close tag makes the whole reply scorable, the draft schema
    object inside the block becomes the last complete object in it, and the
    truncation guard misses the unrecognised OPEN tag on the way past. This
    exact input returned grounding/validity/coverage 1/1/1 before the fix - a
    verdict nobody voted for, written to the judgement table, reused by every
    later pass through judge_slot_reused, and read by P5 calibration as real.

    gates fails the other way on a cased tag (the whole content flows to the
    answer-side gates, which reject rather than admit), which is why the
    asymmetry is kept rather than harmonized."""
    sealed = (
        "<THINK>\nThe schema wants something like "
        '{"grounding": 1, "validity": 1, "coverage": 1, "rationale": "..."} '
        "which I should fill in once I have weighed the trace.\n</THINK>\nno verdict."
    )
    with pytest.raises(JudgeParseError):
        parse_judge_reply(sealed)


def test_parser_reads_a_verdict_after_a_mixed_case_close_tag():
    """Case-insensitivity may not cost the reply that DOES answer: a model
    whose tags are cased is still entitled to be scored on what follows
    them."""
    parsed = parse_judge_reply(
        "<Think>weighing the trace, and the chain has a gap</Think>\n"
        '{"grounding": 4, "validity": 4, "coverage": 5, "rationale": "gap"}'
    )
    assert (parsed.grounding, parsed.validity, parsed.coverage) == (4, 4, 5)


def test_parser_takes_the_verdict_after_the_LAST_close_not_the_first():
    """LAST, not first, and this is the only fixture that can tell them apart.

    Every other reply in this suite carries exactly one </think>, so
    `rfind -> find` survives all of them. Here a model thinks, says something,
    thinks again with a DECOY object in the second block, and only then
    answers. Reading from the first close hands the whole second thought back
    as scorable text - which trips the unclosed-block guard on the reopened
    tag, so the mutant fails loudly rather than scoring the decoy. Either way
    the real verdict is not what comes out, which is what this pins."""
    parsed = parse_judge_reply(
        "<think>first pass</think>\n"
        "Let me reconsider.\n"
        '<think>on reflection, the schema example is '
        '{"grounding": 1, "validity": 1, "coverage": 1, "rationale": "decoy"}'
        "</think>\n"
        '{"grounding": 5, "validity": 4, "coverage": 4, "rationale": "settled"}'
    )
    assert (parsed.grounding, parsed.validity, parsed.coverage) == (5, 4, 4)
    assert parsed.rationale == "settled"


def test_the_reasoning_block_diagnostic_needs_an_opened_block_not_a_stray_tag():
    """A rationale that quotes "</think>" inside a JSON string leaves a
    non-empty `think` prefix with no block ever opened in it. Naming a
    reasoning block there sends the operator after model behaviour that did
    not happen; the message has to key on the OPEN tag."""
    stray = '{"grounding": 4, "validity": 4, "note": "the model wrote </think> here"}'
    with pytest.raises(JudgeParseError) as excinfo:
        parse_judge_reply(stray)
    assert "after the reasoning block" not in str(excinfo.value)


def test_split_reply_think_returns_none_when_there_is_no_block():
    """think=None is what tells parse_judge_reply not to blame a reasoning
    block for a reply that never had one."""
    assert split_reply_think('{"grounding": 4}') == (None, '{"grounding": 4}')
    think, scorable = split_reply_think("<think>weighing</think> verdict follows")
    assert think == "<think>weighing</think>"
    assert scorable == " verdict follows"


def test_verdict_bands():
    assert scores(4, 4, 4).verdict == PASS
    assert scores(5, 5, 3).verdict == BORDERLINE
    assert scores(5, 5, 2).verdict == FAIL
    assert scores(1, 5, 5).min_axis == 1


# --------------------------------------------------------------------------
# The decision matrix (pure).
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "verdicts,already,expected",
    [
        ([PASS, PASS], False, "accept"),
        ([FAIL, FAIL], False, "reject"),
        ([PASS, FAIL], False, "tiebreak"),
        ([FAIL, PASS], False, "tiebreak"),
        ([PASS, BORDERLINE], False, "tiebreak"),
        ([BORDERLINE, BORDERLINE], False, "regenerate"),
        ([BORDERLINE, FAIL], False, "regenerate"),
        ([BORDERLINE, BORDERLINE], True, "reject"),
        ([PASS, FAIL, PASS], False, "accept"),
        ([PASS, FAIL, FAIL], False, "reject"),
        ([PASS, FAIL, BORDERLINE], False, "regenerate"),
        ([PASS, FAIL, BORDERLINE], True, "reject"),
    ],
)
def test_decision_matrix(verdicts, already, expected):
    assert decide(verdicts, already_regenerated=already) == expected


def test_decide_needs_a_verdict():
    with pytest.raises(ValueError):
        decide([], already_regenerated=False)


def test_failing_rationale_takes_the_harshest_judge():
    outcomes = [
        SlotOutcome(slot="a", scores=scores(5, 5, 5, "all good")),
        SlotOutcome(slot="b", scores=scores(5, 2, 4, "the conclusion does not follow")),
    ]
    assert failing_rationale(outcomes) == "the conclusion does not follow"
    assert failing_rationale([SlotOutcome(slot="a")]) == ""


# --------------------------------------------------------------------------
# Routing (contracts 2 and 5).
# --------------------------------------------------------------------------

def test_undersized_families_excludes_the_judges_too_small_for_the_row(cfg):
    """The judge pool has NO SMALL TIER AT ALL since 2026-08-19. zai-glm-4.7
    (8k) left archived on 2026-08-18; mistral-small (32k) lost the judge seat
    after human calibration disqualified it, and gemma - promoted into that
    seat - is 131k. Every judge now declares 131k or more.

    The EMPTY sets are the assertion, not placeholders: they say this filter
    cannot remove a judge at any row size this build produces, so a routing
    test that wants a judge excluded ON LENGTH has to supply one."""
    assert undersized_families(cfg, "judge", 4000) == frozenset()
    assert undersized_families(cfg, "judge", 20000) == frozenset()
    assert undersized_families(cfg, "judge", 40000) == frozenset()
    # The smallest judge window is 131,072, so the first exclusion is there.
    assert undersized_families(cfg, "judge", 104_858) == frozenset()
    assert undersized_families(cfg, "judge", 104_859) == frozenset({"gemma", "qwen"})


def test_generation_family_falls_back_to_the_config(cfg):
    assert generation_family(cfg, {"model_family": "gpt-oss"}) == "gpt-oss"
    assert generation_family(
        cfg, {"model_family": None, "provider": "groq", "model": "qwen/qwen3.6-27b"}
    ) == "qwen"
    assert generation_family(cfg, {"provider": "nope", "model": "nope"}) is None


def test_judges_exclude_the_generator_family_and_each_other(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        router = FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]})
        run_judge(store, cfg, router, paths)
        calls = router.calls_for("judge")
        assert len(calls) == 2
        # The generator was cerebras/gpt-oss-120b (family gpt-oss).
        assert "gpt-oss" in calls[0]["exclude_families"]
        assert calls[0]["ref"].model == "qwen/qwen3.6-27b"
        # Judge B also excludes judge A's family, so it is a different model.
        assert {"gpt-oss", "qwen"} <= calls[1]["exclude_families"]
        assert calls[1]["ref"].model == "gemma-4-31b"


def test_a_long_candidate_past_8k_still_fills_both_slots(tmp_path, cfg, paths):
    """What retiring the 8k judge cost at this row size: nothing.

    A row past 8,192 routing tokens used to have zai-glm-4.7 struck out of the
    pool on context length before either slot was picked. The model is gone
    (archived upstream, 2026-08-18) and NO family is excluded on length here
    any more - since 2026-08-19 the smallest judge window is 131,072 - so both
    slots fill from the same two models a short row uses.

    The misreading it rejects is that dropping a model from a pool must make
    some row harder to route. It does not here, because the model dropped could
    never serve this row either: it was excluded on length exactly when it was
    needed, and all it did in the meantime was spend an attempt per route."""
    with judged_store(tmp_path, paths, cfg) as store:
        gen = store.latest_generation(only_task(store)["task_id"])
        store.conn.execute(
            "UPDATE generation SET think = ? WHERE gen_id = ?",
            ("word " * 12000, gen["gen_id"]),
        )
        router = FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]})
        run_judge(store, cfg, router, paths)
        calls = router.calls_for("judge")
        assert calls[0]["est_tokens"] > 8192
        assert calls[0]["exclude_families"] == frozenset({"gpt-oss"})
        assert calls[0]["ref"].model == "qwen/qwen3.6-27b"
        assert calls[1]["exclude_families"] == frozenset({"gpt-oss", "qwen"})
        assert calls[1]["ref"].model == "gemma-4-31b"


def test_a_candidate_past_every_small_judge_lands_on_the_131k_model(tmp_path, cfg, paths):
    """A NARROWED gemma is what makes this measure anything.

    The premise used to come free: qwen was 131k and mistral-small 32k, so a
    40,000-word trace excluded mistral and landed on qwen. Since 2026-08-19
    every judge is 131k or more and nothing is excluded on length at any row
    size, so the exclusion has to be constructed.
    """
    cfg = _narrow_judge(cfg)
    with judged_store(tmp_path, paths, cfg) as store:
        gen = store.latest_generation(only_task(store)["task_id"])
        store.conn.execute(
            "UPDATE generation SET think = ? WHERE gen_id = ?",
            ("word " * 40000, gen["gen_id"]),
        )
        router = FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]})
        run_judge(store, cfg, router, paths)
        calls = router.calls_for("judge")
        assert {"gpt-oss", "gemma"} <= calls[0]["exclude_families"]
        assert calls[0]["ref"].model == "qwen/qwen3.6-27b"


def test_the_judge_sees_the_same_materials_as_the_generator(tmp_path, cfg, paths):
    from pipeline_fakes import SEED_TEXT

    with judged_store(tmp_path, paths, cfg) as store:
        router = FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]})
        run_judge(store, cfg, router, paths)
        prompt = router.calls_for("judge")[0]["messages"][-1]["content"]
        assert SEED_TEXT in prompt
        gen = store.latest_generation(only_task(store)["task_id"])
        assert gen["think"][:80] in prompt
        assert gen["answer"][:80] in prompt


# --------------------------------------------------------------------------
# Decisions end to end.
# --------------------------------------------------------------------------

def test_two_passes_accept_and_record_both_slots(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        totals = run_judge(store, cfg, FakeRouter(cfg, {"judge": [judge_reply(5, 4, 4)]}), paths)
        assert totals["accepted"] == 1
        task = only_task(store)
        assert task["state"] == "accepted"
        assert task["disposition"] == "judge:accept"
        gen = store.latest_generation(task["task_id"])
        judgements = store.judgements_for(gen["gen_id"])
        assert [j["judge_slot"] for j in judgements] == ["a", "b"]
        assert judgements[0]["grounding"] == 5
        assert judgements[0]["provider"] == "groq"
        assert store.accepted_count("synthesis") == 1


def test_two_fails_reject(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        totals = run_judge(store, cfg, FakeRouter(cfg, {"judge": [judge_reply(1, 2, 2)]}), paths)
        assert totals["rejected"] == 1
        assert only_task(store)["state"] == "rejected"


def test_disagreement_goes_to_a_third_family(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        router = FakeRouter(
            cfg,
            {
                "judge": [judge_reply(5, 5, 5), judge_reply(2, 2, 2)],
                "tiebreak": [judge_reply(5, 5, 5)],
            },
        )
        totals = run_judge(store, cfg, router, paths)
        assert totals["tiebreaks"] == 1
        assert totals["accepted"] == 1
        tiebreak = router.calls_for("tiebreak")[0]
        # Excludes the generator AND both judges: a genuinely third family.
        # Since 2026-08-19 that third family is mistral, which is exactly why
        # mistral-large-latest was put in the tiebreak seat: it is the one
        # family left once {gpt-oss, qwen, gemma} are spent.
        assert {"gpt-oss", "qwen", "gemma"} <= tiebreak["exclude_families"]
        assert tiebreak["ref"].model == "mistral-large-latest"
        gen = store.latest_generation(only_task(store)["task_id"])
        assert [j["judge_slot"] for j in store.judgements_for(gen["gen_id"])] == [
            "a", "b", "tiebreak",
        ]


def test_a_failing_tiebreak_rejects(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        router = FakeRouter(
            cfg,
            {
                "judge": [judge_reply(5, 5, 5), judge_reply(1, 1, 1)],
                "tiebreak": [judge_reply(2, 2, 2)],
            },
        )
        run_judge(store, cfg, router, paths)
        assert only_task(store)["state"] == "rejected"


def test_a_borderline_pair_buys_exactly_one_regeneration(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        router = FakeRouter(
            cfg,
            {
                "judge": [judge_reply(3, 5, 5, "a material issue is left unresolved")],
                "generator": [chat_response()],
            },
        )
        totals = run_judge(store, cfg, router, paths)
        # One regeneration, then the second borderline reading is terminal.
        assert totals["regenerated"] == 1
        assert len(router.calls_for("generator")) == 1
        assert only_task(store)["state"] == "rejected"
        note_call = router.calls_for("generator")[0]
        assert "a material issue is left unresolved" in note_call["messages"][-1]["content"]
        assert store.events("judge_regeneration")
        # Two generations, the second flagged as the rationale-fed one.
        params = [
            json.loads(row[0])
            for row in store.conn.execute(
                "SELECT params_json FROM generation ORDER BY attempt"
            ).fetchall()
        ]
        assert [p["reviewer_note_applied"] for p in params] == [False, True]


def test_a_regeneration_that_satisfies_the_judges_is_accepted(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        router = FakeRouter(
            cfg,
            {
                "judge": [
                    judge_reply(3, 5, 5, "thin"),
                    judge_reply(3, 5, 5, "thin"),
                    judge_reply(5, 5, 5),
                ],
                "generator": [chat_response()],
            },
        )
        totals = run_judge(store, cfg, router, paths)
        assert totals["regenerated"] == 1
        assert only_task(store)["state"] == "accepted"
        # The new generation was judged, not the old one.
        gen = store.latest_generation(only_task(store)["task_id"])
        assert gen["attempt"] == 2
        assert len(store.judgements_for(gen["gen_id"])) == 2


def test_a_regeneration_that_fails_the_gates_takes_the_gate_disposition(tmp_path, cfg, paths):
    from pipeline_fakes import FABRICATED_ANSWER, CLEAN_THINK

    with judged_store(tmp_path, paths, cfg) as store:
        router = FakeRouter(
            cfg,
            {
                "judge": [judge_reply(3, 3, 3, "thin")],
                "generator": [chat_response(FABRICATED_ANSWER, CLEAN_THINK)],
            },
        )
        run_judge(store, cfg, router, paths)
        task = only_task(store)
        assert task["state"] == "rejected"
        assert task["disposition"].startswith("reject:")
        assert "citations" in task["disposition"]


def test_the_decision_event_is_marked_provisional(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        assert thresholds_active(store) == 0
        run_judge(store, cfg, FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]}), paths)
        event = json.loads(store.events("judge_decision")[0]["detail_json"])
        assert event["provisional"] is True
        assert event["action"] == "accept"
        assert event["verdicts"] == ["pass", "pass"]
        assert event["generator_family"] == "gpt-oss"
        assert [s["slot"] for s in event["scores"]] == ["a", "b"]


def test_calibrated_thresholds_flip_the_provisional_flag(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        _activate_rules(store, (QWEN, "min_axis", 4), (GEMMA, "min_axis", 4))
        assert thresholds_active(store) == 2
        run_judge(store, cfg, FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]}), paths)
        event = json.loads(store.events("judge_decision")[0]["detail_json"])
        assert event["provisional"] is False


# --------------------------------------------------------------------------
# Raw-first durability for judgements.
# --------------------------------------------------------------------------

def test_judgement_raw_envelope_round_trips(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        run_judge(store, cfg, FakeRouter(cfg, {"judge": [judge_reply(5, 4, 4, "ok")]}), paths)
        gen = store.latest_generation(only_task(store)["task_id"])
        row = store.judgements_for(gen["gen_id"])[0]
        record = read_at(row["raw_path"], row["raw_offset"])
        assert record["kind"] == "judgement"
        assert record["task_id"] == gen["task_id"]
        assert record["attempt"] == gen["attempt"]
        assert record["judge_slot"] == "a"
        assert record["grounding"] == 5
        assert record["prompt_id"] == "judge_pointwise_v1"
        assert '"grounding": 5' in record["reply_text"]


def test_judgements_reconcile_into_a_fresh_database(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        run_judge(store, cfg, FakeRouter(cfg, {"judge": [judge_reply(4, 4, 4)]}), paths)
        gen = store.latest_generation(only_task(store)["task_id"])
        judge_raw = store.judgements_for(gen["gen_id"])[0]["raw_path"]
        gen_raw = gen["raw_path"]

    with open_store(tmp_path / "rebuild", n_seeds=1) as rebuilt:
        plan_wave(rebuilt, cfg, "synthesis", 1, task_type_mix={"irac_analysis": 1.0})
        # Judge log FIRST, so the deferred-orphan path has to resolve it.
        assert rebuilt.reconcile_raw([judge_raw, gen_raw]) == 3
        recovered = rebuilt.latest_generation(gen["task_id"])
        judgements = rebuilt.judgements_for(recovered["gen_id"])
        assert [j["judge_slot"] for j in judgements] == ["a", "b"]
        assert [j["grounding"] for j in judgements] == [4, 4]
        # The surrogate gen_id was re-issued by the rebuild; the judgements
        # still bound to the right generation because the envelope carries
        # the natural key.
        assert rebuilt.reconcile_raw([judge_raw, gen_raw]) == 0


def test_an_unparsable_reply_is_durable_but_never_a_judgement(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        router = FakeRouter(cfg, {"judge": [chat_response("I refuse to grade this.", None)]})
        totals = run_judge(store, cfg, router, paths)
        assert totals["slot_errors"] == 1
        # Retried once via the router, then the slot gives up.
        assert len(router.calls_for("judge")) == 2
        assert len(store.events("judge_parse_error")) == 2
        task = only_task(store)
        assert task["state"] == JUDGE_STATE_FROM
        gen = store.latest_generation(task["task_id"])
        assert store.judgements_for(gen["gen_id"]) == []
        # The paid reply is on disk under a kind reconcile_raw will not index.
        record = read_at(str(next((paths.root / "raw" / "judge").rglob("judge.ndjson"))), 0)
        assert record["kind"] == "judge_error"
        assert record["reply_text"] == "I refuse to grade this."


def test_an_unparsable_reply_is_not_recovered_as_a_judgement(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        run_judge(store, cfg, FakeRouter(cfg, {"judge": [chat_response("no.", None)]}), paths)
        raw = str(next((paths.root / "raw" / "judge").rglob("judge.ndjson")))
        assert store.reconcile_raw([raw]) == 0
        assert store.events("reconcile_unknown_kind")


def test_judging_is_parked_after_too_many_failed_attempts(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        task_id = only_task(store)["task_id"]
        store.conn.execute(
            "UPDATE task SET attempts = ? WHERE task_id = ?", (MAX_JUDGE_ATTEMPTS, task_id)
        )
        run_judge(store, cfg, FakeRouter(cfg, {"judge": [chat_response("no.", None)]}), paths)
        task = only_task(store)
        assert task["state"] == "judge_error"
        assert task["disposition"].startswith("judge-slot-a:")


# --------------------------------------------------------------------------
# Rows that must never reach a judge (contract 4).
# --------------------------------------------------------------------------

def test_an_empty_trace_is_never_judged(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        gen = store.latest_generation(only_task(store)["task_id"])
        store.conn.execute("UPDATE generation SET think = '' WHERE gen_id = ?", (gen["gen_id"],))
        router = FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]})
        totals = run_judge(store, cfg, router, paths)
        assert router.calls == []
        assert totals["skipped"] == 1
        assert only_task(store)["state"] == "judge_skipped"
        assert only_task(store)["disposition"] == "empty-think"


def test_a_non_judgeable_stream_is_parked(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        store.conn.execute("UPDATE task SET stream = 'replay'")
        router = FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]})
        run_judge(store, cfg, router, paths, streams=["replay"])
        assert router.calls == []
        assert only_task(store)["state"] == "judge_skipped"
        assert only_task(store)["disposition"] == "stream-not-judgeable"


def test_a_task_without_a_generation_is_parked(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        gen = store.latest_generation(only_task(store)["task_id"])
        store.conn.execute("DELETE FROM gate_result WHERE gen_id = ?", (gen["gen_id"],))
        store.conn.execute("DELETE FROM generation WHERE gen_id = ?", (gen["gen_id"],))
        run_judge(store, cfg, FakeRouter(cfg), paths)
        assert only_task(store)["disposition"] == "no-generation"


# --------------------------------------------------------------------------
# The store's claim extension.
# --------------------------------------------------------------------------

def test_judge_claim_takes_only_judging_tasks(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg, n=2) as store:
        plan_wave(store, cfg, "synthesis", 4, task_type_mix={"statute_qa": 1.0})
        claimed = store.claim_tasks(
            "judge-1", 10, state_from=JUDGE_STATE_FROM, state_to=JUDGE_STATE_TO
        )
        assert len(claimed) == 2
        assert {row["state"] for row in claimed} == {JUDGE_STATE_TO}
        assert {row["claimed_by"] for row in claimed} == {"judge-1"}


def test_generation_claim_ignores_judging_tasks(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg, n=2) as store:
        assert store.claim_tasks("gen-1", 10) == []


def test_a_live_judge_lease_is_not_re_claimed(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg, n=2) as store:
        first = store.claim_tasks(
            "judge-1", 10, state_from=JUDGE_STATE_FROM, state_to=JUDGE_STATE_TO
        )
        assert len(first) == 2
        assert store.claim_tasks(
            "judge-2", 10, state_from=JUDGE_STATE_FROM, state_to=JUDGE_STATE_TO
        ) == []


def test_an_expired_judge_lease_is_recovered(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg, n=1) as store:
        store.claim_tasks("judge-1", 1, state_from=JUDGE_STATE_FROM, state_to=JUDGE_STATE_TO)
        # Age the lease by hand: the store compares fixed-width timestamps,
        # and on Windows two claims inside one clock tick stamp identically.
        stale = (datetime.now(UTC) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        store.conn.execute("UPDATE task SET claimed_at = ?", (stale,))
        recovered = store.claim_tasks(
            "judge-2", 1, state_from=JUDGE_STATE_FROM, state_to=JUDGE_STATE_TO
        )
        assert [row["claimed_by"] for row in recovered] == ["judge-2"]
        assert [row["state"] for row in recovered] == [JUDGE_STATE_TO]


def test_identical_claim_states_are_refused(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg, n=1) as store:
        with pytest.raises(ValueError, match="distinct states"):
            store.claim_tasks("judge-1", 1, state_from="judging", state_to="judging")


@pytest.mark.parametrize(
    "state", ["rejected", "accepted", "judge_unroutable", "judge_error", "judge_skipped",
              "gen_unroutable"]
)
def test_terminal_and_parked_rows_are_invisible_to_both_queues(tmp_path, cfg, paths, state):
    """Parking only works if it actually stops the claim: a state that a
    worker can still lease is a re-claim loop with a different name, and the
    money is spent before anyone notices."""
    with judged_store(tmp_path, paths, cfg, n=1) as store:
        store.conn.execute("UPDATE task SET state = ?", (state,))
        assert store.claim_tasks("gen-1", 10) == []
        assert store.claim_tasks(
            "judge-1", 10, state_from=JUDGE_STATE_FROM, state_to=JUDGE_STATE_TO
        ) == []


def test_default_claim_behaviour_is_unchanged(tmp_path, cfg):
    with open_store(tmp_path, n_seeds=2) as store:
        plan_wave(store, cfg, "synthesis", 2, task_type_mix={"irac_analysis": 1.0})
        claimed = store.claim_tasks("gen-1", 2)
        assert {row["state"] for row in claimed} == {"generating"}
        assert {row["attempts"] for row in claimed} == {1}


# --------------------------------------------------------------------------
# When the pool runs out (CRITICAL 1: separation + context can empty a role).
# --------------------------------------------------------------------------

def _lengthen(store, task_id, words):
    gen = store.latest_generation(task_id)
    store.conn.execute(
        "UPDATE generation SET think = ? WHERE gen_id = ?", ("word " * words, gen["gen_id"])
    )
    return gen


def _past_every_judge_window(cfg) -> int:
    """`_lengthen` words that overflow EVERY judge in the pool, DERIVED.

    The largest judge window has moved twice - 8,192 when zai-glm-4.7 held a
    seat, 131,072 after it left, and 800,000 since bai/deepseek-v4-flash took
    slot B on 2026-08-27. A pinned word count is how a test about "no eligible
    model at all" quietly becomes a test about a judge that DID fit: the row
    routes, a slot answers, and the assertion moves to the next slot without
    anything going red about the premise.

    `undersized_families` excludes a model when `needed * CONTEXT_SAFETY_MARGIN`
    exceeds its `max_context`, `context_estimate` counts latin text at
    CHARS_PER_TOKEN_LATIN characters per token, and one `_lengthen` word is the
    5 characters of "word ". Halved margin on top, because the estimate is over
    characters and the exact tokenisation is not the point here.
    """
    from tuned.data.providers import CHARS_PER_TOKEN_LATIN, CONTEXT_SAFETY_MARGIN

    widest = max(
        m.limits.get("max_context") or 0
        for p in cfg.providers
        for m in p.models
        if "judge" in m.roles
    )
    tokens = widest / CONTEXT_SAFETY_MARGIN
    return int(tokens * CHARS_PER_TOKEN_LATIN / 5 * 1.5)


def test_an_unroutable_judge_parks_instead_of_re_queueing(tmp_path, cfg, paths):
    """Past every judge's context window there is no eligible model at all.
    Re-queueing would re-pay whichever slots did answer and meet the same
    wall tomorrow, so the row parks with a diagnostic."""
    with judged_store(tmp_path, paths, cfg) as store:
        _lengthen(store, only_task(store)["task_id"], _past_every_judge_window(cfg))
        router = FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]})
        totals = run_judge(store, cfg, router, paths)
        task = only_task(store)
        assert task["state"] == "judge_unroutable"
        assert task["disposition"].startswith("judge-a-unroutable:")
        assert totals["decided"] == 0
        # One routing attempt, no reply, nothing paid for.
        assert len(router.calls_for("judge")) == 1
        assert router.calls_for("judge")[0]["ref"] is None
        event = json.loads(store.events("judge_route_error")[0]["detail_json"])
        assert event["unroutable"] is True
        assert {"gpt-oss", "qwen", "gemma"} <= set(event["excluded"])
        # And it stays parked: a second sweep does not re-claim it.
        assert run_judge(store, cfg, router, paths)["claimed"] == 0


def test_a_keyless_judge_pool_re_queues_rather_than_parking(tmp_path, cfg, paths):
    """The same distinction the generator now makes, on the judge side: a
    missing key is a fact about the FLEET, so the row goes back to the queue
    it came from rather than parking as if nothing could ever judge it."""
    with judged_store(tmp_path, paths, cfg) as store:
        # Every provider, derived: one left keyed is one judge still routable,
        # and the row then gets judged instead of re-queued.
        router = FakeRouter(cfg, missing_keys={p.name for p in cfg.providers})
        totals = run_judge(store, cfg, router, paths)
        assert only_task(store)["state"] == JUDGE_STATE_FROM
        assert totals["unroutable"] == 0
        event = json.loads(store.events("judge_route_error")[0]["detail_json"])
        assert event["unroutable"] is False
        # "family-excluded" rides along because the judge pool now contains the
        # generator's OWN family (the openai backstop is lumped into gpt-oss),
        # and eligible_refs tests family before key. That makes this the harder
        # case rather than a weaker one: the set CONTAINS a row-shaped reason
        # and still must not read as row-shaped, because
        # `skips <= ROW_SHAPED_SKIPS` is a subset test and a missing key is not
        # in it. Read as row-shaped, this row would park instead of re-queue.
        assert event["skipped"] == ["family-excluded", "missing-key"]


def test_a_recorded_slot_is_never_bought_twice(tmp_path, cfg, paths):
    """Judge A answers, judge B garbles, the task goes back to the queue.
    The retry must re-buy B only - A's judgement is already in the table."""
    with judged_store(tmp_path, paths, cfg) as store:
        first = FakeRouter(
            cfg,
            {
                "judge": [
                    judge_reply(5, 5, 5, "a says fine"),
                    chat_response("I refuse to grade this.", None),
                ]
            },
        )
        run_judge(store, cfg, first, paths)
        gen = store.latest_generation(only_task(store)["task_id"])
        assert [j["judge_slot"] for j in store.judgements_for(gen["gen_id"])] == ["a"]
        assert only_task(store)["state"] == JUDGE_STATE_FROM
        assert len(first.calls_for("judge")) == 3  # a once, b twice (parse retry)

        second = FakeRouter(cfg, {"judge": [judge_reply(4, 4, 4, "b agrees")]})
        run_judge(store, cfg, second, paths)
        # Exactly one call this pass: slot b. Slot a came from the table.
        assert len(second.calls_for("judge")) == 1
        reuse = json.loads(store.events("judge_slot_reused")[0]["detail_json"])
        assert reuse["slot"] == "a"
        # Family separation survives the reuse: b still excludes a's family.
        assert {"gpt-oss", "qwen"} <= second.calls_for("judge")[0]["exclude_families"]
        assert only_task(store)["state"] == "accepted"
        judgements = store.judgements_for(gen["gen_id"])
        assert [j["judge_slot"] for j in judgements] == ["a", "b"]
        assert judgements[0]["rationale"] == "a says fine"


def _narrow_generator(cfg):
    """Every generator family cut back to the window the pilot ran against.

    Needed since 2026-08-19: the probes put cerebras at 131k, so LONG_SEED_TEXT
    no longer diverts anywhere and the judge-side paths only a diverted row
    reaches were never entered. cfg_with_context refuses if the (family, role)
    is missing, so this cannot silently no-op.

    BOTH gpt-oss AND deepseek are narrowed since 2026-08-25: bai (family
    deepseek) joined routing.generator with an 800,000-token window, and
    leaving it alone would make it the family that answers instead of the
    long row diverting to the fixture's own second family.
    """
    narrowed = cfg_with_context(
        cfg, family="gpt-oss", role="generator", max_context=NARROW_GENERATOR_CONTEXT
    )
    return cfg_with_context(
        narrowed, family="deepseek", role="generator", max_context=NARROW_GENERATOR_CONTEXT
    )


def _narrow_judge(cfg):
    """gemma cut back the same way.

    RENAMED FROM _narrow_tiebreak ON 2026-08-19 because it stopped being one:
    gemma took a JUDGE seat that day, and cfg_with_context rewrites the whole
    model, so narrowing "gemma tiebreak" narrows the gemma judge as well. The
    honest name is what it does. Anything that wants a missing TIEBREAK should
    use cfg_without_the_free_tiebreak, which touches routing only.
    """
    return cfg_with_context(cfg, family="gemma", role="judge", max_context=8192)


def test_an_unroutable_tiebreak_decides_on_the_two_judges(tmp_path, cfg, paths):
    """When no third family can take the row the disagreement stands
    unresolved - which is not an accept.

    THE CONDITION IS CONSTRUCTED BY ROUTING, not by a window. It used to be
    the shipped pool's normal behaviour for a long gpt-oss row (gemma pinned at
    8192, removed on length); the probe put gemma at its real 131k, and then
    the 2026-08-19 judge surgery gave the tiebreak seat to
    mistral-large-latest precisely so a gpt-oss row judged by qwen and gemma
    still has a third family. Dropping mistral back out of routing.tiebreak is
    the smallest way to reach the park-loudly path, and it leaves both judge
    slots exactly as they ship. What is under test is that HANDLING, which must
    keep working for any pool that does run out.
    """
    cfg = cfg_without_the_free_tiebreak(cfg)
    with judged_store(tmp_path, paths, cfg) as store:
        _lengthen(store, only_task(store)["task_id"], 5000)
        router = FakeRouter(
            cfg, {"judge": [judge_reply(5, 5, 5), judge_reply(2, 2, 2)]}
        )
        totals = run_judge(store, cfg, router, paths)
        assert totals["tiebreaks"] == 1
        assert totals["rejected"] == 1
        assert router.calls_for("tiebreak")[0]["ref"] is None
        task = only_task(store)
        assert task["state"] == "rejected"
        assert task["disposition"] == "judge:reject-tiebreak-unroutable"
        fallback = json.loads(
            store.events("tiebreak_unroutable_two_judge_decision")[0]["detail_json"]
        )
        assert fallback["verdicts"] == ["pass", "fail"]
        decision = json.loads(store.events("judge_decision")[0]["detail_json"])
        assert decision["tiebreak_unroutable"] is True
        assert decision["action"] == "reject"


def test_slot_b_can_be_unroutable_after_slot_a_has_been_paid_for(tmp_path, cfg, paths):
    """R2-C3, the realistic shape. A long row routes to magistral, family
    separation removes mistral, the 8k glm judge is out on length, slot A
    takes qwen - and slot B has nothing left. The row parks having already
    paid for judge A, which is the money this costs.

    Run against the pool WITHOUT the paid backstop. The shipped pool no longer
    has this hole - closing it is what the openai judges are for - but the
    handling is what is under test here, not the config: a slot B that comes
    back empty must bank judge A and park recoverably, and that has to keep
    working for any pool that runs out. It is also the live behaviour for an
    operator who has not funded OPENAI_API_KEY."""
    # The row under test is LONG, and a long row only reaches a judge if a
    # generator can hold it. mistral was demoted to judge-only on 2026-08-18,
    # so the second generator family is supplied explicitly - otherwise this
    # parks at the generator and never exercises the judge path at all. The
    # cerebras window is narrowed for the same reason in reverse: since the
    # 2026-08-19 probe it holds this row comfortably, so without the narrowing
    # the prompt never diverts and the divert is this test's premise.
    # The JUDGE pool also has to run out, which the shipped one no longer
    # does: gemma took the second judge seat on 2026-08-19, so dropping only
    # the paid backstop still leaves qwen for slot A and gemma for slot B.
    holed = _narrow_generator(
        cfg_without_the_promoted_judge(
            cfg_without_the_paid_judges(cfg_with_two_generator_families(cfg))
        )
    )
    store = open_store(tmp_path, n_seeds=1, text=LONG_SEED_TEXT)
    plan_wave(store, holed, "synthesis", 1, task_type_mix={"irac_analysis": 1.0})
    with store:
        asyncio.run(
            run_workers(
                store, holed, FakeRouter(holed), paths=paths, streams=["synthesis"],
                n_workers=1, max_batches=1,
            )
        )
        gen = store.latest_generation(only_task(store)["task_id"])
        assert gen["model_family"] == "secondgen"  # the long prompt diverted
        _lengthen(store, only_task(store)["task_id"], 2000)

        router = FakeRouter(holed, {"judge": [judge_reply(5, 5, 5, "a says fine")]})
        totals = run_judge(store, holed, router, paths)
        calls = router.calls_for("judge")
        assert len(calls) == 2
        assert calls[0]["ref"].model == "qwen/qwen3.6-27b"  # slot A paid for
        assert calls[1]["ref"] is None  # slot B had no pool at all
        task = only_task(store)
        assert task["state"] == "judge_unroutable"
        assert task["disposition"].startswith("judge-b-unroutable:")
        assert totals["unroutable"] == 1
        # The paid judgement is banked, not thrown away with the row.
        assert [j["judge_slot"] for j in store.judgements_for(gen["gen_id"])] == ["a"]


def test_a_reopened_row_never_re_pays_the_judge_it_already_bought(tmp_path, cfg, paths):
    """The other half of R2-C3: parking is only survivable if something can
    un-park it, and the re-opened row must cost one call, not two.

    On the backstop-less pool, for the same reason as the test above - the
    shipped pool parks nothing here, and what is under test is the recovery,
    which has to work for any pool that ran out."""
    # The row under test is LONG, and a long row only reaches a judge if a
    # generator can hold it. mistral was demoted to judge-only on 2026-08-18,
    # so the second generator family is supplied explicitly - otherwise this
    # parks at the generator and never exercises the judge path at all. The
    # cerebras window is narrowed for the same reason in reverse: since the
    # 2026-08-19 probe it holds this row comfortably, so without the narrowing
    # the prompt never diverts and the divert is this test's premise.
    # The JUDGE pool also has to run out, which the shipped one no longer
    # does: gemma took the second judge seat on 2026-08-19, so dropping only
    # the paid backstop still leaves qwen for slot A and gemma for slot B.
    holed = _narrow_generator(
        cfg_without_the_promoted_judge(
            cfg_without_the_paid_judges(cfg_with_two_generator_families(cfg))
        )
    )
    store = open_store(tmp_path, n_seeds=1, text=LONG_SEED_TEXT)
    plan_wave(store, holed, "synthesis", 1, task_type_mix={"irac_analysis": 1.0})
    with store:
        asyncio.run(
            run_workers(
                store, holed, FakeRouter(holed), paths=paths, streams=["synthesis"],
                n_workers=1, max_batches=1,
            )
        )
        task_id = only_task(store)["task_id"]
        gen = _lengthen(store, task_id, 2000)
        run_judge(store, holed, FakeRouter(holed, {"judge": [judge_reply(5, 5, 5, "a")]}), paths)
        assert only_task(store)["state"] == "judge_unroutable"

        # The operator adds a 32k+ fourth-family judge and re-opens the parked
        # rows. Family separation is untouched: slot B goes to the NEW family,
        # never back to one already used.
        widened = cfg_with_fourth_judge_family(holed)
        assert reopen_tasks(store, ["judge_unroutable"]) == {"judge_unroutable": 1}
        assert only_task(store)["state"] == JUDGE_STATE_FROM

        second = FakeRouter(widened, {"judge": [judge_reply(4, 4, 4, "b agrees")]})
        run_judge(store, widened, second, paths)
        assert len(second.calls_for("judge")) == 1  # slot a came from the table
        assert second.calls_for("judge")[0]["ref"].model == "fourth-judge"
        # Slot B goes to the NEW family, never back to one already used.
        assert {"secondgen", "qwen"} <= second.calls_for("judge")[0]["exclude_families"]
        assert store.events("judge_slot_reused")
        judgements = store.judgements_for(gen["gen_id"])
        assert [j["judge_slot"] for j in judgements] == ["a", "b"]
        assert judgements[0]["rationale"] == "a"
        assert only_task(store)["state"] == "accepted"


def test_a_reused_slot_whose_family_cannot_be_resolved_is_re_bought(tmp_path, cfg, paths):
    """I5: `if o.family` silently drops the constraint when the recorded model
    has left the config, and slot B is then free to be the same family twice.
    Re-buying the slot costs one call; two judges from one family costs the
    invariant."""
    with judged_store(tmp_path, paths, cfg) as store:
        gen = store.latest_generation(only_task(store)["task_id"])
        store.record_judgement(
            gen["gen_id"], "a",
            {"provider": "retired", "model": "gone-v1", "grounding": 5, "validity": 5,
             "coverage": 5, "rationale": "from a model the config no longer knows"},
        )
        router = FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]})
        run_judge(store, cfg, router, paths)
        calls = router.calls_for("judge")
        assert len(calls) == 2  # slot a bought again rather than reused blind
        assert store.events("judge_slot_unresolved")
        families = {call["ref"] for call in calls}
        assert len(families) == 2


def test_a_transient_tiebreak_failure_still_re_queues(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        router = FakeRouter(
            cfg,
            {
                "judge": [judge_reply(5, 5, 5), judge_reply(2, 2, 2)],
                "tiebreak": [chat_response("nope", None)],
            },
        )
        run_judge(store, cfg, router, paths)
        assert only_task(store)["state"] == JUDGE_STATE_FROM
        assert store.events("judge_parse_error")


# --------------------------------------------------------------------------
# The transition stream's judge view (the scenario carries the dates).
# --------------------------------------------------------------------------

def test_the_judge_sees_the_scenario_on_the_transition_stream(tmp_path, cfg, paths):
    store = open_store(tmp_path, n_seeds=1, meta=TRANSITION_META)
    plan_wave(store, cfg, "transition", 1, task_type_mix={"transition": 1.0})
    with store:
        asyncio.run(
            run_workers(
                store, cfg, FakeRouter(cfg), paths=paths, streams=["transition"],
                n_workers=1, max_batches=1,
            )
        )
        assert only_task(store)["state"] == "judging"
        router = FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]})
        run_judge(store, cfg, router, paths, streams=["transition"])
        prompt = router.calls_for("judge")[0]["messages"][-1]["content"]
        # The dates that decide which enactment governs are in front of it.
        assert TRANSITION_META["scenario"] in prompt
        assert "1 July 2024" in prompt
        assert TRANSITION_META["savings_text"] in prompt
        # But the gates' allow-list did not widen: the scenario is not part
        # of what a citation may be checked against.
        gen = store.latest_generation(only_task(store)["task_id"])
        record = read_at(gen["raw_path"], gen["raw_offset"])
        assert TRANSITION_META["scenario"] in record["messages"][-1]["content"]
        assert only_task(store)["state"] == "accepted"


# --------------------------------------------------------------------------
# Loop robustness and lease hygiene.
# --------------------------------------------------------------------------

def test_a_poisoned_task_does_not_kill_the_judge_batch(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg, n=2) as store:
        router = FakeRouter(cfg, {"judge": [RuntimeError("boom"), judge_reply(5, 5, 5)]})
        totals = run_judge(store, cfg, router, paths)
        assert totals["claimed"] == 2
        assert totals["accepted"] == 1
        event = json.loads(store.events("worker_task_error")[0]["detail_json"])
        assert event["error"].startswith("RuntimeError: boom")
        assert store.task_counts()["accepted"] == 1


def test_the_judge_logs_a_lost_lease(tmp_path, cfg, paths):
    """A worker that no longer holds the lease neither spends nor writes: the
    check happens before the first paid call, so the stale worker's verdict
    never exists to be dropped."""
    with judged_store(tmp_path, paths, cfg) as store:
        task = only_task(store)
        # Claim it as somebody else, then hand the stale worker's verdict in.
        store.claim_tasks(
            "other-worker", 1, state_from=JUDGE_STATE_FROM, state_to=JUDGE_STATE_TO
        )
        router = FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]})
        ended = asyncio.run(
            judge_task(
                store, cfg, router, task,
                paths=paths, worker_id="stale-worker",
            )
        )
        assert ended == "lost-lease"
        assert router.calls == []
        assert only_task(store)["state"] == JUDGE_STATE_TO
        assert only_task(store)["claimed_by"] == "other-worker"
        gen = store.latest_generation(task["task_id"])
        assert store.judgements_for(gen["gen_id"]) == []
        lost = json.loads(store.events("lost_lease")[0]["detail_json"])
        assert lost["worker"] == "stale-worker"
        # Site 1 by name: the check at the TOP of judge_task. Without it the
        # per-slot check catches the same worker one window later, which is a
        # different (and dearer) fact, so the site has to be pinned.
        assert lost["wanted_state"] == "judge"


def test_a_stale_workers_judgement_cannot_overwrite_the_live_decision(tmp_path, cfg, paths):
    """R2-C4. Worker A buys slot a and stalls on slot b; its lease expires,
    worker B claims the row, reuses a, buys its own b and accepts. A's late
    reply must not land on top of the scores that decision was made on -
    otherwise the row reads `accepted` on a judgement nobody decided with,
    which is exactly what P5 calibration and gold labelling read."""
    with judged_store(tmp_path, paths, cfg) as store:
        task = only_task(store)
        gen = store.latest_generation(task["task_id"])
        store.claim_tasks("worker-a", 1, state_from=JUDGE_STATE_FROM, state_to=JUDGE_STATE_TO)
        # Worker A's slot a lands while it still holds the lease.
        assert store.record_judgement(
            gen["gen_id"], "a",
            {"provider": "groq", "model": "qwen/qwen3.6-27b", "grounding": 5,
             "validity": 5, "coverage": 5, "rationale": "a says fine"},
            expect_worker="worker-a",
        ) is True

        # The lease expires and worker B takes the row, reuses a and decides.
        stale = (datetime.now(UTC) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        store.conn.execute("UPDATE task SET claimed_at = ?", (stale,))
        claimed = store.claim_tasks(
            "worker-b", 1, state_from=JUDGE_STATE_FROM, state_to=JUDGE_STATE_TO
        )
        router = FakeRouter(cfg, {"judge": [judge_reply(4, 4, 4, "b agrees")]})
        asyncio.run(
            judge_task(store, cfg, router, claimed[0], paths=paths, worker_id="worker-b")
        )
        assert only_task(store)["state"] == "accepted"
        decided = {j["judge_slot"]: j["grounding"] for j in store.judgements_for(gen["gen_id"])}
        assert decided == {"a": 5, "b": 4}

        # ...and now worker A wakes up with its own slot b.
        assert store.record_judgement(
            gen["gen_id"], "b",
            {"provider": "cerebras", "model": "zai-glm-4.7", "grounding": 1,
             "validity": 1, "coverage": 1, "rationale": "a's stalled reply"},
            expect_worker="worker-a",
        ) is False
        after = {j["judge_slot"]: j["grounding"] for j in store.judgements_for(gen["gen_id"])}
        assert after == decided


def test_a_lost_lease_stops_the_decision_being_logged_twice(tmp_path, cfg, paths):
    """Both slots already recorded, so nothing is bought and the lease check
    on the paid path never runs - the decision itself has to be fenced too,
    or a stale worker logs a second judge_decision and re-counts it."""
    with judged_store(tmp_path, paths, cfg) as store:
        task = only_task(store)
        gen = store.latest_generation(task["task_id"])
        for slot, provider, model in (
            ("a", "groq", "qwen/qwen3.6-27b"),
            ("b", "groq", "qwen/qwen3.6-27b"),
        ):
            store.record_judgement(
                gen["gen_id"], slot,
                {"provider": provider, "model": model, "grounding": 5, "validity": 5,
                 "coverage": 5, "rationale": "fine"},
            )
        store.claim_tasks("live-worker", 1, state_from=JUDGE_STATE_FROM, state_to=JUDGE_STATE_TO)
        stats = JudgeStats()
        ended = asyncio.run(
            judge_task(
                store, cfg, FakeRouter(cfg), task,
                paths=paths, worker_id="stale-worker", stats=stats,
            )
        )
        assert ended == "lost-lease"
        assert store.events("judge_decision") == []
        assert stats.decided == 0
        assert stats.accepted == 0
        assert only_task(store)["state"] == JUDGE_STATE_TO


# --------------------------------------------------------------------------
# The C4 fence, per site. Deleting any ONE of the four checks used to leave
# the suite green: `test_a_stale_workers_judgement_cannot_overwrite_the_live
# _decision` calls store.record_judgement directly and never drives
# judge_slot, so what the suite pinned was "some check exists somewhere".
# Each test below steals the lease in ONE window and names the site that has
# to catch it.
# --------------------------------------------------------------------------

def _claimed(store, worker="worker-a"):
    return store.claim_tasks(worker, 1, state_from=JUDGE_STATE_FROM, state_to=JUDGE_STATE_TO)[0]


def _lost_lease_event(store):
    return json.loads(store.events("lost_lease")[-1]["detail_json"])


def test_a_lease_lost_mid_call_costs_the_judgement_not_the_row(tmp_path, cfg, paths):
    """The lease moves while judge A's reply is in flight - the one window
    holds_lease cannot close. The reply is durable in the raw log, but the
    judgement row belongs to the live holder's pass, so the fenced write must
    refuse it and this pass must end without deciding."""
    with judged_store(tmp_path, paths, cfg) as store:
        task = _claimed(store)
        gen = store.latest_generation(task["task_id"])

        def steal_then_answer(ref, messages):
            store.conn.execute("UPDATE task SET claimed_by = 'thief-worker'")
            return judge_reply(5, 5, 5, "scored for somebody else")

        router = FakeRouter(cfg, {"judge": [steal_then_answer]})
        stats = JudgeStats()
        ended = asyncio.run(
            judge_task(store, cfg, router, task, paths=paths, worker_id="worker-a", stats=stats)
        )

        assert ended == "lost-lease"
        assert store.judgements_for(gen["gen_id"]) == []
        # Exactly one paid call: the fence stops the pass, it does not let it
        # run on and buy slot b as well.
        assert len(router.calls_for("judge")) == 1
        assert store.events("judge_decision") == []
        assert _lost_lease_event(store)["wanted_state"] == "judgement:a"
        assert stats.lost_leases == 1
        # ...and the scores are still on disk, under the slot they were for.
        raw = str(next((paths.root / "raw" / "judge").rglob("judge.ndjson")))
        assert "scored for somebody else" in open(raw, encoding="utf-8").read()


def test_the_lease_is_checked_before_the_first_slot_is_bought(tmp_path, cfg, paths):
    """Site 2: the per-slot check. The lease is live when judge_task starts
    and gone by the time the slot loop reaches it, so only the check inside
    the loop can stop this - and stopping it is the difference between a
    wasted call and no call."""
    with judged_store(tmp_path, paths, cfg) as store:
        task = _claimed(store)
        proxy = StealsTheLease(store, at="judgements_for")
        router = FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]})
        stats = JudgeStats()

        ended = asyncio.run(
            judge_task(proxy, cfg, router, task, paths=paths, worker_id="worker-a", stats=stats)
        )

        assert (ended, proxy.stolen) == ("lost-lease", True)
        assert router.calls == []
        assert _lost_lease_event(store)["wanted_state"] == "judge-slot-a"
        assert stats.lost_leases == 1


def test_the_lease_is_checked_before_the_tiebreak_is_bought(tmp_path, cfg, paths):
    """Site 3: the tiebreak check. Both judge slots are reused from the table,
    so the per-slot check never runs at all and the tiebreak is the first
    thing this pass would spend money on."""
    with judged_store(tmp_path, paths, cfg) as store:
        task = _claimed(store)
        gen = store.latest_generation(task["task_id"])
        for slot, provider, model, score in (
            ("a", "groq", "qwen/qwen3.6-27b", 5),
            ("b", "groq", "qwen/qwen3.6-27b", 1),
        ):
            store.record_judgement(
                gen["gen_id"], slot,
                {"provider": provider, "model": model, "grounding": score,
                 "validity": score, "coverage": score, "rationale": "recorded"},
            )
        proxy = StealsTheLease(store, at="judgements_for")
        router = FakeRouter(cfg, {"tiebreak": [judge_reply(5, 5, 5)]})
        stats = JudgeStats()

        ended = asyncio.run(
            judge_task(proxy, cfg, router, task, paths=paths, worker_id="worker-a", stats=stats)
        )

        assert (ended, proxy.stolen) == ("lost-lease", True)
        assert router.calls == []  # one pass/one fail is a tiebreak, and it was not bought
        assert _lost_lease_event(store)["wanted_state"] == "judge-tiebreak"
        assert stats.lost_leases == 1


def test_a_lease_lost_after_the_decision_is_logged_is_not_counted(tmp_path, cfg, paths):
    """Site 4b: the accept write itself. The decision check passed, so the
    event is logged - and then the fence refuses the state write. What must
    NOT happen is the batch line reporting a decision this worker did not
    make: the counters follow the write, never the intention."""
    with judged_store(tmp_path, paths, cfg) as store:
        task = _claimed(store)
        gen = store.latest_generation(task["task_id"])
        for slot, provider, model in (
            ("a", "groq", "qwen/qwen3.6-27b"),
            ("b", "groq", "qwen/qwen3.6-27b"),
        ):
            store.record_judgement(
                gen["gen_id"], slot,
                {"provider": provider, "model": model, "grounding": 5, "validity": 5,
                 "coverage": 5, "rationale": "fine"},
            )
        # The reviewer's window, exactly: the re-claim lands INSIDE log_event,
        # which is after the decision fence and before the state write. No
        # holds_lease check can close it - only the fence on the write itself.
        proxy = StealsTheLease(
            store, at="log_event", when=lambda kind, *a, **k: kind == "judge_decision"
        )
        stats = JudgeStats()

        ended = asyncio.run(
            judge_task(proxy, cfg, FakeRouter(cfg), task, paths=paths,
                       worker_id="worker-a", stats=stats)
        )

        assert (ended, proxy.stolen) == ("lost-lease", True)
        assert only_task(store)["state"] == JUDGE_STATE_TO
        assert only_task(store)["claimed_by"] == "thief-worker"
        # The event is there (it was logged before the fence refused) - the
        # COUNTERS are what must not claim it.
        assert len(store.events("judge_decision")) == 1
        assert (stats.decided, stats.accepted, stats.lost_leases) == (0, 0, 1)
        assert stats.outcomes == {}


def test_the_parse_retry_re_checks_the_lease_before_paying_again(tmp_path, cfg, paths):
    """An unparsable reply buys one more call. That call is a purchase like
    any other and gets the same check: the fenced write makes a stale
    worker's SCORES harmless but cannot make its CALLS free."""
    with judged_store(tmp_path, paths, cfg) as store:
        task = _claimed(store)

        def garble_and_steal(ref, messages):
            store.conn.execute("UPDATE task SET claimed_by = 'thief-worker'")
            return chat_response("I will not grade this.", None)

        router = FakeRouter(cfg, {"judge": [garble_and_steal, judge_reply(5, 5, 5)]})
        stats = JudgeStats()
        ended = asyncio.run(
            judge_task(store, cfg, router, task, paths=paths, worker_id="worker-a", stats=stats)
        )

        assert ended == "lost-lease"
        assert len(router.calls_for("judge")) == 1
        assert _lost_lease_event(store)["wanted_state"] == "judge-retry:a"
        assert stats.lost_leases == 1


def test_the_lease_is_checked_before_the_decision_is_logged(tmp_path, cfg, paths):
    """Site 4a. Both slots are reused, so nothing is bought and the per-slot
    check never runs; the pass would otherwise log a second judge_decision for
    a row somebody else already decided. The existing lost-lease test cannot
    reach this site - it starts with the lease already gone, so the check at
    the top of judge_task returns first."""
    with judged_store(tmp_path, paths, cfg) as store:
        task = _claimed(store)
        gen = store.latest_generation(task["task_id"])
        for slot, provider, model in (
            ("a", "groq", "qwen/qwen3.6-27b"),
            ("b", "groq", "qwen/qwen3.6-27b"),
        ):
            store.record_judgement(
                gen["gen_id"], slot,
                {"provider": provider, "model": model, "grounding": 5, "validity": 5,
                 "coverage": 5, "rationale": "fine"},
            )
        proxy = StealsTheLease(store, at="judgements_for")
        stats = JudgeStats()

        ended = asyncio.run(
            judge_task(proxy, cfg, FakeRouter(cfg), task, paths=paths,
                       worker_id="worker-a", stats=stats)
        )

        assert (ended, proxy.stolen) == ("lost-lease", True)
        assert store.events("judge_decision") == []
        assert _lost_lease_event(store)["wanted_state"] == "judge-decision"
        assert (stats.decided, stats.accepted, stats.lost_leases) == (0, 0, 1)


def test_a_park_the_fence_refused_is_neither_written_nor_counted(tmp_path, cfg, paths):
    """_park's fence-rejected branch. The judge test rewrite left it
    unreached on this side: a stale worker must not park a row the live
    holder is working on, and must not report a park it did not make."""
    with judged_store(tmp_path, paths, cfg) as store:
        task = _claimed(store)
        gen = store.latest_generation(task["task_id"])
        store.conn.execute("UPDATE generation SET think = '' WHERE gen_id = ?", (gen["gen_id"],))
        # The lease moves between the top-of-pass check and the empty-think park.
        proxy = StealsTheLease(store, at="latest_generation")
        stats = JudgeStats()

        ended = asyncio.run(
            judge_task(proxy, cfg, FakeRouter(cfg), task, paths=paths,
                       worker_id="worker-a", stats=stats)
        )

        assert (ended, proxy.stolen) == ("lost-lease", True)
        assert only_task(store)["state"] == JUDGE_STATE_TO  # not judge_skipped
        assert (stats.skipped, stats.lost_leases) == (0, 1)
        # ...and no `judge_parked` event either. It was logged BEFORE the
        # fence, so a stale worker announced a park that never happened - in
        # the log P5 calibration reads, and with no state change to contradict
        # it. Same shape as the second `judge_decision`.
        assert store.events("judge_parked") == []


def test_a_park_that_landed_is_announced(tmp_path, cfg, paths):
    """The other side of moving that log after the fence: a park that DID
    happen still says so, with the state and the reason on it."""
    with judged_store(tmp_path, paths, cfg) as store:
        gen = store.latest_generation(only_task(store)["task_id"])
        store.conn.execute("UPDATE generation SET think = '' WHERE gen_id = ?", (gen["gen_id"],))
        run_judge(store, cfg, FakeRouter(cfg), paths)

        assert only_task(store)["state"] == "judge_skipped"
        parked = json.loads(store.events("judge_parked")[-1]["detail_json"])
        assert (parked["state"], parked["reason"]) == ("judge_skipped", "empty-think")


def test_a_reject_the_fence_refused_is_not_counted_as_a_decision(tmp_path, cfg, paths):
    """The fourth fenced write on this side. The lease moves INSIDE
    log_event("judge_decision") - after the probe that guards the decision and
    before the write it guards - so only the write can refuse it, and the
    counters have to follow the write here as they do on the accept path."""
    with judged_store(tmp_path, paths, cfg) as store:
        task = _claimed(store)
        proxy = StealsTheLease(
            store, at="log_event", when=lambda kind, detail: kind == "judge_decision"
        )
        stats = JudgeStats()
        # Both judges fail it: decide() rejects without buying a tiebreak.
        router = FakeRouter(cfg, {"judge": [judge_reply(1, 1, 1)]})

        ended = asyncio.run(
            judge_task(proxy, cfg, router, task, paths=paths, worker_id="worker-a", stats=stats)
        )

        assert (ended, proxy.stolen) == ("lost-lease", True)
        assert only_task(store)["state"] == JUDGE_STATE_TO  # the live holder still has it
        assert (stats.decided, stats.rejected, stats.lost_leases) == (0, 0, 1)
        assert _lost_lease_event(store)["wanted_state"] == "rejected"


def test_a_requeue_the_fence_refused_after_a_tiebreak_failure_is_a_lost_lease(
    tmp_path, cfg, paths
):
    """A transient tiebreak failure hands the task back to the judging queue -
    a fenced write like every other, and the one path where a refused write
    used to be reported as a clean re-queue."""
    transient = ProviderError(
        "role 'tiebreak': all 3 eligible model(s) failed; last: 429",
        status=429, provider="groq", model="openai/gpt-oss-20b", retryable=True,
    )
    with judged_store(tmp_path, paths, cfg) as store:
        task = _claimed(store)
        proxy = StealsTheLease(
            store, at="log_event", when=lambda kind, detail: kind == "judge_route_error"
        )
        stats = JudgeStats()
        router = FakeRouter(
            cfg,
            {"judge": [judge_reply(5, 5, 5), judge_reply(1, 1, 1)], "tiebreak": [transient]},
        )

        ended = asyncio.run(
            judge_task(proxy, cfg, router, task, paths=paths, worker_id="worker-a", stats=stats)
        )

        assert (ended, proxy.stolen) == ("lost-lease", True)
        assert stats.lost_leases == 1
        assert only_task(store)["state"] == JUDGE_STATE_TO  # NOT handed back
        assert _lost_lease_event(store)["wanted_state"] == JUDGE_STATE_FROM


def test_a_recorded_slot_with_no_resolvable_ref_is_never_reused(cfg):
    """The three ways a recorded judgement cannot stand in for a call, at the
    unit. The ref cases collapse into one: an absent ref makes an empty
    ModelRef, which no config resolves."""
    live = {"provider": "groq", "model": "qwen/qwen3.6-27b",
            "grounding": 5, "validity": 4, "coverage": 4, "rationale": "ok"}
    assert _outcome_from_row(cfg, "a", live).family == "qwen"
    assert _outcome_from_row(cfg, "a", {**live, "coverage": None}) is None
    assert _outcome_from_row(cfg, "a", {**live, "model": "retired-model"}) is None
    assert _outcome_from_row(cfg, "a", {k: v for k, v in live.items()
                                        if k not in ("provider", "model")}) is None


def test_a_judgement_row_that_lost_its_model_is_re_bought(tmp_path, cfg, paths):
    """_outcome_from_row returns None when the recorded ref cannot be resolved
    to a family - including when the row carries no ref at all. Reusing it
    would free slot B to be the same family as slot A, which is the invariant
    the separation exists for."""
    with judged_store(tmp_path, paths, cfg) as store:
        task = only_task(store)
        gen = store.latest_generation(task["task_id"])
        store.record_judgement(
            gen["gen_id"], "a",
            {"grounding": 5, "validity": 5, "coverage": 5, "rationale": "no ref recorded"},
        )
        router = FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]})
        run_judge(store, cfg, router, paths)

        assert store.events("judge_slot_unresolved")
        assert store.events("judge_slot_reused") == []
        # Slot a was bought again, so both slots cost a call this pass.
        assert len(router.calls_for("judge")) == 2
        assert only_task(store)["state"] == "accepted"


def test_a_payload_400_at_the_judge_costs_one_call_not_eight(tmp_path, cfg, paths):
    """Retiring `permanent` raised the judge's bound on a payload 400 from one
    paid call to eight: a plain 400 is unroutable=False, so the task
    re-queued and slot A was re-bought on every claim up to
    MAX_JUDGE_ATTEMPTS. For a systemic payload bug that is eight times the
    wave in paid judge calls before anything parks."""
    payload_400 = ProviderError(
        "mistral/mistral-small-latest: HTTP 400 (bad request, aborting the call): "
        '{"error": "unknown parameter"}',
        status=400, provider="mistral", model="mistral-small-latest", retryable=False,
    )
    with judged_store(tmp_path, paths, cfg) as store:
        router = FakeRouter(cfg, {"judge": [payload_400]})
        totals = None
        for _ in range(MAX_JUDGE_ATTEMPTS):
            totals = totals or run_judge(store, cfg, router, paths)
            run_judge(store, cfg, router, paths)

        assert len(router.calls_for("judge")) == 1
        assert totals["judge_errors"] == 1
        task = only_task(store)
        assert task["state"] == "judge_error"
        assert task["disposition"].startswith("judge-a-payload:")
        # ...and it is a park, so --reopen brings it back once the bug is fixed.
        assert reopen_tasks(store, ["judge_error"]) == {"judge_error": 1}


def test_a_payload_400_at_the_tiebreak_costs_one_call_not_seven(tmp_path, cfg, paths):
    """The same bound on the OTHER role, and it is worth more than it looks:
    slots a and b are reused from the judgement table on every re-claim, so a
    tiebreak that re-queues buys nothing else - the whole cost of the extra
    seven claims lands on the tiebreak role, whose model list is not even the
    same one."""
    payload_400 = ProviderError(
        "groq/openai/gpt-oss-20b: HTTP 400 (bad request, aborting the call): "
        '{"error": "unknown parameter"}',
        status=400, provider="groq", model="openai/gpt-oss-20b", retryable=False,
    )
    with judged_store(tmp_path, paths, cfg) as store:
        # One pass, one fail: the disagreement decide() sends to a tiebreak.
        router = FakeRouter(
            cfg,
            {"judge": [judge_reply(5, 5, 5), judge_reply(1, 1, 1)], "tiebreak": [payload_400]},
        )
        for _ in range(MAX_JUDGE_ATTEMPTS):
            run_judge(store, cfg, router, paths)

        assert len(router.calls_for("tiebreak")) == 1
        assert len(router.calls_for("judge")) == 2  # both slots banked on pass 1
        task = only_task(store)
        assert task["state"] == "judge_error"
        assert task["disposition"].startswith("tiebreak:")
        assert reopen_tasks(store, ["judge_error"]) == {"judge_error": 1}


def test_the_tiebreak_parks_at_the_attempt_cap_not_one_claim_past_it(tmp_path, cfg, paths):
    """A TRANSIENT tiebreak failure still spends claims, and the cap is read
    with >=: at exactly MAX_JUDGE_ATTEMPTS the row parks. With > it buys one
    more paid tiebreak per row, wave-wide."""
    transient = ProviderError(
        "role 'tiebreak': all 3 eligible model(s) failed; last: 429",
        status=429, provider="groq", model="openai/gpt-oss-20b", retryable=True,
    )
    with judged_store(tmp_path, paths, cfg) as store:
        router = FakeRouter(
            cfg,
            {"judge": [judge_reply(5, 5, 5), judge_reply(1, 1, 1)], "tiebreak": [transient]},
        )
        # The claim bumps attempts, so the pass that must park is the one that
        # arrives holding exactly the cap.
        store.conn.execute("UPDATE task SET attempts = ?", (MAX_JUDGE_ATTEMPTS - 1,))
        run_judge(store, cfg, router, paths)

        task = only_task(store)
        assert task["attempts"] == MAX_JUDGE_ATTEMPTS
        assert task["state"] == "judge_error"
        assert task["disposition"].startswith("tiebreak:")
        # ...and the parked row is invisible to the next sweep.
        run_judge(store, cfg, router, paths)
        assert len(router.calls_for("tiebreak")) == 1


def test_a_context_overflow_at_the_judge_is_not_logged_as_our_payload_bug(tmp_path, cfg, paths):
    """Overflow at every ref aggregates as status=400, retryable=False,
    context_exceeded=True - which is what `payload_error`'s context clause is
    there for. The row parks as unroutable either way, so the fact this
    protects is the DIAGNOSTIC: `payload_error: true` would file a
    pool-shaped failure as a code defect, in the log P5 calibration reads."""
    overflow = ProviderError(
        "role 'judge': all 3 eligible model(s) failed; last: 400",
        status=400, provider="mistral", model="mistral-small-latest",
        retryable=False, context_exceeded=True,
    )
    with judged_store(tmp_path, paths, cfg) as store:
        run_judge(store, cfg, FakeRouter(cfg, {"judge": [overflow]}), paths)

        event = json.loads(store.events("judge_route_error")[-1]["detail_json"])
        assert (event["unroutable"], event["payload_error"]) == (True, False)
        assert only_task(store)["state"] == "judge_unroutable"


def test_a_transient_judge_failure_still_spends_its_claims(tmp_path, cfg, paths):
    """The narrower bound is for the no-progress payload case only: a 429 is
    the ordinary weather of a free tier and must still be retried."""
    transient = ProviderError(
        "role 'judge': all 3 eligible model(s) failed; last: 429",
        status=429, provider="mistral", model="mistral-small-latest", retryable=True,
    )
    with judged_store(tmp_path, paths, cfg) as store:
        router = FakeRouter(cfg, {"judge": [transient]})
        for _ in range(3):
            run_judge(store, cfg, router, paths)
        assert len(router.calls_for("judge")) == 3
        assert only_task(store)["state"] == JUDGE_STATE_FROM


def test_the_judge_attempt_cap_is_eight_and_the_eighth_claim_is_the_last(tmp_path, cfg, paths):
    """Every exhaustion test in this file loops `for _ in range(
    MAX_JUDGE_ATTEMPTS)` - parameterised by the value under test, so it holds
    at 8 and at 99 alike. The number matters more since --reopen re-arms the
    counter: these caps are now the only bound on what one row can spend
    across a reopen cycle, on a fleet running for days against hard daily
    caps. So: the literal, and a claim past it that does not happen."""
    assert MAX_JUDGE_ATTEMPTS == 8
    transient = ProviderError(
        "role 'judge': all 3 eligible model(s) failed; last: 429",
        status=429, provider="mistral", model="mistral-small-latest", retryable=True,
    )
    with judged_store(tmp_path, paths, cfg) as store:
        router = FakeRouter(cfg, {"judge": [transient]})
        for _ in range(MAX_JUDGE_ATTEMPTS + 3):
            run_judge(store, cfg, router, paths)

        # Eight claims total and then the row is parked and unclaimable - not
        # a ninth, whatever the loop asks for. Seven of them are judge calls
        # because the GENERATION claim spent the first: both caps read the one
        # `attempts` counter (concern 6), which is exactly why the number has
        # to be pinned as a number.
        task = only_task(store)
        assert task["attempts"] == MAX_JUDGE_ATTEMPTS
        assert len(router.calls_for("judge")) == MAX_JUDGE_ATTEMPTS - 1
        assert task["state"] == "judge_error"


def test_the_judge_sizes_the_call_it_is_about_to_buy_the_way_the_preflight_does(
    tmp_path, cfg, paths
):
    """R3-C2's SPENDING half. The ruling was ONE sizer called by both sides;
    the preflight caller has four tests and the spender had none, so reverting
    judge_slot to `sum(len(m["content"]) for m in messages) // 4 + max_tokens`
    left the suite green. On a Devanagari row - which is what this corpus is -
    the two disagree by enough to change the routing decision."""
    # The row under test is LONG, and a long row only reaches a judge if a
    # generator can hold it. mistral was demoted to judge-only on 2026-08-18,
    # so the second generator family is supplied explicitly - otherwise this
    # parks at the generator and never exercises the judge path at all.
    # The 8k judge this row has to be too big for is SUPPLIED rather than
    # borrowed from the pool. zai-glm-4.7 carried that window until 2026-08-18,
    # when it was retired as archived, and every judge left is 32k or larger -
    # so no shipped model now sits between what the two sizings ask for on this
    # row (11,247 tokens of declared window in the routing currency, 6,777 in
    # chars/4), and the disagreement would be unobservable.
    widened = cfg_with_fourth_judge_family(
        cfg_with_two_generator_families(cfg), max_context=8192
    )
    devanagari = "अभियुक्त को भारतीय दंड संहिता की धारा तीन सौ दो के अंतर्गत दोषी ठहराया गया। " * 140
    with open_store(tmp_path, n_seeds=1, text=devanagari) as store:
        plan_wave(store, widened, "synthesis", 1, task_type_mix={"irac_analysis": 1.0})
        asyncio.run(
            run_workers(
                store, widened, FakeRouter(widened), paths=paths,
                streams=["synthesis"], n_workers=1, max_batches=1,
            )
        )
        task = only_task(store)
        gen = store.latest_generation(task["task_id"])
        router = FakeRouter(widened, {"judge": [judge_reply(5, 5, 5)]})
        run_judge(store, widened, router, paths)

        call = router.calls_for("judge")[0]
        source = build_prompt(widened, task, store.get_seed(task["seed_id"])).judge_source
        messages = judge_messages(source, gen["think"], gen["answer"])
        assert call["messages"] == messages
        assert call["est_tokens"] == judge_needed_tokens(messages, reply_tokens=JUDGE_MAX_TOKENS)
        # ...and the number is load-bearing: it is what takes the 8k judge out
        # of the pool for this row. The chars/4 sizing leaves it in.
        assert "fourth" in call["exclude_families"]
        chars_over_four = sum(len(m["content"]) for m in messages) // 4 + JUDGE_MAX_TOKENS
        assert "fourth" not in undersized_families(widened, "judge", chars_over_four)


def test_a_reopened_judge_row_gets_its_attempt_budget_back(tmp_path, cfg, paths):
    """R3-C1 on the judge side: the same cliff at MAX_JUDGE_ATTEMPTS. A
    re-opened row that spends its one remaining claim on a paid retry and then
    re-parks has not been recovered, it has been charged."""
    with judged_store(tmp_path, paths, cfg) as store:
        task_id = only_task(store)["task_id"]
        store.conn.execute(
            "UPDATE task SET attempts = ? WHERE task_id = ?", (MAX_JUDGE_ATTEMPTS, task_id)
        )
        garbled = {"judge": [chat_response("I will not grade this.", None)]}
        run_judge(store, cfg, FakeRouter(cfg, garbled), paths)
        assert only_task(store)["state"] == "judge_error"

        reopen_tasks(store, ["judge_error"])
        assert only_task(store)["attempts"] == 0

        run_judge(store, cfg, FakeRouter(cfg, garbled), paths)
        task = only_task(store)
        assert task["state"] == JUDGE_STATE_FROM
        assert task["attempts"] == 1


def test_a_parked_judge_error_appears_in_the_batch_line(tmp_path, cfg, paths, capsys):
    """_park(counter=None) meant judge_error parks were in no column at all:
    the batch line said claimed=1 and then accounted for nothing."""
    with judged_store(tmp_path, paths, cfg) as store:
        store.conn.execute("UPDATE task SET attempts = ?", (MAX_JUDGE_ATTEMPTS,))
        run_judge(store, cfg, FakeRouter(cfg, {"judge": [chat_response("no.", None)]}), paths)
        line = next(
            ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("judge batch")
        )
        assert "judge-err=1" in line


def test_idle_judge_batches_are_announced_once(tmp_path, cfg, paths, capsys):
    slept: list[float] = []

    async def sleeper(delay):
        slept.append(delay)

    with open_store(tmp_path, n_seeds=0) as store:
        asyncio.run(
            run_judges(
                store, cfg, FakeRouter(cfg), paths=paths, streams=["synthesis"],
                n_workers=1, forever=True, max_batches=3, sleeper=sleeper, idle_sleep_s=0.01,
            )
        )
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("judge batch")]
        assert len(lines) == 1
        assert len(slept) == 3


def test_the_judge_cli_refuses_the_pool_hole_before_claiming(tmp_path, cfg, monkeypatch, capsys):
    """The judge worker is the one that discovers the hole by paying for
    slot A, so it is the one that must not start into it.

    THE HOLE IS NOW KEY-SHAPED ON THE SHIPPED POOL, and this test changed with
    the pool rather than around it. Until 2026-08-19 the judge pool had one
    free family (qwen) plus mistral, so withholding OPENAI_API_KEY emptied slot
    B for a mistral generation and the test needed a second generator family to
    reach it. mistral is gone (calibration: holdout precision 0.237) and gemma
    is promoted, so there are now TWO free judge families and withholding the
    paid backstop alone leaves slot B filled by gemma.

    Withholding GROQ as well is what empties it, and it is reachable from an
    ordinary gpt-oss generation with no second family at all: judge families
    keyed are {gemma}, separation removes nothing (gemma is not gpt-oss), slot
    A takes gemma, and slot B has no family left. Both keys are withheld
    EXPLICITLY, because a machine that exports either turns this into a test of
    nothing.

    THIS TEST HUNG when the pool changed under it, and that is why the
    construction is spelled out. With no hole the CLI stops refusing, proceeds
    past the preflight and starts a real run - which with keys in the
    environment means live network calls out of a unit test.

    RE-BASELINED 2026-08-28: bai/deepseek-v4-flash is now the SOLE
    routing.generator ref (operator directive - deepseek is the sole
    generator, cerebras spends only on judging). Withholding BAI_API_KEY, as
    this test used to, now trips the EARLIER "routing.generator has no
    usable API key" refusal before the CLI ever reaches the judge pool, so
    BAI_API_KEY must be set for the scenario below to still exercise the
    judge-pool hole this test is about. Setting it does not fill the hole:
    with a deepseek generation, family_separation excludes {deepseek} from
    the judge pool, which removes bai/deepseek-v4-flash itself (it cannot
    judge its own row) while leaving gemma untouched (family gemma) - slot A
    still takes gemma, and slot B still has no family left, the same shape
    as before, just reached by generator-self-exclusion instead of a
    gpt-oss family lump.
    """
    monkeypatch.setenv("CEREBRAS_API_KEY", "sk-test")
    monkeypatch.setenv("BAI_API_KEY", "sk-test")
    for env in ("GROQ_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr("tuned.data.providers.load_dotenv_keys", lambda path=None: 0)
    config_path = temp_config(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        judge_main(["--config", config_path, "--max-batches", "1"])
    assert excinfo.value.code == 2
    out = capsys.readouterr().out
    assert "REFUSING" in out and "routing.judge slot b" in out
    assert "--allow-pool-gaps" in out
    assert not (tmp_path / "build" / "state").exists()


def test_judge_batch_line(tmp_path, cfg, paths, capsys):
    with judged_store(tmp_path, paths, cfg) as store:
        run_judge(store, cfg, FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]}), paths)
        out = capsys.readouterr().out
        assert "judge batch 1: claimed=1 decided=1 accepted=1" in out


# --------------------------------------------------------------------------
# Task 4: apply active per-model rules; unfitted seats cannot mint accepts.
# --------------------------------------------------------------------------

def resolve_slot(*args, **kwargs):
    fn = getattr(judge_mod, "resolve_slot", None)
    assert fn is not None, "judge.resolve_slot must apply per-model threshold rules"
    return fn(*args, **kwargs)


def test_no_thresholds_keeps_the_provisional_four_four_four_pass():
    """The shipped 4/2 bands stay when the fleet has no active row at all."""
    four = resolve_slot(scores(4, 4, 4), model=QWEN, rules={}, fleet_active=False)
    thin = resolve_slot(scores(5, 5, 3), model=QWEN, rules={}, fleet_active=False)
    bad = resolve_slot(scores(5, 5, 2), model=QWEN, rules={}, fleet_active=False)
    assert (four.verdict, four.provisional, four.unfitted, four.coerced) == (
        PASS, True, False, False,
    )
    assert four.rule == "min_axis" and four.threshold == 4
    assert thin.verdict == BORDERLINE and thin.provisional is True
    assert bad.verdict == FAIL and bad.provisional is True


_EQUIV_AXES = (
    (1, 1, 1),
    (2, 2, 2),
    (3, 3, 3),
    (4, 4, 4),
    (5, 5, 5),
    (5, 5, 3),
    (5, 3, 3),
    (4, 4, 5),
    (5, 1, 5),
    (2, 5, 5),
    (5, 5, 2),
    (1, 5, 5),
    (4, 3, 5),
    (3, 4, 5),
    (5, 4, 3),
)


@pytest.mark.parametrize("rule", ["min_axis", "mean", "both"])
@pytest.mark.parametrize("threshold", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("axes", _EQUIV_AXES)
def test_judge_pass_predicate_matches_calibrate_decides_pass(rule, threshold, axes):
    """Both implementations must stay one predicate. A drift here ships a
    different accept rule than the one P5 fitted."""
    from tuned.data.calibrate import Candidate, decides_pass

    scored = JudgeScores(grounding=axes[0], validity=axes[1], coverage=axes[2])
    judge_fn = getattr(judge_mod, "rule_passes", None)
    assert judge_fn is not None, "judge.rule_passes must exist to lock against calibrate.decides_pass"
    assert judge_fn(rule, threshold, scored) is decides_pass(Candidate(rule, threshold), axes)
    decision = resolve_slot(
        scored,
        model="locked-model",
        rules={"locked-model": {"rule": rule, "threshold": threshold}},
        fleet_active=True,
    )
    assert (decision.verdict == PASS) is decides_pass(Candidate(rule, threshold), axes)


def test_fitted_min_axis_mean_and_both_rules_change_the_slot_verdict():
    """A (5,5,3) is borderline on min_axis>=4, a pass on mean>=4, and not both."""
    axes = scores(5, 5, 3)
    min_axis = resolve_slot(
        axes, model=QWEN, rules={QWEN: {"rule": "min_axis", "threshold": 4}},
        fleet_active=True,
    )
    mean = resolve_slot(
        axes, model=QWEN, rules={QWEN: {"rule": "mean", "threshold": 4}},
        fleet_active=True,
    )
    both = resolve_slot(
        axes, model=QWEN, rules={QWEN: {"rule": "both", "threshold": 4}},
        fleet_active=True,
    )
    both_clear = resolve_slot(
        scores(4, 4, 5), model=QWEN, rules={QWEN: {"rule": "both", "threshold": 4}},
        fleet_active=True,
    )
    assert (min_axis.verdict, min_axis.fitted, min_axis.provisional) == (BORDERLINE, True, False)
    assert mean.verdict == PASS and mean.rule == "mean" and mean.threshold == 4
    assert both.verdict == BORDERLINE
    assert both_clear.verdict == PASS


def test_exact_model_identity_is_the_only_lookup():
    """A provider-prefixed or family-only key must not steal another model's rule."""
    rules = {"groq/qwen/qwen3.6-27b": {"rule": "min_axis", "threshold": 1}}
    decision = resolve_slot(scores(4, 4, 4), model=QWEN, rules=rules, fleet_active=True)
    assert decision.unfitted is True
    assert decision.verdict == BORDERLINE
    assert decision.coerced is True


def test_an_unfitted_model_cannot_supply_a_pass_once_any_threshold_is_active():
    decision = resolve_slot(scores(5, 5, 5), model=GEMMA, rules={QWEN: {"rule": "min_axis", "threshold": 4}}, fleet_active=True)
    assert decision.unfitted is True
    assert decision.coerced is True
    assert decision.verdict == BORDERLINE
    assert decision.verdict != PASS
    fail = resolve_slot(scores(1, 1, 1), model=GEMMA, rules={QWEN: {"rule": "min_axis", "threshold": 4}}, fleet_active=True)
    assert fail.verdict == FAIL
    assert fail.coerced is False


def test_a_malformed_or_unknown_active_rule_fails_closed():
    unknown = resolve_slot(
        scores(5, 5, 5), model=QWEN,
        rules={QWEN: {"rule": "median", "threshold": 4}}, fleet_active=True,
    )
    bad_threshold = resolve_slot(
        scores(5, 5, 5), model=QWEN,
        rules={QWEN: {"rule": "min_axis", "threshold": 9}}, fleet_active=True,
    )
    hyphen = resolve_slot(
        scores(5, 5, 5), model=QWEN,
        rules={QWEN: {"rule": "min-axis", "threshold": 4}}, fleet_active=True,
    )
    for decision in (unknown, bad_threshold, hyphen):
        assert decision.valid is False
        assert decision.verdict == FAIL
        assert decision.verdict != PASS
        assert decision.reason


def test_a_fitted_mean_rule_accepts_a_row_the_provisional_band_would_regenerate(
    tmp_path, cfg, paths
):
    """(5,5,3) is borderline under min_axis>=4 and a pass under mean>=4."""
    with judged_store(tmp_path, paths, cfg) as store:
        _activate_rules(store, (QWEN, "mean", 4), (GEMMA, "mean", 4))
        totals = run_judge(store, cfg, FakeRouter(cfg, {"judge": [judge_reply(5, 5, 3)]}), paths)
        assert totals["accepted"] == 1
        assert only_task(store)["state"] == "accepted"
        event = json.loads(store.events("judge_decision")[0]["detail_json"])
        assert event["provisional"] is False
        assert event["verdicts"] == ["pass", "pass"]
        assert event["any_unfitted"] is False
        used = {row["model"]: row for row in event["slot_rules"]}
        assert used[QWEN]["rule"] == "mean" and used[QWEN]["threshold"] == 4
        assert used[GEMMA]["rule"] == "mean"


def test_a_stricter_min_axis_stops_a_provisional_four_four_four_accept(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        _activate_rules(store, (QWEN, "min_axis", 5), (GEMMA, "min_axis", 5))
        run_judge(store, cfg, FakeRouter(cfg, {"judge": [judge_reply(4, 4, 4)]}), paths)
        event = json.loads(store.events("judge_decision")[0]["detail_json"])
        assert event["verdicts"] == ["borderline", "borderline"]
        assert event["action"] != "accept"
        assert only_task(store)["state"] != "accepted"


def test_an_unfitted_seat_cannot_mint_an_accept_once_thresholds_exist(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        _activate_rules(store, ("some-other-judge", "min_axis", 4))
        router = FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)], "generator": [chat_response()]})
        run_judge(store, cfg, router, paths)
        first = json.loads(store.events("judge_decision")[0]["detail_json"])
        assert first["verdicts"] == ["borderline", "borderline"]
        assert first["action"] == "regenerate"
        assert first["any_unfitted"] is True
        assert first["provisional"] is False
        assert store.events("judge_unfitted_coerced")
        assert only_task(store)["state"] != "accepted"
        assert only_task(store)["disposition"] != "judge:accept"


def test_mixed_fitted_and_unfitted_slots_cannot_accept_through_tiebreak(tmp_path, cfg, paths):
    """Only gemma is fitted. A 5/5/5 from qwen must not pair with it into an accept."""
    with judged_store(tmp_path, paths, cfg) as store:
        _activate_rules(store, (GEMMA, "min_axis", 4))
        router = FakeRouter(
            cfg,
            {
                "judge": [judge_reply(5, 5, 5)],
                "tiebreak": [judge_reply(5, 5, 5)],
                "generator": [chat_response()],
            },
        )
        run_judge(store, cfg, router, paths)
        first = json.loads(store.events("judge_decision")[0]["detail_json"])
        assert first["verdicts"][0] == "borderline"  # qwen unfitted
        assert first["verdicts"][1] == "pass"        # gemma fitted
        assert first["any_unfitted"] is True
        assert first["action"] in {"tiebreak", "regenerate", "reject"}
        assert only_task(store)["state"] != "accepted"
        assert only_task(store)["disposition"] != "judge:accept"
        coerced = [json.loads(e["detail_json"]) for e in store.events("judge_unfitted_coerced")]
        assert any(row["model"] == QWEN for row in coerced)


def test_a_malformed_active_rule_rejects_closed_and_names_the_reason(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        store.conn.execute(
            "INSERT INTO judge_threshold (calib_id, model, rule, threshold, active) "
            "VALUES ('bad-qwen', ?, 'median', 4, 1), ('bad-gemma', ?, 'median', 4, 1)",
            (QWEN, GEMMA),
        )
        run_judge(store, cfg, FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]}), paths)
        task = only_task(store)
        assert task["state"] == "rejected"
        assert task["disposition"] == "judge:reject"
        events = [json.loads(e["detail_json"]) for e in store.events("judge_threshold_invalid")]
        assert events
        assert any("median" in str(row.get("reason")) for row in events)
        decision = json.loads(store.events("judge_decision")[0]["detail_json"])
        assert decision["verdicts"] == ["fail", "fail"]
        assert all(row["valid"] is False for row in decision["slot_rules"])


def test_a_cooling_judge_pool_parks_without_a_quality_reject(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        cooling = {f"{ref.provider}/{ref.model}" for ref in cfg.routing_refs("judge")}
        router = FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]}, cooling=cooling)
        totals = run_judge(store, cfg, router, paths)
        task = only_task(store)
        assert task["state"] != "rejected"
        assert task["disposition"] != "judge:reject"
        assert totals["rejected"] == 0
        assert task["state"] == JUDGE_STATE_FROM


def test_an_unroutable_judge_is_still_not_a_quality_reject(tmp_path, cfg, paths):
    """Routing emptiness stays an operational park after thresholds exist."""
    with judged_store(tmp_path, paths, cfg) as store:
        _activate_rules(store, (QWEN, "min_axis", 4), (GEMMA, "min_axis", 4))
        _lengthen(store, only_task(store)["task_id"], _past_every_judge_window(cfg))
        run_judge(store, cfg, FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]}), paths)
        task = only_task(store)
        assert task["state"] == "judge_unroutable"
        assert task["disposition"] != "judge:reject"
        assert not str(task["disposition"]).startswith("judge:reject")


def test_ground_faithfulness_is_read_as_the_grounding_axis():
    """One exp_harmony reply used ground_faithfulness and was discarded.

    The reply was complete and well formed; only the axis key was one
    character off the rubric's own name. Throwing away a paid verdict over
    that is the parser being brittle, not defensive.
    """
    reply = (
        '{"ground_faithfulness": 4, "reasoning_validity": 3, '
        '"issue_coverage": 5, "rationale": "sound"}'
    )
    scores = parse_judge_reply(reply)
    assert (scores.grounding, scores.validity, scores.coverage) == (4, 3, 5)
    assert scores.rationale == "sound"
