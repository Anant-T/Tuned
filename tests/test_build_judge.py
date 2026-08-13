import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from pipeline_fakes import (
    LONG_SEED_TEXT,
    TRANSITION_META,
    FakeRouter,
    StealsTheLease,
    build_cfg,
    cfg_with_fourth_judge_family,
    chat_response,
    judge_reply,
    open_store,
    paths_for,
    temp_config,
)

from tuned.data.generate import build_prompt, judge_messages, judge_needed_tokens, run_workers
from tuned.data.jsonl import read_at
from tuned.data.providers import ProviderError
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
    """A store whose n tasks have been generated and are waiting in 'judging'."""
    store = open_store(tmp_path, n_seeds=n)
    plan_wave(store, cfg, "synthesis", n, task_type_mix={"irac_analysis": 1.0})
    asyncio.run(
        run_workers(
            store, cfg, FakeRouter(cfg, generator_script), paths=paths,
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

def test_undersized_families_excludes_the_8k_judges(cfg):
    assert undersized_families(cfg, "judge", 4000) == frozenset()
    assert undersized_families(cfg, "judge", 20000) == frozenset({"glm"})
    # Past 32k only the 131k judge survives.
    assert undersized_families(cfg, "judge", 40000) == frozenset({"glm", "mistral"})


def test_generation_family_falls_back_to_the_config(cfg):
    assert generation_family(cfg, {"model_family": "gpt-oss"}) == "gpt-oss"
    assert generation_family(
        cfg, {"model_family": None, "provider": "mistral", "model": "magistral-small-latest"}
    ) == "mistral"
    assert generation_family(cfg, {"provider": "nope", "model": "nope"}) is None


def test_judges_exclude_the_generator_family_and_each_other(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        router = FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]})
        run_judge(store, cfg, router, paths)
        calls = router.calls_for("judge")
        assert len(calls) == 2
        # The generator was cerebras/gpt-oss-120b (family gpt-oss).
        assert "gpt-oss" in calls[0]["exclude_families"]
        assert calls[0]["ref"].provider == "mistral"
        # Judge B also excludes judge A's family, so it is a different model.
        assert {"gpt-oss", "mistral"} <= calls[1]["exclude_families"]
        assert calls[1]["ref"].model == "qwen/qwen3.6-27b"


def test_a_long_candidate_is_routed_past_the_8k_judges(tmp_path, cfg, paths):
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
        assert "glm" in calls[0]["exclude_families"]
        assert {"glm", "gpt-oss"} <= calls[1]["exclude_families"]
        assert all(call["ref"].model != "zai-glm-4.7" for call in calls)


def test_a_candidate_past_every_small_judge_lands_on_the_131k_model(tmp_path, cfg, paths):
    with judged_store(tmp_path, paths, cfg) as store:
        gen = store.latest_generation(only_task(store)["task_id"])
        store.conn.execute(
            "UPDATE generation SET think = ? WHERE gen_id = ?",
            ("word " * 40000, gen["gen_id"]),
        )
        router = FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]})
        run_judge(store, cfg, router, paths)
        calls = router.calls_for("judge")
        assert {"glm", "mistral"} <= calls[0]["exclude_families"]
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
        assert judgements[0]["provider"] == "mistral"
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
        assert {"gpt-oss", "mistral", "qwen"} <= tiebreak["exclude_families"]
        assert tiebreak["ref"].model == "gemma-4-31b"
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
        store.conn.execute(
            "INSERT INTO judge_threshold (calib_id, judge_slot, rule, threshold, active) "
            "VALUES ('c1', 'a', 'min-axis', 4, 1)"
        )
        assert thresholds_active(store) == 1
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


def test_an_unroutable_judge_parks_instead_of_re_queueing(tmp_path, cfg, paths):
    """Past every judge's context window there is no eligible model at all.
    Re-queueing would re-pay whichever slots did answer and meet the same
    wall tomorrow, so the row parks with a diagnostic."""
    with judged_store(tmp_path, paths, cfg) as store:
        _lengthen(store, only_task(store)["task_id"], 200000)
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
        assert {"glm", "mistral", "qwen"} <= set(event["excluded"])
        # And it stays parked: a second sweep does not re-claim it.
        assert run_judge(store, cfg, router, paths)["claimed"] == 0


def test_a_keyless_judge_pool_re_queues_rather_than_parking(tmp_path, cfg, paths):
    """The same distinction the generator now makes, on the judge side: a
    missing key is a fact about the FLEET, so the row goes back to the queue
    it came from rather than parking as if nothing could ever judge it."""
    with judged_store(tmp_path, paths, cfg) as store:
        router = FakeRouter(cfg, missing_keys={"mistral", "groq", "cerebras"})
        totals = run_judge(store, cfg, router, paths)
        assert only_task(store)["state"] == JUDGE_STATE_FROM
        assert totals["unroutable"] == 0
        event = json.loads(store.events("judge_route_error")[0]["detail_json"])
        assert event["unroutable"] is False
        assert event["skipped"] == ["missing-key"]


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
        assert {"gpt-oss", "mistral"} <= second.calls_for("judge")[0]["exclude_families"]
        assert only_task(store)["state"] == "accepted"
        judgements = store.judgements_for(gen["gen_id"])
        assert [j["judge_slot"] for j in judgements] == ["a", "b"]
        assert judgements[0]["rationale"] == "a says fine"


def test_an_unroutable_tiebreak_decides_on_the_two_judges(tmp_path, cfg, paths):
    """The shipped tiebreak pool is gpt-oss + two 8k models, so a long
    candidate from the gpt-oss generator has no third family. The
    disagreement then stands unresolved - which is not an accept."""
    with judged_store(tmp_path, paths, cfg) as store:
        _lengthen(store, only_task(store)["task_id"], 5000)  # ~6k tokens: 8k judges out
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
    paid for judge A, which is the money this costs."""
    store = open_store(tmp_path, n_seeds=1, text=LONG_SEED_TEXT)
    plan_wave(store, cfg, "synthesis", 1, task_type_mix={"irac_analysis": 1.0})
    with store:
        asyncio.run(
            run_workers(
                store, cfg, FakeRouter(cfg), paths=paths, streams=["synthesis"],
                n_workers=1, max_batches=1,
            )
        )
        gen = store.latest_generation(only_task(store)["task_id"])
        assert gen["model_family"] == "mistral"  # the long prompt diverted
        _lengthen(store, only_task(store)["task_id"], 2000)

        router = FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5, "a says fine")]})
        totals = run_judge(store, cfg, router, paths)
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
    un-park it, and the re-opened row must cost one call, not two."""
    store = open_store(tmp_path, n_seeds=1, text=LONG_SEED_TEXT)
    plan_wave(store, cfg, "synthesis", 1, task_type_mix={"irac_analysis": 1.0})
    with store:
        asyncio.run(
            run_workers(
                store, cfg, FakeRouter(cfg), paths=paths, streams=["synthesis"],
                n_workers=1, max_batches=1,
            )
        )
        task_id = only_task(store)["task_id"]
        gen = _lengthen(store, task_id, 2000)
        run_judge(store, cfg, FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5, "a")]}), paths)
        assert only_task(store)["state"] == "judge_unroutable"

        # The operator adds the 32k+ fourth-family judge the config is missing
        # and re-opens the parked rows. Family separation is untouched: slot B
        # goes to the NEW family, never back to one already used.
        widened = cfg_with_fourth_judge_family(cfg)
        assert reopen_tasks(store, ["judge_unroutable"]) == {"judge_unroutable": 1}
        assert only_task(store)["state"] == JUDGE_STATE_FROM

        second = FakeRouter(widened, {"judge": [judge_reply(4, 4, 4, "b agrees")]})
        run_judge(store, widened, second, paths)
        assert len(second.calls_for("judge")) == 1  # slot a came from the table
        assert second.calls_for("judge")[0]["ref"].model == "fourth-judge"
        assert {"mistral", "qwen"} <= second.calls_for("judge")[0]["exclude_families"]
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
            {"provider": "mistral", "model": "mistral-small-latest", "grounding": 5,
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
            ("a", "mistral", "mistral-small-latest"),
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
            ("a", "mistral", "mistral-small-latest", 5),
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
            ("a", "mistral", "mistral-small-latest"),
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
            ("a", "mistral", "mistral-small-latest"),
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
    live = {"provider": "mistral", "model": "mistral-small-latest",
            "grounding": 5, "validity": 4, "coverage": 4, "rationale": "ok"}
    assert _outcome_from_row(cfg, "a", live).family == "mistral"
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
    widened = cfg_with_fourth_judge_family(cfg)
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
        assert "glm" in call["exclude_families"]
        chars_over_four = sum(len(m["content"]) for m in messages) // 4 + JUDGE_MAX_TOKENS
        assert "glm" not in undersized_families(widened, "judge", chars_over_four)


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
    slot A, so it is the one that must not start into it."""
    for env in ("CEREBRAS_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(env, "sk-test")
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
