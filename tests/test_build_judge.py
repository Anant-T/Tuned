import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from pipeline_fakes import (
    TRANSITION_META,
    FakeRouter,
    build_cfg,
    chat_response,
    judge_reply,
    open_store,
    paths_for,
)

from tuned.data.generate import run_workers
from tuned.data.jsonl import read_at
from tuned.data.judge import (
    BORDERLINE,
    FAIL,
    JUDGE_STATE_FROM,
    JUDGE_STATE_TO,
    MAX_JUDGE_ATTEMPTS,
    PASS,
    JudgeParseError,
    JudgeScores,
    SlotOutcome,
    decide,
    failing_rationale,
    generation_family,
    judge_task,
    parse_judge_reply,
    run_judges,
    thresholds_active,
    undersized_families,
)
from tuned.data.tasks import plan_wave


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
    with judged_store(tmp_path, paths, cfg) as store:
        task = only_task(store)
        # Claim it as somebody else, then hand the stale worker's verdict in.
        store.claim_tasks(
            "other-worker", 1, state_from=JUDGE_STATE_FROM, state_to=JUDGE_STATE_TO
        )
        asyncio.run(
            judge_task(
                store, cfg, FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]}), task,
                paths=paths, worker_id="stale-worker",
            )
        )
        assert only_task(store)["state"] == JUDGE_STATE_TO
        assert only_task(store)["claimed_by"] == "other-worker"
        lost = json.loads(store.events("lost_lease")[0]["detail_json"])
        assert lost["worker"] == "stale-worker"
        assert lost["wanted_state"] == "accepted"


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


def test_judge_batch_line(tmp_path, cfg, paths, capsys):
    with judged_store(tmp_path, paths, cfg) as store:
        run_judge(store, cfg, FakeRouter(cfg, {"judge": [judge_reply(5, 5, 5)]}), paths)
        out = capsys.readouterr().out
        assert "judge batch 1: claimed=1 decided=1 accepted=1" in out
