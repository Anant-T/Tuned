import asyncio
import json

import pytest
from pipeline_fakes import (
    CLEAN_ANSWER,
    CLEAN_THINK,
    FABRICATED_ANSWER,
    FABRICATED_SUSPECT,
    LONG_SEED_TEXT,
    SEED_TEXT,
    TRANSITION_META,
    FakeRouter,
    build_cfg,
    chat_response,
    open_store,
    paths_for,
    seed_rows,
)

from tuned.data import gates
from tuned.data.config import ModelRef
from tuned.data.generate import (
    MAX_ATTEMPTS,
    QUESTION_BY_TASK_TYPE,
    GenResult,
    SlotError,
    append_reviewer_note,
    apply_gate_disposition,
    assemble_content,
    budget_ok_for,
    build_prompt,
    build_slots,
    effort_for_attempt,
    gate_context,
    generate_once,
    grounding_text,
    next_attempt,
    run_workers,
)
from tuned.data.jsonl import read_at
from tuned.data.providers import ProviderError
from tuned.data.tasks import plan_wave


@pytest.fixture
def cfg():
    return build_cfg()


@pytest.fixture
def paths(tmp_path):
    return paths_for(tmp_path)


def make_store(tmp_path, *, n_seeds=1, n_tasks=1, mix=None, meta=None, stream="synthesis",
               text=SEED_TEXT):
    store = open_store(tmp_path, n_seeds=n_seeds, meta=meta, text=text)
    plan_wave(
        store, build_cfg(), stream, n_tasks,
        task_type_mix=mix or {"irac_analysis": 1.0},
    )
    return store


def run(store, cfg, router, paths, **kwargs):
    kwargs.setdefault("streams", ["synthesis"])
    kwargs.setdefault("n_workers", 4)
    kwargs.setdefault("max_batches", 1)
    return asyncio.run(run_workers(store, cfg, router, paths=paths, **kwargs))


def only_task(store):
    return dict(store.conn.execute("SELECT * FROM task LIMIT 1").fetchone())


# --------------------------------------------------------------------------
# Prompt assembly and the grounding contract.
# --------------------------------------------------------------------------

def test_grounding_is_the_union_of_the_material_slots():
    slots = {
        "source": "AAA",
        "question": "ignored",
        "section_text": "BBB",
        "focus_issue": "ignored too",
    }
    assert grounding_text(slots) == "AAA\n\nBBB"


def test_statute_qa_grounding_carries_the_section_text(tmp_path, cfg):
    meta = {"section_text": f"Section 9 reads as follows, per {FABRICATED_SUSPECT}."}
    with make_store(tmp_path, meta=meta, mix={"statute_qa": 1.0}) as store:
        task = only_task(store)
        seed = store.get_seed(task["seed_id"])
        bundle = build_prompt(cfg, task, seed)
        assert meta["section_text"] in bundle.grounding
        assert SEED_TEXT in bundle.grounding
        # CONTRACT 1: a citation carried IN by a grounding slot is not the
        # model's invention. With the full grounding the gate passes; with
        # {source} alone the same answer takes a PERMANENT citation reject.
        answer = f"Issue\nX\n\nConclusion\nY, see {FABRICATED_SUSPECT}."
        full = gates.check_citations(answer, gate_context(cfg, task, seed, bundle.grounding))
        source_only = gates.check_citations(answer, gate_context(cfg, task, seed, seed["text"]))
        assert full.passed
        assert not source_only.passed
        assert source_only.detail["suspect"] == [FABRICATED_SUSPECT]


def test_statute_qa_falls_back_to_the_seed_text(tmp_path, cfg):
    with make_store(tmp_path, mix={"statute_qa": 1.0}) as store:
        task = only_task(store)
        slots = build_slots(cfg, task, store.get_seed(task["seed_id"]))
        assert slots["section_text"] == SEED_TEXT


def test_seed_meta_question_overrides_the_standing_ask(tmp_path, cfg):
    with make_store(tmp_path, meta={"question": "Is the appeal maintainable?"}) as store:
        task = only_task(store)
        slots = build_slots(cfg, task, store.get_seed(task["seed_id"]))
        assert slots["question"] == "Is the appeal maintainable?"
    with make_store(tmp_path / "b") as store:
        task = only_task(store)
        slots = build_slots(cfg, task, store.get_seed(task["seed_id"]))
        assert slots["question"] == QUESTION_BY_TASK_TYPE["irac_analysis"]


def test_transition_slots_are_never_invented(tmp_path, cfg):
    with make_store(tmp_path, stream="transition", mix={"transition": 1.0}) as store:
        task = only_task(store)
        with pytest.raises(SlotError):
            build_slots(cfg, task, store.get_seed(task["seed_id"]))


def test_reviewer_note_extends_the_last_user_turn():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "do it"}]
    out = append_reviewer_note(messages, "the conclusion does not follow")
    assert len(out) == 2
    assert out[1]["role"] == "user"
    assert out[1]["content"].startswith("do it")
    assert "the conclusion does not follow" in out[1]["content"]
    # The original list is untouched (never edit the caller's messages).
    assert messages[1]["content"] == "do it"


# --------------------------------------------------------------------------
# Content assembly.
# --------------------------------------------------------------------------

def test_assemble_content_from_the_reasoning_channel(cfg):
    content, think, answer = assemble_content(cfg, chat_response("ANSWER", "TRACE"))
    assert content == f"{cfg.think_open}\nTRACE\n{cfg.think_close}\n\nANSWER"
    assert (think, answer) == ("TRACE", "ANSWER")


def test_assemble_content_from_inline_think_tags(cfg):
    text = f"{cfg.think_open}TRACE{cfg.think_close}\nANSWER"
    content, think, answer = assemble_content(cfg, chat_response(text, None))
    assert think == "TRACE"
    assert answer == "ANSWER"
    assert content.count(cfg.think_open) == 1


def test_assemble_content_never_fabricates_a_trace(cfg):
    content, think, answer = assemble_content(cfg, chat_response("just an answer", None))
    assert think is None
    assert content == "just an answer"
    assert cfg.think_open not in content


def test_effort_ladder():
    assert effort_for_attempt(1, "medium") == "medium"
    assert effort_for_attempt(2, "medium") == "high"
    assert effort_for_attempt(3, "medium") == "high"
    assert effort_for_attempt(2, "low") == "medium"


# --------------------------------------------------------------------------
# The happy path, end to end.
# --------------------------------------------------------------------------

def test_clean_generation_lands_in_judging(tmp_path, cfg, paths):
    with make_store(tmp_path) as store:
        router = FakeRouter(cfg)
        totals = run(store, cfg, router, paths)
        assert totals["gen_ok"] == 1
        assert totals["gated_out"] == 0
        task = only_task(store)
        assert task["state"] == "judging"
        assert task["claimed_by"] is None
        gen = store.latest_generation(task["task_id"])
        assert gen["attempt"] == 1
        assert gen["think"].startswith("I start with")
        assert gen["answer"].startswith("Issue")
        assert gen["provider"] == "cerebras"
        assert gen["model_family"] == "gpt-oss"
        assert store.gates_for(gen["gen_id"]) == {gate: True for gate in gates.GATE_ORDER}


def test_generation_records_usage_against_the_ledger(tmp_path, cfg, paths):
    with make_store(tmp_path) as store:
        run(store, cfg, FakeRouter(cfg), paths)
        used = store.usage_today("cerebras", "gpt-oss-120b")
        assert used == {
            "requests": 1,
            "prompt_tokens": 900,
            "completion_tokens": 800,
            "errors_429": 0,
        }


def test_generator_call_carries_the_rendered_prompt(tmp_path, cfg, paths):
    with make_store(tmp_path) as store:
        router = FakeRouter(cfg)
        run(store, cfg, router, paths)
        call = router.calls_for("generator")[0]
        assert call["params"] == {}
        assert SEED_TEXT in call["messages"][-1]["content"]
        assert call["max_tokens"] == cfg.build.length_band.think_max + 1000
        assert call["est_tokens"] > call["max_tokens"]


# --------------------------------------------------------------------------
# Raw-first durability.
# --------------------------------------------------------------------------

def test_raw_envelope_round_trips_by_offset(tmp_path, cfg, paths):
    with make_store(tmp_path) as store:
        run(store, cfg, FakeRouter(cfg), paths)
        gen = store.latest_generation(only_task(store)["task_id"])
        record = read_at(gen["raw_path"], gen["raw_offset"])
        assert record["kind"] == "generation"
        assert record["task_id"] == gen["task_id"]
        assert record["attempt"] == gen["attempt"]
        assert record["think"] == gen["think"]
        assert record["answer"] == gen["answer"]
        assert record["content"].startswith(cfg.think_open)
        assert record["messages"][-1]["role"] == "user"


def test_a_raw_envelope_reconciles_into_a_fresh_database(tmp_path, cfg, paths):
    """The crash window: raw log written, index row lost. reconcile_raw must
    rebuild the row from the envelope this module writes - which is the only
    reason the envelope carries every generation column."""
    with make_store(tmp_path) as store:
        run(store, cfg, FakeRouter(cfg), paths)
        gen = store.latest_generation(only_task(store)["task_id"])
        raw_path = gen["raw_path"]

    with open_store(tmp_path / "rebuild", n_seeds=1) as rebuilt:
        plan_wave(rebuilt, cfg, "synthesis", 1, task_type_mix={"irac_analysis": 1.0})
        assert rebuilt.reconcile_raw([raw_path]) == 1
        recovered = rebuilt.latest_generation(gen["task_id"])
        assert recovered["think"] == gen["think"]
        assert recovered["answer"] == gen["answer"]
        assert recovered["prompt_tokens"] == gen["prompt_tokens"]
        assert recovered["completion_tokens"] == gen["completion_tokens"]
        assert recovered["model_family"] == gen["model_family"]
        assert recovered["raw_offset"] == gen["raw_offset"]
        assert json.loads(recovered["params_json"])["reviewer_note_applied"] is False
        # Idempotent: a second sweep recovers nothing.
        assert rebuilt.reconcile_raw([raw_path]) == 0


def test_a_crash_before_indexing_cannot_double_spend_the_attempt(tmp_path, cfg, paths):
    """A task re-claimed after a crash comes back with a HIGHER attempt, so
    the orphaned raw record and the retry are different rows and the
    UNIQUE(task_id, attempt) constraint is never contended."""
    with make_store(tmp_path) as store:
        router = FakeRouter(cfg)
        run(store, cfg, router, paths)
        task = only_task(store)
        gen = store.latest_generation(task["task_id"])
        # Simulate the lost index row: drop it, leaving the raw log intact.
        store.conn.execute("DELETE FROM gate_result WHERE gen_id = ?", (gen["gen_id"],))
        store.conn.execute("DELETE FROM generation WHERE gen_id = ?", (gen["gen_id"],))
        store.set_task_state(task["task_id"], "pending")

        run(store, cfg, router, paths)
        retry = store.latest_generation(task["task_id"])
        assert retry["attempt"] == 2
        assert store.reconcile_raw([gen["raw_path"]]) == 1
        attempts = [
            row[0]
            for row in store.conn.execute(
                "SELECT attempt FROM generation WHERE task_id = ? ORDER BY attempt",
                (task["task_id"],),
            ).fetchall()
        ]
        assert attempts == [1, 2]


def test_a_duplicate_attempt_adopts_the_row_it_belongs_to(tmp_path, cfg, paths):
    with make_store(tmp_path) as store:
        task = only_task(store)
        first = asyncio.run(
            generate_once(store, cfg, FakeRouter(cfg), task, paths=paths, attempt=1)
        )
        again = asyncio.run(
            generate_once(store, cfg, FakeRouter(cfg), task, paths=paths, attempt=1)
        )
        assert again.ok is True
        assert again.gen_id == first.gen_id
        assert store.events("generation_duplicate")


def test_a_duplicate_attempt_never_adopts_a_different_answer(tmp_path, cfg, paths):
    with make_store(tmp_path) as store:
        task = only_task(store)
        asyncio.run(generate_once(store, cfg, FakeRouter(cfg), task, paths=paths, attempt=1))
        asyncio.run(generate_once(store, cfg, FakeRouter(cfg), task, paths=paths, attempt=2))
        stale = asyncio.run(
            generate_once(store, cfg, FakeRouter(cfg), task, paths=paths, attempt=1)
        )
        assert stale.ok is False
        assert "already indexed" in stale.error
        # Attempt 2's gate results are untouched by the collision.
        assert store.gates_for(store.latest_generation(task["task_id"])["gen_id"])


def test_next_attempt_reconciles_claims_and_generations(tmp_path, cfg, paths):
    with make_store(tmp_path) as store:
        task = only_task(store)
        assert next_attempt(store, {**task, "attempts": 0}) == 1
        assert next_attempt(store, {**task, "attempts": 3}) == 3
        run(store, cfg, FakeRouter(cfg), paths)
        assert next_attempt(store, {**task, "attempts": 0}) == 2


# --------------------------------------------------------------------------
# Gate wiring.
# --------------------------------------------------------------------------

def test_a_fabricated_citation_is_a_permanent_reject(tmp_path, cfg, paths):
    with make_store(tmp_path) as store:
        router = FakeRouter(cfg, {"generator": [chat_response(FABRICATED_ANSWER, CLEAN_THINK)]})
        totals = run(store, cfg, router, paths)
        assert totals["gated_out"] == 1
        task = only_task(store)
        assert task["state"] == "rejected"
        assert task["disposition"].startswith("reject:")
        assert "citations" in task["disposition"]
        gen = store.latest_generation(task["task_id"])
        assert store.gates_for(gen["gen_id"])["citations"] is False


def test_a_missing_trace_is_a_regeneration_not_a_reject(tmp_path, cfg, paths):
    with make_store(tmp_path) as store:
        router = FakeRouter(cfg, {"generator": [chat_response(CLEAN_ANSWER, None)]})
        run(store, cfg, router, paths)
        task = only_task(store)
        assert task["state"] == "pending"
        assert task["disposition"].startswith("regenerate:")
        assert "think_format" in task["disposition"]
        assert task["attempts"] == 1


def test_the_retry_bumps_reasoning_effort_and_says_so(tmp_path, cfg, paths):
    with make_store(tmp_path) as store:
        router = FakeRouter(cfg, {"generator": [chat_response(CLEAN_ANSWER, None)]})
        run(store, cfg, router, paths)
        run(store, cfg, router, paths)
        calls = router.calls_for("generator")
        assert calls[0]["params"] == {}
        assert calls[1]["params"] == {"reasoning_effort": "high"}
        events = store.events("effort_bump")
        assert len(events) == 1
        assert json.loads(events[0]["detail_json"])["attempt"] == 2


def test_regeneration_is_exhausted_at_the_attempt_cap(tmp_path, cfg, paths):
    with make_store(tmp_path) as store:
        router = FakeRouter(cfg, {"generator": [chat_response(CLEAN_ANSWER, None)]})
        for _ in range(MAX_ATTEMPTS):
            run(store, cfg, router, paths)
        task = only_task(store)
        assert task["attempts"] == MAX_ATTEMPTS
        assert task["state"] == "rejected"
        assert task["disposition"].startswith("exhausted:regenerate:")


def test_gates_run_without_an_index_and_say_so(tmp_path, cfg, paths):
    with make_store(tmp_path) as store:
        run(store, cfg, FakeRouter(cfg), paths)
        gen = store.latest_generation(only_task(store)["task_id"])
        detail = json.loads(
            store.conn.execute(
                "SELECT detail_json FROM gate_result WHERE gen_id = ? AND gate = 'citations'",
                (gen["gen_id"],),
            ).fetchone()[0]
        )
        assert detail["novel_skipped"] == "no-index"


# --------------------------------------------------------------------------
# Failure paths that must not spend, crash or strand a task.
# --------------------------------------------------------------------------

def test_provider_failure_returns_the_task_to_the_queue(tmp_path, cfg, paths):
    error = ProviderError(
        "429 everywhere", status=429, provider="cerebras", model="gpt-oss-120b", retryable=True
    )
    with make_store(tmp_path) as store:
        totals = run(store, cfg, FakeRouter(cfg, {"generator": [error]}), paths)
        assert totals["errors"] == 1
        assert totals["gen_ok"] == 0
        task = only_task(store)
        assert task["state"] == "pending"
        assert store.usage_today("cerebras", "gpt-oss-120b")["errors_429"] == 1
        assert store.events("generation_error")


def test_provider_failure_at_the_cap_rejects(tmp_path, cfg, paths):
    error = ProviderError("dead", status=500, provider="cerebras", model="gpt-oss-120b",
                          retryable=True)
    with make_store(tmp_path) as store:
        for _ in range(MAX_ATTEMPTS):
            run(store, cfg, FakeRouter(cfg, {"generator": [error]}), paths)
        assert only_task(store)["disposition"] == "exhausted:error"
        assert only_task(store)["state"] == "rejected"


def test_a_missing_seed_is_skipped_without_spending(tmp_path, cfg, paths):
    with make_store(tmp_path) as store:
        router = FakeRouter(cfg)
        task = {**only_task(store), "seed_id": "not-a-seed"}
        result = asyncio.run(generate_once(store, cfg, router, task, paths=paths))
        assert result.skipped == "missing-seed"
        assert router.calls == []
        assert apply_gate_disposition(store, task, result, worker_id=None) == "rejected"
        assert only_task(store)["disposition"] == "skip:missing-seed"


def test_unrenderable_slots_are_skipped_without_spending(tmp_path, cfg, paths):
    with make_store(tmp_path, stream="transition", mix={"transition": 1.0}) as store:
        router = FakeRouter(cfg)
        task = only_task(store)
        result = asyncio.run(generate_once(store, cfg, router, task, paths=paths))
        assert result.skipped == "slots"
        assert router.calls == []
        assert store.events("generation_skipped")


def test_a_lost_lease_drops_the_result_instead_of_clobbering(tmp_path, cfg, paths):
    with make_store(tmp_path) as store:
        task = only_task(store)
        store.set_task_state(task["task_id"], "generating")
        store.claim_tasks("other-worker", 1)
        result = GenResult(task_id=task["task_id"], attempt=1, ok=True)
        assert apply_gate_disposition(store, task, result, worker_id="me") == "lost-lease"
        assert only_task(store)["state"] == "generating"
        assert only_task(store)["claimed_by"] == "other-worker"
        assert store.events("lost_lease")


# --------------------------------------------------------------------------
# Budget wiring (contract 7).
# --------------------------------------------------------------------------

def test_budget_ok_reads_the_daily_ledger(tmp_path, cfg):
    with open_store(tmp_path, n_seeds=0) as store:
        budget_ok = budget_ok_for(store, cfg)
        assert budget_ok("cerebras", "gpt-oss-120b", 1000) is True
        _, model = cfg.model_for(ModelRef("cerebras", "gpt-oss-120b"))
        store.record_usage(
            "cerebras", "gpt-oss-120b",
            prompt_tokens=int(model.limits["tpd"]), completion_tokens=0,
        )
        assert budget_ok("cerebras", "gpt-oss-120b", 1) is False
        # A ref the config does not know is allowed rather than blocked.
        assert budget_ok("nowhere", "nothing", 10**9) is True


# --------------------------------------------------------------------------
# The loop itself.
# --------------------------------------------------------------------------

def test_batch_processes_every_claimed_task(tmp_path, cfg, paths):
    with make_store(tmp_path, n_seeds=4, n_tasks=4) as store:
        totals = run(store, cfg, FakeRouter(cfg), paths, n_workers=4)
        assert totals["claimed"] == 4
        assert totals["gen_ok"] == 4
        assert store.task_counts() == {"judging": 4}


def test_loop_stops_when_the_queue_is_empty(tmp_path, cfg, paths, capsys):
    with make_store(tmp_path, n_seeds=2, n_tasks=2) as store:
        totals = run(store, cfg, FakeRouter(cfg), paths, n_workers=1, max_batches=None)
        # 2 tasks at 1 per batch, then one empty batch that ends the loop.
        assert totals["batches"] == 3
        assert totals["claimed"] == 2
        assert "batch 3: claimed=0" in capsys.readouterr().out


def test_status_line_reports_the_batch_and_the_budget(tmp_path, cfg, paths, capsys):
    with make_store(tmp_path) as store:
        run(store, cfg, FakeRouter(cfg), paths)
        line = capsys.readouterr().out.strip().splitlines()[-1]
        assert line.startswith("batch 1: claimed=1 gen-ok=1 gated-out=0 err=0 tokens=1700")
        assert "cerebras/gpt-oss-120b spent=1.7k" in line


def test_streams_are_claimed_separately(tmp_path, cfg, paths):
    with make_store(tmp_path, n_seeds=4, n_tasks=2) as store:
        plan_wave(store, cfg, "curated_c2", 2, task_type_mix={"summarization": 1.0})
        run(store, cfg, FakeRouter(cfg), paths, streams=["synthesis"], n_workers=10)
        counts = store.task_counts()
        assert counts["judging"] == 2
        assert counts["pending"] == 2


# --------------------------------------------------------------------------
# Context routing for the GENERATOR (an 8k model is first in the list).
# --------------------------------------------------------------------------

def test_a_long_prompt_is_routed_past_the_8k_generator(tmp_path, cfg, paths):
    with make_store(tmp_path, text=LONG_SEED_TEXT) as store:
        router = FakeRouter(cfg)
        run(store, cfg, router, paths)
        call = router.calls_for("generator")[0]
        assert call["est_tokens"] > 8192
        # cerebras/gpt-oss-120b is 8k and first in the preference list; an
        # over-long prompt there is a 400, and a 400 does NOT fail over.
        assert "gpt-oss" in call["exclude_families"]
        assert call["ref"] == ModelRef("mistral", "magistral-small-latest")
        assert only_task(store)["state"] == "judging"


def test_a_short_prompt_excludes_no_generator(tmp_path, cfg, paths):
    with make_store(tmp_path) as store:
        router = FakeRouter(cfg)
        run(store, cfg, router, paths)
        call = router.calls_for("generator")[0]
        assert call["est_tokens"] <= 8192
        assert call["exclude_families"] == frozenset()
        assert call["ref"] == ModelRef("cerebras", "gpt-oss-120b")


def test_the_effort_bump_is_chosen_for_the_ref_that_answers(tmp_path, cfg, paths):
    """reasoning_effort is a gpt-oss parameter. On a long prompt the call
    lands on magistral, which does not declare it - and an unknown parameter
    is a 400 that never fails over."""
    with make_store(tmp_path, text=LONG_SEED_TEXT) as store:
        router = FakeRouter(cfg)
        task = only_task(store)
        asyncio.run(generate_once(store, cfg, router, task, paths=paths, attempt=2))
        call = router.calls_for("generator")[0]
        assert call["ref"].provider == "mistral"
        assert call["params"] == {}
        assert store.events("effort_bump") == []

    with make_store(tmp_path / "short") as store:
        router = FakeRouter(cfg)
        task = only_task(store)
        asyncio.run(generate_once(store, cfg, router, task, paths=paths, attempt=2))
        call = router.calls_for("generator")[0]
        assert call["ref"].provider == "cerebras"
        assert call["params"] == {"reasoning_effort": "high"}
        assert json.loads(store.events("effort_bump")[0]["detail_json"])["ref"] == (
            "cerebras/gpt-oss-120b"
        )


def test_an_unroutable_generator_closes_the_task_at_once(tmp_path, cfg, paths):
    """No eligible model is a fact about the row, not the moment: three more
    claims would meet the same wall, so the task is closed with a diagnostic
    instead of cycling."""
    with make_store(tmp_path) as store:
        # A seed longer than every generator's context window.
        store.upsert_seeds(
            [{**seed_rows(1)[0], "text": "word " * 60000}]
        )
        task = only_task(store)
        router = FakeRouter(cfg)
        result = asyncio.run(generate_once(store, cfg, router, task, paths=paths))
        assert result.ok is False
        assert result.permanent is True
        assert apply_gate_disposition(store, task, result, worker_id=None) == "rejected"
        assert only_task(store)["disposition"] == "unroutable:generator"
        event = json.loads(store.events("generation_error")[0]["detail_json"])
        assert event["permanent"] is True
        assert set(event["excluded_families"]) == {"gpt-oss", "mistral"}


# --------------------------------------------------------------------------
# Grounding hygiene and the transition stream's date requirement.
# --------------------------------------------------------------------------

def test_grounding_collapses_a_duplicated_slot(tmp_path, cfg):
    """The pilot's statute_qa fallback puts the seed text in {source} AND
    {section_text}; concatenating both doubles every judge prompt and every
    context estimate for no added grounding."""
    with make_store(tmp_path, mix={"statute_qa": 1.0}) as store:
        task = only_task(store)
        bundle = build_prompt(cfg, task, store.get_seed(task["seed_id"]))
        assert bundle.grounding == SEED_TEXT
        assert bundle.grounding.count(SEED_TEXT) == 1


def test_grounding_keeps_a_distinct_section_text(tmp_path, cfg):
    meta = {"section_text": "Section 9. A distinct provision, not the seed text."}
    with make_store(tmp_path, meta=meta, mix={"statute_qa": 1.0}) as store:
        task = only_task(store)
        bundle = build_prompt(cfg, task, store.get_seed(task["seed_id"]))
        assert bundle.grounding == f"{SEED_TEXT}\n\n{meta['section_text']}"


def test_transition_refuses_a_seed_without_its_dates(tmp_path, cfg, paths):
    slots_only = {k: v for k, v in TRANSITION_META.items() if not k.endswith("date")}
    slots_only.pop("proceeding_started")
    with make_store(
        tmp_path, stream="transition", mix={"transition": 1.0}, meta=slots_only
    ) as store:
        task = only_task(store)
        with pytest.raises(SlotError, match="offence_date"):
            build_slots(cfg, task, store.get_seed(task["seed_id"]))
        # And the worker refuses it unspent rather than paying for a row the
        # temporal gate will permanently reject.
        router = FakeRouter(cfg)
        result = asyncio.run(generate_once(store, cfg, router, task, paths=paths))
        assert result.skipped == "slots"
        assert router.calls == []


def test_transition_renders_once_the_dates_are_there(tmp_path, cfg):
    with make_store(
        tmp_path, stream="transition", mix={"transition": 1.0}, meta=TRANSITION_META
    ) as store:
        task = only_task(store)
        seed = store.get_seed(task["seed_id"])
        bundle = build_prompt(cfg, task, seed)
        # The scenario is judge-visible but NOT part of the citation allow-list.
        assert TRANSITION_META["scenario"] not in bundle.grounding
        assert TRANSITION_META["scenario"] in bundle.judge_source
        assert bundle.judge_source.startswith(bundle.grounding)
        assert TRANSITION_META["old_section_text"] in bundle.grounding
        assert TRANSITION_META["savings_text"] in bundle.grounding
        ctx = gate_context(cfg, task, seed, bundle.grounding)
        assert ctx.source_text == bundle.grounding
        assert ctx.offence_date.isoformat() == "2024-03-12"
        assert ctx.proceeding_started.isoformat() == "2024-09-04"


def test_judge_source_equals_grounding_off_the_transition_stream(tmp_path, cfg):
    with make_store(tmp_path) as store:
        task = only_task(store)
        bundle = build_prompt(cfg, task, store.get_seed(task["seed_id"]))
        assert bundle.judge_source == bundle.grounding


# --------------------------------------------------------------------------
# Ledger and loop robustness.
# --------------------------------------------------------------------------

def test_every_attempt_is_ledgered_including_the_429(tmp_path, cfg, paths):
    error = ProviderError(
        "rate limited", status=429, provider="cerebras", model="gpt-oss-120b", retryable=True
    )
    with make_store(tmp_path) as store:
        run(store, cfg, FakeRouter(cfg, {"generator": [error]}), paths)
        used = store.usage_today("cerebras", "gpt-oss-120b")
        assert used["requests"] == 1
        assert used["errors_429"] == 1
        assert used["prompt_tokens"] == 0


def test_a_poisoned_task_does_not_kill_the_batch(tmp_path, cfg, paths):
    with make_store(tmp_path, n_seeds=2, n_tasks=2) as store:
        # The first task's call blows up with something nobody anticipated.
        router = FakeRouter(cfg, {"generator": [RuntimeError("boom"), chat_response()]})
        totals = run(store, cfg, router, paths, n_workers=2)
        assert totals["errors"] == 1
        assert totals["gen_ok"] == 1
        event = json.loads(store.events("worker_task_error")[0]["detail_json"])
        assert event["error"].startswith("RuntimeError: boom")
        # The survivor still went through; the poisoned one keeps its lease
        # and is recovered when it expires.
        counts = store.task_counts()
        assert counts["judging"] == 1
        assert counts["generating"] == 1


def test_idle_batches_are_announced_once(tmp_path, cfg, paths, capsys):
    slept: list[float] = []

    async def sleeper(delay):
        slept.append(delay)

    with make_store(tmp_path, n_seeds=0, n_tasks=0) as store:
        asyncio.run(
            run_workers(
                store, cfg, FakeRouter(cfg), paths=paths, streams=["synthesis"],
                n_workers=1, forever=True, max_batches=4, sleeper=sleeper, idle_sleep_s=0.01,
            )
        )
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("batch")]
        assert len(lines) == 1
        assert "claimed=0" in lines[0]
        assert len(slept) == 4


def test_gate_context_is_built_from_the_seed(tmp_path, cfg):
    with make_store(tmp_path) as store:
        task = only_task(store)
        seed = dict(store.get_seed(task["seed_id"]))
        seed["offence_date"] = "2023-05-04"
        seed["decision_date"] = "2025-01-09"
        seed["meta_json"] = json.dumps({"proceeding_started": "2024-02-02"})
        seed["answer_key_json"] = json.dumps({"governing_family": "old"})
        ctx = gate_context(cfg, task, seed, "GROUND")
        assert ctx.source_text == "GROUND"
        assert ctx.offence_date.year == 2023
        # decision_date is NOT a stand-in for when the proceeding started.
        assert ctx.proceeding_started.isoformat() == "2024-02-02"
        assert ctx.answer_key == {"governing_family": "old"}
        assert ctx.expect_reasoning is True
        assert ctx.citation_index is None
        assert ctx.band is cfg.build.length_band
