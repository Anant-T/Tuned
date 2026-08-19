import asyncio
import json

import pytest
from pipeline_fakes import (
    CLEAN_ANSWER,
    DATA_CONFIG,
    CLEAN_THINK,
    FABRICATED_ANSWER,
    FABRICATED_SUSPECT,
    LONG_SEED_TEXT,
    NARROW_GENERATOR_CONTEXT,
    OVERSIZE_SEED_TEXT,
    SEED_TEXT,
    TRANSITION_META,
    FakeRouter,
    StealsTheLease,
    build_cfg,
    cfg_with_fourth_judge_family,
    cfg_with_context,
    cfg_with_extra_judge,
    cfg_with_two_generator_families,
    cfg_with_split_pools,
    chat_response,
    open_store,
    paths_for,
    seed_rows,
    temp_config,
)

from tuned.data import gates
from tuned.data.config import ModelRef, load_build_config
from tuned.data.generate import (
    GEN_UNROUTABLE_STATE,
    GENERATOR_PREFLIGHT_ROLES,
    MAX_ATTEMPTS,
    QUESTION_BY_TASK_TYPE,
    REPLY_BUDGET_GATE,
    STALE_PROMPT_STATE,
    BatchStats,
    GenResult,
    SlotError,
    append_reviewer_note,
    apply_gate_disposition,
    assemble_content,
    budget_ok_for,
    build_prompt,
    ANSWER_TOKEN_ALLOWANCE,
    build_slots,
    effort_params_for_ref,
    format_batch_line,
    gate_context,
    generate_once,
    grounding_text,
    judge_sizer,
    legal_reply_chars,
    max_output_tokens,
    next_attempt,
    preflight_messages,
    reply_budget_chars,
    reply_over_budget,
    run_workers,
)
from tuned.data.generate import (
    GENERATION_OUTPUT_TOKENS,
    REPLY_BUDGET_CHARS_PER_TOKEN,
)
from tuned.data.generate import main as generate_main
from tuned.data.jsonl import read_at
from tuned.data.providers import (
    QUOTA_OBSERVATION_TTL_S,
    ProviderError,
    QuotaLedger,
    parse_quota_headers,
    required_context,
)
from tuned.data.tasks import plan_wave, reopen_tasks


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


def test_statute_qa_grounding_carries_the_section_number_the_gate_checks(tmp_path, cfg):
    """CONTRACT 1 again, for statutory_grounding. The provision arrives through
    {section_text}, so a gate wired to the SEED sees no s.9 anywhere and reads
    the one citation the task exists to elicit as a fabrication. statute_qa is
    125 of the 416-task backlog, which is what makes the distinction
    load-bearing rather than tidy.

    The seed text has to name SOME provision of its own, or the seed-only
    context has an empty allow-list and skips - which would leave the test
    green for a reason that has nothing to do with the slot union. See the
    no-material-references skip.
    """
    meta = {"section_text": "Section 9. No suit shall lie in respect of a claim so barred."}
    seed_text = SEED_TEXT + "\n\nThe suit was instituted under Section 34 of the Act."
    with make_store(tmp_path, meta=meta, mix={"statute_qa": 1.0}, text=seed_text) as store:
        task = only_task(store)
        seed = store.get_seed(task["seed_id"])
        bundle = build_prompt(cfg, task, seed)
        assert "Section 9" not in seed["text"]

        answer = "Issue\nWhether the suit lies.\n\nConclusion\nSection 9 bars it."
        full = gates.check_statutory_grounding(
            answer, gate_context(cfg, task, seed, bundle.grounding)
        )
        source_only = gates.check_statutory_grounding(
            answer, gate_context(cfg, task, seed, seed["text"])
        )
        assert full.passed
        assert not source_only.passed
        assert source_only.detail["ungrounded"][0]["number"] == "9"


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


@pytest.mark.parametrize("attempt", [1, 2, 3, 4])
def test_a_retry_re_rolls_the_same_request_and_never_bumps_effort(cfg, attempt):
    """RETIRED LADDER (2026-08-18). This used to assert the opposite -
    effort_for_attempt(2, "medium") == "high" - and the ladder it pinned is the
    single largest contributor to the pilot's 0-clean-generations run.

    Measured on cerebras/gpt-oss-120b across 176 generations, attempts 1/2/3:
    mean trace 2,411 -> 12,353 -> 12,536 chars, and with it verbatim_overlap
    35/60 -> 58/59 -> 58/58, banned_meta 11/60 -> 51/59 -> 53/58,
    irac_placement 6/60 -> 39/59 -> 45/58, finish_reason=length 0 -> 18 -> 16.
    The one gate it improved (self_verification 51/60 -> 22/59) it improved by
    accident, a longer trace being likelier to contain one of twelve literal
    cues; that is now handled in the prompt, which hands the cues over.

    So no attempt may send a DIFFERENT reasoning_effort from any other: what a
    generator is asked for is a property of its family, never of how many times
    the row has been tried. That is the attempt-1 condition all of the above
    was measured under.

    (What a family is asked for is a separate question, settled by
    GENERATOR_REASONING_PARAMS: gpt-oss reasons by default and is sent nothing,
    mistral's channel is opt-in and is sent the same value every time. This
    test pins the INVARIANCE; test_the_generator_opts_mistral_into_reasoning
    pins the value.)
    """
    first = effort_params_for_ref(1)
    this = effort_params_for_ref(attempt)
    for provider in cfg.providers:
        for model in provider.models:
            ref = ModelRef(provider=provider.name, model=model.id)
            assert this(ref, model) == first(ref, model)


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


# The shape the length band cannot see: a trace at the band's own think_max
# and an answer built from the remainder. Every gate passes and the reply is
# still more than twice the max_tokens the call asked for - which is the
# premise the per-generator-window judge sizing rests on.
OVERSIZE_THINK = (CLEAN_THINK * 5)[:11_900]
OVERSIZE_ANSWER = CLEAN_ANSWER + " " + (
    "The gap in the chain is the point to press, and it is the point on which this turns. " * 190
)

# A trace that trips exactly one REGENERATE gate: it clears think_min, carries
# no banned phrase, no line-initial IRAC heading and no citation, and simply
# never doubts itself - so check_self_verification fails and nothing else does.
#
# It exists because a traceless response used to be the convenient way to make
# a test produce a regeneration, and since 2026-08-18 that shape parks as a
# provider fact instead (see
# test_a_missing_trace_parks_as_a_provider_fact_not_a_content_failure). Tests
# that want a CONTENT regeneration have to ask for one.
NO_CUE_THINK = (
    "The right pleaded rests on the section both parties invoke, and the "
    "question is whether the facts as recorded bring the case within it. "
) * 20


def test_the_generation_budget_covers_the_largest_gate_legal_reply(cfg):
    """The inequality that replaced an identity (2026-08-18).

    max_output_tokens used to BE `think_max + ANSWER_TOKEN_ALLOWANCE`, which
    added a gate threshold counted in chars//4 to an allowance counted in
    chars//4 and sent the sum to a provider as `max_tokens`, counted in real
    tokens. The two currencies differ by ~17% on this corpus (measured
    4.24-5.13 chars/token over 155 gpt-oss generations against the 4.0 the
    estimate assumes), and the coupling meant raising a gate ceiling silently
    re-priced every generation call and the judge-pool sizing that reads it.

    What actually has to hold is this: the budget must carry the largest reply
    the GATES would pass, converted at the measured WORST-CASE chars/token -
    worst case being the SMALLEST ratio, since fewer chars per token means more
    tokens for the same text. Raising think_max now fails here instead of
    quietly re-pricing the fleet.
    """
    # Measured minimum over the pilot; see GENERATION_OUTPUT_TOKENS.
    measured_min_chars_per_token = 4.24
    worst_case_tokens = legal_reply_chars(cfg) / measured_min_chars_per_token
    assert max_output_tokens(cfg) >= worst_case_tokens
    # ...and it is not derived from the band any more, so moving the band does
    # not move it.
    assert max_output_tokens(cfg) == GENERATION_OUTPUT_TOKENS
    # The primary generator's declared ceiling, above which the cerebras
    # request hook would clamp us without saying so.
    assert max_output_tokens(cfg) <= 4096


def test_the_generation_budget_does_not_move_when_the_band_moves(tmp_path):
    """THE DECOUPLING, TESTED WHERE IT CAN ACTUALLY FAIL (review round 2, I1).

    The assertions above pass against the OLD derivation too, because the
    shipped config makes think_max + ANSWER_TOKEN_ALLOWANCE == 3000 + 1000 ==
    4000 == GENERATION_OUTPUT_TOKENS. A mutation that restores
    `band.think_max + ANSWER_TOKEN_ALLOWANCE` survives every one of them - the
    reviewer's M4, and it survived.

    So the coincidence has to be broken: move think_max in a real config and
    assert the CALL budget does not follow it. Under the old derivation this
    reads 2500; under the new one it stays 4000.
    """
    raw = DATA_CONFIG.read_text(encoding="utf-8")
    moved = raw.replace("think_min: 500, think_max: 3000", "think_min: 500, think_max: 1500")
    assert moved != raw, "the length_band line moved; update this fixture"
    path = tmp_path / "band_moved.yaml"
    path.write_text(moved, encoding="utf-8")
    shifted = load_build_config(str(path), allow_unpinned=True)

    assert shifted.build.length_band.think_max == 1500
    # The old derivation would give 1500 + 1000 = 2500.
    assert shifted.build.length_band.think_max + ANSWER_TOKEN_ALLOWANCE == 2500
    assert max_output_tokens(shifted) == 4000
    # ...and the inequality that replaced the identity now BINDS: a band this
    # small is comfortably covered, and the test above is what fails if the
    # band is ever raised past what the budget can carry.
    assert max_output_tokens(shifted) >= legal_reply_chars(shifted) / 4.24


def test_the_reply_budget_is_the_max_tokens_the_call_actually_sent(cfg):
    """A bound on a physical fact about the call, not a taste judgement: the
    worker sends max_tokens=max_output_tokens(cfg), and
    REPLY_BUDGET_CHARS_PER_TOKEN is the loosest chars-per-token MEASURED, so a
    well-behaved provider cannot exceed it.

    The constant used to be providers.CHARS_PER_TOKEN_LATIN (4.0) on the stated
    premise that "no tokenizer here gives a token more characters than that".
    The pilot measured 4.24-5.13, so the bound was too tight and the check
    fired on 53 compliant calls - 34 of which reported completion_tokens
    exactly equal to the max_tokens sent, i.e. the provider was billing
    correctly at the moment the check said it was not.
    """
    budget = reply_budget_chars(cfg)
    assert budget == int(max_output_tokens(cfg) * REPLY_BUDGET_CHARS_PER_TOKEN)
    assert REPLY_BUDGET_CHARS_PER_TOKEN > 5.13, "must clear the measured maximum"
    assert reply_over_budget(cfg, "a" * budget, "") == 0
    assert reply_over_budget(cfg, "a" * (budget - 1), "b") == 0
    assert reply_over_budget(cfg, "a" * budget, "b") == 1
    # ...and a normal candidate is nowhere near it, which is why enforcing it
    # is a no-op on every well-behaved provider.
    assert reply_over_budget(cfg, CLEAN_THINK, CLEAN_ANSWER) == 0


def test_a_reply_the_gates_pass_can_still_break_the_judge_sizing_premise(tmp_path, cfg, paths):
    """The premise behind judge_tokens_for_generator_window - "the candidate is
    at most max_output_tokens of reply" - is not something check_length_band
    tests. It bounds prompt + think + answer in chars/4 and the trace on its
    own; a short prompt leaves the whole remainder for the reply. This row
    passes all nine gates with a reply of more than twice the budget, and it is
    exactly the shape a provider that does not bill its reasoning channel
    against max_tokens would return. It goes back for a regeneration."""
    with make_store(tmp_path) as store:
        router = FakeRouter(cfg, {"generator": [chat_response(OVERSIZE_ANSWER, OVERSIZE_THINK)]})
        run(store, cfg, router, paths)
        task = only_task(store)
        gen = store.latest_generation(task["task_id"])

        # The gates - all of them - are happy with it.
        assert store.gates_for(gen["gen_id"]) == {gate: True for gate in gates.GATE_ORDER}
        # ...and the reply is most of the way to twice what was asked for. The
        # headroom is structural, not a fixture trick: the band's ceiling on
        # prompt + think + answer is total_max * 4 characters, and the reply
        # budget is a fraction of it, so any row with a short prompt has room
        # to spare.
        reply_chars = len(gen["think"]) + len(gen["answer"])
        assert reply_chars > reply_budget_chars(cfg)
        # The headroom NARROWED on 2026-08-18 and the assertions say so rather
        # than being loosened quietly: correcting the chars/token premise from
        # 4.0 to the measured 5.5 raised the budget 16,000 -> 22,000 against an
        # unchanged band ceiling of total_max * 4 = 32,768 characters, so the
        # window in which a gates-passing row can still breach the premise went
        # from ~1.79x the budget to ~1.30x on this fixture (28,627 reply
        # characters against 16,000 then and 22,000 now - the earlier "~1.5x"
        # here was written from the band ratio rather than measured, M-5). It
        # is a window, not a gap: the shape
        # this test exists for is still reachable, which is why the test still
        # has something to catch.
        assert cfg.build.length_band.total_max * 4 > reply_budget_chars(cfg)
        over = reply_over_budget(cfg, gen["think"], gen["answer"])

        assert task["state"] == "pending"
        assert task["disposition"] == f"regenerate:{REPLY_BUDGET_GATE}"
        event = json.loads(store.events("reply_over_budget")[0]["detail_json"])
        assert event["over_by"] == over
        assert event["budget_chars"] == reply_budget_chars(cfg)
        assert event["ref"] == "cerebras/gpt-oss-120b"


def test_a_permanent_gate_still_decides_an_over_budget_reply(tmp_path, cfg, paths):
    """An over-long reply is a shape problem and asks for a regeneration; a
    fabricated citation is a statement about the law and burns the seed.
    Nothing here promotes the second back into a retry."""
    with make_store(tmp_path) as store:
        long_fabrication = FABRICATED_ANSWER + " " + OVERSIZE_ANSWER
        router = FakeRouter(cfg, {"generator": [chat_response(long_fabrication, OVERSIZE_THINK)]})
        run(store, cfg, router, paths)
        task = only_task(store)
        assert task["state"] == "rejected"
        assert task["disposition"].startswith("reject:citations")
        assert REPLY_BUDGET_GATE in task["disposition"]


def test_a_missing_trace_parks_as_a_provider_fact_not_a_content_failure(
    tmp_path, cfg, paths
):
    """A generator with no reasoning channel is a fact about the POOL.

    CHANGED 2026-08-18. This used to assert `pending` + `regenerate:...`, i.e.
    a content problem worth two more attempts. Measured over the pilot: 43/43
    mistral/mistral-small-latest generations came back with no trace and
    0/176 cerebras/gpt-oss-120b did, so the row was never going to improve by
    being asked again - every retry bought the identical wall and the run
    filed ~20% of its spend as three content-gate failures (think_format,
    length_band, self_verification) that describe the harness, not the answer.

    A single live probe settled that there is nothing to extract: the message
    carries keys ['content','role','tool_calls'], content is a plain string
    with no [THINK]/<think> markers, and `prompt_mode: "reasoning"` is refused
    with 400 "Reasoning prompt mode is not enabled for this model" (code 3051).
    So the harness reports the provider fact and parks the row where
    tasks.REOPEN_STATES can bring it back, rather than spending the seed's
    attempts proving the same thing three times.

    The gates still ran and their rows are still stored - this suppresses no
    instrumentation, it only stops the row being counted as a rejection.
    """
    with make_store(tmp_path) as store:
        router = FakeRouter(cfg, {"generator": [chat_response(CLEAN_ANSWER, None)]})
        run(store, cfg, router, paths)
        task = only_task(store)
        assert task["state"] == GEN_UNROUTABLE_STATE
        assert task["disposition"] == "unroutable:no-reasoning-channel"

        # The provider-level fact is recorded, and names the ref responsible.
        event = json.loads(store.events("no_reasoning_channel")[0]["detail_json"])
        assert event["ref"] == "cerebras/gpt-oss-120b"
        assert "think_format" in event["gates_that_fired"]

        # ...and the gate rows are still there to be read.
        gen = store.latest_generation(task["task_id"])
        assert store.gates_for(gen["gen_id"])["think_format"] is False

        # Parked, not burned: the documented re-open path returns it to the
        # generator queue. It does NOT get its attempt budget back - the call
        # was made and billed, so the cycle has to be bounded. See
        # test_reopening_a_billed_park_is_bounded.
        assert reopen_tasks(store, [GEN_UNROUTABLE_STATE]) == {GEN_UNROUTABLE_STATE: 1}
        reopened = only_task(store)
        assert reopened["state"] == "pending"
        assert reopened["attempts"] == 1
        assert reopened["disposition"] == "reopened:unroutable:no-reasoning-channel"


def test_a_dead_generator_is_counted_as_parked_and_named(tmp_path, cfg, paths):
    """A PROVIDER THAT PARKS EVERY ROW MUST NOT MAKE THE BATCH LOOK HEALTHIER.

    REVIEW ROUND 2, I2. BatchStats.absorb keyed everything on the GATE
    disposition, so six rows parked for having no reasoning channel were
    reported as six ordinary `regenerate`s and nothing anywhere aggregated by
    provider. Since a parked row never reaches `rejected`, a generator that
    answered every call with no trace actually IMPROVED the visible numbers -
    the operator's only route to the truth was joining gate_result to
    generation.model by hand, which is how 43 pilot rows and ~20% of spend
    stayed invisible for a whole run.
    """
    with make_store(tmp_path, n_seeds=6, n_tasks=6) as store:
        traceless = chat_response(CLEAN_ANSWER, None)
        router = FakeRouter(cfg, {"generator": [traceless]})
        totals = run(store, cfg, router, paths, n_workers=6)

        assert totals["parked"] == {"cerebras/gpt-oss-120b": 6}
        # Not laundered through the gate counters.
        assert totals["gated_out"] == 0
        assert totals["dispositions"] == {}
        assert totals["gen_ok"] == 6

        stats = BatchStats()
        for _ in range(6):
            stats.absorb(
                GenResult(
                    task_id="t", attempt=1, ok=True,
                    ref=ModelRef(provider="mistral", model="mistral-small-latest"),
                    no_reasoning_channel=True,
                )
            )
        line = format_batch_line(1, stats, [])
        assert "parked{mistral/mistral-small-latest:6}" in line


def test_a_task_planned_against_an_edited_template_is_refused(tmp_path, cfg, paths):
    """THE PROVENANCE GUARD (review round 2, C1).

    task.prompt_sha is stamped at plan time and copied into every raw envelope
    and generation row, where it is the only record of which bytes the teacher
    saw. Nothing compared it to the live registry, so a template edit silently
    decoupled the two: the call went out under the NEW template and was filed
    under the OLD sha, pooling two different prompts under one label - which is
    fatal to any paraphrase A/B, and all 419 non-terminal pilot tasks were in
    exactly that state after the first fix round.

    The row is REFUSED, not re-stamped. Re-stamping would make the provenance
    agree with itself by destroying the evidence that it had disagreed.
    """
    with make_store(tmp_path) as store:
        task = only_task(store)
        planned_sha = task["prompt_sha"]
        # The template moves under the plan.
        store.conn.execute(
            "UPDATE task SET prompt_sha = ? WHERE task_id = ?",
            ("deadbeef0000", task["task_id"]),
        )
        store.conn.commit()

        router = FakeRouter(cfg)
        run(store, cfg, router, paths)

        moved = only_task(store)
        assert moved["state"] == STALE_PROMPT_STATE
        assert moved["disposition"] == (
            f"stale-prompt:{task['prompt_id']}:deadbeef0000!={planned_sha}"
        )
        # NOTHING WAS SPENT: no provider call, no generation row.
        assert router.calls_for("generator") == []
        assert store.latest_generation(task["task_id"]) is None

        event = json.loads(store.events("prompt_sha_mismatch")[0]["detail_json"])
        assert event["planned_sha"] == "deadbeef0000"
        assert event["live_sha"] == planned_sha

        # Restoring the template makes the row valid again, and the re-open
        # path is what brings it back.
        store.conn.execute(
            "UPDATE task SET prompt_sha = ? WHERE task_id = ?",
            (planned_sha, task["task_id"]),
        )
        store.conn.commit()
        assert reopen_tasks(store, [STALE_PROMPT_STATE]) == {STALE_PROMPT_STATE: 1}
        run(store, cfg, router, paths)
        assert only_task(store)["state"] == "judging"


def test_reopening_a_billed_park_is_bounded(tmp_path, cfg, paths):
    """A park that cost a call must not buy another one every cycle.

    REVIEW ROUND 2, I3. reopen_tasks reset `attempts` unconditionally and
    overwrote the disposition with `reopened:from-gen_unroutable`. For a
    no-reasoning-channel park - which is reached by MAKING the call, paying for
    it and finding no trace in the reply - that made the loop unbounded: every
    `--reopen gen_unroutable` bought one more live generation, and the
    overwritten disposition meant an expensive park was indistinguishable from
    a pool that was never reached.

    The loop now terminates: attempts survive the re-open, so MAX_ATTEMPTS
    cycles exhaust the row and the re-open stops offering it.
    """
    with make_store(tmp_path) as store:
        traceless = chat_response(CLEAN_ANSWER, None)
        router = FakeRouter(cfg, {"generator": [traceless]})

        calls = 0
        for _ in range(MAX_ATTEMPTS + 3):
            run(store, cfg, router, paths)
            calls = len(router.calls_for("generator"))
            if only_task(store)["state"] != GEN_UNROUTABLE_STATE:
                break
            if not reopen_tasks(store, [GEN_UNROUTABLE_STATE])[GEN_UNROUTABLE_STATE]:
                break

        assert calls == MAX_ATTEMPTS, (
            f"the reopen cycle bought {calls} live calls; it must stop at "
            f"{MAX_ATTEMPTS}"
        )
        task = only_task(store)
        assert task["attempts"] == MAX_ATTEMPTS
        # ...and the row is still parked, still saying what parked it.
        assert task["state"] == GEN_UNROUTABLE_STATE
        assert task["disposition"] == "unroutable:no-reasoning-channel"
        # A further re-open is refused rather than silently looping.
        assert reopen_tasks(store, [GEN_UNROUTABLE_STATE]) == {GEN_UNROUTABLE_STATE: 0}


def test_reopening_a_free_park_still_restores_the_budget(tmp_path, cfg, paths):
    """The blanket reset was right for the park it was written for, and I3 must
    not take that away: a keyless wave burns claims without ever reaching a
    provider, so those attempts were never spent on an answer."""
    with make_store(tmp_path) as store:
        _park_keyless(store, cfg, paths)
        assert only_task(store)["attempts"] > 0
        assert reopen_tasks(store, [GEN_UNROUTABLE_STATE]) == {GEN_UNROUTABLE_STATE: 1}
        task = only_task(store)
        assert task["state"] == "pending"
        assert task["attempts"] == 0
        assert task["disposition"] == f"reopened:from-{GEN_UNROUTABLE_STATE}"


def test_a_retry_sends_the_identical_request_and_logs_no_effort_bump(
    tmp_path, cfg, paths
):
    """INVERTED 2026-08-18 - this used to pin `calls[1]["params"] ==
    {"reasoning_effort": "high"}` and one effort_bump event per retry. The
    ladder is retired; see
    test_a_retry_re_rolls_the_same_request_and_never_bumps_effort for the
    measurement, and generate.EFFORT_LADDER_RETIRED for the numbers."""
    with make_store(tmp_path) as store:
        router = FakeRouter(cfg, {"generator": [chat_response(CLEAN_ANSWER, NO_CUE_THINK)]})
        run(store, cfg, router, paths)
        run(store, cfg, router, paths)
        calls = router.calls_for("generator")
        assert calls[0]["params"] == {}
        assert calls[1]["params"] == {}
        assert store.events("effort_bump") == []


def test_regeneration_is_exhausted_at_the_attempt_cap(tmp_path, cfg, paths):
    with make_store(tmp_path) as store:
        router = FakeRouter(cfg, {"generator": [chat_response(CLEAN_ANSWER, NO_CUE_THINK)]})
        for _ in range(MAX_ATTEMPTS):
            run(store, cfg, router, paths)
        task = only_task(store)
        assert task["attempts"] == MAX_ATTEMPTS
        assert task["state"] == "rejected"
        assert task["disposition"].startswith("exhausted:regenerate:")
        assert "self_verification" in task["disposition"]


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


def test_a_fleet_wide_provider_fault_parks_instead_of_rejecting(tmp_path, cfg, paths):
    """A rotated key, a 403, a 5xx outage or a 429 storm the client cannot
    ride out arrives as retryable=True with an EMPTY skip set, so neither
    `unroutable` nor `no_eligible_model` fires and three claims used to land
    the row in `rejected` - terminal, not re-openable, and counted as a
    legal-quality reject. Nothing about the answer was ever tested."""
    outage = ProviderError(
        "role 'generator': all 2 eligible model(s) failed; last: 503",
        status=503, provider="cerebras", model="gpt-oss-120b", retryable=True,
    )
    with make_store(tmp_path) as store:
        for _ in range(MAX_ATTEMPTS):
            run(store, cfg, FakeRouter(cfg, {"generator": [outage]}), paths)
        task = only_task(store)
        assert task["state"] == GEN_UNROUTABLE_STATE
        assert task["disposition"] == "exhausted:provider-fault"
        assert reopen_tasks(store, [GEN_UNROUTABLE_STATE]) == {GEN_UNROUTABLE_STATE: 1}


def test_provider_failure_at_the_cap_rejects(tmp_path, cfg, paths):
    """The payload class keeps the old ending, and it has to: a 400 with no
    context marker is OUR bug, it is not recoverable by re-opening the row,
    and burying it in a parking state would hide it behind a number the
    operator reads as "the pool was short"."""
    error = ProviderError("bad payload", status=400, provider="cerebras",
                          model="gpt-oss-120b", retryable=False)
    with make_store(tmp_path) as store:
        for _ in range(MAX_ATTEMPTS):
            run(store, cfg, FakeRouter(cfg, {"generator": [error]}), paths)
        assert only_task(store)["disposition"] == "exhausted:error"
        assert only_task(store)["state"] == "rejected"


def test_the_generation_attempt_cap_is_three_and_there_is_no_fourth_claim(tmp_path, cfg, paths):
    """Every exhaustion test here loops `for _ in range(MAX_ATTEMPTS)`, which
    is parameterised by the value under test and holds at 3 and at 99 alike.
    The number is load-bearing now that --reopen re-arms the counter: it is
    the only bound on what one row can spend across a reopen cycle, on a fleet
    that runs for days against hard daily caps. So: the literal, and a fourth
    claim that does not happen."""
    assert MAX_ATTEMPTS == 3
    with make_store(tmp_path) as store:
        # A trace that never doubts itself: check_self_verification fails the
        # row into a regeneration every time, which is the path that spends.
        # (A traceless reply used to be the stimulus here; since 2026-08-18 it
        # parks as a provider fact and spends nothing, which is the point of
        # that change and would make this test vacuous.)
        router = FakeRouter(cfg, {"generator": [chat_response(CLEAN_ANSWER, NO_CUE_THINK)]})
        for _ in range(MAX_ATTEMPTS + 2):
            run(store, cfg, router, paths)

        assert len(router.calls_for("generator")) == MAX_ATTEMPTS
        task = only_task(store)
        assert task["attempts"] == MAX_ATTEMPTS
        assert task["state"] == "rejected"


def test_run_workers_does_not_count_a_disposition_the_fence_refused(tmp_path, cfg, paths):
    """The other half of I2, driven rather than hand-fed. The test below feeds
    `absorb(landed=False)` literally - i.e. the value run_workers is supposed
    to DERIVE - so `landed = True` survives it. Here the lease really moves
    mid-pass and the batch totals have to show it."""
    with make_store(tmp_path) as store:
        proxy = StealsTheLease(store, at="record_generation")
        totals = run(proxy, cfg, FakeRouter(cfg), paths)

        assert proxy.stolen
        assert (totals["lost_leases"], totals["gen_ok"]) == (1, 0)
        assert totals["dispositions"] == {}
        # The tokens were spent whoever owns the row now.
        assert totals["prompt_tokens"] + totals["completion_tokens"] > 0
        # ...and the live holder still has the task, untouched by the loser.
        task = only_task(store)
        assert (task["claimed_by"], task["state"]) == ("thief-worker", "generating")


def test_a_disposition_the_fence_refused_is_not_counted_as_one(tmp_path, cfg, paths):
    """run_workers absorbed a result's disposition whether or not the write
    landed - the generator twin of the judge's counter bug. A batch that lost
    every lease reported a full batch of clean generations."""
    with make_store(tmp_path) as store:
        task = store.claim_tasks("stale-worker", 1)[0]
        result = asyncio.run(
            generate_once(store, cfg, FakeRouter(cfg), task, paths=paths)
        )
        assert result.ok
        store.set_task_state(task["task_id"], "pending")
        store.claim_tasks("live-worker", 1)

        stats = BatchStats()
        assert apply_gate_disposition(
            store, task, result, worker_id="stale-worker"
        ) == "lost-lease"
        stats.absorb(result, landed=False)
        assert stats.lost_leases == 1
        assert stats.gen_ok == 0
        # The tokens were still spent, so they are still counted.
        assert stats.tokens == result.prompt_tokens + result.completion_tokens


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

def test_budget_ok_falls_back_to_the_daily_ledger_when_nothing_is_observed(tmp_path, cfg):
    """With no provider observation the ledger still decides - but see the
    probe grant: the FIRST call after the ledger says exhausted is let through
    to go and ask the provider, because otherwise no call is made, no headers
    are seen, and the ledger decides forever."""
    with open_store(tmp_path, n_seeds=0) as store:
        budget_ok = budget_ok_for(store, cfg, quota=QuotaLedger())
        assert budget_ok("cerebras", "gpt-oss-120b", 1000) is True
        _, model = cfg.model_for(ModelRef("cerebras", "gpt-oss-120b"))
        store.record_usage(
            "cerebras", "gpt-oss-120b",
            prompt_tokens=int(model.limits["tpd"]), completion_tokens=0,
        )
        # One probe through...
        assert budget_ok("cerebras", "gpt-oss-120b", 1) is True
        assert len(store.events("budget_probe_grant")) == 1
        # ...and then the ledger holds the door.
        assert budget_ok("cerebras", "gpt-oss-120b", 1) is False
        assert budget_ok("cerebras", "gpt-oss-120b", 1) is False
        assert len(store.events("budget_probe_grant")) == 1
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
        assert line.startswith(
            "batch 1: claimed=1 gen-ok=1 gated-out=0 err=0 lost-lease=0 tokens=1700"
        )
        assert "cerebras/gpt-oss-120b spent=1.7k" in line


def test_streams_are_claimed_separately(tmp_path, cfg, paths):
    with make_store(tmp_path, n_seeds=4, n_tasks=2) as store:
        plan_wave(store, cfg, "curated_c2", 2, task_type_mix={"summarization": 1.0})
        run(store, cfg, FakeRouter(cfg), paths, streams=["synthesis"], n_workers=10)
        counts = store.task_counts()
        assert counts["judging"] == 2
        assert counts["pending"] == 2


# --------------------------------------------------------------------------
# Context routing for the GENERATOR.
#
# THE WINDOW IS NARROWED BY FIXTURE, NEVER BY THE SHIPPED CONFIG. Until
# 2026-08-19 the shipped cerebras generator declared 8192 and LONG_SEED_TEXT
# was sized to beat it, so these tests exercised the filter for free. The
# probed window is 131,072 and no prompt this build can produce comes near it,
# so the exclusion has to be CONSTRUCTED - cfg_with_context refuses if the
# (family, role) it was asked to narrow is not there, which is what stops
# these tests quietly measuring the shipped pool again.
# --------------------------------------------------------------------------

def _narrow_generator(cfg):
    """The config with the sole generator family cut back to the window the
    pilot actually ran against."""
    return cfg_with_context(
        cfg, family="gpt-oss", role="generator", max_context=NARROW_GENERATOR_CONTEXT
    )


def test_the_shipped_generator_window_holds_the_longest_row_this_build_makes(
    tmp_path, cfg, paths
):
    """The 2026-08-19 probe, asserted where it changes behaviour.

    LONG_SEED_TEXT needs 11,008 tokens of declared window. Against the stale
    8192 pin that emptied the generator pool and parked the row; against the
    probed 131,072 it is an ordinary prompt. This is the test that would have
    caught the pin, and it is the one that fails if anybody lowers it again.
    """
    with make_store(tmp_path, text=LONG_SEED_TEXT) as store:
        router = FakeRouter(cfg)
        run(store, cfg, router, paths)
        (attempt,) = router.calls_for("generator")
        assert attempt["est_tokens"] > NARROW_GENERATOR_CONTEXT
        assert attempt["exclude_families"] == frozenset()
        assert attempt["ref"] == ModelRef("cerebras", "gpt-oss-120b")
        assert only_task(store)["state"] == "judging"


def test_a_long_prompt_parks_when_no_generator_can_hold_it(tmp_path, cfg, paths):
    """WHAT THE 2026-08-18 DEMOTION COSTS, asserted rather than assumed.

    This used to divert to mistral. mistral is judge-only now - over 56
    re-pilot generations it produced zero gate-passing rows and failed
    irac_placement on 89% of them by outlining IRAC inside its trace - so a
    prompt the generator cannot hold has nowhere to go.

    The row PARKS and spends NOTHING: no call, recoverable via --reopen. That
    is the trade the demotion made, and it is the cheap side of it - the
    alternative was three billed attempts at an 89% failure rate.

    The window is narrowed by fixture: on the probed 131k pool no prompt the
    length band permits reaches this path, so the 3-of-100 figure the pilot
    measured was an artefact of the stale pin, not a property of the corpus.
    """
    narrow = _narrow_generator(cfg)
    with make_store(tmp_path, text=LONG_SEED_TEXT) as store:
        router = FakeRouter(narrow)
        run(store, narrow, router, paths)
        # The attempt is recorded, but `ref is None` - the context filter
        # emptied the pool before any provider was reached, so nothing was
        # spent.
        (attempt,) = router.calls_for("generator")
        assert attempt["est_tokens"] > NARROW_GENERATOR_CONTEXT
        assert attempt["exclude_families"] == frozenset({"gpt-oss"})
        assert attempt["ref"] is None
        task = only_task(store)
        assert task["state"] == GEN_UNROUTABLE_STATE
        assert task["disposition"] == "unroutable:generator"

    # ...and with a second generator family present it still diverts, so the
    # routing itself is intact and only the pool changed.
    two = _narrow_generator(cfg_with_two_generator_families(cfg))
    with make_store(tmp_path / "two", text=LONG_SEED_TEXT) as store:
        router = FakeRouter(two)
        run(store, two, router, paths)
        call = router.calls_for("generator")[0]
        assert call["est_tokens"] > NARROW_GENERATOR_CONTEXT
        assert "gpt-oss" in call["exclude_families"]
        assert call["ref"] == ModelRef("mistral", "mistral-small-latest")
        assert only_task(store)["state"] == "judging"


def test_a_short_prompt_excludes_no_generator(tmp_path, cfg, paths):
    with make_store(tmp_path) as store:
        router = FakeRouter(cfg)
        run(store, cfg, router, paths)
        call = router.calls_for("generator")[0]
        assert call["est_tokens"] <= 8192
        assert call["exclude_families"] == frozenset()
        assert call["ref"] == ModelRef("cerebras", "gpt-oss-120b")


def test_the_generator_opts_mistral_into_reasoning_and_leaves_gpt_oss_alone(
    tmp_path, cfg, paths
):
    """The per-ref hook is what makes an OPT-IN reasoning channel role-correct.

    Mistral Small 4 only reasons when asked (live probe 2026-08-18: with
    `reasoning_effort` the content is a list carrying a thinking chunk; without
    it, a plain string and no trace anywhere). gpt-oss reasons by default and
    declares `medium` in the config, so sending it anything here would override
    a configured value for no reason.

    THE HOOK SHAPE IS THE POINT, and it is why this is asserted per route
    rather than globally: the params are chosen for the ref the Router is ABOUT
    TO CALL, not the one it would have picked first. A long prompt fails over
    from the 8k cerebras model to mistral, and if the parameter had been chosen
    for cerebras the mistral call would carry a field it does not accept - a
    400, which providers.py reads as our payload being broken everywhere and
    raises through WITHOUT failing over, turning every failover into a dead
    task. The long-prompt route below is exactly that failover.
    """
    # mistral is judge-only in the shipped config since 2026-08-18, so the
    # generator half of this contract is exercised against the two-family
    # fixture. The CODE is deliberately kept - it is family-keyed, harmless
    # while the family serves no generator role, and it records the finding
    # that Small 4's reasoning channel is opt-in.
    # The window is narrowed by fixture: since the 2026-08-19 probe the
    # shipped cerebras generator holds 131k, so LONG_SEED_TEXT no longer
    # forces the failover this contract is about. The failover is the
    # subject, so it is constructed rather than waited for.
    two = _narrow_generator(cfg_with_two_generator_families(cfg))
    with make_store(tmp_path, text=LONG_SEED_TEXT) as store:
        router = FakeRouter(two)
        task = only_task(store)
        asyncio.run(generate_once(store, two, router, task, paths=paths, attempt=2))
        call = router.calls_for("generator")[0]
        assert call["ref"].provider == "mistral"
        assert call["params"] == {"reasoning_effort": "high"}
        assert store.events("effort_bump") != []

    with make_store(tmp_path / "short") as store:
        router = FakeRouter(cfg)
        task = only_task(store)
        asyncio.run(generate_once(store, cfg, router, task, paths=paths, attempt=2))
        call = router.calls_for("generator")[0]
        assert call["ref"].provider == "cerebras"
        assert call["params"] == {}
        assert store.events("effort_bump") == []


def test_a_row_no_generator_can_hold_parks_recoverably(tmp_path, cfg, paths):
    """No eligible model is a fact about the ROW, not the moment: three more
    claims would meet the same wall. But it is not a fact about the LAW
    either, so it parks in its own state instead of landing in `rejected`
    alongside the answers that were legally wrong - and the write is fenced by
    the lease the claim actually handed out."""
    with make_store(tmp_path) as store:
        # A seed longer than every generator's context window - sized
        # against the REAL 131k window, not a narrowed one, because "no
        # generator can hold this row" is exactly what is being asserted.
        store.upsert_seeds(
            [{**seed_rows(1)[0], "text": OVERSIZE_SEED_TEXT}]
        )
        worker = "gen-fence-1"
        task = store.claim_tasks(worker, 1)[0]
        router = FakeRouter(cfg)
        result = asyncio.run(generate_once(store, cfg, router, task, paths=paths))
        assert result.ok is False
        assert result.unroutable is True
        assert apply_gate_disposition(
            store, task, result, worker_id=worker
        ) == GEN_UNROUTABLE_STATE
        assert only_task(store)["state"] == GEN_UNROUTABLE_STATE
        assert only_task(store)["disposition"] == "unroutable:generator"
        assert only_task(store)["claimed_by"] is None
        event = json.loads(store.events("generation_error")[0]["detail_json"])
        assert event["unroutable"] is True
        # One generator family since the 2026-08-18 demotion.
        assert set(event["excluded_families"]) == {"gpt-oss"}


def test_a_stale_worker_cannot_park_a_task_it_no_longer_holds(tmp_path, cfg, paths):
    """The fence is the point of passing worker_id at all: with it disabled
    the permanent-close path was never tested against a live lease."""
    with make_store(tmp_path) as store:
        store.upsert_seeds([{**seed_rows(1)[0], "text": OVERSIZE_SEED_TEXT}])
        task = store.claim_tasks("stale-worker", 1)[0]
        result = asyncio.run(generate_once(store, cfg, FakeRouter(cfg), task, paths=paths))
        assert result.unroutable is True
        # The lease expired and somebody else took the task while the call ran.
        store.set_task_state(task["task_id"], "pending")
        store.claim_tasks("live-worker", 1)
        assert apply_gate_disposition(
            store, task, result, worker_id="stale-worker"
        ) == "lost-lease"
        assert only_task(store)["claimed_by"] == "live-worker"
        assert only_task(store)["state"] == "generating"


# --------------------------------------------------------------------------
# R2-C1: a fleet with no keys must never close the wave it cannot start.
# --------------------------------------------------------------------------

def test_a_keyless_batch_leaves_the_wave_intact(tmp_path, cfg, paths):
    """The operator's .env carries no provider keys yet, so this is what the
    FIRST pilot launch does. It used to take {'pending': 3} to
    {'rejected': 3} with zero calls made, and `rejected` is terminal."""
    with make_store(tmp_path, n_seeds=3, n_tasks=3) as store:
        router = FakeRouter(cfg, missing_keys={"cerebras", "mistral"})
        totals = run(store, cfg, router, paths, n_workers=3)
        assert totals["gen_ok"] == 0
        assert totals["errors"] == 3
        assert store.task_counts() == {"pending": 3}
        assert all(call["ref"] is None for call in router.calls_for("generator"))
        event = json.loads(store.events("generation_error")[0]["detail_json"])
        assert event["unroutable"] is False
        assert event["skipped"] == ["missing-key"]


def test_a_missing_key_is_never_a_row_shaped_failure(tmp_path, cfg, paths):
    with make_store(tmp_path) as store:
        router = FakeRouter(cfg, missing_keys={"cerebras", "mistral"})
        result = asyncio.run(
            generate_once(store, cfg, router, only_task(store), paths=paths)
        )
        assert result.unroutable is False
        assert result.no_eligible_model is True
        assert result.route_skips == ("missing-key",)


def test_a_keyless_task_parks_recoverably_at_the_attempt_cap(tmp_path, cfg, paths):
    """Even the exhausted end of the keyless path stays out of `rejected`:
    nothing about the row was ever judged, so it must not be counted as a
    reject, and re-opening it must be able to bring it back."""
    with make_store(tmp_path) as store:
        router = FakeRouter(cfg, missing_keys={"cerebras", "mistral"})
        for _ in range(MAX_ATTEMPTS):
            run(store, cfg, router, paths)
        task = only_task(store)
        assert task["state"] == GEN_UNROUTABLE_STATE
        assert task["disposition"].startswith("exhausted:unroutable")
        assert "missing-key" in task["disposition"]


def test_a_transient_pool_skip_still_re_queues(tmp_path, cfg, paths):
    """Cooling and over-budget lift on their own, so the row goes back to the
    queue rather than parking."""
    with make_store(tmp_path) as store:
        router = FakeRouter(
            cfg,
            cooling={"cerebras/gpt-oss-120b"},
            over_budget={"mistral/mistral-small-latest"},
        )
        result = asyncio.run(
            generate_once(store, cfg, router, only_task(store), paths=paths)
        )
        assert result.unroutable is False
        run(store, cfg, router, paths)
        assert only_task(store)["state"] == "pending"


# --------------------------------------------------------------------------
# R3-C1: --reopen must hand the row back with a budget, not just a state.
# --------------------------------------------------------------------------

def _park_keyless(store, cfg, paths):
    """The motivating park: nothing routable, so the row exhausts unspent."""
    router = FakeRouter(cfg, missing_keys={"cerebras", "mistral"})
    for _ in range(MAX_ATTEMPTS):
        run(store, cfg, router, paths)
    task = only_task(store)
    assert (task["state"], task["attempts"]) == (GEN_UNROUTABLE_STATE, MAX_ATTEMPTS)
    return task


def test_a_reopened_row_survives_the_first_failure_after_recovery(tmp_path, cfg, paths):
    """R3-C1. The park happens AT the cap, so a re-open that restores only the
    state hands the row back already exhausted: one 429 - the ordinary weather
    of a free tier - closed it as `rejected`, which is deliberately not
    re-openable and has already spent its per-seed slot."""
    with make_store(tmp_path) as store:
        _park_keyless(store, cfg, paths)
        reopen_tasks(store, [GEN_UNROUTABLE_STATE])
        assert only_task(store)["attempts"] == 0

        error = ProviderError(
            "429 everywhere", status=429, provider="cerebras", model="gpt-oss-120b",
            retryable=True,
        )
        run(store, cfg, FakeRouter(cfg, {"generator": [error]}), paths)
        task = only_task(store)
        assert task["state"] == "pending"
        assert task["attempts"] == 1


def test_a_reopened_row_still_buys_the_regenerations_it_never_used(tmp_path, cfg, paths):
    """The worse half of R3-C1: a reply the gates send back is the ordinary
    free-tier outcome, and post-reopen it cost a PAID generation and then
    rejected with zero regenerations - as `exhausted:regenerate:...`, i.e.
    counted as a legal-quality reject.

    The stimulus changed on 2026-08-18 from a traceless reply to one that
    fails check_self_verification. A traceless reply is no longer a content
    regeneration at all - it parks as a provider fact and spends nothing - so
    it can no longer demonstrate the attempt accounting this test is about."""
    with make_store(tmp_path) as store:
        _park_keyless(store, cfg, paths)
        reopen_tasks(store, [GEN_UNROUTABLE_STATE])

        sent_back = chat_response(CLEAN_ANSWER, reasoning=NO_CUE_THINK)
        run(store, cfg, FakeRouter(cfg, {"generator": [sent_back]}), paths)
        task = only_task(store)
        assert task["state"] == "pending"
        assert task["disposition"].startswith("regenerate:")
        assert "self_verification" in task["disposition"]


# --------------------------------------------------------------------------
# R2-C2: a 400 must not burn the row, and the estimate needs headroom.
# --------------------------------------------------------------------------

def test_a_context_overflow_at_every_model_parks_instead_of_rejecting(tmp_path, cfg, paths):
    """A 400 context_length_exceeded used to close the task on claim #1 - the
    exact failure the generator's context routing was added to prevent."""
    overflow = ProviderError(
        "role 'generator': all 2 eligible model(s) failed; last: 400 context_length_exceeded",
        status=400,
        provider="cerebras",
        model="gpt-oss-120b",
        retryable=False,
        context_exceeded=True,
    )
    with make_store(tmp_path) as store:
        router = FakeRouter(cfg, {"generator": [overflow]})
        worker = "gen-1"
        task = store.claim_tasks(worker, 1)[0]
        result = asyncio.run(generate_once(store, cfg, router, task, paths=paths))
        assert result.unroutable is True
        assert apply_gate_disposition(
            store, task, result, worker_id=worker
        ) == GEN_UNROUTABLE_STATE
        assert only_task(store)["state"] != "rejected"


def test_a_plain_payload_400_costs_attempts_not_the_row(tmp_path, cfg, paths):
    """A genuine payload bug is bounded by the attempt cap like any other
    failed call - it never closes the task on the first claim."""
    bad_payload = ProviderError(
        "unknown parameter", status=400, provider="cerebras", model="gpt-oss-120b",
        retryable=False,
    )
    with make_store(tmp_path) as store:
        router = FakeRouter(cfg, {"generator": [bad_payload]})
        run(store, cfg, router, paths)
        assert only_task(store)["state"] == "pending"
        assert only_task(store)["attempts"] == 1


def test_the_context_estimate_routes_devanagari_past_a_generator_latin_fits(
    tmp_path, cfg, paths
):
    """Same character count, different script: chars/4 says both fit, and for
    the Devanagari one that is a 400 nobody fails over.

    Run against a fixture-narrowed window. The property is about the ESTIMATE
    - that Devanagari is charged 2-4x harder than Latin - and that property
    needs a window the Latin text clears and the Devanagari text does not.
    The shipped 131k window is cleared by both, so at the shipped size this
    test would pass while measuring nothing.
    """
    narrow = _narrow_generator(cfg)
    latin = "the accused was convicted under section 302 of the code " * 90
    devanagari = "अभियुक्त को भारतीय दंड संहिता की धारा तीन सौ दो के अंतर्गत " * 90
    assert abs(len(latin) - len(devanagari)) < len(latin) * 0.25

    with make_store(tmp_path, text=latin) as store:
        router = FakeRouter(narrow)
        asyncio.run(generate_once(store, narrow, router, only_task(store), paths=paths))
        assert router.calls_for("generator")[0]["ref"].provider == "cerebras"

    with make_store(tmp_path / "indic", text=devanagari) as store:
        router = FakeRouter(narrow)
        result = asyncio.run(
            generate_once(store, narrow, router, only_task(store), paths=paths)
        )
        # The script-aware estimate still excludes the narrowed family - that
        # is the property under test - but since the 2026-08-18 demotion there
        # is no second generator to receive the row, so it parks unspent
        # instead of diverting. See
        # test_a_long_prompt_parks_when_no_generator_can_hold_it.
        (attempt,) = router.calls_for("generator")
        assert attempt["ref"] is None
        assert result.unroutable is True
        event = json.loads(store.events("generation_error")[0]["detail_json"])
        assert "gpt-oss" in event["excluded_families"]

    # ...and on the SHIPPED window both scripts route, which is the 2026-08-19
    # probe showing up as behaviour rather than as a config line.
    with make_store(tmp_path / "shipped", text=devanagari) as store:
        router = FakeRouter(cfg)
        asyncio.run(generate_once(store, cfg, router, only_task(store), paths=paths))
        assert router.calls_for("generator")[0]["ref"].provider == "cerebras"


def test_the_gate_estimate_is_not_moved_by_the_routing_estimate(tmp_path, cfg):
    """The length band is calibrated in the gates' chars/4 currency. The
    routing estimate is deliberately more pessimistic, and mixing the two
    would silently re-scale every length gate in the build."""
    with make_store(tmp_path) as store:
        task = only_task(store)
        bundle = build_prompt(cfg, task, store.get_seed(task["seed_id"]))
        assert bundle.prompt_est_tokens == sum(
            len(m.get("content") or "") for m in bundle.messages
        ) // 4
        assert bundle.context_est_tokens > bundle.prompt_est_tokens


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


@pytest.mark.parametrize("missing", ["offence_date", "proceeding_started"])
def test_transition_refuses_a_seed_without_its_dates(tmp_path, cfg, paths, missing):
    """Each date is required on its own account: check_temporal picks the
    substantive code from the offence date and the PROCEDURAL code from the
    proceeding date, so a test that strips both proves nothing about either."""
    partial = {k: v for k, v in TRANSITION_META.items() if k != missing}
    with make_store(
        tmp_path, stream="transition", mix={"transition": 1.0}, meta=partial
    ) as store:
        task = only_task(store)
        with pytest.raises(SlotError, match=missing):
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


# --------------------------------------------------------------------------
# The startup preflight: what the fleet refuses to begin.
# --------------------------------------------------------------------------

def test_the_fleet_refuses_to_start_without_a_key_for_a_routed_role(tmp_path, cfg, monkeypatch):
    """"loaded 0 key(s) from .env" and then running anyway is how a wave gets
    claimed, failed and reported one row at a time instead of once."""
    for env in ("CEREBRAS_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    refusals, _ = preflight_messages(cfg, ("generator",))
    assert any("routing.generator has no usable API key" in line for line in refusals)
    assert any("CEREBRAS_API_KEY" in line for line in refusals)
    # ...and no override exists for it: there is nothing to override.
    forced, _ = preflight_messages(cfg, ("generator",), allow_pool_gaps=True)
    assert any("no usable API key" in line for line in forced)


def test_the_fleet_refuses_to_start_with_a_judge_slot_it_cannot_fill(cfg, monkeypatch):
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    cfg = cfg_with_two_generator_families(cfg)
    for env in ("CEREBRAS_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(env, "sk-test")
    # Withheld EXPLICITLY: the paid backstop is what fills slot B here, so a
    # machine that happens to export it would turn this into a test of nothing.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    refusals, warnings = preflight_messages(cfg, ("generator",))
    assert any("routing.judge slot b" in line for line in refusals)
    # A tiebreak gap warns rather than refuses, because it has a defined
    # fallback (judge.py decides on the two judges). The SHIPPED pool no longer
    # has one to demonstrate: gemma was the family that separation left and a
    # stale 8192 removed on length, and the 2026-08-19 probe put it at 131k.
    # Narrow it back and the warning returns, unchanged in kind.
    assert not any("routing.tiebreak" in line for line in warnings)
    assert not any("routing.tiebreak" in line for line in refusals)
    narrow_tb = cfg_with_context(cfg, family="gemma", role="tiebreak", max_context=8192)
    _, tb_warnings = preflight_messages(narrow_tb, ("generator",))
    assert any("routing.tiebreak" in line for line in tb_warnings)
    assert not any(
        "routing.tiebreak" in line
        for line in preflight_messages(narrow_tb, ("generator",))[0]
    )
    # The judge gap SURVIVES the override, and that is the 2026-08-18 change
    # rather than a regression: the slot-B candidate a short mistral row used
    # to have here was cerebras/zai-glm-4.7, and it was retired as archived.
    # With it gone the slot is empty at every row size, and a flag whose whole
    # justification is "the short rows still run" has no short rows to run.
    # test_a_two_generator_judge_gap_is_a_key_the_override_cannot_cover takes
    # that apart; here it is enough that the refusal does not move.
    allowed, _ = preflight_messages(cfg, ("generator",), allow_pool_gaps=True)
    assert any("routing.judge slot b" in line for line in allowed)


def test_a_widened_pool_starts_clean(cfg, monkeypatch):
    for env in ("CEREBRAS_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(env, "sk-test")
    refusals, warnings = preflight_messages(cfg_with_fourth_judge_family(cfg), ("generator",))
    assert (refusals, warnings) == ([], [])


def test_the_generator_cli_exits_rather_than_claiming_anything(tmp_path, cfg, monkeypatch, capsys):
    for env in ("CEREBRAS_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr("tuned.data.providers.load_dotenv_keys", lambda path=None: 0)
    config_path = temp_config(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        generate_main(["--config", config_path, "--max-batches", "1"])
    assert excinfo.value.code == 2
    out = capsys.readouterr().out
    assert "REFUSING" in out
    assert "routing.generator has no usable API key" in out
    # Nothing was opened, claimed or spent.
    assert not (tmp_path / "build" / "state").exists()


def test_the_preflight_refuses_a_16k_fourth_family_judge(cfg, monkeypatch):
    """R3-C2 end to end. 16k is above the >= 11520 the preflight used to
    print, so this config STARTED: a Devanagari row then passed the length
    gate, paid for judge A and parked in judge_unroutable at needed=14042.
    Many free-tier candidates for the pending operator decision are 16k."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    cfg = cfg_with_two_generator_families(cfg)
    for env in ("CEREBRAS_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(env, "sk-test")
    sixteen_k = cfg_with_fourth_judge_family(cfg, max_context=16384)
    refusals, _ = preflight_messages(sixteen_k, ("generator",))
    assert any("routing.judge" in line for line in refusals)
    # ...and the same config with a 32k model in the same slot starts clean.
    thirty_two_k = cfg_with_fourth_judge_family(cfg, max_context=32768)
    assert preflight_messages(thirty_two_k, ("generator",)) == ([], [])


def test_allow_pool_gaps_cannot_override_a_gap_no_row_size_escapes(cfg, monkeypatch):
    """R4-C1. The flag's justification - "running short rows while a key is
    pending is a real choice" - is true of a CONTEXT gap and false of this
    one: an unkeyed judge family is skipped at every size, so a mistral row
    has no slot B whatever its length.

    IT IS NO LONGER "the likely first launch", and that clause was retired on
    2026-08-18 rather than reworded: it rested on the shipped config printing a
    gap that told the operator to reach for the flag. Measured on the shipped
    config with the three free keys and no OPENAI_API_KEY, the preflight now
    returns 0 refusals and 1 tiebreak warning - there is nothing there to
    reach for the flag about. What this test measures is the fixture pool
    below, and the honest reason to keep measuring it is that the flag's
    contract has to hold for the pool an operator assembles next."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    cfg = cfg_with_two_generator_families(cfg)
    for env in ("CEREBRAS_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.setenv(env, "sk-test")
    for env in ("GROQ_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    widened = cfg_with_fourth_judge_family(cfg)  # the new judge is a groq model
    # ...plus the KEYED 8k judge the shipped pool carried until 2026-08-18. It
    # is what makes the contrast this test is about expressible at all: with
    # one keyed judge family, slot B is empty at every size for BOTH generator
    # families, every gap classifies unservable, and "the gap the override IS
    # for" has no instance left. See cfg_with_extra_judge.
    widened = cfg_with_extra_judge(
        widened, provider="cerebras", family="small", model_id="small-judge", max_context=8192
    )

    forced, warnings = preflight_messages(
        widened, GENERATOR_PREFLIGHT_ROLES, allow_pool_gaps=True
    )
    assert any("routing.judge" in line and "GROQ_API_KEY" in line for line in forced)
    assert any("no row size is servable" in line for line in forced)
    # The gap the override IS for still moves to warnings under the flag.
    assert any("routing.judge" in line for line in warnings)

    # ...and the same config with the key set starts clean, so the refusal is
    # about the key and not about the flag.
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    assert preflight_messages(widened, GENERATOR_PREFLIGHT_ROLES) == ([], [])


def test_a_two_generator_judge_gap_is_a_key_the_override_cannot_cover(cfg, monkeypatch):
    """The other half of R4-C1, INVERTED on 2026-08-18 - and kept inverted
    rather than deleted, because the inversion is what an operator assembling
    this pool has to know.

    It used to read: this gap is a MODEL the operator is sourcing, not a key,
    so short rows really do route and `--allow-pool-gaps` opens a pilot. What
    made those short rows route was cerebras/zai-glm-4.7 - a keyed 8k judge
    that could take slot B on a short row from the mistral generator - and it
    was retired as archived. With it gone the slot is empty at EVERY row size
    for an operator who has not funded OPENAI_API_KEY, so the override cannot
    open a pilot on this pool any more. The key can.

    NOT THE SHIPPED CONFIG'S GAP, and the name used to say it was. Measured
    2026-08-18: the shipped one-generator config with CEREBRAS/MISTRAL/GROQ
    keyed and OPENAI_API_KEY unset returns 0 refusals and 1 tiebreak warning.
    The gap below is reachable only from a MISTRAL generation, which is what
    removes mistral from the judge pool and empties slot B, and mistral has
    been judge-only since that day - so the second generator family is
    supplied by the fixture and the gap belongs to the fixture, not to what
    ships."""
    # See cfg_with_two_generator_families: this property is about the ALGORITHM
    # that walks generator families, and one family cannot exercise it.
    cfg = cfg_with_two_generator_families(cfg)
    for env in ("CEREBRAS_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(env, "sk-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    refusals, _ = preflight_messages(cfg, GENERATOR_PREFLIGHT_ROLES)
    assert any("routing.judge slot b" in line for line in refusals)
    forced, _ = preflight_messages(cfg, GENERATOR_PREFLIGHT_ROLES, allow_pool_gaps=True)
    assert any("no row size is servable" in line for line in forced)
    assert any("OPENAI_API_KEY" in line for line in forced)
    # ...and the key really is the fix, so the refusal is about the pool and
    # not about the flag.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert preflight_messages(cfg, GENERATOR_PREFLIGHT_ROLES)[0] == []


def test_the_preflight_advises_one_model_that_closes_every_gap_it_reports(cfg, monkeypatch):
    """The advice is what the operator is choosing a model against right now.
    It was judge-only: the preflight sizes the tiebreak separately (23,733 vs
    23,729), so a model of EXACTLY the advised size closed the judge gap and
    opened a tiebreak warning. Driven through preflight_messages, which is the
    production call - the old pinning test called pool_gaps WITHOUT
    tiebreak_needed_tokens and could not see it."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    cfg = cfg_with_two_generator_families(cfg)
    for env in ("CEREBRAS_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(env, "sk-test")
    refusals, warnings = preflight_messages(cfg, GENERATOR_PREFLIGHT_ROLES)
    advised = {
        int(line.split("max_context >= ")[1].split()[0]) for line in refusals + warnings
    }
    assert len(advised) == 1, "the operator adds ONE model; it needs ONE number"
    advice = advised.pop()

    exact = cfg_with_fourth_judge_family(cfg, max_context=advice)
    assert preflight_messages(exact, GENERATOR_PREFLIGHT_ROLES) == ([], [])
    short = cfg_with_fourth_judge_family(cfg, max_context=advice - 1)
    assert preflight_messages(short, GENERATOR_PREFLIGHT_ROLES) != ([], [])


def test_the_preflight_checks_each_generator_family_at_its_own_window(cfg, monkeypatch):
    """A NARROW generator is diverted long before the length band's longest
    row, so a judge gap reported for it at that length is a refusal about a
    combination that cannot occur. The 32k generator's gap is real and stays.

    The probe judge is 26k, not 20k, since 2026-08-18 (review round 2, I6):
    correcting the reply conversion from 4.0 to the measured 5.5 chars/token
    raised the 8k-window check from 15,104 to 19,104 tokens (23,880 of required
    context), so a 20k judge no longer clears it and this test would be
    asserting the absence of a refusal for the wrong reason.

    THE NARROW WINDOW IS NOW A FIXTURE. Until 2026-08-19 the shipped cerebras
    generator supplied it; the probe put that model at 131k, which narrows
    nothing, so the suppression this test is about had no family left to act
    on and the test was asserting the absence of a refusal that could not have
    been raised. Narrowed explicitly, the property is measured again.
    """
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    cfg = _narrow_generator(cfg_with_two_generator_families(cfg))
    for env in ("CEREBRAS_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(env, "sk-test")
    patched = cfg_with_context(cfg, family="qwen", role="judge", max_context=26000)
    refusals, _ = preflight_messages(patched, GENERATOR_PREFLIGHT_ROLES)
    assert any("a mistral generation" in line for line in refusals)
    assert not any("a gpt-oss generation" in line for line in refusals)


@pytest.mark.parametrize("generator_window", [8192, 131072])
def test_the_preflight_sizes_the_tiebreak_with_the_tiebreak_prompt(
    cfg, monkeypatch, generator_window
):
    """The judge and the tiebreak are different prompts and the preflight
    passes both sizes - the flat worst case AND the per-window hook. Drop
    either and the tiebreak is checked against a prompt nobody sends. They are
    four tokens apart on this config, so the pin is a model that sits between
    the two requirements; the parameters put the generator on each side of the
    band, because the flat number only reaches a family whose window does not
    cap it."""
    for env in ("CEREBRAS_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(env, "sk-test")
    base = cfg_with_context(
        cfg, family="gpt-oss", role="generator", max_context=generator_window
    )
    sizer = judge_sizer(base)
    judge_required = required_context(sizer(generator_window, "judge"))
    tiebreak_required = required_context(sizer(generator_window, "tiebreak"))
    assert judge_required < tiebreak_required

    tight = cfg_with_split_pools(
        base, judge_context=131072, tiebreak_context=tiebreak_required - 1
    )
    refusals, warnings = preflight_messages(tight, GENERATOR_PREFLIGHT_ROLES)
    assert refusals == []
    assert any("routing.tiebreak" in line for line in warnings)

    opened = cfg_with_split_pools(base, judge_context=131072, tiebreak_context=tiebreak_required)
    assert preflight_messages(opened, GENERATOR_PREFLIGHT_ROLES) == ([], [])


def test_the_preflight_refuses_a_judge_slot_that_is_only_missing_a_key(cfg, monkeypatch):
    """R3-C3 end to end, in the shape the operator will actually meet it:
    keys arrive piecemeal, the fourth-family judge lands behind the provider
    whose key has not arrived, and every routed role still passes
    unkeyed_roles because one of its refs IS keyed. The fleet used to start
    and buy judge A for every row in the wave."""
    for env in ("CEREBRAS_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.setenv(env, "sk-test")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    widened = cfg_with_fourth_judge_family(cfg)  # the new judge is a groq model
    refusals, _ = preflight_messages(widened, ("generator", "judge", "tiebreak"))
    assert any("routing.judge" in line and "GROQ_API_KEY" in line for line in refusals)


def test_the_generation_fleet_checks_the_judging_keys_too(cfg, monkeypatch):
    """generate.main's own comment says it checks the judge pool "because
    filling a queue no judge pool can drain is money spent on rows that will
    park" - which is just as true of the judge's KEYS. It passed only
    ("generator",)."""
    for env in ("CEREBRAS_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    refusals, _ = preflight_messages(cfg, GENERATOR_PREFLIGHT_ROLES)
    assert any("routing.generator has no usable API key" in line for line in refusals)
    assert any("routing.judge has no usable API key" in line for line in refusals)
    assert any("routing.tiebreak has no usable API key" in line for line in refusals)


def test_the_judge_reply_allowance_is_shared_with_the_preflight():
    """The preflight sizes the judge pool from providers.py's copy of the
    judge's reply budget; a drift between the two makes the check measure a
    prompt nobody sends."""
    from tuned.data.judge import JUDGE_MAX_TOKENS
    from tuned.data.providers import DEFAULT_JUDGE_REPLY_TOKENS

    assert JUDGE_MAX_TOKENS == DEFAULT_JUDGE_REPLY_TOKENS


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


# --------------------------------------------------------------------------
# The budget gate's authority: the provider's own headers beat our ledger.
# --------------------------------------------------------------------------
# The live observation that produced this work, 2026-08-18: our ledger read
# 1,015,901 tokens spent against a configured tpd of 1,000,000 and refused to
# spend anything, while the provider answered 200 to a 12-token probe with
# x-ratelimit-remaining-tokens-day: 458408 of 1000000 and
# x-ratelimit-remaining-requests-day: 2399 of 2400.
CEREBRAS_HEADERS_WITH_BUDGET = {
    "x-ratelimit-limit-tokens-day": "1000000",
    "x-ratelimit-remaining-tokens-day": "458408",
    "x-ratelimit-limit-requests-day": "2400",
    "x-ratelimit-remaining-requests-day": "2399",
    "x-ratelimit-remaining-tokens-minute": "29988",
}
CEREBRAS_HEADERS_EXHAUSTED = {
    "x-ratelimit-limit-tokens-day": "1000000",
    "x-ratelimit-remaining-tokens-day": "0",
    "x-ratelimit-limit-requests-day": "2400",
    "x-ratelimit-remaining-requests-day": "1200",
}


def _exhaust_ledger(store, cfg, provider="cerebras", model="gpt-oss-120b"):
    _, model_cfg = cfg.model_for(ModelRef(provider, model))
    store.record_usage(
        provider, model,
        prompt_tokens=int(model_cfg.limits["tpd"]) + 15_901, completion_tokens=0,
    )


def test_the_gate_opens_when_the_provider_says_budget_remains(tmp_path, cfg):
    """THE MEASURED FAILURE, as a test. Ledger over cap, provider says 458,408
    tokens left - the fleet must spend, because the provider is the authority
    on its own window and our UTC-day sum is only a proxy for it."""
    quota = QuotaLedger()
    with open_store(tmp_path, n_seeds=0) as store:
        _exhaust_ledger(store, cfg)
        assert store.reserve_budget(
            "cerebras", "gpt-oss-120b", 1,
            limits=cfg.model_for(ModelRef("cerebras", "gpt-oss-120b"))[1].limits,
        ) is False

        quota.record("cerebras", "gpt-oss-120b", CEREBRAS_HEADERS_WITH_BUDGET)
        budget_ok = budget_ok_for(store, cfg, quota=quota)
        assert budget_ok("cerebras", "gpt-oss-120b", 4000) is True
        # ...and it is not blanket permission: a call bigger than what remains
        # is still refused, on the provider's number.
        assert budget_ok("cerebras", "gpt-oss-120b", 500_000) is False


def test_the_gate_closes_when_the_provider_says_exhausted(tmp_path, cfg):
    """The other direction, and the one that protects the account: the ledger
    is comfortably under cap and the provider says the window is spent."""
    quota = QuotaLedger()
    with open_store(tmp_path, n_seeds=0) as store:
        store.record_usage("cerebras", "gpt-oss-120b", prompt_tokens=10, completion_tokens=10)
        quota.record("cerebras", "gpt-oss-120b", CEREBRAS_HEADERS_EXHAUSTED)
        budget_ok = budget_ok_for(store, cfg, quota=quota)
        assert budget_ok("cerebras", "gpt-oss-120b", 1) is False


def test_a_stale_observation_falls_back_to_the_ledger(tmp_path, cfg):
    """An observation is a statement about a window the provider is metering
    NOW. Past the TTL it stops deciding - see QUOTA_OBSERVATION_TTL_S for the
    error budget that number was chosen against."""
    now = [1000.0]
    quota = QuotaLedger(clock=lambda: now[0])
    with open_store(tmp_path, n_seeds=0) as store:
        _exhaust_ledger(store, cfg)
        quota.record("cerebras", "gpt-oss-120b", CEREBRAS_HEADERS_WITH_BUDGET)
        budget_ok = budget_ok_for(store, cfg, quota=quota)
        assert budget_ok("cerebras", "gpt-oss-120b", 4000) is True

        now[0] += QUOTA_OBSERVATION_TTL_S + 1
        # Stale: the ledger decides again - via one probe grant, then closed.
        assert budget_ok("cerebras", "gpt-oss-120b", 4000) is True
        assert store.events("budget_probe_grant")
        assert budget_ok("cerebras", "gpt-oss-120b", 4000) is False


def test_the_divergence_is_logged_once_per_provider_model_day(tmp_path, cfg):
    """The event that would have caught the previous pilot. It carries BOTH
    numbers, because the whole point is that they disagreed and nothing said
    so."""
    quota = QuotaLedger()
    with open_store(tmp_path, n_seeds=0) as store:
        _exhaust_ledger(store, cfg)
        quota.record("cerebras", "gpt-oss-120b", CEREBRAS_HEADERS_WITH_BUDGET)
        budget_ok = budget_ok_for(store, cfg, quota=quota)
        for _ in range(5):
            assert budget_ok("cerebras", "gpt-oss-120b", 4000) is True

        events = store.events("budget_source_divergence")
        assert len(events) == 1
        detail = json.loads(events[0]["detail_json"])
        assert detail["decided_by"] == "provider"
        assert detail["provider_allows"] is True
        assert detail["ledger_allows"] is False
        assert detail["remaining_tokens_day"] == 458408
        assert detail["limit_tokens_day"] == 1000000
        assert detail["ledger_tokens_today"] == 1_015_901
        assert detail["configured_tpd"] == 1_000_000


def test_quota_headers_are_parsed_and_missing_ones_are_not_guessed():
    obs = parse_quota_headers(CEREBRAS_HEADERS_WITH_BUDGET, now=0.0)
    assert obs.remaining_tokens_day == 458408
    assert obs.remaining_requests_day == 2399
    assert obs.limit_requests_day == 2400
    # A provider that publishes nothing usable yields nothing to decide on.
    assert parse_quota_headers({}, now=0.0) is None
    assert parse_quota_headers({"x-ratelimit-remaining-tokens-day": "n/a"}, now=0.0) is None
    assert parse_quota_headers(None, now=0.0) is None
    # A provider that publishes only the minute counters cannot decide the DAY
    # gate, and must fall through rather than be read as exhausted.
    minute_only = parse_quota_headers({"x-ratelimit-remaining-tokens-minute": "10"}, now=0.0)
    assert minute_only is not None and minute_only.allows(1) is None
