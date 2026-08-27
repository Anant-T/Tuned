"""Pre-registered 80-seed matched evaluator (law-v1 recovery Task 5).

Statistical helpers and promotion branches are pinned on known values.
Fixtures stand in for live control / recovery stores; nothing here opens
the frozen live SQLite file or writes gold labels.
"""

from __future__ import annotations

import ast
import json
import math
import sqlite3
from pathlib import Path

import pytest

from tuned.data.gates import DIAGNOSTIC_GATES, GATE_ORDER, PERMANENT_GATES
from tuned.data.store import Store

from tuned.data import eval_matched as E

EVAL_SRC = Path(E.__file__)

TASK_TYPES = ("irac_analysis", "statute_qa", "drafting", "summarization")
SECTION = "Section 34. Distinct provision text, not the judgment body."
SOURCE_BODY = "judgment body text that is not a statute provision"

# Independent Wilson form (center/margin via 2(n+z^2)), not the production
# (1 + z^2/n) arrangement. z=1.96 is the pre-registered 95% constant.
_Z = 1.96


def _reference_wilson(successes: int, n: int, z: float = _Z):
    p = successes / n
    den = 2.0 * (n + z * z)
    center = (2.0 * n * p + z * z) / den
    margin = z * math.sqrt(z * z + 4.0 * n * p * (1.0 - p)) / den
    return (center - margin, p, center + margin)


@pytest.fixture
def store(tmp_path):
    with Store.open(tmp_path / "state" / "law_v1.sqlite3") as s:
        s.upsert_source("ikanoon", "CC-BY-4.0")
        yield s


def _contract(**extra):
    base = dict(
        think_min=500,
        teacher_family="gpt-oss",
        teacher_model="gpt-oss-120b",
        gate_contract=tuple(GATE_ORDER),
    )
    base.update(extra)
    return E.EvalContract(**base)


def _write_pre_manifest(path, control, treatment, contract=None):
    return E.write_manifest(
        path,
        control,
        treatment_store=treatment,
        contract=contract or _contract(),
    )


def _baseline_candidate_map(control):
    labels = list(control.gold_labels())
    if not labels:
        return {}
    judgements = control.judgements_by_gen(int(row["gen_id"]) for row in labels)
    out = {}
    for row in labels:
        gid = int(row["gen_id"])
        already = E._already_regenerated_for_gen(control, gid)
        out[gid] = E.dual_judge_decision(
            judgements.get(gid, {}), already_regenerated=already
        )
    return out


def _eval(
    control,
    treatment,
    *,
    contract=None,
    treatment_contract=None,
    manifest_path=None,
    candidate_decisions=None,
    candidate_decisions_path=None,
):
    contract = contract or _contract()
    treatment_contract = treatment_contract or contract
    kwargs = dict(control=contract, treatment=treatment_contract)
    if manifest_path is not None:
        kwargs["manifest_path"] = manifest_path
    if candidate_decisions_path is not None:
        kwargs["candidate_decisions_path"] = candidate_decisions_path
    elif candidate_decisions is not None:
        kwargs["candidate_decisions"] = candidate_decisions
    else:
        kwargs["candidate_decisions"] = _baseline_candidate_map(control)
    return E.evaluate(control, treatment, **kwargs)


def _seed_row(seed_id, *, section=SECTION, text=SOURCE_BODY):
    meta = json.dumps({"section_text": section}) if section is not None else None
    return {
        "seed_id": seed_id,
        "source_id": "ikanoon",
        "text": text,
        "meta_json": meta,
    }


def _task_row(task_id, seed_id, task_type, *, state="pending", prompt_sha="abc"):
    return {
        "task_id": task_id,
        "seed_id": seed_id,
        "stream": "synthesis",
        "task_type": task_type,
        "prompt_id": f"gen_{task_type}",
        "prompt_sha": prompt_sha,
        "sample_ix": 0,
        "state": state,
    }


def _gate_rows(*, passed=True, **overrides):
    flags = {gate: passed for gate in GATE_ORDER}
    flags.update(overrides)
    return [(gate, flags[gate], None) for gate in GATE_ORDER]


def _record_gen(
    store,
    task_id,
    *,
    attempt=1,
    model="gpt-oss-120b",
    family="gpt-oss",
    finish_reason="stop",
    gates=None,
    judgements=None,
    raw_offset=None,
    params=None,
):
    row = {
        "task_id": task_id,
        "attempt": attempt,
        "provider": "cerebras",
        "model": model,
        "model_family": family,
        "raw_path": "raw.ndjson",
        "raw_offset": attempt if raw_offset is None else raw_offset,
        "think": "think",
        "answer": "answer",
        "finish_reason": finish_reason,
    }
    if params is not None:
        row["params_json"] = json.dumps(params)
    gen_id = store.record_generation(row)
    if gates is not None:
        store.record_gates(gen_id, gates)
    if judgements:
        for slot, axes in judgements.items():
            grounding, validity, coverage = axes
            store.record_judgement(
                gen_id,
                slot,
                {
                    "provider": "groq",
                    "model": "judge",
                    "grounding": grounding,
                    "validity": validity,
                    "coverage": coverage,
                },
            )
    return gen_id


def _plant_unit(
    store,
    *,
    seed_id,
    task_type,
    task_id=None,
    state="pending",
    attempt=1,
    model="gpt-oss-120b",
    gates=None,
    judgements=None,
    finish_reason="stop",
    section=SECTION,
    extra_attempts=(),
    params=None,
):
    task_id = task_id or f"{seed_id}:{task_type}"
    if store.get_seed(seed_id) is None:
        store.upsert_seeds([_seed_row(seed_id, section=section)])
    if store.get_task(task_id) is None:
        store.create_tasks([_task_row(task_id, seed_id, task_type, state=state)])
    elif state != "pending":
        store.set_task_state(task_id, state)
    gen_id = _record_gen(
        store,
        task_id,
        attempt=attempt,
        model=model,
        gates=gates,
        judgements=judgements,
        finish_reason=finish_reason,
        params=params,
    )
    for extra in extra_attempts:
        _record_gen(store, task_id, **extra)
    return gen_id


# --------------------------------------------------------------------------
# Statistics.
# --------------------------------------------------------------------------


def test_wilson_interval_matches_known_values():
    for k, n in ((0, 10), (1, 10), (8, 20), (15, 50), (20, 52)):
        got = E.wilson_interval(k, n, z=_Z)
        ref = _reference_wilson(k, n)
        assert got[1] == pytest.approx(k / n)
        assert got[0] == pytest.approx(ref[0])
        assert got[2] == pytest.approx(ref[2])
    assert E.wilson_interval(0, 0) is None


def test_mcnemar_one_sided_matches_known_binomial_tails():
    assert E.mcnemar_one_sided(8, 2) == pytest.approx(56 / 1024)
    assert E.mcnemar_one_sided(9, 1) == pytest.approx(11 / 1024)
    assert E.mcnemar_one_sided(6, 0) == pytest.approx(1 / 64)
    assert E.mcnemar_one_sided(0, 0) == 1.0
    assert E.mcnemar_one_sided(42, 0) < 0.05


# --------------------------------------------------------------------------
# Format / legal / operational contracts.
# --------------------------------------------------------------------------


def test_empty_or_incomplete_gates_fail_and_never_implicitly_pass():
    assert E.hard_format_pass({}) is False
    assert E.hard_format_pass(None) is False
    incomplete = {gate: True for gate in GATE_ORDER if gate != "citations"}
    assert E.hard_format_pass(incomplete) is False
    full = {gate: True for gate in GATE_ORDER}
    assert E.hard_format_pass(full) is True


def test_diagnostics_are_excluded_from_hard_format():
    flags = {gate: True for gate in GATE_ORDER}
    for name in DIAGNOSTIC_GATES:
        flags[name] = False
    assert E.hard_format_pass(flags) is True
    flags["think_format"] = False
    assert E.hard_format_pass(flags) is False


def test_synthetic_truncation_and_reply_budget_fail_hard_format():
    flags = {gate: True for gate in GATE_ORDER}
    assert E.hard_format_pass(flags, finish_reason="length") is False
    assert E.hard_format_pass(flags, finish_reason="max_tokens") is False
    assert E.hard_format_pass(flags, finish_reason="stop") is True
    with_trunc = dict(flags)
    with_trunc["truncated"] = False
    assert E.hard_format_pass(with_trunc) is False
    with_budget = dict(flags)
    with_budget["reply_budget"] = False
    assert E.hard_format_pass(with_budget) is False


def test_live_style_disposition_and_event_reply_budget_fail_without_planted_gates():
    flags = {gate: True for gate in GATE_ORDER}
    assert "truncated" not in flags and "reply_budget" not in flags
    # Stale task disposition on a later accepted attempt is not a synthetic fail.
    assert E.hard_format_pass(
        flags,
        finish_reason="stop",
        disposition="exhausted:format:reply_budget",
        state="accepted",
    ) is True
    for operational in ("judging", "judging_active", "generating", "rejected"):
        assert E.hard_format_pass(
            flags,
            finish_reason="stop",
            disposition="regenerate:reply_budget",
            state=operational,
        ) is True
    # Fallback only when the latest attempt itself is parked or pending.
    assert E.hard_format_pass(
        flags,
        finish_reason="stop",
        disposition="exhausted:format:reply_budget",
        state="format_parked",
    ) is False
    assert E.hard_format_pass(
        flags,
        finish_reason="stop",
        disposition="regenerate:truncated",
        state="pending",
    ) is False
    assert E.hard_format_pass(
        flags,
        finish_reason="stop",
        events=[{"kind": "reply_over_budget", "detail": {"gen_id": 11}}],
        gen_id=11,
    ) is False
    assert E.hard_format_pass(
        flags,
        finish_reason="stop",
        events=[{"kind": "generation_truncated", "detail": {"gen_id": 11}}],
        gen_id=11,
    ) is False
    assert E.hard_format_pass(flags, finish_reason="stop", disposition="judge:accept") is True


def test_legal_fail_is_permanent_gates_and_grounding_is_separate():
    flags = {gate: True for gate in GATE_ORDER}
    assert E.legal_fail(flags) is False
    assert E.statutory_grounding_fail(flags) is False
    flags["citations"] = False
    assert E.legal_fail(flags) is True
    flags = {gate: True for gate in GATE_ORDER}
    flags["statutory_grounding"] = False
    assert E.legal_fail(flags) is False
    assert E.statutory_grounding_fail(flags) is True
    assert PERMANENT_GATES == frozenset({"citations", "temporal", "answer_key"})


def test_operational_states_are_excluded_from_the_judge_denominator():
    flags = {gate: True for gate in GATE_ORDER}
    judgements = {"a": (5, 5, 5), "b": (5, 5, 5)}
    accepted = E.MatchedRow(
        seed_id="s",
        task_type="drafting",
        task_id="t",
        gen_id=1,
        attempt=1,
        state="accepted",
        model="gpt-oss-120b",
        finish_reason="stop",
        gates=flags,
        judgements=judgements,
    )
    judging = E.MatchedRow(
        **{**accepted.__dict__, "state": "judging", "gen_id": 2, "task_id": "t2"}
    )
    cooling = E.MatchedRow(
        **{**accepted.__dict__, "state": "judge_unroutable", "gen_id": 3, "task_id": "t3"}
    )
    error = E.MatchedRow(
        **{**accepted.__dict__, "state": "judge_error", "gen_id": 4, "task_id": "t4"}
    )
    parked = E.MatchedRow(
        **{**accepted.__dict__, "state": "format_parked", "gen_id": 5, "task_id": "t5"}
    )
    rejected = E.MatchedRow(
        **{
            **accepted.__dict__,
            "state": "rejected",
            "gen_id": 6,
            "task_id": "t6",
            "judgements": {"a": (1, 1, 1), "b": (1, 1, 1)},
        }
    )
    cond = E.judge_accept_cond([accepted, judging, cooling, error, parked, rejected])
    assert cond.n == 2
    assert cond.successes == 1
    assert cond.excluded_operational == 4


# --------------------------------------------------------------------------
# Dual-judge lockbox (by gen_id, not task state).
# --------------------------------------------------------------------------


def test_pre_registered_dual_judge_accepts_only_two_passes():
    assert E.dual_judge_decision({"a": (5, 5, 5), "b": (4, 4, 4)}) == "accept"
    assert E.dual_judge_decision({"a": (1, 1, 1), "b": (1, 2, 1)}) == "reject"
    assert E.dual_judge_decision({"a": (5, 5, 5), "b": (1, 1, 1)}) == "tiebreak"
    assert (
        E.dual_judge_decision({"a": (5, 5, 5), "b": (1, 1, 1), "tiebreak": (5, 5, 5)})
        == "accept"
    )
    assert E.dual_judge_decision({"a": (5, 5, 5)}) is None


_MATRIX = (
    ((5, 5, 5), (4, 4, 4), None, False, "accept"),
    ((1, 1, 1), (1, 2, 1), None, False, "reject"),
    ((5, 5, 5), (1, 1, 1), None, False, "tiebreak"),
    ((5, 5, 5), (3, 3, 3), None, False, "tiebreak"),
    ((3, 3, 3), (3, 4, 3), None, False, "regenerate"),
    ((3, 3, 3), (1, 1, 1), None, False, "regenerate"),
    ((3, 3, 3), (3, 3, 3), None, True, "reject"),
    ((5, 5, 5), (1, 1, 1), (5, 5, 5), False, "accept"),
    ((5, 5, 5), (1, 1, 1), (1, 1, 1), False, "reject"),
    ((5, 5, 5), (1, 1, 1), (3, 3, 3), False, "regenerate"),
    ((5, 5, 5), (1, 1, 1), (3, 3, 3), True, "reject"),
)


@pytest.mark.parametrize("a,b,tie,already,expected", _MATRIX)
def test_dual_judge_matches_judge_decide_matrix(a, b, tie, already, expected):
    from tuned.data.judge import JudgeScores, decide

    slots = {"a": a, "b": b}
    if tie is not None:
        slots["tiebreak"] = tie
    assert E.dual_judge_decision(slots, already_regenerated=already) == expected
    verdicts = [JudgeScores(*a).verdict, JudgeScores(*b).verdict]
    if tie is not None:
        verdicts.append(JudgeScores(*tie).verdict)
    assert decide(verdicts, already_regenerated=already) == expected
    assert E.dual_judge_decision(slots, already_regenerated=already) == decide(
        verdicts, already_regenerated=already
    )


def test_lockbox_uses_labelled_gen_id_not_task_state(store):
    store.upsert_seeds([_seed_row("gold-seed")])
    store.create_tasks(
        [_task_row("gold-task", "gold-seed", "irac_analysis", state="accepted")]
    )
    gen_id = _record_gen(
        store,
        "gold-task",
        judgements={"a": (1, 1, 1), "b": (1, 1, 1)},
    )
    store.upsert_gold_labels([{"gen_id": gen_id, "verdict": "reject"}])
    store.set_task_state("gold-task", "accepted", "judge:accept")
    task = store.get_task("gold-task")
    assert task["state"] == "accepted"

    report = E.lockbox_confirm(
        store.gold_labels(),
        store.judgements_by_gen([gen_id]),
        store=store,
        candidate_decisions={gen_id: "reject"},
    )
    assert report.n_reject == 1
    assert report.n_accept == 0
    assert report.reject_flips == 0
    assert report.new_accept_fns == 0
    assert report.used_task_state is False
    # The rule read the labelled generation, not the (wrong) accepted state.
    assert report.baseline_decisions[gen_id] == "reject"


def _lockbox_46():
    labels = [{"gen_id": i, "verdict": "reject"} for i in range(1, 37)]
    labels += [{"gen_id": i, "verdict": "accept"} for i in range(37, 47)]
    judgements = {i: {"a": (1, 1, 1), "b": (1, 1, 1)} for i in range(1, 37)}
    judgements.update({i: {"a": (5, 5, 5), "b": (5, 5, 5)} for i in range(37, 47)})
    return labels, judgements


def test_one_reject_flip_refuses_promotion():
    labels, judgements = _lockbox_46()
    candidate = {i: "reject" for i in range(1, 37)}
    candidate.update({i: "accept" for i in range(37, 47)})
    candidate[1] = "accept"
    box = E.lockbox_confirm(labels, judgements, candidate_decisions=candidate)
    assert box.baseline_decisions[1] == "reject"
    assert box.reject_flips == 1
    assert box.new_accept_fns == 0
    verdict, reasons = E.decide_promotion(**_promote_kwargs(lockbox_reject_flips=1))
    assert verdict == "refuse"
    assert any("reject-flip" in r for r in reasons)


def test_a_new_human_accept_false_negative_refuses():
    labels, judgements = _lockbox_46()
    candidate = {i: "reject" for i in range(1, 37)}
    candidate.update({i: "accept" for i in range(37, 47)})
    candidate[37] = "reject"
    box = E.lockbox_confirm(labels, judgements, candidate_decisions=candidate)
    assert box.baseline_decisions[37] == "accept"
    assert box.new_accept_fns == 1
    assert box.new_false_negatives == 1
    assert box.reject_flips == 0
    verdict, reasons = E.decide_promotion(**_promote_kwargs(lockbox_new_false_negatives=1))
    assert verdict == "refuse"
    assert any("false-negative" in r for r in reasons)


def test_lockbox_without_candidate_is_incomplete_not_baseline():
    labels, judgements = _lockbox_46()
    judgements[1] = {"a": (5, 5, 5), "b": (5, 5, 5)}
    judgements[37] = {"a": (1, 1, 1), "b": (1, 1, 1)}
    box = E.lockbox_confirm(labels, judgements)
    assert box.baseline_decisions[1] == "accept"
    assert box.baseline_decisions[37] != "accept"
    assert box.reject_flips == 0
    assert box.new_accept_fns == 0
    assert box.valid is False
    assert "lockbox-incomplete" in box.reason


def test_lockbox_reads_already_regenerated_from_persisted_params(store):
    store.upsert_seeds([_seed_row("regen-seed")])
    store.create_tasks([_task_row("regen-task", "regen-seed", "irac_analysis")])
    gen_id = _record_gen(
        store,
        "regen-task",
        judgements={"a": (3, 3, 3), "b": (3, 3, 3)},
        params={"reviewer_note_applied": True},
    )
    store.upsert_gold_labels([{"gen_id": gen_id, "verdict": "reject"}])
    judgements = store.judgements_by_gen([gen_id])
    without_store = E.lockbox_confirm(
        store.gold_labels(),
        judgements,
        candidate_decisions={gen_id: "regenerate"},
    )
    assert without_store.baseline_decisions[gen_id] == "regenerate"
    box = E.lockbox_confirm(
        store.gold_labels(),
        judgements,
        store=store,
        candidate_decisions={gen_id: "reject"},
    )
    assert box.baseline_decisions[gen_id] == "reject"
    assert box.reject_flips == 0
    assert box.new_accept_fns == 0


def test_lockbox_candidate_cannot_change_cardinality_or_labels():
    labels, judgements = _lockbox_46()
    extra = {i: "accept" for i in range(1, 80)}
    extra[37] = "reject"
    box = E.lockbox_confirm(labels, judgements, candidate_decisions=extra)
    assert box.n_reject == 36
    assert box.n_accept == 10
    assert set(box.baseline_decisions) == {i for i in range(1, 47)}
    assert 79 not in box.baseline_decisions
    assert box.valid is False
    assert "lockbox-incomplete" in box.reason


def test_lockbox_candidate_map_must_cover_exactly_the_labelled_gen_ids():
    labels, judgements = _lockbox_46()
    missing = {i: "reject" for i in range(1, 37)}
    missing.update({i: "accept" for i in range(37, 46)})
    box = E.lockbox_confirm(labels, judgements, candidate_decisions=missing)
    assert box.valid is False
    assert "lockbox-incomplete" in box.reason
    assert box.reject_flips == 0

    complete = {i: "reject" for i in range(1, 37)}
    complete.update({i: "accept" for i in range(37, 47)})
    ok = E.lockbox_confirm(labels, judgements, candidate_decisions=complete)
    assert ok.valid is True
    assert ok.reason == ""
    assert ok.reject_flips == 0
    assert ok.new_accept_fns == 0


def test_lockbox_candidate_map_rejects_invalid_values():
    labels, judgements = _lockbox_46()
    candidate = {i: "reject" for i in range(1, 37)}
    candidate.update({i: "accept" for i in range(37, 47)})
    candidate[1] = "baseline"
    box = E.lockbox_confirm(labels, judgements, candidate_decisions=candidate)
    assert box.valid is False
    assert "lockbox-incomplete" in box.reason
    assert box.reject_flips == 0


# --------------------------------------------------------------------------
# Selection.
# --------------------------------------------------------------------------


def _eligible_pool(store, *, per_type=25, gold_at=None, extra=None):
    gold_at = gold_at or {}
    extra = extra or []
    for task_type in TASK_TYPES:
        for i in range(per_type):
            seed_id = f"{task_type}-{i:03d}"
            _plant_unit(
                store,
                seed_id=seed_id,
                task_type=task_type,
                gates=_gate_rows(),
                state="accepted",
            )
            if gold_at.get(task_type) == i:
                task = store.get_task(f"{seed_id}:{task_type}")
                latest = store.latest_generation(task["task_id"])
                store.upsert_gold_labels([{"gen_id": latest["gen_id"], "verdict": "reject"}])
    for spec in extra:
        _plant_unit(store, **spec)


def test_selection_is_one_latest_row_per_seed_and_task_type(store):
    _plant_unit(
        store,
        seed_id="s1",
        task_type="drafting",
        task_id="s1-old",
        attempt=1,
        gates=_gate_rows(passed=False),
    )
    _plant_unit(
        store,
        seed_id="s1",
        task_type="drafting",
        task_id="s1-new",
        attempt=3,
        gates=_gate_rows(),
        state="accepted",
    )
    # Same seed, different type: a second unit, not a duplicate.
    _plant_unit(
        store,
        seed_id="s1",
        task_type="summarization",
        gates=_gate_rows(),
        state="accepted",
    )
    units = E.latest_units(store)
    drafting = [u for u in units if u.seed_id == "s1" and u.task_type == "drafting"]
    assert len(drafting) == 1
    assert drafting[0].attempt == 3
    assert drafting[0].task_id == "s1-new"


def test_format_first_uses_attempt_one_and_format_latest_uses_the_newest(store):
    store.upsert_seeds([_seed_row("s1")])
    store.create_tasks([_task_row("t1", "s1", "irac_analysis")])
    _record_gen(store, "t1", attempt=1, gates=_gate_rows(think_format=False))
    _record_gen(store, "t1", attempt=2, gates=_gate_rows())
    unit = E.latest_units(store)[0]
    first = E.first_attempt_unit(store, unit)
    assert E.hard_format_pass(unit.gates) is True
    assert E.hard_format_pass(first.gates) is False


def test_deterministic_twenty_by_four_excludes_gold_seeds(store):
    _eligible_pool(store, per_type=25, gold_at={"irac_analysis": 0, "drafting": 1})
    first = E.select_cohort(store, n_per=20)
    second = E.select_cohort(store, n_per=20)
    assert first.blocked is False
    assert first.stratum_counts == {t: 20 for t in TASK_TYPES}
    assert len(first.pairs) == 80
    assert [(p.seed_id, p.task_type) for p in first.pairs] == [
        (p.seed_id, p.task_type) for p in second.pairs
    ]
    selected_seeds = {p.seed_id for p in first.pairs}
    assert "irac_analysis-000" not in selected_seeds
    assert "drafting-001" not in selected_seeds
    gold_seeds = E.gold_linked_seeds(store)
    assert gold_seeds <= {"irac_analysis-000", "drafting-001"}
    assert selected_seeds.isdisjoint(gold_seeds)


def test_stale_no_generation_and_non_gpt_oss_rows_are_excluded(store):
    for i, task_type in enumerate(TASK_TYPES):
        for j in range(20):
            _plant_unit(
                store,
                seed_id=f"ok-{task_type}-{j:03d}",
                task_type=task_type,
                gates=_gate_rows(),
                state="accepted",
            )
    store.upsert_seeds([_seed_row("stale-seed"), _seed_row("nogen-seed"), _seed_row("qwen-seed")])
    store.create_tasks(
        [
            _task_row("stale-task", "stale-seed", "irac_analysis", state="stale_prompt"),
            _task_row("nogen-task", "nogen-seed", "irac_analysis"),
            _task_row("qwen-task", "qwen-seed", "irac_analysis"),
        ]
    )
    _record_gen(store, "stale-task", gates=_gate_rows())
    _record_gen(store, "qwen-task", model="qwen/qwen3.6-27b", family="qwen", gates=_gate_rows())
    selected = E.select_cohort(store, n_per=20)
    seeds = {p.seed_id for p in selected.pairs}
    assert "stale-seed" not in seeds
    assert "nogen-seed" not in seeds
    assert "qwen-seed" not in seeds
    assert selected.excluded["stale"] >= 1
    assert selected.excluded["no_generation"] >= 1
    assert selected.excluded["non_gpt_oss"] >= 1


def test_underfilled_statute_stratum_is_blocked_and_inconclusive(store):
    for task_type in TASK_TYPES:
        n = 8 if task_type == "statute_qa" else 22
        for i in range(n):
            section = SOURCE_BODY if task_type == "statute_qa" and i >= 4 else SECTION
            _plant_unit(
                store,
                seed_id=f"{task_type}-{i:03d}",
                task_type=task_type,
                gates=_gate_rows(),
                state="accepted",
                section=section,
            )
    selected = E.select_cohort(store, n_per=20)
    assert selected.blocked is True
    assert selected.stratum_counts["statute_qa"] < 20
    assert selected.decision == "inconclusive"
    assert "statute_qa" in selected.reason
    # No silent reallocation into another type.
    assert selected.stratum_counts["irac_analysis"] <= 20
    assert len(selected.pairs) < 80

    report = _eval(store, store)
    assert report.decision == "inconclusive"
    assert any("underfilled" in r or "statute" in r for r in report.reasons)


# --------------------------------------------------------------------------
# Promotion decision.
# --------------------------------------------------------------------------


def _promote_kwargs(**overrides):
    kwargs = dict(
        format_latest=0.65,
        mcnemar_p=0.001,
        judging_complete=True,
        judge_accept_wilson_lo=0.26,
        treatment_legal_fails=0,
        control_legal_fails=1,
        lockbox_reject_flips=0,
        lockbox_new_false_negatives=0,
        types_with_format_pass=set(TASK_TYPES),
        generated_types=set(TASK_TYPES),
        selection_blocked=False,
        contract_ok=True,
        missing_gate_data=False,
        n_pairs=80,
    )
    kwargs.update(overrides)
    return kwargs


def test_decide_promotion_promote_and_the_refusal_inconclusive_branches():
    verdict, reasons = E.decide_promotion(**_promote_kwargs())
    assert verdict == "promote"
    assert reasons == ()

    v, r = E.decide_promotion(**_promote_kwargs(format_latest=0.49))
    assert v == "refuse" and any("format_latest" in x for x in r)

    v, r = E.decide_promotion(**_promote_kwargs(mcnemar_p=0.06))
    assert v == "inconclusive" and any("mcnemar" in x or "insufficient" in x for x in r)

    v, r = E.decide_promotion(**_promote_kwargs(judging_complete=False, judge_accept_wilson_lo=None))
    assert v == "inconclusive" and any("judging" in x for x in r)

    v, r = E.decide_promotion(**_promote_kwargs(judge_accept_wilson_lo=0.17))
    assert v == "refuse" and any("wilson" in x or "judge_accept" in x for x in r)

    v, r = E.decide_promotion(**_promote_kwargs(treatment_legal_fails=3, control_legal_fails=2))
    assert v == "refuse" and any("legal" in x for x in r)

    v, r = E.decide_promotion(
        **_promote_kwargs(types_with_format_pass={"irac_analysis", "statute_qa", "drafting"})
    )
    assert v == "refuse" and any("format" in x and "summarization" in x for x in r)

    v, r = E.decide_promotion(**_promote_kwargs(selection_blocked=True))
    assert v == "inconclusive"

    v, r = E.decide_promotion(**_promote_kwargs(contract_ok=False))
    assert v == "inconclusive" and any("contract" in x for x in r)

    v, r = E.decide_promotion(**_promote_kwargs(missing_gate_data=True))
    assert v == "inconclusive" and any("gate" in x for x in r)


def _fill_stratum(store, task_type, n, *, prefix, format_pass, accept, legal_fail=False):
    for i in range(n):
        gates = _gate_rows()
        if not format_pass(i):
            gates = _gate_rows(think_format=False)
        if legal_fail and i == 0:
            gates = _gate_rows(citations=False)
        judgements = None
        state = "format_parked"
        if format_pass(i):
            if accept(i):
                judgements = {"a": (5, 5, 5), "b": (5, 5, 5)}
                state = "accepted"
            else:
                judgements = {"a": (1, 1, 1), "b": (1, 1, 1)}
                state = "rejected"
        _plant_unit(
            store,
            seed_id=f"{prefix}-{task_type}-{i:03d}",
            task_type=task_type,
            gates=gates,
            judgements=judgements,
            state=state,
        )


def _lockbox_rows(store, n_reject=36, n_accept=10):
    labels = []
    for i in range(n_reject):
        seed_id = f"lock-rej-{i:03d}"
        gen_id = _plant_unit(
            store,
            seed_id=seed_id,
            task_type="irac_analysis",
            task_id=f"lock-rej-{i:03d}",
            gates=_gate_rows(),
            judgements={"a": (1, 1, 1), "b": (1, 1, 1)},
            state="rejected",
        )
        labels.append({"gen_id": gen_id, "verdict": "reject"})
    for i in range(n_accept):
        seed_id = f"lock-acc-{i:03d}"
        gen_id = _plant_unit(
            store,
            seed_id=seed_id,
            task_type="irac_analysis",
            task_id=f"lock-acc-{i:03d}",
            gates=_gate_rows(),
            judgements={"a": (5, 5, 5), "b": (5, 5, 5)},
            state="accepted",
        )
        labels.append({"gen_id": gen_id, "verdict": "accept"})
    store.upsert_gold_labels(labels)
    return labels


def _promote_pair(tmp_path):
    control = Store.open(tmp_path / "control" / "law_v1.sqlite3")
    treatment = Store.open(tmp_path / "treat" / "law_v1.sqlite3")
    control.upsert_source("ikanoon", "CC-BY-4.0")
    treatment.upsert_source("ikanoon", "CC-BY-4.0")
    for task_type in TASK_TYPES:
        # Control: first 3 format-pass (12 total across types after 4*3).
        _fill_stratum(
            control,
            task_type,
            20,
            prefix="c",
            format_pass=lambda i: i < 3,
            accept=lambda i: False,
        )
    _lockbox_rows(control)
    manifest_path = tmp_path / "cohort.json"
    _write_pre_manifest(manifest_path, control, treatment)
    for task_type in TASK_TYPES:
        # Treatment: first 13 format-pass; first 5 of those accepted.
        _fill_stratum(
            treatment,
            task_type,
            20,
            prefix="c",
            format_pass=lambda i: i < 13,
            accept=lambda i: i < 5,
        )
    return control, treatment, manifest_path


def test_successful_synthetic_promote_fixture(tmp_path):
    control, treatment, manifest_path = _promote_pair(tmp_path)
    try:
        report = _eval(control, treatment, manifest_path=manifest_path)
        assert report.selection.blocked is False
        assert len(report.selection.pairs) == 80
        assert report.metrics["format_latest"].n == 80
        assert report.metrics["format_latest"].point >= 0.50
        assert report.metrics["format_first"].n == 80
        assert report.metrics["mcnemar_p"] < 0.05
        assert report.metrics["judge_accept_cond"].wilson_lo >= 0.18
        assert report.metrics["legal_fail"].treatment <= report.metrics["legal_fail"].control
        assert report.metrics["statutory_grounding"].treatment == 0
        assert report.metrics["statutory_grounding"].control == 0
        assert report.lockbox.reject_flips == 0
        assert report.lockbox.new_false_negatives == 0
        assert report.lockbox.n_reject == 36
        assert report.lockbox.n_accept == 10
        assert report.lockbox.valid is True
        assert report.decision == "promote"
        assert "high-confidence pseudo-gold" in report.synthesis_label
        assert report.synthesis_label != "gold"
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "verdict" not in json.dumps(loaded)
        assert "gold" not in json.dumps(loaded).lower()
        assert loaded["n"] == 80
        assert set(loaded["task_types"]) == set(TASK_TYPES)
        assert loaded["control_fingerprint"]
    finally:
        control.close()
        treatment.close()


def test_contract_mismatch_is_inconclusive(tmp_path):
    control, treatment, manifest_path = _promote_pair(tmp_path)
    try:
        report = _eval(
            control,
            treatment,
            contract=_contract(),
            treatment_contract=_contract(think_min=200),
            manifest_path=manifest_path,
        )
        assert report.decision == "inconclusive"
        assert any("contract" in r or "think_min" in r for r in report.reasons)
    finally:
        control.close()
        treatment.close()


def test_incomplete_judging_is_inconclusive(tmp_path):
    control, treatment, manifest_path = _promote_pair(tmp_path)
    try:
        # Leave one treatment format-passer without a completed decision.
        pair = next(
            p
            for p in E.select_cohort(control, n_per=20).pairs
            if p.task_type == "drafting" and p.seed_id.endswith("-000")
        )
        treat_task = treatment.conn.execute(
            "SELECT task_id FROM task WHERE seed_id = ? AND task_type = ?",
            (pair.seed_id, pair.task_type),
        ).fetchone()
        treatment.set_task_state(treat_task[0], "judging")
        latest = treatment.latest_generation(treat_task[0])
        treatment.conn.execute("DELETE FROM judgement WHERE gen_id = ?", (latest["gen_id"],))
        report = _eval(control, treatment, manifest_path=manifest_path)
        assert report.decision == "inconclusive"
        assert any("judging" in r or "incomplete" in r for r in report.reasons)
    finally:
        control.close()
        treatment.close()


def test_zero_format_passers_in_a_generated_type_refuses(tmp_path):
    control, treatment, manifest_path = _promote_pair(tmp_path)
    try:
        for row in treatment.conn.execute(
            "SELECT task_id FROM task WHERE task_type = 'summarization'"
        ).fetchall():
            latest = treatment.latest_generation(row[0])
            treatment.record_gates(latest["gen_id"], _gate_rows(think_format=False))
            treatment.set_task_state(row[0], "format_parked")
        report = _eval(control, treatment, manifest_path=manifest_path)
        assert report.decision == "refuse"
        assert any("summarization" in r for r in report.reasons)
    finally:
        control.close()
        treatment.close()


# --------------------------------------------------------------------------
# Safety: no fit, no labels, no providers, live URI is read-only.
# --------------------------------------------------------------------------


def test_evaluator_source_cannot_fit_mutate_labels_or_invoke_providers():
    src = EVAL_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden_modules = {
        "tuned.data.generate",
        "tuned.data.judge",
        "tuned.data.providers",
        "tuned.data.calibrate",
        "httpx",
        "tuned.data.providers",
    }
    assert imported.isdisjoint(forbidden_modules)
    for token in (
        "upsert_gold_labels",
        "record_judge_thresholds",
        "run_calibration",
        "fit_judge",
        "ingest_gold",
        "sanity_fit",
        "candidate_rule",
        "Router",
        "httpx",
        "async def",
        "await ",
    ):
        assert token not in src, token
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "fit_judge" not in names
    assert "high-confidence pseudo-gold" in src


def test_read_only_live_uri_is_required_and_refused_otherwise(tmp_path, monkeypatch):
    db = tmp_path / "state" / "law_v1.sqlite3"
    with Store.open(db) as writable:
        writable.upsert_source("ikanoon", "CC-BY-4.0")

    uri = E.readonly_uri(db)
    assert "mode=ro" in uri
    assert uri.startswith("file:")

    with pytest.raises(ValueError, match="mode=ro|read-only"):
        E.open_eval_store(db, readonly=False)

    ro = E.open_eval_store(db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            ro.upsert_source("other", "CC-BY-4.0")
    finally:
        ro.close()

    live = Store.open(db)
    monkeypatch.setattr(E, "is_live_store", lambda store: True)
    try:
        with pytest.raises(ValueError, match="mode=ro|read-only"):
            E.evaluate(
                live,
                live,
                control=_contract(),
                treatment=_contract(),
                manifest_path=tmp_path / "unused.json",
            )
    finally:
        live.close()


def test_manifest_never_copies_labels(tmp_path):
    control, treatment, path = _promote_pair(tmp_path)
    try:
        text = path.read_text(encoding="utf-8")
        assert "verdict" not in text
        assert "gold_label" not in text
        assert "accept" not in text
        assert "reject" not in text
        loaded = json.loads(text)
        assert loaded["n"] == 80
        assert "pairs" in loaded
        assert loaded["control_fingerprint"]
    finally:
        control.close()
        treatment.close()


def test_mcnemar_one_sided_is_oriented_to_treatment_improvement():
    assert E.mcnemar_one_sided(8, 2) == pytest.approx(56 / 1024)
    assert E.mcnemar_one_sided(2, 8) == pytest.approx(1013 / 1024)
    verdict, reasons = E.decide_promotion(**_promote_kwargs(mcnemar_p=E.mcnemar_one_sided(2, 8)))
    assert verdict == "inconclusive"
    assert any("mcnemar" in r or "insufficient" in r for r in reasons)


def test_treatment_worse_mcnemar_does_not_promote(tmp_path):
    control = Store.open(tmp_path / "control" / "law_v1.sqlite3")
    treatment = Store.open(tmp_path / "treat" / "law_v1.sqlite3")
    control.upsert_source("ikanoon", "CC-BY-4.0")
    treatment.upsert_source("ikanoon", "CC-BY-4.0")
    try:
        for task_type in TASK_TYPES:
            _fill_stratum(
                control, task_type, 20, prefix="c",
                format_pass=lambda i: i < 13, accept=lambda i: False,
            )
        _lockbox_rows(control)
        path = tmp_path / "cohort.json"
        _write_pre_manifest(path, control, treatment)
        for task_type in TASK_TYPES:
            _fill_stratum(
                treatment, task_type, 20, prefix="c",
                format_pass=lambda i: i < 11, accept=lambda i: i < 5,
            )
        report = _eval(control, treatment, manifest_path=path)
        assert report.metrics["format_latest"].point >= 0.50
        assert report.metrics["mcnemar_p"] >= 0.05
        assert report.decision == "inconclusive"
        assert any("mcnemar" in r or "insufficient" in r for r in report.reasons)
    finally:
        control.close()
        treatment.close()


def test_statutory_grounding_counts_are_reported_separately(tmp_path):
    control, treatment, manifest_path = _promote_pair(tmp_path)
    try:
        for i, store in enumerate((control, treatment)):
            row = store.conn.execute(
                "SELECT task_id FROM task WHERE task_type = 'statute_qa' ORDER BY task_id"
            ).fetchone()
            latest = store.latest_generation(row[0])
            store.record_gates(latest["gen_id"], _gate_rows(statutory_grounding=False))
        report = _eval(control, treatment, manifest_path=manifest_path)
        assert report.metrics["statutory_grounding"].control == 1
        assert report.metrics["statutory_grounding"].treatment == 1
        assert report.metrics["legal_fail"].control == 0
        assert report.metrics["legal_fail"].treatment == 0
    finally:
        control.close()
        treatment.close()


def test_evaluate_without_a_manifest_is_inconclusive(tmp_path):
    control, treatment, _path = _promote_pair(tmp_path)
    try:
        report = E.evaluate(
            control, treatment, control=_contract(), treatment=_contract()
        )
        assert report.decision == "inconclusive"
        assert any("manifest" in r for r in report.reasons)
    finally:
        control.close()
        treatment.close()


def test_evaluate_rejects_an_in_memory_manifest_dict(tmp_path):
    control, treatment, path = _promote_pair(tmp_path)
    try:
        ad_hoc = json.loads(path.read_text(encoding="utf-8"))
        with pytest.raises(TypeError, match="manifest_path|path"):
            E.evaluate(
                control,
                treatment,
                control=_contract(),
                treatment=_contract(),
                manifest=ad_hoc,
            )
    finally:
        control.close()
        treatment.close()


def test_manifest_id_or_contract_mismatch_is_inconclusive(tmp_path):
    control, treatment, path = _promote_pair(tmp_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        broken = dict(manifest)
        broken["pairs"] = [dict(manifest["pairs"][0], control_gen_id=-1)] + manifest["pairs"][1:]
        broken_path = tmp_path / "broken.json"
        broken_path.write_text(json.dumps(broken), encoding="utf-8")
        report = _eval(control, treatment, manifest_path=broken_path)
        assert report.decision == "inconclusive"
        assert any("manifest" in r or "contract" in r for r in report.reasons)

        wrong_gates = dict(manifest)
        wrong_gates["gate_contract"] = list(GATE_ORDER[:-1])
        wrong_path = tmp_path / "wrong-gates.json"
        wrong_path.write_text(json.dumps(wrong_gates), encoding="utf-8")
        report = _eval(control, treatment, manifest_path=wrong_path)
        assert report.decision == "inconclusive"
        assert any("gate" in r or "contract" in r for r in report.reasons)
    finally:
        control.close()
        treatment.close()


def test_lockbox_wrong_cardinality_is_inconclusive_and_cannot_promote(tmp_path):
    control, treatment, manifest_path = _promote_pair(tmp_path)
    try:
        extra = store_gen = _plant_unit(
            control,
            seed_id="lock-extra",
            task_type="irac_analysis",
            task_id="lock-extra",
            gates=_gate_rows(),
            judgements={"a": (1, 1, 1), "b": (1, 1, 1)},
            state="rejected",
        )
        control.upsert_gold_labels([{"gen_id": extra, "verdict": "reject"}])
        report = _eval(control, treatment, manifest_path=manifest_path)
        assert report.lockbox.valid is False
        assert report.lockbox.n_reject != 36 or report.lockbox.n_accept != 10 or (
            report.lockbox.n_reject + report.lockbox.n_accept != 46
        )
        assert report.decision == "inconclusive"
        assert any("lockbox" in r or "cardinality" in r for r in report.reasons)
        assert store_gen
    finally:
        control.close()
        treatment.close()


def test_incomplete_permanent_gates_are_missing_gate_data(tmp_path):
    control, treatment, manifest_path = _promote_pair(tmp_path)
    try:
        pair = next(
            p
            for p in E.select_cohort(control, n_per=20).pairs
            if p.task_type == "drafting" and p.seed_id.endswith("-010")
        )
        treat_task = treatment.conn.execute(
            "SELECT task_id FROM task WHERE seed_id = ? AND task_type = ?",
            (pair.seed_id, pair.task_type),
        ).fetchone()
        latest = treatment.latest_generation(treat_task[0])
        incomplete = [(g, True, None) for g in GATE_ORDER if g != "citations"]
        treatment.record_gates(latest["gen_id"], incomplete)
        treatment.conn.execute(
            "DELETE FROM gate_result WHERE gen_id = ? AND gate = ?",
            (latest["gen_id"], "citations"),
        )
        report = _eval(control, treatment, manifest_path=manifest_path)
        assert report.decision == "inconclusive"
        assert any("gate" in r for r in report.reasons)
        assert report.metrics["legal_fail"].treatment <= report.metrics["legal_fail"].control
    finally:
        control.close()
        treatment.close()


def test_contract_from_config_reads_think_min_and_teacher(tmp_path):
    from tuned.data.config import load_build_config

    live = load_build_config(
        Path(__file__).parent.parent / "configs" / "data_law_v1.yaml",
        allow_unpinned=True,
    )
    recovery = load_build_config(
        Path(__file__).parent.parent / "configs" / "data_law_v1_exp_recovery.yaml",
        allow_unpinned=True,
    )
    control_c = E.contract_from_config(live)
    treat_c = E.contract_from_config(recovery)
    assert control_c.think_min == treat_c.think_min == 500
    # NOT a cross-config invariant, even though both sides currently agree:
    # data_law_v1_exp_recovery.yaml's own header pins it to "Cerebras
    # gpt-oss-120b is the only generator" as an ISOLATED, frozen experiment,
    # so it is read against ITSELF rather than against the live side. The live
    # side cycled bai (family deepseek) through the lead generator slot
    # 2026-08-25 to 2026-08-27 and back - see routing.generator's own comment
    # in data_law_v1.yaml - so the two configs coincide here only because
    # gpt-oss currently leads on both, not because anything requires it.
    assert control_c.teacher_family == "gpt-oss"
    assert control_c.teacher_model == "gpt-oss-120b"
    assert treat_c.teacher_family == "gpt-oss"
    assert treat_c.teacher_model == "gpt-oss-120b"
    assert control_c.gate_contract == tuple(GATE_ORDER)


def _cohort_selection(strata, *, n_per=20, gen_id_start=1):
    """A Selection with dummy MatchedRows, shaped only enough for
    cohort_manifest (seed_id/task_type/task_id/gen_id/attempt)."""
    pairs = []
    gen_id = gen_id_start
    for task_type in strata:
        for i in range(n_per):
            seed_id = f"{task_type}-{i:03d}"
            pairs.append(
                E.MatchedRow(
                    seed_id=seed_id,
                    task_type=task_type,
                    task_id=f"{seed_id}:{task_type}",
                    gen_id=gen_id,
                    attempt=1,
                    state="accepted",
                    model="dummy",
                    finish_reason="stop",
                    gates={},
                    judgements={},
                )
            )
            gen_id += 1
    return E.Selection(
        pairs=pairs,
        blocked=False,
        stratum_counts={t: n_per for t in strata},
        excluded={},
        decision=None,
        reason="",
    )


def test_require_pretreatment_manifest_enforces_recovery_configs_declared_strata(
    tmp_path,
):
    """require_pretreatment_manifest (eval_matched.py:1219) is the seam that
    connects configs/data_law_v1_exp_recovery.yaml's eval_cohort_strata to
    validate_pretreatment_manifest, the production gate generate.main calls
    before creating the recovery workdir. If getattr(cfg.build,
    "eval_cohort_strata", None) at eval_matched.py:1240 were ever wrong, the
    rest of the suite would still pass and the arm would still be silently
    blocked (or silently NOT blocked) - this test is the only thing pinning
    that one line against the real config.

    Proves both directions: a manifest matching the recovery config's
    3-stratum declaration is accepted, and the 4-stratum default is refused
    by name (contract-mismatch:strata).
    """
    import dataclasses

    from tuned.data.config import load_build_config

    recovery = load_build_config(
        Path(__file__).parent.parent / "configs" / "data_law_v1_exp_recovery.yaml",
        allow_unpinned=True,
    )
    assert recovery.build.eval_cohort_strata == (
        "irac_analysis",
        "drafting",
        "summarization",
    )
    contract = E.contract_from_config(recovery)

    # -- Direction 1: a 60-pair / 3-strata manifest matching the config is
    # accepted and returned as-is.
    matching_manifest = E.cohort_manifest(
        _cohort_selection(recovery.build.eval_cohort_strata),
        contract=contract,
        strata=recovery.build.eval_cohort_strata,
    )
    assert matching_manifest["n"] == 60
    assert matching_manifest["strata"] == list(recovery.build.eval_cohort_strata)
    matching_path = tmp_path / "matching-manifest.json"
    matching_path.write_text(json.dumps(matching_manifest), encoding="utf-8")
    assert matching_path.is_absolute()
    accepted_cfg = dataclasses.replace(
        recovery,
        build=dataclasses.replace(
            recovery.build, pretreatment_manifest=str(matching_path)
        ),
    )
    result = E.require_pretreatment_manifest(accepted_cfg)
    assert result == matching_manifest

    # -- Direction 2: an 80-pair / 4-strata manifest (the eval_matched
    # four-way default) is refused, and named as a strata mismatch - not
    # silently swallowed into some other reason.
    mismatched_manifest = E.cohort_manifest(
        _cohort_selection(E.TASK_TYPES, gen_id_start=1000),
        contract=contract,
        strata=E.TASK_TYPES,
    )
    assert mismatched_manifest["n"] == 80
    mismatched_path = tmp_path / "mismatched-manifest.json"
    mismatched_path.write_text(json.dumps(mismatched_manifest), encoding="utf-8")
    refused_cfg = dataclasses.replace(
        recovery,
        build=dataclasses.replace(
            recovery.build, pretreatment_manifest=str(mismatched_path)
        ),
    )
    with pytest.raises(ValueError, match="contract-mismatch:strata"):
        E.require_pretreatment_manifest(refused_cfg)


def test_live_style_store_event_is_seen_on_latest_unit(store):
    gen_id = _plant_unit(
        store,
        seed_id="s-budget",
        task_type="drafting",
        gates=_gate_rows(),
        finish_reason="stop",
        state="format_parked",
    )
    store.log_event("reply_over_budget", {"gen_id": gen_id, "over_by": 12})
    unit = [u for u in E.latest_units(store) if u.seed_id == "s-budget"][0]
    assert unit.gen_id == gen_id
    assert any(event.get("kind") == "reply_over_budget" for event in unit.events)
    assert E.hard_format_pass(
        unit.gates,
        finish_reason=unit.finish_reason,
        events=unit.events,
        gen_id=unit.gen_id,
    ) is False
    assert E.hard_format_pass(unit.gates, finish_reason=unit.finish_reason) is True


def test_judging_latest_does_not_inherit_sticky_reply_budget(store):
    store.upsert_seeds([_seed_row("s-judge")])
    store.create_tasks([_task_row("s-judge:drafting", "s-judge", "drafting")])
    first = _record_gen(store, "s-judge:drafting", attempt=1, gates=_gate_rows())
    store.log_event("reply_over_budget", {"gen_id": first, "over_by": 9})
    store.set_task_state(
        "s-judge:drafting", "format_parked", "regenerate:reply_budget"
    )
    second = _record_gen(store, "s-judge:drafting", attempt=2, gates=_gate_rows())
    store.set_task_state("s-judge:drafting", "judging")
    latest = [u for u in E.latest_units(store) if u.seed_id == "s-judge"][0]
    first_unit = E.first_attempt_unit(store, latest)
    assert latest.gen_id == second
    assert latest.state == "judging"
    assert "reply_budget" in latest.disposition
    assert E._format_pass(latest) is True
    assert first_unit.gen_id == first
    assert any(event.get("kind") == "reply_over_budget" for event in first_unit.events)
    assert E._format_pass(first_unit) is False


def test_later_accepted_attempt_does_not_inherit_old_reply_budget(store):
    store.upsert_seeds([_seed_row("s-retry")])
    store.create_tasks([_task_row("s-retry:drafting", "s-retry", "drafting")])
    first = _record_gen(store, "s-retry:drafting", attempt=1, gates=_gate_rows())
    store.log_event("reply_over_budget", {"gen_id": first, "over_by": 9})
    store.set_task_state(
        "s-retry:drafting", "format_parked", "exhausted:format:reply_budget"
    )
    second = _record_gen(store, "s-retry:drafting", attempt=2, gates=_gate_rows())
    store.set_task_state("s-retry:drafting", "accepted", "judge:accept")
    unit = [u for u in E.latest_units(store) if u.seed_id == "s-retry"][0]
    assert unit.gen_id == second
    assert unit.state == "accepted"
    assert unit.disposition == "judge:accept" or "reply_budget" in unit.disposition
    assert E.hard_format_pass(
        unit.gates,
        finish_reason=unit.finish_reason,
        disposition=unit.disposition,
        events=unit.events,
        gen_id=unit.gen_id,
        state=unit.state,
    ) is True


def test_pretreatment_manifest_refused_after_treatment_generations(tmp_path):
    control, treatment, path = _promote_pair(tmp_path)
    try:
        with pytest.raises(ValueError, match="treatment|pre-treatment|generation"):
            E.write_manifest(
                tmp_path / "late.json",
                control,
                treatment_store=treatment,
                contract=_contract(),
            )
        assert path.exists()
    finally:
        control.close()
        treatment.close()


def test_unblocked_evaluate_without_candidate_map_is_lockbox_incomplete(tmp_path):
    control, treatment, manifest_path = _promote_pair(tmp_path)
    try:
        report = E.evaluate(
            control,
            treatment,
            control=_contract(),
            treatment=_contract(),
            manifest_path=manifest_path,
        )
        assert report.decision == "inconclusive"
        assert any("lockbox-incomplete" in r for r in report.reasons)
        assert report.lockbox.valid is False
        assert "lockbox-incomplete" in report.lockbox.reason
        assert report.lockbox.reject_flips == 0
    finally:
        control.close()
        treatment.close()


def test_evaluate_loads_persisted_candidate_decision_map(tmp_path):
    control, treatment, manifest_path = _promote_pair(tmp_path)
    try:
        mapping = _baseline_candidate_map(control)
        reject_gid = next(
            int(row["gen_id"])
            for row in control.gold_labels()
            if row["verdict"] == "reject"
        )
        mapping[reject_gid] = "accept"
        path = tmp_path / "candidate-decisions.json"
        path.write_text(
            json.dumps({str(gid): decision for gid, decision in mapping.items()}),
            encoding="utf-8",
        )
        report = E.evaluate(
            control,
            treatment,
            control=_contract(),
            treatment=_contract(),
            manifest_path=manifest_path,
            candidate_decisions_path=path,
        )
        assert report.lockbox.valid is True
        assert report.lockbox.reject_flips == 1
        assert report.decision == "refuse"
        assert any("reject-flip" in r for r in report.reasons)
    finally:
        control.close()
        treatment.close()


def test_blocked_underfill_short_circuits_before_candidate_map(store):
    for task_type in TASK_TYPES:
        n = 8 if task_type == "statute_qa" else 22
        for i in range(n):
            section = SOURCE_BODY if task_type == "statute_qa" and i >= 4 else SECTION
            _plant_unit(
                store,
                seed_id=f"{task_type}-{i:03d}",
                task_type=task_type,
                gates=_gate_rows(),
                state="accepted",
                section=section,
            )
    report = E.evaluate(
        store,
        store,
        control=_contract(),
        treatment=_contract(),
    )
    assert report.decision == "inconclusive"
    assert any("underfilled" in r or "statute" in r for r in report.reasons)
    assert not any("lockbox-incomplete" in r for r in report.reasons)


def test_evaluate_cli_requires_and_loads_candidate_map(monkeypatch, tmp_path):
    from types import SimpleNamespace

    captured = {}

    def fake_evaluate(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            decision="inconclusive",
            reasons=("ok",),
            synthesis_label="high-confidence pseudo-gold",
        )

    class _Paths:
        state_db = tmp_path / "unused.sqlite3"

    monkeypatch.setattr(E, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        "tuned.data.config.load_build_config",
        lambda *_a, **_k: SimpleNamespace(build=SimpleNamespace(workdir=str(tmp_path))),
    )
    monkeypatch.setattr("tuned.data.paths.build_paths", lambda *_a, **_k: _Paths())
    monkeypatch.setattr(
        E,
        "open_eval_store",
        lambda *_a, **_k: SimpleNamespace(close=lambda: None),
    )

    with pytest.raises(SystemExit):
        E.main(["evaluate", "--manifest", str(tmp_path / "cohort.json")])

    cand = tmp_path / "candidates.json"
    cand.write_text("{}", encoding="utf-8")
    assert (
        E.main(
            [
                "evaluate",
                "--manifest",
                str(tmp_path / "cohort.json"),
                "--candidate-decisions",
                str(cand),
            ]
        )
        == 0
    )
    assert captured["manifest_path"] == str(tmp_path / "cohort.json")
    assert captured.get("candidate_decisions_path") == str(cand)


def test_write_manifest_cli_writes_before_treatment(tmp_path):
    control = Store.open(tmp_path / "control" / "law_v1.sqlite3")
    treatment = Store.open(tmp_path / "treat" / "law_v1.sqlite3")
    control.upsert_source("ikanoon", "CC-BY-4.0")
    treatment.upsert_source("ikanoon", "CC-BY-4.0")
    try:
        for task_type in TASK_TYPES:
            _fill_stratum(
                control,
                task_type,
                20,
                prefix="c",
                format_pass=lambda i: True,
                accept=lambda i: False,
            )
        out = tmp_path / "cli-cohort.json"
        written = E.write_manifest(
            out, control, treatment_store=treatment, contract=_contract()
        )
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["control_fingerprint"] == written["control_fingerprint"]
        assert loaded["n"] == 80
        src = EVAL_SRC.read_text(encoding="utf-8")
        assert "write-manifest" in src
        assert "add_subparsers" in src
    finally:
        control.close()
        treatment.close()


def test_declared_strata_fill_where_the_four_way_default_blocks(store):
    """One pool, two contracts: four strata block on statute_qa, three do not.

    The statute_qa seeds carry a section_text equal to the seed body, which
    is the live control store's actual condition - 270 tasks, 0 eligible
    seeds - so statute_section_eligible refuses every one of them.
    """
    for task_type in TASK_TYPES:
        for i in range(22):
            _plant_unit(
                store,
                seed_id=f"{task_type}-{i:03d}",
                task_type=task_type,
                gates=_gate_rows(),
                state="accepted",
                section=SOURCE_BODY if task_type == "statute_qa" else SECTION,
            )

    four = E.select_cohort(store, n_per=20)
    assert four.blocked is True
    assert "underfilled-stratum:statute_qa" in four.reason

    declared = ("irac_analysis", "drafting", "summarization")
    three = E.select_cohort(store, n_per=20, strata=declared)
    assert three.blocked is False
    assert len(three.pairs) == 60
    assert set(three.stratum_counts) == set(declared)
    assert "statute_qa" not in {row.task_type for row in three.pairs}


def test_unknown_stratum_is_refused_rather_than_silently_empty(store):
    """A typo'd stratum would otherwise select nothing and block on itself."""
    with pytest.raises(ValueError, match="unknown cohort strata"):
        E.select_cohort(store, strata=("irac_analysis", "not_a_task_type"))



def test_manifest_records_strata_and_sizes_n_from_them(tmp_path, store):
    _eligible_pool(store, per_type=25)
    strata = ("irac_analysis", "drafting", "summarization")
    with Store.open(tmp_path / "treat" / "law_v1.sqlite3") as treatment:
        manifest = E.write_manifest(
            tmp_path / "cohort.json",
            store,
            treatment_store=treatment,
            contract=_contract(),
            strata=strata,
        )
    assert manifest["strata"] == list(strata)
    assert manifest["n"] == 60
    assert len(manifest["pairs"]) == 60
    ok, reasons = E.validate_pretreatment_manifest(
        manifest, treatment=_contract(), strata=strata
    )
    assert ok, reasons


def test_manifest_written_for_three_strata_is_refused_against_four(store, tmp_path):
    """A 60-pair cohort must not validate as the 80-pair contract."""
    _eligible_pool(store, per_type=25)
    strata = ("irac_analysis", "drafting", "summarization")
    with Store.open(tmp_path / "treat" / "law_v1.sqlite3") as treatment:
        manifest = E.write_manifest(
            tmp_path / "cohort.json",
            store,
            treatment_store=treatment,
            contract=_contract(),
            strata=strata,
        )
    ok, reasons = E.validate_pretreatment_manifest(manifest, treatment=_contract())
    assert ok is False
    assert "contract-mismatch:strata" in reasons

