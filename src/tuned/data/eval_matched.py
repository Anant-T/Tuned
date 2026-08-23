"""Read-only matched evaluator for the law-v1 recovery cohort.

Selects a fixed 80-seed (20x4) control cohort, pairs latest generations by
(seed_id, task_type), computes locked denominators with Wilson / one-sided
exact McNemar, confirms the 46-label lockbox by gen_id under the
pre-registered dual-judge rule, and returns promote | refuse | inconclusive.

Accepted synthesis is high-confidence pseudo-gold, never gold. This module
does not fit thresholds, write labels, or call a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from tuned.data.gates import DIAGNOSTIC_GATES, GATE_ORDER, PERMANENT_GATES
from tuned.data.judge_policy import decide, slot_verdict
from tuned.data.store import Store

TASK_TYPES = ("irac_analysis", "statute_qa", "drafting", "summarization")
STRATUM_N = 20
FORMAT_PROMOTE_MIN = 0.50
MCNEMAR_ALPHA = 0.05
JUDGE_WILSON_LO_MIN = 0.18
WILSON_Z = 1.96
LOCKBOX_N = 46
LOCKBOX_REJECTS = 36
LOCKBOX_ACCEPTS = 10
TRUNCATION_REASONS = frozenset({"length", "max_tokens"})
SYNTHETIC_GATES = frozenset({"truncated", "reply_budget"})
HARD_FORMAT_GATES = tuple(g for g in GATE_ORDER if g not in DIAGNOSTIC_GATES)
DISPOSITION_FALLBACK_STATES = frozenset({"format_parked", "pending"})
OPERATIONAL_STATES = frozenset(
    {
        "pending",
        "generating",
        "judging",
        "judging_active",
        "gen_unroutable",
        "judge_skipped",
        "judge_error",
        "judge_unroutable",
        "format_parked",
        "input_ineligible",
        "stale_prompt",
    }
)
PSEUDO_GOLD = "high-confidence pseudo-gold"
ALLOWED_CANDIDATE_DECISIONS = frozenset({"accept", "reject", "regenerate"})


# --------------------------------------------------------------------------
# Types.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalContract:
    think_min: int
    teacher_family: str
    teacher_model: str
    gate_contract: tuple[str, ...]


@dataclass
class MatchedRow:
    seed_id: str
    task_type: str
    task_id: str
    gen_id: int
    attempt: int
    state: str
    model: str
    finish_reason: str
    gates: dict
    judgements: dict
    disposition: str = ""
    already_regenerated: bool = False
    model_family: str = ""
    events: tuple = ()


@dataclass(frozen=True)
class Rate:
    successes: int
    n: int
    excluded_operational: int = 0

    @property
    def point(self) -> float | None:
        return None if self.n <= 0 else self.successes / self.n

    @property
    def wilson_lo(self) -> float | None:
        interval = wilson_interval(self.successes, self.n)
        return None if interval is None else interval[0]


@dataclass(frozen=True)
class LegalCounts:
    treatment: int
    control: int


@dataclass
class Selection:
    pairs: list[MatchedRow]
    blocked: bool
    stratum_counts: dict[str, int]
    excluded: dict[str, int]
    decision: str | None
    reason: str


@dataclass
class LockboxReport:
    n_reject: int
    n_accept: int
    reject_flips: int
    new_false_negatives: int
    used_task_state: bool
    baseline_decisions: dict
    valid: bool = True
    reason: str = ""
    new_accept_fns: int = 0


@dataclass
class EvalReport:
    decision: str
    reasons: tuple[str, ...]
    selection: Selection
    metrics: dict
    lockbox: LockboxReport
    synthesis_label: str = PSEUDO_GOLD
    contract_ok: bool = True


# --------------------------------------------------------------------------
# Statistics.
# --------------------------------------------------------------------------


def wilson_interval(
    successes: int, n: int, z: float = WILSON_Z
) -> tuple[float, float, float] | None:
    """Two-sided Wilson score interval (lo, p, hi). None when n==0."""
    if n <= 0:
        return None
    if successes < 0 or successes > n:
        raise ValueError(f"successes {successes} out of range for n={n}")
    p = successes / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / den
    rad = math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    margin = z * rad / den
    return (max(0.0, center - margin), p, min(1.0, center + margin))


def mcnemar_one_sided(b: int, c: int) -> float:
    """One-sided exact McNemar: P(X >= b) for X~Bin(b+c, 1/2).

    `b` is treatment-better discordant pairs; `c` is control-better.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(b, n + 1)) / (2 ** n)


# --------------------------------------------------------------------------
# Format / legal / judge.
# --------------------------------------------------------------------------


def required_gates_complete(gates: Mapping[str, bool] | None) -> bool:
    flags = gates or {}
    return all(gate in flags for gate in HARD_FORMAT_GATES)


def _event_kind_and_gen(event: Mapping) -> tuple[str, int | None]:
    kind = str(event.get("kind") or "")
    detail = event.get("detail")
    if detail is None:
        raw = event.get("detail_json")
        if isinstance(raw, str) and raw:
            try:
                detail = json.loads(raw)
            except json.JSONDecodeError:
                detail = {}
        elif isinstance(raw, Mapping):
            detail = raw
        else:
            detail = {}
    if not isinstance(detail, Mapping):
        detail = {}
    gen_id = detail.get("gen_id")
    return kind, (int(gen_id) if gen_id is not None else None)


def _disposition_applies(state: str | None) -> bool:
    """Task disposition applies only when the latest attempt is parked/pending.

    Sticky regenerate:reply_budget / truncated text must not fail a later
    clean generation that has already moved into judging, judging_active,
    accepted, rejected, or generating.
    """
    return str(state or "") in DISPOSITION_FALLBACK_STATES


def synthetic_failures(
    *,
    finish_reason: str | None = None,
    disposition: str | None = None,
    events: Sequence[Mapping] | None = None,
    gen_id: int | None = None,
    gates: Mapping[str, bool] | None = None,
    state: str | None = None,
) -> frozenset[str]:
    found: set[str] = set()
    if str(finish_reason or "").strip().lower() in TRUNCATION_REASONS:
        found.add("truncated")
    if _disposition_applies(state):
        disp = str(disposition or "")
        if "reply_budget" in disp:
            found.add("reply_budget")
        if "truncated" in disp:
            found.add("truncated")
    for event in events or ():
        kind, event_gen = _event_kind_and_gen(event)
        if event_gen is None:
            continue
        if gen_id is not None and int(event_gen) != int(gen_id):
            continue
        if kind == "reply_over_budget":
            found.add("reply_budget")
        if kind == "generation_truncated":
            found.add("truncated")
    flags = gates or {}
    for name in SYNTHETIC_GATES:
        if flags.get(name) is False:
            found.add(name)
    return frozenset(found)


def hard_format_pass(
    gates: Mapping[str, bool] | None,
    *,
    finish_reason: str | None = None,
    disposition: str | None = None,
    events: Sequence[Mapping] | None = None,
    gen_id: int | None = None,
    state: str | None = None,
) -> bool:
    """Required non-diagnostic gates present and passing; no synthetic cut."""
    if not required_gates_complete(gates):
        return False
    flags = gates or {}
    if any(flags.get(gate) is not True for gate in HARD_FORMAT_GATES):
        return False
    return not synthetic_failures(
        finish_reason=finish_reason,
        disposition=disposition,
        events=events,
        gen_id=gen_id,
        gates=flags,
        state=state,
    )


def legal_fail(gates: Mapping[str, bool] | None) -> bool:
    flags = gates or {}
    return any(flags.get(gate) is False for gate in PERMANENT_GATES)


def statutory_grounding_fail(gates: Mapping[str, bool] | None) -> bool:
    return (gates or {}).get("statutory_grounding") is False


def _as_slot_map(raw) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        if not raw:
            return {}
        first = next(iter(raw.values()))
        if isinstance(first, tuple) and len(first) >= 3:
            return {str(key): tuple(raw[key][:3]) for key in raw}
        if isinstance(first, Mapping) and "grounding" in first:
            out = {}
            for key, row in raw.items():
                axes = (row.get("grounding"), row.get("validity"), row.get("coverage"))
                if None not in axes:
                    out[str(row.get("judge_slot") or key)] = tuple(int(x) for x in axes)
            return out
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        out = {}
        for row in raw:
            if not isinstance(row, Mapping):
                continue
            slot = str(row.get("judge_slot") or "")
            axes = (row.get("grounding"), row.get("validity"), row.get("coverage"))
            if slot and None not in axes:
                out[slot] = (int(axes[0]), int(axes[1]), int(axes[2]))
        return out
    return {}


def dual_judge_decision(slots, *, already_regenerated: bool = False) -> str | None:
    """Pre-registered provisional dual-judge rule. None if a or b is missing."""
    mapped = _as_slot_map(slots)
    if "a" not in mapped or "b" not in mapped:
        return None
    verdicts = [slot_verdict(*mapped["a"][:3]), slot_verdict(*mapped["b"][:3])]
    if "tiebreak" in mapped:
        verdicts.append(slot_verdict(*mapped["tiebreak"][:3]))
    return decide(verdicts, already_regenerated=already_regenerated)


def judge_accept_cond(rows: Sequence[MatchedRow]) -> Rate:
    successes = 0
    n = 0
    excluded = 0
    for row in rows:
        if not hard_format_pass(
            row.gates,
            finish_reason=row.finish_reason,
            disposition=row.disposition,
            events=row.events,
            gen_id=row.gen_id,
            state=row.state,
        ):
            continue
        if row.state in OPERATIONAL_STATES:
            excluded += 1
            continue
        decision = dual_judge_decision(
            row.judgements, already_regenerated=row.already_regenerated
        )
        if decision not in ("accept", "reject"):
            excluded += 1
            continue
        n += 1
        if decision == "accept":
            successes += 1
    return Rate(successes=successes, n=n, excluded_operational=excluded)


# --------------------------------------------------------------------------
# Lockbox.
# --------------------------------------------------------------------------


def _already_regenerated_for_gen(store, gen_id: int) -> bool:
    row = store.conn.execute(
        "SELECT task_id FROM generation WHERE gen_id = ?",
        (int(gen_id),),
    ).fetchone()
    if row is None:
        return False
    return _already_regenerated(store, str(row[0]))


def _normalize_candidate_decisions(
    candidate_decisions: Mapping | None,
) -> dict[int, str] | None:
    if candidate_decisions is None:
        return None
    if not isinstance(candidate_decisions, Mapping):
        return None
    out: dict[int, str] = {}
    try:
        for key, value in candidate_decisions.items():
            out[int(key)] = str(value)
    except (TypeError, ValueError):
        return None
    return out


def candidate_map_complete(
    candidate_decisions: Mapping | None, labelled_ids: Sequence[int]
) -> bool:
    normalized = _normalize_candidate_decisions(candidate_decisions)
    if normalized is None:
        return False
    wanted = {int(gid) for gid in labelled_ids}
    if set(normalized) != wanted or len(normalized) != len(wanted):
        return False
    return all(value in ALLOWED_CANDIDATE_DECISIONS for value in normalized.values())


def load_candidate_decisions(path: str | Path) -> dict[int, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    normalized = _normalize_candidate_decisions(raw)
    if normalized is None:
        raise ValueError("lockbox-incomplete: candidate decisions must be a JSON object")
    return normalized


def lockbox_confirm(
    labels: Sequence[Mapping],
    judgements_by_gen: Mapping,
    *,
    store=None,
    candidate_decisions: Mapping[int, str] | None = None,
) -> LockboxReport:
    """Confirm the locked labels by gen_id. Never reads task.state.

    Baseline is the frozen R0 dual-judge rule on stored judgements, with
    `already_regenerated` taken from each labelled generation's persisted
    params/history. `candidate_decisions` is a required precomputed map
    from recovery re-judging under that same rule, keyed by the frozen
    labelled gen_ids. It cannot change labels or cardinality. A missing
    or incomplete map is lockbox-incomplete; it does not default to baseline.
    """
    baseline_decisions = {}
    reject_flips = 0
    new_fn = 0
    n_reject = 0
    n_accept = 0
    seen: list[int] = []
    for label in labels:
        seen.append(int(label["gen_id"]))
    normalized = _normalize_candidate_decisions(candidate_decisions)
    map_ok = candidate_map_complete(normalized, seen)
    if map_ok:
        assert normalized is not None
    for label in labels:
        gid = int(label["gen_id"])
        verdict = str(label.get("verdict") or "")
        slots = _as_slot_map(judgements_by_gen.get(gid, {}))
        already = False
        if store is not None:
            already = _already_regenerated_for_gen(store, gid)
        baseline = dual_judge_decision(slots, already_regenerated=already)
        baseline_decisions[gid] = baseline
        candidate = normalized[gid] if map_ok else None
        if verdict == "reject":
            n_reject += 1
            if map_ok and candidate == "accept" and baseline != "accept":
                reject_flips += 1
        elif verdict == "accept":
            n_accept += 1
            if map_ok and baseline == "accept" and candidate != "accept":
                new_fn += 1
    unique = set(seen)
    label_ok = (
        len(seen) == LOCKBOX_N
        and len(unique) == LOCKBOX_N
        and n_reject == LOCKBOX_REJECTS
        and n_accept == LOCKBOX_ACCEPTS
    )
    if not label_ok:
        reason = "lockbox-cardinality"
        valid = False
    elif not map_ok:
        reason = "lockbox-incomplete"
        valid = False
    else:
        reason = ""
        valid = True
    return LockboxReport(
        n_reject=n_reject,
        n_accept=n_accept,
        reject_flips=reject_flips,
        new_false_negatives=new_fn,
        used_task_state=False,
        baseline_decisions=baseline_decisions,
        valid=valid,
        reason=reason,
        new_accept_fns=new_fn,
    )


# --------------------------------------------------------------------------
# Cohort selection.
# --------------------------------------------------------------------------


def _rank(seed_id: str, task_type: str) -> str:
    return hashlib.sha256(f"{seed_id}\0{task_type}".encode("utf-8")).hexdigest()


def _norm_model(model) -> str:
    name = str(model or "")
    return name.rsplit("/", 1)[-1] if name else ""


def is_gpt_oss_120b(model) -> bool:
    return _norm_model(model) == "gpt-oss-120b"


def statute_section_eligible(seed: Mapping | None) -> bool:
    if seed is None:
        return False
    meta = seed.get("meta_json")
    if isinstance(meta, str) and meta:
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {}
    if not isinstance(meta, Mapping):
        meta = {}
    provision = " ".join(str(meta.get("section_text") or "").split())
    if not provision:
        return False
    body = " ".join(str(seed.get("text") or "").split())
    return provision != body


def gold_linked_seeds(store) -> set[str]:
    return {
        row[0]
        for row in store.conn.execute(
            "SELECT DISTINCT t.seed_id FROM gold_label g "
            "JOIN generation n ON n.gen_id = g.gen_id "
            "JOIN task t ON t.task_id = n.task_id"
        )
    }


def _events_by_gen(store) -> dict[int, tuple]:
    out: dict[int, list] = {}
    for event in store.events():
        kind, gen_id = _event_kind_and_gen(event)
        if gen_id is None:
            continue
        packed = {"kind": kind, "detail": {"gen_id": gen_id}}
        out.setdefault(gen_id, []).append(packed)
    return {gid: tuple(rows) for gid, rows in out.items()}


def _already_regenerated(store, task_id: str) -> bool:
    rows = store.conn.execute(
        "SELECT params_json FROM generation WHERE task_id = ?", (task_id,)
    ).fetchall()
    for (params,) in rows:
        try:
            data = json.loads(params) if params else {}
        except ValueError:
            continue
        if isinstance(data, dict) and data.get("reviewer_note_applied"):
            return True
    return False


def _unit_from(
    row: Mapping,
    gates_map: Mapping,
    judgements_map: Mapping,
    *,
    events_map: Mapping | None = None,
    disposition: str = "",
    already_regenerated: bool = False,
) -> MatchedRow:
    gid = int(row["gen_id"])
    events = tuple((events_map or {}).get(gid) or ())
    return MatchedRow(
        seed_id=str(row["seed_id"]),
        task_type=str(row["task_type"]),
        task_id=str(row["task_id"]),
        gen_id=gid,
        attempt=int(row["attempt"]),
        state=str(row.get("task_state") or row.get("state") or "pending"),
        model=str(row.get("model") or ""),
        finish_reason=str(row.get("finish_reason") or "stop"),
        gates=dict(gates_map.get(gid) or {}),
        judgements=_as_slot_map(judgements_map.get(gid, [])),
        disposition=str(row.get("disposition") or disposition or ""),
        already_regenerated=already_regenerated,
        model_family=str(row.get("model_family") or ""),
        events=events,
    )


def latest_units(store) -> list[MatchedRow]:
    """One latest generation per (seed_id, task_type)."""
    best: dict[tuple[str, str], dict] = {}
    events_map = _events_by_gen(store)
    for row in store.latest_generations():
        key = (row["seed_id"], row["task_type"])
        cur = best.get(key)
        if cur is None or (row["attempt"], row["gen_id"]) > (cur["attempt"], cur["gen_id"]):
            best[key] = row
    rows = list(best.values())
    gates_map = store.gates_by_gen(r["gen_id"] for r in rows)
    judgements_map = store.judgements_by_gen(r["gen_id"] for r in rows)
    units = []
    for row in rows:
        task = store.get_task(row["task_id"]) or {}
        units.append(
            _unit_from(
                row,
                gates_map,
                judgements_map,
                events_map=events_map,
                disposition=str(task.get("disposition") or ""),
                already_regenerated=_already_regenerated(store, row["task_id"]),
            )
        )
    return units


def first_attempt_unit(store, unit: MatchedRow) -> MatchedRow:
    row = store.conn.execute(
        "SELECT * FROM generation WHERE task_id = ? ORDER BY attempt ASC LIMIT 1",
        (unit.task_id,),
    ).fetchone()
    if row is None:
        return unit
    row = dict(row)
    events_map = _events_by_gen(store)
    gates = store.gates_for(int(row["gen_id"]))
    judgements = store.judgements_for(int(row["gen_id"]))
    return replace(
        unit,
        gen_id=int(row["gen_id"]),
        attempt=int(row["attempt"]),
        finish_reason=str(row.get("finish_reason") or "stop"),
        gates=dict(gates),
        judgements=_as_slot_map(judgements),
        disposition="",
        already_regenerated=False,
        model=str(row.get("model") or unit.model),
        model_family=str(row.get("model_family") or unit.model_family),
        events=tuple(events_map.get(int(row["gen_id"])) or ()),
    )


def select_cohort(store, n_per: int = STRATUM_N) -> Selection:
    gold = gold_linked_seeds(store)
    excluded = {
        "stale": 0,
        "no_generation": 0,
        "non_gpt_oss": 0,
        "gold": 0,
        "ineligible_statute": 0,
    }
    latest_by_task = {row["task_id"]: row for row in store.latest_generations()}
    events_map = _events_by_gen(store)
    seed_cache: dict[str, dict | None] = {}
    buckets: dict[str, dict[str, dict]] = {name: {} for name in TASK_TYPES}

    for task in store.conn.execute("SELECT * FROM task"):
        task = dict(task)
        task_type = task["task_type"]
        if task_type not in TASK_TYPES:
            continue
        seed_id = task["seed_id"]
        if seed_id in gold:
            excluded["gold"] += 1
            continue
        disposition = str(task.get("disposition") or "")
        if task["state"] == "stale_prompt" or disposition.startswith("stale-prompt"):
            excluded["stale"] += 1
            continue
        gen = latest_by_task.get(task["task_id"])
        if gen is None:
            excluded["no_generation"] += 1
            continue
        if not is_gpt_oss_120b(gen.get("model")):
            excluded["non_gpt_oss"] += 1
            continue
        if task_type == "statute_qa":
            if seed_id not in seed_cache:
                seed_cache[seed_id] = store.get_seed(seed_id)
            if not statute_section_eligible(seed_cache[seed_id]):
                excluded["ineligible_statute"] += 1
                continue
        packed = dict(gen)
        packed["seed_id"] = seed_id
        packed["task_type"] = task_type
        packed["task_state"] = task["state"]
        packed["disposition"] = disposition
        prev = buckets[task_type].get(seed_id)
        if prev is None or (packed["attempt"], packed["gen_id"]) > (
            prev["attempt"],
            prev["gen_id"],
        ):
            buckets[task_type][seed_id] = packed

    pairs: list[MatchedRow] = []
    counts: dict[str, int] = {}
    blocked = False
    reasons: list[str] = []
    for task_type in TASK_TYPES:
        ranked = sorted(
            buckets[task_type].values(),
            key=lambda row: _rank(row["seed_id"], task_type),
        )
        taken = ranked[:n_per]
        counts[task_type] = len(taken)
        if len(taken) < n_per:
            blocked = True
            reasons.append(f"underfilled-stratum:{task_type}")
        gates_map = store.gates_by_gen(r["gen_id"] for r in taken)
        judgements_map = store.judgements_by_gen(r["gen_id"] for r in taken)
        pairs.extend(
            _unit_from(
                row,
                gates_map,
                judgements_map,
                events_map=events_map,
                disposition=str(row.get("disposition") or ""),
                already_regenerated=_already_regenerated(store, row["task_id"]),
            )
            for row in taken
        )

    return Selection(
        pairs=pairs,
        blocked=blocked,
        stratum_counts=counts,
        excluded=excluded,
        decision="inconclusive" if blocked else None,
        reason=";".join(reasons),
    )


# --------------------------------------------------------------------------
# Contracts and promotion.
# --------------------------------------------------------------------------


def contract_from_config(cfg) -> EvalContract:
    refs = cfg.routing_refs("generator")
    _provider, model = cfg.model_for(refs[0])
    return EvalContract(
        think_min=int(cfg.build.length_band.think_min),
        teacher_family=str(model.family),
        teacher_model=str(model.id),
        gate_contract=tuple(GATE_ORDER),
    )


def _teacher_of(rows: Sequence[MatchedRow]) -> tuple[set[str], set[str]]:
    models = {_norm_model(row.model) for row in rows if row.gen_id >= 0 and row.model}
    families = {row.model_family for row in rows if row.gen_id >= 0 and row.model_family}
    return models, families


def validate_manifest(
    manifest: Mapping | None,
    selection: Selection,
    *,
    control: EvalContract,
    treatment: EvalContract,
    treatment_rows: Sequence[MatchedRow],
) -> tuple[bool, tuple[str, ...]]:
    if manifest is None:
        return False, ("missing-manifest",)
    reasons: list[str] = []
    if tuple(manifest.get("gate_contract") or ()) != tuple(GATE_ORDER):
        reasons.append("contract-mismatch:gate_contract")
    if tuple(control.gate_contract) != tuple(GATE_ORDER) or tuple(
        treatment.gate_contract
    ) != tuple(GATE_ORDER):
        reasons.append("contract-mismatch:gate_contract")
    if (
        manifest.get("think_min") != control.think_min
        or control.think_min != treatment.think_min
    ):
        reasons.append("contract-mismatch:think_min")
    if control.teacher_family != treatment.teacher_family or control.teacher_model != (
        treatment.teacher_model
    ):
        reasons.append("contract-mismatch:teacher")
    if (
        manifest.get("teacher_family") not in (None, control.teacher_family)
        or manifest.get("teacher_model") not in (None, control.teacher_model)
    ):
        reasons.append("contract-mismatch:teacher")
    control_models, control_families = _teacher_of(selection.pairs)
    treat_models, treat_families = _teacher_of(treatment_rows)
    expected_model = _norm_model(control.teacher_model)
    if control_models - {expected_model} or treat_models - {expected_model}:
        reasons.append("contract-mismatch:observed-teacher")
    if control_families - {control.teacher_family} or treat_families - {
        treatment.teacher_family
    }:
        reasons.append("contract-mismatch:observed-teacher")
    selected_keys = {
        (row.seed_id, row.task_type, row.task_id, row.gen_id) for row in selection.pairs
    }
    man_pairs = list(manifest.get("pairs") or ())
    man_keys = {
        (
            row["seed_id"],
            row["task_type"],
            row["control_task_id"],
            row["control_gen_id"],
        )
        for row in man_pairs
    }
    if selected_keys != man_keys:
        reasons.append("contract-mismatch:manifest-ids")
    treat_keys = {(row.seed_id, row.task_type) for row in treatment_rows if row.gen_id >= 0}
    if any((row["seed_id"], row["task_type"]) not in treat_keys for row in man_pairs):
        reasons.append("contract-mismatch:treatment-pairs")
    expected_fp = control_snapshot_fingerprint(selection, control)
    if manifest.get("control_fingerprint") != expected_fp:
        reasons.append("contract-mismatch:fingerprint")
    return (not reasons, tuple(dict.fromkeys(reasons)))


def decide_promotion(
    *,
    format_latest: float,
    mcnemar_p: float,
    judging_complete: bool,
    judge_accept_wilson_lo: float | None,
    treatment_legal_fails: int,
    control_legal_fails: int,
    lockbox_reject_flips: int,
    lockbox_new_false_negatives: int,
    types_with_format_pass: set[str],
    generated_types: set[str],
    selection_blocked: bool,
    contract_ok: bool,
    missing_gate_data: bool,
    n_pairs: int,
    lockbox_valid: bool = True,
    lockbox_reason: str = "",
) -> tuple[str, tuple[str, ...]]:
    if selection_blocked:
        return "inconclusive", ("underfilled-stratum",)
    if not contract_ok:
        return "inconclusive", ("contract-mismatch",)
    if missing_gate_data:
        return "inconclusive", ("missing-gate-data",)
    if not lockbox_valid:
        return "inconclusive", (lockbox_reason or "lockbox-cardinality",)
    if not judging_complete:
        return "inconclusive", ("incomplete-judging",)

    reasons: list[str] = []
    inconclusive: list[str] = []
    if format_latest < FORMAT_PROMOTE_MIN:
        reasons.append(f"format_latest<{FORMAT_PROMOTE_MIN}")
    if mcnemar_p >= MCNEMAR_ALPHA:
        inconclusive.append("insufficient-evidence:mcnemar")
    if judge_accept_wilson_lo is None or judge_accept_wilson_lo < JUDGE_WILSON_LO_MIN:
        reasons.append("judge_accept-wilson-lo")
    if treatment_legal_fails > control_legal_fails:
        reasons.append("legal-fail-increase")
    if lockbox_reject_flips:
        reasons.append("lockbox-reject-flip")
    if lockbox_new_false_negatives:
        reasons.append("lockbox-false-negative")
    missing_types = sorted(generated_types - types_with_format_pass)
    if missing_types:
        reasons.append("zero-format-type:" + ",".join(missing_types))
    if inconclusive and not reasons:
        return "inconclusive", tuple(inconclusive)
    if reasons:
        return "refuse", tuple(reasons + inconclusive)
    return "promote", ()


def _missing_unit(seed_id: str, task_type: str) -> MatchedRow:
    return MatchedRow(
        seed_id=seed_id,
        task_type=task_type,
        task_id="",
        gen_id=-1,
        attempt=0,
        state="pending",
        model="",
        finish_reason="stop",
        gates={},
        judgements={},
    )


def _empty_metrics() -> dict:
    return {
        "format_latest": Rate(0, 0),
        "format_first": Rate(0, 0),
        "judge_accept_cond": Rate(0, 0),
        "joint_yield": Rate(0, 0),
        "legal_fail": LegalCounts(0, 0),
        "statutory_grounding": LegalCounts(0, 0),
        "mcnemar_p": 1.0,
    }


def _format_pass(row: MatchedRow) -> bool:
    return hard_format_pass(
        row.gates,
        finish_reason=row.finish_reason,
        disposition=row.disposition,
        events=row.events,
        gen_id=row.gen_id,
        state=row.state,
    )


def evaluate(
    control_store,
    treatment_store,
    *,
    control: EvalContract | None = None,
    treatment: EvalContract | None = None,
    control_cfg=None,
    treatment_cfg=None,
    manifest_path: str | Path | None = None,
    candidate_decisions: Mapping[int, str] | None = None,
    candidate_decisions_path: str | Path | None = None,
    manifest=None,
    n_per: int = STRATUM_N,
) -> EvalReport:
    if manifest is not None:
        raise TypeError("evaluate requires manifest_path, not an in-memory manifest")
    if is_live_store(control_store) and not is_readonly(control_store):
        raise ValueError("read-only live URI is required (mode=ro)")
    if is_live_store(treatment_store) and not is_readonly(treatment_store):
        raise ValueError("read-only live URI is required (mode=ro)")

    if control_cfg is not None:
        control = contract_from_config(control_cfg)
    if treatment_cfg is not None:
        treatment = contract_from_config(treatment_cfg)
    if control is None or treatment is None:
        raise ValueError("evaluate needs control and treatment contracts or configs")

    selected = select_cohort(control_store, n_per=n_per)
    empty_box = LockboxReport(0, 0, 0, 0, False, {}, valid=False, reason="lockbox-cardinality")
    empty_metrics = _empty_metrics()

    if selected.blocked:
        return EvalReport(
            decision="inconclusive",
            reasons=(selected.reason or "underfilled-stratum",),
            selection=selected,
            metrics=empty_metrics,
            lockbox=empty_box,
            contract_ok=True,
        )
    if manifest_path is None:
        return EvalReport(
            decision="inconclusive",
            reasons=("missing-manifest",),
            selection=selected,
            metrics=empty_metrics,
            lockbox=empty_box,
            contract_ok=False,
        )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    treat_index = {(u.seed_id, u.task_type): u for u in latest_units(treatment_store)}
    paired: list[tuple[MatchedRow, MatchedRow]] = []
    missing_gate_data = False
    for cu in selected.pairs:
        tu = treat_index.get((cu.seed_id, cu.task_type))
        if tu is None:
            tu = _missing_unit(cu.seed_id, cu.task_type)
            missing_gate_data = True
        paired.append((cu, tu))

    contract_ok, contract_reasons = validate_manifest(
        manifest,
        selected,
        control=control,
        treatment=treatment,
        treatment_rows=[tu for _cu, tu in paired],
    )
    if not contract_ok:
        return EvalReport(
            decision="inconclusive",
            reasons=contract_reasons or ("contract-mismatch",),
            selection=selected,
            metrics=empty_metrics,
            lockbox=empty_box,
            contract_ok=False,
        )
    if candidate_decisions_path is not None:
        candidate_decisions = load_candidate_decisions(candidate_decisions_path)
    if candidate_decisions is None:
        incomplete = LockboxReport(
            0, 0, 0, 0, False, {}, valid=False, reason="lockbox-incomplete"
        )
        return EvalReport(
            decision="inconclusive",
            reasons=("lockbox-incomplete",),
            selection=selected,
            metrics=empty_metrics,
            lockbox=incomplete,
            contract_ok=True,
        )

    treat_latest = 0
    treat_first = 0
    control_format = 0
    b = c = 0
    treat_legal = 0
    control_legal = 0
    treat_ground = 0
    control_ground = 0
    types_with_pass: set[str] = set()
    generated_types: set[str] = set()
    judging_complete = True
    treat_rows: list[MatchedRow] = []

    for cu, tu in paired:
        generated_types.add(tu.task_type or cu.task_type)
        if cu.gen_id >= 0 and not required_gates_complete(cu.gates):
            missing_gate_data = True
        if tu.gen_id >= 0 and not required_gates_complete(tu.gates):
            missing_gate_data = True
        c_pass = _format_pass(cu)
        t_pass = _format_pass(tu)
        if t_pass:
            treat_latest += 1
            types_with_pass.add(tu.task_type)
        if c_pass:
            control_format += 1
        if t_pass and not c_pass:
            b += 1
        elif c_pass and not t_pass:
            c += 1
        if legal_fail(tu.gates):
            treat_legal += 1
        if legal_fail(cu.gates):
            control_legal += 1
        if statutory_grounding_fail(tu.gates):
            treat_ground += 1
        if statutory_grounding_fail(cu.gates):
            control_ground += 1
        first = first_attempt_unit(treatment_store, tu) if tu.gen_id >= 0 else tu
        if _format_pass(first):
            treat_first += 1
        if t_pass and dual_judge_decision(
            tu.judgements, already_regenerated=tu.already_regenerated
        ) not in ("accept", "reject"):
            judging_complete = False
        treat_rows.append(tu)

    labels = control_store.gold_labels()
    box = lockbox_confirm(
        labels,
        control_store.judgements_by_gen(int(row["gen_id"]) for row in labels),
        store=control_store,
        candidate_decisions=candidate_decisions,
    )
    accept_cond = judge_accept_cond(treat_rows)
    accepted = sum(
        1
        for row in treat_rows
        if dual_judge_decision(row.judgements, already_regenerated=row.already_regenerated)
        == "accept"
    )
    n_pairs = len(paired)
    metrics = {
        "format_latest": Rate(treat_latest, n_pairs),
        "format_first": Rate(treat_first, n_pairs),
        "judge_accept_cond": accept_cond,
        "joint_yield": Rate(accepted, n_pairs),
        "legal_fail": LegalCounts(treat_legal, control_legal),
        "statutory_grounding": LegalCounts(treat_ground, control_ground),
        "mcnemar_p": mcnemar_one_sided(b, c),
        "control_format_latest": Rate(control_format, n_pairs),
    }
    verdict, reasons = decide_promotion(
        format_latest=metrics["format_latest"].point or 0.0,
        mcnemar_p=metrics["mcnemar_p"],
        judging_complete=judging_complete,
        judge_accept_wilson_lo=accept_cond.wilson_lo,
        treatment_legal_fails=treat_legal,
        control_legal_fails=control_legal,
        lockbox_reject_flips=box.reject_flips,
        lockbox_new_false_negatives=box.new_false_negatives,
        types_with_format_pass=types_with_pass,
        generated_types=generated_types or set(TASK_TYPES),
        selection_blocked=False,
        contract_ok=True,
        missing_gate_data=missing_gate_data,
        n_pairs=n_pairs,
        lockbox_valid=box.valid,
        lockbox_reason=box.reason,
    )
    return EvalReport(
        decision=verdict,
        reasons=reasons,
        selection=selected,
        metrics=metrics,
        lockbox=box,
        contract_ok=True,
    )


# --------------------------------------------------------------------------
# Manifest and read-only open.
# --------------------------------------------------------------------------


def control_snapshot_fingerprint(selection: Selection, contract: EvalContract) -> str:
    payload = {
        "gate_contract": list(contract.gate_contract),
        "n_per_stratum": STRATUM_N,
        "pairs": [
            {
                "control_attempt": row.attempt,
                "control_gen_id": row.gen_id,
                "control_task_id": row.task_id,
                "seed_id": row.seed_id,
                "task_type": row.task_type,
            }
            for row in selection.pairs
        ],
        "teacher_family": contract.teacher_family,
        "teacher_model": contract.teacher_model,
        "think_min": contract.think_min,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cohort_manifest(selection: Selection, *, contract: EvalContract) -> dict:
    pairs = [
        {
            "seed_id": row.seed_id,
            "task_type": row.task_type,
            "control_task_id": row.task_id,
            "control_gen_id": row.gen_id,
            "control_attempt": row.attempt,
        }
        for row in selection.pairs
    ]
    return {
        "n": len(selection.pairs),
        "n_per_stratum": STRATUM_N,
        "task_types": list(TASK_TYPES),
        "think_min": contract.think_min,
        "teacher_family": contract.teacher_family,
        "teacher_model": contract.teacher_model,
        "gate_contract": list(contract.gate_contract),
        "control_fingerprint": control_snapshot_fingerprint(selection, contract),
        "pairs": pairs,
    }


def validate_pretreatment_manifest(
    manifest: Mapping | None, *, treatment: EvalContract
) -> tuple[bool, tuple[str, ...]]:
    """Validate a persisted 80-pair pre-treatment manifest against treatment."""
    if not isinstance(manifest, Mapping):
        return False, ("pretreatment-manifest-invalid",)
    reasons: list[str] = []
    pairs = list(manifest.get("pairs") or ())
    if manifest.get("n") != 80 or len(pairs) != 80:
        reasons.append("pretreatment-manifest-n")
    counts = {name: 0 for name in TASK_TYPES}
    for row in pairs:
        if not isinstance(row, Mapping):
            continue
        task_type = row.get("task_type")
        if task_type in counts:
            counts[task_type] += 1
    for task_type in TASK_TYPES:
        if counts[task_type] != STRATUM_N:
            reasons.append(f"underfilled-stratum:{task_type}")
    if tuple(manifest.get("gate_contract") or ()) != tuple(GATE_ORDER):
        reasons.append("contract-mismatch:gate_contract")
    if tuple(treatment.gate_contract) != tuple(GATE_ORDER):
        reasons.append("contract-mismatch:gate_contract")
    if manifest.get("think_min") != treatment.think_min:
        reasons.append("contract-mismatch:think_min")
    if manifest.get("teacher_family") != treatment.teacher_family or _norm_model(
        manifest.get("teacher_model")
    ) != _norm_model(treatment.teacher_model):
        reasons.append("contract-mismatch:teacher")
    return (not reasons, tuple(dict.fromkeys(reasons)))


def require_pretreatment_manifest(cfg, *, repo_root=None) -> dict:
    """Load and validate the opted-in pre-treatment manifest, or no-op.

    Called from generate.main before BuildPaths.ensure / router / provider.
    """
    if not getattr(cfg.build, "require_pretreatment_manifest", False):
        return {}
    raw_path = cfg.build.pretreatment_manifest
    if not raw_path:
        raise ValueError("pretreatment manifest path is required")
    path = Path(raw_path)
    if not path.is_absolute():
        from tuned.data.paths import package_repo_root

        path = (repo_root or package_repo_root()) / path
    if not path.is_file():
        raise ValueError(f"pretreatment manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    ok, reasons = validate_pretreatment_manifest(
        manifest, treatment=contract_from_config(cfg)
    )
    if not ok:
        raise ValueError("pretreatment manifest refused: " + ",".join(reasons))
    return manifest


def treatment_has_generations(store) -> bool:
    row = store.conn.execute("SELECT 1 FROM generation LIMIT 1").fetchone()
    return row is not None


def write_manifest(
    path: str | Path,
    control_store,
    *,
    treatment_store=None,
    treatment_empty: bool = False,
    contract: EvalContract,
    n_per: int = STRATUM_N,
) -> dict:
    if treatment_store is None and not treatment_empty:
        raise ValueError("treatment store is required to prove pre-treatment")
    if treatment_store is not None and treatment_has_generations(treatment_store):
        raise ValueError(
            "pre-treatment manifest refused: treatment already has generations"
        )
    selected = select_cohort(control_store, n_per=n_per)
    if selected.blocked:
        raise ValueError(selected.reason or "underfilled-stratum")
    manifest = cohort_manifest(selected, contract=contract)
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def readonly_uri(path: str | Path) -> str:
    return Path(path).resolve().as_uri() + "?mode=ro"


def open_eval_store(path: str | Path, *, readonly: bool = True) -> Store:
    if not readonly:
        raise ValueError("read-only live URI is required (mode=ro)")
    uri = readonly_uri(path)
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA foreign_keys = ON")
    return Store(conn, Path(path))


def is_readonly(store) -> bool:
    try:
        return bool(store.conn.execute("PRAGMA query_only").fetchone()[0])
    except sqlite3.Error:
        return False


def is_live_store(store) -> bool:
    from tuned.data.paths import (
        LIVE_STATE_DB,
        package_repo_root,
        resolve_config_workdir,
        workdir_key,
    )

    live = resolve_config_workdir(LIVE_STATE_DB, repo_root=package_repo_root())
    return workdir_key(store.path) == workdir_key(live)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from tuned.data.config import load_build_config
    from tuned.data.paths import build_paths

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--control-config", default="configs/data_law_v1.yaml")
    parser.add_argument("--treatment-config", default="configs/data_law_v1_exp_recovery.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser(
        "write-manifest",
        help="Write the pre-treatment cohort manifest (refuses if treatment has generations).",
    )
    write.add_argument("--out", required=True)
    ev = sub.add_parser("evaluate", help="Run the matched evaluator against a persisted manifest.")
    ev.add_argument("--manifest", required=True)
    ev.add_argument(
        "--candidate-decisions",
        required=True,
        help="Persisted JSON map of lockbox gen_id -> candidate decision.",
    )
    args = parser.parse_args(argv)

    control_cfg = load_build_config(args.control_config, allow_unpinned=True)
    treatment_cfg = load_build_config(args.treatment_config, allow_unpinned=True)
    control_db = build_paths(control_cfg.build.workdir).state_db
    treatment_db = build_paths(treatment_cfg.build.workdir).state_db

    if args.command == "write-manifest":
        control_contract = contract_from_config(control_cfg)
        control_store = open_eval_store(control_db)
        try:
            if Path(treatment_db).exists():
                treatment_store = open_eval_store(treatment_db)
                try:
                    write_manifest(
                        args.out,
                        control_store,
                        treatment_store=treatment_store,
                        contract=control_contract,
                    )
                finally:
                    treatment_store.close()
            else:
                write_manifest(
                    args.out,
                    control_store,
                    treatment_empty=True,
                    contract=control_contract,
                )
        finally:
            control_store.close()
        return 0

    control_store = open_eval_store(control_db)
    treatment_store = open_eval_store(treatment_db)
    try:
        report = evaluate(
            control_store,
            treatment_store,
            control_cfg=control_cfg,
            treatment_cfg=treatment_cfg,
            manifest_path=args.manifest,
            candidate_decisions_path=args.candidate_decisions,
        )
        print(f"decision={report.decision} synthesis={report.synthesis_label}")
        print("reasons=" + ",".join(report.reasons))
    finally:
        control_store.close()
        treatment_store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
