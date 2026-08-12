"""Judge worker - two blind graders per generation, and what to do when they disagree.

Consumes the tasks generate.py parked in 'judging' and decides whether the
row enters the dataset. Four things here are load-bearing and each one is a
contract from an earlier review, not a preference.

FAMILY SEPARATION IS PER CALL (contract 5). A judge may not share a model
family with the generator that produced the row - a model grading its own
family's prose is grading its own habits. config.py only checks that the
pool makes this POSSIBLE; the exclusion itself can only be applied here,
because only here is it known which model actually answered. Judge B
additionally excludes judge A's family, so "two judges" never means "one
model twice".

CONTEXT LENGTH IS A ROUTING INPUT (contract 2). The judge prompt is the
longest in the pipeline: the same materials the generator saw PLUS the
candidate's trace and answer. Two judge-pool models are 8k-context. A
candidate that does not fit must be routed to a 32k+ judge, because a
silently truncated judge prompt produces a score for an answer nobody read -
which is worse than not judging it at all. Router.pick exposes no
context filter, so the length check is turned into a family exclusion here
(undersized_families), which is exact as long as a family's models in a role
are all the same size - and it degrades safely if they are not: a family is
only excluded when EVERY one of its models in that role is too small.

THE PARSER IS DEFENSIVE (contract 3). Judges are free-tier models that
occasionally wrap their JSON in prose, in a fence, or in an apology. The
parser accepts the axis aliases the rubric uses (grounding_faithfulness /
reasoning_validity / issue_coverage as well as the short names), finds JSON
anywhere in the reply, and treats an unparsable answer as ONE retried judge
slot - never as a crash, and never as a score.

EMPTY-THINK ROWS ARE NEVER JUDGED (contract 4). The empty-think slice is
copied, not generated, so it never reaches this queue; if one ever does -
a routing bug - it is parked with a diagnostic instead of being scored on a
trace that does not exist.

Raw-first durability is the same rule as generate.py: the judge's reply is
appended to raw/judge/<day>/ BEFORE record_judgement. An UNPARSABLE reply is
appended too, but under kind="judge_error" rather than "judgement", so that
store.reconcile_raw skips it: recovering it as a judgement would write a row
with three NULL scores that later reads as a judgement nobody made.

Run:  python -m tuned.data.judge --config configs/data_law_v1.yaml
      --stream synthesis --n-workers 4 [--forever] [--max-batches N]
"""

import asyncio
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from tuned.data import prompt_registry
from tuned.data.config import ModelRef
from tuned.data.generate import (
    SlotError,
    apply_gate_disposition,
    build_prompt,
    generate_once,
    make_router,
    record_usage,
    worker_name,
)
from tuned.data.jsonl import append_ndjson
from tuned.data.providers import ProviderError
from tuned.data.store import utcday, utcnow

# Queue states. state_from -> state_to must differ (store.claim_tasks
# enforces it): a shared name would disable lease fencing.
JUDGE_STATE_FROM = "judging"
JUDGE_STATE_TO = "judging_active"
ACCEPTED_STATE = "accepted"
REJECTED_STATE = "rejected"
# Terminal-ish parking states: not claimable by any worker, visible in
# store.task_counts, and re-openable by hand once the cause is fixed.
SKIPPED_STATE = "judge_skipped"
ERROR_STATE = "judge_error"

JUDGE_PROMPT = "judge_pointwise_v1"
TIEBREAK_PROMPT = "judge_tiebreak_v1"
JUDGE_SLOTS = ("a", "b")
TIEBREAK_SLOT = "tiebreak"

# Enough for three integers and an 80-word rationale, with headroom for a
# judge that emits reasoning tokens on its way there.
JUDGE_MAX_TOKENS = 1024

# Provisional thresholds until P5 calibration writes judge_threshold rows.
PASS_MIN = 4
FAIL_MAX = 2
SCORE_RANGE = (1, 5)

# Used only when the harshest judge returned no rationale at all - the
# teacher still has to be told what to fix, and silence is not a note.
DEFAULT_REVIEWER_NOTE = "the reasoning did not carry the conclusion it announced"

PASS, BORDERLINE, FAIL = "pass", "borderline", "fail"

# Streams whose rows are teacher-generated and therefore judgeable. replay
# (empty-think) rows never enter the task table at all; this is a tripwire.
JUDGEABLE_STREAMS = frozenset({"synthesis", "curated_c2", "transition"})

# Total claims (generation + judge) a task may take before it is parked. A
# judge slot that keeps failing to parse or to route would otherwise re-claim
# the same task forever.
MAX_JUDGE_ATTEMPTS = 8

_AXIS_ALIASES = {
    "grounding": ("grounding", "grounding_faithfulness"),
    "validity": ("validity", "reasoning_validity"),
    "coverage": ("coverage", "issue_coverage"),
}


class JudgeParseError(ValueError):
    """The reply carried no scorable JSON object."""


@dataclass(frozen=True)
class JudgeScores:
    grounding: int
    validity: int
    coverage: int
    rationale: str = ""

    @property
    def min_axis(self) -> int:
        return min(self.grounding, self.validity, self.coverage)

    @property
    def verdict(self) -> str:
        if self.min_axis >= PASS_MIN:
            return PASS
        if self.min_axis <= FAIL_MAX:
            return FAIL
        return BORDERLINE

    def as_row(self) -> dict:
        return {
            "grounding": self.grounding,
            "validity": self.validity,
            "coverage": self.coverage,
            "rationale": self.rationale,
        }


@dataclass
class SlotOutcome:
    slot: str
    scores: JudgeScores | None = None
    ref: ModelRef | None = None
    family: str | None = None
    error: str | None = None
    attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class JudgeStats:
    claimed: int = 0
    decided: int = 0
    accepted: int = 0
    rejected: int = 0
    tiebreaks: int = 0
    regenerated: int = 0
    slot_errors: int = 0
    skipped: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    outcomes: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# --------------------------------------------------------------------------
# Parsing (contract 3).
# --------------------------------------------------------------------------

def _json_objects(text: str):
    """Every balanced {...} span in `text`, parsed, in order of appearance.

    A brace scanner rather than a regex because the rationale is free text
    that may itself contain braces or quotes, and rather than "strip the
    fence" because the fence is optional, sometimes mislabelled
    (```JSON, ```python) and sometimes absent while prose surrounds the
    object. String state is tracked so a brace inside the rationale cannot
    close the object early.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text or ""):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                    except ValueError:
                        parsed = None
                    if isinstance(parsed, dict):
                        yield parsed
    return


def _score(value) -> int:
    """One axis value -> an integer in 1..5, or raise.

    Accepts 4, "4", 4.0 and "4/5" because free-tier judges emit all four.
    Anything outside the range is a parse failure, not a clamp: a judge that
    returns 0 or 7 did not follow the rubric, and pretending it said 1 or 5
    would feed calibration a score nobody gave.
    """
    if isinstance(value, bool):
        raise JudgeParseError(f"axis value is a bool: {value!r}")
    if isinstance(value, str):
        value = value.strip().split("/")[0].strip()
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise JudgeParseError(f"axis value is not a number: {value!r}") from None
    score = int(round(number))
    if not (SCORE_RANGE[0] <= score <= SCORE_RANGE[1]):
        raise JudgeParseError(f"axis value out of range 1-5: {value!r}")
    return score


def parse_judge_reply(text: str) -> JudgeScores:
    """The three axes and the rationale out of whatever the judge said.

    The LAST complete object wins. A model that restates the contract's
    example object before answering (they do) would otherwise be scored on
    the example.
    """
    best: JudgeScores | None = None
    last_error: str | None = None
    for obj in _json_objects(text):
        lowered = {str(k).strip().lower(): v for k, v in obj.items()}
        values = {}
        try:
            for axis, aliases in _AXIS_ALIASES.items():
                found = next((lowered[a] for a in aliases if a in lowered), None)
                if found is None:
                    raise JudgeParseError(f"missing axis {axis!r}")
                values[axis] = _score(found)
        except JudgeParseError as exc:
            last_error = str(exc)
            continue
        rationale = lowered.get("rationale") or lowered.get("reason") or ""
        best = JudgeScores(**values, rationale=str(rationale)[:2000])
    if best is None:
        raise JudgeParseError(
            f"no scorable JSON object in judge reply ({last_error or 'no object found'}): "
            f"{' '.join((text or '').split())[:200]!r}"
        )
    return best


# --------------------------------------------------------------------------
# Routing (contracts 2 and 5).
# --------------------------------------------------------------------------

def undersized_families(cfg, role: str, needed_tokens: int) -> frozenset[str]:
    """Families that cannot hold `needed_tokens` in ANY of their `role` models.

    Only a family whose every model in the role is too small is excluded, so
    a mixed family keeps its large model and Router.pick's preference order
    still decides between them.
    """
    fits_by_family: dict[str, bool] = {}
    for ref in cfg.routing_refs(role):
        _, model = cfg.model_for(ref)
        cap = model.limits.get("max_context")
        fits = cap is None or int(cap) >= needed_tokens
        fits_by_family[model.family] = fits_by_family.get(model.family, False) or fits
    return frozenset(family for family, fits in fits_by_family.items() if not fits)


def generation_family(cfg, gen: Mapping) -> str | None:
    """The generator's family - from the row, falling back to the config."""
    family = gen.get("model_family")
    if family:
        return str(family)
    try:
        _, model = cfg.model_for(ModelRef(gen.get("provider"), gen.get("model")))
    except KeyError:
        return None
    return model.family


# --------------------------------------------------------------------------
# The decision matrix.
# --------------------------------------------------------------------------

def decide(verdicts: Sequence[str], *, already_regenerated: bool) -> str:
    """What to do with a set of judge verdicts. Pure; the whole matrix.

    Two judges:
      pass  + pass              -> accept
      fail  + fail              -> reject
      exactly one pass          -> tiebreak (a third family scores it blind)
      no pass, some borderline  -> ONE regeneration, then reject

    Three (a tiebreak was run): the tiebreak decides, because it is the only
    judge that saw the work without a disagreement to split. Its own
    borderline goes down the same one-regeneration path.

    The one-regeneration cap is what stops a genuinely middling seed from
    cycling forever: a 3 means the work is repairable, and one repair attempt
    is the budget. A second 3 is a reject.
    """
    if not verdicts:
        raise ValueError("decide() needs at least one verdict")
    if len(verdicts) >= 3:
        final = verdicts[2]
        if final == PASS:
            return "accept"
        if final == FAIL:
            return "reject"
        return "reject" if already_regenerated else "regenerate"

    passes = sum(1 for v in verdicts if v == PASS)
    if passes == len(verdicts):
        return "accept"
    if all(v == FAIL for v in verdicts):
        return "reject"
    if passes:
        return "tiebreak"
    return "reject" if already_regenerated else "regenerate"


def thresholds_active(store) -> int:
    """How many calibrated judge thresholds exist (P5 writes them).

    Zero means every decision below is PROVISIONAL - taken on the shipped
    both-axes>=4 rule rather than on anything fitted to human labels - and
    every decision event says so, so the pilot's accept rate is never read
    later as if it had been calibrated.
    """
    return int(
        store.conn.execute(
            "SELECT COUNT(*) FROM judge_threshold WHERE active = 1"
        ).fetchone()[0]
    )


def has_regenerated(store, task_id: str) -> bool:
    """Has this task already had its one rationale-fed regeneration?

    Read from the generation rows rather than from memory so it survives a
    crash: a worker that dies between the regeneration and the re-judge must
    not hand the task a second one when it comes back.
    """
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


def failing_rationale(outcomes: Sequence[SlotOutcome]) -> str:
    """The rationale of the harshest judge - the note the teacher gets back.

    Harshest, not first: the rubric asks every judge to name the decisive
    reason for its LOWEST score, so the lowest-scoring judge's sentence is
    the one that says what is actually wrong.
    """
    scored = [o for o in outcomes if o.scores is not None]
    if not scored:
        return ""
    worst = min(scored, key=lambda o: o.scores.min_axis)
    return worst.scores.rationale or ""


# --------------------------------------------------------------------------
# One judge call.
# --------------------------------------------------------------------------

def raw_judge_path(paths, day: str | None = None) -> str:
    return str(paths.raw_judge_dir(utcday(day)) / "judge.ndjson")


def _judgement_envelope(kind, task, gen, slot, ref, scores, reply_text, prompt_id) -> dict:
    """One judge reply, durable before the DB row.

    (task_id, attempt) is the natural key store.reconcile_raw resolves a
    gen_id from - it prefers it over the envelope's own gen_id precisely
    because gen_id is a surrogate that a rebuild reassigns. `attempt` is
    therefore the GENERATION's attempt, never a judge-side counter.
    """
    return {
        "kind": kind,
        "task_id": task["task_id"],
        "attempt": int(gen["attempt"]),
        "gen_id": int(gen["gen_id"]),
        "judge_slot": slot,
        "provider": ref.provider,
        "model": ref.model,
        "grounding": scores.grounding if scores else None,
        "validity": scores.validity if scores else None,
        "coverage": scores.coverage if scores else None,
        "rationale": scores.rationale if scores else None,
        "created_at": utcnow(),
        # Not columns: the paid artifact itself, and what produced it.
        "prompt_id": prompt_id,
        "prompt_sha": prompt_registry.load(prompt_id).sha,
        "reply_text": reply_text,
        "stream": task["stream"],
    }


async def judge_slot(
    store,
    cfg,
    router,
    *,
    task: Mapping,
    gen: Mapping,
    source: str,
    slot: str,
    role: str = "judge",
    prompt_id: str = JUDGE_PROMPT,
    exclude_families: frozenset[str] = frozenset(),
    paths,
    day: str | None = None,
    max_tokens: int = JUDGE_MAX_TOKENS,
) -> SlotOutcome:
    """Score one candidate with one judge; retry ONCE on an unparsable reply.

    The retry is a fresh Router call, so it is a different attempt (and may
    land on a different provider). The family eligibility is deliberately
    NOT narrowed for the retry: with three judge families, minus the
    generator's and minus the other slot's, narrowing again would leave
    nothing eligible and turn a garbled sentence into a routing failure.
    """
    outcome = SlotOutcome(slot=slot)
    messages = prompt_registry.render(
        prompt_id,
        source=source,
        candidate_think=gen.get("think") or "",
        candidate_answer=gen.get("answer") or "",
    )
    prompt_est = sum(len(m.get("content") or "") for m in messages) // 4
    needed = prompt_est + max_tokens
    exclude = frozenset(exclude_families) | undersized_families(cfg, role, needed)

    for attempt in (1, 2):
        outcome.attempts = attempt
        try:
            ref, response = await router.complete(
                role,
                messages,
                max_tokens=max_tokens,
                est_tokens=needed,
                exclude_families=exclude,
            )
        except ProviderError as exc:
            if exc.status == 429 and exc.provider and exc.model:
                record_usage(store, ModelRef(exc.provider, exc.model), is_429=True, day=day)
            outcome.error = str(exc)
            store.log_event(
                "judge_route_error",
                {
                    "task_id": task["task_id"],
                    "slot": slot,
                    "role": role,
                    "needed_tokens": needed,
                    "excluded": sorted(exclude),
                    "error": str(exc)[:500],
                },
            )
            return outcome

        record_usage(store, ref, response, day=day)
        outcome.ref = ref
        outcome.prompt_tokens += int(response.prompt_tokens or 0)
        outcome.completion_tokens += int(response.completion_tokens or 0)
        _, model_cfg = cfg.model_for(ref)
        outcome.family = model_cfg.family

        try:
            scores = parse_judge_reply(response.text or "")
        except JudgeParseError as exc:
            # Durable, but NOT as a judgement: reconcile_raw must not rebuild
            # a scoreless row that later reads as a score.
            append_ndjson(
                raw_judge_path(paths, day),
                _judgement_envelope(
                    "judge_error", task, gen, slot, ref, None, response.text, prompt_id
                ),
            )
            outcome.error = str(exc)
            store.log_event(
                "judge_parse_error",
                {
                    "task_id": task["task_id"],
                    "slot": slot,
                    "attempt": attempt,
                    "ref": f"{ref.provider}/{ref.model}",
                    "error": str(exc)[:500],
                },
            )
            continue

        envelope = _judgement_envelope(
            "judgement", task, gen, slot, ref, scores, response.text, prompt_id
        )
        # ---- raw FIRST ----
        raw_path = raw_judge_path(paths, day)
        raw_offset = append_ndjson(raw_path, envelope)
        row = scores.as_row()
        row.update(
            {
                "provider": ref.provider,
                "model": ref.model,
                "raw_path": raw_path,
                "raw_offset": raw_offset,
                "created_at": envelope["created_at"],
            }
        )
        store.record_judgement(int(gen["gen_id"]), slot, row)
        outcome.scores = scores
        outcome.error = None
        return outcome

    return outcome


# --------------------------------------------------------------------------
# One task.
# --------------------------------------------------------------------------

def _park(store, task, state, reason, *, worker_id) -> str:
    store.log_event("judge_parked", {"task_id": task["task_id"], "state": state, "reason": reason})
    store.set_task_state(task["task_id"], state, reason, expect_worker=worker_id)
    return state


async def judge_task(
    store,
    cfg,
    router,
    task: Mapping,
    *,
    paths,
    worker_id: str,
    day: str | None = None,
    stats: JudgeStats | None = None,
    _regenerating: bool = False,
) -> str:
    """Judge one generation and move the task to its outcome state.

    Returns the state the task ended in ("accepted", "rejected", the queue
    state it was handed back to, or a parking state).
    """
    stats = stats if stats is not None else JudgeStats()
    task_id = task["task_id"]

    if task["stream"] not in JUDGEABLE_STREAMS:
        stats.skipped += 1
        return _park(store, task, SKIPPED_STATE, "stream-not-judgeable", worker_id=worker_id)

    gen = store.latest_generation(task_id)
    if gen is None:
        stats.skipped += 1
        return _park(store, task, SKIPPED_STATE, "no-generation", worker_id=worker_id)
    if not (gen.get("think") or "").strip():
        # Contract 4. An empty trace has nothing to grade and the rubric's
        # first axis is about the trace; scoring it would manufacture a number.
        stats.skipped += 1
        return _park(store, task, SKIPPED_STATE, "empty-think", worker_id=worker_id)

    seed = store.get_seed(task["seed_id"])
    if seed is None:
        stats.skipped += 1
        return _park(store, task, SKIPPED_STATE, "missing-seed", worker_id=worker_id)
    try:
        # The judge sees EXACTLY the materials the generator saw - same
        # builder, same concatenation (contract 1).
        source = build_prompt(cfg, task, seed).grounding
    except (SlotError, KeyError) as exc:
        stats.skipped += 1
        return _park(store, task, SKIPPED_STATE, f"slots:{exc}"[:200], worker_id=worker_id)

    gen_family = generation_family(cfg, gen)
    base_exclude = frozenset({gen_family} if gen_family else ())

    outcomes: list[SlotOutcome] = []
    for slot in JUDGE_SLOTS:
        exclude = base_exclude | frozenset(o.family for o in outcomes if o.family)
        outcome = await judge_slot(
            store, cfg, router,
            task=task, gen=gen, source=source, slot=slot,
            exclude_families=exclude, paths=paths, day=day,
        )
        stats.prompt_tokens += outcome.prompt_tokens
        stats.completion_tokens += outcome.completion_tokens
        outcomes.append(outcome)
        if outcome.scores is None:
            stats.slot_errors += 1
            # Hand the task back to the queue: the judges are transiently
            # unusable, the generation is fine, and a later pass can score it.
            if int(task.get("attempts") or 0) >= MAX_JUDGE_ATTEMPTS:
                return _park(
                    store, task, ERROR_STATE, f"judge-slot-{slot}:{outcome.error}"[:200],
                    worker_id=worker_id,
                )
            store.set_task_state(task_id, JUDGE_STATE_FROM, expect_worker=worker_id)
            return JUDGE_STATE_FROM

    verdicts = [o.scores.verdict for o in outcomes]
    already = _regenerating or has_regenerated(store, task_id)
    action = decide(verdicts, already_regenerated=already)

    if action == "tiebreak":
        stats.tiebreaks += 1
        exclude = base_exclude | frozenset(o.family for o in outcomes if o.family)
        tiebreak = await judge_slot(
            store, cfg, router,
            task=task, gen=gen, source=source, slot=TIEBREAK_SLOT,
            role="tiebreak", prompt_id=TIEBREAK_PROMPT,
            exclude_families=exclude, paths=paths, day=day,
        )
        stats.prompt_tokens += tiebreak.prompt_tokens
        stats.completion_tokens += tiebreak.completion_tokens
        outcomes.append(tiebreak)
        if tiebreak.scores is None:
            stats.slot_errors += 1
            if int(task.get("attempts") or 0) >= MAX_JUDGE_ATTEMPTS:
                return _park(
                    store, task, ERROR_STATE, f"tiebreak:{tiebreak.error}"[:200],
                    worker_id=worker_id,
                )
            store.set_task_state(task_id, JUDGE_STATE_FROM, expect_worker=worker_id)
            return JUDGE_STATE_FROM
        verdicts.append(tiebreak.scores.verdict)
        action = decide(verdicts, already_regenerated=already)

    provisional = thresholds_active(store) == 0
    store.log_event(
        "judge_decision",
        {
            "task_id": task_id,
            "gen_id": int(gen["gen_id"]),
            "attempt": int(gen["attempt"]),
            "action": action,
            "provisional": provisional,
            "rule": f"min-axis>={PASS_MIN} pass, <={FAIL_MAX} fail",
            "generator_family": gen_family,
            "verdicts": verdicts,
            "scores": [
                {
                    "slot": o.slot,
                    "ref": f"{o.ref.provider}/{o.ref.model}" if o.ref else None,
                    "family": o.family,
                    **(o.scores.as_row() if o.scores else {}),
                }
                for o in outcomes
            ],
            "already_regenerated": already,
        },
    )
    stats.decided += 1
    stats.outcomes[action] = stats.outcomes.get(action, 0) + 1

    if action == "accept":
        stats.accepted += 1
        store.set_task_state(task_id, ACCEPTED_STATE, "judge:accept", expect_worker=worker_id)
        return ACCEPTED_STATE
    if action == "reject":
        stats.rejected += 1
        store.set_task_state(task_id, REJECTED_STATE, "judge:reject", expect_worker=worker_id)
        return REJECTED_STATE

    # ---- the one rationale-fed regeneration ----
    stats.regenerated += 1
    note = failing_rationale(outcomes)
    store.log_event(
        "judge_regeneration",
        {"task_id": task_id, "gen_id": int(gen["gen_id"]), "note": note[:500]},
    )
    result = await generate_once(
        store, cfg, router, task,
        paths=paths, reviewer_note=note or DEFAULT_REVIEWER_NOTE, day=day,
    )
    stats.prompt_tokens += result.prompt_tokens
    stats.completion_tokens += result.completion_tokens
    if not result.ok or result.disposition is not None:
        # The regeneration failed or was gated out; the gate disposition is
        # the answer (rejected, or back to pending for another attempt).
        return apply_gate_disposition(store, task, result, worker_id=worker_id)
    # Clean regeneration: judge it once more, with the cap now spent.
    return await judge_task(
        store, cfg, router, task,
        paths=paths, worker_id=worker_id, day=day, stats=stats, _regenerating=True,
    )


# --------------------------------------------------------------------------
# The loop.
# --------------------------------------------------------------------------

def format_batch_line(batch_ix: int, stats: JudgeStats) -> str:
    outcomes = " ".join(f"{k}={v}" for k, v in sorted(stats.outcomes.items()))
    return (
        f"judge batch {batch_ix}: claimed={stats.claimed} decided={stats.decided} "
        f"accepted={stats.accepted} rejected={stats.rejected} tiebreaks={stats.tiebreaks} "
        f"regen={stats.regenerated} slot-err={stats.slot_errors} skipped={stats.skipped} "
        f"tokens={stats.tokens} [{outcomes}]"
    )


async def run_judges(
    store,
    cfg,
    router,
    *,
    streams: Sequence[str],
    n_workers: int,
    forever: bool = False,
    max_batches: int | None = None,
    paths=None,
    worker_id: str | None = None,
    day: str | None = None,
    idle_sleep_s: float = 5.0,
    sleeper=asyncio.sleep,
) -> dict:
    """Claim 'judging' tasks and decide them, bounded the same way as generate.py.

    Tasks are judged SEQUENTIALLY inside a batch even though generation runs
    concurrently: a judged task can trigger a tiebreak and a regeneration
    (three or four more calls), and running n of those in parallel makes the
    judge role's per-minute buckets the bottleneck for every worker at once.
    """
    if paths is None:
        from tuned.data.paths import build_paths

        paths = build_paths(cfg.build.workdir).ensure()
    worker_id = worker_id or worker_name("judge")
    totals = JudgeStats()
    batches = 0

    while max_batches is None or batches < max_batches:
        stats = JudgeStats()
        for stream in streams:
            claimed = store.claim_tasks(
                worker_id, n_workers, stream=stream,
                state_from=JUDGE_STATE_FROM, state_to=JUDGE_STATE_TO,
            )
            stats.claimed += len(claimed)
            for task in claimed:
                await judge_task(
                    store, cfg, router, task,
                    paths=paths, worker_id=worker_id, day=day, stats=stats,
                )
        batches += 1
        for name in (
            "claimed", "decided", "accepted", "rejected", "tiebreaks",
            "regenerated", "slot_errors", "skipped", "prompt_tokens", "completion_tokens",
        ):
            setattr(totals, name, getattr(totals, name) + getattr(stats, name))
        for key, count in stats.outcomes.items():
            totals.outcomes[key] = totals.outcomes.get(key, 0) + count
        print(format_batch_line(batches, stats))
        if stats.claimed == 0:
            if not forever:
                break
            await sleeper(idle_sleep_s)

    return {
        "batches": batches,
        "claimed": totals.claimed,
        "decided": totals.decided,
        "accepted": totals.accepted,
        "rejected": totals.rejected,
        "tiebreaks": totals.tiebreaks,
        "regenerated": totals.regenerated,
        "slot_errors": totals.slot_errors,
        "skipped": totals.skipped,
        "prompt_tokens": totals.prompt_tokens,
        "completion_tokens": totals.completion_tokens,
        "outcomes": dict(totals.outcomes),
        "worker_id": worker_id,
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from tuned.data.config import load_build_config
    from tuned.data.paths import build_paths
    from tuned.data.providers import load_dotenv_keys
    from tuned.data.store import Store

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data_law_v1.yaml")
    parser.add_argument("--stream", action="append", default=None, help="repeatable")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--forever", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_build_config(args.config)
    print(f"loaded {load_dotenv_keys()} key(s) from .env")
    paths = build_paths(cfg.build.workdir).ensure()
    store = Store.open(paths.state_db)
    router = make_router(store, cfg)
    streams = args.stream or ["synthesis"]

    if thresholds_active(store) == 0:
        print("NOTE: no active judge_threshold rows - decisions are PROVISIONAL (P5 calibrates)")

    async def drive():
        # Closed inside the loop that owns the httpx pools (see generate.py).
        try:
            return await run_judges(
                store, cfg, router,
                streams=streams,
                n_workers=args.n_workers,
                forever=args.forever,
                max_batches=args.max_batches,
                paths=paths,
            )
        finally:
            await router.aclose()

    totals = asyncio.run(drive())
    print(
        f"done: batches={totals['batches']} decided={totals['decided']} "
        f"accepted={totals['accepted']} rejected={totals['rejected']} "
        f"tiebreaks={totals['tiebreaks']} regen={totals['regenerated']} "
        f"slot-err={totals['slot_errors']} tokens="
        f"{totals['prompt_tokens'] + totals['completion_tokens']}"
    )
    print("task states: " + ", ".join(f"{k}={v}" for k, v in sorted(store.task_counts().items())))
    store.close()

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
