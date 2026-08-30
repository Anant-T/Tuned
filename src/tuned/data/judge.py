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
candidate's trace and answer. Since 2026-08-19 EVERY JUDGE IS 131k or more -
the 8k zai-glm-4.7 left archived on 2026-08-18, mistral-small (32k) lost the
judge seat to human calibration, and gemma, promoted into it, is 131k. The
TIEBREAK pool's smallest is mistral-large at a probed 52,812. A candidate that
does not fit must be routed past it,
because a silently truncated judge prompt produces a score for an answer
nobody read - which is worse than not judging it at all. Router.pick exposes
no context filter, so the length check is turned into a family exclusion
(providers.undersized_families), which degrades safely: a family is excluded
only when EVERY one of its models in that role is too small.

WHEN THE POOL RUNS OUT, IT RUNS OUT LOUDLY AND ONCE. Family separation and
context length together can empty a role for a given row. That combination
used to bite the SHIPPED pool - the tiebreak was gpt-oss plus gemma, gemma was
pinned at 8192, so separation removed the first and length removed the second
and a long candidate from the gpt-oss generator had no eligible tiebreak at
all. It does not bite it today: the 2026-08-19 probes put gemma at 131k and
the preflight now reports zero gaps at every row size this build can produce.
The three rules below stay, because the condition is one config edit or one
model retirement away and they are what keep it from becoming a paid loop:

  * a NON-RETRYABLE routing failure parks the task in 'judge_unroutable'
    immediately instead of re-queueing it - nothing about tomorrow's claim
    would be different;
  * a slot that already has a judgement for this generation is REUSED from
    the judgement table, never bought again, so a retry after a failed slot
    costs one call and not two;
  * an unroutable TIEBREAK falls back to deciding on the two judges that did
    answer (logged as tiebreak_unroutable_two_judge_decision) - and since
    the disagreement it was called in to settle still stands, the
    conservative outcome is a reject, never an accept.

The real fix for the third is a 32k+ model in a fourth family, in BOTH
routing.judge and routing.tiebreak; the fallback is what keeps the fleet
moving until there is one. That gap is now checked BEFORE anything is
claimed (providers.pool_gaps via generate.print_preflight): a judge slot the
pool cannot fill for the longest row the length band permits refuses the
start outright, because discovering it at runtime means discovering it one
paid judge A at a time. A parked row is recovered with
`python -m tuned.data.tasks --reopen judge_unroutable`, and the slot it
already bought is reused rather than re-paid.

EVERY WRITE IS FENCED, INCLUDING THE JUDGEMENTS. The task-state writes were
always fenced; the judgement rows were not, and they are an INSERT OR
REPLACE keyed on (gen_id, judge_slot). A worker that stalls between reading
the recorded slots and its own reply can therefore overwrite a slot the live
holder already decided on - the row ends `accepted` carrying scores the
accept was never made with, which is precisely the pair P5 calibration and
gold labelling read. record_judgement takes the same expect_worker fence,
and the lease is re-checked before every paid call and before the decision
is logged, so a lost lease costs nothing rather than costing a judge.

THE PARSER IS DEFENSIVE (contract 3). Judges are free-tier models that
occasionally wrap their JSON in prose, in a fence, or in an apology. The
parser accepts the axis aliases the rubric uses (grounding_faithfulness /
reasoning_validity / issue_coverage as well as the short names), finds JSON
anywhere in the reply, and treats an unparsable answer as ONE retried judge
slot - never as a crash, and never as a score.

DEFENSIVE IS NOT CREDULOUS: A THINK BLOCK IS NOT A VERDICT. Some judges
inline their reasoning as <think>...</think> before answering, and inside
that block they restate the schema, argue with it, and try scores on. Those
objects are the model THINKING ABOUT the rubric, not applying it, so
split_reply_think puts the scorable region strictly after the closed block
and nothing inside it can be scored. Reading one would be the failure this
codebase keeps finding in its own instruments - the machinery reporting
HEALTHY in exactly the case it exists to catch - and it would be worse here
than elsewhere, because the number goes into the judgement table that P5
calibration and gold labelling read as a verdict somebody gave.

A block that never CLOSES is a parse error, loudly, and that is the shape
the qwen judge actually failed in (2026-08-18: 7 replies, 7 truncations, all
1,024 completion tokens spent inside <think>). Guessing a verdict out of
half a thought would turn a truncated reply into a score.

EMPTY-THINK ROWS ARE NEVER JUDGED (contract 4). The empty-think slice is
copied, not generated, so it never reaches this queue; if one ever does -
a routing bug - it is parked with a diagnostic instead of being scored on a
trace that does not exist.

THE JUDGE'S {source} IS THE GATES' GROUNDING PLUS, ON TRANSITION ONLY, THE
DATED POSTURE ({scenario}). Which enactment governs is decided by those
dates; a judge without them is scoring the one axis it cannot see.
GateContext.source_text stays the narrower string, because that one is the
citation allow-list and widening it would authorise whatever a scenario
happens to name.

Raw-first durability is the same rule as generate.py: the judge's reply is
appended to raw/judge/<day>/ BEFORE record_judgement. An UNPARSABLE reply is
appended too, but under kind="judge_error" rather than "judgement", so that
store.reconcile_raw skips it: recovering it as a judgement would write a row
with three NULL scores that later reads as a judgement nobody made.

Run:  python -m tuned.data.judge --config data/configs/data_law_v1.yaml
      --stream synthesis --n-workers 4 [--forever] [--max-batches N]
"""

import asyncio
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from tuned.data import prompt_registry
from tuned.data.calibrate import Candidate, decides_pass
from tuned.data.config import CALIBRATION_RULES, ModelRef
from tuned.data.generate import (
    JUDGE_PROMPT_ID,
    ROW_SHAPED_SKIPS,
    TIEBREAK_PROMPT_ID,
    SlotError,
    apply_gate_disposition,
    build_prompt,
    generate_once,
    judge_messages,
    judge_needed_tokens,
    make_router,
    print_preflight,
    usage_recorder,
    worker_name,
)
from tuned.data.jsonl import append_ndjson
from tuned.data.judge_policy import (
    BORDERLINE,
    FAIL,
    FAIL_MAX,
    PASS,
    PASS_MIN,
    decide,
)
from tuned.data.providers import (
    DEFAULT_JUDGE_REPLY_TOKENS,
    ProviderError,
    undersized_families,
)
from tuned.data.store import utcday, utcnow
from tuned.data.paths import DEFAULT_CONFIG

# Queue states. state_from -> state_to must differ (store.claim_tasks
# enforces it): a shared name would disable lease fencing.
JUDGE_STATE_FROM = "judging"
JUDGE_STATE_TO = "judging_active"
ACCEPTED_STATE = "accepted"
REJECTED_STATE = "rejected"
# Not a task state: what judge_task RETURNS when this worker no longer holds
# the lease. Nothing was written, nothing was counted, and the live holder's
# row is untouched. Same vocabulary generate.apply_gate_disposition uses.
LOST_LEASE = "lost-lease"
# Terminal-ish parking states: not claimable by any worker, visible in
# store.task_counts, and re-opened with `tuned.data.tasks --reopen` (which
# is a real command now, not a hand-written UPDATE) once the cause is fixed.
SKIPPED_STATE = "judge_skipped"
ERROR_STATE = "judge_error"
# No model in the pool CAN take this row (every family excluded by
# separation + context length, or no key). Re-queueing it would re-pay the
# judges that did answer and arrive at the same wall, so it parks at once.
UNROUTABLE_STATE = "judge_unroutable"

# routing.judge_mode == "audit" (2026-08-29): with one groq key the dual
# fleet decides ~35-40 rows/UTC-day, so dual-judging every row cannot keep
# pace with generation. In audit mode only a hash-sampled fraction of rows
# buys the full dual treatment; the rest exit here, gate-clean but unjudged,
# and the sample's accept rate is the quality evidence for the whole batch.
AUDIT_ACCEPT_DISPOSITION = "audit:gate-accept"
# Same accept, different WARRANT, so the readout can tell them apart: this one
# is a row the hash SELECTED for judging that the judge fleet could not serve
# (every eligible ref over quota, attempts exhausted). It ships on its gates,
# exactly like an unsampled row - what it must not do is end in judge_error,
# which would make a row's survival depend on whether the hash picked it.
AUDIT_UNJUDGED_DISPOSITION = "audit:gate-accept:unjudged"
DEFAULT_AUDIT_SAMPLE = 0.05

# Defined in generate.py and re-exported here: the startup preflight sizes
# the judge's largest possible call with the SAME renderer this worker uses,
# and generate.py is the module both fleets' preflight lives in.
JUDGE_PROMPT = JUDGE_PROMPT_ID
TIEBREAK_PROMPT = TIEBREAK_PROMPT_ID
JUDGE_SLOTS = ("a", "b")
TIEBREAK_SLOT = "tiebreak"

# Enough for three integers and an 80-word rationale. providers.py keeps a
# copy (DEFAULT_JUDGE_REPLY_TOKENS) so the startup preflight can size the
# judge pool without importing this module; they must agree, and a test pins
# that they do.
#
# IT IS NOT "with headroom for a judge that emits reasoning tokens on its way
# there", which is what this comment used to claim and what 2026-08-18
# measured false. A reasoning judge does not overshoot this budget a little;
# it never reaches the verdict at all - groq/qwen/qwen3.6-27b spent EXACTLY
# 1,024 completion tokens on each of 7 calls and every reply was cut mid-word
# still inside <think>. The fix is per-model and in the config
# (role_params.judge.reasoning_effort = none), because the number here cannot
# be the answer:
#
#   * it is FLEET-WIDE. judge_needed_tokens adds it to every judge prompt and
#     undersized_families turns the sum into a family exclusion, so raising it
#     raises the pool's context bar. The worst-case judge call is 23,729
#     routing tokens and CONTEXT_SAFETY_MARGIN makes that 29,661 of required
#     window.
#
#     THAT ARITHMETIC WAS WRITTEN AGAINST A POOL THAT NO LONGER EXISTS, and it
#     is worth saying rather than deleting because the SHAPE of the argument is
#     still the reason this number is not the answer. It used to read "against
#     slot A's 32,000; +2,048 takes the requirement to 32,221 and retires
#     mistral from every long row". Slot A is qwen at 131k now (2026-08-19),
#     mistral-small is out of the judge pool entirely, and both cerebras
#     windows were probed at 131,072 - so the smallest judge window is 131k and
#     +2,048 retires nothing at all. The fleet-wide objection stands; the
#     specific casualty it named is gone;
#   * it is not per-model, and the model that needs a bigger reply is the one
#     whose tpm (6,000) is already under one call at this size (measured judge
#     prompts 4,914-5,661 routing tokens).
JUDGE_MAX_TOKENS = DEFAULT_JUDGE_REPLY_TOKENS

# Provisional bands live in judge_policy so the matched evaluator cannot drift.
SCORE_RANGE = (1, 5)

# Used only when the harshest judge returned no rationale at all - the
# teacher still has to be told what to fix, and silence is not a note.
DEFAULT_REVIEWER_NOTE = "the reasoning did not carry the conclusion it announced"

# Streams whose rows are teacher-generated and therefore judgeable. replay
# (empty-think) rows never enter the task table at all; this is a tripwire.
JUDGEABLE_STREAMS = frozenset({"synthesis", "curated_c2", "transition"})

# Total claims (generation + judge) a task may take before it is parked. A
# judge slot that keeps failing to parse or to route would otherwise re-claim
# the same task forever.
MAX_JUDGE_ATTEMPTS = 8

# "ground_faithfulness" is not a spelling we invented: a gpt-5-mini judge
# emitted it on exp_harmony 2026-08-21 and a complete, well-formed, PAID
# verdict was discarded for it. Aliases are read-side only - the rubric still
# asks for one spelling, and nothing here loosens what a score has to be.
_AXIS_ALIASES = {
    "grounding": ("grounding", "grounding_faithfulness", "ground_faithfulness"),
    "validity": ("validity", "reasoning_validity"),
    "coverage": ("coverage", "issue_coverage"),
}

# The delimiters a PROVIDER wraps inlined reasoning in. Deliberately NOT
# cfg.think_open/think_close, which are the TRAINER's tags for the dataset:
# they are the same two strings today, and binding them would make a future
# re-tag of the corpus silently change how judge replies parse. A judge reply
# is never trained on; these are wire format, not dataset format.
REPLY_THINK_OPEN = "<think>"
REPLY_THINK_CLOSE = "</think>"

# Matched case-INSENSITIVELY - see split_reply_think for why the judge parser
# and gates._tag_positions differ on this deliberately.
#
# Compiled patterns over the ORIGINAL string rather than a `body.lower()`
# search whose offsets are reused: str.lower() is not length-preserving for
# every codepoint (U+0130 lowercases to two characters), so on a reply
# carrying one the offsets would slide and the split would land mid-token.
# The tags are ASCII, the corpus is not.
_REPLY_THINK_OPEN_RE = re.compile(re.escape(REPLY_THINK_OPEN), re.IGNORECASE)
_REPLY_THINK_CLOSE_RE = re.compile(re.escape(REPLY_THINK_CLOSE), re.IGNORECASE)


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


@dataclass(frozen=True)
class SlotDecision:
    """One slot's verdict under the fleet's active rule, or the provisional bands."""

    verdict: str
    rule: str | None = None
    threshold: int | None = None
    fitted: bool = False
    unfitted: bool = False
    provisional: bool = False
    coerced: bool = False
    valid: bool = True
    reason: str | None = None
    model: str | None = None

    def as_event(self) -> dict:
        return {
            "model": self.model,
            "rule": self.rule,
            "threshold": self.threshold,
            "fitted": self.fitted,
            "unfitted": self.unfitted,
            "provisional": self.provisional,
            "coerced": self.coerced,
            "valid": self.valid,
            "reason": self.reason,
            "verdict": self.verdict,
        }


def active_rules_by_model(store) -> dict[str, dict]:
    """Exact model id -> its active threshold row.

    A model with two active rows is invalid rather than last-write-wins: the
    fleet must fail closed, not pick a rule by accident.
    """
    out: dict[str, dict] = {}
    for row in store.judge_thresholds(active_only=True):
        model = str(row.get("model") or "").strip()
        if not model:
            continue
        packed = dict(row)
        if model in out:
            packed["_invalid"] = True
            packed["reason"] = "ambiguous-active-rows"
        out[model] = packed
    return out


def _parse_threshold(row: Mapping) -> tuple[str, int] | str:
    """`(rule, threshold)` or an error reason. Unknown names fail closed."""
    if row.get("_invalid"):
        return str(row.get("reason") or "invalid-active-row")
    rule = row.get("rule")
    if rule not in CALIBRATION_RULES:
        return f"unknown-rule:{rule}"
    raw = row.get("threshold")
    if isinstance(raw, bool) or raw is None:
        return f"malformed-threshold:{raw!r}"
    try:
        threshold = int(raw)
    except (TypeError, ValueError):
        return f"malformed-threshold:{raw!r}"
    if isinstance(raw, float) and raw != threshold:
        return f"malformed-threshold:{raw!r}"
    if not (SCORE_RANGE[0] <= threshold <= SCORE_RANGE[1]):
        return f"malformed-threshold:{threshold!r}"
    return str(rule), threshold


def rule_passes(rule: str, threshold: int, scored: JudgeScores) -> bool:
    """The same predicate calibrate.decides_pass fitted. Do not restate it."""
    return decides_pass(
        Candidate(rule, threshold),
        (scored.grounding, scored.validity, scored.coverage),
    )


def resolve_slot(
    scored: JudgeScores,
    *,
    model: str | None,
    rules: Mapping[str, Mapping],
    fleet_active: bool,
) -> SlotDecision:
    """Apply the exact model's active rule, or the safe unfitted / provisional path.

    No active rows anywhere: the shipped min_axis>=4 / <=2 bands, marked
    provisional. Active rows exist but this model has none: a would-be
    provisional pass is coerced to borderline so an unfitted seat cannot mint
    an accept; fail evidence stays fail. A present but unknown or malformed
    rule is FAIL, never a silent fallback to the provisional pass band.
    """
    model_key = (model or "").strip() or None
    if not fleet_active:
        return SlotDecision(
            verdict=scored.verdict,
            rule="min_axis",
            threshold=PASS_MIN,
            provisional=True,
            model=model_key,
        )
    row = rules.get(model_key) if model_key else None
    if row is None:
        coerced = scored.verdict == PASS
        return SlotDecision(
            verdict=BORDERLINE if coerced else scored.verdict,
            unfitted=True,
            provisional=True,
            coerced=coerced,
            reason="unfitted-model",
            model=model_key,
        )
    parsed = _parse_threshold(row)
    if isinstance(parsed, str):
        return SlotDecision(
            verdict=FAIL,
            rule=row.get("rule") if isinstance(row, Mapping) else None,
            threshold=row.get("threshold") if isinstance(row, Mapping) else None,
            fitted=True,
            valid=False,
            reason=parsed,
            model=model_key,
        )
    rule, threshold = parsed
    if rule_passes(rule, threshold, scored):
        verdict = PASS
    elif scored.min_axis <= FAIL_MAX:
        verdict = FAIL
    else:
        verdict = BORDERLINE
    return SlotDecision(
        verdict=verdict,
        rule=rule,
        threshold=threshold,
        fitted=True,
        model=model_key,
    )


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
    # Read back from a judgement recorded on an earlier pass instead of
    # bought again (see judge_task).
    reused: bool = False
    # No model in the pool can serve this ROW, rather than a model that
    # failed: re-queueing cannot fix it.
    unroutable: bool = False
    # Why the Router had nothing to try, when it had nothing to try.
    route_skips: tuple[str, ...] = ()
    # A provider ANSWERED and refused the payload (a 400/413/422 with no
    # context marker). Distinct from `unroutable`: something was tried, and
    # the identical request will be refused identically on the next claim, so
    # re-queueing buys the same call again. Never earned from `not retryable`
    # - a missing key is not retryable either.
    payload_error: bool = False
    # The judgement write was fenced out: this worker no longer holds the
    # task's lease, so the scores it just paid for belong to nobody and the
    # whole task must be abandoned rather than decided on.
    lost_lease: bool = False


@dataclass
class JudgeStats:
    claimed: int = 0
    decided: int = 0
    accepted: int = 0
    rejected: int = 0
    tiebreaks: int = 0
    regenerated: int = 0
    slot_errors: int = 0
    # Rows parked because the pool had nothing for them. Counted apart from
    # `skipped`, which is the free, unpaid refusals (empty trace, wrong
    # stream): an unroutable row has usually PAID for a judge already, and
    # burying it in the same number hides the one that costs money.
    unroutable: int = 0
    skipped: int = 0
    # Rows parked in judge_error - a slot that kept failing, or a payload a
    # provider refuses. Its own column because `_park(counter=None)` put these
    # in none at all: the batch line reported the claim and then accounted for
    # nothing, which reads as a worker doing nothing rather than a row parking.
    judge_errors: int = 0
    lost_leases: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    outcomes: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# --------------------------------------------------------------------------
# Parsing (contract 3).
# --------------------------------------------------------------------------

def split_reply_think(text: str) -> tuple[str | None, str]:
    """(think, scorable) for a judge reply; raises on a block that never closes.

    Mirrors generate.assemble_content's second shape - the model that inlines
    <think>...</think> in the text rather than returning a reasoning channel -
    but answers a narrower question, because a judge reply is scored rather
    than stored: WHERE IS A VERDICT ALLOWED TO BE.

    Everything after the LAST close tag, and nothing before it. Not
    gates.split_think's prefix+suffix, and the difference is the whole point:
    that one keeps the prefix so no answer-side gate is handed a blank, while
    here a "prefix" is text the model wrote before it had finished thinking,
    and scoring an object out of it credits a draft as a decision.

    The LAST, not the first, and that is load-bearing rather than tidy: a reply
    that thinks, answers, thinks again and answers again has its verdict after
    the final close, and taking the first one would hand the whole second
    thought back as scorable text.

    Truncation is ONE test, not two, and it covers both ways a reply can end
    mid-thought: the scorable region may not contain an OPEN tag. A reply that
    opened and never closed leaves the whole text scorable-region and trips it;
    so does a reply that closed one block, answered, and opened another it
    never finished. Either way the reply stops inside a thought, and a verdict
    read out of it would be a verdict the model had not arrived at.

    CASE-INSENSITIVE, and gates._tag_positions is not. That asymmetry is a
    decision, not an oversight, and must not be "harmonized" away: the two
    fail in opposite directions on a cased tag. gates fails SAFE - an
    unrecognised <THINK> leaves the whole content flowing to the answer-side
    gates, which then judge more text than they should and reject rather than
    admit. This parser fails SILENT - an unrecognised close tag makes the
    model's own draft schema object scorable, and a 1/1/1 nobody voted for is
    written to the judgement table, reused by every later pass through
    judge_slot_reused, and read by P5 calibration as a verdict. Measured, not
    imagined: <THINK>...draft object...</THINK> with no verdict after it
    scored 1/1/1 before this.

    A reply with no tags at all is returned whole (think=None) - that is the
    mistral shape. That was most of the judgements this build had when the
    parser was written; mistral-small left the judge pool on 2026-08-19, so it
    is now a shape the parser must still handle rather than the common case.
    """
    body = text or ""
    closes = list(_REPLY_THINK_CLOSE_RE.finditer(body))
    cut = closes[-1].end() if closes else 0
    think = body[:cut] if closes else None
    scorable = body[cut:]
    if _REPLY_THINK_OPEN_RE.search(scorable):
        raise JudgeParseError(
            f"judge reply ends inside an unclosed {REPLY_THINK_OPEN} block - it was "
            f"truncated before any verdict: {' '.join(body.split())[:200]!r}"
        )
    return think, scorable


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
    would feed calibration a score nobody gave. A FRACTION is the same defect
    in the middle of the range and is refused for the same reason: the rubric
    asks for "an integer from 1 to 5" (judge_pointwise_v1), so 3.5 is a judge
    declining to commit - and rounding it to 4 crosses PASS_MIN and mints a
    pass on the very axis that decides the verdict (JudgeScores.verdict is
    min_axis >= PASS_MIN). One retry per slot is already budgeted, so the cost
    of refusing is at most one re-ask of a judge that is off-rubric anyway.
    """
    if isinstance(value, bool):
        raise JudgeParseError(f"axis value is a bool: {value!r}")
    if isinstance(value, str):
        value = value.strip().split("/")[0].strip()
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise JudgeParseError(f"axis value is not a number: {value!r}") from None
    if not math.isfinite(number):
        # inf/nan reach int() as OverflowError/ValueError, which would escape
        # the slot's JudgeParseError handling as a crash rather than a re-ask.
        raise JudgeParseError(f"axis value is not finite: {value!r}")
    if number != int(number):
        raise JudgeParseError(f"axis value is not an integer: {value!r}")
    score = int(number)
    if not (SCORE_RANGE[0] <= score <= SCORE_RANGE[1]):
        raise JudgeParseError(f"axis value out of range 1-5: {value!r}")
    return score


def parse_judge_reply(text: str) -> JudgeScores:
    """The three axes and the rationale out of whatever the judge said.

    The LAST complete object wins, out of the region a verdict is allowed to
    be in (split_reply_think). A model that restates the contract's example
    object before answering (they do) would otherwise be scored on the
    example - and a model that reasons about the schema inside a think block
    would be scored on the reasoning.
    """
    think, scorable = split_reply_think(text)
    best: JudgeScores | None = None
    last_error: str | None = None
    for obj in _json_objects(scorable):
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
        # Which REGION was searched is named when it was narrowed. A reply that
        # visibly contains a well-formed object and is reported as having none
        # reads as a parser bug, and the operator goes looking for one instead
        # of at the model that put its verdict inside its own reasoning.
        #
        # Keyed on the OPEN tag, not on `think is not None`: a stray "</think>"
        # inside a rationale string makes `think` a non-empty prefix with no
        # reasoning block in it at all, and blaming a block that was never
        # opened sends the operator after the wrong model behaviour.
        where = (
            " after the reasoning block"
            if _REPLY_THINK_OPEN_RE.search(think or "")
            else ""
        )
        raise JudgeParseError(
            f"no scorable JSON object in judge reply{where} "
            f"({last_error or 'no object found'}): "
            f"{' '.join((text or '').split())[:200]!r}"
        )
    return best


# --------------------------------------------------------------------------
# Routing (contracts 2 and 5).
# --------------------------------------------------------------------------

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
    worker_id: str | None = None,
) -> SlotOutcome:
    """Score one candidate with one judge; retry ONCE on an unparsable reply.

    The retry is a fresh Router call, so it is a different attempt (and may
    land on a different provider). The family eligibility is deliberately
    NOT narrowed for the retry: with three judge families, minus the
    generator's and minus the other slot's, narrowing again would leave
    nothing eligible and turn a garbled sentence into a routing failure.
    """
    outcome = SlotOutcome(slot=slot)
    # The renderer and the sizer the STARTUP PREFLIGHT runs, on this row
    # rather than on a constructed worst case. Two implementations of "how big
    # is this judge call" is how the preflight came to clear a pool that then
    # parked rows half-paid, so there is only one of each.
    messages = judge_messages(
        source, gen.get("think") or "", gen.get("answer") or "", prompt_id=prompt_id
    )
    needed = judge_needed_tokens(messages, reply_tokens=max_tokens)
    exclude = frozenset(exclude_families) | undersized_families(cfg, role, needed)

    for attempt in (1, 2):
        # The retry is a SECOND paid call, so it gets the same check every
        # other purchase gets. The write below is fenced either way; what
        # this saves is the call, which the fence cannot refund.
        if attempt > 1 and not holds_lease(store, task["task_id"], worker_id):
            outcome.lost_lease = True
            outcome.error = "lost-lease"
            store.log_event(
                "lost_lease",
                {
                    "task_id": task["task_id"],
                    "worker": worker_id,
                    "wanted_state": f"judge-retry:{slot}",
                },
            )
            return outcome
        outcome.attempts = attempt
        try:
            ref, response = await router.complete(
                role,
                messages,
                max_tokens=max_tokens,
                est_tokens=needed,
                exclude_families=exclude,
                on_attempt=usage_recorder(store, day),
            )
        except ProviderError as exc:
            outcome.error = str(exc)
            skips = frozenset(getattr(exc, "skipped", frozenset()))
            outcome.route_skips = tuple(sorted(skips))
            # UNROUTABLE means "nothing in this pool can take this ROW":
            # every family that could hold the prompt was excluded by
            # separation plus context length, or every model tried said the
            # prompt is longer than its window. Those are facts about the row,
            # so the caller parks rather than re-queueing - a re-queue would
            # re-pay whichever slots DID answer and hit the same wall.
            #
            # `not exc.retryable` is NOT the same test, and using it was the
            # generator's round-2 bug in judge clothing: a missing key is not
            # retryable either, and a keyless fleet would park every row it
            # touched instead of leaving them where the queue found them.
            outcome.unroutable = bool(exc.context_exceeded) or (
                bool(skips) and skips <= ROW_SHAPED_SKIPS
            )
            # A provider answered and refused the PAYLOAD. The test is a
            # status plus the absence of every per-provider explanation -
            # never `not retryable`, which a missing key also satisfies. It
            # matters because such a call is bought again on every claim: the
            # generator's equivalent is bounded at 3 attempts, and the judge's
            # at 8, so a systemic payload bug costs eight passes over the wave.
            #
            # Two of the five clauses are DEFENCE IN DEPTH and unpinnable by
            # construction, which is stated here rather than left for the next
            # reader to rediscover: through Router.complete a provider_dead
            # error is never re-raised (it fails over, and the aggregate does
            # not carry the flag), and a `skipped` set only ever arrives on
            # the "nothing was tried" error, which has no status. Neither can
            # be reached with the first clause true, so no test can fail on
            # their removal - they are kept because each names a distinct fact
            # a future error shape could carry, and they are NOT counted in
            # any mutation-verified claim about this file. `context_exceeded`
            # IS reachable (overflow at every ref aggregates that way) and is
            # pinned on the logged event below.
            outcome.payload_error = (
                exc.status is not None
                and not exc.retryable
                and not exc.provider_dead
                and not exc.context_exceeded
                and not skips
            )
            store.log_event(
                "judge_route_error",
                {
                    "task_id": task["task_id"],
                    "slot": slot,
                    "role": role,
                    "needed_tokens": needed,
                    "excluded": sorted(exclude),
                    "unroutable": outcome.unroutable,
                    "payload_error": outcome.payload_error,
                    "skipped": list(outcome.route_skips),
                    "status": exc.status,
                    "context_exceeded": bool(exc.context_exceeded),
                    "error": str(exc)[:500],
                },
            )
            return outcome

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
        # FENCED like every task-state write. The reply is already durable in
        # the raw log either way; what the fence protects is the DB row, which
        # the live holder of this task may already have decided on.
        written = store.record_judgement(
            int(gen["gen_id"]), slot, row, expect_worker=worker_id
        )
        if not written:
            outcome.lost_lease = True
            outcome.error = "lost-lease"
            store.log_event(
                "lost_lease",
                {
                    "task_id": task["task_id"],
                    "worker": worker_id,
                    "wanted_state": f"judgement:{slot}",
                    "gen_id": int(gen["gen_id"]),
                },
            )
            return outcome
        outcome.scores = scores
        outcome.error = None
        return outcome

    return outcome


# --------------------------------------------------------------------------
# One task.
# --------------------------------------------------------------------------

def _set_state(store, task_id: str, state: str, disposition: str | None, *, worker_id) -> bool:
    """Lease-fenced state write; a lost lease is logged, never silent.

    Same fence generate.py uses: a worker that stalled past its lease has had
    its task legitimately re-claimed, so its late write must no-op rather
    than overwrite the live holder - and the operator has to be able to SEE
    that happening, which an unchecked return value hides.
    """
    moved = store.set_task_state(task_id, state, disposition, expect_worker=worker_id)
    if not moved:
        store.log_event(
            "lost_lease",
            {"task_id": task_id, "worker": worker_id, "wanted_state": state},
        )
    return moved


def _audit_accept(
    store,
    task_id,
    gen,
    *,
    worker_id,
    stats: "JudgeStats",
    sample: float,
    disposition: str,
    outcome: str,
    reason: str | None = None,
) -> str:
    """Accept a gate-clean row without a judge verdict, and say which kind.

    Two callers, one write, because they must agree: the sampler (this row was
    not selected) and the attempt cap (this row WAS selected but no judge could
    be reached). Before, only the first existed, so a sampled row whose every
    judge ref was over quota burned MAX_JUDGE_ATTEMPTS claims and parked in
    judge_error - lost from the corpus entirely, when the identical row would
    have shipped had the hash not picked it. That also biased the sample: the
    surviving verdicts were whichever rows the fleet had quota for, i.e. the
    first ~30 rows of a UTC day, not the uniform hash sample audit_sampled's
    docstring promises.
    """
    if not _set_state(store, task_id, ACCEPTED_STATE, disposition, worker_id=worker_id):
        stats.lost_leases += 1
        return LOST_LEASE
    payload = {"task_id": task_id, "gen_id": int(gen["gen_id"]), "sample": sample}
    if reason:
        payload["reason"] = reason
    store.log_event("audit_gate_accept", payload)
    stats.decided += 1
    stats.accepted += 1
    stats.outcomes[outcome] = stats.outcomes.get(outcome, 0) + 1
    return ACCEPTED_STATE


def _park(
    store,
    task,
    state,
    reason,
    *,
    worker_id,
    stats: "JudgeStats | None" = None,
    counter: str | None = "skipped",
) -> str:
    """Park the task, or report the lost lease that stopped us.

    Returns the state the task actually ended in - never the one we wanted -
    and increments `counter` only when the fenced write LANDED. Counting the
    intention instead is how batch totals come to over-report every outcome
    the fence rejected.

    The EVENT follows the write for the same reason the counter does: a stale
    worker that logged `judge_parked` for a row it did not park put a park in
    the event log that no state change ever matched, and this log is what P5
    calibration reads. Same fix as the second `judge_decision`.
    """
    if not _set_state(store, task["task_id"], state, reason, worker_id=worker_id):
        if stats is not None:
            stats.lost_leases += 1
        return LOST_LEASE
    store.log_event("judge_parked", {"task_id": task["task_id"], "state": state, "reason": reason})
    if stats is not None and counter is not None:
        setattr(stats, counter, getattr(stats, counter) + 1)
    return state


def holds_lease(store, task_id: str, worker_id: str | None) -> bool:
    """Does `worker_id` still hold this task? Cheap, and checked before money.

    The fenced writes make a stale worker's RESULT harmless, but they cannot
    make its CALLS free: by the time the fence rejects the write, the judge
    has already been paid for. One SELECT before each purchase turns a lost
    lease from a wasted call into no call at all.

    worker_id=None means "unfenced", the same convention set_task_state uses.
    """
    if worker_id is None:
        return True
    task = store.get_task(task_id)
    return task is not None and task.get("claimed_by") == worker_id


def _lost_lease(store, task_id: str, worker_id, stats: "JudgeStats", where: str) -> str:
    stats.lost_leases += 1
    store.log_event(
        "lost_lease", {"task_id": task_id, "worker": worker_id, "wanted_state": where}
    )
    return LOST_LEASE


def audit_sampled(task_id: str, fraction: float) -> bool:
    """Is this row in the audit sample? Deterministic and monotone in fraction.

    Hashing the task_id (itself a content hash) rather than rolling a die
    means a row's fate never flips between claims - a re-claimed sampled row
    is still sampled, so a half-judged row cannot exit as an audit-accept -
    and raising the fraction later only ADDS rows to the sample, never
    silently un-judges one. Uniform over task_ids, so the sample is
    proportionally stratified across task types and streams for free.
    """
    return int(task_id[:8], 16) / 0xFFFFFFFF < fraction


def _outcome_from_row(cfg, slot: str, row: Mapping) -> SlotOutcome | None:
    """Rebuild a slot's outcome from a judgement already in the DB.

    None when the row cannot stand in for a judge call, in which case the
    slot is simply bought again. Two ways to fail:

    * a score is missing - there is nothing to reuse;
    * the recorded ref cannot be resolved to a FAMILY, either because the
      model has left the config or because the row carries no ref at all.
      That one matters more than it looks: a reused slot A with family=None
      stops constraining slot B, and "two judges" quietly becomes one family
      grading itself twice - which is the invariant the whole separation
      exists for. One re-bought call is cheaper than that.
    """
    values = [row.get(axis) for axis in ("grounding", "validity", "coverage")]
    if any(value is None for value in values):
        return None
    ref = ModelRef(str(row.get("provider") or ""), str(row.get("model") or ""))
    try:
        family = cfg.model_for(ref)[1].family
    except KeyError:
        # BOTH ref failures land here, and they are the same failure. A
        # separate `not (provider and model)` guard in front of this was
        # unreachable-in-effect - an absent ref makes an empty ModelRef, which
        # no config resolves either - so no mutation of it could ever fail a
        # test. An unreachable branch reads as covered while nothing covers it.
        return None
    return SlotOutcome(
        slot=slot,
        scores=JudgeScores(
            grounding=int(values[0]),
            validity=int(values[1]),
            coverage=int(values[2]),
            rationale=str(row.get("rationale") or ""),
        ),
        ref=ref,
        family=family,
        reused=True,
    )


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
    audit_sample: float = DEFAULT_AUDIT_SAMPLE,
    _regenerating: bool = False,
) -> str:
    """Judge one generation and move the task to its outcome state.

    Returns the state the task ended in ("accepted", "rejected", the queue
    state it was handed back to, or a parking state).
    """
    stats = stats if stats is not None else JudgeStats()
    task_id = task["task_id"]

    # Before anything: do we still own this task? A stalled worker's whole
    # pass is void, and finding that out at the first fenced write means
    # having paid for a judge on somebody else's row.
    if not holds_lease(store, task_id, worker_id):
        return _lost_lease(store, task_id, worker_id, stats, "judge")

    if task["stream"] not in JUDGEABLE_STREAMS:
        return _park(
            store, task, SKIPPED_STATE, "stream-not-judgeable",
            worker_id=worker_id, stats=stats,
        )

    gen = store.latest_generation(task_id)
    if gen is None:
        return _park(
            store, task, SKIPPED_STATE, "no-generation", worker_id=worker_id, stats=stats
        )
    if not (gen.get("think") or "").strip():
        # Contract 4. An empty trace has nothing to grade and the rubric's
        # first axis is about the trace; scoring it would manufacture a number.
        return _park(
            store, task, SKIPPED_STATE, "empty-think", worker_id=worker_id, stats=stats
        )

    seed = store.get_seed(task["seed_id"])
    if seed is None:
        return _park(
            store, task, SKIPPED_STATE, "missing-seed", worker_id=worker_id, stats=stats
        )

    # Audit mode: rows outside the sample exit here - gate-clean, unjudged,
    # zero calls. Placed AFTER the free guards above (an empty trace or a
    # missing generation must park, never audit-accept) and BEFORE any
    # rendering or routing. The counters follow the WRITE, same as every
    # decision below.
    if cfg.routing.judge_mode == "audit" and not audit_sampled(task_id, audit_sample):
        return _audit_accept(
            store, task_id, gen, worker_id=worker_id, stats=stats,
            sample=audit_sample, disposition=AUDIT_ACCEPT_DISPOSITION,
            outcome="audit-accept",
        )

    try:
        # The judge sees the materials the generator saw - same builder, same
        # concatenation (contract 1) - plus, on the transition stream only,
        # the dated posture, without which the enactment question is not
        # decidable at all. GateContext.source_text stays the narrower
        # `grounding`: it is the citation allow-list.
        source = build_prompt(cfg, task, seed).judge_source
    except (SlotError, KeyError) as exc:
        return _park(
            store, task, SKIPPED_STATE, f"slots:{exc}"[:200],
            worker_id=worker_id, stats=stats,
        )

    gen_family = generation_family(cfg, gen)
    base_exclude = frozenset({gen_family} if gen_family else ())
    # Slots already paid for on an earlier pass over this generation. A judge
    # slot that errored sends the task back to the queue, and re-buying the
    # slot that DID answer on every retry is money spent to learn what the
    # judgement table already says.
    recorded = {j["judge_slot"]: j for j in store.judgements_for(int(gen["gen_id"]))}

    outcomes: list[SlotOutcome] = []
    for slot in JUDGE_SLOTS:
        prior = recorded.get(slot)
        reused = _outcome_from_row(cfg, slot, prior) if prior else None
        if reused is not None:
            store.log_event(
                "judge_slot_reused",
                {"task_id": task_id, "gen_id": int(gen["gen_id"]), "slot": slot},
            )
            outcomes.append(reused)
            continue
        if prior is not None:
            # A recorded judgement that cannot stand in for a call (see
            # _outcome_from_row). Say so: it is one paid call, and silently
            # re-buying a slot that looks recorded is confusing in the ledger.
            store.log_event(
                "judge_slot_unresolved",
                {
                    "task_id": task_id,
                    "gen_id": int(gen["gen_id"]),
                    "slot": slot,
                    "ref": f"{prior.get('provider')}/{prior.get('model')}",
                },
            )
        exclude = base_exclude | frozenset(o.family for o in outcomes if o.family)
        if not holds_lease(store, task_id, worker_id):
            return _lost_lease(store, task_id, worker_id, stats, f"judge-slot-{slot}")
        outcome = await judge_slot(
            store, cfg, router,
            task=task, gen=gen, source=source, slot=slot,
            exclude_families=exclude, paths=paths, day=day, worker_id=worker_id,
        )
        stats.prompt_tokens += outcome.prompt_tokens
        stats.completion_tokens += outcome.completion_tokens
        outcomes.append(outcome)
        if outcome.lost_lease:
            # The scores are durable in the raw log; the row belongs to
            # somebody else, so this pass ends here rather than deciding it.
            stats.lost_leases += 1
            return LOST_LEASE
        if outcome.scores is None:
            stats.slot_errors += 1
            if outcome.unroutable:
                # Nothing in the pool can take this row: park it now. Coming
                # back tomorrow finds the same families excluded for the same
                # reasons, having re-paid the slots that did answer.
                return _park(
                    store, task, UNROUTABLE_STATE,
                    f"judge-{slot}-unroutable:{outcome.error}"[:200],
                    worker_id=worker_id, stats=stats, counter="unroutable",
                )
            if outcome.payload_error:
                # A provider refused the request itself, so the next claim
                # buys the identical refusal. Park at once rather than eight
                # times: judge_error is re-openable, and the fix is a code or
                # config change the operator has to make first.
                return _park(
                    store, task, ERROR_STATE,
                    f"judge-{slot}-payload:{outcome.error}"[:200],
                    worker_id=worker_id, stats=stats, counter="judge_errors",
                )
            # Transient: hand the task back to the queue. The generation is
            # fine and a later pass can score it - reusing this slot's
            # judgement if it landed.
            #
            # A TIEBREAK-PROVIDER KEY FAULT IS BOUNDED BUT NOT LOUD, recorded
            # 2026-08-19 rather than fixed. providers.ROW_SHAPED_SKIPS covers
            # `family-excluded` only, so a MISSING OR REJECTED KEY on the
            # tiebreak provider (no LIGHTNING/MISTRAL key set, or a 401) is not
            # classed as row-shaped: the tied row is re-queued here and comes
            # back to the same wall until MAX_JUDGE_ATTEMPTS, then parks in
            # judge_error. Bounded - 8 claims, no paid calls, re-openable, and
            # judge_error is exactly where an operator looks - but noisier and
            # slower than the unroutable park it deserves, which would say
            # "your key is missing" on the first pass. Left alone because
            # widening ROW_SHAPED_SKIPS to key faults would change the
            # GENERATOR's classification too, and that one is load-bearing:
            # generate.py already distinguishes no_eligible_model from
            # provider_fault and a keyless wave must not be marked terminal.
            #
            # LEDGER'D, not fixed (round 4): there is no provider_fault
            # equivalent here. generate.py parks a fleet-wide outage after 3
            # claims with `exhausted:provider-fault`; the judge spends all 8
            # discovering the same outage and then parks in judge_error. Both
            # end re-openable and neither ends in `rejected`, so the cost is
            # five extra claims per row during an outage - real, bounded, and
            # cheaper to pay than a second classification of "the provider is
            # down" that could disagree with the generator's.
            if int(task.get("attempts") or 0) >= MAX_JUDGE_ATTEMPTS:
                # In AUDIT mode a sampled row that the fleet could not serve
                # ships on its gates rather than dying here: an unsampled twin
                # would already have been accepted with zero calls, so parking
                # this one makes survival depend on the hash. Kept at the CAP,
                # never on the first transient failure - bailing out early
                # would collapse the sample to zero during any 60 s cooling
                # window and there would be no quality evidence at all.
                if cfg.routing.judge_mode == "audit":
                    return _audit_accept(
                        store, task_id, gen, worker_id=worker_id, stats=stats,
                        sample=audit_sample,
                        disposition=AUDIT_UNJUDGED_DISPOSITION,
                        outcome="audit-accept-unjudged",
                        reason=f"judge-slot-{slot}:{outcome.error}"[:200],
                    )
                return _park(
                    store, task, ERROR_STATE, f"judge-slot-{slot}:{outcome.error}"[:200],
                    worker_id=worker_id, stats=stats, counter="judge_errors",
                )
            if not _set_state(store, task_id, JUDGE_STATE_FROM, None, worker_id=worker_id):
                stats.lost_leases += 1
                return LOST_LEASE
            return JUDGE_STATE_FROM

    fleet_active = thresholds_active(store) > 0
    rules = active_rules_by_model(store)
    slot_decisions: list[SlotDecision] = []

    def _apply_slot(outcome: SlotOutcome) -> SlotDecision:
        model = outcome.ref.model if outcome.ref else None
        decision = resolve_slot(
            outcome.scores, model=model, rules=rules, fleet_active=fleet_active
        )
        if decision.coerced:
            store.log_event(
                "judge_unfitted_coerced",
                {
                    "task_id": task_id,
                    "slot": outcome.slot,
                    "model": model,
                    "would_be": PASS,
                    "verdict": decision.verdict,
                    "reason": decision.reason,
                },
            )
        if not decision.valid:
            store.log_event(
                "judge_threshold_invalid",
                {
                    "task_id": task_id,
                    "slot": outcome.slot,
                    "model": model,
                    "rule": decision.rule,
                    "threshold": decision.threshold,
                    "reason": decision.reason,
                    "verdict": decision.verdict,
                },
            )
        return decision

    slot_decisions = [_apply_slot(o) for o in outcomes]
    verdicts = [d.verdict for d in slot_decisions]
    already = _regenerating or has_regenerated(store, task_id)
    action = decide(verdicts, already_regenerated=already)
    tiebreak_unroutable = False

    if action == "tiebreak":
        stats.tiebreaks += 1
        prior = recorded.get(TIEBREAK_SLOT)
        tiebreak = _outcome_from_row(cfg, TIEBREAK_SLOT, prior) if prior else None
        if tiebreak is None:
            exclude = base_exclude | frozenset(o.family for o in outcomes if o.family)
            if not holds_lease(store, task_id, worker_id):
                return _lost_lease(store, task_id, worker_id, stats, "judge-tiebreak")
            tiebreak = await judge_slot(
                store, cfg, router,
                task=task, gen=gen, source=source, slot=TIEBREAK_SLOT,
                role="tiebreak", prompt_id=TIEBREAK_PROMPT,
                exclude_families=exclude, paths=paths, day=day, worker_id=worker_id,
            )
            stats.prompt_tokens += tiebreak.prompt_tokens
            stats.completion_tokens += tiebreak.completion_tokens
            if tiebreak.lost_lease:
                stats.lost_leases += 1
                return LOST_LEASE
        outcomes.append(tiebreak)
        if tiebreak.scores is None:
            stats.slot_errors += 1
            if not tiebreak.unroutable:
                if tiebreak.payload_error or int(task.get("attempts") or 0) >= (
                    MAX_JUDGE_ATTEMPTS
                ):
                    # Same rule as the slot arm, with one exception kept: a
                    # payload_error is a defect in the request WE built, not a
                    # fleet that is busy, so it stays an error to be looked at.
                    if cfg.routing.judge_mode == "audit" and not tiebreak.payload_error:
                        return _audit_accept(
                            store, task_id, gen, worker_id=worker_id, stats=stats,
                            sample=audit_sample,
                            disposition=AUDIT_UNJUDGED_DISPOSITION,
                            outcome="audit-accept-unjudged",
                            reason=f"tiebreak:{tiebreak.error}"[:200],
                        )
                    return _park(
                        store, task, ERROR_STATE, f"tiebreak:{tiebreak.error}"[:200],
                        worker_id=worker_id, stats=stats, counter="judge_errors",
                    )
                if not _set_state(store, task_id, JUDGE_STATE_FROM, None, worker_id=worker_id):
                    stats.lost_leases += 1
                    return LOST_LEASE
                return JUDGE_STATE_FROM
            # No third family can take this row. On the shipped pool that is
            # now a rare event rather than the norm - it used to fire on every
            # long gpt-oss row, because gemma was the only tiebreak family
            # separation left and a stale 8192 pin removed it on length; both
            # cerebras windows were probed and corrected on 2026-08-19 and the
            # preflight reports no gaps. Decide on the two judges we have
            # rather than park a fully-judged row: the disagreement stands
            # unresolved, and an unresolved disagreement is not an accept.
            tiebreak_unroutable = True
            action = "reject"
            store.log_event(
                "tiebreak_unroutable_two_judge_decision",
                {
                    "task_id": task_id,
                    "gen_id": int(gen["gen_id"]),
                    "verdicts": verdicts,
                    "error": str(tiebreak.error)[:300],
                },
            )
        else:
            tiebreak_decision = _apply_slot(tiebreak)
            slot_decisions.append(tiebreak_decision)
            verdicts.append(tiebreak_decision.verdict)
            action = decide(verdicts, already_regenerated=already)

    # Fenced BEFORE the decision is logged, not only before it is written: a
    # pass that reused both recorded slots buys nothing, so the per-slot check
    # above never ran, and a stale worker would otherwise log a second
    # judge_decision for a row somebody else already decided - two decision
    # events for one row, and the batch counters double.
    if not holds_lease(store, task_id, worker_id):
        return _lost_lease(store, task_id, worker_id, stats, "judge-decision")

    provisional = not fleet_active
    if provisional:
        rule_used = f"min-axis>={PASS_MIN} pass, <={FAIL_MAX} fail"
    else:
        parts = []
        for decision in slot_decisions:
            if decision.fitted and decision.valid:
                parts.append(f"{decision.model}:{decision.rule}>={decision.threshold}")
            elif not decision.valid:
                parts.append(f"{decision.model}:invalid:{decision.reason}")
            else:
                parts.append(f"{decision.model}:unfitted")
        rule_used = "; ".join(parts)
    store.log_event(
        "judge_decision",
        {
            "task_id": task_id,
            "gen_id": int(gen["gen_id"]),
            "attempt": int(gen["attempt"]),
            "action": action,
            "provisional": provisional,
            "any_unfitted": any(d.unfitted for d in slot_decisions),
            "rule": rule_used,
            "slot_rules": [
                {"slot": outcome.slot, **decision.as_event()}
                for outcome, decision in zip(outcomes, slot_decisions)
            ],
            "generator_family": gen_family,
            "verdicts": verdicts,
            "tiebreak_unroutable": tiebreak_unroutable,
            "scores": [
                {
                    "slot": o.slot,
                    "ref": f"{o.ref.provider}/{o.ref.model}" if o.ref else None,
                    "family": o.family,
                    "reused": o.reused,
                    **(o.scores.as_row() if o.scores else {}),
                }
                for o in outcomes
            ],
            "already_regenerated": already,
        },
    )

    # The counters follow the WRITE, not the intention - `decided` and
    # `outcomes` included. The fence check above is a PROBE, not a guarantee:
    # a re-claim landing inside log_event still leaves this pass unable to
    # write, and counting the decision here reported a decision this worker
    # did not make (the batch line read decided=1 accepted=0 lost-lease=0
    # beside a judge_decision event for somebody else's row).
    def decided(outcome_action: str) -> None:
        stats.decided += 1
        stats.outcomes[outcome_action] = stats.outcomes.get(outcome_action, 0) + 1

    if action == "accept":
        if not _set_state(store, task_id, ACCEPTED_STATE, "judge:accept", worker_id=worker_id):
            stats.lost_leases += 1
            return LOST_LEASE
        decided(action)
        stats.accepted += 1
        return ACCEPTED_STATE
    if action == "reject":
        reason = "judge:reject-tiebreak-unroutable" if tiebreak_unroutable else "judge:reject"
        if not _set_state(store, task_id, REJECTED_STATE, reason, worker_id=worker_id):
            stats.lost_leases += 1
            return LOST_LEASE
        decided(action)
        stats.rejected += 1
        return REJECTED_STATE

    # ---- the one rationale-fed regeneration ----
    # Counted here rather than after a write because the ACT this decision
    # authorises is the paid call below, not a state change; if the
    # regeneration's own disposition is fenced out, that path reports its own
    # lost lease.
    decided(action)
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
        paths=paths, worker_id=worker_id, day=day, stats=stats,
        audit_sample=audit_sample, _regenerating=True,
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
        f"unroutable={stats.unroutable} judge-err={stats.judge_errors} "
        f"lost-lease={stats.lost_leases} tokens={stats.tokens} [{outcomes}]"
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
    audit_sample: float = DEFAULT_AUDIT_SAMPLE,
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
    idle_announced = False

    while max_batches is None or batches < max_batches:
        stats = JudgeStats()
        for stream in streams:
            claimed = store.claim_tasks(
                worker_id, n_workers, stream=stream,
                state_from=JUDGE_STATE_FROM, state_to=JUDGE_STATE_TO,
            )
            stats.claimed += len(claimed)
            for task in claimed:
                try:
                    await judge_task(
                        store, cfg, router, task,
                        paths=paths, worker_id=worker_id, day=day, stats=stats,
                        audit_sample=audit_sample,
                    )
                except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:
                    # One unforeseen failure must not abandon the rest of the
                    # batch. The task keeps its lease and is recovered when
                    # that expires; everything paid for is already on disk.
                    store.log_event(
                        "worker_task_error",
                        {
                            "task_id": task["task_id"],
                            "worker": worker_id,
                            "error": f"{type(exc).__name__}: {exc}"[:500],
                        },
                    )
                    stats.slot_errors += 1
        batches += 1
        for name in (
            "claimed", "decided", "accepted", "rejected", "tiebreaks",
            "regenerated", "slot_errors", "unroutable", "judge_errors", "skipped",
            "lost_leases", "prompt_tokens", "completion_tokens",
        ):
            setattr(totals, name, getattr(totals, name) + getattr(stats, name))
        for key, count in stats.outcomes.items():
            totals.outcomes[key] = totals.outcomes.get(key, 0) + count
        # Idle ticks are announced once, not every poll (see generate.py).
        if stats.claimed or not idle_announced:
            print(format_batch_line(batches, stats))
        idle_announced = stats.claimed == 0
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
        "unroutable": totals.unroutable,
        "judge_errors": totals.judge_errors,
        "skipped": totals.skipped,
        "lost_leases": totals.lost_leases,
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
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stream", action="append", default=None, help="repeatable")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--forever", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--allow-pool-gaps",
        action="store_true",
        help="start even though a judge slot cannot be filled for long rows",
    )
    parser.add_argument(
        "--audit-sample",
        type=float,
        default=DEFAULT_AUDIT_SAMPLE,
        help="audit mode only: fraction of rows that buy the full dual treatment",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.audit_sample <= 1.0:
        parser.error(f"--audit-sample must be in [0, 1], got {args.audit_sample}")

    cfg = load_build_config(args.config)
    print(f"loaded {load_dotenv_keys()} key(s) from .env")
    if cfg.routing.judge_mode == "audit":
        print(f"judge_mode=audit audit_sample={args.audit_sample}")
    else:
        print(f"judge_mode={cfg.routing.judge_mode} (--audit-sample ignored)")
    # The generator role is in here because a borderline pair buys ONE
    # rationale-fed regeneration through generate_once - a judge fleet that
    # cannot generate would park those rows half-decided.
    if not print_preflight(
        cfg,
        ("judge", "tiebreak", "generator"),
        allow_pool_gaps=args.allow_pool_gaps,
        judge_reply_tokens=JUDGE_MAX_TOKENS,
    ):
        raise SystemExit(2)
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
                audit_sample=args.audit_sample,
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
