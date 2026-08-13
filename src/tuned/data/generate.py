"""Async generation worker - claim a task, call a teacher, gate the answer.

This is the loop that spends money, so its shape is dictated by two rules
that outrank throughput.

RAW FIRST, ALWAYS. Every paid response is appended to an immutable NDJSON
log under raw/gen/<day>/ BEFORE any row reaches SQLite, and the envelope it
writes carries every DB-row key store.reconcile_raw needs to rebuild that
row (kind, task_id, attempt, provider, model, think, answer, usage,
timings). A crash between the two writes therefore loses an INDEX row, never
an answer somebody paid for: reconcile_raw re-reads the log and re-indexes
it. The reverse order would lose the answer itself.

The crash window costs at most one re-generation, never a double-spend of a
recorded row: `attempt` is taken from the task's own claim counter, the
generation table is UNIQUE(task_id, attempt), and a task re-claimed after a
crash always comes back with a HIGHER attempt. So the orphaned raw record
and the fresh one are different rows, both durable, and reconcile_raw
recovers the orphan without ever colliding with the retry.

ONE CLAIM, ONE LEASE, ONE WRITER. Every task-state write is fenced with
expect_worker=. A worker that stalled past its lease has already had its
task legitimately re-claimed by somebody else; when it wakes up and reports,
the fence turns its write into a no-op and it drops the result instead of
overwriting the live holder's row.

A ROUTING FAILURE IS NOT A REJECT. `rejected` means "this example is wrong
about the law" and it is what reject-rate statistics are read over; a row
nothing in the pool could serve parks in `gen_unroutable` instead, where
`tasks.py --reopen` can bring it back once the pool is widened. Nothing
here closes a task because the Router said "not retryable" - a missing key
is not retryable either, and treating that as terminal marked an entire
keyless wave dead with zero calls made. The fleet also REFUSES TO START
when a routed role has no key or the judge pool cannot fill a slot for the
longest row the length band permits (preflight_messages): both facts are
knowable in a second and cost a wave of half-paid rows to discover at
runtime.

WHAT THE TEACHER IS SHOWN, THE GATES AND THE JUDGE SEE TOO. prompt_registry's
first caller contract: GateContext.source_text - and later the judge's
{source} slot - must be the concatenation of EVERY grounding slot the
generator was given ({source}, {section_text}, {old/new_section_text},
{savings_text}), not {source} alone. Pass less and a teacher that correctly
cites the very section it was handed is scored as having invented it, which
is a PERMANENT reject that burns the seed. build_prompt() returns that
concatenation as `grounding` and it is the only string any downstream check
is allowed to use; judge.py imports build_prompt for exactly this reason.

Gates run with citation_index=None during the pilot: the existence half of
the citation gate is SKIPPED (detail carries novel_skipped) and verify.py
must re-run it with the real index before any row is promoted. A gate row
carrying novel_skipped means "unverified", never "passed".

Run:  python -m tuned.data.generate --config configs/data_law_v1.yaml
      --stream synthesis --n-workers 4 [--forever] [--max-batches N]
"""

import asyncio
import json
import os
import socket
import sqlite3
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

from tuned.data import gates, prompt_registry
from tuned.data.config import ModelRef
from tuned.data.gates import GateContext, split_think
from tuned.data.jsonl import append_ndjson
from tuned.data.providers import (
    CHARS_PER_TOKEN_INDIC,
    CHARS_PER_TOKEN_LATIN,
    CONTEXT_SAFETY_MARGIN,
    DEFAULT_JUDGE_REPLY_TOKENS,
    ProviderError,
    context_estimate,
    pool_gaps,
    undersized_families,
    unkeyed_roles,
)
from tuned.data.store import utcday, utcnow

# Task states this module owns. 'judging' is the hand-off to judge.py.
JUDGING_STATE = "judging"
PENDING_STATE = "pending"
REJECTED_STATE = "rejected"
# No model in the pool can serve this ROW (every family excluded by context
# length, or nothing keyed at all). Parked, not claimable, visible in
# task_counts, and re-opened by `tasks.py --reopen` once the pool is widened.
#
# Deliberately NOT `rejected`: that state means "this example is wrong about
# the law" and it is what reject-rate statistics are computed over. A routing
# failure is a fact about the fleet, and mixing the two makes both unreadable
# - which is exactly what happened when a keyless batch marked a whole wave
# `rejected` and the planner then read the wave as complete.
GEN_UNROUTABLE_STATE = "gen_unroutable"

# Skip reasons that are facts about the ROW rather than about the moment or
# the configuration. "family-excluded" here always means the context filter:
# generate.py excludes exactly the families too small to hold this prompt, so
# a pool emptied by it will be just as empty on the next claim. A missing key,
# a cooling breaker and a spent daily budget are all about the fleet, and all
# three lift without the row changing at all.
ROW_SHAPED_SKIPS = frozenset({"family-excluded"})

# How many times a task may be claimed before a regenerate-disposition stops
# buying another attempt. Counted on task.attempts, which claim_tasks bumps -
# and which tasks.reopen_tasks zeroes, because a row parked at the cap by a
# fleet-wide failure has spent none of its budget on the answer.
MAX_ATTEMPTS = 3

# Prompt ids for the judge's two calls. They live in this module, beside the
# judge sizing that the startup preflight needs, because judge.py imports
# generate.py and not the reverse; judge.py re-exports them under its own
# names so there is one definition of each.
JUDGE_PROMPT_ID = "judge_pointwise_v1"
TIEBREAK_PROMPT_ID = "judge_tiebreak_v1"

# The character the worst-case judge prompt is built out of: Devanagari DA.
# The point is the SCRIPT, not the letter - the corpus is largely Indic and
# those codepoints cost ~2.7x what the gates' chars/4 band charges for them.
WORST_CASE_CHAR = "द"

# Roles the GENERATION fleet refuses to start without. The judging roles are
# in here for the same reason the judge pool is checked: filling a queue no
# judge can drain is money spent on rows that will park, and a judge role
# with no key drains nothing at all.
GENERATOR_PREFLIGHT_ROLES = ("generator", "judge", "tiebreak")

# Reasoning-effort ladder for retries. A regeneration that failed on format,
# an empty trace or a missing self-check is usually a model that did not
# think hard enough, so the next attempt asks for more. Only sent to models
# whose config actually declares reasoning_effort - an unknown parameter is a
# 400 at some providers, and providers.py treats 400 as "our payload is
# broken everywhere", aborting the call instead of failing over.
EFFORT_LADDER = ("low", "medium", "high")
DEFAULT_EFFORT = "medium"

# Answer-side token allowance on top of the band's think ceiling. The band is
# an estimate in the same chars/4 currency the gates use.
ANSWER_TOKEN_ALLOWANCE = 1000

# The slots that are MATERIAL - what the teacher may cite, and therefore what
# the gates and the judge must be shown. Fixed order so the same task always
# produces the same grounding string (it is compared, hashed and re-derived).
# {scenario} is deliberately NOT here: it is posture and dates, not citable
# material, and prompt_registry's contract enumerates the grounding slots.
GROUNDING_SLOTS = (
    "source",
    "section_text",
    "old_section_text",
    "new_section_text",
    "savings_text",
)

# The standing ask per task type. The templates carry the craft instructions;
# {question} is the specific thing to answer, and for pilot seeds (case text
# with no question attached) it is this. A seed whose meta_json carries its
# own "question" overrides it.
QUESTION_BY_TASK_TYPE = {
    "irac_analysis": (
        "Decide what this matter turns on and what follows for the party you act for."
    ),
    "statute_qa": (
        "What does this provision require, and how does it apply to the matter above?"
    ),
    "drafting": (
        "Settle the instrument this matter calls for, and say what it achieves."
    ),
    "summarization": (
        "State what was decided here and the reasoning that decides it."
    ),
    "transition": (
        "Which enactment governs each limb of this matter, and on what date does "
        "each of those questions turn?"
    ),
}

# Appended to the generator's user turn when a judge sends work back once.
# Appended to the existing turn rather than added as a second user message:
# consecutive same-role turns are rejected by some OpenAI-compatible servers
# with a 400, which providers.py (correctly) treats as non-retryable and does
# not fail over. The original output is never edited or shown back - the
# teacher writes a fresh answer.
REVIEWER_NOTE_TEMPLATE = (
    "\n\nA reviewer who read an earlier attempt at this same matter noted: "
    "{note}\n"
    "Write the answer afresh and in full, on your own terms, with that in mind. "
    "Do not refer to the earlier attempt or to this note."
)


class SlotError(ValueError):
    """The task cannot be rendered from what the seed carries.

    Permanent: no teacher call is made, no money is spent, and the task is
    rejected with a diagnostic rather than retried into the same wall.
    """


@dataclass(frozen=True)
class PromptBundle:
    messages: list[dict]
    # CONTRACT: every grounding slot, concatenated. GateContext.source_text
    # is THIS string and nothing else - it is the citation allow-list, and
    # widening it would let an invented authority in.
    grounding: str
    # What the JUDGE is shown as {source}. Identical to `grounding` on every
    # stream but transition, where it also carries {scenario} - see
    # judge_source_text.
    judge_source: str
    slots: dict
    # TWO estimates, in two currencies, on purpose:
    #
    # prompt_est_tokens is chars/4 and belongs to the GATES. build.length_band
    # is calibrated in that currency across the whole build, so changing it
    # would silently re-scale every length gate in the dataset.
    #
    # context_est_tokens belongs to ROUTING. It counts chat-template overhead
    # and charges Indic script at its real rate, because comparing it against
    # a model's max_context is a decision where being wrong low is a 400 and
    # being wrong high is a failover to a bigger model.
    prompt_est_tokens: int
    context_est_tokens: int


@dataclass
class GenResult:
    task_id: str
    attempt: int
    ok: bool = False
    gen_id: int | None = None
    disposition: str | None = None
    failed_gates: tuple[str, ...] = ()
    ref: ModelRef | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    skipped: str | None = None
    state: str | None = None
    # Nothing in the POOL can serve this row: every family that could hold the
    # prompt was excluded, or every model that was tried answered "too long
    # for my window". Both are facts about the row, so the task parks rather
    # than burning two more claims arriving at the same wall - and it parks in
    # gen_unroutable, not rejected, because it is recoverable.
    unroutable: bool = False
    # The Router had nothing to try at all (as opposed to trying and failing).
    # Its reasons are in route_skips; a missing key belongs here and must NOT
    # close the task, because the key can land at any moment.
    no_eligible_model: bool = False
    route_skips: tuple[str, ...] = ()
    # Something WAS tried and the fault was the provider's: a 5xx, a 403, a
    # revoked key, a 429 storm the client could not ride out. Nothing about
    # this row was tested, so exhausting on it must not spell `rejected` -
    # which is read as "this example is wrong about the law". The payload
    # class (a 400 with no context marker) is deliberately NOT this: that one
    # is our bug and hiding it in a parking state hides it entirely.
    provider_fault: bool = False


@dataclass
class BatchStats:
    claimed: int = 0
    gen_ok: int = 0
    gated_out: int = 0
    errors: int = 0
    lost_leases: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    dispositions: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def absorb(self, result: "GenResult", *, landed: bool = True) -> None:
        """Fold one result into the batch. `landed` is whether its state write
        took effect.

        The counters follow the WRITE, not the intention - the same rule
        judge.py's decision counters follow. A worker that stalled past its
        lease still spent the tokens (so those are always counted) but the
        disposition it computed belongs to nobody: reporting it makes a batch
        that lost every lease read as a batch of clean generations.
        """
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        if not landed:
            self.lost_leases += 1
            return
        if result.error is not None or result.skipped is not None:
            self.errors += 1
            return
        self.gen_ok += 1
        if result.disposition is not None:
            self.gated_out += 1
            self.dispositions[result.disposition] = (
                self.dispositions.get(result.disposition, 0) + 1
            )


# --------------------------------------------------------------------------
# Wiring: budget-gated router, worker identity.
# --------------------------------------------------------------------------

def budget_ok_for(store, cfg):
    """The `budget_ok` callable providers.Router takes.

    providers.py never imports the store on purpose (per-minute limits are
    client-side, per-DAY limits are a ledger that must survive restarts), so
    the daily cap arrives as this injected callable. A ref the config does not
    know is allowed through rather than blocked: the router only ever asks
    about refs it routed from that same config, so an unknown one means the
    config changed under a running fleet, and blocking every call would be a
    worse failure than briefly not enforcing a cap.
    """

    def budget_ok(provider: str, model: str, est_tokens: int) -> bool:
        try:
            _, model_cfg = cfg.model_for(ModelRef(provider, model))
        except KeyError:
            return True
        return store.reserve_budget(provider, model, est_tokens, limits=model_cfg.limits)

    return budget_ok


def make_router(store, cfg, **kwargs):
    """Router wired to the store's daily ledger (contract 7)."""
    from tuned.data.providers import Router

    return Router(cfg, budget_ok=budget_ok_for(store, cfg), **kwargs)


def judge_messages(
    source: str,
    candidate_think: str,
    candidate_answer: str,
    *,
    prompt_id: str = JUDGE_PROMPT_ID,
) -> list[dict]:
    """Render one judge call. THE renderer - judge_slot calls this one.

    It lives here rather than in judge.py because the startup preflight has to
    size the judge's largest possible call and judge.py imports this module,
    not the other way round. judge.py re-exports both names, so there is
    exactly one definition of each and the preflight cannot end up measuring a
    prompt nobody sends.
    """
    return prompt_registry.render(
        prompt_id,
        source=source,
        candidate_think=candidate_think,
        candidate_answer=candidate_answer,
    )


def judge_needed_tokens(
    messages: Sequence[Mapping], *, reply_tokens: int = DEFAULT_JUDGE_REPLY_TOKENS
) -> int:
    """Context a judge call needs: the rendered prompt plus its own reply.

    ROUTING currency (script- and template-aware), never the gates' chars/4:
    this number is compared against hard context ceilings, where under-
    counting means a truncated judge prompt or a 400.
    """
    return context_estimate(messages) + int(reply_tokens)


def judge_tokens_for_generator_window(
    cfg,
    window: int | None,
    *,
    reply_tokens: int = DEFAULT_JUDGE_REPLY_TOKENS,
    prompt_id: str = JUDGE_PROMPT_ID,
) -> int:
    """The largest judge call a generation routed to a `window`-token model can make.

    `window=None` means "no ceiling", which is the build-wide worst case:
    everything the length band permits, in the script that tokenizes hardest.

    With a ceiling, the row is bounded twice over and the smaller bound wins.
    Everything the judge is shown came through that one generation call:

    * the {source} is the grounding slots, a SUBSET of the generator's own
      prompt - and that prompt had to satisfy `required_context(prompt +
      max_output) <= window`, which is what `undersized_families` enforces
      before the call is made;
    * the candidate (trace + answer) is what the call produced, so it is at
      most `max_output_tokens(cfg)` tokens of reply. That premise is ENFORCED,
      not assumed: `reply_over_budget` fails a longer candidate into a
      regeneration in `generate_once`. It used to rest on nothing -
      `check_length_band` bounds `prompt + think + answer` in chars/4 and a
      short prompt leaves room for a reply of twice this budget, at which
      point the number below is no narrowing at all.

    Converting the reply back into characters is the one step that is not a
    rearrangement of numbers the code already enforces: it charges every reply
    token at CHARS_PER_TOKEN_LATIN, the most characters this module's own
    model ever gives a token, so re-estimating those characters in Devanagari
    over-counts rather than under-counts. That direction matters - under-
    counting here would clear a judge that then cannot hold the row, which is
    R3-C2 with extra steps.
    """
    band_chars = int(cfg.build.length_band.total_max * CHARS_PER_TOKEN_LATIN)
    if window is None:
        chars = band_chars
    else:
        reply = max_output_tokens(cfg)
        material = max(0.0, window / CONTEXT_SAFETY_MARGIN - reply) * CHARS_PER_TOKEN_INDIC
        chars = min(band_chars, int(material) + int(reply * CHARS_PER_TOKEN_LATIN))
    return judge_needed_tokens(
        judge_messages(WORST_CASE_CHAR * chars, "", "", prompt_id=prompt_id),
        reply_tokens=reply_tokens,
    )


def min_judge_tokens(
    cfg,
    *,
    reply_tokens: int = DEFAULT_JUDGE_REPLY_TOKENS,
    prompt_id: str = JUDGE_PROMPT_ID,
) -> int:
    """The SMALLEST judge call this build can make - a floor, not an estimate.

    Three terms, every one of them present in every judge call: the judge
    template, the reply allowance, and the least candidate the length band
    lets through. `check_length_band` fails a trace under `think_min` and an
    answer under `answer_min`, and every judgeable stream expects a trace
    (`expects_reasoning` is false only for replay, which `judge.py` never
    judges), so no judged row carries less than that.

    Deliberately UNDER-stated at each step, because its only job is to stop a
    model that can serve nothing from reading as a filled slot - over-stating
    it would invent gaps. The {source} is taken as empty (a real row always
    has grounding), and the candidate's characters are counted as LATIN, the
    script that gives the fewest tokens per character.

    `providers.pool_gaps` takes it as `servable_floor_tokens`, which is what
    `PoolGap.unservable` is asked at. Without it the question is asked at size
    ZERO - a size the length band cannot produce - so a judge too small for
    any row at all reads as servable, and `--allow-pool-gaps` then clears the
    refusal on the grounds that the short rows still run.
    """
    band = cfg.build.length_band
    think = "a" * int(band.think_min * CHARS_PER_TOKEN_LATIN)
    answer = "a" * int(band.answer_min * CHARS_PER_TOKEN_LATIN)
    return judge_needed_tokens(
        judge_messages("", think, answer, prompt_id=prompt_id), reply_tokens=reply_tokens
    )


def judge_sizer(cfg, *, reply_tokens: int = DEFAULT_JUDGE_REPLY_TOKENS):
    """The per-generator-window sizing hook `providers.pool_gaps` takes.

    Same renderer and same estimator as everything else in this file; what the
    hook adds is that a generator family which CANNOT be handed the longest row
    the band permits is not checked as though it could. A refusal over a
    combination that cannot occur is not a free refusal: it is the operator
    reaching for --allow-pool-gaps.
    """

    def needed_for_window(window: int | None, role: str) -> int:
        return judge_tokens_for_generator_window(
            cfg,
            window,
            reply_tokens=reply_tokens,
            prompt_id=TIEBREAK_PROMPT_ID if role == "tiebreak" else JUDGE_PROMPT_ID,
        )

    return needed_for_window


def worst_case_judge_tokens(
    cfg,
    *,
    reply_tokens: int = DEFAULT_JUDGE_REPLY_TOKENS,
    prompt_id: str = JUDGE_PROMPT_ID,
) -> int:
    """The largest judge call this build can produce, measured the real way.

    Constructed rather than sampled, and then pushed through the SAME renderer
    and the SAME estimator the judge worker uses - which is the whole point.
    Deriving it instead (length_band.total_max + reply) mixes currencies:
    `total_max` is chars/4 by definition, because that is what
    gates.check_length_band compares against, while the judge is sized with
    context_estimate. The two disagree by 9% on English and by 2.3x on
    Devanagari, and the preflight was reading the small one.

    The worst case: the length gate passes a row whose prompt + trace + answer
    is at most `total_max` chars/4 tokens, i.e. `total_max * 4` characters.
    The judge's {source} is a SUBSET of the generator's prompt (the grounding
    slots), so that character budget bounds everything the judge is shown -
    and the hardest way to spend it is entirely in Devanagari, which is a
    large fraction of this corpus and tokenizes at 1.5 chars/token.
    """
    return judge_tokens_for_generator_window(
        cfg, None, reply_tokens=reply_tokens, prompt_id=prompt_id
    )


def preflight_messages(
    cfg,
    roles: Sequence[str],
    *,
    allow_pool_gaps: bool = False,
    judge_reply_tokens: int = DEFAULT_JUDGE_REPLY_TOKENS,
) -> tuple[list[str], list[str]]:
    """(refusals, warnings) for a fleet about to start. Pure over cfg + env.

    Two facts are knowable in one second here and cost a wave of half-paid
    rows to discover at runtime:

    * a routed ROLE with no API key anywhere. The worker would claim tasks,
      fail to route every one of them and report a routing error per row.
      Never overridable - there is nothing to override, the calls cannot be
      made.
    * a judge SLOT the pool cannot fill for the longest row the length band
      permits - on family separation, on context length, or because the only
      family left sits behind an API key that has not arrived. All three empty
      the judge role for a whole class of rows, and the row only discovers
      that AFTER paying for the other judge.

    allow_pool_gaps overrides the second one, and ONLY where the override's
    justification is true. A context-length gap empties the slot above a size,
    so an operator who knows about it can legitimately run the short rows while
    the fourth-family judge is sourced. A gap the same walk still reports with
    the length filter removed entirely - typically because an unkeyed family is
    skipped at every size - has no short rows to run: every row pays judge A
    and parks, which is the failure the key filter was added to prevent. Those
    stay refusals whatever the flag says, and `PoolGap.unservable` is the fact
    that decides it - asked at `min_judge_tokens()`, the smallest call this
    build can make, because a judge under that window serves no row at all and
    asking at size zero would let it read as a slot the short rows can use.

    The tiebreak's own gap is a warning, never a refusal: judge.py has a
    defined, unpaid fallback for it (decide on the two judges, reject the
    unresolved disagreement) and round 1 shipped it deliberately.

    The size the pool is checked at comes from worst_case_judge_tokens, i.e.
    from the judge's own renderer and estimator, capped per generator family
    by what that family's own window permits (judge_sizer). Nothing here
    re-derives either number.
    """
    refusals: list[str] = []
    warnings: list[str] = []
    for role, envs in sorted(unkeyed_roles(cfg, roles).items()):
        refusals.append(
            f"routing.{role} has no usable API key: none of "
            f"{', '.join(envs)} is set, so every {role} call would fail to route. "
            f"Put the key in .env (providers.load_dotenv_keys reads it)."
        )
    gaps = pool_gaps(
        cfg,
        needed_tokens=worst_case_judge_tokens(cfg, reply_tokens=judge_reply_tokens),
        tiebreak_needed_tokens=worst_case_judge_tokens(
            cfg, reply_tokens=judge_reply_tokens, prompt_id=TIEBREAK_PROMPT_ID
        ),
        needed_for_window=judge_sizer(cfg, reply_tokens=judge_reply_tokens),
        servable_floor_tokens=min_judge_tokens(cfg, reply_tokens=judge_reply_tokens),
    )
    for gap in gaps:
        line = f"routing.{gap.role} slot {gap.slot}: {gap.detail}"
        if gap.fatal and (not allow_pool_gaps or gap.unservable):
            refusals.append(line)
        else:
            warnings.append(line)
    return refusals, warnings


def print_preflight(cfg, roles: Sequence[str], *, allow_pool_gaps: bool = False, **kwargs) -> bool:
    """Print the preflight and return True when the fleet may start."""
    refusals, warnings = preflight_messages(
        cfg, roles, allow_pool_gaps=allow_pool_gaps, **kwargs
    )
    for line in warnings:
        print(f"WARNING: {line}")
    for line in refusals:
        print(f"REFUSING: {line}")
    if refusals:
        print(
            "not starting: fix the above, or pass --allow-pool-gaps to run "
            "anyway where the gap only bites ABOVE a row size - running the "
            "short rows while a bigger judge is sourced is a real choice. It "
            "does NOT override a gap that no row size escapes (an unkeyed "
            "judge family is skipped at every size): there are no short rows "
            "to run, so every row would pay for judge A and park. A role with "
            "no usable key at all is never overridable either: those calls "
            "cannot be made."
        )
    return not refusals


def worker_name(prefix: str = "gen") -> str:
    """Stable-ish, unique-per-process worker id: host, pid, short uuid."""
    return f"{prefix}-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"


def usage_recorder(store, day: str | None = None):
    """The Router's on_attempt hook, wired to the daily ledger (contract 7).

    Every HTTP attempt is ledgered, not just the one that returned: the
    client retries a 429 up to six times internally and the Router then fails
    over, so a ledger fed only by the successful response counts one request
    where the providers saw seven - and under-counts hardest exactly when the
    pool is under pressure and the operator most needs the number.

    A transport failure (status None) never reached the provider, so it is
    not a request against anybody's quota and is skipped.
    """

    def on_attempt(ref: ModelRef, status: int | None, usage: dict | None) -> None:
        if status is None:
            return
        usage = usage or {}
        store.record_usage(
            ref.provider,
            ref.model,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            is_429=(status == 429),
            day=day,
        )

    return on_attempt


# --------------------------------------------------------------------------
# Prompt assembly.
# --------------------------------------------------------------------------

def _decode_json(value, default):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return default
    return decoded


def seed_meta(seed: Mapping) -> dict:
    meta = _decode_json(seed.get("meta_json"), {})
    return meta if isinstance(meta, dict) else {}


def seed_answer_key(seed: Mapping) -> dict | None:
    key = _decode_json(seed.get("answer_key_json"), None)
    return key if isinstance(key, dict) else None


def _require(meta: Mapping, name: str, task_type: str, seed_id: str) -> str:
    value = meta.get(name)
    if not value or not str(value).strip():
        raise SlotError(
            f"seed {seed_id!r} carries no {name!r}, which the {task_type!r} template "
            f"requires; plan this task type only over seeds that have it"
        )
    return str(value)


def _require_date(seed: Mapping, meta: Mapping, name: str, task_type: str, seed_id: str) -> date:
    """A date the TRANSITION gates cannot run without.

    check_temporal's undecidable channel is FATAL on this stream, and fatal
    means a PERMANENT reject - after the generation has been paid for. A
    transition seed missing its dates is therefore refused here, unspent,
    rather than bought and thrown away.
    """
    parsed = _parse_date(seed.get(name) or meta.get(name))
    if parsed is None:
        raise SlotError(
            f"seed {seed_id!r} carries no usable {name!r} (ISO yyyy-mm-dd), which the "
            f"{task_type!r} stream needs before it can be gated: without it "
            f"check_temporal cannot decide which code governs and rejects the row "
            f"permanently after paying for it"
        )
    return parsed


def build_slots(cfg, task: Mapping, seed: Mapping) -> dict:
    """Slot values for this task's template, from the seed row.

    Pilot fallbacks are explicit and narrow, because a silent fallback is how
    a stream ends up measuring something other than what it claims:

    * statute_qa - {section_text} comes from the seed's meta_json when the
      seed has one, and otherwise from the seed TEXT itself, so a case-text
      seed reads as "here is the matter, here is the provision" with the same
      material in both. That is honest for the pilot (the teacher is shown
      nothing it is not also being asked about) and it is what the statute
      corpus in P7 replaces.
    * irac_analysis - {focus_issue} falls back to the seed's case_type.
    * drafting - {document_kind}/{party_context} fall back to neutral phrases.
    * transition - NO fallbacks, and the DATES are required too even though
      no template slot holds them. Its slots are the old/new provision texts,
      the savings clause and the dated posture; inventing any of them would
      manufacture the very thing the stream exists to test, and a missing
      offence/proceeding date makes check_temporal fatal-undecidable, i.e. a
      permanent reject of an answer already paid for. Either way the task is
      refused here, unspent.
    """
    meta = seed_meta(seed)
    task_type = task["task_type"]
    seed_id = task["seed_id"]
    source = seed.get("text") or ""
    if not source.strip():
        raise SlotError(f"seed {seed_id!r} has empty text")

    slots: dict = {
        "source": source,
        "question": str(meta.get("question") or QUESTION_BY_TASK_TYPE.get(task_type) or ""),
    }
    if task_type == "irac_analysis":
        case_type = (seed.get("case_type") or "").strip() or "legal"
        slots["focus_issue"] = str(
            meta.get("focus_issue") or f"how the {case_type} point in these papers should be decided"
        )
    elif task_type == "statute_qa":
        slots["section_text"] = str(meta.get("section_text") or source)
    elif task_type == "drafting":
        slots["document_kind"] = str(
            meta.get("document_kind") or "the document this matter now calls for"
        )
        slots["party_context"] = str(
            meta.get("party_context")
            or "the party whose papers these are, on the footing set out above"
        )
    elif task_type == "transition":
        for name in ("scenario", "old_section_text", "new_section_text", "savings_text"):
            slots[name] = _require(meta, name, task_type, seed_id)
        # Not template slots - gate inputs. Validated here so the refusal
        # costs nothing; gate_context reads them from the same places.
        for name in ("offence_date", "proceeding_started"):
            _require_date(seed, meta, name, task_type, seed_id)
    return slots


def grounding_text(slots: Mapping) -> str:
    """The union of the grounding slots, in GROUNDING_SLOTS order, deduped.

    This string is the citation allow-list and the verbatim-scan corpus. See
    the module docstring: anything less turns a correct citation into a
    permanent reject.

    Deduped because a slot is often filled from another slot - the pilot's
    statute_qa fallback puts the seed text in both {source} and
    {section_text}. A part already carried (as normalized text) by one that
    was kept is dropped.

    What that saves, precisely: this string is the JUDGE's {source} and the
    gates' verbatim/citation corpus, so the dedup halves the judge prompt and
    the judge's context estimate. It does NOT shrink the generator's own
    prompt - the teacher is still shown both rendered slots, because that is
    what the template says and trimming it would change what was asked.
    """
    parts: list[str] = []
    kept: list[str] = []
    for name in GROUNDING_SLOTS:
        value = slots.get(name)
        if not value:
            continue
        text = str(value)
        normalized = " ".join(text.split())
        if any(normalized in seen for seen in kept):
            continue
        parts.append(text)
        kept.append(normalized)
    return "\n\n".join(parts)


def judge_source_text(task: Mapping, slots: Mapping, grounding: str) -> str:
    """What the judge is shown as {source}.

    Same as the gates' grounding everywhere except the TRANSITION stream,
    where {scenario} - the posture and the DATES - is appended. Which
    enactment governs is decided by those dates and by nothing else, so a
    judge without them is scoring the one axis it cannot see. It is
    deliberately NOT folded into `grounding`: that string is the citation
    allow-list, and a scenario naming a section would quietly authorise it.
    """
    scenario = slots.get("scenario")
    if task.get("stream") != gates.TRANSITION_STREAM or not scenario:
        return grounding
    return f"{grounding}\n\n{scenario}"


def build_prompt(cfg, task: Mapping, seed: Mapping, *, reviewer_note: str | None = None) -> PromptBundle:
    slots = build_slots(cfg, task, seed)
    messages = prompt_registry.render(task["prompt_id"], **slots)
    if reviewer_note:
        messages = append_reviewer_note(messages, reviewer_note)
    grounding = grounding_text(slots)
    return PromptBundle(
        messages=messages,
        grounding=grounding,
        judge_source=judge_source_text(task, slots, grounding),
        slots=slots,
        # chars/4 for the gates (see PromptBundle), and the conservative,
        # script-aware count for the router.
        prompt_est_tokens=sum(len(m.get("content") or "") for m in messages) // 4,
        context_est_tokens=context_estimate(messages),
    )


def append_reviewer_note(messages: Sequence[Mapping], note: str) -> list[dict]:
    """Append the reviewer's sentence to the LAST user turn (never a new one).

    The note is a critique of an earlier attempt; the earlier attempt itself
    is never shown back, so the teacher cannot patch it and must write the
    answer again.
    """
    out = [dict(m) for m in messages]
    for message in reversed(out):
        if message.get("role") == "user":
            message["content"] = (message.get("content") or "") + REVIEWER_NOTE_TEMPLATE.format(
                note=" ".join(str(note).split())
            )
            return out
    out.append({"role": "user", "content": REVIEWER_NOTE_TEMPLATE.format(note=note).strip()})
    return out


# --------------------------------------------------------------------------
# Gate context and content assembly.
# --------------------------------------------------------------------------

def _parse_date(value) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def expects_reasoning(stream: str) -> bool:
    """Every stream that reaches a teacher expects a trace; replay rows (the
    empty-think slice) are copied, never generated, and never get here."""
    return stream != gates.REPLAY_STREAM


def gate_context(cfg, task: Mapping, seed: Mapping, grounding: str, *, citation_index=None) -> GateContext:
    """The GateContext for one generation - built in ONE place.

    generate.py gates a fresh answer and verify.py re-gates it months later
    with the real citation index; if they built the context differently the
    second pass would measure a different thing and its demotions would be
    noise. Both call this.

    Both dates are read seed-column-first, then meta - the SAME two places
    _require_date checks before the task is allowed to spend. That has to
    hold literally: if this function read one of them from meta alone, a seed
    carrying it in a column would pass the pre-spend check and then arrive
    here undated, which on the transition stream is a fatal-undecidable gate
    and a permanent reject of an answer already paid for.

    decision_date is deliberately not used as a stand-in for
    proceeding_started. They are different facts (when the matter was decided
    vs when it was instituted), and check_temporal uses proceeding_started to
    decide which PROCEDURAL code governs: substituting a later date would
    move a CrPC-era proceeding into the BNSS era and flag a correct citation
    as cross-code. An unknown date is left unknown, which is what the
    undecidable channel is for.
    """
    meta = seed_meta(seed)
    return GateContext(
        think_open=cfg.think_open,
        think_close=cfg.think_close,
        band=cfg.build.length_band,
        citation_index=citation_index,
        source_text=grounding,
        offence_date=_parse_date(seed.get("offence_date") or meta.get("offence_date")),
        proceeding_started=_parse_date(
            seed.get("proceeding_started") or meta.get("proceeding_started")
        ),
        stream=task["stream"],
        expect_reasoning=expects_reasoning(task["stream"]),
        answer_key=seed_answer_key(seed),
    )


def assemble_content(cfg, response) -> tuple[str, str | None, str]:
    """(content, think, answer) from a provider response.

    Three shapes arrive, and only the first two are usable:

    * the model returned reasoning on its own channel (gpt-oss, magistral) -
      wrap it in the trainer's think tags and append the answer text;
    * the model inlined <think>...</think> in the text - split it and
      re-emit it in canonical form, so both shapes index identically;
    * neither - return the text UNCHANGED, with think=None. Nothing is
      fabricated: an empty tag pair here would be a trace we invented, and
      check_think_format is left to fail the row into a regeneration, which
      is the honest outcome.
    """
    text = response.text or ""
    inline_think, inline_answer = split_think(text, cfg.think_open, cfg.think_close)
    reasoning = response.reasoning if response.reasoning else inline_think
    answer = inline_answer if inline_think is not None else text

    if not (reasoning and reasoning.strip()):
        return text, None, text
    think = reasoning.strip()
    body = answer.strip()
    content = f"{cfg.think_open}\n{think}\n{cfg.think_close}\n\n{body}"
    return content, think, body


def max_output_tokens(cfg) -> int:
    band = cfg.build.length_band
    return int(band.think_max + ANSWER_TOKEN_ALLOWANCE)


# The name the reply-budget breach travels under in a disposition string. It
# is NOT a gates.py gate: gates.run_all is re-run offline by verify.py over
# the stored bytes and its result set is a fixed vocabulary, while this is a
# statement about the CALL that produced them (what max_tokens was asked for),
# which the stored row does not carry.
REPLY_BUDGET_GATE = "reply_budget"


def reply_budget_chars(cfg) -> int:
    """The most characters one generation call can physically return.

    `generate_once` sends `max_tokens=max_output_tokens(cfg)`, which bounds
    the reply in TOKENS, and CHARS_PER_TOKEN_LATIN is the loosest
    chars-per-token this module models - no tokenizer here gives a token more
    characters than that. So a well-behaved provider cannot produce a reply
    longer than this, and the bound over-states rather than under-states.
    """
    return int(max_output_tokens(cfg) * CHARS_PER_TOKEN_LATIN)


def reply_over_budget(cfg, think: str | None, answer: str | None) -> int:
    """Characters this candidate is OVER the reply budget; 0 when inside it.

    This enforces the premise `judge_tokens_for_generator_window` rests on -
    "the candidate is at most `max_output_tokens(cfg)` tokens of reply" - and
    it has to be enforced here because nothing else does. `check_length_band`
    tests `prompt + think + answer <= total_max` and `think <= think_max`, in
    chars/4; on the shipped config a row whose prompt is short can spend that
    remainder on a reply of 32,760 characters, twice this budget, and pass
    every gate. The per-family judge sizing would then have cleared a judge
    that cannot hold the row: the whole 23,729 -> 15,104 narrowing IS this
    assumption, and the window contributes almost nothing beside it.

    On a well-behaved provider this never fires - which is the point of
    picking a bound that is a physical fact about the call rather than a
    taste judgement about the answer. What it does catch is the case nothing
    here has met yet: `assemble_content` reads `response.reasoning`, a
    SEPARATE API field, and a provider that does not bill its reasoning
    channel against `max_tokens` can return the gate's full 12,000-character
    trace ON TOP OF a full-length answer. That row passes every gate today.

    Disposition is `regenerate`, not `reject`: an over-long reply is a badly
    shaped answer, not a false statement about the law, and the seed is worth
    asking again. A SYSTEMIC breach therefore shows up first as a run of
    `reply_over_budget` events with the same provider on them, before it
    costs any row its attempt cap.
    """
    return max(0, len(think or "") + len(answer or "") - reply_budget_chars(cfg))


def effort_for_attempt(attempt: int, base: str = DEFAULT_EFFORT) -> str:
    """One rung up the ladder per retry, saturating at the top."""
    try:
        start = EFFORT_LADDER.index(base)
    except ValueError:
        start = EFFORT_LADDER.index(DEFAULT_EFFORT)
    return EFFORT_LADDER[min(start + max(0, attempt - 1), len(EFFORT_LADDER) - 1)]


def effort_params_for_ref(attempt: int):
    """Build the Router's per-ref params hook for this attempt.

    Attempt 1 sends nothing and every model's configured defaults stand. A
    retry asks for more reasoning effort - but the parameter is chosen for
    the ref the Router is ABOUT TO CALL, not for the one it would have
    picked first. That distinction is the whole point: reasoning_effort is a
    gpt-oss parameter, magistral does not declare it, and an unknown field
    earns a 400 - which providers.py classifies as our payload being broken
    everywhere and raises straight through WITHOUT failing over. Choosing the
    params before the failover decision would therefore turn every
    second-attempt failover into a dead task.
    """

    def params_for_ref(ref: ModelRef, model_cfg) -> dict:
        if attempt <= 1:
            return {}
        declared = dict(getattr(model_cfg, "params", None) or {})
        if "reasoning_effort" not in declared:
            return {}
        return {"reasoning_effort": effort_for_attempt(attempt, str(declared["reasoning_effort"]))}

    return params_for_ref


def next_attempt(store, task: Mapping) -> int:
    """The attempt number this generation must claim.

    task.attempts counts CLAIMS and the generation table is
    UNIQUE(task_id, attempt), so the two have to be reconciled: a claim that
    produced no generation (provider down) leaves attempts ahead of the
    generations, and a judge-driven regeneration produces a generation
    without a fresh claim. Taking the max of the two keeps the number both
    unique and monotonic under either skew.
    """
    latest = store.latest_generation(task["task_id"])
    latest_attempt = int(latest["attempt"]) if latest else 0
    return max(int(task.get("attempts") or 0), latest_attempt + 1, 1)


# --------------------------------------------------------------------------
# The paid step.
# --------------------------------------------------------------------------

def raw_gen_path(paths, day: str | None = None) -> str:
    day = utcday(day)
    return str(paths.raw_gen_dir(day) / "gen.ndjson")


def _gen_envelope(task: Mapping, attempt: int, ref: ModelRef, model_family, response,
                  content: str, think: str | None, answer: str, params: Mapping,
                  bundle: PromptBundle, reviewer_note: str | None) -> dict:
    """The immutable record of one paid call.

    Every key store._GEN_COLS reads is here, so store.reconcile_raw can
    rebuild the DB row from this line alone (it supplies raw_path/raw_offset
    itself, from where the line actually sits - never from the envelope).
    The extra keys are ignored by the store's column projection and exist for
    verify.py and for anybody reading the log by hand: `content` in
    particular is the exact string the gates scored, so a re-gate months
    later can score the same bytes instead of a re-assembly of them.
    """
    return {
        "kind": "generation",
        "task_id": task["task_id"],
        "attempt": attempt,
        "provider": ref.provider,
        "model": ref.model,
        "model_family": model_family,
        "params_json": dict(params),
        "think": think,
        "answer": answer,
        "prompt_tokens": int(response.prompt_tokens or 0),
        "completion_tokens": int(response.completion_tokens or 0),
        "think_tokens": len(think or "") // 4,
        "total_tokens": int((response.prompt_tokens or 0) + (response.completion_tokens or 0)),
        "latency_ms": int(response.latency_ms or 0),
        "finish_reason": response.finish_reason,
        "error": None,
        "created_at": utcnow(),
        # Not columns - context for verify.py and for human forensics.
        "content": content,
        "stream": task["stream"],
        "task_type": task["task_type"],
        "seed_id": task["seed_id"],
        "prompt_id": task["prompt_id"],
        "prompt_sha": task["prompt_sha"],
        "reviewer_note": reviewer_note,
        "messages": [dict(m) for m in bundle.messages],
        "response_text": response.text,
        "response_reasoning": response.reasoning,
        "status": response.status,
    }


async def generate_once(
    store,
    cfg,
    router,
    task: Mapping,
    *,
    paths,
    citation_index=None,
    reviewer_note: str | None = None,
    attempt: int | None = None,
    day: str | None = None,
) -> GenResult:
    """One teacher call for one task: render, spend, persist, gate.

    Writes generation/gate/usage rows and run_events; deliberately does NOT
    write the task's state. The caller owns that, because judge.py drives a
    regeneration through this same function and has to decide the outcome
    from the gate result plus its own judgement, not from a state this
    function guessed.
    """
    attempt = next_attempt(store, task) if attempt is None else attempt
    result = GenResult(task_id=task["task_id"], attempt=attempt)

    seed = store.get_seed(task["seed_id"])
    if seed is None:
        result.skipped = "missing-seed"
        store.log_event("generation_skipped", {"task_id": task["task_id"], "reason": "missing-seed"})
        return result
    try:
        bundle = build_prompt(cfg, task, seed, reviewer_note=reviewer_note)
    except (SlotError, KeyError) as exc:
        result.skipped = "slots"
        result.error = str(exc)
        store.log_event(
            "generation_skipped",
            {"task_id": task["task_id"], "reason": "slots", "error": str(exc)},
        )
        return result

    max_tokens = max_output_tokens(cfg)
    # ROUTING currency, not the gates' - see PromptBundle. chars/4 over the
    # rendered messages models neither the chat template's per-turn overhead
    # nor Devanagari (which tokenizes 2-4x harder than the estimate assumes),
    # and under-counting here is what puts an over-long prompt at an 8k model.
    est_tokens = bundle.context_est_tokens + max_tokens
    # CONTEXT ROUTING. cerebras/gpt-oss-120b is an 8k-context model and the
    # first generator in the preference list; a long seed makes
    # prompt + max_tokens exceed that, which is a 400. providers.py now
    # RECOGNISES a context-overflow 400 and fails over rather than aborting,
    # but reaching that point still costs a paid round trip and a wasted
    # attempt, so the families that cannot hold this prompt are excluded
    # before the call and it routes to magistral (40k) directly.
    too_small = undersized_families(cfg, "generator", est_tokens)
    params_for_ref = effort_params_for_ref(attempt)

    try:
        ref, response = await router.complete(
            "generator",
            bundle.messages,
            params_for_ref=params_for_ref,
            max_tokens=max_tokens,
            est_tokens=est_tokens,
            exclude_families=too_small,
            on_attempt=usage_recorder(store, day),
        )
    except ProviderError as exc:
        # Every attempt this call made is already ledgered by on_attempt,
        # including the 429s the client retried through.
        result.error = str(exc)
        skips = frozenset(getattr(exc, "skipped", frozenset()))
        result.no_eligible_model = bool(skips)
        result.route_skips = tuple(sorted(skips))
        # UNROUTABLE means "no model in this pool can serve this ROW", and it
        # is the only shape that closes a task without spending its attempts.
        # Two ways to earn it: every family that could hold the prompt was
        # excluded by the context filter, or every model that WAS tried came
        # back saying the prompt is longer than its window. `not retryable`
        # is NOT one of them - a missing key is also not retryable, and
        # treating it as row-shaped is what marked a whole keyless wave
        # terminal with zero calls made.
        result.unroutable = bool(exc.context_exceeded) or (
            bool(skips) and skips <= ROW_SHAPED_SKIPS
        )
        # A provider answered (or refused to answer) and the fault was on its
        # side. Only meaningful when something was actually TRIED - with a
        # skip set, nothing was, and no_eligible_model already carries that.
        #
        # `not skips` is DEFENCE IN DEPTH and unpinnable by construction: the
        # only error carrying a skip set is the "nothing was tried" one, and
        # apply_gate_disposition consults no_eligible_model before it consults
        # provider_fault, so no mutation of this clause can change a task's
        # state. Kept because the two facts are different facts; never counted
        # as mutation-verified.
        result.provider_fault = not skips and bool(exc.retryable or exc.provider_dead)
        store.log_event(
            "generation_error",
            {
                "task_id": task["task_id"],
                "attempt": attempt,
                "status": exc.status,
                "retryable": exc.retryable,
                "unroutable": result.unroutable,
                "provider_fault": result.provider_fault,
                "skipped": list(result.route_skips),
                "context_exceeded": bool(exc.context_exceeded),
                "est_tokens": est_tokens,
                "excluded_families": sorted(too_small),
                "error": str(exc)[:500],
            },
        )
        return result

    result.ref = ref
    result.prompt_tokens = int(response.prompt_tokens or 0)
    result.completion_tokens = int(response.completion_tokens or 0)

    _, model_cfg = cfg.model_for(ref)
    # What was actually sent - resolved against the ref that answered, which
    # after a failover need not be the one the params were first shaped for.
    params = dict(params_for_ref(ref, model_cfg))
    if params:
        store.log_event(
            "effort_bump",
            {
                "task_id": task["task_id"],
                "attempt": attempt,
                "ref": f"{ref.provider}/{ref.model}",
                "params": params,
            },
        )
    content, think, answer = assemble_content(cfg, response)
    envelope = _gen_envelope(
        task, attempt, ref, model_cfg.family, response, content, think, answer,
        {**params, "reviewer_note_applied": bool(reviewer_note)}, bundle, reviewer_note,
    )

    # ---- raw FIRST. Nothing below this line may run before it. ----
    raw_path = raw_gen_path(paths, day)
    raw_offset = append_ndjson(raw_path, envelope)

    row = dict(envelope)
    row["raw_path"] = raw_path
    row["raw_offset"] = raw_offset
    try:
        gen_id = store.record_generation(row)
    except sqlite3.IntegrityError:
        # (task_id, attempt) is already indexed - a duplicate this worker
        # should not have produced. The response is safe in the raw log
        # either way; adopt the existing row so the gate results still land,
        # but ONLY if it really is the same attempt. Adopting whatever row
        # happens to be latest would attach this answer's gate results to a
        # different answer, which is worse than losing them.
        existing = store.latest_generation(task["task_id"])
        gen_id = (
            int(existing["gen_id"])
            if existing is not None and int(existing["attempt"]) == attempt
            else None
        )
        store.log_event(
            "generation_duplicate",
            {"task_id": task["task_id"], "attempt": attempt, "gen_id": gen_id},
        )
        if gen_id is None:
            result.error = f"attempt {attempt} already indexed under another row"
            return result

    result.ok = True
    result.gen_id = gen_id

    ctx = gate_context(cfg, task, seed, bundle.grounding, citation_index=citation_index)
    gate_results = gates.run_all(content, bundle.prompt_est_tokens, ctx)
    store.record_gates(gen_id, [g.as_row() for g in gate_results])
    result.disposition = gates.disposition(gate_results)
    result.failed_gates = tuple(g.gate for g in gate_results if not g.passed)

    # The premise the judge-pool sizing rests on, made true by construction.
    # See reply_over_budget: `max_tokens` was sent, so a well-behaved provider
    # cannot breach this, and the one shape that can (a reasoning channel not
    # billed against max_tokens) is exactly what the preflight's per-family
    # narrowing would then be wrong about. Recorded as an EVENT rather than a
    # gate row because verify.py re-runs the gates over the stored bytes and
    # would have no way to re-derive the max_tokens this call was made with.
    over = reply_over_budget(cfg, think, answer)
    if over:
        store.log_event(
            "reply_over_budget",
            {
                "task_id": task["task_id"],
                "attempt": attempt,
                "gen_id": gen_id,
                "ref": f"{ref.provider}/{ref.model}",
                "reply_chars": len(think or "") + len(answer or ""),
                "budget_chars": reply_budget_chars(cfg),
                "over_by": over,
                "finish_reason": response.finish_reason,
            },
        )
        result.failed_gates = result.failed_gates + (REPLY_BUDGET_GATE,)
        # A permanent gate already burned the seed; nothing here promotes a
        # reject back to a retry.
        if result.disposition is None:
            result.disposition = "regenerate"
    return result


def apply_gate_disposition(store, task: Mapping, result: GenResult, *, worker_id: str) -> str:
    """Move the task on from what the gates said. Returns the new state.

    * clean            -> judging (judge.py's queue)
    * regenerate       -> pending while attempts remain, else rejected. The
                          next claim bumps reasoning_effort
                          (effort_params_for_ref).
    * reject           -> rejected. A permanent gate means the example is
                          wrong about the law; the seed is burned, never
                          retried.
    * no generation    -> pending while attempts remain (a provider outage is
                          not the task's fault). At the cap it lands in
                          rejected if a provider actually refused the work,
                          and in gen_unroutable if nothing was ever called.
    * unroutable       -> gen_unroutable at once: no model in the pool can
                          hold this row, so the next two claims would meet
                          the identical wall.

    NOTHING here closes a task because the Router said "not retryable". A
    missing key is not retryable either, and a keyless fleet must leave its
    wave exactly where it found it.

    Every write is lease-fenced: if this worker lost its lease while the call
    was in flight, the task already belongs to somebody else and this result
    is dropped rather than allowed to overwrite theirs.
    """
    task_id = task["task_id"]
    attempts = int(task.get("attempts") or 0)
    exhausted = attempts >= MAX_ATTEMPTS

    if not result.ok:
        if result.skipped is not None:
            # A missing seed or an unrenderable slot set is permanent: the
            # next attempt would hit the identical wall, unspent.
            state, disposition = REJECTED_STATE, f"skip:{result.skipped}"
        elif result.unroutable:
            state, disposition = GEN_UNROUTABLE_STATE, "unroutable:generator"
        elif exhausted and result.no_eligible_model:
            # Nothing was ever called, so nothing about this row was judged
            # and it must not be counted as a reject. Park it where re-opening
            # can bring it back once the fleet is fixed.
            state, disposition = (
                GEN_UNROUTABLE_STATE,
                "exhausted:unroutable:" + ",".join(result.route_skips),
            )
        elif exhausted and result.provider_fault:
            # Something was tried and the PROVIDER failed - an outage, a
            # revoked key, a 403, a 429 storm. Same argument as above: the
            # answer was never produced, let alone found wanting, so this is
            # not a reject. It is also the one fleet-wide failure that leaves
            # a wave looking individually rejected, row by row.
            state, disposition = GEN_UNROUTABLE_STATE, "exhausted:provider-fault"
        elif exhausted:
            # What is left is the payload class: a 400/413/422 with no context
            # marker, i.e. our bug. It stays in `rejected` - parking it would
            # file a code defect under "the pool was short".
            #
            # LEDGER'D, not fixed (round 4): `rejected` is terminal and not
            # re-openable, so a SYSTEMIC payload bug costs the whole wave -
            # every row rejected, and the per-seed slots spent with it. The
            # remedy would be `--reopen rejected --disposition exhausted:error`
            # (a re-open guarded on the disposition, so a gate decision can
            # never come back through it), which is a tasks.py change with its
            # own review; the reason for leaving it here is that this state is
            # what makes a code defect visible AS a code defect, and the wave
            # is re-plannable through the per-seed cap either way.
            state, disposition = REJECTED_STATE, "exhausted:error"
        else:
            state, disposition = PENDING_STATE, None
    elif result.disposition is None:
        state, disposition = JUDGING_STATE, None
    elif result.disposition == "regenerate":
        state, disposition = (
            (REJECTED_STATE, "exhausted:regenerate:" + ",".join(result.failed_gates))
            if exhausted
            else (PENDING_STATE, "regenerate:" + ",".join(result.failed_gates))
        )
    else:
        state, disposition = REJECTED_STATE, "reject:" + ",".join(result.failed_gates)

    moved = store.set_task_state(task_id, state, disposition, expect_worker=worker_id)
    if not moved:
        store.log_event(
            "lost_lease",
            {"task_id": task_id, "worker": worker_id, "wanted_state": state},
        )
        result.state = None
        return "lost-lease"
    result.state = state
    return state


# --------------------------------------------------------------------------
# The loop.
# --------------------------------------------------------------------------

def budget_lines(store, cfg, day: str | None = None) -> list[str]:
    """"provider/model spent=X left=Y" for every model used TODAY.

    Only models with spend, so the status line stays readable: a fleet using
    two of the eight configured models should not print six zero rows every
    batch. A model with no daily cap reports "left=-".
    """
    lines = []
    for provider in cfg.providers:
        for model in provider.models:
            used = store.usage_today(provider.name, model.id, day=day)
            spent = used["prompt_tokens"] + used["completion_tokens"]
            if not spent and not used["requests"]:
                continue
            cap = model.limits.get("tpd")
            left = "-" if cap is None else f"{max(0, int(cap) - spent) / 1000:.0f}k"
            lines.append(f"{provider.name}/{model.id} spent={spent / 1000:.1f}k left={left}")
    return lines


def format_batch_line(batch_ix: int, stats: BatchStats, budget: Sequence[str]) -> str:
    dispositions = " ".join(f"{k}={v}" for k, v in sorted(stats.dispositions.items()))
    return (
        f"batch {batch_ix}: claimed={stats.claimed} gen-ok={stats.gen_ok} "
        f"gated-out={stats.gated_out} err={stats.errors} "
        f"lost-lease={stats.lost_leases} "
        f"tokens={stats.tokens} [{dispositions}] | " + " | ".join(budget)
    )


async def run_workers(
    store,
    cfg,
    router,
    *,
    streams: Sequence[str],
    n_workers: int,
    forever: bool = False,
    max_batches: int | None = None,
    paths=None,
    citation_index=None,
    worker_id: str | None = None,
    day: str | None = None,
    idle_sleep_s: float = 5.0,
    sleeper=asyncio.sleep,
) -> dict:
    """Claim -> generate -> gate -> dispose, until the queue or the batch cap runs out.

    n_workers is the in-flight concurrency AND the claim size: one batch
    claims up to n_workers tasks per stream and runs them together, so the
    lease a worker holds is never much older than one call.

    Bounded by construction: without forever=True the loop stops the first
    time a batch claims nothing, and max_batches stops it regardless. Tests
    pass max_batches=1 and never reach the idle sleep.
    """
    if paths is None:
        from tuned.data.paths import build_paths

        paths = build_paths(cfg.build.workdir).ensure()
    worker_id = worker_id or worker_name("gen")
    totals = BatchStats()
    batches = 0
    idle_announced = False

    while max_batches is None or batches < max_batches:
        stats = BatchStats()
        for stream in streams:
            claimed = store.claim_tasks(worker_id, n_workers, stream=stream)
            stats.claimed += len(claimed)
            if not claimed:
                continue
            # return_exceptions: one task that raises something nobody
            # anticipated must not take the other n-1 paid calls in this
            # batch down with it. Its lease simply expires and it is
            # re-claimed later; the batch reports and carries on.
            results = await asyncio.gather(
                *(
                    generate_once(
                        store, cfg, router, task,
                        paths=paths, citation_index=citation_index, day=day,
                    )
                    for task in claimed
                ),
                return_exceptions=True,
            )
            for task, result in zip(claimed, results):
                if isinstance(result, BaseException):
                    if isinstance(result, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                        raise result
                    store.log_event(
                        "worker_task_error",
                        {
                            "task_id": task["task_id"],
                            "worker": worker_id,
                            "error": f"{type(result).__name__}: {result}"[:500],
                        },
                    )
                    stats.errors += 1
                    continue
                landed = apply_gate_disposition(
                    store, task, result, worker_id=worker_id
                ) != "lost-lease"
                stats.absorb(result, landed=landed)
        batches += 1
        totals.claimed += stats.claimed
        totals.gen_ok += stats.gen_ok
        totals.gated_out += stats.gated_out
        totals.errors += stats.errors
        totals.lost_leases += stats.lost_leases
        totals.prompt_tokens += stats.prompt_tokens
        totals.completion_tokens += stats.completion_tokens
        for key, count in stats.dispositions.items():
            totals.dispositions[key] = totals.dispositions.get(key, 0) + count
        # An idle --forever worker polls every few seconds for hours; printing
        # each empty tick buries the batches that did something. Say "idle"
        # once, then stay quiet until the queue produces work again.
        if stats.claimed or not idle_announced:
            print(format_batch_line(batches, stats, budget_lines(store, cfg, day=day)))
        idle_announced = stats.claimed == 0
        if stats.claimed == 0:
            if not forever:
                break
            await sleeper(idle_sleep_s)

    return {
        "batches": batches,
        "claimed": totals.claimed,
        "gen_ok": totals.gen_ok,
        "gated_out": totals.gated_out,
        "errors": totals.errors,
        "lost_leases": totals.lost_leases,
        "prompt_tokens": totals.prompt_tokens,
        "completion_tokens": totals.completion_tokens,
        "dispositions": dict(totals.dispositions),
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
    parser.add_argument(
        "--allow-pool-gaps",
        action="store_true",
        help="start even though a judge slot cannot be filled for long rows",
    )
    args = parser.parse_args(argv)

    cfg = load_build_config(args.config)
    print(f"loaded {load_dotenv_keys()} key(s) from .env")
    # Before anything is claimed: a role with no key, or a judge slot no
    # model can fill, is knowable now and costs paid rows to discover later.
    # The generation worker checks the JUDGE roles too - filling a queue no
    # judge pool can drain is money spent on rows that will park.
    if not print_preflight(
        cfg, GENERATOR_PREFLIGHT_ROLES, allow_pool_gaps=args.allow_pool_gaps
    ):
        raise SystemExit(2)
    paths = build_paths(cfg.build.workdir).ensure()
    store = Store.open(paths.state_db)
    router = make_router(store, cfg)
    streams = args.stream or ["synthesis"]

    async def drive():
        # The router's httpx clients belong to the loop that created them,
        # so they are closed INSIDE it - a second asyncio.run() would be
        # closing pools owned by a loop that no longer exists.
        try:
            return await run_workers(
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
        f"done: batches={totals['batches']} claimed={totals['claimed']} "
        f"gen-ok={totals['gen_ok']} gated-out={totals['gated_out']} "
        f"errors={totals['errors']} tokens={totals['prompt_tokens'] + totals['completion_tokens']}"
    )
    print("task states: " + ", ".join(f"{k}={v}" for k, v in sorted(store.task_counts().items())))
    store.close()

    # Same reasoning as the other data CLIs: httpx/anyio worker threads can
    # outlive the loop and wedge interpreter shutdown after all output is
    # written. Everything durable is already on disk.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
