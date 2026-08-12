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
from tuned.data.providers import ProviderError, undersized_families
from tuned.data.store import utcday, utcnow

# Task states this module owns. 'judging' is the hand-off to judge.py.
JUDGING_STATE = "judging"
PENDING_STATE = "pending"
REJECTED_STATE = "rejected"

# How many times a task may be claimed before a regenerate-disposition stops
# buying another attempt. Counted on task.attempts, which claim_tasks bumps.
MAX_ATTEMPTS = 3

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
    prompt_est_tokens: int


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
    # The call failed for a reason another identical claim cannot fix: no
    # eligible model at all (every family excluded, no key), or a payload the
    # provider refuses. Re-queueing burns two more claims for the same
    # failure, so the task is closed out instead.
    permanent: bool = False


@dataclass
class BatchStats:
    claimed: int = 0
    gen_ok: int = 0
    gated_out: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    dispositions: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def absorb(self, result: "GenResult") -> None:
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
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
    {section_text}, and concatenating them would double every judge prompt
    and every context-length estimate for no added grounding at all. A part
    already carried (as normalized text) by one that was kept is dropped.
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
    est = sum(len(m.get("content") or "") for m in messages) // 4
    grounding = grounding_text(slots)
    return PromptBundle(
        messages=messages,
        grounding=grounding,
        judge_source=judge_source_text(task, slots, grounding),
        slots=slots,
        prompt_est_tokens=est,
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

    proceeding_started comes from the seed's meta and NOWHERE else -
    decision_date is deliberately not used as a stand-in. They are different
    facts (when the matter was decided vs when it was instituted), and
    check_temporal uses proceeding_started to decide which PROCEDURAL code
    governs: substituting a later date would move a CrPC-era proceeding into
    the BNSS era and flag a correct citation as cross-code. An unknown date
    is left unknown, which is what the undecidable channel is for.
    """
    meta = seed_meta(seed)
    return GateContext(
        think_open=cfg.think_open,
        think_close=cfg.think_close,
        band=cfg.build.length_band,
        citation_index=citation_index,
        source_text=grounding,
        offence_date=_parse_date(seed.get("offence_date") or meta.get("offence_date")),
        proceeding_started=_parse_date(meta.get("proceeding_started")),
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
    est_tokens = bundle.prompt_est_tokens + max_tokens
    # CONTEXT ROUTING. cerebras/gpt-oss-120b is an 8k-context model and the
    # first generator in the preference list; a long seed makes
    # prompt + max_tokens exceed that, which is a 400 - and a 400 does not
    # fail over (providers.py treats it as our payload being wrong
    # everywhere), so the task would burn its whole attempt budget on three
    # identical refusals and the pilot's pass rates would be computed over
    # short seeds only. Excluding the families that cannot hold this prompt
    # routes it to magistral (40k) instead.
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
        result.permanent = not exc.retryable
        store.log_event(
            "generation_error",
            {
                "task_id": task["task_id"],
                "attempt": attempt,
                "status": exc.status,
                "retryable": exc.retryable,
                "permanent": result.permanent,
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
                          not the task's fault), else rejected. But a
                          PERMANENT routing/payload failure closes the task
                          immediately: nothing about the next claim would be
                          different, so re-queueing only spends two more
                          claims arriving at the same refusal.

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
        elif result.permanent:
            state, disposition = REJECTED_STATE, "unroutable:generator"
        elif exhausted:
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
                apply_gate_disposition(store, task, result, worker_id=worker_id)
                stats.absorb(result)
        batches += 1
        totals.claimed += stats.claimed
        totals.gen_ok += stats.gen_ok
        totals.gated_out += stats.gated_out
        totals.errors += stats.errors
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
    args = parser.parse_args(argv)

    cfg = load_build_config(args.config)
    print(f"loaded {load_dotenv_keys()} key(s) from .env")
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
