"""Pluggable OpenAI-compatible provider layer for the law_v1 generation pipeline.

Every LLM call in the build goes through here, so the shape of this module
is dictated by three facts about free-tier APIs:

1. **Adding a provider must be pure YAML.**  There is exactly ONE client
   class; per-provider deviation from the OpenAI chat-completions shape
   lives in a named ``Quirk`` (request hook / response hook / Retry-After
   parser) that ``configs/data_law_v1.yaml`` selects by name.  If a new
   provider speaks plain OpenAI, adding it is a config edit and nothing
   more.

2. **Rate limits come in two flavours and only one of them is ours.**
   Per-minute limits (rpm/tpm) are enforced client-side by a ``TokenBucket``
   per (provider, model).  Per-day limits (tpd/rpd) are a *ledger* that must
   survive process restarts, which is the SQLite store's job - so this
   module never imports the store: the daily budget arrives as an injected
   ``budget_ok(provider, model, est_tokens) -> bool`` callable.  That
   decoupling is deliberate; do not import ``tuned.data.store`` here.

3. **Free tiers fail constantly.**  Retries with full-jitter backoff live in
   the client; cross-provider failover and a per-ref circuit breaker live in
   the ``Router``.  A provider that is down must stop costing us latency
   within a few calls, and family separation (a judge may not share a model
   family with the generator it is grading) is enforced per CALL here -
   ``config.py``'s check is only static feasibility.

Everything time-related (``clock``, ``sleeper``) and every transport is
injectable so the whole layer is testable offline with fake clocks and
``httpx.MockTransport``; the test suite never sleeps and never dials out.
"""

import asyncio
import math
import os
import random
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from tuned.data.config import BuildConfig, ModelCfg, ModelRef, ProviderCfg

_CHAT_PATH = "chat/completions"
_BACKOFF_BASE_S = 1.0
_BACKOFF_CAP_S = 60.0
# TOTAL seconds one call may spend parked in backoff/Retry-After sleeps.  A
# per-attempt cap is not enough: six attempts each honouring "Retry-After: 300"
# would park a single call for 25 minutes, and a Router failover pass over
# three refs for over an hour.  Budgeting the whole call instead means a
# provider that wants a long wait costs us one attempt, then the error comes
# back *retryable* and the Router moves to the next provider - which is the
# entire reason we route across providers.  Pure full-jitter backoff at
# max_retries=6 tops out near 31s, so this only ever bites on Retry-After.
_MAX_RETRY_SLEEP_S = 120.0
# Wall-clock ceiling for ONE ChatClient.complete, counting time spent INSIDE
# attempts as well as between them.  The sleep budget alone cannot bound a
# provider that accepts the connection and then hangs: six attempts each
# burning the full HTTP timeout is ~12 minutes on one ref and ~36 on a
# three-ref failover pass.  Checked before every attempt and before every
# sleep, so a hanging provider is abandoned and the Router moves on.
_CALL_DEADLINE_S = 300.0
_DEFAULT_TIMEOUT_S = 120.0

# How a non-429 4xx is classified.  400/413/422 are PAYLOAD-shaped: MOST of
# the time the same request would fail at every provider, so the call aborts
# and the bug surfaces rather than touring the pool.  Everything else in the
# 4xx range (401/403/404, and unlisted codes such as 402 payment-required) is
# about our standing with THIS provider, so the Router fails over.  Defaulting
# the unlisted codes to "fail over" keeps a weeks-long build running; a genuine
# payload bug hiding in one of them still surfaces, as "all N eligible model(s)
# failed".
#
# "MOST of the time" is doing real work in that sentence, and getting it wrong
# cost a row: a 400 carrying context_length_exceeded is a fact about ONE
# model's window, and the next, larger ref in the failover list serves the
# identical request.  Those are detected below (_CONTEXT_OVERFLOW_MARKERS) and
# failed over instead of aborted.  A safety-filter 400 is per-provider in the
# same way and is NOT detected - it is not distinguishable from a payload bug
# by status alone, and inventing a marker list for refusal prose would misfire
# on genuine bugs.  It costs the task attempts, never the row.
_ABORT_STATUSES = frozenset({400, 413, 422})

# Lower-cased substrings that identify a 400/413/422 as "this prompt does not
# fit THIS model's context window".  Drawn from the OpenAI error code every
# compatible server copies plus the prose the shipped providers actually
# return.  Matched against the response body, so a body that merely mentions
# tokens is not enough.
_CONTEXT_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "context length exceeded",
    "maximum context length",
    "context window",
    "prompt is too long",
    "input is too long",
    "input length exceeds",
    "too many tokens",
    "reduce the length",
)


def looks_like_context_overflow(body: str) -> bool:
    """True when a 4xx body says the prompt was too long for THAT model.

    Pure and public because it is the difference between failing over to a
    larger model and closing out a row that nothing was ever wrong with.
    """
    lowered = (body or "").lower()
    return any(marker in lowered for marker in _CONTEXT_OVERFLOW_MARKERS)


# --- .env loading -----------------------------------------------------------


def load_dotenv_keys(path: Path | None = None) -> int:
    """Seed os.environ from the repo-root .env (API keys).  Never overrides.

    Returns the number of keys NEWLY set - a key already present in the
    environment wins and is not counted, so an explicit
    ``GROQ_API_KEY=... python -m ...`` always beats the file.
    """
    if path is None:
        path = Path(__file__).resolve().parents[3] / ".env"
    path = Path(path)
    if not path.is_file():
        return 0
    count = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not key or key in os.environ:
            continue
        os.environ.setdefault(key, value)
        count += 1
    return count


# --- errors and wire types --------------------------------------------------


class ProviderError(RuntimeError):
    """A call failed.  Four outcomes the Router must be able to tell apart:

    * ``retryable``        - 429/5xx/transport/timeout.  Try again, here or
      somewhere else; the request itself was fine.
    * ``provider_dead``    - 401/403/404 and friends.  THIS provider cannot
      serve the call (revoked key, retired preview model) but the payload is
      fine, so the Router marks the ref failed and fails over.  Not retryable
      *at this ref* - retrying the same provider would just fail identically.
    * ``context_exceeded`` - a 4xx whose body says the prompt is longer than
      THIS model's window.  Also a per-provider fact, but not a fault: the
      Router fails over to a larger model WITHOUT charging the breaker,
      because nothing is wrong with the provider it just left.
    * none of the above    - 400/422/413 with no context marker.  Our payload
      is malformed, so it is malformed everywhere; the call aborts
      immediately rather than making the same bad request at every provider
      and hiding the bug.

    ``skipped`` is set only on the "no eligible model" error and carries the
    REASONS refs were passed over ({"missing-key"}, {"cooling"}, ...).  It
    exists because "nothing ran" has two very different meanings to a worker:
    a missing key is a fleet-configuration fact that the row survives, while
    "every family was excluded" is a fact about the row itself.  Callers that
    decide a task's state on ``retryable`` alone cannot tell them apart, and
    the difference is whether a wave is recoverable.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        provider: str | None = None,
        model: str | None = None,
        retryable: bool = False,
        provider_dead: bool = False,
        context_exceeded: bool = False,
        skipped: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__(message)
        self.status = status
        self.provider = provider
        self.model = model
        self.retryable = retryable
        self.provider_dead = provider_dead
        self.context_exceeded = context_exceeded
        self.skipped = frozenset(skipped)


@dataclass(frozen=True)
class ChatRequest:
    messages: tuple[dict, ...]
    ref: ModelRef
    params: dict = field(default_factory=dict)
    max_tokens: int | None = None


@dataclass
class ChatResponse:
    text: str
    reasoning: str | None
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    latency_ms: int
    status: int
    raw: dict


# --- quirks -----------------------------------------------------------------


def _default_request_hook(payload: dict, model: ModelCfg) -> dict:
    return payload


def _default_response_hook(data: dict) -> tuple[str, str | None]:
    """OpenAI shape.  ``content`` may legitimately be null on reasoning models."""
    choices = data.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    text = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    return text, reasoning


def _default_retry_after(response: object) -> float | None:
    """Parse ``Retry-After: <seconds>``.  HTTP-date form is ignored (rare here)."""
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    if not value:
        return None
    try:
        seconds = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


@dataclass(frozen=True)
class Quirk:
    request_hook: Callable[[dict, ModelCfg], dict]
    response_hook: Callable[[dict], tuple[str, str | None]]
    retry_after: Callable[[object], float | None]


def _cerebras_request_hook(payload: dict, model: ModelCfg) -> dict:
    """Cerebras 400s on max_tokens above the model's own output ceiling."""
    max_output = model.limits.get("max_output")
    requested = payload.get("max_tokens")
    if max_output is None or requested is None or requested <= max_output:
        return payload
    clamped = dict(payload)
    clamped["max_tokens"] = max_output
    return clamped


DEFAULT_QUIRK = Quirk(
    request_hook=_default_request_hook,
    response_hook=_default_response_hook,
    retry_after=_default_retry_after,
)

# Named entries for every provider in the config even where behaviour is the
# default: the YAML names must resolve, and a provider that starts deviating
# later gets its hook edited here rather than a config change.
QUIRKS: dict[str, Quirk] = {
    "default": DEFAULT_QUIRK,
    "cerebras": Quirk(
        request_hook=_cerebras_request_hook,
        response_hook=_default_response_hook,
        retry_after=_default_retry_after,
    ),
    "groq": DEFAULT_QUIRK,
    "mistral": DEFAULT_QUIRK,
    "openrouter": DEFAULT_QUIRK,
}


def resolve_quirks(names: Sequence[str]) -> Quirk:
    """Compose named quirks in order.

    Request hooks chain (each transforms the payload the previous produced).
    ``response_hook``/``retry_after`` are single-valued, so the LAST name that
    actually overrides the default wins - "most specific last", matching how
    the request chain reads.
    """
    unknown = [n for n in names if n not in QUIRKS]
    if unknown:
        raise KeyError(
            f"unknown quirk(s) {unknown} in provider config; known quirks: {sorted(QUIRKS)}"
        )
    quirks = [QUIRKS[n] for n in names]
    if not quirks:
        return DEFAULT_QUIRK
    if len(quirks) == 1:
        return quirks[0]

    def chained_request_hook(payload: dict, model: ModelCfg) -> dict:
        for quirk in quirks:
            payload = quirk.request_hook(payload, model)
        return payload

    response_hook = next(
        (q.response_hook for q in reversed(quirks) if q.response_hook is not _default_response_hook),
        _default_response_hook,
    )
    retry_after = next(
        (q.retry_after for q in reversed(quirks) if q.retry_after is not _default_retry_after),
        _default_retry_after,
    )
    return Quirk(
        request_hook=chained_request_hook,
        response_hook=response_hook,
        retry_after=retry_after,
    )


# --- rate limiting ----------------------------------------------------------


class TokenBucket:
    """Continuous-refill rpm + tpm limiter for ONE (provider, model).

    Both buckets start full and refill at ``limit/60`` per second, capacity
    ``limit``.  ``None`` disables that dimension entirely.  ``acquire`` holds
    an ``asyncio.Lock`` across its wait, which makes waiters FIFO and stops a
    thundering herd from all deciding "there is room" at the same instant;
    the cost is head-of-line blocking behind one very large request.

    Claims are never refunded: if the call downstream fails we still burned
    the provider's quota, so the bucket errs toward under-issuing.
    """

    def __init__(
        self,
        rpm: int | None,
        tpm: int | None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], object] = asyncio.sleep,
    ) -> None:
        # 0 is never a sane limit and both readings of it are wrong: treating
        # it as "unlimited" silently removes a limit the config asked for,
        # treating it as "no capacity" wedges the bucket forever.  It is
        # almost always a typo or a missing YAML value, so say so loudly.
        for name, value in (("rpm", rpm), ("tpm", tpm)):
            if value is not None and value <= 0:
                raise ValueError(
                    f"TokenBucket {name} must be a positive int or None "
                    f"(None means that dimension is unlimited), got {value!r}"
                )
        self.rpm = rpm
        self.tpm = tpm
        self._clock = clock
        self._sleeper = sleeper
        self._req_cap = float(self.rpm or 0)
        self._tok_cap = float(self.tpm or 0)
        self._req = self._req_cap
        self._tok = self._tok_cap
        self._updated = clock()
        self._lock = asyncio.Lock()

    # rate per second for each dimension
    @property
    def _req_rate(self) -> float:
        return (self.rpm or 0) / 60.0

    @property
    def _tok_rate(self) -> float:
        return (self.tpm or 0) / 60.0

    def _need_tokens(self, est_tokens: int) -> float:
        """Tokens to charge, CLAMPED to capacity.

        Without the clamp an ``est_tokens`` above tpm would wait forever: the
        bucket can never hold more than its capacity, so the wait condition
        would never become true.  Clamping degrades to "wait for a full
        bucket, then go", which is the only sane behaviour - the request is
        going to blow the per-minute limit either way and the provider's own
        429 handling (with backoff) takes it from there.
        """
        if not self.tpm or est_tokens <= 0:
            return 0.0
        return min(float(est_tokens), self._tok_cap)

    def _peek(self, now: float) -> tuple[float, float]:
        elapsed = max(0.0, now - self._updated)
        req = min(self._req_cap, self._req + elapsed * self._req_rate)
        tok = min(self._tok_cap, self._tok + elapsed * self._tok_rate)
        return req, tok

    def _refill(self, now: float) -> None:
        self._req, self._tok = self._peek(now)
        self._updated = now

    def next_wait(self, est_tokens: int = 0) -> float:
        """Seconds until ``acquire(est_tokens)`` would proceed; 0.0 if now.  Pure."""
        req, tok = self._peek(self._clock())
        wait = 0.0
        if self.rpm and req < 1.0:
            wait = max(wait, (1.0 - req) / self._req_rate)
        need = self._need_tokens(est_tokens)
        if need and tok < need:
            wait = max(wait, (need - tok) / self._tok_rate)
        return wait

    async def acquire(self, est_tokens: int = 0) -> None:
        if not self.rpm and not self.tpm:
            return
        async with self._lock:
            while True:
                self._refill(self._clock())
                need = self._need_tokens(est_tokens)
                if (not self.rpm or self._req >= 1.0) and (not self.tpm or self._tok >= need):
                    if self.rpm:
                        self._req -= 1.0
                    if self.tpm:
                        self._tok -= need
                    return
                # Floor guards against a float-rounding zero spinning the loop.
                await self._sleeper(max(self.next_wait(est_tokens), 1e-6))


# --- the one client ---------------------------------------------------------


def _snippet(text: str, limit: int) -> str:
    """Collapse whitespace and truncate - keeps error detail and table rows
    to a single line.  Used by the client's error paths and by the CLI."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


class ChatClient:
    """OpenAI-compatible chat client for ONE (provider, model), with retries.

    The API key is read at CALL time, not construction: the CLI and the
    Router both want to build clients for models whose key may be absent and
    report that as a row/skip rather than an import-time explosion.
    """

    def __init__(
        self,
        provider: ProviderCfg,
        model: ModelCfg,
        *,
        transport: object | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], object] = asyncio.sleep,
        rng: random.Random | None = None,
        max_retries: int = 6,
        timeout: float = _DEFAULT_TIMEOUT_S,
        max_retry_sleep_s: float = _MAX_RETRY_SLEEP_S,
        call_deadline_s: float = _CALL_DEADLINE_S,
    ) -> None:
        self.provider = provider
        self.model = model
        self.quirk = resolve_quirks(provider.quirks)
        self.max_retries = max(1, max_retries)
        self.max_retry_sleep_s = max_retry_sleep_s
        self.call_deadline_s = call_deadline_s
        self._clock = clock
        self._sleeper = sleeper
        self._rng = rng if rng is not None else random.Random()
        self._client = httpx.AsyncClient(
            base_url=provider.base_url, transport=transport, timeout=timeout
        )

    def _error(
        self,
        message: str,
        *,
        status: int | None,
        retryable: bool,
        provider_dead: bool = False,
        context_exceeded: bool = False,
    ) -> ProviderError:
        return ProviderError(
            f"{self.provider.name}/{self.model.id}: {message}",
            status=status,
            provider=self.provider.name,
            model=self.model.id,
            retryable=retryable,
            provider_dead=provider_dead,
            context_exceeded=context_exceeded,
        )

    def build_payload(self, req: ChatRequest) -> dict:
        """Merge model defaults with per-request params (request wins) + quirks."""
        payload: dict = {
            "model": self.model.id,
            "messages": [dict(m) for m in req.messages],
            **dict(self.model.params or {}),
            **dict(req.params or {}),
        }
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        return self.quirk.request_hook(payload, self.model)

    def _backoff(self, attempt: int) -> float:
        """Exponential base 1s capped at 60s with FULL jitter."""
        ceiling = min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (2**attempt))
        return self._rng.uniform(0.0, ceiling)

    def _check_deadline(
        self, call_started: float, attempt: int, last_status: int | None, last_detail: str
    ) -> None:
        """Raise if this call has already outlived its wall-clock ceiling."""
        elapsed = self._clock() - call_started
        if elapsed >= self.call_deadline_s:
            raise self._error(
                f"call deadline: {elapsed:.0f}s elapsed exceeds the "
                f"{self.call_deadline_s:.0f}s ceiling after {attempt} attempt(s) "
                f"- failing over instead of waiting; last: {last_detail}",
                status=last_status,
                retryable=True,
            )

    async def complete(
        self,
        req: ChatRequest,
        *,
        before_attempt: Callable[[], Awaitable[None]] | None = None,
        on_attempt: Callable[[int | None, dict | None], None] | None = None,
    ) -> ChatResponse:
        """Send ``req``, retrying transient failures within two ceilings.

        ``before_attempt`` is awaited before every attempt AFTER the first.
        The Router uses it to charge the rate-limit bucket for retries: it
        acquires a slot for attempt 0 itself, but every retry is another real
        request against the same per-minute quota, and without this hook a
        flapping provider silently overruns its own rpm limit.

        ``on_attempt`` is called AFTER every attempt with
        ``(status, usage_or_None)`` - the HTTP status this attempt actually
        got (None for a transport failure that never reached the provider)
        and the response's usage block when there was one.  It exists because
        the retries in here are invisible from the outside: five 429s
        followed by a success look like one call to the caller, so a daily
        ledger fed only by the return value under-counts requests exactly
        when the provider is under the most pressure.  Never swallowed - a
        raising callback is a broken ledger and must surface.
        """
        key = os.environ.get(self.provider.api_key_env)
        if not key:
            raise self._error(
                f"API key env {self.provider.api_key_env} is not set", status=None, retryable=False
            )
        payload = self.build_payload(req)
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        last_status: int | None = None
        last_detail = "no attempt made"
        slept_total = 0.0
        call_started = self._clock()
        for attempt in range(self.max_retries):
            if attempt:
                # Guards what the sleep budget cannot see: time burnt INSIDE
                # attempts (an HTTP timeout) or inside before_attempt (a
                # bucket wait) rather than between them.  Checked on BOTH
                # sides of before_attempt - once so we do not start a long
                # wait we have no time for, once because the wait itself may
                # have pushed us over.
                self._check_deadline(call_started, attempt, last_status, last_detail)
                if before_attempt is not None:
                    await before_attempt()
                    self._check_deadline(call_started, attempt, last_status, last_detail)
            retry_after: float | None = None
            started = self._clock()
            try:
                response = await self._client.post(_CHAT_PATH, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last_status = None
                last_detail = f"transport error {type(exc).__name__}: {exc}"
                if on_attempt is not None:
                    on_attempt(None, None)
            else:
                status = response.status_code
                if 200 <= status < 300:
                    latency_ms = int(max(0.0, self._clock() - started) * 1000)
                    try:
                        data = response.json()
                    except ValueError as exc:
                        last_status = status
                        last_detail = f"malformed JSON body: {exc}"
                        if on_attempt is not None:
                            on_attempt(status, None)
                    else:
                        if on_attempt is not None:
                            on_attempt(status, data.get("usage") or None)
                        return self._to_response(data, status=status, latency_ms=latency_ms)
                else:
                    last_status = status
                    last_detail = _snippet(response.text, 200) or f"HTTP {status}"
                    if on_attempt is not None:
                        on_attempt(status, None)
                    if 400 <= status < 500 and status != 429:
                        dead = status not in _ABORT_STATUSES
                        # A 400 that says "too long for this window" is about
                        # THIS model, not about the payload: the larger ref
                        # next in the list takes the identical request.
                        overflow = not dead and looks_like_context_overflow(last_detail)
                        if dead:
                            kind = "provider unusable, failing over"
                        elif overflow:
                            kind = "prompt exceeds this context window, failing over"
                        else:
                            kind = "bad request, aborting the call"
                        raise self._error(
                            f"HTTP {status} ({kind}): {last_detail}",
                            status=status,
                            retryable=False,
                            provider_dead=dead,
                            context_exceeded=overflow,
                        )
                    retry_after = self.quirk.retry_after(response)

            if attempt == self.max_retries - 1:
                break
            delay = max(self._backoff(attempt), retry_after or 0.0)
            elapsed = self._clock() - call_started
            over_sleep = slept_total + delay > self.max_retry_sleep_s
            over_deadline = elapsed + delay > self.call_deadline_s
            if over_sleep or over_deadline:
                asked = f", Retry-After {retry_after:.0f}s" if retry_after else ""
                bound = (
                    f"sleep budget ({slept_total:.0f}s slept + {delay:.0f}s more "
                    f"> {self.max_retry_sleep_s:.0f}s cap)"
                    if over_sleep
                    else f"call deadline ({elapsed:.0f}s elapsed + {delay:.0f}s more "
                    f"> {self.call_deadline_s:.0f}s ceiling)"
                )
                raise self._error(
                    f"{bound} would be exceeded{asked} - failing over instead of "
                    f"waiting; last: {last_detail}",
                    status=last_status,
                    retryable=True,
                )
            await self._sleeper(delay)
            slept_total += delay

        raise self._error(
            f"exhausted {self.max_retries} attempts; last: {last_detail}",
            status=last_status,
            retryable=True,
        )

    def _to_response(self, data: dict, *, status: int, latency_ms: int) -> ChatResponse:
        text, reasoning = self.quirk.response_hook(data)
        usage = data.get("usage") or {}
        choices = data.get("choices") or []
        finish_reason = choices[0].get("finish_reason") if choices else None
        return ChatResponse(
            text=text,
            reasoning=reasoning,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            status=status,
            raw=data,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


# --- routing ----------------------------------------------------------------


@dataclass(frozen=True)
class RoutedModel:
    ref: ModelRef
    provider_cfg: ProviderCfg
    model_cfg: ModelCfg
    client: ChatClient
    bucket: TokenBucket


@dataclass
class _BreakerState:
    consecutive_failures: int = 0
    cooling_until: float = 0.0


# Skip reasons that lift on their own: a caller that got nothing because of
# these may usefully try again shortly.  "missing-key" and "family-excluded"
# are structural - they will still hold in a minute.  Public because a worker
# deciding a task's fate has to classify the reasons the same way this module
# does; a private copy in the workers would drift.
TRANSIENT_SKIPS = frozenset({"cooling", "over-budget"})
_TRANSIENT_SKIPS = TRANSIENT_SKIPS  # legacy alias, kept for in-module reads

# --- context estimation -----------------------------------------------------
#
# Everything here is an ESTIMATE compared against a hard ceiling, so every
# constant errs the same way: over-count rather than under-count.  Under-
# counting routes an over-long prompt at a small model, which is a 400.

# Characters per token.  Latin legal English runs ~4 chars/token under the
# BPE vocabularies these providers use.  Devanagari and the other Indic
# scripts run 1-2, because those vocabularies barely model them - so a
# chars/4 estimate under-counts an Indic passage by 2-4x, on exactly the
# corpus this build is made of.
CHARS_PER_TOKEN_LATIN = 4.0
CHARS_PER_TOKEN_INDIC = 1.5

# Per-message chat-template overhead.  Every OpenAI-compatible server wraps
# each turn in role markers and delimiters that the caller never sees but the
# context window is charged for; ChatML-family templates spend roughly this
# many tokens a turn.
TEMPLATE_TOKENS_PER_MESSAGE = 8

# Multiplied into the estimate before it is compared with a model's
# max_context.  A cap that merely EQUALS the estimate is not headroom: the
# estimate is a heuristic over characters, the provider counts real tokens,
# and the two disagree by a few percent on ordinary English and by more on
# mixed script.  This is the margin that turns "probably fits" into "routes
# there".
CONTEXT_SAFETY_MARGIN = 1.25

# Codepoint ranges of the Indic scripts this corpus contains (Devanagari
# through Sinhala, plus Devanagari Extended).  A range test rather than a
# per-script list: they all tokenize badly for the same reason.
_INDIC_RANGES = ((0x0900, 0x0DFF), (0xA8E0, 0xA8FF))


def _is_indic(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _INDIC_RANGES)


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for one string, script-aware.

    NOT the same currency as the gates' length band, which is chars/4 by
    definition and is calibrated that way across the whole build.  This one
    exists solely to answer "will this fit in that window", where being wrong
    low is a 400 and being wrong high is a failover.
    """
    if not text:
        return 0
    indic = sum(1 for ch in text if _is_indic(ch))
    latin = len(text) - indic
    return int(latin / CHARS_PER_TOKEN_LATIN + indic / CHARS_PER_TOKEN_INDIC) + 1


def context_estimate(messages: Sequence[Mapping]) -> int:
    """Conservative token estimate for a rendered message list."""
    return sum(
        estimate_tokens(m.get("content") or "") + TEMPLATE_TOKENS_PER_MESSAGE for m in messages
    )


def undersized_families(cfg: BuildConfig, role: str, needed_tokens: int) -> frozenset[str]:
    """Families that cannot hold ``needed_tokens`` in ANY of their ``role`` models.

    Context length is not something ``eligible`` can filter on - the Router
    routes by role and family, and a model's ``max_context`` is a property of
    the PROMPT, not of the pool.  Turning "too small for this prompt" into a
    family exclusion is what lets a caller express it with the filter the
    Router does have.

    ``needed_tokens`` is scaled by CONTEXT_SAFETY_MARGIN before the
    comparison: the caller's number is an estimate over characters and the
    provider enforces a count of real tokens, so a model whose cap merely
    equals the estimate is a coin flip on a 400.

    A family is excluded only when EVERY one of its models in that role is
    too small, so a mixed-size family keeps its large model and the
    preference order still decides between them.  A model that declares no
    ``max_context`` is assumed to fit: an absent limit is unknown, and
    guessing "too small" would silently remove a working provider.

    Callers on both sides of a generation need this - the generator because
    an over-long prompt at an 8k model is a 400, the judge because its prompt
    carries the same materials PLUS the candidate and is the longest in the
    pipeline.
    """
    required = int(max(0, needed_tokens) * CONTEXT_SAFETY_MARGIN)
    fits_by_family: dict[str, bool] = {}
    for ref in cfg.routing_refs(role):
        _, model = cfg.model_for(ref)
        cap = model.limits.get("max_context")
        fits = cap is None or int(cap) >= required
        fits_by_family[model.family] = fits_by_family.get(model.family, False) or fits
    return frozenset(family for family, fits in fits_by_family.items() if not fits)


# --- startup preflight ------------------------------------------------------
#
# Both checks below exist for the same reason: the two facts they test are
# knowable in one second at startup and cost a wave of half-paid rows to
# discover at runtime.


def unkeyed_roles(cfg: BuildConfig, roles: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """role -> the api_key_env names it needs, for roles with NO usable model.

    A role is fine if ONE of its refs has a key; the rest is failover.  A role
    with none of them cannot make a single call, and a fleet started in that
    state claims tasks, fails to route every one of them, and reports a
    routing error per row instead of the one line the operator needed.
    """
    missing: dict[str, tuple[str, ...]] = {}
    for role in roles:
        envs: list[str] = []
        usable = False
        for ref in cfg.routing_refs(role):
            provider, _ = cfg.model_for(ref)
            if os.environ.get(provider.api_key_env):
                usable = True
                break
            if provider.api_key_env not in envs:
                envs.append(provider.api_key_env)
        if not usable:
            missing[role] = tuple(envs)
    return missing


# The judge's own reply allowance, in tokens.  judge.py owns the runtime
# value (JUDGE_MAX_TOKENS) and passes it in; this default is what lets the
# GENERATION worker run the same preflight, because a generator that fills a
# queue no judge pool can drain is spending money on rows that will park.
DEFAULT_JUDGE_REPLY_TOKENS = 1024


@dataclass(frozen=True)
class PoolGap:
    """A (generator family, judge slot) pair the pool cannot serve at size."""

    role: str  # "judge" | "tiebreak"
    generator_family: str
    slot: str  # "a" | "b" | "tiebreak"
    detail: str
    # A judge-slot gap parks a row that has ALREADY paid for the other slot,
    # so it is fatal.  A tiebreak gap has a defined, unpaid fallback
    # (judge.py decides on the two judges and rejects the unresolved
    # disagreement), so it is reported and survivable.
    fatal: bool


def max_judge_prompt_tokens(
    cfg: BuildConfig, *, judge_reply_tokens: int = DEFAULT_JUDGE_REPLY_TOKENS
) -> int:
    """The largest judge call the pipeline can produce, in routing currency.

    The judge prompt is the row's own material plus the candidate, and the
    length band caps exactly that sum (``total_max``) - a row above it fails
    the length gate and never reaches a judge.  Add the judge's reply
    allowance, because the window has to hold both.
    """
    return int(cfg.build.length_band.total_max) + int(judge_reply_tokens)


def _first_family(
    cfg: BuildConfig, role: str, exclude: set[str], undersized: frozenset[str]
) -> str | None:
    """The family Router.pick would land on, walking the real preference order."""
    for ref in cfg.routing_refs(role):
        _, model = cfg.model_for(ref)
        if model.family in exclude or model.family in undersized:
            continue
        return model.family
    return None


def pool_gaps(
    cfg: BuildConfig, *, judge_reply_tokens: int = DEFAULT_JUDGE_REPLY_TOKENS
) -> list[PoolGap]:
    """Every judge/tiebreak slot the pool cannot fill for the LONGEST row.

    Family separation and context length are both per-call filters, so their
    interaction is invisible in the config and only shows up once a row is
    long enough - by which time the fleet has paid for judge A and parked the
    row waiting for a judge B that does not exist.  This walks the same
    preference order judge.py walks, at the largest prompt the length band
    permits, and says so before anything is claimed.

    Deliberately NOT a fallback: reusing a family when the pool runs out
    would put a model's own family in front of its own prose, which is the
    invariant the separation exists to protect.  The fix is one more model in
    the config; this is how the operator learns that in one second.
    """
    needed = max_judge_prompt_tokens(cfg, judge_reply_tokens=judge_reply_tokens)
    too_small_judge = undersized_families(cfg, "judge", needed)
    too_small_tiebreak = undersized_families(cfg, "tiebreak", needed)
    gaps: list[PoolGap] = []
    seen: set[tuple[str, str]] = set()
    for gen_ref in cfg.routing_refs("generator"):
        _, gen_model = cfg.model_for(gen_ref)
        gen_family = gen_model.family
        if (gen_family, "seen") in seen:
            continue
        seen.add((gen_family, "seen"))
        chosen: list[str] = []
        blocked = False
        for slot in ("a", "b"):
            family = _first_family(
                cfg, "judge", {gen_family, *chosen}, too_small_judge
            )
            if family is None:
                gaps.append(
                    PoolGap(
                        role="judge",
                        generator_family=gen_family,
                        slot=slot,
                        detail=(
                            f"a {gen_family} generation of {needed} tokens has no judge "
                            f"for slot {slot}: the judge pool is "
                            f"{sorted({cfg.model_for(r)[1].family for r in cfg.routing_refs('judge')})}, "
                            f"minus the generator's own {gen_family!r}, minus "
                            f"{sorted(too_small_judge)} on context length, minus "
                            f"{chosen} already used. Add one judge in a further family "
                            f"with max_context >= {int(needed * CONTEXT_SAFETY_MARGIN)}"
                        ),
                        fatal=True,
                    )
                )
                blocked = True
                break
            chosen.append(family)
        if blocked:
            continue
        tiebreak = _first_family(
            cfg, "tiebreak", {gen_family, *chosen}, too_small_tiebreak
        )
        if tiebreak is None:
            gaps.append(
                PoolGap(
                    role="tiebreak",
                    generator_family=gen_family,
                    slot="tiebreak",
                    detail=(
                        f"a {gen_family} generation of {needed} tokens judged by "
                        f"{chosen} has no third family left to break a tie; judge.py "
                        f"decides on the two judges and rejects the unresolved "
                        f"disagreement (run_event "
                        f"tiebreak_unroutable_two_judge_decision)"
                    ),
                    fatal=False,
                )
            )
    return gaps


class Router:
    """Role -> model selection with call-time family separation and a breaker.

    ``pick`` walks ``cfg.routing_refs(role)`` in configured (preference) order
    and returns the first ref that is eligible: family not excluded BY THIS
    CALL (the judge of a gpt-oss generation may not itself be gpt-oss - a
    static config check cannot know which generator produced the row), not
    cooling behind the circuit breaker, key present in the environment, and
    within the injected daily budget.

    NOTE that failover is only as wide as the role's list.  ``routing.probe``
    is a SINGLE-ref role in the shipped config, so ``complete("probe", ...)``
    has nowhere to fail over to: if that one ref is cooling, out of budget or
    simply broken, the call raises.  The Router deliberately does not loop or
    wait for a cooldown to expire - callers that need probe to survive a blip
    must retry at their own level, where they can decide what a stall costs.
    """

    def __init__(
        self,
        cfg: BuildConfig,
        *,
        budget_ok: Callable[[str, str, int], bool] = lambda provider, model, tokens: True,
        clock: Callable[[], float] = time.monotonic,
        cooldown_s: float = 300.0,
        breaker_threshold: int = 4,
        transport: object | None = None,
        sleeper: Callable[[float], object] = asyncio.sleep,
        rng: random.Random | None = None,
        max_retries: int = 6,
        timeout: float = _DEFAULT_TIMEOUT_S,
        max_retry_sleep_s: float = _MAX_RETRY_SLEEP_S,
        call_deadline_s: float = _CALL_DEADLINE_S,
        client_factory: Callable[[ProviderCfg, ModelCfg], ChatClient] | None = None,
    ) -> None:
        self.cfg = cfg
        self.budget_ok = budget_ok
        self.cooldown_s = cooldown_s
        self.breaker_threshold = breaker_threshold
        self._clock = clock
        self._sleeper = sleeper
        self._rng = rng
        self._transport = transport
        self._max_retries = max_retries
        self._timeout = timeout
        self._max_retry_sleep_s = max_retry_sleep_s
        self._call_deadline_s = call_deadline_s
        self._client_factory = client_factory
        self._routed: dict[ModelRef, RoutedModel] = {}
        self._breakers: dict[ModelRef, _BreakerState] = {}

    # -- construction (lazy: never build a client for a model we never call)

    def _make_client(self, provider: ProviderCfg, model: ModelCfg) -> ChatClient:
        if self._client_factory is not None:
            return self._client_factory(provider, model)
        return ChatClient(
            provider,
            model,
            transport=self._transport,
            clock=self._clock,
            sleeper=self._sleeper,
            rng=self._rng,
            max_retries=self._max_retries,
            timeout=self._timeout,
            max_retry_sleep_s=self._max_retry_sleep_s,
            call_deadline_s=self._call_deadline_s,
        )

    def routed(self, ref: ModelRef) -> RoutedModel:
        existing = self._routed.get(ref)
        if existing is not None:
            return existing
        provider, model = self.cfg.model_for(ref)
        routed = RoutedModel(
            ref=ref,
            provider_cfg=provider,
            model_cfg=model,
            client=self._make_client(provider, model),
            bucket=TokenBucket(
                model.limits.get("rpm"),
                model.limits.get("tpm"),
                clock=self._clock,
                sleeper=self._sleeper,
            ),
        )
        self._routed[ref] = routed
        return routed

    # -- circuit breaker

    def _breaker(self, ref: ModelRef) -> _BreakerState:
        return self._breakers.setdefault(ref, _BreakerState())

    def is_cooling(self, ref: ModelRef) -> bool:
        return self._clock() < self._breaker(ref).cooling_until

    def report_success(self, ref: ModelRef) -> None:
        self._breaker(ref).consecutive_failures = 0

    def report_failure(self, ref: ModelRef) -> None:
        state = self._breaker(ref)
        state.consecutive_failures += 1
        if state.consecutive_failures >= self.breaker_threshold:
            state.cooling_until = self._clock() + self.cooldown_s
            state.consecutive_failures = 0

    # -- selection

    def eligible(
        self,
        role: str,
        *,
        est_tokens: int = 0,
        exclude_families: frozenset[str] = frozenset(),
        skipped: set[str] | None = None,
    ) -> Iterator[RoutedModel]:
        """Yield eligible refs in preference order, re-checking state each step.

        Re-checking matters for ``complete``: a ref that trips the breaker
        part-way through a failover pass must not be re-offered later in the
        same pass.

        ``skipped``, if given, collects the REASONS refs were passed over.
        That is what lets a caller tell "nothing is eligible because the whole
        pool is cooling" (transient - try again in a minute) from "nothing is
        eligible because no key is set" (structural - trying again is futile).
        Because this is a generator, a caller that stops at the first hit sees
        only the reasons encountered before it.
        """
        for ref in self.cfg.routing_refs(role):
            provider, model = self.cfg.model_for(ref)
            reason: str | None = None
            if model.family in exclude_families:
                reason = "family-excluded"
            elif self.is_cooling(ref):
                reason = "cooling"
            elif not os.environ.get(provider.api_key_env):
                reason = "missing-key"
            elif not self.budget_ok(ref.provider, ref.model, est_tokens):
                reason = "over-budget"
            if reason is not None:
                if skipped is not None:
                    skipped.add(reason)
                continue
            yield self.routed(ref)

    def pick(
        self,
        role: str,
        *,
        est_tokens: int = 0,
        exclude_families: frozenset[str] = frozenset(),
        skipped: set[str] | None = None,
    ) -> RoutedModel | None:
        return next(
            self.eligible(
                role,
                est_tokens=est_tokens,
                exclude_families=exclude_families,
                skipped=skipped,
            ),
            None,
        )

    async def complete(
        self,
        role: str,
        req_messages: Sequence[Mapping],
        *,
        params: dict | None = None,
        params_for_ref: Callable[[ModelRef, ModelCfg], Mapping] | None = None,
        max_tokens: int | None = None,
        est_tokens: int = 0,
        exclude_families: frozenset[str] = frozenset(),
        on_attempt: Callable[[ModelRef, int | None, dict | None], None] | None = None,
    ) -> tuple[ModelRef, ChatResponse]:
        """One pass down the role's preference list; returns (ref, response).

        Three failure shapes, three behaviours:

        * *retryable* (429/5xx/transport, after the client's own in-provider
          retries) - ``report_failure`` and move to the next eligible ref.
        * *provider_dead* (401/403/404 - revoked key, retired preview model)
          - also ``report_failure`` and move on.  These are PER-PROVIDER
          facts: qwen3.6-27b is a preview model that may 404 out of existence
          any day, and that must not take the whole judge role down.
        * anything else non-retryable (400/422/413 - our payload) - raised
          straight through, because the same request would fail identically
          at every other provider and failing over would only hide the bug.

        ``params_for_ref``, when given, is called with the ref about to be
        used and its ModelCfg, and its return value is merged OVER ``params``
        for that attempt (key by key - the rest of ``params`` still ships).
        Failover is what makes this necessary: a parameter
        that is valid for the first ref (``reasoning_effort`` on a gpt-oss
        model) is an unknown field at the next one, and an unknown field is a
        400 - which this method classifies as OUR payload being broken and
        raises straight through instead of failing over.  A caller that sends
        model-specific parameters must therefore choose them per ref, not per
        call.

        ``on_attempt`` is called after EVERY HTTP attempt, including the
        client's internal retries and every failover, with
        ``(ref, status, usage_or_None)``.  It is how a caller keeps a daily
        request/token ledger that matches what the providers actually saw.
        """
        messages = tuple(dict(m) for m in req_messages)
        last_error: ProviderError | None = None
        attempted = 0
        skipped: set[str] = set()
        for routed in self.eligible(
            role, est_tokens=est_tokens, exclude_families=exclude_families, skipped=skipped
        ):
            attempted += 1
            # The hook OVERRIDES the call-wide params key by key; it does not
            # replace the dict.  Dropping the whole of `params` when a hook is
            # present is how a caller ends up believing it sent a temperature
            # it never sent - the hook's job is per-model deviation, not
            # re-stating everything the call already asked for.
            per_ref = dict(params or {})
            if params_for_ref is not None:
                per_ref.update(params_for_ref(routed.ref, routed.model_cfg))
            req = ChatRequest(
                messages=messages,
                ref=routed.ref,
                params=per_ref,
                max_tokens=max_tokens,
            )
            attempt_hook = (
                None
                if on_attempt is None
                else (lambda status, usage, ref=routed.ref: on_attempt(ref, status, usage))
            )
            await routed.bucket.acquire(est_tokens)
            try:
                response = await routed.client.complete(
                    req,
                    before_attempt=lambda bucket=routed.bucket: bucket.acquire(0),
                    on_attempt=attempt_hook,
                )
            except ProviderError as exc:
                if exc.context_exceeded:
                    # This model's window, not this provider's health and not
                    # our payload: move to a larger ref and leave the breaker
                    # alone, or a long row would cool a perfectly good
                    # provider for everybody else.
                    last_error = exc
                    continue
                if not exc.retryable and not exc.provider_dead:
                    raise
                self.report_failure(routed.ref)
                last_error = exc
                continue
            self.report_success(routed.ref)
            return routed.ref, response

        if last_error is not None:
            # If the LAST thing that happened was "too long for that window",
            # the pass ran out of models rather than out of luck: the same
            # prompt tomorrow gets the same answer, so this is not retryable
            # and the caller must treat it as a fact about the row.
            overflow = last_error.context_exceeded
            raise ProviderError(
                f"role {role!r}: all {attempted} eligible model(s) failed; last: {last_error}",
                status=last_error.status,
                provider=last_error.provider,
                model=last_error.model,
                retryable=not overflow,
                context_exceeded=overflow,
            ) from last_error
        # Nothing was even tried.  Cooling and over-budget lift on their own,
        # so the caller may usefully come back; a missing key or a family
        # exclusion will still be true in a minute, so it may not.  The
        # reasons ride along on the error: "no key set" and "this row fits
        # nowhere" are both non-retryable and mean opposite things to a task.
        transient = sorted(skipped & TRANSIENT_SKIPS)
        reasons = ", ".join(sorted(skipped)) if skipped else "role list is empty"
        raise ProviderError(
            f"role {role!r}: no eligible model (skipped: {reasons})",
            retryable=bool(transient),
            skipped=frozenset(skipped),
        )

    async def aclose(self) -> None:
        for routed in self._routed.values():
            await routed.client.aclose()


# --- CLI --------------------------------------------------------------------

CHECK_PROMPT = "Reply with the single word OK."


@dataclass(frozen=True)
class CheckResult:
    ref: str
    key_present: bool
    status: int | None
    text: str
    usage_present: bool
    reasoning_present: bool
    latency_ms: int | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.status is not None and 200 <= self.status < 300


def build_check_request(ref: ModelRef, *, max_tokens: int = 32) -> ChatRequest:
    """The one-shot liveness probe.  Pure, so tests cover it without a network."""
    return ChatRequest(
        messages=({"role": "user", "content": CHECK_PROMPT},),
        ref=ref,
        params={},
        max_tokens=max_tokens,
    )


def format_check_header() -> str:
    return (
        f"{'ref':<34} | {'key':<3} | {'status':<6} | {'text':<32} | "
        f"{'usage':<5} | {'reason':<6} | {'ms':>6}"
    )


def format_check_row(result: CheckResult) -> str:
    """Render one probe outcome as a fixed-width table row (pure)."""
    if result.error is not None:
        body = _snippet(f"FAIL {result.error}", 32)
    else:
        body = _snippet(result.text, 32)
    status = str(result.status) if result.status is not None else "-"
    latency = str(result.latency_ms) if result.latency_ms is not None else "-"
    return (
        f"{_snippet(result.ref, 34):<34} | {('yes' if result.key_present else 'no '):<3} | "
        f"{status:<6} | {body:<32} | {('yes' if result.usage_present else 'no '):<5} | "
        f"{('yes' if result.reasoning_present else 'no '):<6} | {latency:>6}"
    )


def check_refs(cfg: BuildConfig, ref: str | None = None) -> tuple[ModelRef, ...]:
    """Every configured (provider, model), or just the one named by --ref."""
    if ref is not None:
        if "/" not in ref:
            raise ValueError(f"--ref must be 'provider/model-id', got {ref!r}")
        parsed = ModelRef(*ref.split("/", 1))
        cfg.model_for(parsed)  # raises KeyError with a clear message
        return (parsed,)
    return tuple(
        ModelRef(provider.name, model.id)
        for provider in cfg.providers
        for model in provider.models
    )


async def _check_one(cfg: BuildConfig, ref: ModelRef, *, max_tokens: int = 32) -> CheckResult:
    label = f"{ref.provider}/{ref.model}"
    provider, model = cfg.model_for(ref)
    key_present = bool(os.environ.get(provider.api_key_env))
    if not key_present:
        return CheckResult(
            ref=label,
            key_present=False,
            status=None,
            text="",
            usage_present=False,
            reasoning_present=False,
            latency_ms=None,
            error=f"{provider.api_key_env} not set",
        )
    client = ChatClient(provider, model, max_retries=2)
    try:
        response = await client.complete(build_check_request(ref, max_tokens=max_tokens))
    except ProviderError as exc:
        return CheckResult(
            ref=label,
            key_present=True,
            status=exc.status,
            text="",
            usage_present=False,
            reasoning_present=False,
            latency_ms=None,
            error=str(exc),
        )
    except Exception as exc:  # a check must never abort the sweep
        return CheckResult(
            ref=label,
            key_present=True,
            status=None,
            text="",
            usage_present=False,
            reasoning_present=False,
            latency_ms=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        await client.aclose()
    usage = response.raw.get("usage") or {}
    return CheckResult(
        ref=label,
        key_present=True,
        status=response.status,
        text=response.text,
        usage_present=bool(usage),
        reasoning_present=response.reasoning is not None,
        latency_ms=response.latency_ms,
    )


async def run_check(cfg: BuildConfig, refs: Sequence[ModelRef]) -> list[CheckResult]:
    return [await _check_one(cfg, ref) for ref in refs]


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from tuned.data.config import load_build_config

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data_law_v1.yaml")
    parser.add_argument("--check", action="store_true", help="one live call per configured model")
    parser.add_argument("--ref", default=None, help="check only this provider/model ref")
    args = parser.parse_args(argv)

    if not args.check:
        parser.error("nothing to do: pass --check")

    cfg = load_build_config(args.config, allow_unpinned=True)
    loaded = load_dotenv_keys()
    print(f"loaded {loaded} key(s) from .env")
    results = asyncio.run(run_check(cfg, check_refs(cfg, args.ref)))
    print(format_check_header())
    for result in results:
        print(format_check_row(result))
    failed = [r for r in results if not r.ok]
    print(f"{len(results) - len(failed)}/{len(results)} model(s) OK")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
