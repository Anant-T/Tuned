"""Offline tests for the provider layer.

Everything here runs with a fake clock, a fake sleeper and
``httpx.MockTransport``: no test sleeps for real and no test opens a socket.
The fake sleeper ADVANCES the fake clock by the requested delay, so retry
backoff and token-bucket refill are exercised end to end in microseconds.
"""

import asyncio
import json
import os
import random
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")

from pipeline_fakes import (  # noqa: E402
    cfg_with_context,
    cfg_with_two_generator_families,
    cfg_with_split_pools,
    cfg_without_the_paid_judges,
)

from tuned.data.config import (  # noqa: E402
    ModelCfg,
    ModelRef,
    ProviderCfg,
    load_build_config,
)
from tuned.data.generate import (  # noqa: E402
    TIEBREAK_PROMPT_ID,
    WORST_CASE_CHAR,
    judge_messages,
    judge_needed_tokens,
    judge_sizer,
    judge_tokens_for_generator_window,
    REPLY_BUDGET_CHARS_PER_TOKEN,
    max_output_tokens,
    min_judge_tokens,
    preflight_messages,
    worst_case_judge_tokens,
)
from tuned.data.providers import (  # noqa: E402
    CHARS_PER_TOKEN_LATIN,
    CONTEXT_SAFETY_MARGIN,
    DEFAULT_JUDGE_REPLY_TOKENS,
    QUIRKS,
    TEMPLATE_TOKENS_PER_MESSAGE,
    ChatClient,
    ChatRequest,
    CheckResult,
    PoolGap,
    ProviderError,
    Router,
    TokenBucket,
    _blocking_key_envs,
    _generator_windows,
    build_check_request,
    check_refs,
    context_estimate,
    estimate_tokens,
    format_check_header,
    format_check_row,
    load_dotenv_keys,
    looks_like_context_overflow,
    pool_gaps,
    required_context,
    resolve_quirks,
    undersized_families,
    unkeyed_roles,
)

DATA_CONFIG = Path(__file__).parent.parent / "configs" / "data_law_v1.yaml"

MISTRAL_JUDGE = ModelRef("mistral", "mistral-small-latest")
GROQ_JUDGE = ModelRef("groq", "qwen/qwen3.6-27b")
GLM_JUDGE = ModelRef("cerebras", "zai-glm-4.7")


# --- fakes ------------------------------------------------------------------


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


class FakeSleeper:
    """Records every requested delay and advances the fake clock by it."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.slept: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.slept.append(delay)
        self.clock.advance(delay)


def _model(
    model_id: str = "m1",
    *,
    family: str = "fam",
    roles: tuple[str, ...] = ("generator",),
    limits: dict | None = None,
    params: dict | None = None,
) -> ModelCfg:
    return ModelCfg(
        id=model_id,
        family=family,
        roles=roles,
        limits=limits if limits is not None else {"rpm": 60, "tpm": 60000, "max_output": 4096},
        params=params if params is not None else {"temperature": 0.7, "top_p": 0.95},
    )


def _provider(
    name: str = "groq",
    *,
    quirks: tuple[str, ...] = ("groq",),
    api_key_env: str = "TUNED_TEST_KEY",
    models: tuple[ModelCfg, ...] | None = None,
) -> ProviderCfg:
    return ProviderCfg(
        name=name,
        base_url="https://api.test.local/v1",
        api_key_env=api_key_env,
        quirks=quirks,
        models=models if models is not None else (_model(),),
    )


def _body(content="OK", *, reasoning=None, reasoning_key="reasoning", prompt=11, completion=7):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message[reasoning_key] = reasoning
    return {
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": 18},
    }


def _request(**kw) -> ChatRequest:
    kw.setdefault("messages", ({"role": "user", "content": "hi"},))
    kw.setdefault("ref", ModelRef("groq", "m1"))
    kw.setdefault("params", {})
    return ChatRequest(**kw)


def _client(handler, *, provider=None, model=None, clock=None, sleeper=None, **kw) -> ChatClient:
    clock = clock or FakeClock()
    return ChatClient(
        provider or _provider(),
        model or _model(),
        transport=httpx.MockTransport(handler),
        clock=clock,
        sleeper=sleeper or FakeSleeper(clock),
        rng=random.Random(1234),
        **kw,
    )


def _complete(client: ChatClient, req: ChatRequest):
    async def run():
        try:
            return await client.complete(req)
        finally:
            await client.aclose()

    return asyncio.run(run())


def _unset(monkeypatch, *names: str) -> None:
    """Delete env vars in a way monkeypatch will restore even if absent now."""
    for name in names:
        monkeypatch.setenv(name, "sentinel")
        monkeypatch.delenv(name)


# --- 1. happy path ----------------------------------------------------------


def test_complete_parses_body_and_carries_auth(monkeypatch):
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")
    clock = FakeClock()
    seen = []

    def handler(request):
        seen.append(request)
        clock.advance(0.25)  # latency comes from clock deltas
        return httpx.Response(200, json=_body("OK", reasoning="step one"))

    response = _complete(
        _client(handler, clock=clock),
        _request(params={"temperature": 0.2}, max_tokens=64),
    )

    assert response.text == "OK"
    assert response.reasoning == "step one"
    assert (response.prompt_tokens, response.completion_tokens) == (11, 7)
    assert response.finish_reason == "stop"
    assert response.status == 200
    assert response.latency_ms == 250
    assert response.raw["id"] == "chatcmpl-1"

    assert len(seen) == 1
    assert seen[0].headers["authorization"] == "Bearer sk-secret"
    assert str(seen[0].url) == "https://api.test.local/v1/chat/completions"
    payload = json.loads(seen[0].content)
    assert payload["model"] == "m1"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["top_p"] == 0.95  # model.params carried through
    assert payload["temperature"] == 0.2  # req.params WINS over model.params (0.7)
    assert payload["max_tokens"] == 64


def test_reasoning_content_alias_and_null_content(monkeypatch):
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")

    def handler(request):
        return httpx.Response(
            200, json=_body(None, reasoning="thinking", reasoning_key="reasoning_content")
        )

    response = _complete(_client(handler), _request())
    assert response.text == ""
    assert response.reasoning == "thinking"


def test_missing_usage_block_counts_zero(monkeypatch):
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")
    body = _body()
    del body["usage"]

    def handler(request):
        return httpx.Response(200, json=body)

    response = _complete(_client(handler), _request())
    assert (response.prompt_tokens, response.completion_tokens) == (0, 0)


# --- 2. quirks --------------------------------------------------------------


def test_cerebras_quirk_clamps_max_tokens(monkeypatch):
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=_body())

    cerebras = _provider("cerebras", quirks=("cerebras",))
    _complete(_client(handler, provider=cerebras), _request(max_tokens=100_000))
    assert payloads[-1]["max_tokens"] == 4096  # clamped to limits["max_output"]

    _complete(_client(handler, provider=cerebras), _request(max_tokens=128))
    assert payloads[-1]["max_tokens"] == 128  # already under the ceiling: untouched

    _complete(_client(handler, provider=_provider("groq")), _request(max_tokens=100_000))
    assert payloads[-1]["max_tokens"] == 100_000  # default quirk passes through


def test_openai_quirk_renames_max_tokens_and_never_sends_temperature(monkeypatch):
    """Both measured as 400s against the live gpt-5 API 2026-08-15, and a 400
    with no context marker is in _ABORT_STATUSES - it aborts the call rather
    than failing over, so either field left in place is a judge role that
    fails every call it is handed instead of quietly degrading.

    The temperature has to be dropped at the HOOK and not merely left out of
    the config: judge.py sends `params={"temperature": ...}` per call, and
    `build_payload` merges request params OVER the model's own, so the config
    carrying none is only half the fix."""
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=_body())

    openai = _provider("openai", quirks=("openai",))
    gpt5 = _model("gpt-5-mini", family="gpt-oss", roles=("judge",), params={})

    # The per-CALL temperature - the one the config cannot prevent.
    _complete(
        _client(handler, provider=openai, model=gpt5),
        _request(max_tokens=1024, params={"temperature": 0.2}),
    )
    assert payloads[-1]["max_completion_tokens"] == 1024
    assert "max_tokens" not in payloads[-1]
    assert "temperature" not in payloads[-1]
    # ...and everything else the caller asked for still ships.
    assert payloads[-1]["model"] == "gpt-5-mini"
    assert payloads[-1]["messages"] == [{"role": "user", "content": "hi"}]

    # A temperature carried by the MODEL's own params is dropped too.
    warm = _model("gpt-5-nano", family="gpt-oss", roles=("judge",), params={"temperature": 0.7})
    _complete(_client(handler, provider=openai, model=warm), _request(max_tokens=64))
    assert "temperature" not in payloads[-1]
    assert payloads[-1]["max_completion_tokens"] == 64

    # A call with no reply allowance sends neither name - the hook renames a
    # field, it does not invent one.
    _complete(_client(handler, provider=openai, model=gpt5), _request(params={"top_p": 0.95}))
    assert "max_tokens" not in payloads[-1] and "max_completion_tokens" not in payloads[-1]
    assert payloads[-1]["top_p"] == 0.95

    # ...and none of this leaks onto the providers that still take both fields.
    _complete(
        _client(handler, provider=_provider("groq")),
        _request(max_tokens=512, params={"temperature": 0.2}),
    )
    assert payloads[-1]["max_tokens"] == 512 and payloads[-1]["temperature"] == 0.2
    assert "max_completion_tokens" not in payloads[-1]


def test_openai_quirk_refuses_two_reply_allowances(monkeypatch):
    """Both names set means two different budgets are in play and the rename is
    about to drop one of them; which one survives would come down to dict
    ordering. The loser is a ceiling somebody set on purpose, so this fails
    loudly here rather than on the wire."""
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")
    openai = _provider("openai", quirks=("openai",))
    gpt5 = _model("gpt-5-mini", family="gpt-oss", roles=("judge",), params={})
    client = _client(lambda request: httpx.Response(200, json=_body()), provider=openai, model=gpt5)

    with pytest.raises(ValueError) as excinfo:
        client.build_payload(_request(max_tokens=1024, params={"max_completion_tokens": 256}))
    message = str(excinfo.value)
    assert "gpt-5-mini" in message
    assert "1024" in message and "256" in message

    # One of them alone is fine, whichever name it arrives under.
    assert client.build_payload(_request(max_tokens=1024))["max_completion_tokens"] == 1024
    only_new = client.build_payload(_request(params={"max_completion_tokens": 256}))
    assert only_new["max_completion_tokens"] == 256 and "max_tokens" not in only_new
    asyncio.run(client.aclose())


def test_unknown_quirk_name_raises_at_construction():
    with pytest.raises(KeyError) as excinfo:
        ChatClient(_provider(quirks=("nosuchprovider",)), _model())
    message = str(excinfo.value)
    assert "nosuchprovider" in message
    assert "cerebras" in message  # lists the known names


def test_resolve_quirks_composes_request_hooks_in_order():
    composed = resolve_quirks(("groq", "cerebras"))
    clamped = composed.request_hook({"max_tokens": 99999}, _model())
    assert clamped["max_tokens"] == 4096
    assert resolve_quirks(()) is QUIRKS["default"]
    assert resolve_quirks(("groq",)) is QUIRKS["groq"]


# --- 3. retries -------------------------------------------------------------


def test_429_then_success_honors_retry_after(monkeypatch):
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "7"},
                json={"error": "rate limited", "usage": {"prompt_tokens": 999}},
            )
        return httpx.Response(200, json=_body("OK", prompt=11, completion=7))

    response = _complete(_client(handler, clock=clock, sleeper=sleeper), _request())

    assert len(calls) == 2
    assert sleeper.slept == [7.0]  # max(Retry-After, jittered backoff)
    assert response.status == 200
    assert (response.prompt_tokens, response.completion_tokens) == (11, 7)  # SUCCESS body only


def test_absurd_retry_after_fails_over_instead_of_sleeping(monkeypatch):
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "86400"}, text="come back tomorrow")

    with pytest.raises(ProviderError) as excinfo:
        _complete(_client(handler, clock=clock, sleeper=sleeper), _request())

    assert excinfo.value.retryable is True  # the Router may try another provider
    assert excinfo.value.status == 429
    assert sleeper.slept == []  # never parks the pipeline for a day
    assert len(calls) == 1


def test_retry_sleep_budget_bounds_total_parking(monkeypatch):
    """Individually-reasonable Retry-Afters must not ADD UP to a long park.

    Six attempts each honouring "Retry-After: 50" would sit on one call for
    four minutes (and a three-ref Router pass for twelve).  The budget is
    per-call, not per-attempt, so the third wait is refused and the Router
    gets a retryable error it can fail over on.
    """
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "50"}, text="slow down")

    with pytest.raises(ProviderError) as excinfo:
        _complete(
            _client(
                handler, clock=clock, sleeper=sleeper, max_retries=6, max_retry_sleep_s=120.0
            ),
            _request(),
        )

    assert excinfo.value.retryable is True
    assert excinfo.value.status == 429
    assert sleeper.slept == [50.0, 50.0]  # 50 + 50 fits; a third would blow 120
    assert sum(sleeper.slept) <= 120.0
    assert len(calls) == 3  # stopped early - did NOT burn all 6 attempts
    assert "budget" in str(excinfo.value)


def test_plain_backoff_never_trips_the_sleep_budget(monkeypatch):
    """Full-jitter backoff at the default depth must stay well inside the cap."""
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500, text="upstream boom")

    with pytest.raises(ProviderError) as excinfo:
        _complete(_client(handler, clock=clock, sleeper=sleeper, max_retries=6), _request())

    assert len(calls) == 6  # all attempts spent on retries, none on the budget
    assert sum(sleeper.slept) < 120.0  # 1+2+4+8+16 is the ceiling of 5 jittered sleeps
    assert "exhausted 6 attempts" in str(excinfo.value)
    assert excinfo.value.retryable is True


def test_non_429_4xx_is_not_retried(monkeypatch):
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, text="invalid api key")

    with pytest.raises(ProviderError) as excinfo:
        _complete(_client(handler), _request())

    assert excinfo.value.retryable is False
    assert excinfo.value.status == 401
    assert excinfo.value.provider == "groq"
    assert excinfo.value.model == "m1"
    assert len(calls) == 1


@pytest.mark.parametrize("status", [401, 403, 404, 402])
def test_credential_and_missing_model_4xx_mark_the_provider_dead(status, monkeypatch):
    """Per-PROVIDER facts: another provider can still serve the same payload."""
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")

    def handler(request):
        return httpx.Response(status, text="nope")

    with pytest.raises(ProviderError) as excinfo:
        _complete(_client(handler), _request())

    assert excinfo.value.retryable is False  # retrying HERE is pointless
    assert excinfo.value.provider_dead is True  # but the Router should move on
    assert excinfo.value.status == status


@pytest.mark.parametrize("status", [400, 413, 422])
def test_payload_4xx_aborts_rather_than_failing_over(status, monkeypatch):
    """A malformed payload is malformed everywhere - surface it, don't hide it."""
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")

    def handler(request):
        return httpx.Response(status, text="bad request")

    with pytest.raises(ProviderError) as excinfo:
        _complete(_client(handler), _request())

    assert excinfo.value.retryable is False
    assert excinfo.value.provider_dead is False
    assert excinfo.value.status == status


def test_persistent_500_exhausts_retries(monkeypatch):
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500, text="upstream boom")

    with pytest.raises(ProviderError) as excinfo:
        _complete(_client(handler, clock=clock, sleeper=sleeper, max_retries=3), _request())

    assert len(calls) == 3  # exactly max_retries requests
    assert len(sleeper.slept) == 2  # no pointless sleep after the last attempt
    assert all(0.0 <= s <= 60.0 for s in sleeper.slept)  # full jitter, capped
    assert excinfo.value.retryable is True
    assert excinfo.value.status == 500


def test_call_deadline_stops_a_hanging_provider_before_max_retries(monkeypatch):
    """Sleeps are bounded, but a provider can also burn the clock INSIDE an
    attempt.  Six attempts x a 120s HTTP timeout is 12 minutes on one ref and
    ~36 across a three-ref pass; the deadline counts attempt time too."""
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    calls = []

    def handler(request):
        calls.append(request)
        clock.advance(120.0)  # the request "hangs" for the full HTTP timeout
        return httpx.Response(500, text="too slow")

    with pytest.raises(ProviderError) as excinfo:
        _complete(
            _client(
                handler, clock=clock, sleeper=sleeper, max_retries=6, call_deadline_s=300.0
            ),
            _request(),
        )

    assert len(calls) == 3  # NOT the full 6 - the deadline cut it short
    assert clock.now - 1000.0 < 400.0  # bounded near the 300s ceiling
    assert excinfo.value.retryable is True  # the Router may try another provider
    assert "deadline" in str(excinfo.value)
    assert sum(sleeper.slept) < 120.0  # the sleep budget was never the binding limit


def test_call_deadline_also_counts_time_spent_in_before_attempt(monkeypatch):
    """A bucket wait before a retry burns the deadline just like a slow request."""
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500, text="boom")

    async def slow_bucket_wait():
        clock.advance(200.0)  # e.g. waiting out a 1-rpm token bucket

    async def run():
        client = _client(
            handler, clock=clock, sleeper=sleeper, max_retries=6, call_deadline_s=300.0
        )
        try:
            return await client.complete(_request(), before_attempt=slow_bucket_wait)
        finally:
            await client.aclose()

    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(run())

    assert len(calls) == 2  # attempt 0, one retry, then the deadline bites
    assert "deadline" in str(excinfo.value)
    assert excinfo.value.retryable is True


def test_backoff_is_full_jitter_and_not_a_constant():
    """Pin the seeded sequence: a regression to constant backoff must fail."""
    client = _client(lambda request: httpx.Response(200, json=_body()))
    delays = [client._backoff(a) for a in range(5)]

    assert delays == [
        pytest.approx(0.966453535692),
        pytest.approx(0.881465198351),
        pytest.approx(0.029965880234),
        pytest.approx(7.287807699593),
        pytest.approx(15.02830395782),
    ]
    # Full jitter: every draw sits in [0, 2**attempt], and they are not equal.
    for attempt, delay in enumerate(delays):
        assert 0.0 <= delay <= min(60.0, 2**attempt)
    assert len(set(delays)) == len(delays)
    assert client._backoff(20) <= 60.0  # ceiling clamps the exponential


def test_transport_error_is_retryable(monkeypatch):
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("connection reset", request=request)
        return httpx.Response(200, json=_body())

    response = _complete(_client(handler, max_retries=3), _request())
    assert response.text == "OK"
    assert len(calls) == 2


# --- 6. credentials ---------------------------------------------------------


def test_missing_api_key_raises_at_call_time_not_construction(monkeypatch):
    _unset(monkeypatch, "TUNED_TEST_KEY")
    calls = []

    def handler(request):  # pragma: no cover - must never run
        calls.append(request)
        return httpx.Response(200, json=_body())

    client = _client(handler)  # construction must NOT raise

    with pytest.raises(ProviderError) as excinfo:
        _complete(client, _request())

    assert excinfo.value.retryable is False
    assert excinfo.value.status is None
    assert "TUNED_TEST_KEY" in str(excinfo.value)
    assert calls == []


# --- 7. token bucket --------------------------------------------------------


def test_token_bucket_rpm_refills_continuously():
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    bucket = TokenBucket(2, None, clock=clock, sleeper=sleeper)

    asyncio.run(bucket.acquire(0))
    asyncio.run(bucket.acquire(0))
    assert sleeper.slept == []  # starts full
    assert bucket.next_wait(0) == pytest.approx(30.0)  # 2/60 per second

    clock.advance(29.0)
    assert bucket.next_wait(0) == pytest.approx(1.0)
    clock.advance(1.0)
    assert bucket.next_wait(0) == 0.0
    asyncio.run(bucket.acquire(0))
    assert sleeper.slept == []


def test_token_bucket_acquire_sleeps_until_refilled():
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    bucket = TokenBucket(2, None, clock=clock, sleeper=sleeper)
    asyncio.run(bucket.acquire(0))
    asyncio.run(bucket.acquire(0))

    asyncio.run(bucket.acquire(0))
    assert sleeper.slept == [pytest.approx(30.0)]
    assert clock.now == pytest.approx(1030.0)


def test_token_bucket_tpm_accounting():
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    bucket = TokenBucket(None, 6000, clock=clock, sleeper=sleeper)  # 100 tok/s

    asyncio.run(bucket.acquire(3000))
    assert sleeper.slept == []
    assert bucket.next_wait(3000) == 0.0
    assert bucket.next_wait(6000) == pytest.approx(30.0)

    clock.advance(30.0)
    assert bucket.next_wait(6000) == 0.0


def test_token_bucket_does_not_deadlock_above_capacity():
    """est_tokens > tpm capacity must degrade to 'wait for a full bucket', not hang."""
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    bucket = TokenBucket(None, 1000, clock=clock, sleeper=sleeper)

    asyncio.run(bucket.acquire(5000))  # full bucket: immediate
    assert sleeper.slept == []
    assert bucket.next_wait(5000) == pytest.approx(60.0)  # a full refill, never forever

    asyncio.run(bucket.acquire(5000))
    assert sleeper.slept == [pytest.approx(60.0)]


def test_token_bucket_serializes_concurrent_waiters():
    """Two coroutines must not both decide there is room for the same slot."""
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    bucket = TokenBucket(1, None, clock=clock, sleeper=sleeper)
    order = []

    async def worker(name):
        await bucket.acquire(0)
        order.append((name, clock.now))

    async def run():
        await asyncio.gather(worker("a"), worker("b"))

    asyncio.run(run())
    assert order == [("a", 1000.0), ("b", 1060.0)]  # 1 rpm -> the second waits a minute
    assert sleeper.slept == [pytest.approx(60.0)]


@pytest.mark.parametrize("rpm,tpm", [(0, None), (None, 0), (0, 0), (-1, None), (None, -5)])
def test_token_bucket_rejects_non_positive_limits(rpm, tpm):
    """0 is a typo, not a limit - and BOTH readings of it are wrong (unlimited
    silently drops a configured cap, no-capacity wedges forever)."""
    with pytest.raises(ValueError) as excinfo:
        TokenBucket(rpm, tpm, clock=FakeClock())
    assert "positive" in str(excinfo.value)


def test_token_bucket_without_limits_never_waits():
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    bucket = TokenBucket(None, None, clock=clock, sleeper=sleeper)
    for _ in range(50):
        asyncio.run(bucket.acquire(10**6))
    assert sleeper.slept == []
    assert bucket.next_wait(10**6) == 0.0


# --- router fixtures --------------------------------------------------------


@pytest.fixture
def cfg():
    return load_build_config(DATA_CONFIG, allow_unpinned=True)


@pytest.fixture
def keys(cfg, monkeypatch):
    """Every provider in the SHIPPED config, keyed.

    Derived from the config rather than listed, because a provider added to
    the config and missed here does not fail: `eligible_refs` skips its models
    as "missing-key", so every preflight test in this module goes on quietly
    measuring the OLD pool and passing. That is how a judge added to close a
    gap can leave the test that asserts the gap green.
    """
    for provider in cfg.providers:
        monkeypatch.setenv(provider.api_key_env, "sk-test")


def _router(cfg, **kw) -> Router:
    clock = kw.setdefault("clock", FakeClock())
    # Default the sleeper to the fake too: a Router built without one would
    # fall back to real asyncio.sleep and a rate-limit wait would really sleep.
    kw.setdefault("sleeper", FakeSleeper(clock))

    def handler(request):  # pragma: no cover - pick() tests never send
        return httpx.Response(200, json=_body())

    kw.setdefault("transport", httpx.MockTransport(handler))
    return Router(cfg, **kw)


# --- 8. pick ----------------------------------------------------------------


def test_pick_respects_configured_order(cfg, keys):
    router = _router(cfg)
    assert router.pick("judge").ref == MISTRAL_JUDGE
    assert router.pick("generator").ref == ModelRef("cerebras", "gpt-oss-120b")
    assert router.pick("probe").ref == ModelRef("groq", "openai/gpt-oss-20b")


def test_pick_excludes_families_at_call_time(cfg, keys):
    router = _router(cfg)

    # A gpt-oss generation may never be judged (or tie-broken) by gpt-oss.
    for role in ("judge", "tiebreak"):
        for routed in router.eligible(role, exclude_families=frozenset({"gpt-oss"})):
            assert routed.model_cfg.family != "gpt-oss"
        picked = router.pick(role, exclude_families=frozenset({"gpt-oss"}))
        assert picked is not None and picked.model_cfg.family != "gpt-oss"

    # tiebreak's first ref IS gpt-oss, so exclusion must move the pick along.
    assert router.pick("tiebreak").ref == ModelRef("groq", "openai/gpt-oss-20b")
    assert router.pick("tiebreak", exclude_families=frozenset({"gpt-oss"})).ref == ModelRef(
        "cerebras", "gemma-4-31b"
    )
    # ...and since the 2026-08-18 demotion there IS no second generator family,
    # so excluding gpt-oss leaves the generator role with nothing to pick. That
    # is the routing half of "a long prompt now parks instead of diverting".
    assert router.pick("generator", exclude_families=frozenset({"gpt-oss"})) is None


def test_pick_skips_missing_key(cfg, keys, monkeypatch):
    _unset(monkeypatch, "MISTRAL_API_KEY")
    router = _router(cfg)
    assert router.pick("judge").ref == GROQ_JUDGE


def test_pick_skips_over_budget(cfg, keys):
    def budget_ok(provider, model, tokens):
        assert tokens == 500
        return not (provider == "mistral" or model == "qwen/qwen3.6-27b")

    router = _router(cfg, budget_ok=budget_ok)
    assert router.pick("judge", est_tokens=500).ref == GLM_JUDGE


def test_pick_returns_none_when_nothing_eligible(cfg, keys):
    router = _router(cfg, budget_ok=lambda provider, model, tokens: False)
    assert router.pick("judge") is None
    assert router.pick("probe", exclude_families=frozenset({"gpt-oss"})) is None


def test_router_forwards_the_client_knobs(cfg, keys):
    """The retry/timeout ceilings must be reachable through the Router - they
    are useless if only a hand-built ChatClient can set them."""
    router = _router(
        cfg, max_retries=2, timeout=7.0, max_retry_sleep_s=11.0, call_deadline_s=13.0
    )
    client = router.routed(MISTRAL_JUDGE).client
    assert client.max_retries == 2
    assert client.max_retry_sleep_s == 11.0
    assert client.call_deadline_s == 13.0
    assert client._client.timeout.read == 7.0


def test_pick_caches_routed_models(cfg, keys):
    router = _router(cfg)
    first = router.pick("judge")
    assert router.pick("judge") is first  # one client + one bucket per ref
    # The bucket is built from THAT model's own limits, and the mistral figures
    # are where that matters. The workspace allows 50 rpm / 50k tpm and the
    # bucket is per (provider, model) - header-verified 2026-08-14, the
    # remaining-token counter decrements across calls made with different
    # models, so the ceiling is per WORKSPACE.
    #
    # It used to be configured as 25/25k, and that halving was arithmetic, not
    # policy: TWO mistral models shared the one workspace bucket, so each was
    # given half or the pair would have issued at twice the real refill rate.
    # The magistral line was retired upstream on 2026-07-31 and its block is
    # gone, so there is one model, one bucket, and the whole allowance - which
    # is a restoration rather than a raise.
    assert (first.bucket.rpm, first.bucket.tpm) == (50, 50000)
    # ...and now that one model carries both roles, the generator and the judge
    # really do share that single bucket rather than each getting one.
    generator = router.routed(ModelRef("mistral", "mistral-small-latest"))
    assert generator is first
    assert generator.bucket is first.bucket


# --- 9. circuit breaker -----------------------------------------------------


def test_circuit_breaker_cools_then_recovers(cfg, keys):
    clock = FakeClock()
    router = _router(cfg, clock=clock, breaker_threshold=2, cooldown_s=300.0)

    router.report_failure(MISTRAL_JUDGE)
    assert router.pick("judge").ref == MISTRAL_JUDGE  # below threshold
    router.report_failure(MISTRAL_JUDGE)

    assert router.is_cooling(MISTRAL_JUDGE)
    assert router.pick("judge").ref == GROQ_JUDGE

    clock.advance(299.0)
    assert router.pick("judge").ref == GROQ_JUDGE
    clock.advance(2.0)
    assert not router.is_cooling(MISTRAL_JUDGE)
    assert router.pick("judge").ref == MISTRAL_JUDGE


def test_report_success_resets_the_failure_run(cfg, keys):
    router = _router(cfg, breaker_threshold=2)
    router.report_failure(MISTRAL_JUDGE)
    router.report_success(MISTRAL_JUDGE)
    router.report_failure(MISTRAL_JUDGE)
    assert not router.is_cooling(MISTRAL_JUDGE)  # failures must be CONSECUTIVE
    assert router.pick("judge").ref == MISTRAL_JUDGE


# --- 10. Router.complete ----------------------------------------------------


def _factory(clock, sleeper, seen, statuses, *, max_retries=2):
    """Per-provider MockTransport whose status is looked up by provider name."""

    def factory(provider, model):
        def handler(request):
            seen.append(provider.name)
            status = statuses.get(provider.name, 200)
            if status == 200:
                return httpx.Response(200, json=_body(f"OK from {provider.name}"))
            return httpx.Response(status, text=f"{provider.name} says {status}")

        return ChatClient(
            provider,
            model,
            transport=httpx.MockTransport(handler),
            clock=clock,
            sleeper=sleeper,
            rng=random.Random(7),
            max_retries=max_retries,
        )

    return factory


def test_complete_fails_over_to_the_next_ref(cfg, keys):
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    seen: list[str] = []
    router = _router(
        cfg,
        clock=clock,
        sleeper=sleeper,
        breaker_threshold=1,  # one failure cools it: proves report_failure ran
        client_factory=_factory(clock, sleeper, seen, {"mistral": 500}),
    )

    ref, response = asyncio.run(
        router.complete("judge", [{"role": "user", "content": "grade this"}], est_tokens=250)
    )

    assert ref == GROQ_JUDGE
    assert response.text == "OK from groq"
    assert seen == ["mistral", "mistral", "groq"]  # 2 in-provider retries, then failover
    assert router.is_cooling(MISTRAL_JUDGE)
    assert not router.is_cooling(GROQ_JUDGE)


def test_complete_raises_when_every_ref_fails(cfg, keys):
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    seen: list[str] = []
    router = _router(
        cfg,
        clock=clock,
        sleeper=sleeper,
        client_factory=_factory(
            clock, sleeper, seen, {"mistral": 500, "groq": 503, "cerebras": 500, "openai": 500}
        ),
    )

    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(router.complete("judge", [{"role": "user", "content": "grade this"}]))

    assert excinfo.value.retryable is True
    # Five refs over four providers: the openai backstop contributes two, and
    # BOTH have to be tried before the role is out of options.
    assert "all 5 eligible model(s) failed" in str(excinfo.value)
    assert seen.count("mistral") == 2 and seen.count("groq") == 2 and seen.count("cerebras") == 2
    assert seen.count("openai") == 4  # two refs, two in-provider attempts each


def test_complete_does_not_fail_over_on_non_retryable(cfg, keys):
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    seen: list[str] = []
    router = _router(
        cfg,
        clock=clock,
        sleeper=sleeper,
        client_factory=_factory(clock, sleeper, seen, {"mistral": 400}),
    )

    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(router.complete("judge", [{"role": "user", "content": "grade this"}]))

    assert excinfo.value.retryable is False
    assert seen == ["mistral"]  # a bad payload would be bad everywhere
    assert not router.is_cooling(MISTRAL_JUDGE)


@pytest.mark.parametrize("status", [401, 403, 404])
def test_complete_fails_over_when_a_provider_is_dead(status, cfg, keys):
    """A revoked key or a retired preview model must not sink the whole role."""
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    seen: list[str] = []
    router = _router(
        cfg,
        clock=clock,
        sleeper=sleeper,
        breaker_threshold=1,  # one failure cools it: proves report_failure ran
        client_factory=_factory(clock, sleeper, seen, {"mistral": status}),
    )

    ref, response = asyncio.run(
        router.complete("judge", [{"role": "user", "content": "grade this"}])
    )

    assert ref == GROQ_JUDGE  # moved on to the next eligible judge
    assert response.text == "OK from groq"
    assert seen == ["mistral", "groq"]  # dead provider tried ONCE, not retried
    assert router.is_cooling(MISTRAL_JUDGE)  # and marked failed


def test_complete_aborts_the_whole_call_on_a_payload_4xx(cfg, keys):
    """400 is our bug: every provider would reject it, so do not hide it."""
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    seen: list[str] = []
    router = _router(
        cfg,
        clock=clock,
        sleeper=sleeper,
        client_factory=_factory(clock, sleeper, seen, {"mistral": 400}),
    )

    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(router.complete("judge", [{"role": "user", "content": "grade this"}]))

    assert excinfo.value.retryable is False
    assert excinfo.value.provider_dead is False
    assert seen == ["mistral"]  # ref2 never called
    assert not router.is_cooling(MISTRAL_JUDGE)


def test_no_eligible_model_is_retryable_only_when_the_reason_is_transient(cfg, keys, monkeypatch):
    """"Everything is cooling" lifts on its own; "no key set" does not."""
    # Transient: over budget today.
    router = _router(cfg, budget_ok=lambda provider, model, tokens: False)
    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(router.complete("judge", [{"role": "user", "content": "hi"}]))
    assert excinfo.value.retryable is True
    assert "over-budget" in str(excinfo.value)

    # Transient: the whole judge pool is cooling behind the breaker.
    clock = FakeClock()
    cooling = _router(cfg, clock=clock, breaker_threshold=1)
    for ref in cfg.routing_refs("judge"):
        cooling.report_failure(ref)
    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(cooling.complete("judge", [{"role": "user", "content": "hi"}]))
    assert excinfo.value.retryable is True
    assert "cooling" in str(excinfo.value)

    # Structural: no keys at all - coming back in a minute changes nothing.
    # Every provider in the config, derived: one left keyed is one ref still
    # eligible, and the call then succeeds instead of raising.
    _unset(monkeypatch, *(p.api_key_env for p in cfg.providers))
    structural = _router(cfg)
    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(structural.complete("judge", [{"role": "user", "content": "hi"}]))
    assert excinfo.value.retryable is False
    assert "missing-key" in str(excinfo.value)


def test_pick_reports_why_it_skipped(cfg, keys):
    router = _router(cfg, budget_ok=lambda provider, model, tokens: False)
    skipped: set[str] = set()
    assert router.pick("judge", skipped=skipped) is None
    assert skipped == {"over-budget"}

    router2 = _router(cfg)
    skipped2: set[str] = set()
    assert router2.pick("tiebreak", exclude_families=frozenset({"gpt-oss"}), skipped=skipped2)
    assert skipped2 == {"family-excluded"}  # only reasons seen BEFORE the hit


def test_retries_charge_the_rpm_bucket_too(cfg, keys):
    """Every retry is a real request against the same per-minute quota."""
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    seen: list[str] = []
    router = _router(
        cfg,
        clock=clock,
        sleeper=sleeper,
        breaker_threshold=99,  # keep it eligible; we only care about the bucket
        client_factory=_factory(clock, sleeper, seen, {"groq": 500}, max_retries=3),
    )
    # Spy on the bucket rather than reading its level: the fake sleeper
    # advances the clock during backoff, so the bucket legitimately refills
    # mid-call and the level alone would not show the charges.
    bucket = router.routed(GROQ_JUDGE).bucket
    charges: list[int] = []
    real_acquire = bucket.acquire

    async def counting_acquire(est_tokens=0):
        charges.append(est_tokens)
        await real_acquire(est_tokens)

    bucket.acquire = counting_acquire

    with pytest.raises(ProviderError):
        asyncio.run(
            router.complete(
                "judge",
                [{"role": "user", "content": "hi"}],
                # Isolate the groq ref: every OTHER family in the judge pool,
                # which now includes the gpt-oss backstop.
                exclude_families=frozenset({"mistral", "glm", "gpt-oss"}),
                est_tokens=100,
            )
        )

    assert seen == ["groq", "groq", "groq"]  # 3 real HTTP requests
    # ...so 3 rpm charges: one from Router.complete carrying the token
    # estimate, then one zero-token charge per RETRY via before_attempt.
    assert charges == [100, 0, 0]


# NOTE: an earlier test asserted that "nothing eligible" is ALWAYS
# retryable=False.  That is superseded by
# test_no_eligible_model_is_retryable_only_when_the_reason_is_transient,
# which distinguishes transient causes (cooling, over-budget) from
# structural ones (missing key, family exclusion) and covers both.


def test_complete_charges_the_bucket_of_the_ref_it_used(cfg, keys):
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    seen: list[str] = []
    router = _router(
        cfg, clock=clock, sleeper=sleeper, client_factory=_factory(clock, sleeper, seen, {})
    )
    asyncio.run(router.complete("probe", [{"role": "user", "content": "hi"}], est_tokens=4000))

    bucket = router.routed(ModelRef("groq", "openai/gpt-oss-20b")).bucket
    # probe limits: rpm 30, tpm 8000 -> 4000 charged leaves 4000.
    assert bucket.next_wait(4000) == 0.0
    assert bucket.next_wait(8000) == pytest.approx(30.0)


def test_complete_forwards_params_and_max_tokens(cfg, keys):
    clock = FakeClock()
    payloads = []

    def factory(provider, model):
        def handler(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json=_body())

        return ChatClient(provider, model, transport=httpx.MockTransport(handler), clock=clock)

    router = _router(cfg, clock=clock, client_factory=factory)
    asyncio.run(
        router.complete(
            "judge",
            [{"role": "user", "content": "hi"}],
            params={"temperature": 0.0},
            max_tokens=256,
        )
    )
    assert payloads[0]["temperature"] == 0.0
    assert payloads[0]["max_tokens"] == 256
    assert payloads[0]["model"] == "mistral-small-latest"


# --- 10b. per-ref params, attempt reporting, context sizing -----------------


def test_on_attempt_reports_every_http_attempt_including_retries(monkeypatch):
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")
    """The client's internal retries are invisible in the return value: five
    429s and a success look like one call. A daily ledger fed only by the
    response therefore under-counts requests exactly when a provider is
    rate-limiting hardest."""
    clock = FakeClock()
    seen: list[tuple[int | None, dict | None]] = []
    statuses = iter([429, 500, 200])

    def handler(request):
        status = next(statuses)
        if status == 200:
            return httpx.Response(200, json=_body(prompt=13, completion=5))
        return httpx.Response(status, text="nope")

    client = _client(handler, clock=clock, sleeper=FakeSleeper(clock), max_retries=4)

    async def run():
        try:
            return await client.complete(
                _request(), on_attempt=lambda status, usage: seen.append((status, usage))
            )
        finally:
            await client.aclose()

    response = asyncio.run(run())
    assert response.status == 200
    assert [status for status, _ in seen] == [429, 500, 200]
    assert seen[0][1] is None and seen[1][1] is None
    assert seen[2][1]["prompt_tokens"] == 13


def test_on_attempt_reports_a_transport_failure_as_no_status(monkeypatch):
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")
    clock = FakeClock()
    seen: list[tuple[int | None, dict | None]] = []

    def handler(request):
        raise httpx.ConnectError("down")

    client = _client(handler, clock=clock, sleeper=FakeSleeper(clock), max_retries=1)

    async def run():
        try:
            await client.complete(
                _request(), on_attempt=lambda status, usage: seen.append((status, usage))
            )
        finally:
            await client.aclose()

    with pytest.raises(ProviderError):
        asyncio.run(run())
    assert seen == [(None, None)]


def test_router_on_attempt_carries_the_ref_across_a_failover(cfg, keys):
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    seen: list[tuple[str, int | None]] = []
    router = _router(
        cfg,
        clock=clock,
        sleeper=sleeper,
        client_factory=_factory(clock, sleeper, [], {"mistral": 500}, max_retries=1),
    )
    ref, _ = asyncio.run(
        router.complete(
            "judge",
            [{"role": "user", "content": "grade"}],
            on_attempt=lambda ref, status, usage: seen.append((ref.provider, status)),
        )
    )
    assert ref.provider == "groq"
    # The failed provider's attempt is reported under ITS ref, not the winner's.
    assert seen == [("mistral", 500), ("groq", 200)]


def test_params_for_ref_is_resolved_against_the_ref_about_to_be_called(cfg, keys):
    """A per-call param must be chosen for the ref that ANSWERS, not the one
    the Router would have tried first: an unknown field is a 400, which
    Router.complete raises straight through instead of failing over, so a
    param picked for the first ref turns every failover into a dead call.

    Needs two generator families to have something to fail over to; the
    shipped config has carried one since the 2026-08-18 mistral demotion."""
    cfg = cfg_with_two_generator_families(cfg)
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    payloads: list[dict] = []

    def factory(provider, model):
        def handler(request):
            payloads.append(json.loads(request.content))
            if provider.name == "cerebras":
                return httpx.Response(500, text="down")
            return httpx.Response(200, json=_body())

        return ChatClient(
            provider, model, transport=httpx.MockTransport(handler),
            clock=clock, sleeper=sleeper, rng=random.Random(3), max_retries=1,
        )

    router = _router(cfg, clock=clock, sleeper=sleeper, client_factory=factory)

    def params_for_ref(ref, model_cfg):
        return (
            {"reasoning_effort": "high"}
            if "reasoning_effort" in (model_cfg.params or {})
            else {}
        )

    ref, _ = asyncio.run(
        router.complete(
            "generator", [{"role": "user", "content": "go"}], params_for_ref=params_for_ref
        )
    )
    assert ref == ModelRef("mistral", "mistral-small-latest")
    assert payloads[0]["reasoning_effort"] == "high"      # cerebras/gpt-oss-120b
    assert "reasoning_effort" not in payloads[1]          # mistral/magistral


def test_params_for_ref_overrides_params(cfg, keys):
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    payloads: list[dict] = []

    def factory(provider, model):
        def handler(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json=_body())

        return ChatClient(
            provider, model, transport=httpx.MockTransport(handler),
            clock=clock, sleeper=sleeper, rng=random.Random(3), max_retries=1,
        )

    router = _router(cfg, clock=clock, sleeper=sleeper, client_factory=factory)
    asyncio.run(
        router.complete(
            "judge", [{"role": "user", "content": "go"}],
            params={"temperature": 0.9},
            params_for_ref=lambda ref, model_cfg: {"temperature": 0.1},
        )
    )
    assert payloads[0]["temperature"] == 0.1


def test_params_for_ref_merges_over_params_rather_than_dropping_them(cfg, keys):
    """The hook expresses per-model deviation; the call-wide params it does
    not mention still ship. Dropping them silently sent a payload the caller
    never asked for."""
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    payloads: list[dict] = []

    def factory(provider, model):
        def handler(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json=_body())

        return ChatClient(
            provider, model, transport=httpx.MockTransport(handler),
            clock=clock, sleeper=sleeper, rng=random.Random(3), max_retries=1,
        )

    router = _router(cfg, clock=clock, sleeper=sleeper, client_factory=factory)
    asyncio.run(
        router.complete(
            "judge", [{"role": "user", "content": "go"}],
            params={"temperature": 0.9, "top_p": 0.5},
            params_for_ref=lambda ref, model_cfg: {"temperature": 0.1},
        )
    )
    assert payloads[0]["temperature"] == 0.1
    assert payloads[0]["top_p"] == 0.5


def test_undersized_families_over_the_real_pool(cfg):
    # Nothing is too small for a short prompt.
    assert undersized_families(cfg, "judge", 4000) == frozenset()
    # The 8k judge drops out first, then the 32k one.
    assert undersized_families(cfg, "judge", 20000) == frozenset({"glm"})
    assert undersized_families(cfg, "judge", 40000) == frozenset({"glm", "mistral"})
    # The generator pool is ONE family since 2026-08-18: cerebras/gpt-oss-120b
    # at 8k. Past that window there is no other family to fall back to, which
    # is why an over-long row now parks instead of diverting.
    assert undersized_families(cfg, "generator", 4000) == frozenset()
    assert undersized_families(cfg, "generator", 20000) == frozenset({"gpt-oss"})
    assert undersized_families(cfg, "generator", 60000) == frozenset({"gpt-oss"})


def test_undersized_families_keeps_a_mixed_family_with_one_large_model(cfg):
    # groq's tiebreak gpt-oss model is 131k; a family is excluded only when
    # EVERY one of its role models is too small.
    assert "gpt-oss" not in undersized_families(cfg, "tiebreak", 100000)
    assert undersized_families(cfg, "tiebreak", 100000) == frozenset({"gemma", "glm"})


# --- 11. .env ---------------------------------------------------------------


def test_load_dotenv_keys(tmp_path, monkeypatch):
    _unset(monkeypatch, "TUNED_PLAIN", "TUNED_DQ", "TUNED_SQ", "TUNED_EMPTY")
    monkeypatch.setenv("TUNED_PRESET", "from-environment")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n"
        "\n"
        "TUNED_PLAIN=abc123\n"
        '  TUNED_DQ = "quoted value"  \n'
        "TUNED_SQ='single'\n"
        "TUNED_EMPTY=\n"
        "TUNED_PRESET=from-file\n"
        "garbage line without an equals sign\n",
        encoding="utf-8",
    )

    assert load_dotenv_keys(env_file) == 4  # PRESET already set -> not counted
    assert os.environ["TUNED_PLAIN"] == "abc123"
    assert os.environ["TUNED_DQ"] == "quoted value"
    assert os.environ["TUNED_SQ"] == "single"
    assert os.environ["TUNED_EMPTY"] == ""
    assert os.environ["TUNED_PRESET"] == "from-environment"  # setdefault: env wins


def test_load_dotenv_keys_missing_file_is_zero(tmp_path):
    assert load_dotenv_keys(tmp_path / "nope.env") == 0


# --- 12. CLI purity ---------------------------------------------------------


def test_build_check_request_is_pure():
    req = build_check_request(ModelRef("groq", "qwen/qwen3.6-27b"), max_tokens=32)
    assert req.messages == ({"role": "user", "content": "Reply with the single word OK."},)
    assert req.max_tokens == 32
    assert req.ref == ModelRef("groq", "qwen/qwen3.6-27b")
    assert req.params == {}


def test_check_refs_covers_every_configured_model(cfg):
    refs = check_refs(cfg)
    expected = sum(len(p.models) for p in cfg.providers)
    # 8, not 9: the magistral generator block was removed when the line was
    # retired upstream and mistral-small-latest took both roles.
    assert len(refs) == expected == 8
    assert ModelRef("groq", "qwen/qwen3.6-27b") in refs
    assert check_refs(cfg, "groq/qwen/qwen3.6-27b") == (ModelRef("groq", "qwen/qwen3.6-27b"),)
    with pytest.raises(KeyError):
        check_refs(cfg, "groq/ghost-model")
    with pytest.raises(ValueError):
        check_refs(cfg, "noslash")


def test_format_check_row_is_single_line_and_flags_failures():
    header = format_check_header()
    assert "ref" in header and "status" in header and "usage" in header

    good = CheckResult(
        ref="groq/qwen/qwen3.6-27b",
        key_present=True,
        status=200,
        text="OK\nsecond line",
        usage_present=True,
        reasoning_present=False,
        latency_ms=412,
    )
    row = format_check_row(good)
    assert good.ok
    assert "\n" not in row  # a table row stays a row
    assert "groq/qwen/qwen3.6-27b" in row
    assert "200" in row and "412" in row and "OK second line" in row
    assert len(row.split("|")) == len(header.split("|"))

    bad = CheckResult(
        ref="mistral/mistral-small-latest",
        key_present=False,
        status=None,
        text="",
        usage_present=False,
        reasoning_present=False,
        latency_ms=None,
        error="MISTRAL_API_KEY not set",
    )
    bad_row = format_check_row(bad)
    assert not bad.ok
    assert "FAIL" in bad_row and "MISTRAL_API_KEY not set" in bad_row
    assert CheckResult(
        ref="x", key_present=True, status=500, text="", usage_present=False,
        reasoning_present=False, latency_ms=None,
    ).ok is False


# --- 13. structural refusals: skip reasons, context overflow, pool preflight -
#
# Round-2 review: `retryable=bool(skipped & TRANSIENT_SKIPS)` is the single
# hinge that decides whether a worker re-queues a row or closes it, and a 400
# that means "this prompt does not fit THIS window" is a per-provider fact,
# not a payload bug. Both are covered here at the transport level.


def _overflow_factory(clock, sleeper, seen, overflowing, body, *, max_retries=2):
    """Per-provider MockTransport that answers 400 with a context-overflow body."""

    def factory(provider, model):
        def handler(request):
            seen.append(provider.name)
            if provider.name in overflowing:
                return httpx.Response(400, text=body)
            return httpx.Response(200, json=_body(f"OK from {provider.name}"))

        return ChatClient(
            provider,
            model,
            transport=httpx.MockTransport(handler),
            clock=clock,
            sleeper=sleeper,
            rng=random.Random(7),
            max_retries=max_retries,
        )

    return factory


OVERFLOW_BODY = (
    '{"error": {"message": "This model has a maximum context length of 8192 tokens, '
    'however you requested 11400 tokens.", "code": "context_length_exceeded"}}'
)


def test_no_eligible_model_carries_the_reasons_it_skipped(cfg, keys, monkeypatch):
    """A caller has to tell "no key anywhere" from "this row fits nowhere":
    one is a fleet-configuration fact a re-queue survives, the other is a fact
    about the row. The message string alone is not a contract."""
    _unset(monkeypatch, *(p.api_key_env for p in cfg.providers))
    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(_router(cfg).complete("judge", [{"role": "user", "content": "hi"}]))
    assert excinfo.value.skipped == frozenset({"missing-key"})
    assert excinfo.value.retryable is False


def test_a_family_exclusion_is_reported_separately_from_a_missing_key(cfg, keys):
    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(
            _router(cfg).complete(
                "probe",
                [{"role": "user", "content": "hi"}],
                exclude_families=frozenset({"gpt-oss"}),
            )
        )
    assert excinfo.value.skipped == frozenset({"family-excluded"})


def test_a_context_overflow_400_fails_over_instead_of_aborting(cfg, keys):
    """The exact failure the generator's context routing exists to prevent.
    400 context_length_exceeded is a per-MODEL fact - the next, larger ref in
    the list serves the identical request - so it must fail over, and it must
    NOT charge the breaker: nothing is wrong with that provider."""
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    seen: list[str] = []
    router = _router(
        cfg,
        clock=clock,
        sleeper=sleeper,
        breaker_threshold=1,  # one charged failure would cool it
        client_factory=_overflow_factory(clock, sleeper, seen, {"mistral"}, OVERFLOW_BODY),
    )

    ref, response = asyncio.run(
        router.complete("judge", [{"role": "user", "content": "grade this"}])
    )

    assert ref == GROQ_JUDGE
    assert response.text == "OK from groq"
    assert seen == ["mistral", "groq"]  # tried once, then moved on
    assert not router.is_cooling(MISTRAL_JUDGE)


@pytest.mark.parametrize(
    "body",
    [
        OVERFLOW_BODY,
        "Prompt is too long: 11400 tokens > 8192 maximum",
        "input length exceeds the context window of this model",
    ],
)
def test_context_overflow_bodies_are_recognised(body, monkeypatch):
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")

    def handler(request):
        return httpx.Response(400, text=body)

    with pytest.raises(ProviderError) as excinfo:
        _complete(_client(handler), _request())

    assert excinfo.value.context_exceeded is True
    assert excinfo.value.provider_dead is False


@pytest.mark.parametrize(
    "body",
    [
        # The two markers round 3 REMOVED, in the bodies that made them wrong:
        # both also describe a max_tokens larger than the model's max_output,
        # which is a payload bug of ours.
        "too many tokens requested: max_tokens 9000 exceeds max_output 4096",
        "please reduce the length of the requested completion",
        '{"error": {"message": "unknown parameter: reasoning_effort", "code": 400}}',
        "rate limit exceeded, please try again in 12s",
        "the model produced 4096 tokens and stopped",
        "",
    ],
)
def test_bodies_that_do_not_say_the_prompt_was_too_long(body):
    """The positive cases alone pin nothing: re-adding a marker leaves them
    all green. A false positive here costs a WAVE - the 400 fails over at
    every ref, aggregates as context_exceeded, and parks the lot in
    gen_unroutable, which reads as a pool gap rather than as our bug."""
    assert looks_like_context_overflow(body) is False


def test_a_genuine_payload_400_still_aborts(monkeypatch):
    """The abort class is not gone, only narrowed. A malformed payload is
    still malformed everywhere and must surface rather than tour the pool."""
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")

    def handler(request):
        return httpx.Response(400, text='{"error": "unknown parameter: reasoning_effort"}')

    with pytest.raises(ProviderError) as excinfo:
        _complete(_client(handler), _request())

    assert excinfo.value.context_exceeded is False
    assert excinfo.value.retryable is False
    assert excinfo.value.provider_dead is False


def test_overflow_at_every_ref_is_reported_as_a_row_shaped_failure(cfg, keys):
    """When every eligible model says the prompt does not fit, coming back
    tomorrow with the same prompt changes nothing - so the aggregate error is
    NOT retryable and says why."""
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    seen: list[str] = []
    router = _router(
        cfg,
        clock=clock,
        sleeper=sleeper,
        client_factory=_overflow_factory(
            clock, sleeper, seen, {"mistral", "groq", "cerebras", "openai"}, OVERFLOW_BODY
        ),
    )

    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(router.complete("judge", [{"role": "user", "content": "grade this"}]))

    assert excinfo.value.context_exceeded is True
    assert excinfo.value.retryable is False
    # Every ref was offered it, the two paid backstops included - a 400 that
    # says "too long for this window" never charges the breaker, so the pass
    # runs to the end of the list rather than stopping at the first refusal.
    assert seen == ["mistral", "groq", "cerebras", "openai", "openai"]


def test_undersized_families_keeps_an_explicit_safety_margin(cfg):
    """chars/4 is an estimate, so a cap that merely EQUALS it is not headroom:
    the 8k judge is excluded at an 8192-token estimate, not only past it."""
    assert CONTEXT_SAFETY_MARGIN > 1.0
    assert "glm" in undersized_families(cfg, "judge", 8192)
    assert "gpt-oss" in undersized_families(cfg, "generator", 8192)
    # ...and the margin does not start excluding models with real headroom.
    assert undersized_families(cfg, "judge", 4000) == frozenset()


def test_the_token_estimate_counts_indic_script_far_harder_than_latin():
    """A BPE vocabulary trained on English runs ~1-2 chars/token on
    Devanagari, so chars/4 under-counts an Indic passage by 2-4x - on exactly
    this corpus, and the under-count is what lands the prompt at an 8k model."""
    latin = "the accused was convicted under section 302 of the code " * 20
    devanagari = "अभियुक्त को भारतीय दंड संहिता की धारा तीन सौ दो के अंतर्गत " * 20
    assert abs(len(latin) - len(devanagari)) < len(latin) * 0.25
    assert estimate_tokens(devanagari) > 2 * estimate_tokens(latin)


def test_the_context_estimate_counts_chat_template_overhead():
    """The overhead is a NUMBER, not a direction: `> 80 // 4` also holds at
    TEMPLATE_TOKENS_PER_MESSAGE = 0 (estimate_tokens rounds up per string), so
    the delta is what has to be pinned."""
    messages = [{"role": "system", "content": "a" * 40}, {"role": "user", "content": "b" * 40}]
    bare = sum(estimate_tokens(m["content"]) for m in messages)
    assert context_estimate(messages) - bare == 2 * TEMPLATE_TOKENS_PER_MESSAGE
    assert TEMPLATE_TOKENS_PER_MESSAGE > 0
    # An empty turn still costs its role markers.
    assert context_estimate([{"role": "user", "content": ""}]) == TEMPLATE_TOKENS_PER_MESSAGE
    assert context_estimate([]) == 0


def test_unkeyed_roles_names_the_role_and_the_env_vars(cfg, monkeypatch):
    """The fleet must refuse to start when a role it routes has no usable key
    - the alternative is a wave of tasks that never reaches a provider."""
    _unset(monkeypatch, "MISTRAL_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY")
    gaps = unkeyed_roles(cfg, ("generator", "judge"))
    assert set(gaps) == {"generator", "judge"}
    # The generator role routes to ONE provider since the 2026-08-18 demotion,
    # so its key list is exactly cerebras - and that key is now load-bearing
    # rather than merely preferred.
    assert gaps["generator"] == ("CEREBRAS_API_KEY",)
    assert "MISTRAL_API_KEY" in gaps["judge"]
    # One key is enough to make a role usable: the rest is failover.
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")
    assert set(unkeyed_roles(cfg, ("generator", "judge"))) == {"generator"}
    monkeypatch.setenv("CEREBRAS_API_KEY", "sk-test")
    assert unkeyed_roles(cfg, ("generator", "judge")) == {}
    # ...and a role whose every provider is still unkeyed keeps reporting.
    assert unkeyed_roles(cfg, ("probe",))["probe"] == ("GROQ_API_KEY",)


def test_the_shipped_config_has_no_fatal_judge_hole_left(cfg, keys):
    """R2-C3 CLOSED. The hole was: long rows route to magistral, family
    separation removes mistral, the 8k glm judge is out on length, slot A takes
    qwen and slot B has NOTHING - so the row parks having already paid for
    judge A. The openai backstop is the fourth family that fills slot B.

    Pinned as "no fatal gap", not as "no gap": the tiebreak holes are still
    there and still only warn, which is what makes this a closure rather than
    a config that stopped being checked."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    cfg = cfg_with_two_generator_families(cfg)
    gaps = pool_gaps(cfg, needed_tokens=worst_case_judge_tokens(cfg))
    assert [g for g in gaps if g.role == "judge"] == []
    assert [g for g in gaps if g.fatal] == []

    # ...and it is the openai refs doing it, at the size that used to empty the
    # slot. Slot B on a mistral row lands on gpt-oss, and the walk that finds it
    # is the Router's own.
    router = Router(cfg)
    needed = worst_case_judge_tokens(cfg)
    too_small = undersized_families(cfg, "judge", needed)
    slot_a = next(router.eligible_refs("judge", exclude_families=frozenset({"mistral"}) | too_small))
    slot_b = next(
        router.eligible_refs(
            "judge", exclude_families=frozenset({"mistral", "qwen"}) | too_small
        )
    )
    assert slot_a == GROQ_JUDGE
    assert slot_b == ModelRef("openai", "gpt-5-mini")

    # The tiebreak holes remain, for BOTH generator families, and remain
    # survivable: judge.py decides on the two judges it has.
    tiebreak_gaps = [g for g in gaps if g.role == "tiebreak"]
    assert {g.generator_family for g in tiebreak_gaps} == {"gpt-oss", "mistral"}
    assert all(not g.fatal for g in tiebreak_gaps)

    # ...and the hole really is closed by the ADDITION, not by the check going
    # quiet: take the backstop back out and the old gap is reported verbatim.
    before = pool_gaps(
        cfg_without_the_paid_judges(cfg), needed_tokens=worst_case_judge_tokens(cfg)
    )
    judge_gaps = [g for g in before if g.role == "judge"]
    assert [(g.generator_family, g.slot) for g in judge_gaps] == [("mistral", "b")]
    assert judge_gaps[0].fatal is True
    assert "a mistral generation of" in judge_gaps[0].detail
    assert "minus ['qwen'] already used" in judge_gaps[0].detail


def test_the_shipped_openai_models_send_no_temperature_and_declare_no_daily_cap(cfg):
    """Two decisions that are invisible until they are wrong.

    No `temperature`: the gpt-5 family 400s on any value but the default, and a
    400 with no context marker ABORTS the call rather than failing over. The
    quirk strips it as well; this is the config's half, and the half that keeps
    the payload clean when no per-call temperature is sent at all.

    No `tpd`/`rpd`: the operator chose UNCAPPED on 2026-08-15, and an absent key
    is how "unlimited" is spelt here. Pinned as a MEANING and not a spelling -
    the ledger's own reader is asked - because an absent number reads like an
    oversight, and the obvious repair for an oversight is to invent one. Any
    spend brake for this provider lives server-side, not in this file."""
    from tuned.data.store import _cap

    provider, _ = cfg.model_for(ModelRef("openai", "gpt-5-mini"))
    assert provider.quirks == ("openai",)
    assert provider.api_key_env == "OPENAI_API_KEY"
    models = {m.id: m for m in provider.models}
    assert set(models) == {"gpt-5-mini", "gpt-5-nano"}

    judge_bar = required_context(worst_case_judge_tokens(cfg))
    for model in models.values():
        assert "temperature" not in model.params, model.id
        for cap in ("tpd", "rpd"):
            assert cap not in model.limits, model.id
            assert _cap(model.limits, cap) == float("inf"), model.id
        # The rate limits that ARE declared are real per-minute limits, not
        # brakes wearing a limit's name.
        assert (model.limits["rpm"], model.limits["tpm"]) == (500, 200000), model.id
        # ...and each clears the threshold the preflight enforces, which is
        # what makes it a judge rather than another 16k candidate.
        assert model.limits["max_context"] >= judge_bar, model.id
        assert model.family == "gpt-oss", model.id


def test_the_paid_backstop_is_last_in_both_routing_lists(cfg, keys):
    """Position IS the preference: Router.pick walks the role's list in order
    and takes the first eligible ref. Anywhere but last, a paid call gets made
    for a row a free judge would have taken - which is the whole of "uncapped
    does not mean preferred"."""
    paid = ("openai/gpt-5-mini", "openai/gpt-5-nano")
    router = Router(cfg)
    for role in ("judge", "tiebreak"):
        refs = getattr(cfg.routing, role)
        assert refs[-2:] == paid, role
        assert not any(r.startswith("openai/") for r in refs[:-2]), role
        # ...and the Router's own walk agrees, which is the half that decides
        # anything: the config list is only a preference if this is its order.
        walked = [f"{r.provider}/{r.model}" for r in router.eligible_refs(role)]
        assert tuple(walked[-2:]) == paid, role
        assert router.pick(role).ref.provider != "openai", role


def _with_fourth_judge(cfg, *, max_context: int, provider: str = "groq"):
    """The shipped config plus one fourth-family judge of a given size."""
    from dataclasses import replace

    extra = ModelCfg(
        id="big-judge",
        family="fourth",
        roles=("judge", "tiebreak"),
        limits={"rpm": 30, "tpm": 8000, "max_context": max_context, "max_output": 8192},
        params={"temperature": 0.2},
    )
    providers = tuple(
        replace(p, models=p.models + (extra,)) if p.name == provider else p for p in cfg.providers
    )
    return replace(
        cfg,
        providers=providers,
        routing=replace(
            cfg.routing,
            judge=cfg.routing.judge + (f"{provider}/big-judge",),
            tiebreak=cfg.routing.tiebreak + (f"{provider}/big-judge",),
        ),
    )


def test_a_fourth_judge_family_closes_the_pool_gap(cfg, keys):
    """The other half of the fix is one config line; this proves the preflight
    goes quiet the moment the operator adds it."""
    patched = _with_fourth_judge(cfg, max_context=131072)
    assert pool_gaps(patched, needed_tokens=worst_case_judge_tokens(patched)) == []


# --- R3-C2/C3: the preflight runs the spender's own sizing and eligibility --
#
# Both round-3 Criticals are one defect: pool_gaps used to MODEL what
# judge_slot does (its own prompt sizing in the gates' currency, its own
# eligibility walk with no key filter) instead of executing the same code.
# These pin the two halves back together.


def test_the_preflight_sizes_the_judge_prompt_in_the_routing_currency(cfg):
    """R3-C2. `length_band.total_max` is chars/4 BY DEFINITION (it is what
    gates.check_length_band compares against); judge_slot spends in the
    routing currency, which charges Indic script at 1.5 chars/token and every
    turn its template overhead. Sizing the pool in the gates' currency
    under-states the corpus this build is actually made of by ~2.3x."""
    gates_currency = cfg.build.length_band.total_max + DEFAULT_JUDGE_REPLY_TOKENS
    needed = worst_case_judge_tokens(cfg)
    assert needed > 2 * gates_currency

    # ...and it is not a second formula: the same render and the same
    # estimate judge_slot uses, over the largest row the length gate permits,
    # in the script that tokenizes hardest.
    material = "क" * (cfg.build.length_band.total_max * 4)
    assert needed == judge_needed_tokens(judge_messages(material, "", ""))
    # The Latin worst case is smaller, and still over the old number.
    latin = judge_needed_tokens(judge_messages("a" * (cfg.build.length_band.total_max * 4), "", ""))
    assert gates_currency < latin < needed


def test_a_16k_fourth_family_judge_is_a_fatal_pool_gap(cfg, keys):
    """Many free-tier candidates are 16k - which is ABOVE the >= 11520 the
    preflight used to print - and a 16k judge cannot hold the longest row this
    build makes. It has to be reported as the gap it is, before the fleet
    starts.

    Run against the pool WITHOUT the paid backstop, because that is the pool in
    which a fourth-family judge is the thing filling slot B. With the 400k
    openai judges in the list the slot is filled whatever the 16k model does,
    so the fixture would guarantee the null result and prove nothing about
    size."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    cfg = cfg_with_two_generator_families(cfg)
    size = 16384
    patched = _with_fourth_judge(cfg_without_the_paid_judges(cfg), max_context=size)
    needed = worst_case_judge_tokens(patched)
    assert "fourth" in undersized_families(patched, "judge", needed)
    fatal = [g for g in pool_gaps(patched, needed_tokens=needed) if g.fatal]
    assert fatal, "a 16k judge cannot hold the longest row the length gate passes"
    # It is a CONTEXT gap and not a KEY gap - the R3-C2/R3-C3 distinction, and
    # the one that decides whether --allow-pool-gaps may run the short rows.
    # (`"on context length" in detail` cannot fail: it is a constant in the
    # f-string, present whether or not any family is undersized.)
    assert (fatal[0].key_shaped, fatal[0].key_envs) == (False, ())
    assert fatal[0].unservable is False
    # Named as a family that is TOO SMALL, which is the conditional half of
    # the sentence. `"'fourth'" in detail` was not: every judge family is
    # listed unconditionally by the "the judge pool is [...]" segment of the
    # same f-string, so it held whether or not 'fourth' was undersized.
    assert "minus ['fourth', 'glm'] on context length" in fatal[0].detail
    # The old preflight sized this pool in the gates' currency and advised
    # exactly this, so a 16k model read as comfortably large and the fleet
    # started; what is enforced now is above it.
    old_advice = required_context(cfg.build.length_band.total_max + DEFAULT_JUDGE_REPLY_TOKENS)
    assert old_advice < size < required_context(needed)


def test_a_key_shaped_judge_gap_is_a_gap_at_every_row_size(cfg, monkeypatch):
    """R4-C1. --allow-pool-gaps justifies itself with "running short rows
    while a key is pending is a real choice". That is true of a CONTEXT gap
    and false of this one: eligible_refs skips an unkeyed family at EVERY
    size, so there is no subset of rows the override lets through safely."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    cfg = cfg_with_two_generator_families(cfg)
    for env in ("CEREBRAS_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.setenv(env, "sk-test")
    _unset(monkeypatch, "GROQ_API_KEY", "OPENAI_API_KEY")
    # The fourth-family judge the operator is sourcing, behind the key that
    # has not arrived - and qwen is behind the same one. The paid backstop is
    # unkeyed too: keyed, it fills every slot at every size and there is no
    # unservable gap left to classify.
    widened = _with_fourth_judge(cfg, max_context=131072)

    for needed in (500, 2000, worst_case_judge_tokens(widened)):
        gaps = [g for g in pool_gaps(widened, needed_tokens=needed) if g.fatal]
        # A magistral generation is the shape R3-C3 was found in: mistral is
        # the generator's own family, and both remaining judge families sit
        # behind the key that has not arrived.
        gap = next(g for g in gaps if g.generator_family == "mistral")
        assert gap.unservable is True, f"servable at needed={needed}"
        assert gap.key_shaped is True
        # BOTH pending keys are named: either one would fill the slot, and a
        # remedy that named only the first would send the operator after half
        # of what they have.
        assert gap.key_envs == ("GROQ_API_KEY", "OPENAI_API_KEY")
        assert "no row size is servable" in gap.detail

    # ...and the classification is per GAP, not per config: a gpt-oss row can
    # still be judged by mistral + glm, so at 2,000 tokens it has no gap at
    # all and at 23,729 its gap is one the override may legitimately cover.
    small = pool_gaps(widened, needed_tokens=2000)
    assert [g.generator_family for g in small if g.fatal] == ["mistral"]
    big = pool_gaps(widened, needed_tokens=worst_case_judge_tokens(widened))
    gpt_oss = next(g for g in big if g.generator_family == "gpt-oss" and g.fatal)
    assert (gpt_oss.key_shaped, gpt_oss.unservable) == (True, False)

    # ...and the same walk goes quiet the moment the key lands.
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    assert pool_gaps(widened, needed_tokens=worst_case_judge_tokens(widened)) == []


def test_a_context_shaped_judge_gap_has_row_sizes_the_pool_still_serves(cfg, keys):
    """The other side of R4-C1, and the reason the override exists at all: a
    16k fourth-family judge empties slot B only ABOVE a size, so short rows
    really are servable and running them is a real choice.

    Without the paid backstop, for the same reason as the 16k test above: a
    400k judge in the list fills slot B at every size, so there would be no
    context-shaped gap to have sizes on either side of."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    cfg = cfg_with_two_generator_families(cfg)
    sixteen_k = _with_fourth_judge(cfg_without_the_paid_judges(cfg), max_context=16384)
    assert [g for g in pool_gaps(sixteen_k, needed_tokens=2000) if g.fatal] == []
    assert [g for g in pool_gaps(sixteen_k, needed_tokens=8000) if g.fatal] == []
    fatal = [g for g in pool_gaps(sixteen_k, needed_tokens=20000) if g.fatal]
    assert fatal and fatal[0].unservable is False


def test_the_advice_the_preflight_prints_is_the_threshold_it_enforces(cfg, keys):
    """The gap detail names a max_context; a model of exactly that size must
    close EVERY gap it reports and one token smaller must not. Anything else
    is a preflight that tells the operator to buy the wrong model - and the
    number is judge-only if the tiebreak is sized apart from it, so a model of
    exactly the advised size closed the judge gap and opened a tiebreak
    warning."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    cfg = cfg_with_two_generator_families(cfg)
    gaps = pool_gaps(
        cfg,
        needed_tokens=worst_case_judge_tokens(cfg),
        tiebreak_needed_tokens=worst_case_judge_tokens(cfg, prompt_id=TIEBREAK_PROMPT_ID),
        needed_for_window=judge_sizer(cfg),
    )
    advised = {int(g.detail.split("max_context >= ")[1].split()[0]) for g in gaps}
    assert len(advised) == 1, "one gap, one number: the operator buys one model"
    advice = advised.pop()

    def remaining(max_context: int) -> list[PoolGap]:
        patched = _with_fourth_judge(cfg, max_context=max_context)
        return pool_gaps(
            patched,
            needed_tokens=worst_case_judge_tokens(patched),
            tiebreak_needed_tokens=worst_case_judge_tokens(patched, prompt_id=TIEBREAK_PROMPT_ID),
            needed_for_window=judge_sizer(patched),
        )

    assert remaining(advice) == []  # every gap, including the tiebreak warning
    assert remaining(advice - 1) != []


def _advice(config, **kw):
    gaps = pool_gaps(
        config,
        needed_tokens=worst_case_judge_tokens(config),
        tiebreak_needed_tokens=worst_case_judge_tokens(config, prompt_id=TIEBREAK_PROMPT_ID),
        needed_for_window=judge_sizer(config),
        servable_floor_tokens=min_judge_tokens(config),
        **kw,
    )
    return {int(g.detail.split("max_context >= ")[1].split()[0]) for g in gaps}


def test_the_advice_never_falls_below_the_flat_worst_case(cfg, monkeypatch):
    """The advice is a PURCHASE, and the per-family narrowing made it a
    function of which keys happened to be set that minute. With MISTRAL_API_KEY
    pending, the 40k generator is not eligible and the only family left is the
    8k one, whose judge check is narrowed to 15,104 - so the shipped config
    asked for 18,880. The operator buys exactly that, adds it, the fleet
    starts; the key lands, the 40k generator becomes eligible, and the same
    config now wants 29,661 and refuses. Being told to buy a bigger model
    costs nothing; being sent shopping twice costs a purchase."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    cfg = cfg_with_two_generator_families(cfg)
    flat = required_context(worst_case_judge_tokens(cfg))

    def advice_with(*set_envs):
        _unset(monkeypatch, "CEREBRAS_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY")
        for env in set_envs:
            monkeypatch.setenv(env, "sk-test")
        return _advice(cfg)

    fully_keyed = advice_with("CEREBRAS_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY")
    assert fully_keyed == {flat}
    # The pending-key state: a real gap, narrowed to a real 15,104 check
    # (which is right - that IS the largest row an 8k generator can make) and
    # still advising the number that survives the key landing.
    pending = advice_with("CEREBRAS_API_KEY", "GROQ_API_KEY")
    assert judge_sizer(cfg)(8192, "judge") < worst_case_judge_tokens(cfg)
    assert pending == fully_keyed
    # ...in every other partially-keyed state too. Whichever keys are set, the
    # operator is told one number.
    for envs in (("CEREBRAS_API_KEY",), ("MISTRAL_API_KEY",),
                 ("CEREBRAS_API_KEY", "MISTRAL_API_KEY"),
                 ("GROQ_API_KEY", "MISTRAL_API_KEY")):
        assert advice_with(*envs) == fully_keyed, envs


def test_unservable_is_asked_at_the_smallest_call_this_build_can_make(cfg, keys):
    """`unservable` is the one fact --allow-pool-gaps cannot override, and it
    was asked with the context filter removed ENTIRELY - i.e. at size zero, a
    size the length band cannot produce. Every judge call renders the judge
    template and asks for a reply, and a judged row carries at least
    think_min + answer_min on top, so a judge under that floor serves no row
    at any length while reading here as the slot the short rows would use."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    cfg = cfg_with_two_generator_families(cfg)
    floor = min_judge_tokens(cfg)
    tiny = 3000
    assert tiny < required_context(floor), "the fixture has to be under the real floor"
    # Without the paid backstop: an unbounded judge in the list serves the
    # short rows too, so nothing would be unservable and the rule would have
    # no case.
    patched = cfg_with_context(
        cfg_without_the_paid_judges(cfg), family="glm", role="judge", max_context=tiny
    )

    def mistral_gap(config):
        gaps = pool_gaps(
            config,
            needed_tokens=worst_case_judge_tokens(config),
            servable_floor_tokens=floor,
        )
        return next(g for g in gaps if (g.generator_family, g.slot) == ("mistral", "b"))

    gap = mistral_gap(patched)
    assert gap.unservable is True
    assert "no row size is servable" in gap.detail
    # The floor is what decides it: glm at 8,192 really can serve the short
    # rows, and that gap stays overridable.
    assert mistral_gap(cfg_without_the_paid_judges(cfg)).unservable is False

    # ...and the consequence, at the flag: an unservable gap is a refusal
    # whatever --allow-pool-gaps says, so the fleet cannot be started against
    # a judge pool with nothing in it for any row.
    refusals, _ = preflight_messages(patched, ("judge",), allow_pool_gaps=True)
    assert any("no row size is servable" in line for line in refusals)


def _with_tiebreak_family(cfg, *, api_key_env: str, max_context: int = 131072):
    """The shipped config plus a fifth-family TIEBREAK behind its own key."""
    from dataclasses import replace

    model = ModelCfg(
        id="fifth-tiebreak",
        family="fifth",
        roles=("tiebreak",),
        limits={"rpm": 30, "tpm": 8000, "max_context": max_context, "max_output": 4096},
        params={},
    )
    provider = ProviderCfg(
        name="fifthparty",
        base_url="https://example.test/v1",
        api_key_env=api_key_env,
        quirks=(),
        models=(model,),
    )
    return replace(
        cfg,
        providers=cfg.providers + (provider,),
        routing=replace(
            cfg.routing, tiebreak=cfg.routing.tiebreak + ("fifthparty/fifth-tiebreak",)
        ),
    )


def test_a_tiebreak_gap_names_the_key_that_would_fill_it(cfg, keys, monkeypatch):
    """The judge side of this is pinned; the tiebreak side was not, so the
    remedy could quietly stop naming the key that would close the gap and go
    on advising a purchase the operator does not need to make."""
    patched = _with_tiebreak_family(cfg, api_key_env="FIFTHPARTY_API_KEY")
    _unset(monkeypatch, "FIFTHPARTY_API_KEY")
    gap = next(
        g
        for g in pool_gaps(patched, needed_tokens=worst_case_judge_tokens(patched))
        if g.role == "tiebreak"
    )
    assert gap.key_envs == ("FIFTHPARTY_API_KEY",)
    assert "set FIFTHPARTY_API_KEY, or add one tiebreak" in gap.detail
    # ...and the key landing is what closes it, not a purchase.
    monkeypatch.setenv("FIFTHPARTY_API_KEY", "sk-test")
    assert [
        g for g in pool_gaps(patched, needed_tokens=worst_case_judge_tokens(patched))
        if g.role == "tiebreak"
    ] == []


def _with_extra_generator(cfg, *, family: str, api_key_env: str, max_context: int = 131072):
    """The shipped config plus a generator family behind its own API key."""
    from dataclasses import replace

    model = ModelCfg(
        id="extra-gen",
        family=family,
        roles=("generator",),
        limits={"rpm": 30, "tpm": 8000, "max_context": max_context, "max_output": 4096},
        params={},
    )
    provider = ProviderCfg(
        name="fourthparty",
        base_url="https://example.test/v1",
        api_key_env=api_key_env,
        quirks=(),
        models=(model,),
    )
    return replace(
        cfg,
        providers=cfg.providers + (provider,),
        routing=replace(
            cfg.routing, generator=cfg.routing.generator + ("fourthparty/extra-gen",)
        ),
    )


def _with_judge_model(cfg, *, family: str, model_id: str, max_context: int | None):
    """The shipped config plus one more judge in an EXISTING family."""
    from dataclasses import replace

    limits = {"rpm": 30, "tpm": 8000, "max_output": 4096}
    if max_context is not None:
        limits["max_context"] = max_context
    extra = ModelCfg(
        id=model_id, family=family, roles=("judge",), limits=limits, params={}
    )
    providers = tuple(
        replace(p, models=p.models + (extra,)) if p.name == "cerebras" else p
        for p in cfg.providers
    )
    return replace(
        cfg,
        providers=providers,
        routing=replace(cfg.routing, judge=cfg.routing.judge + (f"cerebras/{model_id}",)),
    )


def test_pool_gaps_walks_the_generator_role_through_the_routers_own_filter(cfg, keys, monkeypatch):
    """The R3 unification was one-sided: judge and tiebreak went through
    Router.eligible_refs while the generator role stayed a raw cfg.routing_refs
    walk. A generator family behind a key that has not arrived produces no rows
    at all, so a judge gap reported for it is a refusal about a combination
    that cannot occur - and a spurious refusal is not free, it is the operator
    reaching for --allow-pool-gaps.

    Run without the paid backstop, so that the gap the unkeyed family would
    contribute is a FATAL one and the rule is tested where it costs: with the
    400k judges in the list every judge slot fills for every family, and the
    only thing an unkeyed generator could add is another tiebreak warning."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    cfg = cfg_with_two_generator_families(cfg)
    patched = _with_extra_generator(
        cfg_without_the_paid_judges(cfg), family="qwen", api_key_env="FOURTHPARTY_API_KEY"
    )
    _unset(monkeypatch, "FOURTHPARTY_API_KEY")
    needed = worst_case_judge_tokens(patched)

    assert [(g.generator_family, g.slot) for g in pool_gaps(patched, needed_tokens=needed)] == [
        ("gpt-oss", "tiebreak"),
        ("mistral", "b"),
    ]

    # ...and the moment that key lands the family is real, and so is its gap:
    # a qwen generation may not be judged by the qwen judge, and glm is 8k.
    monkeypatch.setenv("FOURTHPARTY_API_KEY", "sk-test")
    assert ("qwen", "b") in [
        (g.generator_family, g.slot) for g in pool_gaps(patched, needed_tokens=needed) if g.fatal
    ]


def test_each_generator_family_is_checked_at_the_size_its_own_window_permits(cfg, keys):
    """The 8k generator cannot be handed the longest row the length band
    permits - `undersized_families` diverts it long before - so checking its
    judge slots at that length invents a gap. Sized at what its own window
    permits the same pool serves it, while the 40k generator, which really can
    produce that row, still has the gap the config TODO is about."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    cfg = cfg_with_two_generator_families(cfg)
    # 26k, not 20k: correcting the reply conversion (2026-08-18, review round
    # 2 / I6) raised the 8k-window check from 15,104 to 19,104 tokens, i.e.
    # 23,880 of required context, so a 20k judge no longer demonstrates the
    # narrowing - it is simply too small either way. The flat worst case is
    # untouched at 29,661.
    patched = cfg_with_context(cfg, family="qwen", role="judge", max_context=26000)
    flat = worst_case_judge_tokens(patched)
    sizer = judge_sizer(patched)
    assert sizer(8192, "judge") < flat == sizer(None, "judge")

    def fatal(**kw):
        return {
            (g.generator_family, g.slot)
            for g in pool_gaps(patched, needed_tokens=flat, **kw)
            if g.fatal
        }

    # Both generator families lose slot B at the flat size: qwen is now too
    # small and glm always was, so the pool is mistral + the gpt-oss backstop
    # and each family's own is one of them.
    assert fatal() == {("gpt-oss", "b"), ("mistral", "b")}
    # Sized at what its own 8k window permits, the gpt-oss family's judges fit
    # again and its refusal disappears; the 40k mistral family really can make
    # the long row, so its gap stays.
    assert fatal(needed_for_window=sizer) == {("mistral", "b")}


def test_the_family_window_bound_never_sizes_above_the_flat_worst_case(cfg, keys):
    """`min` with the flat number is load-bearing: the hook can only ever make
    a check smaller, so a caller that passes a wrong one cannot widen what the
    preflight checks behind the operator's back."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    cfg = cfg_with_two_generator_families(cfg)
    huge = judge_tokens_for_generator_window(cfg, 10**9)
    assert huge == worst_case_judge_tokens(cfg)
    # A window too small to hold the reply allowance contributes NO material -
    # the term clamps at zero rather than going negative and quietly buying
    # back some of the candidate's budget.
    # REPLY_BUDGET_CHARS_PER_TOKEN, not CHARS_PER_TOKEN_LATIN (2026-08-18,
    # review round 2 / I6): this mirrors the conversion inside
    # judge_tokens_for_generator_window, which was under-counting the reply
    # at 4.0 while reply_over_budget had already been corrected to the
    # measured 5.5. The two have to use one constant or the sizing budgets
    # for a smaller candidate than enforcement permits.
    reply_chars = max_output_tokens(cfg) * REPLY_BUDGET_CHARS_PER_TOKEN
    reply_only = judge_needed_tokens(judge_messages(WORST_CASE_CHAR * int(reply_chars), "", ""))
    assert judge_tokens_for_generator_window(cfg, max_output_tokens(cfg)) == reply_only
    # ...and a hook that answers nonsense narrows nothing: the gap set is the
    # flat one, not a wider check nobody asked for.
    flat = pool_gaps(cfg, needed_tokens=worst_case_judge_tokens(cfg))
    assert (
        pool_gaps(
            cfg,
            needed_tokens=worst_case_judge_tokens(cfg),
            needed_for_window=lambda window, role: 10**9,
        )
        == flat
    )


def test_the_tiebreak_slot_is_sized_by_its_own_prompt(cfg, keys):
    """`tiebreak_needed_tokens` defaults to the judge's number, and the two
    prompts are not the same prompt. The difference is four tokens on this
    config, so the only way to pin the plumbing is a model that sits between
    them: it must clear the judge's requirement and fail the tiebreak's."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    cfg = cfg_with_two_generator_families(cfg)
    judge_needed = worst_case_judge_tokens(cfg)
    tiebreak_needed = worst_case_judge_tokens(cfg, prompt_id=TIEBREAK_PROMPT_ID)
    assert required_context(judge_needed) < required_context(tiebreak_needed)

    patched = cfg_with_split_pools(
        cfg, judge_context=131072, tiebreak_context=required_context(tiebreak_needed) - 1
    )
    sized = pool_gaps(
        patched, needed_tokens=judge_needed, tiebreak_needed_tokens=tiebreak_needed
    )
    # Both generator families, because the paid backstop is spent on judge slot
    # B for a mistral row and is therefore gone from its tiebreak too. What the
    # test turns on is unchanged: every one of these flips on the four tokens
    # between the judge prompt and the tiebreak prompt.
    assert [(g.role, g.generator_family) for g in sized] == [
        ("tiebreak", "gpt-oss"),
        ("tiebreak", "mistral"),
    ]
    # Sized by the JUDGE's number instead, that same model reads as big enough.
    assert pool_gaps(patched, needed_tokens=judge_needed) == []
    # ...and one token more really does close it.
    opened = cfg_with_split_pools(
        cfg, judge_context=131072, tiebreak_context=required_context(tiebreak_needed)
    )
    assert pool_gaps(
        opened, needed_tokens=judge_needed, tiebreak_needed_tokens=tiebreak_needed
    ) == []


def test_blocking_key_envs_names_only_keys_that_would_change_something(cfg, keys, monkeypatch):
    """Its entire purpose: naming the env var of a model that is excluded on
    family or on context length would send the operator after a key that opens
    nothing, in the middle of a launch where keys are the scarce thing."""
    _unset(monkeypatch, "CEREBRAS_API_KEY")  # the glm judge lives there

    assert _blocking_key_envs(cfg, "judge", frozenset()) == ("CEREBRAS_API_KEY",)
    assert _blocking_key_envs(cfg, "judge", frozenset({"glm"})) == ()
    # ...and a keyed provider is never named at all.
    monkeypatch.setenv("CEREBRAS_API_KEY", "sk-test")
    assert _blocking_key_envs(cfg, "judge", frozenset()) == ()


def test_undersized_families_needs_every_model_in_the_family_to_be_too_small(cfg):
    """EVERY, not ANY: a mixed-size family keeps its large model and the
    preference order decides between them. The distinction is invisible on
    today's config because no family has two models in one role - and the
    config TODO asks the operator to ADD one, at which point this rule decides
    whether the family they added is usable at all."""
    assert "glm" in undersized_families(cfg, "judge", 20000)
    mixed = _with_judge_model(cfg, family="glm", model_id="glm-big", max_context=131072)
    assert "glm" not in undersized_families(mixed, "judge", 20000)
    # ...and a second SMALL model does not rescue it.
    small = _with_judge_model(cfg, family="glm", model_id="glm-small", max_context=8192)
    assert "glm" in undersized_families(small, "judge", 20000)


def test_a_judge_that_declares_no_context_limit_is_assumed_to_fit(cfg):
    """An absent limit is unknown, not small. Reading it as "too small" would
    silently retire a working provider from every long row; reading it as
    "fits" costs one 400 that providers.py fails over."""
    unknown = _with_judge_model(cfg, family="glm", model_id="glm-unknown", max_context=None)
    assert "glm" not in undersized_families(unknown, "judge", 10**6)


def _with_generator_models(cfg, *, caps, family: str = "extra"):
    """The shipped config plus `len(caps)` generators in one family.

    In the order given, because that is the order `_generator_windows` walks
    and the rule has to hold whichever size the operator happens to add first.
    A cap of None declares no `max_context` at all.
    """
    from dataclasses import replace

    def model(ix, cap):
        limits = {"rpm": 30, "tpm": 8000, "max_output": 4096}
        if cap is not None:
            limits["max_context"] = cap
        return ModelCfg(
            id=f"gen-{family}-{ix}", family=family, roles=("generator",), limits=limits, params={}
        )

    extra = tuple(model(ix, cap) for ix, cap in enumerate(caps))
    providers = tuple(
        replace(p, models=p.models + extra) if p.name == "cerebras" else p for p in cfg.providers
    )
    return replace(
        cfg,
        providers=providers,
        routing=replace(
            cfg.routing,
            generator=cfg.routing.generator + tuple(f"cerebras/{m.id}" for m in extra),
        ),
    )


def test_a_generator_familys_window_is_the_largest_model_it_offers(cfg, keys):
    """The twin of the EVERY-vs-ANY rule above, and the INPUT to the narrowed
    judge check: `pool_gaps` sizes each generator family's judge slots at one
    window, and `undersized_families` excludes a family only when EVERY model
    in it is too small - so the window that decides the check has to be the
    LARGEST the family offers, or the check is sized for a row the family can
    still be handed.

    Invisible on today's config (no generator family has two models) and the
    config TODO asks the operator to ADD models, which is when it decides."""

    def window(*caps):
        patched = _with_generator_models(cfg, caps=caps)
        return _generator_windows(patched, Router(patched))["extra"]

    # Two declared windows: the largest, and the order they were configured
    # in cannot matter.
    assert window(8192, 131072) == 131072
    assert window(131072, 8192) == 131072
    # An undeclared limit is UNKNOWN, not small - `undersized_families` lets
    # that model through at every size, so the family has no ceiling to cap
    # its check with. Again in both orders: a later uncapped model has to
    # CLEAR a cap an earlier one set, not be swallowed by it.
    assert window(8192, None) is None
    assert window(None, 8192) is None
    # ...and alone it is None, never 0. Zero is a real window here: it would
    # size the family's judge check at the reply allowance alone and clear a
    # judge that cannot hold a single one of its rows.
    assert window(None) is None


def test_a_bigger_model_in_a_generator_family_raises_the_size_its_judges_are_checked_at(cfg, keys):
    """What the rule above costs when it is wrong, at the only place it is
    spent. The operator adds a 128k variant beside the 8k one in an existing
    generator family; the family can now produce the longest row the band
    permits, so its judges must be checked at that length. Sized at the
    SMALLEST model instead, the check runs at 19,104 (required 23,880), a 26k
    judge clears, and the long row routes to the 128k generator and parks at
    slot B having already paid for judge A.

    The probe window moved 20k -> 26k on 2026-08-18 (review round 2, I6):
    the reply half of this sizing was being converted at 4.0 chars/token on a
    premise the pilot measured false, and correcting it to the measured 5.5
    raised the 8k-window check from 15,104 to 19,104 tokens. The FLAT
    worst case is unchanged at 23,729 (29,661 of context), because that end
    is bounded by the length band, whose chars//4 definition really is 4.0."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    cfg = cfg_with_two_generator_families(cfg)
    small = cfg_with_context(cfg, family="qwen", role="judge", max_context=26000)
    sizer, flat = judge_sizer(small), worst_case_judge_tokens(small)

    def fatal(config):
        return {
            (g.generator_family, g.slot)
            for g in pool_gaps(config, needed_tokens=flat, needed_for_window=sizer)
            if g.fatal
        }

    # As shipped, the gpt-oss family really is 8k-only and the narrowing is
    # right: at 19,104 the 26k judge holds every row it can produce.
    assert required_context(sizer(8192, "judge")) < 26000 < required_context(flat)
    assert ("gpt-oss", "b") not in fatal(small)
    # Add the big sibling and the same pool no longer serves that family.
    mixed = _with_generator_models(small, caps=(131072,), family="gpt-oss")
    assert ("gpt-oss", "b") in fatal(mixed)


def test_pool_gaps_applies_the_routers_own_key_filter(cfg, monkeypatch):
    """R3-C3. Router.eligible skips an unkeyed ref as "missing-key"; a
    preflight that walks routing_refs without that filter reports a pool it
    cannot call. unkeyed_roles does not cover it - it passes a role as soon as
    ONE ref is keyed, which is the right question for "can this role call at
    all" and the wrong one for "can slot B be filled". Keys arrive piecemeal,
    so a partially-keyed start is the likely first real launch."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    cfg = cfg_with_two_generator_families(cfg)
    _unset(monkeypatch, "GROQ_API_KEY")
    for env in ("MISTRAL_API_KEY", "CEREBRAS_API_KEY"):
        monkeypatch.setenv(env, "sk-test")
    # Small rows on purpose: nothing is undersized at this length, so a
    # missing key is the only thing that can empty a slot.
    assert unkeyed_roles(cfg, ("judge",)) == {}
    assert undersized_families(cfg, "judge", 4000) == frozenset()

    fatal = [g for g in pool_gaps(cfg, needed_tokens=4000) if g.fatal]
    assert fatal
    assert any("GROQ_API_KEY" in g.detail for g in fatal)

    # ...and the same walk goes quiet once the key lands.
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    assert [g for g in pool_gaps(cfg, needed_tokens=4000) if g.fatal] == []


def test_a_key_removed_mid_flight_is_still_a_missing_key(monkeypatch):
    """The Router checked the key and the client found it gone - a narrow
    race, but its error carried no `skipped` at all, so the worker read it as
    a provider failure and burnt the row's attempts into `rejected`."""
    _unset(monkeypatch, "TUNED_TEST_KEY")

    def handler(request):  # pragma: no cover - the call never leaves
        return httpx.Response(200, json=_body())

    with pytest.raises(ProviderError) as excinfo:
        _complete(_client(handler), _request())
    assert excinfo.value.skipped == frozenset({"missing-key"})


def test_an_empty_role_list_is_named_rather_than_silent(cfg, keys):
    """Nothing was tried and nothing was SKIPPED, so the error carried an
    empty reason set and the worker could not tell it from a provider that
    refused it - the row burnt three attempts into `rejected` instead of
    parking where --reopen can reach it."""
    from dataclasses import replace

    patched = replace(cfg, routing=replace(cfg.routing, judge=()))
    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(_router(patched).complete("judge", [{"role": "user", "content": "hi"}]))
    assert excinfo.value.skipped == frozenset({"empty-role-list"})
    assert excinfo.value.retryable is False


def _divert_point(cfg, role: str, family: str, reply_tokens: int) -> int:
    """The prompt size at which `family` stops being routable for `role`."""
    size = 1
    while family not in undersized_families(cfg, role, size + reply_tokens):
        size += 1
    return size


def test_the_config_block_quotes_the_numbers_the_code_enforces(cfg, keys):
    """That block is the operator's spec for the judge pool, and its arithmetic
    was pre-margin and ~40% high (it said ~4.2k where the real divert point is
    2555, and ~7.2k where slot B really dies at 5531) while pool_gaps printed a
    third number. All of them come from here now - and the advice is read out
    of the preflight's own output, not recomputed beside it, because that
    string is what the operator shops against.

    Two numbers, not one, and the difference is easy to get wrong. What is
    still ADVISED is the tiebreak's threshold, because the tiebreak is the only
    gap the shipped pool still has and its prompt is a little longer than the
    judge's. What the closed judge gap was measured against is the judge's, and
    the block has to keep it: it is the bar any replacement judge must clear."""
    from tuned.data.generate import preflight_messages

    text = DATA_CONFIG.read_text(encoding="utf-8")
    refusals, warnings = preflight_messages(cfg, ("generator",))
    assert refusals == [], "the judge gap is closed; nothing here should refuse"
    advised = {int(line.split("max_context >= ")[1].split()[0]) for line in warnings}
    assert len(advised) == 1, "one number: the operator buys one model"
    required = advised.pop()

    # THE ADVISED NUMBER DROPPED 29,666 -> 29,661 ON 2026-08-18, and it is the
    # documented behaviour rather than drift. `advice` is
    # max(required_context(needed_tokens), each gap's own requirement), and the
    # gap that used to carry the larger tiebreak-prompt number was a MISTRAL
    # generation. With mistral demoted to judge-only that row cannot be
    # produced, so the flat judge floor is what remains - which is exactly the
    # "never falls below required_context(needed_tokens)" rule the advice was
    # given so the operator is not quoted a different number on Tuesday than on
    # Wednesday.
    judge_required = required_context(worst_case_judge_tokens(cfg))
    assert required == judge_required
    assert f"max_context >= {required}" in text
    assert f"{required:,}" in text
    # The tiebreak threshold is no longer ADVISED, but the block still has to
    # quote it: it is the bar a tiebreak replacement must clear, and it is the
    # larger of the two because the tiebreak prompt is longer than the judge's.
    tiebreak_required = required_context(
        worst_case_judge_tokens(cfg, prompt_id=TIEBREAK_PROMPT_ID)
    )
    assert tiebreak_required > judge_required
    assert f"max_context >= {tiebreak_required}" in text
    assert f"{tiebreak_required:,}" in text
    # ...and the two thresholds the block explains the (now closed) gap with.
    from tuned.data.generate import max_output_tokens

    assert f"{_divert_point(cfg, 'generator', 'gpt-oss', max_output_tokens(cfg)):,}" in text
    assert f"{_divert_point(cfg, 'judge', 'glm', DEFAULT_JUDGE_REPLY_TOKENS):,}" in text
    # The size a 16k candidate would fail is stated, because most free-tier
    # candidates are 16k.
    assert "16k" in text


def test_eligible_refs_is_the_filter_eligible_itself_uses(cfg, keys):
    """The preflight walks refs and the spender walks clients, but there is
    one filter: `eligible` is `eligible_refs` plus client construction. A
    second implementation is what let the preflight report a pool the Router
    would not have called."""
    router = _router(cfg)
    walked = list(router.eligible_refs("judge"))
    assert walked == [routed.ref for routed in router.eligible("judge")]
    # ...including the reasons, in the Router's own order.
    seen: set[str] = set()
    assert list(router.eligible_refs("judge", exclude_families=frozenset({"mistral"}),
                                    skipped=seen)) == walked[1:]
    assert seen == {"family-excluded"}


# --------------------------------------------------------------------------
# The mistral quirk: Mistral Small 4's typed content chunks.
# --------------------------------------------------------------------------
# Captured from a live probe on 2026-08-18, byte-shape included: the thinking
# chunk's body is a NESTED LIST of typed parts, not a string, and the field is
# named for the chunk's own type. `reasoning` and `reasoning_content` are both
# absent - the trace is only ever inside `content`.
SMALL4_REASONING_REPLY = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "tool_calls": None,
                "content": [
                    {
                        "type": "thinking",
                        "closed": True,
                        "thinking": [
                            {"type": "text", "text": "I need to decide whether "},
                            {"type": "text", "text": "the intent element is made out."},
                        ],
                    },
                    {"type": "text", "text": "No, that is not theft."},
                ],
            }
        }
    ]
}

# The same model called WITHOUT the parameter: a plain string and no trace.
SMALL4_PLAIN_REPLY = {
    "choices": [{"message": {"role": "assistant", "content": "No, that is not theft."}}]
}


def test_the_mistral_quirk_flattens_the_nested_thinking_chunk():
    text, reasoning = QUIRKS["mistral"].response_hook(SMALL4_REASONING_REPLY)
    assert text == "No, that is not theft."
    # Flattened, in order, with no list repr leaking into the trace.
    assert reasoning == "I need to decide whether the intent element is made out."
    assert "{" not in reasoning and "[" not in reasoning


def test_the_mistral_quirk_reports_a_plain_reply_as_having_no_trace():
    """The opt-in half of the contract, and the reason the hook must not invent
    a trace: a generator called without `reasoning_effort` genuinely produced
    none, and generate.py's no-reasoning-channel park is the correct outcome.
    """
    text, reasoning = QUIRKS["mistral"].response_hook(SMALL4_PLAIN_REPLY)
    assert text == "No, that is not theft."
    assert reasoning is None


def test_the_mistral_quirk_still_reads_an_ordinary_reasoning_field():
    """Shape-tolerant: if the API ever moves the trace back onto its own field,
    the hook keeps working rather than silently dropping it."""
    data = {"choices": [{"message": {"content": "answer", "reasoning": "trace"}}]}
    assert QUIRKS["mistral"].response_hook(data) == ("answer", "trace")


def test_the_default_hook_would_have_mangled_the_small4_shape():
    """Why this quirk exists at all. The default hook returns `content`
    untouched, so the whole typed-chunk list - thinking included - would be
    handed downstream as the answer, and `reasoning` would be None: the trace
    would be both lost AND pasted into the answer."""
    text, reasoning = QUIRKS["default"].response_hook(SMALL4_REASONING_REPLY)
    assert reasoning is None
    assert not isinstance(text, str)


# --------------------------------------------------------------------------
# role_params: one model, two roles, two sets of sampling.
# --------------------------------------------------------------------------

def _params_for_role(cfg, ref, role):
    """The sampling params a call in `role` would actually put on the wire."""
    client = _router(cfg).routed(ref).client
    payload = client.build_payload(
        ChatRequest(messages=({"role": "user", "content": "x"},), ref=ref, role=role)
    )
    return {k: v for k, v in payload.items() if k in ("temperature", "top_p")}


def test_role_params_send_generator_sampling_to_generator_calls_only(cfg, keys):
    """N1. One model, two roles, two temperatures - against the REAL config.

    While mistral generated and judged from two config blocks, the sampling
    difference was free. Merging them for Mistral Small 4 put the generator's
    `temperature: 0.7, top_p: 0.95` into `params`, which
    ModelClient.build_payload merges into EVERY payload - and judge.py sends no
    per-call params at all, so slot A would have scored at 0.7/0.95 while every
    test that looked at a judge temperature was reading a hard-coded fake.

    Before (both roles)      temperature 0.7,  top_p 0.95
    After  generator         temperature 0.7,  top_p 0.95
    After  judge             temperature 0.2,  top_p absent

    which is exactly what the two deleted blocks carried.

    SPLIT IN TWO ON 2026-08-18: the shipped config demoted this model to
    judge-only, so the generator half is now asserted against the two-family
    fixture. Keeping it is the point - the mechanism is what stops a judge
    inheriting generator sampling if the model is ever re-promoted, and a test
    that only ever saw one role would not notice it rotting.
    """
    # Shipped reality: one role, and it is the judge's temperature.
    assert _params_for_role(cfg, MISTRAL_JUDGE, "judge") == {"temperature": 0.2}
    # No role layer at all -> the model's own defaults, which are empty here.
    assert _params_for_role(cfg, MISTRAL_JUDGE, None) == {}

    promoted = cfg_with_two_generator_families(cfg)
    assert _params_for_role(promoted, MISTRAL_JUDGE, "generator") == {
        "temperature": 0.7,
        "top_p": 0.95,
    }
    assert _params_for_role(promoted, MISTRAL_JUDGE, "judge") == {"temperature": 0.2}


def test_role_params_do_not_leak_between_models(cfg, keys):
    """A model with no role_params is unaffected: cerebras keeps its single
    configured temperature whichever role asks for it."""
    ref = ModelRef("cerebras", "gpt-oss-120b")
    assert _params_for_role(cfg, ref, "generator") == _params_for_role(cfg, ref, "judge")


def test_a_per_call_param_still_beats_the_role_layer(cfg, keys):
    """Precedence is model.params < role_params[role] < per-call params. The
    caller has to stay able to override, or judge.py's own temperature would be
    silently ignored the day a role entry is added for its model."""
    client = _router(cfg).routed(MISTRAL_JUDGE).client
    payload = client.build_payload(
        ChatRequest(
            messages=({"role": "user", "content": "x"},),
            ref=MISTRAL_JUDGE,
            role="judge",
            params={"temperature": 0.9},
        )
    )
    assert payload["temperature"] == 0.9


def test_role_params_naming_a_role_the_model_does_not_serve_is_refused(tmp_path):
    """A stale or typo'd role key would sit in the config looking like it
    configures something and never apply - the same silent-no-op class as an
    answer-key entry that can never fire."""
    raw = DATA_CONFIG.read_text(encoding="utf-8")
    broken = raw.replace("          judge: {temperature: 0.2}",
                         "          judgge: {temperature: 0.2}")
    assert broken != raw
    path = tmp_path / "bad_role.yaml"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_build_config(str(path), allow_unpinned=True)
    assert "judgge" in str(exc.value) and "role_params" in str(exc.value)
