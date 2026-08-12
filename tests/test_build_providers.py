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

from tuned.data.config import (  # noqa: E402
    ModelCfg,
    ModelRef,
    ProviderCfg,
    load_build_config,
)
from tuned.data.generate import (  # noqa: E402
    judge_messages,
    judge_needed_tokens,
    worst_case_judge_tokens,
)
from tuned.data.providers import (  # noqa: E402
    CONTEXT_SAFETY_MARGIN,
    DEFAULT_JUDGE_REPLY_TOKENS,
    QUIRKS,
    TEMPLATE_TOKENS_PER_MESSAGE,
    ChatClient,
    ChatRequest,
    CheckResult,
    ProviderError,
    Router,
    TokenBucket,
    build_check_request,
    check_refs,
    context_estimate,
    estimate_tokens,
    format_check_header,
    format_check_row,
    load_dotenv_keys,
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
def keys(monkeypatch):
    for env in ("CEREBRAS_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.setenv(env, "sk-test")


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
    assert router.pick("probe").ref == ModelRef("groq", "llama-3.1-8b-instant")


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
    assert router.pick("generator", exclude_families=frozenset({"gpt-oss"})).ref == ModelRef(
        "mistral", "magistral-small-latest"
    )


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
    assert router.pick("probe", exclude_families=frozenset({"llama"})) is None


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
    assert first.bucket.rpm == 60
    assert first.bucket.tpm == 500000


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
            clock, sleeper, seen, {"mistral": 500, "groq": 503, "cerebras": 500}
        ),
    )

    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(router.complete("judge", [{"role": "user", "content": "grade this"}]))

    assert excinfo.value.retryable is True
    assert "all 3 eligible model(s) failed" in str(excinfo.value)
    assert seen.count("mistral") == 2 and seen.count("groq") == 2 and seen.count("cerebras") == 2


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
    _unset(monkeypatch, "MISTRAL_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY")
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
                exclude_families=frozenset({"mistral", "glm"}),  # isolate the groq ref
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

    bucket = router.routed(ModelRef("groq", "llama-3.1-8b-instant")).bucket
    # probe limits: rpm 30, tpm 6000 -> 4000 charged leaves 2000.
    assert bucket.next_wait(2000) == 0.0
    assert bucket.next_wait(6000) == pytest.approx(40.0)


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
    """reasoning_effort is a gpt-oss parameter. Sent to magistral it is an
    unknown field, i.e. a 400, which Router.complete raises straight through
    instead of failing over - so per-call params must be chosen per REF."""
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
    assert ref == ModelRef("mistral", "magistral-small-latest")
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
    # The generator pool: cerebras/gpt-oss-120b is 8k, magistral is 40k.
    assert undersized_families(cfg, "generator", 4000) == frozenset()
    assert undersized_families(cfg, "generator", 20000) == frozenset({"gpt-oss"})
    assert undersized_families(cfg, "generator", 60000) == frozenset({"gpt-oss", "mistral"})


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
    _unset(monkeypatch, "MISTRAL_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY")
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
                exclude_families=frozenset({"llama"}),
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
            clock, sleeper, seen, {"mistral", "groq", "cerebras"}, OVERFLOW_BODY
        ),
    )

    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(router.complete("judge", [{"role": "user", "content": "grade this"}]))

    assert excinfo.value.context_exceeded is True
    assert excinfo.value.retryable is False
    assert seen == ["mistral", "groq", "cerebras"]  # every ref was offered it


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
    assert "CEREBRAS_API_KEY" in gaps["generator"]
    assert "MISTRAL_API_KEY" in gaps["generator"]
    # One key is enough to make a role usable: the rest is failover.
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")
    assert unkeyed_roles(cfg, ("generator", "judge")) == {}
    # ...and a role whose every provider is still unkeyed keeps reporting.
    assert unkeyed_roles(cfg, ("probe",))["probe"] == ("GROQ_API_KEY",)


def test_pool_gaps_finds_the_judge_hole_the_shipped_config_still_has(cfg, keys):
    """R2-C3: long rows route to magistral, family separation then removes
    mistral, the 8k glm judge is out on length, slot A takes qwen and slot B
    has NOTHING. The row parks having already paid for judge A."""
    gaps = pool_gaps(cfg, needed_tokens=worst_case_judge_tokens(cfg))
    judge_gaps = [g for g in gaps if g.role == "judge"]
    assert [(g.generator_family, g.slot) for g in judge_gaps] == [("mistral", "b")]
    assert judge_gaps[0].fatal is True
    assert "mistral" in judge_gaps[0].detail
    # The tiebreak hole is the one round 1 documented and gave a defined,
    # unpaid fallback for, so it is reported but not fatal.
    tiebreak_gaps = [g for g in gaps if g.role == "tiebreak"]
    assert tiebreak_gaps and all(not g.fatal for g in tiebreak_gaps)


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
    """The operator is choosing this model now, and many free-tier candidates
    are 16k - which is ABOVE the >= 11520 the preflight used to print. It has
    to be reported as the gap it is, before the fleet starts."""
    patched = _with_fourth_judge(cfg, max_context=16384)
    needed = worst_case_judge_tokens(patched)
    assert "fourth" in undersized_families(patched, "judge", needed)
    fatal = [g for g in pool_gaps(patched, needed_tokens=needed) if g.fatal]
    assert fatal, "a 16k judge cannot hold the longest row the length gate passes"
    assert "on context length" in fatal[0].detail
    # The old preflight sized this pool at 9216 and advised >= 11520, so a
    # 16k model read as comfortably large and the fleet started.
    assert 11520 < 16384 < required_context(needed)


def test_the_advice_the_preflight_prints_is_the_threshold_it_enforces(cfg, keys):
    """The gap detail names a max_context; a model of exactly that size must
    close the gap and one token smaller must not. Anything else is a preflight
    that tells the operator to buy the wrong model."""
    gaps = pool_gaps(cfg, needed_tokens=worst_case_judge_tokens(cfg))
    fatal = next(g for g in gaps if g.fatal)
    advised = int(fatal.detail.split("max_context >= ")[1].split()[0])

    exact = _with_fourth_judge(cfg, max_context=advised)
    assert pool_gaps(exact, needed_tokens=worst_case_judge_tokens(exact)) == []
    short = _with_fourth_judge(cfg, max_context=advised - 1)
    assert [g for g in pool_gaps(short, needed_tokens=worst_case_judge_tokens(short)) if g.fatal]


def test_pool_gaps_applies_the_routers_own_key_filter(cfg, monkeypatch):
    """R3-C3. Router.eligible skips an unkeyed ref as "missing-key"; a
    preflight that walks routing_refs without that filter reports a pool it
    cannot call. unkeyed_roles does not cover it - it passes a role as soon as
    ONE ref is keyed, which is the right question for "can this role call at
    all" and the wrong one for "can slot B be filled". Keys arrive piecemeal,
    so a partially-keyed start is the likely first real launch."""
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


def test_the_config_todo_quotes_the_numbers_the_code_enforces(cfg):
    """That block is the operator's spec for choosing the fourth-family judge,
    and its arithmetic was pre-margin and ~40% high (it said ~4.2k where the
    real divert point is 2555, and ~7.2k where slot B really dies at 5531)
    while pool_gaps printed a third number. All three now come from here."""
    text = DATA_CONFIG.read_text(encoding="utf-8")
    required = required_context(worst_case_judge_tokens(cfg))
    assert f"max_context >= {required}" in text
    assert f"{required:,}" in text
    # ...and the two thresholds the block explains the gap with.
    from tuned.data.generate import max_output_tokens

    assert f"{_divert_point(cfg, 'generator', 'gpt-oss', max_output_tokens(cfg)):,}" in text
    assert f"{_divert_point(cfg, 'judge', 'glm', DEFAULT_JUDGE_REPLY_TOKENS):,}" in text
    # The advice a 16k candidate would fail is stated, because most free-tier
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
