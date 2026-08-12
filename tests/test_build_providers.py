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
from tuned.data.providers import (  # noqa: E402
    QUIRKS,
    ChatClient,
    ChatRequest,
    CheckResult,
    ProviderError,
    Router,
    TokenBucket,
    build_check_request,
    check_refs,
    format_check_header,
    format_check_row,
    load_dotenv_keys,
    resolve_quirks,
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
    kw.setdefault("clock", FakeClock())

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


def test_complete_with_nothing_eligible_raises_non_retryable(cfg, keys):
    router = _router(cfg, budget_ok=lambda provider, model, tokens: False)
    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(router.complete("judge", [{"role": "user", "content": "hi"}]))
    assert excinfo.value.retryable is False
    assert "no eligible model" in str(excinfo.value)


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
