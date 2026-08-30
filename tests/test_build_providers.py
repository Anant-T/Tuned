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
    NARROW_GENERATOR_CONTEXT,
    PAID_PROVIDERS,
    SECOND_GENERATOR_CONTEXT,
    cfg_without_the_free_tiebreak,
    cfg_without_the_promoted_judge,
    cfg_with_context,
    cfg_with_gpt_oss_reinstated_as_generator,
    cfg_with_two_generator_families,
    cfg_with_split_pools,
    judge_prompt_overlay_with_pinned_tiebreak_gap,
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
from tuned.data.providers import (
    QuotaLedger,  # noqa: E402
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
from tuned.data import providers  # noqa: E402 - module access for private hook internals below

DATA_CONFIG = Path(__file__).parent.parent / "data" / "configs" / "data_law_v1.yaml"

# SLOTS A AND B since 2026-08-19, when mistral was removed from the build
# after human calibration disqualified it (holdout precision 0.237 on n=40,
# see the fence in the config) and gemma was promoted into the judge role.
# qwen is first in routing.judge and gemma second, so on a gpt-oss generation
# these are the two slots - and the failover target for a judge call is gemma
# where it used to be qwen.
GROQ_JUDGE = ModelRef("groq", "qwen/qwen3.6-27b")
GEMMA_JUDGE = ModelRef("cerebras", "gemma-4-31b")
# Third in routing.judge since 2026-08-18, when cerebras/zai-glm-4.7 left the
# pool: the model is archived upstream and 404s on every call.
PAID_JUDGE = ModelRef("openai", "gpt-5-mini")


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


def _narrow_generator(cfg):
    """Every generator family cut back to the window the pilot ran against.

    Needed since 2026-08-19. Both cerebras models were pinned at 8192 by a
    config value nobody had probed; the probes put them at 131,072, so every
    property that is ABOUT a generator too small for the row it is handed now
    has to construct one. cfg_with_context refuses a (family, role) it did not
    find, so this cannot quietly no-op the way the stale pin did.

    NARROWS BOTH gpt-oss AND deepseek, mirroring test_build_generate.py's
    fixture of the same name (there since 2026-08-25, when bai joined
    routing.generator with an 800,000-token window that would otherwise
    swallow every "too small" property below). This module's own copy went
    unfixed then because the shipped config's REAL generator was still
    gpt-oss, so narrowing only gpt-oss kept working by coincidence - until
    2026-08-28, when cerebras/gpt-oss-120b was removed from routing.generator
    outright (operator directive: deepseek is the sole generator) and
    deepseek became the only real generator family, which this fixture was
    not narrowing at all.
    """
    narrowed = cfg_with_context(
        cfg, family="gpt-oss", role="generator", max_context=NARROW_GENERATOR_CONTEXT
    )
    return cfg_with_context(
        narrowed, family="deepseek", role="generator", max_context=NARROW_GENERATOR_CONTEXT
    )


def _narrow_judge(cfg):
    """gemma cut back the same way.

    RENAMED FROM _narrow_tiebreak ON 2026-08-19: gemma took a JUDGE seat that
    day, and cfg_with_context rewrites the whole model, so narrowing "gemma
    tiebreak" narrows the gemma judge too. The honest name is what it does.
    Anything wanting a missing TIEBREAK should use cfg_without_the_free_tiebreak,
    which touches routing only.
    """
    return cfg_with_context(cfg, family="gemma", role="judge", max_context=8192)


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
    clamped = composed.request_hook({"max_tokens": 99999}, _model(), "generator")
    assert clamped["max_tokens"] == 4096
    assert resolve_quirks(()) is QUIRKS["default"]
    assert resolve_quirks(("groq",)) is QUIRKS["groq"]


def _bai_reply(content, *, finish_reason, reasoning="weighing the section...", completion=4096):
    return {
        "id": "chatcmpl-1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": completion,
            "total_tokens": 11 + completion,
        },
    }


def test_bai_quirk_raises_the_reply_budget_to_the_models_ceiling(monkeypatch):
    """The caller's budget is an ANSWER budget; b.ai's is a REPLY budget.

    deepseek-v4-flash bills reasoning against max_tokens and emits it first, so
    a 4000-token budget is not 4000 tokens of answer - it is 4000 tokens of
    reasoning-then-answer, and the measured empty-content rate at 4096 was
    10/20 on a real synthesis prompt while at 12288 it was 0/4. Sending the
    caller's number unchanged put the THEN-shipped GENERATION_OUTPUT_TOKENS
    (4000) exactly on the 50% empty point. (Since 2026-08-28 the shipped
    constant IS 16384 - aligned with what this hook was sending - so on the
    live config the raise is a no-op; the hook stays as the floor for any
    caller that arrives with less.)

    The hook is the ONLY place this can be fixed: build_payload applies
    req.max_tokens AFTER merging model params, so neither params nor
    role_params can raise it.

    This does not loosen any length gate. reply_budget_chars derives from
    max_output_tokens(cfg), a config value - not from what went on the wire -
    so over-long answers are still caught while the model gets room to think.
    """
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=_body())

    bai = _provider("bai", quirks=("bai",))
    deepseek = _model(
        "deepseek-v4-flash",
        family="deepseek",
        limits={"rpm": 13, "max_context": 800_000, "max_output": 16384},
    )

    # The shipped generation budget is raised to the model's reply ceiling.
    _complete(_client(handler, provider=bai, model=deepseek), _request(max_tokens=4000))
    assert payloads[-1]["max_tokens"] == 16384

    # A caller already at or above the ceiling is left alone - the hook raises
    # a floor, it does not clamp.
    _complete(_client(handler, provider=bai, model=deepseek), _request(max_tokens=32768))
    assert payloads[-1]["max_tokens"] == 32768

    # A call with no reply allowance stays that way: the hook does not invent one.
    _complete(_client(handler, provider=bai, model=deepseek), _request(params={"top_p": 0.9}))
    assert "max_tokens" not in payloads[-1]

    # ...and none of this leaks onto other providers.
    _complete(_client(handler, provider=_provider("groq")), _request(max_tokens=4000))
    assert payloads[-1]["max_tokens"] == 4000


def test_bai_quirk_rejects_the_truncated_empty_reply(monkeypatch):
    """deepseek-v4-flash bills reasoning against max_tokens and emits it FIRST,
    so a budget that runs out mid-reasoning returns a well-formed HTTP 200
    whose content is the empty string. Measured 2026-08-25: empty content and
    finish_reason == "length" coincided on every observed call, at every budget
    from 256 to 32768 - 3/3 empty at 512, 10/20 at 4096 on the real synthesis
    prompt, 0/4 at 12288.

    Without this the default hook returns text="" and the row is stored as if
    the model had answered with nothing. Retryable rather than fatal because
    reasoning length is not deterministic even at temperature 0 (298-10,426
    tokens observed), so the same call at the same budget genuinely may fit."""
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")
    bai = _provider("bai", quirks=("bai",))
    client = _client(
        lambda request: httpx.Response(200, json=_bai_reply("", finish_reason="length")),
        provider=bai,
    )

    with pytest.raises(ProviderError) as excinfo:
        _complete(client, _request(max_tokens=4096))

    assert excinfo.value.retryable is True
    assert not excinfo.value.context_exceeded  # the prompt fit; the REPLY did not
    message = str(excinfo.value).lower()
    assert "truncated" in message and "length" in message


def test_bai_quirk_passes_through_a_complete_reply(monkeypatch):
    """The rejection is scoped to the truncated-empty case. A finished reply is
    returned untouched, and a truncated reply that still carries content is a
    partial answer for the gates to judge - not this hook's call to discard."""
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")
    bai = _provider("bai", quirks=("bai",))

    done = _complete(
        _client(
            lambda request: httpx.Response(
                200, json=_bai_reply("Section 103 BNS.", finish_reason="stop")
            ),
            provider=bai,
        ),
        _request(max_tokens=4096),
    )
    assert done.text == "Section 103 BNS."
    assert done.reasoning == "weighing the section..."  # reasoning_content, not reasoning
    assert done.finish_reason == "stop"

    partial = _complete(
        _client(
            lambda request: httpx.Response(
                200, json=_bai_reply("Section 103 of the", finish_reason="length")
            ),
            provider=bai,
        ),
        _request(max_tokens=4096),
    )
    assert partial.text == "Section 103 of the"
    assert partial.finish_reason == "length"


def test_the_bai_hook_leaves_a_judge_reply_budget_alone():
    model = _model(limits={"max_output": 16384})
    payload = {"max_tokens": providers.DEFAULT_JUDGE_REPLY_TOKENS}
    for role in ("judge", "tiebreak"):
        out = providers._bai_request_hook(dict(payload), model, role)
        assert out["max_tokens"] == providers.DEFAULT_JUDGE_REPLY_TOKENS, role


def test_the_bai_hook_still_raises_a_generator_budget():
    model = _model(limits={"max_output": 16384})
    out = providers._bai_request_hook({"max_tokens": 4000}, model, "generator")
    assert out["max_tokens"] == 16384


def test_shipped_bai_judge_payload_disables_thinking_and_keeps_the_small_budget():
    """The two halves of the slot-B guarantee, on the config as SHIPPED.

    The hook (above) is what stops the 1,024-token verdict budget being raised
    16x into room to think; `thinking: disabled` is what stops the model
    spending the 1,024 it keeps. Neither is sufficient alone: reasoning is
    billed against max_tokens here and emitted FIRST, so a judge that
    deliberates returns HTTP 200 with empty content and finish_reason=length
    and the row reads as a content failure rather than the provider fact it is.

    Asserted through build_payload rather than by reading the YAML, because the
    key has to survive the merge to reach the wire: role_params is the middle
    of three layers and req.max_tokens is applied after all of them.

    Role-SCOPED, and the generator half is asserted here too: deepseek is still
    routing.generator ref 2, and a `thinking: disabled` that leaked onto the
    generator would silently strip the reasoning trace this corpus is made of.
    """
    cfg = load_build_config(DATA_CONFIG, allow_unpinned=True)
    ref = ModelRef("bai", "deepseek-v4-flash")
    provider, model = cfg.model_for(ref)

    def _never_called(request):  # build_payload performs no request
        raise AssertionError("build_payload must not perform a request")

    client = ChatClient(provider, model, transport=httpx.MockTransport(_never_called))

    for role in ("judge", "tiebreak"):
        payload = client.build_payload(
            ChatRequest(
                messages=({"role": "user", "content": "score this"},),
                ref=ref,
                role=role,
                max_tokens=DEFAULT_JUDGE_REPLY_TOKENS,
            )
        )
        assert payload["thinking"] == {"type": "disabled"}, role
        assert payload["max_tokens"] == DEFAULT_JUDGE_REPLY_TOKENS, role
        # The repo's judge convention, and the reason build_payload has a role
        # layer at all (providers.py:955-959). `params` sets 0.7 for the
        # generator and judge.py sends no per-call temperature, so without the
        # role layer every judge call would inherit generator sampling - which
        # on this model produced (4,3,3) and (5,5,5) verdicts on ONE unchanging
        # candidate, either side of a min_axis 4 rule.
        assert payload["temperature"] == 0.2, role

    generating = client.build_payload(
        ChatRequest(
            messages=({"role": "user", "content": "answer this"},),
            ref=ref,
            role="generator",
            max_tokens=4000,
        )
    )
    assert "thinking" not in generating
    assert generating["max_tokens"] == 16384
    assert generating["temperature"] == 0.7  # the generator keeps its own


def test_the_shipped_routing_lists_are_the_free_fleet():
    """The 2026-08-28 free-fleet ruling, pinned as list EQUALITY: no paid ref
    anywhere (the openai backstops and their provider block were deleted -
    prev_rep.md holds them), deepseek in both slots it can serve, and the
    order is the policy - the Router walks these lists front to back.
    """
    cfg = load_build_config(DATA_CONFIG, allow_unpinned=True)
    assert list(cfg.routing.generator) == ["bai/deepseek-v4-flash"]
    assert list(cfg.routing.judge) == [
        "groq/qwen/qwen3.6-27b",
        "cerebras/gemma-4-31b",
        "bai/deepseek-v4-flash",
        "groq/openai/gpt-oss-20b",
    ]
    assert list(cfg.routing.tiebreak) == [
        "mistral/mistral-large-latest",
        "groq/openai/gpt-oss-20b",
        "cerebras/gemma-4-31b",
        "bai/deepseek-v4-flash",
    ]
    assert not any(
        p.name in ("openai", "lightning") for p in cfg.providers
    ), "a paid provider block came back without a free-fleet ruling"

    # The seat exists only because the model declares the roles; role_params
    # for a role the model does not serve is refused at load. Pinned by
    # EQUALITY like every other roles assertion in this module: a superset
    # test would let a fourth role appear here unnoticed, and the roles a
    # model declares are what the Router walks.
    _provider, model = cfg.model_for(ModelRef("bai", "deepseek-v4-flash"))
    assert tuple(model.roles) == ("generator", "judge", "tiebreak")


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
    assert router.pick("judge").ref == GROQ_JUDGE
    # The generator's head-of-list provider has changed more than once (most
    # recently to bai on 2026-08-25); what this test is actually about is that
    # pick() walks routing.generator IN ORDER, not which provider sits first,
    # so the expectation is read off the config instead of pinned by name.
    assert router.pick("generator").ref == ModelRef(*cfg.routing.generator[0].split("/", 1))
    assert router.pick("probe").ref == ModelRef("groq", "openai/gpt-oss-20b")


def test_pick_excludes_families_at_call_time(cfg, keys):
    router = _router(cfg)

    # A gpt-oss generation may never be judged (or tie-broken) by gpt-oss.
    for role in ("judge", "tiebreak"):
        for routed in router.eligible(role, exclude_families=frozenset({"gpt-oss"})):
            assert routed.model_cfg.family != "gpt-oss"
        picked = router.pick(role, exclude_families=frozenset({"gpt-oss"}))
        assert picked is not None and picked.model_cfg.family != "gpt-oss"

    # tiebreak's first ref is mistral since the 2026-08-25 reorder (see the
    # routing.tiebreak comment in the shipped config - it moved mistral back
    # ahead of gpt-oss-20b once the generator's own family stopped being
    # gpt-oss on every row), so excluding mistral's family is what must move
    # the pick along here.
    assert router.pick("tiebreak").ref == ModelRef("mistral", "mistral-large-latest")
    assert router.pick("tiebreak", exclude_families=frozenset({"mistral"})).ref == ModelRef(
        "groq", "openai/gpt-oss-20b"
    )
    # ...and since b.ai (family deepseek) joined the generator role ahead of
    # cerebras on 2026-08-25, excluding gpt-oss no longer empties it: deepseek
    # is a distinct family and already the preferred ref, so it still answers.
    assert router.pick("generator", exclude_families=frozenset({"gpt-oss"})).ref == ModelRef(
        "bai", "deepseek-v4-flash"
    )


def test_pick_skips_missing_key(cfg, keys, monkeypatch):
    _unset(monkeypatch, "GROQ_API_KEY")
    router = _router(cfg)
    assert router.pick("judge").ref == GEMMA_JUDGE


def test_pick_skips_over_budget(cfg, keys):
    def budget_ok(provider, model, tokens):
        assert tokens == 500
        # Every judge over budget. Since the 2026-08-28 free-fleet ruling the
        # whole judge pool is groq/cerebras/bai - there is no paid backstop
        # left behind them, so an all-over-budget pool picks NOTHING rather
        # than falling through to a paid ref.
        return provider not in ("groq", "cerebras", "bai")

    router = _router(cfg, budget_ok=budget_ok)
    assert router.pick("judge", est_tokens=500) is None


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
    client = router.routed(GROQ_JUDGE).client
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
    # THE SHARED-WORKSPACE EXAMPLE LEFT WITH ITS MODEL. This used to read the
    # mistral pair, which shared one workspace bucket and was configured at
    # half each so it could not issue at twice the real refill rate; mistral
    # was removed from the build on 2026-08-19. What is under test is the
    # CACHE, not that arithmetic, so it reads the limits the surviving slot-A
    # judge actually declares.
    assert (first.bucket.rpm, first.bucket.tpm) == (30, 6000)
    # ...and asking for the same ref twice hands back the same object, bucket
    # and all - which is the property that stops two callers issuing against
    # two private views of one provider's allowance.
    generator = router.routed(GROQ_JUDGE)
    assert generator is first
    assert generator.bucket is first.bucket


# --- 9. circuit breaker -----------------------------------------------------


def test_circuit_breaker_cools_then_recovers(cfg, keys):
    clock = FakeClock()
    router = _router(cfg, clock=clock, breaker_threshold=2, cooldown_s=300.0)

    router.report_failure(GROQ_JUDGE)
    assert router.pick("judge").ref == GROQ_JUDGE  # below threshold
    router.report_failure(GROQ_JUDGE)

    assert router.is_cooling(GROQ_JUDGE)
    assert router.pick("judge").ref == GEMMA_JUDGE

    clock.advance(299.0)
    assert router.pick("judge").ref == GEMMA_JUDGE
    clock.advance(2.0)
    assert not router.is_cooling(GROQ_JUDGE)
    assert router.pick("judge").ref == GROQ_JUDGE


def test_report_success_resets_the_failure_run(cfg, keys):
    router = _router(cfg, breaker_threshold=2)
    router.report_failure(GROQ_JUDGE)
    router.report_success(GROQ_JUDGE)
    router.report_failure(GROQ_JUDGE)
    assert not router.is_cooling(GROQ_JUDGE)  # failures must be CONSECUTIVE
    assert router.pick("judge").ref == GROQ_JUDGE


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
        client_factory=_factory(clock, sleeper, seen, {"groq": 500}),
    )

    ref, response = asyncio.run(
        router.complete("judge", [{"role": "user", "content": "grade this"}], est_tokens=250)
    )

    assert ref == GEMMA_JUDGE
    assert response.text == "OK from cerebras"
    assert seen == ["groq", "groq", "cerebras"]  # 2 in-provider retries, then failover
    assert router.is_cooling(GROQ_JUDGE)
    assert not router.is_cooling(GEMMA_JUDGE)


def test_complete_raises_when_every_ref_fails(cfg, keys):
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    seen: list[str] = []
    router = _router(
        cfg,
        clock=clock,
        sleeper=sleeper,
        client_factory=_factory(
            clock, sleeper, seen, {"groq": 500, "cerebras": 503, "bai": 500, "openai": 500}
        ),
    )

    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(router.complete("judge", [{"role": "user", "content": "grade this"}]))

    assert excinfo.value.retryable is True
    # Four refs over three providers since the 2026-08-28 free-fleet ruling
    # (the two openai backstops left with their provider block). groq
    # contributes two (qwen since the start; groq/openai/gpt-oss-20b since
    # 2026-08-27, for the slot-B seat a DEEPSEEK generation leaves empty),
    # cerebras one (gemma, promoted 2026-08-19), bai one (deepseek,
    # 2026-08-27) - and ALL of them have to be tried before the role is out
    # of options.
    assert "all 4 eligible model(s) failed" in str(excinfo.value)
    assert seen.count("groq") == 4  # two refs, two in-provider attempts each
    assert seen.count("cerebras") == 2
    assert seen.count("bai") == 2
    assert seen.count("openai") == 0  # the paid backstops left the fleet 2026-08-28


def test_complete_does_not_fail_over_on_non_retryable(cfg, keys):
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    seen: list[str] = []
    router = _router(
        cfg,
        clock=clock,
        sleeper=sleeper,
        client_factory=_factory(clock, sleeper, seen, {"groq": 400}),
    )

    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(router.complete("judge", [{"role": "user", "content": "grade this"}]))

    assert excinfo.value.retryable is False
    assert seen == ["groq"]  # a bad payload would be bad everywhere
    assert not router.is_cooling(GROQ_JUDGE)


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
        client_factory=_factory(clock, sleeper, seen, {"groq": status}),
    )

    ref, response = asyncio.run(
        router.complete("judge", [{"role": "user", "content": "grade this"}])
    )

    assert ref == GEMMA_JUDGE  # moved on to the next eligible judge
    assert response.text == "OK from cerebras"
    assert seen == ["groq", "cerebras"]  # dead provider tried ONCE, not retried
    assert router.is_cooling(GROQ_JUDGE)  # and marked failed


def test_complete_aborts_the_whole_call_on_a_payload_4xx(cfg, keys):
    """400 is our bug: every provider would reject it, so do not hide it."""
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    seen: list[str] = []
    router = _router(
        cfg,
        clock=clock,
        sleeper=sleeper,
        client_factory=_factory(clock, sleeper, seen, {"groq": 400}),
    )

    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(router.complete("judge", [{"role": "user", "content": "grade this"}]))

    assert excinfo.value.retryable is False
    assert excinfo.value.provider_dead is False
    assert seen == ["groq"]  # ref2 never called
    assert not router.is_cooling(GROQ_JUDGE)


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
    # mistral is tiebreak's first ref since the 2026-08-25 reorder, so
    # excluding IT is what surfaces "family-excluded" before the walk lands on
    # the next ref (groq/openai/gpt-oss-20b).
    assert router2.pick("tiebreak", exclude_families=frozenset({"mistral"}), skipped=skipped2)
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
                # which since 2026-08-19 is gemma plus the gpt-oss backstop,
                # and since 2026-08-27 deepseek in slot B as well.
                exclude_families=frozenset({"gemma", "glm", "gpt-oss", "deepseek"}),
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
    assert payloads[0]["model"] == "qwen/qwen3.6-27b"


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
        client_factory=_factory(clock, sleeper, [], {"groq": 500}, max_retries=1),
    )
    ref, _ = asyncio.run(
        router.complete(
            "judge",
            [{"role": "user", "content": "grade"}],
            on_attempt=lambda ref, status, usage: seen.append((ref.provider, status)),
        )
    )
    assert ref.provider == "cerebras"
    # The failed provider's attempt is reported under ITS ref, not the winner's.
    assert seen == [("groq", 500), ("cerebras", 200)]


def test_params_for_ref_is_resolved_against_the_ref_about_to_be_called(cfg, keys):
    """A per-call param must be chosen for the ref that ANSWERS, not the one
    the Router would have tried first: an unknown field is a 400, which
    Router.complete raises straight through instead of failing over, so a
    param picked for the first ref turns every failover into a dead call.

    Needs a family the params-under-test never touch to fail over to. The
    shipped config has carried two generator families (deepseek and gpt-oss)
    since b.ai joined on 2026-08-25, but both configure reasoning_effort, so
    a fixture-synthesised third family is still what makes "the answering
    ref's params differ from the first tried" observable."""
    cfg = cfg_with_two_generator_families(cfg)
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    payloads: list[dict] = []

    def factory(provider, model):
        def handler(request):
            payloads.append(json.loads(request.content))
            # BY MODEL, not by provider: since 2026-08-19 the fixture's
            # second generator family lives under cerebras too, so failing the
            # whole provider would fail both refs and there would be nothing
            # to fail over TO.
            # bai/deepseek-v4-flash (family deepseek) leads routing.generator
            # since 2026-08-25, so it must fail too or it answers on the first
            # attempt and there is no failover to observe. Both cerebras
            # generators AND the paid lightning overflow are failed as well,
            # so the fixture's own second family is what answers. The test is
            # about WHICH ref the params were resolved for, so it needs the
            # answering ref to be a different one from the first tried.
            if model.family in ("deepseek", "gpt-oss"):
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
    assert ref == ModelRef("cerebras", "second-generator")
    assert payloads[0]["reasoning_effort"] == "high"      # bai/deepseek-v4-flash
    assert "reasoning_effort" not in payloads[-1]         # the second family


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
    # NOTHING drops out at 20k either, and the empty set is the assertion: the
    # judge pool lost its whole 8k tier on 2026-08-18 when zai-glm-4.7 was
    # retired (archived upstream, 404 on every call), so the first family this
    # filter can remove is the 32k one.
    assert undersized_families(cfg, "judge", 20000) == frozenset()
    assert undersized_families(cfg, "judge", 40000) == frozenset()
    # The generator pool is ONE family since 2026-08-28: bai/deepseek-v4-flash,
    # the SOLE routing.generator ref (operator directive - deepseek is the
    # sole generator, cerebras spends only on judging; see routing.generator's
    # own comment). Its declared window is 800,000. RE-BASELINED from
    # cerebras/gpt-oss-120b's 131,072: that model is REMOVED from
    # routing.generator outright, not merely reordered, so this property now
    # reads deepseek's own window rather than gpt-oss's. Past the window
    # there is no other family to fall back to, which is why an over-long row
    # parks instead of diverting - but the window is orders of magnitude
    # above anything this build produces (pilot prompts ran 1,445-2,799).
    assert undersized_families(cfg, "generator", 4000) == frozenset()
    assert undersized_families(cfg, "generator", 20000) == frozenset()
    assert undersized_families(cfg, "generator", 60000) == frozenset()
    # The cliff is where the code puts it, not where a comment says: 800,000 /
    # CONTEXT_SAFETY_MARGIN. One token either side of it.
    assert undersized_families(cfg, "generator", 640_000) == frozenset()
    assert undersized_families(cfg, "generator", 640_001) == frozenset({"deepseek"})
    # ...and the filter itself is unchanged - narrow the window back to the
    # value the pilot ran against and the old exclusions return verbatim. The
    # NUMBERS (6,554 / 6,555) are unaffected by which family is narrowed -
    # they come from NARROW_GENERATOR_CONTEXT (8192) and CONTEXT_SAFETY_MARGIN
    # alone - only the family name in the result changes.
    narrow = cfg_with_context(
        cfg, family="deepseek", role="generator", max_context=NARROW_GENERATOR_CONTEXT
    )
    assert undersized_families(narrow, "generator", 20000) == frozenset({"deepseek"})
    assert undersized_families(narrow, "generator", 6_554) == frozenset()
    assert undersized_families(narrow, "generator", 6_555) == frozenset({"deepseek"})


def test_undersized_families_excludes_a_family_only_past_its_largest_window(cfg):
    # RENAMED 2026-08-28: the mixed gpt-oss family this test used to exercise
    # (groq's 131k model plus two 400k openai backstops) lost its large side
    # when the paid refs were deleted - the rule is unchanged (a family is
    # excluded only when EVERY one of its role models is too small), and the
    # family that now demonstrates the far end is deepseek at the probed 800k.
    assert undersized_families(cfg, "tiebreak", 40_000) == frozenset()
    # mistral falls out first: its window is the PROBED 52,812 floor.
    assert undersized_families(cfg, "tiebreak", 42_250) == frozenset()
    assert undersized_families(cfg, "tiebreak", 42_251) == frozenset({"mistral"})
    # gemma and gpt-oss (both 131k) go together past 131,072/1.25.
    assert undersized_families(cfg, "tiebreak", 104_858) == frozenset({"mistral"})
    assert undersized_families(cfg, "tiebreak", 150_000) == frozenset(
        {"gemma", "gpt-oss", "mistral"}
    )
    # ...and deepseek (800k probed bracket) is the last family standing until
    # even its window cannot hold the row.
    assert undersized_families(cfg, "tiebreak", 640_000) == frozenset(
        {"gemma", "gpt-oss", "mistral"}
    )
    assert undersized_families(cfg, "tiebreak", 640_001) == frozenset(
        {"deepseek", "gemma", "gpt-oss", "mistral"}
    )


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
    # 6 since the 2026-08-28 free-fleet ruling (lightning and both openai
    # gpt-5 models left with their provider blocks): bai/deepseek-v4-flash,
    # cerebras/gpt-oss-120b (judging-era block, unrouted as generator),
    # cerebras/gemma-4-31b, groq/qwen/qwen3.6-27b, groq/openai/gpt-oss-20b,
    # mistral/mistral-large-latest.
    assert len(refs) == expected == 6
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
        ref="groq/qwen/qwen3.6-27b",
        key_present=False,
        status=None,
        text="",
        usage_present=False,
        reasoning_present=False,
        latency_ms=None,
        error="GROQ_API_KEY not set",
    )
    bad_row = format_check_row(bad)
    assert not bad.ok
    assert "FAIL" in bad_row and "GROQ_API_KEY not set" in bad_row
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
        client_factory=_overflow_factory(clock, sleeper, seen, {"groq"}, OVERFLOW_BODY),
    )

    ref, response = asyncio.run(
        router.complete("judge", [{"role": "user", "content": "grade this"}])
    )

    assert ref == GEMMA_JUDGE
    assert response.text == "OK from cerebras"
    assert seen == ["groq", "cerebras"]  # tried once, then moved on
    assert not router.is_cooling(GROQ_JUDGE)


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
            clock, sleeper, seen, {"groq", "cerebras", "bai", "openai"}, OVERFLOW_BODY
        ),
    )

    with pytest.raises(ProviderError) as excinfo:
        asyncio.run(router.complete("judge", [{"role": "user", "content": "grade this"}]))

    assert excinfo.value.context_exceeded is True
    assert excinfo.value.retryable is False
    # Every ref was offered it - a 400 that says "too long for this window"
    # never charges the breaker, so the pass runs to the end of the list
    # rather than stopping at the first refusal. groq appears twice since
    # 2026-08-27: qwen, then groq/openai/gpt-oss-20b, added to routing.judge
    # for the slot-B seat a DEEPSEEK generation leaves empty. The two openai
    # entries that used to close this walk left with the paid backstops
    # (2026-08-28, free fleet only).
    assert seen == ["groq", "cerebras", "bai", "groq"]


def test_undersized_families_keeps_an_explicit_safety_margin(cfg):
    """chars/4 is an estimate, so a cap that merely EQUALS it is not headroom:
    the 32k judge is excluded at a 32,000-token estimate, not only past it.

    The 8k judge carried this until 2026-08-18; with zai-glm-4.7 retired the
    smallest judge in the pool is mistral, so the same rule is read at its
    window instead. The generator half is read at deepseek's own 800,000
    since 2026-08-28 (operator directive: deepseek is the sole generator,
    replacing cerebras/gpt-oss-120b's 131,072 from the 2026-08-19 probe): an
    estimate that merely equals the declared window is a coin flip on a 400,
    exactly as it was at 8192."""
    assert CONTEXT_SAFETY_MARGIN > 1.0
    # 131,072, not 32,000: mistral left on 2026-08-19 and the smallest judge in
    # the pool is now 131k, so the rule is read at that window on both sides.
    assert "qwen" in undersized_families(cfg, "judge", 131072)
    # 800,000, not 131,072: cerebras/gpt-oss-120b is removed from
    # routing.generator (2026-08-28), so the generator half now reads
    # deepseek's own declared window.
    assert "deepseek" in undersized_families(cfg, "generator", 800000)
    # ...and the margin does not start excluding models with real headroom.
    assert undersized_families(cfg, "judge", 4000) == frozenset()


def test_the_token_estimate_counts_indic_script_far_harder_than_latin():
    """A BPE vocabulary trained on English runs ~1-2 chars/token on
    Devanagari, so chars/4 under-counts an Indic passage by 2-4x - on exactly
    this corpus, and the under-count is what would land a prompt at a model
    that cannot hold it. The margin this buys is what the routing spends; it
    does not depend on any particular window, which is just as well, because
    the one it was written against was wrong by 16x (2026-08-19 probe)."""
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
    _unset(monkeypatch, "GROQ_API_KEY", "CEREBRAS_API_KEY", "OPENAI_API_KEY", "BAI_API_KEY")
    gaps = unkeyed_roles(cfg, ("generator", "judge"))
    assert set(gaps) == {"generator", "judge"}
    # ONE provider on the generator role since 2026-08-28: bai/deepseek-v4-
    # flash is the SOLE routing.generator ref (operator directive - deepseek
    # is the sole generator, cerebras spends only on judging; see
    # routing.generator's own comment). RE-BASELINED from the two-provider
    # (bai, cerebras) shape that held from lightning's removal (2026-08-27)
    # to cerebras's removal (2026-08-28): there is no longer a failover list
    # here, just the one key that makes the role usable at all.
    expected_generator_envs = []
    for ref in cfg.routing_refs("generator"):
        provider, _ = cfg.model_for(ref)
        if provider.api_key_env not in expected_generator_envs:
            expected_generator_envs.append(provider.api_key_env)
    assert gaps["generator"] == tuple(expected_generator_envs)
    assert set(gaps["generator"]) == {"BAI_API_KEY"}
    assert "GROQ_API_KEY" in gaps["judge"]
    # The generator's one key is what makes the role usable now - there is no
    # second ref left to fail over to.
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    assert set(unkeyed_roles(cfg, ("generator", "judge"))) == {"generator"}
    monkeypatch.setenv("BAI_API_KEY", "sk-test")
    assert unkeyed_roles(cfg, ("generator", "judge")) == {}
    # ...and a role whose every provider is still unkeyed keeps reporting.
    # OPENAI is the one held back here: probe routes to groq, which is keyed
    # above, so the role that can still report is the judge's paid backstop.
    _unset(monkeypatch, "GROQ_API_KEY")
    assert unkeyed_roles(cfg, ("probe",))["probe"] == ("GROQ_API_KEY",)


def test_the_shipped_config_has_no_fatal_judge_hole_left(cfg, keys):
    """R2-C3 CLOSED. The hole was: long rows route to magistral, family
    separation removed the generator's own family, the 8k glm judge was out on
    length, slot A took qwen and slot B had NOTHING - so the row parked having
    already paid for judge A. The openai backstop is the fourth family that
    fills slot B.

    Pinned as "no fatal JUDGE gap". The tiebreak used to be a different
    matter (2026-08-19 to 2026-08-28): gemma was promoted into the judge
    role, so on a gpt-oss row it filled slot B and the tiebreak had no
    family it had not already spent - a WARN, not a refusal. That gap does
    not reproduce on deepseek (see the RE-BASELINED note below), so what is
    asserted now is that removing the free tiebreak survivors leaves no
    tiebreak gap at all - a strictly cleaner property than the one this
    docstring originally described."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    # The promoted judge STAYS here: this test asserts the hole is closed, and
    # gemma joining routing.judge on 2026-08-19 is part of what closes it.
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
    slot_a = next(router.eligible_refs("judge", exclude_families=frozenset({"secondgen"}) | too_small))
    slot_b = next(
        router.eligible_refs(
            "judge", exclude_families=frozenset({"secondgen", "qwen"}) | too_small
        )
    )
    assert slot_a == GROQ_JUDGE
    assert slot_b == GEMMA_JUDGE

    # AND NO TIEBREAK GAP EITHER, which is the 2026-08-19 surgery's second
    # half: qwen and gemma take the judge slots on a deepseek row, the
    # tiebreak excludes {deepseek, qwen, gemma}, and groq/openai/gpt-oss-20b
    # (family gpt-oss, untouched by that exclusion) decides it.
    assert [g for g in gaps if g.role == "tiebreak"] == []
    without = pool_gaps(
        cfg_without_the_free_tiebreak(cfg), needed_tokens=worst_case_judge_tokens(cfg)
    )
    # RE-BASELINED 2026-08-28: with cerebras/gpt-oss-120b removed from
    # routing.generator outright (operator directive - deepseek is the sole
    # generator), the "gpt-oss row" this used to describe no longer exists.
    # Stripping mistral and bai (FREE_TIEBREAKS) from routing.tiebreak used
    # to reopen a gap for a gpt-oss-generated row specifically, because
    # gpt-oss lumps groq/openai/gpt-oss-20b AND both openai backstops into
    # the SAME family as the generator, so a gpt-oss row's exclusion set
    # removed every one of them at once. deepseek shares no family with any
    # tiebreak-role model (mistral/gpt-oss/gemma/openai are all distinct
    # from deepseek), so stripping mistral+bai leaves gpt-oss-20b standing
    # for BOTH the secondgen and the real deepseek row - nothing here
    # reopens a gap for either generator family any more.
    tiebreak_gaps = [g for g in without if g.role == "tiebreak"]
    assert tiebreak_gaps == []

    # ...and the hole really is closed by the ADDITION, not by the check going
    # quiet: take the backstop back out and the old gap is reported verbatim.
    before = pool_gaps(
        cfg_without_the_promoted_judge(cfg),
        needed_tokens=worst_case_judge_tokens(cfg),
    )
    judge_gaps = [g for g in before if g.role == "judge"]
    assert [(g.generator_family, g.slot) for g in judge_gaps] == [("secondgen", "b")]
    assert judge_gaps[0].fatal is True
    assert "a secondgen generation of" in judge_gaps[0].detail
    assert "minus ['qwen'] already used" in judge_gaps[0].detail


def test_the_archived_glm_model_is_in_no_pool_and_no_provider_block(cfg):
    """zai-glm-4.7 was archived upstream and answers HTTP 404
    (model_archived_error) to every call. Measured 2026-08-18 on four live
    slot-B routes: it was tried, it failed, and the route fell through - so its
    only effect was to spend one eligible-model attempt on every route that
    reached it, forever, because a 404 is not a transient the breaker learns
    from within a run.

    Both routing lists AND the provider block, which is three assertions rather
    than one on purpose: dropping it from routing.judge alone leaves the
    tiebreak paying the same attempt, and leaving the model block behind keeps
    `providers --check` reporting on a ref nothing can call.

    The family goes with it. Nothing in this suite may quietly reintroduce
    'glm' as a live judge without this failing first, because several rules
    here used it as the pool's only 8k model and would read as still covered."""
    configured = {f"{p.name}/{m.id}" for p in cfg.providers for m in p.models}
    assert "cerebras/zai-glm-4.7" not in configured
    for role in ("judge", "tiebreak"):
        assert "cerebras/zai-glm-4.7" not in getattr(cfg.routing, role), role
        families = {cfg.model_for(r)[1].family for r in cfg.routing_refs(role)}
        assert "glm" not in families, role


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

    Run against a pool whose slot B is empty, because that is the pool in
    which a fourth-family judge is the thing filling it. With a promoted judge
    in the list the slot is filled whatever the 16k model does, so the fixture
    would guarantee the null result and prove nothing about size."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    # ...and the promoted judge dropped with it: gemma joined routing.judge on
    # 2026-08-19, so slot B fills at every size and the gap under test here
    # cannot occur. See cfg_without_the_promoted_judge.
    cfg = cfg_without_the_promoted_judge(cfg_with_two_generator_families(cfg))
    size = 16384
    patched = _with_fourth_judge(cfg, max_context=size)
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
    assert "minus ['fourth'] on context length" in fatal[0].detail
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
    # ...and the promoted judge dropped with it: gemma joined routing.judge on
    # 2026-08-19, so slot B fills at every size and the gap under test here
    # cannot occur. See cfg_without_the_promoted_judge.
    cfg = cfg_without_the_promoted_judge(cfg_with_two_generator_families(cfg))
    # BAI_API_KEY added 2026-08-28: cerebras/gpt-oss-120b is removed from
    # routing.generator (operator directive - deepseek is the sole
    # generator), so bai/deepseek-v4-flash is now the only REAL generator
    # ref, and it needs its own key to be an eligible (keyed) generator
    # family at all - CEREBRAS_API_KEY alone now only keys the synthetic
    # secondgen family this fixture adds under the cerebras provider.
    for env in ("BAI_API_KEY", "CEREBRAS_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.setenv(env, "sk-test")
    _unset(monkeypatch, "GROQ_API_KEY", "OPENAI_API_KEY")
    # The fourth-family judge the operator is sourcing, behind the key that
    # has not arrived - and qwen is behind the same one. The paid backstop is
    # unkeyed too: keyed, it fills every slot at every size and there is no
    # unservable gap left to classify.
    #
    # The KEYED 8k judge below is what makes the contrast at the end of this
    # test expressible. It is the shape cerebras/zai-glm-4.7 held until it was
    # retired on 2026-08-18 (archived upstream): a third family, keyed, that
    # serves the short rows and not the long ones. Without one in the pool a
    # deepseek row has no keyed slot B at any size either, both gaps classify
    # unservable, and the per-gap distinction this test is about cannot be seen.
    widened = _with_judge_model(
        _with_fourth_judge(cfg, max_context=131072),
        family="small", model_id="small-judge", max_context=8192,
    )

    for needed in (500, 2000, worst_case_judge_tokens(widened)):
        gaps = [g for g in pool_gaps(widened, needed_tokens=needed) if g.fatal]
        # A magistral generation is the shape R3-C3 was found in: mistral is
        # the generator's own family, and both remaining judge families sit
        # behind the key that has not arrived.
        gap = next(g for g in gaps if g.generator_family == "secondgen")
        # GROQ is the one pending key since 2026-08-28: the unkeyed openai
        # backstop that used to co-appear in this remedy left with its
        # provider block (free fleet only).
        # remedy that named only the first would send the operator after half
        # of what they have.
        assert gap.key_envs == ("GROQ_API_KEY",)  # openai backstop deleted 2026-08-28
        assert "no row size is servable" in gap.detail

    # ...and the classification is per GAP, not per config: a deepseek row can
    # still be judged by mistral + the keyed 8k judge, so at 2,000 tokens it
    # has no gap at all and at 23,729 its gap is one the override may
    # legitimately cover. RE-BASELINED 2026-08-28: the real generator family
    # is deepseek now, not gpt-oss - cerebras/gpt-oss-120b is removed from
    # routing.generator (operator directive: deepseek is the sole generator).
    small = pool_gaps(widened, needed_tokens=2000)
    assert [g.generator_family for g in small if g.fatal] == ["secondgen"]
    big = pool_gaps(widened, needed_tokens=worst_case_judge_tokens(widened))
    deepseek_gap = next(g for g in big if g.generator_family == "deepseek" and g.fatal)
    assert (deepseek_gap.key_shaped, deepseek_gap.unservable) == (True, False)

    # ...and the same walk goes quiet the moment the key lands.
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    assert pool_gaps(widened, needed_tokens=worst_case_judge_tokens(widened)) == []


def test_a_context_shaped_judge_gap_has_row_sizes_the_pool_still_serves(cfg, keys):
    """The other side of R4-C1, and the reason the override exists at all: a
    16k fourth-family judge empties slot B only ABOVE a size, so short rows
    really are servable and running them is a real choice.

    On the same emptied slot B as the 16k test above: any wide judge left in
    the list fills slot B at every size, so there would be no context-shaped
    gap to have sizes on either side of."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    # ...and the promoted judge dropped with it: gemma joined routing.judge on
    # 2026-08-19, so slot B fills at every size and the gap under test here
    # cannot occur. See cfg_without_the_promoted_judge.
    cfg = cfg_without_the_promoted_judge(cfg_with_two_generator_families(cfg))
    sixteen_k = _with_fourth_judge(cfg, max_context=16384)
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
    warning. Run under judge_prompt_overlay_with_pinned_tiebreak_gap so that
    "the tiebreak is sized apart from it" is a fact this test controls rather
    than one it reads off the shipped templates' incidental relative sizes."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    #
    # gemma is narrowed back to the stale 8192 because THE SHIPPED POOL NOW HAS
    # NO GAP AT ALL (2026-08-19 probe), and a property about what a gap advises
    # needs a gap. It is also the honest fixture: this is the exact pool the
    # advice was designed against.
    #
    # cfg_with_gpt_oss_reinstated_as_generator ADDED 2026-08-28:
    # cerebras/gpt-oss-120b is removed from routing.generator outright
    # (operator directive - deepseek is the sole generator, cerebras spends
    # only on judging), so stripping mistral+bai from routing.tiebreak
    # (cfg_without_the_free_tiebreak) no longer reopens a gap by itself -
    # deepseek shares no family with groq/openai/gpt-oss-20b or the openai
    # backstops, so it is never excluded from them and always has a tiebreak.
    # Only a gpt-oss-generated row reproduces the single self-referential
    # gap this test needs (gpt-oss lumps all three of those into one
    # excluded family), so gpt-oss-120b is put back for this test only.
    cfg = cfg_without_the_free_tiebreak(
        cfg_with_gpt_oss_reinstated_as_generator(cfg_with_two_generator_families(cfg))
    )
    with judge_prompt_overlay_with_pinned_tiebreak_gap():
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
                tiebreak_needed_tokens=worst_case_judge_tokens(
                    patched, prompt_id=TIEBREAK_PROMPT_ID
                ),
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
    # ...and the promoted judge dropped with it: gemma joined routing.judge on
    # 2026-08-19, so slot B fills at every size and the gap under test here
    # cannot occur. See cfg_without_the_promoted_judge.
    cfg = cfg_without_the_promoted_judge(cfg_with_two_generator_families(cfg))
    # The 2026-08-28 alignment (GENERATION_OUTPUT_TOKENS 16384, the sole
    # generator's real reply ceiling) puts the reply term alone above the band
    # ceiling, so per-family narrowing is unreachable at ANY window on the
    # shipped constant. The narrowing ALGORITHM is the subject here - it is
    # what any future narrow-window generator rests on - so it runs at the
    # pre-alignment budget where the cliff is reachable.
    monkeypatch.setattr("tuned.data.generate.GENERATION_OUTPUT_TOKENS", 4000)
    flat = required_context(worst_case_judge_tokens(cfg))

    def advice_with(*set_envs):
        _unset(monkeypatch, "CEREBRAS_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY")
        for env in set_envs:
            monkeypatch.setenv(env, "sk-test")
        return _advice(cfg)

    fully_keyed = advice_with("CEREBRAS_API_KEY", "GROQ_API_KEY")
    assert fully_keyed == {flat}
    # The pending-key state: a real gap, narrowed to a real 15,104 check
    # (which is right - that IS the largest row an 8k generator can make) and
    # still advising the number that survives the key landing.
    pending = advice_with("CEREBRAS_API_KEY")
    assert judge_sizer(cfg)(8192, "judge") < worst_case_judge_tokens(cfg)
    assert pending == fully_keyed
    # ...in every other partially-keyed state too. Whichever keys are set, the
    # operator is told one number.
    # Both generator families live under CEREBRAS since 2026-08-19, so a state
    # without that key has no generator to walk and nothing to advise about -
    # which is a different question from the one under test.
    for envs in (("CEREBRAS_API_KEY",), ("CEREBRAS_API_KEY", "GROQ_API_KEY")):
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
    # ...and the promoted judge dropped with it: gemma joined routing.judge on
    # 2026-08-19, so slot B fills at every size and the gap under test here
    # cannot occur. See cfg_without_the_promoted_judge.
    cfg = cfg_without_the_promoted_judge(cfg_with_two_generator_families(cfg))
    floor = min_judge_tokens(cfg)
    tiny = 3000
    assert tiny < required_context(floor), "the fixture has to be under the real floor"
    # Slot B has to be empty: any unbounded judge left in the list serves the
    # short rows too, so nothing would be unservable and the rule would have
    # no case.
    #
    # The small third-family judge is SUPPLIED rather than borrowed from the
    # pool. cerebras/zai-glm-4.7 held that window until 2026-08-18, when it was
    # retired as archived, and the two configs this rule contrasts - one whose
    # slot B serves the short rows, one whose slot B serves none - both need a
    # judge whose window is the only thing that moved between them.
    base = _with_judge_model(
        cfg, family="small", model_id="small-judge", max_context=8192,
    )
    patched = cfg_with_context(base, family="small", role="judge", max_context=tiny)

    def second_family_gap(config):
        gaps = pool_gaps(
            config,
            needed_tokens=worst_case_judge_tokens(config),
            servable_floor_tokens=floor,
        )
        return next(g for g in gaps if (g.generator_family, g.slot) == ("secondgen", "b"))

    gap = second_family_gap(patched)
    assert gap.unservable is True
    assert "no row size is servable" in gap.detail
    # The floor is what decides it: the same judge at 8,192 really can serve
    # the short rows, and that gap stays overridable.
    assert second_family_gap(base).unservable is False

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
    # gemma dropped from routing.judge, NOT narrowed. Since 2026-08-19 it
    # serves both roles, and cfg_with_context rewrites the whole model - so
    # narrowing "gemma tiebreak" would narrow the gemma JUDGE as well and move
    # slot B underneath the gap this is reading. Dropping it from the judge
    # list leaves it whole as a tiebreak, which is what the remedy is about.
    #
    # cfg_with_gpt_oss_reinstated_as_generator ADDED 2026-08-28:
    # cerebras/gpt-oss-120b is removed from routing.generator outright
    # (operator directive - deepseek is the sole generator, cerebras spends
    # only on judging), so stripping mistral+bai from routing.tiebreak no
    # longer empties it on its own - deepseek shares no family with
    # groq/openai/gpt-oss-20b or the openai backstops, so a deepseek row is
    # never excluded from them and this fifth-family key-shaped gap would
    # never surface (gpt-oss-20b fills the seat before the pool ever reaches
    # fifthparty). Reinstating gpt-oss-120b as an additional real generator
    # reproduces the self-referential exclusion (gpt-oss lumps
    # groq/openai/gpt-oss-20b and both openai backstops into one family) a
    # gpt-oss-generated row needs to actually reach fifthparty.
    patched = _with_tiebreak_family(
        cfg_without_the_free_tiebreak(cfg_with_gpt_oss_reinstated_as_generator(cfg)),
        api_key_env="FIFTHPARTY_API_KEY",
    )
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

    Run on the emptied slot B, so that the gap the unkeyed family would
    contribute is a FATAL one and the rule is tested where it costs: with a
    promoted judge in the list every judge slot fills for every family, and the
    only thing an unkeyed generator could add is another tiebreak warning."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    # gemma narrowed: the ("gpt-oss", "tiebreak") entry this asserts is the
    # context check removing gemma, which at its probed 131k it no longer does.
    # gemma dropped from routing.judge: the gap this asserts is slot B running
    # out of families, which since the 2026-08-19 promotion it does not.
    cfg = cfg_without_the_promoted_judge(cfg_with_two_generator_families(cfg))
    patched = _with_extra_generator(
        cfg, family="qwen", api_key_env="FOURTHPARTY_API_KEY"
    )
    _unset(monkeypatch, "FOURTHPARTY_API_KEY")
    needed = worst_case_judge_tokens(patched)

    assert [(g.generator_family, g.slot) for g in pool_gaps(patched, needed_tokens=needed)] == [
        ("secondgen", "b"),
    ]

    # ...and the moment that key lands the family is real, and so is its gap:
    # a qwen generation may not be judged by the qwen judge, and glm is 8k.
    monkeypatch.setenv("FOURTHPARTY_API_KEY", "sk-test")
    assert ("qwen", "b") in [
        (g.generator_family, g.slot) for g in pool_gaps(patched, needed_tokens=needed) if g.fatal
    ]


def test_each_generator_family_is_checked_at_the_size_its_own_window_permits(cfg, keys, monkeypatch):
    """A NARROW generator cannot be handed the longest row the length band
    permits - `undersized_families` diverts it long before - so checking its
    judge slots at that length invents a gap. Sized at what its own window
    permits the same pool serves it, while the 32k generator, which really can
    produce that row, still has the gap the config TODO is about.

    The narrow window is a FIXTURE since 2026-08-19. The shipped cerebras
    generator supplied it until the probes put that model at 131k, at which
    point it narrows nothing and this property had no family to act on."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    # ...and the promoted judge dropped with it: gemma joined routing.judge on
    # 2026-08-19, so slot B fills at every size and the gap under test here
    # cannot occur. See cfg_without_the_promoted_judge.
    #
    # cfg_with_gpt_oss_reinstated_as_generator ADDED 2026-08-28:
    # cerebras/gpt-oss-120b is removed from routing.generator outright
    # (operator directive - deepseek is the sole generator, cerebras spends
    # only on judging), so the shipped pool no longer supplies the property
    # this test is about for free. This test specifically needs the REAL
    # gpt-oss self-lump (it shares family with groq/openai/gpt-oss-20b AND
    # both openai backstops), not cfg_with_two_generator_families' foreign
    # "secondgen" family - deepseek shares no family with anything left in
    # this reduced judge pool, so a deepseek row always has two independent
    # rescue candidates (the openai family and secondgen) and never gaps
    # here at all; only gpt-oss and secondgen compete for the same scarce
    # remainder, which is the scarcity this test measures.
    cfg = _narrow_generator(
        cfg_without_the_promoted_judge(
            cfg_with_gpt_oss_reinstated_as_generator(cfg_with_two_generator_families(cfg))
        )
    )
    # The 2026-08-28 alignment (GENERATION_OUTPUT_TOKENS 16384) makes the
    # narrowing unreachable at any window on the shipped constant - the
    # ALGORITHM is the subject, so it runs at the pre-alignment budget. See
    # test_the_advice_never_falls_below_the_flat_worst_case.
    monkeypatch.setattr("tuned.data.generate.GENERATION_OUTPUT_TOKENS", 4000)
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

    # RE-DERIVED 2026-08-28 (free fleet): with the 400k openai judges gone,
    # the promoted-judge-free pool is qwen alone at 26k. At the FLAT size the
    # narrowed families (gpt-oss and deepseek, both 8k by fixture) each gap
    # at slot B, and secondgen - whose 131k window really can make the flat
    # row - gaps at slot A, where the undersized qwen now sits.
    assert fatal() == {("deepseek", "b"), ("gpt-oss", "b"), ("secondgen", "a")}
    # Sized at what their own 8k windows permit, the narrowed families' rows
    # fit the 26k judge and their refusals disappear; secondgen keeps its gap
    # because it alone can produce the row the pool cannot judge. That
    # asymmetry IS the property under test, unchanged since 2026-08-19.
    assert fatal(needed_for_window=sizer) == {("secondgen", "a")}


def test_the_family_window_bound_never_sizes_above_the_flat_worst_case(cfg, keys):
    """`min` with the flat number is load-bearing: the hook can only ever make
    a check smaller, so a caller that passes a wrong one cannot widen what the
    preflight checks behind the operator's back."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    # ...and the promoted judge dropped with it: gemma joined routing.judge on
    # 2026-08-19, so slot B fills at every size and the gap under test here
    # cannot occur. See cfg_without_the_promoted_judge.
    cfg = cfg_without_the_promoted_judge(cfg_with_two_generator_families(cfg))
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
    #
    # SINCE 2026-08-28 the reply term ALONE (16384 x 5.5 = 90,112 chars)
    # exceeds the band ceiling (total_max * 4 = 32,768), so the min clamps to
    # the flat worst case at EVERY window - the zero-material case included -
    # and window narrowing is retired on the shipped config. The load-bearing
    # property is unchanged and still what this test pins: the hook can only
    # ever make a check SMALLER than the flat number, never larger.
    reply_chars = max_output_tokens(cfg) * REPLY_BUDGET_CHARS_PER_TOKEN
    assert int(reply_chars) > cfg.build.length_band.total_max * 4
    assert judge_tokens_for_generator_window(cfg, max_output_tokens(cfg)) == worst_case_judge_tokens(cfg)
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
    prompts are not the same prompt. Pinned via
    judge_prompt_overlay_with_pinned_tiebreak_gap rather than read off the
    shipped templates: the gap used to be a four-token ACCIDENT of the
    shipped prose, and splitting the grounding rubric's bands (2026-08-24
    judge-calibration Task 2) permanently inverted it. With the gap pinned,
    the only way to pin the plumbing is still a model that sits between the
    two requirements: it must clear the judge's requirement and fail the
    tiebreak's."""
    # Two generator families: this property is about the ALGORITHM that walks
    # them, and the shipped config has carried only one since the 2026-08-18
    # mistral demotion. See cfg_with_two_generator_families.
    # ...and the promoted judge dropped with it: gemma joined routing.judge on
    # 2026-08-19, so slot B fills at every size and the gap under test here
    # cannot occur. See cfg_without_the_promoted_judge.
    #
    # cfg_with_gpt_oss_reinstated_as_generator ADDED 2026-08-28: cerebras/
    # gpt-oss-120b is removed from routing.generator outright (operator
    # directive - deepseek is the sole generator, cerebras spends only on
    # judging), so the shipped pool carries only ONE real generator family
    # today, not two. This property needs THREE families walked (see below),
    # so gpt-oss is put back alongside deepseek for this test.
    cfg = cfg_without_the_promoted_judge(
        cfg_with_two_generator_families(cfg_with_gpt_oss_reinstated_as_generator(cfg))
    )
    with judge_prompt_overlay_with_pinned_tiebreak_gap():
        judge_needed = worst_case_judge_tokens(cfg)
        tiebreak_needed = worst_case_judge_tokens(cfg, prompt_id=TIEBREAK_PROMPT_ID)
        assert required_context(judge_needed) < required_context(tiebreak_needed)

        patched = cfg_with_split_pools(
            cfg, judge_context=131072, tiebreak_context=required_context(tiebreak_needed) - 1
        )
        sized = pool_gaps(
            patched, needed_tokens=judge_needed, tiebreak_needed_tokens=tiebreak_needed
        )
        # All three generator families now: gpt-oss (reinstated above) and
        # secondgen join deepseek because the paid backstop is spent on judge
        # slot B for a mistral row and is therefore gone from its tiebreak
        # too. The ORDER here follows routing.generator: deepseek is the
        # SOLE shipped ref since 2026-08-28 (operator directive), so it comes
        # first; cfg_with_gpt_oss_reinstated_as_generator appends gpt-oss
        # right after it, and cfg_with_two_generator_families always appends
        # secondgen last. What the test turns on is unchanged regardless of
        # this order: every one of these flips on the pinned gap between the
        # judge prompt and the tiebreak prompt.
        assert [(g.role, g.generator_family) for g in sized] == [
            ("tiebreak", "deepseek"),
            ("tiebreak", "gpt-oss"),
            ("tiebreak", "secondgen"),
        ]
        # Sized by the JUDGE's number instead, that same model reads as big
        # enough.
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
    # groq, not cerebras: since 2026-08-19 cerebras holds BOTH the generator
    # and the promoted gemma judge, so withholding its key would make every
    # assertion below true for more than one reason at once. groq held exactly
    # one judge when this test was written; since 2026-08-27 it holds two -
    # qwen (family qwen) and groq/openai/gpt-oss-20b (family gpt-oss, added
    # for the slot-B seat a DEEPSEEK generation leaves empty) - so excluding
    # only qwen's family no longer accounts for every groq-keyed judge. Both
    # families have to be excluded for "this key changes nothing" to hold;
    # gpt-oss also happens to be the family openai/gpt-5-mini and
    # openai/gpt-5-nano are lumped into, which is fine here since they are
    # keyed by `keys` regardless.
    _unset(monkeypatch, "GROQ_API_KEY")

    assert _blocking_key_envs(cfg, "judge", frozenset()) == ("GROQ_API_KEY",)
    assert _blocking_key_envs(cfg, "judge", frozenset({"qwen", "gpt-oss"})) == ()
    # ...and a keyed provider is never named at all.
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    assert _blocking_key_envs(cfg, "judge", frozenset()) == ()


def test_undersized_families_needs_every_model_in_the_family_to_be_too_small(cfg):
    """EVERY, not ANY: a mixed-size family keeps its large model and the
    preference order decides between them. The distinction is invisible on
    today's config because no family has two models in one role - and the
    config TODO asks the operator to ADD one, at which point this rule decides
    whether the family they added is usable at all."""
    # The family is SYNTHESISED, because no shipped judge family is small
    # enough to be excluded at any row size since 2026-08-19.
    small = _with_judge_model(cfg, family="tiny", model_id="tiny-judge", max_context=8192)
    assert "tiny" in undersized_families(small, "judge", 40000)
    mixed = _with_judge_model(small, family="tiny", model_id="tiny-big", max_context=131072)
    assert "tiny" not in undersized_families(mixed, "judge", 40000)
    # ...and a second SMALL model does not rescue it.
    small = _with_judge_model(cfg, family="mistral", model_id="mistral-tiny", max_context=8192)
    assert "mistral" in undersized_families(small, "judge", 40000)


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


def test_a_bigger_model_in_a_generator_family_raises_the_size_its_judges_are_checked_at(cfg, keys, monkeypatch):
    """What the rule above costs when it is wrong, at the only place it is
    spent. The operator adds a 128k variant beside a narrow one in an existing
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
    # The narrow generator is a fixture since the 2026-08-19 probe - see
    # _narrow_generator. Without it "the family is 8k-only" is false by
    # construction and the narrowing under test never happens.
    # ...and the promoted judge dropped with it: gemma joined routing.judge on
    # 2026-08-19, so slot B fills at every size and the gap under test here
    # cannot occur. See cfg_without_the_promoted_judge.
    cfg = _narrow_generator(
        cfg_without_the_promoted_judge(cfg_with_two_generator_families(cfg))
    )
    # The 2026-08-28 alignment (GENERATION_OUTPUT_TOKENS 16384) makes the
    # narrowing unreachable at any window on the shipped constant - the
    # ALGORITHM is the subject, so it runs at the pre-alignment budget. See
    # test_the_advice_never_falls_below_the_flat_worst_case.
    monkeypatch.setattr("tuned.data.generate.GENERATION_OUTPUT_TOKENS", 4000)
    small = cfg_with_context(cfg, family="qwen", role="judge", max_context=26000)
    sizer, flat = judge_sizer(small), worst_case_judge_tokens(small)

    def fatal(config):
        return {
            (g.generator_family, g.slot)
            for g in pool_gaps(config, needed_tokens=flat, needed_for_window=sizer)
            if g.fatal
        }

    # With the family narrowed to 8k the narrowing is right: at 19,104 the
    # 26k judge holds every row it can produce.
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
    # ...and the promoted judge dropped with it: gemma joined routing.judge on
    # 2026-08-19, so slot B fills at every size and the gap under test here
    # cannot occur. See cfg_without_the_promoted_judge.
    cfg = cfg_without_the_promoted_judge(cfg_with_two_generator_families(cfg))
    _unset(monkeypatch, "GROQ_API_KEY", "OPENAI_API_KEY")
    for env in ("MISTRAL_API_KEY", "CEREBRAS_API_KEY"):
        monkeypatch.setenv(env, "sk-test")
    # A judge behind CEREBRAS_API_KEY, which the shipped config stopped having
    # on 2026-08-18 when zai-glm-4.7 was retired as archived. Without one the
    # walk cannot go quiet at the end: the key that lands is GROQ's, and with
    # every other judge family behind it or behind OPENAI's there would still
    # be an empty slot afterwards - which would pass this test's first half for
    # the wrong reason and fail its second.
    cfg = _with_judge_model(cfg, family="small", model_id="small-judge", max_context=131072)
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
    """The prompt size at which `family` stops being routable for `role`.

    Bounded, and the bound is not decoration: a family that has LEFT the role's
    pool is never undersized for it, so the unbounded loop this replaced ran
    forever and took the whole suite with it the day zai-glm-4.7 was retired.
    A helper that hangs on a config change reports nothing at all.

    THE BOUND IS ALSO WHY THIS IS NOT THE HELPER FOR THE SHIPPED GENERATOR any
    more: at the probed 131k window that family is never undersized below
    4 x total_max, so calling it there RAISES rather than lying. That is the
    designed behaviour - see _largest_prompt_that_fits for the number the
    config block now quotes."""
    ceiling = 4 * cfg.build.length_band.total_max
    for size in range(1, ceiling):
        if family in undersized_families(cfg, role, size + reply_tokens):
            return size
    raise AssertionError(
        f"family {family!r} is never undersized for role {role!r} below "
        f"{ceiling} tokens - it is not in that pool at all"
    )


def _largest_prompt_that_fits(cfg, role: str, family: str, reply_tokens: int = 0) -> int:
    """The largest prompt (routing tokens, reply excluded) `family` still holds.

    The mirror of `_divert_point`, and the number the config block now quotes
    for the generator: since the 2026-08-19 probe the cliff sits at 104,858
    rather than 6,554, which is past `_divert_point`'s bound and would make it
    raise. Bisected rather than looped, because the range is now six digits.
    """
    lo, hi = 0, 4_000_000
    assert family not in undersized_families(cfg, role, lo + reply_tokens)
    assert family in undersized_families(cfg, role, hi + reply_tokens)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if family in undersized_families(cfg, role, mid + reply_tokens):
            hi = mid
        else:
            lo = mid
    return lo


def test_the_preflight_advises_the_thresholds_the_code_computes(cfg, keys):
    """The advice is read out of the preflight's own output, not recomputed
    beside it, because that string is what the operator shops against.

    This checks the NUMBERS, not the comment block that quotes them. It used
    to do both, and the prose half made every edit to that block a suite
    failure for no behaviour change - which is why the block grew instead of
    being pruned. The config's arithmetic was once pre-margin and ~40% high
    (it said ~4.2k where the real divert point is 2555, and ~7.2k where slot B
    really dies at 5531); what stopped that recurring is that these numbers
    are derived here rather than transcribed, and that survives the cut.

    Two numbers, not one, and the difference is easy to get wrong. Both
    thresholds are derived: the judge's is the bar any replacement judge must
    clear, and the tiebreak's is the bar any replacement tiebreak must clear.
    Which one is larger is NOT a design requirement, only a fact about which
    of the two prompts currently renders longer - the preflight sizes the two
    separately regardless of which way that falls, and a prompt edit to
    either template can flip it (judge-calibration Task 2, 2026-08-24, did
    exactly that).
    """
    from tuned.data.generate import preflight_messages

    # cfg_with_gpt_oss_reinstated_as_generator ADDED 2026-08-28: cerebras/
    # gpt-oss-120b is removed from routing.generator outright (operator
    # directive - deepseek is the sole generator, cerebras spends only on
    # judging). The JUDGE gap this test reads its advice off of only exists
    # on a gpt-oss-generated row - gpt-oss lumps groq/openai/gpt-oss-20b and
    # both openai backstops into one family, so removing the promoted judges
    # empties slot B for gpt-oss specifically; deepseek shares no family with
    # any of them and always has openai left, so it never gaps this way.
    # Reinstating gpt-oss as an additional real generator (the model itself
    # is untouched in cfg.providers) reproduces the pool this test's numbers
    # were designed against.
    cfg = cfg_with_gpt_oss_reinstated_as_generator(cfg)

    # THE SHIPPED POOL HAS NO GAP AT ALL since the 2026-08-19 judge surgery, so
    # each threshold is read from the pool that still produces its own kind of
    # gap. Both remain the numbers the code enforces; neither is transcribed.
    assert preflight_messages(cfg, ("generator",)) == ([], []), (
        "the probed and re-seated pool has no gap left to report"
    )

    def advice_from(lines):
        found = {
            int(line.split("max_context >= ")[1].split()[0])
            for line in lines
            if "max_context >= " in line
        }
        assert len(found) == 1, ("one number: the operator buys one model", found)
        return found.pop()

    # A JUDGE gap advises the judge threshold. Dropping gemma out of
    # routing.judge is the smallest pool that still has one.
    judge_refusals, _ = preflight_messages(
        cfg_without_the_promoted_judge(cfg), ("generator",)
    )
    required = advice_from(judge_refusals)

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
    # The tiebreak threshold is no longer ADVISED, but it is still the bar any
    # replacement tiebreak must clear, so it is still derived. It is NOT
    # reliably the larger of the two - that used to be true because the
    # tiebreak prompt rendered longer, but judge-calibration Task 2
    # (2026-08-24) split the grounding_faithfulness rubric's bands so that
    # absence of authority scores 3 and misstatement scores 2, which
    # lengthened judge_pointwise_v1 past judge_tiebreak_v1 and made the JUDGE
    # threshold the larger one instead. That was a deliberate consequence of
    # the rubric edit, not drift, and it is exactly the kind of thing a future
    # prompt edit can flip again - so this only asserts the two numbers are
    # DIFFERENT, never which one leads.
    tiebreak_required = required_context(
        worst_case_judge_tokens(cfg, prompt_id=TIEBREAK_PROMPT_ID)
    )
    assert tiebreak_required != judge_required
    # ...and it really is what a TIEBREAK gap prints: take
    # mistral-large-latest back out of the tiebreak seat and the
    # preflight asks for exactly what pool_gaps computes. That is NOT always
    # tiebreak_required in isolation - pool_gaps' own floor rule ("never falls
    # below required_context(needed_tokens)", i.e. the judge's flat number)
    # means a tiebreak-only gap is advised at max(judge_required,
    # tiebreak_required). Today that floor is judge_required (29,708 >
    # 29,666), so the number actually printed is the judge's, not the
    # tiebreak's own - the same rubric-split consequence as above, not a
    # second bug. The invariant this line checks - the preflight prints
    # exactly what the config block claims - holds regardless of which of the
    # two is larger.
    _, tiebreak_warnings = preflight_messages(
        cfg_without_the_free_tiebreak(cfg), ("generator",)
    )
    assert advice_from(tiebreak_warnings) == max(judge_required, tiebreak_required)


def test_the_config_block_records_the_probe_that_set_the_window(cfg, keys):
    """A max_context is a claim about someone else's server, and this one was
    wrong by 16x for the life of the project because nothing made it carry its
    evidence. The block must name the probe: date, both models, the
    prompt_tokens each accepted, and the status.

    Pinned because the cost is not hypothetical - the stale value unroutable'd
    85% of the statute_qa stream, and it survived three review rounds by
    looking like a fact."""
    # ANCHORED TO THE CEREBRAS BLOCK, not to the file. Deleting the whole
    # 41-line cerebras provenance block used to leave 5 of 6 assertions green,
    # because later blocks (the mistral fence, the lightning block) also carry
    # "2026-08-19" and "HTTP 200" - so the test passed while the evidence it
    # exists to protect was gone. Each assertion now carries a
    # cerebras-unique token.
    text = DATA_CONFIG.read_text(encoding="utf-8")
    for evidence in (
        "cerebras/gpt-oss-120b   prompt_tokens 13515   HTTP 200",
        "cerebras/gemma-4-31b    prompt_tokens 10252   HTTP 200",
    ):
        assert evidence in text, evidence
    # The date must sit ON one of those probe lines, not merely somewhere.
    assert "2026-08-19  cerebras/gpt-oss-120b" in text
    # ...and the tpm interaction the bigger window opens is answered in the
    # same place, because 131k against a 30k/minute budget is the first thing
    # a reader will worry about.
    assert "_need_tokens" in text
    assert "tpm 30,000" in text or "tpm STAYS 30,000" in text


def test_the_probed_windows_are_what_the_config_declares(cfg):
    """The 2026-08-19 probes, pinned as config values.

    This is the assertion that would have failed for the life of the project.
    Both cerebras models declared 8192 - a number nothing measured, traceable
    to a research note that read this build's 8192 TRAINING sequence length as
    the provider's context window - while the provider served 131,072. It cost
    85% of the statute_qa stream.

    What the wrong number cost downstream, measured: the generator's window
    feeds the per-family judge sizing, so at 8192 the gpt-oss family was
    checked at 19,108 tokens rather than 23,733, and the advice a tiebreak gap
    printed was 29,661 rather than 29,666.
    """
    caps = {
        f"{p.name}/{m.id}": m.limits.get("max_context")
        for p in cfg.providers
        for m in p.models
    }
    assert caps["cerebras/gpt-oss-120b"] == 131072
    assert caps["cerebras/gemma-4-31b"] == 131072
    # EVERY cerebras model, not just the two named: a third one added at the
    # old 8192 default is the same defect returning under a new id.
    for provider in cfg.providers:
        if provider.name == "cerebras":
            for model in provider.models:
                assert model.limits["max_context"] == 131072, model.id
    # Neither is the training sequence length, and the two must never be
    # confused again: 8192 belongs to the length band, not to a provider.
    assert cfg.build.length_band.total_max == 8192
    assert caps["cerebras/gpt-oss-120b"] != cfg.build.length_band.total_max


def test_a_statute_qa_sized_prompt_now_routes_to_the_generator(cfg):
    """THE ROUTING CONSEQUENCE, which is the whole point of the correction.

    statute_qa is the stream the stale pin destroyed: its prompt carries the
    seed AND the provision, so it is the longest generator prompt this build
    makes, and at 8192 the only generator family was excluded for anything
    over 6,554 routing tokens. The exclusion case is still constructible - the
    same sizes against a fixture-narrowed window - so this pins the CHANGE,
    not merely the current state.
    """
    probed, narrow = cfg, _narrow_generator(cfg)
    # RE-BASELINED 2026-08-28: the sole shipped generator family is deepseek
    # now, not gpt-oss - cerebras/gpt-oss-120b is removed from
    # routing.generator outright (operator directive: deepseek is the sole
    # generator). _narrow_generator narrows both families; only deepseek is
    # actually routed, so it is the one that excludes here.
    for needed in (7_000, 10_000, 13_515, 25_000):
        assert undersized_families(probed, "generator", needed) == frozenset(), needed
        assert undersized_families(narrow, "generator", needed) == frozenset({"deepseek"}), needed
    # 13,515 is not arbitrary: it is the prompt_tokens the live probe sent and
    # the provider accepted with a 200.
    assert undersized_families(probed, "generator", 13_515) == frozenset()


def test_the_tiebreak_route_is_open_for_a_long_row(cfg, keys):
    """The other half of the probe: gemma at 8192 meant no long row could
    reach a third family at all, so every tie on a full-length generation was
    undecidable by construction rather than by pool composition."""
    from tuned.data.generate import preflight_messages

    assert undersized_families(cfg, "tiebreak", 20_000) == frozenset()
    # ...and end to end: the shipped pool preflights CLEAN at every role. Two
    # separate fixes were needed and this assertion fails if either is undone -
    # the cerebras probes cleared the context-shaped tiebreak gap, and the
    # 2026-08-19 judge surgery cleared the separation-shaped one by giving the
    # tiebreak seat to mistral-large-latest.
    assert preflight_messages(cfg, ("generator", "judge", "tiebreak")) == ([], [])
    # Take that seat away and the separation-shaped gap comes straight back,
    # which is what says the seat is what closed it.
    #
    # cfg_with_gpt_oss_reinstated_as_generator ADDED 2026-08-28: cerebras/
    # gpt-oss-120b is removed from routing.generator outright (operator
    # directive - deepseek is the sole generator, cerebras spends only on
    # judging), so stripping mistral+bai from routing.tiebreak no longer
    # reopens a gap on the shipped pool by itself - deepseek shares no family
    # with groq/openai/gpt-oss-20b or the openai backstops, so it is never
    # excluded from them and the tiebreak stays open. Only a gpt-oss row
    # reproduces the separation-shaped gap this test is about (gpt-oss lumps
    # all three of those into one excluded family), so gpt-oss-120b is put
    # back as an additional real generator for this half of the test.
    _, without = preflight_messages(
        cfg_without_the_free_tiebreak(cfg_with_gpt_oss_reinstated_as_generator(cfg)),
        ("generator", "judge", "tiebreak"),
    )
    assert any("routing.tiebreak" in line for line in without)


def test_a_single_call_larger_than_tpm_waits_rather_than_deadlocking(cfg):
    """THE EDGE THE BIGGER WINDOW OPENS, answered rather than assumed.

    max_context is 131,072 and tpm is 30,000, so one call can be worth more
    than a whole minute's token budget. The bucket can never hold more than
    its capacity, so an unclamped wait condition would never become true and
    the worker would hang forever holding the lock.

    TokenBucket._need_tokens clamps the charge to capacity, which degrades to
    "wait for a full bucket, then send" - the provider's own 429 handling with
    backoff takes it from there. Asserted on a fake clock so the test neither
    sleeps nor dials out: a full bucket admits the call immediately, a drained
    one admits it after exactly one refill, and next_wait agrees with acquire.

    No per-call ceiling was added because none is needed; this test is what
    stops one being added later on a hunch, and what catches the clamp being
    removed.
    """
    tpm = 30_000
    oversize = 131_072
    assert oversize > tpm, "the premise: one call can exceed a minute's budget"

    now = [0.0]
    slept = []

    async def sleeper(delay):
        slept.append(delay)
        now[0] += delay
        assert len(slept) < 100, "spinning instead of waiting"

    bucket = TokenBucket(rpm=5, tpm=tpm, clock=lambda: now[0], sleeper=sleeper)
    # A full bucket takes it at once - the charge is capped at capacity.
    asyncio.run(asyncio.wait_for(bucket.acquire(oversize), timeout=5))
    assert slept == []
    # Drained, it waits exactly one refill and then proceeds.
    assert bucket.next_wait(oversize) == pytest.approx(60.0)
    asyncio.run(asyncio.wait_for(bucket.acquire(oversize), timeout=5))
    assert slept == [pytest.approx(60.0)]


def test_the_judge_pool_is_the_one_calibration_left_behind(cfg):
    """THE POOL COMPOSITION, pinned as config.

    mistral-small-latest was DISQUALIFIED as a judge on 2026-08-19 by human
    calibration - holdout precision 0.237, recall 1.000, phi 0.124 on n=40
    against a 21.7% base rate, where the gate is 0.75 - so it holds no judge
    seat. gemma took that seat. mistral-LARGE is a different model on different
    weights and is UNPROVEN rather than disqualified, so it sits in the
    tiebreak seat where its verdicts earn gold-label coverage.

    This asserts the shape, not the merits: which refs hold which roles, and
    that the disqualified model is not among the judges under any spelling.
    """
    roles = {
        f"{p.name}/{m.id}": tuple(m.roles) for p in cfg.providers for m in p.models
    }
    assert "mistral/mistral-small-latest" not in roles, (
        "the disqualified judge is back in the config"
    )
    assert roles["mistral/mistral-large-latest"] == ("tiebreak",), (
        "mistral-large is UNPROVEN; the tiebreak seat is where it gets proven, "
        "and a judge seat would need gold labels first"
    )
    assert roles["cerebras/gemma-4-31b"] == ("judge", "tiebreak")
    # bai/deepseek-v4-flash, 2026-08-27, and it is UNPROVEN in exactly the
    # sense mistral-large is: it was measured to ANSWER as a judge and never
    # measured to judge WELL, because no gold-labelled calibration has been
    # run on it. The answer-rate, stated as it was measured rather than at its
    # best batch: 9/10 on the pre-registered batch against a 9/10 pass line,
    # 20/21 pooled over every call. The one failure was NOT a bad verdict but
    # an empty one - HTTP 200, finish_reason=length, the whole 1,024-token
    # reply budget spent on reasoning - which is a live failure mode of this
    # provider and not a rounding error. See
    # docs/reports/2026-08-27-deepseek-as-judge-slot-b.md.
    # It holds a judge seat rather than a tiebreak-only one because the
    # alternative that day was not a proven judge: gemma answers HTTP 402 and
    # the next ref in the list is paid. That trade is the config's, not this
    # test's; what is pinned here is that it is visible.
    #
    # groq/openai/gpt-oss-20b, also 2026-08-27: on a DEEPSEEK generation,
    # family separation excludes deepseek itself, and the pool above (qwen,
    # gemma) left slot B with one candidate - gemma, HTTP 402 - so every
    # deepseek row parked in judge_error. It is UNPROVEN in the same sense as
    # mistral-large and deepseek: no gold-labelled calibration of it AS A
    # JUDGE has been run. What IS measured, and measured badly, is unrelated
    # to judging - 0/10 on IPC->BNS ground truth, recited from parameters with
    # no source text in front of it - which is exactly why routing.tiebreak
    # puts mistral-large ahead of it: that ordering keeps it out of the seat
    # that decides a contested row outright. Here it is not that seat; it is
    # one of two independent opinions, placed last among the free judges so
    # any more-trusted or revived free ref is still chosen first.
    assert roles["groq/openai/gpt-oss-20b"] == ("judge", "tiebreak", "probe")
    assert list(cfg.routing.judge) == [
        "groq/qwen/qwen3.6-27b",
        "cerebras/gemma-4-31b",
        "bai/deepseek-v4-flash",
        "groq/openai/gpt-oss-20b",
    ]
    assert "mistral/mistral-large-latest" in cfg.routing.tiebreak
    # mistral leads the tiebreak list - the ordering that keeps the deciding
    # seat off the family measured 0/10 on IPC->BNS recall. (The paid
    # backstops this comment used to order against left the fleet entirely
    # on 2026-08-28.)
    assert cfg.routing.tiebreak.index("mistral/mistral-large-latest") == 0


def test_the_probed_mistral_window_is_what_the_config_declares(cfg):
    """max_context is a claim about someone else's server, and this build has
    been bitten once by writing one down unprobed (cerebras, 8192 against a
    real 131,072, which unroutable'd 85% of a stream).

    52,812 is a MEASURED FLOOR: probed 2026-08-19, one call, max_tokens 16,
    temperature 0, HTTP 200, finish_reason stop. It is not the model's ceiling
    and the config says so. What it has to do is clear the worst-case tiebreak
    requirement, which is the assertion that matters here.
    """
    caps = {
        f"{p.name}/{m.id}": m.limits.get("max_context")
        for p in cfg.providers
        for m in p.models
    }
    assert caps["mistral/mistral-large-latest"] == 52812
    tiebreak_required = required_context(
        worst_case_judge_tokens(cfg, prompt_id=TIEBREAK_PROMPT_ID)
    )
    assert caps["mistral/mistral-large-latest"] > tiebreak_required, (
        "the probed floor must clear the longest tiebreak call this build makes"
    )
    # ...so it is never excluded on length at any row size the band permits.
    assert "mistral" not in undersized_families(cfg, "tiebreak", tiebreak_required)


def test_a_tiebreak_resolves_to_mistral_for_a_row_qwen_and_gemma_judged(cfg, keys):
    """THE CONSEQUENCE THAT CHANGED, and the reason mistral-large is in the
    pool at all.

    A gpt-oss generation is judged by qwen (slot A) and gemma (slot B). The
    tiebreak then excludes the generator's family AND both judges' -
    {gpt-oss, qwen, gemma} - which before 2026-08-19 was every family
    routing.tiebreak contained, so a disagreement had no third opinion and took
    judge.py's park-loudly path. mistral is the one family that survives that
    exclusion, which is exactly the seat it was given.

    Walked through the Router's own eligible_refs, because that is what
    judge.py calls; the exclusion set is built the same way judge.py builds it
    (base_exclude = the generator's family, plus each answered slot's).
    """
    router = Router(cfg)
    families = {
        f"{p.name}/{m.id}": m.family for p in cfg.providers for m in p.models
    }
    base = frozenset({"gpt-oss"})
    slot_a = next(router.eligible_refs("judge", exclude_families=base))
    fam_a = families[f"{slot_a.provider}/{slot_a.model}"]
    slot_b = next(router.eligible_refs("judge", exclude_families=base | {fam_a}))
    fam_b = families[f"{slot_b.provider}/{slot_b.model}"]
    assert (slot_a, slot_b) == (GROQ_JUDGE, GEMMA_JUDGE)
    assert {fam_a, fam_b} == {"qwen", "gemma"}

    tiebreak = next(
        router.eligible_refs("tiebreak", exclude_families=base | {fam_a, fam_b}), None
    )
    assert tiebreak == ModelRef("mistral", "mistral-large-latest")

    # ...and it is the ONLY thing holding that seat open: take it out and the
    # row is back to two judges and a park.
    without = Router(cfg_without_the_free_tiebreak(cfg))
    assert next(
        without.eligible_refs("tiebreak", exclude_families=base | {fam_a, fam_b}), None
    ) is None


def test_the_judge_sizer_only_narrows_below_the_flat_worst_case(cfg, keys, monkeypatch):
    """WHERE THE PER-FAMILY NARROWING ACTUALLY TURNS ON, measured.

    `needed_for_window` exists so a generator family too small to produce the
    length band's longest row does not have its judge slots checked at that
    length. The curve has a cliff and the fixtures were on the wrong side of
    it: pipeline_fakes.SECOND_GENERATOR_CONTEXT (32,000) narrows NOTHING, and
    the comment there claimed the opposite until the review set it to 131072
    and found the whole suite still green.

    So the curve is pinned here instead. A change to the sizing rule moves
    these numbers and fails loudly, which is the protection the inert constant
    was pretending to give.
    """
    two = cfg_with_two_generator_families(cfg)
    # The 2026-08-28 alignment (GENERATION_OUTPUT_TOKENS 16384) makes the
    # narrowing unreachable at any window on the shipped constant - the
    # cliff itself is the subject here, so the curve is pinned at the
    # pre-alignment budget where it exists. See
    # test_the_advice_never_falls_below_the_flat_worst_case.
    monkeypatch.setattr("tuned.data.generate.GENERATION_OUTPUT_TOKENS", 4000)
    sizer, flat = judge_sizer(two), worst_case_judge_tokens(two)

    # At and above the cliff the hook is a no-op...
    assert sizer(16384, "judge") == flat
    assert sizer(SECOND_GENERATOR_CONTEXT, "judge") == flat
    assert sizer(131072, "judge") == flat
    assert sizer(None, "judge") == flat
    # ...and below it the family is checked at what it can really produce.
    assert sizer(12000, "judge") < flat
    assert sizer(NARROW_GENERATOR_CONTEXT, "judge") < sizer(12000, "judge")
    # The fixture window and the constant cannot drift apart.
    windows = {
        m.id: m.limits["max_context"]
        for p in two.providers
        for m in p.models
        if m.id == "second-generator"
    }
    assert windows["second-generator"] == SECOND_GENERATOR_CONTEXT


def test_the_generator_prefers_the_free_provider_and_parks_once_all_are_ineligible(cfg, keys):
    """THE COST POLICY IS THE LIST ORDER, so it is pinned like one.

    Lightning - the sole paid ref this role ever had - was removed from
    routing.generator on 2026-08-27: a usd_cap declared on any provider but
    the one literally named "openai" was silently unreachable
    (generate._provider_usd_cap, née _openai_usd_cap, and the
    `if provider == "openai":` gate around it), so lightning carried a paid
    ref with no fence actually behind it. It is not replaced with another
    paid ref here - that would just re-open the same hole with a different
    name - so this role now has no paid overflow at all.

    Router.pick walks routing.generator in order and only moves on when a
    ref is INELIGIBLE. With nothing paid left to fail over to, once every
    free ref is ineligible pick returns None and the row parks rather than
    silently spending money - that is the intended trade
    routing.generator's own comment records. This asserts the ORDERING
    preference among whichever free refs are configured, and the parking
    outcome, rather than pinning specific provider names - those have
    already reordered once (bai overtook cerebras 2026-08-25) without the
    cost policy itself breaking.

    THE "no paid overflow" GUARD WAS NAME-BASED UNTIL A 2026-08-27 REVIEW
    caught it: it hardcoded `"lightning/lightning-ai/gpt-oss-120b" not in
    generator_refs`, which is Task 1's exact bug - a fence that recognises
    only one provider by literal name - reborn as a test assertion instead of
    production code. A future paid ref under any OTHER name would pass this
    check while reopening the hole lightning was removed to close.

    THE FIRST REPLACEMENT (same day) TRADED COVERAGE RATHER THAN WIDENING IT,
    and a second review caught that too: a price-only invariant - "any ref
    that declares a PRICE also declares a usd_cap" - cannot see a provider
    that declares NEITHER, and lightning is exactly that provider (see its
    block: no usd_per_1m_prompt/completion, no usd_cap, PAID regardless).
    Verified directly: wiring in a priced-and-uncapped ref still fails this
    test; re-adding "lightning/lightning-ai/gpt-oss-120b" UNCHANGED passed it,
    because there was nothing price-shaped to trip on.

    So paid-ness cannot be inferred from price presence alone - a provider
    that declares no prices is not thereby free. PAID_PROVIDERS below is a
    maintained allowlist, the same shape as pipeline_fakes.PROMOTED_JUDGES:
    it does not derive from anything else in the config, so a NEW paid
    provider silently passes this test until someone adds its name here -
    which is a real, named risk rather than a hidden one, and cheaper than
    the alternative of guessing paid-ness from partial declarations.
    """
    generator_refs = list(cfg.routing.generator)
    for ref in cfg.routing_refs("generator"):
        provider, model = cfg.model_for(ref)
        priced = (
            "usd_per_1m_prompt" in model.limits or "usd_per_1m_completion" in model.limits
        )
        paid = priced or provider.name in PAID_PROVIDERS
        assert not paid or "usd_cap" in model.limits, (
            f"{ref.provider}/{ref.model} is priced, or under a known-paid "
            "provider, but declares no usd_cap - unmetered paid overflow "
            "would be reachable from routing.generator"
        )
    refs = [ModelRef(*ref.split("/", 1)) for ref in generator_refs]
    assert refs, "need at least one generator ref"

    # Preference: with everything eligible, a free provider answers.
    assert _router(cfg).pick("generator").ref in refs

    # ...and once every ref is cooling there is nothing left to fail over to
    # - no paid overflow exists in this role any more - so the row parks.
    cooling = _router(cfg)
    for ref in refs:
        for _ in range(11):
            cooling.report_failure(ref)
    assert cooling.pick("generator") is None


def test_a_lightning_reply_assembles_into_a_real_think_block(cfg):
    """The other end of the same check: what the gates will actually see.

    A traceless park is the failure this is guarding against, so the assertion
    is that `think` is NOT None and that the content carries the trainer's tag
    pair around the reasoning the provider returned.
    """
    from types import SimpleNamespace

    from tuned.data.generate import assemble_content

    response = SimpleNamespace(
        text="The tenant may withhold rent.",
        reasoning="First, identify the covenant. Then the remedy.",
    )
    content, think, answer = assemble_content(cfg, response)
    assert think == "First, identify the covenant. Then the remedy."
    assert answer == "The tenant may withhold rent."
    assert content.startswith(cfg.think_open)
    assert cfg.think_close in content
    # ...and the traceless shape still parks rather than inventing a trace.
    bare = SimpleNamespace(text="No trace here.", reasoning=None)
    assert assemble_content(cfg, bare)[1] is None


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
    assert list(router.eligible_refs("judge", exclude_families=frozenset({"qwen"}),
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
    # Shipped reality: mistral-large's tiebreak layer is its temperature, and
    # asking for any other role gets nothing from this layer. It is the only
    # model left carrying a role_params block with a temperature in it, which
    # is why the worked example moved here on 2026-08-19.
    mistral_tb = ModelRef("mistral", "mistral-large-latest")
    assert _params_for_role(cfg, mistral_tb, "tiebreak") == {"temperature": 0.2}
    assert _params_for_role(cfg, mistral_tb, "judge") == {}
    assert _params_for_role(cfg, mistral_tb, None) == {}

    # The two-role half is read against the fixture's own family, because no
    # shipped model serves both a generator and a judge role since 2026-08-19.
    promoted = cfg_with_two_generator_families(cfg)
    second = ModelRef("cerebras", "second-generator")
    assert _params_for_role(promoted, second, "generator") == {
        "temperature": 0.7,
        "top_p": 0.95,
    }
    assert _params_for_role(promoted, second, "judge") == {"temperature": 0.2}


def _payload_for_role(cfg, ref, role):
    """The WHOLE sampling payload a call in `role` would put on the wire."""
    client = _router(cfg).routed(ref).client
    payload = client.build_payload(
        ChatRequest(messages=({"role": "user", "content": "x"},), ref=ref, role=role)
    )
    return {k: v for k, v in payload.items() if k not in ("model", "messages")}


def test_the_qwen_judge_sends_reasoning_effort_none_and_only_as_a_judge(cfg, keys):
    """qwen3.6-27b cannot answer as a judge with reasoning on: measured
    2026-08-18, 7 slot-B calls spent exactly 1,024 completion tokens each
    inside an unclosed <think> and returned no verdict at all. `none` is
    Groq's documented off switch for this family and was live-probed at 200.

    IN role_params, NOT params, and that is the assertion the second half
    makes. The same key in `params` would follow this model into a generator
    promotion and produce a traceless generator - the pilot failure the mistral
    block records, which cost 43 traceless generations before it was found.
    build_payload keys the middle layer off the CALL's role, so a call made in
    any other role gets the model's own defaults and nothing else.

    The generator's own reasoning setting is checked in the same breath
    because it is the value this change must not touch: cerebras/gpt-oss-120b
    declares `reasoning_effort: medium` and reasons by default, and the
    retired effort ladder is the measurement that says leave it alone."""
    assert _payload_for_role(cfg, GROQ_JUDGE, "judge") == {
        "temperature": 0.2,
        "reasoning_effort": "none",
    }
    # No role layer -> the model's own params, and the suppression is not there.
    assert _payload_for_role(cfg, GROQ_JUDGE, None) == {"temperature": 0.2}
    assert _payload_for_role(cfg, GROQ_JUDGE, "generator") == {"temperature": 0.2}
    # It reaches no other judge: gemma took a judge role on 2026-08-19 and
    # carries no role layer, so the suppression must not follow it.
    assert "reasoning_effort" not in _payload_for_role(cfg, GEMMA_JUDGE, "judge")
    # ...and the generator's own effort is untouched, in every role.
    generator = ModelRef("cerebras", "gpt-oss-120b")
    assert _payload_for_role(cfg, generator, "generator")["reasoning_effort"] == "medium"
    assert _payload_for_role(cfg, generator, "judge")["reasoning_effort"] == "medium"


def test_role_params_do_not_leak_between_models(cfg, keys):
    """A model with no role_params is unaffected: cerebras keeps its single
    configured temperature whichever role asks for it."""
    ref = ModelRef("cerebras", "gpt-oss-120b")
    assert _params_for_role(cfg, ref, "generator") == _params_for_role(cfg, ref, "judge")


def test_a_per_call_param_still_beats_the_role_layer(cfg, keys):
    """Precedence is model.params < role_params[role] < per-call params. The
    caller has to stay able to override, or judge.py's own temperature would be
    silently ignored the day a role entry is added for its model."""
    client = _router(cfg).routed(GROQ_JUDGE).client
    payload = client.build_payload(
        ChatRequest(
            messages=({"role": "user", "content": "x"},),
            ref=GROQ_JUDGE,
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
    broken = raw.replace("          tiebreak: {temperature: 0.2}",
                         "          tiebreakk: {temperature: 0.2}")
    assert broken != raw
    path = tmp_path / "bad_role.yaml"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_build_config(str(path), allow_unpinned=True)
    assert "tiebreakk" in str(exc.value) and "role_params" in str(exc.value)


# --------------------------------------------------------------------------
# Observed quota: the headers the budget gate is now the authority for.
# --------------------------------------------------------------------------

def _quota_client(cfg, handler, quota):
    provider = next(p for p in cfg.providers if p.name == "cerebras")
    model = next(m for m in provider.models if m.id == "gpt-oss-120b")
    return ChatClient(
        provider, model, transport=httpx.MockTransport(handler),
        clock=FakeClock(), sleeper=lambda d: asyncio.sleep(0), rng=random.Random(0),
        max_retries=1, quota=quota,
    )


def test_rate_limit_headers_are_captured_from_a_200(cfg, keys):
    quota = QuotaLedger()
    headers = {
        "x-ratelimit-limit-tokens-day": "1000000",
        "x-ratelimit-remaining-tokens-day": "458408",
        "x-ratelimit-remaining-requests-day": "2399",
    }
    client = _quota_client(cfg, lambda r: httpx.Response(200, json=_body(), headers=headers), quota)
    asyncio.run(client.complete(ChatRequest(messages=({"role": "user", "content": "x"},),
                                            ref=ModelRef("cerebras", "gpt-oss-120b"))))
    obs = quota.observation("cerebras", "gpt-oss-120b")
    assert obs is not None
    assert obs.remaining_tokens_day == 458408
    assert obs.limit_tokens_day == 1000000


def test_rate_limit_headers_are_captured_from_a_429_too(cfg, keys):
    """The 429 is the response that says the window really IS spent, so it is
    the one the gate most needs. Capturing only 200s would leave the gate
    trusting the last healthy number right through an exhaustion."""
    quota = QuotaLedger()
    headers = {"x-ratelimit-limit-tokens-day": "1000000",
               "x-ratelimit-remaining-tokens-day": "0"}
    client = _quota_client(cfg, lambda r: httpx.Response(429, text="slow down", headers=headers), quota)
    with pytest.raises(ProviderError):
        asyncio.run(client.complete(ChatRequest(messages=({"role": "user", "content": "x"},),
                                                ref=ModelRef("cerebras", "gpt-oss-120b"))))
    obs = quota.observation("cerebras", "gpt-oss-120b")
    assert obs is not None and obs.remaining_tokens_day == 0
    assert obs.allows(1) is False


def test_a_response_without_quota_headers_leaves_no_observation(cfg, keys):
    quota = QuotaLedger()
    client = _quota_client(cfg, lambda r: httpx.Response(200, json=_body()), quota)
    asyncio.run(client.complete(ChatRequest(messages=({"role": "user", "content": "x"},),
                                            ref=ModelRef("cerebras", "gpt-oss-120b"))))
    assert quota.observation("cerebras", "gpt-oss-120b") is None
