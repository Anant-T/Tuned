"""Harmony Completions prefill: renderer, parser, and the /v1/completions wire.

Chat Completions `assistant.reasoning` did not bind gpt-oss's analysis
channel (12/12 dialect rows still opened on instruction restatement). This
module is the hosted alternative: render a Harmony prompt that already
contains the first analysis tokens, complete it, parse analysis vs final.
"""

import asyncio
import json
import random
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")

from tuned.data.harmony import (
    DEFAULT_PREFILL,
    HARMONY_STOP,
    S1_WAIT,
    needs_s1_continue,
    parse_completion,
    render_for_analysis_prefill,
    restated_opening,
    stitch_s1,
)
from tuned.data.providers import ChatClient, ChatRequest
from tuned.data.config import ModelCfg, ModelRef, ProviderCfg


# --- renderer ---------------------------------------------------------------


def test_render_puts_pipeline_system_on_developer_and_prefills_analysis():
    prompt = render_for_analysis_prefill(
        [
            {"role": "system", "content": "You are a judge of an Indian court."},
            {"role": "user", "content": "Decide whether s.302 applies."},
        ],
        DEFAULT_PREFILL,
        reasoning_effort="medium",
        current_date="2026-08-21",
    )

    assert prompt.startswith("<|start|>system<|message|>")
    assert "Reasoning: medium" in prompt
    assert "Current date: 2026-08-21" in prompt
    assert "<|start|>developer<|message|>" in prompt
    assert "You are a judge of an Indian court." in prompt
    assert prompt.count("<|start|>system<|message|>") == 1
    assert "<|start|>user<|message|>Decide whether s.302 applies.<|end|>" in prompt
    assert prompt.endswith(
        "<|start|>assistant<|channel|>analysis<|message|>" + DEFAULT_PREFILL
    )
    # Mid-message: no closing token after the prefill, or the model would
    # start a new channel instead of continuing analysis.
    assert not prompt.endswith("<|end|>")
    # openai-harmony SystemContent, not a hand-rolled identity string.
    assert "You are ChatGPT, a large language model trained by OpenAI." in prompt
    assert "Knowledge cutoff: 2024-06" in prompt


def test_render_concatenates_harmony_turns_without_inserted_newlines():
    prompt = render_for_analysis_prefill(
        [{"role": "user", "content": "Hi"}],
        "Let me check ",
        current_date="2026-08-21",
    )
    assert "><|start|>" in prompt
    assert "\n<|start|>user" not in prompt


# --- parser -----------------------------------------------------------------


def test_parse_continues_analysis_then_splits_final_channel():
    generated = (
        "whether s.302 is made out on these facts. Actually, the injury is not fatal.\n"
        "<|end|><|start|>assistant<|channel|>final<|message|>"
        "Issue\nWhether s.302 applies.\n\nConclusion\nIt does not.<|return|>"
    )
    parsed = parse_completion(generated, DEFAULT_PREFILL)

    assert parsed.continued is True
    assert parsed.restated is False
    assert parsed.analysis_continuation.startswith("whether s.302")
    assert parsed.think.startswith(DEFAULT_PREFILL)
    assert "whether s.302 is made out" in parsed.think
    assert parsed.final.startswith("Issue")
    assert "<|return|>" not in parsed.final
    assert "<|end|>" not in parsed.think


def test_parse_does_not_prepend_prefill_when_the_model_restates_instructions():
    generated = (
        "We need to produce a legal analysis in first person with 450-700 words."
    )
    parsed = parse_completion(generated, DEFAULT_PREFILL)

    assert parsed.continued is False
    assert parsed.restated is True
    assert parsed.think == generated
    assert not parsed.think.startswith(DEFAULT_PREFILL)


def test_parse_does_not_double_the_prefill_when_the_model_echoes_it():
    generated = DEFAULT_PREFILL + "whether the notice was served in time."
    parsed = parse_completion(generated, DEFAULT_PREFILL)

    assert parsed.continued is True
    assert parsed.think.startswith(DEFAULT_PREFILL)
    assert parsed.think.count("I start from the facts") == 1


def test_parse_handles_stripped_assistantfinal_marker():
    generated = (
        "the limitation clock. Am I sure about the date?\n"
        "assistantfinalIssue\nLimitation bars the suit."
    )
    parsed = parse_completion(generated, DEFAULT_PREFILL)
    assert parsed.continued is True
    assert parsed.final.startswith("Issue")
    assert "assistantfinal" not in parsed.think
    assert "assistantfinal" not in parsed.final


def test_restated_opening_detects_instruction_echo():
    assert restated_opening("We need to produce a reasoning in first person")
    assert restated_opening("We need to produce a response/reasoning in first person")
    assert restated_opening("The user wants a 450-700 word deliberation")
    assert not restated_opening("whether the complaint discloses an offence")
    assert not restated_opening("I need to write the plaint from the pleaded facts")
    assert not restated_opening("I must produce the written statement next")


def test_s1_wait_is_not_itself_a_verification_cue():
    from tuned.data.harmony import HarmonyParse, continuation_has_cue

    assert S1_WAIT == " Wait"
    assert not continuation_has_cue("the notice was served on the facts.")
    assert not continuation_has_cue("the notice was served." + S1_WAIT)
    assert continuation_has_cue("the notice was served. Actually, the date is wrong.")
    first = HarmonyParse(
        analysis_continuation="the notice was served on the facts.",
        think=DEFAULT_PREFILL + "the notice was served on the facts.",
        final="",
        continued=True,
        restated=False,
    )
    assert needs_s1_continue(first)
    second = HarmonyParse(
        analysis_continuation="Actually, the limitation clock had already run.",
        think="Actually, the limitation clock had already run.",
        final="Issue\nLimitation bars the suit.",
        continued=True,
        restated=False,
    )
    stitched = stitch_s1(first, second, DEFAULT_PREFILL)
    assert S1_WAIT in stitched.think
    assert stitched.think.startswith(DEFAULT_PREFILL)
    assert "Actually, the limitation clock" in stitched.think
    assert stitched.final.startswith("Issue")
    assert needs_s1_continue(stitched) is False


# --- Completions wire -------------------------------------------------------


def _provider():
    return ProviderCfg(
        name="cerebras",
        base_url="https://api.test.local/v1",
        api_key_env="TUNED_TEST_KEY",
        quirks=("cerebras",),
        models=(),
    )


def _model():
    return ModelCfg(
        id="gpt-oss-120b",
        family="gpt-oss",
        roles=("generator",),
        limits={"max_output": 4096},
        params={"temperature": 0.7, "top_p": 0.95, "reasoning_effort": "medium"},
    )


def _complete(client, req):
    async def run():
        try:
            return await client.complete(req)
        finally:
            await client.aclose()

    return asyncio.run(run())


def test_completions_payload_drops_chat_only_fields_and_posts_completions(monkeypatch):
    monkeypatch.setenv("TUNED_TEST_KEY", "sk-secret")
    seen = []

    def handler(request):
        seen.append(request)
        body = json.loads(request.content)
        assert "messages" not in body
        assert "reasoning_effort" not in body
        return httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "object": "text_completion",
                "choices": [
                    {
                        "index": 0,
                        "text": "whether notice was served. assistantfinalIssue\nYes.",
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
            },
        )

    client = ChatClient(
        _provider(),
        _model(),
        transport=httpx.MockTransport(handler),
        rng=random.Random(1),
    )
    prompt = render_for_analysis_prefill(
        [{"role": "user", "content": "Was notice served?"}],
        DEFAULT_PREFILL,
        current_date="2026-08-21",
    )
    response = _complete(
        client,
        ChatRequest(
            messages=({"role": "user", "content": "Was notice served?"},),
            ref=ModelRef("cerebras", "gpt-oss-120b"),
            prompt=prompt,
            max_tokens=64,
            role="generator",
        ),
    )

    assert len(seen) == 1
    assert str(seen[0].url) == "https://api.test.local/v1/completions"
    payload = json.loads(seen[0].content)
    assert payload["prompt"] == prompt
    assert payload["model"] == "gpt-oss-120b"
    assert payload["max_tokens"] == 64
    assert payload["stop"] == list(HARMONY_STOP)
    assert payload["temperature"] == 0.7
    assert response.text.startswith("whether notice was served")
    assert response.reasoning is None


def test_generate_once_harmony_completions_sends_prefill_and_stores_continued_think(tmp_path):
    from dataclasses import replace

    from pipeline_fakes import FakeRouter, build_cfg, chat_response, open_store, paths_for
    from tuned.data.generate import generate_once
    from tuned.data.tasks import plan_wave

    cfg = replace(
        build_cfg(),
        build=replace(
            build_cfg().build,
            harmony_completions=True,
            harmony_prefill=DEFAULT_PREFILL,
        ),
    )
    generated = (
        "whether the complaint discloses an offence under s.302. Am I sure about the date?\n"
        "<|end|><|start|>assistant<|channel|>final<|message|>"
        "Issue\nWhether s.302 applies.\n\nRule\nSection 302.\n\n"
        "Application\nThe facts do not make it out.\n\nConclusion\nIt does not."
    )
    router = FakeRouter(cfg, script={"generator": [chat_response(text=generated, reasoning=None)]})
    store = open_store(tmp_path, n_seeds=1)
    plan_wave(store, cfg, "synthesis", 1, task_type_mix={"irac_analysis": 1.0})
    task = store.claim_tasks("w", 1)[0]
    result = asyncio.run(
        generate_once(store, cfg, router, task, paths=paths_for(tmp_path), attempt=1)
    )
    assert result.ok
    call = router.calls_for("generator")[0]
    assert call.get("prompt", "").endswith(DEFAULT_PREFILL)
    assert "<|start|>assistant<|channel|>analysis<|message|>" in call["prompt"]
    gen = store.latest_generation(task["task_id"])
    assert gen["think"].startswith(DEFAULT_PREFILL)
    assert "whether the complaint discloses" in gen["think"]
    assert gen["answer"].startswith("Issue")


def test_generate_once_s1_continue_appends_wait_when_first_think_has_no_cue(tmp_path):
    from dataclasses import replace

    from pipeline_fakes import FakeRouter, build_cfg, chat_response, open_store, paths_for
    from tuned.data.generate import generate_once
    from tuned.data.harmony import S1_WAIT
    from tuned.data.tasks import plan_wave

    cfg = replace(
        build_cfg(),
        build=replace(
            build_cfg().build,
            harmony_completions=True,
            harmony_prefill=DEFAULT_PREFILL,
            harmony_s1_continue=True,
        ),
    )
    first = (
        "whether the complaint discloses an offence under s.302 on these papers.\n"
        "<|end|><|start|>assistant<|channel|>final<|message|>"
        "Issue\nWhether s.302 applies.\n\nConclusion\nIt does not."
    )
    second = (
        "the dates do not line up. Actually, limitation had already run.\n"
        "<|end|><|start|>assistant<|channel|>final<|message|>"
        "Issue\nWhether s.302 applies.\n\nRule\nSection 302.\n\n"
        "Application\nThe facts do not make it out.\n\nConclusion\nIt does not."
    )
    router = FakeRouter(
        cfg,
        script={
            "generator": [
                chat_response(text=first, reasoning=None),
                chat_response(text=second, reasoning=None),
            ]
        },
    )
    store = open_store(tmp_path, n_seeds=1)
    plan_wave(store, cfg, "synthesis", 1, task_type_mix={"irac_analysis": 1.0})
    task = store.claim_tasks("w", 1)[0]
    result = asyncio.run(
        generate_once(store, cfg, router, task, paths=paths_for(tmp_path), attempt=1)
    )
    assert result.ok
    calls = router.calls_for("generator")
    assert len(calls) == 2
    assert calls[0].get("prompt", "").endswith(DEFAULT_PREFILL)
    assert calls[1].get("prompt", "").endswith(S1_WAIT)
    assert "whether the complaint discloses" in (calls[1].get("prompt") or "")
    assert "assistant<|channel|>final" not in (calls[1].get("prompt") or "")
    gen = store.latest_generation(task["task_id"])
    assert gen["think"].startswith(DEFAULT_PREFILL)
    assert S1_WAIT in gen["think"]
    assert "Actually, limitation had already run" in gen["think"]
    assert gen["answer"].startswith("Issue")


def _harmony_completion(analysis: str, answer: str) -> str:
    return (
        f"{analysis}\n"
        "<|end|><|start|>assistant<|channel|>final<|message|>"
        f"{answer}"
    )


_S1_NO_CUE_CONT = (
    "The right pleaded rests on the section both parties invoke, and the "
    "question is whether the facts as recorded bring the case within it. "
) * 8


def test_s1_stitch_uses_the_last_call_finish_reason_when_the_second_completes(tmp_path):
    """A first Completions call can end on length and still s1-continue.

    Truncation is a fact about the LAST provider call. After a successful
    stitch whose second call stopped cleanly, the first length must not
    force regeneration.
    """
    from dataclasses import replace

    from pipeline_fakes import (
        CLEAN_ANSWER,
        CLEAN_THINK,
        FakeRouter,
        build_cfg,
        chat_response,
        open_store,
        paths_for,
    )
    from tuned.data.generate import apply_gate_disposition, generate_once
    from tuned.data.tasks import plan_wave

    cfg = replace(
        build_cfg(),
        build=replace(
            build_cfg().build,
            harmony_completions=True,
            harmony_prefill=DEFAULT_PREFILL,
            harmony_s1_continue=True,
        ),
    )
    first = _harmony_completion(_S1_NO_CUE_CONT, "partial")
    second = _harmony_completion(CLEAN_THINK, CLEAN_ANSWER)
    router = FakeRouter(
        cfg,
        script={
            "generator": [
                chat_response(text=first, reasoning=None, finish_reason="length"),
                chat_response(text=second, reasoning=None, finish_reason="stop"),
            ]
        },
    )
    store = open_store(tmp_path, n_seeds=1)
    plan_wave(store, cfg, "synthesis", 1, task_type_mix={"irac_analysis": 1.0})
    task = store.claim_tasks("w", 1)[0]
    result = asyncio.run(
        generate_once(store, cfg, router, task, paths=paths_for(tmp_path), attempt=1)
    )
    assert result.ok
    assert len(router.calls_for("generator")) == 2
    assert "truncated" not in result.failed_gates
    assert result.disposition is None
    gen = store.latest_generation(task["task_id"])
    assert gen["finish_reason"] == "stop"
    assert apply_gate_disposition(store, task, result, worker_id="w") == "judging"
    assert store.events("generation_truncated") == []


def test_s1_stitch_stays_truncated_when_the_second_call_is_still_length(tmp_path):
    """The last call is the one that can still cut the stitched final short.

    A first call that stopped cleanly must not hide a second call that
    returned length / max_tokens.
    """
    from dataclasses import replace

    from pipeline_fakes import (
        CLEAN_ANSWER,
        CLEAN_THINK,
        FakeRouter,
        build_cfg,
        chat_response,
        open_store,
        paths_for,
    )
    from tuned.data.generate import apply_gate_disposition, generate_once
    from tuned.data.tasks import plan_wave

    cfg = replace(
        build_cfg(),
        build=replace(
            build_cfg().build,
            harmony_completions=True,
            harmony_prefill=DEFAULT_PREFILL,
            harmony_s1_continue=True,
        ),
    )
    first = _harmony_completion(_S1_NO_CUE_CONT, "partial")
    second = _harmony_completion(CLEAN_THINK, CLEAN_ANSWER)
    router = FakeRouter(
        cfg,
        script={
            "generator": [
                chat_response(text=first, reasoning=None, finish_reason="stop"),
                chat_response(text=second, reasoning=None, finish_reason="length"),
            ]
        },
    )
    store = open_store(tmp_path, n_seeds=1)
    plan_wave(store, cfg, "synthesis", 1, task_type_mix={"irac_analysis": 1.0})
    task = store.claim_tasks("w", 1)[0]
    result = asyncio.run(
        generate_once(store, cfg, router, task, paths=paths_for(tmp_path), attempt=1)
    )
    assert result.ok
    assert len(router.calls_for("generator")) == 2
    assert "truncated" in result.failed_gates
    assert result.disposition == "regenerate"
    gen = store.latest_generation(task["task_id"])
    assert gen["finish_reason"] == "length"
    assert apply_gate_disposition(store, task, result, worker_id="w") == "pending"
    assert store.events("generation_truncated")


def test_overlay_strips_word_count_packet_without_moving_live_shas():
    from tuned.data import prompt_registry as reg
    from test_build_prompts import EXPECTED_SHAS

    overlay = Path(__file__).parent.parent / "src" / "tuned" / "data" / "prompts_harmony"
    live = reg.load("gen_irac_analysis_v1")
    assert "450 to 700" in (live.user or "")
    assert live.sha == EXPECTED_SHAS["gen_irac_analysis_v1"]
    try:
        reg.set_overlay(overlay)
        over = reg.load("gen_irac_analysis_v1")
        assert "450 to 700" not in (over.user or "")
        assert over.sha != live.sha
        assert "Work it out before you commit" in (over.user or "")
    finally:
        reg.set_overlay(None)
    restored = reg.load("gen_irac_analysis_v1")
    assert restored.sha == live.sha


def test_exp_harmony_openai_judge_is_split_from_gpt_oss_under_usd_cap():
    """Live lumps gpt-5-* as family gpt-oss so they never grade gpt-oss-120b.

    The Harmony experiment splits that family so OpenAI can judge those rows,
    under a $2 TOTAL hard usd_cap. Without the split, OPENAI_API_KEY is loaded
    and then skipped on every call.
    """
    from tuned.data.config import load_build_config

    root = Path(__file__).parent.parent
    cfg = load_build_config(
        root / "configs" / "data_law_v1_exp_harmony.yaml", allow_unpinned=True
    )
    _, mini = cfg.model_for(ModelRef("openai", "gpt-5-mini"))
    _, nano = cfg.model_for(ModelRef("openai", "gpt-5-nano"))
    assert mini.family == "gpt-5"
    assert nano.family == "gpt-5"
    assert mini.limits["usd_cap"] == 2.0
    assert nano.limits["usd_cap"] == 2.0


def test_loading_the_live_config_clears_a_prior_harmony_overlay():
    from tuned.data import prompt_registry as reg
    from tuned.data.config import load_build_config
    from test_build_prompts import EXPECTED_SHAS

    root = Path(__file__).parent.parent
    load_build_config(root / "configs" / "data_law_v1_exp_harmony.yaml", allow_unpinned=True)
    overlaid = reg.load("gen_irac_analysis_v1")
    assert "450 to 700" not in (overlaid.user or "")
    load_build_config(root / "configs" / "data_law_v1.yaml", allow_unpinned=True)
    live = reg.load("gen_irac_analysis_v1")
    assert live.sha == EXPECTED_SHAS["gen_irac_analysis_v1"]
    assert "450 to 700" in (live.user or "")


def test_live_prompt_sha_is_stable_after_a_prior_harmony_config_load():
    """Independently load Harmony, then prove live SHAs return after reset.

    Must not rely on collection order. The baseline leak was: Harmony yaml
    arms the overlay and a later load() sees overlay bytes.
    """
    from tuned.data import prompt_registry as reg
    from tuned.data.config import load_build_config
    from test_build_prompts import EXPECTED_SHAS

    root = Path(__file__).parent.parent
    load_build_config(root / "configs" / "data_law_v1_exp_harmony.yaml", allow_unpinned=True)
    overlaid = reg.load("gen_irac_analysis_v1")
    assert overlaid.sha != EXPECTED_SHAS["gen_irac_analysis_v1"]
    assert "450 to 700" not in (overlaid.user or "")
    load_build_config(root / "configs" / "data_law_v1.yaml", allow_unpinned=True)
    live = reg.load("gen_irac_analysis_v1")
    assert live.sha == EXPECTED_SHAS["gen_irac_analysis_v1"]
    assert "450 to 700" in (live.user or "")


OVERLAY_DIR = Path(__file__).parent.parent / "src" / "tuned" / "data" / "prompts_harmony"
IRAC_ANSWER_CONTRACT = (
    "under four headings, each on its own line — Issue, Rule, Application, Conclusion"
)
CUE_HANDOVER = (
    "Let me check this, or actually, that does not follow, is a real thought"
)
HIDDEN_REASONING_TARGETS = ("450 to 700", "450–700", "450-700")
# Current overlay freeze. Re-pin after a deliberate overlay edit.
EXPECTED_OVERLAY_SHAS = {
    "gen_drafting_v1": "609834efa759",
    "gen_drafting_v2": "ea66b7bba577",
    "gen_irac_analysis_v1": "088c0442f674",
    "gen_irac_analysis_v2": "5fa4ce5dba19",
    "gen_irac_analysis_v3": "d3635ee18266",
    "gen_irac_analysis_v4": "5bc40d3c1bef",
    "gen_statute_qa_v1": "bf49860e80dc",
    "gen_statute_qa_v2": "aaaaec660f01",
    "gen_statute_qa_v3": "9d2859618af8",
    "gen_statute_qa_v4": "598deaeafd23",
    "gen_summarization_v1": "42ee72ab542c",
    "gen_summarization_v2": "84e8a00c5425",
    "gen_transition_v1": "717e0c99aea7",
    "gen_transition_v2": "73a97936afa1",
}
EXPECTED_OVERLAY_JUDGE_SHAS = {
    "judge_pointwise_v1": "e2798dd5c81c",
    "judge_tiebreak_v1": "85a0c7f8da47",
}


def _overlay_ids(prefix: str) -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in OVERLAY_DIR.glob(f"{prefix}*.md")))


def _overlay_text(prompt_id: str) -> str:
    from tuned.data import prompt_registry as reg

    try:
        reg.set_overlay(OVERLAY_DIR)
        template = reg.load(prompt_id)
        return f"{template.system or ''}\n{template.user}"
    finally:
        reg.set_overlay(None)


def test_live_prompt_bytes_and_shas_are_unchanged():
    import hashlib

    from tuned.data import prompt_registry as reg
    from test_build_prompts import EXPECTED_SHAS

    assert set(reg.all_ids()) == set(EXPECTED_SHAS)
    for prompt_id, sha in EXPECTED_SHAS.items():
        raw = (reg.PROMPTS_DIR / f"{prompt_id}.md").read_bytes()
        assert hashlib.sha256(raw).hexdigest()[:12] == sha
        assert reg.load(prompt_id).sha == sha


def test_overlay_drops_cue_enumeration_and_hidden_reasoning_word_targets():
    for prompt_id in _overlay_ids("gen_"):
        text = _overlay_text(prompt_id)
        assert CUE_HANDOVER not in text, prompt_id
        assert "Ask am I sure" not in text, prompt_id
        lowered = text.lower()
        for target in HIDDEN_REASONING_TARGETS:
            assert target not in lowered, (prompt_id, target)


def test_overlay_drafting_and_summarization_are_legal_products_not_irac():
    for prompt_id in _overlay_ids("gen_drafting_"):
        text = _overlay_text(prompt_id)
        assert IRAC_ANSWER_CONTRACT not in text, prompt_id
        assert "{document_kind}" in text
        assert "{party_context}" in text
        lowered = text.lower()
        assert "relief" in lowered
        assert "never opens a line" in lowered
        assert "in your own words" in lowered

    for prompt_id in _overlay_ids("gen_summarization_"):
        text = _overlay_text(prompt_id)
        assert IRAC_ANSWER_CONTRACT not in text, prompt_id
        lowered = text.lower()
        assert "posture" in lowered
        assert "holding" in lowered or "ratio" in lowered
        assert "order" in lowered
        assert "never opens a line" in lowered
        assert "in your own words" in lowered


def test_overlay_analysis_statute_and_transition_keep_their_answer_contracts():
    for prompt_id in (
        *_overlay_ids("gen_irac_analysis_"),
        *_overlay_ids("gen_statute_qa_"),
        *_overlay_ids("gen_transition_"),
    ):
        text = _overlay_text(prompt_id)
        assert IRAC_ANSWER_CONTRACT in text, prompt_id
        assert "never inside your reasoning" in text
        assert "in your own words" in text
        assert (
            "You must not rely on any statutory provision, case name, or authority "
            "that does not appear in the materials above."
        ) in text
    for prompt_id in _overlay_ids("gen_statute_qa_"):
        assert "{section_text}" in _overlay_text(prompt_id)
    for prompt_id in _overlay_ids("gen_transition_"):
        text = _overlay_text(prompt_id)
        assert "enactment" in text.lower()
        assert "named expressly" in text


def test_overlay_never_enumerates_banned_meta_phrases():
    from tuned.data.gates import BANNED_META

    for prompt_id in _overlay_ids("gen_"):
        lowered = _overlay_text(prompt_id).lower()
        present = [phrase for phrase in BANNED_META if phrase in lowered]
        assert not present, (prompt_id, present)


def test_overlay_shas_are_pinned_independently_of_live():
    import hashlib

    from tuned.data import prompt_registry as reg
    from test_build_prompts import EXPECTED_SHAS

    overlay_ids = set(_overlay_ids("gen_"))
    assert overlay_ids == set(EXPECTED_OVERLAY_SHAS)
    try:
        reg.set_overlay(OVERLAY_DIR)
        for prompt_id, sha in EXPECTED_OVERLAY_SHAS.items():
            raw = (OVERLAY_DIR / f"{prompt_id}.md").read_bytes()
            assert hashlib.sha256(raw).hexdigest()[:12] == sha
            loaded = reg.load(prompt_id)
            assert loaded.sha == sha
            assert sha != EXPECTED_SHAS[prompt_id]
    finally:
        reg.set_overlay(None)


def test_recovery_config_has_matched_think_min_and_disables_s1():
    from tuned.data.config import load_build_config

    root = Path(__file__).parent.parent
    cfg = load_build_config(
        root / "configs" / "data_law_v1_exp_recovery.yaml", allow_unpinned=True
    )
    live = load_build_config(root / "configs" / "data_law_v1.yaml", allow_unpinned=True)
    assert cfg.build.harmony_s1_continue is False
    assert cfg.build.length_band.think_min == 500
    assert cfg.build.length_band.think_min == live.build.length_band.think_min
    assert cfg.routing.generator == ("cerebras/gpt-oss-120b",)
    assert cfg.build.harmony_completions is True
    assert cfg.build.prompt_overlay == "src/tuned/data/prompts_harmony"


COPYABLE_SCORE_OBJECT = (
    r'\{\s*"grounding"\s*:\s*[1-5]\s*,\s*"validity"\s*:\s*[1-5]\s*,'
    r'\s*"coverage"\s*:\s*[1-5]'
)


def test_recovery_judge_overlay_has_no_copyable_score_exemplar():
    import re

    from test_build_prompts import EXPECTED_SHAS, JUDGE_LEAKS

    for prompt_id in ("judge_pointwise_v1", "judge_tiebreak_v1"):
        path = OVERLAY_DIR / f"{prompt_id}.md"
        assert path.is_file(), path
        raw = path.read_text(encoding="utf-8")
        assert re.search(COPYABLE_SCORE_OBJECT, raw) is None, prompt_id
        assert "4/2/3" not in raw
        assert '"grounding": 4' not in raw
        assert '"validity": 2' not in raw
        assert '"coverage": 3' not in raw
        lowered = raw.lower()
        for leak in JUDGE_LEAKS:
            assert leak not in lowered, (prompt_id, leak)
        assert "grounding" in lowered and "validity" in lowered and "coverage" in lowered
        assert "json" in lowered
        assert "integer" in lowered
        assert "{source}" in raw and "{candidate_think}" in raw and "{candidate_answer}" in raw
        assert "grounding_faithfulness" in raw
        assert "reasoning_validity" in raw
        assert "issue_coverage" in raw

    from tuned.data import prompt_registry as reg

    try:
        reg.set_overlay(OVERLAY_DIR)
        for prompt_id, sha in EXPECTED_OVERLAY_JUDGE_SHAS.items():
            loaded = reg.load(prompt_id)
            assert loaded.sha == sha
            assert sha != EXPECTED_SHAS[prompt_id]
            live_bytes = (reg.PROMPTS_DIR / f"{prompt_id}.md").read_bytes()
            overlay_bytes = (OVERLAY_DIR / f"{prompt_id}.md").read_bytes()
            assert live_bytes != overlay_bytes
    finally:
        reg.set_overlay(None)
    from tuned.data import prompt_registry as reg2
    assert reg2.load("judge_pointwise_v1").sha == EXPECTED_SHAS["judge_pointwise_v1"]
    assert reg2.load("judge_tiebreak_v1").sha == EXPECTED_SHAS["judge_tiebreak_v1"]


def test_recovery_s1_flag_does_not_force_a_second_call(tmp_path):
    from dataclasses import replace

    from pipeline_fakes import FakeRouter, build_cfg, chat_response, open_store, paths_for
    from tuned.data.generate import generate_once
    from tuned.data.tasks import plan_wave

    cfg = replace(
        build_cfg(),
        build=replace(
            build_cfg().build,
            harmony_completions=True,
            harmony_prefill=DEFAULT_PREFILL,
            harmony_s1_continue=False,
        ),
    )
    first = (
        "whether the complaint discloses an offence under s.302 on these papers.\n"
        "<|end|><|start|>assistant<|channel|>final<|message|>"
        "Issue\nWhether s.302 applies.\n\nConclusion\nIt does not."
    )
    router = FakeRouter(
        cfg, script={"generator": [chat_response(text=first, reasoning=None)]}
    )
    store = open_store(tmp_path, n_seeds=1)
    plan_wave(store, cfg, "synthesis", 1, task_type_mix={"irac_analysis": 1.0})
    task = store.claim_tasks("w", 1)[0]
    result = asyncio.run(
        generate_once(store, cfg, router, task, paths=paths_for(tmp_path), attempt=1)
    )
    assert result.ok
    assert len(router.calls_for("generator")) == 1

