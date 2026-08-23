"""Harmony prompt render + completion parse for gpt-oss Completions prefill.

Chat Completions on Cerebras wraps Harmony for us and ignored
``assistant.reasoning`` as a think-channel prefill. Completions takes a raw
prompt string. For gpt-oss that string has to be Harmony, ending mid
analysis-message, so token-1 of the completion is a continuation of the
trace rather than a fresh instruction echo.

Render and parse go through ``openai-harmony``
(https://github.com/openai/harmony): string-concat Harmony drifted from the
tokenizer's special tokens, and Completions hosts strip markers
(``assistantfinal``). Restore those to the library's tokens, then parse.

s1 continue (https://github.com/simplescaling/s1) appends `` Wait`` — not
``Wait,`` / ``am I sure``, which are ``self_verification`` cues — when the
first analysis continuation has none.

This module does not call the network. providers.ChatClient posts the
rendered prompt to ``/v1/completions``; generate.py stores the parsed think
and final.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

DEFAULT_PREFILL = "I start from the facts. "
HARMONY_STOP = ("<|return|>", "<|call|>")
KNOWLEDGE_CUTOFF = "2024-06"
# simplescaling/s1 budget forcing. Leading space continues the analysis
# sentence. Bare "Wait" is not a VERIFICATION_CUES hit ("wait," / "wait:" are).
S1_WAIT = " Wait"

_ANALYSIS_HEADER = "<|start|>assistant<|channel|>analysis<|message|>"
_FINAL_MARKERS = (
    "<|end|><|start|>assistant<|channel|>final<|message|>",
    "<|start|>assistant<|channel|>final<|message|>",
    "<|end|><|start|>assistant<|channel|>commentary<|message|>",
    "assistantfinal",
)
_LEADING_ANALYSIS_HEADERS = (
    "<|channel|>analysis<|message|>",
    "<|start|>assistant<|channel|>analysis<|message|>",
    "analysis<|message|>",
)
_STRIPPED_MARKERS = (
    ("assistantfinal", "<|end|><|start|>assistant<|channel|>final<|message|>"),
    ("assistantcommentary", "<|end|><|start|>assistant<|channel|>commentary<|message|>"),
    ("assistantanalysis", "<|end|><|start|>assistant<|channel|>analysis<|message|>"),
)
_EFFORT = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}


@dataclass(frozen=True)
class HarmonyParse:
    analysis_continuation: str
    think: str
    final: str
    continued: bool
    restated: bool


def restated_opening(text: str) -> bool:
    """True when the first generated analysis tokens echo the instruction packet."""
    from tuned.data.gates import restated_opening as _gate_restated

    return _gate_restated(text)


@lru_cache(maxsize=1)
def _encoding():
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def render_for_analysis_prefill(
    messages: Sequence[Mapping],
    prefill: str,
    *,
    reasoning_effort: str = "medium",
    current_date: str,
    knowledge_cutoff: str = KNOWLEDGE_CUTOFF,
) -> str:
    """Render chat messages as Harmony, ending inside the analysis message.

    Pipeline ``system`` turns become Harmony *developer* instructions. The
    Harmony *system* turn is openai-harmony's SystemContent (identity / date /
    effort). The prompt MUST end with ``prefill`` and no ``<|end|>``, so the
    Completions continuation is still the analysis channel.
    """
    from openai_harmony import (
        Conversation,
        DeveloperContent,
        Message,
        ReasoningEffort,
        Role,
        SystemContent,
    )

    developer_parts: list[str] = []
    user_parts: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if role == "system":
            developer_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        elif role == "developer":
            developer_parts.append(content)

    effort_name = _EFFORT.get((reasoning_effort or "medium").lower(), "MEDIUM")
    system = Message.from_role_and_content(
        Role.SYSTEM,
        SystemContent.new()
        .with_reasoning_effort(getattr(ReasoningEffort, effort_name))
        .with_conversation_start_date(current_date)
        .with_knowledge_cutoff(knowledge_cutoff),
    )
    turns = [system]
    developer = "\n\n".join(part for part in developer_parts if part)
    if developer:
        turns.append(
            Message.from_role_and_content(
                Role.DEVELOPER,
                DeveloperContent.new().with_instructions(developer),
            )
        )
    user = "\n\n".join(part for part in user_parts if part)
    turns.append(Message.from_role_and_content(Role.USER, user))
    turns.append(
        Message.from_role_and_content(Role.ASSISTANT, prefill).with_channel("analysis")
    )
    tokens = _encoding().render_conversation(Conversation.from_messages(turns))
    # render() closes the analysis message with <|end|>. Drop it so token-1
    # continues the prefill instead of starting a new channel.
    if tokens:
        tokens = tokens[:-1]
    return _encoding().decode_utf8(tokens)


def parse_completion(generated: str, prefill: str) -> HarmonyParse:
    """Split a Completions continuation into analysis vs final.

    ``generated`` is echo=false: it does not include ``prefill``. Reconstruct
    the full think as ``prefill + continuation`` only when the continuation
    actually continues the sentence. Prepending on a restatement would put
    dialect cues into a trace the model did not write, which is how
    ``self_verification`` would get faked.
    """
    analysis, final = _split_channels(generated)
    continuation = analysis
    echoed = bool(prefill) and continuation.startswith(prefill)
    if echoed:
        continuation_after = continuation[len(prefill) :]
        restated = restated_opening(continuation_after)
        continued = bool(continuation_after.strip()) and not restated
        think = continuation if continued else continuation_after
        return HarmonyParse(
            analysis_continuation=continuation_after,
            think=think.strip(),
            final=final,
            continued=continued,
            restated=restated,
        )

    restated = restated_opening(continuation)
    continued = bool(continuation.strip()) and not restated
    think = (prefill + continuation) if continued else continuation
    return HarmonyParse(
        analysis_continuation=continuation,
        think=think.strip(),
        final=final,
        continued=continued,
        restated=restated,
    )


def continuation_has_cue(text: str) -> bool:
    """Whether the analysis continuation already contains a verification cue."""
    from tuned.data.gates import VERIFICATION_CUES

    lowered = (text or "").lower()
    return any(cue.lower() in lowered for cue in VERIFICATION_CUES)


def needs_s1_continue(parsed: HarmonyParse) -> bool:
    """Second Completions call only when token-1 already continued without a cue."""
    return parsed.continued and not parsed.restated and not continuation_has_cue(
        parsed.analysis_continuation
    )


def stitch_s1(first: HarmonyParse, second: HarmonyParse, prefill: str) -> HarmonyParse:
    """Join a Wait-forced second analysis onto the first continuation."""
    continuation = first.analysis_continuation + S1_WAIT + second.analysis_continuation
    restated = restated_opening(continuation)
    continued = bool(continuation.strip()) and not restated
    think = (prefill + continuation) if continued else continuation
    return HarmonyParse(
        analysis_continuation=continuation,
        think=think.strip(),
        final=(second.final or first.final),
        continued=continued,
        restated=restated,
    )


def _restore_stripped_tokens(text: str) -> str:
    """Hosts often decode Harmony specials away (``assistantfinal``)."""
    restored = text
    for stripped, full in _STRIPPED_MARKERS:
        restored = restored.replace(stripped, full)
    return restored


def _split_channels(generated: str) -> tuple[str, str]:
    from openai_harmony import Role

    restored = _restore_stripped_tokens(generated or "")
    for stop in HARMONY_STOP:
        if stop in restored:
            restored = restored.split(stop, 1)[0]
    restored = _strip_leading_analysis_header(restored)
    payload = "<|channel|>analysis<|message|>" + restored
    enc = _encoding()
    try:
        tokens = enc.encode(payload, allowed_special="all")
    except TypeError:
        tokens = enc.encode(payload)
    stop_ids = set(enc.stop_tokens_for_assistant_actions())
    tokens = [token for token in tokens if token not in stop_ids]
    try:
        messages = enc.parse_messages_from_completion_tokens(
            tokens, Role.ASSISTANT, strict=False
        )
    except Exception:
        return _split_channels_fallback(generated)
    analysis_parts: list[str] = []
    final_parts: list[str] = []
    for message in messages:
        text = "".join(getattr(part, "text", "") or "" for part in message.content)
        if message.channel == "final":
            final_parts.append(text)
        else:
            analysis_parts.append(text)
    if not analysis_parts and not final_parts:
        return _split_channels_fallback(generated)
    return "".join(analysis_parts), "".join(final_parts).strip()


def _split_channels_fallback(generated: str) -> tuple[str, str]:
    text = generated or ""
    for stop in HARMONY_STOP:
        if stop in text:
            text = text.split(stop, 1)[0]
    analysis, final = text, ""
    for marker in _FINAL_MARKERS:
        if marker in text:
            analysis, _, rest = text.partition(marker)
            final = rest
            break
    return _strip_leading_analysis_header(analysis), final.strip()


def _strip_leading_analysis_header(text: str) -> str:
    stripped = text.lstrip("\n")
    for header in _LEADING_ANALYSIS_HEADERS:
        if stripped.startswith(header):
            return stripped[len(header) :]
    return stripped


_SMOKE_SYSTEM = (
    "You are a judge of an Indian court. You think from the facts to the "
    "questions they raise, then to the law, then to a result."
)
_SMOKE_USER = """The facts, the record and the law relied on are before you:

A stabbed B once in the chest in Delhi on 12 March 2023. B died that night. The prosecution relies on s.302 of the Indian Penal Code. The defence says the single injury was not intended to kill.

The point on which the matter turns is whether the injury was sufficient in the ordinary course of nature to cause death.

Work it out before you commit to anything. Reason in the first person and in the present tense.{word_count}

When the thinking is done, write the judgment's analysis under four headings, each on its own line — Issue, Rule, Application, Conclusion.
"""
_SMOKE_WORD_COUNT = (
    " Roughly 450 to 700 words of deliberation is normal for a matter of any substance."
)
_GROUNDED_PREFILL = (
    "I start from the facts. A stabbed B once in the chest in Delhi. Let me check whether "
)


def main(argv: Sequence[str] | None = None) -> int:
    """One paid Cerebras Completions call. Scores token-1 of the analysis continuation."""
    import argparse
    import asyncio
    import json
    from datetime import date
    from pathlib import Path

    from tuned.data.config import ModelRef, load_build_config
    from tuned.data.providers import ChatClient, ChatRequest, ProviderError, load_dotenv_keys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_law_v1.yaml")
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--prefill", default=DEFAULT_PREFILL)
    parser.add_argument(
        "--no-word-count",
        action="store_true",
        help="omit the 450-700 word packet from the user turn",
    )
    parser.add_argument(
        "--grounded-prefill",
        action="store_true",
        help="prefill with the case facts already in the analysis channel",
    )
    parser.add_argument("--effort", default="medium", choices=("low", "medium", "high"))
    args = parser.parse_args(argv)

    load_dotenv_keys()
    cfg = load_build_config(args.config, allow_unpinned=True)
    ref = ModelRef("cerebras", "gpt-oss-120b")
    provider, model = cfg.model_for(ref)
    prefill = _GROUNDED_PREFILL if args.grounded_prefill else args.prefill
    user = _SMOKE_USER.format(word_count="" if args.no_word_count else _SMOKE_WORD_COUNT)
    messages = (
        {"role": "system", "content": _SMOKE_SYSTEM},
        {"role": "user", "content": user},
    )
    prompt = render_for_analysis_prefill(
        messages,
        prefill,
        reasoning_effort=args.effort,
        current_date=date.today().isoformat(),
    )
    req = ChatRequest(
        messages=messages,
        ref=ref,
        prompt=prompt,
        max_tokens=args.max_tokens,
        role="generator",
    )

    async def run():
        client = ChatClient(provider, model, max_retries=2)
        try:
            return await client.complete(req)
        finally:
            await client.aclose()

    print(f"harmony smoke: cerebras/gpt-oss-120b completions max_tokens={args.max_tokens}")
    print(f"prefill: {prefill!r}")
    print(f"word_count_packet={not args.no_word_count} effort={args.effort}")
    print(f"prompt_chars={len(prompt)}")
    try:
        response = asyncio.run(run())
    except ProviderError as exc:
        print(f"FAIL status={exc.status} {exc}")
        return 1

    parsed = parse_completion(response.text, prefill)
    opening = " ".join(parsed.analysis_continuation.split())[:180]
    print(f"status={response.status} finish={response.finish_reason}")
    print(
        f"usage prompt={response.prompt_tokens} completion={response.completion_tokens} "
        f"latency_ms={response.latency_ms}"
    )
    print(f"continued={parsed.continued} restated={parsed.restated}")
    print(f"analysis_opening: {opening!r}")
    print(f"final_chars={len(parsed.final)} think_chars={len(parsed.think)}")
    from tuned.data.gates import BANNED_META, VERIFICATION_CUES

    think_l = parsed.think.lower()
    cont_l = parsed.analysis_continuation.lower()
    cues = [c for c in VERIFICATION_CUES if c.lower() in think_l]
    cues_continuation = [c for c in VERIFICATION_CUES if c.lower() in cont_l]
    meta = [p for p in BANNED_META if p in think_l]
    print(f"verification_cues_in_think={cues[:8]}")
    print(f"verification_cues_in_continuation={cues_continuation[:8]}")
    print(f"banned_meta={meta}")
    print(f"we_need_to_produce={'we need to produce' in think_l}")
    headings = [
        name
        for name in ("Issue", "Rule", "Application", "Conclusion")
        if any(line.strip().lower().startswith(name.lower()) for line in parsed.final.splitlines())
    ]
    print(f"final_irac_headings={headings}")
    dump_dir = Path("data/build/exp_harmony")
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / "last_smoke.json"
    dump_path.write_text(
        json.dumps(
            {
                "continued": parsed.continued,
                "restated": parsed.restated,
                "prefill": prefill,
                "word_count_packet": not args.no_word_count,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "think": parsed.think,
                "final": parsed.final,
                "raw": response.text,
                "cues": cues,
                "banned_meta": meta,
                "final_irac_headings": headings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"dump={dump_path}")
    raw = " ".join((response.text or "").split())[:400]
    print(f"raw_continuation: {raw!r}")
    if parsed.continued:
        print("WIN: token-1 continued the analysis prefill")
        return 0
    print("LOSS: token-1 did not continue the analysis prefill")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

