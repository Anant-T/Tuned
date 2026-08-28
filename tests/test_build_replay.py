import ast
import json
from pathlib import Path

import pytest

from tuned.data.config import load_build_config
from tuned.data.replay import (
    DEFAULT_COUNTS,
    SLICE_ORDER,
    answer_length_ok,
    build_replay,
    empty_think,
    has_emoji,
    has_markup,
    is_advisory_question,
    is_mostly_ascii,
    is_refusal,
    is_single_turn,
    legal_qa_row,
    nemotron_row,
    ot_row,
    parse_counts,
    sha256_hex,
    smoltalk_row,
    wildchat_row,
)

REPLAY_SRC = Path(__file__).parent.parent / "src" / "tuned" / "data" / "replay.py"


# --------------------------------------------------------------------------
# Real-config precedent: empty-think block must be byte-exact.
# --------------------------------------------------------------------------

def _real_think_tags():
    cfg = load_build_config("data/configs/data_law_v1.yaml", allow_unpinned=True)
    return cfg.think_open, cfg.think_close


def test_empty_think_byte_exact_against_real_config():
    think_open, think_close = _real_think_tags()
    assert think_open == "<think>"
    assert think_close == "</think>"
    assert empty_think(think_open, think_close) == "<think>\n\n</think>"


def test_empty_think_generic_shape():
    assert empty_think("[T]", "[/T]") == "[T]\n\n[/T]"


# --------------------------------------------------------------------------
# Pure quality-filter primitives.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "I can't help with that.",
    "I cannot do that.",
    "I'm sorry, but I won't.",
    "I am sorry, I won't do that.",
    "As an AI, I don't have opinions.",
    "I'm unable to assist with this request.",
    "That goes against my guidelines.",
    "I CANNOT do that.",  # case-insensitive
])
def test_is_refusal_true(text):
    assert is_refusal(text) is True


@pytest.mark.parametrize("text", [
    "Section 1 of BNS 2023 defines the scope.",
    "Here is a draft memo on contract termination.",
    "",
])
def test_is_refusal_false(text):
    assert is_refusal(text) is False


def test_has_emoji_true_for_astral_and_bmp_ranges():
    assert has_emoji("great job \U0001F600") is True  # astral emoji block
    assert has_emoji("weather ☀ today") is True  # BMP dingbat block


def test_has_emoji_false_for_plain_text():
    assert has_emoji("Section 1 of BNS 2023 establishes jurisdiction.") is False


def test_is_single_turn_true():
    assert is_single_turn([
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]) is True


def test_is_single_turn_ignores_system():
    assert is_single_turn([
        {"role": "system", "content": ""},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]) is True


def test_is_single_turn_false_for_multi_turn():
    assert is_single_turn([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]) is False


def test_is_single_turn_false_for_missing_role():
    assert is_single_turn([{"role": "user", "content": "q"}]) is False


def test_has_markup():
    assert has_markup("hello <|im_start|>") is True
    assert has_markup("hello", "world <|end|>") is True
    assert has_markup("clean text", "also clean") is False


def test_is_advisory_question():
    assert is_advisory_question("What is the penalty under Section 5?") is True
    assert is_advisory_question("How should I draft a termination letter?") is True
    assert is_advisory_question("Write a professional email declining an offer.") is True
    assert is_advisory_question("lol nice") is False
    assert is_advisory_question("") is False


def test_is_mostly_ascii():
    assert is_mostly_ascii("plain english text") is True
    assert is_mostly_ascii("Bharatiya Nyaya Sanhita संहिता") is False
    assert is_mostly_ascii("") is True


def test_answer_length_ok():
    assert answer_length_ok("x" * 200) is True
    assert answer_length_ok("x" * 1600) is True
    assert answer_length_ok("x" * 199) is False
    assert answer_length_ok("x" * 1601) is False


def test_sha256_hex_deterministic_and_distinct():
    assert sha256_hex("hello") == sha256_hex("hello")
    assert sha256_hex("hello") != sha256_hex("world")


# --------------------------------------------------------------------------
# ot_row
# --------------------------------------------------------------------------

def _ot_raw(user="What is 2+2?", reasoning="Two plus two is four.", solution="4"):
    return {
        "conversations": [
            {"from": "user", "value": user},
            {"from": "assistant", "value": f"<|begin_of_thought|>{reasoning}<|end_of_thought|>\n<|begin_of_solution|>{solution}<|end_of_solution|>"},
        ]
    }


def test_ot_row_accept():
    row, reason = ot_row(_ot_raw(), "<think>", "</think>")
    assert reason is None
    assert row["messages"][0] == {"role": "user", "content": "What is 2+2?"}
    assert row["messages"][1]["content"] == "<think>Two plus two is four.</think>4"
    assert row["_prov"] == {
        "source": "open-thoughts/OpenThoughts-114k",
        "license": "Apache-2.0",
        "native_id": None,
        "reasoning": True,
    }
    assert "<|" not in row["messages"][1]["content"]


def test_ot_row_reject_multi_turn():
    raw = _ot_raw()
    raw["conversations"].extend([
        {"from": "user", "value": "follow up"},
        {"from": "assistant", "value": "<|begin_of_thought|>x<|end_of_thought|>\n<|begin_of_solution|>y<|end_of_solution|>"},
    ])
    row, reason = ot_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "not_single_turn"


def test_ot_row_reject_no_end_of_thought():
    raw = {"conversations": [
        {"from": "user", "value": "q"},
        {"from": "assistant", "value": "just an answer, no markers"},
    ]}
    row, reason = ot_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "no_end_of_thought"


def test_ot_row_reject_empty_after_strip():
    raw = _ot_raw(reasoning="", solution="")
    row, reason = ot_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "empty"


def test_ot_row_reject_too_long():
    raw = _ot_raw(reasoning="x" * 20000, solution="y" * 5000)
    row, reason = ot_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "too_long"


def test_ot_row_reject_refusal():
    raw = _ot_raw(solution="I'm sorry, I cannot help with that.")
    row, reason = ot_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "refusal"


def test_ot_row_reject_emoji():
    raw = _ot_raw(solution="The answer is 4 \U0001F600")
    row, reason = ot_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "emoji"


def test_ot_row_reject_markup():
    raw = _ot_raw(solution="4<|im_start|>")
    row, reason = ot_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "markup"


def test_ot_row_reject_non_ascii():
    raw = _ot_raw(solution="中文答案" * 10)
    row, reason = ot_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "non_ascii"


# --------------------------------------------------------------------------
# nemotron_row
# --------------------------------------------------------------------------

def _nemotron_raw(user="What is the capital of India?", trace="Let me think about this.",
                   answer="The capital of India is New Delhi.", license_="CC BY 4.0", reasoning="on"):
    return {
        "uuid": "uuid-123",
        "license": license_,
        "generator": "some-model",
        "version": "v2",
        "category": "chat",
        "reasoning": reasoning,
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": user},
            {"role": "assistant", "content": f"<think>\n{trace}\n</think>\n{answer}"},
        ],
    }


def test_nemotron_row_accept():
    row, reason = nemotron_row(_nemotron_raw(), "<think>", "</think>")
    assert reason is None
    assert row["messages"][1]["content"] == "<think>Let me think about this.</think>The capital of India is New Delhi."
    assert row["_prov"] == {
        "source": "nvidia/Nemotron-Post-Training-Dataset-v2",
        "license": "CC-BY-4.0",
        "native_id": "uuid-123",
        "reasoning": True,
    }


def test_nemotron_row_license_normalization_accepts_hyphen_variant():
    row, reason = nemotron_row(_nemotron_raw(license_="cc-by-4.0"), "<think>", "</think>")
    assert reason is None


def test_nemotron_row_reject_license():
    row, reason = nemotron_row(_nemotron_raw(license_="CC BY-SA 4.0"), "<think>", "</think>")
    assert row is None
    assert reason == "license"


def test_nemotron_row_reject_license_key_absent():
    raw = _nemotron_raw()
    del raw["license"]
    row, reason = nemotron_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "license"


def test_nemotron_row_reject_reasoning_key_absent():
    raw = _nemotron_raw()
    del raw["reasoning"]
    row, reason = nemotron_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "not_reasoning_on"


def test_nemotron_row_reject_non_string_license_does_not_raise():
    raw = _nemotron_raw()
    raw["license"] = True  # non-string truthy value - pins the str(...) hardening
    row, reason = nemotron_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "license"


def test_nemotron_row_reject_non_string_reasoning_does_not_raise():
    raw = _nemotron_raw()
    raw["reasoning"] = True  # non-string truthy value - pins the str(...) hardening
    row, reason = nemotron_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "not_reasoning_on"


def test_nemotron_row_reject_license_odc_by():
    row, reason = nemotron_row(_nemotron_raw(license_="ODC-BY"), "<think>", "</think>")
    assert row is None
    assert reason == "license"


def test_nemotron_row_reject_not_reasoning_on():
    row, reason = nemotron_row(_nemotron_raw(reasoning="off"), "<think>", "</think>")
    assert row is None
    assert reason == "not_reasoning_on"


def test_nemotron_row_reject_multi_turn():
    raw = _nemotron_raw()
    raw["messages"].extend([
        {"role": "user", "content": "follow up"},
        {"role": "assistant", "content": "<think>x</think>y"},
    ])
    row, reason = nemotron_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "not_single_turn"


def test_nemotron_row_reject_no_think_markers():
    raw = _nemotron_raw()
    raw["messages"][-1]["content"] = "just a direct answer, no think block"
    row, reason = nemotron_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "no_think_markers"


def test_nemotron_row_reject_empty_trace():
    raw = _nemotron_raw()
    raw["messages"][-1]["content"] = "<think>\n\n</think>\nanswer text here"
    row, reason = nemotron_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "empty"


def test_nemotron_row_reject_refusal():
    row, reason = nemotron_row(_nemotron_raw(answer="I'm sorry, I cannot help with that."), "<think>", "</think>")
    assert row is None
    assert reason == "refusal"


def test_nemotron_row_reject_emoji():
    row, reason = nemotron_row(_nemotron_raw(answer="New Delhi \U0001F600"), "<think>", "</think>")
    assert row is None
    assert reason == "emoji"


def test_nemotron_row_reject_markup():
    row, reason = nemotron_row(_nemotron_raw(answer="New Delhi <|im_start|>"), "<think>", "</think>")
    assert row is None
    assert reason == "markup"


def test_nemotron_row_reject_non_ascii():
    row, reason = nemotron_row(_nemotron_raw(answer="中文答案" * 10), "<think>", "</think>")
    assert row is None
    assert reason == "non_ascii"


# --------------------------------------------------------------------------
# smoltalk_row
# --------------------------------------------------------------------------

def _smoltalk_raw(user="How can you help me revise an essay?", answer=None, source="smoltalk-smollm3_smol-magpie-ultra"):
    if answer is None:
        answer = "I can help review structure, clarity, and flow. " * 5
        answer = answer[:400].strip()
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "source": source,
    }


def test_smoltalk_row_accept():
    row, reason = smoltalk_row(_smoltalk_raw(), "<think>", "</think>")
    assert reason is None
    assert row["messages"][1]["content"].startswith("<think>\n\n</think>")
    assert row["_prov"]["reasoning"] is False
    assert row["_prov"]["source"] == "HuggingFaceTB/smoltalk2:smoltalk-smollm3_smol-magpie-ultra"


def test_smoltalk_row_reject_multi_turn():
    raw = _smoltalk_raw()
    raw["messages"].extend([
        {"role": "user", "content": "another question"},
        {"role": "assistant", "content": "another answer " * 30},
    ])
    row, reason = smoltalk_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "not_single_turn"


def test_smoltalk_row_reject_length_too_short():
    row, reason = smoltalk_row(_smoltalk_raw(answer="too short"), "<think>", "</think>")
    assert row is None
    assert reason == "length"


def test_smoltalk_row_reject_length_too_long():
    row, reason = smoltalk_row(_smoltalk_raw(answer="x" * 2000), "<think>", "</think>")
    assert row is None
    assert reason == "length"


def test_smoltalk_row_reject_refusal():
    row, reason = smoltalk_row(_smoltalk_raw(answer="I'm sorry, I cannot help with that." + " padding" * 25), "<think>", "</think>")
    assert row is None
    assert reason == "refusal"


def test_smoltalk_row_reject_emoji():
    row, reason = smoltalk_row(_smoltalk_raw(answer="Sure, happy to help! \U0001F600" + " padding" * 25), "<think>", "</think>")
    assert row is None
    assert reason == "emoji"


def test_smoltalk_row_reject_markup():
    row, reason = smoltalk_row(_smoltalk_raw(answer="Sure, here you go <|im_start|>" + " padding" * 25), "<think>", "</think>")
    assert row is None
    assert reason == "markup"


def test_smoltalk_row_reject_non_ascii():
    row, reason = smoltalk_row(_smoltalk_raw(answer="中文答案" * 60), "<think>", "</think>")
    assert row is None
    assert reason == "non_ascii"


# --------------------------------------------------------------------------
# legal_qa_row
# --------------------------------------------------------------------------

def _legal_qa_raw(question="Who is liable under Section 1 of BNS 2023?", answer=None, chunk_id="BNS_1"):
    if answer is None:
        answer = ("Any person liable under Indian law to be tried for an offence "
                   "committed beyond India will be dealt with as if the act was "
                   "committed within India, per Section 1 of BNS 2023. This extends "
                   "to citizens abroad and persons on Indian-registered ships too.")
    return {"chunk_id": chunk_id, "act": "BNS 2023", "section_number": "1",
            "question": question, "answer": answer, "question_type": "definitional_topic"}


def test_legal_qa_row_accept():
    row, reason = legal_qa_row(_legal_qa_raw(), "<think>", "</think>")
    assert reason is None
    assert row["messages"][1]["content"].startswith("<think>\n\n</think>")
    assert row["_prov"]["native_id"] == "BNS_1"
    assert row["_prov"]["reasoning"] is False


def test_legal_qa_row_allows_devanagari_and_non_ascii():
    answer = ("भारतीय न्याय संहिता "
              "(Bharatiya Nyaya Sanhita) Section 1 applies across all of India and to " * 3)
    assert not is_mostly_ascii(answer)  # sanity: this really is non-ASCII heavy
    row, reason = legal_qa_row(_legal_qa_raw(answer=answer[:1500]), "<think>", "</think>")
    assert reason is None, f"unexpected reject reason: {reason}"


def test_legal_qa_row_reject_empty():
    row, reason = legal_qa_row(_legal_qa_raw(question=""), "<think>", "</think>")
    assert row is None
    assert reason == "empty"


def test_legal_qa_row_reject_length():
    row, reason = legal_qa_row(_legal_qa_raw(answer="Too short."), "<think>", "</think>")
    assert row is None
    assert reason == "length"


def test_legal_qa_row_reject_refusal():
    answer = "I'm sorry, I cannot help with that. " * 10
    row, reason = legal_qa_row(_legal_qa_raw(answer=answer[:1500]), "<think>", "</think>")
    assert row is None
    assert reason == "refusal"


def test_legal_qa_row_reject_emoji():
    answer = ("This section applies to the whole of India. " * 6) + "\U0001F600"
    row, reason = legal_qa_row(_legal_qa_raw(answer=answer), "<think>", "</think>")
    assert row is None
    assert reason == "emoji"


def test_legal_qa_row_reject_markup():
    answer = ("This section applies to the whole of India. " * 6) + "<|im_start|>"
    row, reason = legal_qa_row(_legal_qa_raw(answer=answer), "<think>", "</think>")
    assert row is None
    assert reason == "markup"


# --------------------------------------------------------------------------
# wildchat_row
# --------------------------------------------------------------------------

def _wildchat_raw(user="What should I include in a professional resignation letter?", answer=None,
                   language="English", turn=1, toxic=False, redacted=False):
    if answer is None:
        answer = ("A professional resignation letter should include your intended last "
                   "day, a brief thank-you for the opportunity, and an offer to help "
                   "with the transition. Keep the tone courteous and concise throughout.")
    return {
        "conversation_hash": "hash-abc",
        "language": language,
        "turn": turn,
        "toxic": toxic,
        "redacted": redacted,
        "conversation": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
    }


def test_wildchat_row_accept():
    row, reason = wildchat_row(_wildchat_raw(), "<think>", "</think>")
    assert reason is None
    assert row["messages"][1]["content"].startswith("<think>\n\n</think>")
    assert row["_prov"]["native_id"] == "hash-abc"
    assert row["_prov"]["reasoning"] is False


def test_wildchat_row_reject_language():
    row, reason = wildchat_row(_wildchat_raw(language="Spanish"), "<think>", "</think>")
    assert row is None
    assert reason == "language"


def test_wildchat_row_reject_flagged_toxic():
    row, reason = wildchat_row(_wildchat_raw(toxic=True), "<think>", "</think>")
    assert row is None
    assert reason == "flagged"


def test_wildchat_row_reject_flagged_redacted():
    row, reason = wildchat_row(_wildchat_raw(redacted=True), "<think>", "</think>")
    assert row is None
    assert reason == "flagged"


def test_wildchat_row_reject_multi_turn():
    raw = _wildchat_raw()
    raw["conversation"].extend([
        {"role": "user", "content": "one more thing"},
        {"role": "assistant", "content": "sure, here it is"},
    ])
    row, reason = wildchat_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "not_single_turn"


def test_wildchat_row_reject_not_advisory():
    row, reason = wildchat_row(_wildchat_raw(user="lol nice one"), "<think>", "</think>")
    assert row is None
    assert reason == "not_advisory"


def test_wildchat_row_reject_length():
    row, reason = wildchat_row(_wildchat_raw(answer="Too short."), "<think>", "</think>")
    assert row is None
    assert reason == "length"


def test_wildchat_row_reject_refusal():
    answer = "I'm sorry, I cannot help with that request. " * 5
    row, reason = wildchat_row(_wildchat_raw(answer=answer), "<think>", "</think>")
    assert row is None
    assert reason == "refusal"


def test_wildchat_row_reject_emoji():
    answer = ("A professional resignation letter should be courteous and concise. " * 4) + "\U0001F600"
    row, reason = wildchat_row(_wildchat_raw(answer=answer), "<think>", "</think>")
    assert row is None
    assert reason == "emoji"


def test_wildchat_row_reject_markup():
    answer = ("A professional resignation letter should be courteous and concise. " * 4) + "<|im_start|>"
    row, reason = wildchat_row(_wildchat_raw(answer=answer), "<think>", "</think>")
    assert row is None
    assert reason == "markup"


def test_wildchat_row_reject_non_ascii():
    row, reason = wildchat_row(_wildchat_raw(answer="中文答案" * 60), "<think>", "</think>")
    assert row is None
    assert reason == "non_ascii"


# --------------------------------------------------------------------------
# build_replay end-to-end.
# --------------------------------------------------------------------------

def _synthetic_sources(n_each=6):
    ot = [_ot_raw(user=f"ot q{i}", reasoning=f"reasoning {i}", solution=f"solution {i}") for i in range(n_each)]
    nemotron = [_nemotron_raw(user=f"nem q{i}", trace=f"trace {i}", answer=f"The answer to question {i} is here in full detail.") for i in range(n_each)]
    smoltalk = [_smoltalk_raw(user=f"smalltalk q{i}", answer=("A reasonably detailed helpful answer padded out. " * 5)[:400] + str(i)) for i in range(n_each)]
    legal = [_legal_qa_raw(question=f"legal q{i}", chunk_id=f"BNS_{i}") for i in range(n_each)]
    wildchat = [_wildchat_raw(user=f"What should I do about issue {i} at work?") for i in range(n_each)]
    return {
        "ot_reasoning": iter(ot),
        "nemotron_reasoning": iter(nemotron),
        "smoltalk_nothink": iter(smoltalk),
        "legal_qa_empty": iter(legal),
        "wildchat_prof": iter(wildchat),
    }


class _FakeBuildCfg:
    def __init__(self, workdir):
        self.workdir = workdir


class _FakeCfg:
    def __init__(self, workdir, think_open="<think>", think_close="</think>"):
        self.think_open = think_open
        self.think_close = think_close
        self.build = _FakeBuildCfg(workdir)


def test_build_replay_writes_expected_counts_and_prov(tmp_path):
    cfg = _FakeCfg(str(tmp_path / "workdir"))
    counts = (2, 2, 2, 2, 2)
    out_path = tmp_path / "replay.jsonl"
    stats = build_replay(cfg, counts, rows_by_source=_synthetic_sources(), out_path=out_path)

    assert stats["total"] == 10
    for name, n in zip(SLICE_ORDER, counts):
        assert stats[name]["accepted"] == n

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 10
    rows = [json.loads(line) for line in lines]

    reasoning_by_slice = {
        "open-thoughts/OpenThoughts-114k": True,
        "nvidia/Nemotron-Post-Training-Dataset-v2": True,
    }
    for row in rows:
        assert "_prov" in row
        prov = row["_prov"]
        assert set(prov.keys()) == {"source", "license", "native_id", "reasoning"}
        if prov["source"] in reasoning_by_slice:
            assert prov["reasoning"] is True
        else:
            assert prov["reasoning"] is False
        assert "<|" not in row["messages"][0]["content"]
        assert "<|" not in row["messages"][1]["content"]


def test_build_replay_raises_on_shortfall_names_source(tmp_path):
    cfg = _FakeCfg(str(tmp_path / "workdir"))
    sources = _synthetic_sources(n_each=1)  # legal_qa_empty will run out at 1 < 2
    counts = (1, 1, 1, 2, 1)
    with pytest.raises(RuntimeError, match="legal_qa_empty"):
        build_replay(cfg, counts, rows_by_source=sources, out_path=tmp_path / "replay.jsonl")


def test_build_replay_dedup_within_slice(tmp_path):
    cfg = _FakeCfg(str(tmp_path / "workdir"))
    dup_question = "the exact same legal question every time"
    legal = [_legal_qa_raw(question=dup_question, chunk_id=f"BNS_{i}") for i in range(5)]
    legal.append(_legal_qa_raw(question="a genuinely different question", chunk_id="BNS_99"))
    sources = {
        "ot_reasoning": iter([]),
        "nemotron_reasoning": iter([]),
        "smoltalk_nothink": iter([]),
        "legal_qa_empty": iter(legal),
        "wildchat_prof": iter([]),
    }
    counts = (0, 0, 0, 2, 0)
    stats = build_replay(cfg, counts, rows_by_source=sources, out_path=tmp_path / "replay.jsonl")
    assert stats["legal_qa_empty"]["accepted"] == 2
    assert stats["legal_qa_empty"]["rejects"].get("duplicate") == 4


def test_build_replay_skips_zero_count_slices_without_touching_source(tmp_path):
    cfg = _FakeCfg(str(tmp_path / "workdir"))
    # rows_by_source deliberately omits nemotron_reasoning/wildchat_prof keys -
    # a 0 count must never even look them up.
    sources = {
        "ot_reasoning": iter([_ot_raw()]),
        "smoltalk_nothink": iter([_smoltalk_raw()]),
        "legal_qa_empty": iter([_legal_qa_raw()]),
    }
    counts = (1, 0, 1, 1, 0)
    stats = build_replay(cfg, counts, rows_by_source=sources, out_path=tmp_path / "replay.jsonl")
    assert stats["nemotron_reasoning"]["accepted"] == 0
    assert stats["wildchat_prof"]["accepted"] == 0
    assert stats["total"] == 3


def test_build_replay_default_out_path_uses_build_paths(tmp_path):
    cfg = _FakeCfg(str(tmp_path / "workdir"))
    stats = build_replay(cfg, (1, 1, 1, 1, 1), rows_by_source=_synthetic_sources())
    expected = tmp_path / "workdir" / "streams" / "replay.jsonl"
    assert Path(stats["out_path"]) == expected
    assert expected.exists()


# --------------------------------------------------------------------------
# --counts parsing.
# --------------------------------------------------------------------------

def test_parse_counts_valid():
    assert parse_counts("2520,600,600,300,300") == DEFAULT_COUNTS


def test_parse_counts_wrong_length():
    with pytest.raises(ValueError, match="5 comma-separated"):
        parse_counts("1,2,3")


def test_parse_counts_non_integer():
    with pytest.raises(ValueError):
        parse_counts("a,b,c,d,e")


# --------------------------------------------------------------------------
# Module-import / CLI hygiene.
# --------------------------------------------------------------------------

def test_cli_hard_exits_after_success():
    text = REPLAY_SRC.read_text(encoding="utf-8")
    assert "os._exit(0)" in text


def test_module_import_never_touches_datasets_at_top_level():
    """datasets/pyarrow/huggingface_hub must only be imported lazily inside
    function bodies - importing tuned.data.replay must never hit the
    network. Verified via AST rather than sys.modules because the test
    venv has datasets installed for other reasons."""
    tree = ast.parse(REPLAY_SRC.read_text(encoding="utf-8"))
    banned = {"datasets", "pyarrow", "huggingface_hub"}
    for node in tree.body:  # module-level statements only, not nested in defs
        if isinstance(node, ast.Import):
            names = {alias.name.split(".")[0] for alias in node.names}
            assert not (names & banned), f"top-level import of {names & banned}"
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in banned, f"top-level `from {node.module} import ...`"


def test_module_importable_without_error():
    import importlib

    import tuned.data.replay as replay_mod
    importlib.reload(replay_mod)
    assert hasattr(replay_mod, "build_replay")
