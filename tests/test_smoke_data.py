import json

import pytest

from tuned.data.smoke import build_smoke, format_example


def test_format_example_shape():
    ex = format_example("What is 2+2?", "Two plus two makes four.", "4")
    assert ex["messages"][0] == {"role": "user", "content": "What is 2+2?"}
    assert ex["messages"][1]["role"] == "assistant"
    assert "Two plus two makes four." in ex["messages"][1]["content"]
    assert ex["messages"][1]["content"].endswith("4")


def test_build_smoke_writes_jsonl(tmp_path):
    # Real OpenThoughts-114k schema: conversations list with user/assistant messages
    rows = [
        {
            "conversations": [
                {"from": "user", "value": f"q{i}"},
                {"from": "assistant", "value": f"<|begin_of_thought|>r{i}<|end_of_thought|>\n<|begin_of_solution|>s{i}<|end_of_solution|>"}
            ]
        }
        for i in range(5)
    ]
    out = tmp_path / "smoke.jsonl"
    n = build_smoke(out, n=3, rows=iter(rows))
    assert n == 3
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["messages"][0]["content"] == "q0"
    # Verify no markup tags in output
    assert "<|" not in first["messages"][1]["content"]


def test_build_smoke_skips_incomplete_rows(tmp_path):
    rows = [
        {
            "conversations": [
                {"from": "user", "value": "q0"},
                {"from": "assistant", "value": ""}  # Empty reasoning/solution
            ]
        },
        {
            "conversations": [
                {"from": "user", "value": "q1"},
                {"from": "assistant", "value": "<|begin_of_thought|>r1<|end_of_thought|>\n<|begin_of_solution|>s1<|end_of_solution|>"}
            ]
        },
    ]
    out = tmp_path / "smoke.jsonl"
    with pytest.raises(RuntimeError, match="only wrote 1 of 5"):
        build_smoke(out, n=5, rows=iter(rows))
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["messages"][0]["content"] == "q1"


def test_build_smoke_no_markup_in_output(tmp_path):
    """Verify extracted reasoning and solution have no thinking-tag markup."""
    rows = [
        {
            "conversations": [
                {"from": "user", "value": "What is 2+2?"},
                {"from": "assistant", "value": "<|begin_of_thought|>Let me think about this<|end_of_thought|>\n<|begin_of_solution|>The answer is 4<|end_of_solution|>"}
            ]
        },
    ]
    out = tmp_path / "smoke.jsonl"
    n = build_smoke(out, n=1, rows=iter(rows))
    assert n == 1
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    data = json.loads(lines[0])
    assistant_content = data["messages"][1]["content"]
    # No thinking tags should be present
    assert "<|begin_of_thought|>" not in assistant_content
    assert "<|end_of_thought|>" not in assistant_content
    assert "<|begin_of_solution|>" not in assistant_content
    assert "<|end_of_solution|>" not in assistant_content
    assert "<|" not in assistant_content


def test_format_example_with_think_tags():
    ex = format_example(
        "What is 2+2?", "Two plus two makes four.", "4",
        think_open="[THINK]", think_close="[/THINK]",
    )
    content = ex["messages"][1]["content"]
    assert content == "[THINK]Two plus two makes four.[/THINK]4"


def test_build_smoke_wraps_with_think_tags(tmp_path):
    rows = [
        {
            "conversations": [
                {"from": "user", "value": "q0"},
                {"from": "assistant", "value": "<|begin_of_thought|>r0<|end_of_thought|>\n<|begin_of_solution|>s0<|end_of_solution|>"},
            ]
        },
    ]
    out = tmp_path / "smoke.jsonl"
    n = build_smoke(out, n=1, rows=iter(rows), think_open="[THINK]", think_close="[/THINK]")
    assert n == 1
    data = json.loads(out.read_text(encoding="utf-8").strip())
    assert data["messages"][1]["content"] == "[THINK]r0[/THINK]s0"


