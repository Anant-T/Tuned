import json

from tuned.data.smoke import build_smoke, format_example


def test_format_example_shape():
    ex = format_example("What is 2+2?", "Two plus two makes four.", "4")
    assert ex["messages"][0] == {"role": "user", "content": "What is 2+2?"}
    assert ex["messages"][1]["role"] == "assistant"
    assert "Two plus two makes four." in ex["messages"][1]["content"]
    assert ex["messages"][1]["content"].endswith("4")


def test_build_smoke_writes_jsonl(tmp_path):
    rows = [
        {"problem": f"q{i}", "deepseek_reasoning": f"r{i}", "deepseek_solution": f"s{i}"}
        for i in range(5)
    ]
    out = tmp_path / "smoke.jsonl"
    n = build_smoke(out, n=3, rows=iter(rows))
    assert n == 3
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["messages"][0]["content"] == "q0"


def test_build_smoke_skips_incomplete_rows(tmp_path):
    rows = [
        {"problem": "q0", "deepseek_reasoning": "", "deepseek_solution": "s0"},
        {"problem": "q1", "deepseek_reasoning": "r1", "deepseek_solution": "s1"},
    ]
    out = tmp_path / "smoke.jsonl"
    n = build_smoke(out, n=5, rows=iter(rows))
    assert n == 1
