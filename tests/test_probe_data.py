"""build_probe: the seq-ceiling probe is only honest if every example
really exceeds the target length - undershoot is a false-green probe."""

import json

import pytest

from tuned.data.probe import build_probe, message_chars


def _write_smoke(path, n_rows, content_chars):
    rows = [
        {
            "messages": [
                {"role": "user", "content": "q" * (content_chars // 2)},
                {"role": "assistant", "content": "a" * (content_chars // 2)},
            ]
        }
        for _ in range(n_rows)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return rows


def test_probe_examples_exceed_char_threshold(tmp_path):
    src, out = tmp_path / "smoke.jsonl", tmp_path / "probe.jsonl"
    _write_smoke(src, n_rows=200, content_chars=4000)
    assert build_probe(out, src, n=8, target_tokens=6144, chars_per_token=6) == 8
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8
    for line in lines:
        ex = json.loads(line)
        assert message_chars(ex) >= 6 * 6144


def test_probe_preserves_alternating_roles(tmp_path):
    src, out = tmp_path / "smoke.jsonl", tmp_path / "probe.jsonl"
    _write_smoke(src, n_rows=60, content_chars=10000)
    build_probe(out, src, n=2)
    for line in out.read_text(encoding="utf-8").splitlines():
        roles = [m["role"] for m in json.loads(line)["messages"]]
        assert roles[::2] == ["user"] * (len(roles) // 2)
        assert roles[1::2] == ["assistant"] * (len(roles) // 2)


def test_probe_fails_loudly_on_small_source(tmp_path):
    src, out = tmp_path / "smoke.jsonl", tmp_path / "probe.jsonl"
    _write_smoke(src, n_rows=3, content_chars=1000)  # 3k chars total < one example
    with pytest.raises(RuntimeError, match="too small"):
        build_probe(out, src, n=8)
