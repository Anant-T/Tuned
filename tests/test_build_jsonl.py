import gzip
import json

import pytest

from tuned.data.jsonl import append_ndjson, read_at, read_jsonl, write_jsonl


def test_write_read_round_trip(tmp_path):
    out = tmp_path / "rows.jsonl"
    rows = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
    n = write_jsonl(out, rows)
    assert n == 2
    result = list(read_jsonl(out))
    assert result == rows


def test_write_read_round_trip_non_ascii(tmp_path):
    out = tmp_path / "rows.jsonl"
    rows = [{"text": "धारा 302"}, {"text": "मृत्युदंड"}]
    n = write_jsonl(out, rows)
    assert n == 2
    result = list(read_jsonl(out))
    assert result == rows
    # ensure_ascii=False: the raw bytes on disk contain the Devanagari text,
    # not \u escape sequences
    raw = out.read_text(encoding="utf-8")
    assert "धारा 302" in raw
    assert "\\u" not in raw


def test_write_jsonl_creates_parent_dirs(tmp_path):
    out = tmp_path / "nested" / "deep" / "rows.jsonl"
    n = write_jsonl(out, [{"a": 1}])
    assert n == 1
    assert out.exists()


def test_write_jsonl_atomic_on_mid_iteration_failure(tmp_path):
    out = tmp_path / "rows.jsonl"
    original_rows = [{"id": 1}, {"id": 2}]
    write_jsonl(out, original_rows)
    original_content = out.read_text(encoding="utf-8")

    def bad_rows():
        yield {"id": 99}
        raise RuntimeError("boom mid-write")

    with pytest.raises(RuntimeError, match="boom mid-write"):
        write_jsonl(out, bad_rows())

    # Original file must be untouched - no partial/replaced content.
    assert out.read_text(encoding="utf-8") == original_content
    assert list(read_jsonl(out)) == original_rows
    # No stray .tmp file left behind on the happy path check isn't required,
    # but the real target file must never have been swapped.


def test_read_jsonl_skips_blank_lines(tmp_path):
    out = tmp_path / "rows.jsonl"
    out.write_text('{"a": 1}\n\n{"a": 2}\n   \n{"a": 3}\n', encoding="utf-8")
    result = list(read_jsonl(out))
    assert result == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_read_jsonl_gzip(tmp_path):
    out = tmp_path / "rows.jsonl.gz"
    rows = [{"a": 1}, {"a": 2, "text": "धारा 302"}]
    with gzip.open(out, mode="wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = list(read_jsonl(out))
    assert result == rows


def test_append_ndjson_offsets_round_trip(tmp_path):
    out = tmp_path / "log.ndjson"
    objs = [{"i": 0, "msg": "first"}, {"i": 1, "msg": "second"}, {"i": 2, "msg": "धारा"}]
    offsets = [append_ndjson(out, obj) for obj in objs]
    # offsets are strictly increasing byte positions
    assert offsets == sorted(offsets)
    assert len(set(offsets)) == 3
    for offset, expected in zip(offsets, objs):
        assert read_at(out, offset) == expected


def test_append_ndjson_creates_parent_dirs(tmp_path):
    out = tmp_path / "nested" / "log.ndjson"
    offset = append_ndjson(out, {"a": 1})
    assert offset == 0
    assert read_at(out, offset) == {"a": 1}


def test_append_ndjson_first_offset_is_zero(tmp_path):
    out = tmp_path / "log.ndjson"
    offset = append_ndjson(out, {"a": 1})
    assert offset == 0
