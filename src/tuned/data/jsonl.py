"""JSONL/NDJSON I/O primitives for the law_v1 data-build pipeline.

read_jsonl / write_jsonl cover whole-file round-trips (write_jsonl is
atomic via a temp-file + os.replace swap). append_ndjson is the
streaming-log primitive: every call returns the byte offset the record
was written at, so a caller can persist (path, offset) and later seek
straight back to that exact line with read_at - this is how downstream
modules index generation/judge logs without re-scanning them.
"""

import gzip
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, mode="rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    written = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    os.replace(tmp_path, path)
    return written


def append_ndjson(path: str | Path, obj: dict) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as f:
        offset = f.tell()
        f.write(json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n")
        f.flush()
        os.fsync(f.fileno())
    return offset


def read_at(path: str | Path, offset: int) -> dict:
    path = Path(path)
    with path.open("rb") as f:
        f.seek(offset)
        line = f.readline()
    return json.loads(line.decode("utf-8"))
