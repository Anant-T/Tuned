"""data/scripts/actions_worker.py - the remote-baton supervisor's pure parts.

No network and no children here: the HF wrapper and the Popen loop are thin
and injectable; what must not drift silently is the fleet shape (exactly one
generator), the assembly-chain order, and the snapshot/stage/restore
round-trip the baton depends on.
"""

import sqlite3
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parent.parent / "data" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import actions_worker  # noqa: E402


def test_the_fleet_is_one_generator_and_one_judge():
    argvs = actions_worker.child_argvs("cfg.yaml", n_workers=8, audit_sample=0.05)
    assert len(argvs) == 2
    gen, judge = argvs
    # Exactly ONE generating process, ever: two would share a raw append
    # target and double-run the per-process rate bucket.
    assert sum("tuned.data.generate" in a for a in (gen, judge)) == 1
    assert "tuned.data.generate" in gen
    assert gen[gen.index("--n-workers") + 1] == "8"
    assert "--forever" in gen
    assert "tuned.data.judge" in judge
    assert judge[judge.index("--audit-sample") + 1] == "0.05"
    for argv in argvs:
        for stream in actions_worker.STREAMS:
            assert stream in argv
        assert argv[argv.index("--config") + 1] == "cfg.yaml"


def test_the_assembly_chain_runs_in_order_and_stats_gates_last():
    argvs = actions_worker.assemble_argvs("cfg.yaml")
    modules = [a[3] for a in argvs]
    assert modules == [
        "tuned.data.verify", "tuned.data.decontaminate", "tuned.data.dedupe",
        "tuned.data.split", "tuned.data.assemble", "tuned.data.stats",
    ]
    assert argvs[-1][-2:] == ["--profile", "v1.0-MVP"]


def _tiny_db(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (v TEXT)")
    conn.execute("INSERT INTO t VALUES (?)", (marker,))
    conn.commit()
    return conn


def test_snapshot_db_is_consistent_while_the_source_is_open(tmp_path):
    src = tmp_path / "state" / "law_v1.sqlite3"
    conn = _tiny_db(src, "committed")
    # An OPEN WAL handle with a committed row: the snapshot must read
    # through the -wal, which a plain file copy of the main db would miss.
    dest = tmp_path / "snap.sqlite3"
    actions_worker.snapshot_db(src, dest)
    conn.close()
    out = sqlite3.connect(f"file:{dest.as_posix()}?mode=ro", uri=True)
    assert out.execute("SELECT v FROM t").fetchone() == ("committed",)
    out.close()
    # And it overwrites: VACUUM INTO refuses an existing target on its own.
    actions_worker.snapshot_db(src, dest)


def test_stage_and_restore_round_trip_the_baton(tmp_path):
    root = tmp_path / "build"
    _tiny_db(root / actions_worker.DB_RELPATH, "baton").close()
    (root / "raw" / "gen" / "2026-08-29").mkdir(parents=True)
    (root / "raw" / "gen" / "2026-08-29" / "gen.ndjson").write_text('{"kind":"generation"}\n')
    (root / "streams").mkdir()
    (root / "streams" / "replay.jsonl").write_text('{"row":1}\n')

    staging = actions_worker.stage_bundle(root, tmp_path / "staging")
    assert (staging / actions_worker.DB_RELPATH).is_file()
    assert (staging / "raw" / "gen" / "2026-08-29" / "gen.ndjson").is_file()
    assert (staging / "streams" / "replay.jsonl").is_file()

    root2 = tmp_path / "build2"
    root2.mkdir()
    # A stale -wal beside the destination DB would be replayed over the
    # self-contained snapshot; restore must drop it.
    stale_wal = (root2 / actions_worker.DB_RELPATH).with_name(
        actions_worker.DB_RELPATH.name + "-wal"
    )
    stale_wal.parent.mkdir(parents=True)
    stale_wal.write_bytes(b"stale")
    actions_worker.restore_bundle(staging, root2)

    assert not stale_wal.exists()
    conn = sqlite3.connect(f"file:{(root2 / actions_worker.DB_RELPATH).as_posix()}?mode=ro", uri=True)
    assert conn.execute("SELECT v FROM t").fetchone() == ("baton",)
    conn.close()
    assert (root2 / "streams" / "replay.jsonl").read_text() == '{"row":1}\n'
    assert (root2 / "raw" / "gen" / "2026-08-29" / "gen.ndjson").is_file()
