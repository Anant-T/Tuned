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
    # Without an index the verify step must NOT pass --index (verify warns
    # UNVERIFIED); with one, the existence half arms.
    assert "--index" not in argvs[0]
    armed = actions_worker.assemble_argvs("cfg.yaml", citation_index=Path("x/citation_index.txt"))
    assert armed[0][-2:] == ["--index", str(Path("x/citation_index.txt"))]


def test_naming_the_streams_inserts_shape_and_feeds_decontaminate_its_output():
    """The pools are sized for the FINISHED corpus; feeding them whole to a
    half-generated one is what put mix/trace/empty_think red on 2026-08-29.
    Shape runs first and decontaminate must read ITS files, not the pools."""
    out = Path("w/out")
    argvs = actions_worker.assemble_argvs(
        "cfg.yaml", streams=["curated_c1", "replay"], out_dir=out
    )
    modules = [a[3] for a in argvs]
    assert modules == [
        "tuned.data.verify", "tuned.data.shape", "tuned.data.decontaminate",
        "tuned.data.dedupe", "tuned.data.split", "tuned.data.assemble",
        "tuned.data.stats",
    ]
    assert argvs[1][-2:] == ["--profile", "v1.0-MVP"]
    decon = argvs[2]
    assert decon[-4:] == [
        "--in", str(out / "shaped_curated_c1.jsonl"),
        "--in", str(out / "shaped_replay.jsonl"),
    ]
    # Both stages must be told the SAME profile, or shape sizes the corpus
    # against one mix and stats gates it against another.
    assert argvs[1][argvs[1].index("--profile") + 1] == \
        argvs[-1][argvs[-1].index("--profile") + 1]


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

    # The citation index rides along as ONE file; the corpus dir's source
    # text must never enter the baton.
    (root / actions_worker.INDEX_RELPATH).parent.mkdir(parents=True)
    (root / actions_worker.INDEX_RELPATH).write_text("2023 insc 45\n")
    (root / "corpus" / "extraction.jsonl").write_text('{"big":"source text"}\n')

    staging = actions_worker.stage_bundle(root, tmp_path / "staging")
    assert (staging / actions_worker.DB_RELPATH).is_file()
    assert (staging / "raw" / "gen" / "2026-08-29" / "gen.ndjson").is_file()
    assert (staging / "streams" / "replay.jsonl").is_file()
    assert (staging / actions_worker.INDEX_RELPATH).is_file()
    assert not (staging / "corpus" / "extraction.jsonl").exists()

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
    assert (root2 / actions_worker.INDEX_RELPATH).is_file()


import audit_readout  # noqa: E402  (same data/scripts sys.path as above)


def test_audit_readout_computes_the_sample_accept_rate(tmp_path):
    db = tmp_path / "law_v1.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE task (task_id TEXT, state TEXT, disposition TEXT)")
    rows = (
        [("a%d" % i, "accepted", "audit:gate-accept") for i in range(7)]
        + [("s%d" % i, "accepted", "judge:accept") for i in range(3)]
        + [("r0", "rejected", "judge:reject")]
        + [("p0", "pending", None)]
    )
    conn.executemany("INSERT INTO task VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()

    ro = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    s = audit_readout.summarize(ro)
    ro.close()
    assert s["audit_accepts"] == 7
    assert (s["sampled_decided"], s["sampled_accepted"]) == (4, 3)
    assert abs(s["sample_accept_rate"] - 0.75) < 1e-9
    assert s["states"] == {"accepted": 10, "pending": 1, "rejected": 1}


def test_audit_readout_survives_an_unjudged_store(tmp_path):
    db = tmp_path / "law_v1.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE task (task_id TEXT, state TEXT, disposition TEXT)")
    conn.commit()
    conn.close()
    ro = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    s = audit_readout.summarize(ro)
    ro.close()
    assert s["sample_accept_rate"] is None
    assert s["audit_accepts"] == 0
