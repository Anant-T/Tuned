"""data/scripts/actions_worker.py - the remote-baton supervisor's pure parts.

No network and no children here: the HF wrapper and the Popen loop are thin
and injectable; what must not drift silently is the fleet shape (exactly one
generator), the assembly-chain order, and the snapshot/stage/restore
round-trip the baton depends on.
"""

import sqlite3
import sys

import pytest
from pathlib import Path

SCRIPTS = Path(__file__).parent.parent / "data" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import actions_worker  # noqa: E402


def test_the_supervisor_default_sample_is_the_judges_own_constant():
    """The audit fraction IS the quality warrant for every audit-accepted row.

    While this was a bare 0.05 literal here, editing DEFAULT_AUDIT_SAMPLE in
    judge.py changed nothing about the unattended run - the supervisor always
    passed its own copy explicitly, so the value looked tunable and was not.
    """
    from tuned.data.judge import DEFAULT_AUDIT_SAMPLE

    parsed = actions_worker.main_parser().parse_args(
        ["--phase", "worker", "--hf-repo", "u/r"]
    )
    assert parsed.audit_sample == DEFAULT_AUDIT_SAMPLE


def test_the_fleet_is_one_generator_and_one_judge():
    from tuned.data.judge import DEFAULT_AUDIT_SAMPLE

    argvs = actions_worker.child_argvs(
        "cfg.yaml", n_workers=8, audit_sample=DEFAULT_AUDIT_SAMPLE
    )
    assert len(argvs) == 2
    gen, judge = argvs
    # Exactly ONE generating process, ever: two would share a raw append
    # target and double-run the per-process rate bucket.
    assert sum("tuned.data.generate" in a for a in (gen, judge)) == 1
    assert "tuned.data.generate" in gen
    assert gen[gen.index("--n-workers") + 1] == "8"
    assert "--forever" in gen
    assert "tuned.data.judge" in judge
    assert judge[judge.index("--audit-sample") + 1] == str(DEFAULT_AUDIT_SAMPLE)
    for argv in argvs:
        for stream in actions_worker.STREAMS:
            assert stream in argv
        assert argv[argv.index("--config") + 1] == "cfg.yaml"


def test_dropping_a_stream_from_STREAMS_stops_the_fleet_claiming_it():
    """STREAMS is the only brake on a stream that is over-generating.

    `--stream` becomes store.claim_tasks(stream=...), so a stream missing
    from this tuple is one the generator never claims - its planned tasks
    simply stay pending, which is why editing the tuple is a reversible
    throttle that touches neither the store nor the baton. The sibling test
    above pins that every LISTED stream reaches both children; this pins the
    converse, which is the half the brake actually rests on. Without it,
    child_argvs could restate the stream list inline and disarm the throttle
    while every other test stayed green.
    """
    from tuned.data.judge import DEFAULT_AUDIT_SAMPLE

    argvs = actions_worker.child_argvs(
        "cfg.yaml", n_workers=8, audit_sample=DEFAULT_AUDIT_SAMPLE
    )
    for argv in argvs:
        passed = {argv[i + 1] for i, a in enumerate(argv) if a == "--stream"}
        assert passed == set(actions_worker.STREAMS)


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


def _audit_db(tmp_path, rows, name="law_v1.sqlite3"):
    """rows: (task_id, state, disposition, stream) tuples."""
    db = tmp_path / name
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE task (task_id TEXT, state TEXT, disposition TEXT, stream TEXT)")
    conn.executemany("INSERT INTO task VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return db


def test_audit_readout_computes_the_sample_accept_rate(tmp_path):
    rows = (
        [("a%d" % i, "accepted", "audit:gate-accept", "synthesis") for i in range(7)]
        + [("s%d" % i, "accepted", "judge:accept", "synthesis") for i in range(3)]
        + [("r0", "rejected", "judge:reject", "synthesis")]
        + [("p0", "pending", None, "synthesis")]
    )
    db = _audit_db(tmp_path, rows)

    ro = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    s = audit_readout.summarize(ro)
    ro.close()
    assert s["audit_accepts"] == 7
    assert (s["sampled_decided"], s["sampled_accepted"]) == (4, 3)
    assert abs(s["sample_accept_rate"] - 0.75) < 1e-9
    assert s["states"] == {"accepted": 10, "pending": 1, "rejected": 1}


def test_audit_readout_survives_an_unjudged_store(tmp_path):
    db = _audit_db(tmp_path, [])
    ro = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    s = audit_readout.summarize(ro)
    ro.close()
    assert s["sample_accept_rate"] is None
    assert s["audit_accepts"] == 0


def test_audit_readout_breaks_the_sample_out_by_stream(tmp_path):
    """(b) of P1.7: a pooled-only rate cannot tell an operator WHICH stream
    is failing - synthesis and transition here disagree completely, and the
    pooled rate alone (2/4 = 50%) hides that."""
    rows = (
        [("s0", "accepted", "judge:accept", "synthesis"),
         ("s1", "accepted", "judge:accept", "synthesis"),
         ("t0", "rejected", "judge:reject", "transition"),
         ("t1", "rejected", "judge:reject", "transition")]
    )
    db = _audit_db(tmp_path, rows)
    ro = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    s = audit_readout.summarize(ro)
    ro.close()
    assert s["by_stream"]["synthesis"] == {"decided": 2, "accepted": 2, "accept_rate": 1.0}
    assert s["by_stream"]["transition"] == {"decided": 2, "accepted": 0, "accept_rate": 0.0}
    lines = "\n".join(audit_readout.format_summary(s))
    assert "synthesis: 2 decided, 2 accepted -> accept rate 100.0%" in lines
    assert "transition: 2 decided, 0 accepted -> accept rate 0.0%" in lines


def test_audit_readout_since_excludes_pre_window_judgements(tmp_path):
    """(a) of P1.7: since=None pools everything (today's behaviour,
    unchanged); a since timestamp restricts the sample to judgements whose
    LATEST judgement.created_at falls on or after it. judgement.created_at,
    not task.updated_at, is what dates a judgement - see summarize's own
    docstring for why."""
    db = tmp_path / "law_v1.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE task (task_id TEXT, state TEXT, disposition TEXT, stream TEXT)")
    conn.execute("CREATE TABLE generation (gen_id INTEGER PRIMARY KEY, task_id TEXT)")
    conn.execute("CREATE TABLE judgement (gen_id INTEGER, judge_slot TEXT, created_at TEXT)")
    conn.executemany(
        "INSERT INTO task VALUES (?, ?, ?, ?)",
        [
            ("old", "accepted", "judge:accept", "synthesis"),   # pre-window
            ("new", "rejected", "judge:reject", "synthesis"),   # in-window
        ],
    )
    conn.execute("INSERT INTO generation VALUES (1, 'old')")
    conn.execute("INSERT INTO generation VALUES (2, 'new')")
    conn.execute("INSERT INTO judgement VALUES (1, 'A', '2026-08-01T00:00:00Z')")
    conn.execute("INSERT INTO judgement VALUES (2, 'A', '2026-08-30T00:00:00Z')")
    conn.commit()
    conn.close()

    ro = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    unfiltered = audit_readout.summarize(ro)
    assert (unfiltered["sampled_decided"], unfiltered["sampled_accepted"]) == (2, 1)

    windowed = audit_readout.summarize(ro, since="2026-08-15T00:00:00Z")
    ro.close()
    assert (windowed["sampled_decided"], windowed["sampled_accepted"]) == (1, 0)
    assert windowed["since"] == "2026-08-15T00:00:00Z"
    assert "since=2026-08-15T00:00:00Z" in "\n".join(audit_readout.format_summary(windowed))


def test_audit_readout_since_uses_the_latest_judgement_per_task(tmp_path):
    """A task with two judge slots (a tiebreak) is windowed on the LATEST
    slot's created_at, not the earliest - the disposition is only decided
    once every required slot is in, so the last slot is when that decision
    was actually made. A task whose slot A predates the window but whose
    slot B (the tiebreak) falls inside it must still be counted."""
    db = tmp_path / "law_v1.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE task (task_id TEXT, state TEXT, disposition TEXT, stream TEXT)")
    conn.execute("CREATE TABLE generation (gen_id INTEGER PRIMARY KEY, task_id TEXT)")
    conn.execute("CREATE TABLE judgement (gen_id INTEGER, judge_slot TEXT, created_at TEXT)")
    conn.execute(
        "INSERT INTO task VALUES ('t0', 'accepted', 'judge:accept', 'synthesis')"
    )
    conn.execute("INSERT INTO generation VALUES (1, 't0')")
    # slot A well before the window, slot B (tiebreak) inside it.
    conn.execute("INSERT INTO judgement VALUES (1, 'A', '2026-08-01T00:00:00Z')")
    conn.execute("INSERT INTO judgement VALUES (1, 'B', '2026-08-30T00:00:00Z')")
    conn.commit()
    conn.close()

    ro = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    s = audit_readout.summarize(ro, since="2026-08-15T00:00:00Z")
    ro.close()
    assert s["sampled_decided"] == 1


# --------------------------------------------------------------------------
# The audit-sample collapse refusal (P1.7 part 2): a Wilson upper bound on
# the dual-judged sample, gating run_assemble's push step. Hand-checked
# against the closed-form Wilson formula independently of the implementation.
# --------------------------------------------------------------------------

def test_wilson_upper_bound_hand_checked_cases():
    assert abs(actions_worker.wilson_upper_bound(0, 50) - 0.071350) < 1e-5
    assert abs(actions_worker.wilson_upper_bound(25, 50) - 0.633557) < 1e-5


def test_wilson_upper_bound_sits_above_the_point_estimate():
    """The whole reason the refusal reads the upper bound instead of the raw
    accepted/decided ratio: it must be the BEST case the sample is
    consistent with, not the (noisier) point estimate."""
    accepted, decided = 5, 100
    upper = actions_worker.wilson_upper_bound(accepted, decided)
    assert upper > accepted / decided
    assert abs(upper - 0.111752) < 1e-5


def test_wilson_upper_bound_of_no_evidence_is_not_collapse():
    assert actions_worker.wilson_upper_bound(0, 0) == 1.0


# routing.judge_mode_since ships as the commit that turned audit mode on.
# A judgement stamped before it was made under the DUAL regime by a fleet
# since retired, and pooling the two eras is what made "30.4%" look like an
# audit number.
AUDIT_ERA = "2026-08-30T12:00:00.000000Z"
PRE_AUDIT = "2026-08-01T12:00:00.000000Z"


def _judged_tables(conn):
    """The three tables the windowed sample reads, with the real column names.

    A task carrying a `judge:%` disposition and NO judgement row is not a
    shape the store can produce - record_judgement is what puts the verdict
    there in the first place - so a fixture that omits them is testing a
    population that cannot exist.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS task (task_id TEXT, state TEXT, disposition TEXT, "
        "stream TEXT, claimed_at TEXT)"
    )
    conn.execute("CREATE TABLE IF NOT EXISTS generation (gen_id INTEGER, task_id TEXT)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS judgement (gen_id INTEGER, judge_slot TEXT, created_at TEXT)"
    )


def _judged_rows(conn, dispositions, *, at=AUDIT_ERA, stream="synthesis"):
    for i, disposition in enumerate(dispositions):
        conn.execute(
            "INSERT INTO task (task_id, state, disposition, stream, claimed_at) "
            "VALUES (?, 'accepted', ?, ?, NULL)",
            (f"t{i}-{stream}-{at}", disposition, stream),
        )
        gen_id = conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0] + 1
        conn.execute("INSERT INTO generation VALUES (?, ?)",
                     (gen_id, f"t{i}-{stream}-{at}"))
        conn.execute("INSERT INTO judgement VALUES (?, 'a', ?)", (gen_id, at))


def _collapse_db(tmp_path, dispositions, *, at=AUDIT_ERA):
    db = tmp_path / "law_v1.sqlite3"
    conn = sqlite3.connect(db)
    _judged_tables(conn)
    _judged_rows(conn, dispositions, at=at)
    conn.commit()
    conn.close()
    return db


def test_collapse_refusal_none_below_the_minimum_sample_even_at_zero_percent(tmp_path):
    """Below MIN_SAMPLE_FOR_COLLAPSE_CHECK the sample says nothing - it must
    not refuse even when every decided row so far was rejected."""
    db = _collapse_db(tmp_path, ["judge:reject"] * 49)
    assert actions_worker._audit_collapse_refusal(db, 0.20) is None


def test_collapse_refusal_fires_on_a_large_sample_below_the_floor(tmp_path):
    db = _collapse_db(tmp_path, ["judge:accept"] * 5 + ["judge:reject"] * 95)
    msg = actions_worker._audit_collapse_refusal(db, 0.20)
    assert msg is not None
    assert "AUDIT SAMPLE COLLAPSE" in msg
    assert "5/100" in msg


def test_collapse_refusal_none_on_a_large_sample_above_the_floor(tmp_path):
    db = _collapse_db(tmp_path, ["judge:accept"] * 70 + ["judge:reject"] * 30)
    assert actions_worker._audit_collapse_refusal(db, 0.20) is None


def test_collapse_refusal_none_when_the_state_db_does_not_exist(tmp_path):
    assert actions_worker._audit_collapse_refusal(tmp_path / "nope.sqlite3", 0.20) is None


def test_the_window_keeps_the_pre_audit_era_out_of_the_gate(tmp_path):
    """The reason `since` is load-bearing rather than a nicety.

    verify.py's armed cut rewrites an off-teacher row's disposition out of
    `judge:%` when the row sits in accepted/judging - but a `rejected` row is
    a decision verify never touches. Pooling the eras therefore drops the old
    ACCEPTS and keeps the old REJECTS, which depresses the rate this gate
    reads and makes a FALSE refusal more likely, not less. Here that is the
    difference between refusing and not.
    """
    db = tmp_path / "law_v1.sqlite3"
    conn = sqlite3.connect(db)
    _judged_tables(conn)
    # The old regime's rejects survive verify's cut (a rejected row is a
    # decision it never touches) while its accepts do not, so this is the
    # asymmetry the window exists to undo - not a contrived ratio.
    _judged_rows(conn, ["judge:reject"] * 400, at=PRE_AUDIT)
    _judged_rows(conn, ["judge:accept"] * 50, at=AUDIT_ERA)
    conn.commit()
    conn.close()

    # Pooled, the old rejects swamp a healthy audit era and the gate refuses.
    assert actions_worker._audit_collapse_refusal(db, 0.20) is not None
    # Windowed, it reads the era it is actually gating.
    assert actions_worker._audit_collapse_refusal(db, 0.20, AUDIT_ERA) is None


def test_a_verdict_the_window_cannot_date_is_left_out(tmp_path):
    """Conservative on purpose: a `judge:%` row with no judgement to date is
    dropped from the sample rather than assumed to be in it. A smaller sample
    is harder to trip a collapse floor with, which is the right direction for
    a refusal that has to be certain."""
    db = tmp_path / "law_v1.sqlite3"
    conn = sqlite3.connect(db)
    _judged_tables(conn)
    conn.executemany(
        "INSERT INTO task (task_id, state, disposition, stream, claimed_at) "
        "VALUES (?, 'accepted', 'judge:reject', 'synthesis', NULL)",
        [(f"orphan{i}",) for i in range(100)],
    )
    conn.commit()
    conn.close()

    assert actions_worker._audit_collapse_refusal(db, 0.20, AUDIT_ERA) is None
    # ...and pooled, with no window to date them against, they still count.
    assert actions_worker._audit_collapse_refusal(db, 0.20) is not None


def _assemble_root_with(tmp_path, dispositions):
    root = tmp_path / "build"
    db = root / actions_worker.DB_RELPATH
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    _judged_tables(conn)
    _judged_rows(conn, dispositions)
    conn.commit()
    conn.close()
    return root


class _CallRecorder:
    """Stands in for subprocess.run across run_assemble's whole chain: every
    step reports rc=0, and every argv is kept so a test can tell whether
    push.py was actually invoked."""

    def __init__(self):
        self.argvs: list[list[str]] = []

    def __call__(self, argv, *a, **k):
        self.argvs.append(argv)
        return type("R", (), {"returncode": 0})()


def _run_assemble(tmp_path, monkeypatch, dispositions, with_out=False):
    from pipeline_fakes import temp_config

    cfg_path = temp_config(tmp_path)
    root = _assemble_root_with(tmp_path, dispositions)
    if with_out:
        # The artifact-upload branch is guarded by out_dir.is_dir(); in
        # production paths.ensure() has always made it, so leaving it out of
        # the fixture is what kept that branch unexecuted by the suite.
        (root / "out").mkdir(parents=True)
        (root / "out" / "stats.json").write_text("{}", encoding="utf-8")
    args = actions_worker.main_parser().parse_args(
        ["--phase", "assemble", "--hf-repo", "u/r", "--config", cfg_path]
    )
    recorder = _CallRecorder()
    monkeypatch.setattr(actions_worker.subprocess, "run", recorder)
    monkeypatch.setattr(actions_worker.time, "sleep", lambda s: None)
    bundle = _FakeBundle()
    rc = actions_worker.run_assemble(args, root, bundle)
    return rc, recorder.argvs, bundle


def test_run_assemble_refuses_the_push_step_on_a_collapsed_sample(tmp_path, monkeypatch, capsys):
    dispositions = ["judge:accept"] * 5 + ["judge:reject"] * 95
    rc, argvs, _ = _run_assemble(tmp_path, monkeypatch, dispositions)
    assert rc != 0
    assert not any("tuned.data.push" in argv for argv in argvs)
    assert "AUDIT SAMPLE COLLAPSE" in capsys.readouterr().out


def test_run_assemble_pushes_when_the_sample_is_healthy(tmp_path, monkeypatch):
    dispositions = ["judge:accept"] * 90 + ["judge:reject"] * 10
    rc, argvs, _ = _run_assemble(tmp_path, monkeypatch, dispositions)
    assert rc == 0
    assert any("tuned.data.push" in argv for argv in argvs)


def test_the_artifacts_upload_through_the_fence_and_before_the_checkpoint(
    tmp_path, monkeypatch
):
    """Covers the branch that shipped broken because nothing executed it.

    run_assemble uploads out/ and then pushes the post-assembly checkpoint.
    Both must go through Bundle, in that order: the artifact commit moves the
    remote head, so a checkpoint that does not fence against IT is refused as
    a 412 and re-raised as BATON STOLEN - which is what discarded the re-gate
    results and the off_teacher demotions on every dispatch.
    """
    dispositions = ["judge:accept"] * 90 + ["judge:reject"] * 10
    rc, _, bundle = _run_assemble(
        tmp_path, monkeypatch, dispositions, with_out=True
    )
    assert rc == 0
    assert bundle.pushes == [
        "replace_dir:out", "post-assembly checkpoint",
    ], "the artifacts go up through Bundle, then the checkpoint fences on them"


# --------------------------------------------------------------------------
# The supervisor loop: what it reports, and what it exits with. run_worker
# used to return 0 unconditionally, so no build failure could fail the job or
# fire the Actions failure mail - the only notification this build has.
# --------------------------------------------------------------------------


class _FakeProc:
    """A child that survives `alive_polls` polls, then exits with `rc`."""

    def __init__(self, rc=0, alive_polls=10**6, pid=4242):
        self.returncode = None
        self.pid = pid
        self._rc = rc
        self._left = alive_polls
        self.stdout = None

    def poll(self):
        if self._left <= 0 and self.returncode is None:
            self.returncode = self._rc
        else:
            self._left -= 1
        return self.returncode

    def terminate(self):
        if self.returncode is None:
            self.returncode = -15  # no SIGTERM handler exists in either child

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = -15
        return self.returncode


class _FakeBundle:
    def __init__(self, fail_times=0):
        self.pushes, self.stages, self._fail = [], [], fail_times

    def pull(self, dest):
        Path(dest).mkdir(parents=True, exist_ok=True)
        return Path(dest)

    def push(self, staged, message):
        self.stages.append(str(staged))
        if self._fail > 0:
            self._fail -= 1
            raise RuntimeError("HF 503")
        self.pushes.append(message)

    def replace_dir(self, local, path_in_repo, message):
        # Recorded into the SAME list as push() so a test can assert the
        # ORDER of the two - the artifact upload has to land before the
        # post-assembly checkpoint, and it is the head it leaves behind that
        # the checkpoint fences against.
        self.pushes.append(f"replace_dir:{path_in_repo}")


class _FakePump:
    """The pump threads are real threads in production; here only join() is
    reached, so a stub avoids starting anything."""

    def join(self, timeout=None):
        return None


def _run_worker(tmp_path, monkeypatch, procs, bundle=None, **over):
    root = tmp_path / "build"
    (root / "logs").mkdir(parents=True)
    bundle = bundle or _FakeBundle()
    args = actions_worker.main_parser().parse_args(
        ["--phase", "worker", "--hf-repo", "u/r", "--minutes", "0.001",
         "--push-every", "10000"]
    )
    for key, value in over.items():
        setattr(args, key, value)
    monkeypatch.setattr(actions_worker.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(actions_worker, "stage_bundle", lambda r, s, db=True: s)
    monkeypatch.setattr(actions_worker, "_pump", lambda p, n, path: _FakePump())
    monkeypatch.setattr(actions_worker.time, "sleep", lambda s: None)
    it = iter(procs)
    monkeypatch.setattr(actions_worker.subprocess, "Popen", lambda *a, **k: next(it))
    return actions_worker.run_worker(args, root, bundle), bundle


def test_a_clean_deadline_stop_returns_zero_despite_sigterm_rcs(tmp_path, monkeypatch):
    """THE regression guard for this change.

    Neither child installs a SIGTERM handler, so the ordinary end-of-deadline
    terminate leaves returncode == -15 on BOTH of them on a perfectly healthy
    run. Keying the exit code on all(rc == 0) - the obvious reading - would
    red every successful run instead.
    """
    rc, bundle = _run_worker(tmp_path, monkeypatch, [_FakeProc(), _FakeProc()])
    assert rc == 0
    assert bundle.pushes == ["end-of-job checkpoint"]


def test_a_dead_generator_ends_the_run_red_instead_of_sleeping_it_out(tmp_path, monkeypatch):
    """generate.py raises SystemExit(2) within seconds of start on a missing
    key or an unfillable judge slot. The old any(alive) guard then left the
    judge polling an empty queue for the rest of the 5.25 h window - a quarter
    of the day generation ceiling - and the job still finished green.
    """
    rc, _ = _run_worker(tmp_path, monkeypatch, [_FakeProc(rc=2, alive_polls=0), _FakeProc()])
    assert rc == 1


def test_a_dead_judge_does_not_stop_the_generator(tmp_path, monkeypatch):
    """Both live judges are keyed by one GROQ_API_KEY, so groq going down takes
    them together - which must not stop the one process on the critical path.
    """
    rc, _ = _run_worker(tmp_path, monkeypatch, [_FakeProc(), _FakeProc(rc=1, alive_polls=0)])
    assert rc == 0


def test_the_final_push_retries_and_stages_only_once(tmp_path, monkeypatch):
    """The periodic path is best-effort BECAUSE the final push retries - and
    that delegation was written down while the final push was a single
    unguarded upload_folder. Staging must not repeat: it VACUUMs a ~565 MB
    database and copies the raw tree.
    """
    bundle = _FakeBundle(fail_times=2)
    rc, bundle = _run_worker(tmp_path, monkeypatch, [_FakeProc(), _FakeProc()], bundle=bundle)
    assert rc == 0
    assert bundle.pushes == ["end-of-job checkpoint"]
    assert len(bundle.stages) == 3           # three attempts...
    assert len(set(bundle.stages)) == 1      # ...of ONE staged tree


def test_a_failed_final_push_is_reported_and_fails_the_job(tmp_path, monkeypatch):
    bundle = _FakeBundle(fail_times=99)
    rc, _ = _run_worker(tmp_path, monkeypatch, [_FakeProc(), _FakeProc()], bundle=bundle)
    assert rc == 1


def _store_with(tmp_path, rows):
    """rows: (state, disposition) pairs. Every row gets a fixed stream -
    these tests exercise the run report, not the per-stream breakdown, and
    summarize() now always selects task.stream (see audit_readout.py P1.7)."""
    root = tmp_path / "build"
    db = root / actions_worker.DB_RELPATH
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE task (state TEXT, disposition TEXT, stream TEXT)")
    conn.executemany(
        "INSERT INTO task (state, disposition, stream) VALUES (?, ?, ?)",
        [(r[0], r[1], r[2] if len(r) > 2 else "synthesis") for r in rows],
    )
    conn.commit()
    conn.close()
    return root


def test_the_run_report_names_the_queue_and_the_audit_rate(tmp_path, capsys, monkeypatch):
    """One readout per run, in the job log AND the step summary. Before it the
    only operator surface was a 5.25 h log, and audit_readout.py - the
    documented ship gate - was invoked by no workflow at all.
    """
    root = _store_with(tmp_path, [
        ("accepted", "judge:accept"), ("accepted", "judge:accept"),
        ("rejected", "judge:reject"), ("accepted", "audit:gate-accept"),
        ("pending", None),
    ])
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    rc = actions_worker._finish(
        root, [_FakeProc(), _FakeProc()], ("gen", "judge"),
        gen_died_early=False, final_push_ok=True,
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "[gen] exited rc=" in out
    assert "task states:" in out and "accepted=3" in out
    assert "accept rate 66.7%" in out
    # the same text reaches the Actions run summary, not only the log
    assert "task states:" in summary.read_text(encoding="utf-8")


def test_an_empty_queue_says_so_unmissably(tmp_path, capsys):
    """A drained queue is otherwise indistinguishable from a wedged one: the
    generator prints claimed=0 once and then sleeps to the deadline.
    """
    root = _store_with(tmp_path, [("accepted", "judge:accept"), ("rejected", "judge:reject")])
    actions_worker._finish(root, [_FakeProc()], ("gen",),
                           gen_died_early=False, final_push_ok=True)
    assert "QUEUE EMPTY" in capsys.readouterr().out


def test_a_queue_with_work_left_does_not_claim_to_be_empty(tmp_path, capsys):
    root = _store_with(tmp_path, [("pending", None), ("accepted", "judge:accept")])
    actions_worker._finish(root, [_FakeProc()], ("gen",),
                           gen_died_early=False, final_push_ok=True)
    assert "QUEUE EMPTY" not in capsys.readouterr().out


def test_the_report_separates_servable_work_from_a_throttled_backlog(tmp_path, capsys):
    """`task states:` counts the WHOLE store, which is right - a throttled
    stream's backlog is the operator's cue to re-open it. On its own, though,
    it overstates what the run will do: with curated_c2 off STREAMS the live
    summary reads pending=5,278 while only 3,465 of those are servable. Both
    numbers have to appear, or the line quietly describes a queue 1,837 rows
    deeper than the one the fleet can touch.
    """
    root = _store_with(tmp_path, [
        ("pending", None, "synthesis"),
        ("pending", None, "curated_c2"),
        ("pending", None, "curated_c2"),
    ])
    # curated_c2 leaves the served list at RUNTIME now, when the ceiling guard
    # drops it - so the report is told what this run served rather than reading
    # the module constant, which no longer knows.
    actions_worker._finish(root, [_FakeProc()], ("gen",),
                           gen_died_early=False, final_push_ok=True,
                           streams=("synthesis", "transition"))
    out = capsys.readouterr().out
    assert "pending=3" in out, "the whole-store line must not shrink"
    assert "1 claimable in synthesis, transition" in out
    assert "2 pending in streams this run did not serve" in out


def test_the_backlog_report_covers_every_plannable_stream(tmp_path, capsys):
    """ALL_STREAMS must track the planner, not a copy of it. If a new stream
    were added and this list were hand-maintained, the report would silently
    stop counting its backlog - failing exactly where it is needed."""
    from tuned.data.tasks import PLANNABLE_STREAMS

    assert set(actions_worker.ALL_STREAMS) == set(PLANNABLE_STREAMS)
    assert set(actions_worker.STREAMS) <= set(actions_worker.ALL_STREAMS)


def test_the_servable_line_is_absent_when_every_stream_is_served(tmp_path, capsys):
    """It exists to explain a DIVERGENCE. With nothing throttled there is
    none, and a second count saying the same as the first is noise in the
    one readout an operator actually reads."""
    root = _store_with(tmp_path, [("pending", None, "synthesis")])
    actions_worker._finish(root, [_FakeProc()], ("gen",),
                           gen_died_early=False, final_push_ok=True)
    assert "does not serve" not in capsys.readouterr().out


def test_logs_are_scoped_per_run_so_the_baton_stops_overwriting_them(monkeypatch):
    """logs/ is staged into the baton but never restored, so every run began
    with an empty gen.log and its push REPLACED the previous run copy at the
    repo tip - the per-job logs README points the operator at only ever
    described the last ~5.25 h.
    """
    monkeypatch.setenv("GITHUB_RUN_ID", "1234567")
    assert actions_worker._run_scope() == "1234567"
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    assert actions_worker._run_scope() == "local"


# --------------------------------------------------------------------------
# The baton fence. The one-generator invariant rested entirely on the Actions
# concurrency group: upload_folder is unconditional last-writer-wins, and a
# second holder rewound the build silently - visible only as task counts
# going DOWN.
# --------------------------------------------------------------------------


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


class _Conflict(Exception):
    def __init__(self):
        super().__init__("412 Client Error: Precondition Failed")
        self.response = _Response(412)


class _RecordingApi:
    def __init__(self, head="head-sha", conflict=False, exists=False):
        self.head = head
        self.conflict = conflict
        self.exists = exists
        self.uploads = []

    def dataset_info(self, repo_id):
        return type("Info", (), {"sha": self.head})()

    def create_repo(self, *a, **k):
        return None

    def file_exists(self, repo_id, filename, repo_type=None):
        return self.exists

    def upload_folder(self, **kw):
        self.uploads.append(kw)
        if self.conflict:
            raise _Conflict()
        return type("Commit", (), {"oid": "new-sha"})()


_REAL_BUNDLE = actions_worker.Bundle


def _bundle_with(api):
    bundle = _REAL_BUNDLE.__new__(_REAL_BUNDLE)
    bundle.repo_id = "u/r"
    bundle.api = api
    bundle.head = None
    return bundle


def test_the_push_declares_the_revision_it_pulled_as_its_parent(tmp_path, monkeypatch):
    api = _RecordingApi(head="pulled-sha")
    bundle = _bundle_with(api)
    monkeypatch.setattr(
        actions_worker, "snapshot_download", lambda *a, **k: str(tmp_path), raising=False
    )
    import huggingface_hub

    seen = {}

    def _fake_snapshot(repo_id, **kw):
        seen.update(kw)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot)
    bundle.pull(tmp_path)
    # the tree we restore IS the revision we will push against
    assert seen["revision"] == "pulled-sha"

    bundle.push(tmp_path, "checkpoint")
    assert api.uploads[0]["parent_commit"] == "pulled-sha"
    # successive pushes in one job chain onto our own commit, not the pull
    bundle.push(tmp_path, "second")
    assert api.uploads[1]["parent_commit"] == "new-sha"


def test_a_stolen_baton_fails_the_job_instead_of_being_retried(tmp_path, capsys, monkeypatch):
    """A 412 says our parent is no longer the head. Retrying it would either
    fail identically or - worse - succeed against a moved parent and rewind
    the other holder's work, so it must not be swallowed by the best-effort
    push path."""
    monkeypatch.setattr(actions_worker, "stage_bundle", lambda r, s, db=True: s)
    bundle = _bundle_with(_RecordingApi(conflict=True))
    bundle.head = "stale"
    with pytest.raises(Exception, match="412"):
        actions_worker._push(
            bundle, tmp_path, tmp_path, "checkpoint", attempts=3,
            sleep=lambda s: None,
        )
    assert "BATON STOLEN" in capsys.readouterr().out


def test_a_transient_failure_is_still_retried(tmp_path, monkeypatch):
    # the fence must not turn ordinary 5xx flakiness into a hard failure
    class _Flaky(_RecordingApi):
        def __init__(self):
            super().__init__()
            self.left = 2

        def upload_folder(self, **kw):
            if self.left:
                self.left -= 1
                raise RuntimeError("HF 503")
            return super().upload_folder(**kw)

    monkeypatch.setattr(actions_worker, "stage_bundle", lambda r, s, db=True: s)
    bundle = _bundle_with(_Flaky())
    assert actions_worker._push(
        bundle, tmp_path, tmp_path, "checkpoint", attempts=3, sleep=lambda s: None
    ) is True


def test_seed_push_refuses_to_clobber_a_remote_that_owns_the_baton(tmp_path, capsys, monkeypatch):
    """The handoff is one-time, and re-running it overwrites the remote with
    this machine's stale copy. README warned about it in prose while the code
    did nothing."""
    api = _RecordingApi(exists=True)

    monkeypatch.setattr(actions_worker, "Bundle", lambda repo: _bundle_with(api))
    monkeypatch.setattr(
        actions_worker, "stage_bundle", lambda r, s, db=True: s
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("x: 1", encoding="utf-8")

    class _Cfg:
        class build:
            workdir = str(tmp_path / "build")

    monkeypatch.setitem(
        __import__("sys").modules, "_", type("m", (), {})()
    )
    import tuned.data.config as data_config
    import tuned.data.paths as data_paths

    monkeypatch.setattr(data_config, "load_build_config", lambda p: _Cfg())
    monkeypatch.setattr(
        data_paths, "build_paths",
        lambda w: type("P", (), {"ensure": lambda self: type("R", (), {"root": tmp_path})()})(),
    )

    rc = actions_worker.main(
        ["--phase", "seed-push", "--hf-repo", "u/r", "--config", str(cfg_path)]
    )
    assert rc == 3
    assert "REFUSING" in capsys.readouterr().out
    assert api.uploads == []  # nothing was overwritten


def test_the_ship_path_arms_the_teacher_cut():
    """(b) of P1.1, and the only place it is armed.

    verify's filters default OFF so an operator re-gating between waves cannot
    demote the corpus by accident - which means the corpus that actually
    leaves this machine is only one teacher at current prompts if THIS chain
    says so. Both flags, on the verify step, with or without an index.
    """
    for argvs in (
        actions_worker.assemble_argvs("cfg.yaml"),
        actions_worker.assemble_argvs("cfg.yaml", citation_index=Path("x/i.txt")),
        actions_worker.assemble_argvs("cfg.yaml", streams=["replay"], out_dir=Path("w/out")),
    ):
        verify = argvs[0]
        assert verify[3] == "tuned.data.verify"
        assert "--require-generator" in verify
        assert "--require-current-prompt" in verify
        # And swept over the only state that can ship. Not a filter on the
        # cut - a row this never looks at is a row decontaminate's
        # `state = 'accepted'` select cannot read either.
        assert verify[verify.index("--state") + 1] == "accepted"


def test_the_ship_paths_verify_state_is_the_state_decontaminate_reads():
    """Two spellings of "which rows ship" that must not drift apart.

    If the chain re-gated a state decontaminate does not select, it would
    spend the sweep on rows that cannot ship; if decontaminate selected a
    state the chain does not re-gate, rows would ship whose citation gate was
    never re-run with the real index - which is the failure verify.py exists
    to make impossible.
    """
    import inspect

    from tuned.data.decontaminate import generated_rows

    shipped = inspect.signature(generated_rows).parameters["state"].default
    verify = actions_worker.assemble_argvs("cfg.yaml")[0]
    assert verify[verify.index("--state") + 1] == shipped


def test_the_chain_fetches_exactly_the_eval_sets_decontaminate_demands():
    """`--kind hf` with no source list snapshots all six HF_SOURCES - the
    three eval corpora plus three full-text corpus inputs (predex,
    tathyanyaya, injudgements) that belong to seeds/select, phases which run
    on the operator's machine and never in this job.

    The keys are derived from EVAL_SETS, so this asserts the derivation is
    legal: every eval key must also be an acquire source, or argparse rejects
    the whole call with exit 2, no eval set lands, and decontaminate refuses
    the run.
    """
    from tuned.data.acquire import HF_SOURCES
    from tuned.data.eval_sets import EVAL_SETS

    assert set(EVAL_SETS) <= set(HF_SOURCES), (
        f"eval sets with no acquire source: {sorted(set(EVAL_SETS) - set(HF_SOURCES))}"
    )
    assert set(EVAL_SETS) < set(HF_SOURCES), "nothing left to skip - is the derivation still worth it?"
    src = Path(actions_worker.__file__).read_text(encoding="utf-8")
    assert "for key in sorted(EVAL_SETS)" in src, "the source list must stay derived, not literal"


def test_the_worker_releases_its_leases_only_after_the_children_are_dead():
    """Ordering is the whole safety argument.

    A lease is what stops two workers running one task, so clearing one while
    a child still holds it is exactly the corruption the lease exists to
    prevent. release_claims() therefore has to sit AFTER _stop() and after the
    pumps are joined - and before the end-of-job push, or the released stamps
    never leave this host.
    """
    import inspect

    src = inspect.getsource(actions_worker.run_worker)
    stop = src.index("_stop(procs)")
    release = src.index("release_claims()")
    push = src.index('"end-of-job checkpoint"')
    assert stop < release < push, "release_claims must run after _stop and before the final push"


def test_the_assembly_no_longer_waits_out_a_whole_lease_window():
    """900 was DEFAULT_LEASE_S: this job was paying, every dispatch, for the
    lease release the worker had not been written to do yet. With the worker
    releasing its own claims the expected count here is zero, so what is left
    to absorb is a baton snapshotted mid-flight - stamps already most of a
    lease old."""
    import inspect

    from tuned.data.store import DEFAULT_LEASE_S

    src = inspect.getsource(actions_worker.run_assemble)
    assert "waited < 120" in src
    assert f"waited < {DEFAULT_LEASE_S}" not in src
    assert 120 < DEFAULT_LEASE_S


def test_a_lease_that_outlives_the_wait_stops_the_chain_by_name():
    """Falling through was the old behaviour and it is the wrong one twice
    over: verify would refuse anyway, mid-chain, and a lease still live after
    the worker's release means another host is holding this baton - which the
    data-build concurrency group is supposed to make impossible. Assembling a
    corpus underneath it is worse than not assembling one."""
    import inspect

    src = inspect.getsource(actions_worker.run_assemble)
    refusal = src.index("REFUSING")
    acquire = src.index("acquire eval sets")
    assert refusal < acquire, "the refusal must precede any chain work"
    assert "return 5" in src[refusal:acquire]
    # distinct from the audit-collapse refusal, so a red run says which one
    assert "rc = 4" in src


def test_the_pull_fetches_only_what_the_restore_will_land():
    """`logs/` and `out/` live in the baton repo and neither is ever restored,
    so downloading them was bandwidth and runner disk spent to be ignored -
    and out/ gains a full set of assembly artifacts per dispatch.

    Both the download filter and the restore walk read RESTORE_SUBS, because
    two lists drift into either a sub that is fetched and ignored or - far
    worse - one that is restored and was never fetched.
    """
    import inspect

    from huggingface_hub.utils import filter_repo_objects

    assert "RESTORE_SUBS" in inspect.getsource(actions_worker.restore_bundle)
    assert "RESTORE_SUBS" in inspect.getsource(actions_worker.Bundle.pull)

    patterns = [f"{sub}/**" for sub in actions_worker.RESTORE_SUBS]
    repo = [
        "state/law_v1.sqlite3", "raw/gen/2026-08-29/gen.ndjson",
        "streams/replay.jsonl", str(actions_worker.INDEX_RELPATH).replace("\\", "/"),
        "logs/run-1/gen.log", "out/assemble.json", "out/law_v1_train.jsonl",
    ]
    kept = set(filter_repo_objects(repo, allow_patterns=patterns))
    assert "logs/run-1/gen.log" not in kept
    assert not any(p.startswith("out/") for p in kept)
    # everything restore_bundle would look for still arrives
    assert kept == {p for p in repo if p.split("/")[0] in actions_worker.RESTORE_SUBS}


def test_the_pulled_copy_is_dropped_once_it_has_been_restored(tmp_path, monkeypatch):
    """The pull writes a second whole copy of the working state onto the
    runner and restore_bundle copies it into place; keeping it doubles the
    baton's footprint for the rest of the job."""
    root = tmp_path / "build"
    root.mkdir(parents=True)

    class _FakeBundle:
        def pull(self, dest):
            (dest / "streams").mkdir(parents=True)
            (dest / "streams" / "replay.jsonl").write_text('{"row":1}\n', encoding="utf-8")
            return dest

    actions_worker.pull_and_restore(_FakeBundle(), root)
    assert (root / "streams" / "replay.jsonl").read_text(encoding="utf-8") == '{"row":1}\n'
    assert not (root.parent / "bundle_in").exists(), "the pulled copy is still on disk"


def test_the_out_upload_deletes_only_because_it_is_scoped_to_out(tmp_path):
    """delete_patterns is resolved RELATIVE TO path_in_repo. With
    path_in_repo="out" it clears the previous assembly's artifacts; without
    it, the same ["**"] means the whole repo - which is the baton. The two
    kwargs are one decision, so replace_dir's signature makes the scope
    non-optional and this drives the real call to prove they travel together.
    """
    api = _RecordingApi(head="pulled-sha")
    bundle = _bundle_with(api)
    bundle.head = "pulled-sha"

    bundle.replace_dir(tmp_path, "out", "assembly artifacts")

    kw = api.uploads[-1]
    assert kw["path_in_repo"] == "out"
    assert kw["delete_patterns"] == ["**"]
    # and nowhere else in the file may pass it - a second, unscoped use is
    # the failure this test exists for. File-scope, so it cannot be driven.
    whole = Path(actions_worker.__file__).read_text(encoding="utf-8")
    assert whole.count("delete_patterns=") == 1


def test_the_out_upload_fences_and_hands_the_head_on(tmp_path):
    """The bug this replaces: a bare api.upload_folder for out/ still makes a
    commit, so the remote head moved while bundle.head did not, and the very
    next push - the post-assembly checkpoint carrying the re-gate results and
    the off_teacher demotions - was refused as a 412 and re-raised as BATON
    STOLEN. Every dispatch ended in a false alarm and lost its DB write.

    So: the artifact upload must declare the head it saw, and must hand the
    commit it made to whatever pushes next.
    """
    api = _RecordingApi(head="pulled-sha")
    bundle = _bundle_with(api)
    bundle.head = "pulled-sha"

    bundle.replace_dir(tmp_path, "out", "assembly artifacts")
    assert api.uploads[-1]["parent_commit"] == "pulled-sha", "the artifact upload must fence too"
    assert bundle.head == "new-sha", "it must adopt the commit it just made"

    bundle.push(tmp_path, "post-assembly checkpoint")
    assert api.uploads[-1]["parent_commit"] == "new-sha", (
        "the checkpoint must fence against the artifact commit, not the pull - "
        "this is the 412 that made every assemble dispatch cry BATON STOLEN"
    )


def test_a_db_less_stage_keeps_everything_else(tmp_path):
    """The DB is a ~565 MB VACUUMed file rewritten whole and stored as a new
    blob per push - essentially all of this repo's growth. The appends beside
    it are cheap and must keep their own cadence."""
    root = tmp_path / "build"
    _tiny_db(root / actions_worker.DB_RELPATH, "baton").close()
    (root / "streams").mkdir()
    (root / "streams" / "replay.jsonl").write_text('{"row":1}\n', encoding="utf-8")
    (root / "raw" / "gen").mkdir(parents=True)
    (root / "raw" / "gen" / "gen.ndjson").write_text('{"kind":"generation"}\n', encoding="utf-8")

    light = actions_worker.stage_bundle(root, tmp_path / "s1", db=False)
    assert not (light / actions_worker.DB_RELPATH).exists()
    assert (light / "streams" / "replay.jsonl").is_file()
    assert (light / "raw" / "gen" / "gen.ndjson").is_file()

    full = actions_worker.stage_bundle(root, tmp_path / "s2")
    assert (full / actions_worker.DB_RELPATH).is_file()


def test_a_db_less_push_does_not_remove_the_database_from_the_remote():
    """The cheap variant only works because Bundle.push is a plain
    upload_folder with no delete_patterns: a staging tree without a DB leaves
    the last one pushed in place. If a delete ever appears on THAT call, a
    raw-only checkpoint would wipe the store off the baton."""
    import inspect

    src = inspect.getsource(actions_worker.Bundle.push)
    assert "delete_patterns" not in src, (
        "Bundle.push must not delete: a db=False checkpoint would erase the store"
    )


def test_the_database_rides_a_slower_lane_and_the_final_push_always_carries_it():
    import inspect

    parsed = actions_worker.main_parser().parse_args(["--phase", "worker", "--hf-repo", "u/r"])
    assert parsed.push_every == 900
    assert parsed.db_every == 3600
    assert parsed.db_every > parsed.push_every, "a DB cadence at or under the append cadence buys nothing"

    src = inspect.getsource(actions_worker.run_worker)
    # the periodic push decides; the end-of-job push never does
    assert "db=with_db" in src
    final = src[src.index('"end-of-job checkpoint"') - 200:]
    assert "db=False" not in final.split("_finish")[0]


def test_the_db_clock_only_advances_on_a_push_that_landed():
    """Otherwise a failed hourly checkpoint would push the next attempt an
    hour further out, and the crash-loss bound --db-every names would quietly
    stop being true."""
    import inspect

    src = inspect.getsource(actions_worker.run_worker)
    advance = src.index("next_db_push = time.monotonic() + args.db_every", src.index("with_db ="))
    failure = src.index("push_failures += 1")
    assert advance < failure, "the advance must sit in the success branch"
    assert "if with_db:" in src[advance - 120:advance]


def test_the_reopen_on_empty_list_is_the_free_parks_and_no_others():
    """Every member is a park that costs NOTHING to retry, and that is the
    rule rather than a hand-kept list.

    gen_unroutable and format_parked are POOL failures - no key, every model
    down, soft-gate attempts exhausted - which a fleet or policy change makes
    valid again. off_teacher is the 2026-08-30 one-teacher cut: the row must
    be bought again from the current teacher, and nothing else recovers it,
    because it is invisible to the queue (not claimable) while still counting
    as live to the planner (not terminally dead), so a re-plan inserts
    nothing.

    judge_error is reopenable by hand and must NEVER be automatic: its
    attempts RESET on re-open, so a state that comes back with a full budget
    every time the queue empties is an unbounded re-spend loop, not a
    recovery. stale_prompt is in TERMINALLY_DEAD - generate.py re-parks it
    before any render, so re-opening it 6x a day churns the store to arrive
    back where it started.
    """
    from tuned.data.tasks import REOPEN_STATES, TERMINALLY_DEAD

    assert actions_worker.REOPEN_ON_EMPTY == (
        "gen_unroutable", "format_parked", "off_teacher",
    )
    # DERIVED, not a second copy of the list: the members are exactly the
    # re-openable states that go back to the GENERATOR and can still leave
    # under their own power. Re-opening to "judging" (judge_error,
    # judge_unroutable) resets attempts already spent on a good answer, and a
    # TERMINALLY_DEAD state re-parks before any render. If someone adds a
    # state to REOPEN_STATES that satisfies both, this fails until they
    # decide about it deliberately.
    expected = tuple(
        s for s, back_to in REOPEN_STATES.items()
        if back_to == "pending" and s not in TERMINALLY_DEAD
    )
    assert set(actions_worker.REOPEN_ON_EMPTY) == set(expected)
    assert "judge_error" not in actions_worker.REOPEN_ON_EMPTY
    assert "stale_prompt" not in actions_worker.REOPEN_ON_EMPTY


def test_off_teacher_is_unrecoverable_without_the_automatic_reopen():
    """The demotion is a one-way door unless REOPEN_ON_EMPTY names it.

    P1.1 says the demoted rows "must regenerate". They cannot reach a worker
    on their own: off_teacher is not claimable, so the queue reads empty; and
    it is not terminally dead, so _existing_in_queue counts those rows as
    live and the planner reports "already at target" without inserting. This
    pins both halves of that trap, so a future edit to either list cannot
    quietly re-close the door.
    """
    from tuned.data.tasks import TERMINALLY_DEAD

    assert "off_teacher" not in actions_worker.CLAIMABLE_STATES
    assert "off_teacher" not in TERMINALLY_DEAD
    assert "off_teacher" in actions_worker.REOPEN_ON_EMPTY


def _task_db(path, rows):
    """(state, stream) rows in the shape run_worker reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE task (state TEXT, stream TEXT)")
    conn.executemany("INSERT INTO task VALUES (?, ?)", rows)
    conn.commit()
    conn.close()
    return path


def test_pending_work_in_an_UNSERVED_stream_does_not_count_as_claimable(tmp_path):
    """Throttling a stream must not also disable the empty-queue recovery.

    Dropping a stream from STREAMS leaves its planned tasks pending forever -
    that is the point, it is what makes the throttle reversible. But the
    queue-empty guard reads a whole-store `SELECT state, COUNT(*)`, so those
    permanently-pending rows would read as work the fleet is about to do.
    Nothing would ever be re-opened and every run would sit its full ~5 h
    claiming nothing, which is the exact stall the guard was added to end.
    """
    db = _task_db(tmp_path / "s.sqlite3", [("pending", "curated_c2")])
    assert actions_worker._claimable_in(db, ("synthesis", "transition")) == 0
    assert actions_worker._claimable_in(db, ("synthesis", "curated_c2")) == 1


def test_the_reported_task_states_still_cover_every_stream(tmp_path):
    """Claimability narrows to the served streams; VISIBILITY must not.

    A throttled stream's backlog is the operator's cue to re-open it, so it
    has to keep appearing in the job summary even though no run will touch
    it."""
    db = _task_db(
        tmp_path / "s.sqlite3",
        [("pending", "curated_c2"), ("accepted", "synthesis")],
    )
    assert actions_worker._task_counts(db) == {"pending": 1, "accepted": 1}


def test_the_reopen_runs_only_when_there_is_nothing_left_to_claim():
    """Unconditionally, it is a churn loop against a pool gap that is usually
    still there - re-open, fail to route, park, re-open - six times a day. On
    the empty branch it is the last thing tried before declaring the queue
    dead."""
    import inspect

    src = inspect.getsource(actions_worker.run_worker)
    guard = src.index("if counts is not None and not servable:")
    reopen = src.index("REOPEN_ON_EMPTY", guard)
    spawn = src.index("subprocess.Popen")
    assert guard < reopen < spawn, "the re-open must sit inside the empty branch, before the children"
    assert src.count("REOPEN_ON_EMPTY") == 3  # the print, the argv, the report


def test_an_empty_queue_skips_both_children_and_says_so_in_the_job_summary(tmp_path, monkeypatch, capsys):
    """5.25 h of two processes polling an empty queue, a pull, a push and a
    cron slot, to report what the first SELECT already knew."""
    root = tmp_path / "build"
    _tiny_db(root / actions_worker.DB_RELPATH, "x").close()
    conn = sqlite3.connect(root / actions_worker.DB_RELPATH)
    conn.execute("CREATE TABLE task (state TEXT)")
    conn.executemany("INSERT INTO task VALUES (?)", [("rejected",), ("stale_prompt",)])
    conn.commit()
    conn.close()

    ran = []

    def _run(argv, **kwargs):
        ran.append(argv)
        return type("R", (), {"returncode": 0})()

    def _popen(*args, **kwargs):
        pytest.fail("a child was spawned on an empty queue")

    monkeypatch.setattr(actions_worker, "pull_and_restore", lambda b, r: None)
    monkeypatch.setattr(actions_worker, "stage_bundle", lambda r, s, db=True: s)
    monkeypatch.setattr(actions_worker.subprocess, "run", _run)
    monkeypatch.setattr(actions_worker.subprocess, "Popen", _popen)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    pushed = []

    class _Bundle:
        repo_id = "u/r"

        def push(self, staging, message):
            pushed.append(message)

    args = actions_worker.main_parser().parse_args(["--phase", "worker", "--hf-repo", "u/r"])
    assert actions_worker.run_worker(args, root, _Bundle()) == 0
    assert pushed, "the re-open writes task states; they have to leave the host"
    assert "QUEUE EMPTY" in summary.read_text(encoding="utf-8")
    assert any("--reopen" in a for argv in ran for a in argv), "the re-open was never tried"


def _worker_workflow() -> dict:
    import yaml

    path = Path(__file__).parent.parent / ".github" / "workflows" / "data-worker.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_the_cron_period_is_under_the_cycle_it_schedules():
    """The point of the change: at a period longer than one run, the machine
    idles. The data-build group holds one running and one pending, so any
    period below the cycle chains back to back.

    The cycle is --minutes plus the pull, the pushes and setup. If --minutes
    ever grows past the period this test is what says the schedule no longer
    makes sense.
    """
    workflow = _worker_workflow()
    crons = [entry["cron"] for entry in workflow[True]["schedule"]]
    assert crons == ["17 */4 * * *"]
    period_min = 4 * 60

    parsed = actions_worker.main_parser().parse_args(["--phase", "worker", "--hf-repo", "u/r"])
    cycle_min = parsed.minutes  # the floor; the pull and pushes add ~10-15
    assert period_min < cycle_min, (
        "a period at or above the run length leaves the machine idle between runs"
    )
    # and the queued run must start well before the NEXT trigger, or the
    # group replaces a pending run - possibly an operator's assemble dispatch
    assert cycle_min - period_min >= 60


def test_the_job_timeout_still_leaves_the_deadline_room_to_stop_cleanly():
    """--minutes is a clean stop that pushes; timeout-minutes is a kill that
    loses everything since the last checkpoint. Raising --minutes toward the
    timeout - the change deliberately NOT made with the cron - is what this
    guards."""
    workflow = _worker_workflow()
    timeout = workflow["jobs"]["worker"]["timeout-minutes"]
    parsed = actions_worker.main_parser().parse_args(["--phase", "worker", "--hf-repo", "u/r"])
    assert parsed.minutes == 315
    assert timeout - parsed.minutes >= 30, "not enough room for the baton pull and the final push"


# --------------------------------------------------------------------------
# --phase plan: the only operator path to the remote queue.
# --------------------------------------------------------------------------

def _plan_env(tmp_path, monkeypatch, *, leases=0, planner_rc=0):
    """Wire main() far enough to reach run_plan without touching a real baton."""
    api = _RecordingApi()
    monkeypatch.setattr(actions_worker, "Bundle", lambda repo: _bundle_with(api))
    monkeypatch.setattr(actions_worker, "pull_and_restore", lambda b, r: None)

    calls: list[list[str]] = []

    def _run(argv, *a, **k):
        calls.append(list(argv))
        rc = planner_rc if "tuned.data.tasks" in argv else 0
        return type("CP", (), {"returncode": rc})()

    monkeypatch.setattr(actions_worker.subprocess, "run", _run)

    pushed: list[str] = []
    monkeypatch.setattr(
        actions_worker, "_push",
        lambda bundle, root, staging, message, **kw: pushed.append(message) or True,
    )

    import tuned.data.store as data_store
    import tuned.data.verify as data_verify

    class _Store:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(data_store.Store, "open", staticmethod(lambda p: _Store()))
    monkeypatch.setattr(data_verify, "live_leases", lambda store: leases)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("x: 1", encoding="utf-8")

    class _Cfg:
        class build:
            workdir = str(tmp_path / "build")

    import tuned.data.config as data_config
    import tuned.data.paths as data_paths

    monkeypatch.setattr(data_config, "load_build_config", lambda p: _Cfg())
    monkeypatch.setattr(
        data_paths, "build_paths",
        lambda w: type("P", (), {"ensure": lambda self: type("R", (), {"root": tmp_path})()})(),
    )
    return cfg_path, calls, pushed


def test_plan_refuses_without_an_explicit_mix(tmp_path, monkeypatch, capsys):
    """The default mix under-fills SILENTLY, so the phase has no default.

    tasks.SYNTHESIS_MIX sends 0.25 to statute_qa, whose seeds
    statute_section_eligible refuses in their entirety, and 0.00 to the parked
    drafting stream. A wave planned on it comes up a quarter short and says
    nothing - which is the failure this refusal exists to make loud.
    """
    cfg_path, calls, pushed = _plan_env(tmp_path, monkeypatch)
    rc = actions_worker.main(
        ["--phase", "plan", "--hf-repo", "u/r", "--config", str(cfg_path), "--plan-n", "9000"]
    )
    assert rc == 2
    out = capsys.readouterr().out
    assert "--plan-mix" in out and "statute_qa" in out
    # and it stopped BEFORE the baton moved
    assert pushed == [] and calls == []


def test_plan_refuses_without_a_target(tmp_path, monkeypatch, capsys):
    cfg_path, calls, pushed = _plan_env(tmp_path, monkeypatch)
    rc = actions_worker.main(
        ["--phase", "plan", "--hf-repo", "u/r", "--config", str(cfg_path),
         "--plan-mix", "irac_analysis=1.0"]
    )
    assert rc == 2
    assert "--plan-n" in capsys.readouterr().out
    assert pushed == []


def test_plan_hands_the_target_and_the_mix_to_the_planner_then_pushes(tmp_path, monkeypatch):
    """The whole point of the phase: widen the queue ON the baton, in one
    fenced round trip, rather than on a laptop racing the cron."""
    cfg_path, calls, pushed = _plan_env(tmp_path, monkeypatch)
    rc = actions_worker.main(
        ["--phase", "plan", "--hf-repo", "u/r", "--config", str(cfg_path),
         "--plan-stream", "synthesis", "--plan-n", "12000",
         "--plan-mix", "irac_analysis=0.55,summarization=0.45"]
    )
    assert rc == 0
    planner = next(c for c in calls if "tuned.data.tasks" in c)
    assert planner[planner.index("--stream") + 1] == "synthesis"
    assert planner[planner.index("--n") + 1] == "12000"
    assert planner[planner.index("--mix") + 1] == "irac_analysis=0.55,summarization=0.45"
    # reconcile runs first, exactly as the assemble phase does
    assert any("tuned.data.reconcile" in c for c in calls)
    assert len(pushed) == 1 and "12000" in pushed[0]


def test_plan_does_not_push_a_queue_the_planner_failed_to_build(tmp_path, monkeypatch, capsys):
    """A failed planner leaves the REMOTE holding the pre-plan queue, which is
    only true if nothing is pushed."""
    cfg_path, calls, pushed = _plan_env(tmp_path, monkeypatch, planner_rc=7)
    rc = actions_worker.main(
        ["--phase", "plan", "--hf-repo", "u/r", "--config", str(cfg_path),
         "--plan-n", "12000", "--plan-mix", "irac_analysis=1.0"]
    )
    assert rc == 7
    assert pushed == []
    assert "remote queue is unchanged" in capsys.readouterr().out


def test_plan_refuses_to_push_over_a_host_that_still_holds_the_baton(tmp_path, monkeypatch, capsys):
    """Same refusal as run_assemble and for the same reason: planning only
    INSERTs, but the push that carries it home rewrites the whole DB."""
    cfg_path, calls, pushed = _plan_env(tmp_path, monkeypatch, leases=2)
    monkeypatch.setattr(actions_worker.time, "sleep", lambda s: None)
    rc = actions_worker.main(
        ["--phase", "plan", "--hf-repo", "u/r", "--config", str(cfg_path),
         "--plan-n", "12000", "--plan-mix", "irac_analysis=1.0"]
    )
    assert rc == 5
    assert "REFUSING" in capsys.readouterr().out
    assert pushed == []
    assert not any("tuned.data.tasks" in c for c in calls)


def test_the_plan_workflow_serializes_against_the_worker(tmp_path):
    """The fence, read off the workflow rather than asserted about it.

    A plan that ran beside a worker would push a queue over the generations
    that worker had not checkpointed yet. `data-build` is what makes that
    impossible, and it only works if all three workflows name it.
    """
    import yaml

    root = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    groups = {
        name: yaml.safe_load((root / f"{name}.yml").read_text(encoding="utf-8"))
        for name in ("data-plan", "data-worker", "data-assemble")
    }
    assert {g["concurrency"]["group"] for g in groups.values()} == {"data-build"}
    assert all(g["concurrency"]["cancel-in-progress"] is False for g in groups.values())

    plan = groups["data-plan"]
    # Dispatch-only: a planner on a cron would widen the queue unattended.
    assert set(plan[True]) == {"workflow_dispatch"}
    inputs = plan[True]["workflow_dispatch"]["inputs"]
    assert all(inputs[k]["required"] for k in ("stream", "n", "mix"))


def test_plan_pins_the_wave_to_chosen_variants(tmp_path, monkeypatch):
    """The allowlist has to survive the ONE operator path to the queue.

    `--variant` landed on tuned.data.tasks, which runs on a laptop that must
    not touch the baton. This phase is the only way a wave reaches the remote,
    so an allowlist it cannot carry is an allowlist that cannot be used.
    Repeatable AND comma-separated, because workflow_dispatch has no list
    input type: the operator types one string.
    """
    cfg_path, calls, pushed = _plan_env(tmp_path, monkeypatch)
    rc = actions_worker.main(
        ["--phase", "plan", "--hf-repo", "u/r", "--config", str(cfg_path),
         "--plan-n", "1600", "--plan-mix", "irac_analysis=0.55,summarization=0.45",
         "--plan-variant", "gen_irac_analysis_v1, gen_irac_analysis_v2",
         "--plan-variant", "gen_summarization_v2"]
    )
    assert rc == 0
    planner = next(c for c in calls if "tuned.data.tasks" in c)
    named = [planner[i + 1] for i, a in enumerate(planner) if a == "--variant"]
    assert named == ["gen_irac_analysis_v1", "gen_irac_analysis_v2", "gen_summarization_v2"]


def test_plan_treats_a_blank_variant_input_as_the_full_pool(tmp_path, monkeypatch):
    """Load-bearing, and the same shape as the empty-skip-set guard in
    generate.py: a workflow input left blank arrives as "", and forwarding it
    would hand the planner an allowlist naming nothing."""
    cfg_path, calls, pushed = _plan_env(tmp_path, monkeypatch)
    rc = actions_worker.main(
        ["--phase", "plan", "--hf-repo", "u/r", "--config", str(cfg_path),
         "--plan-n", "1600", "--plan-mix", "irac_analysis=1.0",
         "--plan-variant", "  ,  "]
    )
    assert rc == 0
    planner = next(c for c in calls if "tuned.data.tasks" in c)
    assert "--variant" not in planner, "a blank input must plan on every template"


def test_the_plan_workflow_can_carry_an_allowlist(tmp_path):
    """Read off the workflow, because the CLI flag existing is not the same as
    the operator being able to reach it."""
    import yaml

    root = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    plan = yaml.safe_load((root / "data-plan.yml").read_text(encoding="utf-8"))
    inputs = plan[True]["workflow_dispatch"]["inputs"]
    assert "variants" in inputs, "no operator path to the variant allowlist"
    # Optional: the default wave draws from every paraphrase, and that stays
    # the default here too.
    assert not inputs["variants"].get("required", False)
    assert not inputs["variants"].get("default", "")
    run = plan["jobs"]["plan"]["steps"][-1]["run"]
    assert "--plan-variant" in run and "inputs.variants" in run


# --------------------------------------------------------------------------
# The curated_c2 ceiling guard
#
# curated_c2 was throttled out of STREAMS by hand on 2026-08-31 on the
# reasoning that draining its ~1,825 queued tasks would land ~2,149 accepted
# rows, within 26 of the one-way ceiling. That arithmetic multiplied PENDING
# tasks by 1.0. curated_c2's measured terminal accept rate is ~65%, so the
# drain actually lands ~1,190 accepted with ~450 effective rows of margin -
# and the hand throttle cost ~4,800 rows of assembled corpus to prevent a
# breach the queue could not cause. The policy below replaces the guess.


def test_curated_c2_is_served_while_the_corpus_is_far_from_the_ceiling():
    assert actions_worker.served_streams(
        ("synthesis", "transition", "curated_c2"),
        generated_curated=466, ceiling=2050,
    ) == ("synthesis", "transition", "curated_c2")


def test_curated_c2_is_dropped_once_the_corpus_is_inside_the_margin():
    """Generated rows cannot be dropped, so the last safe moment to stop is
    BEFORE the count that makes the corpus unassemblable, not at it."""
    assert actions_worker.served_streams(
        ("synthesis", "transition", "curated_c2"),
        generated_curated=2050 - actions_worker.CEILING_MARGIN_EFFECTIVE + 1,
        ceiling=2050,
    ) == ("synthesis", "transition")


def test_an_unmeasurable_ceiling_stops_curated_c2_rather_than_risking_it():
    """Fail CLOSED. Not serving the stream costs assembly delay and is undone
    by serving it again; serving it past the ceiling is permanent. The two
    errors are not the same size, so the unknown case takes the recoverable
    one."""
    assert actions_worker.served_streams(
        ("synthesis", "transition", "curated_c2"),
        generated_curated=0, ceiling=None,
    ) == ("synthesis", "transition")


def test_the_guard_touches_no_stream_but_curated_c2():
    """synthesis is the bucket the corpus is short of from BOTH sides; a
    ceiling on curated must never be a reason to stop generating it."""
    assert actions_worker.served_streams(
        ("synthesis", "transition"), generated_curated=10**6, ceiling=None,
    ) == ("synthesis", "transition")


def test_the_ceiling_state_counts_ACCEPTED_curated_rows_not_PENDING_ones():
    """THE regression test for the throttle that should not have happened.

    A store with a large curated_c2 backlog and few accepted rows is FAR from
    the ceiling: pending tasks are not corpus rows, and ~35% of them never
    will be. Reading the backlog as though it were already accepted is what
    made a stream with 450 effective rows of headroom look 26 rows away from
    a one-way door.
    """
    class _Store:
        def accepted_count(self, stream):
            return {"curated_c2": 100}.get(stream, 0)

    from tuned.data.config import load_build_config
    cfg = load_build_config(actions_worker.DEFAULT_CONFIG)
    generated_curated = actions_worker.measured_curated(_Store(), cfg.assembly)
    assert generated_curated < 200, (
        "100 accepted curated rows must not measure as thousands - the guard "
        "would then throttle a stream that has all of its headroom left"
    )


def test_the_ceiling_guard_reaches_the_children_that_do_the_claiming():
    """A guard the fleet never sees is not a guard.

    `served_streams` deciding to drop curated_c2 means nothing unless the
    decision becomes the children's `--stream` arguments - so child_argvs
    takes the served list rather than reading the module constant, which by
    design no longer knows what this run decided.
    """
    from tuned.data.judge import DEFAULT_AUDIT_SAMPLE

    argvs = actions_worker.child_argvs(
        "cfg.yaml", n_workers=8, audit_sample=DEFAULT_AUDIT_SAMPLE,
        streams=("synthesis", "transition"),
    )
    for argv in argvs:
        passed = {argv[i + 1] for i, a in enumerate(argv) if a == "--stream"}
        assert passed == {"synthesis", "transition"}
        assert "curated_c2" not in argv


def test_an_absent_build_leaves_the_ceiling_unmeasurable_rather_than_crashing(tmp_path):
    """The guard runs before the baton is unpacked on a cold checkout, and a
    run that dies measuring its own safety margin is worse than one that
    pauses a stream. Unmeasurable resolves to the recoverable error."""
    generated_curated, ceiling = actions_worker.ceiling_state(
        actions_worker.DEFAULT_CONFIG, tmp_path / "nothing-here"
    )
    assert ceiling is None
    assert actions_worker.served_streams(
        actions_worker.STREAMS,
        generated_curated=generated_curated, ceiling=ceiling,
    ) == ("synthesis", "transition")


def test_the_ceiling_is_measured_for_the_profile_the_chain_assembles():
    """A ceiling is a property of a PROFILE - the shares decide which pool
    binds. Two copies of the profile name would let the guard clear a stream
    against one target while the chain assembled another, and the corpus that
    then failed to assemble would be permanently unassemblable."""
    import inspect

    sig = inspect.signature(actions_worker.assemble_argvs)
    assert sig.parameters["profile"].default is actions_worker.PROFILE


def test_the_ceiling_guards_decision_is_left_where_the_baton_will_carry_it(
    tmp_path, monkeypatch
):
    """The guard's line has to outlive the run that printed it.

    GitHub publishes a job's log only when the job ENDS, and there is no log
    blob to fetch before that - so a 5h16m worker hides its single most
    consequential decision (which streams it is allowed to serve) for 5h16m.
    The job summary is no better: it renders when the STEP finishes, and the
    worker is one step. logs/ is staged onto the baton on every checkpoint,
    so the same line written there is readable within one checkpoint instead.
    """
    rc, _ = _run_worker(tmp_path, monkeypatch, [_FakeProc(), _FakeProc()])
    assert rc == 0
    written = (tmp_path / "build" / "logs" / actions_worker._run_scope()
               / "worker.log").read_text(encoding="utf-8")
    assert "ceiling guard:" in written


def test_the_reopen_decision_is_left_on_the_baton_too(tmp_path, monkeypatch):
    """Same blind spot as the ceiling guard, same fix.

    "no claimable work ... trying gen_unroutable" is the line that says a run
    found the queue dead and reached for the recovery - and it is printed in
    the first minute of a job whose log GitHub will not publish for another
    five hours. It belongs where a checkpoint can carry it.
    """
    root = tmp_path / "build"
    _tiny_db(root / actions_worker.DB_RELPATH, "x").close()
    conn = sqlite3.connect(root / actions_worker.DB_RELPATH)
    conn.execute("CREATE TABLE task (state TEXT)")
    conn.executemany("INSERT INTO task VALUES (?)", [("rejected",), ("stale_prompt",)])
    conn.commit()
    conn.close()

    monkeypatch.setattr(actions_worker, "pull_and_restore", lambda b, r: None)
    monkeypatch.setattr(actions_worker, "stage_bundle", lambda r, s, db=True: s)
    monkeypatch.setattr(actions_worker.subprocess, "run",
                        lambda argv, **kw: type("R", (), {"returncode": 0})())
    monkeypatch.setattr(actions_worker.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("a child was spawned on an empty queue"))

    class _Bundle:
        repo_id = "u/r"

        def push(self, staging, message):
            return None

    args = actions_worker.main_parser().parse_args(["--phase", "worker", "--hf-repo", "u/r"])
    assert actions_worker.run_worker(args, root, _Bundle()) == 0
    written = (root / "logs" / actions_worker._run_scope()
               / "worker.log").read_text(encoding="utf-8")
    assert "no claimable work:" in written


class _FakeStorageApi:
    """Enough of HfApi to answer "how much of the account quota is left".

    `boom` makes every reading raise, which is the case that must not be able
    to take a run down: a quota READING is a convenience, and the job's work
    does not depend on it.
    """

    def __init__(self, sizes, boom=False):
        self.sizes, self.boom = sizes, boom

    def list_models(self, author=None):
        return [type("R", (), {"id": i})() for i in self.sizes if "ckpt" in i]

    def list_datasets(self, author=None):
        return [type("R", (), {"id": i})() for i in self.sizes if "ckpt" not in i]

    def repo_info(self, repo_id, repo_type=None, expand=None):
        if self.boom:
            raise RuntimeError("hub down")
        return type("I", (), {"used_storage": self.sizes[repo_id]})()


class _StorageBundle(_FakeBundle):
    repo_id = "u/r"

    def __init__(self, api, **kw):
        super().__init__(**kw)
        self.api = api


def test_the_run_reports_how_much_of_the_storage_quota_is_left(tmp_path, monkeypatch):
    """The quota wall is a recurring, unattended failure - so it has to count
    itself down.

    The private-storage quota is ACCOUNT-WIDE, and the build walks into it
    every few days: the baton is an LFS repo that keeps every version of
    everything it has ever pushed. Twice now the wall has been found by an
    agent measuring it by hand, and the second measurement was 2.8x the first.
    A run that states its own headroom turns that into a number in every log,
    so the next person to look does not have to re-derive it.
    """
    api = _FakeStorageApi({"u/r": 44_000_000_000, "u/x-ckpt": 6_000_000_000})
    rc, _ = _run_worker(tmp_path, monkeypatch, [_FakeProc(), _FakeProc()],
                        bundle=_StorageBundle(api))
    assert rc == 0
    written = (tmp_path / "build" / "logs" / actions_worker._run_scope()
               / "worker.log").read_text(encoding="utf-8")
    assert "storage: 50.0 GB of 100 used, 50.0 GB headroom" in written


def test_the_storage_line_names_no_repo_because_the_actions_log_is_public(
    tmp_path, monkeypatch
):
    """`_run_log` prints to stdout, and this repo's Actions logs are public.

    The account being measured is full of PRIVATE repositories, and their
    names are the one thing in the reading that is not already public. Totals
    say everything an operator needs; the ids say nothing they need and leak
    what the account holds.
    """
    api = _FakeStorageApi({"u/r": 1_000_000_000, "u/secret-lane-ckpt": 2_000_000_000})
    rc, _ = _run_worker(tmp_path, monkeypatch, [_FakeProc(), _FakeProc()],
                        bundle=_StorageBundle(api))
    assert rc == 0
    written = (tmp_path / "build" / "logs" / actions_worker._run_scope()
               / "worker.log").read_text(encoding="utf-8")
    assert "storage:" in written
    assert "secret-lane-ckpt" not in written


def test_a_storage_reading_that_fails_does_not_fail_the_run(tmp_path, monkeypatch):
    """Same rule as _run_log: a run must not die because it could not narrate
    itself. The Hub being unreachable for a metadata call says nothing about
    whether this job can generate rows."""
    api = _FakeStorageApi({"u/r": 1_000_000_000}, boom=True)
    rc, _ = _run_worker(tmp_path, monkeypatch, [_FakeProc(), _FakeProc()],
                        bundle=_StorageBundle(api))
    assert rc == 0
    written = (tmp_path / "build" / "logs" / actions_worker._run_scope()
               / "worker.log").read_text(encoding="utf-8")
    assert "storage: reading unavailable" in written
