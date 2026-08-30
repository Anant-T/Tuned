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


def _run_assemble(tmp_path, monkeypatch, dispositions):
    from pipeline_fakes import temp_config

    cfg_path = temp_config(tmp_path)
    root = _assemble_root_with(tmp_path, dispositions)
    args = actions_worker.main_parser().parse_args(
        ["--phase", "assemble", "--hf-repo", "u/r", "--config", cfg_path]
    )
    recorder = _CallRecorder()
    monkeypatch.setattr(actions_worker.subprocess, "run", recorder)
    monkeypatch.setattr(actions_worker.time, "sleep", lambda s: None)
    rc = actions_worker.run_assemble(args, root, _FakeBundle())
    return rc, recorder.argvs


def test_run_assemble_refuses_the_push_step_on_a_collapsed_sample(tmp_path, monkeypatch, capsys):
    dispositions = ["judge:accept"] * 5 + ["judge:reject"] * 95
    rc, argvs = _run_assemble(tmp_path, monkeypatch, dispositions)
    assert rc != 0
    assert not any("tuned.data.push" in argv for argv in argvs)
    assert "AUDIT SAMPLE COLLAPSE" in capsys.readouterr().out


def test_run_assemble_pushes_when_the_sample_is_healthy(tmp_path, monkeypatch):
    dispositions = ["judge:accept"] * 90 + ["judge:reject"] * 10
    rc, argvs = _run_assemble(tmp_path, monkeypatch, dispositions)
    assert rc == 0
    assert any("tuned.data.push" in argv for argv in argvs)


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
        [(state, disposition, "synthesis") for state, disposition in rows],
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


def test_the_out_upload_deletes_only_because_it_is_scoped_to_out():
    """delete_patterns is resolved RELATIVE TO path_in_repo. With
    path_in_repo="out" it clears the previous assembly's artifacts; without
    it, the same ["**"] means the whole repo - which is the baton. The two
    kwargs are one decision, so they are pinned in one place.
    """
    import inspect

    src = inspect.getsource(actions_worker.run_assemble)
    call = src[src.index("out_dir = root / \"out\""):]
    call = call[:call.index(")\n", call.index("delete_patterns"))]
    assert 'path_in_repo="out"' in call
    assert 'delete_patterns=["**"]' in call
    # and nowhere else in the file may pass it - a second, unscoped use is
    # the failure this test exists for
    whole = Path(actions_worker.__file__).read_text(encoding="utf-8")
    assert whole.count("delete_patterns=") == 1


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
