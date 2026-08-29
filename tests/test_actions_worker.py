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
    monkeypatch.setattr(actions_worker, "stage_bundle", lambda r, s: s)
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
    root = tmp_path / "build"
    db = root / actions_worker.DB_RELPATH
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE task (state TEXT, disposition TEXT)")
    conn.executemany("INSERT INTO task VALUES (?, ?)", rows)
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
