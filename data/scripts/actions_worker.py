"""Remote build worker: run the generate+judge fleet on a CI host, state via HF.

The store is a local SQLite file and only ONE host may generate at a time
(the generator's rate bucket is account-level), so the build travels as a
baton: the whole ``data/build`` working state - a consistent DB snapshot,
the append-only raw NDJSON logs, and the stream files - lives in a PRIVATE
HuggingFace dataset repo, and whichever host holds the baton restores it,
works, and pushes it back. GitHub Actions chains ~5.5h jobs on a cron; the
workflow-level ``concurrency`` group is the fence between overlapping runs.

Phases:
  seed-push  upload the local data/build state to the bundle repo (the
             one-time handoff, run on the operator's machine)
  worker     restore bundle -> reconcile -> run generate + judge until the
             deadline, pushing raw+streams every --push-every seconds and
             the database every --db-every -> final push, always with the
             database
  assemble   restore bundle -> reconcile -> wait for leases to expire ->
             verify -> decontaminate -> dedupe -> split -> assemble ->
             stats -> push.py only if stats is green; the out/ tree is
             uploaded to the bundle repo either way

Auth is the HF_TOKEN env var (huggingface_hub reads it itself); provider
keys reach the children through the environment. Nothing here ever prints a
key value - CI logs on a public repo are public.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

# Imported rather than restated: the shaped filenames are a contract between
# this chain's shape step and its decontaminate step, and a second copy of
# the prefix would let them disagree silently - decontaminate would fall
# back to globbing the unshaped pools and the gates would go red for a
# reason nothing in this file mentions. The config path is here for the same
# reason: it moved once already in the 2026-08-28 restructure.
from tuned.data.paths import DEFAULT_CONFIG
from tuned.data.shape import SHAPED_PREFIX

STREAMS = ("synthesis", "transition", "curated_c2")
DB_RELPATH = Path("state") / "law_v1.sqlite3"
GRACE_S = 90  # SIGTERM -> this long -> SIGKILL, wide enough for a final fsync


def child_argvs(config: str, *, n_workers: int, audit_sample: float) -> list[list[str]]:
    """The two children: one generating process, one judging process.

    One of each is the store's intended safe topology - disjoint claim state
    pairs, disjoint raw append targets. NEVER two generators: they would
    share one raw file (jsonl.append_ndjson is single-writer) and double-run
    the per-process rate bucket into real 429s.
    """
    stream_args = [a for s in STREAMS for a in ("--stream", s)]
    return [
        [sys.executable, "-u", "-m", "tuned.data.generate", "--config", config,
         *stream_args, "--n-workers", str(n_workers), "--forever"],
        [sys.executable, "-u", "-m", "tuned.data.judge", "--config", config,
         *stream_args, "--n-workers", "4", "--forever",
         "--audit-sample", str(audit_sample)],
    ]


def assemble_argvs(
    config: str,
    citation_index: Path | None = None,
    streams: list[str] | None = None,
    out_dir: Path | None = None,
    profile: str = "v1.0-MVP",
) -> list[list[str]]:
    """The assembly chain, in order. stats is last and gates push.py.

    citation_index arms verify's existence half - without it every row ships
    citation-UNVERIFIED (verify warns loudly). The bundle carries the index
    when the operator has built it (tuned.data.citations --build).

    `streams` (the stream file STEMS, e.g. ["replay", "curated_c1"]) inserts
    the shape stage and points decontaminate at its output instead of the
    pools. Without it the pools ship whole, which is the documented escape
    and also what every pre-2026-08-29 run did - and what put three gates
    red, because the pools are sized for the FINISHED corpus and feeding
    them to a half-generated one guarantees a replay-dominated mix. The
    shaped names are a pure function of the stem, so they can be named here
    before shape has run.
    """
    # THE CUT IS ARMED HERE AND ONLY HERE. An ad-hoc `verify` run stays a
    # pure citation re-check; the corpus that leaves this machine is one
    # teacher at the prompt templates on disk (2026-08-30 ruling), so a row
    # from a retired provider or a superseded template is demoted out of the
    # shippable pool rather than blended into it silently. Expect the first
    # armed run to shrink the pool - that is the ruling, not a fault.
    #
    # --state accepted, because only an accepted row can ship. The sweep is
    # over latest_generations, and on a real store four fifths of that is
    # rejected, stale_prompt, pending or generating - states decontaminate's
    # `state = 'accepted'` select cannot reach, being re-gated so their gate
    # rows can be recomputed for nobody. `judging` is the one state this drops
    # that could ship LATER, and it cannot ship from THIS run: the fleet is
    # idle by here (run_assemble waits out the leases, and verify refuses to
    # start while one is live), so no row becomes accepted between this step
    # and decontaminate's read. It is swept by the next build, when it is.
    verify_step = [
        "verify", "--require-generator", "--require-current-prompt", "--state", "accepted",
    ] + (["--index", str(citation_index)] if citation_index else [])
    if streams:
        base = out_dir or Path("data/build/out")
        shaped = []
        for stem in streams:
            shaped += ["--in", str(base / f"{SHAPED_PREFIX}{stem}.jsonl")]
        head = [["shape", "--profile", profile], ["decontaminate", *shaped]]
    else:
        head = [["decontaminate"]]
    chain = [
        verify_step, *head, ["dedupe"], ["split"], ["assemble"],
        ["stats", "--profile", profile],
    ]
    return [
        [sys.executable, "-u", "-m", f"tuned.data.{step[0]}", "--config", config, *step[1:]]
        for step in chain
    ]


def snapshot_db(state_db: Path, dest: Path) -> None:
    """A consistent copy of a live WAL database, via VACUUM INTO.

    Never file-copy a live WAL db: the main file alone is a torn snapshot
    (committed pages still live in the -wal). VACUUM INTO reads through the
    WAL and writes a compact, self-contained file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)  # VACUUM INTO refuses an existing target
    conn = sqlite3.connect(f"file:{state_db.as_posix()}?mode=ro", uri=True)
    try:
        conn.execute("VACUUM INTO ?", (dest.as_posix(),))
    finally:
        conn.close()


INDEX_RELPATH = Path("corpus") / "citation_index.txt"

# The baton's WORKING state - the only part of the repo a new holder needs.
# One definition, used twice: restore_bundle walks these, and Bundle.pull
# turns them into the allow_patterns it downloads. The repo also carries
# `logs/` (staged for the operator to read, never restored) and `out/` (the
# assembly artifacts, which grow with every dispatch), and pulling those was
# hundreds of megabytes per run fetched so that restore_bundle could skip
# them. Two lists would drift into either a sub that is downloaded and
# ignored or - much worse - one that is restored and was never fetched.
RESTORE_SUBS = ("state", "raw", "streams", "corpus")


def stage_bundle(root: Path, staging: Path, *, db: bool = True) -> Path:
    """Copy everything the baton carries into a clean staging tree.

    The citation index rides along when it exists - ONE file, never the
    corpus dir (1.9 GB of source text the remote worker has no use for).

    `db=False` stages everything EXCEPT the database snapshot. The DB is by
    far the largest thing here and it is rewritten whole every time - a
    VACUUMed ~565 MB file, uploaded as a new blob on each checkpoint, is
    almost all of this repo's growth. Bundle.push passes no delete_patterns,
    so a staging tree without one simply leaves the last DB pushed in place
    on the remote, and the raw/stream appends still land at full cadence.

    SKIPPING IT ENTIRELY IS NOT AN OPTION, and was refuted three times in the
    audit before it stuck: reconcile_raw rebuilds generation and judgement
    rows from the raw log, but it restores NEITHER task state NOR gate_result.
    Without a DB, every row whose answer was already bought sits at `pending`
    and the next claim pays for it again. So the DB goes at its own, slower
    cadence, and the end-of-job push always sends one.
    """
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    if db:
        snapshot_db(root / DB_RELPATH, staging / DB_RELPATH)
    for sub in ("raw", "streams", "logs"):
        src = root / sub
        if src.is_dir():
            shutil.copytree(src, staging / sub)
    index = root / INDEX_RELPATH
    if index.is_file():
        (staging / INDEX_RELPATH).parent.mkdir(parents=True)
        shutil.copy2(index, staging / INDEX_RELPATH)
    return staging


def restore_bundle(bundle: Path, root: Path) -> None:
    """Land a pulled bundle in the workdir. Fresh-checkout semantics: the
    bundle's copy of a file wins, and any stale -wal/-shm beside the DB is
    dropped (the snapshot is self-contained; a leftover WAL from a previous
    life would be replayed over it)."""
    for sub in RESTORE_SUBS:
        src = bundle / sub
        if not src.is_dir():
            continue
        for f in sorted(p for p in src.rglob("*") if p.is_file()):
            dest = root / f.relative_to(bundle)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
    for suffix in ("-wal", "-shm"):
        (root / DB_RELPATH).with_name(DB_RELPATH.name + suffix).unlink(missing_ok=True)


def pull_and_restore(bundle: "Bundle", root: Path) -> None:
    """Land the baton in `root`, then delete the copy it was landed from.

    The pull writes a whole second copy of the working state onto the runner
    disk and restore_bundle then copies it into place; keeping it afterwards
    doubles the baton's footprint for the rest of the job, and every byte of
    it is either already in `root` or in a sub restore_bundle does not read.
    Both phases did these steps in the same order, so the drop lives here
    rather than being remembered separately in each of them.
    """
    pulled = bundle.pull(root.parent / "bundle_in")
    restore_bundle(pulled, root)
    # ignore_errors: failing to reclaim disk must not fail a job that has
    # already restored successfully.
    shutil.rmtree(pulled, ignore_errors=True)


class Bundle:
    """The HF dataset repo the baton lives in. Thin, injectable for tests."""

    def __init__(self, repo_id: str):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.api = HfApi()
        self.head: str | None = None  # the revision this holder pulled

    def ensure(self) -> None:
        self.api.create_repo(self.repo_id, repo_type="dataset", private=True, exist_ok=True)

    def pull(self, dest: Path) -> Path:
        from huggingface_hub import snapshot_download

        # Record the head we pulled and download THAT revision, so the tree we
        # restore and the parent we later push against are the same commit by
        # construction rather than by timing.
        self.head = self.api.dataset_info(self.repo_id).sha
        return Path(
            snapshot_download(
                self.repo_id, repo_type="dataset",
                revision=self.head, local_dir=dest,
                # Only what restore_bundle will actually land, from the one
                # list that decides it. `logs/` and `out/` live in this repo
                # too and neither is ever restored, so downloading them was
                # bandwidth and runner disk spent to be ignored - and `out/`
                # accumulates a full set of assembly artifacts per dispatch.
                allow_patterns=[f"{sub}/**" for sub in RESTORE_SUBS],
            )
        )

    def push(self, staging: Path, message: str) -> None:
        """Push the staged tree, refusing to overwrite a baton someone moved.

        The one-generator invariant rests entirely on the Actions concurrency
        group; nothing in the repo itself could detect a second holder, and
        upload_folder is unconditional last-writer-wins. A stale local
        seed-push, a dispatch from a branch whose concurrency block was edited,
        or a local run against data/build would silently REWIND the build -
        detectable only by noticing the task counts went down.

        parent_commit turns that into a 412 from the Hub. No new state and no
        generation counter: the Hub already knows what the head was.
        """
        info = self.api.upload_folder(
            repo_id=self.repo_id, repo_type="dataset",
            folder_path=str(staging), commit_message=message,
            parent_commit=getattr(self, "head", None),
        )
        # Track our own commit so successive pushes in one job chain correctly
        # (the second push's parent is the first push's commit, not the pull).
        oid = getattr(info, "oid", None)
        if oid:
            self.head = oid


def _pump(proc: subprocess.Popen, name: str, log_path: Path) -> threading.Thread:
    """Tee a child's output to our stdout and a log file, line by line.

    CI shows nothing a buffered pipe holds; a wedged child with a full pipe
    buffer deadlocks. One daemon thread per child drains it continuously.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def run():
        with log_path.open("a", encoding="utf-8") as log:
            for line in proc.stdout:  # type: ignore[union-attr]
                sys.stdout.write(f"[{name}] {line}")
                sys.stdout.flush()
                log.write(line)
                log.flush()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def _stop(procs: Sequence[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
    deadline = time.monotonic() + GRACE_S
    for p in procs:
        if p.poll() is None:
            try:
                p.wait(timeout=max(1.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


# States a future claim can still pick up. Anything else is terminal for this
# wave, so a queue holding only those has no work left for either child.
CLAIMABLE_STATES = ("pending", "generating", "judging", "judging_active")


def _run_scope() -> str:
    """One log directory per Actions run.

    logs/ is staged into the baton but never restored, so every run started
    with an empty logs/gen.log and its push REPLACED the previous run's copy
    at the repo tip - the per-job logs README points the operator at only ever
    described the last ~5.25 h."""
    return os.environ.get("GITHUB_RUN_ID", "local")


def _task_counts(state_db: Path) -> dict[str, int] | None:
    if not state_db.is_file():
        return None
    try:
        with sqlite3.connect(f"file:{state_db.as_posix()}?mode=ro", uri=True) as conn:
            return dict(conn.execute("SELECT state, COUNT(*) FROM task GROUP BY state"))
    except sqlite3.Error:
        return None


def _claimable(counts: dict[str, int]) -> int:
    return sum(counts.get(state, 0) for state in CLAIMABLE_STATES)


def _report(lines: Sequence[str]) -> None:
    """Say it on stdout AND in the Actions job summary.

    The summary is the only surface an operator sees without opening the run,
    so anything worth ending a job over belongs in both. Extracted from
    _finish when the queue-empty exit became a second thing that ends a run
    early and has to be exactly as visible as a crash.
    """
    text = "\n".join(lines)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except OSError as exc:
            print(f"step summary unavailable ({exc})")


# Parked states worth one automatic re-open when the queue has run dry, and
# the only two. Both are POOL failures rather than decisions about an answer:
# gen_unroutable is a wave that could not route at all (no key, every model
# down) and format_parked is a row that exhausted its soft-gate attempts,
# which a template or policy change makes valid again.
#
# NOT judge_error, though it is reopenable by hand: its attempts RESET on
# re-open (a judge park happens after a good generation exists, so the
# attempts were spent on the answer, not on discovering the pool was short),
# and a state that comes back with a full budget every time the queue empties
# is an unbounded re-spend loop rather than a recovery.
#
# NOT stale_prompt either: it is in TERMINALLY_DEAD. The row was planned
# against template bytes that no longer exist, so generate.py's guard re-parks
# it before any render - re-opening it 6x/day churns the store to arrive back
# where it started. Only a re-plan produces a usable row for that seed.
REOPEN_ON_EMPTY = ("gen_unroutable", "format_parked")


def _is_baton_conflict(exc: BaseException) -> bool:
    """A 412 from the Hub means our parent_commit is no longer the head."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status == 412 or "412" in f"{exc}" and "parent" in f"{exc}".lower()


PUSH_BACKOFF_S = (30, 90, 240)


def _push(
    bundle: Bundle,
    root: Path,
    staging: Path,
    message: str,
    *,
    attempts: int = 1,
    sleep=None,
    db: bool = True,
) -> bool:
    """Push the baton, optionally retrying. Returns whether it landed.

    Periodic pushes stay best-effort at attempts=1: a transient HF failure
    must not kill the fleet mid-run, because the next interval retries.

    The FINAL push is the one that cannot be best-effort, and it used to be
    the only push with no retry at all - the docstring above delegated
    recovery to "the final push" while that push was a single unguarded
    upload_folder. A transient 5xx there discarded everything generated since
    the last checkpoint AND failed the job with a traceback.

    stage_bundle runs ONCE, outside the retry loop: it VACUUMs a ~565 MB
    database and copies the raw tree, so re-staging per attempt would triple
    that work to re-send bytes that have not changed.

    `db=False` sends everything but the database - see stage_bundle. It is
    the caller's cadence decision, not a property of a push.
    """
    # Resolved at CALL time, not bound as a default: a default would capture
    # time.sleep at import and make the backoff unpatchable, which is a real
    # 2-minute stall in any test that exercises the retry path.
    sleep = sleep or time.sleep
    staged = stage_bundle(root, staging, db=db)
    for attempt in range(1, attempts + 1):
        try:
            bundle.push(staged, message)
            print(f"bundle pushed: {message}")
            return True
        except Exception as exc:  # noqa: BLE001
            if _is_baton_conflict(exc):
                # NOT retryable and NOT best-effort: another holder pushed
                # since we pulled, so our tree is built on a state that is no
                # longer the head. Retrying would either fail identically or,
                # worse, succeed against a moved parent and rewind their work.
                print(
                    "BATON STOLEN - another holder pushed since we pulled; "
                    "refusing to overwrite. Check for a second worker run or a "
                    "local run against data/build."
                )
                raise
            last = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                wait = PUSH_BACKOFF_S[min(attempt - 1, len(PUSH_BACKOFF_S) - 1)]
                print(f"bundle push failed ({last}) - retrying in {wait}s")
                sleep(wait)
    print(f"bundle push failed ({last}) - continuing")
    return False


def _tail(path: Path, lines: int = 20) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def _audit_readout():
    """Import data/scripts/audit_readout.py. That directory sits outside the
    `tuned` package, so it is only reachable by putting this file's own
    directory on sys.path first - shared by every call site below rather
    than each repeating the same two lines."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:  # called per push; sys.path is not a log
        sys.path.insert(0, here)
    import audit_readout

    return audit_readout


def _audit_summary(state_db: Path, since: str | None = None) -> list[str]:
    """The dual-judged sample's accept rate, computed where the operator will
    see it. audit_readout.py has been the documented ship gate since
    2026-08-29 while being invoked by no workflow at all, so the one number
    standing behind every audit-accepted row was machine-computed and
    machine-ignored."""
    if not state_db.is_file():
        return []
    try:
        audit_readout = _audit_readout()
        with sqlite3.connect(f"file:{state_db.as_posix()}?mode=ro", uri=True) as conn:
            return audit_readout.format_summary(
                audit_readout.summarize(conn, since=since)
            )
    except Exception as exc:  # noqa: BLE001 - a readout must never fail a run
        return [f"audit readout unavailable ({type(exc).__name__}: {exc})"]


# Two-sided 95% critical value. Not tuned for this pipeline - it is the
# conventional default for "is this number believable at all", the question
# the collapse refusal below is actually asking.
WILSON_Z = 1.96

# Below this many DECIDED rows, the sample is too small to say anything: a
# 10-row sample at 0% accepted still has a Wilson upper bound near 30%, so in
# practice it would rarely trip a 0.20 floor anyway - requiring 50 decided
# rows up front is the more legible statement of the same fact. The refusal
# below names "the sample is too small to judge" instead of leaning on an
# interval that happens to average out.
MIN_SAMPLE_FOR_COLLAPSE_CHECK = 50


def wilson_upper_bound(successes: int, n: int, *, z: float = WILSON_Z) -> float:
    """The Wilson score interval's UPPER bound for a binomial proportion.

    Solves |p_hat - p| = z*sqrt(p*(1-p)/n) for p directly, rather than the
    naive p_hat +/- z*sqrt(p_hat*(1-p_hat)/n) - the naive interval can
    undershoot 0 or overshoot 1 and is a poor approximation exactly where a
    collapsing accept rate lives (p_hat near 0, n in the tens rather than the
    thousands). Reading only the UPPER root is deliberate: the refusal below
    must be CERTAIN the batch has collapsed, so it asks "even in the best
    case this sample is consistent with, is the accept rate still below the
    floor?" - never the point estimate, which an unlucky small sample could
    dip under by chance alone.

        p_hat      = successes / n
        denom      = 1 + z^2/n
        center     = (p_hat + z^2/(2n)) / denom
        half_width = (z/denom) * sqrt(p_hat*(1-p_hat)/n + z^2/(4n^2))
        upper      = center + half_width

    n<=0 returns 1.0 (no evidence at all is not evidence of collapse) - the
    caller's MIN_SAMPLE_FOR_COLLAPSE_CHECK gate never lets n=0 reach here in
    practice, but the function stays total rather than raising on it.
    """
    if n <= 0:
        return 1.0
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    half_width = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
    return center + half_width


def _audit_collapse_refusal(
    state_db: Path, floor: float, since: str | None = None
) -> str | None:
    """None if the build may proceed to push; else the refusal message.

    Reads audit_readout.summarize() straight - the SAME sampled_decided /
    sampled_accepted the operator-facing readout prints - so this gate and
    the number the operator reads before publishing can never disagree. No
    state DB is not this function's refusal to make (an earlier step already
    failed the run in that case); it returns None rather than raising.

    `since` IS LOAD-BEARING, not a nicety. Without it the sample pools both
    judge regimes, and verify.py's armed cut makes that pooling ASYMMETRIC:
    it rewrites an off-teacher row's disposition out of `judge:%` when the
    row sits in accepted/judging, but a `rejected` row is a decision verify
    never touches - so the pre-audit ACCEPTS leave the population and the
    pre-audit REJECTS stay in it, depressing the rate this gate reads and
    making a FALSE refusal more likely, not less. routing.judge_mode_since
    is where the window comes from.
    """
    if not state_db.is_file():
        return None
    audit_readout = _audit_readout()
    with sqlite3.connect(f"file:{state_db.as_posix()}?mode=ro", uri=True) as conn:
        s = audit_readout.summarize(conn, since=since)
    decided, accepted = s["sampled_decided"], s["sampled_accepted"]
    if decided < MIN_SAMPLE_FOR_COLLAPSE_CHECK:
        return None
    upper = wilson_upper_bound(accepted, decided)
    if upper >= floor:
        return None
    return (
        f"AUDIT SAMPLE COLLAPSE - refusing to push. {accepted}/{decided} dual-judged "
        f"rows accepted since {since or 'the beginning of the store'}; the Wilson "
        f"upper bound on that rate is {upper:.1%}, below the "
        f"{floor:.0%} collapse floor (routing.judge_collapse_floor). This is not a "
        f"quality bar - it means the sample is consistent with almost nothing being "
        f"acceptable even in the best case. Check the judge fleet (provider outage, a "
        f"broken prompt template, a routing regression) before assuming the corpus "
        f"itself is bad; a build that is merely mediocre would not trip this floor."
    )


def _finish(
    root: Path,
    procs: Sequence[subprocess.Popen],
    names: Sequence[str],
    *,
    gen_died_early: bool,
    final_push_ok: bool,
) -> int:
    """Say what happened, everywhere the operator might look, and exit
    accordingly.

    run_worker used to `return 0` unconditionally, so NO build failure could
    fail the job or send a notification: a generator that SystemExit(2)'d at
    t~5s on a missing key left the judge polling an empty queue for 5.25 h and
    the run went green. That unconditional zero is also why the Actions
    failure mail - the only notification channel this build has - could never
    fire.
    """
    report: list[str] = []
    for name, proc in zip(names, procs):
        report.append(f"[{name}] exited rc={proc.returncode}")

    counts = _task_counts(root / DB_RELPATH)
    if counts is not None:
        report.append("task states: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        if not _claimable(counts):
            report.append(
                "QUEUE EMPTY - nothing left to claim; re-plan with "
                "`python -m tuned.data.tasks --config <cfg> --n <rows>`"
            )
    report += _audit_summary(root / DB_RELPATH)

    gen_log = _tail(root / "logs" / _run_scope() / "gen.log")
    if gen_log:
        report.append("--- last lines of gen.log ---")
        report += gen_log

    if gen_died_early:
        report.append("GENERATOR DIED EARLY - this run generated nothing after that point")
    if not final_push_ok:
        report.append("FINAL PUSH FAILED - work since the last checkpoint is unrecovered")

    _report(report)
    return 1 if (gen_died_early or not final_push_ok) else 0


def run_worker(args, root: Path, bundle: Bundle) -> int:
    from tuned.data.store import Store  # local, like run_assemble's - see the module head

    pull_and_restore(bundle, root)
    subprocess.run(
        [sys.executable, "-m", "tuned.data.reconcile", "--config", args.config], check=False
    )

    staging = root.parent / "bundle_out"

    # NOTHING TO CLAIM? Do not spend 5.25 h finding that out.
    #
    # Both children poll a queue; with none of CLAIMABLE_STATES populated they
    # poll an empty one until the deadline, and the run costs a full job slot,
    # a pull, a push and a cron cycle to report a fact the first SELECT knew.
    # Denser cron only multiplies that.
    #
    # The re-open runs ONLY on this branch. Doing it every run would be a
    # churn loop against a pool gap that is usually still there - re-open,
    # fail to route, park, re-open - 6x a day. Here it is the last thing to
    # try before declaring the queue dead, and it either finds work (in which
    # case the run proceeds normally) or it does not.
    counts = _task_counts(root / DB_RELPATH)
    if counts is not None and not _claimable(counts):
        print(f"no claimable work: {counts} - trying {', '.join(REOPEN_ON_EMPTY)}")
        subprocess.run(
            [sys.executable, "-m", "tuned.data.tasks", "--config", args.config,
             *[a for state in REOPEN_ON_EMPTY for a in ("--reopen", state)]],
            check=False,
        )
        counts = _task_counts(root / DB_RELPATH) or counts
        if not _claimable(counts):
            _report([
                "task states: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                "QUEUE EMPTY - nothing left to claim, and re-opening "
                f"{'/'.join(REOPEN_ON_EMPTY)} found nothing either. Skipped both "
                "children. Re-plan with `python -m tuned.data.tasks --config "
                "<cfg> --n <rows>`; until then every run of this workflow will "
                "end here in ~3 min.",
            ])
            # Pushed because the re-open above writes task states, and exit 0
            # because an empty queue is a fact about the plan, not a fault in
            # this job - a red run here would train the operator to ignore red.
            _push(bundle, root, staging, "queue-empty checkpoint", attempts=3)
            return 0
        print(f"re-open found work: {_claimable(counts)} claimable - continuing")

    deadline = time.monotonic() + args.minutes * 60
    next_push = time.monotonic() + args.push_every
    next_db_push = time.monotonic() + args.db_every
    procs, pumps = [], []
    for name, argv in zip(("gen", "judge"), child_argvs(
        args.config, n_workers=args.n_workers, audit_sample=args.audit_sample
    )):
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        procs.append(proc)
        pumps.append(_pump(proc, name, root / "logs" / _run_scope() / f"{name}.log"))
        print(f"[{name}] started pid={proc.pid}")

    names = ("gen", "judge")
    seen_dead: set[str] = set()
    gen_died_early = False
    push_failures = 0
    try:
        # Deliberately NOT `any(alive)`: that kept the supervisor sleeping for
        # the rest of the 5.25 h whenever the GENERATOR died and the judge
        # kept polling an empty queue. Nor `all(alive)` - a judge death (both
        # live judges share one GROQ_API_KEY, so groq going down takes them
        # together) must not stop the one process on the critical path.
        while time.monotonic() < deadline:
            for name, proc in zip(names, procs):
                if proc.poll() is not None and name not in seen_dead:
                    seen_dead.add(name)
                    print(f"[{name}] EXITED EARLY rc={proc.returncode}")
                    if name == "gen":
                        gen_died_early = True
            if gen_died_early or len(seen_dead) == len(procs):
                break
            time.sleep(15)
            if time.monotonic() >= next_push:
                # TWO CADENCES, ONE LOOP. raw/ and streams/ are appends and
                # cheap; the DB is a ~565 MB VACUUMed file rewritten whole,
                # uploaded as a brand-new blob every time, and it is almost
                # all of this repo's growth (~56 GB/day at 15-minute pushes,
                # against ~29 with the database hourly).
                #
                # What a DB-less checkpoint costs is bounded and known: on a
                # crash, reconcile_raw replays the generation and judgement
                # rows from the raw log, but not task state and not
                # gate_result, so up to --db-every of ALREADY-PAID answers
                # return to `pending` and get bought again. An hour of that
                # is the trade the storage buys, and the end-of-job push
                # always sends a database.
                with_db = time.monotonic() >= next_db_push
                label = "periodic checkpoint" + ("" if with_db else " (raw+streams)")
                if _push(bundle, root, staging, label, db=with_db):
                    push_failures = 0
                    if with_db:
                        next_db_push = time.monotonic() + args.db_every
                else:
                    push_failures += 1
                    if push_failures >= 3:
                        print(
                            f"CHECKPOINTS FAILING ({push_failures} in a row) - "
                            "the baton is not being saved"
                        )
                next_push = time.monotonic() + args.push_every
    finally:
        _stop(procs)
        for t in pumps:
            t.join(timeout=30)
        # AFTER _stop, never before: a lease is what stops two workers running
        # one task, so releasing one out from under a live child is the
        # corruption the lease exists to prevent. By here both children have
        # been terminated and waited on, so every remaining stamp belongs to a
        # process that no longer exists.
        #
        # Left alone they expire on the clock, 15 minutes after the last
        # claim - which is 15 minutes the NEXT assembly spends waiting for
        # permission to write task states, and 15 minutes those rows are
        # unqueueable for a worker that could have re-claimed them
        # immediately. The push below carries the released stamps out with
        # the rest of the DB, so the next host restores a store with no
        # phantom holders in it.
        #
        # Caught, because this runs in a `finally`: an exception raised here
        # would REPLACE whatever sent us into it, and losing the real failure
        # of an unattended run to a bookkeeping error is a bad trade. The
        # leases still expire on the clock if this cannot write.
        try:
            with Store.open(root / DB_RELPATH) as store:
                released = store.release_claims()
            print(f"released {released} lease(s) held by the stopped children")
        except Exception as exc:  # noqa: BLE001 - see above
            print(f"could not release leases ({exc!r}); they expire on the clock instead")

    # attempts=3: the periodic path delegates recovery to this push, so this
    # is the one that must not fail on a transient 5xx.
    final_push_ok = _push(
        bundle, root, staging, "end-of-job checkpoint", attempts=3
    )
    return _finish(
        root, procs, names,
        gen_died_early=gen_died_early, final_push_ok=final_push_ok,
    )


def run_assemble(args, root: Path, bundle: Bundle) -> int:
    from tuned.data.store import Store
    from tuned.data.verify import live_leases

    pull_and_restore(bundle, root)
    subprocess.run(
        [sys.executable, "-m", "tuned.data.reconcile", "--config", args.config], check=False
    )

    # verify refuses while any lease is live (it writes states unfenced).
    #
    # 120 s, not the old 900. A worker that stopped normally now releases its
    # own claims before it pushes (run_worker's finally), so the expected
    # number of live leases here is zero and the loop should not run at all.
    # 900 was DEFAULT_LEASE_S - the time a stamp takes to age out on the
    # clock - and waiting a full lease window was how this job paid for the
    # release that had not been written yet. What is left to wait for is a
    # baton snapshot taken mid-flight, whose stamps are already most of a
    # lease old, so two minutes is the margin, not the mechanism.
    #
    # And it REFUSES by name rather than falling through. Past the wait,
    # verify would exit on its own live-lease check with the chain half
    # started; worse, a lease still live after the release means something is
    # holding this baton that this job does not know about - a worker whose
    # concurrency fence failed - and the right move there is to stop, not to
    # assemble a corpus underneath it.
    waited = 0
    with Store.open(root / DB_RELPATH) as store:
        while live_leases(store) and waited < 120:
            print(f"waiting for {live_leases(store)} live lease(s) to expire...")
            time.sleep(30)
            waited += 30
        still_live = live_leases(store)
    if still_live:
        print(
            f"REFUSING: {still_live} lease(s) still live after {waited}s. A worker "
            "that stopped cleanly releases its claims, so this means either a run "
            "that died without its finally block (wait out DEFAULT_LEASE_S = 900s "
            "and re-dispatch) or another host holding this baton right now, which "
            "the data-build concurrency group is supposed to make impossible."
        )
        return 5  # this refusal's own code, distinct from a chain step's rc

    # The eval sets decontaminate screens against (~1 GB of hub parquet) are
    # deliberately NOT in the baton - they are public and re-fetchable, and a
    # gigabyte per checkpoint would dominate every push. Fetch them here, on
    # the runner, before the chain: decontaminate REFUSES outright if an eval
    # set it is measured against cannot be read, which is the correct
    # behaviour and would otherwise fail this job at its first step.
    #
    # THE EVAL SETS, and only those. Bare `--kind hf` snapshots every key in
    # acquire.HF_SOURCES, which is six: the three eval corpora this chain
    # needs, plus predex, tathyanyaya and injudgements - full-text corpus
    # inputs for seeds/select, phases that run on the operator's machine and
    # never here. injudgements alone is the largest of them.
    #
    # The keys come from EVAL_SETS rather than a literal list, so the set this
    # fetches is by construction the set decontaminate refuses to run without.
    print("== acquire eval sets ==")
    from tuned.data.eval_sets import EVAL_SETS

    eval_source_args = [a for key in sorted(EVAL_SETS) for a in ("--hf-source", key)]
    subprocess.run(
        [sys.executable, "-u", "-m", "tuned.data.acquire", "--kind", "hf",
         "--config", args.config, *eval_source_args],
        check=False,
    )

    index = root / INDEX_RELPATH
    if not index.is_file():
        index = None
        print("no citation index in the bundle - verify's existence half stays UNVERIFIED")
    # The stems the shape stage will write, read off the bundle's own stream
    # files so a stream added later is shaped without editing this script.
    streams = sorted(p.stem for p in (root / "streams").glob("*.jsonl"))
    if not streams:
        print("no stream files in the bundle - the pools ship unshaped")
    rc = 0
    for argv in assemble_argvs(args.config, citation_index=index,
                               streams=streams, out_dir=root / "out"):
        step = argv[3].rsplit(".", 1)[-1]
        print(f"== {step} ==")
        rc = subprocess.run(argv).returncode
        if rc != 0:
            print(f"{step} exited rc={rc} - stopping the chain")
            break
    if rc == 0:
        from tuned.data.config import load_build_config

        cfg = load_build_config(args.config)
        refusal = _audit_collapse_refusal(
            root / DB_RELPATH,
            cfg.routing.judge_collapse_floor,
            cfg.routing.judge_mode_since,
        )
        if refusal:
            print(refusal)
            rc = 4  # this refusal's own code - distinct from a bubbled-up chain-step rc
        else:
            print("== push ==")
            rc = subprocess.run(
                [sys.executable, "-u", "-m", "tuned.data.push", "--config", args.config]
            ).returncode
    else:
        print("stats not green - push.py skipped, artifacts still uploaded")

    out_dir = root / "out"
    if out_dir.is_dir():
        bundle.api.upload_folder(
            repo_id=bundle.repo_id, repo_type="dataset",
            folder_path=str(out_dir), path_in_repo="out",
            commit_message="assembly artifacts",
            # THIS KWARG IS ONLY SAFE BESIDE path_in_repo="out". delete_patterns
            # is resolved RELATIVE TO path_in_repo, so here it means "delete
            # everything under out/ that this upload did not just write" - and
            # with the path_in_repo line removed or changed it would mean the
            # whole repo, i.e. the baton. They are one decision and are pinned
            # together in one test for that reason; do not edit either alone.
            #
            # It exists because out/ is the only part of this repo that
            # accumulates: each dispatch writes a fresh set of artifacts under
            # the same names, but a run whose chain stopped early leaves the
            # PREVIOUS run's later stages sitting beside the new ones, where
            # they read as this run's output. Replacing the directory makes
            # out/ mean "the last assembly", which is the only thing anyone
            # reads it as.
            delete_patterns=["**"],
        )
    _push(bundle, root, root.parent / "bundle_out", "post-assembly checkpoint")
    return rc


def main_parser() -> argparse.ArgumentParser:
    """The CLI surface, separated so tests can read the DEFAULTS the
    unattended run actually uses rather than re-asserting literals."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", required=True, choices=("worker", "assemble", "seed-push"))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--hf-repo", required=True, help="private HF dataset repo for the bundle")
    parser.add_argument("--minutes", type=float, default=315)
    parser.add_argument("--push-every", type=float, default=900)
    # The DB rides a slower lane than the appends beside it. It is a ~565 MB
    # VACUUMed file rewritten whole and stored as a new blob per push, so at
    # the raw cadence it is essentially all of the baton repo's growth
    # (~56 GB/day, against ~29 hourly). The number IS the crash-loss bound:
    # reconcile_raw replays generations and judgements from the raw log but
    # neither task state nor gate_result, so a crash returns up to this much
    # already-paid work to `pending` to be bought again.
    parser.add_argument("--db-every", type=float, default=3600)
    # 12 per stream x 3 streams = 36 calls in one gather. Admission runs at
    # the generator bucket's rate (~7.5 s/call at rpm 8), so the last call is
    # admitted at ~270 s and runs at most ~120 s: ~390 s against a 900 s
    # lease. Raising this further buys less and less - the bucket, not
    # concurrency, is the ceiling (Little's law puts saturation at ~5
    # in-flight); what it actually buys is amortising the gather's tail
    # stall over more work.
    parser.add_argument("--n-workers", type=int, default=12)
    # Imported, not restated, for the same reason as SHAPED_PREFIX above: this
    # fraction IS the quality warrant for every audit-accepted row, and while
    # it was a bare literal here, changing DEFAULT_AUDIT_SAMPLE in judge.py had
    # no effect whatsoever on the unattended run - the only run that matters.
    # Function-local because judge -> generate -> providers pulls httpx into
    # the supervisor at import time, which the shape import does not.
    from tuned.data.judge import DEFAULT_AUDIT_SAMPLE

    parser.add_argument("--audit-sample", type=float, default=DEFAULT_AUDIT_SAMPLE)
    parser.add_argument(
        "--seed-push-clobbers-remote", action="store_true",
        help="allow --phase seed-push to overwrite a remote that already holds a baton",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = main_parser().parse_args(argv)

    from tuned.data.config import load_build_config
    from tuned.data.paths import build_paths

    cfg = load_build_config(args.config)
    root = build_paths(cfg.build.workdir).ensure().root
    bundle = Bundle(args.hf_repo)

    if args.phase == "seed-push":
        bundle.ensure()
        # The handoff is ONE-TIME. Re-running it once the remote owns the
        # baton overwrites the remote state with whatever this machine happens
        # to hold - the README has warned about it in prose since the handoff,
        # while the code did nothing. Make the precondition explicit instead.
        if not args.seed_push_clobbers_remote and bundle.api.file_exists(
            args.hf_repo, str(DB_RELPATH.as_posix()), repo_type="dataset"
        ):
            print(
                f"REFUSING: {args.hf_repo} already holds a state DB - the remote "
                "owns the baton. Re-running seed-push would clobber it with this "
                "machine's stale copy; single files go up via HfApi.upload_file. "
                "Pass --seed-push-clobbers-remote if that is really what you want."
            )
            return 3
        bundle.push(stage_bundle(root, root.parent / "bundle_out"), "seed: local handoff")
        print(f"seeded {args.hf_repo} from {root}")
        return 0
    if args.phase == "worker":
        return run_worker(args, root, bundle)
    return run_assemble(args, root, bundle)


if __name__ == "__main__":
    sys.exit(main())
