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
             deadline, pushing the bundle every --push-every seconds ->
             final push
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
# reason nothing in this file mentions.
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
    verify_step = ["verify"] + (
        ["--index", str(citation_index)] if citation_index else []
    )
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


def stage_bundle(root: Path, staging: Path) -> Path:
    """Copy everything the baton carries into a clean staging tree.

    The citation index rides along when it exists - ONE file, never the
    corpus dir (1.9 GB of source text the remote worker has no use for).
    """
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
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
    for sub in ("state", "raw", "streams", "corpus"):
        src = bundle / sub
        if not src.is_dir():
            continue
        for f in sorted(p for p in src.rglob("*") if p.is_file()):
            dest = root / f.relative_to(bundle)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
    for suffix in ("-wal", "-shm"):
        (root / DB_RELPATH).with_name(DB_RELPATH.name + suffix).unlink(missing_ok=True)


class Bundle:
    """The HF dataset repo the baton lives in. Thin, injectable for tests."""

    def __init__(self, repo_id: str):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.api = HfApi()

    def ensure(self) -> None:
        self.api.create_repo(self.repo_id, repo_type="dataset", private=True, exist_ok=True)

    def pull(self, dest: Path) -> Path:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(self.repo_id, repo_type="dataset", local_dir=dest)
        )

    def push(self, staging: Path, message: str) -> None:
        self.api.upload_folder(
            repo_id=self.repo_id, repo_type="dataset",
            folder_path=str(staging), commit_message=message,
        )


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


PUSH_BACKOFF_S = (30, 90, 240)


def _push(
    bundle: Bundle,
    root: Path,
    staging: Path,
    message: str,
    *,
    attempts: int = 1,
    sleep=None,
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
    """
    # Resolved at CALL time, not bound as a default: a default would capture
    # time.sleep at import and make the backoff unpatchable, which is a real
    # 2-minute stall in any test that exercises the retry path.
    sleep = sleep or time.sleep
    staged = stage_bundle(root, staging)
    for attempt in range(1, attempts + 1):
        try:
            bundle.push(staged, message)
            print(f"bundle pushed: {message}")
            return True
        except Exception as exc:  # noqa: BLE001
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


def _audit_summary(state_db: Path) -> list[str]:
    """The dual-judged sample's accept rate, computed where the operator will
    see it. audit_readout.py has been the documented ship gate since
    2026-08-29 while being invoked by no workflow at all, so the one number
    standing behind every audit-accepted row was machine-computed and
    machine-ignored."""
    if not state_db.is_file():
        return []
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import audit_readout

        with sqlite3.connect(f"file:{state_db.as_posix()}?mode=ro", uri=True) as conn:
            return audit_readout.format_summary(audit_readout.summarize(conn))
    except Exception as exc:  # noqa: BLE001 - a readout must never fail a run
        return [f"audit readout unavailable ({type(exc).__name__}: {exc})"]


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

    text = "\n".join(report)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except OSError as exc:
            print(f"step summary unavailable ({exc})")
    return 1 if (gen_died_early or not final_push_ok) else 0


def run_worker(args, root: Path, bundle: Bundle) -> int:
    pulled = bundle.pull(root.parent / "bundle_in")
    restore_bundle(pulled, root)
    subprocess.run(
        [sys.executable, "-m", "tuned.data.reconcile", "--config", args.config], check=False
    )

    staging = root.parent / "bundle_out"
    deadline = time.monotonic() + args.minutes * 60
    next_push = time.monotonic() + args.push_every
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
                if _push(bundle, root, staging, "periodic checkpoint"):
                    push_failures = 0
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

    pulled = bundle.pull(root.parent / "bundle_in")
    restore_bundle(pulled, root)
    subprocess.run(
        [sys.executable, "-m", "tuned.data.reconcile", "--config", args.config], check=False
    )

    # verify refuses while any lease is live (it writes states unfenced);
    # a just-killed worker's leases expire within DEFAULT_LEASE_S = 900.
    waited = 0
    with Store.open(root / DB_RELPATH) as store:
        while live_leases(store) and waited < 900:
            print(f"waiting for {live_leases(store)} live lease(s) to expire...")
            time.sleep(30)
            waited += 30

    # The eval sets decontaminate screens against (~1 GB of hub parquet) are
    # deliberately NOT in the baton - they are public and re-fetchable, and a
    # gigabyte per checkpoint would dominate every push. Fetch them here, on
    # the runner, before the chain: decontaminate REFUSES outright if an eval
    # set it is measured against cannot be read, which is the correct
    # behaviour and would otherwise fail this job at its first step.
    print("== acquire eval sets ==")
    subprocess.run(
        [sys.executable, "-u", "-m", "tuned.data.acquire", "--kind", "hf",
         "--config", args.config],
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
        )
    _push(bundle, root, root.parent / "bundle_out", "post-assembly checkpoint")
    return rc


def main_parser() -> argparse.ArgumentParser:
    """The CLI surface, separated so tests can read the DEFAULTS the
    unattended run actually uses rather than re-asserting literals."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", required=True, choices=("worker", "assemble", "seed-push"))
    parser.add_argument("--config", default="data/configs/data_law_v1.yaml")
    parser.add_argument("--hf-repo", required=True, help="private HF dataset repo for the bundle")
    parser.add_argument("--minutes", type=float, default=315)
    parser.add_argument("--push-every", type=float, default=900)
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
        bundle.push(stage_bundle(root, root.parent / "bundle_out"), "seed: local handoff")
        print(f"seeded {args.hf_repo} from {root}")
        return 0
    if args.phase == "worker":
        return run_worker(args, root, bundle)
    return run_assemble(args, root, bundle)


if __name__ == "__main__":
    sys.exit(main())
