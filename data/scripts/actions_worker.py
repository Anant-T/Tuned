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
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

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


def assemble_argvs(config: str) -> list[list[str]]:
    """The assembly chain, in order. stats is last and gates push.py."""
    chain = [
        ["verify"], ["decontaminate"], ["dedupe"], ["split"], ["assemble"],
        ["stats", "--profile", "v1.0-MVP"],
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


def stage_bundle(root: Path, staging: Path) -> Path:
    """Copy everything the baton carries into a clean staging tree."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    snapshot_db(root / DB_RELPATH, staging / DB_RELPATH)
    for sub in ("raw", "streams", "logs"):
        src = root / sub
        if src.is_dir():
            shutil.copytree(src, staging / sub)
    return staging


def restore_bundle(bundle: Path, root: Path) -> None:
    """Land a pulled bundle in the workdir. Fresh-checkout semantics: the
    bundle's copy of a file wins, and any stale -wal/-shm beside the DB is
    dropped (the snapshot is self-contained; a leftover WAL from a previous
    life would be replayed over it)."""
    for sub in ("state", "raw", "streams"):
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


def _push_quietly(bundle: Bundle, root: Path, staging: Path, message: str) -> None:
    """Periodic pushes are best-effort: a transient HF failure must not kill
    the fleet mid-run - the next interval (or the final push) retries."""
    try:
        bundle.push(stage_bundle(root, staging), message)
        print(f"bundle pushed: {message}")
    except Exception as exc:  # noqa: BLE001
        print(f"bundle push failed ({type(exc).__name__}: {exc}) - continuing")


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
        pumps.append(_pump(proc, name, root / "logs" / f"{name}.log"))
        print(f"[{name}] started pid={proc.pid}")

    try:
        while time.monotonic() < deadline and any(p.poll() is None for p in procs):
            time.sleep(15)
            if time.monotonic() >= next_push:
                _push_quietly(bundle, root, staging, "periodic checkpoint")
                next_push = time.monotonic() + args.push_every
    finally:
        _stop(procs)
        for t in pumps:
            t.join(timeout=30)

    for name, proc in zip(("gen", "judge"), procs):
        print(f"[{name}] exited rc={proc.returncode}")
    bundle.push(stage_bundle(root, staging), "end-of-job checkpoint")
    print("final bundle push done")
    return 0


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

    rc = 0
    for argv in assemble_argvs(args.config):
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
    _push_quietly(bundle, root, root.parent / "bundle_out", "post-assembly checkpoint")
    return rc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", required=True, choices=("worker", "assemble", "seed-push"))
    parser.add_argument("--config", default="data/configs/data_law_v1.yaml")
    parser.add_argument("--hf-repo", required=True, help="private HF dataset repo for the bundle")
    parser.add_argument("--minutes", type=float, default=315)
    parser.add_argument("--push-every", type=float, default=900)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--audit-sample", type=float, default=0.05)
    args = parser.parse_args(argv)

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
