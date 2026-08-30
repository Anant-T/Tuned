"""Run the training child under a supervisor, for a Kaggle batch session.

WHY A SUPERVISOR AND NOT `!torchrun ...`: on Kaggle batch a cell's output is
only flushed when the cell COMPLETES, and cancelling a cell discards its
output entirely - so a run that hangs and gets cancelled leaves no evidence
at all (the v6/v7 sessions). Popen on plain pipes streams line-by-line, tees
to a persisted log under /kaggle/working (which survives the session), pushes
that log to the checkpoint repo every 5 min for live remote visibility,
heartbeats with GPU stats during silence, and hard-fails on a non-zero exit.
Never cancel the cell; let the watchdog kill and flush.

WHY A SUBPROCESS AND NOT AN IMPORT: the notebook's kernel starts BEFORE the
editable install in its install cell, and an editable .pth is only read at
interpreter startup - so no notebook cell can `import tuned`, and this one is
no exception. It is invoked as `python -m tuned.train.supervise`, which is
also why the module carries a CLI rather than only `supervise()`.

WHY A MODULE AND NOT 233 LINES IN THE CELL: three threads, two signal
escalation paths and a /proc scan are the part of this notebook most likely
to be wrong and least likely to be noticed, and a notebook cell is edited in
a browser with no tests and no diff. The kill correctness below was
adjudicated once, on 2026-08-09, and every clause of it is load-bearing;
test_supervise.py now pins those clauses where test_notebook.py used to pin
them as cell bytes.
"""

import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

PROBE_DATASET = "data/probe_long.jsonl"
LOG_PATH = "/kaggle/working/train.log"  # persisted output dir - survives the session
HEARTBEAT_S = 60
PUSH_EVERY_S = 300

# RESUME extends max_steps past SMOKE's 60: the final checkpoint sits AT
# max_steps, so a bare --resume would load it and exit without a single step -
# a no-op false green. 4 extra steps force a real optimizer/scaler/rng reload;
# green = first logged step is 61, not 1. sft.py refuses a resume whose
# max_steps differs from the checkpoint state unless --allow-schedule-change
# is passed: warmup and the LR decay denominator are rebuilt from max_steps,
# so the LR jumps at the resume step - fine for a 4-step gate, never for the
# main run.
#
# MAIN is the production entry: --mode main selects the config's train.main
# block (ga=6, save_steps=10) and the clean-stop budget 37800 s = 10.5 h
# measured from process start - ~30 min inside the 11 h watchdog, so
# _TimeBudget checkpoints and exits rc=0 BEFORE any kill signal. It must NEVER
# carry --allow-schedule-change: on a production resume that jump is the +134%
# LR bug the schedule guard exists to prevent. sft.py refuses MAIN while
# train.main.max_steps is still the underived 0 sentinel.
#
# ONE production entry. The retired second one carried nothing the checkpoint
# repo did not already hold, and forgetting to flip MODE on session 2 started
# at step 0 - whose first save, ten steps later, OVERWRITES last-checkpoint/ at
# the fixed path_in_repo, discarding a whole session silently.
# --resume-if-available decides from the checkpoint itself and compares
# max_steps, so it cannot resume a 60-step smoke run into a main run.
MODE_ARGS = {
    # One gate, not two: the checkpoint-push path does not depend on row
    # length, so it rides along on the probe dataset instead of costing a
    # second model load. --save-steps 1 pushes at step 1; dropping --no-hub
    # is what makes it a save gate. This also proves a push succeeds AT the
    # new sequence length, which the split PROBE/SAVETEST never tested.
    "PROBE": ["--max-steps", "2", "--save-steps", "1",
              "--dataset", PROBE_DATASET],
    "SMOKE": [],
    "RESUME": ["--resume", "--max-steps", "64", "--allow-schedule-change"],
    "MAIN": ["--resume-if-available", "--time-budget-s", "37800"],
}

# Model is pre-downloaded by the notebook's acquisition cell, so 45 min covers
# load + LoRA attach + 2 steps + 1 checkpoint push with slack; a stall now
# flushes its evidence in <=45 min instead of burning 2 h blind. PROBE: 2 steps
# at long seq plus the model load and the push fit the same budget. MAIN:
# --time-budget-s stops cleanly ~30 min before this last-resort kill fires.
PROBE_TIMEOUT_S = 45 * 60
DEFAULT_TIMEOUT_S = 11 * 3600


def mode_args(mode, probe_seq=None):
    """The child's flags for `mode`, with PROBE's optional sequence override."""
    if mode not in MODE_ARGS:
        raise ValueError(f"unknown MODE {mode!r}")
    args = list(MODE_ARGS[mode])
    if mode == "PROBE" and probe_seq:
        args += ["--max-seq-length", str(probe_seq)]
    return args


def ensure_probe_dataset(config, probe_seq=None):
    """Build the PROBE corpus here if the notebook's dataset cell did not.

    Belt-and-suspenders: a stale kernel whose dataset cell predates PROBE
    would otherwise die 55 s into the child with FileNotFoundError (the
    2026-08-07 lesson). The child depends on this file existing.
    """
    if Path(PROBE_DATASET).exists():
        return False
    announce("probe dataset missing - building it now")
    cmd = ["python", "-m", "tuned.data.probe", "--config", config]
    if probe_seq:
        cmd += ["--target-tokens", str(probe_seq)]
    if subprocess.run(cmd, timeout=5 * 60).returncode != 0:
        raise RuntimeError("probe dataset build failed")
    return True


def build_command(mode, config, probe_seq=None):
    """torchrun spawns one rank per GPU; unsloth places rank N on cuda:N (needs
    the 0,1 visibility set in the notebook's env cell). Each rank holds the
    FULL model - the parallelism is over data, not layers."""
    launcher = ["torchrun", "--nproc_per_node=2"]
    mode_flag = "main" if mode.startswith("MAIN") else "smoke"
    return [*launcher, "-m", "tuned.train.sft",
            "--config", config, "--mode", mode_flag, *mode_args(mode, probe_seq)]


def timeout_for(mode):
    return PROBE_TIMEOUT_S if mode == "PROBE" else DEFAULT_TIMEOUT_S


def announce(msg):
    """Both channels: the iopub one the notebook renders, and raw fd 2, which
    is what actually survives when the cell is killed rather than completing."""
    line = msg.rstrip("\n") + "\n"
    print(line, end="", flush=True)
    try:
        os.write(2, line.encode())
    except OSError:
        pass


def _rank_pids(parent):
    """Direct children of the launcher = the rank workers. torchrun spawns
    each rank with start_new_session=True (pytorch subprocess_handler), so
    they are NOT in the launcher's process group - killing that group alone
    would orphan two processes holding ~13 GiB of VRAM each. /proc scan, no
    psutil dependency."""
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as fh:
                fields = fh.read().rsplit(b") ", 1)[1].split()  # comm may contain spaces
            if int(fields[1]) == parent:                         # fields[1] = ppid
                pids.append(int(entry))
        except (OSError, IndexError, ValueError):
            continue
    return pids


def _killpg(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def terminate_child(proc, ranks, grace_s=60):
    """SIGTERM the launcher and let torchrun reap its own workers: its elastic
    agent catches SIGTERM and runs killpg(TERM) -> 30 s -> killpg(KILL) per
    rank; grace_s covers that escalation. Then SIGKILL our group and each
    surviving rank (each rank is its own session leader: pgid == pid). A
    longer grace does NOT save an in-flight checkpoint push - SIGTERM kills a
    rank outright (no handler installed) - and Hub commits are atomic, so the
    repo stays consistent either way."""
    _killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        announce(f"[watchdog] launcher alive {grace_s}s after SIGTERM - SIGKILL")
        _killpg(os.getpgid(proc.pid), signal.SIGKILL)
    for pid in ranks:
        _killpg(pid, signal.SIGKILL)


def _gpu_stats():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,"
             "clocks.sm,clocks.max.sm,temperature.gpu,power.draw",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().replace("\n", " | ")
    except Exception as exc:
        return f"nvidia-smi failed: {exc!r}"


class _ProgressPush:
    """Tee the log to the ckpt repo - the only channel visible mid-run.

    Uploads in a daemon thread: a slow or hung upload must never block the
    supervisor loop, which owns the heartbeats and the watchdog timeout check.
    Skips if the previous push is still in flight."""

    def __init__(self, log_path, ckpt_repo):
        self.log_path = log_path
        self.ckpt_repo = ckpt_repo
        self._inflight = threading.Event()

    def __call__(self, elapsed):
        if self._inflight.is_set():
            announce(f"[progress-push skipped +{elapsed:.0f}s - previous push still in flight]")
            return None
        self._inflight.set()

        def _upload():
            try:
                from huggingface_hub import HfApi  # lazy: import must not need the hub
                HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
                    path_or_fileobj=self.log_path, path_in_repo="progress/train.log",
                    repo_id=self.ckpt_repo, commit_message=f"progress +{elapsed:.0f}s",
                )
            except Exception as exc:
                announce(f"[progress-push failed] {exc!r}")
            finally:
                self._inflight.clear()

        t = threading.Thread(target=_upload, daemon=True)
        t.start()
        return t


def supervise(mode, config, ckpt_repo, *, probe_seq=None, log_path=LOG_PATH):
    """Run the training child to completion. Raises RuntimeError on failure."""
    # MAIN's corpus is NOT a local file this can check for: data/ is gitignored,
    # so the assembled corpus never reaches the clone and the old
    # `Path(main_dataset).exists()` belt refused every MAIN run. sft.py's
    # resolve_main_dataset fetches it from the private HF dataset repo at
    # hub.dataset_revision and refuses on a sha256 mismatch - both earlier and a
    # better message than an assert here could give.
    if mode == "PROBE":
        ensure_probe_dataset(config, probe_seq)
    cmd = build_command(mode, config, probe_seq)
    timeout_s = timeout_for(mode)
    push_progress = _ProgressPush(log_path, ckpt_repo)

    child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # replaces `python -u`
    # start_new_session: WITHOUT it the child shares the Jupyter KERNEL's
    # process group - any killpg would take the kernel down with it, and Kaggle
    # batch discards a dead cell's buffered output (v6/v7 lesson). With it,
    # killpg is scoped to the launcher's own subtree.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=child_env, start_new_session=True)  # stderr merged: one ordered stream
    announce(f"training child spawned pid={proc.pid} cmd={shlex.join(cmd)} tee={log_path}")

    last_line = [time.monotonic()]  # last time the child said anything

    def pump():
        with open(log_path, "ab", buffering=0) as log:  # unbuffered: survives SIGKILL
            for raw in iter(proc.stdout.readline, b""):
                log.write(raw)
                last_line[0] = time.monotonic()
                sys.stdout.write(raw.decode("utf-8", "replace"))
                sys.stdout.flush()
        proc.stdout.close()

    pump_t = threading.Thread(target=pump, daemon=True)
    pump_t.start()

    start = time.monotonic()
    last_beat = start
    last_push = start
    ranks = []
    try:
        while proc.poll() is None:
            time.sleep(5)
            # refresh while the launcher lives: once it exits the ppid link is
            # gone and any orphan becomes unfindable
            ranks = _rank_pids(proc.pid) or ranks
            now = time.monotonic()
            if now - start > timeout_s:
                announce(f"[watchdog] timeout after {now - start:.0f}s - terminating pid={proc.pid}")
                terminate_child(proc, ranks)
                break
            if now - last_line[0] >= HEARTBEAT_S and now - last_beat >= HEARTBEAT_S:
                announce(f"[heartbeat +{now - start:.0f}s] pid={proc.pid} "
                         f"silent {now - last_line[0]:.0f}s; gpu: {_gpu_stats()}")
                last_beat = now
            if now - last_push >= PUSH_EVERY_S:
                push_progress(now - start)
                last_push = now
    finally:
        # A cell exception must not orphan the ranks: start_new_session means
        # they no longer die with the kernel. Idempotent - already-reaped pids
        # just miss (errors swallowed in _killpg).
        if proc.poll() is None:
            terminate_child(proc, ranks)
        for pid in ranks:
            _killpg(pid, signal.SIGKILL)

    pump_t.join(timeout=30)
    rc = proc.wait()
    announce(f"training child exited rc={rc} after {time.monotonic() - start:.0f}s")
    final_push = push_progress(time.monotonic() - start)  # wins even on failure
    if final_push is not None:
        final_push.join(timeout=120)  # bounded: a hung upload must not wedge the cell either
    if rc != 0:
        raise RuntimeError(f"training failed with exit code {rc} - see {log_path}")
    return rc


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="run the training child under a supervisor")
    ap.add_argument("--mode", required=True, choices=sorted(MODE_ARGS))
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt-repo", required=True)
    ap.add_argument("--probe-seq", type=int, default=None)
    ap.add_argument("--log-path", default=LOG_PATH)
    args = ap.parse_args(argv)
    try:
        supervise(args.mode, args.config, args.ckpt_repo,
                  probe_seq=args.probe_seq, log_path=args.log_path)
    except RuntimeError as exc:
        announce(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    exit_code = main()
    # The hf upload thread is a daemon, but huggingface_hub/hf-xet can still
    # leave non-daemon machinery that wedges interpreter shutdown after every
    # byte is written - and this process is the last thing a Kaggle session
    # runs, where that hang is indistinguishable from a training stall.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
