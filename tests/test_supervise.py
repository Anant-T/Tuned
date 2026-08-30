"""The kill correctness that used to be pinned as bytes of notebook cell 10.

These assertions did not change when the supervisor moved out of
kaggle_smoke.ipynb into tuned.train.supervise - only what they read. They are
text assertions rather than executions on purpose: `_rank_pids` reads /proc and
`terminate_child` calls os.killpg with SIGKILL, none of which exist off Linux,
so running them would make the suite pass or fail on which machine ran it. That
is the same shape the notebook tests had, and it is the right shape - what is
being defended here is a set of clauses that were adjudicated once, on
2026-08-09, and that a later edit could quietly drop without any test noticing.
"""

import ast
from pathlib import Path

import pytest

from tuned.train import supervise

SRC = Path(supervise.__file__)
TEXT = SRC.read_text(encoding="utf-8")


def test_torchrun_unconditionally_one_rank_per_gpu():
    """One rank per GPU, full model per rank - the parallelism is over data."""
    assert 'launcher = ["torchrun", "--nproc_per_node=2"]' in TEXT
    assert supervise.build_command("SMOKE", "c.yaml")[:2] == ["torchrun", "--nproc_per_node=2"]


def test_the_child_mode_follows_the_notebook_mode():
    """Gates run the smoke config block, MAIN runs train.main (ga=6,
    save_steps=10, the assembled corpus)."""
    assert 'MODE.startswith("MAIN")' not in TEXT  # the cell's form; now a parameter

    def mode_flag(mode):
        cmd = supervise.build_command(mode, "c.yaml")
        return cmd[cmd.index("--mode") + 1]

    assert mode_flag("MAIN") == "main"
    for gate in ("PROBE", "SMOKE", "RESUME"):
        assert mode_flag(gate) == "smoke"


def test_resume_extends_past_the_smoke_runs_sixty_steps():
    """SMOKE's final checkpoint sits AT max_steps, so a bare --resume loads it
    and exits without stepping - a no-op false green. Forcing 4 extra steps
    proves optimizer/scaler/rng state actually reloads and trains (2026-08-08
    SMOKE-green lesson). --allow-schedule-change is REQUIRED here: sft.py
    refuses a resume whose max_steps differs from the checkpoint's, because
    warmup and the decay denominator are rebuilt from the session's max_steps
    while scheduler.pt restores only the step counter (the gate's LR jumped
    +134% at step 62). The gate accepts that jump; the main run must never opt
    into it."""
    assert supervise.MODE_ARGS["RESUME"] == [
        "--resume", "--max-steps", "64", "--allow-schedule-change",
    ]


def test_main_is_the_one_production_entry_and_never_changes_the_schedule():
    """MAIN carries the clean-stop budget (10.5 h, ~30 min inside the 11 h
    watchdog SIGKILL - without it every session discards up to save_steps-1
    steps, worst case ~30 min of compute) and must NEVER carry
    --allow-schedule-change: on a production resume that jump is the +134% LR
    bug the guard exists to prevent.

    ONE production entry: MAIN_RESUME carried nothing the checkpoint repo did
    not already hold, and forgetting to flip MODE on session 2 restarted at
    step 0 - whose first save overwrites last-checkpoint/ at the fixed
    path_in_repo, silently discarding a session of a multi-session epoch."""
    assert supervise.MODE_ARGS["MAIN"] == ["--resume-if-available", "--time-budget-s", "37800"]
    assert "MAIN_RESUME" not in TEXT
    for mode, args in supervise.MODE_ARGS.items():
        if mode.startswith("MAIN"):
            assert "--allow-schedule-change" not in args, mode
    for line in TEXT.splitlines():
        if '"MAIN' in line:
            assert "--allow-schedule-change" not in line, line


def test_the_watchdog_kills_the_ranks_and_not_the_kernel():
    """2026-08-09 adjudication. Without start_new_session the child shares the
    KERNEL's process group - any killpg would take the kernel down and Kaggle
    discards its buffered output. And torchrun spawns each rank as its own
    session leader, so killing the launcher's group alone orphans two ~13 GiB
    processes: SIGTERM lets torchrun reap its own workers (killpg TERM -> 30 s
    -> KILL per rank), and the /proc ppid sweep SIGKILLs any survivor."""
    assert "start_new_session=True" in TEXT
    assert "signal.SIGTERM" in TEXT
    assert "os.killpg" in TEXT
    assert "_rank_pids(" in TEXT
    # proc.kill() kills the launcher only, leaving both ranks holding their VRAM
    assert "proc.kill()" not in TEXT


def test_a_progress_push_never_blocks_the_watchdog_loop():
    """A hung upload would freeze the heartbeats AND the timeout check, which
    is the one thing the supervisor exists to keep alive."""
    assert "_inflight" in TEXT
    push = supervise._ProgressPush("/tmp/x.log", "org/repo")
    push._inflight.set()
    assert push(elapsed=1.0) is None  # skipped, not queued, while one is in flight


def test_the_probe_dataset_is_built_here_when_the_dataset_cell_did_not():
    """Belt-and-suspenders: a stale kernel whose dataset cell predates PROBE
    would otherwise die 55 s into the child with FileNotFoundError (the
    2026-08-07 lesson)."""
    assert "_probe_cmd, timeout=5 * 60" not in TEXT  # the cell's name for it
    assert "timeout=5 * 60" in TEXT
    assert supervise.PROBE_DATASET in supervise.MODE_ARGS["PROBE"]


def test_mains_corpus_is_not_checked_for_as_a_local_path():
    """data/ is gitignored, so the assembled corpus never reaches the clone and
    the old `Path(main_dataset).exists()` belt refused every MAIN run. sft.py's
    resolve_main_dataset fetches it from the private HF dataset repo at the
    pinned revision and refuses on a sha256 mismatch, which is both earlier and
    a better message than an assert here could give."""
    assert '["train"]["main"]["dataset"]' not in TEXT
    assert "resolve_main_dataset" in TEXT  # named, so the reader is sent there


def test_probe_seq_only_reaches_the_probe_mode():
    assert supervise.mode_args("PROBE", 12288)[-2:] == ["--max-seq-length", "12288"]
    assert "--max-seq-length" not in supervise.mode_args("PROBE", None)
    assert "--max-seq-length" not in supervise.mode_args("SMOKE", 12288)


def test_the_watchdog_budget_leaves_the_clean_stop_room_to_fire_first():
    """MAIN's --time-budget-s must stop the child cleanly BEFORE the watchdog's
    last-resort kill, or every session loses up to save_steps-1 steps."""
    budget = int(supervise.MODE_ARGS["MAIN"][supervise.MODE_ARGS["MAIN"].index("--time-budget-s") + 1])
    assert budget < supervise.DEFAULT_TIMEOUT_S
    assert supervise.DEFAULT_TIMEOUT_S - budget >= 25 * 60  # ~30 min of margin
    assert supervise.timeout_for("PROBE") == supervise.PROBE_TIMEOUT_S
    assert supervise.timeout_for("MAIN") == supervise.DEFAULT_TIMEOUT_S


def test_the_hub_client_is_imported_lazily():
    """Importing the supervisor must not need huggingface_hub: the module is
    read by this suite on machines that never push a checkpoint."""
    tree = ast.parse(TEXT)
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert "huggingface_hub" not in {a.name.split(".")[0] for a in node.names}
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] != "huggingface_hub"


def test_an_unknown_mode_is_refused_before_anything_is_spawned():
    with pytest.raises(ValueError, match="unknown MODE"):
        supervise.mode_args("MAIN_RESUME")


def test_the_probe_gate_is_one_gate_and_not_two():
    """The checkpoint-push path does not depend on row length, so it rides
    along on the probe dataset instead of costing a second model load.
    --save-steps 1 pushes at step 1, and NOT passing --no-hub is what makes it
    a save gate - which also proves a push succeeds AT the new sequence length,
    something the split PROBE/SAVETEST never tested."""
    probe = supervise.MODE_ARGS["PROBE"]
    assert probe[probe.index("--save-steps") + 1] == "1"
    assert "--no-hub" not in probe, "the merged gate must push a checkpoint"
    # the word still appears in the module's history above; what must not come
    # back is the MODE
    assert "SAVETEST" not in supervise.MODE_ARGS
