import json
from pathlib import Path

NB = Path(__file__).parent.parent / "training" / "notebooks" / "kaggle_smoke.ipynb"


def test_notebook_is_valid_and_complete():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    sources = ["".join(c["source"]) for c in nb["cells"]]
    joined = "\n".join(sources)
    # the operator's only switch is the gate: the 2026-08-09 hardening
    # (schedule guard, step-0 gates, max_grad_norm 0.3, reserved peaks) gets a
    # validation pass on the known-green recipe, and that ladder starts at
    # SMOKE - RESUME follows in a fresh session and exercises the guard
    assert 'MODE = "SMOKE"' in joined
    # re-import preflight: a re-imported notebook is a FRESH kernel and Kaggle
    # silently drops attached Secrets and Input mounts (2026-08-07, twice).
    # The notebook must discover that in seconds - before clone/install - and
    # must warn when the staged snapshot is missing, because its absence
    # silently re-opens the v6-v9 hub-download stall class
    assert 'socket.create_connection(("github.com", 443)' in joined
    assert "re-attach before running" in joined
    assert joined.count("qwen3-8b-staged/REVISION.txt") == 2
    assert joined.count('get_secret("HF_TOKEN")') == 2
    # ONE lane: the notebook IS training/configs/law_v1_8b_ddp.yaml. No lane flags, no
    # config ternary - the config, the GPU mask and the launcher are fixed and
    # can no longer drift apart from each other.
    assert 'CONFIG = "training/configs/law_v1_8b_ddp.yaml"' in joined
    # every rank must see BOTH GPUs (rank N places itself on cuda:N): a leaked
    # single-GPU mask killed rank 1 with "invalid device ordinal" on 2026-08-06
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"' in joined
    # torchrun unconditionally - one rank per GPU, full model per rank
    assert 'launcher = ["torchrun", "--nproc_per_node=2"]' in joined
    # the chunked-CE env var must be set BEFORE `import unsloth` in the
    # torchrun children, so it has to sit in the parent env. 32, not 16: the
    # CE transient scales with sequence length; the qualified peaks were
    # measured at 16, so 32 only adds margin (unquantified but real). Spent
    # 2026-08-26 and kept after the 12288 raise was reverted.
    assert 'UNSLOTH_CE_LOSS_N_CHUNKS"] = "32"' in joined
    # the retired lanes must not creep back in as dead switches
    for dead in ("DDP = ", "MP = ", "DDP_8B", "law_v1_mp.yaml", "law_v1_ddp.yaml",
                 '"configs/law_v1.yaml"', '"configs/law_v1_run.yaml"', "device_map"):
        assert dead not in joined, f"retired-lane leftover in the notebook: {dead}"
    # the session-local config copy is written BESIDE the source config. The
    # 2026-08-28 restructure moved configs/ -> training/configs/ and updated the
    # READ path above but not this WRITE path, so a bare "configs/..." literal
    # raised FileNotFoundError in every mode before any GPU work - the lane
    # could not start at all. Deriving it from CONFIG cannot drift again.
    assert 'Path(CONFIG).with_name("law_v1_run.yaml")' in joined
    # PROBE runs the long probe dataset (short unpacked examples would make
    # the long-seq VRAM probe a false green) and now PUSHES: the checkpoint
    # path does not depend on row length, so the old separate SAVETEST mode
    # cost a second model load for nothing.
    assert '"data/probe_long.jsonl"' in joined
    assert "PROBE_SEQ" in joined
    assert "--max-seq-length" in joined
    assert "tuned.data.probe" in joined
    probe_line = next(
        line for line in joined.splitlines() if line.strip().startswith('"PROBE":')
    )
    assert '"--save-steps", "1"' in probe_line
    assert "--no-hub" not in probe_line, "the merged gate must push a checkpoint"
    assert '"SAVETEST"' not in joined, "SAVETEST is retired into the PROBE gate"
    # 8192 is the qualified cap; 12288 OOM'd on 2026-08-26 and was reverted
    assert "PROBE_SEQ = 8192" in joined
    # allocator headroom - peaks land ~1.4 GiB from the 14.56 GiB cap
    assert 'PYTORCH_ALLOC_CONF"] = "expandable_segments:True"' in joined
    # scratch cache - never /kaggle/working
    assert "/tmp/hf_cache" in joined
    assert "/kaggle/working/hf_cache" not in joined
    # secrets come from Kaggle, never hardcoded
    assert "UserSecretsClient" in joined
    # W&B run names carry the lane so Hub/W&B history stays readable
    assert "8b-ddp" in joined
    # token hunt must be depth-bounded: rglob walks EVERY attached dataset, and
    # the not-found diagnostic must stream lazily (islice), never materialize
    # the whole /kaggle/input tree - a mounted multi-GB dataset turns either
    # into a minutes-long network-FS crawl
    assert 'rglob("token.txt")' not in joined
    assert '"*/token.txt"' in joined
    assert "islice" in joined
    # xet turbo is scoped to the pre-download child only (hub 1.x ignores
    # HF_HUB_ENABLE_HF_TRANSFER; this is the xet-native equivalent), and the
    # 25-min timeout still bounds the phase
    assert 'HF_XET_HIGH_PERFORMANCE": "1"' in joined
    # staged-snapshot fast path: pre-download prefers a mounted snapshot whose
    # REVISION.txt matches the config pin (staleness guard), else falls back
    # to the bounded hub download
    assert "TUNED_MODEL_PATH" in joined
    assert "qwen3-8b-staged/REVISION.txt" in joined
    # progress pushes must never block the watchdog loop (a hung upload would
    # freeze heartbeats AND the timeout check)
    assert "_push_inflight" in joined
    assert "hf_" not in joined.replace("hf_cache", "").replace("hf_transfer", "").replace("HF_", "")
    # hf_transfer must be explicitly disabled: unsloth_zoo force-enables it when
    # the var is absent, and its no-retry fast path can stall downloads silently
    assert 'HF_HUB_ENABLE_HF_TRANSFER"] = "0"' in joined
    # xet must stay ENABLED: disabling it forces the legacy bridge path, which
    # stalls from Kaggle (v6-v8 all hung on downloads with xet off; v1-v5 were fine)
    assert "HF_HUB_DISABLE_XET" not in joined.replace("HF_HUB_DISABLE_XET: v", "")
    assert "UNSLOTH_STABLE_DOWNLOADS" not in joined.replace("UNSLOTH_STABLE_DOWNLOADS / ", "")
    # dataset build follows CONFIG (think tags must match the model)
    assert '"--config", CONFIG' in joined
    # dataset-build phases must be BOUNDED: an unbounded subprocess.run wedged
    # interactive sessions when the smoke-build child hung at interpreter
    # shutdown after finishing its work (2026-08-08; likely v9's real stall)
    assert 'CONFIG], timeout=20 * 60' in joined
    assert joined.count("_probe_cmd, timeout=5 * 60") == 2
    # RESUME must extend past the smoke run's 60 steps: SMOKE's final checkpoint
    # sits AT max_steps, so a bare --resume loads it and exits without stepping
    # - a no-op false green. Forcing 4 extra steps proves optimizer/scaler/rng
    # state actually reloads and trains (2026-08-08 SMOKE-green lesson).
    # --allow-schedule-change is REQUIRED here: sft.py refuses a resume whose
    # max_steps differs from the checkpoint's, because warmup and the decay
    # denominator are rebuilt from the session's max_steps while scheduler.pt
    # restores only the step counter (the gate's LR jumped +134% at step 62).
    # The gate accepts that jump; the main run must never opt into it.
    assert '"RESUME": ["--resume", "--max-steps", "64", "--allow-schedule-change"]' in joined
    # MAIN/MAIN_RESUME are the production entries (2026-08-09 audit): they
    # carry the clean-stop budget (10.5 h, ~30 min inside the 11 h watchdog
    # SIGKILL - without it every session discards up to save_steps-1 steps,
    # worst case ~30 min of compute) and must NEVER carry
    # --allow-schedule-change: a schedule change on a production resume is
    # the +134% LR jump the guard exists to prevent.
    # ONE production entry: MAIN_RESUME carried nothing the checkpoint repo did
    # not already hold, and forgetting to flip MODE on session 2 restarted at
    # step 0 - whose first save overwrites last-checkpoint/ at the fixed
    # path_in_repo, silently discarding a session of a multi-session epoch.
    assert '"MAIN": ["--resume-if-available", "--time-budget-s", "37800"]' in joined
    assert "MAIN_RESUME" not in joined
    for line in joined.splitlines():
        if '"MAIN' in line:
            assert "--allow-schedule-change" not in line, line
    # the child's --mode follows the notebook MODE: gates run the smoke
    # config block, MAIN* runs train.main (ga=6, save_steps=10, law_v1)
    assert 'MODE.startswith("MAIN")' in joined
    # watchdog kill correctness (2026-08-09 adjudication): without
    # start_new_session the child shares the KERNEL's process group - any
    # killpg would take the kernel down and Kaggle discards its buffered
    # output. And torchrun spawns each rank as its own session leader, so
    # killing the launcher's group alone orphans two ~13 GiB processes:
    # SIGTERM lets torchrun reap its own workers (killpg TERM -> 30s -> KILL
    # per rank), the /proc ppid sweep SIGKILLs any survivor.
    assert "start_new_session=True" in joined
    assert "signal.SIGTERM" in joined
    assert "os.killpg" in joined
    assert "_rank_pids(" in joined
    assert "proc.kill()" not in joined
    # MAIN's corpus is NOT a local file the notebook can check for: data/ is
    # gitignored, so the assembled corpus never reaches this clone and the old
    # `Path(main_dataset).exists()` belt refused every MAIN run. sft.py's
    # resolve_main_dataset fetches it from the private HF dataset repo at the
    # pinned revision and refuses on a sha256 mismatch, which is both earlier
    # and a better message than an assert here could give.
    assert '["train"]["main"]["dataset"]' not in joined
    # the 13.5 abort line applies to RESERVED (the allocator high-water that
    # actually OOMs) - the PROBE bullet used to point readers at the smaller
    # allocated number, and the two mentions disagreed with each other
    assert "peak_vram_reserved_gb_dev0/1` below **13.5 GiB**" in joined
    assert "abort-and-rethink at ~13.5." not in joined
    # notebook cells read configs with plain yaml and never import the package:
    # a kernel started before the editable install misses the .pth, so only
    # subprocesses can import tuned
    assert "from tuned" not in joined
    # every run is re-homed to the session account's own HF namespace
    assert "whoami" in joined


def test_every_path_the_notebook_writes_has_a_parent_that_exists_in_the_clone():
    """A write into a directory the repo does not carry is a session killer.

    This is the CLASS of the 2026-08-28 defect, not just its instance: cell 7
    wrote `configs/law_v1_run.yaml` after the restructure had moved that
    directory to `training/configs/`, so `Path.write_text` raised
    FileNotFoundError on a fresh `git clone --depth 1` - in PROBE, SMOKE,
    RESUME and MAIN alike, before the model download and before any GPU work.
    Nothing caught it because the only assertions were on the READ path.

    Only literal targets are checkable; a target derived from CONFIG (which is
    what the fix uses) resolves at runtime and is skipped here - the literal
    ban in test_notebook_is_valid_and_complete covers the regression.
    """
    import re

    nb = json.loads(NB.read_text(encoding="utf-8"))
    joined = "\n".join("".join(c["source"]) for c in nb["cells"])
    repo = NB.parent.parent.parent

    literals = re.findall(r'Path\(\s*"([^"]+)"\s*\)\s*\.write_text', joined)
    for target in literals:
        parent = (repo / target).parent
        assert parent.is_dir(), (
            f"the notebook writes {target!r}, whose parent {parent} does not "
            "exist in a fresh clone - the session dies before the GPU"
        )


def test_stage_model_notebook_matches_the_8b_pin():
    import yaml

    stage = json.loads((NB.parent / "stage_model.ipynb").read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in stage["cells"])
    cfg = yaml.safe_load(
        (Path(__file__).parent.parent / "training" / "configs" / "law_v1_8b_ddp.yaml").read_text(encoding="utf-8")
    )
    # the staging notebook must stage EXACTLY the lane's pinned repo+revision -
    # a drifted pin would make the fast path silently fall back (or worse,
    # stage a snapshot no lane can use)
    assert cfg["model"]["repo"] in src
    assert cfg["model"]["revision"] in src
    # staged layout contract shared with kaggle_smoke's pre-download cell
    assert "qwen3-8b-staged" in src
    assert "REVISION.txt" in src


def test_wandb_run_name_carries_the_config_variant():
    # Experiment configs reuse MODE="SMOKE", so without a suffix every arm
    # would land in W&B as the same "8b-ddp-smoke", told apart only by run
    # id. The suffix must DERIVE from CONFIG - the switch the operator
    # actually flips - and stay bare for the production yaml; hardcoding one
    # experiment's name broke the moment a second experiment existed.
    nb = json.loads(NB.read_text(encoding="utf-8"))
    joined = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert 'removeprefix("law_v1_8b_ddp")' in joined
    assert '"-rslora" if "rslora" in CONFIG' not in joined
