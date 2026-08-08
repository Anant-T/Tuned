"""The production lane (configs/law_v1_8b_ddp.yaml): Qwen3-8B under plain
2x T4 data-parallel via torchrun. FULLY QUALIFIED 2026-08-08 - all four gates
green (PROBE 12.80/13.00 GiB, SAVETEST, SMOKE 60/60 at 74.7 s/step with peaks
12.98/13.18 GiB, RESUME). Every value asserted here is a live contract with
the Kaggle notebook, notebooks/stage_model.ipynb, or the Hub checkpoint repo -
none of it may drift without re-running the ladder."""

from pathlib import Path

from tuned.train.config import load_config

CONFIGS = Path(__file__).parent.parent / "configs"


def test_model_repo_and_revision_pinned():
    cfg = load_config(CONFIGS / "law_v1_8b_ddp.yaml")
    assert cfg.model.repo == "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    assert cfg.model.revision == "62efd7f9d748e394734a7adae2adf96e13a2abc8"


def test_checkpoint_repo_is_the_live_one():
    # The gates' checkpoints (and --resume) live here. The notebook re-homes
    # the namespace per session but keeps the repo NAME, so this string is the
    # contract: change it and --resume silently starts from scratch.
    cfg = load_config(CONFIGS / "law_v1_8b_ddp.yaml")
    assert cfg.hub.checkpoint_repo == "tantan01/tuned-law-v1-qwen8b-ckpt-ddp"


def test_tokens_per_optimizer_step():
    run = load_config(CONFIGS / "law_v1_8b_ddp.yaml").train.smoke
    # 8192 x 1 x 2 x 2 ranks = 32,768 tokens/optimizer-step - the budget the
    # measured 74.7 s/step and the peak-VRAM numbers were taken under.
    assert run.max_seq_length * run.per_device_train_batch_size * run.gradient_accumulation_steps * 2 == 32768


def test_smoke_run_shape():
    run = load_config(CONFIGS / "law_v1_8b_ddp.yaml").train.smoke
    assert run.max_seq_length == 8192
    assert run.per_device_train_batch_size == 1
    assert run.gradient_accumulation_steps == 2
    assert run.max_steps == 60
    assert run.save_steps == 25
    assert run.dataset == "data/smoke_v1.jsonl"
