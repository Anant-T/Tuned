"""8B DDP lane (configs/law_v1_8b_ddp.yaml): Qwen3-8B under plain 2x T4
data-parallel via torchrun - UNPROBED as of 2026-08-07 (quota exhausted
before a Kaggle run). Same family template as law_v1_ddp.yaml (ChatML,
identical LoRA/data markers); the lane-specific deltas are the base model
pin, max_seq_length, and an isolated checkpoint repo."""

import dataclasses
from pathlib import Path

from tuned.train.config import load_config

CONFIGS = Path(__file__).parent.parent / "configs"


def test_model_repo_and_revision_pinned():
    cfg = load_config(CONFIGS / "law_v1_8b_ddp.yaml")
    assert cfg.model.repo == "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    assert cfg.model.revision == "62efd7f9d748e394734a7adae2adf96e13a2abc8"


def test_checkpoint_repo_isolated():
    repos = {
        name: load_config(CONFIGS / f"law_v1{suffix}.yaml").hub.checkpoint_repo
        for name, suffix in [
            ("primary", ""),
            ("ddp", "_ddp"),
            ("mp", "_mp"),
            ("8b_ddp", "_8b_ddp"),
        ]
    }
    assert repos["8b_ddp"].endswith("-ddp")
    assert len(set(repos.values())) == 4  # no two lanes share a repo


def test_token_parity_with_other_lanes():
    run = load_config(CONFIGS / "law_v1_8b_ddp.yaml").train.smoke
    # 8192 x 1 x 2 x 2 ranks = 32,768 tokens/optimizer-step, matching every
    # other lane's token budget per step.
    assert run.max_seq_length * run.per_device_train_batch_size * run.gradient_accumulation_steps * 2 == 32768


def test_no_device_map():
    # DDP-style lane: each rank holds the FULL model, no layer split.
    assert load_config(CONFIGS / "law_v1_8b_ddp.yaml").model.device_map is None


def test_matches_ddp_family_template():
    ddp = load_config(CONFIGS / "law_v1_ddp.yaml")
    eightb = load_config(CONFIGS / "law_v1_8b_ddp.yaml")
    # Same Qwen3 ChatML template - instruction/response markers and think
    # tags are identical across the family regardless of model size.
    assert eightb.model.instruction_part == ddp.model.instruction_part
    assert eightb.model.response_part == ddp.model.response_part
    assert eightb.data == ddp.data
    assert eightb.lora == ddp.lora
    # LoRA swappability: same target-module scoping as the rest of the family.
    assert eightb.lora.target_modules == ddp.lora.target_modules


def test_smoke_run_shape_matches_family_except_seq_and_ga():
    ddp = load_config(CONFIGS / "law_v1_ddp.yaml").train.smoke
    eightb = load_config(CONFIGS / "law_v1_8b_ddp.yaml").train.smoke
    assert eightb.max_seq_length == 8192
    assert eightb.per_device_train_batch_size == 1
    assert eightb.gradient_accumulation_steps == 2
    assert eightb.max_steps == 60
    assert eightb.save_steps == 25
    assert eightb.dataset == "data/smoke_v1.jsonl"
    assert dataclasses.replace(
        eightb, max_seq_length=ddp.max_seq_length, gradient_accumulation_steps=ddp.gradient_accumulation_steps
    ) == ddp
