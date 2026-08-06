"""DDP lane (configs/law_v1_ddp.yaml): 2x T4 data-parallel, qualified by the
2026-08-06 SAVETEST. The lane must stay in lockstep with the primary config
except for what DDP itself changes: half the grad accumulation (2 ranks keep
the effective batch at 16) and an isolated checkpoint repo."""

import dataclasses
from pathlib import Path

import pytest

from tuned.train.config import load_config
from tuned.train.sft import build_sft_config, check_ddp_visibility

CONFIGS = Path(__file__).parent.parent / "configs"


def test_ddp_matches_primary_except_ga_and_repo():
    primary = load_config(CONFIGS / "law_v1.yaml")
    ddp = load_config(CONFIGS / "law_v1_ddp.yaml")
    assert ddp.model == primary.model  # same base model, same pin
    assert ddp.data == primary.data
    assert ddp.lora == primary.lora
    assert ddp.train.smoke.gradient_accumulation_steps == 8  # x2 ranks = eff 16
    assert dataclasses.replace(
        ddp.train.smoke, gradient_accumulation_steps=16
    ) == primary.train.smoke
    assert dataclasses.replace(
        ddp.train, smoke=primary.train.smoke
    ) == primary.train


def test_ddp_checkpoint_repo_isolated():
    primary = load_config(CONFIGS / "law_v1.yaml")
    ddp = load_config(CONFIGS / "law_v1_ddp.yaml")
    # Two lanes sharing a repo would silently clobber each other's
    # last-checkpoint (last push wins) and cross-load on --resume.
    assert ddp.hub.checkpoint_repo != primary.hub.checkpoint_repo
    assert ddp.hub.checkpoint_repo.endswith("-ddp")


def test_find_unused_parameters_disabled():
    cfg = load_config(CONFIGS / "law_v1_ddp.yaml")
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o")
    # TRL defaults this to True under DDP - an extra autograd-graph traversal
    # every step for nothing (every LoRA param gets a grad each step).
    assert kw["ddp_find_unused_parameters"] is False


def test_visibility_guard_rejects_masked_ranks():
    # The exact 2026-08-06 failure: notebook's CUDA_VISIBLE_DEVICES=0 leaked
    # into torchrun, each rank saw one GPU, rank 1 asked for cuda:1 and died
    # with "invalid device ordinal" mid-load.
    with pytest.raises(SystemExit, match="CUDA_VISIBLE_DEVICES"):
        check_ddp_visibility(world_size=2, visible_gpus=1)


def test_visibility_guard_passes_valid_setups():
    check_ddp_visibility(world_size=1, visible_gpus=1)  # single GPU
    check_ddp_visibility(world_size=2, visible_gpus=2)  # DDP, both visible
