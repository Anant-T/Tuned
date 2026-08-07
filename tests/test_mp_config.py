"""MP lane (configs/law_v1_mp.yaml): 2x T4 model-parallel via device_map.
Must stay in lockstep with the primary config except for what MP itself
changes: seq 6144, token-matched grad accumulation, device_map, and an
isolated checkpoint repo. Plus the fail-fast guards that keep a bad launch
from burning quota."""

import dataclasses
from pathlib import Path

import pytest

from tuned.train.config import load_config
from tuned.train.sft import (
    apply_overrides,
    build_sft_config,
    check_max_memory_requires_device_map,
    check_model_split,
    check_mp_gpu_count,
    check_mp_torchrun_conflict,
)

CONFIGS = Path(__file__).parent.parent / "configs"


def test_mp_matches_primary_except_seq_ga_devicemap_repo():
    primary = load_config(CONFIGS / "law_v1.yaml")
    mp = load_config(CONFIGS / "law_v1_mp.yaml")
    assert mp.model == dataclasses.replace(
        primary.model, device_map="sequential", max_memory={0: "8GiB", 1: "13GiB"}
    )
    assert mp.data == primary.data
    assert mp.lora == primary.lora  # swappability: same target-module scoping
    assert mp.train.smoke.max_seq_length == 6144
    # 6144 x 5 = 30,720 tokens/step ~= the 2048 lanes' 32,768.
    assert mp.train.smoke.gradient_accumulation_steps == 5
    assert dataclasses.replace(
        mp.train.smoke, max_seq_length=2048, gradient_accumulation_steps=16
    ) == primary.train.smoke
    assert dataclasses.replace(mp.train, smoke=primary.train.smoke) == primary.train


def test_mp_checkpoint_repo_isolated():
    repos = {
        name: load_config(CONFIGS / f"law_v1{suffix}.yaml").hub.checkpoint_repo
        for name, suffix in [("primary", ""), ("ddp", "_ddp"), ("mp", "_mp")]
    }
    assert repos["mp"].endswith("-mp")
    assert len(set(repos.values())) == 3  # no two lanes share a repo


def test_existing_lanes_have_no_device_map():
    # device_map=None must remain the byte-for-byte behavior of the two
    # qualified lanes; only the MP lane sets it.
    assert load_config(CONFIGS / "law_v1.yaml").model.device_map is None
    assert load_config(CONFIGS / "law_v1_ddp.yaml").model.device_map is None


def test_mp_guard_rejects_torchrun():
    with pytest.raises(SystemExit, match="torchrun"):
        check_mp_torchrun_conflict("balanced", world_size=2)


def test_mp_guard_rejects_single_visible_gpu():
    with pytest.raises(SystemExit, match="CUDA_VISIBLE_DEVICES"):
        check_mp_gpu_count("balanced", visible_gpus=1)


def test_mp_guards_pass_valid_setups():
    check_mp_torchrun_conflict("balanced", world_size=1)  # plain python launch
    check_mp_gpu_count("balanced", visible_gpus=2)
    # No device_map = guards are inert whatever the launch looks like.
    check_mp_torchrun_conflict(None, world_size=2)
    check_mp_gpu_count(None, visible_gpus=1)


def test_split_check_rejects_single_device():
    with pytest.raises(SystemExit, match="split did not happen"):
        check_model_split(["cuda:0"])
    with pytest.raises(SystemExit, match="split did not happen"):
        check_model_split(["cuda:0", "cpu"])  # offload is not a 2-GPU split


def test_split_check_passes_two_gpus():
    check_model_split(["cuda:0", "cuda:1"])


def test_max_memory_skew_keeps_last_gpu_light():
    cfg = load_config(CONFIGS / "law_v1_mp.yaml").model
    # sequential fills dev0 FIRST, so the dev0 cap is what bounds dev1's
    # weight spill - dev0's cap must be the small one, and it must be below
    # the ~10.4 GiB total 4-bit footprint or everything lands on dev0.
    # (balanced + max_memory instead pushes the surplus to CPU and bnb
    # refuses - the 2026-08-07 failure.)
    assert cfg.device_map == "sequential"
    mm = cfg.max_memory
    assert int(mm[0].replace("GiB", "")) < int(mm[1].replace("GiB", ""))
    assert int(mm[0].replace("GiB", "")) < 10


def test_max_memory_guard_requires_device_map():
    with pytest.raises(SystemExit, match="device_map"):
        check_max_memory_requires_device_map(None, {0: "13GiB", 1: "5GiB"})
    check_max_memory_requires_device_map("balanced", {0: "13GiB", 1: "5GiB"})
    check_max_memory_requires_device_map(None, None)


def test_dataset_and_seq_overrides():
    run = load_config(CONFIGS / "law_v1_mp.yaml").train.smoke
    assert apply_overrides(run).dataset == run.dataset
    assert apply_overrides(run, dataset="data/probe_long.jsonl").dataset == "data/probe_long.jsonl"
    assert apply_overrides(run).max_seq_length == 6144
    assert apply_overrides(run, max_seq_length=8192).max_seq_length == 8192


def test_warmup_converted_to_steps():
    # warmup_ratio is deprecated in transformers 5.5 (lr logged 0 in the
    # 2026-08-07 probe); build_sft_config converts it.
    cfg = load_config(CONFIGS / "law_v1_mp.yaml")
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o")
    assert "warmup_ratio" not in kw
    assert kw["warmup_steps"] == 2  # round(0.03 * 60)
