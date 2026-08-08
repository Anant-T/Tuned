from pathlib import Path

from tuned.train.config import load_config
from tuned.train.sft import build_sft_config

CONFIG = Path(__file__).parent.parent / "configs" / "law_v1.yaml"


def test_smoke_sft_kwargs():
    cfg = load_config(CONFIG, allow_unpinned=True)
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="outputs/smoke")
    assert kw["max_steps"] == 60
    assert kw["per_device_train_batch_size"] == 1
    assert kw["gradient_accumulation_steps"] == 16
    assert kw["learning_rate"] == 2.0e-4
    assert kw["optim"] == "adamw_8bit"
    assert kw["seed"] == 3407
    assert kw["save_steps"] == 25
    assert kw["save_strategy"] == "steps"
    assert kw["output_dir"] == "outputs/smoke"
    assert kw["max_length"] == 2048


def test_hub_kwargs_only_when_repo_set():
    cfg = load_config(CONFIG, allow_unpinned=True)
    cfg.hub.checkpoint_repo = None
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o")
    assert "hub_model_id" not in kw
    cfg.hub.checkpoint_repo = "user/ckpt"
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o")
    assert kw["hub_model_id"] == "user/ckpt"
    assert kw["hub_strategy"] == "checkpoint"
    assert kw["push_to_hub"] is True
    assert kw["hub_private_repo"] is True


from tuned.train.sft import apply_overrides, check_gpu_capability

import pytest


def test_precision_flags_fp16_when_no_bf16():
    cfg = load_config(CONFIG, allow_unpinned=True)
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o", bf16_supported=False)
    assert kw["fp16"] is True
    assert kw["bf16"] is False


def test_precision_flags_bf16_when_supported():
    cfg = load_config(CONFIG, allow_unpinned=True)
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o", bf16_supported=True)
    assert kw["fp16"] is False
    assert kw["bf16"] is True


def test_report_to_gated_on_wandb_key(monkeypatch):
    cfg = load_config(CONFIG, allow_unpinned=True)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o")
    assert kw["report_to"] == "none"
    monkeypatch.setenv("WANDB_API_KEY", "k")
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o")
    assert kw["report_to"] == "wandb"


def test_apply_overrides_replaces_steps():
    cfg = load_config(CONFIG, allow_unpinned=True)
    run = apply_overrides(cfg.train.smoke, max_steps=4, save_steps=2)
    assert run.max_steps == 4
    assert run.save_steps == 2
    # untouched fields survive
    assert run.max_seq_length == cfg.train.smoke.max_seq_length
    # original is not mutated
    assert cfg.train.smoke.max_steps == 60


def test_apply_overrides_none_is_noop():
    cfg = load_config(CONFIG, allow_unpinned=True)
    run = apply_overrides(cfg.train.smoke)
    assert run == cfg.train.smoke


def test_capability_gate_rejects_p100():
    with pytest.raises(SystemExit, match="T4 x2"):
        check_gpu_capability((6, 0))


def test_capability_gate_accepts_t4():
    check_gpu_capability((7, 5))  # must not raise


def test_read_gpu_capability_no_crash():
    from tuned.train.sft import read_gpu_capability

    cap = read_gpu_capability()
    assert cap is None or (isinstance(cap, tuple) and len(cap) == 2)
