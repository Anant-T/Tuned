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


def test_hub_kwargs_only_when_repo_set():
    cfg = load_config(CONFIG, allow_unpinned=True)
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o")
    if cfg.hub.checkpoint_repo is None:
        assert "hub_model_id" not in kw
    cfg.hub.checkpoint_repo = "user/ckpt"
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o")
    assert kw["hub_model_id"] == "user/ckpt"
    assert kw["hub_strategy"] == "checkpoint"
    assert kw["push_to_hub"] is True
    assert kw["hub_private_repo"] is True
