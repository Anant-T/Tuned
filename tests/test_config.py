from pathlib import Path

import pytest

from tuned.train.config import load_config

CONFIG = Path(__file__).parent.parent / "configs" / "law_v1.yaml"


def test_loads_repo_and_lora():
    cfg = load_config(CONFIG, allow_unpinned=True)
    assert cfg.model.repo == "unsloth/gemma-4-31B-it-unsloth-bnb-4bit"
    assert cfg.lora.r == 32
    assert cfg.lora.alpha == 32
    assert cfg.lora.target_modules == [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]


def test_smoke_run_settings():
    cfg = load_config(CONFIG, allow_unpinned=True)
    assert cfg.train.smoke.max_seq_length == 2048
    assert cfg.train.smoke.max_steps == 60
    assert cfg.train.seed == 3407


def test_unpinned_revision_rejected():
    with pytest.raises(ValueError, match="revision"):
        load_config(CONFIG)
