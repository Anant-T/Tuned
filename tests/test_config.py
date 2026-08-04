from pathlib import Path

import pytest

from tuned.train.config import load_config

CONFIG = Path(__file__).parent.parent / "configs" / "law_v1.yaml"

TARGET_REGEX = r"language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def test_loads_repo_and_lora():
    cfg = load_config(CONFIG, allow_unpinned=True)
    assert cfg.model.repo == "unsloth/Ministral-3-14B-Reasoning-2512-unsloth-bnb-4bit"
    assert cfg.lora.r == 32
    assert cfg.lora.alpha == 32
    # Regex string scoped to the language model - keeps LoRA off the vision
    # tower (unsloth#5677 save-failure workaround).
    assert cfg.lora.target_modules == TARGET_REGEX


def test_masking_markers_and_think_tags():
    cfg = load_config(CONFIG, allow_unpinned=True)
    assert cfg.model.instruction_part == "[INST]"
    assert cfg.model.response_part == "[/INST]"
    assert cfg.data.think_open == "[THINK]"
    assert cfg.data.think_close == "[/THINK]"


def test_smoke_run_settings():
    cfg = load_config(CONFIG, allow_unpinned=True)
    assert cfg.train.smoke.max_seq_length == 2048
    assert cfg.train.smoke.max_steps == 60
    assert cfg.train.seed == 3407


def test_pinned_config_loads_strictly():
    cfg = load_config(CONFIG)
    assert cfg.model.revision == "ec1befbd41647354531b2e09bd036cd1dc94b076"


def test_unpinned_revision_rejected(tmp_path):
    import re

    tmp = tmp_path / "c.yaml"
    text = CONFIG.read_text(encoding="utf-8")
    tmp.write_text(re.sub(r"revision: \S+", "revision: null", text, count=1), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        load_config(tmp)


def test_list_target_modules_still_accepted(tmp_path):
    tmp = tmp_path / "c.yaml"
    text = CONFIG.read_text(encoding="utf-8")
    text = text.replace(
        f"target_modules: '{TARGET_REGEX}'",
        "target_modules: [q_proj, v_proj]",
    )
    tmp.write_text(text, encoding="utf-8")
    cfg = load_config(tmp)
    assert cfg.lora.target_modules == ["q_proj", "v_proj"]
