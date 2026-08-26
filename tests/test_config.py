from pathlib import Path

import pytest

from tuned.train.config import load_config

CONFIG = Path(__file__).parent.parent / "configs" / "law_v1_8b_ddp.yaml"

MODULE_LIST = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def test_loads_repo_and_lora():
    cfg = load_config(CONFIG, allow_unpinned=True)
    assert cfg.model.repo == "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    assert cfg.lora.r == 32
    assert cfg.lora.alpha == 32
    # Qwen3 is text-only - plain module list, no vision tower to exclude.
    assert cfg.lora.target_modules == MODULE_LIST


def test_masking_markers_and_think_tags():
    cfg = load_config(CONFIG, allow_unpinned=True)
    assert cfg.model.instruction_part == "<|im_start|>user\n"
    assert cfg.model.response_part == "<|im_start|>assistant\n"
    assert cfg.data.think_open == "<think>"
    assert cfg.data.think_close == "</think>"


def test_smoke_run_settings():
    cfg = load_config(CONFIG, allow_unpinned=True)
    assert cfg.train.smoke.max_seq_length == 8192
    assert cfg.train.smoke.max_steps == 60
    assert cfg.train.seed == 3407


def test_pinned_config_loads_strictly():
    cfg = load_config(CONFIG)
    assert cfg.model.revision == "62efd7f9d748e394734a7adae2adf96e13a2abc8"


def test_unpinned_revision_rejected(tmp_path):
    import re

    tmp = tmp_path / "c.yaml"
    text = CONFIG.read_text(encoding="utf-8")
    tmp.write_text(re.sub(r"revision: \S+", "revision: null", text, count=1), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        load_config(tmp)


def test_regex_target_modules_still_accepted(tmp_path):
    # The loader must keep accepting a regex STRING alongside the plain list:
    # it is how a vision-tower base model would scope LoRA to the language
    # tower (unsloth#5677), and sft.py's vision guard points at that remedy.
    tmp = tmp_path / "c.yaml"
    text = CONFIG.read_text(encoding="utf-8")
    text = text.replace(
        "target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]",
        r"target_modules: '(?:.*\.)?language_model\..*\.(?:q_proj|v_proj)'",
    )
    tmp.write_text(text, encoding="utf-8")
    cfg = load_config(tmp)
    assert cfg.lora.target_modules == r"(?:.*\.)?language_model\..*\.(?:q_proj|v_proj)"
