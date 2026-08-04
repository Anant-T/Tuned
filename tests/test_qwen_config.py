from pathlib import Path

from tuned.train.config import load_config

CONFIG = Path(__file__).parent.parent / "configs" / "law_v1_qwen.yaml"


def test_qwen_fallback_loads_strictly():
    cfg = load_config(CONFIG)  # strict: must be pinned
    assert cfg.model.repo == "unsloth/Qwen3-14B-unsloth-bnb-4bit"
    assert cfg.model.revision == "46105e245750aad3be7fd1d81c21cb03a0e438ed"
    assert cfg.model.instruction_part == "<|im_start|>user\n"
    assert cfg.model.response_part == "<|im_start|>assistant\n"
    assert cfg.data.think_open == "<think>"
    assert cfg.data.think_close == "</think>"
    # Qwen3 is text-only - plain list, no vision tower to exclude.
    assert cfg.lora.target_modules == [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]


def test_qwen_fallback_matches_primary_hyperparams():
    primary = load_config(CONFIG.parent / "law_v1.yaml")
    qwen = load_config(CONFIG)
    assert qwen.train == primary.train
    assert qwen.lora.r == primary.lora.r
    assert qwen.lora.alpha == primary.lora.alpha
