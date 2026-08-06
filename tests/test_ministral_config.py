"""Archived Ministral config: disqualified on T4 (2026-08-06, see
docs/ministral-t4-disqualification.md) but kept as the reference for the
vision-tower LoRA regex. These tests keep that regex honest and make sure the
archived config can never push checkpoints."""

from pathlib import Path

from tuned.train.config import load_config

CONFIG = Path(__file__).parent.parent / "configs" / "law_v1_ministral.yaml"

TARGET_REGEX = (
    r"(?:.*\.)?language_model\..*\."
    r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)


def test_archived_config_loads_strictly():
    cfg = load_config(CONFIG)  # revision stays pinned even in the archive
    assert cfg.model.repo == "unsloth/Ministral-3-14B-Reasoning-2512-unsloth-bnb-4bit"
    assert cfg.model.revision == "ec1befbd41647354531b2e09bd036cd1dc94b076"
    assert cfg.model.instruction_part == "[INST]"
    assert cfg.model.response_part == "[/INST]"
    assert cfg.data.think_open == "[THINK]"
    assert cfg.data.think_close == "[/THINK]"
    # Regex string scoped to the language model - keeps LoRA off the vision
    # tower (unsloth#5677 save-failure workaround).
    assert cfg.lora.target_modules == TARGET_REGEX


def test_disqualified_config_cannot_push():
    cfg = load_config(CONFIG)
    # sft.py's preflight refuses a null checkpoint_repo without --no-hub, so a
    # disqualified config can never overwrite a live lane's checkpoints.
    assert cfg.hub.checkpoint_repo is None


def test_target_regex_fullmatches_real_module_keys():
    import re

    cfg = load_config(CONFIG)
    pat = re.compile(cfg.lora.target_modules)
    # PEFT matches string target_modules with re.fullmatch against the full
    # module path, which starts with "model." on this architecture (the LoRA
    # keys quoted in unsloth#5677 prove the prefix).
    assert pat.fullmatch("model.language_model.layers.0.self_attn.q_proj")
    assert pat.fullmatch("model.language_model.layers.39.mlp.down_proj")
    assert not pat.fullmatch("model.vision_tower.transformer.layers.0.feed_forward.gate_proj")
    assert not pat.fullmatch("model.multi_modal_projector.linear_1")
    assert not pat.fullmatch("lm_head")
