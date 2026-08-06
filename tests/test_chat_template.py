"""Render a sample conversation through the real tokenizer and assert the
config's masking markers appear. Skips locally (no [train] extra); runs on
Kaggle where transformers is installed and internet is on.

Catches: a transformers/tokenizer change silently altering the template, or a
config pointing at markers from the wrong template family (ChatML vs Mistral).
"""

from pathlib import Path

import pytest

from tuned.train.config import load_config

CONFIGS = Path(__file__).parent.parent / "configs"


# Only live lanes are checked (law_v1_ddp.yaml shares law_v1.yaml's model; the
# archived Ministral config is not worth a tokenizer fetch per Kaggle session).
@pytest.mark.parametrize("name", ["law_v1.yaml"])
def test_markers_and_think_tags_render(name):
    transformers = pytest.importorskip("transformers")
    cfg = load_config(CONFIGS / name)
    tok = transformers.AutoTokenizer.from_pretrained(
        cfg.model.repo, revision=cfg.model.revision
    )
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": f"{cfg.data.think_open}Adding.{cfg.data.think_close}4",
        },
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    assert cfg.model.instruction_part in text
    assert cfg.model.response_part in text
    # instruction marker must precede response marker
    assert text.index(cfg.model.instruction_part) < text.index(cfg.model.response_part)
    # the reasoning scaffold survives rendering (single-turn: last assistant
    # message - Qwen3's template only strips think blocks from earlier turns)
    # and must not be doubled by a template that re-wraps assistant reasoning.
    # Count within the response region only: Ministral's auto-injected default
    # system prompt legitimately mentions [THINK] once in its instruction text.
    response_region = text[text.index(cfg.model.response_part) :]
    assert response_region.count(cfg.data.think_open) == 1
    assert response_region.count(cfg.data.think_close) == 1
    # the trainable answer sits after the response marker
    assert text.rindex("4") > text.index(cfg.model.response_part)
