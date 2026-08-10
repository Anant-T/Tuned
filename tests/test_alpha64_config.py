"""The alpha-64 arm: configs/law_v1_8b_ddp_alpha64.yaml must be the qualified
lane with exactly two knobs turned - lora.alpha 32 -> 64 (plain-LoRA scale
alpha/r doubles, 1.0 -> 2.0; NO rsLoRA riding along - this is the third cell
of the design after baseline and the rsLoRA isolate) and hub.checkpoint_repo
(a shared repo means silent last-push-wins clobbering and cross-loading on
--resume - the yaml's own contract). max_grad_norm stays at the production
0.3 ON PURPOSE: the question is whether alpha 64 helps the production config
as-is, so the clip is part of the treatment being tested. A third difference
would make the A/B unreadable.
"""

from pathlib import Path

import yaml

from tuned.train.config import load_config

BASE = Path(__file__).parent.parent / "configs" / "law_v1_8b_ddp.yaml"
ALPHA64 = Path(__file__).parent.parent / "configs" / "law_v1_8b_ddp_alpha64.yaml"


def test_base_lane_keeps_qualified_alpha():
    raw = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    assert "use_rslora" not in raw["lora"]
    cfg = load_config(BASE, allow_unpinned=True)
    assert cfg.lora.alpha == 32
    assert cfg.train.max_grad_norm == 0.3


def test_alpha64_lane_doubles_alpha_and_nothing_else_hot():
    raw = yaml.safe_load(ALPHA64.read_text(encoding="utf-8"))
    # Plain LoRA - the key must be absent, not "false"; its presence would
    # mean this arm re-runs the closed rsLoRA question instead of isolating
    # alpha.
    assert "use_rslora" not in raw["lora"]
    cfg = load_config(ALPHA64, allow_unpinned=True)
    assert cfg.lora.alpha == 64
    assert cfg.lora.r == 32
    # Production clip retained - clip interaction is part of the question.
    assert cfg.train.max_grad_norm == 0.3


def test_alpha64_lane_has_its_own_checkpoint_repo():
    base = load_config(BASE, allow_unpinned=True)
    a64 = load_config(ALPHA64, allow_unpinned=True)
    assert a64.hub.checkpoint_repo != base.hub.checkpoint_repo
    assert "alpha64" in a64.hub.checkpoint_repo


def test_alpha64_lane_differs_in_exactly_two_fields():
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    a64 = yaml.safe_load(ALPHA64.read_text(encoding="utf-8"))
    a64["lora"]["alpha"] = base["lora"]["alpha"]
    a64["hub"]["checkpoint_repo"] = base["hub"]["checkpoint_repo"]
    assert a64 == base
