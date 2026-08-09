"""The rsLoRA A/B lane: configs/law_v1_8b_ddp_rslora.yaml must be the
qualified lane with exactly three knobs turned - lora.use_rslora and
lora.alpha=64 (the joint treatment under test: adapter scale alpha/sqrt(r)
= 64/5.66 = 11.3x the baseline's alpha/r = 1.0) and hub.checkpoint_repo
(a shared repo means silent last-push-wins clobbering and cross-loading on
--resume - the yaml's own contract). A fourth difference would make the
A/B unreadable.
"""

from pathlib import Path

import yaml

from tuned.train.config import load_config

BASE = Path(__file__).parent.parent / "configs" / "law_v1_8b_ddp.yaml"
RSLORA = Path(__file__).parent.parent / "configs" / "law_v1_8b_ddp_rslora.yaml"


def test_base_lane_defaults_rslora_off():
    # The production yaml must not even carry the key - absence, not "false",
    # is the qualified state; the dataclass default supplies the False.
    raw = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    assert "use_rslora" not in raw["lora"]
    cfg = load_config(BASE, allow_unpinned=True)
    assert cfg.lora.use_rslora is False


def test_rslora_lane_turns_it_on():
    cfg = load_config(RSLORA, allow_unpinned=True)
    assert cfg.lora.use_rslora is True
    # operator-chosen treatment arm: alpha 64 under rsLoRA, scale 64/sqrt(32)
    assert cfg.lora.alpha == 64
    assert cfg.lora.r == 32


def test_rslora_lane_has_its_own_checkpoint_repo():
    base = load_config(BASE, allow_unpinned=True)
    rs = load_config(RSLORA, allow_unpinned=True)
    assert rs.hub.checkpoint_repo != base.hub.checkpoint_repo
    assert "rslora" in rs.hub.checkpoint_repo


def test_rslora_lane_differs_in_exactly_three_fields():
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    rs = yaml.safe_load(RSLORA.read_text(encoding="utf-8"))
    rs["lora"].pop("use_rslora")
    rs["lora"]["alpha"] = base["lora"]["alpha"]
    rs["hub"]["checkpoint_repo"] = base["hub"]["checkpoint_repo"]
    assert rs == base
