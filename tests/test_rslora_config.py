"""The rsLoRA A/B lane: configs/law_v1_8b_ddp_rslora.yaml must be the
qualified lane with exactly two knobs turned - lora.use_rslora (the variable
under test: adapter scale alpha/sqrt(r) instead of alpha/r, a 5.66x jump at
r=32/alpha=32) and hub.checkpoint_repo (a shared repo means silent
last-push-wins clobbering and cross-loading on --resume - the yaml's own
contract). A third difference would make the A/B unreadable.
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


def test_rslora_lane_has_its_own_checkpoint_repo():
    base = load_config(BASE, allow_unpinned=True)
    rs = load_config(RSLORA, allow_unpinned=True)
    assert rs.hub.checkpoint_repo != base.hub.checkpoint_repo
    assert "rslora" in rs.hub.checkpoint_repo


def test_rslora_lane_differs_in_exactly_two_fields():
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    rs = yaml.safe_load(RSLORA.read_text(encoding="utf-8"))
    rs["lora"].pop("use_rslora")
    rs["hub"]["checkpoint_repo"] = base["hub"]["checkpoint_repo"]
    assert rs == base
