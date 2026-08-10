"""The rsLoRA isolate arm: configs/law_v1_8b_ddp_rslora_a32.yaml must be the
qualified lane with exactly three knobs turned - lora.use_rslora (alpha stays
at the base 32, so the adapter scale is a pure 32/sqrt(32) = 5.66x rsLoRA
treatment), train.max_grad_norm 0.3 -> 1.5 (the first A/B's 0.3 clip bound
every step under rsLoRA and cancelled the treatment; 1.5 is baseline-neutral
because base norms run 0.06-0.16), and hub.checkpoint_repo (a shared repo
means silent last-push-wins clobbering and cross-loading on --resume - the
yaml's own contract). A fourth difference would make the A/B unreadable.
"""

from pathlib import Path

import yaml

from tuned.train.config import load_config

BASE = Path(__file__).parent.parent / "configs" / "law_v1_8b_ddp.yaml"
RSLORA = Path(__file__).parent.parent / "configs" / "law_v1_8b_ddp_rslora_a32.yaml"


def test_base_lane_keeps_rslora_off_and_clip_at_qualified_value():
    # The production yaml must not even carry the key - absence, not "false",
    # is the qualified state - and its clip stays at the QLoRA-paper 0.3.
    raw = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    assert "use_rslora" not in raw["lora"]
    cfg = load_config(BASE, allow_unpinned=True)
    assert cfg.lora.use_rslora is False
    assert cfg.train.max_grad_norm == 0.3


def test_rslora_a32_lane_isolates_rslora_with_an_open_clip():
    cfg = load_config(RSLORA, allow_unpinned=True)
    assert cfg.lora.use_rslora is True
    # alpha must equal the base value - any other alpha re-creates the joint
    # treatment that made the first A/B unattributable.
    assert cfg.lora.alpha == 32
    assert cfg.lora.r == 32
    # 1.5 sits just above the first A/B's observed pre-clip ceiling at 11.3x
    # scale; at 5.66x the norms should clear it, so clipping stops being the
    # treatment.
    assert cfg.train.max_grad_norm == 1.5


def test_rslora_a32_lane_has_its_own_checkpoint_repo():
    base = load_config(BASE, allow_unpinned=True)
    rs = load_config(RSLORA, allow_unpinned=True)
    assert rs.hub.checkpoint_repo != base.hub.checkpoint_repo
    # ...and not the retired alpha-64 arm's repo either - it still holds a
    # stale checkpoint-25 that a --resume would silently cross-load.
    assert rs.hub.checkpoint_repo != "tantan01/tuned-law-v1-qwen8b-ckpt-ddp-rslora"
    assert "rslora" in rs.hub.checkpoint_repo


def test_rslora_a32_lane_differs_in_exactly_three_fields():
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    rs = yaml.safe_load(RSLORA.read_text(encoding="utf-8"))
    rs["lora"].pop("use_rslora")
    rs["train"]["max_grad_norm"] = base["train"]["max_grad_norm"]
    rs["hub"]["checkpoint_repo"] = base["hub"]["checkpoint_repo"]
    assert rs == base
