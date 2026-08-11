import sys
from pathlib import Path

import pytest
import yaml

from tuned.data.config import load_build_config
from tuned.train.config import load_config

DATA_CONFIG = Path(__file__).parent.parent / "configs" / "data_law_v1.yaml"
TRAIN_CONFIG = Path(__file__).parent.parent / "configs" / "law_v1_8b_ddp.yaml"

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from pin_dataset import rewrite_dataset_revision  # noqa: E402


# --- happy path -------------------------------------------------------------


def test_happy_path_resolves_fields_from_train_config():
    cfg = load_build_config(DATA_CONFIG, allow_unpinned=True)
    train_cfg = load_config(TRAIN_CONFIG, allow_unpinned=True)
    assert cfg.think_open == "<think>"
    assert cfg.think_close == "</think>"
    assert cfg.model_repo == train_cfg.model.repo
    assert cfg.model_revision == train_cfg.model.revision
    assert cfg.instruction_part == train_cfg.model.instruction_part
    assert cfg.response_part == train_cfg.model.response_part
    assert cfg.main_dataset_path == train_cfg.train.main.dataset


def test_routing_refs_and_model_for_resolve_double_slash_ref():
    cfg = load_build_config(DATA_CONFIG, allow_unpinned=True)
    judge_refs = cfg.routing_refs("judge")
    double_slash = [r for r in judge_refs if r.provider == "groq" and r.model == "qwen/qwen3.6-27b"]
    assert double_slash, judge_refs
    provider, model = cfg.model_for(double_slash[0])
    assert provider.name == "groq"
    assert model.id == "qwen/qwen3.6-27b"
    assert model.family == "qwen"


def test_model_for_unknown_ref_raises_keyerror():
    cfg = load_build_config(DATA_CONFIG, allow_unpinned=True)
    from tuned.data.config import ModelRef

    with pytest.raises(KeyError):
        cfg.model_for(ModelRef("ghost", "nomodel"))


# --- validation fixtures ------------------------------------------------


def _base_doc() -> dict:
    return {
        "build": {
            "train_config": "configs/law_v1_8b_ddp.yaml",
            "workdir": "data/build",
            "target_total": 100,
            "mvp_total": 50,
            "mix": {"a": 0.5, "b": 0.5},
            "overgeneration": 2.0,
            "held_out_frac": 0.1,
            "length_band": {
                "total_max": 8192,
                "total_min": 300,
                "think_min": 500,
                "think_max": 3000,
                "answer_min": 120,
            },
            "difficulty_target": {"easy": 0.5, "hard": 0.5},
            "appointed_day": "2024-07-01",
        },
        "providers": [
            {
                "name": "p1",
                "base_url": "https://p1.example",
                "api_key_env": "P1_KEY",
                "quirks": [],
                "models": [
                    {"id": "gen1", "family": "fam-gen", "roles": ["generator"], "limits": {}, "params": {}},
                ],
            },
            {
                "name": "p2",
                "base_url": "https://p2.example",
                "api_key_env": "P2_KEY",
                "quirks": [],
                "models": [
                    {"id": "judge1", "family": "fam-j1", "roles": ["judge"], "limits": {}, "params": {}},
                    {"id": "judge2", "family": "fam-j2", "roles": ["judge"], "limits": {}, "params": {}},
                    {"id": "tie1", "family": "fam-t1", "roles": ["tiebreak"], "limits": {}, "params": {}},
                    {"id": "probe1", "family": "fam-p1", "roles": ["probe"], "limits": {}, "params": {}},
                ],
            },
        ],
        "routing": {
            "generator": ["p1/gen1"],
            "judge": ["p2/judge1", "p2/judge2"],
            "tiebreak": ["p2/tie1"],
            "probe": ["p2/probe1"],
            "family_separation": True,
            "judge_mode": "dual",
        },
    }


def _write(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def test_base_fixture_is_valid(tmp_path):
    # Sanity-checks the fixture itself so failures below are attributable to
    # the specific mutation, not a broken base doc.
    cfg = load_build_config(_write(tmp_path, _base_doc()), allow_unpinned=True)
    assert cfg.routing.judge_mode == "dual"


def test_rule1_unknown_routing_ref_rejected(tmp_path):
    doc = _base_doc()
    doc["routing"]["probe"] = ["ghost/nomodel"]
    path = _write(tmp_path, doc)
    with pytest.raises(ValueError, match="ghost"):
        load_build_config(path, allow_unpinned=True)


def test_rule2_role_mismatch_rejected(tmp_path):
    doc = _base_doc()
    # gen1 only lists role "generator" - routing it as a tiebreak must fail.
    doc["routing"]["tiebreak"] = ["p1/gen1"]
    path = _write(tmp_path, doc)
    with pytest.raises(ValueError, match="tiebreak"):
        load_build_config(path, allow_unpinned=True)


def test_rule3_insufficient_cross_family_judges_rejected(tmp_path):
    doc = _base_doc()
    # Only one judge family left - family_separation requires >=2 distinct
    # judge families outside the generator's own family.
    doc["routing"]["judge"] = ["p2/judge1"]
    path = _write(tmp_path, doc)
    with pytest.raises(ValueError, match="family"):
        load_build_config(path, allow_unpinned=True)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda d: d["routing"].__setitem__("judge_mode", "trial"), "judge_mode"),
        (lambda d: d["build"].__setitem__("mix", {"a": 0.5, "b": 0.6}), "mix"),
        (lambda d: d["build"].__setitem__("held_out_frac", 0.9), "held_out_frac"),
        (lambda d: d["build"].__setitem__("overgeneration", 0.5), "overgeneration"),
    ],
)
def test_rule4_scalar_sanity_checks_rejected(tmp_path, mutate, match):
    doc = _base_doc()
    mutate(doc)
    path = _write(tmp_path, doc)
    with pytest.raises(ValueError, match=match):
        load_build_config(path, allow_unpinned=True)


# --- HubCfg dataset_* fields ----------------------------------------------


def test_hub_dataset_fields_default_none():
    cfg = load_config(TRAIN_CONFIG, allow_unpinned=True)
    assert cfg.hub.dataset_repo is None
    assert cfg.hub.dataset_revision is None
    assert cfg.hub.dataset_sha256 is None


def test_hub_dataset_fields_load_when_set(tmp_path):
    text = TRAIN_CONFIG.read_text(encoding="utf-8").rstrip("\n") + "\n"
    text += (
        "  dataset_repo: tantan01/tuned-law-v1-data\n"
        "  dataset_revision: deadbeefcafe\n"
        "  dataset_sha256: abc123\n"
    )
    tmp = tmp_path / "c.yaml"
    tmp.write_text(text, encoding="utf-8")
    cfg = load_config(tmp, allow_unpinned=True)
    assert cfg.hub.dataset_repo == "tantan01/tuned-law-v1-data"
    assert cfg.hub.dataset_revision == "deadbeefcafe"
    assert cfg.hub.dataset_sha256 == "abc123"


# --- pin_dataset.py pure rewrite --------------------------------------------


def test_rewrite_dataset_revision_replaces_existing_line():
    text = "hub:\n  checkpoint_repo: x\n  dataset_revision: oldsha\n"
    new = rewrite_dataset_revision(text, "newsha")
    assert "dataset_revision: newsha" in new
    assert "oldsha" not in new


def test_rewrite_dataset_revision_inserts_when_missing():
    text = "hub:\n  checkpoint_repo: x\n"
    new = rewrite_dataset_revision(text, "newsha")
    assert new == "hub:\n  dataset_revision: newsha\n  checkpoint_repo: x\n"
