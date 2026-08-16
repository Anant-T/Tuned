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
        "assembly": {
            "default_profile": "v1.1-full",
            "profiles": {"lean": {"a": 0.7, "b": 0.3}},
            "source_streams": {"src/one": "a", "src/two": "b"},
            "gates": {
                "mix_tolerance_pp": 2.0,
                "trace_floor": 0.8,
                "empty_think_min": 0.18,
                "empty_think_max": 0.22,
                "dup_ceiling": 0.005,
                "markup": True,
                "require_license": True,
                "cross_code_red": False,
                "old_code_sources": ["old/corpus"],
                "require_chain": True,
            },
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


# --- the assembly block -----------------------------------------------------


def test_the_full_profile_is_build_mix_and_is_not_written_twice():
    """One definition of 60/16/24, in build.mix, reachable as a profile.

    A second copy in assembly.profiles would be a fence that can disagree with
    the fencing - and the copy nothing else reads is the one that drifts.
    """
    cfg = load_build_config(DATA_CONFIG, allow_unpinned=True)
    assert cfg.assembly.targets("v1.1-full") == cfg.build.mix
    assert cfg.assembly.targets() == cfg.build.mix  # default_profile is v1.1-full
    raw = yaml.safe_load(DATA_CONFIG.read_text(encoding="utf-8"))
    assert "v1.1-full" not in (raw["assembly"]["profiles"] or {})


def test_restating_the_full_profile_is_refused(tmp_path):
    doc = _base_doc()
    doc["assembly"]["profiles"]["v1.1-full"] = {"a": 0.5, "b": 0.5}
    with pytest.raises(ValueError, match="restates build.mix"):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)


def test_a_build_config_without_an_assembly_block_is_refused(tmp_path):
    doc = _base_doc()
    del doc["assembly"]
    with pytest.raises(ValueError, match="assembly"):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)


def test_the_mvp_profile_is_the_arithmetic_of_the_cut_not_a_preference():
    """v1.0-MVP's targets are DERIVED, and this recomputes the derivation.

    The MVP cut ships every zero-API row plus whatever synthesis exists, so its
    shares are the component counts over mvp_total. Recomputed here from
    build.mix/target_total/mvp_total and cross-checked against the stream
    builders' own totals, because a hand-edited share would otherwise regrade
    the corpus with nothing disagreeing.
    """
    from tuned.data.curated import DEFAULT_COUNTS as CURATED_COUNTS
    from tuned.data.replay import DEFAULT_COUNTS as REPLAY_COUNTS

    cfg = load_build_config(DATA_CONFIG, allow_unpinned=True)
    total, mvp = cfg.build.target_total, cfg.build.mvp_total
    replay = round(total * cfg.build.mix["replay"])
    curated = round(total * cfg.build.mix["curated"])
    # The full-run replay target IS what replay.py builds, which is what makes
    # this a cross-check rather than a restatement.
    assert replay == sum(REPLAY_COUNTS) == 4320
    # The curated BUCKET is bigger than curated.py's zero-API stream: the
    # remainder is the curated_c2 teacher-rewrite slice.
    assert curated == 2880 > sum(CURATED_COUNTS) == 1700
    synthesis = mvp - replay - curated
    assert synthesis == 3100  # transition ~1,100 + the 2,000-row synthesis core

    mvp_targets = cfg.assembly.targets("v1.0-MVP")
    assert mvp_targets == {
        "grounded_synthesis": round(synthesis / mvp, 4),
        "curated": round(curated / mvp, 4),
        "replay": round(replay / mvp, 4),
    }
    # And the MVP cut really is synthesis-light, which is the whole reason it
    # needs its own profile: 30% where the full run wants 60%.
    assert mvp_targets["grounded_synthesis"] < cfg.build.mix["grounded_synthesis"] - 0.25


def test_every_shipped_source_maps_to_a_stream():
    """Every source string the stream builders can emit resolves to a bucket.

    stats.py reds on an unmapped source, so this is the check that keeps that
    gate from firing at the END of a multi-day build over a source that was
    known all along.
    """
    from tuned.data.tasks import PLANNABLE_STREAMS

    cfg = load_build_config(DATA_CONFIG, allow_unpinned=True)
    shipped = [
        "open-thoughts/OpenThoughts-114k",
        "nvidia/Nemotron-Post-Training-Dataset-v2",
        "HuggingFaceTB/smoltalk2:smoltalk_smollm3_smol_magpie_ultra_no_think",
        "HuggingFaceTB/smoltalk2:OpenHermes_2.5",
        "GSMS-B/Indian-Legal-QA-BNS-BNSS-BSA",
        "allenai/WildChat-4.8M",
        "L-NLProc/PredEx_Instruction-Tuning_Pred-Exp",
        "opennyaiorg/aalap_instruction_dataset",
        "169Pi/indian_law",
        *PLANNABLE_STREAMS,
    ]
    unmapped = [s for s in shipped if cfg.assembly.stream_of(s) is None]
    assert unmapped == []
    # The subset half of a replay source is open-ended, so the DATASET half is
    # what carries the mapping - both smoltalk subsets land in replay.
    assert cfg.assembly.stream_of("HuggingFaceTB/smoltalk2:anything-at-all") == "replay"
    # curated_c2 reaches a teacher but counts as CURATED, not as synthesis.
    assert cfg.assembly.stream_of("curated_c2") == "curated"
    assert cfg.assembly.stream_of("synthesis") == "grounded_synthesis"


def test_an_unmapped_source_is_none_and_not_a_default_bucket():
    cfg = load_build_config(DATA_CONFIG, allow_unpinned=True)
    assert cfg.assembly.stream_of("nobody/mapped-this") is None
    assert cfg.assembly.stream_of("") is None
    assert cfg.assembly.stream_of(None) is None


def test_the_whole_source_string_beats_the_dataset_half(tmp_path):
    """A subset CAN be pulled out of its dataset's bucket, and that is what the
    two-step lookup is for - a prefix-only rule would silently ignore the more
    specific line, and a full-string-only rule could not map smoltalk at all."""
    doc = _base_doc()
    doc["assembly"]["source_streams"] = {"ds/one": "a", "ds/one:special": "b"}
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.assembly.stream_of("ds/one:special") == "b"
    assert cfg.assembly.stream_of("ds/one:ordinary") == "a"
    assert cfg.assembly.stream_of("ds/one") == "a"


def test_the_eval_fraction_and_the_length_bucket_are_not_duplicated_here():
    """The two numbers the assembly tail needs that already existed.

    build.held_out_frac had NO reader in src/ before split.py; the length
    bucket is the trainer's max_seq_length. Adding `assembly.eval_fraction` or
    `assembly.max_tokens` would have made two of each, and this fails if
    either one comes back.
    """
    cfg = load_build_config(DATA_CONFIG, allow_unpinned=True)
    train_cfg = load_config(TRAIN_CONFIG, allow_unpinned=True)
    assert cfg.build.held_out_frac == 0.10
    assert cfg.max_seq_length == train_cfg.train.main.max_seq_length == 8192
    raw = yaml.safe_load(DATA_CONFIG.read_text(encoding="utf-8"))
    assert not {"eval_fraction", "max_tokens", "length"} & set(raw["assembly"])


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda d: d["assembly"].__setitem__("default_profile", "ghost"), "default_profile"),
        (lambda d: d["assembly"]["profiles"].__setitem__("lean", {"a": 0.7, "b": 0.5}),
         "sum to 1.0"),
        (lambda d: d["assembly"]["profiles"].__setitem__("lean", {"a": 0.7, "c": 0.3}),
         "same buckets"),
        (lambda d: d["assembly"]["profiles"].__setitem__("lean", {"a": 1.7, "b": -0.7}),
         r"must be in \[0, 1\]"),
        (lambda d: d["assembly"]["source_streams"].__setitem__("src/three", "z"),
         "no profile grades"),
        (lambda d: d["assembly"]["gates"].__setitem__("mix_tolerance_pp", -1.0),
         "mix_tolerance_pp"),
        (lambda d: d["assembly"]["gates"].__setitem__("trace_floor", 1.5), "trace_floor"),
        (lambda d: d["assembly"]["gates"].__setitem__("empty_think_min", 0.5),
         "empty_think_min/max"),
        (lambda d: d["assembly"]["gates"].__setitem__("empty_think_max", 1.5),
         "empty_think_min/max"),
        (lambda d: d["assembly"]["gates"].__setitem__("dup_ceiling", 2.0), "dup_ceiling"),
    ],
)
def test_rule5_assembly_checks_rejected(tmp_path, mutate, match):
    doc = _base_doc()
    mutate(doc)
    with pytest.raises(ValueError, match=match):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)


def test_targets_for_an_unknown_profile_raises_rather_than_defaulting():
    """--profile is a CLI flag, so a typo must not silently grade against the
    default targets and record the profile name it did not use."""
    cfg = load_build_config(DATA_CONFIG, allow_unpinned=True)
    with pytest.raises(KeyError, match="v1.0-mvp"):
        cfg.assembly.targets("v1.0-mvp")  # real profile, wrong case
    assert cfg.assembly.targets("v1.0-MVP")["replay"] == 0.4194


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
