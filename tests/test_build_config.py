import os
import sys
from pathlib import Path

import pytest
import yaml

from tuned.data.config import (
    CALIBRATION_RULES,
    JUDGE_SCORE_RANGE,
    DifficultyCfg,
    ModelRef,
    TransitionCfg,
    load_build_config,
)
from tuned.train.config import load_config

DATA_CONFIG = Path(__file__).parent.parent / "data" / "configs" / "data_law_v1.yaml"
RECOVERY_CONFIG = Path(__file__).parent.parent / "configs" / "data_law_v1_exp_recovery.yaml"
TRAIN_CONFIG = Path(__file__).parent.parent / "training" / "configs" / "law_v1_8b_ddp.yaml"

sys.path.insert(0, str(Path(__file__).parent.parent / "training" / "scripts"))
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
            "train_config": "training/configs/law_v1_8b_ddp.yaml",
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
                # The floor's complement, which rule 5 now requires - 0.22
                # against a 0.8 floor is the dead band the ruling closed.
                "empty_think_max": 0.20,
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
        # The three share bounds are ONE system under the identity assemble.py
        # enforces, and two combinations of them are unsatisfiable as
        # arithmetic rather than as a corpus.
        (lambda d: d["assembly"]["gates"].__setitem__("trace_floor", 0.9),
         "cannot both be satisfied"),
        (lambda d: d["assembly"]["gates"].__setitem__("empty_think_max", 0.22),
         "DEAD BAND"),
        # A bare string iterates as 16 single characters and the cross-code
        # gate goes silently dead.
        (lambda d: d["assembly"]["gates"].__setitem__("old_code_sources", "169Pi/indian_law"),
         "must be a LIST"),
        (lambda d: d["assembly"]["gates"].__setitem__("old_code_sources", ["ok", 7]),
         "only source strings"),
        (lambda d: d["assembly"]["gates"].__setitem__("old_code_sources", 7),
         "must be a LIST"),
        # bool("false") is True, so a quoted YAML boolean inverts a gate.
        (lambda d: d["assembly"]["gates"].__setitem__("cross_code_red", "false"),
         "cross_code_red must be a YAML boolean"),
        (lambda d: d["assembly"]["gates"].__setitem__("markup", "no"),
         "markup must be a YAML boolean"),
        (lambda d: d["assembly"]["gates"].__setitem__("require_chain", 1),
         "require_chain must be a YAML boolean"),
        (lambda d: d["assembly"]["gates"].__setitem__("require_license", "yes"),
         "require_license must be a YAML boolean"),
        # ...and the same strictness the other way: a toggle is not a threshold.
        (lambda d: d["assembly"]["gates"].__setitem__("trace_floor", True),
         "trace_floor must be a number"),
        (lambda d: d["assembly"]["gates"].__setitem__("dup_ceiling", "half a percent"),
         "dup_ceiling must be a number"),
    ],
)
def test_rule5_assembly_checks_rejected(tmp_path, mutate, match):
    doc = _base_doc()
    mutate(doc)
    with pytest.raises(ValueError, match=match):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)


@pytest.mark.parametrize(
    "key", ["default_profile", "source_streams", "gates"]
)
def test_a_partial_assembly_block_names_the_key_it_is_missing(tmp_path, key):
    """The whole-block-absent path had a good message; the PARTIAL path died
    with a bare `KeyError: 'gates'` from whichever line reached for it first -
    which names the key and nothing about what it is for or why the build
    cannot run without it."""
    doc = _base_doc()
    del doc["assembly"][key]
    with pytest.raises(ValueError, match=f"assembly.{key} is missing"):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)


@pytest.mark.parametrize(
    "key",
    ["mix_tolerance_pp", "trace_floor", "empty_think_min", "empty_think_max",
     "dup_ceiling", "markup", "require_license", "cross_code_red", "require_chain"],
)
def test_every_missing_gate_key_is_a_named_refusal(tmp_path, key):
    doc = _base_doc()
    del doc["assembly"]["gates"][key]
    with pytest.raises(ValueError, match=f"assembly.gates.{key} is missing"):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)


def test_old_code_sources_may_be_absent_or_empty_but_never_a_string(tmp_path):
    """Absent is a legitimate configuration - the gate then fires only on
    `_prov.code_era == "ipc"` - and that is why the type has to be checked
    rather than the truthiness."""
    doc = _base_doc()
    del doc["assembly"]["gates"]["old_code_sources"]
    assert load_build_config(_write(tmp_path, doc),
                             allow_unpinned=True).assembly.gates.old_code_sources == ()
    doc["assembly"]["gates"]["old_code_sources"] = []
    assert load_build_config(_write(tmp_path, doc),
                             allow_unpinned=True).assembly.gates.old_code_sources == ()


def test_the_shipped_gate_bounds_are_a_coherent_system(tmp_path):
    """The two couplings, against the config that actually ships.

    0.80 + 0.18 = 0.98 <= 1, and 0.20 IS 1 - 0.80 - which is where the check
    needs its epsilon, because `1 - 0.80` is 0.19999999999999996 in binary
    floating point and a bare `>` refuses the very pair the rule exists to
    require. Measured here rather than trusted.
    """
    cfg = load_build_config(DATA_CONFIG, allow_unpinned=True)
    gates = cfg.assembly.gates
    assert gates.trace_floor + gates.empty_think_min <= 1.0
    assert gates.empty_think_max == 0.20 and gates.trace_floor == 0.80
    assert 1.0 - gates.trace_floor == 0.19999999999999996  # ...the reason for the slack
    assert gates.empty_think_max > 1.0 - gates.trace_floor  # a bare `>` WOULD refuse it
    assert gates.empty_think_max - (1.0 - gates.trace_floor) < 1e-9


def test_targets_for_an_unknown_profile_raises_rather_than_defaulting():
    """--profile is a CLI flag, so a typo must not silently grade against the
    default targets and record the profile name it did not use."""
    cfg = load_build_config(DATA_CONFIG, allow_unpinned=True)
    with pytest.raises(KeyError, match="v1.0-mvp"):
        cfg.assembly.targets("v1.0-mvp")  # real profile, wrong case
    assert cfg.assembly.targets("v1.0-MVP")["replay"] == 0.4194


# --- HubCfg dataset_* fields ----------------------------------------------


def test_hub_dataset_fields_default_none(tmp_path):
    # Task 15 fix round 1 (M5) gave the SHIPPED config a real dataset_repo -
    # the pin handoff push.py -> scripts/pin_dataset.py needs the value set
    # there. This test is about HubCfg's own default, not that config's
    # content, so it builds a config that legitimately omits the field
    # rather than asserting an absence the shipped file no longer has.
    text = TRAIN_CONFIG.read_text(encoding="utf-8")
    without_dataset_repo = text.replace(
        "  dataset_repo: tantan01/tuned-law-v1-data\n", ""
    )
    assert without_dataset_repo != text
    tmp = tmp_path / "c.yaml"
    tmp.write_text(without_dataset_repo, encoding="utf-8")
    cfg = load_config(tmp, allow_unpinned=True)
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


# --- the transition / calibration / difficulty blocks -----------------------


def _shipped_blocks() -> dict:
    """The three blocks AS SHIPPED, read out of data/configs/data_law_v1.yaml.

    Copied rather than retyped so the validation table below cannot pass
    against a fixture that has drifted from the config an operator actually
    runs - the mutation tests are only attributable while the unmutated base
    is the real thing (test_each_new_block_loads_when_it_is_correct is the
    other direction of that).
    """
    raw = yaml.safe_load(DATA_CONFIG.read_text(encoding="utf-8"))
    return {name: dict(raw[name]) for name in ("transition", "calibration", "difficulty")}


def _t_doc(tmp_path: Path, **blocks) -> Path:
    doc = _base_doc()
    doc.update(blocks)
    return _write(tmp_path, doc)


def test_the_three_new_blocks_are_absent_from_the_minimal_fixture(tmp_path):
    # OPTIONAL like `push:`: a config that never builds the transition grid,
    # never calibrates and never labels difficulty still loads. The three
    # modules refuse by name; the loader does not.
    cfg = load_build_config(_write(tmp_path, _base_doc()), allow_unpinned=True)
    assert cfg.transition is None
    assert cfg.calibration is None
    assert cfg.difficulty is None


def test_the_shipped_config_carries_all_three_blocks():
    cfg = load_build_config(DATA_CONFIG, allow_unpinned=True)
    assert cfg.transition == TransitionCfg(sample=1100, eval_reserve=150)
    assert cfg.calibration.pilot_export == 180
    assert cfg.calibration.holdout == 40
    assert cfg.calibration.folds == 5
    assert cfg.calibration.thresholds == (3, 4, 5)
    assert cfg.calibration.rules == ("min_axis", "mean", "both")
    assert cfg.calibration.min_recall == 0.60
    assert cfg.calibration.min_precision == 0.75
    assert cfg.difficulty == DifficultyCfg(probe_sample=1000, mix_tolerance=0.05)


def test_the_judge_score_range_has_one_definition():
    # config.py cannot import judge.py (judge.py imports config for
    # LengthBand), so the range is stated here and the copy is pinned - the
    # same treatment JUDGE_MAX_TOKENS / DEFAULT_JUDGE_REPLY_TOKENS get. A
    # threshold outside the range fits nothing, and the refusal that says so
    # is only correct while the two agree.
    from tuned.data.judge import SCORE_RANGE

    assert SCORE_RANGE == JUDGE_SCORE_RANGE


def test_the_calibration_rule_vocabulary_is_closed():
    # A rule name that is not in this tuple is refused at LOAD, months before
    # the fit would have quietly skipped it.
    assert CALIBRATION_RULES == ("min_axis", "mean", "both")


@pytest.mark.parametrize(
    "block,mutate,match",
    [
        # transition
        ("transition", lambda b: b.pop("sample"), "transition.sample"),
        ("transition", lambda b: b.pop("eval_reserve"), "transition.eval_reserve"),
        ("transition", lambda b: b.__setitem__("sample", 0), ">= 1"),
        ("transition", lambda b: b.__setitem__("sample", 1.5), "whole number"),
        ("transition", lambda b: b.__setitem__("sample", True), "whole number"),
        ("transition", lambda b: b.__setitem__("eval_reserve", 1100), "smaller than"),
        ("transition", lambda b: b.__setitem__("eval_reserve", 2000), "smaller than"),
        # calibration
        ("calibration", lambda b: b.pop("min_recall"), "calibration.min_recall"),
        ("calibration", lambda b: b.__setitem__("folds", 1), ">= 2"),
        ("calibration", lambda b: b.__setitem__("thresholds", []), "non-empty"),
        ("calibration", lambda b: b.__setitem__("thresholds", "345"), "non-empty"),
        ("calibration", lambda b: b.__setitem__("thresholds", [6]), "score range"),
        ("calibration", lambda b: b.__setitem__("thresholds", [0]), "score range"),
        ("calibration", lambda b: b.__setitem__("rules", ["median"]), "median"),
        ("calibration", lambda b: b.__setitem__("rules", "mean"), "non-empty"),
        ("calibration", lambda b: b.__setitem__("min_recall", 0.0), r"\(0, 1\]"),
        ("calibration", lambda b: b.__setitem__("min_precision", 1.5), r"\(0, 1\]"),
        ("calibration", lambda b: b.__setitem__("holdout", 180), "at least one row per fold"),
        ("calibration", lambda b: b.__setitem__("holdout", 177), "at least one row per fold"),
        # difficulty
        ("difficulty", lambda b: b.pop("probe_sample"), "difficulty.probe_sample"),
        ("difficulty", lambda b: b.__setitem__("mix_tolerance", -0.1), r"\[0, 1\]"),
        ("difficulty", lambda b: b.__setitem__("mix_tolerance", 1.1), r"\[0, 1\]"),
        ("difficulty", lambda b: b.__setitem__("mix_tolerance", "0.05"), "must be a number"),
    ],
)
def test_each_new_block_is_validated_key_by_key(tmp_path, block, mutate, match):
    blocks = _shipped_blocks()
    mutate(blocks[block])
    with pytest.raises(ValueError, match=match):
        load_build_config(_t_doc(tmp_path, **blocks), allow_unpinned=True)


def test_each_new_block_loads_when_it_is_correct(tmp_path):
    # The other direction of the table above: the same three blocks, unmutated,
    # on the same minimal fixture. Without this the refusals could all be
    # firing on the fixture rather than on the mutation.
    cfg = load_build_config(_t_doc(tmp_path, **_shipped_blocks()), allow_unpinned=True)
    assert (cfg.transition.sample, cfg.transition.eval_reserve) == (1100, 150)
    assert cfg.calibration.folds == 5
    assert cfg.difficulty.probe_sample == 1000


@pytest.mark.parametrize("name", ["transition", "calibration", "difficulty"])
def test_a_new_block_that_is_not_a_block_is_refused(tmp_path, name):
    blocks = _shipped_blocks()
    blocks[name] = ["sample", 1100]
    with pytest.raises(ValueError, match=f"`{name}:`"):
        load_build_config(_t_doc(tmp_path, **blocks), allow_unpinned=True)


@pytest.mark.parametrize("name", ["transition", "calibration", "difficulty"])
def test_an_empty_new_block_reads_as_absent_not_as_a_partial_one(tmp_path, name):
    # `transition:` with nothing under it parses as None, which YAML cannot
    # tell from an absent key. Treating it as absent is the only reading that
    # does not depend on a distinction the file format does not carry.
    blocks = _shipped_blocks()
    blocks[name] = None
    cfg = load_build_config(_t_doc(tmp_path, **blocks), allow_unpinned=True)
    assert getattr(cfg, name) is None


def test_thresholds_are_deduped_and_sorted_so_the_sweep_order_is_fixed(tmp_path):
    blocks = _shipped_blocks()
    blocks["calibration"]["thresholds"] = [5, 3, 4, 3]
    blocks["calibration"]["rules"] = ["mean", "min_axis", "mean"]
    cfg = load_build_config(_t_doc(tmp_path, **blocks), allow_unpinned=True)
    assert cfg.calibration.thresholds == (3, 4, 5)
    # Rules keep the operator's ORDER (first-wins ties are broken by it),
    # duplicates removed.
    assert cfg.calibration.rules == ("mean", "min_axis")


def test_the_difficulty_target_is_graded_like_the_mix_it_is(tmp_path):
    doc = _base_doc()
    doc["build"]["difficulty_target"] = {"easy": 0.5, "hard": 0.4}
    with pytest.raises(ValueError, match="difficulty_target"):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)

    doc["build"]["difficulty_target"] = {"easy": 1.2, "hard": -0.2}
    with pytest.raises(ValueError, match=r"difficulty_target.easy"):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)

    doc["build"]["difficulty_target"] = {"easy": 0.34, "medium": 0.50, "hard": 0.16}
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.build.difficulty_target["medium"] == 0.50


def test_the_difficulty_target_is_not_restated_in_the_difficulty_block():
    # Same rule as assembly.profiles' v1.1-full: build.difficulty_target is
    # the ONE definition, and difficulty.py reads it there.
    raw = yaml.safe_load(DATA_CONFIG.read_text(encoding="utf-8"))
    assert "difficulty_target" not in raw["difficulty"]
    assert set(raw["difficulty"]) == {"probe_sample", "mix_tolerance"}


# --- recovery isolation -----------------------------------------------------

_RECOVERY_KNOBS = (
    "harmony_prefill",
    "harmony_completions",
    "harmony_s1_continue",
    "prompt_overlay",
)


def test_live_config_has_no_recovery_or_harmony_experiment_knobs():
    raw = yaml.safe_load(DATA_CONFIG.read_text(encoding="utf-8"))
    for key in _RECOVERY_KNOBS:
        assert key not in raw["build"], key
    assert "require_pretreatment_manifest" not in raw["build"]
    assert "pretreatment_manifest" not in raw["build"]
    cfg = load_build_config(DATA_CONFIG, allow_unpinned=True)
    assert cfg.build.workdir == "data/build"
    assert cfg.build.harmony_prefill is None
    assert cfg.build.harmony_completions is False
    assert cfg.build.harmony_s1_continue is False
    assert cfg.build.prompt_overlay is None
    assert cfg.build.require_pretreatment_manifest is False
    assert cfg.build.pretreatment_manifest is None


_RECOVERY_SPEND_KEYS = ("usd_cap", "usd_per_1m_prompt", "usd_per_1m_completion")


def test_live_openai_limits_carry_a_blocking_cap_not_the_recovery_wallet():
    """The invariant this used to pin - the live OpenAI backstop carries NONE
    of `_RECOVERY_SPEND_KEYS` - stopped holding on 2026-08-27 on purpose: a
    deepseek lead generator made these two judges reachable through
    `family_separation` on a config with no spend cap anywhere, so the live
    block got its OWN usd_cap fence. That is not the leakage this test was
    written against, and the distinction is checkable rather than asserted by
    fiat: the live fence is a HARD BLOCK (usd_cap 0.0, which fails the first
    positive-priced token) where the recovery experiment
    (data_law_v1_exp_recovery.yaml) declares an ENABLING wallet (usd_cap 1.66,
    which lets real spend through up to that amount) on a model relabelled
    `family: gpt-5` for the opposite reason - so it is reachable from a
    gpt-oss row rather than excluded from one. Family staying gpt-oss here is
    still what keeps live family separation excluding these judges from a
    gpt-oss generation.
    """
    cfg = load_build_config(DATA_CONFIG, allow_unpinned=True)
    provider, _ = cfg.model_for(ModelRef("openai", "gpt-5-mini"))
    models = {model.id: model for model in provider.models}
    assert set(models) == {"gpt-5-mini", "gpt-5-nano"}
    for model in models.values():
        assert model.family == "gpt-oss", model.id
        for key in _RECOVERY_SPEND_KEYS:
            assert key in model.limits, (model.id, key)
        assert model.limits["usd_cap"] == 0.0, model.id


def test_recovery_config_is_isolated_and_cerebras_only():
    assert RECOVERY_CONFIG.is_file()
    raw = yaml.safe_load(RECOVERY_CONFIG.read_text(encoding="utf-8"))
    assert raw["build"]["workdir"] == "data/build/exp_recovery"
    assert raw["build"]["prompt_overlay"] == "src/tuned/data/prompts_harmony"
    assert raw["build"]["harmony_completions"] is True
    assert raw["build"]["harmony_s1_continue"] is False
    assert raw["build"]["length_band"]["think_min"] == 500
    cfg = load_build_config(RECOVERY_CONFIG, allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_recovery"
    assert cfg.build.prompt_overlay == "src/tuned/data/prompts_harmony"
    assert cfg.build.harmony_s1_continue is False
    assert cfg.build.length_band.think_min == 500
    assert cfg.routing.generator == ("cerebras/gpt-oss-120b",)
    assert raw["build"]["require_pretreatment_manifest"] is True
    assert raw["build"]["pretreatment_manifest"]
    assert cfg.build.require_pretreatment_manifest is True
    assert cfg.build.pretreatment_manifest == raw["build"]["pretreatment_manifest"]


HARMONY_CONFIG = Path(__file__).parent.parent / "configs" / "data_law_v1_exp_harmony.yaml"
LIVE_PUSH_REPO = "tantan01/tuned-law-v1-data"


def test_harmony_config_does_not_opt_into_pretreatment_manifest():
    raw = yaml.safe_load(HARMONY_CONFIG.read_text(encoding="utf-8"))
    assert "require_pretreatment_manifest" not in raw["build"]
    assert "pretreatment_manifest" not in raw["build"]
    cfg = load_build_config(HARMONY_CONFIG, allow_unpinned=True)
    assert cfg.build.require_pretreatment_manifest is False
    assert cfg.build.pretreatment_manifest is None


def test_recovery_push_target_is_not_the_live_dataset_repo():
    raw = yaml.safe_load(RECOVERY_CONFIG.read_text(encoding="utf-8"))
    assert raw["push"]["repo_id"] != LIVE_PUSH_REPO
    cfg = load_build_config(RECOVERY_CONFIG, allow_unpinned=True)
    live = load_build_config(DATA_CONFIG, allow_unpinned=True)
    harmony = load_build_config(HARMONY_CONFIG, allow_unpinned=True)
    assert cfg.push.repo_id != live.push.repo_id
    assert cfg.push.repo_id != LIVE_PUSH_REPO
    assert live.push.repo_id == LIVE_PUSH_REPO
    assert harmony.push.repo_id == LIVE_PUSH_REPO


def test_require_pretreatment_manifest_must_be_a_yaml_boolean(tmp_path):
    doc = _base_doc()
    doc["build"]["workdir"] = "data/build/exp_recovery"
    doc["build"]["require_pretreatment_manifest"] = "true"
    doc["build"]["pretreatment_manifest"] = "cohort.json"
    with pytest.raises(ValueError, match="require_pretreatment_manifest"):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)


def test_a_recovery_config_that_points_at_the_live_workdir_is_refused(tmp_path):
    doc = _base_doc()
    doc["build"]["workdir"] = "data/build"
    doc["build"]["prompt_overlay"] = "src/tuned/data/prompts_harmony"
    doc["build"]["harmony_completions"] = True
    with pytest.raises(ValueError, match="live workdir"):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)


def test_a_recovery_config_that_resolves_to_the_live_database_is_refused(tmp_path):
    repo_root = Path(__file__).parent.parent.resolve()
    doc = _base_doc()
    doc["build"]["workdir"] = str(repo_root / "data" / "build")
    doc["build"]["harmony_prefill"] = "I start from the facts. "
    with pytest.raises(ValueError, match="live"):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)


def test_a_recovery_config_on_an_isolated_workdir_is_accepted(tmp_path):
    doc = _base_doc()
    doc["build"]["workdir"] = "data/build/exp_recovery"
    doc["build"]["prompt_overlay"] = "src/tuned/data/prompts_harmony"
    doc["build"]["harmony_completions"] = True
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_recovery"


def _recovery_on(workdir: str) -> dict:
    doc = _base_doc()
    doc["build"]["workdir"] = workdir
    doc["build"]["prompt_overlay"] = "src/tuned/data/prompts_harmony"
    doc["build"]["harmony_completions"] = True
    return doc


_LIVE_CONTROL_CHILDREN = (
    "data/build/state",
    "data/build/raw",
    "data/build/corpus",
    "data/build/gold",
    "data/build/logs",
    "data/build/streams",
    "data/build/out",
    "data/build/raw/gen",
)


@pytest.mark.parametrize("workdir", _LIVE_CONTROL_CHILDREN)
def test_a_recovery_config_that_points_at_a_live_control_child_is_refused(
    tmp_path, workdir
):
    with pytest.raises(ValueError, match="live"):
        load_build_config(_write(tmp_path, _recovery_on(workdir)), allow_unpinned=True)


@pytest.mark.parametrize(
    "workdir",
    [
        "data/build/../build",
        "data/build/./state",
        "data/build/../build/state",
        str(Path("data") / "build" / "state"),
        *([r"data\build\state", r"data\build\raw", "DATA/BUILD", "data/BUILD/State"]
          if os.name == "nt"
          else []),
    ],
)
def test_recovery_refuses_canonical_aliases_of_the_live_control(tmp_path, workdir):
    with pytest.raises(ValueError, match="live"):
        load_build_config(_write(tmp_path, _recovery_on(workdir)), allow_unpinned=True)


@pytest.mark.parametrize(
    "workdir",
    [
        "data/build/exp_recovery",
        "data/build/exp_harmony",
        "data/build/exp_s1",
        "data/build/exp_measure",
    ],
)
def test_isolated_experiment_siblings_are_not_treated_as_live_control(
    tmp_path, workdir
):
    cfg = load_build_config(_write(tmp_path, _recovery_on(workdir)), allow_unpinned=True)
    assert cfg.build.workdir == workdir


def test_build_paths_resolves_relative_workdir_against_the_repo_not_cwd(
    tmp_path, monkeypatch
):
    from tuned.data.paths import build_paths

    repo_root = Path(__file__).resolve().parent.parent
    monkeypatch.chdir(tmp_path)
    paths = build_paths("data/build/exp_recovery")
    expected = (repo_root / "data" / "build" / "exp_recovery").resolve()
    assert paths.root.resolve() == expected
    assert not (tmp_path / "data" / "build").exists()


def test_build_paths_leaves_absolute_runtime_paths_alone(tmp_path):
    from tuned.data.paths import build_paths

    root = tmp_path / "build"
    assert build_paths(root).root == root


def test_refusal_and_build_paths_agree_on_a_relative_live_child(tmp_path, monkeypatch):
    from tuned.data.paths import build_paths

    repo_root = Path(__file__).resolve().parent.parent
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="live"):
        load_build_config(
            _write(tmp_path, _recovery_on("data/build/state")), allow_unpinned=True
        )
    # Same string the yaml would have used: BuildPaths must land on the live
    # control tree under the repo, not cwd/data/build/state.
    paths = build_paths("data/build/state")
    live_state = repo_root / "data" / "build" / "state"
    assert paths.root.resolve() == live_state.resolve()
    assert not (tmp_path / "data" / "build").exists()


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


def test_recovery_openai_cap_is_the_remaining_headroom_not_a_fresh_two_dollars():
    """budget_ledger is per-store, so a new store resets usd_cap.

    exp_harmony already spent $0.3396 of the operator's $2.00 total
    (gpt-5-mini, 124 requests, 377,537 prompt / 122,607 completion tokens).
    The recovery yaml must declare the REMAINDER, or the arm silently
    authorises a second full wallet.
    """
    cfg = load_build_config(
        RECOVERY_CONFIG, allow_unpinned=True
    )
    for name in ("gpt-5-mini", "gpt-5-nano"):
        _provider, model = cfg.model_for(ModelRef("openai", name))
        assert model.limits["usd_cap"] == 1.66, (
            f"openai/{name} declares usd_cap {model.limits['usd_cap']!r}; the "
            "$2.00 operator total already has $0.3396 spent in exp_harmony"
        )


def test_gpt5_judge_and_tiebreak_calls_carry_minimal_reasoning_effort():
    """gpt-5 bills reasoning against max_completion_tokens.

    With params {} the model spent its whole 1,024-token reply budget
    thinking and returned empty content on 95 of 96 parse failures in
    exp_harmony. JUDGE_MAX_TOKENS cannot be the answer - it is fleet-wide,
    and build_payload assigns max_tokens AFTER the role_params merge - so
    the effort knob is the fix, as judge.py's own comment says.
    """
    import httpx

    from tuned.data.providers import ChatClient, ChatRequest

    def _never_called(request):  # build_payload performs no request
        raise AssertionError("build_payload must not perform a request")

    cfg = load_build_config(
        RECOVERY_CONFIG, allow_unpinned=True
    )
    for name in ("gpt-5-mini", "gpt-5-nano"):
        ref = ModelRef("openai", name)
        provider, model = cfg.model_for(ref)
        for role in ("judge", "tiebreak"):
            assert model.role_params[role]["reasoning_effort"] == "minimal", (
                f"openai/{name} role_params[{role!r}] does not pin "
                "reasoning_effort"
            )
            client = ChatClient(
                provider, model, transport=httpx.MockTransport(_never_called)
            )
            payload = client.build_payload(
                ChatRequest(
                    messages=({"role": "user", "content": "score this"},),
                    ref=ref,
                    role=role,
                    max_tokens=1024,
                )
            )
            assert payload["reasoning_effort"] == "minimal"
            # the openai quirk renames the allowance and drops temperature
            assert payload["max_completion_tokens"] == 1024
            assert "max_tokens" not in payload
            assert "temperature" not in payload


def test_eval_cohort_strata_defaults_to_none_on_the_live_config():
    cfg = load_build_config("data/configs/data_law_v1.yaml", allow_unpinned=True)
    assert cfg.build.eval_cohort_strata is None


def test_recovery_config_declares_three_strata_as_a_tuple():
    cfg = load_build_config(
        "configs/data_law_v1_exp_recovery.yaml", allow_unpinned=True
    )
    assert cfg.build.eval_cohort_strata == (
        "irac_analysis",
        "drafting",
        "summarization",
    )


def test_eval_cohort_strata_refuses_empty_duplicate_and_non_string(tmp_path):
    """A stratum list is a pre-registration, so it is validated at load.

    An empty list, a repeat, or a non-string is a typo that would otherwise
    reach the cohort selector and silently change the cohort's size.
    """
    import yaml as _yaml

    base = _yaml.safe_load(
        Path("configs/data_law_v1_exp_recovery.yaml").read_text(encoding="utf-8")
    )
    for bad in ([], ["irac_analysis", "irac_analysis"], ["irac_analysis", 7], "drafting"):
        doc = dict(base)
        doc["build"] = dict(base["build"])
        doc["build"]["eval_cohort_strata"] = bad
        path = tmp_path / "bad.yaml"
        path.write_text(_yaml.safe_dump(doc), encoding="utf-8")
        with pytest.raises(ValueError, match="eval_cohort_strata"):
            load_build_config(path, allow_unpinned=True)


def test_the_deepseek_arm_is_an_isolated_workdir(tmp_path):
    """data/build/exp_deepseek is an experiment sibling, not the live control.

    The one-line fence that makes everything else in the deepseek arm
    possible: is_live_control_workdir is what load_build_config and the
    write guards consult, and an unlisted name under data/build reads as
    the frozen control.
    """
    from tuned.data.paths import is_live_control_workdir

    assert is_live_control_workdir("data/build/exp_deepseek") is False
    assert is_live_control_workdir("data/build") is True
    # A recovery-capable doc on the new sibling loads; on the live root it
    # is refused. Same shape as the exp_recovery tests above.
    doc = _base_doc()
    doc["build"]["workdir"] = "data/build/exp_deepseek"
    doc["build"]["harmony_prefill"] = "I start from the facts. "
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_deepseek"


def test_the_prompt_v5_arm_is_an_isolated_workdir(tmp_path):
    """data/build/exp_prompt_v5 is an experiment sibling, not the live control.

    The treatment half of the 2026-08-27 prompt-length A/B. Same one-line
    fence as the deepseek arm above: an unlisted name under data/build reads
    as the frozen control, and every write guard in the tree would then
    refuse the arm - or, worse, aim it at the control store.
    """
    from tuned.data.paths import is_live_control_workdir

    assert is_live_control_workdir("data/build/exp_prompt_v5") is False
    assert is_live_control_workdir("data/build") is True
    doc = _base_doc()
    doc["build"]["workdir"] = "data/build/exp_prompt_v5"
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_prompt_v5"


DEEPSEEK_CONFIG = Path(__file__).parent.parent / "configs" / "data_law_v1_exp_deepseek.yaml"


def test_the_deepseek_arm_config_is_fenced(tmp_path):
    """The two holes the live config has that an experiment arm may not.

    1. The live generator list falls over bai -> cerebras/gpt-oss -> paid
       lightning; a 429 storm would turn a deepseek arm into a gpt-oss arm
       without anything noticing. The arm pins the single ref.
    2. The live config declares no openai usd_cap, which _provider_usd_cap
       reads as UNCAPPED, and gpt-5-mini/nano sit in judge and tiebreak as
       backstops. usd_cap 0.0 ALONE does not block either - _usd_per_1m
       returns 0.0 for a missing price, so est_cost is 0 and 0 > 0.0 is
       False. The price must be present too (exp_measure's precedent).
    Plus: none of the Harmony flags, which are gpt-oss's chat format.
    """
    import yaml

    from tuned.data.generate import _provider_usd_cap

    cfg = load_build_config(DEEPSEEK_CONFIG, allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_deepseek"
    assert list(cfg.routing.generator) == ["bai/deepseek-v4-flash"]
    assert _provider_usd_cap(cfg, "openai") == 0.0
    openai = next(p for p in cfg.providers if p.name == "openai")
    for model in openai.models:
        assert model.limits["usd_cap"] == 0.0
        assert model.limits["usd_per_1m_prompt"] > 0
        assert model.limits["usd_per_1m_completion"] > 0
    raw = yaml.safe_load(DEEPSEEK_CONFIG.read_text(encoding="utf-8"))
    for key in ("harmony_completions", "harmony_prefill", "harmony_s1_continue",
                "prompt_overlay", "require_pretreatment_manifest", "pretreatment_manifest"):
        assert key not in raw["build"], key
    # The judge/tiebreak order used to be asserted as EQUAL to the live
    # config's. That made a committed test depend on a file's uncommitted
    # working state - data/configs/data_law_v1.yaml carries the bai block and the
    # reordered tiebreak as unstaged edits, so the equality passes here and
    # fails on anyone else's checkout. The INVARIANT is asserted directly on
    # the arm config instead, and it is the one this arm's header spells out:
    #
    # on a DEEPSEEK generation family separation excludes {deepseek, qwen,
    # gemma}, which makes groq/openai/gpt-oss-20b eligible for the tiebreak
    # seat for the first time - and with three verdicts that seat decides
    # outright (judge_policy.resolve). mistral must therefore come BEFORE
    # gpt-oss-20b, or every contested row goes to the family measured 0/10 on
    # IPC->BNS mapping.
    tiebreak = list(cfg.routing.tiebreak)
    assert tiebreak.index("mistral/mistral-large-latest") < tiebreak.index(
        "groq/openai/gpt-oss-20b"
    ), tiebreak
    # ...and the two slots the dual judge actually fills are the free qwen and
    # gemma families, ahead of the $0-fenced openai backstops.
    assert list(cfg.routing.judge)[:2] == ["groq/qwen/qwen3.6-27b", "cerebras/gemma-4-31b"]
    bai = next(p for p in cfg.providers if p.name == "bai")
    assert bai.models[0].params["reasoning_effort"] == "low"


PROMPT_V5_CONFIG = (
    Path(__file__).parent.parent / "configs" / "data_law_v1_exp_prompt_v5.yaml"
)


def test_the_prompt_v5_arm_config_is_fenced_and_matches_its_control(tmp_path):
    """The treatment arm carries both of the deepseek arm's cost fences, and
    differs from it ONLY in the workdir and the header.

    Two separate properties, both load-bearing for the A/B:

    1. The fences. Same two holes as the control arm - the single-ref
       generator (a 429 storm must not silently turn this into a gpt-oss
       arm) and usd_cap 0.0 WITH prices on both gpt-5 models (a bare
       usd_cap 0.0 blocks nothing: _usd_per_1m returns 0.0 for a missing
       price, so est_cost is 0 and 0 > 0.0 is False).
    2. The pairing. Control and treatment are a matched pair whose only
       intended difference is the prompt text on disk. Any third difference
       between the two config files silently confounds the comparison, so
       the byte-level equality is asserted rather than trusted. If a future
       edit to either arm is deliberate, re-pair them here on purpose.
    """
    from tuned.data.generate import _provider_usd_cap

    cfg = load_build_config(PROMPT_V5_CONFIG, allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_prompt_v5"
    assert list(cfg.routing.generator) == ["bai/deepseek-v4-flash"]
    assert _provider_usd_cap(cfg, "openai") == 0.0
    openai = next(p for p in cfg.providers if p.name == "openai")
    for model in openai.models:
        assert model.limits["usd_cap"] == 0.0
        assert model.limits["usd_per_1m_prompt"] > 0
        assert model.limits["usd_per_1m_completion"] > 0

    def _body(path: Path) -> list[str]:
        lines = path.read_text(encoding="utf-8").splitlines()
        first_key = next(i for i, ln in enumerate(lines) if not ln.startswith("#"))
        return [ln for ln in lines[first_key:] if not ln.startswith("  workdir:")]

    assert _body(PROMPT_V5_CONFIG) == _body(DEEPSEEK_CONFIG)
    assert PROMPT_V5_CONFIG.read_bytes().count(b"\r") == 0


def test_the_gptoss_control_arm_is_an_isolated_workdir(tmp_path):
    """data/build/exp_gptoss_ctl is an experiment sibling, not the live control.

    The pre-edit-prompt half of the 2026-08-27 gpt-oss floor A/B. Same
    one-line fence as the arms above, with one extra thing to prove: this
    arm carries `prompt_overlay`, which makes it a RECOVERY experiment
    (config._is_recovery_experiment), and a recovery config aimed at the
    live workdir is refused outright. An unlisted name under data/build
    would therefore not merely read as the frozen control - it would make
    the arm unloadable.
    """
    from tuned.data.paths import is_live_control_workdir

    assert is_live_control_workdir("data/build/exp_gptoss_ctl") is False
    assert is_live_control_workdir("data/build") is True
    doc = _base_doc()
    doc["build"]["workdir"] = "data/build/exp_gptoss_ctl"
    doc["build"]["prompt_overlay"] = "data/build/exp_gptoss_ctl/prompts_preedit"
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_gptoss_ctl"
    assert cfg.build.prompt_overlay == "data/build/exp_gptoss_ctl/prompts_preedit"


def test_the_gptoss_treatment_arm_is_an_isolated_workdir(tmp_path):
    """data/build/exp_gptoss_new is an experiment sibling, not the live control.

    The shipped-prompt half of the same A/B. It sets no overlay - it reads
    the same src/tuned/data/prompts/ bytes the live config reads, which is
    the whole point of it - so the only fence it needs is this one.
    """
    from tuned.data.paths import is_live_control_workdir

    assert is_live_control_workdir("data/build/exp_gptoss_new") is False
    assert is_live_control_workdir("data/build") is True
    doc = _base_doc()
    doc["build"]["workdir"] = "data/build/exp_gptoss_new"
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_gptoss_new"


GPTOSS_CTL_CONFIG = (
    Path(__file__).parent.parent / "configs" / "data_law_v1_exp_gptoss_ctl.yaml"
)
GPTOSS_NEW_CONFIG = (
    Path(__file__).parent.parent / "configs" / "data_law_v1_exp_gptoss_new.yaml"
)


def test_the_gptoss_arms_are_fenced_and_differ_only_in_the_prompt_overlay():
    """Both gpt-oss arms carry both cost fences, and differ in ONE key.

    Three properties, all load-bearing for the A/B:

    1. The single-ref generator. The live list falls over cerebras ->
       bai/deepseek -> paid lightning, so a 429 storm would turn a gpt-oss
       arm into a DEEPSEEK arm - a different family with a very different
       reasoning length, silently confounding the one number this A/B
       exists to produce.
    2. usd_cap 0.0 WITH prices on both gpt-5 models. A bare usd_cap 0.0
       blocks nothing: _usd_per_1m returns 0.0 for a missing price, so
       est_cost is 0 and 0 > 0.0 is False. The price must be present too.
    3. The pairing. The two arms are a matched pair whose ONLY intended
       difference is build.prompt_overlay - the control reads the pre-edit
       templates, the treatment reads the shipped ones. Any third
       difference confounds the comparison, so the line-level equality is
       asserted rather than trusted.

    The band is asserted equal to the LIVE config's rather than to
    literals: both arms must grade against exactly what production grades
    against, and a band that drifted from the live file would make the
    measured breach rate answer a question nobody asked.
    """
    import yaml

    from tuned.data.generate import _provider_usd_cap

    live = load_build_config(
        Path(__file__).parent.parent / "data" / "configs" / "data_law_v1.yaml",
        allow_unpinned=True,
    )
    for path, workdir, overlay in (
        (GPTOSS_CTL_CONFIG, "data/build/exp_gptoss_ctl",
         "data/build/exp_gptoss_ctl/prompts_preedit"),
        (GPTOSS_NEW_CONFIG, "data/build/exp_gptoss_new", None),
    ):
        cfg = load_build_config(path, allow_unpinned=True)
        assert cfg.build.workdir == workdir
        assert cfg.build.prompt_overlay == overlay
        assert list(cfg.routing.generator) == ["cerebras/gpt-oss-120b"]
        assert _provider_usd_cap(cfg, "openai") == 0.0
        openai = next(p for p in cfg.providers if p.name == "openai")
        for model in openai.models:
            assert model.limits["usd_cap"] == 0.0
            assert model.limits["usd_per_1m_prompt"] > 0
            assert model.limits["usd_per_1m_completion"] > 0
        assert cfg.build.length_band == live.build.length_band
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key in ("harmony_completions", "harmony_prefill", "harmony_s1_continue",
                    "require_pretreatment_manifest", "pretreatment_manifest"):
            assert key not in raw["build"], key
        assert path.read_bytes().count(b"\r") == 0

    def _body(path: Path) -> list[str]:
        lines = path.read_text(encoding="utf-8").splitlines()
        first_key = next(i for i, ln in enumerate(lines) if not ln.startswith("#"))
        return [ln for ln in lines[first_key:]
                if not ln.startswith("  workdir:")
                and not ln.startswith("  prompt_overlay:")]

    assert _body(GPTOSS_CTL_CONFIG) == _body(GPTOSS_NEW_CONFIG)


def test_the_ds_v4rerun_arm_is_an_isolated_workdir(tmp_path):
    """data/build/exp_ds_v4rerun is an experiment sibling, not the live control.

    The pre-edit-prompt half of the 2026-08-28 clean rerun of the v4-vs-v5
    deepseek generation-yield A/B (the original ran its two arms 13h41m apart
    across b.ai's hidden multi-upstream pool and is uninterpretable). Same
    fence as the gpt-oss floor arms above, with the same extra thing to
    prove: this arm carries `prompt_overlay`, which makes it a RECOVERY
    experiment (config._is_recovery_experiment), and a recovery config aimed
    at the live workdir is refused outright.
    """
    from tuned.data.paths import is_live_control_workdir

    assert is_live_control_workdir("data/build/exp_ds_v4rerun") is False
    assert is_live_control_workdir("data/build") is True
    doc = _base_doc()
    doc["build"]["workdir"] = "data/build/exp_ds_v4rerun"
    doc["build"]["prompt_overlay"] = "data/build/exp_gptoss_ctl/prompts_preedit"
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_ds_v4rerun"
    assert cfg.build.prompt_overlay == "data/build/exp_gptoss_ctl/prompts_preedit"


def test_the_ds_v5rerun_arm_is_an_isolated_workdir(tmp_path):
    """data/build/exp_ds_v5rerun is an experiment sibling, not the live control.

    The shipped-prompt half of the same rerun. It sets no overlay - it reads
    the same src/tuned/data/prompts/ bytes the live config reads, which is
    the whole point of it - so the only fence it needs is this one.
    """
    from tuned.data.paths import is_live_control_workdir

    assert is_live_control_workdir("data/build/exp_ds_v5rerun") is False
    assert is_live_control_workdir("data/build") is True
    doc = _base_doc()
    doc["build"]["workdir"] = "data/build/exp_ds_v5rerun"
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_ds_v5rerun"


DS_V4RERUN_CONFIG = (
    Path(__file__).parent.parent / "configs" / "data_law_v1_exp_ds_v4rerun.yaml"
)
DS_V5RERUN_CONFIG = (
    Path(__file__).parent.parent / "configs" / "data_law_v1_exp_ds_v5rerun.yaml"
)


def test_the_ds_rerun_arms_are_fenced_and_differ_only_in_the_prompt_overlay():
    """Both deepseek-rerun arms carry both cost fences, and differ in ONE key.

    This is the clean re-run of the confounded v4-vs-v5 comparison (original
    result: v4 pass 49.5% n=99 vs v5 31.9% n=94, arms 13h41m apart across
    b.ai's hidden multi-upstream pool - uninterpretable). Four properties, all
    load-bearing:

    1. The single-ref generator, deepseek only - a 429 storm falling over to
       another family would silently confound the length-yield measurement
       this rerun exists to produce.
    2. usd_cap 0.0 WITH prices on both gpt-5 models. A bare usd_cap 0.0
       blocks nothing: _usd_per_1m returns 0.0 for a missing price, so
       est_cost is 0 and 0 > 0.0 is False. The price must be present too.
    3. The length_band matches the shipped live config's - this measures the
       generation-time length_band gate under two prompt eras, not a
       different band.
    4. The pairing. The two arms are a matched pair whose ONLY intended
       difference is build.prompt_overlay - v4rerun reads the pre-edit
       templates (REUSED from data/build/exp_gptoss_ctl/prompts_preedit
       rather than duplicated), v5rerun reads the shipped ones. Any third
       difference confounds the comparison, so the line-level equality is
       asserted rather than trusted.
    """
    import yaml

    from tuned.data.generate import _provider_usd_cap

    live = load_build_config(
        Path(__file__).parent.parent / "data" / "configs" / "data_law_v1.yaml",
        allow_unpinned=True,
    )
    for path, workdir, overlay in (
        (DS_V4RERUN_CONFIG, "data/build/exp_ds_v4rerun",
         "data/build/exp_gptoss_ctl/prompts_preedit"),
        (DS_V5RERUN_CONFIG, "data/build/exp_ds_v5rerun", None),
    ):
        cfg = load_build_config(path, allow_unpinned=True)
        assert cfg.build.workdir == workdir
        assert cfg.build.prompt_overlay == overlay
        assert list(cfg.routing.generator) == ["bai/deepseek-v4-flash"]
        assert _provider_usd_cap(cfg, "openai") == 0.0
        openai = next(p for p in cfg.providers if p.name == "openai")
        for model in openai.models:
            assert model.limits["usd_cap"] == 0.0
            assert model.limits["usd_per_1m_prompt"] > 0
            assert model.limits["usd_per_1m_completion"] > 0
        assert cfg.build.length_band == live.build.length_band
        assert cfg.build.length_band.think_max == 4000
        assert cfg.build.length_band.total_max == 8192
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key in ("harmony_completions", "harmony_prefill", "harmony_s1_continue",
                    "require_pretreatment_manifest", "pretreatment_manifest"):
            assert key not in raw["build"], key
        assert path.read_bytes().count(b"\r") == 0
        bai = next(p for p in cfg.providers if p.name == "bai")
        assert bai.models[0].limits["rpm"] == 8

    def _body(path: Path) -> list[str]:
        lines = path.read_text(encoding="utf-8").splitlines()
        first_key = next(i for i, ln in enumerate(lines) if not ln.startswith("#"))
        return [ln for ln in lines[first_key:]
                if not ln.startswith("  workdir:")
                and not ln.startswith("  prompt_overlay:")]

    assert _body(DS_V4RERUN_CONFIG) == _body(DS_V5RERUN_CONFIG)


def test_the_ds_ctl2_arm_is_an_isolated_workdir(tmp_path):
    """data/build/exp_ds_ctl2 is an experiment sibling, not the live control.

    The shared control for the 2026-08-28 three-arm clause/cap A/B: two
    independent levers (a prompt-overlay clause, a lowered bai max_output)
    are each measured against this ONE time-local control, run back to
    back, so neither measurement is confounded by b.ai's hidden
    multi-upstream pool drifting between arms - the lesson of the 13h41m
    v4-vs-v5 confound this rerun family already paid for once. This arm
    sets no overlay and no limit override, so it reads the current shipped
    src/tuned/data/prompts/ templates and the live max_output.
    """
    from tuned.data.paths import is_live_control_workdir

    assert is_live_control_workdir("data/build/exp_ds_ctl2") is False
    assert is_live_control_workdir("data/build") is True
    doc = _base_doc()
    doc["build"]["workdir"] = "data/build/exp_ds_ctl2"
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_ds_ctl2"
    assert cfg.build.prompt_overlay is None


def test_the_ds_clause_arm_is_an_isolated_workdir(tmp_path):
    """data/build/exp_ds_clause is an experiment sibling, not the live control.

    The clause-treatment half of the three-arm A/B. Carries `prompt_overlay`,
    which makes it a RECOVERY experiment (config._is_recovery_experiment),
    and a recovery config aimed at the live workdir is refused outright - so
    an unlisted name under data/build would make the arm unloadable, not
    merely misread as the frozen control.
    """
    from tuned.data.paths import is_live_control_workdir

    assert is_live_control_workdir("data/build/exp_ds_clause") is False
    assert is_live_control_workdir("data/build") is True
    doc = _base_doc()
    doc["build"]["workdir"] = "data/build/exp_ds_clause"
    doc["build"]["prompt_overlay"] = "data/build/exp_ds_clause/prompts_clause"
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_ds_clause"
    assert cfg.build.prompt_overlay == "data/build/exp_ds_clause/prompts_clause"


def test_the_ds_cap_arm_is_an_isolated_workdir(tmp_path):
    """data/build/exp_ds_cap is an experiment sibling, not the live control.

    The cap-treatment half of the three-arm A/B. Sets no overlay - the lever
    under test here is the bai model's max_output limit, not the prompts -
    so the only fence it needs is this one.
    """
    from tuned.data.paths import is_live_control_workdir

    assert is_live_control_workdir("data/build/exp_ds_cap") is False
    assert is_live_control_workdir("data/build") is True
    doc = _base_doc()
    doc["build"]["workdir"] = "data/build/exp_ds_cap"
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_ds_cap"
    assert cfg.build.prompt_overlay is None


def test_the_hy3_arm_is_an_isolated_workdir(tmp_path):
    """data/build/exp_hy3 is an experiment sibling, not the live control.

    The 2026-08-28 hy3 think_low qualification probe: can b.ai's free `hy3`
    model (Tencent Hunyuan) serve as a second generator family, tested at
    its documented LOW thinking tier. Isolated the same way every other
    single-arm probe this week was - no overlay, no shared state with the
    live control.
    """
    from tuned.data.paths import is_live_control_workdir

    assert is_live_control_workdir("data/build/exp_hy3") is False
    assert is_live_control_workdir("data/build") is True
    doc = _base_doc()
    doc["build"]["workdir"] = "data/build/exp_hy3"
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_hy3"
    assert cfg.build.prompt_overlay is None


DS_CTL2_CONFIG = (
    Path(__file__).parent.parent / "configs" / "data_law_v1_exp_ds_ctl2.yaml"
)
DS_CLAUSE_CONFIG = (
    Path(__file__).parent.parent / "configs" / "data_law_v1_exp_ds_clause.yaml"
)
DS_CAP_CONFIG = (
    Path(__file__).parent.parent / "configs" / "data_law_v1_exp_ds_cap.yaml"
)


def _assert_ds_ab_common_fences(cfg, path: Path, live) -> None:
    """The four invariants every arm of the 2026-08-28 clause/cap A/B shares:
    single-ref deepseek generator, the openai cost fence, the live band, and
    LF-only bytes. Factored out so both pairing tests below check the same
    ground before comparing the one key each is allowed to differ on."""
    import yaml

    from tuned.data.generate import _provider_usd_cap

    assert list(cfg.routing.generator) == ["bai/deepseek-v4-flash"]
    assert _provider_usd_cap(cfg, "openai") == 0.0
    openai = next(p for p in cfg.providers if p.name == "openai")
    for model in openai.models:
        assert model.limits["usd_cap"] == 0.0
        assert model.limits["usd_per_1m_prompt"] > 0
        assert model.limits["usd_per_1m_completion"] > 0
    assert cfg.build.length_band == live.build.length_band
    assert cfg.build.length_band.think_max == 4000
    assert cfg.build.length_band.total_max == 8192
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("harmony_completions", "harmony_prefill", "harmony_s1_continue",
                "require_pretreatment_manifest", "pretreatment_manifest"):
        assert key not in raw["build"], key
    assert path.read_bytes().count(b"\r") == 0


def test_the_ds_ctl2_and_clause_arms_are_fenced_and_differ_only_in_the_prompt_overlay():
    """The shared control and the clause treatment differ in ONE key.

    E2 of the 2026-08-28 clause/cap A/B: does one added sentence, folding the
    model's own "have I covered all four parts" self-check into the prose it
    is already writing rather than a labelled Issue/Rule/Application/
    Conclusion rehearsal inside <think>, cut the irac_placement gate failure
    this rehearsal causes (E2-offline: 122/123 of that gate's failures).
    Same three properties as the v4/v5 rerun pairing test above - single-ref
    generator, the openai cost fence, the live band - plus the pairing
    itself: prompt_overlay is the ONLY intended difference from the control,
    so line-level equality is asserted rather than trusted.
    """
    live = load_build_config(
        Path(__file__).parent.parent / "data" / "configs" / "data_law_v1.yaml",
        allow_unpinned=True,
    )
    for path, workdir, overlay in (
        (DS_CTL2_CONFIG, "data/build/exp_ds_ctl2", None),
        (DS_CLAUSE_CONFIG, "data/build/exp_ds_clause",
         "data/build/exp_ds_clause/prompts_clause"),
    ):
        cfg = load_build_config(path, allow_unpinned=True)
        assert cfg.build.workdir == workdir
        assert cfg.build.prompt_overlay == overlay
        _assert_ds_ab_common_fences(cfg, path, live)
        bai = next(p for p in cfg.providers if p.name == "bai")
        assert bai.models[0].limits["rpm"] == 8
        assert bai.models[0].limits["max_output"] == 16384

    def _body(path: Path) -> list[str]:
        lines = path.read_text(encoding="utf-8").splitlines()
        first_key = next(i for i, ln in enumerate(lines) if not ln.startswith("#"))
        return [ln for ln in lines[first_key:]
                if not ln.startswith("  workdir:")
                and not ln.startswith("  prompt_overlay:")]

    assert _body(DS_CTL2_CONFIG) == _body(DS_CLAUSE_CONFIG)


def test_the_ds_ctl2_and_cap_arms_are_fenced_and_differ_only_in_the_bai_max_output():
    """The shared control and the cap treatment differ in ONE key.

    E1 of the 2026-08-28 clause/cap A/B: does lowering the bai model's reply
    ceiling 16384 -> 5000 raise real completion tokens per length-passing
    row without moving the length_band pass rate. Neither arm carries a
    prompt_overlay - the lever under test is the limit, not the prompts -
    so the pairing is asserted on the bai model's max_output specifically,
    and line-level equality on everything else.
    """
    live = load_build_config(
        Path(__file__).parent.parent / "data" / "configs" / "data_law_v1.yaml",
        allow_unpinned=True,
    )
    expected_max_output = {"data/build/exp_ds_ctl2": 16384, "data/build/exp_ds_cap": 5000}
    for path, workdir in (
        (DS_CTL2_CONFIG, "data/build/exp_ds_ctl2"),
        (DS_CAP_CONFIG, "data/build/exp_ds_cap"),
    ):
        cfg = load_build_config(path, allow_unpinned=True)
        assert cfg.build.workdir == workdir
        assert cfg.build.prompt_overlay is None
        _assert_ds_ab_common_fences(cfg, path, live)
        bai = next(p for p in cfg.providers if p.name == "bai")
        assert bai.models[0].limits["rpm"] == 8
        assert bai.models[0].limits["max_output"] == expected_max_output[workdir]

    # The bai limits line is the only one this pairing may differ on -
    # matched by its unique prefix (rpm 8 + max_context 800000 belong to no
    # other model block) rather than by the max_output value itself, so a
    # coincidental "max_output: 16384" on an unrelated model (groq/qwen,
    # both openai gpt-5 models) is never mistaken for the lever under test.
    bai_limits_prefix = "        limits: {rpm: 8, max_context: 800000, max_output:"

    def _body(path: Path) -> list[str]:
        lines = path.read_text(encoding="utf-8").splitlines()
        first_key = next(i for i, ln in enumerate(lines) if not ln.startswith("#"))
        return [ln for ln in lines[first_key:]
                if not ln.startswith("  workdir:")
                and not ln.startswith(bai_limits_prefix)]

    assert _body(DS_CTL2_CONFIG) == _body(DS_CAP_CONFIG)


HY3_CONFIG = Path(__file__).parent.parent / "configs" / "data_law_v1_exp_hy3.yaml"


def test_the_hy3_probe_config_is_fenced_and_carries_the_new_model():
    """The 2026-08-28 hy3 think_low qualification probe apparatus.

    Single-ref generator on a NEW family (`hy`, distinct from every family
    already in the fleet), the same cost/band fences every isolated arm this
    week carries, and the deepseek-v4-flash entry left declared-but-unused
    in the same bai block (routing.generator names only bai/hy3, so it is
    never called this run).
    """
    import yaml

    from tuned.data.generate import _provider_usd_cap

    live = load_build_config(
        Path(__file__).parent.parent / "data" / "configs" / "data_law_v1.yaml",
        allow_unpinned=True,
    )
    cfg = load_build_config(HY3_CONFIG, allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_hy3"
    assert cfg.build.prompt_overlay is None

    # The same non-generator invariants _assert_ds_ab_common_fences checks
    # for the clause/cap arms - reimplemented rather than reused, because
    # that helper hardcodes the deepseek single-ref generator this arm
    # deliberately does not carry.
    assert _provider_usd_cap(cfg, "openai") == 0.0
    openai = next(p for p in cfg.providers if p.name == "openai")
    for model in openai.models:
        assert model.limits["usd_cap"] == 0.0
        assert model.limits["usd_per_1m_prompt"] > 0
        assert model.limits["usd_per_1m_completion"] > 0
    assert cfg.build.length_band == live.build.length_band
    assert cfg.build.length_band.think_max == 4000
    assert cfg.build.length_band.total_max == 8192
    raw = yaml.safe_load(HY3_CONFIG.read_text(encoding="utf-8"))
    for key in ("harmony_completions", "harmony_prefill", "harmony_s1_continue",
                "require_pretreatment_manifest", "pretreatment_manifest"):
        assert key not in raw["build"], key
    assert HY3_CONFIG.read_bytes().count(b"\r") == 0

    bai = next(p for p in cfg.providers if p.name == "bai")
    ids = {m.id: m for m in bai.models}
    assert set(ids) == {"deepseek-v4-flash", "hy3"}

    hy3 = ids["hy3"]
    assert hy3.family == "hy"
    assert hy3.family not in {"deepseek", "gpt-oss", "gemma", "qwen", "mistral"}
    assert hy3.roles == ("generator",)
    assert hy3.limits == {"rpm": 8, "max_context": 192000, "max_output": 16384}
    assert hy3.params == {
        "temperature": 0.7,
        "top_p": 0.95,
        "reasoning_effort": "low",
    }

    assert list(cfg.routing.generator) == ["bai/hy3"]
    assert cfg.routing.family_separation is True
    # This arm is generate-only (see header) and was forked from the older
    # data_law_v1_exp_deepseek.yaml lineage rather than today's live
    # data_law_v1.yaml, so its judge/tiebreak/probe lists are not asserted
    # against the live config's - they were never meant to track it.


def test_the_irac_ctl_arm_is_an_isolated_workdir(tmp_path):
    """data/build/exp_irac_ctl is an experiment sibling, not the live control.

    The control half of the 2026-08-28 irac stop-timing A/B. Sets no
    overlay - it reads the shipped templates, which is what makes it the
    baseline - so this fence is the only one it needs.
    """
    from tuned.data.paths import is_live_control_workdir

    assert is_live_control_workdir("data/build/exp_irac_ctl") is False
    assert is_live_control_workdir("data/build") is True
    doc = _base_doc()
    doc["build"]["workdir"] = "data/build/exp_irac_ctl"
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_irac_ctl"
    assert cfg.build.prompt_overlay is None


def test_the_irac_fix_arm_is_an_isolated_workdir(tmp_path):
    """data/build/exp_irac_fix is an experiment sibling, not the live control.

    The treatment half of the 2026-08-28 irac stop-timing A/B. Carries
    `prompt_overlay`, which makes it a RECOVERY experiment
    (config._is_recovery_experiment), and a recovery config aimed at the live
    workdir is refused outright - so an unlisted name under data/build would
    make the arm unloadable, not merely misread as the frozen control.
    """
    from tuned.data.paths import is_live_control_workdir

    assert is_live_control_workdir("data/build/exp_irac_fix") is False
    assert is_live_control_workdir("data/build") is True
    doc = _base_doc()
    doc["build"]["workdir"] = "data/build/exp_irac_fix"
    doc["build"]["prompt_overlay"] = "data/build/exp_irac_fix/prompts_stop_timing"
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_irac_fix"
    assert (
        cfg.build.prompt_overlay == "data/build/exp_irac_fix/prompts_stop_timing"
    )


IRAC_CTL_CONFIG = (
    Path(__file__).parent.parent / "configs" / "data_law_v1_exp_irac_ctl.yaml"
)
IRAC_FIX_CONFIG = (
    Path(__file__).parent.parent / "configs" / "data_law_v1_exp_irac_fix.yaml"
)


def test_the_irac_arms_are_fenced_and_differ_only_in_the_overlay_and_judge_seat():
    """The stop-timing control and treatment differ in exactly two things.

    The lever is `prompt_overlay` (F1: the how-to-think paragraph of the six
    IRAC-answer templates now names WHEN the thinking stops; F2: the two
    summarization templates ask for the genre's own prose form instead of
    the four headings `gates.IRAC_ANSWER_TASK_TYPES` already stopped
    requiring of a summarization answer).

    The SECOND difference is deliberate and is not part of the lever: only
    the treatment arm is judged, because F2 changes the shape of the
    summarization answer and a spot-check has to confirm the free fleet
    still accepts it. So the treatment restores
    data_law_v1_exp_deepseek.yaml's judge list and the groq gpt-oss-20b
    `roles` line that goes with it, while the control keeps ds_ctl2's
    generate-only pair. Both are asserted here rather than trusted, and
    line-level equality is asserted on everything else - a third difference
    would be a confound, and this test is what makes one impossible to
    introduce silently.
    """
    live = load_build_config(
        Path(__file__).parent.parent / "data" / "configs" / "data_law_v1.yaml",
        allow_unpinned=True,
    )
    for path, workdir, overlay in (
        (IRAC_CTL_CONFIG, "data/build/exp_irac_ctl", None),
        (IRAC_FIX_CONFIG, "data/build/exp_irac_fix",
         "data/build/exp_irac_fix/prompts_stop_timing"),
    ):
        cfg = load_build_config(path, allow_unpinned=True)
        assert cfg.build.workdir == workdir
        assert cfg.build.prompt_overlay == overlay
        _assert_ds_ab_common_fences(cfg, path, live)
        bai = next(p for p in cfg.providers if p.name == "bai")
        assert bai.models[0].limits["rpm"] == 8
        assert bai.models[0].limits["max_output"] == 16384

    # The judge seat, stated as values rather than left to the line filter
    # below. qwen leads (slot A) and gemma follows (slot B) in both arms;
    # the treatment additionally carries gpt-oss-20b behind them so a qwen
    # tpd exhaustion cannot empty slot B, and mistral holds the tiebreak in
    # both.
    ctl = load_build_config(IRAC_CTL_CONFIG, allow_unpinned=True)
    fix = load_build_config(IRAC_FIX_CONFIG, allow_unpinned=True)
    deepseek = load_build_config(
        Path(__file__).parent.parent / "configs" / "data_law_v1_exp_deepseek.yaml",
        allow_unpinned=True,
    )
    assert list(ctl.routing.judge) == [
        "groq/qwen/qwen3.6-27b", "cerebras/gemma-4-31b",
        "openai/gpt-5-mini", "openai/gpt-5-nano",
    ]
    assert list(fix.routing.judge) == list(deepseek.routing.judge)
    assert list(fix.routing.judge)[:2] == [
        "groq/qwen/qwen3.6-27b", "cerebras/gemma-4-31b",
    ]
    assert list(fix.routing.tiebreak) == list(ctl.routing.tiebreak)
    assert list(fix.routing.tiebreak)[0] == "mistral/mistral-large-latest"

    # Everything outside the two intended differences is equal line for
    # line. The four filtered prefixes are exactly the keys named in the two
    # arm headers; a fifth divergence fails this test.
    def _body(path: Path) -> list[str]:
        lines = path.read_text(encoding="utf-8").splitlines()
        first_key = next(i for i, ln in enumerate(lines) if not ln.startswith("#"))
        return [ln for ln in lines[first_key:]
                if not ln.startswith("  workdir:")
                and not ln.startswith("  prompt_overlay:")
                and not ln.startswith("  judge:")
                and not ln.startswith("  # added 2026-08-28 for judge")
                and ln.strip() not in {"roles: [tiebreak, probe]",
                                       "roles: [judge, tiebreak, probe]"}]

    assert _body(IRAC_CTL_CONFIG) == _body(IRAC_FIX_CONFIG)


def test_the_irac_ctl3_arm_is_an_isolated_workdir(tmp_path):
    """data/build/exp_irac_ctl3 is an experiment sibling, not the live control.

    The control half of the F2-only confirm pair - the follow-up the first
    stop-timing A/B's own report asked for, which re-runs F2 without F1 so
    the length_band cost can be attributed to one edit or the other.
    """
    from tuned.data.paths import is_live_control_workdir

    assert is_live_control_workdir("data/build/exp_irac_ctl3") is False
    assert is_live_control_workdir("data/build") is True
    doc = _base_doc()
    doc["build"]["workdir"] = "data/build/exp_irac_ctl3"
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_irac_ctl3"
    assert cfg.build.prompt_overlay is None


def test_the_irac_f2only_arm_is_an_isolated_workdir(tmp_path):
    """data/build/exp_irac_f2only is an experiment sibling, not the live control.

    The treatment half of the F2-only confirm pair. Carries `prompt_overlay`,
    which makes it a RECOVERY experiment (config._is_recovery_experiment),
    and a recovery config aimed at the live workdir is refused outright - so
    an unlisted name under data/build would make the arm unloadable, not
    merely misread as the frozen control.
    """
    from tuned.data.paths import is_live_control_workdir

    assert is_live_control_workdir("data/build/exp_irac_f2only") is False
    assert is_live_control_workdir("data/build") is True
    doc = _base_doc()
    doc["build"]["workdir"] = "data/build/exp_irac_f2only"
    doc["build"]["prompt_overlay"] = "data/build/exp_irac_f2only/prompts_f2only"
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_irac_f2only"
    assert cfg.build.prompt_overlay == "data/build/exp_irac_f2only/prompts_f2only"


IRAC_CTL3_CONFIG = (
    Path(__file__).parent.parent / "configs" / "data_law_v1_exp_irac_ctl3.yaml"
)
IRAC_F2ONLY_CONFIG = (
    Path(__file__).parent.parent / "configs" / "data_law_v1_exp_irac_f2only.yaml"
)


def test_the_f2only_confirm_arms_are_fenced_and_differ_only_in_the_prompt_overlay():
    """The F2-only confirm pair differs in ONE key.

    Stricter than the first stop-timing pairing, which had to allow a second
    difference for its judge seat: this measurement dispatches no judge at
    all - the format question was answered by the first run's spot-check
    (10/11 accept on the new summarization form) - so both arms keep the
    identical generate-only routing and `prompt_overlay` is the only
    intended divergence.

    The four gen_irac_analysis templates are byte-identical across the two
    arms by construction, which is what makes irac_analysis the untreated
    task type and its gate rates a measurement of arm noise rather than of
    any treatment.
    """
    live = load_build_config(
        Path(__file__).parent.parent / "data" / "configs" / "data_law_v1.yaml",
        allow_unpinned=True,
    )
    for path, workdir, overlay in (
        (IRAC_CTL3_CONFIG, "data/build/exp_irac_ctl3", None),
        (IRAC_F2ONLY_CONFIG, "data/build/exp_irac_f2only",
         "data/build/exp_irac_f2only/prompts_f2only"),
    ):
        cfg = load_build_config(path, allow_unpinned=True)
        assert cfg.build.workdir == workdir
        assert cfg.build.prompt_overlay == overlay
        _assert_ds_ab_common_fences(cfg, path, live)
        bai = next(p for p in cfg.providers if p.name == "bai")
        assert bai.models[0].limits["rpm"] == 8
        assert bai.models[0].limits["max_output"] == 16384

    # Generate-only: neither arm may carry the judge seat the combined
    # stop-timing treatment needed for its spot-check.
    ctl3 = load_build_config(IRAC_CTL3_CONFIG, allow_unpinned=True)
    f2only = load_build_config(IRAC_F2ONLY_CONFIG, allow_unpinned=True)
    assert list(ctl3.routing.judge) == list(f2only.routing.judge)
    assert list(ctl3.routing.tiebreak) == list(f2only.routing.tiebreak)

    def _body(path: Path) -> list[str]:
        lines = path.read_text(encoding="utf-8").splitlines()
        first_key = next(i for i, ln in enumerate(lines) if not ln.startswith("#"))
        return [ln for ln in lines[first_key:]
                if not ln.startswith("  workdir:")
                and not ln.startswith("  prompt_overlay:")]

    assert _body(IRAC_CTL3_CONFIG) == _body(IRAC_F2ONLY_CONFIG)
