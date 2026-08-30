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
    # the pin handoff push.py -> training/scripts/pin_dataset.py needs the value set
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


def test_the_shipped_judge_mode_is_audit():
    # audit since 2026-08-29: one groq key caps the dual fleet at ~35-40
    # decided rows/day, so the shipped config accepts unsampled gate-clean
    # rows and dual-judges only the --audit-sample fraction. Flipping this
    # back to dual is a deliberate operator decision (judge capacity
    # returned), not drift - hence the pin.
    cfg = load_build_config(DATA_CONFIG, allow_unpinned=True)
    assert cfg.routing.judge_mode == "audit"


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
    # The harmony/pretreatment knobs themselves were deleted 2026-08-28 (a
    # yaml carrying them now refuses to load at all); prompt_overlay is the
    # one surviving experiment knob and the live wave must not set it.
    assert cfg.build.prompt_overlay is None


_RECOVERY_SPEND_KEYS = ("usd_cap", "usd_per_1m_prompt", "usd_per_1m_completion")


LIVE_PUSH_REPO = "tantan01/tuned-law-v1-data"


def test_a_recovery_config_that_points_at_the_live_workdir_is_refused(tmp_path):
    doc = _base_doc()
    doc["build"]["workdir"] = "data/build"
    doc["build"]["prompt_overlay"] = "src/tuned/data/prompts"
    with pytest.raises(ValueError, match="live workdir"):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)


def test_a_recovery_config_that_resolves_to_the_live_database_is_refused(tmp_path):
    repo_root = Path(__file__).parent.parent.resolve()
    doc = _base_doc()
    doc["build"]["workdir"] = str(repo_root / "data" / "build")
    doc["build"]["prompt_overlay"] = "src/tuned/data/prompts"
    with pytest.raises(ValueError, match="live"):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)


def test_a_recovery_config_on_an_isolated_workdir_is_accepted(tmp_path):
    doc = _base_doc()
    doc["build"]["workdir"] = "data/build/exp_recovery"
    doc["build"]["prompt_overlay"] = "src/tuned/data/prompts"
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_recovery"


def _recovery_on(workdir: str) -> dict:
    doc = _base_doc()
    doc["build"]["workdir"] = workdir
    doc["build"]["prompt_overlay"] = "src/tuned/data/prompts"
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


def test_the_pin_rewrite_is_one_function_for_both_pins():
    # dataset_revision says WHICH commit to fetch; dataset_sha256 says which
    # BYTES that fetch must produce (sft.check_dataset_pin refuses a mismatch,
    # because resume replays a sampler permutation derived from the file). Two
    # copies of this regex would be two things to keep in step.
    from pin_dataset import rewrite_pin

    text = "hub:\n  checkpoint_repo: x\n  dataset_revision: oldsha\n"
    new = rewrite_pin(rewrite_pin(text, "dataset_revision", "newsha"),
                      "dataset_sha256", "deadbeef")
    assert "dataset_revision: newsha" in new
    assert "dataset_sha256: deadbeef" in new
    assert "oldsha" not in new


def test_the_corpus_digest_comes_from_the_shipped_manifest(tmp_path):
    # push.py records the sha256 it uploaded in build_manifest.json, so the pin
    # describes what actually shipped rather than a local rebuild that may
    # differ - which is the whole failure this pin exists to catch.
    import json

    from pin_dataset import TRAIN_FILENAME, resolve_dataset_sha256

    manifest = tmp_path / "build_manifest.json"
    manifest.write_text(json.dumps({"outputs": [
        {"path": "law_v1_eval.jsonl", "sha256": "eval-digest"},
        {"path": TRAIN_FILENAME, "sha256": "train-digest"},
    ]}), encoding="utf-8")
    got = resolve_dataset_sha256("u/d", "rev", download=lambda **kw: str(manifest))
    assert got == "train-digest"

    # a repo with no manifest still pins its revision rather than failing
    def _missing(**kw):
        raise FileNotFoundError("no manifest")

    assert resolve_dataset_sha256("u/d", "rev", download=_missing) is None


def test_eval_cohort_strata_defaults_to_none_on_the_live_config():
    cfg = load_build_config("data/configs/data_law_v1.yaml", allow_unpinned=True)
    assert cfg.build.eval_cohort_strata is None


def test_eval_cohort_strata_refuses_empty_duplicate_and_non_string(tmp_path):
    """A stratum list is a pre-registration, so it is validated at load.

    An empty list, a repeat, or a non-string is a typo that would otherwise
    reach the cohort selector and silently change the cohort's size.
    """
    import yaml as _yaml

    # Retargeted 2026-08-28: the recovery arm's config (the original fixture
    # base) was retired to prev_rep.md; the validator under test is config
    # machinery, so the LIVE config plus an injected key exercises it the same.
    base = _yaml.safe_load(
        Path("data/configs/data_law_v1.yaml").read_text(encoding="utf-8")
    )
    for bad in ([], ["irac_analysis", "irac_analysis"], ["irac_analysis", 7], "drafting"):
        doc = dict(base)
        doc["build"] = dict(base["build"])
        doc["build"]["eval_cohort_strata"] = bad
        path = tmp_path / "bad.yaml"
        path.write_text(_yaml.safe_dump(doc), encoding="utf-8")
        with pytest.raises(ValueError, match="eval_cohort_strata"):
            load_build_config(path, allow_unpinned=True)


