"""Typed loader for data/configs/data_law_v1.yaml, the dataset-curation build config.

This config is churn-heavy (API providers/models/limits change weekly) so it
lives separately from the training config and only ever REFERENCES it
(build.train_config) for anything the trainer owns - think tags, tokenizer
pin, dataset path. load_build_config resolves those fields out of the
referenced training config at load time so they are never duplicated here.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from tuned.data.paths import is_live_control_workdir, package_repo_root
from tuned.train.config import load_config

_ROLES = ("generator", "judge", "tiebreak", "probe")


@dataclass(frozen=True)
class LengthBand:
    total_max: int
    total_min: int
    think_min: int
    think_max: int
    answer_min: int


@dataclass(frozen=True)
class BuildCfg:
    train_config: str
    workdir: str
    target_total: int
    mvp_total: int
    mix: dict[str, float]
    overgeneration: float
    held_out_frac: float
    length_band: LengthBand
    difficulty_target: dict[str, float]
    appointed_day: str
    # Completions-path Harmony analysis prefill (gpt-oss). Off on the live
    # wave. Isolated experiment yaml sets both; generate.py then posts to
    # /v1/completions instead of chat/completions.
    harmony_prefill: str | None = None
    harmony_completions: bool = False
    # Second Completions call appending s1's " Wait" when the first analysis
    # continuation has no verification cue. Off on the live wave.
    harmony_s1_continue: bool = False
    # Optional directory of *.md templates that override `prompts/` by stem.
    # Live SHAs stay pinned; experiment yaml points here. Relative to repo root.
    prompt_overlay: str | None = None
    # Opt-in generate.main gate: a complete persisted 80-pair pre-treatment
    # manifest must exist before the recovery workdir is created. Off unless
    # a yaml sets the flag; Harmony stays off until it opts in.
    require_pretreatment_manifest: bool = False
    pretreatment_manifest: str | None = None
    # Which task-type strata the matched evaluator's cohort draws from.
    # None = every stratum eval_matched knows (the four-way default), which
    # is what the live wave declares by saying nothing.
    #
    # An experiment declares a SHORTER list only when a stratum cannot be
    # filled for a DATA reason, and the reason belongs in the yaml next to
    # the list. Measured 2026-08-23: the control store holds 270 statute_qa
    # tasks and 0 seeds carrying a section_text distinct from the seed body,
    # so statute_section_eligible refuses every one of them and the stratum
    # is unfillable until real Gazette provision text exists. Declaring it
    # here - rather than letting the selector return a short cohort - is what
    # keeps a 60-pair result from being read as an 80-pair one.
    eval_cohort_strata: tuple[str, ...] | None = None


@dataclass(frozen=True)
class GateCfg:
    """stats.py's thresholds. Every one of them is a gate, not a report knob.

    `cross_code_red` and `require_chain` are the two toggles: the first because
    the measurement is a regex over prose and its false-positive direction
    costs real rows, the second because a build that never ran decontamination
    is a different kind of problem from a threshold miss and an operator may
    want to see the rest of the report before it refuses.
    """

    mix_tolerance_pp: float
    trace_floor: float
    empty_think_min: float
    empty_think_max: float
    dup_ceiling: float
    markup: bool
    require_license: bool
    cross_code_red: bool
    old_code_sources: tuple[str, ...]
    require_chain: bool


# The profile whose targets are build.mix. Named rather than spelled twice:
# `assembly.profiles` may not redefine it (see _validate), so this constant is
# the one place that says WHICH profile the top-level mix is.
FULL_PROFILE = "v1.1-full"

# Slack for the ONE comparison in _validate that subtracts two share bounds.
# `1 - 0.80` is 0.19999999999999996 in binary floating point, so a bare
# `empty_think_max > 1 - trace_floor` refuses the shipped 0.80/0.20 pair - the
# exact coherence the check exists to require. Nine places is far below any
# threshold anyone would write and far above the noise; the same reasoning,
# and the same number, as stats.gate_mix's rounding.
_SHARE_EPS = 1e-9

@dataclass(frozen=True)
class AssemblyCfg:
    """split/assemble/stats configuration - thresholds and the mix mapping.

    Deliberately does NOT carry an eval fraction or a length bucket: those are
    build.held_out_frac and the train config's train.main.max_seq_length, both
    already in force, and a second copy is a fence that can disagree with the
    fencing.
    """

    default_profile: str
    profiles: dict[str, dict[str, float]]
    source_streams: dict[str, str]
    gates: GateCfg

    def targets(self, profile: str | None = None) -> dict[str, float]:
        name = profile or self.default_profile
        if name not in self.profiles:
            raise KeyError(
                f"no mix profile {name!r}; known profiles are {sorted(self.profiles)}"
            )
        return dict(self.profiles[name])

    def stream_of(self, source) -> str | None:
        """Which mix bucket a `_prov.source` counts toward, or None.

        None is a RED GATE upstream, never a default bucket - a source nobody
        mapped is a stream nobody sized.

        Two lookups, in this order: the whole source string, then the part
        before the first ":". replay.py builds "HuggingFaceTB/smoltalk2:{sub}"
        out of the upstream row's own column, so the subset half cannot be
        enumerated in config - but a subset that needs its OWN bucket can still
        say so, because the full string is tried first.
        """
        key = str(source or "")
        if not key:
            return None
        if key in self.source_streams:
            return self.source_streams[key]
        return self.source_streams.get(key.split(":", 1)[0])


@dataclass(frozen=True)
class ModelCfg:
    id: str
    family: str
    roles: tuple[str, ...]
    limits: dict
    params: dict
    # role -> params that override `params` for calls made IN that role.
    #
    # Needed because one model can serve two roles with different sampling.
    # One model can serve two roles - mistral-small-latest generated AND
    # judged until 2026-08-19 - and a generator wants
    # temperature 0.7 / top_p 0.95 and a judge wants 0.2, and before this
    # existed the two roles lived in two config blocks so the distinction was
    # free. Merging those blocks (Mistral Small 4, 2026-08-18) would have sent
    # the generator's sampling to every judge call - the same hazard
    # reasoning_effort had, and the same shape of fix.
    #
    # Precedence, applied in providers.ModelClient.build_payload:
    #     model.params  <  model.role_params[role]  <  per-call params
    # so a role entry overrides the model default and a caller still overrides
    # both. Roles named here must be roles the model actually declares.
    role_params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderCfg:
    name: str
    base_url: str
    api_key_env: str
    quirks: tuple[str, ...]
    models: tuple[ModelCfg, ...]


@dataclass(frozen=True)
class RoutingCfg:
    generator: tuple[str, ...]
    judge: tuple[str, ...]
    tiebreak: tuple[str, ...]
    probe: tuple[str, ...]
    family_separation: bool
    judge_mode: str


@dataclass(frozen=True)
class ModelRef:
    provider: str
    model: str


def _parse_ref(ref: str) -> ModelRef:
    # groq/qwen/qwen3.6-27b has TWO slashes - split on the FIRST only, so the
    # provider is "groq" and the model id is "qwen/qwen3.6-27b".
    return ModelRef(*ref.split("/", 1))


# The judge score range, owned here because CalibrationCfg.thresholds are
# graded against it and config.py cannot import judge.py (judge.py imports
# this module for LengthBand). judge.SCORE_RANGE is the other copy and a test
# pins the two together - the same treatment JUDGE_MAX_TOKENS /
# DEFAULT_JUDGE_REPLY_TOKENS already get.
JUDGE_SCORE_RANGE = (1, 5)

# The decision rules calibrate.py fits over. Owned here for the same reason:
# `calibration.rules` is validated at load, calibrate.py imports this tuple
# rather than restating it, so a rule name cannot exist in one place only.
#
#   min_axis  the harshest axis clears the threshold
#   mean      the mean of the three axes clears it
#   both      min_axis AND mean clear it
CALIBRATION_RULES = ("min_axis", "mean", "both")


@dataclass(frozen=True)
class TransitionCfg:
    """transition.py's grid sizing. The GRID itself is not sized here.

    Its size is `verified mapping rows x date postures x procedural postures
    x question forms`, and every one of those factors is owned by code or by
    the mapping resource - so a `cells:` key here would be a number that
    disagrees with the grid the moment an operator signs off one more mapping
    row. What IS a choice is how much of the grid is drawn: `sample` training
    cells and `eval_reserve` cells held back as the transition-accuracy eval.
    """

    sample: int
    eval_reserve: int


@dataclass(frozen=True)
class CalibrationCfg:
    """calibrate.py's bounds. `pilot_export` generations are exported for the
    operator to label by hand, `holdout` of them are kept out of the fit
    entirely and the rest are split into `folds` cross-validation folds.

    min_recall / min_precision are the P5 gate: maximise precision subject to
    recall >= min_recall, then disqualify any judge whose HOLDOUT precision is
    below min_precision.
    """

    pilot_export: int
    holdout: int
    folds: int
    thresholds: tuple[int, ...]
    rules: tuple[str, ...]
    min_recall: float
    min_precision: float


@dataclass(frozen=True)
class DifficultyCfg:
    """difficulty.py's bounds. `probe_sample` is a CEILING on how many rows
    ever reach the probe model - the whole design is one calibration sample
    and a length proxy for everything else, because per-row probing was
    measured at 32M tokens / 65 days.

    The target mix is build.difficulty_target and is deliberately NOT
    restated here; `mix_tolerance` is how far the labelled corpus may sit
    from it before difficulty.py refuses its own bands.
    """

    probe_sample: int
    mix_tolerance: float


@dataclass(frozen=True)
class PushCfg:
    """push.py's target: the HF dataset repo, and nothing push.py should ever
    hardcode. Optional at the top level - a build config that never pushes
    (every fixture in this test suite, for one) does not need this block, and
    split.py/assemble.py/stats.py never read it."""

    repo_id: str
    private: bool = True
    card_extra: str | None = None


@dataclass(frozen=True)
class BuildConfig:
    build: BuildCfg
    providers: tuple[ProviderCfg, ...]
    routing: RoutingCfg
    assembly: AssemblyCfg
    # Resolved from the referenced train config at load time - never
    # duplicated in data/configs/data_law_v1.yaml itself.
    think_open: str
    think_close: str
    model_repo: str
    model_revision: str | None
    instruction_part: str
    response_part: str
    main_dataset_path: str
    # The length bucket assemble.py drops rows against. It is the TRAINER's
    # number (train.main.max_seq_length) resolved here for the same reason the
    # think tags are: a builder that carried its own 8192 could pass a corpus
    # the trainer then truncates or refuses.
    max_seq_length: int
    # None when the config carries no `push:` block at all - see PushCfg.
    push: PushCfg | None = None
    # None when the config carries no block of that name. OPTIONAL like
    # `push:` and unlike `assembly:`, and the split is not a preference: every
    # module in the assembly tail grades EVERY corpus, so a missing
    # `assembly:` is a builder with no targets, whereas a config that never
    # builds the transition grid, never calibrates a judge and never labels
    # difficulty is a real configuration (every fixture in this suite is one).
    # The three modules refuse by name when their own block is absent, which
    # is a better message than a loader error can give.
    transition: TransitionCfg | None = None
    calibration: CalibrationCfg | None = None
    difficulty: DifficultyCfg | None = None

    def model_for(self, ref: ModelRef) -> tuple[ProviderCfg, ModelCfg]:
        for provider in self.providers:
            if provider.name == ref.provider:
                for model in provider.models:
                    if model.id == ref.model:
                        return provider, model
                raise KeyError(f"no model {ref.model!r} on provider {ref.provider!r}")
        raise KeyError(f"no provider {ref.provider!r} (from ref {ref.provider}/{ref.model})")

    def routing_refs(self, role: str) -> tuple[ModelRef, ...]:
        if role not in _ROLES:
            raise ValueError(f"unknown role {role!r}, must be one of {_ROLES}")
        return tuple(_parse_ref(s) for s in getattr(self.routing, role))


def _validate(cfg: BuildConfig) -> None:
    # 1 & 2: every routing ref resolves to an existing provider+model, and
    # that model actually lists a compatible role.
    for role in _ROLES:
        for ref_str in getattr(cfg.routing, role):
            ref = _parse_ref(ref_str)
            try:
                provider, model = cfg.model_for(ref)
            except KeyError as exc:
                raise ValueError(f"routing.{role} ref {ref_str!r}: {exc.args[0]}") from None
            if role not in model.roles:
                raise ValueError(
                    f"routing.{role} ref {ref_str!r} resolves to "
                    f"{provider.name}/{model.id}, which does not list role "
                    f"{role!r} (has roles {model.roles})"
                )

    # 2b: role_params may only name roles the model actually serves. A typo'd
    # or stale key ("judgge", or a role deleted from `roles` afterwards) would
    # otherwise sit in the config looking like it configures something and
    # silently never apply - the same class of failure as a forbidden-section
    # entry that can never fire.
    for provider in cfg.providers:
        for model in provider.models:
            unknown = sorted(set(model.role_params) - set(model.roles))
            if unknown:
                raise ValueError(
                    f"{provider.name}/{model.id}: role_params names "
                    f"{unknown}, which {'is' if len(unknown) == 1 else 'are'} "
                    f"not in its roles {list(model.roles)}"
                )
            bad = sorted(k for k, v in model.role_params.items() if not isinstance(v, dict))
            if bad:
                raise ValueError(
                    f"{provider.name}/{model.id}: role_params entries {bad} "
                    f"are not mappings"
                )

    # 3: cross-family judging must be possible for every generator - the
    # judge pool must contain >=2 distinct families other than the
    # generator's own family. Per-call enforcement is providers.py's job;
    # this is only the static feasibility check.
    if cfg.routing.family_separation:
        judge_families = {cfg.model_for(_parse_ref(r))[1].family for r in cfg.routing.judge}
        for gen_ref_str in cfg.routing.generator:
            gen_family = cfg.model_for(_parse_ref(gen_ref_str))[1].family
            other_families = judge_families - {gen_family}
            if len(other_families) < 2:
                raise ValueError(
                    f"routing.generator ref {gen_ref_str!r} (family {gen_family!r}) "
                    f"has fewer than 2 judge families outside its own "
                    f"(judge pool families: {sorted(judge_families)})"
                )

    # 4: scalar sanity checks.
    if cfg.routing.judge_mode not in ("dual", "audit"):
        raise ValueError(
            f"routing.judge_mode must be 'dual' or 'audit', got {cfg.routing.judge_mode!r}"
        )
    mix_total = sum(cfg.build.mix.values())
    if abs(mix_total - 1.0) > 0.001:
        raise ValueError(f"build.mix values must sum to 1.0, got {mix_total}")
    if not (0 < cfg.build.held_out_frac < 0.5):
        raise ValueError(
            f"build.held_out_frac must be in (0, 0.5), got {cfg.build.held_out_frac}"
        )
    if cfg.build.overgeneration < 1.0:
        raise ValueError(f"build.overgeneration must be >= 1.0, got {cfg.build.overgeneration}")
    # build.difficulty_target is a SHARE table over every row difficulty.py
    # labels, exactly like build.mix is over streams, so it is graded the same
    # way. It went unchecked while nothing read it; difficulty.py grades its
    # own bands against it (within difficulty.mix_tolerance), and a table that
    # sums to 0.9 makes the band that can satisfy it non-existent.
    for label, share in sorted(cfg.build.difficulty_target.items()):
        if isinstance(share, bool) or not isinstance(share, (int, float)):
            raise ValueError(f"build.difficulty_target.{label} must be a number, got {share!r}")
        if not (0.0 <= float(share) <= 1.0):
            raise ValueError(f"build.difficulty_target.{label} must be in [0, 1], got {share}")
    difficulty_total = sum(float(v) for v in cfg.build.difficulty_target.values())
    if abs(difficulty_total - 1.0) > 0.001:
        raise ValueError(
            f"build.difficulty_target values must sum to 1.0, got {difficulty_total}"
        )

    # 5: the assembly block. Every rule here is one stats.py would otherwise
    # discover at the end of a multi-day build, on the corpus.
    assembly = cfg.assembly
    if assembly.default_profile not in assembly.profiles:
        raise ValueError(
            f"assembly.default_profile {assembly.default_profile!r} is not a profile; "
            f"known profiles are {sorted(assembly.profiles)}"
        )
    buckets = set(assembly.profiles[FULL_PROFILE])
    for name, targets in sorted(assembly.profiles.items()):
        total = sum(targets.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"assembly.profiles.{name} shares must sum to 1.0, got {total}"
            )
        if set(targets) != buckets:
            # Two profiles over different buckets are two different gates
            # wearing one name: --profile would silently change WHAT is
            # measured rather than what it is measured against.
            raise ValueError(
                f"assembly.profiles.{name} names streams {sorted(targets)}, but "
                f"{FULL_PROFILE} names {sorted(buckets)} - every profile must grade the "
                f"same buckets"
            )
        for stream, share in sorted(targets.items()):
            if not (0.0 <= share <= 1.0):
                raise ValueError(
                    f"assembly.profiles.{name}.{stream} must be in [0, 1], got {share}"
                )
    unknown = sorted(
        f"{source} -> {stream}"
        for source, stream in assembly.source_streams.items()
        if stream not in buckets
    )
    if unknown:
        raise ValueError(
            f"assembly.source_streams maps sources to streams no profile grades "
            f"({'; '.join(unknown)}); the streams are {sorted(buckets)}"
        )
    gates = assembly.gates
    if gates.mix_tolerance_pp < 0:
        raise ValueError(
            f"assembly.gates.mix_tolerance_pp must be >= 0, got {gates.mix_tolerance_pp}"
        )
    if not (0.0 <= gates.trace_floor <= 1.0):
        raise ValueError(
            f"assembly.gates.trace_floor must be in [0, 1], got {gates.trace_floor}"
        )
    if not (0.0 <= gates.empty_think_min <= gates.empty_think_max <= 1.0):
        raise ValueError(
            f"assembly.gates.empty_think_min/max must satisfy 0 <= min <= max <= 1, got "
            f"{gates.empty_think_min}/{gates.empty_think_max}"
        )
    # THE THREE SHARE BOUNDS ARE ONE SYSTEM, not three independent numbers.
    # assemble.py drops every row that is neither traced nor empty-scaffolded
    # (ASSEMBLE_VERSION 2), so over the corpus stats.py grades
    # trace_share + empty_think_share == 1 exactly. Two combinations are then
    # unsatisfiable as ARITHMETIC rather than as a corpus, and a build that
    # meets them at the end of a multi-day run meets them on the corpus.
    complement = 1.0 - gates.trace_floor
    if gates.trace_floor + gates.empty_think_min > 1.0 + _SHARE_EPS:
        raise ValueError(
            f"assembly.gates.trace_floor {gates.trace_floor} and empty_think_min "
            f"{gates.empty_think_min} cannot both be satisfied: a corpus with at least "
            f"{gates.trace_floor:.1%} reasoning traces carries at most {complement:.1%} "
            f"empty-scaffold rows, which is below that floor. EVERY corpus reds."
        )
    if gates.empty_think_max > complement + _SHARE_EPS:
        raise ValueError(
            f"assembly.gates.empty_think_max {gates.empty_think_max} is above "
            f"1 - trace_floor ({complement:.4g}), which leaves a DEAD BAND: a corpus "
            f"between {complement:.1%} and {gates.empty_think_max:.1%} empty-scaffold "
            f"rows passes the empty-think band and fails the trace floor, so the two "
            f"gates disagree about the same corpus. The ceiling is the trace floor's "
            f"complement, not an independent number."
        )
    if not (0.0 <= gates.dup_ceiling <= 1.0):
        raise ValueError(
            f"assembly.gates.dup_ceiling must be in [0, 1], got {gates.dup_ceiling}"
        )


# What each required `assembly:` key is FOR. A partial block used to die with
# a bare `KeyError: 'gates'` from whichever line reached for it first, which
# names the key and nothing else - so these are rule 5's own refusals, written
# where the raw dict still exists.
_ASSEMBLY_KEYS = {
    "default_profile": "the mix profile stats.py grades against when --profile is not passed",
    "source_streams": "the _prov.source -> mix bucket mapping, without which every source "
                      "is unmapped and every corpus reds",
    "gates": "every threshold stats.py compares against",
}
_GATE_NUMBERS = {
    "mix_tolerance_pp": "the mix tolerance in percentage points",
    "trace_floor": "the reasoning-trace floor",
    "empty_think_min": "the empty-think floor",
    "empty_think_max": "the empty-think ceiling",
    "dup_ceiling": "the duplicate-share ceiling",
}
# Toggles, and the reason they are parsed strictly rather than with bool():
# `bool("false")` is True, so a QUOTED YAML boolean inverts every one of them.
_GATE_TOGGLES = {
    "markup": "whether control-token markup reds the build",
    "require_license": "whether an unlicensed row reds the build",
    "cross_code_red": "whether the cross-code measurement gates or only reports",
    "require_chain": "whether a broken custody chain reds the build",
}


def _required(block, key: str, *, where: str, purpose: str):
    if not isinstance(block, dict):
        raise ValueError(
            f"`{where}:` must be a block of keys, got {type(block).__name__}"
        )
    if key not in block:
        raise ValueError(
            f"{where}.{key} is missing, and it is {purpose}. A PARTIAL `{where}:` block "
            f"is a builder grading a corpus against thresholds nobody wrote."
        )
    return block[key]


def _gate_number(raw: dict, key: str) -> float:
    value = _required(raw, key, where="assembly.gates", purpose=_GATE_NUMBERS[key])
    # bool is a subclass of int, so `float(True)` is 1.0 and a stray `true`
    # would load as a threshold rather than as the mistake it is.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"assembly.gates.{key} must be a number, got {value!r}")
    return float(value)


def _gate_toggle(raw: dict, key: str) -> bool:
    value = _required(raw, key, where="assembly.gates", purpose=_GATE_TOGGLES[key])
    if not isinstance(value, bool):
        raise ValueError(
            f"assembly.gates.{key} must be a YAML boolean (true/false), got {value!r}. "
            f"It is not coerced: bool() of any non-empty string is True, so a quoted "
            f'"false" would ARM this gate and a quoted "no" would read as yes.'
        )
    return value


def _old_code_sources(raw: dict) -> tuple[str, ...]:
    """The cross-code gate's pre-transition corpora - a LIST, always.

    A bare string is the failure worth spelling out: `tuple("169Pi/indian_law")`
    is sixteen single characters, no source string ever equals one of them, and
    the gate goes silently dead while the config still reads as if it is armed.
    """
    value = raw.get("old_code_sources")
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(
            f"assembly.gates.old_code_sources must be a LIST of source strings, got "
            f"{value!r}. A bare string iterates as its own characters, which empties "
            f"the cross-code gate silently: no source matches a single letter."
        )
    wrong = [item for item in value if not isinstance(item, str)]
    if wrong:
        raise ValueError(
            f"assembly.gates.old_code_sources must contain only source strings, got "
            f"{wrong!r}"
        )
    return tuple(value)


def _assembly_of(raw: dict, build: BuildCfg) -> AssemblyCfg:
    """The `assembly:` block, with build.mix folded in as the full profile.

    build.mix IS the v1.1-full mix target - it is already there, already
    validated to sum to 1.0, and stats.py grades against it. So the profile
    table is assembled here rather than written twice, and a YAML that tries to
    restate v1.1-full is refused: two copies of 60/16/24 can drift, and the one
    that drifts is the one nothing else reads.
    """
    block = raw.get("assembly")
    if block is None:
        raise ValueError(
            "this build config has no `assembly:` block, so split.py/assemble.py/stats.py "
            "have no mix targets, no source->stream mapping and no gate thresholds. The "
            "builder cannot grade a corpus it has no targets for."
        )
    for key, purpose in _ASSEMBLY_KEYS.items():
        _required(block, key, where="assembly", purpose=purpose)
    profiles = {name: dict(targets) for name, targets in (block.get("profiles") or {}).items()}
    if FULL_PROFILE in profiles:
        raise ValueError(
            f"assembly.profiles.{FULL_PROFILE} restates build.mix. build.mix is the ONE "
            f"definition of that profile's targets; delete the copy here."
        )
    profiles[FULL_PROFILE] = dict(build.mix)
    gates_raw = block["gates"]
    if not isinstance(gates_raw, dict):
        raise ValueError(f"`assembly.gates:` must be a block of keys, got {gates_raw!r}")
    gates = GateCfg(
        **{key: _gate_number(gates_raw, key) for key in _GATE_NUMBERS},
        **{key: _gate_toggle(gates_raw, key) for key in _GATE_TOGGLES},
        old_code_sources=_old_code_sources(gates_raw),
    )
    streams = block["source_streams"]
    if not isinstance(streams, dict):
        raise ValueError(
            f"assembly.source_streams must be a mapping of source -> stream, got "
            f"{streams!r}"
        )
    return AssemblyCfg(
        default_profile=block["default_profile"],
        profiles=profiles,
        source_streams=dict(streams),
        gates=gates,
    )


def _push_of(raw: dict) -> PushCfg | None:
    """The optional `push:` block, or None if this config never pushes.

    Unlike `assembly:`, absence is a legitimate configuration - split.py,
    assemble.py and stats.py all run to a green report without one, and only
    push.py itself needs to fail on a missing block (named there, not here,
    because the message push.py can give ["run stats.py first" vs "this
    config has no push target"] is more useful than a generic loader error).
    Once the block IS present, every key in it is checked the same strict way
    assembly.gates is: no coercion, a named refusal per key.
    """
    block = raw.get("push")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError(f"`push:` must be a block of keys, got {type(block).__name__}")
    if "repo_id" not in block:
        raise ValueError(
            "push.repo_id is missing, and it is the HuggingFace dataset repo push.py "
            "writes to. A `push:` block without it is a target nobody named."
        )
    repo_id = block["repo_id"]
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError(f"push.repo_id must be a non-empty string, got {repo_id!r}")
    # Control characters first, and named by SHAPE only - never echoed whole:
    # a repo_id is written into refusal messages and printed output elsewhere
    # in this pipeline, and a value carrying an embedded newline could forge
    # a fake log line there. Every check after this one is safe to quote in
    # full, because a string that reaches it is already known printable.
    if not repo_id.isprintable():
        raise ValueError(
            f"push.repo_id contains a non-printable or control character "
            f"({len(repo_id)} chars total) - the value itself is never echoed here; "
            f"fix it in the config"
        )
    if repo_id != repo_id.strip():
        raise ValueError(
            f"push.repo_id must not carry leading/trailing whitespace, got {repo_id!r}"
        )
    halves = repo_id.split("/")
    if len(halves) != 2 or not halves[0] or not halves[1]:
        raise ValueError(
            f"push.repo_id must be `namespace/name` (exactly one `/`, both halves "
            f"non-empty), got {repo_id!r}"
        )
    private = block.get("private", True)
    # Strict, like assembly.gates' toggles: bool("false") is True, so a quoted
    # "false" here would make a PRIVATE dataset repo PUBLIC, which is the one
    # direction this config must never coerce its way into.
    if not isinstance(private, bool):
        raise ValueError(
            f"push.private must be a YAML boolean (true/false), got {private!r}. Not "
            f'coerced: bool() of any non-empty string is True, so a quoted "false" '
            f"would make a private dataset repo PUBLIC."
        )
    card_extra = block.get("card_extra")
    if card_extra is not None and not isinstance(card_extra, str):
        raise ValueError(f"push.card_extra must be a string path, got {card_extra!r}")
    return PushCfg(repo_id=repo_id, private=private, card_extra=card_extra)


def _block(raw: dict, name: str) -> dict | None:
    """An optional top-level block, refused if it is present but not a block.

    `foo:` with nothing under it parses as None, which is indistinguishable
    from an absent key and is treated as absent. Anything else that is not a
    mapping is a typo worth naming.
    """
    value = raw.get(name)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"`{name}:` must be a block of keys, got {type(value).__name__}")
    return value


def _positive_int(block: dict, key: str, *, where: str, purpose: str) -> int:
    value = _required(block, key, where=where, purpose=purpose)
    # bool before int: `True` is an int in Python and would load as 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}.{key} must be a whole number, got {value!r}")
    if value < 1:
        raise ValueError(f"{where}.{key} must be >= 1, got {value}")
    return value


def _unit_float(block: dict, key: str, *, where: str, purpose: str) -> float:
    value = _required(block, key, where=where, purpose=purpose)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}.{key} must be a number, got {value!r}")
    if not (0.0 < float(value) <= 1.0):
        raise ValueError(f"{where}.{key} must be in (0, 1], got {value}")
    return float(value)


_TRANSITION_KEYS = {
    "sample": "how many grid cells become training seeds",
    "eval_reserve": "how many cells are held back as the transition-accuracy eval",
}
_CALIBRATION_KEYS = {
    "pilot_export": "how many pilot generations the operator is asked to label",
    "holdout": "how many of them never enter the fit at all",
    "folds": "the cross-validation fold count over the rest",
    "thresholds": "the candidate score thresholds the fit sweeps",
    "rules": "the candidate decision rules the fit sweeps",
    "min_recall": "the recall floor the precision maximisation is subject to",
    "min_precision": "the HOLDOUT precision below which a judge is disqualified",
}
_DIFFICULTY_KEYS = {
    "probe_sample": "the ceiling on how many rows ever reach the probe model",
    "mix_tolerance": "how far the labelled mix may sit from build.difficulty_target",
}


def _transition_of(raw: dict) -> TransitionCfg | None:
    block = _block(raw, "transition")
    if block is None:
        return None
    cfg = TransitionCfg(
        **{
            key: _positive_int(block, key, where="transition", purpose=purpose)
            for key, purpose in _TRANSITION_KEYS.items()
        }
    )
    if cfg.eval_reserve >= cfg.sample:
        # Not an arbitrary ordering: the reserve is a held-back MEASUREMENT of
        # a stream whose rows are the sample. A reserve at or above the sample
        # is a build that holds back more than it ships, which is a
        # mis-transcribed pair of numbers every time.
        raise ValueError(
            f"transition.eval_reserve ({cfg.eval_reserve}) must be smaller than "
            f"transition.sample ({cfg.sample}): the reserve is an eval slice held "
            f"back from the stream, not the stream"
        )
    return cfg


def _calibration_of(raw: dict) -> CalibrationCfg | None:
    block = _block(raw, "calibration")
    if block is None:
        return None
    for key, purpose in _CALIBRATION_KEYS.items():
        _required(block, key, where="calibration", purpose=purpose)

    thresholds = block["thresholds"]
    if isinstance(thresholds, str) or not isinstance(thresholds, (list, tuple)) or not thresholds:
        raise ValueError(
            f"calibration.thresholds must be a non-empty LIST of scores, got {thresholds!r}"
        )
    low, high = JUDGE_SCORE_RANGE
    for value in thresholds:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"calibration.thresholds must be whole scores, got {value!r}")
        if not (low <= value <= high):
            raise ValueError(
                f"calibration.thresholds entry {value} is outside the judge score range "
                f"{low}-{high}; a threshold no judgement can reach fits nothing"
            )

    rules = block["rules"]
    if isinstance(rules, str) or not isinstance(rules, (list, tuple)) or not rules:
        raise ValueError(f"calibration.rules must be a non-empty LIST of rule names, got {rules!r}")
    unknown = [rule for rule in rules if rule not in CALIBRATION_RULES]
    if unknown:
        raise ValueError(
            f"calibration.rules names {unknown}, which calibrate.py cannot fit; the rules "
            f"are {list(CALIBRATION_RULES)}"
        )

    cfg = CalibrationCfg(
        pilot_export=_positive_int(
            block, "pilot_export", where="calibration", purpose=_CALIBRATION_KEYS["pilot_export"]
        ),
        holdout=_positive_int(
            block, "holdout", where="calibration", purpose=_CALIBRATION_KEYS["holdout"]
        ),
        folds=_positive_int(block, "folds", where="calibration", purpose=_CALIBRATION_KEYS["folds"]),
        thresholds=tuple(sorted(dict.fromkeys(thresholds))),
        rules=tuple(dict.fromkeys(rules)),
        min_recall=_unit_float(
            block, "min_recall", where="calibration", purpose=_CALIBRATION_KEYS["min_recall"]
        ),
        min_precision=_unit_float(
            block, "min_precision", where="calibration", purpose=_CALIBRATION_KEYS["min_precision"]
        ),
    )
    if cfg.folds < 2:
        raise ValueError(
            f"calibration.folds must be >= 2, got {cfg.folds}: one fold is not "
            f"cross-validation, it is the fit measuring itself"
        )
    fitted = cfg.pilot_export - cfg.holdout
    if fitted < cfg.folds:
        raise ValueError(
            f"calibration.pilot_export ({cfg.pilot_export}) minus holdout ({cfg.holdout}) "
            f"leaves {fitted} labelled rows for {cfg.folds} folds. The holdout comes out of "
            f"the export, so the export must exceed it by at least one row per fold."
        )
    return cfg


def _difficulty_of(raw: dict) -> DifficultyCfg | None:
    block = _block(raw, "difficulty")
    if block is None:
        return None
    probe_sample = _positive_int(
        block, "probe_sample", where="difficulty", purpose=_DIFFICULTY_KEYS["probe_sample"]
    )
    tolerance = _required(
        block, "mix_tolerance", where="difficulty", purpose=_DIFFICULTY_KEYS["mix_tolerance"]
    )
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError(f"difficulty.mix_tolerance must be a number, got {tolerance!r}")
    if not (0.0 <= float(tolerance) <= 1.0):
        raise ValueError(f"difficulty.mix_tolerance must be in [0, 1], got {tolerance}")
    return DifficultyCfg(probe_sample=probe_sample, mix_tolerance=float(tolerance))


def _is_recovery_experiment(build: BuildCfg) -> bool:
    return bool(
        build.prompt_overlay
        or build.harmony_completions
        or build.harmony_prefill
        or build.harmony_s1_continue
    )


def _refuse_recovery_on_live_store(build: BuildCfg, repo_root: Path) -> None:
    """Recovery/Harmony knobs may not target the frozen live workdir or DB.

    Checked before the train-config pin so a recovery-capable CLI refuses
    the live store even when the referenced trainer yaml is unpinned.
    Uses the same repository-root resolution as BuildPaths.
    """
    if not _is_recovery_experiment(build):
        return
    if is_live_control_workdir(build.workdir, repo_root=repo_root):
        raise ValueError(
            "recovery configuration must not point at the live workdir or "
            f"database (data/build / data/build/state/law_v1.sqlite3); "
            f"got workdir {build.workdir!r}"
        )


def load_build_config(path: str | Path, *, allow_unpinned: bool = False) -> BuildConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    build_raw = dict(raw["build"])
    length_band = LengthBand(**build_raw.pop("length_band"))
    require_manifest = build_raw.pop("require_pretreatment_manifest", False)
    if not isinstance(require_manifest, bool):
        raise ValueError(
            "build.require_pretreatment_manifest must be a YAML boolean "
            f"(true/false), got {require_manifest!r}. Not coerced."
        )
    pretreatment_manifest = build_raw.pop("pretreatment_manifest", None)
    if pretreatment_manifest is not None:
        if not isinstance(pretreatment_manifest, str) or not pretreatment_manifest.strip():
            raise ValueError(
                "build.pretreatment_manifest must be a non-empty string path, "
                f"got {pretreatment_manifest!r}"
            )
        pretreatment_manifest = pretreatment_manifest.strip()
    if require_manifest and not pretreatment_manifest:
        raise ValueError(
            "build.pretreatment_manifest is required when "
            "require_pretreatment_manifest is true"
        )
    strata = build_raw.pop("eval_cohort_strata", None)
    if strata is not None:
        if not isinstance(strata, list) or not strata:
            raise ValueError(
                "build.eval_cohort_strata must be a non-empty YAML list of "
                f"task-type names, got {strata!r}"
            )
        if not all(isinstance(name, str) and name.strip() for name in strata):
            raise ValueError(
                "build.eval_cohort_strata entries must be non-empty strings, "
                f"got {strata!r}"
            )
        strata = tuple(name.strip() for name in strata)
        if len(set(strata)) != len(strata):
            raise ValueError(
                f"build.eval_cohort_strata repeats a stratum: {strata!r}. A "
                "repeat would double one stratum's share of the cohort."
            )
    build = BuildCfg(
        length_band=length_band,
        require_pretreatment_manifest=require_manifest,
        pretreatment_manifest=pretreatment_manifest,
        eval_cohort_strata=strata,
        **build_raw,
    )

    # Same repo-root convention as the train_config resolve below. Isolation
    # is checked here, before any provider/train work, so a recovery yaml
    # aimed at data/build never opens the live SQLite file.
    repo_root = package_repo_root()
    _refuse_recovery_on_live_store(build, repo_root)

    providers = tuple(
        ProviderCfg(
            name=p["name"],
            base_url=p["base_url"],
            api_key_env=p["api_key_env"],
            quirks=tuple(p["quirks"]),
            models=tuple(
                ModelCfg(
                    id=m["id"],
                    family=m["family"],
                    roles=tuple(m["roles"]),
                    limits=m["limits"],
                    params=m["params"],
                    role_params=dict(m.get("role_params") or {}),
                )
                for m in p["models"]
            ),
        )
        for p in raw["providers"]
    )

    assembly = _assembly_of(raw, build)

    r = raw["routing"]
    routing = RoutingCfg(
        generator=tuple(r["generator"]),
        judge=tuple(r["judge"]),
        tiebreak=tuple(r["tiebreak"]),
        probe=tuple(r["probe"]),
        family_separation=r["family_separation"],
        judge_mode=r["judge_mode"],
    )

    # build.train_config is repo-root-relative, same convention the tests
    # use - resolve against this module's own location (src/tuned/data/),
    # not against `path`, so a fixture yaml living anywhere can still point
    # at the real shared training config.
    train_cfg = load_config(repo_root / build.train_config, allow_unpinned=allow_unpinned)

    cfg = BuildConfig(
        build=build,
        providers=providers,
        routing=routing,
        assembly=assembly,
        think_open=train_cfg.data.think_open,
        think_close=train_cfg.data.think_close,
        model_repo=train_cfg.model.repo,
        model_revision=train_cfg.model.revision,
        instruction_part=train_cfg.model.instruction_part,
        response_part=train_cfg.model.response_part,
        main_dataset_path=train_cfg.train.main.dataset,
        max_seq_length=train_cfg.train.main.max_seq_length,
        push=_push_of(raw),
        transition=_transition_of(raw),
        calibration=_calibration_of(raw),
        difficulty=_difficulty_of(raw),
    )
    _validate(cfg)
    from tuned.data.prompt_registry import set_overlay

    if cfg.build.prompt_overlay:
        set_overlay(repo_root / cfg.build.prompt_overlay)
    else:
        set_overlay(None)
    return cfg
