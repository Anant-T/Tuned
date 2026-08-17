"""Typed loader for configs/data_law_v1.yaml, the dataset-curation build config.

This config is churn-heavy (API providers/models/limits change weekly) so it
lives separately from the training config and only ever REFERENCES it
(build.train_config) for anything the trainer owns - think tags, tokenizer
pin, dataset path. load_build_config resolves those fields out of the
referenced training config at load time so they are never duplicated here.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

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
    # duplicated in configs/data_law_v1.yaml itself.
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


def load_build_config(path: str | Path, *, allow_unpinned: bool = False) -> BuildConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    build_raw = dict(raw["build"])
    length_band = LengthBand(**build_raw.pop("length_band"))
    build = BuildCfg(length_band=length_band, **build_raw)

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
    repo_root = Path(__file__).resolve().parents[3]
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
    )
    _validate(cfg)
    return cfg
