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
class BuildConfig:
    build: BuildCfg
    providers: tuple[ProviderCfg, ...]
    routing: RoutingCfg
    # Resolved from the referenced train config at load time - never
    # duplicated in configs/data_law_v1.yaml itself.
    think_open: str
    think_close: str
    model_repo: str
    model_revision: str | None
    instruction_part: str
    response_part: str
    main_dataset_path: str

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
        think_open=train_cfg.data.think_open,
        think_close=train_cfg.data.think_close,
        model_repo=train_cfg.model.repo,
        model_revision=train_cfg.model.revision,
        instruction_part=train_cfg.model.instruction_part,
        response_part=train_cfg.model.response_part,
        main_dataset_path=train_cfg.train.main.dataset,
    )
    _validate(cfg)
    return cfg
