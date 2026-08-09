"""Typed loader for configs/law_v1_8b_ddp.yaml. The revision pin is enforced here."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ModelCfg:
    repo: str
    revision: str | None
    instruction_part: str
    response_part: str


@dataclass
class DataCfg:
    think_open: str
    think_close: str


@dataclass
class LoraCfg:
    r: int
    alpha: int
    dropout: float
    target_modules: list[str] | str  # list of module names, or a regex string
    # rsLoRA scales the adapter by alpha/sqrt(r) instead of alpha/r - at
    # r=32/alpha=32 that is a 5.66x jump in effective scale, so the production
    # lane never sets this; only the _rslora A/B config flips it.
    use_rslora: bool = False


@dataclass
class RunCfg:
    max_seq_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    save_steps: int
    dataset: str


@dataclass
class TrainCfg:
    seed: int
    lr: float
    warmup_ratio: float
    weight_decay: float
    optim: str
    lr_scheduler_type: str
    max_grad_norm: float
    smoke: RunCfg
    main: RunCfg


@dataclass
class HubCfg:
    checkpoint_repo: str | None


@dataclass
class Config:
    model: ModelCfg
    data: DataCfg
    lora: LoraCfg
    train: TrainCfg
    hub: HubCfg


def load_config(path: str | Path, *, allow_unpinned: bool = False) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = Config(
        model=ModelCfg(**raw["model"]),
        data=DataCfg(**raw["data"]),
        lora=LoraCfg(**raw["lora"]),
        train=TrainCfg(
            smoke=RunCfg(**raw["train"].pop("smoke")),
            main=RunCfg(**raw["train"].pop("main")),
            **raw["train"],
        ),
        hub=HubCfg(**raw["hub"]),
    )
    if cfg.model.revision is None and not allow_unpinned:
        raise ValueError(
            "model.revision is null - run scripts/pin_revision.py and commit the pin"
        )
    return cfg
