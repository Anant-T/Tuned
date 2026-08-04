"""Typed loader for configs/law_v1.yaml. The revision pin is enforced here."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ModelCfg:
    repo: str
    revision: str | None


@dataclass
class LoraCfg:
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]


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
    smoke: RunCfg


@dataclass
class HubCfg:
    checkpoint_repo: str | None


@dataclass
class Config:
    model: ModelCfg
    lora: LoraCfg
    train: TrainCfg
    hub: HubCfg


def load_config(path: str | Path, *, allow_unpinned: bool = False) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = Config(
        model=ModelCfg(**raw["model"]),
        lora=LoraCfg(**raw["lora"]),
        train=TrainCfg(
            smoke=RunCfg(**raw["train"].pop("smoke")),
            **raw["train"],
        ),
        hub=HubCfg(**raw["hub"]),
    )
    if cfg.model.revision is None and not allow_unpinned:
        raise ValueError(
            "model.revision is null - run scripts/pin_revision.py and commit the pin"
        )
    return cfg
