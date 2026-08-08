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
            **raw["train"],
        ),
        hub=HubCfg(**raw["hub"]),
    )
    if cfg.model.revision is None and not allow_unpinned:
        raise ValueError(
            "model.revision is null - run scripts/pin_revision.py and commit the pin"
        )
    return cfg
