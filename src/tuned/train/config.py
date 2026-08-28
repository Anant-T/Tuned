"""Typed loader for training/configs/law_v1_8b_ddp.yaml. The revision pin is enforced here."""

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
    # rsLoRA scales the adapter by alpha/sqrt(r) instead of alpha/r. Rejected
    # on two 2026-08-10 SMOKE A/Bs: the alpha-64 arm (11.3x, W&B bh920zyh) was
    # cancelled by the 0.3 clip binding every step; the clean isolate arm
    # (alpha 32 = 5.66x, clip opened to 1.5, W&B wl5estcl, 60/60) finished
    # 0.5585 vs baseline 0.5601 (-0.3%, mixed sign per-step) while running
    # 3-4x hotter grad norms (0.21-0.45, spikes to 1.0) that are incompatible
    # with the qualified 0.3 clip - same loss, less stability margin, so the
    # production lane keeps this False.
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
    dataset_repo: str | None = None       # private HF DATASET repo, e.g. tantan01/tuned-law-v1-data
    dataset_revision: str | None = None   # pinned by training/scripts/pin_dataset.py
    dataset_sha256: str | None = None     # optional integrity pin of law_v1.jsonl


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
            "model.revision is null - run training/scripts/pin_revision.py and commit the pin"
        )
    return cfg
