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
    # The held-out half of the SAME build. CONFIG-DRIVEN, NOT MODE-GATED: an
    # `if mode == "main"` in the code would mean the production memory shape
    # is never exercised by the PROBE/SMOKE ladder, on a lane whose measured
    # peaks are 12.98/13.18 GiB against the 13.5 GiB line _ReservedCeiling
    # enforces by RAISING. An eval forward materialises a [1, seq, 151936]
    # logits tensor outside unsloth's chunked-CE path, so this genuinely
    # moves the profile and goes through PROBE first like any other change to
    # it - which is only possible if the smoke lane can turn it on from the
    # config alone.
    #
    # eval_rows: 0 is "this run does not evaluate". A positive value is a
    # PROMISE: if the file cannot be resolved the run REFUSES rather than
    # quietly training without an eval signal.
    eval_dataset: str | None = None
    eval_rows: int = 0


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
