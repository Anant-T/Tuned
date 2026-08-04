# Kaggle Smoke-Run Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the training workflow from Lightning.ai to Kaggle free tier (single T4) with `Ministral-3-14B-Reasoning-2512` as the base model, ending at a runnable Kaggle smoke notebook with a Qwen3-14B escape-hatch config.

**Architecture:** The repo stays the tested source of truth; a thin committed notebook clones it on Kaggle and runs the existing CLI entrypoints. Model-specific strings (masking markers, think tags) move from code into config so the fallback model is a pure config swap. All GPU-touching behavior is behind `python -m tuned.train.sft`; everything else is local-testable.

**Tech Stack:** Python 3.12, uv, pytest, PyYAML dataclass config loader, Unsloth + TRL + transformers 5.5.0 (train extra only), Kaggle notebooks.

**Spec:** `docs/superpowers/specs/2026-08-05-kaggle-migration-design.md`

## Global Constraints

- Python `>=3.12` (pyproject pin unchanged); local dev is Windows/PowerShell with `.venv\Scripts\Activate.ps1`, **no GPU locally — never run training locally**.
- Local test command: `python -m pytest tests/ -q` (venv activated, repo root). Train extras are NOT installed locally — any test needing `transformers` must `pytest.importorskip`.
- Primary base model: `unsloth/Ministral-3-14B-Reasoning-2512-unsloth-bnb-4bit` @ revision `ec1befbd41647354531b2e09bd036cd1dc94b076`.
- Fallback base model: `unsloth/Qwen3-14B-unsloth-bnb-4bit` @ revision `46105e245750aad3be7fd1d81c21cb03a0e438ed`.
- Masking markers (verified from the pinned chat_template.jinja): Ministral `[INST]` / `[/INST]`; Qwen `<|im_start|>user\n` / `<|im_start|>assistant\n`.
- Think tags: Ministral `[THINK]` / `[/THINK]`; Qwen `<think>` / `</think>`.
- fp16 explicit everywhere; no config may ship with `revision: null`; seed 3407; LoRA r=32/alpha=32.
- YAML regexes must be single-quoted (`\.` is an invalid escape in double-quoted YAML).
- Commit after every task with the trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Config schema (markers, data tags, regex target_modules) + Ministral re-pin

**Files:**
- Modify: `src/tuned/train/config.py`
- Modify: `configs/law_v1.yaml`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: `ModelCfg` gains `instruction_part: str`, `response_part: str`; new `DataCfg(think_open: str, think_close: str)` exposed as `cfg.data`; `LoraCfg.target_modules: list[str] | str`. Later tasks read `cfg.model.instruction_part`, `cfg.model.response_part`, `cfg.data.think_open`, `cfg.data.think_close`.

- [ ] **Step 1: Update `tests/test_config.py` to the new schema (failing tests first)**

Replace the whole file with:

```python
from pathlib import Path

import pytest

from tuned.train.config import load_config

CONFIG = Path(__file__).parent.parent / "configs" / "law_v1.yaml"

TARGET_REGEX = r"language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def test_loads_repo_and_lora():
    cfg = load_config(CONFIG, allow_unpinned=True)
    assert cfg.model.repo == "unsloth/Ministral-3-14B-Reasoning-2512-unsloth-bnb-4bit"
    assert cfg.lora.r == 32
    assert cfg.lora.alpha == 32
    # Regex string scoped to the language model - keeps LoRA off the vision
    # tower (unsloth#5677 save-failure workaround).
    assert cfg.lora.target_modules == TARGET_REGEX


def test_masking_markers_and_think_tags():
    cfg = load_config(CONFIG, allow_unpinned=True)
    assert cfg.model.instruction_part == "[INST]"
    assert cfg.model.response_part == "[/INST]"
    assert cfg.data.think_open == "[THINK]"
    assert cfg.data.think_close == "[/THINK]"


def test_smoke_run_settings():
    cfg = load_config(CONFIG, allow_unpinned=True)
    assert cfg.train.smoke.max_seq_length == 2048
    assert cfg.train.smoke.max_steps == 60
    assert cfg.train.seed == 3407


def test_pinned_config_loads_strictly():
    cfg = load_config(CONFIG)
    assert cfg.model.revision == "ec1befbd41647354531b2e09bd036cd1dc94b076"


def test_unpinned_revision_rejected(tmp_path):
    import re

    tmp = tmp_path / "c.yaml"
    text = CONFIG.read_text(encoding="utf-8")
    tmp.write_text(re.sub(r"revision: \S+", "revision: null", text, count=1), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        load_config(tmp)


def test_list_target_modules_still_accepted(tmp_path):
    tmp = tmp_path / "c.yaml"
    text = CONFIG.read_text(encoding="utf-8")
    text = text.replace(
        f"target_modules: '{TARGET_REGEX}'",
        "target_modules: [q_proj, v_proj]",
    )
    tmp.write_text(text, encoding="utf-8")
    cfg = load_config(tmp)
    assert cfg.lora.target_modules == ["q_proj", "v_proj"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py -q`
Expected: FAIL — old yaml still points at Gemma, `DataCfg`/markers don't exist yet.

- [ ] **Step 3: Update `src/tuned/train/config.py`**

Replace the `ModelCfg` and `LoraCfg` dataclasses and add `DataCfg`; wire into `Config`/`load_config`:

```python
"""Typed loader for configs/law_v1.yaml. The revision pin is enforced here."""

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
```

- [ ] **Step 4: Rewrite `configs/law_v1.yaml`**

Full new content (note the single-quoted regex — `\.` is invalid inside YAML double quotes):

```yaml
model:
  repo: unsloth/Ministral-3-14B-Reasoning-2512-unsloth-bnb-4bit
  revision: ec1befbd41647354531b2e09bd036cd1dc94b076
  # Completion-masking markers, verified against this revision's chat_template.jinja:
  # user turns render as [INST]...[/INST], the assistant reply follows [/INST].
  instruction_part: '[INST]'
  response_part: '[/INST]'

data:
  # Ministral Reasoning's native scaffold; the smoke builder wraps traces in these.
  think_open: '[THINK]'
  think_close: '[/THINK]'

lora:
  r: 32
  alpha: 32
  dropout: 0.0
  # Regex, not a list: scoped to language_model so the vision tower never gets
  # LoRA modules (works around unsloth#5677 LoRA-save failure). Single quotes required.
  target_modules: 'language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'

train:
  seed: 3407
  lr: 2.0e-4
  warmup_ratio: 0.03
  weight_decay: 0.001
  optim: adamw_8bit
  lr_scheduler_type: linear
  smoke:
    max_seq_length: 2048
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 16
    max_steps: 60
    save_steps: 25
    dataset: data/smoke_v1.jsonl

hub:
  checkpoint_repo: null   # operator sets e.g. "<hf-user>/tuned-law-v1-ckpt" before the smoke run
```

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: `tests/test_config.py` PASSES. `tests/test_sft_args.py` and `tests/test_smoke_data.py` must also still pass (they don't touch the new keys). If `test_sft_args.py` fails, something in the schema broke `build_sft_config` — fix before proceeding.

- [ ] **Step 6: Commit**

```powershell
git add configs/law_v1.yaml src/tuned/train/config.py tests/test_config.py
git commit -m "feat: re-pin base to ministral-3-14b-reasoning, config-driven markers and think tags"
```

---

### Task 2: Qwen escape-hatch config

**Files:**
- Create: `configs/law_v1_qwen.yaml`
- Create: `tests/test_qwen_config.py`

**Interfaces:**
- Consumes: `load_config` from Task 1 (unchanged signature).
- Produces: a strictly-pinned fallback config selectable with `--config configs/law_v1_qwen.yaml`. No code reads it by default.

- [ ] **Step 1: Write the failing test**

Create `tests/test_qwen_config.py`:

```python
from pathlib import Path

from tuned.train.config import load_config

CONFIG = Path(__file__).parent.parent / "configs" / "law_v1_qwen.yaml"


def test_qwen_fallback_loads_strictly():
    cfg = load_config(CONFIG)  # strict: must be pinned
    assert cfg.model.repo == "unsloth/Qwen3-14B-unsloth-bnb-4bit"
    assert cfg.model.revision == "46105e245750aad3be7fd1d81c21cb03a0e438ed"
    assert cfg.model.instruction_part == "<|im_start|>user\n"
    assert cfg.model.response_part == "<|im_start|>assistant\n"
    assert cfg.data.think_open == "<think>"
    assert cfg.data.think_close == "</think>"
    # Qwen3 is text-only - plain list, no vision tower to exclude.
    assert cfg.lora.target_modules == [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]


def test_qwen_fallback_matches_primary_hyperparams():
    primary = load_config(CONFIG.parent / "law_v1.yaml")
    qwen = load_config(CONFIG)
    assert qwen.train == primary.train
    assert qwen.lora.r == primary.lora.r
    assert qwen.lora.alpha == primary.lora.alpha
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_qwen_config.py -q`
Expected: FAIL — file not found.

- [ ] **Step 3: Create `configs/law_v1_qwen.yaml`**

```yaml
# Escape hatch: switch with --config configs/law_v1_qwen.yaml if the Ministral
# LoRA-save path fails on Kaggle (unsloth#5677) after one session of debugging.
model:
  repo: unsloth/Qwen3-14B-unsloth-bnb-4bit
  revision: 46105e245750aad3be7fd1d81c21cb03a0e438ed
  # Qwen3 ChatML markers (unsloth's documented train_on_responses_only pair).
  instruction_part: "<|im_start|>user\n"
  response_part: "<|im_start|>assistant\n"

data:
  think_open: '<think>'
  think_close: '</think>'

lora:
  r: 32
  alpha: 32
  dropout: 0.0
  # Text-only model - no vision tower, plain module list is safe.
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

train:
  seed: 3407
  lr: 2.0e-4
  warmup_ratio: 0.03
  weight_decay: 0.001
  optim: adamw_8bit
  lr_scheduler_type: linear
  smoke:
    max_seq_length: 2048
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 16
    max_steps: 60
    save_steps: 25
    dataset: data/smoke_v1.jsonl

hub:
  checkpoint_repo: null   # operator sets before use (may reuse the same ckpt repo)
```

Note: the markers use double quotes deliberately — they contain `\n`, which double-quoted YAML turns into a real newline (single quotes would keep it literal, which is wrong here).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_qwen_config.py -q`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```powershell
git add configs/law_v1_qwen.yaml tests/test_qwen_config.py
git commit -m "feat: qwen3-14b escape-hatch config"
```

---

### Task 3: Smoke builder emits think tags

**Files:**
- Modify: `src/tuned/data/smoke.py`
- Modify: `tests/test_smoke_data.py`

**Interfaces:**
- Consumes: `cfg.data.think_open` / `cfg.data.think_close` (Task 1).
- Produces: `format_example(problem, reasoning, solution, think_open="", think_close="")` and `build_smoke(out_path, n=1000, rows=None, think_open="", think_close="")`. CLI: `python -m tuned.data.smoke --config configs/law_v1.yaml`.

- [ ] **Step 1: Add failing tests**

Append to `tests/test_smoke_data.py` (keep every existing test unchanged — they cover the untagged default path):

```python
def test_format_example_with_think_tags():
    ex = format_example(
        "What is 2+2?", "Two plus two makes four.", "4",
        think_open="[THINK]", think_close="[/THINK]",
    )
    content = ex["messages"][1]["content"]
    assert content == "[THINK]Two plus two makes four.[/THINK]4"


def test_build_smoke_wraps_with_think_tags(tmp_path):
    rows = [
        {
            "conversations": [
                {"from": "user", "value": "q0"},
                {"from": "assistant", "value": "<|begin_of_thought|>r0<|end_of_thought|>\n<|begin_of_solution|>s0<|end_of_solution|>"},
            ]
        },
    ]
    out = tmp_path / "smoke.jsonl"
    n = build_smoke(out, n=1, rows=iter(rows), think_open="[THINK]", think_close="[/THINK]")
    assert n == 1
    data = json.loads(out.read_text(encoding="utf-8").strip())
    assert data["messages"][1]["content"] == "[THINK]r0[/THINK]s0"
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `python -m pytest tests/test_smoke_data.py -q`
Expected: 2 new FAIL (unexpected keyword `think_open`), existing tests PASS.

- [ ] **Step 3: Implement**

In `src/tuned/data/smoke.py`:

1. Replace the module docstring's second paragraph — it currently says tag fidelity is "handled in the main data pipeline, not here", which is no longer true:

```python
"""Build the ~1k-example smoke dataset from OpenThoughts-114k (Apache-2.0).

Assistant content is wrapped in the base model's reasoning scaffold
(config data.think_open / data.think_close), e.g. [THINK]trace[/THINK]solution
for Ministral-3 Reasoning, so training matches the model's native template.
"""
```

2. Replace `format_example` with:

```python
def format_example(
    problem: str,
    reasoning: str,
    solution: str,
    think_open: str = "",
    think_close: str = "",
) -> dict:
    if think_open or think_close:
        content = f"{think_open}{reasoning}{think_close}{solution}"
    else:
        content = f"{reasoning}\n\n{solution}"
    return {
        "messages": [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": content},
        ]
    }
```

3. Give `build_smoke` the passthrough params — change its signature to
`def build_smoke(out_path: str | Path, n: int = 1000, rows=None, think_open: str = "", think_close: str = "") -> int:`
and change the `f.write(...)` line to:

```python
            f.write(
                json.dumps(
                    format_example(problem, reasoning, solution, think_open, think_close)
                ) + "\n"
            )
```

4. Replace the `__main__` block with a config-driven CLI:

```python
if __name__ == "__main__":
    import argparse

    from tuned.train.config import load_config

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/law_v1.yaml")
    p.add_argument("--out", default="data/smoke_v1.jsonl")
    args = p.parse_args()

    cfg = load_config(args.config)
    count = build_smoke(
        args.out,
        think_open=cfg.data.think_open,
        think_close=cfg.data.think_close,
    )
    print(f"wrote {count} examples to {args.out}")
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/tuned/data/smoke.py tests/test_smoke_data.py
git commit -m "feat: smoke builder wraps traces in config-driven think tags"
```

---

### Task 4: sft.py — fp16, GPU preflight, config markers, CLI overrides, telemetry

**Files:**
- Modify: `src/tuned/train/sft.py`
- Modify: `tests/test_sft_args.py`

**Interfaces:**
- Consumes: `cfg.model.instruction_part` / `cfg.model.response_part` (Task 1); `RunCfg` dataclass.
- Produces: `build_sft_config(cfg, run, output_dir, bf16_supported=False)` (new kwarg, sets `fp16`/`bf16`); `apply_overrides(run, max_steps=None, save_steps=None) -> RunCfg`; `check_gpu_capability(capability: tuple) -> None` (raises `SystemExit` below (7, 0)); CLI flags `--max-steps N` / `--save-steps N`.

- [ ] **Step 1: Add failing tests**

Append to `tests/test_sft_args.py`:

```python
from tuned.train.sft import apply_overrides, check_gpu_capability

import pytest


def test_precision_flags_fp16_when_no_bf16():
    cfg = load_config(CONFIG, allow_unpinned=True)
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o", bf16_supported=False)
    assert kw["fp16"] is True
    assert kw["bf16"] is False


def test_precision_flags_bf16_when_supported():
    cfg = load_config(CONFIG, allow_unpinned=True)
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o", bf16_supported=True)
    assert kw["fp16"] is False
    assert kw["bf16"] is True


def test_apply_overrides_replaces_steps():
    cfg = load_config(CONFIG, allow_unpinned=True)
    run = apply_overrides(cfg.train.smoke, max_steps=4, save_steps=2)
    assert run.max_steps == 4
    assert run.save_steps == 2
    # untouched fields survive
    assert run.max_seq_length == cfg.train.smoke.max_seq_length
    # original is not mutated
    assert cfg.train.smoke.max_steps == 60


def test_apply_overrides_none_is_noop():
    cfg = load_config(CONFIG, allow_unpinned=True)
    run = apply_overrides(cfg.train.smoke)
    assert run == cfg.train.smoke


def test_capability_gate_rejects_p100():
    with pytest.raises(SystemExit, match="T4 x2"):
        check_gpu_capability((6, 0))


def test_capability_gate_accepts_t4():
    check_gpu_capability((7, 5))  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sft_args.py -q`
Expected: new tests FAIL (ImportError on `apply_overrides`), existing two PASS.

- [ ] **Step 3: Implement in `src/tuned/train/sft.py`**

Replace the whole file with:

```python
"""Unsloth QLoRA SFT entrypoint. Run on a Kaggle GPU (accelerator "GPU T4 x2",
training pinned to one T4), never locally.

Savetest: python -m tuned.train.sft --config configs/law_v1.yaml --mode smoke --max-steps 4 --save-steps 2
Smoke:    python -m tuned.train.sft --config configs/law_v1.yaml --mode smoke
Resume:   python -m tuned.train.sft --config configs/law_v1.yaml --mode smoke --resume
"""

import argparse
import dataclasses
from pathlib import Path

from tuned.train.config import Config, RunCfg, load_config


def build_sft_config(
    cfg: Config, run: RunCfg, output_dir: str, bf16_supported: bool = False
) -> dict:
    kw = {
        "output_dir": output_dir,
        "max_steps": run.max_steps,
        "per_device_train_batch_size": run.per_device_train_batch_size,
        "gradient_accumulation_steps": run.gradient_accumulation_steps,
        "learning_rate": cfg.train.lr,
        "warmup_ratio": cfg.train.warmup_ratio,
        "weight_decay": cfg.train.weight_decay,
        "optim": cfg.train.optim,
        "lr_scheduler_type": cfg.train.lr_scheduler_type,
        "seed": cfg.train.seed,
        # T4 (sm_75) has no bf16; flags are explicit so a bf16 default can
        # never sneak in ("BFloat16 != Half" is the classic Kaggle failure).
        "fp16": not bf16_supported,
        "bf16": bf16_supported,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": run.save_steps,
        "save_total_limit": 2,
        "report_to": "none",
    }
    if cfg.hub.checkpoint_repo is not None:
        kw.update(
            push_to_hub=True,
            hub_model_id=cfg.hub.checkpoint_repo,
            hub_strategy="checkpoint",
            hub_private_repo=True,
        )
    return kw


def apply_overrides(
    run: RunCfg, max_steps: int | None = None, save_steps: int | None = None
) -> RunCfg:
    if max_steps is not None:
        run = dataclasses.replace(run, max_steps=max_steps)
    if save_steps is not None:
        run = dataclasses.replace(run, save_steps=save_steps)
    return run


def check_gpu_capability(capability: tuple) -> None:
    """Abort before any quota-burning work on unsupported GPUs (e.g. P100)."""
    if tuple(capability) < (7, 0):
        raise SystemExit(
            f"GPU compute capability {capability[0]}.{capability[1]} is below 7.0 "
            "(P100 is 6.0 - unsupported by current unsloth/bitsandbytes). "
            "In Kaggle: Settings -> Accelerator -> 'GPU T4 x2'."
        )


def print_versions() -> None:
    from importlib.metadata import version

    for pkg in ("torch", "transformers", "trl", "unsloth", "bitsandbytes", "peft"):
        try:
            print(f"{pkg}=={version(pkg)}")
        except Exception:
            print(f"{pkg}: not installed")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/law_v1.yaml")
    p.add_argument("--mode", choices=["smoke"], default="smoke")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-hub", action="store_true")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--save-steps", type=int, default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)  # strict: refuses unpinned revision

    # Preflight - before any GPU import or model load.
    if cfg.hub.checkpoint_repo is None and not args.no_hub:
        raise SystemExit(
            "hub.checkpoint_repo is null - set it in the config (checkpoint "
            "push/resume is the point of the smoke run), or pass --no-hub to "
            "train without it"
        )
    if args.resume and cfg.hub.checkpoint_repo is None:
        raise SystemExit("--resume requires hub.checkpoint_repo in the config")

    run = apply_overrides(
        getattr(cfg.train, args.mode), max_steps=args.max_steps, save_steps=args.save_steps
    )
    output_dir = f"outputs/{args.mode}"

    print_versions()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device - in Kaggle set Accelerator to 'GPU T4 x2'")
    check_gpu_capability(torch.cuda.get_device_capability(0))

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg.model.repo,
        revision=cfg.model.revision,
        max_seq_length=run.max_seq_length,
        dtype=torch.float16 if not is_bfloat16_supported() else None,
        load_in_4bit=True,
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.train.seed,
    )

    ds = load_dataset("json", data_files=run.dataset, split="train")
    ds = ds.map(
        lambda ex: {
            "text": tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
        },
        remove_columns=ds.column_names,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            **build_sft_config(cfg, run, output_dir, bf16_supported=is_bfloat16_supported()),
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part=cfg.model.instruction_part,
        response_part=cfg.model.response_part,
    )

    resume = False
    if args.resume:
        from huggingface_hub import snapshot_download

        snapshot_download(
            cfg.hub.checkpoint_repo,
            allow_patterns=["last-checkpoint/*"],
            local_dir=output_dir,
        )
        resume = f"{output_dir}/last-checkpoint"
        if not Path(resume).is_dir():
            raise SystemExit(f"no last-checkpoint found in {cfg.hub.checkpoint_repo}")

    stats = trainer.train(resume_from_checkpoint=resume)
    print(f"train_loss={stats.training_loss:.4f}")
    print(f"peak_vram_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}")

    runtime = stats.metrics.get("train_runtime")
    if runtime:
        tokens = (
            run.max_steps
            * run.per_device_train_batch_size
            * run.gradient_accumulation_steps
            * run.max_seq_length
        )
        print(
            f"train_runtime_s={runtime:.0f} "
            f"approx_tokens_per_sec={tokens / runtime:.0f} "
            "(upper bound - assumes every sequence is max_seq_length)"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all PASS (the GPU branch of `main()` is never executed by tests).

- [ ] **Step 5: Commit**

```powershell
git add src/tuned/train/sft.py tests/test_sft_args.py
git commit -m "feat: fp16-explicit sft with gpu preflight, config markers, savetest overrides"
```

---

### Task 5: Template-drift test (runs on Kaggle, skips locally)

**Files:**
- Create: `tests/test_chat_template.py`

**Interfaces:**
- Consumes: both configs (Tasks 1-2); `transformers` (train extra — absent locally, so the test self-skips via `importorskip`).
- Produces: nothing consumed later; this is the tripwire that catches a transformers upgrade silently changing the chat template.

- [ ] **Step 1: Write the test**

Create `tests/test_chat_template.py`:

```python
"""Render a sample conversation through the real tokenizer and assert the
config's masking markers appear. Skips locally (no [train] extra); runs on
Kaggle where transformers is installed and internet is on.

Catches: a transformers/tokenizer change silently altering the template, or a
config pointing at markers from the wrong template family (ChatML vs Mistral).
"""

from pathlib import Path

import pytest

from tuned.train.config import load_config

transformers = pytest.importorskip("transformers")

CONFIGS = Path(__file__).parent.parent / "configs"


@pytest.mark.parametrize("name", ["law_v1.yaml", "law_v1_qwen.yaml"])
def test_markers_and_think_tags_render(name):
    cfg = load_config(CONFIGS / name)
    tok = transformers.AutoTokenizer.from_pretrained(
        cfg.model.repo, revision=cfg.model.revision
    )
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": f"{cfg.data.think_open}Adding.{cfg.data.think_close}4",
        },
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    assert cfg.model.instruction_part in text
    assert cfg.model.response_part in text
    # instruction marker must precede response marker
    assert text.index(cfg.model.instruction_part) < text.index(cfg.model.response_part)
    # the reasoning scaffold survives rendering (single-turn: last assistant
    # message - Qwen3's template only strips think blocks from earlier turns)
    assert cfg.data.think_open in text
    # the trainable answer sits after the response marker
    assert text.rindex("4") > text.index(cfg.model.response_part)
```

- [ ] **Step 2: Run locally to verify it skips (not fails)**

Run: `python -m pytest tests/test_chat_template.py -q`
Expected: `2 skipped` (transformers not installed locally). If it errors instead of skipping, the `importorskip` line is misplaced — it must run at module level before any `transformers` use.

- [ ] **Step 3: Commit**

```powershell
git add tests/test_chat_template.py
git commit -m "test: template-drift tripwire for masking markers and think tags"
```

---

### Task 6: Package metadata — Kaggle detection, pyproject, docstrings

**Files:**
- Modify: `src/tuned/__init__.py`
- Modify: `pyproject.toml`
- Create: `tests/test_where_am_i.py`

**Interfaces:**
- Produces: `where_am_i()` output gains a `"platform"` key (`"kaggle (Interactive)"`, `"kaggle (Batch)"`, or `"local/other"`). `[train]` extra gains `hf_transfer`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_where_am_i.py`:

```python
from tuned import where_am_i


def test_detects_kaggle(monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    info = where_am_i()
    assert info["platform"] == "kaggle (Interactive)"


def test_detects_non_kaggle(monkeypatch):
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    info = where_am_i()
    assert info["platform"] == "local/other"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_where_am_i.py -q`
Expected: FAIL with `KeyError: 'platform'`.

- [ ] **Step 3: Implement**

In `src/tuned/__init__.py`:

1. Replace the module docstring (line 1) with:

```python
"""tuned - multi-adapter fine-tuning of Ministral-3-14B-Reasoning on Kaggle free-tier GPUs.

Model weights never live in this package.
"""
```

2. In `where_am_i()`, replace the docstring line "Run this on a fresh Studio…" with "Run this in a fresh Kaggle session to verify the environment." and add `platform` to the `info` dict:

```python
    run_type = os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
    info = {
        "tuned_version": __version__,
        "platform": f"kaggle ({run_type})" if run_type else "local/other",
        "host": platform.node(),
        "python": sys.version.split()[0],
        "gpu": gpu,
        "package_path": os.path.dirname(__file__),
    }
```

In `pyproject.toml`:

3. Change the description line to:

```toml
description = "Multi-adapter fine-tuning of Ministral-3-14B-Reasoning on Kaggle free-tier GPUs."
```

4. Add `hf_transfer` to the train extra (fast Hub downloads; the notebook sets `HF_HUB_ENABLE_HF_TRANSFER=1` and the ~9 GB base model re-downloads every session because the cache lives in ephemeral `/tmp`):

```toml
train = [
    "unsloth>=2026.8.2",
    "transformers==5.5.0",
    "trl",
    "datasets",
    "huggingface_hub",
    "hf_transfer",
]
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/tuned/__init__.py pyproject.toml tests/test_where_am_i.py
git commit -m "feat: kaggle platform detection, hf_transfer dep, kaggle metadata"
```

---

### Task 7: Kaggle notebook

**Files:**
- Create: `notebooks/kaggle_smoke.ipynb`
- Create: `tests/test_notebook.py`

**Interfaces:**
- Consumes: the CLI entrypoints (`python -m tuned.data.smoke --config …`, `python -m tuned.train.sft …` with the Task 4 flags), Kaggle Secrets (`HF_TOKEN`).
- Produces: the single artifact the operator uploads to Kaggle.

- [ ] **Step 1: Write the failing test**

Create `tests/test_notebook.py`:

```python
import json
from pathlib import Path

NB = Path(__file__).parent.parent / "notebooks" / "kaggle_smoke.ipynb"


def test_notebook_is_valid_and_complete():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    sources = ["".join(c["source"]) for c in nb["cells"]]
    joined = "\n".join(sources)
    # the operator's mode switch exists and defaults to the cheap gate
    assert 'MODE = "SAVETEST"' in joined
    # single-GPU pin and scratch cache - never /kaggle/working
    assert 'CUDA_VISIBLE_DEVICES"] = "0"' in joined
    assert "/tmp/hf_cache" in joined
    assert "/kaggle/working/hf_cache" not in joined
    # secrets come from Kaggle, never hardcoded
    assert "UserSecretsClient" in joined
    assert "hf_" not in joined.replace("hf_cache", "").replace("hf_transfer", "").replace("HF_", "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_notebook.py -q`
Expected: FAIL — file not found.

- [ ] **Step 3: Create `notebooks/kaggle_smoke.ipynb`**

Write the file with exactly this JSON (each cell's `source` is shown as readable code here; store each line as a list item ending in `\n` per nbformat):

Cell 1 — markdown:

```markdown
# tuned - Kaggle smoke run
Prereqs: phone-verified account, Accelerator = **GPU T4 x2** (never P100), Internet **On**,
`HF_TOKEN` added under Add-ons -> Secrets. Set `MODE` below, then Run All
(SAVETEST interactively first; SMOKE via *Save & Run All* in the background).
```

Cell 2 — code (env + GPU gate + disk report):

```python
MODE = "SAVETEST"  # SAVETEST (4-step save/push gate) | SMOKE (60 steps) | RESUME

import os, subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # single-T4 training (spec: DDP deferred)
os.environ["HF_HOME"] = "/tmp/hf_cache"          # scratch, NOT the 20GB persisted /kaggle/working
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
print(gpus)
assert "T4" in gpus, "Wrong accelerator - Settings -> Accelerator -> 'GPU T4 x2'"
print(subprocess.run(["df", "-h", "/tmp", "/kaggle/working"], capture_output=True, text=True).stdout)
```

Cell 3 — code (clone):

```python
%cd /tmp
!rm -rf /tmp/tuned
!git clone --depth 1 https://github.com/Anant-T/Tuned /tmp/tuned
%cd /tmp/tuned
```

Cell 4 — code (deps):

```python
!pip install -q uv
!uv pip install --system -e ".[dev,train]"
```

Cell 5 — code (secrets):

```python
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
print("HF token loaded (not printed).")
```

Cell 6 — code (versions + local tests; the template-drift test runs here for real):

```python
from importlib.metadata import version

for pkg in ("torch", "transformers", "trl", "unsloth", "bitsandbytes", "peft", "hf_transfer"):
    try:
        print(f"{pkg}=={version(pkg)}")
    except Exception:
        print(f"{pkg}: NOT INSTALLED")

!python -m pytest tests/ -q
```

Cell 7 — code (dataset):

```python
!python -m tuned.data.smoke --config configs/law_v1.yaml
```

Cell 8 — code (train per MODE):

```python
CONFIG = "configs/law_v1.yaml"  # escape hatch: configs/law_v1_qwen.yaml (see runbook)

if MODE == "SAVETEST":
    !python -m tuned.train.sft --config {CONFIG} --mode smoke --max-steps 4 --save-steps 2
elif MODE == "SMOKE":
    !python -m tuned.train.sft --config {CONFIG} --mode smoke
elif MODE == "RESUME":
    !python -m tuned.train.sft --config {CONFIG} --mode smoke --resume
else:
    raise ValueError(f"unknown MODE {MODE!r}")
```

Cell 9 — markdown (green criteria):

```markdown
## Green means
- **SAVETEST**: no `# of LoRAs ... does not match` error (unsloth#5677); `last-checkpoint/`
  visible in the private HF checkpoint repo. If it fails twice after the regex scoping,
  switch `CONFIG` above to `configs/law_v1_qwen.yaml` (see runbook in the plan doc).
- **SMOKE**: 60 steps complete, loss trending down, **no NaN** (fp16 canary),
  `peak_vram_gb` < 14. Expected duration 4-6 h - record `approx_tokens_per_sec`
  and total session hours for the main-run plan.
- **RESUME**: run in a *fresh* session; training continues from step 25/50, not step 0.
```

Notebook JSON skeleton (fill `cells` with the 9 cells above, code cells with `"outputs": [], "execution_count": null`):

```json
{
  "nbformat": 4,
  "nbformat_minor": 5,
  "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}
  },
  "cells": [ ... ]
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_notebook.py -q`
Expected: PASS. Also validate the JSON parses: `python -c "import json; json.load(open('notebooks/kaggle_smoke.ipynb'))"`.

- [ ] **Step 5: Commit**

```powershell
git add notebooks/kaggle_smoke.ipynb tests/test_notebook.py
git commit -m "feat: kaggle smoke notebook (savetest/smoke/resume modes)"
```

---

### Task 8: README rewrite + Lightning cleanup + final green suite

**Files:**
- Modify: `README.md`
- Delete: `scripts/lightning_bootstrap.sh`

**Interfaces:**
- Consumes: everything above (documents the final workflow).
- Produces: the repo's public story matches reality; no Lightning references outside `docs/` history.

- [ ] **Step 1: Delete the Lightning bootstrap**

```powershell
git rm scripts/lightning_bootstrap.sh
```

- [ ] **Step 2: Rewrite `README.md`**

Full new content:

```markdown
# tuned

Local code, trained on Kaggle free-tier GPUs ($0). Multi-adapter fine-tuning of
Ministral-3-14B-Reasoning — one LoRA per domain (Indian law first).

    edit locally -> git push -> Kaggle notebook: clone -> train

## Layout

| Path | Purpose |
|---|---|
| `src/tuned/` | Importable package (data, train — eval and serve arrive with the main-run plan). |
| `configs/law_v1.yaml` | Single source of truth: model pin, LoRA, markers, run settings. |
| `configs/law_v1_qwen.yaml` | Escape hatch (Qwen3-14B) if the Ministral LoRA-save bug fires. |
| `notebooks/kaggle_smoke.ipynb` | The one artifact uploaded to Kaggle; clones this repo and runs the CLI. |
| `scripts/` | Revision pinning. |
| `docs/superpowers/` | Design specs and implementation plans. |

## Local setup (Windows, no GPU)

    uv venv
    .venv\Scripts\Activate.ps1
    uv pip install -e ".[dev]"
    python -m pytest tests/ -q

Training deps (`[train]`: unsloth, transformers 5.5.0) install only on Kaggle.
The template-drift test self-skips locally and runs on Kaggle.

## Kaggle setup (once)

1. kaggle.com account -> Settings -> verify phone number (gates GPU + internet).
2. Create a private HF checkpoint repo and a **write** token.
3. Set `hub.checkpoint_repo` in `configs/law_v1.yaml` to `<hf-user>/tuned-law-v1-ckpt`; commit and push.
4. Kaggle -> Create -> Notebook -> File -> Import Notebook -> upload `notebooks/kaggle_smoke.ipynb`.
5. Notebook settings: Accelerator **GPU T4 x2** (never P100 — unsupported), Internet **On**.
6. Add-ons -> Secrets -> add `HF_TOKEN`.

## Smoke run (free, ~5-7 GPU-h of the 30 h/week quota)

1. `MODE = "SAVETEST"` -> Run All interactively (~15 min). Green = checkpoint in the
   HF repo, no LoRA-save error. This gate exists because of unsloth#5677.
2. `MODE = "SMOKE"` -> **Save & Run All** (background, 4-6 h; immune to the
   20-min idle timeout). Green = loss down, no NaN, peak VRAM < 14 GB.
3. Fresh session, `MODE = "RESUME"` -> verifies checkpoint resume from the Hub.

If SAVETEST fails on the LoRA save after one session of debugging: set
`CONFIG = "configs/law_v1_qwen.yaml"` in the notebook and rerun from step 1.

## Rules that keep adapters swappable

- The base model revision is **pinned** in the config. Never train against
  `main`. Re-pin deliberately with `python scripts/pin_revision.py`.
- Every domain adapter uses the same base revision and the same
  `lora.target_modules` scoping.
- fp16 only on T4 (no bf16) — precision flags are explicit in code, never "auto".
- Save adapters only; never merge to 16-bit on Kaggle (blows the 20 GB disk).
- Secrets are env vars / Kaggle Secrets. `data/` and `outputs/` never enter git.
```

- [ ] **Step 3: Grep for leftover Lightning references**

Run: `git grep -il lightning -- ':!docs/'`
Expected: no output (docs/ keeps its dated history). If anything else surfaces, fix it now.

- [ ] **Step 4: Full suite, then commit**

Run: `python -m pytest tests/ -q`
Expected: all PASS, template-drift tests skipped.

```powershell
git add README.md
git commit -m "docs: kaggle workflow README, remove lightning bootstrap"
```

- [ ] **Step 5: Push (the notebook clones `main`, so push is part of done)**

```powershell
git push origin main
```

---

### Task 9: Operator runbook (no code — the user executes this on Kaggle)

**Files:** none (this section IS the deliverable; it stays in this plan doc).

Prerequisites checklist (once):
- [ ] Kaggle account phone-verified (Settings → Verify phone; gates GPU and internet).
- [ ] HF **write** token created; private checkpoint repo name chosen (e.g. `<hf-user>/tuned-law-v1-ckpt`).
- [ ] `hub.checkpoint_repo` set in `configs/law_v1.yaml` (and `law_v1_qwen.yaml` if ever used), committed, pushed.
- [ ] `notebooks/kaggle_smoke.ipynb` imported into a new private Kaggle notebook; Accelerator = **GPU T4 x2**; Internet = **On**; Secret `HF_TOKEN` added.

Run sequence:
- [ ] **Savetest (interactive, ~15 min GPU):** `MODE="SAVETEST"`, Run All. Watch cell 6's version printout — record torch/CUDA/bitsandbytes versions in the repo (issue or commit message) for pin decisions. Green gate: checkpoint pushed, no `# of LoRAs` error.
- [ ] If savetest fails on LoRA save: retry once after reading the error; if it is the #5677 module-count signature, debug within this one session only (check that no `vision_tower` modules appear in `model.print_trainable_parameters()`); else flip `CONFIG` to `configs/law_v1_qwen.yaml` and repeat savetest.
- [ ] **Smoke (background, 4-6 h GPU):** `MODE="SMOKE"`, then **Save & Run All (Commit)**. Check back after ~6 h; the committed notebook's output shows loss curve, `peak_vram_gb`, `approx_tokens_per_sec`.
- [ ] **Resume (interactive, ~30 min GPU):** fresh session, `MODE="RESUME"`, Run All. Green = training resumes from the last saved step, not step 0.
- [ ] Record in the next planning session: measured tokens/sec, total GPU-hours spent, peak VRAM, and which model config went green. These numbers drive the main-run plan (quota budget, DDP probe decision, 4096-token data length policy).

---

## Self-review notes

- Spec §3 (config layer) → Tasks 1-2. §4 (sft.py) → Task 4. §5 (smoke builder) → Task 3. §6 (notebook) → Task 7. §7 (cleanup) → Tasks 6, 8. §8 (tests) → Tasks 1-7 inline + Task 5. §9 (success criteria) → Task 9 runbook. §10 is informational for the next plan — no task needed.
- Marker values and both revision pins are live-verified (HF API / raw template fetch, 2026-08-05) — no implementation-time lookups remain.
- Type consistency: `build_sft_config(cfg, run, output_dir, bf16_supported)` matches between Task 4 code and tests; `apply_overrides` and `check_gpu_capability` defined and tested in the same task; `DataCfg` fields (`think_open`/`think_close`) used identically in Tasks 3, 5, and the configs.
```
