"""Unsloth QLoRA SFT entrypoint. Run on a Kaggle GPU (accelerator "GPU T4 x2"),
never locally. Single-GPU by default; the DDP lane uses both T4s via torchrun
(unsloth auto-assigns one rank per GPU when no device_map is passed).

Savetest: python -m tuned.train.sft --config configs/law_v1.yaml --mode smoke --max-steps 4 --save-steps 2
Smoke:    python -m tuned.train.sft --config configs/law_v1.yaml --mode smoke
Resume:   python -m tuned.train.sft --config configs/law_v1.yaml --mode smoke --resume
DDP:      CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m tuned.train.sft --config configs/law_v1_ddp.yaml --mode smoke
"""

import argparse
import dataclasses
import os
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
        "max_length": run.max_seq_length,
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
        # No-op single-GPU; under DDP the trainer otherwise defaults this to
        # True and burns an extra autograd-graph traversal every step (torch
        # warned about it on the qualified 2026-08-06 SAVETEST). Every LoRA
        # param gets a grad each step, so False is safe.
        "ddp_find_unused_parameters": False,
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


def check_ddp_visibility(world_size: int, visible_gpus: int) -> None:
    """Under torchrun, every rank must see every GPU (rank N places itself on
    cuda:N). A leaked single-GPU CUDA_VISIBLE_DEVICES mask makes rank 1 die
    minutes later inside the model load with "invalid device ordinal" - die
    here in milliseconds instead."""
    if world_size > 1 and visible_gpus < world_size:
        raise SystemExit(
            f"WORLD_SIZE={world_size} but only {visible_gpus} CUDA device(s) "
            "visible - a CUDA_VISIBLE_DEVICES mask (e.g. the notebook's "
            "single-GPU default) leaked into the torchrun launch. Prefix the "
            "command with CUDA_VISIBLE_DEVICES=0,1."
        )


def read_gpu_capability() -> tuple | None:
    """Compute capability via nvidia-smi, before any CUDA library loads. None = undetermined."""
    import subprocess

    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    line = probe.stdout.strip().splitlines()[0].strip() if probe.returncode == 0 and probe.stdout.strip() else ""
    if not line:
        return None
    try:
        major, minor = line.split(".")
        return (int(major), int(minor))
    except ValueError:
        return None


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

    cap = read_gpu_capability()
    if cap is not None:
        check_gpu_capability(cap)

    # Unsloth MUST be imported before torch/transformers/trl so its patches apply.
    from unsloth import FastModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    # unsloth 2026.8.3: if bitsandbytes' native kernels fail to load, this flag
    # flips False and loader.py silently strips the -unsloth-bnb-4bit suffix AND
    # drops revision= - a doomed env would re-download ~28 GB fp16 inside the
    # watchdog. Die here in milliseconds instead.
    from unsloth import device_type as _unsloth_device

    if not getattr(_unsloth_device, "ALLOW_PREQUANTIZED_MODELS", True):
        raise SystemExit(
            "unsloth ALLOW_PREQUANTIZED_MODELS is False - bitsandbytes native "
            "kernels failed to load; the pre-quantized repo and its pinned "
            "revision would be silently swapped for a full fp16 download. "
            "Fix the bitsandbytes install; do not train."
        )

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device - in Kaggle set Accelerator to 'GPU T4 x2'")
    check_ddp_visibility(int(os.environ.get("WORLD_SIZE", "1")), torch.cuda.device_count())

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg.model.repo,
        revision=cfg.model.revision,
        max_seq_length=run.max_seq_length,
        dtype=torch.float16 if not is_bfloat16_supported() else torch.bfloat16,
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
    model.print_trainable_parameters()
    lora_modules = sorted({n.rsplit(".", 2)[0] for n, _ in model.named_parameters() if "lora_" in n})
    print(f"lora_target_modules_sample={lora_modules[:5]}")
    vision_hits = [m for m in lora_modules if "vision" in m.lower()]
    if vision_hits:
        raise SystemExit(f"LoRA attached to vision tower modules {vision_hits[:3]} - unsloth#5677 risk; fix target_modules regex")

    ds = load_dataset("json", data_files=run.dataset, split="train")
    ds = ds.map(
        lambda ex: {
            "text": tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
        },
        remove_columns=ds.column_names,
    )

    import inspect

    sft_kw = build_sft_config(cfg, run, output_dir, bf16_supported=is_bfloat16_supported())
    if "max_length" not in inspect.signature(SFTConfig.__init__).parameters:
        sft_kw["max_seq_length"] = sft_kw.pop("max_length")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            **sft_kw,
        ),
    )
    print(f"trainer_max_len={getattr(trainer.args, 'max_length', None) or getattr(trainer.args, 'max_seq_length', None)}")
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

    import math

    if not math.isfinite(stats.training_loss):
        raise SystemExit(
            f"train_loss={stats.training_loss} - fp16 divergence; first lever: set max_grad_norm: 0.3"
        )

    print(f"peak_vram_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}")

    runtime = stats.metrics.get("train_runtime")
    if runtime and not args.resume:
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
