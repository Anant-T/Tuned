"""Unsloth QLoRA SFT entrypoint. Run on a Kaggle GPU (accelerator "GPU T4 x2"),
never locally. Single-GPU by default; the DDP lane uses both T4s via torchrun
(unsloth auto-assigns one rank per GPU when no device_map is passed).

Savetest: python -m tuned.train.sft --config configs/law_v1.yaml --mode smoke --max-steps 4 --save-steps 2
Smoke:    python -m tuned.train.sft --config configs/law_v1.yaml --mode smoke
Resume:   python -m tuned.train.sft --config configs/law_v1.yaml --mode smoke --resume
DDP:      CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m tuned.train.sft --config configs/law_v1_ddp.yaml --mode smoke
MP:       CUDA_VISIBLE_DEVICES=0,1 python -m tuned.train.sft --config configs/law_v1_mp.yaml --mode smoke
MP probe: CUDA_VISIBLE_DEVICES=0,1 python -m tuned.train.sft --config configs/law_v1_mp.yaml --mode smoke --max-steps 2 --no-hub --dataset data/probe_long.jsonl --max-seq-length 8192
"""

import argparse
import dataclasses
import math
import os
from pathlib import Path

from tuned.train.config import Config, HubCfg, RunCfg, load_config


class _NonFiniteWindow:
    """Divergence detector for the trainer's log stream.

    Keyed on grad_norm, never loss: logging_nan_inf_filter defaults True and
    rewrites nan losses in logs, so loss cannot show divergence. grace_steps
    covers DDP GradScaler calibration (steps 1-2 log grad_norm=nan on a
    healthy run - observed in both green 8B gates); after that only `window`
    CONSECUTIVE non-finite values count - a lone nan is ordinary GradScaler
    backoff. Unparseable values neither advance nor reset the streak.
    """

    def __init__(self, grace_steps: int = 2, window: int = 3):
        self.grace_steps = grace_steps
        self.window = window
        self._streak = 0

    def observe(self, step: int, grad_norm) -> bool:
        if step <= self.grace_steps:
            return False
        try:
            finite = math.isfinite(float(grad_norm))
        except (TypeError, ValueError):
            return False
        self._streak = 0 if finite else self._streak + 1
        return self._streak >= self.window


def resolve_model_source(
    repo: str, revision: str | None, staged_path: str | None
) -> tuple[str, str | None]:
    """Prefer a pre-staged local snapshot (TUNED_MODEL_PATH, set by the
    notebook after verifying the staged REVISION.txt against the config pin)
    over the hub repo. A local path carries no revision - and loading by path
    never touches the network, which sidesteps both the hub-stall failure
    class and unsloth's history of ignoring HF_HUB_OFFLINE (unsloth#5316)."""
    if staged_path:
        p = Path(staged_path)
        if not (p / "config.json").is_file():
            raise SystemExit(
                f"TUNED_MODEL_PATH={staged_path} has no config.json - not a model snapshot"
            )
        return str(p), None
    return repo, revision


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
        # warmup_ratio is deprecated in transformers 5.5 (it logged lr=0 in the
        # 2026-08-07 MP probe); the ratio stays the config's semantic knob and
        # is converted to steps here.
        "warmup_steps": max(0, round(cfg.train.warmup_ratio * run.max_steps)),
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
    run: RunCfg,
    max_steps: int | None = None,
    save_steps: int | None = None,
    dataset: str | None = None,
    max_seq_length: int | None = None,
) -> RunCfg:
    if max_steps is not None:
        run = dataclasses.replace(run, max_steps=max_steps)
    if save_steps is not None:
        run = dataclasses.replace(run, save_steps=save_steps)
    if dataset is not None:
        run = dataclasses.replace(run, dataset=dataset)
    if max_seq_length is not None:
        run = dataclasses.replace(run, max_seq_length=max_seq_length)
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


def check_mp_torchrun_conflict(device_map: str | None, world_size: int) -> None:
    """Model-parallel (device_map) and torchrun DDP are mutually exclusive:
    each rank would try to split the model across all GPUs while DDP also
    replicates it per rank. Die before the model load."""
    if device_map is not None and world_size > 1:
        raise SystemExit(
            f"model.device_map={device_map!r} under torchrun (WORLD_SIZE="
            f"{world_size}) - the MP lane launches with plain python, not "
            "torchrun. Use the DDP config for torchrun launches."
        )


def check_mp_gpu_count(device_map: str | None, visible_gpus: int) -> None:
    """A leaked single-GPU CUDA_VISIBLE_DEVICES mask would make device_map
    cram the full model onto one T4 and OOM mid-load. Die in milliseconds."""
    if device_map is not None and visible_gpus < 2:
        raise SystemExit(
            f"model.device_map={device_map!r} but only {visible_gpus} CUDA "
            "device(s) visible - the MP lane needs both T4s. Prefix the "
            "launch with CUDA_VISIBLE_DEVICES=0,1."
        )


def check_max_memory_requires_device_map(device_map: str | None, max_memory: dict | None) -> None:
    """max_memory only shapes a device_map split; alone it silently does
    nothing - refuse the misconfiguration instead."""
    if max_memory is not None and device_map is None:
        raise SystemExit(
            "model.max_memory is set but model.device_map is null - max_memory "
            "only applies to a device_map split. Set device_map: balanced or "
            "remove max_memory."
        )


def check_model_split(param_devices: list[str]) -> None:
    """After load with a device_map, parameters must actually live on >= 2
    CUDA devices - a silently ignored device_map looks green until OOM."""
    cuda_devices = {d for d in param_devices if d.startswith("cuda")}
    if len(cuda_devices) < 2:
        raise SystemExit(
            f"device_map was set but parameters sit on {sorted(param_devices)} "
            "- the split did not happen (unsloth ignored device_map, or a GPU "
            "mask leaked). Do not trust this run's memory numbers."
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
    p.add_argument("--dataset", default=None, help="override run dataset path (PROBE runs)")
    p.add_argument("--max-seq-length", type=int, default=None, help="override run seq length (PROBE runs)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)  # strict: refuses unpinned revision
    if args.no_hub:
        # Actually strip the repo, not just skip the preflight: a PROBE run
        # must never push to (or depend on) the lane's checkpoint repo.
        cfg = dataclasses.replace(cfg, hub=HubCfg(checkpoint_repo=None))

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
        getattr(cfg.train, args.mode),
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        dataset=args.dataset,
        max_seq_length=args.max_seq_length,
    )
    check_mp_torchrun_conflict(cfg.model.device_map, int(os.environ.get("WORLD_SIZE", "1")))
    check_max_memory_requires_device_map(cfg.model.device_map, cfg.model.max_memory)
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
    check_mp_gpu_count(cfg.model.device_map, torch.cuda.device_count())

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    load_kw = {}
    if cfg.model.device_map is not None:
        load_kw["device_map"] = cfg.model.device_map
        if cfg.model.max_memory is not None:
            load_kw["max_memory"] = cfg.model.max_memory
    model_source, model_revision = resolve_model_source(
        cfg.model.repo, cfg.model.revision, os.environ.get("TUNED_MODEL_PATH")
    )
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_source,
        revision=model_revision,
        max_seq_length=run.max_seq_length,
        dtype=torch.float16 if not is_bfloat16_supported() else torch.bfloat16,
        load_in_4bit=True,
        full_finetuning=False,
        **load_kw,
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

    if cfg.model.device_map is not None:
        param_devices = sorted({str(p.device) for p in model.parameters()})
        print(f"param_devices={param_devices}")
        check_model_split(param_devices)

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

    from transformers import TrainerCallback

    class _NonFiniteGuard(TrainerCallback):
        """Abort a diverged run after ~3 steps instead of burning the whole
        budget. Must RAISE, not set control.should_training_stop: that flag
        ends the run rc=0 - the exact flag normal completion uses - so the
        notebook supervisor would read a divergence as green. Under torchrun
        the raising rank takes the whole job down nonzero."""

        def __init__(self):
            self._window = _NonFiniteWindow()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "grad_norm" not in logs:
                return
            if self._window.observe(state.global_step, logs["grad_norm"]):
                raise RuntimeError(
                    f"grad_norm non-finite {self._window.window} logs in a row "
                    f"(through step {state.global_step}) - fp16 divergence; "
                    "aborting early to save quota. First lever: max_grad_norm: 0.3"
                )

    trainer.add_callback(_NonFiniteGuard())

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

    if not math.isfinite(stats.training_loss):
        raise SystemExit(
            f"train_loss={stats.training_loss} - fp16 divergence; first lever: set max_grad_norm: 0.3"
        )

    peaks = [
        torch.cuda.max_memory_allocated(i) / 1e9 for i in range(torch.cuda.device_count())
    ]
    print(f"peak_vram_gb={max(peaks):.2f}")
    for i, gb in enumerate(peaks):
        print(f"peak_vram_gb_dev{i}={gb:.2f}")

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
