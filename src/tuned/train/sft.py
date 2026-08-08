"""Unsloth QLoRA SFT entrypoint. Run on a Kaggle GPU (accelerator "GPU T4 x2"),
never locally. Always 2x T4 data-parallel under torchrun - unsloth auto-assigns
one rank per GPU, and each rank holds the full model.

All launches are prefixed CUDA_VISIBLE_DEVICES=0,1 and go through
`torchrun --nproc_per_node=2 -m tuned.train.sft --config configs/law_v1_8b_ddp.yaml --mode smoke`:

Probe:    ... --max-steps 2 --no-hub --dataset data/probe_long.jsonl --max-seq-length 8192
Savetest: ... --max-steps 4 --save-steps 2
Smoke:    ... (no extra args)
Resume:   ... --resume --max-steps 64 --allow-schedule-change

The resume flag is gate-only: changing max_steps rebuilds the LR schedule.
"""

import argparse
import dataclasses
import json
import math
import os
import time
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

    `step` is state.global_step - absolute, never relative to the session - so
    a run resumed at step 61 gets NO fresh grace window. That is correct: the
    restored GradScaler does not recalibrate. Re-keying this on a step-relative
    counter would silently open a 2-step blind spot on every resume.
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


def check_resume_schedule(
    checkpoint_dir: str | Path, max_steps: int, allow_schedule_change: bool = False
) -> None:
    """Refuse a resume that would silently rebuild the LR schedule.

    scheduler.pt restores the step counter and nothing else: warmup_steps and
    the decay denominator are both derived from THIS session's max_steps in
    build_sft_config. A changed max_steps therefore reshapes the whole curve
    mid-run - the RESUME gate's LR jumped +134% at step 62 that way."""
    state = Path(checkpoint_dir) / "trainer_state.json"
    if not state.is_file():
        return
    saved = json.loads(state.read_text(encoding="utf-8")).get("max_steps")
    if saved is None or saved == max_steps or allow_schedule_change:
        return
    raise SystemExit(
        f"checkpoint was trained with max_steps={saved}, this run has "
        f"max_steps={max_steps} - the LR schedule would be REBUILT (warmup and "
        "the decay denominator both derive from the session's max_steps, while "
        "scheduler.pt restores only the step counter) and the learning rate "
        "would jump at the resume step. --allow-schedule-change accepts that; "
        "it is meant for the RESUME gate, never for the main run."
    )


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
        # warmup_ratio is deprecated in transformers 5.5 (it logged lr=0 in a
        # 2026-08-07 probe); the ratio stays the config's semantic knob and
        # is converted to steps here.
        "warmup_steps": max(0, round(cfg.train.warmup_ratio * run.max_steps)),
        "weight_decay": cfg.train.weight_decay,
        # Measured grad_norm 0.08-0.19: the default 1.0 clip never binds.
        "max_grad_norm": cfg.train.max_grad_norm,
        "optim": cfg.train.optim,
        "lr_scheduler_type": cfg.train.lr_scheduler_type,
        "seed": cfg.train.seed,
        # T4 (sm_75) has no bf16; flags are explicit so a bf16 default can
        # never sneak in ("BFloat16 != Half" is the classic Kaggle failure).
        "fp16": not bf16_supported,
        "bf16": bf16_supported,
        "logging_steps": 1,
        # The printed approx_tokens_per_sec assumes every sequence fills
        # max_seq_length - an upper bound. This logs tokens actually consumed.
        "include_num_input_tokens_seen": True,
        # Under DDP the trainer otherwise defaults this to True and burns an
        # extra autograd-graph traversal every step (torch warned about it on
        # the qualified 2026-08-06 SAVETEST). Every LoRA param gets a grad
        # each step, so False is safe.
        "ddp_find_unused_parameters": False,
        "save_strategy": "steps",
        "save_steps": run.save_steps,
        "save_total_limit": 2,
        # Opt-in W&B: keyed on the secret's presence so a notebook without the
        # WANDB_API_KEY secret runs exactly as before. Live metrics matter on
        # Kaggle batch runs, which flush output only per completed cell.
        "report_to": "wandb" if os.environ.get("WANDB_API_KEY") else "none",
    }
    if cfg.hub.checkpoint_repo is not None:
        kw.update(
            push_to_hub=True,
            hub_model_id=cfg.hub.checkpoint_repo,
            hub_strategy="checkpoint",
            hub_private_repo=True,
            # Default False skips a checkpoint push outright when the previous
            # upload is still in flight; on Kaggle the Hub copy is the only
            # artifact that survives the session, so never skip one.
            hub_always_push=True,
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
            "visible - a single-GPU CUDA_VISIBLE_DEVICES mask leaked into the "
            "torchrun launch. Prefix the command with CUDA_VISIBLE_DEVICES=0,1."
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
    p.add_argument("--config", default="configs/law_v1_8b_ddp.yaml")
    p.add_argument("--mode", choices=["smoke"], default="smoke")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-schedule-change", action="store_true",
                   help="permit a resume whose max_steps differs from the checkpoint's (RESUME gate only)")
    p.add_argument("--no-hub", action="store_true")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--save-steps", type=int, default=None)
    p.add_argument("--dataset", default=None, help="override run dataset path (PROBE runs)")
    p.add_argument("--max-seq-length", type=int, default=None, help="override run seq length (PROBE runs)")
    p.add_argument("--time-budget-s", type=float, default=None,
                   help="checkpoint and stop cleanly after this many seconds (default: no budget)")
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
    )
    # unsloth auto-selects <|vision_pad|> as Qwen3's pad; at batch > 1 that
    # pad silently NaNs LoRA-A grads (unsloth#4104 - the step-0 tripwire
    # caught it live, 2026-08-08 21:12 UTC). Pin the pad here, before the
    # trainer and collator capture the tokenizer.
    tokenizer.pad_token = "<|endoftext|>"
    model.config.pad_token_id = tokenizer.pad_token_id
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

    # Step-0 gates - both mandatory before any loss/grad_norm number out of
    # this lane is worth reading, and both only meaningful once the response
    # mask above is applied.
    # unsloth#4104: a <|vision_pad|> pad silently NaNs LoRA-A grads at batch > 1.
    assert tokenizer.pad_token == "<|endoftext|>", (
        f"pad_token is {tokenizer.pad_token!r}, expected '<|endoftext|>' - "
        "unsloth#4104: a <|vision_pad|> pad silently NaNs LoRA-A grads at batch > 1"
    )
    probe_rows = [trainer.train_dataset[i] for i in range(min(8, len(trainer.train_dataset)))]
    probe_labels = trainer.data_collator(probe_rows)["labels"]
    kept = int((probe_labels != -100).sum())
    total = int(probe_labels.numel())
    print(f"label_coverage={kept}/{total} ({100 * kept / total:.1f}%)")
    if kept == 0:
        raise SystemExit(
            "label_coverage=0 - masking or truncation ate every response token; "
            "this run would train on nothing. Check instruction_part/response_part "
            "against the chat template and max_seq_length against the data "
            "(unsloth#2771 / trl#3927)."
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

    class _TimeBudget(TrainerCallback):
        """Spend the wall-clock budget, then checkpoint and stop cleanly.

        Kaggle's 12h ceiling and the notebook watchdog both SIGKILL the child,
        which discards up to save_steps-1 steps every session. The deliberate
        contrast with _NonFiniteGuard: there a clean rc=0 would read a
        divergence as green, so it must raise; here rc=0 IS correct, and the
        signal telling the two apart is the printed line plus a global_step
        below max_steps."""

        def __init__(self, budget_s: float):
            self.budget_s = budget_s
            self._start = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            if time.monotonic() - self._start > self.budget_s:
                print(f"time_budget_reached step={state.global_step} - saving and stopping")
                control.should_save = True
                control.should_training_stop = True
            return control

    trainer.add_callback(_NonFiniteGuard())
    if args.time_budget_s is not None:
        trainer.add_callback(_TimeBudget(args.time_budget_s))

    resume = False
    if args.resume:
        from huggingface_hub import snapshot_download

        # One rank downloads: both share this local_dir, so a second pull is
        # duplicate ~0.5-0.7 GB of bandwidth and two writers on one tree. The
        # barrier holds rank 1 until the checkpoint is fully written.
        if trainer.accelerator.is_main_process:
            snapshot_download(
                cfg.hub.checkpoint_repo,
                allow_patterns=["last-checkpoint/*"],
                local_dir=output_dir,
            )
        trainer.accelerator.wait_for_everyone()
        resume = f"{output_dir}/last-checkpoint"
        if not Path(resume).is_dir():
            raise SystemExit(f"no last-checkpoint found in {cfg.hub.checkpoint_repo}")
        check_resume_schedule(
            resume, run.max_steps, allow_schedule_change=args.allow_schedule_change
        )

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

    # Reserved (the allocator's segment high-water), not allocated, is what
    # OOMs - the ~13.5 GiB abort line must be checked against these numbers.
    reserved = [
        torch.cuda.max_memory_reserved(i) / 1e9 for i in range(torch.cuda.device_count())
    ]
    print(f"peak_vram_reserved_gb={max(reserved):.2f}")
    for i, gb in enumerate(reserved):
        print(f"peak_vram_reserved_gb_dev{i}={gb:.2f}")

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
