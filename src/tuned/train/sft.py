"""Unsloth QLoRA SFT entrypoint. Run on a Lightning Studio GPU, never locally.

Smoke:  python -m tuned.train.sft --config configs/law_v1.yaml --mode smoke
Resume: python -m tuned.train.sft --config configs/law_v1.yaml --mode smoke --resume
"""

import argparse

from tuned.train.config import Config, RunCfg, load_config


def build_sft_config(cfg: Config, run: RunCfg, output_dir: str) -> dict:
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


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/law_v1.yaml")
    p.add_argument("--mode", choices=["smoke"], default="smoke")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config)  # strict: refuses unpinned revision
    run = cfg.train.smoke
    output_dir = f"outputs/{args.mode}"

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel
    from unsloth.chat_templates import train_on_responses_only

    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg.model.repo,
        revision=cfg.model.revision,
        max_seq_length=run.max_seq_length,
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
        }
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=SFTConfig(dataset_text_field="text", **build_sft_config(cfg, run, output_dir)),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    resume = False
    if args.resume:
        from huggingface_hub import snapshot_download

        if cfg.hub.checkpoint_repo is None:
            raise SystemExit("--resume requires hub.checkpoint_repo in the config")
        snapshot_download(
            cfg.hub.checkpoint_repo,
            allow_patterns=["last-checkpoint/*"],
            local_dir=output_dir,
        )
        resume = f"{output_dir}/last-checkpoint"

    stats = trainer.train(resume_from_checkpoint=resume)
    print(f"train_loss={stats.training_loss:.4f}")

    import torch

    print(f"peak_vram_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}")


if __name__ == "__main__":
    main()
