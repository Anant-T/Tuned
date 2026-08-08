from pathlib import Path

from tuned.train.config import load_config
from tuned.train.sft import build_sft_config

CONFIG = Path(__file__).parent.parent / "configs" / "law_v1_8b_ddp.yaml"
# main() is unimportable without the GPU stack, so everything it does inline
# (prints, step-0 gates, the resume path) is asserted against its source.
SFT = Path(__file__).parent.parent / "src" / "tuned" / "train" / "sft.py"


def test_smoke_sft_kwargs():
    cfg = load_config(CONFIG, allow_unpinned=True)
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="outputs/smoke")
    assert kw["max_steps"] == 60
    assert kw["per_device_train_batch_size"] == 1
    assert kw["gradient_accumulation_steps"] == 2
    assert kw["learning_rate"] == 2.0e-4
    assert kw["optim"] == "adamw_8bit"
    assert kw["seed"] == 3407
    assert kw["save_steps"] == 25
    assert kw["save_strategy"] == "steps"
    assert kw["output_dir"] == "outputs/smoke"
    assert kw["max_length"] == 8192


def test_hub_kwargs_only_when_repo_set():
    cfg = load_config(CONFIG, allow_unpinned=True)
    cfg.hub.checkpoint_repo = None
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o")
    assert "hub_model_id" not in kw
    cfg.hub.checkpoint_repo = "user/ckpt"
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o")
    assert kw["hub_model_id"] == "user/ckpt"
    assert kw["hub_strategy"] == "checkpoint"
    assert kw["push_to_hub"] is True
    assert kw["hub_private_repo"] is True


def test_token_counting_and_unconditional_checkpoint_push():
    # approx_tokens_per_sec is synthetic (it assumes every sequence fills
    # max_seq_length), so the trainer's own counter is the only real number.
    # hub_always_push defaults False, which SILENTLY SKIPS a checkpoint push
    # while the previous upload is in flight - and the Hub copy is the only
    # artifact that outlives a Kaggle session.
    cfg = load_config(CONFIG, allow_unpinned=True)
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o")
    assert kw["include_num_input_tokens_seen"] is True
    assert kw["hub_always_push"] is True
    cfg.hub.checkpoint_repo = None
    assert "hub_always_push" not in build_sft_config(cfg, cfg.train.smoke, output_dir="o")


from tuned.train.sft import apply_overrides, check_ddp_visibility, check_gpu_capability

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


def test_report_to_gated_on_wandb_key(monkeypatch):
    cfg = load_config(CONFIG, allow_unpinned=True)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o")
    assert kw["report_to"] == "none"
    monkeypatch.setenv("WANDB_API_KEY", "k")
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o")
    assert kw["report_to"] == "wandb"


def test_find_unused_parameters_disabled():
    cfg = load_config(CONFIG, allow_unpinned=True)
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o")
    # TRL defaults this to True under DDP - an extra autograd-graph traversal
    # every step for nothing (every LoRA param gets a grad each step).
    assert kw["ddp_find_unused_parameters"] is False


def test_warmup_converted_to_steps():
    # warmup_ratio is deprecated in transformers 5.5 (lr logged 0 in the
    # 2026-08-07 probe); build_sft_config converts it.
    cfg = load_config(CONFIG, allow_unpinned=True)
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o")
    assert "warmup_ratio" not in kw
    assert kw["warmup_steps"] == 2  # round(0.03 * 60)


def test_max_grad_norm_is_below_the_measured_grad_norm_band():
    # Measured grad_norms run 0.08-0.19, so the transformers default 1.0 clip
    # never binds - it is not the safety net the divergence guard advertises.
    cfg = load_config(CONFIG, allow_unpinned=True)
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o")
    assert kw["max_grad_norm"] == 0.3


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


def test_dataset_and_seq_overrides():
    # The PROBE gate's two levers: swap in the long probe dataset and probe an
    # above-config sequence length without editing the config.
    run = load_config(CONFIG, allow_unpinned=True).train.smoke
    assert apply_overrides(run).dataset == run.dataset
    assert apply_overrides(run, dataset="data/probe_long.jsonl").dataset == "data/probe_long.jsonl"
    assert apply_overrides(run).max_seq_length == 8192
    assert apply_overrides(run, max_seq_length=10240).max_seq_length == 10240


def test_capability_gate_rejects_p100():
    with pytest.raises(SystemExit, match="T4 x2"):
        check_gpu_capability((6, 0))


def test_capability_gate_accepts_t4():
    check_gpu_capability((7, 5))  # must not raise


def test_visibility_guard_rejects_masked_ranks():
    # The exact 2026-08-06 failure: a CUDA_VISIBLE_DEVICES=0 mask leaked into
    # torchrun, each rank saw one GPU, rank 1 asked for cuda:1 and died with
    # "invalid device ordinal" mid-load.
    with pytest.raises(SystemExit, match="CUDA_VISIBLE_DEVICES"):
        check_ddp_visibility(world_size=2, visible_gpus=1)


def test_visibility_guard_passes_valid_setups():
    check_ddp_visibility(world_size=1, visible_gpus=1)  # not under torchrun
    check_ddp_visibility(world_size=2, visible_gpus=2)  # the production launch


def test_read_gpu_capability_no_crash():
    from tuned.train.sft import read_gpu_capability

    cap = read_gpu_capability()
    assert cap is None or (isinstance(cap, tuple) and len(cap) == 2)


def test_resume_refuses_a_silently_rebuilt_lr_schedule(tmp_path):
    from tuned.train.sft import check_resume_schedule

    # scheduler.pt restores only the step counter; warmup_steps and the decay
    # denominator are both rebuilt from the SESSION's max_steps. That is why
    # the RESUME gate's LR jumped +134% at step 62 - fine for a gate, ruinous
    # for the main run.
    ckpt = tmp_path / "last-checkpoint"
    ckpt.mkdir()
    (ckpt / "trainer_state.json").write_text('{"max_steps": 60}', encoding="utf-8")
    check_resume_schedule(ckpt, 60)  # same schedule: silent
    with pytest.raises(SystemExit, match="allow-schedule-change"):
        check_resume_schedule(ckpt, 64)
    check_resume_schedule(ckpt, 64, allow_schedule_change=True)  # the RESUME gate


def test_checkpoint_download_runs_on_one_rank_only():
    # Both torchrun ranks reach this with the same local_dir: the second
    # download is duplicate ~0.5-0.7 GB of bandwidth and puts two writers on
    # one tree. The barrier is what keeps rank 1 from reading a half-written
    # checkpoint.
    src = SFT.read_text(encoding="utf-8")
    gate = src.rfind("trainer.accelerator.is_main_process")
    download = src.rfind("snapshot_download(")
    barrier = src.rfind("trainer.accelerator.wait_for_everyone()")
    assert -1 not in (gate, download, barrier)
    assert gate < download < barrier


def test_resume_path_runs_the_schedule_guard_before_training():
    # The guard needs the downloaded trainer_state.json, and it is worthless
    # once a single step has run under the rebuilt schedule.
    src = SFT.read_text(encoding="utf-8")
    assert '"--allow-schedule-change"' in src
    assert "allow_schedule_change=args.allow_schedule_change" in src
    # the header's launch recipes are the operator's copy source: the RESUME
    # one is now rejected without the flag
    assert "Resume:   ... --resume --max-steps 64 --allow-schedule-change" in src
    download = src.rfind("snapshot_download(")
    guard = src.rfind("check_resume_schedule(")
    train = src.find("trainer.train(")
    assert -1 not in (download, guard, train)
    assert download < guard < train


def test_time_budget_saves_and_stops_instead_of_raising():
    # Kaggle's 12h ceiling and the notebook watchdog both SIGKILL the child,
    # discarding up to save_steps-1 steps every session. Opposite call to
    # _NonFiniteGuard's: there a clean stop would read a divergence as green,
    # here rc=0 IS the correct outcome, so the flags are the right mechanism.
    src = SFT.read_text(encoding="utf-8")
    assert '"--time-budget-s"' in src
    assert "type=float" in src
    body = src[src.index("class _TimeBudget(TrainerCallback):") :]
    assert "_NonFiniteGuard" in body[:1200]  # the contrast is documented
    assert "def on_step_end(" in body
    assert "time.monotonic()" in body
    assert "time_budget_reached step=" in body
    assert "control.should_save = True" in body
    assert "control.should_training_stop = True" in body


def test_step_zero_gates_sit_between_masking_and_the_first_step():
    # Both are documented mandatory in the config header. pad_token:
    # unsloth#4104 - a <|vision_pad|> pad silently NaNs LoRA-A grads at
    # batch > 1. Coverage: masking or truncation can leave an all -100 label
    # row, which trains on nothing while every logged number stays green.
    # They must run AFTER train_on_responses_only (it applies the mask) and
    # BEFORE the first step, or they gate nothing.
    src = SFT.read_text(encoding="utf-8")
    mask = src.find("train_on_responses_only(")
    pad = src.find('tokenizer.pad_token == "<|endoftext|>"')
    coverage = src.find("label_coverage=")
    train = src.find("trainer.train(")
    assert -1 not in (mask, pad, coverage, train)
    assert mask < pad < coverage < train
    assert "unsloth#4104" in src
    assert "trainer.data_collator" in src
    assert "label_coverage=0" in src  # zero coverage aborts, never warns


def test_reserved_peak_is_reported_beside_allocated():
    # max_memory_allocated is not the OOM number: the allocator's segment
    # high-water (reserved) is what meets the 14.56 GiB cap, so the ~13.5 GiB
    # abort line has to be read off reserved, not allocated.
    src = SFT.read_text(encoding="utf-8")
    assert "torch.cuda.max_memory_reserved" in src
    assert "peak_vram_reserved_gb=" in src
    assert "peak_vram_reserved_gb_dev{i}=" in src
