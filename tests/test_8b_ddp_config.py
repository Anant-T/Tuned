"""The production lane (training/configs/law_v1_8b_ddp.yaml): Qwen3-8B under plain
2x T4 data-parallel via torchrun. Qualified at seq 8192 on 2026-08-08 - all
four gates green (PROBE 12.80/13.00 GiB, SAVETEST, SMOKE 60/60 at 74.7 s/step
with peaks 12.98/13.18 GiB, RESUME). A 12288 raise was tried on 2026-08-26 and
reverted the same day - it OOM'd rank 1 at step 0, ~0.8 GiB over the abort
line - so 8192 is asserted below. Every value asserted here is a live contract
with the Kaggle notebook, notebooks/stage_model.ipynb, or the Hub checkpoint
repo - none of it may drift without re-running the ladder."""

from pathlib import Path

from tuned.train.config import load_config

CONFIGS = Path(__file__).parent.parent / "training" / "configs"


def test_model_repo_and_revision_pinned():
    cfg = load_config(CONFIGS / "law_v1_8b_ddp.yaml")
    assert cfg.model.repo == "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    assert cfg.model.revision == "62efd7f9d748e394734a7adae2adf96e13a2abc8"


def test_checkpoint_repo_is_the_live_one():
    # The gates' checkpoints (and --resume) live here. The notebook re-homes
    # the namespace per session but keeps the repo NAME, so this string is the
    # contract: change it and --resume silently starts from scratch.
    cfg = load_config(CONFIGS / "law_v1_8b_ddp.yaml")
    assert cfg.hub.checkpoint_repo == "tantan01/tuned-law-v1-qwen8b-ckpt-ddp"


def test_tokens_per_optimizer_step():
    run = load_config(CONFIGS / "law_v1_8b_ddp.yaml").train.smoke
    # An UPPER BOUND, not the real batch: 8192 x 1 x 2 x 2 ranks = 32,768.
    # At bs=1 the collator pads to the longest row IN THE BATCH, which is the
    # row itself, so real tokens/step are set by row length (~2.5k p50,
    # 7.6k p100 across the built corpus) and stay ~30k regardless of the cap.
    assert run.max_seq_length * run.per_device_train_batch_size * run.gradient_accumulation_steps * 2 == 32768


def test_smoke_run_shape():
    run = load_config(CONFIGS / "law_v1_8b_ddp.yaml").train.smoke
    assert run.max_seq_length == 8192
    assert run.per_device_train_batch_size == 1
    assert run.gradient_accumulation_steps == 2
    assert run.max_steps == 60
    assert run.save_steps == 25
    assert run.dataset == "data/smoke_v1.jsonl"


def test_main_run_shape():
    # The 2026-08-09 audit's packing verdict: packing=True on this stack would
    # be correct (position_ids -> block-diagonal mask, verified against the
    # pinned trl 0.24.0 / unsloth_zoo 2026.8.3 / transformers 5.5.0 sources)
    # but NET-NEGATIVE - it forfeits SDPA's is_causal fast path and enable_gqa
    # (8->32 KV-head expansion), materializes a 64 MiB mask, and pays 8192^2
    # attention on ~2,500-token segments. ga=6 buys packing's only real
    # benefit (a 3x gradient batch, ~30k real tokens/optimizer step) at zero
    # VRAM or kernel change. save_steps counts OPTIMIZER steps, which get ~3x
    # longer (~224 s) - so it must shrink, not grow: 10 x ~224 s ~= 37 min,
    # the same cadence band the lane qualified at (25 x 74.7 s ~= 31 min).
    # 50 would have meant ~3.1 h between checkpoints.
    run = load_config(CONFIGS / "law_v1_8b_ddp.yaml").train.main
    assert run.max_seq_length == 8192
    assert run.per_device_train_batch_size == 1
    assert run.gradient_accumulation_steps == 6
    assert run.save_steps == 10
    assert run.dataset == "data/law_v1.jsonl"
    # 0 = deliberately underived: max_steps must be set from the POST-FILTER
    # row count (train_on_responses_only drops fully-masked rows with only a
    # print) via a 2-step --no-hub probe, because check_resume_schedule makes
    # it immutable for the whole multi-session run. sft.py refuses to train
    # main with the sentinel still in place.
    assert run.max_steps == 0
