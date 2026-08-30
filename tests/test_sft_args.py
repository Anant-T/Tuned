import dataclasses
import hashlib
import json
from pathlib import Path

from tuned.train.config import load_config
from tuned.train.sft import build_sft_config

CONFIG = Path(__file__).parent.parent / "training" / "configs" / "law_v1_8b_ddp.yaml"
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


def test_pad_token_is_pinned_before_the_trainer_captures_it():
    # unsloth auto-selects <|vision_pad|> as Qwen3's pad; at batch > 1 that
    # pad silently NaNs LoRA-A grads (unsloth#4104) - the step-0 tripwire
    # caught exactly this live on 2026-08-08 21:12 UTC. The pin must run
    # BEFORE SFTTrainer(...) so the collator captures the corrected
    # tokenizer; the assert stays downstream as the tripwire.
    src = SFT.read_text(encoding="utf-8")
    pin = src.find('tokenizer.pad_token = "<|endoftext|>"')
    trainer_ctor = src.find("trainer = SFTTrainer(")
    tripwire = src.find('assert tokenizer.pad_token == "<|endoftext|>"')
    assert -1 not in (pin, trainer_ctor, tripwire)
    assert pin < trainer_ctor < tripwire
    assert "model.config.pad_token_id = tokenizer.pad_token_id" in src


def test_rslora_flag_reaches_get_peft_model():
    # rsLoRA is config-driven: the flag must flow from the yaml into
    # FastModel.get_peft_model, not be hardcoded - the production lane keeps
    # the default False; an experiment yaml can flip it without code changes.
    src = SFT.read_text(encoding="utf-8")
    peft_ctor = src.find("FastModel.get_peft_model(")
    flag = src.find("use_rslora=cfg.lora.use_rslora", peft_ctor)
    trainer_ctor = src.find("trainer = SFTTrainer(")
    assert -1 not in (peft_ctor, flag, trainer_ctor)
    assert peft_ctor < flag < trainer_ctor


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


def _run_cfg(dataset: str):
    cfg = load_config(CONFIG, allow_unpinned=True)
    return dataclasses.replace(cfg.train.main, dataset=dataset)


def test_a_local_corpus_wins_and_never_touches_the_network(tmp_path):
    from tuned.train.config import HubCfg
    from tuned.train.sft import resolve_main_dataset

    corpus = tmp_path / "law_v1.jsonl"
    corpus.write_text('{"messages": []}\n', encoding="utf-8")

    def _never(**kw):  # the fetch must not happen when the file is present
        raise AssertionError("downloaded despite a local corpus")

    path, digest = resolve_main_dataset(
        _run_cfg(str(corpus)), HubCfg(checkpoint_repo=None, dataset_repo="u/d"),
        "main", download=_never,
    )
    assert path == str(corpus)
    assert len(digest) == 64


def test_main_fetches_the_pinned_corpus_from_the_dataset_repo(tmp_path):
    from tuned.train.config import HubCfg
    from tuned.train.sft import MAIN_DATASET_FILENAME, resolve_main_dataset

    # data/ is gitignored (only configs/ and scripts/ are excepted), so the
    # assembled corpus can NEVER be in the Kaggle clone - the hub is the only
    # route, and MAIN aborted here even with a finished, pushed corpus.
    fetched = tmp_path / "cached" / MAIN_DATASET_FILENAME
    fetched.parent.mkdir()
    fetched.write_text('{"messages": [1]}\n', encoding="utf-8")
    calls = []

    def _fake(**kw):
        calls.append(kw)
        return str(fetched)

    hub = HubCfg(
        checkpoint_repo=None, dataset_repo="tantan01/tuned-law-v1-data",
        dataset_revision="abc123",
    )
    path, digest = resolve_main_dataset(
        _run_cfg(str(tmp_path / "absent.jsonl")), hub, "main", download=_fake
    )
    assert path == str(fetched)
    assert digest == hashlib.sha256(fetched.read_bytes()).hexdigest()
    assert calls == [{
        "repo_id": "tantan01/tuned-law-v1-data",
        "filename": MAIN_DATASET_FILENAME,
        "revision": "abc123",
        "repo_type": "dataset",
    }]


def test_a_missing_corpus_with_no_dataset_repo_is_a_named_refusal(tmp_path):
    from tuned.train.config import HubCfg
    from tuned.train.sft import resolve_main_dataset

    with pytest.raises(SystemExit, match="hub.dataset_repo is null"):
        resolve_main_dataset(
            _run_cfg(str(tmp_path / "absent.jsonl")),
            HubCfg(checkpoint_repo=None), "main", download=lambda **kw: "x",
        )


def test_smoke_never_reaches_for_the_hub(tmp_path):
    from tuned.train.config import HubCfg
    from tuned.train.sft import resolve_main_dataset

    # The smoke/probe datasets are built locally by tuned.data.smoke; only the
    # main corpus is a hub artifact.
    path, digest = resolve_main_dataset(
        _run_cfg(str(tmp_path / "smoke_v1.jsonl")),
        HubCfg(checkpoint_repo=None), "smoke",
        download=lambda **kw: pytest.fail("smoke must not fetch"),
    )
    assert path.endswith("smoke_v1.jsonl") and digest == ""


def test_main_refuses_a_corpus_whose_digest_moved():
    from tuned.train.sft import check_dataset_pin

    # A main epoch spans ~3 sessions and resume replays a LengthGroupedSampler
    # permutation derived from the FILE. A corpus rebuilt between sessions
    # retrains some rows and skips others with loss and grad_norm both green -
    # check_resume_schedule guards the LR half of this, nothing guarded the data.
    check_dataset_pin("abc", "abc", "main")          # matching pin: silent
    check_dataset_pin("abc", None, "smoke")          # gates are unpinned
    with pytest.raises(SystemExit, match="was rebuilt"):
        check_dataset_pin("abc", "def", "main")
    with pytest.raises(SystemExit, match="hub.dataset_sha256 is null"):
        check_dataset_pin("abc", None, "main")


def test_the_corpus_digest_is_printed_and_the_commit_recorded():
    src = SFT.read_text(encoding="utf-8")
    # Both land in train.log and therefore in the 5-min progress/train.log push,
    # so the shipped adapter carries its own provenance.
    assert 'print(f"dataset_sha256={dataset_digest}")' in src
    assert 'print(f"git_commit=' in src
    # the commit banner must precede the GPU import, like every other preflight
    commit = src.find("print_git_commit()")
    unsloth = src.find("from unsloth import FastModel")
    assert -1 not in (commit, unsloth) and commit < unsloth


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


def test_the_clip_instrument_did_not_become_a_fourth_callback():
    """The one thing here that source alone can settle.

    _NonFiniteGuard is already subscribed to the grad_norm log key and already
    runs on every logged step; a second subscriber would be two readers of one
    stream. The file has three callback classes and that is the budget - the
    rest of this instrument is tested as behaviour in test_nan_guard.py.
    """
    src = SFT.read_text(encoding="utf-8")
    body = src[
        src.index("class _NonFiniteGuard(TrainerCallback):"):
        src.index("class _TimeBudget(TrainerCallback):")
    ]
    assert src.count("(TrainerCallback):") == 3
    assert "clip_report(" in body
    # Off the args object the callback already receives - no new config key,
    # no new CLI flag, no second copy of the limit to drift from the first.
    assert "args.max_grad_norm" in body
    # PRE-clip is what makes the comparison a binding test rather than a
    # tautology against a value the clip has already flattened to the limit.
    assert "PRE-clip" in body


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
    # The budget must be measured from PROCESS start, not callback
    # construction: the callback is built after model load + dataset prep,
    # and the notebook watchdog's 11 h clock starts at spawn - an unanchored
    # budget silently spends the 30-min kill margin on setup time.
    assert "_TimeBudget(args.time_budget_s, start=_proc_t0)" in src
    assert src.find("_proc_t0 = time.monotonic()") < src.find("from unsloth import FastModel")


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


def test_vram_math_is_binary_gib_not_decimal_gb():
    # The peaks were divided by 1e9 (decimal GB) but labelled _gb and read
    # against GiB reference lines (14.56 usable, 13.5 abort) - understating
    # the real headroom by ~0.9 GiB (2026-08-09 audit). One unit everywhere.
    from tuned.train.sft import _gib

    assert _gib(2**30) == 1.0
    assert abs(_gib(int(14.56 * 2**30)) - 14.56) < 1e-6
    src = SFT.read_text(encoding="utf-8")
    assert "/ 1e9" not in src
    assert "/1e9" not in src


def test_reserved_gate_is_enforced_in_code_not_prose():
    # Until now the ~13.5 GiB abort line lived only in notebook markdown - and
    # its two mentions disagreed on WHICH number it applies to. Reserved (the
    # allocator segment high-water) is what OOMs, so the gate compares
    # reserved and raises: a profile that close to the 14.56 GiB cap must
    # never silently qualify for a multi-session run.
    from tuned.train.sft import check_vram_reserved

    check_vram_reserved([12.98, 13.18])  # the qualified profile: fine
    check_vram_reserved([13.49, 13.50])  # at the line: fine
    with pytest.raises(SystemExit, match="reserved"):
        check_vram_reserved([12.98, 13.51])
    src = SFT.read_text(encoding="utf-8")
    reserved_calc = src.find("torch.cuda.max_memory_reserved")
    gate_call = src.rfind("check_vram_reserved(")
    assert -1 not in (reserved_calc, gate_call)
    assert reserved_calc < gate_call


def test_main_mode_refuses_the_underived_max_steps_sentinel():
    # train.main.max_steps ships as 0 on purpose: check_resume_schedule
    # freezes whatever the first session trains with, so the sentinel would
    # build a nonsense LR schedule for the whole run. Refuse before any GPU
    # work.
    #
    # The refusal must send the operator to the BUILD (counts.train.kept in
    # assemble.json), not to a GPU probe session: assemble.py already dropped
    # every row over this max_seq_length with the pinned tokenizer, so the
    # rows it wrote are the rows the trainer loads. post_filter_rows= stays
    # the cross-check and must still be named, because
    # train_on_responses_only can drop a fully-masked row with only a print.
    from tuned.train.sft import check_main_max_steps

    with pytest.raises(SystemExit, match="counts.train.kept") as exc:
        check_main_max_steps("main", 0)
    assert "post_filter_rows" in str(exc.value)
    assert "probe" not in str(exc.value).lower(), "the derivation no longer costs a GPU session"
    check_main_max_steps("main", 1500)  # derived value: fine
    check_main_max_steps("smoke", 60)  # smoke unaffected
    src = SFT.read_text(encoding="utf-8")
    guard = src.find("check_main_max_steps(args.mode")
    gpu_import = src.find("from unsloth import FastModel")
    assert -1 not in (guard, gpu_import)
    assert guard < gpu_import  # refuse in milliseconds, not after model load


def test_length_grouped_sampling_attacks_ddp_straggler_skew():
    # At bs=1 with variable-length data, every step costs max(rank0, rank1) -
    # rank 0 can draw a 900-token example while rank 1 draws 7,800. The
    # length-grouped sampler makes ranks draw similar lengths at the same
    # time, with no attention-mask change (it cannot demote the SDPA backend
    # or contaminate anything - the safe substitute for packing's other
    # benefit). No-op on the uniform-length smoke data.
    # transformers 5.5 RENAMED the knob: the bool group_by_length field is
    # gone, replaced by train_sampling_strategy="group_by_length" - and
    # unsloth's code-generated UnslothSFTConfig rejects unknown kwargs
    # outright ("unexpected keyword argument 'group_by_length'", the
    # 2026-08-09 02:45 UTC SAVETEST failure, 65 s into the session).
    cfg = load_config(CONFIG, allow_unpinned=True)
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o")
    assert kw["train_sampling_strategy"] == "group_by_length"
    assert "group_by_length" not in kw  # the transformers<5 spelling crashes


def test_dataloader_drop_last_prevents_duplicate_examples():
    # accelerate's even_batches=True default DUPLICATES wrap-around samples so
    # every rank sees equal batch counts - silent example duplication in a
    # one-epoch run. max_steps is floored to full batches anyway, so dropping
    # the partial tail costs nothing.
    cfg = load_config(CONFIG, allow_unpinned=True)
    kw = build_sft_config(cfg, cfg.train.smoke, output_dir="o")
    assert kw["dataloader_drop_last"] is True


def test_reserved_ceiling_aborts_early_not_post_mortem():
    # A pre-training reserved check can never fire: adamw_8bit state appears
    # at the first optimizer step and DDP buckets at the first backward. And
    # the post-run gate fires after the quota is already spent. So the live
    # callback checks the first steps (full memory shape by step 1-2 at bs=1,
    # fixed bucket) and periodically after (fragmentation growth), raising
    # like _NonFiniteGuard - rc=0 must never carry an OOM-bound profile.
    src = SFT.read_text(encoding="utf-8")
    body = src[src.index("class _ReservedCeiling(TrainerCallback):") :]
    assert "def on_step_end(" in body[:2500]
    assert "max_memory_reserved" in body[:2500]
    assert "raise RuntimeError" in body[:2500]
    assert "trainer.add_callback(_ReservedCeiling())" in src


def test_pad_positions_never_carry_labels():
    # The pad IS <|endoftext|>; an unmasked pad position would train the
    # model to emit padding - the exact inverse of the EOS gate. One collator
    # call feeds coverage, EOS and pad-leak checks (two calls could pad
    # differently and check different tensors).
    src = SFT.read_text(encoding="utf-8")
    assert "probe_batch = trainer.data_collator(probe_rows)" in src
    assert 'probe_batch["attention_mask"]' in src
    assert "pad positions carry labels" in src


def test_eos_gate_teaches_stopping_and_never_requires_the_pad():
    # A model whose labels never contain <|im_end|> (the Qwen3 turn
    # terminator) never learns to stop - fatal for the blind-judge eval. The
    # gate must key on <|im_end|>, NEVER on <|endoftext|>: that token is the
    # PAD in this lane, and pad positions are -100 by design. Smoke data
    # truncates single-turn OpenThoughts rows at max_seq (cutting their only
    # <|im_end|>) by design, so zero there is a data artifact -> warn; the
    # main dataset is drop-never-truncate, so zero there is fatal.
    from tuned.train.sft import check_eos_in_labels

    check_eos_in_labels(3, "main")  # present: silent
    check_eos_in_labels(0, "smoke")  # truncation artifact: warns, returns
    with pytest.raises(SystemExit, match="im_end"):
        check_eos_in_labels(0, "main")
    src = SFT.read_text(encoding="utf-8")
    coverage = src.find('print(f"label_coverage=')
    eos = src.find('print(f"eos_in_labels=')
    train = src.find("trainer.train(resume_from_checkpoint")
    assert -1 not in (coverage, eos, train)
    assert coverage < eos < train  # step-0 gate, after masking, before step 1
    assert 'convert_tokens_to_ids("<|im_end|>")' in src


def test_dataset_prep_runs_rank0_first_not_twice():
    # The 2026-08-08 logs show every prep stage twice: two interleaved
    # "Unsloth: Tokenizing (num_proc=8)" bars at ~43 s EACH - both torchrun
    # ranks independently tokenizing, 16 fork workers on a 4-vCPU box (the
    # zoo<2026.8.4 fork-OOM class, doubled). local_main_process_first makes
    # rank 0 compute and write the datasets cache; rank 1 waits at the
    # barrier, then re-runs the same code as a cache hit.
    src = SFT.read_text(encoding="utf-8")
    first = src.find("local_main_process_first()")
    second = src.find("local_main_process_first()", first + 1)
    load = src.find('load_dataset("json"')
    ctor = src.find("trainer = SFTTrainer(")
    assert -1 not in (first, second, load, ctor)
    assert first < load < second < ctor  # one block for load+map, one for the
    # trainer ctor (unsloth's internal num_proc=8 tokenization) + masking map


def test_ceiling_check_due_fires_on_every_step_at_every_1():
    from tuned.train.sft import ceiling_check_due

    # every=1 is the default the variable-bucket cap requires: the peak step
    # is whichever step carries the longest row, so sampling can miss it.
    assert all(ceiling_check_due(s, early=3, every=1) for s in range(1, 120))


def test_ceiling_check_due_still_samples_when_asked():
    from tuned.train.sft import ceiling_check_due

    assert ceiling_check_due(1, early=3, every=25)
    assert ceiling_check_due(3, early=3, every=25)
    assert not ceiling_check_due(4, early=3, every=25)
    assert not ceiling_check_due(10, early=3, every=25)
    assert ceiling_check_due(25, early=3, every=25)


def test_reserved_ceiling_defaults_to_every_step():
    import inspect

    from tuned.train import sft

    src = inspect.getsource(sft.main)
    assert "every: int = 1" in src, "_ReservedCeiling must default to every=1"
    assert "every: int = 25" not in src


def test_reserved_ceiling_guard_is_negated():
    # Pins the call-site guard's polarity. `ceiling_check_due` returns True
    # exactly on steps the ceiling SHOULD be read, so on_step_end must skip
    # (return control) when it is FALSE - "if not ceiling_check_due(...)".
    # Flipping that to "if ceiling_check_due(...): return control" silently
    # disables the reserved-VRAM check on every step it would have fired on
    # (ceiling_check_due itself stays correct, so every test exercising it
    # directly would still pass) - this test is what catches that inversion.
    import inspect

    from tuned.train import sft

    src = inspect.getsource(sft.main)
    assert "if not ceiling_check_due(state.global_step, self.early, self.every):" in src
    assert "if ceiling_check_due(state.global_step, self.early, self.every):" not in src


def test_remediation_ladders_end_at_the_6144_rung():
    import inspect

    import pytest

    from tuned.train import sft

    with pytest.raises(SystemExit) as excinfo:
        sft.check_vram_reserved([14.0])
    # 8192 IS the cap now, so naming it as a rung would tell an operator
    # mid-abort to "drop" to the length they are already running.
    assert "standard-quant repo (-1.31 GiB) -> seq 6144" in str(excinfo.value)
    assert "seq 8192" not in str(excinfo.value)

    # The _ReservedCeiling raise is nested inside main(); read its source.
    src = inspect.getsource(sft.main)
    assert "standard-quant repo -> seq 6144" in src
    assert "seq 8192" not in src


def _state(tmp_path, **body):
    ckpt = tmp_path / "last-checkpoint"
    ckpt.mkdir(exist_ok=True)
    (ckpt / "trainer_state.json").write_text(json.dumps(body), encoding="utf-8")
    return ckpt


def test_resume_decision_starts_fresh_when_there_is_no_checkpoint(tmp_path):
    from tuned.train.sft import resume_decision

    # A first main session has nothing to resume; --resume-if-available must
    # not turn that into a refusal.
    assert resume_decision(tmp_path / "last-checkpoint", 1500) is False


def test_resume_decision_resumes_a_matching_unfinished_run(tmp_path):
    from tuned.train.sft import resume_decision

    assert resume_decision(_state(tmp_path, max_steps=1500, global_step=470), 1500) is True


def test_resume_decision_declines_a_finished_run(tmp_path):
    from tuned.train.sft import resume_decision

    # Resuming AT max_steps loads the checkpoint and exits without stepping -
    # the no-op false green the RESUME gate exists to avoid.
    assert resume_decision(_state(tmp_path, max_steps=1500, global_step=1500), 1500) is False


def test_resume_decision_refuses_a_foreign_schedule_rather_than_restarting(tmp_path):
    """The reason the predicate compares max_steps at all.

    PROBE and SMOKE push to the SAME checkpoint repo as main, so "a checkpoint
    exists" alone would try to resume a 60-step smoke run into a 1500-step main
    run. Silently starting fresh instead would be worse still: the first save
    ten steps later overwrites last-checkpoint/ at the fixed path_in_repo.
    Defer to the schedule guard, which explains the LR-rebuild hazard.
    """
    from tuned.train.sft import resume_decision

    with pytest.raises(SystemExit, match="allow-schedule-change"):
        resume_decision(_state(tmp_path, max_steps=60, global_step=60), 1500)


def test_the_production_entry_never_opts_into_a_schedule_change():
    # --allow-schedule-change on a production resume IS the +134% LR jump the
    # guard exists to prevent; it belongs to the RESUME gate alone.
    src = SFT.read_text(encoding="utf-8")
    assert "--resume-if-available" in src
    # the auto path must go through resume_decision, never straight to resume
    auto = src.find("elif resume_decision(")
    assert auto != -1


# --------------------------------------------------------------------------
# The held-out signal (P1.5). The training lane has never loaded an eval file:
# split.py has written one since it existed, decision 5 asks for 64 rows, and
# a run with no eval signal cannot tell overfitting from progress.
# --------------------------------------------------------------------------

def _eval_run(**fields):
    cfg = load_config(CONFIG, allow_unpinned=True)
    return dataclasses.replace(cfg.train.main, **fields)


def _hub(repo="u/d", revision=None):
    from tuned.train.config import HubCfg

    return HubCfg(checkpoint_repo=None, dataset_repo=repo, dataset_revision=revision)


def test_a_run_that_does_not_evaluate_looks_nothing_up(tmp_path):
    from tuned.train.sft import resolve_eval_dataset

    def _never(**kw):
        raise AssertionError("eval_rows: 0 must not reach the hub")

    run = _eval_run(eval_rows=0, eval_dataset="data/law_v1_eval.jsonl")
    assert resolve_eval_dataset(run, _hub(), _never) is None


def test_a_local_eval_file_wins_over_the_hub(tmp_path):
    from tuned.train.sft import resolve_eval_dataset

    local = tmp_path / "law_v1_eval.jsonl"
    local.write_text("{}\n", encoding="utf-8")

    def _never(**kw):
        raise AssertionError("a local eval file must not be re-fetched")

    run = _eval_run(eval_rows=8, eval_dataset=str(local))
    assert resolve_eval_dataset(run, _hub(), _never) == str(local)


def test_the_eval_half_is_fetched_at_the_corpus_revision(tmp_path):
    """The same revision, not merely the same repo. split.py guarantees
    train/eval disjointness PER BUILD, so an eval file from another build can
    contain rows this one trained on - and nothing downstream would notice."""
    from tuned.train.sft import EVAL_DATASET_FILENAME, resolve_eval_dataset

    seen = {}

    def _download(**kw):
        seen.update(kw)
        return str(tmp_path / "fetched.jsonl")

    run = _eval_run(eval_rows=64, eval_dataset="data/law_v1_eval.jsonl")
    path = resolve_eval_dataset(run, _hub(revision="abc123"), _download)
    assert path == str(tmp_path / "fetched.jsonl")
    assert seen["filename"] == EVAL_DATASET_FILENAME
    assert seen["revision"] == "abc123"
    assert seen["repo_type"] == "dataset"


def test_an_unresolvable_eval_file_refuses_rather_than_training_blind(tmp_path):
    """eval_rows is a PROMISE. A run configured to carry a held-out signal and
    silently training without one is indistinguishable, in the log, from one
    that was never asked for it."""
    from tuned.train.sft import resolve_eval_dataset

    run = _eval_run(eval_rows=64, eval_dataset="data/nowhere.jsonl")
    with pytest.raises(SystemExit, match="eval_rows=64"):
        resolve_eval_dataset(run, _hub(repo=None))


def test_the_eval_knobs_appear_only_when_the_run_evaluates(tmp_path):
    from tuned.train.sft import build_sft_config

    cfg = load_config(CONFIG, allow_unpinned=True)
    off = build_sft_config(cfg, cfg.train.main, str(tmp_path))
    assert "eval_strategy" not in off and "per_device_eval_batch_size" not in off

    on = build_sft_config(cfg, cfg.train.main, str(tmp_path), evaluating=True)
    assert on["eval_strategy"] == "steps"
    # Tied to the save cadence, not a second free-standing number.
    assert on["eval_steps"] == cfg.train.main.save_steps * 5
    # bs=1: the eval forward materialises a [1, seq, 151936] logits tensor
    # outside unsloth's chunked-CE path, so this is a direct multiplier on a
    # peak with ~2.3 GiB of headroom.
    assert on["per_device_eval_batch_size"] == 1


def test_the_lane_is_config_driven_and_never_gated_on_the_mode():
    """A `mode == "main"` gate would mean the production memory shape is never
    exercised by the PROBE/SMOKE ladder - on a lane whose peaks are
    12.98/13.18 GiB against a 13.5 GiB abort line enforced by RAISING."""
    src = SFT.read_text(encoding="utf-8")
    body = src[src.index("def resolve_eval_dataset("):src.index("def check_dataset_pin(")]
    assert "mode" not in body
    assert "run.eval_rows" in body


def test_the_shipped_config_carries_the_eval_half_on_main_and_the_probe_switch():
    import yaml

    raw = yaml.safe_load(
        (Path(__file__).parent.parent / "training/configs/law_v1_8b_ddp.yaml")
        .read_text(encoding="utf-8")
    )
    main, smoke = raw["train"]["main"], raw["train"]["smoke"]
    assert main["eval_rows"] == 64          # decision 5
    assert main["eval_dataset"] == "data/law_v1_eval.jsonl"
    # Off in the smoke lane by default: smoke_v1 has no held-out half, and the
    # PROBE qualification is the operator turning this one line on.
    assert smoke["eval_rows"] == 0
    assert smoke["eval_dataset"] == "data/law_v1_eval.jsonl"


def test_the_eval_mask_gate_refuses_two_numbers_that_are_not_one_quantity():
    """If this build's train_on_responses_only leaves eval_dataset unmasked,
    eval_loss is computed over prompt tokens too - lower, smoother, and not
    comparable to the train loss printed beside it."""
    src = SFT.read_text(encoding="utf-8")
    assert "prompt_masked=" in src
    assert "NOT the same" in src
    # A refusal, not a warning: the two numbers appear side by side in the
    # log and nothing downstream re-derives which is which.
    gate = src[src.index("eval_label_coverage="):]
    assert "raise SystemExit(" in gate[:1200]
