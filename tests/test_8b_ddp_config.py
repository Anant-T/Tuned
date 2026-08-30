"""The production lane (training/configs/law_v1_8b_ddp.yaml): Qwen3-8B under plain
2x T4 data-parallel via torchrun. Qualified at seq 8192 on 2026-08-08 - all
four gates green (PROBE 12.80/13.00 GiB, SAVETEST, SMOKE 60/60 at 74.7 s/step
with peaks 12.98/13.18 GiB, RESUME). A 12288 raise was tried on 2026-08-26 and
reverted the same day - it OOM'd rank 1 at step 0, ~0.8 GiB over the abort
line - so 8192 is asserted below. Every value asserted here is a live contract
with the Kaggle notebook, notebooks/stage_model.ipynb, or the Hub checkpoint
repo - none of it may drift without re-running the ladder."""

from pathlib import Path

import pytest

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
    # VRAM or kernel change. save_steps counts OPTIMIZER steps, which at ga=6
    # hold 3x the micro-batches a ga=2 step did, so it must shrink, not grow.
    run = load_config(CONFIGS / "law_v1_8b_ddp.yaml").train.main
    assert run.max_seq_length == 8192
    assert run.per_device_train_batch_size == 1
    assert run.gradient_accumulation_steps == 6

    # A BAND, because the number it used to pin is not measured yet. The
    # cadence argument for 10 rests on ~224 s/optimizer step, which is 3 x the
    # 74.7 s timed on smoke_v1 - rows built from a fixed template, every one
    # of them filling the 8192 bucket. The law corpus is drop-never-truncate
    # (p50 2288 / p90 5757 on the current build), attention is quadratic in
    # that length, and at bs=1 the collator pads only to the row itself. So
    # 37 min is an upper bound on the cadence, the real value comes off
    # session 1's log, and freezing 10 here would make setting it a two-file
    # edit for no gain.
    #
    # What the band defends is the SHAPE, which does not depend on the step
    # time: small enough that a session killed at the watchdog loses minutes
    # of steps rather than hours, large enough that the adapter push is not
    # most of what the session does. 25 is what the lane qualified at under
    # smoke; 50 was rejected outright.
    assert 5 <= run.save_steps <= 25
    assert run.dataset == "data/law_v1.jsonl"
    # 0 = deliberately underived: max_steps must be set from the POST-FILTER
    # row count (train_on_responses_only drops fully-masked rows with only a
    # print) via a 2-step --no-hub probe, because check_resume_schedule makes
    # it immutable for the whole multi-session run. sft.py refuses to train
    # main with the sentinel still in place.
    assert run.max_steps == 0


def _pushed_filenames() -> set[str]:
    """The filenames push.py actually uploads, read off the call itself.

    Derived rather than grepped: the upload maps {filename: path}, and the
    keys are module constants, so the AST gives the names and the module
    gives their values. A file added to or removed from that dict changes
    this set without anyone updating a literal here.
    """
    import ast

    from tuned.data import push as push_mod

    tree = ast.parse(Path(push_mod.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "upload":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Dict):
                names = [k.id for k in arg.keys if isinstance(k, ast.Name)]
                if names:
                    return {getattr(push_mod, n) for n in names}
    raise AssertionError("could not find push.py's upload({...}) call")


def test_the_max_steps_derivation_names_a_key_the_builder_actually_writes():
    """The derivation moved off a GPU probe session and onto the build's own
    manifest, so the three places that restate it now name a JSON key. A key
    that drifts leaves the operator reading a field that is not there, on the
    one procedure that cannot be re-run cheaply - check_resume_schedule
    freezes max_steps for the whole run.

    And it must name a file the operator can actually OPEN. The three
    restatements said "assemble.json", which push.py does not upload - it
    stays in the builder's out/ dir. The number is one level deeper, in the
    build_manifest.json that push.py does ship, under its `chain` key. The
    old test could not catch that because it checked assemble.py's own
    filename constant instead of push.py's upload set.
    """
    from tuned.data.assemble import MANIFEST_FILENAME as ASSEMBLE_MANIFEST
    from tuned.data.push import MANIFEST_FILENAME as SHIPPED_MANIFEST

    shipped = _pushed_filenames()
    assert SHIPPED_MANIFEST in shipped, "push.py must ship the manifest it embeds chain into"
    assert ASSEMBLE_MANIFEST not in shipped, (
        "assemble.json is NOT shipped - if that changes, the derivation text "
        "may name it again"
    )

    from tuned.train.sft import check_main_max_steps

    with pytest.raises(SystemExit) as exc:
        check_main_max_steps("main", 0)
    message = str(exc.value)

    config = (CONFIGS / "law_v1_8b_ddp.yaml").read_text(encoding="utf-8")
    notebook = (
        Path(__file__).parent.parent / "training" / "notebooks" / "kaggle_smoke.ipynb"
    ).read_text(encoding="utf-8")

    for where, text in (("refusal", message), ("config", config), ("notebook", notebook)):
        assert "chain.counts.train.kept" in text, (
            f"the {where} must name the full key path inside the shipped manifest"
        )
        assert SHIPPED_MANIFEST in text, f"the {where} must name the file push.py ships"


def test_the_length_bucket_the_builder_filters_on_is_the_one_the_trainer_trains_at():
    """What makes the manifest count usable at all. assemble.py measures every
    row against the TRAINER's max_seq_length and drops rather than truncates,
    so the rows it emitted are the rows the trainer loads. If a builder ever
    carried its own bucket, the derived max_steps would describe a corpus
    nobody trains on."""
    import inspect

    from tuned.data.assemble import main as assemble_main

    assert "max_tokens=cfg.max_seq_length" in inspect.getsource(assemble_main)
