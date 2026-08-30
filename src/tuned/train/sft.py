"""Unsloth QLoRA SFT entrypoint. Run on a Kaggle GPU (accelerator "GPU T4 x2"),
never locally. Always 2x T4 data-parallel under torchrun - unsloth auto-assigns
one rank per GPU, and each rank holds the full model.

All launches are prefixed CUDA_VISIBLE_DEVICES=0,1 and go through
`torchrun --nproc_per_node=2 -m tuned.train.sft --config training/configs/law_v1_8b_ddp.yaml --mode smoke`:

Probe:    ... --max-steps 2 --save-steps 1 --dataset data/probe_long.jsonl --max-seq-length 8192
Smoke:    ... (no extra args)
Resume:   ... --resume --max-steps 64 --allow-schedule-change
Main:     ... --mode main --resume-if-available --time-budget-s 37800
          (every session, unchanged; NEVER --allow-schedule-change - that
          jump is the +134% LR bug)

--resume demands a checkpoint and is the gate's flag. --resume-if-available
resumes only a checkpoint whose max_steps matches and which has steps left,
so one production entry serves every session of a multi-session epoch: the
alternative was an operator flipping a MODE by hand, where forgetting once
restarts at step 0 and the next save overwrites last-checkpoint/.
"""

import argparse
import dataclasses
import json
import math
import os
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

from tuned.train.config import Config, HubCfg, RunCfg, load_config

# The filename push.py uploads to the private HF dataset repo (it is
# assemble.TRAIN_FILENAME). Named here rather than imported: the data lane
# must never be importable from the GPU process.
MAIN_DATASET_FILENAME = "law_v1_train.jsonl"
EVAL_DATASET_FILENAME = "law_v1_eval.jsonl"


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


def clip_binding_rate(norms, limit: float) -> float:
    """Fraction of logged grad_norm values that were AT OR ABOVE max_grad_norm.

    P1.6 instrument: max_grad_norm=0.3 was fitted on smoke-lane OpenThoughts
    gradients (build_sft_config's comment records the 0.08-0.19 band that
    justified it) and has never been measured on the legal/think corpus it
    now clips in main. A clip that binds on nearly every step silently turns
    the configured LR schedule into a normalised-gradient schedule - a
    different optimizer than the one the config claims to run - so this is a
    genuine measurement, not decoration. A pure function, not a method on
    _NonFiniteGuard: it must stay importable with no torch/transformers on
    the path, same reason _NonFiniteWindow above is pure.

    Edge cases, decided and not left implicit:

    - Empty input returns 0.0 rather than raising. "No steps observed" and
      "no steps bound" are different questions, but this function has no
      caller context to tell them apart (the caller - _NonFiniteGuard.on_log
      - is what knows whether n is meaningful, and it only prints once n=50
      logs have accumulated). A free function should not raise on an empty
      window; 0.0 is also the literal correct answer to "what fraction of
      nothing is >= limit".
    - Non-finite (nan/inf) and unparseable entries are EXCLUDED from the
      binding count - a non-finite grad_norm is the divergence _NonFiniteWindow
      and _NonFiniteGuard's RuntimeError already own, not evidence the clip
      bound - but they are NOT dropped from the denominator (len(norms)).
      Shrinking n would make the printed rate look like it was computed over
      more, cleaner data than it actually was; keeping them in the
      denominator instead makes a run with many non-finite logs read as a
      LOWER binding rate, which is honest: this function cannot certify
      "the clip bound" for a step whose norm it could not read.
    """
    norms = list(norms)
    if not norms:
        return 0.0
    binding = 0
    for raw in norms:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue  # unparseable: same non-advancing treatment as observe()
        if not math.isfinite(value):
            continue  # _NonFiniteGuard's failure mode, not this one
        if value >= limit:
            binding += 1
    return binding / len(norms)


# The early window the clip is measured over, and the rate above which the
# reading gets a WARN beside it. 50 steps is long enough to be past the
# warm-up spike and short enough to reach an operator in session one.
CLIP_WINDOW_STEPS = 50
CLIP_BINDING_WARN = 0.30


def clip_report(norms, limit: float) -> list[str]:
    """The lines the guard prints once the early window has filled.

    A pure function returning LINES rather than printing them, for the same
    reason clip_binding_rate is pure: the guard that calls it is nested inside
    main() and cannot be constructed without torch, so anything left inside it
    can only be tested by reading the source back as a string. The threshold
    and the sentence a reader will act on both belong out here where a test
    can exercise them.
    """
    rate = clip_binding_rate(norms, limit)
    lines = [f"clip_binding_rate={rate:.3f} max_grad_norm={limit} n={len(list(norms))}"]
    if rate > CLIP_BINDING_WARN:
        lines.append(
            f"WARN clip_binding_rate={rate:.3f} exceeds {CLIP_BINDING_WARN:.2f} - "
            "max_grad_norm was fitted on smoke-lane OpenThoughts gradients, never "
            "on this corpus; the clip may be running this as a normalised-gradient "
            "schedule instead of the configured LR curve"
        )
    return lines


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


def sha256_file(path: str | Path, _blocks: int = 1 << 20) -> str:
    """Digest of the corpus that trained a checkpoint.

    Deliberately stdlib and deliberately local: `tuned.data.acquire` has the
    same helper, but importing it here would drag the whole data lane (httpx,
    the store, the provider fleet) into the GPU process for one hashlib call.
    """
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_blocks), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_main_dataset(
    run: RunCfg, hub: HubCfg, mode: str, *, download=None
) -> tuple[str, str]:
    """Resolve the training corpus to a local path and digest it.

    Mirrors resolve_model_source: the model is pinned by revision, and after
    this the corpus is pinned by sha256. Returns (path, digest).

    THE HUB IS THE ONLY ROUTE for main. `.gitignore` carries `/data/*` with
    exactly two exceptions (configs/, scripts/), so the assembled corpus can
    never be git-tracked and never reaches the Kaggle clone - while push.py
    uploads it to the private HF DATASET repo as `law_v1_train.jsonl`. The
    config's `data/law_v1.jsonl` was therefore a path nothing produced and
    nothing fetched, and MAIN aborted even with a finished, pushed corpus.

    THE DIGEST IS NOT DECORATION. A main run is one epoch spread over ~3
    sessions; correct resumption needs `skip_first_batches` to replay the same
    LengthGroupedSampler permutation, which is a function of the dataset FILE
    (row order, row count, per-row token lengths). Rebuild the corpus between
    two sessions - which the data pipeline is designed to keep doing - and
    some rows train twice while others never train, with loss and grad_norm
    both perfectly green. check_resume_schedule guards the LR half of exactly
    this hazard; this guards the data half.
    """
    local = Path(run.dataset)
    if mode != "main" or local.is_file():
        return str(local), sha256_file(local) if local.is_file() else ""

    if not hub.dataset_repo:
        raise SystemExit(
            f"train.main.dataset={run.dataset} does not exist and hub.dataset_repo "
            "is null - the assembled corpus lives in a private HF dataset repo "
            "(data/ is gitignored, so it is never in the clone). Set "
            "hub.dataset_repo and pin it with training/scripts/pin_dataset.py."
        )
    path = _hub_file(hub, MAIN_DATASET_FILENAME, download)
    return path, sha256_file(path)


def _hub_file(hub: HubCfg, filename: str, download=None) -> str:
    """One file out of the pinned dataset repo, at the pinned revision.

    One function for both halves of the corpus: the train side and the eval
    side must come from the SAME revision or the eval set is not the held-out
    half of what is being trained - split.py's disjointness guarantee is per
    build, and two builds' halves can overlap freely.
    """
    if download is None:  # pragma: no cover - exercised by the fake in tests
        from huggingface_hub import hf_hub_download as download

    return str(download(
        repo_id=hub.dataset_repo,
        filename=filename,
        revision=hub.dataset_revision,
        repo_type="dataset",
    ))


def stratified_head(sources: Sequence[str], n: int) -> list[int]:
    """Deterministically pick n row indexes spread across every source.

    Replaces a plain head, which was silently reading one source. split.py
    writes eval_indexes in INPUT-FILE order and the corpus is strictly
    source-blocked, so the first 64 rows of a real eval file were 64/64
    PredEx out of eight sources - and `synthesis`, the stream this whole
    pipeline exists to produce, is the LAST block and could never appear in a
    head of any size short of the entire file. P1.4 held each source out of
    its own pool; taking a head at the trainer threw that away one commit
    later, and the eval loss printed beside the train loss was a
    PredEx-only number.

    Determinism is preserved exactly as the head had it, and it matters for
    the same reason: a multi-session run must report loss on the SAME rows
    every session or the two numbers are not comparable. So there is no RNG
    and no seed to keep in step with the trainer's - this is a round-robin
    over sources sorted by name, taking rows in file order within each, which
    is a pure function of the file.

    Round-robin rather than proportional on purpose. At 64 rows a
    proportional draw would give zero synthesis rows (0.3% of the corpus) and
    reproduce the bug in a subtler form. This is a diagnostic meant to catch a
    regression in ANY source, not an estimate of the training distribution.
    """
    if n <= 0:
        return []
    groups: dict[str, list[int]] = {}
    for i, source in enumerate(sources):
        groups.setdefault(source, []).append(i)
    picked: list[int] = []
    ordered = [groups[k] for k in sorted(groups)]
    for depth in range(max((len(g) for g in ordered), default=0)):
        for group in ordered:
            if depth < len(group):
                picked.append(group[depth])
                if len(picked) == n:
                    return sorted(picked)
    return sorted(picked)


def resolve_eval_dataset(run: RunCfg, hub: HubCfg, download=None) -> str | None:
    """The held-out file for this run, or None when it does not evaluate.

    `eval_rows: 0` means no evaluation and returns None before anything is
    looked up. A POSITIVE eval_rows that cannot be resolved is a refusal, not
    a fallback: a run configured to carry a held-out signal and silently
    training without one is the failure this whole step exists to remove, and
    in the log it would be indistinguishable from a run never asked for one.

    No sha256 pin, unlike the training corpus. The eval file does not feed the
    LengthGroupedSampler permutation, so the resume hazard resolve_main_dataset
    guards cannot reach it - it is loaded fresh each session and never resumed
    into. The revision pin is what ties it to the right build.
    """
    if not run.eval_rows:
        return None
    local = Path(run.eval_dataset) if run.eval_dataset else None
    if local is not None and local.is_file():
        return str(local)
    if not hub.dataset_repo:
        raise SystemExit(
            f"eval_rows={run.eval_rows} asks for a held-out signal, but "
            f"eval_dataset={run.eval_dataset!r} does not exist and hub.dataset_repo "
            "is null. Set hub.dataset_repo (the eval half rides in the same repo "
            "as the training corpus) or set eval_rows: 0 to say plainly that this "
            "run does not evaluate."
        )
    return _hub_file(hub, EVAL_DATASET_FILENAME, download)


def check_dataset_pin(digest: str, pinned: str | None, mode: str) -> None:
    """Refuse a corpus whose bytes moved under a pinned run (see above)."""
    if mode != "main":
        return
    if not pinned:
        raise SystemExit(
            "hub.dataset_sha256 is null - a main run must pin the corpus it "
            "trains on, because resume replays a sampler permutation derived "
            f"from the file itself. This run's corpus digests to {digest}; "
            "record it with training/scripts/pin_dataset.py and commit."
        )
    if digest != pinned:
        raise SystemExit(
            f"dataset digest {digest} does not match the pinned "
            f"hub.dataset_sha256 {pinned} - the corpus was rebuilt. Resuming "
            "across a rebuilt corpus retrains some rows and skips others "
            "silently. Restore the pinned corpus, or re-pin AND restart the "
            "run from step 0 (a fresh checkpoint repo)."
        )


def print_git_commit() -> None:
    """Record which code trained the adapter.

    The lane refuses an unpinned model revision and ==-pins every training
    dep, then clones whatever is on the default branch. Across the 3+ sessions
    of a main epoch that means session 3 can run different code against a
    checkpoint session 1 produced, with nothing in train.log or the checkpoint
    repo recording it. Must never raise: the package can run from a wheel.
    """
    try:
        probe = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True, text=True, timeout=10,
        )
        sha = probe.stdout.strip() if probe.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        sha = ""
    print(f"git_commit={sha or 'unknown'}")


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


def resume_decision(checkpoint_dir: str | Path, max_steps: int) -> bool:
    """Whether a checkpoint at this path should be resumed from.

    Exists so a multi-session main run needs no per-session notebook edit. The
    hazard that makes it worth code rather than an operator habit: MODE and
    MAIN_RESUME were two entries carrying the same information the checkpoint
    repo already holds, and forgetting to flip it started training at step 0 -
    whose first save, ten steps later, OVERWRITES last-checkpoint/ at the
    fixed path_in_repo. A whole session of a multi-session epoch is discarded
    silently, recoverable only by digging an older revision out of the Hub.

    Comparing max_steps is what makes it safe to point at the shared
    checkpoint repo: PROBE and SMOKE push there too, so "a checkpoint exists"
    alone would try to resume a 60-step smoke run into a main run.

    A checkpoint on a DIFFERENT schedule splits two ways, and conflating them
    is what made MAIN session 1 unstartable. The qualification ladder is
    mandatory and ends with RESUME, which pushes a COMPLETE 64-step
    checkpoint to this same repo; MAIN then wants 437, so the max_steps
    compare fails and the old code refused the run outright - with no escape,
    because --allow-schedule-change is only threaded through the explicit
    --resume branch. The documented gate sequence bricked the run it gates,
    and the only way out was deleting last-checkpoint/ from the Hub by hand.

    But a SPENT checkpoint has nothing to lose. `global_step >= its own
    max_steps` means that run finished; starting fresh past it discards no
    progress and rebuilds no schedule mid-run, because there is no run to be
    mid. An UNFINISHED checkpoint on a different schedule is the real hazard
    this function exists for - a partial main session whose max_steps moved
    under it - and still refuses.
    """
    state = Path(checkpoint_dir) / "trainer_state.json"
    if not state.is_file():
        return False
    saved = json.loads(state.read_text(encoding="utf-8"))
    saved_max = saved.get("max_steps")
    step = int(saved.get("global_step") or 0)
    if saved_max != max_steps:
        if saved_max is not None and step >= int(saved_max):
            print(
                f"checkpoint at {checkpoint_dir} is a COMPLETED "
                f"{saved_max}-step run (global_step={step}); this run wants "
                f"max_steps={max_steps}. Nothing to resume - starting fresh. "
                "This is the normal hand-off from the qualification ladder."
            )
            return False
        # Unfinished and on a different schedule: defer to the guard, which
        # explains the LR-rebuild hazard and offers the escape hatch. Never
        # silently start fresh here - that is the step-0 overwrite this
        # function exists to prevent.
        check_resume_schedule(checkpoint_dir, max_steps)
        return False
    return step < max_steps


def check_main_max_steps(mode: str, max_steps: int) -> None:
    """train.main.max_steps ships as 0 - a deliberate sentinel. The value is
    derived from the corpus the builder emitted, and check_resume_schedule
    freezes whatever the first session trains with, so the sentinel would
    build a nonsense LR schedule for the whole run.

    THE NUMBER COMES OFF THE BUILD, not off a GPU session. assemble.py
    measures every row against `train.main.max_seq_length` - resolved through
    config.py, the trainer's own bucket, under the pinned tokenizer sft.py
    feeds the trainer - and drops what does not fit rather than truncating it.
    So the rows it wrote ARE the rows the trainer loads, and their count is
    `chain.counts.train.kept` in **build_manifest.json**, which push.py ships
    beside the corpus. Deriving it needed a whole Kaggle session only because
    nothing said that.

    The file name matters and was wrong here until 2026-08-31: push.py
    uploads train, eval, README, build_manifest.json and stats.json - and NOT
    assemble.json, which stays in the builder's out/ dir. build_manifest.json
    embeds assemble.json wholesale under its `chain` key, so the number is
    there, one level deeper than the old text said.

    post_filter_rows= is still printed every run, and it is still the number
    that decides: train_on_responses_only drops a fully-masked row with only a
    print, so the manifest count is an upper bound. In practice they agree -
    assemble.py's built_row guarantees the shape, and the step-0
    label_coverage gate refuses a batch that is entirely mask - but the print
    is the cross-check, not decoration, and a disagreement means read the
    print and not this docstring.
    """
    if mode == "main" and max_steps <= 0:
        raise SystemExit(
            f"train.main.max_steps={max_steps} is the underived sentinel - derive "
            "it from the build: read chain.counts.train.kept from the "
            "build_manifest.json beside the corpus (the builder already "
            "dropped every row over max_seq_length, measured with the pinned "
            "tokenizer), set "
            "max_steps = kept // (bs * ga * world_size) in the config and commit. "
            "Cross-check it against the post_filter_rows= line of the first "
            "session's log. It is immutable after that session "
            "(check_resume_schedule refuses a changed schedule on resume)."
        )


def _gib(nbytes: int) -> float:
    """Binary GiB - the unit of every reference line (14.56 usable, 13.5 abort).
    The old math divided by decimal 1e9 under a _gb label, understating true
    headroom by ~0.9 GiB against those GiB references."""
    return nbytes / 2**30


def check_vram_reserved(reserved_gib: list[float], limit_gib: float = 13.5) -> None:
    """The ~13.5 GiB abort line, enforced in code instead of notebook prose
    (whose two mentions disagreed on which number it applies to). Reserved -
    the allocator's segment high-water - is what actually OOMs against the
    14.56 GiB cap; allocated is always smaller and reads falsely green."""
    worst = max(reserved_gib)
    if worst > limit_gib:
        raise SystemExit(
            f"peak reserved VRAM {worst:.2f} GiB exceeds the {limit_gib} GiB "
            "abort line - too close to the 14.56 GiB cap to trust across a "
            "multi-session run (fragmentation only grows). OOM ladder in "
            "training/configs/law_v1_8b_ddp.yaml: standard-quant repo (-1.31 GiB) -> "
            "seq 6144 (UNSLOTH_CE_LOSS_N_CHUNKS is already at "
            "32, not a lever left to spend)."
        )


def ceiling_check_due(step: int, early: int, every: int) -> bool:
    """Whether the reserved-VRAM ceiling should be read at this step.

    torch.cuda.max_memory_reserved() is a MONOTONIC high-water mark and
    nothing in this repo calls reset_peak_memory_stats, so a later sample
    still observes an earlier breach - sampling every 25th step was never at
    risk of missing one, only of reporting it late. every=1 exists so the
    abort fires AT the breaching step: the `at step {N}` in the error message
    then names the step that actually caused the breach, not whichever later
    step happened to be sampled. At ga=6, 24 late steps is on the order of an
    hour of Kaggle quota spent training on a profile already known to be
    OOM-bound - the exact figure depends on a step time this lane has not
    measured on the law corpus (see the save_steps comment in
    law_v1_8b_ddp.yaml; the often-quoted 224 s is an upper bound derived from
    smoke_v1's full-bucket rows, not from real data). It costs nothing to
    check every step either way - a stats-counter read, no CUDA sync, against
    a step measured in tens of seconds. `every` still exists and still samples
    (rather than checking every step) when set above 1."""
    return step <= early or step % every == 0


def check_eos_in_labels(eos_kept: int, mode: str) -> None:
    """A model whose labels never contain <|im_end|> (the Qwen3 turn
    terminator) never learns to STOP - it fails the blind-judge eval by
    rambling, while loss and grad_norm stay green. Keyed on <|im_end|> and
    NEVER on <|endoftext|>: that token is this lane's PAD, and pad positions
    must stay -100. Smoke data truncates single-turn OpenThoughts rows at
    max_seq by design (cutting their only <|im_end|>), so zero there is a
    data artifact -> warn; the main dataset is drop-never-truncate, so zero
    there means the builder or masking is broken -> fatal."""
    if eos_kept > 0:
        return
    msg = (
        "eos_in_labels=0 - no <|im_end|> token among the unmasked labels of "
        "the probe batch; nothing teaches the model to end a turn"
    )
    if mode == "main":
        raise SystemExit(msg + " - main data is drop-never-truncate, so this is a builder/masking bug")
    print(f"WARNING: {msg} (expected artifact when every probe row truncates at max_seq)")


def build_sft_config(
    cfg: Config, run: RunCfg, output_dir: str, bf16_supported: bool = False,
    *, evaluating: bool = False,
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
        # bs=1 + variable-length data: every DDP step costs max(rank0, rank1)
        # - rank 0 can draw a 900-token row while rank 1 draws 7,800. The
        # length-grouped sampler has ranks draw similar lengths at the same
        # time. No attention-mask or kernel change (cannot demote the SDPA
        # backend, cannot contaminate); no-op on uniform-length smoke data.
        # transformers 5.5 spelling: the old bool group_by_length field is
        # GONE, and unsloth's code-generated UnslothSFTConfig rejects unknown
        # kwargs outright (the 2026-08-09 02:45 UTC SAVETEST died on it 65 s
        # into the session - same 5.x-rename class as warmup_ratio above).
        "train_sampling_strategy": "group_by_length",
        # accelerate's even_batches=True default DUPLICATES wrap-around
        # samples so every rank sees equal batch counts - silent example
        # duplication in a one-epoch run. max_steps is floored to full
        # batches anyway (see check_main_max_steps), so the tail is free.
        "dataloader_drop_last": True,
        "save_strategy": "steps",
        "save_steps": run.save_steps,
        "save_total_limit": 2,
        # Opt-in W&B: keyed on the secret's presence so a notebook without the
        # WANDB_API_KEY secret runs exactly as before. Live metrics matter on
        # Kaggle batch runs, which flush output only per completed cell.
        "report_to": "wandb" if os.environ.get("WANDB_API_KEY") else "none",
    }
    if evaluating:
        kw.update(
            eval_strategy="steps",
            # EVERY save_steps * 5, tied to the save cadence rather than given
            # its own number: both are session-relative, and one moving
            # without the other is how a 2-4 minute eval ends up running more
            # often than the checkpoint that lets you resume past it.
            eval_steps=run.save_steps * 5,
            # ONE sequence at a time. The eval forward materialises a
            # [1, seq, 151936] logits tensor outside unsloth's chunked-CE
            # path, so this is a direct multiplier on the peak the lane has
            # ~2.3 GiB of headroom against.
            per_device_eval_batch_size=1,
        )
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
    _proc_t0 = time.monotonic()  # --time-budget-s anchor: process start
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="training/configs/law_v1_8b_ddp.yaml")
    p.add_argument("--mode", choices=["smoke", "main"], default="smoke")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-if-available", action="store_true",
                   help="resume when the checkpoint repo already holds a compatible "
                        "checkpoint, else start fresh (the production main entry)")
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
        # Actually strip the repo, not just skip the preflight: --no-hub is
        # for a throwaway run - e.g. the 2-step row-count probe used to
        # derive train.main.max_steps - that must never push to (or depend
        # on) the lane's checkpoint repo. The PROBE gate itself no longer
        # passes this flag; it pushes a checkpoint on purpose.
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
    # After overrides (a --max-steps 2 probe of the main dataset is legal and
    # is exactly how the real value gets derived), before any GPU import.
    check_main_max_steps(args.mode, run.max_steps)
    output_dir = f"outputs/{args.mode}"

    print_versions()
    print_git_commit()

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
        use_rslora=cfg.lora.use_rslora,
        use_gradient_checkpointing="unsloth",
        random_state=cfg.train.seed,
    )
    model.print_trainable_parameters()
    lora_modules = sorted({n.rsplit(".", 2)[0] for n, _ in model.named_parameters() if "lora_" in n})
    print(f"lora_target_modules_sample={lora_modules[:5]}")
    vision_hits = [m for m in lora_modules if "vision" in m.lower()]
    if vision_hits:
        raise SystemExit(f"LoRA attached to vision tower modules {vision_hits[:3]} - unsloth#5677 risk; fix target_modules regex")

    from accelerate import PartialState

    # Rank 0 computes and writes the datasets cache; rank 1 waits at the
    # barrier, then re-runs the same code as a cache hit. Without this BOTH
    # ranks ran every prep stage - the 2026-08-08 log shows two interleaved
    # "Unsloth: Tokenizing (num_proc=8)" bars at ~43 s each: 16 fork workers
    # on a 4-vCPU box, doubling the zoo<2026.8.4 fork-OOM exposure for zero
    # benefit (the sampler shards data across ranks at iteration time anyway).
    _dist = PartialState()
    with _dist.local_main_process_first():
        # Inside the barrier on purpose: rank 0 does the hub fetch, rank 1
        # takes the cache hit - the same serialization the dataset prep below
        # already relies on, so this needs no new synchronization.
        dataset_path, dataset_digest = resolve_main_dataset(run, cfg.hub, args.mode)
        # Printed unconditionally, so every session log - and the 5-minute
        # progress/train.log push - records which corpus bytes trained the
        # checkpoint, whether or not a pin is set.
        print(f"dataset_sha256={dataset_digest}")
        check_dataset_pin(dataset_digest, cfg.hub.dataset_sha256, args.mode)

        def _as_text(dataset):
            return dataset.map(
                lambda ex: {
                    "text": tokenizer.apply_chat_template(
                        ex["messages"], tokenize=False, add_generation_prompt=False
                    )
                },
                remove_columns=dataset.column_names,
            )

        ds = _as_text(load_dataset("json", data_files=dataset_path, split="train"))
        # The held-out half, inside the SAME barrier: rank 0 fetches and
        # tokenizes, rank 1 takes the cache hit, exactly as the training
        # corpus above does. Resolving it here rather than beside the trainer
        # keeps every hub read on one side of one barrier.
        eval_path = resolve_eval_dataset(run, cfg.hub)
        eval_ds = None
        if eval_path is not None:
            eval_ds = load_dataset("json", data_files=eval_path, split="train")
            # DETERMINISTIC AND STRATIFIED - see stratified_head. Still no RNG
            # and no seed to keep in step with the trainer's own, so two
            # sessions of one run report loss on the same rows; but spread
            # across sources, because split.py writes in input-file order over
            # a source-blocked corpus and a plain head read exactly one of
            # them.
            keep = stratified_head(
                [(r or {}).get("source", "") for r in eval_ds["_prov"]],
                min(run.eval_rows, len(eval_ds)),
            ) if "_prov" in eval_ds.column_names else list(
                range(min(run.eval_rows, len(eval_ds)))
            )
            eval_ds = _as_text(eval_ds.select(keep))
            print(f"eval_rows={len(eval_ds)} <- {eval_path}")

    import inspect

    sft_kw = build_sft_config(
        cfg, run, output_dir, bf16_supported=is_bfloat16_supported(),
        evaluating=eval_ds is not None,
    )
    if "max_length" not in inspect.signature(SFTConfig.__init__).parameters:
        sft_kw["max_seq_length"] = sft_kw.pop("max_length")

    # Same rank-0-first treatment: the SFTTrainer ctor runs unsloth's
    # num_proc=8 tokenization map and train_on_responses_only runs the
    # masking map - the two other stages the log showed twice. Neither does
    # a collective (the first NCCL op happens once training starts), so
    # serializing them under the barrier is safe.
    with _dist.local_main_process_first():
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=ds,
            eval_dataset=eval_ds,
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
    # The max_steps derivation input (see check_main_max_steps): rows LEFT
    # after train_on_responses_only dropped fully-masked ones with a print.
    print(f"post_filter_rows={len(trainer.train_dataset)}")

    # Step-0 gates - both mandatory before any loss/grad_norm number out of
    # this lane is worth reading, and both only meaningful once the response
    # mask above is applied.
    # unsloth#4104: a <|vision_pad|> pad silently NaNs LoRA-A grads at batch > 1.
    assert tokenizer.pad_token == "<|endoftext|>", (
        f"pad_token is {tokenizer.pad_token!r}, expected '<|endoftext|>' - "
        "unsloth#4104: a <|vision_pad|> pad silently NaNs LoRA-A grads at batch > 1"
    )
    probe_rows = [trainer.train_dataset[i] for i in range(min(8, len(trainer.train_dataset)))]
    # ONE collator call feeds all three gates below - two calls could pad
    # differently and silently check different tensors.
    probe_batch = trainer.data_collator(probe_rows)
    probe_labels = probe_batch["labels"]
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
    # Third step-0 gate: the turn terminator must be LEARNED, not just seen.
    # <|im_end|> closes every assistant turn in the Qwen3 template; if none
    # survives unmasked, the model never learns to stop generating.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_kept = int((probe_labels == im_end_id).sum())
    print(f"eos_in_labels={eos_kept}")
    check_eos_in_labels(eos_kept, args.mode)
    # Fourth gate, the EOS gate's inverse: the pad IS <|endoftext|>, so an
    # unmasked pad position would train the model to EMIT padding.
    pad_leak = int((probe_labels[probe_batch["attention_mask"] == 0] != -100).sum())
    if pad_leak:
        raise SystemExit(
            f"{pad_leak} pad positions carry labels - the collator is not "
            "masking padding; every one of them trains the model to emit "
            "<|endoftext|> mid-sequence"
        )

    # Fifth gate, and it only exists because eval does: TWO NUMBERS CALLED
    # "loss" MUST BE THE SAME QUANTITY. train_on_responses_only rewrites the
    # labels of the dataset(s) the trainer holds; if this build's unsloth
    # rewrites only train_dataset, the eval forward scores prompt tokens as
    # well as response tokens and the eval loss printed beside the train loss
    # is measuring something else entirely - lower, smooth, and completely
    # uncomparable. The tell is an attended position that carries a label
    # where the prompt should have been masked out.
    if trainer.eval_dataset is not None:
        eval_probe = [
            trainer.eval_dataset[i] for i in range(min(4, len(trainer.eval_dataset)))
        ]
        eval_batch = trainer.data_collator(eval_probe)
        eval_labels, attended = eval_batch["labels"], eval_batch["attention_mask"] == 1
        eval_masked = int(((eval_labels == -100) & attended).sum())
        eval_kept = int(((eval_labels != -100) & attended).sum())
        print(f"eval_label_coverage={eval_kept}/{int(attended.sum())} "
              f"prompt_masked={eval_masked}")
        if eval_masked == 0 or eval_kept == 0:
            raise SystemExit(
                f"the eval set's response mask did not apply (prompt_masked="
                f"{eval_masked}, response_labels={eval_kept}) - this build's "
                "train_on_responses_only left eval_dataset unmasked, so eval_loss "
                "would be computed over prompt tokens too and is NOT the same "
                "quantity as the train loss beside it. Do not read the two "
                f"together: set train.{args.mode}.eval_rows: 0 to run without an "
                "eval signal, or apply the mask to eval_dataset before trusting it."
            )

    from transformers import TrainerCallback

    class _NonFiniteGuard(TrainerCallback):
        """Abort a diverged run after ~3 steps instead of burning the whole
        budget. Must RAISE, not set control.should_training_stop: that flag
        ends the run rc=0 - the exact flag normal completion uses - so the
        notebook supervisor would read a divergence as green. Under torchrun
        the raising rank takes the whole job down nonzero.

        Also carries the P1.6 clip-binding instrument. max_grad_norm=0.3 (see
        build_sft_config's "0.08-0.19" comment for where that number came
        from) has only ever been measured against smoke-lane OpenThoughts
        grad_norms, never the legal/think corpus this guard actually watches
        in main. It piggybacks HERE rather than in a fourth callback class
        because this is already the callback subscribed to the grad_norm log
        key and already running every logged step - a second subscriber
        would just be two readers of the same stream for no reason."""

        def __init__(self):
            self._window = _NonFiniteWindow()
            # Collected only through CLIP_WINDOW_STEPS - a fixed early
            # window, not the whole run, so the print below fires once and the
            # list never grows unbounded across a multi-thousand-step main run.
            self._clip_norms = []
            self._clip_reported = False

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "grad_norm" not in logs:
                return
            if self._window.observe(state.global_step, logs["grad_norm"]):
                raise RuntimeError(
                    f"grad_norm non-finite {self._window.window} logs in a row "
                    f"(through step {state.global_step}) - fp16 divergence; "
                    "aborting early to save quota. First lever: max_grad_norm: 0.3"
                )
            # logs["grad_norm"] is the PRE-clip norm - the Trainer computes
            # and logs it before torch.nn.utils.clip_grad_norm_ rescales the
            # gradients - so comparing it against args.max_grad_norm below is
            # a genuine test of whether the clip is binding, not a tautology
            # against a value the clip has already flattened to the limit.
            if state.global_step <= CLIP_WINDOW_STEPS:
                self._clip_norms.append(logs["grad_norm"])
            # >=, not ==. logging_steps is 1 today so step 50 is certain to be
            # logged, but that is a config value - at logging_steps: 3 the logs
            # run 48, 51 and an equality test would print nothing at all, which
            # is the silent-instrument failure this file refuses everywhere
            # else. _clip_reported already makes it fire once.
            if state.global_step >= CLIP_WINDOW_STEPS and not self._clip_reported:
                self._clip_reported = True
                for line in clip_report(self._clip_norms, args.max_grad_norm):
                    print(line)

    class _TimeBudget(TrainerCallback):
        """Spend the wall-clock budget, then checkpoint and stop cleanly.

        Kaggle's 12h ceiling and the notebook watchdog both SIGKILL the child,
        which discards up to save_steps-1 steps every session. The deliberate
        contrast with _NonFiniteGuard: there a clean rc=0 would read a
        divergence as green, so it must raise; here rc=0 IS correct, and the
        signal telling the two apart is the printed line plus a global_step
        below max_steps."""

        def __init__(self, budget_s: float, start: float | None = None):
            # Anchored to PROCESS start, not construction: this callback is
            # built after model load + dataset prep, while the notebook
            # watchdog's 11 h SIGKILL clock starts at spawn. Unanchored, the
            # setup minutes would silently spend the 30-min kill margin.
            self.budget_s = budget_s
            self._start = time.monotonic() if start is None else start

        def on_step_end(self, args, state, control, **kwargs):
            if time.monotonic() - self._start > self.budget_s:
                print(f"time_budget_reached step={state.global_step} - saving and stopping")
                control.should_save = True
                control.should_training_stop = True
            return control

    class _ReservedCeiling(TrainerCallback):
        """The 13.5 GiB abort line, live. A pre-training check can never fire
        (adamw_8bit state appears at the first optimizer step, DDP buckets at
        the first backward), and the post-run check_vram_reserved fires after
        the quota is spent - so check EVERY step. The old every-25th
        sampling assumed a fixed bucket, and there never was one: at bs=1 the
        collator pads to the longest row IN THE BATCH (= the row itself), so
        every step carries a different length and the peak step is whichever
        one happens to carry the longest row.
        A stats-counter read, no CUDA sync - free against a ~74 s step."""

        def __init__(self, limit_gib: float = 13.5, early: int = 3, every: int = 1):
            self.limit_gib, self.early, self.every = limit_gib, early, every

        def on_step_end(self, args, state, control, **kwargs):
            if not ceiling_check_due(state.global_step, self.early, self.every):
                return control
            worst = max(
                _gib(torch.cuda.max_memory_reserved(i))
                for i in range(torch.cuda.device_count())
            )
            if worst > self.limit_gib:
                raise RuntimeError(
                    f"peak reserved {worst:.2f} GiB > {self.limit_gib} GiB at "
                    f"step {state.global_step} - OOM-bound profile; ladder: "
                    "standard-quant repo -> seq 6144 "
                    "(UNSLOTH_CE_LOSS_N_CHUNKS already at 32)"
                )
            return control

    trainer.add_callback(_NonFiniteGuard())
    trainer.add_callback(_ReservedCeiling())
    if args.time_budget_s is not None:
        trainer.add_callback(_TimeBudget(args.time_budget_s, start=_proc_t0))

    resume = False
    if args.resume or args.resume_if_available:
        from huggingface_hub import snapshot_download

        # One rank downloads: both share this local_dir, so a second pull is
        # duplicate ~0.5-0.7 GB of bandwidth and two writers on one tree. The
        # barrier holds rank 1 until the checkpoint is fully written.
        if trainer.accelerator.is_main_process:
            try:
                snapshot_download(
                    cfg.hub.checkpoint_repo,
                    allow_patterns=["last-checkpoint/*"],
                    local_dir=output_dir,
                )
            except Exception as exc:  # noqa: BLE001
                # A first main session has nothing to download. --resume still
                # demands one; --resume-if-available treats it as "start fresh".
                if args.resume:
                    raise SystemExit(
                        f"could not fetch last-checkpoint from "
                        f"{cfg.hub.checkpoint_repo}: {exc}"
                    ) from exc
                print(f"no checkpoint to resume ({type(exc).__name__}) - starting fresh")
        trainer.accelerator.wait_for_everyone()
        candidate = f"{output_dir}/last-checkpoint"
        if args.resume:
            resume = candidate
            if not Path(resume).is_dir():
                raise SystemExit(f"no last-checkpoint found in {cfg.hub.checkpoint_repo}")
            check_resume_schedule(
                resume, run.max_steps, allow_schedule_change=args.allow_schedule_change
            )
        elif resume_decision(candidate, run.max_steps):
            resume = candidate
            print(f"resuming from {resume}")
        else:
            print("no compatible checkpoint - starting fresh")

    stats = trainer.train(resume_from_checkpoint=resume)
    print(f"train_loss={stats.training_loss:.4f}")

    if not math.isfinite(stats.training_loss):
        raise SystemExit(
            f"train_loss={stats.training_loss} - fp16 divergence; first lever: set max_grad_norm: 0.3"
        )

    # ONE eval pass, here, and BEFORE the peak readings below - which is the
    # whole point: an eval forward materialises a [1, seq, 151936] logits
    # tensor outside unsloth's chunked-CE path, and the lane has ~2.3 GiB of
    # headroom against a 13.5 GiB ceiling. Running it here is what puts that
    # allocation inside the max_memory_reserved() high-water the qualification
    # reads.
    #
    # Without it the ladder could never measure the thing it exists to
    # measure. eval_steps is save_steps * 5, and every qualification mode's
    # step budget is smaller than its own eval interval - PROBE 2 steps vs 5,
    # SMOKE 60 vs 125, RESUME 64 vs 125 - so the periodic eval never fired
    # and `eval_rows: 8` produced a peak byte-identical to a no-eval run. MAIN
    # (save_steps 10 -> eval_steps 50) then met the allocation for the first
    # time at step 50, against a ceiling _ReservedCeiling enforces by raising.
    # The step-0 gates only run the collator, not a forward, so they cannot
    # cover this either.
    if trainer.eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss")
        if eval_loss is not None:
            print(f"eval_loss={eval_loss:.4f}")
        print(f"eval_rows={len(trainer.eval_dataset)}")

    # Binary GiB throughout (_gib): decimal-GB math under a _gb label
    # understated headroom ~0.9 GiB vs the GiB reference lines.
    peaks = [
        _gib(torch.cuda.max_memory_allocated(i)) for i in range(torch.cuda.device_count())
    ]
    print(f"peak_vram_gb={max(peaks):.2f}")
    for i, gb in enumerate(peaks):
        print(f"peak_vram_gb_dev{i}={gb:.2f}")

    # Reserved (the allocator's segment high-water), not allocated, is what
    # OOMs - the ~13.5 GiB abort line is enforced against these numbers.
    reserved = [
        _gib(torch.cuda.max_memory_reserved(i)) for i in range(torch.cuda.device_count())
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

    # Last so a breach still reports its throughput above; the live
    # _ReservedCeiling already aborted early if the profile was OOM-bound.
    check_vram_reserved(reserved)


if __name__ == "__main__":
    main()
