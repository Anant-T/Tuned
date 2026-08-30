# tuned

Local code, trained on Kaggle free-tier GPUs ($0). Single production lane:
Qwen3-8B QLoRA, 2x T4 data-parallel (DDP) via `torchrun`, one LoRA per domain
(Indian law is the first).

    edit locally -> git push -> Kaggle notebook: clone -> train

## Layout

One project, two lanes. `training/` holds everything the Kaggle lane needs;
`data/` holds everything the dataset build needs; `src/tuned/` is the shared
importable package both run on.

| Path | Purpose |
|---|---|
| `src/tuned/` | Importable package: `tuned.train` (the DDP lane) and `tuned.data` (the dataset pipeline). |
| `training/configs/law_v1_8b_ddp.yaml` | The only training config: model pin, LoRA, markers, run settings. |
| `training/notebooks/kaggle_smoke.ipynb` | The one artifact uploaded to Kaggle; clones this repo and runs the CLI. |
| `training/notebooks/stage_model.ipynb` | Optional one-time, CPU-only notebook: stages the pinned model snapshot as a Kaggle Dataset input so training skips the hub download. |
| `training/scripts/` | Revision pinning (`pin_revision.py` for the base model, `pin_dataset.py` for the dataset repo). |
| `data/configs/data_law_v1.yaml` | The dataset-build config: free-fleet routing (deepseek generator; qwen/gemma/gpt-oss judges), gates, mix. |
| `data/scripts/` | Store seeding and calibration-set export. |
| `data/build/` | Gitignored working area: corpus, the live SQLite store, pulled artifacts, archive. |
| `tests/` | The full suite (data + train), runs locally on CPU. |
| `prev_rep.md` | The archive: full lane history, retired configs, closed questions, campaign records. Check it before re-litigating anything. Retired plans are summarised in §3.6. |

## Local setup (Windows, no GPU)

    uv venv
    .venv\Scripts\Activate.ps1
    uv pip install -e ".[dev]"
    python -m pytest tests/ -q

Training deps (`[train]`: unsloth, transformers 5.5.0) install only on Kaggle.
The template-drift test self-skips locally and runs on Kaggle. The data
pipeline (`[build]` extra) additionally needs provider keys in `.env` (never
committed).

## Kaggle setup (once)

1. kaggle.com account -> Settings -> verify phone number (gates GPU + internet).
2. Create an HF account with a **write** token (the notebook derives the private
   checkpoint repo from your token's account automatically — no config edit needed).
3. Kaggle -> Create -> Notebook -> File -> Import Notebook -> upload `training/notebooks/kaggle_smoke.ipynb`.
4. Notebook settings: Accelerator **GPU T4 x2** (never P100 — unsupported), Internet **On**.
5. Add-ons -> Secrets -> add `HF_TOKEN`. Optionally add `WANDB_API_KEY` too — it
   turns on live Weights & Biases metrics for the training cell (unset = no W&B,
   same as before).
6. Optional, saves ~7 GiB of hub traffic per session: run `training/notebooks/stage_model.ipynb`
   once (CPU only, zero GPU quota), then attach its output as an Input to
   `kaggle_smoke` (**+ Add Input -> Your Work**). It stages the pinned snapshot
   with a `REVISION.txt` guard — `kaggle_smoke` uses it only if that revision
   still matches the config pin, otherwise it falls back to the hub download.

## Smoke run

The lane (`training/configs/law_v1_8b_ddp.yaml`) runs at `max_seq_length` 8192,
where all four gates are green. A raise to 12288 was tried on 2026-08-26 and
**failed** — rank 1 OOM'd inside step 0's backward at ~14.3 GiB reserved
against the 13.5 GiB abort line, roughly 0.8 GiB over, already at
`UNSLOTH_CE_LOSS_N_CHUNKS=32`. It was reverted the same day. Do not re-raise
the cap without spending an OOM-ladder rung first. The numbers below are the
2026-08-08 gate run on Kaggle:

- **PROBE** (2-step VRAM-ceiling check, pushes a checkpoint): per-rank peaks 12.80/13.00 GiB at seq 8192.
- **SMOKE** (60 steps): 60/60 complete, 74.7 s/step, ~438 tok/s aggregate, peaks 12.98/13.18 GiB.
- **RESUME**: fresh session, training continued from step 61 with optimizer/scaler state restored.

Set `MODE` in the first code cell of `kaggle_smoke.ipynb`, then Run All. Gate
ladder: PROBE -> SMOKE -> RESUME.

1. `MODE = "PROBE"` (+ `PROBE_SEQ`, the sequence length being probed) ->
   Run All interactively (~15 min). Checks VRAM headroom at that sequence
   length AND pushes a checkpoint in the same session (`--save-steps 1`, no
   `--no-hub`) - SAVETEST was retired into this gate. Green = checkpoint in
   the HF repo printed by the re-home cell, `grad_norm` finite by step 3.
   **Any change to the cap goes through this gate before SMOKE** - that is
   the step the 12288 attempt skipped, and it cost a session.
2. `MODE = "SMOKE"` -> **Save & Run All** (background, ~1.2 h; immune to the
   20-min idle timeout). Green = loss down, no NaN, peak VRAM < 14 GB.
3. Fresh session, `MODE = "RESUME"` -> verifies checkpoint resume from the Hub.

Training always launches as 2-rank DDP via `torchrun` at seq 8192 — both T4s,
every session, no lane switches left to flip. A Kaggle T4x2 session bills
1x wall-clock regardless of GPU count. See `prev_rep.md` §1 for the full lane
history — including the Ministral, Qwen3-14B single-GPU, 2048-DDP, and 14B
model-parallel lanes this one replaced — and detailed metrics.

## Dataset build: unattended on GitHub Actions

The build runs itself (since 2026-08-29): `.github/workflows/data-worker.yml`
launches a ~5.5 h generate+judge job every 6 hours, resuming from a state
baton in the private HF dataset repo `tantan01/tuned-law-state` (SQLite
snapshot + raw NDJSON + streams, pushed back every 15 min). Judging runs in
**audit mode** — a 5% hash-sample gets the full dual-judge treatment, the
rest of the gate-clean rows ship as `audit:gate-accept`.

Operator surface:

- **Watch**: the repo's Actions tab; per-job logs also land in the baton
  (`logs/gen.log`, `logs/judge.log`).
- **Ship a dataset cut**: Actions -> `data-assemble` -> Run workflow. It
  reconciles, verifies, assembles, and pushes to the HF dataset repo only if
  `stats` is green; the `out/` artifacts upload to the baton either way.
- **Corpus size follows generation**: the chain's first content stage is
  `tuned.data.shape`, which trims the pre-built pools (4,320 replay, 1,700
  curated C1 — both sized for the FINISHED corpus) down to the profile the
  generated rows can support. `grounded_synthesis` can only come from the
  teacher, so it sets the size: one generated row buys ~2.05 corpus rows.
  Shipping the pools whole is what put `mix`, `trace` and `empty_think` red
  before this existed, and no amount of waiting fixed it — at the exact
  target mix the pools' own composition lands no-think at 34%, against a
  gate window of 18–20%. `--replay-nothink-share` moves the no-think budget
  between replay's chat slices and curated's raw legal rows; the default
  keeps the pools' composition, and lowering it buys corpus by changing
  what no-think is trained on, which is a design decision, not a knob.
- **Ship gate**: `python data/scripts/audit_readout.py <store>` prints the
  dual-judged sample's accept rate — the quality evidence for the whole
  audit-accepted batch. Read it before publishing.
- **One generator, ever**: the deepseek rate bucket is account-level. Never
  run `tuned.data.generate` locally while the cron is active, and never
  re-run `--phase seed-push` once the remote owns the baton (it would
  clobber the remote state with the stale local copy — single files go up
  via `HfApi.upload_file` instead).

The lane is qualified end to end; the dataset build is the only blocker for
the main run. History, retired experiment arms, and the design record live
in `prev_rep.md`.

## Rules that keep adapters swappable

- The base model revision is **pinned** in the config. Never train against
  `main`. Re-pin deliberately with `python training/scripts/pin_revision.py`.
- Every domain adapter uses the same base revision and the same
  `lora.target_modules` scoping.
- fp16 only on T4 (no bf16) — precision flags are explicit in code, never "auto".
- Save adapters only; never merge to 16-bit on Kaggle (blows the 20 GB disk).
- Secrets are env vars / Kaggle Secrets. `data/build/` and `outputs/` never enter git.
- One checkpoint repo per lane and per base model — sharing one means silent
  last-push-wins clobbering and cross-loading on `--resume`.
