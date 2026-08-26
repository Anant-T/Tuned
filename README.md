# tuned

Local code, trained on Kaggle free-tier GPUs ($0). Single production lane:
Qwen3-8B QLoRA, 2x T4 data-parallel (DDP) via `torchrun`, one LoRA per domain
(Indian law is the first).

    edit locally -> git push -> Kaggle notebook: clone -> train

## Layout

| Path | Purpose |
|---|---|
| `src/tuned/` | Importable package (data, train — eval and serve arrive with the main-run plan). |
| `configs/law_v1_8b_ddp.yaml` | The only config: model pin, LoRA, markers, run settings for the DDP lane. |
| `notebooks/kaggle_smoke.ipynb` | The one artifact uploaded to Kaggle; clones this repo and runs the CLI. |
| `notebooks/stage_model.ipynb` | Optional one-time, CPU-only notebook: stages the pinned model snapshot as a Kaggle Dataset input so training skips the hub download. |
| `scripts/` | Revision pinning. |
| `docs/reports/2026-08-08-project-record.md` | Full lane history (including retired lanes) and qualification metrics. |
| `docs/superpowers/` | Design specs and implementation plans. |

## Local setup (Windows, no GPU)

    uv venv
    .venv\Scripts\Activate.ps1
    uv pip install -e ".[dev]"
    python -m pytest tests/ -q

Training deps (`[train]`: unsloth, transformers 5.5.0) install only on Kaggle.
The template-drift test self-skips locally and runs on Kaggle.

## Kaggle setup (once)

1. kaggle.com account -> Settings -> verify phone number (gates GPU + internet).
2. Create an HF account with a **write** token (the notebook derives the private
   checkpoint repo from your token's account automatically — no config edit needed).
3. Kaggle -> Create -> Notebook -> File -> Import Notebook -> upload `notebooks/kaggle_smoke.ipynb`.
4. Notebook settings: Accelerator **GPU T4 x2** (never P100 — unsupported), Internet **On**.
5. Add-ons -> Secrets -> add `HF_TOKEN`. Optionally add `WANDB_API_KEY` too — it
   turns on live Weights & Biases metrics for the training cell (unset = no W&B,
   same as before).
6. Optional, saves ~7 GiB of hub traffic per session: run `notebooks/stage_model.ipynb`
   once (CPU only, zero GPU quota), then attach its output as an Input to
   `kaggle_smoke` (**+ Add Input -> Your Work**). It stages the pinned snapshot
   with a `REVISION.txt` guard — `kaggle_smoke` uses it only if that revision
   still matches the config pin, otherwise it falls back to the hub download.

## Smoke run

The lane (`configs/law_v1_8b_ddp.yaml`) raised `max_seq_length` 8192 -> 12288
on 2026-08-26 as deliberate headroom (today's longest row is 7,610 tokens).
**Requalification at 12288 is pending.** The numbers below are the reference
gate run at seq 8192, green on Kaggle on 2026-08-08; the memory shape at real
row lengths is unchanged by the raise, but the merged PROBE gate has not yet
run green at 12288:

- **PROBE** (2-step VRAM-ceiling check, pushes a checkpoint): per-rank peaks 12.80/13.00 GiB at seq 8192.
- **SMOKE** (60 steps): 60/60 complete, 74.7 s/step, ~438 tok/s aggregate, peaks 12.98/13.18 GiB.
- **RESUME**: fresh session, training continued from step 61 with optimizer/scaler state restored.

Set `MODE` in the first code cell of `kaggle_smoke.ipynb`, then Run All. Gate
ladder: PROBE -> SMOKE -> RESUME.

1. `MODE = "PROBE"` (+ `PROBE_SEQ`, the sequence length being requalified) ->
   Run All interactively (~15 min). Checks VRAM headroom at that sequence
   length AND pushes a checkpoint in the same session (`--save-steps 1`, no
   `--no-hub`) - SAVETEST was retired into this gate. Green = checkpoint in
   the HF repo printed by the re-home cell, `grad_norm` finite by step 3.
2. `MODE = "SMOKE"` -> **Save & Run All** (background, ~1.2 h; immune to the
   20-min idle timeout). Green = loss down, no NaN, peak VRAM < 14 GB.
3. Fresh session, `MODE = "RESUME"` -> verifies checkpoint resume from the Hub.

Training always launches as 2-rank DDP via `torchrun` at seq 12288 — both T4s,
every session, no lane switches left to flip. A Kaggle T4x2 session bills
1x wall-clock regardless of GPU count. See
`docs/reports/2026-08-08-project-record.md` for the full lane history —
including the Ministral, Qwen3-14B single-GPU, 2048-DDP, and 14B
model-parallel lanes this one replaced — and detailed metrics.

## Next milestone

The lane is qualified end to end; the only blocker for a main run is the
Indian-law dataset build. Spec:
`docs/superpowers/specs/2026-08-04-indian-law-adapter-design.md` (its sizing
is superseded — see the project record for current numbers).

## Rules that keep adapters swappable

- The base model revision is **pinned** in the config. Never train against
  `main`. Re-pin deliberately with `python scripts/pin_revision.py`.
- Every domain adapter uses the same base revision and the same
  `lora.target_modules` scoping.
- fp16 only on T4 (no bf16) — precision flags are explicit in code, never "auto".
- Save adapters only; never merge to 16-bit on Kaggle (blows the 20 GB disk).
- Secrets are env vars / Kaggle Secrets. `data/` and `outputs/` never enter git.
- One checkpoint repo per lane and per base model — sharing one means silent
  last-push-wins clobbering and cross-loading on `--resume`.
