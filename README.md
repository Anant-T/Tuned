# tuned

Local code, trained on Kaggle free-tier GPUs ($0). Multi-adapter fine-tuning of
Qwen3-14B — one LoRA per domain (Indian law).

    edit locally -> git push -> Kaggle notebook: clone -> train

## Layout

| Path | Purpose |
|---|---|
| `src/tuned/` | Importable package (data, train — eval and serve arrive with the main-run plan). |
| `configs/law_v1.yaml` | Single source of truth: model pin, LoRA, markers, run settings (single-GPU lane). |
| `configs/law_v1_ddp.yaml` | 2x T4 data-parallel lane (torchrun) — same quota cost, ~2.1x tokens/s. |
| `configs/law_v1_ministral.yaml` | Archived: Ministral, disqualified on T4 (see `docs/ministral-t4-disqualification.md`). |
| `notebooks/kaggle_smoke.ipynb` | The one artifact uploaded to Kaggle; clones this repo and runs the CLI. |
| `scripts/` | Revision pinning. |
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
5. Add-ons -> Secrets -> add `HF_TOKEN`.

## Smoke run (free, ~5-7 GPU-h of the 30 h/week quota)

1. `MODE = "SAVETEST"` -> Run All interactively (~15 min). Green = checkpoint in the
   HF repo printed by the re-home cell, `grad_norm` finite by step 3.
2. `MODE = "SMOKE"` -> **Save & Run All** (background, ~3.5 h single-GPU; immune to the
   20-min idle timeout). Green = loss down, no NaN, peak VRAM < 14 GB.
3. Fresh session, `MODE = "RESUME"` -> verifies checkpoint resume from the Hub.

`DDP = True` switches to the 2x T4 lane (`configs/law_v1_ddp.yaml`, own checkpoint
repo, ~2.1x tokens/s at the same quota cost — a T4x2 session bills 1x wall-clock
regardless of GPUs used). Resume in the same lane that saved.

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
