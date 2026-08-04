# tuned

Local code, trained on Kaggle free-tier GPUs ($0). Multi-adapter fine-tuning of
Ministral-3-14B-Reasoning — one LoRA per domain (Indian law first).

    edit locally -> git push -> Kaggle notebook: clone -> train

## Layout

| Path | Purpose |
|---|---|
| `src/tuned/` | Importable package (data, train — eval and serve arrive with the main-run plan). |
| `configs/law_v1.yaml` | Single source of truth: model pin, LoRA, markers, run settings. |
| `configs/law_v1_qwen.yaml` | Escape hatch (Qwen3-14B) if the Ministral LoRA-save bug fires. |
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
2. Create a private HF checkpoint repo and a **write** token.
3. Set `hub.checkpoint_repo` in `configs/law_v1.yaml` to `<hf-user>/tuned-law-v1-ckpt`; commit and push.
4. Kaggle -> Create -> Notebook -> File -> Import Notebook -> upload `notebooks/kaggle_smoke.ipynb`.
5. Notebook settings: Accelerator **GPU T4 x2** (never P100 — unsupported), Internet **On**.
6. Add-ons -> Secrets -> add `HF_TOKEN`.

## Smoke run (free, ~5-7 GPU-h of the 30 h/week quota)

1. `MODE = "SAVETEST"` -> Run All interactively (~15 min). Green = checkpoint in the
   HF repo, no LoRA-save error. This gate exists because of unsloth#5677.
2. `MODE = "SMOKE"` -> **Save & Run All** (background, 4-6 h; immune to the
   20-min idle timeout). Green = loss down, no NaN, peak VRAM < 14 GB.
3. Fresh session, `MODE = "RESUME"` -> verifies checkpoint resume from the Hub.

If SAVETEST fails on the LoRA save after one session of debugging: set
`CONFIG = "configs/law_v1_qwen.yaml"` in the notebook and rerun from step 1.

## Rules that keep adapters swappable

- The base model revision is **pinned** in the config. Never train against
  `main`. Re-pin deliberately with `python scripts/pin_revision.py`.
- Every domain adapter uses the same base revision and the same
  `lora.target_modules` scoping.
- fp16 only on T4 (no bf16) — precision flags are explicit in code, never "auto".
- Save adapters only; never merge to 16-bit on Kaggle (blows the 20 GB disk).
- Secrets are env vars / Kaggle Secrets. `data/` and `outputs/` never enter git.
