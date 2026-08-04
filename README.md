# tuned

Local code, trained on Lightning.ai GPUs. Multi-adapter fine-tuning of Gemma 4 31B —
one LoRA per domain (Indian law first).

    edit locally -> git push -> Studio: git pull -> train

## Layout

| Path | Purpose |
|---|---|
| `src/tuned/` | Importable package (data, train, eval, serve). |
| `configs/law_v1.yaml` | Single source of truth: model pin, LoRA, run settings. |
| `scripts/` | Revision pinning, Studio bootstrap. |
| `docs/superpowers/` | Design specs and implementation plans. |

## Local setup (Windows, no GPU)

    uv venv
    .venv\Scripts\Activate.ps1
    uv pip install -e ".[dev]"
    python -m pytest tests/ -q

Training deps (`[train]`: unsloth, transformers 5.5.0) install only on a Studio.

## Lightning Studio setup (once)

1. lightning.ai -> new Studio (CPU is fine for setup).
2. Settings -> Environment variables -> add `HF_TOKEN` (a HuggingFace write token).
3. Terminal: `curl -fsSL https://raw.githubusercontent.com/Anant-T/Tuned/main/scripts/lightning_bootstrap.sh | bash`
4. Set `hub.checkpoint_repo` in `configs/law_v1.yaml` to `<your-hf-user>/tuned-law-v1-ckpt`, commit and push (or edit on the Studio).

## Smoke run (L4, ~$1.50)

Switch the Studio to an L4 24GB, then:

    python -m tuned.data.smoke                                    # ~1k examples
    python -m tuned.train.sft --config configs/law_v1.yaml --mode smoke

Success = loss trending down over 60 steps, `last-checkpoint/` visible in the HF
checkpoint repo, and a clean resume after killing the process:

    python -m tuned.train.sft --config configs/law_v1.yaml --mode smoke --resume

## Rules that keep adapters swappable

- The base model revision is **pinned** in `configs/law_v1.yaml`. Never train
  against `main`. Re-pin deliberately with `python scripts/pin_revision.py`.
- Every domain adapter uses the same base revision and the same
  `lora.target_modules` list.
- Secrets are env vars. `data/` and `outputs/` never enter git.
