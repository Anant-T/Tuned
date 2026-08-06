# Ministral-3-14B disqualified on Kaggle T4 (2026-08-06)

`unsloth/Ministral-3-14B-Reasoning-2512-unsloth-bnb-4bit` (pinned `ec1befb...`)
cannot train under QLoRA on a 16 GB T4. Two manual runs OOMed at step 0, at
**both** seq 2048 and seq 1024, with `PYTORCH_ALLOC_CONF=expandable_segments:True`.
The config is archived as `configs/law_v1_ministral.yaml` (hub repo nulled so it
cannot push); the project's primary model is now Qwen3-14B (`configs/law_v1.yaml`).

## Root cause chain

1. **Fused CE loss fails to compile.** unsloth_zoo's fused cross-entropy uses
   `torch.func.grad_and_value`; on torch 2.10 + unsloth 2026.8.3 `torch._dynamo`
   raises "Unsupported functorch tracing attempt" and unsloth falls back to the
   eager functorch path.
2. **The eager fallback materializes the lm_head.** The failed allocation is
   exactly 1.25 GiB = fp16 131072 x 5120 — a full `lm_head` weight copy. It is
   **sequence-length-invariant**: dropping seq 2048 → 1024 reproduced the same
   1.25 GiB request, proving seq is not the lever.
3. **No headroom to absorb it.** Ministral ships **untied** embed + lm_head
   ("both present with different values" load warning), costing ~2.5 GB more
   residency than the tied-weights assumption in the original fit estimate,
   plus a vision-tower wrapper (`Mistral3ForConditionalGeneration`). Baseline
   before the loss: 13.5 / 14.56 GiB.

Everything upstream was green in the same runs: LoRA regex attached to
`language_model` only (121.9M trainable = 0.87%), NaN-guard filtering worked,
trainer configured. The unsloth#5677 save gate was never reached.

## Revisit conditions

Any of: a GPU with >= 24 GB (the 1.25 GiB spike plus untied heads fit trivially),
a torch/unsloth combo whose fused CE actually compiles on sm_75, or a tied-head
Ministral variant. Until then, do not spend Kaggle quota on it.

## What replaced it

Qwen3-14B (`unsloth/Qwen3-14B-unsloth-bnb-4bit`, pinned `46105e2...`): text-only,
tied embeddings, official unsloth Kaggle T4 notebook exists. Qualified same day:
single-GPU SAVETEST green (208 s/step, peak 12.9 GiB) and 2x T4 DDP SAVETEST
green (2.14x tokens/s, ~1 GiB headroom per rank).
