# Kaggle Free-Tier Migration — Ministral-3-14B-Reasoning (Design Spec)

**Date:** 2026-08-05
**Status:** Approved by operator (brainstorming complete)
**Supersedes platform/model choices in:** `2026-08-04-indian-law-adapter-design.md` (Lightning.ai + Gemma 4 31B). That spec's data-mix and multi-adapter goals remain in force.

## 1. Goal & scope

Migrate the training workflow from Lightning.ai (paid credits) to **Kaggle free tier** ($0, 2x T4, ~30 GPU-h/week), changing the base model from Gemma 4 31B — which cannot run on this hardware — to **Ministral-3-14B-Reasoning-2512**.

**Scope ends at a green smoke run on Kaggle:**
1. LoRA checkpoint save + HF Hub push proven (the "savetest" gate).
2. 60-step smoke completes with healthy loss.
3. Resume-from-Hub verified in a fresh session.
4. Throughput (tokens/sec) and GPU-hours measured.

The main training run (full Indian-law data pipeline, quota budgeting, DDP decision) is a separate later plan that consumes this run's measurements. The multi-adapter goal (law first, shared base) is unchanged.

## 2. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Platform | Kaggle free tier, accelerator **"GPU T4 x2", never P100** | P100 is sm_60 — below Unsloth's supported floor; current bitsandbytes/torch CUDA-13 wheels start at sm_75 |
| Base model | `Ministral-3-14B-Reasoning-2512` (Apache 2.0) | Strongest trainable model on this hardware: beats the 24B Magistral-Small-2509 on AIME24 (89.8 vs 86.1), AIME25 (85.0 vs 77.3), GPQA-D (71.2 vs 70.1); native `[THINK]` scaffold matches long-CoT training data |
| Bigger model via both T4s | Rejected | `device_map="balanced"` is sequential model-parallel (no speedup), has an open Kaggle T4x2 GPU-detection bug (unsloth#2864), and the only fitting 24B is *worse* than this 14B; Mistral Small 4 (119B MoE), Gemma 4 31B (fp16 overflow on T4), Qwen3.6-27B (quantizes badly) all remain disqualified |
| GPU usage | Single T4 for smoke (`CUDA_VISIBLE_DEVICES=0`); DDP-on-2xT4 probe deferred to main-run planning | 14B QLoRA fits one T4 (~11-12 GB of ~14.7 usable); PCIe-only DDP gain unproven — measure before committing |
| Risk posture | Ministral primary, **Qwen3-14B escape hatch** as a ready config | Open Unsloth bug #5677: vision tower breaks text-only LoRA *saving* on Ministral 3; fallback is one `--config` flag |
| Workflow shape | Thin committed notebook clones repo @ main, runs existing CLI entrypoints | Repo stays the tested source of truth; notebook is ~8 cells of glue |
| Precision | fp16, always explicit, never "auto" | T4 (sm_75, Turing) has no bf16; a bf16 default sneaking in is the top documented Kaggle T4 failure (`BFloat16 != Half`) |

## 3. Config layer

### 3.1 `configs/law_v1.yaml` changes

- `model.repo: unsloth/Ministral-3-14B-Reasoning-2512-unsloth-bnb-4bit`
- `model.revision: ec1befbd41647354531b2e09bd036cd1dc94b076` (sha verified via HF API 2026-08-04). Strict refuse-unpinned-revision loading unchanged.
- `lora.target_modules` becomes a **regex string** scoped to the language model only:
  `language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)`
  PEFT accepts regex target_modules; this prevents the vision tower from sprouting LoRA modules — the root cause of unsloth#5677's save failure. Schema accepts string-regex **or** list. YAML note: single-quote the regex (`\.` is an invalid escape inside YAML double quotes).
- New keys (model-specific strings become data, so the escape hatch is a pure config swap):
  - `model.instruction_part` / `model.response_part` — completion-masking markers for `train_on_responses_only`, replacing the hardcoded Gemma strings. **Exact Mistral-template values are read off the real tokenizer during implementation** — never assumed (external review proposed ChatML markers for Ministral; that is the wrong template family and would corrupt masking silently).
  - `data.think_open: '[THINK]'` / `data.think_close: '[/THINK]'` — reasoning-scaffold tags used by the smoke dataset builder.
- Unchanged: LoRA r=32 / alpha=32 / dropout 0.0, lr 2e-4, warmup_ratio 0.03, weight_decay 0.001, adamw_8bit, linear scheduler, seed 3407, smoke block (seq 2048, batch 1, grad-accum 16, 60 steps, save every 25), `hub.checkpoint_repo` operator-set.
- `max_grad_norm` stays at the TRL default (1.0) to match known-good Unsloth recipes. 0.3 (the QLoRA-paper value) is the first lever if the fp16 loss curve shows spikes — documented in §9, not preemptively applied.

### 3.2 New `configs/law_v1_qwen.yaml` (escape hatch)

Identical structure with: `model.repo: unsloth/Qwen3-14B-unsloth-bnb-4bit` (revision sha fetched from the HF API and pinned as an implementation task — no unpinned config may ship), Qwen ChatML masking markers (`<|im_start|>user\n` / `<|im_start|>assistant\n`, verified off the Qwen tokenizer), `data.think_open: '<think>'` / `data.think_close: '</think>'`, plain-list `target_modules` (Qwen3 is text-only; no vision tower to exclude).

**Trigger:** the savetest save-path failure persists after the regex workaround, timeboxed to one session of debugging. Switch = `--config configs/law_v1_qwen.yaml`. No code edits.

## 4. Training code — `src/tuned/train/sft.py`

- **Explicit fp16:** model load gets `dtype=torch.float16`; `SFTConfig` gets `fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported()` (correct on T4, still correct if run on a newer GPU).
- **Preflight additions** (inside `main()`, never at module import; existing hub checks unchanged):
  - Abort if `torch.cuda.get_device_capability(0) < (7, 0)` with the message to re-select the accelerator as "GPU T4 x2" (catches P100 and anything pre-Volta before quota is burned).
  - Print `torch` / CUDA / bitsandbytes / transformers / unsloth versions — the live-image record for pin adjustments.
- **Masking markers from config** (`model.instruction_part` / `model.response_part`) instead of hardcoded Gemma strings.
- **New CLI overrides `--max-steps` / `--save-steps`** on top of the smoke config. The savetest is `--max-steps 4 --save-steps 2`: an interactive ~15-minute run whose only job is proving LoRA save + Hub push before hours are spent.
- Header/docstring updated: Kaggle, not Lightning.

## 5. Data — `src/tuned/data/smoke.py`

Assistant content becomes `{think_open}{trace}{think_close}{solution}` using the config's tags (builder gains a `--config` argument). Source stays OpenThoughts-114k streaming, 1,000 examples, JSONL, seq 2048.

## 6. Kaggle notebook — `notebooks/kaggle_smoke.ipynb`

~8 cells; a `MODE` variable at the top switches SAVETEST / SMOKE / RESUME:

1. **Environment:** assert 2x T4 visible; set `CUDA_VISIBLE_DEVICES=0`; set `HF_HOME=/tmp/hf_cache` (scratch, ~60 GB, ephemeral — **never** `/kaggle/working`, which is the 20 GB persisted output and would snapshot a ~10 GB model cache into every saved version); set `HF_HUB_ENABLE_HF_TRANSFER=1`; print `df -h` to verify the real disk layout live.
2. **Clone** the repo @ main into scratch.
3. **Install** pinned deps via uv, plus `hf_transfer` (cache is ephemeral, so the ~10 GB base model re-downloads each session — the fast path pays for itself). Exact pins finalized against the live image (first session's version printout).
4. **Secrets:** `HF_TOKEN` from Kaggle User Secrets → env var.
5. **Preflight & versions:** run the CLI preflight path; record versions.
6. **Dataset:** build `data/smoke_v1.jsonl` with think tags.
7. **Train per MODE:** SAVETEST `--max-steps 4 --save-steps 2` (interactive, go/no-go) · SMOKE full 60 steps (run via **Save & Run All** — background, immune to the 20-min idle timeout, well under the 12 h cap) · RESUME `--resume` from Hub.
8. **Telemetry:** print final train loss, peak VRAM, and **tokens/sec** (feeds main-run quota planning).

## 7. Cleanup & docs

- Delete `scripts/lightning_bootstrap.sh`.
- Rewrite README for Kaggle: phone-verification prerequisite (gates both GPU and internet), HF_TOKEN secret setup, "GPU T4 x2 not P100", notebook flow, Save & Run All usage.
- Update `pyproject.toml` description, `src/tuned/__init__.py` docstring, and `where_am_i()` (detect `KAGGLE_KERNEL_RUN_TYPE`).
- Old dated Lightning spec/plan files stay in `docs/superpowers/` as historical records.

## 8. Tests (all local-runnable, no GPU)

1. Config tests re-pinned to the new repo/revision.
2. `target_modules` accepts regex-string or list; regex round-trips through YAML single-quoting.
3. `configs/law_v1_qwen.yaml` loads and validates.
4. **Template-drift test:** render a sample conversation through the real tokenizer chat template and assert the configured masking markers appear in the output — catches silent transformers-version template changes (the failure mode the ChatML-marker review suggestion would have caused).
5. Think-tag wrapping test for the smoke builder.

## 9. Success criteria & risks

**Green smoke means:**
1. Savetest checkpoint saves without the #5677 module-count error and lands in the private HF repo.
2. 60-step smoke completes; loss decreasing; no NaN (the fp16 canary); peak VRAM < ~14 GB (Kaggle T4s expose ~14.7).
3. Resume-from-Hub works in a fresh session (the 12 h-cap survival mechanism).
4. Tokens/sec + GPU-hours recorded. Expected smoke duration is **4-6 hours** (60 steps x 32,768 tokens/step on a T4) — duration is measured, not gated; savetest + smoke + resume ≈ 6-7 h, comfortably inside one 30 h week.

**Risks:**

| Risk | Mitigation |
|---|---|
| Save bug #5677 fires despite regex scoping | One-session timebox → swap to `law_v1_qwen.yaml` |
| Kaggle image dependency conflicts; `transformers==5.5.0` pin vs Ministral 3 (needs transformers v5 — confirm minor-version compat) | Preflight version printout; plan adjusts pins against the live image |
| fp16 loss instability | No Mistral-family fp16 reports exist; smoke loss curve is the check; first lever is `max_grad_norm: 0.3` (QLoRA-paper value) |
| Disk | Model cache in `/tmp` scratch (verified live via `df -h`); `save_total_limit=2`; adapters-only saves — never merge to 16-bit on Kaggle (~28 GB > 20 GB quota) |
| Unknown T4 throughput | Smoke measures it; feeds the main-run plan and the DDP decision |

## 10. Dataset-acquisition deltas for the main-run plan (informational)

Acquisition strategy is unchanged — teacher-ensemble synthesis via free OpenRouter credits, grounded corpora (KanoonGPT, NyayaAnumana), IndicLegalQA/PredEx all happen off-GPU and land on HF Hub. What the platform/model change does affect:

1. **Format target:** synthesized traces must emit the Mistral Reasoning scaffold (`[THINK]…[/THINK]`, Mistral chat template) instead of Gemma format — one config-driven formatting layer.
2. **Length budget:** the L40S-era 4096/8192 bucketing shrinks; on a 16 GB T4, synthesis should target traces that mostly fit 4096, with the long tail truncated or excluded (decided in the main-run plan from measured tokens/sec).
3. **Volume vs quota:** ~50k long-CoT examples at T4 speed may exceed 30 GPU-h/week for one epoch — the main-run plan may trim volume, trim trace length, or span multiple weekly quotas; acquisition should prioritize quality-dense examples.
