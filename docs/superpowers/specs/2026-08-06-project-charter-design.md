# Project Charter & v1 Scope — Grounded-Synthesis Research

**Date:** 2026-08-06
**Status:** Approved by operator (brainstorming complete)
**Relationship to prior specs:** Narrows the *scope* of
`2026-08-04-indian-law-adapter-design.md` (data mix, quality gates, training
recipe, and eval instruments remain the reference design) on top of the
platform decisions in `2026-08-05-kaggle-migration-design.md` (Kaggle T4,
Ministral-3-14B-Reasoning, Qwen3-14B escape hatch — all unchanged).

## 1. Charter

> **tuned** is a research project testing one falsifiable hypothesis: that
> license-safe, citation-verified synthetic data — generated from real Indian
> legal corpora by open-weight teachers and filtered through automated quality
> gates — can measurably improve a small open model's domain competence over
> its base, at $0 compute cost. The vehicle is a QLoRA adapter for
> Ministral-3-14B-Reasoning (Qwen3-14B escape hatch) trained on Kaggle's free
> T4 tier, with dataset volume deliberately trimmed to quality-dense examples
> sized to the measured token throughput of the ~30 GPU-h/week quota. The
> deliverable is evidence **plus a usable artifact**: an adapter on HF Hub
> that is genuinely good at Indian law — trained against the pinned base
> revision with swap-compatible target modules and adapter-only saves, so it
> drops directly into the already-specced llama-server hot-swap serving path
> in the following phase — together with a before/after evaluation report:
> BhashaBench-Legal delta (hypothesis threshold ≥ +3 points), blind LLM-judge
> win-rate on held-out data (≥ 55%), and forgetting guards (MMLU/IFEval within
> −2 of base). Serving deployment, multi-domain expansion, and productization
> are explicitly deferred until the hypothesis is confirmed. Immediate
> milestone: a green three-gate smoke run (SAVETEST → SMOKE → RESUME,
> currently pending at kernel v10) whose measured tokens/sec sets the dataset
> budget for the main run.

## 2. Hypothesis and pre-registered success criteria

**H1:** Grounded, citation-verified synthetic SFT data improves domain
competence over the base model.

| Criterion | Instrument | Threshold (pre-registered) |
|---|---|---|
| Domain competence | BhashaBench-Legal (English), 4-bit inference on T4 | ≥ +3 points over base |
| Generation quality | Blind pointwise LLM-judge on held-out split, both-order averaged | ≥ 55% win-rate vs base |
| No forgetting | MMLU 1k-sample + IFEval subset | within −2 points of base |
| Citation honesty | 200-generation spot-check against grounding corpus | no regression vs base |
| Usability | Adapter-only save, pinned base revision, swap-compatible target modules | loads and generates against the pinned base |

Thresholds are fixed *before* the main run. The report states results against
them whatever the outcome — a clean negative is a valid deliverable.

## 3. Deliverables (v1)

1. **Adapter** on private HF Hub: LoRA weights only, provenance-complete
   config committed to the repo.
2. **Evaluation report** (`docs/` markdown): baseline vs adapter on every
   instrument above, with the decision rule outcome (§5) and measured
   compute cost in GPU-hours.
3. **Dataset** on private HF Hub with per-example provenance metadata
   (source/chunk ID, teacher, license tag, difficulty, judge scores).

## 4. Scope

**In v1:**
- Kaggle smoke validation (in flight) and the main SFT run(s).
- Stage-1 data pipeline (§5): grounded synthesis, quality gates, citation
  verification, dedup, decontamination, held-out split.
- Full evaluation battery and written report.

**Out of v1 (deferred, doors kept open at zero cost):**
- GGUF export, llama-server hot-swap deployment, and the adapter-active
  serving check — next phase, triggered by H1 confirmation. The swappability
  *rules* (pinned revision, shared target-module scoping, adapter-only saves,
  fp16-explicit) remain binding in v1 because they cost nothing and keep the
  adapter usable.
- GRPO, medical adapter, Hindi expansion, app integration (unchanged from the
  Aug 4 spec's out-of-scope list).

## 5. Approach: staged falsification (Approach C)

Stage 1 is the smallest experiment that can falsify H1; scaling is
evidence-gated, not assumed.

**Stage 1 deltas vs the Aug 4 reference design:**

| Dimension | Aug 4 spec | Stage 1 |
|---|---|---|
| Volume | ~50k examples | ~8–12k quality-dense (final N from §6 formula) |
| Teachers | 4-model ensemble | One strong free teacher (Nemotron via OpenRouter free) + gpt-oss-120b for volume where rate limits bind |
| Judging | Two-judge blind on every example | Rule floors + one judge on every example; second-judge blind audit on a 10% sample |
| Citation verification | Every cited case/statute | **Unchanged — every example** (this is the method under test) |
| Dedup / decontamination | MinHash + 13-gram + embedding screen | **Unchanged** |
| Replay share | ~24% | ~15–20% (forgetting-guard insurance) |
| Sequence | 4,096 main + 8,192 long bucket | Single bucket at 2,048 or 4,096 (chosen from smoke VRAM/throughput); long tail truncated or excluded |
| Task mix | 4 grounded task families + curated + replay | Trim to 2–3 grounded families (IRAC analysis, statute QA; drafting only if budget allows) + curated + replay |
| Wall-clock | one L40S day | ~1–2 Kaggle quota-weeks, resume-from-Hub between weeks |

**Decision rule after Stage 1 eval:**
- **Delta ≥ +3 and guards pass** → H1 confirmed; write the report; v1 done.
- **Delta in (0, +3) or noisy** → scale toward the spec-faithful design
  (~25–30k, two-judge everywhere, full task mix) over additional
  quota-weeks; this is Stage 2 and gets its own plan.
- **Delta ≤ 0** → stop training; diagnose data quality (judge audit sample,
  citation-verification stats, contamination screen) before any further GPU
  spend.

## 6. Dataset sizing formula

The 50k figure is dead; N is derived, not assumed:

```
N ≈ (usable GPU-h per stage × measured tok/s × 3600) / avg tokens per example
```

- `measured tok/s` comes from the SMOKE run's telemetry (already
  instrumented in the notebook).
- `usable GPU-h` budgets ~25 h/quota-week (reserve for eval inference,
  resume overhead, and mistakes).
- Baseline eval inference (all instruments, base model) is budgeted *before*
  training and runs once.

Until smoke reports, ~8–12k at seq 2048 is the planning assumption
(one epoch ≈ 500–750 optimizer steps at effective batch 16).

## 7. Tech stack (confirmed)

**Training — unchanged, battle-tested; do not churn:** Kaggle T4 (fp16
explicit) + Unsloth QLoRA + TRL SFTTrainer, pinned model revisions, HF Hub
checkpoint push/resume, Popen supervisor + log-push-to-Hub telemetry, pytest
tripwires, uv.

**Data pipeline — to build, local + free tiers only:**
- `openai` SDK against OpenRouter / Cerebras free endpoints.
- stdlib `sqlite3` resumable state store (chunk → task → status → output),
  exponential backoff, per-provider daily-budget tracking.
- `datasketch` MinHash dedup; small `sentence-transformers` model on CPU for
  the embedding decontamination screen.

**Evaluation — to build:**
- Custom BhashaBench-Legal MCQ runner (4-bit inference on Kaggle T4).
- `lm-evaluation-harness` for MMLU/IFEval guards.
- Judge-eval reusing the data-pipeline API clients.
- Metrics JSON pushed to the Hub alongside checkpoints; report is markdown
  in `docs/`. No experiment-tracking service (existing Hub telemetry covers
  Kaggle-batch observability).

## 8. Current baseline and immediate milestone

- Repo: platform migration complete and merged; smoke infrastructure through
  kernel v10 (xet download fix in, result pending as of 2026-08-06).
- **Immediate milestone (unchanged):** green SAVETEST → SMOKE → RESUME with
  tokens/sec and GPU-hours recorded.
- **Next planning step after green smoke:** the Stage-1 main-run
  implementation plan (data pipeline + eval harness + sized main run),
  consuming this charter and the smoke measurements.

## 9. Risks specific to this framing

| Risk | Mitigation |
|---|---|
| Underpowered experiment: 8–12k examples too few for a +3 signal | The decision rule treats a marginal delta as "scale up", not "failed"; Stage 2 exists for exactly this |
| Single-judge filtering admits noise | Rule floors first; 10% second-judge blind audit with a pre-set disagreement tolerance (audit disagreement > 15% → re-judge the pool) |
| Eval contamination inflates the delta | Decontamination screen kept at full strength in Stage 1; BhashaBench screened explicitly |
| Free-tier API quotas stall generation | Resumable SQLite state store; generation is a multi-day background job by design |
| Smoke reveals throughput far below plan | §6 formula absorbs it: N shrinks or Stage 1 spans more weeks; charter unaffected |
