# Previous findings and retired configuration (consolidated archive)

This file is the single surviving record of `docs/reports/` (15 dated reports, 2026-08-08 to 2026-08-28) and of the configuration surface deleted in the 2026-08-28 single-project restructure. Written 2026-08-28, immediately before those files were removed.
Every number below is measured and is quoted from the report or config that measured it; where a claim was never corroborated it says so.
`git log` holds the full originals — nothing here replaces the history, it replaces the working copy.

---

## 1. Training-lane record

Source: `docs/reports/2026-08-08-project-record.md` (872 lines, the canonical lane history; if it is absent from a future checkout, that path in the main repo's history is the pointer) and `docs/reports/2026-08-09-data-strategy-research.md`.

### 1.1 The lane that ships

**Qwen3-8B, 2× T4 DDP, seq 8192** — `configs/law_v1_8b_ddp.yaml`, `unsloth/Qwen3-8B-unsloth-bnb-4bit` @ `62efd7f9d748e394734a7adae2adf96e13a2abc8`, bs 1 × ga 2 × 2 ranks = 32,768 tok/optimizer step, LoRA r32/α32/dropout 0, ChatML markers, `<think>`/`</think>`, ckpt repo `tantan01/tuned-law-v1-qwen8b-ckpt-ddp`, `UNSLOTH_CE_LOSS_N_CHUNKS=16` exported before `import unsloth`.

| Gate (2026-08-08) | Result |
|---|---|
| PROBE | 2 steps @ 8192, loss 0.6618→0.6299, peaks **12.80 / 13.00 GiB**, nan grad_norm = GradScaler calibration |
| SAVETEST | 4 steps, peaks 12.98 / 12.77, ~82 s/step, 13-file resume set pushed |
| SMOKE | **60/60**, `train_loss` **0.5722**, peaks **12.98 / 13.18**, **74.7 s/step**, ckpts at 25/50/60 |
| RESUME | steps 61–64, losses 0.5955/0.5351/0.5983/0.6038, grad_norms 0.086/0.076/0.110/0.101 — **no calibration nan = `scaler.pt` restored**; verified operator-side against the live ckpt repo, never committed |

Throughput: ~219 tok/s per rank, **~438 tok/s aggregate**, ~1.58M tok/GPU-hour. Whole ladder ≈ 3 of 30 weekly GPU-h.

### 1.2 Retired lanes (all deleted 2026-08-08)

| Lane | Seq | Peaks GiB | s/step | tok/s | Gates | Verdict |
|---|---|---|---|---|---|---|
| Gemma 4 31B (Lightning L40S) | 2048 | — | — | — | none | Died on paper; 31B does not fit a 16 GB T4 |
| Ministral-3-14B-Reasoning | 2048/1024 | 13.5 baseline | — | — | none | **DISQUALIFIED** 2026-08-06: OOM at step 0 at both lengths; failed alloc exactly 1.25 GiB = fp16 131072×5120 (a full `lm_head` copy), sequence-length-invariant; cause = unsloth_zoo fused CE falling back to eager functorch on torch 2.10 + sm_75, with untied embed+lm_head costing ~2.5 GB |
| Qwen3-14B 1-GPU | 2048 | 12.9 | 208 | ~158 | SAVETEST | Superseded — 60-step smoke ≈ 3.5 h |
| Qwen3-14B DDP | 2048 | 13.6 / 13.45 | ~100 | ~338 (2.14×) | SAVETEST | Killed by the seq-length verdict, not by failure |
| Qwen3-14B MP (`sequential`, max_memory {0:8GiB,1:13GiB}) | 6144 / 8192 | 5.98/12.56 · 9.91/8.75 | 179.6 / 295.7 | ~162 / ~131 | PROBE only | Superseded by 8B DDP: same 8192 at 3.3× the token rate |

Load-bearing verdicts behind those retirements: **truncating legal documents at 2048 trains hallucination** (so 8192-MP beat 2048-DDP on quality; DDP's speed edge is concurrency, not an attention tax) — uncommitted, corroborated only by its consequences. The 2026-08-07 model screen was one criterion: **4-bit weight footprint ≤ ~10.4 GiB**; embedding tying is not itself a criterion. `balanced` + `max_memory` fails outright (surplus dispatched to CPU, bnb-4bit refuses).

### 1.3 Ops rules that outlived their incidents

Never train interactively (20-min idle kill); never cancel the training cell (batch mode discards a cancelled cell's buffered output); supervisor tees to `progress/train.log` in the ckpt repo every 5 min with 60 s heartbeats, pushes on a daemon thread; **xet must stay ENABLED** (disabling it was the v6–v8 stall root cause) and `HF_HUB_ENABLE_HF_TRANSFER` must be present and `"0"`; `HF_HOME=/tmp/hf_cache`, never `/kaggle/working`; `PYTORCH_ALLOC_CONF=expandable_segments:True`; `check_ddp_visibility` (a leaked `CUDA_VISIBLE_DEVICES=0` into torchrun cost a session); one checkpoint repo per lane; divergence guard raises on 3 consecutive non-finite `grad_norm` after a 2-step grace window and must **raise**, never set `should_training_stop` (that exits rc=0 and reads as green); RESUME always extends (`--resume --max-steps 64`) because a bare resume at `max_steps` is a no-op false green; edit the Kaggle notebook **in place** — re-import silently loses Secrets and Inputs; every training dep `==`-pinned (`unsloth==2026.8.3`, `transformers==5.5.0`, `trl==0.24.0`, `peft==0.20.0`, `bitsandbytes==0.50.0`); a broken bitsandbytes flips `ALLOW_PREQUANTIZED_MODELS` and the loader then drops the revision pin into a ~28 GB fp16 download.

### 1.4 Dataset spec (2026-08-07 operator decision; supersedes the Aug-6 charter row)

15–20k Indian-law examples · mix 60% grounded synthesis / 16% curated / 24% replay · **≥80% reasoning traces (hard floor)** · single 8192 bucket, **drop, never truncate** · builder ships **UNPACKED**, TRL packs at train time (`packing=True` + default `bfd`); pre-packing in the builder is silent contamination · synthesis T=0.6–0.8 · ~37–50M tokens ≈ **23–32 GPU-h ≈ one quota-week per epoch** · builder gates: tokenized-length histogram p50/p90/p99, label-mask assert, `pad == <|endoftext|>`, citation set-difference, cross-code flag. Two pieces exist only on paper: `build_sft_config` does **not** set `packing`, and the `find_packed_sequence_indices` CPU pre-flight assert is unwritten.

### 1.5 Data-strategy research (2026-08-09), the rules still binding

- **Token budget 2,500/example** (~700 prompt + ~1,400 think + ~400 answer). Target 1,200–1,800 thinking tokens; never paste full judgments (ILDC ≈4.4k, In-Abs ≈5.8k, IL-PCR ≈10.7k tokens).
- **Rewrite, don't synthesize**: the teacher rewrites authentic court reasoning (PredEx 12,178 rows; TathyaNyaya 25.4k) into first-person `<think>` + IRAC answers. Aalap's failure mode becomes structurally impossible.
- **MSLR rule (highest-value single rule)**: never script the `<think>` block; enforce IRAC only in the final answer. Human CoT scaffolds cost QwQ-32B −33.8% LLM score / −10.2% IRAC recall.
- **Citation existence is a hard reject, never a repair** (Lexis+ AI 17%, Westlaw 33%, GPT-4 43% hallucination *with* RAG). O(1) check against 17.1M KanoonGPT `neutral_citation` values, `headnote_text` stripped (EBC v. D.B. Modak: headnotes are copyrighted; raw judgments are not).
- **BNS §358 is the moat**: offences before 2026-07-01 continue under IPC/CrPC/IEA, so both code families are live law for a decade; nearly every public dataset is silently frozen pre-July-2024. Dedicated transition stream ~1,100 examples.
- Licensing: AWS `indian-high-court-judgments` / `indian-supreme-court-judgments` (CC-BY-4.0) are the primary grounding corpus, PDF-only (budget extraction engineering); IL-TUR/IL-PCSR/ILDC/HLDC/BhashaBench/AIBE are **eval + decontamination only**; SCC Online / Manupatra / LiveLaw / Bar & Bench are UNSAFE. Hindi cut from v1.
- Proposed charter amendment (still pending): make the blind pairwise judge the primary gate and demote BhashaBench/MMLU/IFEval to forgetting guards — Aalap moved task-shaped generation but **zero** MCQ movement, so the ≥+3 gate may be unable to detect success.

---

## 2. Data-pipeline campaign history

The `docs/reports/*.md` files these sections consolidate were deleted on
2026-08-28 (commit `3406377`). Code and config cite the section here; to
read an original, `git show 3406377^:docs/reports/<name>.md`.

Chronological. Every arm ran in an isolated `data/build/exp_*` workdir; the live control store `data/build/state/law_v1.sqlite3` was opened `mode=ro` throughout every arm below and its fingerprint (`size 554532864`, `mtime 1787309490`, `sha256 2ea51e4c996273fbee6d79ee1d632b6677c8752d50cb9f45258370f07fcc8f48`) is identical before and after all of them.

Live store as last measured (2026-08-24): 1,396 generation rows; task states `accepted 15, generating 8, judge_error 34, judging 43, pending 127, rejected 414, stale_prompt 419`; **46 `gold_label` rows, 0 `judge_threshold` rows**.

### 2.1 Pilot era — gpt-oss / mistral (to 2026-08-19)

| Finding | Numbers | Verdict |
|---|---|---|
| **Retry effort ladder retired** (2026-08-18, `generate.EFFORT_LADDER_RETIRED`) | Over 221 gated pilot generations, bumping `reasoning_effort` per attempt drove trace length 2,411 → 12,353 → 12,536 chars (5.2×) and every gate with it: verbatim_overlap 35/60 → 58/59 → 58/58, banned_meta 11/60 → 51/59 → 53/58, irac_placement 6/60 → 39/59 → 45/58, length_band 21/60 → 29/59 → 32/58, `finish_reason=length` 0 → 18 → 16. The only gate helped was `self_verification` (51/60 → 22/59) and only by accident — a 5× longer trace hits one of twelve literal cues by chance | **RETIRED, uniformly negative.** Retries now send identical effort params plus a machine-generated repair note |
| **mistral-small demoted from generator** (2026-08-18) | 56 generations, **zero** rows passed the gates; **50/56 = 89%** failed `irac_placement` because it writes a numbered IRAC outline inside its reasoning and fills it in | Generator seat deleted. (magistral retired upstream 2026-07-31; its `-latest` alias rides Small 4, which is why 43/43 pilot generations were traceless — an opt-in reasoning channel, not a model that cannot think) |
| **mistral-small disqualified as judge** (2026-08-19, human calibration on 46 gold labels, 10 accept / 36 reject, base rate 21.7%) | Fitted rule `mean>=3`; **holdout n=40 precision 0.237, recall 1.000, phi 0.124**; cross-val n=6 precision 0.200. Gate is precision ≥ 0.75. Structural: never awarded validity ≥ 4 over 46 judgements (values {2,3}, sd 0.48), emitted the single vector 5/3/4 on 59% of calls, r=+0.22 with the other judge; forced a wasted regeneration on 41% of rows and a paid tiebreak on 100% of accepts — 101 judge calls bought 46 scored generations | **OUT.** Fence covers `mistral-small-latest` specifically; `mistral-large-latest` is *unproven*, not disqualified |
| **gemma-4-31b promoted to judge slot B** (2026-08-19) | Hand-forensics rated it the most accurate of the three judges (caught qwen false positives on gens 391, 441, 470, confirmed by literal source comparison). Could not be fitted because **all 9** of its gold-labelled judgements landed in the holdout fold — a sampling accident, not a verdict | **PROMOTED**, least-bad evidence |
| **`verbatim_overlap` gate re-audit** (2026-08-18) | `DEFAULT_MAX_RUN` 30 → **120**, "where the curve flattens": gpt-oss curve read 24% fails at 80, 13% at 120. At 30 the gate "was rejecting the act of thinking about the matter at all" | Shipped; re-fitted again 2026-08-28 (§2.5) |
| **Prompt rewrite** (2026-08-18) | Added "450 to 700 words of deliberation is normal for a matter of any substance" to all 14 generator templates expressly to push gpt-oss traces **up** — gpt-oss's failure mode is `think < think_min` (381/1,281 = **29.7%** of live gpt-oss generations, median `think_est` 620 tok). Moved IRAC placement; never moved `self_verification` (66% live, unchanged) | Shipped |
| **cerebras window pin corrected** (2026-08-19) | Stale `max_context: 8192` put the unroutable cliff at 6,554 routing tokens and **swallowed 85% of the statute_qa stream**; probed 131,072 moves the cliff to 104,858 against real prompts of 1,445–2,799 | Fixed |

### 2.2 Judge-fleet evolution and the recovery arm (2026-08-23 / 24)

`docs/reports/2026-08-23-recovery-arm-probe.md` — `exp_recovery`, cerebras/gpt-oss-120b, harmony prefill alone (`harmony_s1_continue: false`), 60 pairs, **$0.00 spend**.

- Fleet repairs held: OpenAI spend $0.3396 → **$0**, `judge_parse_error` 96 → **0**, `judge_route_error` 235 → **0**, `judge_error` 34 → **0**. Slot A qwen (46 calls) / slot B gemma (46) are different families, so `family_separation` is satisfiable with no OpenAI at all; the gpt-5 refs were never reached, so the `reasoning_effort: 'minimal'` fix and the `ground_faithfulness` alias remain **correct-by-construction and unrefuted, not confirmed**.
- **The prefill buys format; the s1 `" Wait"` continue buys the verification ritual.** Blocking-gate yield **75.0%** (best ever measured here: summarization 85%, drafting 75%, irac 60%) but `self_verification` fails **88.3%** against 10.4% with s1 on. The reasoning ritual is not a property of the prompt or the prefill.
- Accept 25/60 = 41.7% of cohort, 25/44 = 56.8% of decided — confounded by judge composition, not a clean delta. The kill axis moved: grounding 3.42 became the lowest axis (validity 3.93–4.04, previously binding at 2.86–3.74).
- **Structural blocker found:** `prompt_echo` has zero rows in the entire control store's `gate_result` table (the gate postdates those generations), so `required_gates_complete` demands 11 gates and the frozen control store can supply 10. **`eval_matched` cannot return anything but `inconclusive` against that store, for any treatment arm however good.**
- `statute_qa` absent as a data fact: **270 statute_qa tasks, 0 eligible seeds** — `statute_section_eligible` needs `meta_json.section_text` distinct from the seed body and no seed carries provision text.

`docs/superpowers/plans/2026-08-24-judge-calibration-and-yield.md` evidence base (four converging investigations):

| Finding | Measurement |
|---|---|
| Tiebreak arbiter saturated | mistral-large 9/9 then 18/18 accepts, validity 5.00; same model, same seat, frozen store: validity 2.75 over 12. Cause was the overlay, not the model — `prompts_harmony/judge_tiebreak_v1.md` had deleted the worked example containing `"validity": 2`. Cross-arm inflation: mistral **+2.25**, gemma +0.88, qwen +0.22 |
| Grounding band 3 unreachable | Band 2 ("provision, case or **rule**") and band 3 ("proposition of substance") name the same thing; score distribution 1 / **41** / 7 / 19 / 33, and band 2 is the only band that hard-fails (`FAIL_MAX = 2`) |
| Teacher error is the dominant quality cap | Two independent methods agree: rationale classification **51%**, parent-document forensics **49.0%** |
| Chunking is NOT the dominant cause | Allegedly unsupported items matched against the parent on disk: **18.4%** severed by the chunker, **49.0%** absent from the parent entirely |
| Footnotes amputated | **96.2%** of inline footnote markers (1,909/1,984) point at a `[FOOTNOTES]` block in a different chunk |
| Role-aware segmentation never ran | `roles_json` empty on **60,603/60,603** seeds; 98.5% of SC chunks are `roles_backend_none` packing fallbacks |
| Drafting structurally unwinnable | `document_kind`, `party_context`, `focus_issue`, `question` empty on **all 60,603 seeds**; accepts 4/20 vs summarization 13/20, **p=0.0095** → drafting parked at weight **0.00** in `SYNTHESIS_MIX` |
| s1 continue is the efficiency win | Cost per all-gates-clean row **66,808 → 11,725 tokens (5.7×)** despite 2.3× more tokens per generation; within-s1, higher k gets worse |
| Statute enrichment bounded / case enrichment dead | 915 distinct sections cited, top 100 cover 73.2%, old-code 99.91% of mentions. 25,932 distinct case citations, top 100 cover 6.2%, corpus self-coverage 0.33% |

The s1 A/B, same 60 pairs, one flag, deterministic gates only: all-gates **6.7% → 68.3%**, blocking gates 73.3% → 78.3%, `self_verification` fails 90.0% → **13.3%**. Its accept-rate half (41.7% → 56.7%) is **withdrawn as uninterpretable** — the arms drew different judge fleets (exp_s1 had gpt-5-mini in slot B for 22 rows grading grounding at 4.91 where gemma graded 2.88).

### 2.3 deepseek qualification and the slot-B gap (2026-08-25 to 27)

`docs/reports/2026-08-27-deepseek-as-judge-slot-b.md` — `bai/deepseek-v4-flash` given `routing.judge` position 3 and `routing.tiebreak` position 4 with `role_params: {thinking: {type: disabled}, temperature: 0.2}`.

- Pass line fixed before the run at 9/10 parseable three-axis verdicts. **Measured 9/10 (PASS by one call)**; confirmatory batch 10/10; **20 of 21 across every call made**. The single failure is the predicted mode: HTTP 200, `finish_reason: length`, entire 1,024-token reply budget consumed by reasoning, empty content.
- `thinking: disabled` suppression is **probabilistic and bimodal**: of the 17 calls where `completion_tokens_details` was reported, 14 (~82%) had zero reasoning; the non-zero values were 637, 748, **1,024** — no useful middle. Truncation 1/21 = 4.8%, 95% CI roughly **0.1%–24%**.
- `reasoning_effort: minimal` measured **worse** (7/10 parsed, 3 truncated), so the 2026-08-25 "both work" note reads as "neither knob is deterministic". No effort key was added. The untried arm — `thinking: disabled` with **no** `reasoning_effort` key at all — is unreachable from config: the merge is a dict overlay (`model.params < role_params[role] < req.params`), which can override but never unset.
- Judging quality **entirely unmeasured** here. The probed candidate was deepseek's own generation, so the scores are self-assessment. Twenty verdicts on one unchanging candidate at the inherited `temperature: 0.7`: (5,5,5)×12, (4,3,3)×3, (4,5,5)×2, (4,4,4), (4,4,5), (5,4,5) — under `min_axis 4`, **3 of 20 draws flipped the decision** with nothing about the candidate changing. `temperature: 0.2` was then pinned; every number in that report was measured at 0.7 and was not re-measured.
- **Slot B on a deepseek row was still empty after the change**: `cerebras/gemma-4-31b` and `cerebras/gpt-oss-120b` both returned HTTP 402 that day, and `family_separation` excludes only the generator's family. This is exactly what stalled the 2026-08-23 live drain with 34 `judge_error`. `groq/openai/gpt-oss-20b` was added the same day to fill it — free, alive, family gpt-oss, therefore not excluded on a deepseek row.

### 2.4 The deepseek campaigns (2026-08-26 to 28)

**Validation wave** (`2026-08-27-deepseek-validation-wave.md`, `exp_deepseek`, 40 tasks / 99 calls): infrastructure passed everything (99/99 with content, all deepseek, $0, seed gate held with 35 oversize seeds present and 0 planned against — first live confirmation), but **79/99 generations were gated out** and only 20/40 tasks reached a judge. `think_max` violation **44%** against a 30% pre-registered threshold. The n=4 qualification probe under-counted reasoning by 31% (2,097 projected vs **2,739** measured); row mean 4,440 → **5,677**; rows over 8192 1.5% → **8.1%**. Like-for-like (reweighted to the projection's seed mix) the miss is **+22% on the row mean and 3.7× on the over-cap share**. The 0.77 answer-scaling assumption is **retracted** — deepseek answered 1,036 tokens against gpt-oss's 1,027 (factor 1.01); it does not answer shorter, it thinks ~4× longer. Per-gate fails: `irac_placement` 62%, `length_band` 51%, `verbatim_overlap` 46%, `banned_meta` 14%, `prompt_echo` 13%. Economics measured, not projected: **11 calls per accepted row**, 225 calls/hour (latency-bound at 4 workers, zero 429s) → ~165,000 calls ≈ **31 days** for a 15k corpus, recoverable toward ~11 days by concurrency alone. 8 rows died in `judge_error` on gemma cooling from judge batch 5 — a clean temporal cut, **6 tathya + 2 predex**, so no tathya row was ever fully decided and its "0 accepted" is arithmetic, not a quality signal.

**Prompt-ceiling A/B v4 vs v5** (`2026-08-27-generator-prompt-length-fix.md`): the edit `286fd3a` (700-word hard ceiling in all 14 templates) **did not shorten traces**. Median trace 1,727 → 2,507 words; **2 PASS / 5 FAIL** on pre-registered lines. Matched at attempt 1 over the same 40 seeds and prompt variants: median per-task change **−58 words**, 21 shorter / 19 longer, sign test **p = 0.87**, bootstrap CI [−378, +476]; McNemar **p = 0.80–1.00** on all four gates. Decisive on-the-wire finding: **86% of treatment generations exceeded the instructed 700-word ceiling**, against 84% under the permissive wording — *the prompt is not a lever on this generator's trace length*. The two arms ran **13h41m apart** with no provider-side upstream id, so every pooled between-arm number in that report is confounded; the nulls are not. `irac_placement` failure mode is unambiguous: **62 of 62** failures are an IRAC heading inside the trace, **0** a heading missing from the answer. Rows over the 8,192 cap 8.1% → 22.3%. A smaller output budget was already measured and rejected: 4,096 tokens returned empty content on **10 of 20** synthesis calls vs 0 of 4 at 12,288, and the survivors are biased short — "silent selection on the corpus rather than honest sampling".

**gpt-oss under the ceiling** (`2026-08-27-gptoss-floor-under-the-prompt-ceiling.md`, both arms back to back, 13 s apart, 90 generations each): **1 PASS / 4 FAIL**. `think < think_min` breach 44.4% → **57.8%** (+13.4pp against a +5pp allowance); median trace words 324 → **292** (both arms fail the 400 floor); `length_band` pass 55.6% → **42.2%** (−13.4pp); `self_verification` 31.1% → 25.6%. The edit measurably harms the generator it was never tested on. Caveat: a newly funded cerebras key means these numbers are cross-account against the 1,281-row live baseline, though both arms share the new account.

**Prompt-era rerun** (`2026-08-28-deepseek-prompt-era-rerun.md`, gap **56.18 s**): pre-registered primary `length_band` over all generations — v4rerun **47.42%** (46/97) vs v5rerun **42.45%** (45/106), delta **+4.97pp → WASH** (threshold 5pp). First-attempt-only cut gives +12.5pp, which alone would cross the revert line, but at n=40/arm that is ~1.1 SE and does not overturn the pooled call. Secondaries all lean the same way: full-gate clean 16.49% vs 8.49%; tokens per length-passing row 13,497 vs 15,773. Five workers against the `rpm: 8` bucket cost nothing (0 429s over 203 requests).
**Outcome: the ceiling edit was reverted (`06f588a`)** — proven harmful to gpt-oss, at-best-wash for deepseek, helping nobody.

**Three-arm clause/cap A/B** (`2026-08-28-deepseek-clause-and-cap-ab.md`, `ctl2 → clause → cap`, 63 s and 61 s apart):

| Lever | Line | ctl2 | treatment | Verdict |
|---|---|---|---|---|
| E2 clause | PRIMARY irac_placement fail | 73.39% (80/109) | 68.93% (71/103) = **−4.46pp** vs a ≥15pp bar | **FAIL** |
| E2 clause | GUARD length_band pass | 42.20% | 58.25% = **+16.05pp** vs ±5pp | **GUARD-BREACH** |
| E1 cap 16384→5000 | PRIMARY completion tok / length-passing row | 9,175.7 | 8,193.4 = **−10.71%** (true ledger spend, not the naive 28.45%) | WIN |
| E1 cap | GUARD length_band pass | 42.20% | 50.00% = +7.80pp vs ±3pp | **GUARD-BREACH** |

The clause is a general de-verbosifier (think p50 −18.7%, mean −14.8%) rather than a targeted fix; the rehearsal count only moved 73.4% → 66.0%. The cap's hidden cost, found by task-level cross-reference across the shared 40 task ids: 9 tasks hit truncation-before-content, 6 of them had passed `length_band` in the control, and **4 exhausted all three attempts and parked `gen_unroutable` — 10% of the arm's tasks converted from a usable row to nothing.** **Cap closed, not shipped.** The clause nevertheless **shipped as a think-length lever** (`0637fa0`, six templates) on its secondaries: `length_band` pass 42.2% → 58.3%, ledger tokens per length-passing row 15,210 → 10,359 (**−31.9%**).

**hy3 qualification probe** (`2026-08-28-hy3-think-low-probe.md`, 25 tasks / 72 attempts): **FAIL.** `length_band` **19.44%** (14/72) against a ≥60% bar, think est p50 3,440 against a 3,000 band. Format integrity perfect (0% breakage, 72/72 `think_format`, all `finish_reason: stop`), legal content sound on a 5-row spot read — the model fails this build's band, not the task. `irac_placement` fails **95.83%** (69/72), all rehearsal, worse than deepseek's 73.39% — **the headed-rehearsal pathology is cross-model, not a deepseek quirk.** Docs-vs-API divergence recorded: b.ai's own hy3 page names `no_think`/`think_low`/`think_high`, but `think_low` returns **HTTP 400** — the real enum is `no_think, none, off, minimal, low, medium, high, xhigh, max`.

**The irac stop-timing A/B** (`2026-08-28-irac-stop-timing-fix.md`, two pairs, 38.8 s and 43.0 s apart) — the campaign's largest measured effect:

| Line | ctl (n=108) | fix (n=92) | Verdict |
|---|---|---|---|
| irac_placement fail | 65.74% | **39.13%** (−26.61pp vs a −15pp bar) | PASS |
| length_band pass | 49.07% | 40.22% (−8.86pp vs −5pp) | FAIL |
| full-gate clean | 13.89% | **26.09%** (+12.20pp) | PASS |

Attribution is unambiguous because F2 touched summarization only: irac_analysis (F1 only) **−0.88pp**, summarization (F1+F2) **−58.64pp**; think-side rehearsal on summarization 70.83% → **4.88%**. End-to-end, 24 tasks reached judging against 15, from 92 calls against 108 — **60% more usable rows from 15% fewer calls**. The guard breach runs *against* the treatment (irac-failing traces are longer traces in both arms; restricted to band-passing rows the effect grows to −39.32pp). Judge spot-check on the new prose form: **10/11 summarization accepts** (the single miss was marked down on validity and coverage by both judges independently).
F2-only confirm (`ctl3` vs `f2only`): summarization irac fail 72.73% → **7.89% (−64.83pp)** — F2 alone does slightly better than F2+F1, so **F1 is dead, measured inert twice**. But summarization `length_band` pass 65.91% → 50.00% (**−15.91pp**), reproducing the combined arm's −14.79pp: **the length cost belongs to F2 and is structural** — taking the headings away makes the model deliberate longer in prose (summarization think p50 +23.8%, p90 flat). Pre-registered rule returned **NO-SHIP**; the **operator overrode it on end-to-end totality and F2 shipped in `ebde9a7`**. The untreated task type doubled as a noise channel and earned its place: on byte-identical templates an hour apart, `prompt_echo` drifted **+17.59pp** and `verbatim_overlap` **−13.85pp**, while the treatment's own gates drifted 1.55–2.22pp — which is the only reason the −64.83pp win and the −15.91pp cost can both be called real.
F3 (the rewritten retry note) is **kept but inert**: conditional recovery is 34.6–34.7% in both arms, inside the historical 22–39% band. What the prompt changes is stability — a fix-arm trace that placed IRAC correctly kept doing so on the next attempt 80.8% of the time against the control's 47.4%.

**First judge calibration** (`2026-08-28-deepseek-judge-calibration.md`, E4): pooled accept **82.9% (29/35)** — v4 80.0% (16/20), v5 86.7% (13/15) — on the 100% production pair (qwen slot A + gemma slot B, `groq/openai/gpt-oss-20b` armed but never invoked). Cost **125,516 cerebras tokens ≈ $0.071** of a $0.50 cap; qwen used 83,706 / 200,000 tpd. **The judges are not the bottleneck** — only 20/40 and 16/40 tasks ever reach judging; the binding constraint is pre-judge format compliance. By source: predex 16/17 (94.1%), tathya 9/10 (90.0%), **sc 4/8 (50.0%)** — and `sc` also has the highest format-park share (18/26 = 69%), two independent signals on the same source. All 5 tiebreak invocations scored the identical vector (grounding 4, validity 3, coverage 4). One v5 row parked on a genuine stale-prompt guard (`gen_irac_analysis_v3:63b7780879f2!=09e8c6ffaf80`) — running judge regeneration concurrently with template edits in a shared worktree is a live race, not a theoretical one.

### 2.5 Root causes and recalibration (2026-08-28)

`docs/reports/2026-08-28-verbatim-overlap-drafting-drift.md` — both follow-ups are the same story: **a gpt-oss-era calibration meeting the deepseek-era generator.**

- `verbatim_overlap` is a **generator-calibration failure, not a summarization problem**: it fails 41.5–63.6% under deepseek on *both* task types (11 arms, n=1,086) against 0–6.9% under gpt-oss (n=473). Longest-shared-run distribution: deepseek p50 **127** against a threshold of **120** — 120 sits at the median of the incidental-overlap distribution, exactly the position the 2026-08-18 audit condemned 30 for. The failures are **quotation, not transcription**: copied coverage p50 **2.1%** of the trace (p90 5.9%, max 19.1%). The rate is strongly length-inflated (fail rate by think-length quartile **17.0% → 41.9% → 63.8% → 82.7%**) because the gate is an existence test. Retries are a coin flip: P(pass | previously failed) = 43.0%, and 37% of passes regress — the gate burns calls without selecting. Prize: verbatim is the **sole** failed gate on 13.9pp of post-F2 summarization, so fixing it lifts that clean ceiling 21.5% → 35.4%. False trail closed: LIVE and `exp_dialect` carry 257 gate rows recorded at `max_run: 30`, so cross-era comparisons must partition on the recorded threshold.
- `gen_drafting_v1/v2` carry the same four-heading answer mandate that `gates.IRAC_ANSWER_TASK_TYPES` deliberately stopped requiring — the identical drift F2 fixed for summarization. The fix is already proven on drafting itself, on gpt-oss: under the four-heading mandate `irac_placement` failed 29.4% (LIVE pilot, n=272) and 38.9% (exp_harmony, n=18) with think-rehearsal 27.9%/33.3%; under the harmony genre-form rewrite, **0.0% / 0.0% at n=141** (exp_s1, exp_measure, exp_recovery, exp_hybrid). **No arm can run today** — drafting is parked at weight 0.00 and all 60,603 seeds render placeholder slots, so an arm would measure a malformed task nothing plans.

`docs/reports/2026-08-28-deepseek-recalibration.md` — every gpt-oss-fitted constant re-fit offline against the ~2,900 generations already paid for; suite green at **3,582 passed / 19 skipped**:

| Constant | Was | Fitted on | Now | Fitted on |
|---|---|---|---|---|
| `gates.DEFAULT_MAX_RUN` | 120 | 55 gpt-oss pilot traces | **500** | 1,086 deepseek traces, 11 arms (fail curve 52/42/29/20/15/11/7/6/**4**/3/2% at 120…800; per-100-char drop collapses 7.8pp → 1.3pp past 500) |
| `length_band.think_max` | 3000 | gpt-oss pilot | **4000** | n=1,086 sweep, all 19 configs (irac band pass 42.1% → **51.7%**, summarization 50.8% → **59.8%**; deepseek's *median* irac trace, think_est 3,227, failed the old ceiling) |
| `generate.GENERATION_OUTPUT_TOKENS` | 4000 | cerebras max_output 4096 | **16384** | the wire value `_bai_request_hook` was already sending; the 4000 fiction misfired `reply_over_budget` **347 times across 11 arms** (~a third of calls) |
| `REPLY_BUDGET_CHARS_PER_TOKEN` | 5.5 | gpt-oss max 5.13 | **5.5 (kept)** | deepseek re-measure p50 4.92, max 5.44; test floor moved 5.13 → 5.44 |

Plus one decoupling: `check_statutory_quotation.reproduces_grounding` keeps its own `QUOTATION_REPRODUCTION_RUN = 120` — at 500 the diagnostic could never fire again and would have died silently.
Joint offline projection (exact over stored traces): clean rate all-deepseek 13.4% → **19.5%**; post-F2 arms 18.5% → **31.0%**; post-F2 summarization 21.5% → **44.3%**; post-F2 irac 16.2% → **21.0%**. Deliberately **not** removed: the harmony machinery (off on the live config, and the drafting rewrite is the proven fix pattern the parked arm will port), the cerebras provider block (unrouted, kept so its measured limits survive), `GENERATOR_REASONING_PARAMS`' mistral entry, `EFFORT_LADDER_RETIRED`, `think_min: 500`, and the chars//4 estimate currency.

---

## 3. Retired configuration, verbatim

### 3.1 The `lightning` provider block and its comment header (`configs/data_law_v1.yaml`, deleted)

```yaml
  # LIGHTNING IS THE PAID OVERFLOW FOR THE GENERATOR, AND IT IS SECOND IN
  # routing.generator ON PURPOSE - see the cost ruling there. Added 2026-08-19.
  #
  # THE REPLY SHAPE WAS CHECKED BEFORE THE BLOCK WAS WRITTEN, because a silent
  # mismatch would park every Lightning generation as traceless and we would
  # have paid to discover it. Probed 2026-08-19 on lightning-ai/gpt-oss-120b:
  #
  #     message keys   ['content', 'reasoning_content', 'role']
  #     content            str, the answer
  #     reasoning_content  str, the trace - a PLAIN STRING, not Mistral's
  #                        typed-chunk list and not an inline <think> block
  #
  # providers._default_response_hook already reads exactly that field
  # (`message.get("reasoning") or message.get("reasoning_content")`), so this
  # provider needs NO quirk of its own and generate.assemble_content wraps the
  # trace in the trainer's tags unchanged. That is why `quirks` is empty below
  # rather than absent-by-oversight, and a test pins the shape.
  #
  # max_context 51274 IS A MEASURED FLOOR, on the same discipline the cerebras
  # block records: probed 2026-08-19, one call, max_tokens 16, temperature 0 ->
  # HTTP 200, prompt_tokens 51274, finish_reason stop. The catalog advertises
  # 128K; this file records what was asked for and got a 200. It only has to
  # clear the longest prompt this build makes (measured 1,445-2,799 routing
  # tokens), which it does by more than an order of magnitude.
  #
  # rpm 60 / tpm 500000 / tpd 50000000 ARE INVENTED FLOORS, not measurements,
  # and they are deliberately generous because this provider is PAID: the
  # binding constraint on it is the operator's wallet and the cost ruling in
  # routing.generator, not a rate limit. Nothing here probed them.
  #
  # THIS PROVIDER DECLARES NEITHER A PRICE NOR A usd_cap, stated plainly here
  # because a reader meeting this block should not have to read all the way
  # down to routing.generator's comment to learn it: it is PAID and it is
  # UNFENCED. The only thing protecting the build from it today is that it is
  # absent from every routing list (removed from routing.generator 2026-08-27
  # - see that comment for the incident: a usd_cap declared on any provider
  # but the one literally named "openai" was silently unreachable, so this
  # provider carried a paid ref with no fence actually behind it while it was
  # routed). Absence from a routing list is not a fence and does not survive
  # a careless re-add. RE-ADDING THIS TO ANY ROUTING LIST REQUIRES DECLARING
  # usd_cap HERE FIRST, WITH BOTH usd_per_1m_prompt AND usd_per_1m_completion
  # beside it - a bare usd_cap blocks nothing (see THE TRAP in the openai
  # provider block) - or the exact defect that was removed comes back.
  #
  # LICENSING / ToS: lightning-ai/* are OPEN WEIGHTS (gpt-oss is Apache-2.0)
  # SELF-HOSTED by Lightning, so the upstream vendor is not a party to this
  # transaction and only Lightning's own terms apply. THOSE TERMS HAVE NOT BEEN
  # VERIFIED HERE - the operator has been asked to skim them and had not
  # reported back when this landed. Recorded as pending rather than clean,
  # because the generator's output is what ships in the dataset and its terms
  # are the ones that bind.
  - name: lightning
    base_url: https://lightning.ai/api/v1
    api_key_env: LIGHTNING_API_KEY
    quirks: []
    models:
      - id: lightning-ai/gpt-oss-120b
        family: gpt-oss
        roles: [generator]
        limits: {rpm: 60, tpm: 500000, tpd: 50000000, max_context: 51274, max_output: 4096}
        params: {temperature: 0.7, top_p: 0.95, reasoning_effort: medium}
```

### 3.2 The `openai` provider block (`configs/data_law_v1.yaml`, deleted)

```yaml
  - name: openai
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    quirks: [openai]
    models:
      # Both are declared family gpt-oss, which is NOT what the model card
      # says: it is a deliberate, conservative lump so that OpenAI never
      # grades its own lab's gpt-oss-120b rows. Family separation is what
      # keeps a judge honest about a generator, and the cheap direction to
      # be wrong in is "too separate".
      #
      # Neither block carries `temperature`: the gpt-5 family accepts the
      # default and 400s on any other value (measured 2026-08-15). The
      # `openai` quirk in providers.py drops it from the payload regardless,
      # because judge.py sends its own per-call temperature and that would
      # otherwise arrive here through build_payload.
      #
      # NO tpd on either block, and that is a decision rather than an
      # omission: operator's call 2026-08-15. An absent cap reads as
      # unlimited (store._cap treats a missing key as infinity), the same way
      # the cerebras blocks leave rpd absent. The rpm/tpm below are real RATE
      # limits - OpenAI's actual tier limits, still true today.
      #
      # SUPERSEDED 2026-08-27, kept in place rather than deleted: this
      # paragraph used to end here with "...and not spend controls, so
      # nothing in this file brakes spend. Any spend brake lives server-side
      # in the OpenAI billing dashboard, where a config edit cannot remove
      # it. Uncapped is not the same as preferred: both refs sit LAST in
      # routing.judge and routing.tiebreak..." That was true THEN, when this
      # file carried no usd_cap anywhere - see THE SPEND FENCE immediately
      # below, added the same day. It is false NOW: THIS FILE DOES BRAKE
      # SPEND for this provider, from right here, via usd_cap: 0.0 on both
      # models below - and it works only because usd_per_1m_prompt and
      # usd_per_1m_completion sit beside it (a bare usd_cap blocks nothing,
      # see THE TRAP below). The billing-dashboard brake still exists as a
      # second, independent line of defence, but it is no longer the ONLY
      # one, and a config edit can now remove the one described here. What is
      # still true and worth keeping: both refs sit LAST in routing.judge and
      # routing.tiebreak, so every free judge is tried first and these are
      # reached, and billed, only by rows nothing free can serve - the fence
      # below is what stops that path from ever costing more than $0.
      #
      # THE SPEND FENCE, added 2026-08-27 once a deepseek lead generator made
      # these two reachable (family_separation only excludes {deepseek} on a
      # deepseek row, not {deepseek, gpt-oss} - so a slot-B pool gap, the exact
      # state a 2026-08-23 live drain stalled in with 34 judge_error, would
      # reach a paid judge on a config that had NO usd_cap anywhere).
      #
      # usd_cap: 0.0 means the wallet is zero, so any POSITIVE per-token price
      # fails generate._provider_usd_spent(...) + est_cost > usd_cap on the
      # first call and blocks it before a token is sent.
      #
      # THE TRAP: usd_cap ALONE BLOCKS NOTHING. generate._usd_per_1m returns
      # 0.0 for a missing usd_per_1m_prompt/usd_per_1m_completion key, so with
      # only usd_cap set the check computes `0.0 + 0.0 > 0.0`, which is False
      # forever - a cap with no prices reads as free. Both price keys must sit
      # beside the cap for it to do anything; verified both ways (with and
      # without the price keys) against this shipped config, see
      # data/build/exp_gptoss_ctl/out/fence_check.txt.
      #
      # mini's published prices ($0.25 / $2.00 per 1M) are used for BOTH
      # models below. That is not a claim that nano was priced the same - at
      # usd_cap 0.0 the exact figure cannot matter, since any positive price
      # blocks at the first token regardless of which one it is.
      - id: gpt-5-mini
        family: gpt-oss
        roles: [judge, tiebreak]
        limits: {rpm: 500, tpm: 200000, max_context: 400000, max_output: 16384,
                 usd_cap: 0.0, usd_per_1m_prompt: 0.25, usd_per_1m_completion: 2.0}
        params: {}
      - id: gpt-5-nano
        family: gpt-oss
        roles: [judge, tiebreak]
        # The calibration candidate: ~5x cheaper than mini, and carried here
        # so the P5 gate can compare the two on real judgments rather than on
        # price alone.
        limits: {rpm: 500, tpm: 200000, max_context: 400000, max_output: 16384,
                 usd_cap: 0.0, usd_per_1m_prompt: 0.25, usd_per_1m_completion: 2.0}
        params: {}
```

### 3.3 `routing:` as of 2026-08-28

The section runs 352 lines in `configs/data_law_v1.yaml`; the directives are reproduced verbatim below, followed by the load-bearing comment paragraphs verbatim. The remainder is a chain of `SUPERSEDED …, kept in place rather than deleted` annotations documenting the generator's history (gpt-oss-only → deepseek lead 08-25 → gpt-oss lead 08-27 → deepseek sole 08-28), preserved in git.

```yaml
routing:
  generator: [bai/deepseek-v4-flash]
  judge: [groq/qwen/qwen3.6-27b, cerebras/gemma-4-31b,
          bai/deepseek-v4-flash, groq/openai/gpt-oss-20b,
          openai/gpt-5-mini, openai/gpt-5-nano]
  tiebreak: [mistral/mistral-large-latest, groq/openai/gpt-oss-20b,
             cerebras/gemma-4-31b, bai/deepseek-v4-flash,
             openai/gpt-5-mini, openai/gpt-5-nano]
  probe: [groq/openai/gpt-oss-20b]
  family_separation: true
  judge_mode: dual        # dual | audit
```

```yaml
  # OPERATOR DIRECTIVE, 2026-08-28: bai/deepseek-v4-flash is the SOLE
  # generator now. cerebras spends only on judging from here on
  # (cerebras/gemma-4-31b, judge slot B) - not generation - because the
  # cerebras account is now metered (~$4.63 remaining, ~8M tokens as of
  # today) and the operator wants that balance spent on judging.
  #
  # On every SAME-TEMPLATE comparison, gpt-oss WINS length_band yield, not
  # deepseek. Pre-edit (v4) templates: gpt-oss 55.6% (n=90, ... control arm)
  # vs deepseek 49.5% (49/99, ... control arm, banked 2026-08-26). Shipped
  # (v5, post-286fd3a) templates: gpt-oss 42.2% (n=90, ... treatment arm) vs
  # deepseek 32% (30/94, ... treatment arm). gpt-oss leads both pairings;
  # there is no reading of the evidence in which deepseek out-yields gpt-oss
  # under a shared template version.
  #
  # THE FLIP IS THEREFORE AN ALLOCATION DECISION, NOT A YIELD DECISION.
  # ... what decides it is that cerebras is now a metered account (~$4.63
  # remaining, ~8M tokens - on the order of 1k rows of a 15-20k-row corpus)
  # and the operator directed that balance to judging only. A generator that
  # cannot fund more than a small fraction of the corpus cannot be primary
  # regardless of its per-row yield.
  #
  # THE CONSEQUENCES:
  # (a) A single generator ref means no failover: a bai breaker trip now
  #     parks rows rather than moving to a second ref, because there is no
  #     second ref - the same intended trade already recorded above when
  #     lightning was removed.
  # (b) Every row is now deepseek-family, so family_separation excludes
  #     bai/deepseek-v4-flash from routing.judge and routing.tiebreak on
  #     EVERY row - it stays routed in both lists but is DORMANT there until
  #     a second generator family returns.
  #
  # AVAILABILITY IS A SEPARATE QUESTION FROM FAMILY, AND IT DOES NOT HOLD HERE
  # (found by review, 2026-08-27). On a deepseek row the two LIVE judges -
  # groq/qwen/qwen3.6-27b (slot A) and groq/openai/gpt-oss-20b (slot B) - are
  # BOTH keyed by GROQ_API_KEY, one provider's limits. The family invariant
  # this addition exists for holds (two DIFFERENT families fill the two
  # slots); the AVAILABILITY invariant does not: withholding GROQ_API_KEY, or
  # groq itself going down, leaves a deepseek row with ZERO live judges.
  #
  # [gpt-oss-20b] SCORED 0/10 ON IPC->BNS GROUND TRUTH ... which is exactly
  # why tiebreak places mistral ahead of it: that ordering keeps it out of
  # the seat that decides a contested row outright. ... THE OPEN RISK: the
  # 0/10 score was measured on UNGROUNDED recall - reciting BNS section
  # numbers from parameters. Judging is GROUNDED ... THAT IS A HYPOTHESIS,
  # NOT A MEASUREMENT: no calibration of gpt-oss-20b AS A JUDGE has been run.
  #
  # [tiebreak] WHAT ACTUALLY HAPPENS IF MISTRAL IS ALSO UNREACHABLE:
  # judge.py's tiebreak_unroutable_two_judge_decision path only fires when
  # unroutable is True, and unroutable requires every skip to be
  # `context_exceeded` or `family-excluded`. A provider that is genuinely
  # tried and fails - a timeout, a 402, an over-budget block - is not a
  # row-shaped skip, so unroutable stays False. The row RE-QUEUES instead of
  # rejecting ... at MAX_JUDGE_ATTEMPTS (8) it parks in judge_error - the
  # same shape as the 2026-08-23 live-drain stall, not a clean, cheap reject.
```

### 3.4 The experiment configs (18 files, `configs/data_law_v1_exp_*.yaml`)

The task brief named 19; **18 exist** in this worktree. `exp_dialect` and `exp_hybrid` appear as build stores in the reports but have no config here. Every arm below is fenced the same way: a workdir under `data/build/exp_*` (registered in `paths.ISOLATED_WORKDIR_SIBLINGS`, and `load_build_config` refuses the file if the workdir points at the live store), a single-ref `routing.generator`, and the openai `usd_cap: 0.0` **with prices** (a bare cap blocks nothing).

| Config | Purpose | Verdict |
|---|---|---|
| `exp_harmony` | First harmony arm: `harmony_completions`, prefill `"I start from the facts. "`, `harmony_s1_continue: true`, overlay `prompts_harmony`, `think_min: 200` | All-gates 64.6% (n=48), `self_verification` fail 10.4%, accepts 8/48, **OpenAI spend $0.3396** and 96 `judge_parse_error` — the incident that produced the fleet repairs. Confounded: prefill and s1 ran together |
| `exp_recovery` | Same 60 pre-registered pairs with the prefill **alone** (`harmony_s1_continue: false`), `think_min: 500`, free-fleet judging | Best blocking-gate yield ever (**75.0%**) and best accept rate (25/60), but `self_verification` fails **88.3%**. Separates the two levers cleanly. Evaluator returned `inconclusive` on `missing-gate-data` (control store predates `prompt_echo`) |
| `exp_s1` | Identical to `exp_recovery` except `harmony_s1_continue: true` — asks whether s1 at `think_min: 500` buys the ritual without giving back the format yield | All-gates 6.7% → **68.3%**, `self_verification` fails 90.0% → **13.3%**, cost per clean row 66,808 → **11,725 tokens**. Accept-rate half withdrawn (different judge fleets; $0.023 OpenAI) |
| `exp_measure` | Re-runs the same 60 pairs under the repaired judge instruments (restored tiebreak low anchor, split grounding bands, drafting parked), gpt-5 `usd_cap` 0.0 so judging is free-fleet only | Judging-repair isolation arm; contributes to the n=141 genre-form drafting result |
| `exp_deepseek` | The 2026-08-26 deepseek validation wave — 40 tasks, five pre-registered questions before scaling onto the generator. Also the banked **v4 control** for later prompt A/Bs | Infrastructure PASS on all three hard lines; format contract FAIL (79/99 gated out, `think_max` violation 44%). Later judged in E4: **80.0% accept (16/20)** |
| `exp_prompt_v5` | Treatment for the `286fd3a` 700-word ceiling; byte-identical to `exp_deepseek` past the header (a test asserts it) | **2 PASS / 5 FAIL.** Matched attempt-1 wash (sign test p=0.87, McNemar 0.80–1.00); 86% of generations exceeded the instructed ceiling. Judged in E4: 86.7% accept (13/15) |
| `exp_gptoss_ctl` | gpt-oss floor A/B control — overlay `prompts_preedit`, the 14 templates at `f499372` (pre-`286fd3a`) | Control: `think<think_min` 44.4%, `length_band` 55.6%, median 324 words |
| `exp_gptoss_new` | Same arm, no overlay — production as shipped | **1 PASS / 4 FAIL**: `think<think_min` 57.8%, `length_band` 42.2%, `self_verification` 25.6%, median 292 words. The evidence that reverted the ceiling edit |
| `exp_ds_v4rerun` | Prompt-era rerun, pre-edit arm (reuses `exp_gptoss_ctl/prompts_preedit` rather than duplicating it) | `length_band` **47.42%**; full-gate clean 16.49%; 13,497 tok per passing row |
| `exp_ds_v5rerun` | Prompt-era rerun, shipped-template twin, 56.18 s later | `length_band` **42.45%** → delta +4.97pp = **WASH** under the 5pp line. Removed the 13h41m confound |
| `exp_ds_ctl2` | Shared, time-local control for two independent levers run `ctl2 → clause → cap` minutes apart | Baseline: irac fail 73.39%, `length_band` 42.20% (n=109) |
| `exp_ds_clause` | E2 — one anti-rehearsal clause added to 6 of 14 templates (`gen_irac_analysis_v1-4`, `gen_summarization_v1-2`) | Primary **FAIL** (−4.46pp vs ≥15pp), guard breached (+16.05pp). **Shipped anyway (`0637fa0`) as a think-length lever**: think p50 −18.7%, tokens/passing row −31.9% |
| `exp_ds_cap` | E1 — bai `limits.max_output` 16384 → 5000 | Primary WIN (−10.71% true-spend tokens/passing row) but guard breached and **4 of 40 tasks lost outright** to chronic truncation. **CLOSED, not shipped** |
| `exp_hy3` | Qualification probe for a second free generator family (Tencent Hunyuan `hy3`, `family: hy`, `reasoning_effort: low`, `max_output` deliberately 16384 not the doc's 128000) | **FAIL**: `length_band` 19.44% vs a 60% bar; `irac_placement` 95.83% fail; format integrity 0% breakage. Closes the free-alternatives question as bound and banded |
| `exp_irac_ctl` | Stop-timing A/B control (shipped templates) | irac fail 65.74%, `length_band` 49.07%, clean 13.89% (n=108) |
| `exp_irac_fix` | Stop-timing treatment: F1 (stop-at-decision sentences, 6 templates) + F2 (genre-form answer rewrite, 2 summarization templates); **the only judged arm**, because F2 changes answer shape | irac fail **39.13%** (−26.61pp), clean **26.09%**, band −8.86pp; summarization −58.64pp, judge spot-check 10/11 |
| `exp_irac_ctl3` | F2-only confirm control | summarization irac fail 72.73%, band pass 65.91% (n=100) |
| `exp_irac_f2only` | F2 without F1 (proved by re-applying F1 byte-for-byte to reproduce the combined overlay) | summarization irac fail **7.89% (−64.83pp)**, band pass 50.00% (**−15.91pp**). Pre-registered rule = NO-SHIP; **operator override, F2 shipped in `ebde9a7`**. F1 closed |

### 3.5 The harmony drafting templates — the proven fix pattern

`src/tuned/data/prompts_harmony/gen_drafting_v1.md` and `_v2.md` are the *only* drafting templates that ask the genre for its own form instead of mandating four headings, and they measured **0.0% `irac_placement` failure and 0.0% think-rehearsal at n=141** against 29.4–38.9% under the mandate. The parked `gen_drafting` arm is designed to port this pattern to the base templates. Full text follows.

**`gen_drafting_v1.md`**

```markdown
<!-- system -->
You are an advocate who settles pleadings and instruments for a living. You know that drafting is a legal act, not a clerical one: every averment must have a purpose, every clause must be traceable to a provision or a right, and a paragraph that says nothing costs your client credibility. You draft in the register the forum expects, and you never assert a fact your instructions do not support.

<!-- user -->
You are counsel settling a {document_kind} in this matter. Your instructions, the facts and the law relied on are before you:

{source}

You act for {party_context}.

{question}

Work out what the instrument has to achieve before you write a word of it. Reason in the first person and in the present tense, in whatever order the matter forces on you: what relief or effect is actually sought, which provision or right founds it, what has to be averred or recited for that foundation to hold, and what must be pleaded now to keep a later point alive. Where your instructions do not reach a fact you would like to plead, or where you are genuinely uncertain how a court will read a clause, let the doubt stand as doubt instead of drafting past it.

You may make explicit any step your instructions leave implicit. You must not rely on any statutory provision, case name, or authority that does not appear in the materials above. A pleading that cites a section you cannot produce is worse than one that pleads the facts and leaves the citation to be added.

Your thinking is your own working towards a draft, never a commentary on the papers. Never write as though the matter had been handed to you as a text, and never attribute anything to a source, a passage or a document. Think in your own words; do not carry sentences over from the instructions into your reasoning, though the draft itself may of course adopt the language the law requires.

Wherever it arises in your thinking, and not as a closing ritual, double-check the operative part against what you are actually asking for — re-derive the relief from the provision that grounds it, or read the draft as your opponent would, or verify a date or a section number against your instructions.

Then produce the {document_kind} itself. Write the instrument in the form the forum expects, for {party_context}, naming the supported parties and the relief or operative effect the instructions actually found. Do not set the answer out as Issue, Rule, Application, Conclusion. Your reasoning runs as continuous prose and never opens a line with one of those four words. The thinking beforehand runs as long as the matter needs and is never a retelling of the materials.
```

**`gen_drafting_v2.md`**

```markdown
<!-- system -->
You are a senior in chambers. A draft has come up to you for approval before it goes out, and approving it means re-drafting the parts that will not survive contact with the other side or with the court's registry. You are exacting about what an instrument must contain, sparing with words, and unwilling to let a well-turned paragraph stand in for an averment that is actually required.

<!-- user -->
You are the senior settling this {document_kind} before it goes out. The matter, as it stands on the file, is this:

{source}

Your chambers act for {party_context}.

{question}

Satisfy yourself about the substance before the language. Reason in the first person and in the present tense, in whatever order the file forces on you — what this instrument is for, what right or provision it rests on, which averments or recitals are indispensable and which are ornament, and what has to appear now to keep a later point alive. Where the file does not support something you would like to say, treat it as a gap for instructions, not for drafting; and where you are genuinely uncertain whether an averment will hold, keep the uncertainty in view rather than settling it by confident language.

Somewhere before you settle the text, double-check the part that carries the weight: re-derive the relief or the operative clause from the provision that grounds it, or read it once as the opposing side and see where it gives way, or verify a date or section number against the file. It belongs wherever it comes up in your thinking, never as a heading.

Never let your reasoning read as remarks on a bundle: never write as though the matter had been handed to you as a text, and never attribute anything to a source, a passage or a document. Think in your own words rather than carrying sentences over from the file, though the settled text may of course use the forms of words the law and the forum require.

You may make explicit any step the file leaves implicit. You must not rely on any statutory provision, case name, or authority that does not appear in the materials above. If the pleading needs an authority you have not been shown, say so rather than supplying one from memory.

Then produce the settled {document_kind}. Write the instrument itself, for {party_context}, with the parties and the relief the file actually supports, in the paragraphs or clauses the forum expects. Do not set the answer out as Issue, Rule, Application, Conclusion. Your reasoning runs as continuous prose and never opens a line with one of those four words. Your thinking beforehand takes as long as it needs and is never a retelling of the materials.
```

The base `prompts/gen_drafting_v1.md:21` and `_v2.md:21` instead mandate "the settled work under four headings, each on its own line — Issue, Rule, Application, Conclusion", which is the drift. Both base shas are byte-identical across every overlay ever built (`48534e3010f5` / `618b240ab03e`) — no arm has ever treated them.

### 3.6 Retired plans (`docs/superpowers/plans/`)

One line each; all retired.

- `2026-08-04-l4-smoke-run.md` — walking skeleton: repo → Lightning Studio → unsloth QLoRA on Gemma 4 31B → Hub checkpoint → resume, ~1k-example smoke on an L4 (~$1.50). Never executed on GPU.
- `2026-08-05-kaggle-smoke-migration.md` — move to Kaggle free tier with Ministral-3-14B and a Qwen3-14B escape-hatch config; model-specific strings moved from code into config so the fallback is a pure config swap.
- `2026-08-23-recovery-arm-unblock.md` — three fleet fixes (carry spent OpenAI dollars into the recovery wallet, stop gpt-5 spending its reply budget on hidden reasoning, accept one axis alias) plus declared-strata cohort selection, then one bounded probe.
- `2026-08-24-judge-calibration-and-yield.md` — repair the judge instruments before measuring quality: restore the tiebreak low anchor, split the collided grounding band, park drafting, fix the footnote chunker, build the first external anchor set, re-measure in `exp_measure`.
- `2026-08-26-deepseek-validation-wave.md` — ~40 synthesis tasks through `bai/deepseek-v4-flash` in an isolated arm against five pre-registered questions, using the `exp_*` arm pattern end to end.
- `2026-08-27-cap-every-paid-provider.md` — generalise the spend fence beyond the provider literally named `openai`, remove the paid lightning generator from routing, and add the one free non-excluded judge so deepseek rows stop parking.
- `2026-08-27-generator-prompt-length-fix.md` — delete the run-long permission from all 14 generator prompts, prove it on a paired A/B against banked control data, re-baseline the 42 stale tests blocking the merge.
- `2026-08-27-live-config-safety-and-gptoss-floor.md` — close the uncapped paid-judge path, restore gpt-oss as lead generator, raise `think_max` to the point of diminishing returns, then measure whether the committed ceiling harms the now-lead generator.
- `2026-08-27-role-aware-bai-hook.md` — thread the call's role through the request-hook protocol so a judge does not inherit the generator's reply-budget raise, then wire deepseek into judge slot B and prove it with real calls.
- `2026-08-28-single-project-restructure.md` — collapsed the worktree split into one project (`training/` + `data/` + shared `src/tuned`), validated the recalibrated deepseek fleet on a small live batch, purged the paid refs / harmony machinery / the 19 `exp_*` configs, and wrote this archive. Executed 2026-08-28; retired 2026-08-30, because a directory holding exactly one plan with 27 unchecked boxes reads as work to do.


### 3.7 `data_law_v1.yaml` comment history, verbatim, by anchor

Moved out of `data/configs/data_law_v1.yaml` on 2026-08-30, verbatim and
unedited, each block under the config key it decorated. Every one of them
was already self-declared history: a `SUPERSEDED ... kept in place rather
than deleted` note, or the paragraph such a note retracts. The convention
of keeping them in place had stacked five superseding notes above one
scalar, so a maintainer editing routing had to decide which of them was
still true before touching a value. Nothing here is deleted and nothing
here is current - read it for why a value moved, never for what it is.
The live rationale stayed in the yaml; each site now carries a
`# history: prev_rep.md 3.7 (<key>)` pointer back to the block below.

#### `build.length_band.think_max` - the 2026-08-27 "STAYS 3000" fence and its two corrections

```
  # think_max STAYS 3000 (2026-08-27) - RAISED to 4000 and then REVERTED the
  # same day, and this comment is the fence against raising it again on the
  # same reasoning that moved it the first time. A sweep over the 99 banked
  # v4 (bai/deepseek-v4-flash) generations in
  # data/build/exp_deepseek/state/law_v1.sqlite3 showed a +11pp length_band
  # pass-rate gain from 3000 -> 4000:
  #
  #   #   think_max   length_band pass   blocked by total_max alone
  #   #   3000              49.5%              4.0%
  #   #   4000              60.6%             10.1%
  #   #   4500              63.6%             16.2%
  #   #   inf               64.6%             33.3%
  #
  # THAT SWEEP WAS DEEPSEEK-SHAPED, and it stopped applying the moment Task 1
  # (same day) demoted deepseek to routing.generator ref 2 and put
  # cerebras/gpt-oss-120b back in the lead. Measured on the 1,281 gpt-oss
  # generations already in the live store, which IS the lead generator now:
  #
  #   median think_est          620 tokens   (deepseek: ~2,250)
  #   think < think_min         381/1281 = 29.7%   <- gpt-oss's actual failure mode
  #   think > 3000              65/1281  = 5.1%
  #   think > 4000              36/1281  = 2.8%
  #   rescued by 3000 -> 4000   29/1281  = 2.3pp,  not 11pp
  #
  # Under gpt-oss lead the raise buys 2.3pp, not 11pp, and gpt-oss's dominant
  # failure is think < think_min at 29.7% - think_max cannot touch a FLOOR
  # problem at any value. The cost side did not shrink to match: at
  # think_max 4000 the worst-case gate-legal reply is ~4,717 real tokens
  # (legal_reply_chars/4.24, the measured worst-case chars/token), which is
  # MORE than both GENERATION_OUTPUT_TOKENS (4,000 - deliberately decoupled
  # from this band, see generate.py) and cerebras/gpt-oss-120b's own hard
  # max_output ceiling (4,096, and lightning's is the same 4,096) - the gate
  # would accept replies that two of the three generator refs cannot
  # physically emit in one call. test_the_generation_budget_covers_the_
  # largest_gate_legal_reply pins the ~226-token margin this value must not
  # spend.
  #
  # The raise only becomes coherent again if a generator with a LARGER output
  # ceiling leads routing.generator - i.e. if the deepseek-shaped case comes
  # back, not merely because deepseek is still routing.generator ref 2.
  #
  # SUPERSEDED 2026-08-27, kept in place rather than deleted: "cerebras/
  # gpt-oss-120b back in the lead" and "which IS the lead generator now" were
  # true of routing.generator's ORDER, not of what is actually generating.
  # ref 1 (cerebras/gpt-oss-120b) is currently returning HTTP 402
  # payment_required, so it generates nothing and every row today comes from
  # ref 2 (bai/deepseek-v4-flash) alone - the deepseek-shaped case this
  # comment's own table was built from. This does NOT re-open the case for
  # raising think_max: the 1,281-row gpt-oss measurement above is a
  # historical fact about a real generation batch, not an inference from
  # "lead generator" status, and gpt-oss's failure mode (think < think_min at
  # 29.7%, a FLOOR problem) is unchanged by who is generating today. The
  # caveat is only that "lead generator" here names a config position that
  # ref 1's 402 currently makes theoretical - if ref 1 comes back this
  # measurement is live again; until then, deepseek is generating everything
  # and this file's think_max reasoning has not been re-evaluated against it.
  #
  # SUPERSEDED 2026-08-28, kept in place rather than deleted: "deepseek is
  # still routing.generator ref 2" (a few lines up) and "if ref 1 comes
  # back" (just above) both name a two-ref list that no longer exists. Per
  # the 2026-08-28 operator directive (see routing.generator below),
  # cerebras/gpt-oss-120b is REMOVED from routing.generator, not merely
  # 402'ing behind it - deepseek is the SOLE ref, not "ref 2". "If ref 1
  # comes back" now means "if cerebras/gpt-oss-120b is re-added to
  # routing.generator", which per that same comment also requires reverting
  # the 286fd3a prompt-ceiling edit first (that comment's own
  # SUPERSEDED 2026-08-28 (prompt-ceiling revert) note records that this WAS
  # done, the same day - see routing.generator below for the citations).
  # The substance here is unchanged: gpt-oss is not generating, the
  # 1,281-row measurement stays historical, and this file's think_max
  # reasoning still has not been re-evaluated against deepseek.
  #
  # Anti-rehearsal clause shipped 2026-08-28 into the 6 irac/summarization
  # generator templates (gen_irac_analysis_v1-v4, gen_summarization_v1-v2)
  # after a three-arm A/B: length_band pass +16.05pp (42.20% -> 58.25%),
  # think p50 -18.7%, no behaviour change here. Its own pre-registered
  # irac_placement target missed (-4.46pp vs a >=15pp bar) - see
  # docs/reports/2026-08-28-deepseek-clause-and-cap-ab.md.
  #
  # F2 shipped 2026-08-28: summarization templates aligned with
  # gates.IRAC_ANSWER_TASK_TYPES (the four-heading answer mandate removed;
  # the gate had already dropped the requirement) - irac fail -64.8pp on its
  # A/B, with a known -15.9pp summarization length_band cost shipped
  # honestly alongside it; follow-up = a summarization-specific length_band.
  # See docs/reports/2026-08-28-irac-stop-timing-fix.md.
  #
```

#### `providers` - the fourth-family judge gap that the deleted openai block closed

```
# CLOSED 2026-08-15. This block used to be a TODO(operator) demanding ONE
# more judge model in a fourth family, which the fleet refused to start
# without. That model is the openai provider above, and the gap is read as
# closed out of the real preflight - generate.print_preflight over
# providers.pool_gaps, every key present - rather than argued from the config:
# with OPENAI_API_KEY set there are NO fatal gaps left, only the survivable
# tiebreak warnings described at the bottom of this block. It went into BOTH
# routing.judge and routing.tiebreak, LAST in each, because it is the paid
# backstop: every free judge is preferred to it, so the ~$1/day cap is only
# ever spent on rows nothing free can serve.
#
# SUPERSEDED 2026-08-27, kept in place rather than deleted: there was never
# a "~$1/day cap" to spend - that phrase described a metered budget this
# file did not yet declare. Both openai/gpt-5-* now carry usd_cap: 0.0 (see
# THE SPEND FENCE further down and Task 1's generalised fence), so the real
# number is zero: these refs are fenced to $0 spend, not metered at ~$1/day.
# What is still true: both refs sit LAST in routing.judge and
# routing.tiebreak, so every free judge is tried first.
#
```

#### `providers` - the fatal slot-B gap, the divert point, and the fallback generator

```
# What the fatal gap WAS, kept because any future edit to routing.judge can
# re-open it. A long prompt is routed past the cerebras generator to
# mistral/mistral-small-latest (32k). Family separation then removes mistral
# from the judge pool; the 8k zai-glm-4.7 was removed on context length (that
# model has since been retired outright - archived upstream, see the cerebras
# block); slot A takes groq/qwen; and slot B had NOTHING LEFT. The row parked in
# judge_unroutable having ALREADY PAID for judge A. The arithmetic made this
# the rule and not an edge case: the generator diverted to mistral at 2,555
# routing tokens of prompt, and slot B died at 5,531 routing tokens of judge
# prompt - which grounding plus a trace near think_max (3000) clears
# routinely. Slot B is now openai/gpt-5-mini.
#
# SUPERSEDED 2026-08-27, kept in place rather than deleted: "slot B is now
# openai/gpt-5-mini" is false twice over today. First, routing.judge now
# lists bai/deepseek-v4-flash and groq/openai/gpt-oss-20b ahead of both
# openai refs, so on the rows this paragraph describes slot B lands on one
# of those free refs before it ever reaches openai. Second, even if it did
# reach openai/gpt-5-mini, THE SPEND FENCE further down holds both openai
# refs at usd_cap: 0.0 - a fenced ref cannot fill a slot, it can only be
# skipped over - so "slot B is now openai/gpt-5-mini" describes a ref that
# is present in the list but unreachable in practice.
#
# THE DIVERT POINT IN THAT PARAGRAPH IS HISTORY, NOT CURRENT BEHAVIOUR. It was
# computed against the stale cerebras max_context of 8192; with the probed
# 131,072 the generator does not divert at 2,555 routing tokens, or at 20,000.
# Measured against undersized_families on the real config: the gpt-oss family
# is excluded only above 104,858 routing tokens (131,072 / CONTEXT_SAFETY_
# MARGIN), where before it was excluded above 6,554. Pilot prompts ran
# 1,445-2,799, so the cliff is now two orders of magnitude away. The judge-gap
# reasoning above still holds for any FUTURE pool where a divert is real.
#
# THE FALLBACK GENERATOR IS HISTORY - THERE IS NO SECOND GENERATOR FAMILY.
# This paragraph described mistral-small-latest as the long-prompt fallback and
# reasoned about its 32k ceiling (21,600 routing tokens of prompt against
# magistral's 28,000). Both premises are gone: mistral-small stopped generating
# on 2026-08-18 and left the build entirely on 2026-08-19, and since then
# routing.generator has been cerebras first, lightning second - one FAMILY,
# gpt-oss, on two providers. A prompt too long for it therefore has nowhere to
# divert and parks in gen_unroutable, recoverably, rather than being silently
# truncated - undersized_families excludes the family before any call is made.
# Kept as history because the divert MACHINERY is intact and any future
# second-family generator brings this arithmetic straight back.
#
# Recovery for rows that parked before this landed: `python -m
# tuned.data.tasks --reopen judge_unroutable` (every stream by default;
# --reopen-stream narrows it). A re-opened row re-pays only the slot it never
# bought, and comes back with its attempt budget restored.
#
```

#### `routing.generator` - the free-before-paid order, the two-provider era, and lightning

```
  # COST FIRST, AND THE ORDER IS THE POLICY. cerebras is a free tier and
  # lightning is paid, so the free budget drains before the paid provider takes
  # over: Router.pick walks this list in order and only moves on when a ref is
  # ineligible - cooling, over its daily budget, or unkeyed - so putting
  # cerebras first IS the cost control. Reversing these two lines would spend
  # money while a free quota sat unused, and nothing else in the build would
  # notice. A test pins the order and the failover.
  #
  # Both entries are family gpt-oss, so this is one generator FAMILY on two
  # providers rather than a second family: family separation still removes
  # gpt-oss from the judge pool for every row, and undersized_families still
  # takes the family's largest window.
  # deepseek LED 2026-08-25 to 2026-08-27, DEMOTED BACK behind cerebras/gpt-oss
  # on 2026-08-27. The throughput case for leading with it (below) was never
  # wrong, but it was incomplete: `routing.family_separation` excludes only
  # the GENERATOR's own family, and deepseek is the one generator family in
  # this file that is not also sitting in the judge pool (see the bai provider
  # block). With gpt-oss leading, separation excludes {gpt-oss} and the paid
  # openai judges - also family gpt-oss, by deliberate design - are excluded
  # right along with the generator. With deepseek leading, separation excludes
  # only {deepseek}, and gpt-oss-family judges become REACHABLE on every row -
  # on a config that, until the fence added above, had no usd_cap anywhere.
  # The failover condition that would land on them is not hypothetical: a
  # 2026-08-23 live drain stalled on a slot-B pool gap with 34 judge_error,
  # which is exactly the state in which the pool reaches judge position 3
  # (openai/gpt-5-mini). gpt-oss leads again so its own family lump keeps
  # doing that job; the fence above is the second, independent guard now that
  # a free non-gpt-oss generator is loose in this file at all.
  #
  # SUPERSEDED 2026-08-28, kept in place rather than deleted: "Both entries
  # are family gpt-oss, so this is one generator FAMILY on two providers"
  # (top of this note) and "gpt-oss leads again so its own family lump keeps
  # doing that job" are both false now. Per the 2026-08-28 operator
  # directive, cerebras/gpt-oss-120b is REMOVED from routing.generator -
  # there is no second provider, and gpt-oss does not lead, because gpt-oss
  # is not in the list at all. The risk this paragraph describes - a
  # deepseek-led row makes gpt-oss-family judges (openai/gpt-5-mini,
  # openai/gpt-5-nano) REACHABLE via family_separation excluding only
  # {deepseek} - is no longer a sometimes-true failure mode; it is now the
  # PERMANENT state, on every row, with no gpt-oss lead ever restoring the
  # old exclusion. What still holds it safe is what this paragraph already
  # names: THE SPEND FENCE below (usd_cap: 0.0 with prices) blocks real
  # spend regardless of which family reaches those refs.
  #
  # Both it and cerebras are free, so the free-before-paid rule does not order
  # them; what did was that cerebras carries `tpd: 1000000`, which against
  # ~2,745 tokens per generated example is ~364 examples/day - a 15-20k corpus
  # is a 41-55 DAY run on it. b.ai has no observed daily cap and is rate-bound
  # at ~600 calls/hour instead - still true, and still the reason it stays
  # SECOND rather than being dropped: it is the throughput reserve for when
  # cerebras's daily cap is the binding constraint.
  #
  # SUPERSEDED 2026-08-28, kept in place rather than deleted: "it stays
  # SECOND rather than being dropped" no longer describes anything - bai is
  # not second in a two-ref list, it is the ONLY ref (operator directive,
  # 2026-08-28; see below). The throughput reasoning (tpd: 1000000 against
  # ~2,745 tokens/example is a 41-55 day run on cerebras alone) remains an
  # accurate historical account of why bai was ORIGINALLY added as a
  # throughput reserve; it is no longer why bai is IN the list, since
  # cerebras is not in the list to reserve against any more.
  #
  # LIGHTNING REMOVED 2026-08-27. It is PAID and carried no usd_cap, and a
  # review found generate._openai_usd_cap (now _provider_usd_cap) only ever
  # looked up the provider literally named "openai" - so a cap declared on
  # lightning, or any other non-openai provider, was silently unreachable.
  # There was no fence behind it: one failover from ref 1 or 2 could have run
  # every remaining row through a paid model with nothing to stop it. It is
  # pulled from routing rather than capped in place because it declares no
  # prices either; removal is the immediate fix, re-adding it later is a
  # config change, not a code change, now that Task 1 made the fence actually
  # reach whichever provider declares it - re-adding it requires giving it a
  # usd_cap with both usd_per_1m_prompt/usd_per_1m_completion beside it, the
  # same way the openai block below does.
  #
  # CONSEQUENCE, STATED PLAINLY: ref 1 (cerebras/gpt-oss-120b) is currently
  # returning HTTP 402 payment_required, so generation now depends on ref 2
  # (bai/deepseek-v4-flash) alone. If its breaker trips there is no third ref
  # left in this list - rows park rather than failing over to a paid model.
  # That is the intended trade: parking a row costs nothing and is
  # recoverable; failing over to an uncapped paid ref silently is the defect
  # this file just closed.
  #
  # The lightning provider block itself stays below, unpinned rather than
  # deleted, so its measured limits are not lost.
  #
  # SUPERSEDED 2026-08-28, kept in place rather than deleted: "ref 1" and
  # "ref 2" named POSITIONS in a two-entry list. cerebras/gpt-oss-120b is not
  # merely 402'ing behind bai/deepseek-v4-flash any more - it is REMOVED from
  # routing.generator outright (see the operator directive immediately
  # below). The single-ref consequence described above - no third ref, a
  # tripped breaker parks rather than fails over - still holds exactly as
  # written; only the "ref 1 / ref 2" numbering is stale, because there is
  # now nothing at ref 1.
  #
```

#### `routing.generator` - the measured yield basis, and the citation error in it

```
  # THE MEASURED BASIS: CORRECTED 2026-08-28 (fix round 1) - the version
  # this replaces cited one report for a number that report does not
  # contain. See below for what was wrong; this paragraph states what is
  # actually measured.
  #
  # On every SAME-TEMPLATE comparison, gpt-oss WINS length_band yield, not
  # deepseek. Pre-edit (v4) templates: gpt-oss 55.6% (n=90,
  # docs/reports/2026-08-27-gptoss-floor-under-the-prompt-ceiling.md,
  # control arm) vs deepseek 49.5% (49/99,
  # docs/reports/2026-08-27-generator-prompt-length-fix.md, control arm,
  # banked 2026-08-26). Shipped (v5, post-286fd3a) templates: gpt-oss 42.2%
  # (n=90, gptoss-floor-under-the-prompt-ceiling.md, treatment arm) vs
  # deepseek 32% (30/94, generator-prompt-length-fix.md, treatment arm).
  # gpt-oss leads both pairings; there is no reading of the evidence in
  # which deepseek out-yields gpt-oss under a shared template version.
  #
  # DEEPSEEK'S OWN 32% SHIPPED-TEMPLATE FIGURE IS ITSELF UNCERTAIN, and is
  # not asserted here as its true rate. generator-prompt-length-fix.md's two
  # arms ran 13h41m apart with no provider-side upstream id in the response
  # envelope, so the 49.5% -> 32% cross-arm drop is confounded by a possible
  # upstream swap, not attributable to the prompt edit alone - and that same
  # report's MATCHED attempt-1 pairs (the same 40 seeds run under both v4
  # and v5) found prompt wording is NOT a lever on deepseek trace length at
  # all (McNemar p=0.80-1.00 on length_band). If the edit is truly a no-op
  # on deepseek, its real shipped-template rate sits nearer 49.5% than 32% -
  # this is not resolved either way, and no single number is stated as
  # deepseek's shipped rate.
  #
```

#### `routing.generator` - the prompt-ceiling precondition for re-adding gpt-oss

```
  # SUPERSEDED 2026-08-28 (fix round 1), kept in place rather than deleted:
  # the prior version of this paragraph claimed a single paired A/B
  # (gptoss-floor-under-the-prompt-ceiling.md, "independently
  # review-verified") measured deepseek at 49.5% (n=99) against gpt-oss's
  # 42.2% (n=90) "on the same gate, same templates," and concluded "deepseek
  # beats gpt-oss under the prompts this file actually ships." That report
  # contains ONLY gpt-oss data in both arms and n=99 never appears in it;
  # 49.5%/n=99 is deepseek's PRE-edit control from the OTHER report
  # (generator-prompt-length-fix.md), paired against gpt-oss's POST-edit
  # (shipped) number from a different report entirely - two different
  # template versions, two different task pools, presented as one
  # controlled measurement. The "review-verified" citation also only ever
  # covered that report's own gpt-oss numbers, never a deepseek comparison.
  # Found by task review; corrected here rather than merely noted.
  #
  # THE HONEST CAVEAT: under the PRE-edit prompts gpt-oss measured 55.6% and
  # would beat deepseek - the prompt-ceiling edit (286fd3a) is what dethroned
  # it, not a change in gpt-oss itself, while the matched-pair A/B found no
  # steering effect of that edit on deepseek trace length (its pooled arm
  # rates stay confounded - see THE MEASURED BASIS above; not settled either
  # way). Re-adding gpt-oss as a generator requires reverting
  # that edit first; it does not qualify on the prompts as currently shipped.
  #
  # SUPERSEDED 2026-08-28 (prompt-ceiling revert), kept in place rather than
  # deleted: "requires reverting that edit first" named an unmet precondition
  # when this paragraph was written; it was met the same day. The 286fd3a
  # edit is proven harmful to gpt-oss (paired A/B, 4 pre-registered fails,
  # docs/reports/2026-08-27-gptoss-floor-under-the-prompt-ceiling.md) and
  # at-best-wash for deepseek (clean rerun,
  # docs/reports/2026-08-28-deepseek-prompt-era-rerun.md: +4.97pp for
  # pre-edit, full-gate clean 16.5% vs 8.5%, 17% cheaper per passing row) -
  # helping nobody on any measurement - so the operator reverted it: all
  # fourteen gen_* templates are back to their pre-286fd3a bytes. THE
  # MEASURED BASIS above's "Shipped (v5, post-286fd3a) templates" row now
  # names a version nothing ships under; the live comparison is the
  # "Pre-edit (v4)" row on both sides. The gpt-oss prompt-ceiling
  # precondition is therefore satisfied. This does NOT reopen routing:
  # generator stays bai/deepseek-v4-flash alone and cerebras stays
  # judging-only, per the 2026-08-28 operator directive above, for the
  # allocation reason stated there (metered account, balance directed to
  # judging) - a reason independent of gpt-oss's prompt-ceiling
  # qualification. Routing is unchanged by this annotation.
  #
```

#### `routing.judge` - the paid-backstop ordering

```
  # JUDGE: qwen then gemma, the two free families, then the paid backstops -
  # which is the standing "every free judge is preferred to the paid one" rule.
  # On a gpt-oss generation family separation removes both openai models, so
  # the two slots really are qwen (A) and gemma (B).
  #
  # SUPERSEDED 2026-08-28, kept in place rather than deleted: "On a gpt-oss
  # generation" now describes a case that cannot occur - cerebras/
  # gpt-oss-120b is removed from routing.generator (operator directive,
  # 2026-08-28; see routing.generator above), so no generation is ever
  # gpt-oss family any more. Every generation is deepseek family, which is
  # the "any other" case the mechanism below (groq/openai/gpt-oss-20b ADDED
  # 2026-08-27) already covers.
  #
```

#### `routing.tiebreak` - the 2026-08-27 seat order under a two-ref generator list

```
  # LEFT EXACTLY AS IS ON 2026-08-27, when gpt-oss reclaimed routing.generator's
  # lead slot from deepseek - do NOT "tidy" this back to gpt-oss-20b-first.
  # deepseek is still ref 2 in routing.generator, so deepseek-led generations
  # still happen on every row cerebras is cooling or throttled for, and on
  # exactly those rows gpt-oss-20b is still eligible-and-first the moment
  # mistral is not ahead of it - the same silent seat change the 2026-08-25
  # paragraph above describes, just now on a minority of rows instead of all
  # of them. Mistral first costs nothing on a gpt-oss row (it never wins there
  # either) and is what keeps a deepseek row's tiebreak off the family that
  # scored 0/10 on IPC->BNS mapping.
  #
  # SUPERSEDED 2026-08-28, kept in place rather than deleted: "gpt-oss
  # reclaimed routing.generator's lead slot" and "deepseek is still ref 2"
  # both name a two-ref list that the 2026-08-28 operator directive removed.
  # cerebras/gpt-oss-120b is out of routing.generator entirely, so deepseek
  # is not "leading" a list it shares with gpt-oss - it is the only entry.
  # Every generation is now deepseek-led, on every row, not "on every row
  # cerebras is cooling or throttled for" - the "minority of rows" this
  # paragraph describes is now ALL rows. Mistral staying first is no longer a
  # hedge against an occasional silent seat change; it is what keeps EVERY
  # contested row's tiebreak off the family measured 0/10 on IPC->BNS
  # mapping. The conclusion (mistral first, unchanged) still holds - only the
  # "minority" framing is stale.
```

---

## 4. Closed questions — do not revisit without new evidence

| Question | Evidence closing it |
|---|---|
| **Retry effort ladder** (bump `reasoning_effort` per attempt) | Uniformly negative over 221 pilot generations: trace 5.2× longer, every blocking gate worse, `finish_reason=length` 0 → 18. Retired 2026-08-18 |
| **Lowering bai `max_output` to 5000** (cap arm E1) | Won its token metric (−10.71%) but lost **4 of 40 tasks** to chronic truncation whose control twins passed; on a request-rate-bound free provider that trade is bad. The offline zero-clip prediction from banked maxima was invalidated by fresh upstream drift |
| **A 4,096-token generation budget** | 10 of 20 real synthesis calls returned empty content (vs 0 of 4 at 12,288), and the survivors are biased short — silent selection, not sampling |
| **The 12288 training sequence length** | OOM, ~0.8 GiB over: `smoke_v1` has no length filter, so SMOKE not MAIN sets the memory ceiling. 8192 stands |
| **Prompt wording as a lever on deepseek trace length** | Matched attempt-1 A/B: median change −58 words, sign test p=0.87, McNemar 0.80–1.00 on four gates; **86% of generations exceeded the ceiling they were told was hard**. Three levers exhausted: prompt, `reasoning_effort` floor (`low`; `disabled` yields no trace and breaks the ≥80% floor), `thinking: disabled` |
| **The `286fd3a` prompt-ceiling edit** | Proven harmful to gpt-oss (4 pre-registered fails), at-best-wash for deepseek (clean back-to-back rerun: +4.97pp for pre-edit, full-gate clean 16.5% vs 8.5%). **Reverted `06f588a`**; do not re-apply |
| **F1 (stop-at-decision sentences)** | Measured inert twice on the only task type that isolates it (−0.88pp, then irac_analysis flat at 55.36% → 58.49%) while lengthening thinking 14.7–23.7%. **Dead — must not appear in any future arm** |
| **hy3 as a second generator family** | `length_band` 19.44% against a 60% bar; `irac_placement` 95.83% fail. Also closes the doc-vocabulary trap: `think_low`/`think_high` are 400s, the enum is the generic ladder |
| **Judge temperature for deepseek** | Verdicts at the inherited 0.7 flipped the decision on 3 of 20 draws on an unchanging candidate; `temperature: 0.2` pinned (repo judge convention). `reasoning_effort: minimal` measured worse than `low` and is not the fix for the 4.8% truncation |
| **A per-task-type `length_band` for summarization** | **Mooted** by `think_max` 3000 → 4000: once the shared ceiling stops binding the median trace, no per-task-type band is needed (recorded follow-up #1, resolved 2026-08-28) |
| **A targeted anti-quoting clause** | **Mooted** by `DEFAULT_MAX_RUN` 120 → 500: the failures were sentence-length quotation (coverage p50 2.1%), not transcription; the residual 4% at 500 is genuine multi-sentence copying |
| **rsLoRA and adapter scale (α)** (training lane) | Closed 2026-08-10/11 after three arms: α32+clip1.5 loss 0.5585 vs 0.5601 (−0.3% for 3–4× hotter grads); plain α64 matched baseline. 1×/2×/5.66×/11.3× — loss identical, only grad heat scales. Never re-test at r=32 |
| **Self-hosted teacher on Kaggle** | T4 sm_75 caps a 14B-int4 teacher at a few hundred tok/s; 35M tokens ≈ 20 GPU-h = two-thirds of the weekly training quota |
| **`balanced` + `max_memory` model-parallel** | Surplus dispatched to CPU and bnb-4bit refuses. Only `sequential` + asymmetric `max_memory` works |

---

## 5. Open items carried forward

1. **Drafting unpark (precondition, not a prompt campaign).** Drafting sits at weight **0.00** in `SYNTHESIS_MIX` (2026-08-24) because `document_kind` / `party_context` / `focus_issue` / `question` are empty on **all 60,603 seeds** — every task asks the teacher to draft "for the party whose papers these are" against a judgment that already disposed of the matter. Exit numbers: 4/20 accepts, 66,666 tokens per accepted row, p=0.0095 against summarization. The unpark needs seeds carrying those fields, or the stream retargeted. **The arm design is already settled** (see §3.5): ctl = current base templates, fix = the genre-form rewrite ported from harmony, back to back, an untreated task type as the noise channel, pre-registered on `irac_placement` fail and think-rehearsal rate; prediction on two prior measurements is that rehearsal collapses to ~0.
2. **`statute_qa` provision-text gap.** 270 `statute_qa` tasks, **0 eligible seeds** — `statute_section_eligible` needs `meta_json.section_text` distinct from the seed body, and no seed in the control store carries provision text. Filling it needs official Gazette Act bodies (the `gazette.py` work on `law-v1-foundation`, which holds identities only) *and* a write to the control store. Until then the stream is unmeasurable, and every recovery-era arm makes **no claim whatsoever** about it.
3. **Transition stream.** The IPC↔BNS mapping audit is **closed** — the 2026-08-17 operator ruling signed off the last 17 sheet rows (IPC 146 flipping kind, the other 16 confirming theirs), so `resources/ipc_bns_map.jsonl` verifies all **171** rows and none are refused. What remains unrun is the stream itself: `transition.sample: 1100` with `eval_reserve: 150` reserved first from the same deterministic order and marked `held_out`.
4. **Gold labels: 46 of 180.** `calibration.pilot_export: 180`, `holdout: 40`, 5 folds of 28, `thresholds [3,4,5]`, `rules [min_axis, mean, both]`, `min_recall 0.60`, `min_precision 0.75`. The existing 46 rows are **model-generated references, not human gold** — all 46 carry the identical `labeled_at` timestamp `2026-08-19T10:32:47.555143Z`, a single batch write; no quality warrant may be derived from them. At 46 labels only 6 rows were fittable for any judge, which is why gemma's disqualification says nothing about gemma. `judge_threshold` is still empty, so **every accept rate in this archive is provisional** (the informal all-axes ≥4 rule, which happens to be the pipeline's own fallback). A real fit needs gold round 2 (~180 labels).
5. **Judging throughput on the free fleet** (an operator scheduling fact, not a code fix): qwen ~**30 calls/day** against its 200k tpd at ~6–9k-token judge prompts; gemma ~**1,100 calls** on the remaining ~$4.63 (~8M cerebras tokens); `bai/deepseek-v4-flash` holds judge and tiebreak seats but is **dormant on every row** while it is the sole generator (`family_separation`). Groq multi-key rotation is the ~10× lever and is unbuilt. Also unresolved: both live judges on a deepseek row are keyed by the same `GROQ_API_KEY`, so groq going down leaves that row with **zero** judges, not one.
6. **Watch-items for the next live batch** (the recalibration's projections are exact over stored traces; the first live batches are the real instrument): expect `verbatim_overlap` **~4–9%** (summarization runs hotter), `length_band` `think>think_max` roughly halved, **zero** `reply_over_budget` events, and a clean rate in the high-20s / low-30s on post-F2 templates. **If verbatim comes back materially above ~10%, the ±14pp pool drift is the first suspect — partition by recorded `max_run` (stale-threshold rows still exist in LIVE and `exp_dialect`) before touching anything.**
7. **Still-open gate work.** `irac_placement` still fails ~60% of *irac_analysis* generations — F2 only ever treated summarization, and the same four-heading drift lives in the drafting templates. A second, differently-aimed iteration is the obvious next experiment. Any arm on this provider must run back to back with an untreated task type as a noise channel; ±13–18pp gate drift on byte-identical templates an hour apart is measured, not hypothetical.
8. **Training-side.** Re-verified against source 2026-08-29 — the 2026-08-08 wording of this item was stale on all three fronts and had begun re-baiting sessions into dead work. Actual state: **packing was REJECTED on the merits by the 2026-08-09 perf audit** (`training/configs/law_v1_8b_ddp.yaml` train block carries the full rationale: correct-but-net-negative — forfeits SDPA `is_causal` + `enable_gqa`, materializes a 64 MiB mask, pays 8192² attention over ~2,500-token segments, while bs=1 pads zero tokens and `ga=6` already buys the 3× gradient batch), so the spec-era `packing=True` / `find_packed_sequence_indices` TODOs from §1 are **dead, not open** — the §1 dataset-spec line predates that audit. The LR-rebuild-on-resume jump is **guarded**: `check_resume_schedule` (sft.py:78) refuses a resume whose `max_steps` changed, and main's `max_steps` is frozen after the first session — `--allow-schedule-change` exists for the RESUME gate only. The memory gate **does** read reserved: `check_vram_reserved` runs post-run (sft.py:676) and the live abort line is enforced against reserved. Still genuinely open: true tok/s and reserved peaks have never been measured on a MAIN-shaped run, and the 2026-08-09 changes still want one full SMOKE requalification before MAIN.
