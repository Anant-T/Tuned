# DeepSeek validation wave — measured 2026-08-27

**Arm:** `data/build/exp_deepseek`, config `configs/data_law_v1_exp_deepseek.yaml`
**Spec:** `docs/superpowers/specs/2026-08-26-deepseek-validation-wave-design.md`
**Generator:** `bai/deepseek-v4-flash`, `reasoning_effort: low`, temperature 0.7, top_p 0.95
**Judges:** qwen3.6-27b (A), gemma-4-31b (B), mistral-large (tiebreak); openai fenced to $0
**Live control store:** untouched — `554532864 1787309490` before and after

## Verdict

**The infrastructure passed everything. The generator does not fit the build's format
contract.** All three hard pass lines are green: 99 of 99 calls returned content, every
generation came from deepseek, no paid model was ever called, and the seed gate held with 35
oversize seeds present and zero planned against.

But **79 of 99 generations were gated out**, only 20 of 40 tasks ever reached a judge, and 20
parked as `format_parked`. The `think_max` violation rate is **44%**, well over the 30%
threshold pre-registered as "`low` and `think_max` disagree; one must move."

**The length projection was optimistic and is now corrected.** Real reasoning averages **2,739
tokens** (pinned tokenizer) against the 2,097 the n=4 qualification probe measured — that probe
under-counted by 31%. This arm's rows average **5,677** tokens against a projected 4,440, and
**8.1% exceed 8192** against a projected 1.5%.

Those two arm figures are **not a like-for-like comparison** with the projection: the arm was
stratified ~1/3 per source by design, while the projection ran over the live seed population
(~67% `sc`, the shortest source). Reweighted to the projection's own mix, the arm's row mean is
**5,423 (+22%)** and its over-8192 share is **5.5% (3.7×)** — see §3a. **The like-for-like
verdict on the projection is +22% on the mean and 3.7× on the over-cap share**, not +28%/5×.
Either way it is optimistic and either way rows are landing over the trainer's 8192 cap, but
the miss is the smaller number.

The judge signal is **75% accept (9/12)** against a 17% baseline, but that number is confounded
past the point of usefulness — see §5. The dominant failure is **`irac_placement` at 62%**,
which is a prompt-contract mismatch, not a length problem.

<!-- measured output of data/build/exp_deepseek/out/report_wave.py -->

## 1. Pipe health

- generations recorded: **99**; with content: **99** (100%) — pass line ≥ 90%: **PASS**
- errored rows: 0; finish_reason=length: 0
- latency s: mean 35.5, p50 32.5, p90 53.4, max 113.9
- models seen: `[('deepseek-v4-flash', 99)]` — only deepseek: **PASS**
- openai requests on the ledger: 0 — $0 fence: **PASS**
- ledger: `bai/deepseek-v4-flash req=99 429=0 tok=247671+366112`;
  `groq/qwen3.6-27b req=20 429=0`; `cerebras/gemma-4-31b req=16 429=0`;
  `mistral/mistral-large-latest req=2 429=0`

**Zero 429s across 99 calls** — `rpm: 8` against the measured bucket of 10 is correctly
conservative. The `_bai_response_hook` truncation path never fired: `max_output: 16384` is
comfortably above what `low` produces.

## 2. Seed gate, live

- budget `seed_token_budget(cfg)` = **4692** tokens
- oversize seeds present in the arm store: **35** (need ≥ 1)
- tasks planned against an oversize seed: **0** (need 0) — **PASS**

The gate committed in `70131e1` works. This is the first live confirmation.

## 3. Lengths vs the 2026-08-26 projection

| series | n | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| reasoning (pinned tokenizer) | 99 | 2739 | 2263 | 4721 | 10901 | 11359 |
| reasoning (provider-reported) | 99 | 3236 | 2585 | 5448 | 12332 | 13797 |
| answer | 99 | 1036 | 1033 | 1433 | 1806 | 2125 |
| seed (user turn) | 99 | 1903 | 1722 | 3255 | 3690 | 3690 |
| **templated row** | 99 | **5677** | 5455 | 7504 | 12265 | 14698 (**8 over 8192**) |
| row — arm `sc` | 32 | 5189 | 4980 | 6934 | 9533 | 9533 (1 over) |
| row — arm `predex` | 34 | 5851 | 5653 | 7636 | 11404 | 11404 (3 over) |
| row — arm `tathya` | 33 | 5971 | 5706 | 9745 | 14698 | 14698 (4 over) |

| | projected | measured | error |
|---|---|---|---|
| reasoning mean | 2,097 | **2,739** | +31% |
| row mean | 4,440 | **5,677** | +28% |
| row p90 | 5,514 | **7,504** | +36% |
| row p99 | 9,719 | **12,265** | +26% |
| rows over 8192 | 1.5% | **8%** | 5× |

**The error does not trace to a single input.** An earlier draft of this section said it did —
that the whole row-mean miss was the reasoning constant. The projection is additive, so it
decomposes, and all three terms are wrong in the same direction:

| term | projected | measured | error | share of the miss |
|---|---|---|---|---|
| seed (user turn) | 1,572 | **1,903** | +331 | 27% |
| reasoning | 2,097 | **2,739** | +642 | **52%** |
| answer | 791 | **1,036** | +245 | 20% |
| sum of terms | 4,460 | 5,678 | +1,218 | |
| row mean as reported | 4,440 | 5,677 | **+1,237** | |

(The ~19-token gap between the term sum and the reported row means is template/rounding
overhead, not a fourth term; the shares above are taken against the reported +1,237.)

Reasoning is the largest single term at 52%, and its cause is the one already identified: the
projection used the qualification report's `reasoning_effort: low` figure of 2,097 tokens,
measured at **n=4 on one synthesis prompt**, against a true mean of 2,739 with a p99 of 10,901
— a long right tail four samples could not see. But that is barely half the miss. Seed and
answer together contribute the other 48%, and calling the projection "one bad constant"
understates how much of it has to be re-derived.

**Retracted: the 0.77 answer-scaling assumption.** The projection scaled the answer term by
**0.77** on the theory that deepseek answers more tersely than gpt-oss (791 = 1,027 × 0.77).
Measured, deepseek answered **1,036** tokens against gpt-oss's **1,027** — a factor of
**1.01**. Deepseek does not answer shorter; it answers the same length and thinks ~4× longer.
Both numbers were available and were never compared: 1,036 sits in the §3 table above and
1,027 is the gpt-oss figure the scaling was applied to. That assumption is withdrawn, not
adjusted — any re-projection should carry the answer term through at 1.0.

Note the provider-reported figure (3,236) runs ~18% above the pinned-tokenizer count (2,739).
Both are correct: they are different tokenizers. The pinned count is the one the trainer's cap
gates on and is the number to use.

### 3a. The like-for-like comparison (the raw one is not)

The projected/measured table above is **not like-for-like**, and the direction of that is
against the projection. The projection ran over the **live seed population**, which is ~67%
supreme-court chunks — **939 `sc` / 253 `predex` / 204 `tathya`** generations. This arm
deliberately drew **roughly one third from each source** (13 `sc` / 14 `predex` / 13 `tathya`
tasks): the plan stratified it precisely so the long-tail sources were not swamped by `sc`. The
spec says so in as many words — *"69% of seeds are SC chunks capped at 1,500 tokens. A default
40 would draw ~28 short chunks and ~12 from the two sources that actually produce the length
tail."* So the arm's equal-weighted mean is compared
against a population-weighted projection, and since `sc` is the *shortest* arm (5,189) the
equal weighting inflates the arm's mean.

Reweighting the arm's own §6 per-source figures to the projection's source mix:

| statistic | arm, equal-weighted (raw) | **arm, reweighted to the projection's mix** | projection |
|---|---|---|---|
| row mean | 5,677 (**+28%**) | **5,423 (+22%)** | 4,440 |
| share over 8192 | 8.1% (**5×**) | **5.5% (3.7×)** | 1.5% |

**Use the reweighted column when judging the projection.** On a like-for-like population the
projection was optimistic by **+22% on the row mean and 3.7× on the over-cap share** — still a
real miss, still the wrong side of the 8192 cap, but not the +28%/5× the raw arm numbers
suggest. Use the **raw** column when judging *this arm's output*, because that is what the arm
actually produced and what the gates actually saw.

## 4. `think_max` violation rate (the real unknown)

- length_band gate results: 99; passed: 49
- `think>think_max` (3000): **44/99 = 44%** — threshold 30%: **OVER**
- `total>total_max` (8192): 33; `think<think_min` (500): 3

**Per-gate failure rates over all 99 generations:**

| gate | fails | rate |
|---|---|---|
| `irac_placement` | 61 | **62%** |
| `length_band` | 50 | **51%** |
| `verbatim_overlap` | 46 | 46% |
| `banned_meta` | 14 | 14% |
| `prompt_echo` | 13 | 13% |
| `self_verification` | 13 | 13% |
| `statutory_grounding` | 7 | 7% |
| `think_format` | 2 | 2% |

**There is no knob between these two failure modes.** `reasoning_effort` is an enum —
`low | medium | high | xhigh | max` — and `low` is already the floor. The only setting below it
is `thinking: {"type": "disabled"}`, which yields **exactly zero** reasoning and would violate
the corpus's ≥80% reasoning-trace floor outright. `budget_tokens` was measured advisory, not a
cap. So deepseek has **no configuration that reliably produces a 500–3,000 token trace**, which
is precisely the band `length_band` declares. That band was calibrated for gpt-oss-120b's ~700
token traces; deepseek's floor is ~4× that.

## 5. Judge accept rate vs the gpt-oss baseline

- arm (deepseek): decided **12**, accepted **9 = 75%**; decisions `{accept: 9, regenerate: 2, reject: 1}`
- live baseline (gpt-oss, read-only): decided 90, accepted 15 = **17%**
- n=12 ⇒ roughly a **±28 pp** interval
- judges used: a=`groq/qwen3.6-27b` ×20; b=`cerebras/gemma-4-31b` ×12; tiebreak=`mistral-large` ×2

**The two breakdowns of these same 12 rows do not match, and both are correct.** §4's
task-state line and `judge.log` record `rejected=3, regen=0`; the line above records
`{accept: 9, regenerate: 2, reject: 1}`. The difference is not a discrepancy in the data: the
measurement script recomputes the provisional rule from the stored axis scores by calling
`dual_judge_decision(..., already_regenerated=False)`, while the pipeline's recorded
disposition reflects the regeneration state each task was actually in. `judge_policy.decide`
ends `return "reject" if already_regenerated else "regenerate"`, and all three rejected tasks
carried `attempts` of 2, 3 and 4 from the generate-phase format-gate retries — so they were
already past their one regeneration when the judge saw them, and the pipeline correctly landed
`reject` where a fresh recompute reads `regenerate`. Same 12 rows, two partitions of them,
taken under different assumptions about attempt history.

**Do not read this as "deepseek is 4× better."** Three confounds, each large enough alone to
explain the gap:

1. **Survivorship.** The arm's 12 decided rows are what survived an 80% gate-out. The baseline's
   90 were judged under a regime that parked far fewer. Comparing post-filter acceptance across
   two different filters measures the filters.
2. **Different prompt versions.** The live store's judgements were made under earlier
   `gen_*` prompt revisions — the live store carries 419 tasks in `stale_prompt` for exactly
   this reason.
3. **n=12.** The interval is ±28 pp. A 75% point estimate is consistent with anything from 47%
   to 100%.

The honest reading: **nothing here contradicts deepseek producing acceptable rows, and nothing
here establishes it.** A matched A/B on the same seeds is the design that would settle it, and
it was scoped out of this wave for time.

## 6. Per-arm breakdown

| arm | tasks | calls | with content | accepted |
|---|---|---|---|---|
| `sc` | 13 | 32 | 32 | 2 |
| `predex` | 14 | 34 | 34 | 7 |
| `tathya` | 13 | 33 | 33 | **0** |

`tathya` returned **zero** accepted rows from 33 calls. **That zero is not a quality signal and
must not be read as one.** An earlier draft read it as evidence that the IRAC-heavy mix (0.55)
fits TathyaNyaya's fact-extraction seeds poorly. The store contradicts that on both halves:

- **`tathya` parks *less* than `sc`, not more.** `format_parked` rates: **`sc` 8/13 = 62%**,
  **`tathya` 7/13 = 54%**, **`predex` 5/14 = 36%**. If the format contract fit `tathya` worse
  than the other sources, this is the number that would show it, and it does not.
- **No `tathya` row was ever decided.** Judging outcomes by source: **fully decided (slot A
  *and* slot B) — `sc` 5, `predex` 7, `tathya` 0**; **partial (slot A only, then
  `judge_error`) — `predex` 2, `tathya` 6**. All six of `tathya`'s judgeable rows were scored
  by qwen in slot A and then died waiting for gemma (see Caveats). The denominator for
  `tathya`'s accept rate is zero, so "0 accepted" is arithmetic, not evidence.

**This wave supports no conclusion about IRAC fit for `tathya`.** What it does establish about
that source is a length fact: `tathya` carries the worst tail of the three, p99 **14,698**, and
4 of its 33 rows exceed 8192. That is measured and it stands.

## 7. Cost and wall clock

- generate: 26 batches, 99 calls, **613,783 tokens**, **26.4 min** at 4 workers
- judge: 23 batches, 34 judgements, 173,914 tokens, **≈19 min (approximate)** at 3 workers
- **paid spend: $0** — fence verified, zero openai requests on the ledger
- 429s: **0** across every provider

The generate figure is exact: `generation.created_at` spans
`2026-08-26T18:56:20.600Z` to `2026-08-26T19:22:42.637Z` = **26.4 minutes** for all 99 calls.
The judge figure is **not** exact — `judge.log` carries no timestamps — and is derived from
file mtimes: `generate.log`'s final write (00:52:42 IST) to `judge.log`'s (01:11:29 IST) is
18 min 47 s, which bounds the judging phase from above. Treat it as ≈19 min, not measured.

**The economics are worse than the head-to-head projected.** That comparison assumed roughly
one call per example and derived "25–33 hours for a 15–20k corpus." Measured here: **99 calls
produced 9 accepted rows** — 11 calls per accepted example, so a 15,000-example corpus needs
**~165,000 calls**.

**The throughput term has to come from this run, not from the rate-limit ceiling.** An earlier
draft priced those calls at b.ai's ~600 calls/hour and got ~275 hours ≈ 11 days. That 600 is
the **rate-limit bucket ceiling** from the qualification report — the most the provider would
admit — and this run never approached it. Measured: **99 calls in 26.4 minutes = 225
calls/hour**. The run was **latency-bound, not rate-bound**: 4 workers, batch-synchronous, mean
latency 35.5 s and max 113.9 s, and **zero 429s** across all 99 calls, which is itself the
proof that the bucket was never the binding constraint.

At the measured 225 calls/hour, 165,000 calls is **~732 hours ≈ 31 days** of continuous
running, not 11 days and not 25–33 hours.

Two directions on that 31 days, and they point opposite ways:

- **It is the measured-configuration figure, not a floor.** More workers, or an async rather
  than batch-synchronous loop, would raise throughput toward the 600 calls/hour ceiling —
  which at the limit would put the same 165,000 calls at ~275 hours ≈ 11 days. The gap between
  11 and 31 days is entirely concurrency, and it is recoverable engineering.
- **It is not an upper bound on the pessimism.** That claim applied only to the **yield**
  term — 11 calls per accepted row — where most of the loss is `irac_placement` and
  `verbatim_overlap`, prompt-contract failures that could plausibly be fixed without touching
  the generator. It was never true of the throughput term, and it is withdrawn as applied
  to it.

## 8. What changes in the live config

**Nothing yet — and deliberately not.** This wave was scoped to *measure* whether
`reasoning_effort`, `think_max` and the seed reserve agree. They do not. Changing them is a
separate, evidence-backed decision, and the evidence points at three mutually exclusive routes:

1. **Raise `length_band.think_max` from 3000** to ~5,000 and re-derive
   `GENERATION_OUTPUT_TOKENS`. `generate.py:227-251` states a test deliberately pins an
   inequality so that raising `think_max` *fails a test rather than silently changing budgets*
   — that fence must be walked, not bypassed. **The inequality is not the one an earlier draft
   quoted.** `tests/test_build_generate.py::test_the_generation_budget_covers_the_largest_gate_legal_reply`
   asserts

   ```
   max_output_tokens(cfg) >= legal_reply_chars(cfg) / 4.24
   ```

   where `legal_reply_chars(cfg) == (think_max + ANSWER_TOKEN_ALLOWANCE) * 4` in *characters*
   — the `* 4` is a chars-per-estimate-token conversion that is then divided back out at the
   measured worst-case 4.24 chars/real-token. Quoted without the `/ 4.24`, as
   `GENERATION_OUTPUT_TOKENS >= (think_max + ANSWER_TOKEN_ALLOWANCE) * 4`, it reads
   `4000 >= 16000`, which is false at shipped values and would mean the fence is already
   broken. It is not: `4000 >= 16000 / 4.24 = 3774` holds with 226 tokens of margin.

   **Route 1 is harder than it looks, and the corrected inequality is why.** At
   `think_max: 5000` the requirement becomes `(5000 + 1000) * 4 / 4.24 = 5,660` output tokens.
   Raising `GENERATION_OUTPUT_TOKENS` to cover that immediately trips the **same test's** next
   assertion, `max_output_tokens(cfg) <= 4096` — the cerebras `gpt-oss-120b` declared
   `max_output`, above which `_cerebras_request_hook` clamps the call without saying so. So
   route 1 is not one fence but two in series, and clearing the second means either dropping
   cerebras as the overflow generator or accepting a silent clamp on it. Consequence, still:
   rows get longer, and the 8.1% already over 8192 grows.
2. **Keep deepseek as a minority diversity source**, not the lead generator, and restore
   `cerebras/gpt-oss-120b` to the head of `routing.generator`. Costs the daily-cap problem back
   (~364 examples/day) but keeps the format contract intact.
3. **Fix the prompts, not the model.** `irac_placement` at 62% and `verbatim_overlap` at 46%
   are the two largest losses and are not length problems. If deepseek's output shape can be
   brought into the IRAC contract, the length question shrinks to the 51% `length_band` rate,
   of which `think>think_max` is the tractable part.

Route 3 is the only one that does not trade something away, and it is testable with the same
arm at a fraction of the cost.

## Caveats

- **n=12 decided rows** for the accept rate, ±28 pp, and confounded three ways (§5). Treat §5
  as a null result, not a positive one.
- **One wave, one hour, one upstream draw.** The qualification report established that multiple
  upstreams sit behind the `deepseek-v4-flash` id with materially different behaviour; a run on
  a different day may draw a different mix.
- **The accept rule is UNCALIBRATED.** `judge.log` line 2: `NOTE: no active judge_threshold
  rows - decisions are PROVISIONAL (P5 calibrates)`. Both stores carry zero `judge_threshold`
  rows, so the arm-vs-baseline comparison in §5 is at least *symmetric* — both sides are
  scored by the same provisional rule — but neither side's absolute accept rate means anything
  until P5 fits the thresholds. Do not carry 75% or 17% forward as calibrated numbers.
- **8 rows died in `judge_error`** on `judge-slot-b: no eligible model (cooling,
  family-excluded, over-budget)` — gemma's `tpm: 30000` against ~8–9k-token judge prompts is
  the binding constraint. **The bias this introduces is compositional, not length-based.** An
  earlier draft said the lost 8 were the *longer* rows; that is unsupported and wrong. It is a
  clean **temporal** cut: gemma (slot B) entered cooling at judge batch 5 and never recovered
  (`judge.log`, `slot-err` on every batch from 5 onward, 46 in total), so whichever rows were
  claimed late lost their slot-B judge regardless of length. The composition of the 8 is
  **6 `tathya` + 2 `predex`** — and the resulting bias is **stronger** than the length story it
  replaces. Judging outcomes by source: fully decided (slot A *and* B) — `sc` 5, `predex` 7,
  **`tathya` 0**; partial (slot A only, then `judge_error`) — `predex` 2, **`tathya` 6**.
  **No `tathya` row was ever fully decided.** §5's accept rate therefore describes `sc` and
  `predex` only, and this is a rate-limit artifact concentrated on one arm, not a signal about
  that arm's quality (see §6).
- **The 40-task plan yielded 20 judgeable rows**, so every rate in §5 and §6 rests on half the
  intended sample.
- Baseline judgements in the live store predate the current judge prompt revisions.
