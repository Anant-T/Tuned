# The deepseek recalibration: every gpt-oss-fitted constant re-fit for the generator that generates

2026-08-28. Operator directive: resolve the verbatim_overlap finding and strip the
settings tuned for gpt-oss-120b from the deepseek lane. This is the Phase-4
implementation of the two root-cause reports filed earlier today
(`2026-08-28-verbatim-overlap-drafting-drift.md`) — the fixes those reports
deliberately did not apply.

Every change below was validated OFFLINE against the ~2,900 generations the build
has already paid for (all stores opened `mode=ro`); no teacher was called. The full
suite is green after the change: **3,582 passed, 19 skipped**.

## The defect class, named once

bai/deepseek-v4-flash has been the sole `routing.generator` since the 2026-08-28
operator directive, but three load-bearing constants were still fitted on
gpt-oss-120b measurements, and each was failing or fining the current generator for
being the model it is:

| constant | was | fitted on | now | fitted on |
|---|---|---|---|---|
| `gates.DEFAULT_MAX_RUN` | 120 | 55 gpt-oss pilot traces | **500** | 1,086 deepseek traces, 11 arms |
| `length_band.think_max` | 3000 | gpt-oss pilot + the 08-27 cost case | **4000** | n=1,086 band sweep, all 19 configs |
| `generate.GENERATION_OUTPUT_TOKENS` | 4000 | cerebras max_output 4096 + gpt-oss chars/token | **16384** | the sole generator's declared reply ceiling (the wire value `_bai_request_hook` was already sending) |
| `generate.REPLY_BUDGET_CHARS_PER_TOKEN` | 5.5 | gpt-oss max 5.13 | **5.5 (kept)** | deepseek re-measure: p50 4.92, max 5.44 — holds; test floor moved 5.13 → 5.44 |

Plus one decoupling: `check_statutory_quotation.reproduces_grounding` no longer
reads `DEFAULT_MAX_RUN`; it keeps its own `QUOTATION_REPRODUCTION_RUN = 120`. A
quoted span is sentence-sized — at 500 the diagnostic could never fire again and
would have died silently, the exact "second consumer" drag the 30 → 120 note
already warned about.

## 1. verbatim_overlap 120 → 500

By the 2026-08-18 re-audit's own method, applied to the current generator. The
fine sweep over all 1,086 deepseek generations (longest run shared with
`seed.text`, the gate's own matcher, byte-exact grounding):

    max_run   120   150   200   250   300   350   400   450   500   600   800
    fails     52%   42%   29%   20%   15%   11%    7%    6%    4%    3%    2%

The per-100-char drop collapses from 7.8pp (300→400) to 1.3pp past 500 — the curve
flattens at 500, the same criterion that picked 120 for gpt-oss. 120 sat at
deepseek's median incidental overlap (p50 127), the position the re-audit condemned
30 for. What it failed was quotation (median 2.1% of the trace inside long runs),
not transcription; the residual 4% at 500 is genuine multi-sentence copying.

## 2. think_max 3000 → 4000, in all 19 configs

The "STAYS 3000" fence's own re-open conditions were met, and it is superseded on
its own terms:

* its condition — "a generator with a LARGER output ceiling leads
  routing.generator" — is the shipped state (deepseek max_output 16384, sole ref);
* its cost-side arithmetic was re-derived, not ignored (see §3): the worst-case
  gate-legal reply at 4000 is ~5,089 real tokens at the measured deepseek MINIMUM
  3.93 chars/token, covered 3x by the new budget.

Benefit re-measured at n=1,086 (not the 99-row sweep the fence rested on):

| think_max | irac_analysis band pass | summarization band pass |
|---|---|---|
| 3000 | 42.1% | 50.8% |
| **4000** | **51.7%** | **59.8%** |
| 4500 | 54.0% | 62.9% |
| 5000 | 56.0% | 63.9% |

The curve flattens past 4000 while total_max-alone blockage grows. The decisive
fact: deepseek's **median** irac trace (think_est 3,227) failed the old ceiling —
the band was rejecting this generator's typical deliberation.

This also **resolves recorded follow-up #1** (the summarization-specific
length_band F2 shipped with): once the shared ceiling stops binding the median
trace, no per-task-type band is needed. The verbatim/length coupling measured this
morning (fail rate 17% → 83% by think quartile) is priced into the joint numbers
below.

## 3. GENERATION_OUTPUT_TOKENS 4000 → 16384 — an alignment, not a re-pricing

16384 is what every generation call has ACTUALLY sent since 2026-08-25:
`_bai_request_hook` raises the caller's budget to the model's reply ceiling because
deepseek bills reasoning against `max_tokens` and emits it first. The caller's
4000 was a fiction sized to cerebras/gpt-oss-120b (max_output 4096) — a ref no
longer in `routing.generator` — and the fiction billed real money:

* **`reply_over_budget` misfired 347 times across the 11 deepseek arms** (~a third
  of calls): the bound sat at 22,000 chars (4000 × 5.5) while the wire permitted
  ~90,000, so billed, legitimate replies were sent back for regeneration. Even a
  fully band-legal reply could trip it.
* The alignment flips a structural inequality: the reply budget (90,112 chars) now
  exceeds the band ceiling (total_max × 4 = 32,768), so a gates-passing row can no
  longer breach the judge-sizing premise — it holds by construction. Window
  narrowing in `judge_tokens_for_generator_window` is consequently unreachable on
  the shipped constant (the `min` clamps to the flat, band-derived worst case at
  every window); the flat worst case and required_context numbers are untouched.
* If a gpt-oss generator ever returns, `_cerebras_request_hook` clamps to 4,096 on
  the wire, exactly as before.

`REPLY_BUDGET_CHARS_PER_TOKEN` stays 5.5: deepseek re-measured under it (max 5.44
over 1,086 generations, vs gpt-oss's 5.13). The pinning test's floor moved to 5.44
so the remaining margin is the measured one.

## The joint projection (offline, exact)

Stored verdicts with verbatim and band recomputed at the shipped values:

| population | clean before | clean after |
|---|---|---|
| all deepseek (n=1,086) | 13.4% | **19.5%** |
| post-F2 arms (n=184) | 18.5% | **31.0%** |
| post-F2 summarization (n=79) | 21.5% | **44.3%** |
| post-F2 irac_analysis (n=105) | 16.2% | **21.0%** |

Per generated row the gate-fail load collapses from three sides at once: the ~52%
verbatim coin-flip (retries were P(pass|fail)=43% — pure token burn) drops to ~4%,
the band stops failing the median trace, and the 347-per-1,086 reply-budget
regenerations go to zero. On an 8-requests-per-minute bucket where every call
costs one slot, that IS the throughput fix: fewer wasted calls per accepted row,
no request-shape change needed (batching stays as configured; `reasoning_effort:
low` stays the reliability setting).

## What was deliberately NOT removed

* **The harmony machinery** (`harmony.py`, the `prompts_harmony/` overlay, the
  config flags): gpt-oss-specific but experiment-only and OFF on the live config —
  and the harmony drafting rewrite is the proven fix pattern the parked
  gen_drafting arm will port when drafting unparks. Deleting it would destroy the
  precedent.
* **The cerebras/gpt-oss-120b provider block**: unrouted as generator, kept per
  this file's own convention ("unpinned rather than deleted, so its measured
  limits are not lost"); cerebras spends on judging only (gemma slot B), per the
  operator directive.
* **`GENERATOR_REASONING_PARAMS`' mistral entry, `EFFORT_LADDER_RETIRED`,
  `think_min: 500`**: recorded quirks and a floor deepseek's long traces never
  touch (gpt-oss's 29.7% floor-failure mode is a fact about gpt-oss, preserved in
  the fence).
* **The chars//4 estimate currency**: definitional for the band, not a per-model
  calibration.

## Test/fence changes (all deliberate, all dated in place)

* `test_build_gates.py`: pin 120 → 500 with the new re-audit in the docstring;
  `SOURCE` extended with inert prose so `SOURCE_RUN_LONG` can slice 500 chars;
  `_ALIGNMENT_TEXT` ×3 → ×5; the quotation-diagnostic test unchanged (the
  decoupling is what keeps it true).
* `test_build_config.py`: three think_max pins 3000 → 4000.
* `test_build_generate.py`: budget test converts at the deepseek minimum 3.93 and
  caps at 16384; the decoupling mutation-test literal tracks the new band line;
  the reply-budget floor 5.13 → 5.44; the "gates-passing row breaks the sizing
  premise" test is REWRITTEN as its own closure (the window no longer exists —
  the test now pins the inversion `total_max * 4 < reply_budget_chars`); the
  permanent-gate-precedence test resized to a >90k-char monster (band co-fires
  now, structurally; precedence is what it decides).
* `test_build_providers.py` + two routing tests: the four narrowing-algorithm
  tests run at the pre-alignment budget via monkeypatch — the cliff is unreachable
  at any window on the shipped constant, and the algorithm is what any future
  narrow-window generator rests on.

## What to watch on the next live batch

The projections above are exact over stored traces, but the first live batches are
the real instrument: expect verbatim_overlap ~4–9% (summarization runs hotter),
length_band think>think_max roughly halved, zero reply_over_budget events, and a
clean rate in the high-20s/low-30s on post-F2 templates. If verbatim comes back
materially above ~10%, the ±14pp pool drift the campaign measured is the first
suspect — partition by recorded `max_run` (stale-threshold rows still exist in
LIVE/exp_dialect) before touching anything.

Uncommitted alongside this report: the two root-cause reports' scratch scripts live
in the session scratchpad; the store fingerprints are untouched.
