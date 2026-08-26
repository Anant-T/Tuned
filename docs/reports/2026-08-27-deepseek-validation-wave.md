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
under-counted by 31%. The templated row averages **5,677** tokens against a projected 4,440,
and **8% of rows exceed 8192** against a projected 1.5%.

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

The projection's error traces to a single input: it used the qualification report's
`reasoning_effort: low` figure of 2,097 tokens, measured at **n=4 on one synthesis prompt**.
Across 99 real generations the true mean is 2,739 with a p99 of 10,901 — the arm distribution
has a long right tail that four samples could not see. The **method** was sound (the constant
shift held; see the projection's own correlation check), but the constant was 31% too small.

Note the provider-reported figure (3,236) runs ~18% above the pinned-tokenizer count (2,739).
Both are correct: they are different tokenizers. The pinned count is the one the trainer's cap
gates on and is the number to use.

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

`tathya` returned **zero** accepted rows from 33 calls, and carries the worst length tail
(p99 14,698). TathyaNyaya seeds are fact-extraction cases; combined with a 62% `irac_placement`
failure rate, the IRAC-heavy mix (0.55) appears to be a poor fit for that source.

## 7. Cost and wall clock

- generate: 26 batches, 99 calls, **613,783 tokens**, ~26 min at 4 workers
- judge: 23 batches, 34 judgements, 173,914 tokens, ~22 min at 3 workers
- **paid spend: $0** — fence verified, zero openai requests on the ledger
- 429s: **0** across every provider

**The economics are worse than the head-to-head projected.** That comparison assumed roughly
one call per example and derived "25–33 hours for a 15–20k corpus." Measured here: **99 calls
produced 9 accepted rows** — 11 calls per accepted example. At b.ai's ~600 calls/hour that is
~55 accepted rows/hour, so a 15,000-example corpus is **~275 hours ≈ 11 days** of continuous
running, not 25–33 hours.

That figure is an upper bound on the pessimism: most of the loss is `irac_placement` and
`verbatim_overlap`, which are prompt-contract failures that could plausibly be fixed without
touching the generator.

## 8. What changes in the live config

**Nothing yet — and deliberately not.** This wave was scoped to *measure* whether
`reasoning_effort`, `think_max` and the seed reserve agree. They do not. Changing them is a
separate, evidence-backed decision, and the evidence points at three mutually exclusive routes:

1. **Raise `length_band.think_max` from 3000** to ~5,000 and re-derive
   `GENERATION_OUTPUT_TOKENS`. `generate.py:227-251` states a test deliberately pins the
   inequality `GENERATION_OUTPUT_TOKENS >= (think_max + ANSWER_TOKEN_ALLOWANCE) * 4` so that
   raising `think_max` *fails a test rather than silently changing budgets* — that fence must
   be walked, not bypassed. Consequence: rows get longer, and the 8% already over 8192 grows.
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
- **8 rows died in `judge_error`** on `judge-slot-b: no eligible model (cooling,
  family-excluded, over-budget)` — gemma's `tpm: 30000` against ~8–9k-token judge prompts is
  the binding constraint. Those 8 rows are missing from §5 non-randomly: they are the *longer*
  rows, which is the direction that would have lowered the accept rate.
- **The 40-task plan yielded 20 judgeable rows**, so every rate in §5 and §6 rests on half the
  intended sample.
- Baseline judgements in the live store predate the current judge prompt revisions.
