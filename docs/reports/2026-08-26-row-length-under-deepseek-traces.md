# Does 8192 hold the corpus? — measured 2026-08-26

**Question:** the seq cap was raised to 12,288 to make room for DeepSeek reasoning traces,
and the raise [OOM'd](../superpowers/specs/2026-08-26-seq-length-12288-design.md). Before
spending any more GPU quota, does 8192 actually hold the corpus we are about to build?

**Answer: yes, at `reasoning_effort: low` — which is already the shipped production setting.**
1.5% of rows would exceed the cap, against 1.2% today. The cap is not the constraint, and the
long tail is not made of reasoning.

**Method.** 1,368 real generations from `data/build/state/law_v1.sqlite3`, reconstructed into
rows exactly as `decontaminate.generated_rows` does, rendered through the pinned Qwen3 chat
template and counted with the pinned tokenizer — the same path `assemble.py` gates on. No API
calls; this is all already-spent work.

## What the corpus looks like today

| | n | mean | p50 | p90 | p99 | max | over 8192 |
|---|---|---|---|---|---|---|---|
| **templated row** | 1,368 | 3,297 | 3,055 | 4,707 | 8,366 | 10,466 | **16 (1.2%)** |
| user turn (seed grounding) | 1,368 | 1,572 | 1,387 | 2,696 | 6,547 | 8,726 | 3 |
| reasoning trace | 1,309 | 715 | 531 | 1,297 | 3,440 | 4,185 | — |
| answer | 1,368 | 1,027 | 1,037 | 1,429 | 1,728 | 2,242 | — |

By generator: `gpt-oss-120b` reasons 697 tokens on average (n=1,111),
`lightning-ai/gpt-oss-120b` 516 (n=146), `mistral-small-latest` 1,653 (n=52).
`magistral-small-latest` returned no trace at all on 59/59.

### Correction to an earlier claim

The seq-12288 spec asserted **"0 of 5,420 rows exceed 8192."** That was measured on
`curated_c1.jsonl` and `replay.jsonl` — the PredEx-rewrite streams, whose assistant turns carry
an **empty** `<think></think>` scaffold. The *generated* stream is a third source those two
files do not contain, and it does have a real over-cap tail: 16 rows, already being dropped.
The cap does cost rows. It costs 1.2% of them, and it is still the right cap — for the reason
below.

### The overflow is the seed, not the trace

The 16 rows over 8192 average a **7,597-token user turn** and a **492-token trace**. Three
seeds exceed 8192 on their own, before a single token of answer — unusable at any teacher
setting, any cap this hardware can reach. 39 seeds exceed 4,096.

Raising the cap to make room for reasoning was aimed at the wrong term. What fills the tail is
long *documents*.

## Projected row length under DeepSeek

Substituting `deepseek-v4-flash`'s measured reasoning lengths
([qualification report](2026-08-25-bai-deepseek-qualification.md)) into the same 1,309 rows,
holding the seed fixed and scaling the answer by 0.77 (the head-to-head measured deepseek's
content at ~6,200 chars against gpt-oss's 8,034 — it answers *shorter*):

| arm | reasoning | mean row | p90 | p99 | **over 8192** |
|---|---|---|---|---|---|
| current teachers (measured) | 715 | 3,296 | 4,707 | 8,368 | 16 / 1,309 — **1.2%** |
| **`reasoning_effort: low`** ← shipped | 2,097 | 4,440 | 5,514 | 9,719 | 19 / 1,309 — **1.5%** |
| `reasoning_effort: high` | 4,300 | 6,643 | 7,717 | 11,922 | 89 / 1,309 — 6.8% |
| baseline, no knob | 6,576 | 8,919 | 9,993 | 14,198 | **1,108 / 1,309 — 85%** |
| `reasoning_effort: max` (SSE) | 12,573 | 14,916 | 15,990 | 20,195 | 1,309 / 1,309 — 100% |

**`low` changes nothing.** It costs three extra dropped rows out of 1,309.

**Baseline would have destroyed the corpus** — 85% of rows dropped at 8192, and still 41% at
12,288 had that cap fit. The instinct that drove the raise was right that baseline traces do
not fit; the cap that was chosen would not have saved them either.

### The tail's shape holds up

The projection adds a constant to every row, which would understate the tail if a model reasons
longer about longer documents — the trace would grow exactly where the row is already long.

It does not. Across 1,309 real generations, **corr(seed length, trace length) = −0.068** —
no relationship, very slightly negative. A constant shift is the right model, and the
proportional-tail worry is unfounded.

## Consequences

1. **Keep `max_seq_length: 8192`.** Nothing in the corpus argues for more, and the hardware
   cannot deliver it: 12,288 OOM'd by ~0.8 GiB, and at ~0.35 GiB per 1,024 tokens even 10,240
   lands on the abort line. Chasing p99 would buy ~1% of rows for a cap that does not fit.
2. **Keep `reasoning_effort: low` on the b.ai generator.** Already shipped in
   `configs/data_law_v1.yaml`, already justified there as a reliability setting. This is the
   second, independent reason to keep it.
3. **`reasoning_effort: "minimal"` is not a legal value** — the enum is
   `low | medium | high | xhigh | max`, and one upstream enforces it while the other does not,
   producing a ~20% hard-failure rate. Any note recommending `minimal` (including earlier
   working notes on this question) should read `low`.
4. **The row-length lever that remains is the seed, not the teacher.** Trimming or chunking the
   39 seeds above 4,096 tokens would do more for the drop rate than any reasoning setting.
5. **Nothing here needs GPU quota**, and the corpus question that blocked the main run is now
   answered for the length dimension.

## Caveats

- These 1,368 generations come from the current teacher mix on the current seed mix. The final
  corpus targets 15–20k examples at a 60/16/24 task split; if that shifts the seed-length
  distribution, the tail moves with it. The measurement should be re-run against the built
  corpus before `max_steps` is derived.
- DeepSeek's reasoning figures are from the qualification report (n=4 per arm on the synthesis
  prompt), and baseline reasoning there ranged 298–10,426 tokens at temperature 0. The arm
  *means* are what this projection uses and they are the best available, but per-row variance
  is wide. A live run over these same seeds would tighten the tail estimate — it is
  confirmatory, not decisive, since `low` clears the cap with 2.7k tokens of margin at p90.
- `magistral-small-latest` contributing 59 trace-less generations means the ≥80%
  reasoning-trace floor is a live constraint on generator mix, separate from length.
