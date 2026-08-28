# deepseek quality and efficiency campaign — consolidated close-out

**Date:** 2026-08-28. Sole generator: `bai/deepseek-v4-flash`. Every experiment ran
pre-registered, back-to-back arms on the free bai tier; every report's headline numbers were
independently recomputed from the arm stores by a reviewer (zero deltas across all four
report groups); the live control store stayed byte-identical throughout
(`554532864 1787309490`).

## What shipped (each after its own A/B and review)

1. **The prompt-ceiling edit was reverted** (`06f588a`) — all 14 base templates back to the
   pre-edit wording. Grounds: proven harm to gpt-oss, at-best-wash for deepseek
   (`2026-08-28-deepseek-prompt-era-rerun.md`: +4.97pp for pre-edit, every secondary leaning
   the same way), zero benefit to anyone.
2. **The anti-rehearsal clause** (`0637fa0`) — six templates (`gen_irac_analysis_v1-4`,
   `gen_summarization_v1-2`) now tell the model that a labelled Issue/Rule/Application/
   Conclusion run-through inside its thinking is a hidden first draft of the answer.
   Three-arm A/B (`2026-08-28-deepseek-clause-and-cap-ab.md`), clause vs shared control:

   | metric | control | clause | delta |
   |---|---|---|---|
   | `length_band` pass | 42.2% (n=109) | 58.3% (n=103) | **+16.05pp** |
   | think est-tokens p50 | 2,974 | 2,418 | **−18.7%** |
   | full-gate clean | 11.0% | 14.6% | +3.6pp |
   | total ledger tokens per length-passing row | 15,210 | 10,359 | **−31.9%** |

   Shipped with an honest asterisk: its pre-registered primary (irac_placement −15pp) FAILED
   (−4.5pp); the clause ships as a think-length lever, which its designers did not predict.

## What was measured and closed (negatives, recorded so nobody re-runs them)

- **Judge quality is not the bottleneck.** First-ever calibration
  (`2026-08-28-deepseek-judge-calibration.md`): pooled accept **82.9%** (29/35, production
  qwen+gemma pair, informal all-dims≥4 rule — `judge_threshold` still unfitted). Cost $0.071
  of a $0.50 budget. The yield ceiling is upstream, at the format gates.
- **The `max_output` cap (5000) is closed, not shipped.** It won its token metric (−10.7%
  completion tokens per passing row) but lost 4/40 tasks to chronic truncation whose control
  twins passed. On a free, request-rate-bound provider, that trade is bad. The offline
  zero-clip prediction from banked data was invalidated by fresh upstream drift — banked
  maxima do not bound tomorrow's traces.
- **hy3 is closed as an alternative generator** (`2026-08-28-hy3-think-low-probe.md`):
  19.4% length_band (n=72) with think p50 3,440 against a 3,000 band, and the b.ai doc's
  `think_low` tier name is not the API binding (live 400; the real enum reuses
  `reasoning_effort`). Its legal content was sound and its format never broke — the model
  fails this build's band, not the task.
- **The headed-rehearsal pathology is cross-model** — hy3 rehearses IRAC inside its trace
  worse than deepseek (95.8% vs 73.4%). It is a reasoning-model habit, not a deepseek quirk.
- **Steering levers confirmed dead** (external research + prior measurement): no
  thinking-budget parameter exists anywhere in the DeepSeek/b.ai stack; temperature/top_p are
  documented as ignored in thinking mode; `reasoning_effort` is already at its floor.

## What remains open

- `irac_placement` still fails ~69% of clause-arm generations — the single biggest
  remaining gate tax. The clause moved it −4.5pp; a second, differently-aimed iteration is
  the obvious next experiment.
- `judge_threshold` remains unfitted; accept rates above are the informal rule.
- Groq multi-key rotation (the ~10× judge-throughput lever) — out of scope here, unbuilt.
- The `sc` seed source underperforms on both format parks and judge accept (50% pooled at
  n=8) — not yet statistically load-bearing.

## Ledger

Commits: `1fff019` `58a6334` (E0) · `06f588a` (revert) · `f4555f9` `0d9b89b` (three-arm) ·
`092bc0a` (judge configs) · `5624471` `0150b1c` `0d11d7f` (hy3) · `0637fa0` (clause ship) ·
this commit (E4 report + close-out). Suite at close: 3576 passed / 19 skipped / 0 failed.
Spend: $0 generation (bai free tier), $0.071 cerebras judging.
