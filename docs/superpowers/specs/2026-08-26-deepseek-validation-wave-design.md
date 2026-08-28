# DeepSeek validation wave — an isolated 40-task arm

**Date:** 2026-08-26
**Status:** approved, not yet implemented
**Branch:** `worktree-law-v1-data-pipeline`
**Time box:** 30–40 minutes of wall clock for the run itself

## Summary

Run ~40 synthesis tasks through `bai/deepseek-v4-flash` in an isolated experiment arm, judge
them with the free fleet, and measure five things before the corpus build scales onto this
generator. Nothing touches the live control store.

The generator was qualified on 2026-08-25 with probes
([report](../../reports/2026-08-25-bai-deepseek-qualification.md)) and its row lengths were
*projected* on 2026-08-26 from stored gpt-oss generations
([report](../../reports/2026-08-26-row-length-under-deepseek-traces.md)). Neither has yet
pushed a real seed through the real pipeline — `tasks.py` → `generate.py` (with the `bai`
quirk, the length band, and the new seed gate) → `judge.py`. This arm does that.

## What it measures — pre-registered

| # | question | measurement | pass line |
|---|---|---|---|
| 1 | Does the pipe work end to end? | calls made, HTTP errors, 429s, `finish_reason=length` retries raised by `_bai_response_hook`, latency per call; **every** `generation.model` is `deepseek-v4-flash` and the `budget_ledger` shows $0 | **≥ 90%** of calls return non-empty content; 0 generations from any other model; $0 spent |
| 2 | Does the seed gate work live? | seeds in the arm store with `token_count > seed_token_budget(cfg)` (= 4,692) vs. tasks planned against such seeds | **exactly 0** planned tasks over budget, with **≥ 1** oversize seed present in the store |
| 3 | Were the projections right? | `think_tokens` and templated row length (pinned Qwen3 tokenizer, same path as `assemble.token_length`) vs. projected reasoning mean 2,097 and row mean 4,440 / p99 9,719 | reported; no pass line |
| 4 | **The real unknown:** how often does `reasoning_effort: low` blow the build's own `length_band.think_max = 3000`? | fraction of generations gated `think>think_max` | reported; **> 30%** means `low` and `think_max` disagree and one must move before scaling |
| 5 | Are the rows any good? | dual-judge accept rate (+ tiebreak), against the gpt-oss baseline computed from the live store's `judgement` table | reported — at n=40 the interval is ±15 pp, a signal not a verdict |

The pass lines are written here, before the run, so they cannot move afterwards.

## Isolation

- **Workdir:** `data/build/exp_deepseek`, a new sibling under `data/build`. Added to
  `ISOLATED_WORKDIR_SIBLINGS` in `src/tuned/data/paths.py`, with a test, so
  `is_live_control_workdir` returns `False` for it. Without that line every write guard in the
  tree treats the arm as the frozen live control.
- **Store:** a fresh `Store.open(data/build/exp_deepseek/state/law_v1.sqlite3)`. Seeds and
  sources are copied *out of* the live store over a read-only `ATTACH`
  (`file:...?mode=ro`). The live store is never opened for write by anything in this arm.
- **Config:** `configs/data_law_v1_exp_deepseek.yaml` — a copy of the live
  `configs/data_law_v1.yaml` (which already carries the uncommitted b.ai provider block and
  `bai/deepseek-v4-flash` at the head of `routing.generator`) with `build.workdir` changed, a
  header comment stating the arm's purpose, and **two cost/contamination fences the live config
  does not have**:
  1. `routing.generator: [bai/deepseek-v4-flash]` — the single ref. The live list falls over to
     `cerebras/gpt-oss-120b` (free) and then `lightning` (paid) when b.ai is cooling, and a 429
     storm would silently turn a deepseek arm into a gpt-oss arm, contaminating measurements
     3–5. The report asserts every `generation.model == 'deepseek-v4-flash'`.
  2. `usd_cap: 0.0` on the `openai` provider. The live config declares no cap, and
     `generate._openai_usd_cap` reads that as **uncapped**; gpt-5-mini/nano sit in `judge` and
     `tiebreak` as backstops, so a gap in the free pool would spend money silently. A zero cap
     is `exp_measure`'s precedent and makes spend on this arm impossible outright — judging runs
     entirely on qwen (A), gemma (B) and mistral (tiebreak).
  Copied from the **live** config, not from
  `exp_s1`/`exp_measure`: those arms carry `harmony_completions`, `harmony_prefill`,
  `harmony_s1_continue` and `prompt_overlay: src/tuned/data/prompts_harmony`, which are gpt-oss's
  Harmony chat format and must not reach a deepseek generation. `require_pretreatment_manifest`
  is left unset (its default, `False`): this is a smoke arm, not a matched-cohort eval.

## Seeding the arm — `scripts/seed_exp_store.py`

The previous arms were seeded by hand. This script makes it repeatable and is the one new
reusable piece.

```
python scripts/seed_exp_store.py --config configs/data_law_v1_exp_deepseek.yaml \
    --from data/build/state/law_v1.sqlite3 --per-source 200 --seed 3407
```

Behaviour:

1. `load_build_config(--config)`; refuse if `is_live_control_workdir(cfg.build.workdir)`.
2. Open the target store with `Store.open(paths.state_db)` (creates the schema); `ATTACH` the
   source DB read-only as `live`.
3. `INSERT OR IGNORE INTO source SELECT * FROM live.source`.
4. For each `source_id` in `live.source`, insert a deterministic sample of `--per-source` seed
   rows (`ORDER BY seed_id`, offset from `--seed` mod the source's count — content-derived ids
   are already a stable pseudo-random order, so no RNG is needed). The sample is taken **without
   a length filter**, so seeds above the planner's budget are present: that is what makes
   measurement #2 possible.
5. Print, per source: rows copied, and how many exceed `seed_token_budget(cfg)`.

Idempotent: re-running upserts nothing new (`INSERT OR IGNORE` on the primary key).

Not copied: `chunk_manifest` (extract.py's resume index — irrelevant here), `document`, `task`,
`generation`, `judgement`, `budget_ledger`. An arm must start with an empty ledger and an empty
queue.

## Planning — three arms, one per source

`--n` is a *target* for the `(stream, arm)` queue, so each source gets its own `--arm` label and
its own top-up. The label lands on the task row and is what the report groups by.

```
python -m tuned.data.tasks --config configs/data_law_v1_exp_deepseek.yaml \
    --stream synthesis --arm sc     --n 13 --source s3://indian-supreme-court-judgments \
    --mix irac_analysis=0.55,summarization=0.45
python -m tuned.data.tasks ... --arm predex --n 14 --source L-NLProc/PredEx_Instruction-Tuning_Pred-Exp ...
python -m tuned.data.tasks ... --arm tathya --n 13 --source L-NLProc/TathyaNyaya-and-FactLegalLlama-NyayaFacts-Datasets ...
```

Why stratified: the planner's default order is `(n_tasks ASC, seed_id ASC)`, and 69% of seeds
are SC chunks capped at 1,500 tokens. A default 40 would draw ~28 short chunks and ~12 from the
two sources that actually produce the length tail.

Why this mix: `drafting` is already parked at 0.0 in `SYNTHESIS_MIX`; `statute_qa` has zero
eligible seeds in the live store (measured 2026-08-23) and would spend 25% of the plan on
`skip:slots` rows that cost nothing but produce nothing.

Gate check happens here, before any call: after planning, assert
`SELECT COUNT(*) FROM task t JOIN seed s USING (seed_id) WHERE s.token_count > 4692` is 0 and
`SELECT COUNT(*) FROM seed WHERE token_count > 4692` is ≥ 1.

## Running

```
PYTHONIOENCODING=utf-8 python -m tuned.data.generate --config configs/data_law_v1_exp_deepseek.yaml \
    --n-workers 4 --max-batches 30
PYTHONIOENCODING=utf-8 python -m tuned.data.judge --config configs/data_law_v1_exp_deepseek.yaml \
    --n-workers 3 --max-batches 30
```

`--forever` is not used; both commands exit when the queue or the batch cap runs out. The
worktree `.env` now carries `BAI_API_KEY` (added 2026-08-26) alongside the judge-fleet keys, and
`load_dotenv_keys` resolves the worktree's own `.env`.

**Budget, from measured rates:**

| stage | bound | estimate |
|---|---|---|
| generate | b.ai rpm 8; ~33 s/call at `low`; `MAX_ATTEMPTS = 3` on gate failures → 40–80 calls | 8–12 min |
| judge A | groq/qwen3.6-27b, rpm 30 | not binding |
| judge B | cerebras/gemma-4-31b, **tpm 30k** binds before rpm 5 on ~8–9k-token judge prompts → ~3.5 calls/min | ~12 min for 40 rows |
| tiebreak | mistral-large, rpm 2, disagreements only | overlaps |
| seed + measure | | ~5 min |

≈ 28 min, inside the box. If generation runs long, the generate step is cut at its batch cap
and judging proceeds on whatever landed — a smaller n, reported as such.

Family separation on a deepseek row excludes `{deepseek, qwen, gemma}` from the tiebreak, which
is why `mistral/mistral-large-latest` was moved to the head of `routing.tiebreak` on 2026-08-25
(it keeps gpt-oss-20b — 0/10 on IPC→BNS — out of the deciding seat). This arm inherits that
ordering from the live config.

## Measurement and report

A scratch script (not committed) reads the arm store and produces
`docs/reports/2026-08-26-deepseek-validation-wave.md`, committed. It reuses the row-length
method from the 2026-08-26 report: rebuild each row exactly as `decontaminate.generated_rows`
does, render through the pinned chat template, count with the pinned tokenizer.

Sections, matching the table above: pipe health · seed gate · lengths vs projection ·
`think_max` violation rate · judge accept rate vs baseline · per-arm breakdown (`sc` /
`predex` / `tathya`) · cost (calls, tokens, wall clock) · what changes, if anything, in
`configs/data_law_v1.yaml` as a result.

The gpt-oss baseline for #5 is computed from the live store's `judgement` table (238 rows as of
2026-08-26), read-only, using the same accept rule (`eval_matched.dual_judge_decision`).

## Out of scope

- **Re-baselining the 42 failing worktree tests** that expect `cerebras` as lead generator.
  Pre-existing, from the uncommitted b.ai integration; unaffected by this arm and not fixed by
  it. The arm's own new test (`paths.py` sibling) must pass.
- **A paired A/B against cerebras** on the same seeds. The more informative design, but judging
  80 rows on gemma's rpm 5 breaks the time box. If #5 comes back ambiguous, that is the follow-up.
- **SSE streaming in `ChatClient`.** Only needed above `low`; not touched.
- **Changing `think_max`, `reasoning_effort`, or the seed reserve.** This arm *measures* whether
  they agree; any change is a separate, evidence-backed commit to the live config.

## Acceptance criteria

1. `is_live_control_workdir("data/build/exp_deepseek")` is `False`; test added and green.
2. `scripts/seed_exp_store.py` populates the arm store from the live store read-only, prints
   per-source counts including oversize seeds, and is idempotent.
3. The arm store contains ≥ 1 seed over budget and **0** tasks planned against one.
4. `generate` and `judge` both run to their batch cap or an empty queue without operator
   intervention; ≥ 90% of b.ai calls return content; every generation is deepseek; `$0` on the
   arm's `budget_ledger`.
5. The report exists with every section above filled from the arm store, and states each
   pre-registered pass line beside its measured value.
6. The live store's main file (`data/build/state/law_v1.sqlite3`) has the same size and
   mtime after the run as before it. (`-shm` may be touched by a read-only WAL reader;
   the main file may not.)
