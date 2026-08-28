# deepseek prompt-era rerun: v4 vs v5, measured back to back - a wash

2026-08-27/28. Re-runs the v4-vs-v5 deepseek generation-yield comparison with
the one variable the original run could not control: timing. Both
pre-registered measurements landed, both arms are integrity-clean, and the
**primary line is a WASH** - keep the shipped (v5) prompt templates. A
first-attempt-only robustness cut disagrees sharply and is reported honestly
below, with the reason it should not overturn the primary call.

**This measures the generation-time `length_band` gate (gate yield), not
judge-accepted quality.** No judge was dispatched in either arm; nothing here
speaks to whether a length-band-passing row is actually a good example.

---

## 1. The confound this rerun removes

The original v4-vs-v5 comparison ran its two arms **13h41m apart**, against
b.ai's documented hidden multi-upstream pool (multiple upstream instances
sit behind one model id - see the b.ai qualification notes). Its result -
v4 (pre-edit templates) pass 49.5% (n=99) vs v5 (shipped templates) pass
31.9% (n=94) - is uninterpretable: a 17.6pp gap measured across two
different points in time could be the prompt edit, or it could be two
different upstream instances answering the same prompt differently. Nothing
in that run controls for which.

This rerun puts both arms through the same apparatus, back to back, with the
gap between the first arm's last call and the second arm's first call
measured in *seconds*, not hours - removing the confound rather than arguing
around it.

## 2. Apparatus

Two isolated workdirs, `data/build/exp_ds_v4rerun` and
`data/build/exp_ds_v5rerun`, registered in
`src/tuned/data/paths.py::ISOLATED_WORKDIR_SIBLINGS` alongside the existing
`exp_gptoss_ctl`/`exp_gptoss_new`/`exp_deepseek` siblings. Three new tests in
`tests/test_build_config.py` mirror the existing gpt-oss-floor-arm pattern
exactly: two isolated-workdir tests and one combined fenced+paired-config
test.

Two arm configs, both `configs/data_law_v1_exp_deepseek.yaml` (the deepseek
generator arm: single-ref `routing.generator: [bai/deepseek-v4-flash]`, the
openai `$0` fence with prices, `bai` `rpm: 8`, `length_band` identical to the
shipped live config - `think_max: 3000`, `total_max: 8192`) with only the
workdir (and, for v4rerun, `build.prompt_overlay`) changed:

- `configs/data_law_v1_exp_ds_v4rerun.yaml` - workdir
  `data/build/exp_ds_v4rerun`, `prompt_overlay:
  data/build/exp_gptoss_ctl/prompts_preedit`.
- `configs/data_law_v1_exp_ds_v5rerun.yaml` - workdir
  `data/build/exp_ds_v5rerun`, no overlay (reads shipped
  `src/tuned/data/prompts/`).

The two files are asserted line-for-line identical outside `workdir` and
`prompt_overlay` (`test_the_ds_rerun_arms_are_fenced_and_differ_only_in_the_prompt_overlay`).

### Overlay verification (reused, not rebuilt)

The gpt-oss floor arm had already built the pre-edit template snapshot at
`data/build/exp_gptoss_ctl/prompts_preedit/` (14 `gen_*.md` files, the state
at `f499372` before `286fd3a` added the reasoning ceiling). This rerun
**reuses that directory directly** rather than duplicating it - both
`configs/data_law_v1_exp_ds_v4rerun.yaml` and the gpt-oss control arm point
at the same path on disk.

Verified before seeding:

```
grep -lc "450 to 700 words of deliberation is normal" prompts_preedit/gen_*.md | wc -l  -> 14
grep -l  "700 is a ceiling"                          prompts_preedit/gen_*.md          -> (empty)
```

**14/14** contain the pre-edit sentence fragment; **0/14** contain the
post-edit ceiling text.

### Seeding (identical inputs, both arms)

```
./.venv/Scripts/python.exe scripts/seed_exp_store.py --config configs/data_law_v1_exp_ds_v4rerun.yaml \
    --from data/build/state/law_v1.sqlite3 --per-source 200 --seed 3407
./.venv/Scripts/python.exe scripts/seed_exp_store.py --config configs/data_law_v1_exp_ds_v5rerun.yaml \
    --from data/build/state/law_v1.sqlite3 --per-source 200 --seed 3407
```

Same `--seed`/`--per-source` against the same live store: both stores came
back with **600 seeds**, identically distributed -
`s3://indian-supreme-court-judgments` 200 (1 over the 4,692-token budget),
`L-NLProc/PredEx_Instruction-Tuning_Pred-Exp` 200 (13 over),
`L-NLProc/TathyaNyaya-and-FactLegalLlama-NyayaFacts-Datasets` 200 (21 over);
the other four live sources have zero eligible seed rows and copied 0 in
both stores.

### Planning (matching the prior deepseek arm's stratification)

Same 3-source stratified plan the 2026-08-26 deepseek validation wave used
(the planner's default order over-draws the shortest source otherwise),
run identically against both arm configs:

```
tuned.data.tasks --stream synthesis --arm sc     --n 13 --source s3://indian-supreme-court-judgments                                       --mix irac_analysis=0.55,summarization=0.45
tuned.data.tasks --stream synthesis --arm predex --n 14 --source L-NLProc/PredEx_Instruction-Tuning_Pred-Exp                                --mix irac_analysis=0.55,summarization=0.45
tuned.data.tasks --stream synthesis --arm tathya --n 13 --source L-NLProc/TathyaNyaya-and-FactLegalLlama-NyayaFacts-Datasets                --mix irac_analysis=0.55,summarization=0.45
```

Both stores planned **40 pending tasks** (13+14+13). Seed gate held in both:
oversize seeds present **35** (need >=1), tasks planned against an oversize
seed **0** (need 0) - the gate is what keeps the planner from ever routing a
seed the generator's own budget cannot hold.

## 3. Both arms, as run

```
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tuned.data.generate \
    --config configs/data_law_v1_exp_ds_v4rerun.yaml --n-workers 5 --max-batches 30
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tuned.data.generate \
    --config configs/data_law_v1_exp_ds_v5rerun.yaml --n-workers 5 --max-batches 30
```

`--n-workers 5` is the approved efficiency rider (the prior deepseek wave
ran 4). Generate only - no judge dispatched in either arm, consistent with
this being a gate-yield measurement rather than a quality eval.

| | v4rerun (pre-edit overlay) | v5rerun (shipped) |
|---|---|---|
| tasks planned | 40 | 40 |
| batches to exhaustion | 22 | 23 |
| generations recorded | 97 (claimed 97, gen-ok 97, errors 0) | 106 (claimed 106, gen-ok 106, errors 0) |
| final task states | `format_parked` 23, `judging` 16, `rejected` 1, `pending` 0 | `format_parked` 30, `judging` 9, `rejected` 1, `pending` 0 |
| every generation's model | `deepseek-v4-flash`, 97/97 | `deepseek-v4-flash`, 106/106 |
| `budget_ledger` rows | only `(bai, deepseek-v4-flash, req=97, 429=0, prompt=246814, completion=374054)` | only `(bai, deepseek-v4-flash, req=106, 429=0, prompt=262877, completion=446913)` |
| wall time (`generation.created_at` span) | 20.48 min (`22:23:56.856682Z` - `22:44:25.510970Z`) | 20.06 min (`22:45:21.693025Z` - `23:05:25.086593Z`) |

### Inter-arm timing

```
v4rerun   last generation   2026-08-27T22:44:25.510970Z
v5rerun   first generation  2026-08-27T22:45:21.693025Z
```

**Gap: 56.18 seconds** - well inside "minutes, not hours," and three orders
of magnitude tighter than the original run's 13h41m confound.

## 4. The pre-registered primary measurement

**`length_band` pass rate over ALL generations** (fixed before the run, pass
lines cannot move):

| arm | n | pass | rate |
|---|---|---|---|
| v4rerun (pre-edit) | 97 | 46 | **47.42%** |
| v5rerun (shipped) | 106 | 45 | **42.45%** |

**Delta (v4 - v5): +4.97pp.**

| pass line | threshold | result |
|---|---|---|
| `v4 >= v5 + 10pp` -> "ceiling edit harmful, REVERT recommended" | delta >= +10pp | not met (+4.97pp) |
| `\|v4-v5\| < 5pp` -> "wash, keep shipped templates" | delta < 5pp | **MET** (4.97 < 5) |
| between -> "inconclusive, judgment call" | 5pp <= delta < 10pp | not applicable |

**The primary line is a WASH.** The measured gap (4.97pp) sits just under
the 5pp wash threshold.

### Robustness check: first-attempt only (n=40 both arms, not pass-line-registered)

The pooled population above includes up to 3 attempts per task
(`overgeneration`-budgeted retries on gate failure). Restricting to attempt 1
gives exactly the 40 originally-planned tasks per arm, one row each:

| arm | n | pass | rate |
|---|---|---|---|
| v4rerun, attempt 1 | 40 | 19 | 47.5% |
| v5rerun, attempt 1 | 40 | 14 | 35.0% |

**Delta: +12.5pp** - which, taken alone, would cross the +10pp REVERT
threshold.

**This does not overturn the primary call.** At n=40 per arm with rates near
40-50%, the standard error on a single proportion is roughly 8pp and the
combined SE on the difference of two independent proportions this size is
roughly 11pp - so a 12.5pp gap at this sample size is about 1.1 SE from
zero, not a result that would itself clear significance. The pooled
measurement (n=97/106) has a materially smaller combined SE (roughly 7pp)
and is the more reliable of the two by sample size alone. The two cuts
disagreeing is therefore reported as exactly that - a data point consistent
with ordinary sampling noise at this scale, not evidence of a retry-driven
composition effect - and the pre-registered primary line (computed over ALL
generations) is what this report's verdict rests on, per the protocol fixed
before the run.

## 5. Secondary measurements (reported, no pass line)

Per-edge `length_band` failures, over ALL generations:

| edge | v4rerun (n=97) | v5rerun (n=106) |
|---|---|---|
| `think < think_min` (500) | 4 (4.1%) | 1 (0.9%) |
| `think > think_max` (3000) | 44 (45.4%) | 54 (50.9%) |
| `total > total_max` (8192) | 39 (40.2%) | 49 (46.2%) |

`think`/`total` estimated-token percentiles (the gate's own chars/4 estimate,
`gate_result.detail_json`, not a real tokenizer count):

| series | v4rerun p50 | v4rerun p90 | v5rerun p50 | v5rerun p90 |
|---|---|---|---|---|
| `think_est` | 2,633 | 5,860 | 3,101 | 7,617 |
| `total_est` | 7,437 | 9,985 | 7,756 | 11,535 |

Tokens per length-passing row (real `bai` prompt+completion tokens / count of
`length_band`-passing generations):

| arm | bai tokens | length-passing rows | tokens/passing row |
|---|---|---|---|
| v4rerun | 620,868 | 46 | **13,497** |
| v5rerun | 709,790 | 45 | **15,773** |

Full-gate pass rate (generation-level: every non-diagnostic gate in
`gates.GATE_ORDER` clean, `self_verification` excluded per
`gates.DIAGNOSTIC_GATES`, matching `gates.disposition`):

| arm | n | full-gate clean | rate |
|---|---|---|---|
| v4rerun | 97 | 16 | **16.49%** |
| v5rerun | 106 | 9 | **8.49%** |

The full-gate rate favors v4 by a wider margin than the length-band-only
primary line - consistent with v4's slightly lower `total>total_max` and
`think>think_max` breach rates feeding through to fewer downstream rejects on
gates this report does not otherwise measure (`irac_placement`,
`statutory_grounding`, etc.). This is descriptive context, not a
pre-registered line, and is not the basis for the verdict below.

## 6. Run economics

| | v4rerun | v5rerun |
|---|---|---|
| `bai` requests | 97 | 106 |
| `429`s | 0 | 0 |
| "reply truncated before any content" (retryable `_bai_response_hook` path, `run_event.kind='generation_error'`) | 0 | 0 |
| real tokens (prompt+completion) | 246,814 + 374,054 = 620,868 | 262,877 + 446,913 = 709,790 |
| wall time | 20.48 min | 20.06 min |
| tokens/min | **30,319** | **35,389** |

**`--n-workers 5` observation.** Zero 429s across 203 combined requests, at 5
workers against the `rpm: 8` request-counted bucket (up from the prior
deepseek wave's 4 workers). Both arms' batch logs show steady 5-per-batch
claims with no stall or backoff pattern, consistent with the bucket - not
worker concurrency - being the binding constraint on throughput, as the
`bai` provider block's measured-limits comment describes. The efficiency
rider cost nothing here.

## 7. Integrity (hard requirements)

- **Every generation `deepseek-v4-flash`**: v4rerun 97/97, v5rerun 106/106. **PASS.**
- **$0 non-`bai` spend**: `budget_ledger` in both arm stores contains exactly
  one provider row (`bai`) - no judge was dispatched in either arm, so no
  `groq`/`cerebras`/`mistral`/`openai` rows exist to spend anything. **PASS.**
- **Live control store never opened for write**: fingerprint
  `554532864 1787309490`, checked before seeding, after seeding both arms,
  after planning both arms, and after both generation runs completed -
  identical every time. **PASS.**

## 8. Verdict

**Per the pre-registered primary line, computed over ALL generations and
fixed before this run: the gap is 4.97pp, under the 5pp wash threshold. Verdict:
WASH - keep the shipped (v5) prompt templates.** Once the 13h41m cross-upstream
confound is removed and both arms are measured back to back, the
prompt-ceiling edit is not shown to harm `deepseek-v4-flash`'s generation-time
`length_band` gate yield. The first-attempt-only robustness cut shows a
larger gap that alone would suggest reverting the edit, but at n=40 per arm
that gap is within its own sampling noise and does not carry the weight the
pooled, larger-n primary measurement does; it is reported for the record; it
is not grounds to overturn the pre-registered call above. This measures
generation-time gate yield, not judge-accepted quality - nothing here speaks
to whether a length-band-passing row under either template era is actually a
good training example.
