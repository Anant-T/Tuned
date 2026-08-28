# Three-arm clause/cap A/B: neither lever wins clean, one breaches its own guard

2026-08-28. Two independent levers, tested against one shared, time-local
control, back to back in one sitting: a `prompt_overlay` clause aimed at the
irac_placement rehearsal pathology (E2), and a lowered bai `max_output`
ceiling aimed at token efficiency (E1). **Neither lever clears its
pre-registered bar cleanly.** The clause misses its primary threshold by a
wide margin and blows through its guard band. The cap clears its primary
threshold but also blows through its guard band, and a task-level
cross-check finds a real, uncounted cost the guard exists to catch: four
tasks that passed under the uncapped control are lost outright under the
cap.

**This measures the generation-time gate yield (`length_band`,
`irac_placement`, token spend), not judge-accepted quality.** No judge was
dispatched in any arm; nothing here speaks to whether a gate-passing row is
actually a good example.

---

## 1. Design: one shared control, two independent levers

Running four arms (control x2, treatment x2) would double the b.ai account's
exposure to its own hidden multi-upstream pool drifting between
measurements - the exact confound that made the original v4-vs-v5 comparison
uninterpretable (13h41m apart, 17.6pp gap, no way to attribute it). Instead
this design runs **three** arms - one shared control, `ctl2`, and two
treatments each graded against it - back to back, minutes apart, so both
lever measurements sit inside the same short window of upstream behaviour:

```
ctl2 (shared control) -> clause (treatment, E2) -> cap (treatment, E1)
```

`ctl2` is the current shipped state with no overlay and no limit override -
the post-revert "v4" prompt wording (`06f588a`) and the live bai
`max_output: 16384`. `clause` differs from `ctl2` in exactly one key
(`build.prompt_overlay`); `cap` differs from `ctl2` in exactly one key (the
bai model's `limits.max_output`). Both pairings are asserted line-for-line
equal outside their one intended key in
`tests/test_build_config.py::test_the_ds_ctl2_and_clause_arms_are_fenced_and_differ_only_in_the_prompt_overlay`
and `::test_the_ds_ctl2_and_cap_arms_are_fenced_and_differ_only_in_the_bai_max_output`.

## 2. Apparatus

Three isolated workdirs registered in
`src/tuned/data/paths.py::ISOLATED_WORKDIR_SIBLINGS`: `exp_ds_ctl2`,
`exp_ds_clause`, `exp_ds_cap`. Three configs, all
`configs/data_law_v1_exp_deepseek.yaml` (single-ref
`routing.generator: [bai/deepseek-v4-flash]`, the openai `$0` fence with
prices, bai `rpm: 8`, `length_band` identical to the shipped live config -
`think_max: 3000`, `total_max: 8192`) with one edit each:

- `configs/data_law_v1_exp_ds_ctl2.yaml` - `workdir: data/build/exp_ds_ctl2`.
  No overlay, no other change.
- `configs/data_law_v1_exp_ds_clause.yaml` - `workdir:
  data/build/exp_ds_clause`, `prompt_overlay:
  data/build/exp_ds_clause/prompts_clause`.
- `configs/data_law_v1_exp_ds_cap.yaml` - `workdir: data/build/exp_ds_cap`,
  the bai model's `limits.max_output` lowered `16384 -> 5000`.

### 2.1 The clause overlay

`data/build/exp_ds_clause/prompts_clause/` (gitignored under `/data/`, not
committed - shas below are the record) holds copies of all 14 current base
`gen_*.md` templates. Six carry one added clause, inserted directly after
the anchor sentence "...never opens a line with one of those four words."
and before the following word-count sentence; the other eight are
byte-identical to `src/tuned/data/prompts/`:

```
The same holds if you feel the pull to check, before you commit to the
answer, that you have covered all four parts: take that check as one line
of the prose you are already in - the issue is settled, this is the rule,
applied here it gives this, so this follows - never as a run-through under
the four labels, because a labelled run-through is a hidden first draft of
the answer, wherever in your thinking it falls. Write the issue, the rule,
the application and the conclusion for the first time in the answer itself.
```

Verified before any generation: 6/6 clause files contain the clause
verbatim, 8/8 non-target files are byte-identical to
`src/tuned/data/prompts/`, all 14 still contain "450 to 700 words of
deliberation is normal", zero CR bytes in any of the 14.

| file | sha256(12) | clause? |
|---|---|---|
| gen_drafting_v1.md | `48534e3010f5` | no (base) |
| gen_drafting_v2.md | `618b240ab03e` | no (base) |
| gen_irac_analysis_v1.md | `f2b4a76489cb` | **yes** |
| gen_irac_analysis_v2.md | `a5e62bd4bb3f` | **yes** |
| gen_irac_analysis_v3.md | `c4922e9d298c` | **yes** |
| gen_irac_analysis_v4.md | `78f0e8944ae1` | **yes** |
| gen_statute_qa_v1.md | `94e43b22bf48` | no (base) |
| gen_statute_qa_v2.md | `4d04338ba007` | no (base) |
| gen_statute_qa_v3.md | `5888a6c4461d` | no (base) |
| gen_statute_qa_v4.md | `713a9060835e` | no (base) |
| gen_summarization_v1.md | `52fdcf8dbd04` | **yes** |
| gen_summarization_v2.md | `3b9eefc64d33` | **yes** |
| gen_transition_v1.md | `113813116cfb` | no (base) |
| gen_transition_v2.md | `2f28a53e5259` | no (base) |

### 2.2 The cap

The bai request hook (`_bai_request_hook`) raises a generation call's
`max_tokens` up to the model's configured `limits.max_output`, because
`deepseek-v4-flash` bills reasoning against the same budget and emits it
first. Lowering that ceiling `16384 -> 5000` therefore becomes a hard
completion cap: a trace that has not finished reasoning by 5000 tokens
returns `finish_reason=length` with empty content, which
`_bai_response_hook` raises as a retryable `ProviderError` ("reply
truncated before any content"). Expected and pre-declared in scope for this
run, not a run failure.

## 3. Seeding and planning

Live store fingerprint checked before touching anything:
`554532864 1787309490` (matches the required value exactly). All three
stores seeded identically from the live store, read-only:

```
scripts/seed_exp_store.py --config <arm config> --from data/build/state/law_v1.sqlite3 --per-source 200 --seed 3407
```

600 seeds each (200 per source across `s3://indian-supreme-court-judgments`,
`L-NLProc/PredEx_Instruction-Tuning_Pred-Exp`,
`L-NLProc/TathyaNyaya-and-FactLegalLlama-NyayaFacts-Datasets` - the other
four live sources have zero eligible seed rows). Planning matched the
2026-08-26 deepseek validation wave's stratification exactly, one command
per source per arm:

```
tuned.data.tasks --stream synthesis --arm sc     --n 13 --source s3://indian-supreme-court-judgments                                       --mix irac_analysis=0.55,summarization=0.45
tuned.data.tasks --stream synthesis --arm predex --n 14 --source L-NLProc/PredEx_Instruction-Tuning_Pred-Exp                                --mix irac_analysis=0.55,summarization=0.45
tuned.data.tasks --stream synthesis --arm tathya --n 13 --source L-NLProc/TathyaNyaya-and-FactLegalLlama-NyayaFacts-Datasets                --mix irac_analysis=0.55,summarization=0.45
```

40 pending tasks in every arm (22 irac_analysis, 18 summarization).
Fingerprint re-checked after seeding all three and after planning all
three - unchanged both times.

Task ids are deterministic
(`sha256(seed_id|task_type|prompt_id|sample_ix)[:16]`) and independent of
`prompt_sha`/config, so the **same 40 task ids appear in all three stores** -
what makes the task-level cross-reference in §5 possible.

## 4. Run

```
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tuned.data.generate --config <arm config> --n-workers 5 --max-batches 30
```

Run back to back, `ctl2 -> clause -> cap`, no operator step between them:

| arm | window (UTC) | wall time | gap before |
|---|---|---|---|
| ctl2 | 01:05:09.385 - 01:31:47.366 | 26.6 min | - |
| clause | 01:32:50.614 - 01:51:19.614 | 18.5 min | 63 s |
| cap | 01:52:20.750 - 02:08:22.280 | 16.0 min | 61 s |

Batch totals (`claimed` = generation attempts made, including retries; not
unique tasks):

| arm | batches | claimed | gen-ok | gated-out | errors |
|---|---|---|---|---|---|
| ctl2 | 24 | 109 | 109 | 97 | 0 |
| clause | 22 | 103 | 103 | 88 | 0 |
| cap | 22 | 100 | 86 | 68 | 14 |

Every one of the 14 cap-arm errors is the expected "reply truncated before
any content" `ProviderError` - confirmed by joining `run_event` back to its
`detail_json.error` text, not assumed from the count.

## 5. Measurement

All "n" below counts **every generation attempt** (all retries), not unique
tasks - the same currency the v4-vs-v5 rerun used, since a lever that only
helps first attempts but burns more retries is not free.

### 5.1 E2 - the clause (irac_placement rehearsal)

| line | ctl2 | clause | delta | bar | verdict |
|---|---|---|---|---|---|
| **PRIMARY** irac_placement fail rate | 73.39% (80/109) | 68.93% (71/103) | **-4.46pp** | drop >=15pp | **FAIL** |
| GUARD length_band pass rate | 42.20% (46/109) | 58.25% (60/103) | **+16.05pp** | within +-5pp | **GUARD-BREACH** |

Secondaries (report only):

| metric | ctl2 | clause | delta |
|---|---|---|---|
| think est-tokens (chars/4) p50 | 2,974 | 2,418 | -556 (-18.7%) |
| think est-tokens p90 | 6,564 | 5,814 | -750 (-11.4%) |
| think est-tokens mean | 3,571 | 3,041 | -530 (-14.8%) |
| full-gate clean rate (all non-diagnostic gates) | 11.01% (12/109) | 14.56% (15/103) | +3.55pp |
| rows with a line-initial Issue/Rule/Application/Conclusion heading in `<think>` | 80/109 (73.4%) | 68/103 (66.0%) | -7.3pp |

**Reading it.** The clause did what the bonus hypothesis predicted -
reasoning got shorter across the board (p50 -18.7%, mean -14.8%) - and the
line-initial-heading rehearsal count did drop (73.4% -> 66.0%). But that
drop is nowhere near enough to move the primary gate 15 points: most
irac_placement failures on this generator are not marginal, one-heading
slips that a clause about "the pull to check" would catch - they are a
structural habit the model falls back into on the majority of calls
regardless. The clause reads as a *general de-verbosifier* more than a
*targeted fix* for the specific rehearsal pattern it names.

The guard breach compounds the reading problem rather than adding a second
independent finding: shorter thinking mechanically raises the length_band
pass rate (less of the 8192-token band spent on `<think>`), so the
population of rows entering the irac_placement comparison shifted
underneath it. The primary result should be read as "no", not as "no, but
cleanly measured" - the guard was pre-registered exactly to flag this kind
of confound, and it fired.

### 5.2 E1 - the cap (token efficiency)

| line | ctl2 | cap | delta | bar | verdict |
|---|---|---|---|---|---|
| GUARD length_band pass rate | 42.20% (46/109) | 50.00% (43/86) | **+7.80pp** | within +-3pp | **GUARD-BREACH** |
| **PRIMARY** completion tokens per length-passing row | 9,175.7 (422,084 / 46) | 8,193.4 (352,316 / 43) | **-10.71%** (improvement) | improves >=8% | **WIN** |

The primary uses the **true spend** from `budget_ledger` (which includes
tokens burned on failed attempts), not a sum over only the successful
`generation` rows. The distinction is not cosmetic: 14 truncated-before-any-
content attempts each spent the *entire* 5,000-token cap on reasoning before
failing (14 x 5,000 = 70,000 tokens, exactly the gap between
`budget_ledger`'s completion-token total for cap, 352,316, and the sum over
its 86 successful `generation` rows, 282,316). Computing the ratio from the
generation-row sum alone gives 6,565.5 tok/row and a headline **28.45%**
"improvement" - a real number, but one that hides the cap's own wasted spend
inside a smaller total. 10.71% is the honest figure and it still clears the
8% bar, but by roughly a third of the naive margin.

**Truncation count**: 14 "reply truncated before any content" errors
(`ProviderError`, retryable, no `generation` row created), landing on 9
distinct tasks. Separately, 12 "generation_truncated" events fired -
`finish_reason=length` calls that *did* produce non-empty content and a
`generation` row, flagged with the synthetic `truncated` gate. These are a
related but distinct phenomenon from the 14 above (softer, mid-stream
cutoffs rather than reasoning consuming the whole budget) and are already
folded into the 86-row gate statistics above; they are not double-counted
against the 14.

**Task-level cross-reference** (same 40 task ids in every arm, by
construction - see 3): of the 9 tasks that hit >=1 truncation-before-content
error in `cap`, 6 had passed `length_band` in the `ctl2` control:

| task_id | task_type | passed length_band in ctl2 | recovered in cap (later attempt) | final state in cap |
|---|---|---|---|---|
| `1fc960c376a05464` | irac_analysis | no | yes | format_parked |
| `2b75d28c3aa5bda9` | irac_analysis | no | no | format_parked |
| `4f69d55611f115c1` | summarization | **yes** | yes | judging |
| `540bd5a01a08c307` | irac_analysis | **yes** | no | **gen_unroutable** |
| `70bb5d8b897d3727` | summarization | **yes** | no | **gen_unroutable** |
| `a169a11943012a3f` | irac_analysis | **yes** | no | **gen_unroutable** |
| `eebf5b6fca33ac00` | irac_analysis | no | no | format_parked |
| `f129e1685c91c1e6` | irac_analysis | **yes** | no | **gen_unroutable** |
| `f67ae9b41f362637` | irac_analysis | **yes** | yes | judging |

Of the 6 control-passing tasks caught by a truncation, only 2 recovered on a
later attempt; the other **4 exhausted all 3 attempts on truncation and
parked `gen_unroutable` (`exhausted:provider-fault`) - zero output** for a
task the uncapped control answered cleanly. That is 4 of the arm's 40
planned tasks (10%) converted from a usable row to nothing, a real cost the
pre-registered lines do not price in and the guard breach above is the
symptom of.

## 6. Integrity

- Every generation in every arm is `deepseek-v4-flash`: ctl2 109/109,
  clause 103/103, cap 86/86 - confirmed by `SELECT DISTINCT model FROM
  generation` per store, not sampled.
- `budget_ledger` in every arm store names exactly one provider, `bai` -
  confirmed the same way.
- Zero HTTP 429s across all three arms (`run_event` has no `429`/rate-limit
  kind in any store).
- Live store fingerprint `554532864 1787309490` - checked before touching
  anything, after seeding all three stores, after planning all three, and
  after each of the three generation runs (six checks total). Identical
  every time.
- No `.env` contents were printed at any point; all three `.venv` invocations
  logged only `loaded 18 key(s) from .env`.

## 7. Run economics

| arm | prompt tok | completion tok | total tok | 429s | wall time |
|---|---|---|---|---|---|
| ctl2 | 277,571 | 422,084 | 699,655 | 0 | 26.6 min |
| clause | 269,555 | 352,010 | 621,565 | 0 | 18.5 min |
| cap | 249,073 | 352,316 | 601,389 | 0 | 16.0 min |

All three ran on b.ai's free tier - $0 spend, as with every deepseek arm to
date.

## 8. Verdicts

- **Clause lever (E2): FAIL.** Primary drop -4.46pp against a >=15pp bar.
  Guard breached (+16.05pp, over 3x the +-5pp band). The clause shortens
  reasoning generally but does not suppress the targeted rehearsal pattern
  enough to move the gate that matters. Do not ship.
- **Cap lever (E1): WIN on primary (-10.71% tokens/passing-row, true
  spend), but GUARD-BREACHED (+7.80pp length_band, over 2.5x the +-3pp
  band), and a real cost the pre-registered lines miss: 4 of 40 tasks
  (10%) that passed under the uncapped control are lost outright
  (`gen_unroutable`) under the cap.** The efficiency gain is real but is not
  a clean win - it trades a meaningful slice of usable rows for a
  double-digit token saving on the rows that survive. Recommend against
  shipping 5000 as-is; a looser cap (nearer the E1-offline floor
  measurements) or a raised per-task attempt budget for this failure mode
  would be the next thing to test before either lever is revisited.

**Gate yield measured, judge quality not.**

## 9. Tests

```
./.venv/Scripts/python.exe -m pytest tests/test_build_config.py tests/test_build_paths.py -q -p no:cacheprovider --basetemp=data/build/exp_ds_ctl2/pt
```

156 passed, 0 failed. Full suite not run (not required - no `src/` behaviour
change beyond `ISOLATED_WORKDIR_SIBLINGS` registration).
