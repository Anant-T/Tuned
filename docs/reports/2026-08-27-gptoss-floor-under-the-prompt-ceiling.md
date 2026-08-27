# gpt-oss under the prompt ceiling: measured, and it fails four of five lines

2026-08-27/28. The gpt-oss floor A/B was blocked earlier the same day by a
cerebras account returning HTTP 402 to every call. A newly funded cerebras key
was swapped into the worktree, both arms were reopened from their parked state
and run back to back in one sitting, and all five pre-registered measurements
completed.

**Result: 1 PASS / 4 FAIL.** The prompt-ceiling edit measurably harms
`cerebras/gpt-oss-120b`, the generator this branch put back in the lead the
same day. Section 5 says this plainly and does not soften it, per the brief.

---

## 1. What this measures

On 2026-08-27, commit `286fd3a` gave the reasoning band a hard ceiling in all
fourteen generator templates, deleting a sentence added on 2026-08-18 for the
opposite purpose - to push gpt-oss traces up, because gpt-oss's measured
failure mode is the floor (`think < think_min`), not the roof:

> ...450 to 700 words of deliberation is normal for a matter of any
> substance. -> ...450 to 700 words of deliberation, and 700 is a ceiling
> you do not cross (fourteen surface forms of the same sentence).

Counted over the 1,281 gpt-oss generations already in the live store -
context only, not this report's control - median `think_est` was 620
tokens and `think < think_min` already ran 381/1,281 = 29.7%. The edit was
measured on deepseek and came out inert
(`docs/reports/2026-08-27-generator-prompt-length-fix.md`); it had never been
measured on gpt-oss. Task 1 of the 2026-08-27 plan then made gpt-oss the lead
generator again. That combination - the generator whose dominant failure is
`think < think_min`, reading templates that just lost the sentence written to
raise it - is what these two arms exist to grade.

## 2. The blocker, and what changed

The apparatus (both arm configs, both seeded stores, the pre-edit prompt
overlay) was built and committed as `4113549` earlier in the day, but every
call to `cerebras/gpt-oss-120b` returned `HTTP 402 payment_required` - the
account was out of credit. Both arms produced zero generations and parked
30/40 tasks each as `gen_unroutable` (disposition
`exhausted:unroutable:cooling`); the remaining 10/40 in each arm never got
claimed before the single-ref generator pool cooled.

A newly funded `CEREBRAS_API_KEY` was placed in the worktree `.env` and
independently verified (HTTP 200 on both `gpt-oss-120b` and `gemma-4-31b`)
before this run began. No value from `.env` is reproduced anywhere in this
report, the commands run, or their output.

## 3. What was reopened and run

Neither arm needed rebuilding, reseeding or re-verifying - the prior
implementation's apparatus (arm configs, seeded stores, pre-edit overlay,
config pairing, `$0` fence) stands as committed in `4113549` and was reused
as-is. Both arm configs' `length_band` was confirmed to match the shipped
`configs/data_law_v1.yaml` (`think_max: 3000`, `total_max: 8192`) before
running, per the resolved ambiguity in the task brief.

```
./.venv/Scripts/python.exe -m tuned.data.tasks \
    --config configs/data_law_v1_exp_gptoss_ctl.yaml --reopen gen_unroutable
./.venv/Scripts/python.exe -m tuned.data.tasks \
    --config configs/data_law_v1_exp_gptoss_new.yaml --reopen gen_unroutable
```

Both re-opens moved exactly 30 `gen_unroutable` -> `pending` (0 out of
attempt-budget skips - `exhausted:unroutable:cooling` is a free park), leaving
`pending=40` in both stores, matching the original 40-task plan. Then, back to
back with no operator step between them:

```
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tuned.data.generate \
    --config configs/data_law_v1_exp_gptoss_ctl.yaml --n-workers 3 --max-batches 30
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tuned.data.generate \
    --config configs/data_law_v1_exp_gptoss_new.yaml --n-workers 3 --max-batches 30
```

### Both arms, as run

| | control (pre-edit templates) | treatment (shipped templates) |
|---|---|---|
| tasks planned | 40 | 40 |
| batches | 30 | 30 |
| generations recorded | 90 (claimed 90, gen-ok 90, errors 0) | 90 (claimed 90, gen-ok 90, errors 0) |
| attempts per generation | 1: 33, 2: 29, 3: 28 | 1: 31, 2: 30, 3: 29 |
| final task states | `format_parked` 28, `judging` 2, `pending` 9, `rejected` 1 | `format_parked` 29, `judging` 1, `pending` 10 |
| every generation's (provider, model) | `(cerebras, gpt-oss-120b)`, 90/90 | `(cerebras, gpt-oss-120b)`, 90/90 |
| total tokens (`generation.total_tokens`, matches the CLI's own reported run total) | 373,223 | 368,802 |

`judging` is out of scope for this measurement (generation and gate rates
only) and no judge was dispatched in either arm.

### Inter-arm timing

```
control    first generation 2026-08-27T18:24:02.542Z   last 2026-08-27T18:44:19.877Z
treatment  first generation 2026-08-27T18:44:32.747Z   last 2026-08-27T19:05:39.673Z
```

Inter-arm gap: about 13 seconds (last control generation to first treatment
generation). Well inside the "minutes acceptable, hours not" bound; the
deepseek A/B's 13h41m confound is not present here. Each arm ran roughly
20-21 minutes, matching the `rpm: 5 / tpm: 30000` budget estimate.

## 4. The five pre-registered measurements

Computed by a scratch script
(`data/build/exp_gptoss_new/out/report_gptoss_floor.py`, not committed -
`data/` is gitignored) against both arm stores opened strictly read-only.
Population: every generation with content, all attempts pooled - the
population the pass lines were fixed against, matching the methodology of
`data/build/exp_prompt_v5/out/report_ab.py`. No generation in either arm had
an empty `think` or `answer`, so pooled-all-attempts and "with content" are
the same 90-row population in both arms.

| # | measurement | pass line | control (n) | treatment (n) | verdict |
|---|---|---|---|---|---|
| 1 | `think < think_min` breach rate | treatment <= control + 5pp | 44.4% (n=90) | 57.8% (n=90) | FAIL (+13.4pp, allowance was +5pp) |
| 2 | median trace words | treatment >= 400 absolute | 324 words (n=90) | 292 words (n=90) | FAIL |
| 3 | `length_band` pass rate | treatment >= control - 5pp | 55.6% (n=90) | 42.2% (n=90) | FAIL (-13.4pp, allowance was -5pp) |
| 4 | `self_verification` pass rate | treatment >= control - 5pp | 31.1% (n=90) | 25.6% (n=90) | FAIL (-5.5pp, allowance was -5pp) |
| 5 | every generation `gpt-oss-120b`; `$0` ledgered | hard | 90/90 model-correct; `budget_ledger` = (cerebras, gpt-oss-120b, prompt=237376, completion=135847); no non-cerebras rows | 90/90 model-correct; `budget_ledger` = (cerebras, gpt-oss-120b, prompt=235276, completion=133526); no non-cerebras rows | PASS |

1 PASS / 4 FAIL. Measurement 5's ledger `$0` holds because
`cerebras/gpt-oss-120b` carries no `usd_per_1m_prompt` / `usd_per_1m_completion`
key in either arm config; `generate._usd_per_1m` returns `0.0` for a missing
price regardless of token volume, verified directly against both arm configs'
`cerebras` provider block (no `usd_per_1m_*` keys present on the `gpt-oss-120b`
model). This is a genuine, non-vacuous PASS this time - n=90 in each arm, not
the empty-population vacuous case the prior (blocked) attempt correctly
refused to call PASS.

### Robustness check: first attempt only (not pass-line-registered, reported for context)

The pooled population above retries a task up to the overgeneration budget
(3) when it fails its gates, so a task that keeps failing contributes up to 3
rows and a passing task contributes 1 - a real composition effect, in both
arms, not necessarily the same size in each. Restricting to attempt 1 gives
one row per task, the same 40 seeds and prompt variants in both arms:

| measurement | control, attempt 1 (n=33) | treatment, attempt 1 (n=31) |
|---|---|---|
| `think < think_min` breach rate | 33.3% | 61.3% |
| `length_band` pass rate | 66.7% | 38.7% |
| `self_verification` pass rate | 24.2% | 32.3% |
| median trace words | 368 | 284 |

Measurements 1-3 move the same direction and by a larger margin on
first-attempt-only than pooled - the composition effect, if anything,
understated the treatment's degradation on those three. `self_verification`
flips sign (treatment higher on first-attempt-only, lower pooled) at n=31-33
per arm; at that sample size a single-digit-count flip is well within noise
and is flagged as exactly that, not as a second finding.

## 5. What this means for the prompt edit

The edit harms the generator that is now lead. This is not a soft
reading: three of the four gate-rate measurements moved against the
treatment by more than their allowed tolerance, in the direction the risk
this task was commissioned to test predicted - `think < think_min` breach
rate rose 13.4 points past its +5pp allowance, `length_band` pass rate fell
13.4 points past its -5pp allowance, and `self_verification` pass rate fell
0.5 points past its -5pp allowance. Median trace words fails its own absolute
floor in both arms, control included (see caveat 2 below).

Per the brief's framing: the choice is now between reverting the 286fd3a
ceiling edit and reconsidering gpt-oss's demotion back to lead generator.
This report does not make that choice - it was scoped to measurement, not to
routing policy - but it records that the specific risk the edit created and
was never tested against is now measured, and it is real.

Measurement 5's PASS should not be read as absolving the other four. The
`$0` / model-identity fence is a hard operational guard, not a quality
signal; it being clean says the measurement is trustworthy, not that the
finding is good news.

## 6. Caveats

1. Different cerebras account than every prior gpt-oss measurement on this
   branch. The account behind today's key is not the account that produced
   the 1,281-row live-store baseline (median `think_est` 620 tokens, 29.7%
   `think < think_min`) or the account the blocked attempt probed and found
   402'd - that account's credit was exhausted and a new key was substituted
   the same day. Same endpoint (`https://api.cerebras.ai/v1`), same model id
   (`gpt-oss-120b`), same `params` block (`temperature 0.7, top_p 0.95,
   reasoning_effort medium`) in both arm configs - but account-level upstream
   differences (routing to a different backing deployment, different
   capacity-driven variance) cannot be ruled out from this repo's side. Both
   arms in this A/B ran on the same new account back to back, so the
   within-run comparison (control vs. treatment) is not confounded by the
   account swap - both sides of the fraction moved together. What the swap
   does put in question is any comparison of these numbers against the
   1,281-row live-store baseline or against the blocked attempt's account:
   those are cross-account and are not load-bearing to the verdict in
   section 5, which rests entirely on control vs. treatment within this run.
2. Measurement 2's absolute floor (400 words) is not cleared by either arm,
   including the pre-edit control (324 words). The 620-token figure quoted
   for the 1,281-row live population in section 1 is a token count on a
   different (much larger, mixed-provenance) population; this report's word
   count is a whitespace split on the raw `think` string, per the pass line
   as written and per `report_ab.py`'s established convention. The two units
   are not directly convertible without knowing the tokens-per-word ratio
   realized on this specific sample, and this report does not attempt that
   conversion. The measurement is reported and fails exactly as registered,
   on both arms, without adjustment.
3. n=40 tasks / 90 generations per arm is set by the seeding step from the
   earlier build (`--per-source 200 --seed 3407`, 13/14/13 across
   sc/predex/tathya), unchanged in this resume. It is the population the
   apparatus was built against, not a fresh choice made for this run.
4. Judging is out of scope. No judge was dispatched in either arm; the
   `judging`-state rows (2 control, 1 treatment) sit unresolved. The five
   measurements above are generation- and gate-rate-only, as scoped.
5. `gemma-4-31b`'s restoration is reported, not independently re-verified
   by this task. The task instructions state the new key was verified
   HTTP 200 on both `gpt-oss-120b` and `gemma-4-31b` minutes before this run
   started; this task did not re-probe `gemma-4-31b` or dispatch a judge, so
   the earlier finding that judge slot B fails over to a paid model when
   `gemma-4-31b` is down is not re-tested here one way or the other.
6. Pre-existing, untouched: `configs/data_law_v1_exp_measure.yaml`
   (untracked) and
   `docs/superpowers/plans/2026-08-24-judge-calibration-and-yield.md`
   (modified in the working tree) are neither reviewed nor staged by this
   task.

## 7. Live control store: untouched

Opened read-only (`file:...?mode=ro`) throughout; never opened for write.
Fingerprinted before the reopen and after both generation runs completed:

```
before   554532864  1787309490
after    554532864  1787309490
sha256   2ea51e4c996273fbee6d79ee1d632b6677c8752d50cb9f45258370f07fcc8f48   (both before and after)
```

Size, mtime and full content hash identical, and identical to the values
recorded by the blocked attempt earlier the same day - the store has not
been touched since before this measurement began.

## 8. What remains unmeasured

- Judge-side quality (`self_verification`'s pooled and first-attempt readings
  disagree on direction at small n; nothing here adjudicates which reading
  a live judge would agree with).
- Any matched-cohort or downstream training-signal evaluation - this is a
  gate-rate arm, not an eval arm, by design (see the arm configs' header
  comments).
- Whether reverting `286fd3a` recovers the control-arm rates on a shipped
  (not overlay) config - this report compares the pre-edit overlay against
  the shipped base; it does not test an actual revert-and-reship.
- The judge-slot-B / `gemma-4-31b` question from the blocked attempt's
  section 6, per caveat 5 above.

## 9. Files

| | |
|---|---|
| control config | `configs/data_law_v1_exp_gptoss_ctl.yaml` (unchanged from `4113549`) |
| treatment config | `configs/data_law_v1_exp_gptoss_new.yaml` (unchanged from `4113549`) |
| pre-edit templates | `data/build/exp_gptoss_ctl/prompts_preedit/` (14 files, from `f499372`, unchanged) |
| measurement script | `data/build/exp_gptoss_new/out/report_gptoss_floor.py` (uncommitted; `data/` is gitignored) |
| live-store fingerprints | recorded in section 7 above |
