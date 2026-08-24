# The recovery-arm probe: what the prefill buys, what it does not, and why the evaluator could not rule

Run window 2026-08-23T23:38Z – 2026-08-24T01:20Z. Branch `worktree-law-v1-data-pipeline`,
worktree `.claude/worktrees/law-v1-data-pipeline`. Executes Tasks 6–10 of
`docs/superpowers/plans/2026-08-23-recovery-arm-unblock.md`.

**Headline:** the arm ran end to end and spent **$0.00**. It produced the best
blocking-gate format yield ever measured here (75.0%) and the best accept rate
(25/60), and it isolated a mechanism the previous arm could not. The matched
evaluator returned `inconclusive` — for a reason that has nothing to do with this
arm and that no treatment arm can currently escape.

## 1. What ran and where

| | |
|---|---|
| config | `configs/data_law_v1_exp_recovery.yaml` |
| workdir | `data/build/exp_recovery` (never `data/build`) |
| generator | `cerebras/gpt-oss-120b`, free tier |
| harmony | `harmony_completions: true`, prefill `"I start from the facts. "`, **`harmony_s1_continue: false`** |
| overlay | `src/tuned/data/prompts_harmony`, `think_min: 500` |

The live control store `data/build/state/law_v1.sqlite3` was opened **read-only
throughout** and is byte-identical to its pre-run baseline:

```text
mtime 1787309490.939564   size 554532864   generation rows 1396
[('accepted', 15), ('generating', 8), ('judge_error', 34), ('judging', 43),
 ('pending', 127), ('rejected', 414), ('stale_prompt', 419)]
gold_label rows: 46      judge_threshold rows: 0
```

Every value matches the baseline captured before Task 6 and the line the plan
predicted. Nothing in this run wrote to it.

## 2. Fleet fixes, and the honest limit of what this run proves

| metric | before | this run |
|---|---|---|
| OpenAI spend | $0.3396 (`exp_harmony`) | **$0.0000** |
| `judge_parse_error` | 96 (`exp_harmony`) | **0** |
| `judge_route_error` | 235 (live) | **0** |
| `judge_error` parked | 34 (live) | **0** |

The judge pass ran entirely on free judges: `groq/qwen/qwen3.6-27b` (slot A, 46
requests), `cerebras/gemma-4-31b` (slot B, 46), `mistral/mistral-large-latest`
(tiebreak, 9). Slot A and slot B are different families (qwen vs gemma), so
`family_separation` is satisfiable **without OpenAI at all** — which is why the
gpt-5 refs were never reached and the $1.66 remainder of the $2.00 lifetime cap
is fully intact.

**What this run therefore does NOT prove.** Because gpt-5 was never called:

- the `reasoning_effort: 'minimal'` fix is **not live-verified**. Its correctness
  still rests on the static argument (gpt-5 bills reasoning against
  `max_completion_tokens`; `minimal` is the lowest effort the family accepts).
- the `ground_faithfulness` axis alias is **unexercised** — zero parse errors
  occurred, so no reply needed alias resolution.

Both fixes remain correct-by-construction and unrefuted, not confirmed.

## 3. Cohort

60 pairs, 20 each across `irac_analysis`, `drafting`, `summarization`.
`control_fingerprint`
`09e1367b60146b69bef981e896963d5cc4935be7b9b5e57b3e828f707579609b`.
Manifest at `.superpowers/sdd/law-v1-recovery_893eff3d/cohort-manifest.json`
(uncommitted — `.superpowers/` is gitignored at `.gitignore:32`).

**`statute_qa` is absent, as a data fact.** The control store holds 270
`statute_qa` tasks and **0 eligible seeds**: `statute_section_eligible` requires a
`meta_json.section_text` distinct from the seed body, and no seed in that DB
carries provision text. Filling it needs real Gazette Act bodies (the uncommitted
`gazette.py` work on `law-v1-foundation`, which holds identities only) *and*
writing to the control store. So this arm makes **no claim whatsoever about
`statute_qa`** — which is also the one stream whose live accepts went 0 → 2 under
the corrected prompts.

## 4. Format yield — and a metric that misfires

The plan's Step 4 gate compares "latest-per-task, all gates pass" against a 30.2%
floor. This arm scored **8.3%**, which fires the plan's STOP. That metric is
misleading here, because it counts `self_verification` — the single
**diagnostic** gate (`gates.py:111`), which by design does **not** block
promotion (`gates.py:1872`). Split the two:

| arm | n | all-gates | **blocking gates only** | `self_verification` fails |
|---|---|---|---|---|
| live, corrected prompts | 434 | 30.2% | — | 66% |
| `exp_harmony` | 48 | 64.6% | 68.8% | 10.4% |
| **`exp_recovery`** | 60 | **8.3%** | **75.0%** | **88.3%** |

On the gates that actually gate, this arm is the best ever run here. Per stratum:

| stratum | blocking-clean | all-gates |
|---|---|---|
| summarization | 17/20 = 85.0% | 0/20 = 0.0% |
| drafting | 15/20 = 75.0% | 3/20 = 15.0% |
| irac_analysis | 12/20 = 60.0% | 0/20 = 0.0% |

## 5. The probe's main result: prefill and s1 do different jobs

`exp_harmony` ran `harmony_prefill` **and** `harmony_s1_continue` together and
could not attribute its gains. This arm ran the prefill **alone**
(`harmony_s1_continue: false`). The result separates them cleanly:

- **The prefill buys format.** 59/60 latest traces open with
  `"I start from the facts. "`; think length p50 646, p90 1027, min 272, max 1135
  tokens; 75.0% blocking-clean.
- **The s1 `" Wait"` continue buys the self-verification ritual.** Without it,
  `self_verification` fails 88.3% — against 10.4% when it was on. Not one
  `irac_analysis` or `summarization` trace carries a verification cue.

Qualitatively the traces are otherwise exactly the accepted shape: first-person
deliberation, no IRAC headings inside think, real authority reasoned with (e.g.
*Uday Mohanlal Acharya* on the s.167(2) indefeasible-bail question), and a clean
`Issue` heading opening the answer. They match the six accepted examples in every
respect **except** the moment of doubt.

This is the actionable finding, and it cost nothing: **the reasoning ritual is
not a property of the prompt or the prefill. It comes from the s1 continue.**
The 18 Aug prompt rewrite moved IRAC placement and never moved
`self_verification` (66% live, unchanged) for the same reason.

## 6. Judge outcome

25 accepted, 19 rejected, 14 `format_parked`, 2 `pending`. 101 judgements across
46 `judge_decision` events and 3 `judge_regeneration`.

Accept rate **25/60 = 41.7%** of the cohort, **25/44 = 56.8%** of decided rows —
against `exp_harmony` 8/48 and live 15/429.

| judge | slot | n | grounding | validity | coverage |
|---|---|---|---|---|---|
| `qwen/qwen3.6-27b` | a | 46 | 3.50 | 4.04 | 4.24 |
| `gemma-4-31b` | b | 46 | 3.09 | 3.93 | 4.46 |
| `mistral-large-latest` | tiebreak | 9 | 4.67 | 5.00 | 4.89 |
| **overall** | | 101 | **3.42** | **4.08** | 4.40 |

**The kill axis moved.** Every prior arm had validity binding at 2.86–3.74. Here
validity is 3.93–4.04 and **grounding (3.42) is now the lowest axis**. Whatever
the prefill does to the trace, judges now doubt the sourcing rather than the
reasoning.

Free-tier cost: `cerebras/gpt-oss-120b` 131 requests / 169,129 completion tokens
(78.4k of 1M used in generation pass 1).

## 7. The evaluator's verdict, and why it is not about this arm

```text
decision=inconclusive synthesis=high-confidence pseudo-gold
reasons=missing-gate-data
```

`missing-gate-data` is raised when a paired unit lacks any of the 11
`HARD_FORMAT_GATES`. Measured on the selected cohort:

- **treatment: 0/60 incomplete**, 0 absent from the index — this arm's gate
  records are complete.
- **control: 60/60 incomplete.**

The cause is exact: **`prompt_echo` has zero rows in the entire control store's
`gate_result` table.** The gate was introduced in `104a0e3`; the control
generations span 2026-08-18 → 2026-08-21 and were scored before it existed.
`statutory_grounding` is additionally missing on 21 of the 60.

So `required_gates_complete` demands 11 gates and the frozen control store can
only ever supply 10. **The matched evaluator cannot return anything but
`inconclusive` against this control store — for any treatment arm, however
good.** This is a structural blocker in the pre-registration apparatus, not a
result about the prefill.

Clearing it requires re-gating the control generations against the current
`GATE_ORDER`, which means writing `gate_result` rows to the live control store —
forbidden by this plan's custody constraint and an explicit non-goal.

## 8. What this may not claim

- The 46 lockbox rows are **model-generated references, not human gold**. Any
  `decision` derived from them is a model-agreement statement, never evidence of
  legal correctness. `synthesis=high-confidence pseudo-gold` is a label about
  provenance, not a quality warrant.
- `judge_threshold` is still empty and stays empty. Nothing here calibrates a
  threshold.
- The cohort is 60 pairs, not the 80-pair contract originally pre-registered, so
  McNemar would have run on a smaller discordant pool had it run at all.
- **Nothing about `statute_qa`** (§3).
- The accept-rate comparison to `exp_harmony` and to live is **confounded by
  judge composition**: this run's judges are qwen/gemma/mistral, and the earlier
  `exp_harmony` lift was already known to be confounded by gpt-5-mini's higher
  grounding mean. Treat 41.7% as "this fleet on this cohort", not as a clean
  delta.
- The live wave is **not promoted**. `configs/data_law_v1.yaml` is untouched: it
  still has `think_min: 500`, no `harmony_*` flags, no `prompt_overlay`, and its
  gpt-5 refs still declare `family: gpt-oss`.
- 419 `stale_prompt` and 414 `rejected` tasks on the control store remain
  unregenerated against the corrected templates.

## 9. What this hands back

Three decisions, in the order they gate each other:

1. **Turn `harmony_s1_continue` back on.** §5 is the evidence: the prefill alone
   gives the best format ever measured and loses the verification ritual almost
   entirely. An arm with prefill **and** s1, at `think_min: 500`, is the obvious
   next probe and is free.
2. **Re-gate the control store, or stop pre-registering against it.** §7 is a
   hard stop on the whole matched-evaluation apparatus. Until the control
   generations carry `prompt_echo`, `eval_matched` cannot rule on anything. This
   needs an explicit operator decision about writing to the control store.
3. **Authorise the Gazette acquisition, or accept that `statute_qa` is
   unmeasurable.** 270 tasks, 0 eligible seeds (§3).

Grounding, not validity, is now the axis to attack (§6). And the $1.66 OpenAI
remainder is still unspent and available for the arm that needs it.
