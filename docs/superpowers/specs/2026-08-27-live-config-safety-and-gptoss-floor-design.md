# Live-config safety fix, and the gpt-oss floor measurement — design

**Date:** 2026-08-27
**Status:** approved
**Branch:** `worktree-law-v1-data-pipeline`

Two deliverables. Part A closes a live cost path and corrects a routing decision this branch
made without saying so. Part B measures the one risk the branch created and never tested.

## Part A — three changes to `configs/data_law_v1.yaml`

### A1. The cost fence. This is the urgent one.

`openai/gpt-5-mini` and `openai/gpt-5-nano` are declared `family: gpt-oss` **deliberately**, so
that `routing.family_separation` excludes them whenever the generator is gpt-oss. That
declaration was the entire mechanism keeping paid judges unreachable.

This branch put `bai/deepseek-v4-flash` at the head of `routing.generator`. On a deepseek
generation, separation excludes `{deepseek}` — and the paid judges are no longer excluded. They
sit at judge positions 3 and 4, and tiebreak positions 4 and 5.

**The live config has no `usd_cap` anywhere.** Every experiment arm carried one; the live config
never needed one while gpt-oss led. The failover condition is known to occur: the 2026-08-23
live drain stalled on a slot-B pool gap with 34 `judge_error`s, which is exactly the state in
which the pool reaches position 3.

Add to **both** gpt-5 models:

```yaml
        limits: {rpm: 500, tpm: 200000, max_context: 400000, max_output: 16384,
                 usd_cap: 0.0, usd_per_1m_prompt: 0.25, usd_per_1m_completion: 2.0}
```

**The prices are load-bearing and a bare cap is worthless.** `generate._usd_per_1m` returns
`0.0` for a missing key, so `usd_cap: 0.0` alone computes `0 + 0 > 0.0`, which is `False`, and
blocks nothing. This exact trap was verified live on the experiment arms: with prices, a
1,000-token call computes `0.000250 > 0.0` and BLOCKS; with prices stripped, the identical call
passes. Verify it the same way here, against the shipped config, before trusting it.

`gpt-5-nano`'s real prices differ from mini's, but the cap is zero — any positive price blocks
at the first token. Use mini's figures for both and say so in a comment rather than implying a
nano price was looked up.

### A2. Demote deepseek from lead generator

```yaml
  generator: [cerebras/gpt-oss-120b, bai/deepseek-v4-flash,
              lightning/lightning-ai/gpt-oss-120b]
```

Three reasons, in order of weight:

1. **It is not what deepseek was qualified for.** The 2026-08-25 qualification recommended it as
   a judge/tiebreak provider — a new family against the gpt-oss/gemma/qwen/mistral pool, which
   is what the stalled slot-B judge pool needed — and as a *supplementary* generator only. It
   currently ships `roles: [generator]`, which is the inverse.
2. **Measured yield.** `length_band` passes 49.5% of deepseek generations at `think_max: 3000`,
   and 60.6% even after A3. See the saturation table under A3: about a third of deepseek rows
   do not fit the student's context at any `think_max`.
3. **Generator throughput is not the bottleneck.** The judge pool is — slot A (`groq/qwen`) has
   `tpd: 200000`, roughly 25-33 judge calls a day, and end-to-end throughput was measured at
   ~100 rows/day. A generator that is faster than the judge pool buys nothing.

Leave `routing.tiebreak` in its current order (mistral first). It was reordered because
gpt-oss-20b must not hold the deciding seat on a deepseek row, and deepseek remains ref 2, so
the reason still holds.

### A3. `length_band.think_max: 3000 -> 4000`

This stands on its own merits and is not a concession to any generator: at 3000 the gate
rejects rows whose reasoning is longer than 3000 tokens but whose **total still fits the
student's 8192-token context**. Those rows are trainable and are being thrown away.

Counterfactual sweep over the 99 banked v4 generations. `total_max: 8192` is the student's
training context and **cannot move** — the 2×T4 OOM ladder goes *down* to 6144:

| `think_max` | `length_band` pass | blocked by `total_max` alone |
|---|---|---|
| 3000 (today) | 49.5% | 4.0% |
| 3500 | 55.6% | 6.1% |
| **4000** | **60.6%** | 10.1% |
| 4500 | 63.6% | 16.2% |
| 6000 | 64.6% | 24.2% |
| ∞ | 64.6% | 33.3% |

The gain saturates: +11pp at 4000, +3pp more for everything after. Past 4000 the binding
constraint is `total_max`, which no configuration change can lift. 4000 takes the available
gain and stops where the returns do.

### A4. Adding deepseek to the judge pool — CONDITIONAL, and defer if it is not one change

Restoring deepseek's qualified role means `roles: [generator, judge, tiebreak]` and appending it
to `routing.judge`/`routing.tiebreak`. There is one blocker to check first.

A reasoning model that spends its reply budget thinking returns no verdict — this repo already
carries a commit for exactly that failure (`4bcf014`, "a judge that spends its reply budget
thinking is not a judge"). The qualification's remedy is `thinking: {"type": "disabled"}` on
judge calls only. But `_bai_request_hook(payload, model)` **is not role-aware**, so the quirk
cannot vary by role.

Check whether `role_params` can carry `thinking` through to the payload. If it can, make the
change. **If it cannot, stop and report it — do not ship an untested judge, and do not make the
hook role-aware as a side quest.** A1-A3 are the fix; A4 is a restoration that can wait for its
own validated change.

## Part B — the gpt-oss floor measurement, both arms back to back

Task 2 edited all 14 base templates, replacing "450 to 700 words of deliberation is normal for a
matter of any substance" with a hard ceiling. That sentence was added on 2026-08-18 to push
gpt-oss traces **up**, because gpt-oss's measured failure is `think < think_min`, the mirror of
deepseek's. After A2, gpt-oss is the **lead** generator reading those edited templates.

The live config sets neither `prompt_overlay` nor `harmony_completions`, so gpt-oss reads the
base `prompts/` directory directly. Nothing has measured it there under the new wording.

### The control has to be built, because none exists

The live store's 1,281 gpt-oss generations span many config eras and prompt revisions and are
not a usable baseline (`length_band` pools to 61.7%, against a remembered 1.65% failure — the
two are not measuring the same thing).

Build the control instead: extract the **pre-edit** templates with
`git show f499372:src/tuned/data/prompts/<file>` into a scratch directory, and point the control
arm's `prompt_overlay` at it. The overlay mechanism exists for exactly this. The treatment arm
sets no overlay and reads the edited base.

**Run both arms back to back, in one sitting.** This is the direct lesson of the 2026-08-27
deepseek A/B, whose arms ran 13h41m apart: because multiple upstreams sit behind one model id
with no provider-side identifier recorded, every pooled between-arm number from that run is
uninterpretable. Do not repeat it. Both arms use the same seeds, the same generator, and the
same `length_band` — the **post-A3** one, so the measurement reflects what ships.

### Pre-registered, before the run

Control = pre-edit templates, treatment = edited templates, same seeds.

| # | measurement | pass line |
|---|---|---|
| 1 | `think < think_min` breach rate | treatment **<= control + 5pp** — the whole point |
| 2 | median trace words | treatment **>= 400** in absolute terms |
| 3 | `length_band` pass rate | treatment **>= control − 5pp** |
| 4 | `self_verification` pass rate | treatment **>= control − 5pp** |
| 5 | every generation is `gpt-oss-120b`; `$0` on the ledger | hard pass line |

Measurement 1 is the risk this exists to close. If it fails, the prompt edit is harmful to the
generator that is now lead, and the branch must either revert the edit or keep deepseek leading
— that finding is reported, not worked around.

Report both arms' n. Judging is out of scope; gate rates answer the question.

## Isolation and cost

- New isolated workdirs `data/build/exp_gptoss_ctl` and `data/build/exp_gptoss_new`, both added
  to `paths.ISOLATED_WORKDIR_SIBLINGS` with a test.
- Seed both from the live store **read-only** with `scripts/seed_exp_store.py --seed 3407
  --per-source 200`, the same arguments as every prior arm, so the seeds match.
- Both arm configs carry the A1 fence, and `routing.generator: [cerebras/gpt-oss-120b]` alone —
  a single ref, so a throttle cannot silently turn a gpt-oss arm into a deepseek arm.
- cerebras is free. `rpm: 5` and `tpm: 30000` make it slower than b.ai: budget ~20 minutes per
  arm, ~40 minutes for both. That is the price of a control that is not confounded.
- **The live control store is never opened for write.** Fingerprint it before and after; the
  diff must be empty.

## Out of scope

- Re-running the deepseek A/B with both arms in one sitting. Worth doing, but it answers a
  question that no longer blocks anything now that deepseek is not lead.
- Making `_bai_request_hook` role-aware (see A4).
- Raising `total_max` or `max_seq_length`. Both are the student's context; the OOM ladder goes
  down, not up.
- Reverting the prompt edit. Part B measures whether that becomes necessary; it is not assumed.

## Acceptance criteria

1. Both gpt-5 models carry `usd_cap: 0.0` **with prices**, and the fence is verified against the
   shipped config by calling the real `budget_ok_for`, plus a price-stripped counterfactual
   proving the prices are what block.
2. `routing.generator` leads with `cerebras/gpt-oss-120b`; deepseek is ref 2; tiebreak unchanged.
3. `length_band.think_max` is 4000, with the saturation table's reasoning recorded in a comment.
4. A4 is either done or explicitly reported as deferred with the reason.
5. Both arms run to completion in one sitting, every generation `gpt-oss-120b`, `$0` ledgered.
6. Each pre-registered measurement is reported beside its pass line in
   `docs/reports/2026-08-27-gptoss-floor-under-the-prompt-ceiling.md`.
7. `data/build/state/law_v1.sqlite3` identical in size and mtime before and after.
8. Full suite green.
