# E4 — judge calibration on banked deepseek generations

First real number on deepseek's downstream (post-length-gate) quality. Before this run, 0 calibrated
verdicts existed anywhere and `judge_threshold` was empty in both arm stores — nothing about
deepseek's judge-accepted quality had ever been measured.

## Status

Complete. Both stores judged to exhaustion of their currently-reachable backlog. Live store
untouched (fingerprint identical before/after). No `gpt-5` calls, no `openai` provider calls, in
either run. Cerebras spend: 125,516 tokens (~$0.07), 16.7% of the 750,000-token hard cap. Config
tweak commit: **`092bc0a`**.

## Step 1 — config tweak

Appended `groq/openai/gpt-oss-20b` to `routing.judge` in both arm configs, positioned after
`cerebras/gemma-4-31b` and before the paid `openai/gpt-5-*` backstops (list order verified with
`load_build_config` post-edit: `['groq/qwen/qwen3.6-27b', 'cerebras/gemma-4-31b',
'groq/openai/gpt-oss-20b', 'openai/gpt-5-mini', 'openai/gpt-5-nano']` in both files). The model's
`roles` list also had to gain `judge` (was `[tiebreak, probe]`) — `config._validate` rejects a
routing ref whose model doesn't declare that role, so the append alone would not have loaded.
Both edits were applied identically to `configs/data_law_v1_exp_deepseek.yaml` and
`configs/data_law_v1_exp_prompt_v5.yaml` — required, since `test_the_prompt_v5_arm_config_is_fenced_and_matches_its_control`
asserts the two files' bodies are byte-identical past the header.

`tests/test_build_config.py -q` (148 tests): all passed post-edit, including
`test_the_deepseek_arm_config_is_fenced`'s `list(cfg.routing.judge)[:2] ==
["groq/qwen/qwen3.6-27b", "cerebras/gemma-4-31b"]` (unaffected — the append is at index 2) and the
byte-equality fence between the two arm files. LF-only confirmed (`grep -c $'\r'` = 0 on both
files, before and after). Committed before any API call: **`092bc0a`** — "add gpt-oss-20b as a
non-production judge fallback for qwen tpd exhaustion".

## Step 1a — an operational gap the brief didn't anticipate: nothing was in `judging`

Before running `tuned.data.judge`, the v4 store (`data/build/exp_deepseek`) had **zero** tasks in
state `judging`. Of its 40 planned tasks (13 sc / 14 predex / 13 tathya), the funnel had already
resolved to: `format_parked` 20, `accepted` 9, `judge_error` 8, `rejected` 3. The 8 `judge_error`
tasks are exactly the ones stranded by the prior run's gemma-cooling incident (recorded in the
2026-08-26 wave's progress notes: "gemma entered cooling at judge batch 5, never recovered") — each
carries judge slot A (qwen) already and is missing only slot B (gemma). Per `REOPEN_STATES`,
`judge_error` maps back to `judging` and is a free park (attempts preserved, no re-billing of slot
A). I ran `tuned.data.tasks --config configs/data_law_v1_exp_deepseek.yaml --reopen judge_error`
before invoking the judge — this is the documented recovery path for exactly this situation, not a
new deepseek generation. Without it, the v4 judge run would have claimed 0 tasks and produced no
new data.

I did **not** reopen `format_parked` (20 in v4, 22 in v5): those tasks failed a format gate (very
likely `irac_placement`, per E2's finding that it fails 63.7% of all deepseek evaluations) on every
attempt and never reached judging at all — reopening them targets `pending` (a fresh bai/deepseek
generation call), which is out of scope for a judging-only measurement and would spend the
generation budget this task doesn't own.

**Consequence for "banked passers":** the plan's "~70-75 of 79 banked passers" estimate (49 + 30
length-band-passing *generations*) overstates what is reachable by judging. Length-band passing and
ever-reaching-the-judging-step are different gates. The generations that actually reach judging are
far fewer:

| store | length-band passers (gens) | tasks that ever reached judging | of those: judged-and-resolved this run |
|---|---|---|---|
| v4 | 49 | 20 (12 already decided, 8 reopened) | 20 (8 newly decided) |
| v5 | 30 | 16 (all fresh) | 15 (1 parked, see below) |

## Step 2 — judge runs

Both ran the exact invocation from `docs/superpowers/specs/2026-08-26-deepseek-validation-wave-design.md`
(`PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tuned.data.judge --config <arm> --n-workers 3
--max-batches 30`), v4 first.

**v4** (`data/build/exp_deepseek/out/judge_e4.log`): 4 batches, self-terminated on `claimed=0`.
`done: batches=4 decided=8 accepted=7 rejected=1 tiebreaks=1 regen=0 slot-err=0 tokens=46938`.
Combined with the 12 already-decided tasks: **20/20 reachable tasks now fully decided.**

**v5** (background run, `data/build/exp_prompt_v5/out/judge_e4.log`): 7 batches, self-terminated.
`done: batches=7 decided=16 accepted=13 rejected=2 tiebreaks=2 regen=1 slot-err=0 tokens=178791`.
One of the 16 "decided" tasks (`c0dcdb27…`, predex, borderline tiebreak on attempt 1) triggered the
policy's one-regeneration path, and the regeneration hit a genuine stale-prompt guard:
`stale-prompt:gen_irac_analysis_v3:63b7780879f2!=09e8c6ffaf80` — the on-disk template's sha no
longer matches what this task was planned against. This worktree has concurrent template-editing
work in flight this same day (E0's revert, E2's clause overlay work per the controller ledger), so
a template hash drifting mid-run is a real risk of running a judge/regenerate pass in a shared
worktree while another task is editing prompt templates. This is a park, not a quality verdict —
excluded from the accept-rate denominator below, flagged as an anomaly, and left untouched
(recovering it is templates-team's call, not this task's).

No `gpt-5` anywhere in either log (checked by grep, not just eyeballing). No `openai` provider rows
in either store's `budget_ledger` or `judgement` table, ever. `groq/openai/gpt-oss-20b` was added to
the pool but **never invoked as a judge in either store** — qwen and gemma never failed or
exhausted, so the non-production fallback added in Step 1 was armed but not exercised this run.

## Step 3 — measurement

**Rule applied:** both judges' three axes (grounding, validity, coverage) `>= 4` → accept; this is
explicitly **not a fitted `judge_threshold`** — both stores show 0 active threshold rows, and both
runs printed `NOTE: no active judge_threshold rows - decisions are PROVISIONAL (P5 calibrates)`. The
informal rule I was asked to apply is, as it happens, *exactly* the pipeline's own fallback logic
when no threshold is active (`judge_policy.slot_verdict`, `PASS_MIN=4`/`FAIL_MAX=2`) — so the
store's own `accepted`/`rejected` task states already are this rule's output, verified by
reconstructing the verdict from raw axis scores for all 36 judged tasks and matching every one
against its recorded final state (zero mismatches).

### Judge pair per row

**100% production pair (qwen slot A + gemma slot B) in both stores.** `groq/openai/gpt-oss-20b`
never served a row. Tiebreak seat (a pre-existing routing entry, not part of this task's config
change) went to `mistral/mistral-large-latest` on every tiebreak in both stores.

| store | slot A | slot B | tiebreak |
|---|---|---|---|
| v4 | groq/qwen (20) | cerebras/gemma (20) | mistral/mistral-large (3, incl. 2 pre-existing) |
| v5 | groq/qwen (16) | cerebras/gemma (16) | mistral/mistral-large (2) |

### Accept rate

| store | accepted | rejected(judge) | resolved n | accept rate | excluded (parked, not a verdict) |
|---|---|---|---|---|---|
| v4 | 16 | 4 | 20 | **80.0%** | 0 |
| v5 | 13 | 2 | 15 | **86.7%** | 1 (stale_prompt) |
| **pooled** | **29** | **6** | **35** | **82.9%** | 1 |

Pooled accept rate is **well above** the 60% flag line — this is the opposite finding from "length
is not the binding quality problem." Once a deepseek generation clears length-band **and** the
format gates, the judges accept it 4 times out of 5. Read together with the v4/v5 funnel numbers
above (only 20/40 and 16/40 tasks per store ever reach judging at all, the rest dying at
`format_parked`), the binding quality constraint on this generator is **pre-judge format
compliance** (principally `irac_placement`, per E2's 63.7%-fail finding), not anything the LLM
judges are catching. The judges are a weak filter on rows that already survived the format gates.

**v4 vs v5 (prompt era) comparison:** v5 (86.7%) trends ~7pp above v4 (80.0%) on this small sample
(n=20 vs n=15 resolved) — directionally consistent with v5 shipping being kept over v4 in the E0
decision, but the gap is well inside noise at this n and should not be read as confirmation; E0's
own pre-registered length_band comparison (47.4% vs 42.5%, a wash by its 5pp line) is the load-bearing
result for that question, not this.

### By source arm (resolved rows only; small-n, directional)

| arm | v4 accept | v5 accept | pooled |
|---|---|---|---|
| predex | 9/9 (100%) | 7/8 (87.5%) | 16/17 (94.1%) |
| tathya | 5/6 (83.3%) | 4/4 (100%) | 9/10 (90.0%) |
| sc | 2/5 (40.0%) | 2/3 (66.7%) | 4/8 (50.0%) |

`sc` (raw supreme-court judgment chunks) is the weak arm both pre-judging (highest format-park
share: 18/26 = 69% of its tasks never reach judging) and post-judging (50% accept vs ~90%+ for the
other two sources) — two independent signals pointing the same direction on the same source.

### Tiebreak pattern (n=5, both stores)

All 5 tiebreak invocations across both runs scored the **identical** vector — grounding=4,
validity=3, coverage=4 (min-axis 3 = borderline, never itself a pass). Every one of the 5 was
"already regenerated," so the policy rejected rather than requeued. n is too small to conclude
mistral-large is miscalibrated rather than coincidentally landing on the same borderline case
shape, but the total uniformity (5/5 identical triple) is worth a note for whoever calibrates this
seat next.

### Axis score distributions

```
v4 slot A (qwen, n=20)   grounding {4:3, 5:17}         validity {2:1, 4:1, 5:18}   coverage {4:1, 5:19}
v4 slot B (gemma, n=20)  grounding {2:1,3:2,4:2,5:15}  validity {2:2, 5:18}        coverage {4:1, 5:19}
v5 slot A (qwen, n=16)   grounding {2:1,4:1,5:14}      validity {2:1,4:1,5:14}     coverage {4:2, 5:14}
v5 slot B (gemma, n=16)  grounding {2:1,3:2,4:1,5:12}  validity {2:1,4:1,5:14}     coverage {4:1, 5:15}
```

Gemma (slot B) shows a visibly wider low-score spread than qwen (slot A) in both stores, consistent
with the config's standing note that gemma is "the most accurate of the three judges" from earlier
hand-forensics — it is the one actually finding things to mark down.

### Errors / parked rows

`slot-err=0` and `judge-err=0` across every batch in both runs — the fleet did not fail once. The
only anomaly is the single v5 `stale_prompt` park described above (template drift, not a quality
signal). Two v5 tasks were `rejected` **before** this run started via a gate-only path (disposition
`reject:length_band,citations,...`, zero judgement rows) — these are generation-side gate rejects,
unrelated to judging, and are excluded from every accept-rate figure above.

## Step 4 — budget

Cerebras (gemma) tokens spent, today's ledger rows only:

| store | requests | prompt | completion | total |
|---|---|---|---|---|
| v4 | 8 | 41,405 | 917 | 42,322 |
| v5 | 16 | 81,271 | 1,923 | 83,194 |
| **total** | **24** | **122,676** | **2,840** | **125,516** |

**125,516 / 750,000 = 16.7% of the hard cap.** At $4.63/8.2M tokens: **≈ $0.071**, well inside the
$0.42 target and the $0.50 hard cap. Checked between every batch (both runs logged per-batch
`tokens=` and I queried `budget_ledger` mid-run for v5); no batch came close to requiring a stop.

**Qwen tpd:** today's fresh usage (v5 only — v4's 8 reopened tasks reused slot A, buying nothing
new) = 81,904 + 1,802 = **83,706 / 200,000 (41.9%)**. Not exhausted; the non-production fallback
added in Step 1 was never needed.

**Mistral (tiebreak):** 16,507 tokens today (informational only, not part of the cerebras cap).

## Step 5 — live store integrity

`stat -c '%s %Y' data/build/state/law_v1.sqlite3`:

- Before: `554532864 1787309490`
- After: `554532864 1787309490`

Unchanged. All work targeted `data/build/exp_deepseek/state/law_v1.sqlite3` and
`data/build/exp_prompt_v5/state/law_v1.sqlite3` exclusively; no `exp_ds_*` store or the `bai`
bucket was touched by this task.

## Concerns for whoever reads this next

1. **The judges are not the bottleneck.** 82.9% pooled accept, with only 16.7% of the cerebras
   budget spent, means there is a lot of budget headroom if the actual problem — format-gate
   failure, principally `irac_placement` — gets fixed upstream. Spending more judge budget on the
   current funnel buys very little; the 20/40 (v4) and 16/40 (v5) reach-judging rates are the real
   ceiling on yield, not judge harshness.
2. **`judge_threshold` is still empty.** Every number above is provisional by the pipeline's own
   admission (both run logs print the PROVISIONAL note). This report puts a first real number on
   deepseek quality but does not calibrate anything — P5 gold-labelling is still the open item for
   turning "82.9% under an informal >=4/>=4 rule" into a fitted, trustworthy accept rate.
3. **Shared-worktree race surfaced once.** The v5 stale-prompt park shows that running judge
   regeneration concurrently with another task's template edits in the same worktree is a live risk,
   not just a theoretical one. It cost nothing here (one row parked, recoverable), but is worth
   knowing about if E4-style runs and template-editing tasks (E2) are dispatched at the same time
   again.
4. **`sc` (raw court-judgment chunks) is the weak source on both sides of judging** (highest
   format-park rate pre-judging, lowest accept rate post-judging), at small n (8 resolved pooled).
   Worth a dedicated look if the corpus leans on this source, but not statistically load-bearing yet.
5. **The tiebreak seat's 5-for-5 identical score vector** is an odd pattern worth another look once
   there's enough volume to tell coincidence from a calibration issue.
