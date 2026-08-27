# Generator prompt length fix — the paired A/B, measured 2026-08-27

**Treatment arm:** `data/build/exp_prompt_v5`, config `configs/data_law_v1_exp_prompt_v5.yaml`
**Control arm:** `data/build/exp_deepseek`, config `configs/data_law_v1_exp_deepseek.yaml` (banked 2026-08-26, v4 prompt text)
**Spec:** `docs/superpowers/specs/2026-08-27-generator-prompt-length-fix-design.md`
**Prompt edit under test:** `286fd3a` / `b71d9ac` — all 14 generator templates: the permission clause deleted, and `450 to 700 words of deliberation` given a hard 700-word ceiling
**Generator:** `bai/deepseek-v4-flash`, `reasoning_effort: low`, temperature 0.7, top_p 0.95 — identical in both arms
**Judging:** deliberately not run. Gate pass rates answer the question, and the control's accept rate was already a null result at n=12.
**Live control store:** untouched — `554532864 1787309490`, sha256 `2ea51e4c…8f48`, before and after

## Verdict

**The edit did not shorten the traces. Two of the seven pre-registered lines pass.**

Measurement 1 decides how the rest is read. The pass line was a median trace under **900
words**; the control was **1,727**; the treatment measured **2,507**. Traces did not get
shorter under the ceilinged prompts — pooled, they got longer. That pooled comparison is
confounded and the confound is disclosed below: **the two arms ran fourteen hours apart**
and nothing in the design controls for the provider. What survives it is the matched
attempt-1 result and the compliance figure, and both point the same way.

On the matched pairs — the same 40 seeds and the same 40 prompt variants, one generation
each under v4 and v5 — the per-task change in trace length is **indistinguishable from
zero**: median **-58 words**, 21 tasks shorter against 19 longer, sign test **p = 0.87**,
95% bootstrap interval on the median change **[-378, +476] words**. Per-task changes run
from **-4,526** to **+5,319** words. At temperature 0.7 the run-to-run noise on a single
task is an order of magnitude larger than the shift this edit was meant to produce.

The four gates behave the same way. Paired at attempt 1, McNemar over the 40 tasks returns
**p = 0.80 to 1.00** on `length_band`, `irac_placement`, `verbatim_overlap` and
`self_verification`. Nothing moved.

### What this does and does not establish

This has to be stated precisely, because the tidy version is wrong in both directions.

**It does not show that trace length was innocent.** The spec's diagnosis — one length
failure wearing three gate names — was to be tested by producing shorter traces and seeing
whether the gates recovered. This run never produced shorter traces, so the diagnosis was
not put to the test. The brief anticipated the case where measurement 1 passes and 2-4 fail,
and that case would have refuted the diagnosis outright. That is not what happened.
Measurement 1 failed, so the treatment was never applied in effect. The correct reading is
a failed instrument, not a refuted hypothesis.

**What it does establish is that the prompt is not a lever on this generator's trace
length.** The edit reached the model — verified on the wire rather than assumed. Across all
**40** matched attempt-1 payloads the system prompt and the sampling params are
byte-identical and the user block differs on **exactly one line**, which in every case is
the edited sentence. On the `gen_summarization_v2` pair, for instance,
`takes as long as the decision deserves and` is deleted and `is normal for a matter of any
substance` becomes `and no more than 700`; the other thirteen templates make the same move
in their own wording. Across all **94** treatment payloads, **0** lack a ceiling after the
band marker and **0** carry the licence phrase in their instructions — the only three
occurrences of `as long as` anywhere in the arm are inside the grounding judgment text of a
single seed. deepseek-v4-flash read a prompt naming 700 as a hard ceiling and wrote a median
2,507 words anyway — **86% of treatment generations exceed
the instructed ceiling**, against 84% under the permissive v4 wording. The permission clause
was not what was making the traces long.

**The length-to-failure association is real and it reproduced** on a second independent
sample of 94 generations (§5). Pass rates collapse with trace length in both arms, at
closely matching rates in each bucket: at 2,000-3,000 words `length_band` is 0% in both;
above 3,000 words `irac_placement` is 9% control and 5% treatment. That is an association,
not a demonstrated cause. Trace length was never randomised and no intervention here
succeeded in moving it, so length and whatever else co-varies with a long generation are
**not separable with the data this arm produced**.

### The two guards

Measurement **5 held**, and its pass line was registered as **`>= 80%, must not
regress` (control 87%)** — both halves, not just the threshold. The measured values fall on
the wrong side of a literal reading of the second half: `self_verification` went 87% → 86%
pooled (86/99 to 81/94) and 88% → 85% matched (35/40 to 34/40). Neither is a regression in
any sense the data supports. Paired over the same 40 tasks the gate is a wash — 29 pass in
both arms, 0 fail in both, 6 flip one way and 5 the other, **McNemar p = 1.00** — and the
pooled Wilson intervals, [79%, 92%] control against [78%, 92%] treatment, are all but
identical. A one-row and a one-generation difference at these n are not a direction. The
verdict is **PASS** on that basis, not on the threshold alone. Shortening did not cost the
verification cue — mainly because nothing shortened.

Measurement **7 is recorded FAIL and the label overstates it.** Pooled, `think<think_min`
breaches went 3/99 (3.0%) to 6/94 (6.4%) against a `<= 5%` line. On the matched attempt-1
population the two arms are **identical at 3/40 (7.5%) each**, and the control is over the
line there too — which means the 3/99 anchor the line was set against is a pooled figure the
matched view does not reproduce. There is no evidence here that the ceiling converted
ceiling breaches into floor breaches. The pooled difference comes from the retry path, not
from the first attempt.

### The arms ran fourteen hours apart, and nothing controls for that

This is the largest gap in the design and it is not closed by anything below.

The control generations span **2026-08-26T18:56:20Z to 19:22:42Z**. The treatment
generations span **2026-08-27T08:37:08Z to 09:04:41Z**. First call to first call the gap is
**13 hours 41 minutes**, which the heading rounds. Across it the treatment's latency rose
**24%**: mean 35.5 s to 43.9 s, p50 32.5 s to 41.1 s.

A 24% latency rise is exactly what longer traces would produce. It is *also* exactly what a
different upstream would produce, and this project's own record on b.ai is that **multiple
upstreams sit behind one model id**. Nothing in the response envelope carries a
provider-side identifier — no upstream name, no served-model string, no fingerprint — and
none is recorded in the store. **The two explanations are therefore not separable from this
data.** "v5 made the traces longer" and "the upstream serving deepseek-v4-flash on the
Thursday morning was not the one serving it on the Wednesday evening" fit the same
observations equally well.

The report calls the two arms "paired by construction" in the Method section. That pairing
is real for everything under this repository's control — seeds, task plan, prompt variants,
routing, sampling params, gate thresholds — and it does not extend to the provider. Run-time
separation is an uncontrolled variable and it is named here as one.

**What this weakens:** the pooled wrong-sign headline. Median trace words 1,727 to 2,507,
`length_band` 49% to 32%, rows over the 8,192-token cap 8.1% to 22.3% — every pooled
between-arm difference in this report is confounded by the fourteen hours, and none of them
can be attributed to the prompt edit on this evidence.

**What this does not weaken:**

- **The matched attempt-1 wash.** McNemar p = 0.80 to 1.00 across the four gates and a sign
  test at p = 0.87 on trace length are *null* results. A drifting upstream is a reason a
  between-arm difference might be spurious; it is not a mechanism that manufactures a null.
  If anything it makes the wash more striking, since it survived a fourteen-hour gap.
- **The on-the-wire compliance finding, which is the real basis for the conclusion.**
  **86% of treatment generations exceeded the instructed 700-word ceiling** (81/94), against
  84% under the permissive v4 wording (83/99). That is a statement about the treatment arm
  measured against its own instruction, not a between-arm comparison, so no upstream drift
  touches it. A generator told that 700 words is a hard ceiling wrote past it in six
  generations out of every seven. That, and not the pooled medians, is what supports "the
  prompt is not a lever".

Any follow-up arm should record a provider-side identifier per call and, where the question
is a between-arm one, interleave the two arms in a single run rather than running them
back to back.

### One number that is not comparable, and one that genuinely got worse

`prompt_echo` shows 87% → 80% and **must not be read as a regression**. The gate itself
changed between the two runs: `286fd3a` broadened `gates.INSTRUCTION_ECHO_SPANS` from
`450 to 700 words of deliberation is normal` to `450 to 700 words of deliberation`, which is
strictly more general and therefore strictly more likely to fire. The control was scored
with the narrow span, the treatment with the broad one. That is the **only** behavioural
difference in `src/` between the two runs other than the 14 templates. `git diff
1f6c0c0..HEAD -- src/` touches, in full: `gates.py` (that one line), `harmony.py` (a
smoke-probe constant not on this path), `paths.py` (+9/-1, this task's own
`exp_prompt_v5` workdir declaration, which no generation path reads), `providers.py`, and
the 14 templates.

The `providers.py` entry is the b.ai request and response hooks, and they look new in that
diff without being new *behaviour*: they were already live in the working tree when the
control ran, and 869da9b committed them rather than introducing them. **The data proves
that, and the commit history only suggests it**, so the data is what is cited here. The
control's maximum `completion_tokens` is **12,145** — far above the shipped
`GENERATION_OUTPUT_TOKENS` of 4,000, and nothing but `_bai_request_hook` raising the budget
to the model's 16,384 `max_output` can produce a reply that long. And the control has
**0/99 empty-content rows** against the roughly 50%-at-4,096 empty rate that the same
hook's docstring records measuring. Both are facts about the control run itself. (For what
it is worth the commit message agrees: `ac5db21`, committed before the control run,
documents the hook in the arm config.)

The number that did get worse is the one the trainer cares about. **Rows over the
8,192-token cap went from 8/99 (8.1%) to 21/94 (22.3%)**, and the templated row mean from
5,677 to 6,405 tokens. Under this build's drop-never-truncate rule that is direct yield
loss.

### Pooled versus matched, and why they disagree

Read together, the two views say something the pass lines alone do not. **Matched at
attempt 1, v4 and v5 are indistinguishable on every gate.** **Pooled across all attempts,
v5 is worse** — `length_band` 49% → 32%, with Wilson intervals [40%, 59%] and [23%, 42%]
that barely touch. The difference between those two views lives entirely in the retry path,
where treatment traces escalate (median 2,048 → 2,715 → 3,138 words by attempt) while
control traces stayed roughly flat (1,733 → 1,626 → 2,125). That comparison is **not
matched past attempt 1** — a row only has an attempt 2 because it already failed, and the
failing sets differ between arms — so it is reported as a shape to look at, not a measured
effect of the edit.

And the pooled half of that contrast carries the fourteen-hour confound. `length_band`
49% → 32% is a between-arm difference measured across a provider gap this design does not
control, so the retry path is one candidate explanation for it and a different upstream is
another. The matched half is unaffected: a null does not become a null because the upstream
moved.


## What this changes in `configs/data_law_v1.yaml`

**Nothing, on the strength of this arm — and the two routes the spec put out of scope come
back.**

The spec's rule was: if measurements 1-4 pass and 5 and 7 hold, nothing changes but the
prompts and deepseek stays lead generator. 1-4 did not pass. So the condition for "nothing
changes" is not met, and the live config keeps its current `length_band`
(`think_min: 500, think_max: 3000, total_max: 8192`) and its current generator order
unexamined by this run.

What the numbers do force is a decision this task did not have the scope to take:

- **`think_max: 3000` and this generator are incompatible.** The treatment's trace p50 is
  **3,126 tokens** — the median generation is now over the ceiling, not the tail, and that
  is a fact about the treatment arm on its own rather than a between-arm comparison. (The
  57 of 94 versus 44 of 99 `think>think_max` breach counts *are* a between-arm comparison
  and inherit the fourteen-hour confound; the incompatibility does not rest on them. The
  control alone already breached 44 of 99.) The spec named raising
  `think_max` and demoting deepseek to a minority source as the two routes this fix was the
  alternative to. The fix did not work, so they are back on the table, and they are now the
  only levers left that this build has evidence for.
- **The 8,192 row cap is the harder constraint.** 22.3% of treatment rows exceed it. Raising
  `think_max` alone would convert `length_band` failures into rows the assembler drops.
- **Do not revert the prompt edit on this evidence — and hold that recommendation
  loosely.** Matched, v5 is indistinguishable from v4 on every gate, measurements 5 and 6
  held, and every one of the 14 templates came out word-neutral or shorter. There is no
  measured harm to undo, and equally no measured benefit: the honest status of `286fd3a`
  after this run is **inert on deepseek**, not helpful. Its stated value for gpt-oss —
  holding traces above `think_min` — is untested here and unaffected either way.

  The caveat is that "no measured harm" rests on the matched view, and the matched view
  covers attempt 1 only. The pooled harms — `length_band` 49% → 32%, and 22.3% of rows over
  the trainer's cap — are set aside above as living in the retry path, but **the retry
  escalation is itself arm-differential**: treatment traces grow 2,048 → 2,715 → 3,138 words
  by attempt while control traces did not, and the v5 prompt tail is a live candidate cause
  that this arm cannot rule in or out (see *What is still unmeasured*). If the tail turns
  out to drive that escalation, it is a real harm that the attempt-1 wash is blind to by
  construction. Keeping the edit is the right call on what is measured; it is not a
  finding that the edit is safe.
- **`gates.INSTRUCTION_ECHO_SPANS` is now broader than the wording any prompt issues.** That
  is deliberate and documented, but it means the `prompt_echo` series is discontinuous at
  `286fd3a`. Any future comparison against a pre-`286fd3a` arm has to say so.

## What is still unmeasured

- **Accept rate under v5 prompts is unknown.** Judging was deliberately skipped, per the
  spec. The control's 75% (9/12) was already recorded as a null result at that n, and this
  arm produced 16 tasks in `judging` that were never judged. Nothing here says whether a v5
  row a judge sees is better or worse than a v4 one.
- **Whether shorter traces would fix the gates is still open.** No intervention in this run
  produced them. `reasoning_effort` is already at the enum floor (`low`; `disabled` yields
  no trace at all and violates the >=80% reasoning-trace floor), and the prompt is now
  measured as not being a lever. **A smaller output budget is not on that list, and it is
  worth saying why: it has already been measured and rejected here.**
  `providers._bai_request_hook` exists precisely to raise the budget, and its docstring
  records the measurement — a 4,096-token budget returned empty content on **10 of 20** real
  synthesis calls, against 0 of 4 at 12,288, and the surviving rows are biased toward short
  traces, which the docstring calls "silent selection on the corpus rather than honest
  sampling". Squeezing the budget would buy shorter traces by discarding the long ones
  unrecorded. What genuinely remains untried: a different generator, or accepting the
  lengths and moving the band.
- **Which of the three gate failures is length-driven cannot be told apart.** `length_band`,
  `irac_placement` and `verbatim_overlap` all fail more on long traces in both arms, but
  with length unmoved there is no contrast to attribute anything to. The one thing that is
  clear and unchanged is the failure *mode*: **62 of 62** `irac_placement` failures in the
  treatment are an IRAC heading inside the trace, and **0** are a heading missing from the
  answer (control: 61 of 61, and 1 malformed answer). The answers remain well-formed. It is
  the trace that misbehaves, exactly as the control found.
- **The retry escalation is unexplained.** Treatment traces grow with each attempt where
  control traces did not. The retry nudge in `generate._REPAIR_HINTS` is the obvious
  suspect and this arm cannot test it: the attempt-2 populations are selected differently in
  the two arms, so the two columns are not comparable, and separating the nudge from the
  new prompt tail would need an arm that varies one without the other.
- **Two permanent rejects appeared** (`citations` fired on 2 of 94 rows; the control had
  none). At n=2 that is noise until it is seen again.

## Method

Every number above and below is produced by `data/build/exp_prompt_v5/out/report_ab.py`,
which opens both stores read-only and is modelled on the control's own
`data/build/exp_deepseek/out/report_wave.py`. Row lengths are built the way
`decontaminate.generated_rows` builds a row, rendered through the pinned chat template and
counted with the pinned tokenizer. Trace words are `len(think.split())`, the counter that
reproduces the control's recorded 1,727 median and 836 answer median exactly.

**Two percentile conventions are in play and the report does not reconcile them.** Section 1
reports medians with `statistics.median` (the mean of the two middle values on an even n);
section 3's `p50` column comes from `report_ab.py`'s own `pct`, inherited unchanged from the
control's `report_wave.py`, which takes the nearest index on an `n - 1` scale. On the same
series the two disagree slightly — treatment trace words read **2,507** in section 1 and
**2,500** in section 3 — and on the tails the gap is larger: `pct` puts the control's trace
p90 at **3,664** where a nearest-rank definition gives **3,773**, the figure the spec
quotes. Both are computed from the same rows; neither is wrong; they are different
definitions. `pct` is kept as-is so that every length figure in section 3 is directly
comparable with the control's own banked report, which used it.

The arms are paired by construction: `scripts/seed_exp_store.py --per-source 200 --seed
3407` against the same live store gives both arms the same 600 seeds; the same three
`tasks.py` invocations give the same 40 tasks; and `prompt_registry.pick_variant` keys on
`seed_id`, so both arms drew the same prompt variant for every task. Verified rather than
assumed: the two stores agree on `(task_id, seed_id, arm, task_type, prompt_id)` for all 40
tasks, and disagree on `prompt_sha` for all 40 — which is the edit and nothing else.

Two of the banked control figures the brief quotes are rounded, and the store gives slightly
different numbers. Measurement 6's control is **99.0% (96/97)**, not 100%: one control answer
was genuinely missing a required heading, which the spec's own prose records as "60 of the 61
`irac_placement` failures have a well-formed answer". Measurement 4's control is 53/99, which
rounds to 54%. Both are reported as measured. Neither changes any verdict.

<!-- measured output of data/build/exp_prompt_v5/out/report_ab.py -->
## 0. What was compared

- control store `data/build/exp_deepseek/state/law_v1.sqlite3` (v4 prompts, banked 2026-08-26), treatment store `data/build/exp_prompt_v5/state/law_v1.sqlite3` (v5 prompts). Both opened read-only.
- seed pools identical: **True** (600 vs 600 seeds)
- task plans identical on (task_id, seed_id, arm, task_type, prompt_id): **True** (40 vs 40 tasks)
- `prompt_sha` identical on **0/40** tasks — 0 is the expected value and the only intended difference between the arms: the template bytes changed.
- tokenizer pin: `unsloth/Qwen3-8B-unsloth-bnb-4bit` @ `62efd7f9d748e394734a7adae2adf96e13a2abc8`; length band think_min=500 think_max=3000 total_max=8192

## 1. The seven pre-registered measurements (all attempts, pooled)

This is the population the pass lines were fixed against: every generation with content, all attempts pooled. The control column reproduces the spec's banked figures from the store rather than restating them.

| # | measurement | pass line | control (v4) | treatment (v5) | verdict |
|---|---|---|---|---|---|
| 1 | median trace words | **< 900** | 1,727 words (n=99) | **2,507 words (n=94)** | **FAIL** |
| 2 | `length_band` pass rate | **> 70%** | 49% (49/99) | **32% (30/94)** | **FAIL** |
| 3 | `irac_placement` pass rate | **> 60%** | 38% (38/99) | **34% (32/94)** | **FAIL** |
| 4 | `verbatim_overlap` pass rate | **> 70%** | 54% (53/99) | **45% (42/94)** | **FAIL** |
| 5 | `self_verification` pass rate | **>= 80%, must not regress** | 87% (86/99) | **86% (81/94)** | **PASS** |
| 6 | answer well-formedness (`missing_in_answer` empty) | **>= 95%** | 99.0% (96/97) | **100.0% (91/91)** | **PASS** |
| 7 | `think<think_min` breaches | **<= 5%** | 3.0% (3/99) | **6.4% (6/94)** | **FAIL** |

**2 PASS / 5 FAIL.**

## 2. The same seven on matched first attempts only

The pooled population above is **not** composition-matched: a task that fails its gates is retried, so a failing task contributes up to three rows and a passing task contributes one. That biases every pooled rate downward, by a different amount in each arm. Restricting to attempt 1 gives one row per task from the same 40 tasks, the same seeds and the same prompt variants in both arms — an exactly paired comparison. The pass lines were not registered against this population; it is reported as the like-for-like check on the table above.

| # | measurement | pass line | control (v4) | treatment (v5) | verdict |
|---|---|---|---|---|---|
| 1 | median trace words | **< 900** | 1,733 words (n=40) | **2,048 words (n=40)** | **FAIL** |
| 2 | `length_band` pass rate | **> 70%** | 45% (18/40) | **40% (16/40)** | **FAIL** |
| 3 | `irac_placement` pass rate | **> 60%** | 42% (17/40) | **42% (17/40)** | **FAIL** |
| 4 | `verbatim_overlap` pass rate | **> 70%** | 50% (20/40) | **45% (18/40)** | **FAIL** |
| 5 | `self_verification` pass rate | **>= 80%, must not regress** | 88% (35/40) | **85% (34/40)** | **PASS** |
| 6 | answer well-formedness (`missing_in_answer` empty) | **>= 95%** | 97.4% (37/38) | **100.0% (39/39)** | **PASS** |
| 7 | `think<think_min` breaches | **<= 5%** | 7.5% (3/40) | **7.5% (3/40)** | **FAIL** |

**2 PASS / 5 FAIL.**

### 2b. Per-task paired trace length, attempt 1

The tightest form of the length question: the SAME task, the same seed and the same prompt variant, generated once under v4 and once under v5. One pair per task; no pooling, no composition to reweight.

- paired tasks with a trace in both arms: **40** of 40
- median control trace: **1,733** words; median treatment trace: **2,048** words
- median per-task change: **-58** words; mean **+246**
- tasks whose trace got SHORTER under v5: **21/40**; longer: **19/40** (a coin flip would be 20/20)
- tasks whose v5 trace came in under the instructed 700-word ceiling: **7/40**; under v4: **6/40**
- sign test on the direction of the per-task change: **p = 0.87** (21 shorter / 19 longer, 0 tied)
- 95% bootstrap interval on the median per-task change (10,000 resamples, seed 3407): **[-378, +476]** words
- per-task change ranges from **-4,526** to **+5,319** words: the run-to-run spread on a single task dwarfs the shift being looked for.

| source arm | pairs | median control | median treatment | median delta |
|---|---|---|---|---|
| sc | 13 | 2,134 | 1,712 | -190 |
| predex | 14 | 1,431 | 2,519 | -374 |
| tathya | 13 | 1,727 | 1,977 | +170 |

### 2c. Paired gate outcomes, attempt 1 (McNemar)

Each of the 40 tasks is one pair: the same seed and the same prompt variant, gated once under v4 and once under v5. `v4 pass -> v5 fail` and `v4 fail -> v5 pass` are the only two cells that carry information; the exact binomial over them is the paired test.

| gate | both pass | both fail | v4 pass -> v5 fail | v4 fail -> v5 pass | p |
|---|---|---|---|---|---|
| `length_band` | 7 | 13 | 11 | 9 | 0.82 |
| `irac_placement` | 8 | 14 | 9 | 9 | 1.00 |
| `verbatim_overlap` | 11 | 13 | 9 | 7 | 0.80 |
| `self_verification` | 29 | 0 | 6 | 5 | 1.00 |

## 3. Lengths

### control (v4)

| series | n | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| trace WORDS | 99 | 2115 | 1727 | 3664 | 8016 | 9020 |
| answer WORDS | 99 | 836 | 836 | 1142 | 1408 | 1771 |
| trace tokens (pinned tokenizer) | 99 | 2739 | 2263 | 4721 | 10901 | 11359 |
| answer tokens | 99 | 1036 | 1033 | 1433 | 1806 | 2125 |
| **templated row tokens** | 99 | 5677 | 5455 | 7504 | 12265 | 14698 (8 over 8192) |
| row — arm `sc` | 32 | 5189 | 4980 | 6934 | 9533 | 9533 (1 over 8192) |
| row — arm `predex` | 34 | 5851 | 5653 | 7636 | 11404 | 11404 (3 over 8192) |
| row — arm `tathya` | 33 | 5971 | 5706 | 9745 | 14698 | 14698 (4 over 8192) |

### treatment (v5)

| series | n | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| trace WORDS | 94 | 2682 | 2500 | 4782 | 6626 | 6908 |
| answer WORDS | 94 | 809 | 826 | 1192 | 1336 | 1451 |
| trace tokens (pinned tokenizer) | 94 | 3479 | 3126 | 6570 | 8631 | 9284 |
| answer tokens | 94 | 1009 | 1015 | 1475 | 1641 | 1882 |
| **templated row tokens** | 94 | 6405 | 6011 | 9864 | 10903 | 13294 (21 over 8192) |
| row — arm `sc` | 33 | 5620 | 5021 | 7745 | 10507 | 10507 (2 over 8192) |
| row — arm `predex` | 31 | 6671 | 6003 | 10191 | 13294 | 13294 (8 over 8192) |
| row — arm `tathya` | 30 | 6995 | 6510 | 10234 | 10577 | 10577 (11 over 8192) |

## 4. Every gate, both arms

| gate | control pass | treatment pass | delta (pp) |
|---|---|---|---|
| `answer_key` | 100% (99/99) | 100% (94/94) | +0 |
| `banned_meta` | 86% (85/99) | 83% (78/94) | -3 |
| `citations` | 100% (99/99) | 98% (92/94) | -2 |
| `irac_placement` | 38% (38/99) | 34% (32/94) | -4 |
| `length_band` | 49% (49/99) | 32% (30/94) | -18 |
| `prompt_echo` | 87% (86/99) | 80% (75/94) | -7 |
| `self_verification` | 87% (86/99) | 86% (81/94) | -1 |
| `statutory_grounding` | 93% (92/99) | 97% (91/94) | +4 |
| `statutory_quotation` | 100% (99/99) | 100% (94/94) | +0 |
| `temporal` | 100% (99/99) | 100% (94/94) | +0 |
| `think_format` | 98% (97/99) | 97% (91/94) | -1 |
| `verbatim_overlap` | 54% (53/99) | 45% (42/94) | -9 |

### length_band violation breakdown

| violation | control | treatment |
|---|---|---|
| `think<think_min` | 3/99 | 6/94 |
| `think>think_max` | 44/99 | 57/94 |
| `total>total_max` | 33/99 | 45/94 |

### 95% Wilson intervals on the four gates the pass lines name

Pooled, all attempts — the population the pass lines were fixed against.

| gate | control (n) | control 95% CI | treatment (n) | treatment 95% CI | pass line |
|---|---|---|---|---|---|
| `length_band` | 49% (49/99) | [40%, 59%] | 32% (30/94) | [23%, 42%] | > 70% |
| `irac_placement` | 38% (38/99) | [29%, 48%] | 34% (32/94) | [25%, 44%] | > 60% |
| `verbatim_overlap` | 54% (53/99) | [44%, 63%] | 45% (42/94) | [35%, 55%] | > 70% |
| `self_verification` | 87% (86/99) | [79%, 92%] | 86% (81/94) | [78%, 92%] | >= 80% |

### irac_placement failure mode

The control's story was that the ANSWER is fine and only the TRACE misbehaves: a heading inside the trace, not a heading missing from the answer.

| | control | treatment |
|---|---|---|
| evaluated | 99 | 94 |
| failed | 61 | 62 |
| ...because an IRAC heading is INSIDE the trace | 61 | 62 |
| ...because a required heading is MISSING from the answer | 1 | 0 |

### Compliance with the instructed ceiling

The v5 prompts instruct 450-700 words of trace and name 700 as a ceiling. The v4 prompts named the same band and licensed running past it.

| | control | treatment |
|---|---|---|
| trace within the instructed 450-700 band | 14/99 (14%) | 8/94 (9%) |
| trace over the instructed 700-word ceiling | 83/99 (84%) | 81/94 (86%) |
| trace over 2x the ceiling (1,400 words) | 58/99 (59%) | 66/94 (70%) |

## 4b. Trace length by attempt number

Attempts 2 and 3 exist only for rows that already failed, and the set of rows that failed is not the same in the two arms, so the two columns below are NOT a matched comparison past attempt 1. Read the SHAPE within each column, not the difference across them.

| attempt | control n | control median words | treatment n | treatment median words |
|---|---|---|---|---|
| 1 | 40 | 1,733 | 40 | 2,048 |
| 2 | 34 | 1,626 | 30 | 2,715 |
| 3 | 25 | 2,125 | 24 | 3,138 |

## 5. Gate pass rate by trace length

Descriptive, not causal. Buckets are trace words; `n` is generations with content in that bucket in that arm.

| trace words | arm | n | length_band | irac_placement | verbatim_overlap |
|---|---|---|---|---|---|
| 0–500 | control | 6 | 83% (5/6) | 83% (5/6) | 100% (6/6) |
| 0–500 | treatment | 8 | 50% (4/8) | 100% (8/8) | 100% (8/8) |
| 500–900 | control | 18 | 100% (18/18) | 72% (13/18) | 89% (16/18) |
| 500–900 | treatment | 8 | 100% (8/8) | 88% (7/8) | 100% (8/8) |
| 900–1400 | control | 17 | 100% (17/17) | 35% (6/17) | 65% (11/17) |
| 900–1400 | treatment | 12 | 92% (11/12) | 75% (9/12) | 58% (7/12) |
| 1400–2000 | control | 13 | 69% (9/13) | 54% (7/13) | 62% (8/13) |
| 1400–2000 | treatment | 8 | 88% (7/8) | 12% (1/8) | 62% (5/8) |
| 2000–3000 | control | 23 | 0% (0/23) | 22% (5/23) | 35% (8/23) |
| 2000–3000 | treatment | 21 | 0% (0/21) | 24% (5/21) | 29% (6/21) |
| 3000+ | control | 22 | 0% (0/22) | 9% (2/22) | 18% (4/22) |
| 3000+ | treatment | 37 | 0% (0/37) | 5% (2/37) | 22% (8/37) |

## 6. Pipe health and cost fences (treatment arm)

- generations recorded: **94**; calls returning content: **94** (100%)
- errored rows: 0; finish_reason=length: 0
- latency s: mean 43.9, p50 40.6, p90 72.4, max 118.0
- models seen: [('deepseek-v4-flash', 94)] — only deepseek-v4-flash: **PASS**
- openai requests on the ledger: 0 — $0 fence: **PASS**
- ledger: `bai/deepseek-v4-flash` req=94 429=0 tok=237435+412338
- irac_placement rows the gate skipped (not evaluated): 3 treatment, 2 control

### Task states

| state / disposition | control | treatment |
|---|---|---|
| `judging` / - | 0 | 9 |
| `format_parked` / exhausted:format:length_band,irac_placement,verbatim_overlap,reply_budget | 3 | 5 |
| `format_parked` / exhausted:format:length_band,irac_placement,verbatim_overlap,prompt_echo,reply_budget | 0 | 3 |
| `judging` / regenerate:length_band | 0 | 2 |
| `format_parked` / exhausted:format:think_format,length_band | 0 | 2 |
| `format_parked` / exhausted:format:length_band,irac_placement,prompt_echo,reply_budget | 0 | 2 |
| `format_parked` / exhausted:format:irac_placement | 4 | 1 |
| `format_parked` / exhausted:format:irac_placement,verbatim_overlap | 2 | 1 |
| `format_parked` / exhausted:format:length_band,irac_placement,verbatim_overlap | 2 | 1 |
| `format_parked` / exhausted:format:length_band,irac_placement | 1 | 1 |
| `format_parked` / exhausted:format:length_band,irac_placement,reply_budget | 1 | 1 |
| `format_parked` / exhausted:format:length_band,irac_placement,verbatim_overlap,banned_meta,prompt_echo,reply_budget | 1 | 1 |
| `rejected` / reject:length_band,citations,verbatim_overlap,banned_meta,reply_budget | 0 | 1 |
| `format_parked` / exhausted:format:irac_placement,verbatim_overlap,banned_meta | 0 | 1 |
| `judging` / regenerate:irac_placement | 0 | 1 |
| `judging` / regenerate:irac_placement,verbatim_overlap | 0 | 1 |
| `format_parked` / exhausted:format:length_band,irac_placement,verbatim_overlap,banned_meta | 0 | 1 |
| `judging` / regenerate:length_band,irac_placement,reply_budget | 0 | 1 |
| `format_parked` / exhausted:format:prompt_echo | 0 | 1 |
| `judging` / regenerate:length_band,irac_placement,verbatim_overlap,reply_budget | 0 | 1 |
| `judging` / regenerate:length_band,verbatim_overlap,banned_meta,reply_budget | 0 | 1 |
| `format_parked` / exhausted:format:length_band,statutory_grounding,irac_placement,verbatim_overlap,reply_budget | 0 | 1 |
| `rejected` / reject:length_band,citations,irac_placement,verbatim_overlap,reply_budget | 0 | 1 |
| `accepted` / judge:accept | 9 | 0 |
| `judge_error` / judge-slot-b:role 'judge': no eligible model (skipped: cooling, family-excluded, over-budget) | 8 | 0 |
| `rejected` / judge:reject | 3 | 0 |
| `format_parked` / exhausted:format:length_band,statutory_grounding,irac_placement | 2 | 0 |
| `format_parked` / exhausted:format:length_band,statutory_grounding,irac_placement,banned_meta,prompt_echo | 1 | 0 |
| `format_parked` / exhausted:format:length_band,irac_placement,verbatim_overlap,prompt_echo | 1 | 0 |
| `format_parked` / exhausted:format:length_band,verbatim_overlap,prompt_echo | 1 | 0 |
| `format_parked` / exhausted:format:banned_meta | 1 | 0 |

## 7. Per-arm composition

| source arm | tasks | control gens | treatment gens | control attempts/task | treatment attempts/task |
|---|---|---|---|---|---|
| sc | 13 | 32 | 33 | 2.46 | 2.54 |
| predex | 14 | 34 | 31 | 2.43 | 2.21 |
| tathya | 13 | 33 | 30 | 2.54 | 2.31 |
| **total** | 40 | 99 | 94 | 2.48 | 2.35 |

Composition of the pooled populations, as a share of generations with content:

| source arm | control share | treatment share |
|---|---|---|
| sc | 32% | 35% |
| predex | 34% | 33% |
| tathya | 33% | 32% |

### Median trace words by source arm

| source arm | control | treatment |
|---|---|---|
| sc | 2,130 (n=32) | 2,232 (n=33) |
| predex | 1,357 (n=34) | 2,514 (n=31) |
| tathya | 1,739 (n=33) | 2,637 (n=30) |

### Arm-reweighted pass rates (each source arm weighted equally)

| gate | control | treatment |
|---|---|---|
| `length_band` | 49% | 32% |
| `irac_placement` | 38% | 34% |
| `verbatim_overlap` | 53% | 45% |
| `self_verification` | 87% | 86% |

