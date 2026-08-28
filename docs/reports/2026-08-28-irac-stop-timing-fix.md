# The irac_placement gate is a stop-timing failure: the root-cause A/B

2026-08-28. Two arms, `data/build/exp_irac_ctl` and `data/build/exp_irac_fix`,
run back to back 38.8 seconds apart on `bai/deepseek-v4-flash`.

## 1. Root cause (recap, not re-derived here)

`irac_placement` fails ~69% of this generator's output because of WHEN the
reasoning stops, not what vocabulary it uses: a passing trace ends `<think>`
at the moment it decides to write the answer, while a failing trace treats
that same decision as the cue to keep going and drafts the labelled
`Issue:`/`Rule:`/`Application:`/`Conclusion:` sections inside the reasoning
(Issue-first in 79% of failures, all four labels in 90%, think 1.9x longer).
Compounding it, both summarization templates still mandated a four-headed
ANSWER that `gates.IRAC_ANSWER_TASK_TYPES` (`src/tuned/data/gates.py:116`)
deliberately stopped requiring - template/gate drift that seeds the very
scaffolding the think-side tripwire then fails.

## 2. The treatment, verbatim

Overlay: `data/build/exp_irac_fix/prompts_stop_timing` - copies of all 14
current base `gen_*.md`, 6 edited, 8 byte-identical.

### F1 - name the stop (6 templates)

Three sentences added to each template's how-to-think paragraph, in that
template's own second-person voice. No word counts or lengths anywhere in
the added text.

**`gen_irac_analysis_v1.md`** (appended to the how-to-think paragraph, line 13):

> The thinking ends where the deciding ends. When you reach the point of knowing what the analysis has to say and turn to writing it, let that turn be the last thing your reasoning says — not the opening of a draft you then work through on the bench. Everything that follows the decision to write belongs to the judgment itself.

**`gen_irac_analysis_v2.md`** (appended to the how-to-think paragraph, line 13):

> Your thinking ends when you pick the pen up. The moment you know what the advice has to say and turn to writing it, that turn is the last thing your reasoning does — never the first line of a draft you go on to work through in your head. Everything after that decision belongs to the advice.

**`gen_irac_analysis_v3.md`** (appended to the how-to-think paragraph, line 13):

> The working stops when you know what to dictate. Turning to the note is the last thing your thinking does — not the opening of a note you deliver twice, once silently and once aloud. Everything after that turn belongs to the note.

**`gen_irac_analysis_v4.md`** (appended to the how-to-think paragraph, line 13):

> Your working stops when you know what the model answer has to say. Turning to write it is the last thing your working does — not the first line of an answer you compose privately and then copy out. Everything after that turn belongs to the model answer.

**`gen_summarization_v1.md`** (appended to the how-to-think paragraph, line 11):

> The working ends when you know what the headnote has to say. Turning to settle it is the last thing your reasoning does — not the opening of a headnote you compose once in your head and again on the page. Everything after that turn belongs to the headnote.

**`gen_summarization_v2.md`** (appended to the how-to-think paragraph, line 11):

> Your thinking ends when the decision is straight. The moment you know what the client has to be told and turn to writing it, that turn is the last thing your reasoning does — never the opening of a letter you draft silently first. Everything after that decision belongs to the letter.

### F2 - ask the genre for its own form (2 templates)

The answer-format paragraph rewritten so the deliverable is the document the
template already frames - a headnote as a law report prints one, a note as a
letter goes - in continuous prose, with NO mandate to use the four named
headings. The line-initial-label prohibition on the REASONING is retained
and rephrased; the anti-rehearsal sentences are adapted to the new answer
contract; both length sentences are carried over verbatim.

**`gen_summarization_v1.md`** (line 19, replacing the four-headings
paragraph in full):

> Then settle the headnote. Write it as a law report prints one: continuous prose that states the question the court decided, the proposition of law it laid down, how that proposition works on these facts, and the ratio together with the order made, ordered as the case itself demands and with no labels standing over its parts. Issue, Rule, Application and Conclusion are not words your reasoning may put at the head of a line either; the reasoning runs as continuous prose throughout. The same holds if you feel the pull to check, before you commit to the headnote, that you have the question, the proposition, the application and the ratio all in hand: take that check as one line of the prose you are already in — the issue is settled, this is the rule, applied here it gives this, so this follows — never as a run-through under those four labels, because a labelled run-through is a hidden first draft of the headnote, wherever in your thinking it falls. Write the question, the proposition, the application and the ratio for the first time in the headnote itself. Roughly 250 to 450 words for the headnote; your thinking runs as long as the case requires and is never a retelling of the materials. Work the point through fully — 450 to 700 words of deliberation is normal for a matter of any substance.

**`gen_summarization_v2.md`** (line 19, replacing the four-headings
paragraph in full):

> Then write the note. It goes to the client the way a letter goes: continuous prose telling them what was in dispute, the law the court applied, how it applied to their facts, and what they should understand and do now, ordered as it would best be heard by someone coming to it for the first time and with no labels standing over its parts. Issue, Rule, Application and Conclusion are not words your reasoning may put at the head of a line either; the reasoning runs as continuous prose throughout. The same holds if you feel the pull to check, before you commit to the note, that you have covered the dispute, the law, its application and what follows for the client: take that check as one line of the prose you are already in — the issue is settled, this is the rule, applied here it gives this, so this follows — never as a run-through under those four labels, because a labelled run-through is a hidden first draft of the note, wherever in your thinking it falls. Write the dispute, the law, its application and what follows for the first time in the note itself. Roughly 250 to 450 words for the note; the thinking beforehand takes as long as the decision deserves and is never a retelling of the materials. Work the point through fully — 450 to 700 words of deliberation is normal for a matter of any substance.

## 3. Apparatus and its verification

| check | result |
|---|---|
| 14 `gen_*.md` in the overlay | PASS |
| 8 unmodified files byte-identical to base | PASS |
| 6 modified files show only the intended edits (diff reviewed line by line) | PASS |
| all 14 carry "450 to 700 words of deliberation is normal" | PASS |
| zero CR bytes in any of the 14, and in both configs | PASS |
| all 14 parse through `prompt_registry.load` | PASS |
| `pytest tests/test_build_config.py tests/test_build_paths.py` | **161 passed, 0 failed** |

Every file was written with `open(path, "wb")`; `Path.write_text` was not
used anywhere in this task.

### The 14 shas (sha256[:12], as `prompt_registry` stamps them)

| template | overlay sha | base sha | edit |
|---|---|---|---|
| gen_drafting_v1.md | `48534e3010f5` | `48534e3010f5` | - |
| gen_drafting_v2.md | `618b240ab03e` | `618b240ab03e` | - |
| gen_irac_analysis_v1.md | `4a6bcfa1061c` | `f2b4a76489cb` | F1 |
| gen_irac_analysis_v2.md | `35e23732bdd9` | `a5e62bd4bb3f` | F1 |
| gen_irac_analysis_v3.md | `889ec6ca0dad` | `c4922e9d298c` | F1 |
| gen_irac_analysis_v4.md | `e5f02c422c25` | `78f0e8944ae1` | F1 |
| gen_statute_qa_v1.md | `94e43b22bf48` | `94e43b22bf48` | - |
| gen_statute_qa_v2.md | `4d04338ba007` | `4d04338ba007` | - |
| gen_statute_qa_v3.md | `5888a6c4461d` | `5888a6c4461d` | - |
| gen_statute_qa_v4.md | `713a9060835e` | `713a9060835e` | - |
| gen_summarization_v1.md | `605c2f54c1f7` | `52fdcf8dbd04` | F1+F2 |
| gen_summarization_v2.md | `dcbc9f3ae1e2` | `3b9eefc64d33` | F1+F2 |
| gen_transition_v1.md | `113813116cfb` | `113813116cfb` | - |
| gen_transition_v2.md | `2f28a53e5259` | `2f28a53e5259` | - |

The base column is exactly the `prompt_sha` the CONTROL arm's `task` rows
carry and the overlay column is exactly the `prompt_sha` the TREATMENT arm's
`task` rows carry - so the overlay is provably live in one arm and provably
absent from the other, rather than assumed to be.

### Configs

`configs/data_law_v1_exp_irac_ctl.yaml` and
`configs/data_law_v1_exp_irac_fix.yaml`, both built from
`configs/data_law_v1_exp_ds_ctl2.yaml`. Outside the header the two arms
differ in exactly four lines, and a test asserts there is no fifth:

    workdir              exp_irac_ctl              exp_irac_fix
    prompt_overlay       (absent)                  data/build/exp_irac_fix/prompts_stop_timing
    gpt-oss-20b roles    [tiebreak, probe]         [judge, tiebreak, probe]
    routing.judge        ctl2's pair               exp_deepseek's list (+ its comment line)

The judge asymmetry is deliberate and is not part of the lever: only the
treatment arm is judged, because F2 changes the shape of a summarization
answer and the spot-check in section 9 has to confirm the free fleet still
accepts it. No judge is dispatched against the control, so the asymmetry
cannot touch a generation-time number.

`src/tuned/data/paths.py`: `exp_irac_ctl` and `exp_irac_fix` added to
`ISOLATED_WORKDIR_SIBLINGS`. Tests added:
`test_the_irac_ctl_arm_is_an_isolated_workdir`,
`test_the_irac_fix_arm_is_an_isolated_workdir`,
`test_the_irac_arms_are_fenced_and_differ_only_in_the_overlay_and_judge_seat`.

## 4. Seeding, planning, run

Both stores seeded identically from the live store, read-only:

    scripts/seed_exp_store.py --config <arm> --from data/build/state/law_v1.sqlite3 --per-source 200 --seed 3407

600 seeds each (200 from each of sc / predex / tathya; the other four live
sources have zero eligible rows). Planning, matching the 2026-08-26
validation wave's stratification exactly, one command per source per arm:

    tuned.data.tasks --stream synthesis --arm sc     --n 13 --source s3://indian-supreme-court-judgments                                       --mix irac_analysis=0.55,summarization=0.45
    tuned.data.tasks --stream synthesis --arm predex --n 14 --source L-NLProc/PredEx_Instruction-Tuning_Pred-Exp                                --mix irac_analysis=0.55,summarization=0.45
    tuned.data.tasks --stream synthesis --arm tathya --n 13 --source L-NLProc/TathyaNyaya-and-FactLegalLlama-NyayaFacts-Datasets                --mix irac_analysis=0.55,summarization=0.45

40 pending tasks per arm (22 irac_analysis, 18 summarization); the 40 task
ids are identical across the arms (asserted, not assumed).

    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tuned.data.generate --config <arm cfg> --n-workers 5 --max-batches 30

| arm | window (UTC) | wall | gap before | batches | claimed | gen-ok | errors |
|---|---|---|---|---|---|---|---|
| ctl | 04:41:04.512 - 05:10:28.222 | 29.4 min | - | 24 | 108 | 108 | 0 |
| fix | 05:11:06.982 - 05:35:18.951 | 24.2 min | **38.8 s** | 20 | 92 | 92 | 0 |

**Measurement cutoff.** Every number below is the GENERATION-PHASE snapshot,
frozen at each arm's last generation above (ctl n=108, fix n=92). The judge
spot-check in section 9 ran afterwards and its `regenerate` disposition
added one further generation to the fix store (gen_id 93); it is outside the
snapshot and excluded from every rate here.

## 5. The five pre-registered lines

"n" counts EVERY generation attempt, retries included - the same currency
the clause/cap A/B and the v4/v5 rerun used.

| # | measurement | ctl (n=108) | fix (n=92) | delta | line | verdict |
|---|---|---|---|---|---|---|
| 1 | **irac_placement fail rate** | 65.74% (71/108) | **39.13% (36/92)** | **-26.61pp** | <= -15pp | **PASS** |
| 2 | length_band pass rate | 49.07% (53/108) | 40.22% (37/92) | **-8.86pp** | >= -5pp | **FAIL** |
| 3 | full-gate clean rate | 13.89% (15/108) | **26.09% (24/92)** | **+12.20pp** | >= 0 | **PASS** |
| 4 | summarization, every OTHER gate | see below | see below | 4 of 11 outside band | +-5pp | **FAIL** |
| 5 | integrity (all deepseek, $0 non-bai at generation, fingerprint) | - | - | - | hard | **PASS** |

### Per task type - the attribution

F2 touches summarization only, so the split IS the attribution, and it is
unambiguous: **F2 does all of the measured work and F1 does none of it.**

| task type | edit it received | ctl irac fail | fix irac fail | delta |
|---|---|---|---|---|
| irac_analysis | F1 only | 61.67% (37/60) | 60.78% (31/51) | **-0.88pp** |
| summarization | F1 + F2 | 70.83% (34/48) | **12.20% (5/41)** | **-58.64pp** |

The mechanism itself moves the same way. Counting rows whose `<think>`
contains a line-initial Issue/Rule/Application/Conclusion:

| task type | ctl | fix | delta |
|---|---|---|---|
| irac_analysis | 55.00% (33/60) | 52.94% (27/51) | -2.06pp |
| summarization | 70.83% (34/48) | **4.88% (2/41)** | **-65.95pp** |

Full-gate clean rate by task type: irac_analysis 13.33% (8/60) -> 25.49%
(13/51); summarization 14.58% (7/48) -> 26.83% (11/41).

### Line 4 - summarization only, every other gate

| gate | ctl (n=48) | fix (n=41) | delta | within +-5pp |
|---|---|---|---|---|
| length_band (fail) | 43.75% (21/48) | 58.54% (24/41) | **+14.79pp** | **NO - adverse** |
| self_verification | 18.75% (9/48) | 7.32% (3/41) | -11.43pp | NO - improvement |
| banned_meta | 18.75% (9/48) | 12.20% (5/41) | -6.55pp | NO - improvement |
| prompt_echo | 12.50% (6/48) | 7.32% (3/41) | -5.18pp | NO - improvement |
| verbatim_overlap | 56.25% (27/48) | 60.98% (25/41) | +4.73pp | yes |
| think_format | 0.00% (0/48) | 2.44% (1/41) | +2.44pp | yes |
| answer_key | 0.00% (0/48) | 0.00% (0/41) | 0.00pp | yes |
| citations | 0.00% (0/48) | 0.00% (0/41) | 0.00pp | yes |
| statutory_grounding | 0.00% (0/48) | 0.00% (0/41) | 0.00pp | yes |
| statutory_quotation | 0.00% (0/48) | 0.00% (0/41) | 0.00pp | yes |
| temporal | 0.00% (0/48) | 0.00% (0/41) | 0.00pp | yes |

The line is two-sided as written, so it fails on four gates. Three of the
four moved in the SAFE direction (fewer failures). The one real cost is
`length_band`, and it is the same cost line 2 records.

### Line 5 - integrity

- Every generation in both arms is `deepseek-v4-flash`: ctl 108/108, fix
  92/92, by `SELECT DISTINCT model FROM generation`, not sampled.
- `budget_ledger` at the generation cutoff names exactly ONE provider in
  each arm, `bai`. Non-bai spend at generation time is $0 in both arms.
- Zero HTTP 429s in either arm's generation phase.
- Live store fingerprint `554532864 1787309490` checked six times - before
  anything, after seeding both, after planning both, after each generation
  run, and after the spot-check. Identical every time. The live store was
  only ever opened `mode=ro`.
- No `.env` contents printed at any point; every invocation logged only
  `loaded 18 key(s) from .env`.

## 6. Reading the guard breach - it runs AGAINST the treatment

This is where this A/B differs structurally from the clause arm, whose guard
breach flattered its primary. Here the breach penalises it.

Within BOTH arms, an irac-failing trace is a LONGER trace:

| arm | irac FAIL, think est-tok p50 | irac PASS, think est-tok p50 |
|---|---|---|
| ctl | 3,110 (n=71) | 1,926 (n=37) |
| fix | 4,782 (n=36) | 2,527 (n=56) |

and the fix arm's traces are longer overall (p50 2,758 -> 3,411; p90 6,270
-> 7,676). Longer thinking is the direction that PRODUCES irac failures, so
the treatment posted -26.61pp while carrying a headwind, not a tailwind.
Holding the population fixed makes the effect larger, not smaller
(sensitivity check, NOT pre-registered - restricted to `length_band`-passing
rows only):

| subset | ctl irac fail | fix irac fail | delta |
|---|---|---|---|
| all | 52.83% (28/53) | 13.51% (5/37) | **-39.32pp** |
| irac_analysis | 50.00% (13/26) | 20.00% (4/20) | -30.00pp |
| summarization | 55.56% (15/27) | 5.88% (1/17) | -49.67pp |

Two things follow. The primary is real and is not a population artefact.
And F1's apparent null on `irac_analysis` (-0.88pp over all attempts) is at
least partly a length effect - within band-passing rows F1's own task type
moves -30pp on n=46 - so F1 is better described as UNDERPOWERED AND
EXPENSIVE HERE than as proven inert.

The end-to-end yield says the same thing. From the same 40 planned tasks:

| arm | tasks reaching `judging` | tasks `format_parked` | generation attempts spent |
|---|---|---|---|
| ctl | 15 | 24 | 108 |
| fix | **24** | 15 | **92** |

60% more usable rows from 15% fewer calls.

## 7. Secondaries

| metric | ctl | fix | delta |
|---|---|---|---|
| think est-tokens (chars/4) p50 | 2,758 | 3,411 | +23.7% |
| think est-tokens p90 | 6,270 | 7,676 | +22.4% |
| think real-tokens mean (provider-billed) | 3,239 | 3,933 | +21.4% |
| - irac_analysis est p50 / p90 | 3,058 / 7,466 | 3,507 / 7,686 | +14.7% / +2.9% |
| - summarization est p50 / p90 | 2,273 / 5,249 | 3,386 / 7,691 | +49.0% / +46.5% |
| **completion tokens per length-passing row** | 7,233.8 (383,389/53) | **10,414.9 (385,352/37)** | **+44.0%** |
| summarization answer length, chars p50 | 4,268 | 4,228 | -0.9% |

The deliverable itself did not shrink when the headings came off - the answer
is the same size, it is just prose now. The cost is entirely on the think
side, and it is the largest single reservation about this treatment.

## 8. F3 - the retry note's conditional

Both arms share the same repair hint (`generate.py::_REPAIR_HINTS`, changed
by `e03be76` BEFORE this run), so this is a property of one note under two
prompt regimes rather than an A/B of the note.

| arm | P(attempt N+1 passes irac given attempt N FAILED irac) | P(pass given prior PASSED) |
|---|---|---|
| ctl | 34.69% (17/49) | 47.37% (9/19) |
| fix | 34.62% (9/26) | 80.77% (21/26) |
| historical band | 22-39% | - |

The conditional recovery rate is 34.6-34.7% in BOTH arms, sitting inside the
historical 22-39% band and essentially unmoved by the treatment: the note
recovers about a third of the traces it is shown, and that is a constant.
What the treatment changes is the OTHER column - a fix-arm trace that had
already placed IRAC correctly keeps doing so on the next attempt 80.8% of
the time against the control's 47.4%. The PROMPT is what makes the good
behaviour stable across retries; the note is not.

## 9. Judge spot-check - does the new summarization form survive the fleet?

`-m tuned.data.judge --config configs/data_law_v1_exp_irac_fix.yaml
--n-workers 3` run in increments of 2, 4 and 2 batches, with the arm's
`budget_ledger` read between each.

Seats filled exactly as designed: slot A `groq/qwen/qwen3.6-27b` (26 calls),
slot B `cerebras/gemma-4-31b` (25), tiebreak `mistral/mistral-large-latest`
(2). **No `gpt-5` call line appeared** - grepped, not eyeballed, across all
three logs. Cerebras spend for this step **119,132 tokens** against the
250,000 stop (116,180 prompt + 2,952 completion). One groq 429, recovered.

Accept = all three dimensions >= 4 from BOTH judges (the informal rule; no
`judge_threshold` row is active, so the worker's own dispositions are
PROVISIONAL and are not what this table counts).

| subset | accept | n |
|---|---|---|
| **summarization** | **10** | **11** |
| irac_analysis | 12 | 14 |
| total | 22 | 25 |

The single summarization miss is gen 26 (A 4/2/3, B 3/2/3) - both judges
independently marked it down on validity and coverage, which is a substance
verdict rather than a format one; nine of the ten accepts are straight 5/5/5
from both seats. **10/11 against a blocking bar of 5/10: the format change
does not cost judge acceptance.**

## 10. Run economics

| arm | prompt tok | completion tok | total | requests | 429s | wall |
|---|---|---|---|---|---|---|
| ctl generation | 286,235 | 383,389 | 669,624 | 108 | 0 | 29.4 min |
| fix generation | 251,377 | 385,352 | 636,729 | 92 | 0 | 24.2 min |
| fix judging (spot-check) | 241,898 | 5,927 | 247,825 | 53 | 1 | - |

All generation spend is `bai`, which is free. Judge spend is groq + cerebras
(both free) plus 2 mistral tiebreak calls (free tier). `reply_over_budget`
events: ctl 30, fix 34.

## 11. Verdict

| # | line | verdict |
|---|---|---|
| 1 | irac_placement fail rate, fix <= ctl - 15pp | **PASS** (-26.61pp, 1.8x the bar) |
| 2 | length_band pass rate, fix >= ctl - 5pp | **FAIL** (-8.86pp, breach of 3.86pp) |
| 3 | full-gate clean rate, fix >= ctl | **PASS** (+12.20pp) |
| 4 | summarization, every other gate within +-5pp | **FAIL** (4 of 11; 3 are improvements, the 4th is length_band again) |
| 5 | all deepseek, $0 non-bai, fingerprint unchanged | **PASS** |

### Recommendation: SHIP F2. DO NOT SHIP F1.

**F2 ships.** It is the whole of the measured win: -58.64pp on the gate for
the task type it touches, a think-side rehearsal rate collapsing from 70.8%
to 4.9%, a clean rate that nearly doubles, 10/11 judge acceptance of the new
answer form, and an answer that is the same length as before. It is also the
only change here that closes a real DEFECT rather than adding an
instruction: the four-heading mandate contradicted
`gates.IRAC_ANSWER_TASK_TYPES`, and removing a contradiction the gate had
already resolved is worth doing on its own terms.

**F1 does not ship on this evidence.** On `irac_analysis`, the only task
type that isolates it, it moved the primary -0.88pp while lengthening
thinking 14.7% and costing 4.11pp of `length_band` pass rate. It may be
underpowered rather than inert (section 6), but "may help, measurably costs"
is not a shipping case, and F1 is the change most likely to be carrying the
guard breach that failed lines 2 and 4.

**The breach is real and is not waved through.** Two of five pre-registered
lines failed. The honest summary is that the treatment bought a large,
cleanly attributed gate win and paid for it in reasoning length: 44% more
completion tokens per length-passing row. Against that, end-to-end yield
improved sharply - 24 usable tasks against 15, from 92 calls against 108 -
so the arm is cheaper per USABLE row even while being dearer per
length-passing generation.

### Named follow-ups

1. Re-run the pair with **F2 alone** in the overlay. It isolates F2 and
   tests the specific prediction that dropping F1 recovers most of the
   `length_band` cost while keeping the -58.64pp.
2. If F2-alone still breaches `length_band`, the next lever is the band
   itself for summarization, not another prompt sentence - three prompt
   levers (ceiling wording, `reasoning_effort` floor, `thinking: disabled`)
   are already exhausted on this provider.
3. `verbatim_overlap` now fails ~60% of summarization rows in BOTH arms and
   is the single largest remaining blocker on the clean rate. It is
   untouched by this work and is the obvious next target.

---

# F2-only confirm (same investigation, second pair)

Section 11 above recommended shipping F2 and dropping F1, and named the
follow-up that would have to justify it: **re-run the pair with F2 alone**,
on the prediction that dropping F1 would recover most of the `length_band`
cost while keeping the -58.64pp. That run is below. **The prediction was
wrong**, and the pre-registered decision rule therefore does not return a
ship.

## 12. Apparatus

Two new arms, `data/build/exp_irac_ctl3` and `data/build/exp_irac_f2only`,
registered in `paths.ISOLATED_WORKDIR_SIBLINGS` with three new tests
(`test_the_irac_ctl3_arm_is_an_isolated_workdir`,
`test_the_irac_f2only_arm_is_an_isolated_workdir`,
`test_the_f2only_confirm_arms_are_fenced_and_differ_only_in_the_prompt_overlay`).
Suite: **164 passed, 0 failed**.

Configs `configs/data_law_v1_exp_irac_ctl3.yaml` and
`configs/data_law_v1_exp_irac_f2only.yaml`, both cloned from
`configs/data_law_v1_exp_irac_ctl.yaml`. **Neither arm carries judge
routing** - the format question was settled by the first run's spot-check at
10/11 accept - so unlike the first pair these two differ in `workdir` and
`prompt_overlay` and nothing else, and the pairing test asserts full
line-level equality on the rest.

Overlay `data/build/exp_irac_f2only/prompts_f2only`: all 14 current base
`gen_*.md`, with ONLY `gen_summarization_v1.md` and `_v2.md` changed, by
exactly the F2 answer-format rewrite printed in section 2. The F1 sentences
appear nowhere in it.

| check | result |
|---|---|
| 12 of 14 byte-identical to base | PASS |
| 2 differing, and only by the F2 rewrite | PASS |
| re-applying F1 to those 2 reproduces the combined overlay byte for byte | **PASS** |
| the 8 files neither edit touches identical in both overlays | PASS |
| the 4 irac_analysis files are base copies here, F1-carrying there | PASS |
| all 14 LF-only, floor sentence "450 to 700 words of deliberation is normal" intact | PASS |

The third row is the load-bearing one: it proves this arm differs from the
combined arm by exactly F1, rather than merely asserting it.

The two summarization overlay shas are `b7c53ce9254f`
(`gen_summarization_v1.md`) and `21155223e534` (`_v2.md`); the other twelve
carry their base shas from the table in section 3. The arm stores' own
`prompt_sha` values confirm it live: **all four `gen_irac_analysis_*` shas
are identical across the two arms**, and only the two summarization shas
differ.

Apparatus commit `7402ee1`, made before any generation.

## 13. Run

Seeded and planned exactly as the first pair (`--seed 3407 --per-source
200`, 600 seeds each, same three stratified `tasks` commands, 40 tasks per
arm = 22 irac_analysis + 18 summarization, identical task ids across arms).

| arm | window (UTC) | wall | gap before | batches | claimed | gen-ok | errors |
|---|---|---|---|---|---|---|---|
| ctl3 | 06:16:21.161 - 06:43:38.185 | 27.3 min | - | 22 | 100 | 100 | 0 |
| f2only | 06:44:21.190 - 07:18:28.444 | 34.1 min | **43.0 s** | 20 | 91 | 91 | 0 |

## 14. The five pre-registered lines

| # | measurement | ctl3 | f2only | delta | line | verdict |
|---|---|---|---|---|---|---|
| 1 | **summarization irac_placement fail** | 72.73% (32/44) | **7.89% (3/38)** | **-64.83pp** | <= -30pp | **PASS** |
| 2 | **summarization length_band pass** | 65.91% (29/44) | 50.00% (19/38) | **-15.91pp** | >= -5pp | **FAIL** |
| 3 | irac_analysis, every gate | see below | see below | 2 of 12 outside band | +-5pp | **FAIL - drift warning** |
| 4 | full-gate clean rate | 16.00% (16/100) | 19.78% (18/91) | +3.78pp | >= 0 | **PASS** |
| 5 | all deepseek, bai-only ledger, fingerprint | 100/100, bai only | 91/91, bai only | - | hard | **PASS** |

n counts every generation attempt: ctl3 n=100, f2only n=91.

### Line 1 - F2 survives alone, and then some

72.73% -> 7.89%, **-64.83pp** against a -30pp bar. The combined arm managed
-58.64pp with F1 alongside; F2 by itself does slightly better. The mechanism
tracks it: rows with a line-initial IRAC label inside `<think>` go 65.91%
(29/44) -> **5.26% (2/38)** on summarization, while the untreated
irac_analysis templates sit still at 55.36% -> 58.49%.

**F1 contributed nothing to the win.** That is now measured twice.

### Line 2 - and F2 also owns the cost

This is the line the follow-up existed to test, and it fails, harder than in
the combined arm:

| pair | summarization length_band pass | delta |
|---|---|---|
| combined (F1+F2) | 56.25% -> 41.46% | -14.79pp |
| **F2 alone** | 65.91% -> 50.00% | **-15.91pp** |

Removing F1 did not recover the length cost - it left it exactly where it
was. **The `length_band` regression belongs to F2, not to F1.** Section 11's
reading, that F1 was "the change most likely to be carrying the guard
breach", is disproved by this run and should not be relied on.

The overall (both task types) length_band pass rate tells the same story:
-8.86pp in the combined pair, -8.14pp here.

### Line 3 - ARM NOISE WARNING, flag raised

The four `gen_irac_analysis_*` templates are byte-identical across these two
arms, so every irac_analysis difference below is drift in b.ai's hidden
upstream pool between 06:16 and 07:18, not a treatment effect.

| gate | ctl3 (n=56) | f2only (n=53) | delta | within +-5pp |
|---|---|---|---|---|
| **prompt_echo** | 10.71% (6/56) | 28.30% (15/53) | **+17.59pp** | **NO - DRIFT** |
| **verbatim_overlap** | 55.36% (31/56) | 41.51% (22/53) | **-13.85pp** | **NO - DRIFT** |
| statutory_grounding | 12.50% (7/56) | 7.55% (4/53) | -4.95pp | yes |
| think_format | 0.00% (0/56) | 3.77% (2/53) | +3.77pp | yes |
| self_verification | 7.14% (4/56) | 9.43% (5/53) | +2.29pp | yes |
| irac_placement | 60.71% (34/56) | 58.49% (31/53) | -2.22pp | yes |
| citations | 1.79% (1/56) | 0.00% (0/53) | -1.79pp | yes |
| length_band | 60.71% (34/56) | 62.26% (33/53) | **+1.55pp** | yes |
| banned_meta | 16.07% (9/56) | 15.09% (8/53) | -0.98pp | yes |
| answer_key / statutory_quotation / temporal | 0.00% | 0.00% | 0.00pp | yes |

**Two gates drifted by 13-18 points on templates that did not change.** That
is a real measurement-validity finding and it is why this line was
pre-registered. Read it as follows, and no further:

- The drift is **gate-specific, not global**. It hit `prompt_echo` and
  `verbatim_overlap`. It did not hit either gate the treatment lines are
  scored on.
- **The noise floor under line 1 is 2.22pp.** A -64.83pp effect is roughly
  29x it. Line 1 is not in question.
- **The noise floor under line 2 is 1.55pp.** A -15.91pp cost is roughly 10x
  it, and it reproduces the combined arm's -14.79pp to within 1.1pp on a
  different pool draw. Line 2's failure is not noise either.
- What the drift DOES contaminate is any absolute cross-pair comparison of
  `prompt_echo`, `verbatim_overlap`, and the full-gate clean rate that
  depends on them. Line 4's +3.78pp should be treated as soft for that
  reason.

## 15. Secondaries

| metric | ctl3 | f2only | delta |
|---|---|---|---|
| think est-tokens (chars/4) p50 | 2,718 | 3,109 | +14.4% |
| think est-tokens p90 | 6,784 | 6,757 | -0.4% |
| - summarization est p50 | 2,255 | 2,792 | **+23.8%** |
| - irac_analysis est p50 (untreated) | 3,436 | 3,633 | +5.7% |
| **completion tokens per length-passing row** | 7,430.3 (378,944/51) | **9,051.8 (353,021/39)** | **+21.8%** |
| summarization answer chars p50 | 4,473 | 4,788 | +7.0% |
| tasks reaching `judging` (of 40) | 16 | 18 | +2 |
| - summarization | 6 | **10** | +4 |
| - irac_analysis (untreated) | 10 | 8 | -2 |

The shape of the cost is now unambiguous. F2 makes the model deliberate
longer on summarization (+23.8% at p50) and write a slightly longer answer
(+7.0%), and both land inside the same 8,192-token band. The p90 barely
moved, so this is the middle of the distribution shifting right rather than
a tail of runaway traces.

Run economics: ctl3 263,825 prompt / 378,944 completion / 100 requests / 0
429s; f2only 240,744 / 353,021 / 91 / 0. All spend is `bai`, which is free.

## 16. Verdict

| # | line | verdict |
|---|---|---|
| 1 | summarization irac fail, f2only <= ctl3 - 30pp | **PASS** (-64.83pp, 2.2x the bar) |
| 2 | summarization length_band pass, f2only >= ctl3 - 5pp | **FAIL** (-15.91pp, 3.2x over) |
| 3 | irac_analysis every gate within +-5pp | **FAIL** (prompt_echo +17.59pp, verbatim_overlap -13.85pp - upstream drift, flagged) |
| 4 | full-gate clean rate, f2only >= ctl3 | **PASS** (+3.78pp, soft - see line 3) |
| 5 | integrity | **PASS** |

Ship rule as pre-registered: lines 1, 2 and 4 must all pass. **Line 2 fails.**

### Recommendation: NO-SHIP for F2 as it stands.

Not because the edit does not work - it works better isolated than combined,
and the -64.83pp is the largest, cleanest effect measured anywhere in this
investigation. It is a no-ship because the follow-up asked one question,
"is the length cost F1's?", and the answer came back **no, it is F2's**. The
combined arm's guard breach was never F1's to carry, so removing F1 buys
nothing on that front, and F2 cannot ship on the strength of a guard that
fails identically with or without it.

What this pair did settle, and it is worth the run:

1. **F2 is the entire treatment effect.** Measured twice now: -58.64pp with
   F1 alongside, -64.83pp without it.
2. **F1 is dead.** It contributes nothing to the win in either pair and
   should not appear in any future arm.
3. **The length cost is F2's and is structural**, not incidental: taking the
   four headings away makes the model deliberate longer in prose. It
   reproduces across two independent pairs to within 1.1pp.

### Named next step

One lever, and it is no longer a prompt lever - three of those are already
exhausted on this provider. F2's win is large enough to be worth buying, and
what it costs is band headroom on summarization specifically. The next arm
should pair F2 with a **summarization-specific `length_band`**, pre-registered
on whether the clean rate and the usable-row count rise together once the
band stops charging F2 for the deliberation it causes. Note that
summarization tasks reaching `judging` already rose 6 -> 10 in this pair
despite the band cost, which is the first direct evidence that the trade may
be worth making explicit.

A second, cheaper practice worth keeping: the `prompt_echo` and
`verbatim_overlap` drift in section 14 was large. Any future arm on this
provider should keep an untreated task type in the design purely as a noise
channel - it cost nothing here, and it is the only reason line 2's failure
can be called real rather than atmospheric.

## Ship record (appended post-review)

The F2-only confirm returned NO-SHIP by its pre-registered rule (line 2, the summarization
`length_band` guard, failed at −15.91pp against a −5pp allowance). The operator overrode that
verdict on end-to-end totality and F2 shipped in `ebde9a7`: summarization irac_placement fail
−64.83pp (~29× the run's measured noise floor), tasks reaching the judges 6→10 (+67%), the
new prose format judge-accepted 10/11, and the edit closes a real defect — the templates
mandated a four-heading answer that `gates.IRAC_ANSWER_TASK_TYPES` deliberately stopped
requiring. The −15.91pp band cost ships with it, known and structural (prose deliberation
runs longer at the median; p90 flat).

Dispositions: **F2 shipped** (`ebde9a7`, two summarization templates byte-identical to the
f2only overlay). **F1 closed** — twice measured inert on its own task type and expensive
(think p50 +23.7%). **F3 kept** — the rewritten retry note remains inert at converting
failures (within the historical 22–39% band) but is better aimed and harmless; the prompt,
not the note, is what stabilizes placement.

Recorded follow-ups: a summarization-specific `length_band` (code + config feature) to
recover the band cost; `verbatim_overlap` (~60% summarization fail in both arms) as the next
gate frontier; `gen_drafting_v1/v2` carry the same four-heading-mandate drift outside
`IRAC_ANSWER_TASK_TYPES`, unmeasured today and needing its own arm before any edit.
