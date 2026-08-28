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
