# hy3 think-low qualification probe: docs mislead on the parameter, and the model fails the length gate anyway

2026-08-28. Qualification probe, not a shipping change - ships nothing to
the live config. Question: can b.ai's free `hy3` model (Tencent Hunyuan)
serve as a second generator family alongside `bai/deepseek-v4-flash`, run
at its documented LOW thinking tier. **FAIL.** The primary gate-yield line
misses its bar by a factor of three (19.44% against a >=60% bar), and a
secondary line (`irac_placement`) is worse than the deepseek baseline it
would have to beat. This closes the free-alternatives question for `hy3` as
tested; it does not close the question for every possible binding of it
(see §8).

---

## 1. Doc research: the API surface, and where it stopped being reliable

Two pages fetched 2026-08-28: `https://docs.b.ai/llmservice/api/` and
`https://docs.b.ai/llmservice/models/hy3/`.

**Context/output limits**, quoted from the hy3 model page: "Context window:
256K tokens | Maximum input: 192K tokens | Maximum output: 128K tokens."

**Reasoning tiers**, quoted from the same page: "Reasoning: Supports direct
and deeper reasoning modes: `no_think`, `think_low`, and `think_high`."

**The selection mechanism is not documented for this model.** The hy3 page
names the three modes as a capability and gives no accompanying parameter,
no model-id suffix (no string `hy3-think-low` appears anywhere on it), and
no code/curl/JSON example. The API reference page documents a
`"reasoning": {"effort": ...}` object, but only for the separate
`/v1/responses` endpoint ("Available reasoning levels depend on the
selected model"), never for `/v1/chat/completions` - the endpoint this
build's `ChatClient` actually calls. Third-party rehosts of the same open
weights (DeepInfra, aimlapi) document `extra_body: {chat_template_kwargs:
{reasoning_effort: "..."}}` for their own serving stacks; that convention
is not stated anywhere on `docs.b.ai` and was not assumed here.

This is short of the task's BLOCKED bar - a parameter exists
(`reasoning_effort`, already working for `bai/deepseek-v4-flash` on this
identical endpoint) even though the doc does not bind it to hy3's named
tiers - so the probe proceeded to a live test rather than stopping.

## 2. The live 400: the doc's own tier names are not valid API values

The apparatus was first built with `reasoning_effort: think_low` - hy3's
own doc vocabulary, passed as the value of the same key that already works
for deepseek. A pre-run sanity call (`python -m tuned.data.providers
--config configs/data_law_v1_exp_hy3.yaml --check --ref bai/hy3`) came back
**HTTP 400** before a single generation task ran:

```
{"error":{"message":"The request is invalid: reasoning_effort must be one
of: no_think, none, off, minimal, low, medium, high, xhigh, max. Please
check the request body, required fields, and reques[...]"}}
```

So b.ai's gateway does not expose `think_low`/`think_high` as API strings
at all - it folds hy3 into the same generic ladder every other model in
this fleet's config uses, plus `no_think` kept as its own value. A direct
`ChatClient` call (bypassing the CLI's truncated error rendering) confirmed
both ends of the ladder relevant here:

| `reasoning_effort` value | HTTP | reasoning tokens (trivial "reply OK" prompt) |
|---|---|---|
| `low` | 200 | 23 (a real trace, `message.reasoning` populated) |
| `no_think` | 200 | 0 |

`reasoning_effort: low` is therefore the binding this probe actually ran -
corrected in the config and covered by a live 400 in the commit history
(`0150b1c`) before any of the 25 tasks were dispatched. **The qualification-
relevant fact here is itself a docs-vs-API divergence**: a caller that
trusted hy3's own doc vocabulary literally would have blocked on a 400
before generating a single row.

## 3. Apparatus

- Workdir `data/build/exp_hy3`, registered in
  `src/tuned/data/paths.py::ISOLATED_WORKDIR_SIBLINGS`, covered by
  `tests/test_build_config.py::test_the_hy3_arm_is_an_isolated_workdir`.
- `configs/data_law_v1_exp_hy3.yaml` - `configs/data_law_v1_exp_ds_ctl2.yaml`
  with three edits: `build.workdir -> data/build/exp_hy3`; one new model
  entry in the `bai` provider block, `id: hy3, family: hy, roles:
  [generator], limits: {rpm: 8, max_context: 192000, max_output: 16384},
  params: {temperature: 0.7, top_p: 0.95, reasoning_effort: low}`
  (`deepseek-v4-flash` left declared but unused in the same block);
  `routing.generator -> [bai/hy3]` (single ref). `family: hy` is new and
  distinct from every family already in the fleet (deepseek, gpt-oss,
  gemma, qwen, mistral), so `family_separation` holds trivially. The openai
  `usd_cap: 0.0` fence and `length_band` (`think_max: 3000, total_max:
  8192`) carry over unchanged from the live config. Covered by
  `tests/test_build_config.py::test_the_hy3_probe_config_is_fenced_and_carries_the_new_model`.
- `max_context` is the doc-stated MAX INPUT (192,000), not the larger total
  window (256,000) - the routing-relevant prompt-side ceiling.
  `max_output` is deliberately **not** the doc's 128,000 ceiling:
  `_bai_request_hook` (`src/tuned/data/providers.py`) raises every
  generation call's `max_tokens` up to `limits.max_output` because bai
  models bill reasoning against the same budget and emit it first, so
  128,000 would have let every one of this probe's calls request a
  128,000-token reply against an unverified tier binding. 16,384 mirrors
  `deepseek-v4-flash`'s own proven-safe value in this exact mechanism. This
  paid off: 0 of 72 generation attempts in the run hit a truncation error at
  that ceiling (see §6).
- Apparatus committed before any generation ran: `5624471` (paths.py +
  config + test), `0150b1c` (the `think_low` -> `low` correction after the
  live 400, §2).

## 4. Seeding and planning

Live store fingerprint checked before touching anything: `554532864
1787309490` (matches the required value). Seeded read-only from the live
store:

```
scripts/seed_exp_store.py --config configs/data_law_v1_exp_hy3.yaml --from data/build/state/law_v1.sqlite3 --per-source 200 --seed 3407
```

600 seeds (200 per source across the same three eligible sources every
recent bai arm has used: `s3://indian-supreme-court-judgments`,
`L-NLProc/PredEx_Instruction-Tuning_Pred-Exp`,
`L-NLProc/TathyaNyaya-and-FactLegalLlama-NyayaFacts-Datasets`). Fingerprint
re-checked after seeding - unchanged. Planned to ~25 tasks total, split
across the three sources:

```
tuned.data.tasks --config configs/data_law_v1_exp_hy3.yaml --stream synthesis --arm sc     --n 9 --source "s3://indian-supreme-court-judgments"                                       --mix irac_analysis=0.55,summarization=0.45
tuned.data.tasks --config configs/data_law_v1_exp_hy3.yaml --stream synthesis --arm predex --n 8 --source "L-NLProc/PredEx_Instruction-Tuning_Pred-Exp"                                --mix irac_analysis=0.55,summarization=0.45
tuned.data.tasks --config configs/data_law_v1_exp_hy3.yaml --stream synthesis --arm tathya --n 8 --source "L-NLProc/TathyaNyaya-and-FactLegalLlama-NyayaFacts-Datasets"                --mix irac_analysis=0.55,summarization=0.45
```

25 pending tasks (13 irac_analysis, 12 summarization). Fingerprint
re-checked after planning - unchanged.

## 5. Run

```
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tuned.data.generate --config configs/data_law_v1_exp_hy3.yaml --n-workers 5 --max-batches 15
```

02:36:48 - 02:51:35 UTC, 14.8 minutes, 15 batches, 72 generation attempts
over the 25 tasks (retries included), 0 errors, 0 HTTP 429s. Final task
states: 23 `format_parked` (exhausted 3 attempts on a format/gate failure),
2 `rejected`. **No task reached `judging`** - every task exhausted its
attempts before clearing every non-diagnostic gate at once (see §6.4).

The first three responses were watched live before committing to the full
run (§2's `--check` probe, then the first three real generations):
`finish_reason=stop` on all three, non-empty `think` and `answer`, real
`think_tokens` in the low thousands. No empty-content or unparseable
replies appeared at any point in the run - see §6.2.

## 6. Measurement

All counts are over **72 generation attempts** (every retry), the same
currency the deepseek clause/cap A/B used - not 25 unique tasks, since
retries are real spend.

### 6.1 PRIMARY - length_band gate yield

| line | measured | bar | verdict |
|---|---|---|---|
| length_band pass rate | **19.44%** (14/72) | >= 60% | **FAIL** |

The dominant violation is `think>think_max`: `think_tokens` (chars/4 est.)
ranged 1,810-6,012 across the 72 attempts against a `think_max` of 3,000.
Sampled violation rows show `think>think_max` alone or alongside
`total>total_max`; no row failed on `think<think_min` or `answer<answer_min`.
hy3's `low` tier, as bound here, produces traces that regularly exceed this
build's `think_max` band - a band tuned against `deepseek-v4-flash`'s own
`low` behaviour, not hy3's.

### 6.2 PRIMARY - format integrity

| line | measured | bar | verdict |
|---|---|---|---|
| format breakage | **0%** (0/72) | 0% | **PASS** |

`think_format` gate: 72/72 passed. `generation.think IS NULL`: 0 rows.
`generation.answer` empty/null: 0 rows. `finish_reason`: `stop` on all 72 -
no truncation, no empty-content-on-length event of the kind
`_bai_response_hook` exists to catch. Every non-empty reply parsed into
think + answer through the normal `assemble_content`/`split_think` path.

### 6.3 Secondary - irac_placement (does hy3 pre-draft the headed answer in its trace)

| metric | measured |
|---|---|
| irac_placement fail rate | **95.83%** (69/72 failed, 3/72 passed) |
| - failures caused by IRAC headings appearing in `think` | 69/72 |
| - failures caused by headings missing from the answer | 0/72 |

Every failure is the rehearsal pathology, never a missing-heading defect:
`missing_in_answer` is empty on all 72 rows (the model always labels the
final answer correctly), but 69 of 72 also write a full
Issue/Rule/Application/Conclusion-headed draft inside `<think>` first -
worse than the deepseek `ctl2` control's 73.39% measured 2026-08-28 in
`docs/reports/2026-08-28-deepseek-clause-and-cap-ab.md`. hy3 pre-drafts the
headed answer in its trace *more* often than deepseek does, not less.

### 6.4 Secondary - think tokens, full-gate clean rate, spend per passing row

| metric | value |
|---|---|
| think est-tokens (chars/4) p50 | 3,440 |
| think est-tokens p90 | 5,180 |
| think est-tokens mean | 3,665 |
| think est-tokens range | 1,810 - 6,012 |
| full-gate clean rate (every non-diagnostic gate passed) | **0.00%** (0/72) |
| completion tokens, mean over the 14 length_band-passing rows | 3,266.1 |
| completion tokens per length_band-passing row, true spend (`budget_ledger` completion total / passing count) | **21,424.4** (299,941 / 14) |

The true-spend figure is the honest one (same convention the
clause/cap A/B used): it charges the tokens burned on the 58 attempts that
never passed `length_band` against the 14 that did, rather than averaging
only over the survivors.

Per-gate pass counts, all 72 attempts:

| gate | passed / total |
|---|---|
| answer_key | 72/72 |
| self_verification (diagnostic) | 72/72 |
| statutory_quotation | 72/72 |
| temporal | 72/72 |
| think_format | 72/72 |
| citations | 70/72 |
| statutory_grounding | 68/72 |
| banned_meta | 62/72 |
| prompt_echo | 46/72 |
| verbatim_overlap | 33/72 |
| length_band | 14/72 |
| irac_placement | 3/72 |

## 7. Spot-read (5 generations, legal sanity only - not a judge pass)

Five distinct final-attempt generations, mixed `irac_analysis`/
`summarization`, mixed length_band outcome:

1. **Akhara trust dedication** (gen 43, the one row that passed both
   primary lines *and* irac_placement) - dominant-object charitable-trust
   reasoning, cites the Statute of Elizabeth and the *Saraswathi Ammal*
   dominant-object test correctly for a Hindu Law religious/charitable
   dedication question. Coherent.
2. **SCBA/SCAORA chamber-allotment constitutional challenge** (gen 11) -
   Articles 14/19(1)(c)/19(1)(g) framing is right for the fact pattern;
   quotes *Shayara Bano* ("What is manifestly arbitrary is obviously
   unreasonable and, being contrary to the rule of law, would violate
   Article 14") accurately. Coherent.
3. **Caste-validity-certificate scrutiny** (gen 30) - correctly invokes
   *Kumari Madhuri Patil* for the Scrutiny Committee/Vigilance Cell
   machinery and tracks a quash-then-stay procedural posture without
   confusing the two orders. Coherent.
4. **UP Secondary Education ad hoc appointment** (gen 12) - Section 16/18
   and the Removal of Difficulties Orders are used consistently with the
   *Radha Raizada* Full Bench framing named in the materials. Coherent.
5. **Railway retirement-age headnote** (gen 41) - correctly isolates Rule
   2046(b)'s two conjunctive conditions (entry date, and lien/suspended
   lien/provisional-substantive status as of the same date) and reasons
   through why a later confirmation cannot retroactively supply the second
   limb. Coherent.

**Verdict: no obvious nonsense in any of the 5** - no fabricated statute
numbers, no garbled or invented case names, no repetition loops, correct
use of real, contextually-appropriate precedent in every sample. This is
not a judge-accepted-quality signal (see §8) and n=5 is not a claim of
representativeness; it only rules out gross generation failure.

One incidental pattern worth naming: every sampled `<think>` block includes
explicit self-directed word-count and structure planning ("Need word
counts... Let's craft... Now produce final answer with reasoning first
(no headings) then headnote"), i.e. hy3 narrates its own compliance with
the prompt's formatting rules inside the trace. This is distinct from the
graded irac_placement rehearsal (§6.3) and was not gated on here, but it is
the same family of "the trace is aware of its own output contract" habit.

## 8. Integrity

- Every one of the 72 generation rows is `provider=bai, model=hy3,
  model_family=hy` - confirmed by `SELECT DISTINCT provider, model,
  model_family FROM generation`, not sampled.
- `budget_ledger` names exactly one (provider, model): `(bai, hy3)`. $0
  spend - bai is free tier, and no other provider was touched (this run is
  generate-only; no judge/tiebreak/probe call was dispatched).
- Zero HTTP 429s, zero generation errors across all 72 attempts
  (`errors=0` in the run summary; `budget_ledger.errors_429 = 0`).
- Live store `data/build/state/law_v1.sqlite3` fingerprint `554532864
  1787309490` - checked before seeding, after seeding, after planning, and
  after the run. Identical every time; the store was never opened for
  write.
- No `.env` contents were printed at any point; every invocation logged
  only `loaded 18 key(s) from .env`.
- Apparatus commits: `5624471` (workdir/config/test), `0150b1c` (the
  `reasoning_effort` correction after the live 400). This report:
  see the commit that adds this file.

## 9. Verdict and what it does not establish

**FAIL.** The primary length_band line misses its >=60% bar by a factor of
three (19.44%), driven almost entirely by traces that run well past this
build's `think_max: 3000` band. The irac_placement secondary is worse than
the deepseek baseline it would have needed to match or beat (95.83% fail
vs. 73.39%). Format integrity is clean (0% breakage) and the spot-read
found no gross generation failures, but neither offsets a primary miss this
large. This closes the free-alternatives question for `hy3` **as bound and
banded here** - `reasoning_effort: low` against this build's existing
`length_band`.

**What this does NOT establish:**

- **Not judge-accepted quality.** No judge was dispatched in this run
  (generate-only, matching every recent bai arm). Nothing here speaks to
  whether a gate-passing hy3 row would be accepted by the judge fleet -
  there are only 14 gate-passing rows to begin with, and 0 that cleared
  every non-diagnostic gate.
- **Not full-scale behaviour.** n=72 generation attempts over 25 tasks in
  one ~15-minute window is a probe, not a calibration - the same
  qualification the deepseek arms this week were run under.
- **Not a verdict on every possible hy3 binding.** Only `reasoning_effort:
  low` against the unmodified live `length_band` was tested. `no_think`
  (confirmed live to produce zero reasoning tokens, §2) or a `think_max`
  recalibrated for hy3 specifically were not tried and might move the
  primary line - but doing so is exactly the "follow-up qualification" this
  probe's FAIL closes the door on opening by default, per the task's own
  framing. Nothing here recommends chasing it.
- **Not a claim about b.ai's documentation in general.** Only the hy3 page
  and the reasoning-tier binding were checked; other models' doc pages were
  not re-verified against live behaviour here.
