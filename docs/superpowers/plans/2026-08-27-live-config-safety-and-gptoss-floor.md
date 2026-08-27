# Live-Config Safety Fix and gpt-oss Floor Measurement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close an uncapped paid-judge path in the live config, restore gpt-oss as lead generator, raise `think_max` to the point where returns stop, and then measure whether the committed prompt ceiling harms the generator that is now lead.

**Architecture:** Two tasks. Task 1 is three edits to `configs/data_law_v1.yaml` plus the test re-baselining they force — small, urgent, and entirely offline. Task 2 is a paired A/B with both arms run back to back in one sitting, building its own control from the pre-edit templates via `prompt_overlay`.

**Tech Stack:** Python 3.12, pytest, SQLite, PyYAML, `tuned.data` build pipeline, cerebras `gpt-oss-120b` over an OpenAI-compatible endpoint.

**Spec:** `docs/superpowers/specs/2026-08-27-live-config-safety-and-gptoss-floor-design.md`

## Global Constraints

- **No attribution or watermarks of any kind** in commits, code, comments, docs, or reports. Absolute; overrides any default commit-trailer behaviour.
- **The live control store `data/build/state/law_v1.sqlite3` is never opened for write.** Read-only `file:...?mode=ro` URIs only. Its size and mtime must be identical after every task.
- **`usd_cap: 0.0` alone blocks nothing.** `generate._usd_per_1m` returns `0.0` for a missing price, so `0 + 0 > 0.0` is `False`. A zero cap must ship with `usd_per_1m_prompt` and `usd_per_1m_completion` beside it.
- **`total_max: 8192` and `max_seq_length: 8192` are the student's training context and must not be raised.** The 2×T4 OOM ladder goes *down* to 6144.
- Write config files as LF bytes. Do not use `Path.write_text` on Windows for them — it emits CRLF and has already once buried four real edits in an 886-line diff.
- Never guess a model repo revision sha; use `scripts/pin_revision.py`.
- Tests: `./.venv/Scripts/python.exe -m pytest`. The suite is green at 3572 passed / 19 skipped / 0 failed. Any failure is yours.

---

### Task 1: Close the cost path, restore gpt-oss as lead, raise `think_max`

**Context:** `openai/gpt-5-mini` and `gpt-5-nano` are declared `family: gpt-oss` so that `family_separation` excludes them when the generator is gpt-oss. This branch made `bai/deepseek-v4-flash` the lead generator, which excludes only `{deepseek}` — so the paid judges became reachable, on a config with no spend cap anywhere.

**Files:**
- Modify: `configs/data_law_v1.yaml`
- Modify: whichever tests assert the generator order or `think_max` (expect several; re-baseline them honestly)

**Interfaces:**
- Consumes: nothing.
- Produces: the shipped `length_band` that Task 2's arms both use, and the generator order Task 2 measures.

- [ ] **Step 1: Prove the fence trap before relying on the fence**

```bash
./.venv/Scripts/python.exe -c "
from tuned.data.config import load_build_config
from tuned.data import generate
cfg = load_build_config('configs/data_law_v1.yaml', allow_unpinned=True)
for p in cfg.providers:
    if p.name == 'openai':
        for m in p.models:
            print(m.id, 'usd_cap=', m.limits.get('usd_cap'),
                  'p=', m.limits.get('usd_per_1m_prompt'),
                  'c=', m.limits.get('usd_per_1m_completion'))
"
```
Expected right now: no `usd_cap` at all. Record it.

- [ ] **Step 2: Add the fence to BOTH gpt-5 models**

```yaml
        limits: {rpm: 500, tpm: 200000, max_context: 400000, max_output: 16384,
                 usd_cap: 0.0, usd_per_1m_prompt: 0.25, usd_per_1m_completion: 2.0}
```

Add a comment above them recording: that the cap is zero so any positive price blocks at the first token; that a bare `usd_cap: 0.0` blocks nothing because `_usd_per_1m` returns `0.0` for a missing price; and that mini's prices are used for both because at a zero cap the exact figure cannot matter. Do not imply a nano price was looked up.

- [ ] **Step 3: Verify the fence on the SHIPPED config, both ways**

Call the real `generate.budget_ok_for` against the loaded live config for both gpt-5 models with a non-trivial token count — it must return `False`. Then re-run with the two price keys stripped from the loaded objects — it must return `True`, proving the prices are what block rather than the cap. Save both results to `data/build/exp_gptoss_ctl/out/fence_check.txt` (create the directory) and paste them into your report. A fence you did not watch reject something is not a verified fence.

- [ ] **Step 4: Demote deepseek from lead generator**

```yaml
  generator: [cerebras/gpt-oss-120b, bai/deepseek-v4-flash,
              lightning/lightning-ai/gpt-oss-120b]
```

Leave `routing.tiebreak` exactly as it is — mistral stays first, because gpt-oss-20b must not hold the deciding seat on a deepseek row and deepseek is still ref 2. Record that reason in a comment so the next reader does not "tidy" it back.

- [ ] **Step 5: Raise `think_max` to 4000**

In `build.length_band`, change `think_max: 3000` to `think_max: 4000`. Leave every other band value alone — in particular `total_max: 8192`, which is the student's context.

Record this table in a comment, measured over the 99 banked v4 generations in `data/build/exp_deepseek/state/law_v1.sqlite3`:

```
#   think_max   length_band pass   blocked by total_max alone
#   3000              49.5%              4.0%
#   4000              60.6%             10.1%
#   4500              63.6%             16.2%
#   inf               64.6%             33.3%
```

and the point it makes: the gain saturates because past 4000 the binding constraint is `total_max`, which no config change can lift.

- [ ] **Step 6: A4 — deepseek's judge role, conditional**

Determine whether `role_params` can carry `thinking: {"type": "disabled"}` through to the request payload for the judge role. Read `providers.build_payload` and how `role_params` merges.

- If it can: set deepseek `roles: [generator, judge, tiebreak]`, append it to `routing.judge` and `routing.tiebreak`, add the `role_params`, and add a test that a judge-role bai payload carries thinking-disabled.
- If it cannot: **stop, change nothing for A4, and report it.** `_bai_request_hook(payload, model)` is not role-aware, and this repo already has a commit named "a judge that spends its reply budget thinking is not a judge". Do not make the hook role-aware as a side quest — that is its own change with its own validation.

- [ ] **Step 7: Re-baseline the tests this breaks**

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q 2>&1 | tail -30
```

Expect failures asserting the generator order or `think_max: 3000`. For each, decide and say which in your report: the assertion encodes an invariant that genuinely changed (update the expected value), or it pinned a literal where it meant an invariant (rewrite it to assert the invariant). Prefer the second. Do not weaken an assertion to make it pass — if one cannot pass without changing production behaviour, stop and report.

- [ ] **Step 8: Verify green and confirm the live store is untouched**

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q 2>&1 | tail -5
./.venv/Scripts/python.exe -c "import os;s=os.stat('data/build/state/law_v1.sqlite3');print(s.st_size, int(s.st_mtime))"
```
Expected: 0 failed; `554532864 1787309490`.

- [ ] **Step 9: Commit**

```bash
git add configs/data_law_v1.yaml tests/
git commit -m "fence the paid judges, put gpt-oss back in front, and stop rejecting rows that fit"
```

---

### Task 2: Does the prompt ceiling hurt gpt-oss? Both arms, one sitting

**Context:** Task 1 makes `cerebras/gpt-oss-120b` the lead generator. It reads the base `prompts/` templates — the live config sets neither `prompt_overlay` nor `harmony_completions`. Those templates were edited on this branch to replace a sentence that existed to push gpt-oss traces **up**, because gpt-oss's measured failure is `think < think_min`. Nobody has measured that combination.

**Files:**
- Modify: `src/tuned/data/paths.py` — add `exp_gptoss_ctl` and `exp_gptoss_new` to `ISOLATED_WORKDIR_SIBLINGS`
- Modify: `tests/test_build_config.py` — a test for each, beside the existing `exp_deepseek` one
- Create: `configs/data_law_v1_exp_gptoss_ctl.yaml`, `configs/data_law_v1_exp_gptoss_new.yaml`
- Create: `docs/reports/2026-08-27-gptoss-floor-under-the-prompt-ceiling.md`

**Interfaces:**
- Consumes: Task 1's shipped `length_band` (both arms use it, unchanged, so the comparison is like-for-like); `scripts/seed_exp_store.py` unchanged.
- Produces: the finding that decides whether the prompt edit survives.

- [ ] **Step 1: Declare both isolated workdirs**

Add `exp_gptoss_ctl` and `exp_gptoss_new` beside `exp_deepseek` in `paths.ISOLATED_WORKDIR_SIBLINGS`, with matching tests. Without this, every write guard treats the arms as the frozen live control.

```bash
./.venv/Scripts/python.exe -m pytest tests/test_build_config.py -q
```

- [ ] **Step 2: Build the control's prompt overlay from the pre-edit templates**

```bash
mkdir -p data/build/exp_gptoss_ctl/prompts_preedit
for f in $(git show --name-only --pretty=format: 286fd3a | grep '^src/tuned/data/prompts/'); do
  git show "f499372:$f" > "data/build/exp_gptoss_ctl/prompts_preedit/$(basename $f)"
done
ls data/build/exp_gptoss_ctl/prompts_preedit | wc -l
```
Expected: 14 files. Then confirm they are the PRE-edit versions — each must contain `is normal for a matter of any substance`, and none may contain a ceiling phrase. Report both counts.

Note `prompt_registry` may require the judge/probe templates present in an overlay directory; if the overlay loader errors on a missing id, copy those from the current tree unchanged (they were never edited) and say so.

- [ ] **Step 3: Write both arm configs**

Both are copies of `configs/data_law_v1.yaml` (post-Task-1, so both carry the fence and the new `think_max`) with:
- `build.workdir` set to the arm's own directory
- `routing.generator: [cerebras/gpt-oss-120b]` — a single ref, so a throttle cannot silently turn a gpt-oss arm into a deepseek one
- a header comment naming the arm and its twin

The **control** additionally sets `build.prompt_overlay: data/build/exp_gptoss_ctl/prompts_preedit`. The treatment sets no overlay.

Everything else identical between the two. Write as LF bytes.

- [ ] **Step 4: Fingerprint the live store**

```bash
./.venv/Scripts/python.exe -c "import os;s=os.stat('data/build/state/law_v1.sqlite3');print(s.st_size, int(s.st_mtime))" | tee data/build/exp_gptoss_ctl/out/live_stat_before.txt
```

- [ ] **Step 5: Seed and plan both arms identically**

For each arm, with its own config:

```bash
./.venv/Scripts/python.exe scripts/seed_exp_store.py --config <arm config> \
    --from data/build/state/law_v1.sqlite3 --per-source 200 --seed 3407
./.venv/Scripts/python.exe -m tuned.data.tasks --config <arm config> \
    --stream synthesis --arm sc --n 13 --source s3://indian-supreme-court-judgments \
    --mix irac_analysis=0.55,summarization=0.45
```
then `--arm predex --n 14 --source L-NLProc/PredEx_Instruction-Tuning_Pred-Exp` and `--arm tathya --n 13 --source L-NLProc/TathyaNyaya-and-FactLegalLlama-NyayaFacts-Datasets`, same `--mix`.

`--seed 3407` and `--per-source 200` match every prior arm; the same seeds must land in both.

- [ ] **Step 6: Run both arms BACK TO BACK, control first**

```bash
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tuned.data.generate \
    --config configs/data_law_v1_exp_gptoss_ctl.yaml --n-workers 3 --max-batches 30
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tuned.data.generate \
    --config configs/data_law_v1_exp_gptoss_new.yaml --n-workers 3 --max-batches 30
```

Run them in one sitting with no gap. This is the direct lesson of the deepseek A/B, whose arms ran 13h41m apart and whose every pooled between-arm number is uninterpretable as a result. Record each arm's first and last `created_at`; the gap between arms goes in the report.

cerebras is `rpm: 5`, `tpm: 30000` — budget ~20 min per arm. Abort and report if: any generation is not `gpt-oss-120b`, any `gpt-5` call appears, or the live store's fingerprint changes.

- [ ] **Step 7: Confirm the live store is untouched**

```bash
./.venv/Scripts/python.exe -c "import os;s=os.stat('data/build/state/law_v1.sqlite3');print(s.st_size, int(s.st_mtime))" > data/build/exp_gptoss_ctl/out/live_stat_after.txt
diff data/build/exp_gptoss_ctl/out/live_stat_before.txt data/build/exp_gptoss_ctl/out/live_stat_after.txt && echo UNTOUCHED
```

- [ ] **Step 8: Measure**

Write a scratch measurement script under `data/build/exp_gptoss_new/out/` (not committed), modelled on `data/build/exp_prompt_v5/out/report_ab.py`. Open both arm stores read-only and print a markdown table giving, for each pre-registered measurement, the control value, the treatment value, the pass line, and PASS/FAIL:

| # | measurement | pass line |
|---|---|---|
| 1 | `think < think_min` breach rate | treatment <= control + 5pp |
| 2 | median trace words | treatment >= 400 absolute |
| 3 | `length_band` pass rate | treatment >= control − 5pp |
| 4 | `self_verification` pass rate | treatment >= control − 5pp |
| 5 | all generations `gpt-oss-120b`; `$0` ledgered | hard |

Report n for both arms. Report the inter-arm time gap. Do not report a rate without its n.

- [ ] **Step 9: Write the report**

`docs/reports/2026-08-27-gptoss-floor-under-the-prompt-ceiling.md`. Every pass line beside its measured value.

Then the section that matters: **what this means for the prompt edit.** If measurement 1 fails, the edit harms the generator that is now lead, and the choice is to revert the edit or reconsider the demotion — say so plainly and do not soften it. If it passes, say that the edit is now measured harmless on gpt-oss and still measured inert on deepseek, so it survives on the narrower ground of removing a real internal contradiction rather than on any demonstrated benefit.

Do not claim a benefit this branch has never measured.

- [ ] **Step 10: Commit**

```bash
git add src/tuned/data/paths.py tests/test_build_config.py \
        configs/data_law_v1_exp_gptoss_ctl.yaml configs/data_law_v1_exp_gptoss_new.yaml \
        docs/reports/2026-08-27-gptoss-floor-under-the-prompt-ceiling.md
git commit -m "measure gpt-oss against the ceiling it never asked for"
```
