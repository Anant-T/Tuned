# Generator Prompt Length Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop deepseek-v4-flash overrunning the reasoning band by removing the permission to run long from all 14 generator prompts, prove the fix on a paired A/B against banked control data, and re-baseline the 42 stale tests that block the merge.

**Architecture:** Three independent deliverables. Task 1 repairs a test fixture whose anchor no longer matches the live config and commits the b.ai integration — this restores a green suite so later verification means something. Task 2 edits the prompt templates in place and re-pins their shas. Task 3 runs a generate-only A/B in a fresh isolated workdir and reports measured values against pre-registered pass lines.

**Tech Stack:** Python 3.12, pytest, SQLite, PyYAML, `tuned.data` build pipeline, b.ai `deepseek-v4-flash` over an OpenAI-compatible endpoint.

**Spec:** `docs/superpowers/specs/2026-08-27-generator-prompt-length-fix-design.md`

## Global Constraints

- **No attribution or watermarks of any kind** in commits, code, comments, docs, or reports. No AI co-author trailers, no "generated with" footers. This is absolute.
- **The live control store `data/build/state/law_v1.sqlite3` is never opened for write.** Read-only `ATTACH` via `file:...?mode=ro` only. Its size and mtime must be identical after every task.
- **All prompt template files are LF-only.** `test_templates_are_lf_only` asserts it and the sha pin depends on it. Never use `Path.write_text` for these on Windows — it emits CRLF and silently changes every hash. Write LF bytes explicitly.
- **Never re-pin a sha by copying it from a test failure message.** Read it from `prompt_registry.load(prompt_id).sha` after the edit.
- **Do not create new prompt variant files.** `pick_variant` hashes `seed_id` modulo the variant count; adding a file re-maps every seed assignment corpus-wide.
- **`usd_cap: 0.0` alone blocks nothing.** `generate._usd_per_1m` returns `0.0` for a missing price, so `0 + 0 > 0.0` is False. Any zero cap must ship with `usd_per_1m_prompt` and `usd_per_1m_completion` beside it.
- **Never guess a model repo revision sha.** Use `scripts/pin_revision.py`.
- Run tests with `./.venv/Scripts/python.exe -m pytest`.

---

### Task 1: Re-baseline the b.ai routing expectations and commit the integration

**Context:** The worktree carries an uncommitted b.ai integration (`configs/data_law_v1.yaml`, `src/tuned/data/providers.py`, plus new tests in `tests/test_build_providers.py`). It is complete and well-documented work. It leaves 42 tests failing, and those failures block the merge. They are stale expectations, not broken code — confirmed pre-existing by stashing and diffing the failure sets before this branch's first commit.

**Root cause, already diagnosed — do not re-investigate:** `tests/pipeline_fakes.py:798-803` builds every test config by string-replacing the live config, guarded by `assert redirected.count(old) == 1, old`. One anchor is `"  generator: [cerebras/gpt-oss-120b]"`. The live config's routing block is now:

```yaml
  generator: [bai/deepseek-v4-flash, cerebras/gpt-oss-120b,
              lightning/lightning-ai/gpt-oss-120b]
```

so that anchor matches **0** times, the assert fires, the fixture aborts, and every test built on it fails. Fix the fixture first and most of the 42 resolve together. Only then look at the residue.

**Files:**
- Modify: `tests/pipeline_fakes.py` (the anchor block around line 798)
- Modify: `tests/test_build_providers.py`, `tests/test_build_generate.py`, `tests/test_build_judge.py`, `tests/test_build_eval_matched.py` — only where a literal expectation genuinely needs updating
- Commit (already-written, uncommitted): `configs/data_law_v1.yaml`, `src/tuned/data/providers.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: a green test suite, which Tasks 2 and 3 rely on to tell their own regressions apart from inherited noise.

- [ ] **Step 1: Record the exact failure baseline**

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q 2>&1 | tail -5
```
Expected: `42 failed, 3512 passed, 19 skipped`. If the count differs, stop and report — the baseline moved and this plan's premise needs rechecking.

- [ ] **Step 2: Repair the fixture anchor**

The three replacements in `pipeline_fakes.py` inject a second generator and a third judge into a copy of the live config. Update the `generator` anchor so it matches the live config's current multi-line routing block, and keep the `SECOND_GENERATOR_REF` injection doing what it did before: adding one more generator ref to the list. Preserve the `assert redirected.count(old) == 1` guard on every replacement — it is what turned this into a loud failure instead of a silent one, and it earned its keep.

Note the `judge` anchor spans two lines in the live config and the `generator` anchor now does too. Match what is actually in the file.

- [ ] **Step 3: Re-run and triage the residue**

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q 2>&1 | tail -30
```
Expect the count to drop sharply. For each test still failing, decide one of two things and say which in the report:
- the assertion encodes a real invariant that the new routing genuinely changed → update the expected value
- the assertion pinned a literal where it meant an invariant → rewrite it to assert the invariant

Prefer the second. The validation wave already produced one Critical from a test pinned to a config that was not committed. Where a test cares that a free provider precedes a paid one, assert that ordering — not the provider's name.

- [ ] **Step 4: Verify green**

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q 2>&1 | tail -5
```
Expected: 0 failed. If any test cannot be made to pass without changing production behaviour, STOP and report it rather than weakening the assertion.

- [ ] **Step 5: Commit**

```bash
git add configs/data_law_v1.yaml src/tuned/data/providers.py tests/pipeline_fakes.py tests/test_build_providers.py tests/test_build_generate.py tests/test_build_judge.py tests/test_build_eval_matched.py
git commit -m "route the generator through b.ai and re-baseline the fixture anchors"
```

---

### Task 2: Remove the permission to run long from all 14 generator prompts

**Context:** Each generator template caps the reasoning at 450–700 words and then, in the same breath, grants permission to ignore the cap. gpt-oss obeyed the cap; deepseek obeys the permission and writes a median 1,727 words. Delete the permission, keep the band, add a ceiling.

**Files:**
- Modify: all 14 of `src/tuned/data/prompts/gen_*.md`
- Modify: `tests/test_build_prompts.py` — `EXPECTED_SHAS` (14 `gen_*` entries) and `REASONING_FLOOR_CLAUSE`
- Modify: `src/tuned/data/gates.py` — one entry in `INSTRUCTION_ECHO_SPANS`
- **Do not touch** `src/tuned/data/prompts_harmony/**`. Leaving it is deliberate; the spec's "Out of scope" section says why.

**Interfaces:**
- Consumes: a green suite from Task 1.
- Produces: edited templates that Task 3 measures. No API surface changes.

- [ ] **Step 1: Record the baseline**

```bash
./.venv/Scripts/python.exe -c "
from tuned.data.prompt_registry import all_ids, load
import re
for i in all_ids():
    if not i.startswith('gen_'): continue
    u = load(i).user
    print(f'{i:26s} sha={load(i).sha}  words={len(re.findall(chr(92)+chr(119)+chr(43), u))}')
"
```
Keep this output. Step 6 compares against it.

- [ ] **Step 2: Edit the 14 templates**

In each file, two changes on one sentence-pair, and nothing else:

**(a) Delete the permissive clause.** It has at least four surface forms — **do not find-and-replace on `deserves`, which appears in only 5 of the 14 files.** Read each file. The clause sits between a semicolon and `and is never a retelling`:

| file | line | the clause to remove |
|---|---|---|
| `gen_drafting_v1` | 21 | the thinking beforehand runs as long as the matter needs |
| `gen_drafting_v2` | 21 | your thinking beforehand takes as long as it needs |
| `gen_irac_analysis_v1` | 21 | the reasoning takes as long as it needs to take, |
| `gen_irac_analysis_v2` | 15 | the reasoning runs as long as it needs to |
| `gen_irac_analysis_v3` | 21 | the thinking that precedes it runs as long as the point deserves |
| `gen_irac_analysis_v4` | 15 | your own working beforehand runs as long as the problem deserves |
| `gen_statute_qa_v1` | 23 | the reasoning that precedes it takes as long as the point requires |
| `gen_statute_qa_v2` | 23 | the thinking before it takes as long as the question deserves |
| `gen_statute_qa_v3` | 23 | your thinking beforehand runs as long as the question deserves |
| `gen_statute_qa_v4` | 17 | the thinking before it takes as long as the point requires |
| `gen_summarization_v1` | 19 | your thinking runs as long as the case requires |
| `gen_summarization_v2` | 19 | the thinking beforehand takes as long as the decision deserves |
| `gen_transition_v1` | 33 | your reasoning takes as long as the point needs |
| `gen_transition_v2` | 33 | the thinking beforehand runs as long as the point requires |

The surrounding sentence must still read as English afterwards. `and is never a retelling of the materials` is doing real work and stays.

**(b) Replace the tail of the next sentence.** Every file ends the packet with:

> Work the point through fully — 450 to 700 words of deliberation is normal for a matter of any substance.

Keep `450 to 700 words of deliberation` **as a literal substring** — two test constants match on it. Replace only `is normal for a matter of any substance` with a ceiling. Worked example for `gen_irac_analysis_v4`:

> Work the point through fully — 450 to 700 words of deliberation, and 700 is a ceiling you do not cross.

**Write each file as LF bytes.** `test_templates_are_lf_only` asserts it and every sha depends on it. Do not use `Path.write_text` on these files.

**Keep the 14 files worded differently from one another.** They are deliberate paraphrases and `test_variants_are_real_paraphrases` guards their dissimilarity. Fourteen identical ceiling sentences defeats the mechanism even if the test's 0.8 threshold tolerates it.

- [ ] **Step 3: Update the two test constants**

In `tests/test_build_prompts.py`, `REASONING_FLOOR_CLAUSE` is currently:

```python
REASONING_FLOOR_CLAUSE = "450 to 700 words of deliberation is normal"
```

Its trailing ` is normal` no longer exists in any template. Trim the constant to the part that survives, and extend the comment above it — which explains the 2026-08-18 floor raise — with one sentence recording that a ceiling was added on 2026-08-27 and why. That comment is the only place a reader learns the band is now defended from both sides.

- [ ] **Step 4: Update the echo span in `gates.py`**

`INSTRUCTION_ECHO_SPANS` (around `gates.py:274`) carries the literal `"450 to 700 words of deliberation is normal"`. `check_prompt_echo` matches it against generated traces to catch a model parroting its instructions back. No test exercises this span, so a stale value fails silently — the gate simply stops detecting. Update it to the wording the templates now use.

- [ ] **Step 5: Re-pin the 14 shas**

Read the new values from the registry — **never copy a sha out of a pytest failure message**:

```bash
./.venv/Scripts/python.exe -c "from tuned.data.prompt_registry import all_ids, load; [print(f'    {i!r}: {load(i).sha!r},') for i in all_ids()]"
```

Update only the 14 `gen_*` entries in `EXPECTED_SHAS` in `tests/test_build_prompts.py`. The `judge_*` and `probe_*` entries do not change, and neither do any of the three overlay sha dicts — those hash `prompts_harmony/` bytes, which you have not touched.

- [ ] **Step 6: Check the word budget before running the suite**

Re-run Step 1's command and diff the word counts. `test_generator_user_block_is_the_intended_size` asserts `250 <= words <= 500`, and two files are nearly full:

- `gen_drafting_v1` — 495 words, **5** of headroom
- `gen_transition_v2` — 496 words, **4** of headroom

If either exceeds 500, shorten your ceiling wording in that file. Do not raise the test's bound — it was already raised once, 470 → 500, and raising it again to fit an edit that was supposed to *remove* words is treating the symptom.

- [ ] **Step 7: Run the tests**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_build_prompts.py tests/test_build_harmony.py -q 2>&1 | tail -20
./.venv/Scripts/python.exe -m pytest tests/ -q 2>&1 | tail -5
```
Expected: 0 failed on both. Task 1 left the suite green, so any failure here is yours.

- [ ] **Step 8: Commit**

```bash
git add src/tuned/data/prompts/ src/tuned/data/gates.py tests/test_build_prompts.py
git commit -m "the reasoning band gets a ceiling, not just a floor"
```

---

### Task 3: The paired A/B arm

**Context:** Task 2 changes the prompts. This task measures whether that changed anything, against control data already banked in `data/build/exp_deepseek/state/law_v1.sqlite3` (n=99 generations, v4 prompts, same generator, same seeds).

Precedent to copy closely: the `exp_deepseek` arm built on 2026-08-26 (`31a546d`, `ac5db21`, `1f6c0c0`). This task is the same shape with a different workdir.

**Files:**
- Modify: `src/tuned/data/paths.py` — add `exp_prompt_v5` to `ISOLATED_WORKDIR_SIBLINGS`
- Modify: `tests/test_build_config.py` — a test that `is_live_control_workdir("data/build/exp_prompt_v5")` is `False`
- Create: `configs/data_law_v1_exp_prompt_v5.yaml`
- Create: `docs/reports/2026-08-27-generator-prompt-length-fix.md`

**Interfaces:**
- Consumes: the edited templates from Task 2; `scripts/seed_exp_store.py` from `1f6c0c0` unchanged.
- Produces: the report that decides whether the live config changes.

- [ ] **Step 1: Declare the isolated workdir**

Add `exp_prompt_v5` beside `exp_deepseek` in `paths.ISOLATED_WORKDIR_SIBLINGS`, and add the matching test next to the existing `exp_deepseek` one. Without this every write guard in the tree treats the arm as the frozen live control.

```bash
./.venv/Scripts/python.exe -m pytest tests/test_build_config.py -q
```

- [ ] **Step 2: Create the arm config**

Copy `configs/data_law_v1_exp_prompt_v5.yaml` from `configs/data_law_v1_exp_deepseek.yaml`. Change exactly two things: the header comment (state that this is the v5-prompt treatment arm, and name the control) and `build.workdir: data/build/exp_prompt_v5`.

Everything else carries over untouched — in particular both cost fences: `routing.generator: [bai/deepseek-v4-flash]` alone, and on **both** gpt-5 models the full `usd_cap: 0.0` line with `usd_per_1m_prompt: 0.25, usd_per_1m_completion: 2.0` beside it.

Write the file as LF bytes. A previous task in this repo lost time to `Path.write_text` converting a whole config to CRLF and burying its four real edits in an 886-line diff.

- [ ] **Step 3: Record the live store's fingerprint**

```bash
mkdir -p data/build/exp_prompt_v5/out
./.venv/Scripts/python.exe -c "import os;s=os.stat('data/build/state/law_v1.sqlite3');print(s.st_size, int(s.st_mtime))" > data/build/exp_prompt_v5/out/live_stat_before.txt
cat data/build/exp_prompt_v5/out/live_stat_before.txt
```

- [ ] **Step 4: Seed the arm from the live store, read-only**

```bash
./.venv/Scripts/python.exe scripts/seed_exp_store.py --config configs/data_law_v1_exp_prompt_v5.yaml \
    --from data/build/state/law_v1.sqlite3 --per-source 200 --seed 3407
```
The `--seed 3407` and `--per-source 200` must match the control exactly or the A/B is not paired.

- [ ] **Step 5: Plan the same three arms**

```bash
./.venv/Scripts/python.exe -m tuned.data.tasks --config configs/data_law_v1_exp_prompt_v5.yaml \
    --stream synthesis --arm sc --n 13 --source s3://indian-supreme-court-judgments \
    --mix irac_analysis=0.55,summarization=0.45
```
Repeat for `--arm predex --n 14 --source L-NLProc/PredEx_Instruction-Tuning_Pred-Exp` and `--arm tathya --n 13 --source L-NLProc/TathyaNyaya-and-FactLegalLlama-NyayaFacts-Datasets`, same `--mix` on each.

Then assert the seed gate held, before spending anything:

```bash
./.venv/Scripts/python.exe -c "
import sqlite3
from tuned.data.config import load_build_config
from tuned.data.tasks import seed_token_budget
b = seed_token_budget(load_build_config('configs/data_law_v1_exp_prompt_v5.yaml', allow_unpinned=True))
c = sqlite3.connect('file:data/build/exp_prompt_v5/state/law_v1.sqlite3?mode=ro', uri=True)
over_store = c.execute('SELECT COUNT(*) FROM seed WHERE COALESCE(token_count,0) > ?', (b,)).fetchone()[0]
over_plan = c.execute('SELECT COUNT(*) FROM task t JOIN seed s ON s.seed_id=t.seed_id WHERE COALESCE(s.token_count,0) > ?', (b,)).fetchone()[0]
print('budget', b, 'oversize in store', over_store, 'planned over budget', over_plan)
assert over_plan == 0 and over_store >= 1
"
```

- [ ] **Step 6: Generate — no judging**

```bash
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tuned.data.generate \
    --config configs/data_law_v1_exp_prompt_v5.yaml --n-workers 4 --max-batches 30
```
Watch for: any `gpt-5` call (the $0 fence leaked — stop immediately), any generation whose model is not `deepseek-v4-flash`, and any `reasoning_effort` 400.

- [ ] **Step 7: Confirm the live store is untouched**

```bash
./.venv/Scripts/python.exe -c "import os;s=os.stat('data/build/state/law_v1.sqlite3');print(s.st_size, int(s.st_mtime))" > data/build/exp_prompt_v5/out/live_stat_after.txt
diff data/build/exp_prompt_v5/out/live_stat_before.txt data/build/exp_prompt_v5/out/live_stat_after.txt && echo "UNTOUCHED"
```
Any difference is a stop-everything event.

- [ ] **Step 8: Measure control against treatment**

Write a measurement script under `data/build/exp_prompt_v5/out/` (scratch, not committed) modelled on `data/build/exp_deepseek/out/report_wave.py`. It opens BOTH stores read-only and prints a markdown table with, for each pre-registered measurement, the control value, the treatment value, the pass line, and PASS/FAIL.

The seven pre-registered pass lines, from the spec — these were fixed before the run and do not move:

| # | measurement | control | pass line |
|---|---|---|---|
| 1 | median trace words | 1,727 | < 900 |
| 2 | `length_band` pass rate | 49% | > 70% |
| 3 | `irac_placement` pass rate | 38% | > 60% |
| 4 | `verbatim_overlap` pass rate | 54% | > 70% |
| 5 | `self_verification` pass rate | 87% | >= 80%, must not regress |
| 6 | answer well-formedness (`missing_in_answer` empty) | 100% | >= 95% |
| 7 | `think<think_min` breaches | 3/99 | <= 5% |

Also report, without a pass line: generations recorded, calls returning content, models seen, ledger totals, and per-arm counts.

- [ ] **Step 9: Write the report**

Create `docs/reports/2026-08-27-generator-prompt-length-fix.md`. Paste the measured table. State each pass line beside its measured value.

Then write the two prose sections:
- **What this changes in `configs/data_law_v1.yaml`** — if measurements 1-4 pass and 5 and 7 hold, nothing changes but the prompts, and deepseek stays lead generator. If 1 passes and 2-4 do not, say plainly that length was not the cause and the spec's diagnosis was wrong.
- **What is still unmeasured** — judging was deliberately skipped; accept rate under v5 prompts is unknown.

Do not describe a result the numbers do not support. If a number disappoints, report it.

- [ ] **Step 10: Commit**

```bash
git add src/tuned/data/paths.py tests/test_build_config.py configs/data_law_v1_exp_prompt_v5.yaml docs/reports/2026-08-27-generator-prompt-length-fix.md
git commit -m "measure the shortened prompts against the banked v4 control"
```
