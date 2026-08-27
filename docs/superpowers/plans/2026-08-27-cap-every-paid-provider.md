# Cap Every Paid Provider, and Give a deepseek Row a Second Judge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close a live, uncappable spend path on the generator, and make a deepseek-generated row completable.

**Architecture:** Task 1 generalises the spend fence so a `usd_cap` works on any provider that declares one — today it is hardcoded to the provider literally named `openai` — and removes the paid `lightning` generator from routing until that holds. Task 2 adds the one alive, free, non-excluded model to the judge pool so deepseek rows stop parking, and corrects three config comments a final review found stating things that are no longer true.

**Tech Stack:** Python 3.12, pytest, `tuned.data` provider/generate layer.

**Why now — both verified today.** `generate.py:551` reads `if provider == "openai"`, and `_openai_usd_cap` (`generate.py:479-491`) scans only that provider by name. Declaring `usd_cap: 0.0` plus prices on `lightning` was tested and changed nothing. `lightning/lightning-ai/gpt-oss-120b` is generator ref 3, paid, `tpd: 50000000`, uncapped. Generator ref 1 (`cerebras/gpt-oss-120b`) returns HTTP 402 and ref 2 (`bai/deepseek-v4-flash`) is four consecutive failures from tripping its breaker, after which lightning serves every generation. Separately, on a deepseek row `family_separation` reduces judge slot B to the single ref `cerebras/gemma-4-31b`, which is also 402 — so no deepseek row can complete.

## Global Constraints

- **No attribution or watermarks of any kind** in commits, code, comments, or docs. Absolute.
- **The live control store `data/build/state/law_v1.sqlite3` is never opened for write.** Read-only `file:...?mode=ro` URIs only. Expect size `554532864`, mtime `1787309490` unchanged.
- **Do not weaken the existing OpenAI fence.** Both `openai/gpt-5-*` models must keep `usd_cap: 0.0` and BOTH `usd_per_1m_*` keys. A bare cap blocks nothing: `_usd_per_1m` returns `0.0` for a missing price, so the check computes `0 + 0 > 0.0` = False.
- **Do not change `build.length_band`** (`think_max` 3000, `total_max` 8192). Do not touch `src/tuned/data/prompts/` or `prompts_harmony/`.
- Config files LF-only; never `Path.write_text` on them.
- Tests: `./.venv/Scripts/python.exe -m pytest`. Green at 3579 passed / 19 skipped / 0 failed.
- Two files were already dirty before this plan and are not yours: `docs/superpowers/plans/2026-08-24-judge-calibration-and-yield.md` and `configs/data_law_v1_exp_measure.yaml`. Use explicit paths in `git add`, never `-A`.

---

### Task 1: A `usd_cap` must work on whichever provider declares it

**Files:**
- Modify: `src/tuned/data/generate.py` — `_openai_usd_cap` and the `provider == "openai"` branch around `:551`
- Modify: `configs/data_law_v1.yaml` — remove `lightning` from `routing.generator`
- Modify: `tests/test_build_generate.py` (or wherever the existing openai-fence tests live — find them first)

**Interfaces:**
- Consumes: nothing.
- Produces: a `budget_ok` that consults a cap declared on any provider, so Task 2 and any future paid ref inherit the protection.

- [ ] **Step 1: Write the failing test first**

A cap declared on a NON-openai provider must block. Model it on the existing openai-fence test — find that test and match its construction rather than inventing a fixture:

```python
def test_a_usd_cap_blocks_on_whichever_provider_declares_it(tmp_path):
    # A paid provider that is NOT named "openai", carrying a zero cap and prices.
    # Before this change the fence read only the provider literally named
    # "openai", so this call was allowed through at full price.
    ...
    assert budget_ok(store, cfg, "lightning", "lightning-ai/gpt-oss-120b", est_tokens=100_000) is False
```

Fill in the fixture the way the existing test does. Keep the existing openai test untouched — it must still pass afterwards, unchanged.

- [ ] **Step 2: Run it and watch it fail for the right reason**

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q -k "usd_cap_blocks_on_whichever"
```
Expected: FAIL — the call is currently allowed. Quote the failure in your report. A test that fails for a fixture error proves nothing.

- [ ] **Step 3: Generalise the cap lookup**

Rename `_openai_usd_cap` to something that says what it now does, and give it the provider name to look up rather than hardcoding `"openai"`. Preserve its current precedence exactly: a `usd_cap` attribute on the provider wins, otherwise the first model whose `limits` carries `usd_cap`. Returning `None` still means uncapped.

Then change the `if provider == "openai"` branch to apply whenever that provider declares a cap. Keep the arithmetic as it is — prompt tokens at `usd_per_1m_prompt`, completion at `usd_per_1m_completion` — and keep the existing behaviour that a missing price reads as `0.0`, since the OpenAI fence's correctness depends on prices being present rather than on the lookup failing loudly.

Record in the docstring why the old form was wrong: a cap declared on any other provider was silently ignored, so the config could express a fence that did not exist.

- [ ] **Step 4: Both tests pass**

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q -k "usd_cap or budget_ok or fence"
./.venv/Scripts/python.exe -m pytest tests/ -q 2>&1 | tail -5
```
Expected: the new test passes, the existing openai test still passes unchanged, 0 failed overall.

- [ ] **Step 5: Remove lightning from the generator routing**

```yaml
  generator: [cerebras/gpt-oss-120b, bai/deepseek-v4-flash]
```

Leave the `lightning` provider block itself in place — it is still a valid ref that a future config may route to, and deleting it would lose its measured limits. Add a comment recording: lightning is PAID and carries no cap; it was removed from routing on 2026-08-27 after a review found the fence could not reach it; and re-adding it requires a `usd_cap` with prices, which now functions after Task 1's change.

Note the consequence honestly in the comment: with ref 1 at 402, generation now depends on `bai/deepseek-v4-flash` alone, and if its breaker trips there is no third ref — rows park rather than failing over to a paid model. That is the intended trade.

- [ ] **Step 6: Prove the fence now covers what it claims**

Against the SHIPPED config, call the real `budget_ok`/`budget_ok_for` for `openai/gpt-5-mini`, `openai/gpt-5-nano`, and `lightning/lightning-ai/gpt-oss-120b` at a realistic `est_tokens`. Report each result. The two openai refs must be blocked. Lightning has no cap declared, so it will NOT be blocked — that is why Step 5 removes it from routing; say so rather than implying the cap covers it.

Then add a cap to lightning temporarily in memory and confirm it now blocks, proving the generalisation works. Do not commit that cap.

- [ ] **Step 7: Commit**

```bash
git add src/tuned/data/generate.py configs/data_law_v1.yaml tests/
git commit -m "a cap declared on any provider is a cap, not decoration"
```

---

### Task 2: Give a deepseek row a second judge, and correct three stale comments

**Context:** On a deepseek generation, `family_separation` excludes the deepseek judge, and slot B's remaining candidates are `cerebras/gemma-4-31b` (HTTP 402) and the fenced `openai/gpt-5-*`. So slot B is unfillable and the row parks in `judge_error` after 8 attempts. `groq/openai/gpt-oss-20b` is alive, free, and family `gpt-oss` — not excluded on a deepseek row. It already sits in `routing.tiebreak` and `routing.probe`.

**Files:**
- Modify: `configs/data_law_v1.yaml` — `routing.judge`, plus three comment corrections
- Modify: tests asserting judge-pool shape (expect several; widen honestly, do not repin loosely)

**Interfaces:**
- Consumes: Task 1's generalised fence.
- Produces: a judge pool that can fill both slots for every generator family currently routable.

- [ ] **Step 1: Add the ref**

Place `groq/openai/gpt-oss-20b` in `routing.judge` after `groq/qwen/qwen3.6-27b` and before the paid refs. Record in a comment: it is added because on a deepseek row slot B was otherwise a single 402'd ref; that this model scored **0/10 on IPC→BNS ground truth**, which is why `routing.tiebreak` deliberately places mistral ahead of it to keep it out of the deciding seat; and that as judge B it is one of two opinions with a tiebreak above it, not the decider.

State the open risk plainly in the comment: its BNS weakness was measured on ungrounded recall, while judging is grounded in supplied source, so the weakness may not transfer — **that is a hypothesis and no calibration has been run.**

- [ ] **Step 2: Confirm every generator family can now fill both judge slots**

```bash
./.venv/Scripts/python.exe -c "
from tuned.data.config import load_build_config
cfg = load_build_config('configs/data_law_v1.yaml', allow_unpinned=True)
fam = {f'{p.name}/{m.id}': m.family for p in cfg.providers for m in p.models}
for gen in cfg.routing.generator:
    g = fam[gen]
    pool = [r for r in cfg.routing.judge if fam[r] != g]
    print(f'{gen:34s} family={g:9s} judge pool={pool}')
"
```
Report the output. Every generator ref must leave at least two judge refs of differing families, and say which of those are alive given cerebras is 402.

- [ ] **Step 3: Re-baseline the judge-pool tests**

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q 2>&1 | tail -30
```

For each failure, decide and state in your report whether the assertion pinned a literal where it meant an invariant. Prefer asserting the invariant. Do not weaken a pool-shape or family-separation assertion to accommodate the new ref — if one cannot pass without changing production behaviour, stop and report.

- [ ] **Step 4: Correct three stale comments a final review found**

All in `configs/data_law_v1.yaml`, all stating things that are no longer true. Mark corrections in place as SUPERSEDED with the date, this branch's convention, rather than deleting:

1. Around `:837` — "the ~$1/day cap". No such cap exists; it is `usd_cap: 0.0`.
2. Around `:874` — "Slot B is now openai/gpt-5-mini". False twice over: deepseek was placed ahead of it, and the fence makes the openai refs unreachable.
3. Around `:29-38` — the `think_max` revert is justified on gpt-oss being "the lead generator now". It generates nothing while ref 1 returns 402. Add the caveat; do not re-raise `think_max`.

- [ ] **Step 5: Record what preflight cannot see**

`generate.preflight_messages(cfg, ["generator","judge","tiebreak"])` returns `([], [])` on the shipped config because it builds a budget-blind `Router`, so fenced refs read as filling a slot they cannot fill. Add a comment near the routing block recording this, so a clean preflight is not read as a healthy pool. Do not change preflight's behaviour in this task — that is a separate change.

- [ ] **Step 6: Verify and commit**

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q 2>&1 | tail -5
./.venv/Scripts/python.exe -c "import os;s=os.stat('data/build/state/law_v1.sqlite3');print(s.st_size, int(s.st_mtime))"
```
Expected: 0 failed; `554532864 1787309490`.

```bash
git add configs/data_law_v1.yaml tests/
git commit -m "a deepseek row gets a second judge family, and three comments stop lying"
```
