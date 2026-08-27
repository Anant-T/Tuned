# Role-Aware bai Hook, and deepseek as Judge Slot B — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `bai/deepseek-v4-flash` serve as a judge without inheriting the generator's reply-budget raise, then wire it into judge slot B — which is empty today because `cerebras/gemma-4-31b` returns HTTP 402.

**Architecture:** Task 1 threads the call's role through the request-hook protocol, a fully internal `Callable` with four implementations in one file, and makes the bai hook leave judge budgets alone. Task 2 wires deepseek into the judge and tiebreak pools and proves with real calls that it returns verdicts rather than empty replies.

**Tech Stack:** Python 3.12, pytest, httpx, `tuned.data` provider layer, b.ai `deepseek-v4-flash`.

**Why now:** verified by direct probe on 2026-08-27 — `cerebras/gpt-oss-120b` (generator ref 1) and `cerebras/gemma-4-31b` (judge slot B) both return HTTP 402 `payment_required`. Generation fails over to deepseek; slot B has no free candidate, and its next-in-line is the paid `openai/gpt-5-mini`, which `family_separation` does **not** exclude on a deepseek row. That path is currently held shut only by the `usd_cap: 0.0` fence added earlier today, so rows park instead of being judged. deepseek was qualified on 2026-08-25 precisely to fill this gap as a new family.

## Global Constraints

- **No attribution or watermarks of any kind** in commits, code, comments, or docs. Absolute.
- **The live control store `data/build/state/law_v1.sqlite3` is never opened for write.** Read-only `file:...?mode=ro` URIs only. Expect size `554532864`, mtime `1787309490` unchanged.
- **Do not remove or weaken the `usd_cap: 0.0` fence** on either `openai/gpt-5-*` model, and do not remove their `usd_per_1m_*` keys. A bare cap blocks nothing — `_usd_per_1m` returns `0.0` for a missing price, so the check computes `0 + 0 > 0.0` = False. That fence is load-bearing today.
- **Do not change `length_band`.** `think_max` is 3000 and `total_max` is 8192; the latter is the student's training context.
- **Do not touch `src/tuned/data/prompts/` or `prompts_harmony/`.**
- Config files LF-only; never `Path.write_text` on them.
- Tests: `./.venv/Scripts/python.exe -m pytest`. Suite is green at 3575 passed / 19 skipped / 0 failed (count includes two uncommitted workdir tests from a blocked task).

---

### Task 1: Thread the call's role into the request-hook protocol

**Context:** `_bai_request_hook` raises `max_tokens` to the model's `max_output` (16384) because a reasoning model emits reasoning first and bills it against the same allowance — a generator budget that looks generous gets consumed by thinking before the answer starts. That is right for generation and wrong for judging: a judge is called with `JUDGE_MAX_TOKENS = 1024` (`judge.py:216`, from `providers.DEFAULT_JUDGE_REPLY_TOKENS`), and raising it 16× is the failure behind commit `4bcf014`, "a judge that spends its reply budget thinking is not a judge".

The hook cannot currently tell: its signature is `(payload, model)`. The role is already in scope at both call sites — `build_payload(self, req)` reads `req.role` at `providers.py:930` — it simply is not passed.

**Files:**
- Modify: `src/tuned/data/providers.py` — the `Quirk` type, four `_*_request_hook` functions, `chained_request_hook`, and the two call sites in `build_payload`
- Modify: `tests/test_build_providers.py` — `test_resolve_quirks_composes_request_hooks_in_order` (the only direct caller), plus new tests

**Interfaces:**
- Consumes: nothing.
- Produces: `Quirk.request_hook: Callable[[dict, ModelCfg, str | None], dict]` — every hook receives the role of the call, or `None` when a caller has none.

- [ ] **Step 1: Write the failing tests first**

Two behaviours, both in `tests/test_build_providers.py`:

```python
def test_the_bai_hook_leaves_a_judge_reply_budget_alone():
    model = _model(limits={"max_output": 16384})
    payload = {"max_tokens": providers.DEFAULT_JUDGE_REPLY_TOKENS}
    for role in ("judge", "tiebreak"):
        out = providers._bai_request_hook(dict(payload), model, role)
        assert out["max_tokens"] == providers.DEFAULT_JUDGE_REPLY_TOKENS, role


def test_the_bai_hook_still_raises_a_generator_budget():
    model = _model(limits={"max_output": 16384})
    out = providers._bai_request_hook({"max_tokens": 4000}, model, "generator")
    assert out["max_tokens"] == 16384
```

Match the file's existing `_model(...)` helper signature; read it rather than assuming.

- [ ] **Step 2: Run them and watch them fail**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_build_providers.py -q -k "bai_hook"
```
Expected: `TypeError` — the hook takes 2 positional arguments.

- [ ] **Step 3: Widen the protocol**

Change `Quirk.request_hook` to `Callable[[dict, ModelCfg, str | None], dict]`. Add a third parameter `role: str | None` to all four implementations — `_default_request_hook`, `_cerebras_request_hook`, `_openai_request_hook`, `_bai_request_hook` — and to `chained_request_hook`, which must pass it through to each composed quirk. Update both call sites in `build_payload` to pass `req.role`.

Three of the four hooks ignore the role. Make that explicit in each signature rather than silently accepting an unused argument, and say in one line why the parameter exists at all.

- [ ] **Step 4: Make the bai hook role-aware**

In `_bai_request_hook`, return the payload untouched when the role is a judging role. Do not hardcode a bare string pair if the module already names these roles — look for an existing constant and use it; introduce one only if none exists.

Extend the docstring: the raise exists because reasoning is billed against `max_tokens` and emitted first, which is a *generation* problem; a judge is called with a deliberately small reply budget and raising it reproduces `4bcf014`. Name that commit.

- [ ] **Step 5: Run the new tests, then the whole suite**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_build_providers.py -q -k "bai_hook"
./.venv/Scripts/python.exe -m pytest tests/ -q 2>&1 | tail -5
```
Expected: the two new tests pass; 0 failed overall. Fix `test_resolve_quirks_composes_request_hooks_in_order` (`tests/test_build_providers.py:410`) — it calls `composed.request_hook(...)` directly and needs the new argument.

- [ ] **Step 6: Commit**

```bash
git add src/tuned/data/providers.py tests/test_build_providers.py
git commit -m "a hook that cannot see the role cannot tell a judge from a generator"
```

---

### Task 2: Wire deepseek into judge slot B, and prove it returns verdicts

**Context:** With Task 1 done, a bai judge call keeps its 1024-token budget. That is necessary but not sufficient: `thinking: {"type": "disabled"}` was measured on 2026-08-25 as **~90% effective** — one call in ten still returned 165 reasoning tokens. Against a 1024-token budget that should still leave room for a verdict, but "should" is not a basis for putting a model in a production judge seat. This task ends with real calls.

**Files:**
- Modify: `configs/data_law_v1.yaml` — deepseek `roles`, `role_params`, and the two routing lists
- Modify: `tests/test_build_providers.py` or `tests/test_build_judge.py` — a test that a judge-role bai payload carries thinking-disabled and an unraised budget
- Create: `docs/reports/2026-08-27-deepseek-as-judge-slot-b.md`

**Interfaces:**
- Consumes: Task 1's role-aware hook.
- Produces: a judge pool with a free slot B.

- [ ] **Step 1: Give deepseek the roles and the judging params**

In the `bai` provider's `deepseek-v4-flash` block, set `roles: [generator, judge, tiebreak]` and add:

```yaml
        role_params:
          judge: {thinking: {type: disabled}}
          tiebreak: {thinking: {type: disabled}}
```

`build_payload` merges `model.role_params[role]` (`providers.py:930`) so the key reaches the wire. Verify that by building a payload rather than assuming it.

Append `bai/deepseek-v4-flash` to `routing.judge` and `routing.tiebreak`, placed **before** the paid `openai/gpt-5-*` refs and after the free ones. Add a comment recording why: gemma is 402 as of 2026-08-27, and without a free slot B the next candidate is a paid model that `family_separation` does not exclude on a deepseek row.

- [ ] **Step 2: Assert the payload, offline**

Add a test that a judge-role payload for `bai/deepseek-v4-flash` carries `thinking == {"type": "disabled"}` and `max_tokens == DEFAULT_JUDGE_REPLY_TOKENS`. This is the offline half of the guarantee.

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q 2>&1 | tail -5
```

- [ ] **Step 3: Prove it with real calls — this is the point of the task**

Write a scratch probe (not committed) under `data/build/exp_deepseek/out/`. It must use the SHIPPED config and the real provider layer — not a hand-built payload — and issue **at least 10** judge-role calls against a realistic judge prompt. Reuse a stored generation from `data/build/exp_deepseek/state/law_v1.sqlite3` (read-only) as the candidate so the prompt is the real size, around 8-9k tokens.

For every call record: HTTP status, `finish_reason`, `reasoning_tokens` if reported, whether `content` was non-empty, and whether the reply parsed as a verdict with all three axes.

Pass line, fixed here before the run: **at least 9 of 10 calls return a parseable three-axis verdict.** Below that, deepseek does not go in the judge pool — report it and revert Step 1's routing change rather than shipping a judge that returns empty replies.

Watch the rate limit: b.ai's is a request-counted bucket of 10 per minute and rejected calls consume slots, so pace the probe rather than firing all ten at once.

- [ ] **Step 4: Write the report**

`docs/reports/2026-08-27-deepseek-as-judge-slot-b.md`. Record: the fleet status that motivated this (cerebras 402 on both models, verified by direct probe), the pass line and the measured result with its n, the observed `reasoning_tokens` distribution against the ~90%-effective claim, and what remains unmeasured — in particular that this measures whether deepseek *answers* as a judge, not whether it judges *well*. Do not claim judging quality; no calibration was run.

If the pass line failed, say so plainly and record that the routing change was reverted.

- [ ] **Step 5: Commit**

```bash
git add configs/data_law_v1.yaml tests/ docs/reports/2026-08-27-deepseek-as-judge-slot-b.md
git commit -m "deepseek takes judge slot B, with the reply budget it was given"
```
