# law_v1 recovery-arm unblock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the isolated `exp_recovery` arm to the point where it can generate and judge one bounded, matched cohort — by fixing the two judge-fleet defects that make any yield number meaningless, and by removing the cohort blocker that currently refuses the arm before it spends anything.

**Architecture:** Three config/code fixes land first (all free, all test-first): carry the already-spent OpenAI dollars into the recovery wallet, stop the gpt-5 judges spending their entire reply budget on hidden reasoning, and accept one axis alias the rubric emits. Then the matched evaluator learns to draw its cohort from a *declared* set of task-type strata instead of assuming all four, because the live control store cannot fill the `statute_qa` stratum and will not without provision text it does not have. Only then does one bounded probe run: generate on the free Cerebras tier, judge under the corrected fleet, evaluate against a pre-registered manifest.

**Tech Stack:** Python 3.12, SQLite (`data/build/exp_recovery/state/law_v1.sqlite3`), pytest, YAML config (`configs/data_law_v1_exp_recovery.yaml`), OpenAI-compatible HTTP providers (Cerebras, Groq, Mistral, OpenAI).

## Why this plan exists (measured 2026-08-23)

The prompt rewrite of 18 Aug did fix IRAC placement. Split the live store's 1,396 generations by whether the task's stored `prompt_sha` equals the current file SHA:

| | old prompts (257 gens) | corrected prompts (1,139 gens) |
|---|---|---|
| all-gates-pass | 0 (0%) | 167 (14.7%); latest-per-task 131/434 = 30.2% |
| `irac_placement` fails | 41% | 19% |
| `banned_meta` | 49% | 9% |
| `think_format` | 23% | 0% |
| `self_verification` | 66% | **66% — unmoved** |

The Harmony prefill arm moves the one gate the prompt rewrite could not: `exp_harmony` is 31/48 latest-per-task = 64.6% all-gates-pass with `self_verification` failing 10%. That is the treatment this plan is trying to measure. It cannot be measured yet, for the reasons Tasks 1–5 remove.

## Global Constraints

- The live control store `data/build/state/law_v1.sqlite3` is **READ-ONLY**. Never `Store.open()` it, never `--reopen` it, never migrate it, never relabel it. Use `eval_matched.open_eval_store` (which forces `mode=ro` + `PRAGMA query_only`) and nothing else.
- All work happens on branch `worktree-law-v1-data-pipeline` in `C:\Users\Anant\Desktop\projects\tuned\.claude\worktrees\law-v1-data-pipeline`. The training lane on `main` is untouched. No push, no merge, no training run.
- Never pass `--config configs/data_law_v1.yaml` to any command that writes. The recovery config is `configs/data_law_v1_exp_recovery.yaml`, whose `workdir` is `data/build/exp_recovery`; `load_build_config` already refuses that file if the workdir is pointed at `data/build`.
- **OpenAI spend: $2.00 TOTAL, hard, across `gpt-5-mini` + `gpt-5-nano`** (operator instruction, 2026-08-21). `budget_ledger` is per-store, so a fresh store resets the cap — Task 1 exists because of that. Measured spend already made: `exp_harmony`, `gpt-5-mini`, 124 requests, 377,537 prompt / 122,607 completion tokens = **$0.3396**. Remaining headroom: **$1.6604**.
- OpenAI and Gemini may **judge**. Closed-API **generations** never enter the training mix (spec line 14 / ToS). The teacher stays `cerebras/gpt-oss-120b` at `temperature 0.7` / `top_p 0.95`. Do not revive `EFFORT_LADDER_RETIRED`.
- Do not loosen the law gates: `citations`, `temporal`, `answer_key`, `statutory_grounding`. Only format/style knobs are in scope.
- `judge_threshold` stays empty. The 46 rows in `gold_label` are **Fable-5 model-generated references, not human gold** (operator attestation, 22 Aug). They may support model-agreement comparison only. Nothing here may fit thresholds, activate a fleet rule, or treat `evaluate()`'s `decision` as a promotion authority — Task 10 records it as advisory and says why.
- Every task ends green on its focused test file. The full suite is run once, at Task 5.
- Python is the worktree venv: `.venv/Scripts/python.exe`. All commands run from the worktree root.

## The blocker Tasks 4–5 remove

`configs/data_law_v1_exp_recovery.yaml` sets `require_pretreatment_manifest: true`, so `generate.main` refuses to create the workdir until `.superpowers/sdd/law-v1-recovery_893eff3d/cohort-manifest.json` exists and validates. That file does not exist, and today it cannot be written:

```text
select_cohort(control_store, n_per=20)
  -> blocked=True  reason='underfilled-stratum:statute_qa'
  -> pairs=60      irac_analysis 20, drafting 20, summarization 20, statute_qa 0
  -> excluded: stale 419, no_generation 126, non_gpt_oss 39, gold 41,
               ineligible_statute 103
```

`statute_section_eligible` requires a seed to carry a `meta_json.section_text` **distinct from** the seed body. Measured on the live control store: **270 `statute_qa` tasks, 0 eligible seeds.** No seed in that database carries provision text at all. Fixing that means real Gazette Act bodies attached to seeds — the `gazette.py` / `attach_statute_qa_section` work that sits **uncommitted** on branch `law-v1-foundation`, whose manifest deliberately holds identities and provenance only, no Act text. Attaching it would also mean writing to the control store, which is forbidden.

So the cohort cannot be four strata. Tasks 4–5 make the strata a **declared, validated, manifest-recorded** property of the experiment rather than silently shrinking `n_per` or silently dropping a stratum. The consequence is stated in the manifest and in the final report: **60 pairs, not 80** — McNemar runs on a smaller discordant pool, and no claim about `statute_qa` may be made from this arm at all.

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `configs/data_law_v1_exp_recovery.yaml` | Modify | The arm's whole contract: wallet headroom, gpt-5 judge sampling, declared cohort strata |
| `src/tuned/data/judge.py` | Modify (~line 234) | Axis alias table — one alias the rubric emits and the parser rejects |
| `src/tuned/data/config.py` | Modify (`BuildCfg` ~line 56, loader ~line 878) | New validated `build.eval_cohort_strata` surface |
| `src/tuned/data/eval_matched.py` | Modify (`select_cohort`, `cohort_manifest`, `validate_pretreatment_manifest`, `write_manifest`, `evaluate`, `require_pretreatment_manifest`, `main`) | Cohort selection and manifest honour the declared strata |
| `tests/test_build_config.py` | Modify | Wallet, gpt-5 role_params, strata validation |
| `tests/test_build_judge.py` | Modify | Alias parse |
| `tests/test_build_eval_matched.py` | Modify | Declared-strata selection, manifest round-trip, default still blocks |
| `.superpowers/sdd/law-v1-recovery_893eff3d/cohort-manifest.json` | Create (Task 6) | The pre-registered 60-pair cohort |
| `docs/reports/2026-08-23-recovery-arm-probe.md` | Create (Task 10) | What the probe measured and what it may not claim |

---

### Task 1: Carry the spent OpenAI dollars into the recovery wallet

`generate._openai_usd_spent(store, cfg)` sums `budget_ledger` **from the store it is given**. `exp_recovery` will be a brand-new SQLite file with an empty ledger, so a `usd_cap: 2.0` there authorises a second, full $2 on top of the $0.3396 already spent in `exp_harmony`. The operator's instruction was $2 **total**, hard. The cap in this yaml must therefore be the remaining headroom.

**Files:**
- Modify: `configs/data_law_v1_exp_recovery.yaml` (the two `openai` model blocks, ~lines 589–601)
- Test: `tests/test_build_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `configs/data_law_v1_exp_recovery.yaml` declares `usd_cap: 1.66` on both `openai/gpt-5-mini` and `openai/gpt-5-nano`. Later tasks rely on that ceiling being the arm's real spend brake.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_config.py`:

```python
def test_recovery_openai_cap_is_the_remaining_headroom_not_a_fresh_two_dollars():
    """budget_ledger is per-store, so a new store resets usd_cap.

    exp_harmony already spent $0.3396 of the operator's $2.00 total
    (gpt-5-mini, 124 requests, 377,537 prompt / 122,607 completion tokens).
    The recovery yaml must declare the REMAINDER, or the arm silently
    authorises a second full wallet.
    """
    cfg = load_build_config(
        "configs/data_law_v1_exp_recovery.yaml", allow_unpinned=True
    )
    for name in ("gpt-5-mini", "gpt-5-nano"):
        _provider, model = cfg.model_for(ModelRef("openai", name))
        assert model.limits["usd_cap"] == 1.66, (
            f"openai/{name} declares usd_cap {model.limits['usd_cap']!r}; the "
            "$2.00 operator total already has $0.3396 spent in exp_harmony"
        )
```

`BuildConfig.model_for` takes a `ModelRef`, not a `"provider/model"` string — it reads `ref.provider` and `ref.model`, so a string raises `AttributeError`. Add to the module's imports if not already present:

```python
from tuned.data.config import ModelRef, load_build_config
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/test_build_config.py::test_recovery_openai_cap_is_the_remaining_headroom_not_a_fresh_two_dollars -v
```

Expected: FAIL — `AssertionError: openai/gpt-5-mini declares usd_cap 2.0`

- [ ] **Step 3: Edit the config**

In `configs/data_law_v1_exp_recovery.yaml`, change `usd_cap: 2.0` to `usd_cap: 1.66` in **both** `openai` model blocks, and extend the operator comment above them with:

```yaml
      # THE NUMBER HERE IS 1.66, NOT 2.00, AND THAT IS NOT A TYPO. usd_cap is
      # enforced by generate._openai_usd_spent, which sums budget_ledger from
      # THE STORE IT IS GIVEN - and budget_ledger is per-workdir. A brand-new
      # exp_recovery store starts at zero however much has been spent
      # elsewhere, so declaring 2.00 here would authorise a SECOND full wallet.
      # Measured 2026-08-23 on data/build/exp_harmony/state/law_v1.sqlite3:
      # gpt-5-mini, 124 requests, 377,537 prompt + 122,607 completion tokens
      # = $0.3396 at the prices below. 2.00 - 0.3396 = 1.6604, rounded DOWN.
      # Any future store recomputes the remainder the same way; the arithmetic
      # belongs to the operator's total, not to this file.
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/Scripts/python.exe -m pytest tests/test_build_config.py -v
```

Expected: PASS, whole file green.

- [ ] **Step 5: Commit**

```bash
git add configs/data_law_v1_exp_recovery.yaml tests/test_build_config.py
git commit -m "the recovery wallet inherits what exp_harmony already spent"
```

---

### Task 2: Stop the gpt-5 judges spending their whole reply budget on reasoning

Measured on `exp_harmony`: 124 `gpt-5-mini` judge calls returned **27 usable judgements**. 95 of the store's 96 `judge_parse_error` events are empty replies (`no object found: ''`). Mean completion was 989 tokens against a 1,024 budget. Cause: judge calls send `max_tokens=1024` (`providers.DEFAULT_JUDGE_REPLY_TOKENS`), the `openai` quirk renames it `max_completion_tokens`, and the gpt-5 family bills *reasoning* tokens against that same budget — while the yaml leaves `params: {}`, so no `reasoning_effort` is sent. About 78% of that $0.3396 bought nothing.

Raising the number is not the fix, and `judge.py` says so at length (lines 185–216): `JUDGE_MAX_TOKENS` is fleet-wide, it feeds `judge_needed_tokens` and `undersized_families`, and `providers.build_payload` assigns `payload["max_tokens"] = req.max_tokens` **after** merging `role_params`, so a per-model override cannot express a bigger cap anyway. The fix the code already names is `role_params.judge.reasoning_effort` — exactly what `groq/qwen/qwen3.6-27b` already carries as `'none'`.

**Files:**
- Modify: `configs/data_law_v1_exp_recovery.yaml` (the two `openai` model blocks)
- Test: `tests/test_build_config.py`

**Interfaces:**
- Consumes: Task 1's edited `openai` blocks.
- Produces: both gpt-5 refs carry `role_params: {judge: {reasoning_effort: 'minimal'}, tiebreak: {reasoning_effort: 'minimal'}}`. `providers.ModelClient.build_payload` merges `model.params < model.role_params[role] < req.params`, so every judge/tiebreak call to those refs ships `reasoning_effort: minimal`.

Both refs declare `roles: [judge, tiebreak]`, and `config.py` (~line 349) refuses a `role_params` key naming a role the model does not serve — so both keys are legal and both are needed.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_config.py`. The client class is `providers.ChatClient(provider, model, ...)` - both positional - and `ChatRequest` requires a `ref`. Constructing a `ChatClient` opens an `httpx.AsyncClient`, so hand it a `MockTransport` rather than letting it open a real one; no request is ever sent, because `build_payload` is pure.

```python
def test_gpt5_judge_and_tiebreak_calls_carry_minimal_reasoning_effort():
    """gpt-5 bills reasoning against max_completion_tokens.

    With params {} the model spent its whole 1,024-token reply budget
    thinking and returned empty content on 95 of 96 parse failures in
    exp_harmony. JUDGE_MAX_TOKENS cannot be the answer - it is fleet-wide,
    and build_payload assigns max_tokens AFTER the role_params merge - so
    the effort knob is the fix, as judge.py's own comment says.
    """
    import httpx

    from tuned.data.providers import ChatClient, ChatRequest

    def _never_called(request):  # build_payload performs no request
        raise AssertionError("build_payload must not perform a request")

    cfg = load_build_config(
        "configs/data_law_v1_exp_recovery.yaml", allow_unpinned=True
    )
    for name in ("gpt-5-mini", "gpt-5-nano"):
        ref = ModelRef("openai", name)
        provider, model = cfg.model_for(ref)
        for role in ("judge", "tiebreak"):
            assert model.role_params[role]["reasoning_effort"] == "minimal", (
                f"openai/{name} role_params[{role!r}] does not pin "
                "reasoning_effort"
            )
            client = ChatClient(
                provider, model, transport=httpx.MockTransport(_never_called)
            )
            payload = client.build_payload(
                ChatRequest(
                    messages=({"role": "user", "content": "score this"},),
                    ref=ref,
                    role=role,
                    max_tokens=1024,
                )
            )
            assert payload["reasoning_effort"] == "minimal"
            # the openai quirk renames the allowance and drops temperature
            assert payload["max_completion_tokens"] == 1024
            assert "max_tokens" not in payload
            assert "temperature" not in payload
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/test_build_config.py::test_gpt5_judge_and_tiebreak_calls_carry_minimal_reasoning_effort -v
```

Expected: FAIL with `KeyError: 'judge'` on `model.role_params`.

- [ ] **Step 3: Edit the config**

In `configs/data_law_v1_exp_recovery.yaml`, after `params: {}` on **each** gpt-5 block, add:

```yaml
        # gpt-5 bills REASONING against max_completion_tokens, and judge.py's
        # reply allowance is a fleet-wide 1,024 it cannot raise per model
        # (build_payload assigns max_tokens after this merge). Measured on
        # exp_harmony 2026-08-21: mean completion 989 tokens/request, and 95
        # of 96 parse failures were EMPTY replies - the model spent the whole
        # budget thinking and never reached the verdict. 'minimal' is the
        # lowest effort this family accepts; the free qwen seat carries the
        # same fix as reasoning_effort 'none'.
        role_params:
          judge: {reasoning_effort: 'minimal'}
          tiebreak: {reasoning_effort: 'minimal'}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/Scripts/python.exe -m pytest tests/test_build_config.py -v
```

Expected: PASS, whole file green.

- [ ] **Step 5: Commit**

```bash
git add configs/data_law_v1_exp_recovery.yaml tests/test_build_config.py
git commit -m "a judge that spends its reply budget thinking is not a judge"
```

---

### Task 3: Accept the `ground_faithfulness` axis alias

One of `exp_harmony`'s 96 `judge_parse_error` events is not an empty reply: the model emitted `ground_faithfulness` where the alias table accepts `grounding_faithfulness`. The parser then reported `missing axis 'grounding'` and threw away a complete, well-formed, paid verdict.

**Files:**
- Modify: `src/tuned/data/judge.py` (`_AXIS_ALIASES`, ~line 234)
- Test: `tests/test_build_judge.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `judge.parse_judge_reply(text: str) -> JudgeScores` accepts `ground_faithfulness` as a `grounding` alias. Signature unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_judge.py`:

```python
def test_ground_faithfulness_is_read_as_the_grounding_axis():
    """One exp_harmony reply used ground_faithfulness and was discarded.

    The reply was complete and well formed; only the axis key was one
    character off the rubric's own name. Throwing away a paid verdict over
    that is the parser being brittle, not defensive.
    """
    reply = (
        '{"ground_faithfulness": 4, "reasoning_validity": 3, '
        '"issue_coverage": 5, "rationale": "sound"}'
    )
    scores = parse_judge_reply(reply)
    assert (scores.grounding, scores.validity, scores.coverage) == (4, 3, 5)
    assert scores.rationale == "sound"
```

If `parse_judge_reply` is not already imported at module scope in that file, add it to the existing `from tuned.data.judge import ...` line.

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/test_build_judge.py::test_ground_faithfulness_is_read_as_the_grounding_axis -v
```

Expected: FAIL — `JudgeParseError: no scorable JSON object in judge reply (missing axis 'grounding')`

- [ ] **Step 3: Write the minimal implementation**

In `src/tuned/data/judge.py`, replace the `_AXIS_ALIASES` table:

```python
# "ground_faithfulness" is not a spelling we invented: a gpt-5-mini judge
# emitted it on exp_harmony 2026-08-21 and a complete, well-formed, PAID
# verdict was discarded for it. Aliases are read-side only - the rubric still
# asks for one spelling, and nothing here loosens what a score has to be.
_AXIS_ALIASES = {
    "grounding": ("grounding", "grounding_faithfulness", "ground_faithfulness"),
    "validity": ("validity", "reasoning_validity"),
    "coverage": ("coverage", "issue_coverage"),
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_build_judge.py -v
```

Expected: PASS, whole file green.

- [ ] **Step 5: Commit**

```bash
git add src/tuned/data/judge.py tests/test_build_judge.py
git commit -m "read the axis the judge named, not only the one the rubric spelled"
```

---

### Task 4: Declare which strata the cohort draws from

The evaluator assumes four strata of 20. The control store can fill three. Rather than shrink `n_per` (which weakens every stratum) or let `select_cohort` quietly return a short cohort (which would make a 60-pair result look like an 80-pair one), the experiment **declares** its strata in the config, and everything downstream validates against that declaration.

The default stays all four task types, so the live yaml is unaffected and the existing `test_underfilled_statute_stratum_is_blocked_and_inconclusive` keeps passing.

**Files:**
- Modify: `src/tuned/data/config.py` (`BuildCfg` ~line 56; `load_build_config` ~line 878)
- Modify: `configs/data_law_v1_exp_recovery.yaml` (`build:` block)
- Test: `tests/test_build_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `cfg.build.eval_cohort_strata` — a `tuple[str, ...] | None`. `None` means "every stratum the evaluator knows". Task 5 consumes it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_build_config.py`:

```python
def test_eval_cohort_strata_defaults_to_none_on_the_live_config():
    cfg = load_build_config("configs/data_law_v1.yaml", allow_unpinned=True)
    assert cfg.build.eval_cohort_strata is None


def test_recovery_config_declares_three_strata_as_a_tuple():
    cfg = load_build_config(
        "configs/data_law_v1_exp_recovery.yaml", allow_unpinned=True
    )
    assert cfg.build.eval_cohort_strata == (
        "irac_analysis",
        "drafting",
        "summarization",
    )


def test_eval_cohort_strata_refuses_empty_duplicate_and_non_string(tmp_path):
    """A stratum list is a pre-registration, so it is validated at load.

    An empty list, a repeat, or a non-string is a typo that would otherwise
    reach the cohort selector and silently change the cohort's size.
    """
    import yaml as _yaml

    base = _yaml.safe_load(
        Path("configs/data_law_v1_exp_recovery.yaml").read_text(encoding="utf-8")
    )
    for bad in ([], ["irac_analysis", "irac_analysis"], ["irac_analysis", 7], "drafting"):
        doc = dict(base)
        doc["build"] = dict(base["build"])
        doc["build"]["eval_cohort_strata"] = bad
        path = tmp_path / "bad.yaml"
        path.write_text(_yaml.safe_dump(doc), encoding="utf-8")
        with pytest.raises(ValueError, match="eval_cohort_strata"):
            load_build_config(path, allow_unpinned=True)
```

Ensure `from pathlib import Path`, `import pytest`, and `import yaml` (or the aliased import above) are available in that module.

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_build_config.py -v -k eval_cohort_strata
```

Expected: FAIL — `AttributeError: 'BuildCfg' object has no attribute 'eval_cohort_strata'`

- [ ] **Step 3: Add the field to `BuildCfg`**

In `src/tuned/data/config.py`, after the `pretreatment_manifest` field:

```python
    # Which task-type strata the matched evaluator's cohort draws from.
    # None = every stratum eval_matched knows (the four-way default), which
    # is what the live wave declares by saying nothing.
    #
    # An experiment declares a SHORTER list only when a stratum cannot be
    # filled for a DATA reason, and the reason belongs in the yaml next to
    # the list. Measured 2026-08-23: the control store holds 270 statute_qa
    # tasks and 0 seeds carrying a section_text distinct from the seed body,
    # so statute_section_eligible refuses every one of them and the stratum
    # is unfillable until real Gazette provision text exists. Declaring it
    # here - rather than letting the selector return a short cohort - is what
    # keeps a 60-pair result from being read as an 80-pair one.
    eval_cohort_strata: tuple[str, ...] | None = None
```

- [ ] **Step 4: Parse and validate it in `load_build_config`**

In `src/tuned/data/config.py`, immediately before the `build = BuildCfg(` call, add:

```python
    strata = build_raw.pop("eval_cohort_strata", None)
    if strata is not None:
        if not isinstance(strata, list) or not strata:
            raise ValueError(
                "build.eval_cohort_strata must be a non-empty YAML list of "
                f"task-type names, got {strata!r}"
            )
        if not all(isinstance(name, str) and name.strip() for name in strata):
            raise ValueError(
                "build.eval_cohort_strata entries must be non-empty strings, "
                f"got {strata!r}"
            )
        strata = tuple(name.strip() for name in strata)
        if len(set(strata)) != len(strata):
            raise ValueError(
                f"build.eval_cohort_strata repeats a stratum: {strata!r}. A "
                "repeat would double one stratum's share of the cohort."
            )
```

and add `eval_cohort_strata=strata,` to the `BuildCfg(...)` call alongside `require_pretreatment_manifest=` and `pretreatment_manifest=`.

- [ ] **Step 5: Declare the strata in the recovery config**

In `configs/data_law_v1_exp_recovery.yaml`, in the `build:` block after `pretreatment_manifest:`, add:

```yaml
  # THREE STRATA, NOT FOUR, AND THE MISSING ONE IS A DATA FACT NOT A CHOICE.
  # eval_matched.statute_section_eligible requires a seed whose
  # meta_json.section_text differs from the seed body. Measured 2026-08-23 on
  # the read-only control store: 270 statute_qa tasks, 0 eligible seeds - no
  # seed in that database carries provision text at all, so the stratum
  # cannot be filled and select_cohort blocks with
  # underfilled-stratum:statute_qa before anything is spent.
  #
  # Filling it needs official Gazette Act bodies attached to seeds (the
  # gazette.py work on branch law-v1-foundation, whose manifest holds
  # identities only), and attaching them would mean WRITING to the control
  # store, which this arm may not do.
  #
  # Consequence, recorded here so the report cannot forget it: the cohort is
  # 60 pairs, not 80. McNemar runs on a smaller discordant pool, and this arm
  # may make NO claim about statute_qa - which is also the stream whose live
  # accepts went 0 -> 2 under the corrected prompts, so the gap is real.
  eval_cohort_strata: [irac_analysis, drafting, summarization]
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_build_config.py -v
```

Expected: PASS, whole file green.

- [ ] **Step 7: Commit**

```bash
git add src/tuned/data/config.py configs/data_law_v1_exp_recovery.yaml tests/test_build_config.py
git commit -m "let an experiment declare the strata its control store can fill"
```

---

### Task 5: The cohort selector and manifest honour the declared strata

`select_cohort`, `cohort_manifest`, `validate_pretreatment_manifest`, `write_manifest`, `evaluate` and `require_pretreatment_manifest` all hardcode the four-way cohort — including a literal `!= 80` in the validator and `set(TASK_TYPES)` as `generated_types` in `evaluate`. Thread the declared strata through all six, defaulting to `TASK_TYPES` so nothing existing changes.

The strata are a property of the **cohort**, not of the teacher, so they are threaded as an explicit parameter rather than added to `EvalContract`. That matters at the call site: `main()`'s `write-manifest` builds its contract from the **control** config but must take its strata from the **treatment** config, and folding strata into `EvalContract` would make that mismatch invisible.

**Files:**
- Modify: `src/tuned/data/eval_matched.py`
- Test: `tests/test_build_eval_matched.py`

**Interfaces:**
- Consumes: `cfg.build.eval_cohort_strata` from Task 4.
- Produces:
  - `select_cohort(store, n_per: int = STRATUM_N, *, strata: Sequence[str] | None = None) -> Selection`
  - `cohort_manifest(selection, *, contract: EvalContract, strata: Sequence[str] | None = None) -> dict` — the returned dict gains a `"strata"` key (a list) and its `"n"` is `n_per * len(strata)`
  - `validate_pretreatment_manifest(manifest, *, treatment: EvalContract, strata: Sequence[str] | None = None) -> tuple[bool, tuple[str, ...]]`
  - `write_manifest(path, control_store, *, treatment_store=None, treatment_empty=False, contract, n_per=STRATUM_N, strata=None) -> dict`
  - `evaluate(..., strata: Sequence[str] | None = None) -> EvalReport`
  - `require_pretreatment_manifest(cfg, *, repo_root=None) -> dict` — unchanged signature; reads `cfg.build.eval_cohort_strata` internally

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_build_eval_matched.py`:

```python
def test_declared_strata_fill_where_the_four_way_default_blocks(store):
    """One pool, two contracts: four strata block on statute_qa, three do not.

    The statute_qa seeds carry a section_text equal to the seed body, which
    is the live control store's actual condition - 270 tasks, 0 eligible
    seeds - so statute_section_eligible refuses every one of them.
    """
    for task_type in TASK_TYPES:
        for i in range(22):
            _plant_unit(
                store,
                seed_id=f"{task_type}-{i:03d}",
                task_type=task_type,
                gates=_gate_rows(),
                state="accepted",
                section=SOURCE_BODY if task_type == "statute_qa" else SECTION,
            )

    four = E.select_cohort(store, n_per=20)
    assert four.blocked is True
    assert "underfilled-stratum:statute_qa" in four.reason

    declared = ("irac_analysis", "drafting", "summarization")
    three = E.select_cohort(store, n_per=20, strata=declared)
    assert three.blocked is False
    assert len(three.pairs) == 60
    assert set(three.stratum_counts) == set(declared)
    assert "statute_qa" not in {row.task_type for row in three.pairs}


def test_unknown_stratum_is_refused_rather_than_silently_empty(store):
    """A typo'd stratum would otherwise select nothing and block on itself."""
    with pytest.raises(ValueError, match="unknown cohort strata"):
        E.select_cohort(store, strata=("irac_analysis", "not_a_task_type"))



def test_manifest_records_strata_and_sizes_n_from_them(tmp_path, store):
    _eligible_pool(store, per_type=25)
    strata = ("irac_analysis", "drafting", "summarization")
    with Store.open(tmp_path / "treat" / "law_v1.sqlite3") as treatment:
        manifest = E.write_manifest(
            tmp_path / "cohort.json",
            store,
            treatment_store=treatment,
            contract=_contract(),
            strata=strata,
        )
    assert manifest["strata"] == list(strata)
    assert manifest["n"] == 60
    assert len(manifest["pairs"]) == 60
    ok, reasons = E.validate_pretreatment_manifest(
        manifest, treatment=_contract(), strata=strata
    )
    assert ok, reasons


def test_manifest_written_for_three_strata_is_refused_against_four(store, tmp_path):
    """A 60-pair cohort must not validate as the 80-pair contract."""
    _eligible_pool(store, per_type=25)
    strata = ("irac_analysis", "drafting", "summarization")
    with Store.open(tmp_path / "treat" / "law_v1.sqlite3") as treatment:
        manifest = E.write_manifest(
            tmp_path / "cohort.json",
            store,
            treatment_store=treatment,
            contract=_contract(),
            strata=strata,
        )
    ok, reasons = E.validate_pretreatment_manifest(manifest, treatment=_contract())
    assert ok is False
    assert "contract-mismatch:strata" in reasons
```

These reuse the module's existing helpers: `_plant_unit(..., section=...)` (which the pre-existing `test_underfilled_statute_stratum_is_blocked_and_inconclusive` already uses the same way), `_eligible_pool` at line ~584, `_contract()`, `SECTION`, `SOURCE_BODY`, and `Store` — all already imported in that file. The four-way default keeps its own pre-existing test; nothing here replaces it.

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_build_eval_matched.py -v -k "strata"
```

Expected: FAIL — `TypeError: select_cohort() got an unexpected keyword argument 'strata'`

- [ ] **Step 3: Thread strata through `select_cohort`**

In `src/tuned/data/eval_matched.py`, change the signature and the three places that read `TASK_TYPES`:

```python
def select_cohort(
    store, n_per: int = STRATUM_N, *, strata: Sequence[str] | None = None
) -> Selection:
    """Pair the latest control generation per seed, within the declared strata.

    ``strata`` defaults to TASK_TYPES - the four-way cohort the live contract
    pre-registered. An experiment passes a shorter tuple ONLY when a stratum
    is unfillable for a data reason; the tuple then travels into the manifest
    so a short cohort can never be read as a full one.
    """
    wanted = tuple(strata) if strata else TASK_TYPES
    unknown = sorted(set(wanted) - set(TASK_TYPES))
    if unknown:
        raise ValueError(f"unknown cohort strata: {unknown}")
    gold = gold_linked_seeds(store)
    ...
    buckets: dict[str, dict[str, dict]] = {name: {} for name in wanted}
```

and in the row loop replace `if task_type not in TASK_TYPES:` with `if task_type not in wanted:`, and in the ranking loop replace `for task_type in TASK_TYPES:` with `for task_type in wanted:`.

- [ ] **Step 4: Thread strata through the manifest functions**

```python
def cohort_manifest(
    selection: Selection, *, contract: EvalContract, strata: Sequence[str] | None = None
) -> dict:
    wanted = tuple(strata) if strata else TASK_TYPES
    pairs = [...]  # unchanged
    return {
        "n": len(selection.pairs),
        "n_per_stratum": STRATUM_N,
        "strata": list(wanted),
        "task_types": list(wanted),
        "think_min": contract.think_min,
        "teacher_family": contract.teacher_family,
        "teacher_model": contract.teacher_model,
        "gate_contract": list(contract.gate_contract),
        "control_fingerprint": control_snapshot_fingerprint(selection, contract),
        "pairs": pairs,
    }
```

```python
def validate_pretreatment_manifest(
    manifest: Mapping | None,
    *,
    treatment: EvalContract,
    strata: Sequence[str] | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Validate a persisted pre-treatment manifest against treatment."""
    if not isinstance(manifest, Mapping):
        return False, ("pretreatment-manifest-invalid",)
    wanted = tuple(strata) if strata else TASK_TYPES
    reasons: list[str] = []
    if tuple(manifest.get("strata") or ()) != wanted:
        reasons.append("contract-mismatch:strata")
    expected_n = STRATUM_N * len(wanted)
    pairs = list(manifest.get("pairs") or ())
    if manifest.get("n") != expected_n or len(pairs) != expected_n:
        reasons.append("pretreatment-manifest-n")
    counts = {name: 0 for name in wanted}
    for row in pairs:
        if not isinstance(row, Mapping):
            continue
        task_type = row.get("task_type")
        if task_type in counts:
            counts[task_type] += 1
    for task_type in wanted:
        if counts[task_type] != STRATUM_N:
            reasons.append(f"underfilled-stratum:{task_type}")
    # ...the gate_contract / think_min / teacher checks are unchanged
```

```python
def write_manifest(
    path,
    control_store,
    *,
    treatment_store=None,
    treatment_empty: bool = False,
    contract: EvalContract,
    n_per: int = STRATUM_N,
    strata: Sequence[str] | None = None,
) -> dict:
    ...
    selected = select_cohort(control_store, n_per=n_per, strata=strata)
    if selected.blocked:
        raise ValueError(selected.reason or "underfilled-stratum")
    manifest = cohort_manifest(selected, contract=contract, strata=strata)
    ...
```

- [ ] **Step 5: Thread strata through `evaluate` and `require_pretreatment_manifest`**

In `evaluate`, add `strata: Sequence[str] | None = None` to the keyword-only parameters; change `selected = select_cohort(control_store, n_per=n_per)` to `select_cohort(control_store, n_per=n_per, strata=strata)`; and change `generated_types=generated_types or set(TASK_TYPES)` to:

```python
        generated_types=generated_types or set(strata or TASK_TYPES),
```

In `require_pretreatment_manifest`, pass the running config's declaration:

```python
    ok, reasons = validate_pretreatment_manifest(
        manifest,
        treatment=contract_from_config(cfg),
        strata=getattr(cfg.build, "eval_cohort_strata", None),
    )
```

Add `Sequence` to the module's `collections.abc` import if it is not already there.

- [ ] **Step 6: Wire the CLI to read strata from the TREATMENT config**

In `main()`, the `write-manifest` branch builds its contract from the control config. Its strata must come from the treatment config, because the strata are the experiment's declaration:

```python
    if args.command == "write-manifest":
        control_contract = contract_from_config(control_cfg)
        strata = getattr(treatment_cfg.build, "eval_cohort_strata", None)
```

and pass `strata=strata` to both `write_manifest(...)` calls in that branch. In the `evaluate` branch, pass `strata=getattr(treatment_cfg.build, "eval_cohort_strata", None)` to `evaluate(...)`.

- [ ] **Step 7: Run the focused tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_build_eval_matched.py -v
```

Expected: PASS, whole file green — including the pre-existing `test_underfilled_statute_stratum_is_blocked_and_inconclusive` and `test_deterministic_twenty_by_four_excludes_gold_seeds`.

- [ ] **Step 8: Run the full suite**

```bash
.venv/Scripts/python.exe -m pytest tests -q
```

Expected: all green. This is the one full-suite run in the plan; the baseline to beat is the clean-worktree `3108 passed, 4 skipped`, plus the tests added by Tasks 1–5. If pytest complains about a missing `--basetemp` parent, create the directory first rather than changing the invocation.

- [ ] **Step 9: Commit**

```bash
git add src/tuned/data/eval_matched.py tests/test_build_eval_matched.py
git commit -m "a sixty-pair cohort must not validate as the eighty-pair contract"
```

---

### Task 6: Write the pre-treatment manifest (free, read-only on the control store)

This is the first task that touches real data. It reads the live control store through `open_eval_store` (`mode=ro` + `PRAGMA query_only`) and writes one JSON file. No provider is called, no store is created, nothing is spent.

`write_manifest` refuses if the treatment store already has generations — that is the whole point of a *pre*-treatment manifest, and `data/build/exp_recovery` does not exist yet, so the `treatment_empty=True` branch will be taken.

**Files:**
- Create: `.superpowers/sdd/law-v1-recovery_893eff3d/cohort-manifest.json`

**Interfaces:**
- Consumes: Task 5's `write_manifest(..., strata=...)` and Task 4's `cfg.build.eval_cohort_strata`.
- Produces: the file at the path `configs/data_law_v1_exp_recovery.yaml` names in `build.pretreatment_manifest`, containing `"strata": ["irac_analysis", "drafting", "summarization"]`, `"n": 60`, and 60 `pairs` entries.

- [ ] **Step 1: Confirm the control store is still where the config expects it and is not being written**

```bash
ls -la data/build/state/law_v1.sqlite3
ls -la data/build/exp_recovery 2>&1 | head -3
```

Expected: the control DB exists; `data/build/exp_recovery` does not exist. If `exp_recovery` already exists with a store in it, STOP and report — the manifest can no longer be pre-treatment.

- [ ] **Step 2: Write the manifest**

```bash
.venv/Scripts/python.exe -m tuned.data.eval_matched \
  --control-config configs/data_law_v1.yaml \
  --treatment-config configs/data_law_v1_exp_recovery.yaml \
  write-manifest \
  --out .superpowers/sdd/law-v1-recovery_893eff3d/cohort-manifest.json
```

Expected: exit 0, no output. If it exits with `underfilled-stratum:...` for a stratum other than `statute_qa`, STOP and report — the control store has drifted from the 2026-08-23 measurement and the declared strata need revisiting, not widening.

- [ ] **Step 3: Verify what was written**

```bash
.venv/Scripts/python.exe -c "
import json, collections
m = json.load(open('.superpowers/sdd/law-v1-recovery_893eff3d/cohort-manifest.json'))
print('strata      :', m['strata'])
print('n           :', m['n'], 'pairs:', len(m['pairs']))
print('per stratum :', collections.Counter(p['task_type'] for p in m['pairs']))
print('fingerprint :', m['control_fingerprint'][:16])
print('teacher     :', m['teacher_family'], m['teacher_model'], 'think_min', m['think_min'])
"
```

Expected exactly:

```text
strata      : ['irac_analysis', 'drafting', 'summarization']
n           : 60 pairs: 60
per stratum : Counter({'irac_analysis': 20, 'drafting': 20, 'summarization': 20})
teacher     : gpt-oss gpt-oss-120b think_min 500
```

- [ ] **Step 4: Verify the config gate now accepts it**

```bash
.venv/Scripts/python.exe -c "
from tuned.data.config import load_build_config
from tuned.data.eval_matched import require_pretreatment_manifest
cfg = load_build_config('configs/data_law_v1_exp_recovery.yaml', allow_unpinned=True)
m = require_pretreatment_manifest(cfg)
print('manifest accepted, n =', m['n'])
"
```

Expected: `manifest accepted, n = 60`. A `ValueError: pretreatment manifest refused: ...` here means Task 5's validator and Task 4's declaration disagree — fix that before going further, never by editing the JSON by hand.

- [ ] **Step 5: Commit**

```bash
git add .superpowers/sdd/law-v1-recovery_893eff3d/cohort-manifest.json
git commit -m "pre-register the sixty pairs this arm is allowed to be judged on"
```

If `.superpowers/` is gitignored in this worktree (check `.superpowers/.gitignore`), do not force-add it. Instead record the manifest's `control_fingerprint` and per-stratum counts in the Task 10 report, and note in the commit for Task 5 that the manifest lives outside git.

---

### Task 7: Preflight the pool and the wallet before spending

`generate.print_preflight` runs `providers.pool_gaps`, which walks the same preference order `judge.py` walks and reports every judge/tiebreak slot the pool cannot fill for the longest row the length band permits. The 34 `judge_error` rows on the live store all carry one disposition — `judge-slot-b: role 'judge': no eligible model (skipped: cooling, family-excluded)` — which is exactly what this preflight exists to catch before a paid judge A is bought and its row parked.

On the recovery config the family bug is already fixed (`family: gpt-5`, not `gpt-oss`), so slot B is gemma with the two gpt-5 refs behind it. This task proves that on the real config rather than assuming it.

**Files:** none modified. This is a gate.

**Interfaces:**
- Consumes: Tasks 1–2 (wallet and effort), Task 6 (manifest).
- Produces: a recorded preflight with zero pool gaps, or a STOP.

- [ ] **Step 1: Confirm which API keys are present**

```bash
.venv/Scripts/python.exe -c "
import os
for k in ('CEREBRAS_API_KEY','GROQ_API_KEY','MISTRAL_API_KEY','OPENAI_API_KEY','LIGHTNING_API_KEY'):
    print(f'{k:20} {\"set\" if os.environ.get(k) else \"MISSING\"}')
"
```

Which keys are set changes which families are eligible, and `pool_gaps`' advice depends on it. Record the result in the Task 10 report. `CEREBRAS_API_KEY` (generator) and at least two of Groq / Cerebras-gemma / OpenAI (judge slots A and B) must be set. If `.env` holds them, source it the way the other runbooks in this repo do before running.

- [ ] **Step 2: Run the preflight without generating**

```bash
.venv/Scripts/python.exe -m tuned.data.generate \
  --config configs/data_law_v1_exp_recovery.yaml \
  --max-batches 0
```

Expected: the preflight banner prints, reports **zero pool gaps**, and the process exits without claiming a task. `--max-batches 0` is a genuine preflight-only path, not a trick: `generate.main` calls `print_preflight` at line ~2205 *before* the worker loop, and that loop is `while max_batches is None or batches < max_batches`, so `0 < 0` is false and no batch is ever claimed.

- [ ] **Step 3: STOP-check the output**

Read the preflight output and confirm all four:

1. zero pool gaps for role `judge` **and** role `tiebreak`;
2. the generator resolves to `cerebras/gpt-oss-120b`;
3. the workdir is `data/build/exp_recovery` — **not** `data/build`;
4. the manifest line reports the 60-pair cohort as accepted.

If any gap is reported, STOP. A gap means a slot the pool cannot fill for the longest row this band permits, and running anyway reproduces the exact 34-row park the live store is sitting in. Report the gap and its advised window instead of proceeding.

- [ ] **Step 4: Record, do not commit**

Nothing is committed here. Paste the preflight output into the Task 10 report draft.

---

### Task 8: The bounded probe — PAID, OPERATOR CHECKPOINT

**Do not start this task without the operator saying go.** Everything before it was free; this spends free-tier quota and up to $1.66 of real money.

Authorised envelope: one cohort of 60 tasks on the free `cerebras/gpt-oss-120b` tier, judged under the corrected fleet with the OpenAI refs capped at `usd_cap: 1.66` and only ever reached after every free judge has been tried.

**Files:** creates `data/build/exp_recovery/` (store, streams, logs). No source file changes.

**Interfaces:**
- Consumes: Tasks 1–7.
- Produces: `data/build/exp_recovery/state/law_v1.sqlite3` holding generations for the 60 manifest pairs.

- [ ] **Step 1: Plan the wave**

```bash
.venv/Scripts/python.exe -m tuned.data.tasks \
  --config configs/data_law_v1_exp_recovery.yaml \
  --n 60 \
  --arm exp_recovery \
  --mix irac_analysis=1,drafting=1,summarization=1
```

Expected: 60 tasks planned in `data/build/exp_recovery`, 20 per declared stratum, none `statute_qa`. The `--mix` format is `task_type=weight` comma-separated with float weights (`tasks.parse_mix`, line ~659), so equal weights of 1 give the three-way split.

- [ ] **Step 2: Verify the wave landed in the isolated store only**

```bash
.venv/Scripts/python.exe -c "
import sqlite3, collections
c = sqlite3.connect('file:data/build/exp_recovery/state/law_v1.sqlite3?mode=ro', uri=True)
print('exp_recovery tasks:', list(c.execute('select state, count(*) from task group by 1')))
print('by type:', list(c.execute('select task_type, count(*) from task group by 1')))
c2 = sqlite3.connect('file:data/build/state/law_v1.sqlite3?mode=ro', uri=True)
print('LIVE tasks (must be unchanged):', list(c2.execute('select state, count(*) from task group by 1')))
"
```

Expected: `exp_recovery` has 60 pending tasks across three types. The live store must still read exactly:

```text
stale_prompt 419, rejected 414, pending 127, judging 43, judge_error 34, accepted 15, generating 8
```

If the live numbers moved, STOP immediately — something wrote to the control store.

- [ ] **Step 3: Generate**

```bash
.venv/Scripts/python.exe -m tuned.data.generate \
  --config configs/data_law_v1_exp_recovery.yaml \
  --n-workers 4 \
  --max-batches 20
```

Expected: generations accumulate; `format_parked` rows appear for style exhaustion; `rejected` only for permanent law gates. Watch that `harmony_completions` is active — the parsed think should start with the prefill `"I start from the facts. "` on continued rows.

- [ ] **Step 4: Check the format yield before buying any judges**

```bash
.venv/Scripts/python.exe -c "
import sqlite3
from collections import defaultdict
c = sqlite3.connect('file:data/build/exp_recovery/state/law_v1.sqlite3?mode=ro', uri=True)
best = {}
for tid, gid, att in c.execute('select task_id, gen_id, attempt from generation'):
    if tid not in best or att > best[tid][1]:
        best[tid] = (gid, att)
gids = {g for g, _ in best.values()}
per = defaultdict(dict)
for gid, gate, p in c.execute('select gen_id, gate, passed from gate_result'):
    if gid in gids:
        per[gid][gate] = p
tot = len(per)
ap = sum(1 for g in per.values() if all(g.values()))
print(f'latest-per-task all-gates-pass {ap}/{tot} = {100*ap/max(tot,1):.1f}%')
f = defaultdict(int)
for g in per.values():
    for gate, p in g.items():
        if not p:
            f[gate] += 1
for k, v in sorted(f.items(), key=lambda x: -x[1]):
    print(f'  {k:22} {v:4} ({100*v/max(tot,1):.0f}%)')
"
```

Reference points: live corrected-prompt arm is **30.2%** latest-per-task; `exp_harmony` is **64.6%**. A result at or above the live arm is the minimum for judging to be worth buying. Below it, STOP and report — the prefill did not transfer to this cohort, and judging a broken format wastes the wallet.

- [ ] **Step 5: Judge**

```bash
.venv/Scripts/python.exe -m tuned.data.judge \
  --config configs/data_law_v1_exp_recovery.yaml \
  --n-workers 2 \
  --max-batches 30
```

- [ ] **Step 6: Check the wallet and the parse-error rate**

```bash
.venv/Scripts/python.exe -c "
import sqlite3
price = {'gpt-5-mini': (0.25, 2.00), 'gpt-5-nano': (0.05, 0.40)}
c = sqlite3.connect('file:data/build/exp_recovery/state/law_v1.sqlite3?mode=ro', uri=True)
tot = 0.0
for m, r, pt, ct in c.execute(
    \"select model, sum(requests), sum(prompt_tokens), sum(completion_tokens) \"
    \"from budget_ledger where provider='openai' group by 1\"
):
    pi, po = price.get(m, (0, 0))
    usd = pt / 1e6 * pi + ct / 1e6 * po
    tot += usd
    print(f'{m:12} req={r:4} completion={ct:7} mean={ct/max(r,1):6.0f} usd={usd:.4f}')
print(f'exp_recovery OpenAI spend: \${tot:.4f} of the 1.66 remaining')
print('judge_parse_error events:', c.execute(
    \"select count(*) from run_event where kind='judge_parse_error'\").fetchone()[0])
print('judge_route_error events:', c.execute(
    \"select count(*) from run_event where kind='judge_route_error'\").fetchone()[0])
print('task states:', list(c.execute('select state, count(*) from task group by 1')))
"
```

The Task 2 fix is confirmed when **mean completion tokens per gpt-5 request is well under 1,024** (exp_harmony's broken figure was 989) and `judge_parse_error` is near zero. `judge_route_error` must be 0 — any non-zero value means slot B is unfillable again and Task 7's preflight missed it.

- [ ] **Step 7: Do not commit runtime artifacts**

`data/build/` is gitignored runtime. Nothing is committed in this task. Record the numbers for Task 10.

---

### Task 9: Run the matched evaluation

**Files:** creates `data/build/exp_recovery/candidate_decisions.json`.

**Interfaces:**
- Consumes: Task 6's manifest, Task 8's treatment store.
- Produces: an `EvalReport` printed as `decision=... synthesis=...` plus `reasons=...`.

The lockbox is the 46 `gold_label` rows on the control store — 10 `accept`, 36 `reject`, matching `LOCKBOX_ACCEPTS` / `LOCKBOX_REJECTS`, so `lockbox_confirm` will run. **Those 46 labels are Fable-5 model references, not human gold.** Whatever `decision` comes back is a model-agreement statement, and Task 10 must say so.

- [ ] **Step 1: Build the candidate-decision map**

`evaluate` requires a persisted JSON map of lockbox `gen_id` → decision under the candidate rule. Mirror the helper the tests use (`_baseline_candidate_map` in `tests/test_build_eval_matched.py`):

```bash
.venv/Scripts/python.exe -c "
import json
from tuned.data import eval_matched as E
control = E.open_eval_store('data/build/state/law_v1.sqlite3')
try:
    labels = list(control.gold_labels())
    judgements = control.judgements_by_gen(int(r['gen_id']) for r in labels)
    out = {}
    for row in labels:
        gid = int(row['gen_id'])
        out[gid] = E.dual_judge_decision(
            judgements.get(gid, {}),
            already_regenerated=E._already_regenerated_for_gen(control, gid),
        )
finally:
    control.close()
json.dump({str(k): v for k, v in out.items()},
          open('data/build/exp_recovery/candidate_decisions.json', 'w'), indent=2)
print('candidate decisions:', len(out))
"
```

Expected: `candidate decisions: 46`. A different count means the lockbox is not the 46 rows `LOCKBOX_N` expects and `evaluate` will refuse; report it rather than padding the map.

- [ ] **Step 2: Evaluate**

```bash
.venv/Scripts/python.exe -m tuned.data.eval_matched \
  --control-config configs/data_law_v1.yaml \
  --treatment-config configs/data_law_v1_exp_recovery.yaml \
  evaluate \
  --manifest .superpowers/sdd/law-v1-recovery_893eff3d/cohort-manifest.json \
  --candidate-decisions data/build/exp_recovery/candidate_decisions.json
```

Expected: two lines — `decision=<verdict> synthesis=<label>` and `reasons=<comma-separated>`. Capture both verbatim.

- [ ] **Step 3: Confirm the live store was never written**

```bash
.venv/Scripts/python.exe -c "
import sqlite3
c = sqlite3.connect('file:data/build/state/law_v1.sqlite3?mode=ro', uri=True)
print(sorted(c.execute('select state, count(*) from task group by 1')))
print('gold_label rows:', c.execute('select count(*) from gold_label').fetchone()[0])
print('judge_threshold rows:', c.execute('select count(*) from judge_threshold').fetchone()[0])
"
```

Expected exactly:

```text
[('accepted', 15), ('generating', 8), ('judge_error', 34), ('judging', 43), ('pending', 127), ('rejected', 414), ('stale_prompt', 419)]
gold_label rows: 46
judge_threshold rows: 0
```

Any deviation is a custody failure. Report it and stop.

---

### Task 10: Write the report

**Files:**
- Create: `docs/reports/2026-08-23-recovery-arm-probe.md`

**Interfaces:**
- Consumes: every measurement from Tasks 6–9.
- Produces: the record. No code depends on it.

- [ ] **Step 1: Write the report**

Create `docs/reports/2026-08-23-recovery-arm-probe.md` with these sections, filled from the captured output — no placeholders, no rounded-from-memory numbers:

1. **What ran and where** — branch, worktree, config, workdir, the live store's read-only status and its unchanged task-state line from Task 9 Step 3.
2. **Fleet fixes and what they bought** — the gpt-5 mean-completion figure before ($989/request, 27 usable judgements from 124 calls) and after (Task 8 Step 6); `judge_parse_error` and `judge_route_error` counts; the `ground_faithfulness` alias.
3. **Cohort** — 60 pairs across three strata, the `control_fingerprint`, and the statute_qa exclusion stated as a data fact: 270 tasks, 0 eligible seeds, blocked until real Gazette provision text exists on branch `law-v1-foundation`.
4. **Format yield** — Task 8 Step 4's table against the two reference points (live corrected-prompt arm 30.2%, `exp_harmony` 64.6%).
5. **Judge outcome** — accept/reject/parked counts, the axis means, and the OpenAI spend against the $1.66 headroom.
6. **The evaluator's verdict** — `decision` and `reasons` verbatim.
7. **What this may not claim** — in its own section, in plain words:
   - the lockbox is 46 **Fable-5 model-generated references, not human gold**, so `decision` is a model-agreement statement and not evidence of legal correctness;
   - `judge_threshold` is still empty and stays empty;
   - the cohort is 60 pairs, so McNemar ran on a smaller discordant pool than the 80-pair contract pre-registered;
   - **nothing about `statute_qa`** — the stream that went 0 → 2 live accepts under the corrected prompts is entirely absent from this arm;
   - the live wave is not promoted: `configs/data_law_v1.yaml` still has `think_min: 500`, no `harmony_*` flags, no `prompt_overlay`, and its gpt-5 refs still declare `family: gpt-oss`;
   - 419 `stale_prompt` + 81 old-SHA `rejected` tasks on the control store are still unregenerated against the corrected templates.
8. **What would come next** — the two decisions this probe hands back: whether to promote the prefill to the live wave, and whether to authorise the Gazette acquisition that would make `statute_qa` measurable.

- [ ] **Step 2: Commit**

```bash
git add docs/reports/2026-08-23-recovery-arm-probe.md
git commit -m "record what the recovery probe measured and what it may not claim"
```

---

## Explicit non-goals

- No promotion to the live wave. `configs/data_law_v1.yaml` is not edited by any task here.
- No `--reopen` on the control store, and no migration of its 359 `exhausted:regenerate:*` rows to `format_parked`.
- No threshold fitting, no `calibrate --fit`, no `--ingest`.
- No Gazette acquisition, no statute-text attachment, no merge of the uncommitted `law-v1-foundation` work.
- No training run and no Hub push.
