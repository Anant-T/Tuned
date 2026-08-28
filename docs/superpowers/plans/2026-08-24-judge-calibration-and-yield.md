# Judge Calibration and Yield Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the judge instruments so accept rate means something, then bank the measured efficiency wins — without touching the frozen control store.

**Architecture:** Four independent investigations (2026-08-24) converged on one conclusion: the pipeline's *measurement* is broken before its *quality* is. A prompt-overlay regression saturated the tiebreak arbiter, and a one-word collision in the grounding rubric made its middle band unreachable, funnelling 41 of 101 judgements into the only band that hard-fails. Until both are fixed, no accept-rate comparison between arms is interpretable — which is why the otherwise-decisive `harmony_s1_continue` A/B could only be settled on deterministic gate counts. This plan fixes the instruments first (Tasks 1–2), removes a stream that is structurally unwinnable (Task 3), repairs the one chunker defect that is custody-safe (Task 4), builds the project's first external anchor (Task 5), and only then re-measures (Task 6).

**Tech Stack:** Python 3.12, SQLite, pytest, TRL/transformers downstream. Teacher `cerebras/gpt-oss-120b` (free tier); judges `groq/qwen3.6-27b`, `cerebras/gemma-4-31b`, `mistral/mistral-large-latest`, `openai/gpt-5-mini`.

## Global Constraints

- The live control store `data/build/state/law_v1.sqlite3` is **READ-ONLY**. Never `Store.open()` it, never `--reopen`, never migrate, never re-chunk. Read only via `sqlite3.connect('file:...?mode=ro', uri=True)` or `eval_matched.open_eval_store`.
- **OpenAI spend: $2.00 TOTAL, hard, across `gpt-5-mini` + `gpt-5-nano`.** Spent to date: $0.3396 (exp_harmony) + $0.023 (exp_s1) = **$0.3626. Remaining: $1.637.** `usd_cap` resets per UTC day *and* per store — recompute the remainder from the ledger before any resume that crosses 00:00 UTC.
- Closed-API **generations** (OpenAI/Gemini as teacher) never enter the training mix. They may only judge.
- Do not loosen the law gates: `citations`, `temporal`, `answer_key`, `statutory_grounding`.
- `judge_threshold` stays empty. The 46 `gold_label` rows are **model-generated references, not human gold** — verified: all 46 carry the identical `labeled_at` timestamp `2026-08-19T10:32:47.555143Z`, a single batch write. No quality warrant may be derived from them.
- No push, no merge to `main`, no training run.
- Never pass `--config configs/data_law_v1.yaml` to any command that writes.
- Worktree trap: always invoke `./.venv/Scripts/python.exe` inside `.claude/worktrees/law-v1-data-pipeline`. The shell's bare `python` resolves to the main checkout's venv, which lacks `openai_harmony`. Symptom of getting this wrong: 12 failures and the skip count dropping 19 → 4, which reads as a code regression but is an environment error.
- Set `PYTHONIOENCODING=utf-8` before printing any corpus or model text. The Windows cp1252 console renders curly quotes as U+FFFD and fakes "encoding corruption" — check bytes, never console glyphs.
- No AI-assistant attribution in any commit message, file, or artifact.
- `.git` is a FILE in this linked worktree — writing `.git/<tmpfile>` for a commit message fails. Use `git commit -m` or a temp file elsewhere.

---

## Evidence Base

Every number below is measured, not estimated. Full reports in the session scratchpad (`chunking-forensics.md`, `grounding-rubric.md`, `task-routing-fit.md`, `authority-enrichment.md`).

| finding | measurement |
|---|---|
| Tiebreak arbiter is saturated | mistral-large 9/9 then 18/18 accepts, validity 5.00, floor of 4 across all axis scores. Same model, same seat, frozen store: **validity 2.75 over 12**. |
| Cause is the overlay, not the model | `prompts_harmony/judge_tiebreak_v1.md` deleted the worked example containing `"validity": 2`. Cross-arm inflation: mistral **+2.25**, gemma +0.88, qwen +0.22. |
| Grounding band 3 is unreachable | Band 2 says "provision, case or **rule**"; band 3 says "proposition of substance". Same thing. Score distribution: **1 / 41 / 7 / 19 / 33**. Band 2 is the only band that hard-fails (`judge_policy.py:11`, `FAIL_MAX = 2`). |
| Teacher error is the dominant quality cap | Two independent methods agree: rationale classification **51%**, parent-document forensics **49.0%**. |
| Chunking is NOT the dominant cause | Decisive test — allegedly unsupported items matched against the parent document on disk: **18.4%** severed by chunker, **49.0%** absent from parent entirely. Enrichment control: base rate 0–13% over 400 random judgments, hits ~20× enriched. |
| Footnotes are systematically amputated | **96.2%** of inline footnote markers (1,909/1,984) point at a `[FOOTNOTES]` block in a different chunk. 29.9% of chunks carry an authority whose citation is physically elsewhere. |
| Role-aware segmentation never ran | `roles_json` empty on **60,603/60,603** seeds; **98.5%** of SC chunks are `roles_backend_none` packing fallbacks. |
| Routing is blind | `_candidate_seeds` (`tasks.py:266`) takes no `task_type` argument. `statute_qa` (`tasks.py:433`) is the only content-conditioned line in the planner. |
| Drafting is structurally unwinnable | `document_kind`, `party_context`, `focus_issue`, `question` are empty on **all 60,603 seeds** — every drafting task renders placeholder text against a delivered judgment. Accepts 4/20 vs summarization 13/20, **p=0.0095**. |
| s1 continue is the efficiency win | Cost per all-gates-clean row **66,808 → 11,725 tokens, a 5.7× improvement**, despite 2.3× more tokens per generation. Within-s1, higher k gets *worse* (11,725 → 15,760). |
| Best-of-k is already running | `MAX_ATTEMPTS = 3`. The current 41.7% *is* best-of-3. Judge-side resampling has never run (1/60 tasks has ≥2 judged generations) and its correlation is unmeasurable at n=1. |
| Statute enrichment is bounded | 915 distinct sections cited; **top 100 cover 73.2%**. Old-code (IPC/CrPC/IEA) is **99.91%** of mentions. Zenodo 5088102: 858 Central Acts as JSON, CC-BY-4.0. |
| Case enrichment is dead | 25,932 distinct case citations, top 100 cover 6.2%. Corpus self-coverage **0.33%**. All top-20 most-cited cases absent. |

**The A/B that this plan banks.** Same 60 pre-registered pairs, one flag different, measured on deterministic gates with no judge involved:

| | prefill only | prefill + s1 |
|---|---|---|
| all gates | 6.7% | **68.3%** |
| blocking gates | 73.3% | **78.3%** |
| `self_verification` fails | 90.0% | **13.3%** |

The accept-rate half of that comparison (41.7% → 56.7%) is **withdrawn as uninterpretable**: the arms drew different judge fleets. exp_s1 had gpt-5-mini in slot B for 22 rows grading grounding at 4.91 where gemma graded 2.88, and its tiebreaks doubled 9 → 18. Accepts touched by neither the tiebreak nor gpt-5-mini: exp_recovery 16, exp_s1 6.

---

## File Structure

| file | responsibility | task |
|---|---|---|
| `src/tuned/data/prompts_harmony/judge_tiebreak_v1.md` | tiebreak rubric — restore the low anchor | 1 |
| `src/tuned/data/prompts_harmony/judge_pointwise_v1.md` | pointwise rubric — split band 2 | 2 |
| `src/tuned/data/prompts/judge_pointwise_v1.md` | same split, base overlay | 2 |
| `tests/test_build_prompts.py` | template invariants | 1, 2 |
| `src/tuned/data/tasks.py:76-81` | `SYNTHESIS_MIX` weights | 3 |
| `tests/test_build_tasks.py` | mix invariants | 3 |
| `src/tuned/data/segment.py`, `src/tuned/data/chunks.py` | footnote resolution | 4 |
| `tests/test_build_chunks.py` | chunk-content invariants | 4 |
| `scripts/export_calibration_set.py` (new) | anchor-set export | 5 |
| `tests/test_export_calibration_set.py` (new) | export contract | 5 |
| `configs/data_law_v1_exp_measure.yaml` (new) | re-measurement arm | 6 |

---

### Task 1: Restore the tiebreak scoring anchor

The overlay removed the only low score the arbiter ever saw. The replacement text even says "There is no example verdict in this packet and you must not copy a score from these instructions" — an anti-anchoring measure that produced anchoring in the opposite direction. Restore a worked example, but keep the anti-copying warning that was the original change's legitimate goal.

**Files:**
- Modify: `src/tuned/data/prompts_harmony/judge_tiebreak_v1.md:31`
- Test: `tests/test_build_prompts.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: a tiebreak template containing at least one exemplar score ≤ 2. Task 6 depends on this being in place before it re-measures.

- [ ] **Step 1: Write the failing test**

```python
import re
from pathlib import Path

PROMPTS = Path("src/tuned/data/prompts")
HARMONY = Path("src/tuned/data/prompts_harmony")


def test_tiebreak_templates_carry_a_low_score_anchor():
    """Both tiebreak overlays must show the model a failing exemplar.

    Removing the worked example that contained `"validity": 2` saturated
    mistral-large to 18/18 accepts at validity 5.00, against 2.75 for the
    same model in the same seat on the frozen store. An arbiter that has
    never seen a low score does not produce one.
    """
    for root in (PROMPTS, HARMONY):
        text = (root / "judge_tiebreak_v1.md").read_text(encoding="utf-8")
        scores = [int(n) for n in re.findall(r'"(?:grounding|validity|coverage)":\s*(\d)', text)]
        assert scores, f"{root.name}: tiebreak template has no exemplar verdict at all"
        assert min(scores) <= 2, (
            f"{root.name}: lowest exemplar score is {min(scores)}; "
            "the arbiter needs a failing anchor or it saturates upward"
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m pytest tests/test_build_prompts.py::test_tiebreak_templates_carry_a_low_score_anchor -v`

Expected: FAIL on the harmony overlay — `tiebreak template has no exemplar verdict at all`. The base `prompts/` copy already passes; the test must fail only on `prompts_harmony`.

- [ ] **Step 3: Restore the anchor in the harmony overlay**

Replace line 31 of `src/tuned/data/prompts_harmony/judge_tiebreak_v1.md` with the exemplar plus the retained warning:

```markdown
{{"grounding": 4, "validity": 2, "coverage": 3, "rationale": "Cites the section it was given accurately, but the conclusion on limitation does not follow from the step before it."}}

That is the shape only — the three numbers there are an illustration, not a suggested score, and you must not copy them. The object must use exactly these keys and types: grounding (integer), validity (integer), coverage (integer), rationale (string). Every score is an integer from 1 to 5, scored independently of the other two. The rationale is at most 80 words and names the decisive reason for the lowest score you gave.
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m pytest tests/test_build_prompts.py -v`

Expected: PASS, and no other prompt test regresses.

- [ ] **Step 5: Commit**

```bash
git add src/tuned/data/prompts_harmony/judge_tiebreak_v1.md tests/test_build_prompts.py
git commit -m "an arbiter that never sees a low score never gives one"
```

---

### Task 2: Split the conflated grounding band

Band 2 currently merges two failures that call for opposite responses: *relies on a rule not in the materials* (a teacher correctly recalling real law the chunker amputated — 18.4% of cases are provably this) and *materially misstates one* (a genuine defect — ~49%). Because band 3's "proposition of substance rests on nothing given" describes the same thing as band 2's "rule not in the materials", band 3 is textually unreachable and everything lands in the only hard-failing band.

Separate them: unsupported-but-not-contradicted goes to band 3, misstatement stays at band 2, fabrication stays at band 1. This does **not** loosen a law gate — `citations` and `statutory_grounding` fire on section numbers and citation strings, not on unattributed principles.

**Files:**
- Modify: `src/tuned/data/prompts_harmony/judge_pointwise_v1.md:21`
- Modify: `src/tuned/data/prompts/judge_pointwise_v1.md` (same axis line)
- Test: `tests/test_build_prompts.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a grounding rubric whose bands are mutually exclusive. Task 6 measures the effect.

- [ ] **Step 1: Write the failing test**

```python
def test_grounding_bands_are_mutually_exclusive():
    """Band 2 must not claim the territory band 3 describes.

    Band 2 read "provision, case or rule that is not in the materials" while
    band 3 read "at least one proposition of substance rests on nothing
    given" — the same event. Band 2 is the only band that hard-fails
    (judge_policy.FAIL_MAX = 2), so the collision funnelled 41 of 101
    judgements into a failing band. Band 2 must now require a MISSTATEMENT,
    not mere absence.
    """
    for root in (PROMPTS, HARMONY):
        text = (root / "judge_pointwise_v1.md").read_text(encoding="utf-8")
        line = next(ln for ln in text.splitlines() if ln.startswith("grounding_faithfulness"))
        band2 = line.split("2:")[1].split("1:")[0]
        assert "misstates" in band2 or "contradict" in band2, (
            "band 2 must turn on misstatement, not absence"
        )
        assert "or rule that is not in the materials" not in band2, (
            "band 2 still swallows band 3's territory"
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m pytest tests/test_build_prompts.py::test_grounding_bands_are_mutually_exclusive -v`

Expected: FAIL — `band 2 still swallows band 3's territory`, on both overlays.

- [ ] **Step 3: Rewrite the grounding axis line**

In **both** `src/tuned/data/prompts/judge_pointwise_v1.md` and `src/tuned/data/prompts_harmony/judge_pointwise_v1.md`, replace the `grounding_faithfulness` line with:

```markdown
grounding_faithfulness — is every legal proposition traceable to the materials above, or to law correctly stated within them? 5: every proposition traceable and every citation accurate. 4: traceable throughout, with a slip that carries no weight, or a correctly stated principle of general law that the materials assume rather than set out. 3: at least one proposition of substance rests on authority not present in the materials, but nothing stated is contradicted by them. 2: materially misstates a provision or case that IS in the materials, or attributes to it something it does not say. 1: fabricated authority — an invented section or citation, or a holding attributed to a case that does not carry it.
```

The change in one sentence: absence now scores 3, misstatement scores 2. Correct general law the materials presuppose scores 4.

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m pytest tests/test_build_prompts.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tuned/data/prompts/judge_pointwise_v1.md src/tuned/data/prompts_harmony/judge_pointwise_v1.md tests/test_build_prompts.py
git commit -m "absence of authority and misstatement of it are not the same failure"
```

---

### Task 3: Park the drafting stream pending retarget

Drafting cannot succeed on this corpus as currently posed. `document_kind`, `party_context`, `focus_issue` and `question` are empty on all 60,603 seeds, so every drafting prompt renders placeholder text — "the document this matter now calls for" for "the party whose papers these are" — against a judgment that already disposed of the matter. Six judge rationales flag exactly that. Drafting burns **66,666 tokens per accepted row** against summarization's 18,028.

This is a **temporary park, not a deletion.** A coherent drafting task exists — a *downstream* instrument conditioned on the outcome — with an eligible pool of ~14,225 seeds against a quota need of ~810 (17× oversupply). That retarget is out of scope here and is filed as follow-on work. Reallocating drafting's weight buys **+7.2 accepted rows at zero extra tokens**.

**Files:**
- Modify: `src/tuned/data/tasks.py:76-81`
- Test: `tests/test_build_tasks.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SYNTHESIS_MIX` with `drafting` at 0.0 and weights summing to 1.0.

- [ ] **Step 1: Write the failing test**

```python
from tuned.data.tasks import SYNTHESIS_MIX


def test_drafting_is_parked_and_mix_still_sums_to_one():
    """Drafting is parked until its seeds carry the fields it needs.

    document_kind / party_context / focus_issue / question are empty on all
    60,603 seeds, so a drafting prompt renders placeholders against a
    judgment that already disposed of the matter. 66,666 tok/accepted row
    against summarization's 18,028. Park, do not delete: the retarget to a
    downstream instrument has a 14,225-seed eligible pool.
    """
    assert SYNTHESIS_MIX["drafting"] == 0.0
    assert abs(sum(SYNTHESIS_MIX.values()) - 1.0) < 1e-9
    assert set(SYNTHESIS_MIX) == {"irac_analysis", "statute_qa", "drafting", "summarization"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m pytest tests/test_build_tasks.py::test_drafting_is_parked_and_mix_still_sums_to_one -v`

Expected: FAIL — `assert 0.18 == 0.0`.

- [ ] **Step 3: Reallocate the weight**

Replace `src/tuned/data/tasks.py:76-81`:

```python
# drafting is PARKED at 0.0, not removed (2026-08-24). Every drafting prompt
# renders placeholder slots - document_kind / party_context / focus_issue /
# question are empty on all 60,603 seeds - so the teacher is asked to draft
# for "the party whose papers these are" against a judgment that already
# disposed of the matter. 4/20 accepts vs summarization's 13/20 (p=0.0095)
# and 66,666 tok per accepted row vs 18,028. Its 0.18 goes to summarization,
# the cheapest converter. Restore it when seeds carry the fields, or when the
# stream is retargeted as a downstream instrument (~14,225 eligible seeds).
SYNTHESIS_MIX = {
    "irac_analysis": 0.40,
    "statute_qa": 0.25,
    "drafting": 0.00,
    "summarization": 0.35,
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m pytest tests/test_build_tasks.py -v`

Expected: PASS. If another test asserts the old 0.18, update it to match — do not weaken the new assertion.

- [ ] **Step 5: Commit**

```bash
git add src/tuned/data/tasks.py tests/test_build_tasks.py
git commit -m "park drafting until its seeds carry the fields the prompt asks for"
```

---

### Task 4: Resolve footnote markers into the chunk that cites them

`extract.py:1515` hoists every footnote to one document-tail block and `segment.py:326` makes that block a single segment, so **96.2% of inline footnote markers point at text in a different chunk** and 29.9% of chunks carry an authority whose citation is physically elsewhere. The teacher is told it is in a closed world, then handed a fragment with its authorities amputated — and one grounding-1 rationale names this verbatim.

**Custody note, verified:** `chunk_id_for` hashes `object_key:start:end`. Appending footnote text without moving boundaries leaves `seed_id`s **bit-identical**, so the pre-registration survives. Do not change boundaries in this task — the trailing-heading fix does re-key seed_ids and is deliberately excluded.

**Before writing code, read** `src/tuned/data/segment.py:303-330` and `src/tuned/data/chunks.py:150-220` to see how segments are packed and how `chunk_id_for` derives ids.

**Files:**
- Modify: `src/tuned/data/segment.py` (footnote block handling, around line 326)
- Modify: `src/tuned/data/chunks.py` (chunk assembly, around line 160)
- Test: `tests/test_build_chunks.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: chunk text that carries the footnotes it references. `seed_id` derivation is unchanged — assert this.

- [ ] **Step 1: Write the failing test**

```python
def test_footnote_markers_resolve_into_their_citing_chunk():
    """A chunk that cites footnote 3 must contain footnote 3's text.

    extract.py hoists all footnotes to a document-tail block; 96.2% of
    inline markers (1,909/1,984) therefore point outside the chunk that
    cites them. The teacher is prompted with a closed world, so an amputated
    authority reads to the judge as invented.
    """
    doc = (
        "1. The appellant relies on the rule in Salomon.[^3]\n\n"
        "2. That contention must be rejected.\n\n"
        "[FOOTNOTES]\n"
        "[^3]: Salomon v A Salomon & Co Ltd [1897] AC 22.\n"
    )
    chunks = chunk_text_for_test(doc)
    citing = next(c for c in chunks if "[^3]" in c.text)
    assert "Salomon v A Salomon" in citing.text, (
        "chunk cites footnote 3 but does not carry its text"
    )


def test_footnote_resolution_does_not_move_chunk_boundaries():
    """seed_id is hash(object_key:start:end) — resolution must not re-key.

    A frozen control store holds a 60-pair pre-registration keyed to current
    seed_ids, and re-chunking would orphan 1,060 tasks / 1,396 generations /
    46 gold labels behind a swallowed IntegrityError.
    """
    doc = (
        "1. The appellant relies on the rule in Salomon.[^3]\n\n"
        "2. That contention must be rejected.\n\n"
        "[FOOTNOTES]\n"
        "[^3]: Salomon v A Salomon & Co Ltd [1897] AC 22.\n"
    )
    before = [(c.start, c.end) for c in chunk_text_for_test(doc, resolve_footnotes=False)]
    after = [(c.start, c.end) for c in chunk_text_for_test(doc, resolve_footnotes=True)]
    assert before == after
```

Add the helper alongside the existing fixtures in `tests/test_build_chunks.py`, matching how that file already builds chunks — read it first and reuse its existing construction rather than inventing a second path.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m pytest tests/test_build_chunks.py -k footnote -v`

Expected: FAIL — `chunk cites footnote 3 but does not carry its text`.

- [ ] **Step 3: Implement resolution**

Append the resolved footnotes to each chunk's rendered text, keyed off the markers actually present in that chunk, leaving `start`/`end` untouched:

```python
FOOTNOTE_MARKER = re.compile(r"\[\^([^\]]+)\]")


def resolve_footnotes(text: str, footnotes: dict[str, str]) -> str:
    """Append the footnotes this text cites, so the chunk is self-contained.

    Boundaries are NOT touched: seed_id hashes object_key:start:end, and a
    frozen pre-registration is keyed to the current ids.
    """
    cited = [k for k in dict.fromkeys(FOOTNOTE_MARKER.findall(text)) if k in footnotes]
    if not cited:
        return text
    lines = "\n".join(f"[^{k}]: {footnotes[k]}" for k in cited)
    return f"{text}\n\n[FOOTNOTES CITED ABOVE]\n{lines}\n"
```

Parse the document-tail `[FOOTNOTES]` block into the `footnotes` mapping where `segment.py` currently emits it as one segment, and call `resolve_footnotes` as each chunk's text is assembled in `chunks.py`. Bump `CHUNK_VERSION` so the manifest re-chunks on the next build.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m pytest tests/test_build_chunks.py -v`

Expected: PASS, including the boundary-stability test.

- [ ] **Step 5: Confirm the whole suite is green**

Run: `PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m pytest tests/ -q --basetemp=C:/Users/Anant/AppData/Local/Temp/pt-t4`

Expected: no regressions. The explicit `--basetemp` avoids the shared `pytest-current` symlink collision when two runs overlap.

- [ ] **Step 6: Commit**

```bash
git add src/tuned/data/segment.py src/tuned/data/chunks.py tests/test_build_chunks.py
git commit -m "a chunk that cites a footnote should carry it"
```

---

### Task 5: Export the calibration anchor set

Every accept rate this project has ever reported is model-agreeing-with-model. The 46 `gold_label` rows are model-generated (identical batch timestamp), so they calibrate agreement, not correctness. Without an external anchor, Tasks 1–2 cannot be shown to have improved anything rather than merely moved numbers.

**Minimum viable anchor: ~40 items, one axis, one lawyer-day.** Unit is the *generation*, not the judgement, so one read adjudicates both slots. Anchor on two **rubric-independent bits** rather than the 1–5 scale, so the labels stay valid across any future rewording:

1. Does the answer assert a **false** legal proposition?
2. Does it assert a **correct but unsupported** one?

Those two bits separate the 49% teacher-error mode from the correct-recall mode — the exact question every other decision in this plan depends on. Withhold the judge rationales from the labeller; supply the task instruction the judge never saw.

**Files:**
- Create: `scripts/export_calibration_set.py`
- Test: `tests/test_export_calibration_set.py`

**Interfaces:**
- Consumes: read-only access to any arm store.
- Produces: `data/calibration/anchor-set-<arm>.jsonl`, one object per generation with keys `gen_id`, `task_type`, `task_instruction`, `seed_text`, `answer`, `think`, and empty `asserts_false` / `asserts_unsupported` fields for a human to fill.

- [ ] **Step 1: Write the failing test**

```python
import json
from scripts.export_calibration_set import build_rows


def test_export_withholds_judge_rationales_and_leaves_labels_empty():
    """The labeller must not see what the judge said, or they will agree with it.

    Anchoring on two rubric-independent bits keeps the labels valid across
    rubric rewrites — the 1-5 bands have already been rewritten once.
    """
    rows = build_rows(FAKE_STORE, limit=40)
    assert len(rows) <= 40
    for row in rows:
        assert set(row) >= {
            "gen_id", "task_type", "task_instruction", "seed_text",
            "answer", "asserts_false", "asserts_unsupported",
        }
        assert row["asserts_false"] is None
        assert row["asserts_unsupported"] is None
        blob = json.dumps(row)
        assert "rationale" not in blob
        assert "grounding" not in blob
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m pytest tests/test_export_calibration_set.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.export_calibration_set'`.

- [ ] **Step 3: Implement the exporter**

```python
"""Export a human-labellable anchor set. READ-ONLY on every store it touches.

Selection is the whole failure population, not a sample: every generation
underlying a grounding<=3 judgement, plus the tiebreak-derived accepts (the
rows the saturated arbiter waved through), plus a few clean accepts as
control. Zero sampling error over the population that matters.
"""
import argparse, json, sqlite3
from pathlib import Path

SELECT_FAILING = """
select distinct g.gen_id from judgement j join generation g on g.gen_id = j.gen_id
where j.grounding <= 3
"""
SELECT_TIEBREAK = """
select distinct g.gen_id from judgement j join generation g on g.gen_id = j.gen_id
where j.judge_slot = 'tiebreak'
"""
SELECT_CONTROL = """
select distinct g.gen_id from judgement j join generation g on g.gen_id = j.gen_id
where j.grounding = 5 limit 3
"""


def build_rows(conn, *, limit=40):
    ids = []
    for sql in (SELECT_FAILING, SELECT_TIEBREAK, SELECT_CONTROL):
        for (gen_id,) in conn.execute(sql):
            if gen_id not in ids:
                ids.append(gen_id)
    rows = []
    for gen_id in ids[:limit]:
        gen = conn.execute(
            "select gen_id, task_id, think, answer from generation where gen_id=?",
            (gen_id,),
        ).fetchone()
        task = conn.execute(
            "select task_type, seed_id from task where task_id=?", (gen[1],)
        ).fetchone()
        seed = conn.execute(
            "select text from seed where seed_id=?", (task[1],)
        ).fetchone()
        rows.append({
            "gen_id": gen[0],
            "task_type": task[0],
            "task_instruction": task[0],
            "seed_text": seed[0] if seed else "",
            "think": gen[2],
            "answer": gen[3],
            "asserts_false": None,
            "asserts_unsupported": None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
    conn = sqlite3.connect(f"file:{Path(args.store).as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    rows = build_rows(conn, limit=args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m pytest tests/test_export_calibration_set.py -v`

Expected: PASS.

- [ ] **Step 5: Produce the actual anchor set**

```bash
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe scripts/export_calibration_set.py \
  --store data/build/exp_recovery/state/law_v1.sqlite3 \
  --out data/calibration/anchor-set-exp_recovery.jsonl --limit 40
```

Expected: `wrote 40 rows`. Confirm the file contains no `rationale` or `grounding` key: `grep -c rationale data/calibration/anchor-set-exp_recovery.jsonl` returns 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/export_calibration_set.py tests/test_export_calibration_set.py
git commit -m "the first thing in this pipeline that a human can check"
```

`data/calibration/` output stays untracked unless the operator decides otherwise — it contains corpus text.

---

### Task 6: Re-measure on a fresh arm

Only now is an accept rate worth computing. This arm re-runs the **same 60 pre-registered pairs** with the repaired rubric and the restored anchor, so the comparison against exp_recovery isolates the judging fix.

**Files:**
- Create: `configs/data_law_v1_exp_measure.yaml`
- Modify: `src/tuned/data/paths.py:17` (`ISOLATED_WORKDIR_SIBLINGS`)
- Modify: `tests/test_build_config.py:819` (the parametrized sibling test)

**Interfaces:**
- Consumes: the templates from Tasks 1–2 and the parked mix from Task 3.
- Produces: an accept rate comparable to exp_recovery's 41.7% under repaired judging.

- [ ] **Step 1: Add the new arm to the isolation allowlist**

`ISOLATED_WORKDIR_SIBLINGS` is a deliberate allowlist — `config.py:862-876` refuses any recovery-flavoured config that does not name one, which is what keeps an experiment off the live workdir. Extend it and its test:

```python
ISOLATED_WORKDIR_SIBLINGS = frozenset(
    {"exp_recovery", "exp_harmony", "exp_s1", "exp_measure"}
)
```

```python
@pytest.mark.parametrize(
    "workdir",
    [
        "data/build/exp_recovery",
        "data/build/exp_harmony",
        "data/build/exp_s1",
        "data/build/exp_measure",
    ],
)
def test_isolated_experiment_siblings_are_not_treated_as_live_control(
    tmp_path, workdir
):
```

- [ ] **Step 2: Run the config tests**

Run: `PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m pytest tests/test_build_config.py -v`

Expected: PASS, including the new parametrization.

- [ ] **Step 3: Create the arm config**

```bash
cp configs/data_law_v1_exp_s1.yaml configs/data_law_v1_exp_measure.yaml
```

Then edit exactly three things and nothing else — record the diff in the commit message:
- `build.workdir` → `data/build/exp_measure`
- both gpt-5 `usd_cap` values → `0.0` (this arm measures the *free* fleet; a lenient paid judge in slot B is precisely the confound being removed)
- the header comment → describe the judging-repair measurement

- [ ] **Step 4: Bootstrap the same 60 pairs**

```bash
cp data/build/exp_s1/bootstrap.py data/build/exp_measure/bootstrap.py
```

Edit `CONFIG` to `configs/data_law_v1_exp_measure.yaml` and `ARM` to `exp_measure`. The manifest path stays `configs/pre-registration/law-v1-recovery-cohort-manifest.json`. Run it, then verify 0 missing / 0 extra against the manifest's 60 `seed_id`s before generating.

The bootstrap opens the live store through a **separate read-only connection and never ATTACHes it** — `Store.open` connects without `uri=True`, so a `mode=ro` URI would be read as a literal filename and would open the control store read-write. Do not "simplify" this.

- [ ] **Step 5: Generate and judge**

```bash
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tuned.data.generate --config configs/data_law_v1_exp_measure.yaml --n-workers 4 --max-batches 40
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tuned.data.judge --config configs/data_law_v1_exp_measure.yaml --n-workers 3 --max-batches 40
```

Expected: 60 generations; judging entirely on the free fleet. Confirm `select * from budget_ledger where provider='openai'` returns **no rows**.

- [ ] **Step 6: Report the comparison**

Compute, against exp_recovery on the identical pairs: accept rate, per-axis means, the grounding score distribution (the old `1/41/7/19/33` should flatten if Task 2 worked), and the tiebreak's accept rate (18/18 should no longer hold if Task 1 worked). Write the result to `docs/reports/2026-08-24-judge-calibration.md`.

- [ ] **Step 7: Commit**

```bash
git add configs/data_law_v1_exp_measure.yaml src/tuned/data/paths.py tests/test_build_config.py docs/reports/2026-08-24-judge-calibration.md
git commit -m "measure the judging repair on the pairs it was pre-registered against"
```

---

## Out of Scope — Filed, Not Forgotten

These are real and measured, but each needs its own plan and, in two cases, an operator decision.

1. **Teacher quality — the dominant cap.** ~50% of grounding failures are the teacher genuinely wrong, confirmed by two independent methods. No fix in this plan touches it. Options: a stronger teacher, or a generation-time self-check. **This is where the next month goes.** Note that `MAX_ATTEMPTS = 3` already gives best-of-3, and within the s1 recipe higher k makes cost-per-clean-row *worse* — so resampling is not the answer.
2. **Statute text acquisition.** Bounded and cheap: ~150 sections cover 73.2% of citations, old-code is 99.91% of mentions, and Zenodo 5088102 has 858 Central Acts as CC-BY-4.0 JSON. The `{section_text}` prompt slot already exists. This also unblocks `statute_qa`, which has 270 tasks and 0 eligible seeds.
3. **Case-citation existence index.** All 1,396 `citations` gate rows read `novel_skipped: "no-index"` — the existence check has never actually run. Building it is a prerequisite for ever telling a judge "this citation was verified to exist".
4. **The role-aware segmenter.** `roles_infer.py --worker` exists and is tested; `opennyai`/`spacy` are absent from the venv but the py≥3.13 blocker in the docstring does not bite (venv is 3.12.13). **Blocked on custody:** re-chunking re-keys every `seed_id`, which needs a fresh store and re-labelled gold. Operator decision.
5. **Re-gating the control store.** `prompt_echo` has zero rows in the control `gate_result` table, so `eval_matched` returns `inconclusive / missing-gate-data` for *any* treatment arm. Fixing it means writing to the frozen store. Operator decision.
6. **Drafting retarget.** ~14,225 eligible seeds against a need of ~810. Task 3 parks it; this un-parks it properly.
7. **Trailing-heading carry-forward.** 12.0% of chunks end on an orphaned heading. Excluded from Task 4 because it *does* re-key `seed_id`s.

---

## Self-Review

**Spec coverage.** Judge saturation → Task 1. Rubric collision → Task 2. Unwinnable drafting → Task 3. Severed authorities → Task 4. No external anchor → Task 5. Untrustworthy accept rate → Task 6. Teacher quality, statute acquisition, segmenter, control-store re-gating → explicitly out of scope with reasons.

**Placeholder scan.** Every code step carries real code. Task 4 asks the implementer to read two specific line ranges before writing, because the surrounding assembly must be matched rather than guessed — the tests pin the contract either way.

**Type consistency.** `build_rows(conn, *, limit)` is used identically in test and implementation. `resolve_footnotes(text, footnotes) -> str` is used once. `SYNTHESIS_MIX` keys are unchanged; only weights move. `ISOLATED_WORKDIR_SIBLINGS` stays a `frozenset[str]` of bare directory names, matching how `is_live_control_workdir` consumes it.

**Ordering.** Tasks 1–3 are independent. Task 4 is independent. Task 5 reads existing stores. Task 6 depends on 1, 2 and 3 being merged. Do not reorder 6 earlier — it is the measurement, and measuring before the instrument is fixed is what produced the withdrawn 56.7%.
