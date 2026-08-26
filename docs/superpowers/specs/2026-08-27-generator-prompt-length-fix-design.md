# Generator prompt length fix — design

**Date:** 2026-08-27
**Status:** approved
**Branch:** `worktree-law-v1-data-pipeline`

## Problem

The 2026-08-27 deepseek validation wave gated out 79 of 99 generations. The three
largest gate losses are not three problems. They are one problem measured three ways:

| gate | fail | median trace, FAIL | median trace, PASS |
|---|---|---|---|
| `irac_placement` | 61/99 | 14,320 ch | 6,576 ch |
| `length_band` | 50/99 | 17,158 ch | 6,257 ch |
| `verbatim_overlap` | 46/99 | 16,200 ch | 7,257 ch |
| `prompt_echo` | 13/99 | 14,005 ch | 9,993 ch |
| `banned_meta` | 14/99 | 13,137 ch | 10,235 ch |

Every gate that fails, fails on long traces. A trace that runs long drifts: it
restates the source (`verbatim_overlap`), it organises itself under headings
(`irac_placement`), and it breaches the band (`length_band`).

The root cause is length, and length alone:

- trace: prompts ask **450–700 words**; deepseek produced median **1,727**, p90 **3,773**
- answer: prompts ask **250–450 words**; deepseek produced median **836**

**The answers are fine.** 60 of the 61 `irac_placement` failures have a
well-formed answer with all four headings; `missing_in_answer` is empty. Zero
answers are malformed. Only the trace misbehaves.

### Why this is not a "clarify the prompt" fix

The prompts already state the rules, explicitly:

- *"Those headings belong to the model answer and never inside your reasoning,
  which runs as continuous prose and never opens a line with one of those four words."*
- *"do not carry sentences over from the materials into your thinking"*

Both are ignored. The retry nudge in `generate._REPAIR_HINTS` repeats them
verbatim and makes the gate **worse** — `irac_placement` by attempt: 42% → 41% → 28%.

Clarity is not the defect. The defect is that each prompt asks for length in two
places while also capping it:

1. *"your own working beforehand runs as long as the problem deserves"*
2. *"Work the point through fully — 450 to 700 words of deliberation is normal
   for a matter of any substance."*

These conflict with the stated word bounds. gpt-oss resolved the conflict toward
the bound; deepseek resolves it toward the permission. `reasoning_effort` cannot
resolve it either — `low` is the enum floor and `disabled` yields zero trace,
which violates the >=80% reasoning-trace requirement. The prompt is the only lever left.

## The fix

Remove the permission to run long, and make the bounds bounds. Surgical: nothing
else in any prompt changes.

Per file:

1. Delete the permissive clause (*"runs as long as the problem deserves"* /
   *"takes as long as the decision deserves"*).
2. **Keep the literal substring `450 to 700 words of deliberation`. Replace only
   the tail** — *"is normal for a matter of any substance"* — with a hard ceiling.

   This sentence is not surplus prose, and an earlier draft of this spec was
   wrong to call for its deletion. `tests/test_build_prompts.py` pins it as
   `REASONING_FLOOR_CLAUSE`, and its comment records why it exists:

   > RAISED 2026-08-18 off "a few hundred words". […] Measured: every one of the
   > 20 attempt-1 gpt-oss `length_band` violations was `think<think_min` […]
   > 450-700 words is ~2,900-4,500 chars, ~725-1,125 est tokens: clear of
   > think_min 500 and clear of think_max 3,000.

   It is the mechanism that holds traces **above** the floor, added four days ago
   to fix the mirror image of today's problem. Deleting it would regress that fix
   for gpt-oss while chasing deepseek. The band values were always right; only
   the permission around them is wrong.

   The literal substring must also survive because two test constants match on
   it — `_PACKET_MARKERS` in `test_build_prompts.py` and `HIDDEN_REASONING_TARGETS`
   in `test_build_harmony.py` — where it distinguishes a base template from its
   Harmony overlay (the overlay strips the reasoning-length target by design).
   Preserving it keeps the edit's blast radius to one constant instead of three.
3. Restate the existing word ranges as **bands with a hard ceiling** — for both
   the trace and the answer. Do not invent new numbers: 250–450 for the answer
   and 450–700 for the trace are the numbers already in the files.

   **The floor is load-bearing and must survive the edit.** `length_band` enforces
   `think_min = 500` tokens, and the arm is already breaching it 3/99 with the
   shortest trace at 470 tokens — 30 below the floor. An edit that says only
   "shorter is better" converts a ceiling failure into a floor failure. Each file
   must keep a lower bound as explicit as its upper one.
4. Leave the self-verification paragraph intact. It is a length amplifier, but
   `self_verification` passes 86/99 today and cutting it trades one gate for another.
5. **Spend no net words.** `test_generator_user_block_is_the_intended_size` asserts
   `250 <= words <= 500` over each template's whole user block, and two files are
   nearly full: `gen_drafting_v1` has **5** words of headroom and `gen_transition_v2`
   has **4**. The ceiling was already raised once (470 → 500) for this same reason.
   The edit deletes a clause and rewrites a tail, so it must come out word-neutral
   or shorter on every file — and on those two it must be checked, not assumed.
6. **The permissive clause has at least four surface forms and cannot be matched
   on one word.** Only 5 of 14 files say *"deserves"*; the others say *"as long as
   it needs to"*, *"as long as the point requires"*, *"as long as the case
   requires"*, and so on. A find-and-replace keyed on `deserves` silently misses 9
   of 14 files and would leave most of the corpus unfixed while every test passed.
   Each file gets read.

### One production constant tracks this wording

`gates.py` `INSTRUCTION_ECHO_SPANS` contains the literal
`"450 to 700 words of deliberation is normal"`. `check_prompt_echo` matches it
against **generated traces** to catch a model that parrots its own instructions
back as reasoning — a gate that fails 13/99 on the arm today. Change the prompt
tail without changing this span and the gate keeps hunting a sentence no prompt
issues any more: it would not fail loudly, it would quietly stop detecting.

No test exercises this span, so nothing in CI will catch it. Update it with the
edit. This is functional, not hygiene.

**Scope: all 14 generator prompts** in `src/tuned/data/prompts/` —
`gen_irac_analysis_v1..v4`, `gen_summarization_v1..v2`, `gen_statute_qa_v1..v4`,
`gen_transition_v1..v2`, `gen_drafting_v1..v2`. All 14 carry clause 2; 13 carry
clause 1 (`gen_irac_analysis_v2` words it differently and must be read, not
pattern-matched).

### Two constraints that decide the shape of the edit

**Edit in place. Do not add a v5.** These files are not revisions — they are
paraphrase variants. `prompt_registry.pick_variant(task_type, seed_id, sample_ix)`
assigns one per seed by hashing modulo the variant count. Adding a fifth
`gen_irac_analysis` re-maps every seed's assignment across the entire corpus, which
the code comments already call out as a silent-remapping hazard. In-place edits
keep every assignment stable.

**Do not homogenise the wording.** The 14 files are deliberately distinct
paraphrases; that diversity is the reason `pick_variant` exists. The same
*semantic* change goes into each file, expressed in that file's own voice and
register. Fourteen identical sentences would destroy what the variant mechanism
is for.

Worked example, `gen_irac_analysis_v4` — before:

> Roughly 250 to 450 words is the length you would expect of a strong candidate;
> your own working beforehand runs as long as the problem deserves and is never a
> retelling of the materials. Work the point through fully — 450 to 700 words of
> deliberation is normal for a matter of any substance.

after:

> Roughly 250 to 450 words is the length you would expect of a strong candidate,
> and you do not exceed it. Your own working beforehand is never a retelling of the
> materials. Work the point through fully — 450 to 700 words of deliberation, and
> 700 is a ceiling you do not cross.

Note what survives: the sentence, the band, and the literal `450 to 700 words of
deliberation`. What goes is the permission on either side of it. The other 13 make
the same move in their own words.

### Re-pinning the template shas

Every template's sha is pinned in the test suite, deliberately, so that any edit
forces a look. `tests/test_build_prompts.py` holds `EXPECTED_SHAS` for the 14 base
templates, and `test_template_sha_is_pinned` asserts each one. Editing the prompts
without re-pinning turns 14 green tests red.

Re-pin by reading the new sha from `prompt_registry.load(prompt_id).sha` after the
edit — never by copying it out of a test failure message, which is how a wrong
value gets laundered into a pin. `test_sha_is_twelve_hex_of_the_raw_file` and
`test_templates_are_lf_only` both guard the pin, so **the files must be written
LF-only**; a CRLF write changes every hash. This repo has already lost one task to
`Path.write_text` converting a file to CRLF on Windows.

`prompt_registry.load` hashes each file's bytes into `Template.sha`, so every
edited template gets a new sha automatically. No provenance bookkeeping is needed.

## Validation — a paired A/B

The v4 numbers above are the control (n=99, banked in the existing arm store). The
treatment re-runs the **same seeds** through the edited prompts.

- New isolated workdir `data/build/exp_prompt_v5`, added to
  `paths.ISOLATED_WORKDIR_SIBLINGS` with a test, exactly as `exp_deepseek` was.
- Config `configs/data_law_v1_exp_prompt_v5.yaml`: a copy of
  `configs/data_law_v1_exp_deepseek.yaml` with `build.workdir` changed and a header
  stating its purpose. Both cost fences carry over unchanged — the single-ref
  `routing.generator: [bai/deepseek-v4-flash]` and `usd_cap: 0.0` **with prices**
  on both gpt-5 models (a bare `usd_cap: 0.0` blocks nothing: `_usd_per_1m`
  returns `0.0` for a missing price, so `0 + 0 > 0.0` is False).
- Seed with `scripts/seed_exp_store.py --seed 3407 --per-source 200`, the same
  arguments as the control, so the same seeds land. Plan the same three arms with
  the same counts: `sc`=13, `predex`=14, `tathya`=13.
- Because `pick_variant` keys on `seed_id`, identical seeds draw identical
  variants. The only variable between control and treatment is the prompt text.
- Generate only. **No judging** — the question is whether the gates pass, and
  judging costs the bulk of the wall clock.

### Pre-registered, before the run

| # | measurement | pass line |
|---|---|---|
| 1 | median trace words | **< 900** (control: 1,727) |
| 2 | `length_band` pass rate | **> 70%** (control: 49%) |
| 3 | `irac_placement` pass rate | **> 60%** (control: 38%) |
| 4 | `verbatim_overlap` pass rate | **> 70%** (control: 54%) |
| 5 | `self_verification` pass rate | **>= 80%** — must not regress (control: 87%) |
| 6 | answer well-formedness | `missing_in_answer` stays empty; >= 95% |
| 7 | `think<think_min` breaches | **<= 5%** — must not rise (control: 3/99) |

Measurements 1–4 are the fix working. **5 and 7 are the guards**: if shortening
the trace kills self-verification, or converts a ceiling breach into a floor
breach, the fix has moved the failure rather than removed it.

If 1 passes but 2–4 do not, length was not the cause and the diagnosis in this
spec is wrong — that result is reported, not worked around.

## Also required: unblock the merge

Independent of the prompt work, the branch carries **42 failing tests** from the
uncommitted b.ai integration. They are stale expectations, not broken code: the
config now routes `bai/deepseek-v4-flash` first and the tests still assert
`cerebras/gpt-oss-120b`. Confirmed pre-existing — the failure set was byte-identical
before this branch's first commit.

- `tests/test_build_providers.py` — 8 failures, generator-order assertions
- `tests/test_build_generate.py` — 20 failures
- `tests/test_build_judge.py` — 11 failures
- `tests/test_build_eval_matched.py` — 1 failure

Re-baseline each to the `bai`-first routing and commit the integration
(`configs/data_law_v1.yaml`, `src/tuned/data/providers.py`, and the test files)
alongside them. Where a test asserts a *specific* provider only incidentally,
prefer asserting the invariant over the literal — the wave already produced one
Critical from a test pinned to a config that was not committed.

## Out of scope

- **Raising `think_max`** or demoting deepseek to a minority source. Those are the
  two routes this fix is the alternative to; if validation fails they come back.
- **Judging the treatment arm.** Gate pass rates answer the question. Accept rate
  at n~20 was already recorded as a null result.
- **Editing `prompts_harmony/`.** That overlay is gpt-oss's Harmony format and is
  not on the deepseek path. Leaving it alone is a decision, not an oversight, and
  it is the right one on the merits: the overlay already strips the whole
  reasoning-length packet by design, and gpt-oss's measured failure mode is
  falling *below* `think_min`, not overrunning the ceiling. Removing its
  permission to run long would push it further into the failure this build spent
  2026-08-18 fixing.

  **The cost is real and must be recorded.** After this edit the base templates
  say "ceiling" and the overlay still says "as long as it needs", and **no test
  detects the divergence.** The guard usually credited with catching it —
  `test_generator_overlays_differ_from_their_base_in_exactly_two_lines` — counts
  *how many lines differ*, not how they differ, and the clause being edited already
  sits on one of those two lines. `_PACKET_MARKERS` misses it too. The divergence
  is therefore deliberate, documented here, and invisible to CI; anyone later
  syncing overlay to base must read this paragraph first.
- **Touching the live control store.** Read-only, always.

## Acceptance criteria

1. All 14 generator prompts edited in place; no v5 file created; each file's
   wording remains distinct from the other 13.
2. `prompt_registry` loads all 14 and `all_ids()` is unchanged in length.
3. The full existing prompt/registry test suite stays green.
4. `exp_prompt_v5` is an isolated workdir with a test; the arm runs generate-only
   to its batch cap; every generation is deepseek; `$0` on its ledger.
5. Each pre-registered number above is reported beside its measured value in
   `docs/reports/2026-08-27-generator-prompt-length-fix.md`.
6. The 42 pre-existing failures are re-baselined and the b.ai integration is
   committed; full suite green.
7. `data/build/state/law_v1.sqlite3` has the same size and mtime after as before.
