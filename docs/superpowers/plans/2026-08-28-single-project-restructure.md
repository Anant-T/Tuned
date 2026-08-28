# Single-Project Restructure + Free-Fleet Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the recalibrated deepseek pipeline on 2–5 live examples (user gate for full generation), then collapse the worktree split into ONE project with `training/` and `data/` folders, purging everything gpt-oss-era that no longer serves — paid refs, harmony machinery, experiment configs, dead tests — with all retired knowledge archived in a single `prev_rep.md`.

**Architecture:** Phase A runs a small live batch end-to-end (generate → gates → judge) in the existing worktree, because the pipeline there is qualified and keyed. Phase B commits the worktree. Phase C merges the pipeline branch into `main`, reorganizes into `training/` + `data/`, deletes the retired surface, and removes the worktree. `prev_rep.md` is written by synthesis from the dated reports BEFORE they are deleted.

**Tech Stack:** Python 3.x (worktree `.venv`), pytest, sqlite3 (LIVE store — pipeline commands may write; ad-hoc scripts stay `mode=ro`), git worktrees/merge.

## Global Constraints

- **Operator directives (2026-08-28):** NO OpenRouter, NO paid refs — the fleet is bai/deepseek-v4-flash (generator), groq/qwen3.6-27b, cerebras/gemma-4-31b, groq/openai/gpt-oss-20b (judges/tiebreak). mistral-large keeps its tiebreak seat (free tier, already routed).
- **Full generation is GATED on the user approving the 2–5 sample outputs.** Do not start a full wave.
- NO Claude/Anthropic attribution anywhere: no co-author trailers, no "Generated with", no watermarks (global CLAUDE.md hard rule).
- Never print `.env` contents. Keys load via `providers.load_dotenv_keys` from the repo-root `.env`.
- `data/build/state/law_v1.sqlite3` is written ONLY by pipeline commands (`tuned.data.generate/judge/tasks`); ad-hoc inspection opens `mode=ro`.
- Inside the pipeline tree always invoke `./.venv/Scripts/python.exe` with `PYTHONIOENCODING=utf-8`.
- Suite must be green (currently 3,582 passed / 19 skipped) after every commit-worthy state.
- Templates in `prompts/` stay byte-untouched (no-edit-without-arm rule); cleanup never touches `prompts/gen_*.md`.
- `.git` is a FILE in the linked worktree — never treat it as a directory.

---

### Task A1: Live validation batch — generation

**Files:** none modified (pipeline run). Working dir: `.claude/worktrees/law-v1-data-pipeline`.

- [ ] **Step 1:** Confirm keys + preflight: `PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tuned.data.generate --config configs/data_law_v1.yaml --stream synthesis --n-workers 5 --max-batches 1` — the CLI prints `loaded N key(s) from .env` and a preflight; ONE batch claims ≤5 of the 127 pending synthesis tasks.
- [ ] **Step 2:** Read the batch line (`batch 1: claimed=… gen-ok=… clean=…`). Expected: claimed ~5, gen-ok ≥4 (bai rpm 8; ~33s/call), some clean at the recalibrated gates. If claimed=0, the pending pool is stale-guarded — run `./.venv/Scripts/python.exe -m tuned.data.tasks --reopen stale_prompt` and retry Step 1 once.
- [ ] **Step 3:** If clean rows < 2 after batch 1, run ONE more batch (same command). Hard stop at 2 batches — this is a sample, not a wave.

### Task A2: Live validation batch — judging

- [ ] **Step 1:** `PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tuned.data.judge --config configs/data_law_v1.yaml --stream synthesis --n-workers 2 --max-batches 2` — drains the `judging` queue (43 pre-existing + new). Slots land on qwen (A) and gemma (B) — deepseek is family-excluded on its own rows.
- [ ] **Step 2:** Verify judgements landed: read-only query of `judgement` rows joined to today's gen_ids; note accepts/rejects and scores.

### Task A3: Extract the review samples

**Files:** Create: `sample_review.md` (repo root of worktree; temporary, shown to user, not committed).

- [ ] **Step 1:** Read-only script: pull the 2–5 newest deepseek generations with their gate rows + judge scores; write think-excerpt (~600 chars), full answer, gate verdicts, judge axes to `sample_review.md`.
- [ ] **Step 2:** Present the samples in the final user message with the explicit question: approve → full generation.

### Task B1: Commit the worktree state

- [ ] **Step 1:** In the worktree: `git add -A` then commit in logical chunks if practical (recalibration src+tests+configs; reports; plan). Message style: repo convention, e.g. `refit: deepseek recalibration - verbatim 500, think_max 4000, output budget 16384`. NO attribution trailers.
- [ ] **Step 2:** `git log --oneline -3` to confirm; suite already green (3,582) — do not rerun here.

### Task C1: Merge the pipeline branch into main

**Files:** main repo `c:\Users\Anant\Desktop\projects\tuned`.

- [ ] **Step 1:** `git -C <main> fetch . && git -C <main> merge <pipeline-branch>` (find branch name via `git -C <worktree> branch --show-current`). Resolve conflicts favoring the pipeline branch for pipeline files, main for training docs.
- [ ] **Step 2:** Copy non-git assets the worktree holds: `.env` (if main lacks the keys) and `data/build/**` (corpora + LIVE store) — `robocopy`/`cp -r` to the main checkout, preserving layout. Verify LIVE store fingerprint unchanged (byte size + mtime).
- [ ] **Step 3:** `git worktree remove` the pipeline worktree (after confirming nothing uncommitted remains: `git -C <worktree> status --short` is empty).

### Task C2: Two-folder layout — `training/` and `data/`

**Files:**
- Move: training assets → `training/` (configs/law_v1_8b_ddp.yaml + smoke/other train yamls → `training/configs/`; `scripts/` train-lane scripts; notebooks; training docs/reports).
- Move: `configs/data_law_v1.yaml` → `data/configs/data_law_v1.yaml`; `data/build` stays at `data/build`.
- Keep: `src/tuned/` single shared package (data imports `tuned.train.config` — package surgery is out of scope; the folders organize ASSETS, the package already splits `train`/`data`).
- Modify: every default/reference to the old config paths.

**Interfaces:** Produces the path contract later tasks rely on: `data/configs/data_law_v1.yaml` (pipeline CLIs' default `--config`), `training/configs/law_v1_8b_ddp.yaml` (`build.train_config` value inside the data config).

- [ ] **Step 1:** `git mv` the files per the map above.
- [ ] **Step 2:** Update path literals: grep `configs/data_law_v1` and `configs/law_v1_8b_ddp` across `src/ tests/ scripts/ docs/ *.md` and update every live reference (CLI defaults in generate/judge/tasks/verify/assemble/split/stats/push/difficulty/calibrate mains; tests' `DATA_CONFIG`/config fixtures; `build.train_config:` inside the data config; train-lane scripts). Dated historical mentions inside comments/reports stay as history.
- [ ] **Step 3:** Full suite: `PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m pytest tests/ -q`. Expected: same green count. Fix path fallout until green.
- [ ] **Step 4:** Commit: `restructure: single project - training/ and data/ folders`.

### Task C3: Purge the retired surface (free fleet only)

**Files:**
- Delete: `src/tuned/data/harmony.py`, `src/tuned/data/prompts_harmony/` (drafting genre-form text archived in prev_rep.md first), harmony flags in `config.py` (`harmony_prefill/harmony_completions/harmony_s1_continue` + guards) and their branches in `generate.py`/`providers.py` (HARMONY_STOP payload path), harmony tests.
- Delete: `openai` provider block (gpt-5-mini/nano) and `lightning` provider block from the data config; their refs from `routing.judge`/`routing.tiebreak`; their pinning tests (spend-fence tests that name openai refs get retargeted or deleted — the `_provider_usd_cap` CODE stays, it is provider-generic).
- Delete: `configs/data_law_v1_exp_*.yaml` (all 19) + the arm-fence tests that load them; `rows_dump.json`.
- Move (disk only, gitignored): `data/build/exp_*` stores → `data/build/archive/`.

- [ ] **Step 1:** Write prev_rep.md content FIRST (Task C4 runs before deletions of docs/reports; the config blocks being deleted here are quoted into prev_rep.md in the same commit).
- [ ] **Step 2:** Delete per the map; grep `harmony|lightning|gpt-5|openai/gpt-5|exp_` over `src/ tests/ configs/` until the only survivors are dated comments that narrate history.
- [ ] **Step 3:** Full suite green; fix fallout (routing-list pins, blocking-key tests, preflight tests that expected openai keys).
- [ ] **Step 4:** Commit: `purge: free-fleet only - drop paid refs, harmony lane, experiment configs`.

### Task C4: prev_rep.md — the single archive

**Files:** Create: `prev_rep.md` (repo root). Delete after archiving: `docs/reports/*.md`, superseded `docs/superpowers/plans/*.md` (keep THIS plan), stale root-level notes.

- [ ] **Step 1:** Synthesize (subagent, opus): for every dated report — the finding, the numbers that still bind, the verdict. Sections: (1) training-lane record (from 2026-08-08 project record + perf audit), (2) data-pipeline campaign history (pilot → judge fleet → deepseek qualification → prompt campaigns → F2 → recalibration), (3) retired config blocks VERBATIM (openai/lightning providers, exp_* one-line purposes, harmony drafting template text), (4) closed questions (never revisit: rsLoRA, adapter scale, effort ladder, cap arm, packing…), (5) still-open items carried forward (drafting unpark, statute text, transition audit, gold 46→180, judging throughput arithmetic).
- [ ] **Step 2:** Verify every deleted file's load-bearing conclusion appears in prev_rep.md (spot-check 5 reports against it), then `git rm` the archived files.
- [ ] **Step 3:** Commit: `archive: prev_rep.md consolidates retired reports and configs`.

### Task C5: Post-restructure verification

- [ ] **Step 1:** Full suite in the final layout — green.
- [ ] **Step 2:** Preflight smoke: `python -m tuned.data.generate --config data/configs/data_law_v1.yaml --stream synthesis --n-workers 1 --max-batches 0` equivalent (or tasks --help path check) proves configs resolve from the new paths and keys load.
- [ ] **Step 3:** `git worktree list` shows only the main checkout; `git status` clean; memory + MEMORY.md updated (worktree paths in memories are stale after this — rewrite the law-v1 memory pointers).

### Deliberately out of scope (named so nobody "finishes" them by accident)

- Full generation/synthesis wave — user gate.
- Judging-throughput redesign (free-fleet caps: qwen ~30 calls/day tpd; gemma ~1,100 calls on $4.63) — an operator scheduling fact, not a code fix.
- Drafting unpark, statute text acquisition, transition planning, gold labels — operator queue, carried in prev_rep.md §5.
- Any edit to `prompts/gen_*.md`.
