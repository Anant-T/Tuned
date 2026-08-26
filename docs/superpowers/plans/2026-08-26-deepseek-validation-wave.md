# DeepSeek Validation Wave Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run ~40 synthesis tasks through `bai/deepseek-v4-flash` in an isolated experiment arm, judge them with the free fleet, and write a report answering five pre-registered questions before the corpus build scales onto that generator.

**Architecture:** Follow the repo's `exp_*` arm pattern: a new isolated workdir + SQLite store, a config copied from the live yaml with two cost fences, a reusable seeding script that copies seeds out of the live store read-only, then the existing `tasks` → `generate` → `judge` CLIs, then a measurement script that writes the report. Three code tasks (each with its own test), one operator run, one report.

**Tech Stack:** Python 3.12, pytest, PyYAML, sqlite3, `tuned.data.*` (store, config, paths, tasks), httpx via `providers.ChatClient`; run from the `law-v1-data-pipeline` worktree with its own `.venv` and `.env`.

**Spec:** `docs/superpowers/specs/2026-08-26-deepseek-validation-wave-design.md`

## Global Constraints

- **Work in the worktree:** `C:\Users\Anant\Desktop\projects\tuned\.claude\worktrees\law-v1-data-pipeline`, branch `worktree-law-v1-data-pipeline`. Its own venv: `./.venv/Scripts/python.exe`. Its own `.env` (carries `BAI_API_KEY`).
- **Never open the live control store for write.** `data/build/state/law_v1.sqlite3` is read with `sqlite3.connect("file:...?mode=ro", uri=True)` only. Acceptance: its size and mtime are identical before and after the run.
- **Workdir:** `data/build/exp_deepseek`. Config: `configs/data_law_v1_exp_deepseek.yaml`.
- **Generator list is exactly** `[bai/deepseek-v4-flash]`. No cerebras, no lightning.
- **OpenAI is fenced to zero spend:** `usd_cap: 0.0, usd_per_1m_prompt: 0.25, usd_per_1m_completion: 2.0` on BOTH gpt-5 models' `limits`. The price is REQUIRED — `generate._usd_per_1m` returns 0.0 for a missing price, so `usd_cap: 0.0` alone computes `0 + 0 > 0.0 == False` and does NOT block.
- **No Harmony flags in the arm config:** none of `harmony_completions`, `harmony_prefill`, `harmony_s1_continue`, `prompt_overlay`, `require_pretreatment_manifest`, `pretreatment_manifest`.
- **Seed budget** is `tasks.seed_token_budget(cfg)` = `cfg.max_seq_length - 3500` = **4692** on this config. Never hard-code 4692 in code; tests may assert it.
- **Pre-registered pass lines (spec table):** ≥ 90% of b.ai calls return content; 0 planned tasks over the seed budget with ≥ 1 oversize seed in the store; every `generation.model == 'deepseek-v4-flash'`; `$0` on `budget_ledger`. Others are reported, not gated.
- **No AI-assistant attribution** in commits, files, or comments. No `Co-Authored-By` trailer.
- **Pytest invocation:** `./.venv/Scripts/python.exe -m pytest <file> -q -p no:cacheprovider --basetemp=<scratch>` (the default basetemp hits a Windows permission error). 42 tests in `test_build_generate.py`, `test_build_judge.py`, `test_build_providers.py`, `test_build_eval_matched.py` are **pre-existing failures** from the uncommitted b.ai routing work; they are out of scope and must not be "fixed" here. Run only the files each task names.
- **Do not `git add -A`.** The worktree carries uncommitted changes in `configs/data_law_v1.yaml`, `src/tuned/data/providers.py`, `src/tuned/data/paths.py`, `tests/test_build_config.py`, `tests/test_build_providers.py`, `docs/superpowers/plans/2026-08-24-...md`, and an untracked `configs/data_law_v1_exp_measure.yaml`. Stage only the files each task names. Task 1 edits `paths.py` and `tests/test_build_config.py`, which already have uncommitted hunks — commit those files whole (the existing hunks are the `exp_measure` sibling, which belongs with this change).

---

## File Structure

| path | responsibility |
|---|---|
| `src/tuned/data/paths.py` (modify, 1 line) | declares `exp_deepseek` an isolated sibling so write guards accept it |
| `tests/test_build_config.py` (modify, append) | isolation test for the sibling; load/fence tests for the arm config |
| `configs/data_law_v1_exp_deepseek.yaml` (create) | the arm: live config + workdir + two fences + header |
| `scripts/seed_exp_store.py` (create) | copies `source` + a per-source seed sample from a read-only live DB into an arm store; refuses live workdirs; idempotent |
| `tests/test_seed_exp_store.py` (create) | seeding behaviour on a synthetic "live" DB in tmp |
| `data/build/exp_deepseek/out/report_wave.py` (create, gitignored — under `data/`) | measurement script: reads the arm store + live store (ro), writes the report |
| `docs/reports/2026-08-26-deepseek-validation-wave.md` (create) | the deliverable |

---

### Task 1: Declare `exp_deepseek` an isolated workdir

**Files:**
- Modify: `src/tuned/data/paths.py:17-19` (the `ISOLATED_WORKDIR_SIBLINGS` frozenset)
- Test: `tests/test_build_config.py` (append)

**Interfaces:**
- Consumes: `tuned.data.paths.is_live_control_workdir(workdir) -> bool` (existing).
- Produces: `is_live_control_workdir("data/build/exp_deepseek") is False`. Tasks 2–4 depend on this; without it `load_build_config` and every write guard treat the arm as the live control.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_config.py` (the file already imports `pytest`, `Path`, `load_build_config`, and defines `_base_doc()` / `_write(tmp_path, doc)` helpers — reuse them):

```python
def test_the_deepseek_arm_is_an_isolated_workdir(tmp_path):
    """data/build/exp_deepseek is an experiment sibling, not the live control.

    The one-line fence that makes everything else in the deepseek arm
    possible: is_live_control_workdir is what load_build_config and the
    write guards consult, and an unlisted name under data/build reads as
    the frozen control.
    """
    from tuned.data.paths import is_live_control_workdir

    assert is_live_control_workdir("data/build/exp_deepseek") is False
    assert is_live_control_workdir("data/build") is True
    # A recovery-capable doc on the new sibling loads; on the live root it
    # is refused. Same shape as the exp_recovery tests above.
    doc = _base_doc()
    doc["build"]["workdir"] = "data/build/exp_deepseek"
    doc["build"]["harmony_prefill"] = "I start from the facts. "
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_deepseek"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_build_config.py::test_the_deepseek_arm_is_an_isolated_workdir -q -p no:cacheprovider --basetemp=data/build/exp_deepseek/pt`
Expected: FAIL — `assert True is False` on the first assertion.

- [ ] **Step 3: Add the sibling**

In `src/tuned/data/paths.py`, change:

```python
ISOLATED_WORKDIR_SIBLINGS = frozenset(
    {"exp_recovery", "exp_harmony", "exp_s1", "exp_measure"}
)
```

to:

```python
ISOLATED_WORKDIR_SIBLINGS = frozenset(
    {"exp_recovery", "exp_harmony", "exp_s1", "exp_measure", "exp_deepseek"}
)
```

- [ ] **Step 4: Run the config test file**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_build_config.py -q -p no:cacheprovider --basetemp=data/build/exp_deepseek/pt`
Expected: all pass (the new test plus the existing ones in this file).

- [ ] **Step 5: Commit**

```bash
git add src/tuned/data/paths.py tests/test_build_config.py
git commit -m "declare exp_deepseek (and exp_measure) isolated workdir siblings

Without the name in ISOLATED_WORKDIR_SIBLINGS, is_live_control_workdir
reads data/build/exp_deepseek as the frozen live control and every write
guard refuses it. exp_measure was already in the working tree uncommitted
and belongs with the same fence."
```

---

### Task 2: The arm config

**Files:**
- Create: `configs/data_law_v1_exp_deepseek.yaml`
- Test: `tests/test_build_config.py` (append)

**Interfaces:**
- Consumes: `tuned.data.config.load_build_config(path, allow_unpinned=True) -> BuildConfig` with `.build.workdir`, `.routing.generator` (list of `"provider/model"` strings), `.providers` (each with `.name`, `.models[*].limits`), and `generate._openai_usd_cap(cfg) -> float | None`.
- Produces: a loadable config whose `build.workdir == "data/build/exp_deepseek"`, `routing.generator == ["bai/deepseek-v4-flash"]`, `_openai_usd_cap(cfg) == 0.0`, and each openai model's `limits["usd_per_1m_prompt"] > 0`. Tasks 3–5 pass this path as `--config`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_config.py`:

```python
DEEPSEEK_CONFIG = Path(__file__).parent.parent / "configs" / "data_law_v1_exp_deepseek.yaml"


def test_the_deepseek_arm_config_is_fenced(tmp_path):
    """The two holes the live config has that an experiment arm may not.

    1. The live generator list falls over bai -> cerebras/gpt-oss -> paid
       lightning; a 429 storm would turn a deepseek arm into a gpt-oss arm
       without anything noticing. The arm pins the single ref.
    2. The live config declares no openai usd_cap, which _openai_usd_cap
       reads as UNCAPPED, and gpt-5-mini/nano sit in judge and tiebreak as
       backstops. usd_cap 0.0 ALONE does not block either - _usd_per_1m
       returns 0.0 for a missing price, so est_cost is 0 and 0 > 0.0 is
       False. The price must be present too (exp_measure's precedent).
    Plus: none of the Harmony flags, which are gpt-oss's chat format.
    """
    import yaml

    from tuned.data.generate import _openai_usd_cap

    cfg = load_build_config(DEEPSEEK_CONFIG, allow_unpinned=True)
    assert cfg.build.workdir == "data/build/exp_deepseek"
    assert list(cfg.routing.generator) == ["bai/deepseek-v4-flash"]
    assert _openai_usd_cap(cfg) == 0.0
    openai = next(p for p in cfg.providers if p.name == "openai")
    for model in openai.models:
        assert model.limits["usd_cap"] == 0.0
        assert model.limits["usd_per_1m_prompt"] > 0
        assert model.limits["usd_per_1m_completion"] > 0
    raw = yaml.safe_load(DEEPSEEK_CONFIG.read_text(encoding="utf-8"))
    for key in ("harmony_completions", "harmony_prefill", "harmony_s1_continue",
                "prompt_overlay", "require_pretreatment_manifest", "pretreatment_manifest"):
        assert key not in raw["build"], key
    # Everything else is the live config, byte for byte in intent: same
    # judge and tiebreak order, same bai provider block.
    live = load_build_config(DATA_CONFIG, allow_unpinned=True)
    assert list(cfg.routing.judge) == list(live.routing.judge)
    assert list(cfg.routing.tiebreak) == list(live.routing.tiebreak)
    bai = next(p for p in cfg.providers if p.name == "bai")
    assert bai.models[0].params["reasoning_effort"] == "low"
```

`DATA_CONFIG` already exists at the top of this test file (it points at `configs/data_law_v1.yaml`).

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_build_config.py::test_the_deepseek_arm_config_is_fenced -q -p no:cacheprovider --basetemp=data/build/exp_deepseek/pt`
Expected: FAIL — `FileNotFoundError` for the config.

- [ ] **Step 3: Create the config from the live one**

```bash
cp configs/data_law_v1.yaml configs/data_law_v1_exp_deepseek.yaml
```

Then apply exactly these four edits to `configs/data_law_v1_exp_deepseek.yaml` (use a Python script with `str.replace(old, new, 1)` and `assert old in text` for each, so a miss is loud):

**Edit A — header.** Insert at the very top of the file, before the first existing line:

```yaml
# Isolated deepseek validation arm (2026-08-26). Not the live control.
# ~40 synthesis tasks through bai/deepseek-v4-flash, judged by the free
# fleet, to answer five pre-registered questions before the corpus build
# scales onto this generator - see
# docs/superpowers/specs/2026-08-26-deepseek-validation-wave-design.md.
#
# This is the LIVE configs/data_law_v1.yaml with four edits and nothing else:
#   * build.workdir -> data/build/exp_deepseek
#   * routing.generator -> [bai/deepseek-v4-flash] ALONE. The live list falls
#     over to cerebras/gpt-oss and then paid lightning when b.ai is cooling,
#     and a 429 storm would turn this deepseek arm into a gpt-oss arm without
#     anything noticing. The report asserts every generation is deepseek.
#   * openai usd_cap 0.0 WITH prices on both gpt-5 models. The live config
#     declares no cap, which generate._openai_usd_cap reads as UNCAPPED, and
#     gpt-5-mini/nano sit in judge and tiebreak as backstops. The price is
#     load-bearing: _usd_per_1m returns 0.0 for a missing price, so a bare
#     usd_cap 0.0 computes 0 + 0 > 0.0 == False and blocks nothing.
#   * this header.
# Copied from the LIVE yaml, not from exp_s1/exp_measure: those carry
# harmony_* and prompt_overlay, gpt-oss's Harmony chat format, which must not
# reach a deepseek generation. No pretreatment manifest: a smoke arm, not a
# matched-cohort eval.
#
```

**Edit B — workdir.** Replace the first occurrence of

```yaml
  workdir: data/build
```

with

```yaml
  workdir: data/build/exp_deepseek
```

(It is the only `workdir:` key in the file; the assert on `old in text` and a post-check `text.count("workdir:") == 1` guard it.)

**Edit C — generator list.** Replace

```yaml
  generator: [bai/deepseek-v4-flash, cerebras/gpt-oss-120b,
              lightning/lightning-ai/gpt-oss-120b]
```

with

```yaml
  # ARM FENCE: the single ref. See the header.
  generator: [bai/deepseek-v4-flash]
```

**Edit D — openai fence.** The live openai block has two models, each with the identical line

```yaml
        limits: {rpm: 500, tpm: 200000, max_context: 400000, max_output: 16384}
```

Replace **both** occurrences (use `text.replace(old, new)` with no count, then assert the new string occurs exactly twice) with

```yaml
        limits: {rpm: 500, tpm: 200000, max_context: 400000, max_output: 16384,
                 usd_cap: 0.0, usd_per_1m_prompt: 0.25, usd_per_1m_completion: 2.0}
```

Verify that line appears exactly twice in the live file before relying on this: `grep -c "limits: {rpm: 500, tpm: 200000, max_context: 400000, max_output: 16384}" configs/data_law_v1.yaml` must print `2`. If it does not, stop and report — the live config moved.

- [ ] **Step 4: Run the config test file**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_build_config.py -q -p no:cacheprovider --basetemp=data/build/exp_deepseek/pt`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add configs/data_law_v1_exp_deepseek.yaml tests/test_build_config.py
git commit -m "add the isolated deepseek validation arm config

The live yaml with four edits: workdir, the single-ref generator list, an
openai zero-spend fence WITH prices (a bare usd_cap 0.0 blocks nothing -
_usd_per_1m reads a missing price as 0.0), and a header saying so. Copied
from the live config rather than exp_s1 so no Harmony prefill reaches a
deepseek generation."
```

---

### Task 3: `scripts/seed_exp_store.py`

**Files:**
- Create: `scripts/seed_exp_store.py`
- Test: `tests/test_seed_exp_store.py`

**Interfaces:**
- Consumes: `tuned.data.store.Store.open(path) -> Store` (context manager; `.upsert_source(source_id, license, url=None, version=None)`, `.upsert_seeds(rows: Iterable[dict]) -> int`, `.seed_count(source_id=None) -> int`, `.conn`); `tuned.data.paths.build_paths(workdir).ensure() -> BuildPaths` with `.state_db`; `tuned.data.paths.is_live_control_workdir`; `tuned.data.config.load_build_config`; `tuned.data.tasks.seed_token_budget(cfg) -> int`.
- Produces:
  - `seed_store(store, live_db: Path, *, per_source: int, offset_seed: int, budget: int) -> dict[str, dict]` — returns `{source_id: {"copied": int, "oversize": int}}`.
  - CLI: `python scripts/seed_exp_store.py --config <yaml> --from <live.sqlite3> [--per-source 200] [--seed 3407]`, exit 0 on success, exit 2 if the config's workdir is the live control.
  - Task 4 runs the CLI; Task 5's report reads the `oversize` counts it prints.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_seed_exp_store.py`:

```python
"""scripts/seed_exp_store.py - copy source + a seed sample out of a live
store, read-only, into an isolated arm store."""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from tuned.data.store import Store  # noqa: E402

import seed_exp_store  # noqa: E402

SRC_A = "s3://indian-supreme-court-judgments"
SRC_B = "L-NLProc/PredEx_Instruction-Tuning_Pred-Exp"


def _live(tmp_path: Path) -> Path:
    """A synthetic live store: two sources, 30 seeds each, a known length
    spread so an oversize row exists in one source and not the other."""
    db = tmp_path / "live" / "law_v1.sqlite3"
    with Store.open(db) as store:
        store.upsert_source(SRC_A, "CC-BY-4.0")
        store.upsert_source(SRC_B, "Apache-2.0")
        rows = []
        for i in range(30):
            rows.append({"seed_id": f"a{i:02d}", "source_id": SRC_A, "text": "t" * 10,
                         "token_count": 1200, "case_type": "bail", "code_era": "bns",
                         "meta_json": json.dumps({"kind": "chunk", "oversize": False})})
            rows.append({"seed_id": f"b{i:02d}", "source_id": SRC_B, "text": "t" * 10,
                         "token_count": 9000 if i % 5 == 0 else 2500,
                         "case_type": "bail", "code_era": "ipc",
                         "meta_json": json.dumps({"estimator": "chars/4"})})
        store.upsert_seeds(rows)
    return db


def test_seed_store_copies_sources_and_a_per_source_sample(tmp_path):
    live = _live(tmp_path)
    arm = tmp_path / "arm" / "law_v1.sqlite3"
    with Store.open(arm) as store:
        report = seed_exp_store.seed_store(
            store, live, per_source=10, offset_seed=0, budget=4692
        )
        assert store.seed_count(SRC_A) == 10
        assert store.seed_count(SRC_B) == 10
        sources = {r[0] for r in store.conn.execute("SELECT source_id FROM source")}
        assert sources == {SRC_A, SRC_B}
    # The sample is taken WITHOUT a length filter - oversize rows come
    # through, which is what lets the planner's gate be tested live.
    assert report[SRC_A] == {"copied": 10, "oversize": 0}
    assert report[SRC_B]["copied"] == 10
    assert report[SRC_B]["oversize"] == 2  # b00, b05 in the first ten by seed_id


def test_seed_store_round_trips_meta_json_without_double_encoding(tmp_path):
    live = _live(tmp_path)
    arm = tmp_path / "arm" / "law_v1.sqlite3"
    with Store.open(arm) as store:
        seed_exp_store.seed_store(store, live, per_source=5, offset_seed=0, budget=4692)
        row = store.get_seed("a00")
    assert json.loads(row["meta_json"]) == {"kind": "chunk", "oversize": False}


def test_seed_store_is_idempotent_and_deterministic(tmp_path):
    live = _live(tmp_path)
    arm = tmp_path / "arm" / "law_v1.sqlite3"
    with Store.open(arm) as store:
        first = seed_exp_store.seed_store(store, live, per_source=7, offset_seed=3, budget=4692)
        ids_1 = sorted(r[0] for r in store.conn.execute("SELECT seed_id FROM seed"))
        second = seed_exp_store.seed_store(store, live, per_source=7, offset_seed=3, budget=4692)
        ids_2 = sorted(r[0] for r in store.conn.execute("SELECT seed_id FROM seed"))
    assert ids_1 == ids_2
    assert first == second
    assert len(ids_1) == 14


def test_seed_store_offset_wraps_inside_the_source(tmp_path):
    live = _live(tmp_path)
    arm = tmp_path / "arm" / "law_v1.sqlite3"
    with Store.open(arm) as store:
        # 30 rows per source; offset 28 with per_source 5 must wrap, not
        # come back short.
        seed_exp_store.seed_store(store, live, per_source=5, offset_seed=28, budget=4692)
        assert store.seed_count(SRC_A) == 5
        ids = {r[0] for r in store.conn.execute(
            "SELECT seed_id FROM seed WHERE source_id = ?", (SRC_A,))}
    assert ids == {"a28", "a29", "a00", "a01", "a02"}


def test_seed_store_never_writes_the_live_db(tmp_path):
    live = _live(tmp_path)
    before = (live.stat().st_size, live.stat().st_mtime_ns)
    arm = tmp_path / "arm" / "law_v1.sqlite3"
    with Store.open(arm) as store:
        seed_exp_store.seed_store(store, live, per_source=10, offset_seed=0, budget=4692)
    assert (live.stat().st_size, live.stat().st_mtime_ns) == before


def test_cli_refuses_a_live_control_workdir(tmp_path, monkeypatch, capsys):
    """The config's workdir, not the --from path, is what must be isolated."""
    live = _live(tmp_path)
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "build:\n  train_config: configs/law_v1_8b_ddp.yaml\n  workdir: data/build\n",
        encoding="utf-8",
    )
    rc = seed_exp_store.main(["--config", str(yaml_path), "--from", str(live)])
    assert rc == 2
    assert "live control" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_seed_exp_store.py -q -p no:cacheprovider --basetemp=data/build/exp_deepseek/pt`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'seed_exp_store'`.

- [ ] **Step 3: Write the script**

Create `scripts/seed_exp_store.py`:

```python
"""Seed an isolated experiment store from the live control, read-only.

The exp_* arms before this one were seeded by hand. This copies the
`source` table and a deterministic per-source sample of `seed` rows out of
a live store into the arm's own store, so an arm can be stood up in one
command and the live database is never opened for write.

    python scripts/seed_exp_store.py --config configs/data_law_v1_exp_deepseek.yaml \\
        --from data/build/state/law_v1.sqlite3 --per-source 200 --seed 3407

THE SAMPLE HAS NO LENGTH FILTER, on purpose. tasks._candidate_seeds refuses
a seed longer than seed_token_budget(cfg); the only way to prove that live
is to have oversize seeds in the store and zero tasks planned against them.
The per-source `oversize` count printed here is the numerator the report
needs.

Deterministic without an RNG: seed_ids are content-derived hashes, so
ORDER BY seed_id is already a stable pseudo-random order, and --seed picks
the starting offset inside it (wrapping at the source's end). The same
arguments produce the same rows on any machine.

Idempotent: Store.upsert_seeds is INSERT OR REPLACE on the primary key, so a
re-run rewrites the same rows and adds none.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

from tuned.data.config import load_build_config
from tuned.data.paths import build_paths, is_live_control_workdir
from tuned.data.store import Store
from tuned.data.tasks import seed_token_budget


def _open_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _sample(live: sqlite3.Connection, source_id: str, *, per_source: int, offset_seed: int):
    """`per_source` rows of one source, ORDER BY seed_id, starting at
    offset_seed mod count and wrapping - so a large offset never comes back
    short, and the same arguments always name the same rows."""
    total = live.execute(
        "SELECT COUNT(*) FROM seed WHERE source_id = ?", (source_id,)
    ).fetchone()[0]
    if total == 0:
        return []
    start = offset_seed % total
    take = min(per_source, total)
    head = live.execute(
        "SELECT * FROM seed WHERE source_id = ? ORDER BY seed_id LIMIT ? OFFSET ?",
        (source_id, take, start),
    ).fetchall()
    if len(head) < take:
        head += live.execute(
            "SELECT * FROM seed WHERE source_id = ? ORDER BY seed_id LIMIT ?",
            (source_id, take - len(head)),
        ).fetchall()
    return head


def seed_store(
    store: Store, live_db: Path, *, per_source: int, offset_seed: int, budget: int
) -> dict[str, dict]:
    """Copy `source` and a per-source seed sample from live_db into store.

    Returns {source_id: {"copied": n, "oversize": n_over_budget}}.
    """
    live = _open_ro(Path(live_db))
    try:
        report: dict[str, dict] = {}
        for src in live.execute("SELECT * FROM source ORDER BY source_id").fetchall():
            store.upsert_source(
                src["source_id"], src["license"], url=src["url"], version=src["version"]
            )
            rows = [dict(r) for r in _sample(
                live, src["source_id"], per_source=per_source, offset_seed=offset_seed
            )]
            store.upsert_seeds(rows)
            oversize = sum(1 for r in rows if (r.get("token_count") or 0) > budget)
            report[src["source_id"]] = {"copied": len(rows), "oversize": oversize}
        return report
    finally:
        live.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="the ARM's build config")
    parser.add_argument("--from", dest="live_db", required=True,
                        help="the live store to copy FROM (opened read-only)")
    parser.add_argument("--per-source", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3407, help="offset seed (see docstring)")
    args = parser.parse_args(argv)

    cfg = load_build_config(args.config, allow_unpinned=True)
    if is_live_control_workdir(cfg.build.workdir):
        print(
            f"refusing: {args.config} points at the live control workdir "
            f"{cfg.build.workdir!r}; an arm must have its own",
            file=sys.stderr,
        )
        return 2
    live_db = Path(args.live_db)
    if not live_db.is_file():
        print(f"no such live store: {live_db}", file=sys.stderr)
        return 2
    budget = seed_token_budget(cfg)
    paths = build_paths(cfg.build.workdir).ensure()
    with Store.open(paths.state_db) as store:
        report = seed_store(
            store, live_db, per_source=args.per_source, offset_seed=args.seed, budget=budget
        )
        total = store.seed_count()
    print(f"arm store: {paths.state_db}")
    print(f"seed budget (max_seq_length - reply reserve): {budget} tokens")
    for source_id, counts in report.items():
        print(f"  {source_id}: copied {counts['copied']}, over budget {counts['oversize']}")
    print(f"seeds in arm store: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note on `test_cli_refuses_a_live_control_workdir`: `load_build_config` needs more than `build:` to load. If the minimal yaml in that test fails validation before the workdir check, replace the yaml body with a copy of the live config edited to `workdir: data/build` — i.e. in the test, `doc = yaml.safe_load(Path("configs/data_law_v1.yaml").read_text())`, set `doc["build"]["workdir"] = "data/build"`, dump to `yaml_path`. The assertion stays the same. Report which form was needed.

- [ ] **Step 4: Run the tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_seed_exp_store.py -q -p no:cacheprovider --basetemp=data/build/exp_deepseek/pt`
Expected: 6 passed. If `test_seed_store_round_trips_meta_json_without_double_encoding` fails with a doubly-encoded string, `Store.upsert_seeds` is JSON-encoding a value that is already a string; fix in the script by `json.loads`-ing `meta_json`/`roles_json`/`answer_key_json` when they are non-empty strings before passing rows to `upsert_seeds`, and say so in the commit message.

- [ ] **Step 5: Commit**

```bash
git add scripts/seed_exp_store.py tests/test_seed_exp_store.py
git commit -m "seed an experiment store from the live control, read-only

The exp_* arms were seeded by hand. This copies source and a
deterministic per-source seed sample - ORDER BY content-derived seed_id
from an offset, wrapping - out of a live store opened mode=ro into the
arm's own store. No length filter, on purpose: the planner's seed gate
can only be proved live if oversize seeds are present and none are
planned against. Refuses a config whose workdir is the live control."
```

---

### Task 4: Operator run — seed, plan, gate-check, generate, judge

**Files:** none committed. Produces the arm store `data/build/exp_deepseek/state/law_v1.sqlite3` populated with tasks, generations and judgements, plus two stat lines saved to `data/build/exp_deepseek/out/live_stat_before.txt` / `live_stat_after.txt`.

**Interfaces:**
- Consumes: Tasks 1–3 committed. `BAI_API_KEY`, `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `MISTRAL_API_KEY`, `OPENAI_API_KEY` in the worktree `.env` (all present as of 2026-08-26).
- Produces: the populated arm store Task 5 reads.

All commands run from the worktree root with `PYTHONIOENCODING=utf-8` set (Windows console encoding otherwise breaks on `→` in log lines).

- [ ] **Step 1: Record the live store before anything runs**

```bash
mkdir -p data/build/exp_deepseek/out
stat -c '%s %Y' data/build/state/law_v1.sqlite3 | tee data/build/exp_deepseek/out/live_stat_before.txt
```

- [ ] **Step 2: Seed the arm**

```bash
./.venv/Scripts/python.exe scripts/seed_exp_store.py \
  --config configs/data_law_v1_exp_deepseek.yaml \
  --from data/build/state/law_v1.sqlite3 --per-source 200 --seed 3407
```

Expected output shape: three `copied 200` lines; `over budget` non-zero for PredEx and TathyaNyaya (expected ~15–25 each), ~0–2 for the SC source; `seeds in arm store: 600`. **If every source prints `over budget 0`, re-run with `--seed 4407`** — the sample slice must contain at least one oversize seed for measurement #2.

- [ ] **Step 3: Plan three arms, one per source**

```bash
P=./.venv/Scripts/python.exe
C=configs/data_law_v1_exp_deepseek.yaml
MIX=irac_analysis=0.55,summarization=0.45
$P -m tuned.data.tasks --config $C --stream synthesis --arm sc     --n 13 --mix $MIX --source s3://indian-supreme-court-judgments
$P -m tuned.data.tasks --config $C --stream synthesis --arm predex --n 14 --mix $MIX --source L-NLProc/PredEx_Instruction-Tuning_Pred-Exp
$P -m tuned.data.tasks --config $C --stream synthesis --arm tathya --n 13 --mix $MIX --source L-NLProc/TathyaNyaya-and-FactLegalLlama-NyayaFacts-Datasets
```

Expected: each prints a planned count; totals 13 / 14 / 13.

- [ ] **Step 4: Gate check — before a single call is made**

```bash
./.venv/Scripts/python.exe - <<'PY'
import sqlite3
from tuned.data.config import load_build_config
from tuned.data.tasks import seed_token_budget
cfg = load_build_config("configs/data_law_v1_exp_deepseek.yaml", allow_unpinned=True)
b = seed_token_budget(cfg)
db = sqlite3.connect("data/build/exp_deepseek/state/law_v1.sqlite3")
over_in_store = db.execute("SELECT COUNT(*) FROM seed WHERE COALESCE(token_count,0) > ?", (b,)).fetchone()[0]
over_planned = db.execute("SELECT COUNT(*) FROM task t JOIN seed s ON s.seed_id=t.seed_id WHERE COALESCE(s.token_count,0) > ?", (b,)).fetchone()[0]
tasks = db.execute("SELECT arm, COUNT(*) FROM task GROUP BY arm").fetchall()
print(f"budget={b} oversize_in_store={over_in_store} planned_over_budget={over_planned} tasks={tasks}")
assert over_in_store >= 1, "measurement #2 needs an oversize seed present"
assert over_planned == 0, "SEED GATE FAILED - stop, do not generate"
PY
```

Expected: `planned_over_budget=0`, `oversize_in_store>=1`, `tasks=[('predex', 14), ('sc', 13), ('tathya', 13)]`. If the assert fires, **stop and report** — the gate from `tasks.py` `70131e1` is not doing its job and generating would waste the budget.

- [ ] **Step 5: Generate (≈ 8–12 min)**

```bash
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tuned.data.generate \
  --config configs/data_law_v1_exp_deepseek.yaml --n-workers 4 --max-batches 30 \
  2>&1 | tee data/build/exp_deepseek/out/generate.log
```

The preflight prints role/key coverage first. If it exits 2 with a judge-slot pool gap, re-run with `--allow-pool-gaps` and note it in the report. Watch for: `reply truncated before any content` (the bai hook's retryable error — counted, not fatal), 429s, and the `reasoning_effort` 400 (must NOT appear: the config says `low`, which is in the enum). Expected finish: queue drained or batch cap hit; a summary line of totals.

- [ ] **Step 6: Judge (≈ 12 min)**

```bash
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tuned.data.judge \
  --config configs/data_law_v1_exp_deepseek.yaml --n-workers 3 --max-batches 30 \
  2>&1 | tee data/build/exp_deepseek/out/judge.log
```

Expected: rows move `judging` → `accepted` / `rejected`; some `judge_error` is normal (report it). If the openai fence is working there is **no** line mentioning `gpt-5` making a call; if one appears, stop and report.

- [ ] **Step 7: Record the live store after**

```bash
stat -c '%s %Y' data/build/state/law_v1.sqlite3 | tee data/build/exp_deepseek/out/live_stat_after.txt
diff data/build/exp_deepseek/out/live_stat_before.txt data/build/exp_deepseek/out/live_stat_after.txt && echo "live store untouched"
```

Expected: `live store untouched`.

No commit for this task (nothing under `data/` is tracked).

---

### Task 5: Measure and write the report

**Files:**
- Create: `data/build/exp_deepseek/out/report_wave.py` (gitignored scratch — under `data/`)
- Create: `docs/reports/2026-08-26-deepseek-validation-wave.md`

**Interfaces:**
- Consumes: the arm store from Task 4; the live store (read-only) for the gpt-oss judge baseline; `tuned.data.eval_matched.dual_judge_decision(slots, already_regenerated=False) -> str | None` (accepts a list of `judgement` row dicts — `_as_slot_map` handles `judge_slot`/`grounding`/`validity`/`coverage` keys); `transformers.AutoTokenizer` (tokenizers-only install is enough) with `cfg.model_repo` / `cfg.model_revision`.
- Produces: the report.

- [ ] **Step 1: Write the measurement script**

Create `data/build/exp_deepseek/out/report_wave.py`:

```python
"""Measure the deepseek validation arm and print the report body (markdown).

Reads the ARM store read-write-safe (no writes issued) and the LIVE store
read-only for the gpt-oss judge baseline. Every number in the report comes
from here; nothing is typed in by hand.
"""
import json
import sqlite3
import statistics
from pathlib import Path

from transformers import AutoTokenizer

from tuned.data.config import load_build_config
from tuned.data.eval_matched import dual_judge_decision
from tuned.data.tasks import seed_token_budget

CFG = "configs/data_law_v1_exp_deepseek.yaml"
ARM = Path("data/build/exp_deepseek/state/law_v1.sqlite3")
LIVE = Path("data/build/state/law_v1.sqlite3")
CAP = 8192
PROJ = {"reasoning_mean": 2097, "row_mean": 4440, "row_p90": 5514, "row_p99": 9719}

cfg = load_build_config(CFG, allow_unpinned=True)
BUDGET = seed_token_budget(cfg)
tok = AutoTokenizer.from_pretrained(cfg.model_repo, revision=cfg.model_revision)
THINK_OPEN, THINK_CLOSE = cfg.think_open, cfg.think_close


def ro(path):
    c = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def pct(v, p):
    if not v:
        return 0
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def n_tok(s):
    return len(tok.encode(s, add_special_tokens=False))


def row_tokens(user, think, answer):
    content = f"{THINK_OPEN}\n{think}\n{THINK_CLOSE}\n\n{answer}" if think else answer
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": content}],
        tokenize=False, add_generation_prompt=False,
    )
    return len(tok.encode(rendered, add_special_tokens=False))


def stats_line(label, vals, cap=None):
    if not vals:
        return f"| {label} | 0 | — | — | — | — | — |"
    over = f" ({sum(v > cap for v in vals)} over {cap})" if cap else ""
    return (f"| {label} | {len(vals)} | {statistics.mean(vals):.0f} | {pct(vals,50)} | "
            f"{pct(vals,90)} | {pct(vals,99)} | {max(vals)}{over} |")


arm = ro(ARM)
live = ro(LIVE)
out = []
P = out.append

# ---- 1. pipe health ----------------------------------------------------
gens = arm.execute("""SELECT g.*, t.arm, t.task_type, s.token_count AS seed_tok, s.text AS seed_text
                      FROM generation g JOIN task t ON t.task_id=g.task_id
                      JOIN seed s ON s.seed_id=t.seed_id ORDER BY g.gen_id""").fetchall()
models = arm.execute("SELECT model, COUNT(*) FROM generation GROUP BY model").fetchall()
ledger = arm.execute("SELECT provider, model, requests, prompt_tokens, completion_tokens, errors_429 FROM budget_ledger").fetchall()
with_content = [g for g in gens if (g["answer"] or "").strip()]
errored = [g for g in gens if g["error"]]
trunc = [g for g in gens if g["finish_reason"] == "length"]
lat = [g["latency_ms"] / 1000 for g in gens if g["latency_ms"]]
openai_rows = [r for r in ledger if r["provider"] == "openai" and r["requests"]]
P("## 1. Pipe health\n")
P(f"- generations recorded: **{len(gens)}**; with content: **{len(with_content)}** "
  f"({100*len(with_content)/max(1,len(gens)):.0f}%) — pass line ≥ 90%: "
  f"**{'PASS' if len(with_content) >= 0.9*len(gens) and gens else 'FAIL'}**")
P(f"- errored rows: {len(errored)}; finish_reason=length: {len(trunc)}")
P(f"- latency s: mean {statistics.mean(lat):.1f}, p50 {pct(lat,50):.1f}, p90 {pct(lat,90):.1f}, max {max(lat):.1f}" if lat else "- latency: n/a")
P(f"- models seen: {[(m[0], m[1]) for m in models]} — only deepseek: "
  f"**{'PASS' if all(m[0]=='deepseek-v4-flash' for m in models) else 'FAIL'}**")
P(f"- openai requests on the ledger: {sum(r['requests'] for r in openai_rows)} — $0 fence: "
  f"**{'PASS' if not openai_rows else 'FAIL'}**")
P("- ledger: " + "; ".join(f"{r['provider']}/{r['model']} req={r['requests']} 429={r['errors_429']} "
                          f"tok={r['prompt_tokens']}+{r['completion_tokens']}" for r in ledger))
P("")

# ---- 2. seed gate ------------------------------------------------------
over_store = arm.execute("SELECT COUNT(*) FROM seed WHERE COALESCE(token_count,0) > ?", (BUDGET,)).fetchone()[0]
over_planned = arm.execute("SELECT COUNT(*) FROM task t JOIN seed s ON s.seed_id=t.seed_id WHERE COALESCE(s.token_count,0) > ?", (BUDGET,)).fetchone()[0]
P("## 2. Seed gate, live\n")
P(f"- budget `seed_token_budget(cfg)` = **{BUDGET}** tokens")
P(f"- oversize seeds present in the arm store: **{over_store}** (need ≥ 1)")
P(f"- tasks planned against an oversize seed: **{over_planned}** (need 0) — "
  f"**{'PASS' if over_planned == 0 and over_store >= 1 else 'FAIL'}**\n")

# ---- 3. lengths vs projection -----------------------------------------
recs = []
for g in with_content:
    think, answer = g["think"] or "", g["answer"] or ""
    recs.append({"arm": g["arm"], "think": n_tok(think) if think else 0,
                 "answer": n_tok(answer), "seed": n_tok(g["seed_text"] or ""),
                 "row": row_tokens(g["seed_text"] or "", think, answer),
                 "think_tokens_reported": g["think_tokens"]})
P("## 3. Lengths vs the 2026-08-26 projection\n")
P("| series | n | mean | p50 | p90 | p99 | max |")
P("|---|---|---|---|---|---|---|")
P(stats_line("reasoning (pinned tokenizer)", [r["think"] for r in recs if r["think"]]))
P(stats_line("reasoning (provider-reported)", [r["think_tokens_reported"] for r in recs if r["think_tokens_reported"]]))
P(stats_line("answer", [r["answer"] for r in recs]))
P(stats_line("seed (user turn)", [r["seed"] for r in recs]))
P(stats_line("**templated row**", [r["row"] for r in recs], cap=CAP))
for a in ("sc", "predex", "tathya"):
    P(stats_line(f"row — arm `{a}`", [r["row"] for r in recs if r["arm"] == a], cap=CAP))
P("")
P(f"Projected: reasoning mean {PROJ['reasoning_mean']}, row mean {PROJ['row_mean']}, "
  f"p90 {PROJ['row_p90']}, p99 {PROJ['row_p99']}. Measured row mean "
  f"{statistics.mean(r['row'] for r in recs):.0f}." if recs else "no rows with content")
P("")

# ---- 4. think_max violation rate --------------------------------------
gates = arm.execute("SELECT gen_id, passed, detail_json FROM gate_result WHERE gate='length_band'").fetchall()
viol = [json.loads(r["detail_json"] or "{}").get("violations", []) for r in gates]
n_think_max = sum("think>think_max" in v for v in viol)
n_total_max = sum("total>total_max" in v for v in viol)
n_think_min = sum("think<think_min" in v for v in viol)
P("## 4. `think_max` violation rate (the real unknown)\n")
P(f"- length_band gate results: {len(gates)}; passed: {sum(r['passed'] for r in gates)}")
P(f"- `think>think_max` (3000): **{n_think_max}/{len(gates)}** = "
  f"**{100*n_think_max/max(1,len(gates)):.0f}%** — threshold 30%: "
  f"{'OVER — low and think_max disagree; one must move' if gates and n_think_max > 0.3*len(gates) else 'under'}")
P(f"- `total>total_max` (8192): {n_total_max}; `think<think_min` (500): {n_think_min}")
disp = arm.execute("SELECT state, disposition, COUNT(*) FROM task GROUP BY 1,2 ORDER BY 3 DESC").fetchall()
P("- task states/dispositions: " + "; ".join(f"{s}/{d or '-'}={n}" for s, d, n in disp))
P("")

# ---- 5. judge accept rate vs baseline ---------------------------------
def accept_rate(conn):
    decided = []
    for (gen_id,) in conn.execute("SELECT DISTINCT gen_id FROM judgement"):
        slots = [dict(r) for r in conn.execute("SELECT * FROM judgement WHERE gen_id=?", (gen_id,))]
        d = dual_judge_decision(slots)
        if d is not None:
            decided.append(d)
    n = len(decided)
    acc = sum(d == "accept" for d in decided)
    return n, acc, decided

n_arm, acc_arm, dec_arm = accept_rate(arm)
n_live, acc_live, dec_live = accept_rate(live)
from collections import Counter
P("## 5. Judge accept rate vs the gpt-oss baseline\n")
P(f"- arm (deepseek): decided {n_arm}, accepted {acc_arm} = **{100*acc_arm/max(1,n_arm):.0f}%**; decisions {dict(Counter(dec_arm))}")
P(f"- live baseline (gpt-oss, read-only): decided {n_live}, accepted {acc_live} = **{100*acc_live/max(1,n_live):.0f}%**; decisions {dict(Counter(dec_live))}")
P(f"- n={n_arm} ⇒ a ±{100*1.96*(0.25/max(1,n_arm))**0.5:.0f} pp interval; a signal, not a verdict")
jslots = arm.execute("SELECT judge_slot, provider, model, COUNT(*) FROM judgement GROUP BY 1,2,3").fetchall()
P("- judges used: " + "; ".join(f"{s}={p}/{m} ×{n}" for s, p, m, n in jslots))
P("")

# ---- 6. per-arm --------------------------------------------------------
P("## 6. Per-arm breakdown\n")
P("| arm | tasks | gens | with content | accepted |")
P("|---|---|---|---|---|")
for a in ("sc", "predex", "tathya"):
    t = arm.execute("SELECT COUNT(*) FROM task WHERE arm=?", (a,)).fetchone()[0]
    gg = [g for g in gens if g["arm"] == a]
    acc = arm.execute("SELECT COUNT(*) FROM task WHERE arm=? AND state='accepted'", (a,)).fetchone()[0]
    P(f"| {a} | {t} | {len(gg)} | {sum(1 for g in gg if (g['answer'] or '').strip())} | {acc} |")
P("")
print("\n".join(out))
```

- [ ] **Step 2: Run it**

```bash
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe data/build/exp_deepseek/out/report_wave.py \
  > data/build/exp_deepseek/out/report_body.md
```

Expected: markdown on stdout, sections 1–6, every PASS/FAIL line populated. If `dual_judge_decision` rejects the row-dict list (returns `None` for every row), pass `{r["judge_slot"]: (r["grounding"], r["validity"], r["coverage"]) for r in slots}` instead — `_as_slot_map` accepts that shape too.

- [ ] **Step 3: Write the report**

Create `docs/reports/2026-08-26-deepseek-validation-wave.md` with this frame, pasting the generated body into the marked place and filling the two prose sections from what the numbers say:

```markdown
# DeepSeek validation wave — measured 2026-08-26

**Arm:** `data/build/exp_deepseek`, config `configs/data_law_v1_exp_deepseek.yaml`
**Spec:** `docs/superpowers/specs/2026-08-26-deepseek-validation-wave-design.md`
**Generator:** `bai/deepseek-v4-flash`, `reasoning_effort: low`, temperature 0.7
**Judges:** qwen3.6-27b (A), gemma-4-31b (B), mistral-large (tiebreak); openai fenced to $0
**Live store:** untouched — `live_stat_before.txt` == `live_stat_after.txt`

## Verdict

<3–6 sentences: which pre-registered lines passed, the think_max rate and
what it implies for `low` vs `think_max`, whether the length projection
held, and the judge signal vs baseline with its interval.>

<PASTE report_body.md HERE — sections 1–6>

## 7. Cost and wall clock

- generate: <mm:ss from generate.log timestamps>, judge: <mm:ss from judge.log>
- b.ai tokens: <from ledger line>, 429s: <n>
- paid spend: $0 (fenced)

## 8. What changes in the live config, if anything

<Bullet per finding. If think_max violations > 30%: state which of
`reasoning_effort` / `length_band.think_max` should move and why. If the
row p99 exceeded 8192 beyond the projection: state whether the seed
reserve (3,500) or the source mix is the cause. If accept rate is far
below baseline: name the judge axis that failed most. Otherwise: "none".>

## Caveats

- n=<n> decided rows: the accept-rate interval is ±<pp> pp.
- One wave, one hour: no day-over-day drift on b.ai's upstream mix is measured.
- Baseline judgements in the live store were made under earlier judge
  prompt versions; the comparison is indicative, not matched.
```

- [ ] **Step 4: Verify the report has no unfilled angle-bracket placeholders**

Run: `grep -n "<" docs/reports/2026-08-26-deepseek-validation-wave.md | grep -v "^.*|.*<" | grep "<[a-z0-9 –-]*>" ; echo "exit=$?"`
Expected: no lines printed (exit=1) — every `<...>` prose slot is filled.

- [ ] **Step 5: Commit**

```bash
git add docs/reports/2026-08-26-deepseek-validation-wave.md
git commit -m "docs: deepseek validation wave - <one-line verdict>

<3-5 lines: the pass/fail lines with numbers; the think_max rate; the
projection check; the judge signal. Written from the report, not from
memory.>"
```

---

## Self-review

**Spec coverage.** Isolation (workdir sibling → Task 1; store + read-only ATTACH-equivalent → Task 3; config copied from live with fences → Task 2). Seeding script with per-source sample, no length filter, oversize counts, idempotent, refuses live → Task 3. Planning three arms with `--arm`/`--source`/`--mix` and the pre-call gate check → Task 4 steps 3–4. Running generate/judge with the batch caps → Task 4 steps 5–6. Live-store stat before/after → Task 4 steps 1, 7. Measurement 1–5 + per-arm + cost + "what changes" + caveats → Task 5. Acceptance criteria 1–6 map to Tasks 1, 3, 4(step 4), 4(steps 5–6 + report §1), 5, 4(step 7).

**Placeholder scan.** The only angle-bracket slots are in the Task 5 report frame and the final commit message, both explicitly "fill from the numbers" with a grep step that fails the task if any remain. No "TBD"/"similar to".

**Type consistency.** `seed_store(store, live_db, *, per_source, offset_seed, budget) -> dict[str, dict]` is the same in the script, the tests and the CLI. `seed_token_budget(cfg)` is imported from `tuned.data.tasks` in Tasks 3, 4, 5 with the same signature. `dual_judge_decision(slots)` is called with a list of judgement row dicts, with the documented fallback to a slot map.

**Two things flagged for the implementer rather than resolved here:** (a) Task 3's CLI-refusal test may need a fuller yaml than the three-line one if `load_build_config` validates more before the workdir check — the alternative is written in the step; (b) Task 5 step 2 names the alternative slot shape if `_as_slot_map` does not take row dicts.
