"""scripts/seed_exp_store.py - copy source + a seed sample out of a live
store, read-only, into an isolated arm store."""
import json
import sqlite3
import sys
from pathlib import Path

import pytest
import yaml

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
    # A minimal build: block fails load_build_config's own schema checks
    # (e.g. the required length_band) before it ever reaches the workdir
    # check, so use a full real config with only the workdir overridden.
    doc = yaml.safe_load(Path("configs/data_law_v1.yaml").read_text())
    doc["build"]["workdir"] = "data/build"
    yaml_path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    rc = seed_exp_store.main(["--config", str(yaml_path), "--from", str(live)])
    assert rc == 2
    assert "live control" in capsys.readouterr().err
