import dataclasses
from pathlib import Path

import pytest

from tuned.data.paths import BuildPaths, build_paths


def test_layout_properties_are_exact(tmp_path):
    bp = build_paths(tmp_path)
    assert bp.root == tmp_path
    assert bp.state_db == tmp_path / "state" / "law_v1.sqlite3"
    assert bp.corpus_dir == tmp_path / "corpus"
    assert bp.gold_dir == tmp_path / "gold"
    assert bp.streams_dir == tmp_path / "streams"
    assert bp.out_dir == tmp_path / "out"
    assert bp.logs_dir == tmp_path / "logs"


def test_build_paths_accepts_str(tmp_path):
    bp = build_paths(str(tmp_path))
    assert bp.root == Path(str(tmp_path))
    assert isinstance(bp.root, Path)


def test_ensure_creates_all_static_dirs(tmp_path):
    bp = build_paths(tmp_path / "build")
    assert not (tmp_path / "build").exists()
    result = bp.ensure()
    assert result is bp  # chaining
    assert bp.state_db.parent.is_dir()
    assert (bp.root / "raw" / "gen").is_dir()
    assert (bp.root / "raw" / "judge").is_dir()
    assert bp.corpus_dir.is_dir()
    assert bp.gold_dir.is_dir()
    assert bp.streams_dir.is_dir()
    assert bp.out_dir.is_dir()
    assert bp.logs_dir.is_dir()
    # ensure() itself does not create day dirs
    assert not (bp.root / "raw" / "gen" / "2026-08-11").exists()


def test_raw_gen_dir_creates_day_dir_on_call(tmp_path):
    bp = build_paths(tmp_path)
    day_dir = bp.raw_gen_dir("2026-08-11")
    assert day_dir == tmp_path / "raw" / "gen" / "2026-08-11"
    assert day_dir.is_dir()


def test_raw_judge_dir_creates_day_dir_on_call(tmp_path):
    bp = build_paths(tmp_path)
    day_dir = bp.raw_judge_dir("2026-08-11")
    assert day_dir == tmp_path / "raw" / "judge" / "2026-08-11"
    assert day_dir.is_dir()


def test_raw_dirs_are_independent_days(tmp_path):
    bp = build_paths(tmp_path)
    d1 = bp.raw_gen_dir("2026-08-10")
    d2 = bp.raw_gen_dir("2026-08-11")
    assert d1 != d2
    assert d1.is_dir() and d2.is_dir()


def test_build_paths_is_frozen(tmp_path):
    bp = build_paths(tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        bp.root = tmp_path / "other"


def test_ensure_is_idempotent(tmp_path):
    bp = build_paths(tmp_path)
    bp.ensure()
    bp.ensure()  # must not raise on already-existing dirs
    assert bp.corpus_dir.is_dir()
