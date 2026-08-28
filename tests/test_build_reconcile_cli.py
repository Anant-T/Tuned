"""The reconcile CLI: fold raw NDJSON from elsewhere back into the store.

The raw logs are the system of record; a worker that ran on another machine
comes home as raw files. This suite drives `python -m tuned.data.reconcile`
end to end against a redirected workdir: recover, skip what cannot be
recovered (loudly), and recover nothing the second time.
"""

from pipeline_fakes import temp_config
from test_build_store import _gen_envelope, _populate

from tuned.data.config import load_build_config
from tuned.data.jsonl import append_ndjson
from tuned.data.paths import build_paths
from tuned.data.reconcile import main as reconcile_main
from tuned.data.store import Store


def _workdir_paths(config_path: str):
    cfg = load_build_config(config_path)
    return build_paths(cfg.build.workdir).ensure()


def test_the_cli_sweeps_the_workdir_and_is_idempotent(tmp_path, capsys):
    config = temp_config(tmp_path)
    paths = _workdir_paths(config)
    with Store.open(paths.state_db) as store:
        _populate(store, n=2)

    gen_raw = paths.raw_gen_dir("2026-08-29") / "gen.ndjson"
    append_ndjson(gen_raw, _gen_envelope("t0", 1))
    append_ndjson(gen_raw, _gen_envelope("t1", 1))
    judge_raw = paths.raw_judge_dir("2026-08-29") / "judge.ndjson"
    append_ndjson(
        judge_raw,
        {
            "kind": "judgement", "task_id": "t0", "attempt": 1, "judge_slot": "a",
            "provider": "groq", "model": "qwen/qwen3.6-27b",
            "grounding": 5, "validity": 5, "coverage": 5, "rationale": "fine",
        },
    )
    # An orphan (no such generation anywhere) and a torn tail (the expected
    # shape of a crash) ride along: both must be skipped with a diagnostic,
    # never raised, and never counted as recovered.
    append_ndjson(
        judge_raw,
        {"kind": "judgement", "task_id": "t9", "attempt": 1, "judge_slot": "a", "grounding": 2},
    )
    with gen_raw.open("ab") as f:
        f.write(b'{"kind": "generation", "task_id": "t1", "att')

    assert reconcile_main(["--config", config]) == 0
    out = capsys.readouterr().out
    assert "recovered 3 row(s)" in out  # 2 generations + 1 judgement, orphan excluded

    with Store.open(paths.state_db) as store:
        gen = store.latest_generation("t0")
        assert gen is not None
        assert store.latest_generation("t1") is not None
        assert [j["judge_slot"] for j in store.judgements_for(gen["gen_id"])] == ["a"]
        kinds = [e["kind"] for e in store.events()]
        assert "reconcile_bad_line" in kinds

    # Idempotent: everything recoverable is indexed now.
    assert reconcile_main(["--config", config]) == 0
    assert "recovered 0 row(s)" in capsys.readouterr().out


def test_a_missing_explicit_path_is_a_diagnostic_not_an_error(tmp_path, capsys):
    config = temp_config(tmp_path)
    paths = _workdir_paths(config)
    with Store.open(paths.state_db) as store:
        _populate(store, n=1)

    missing = str(tmp_path / "does-not-exist.ndjson")
    assert reconcile_main(["--config", config, "--raw", missing]) == 0
    assert "recovered 0 row(s)" in capsys.readouterr().out
    with Store.open(paths.state_db) as store:
        assert store.events("reconcile_missing_file") != []
