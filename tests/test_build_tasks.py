import hashlib

import pytest
from pipeline_fakes import SOURCE_ID, build_cfg, open_store, paths_for, seed_rows, temp_config

from tuned.data import prompt_registry
from tuned.data.tasks import main as tasks_main
from tuned.data.tasks import (
    CURATED_C2_MIX,
    PER_SEED_CAP,
    SYNTHESIS_MIX,
    allocate,
    default_mix,
    parse_mix,
    plan_rows,
    plan_wave,
    reopen_tasks,
    task_id_for,
)


@pytest.fixture
def cfg():
    return build_cfg()


@pytest.fixture
def store(tmp_path):
    with open_store(tmp_path, n_seeds=12) as s:
        yield s


# --------------------------------------------------------------------------
# allocate: the mix -> slot split.
# --------------------------------------------------------------------------

def test_allocate_parts_sum_to_n():
    for n in (1, 7, 13, 300, 301):
        assert sum(allocate(SYNTHESIS_MIX, n).values()) == n


def test_allocate_follows_the_weights():
    counts = allocate(SYNTHESIS_MIX, 100)
    assert counts == {"irac_analysis": 40, "statute_qa": 25, "drafting": 18, "summarization": 17}


def test_allocate_is_independent_of_dict_order():
    reversed_mix = dict(reversed(list(SYNTHESIS_MIX.items())))
    assert allocate(reversed_mix, 37) == allocate(SYNTHESIS_MIX, 37)


def test_allocate_drops_zero_weight_types():
    assert allocate({"irac_analysis": 1.0, "drafting": 0.0}, 5) == {"irac_analysis": 5}


def test_allocate_rejects_an_empty_mix():
    with pytest.raises(ValueError):
        allocate({"irac_analysis": 0.0}, 5)


def test_allocate_of_zero_is_empty():
    assert allocate(SYNTHESIS_MIX, 0) == {}


def test_default_mix_per_stream():
    assert default_mix("synthesis") == SYNTHESIS_MIX
    assert default_mix("curated_c2") == CURATED_C2_MIX
    assert default_mix("transition") == {"transition": 1.0}
    with pytest.raises(KeyError):
        default_mix("replay")


def test_parse_mix():
    assert parse_mix("irac_analysis=0.5, drafting=0.5") == {
        "irac_analysis": 0.5,
        "drafting": 0.5,
    }
    with pytest.raises(ValueError):
        parse_mix("irac_analysis")


# --------------------------------------------------------------------------
# task identity.
# --------------------------------------------------------------------------

def test_task_id_is_the_documented_hash():
    expected = hashlib.sha256(b"seed001|irac_analysis|gen_irac_analysis_v2|0").hexdigest()[:16]
    assert task_id_for("seed001", "irac_analysis", "gen_irac_analysis_v2", 0) == expected
    assert len(expected) == 16


def test_task_id_separates_every_field():
    base = task_id_for("s", "irac_analysis", "gen_irac_analysis_v1", 0)
    assert base != task_id_for("s2", "irac_analysis", "gen_irac_analysis_v1", 0)
    assert base != task_id_for("s", "statute_qa", "gen_irac_analysis_v1", 0)
    assert base != task_id_for("s", "irac_analysis", "gen_irac_analysis_v2", 0)
    assert base != task_id_for("s", "irac_analysis", "gen_irac_analysis_v1", 1)


# --------------------------------------------------------------------------
# planning.
# --------------------------------------------------------------------------

def test_plan_wave_creates_the_requested_number(store, cfg):
    assert plan_wave(store, cfg, "synthesis", 8) == 8
    assert store.task_counts() == {"pending": 8}


def test_plan_wave_is_idempotent(store, cfg):
    assert plan_wave(store, cfg, "synthesis", 8) == 8
    assert plan_wave(store, cfg, "synthesis", 8) == 0
    assert store.task_counts() == {"pending": 8}


def test_plan_wave_tops_up_to_the_target(store, cfg):
    plan_wave(store, cfg, "synthesis", 6)
    assert plan_wave(store, cfg, "synthesis", 10) == 4
    assert store.task_counts() == {"pending": 10}


def test_plan_rows_is_deterministic(store, cfg):
    first = plan_rows(store, cfg, "synthesis", 9)
    second = plan_rows(store, cfg, "synthesis", 9)
    assert [r["task_id"] for r in first] == [r["task_id"] for r in second]
    assert first == second


def test_plan_rows_are_unique(store, cfg):
    rows = plan_rows(store, cfg, "synthesis", 12)
    assert len({r["task_id"] for r in rows}) == len(rows)


def test_prompt_assignment_matches_the_registry(store, cfg):
    for row in plan_rows(store, cfg, "synthesis", 12):
        expected = prompt_registry.pick_variant(
            row["task_type"], row["seed_id"], row["sample_ix"]
        )
        assert row["prompt_id"] == expected
        assert row["prompt_sha"] == prompt_registry.load(expected).sha
        assert row["task_id"] == task_id_for(
            row["seed_id"], row["task_type"], row["prompt_id"], row["sample_ix"]
        )


def test_wave_follows_the_mix(store, cfg):
    plan_wave(store, cfg, "synthesis", 12)
    rows = store.conn.execute(
        "SELECT task_type, COUNT(*) FROM task GROUP BY task_type"
    ).fetchall()
    by_type = {r[0]: r[1] for r in rows}
    assert by_type == allocate(SYNTHESIS_MIX, 12)


def test_first_wave_spreads_over_seeds_before_resampling(store, cfg):
    rows = plan_rows(store, cfg, "synthesis", 12)
    assert len({r["seed_id"] for r in rows}) == 12
    assert all(r["sample_ix"] == 0 for r in rows)


def test_sample_ix_advances_on_the_next_wave(tmp_path, cfg):
    with open_store(tmp_path, n_seeds=2) as store:
        plan_wave(store, cfg, "synthesis", 2)
        rows = plan_rows(store, cfg, "synthesis", 4)
        assert len(rows) == 2
        # Same seeds, second sample - and a re-derived (possibly different)
        # paraphrase, which is the point of resampling.
        assert {r["seed_id"] for r in rows} == {"seed000", "seed001"}
        assert all(r["sample_ix"] >= 1 for r in rows)


def test_per_seed_cap_bounds_the_wave(tmp_path, cfg):
    with open_store(tmp_path, n_seeds=2) as store:
        created = plan_wave(store, cfg, "synthesis", 40)
        assert created == 2 * PER_SEED_CAP
        counts = store.conn.execute(
            "SELECT seed_id, COUNT(*) FROM task GROUP BY seed_id"
        ).fetchall()
        assert {row[1] for row in counts} == {PER_SEED_CAP}
        # And a further wave adds nothing: every seed is spent.
        assert plan_wave(store, cfg, "synthesis", 80) == 0


def test_sample_ix_is_per_seed_and_task_type(tmp_path, cfg):
    with open_store(tmp_path, n_seeds=1) as store:
        plan_wave(store, cfg, "synthesis", 1, task_type_mix={"irac_analysis": 1.0})
        rows = plan_rows(store, cfg, "synthesis", 2, task_type_mix={"statute_qa": 1.0})
        # One irac task exists; the first statute_qa sample still starts at 0.
        assert rows[0]["sample_ix"] == 0


def test_arm_labels_the_row_and_scopes_the_queue(store, cfg):
    plan_wave(store, cfg, "synthesis", 4, arm="unscripted")
    assert plan_wave(store, cfg, "synthesis", 4, arm="unscripted") == 0
    # A different arm is a different queue, so it plans its own 4.
    assert plan_wave(store, cfg, "synthesis", 4, arm="scripted") == 4
    arms = dict(
        store.conn.execute("SELECT arm, COUNT(*) FROM task GROUP BY arm").fetchall()
    )
    assert arms == {"unscripted": 4, "scripted": 4}


def test_unarmed_and_armed_queues_do_not_shadow_each_other(store, cfg):
    plan_wave(store, cfg, "synthesis", 4, arm="scripted")
    assert plan_wave(store, cfg, "synthesis", 4) == 4


def test_sources_filter(tmp_path, cfg):
    with open_store(tmp_path, n_seeds=3) as store:
        store.upsert_source("other/source", "CC-BY-4.0")
        other = seed_rows(2)
        for row in other:
            row["seed_id"] = row["seed_id"] + "-other"
            row["source_id"] = "other/source"
        store.upsert_seeds(other)
        rows = plan_rows(store, cfg, "synthesis", 5, sources=["other/source"])
        assert {r["seed_id"] for r in rows} == {"seed000-other", "seed001-other"}


def test_empty_store_plans_nothing(tmp_path, cfg):
    with open_store(tmp_path, n_seeds=0) as store:
        assert plan_wave(store, cfg, "synthesis", 10) == 0


def test_unknown_task_type_fails_before_anything_is_written(store, cfg):
    with pytest.raises(KeyError):
        plan_rows(store, cfg, "synthesis", 4, task_type_mix={"haiku": 1.0})
    assert store.task_counts() == {}


def test_transition_stream_plans_transition_tasks(store, cfg):
    plan_wave(store, cfg, "transition", 3)
    types = {r[0] for r in store.conn.execute("SELECT DISTINCT task_type FROM task").fetchall()}
    assert types == {"transition"}


def test_wave_planned_event_is_logged(store, cfg):
    plan_wave(store, cfg, "synthesis", 5)
    events = store.events("wave_planned")
    assert len(events) == 1
    assert '"created": 5' in events[0]["detail_json"]


def test_cli_reports_a_topped_up_queue_rather_than_phantom_skips(tmp_path, cfg, capsys):
    """"skipped 8" reads as "8 tasks were dropped"; the truth is that the
    queue is already at the target."""
    config_path = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    with open_store(tmp_path, n_seeds=12, db_path=paths.state_db):
        pass

    assert tasks_main(["--config", config_path, "--stream", "synthesis", "--n", "8"]) == 0
    first = capsys.readouterr().out
    assert "planned 8  collided 0" in first
    assert "irac_analysis" in first

    assert tasks_main(["--config", config_path, "--stream", "synthesis", "--n", "8"]) == 0
    second = capsys.readouterr().out
    assert "planned 0  (already at target: queue holds 8, target 8)" in second
    assert "skipped" not in second


def test_cli_names_the_per_seed_cap_when_that_is_what_stopped_it(tmp_path, cfg, capsys):
    config_path = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    with open_store(tmp_path, n_seeds=1, db_path=paths.state_db):
        pass
    tasks_main(["--config", config_path, "--stream", "synthesis", "--n", "20"])
    capsys.readouterr()
    tasks_main(["--config", config_path, "--stream", "synthesis", "--n", "20"])
    assert "no seeds under the per-seed cap" in capsys.readouterr().out


# --------------------------------------------------------------------------
# What counts toward the queue target, and what re-opens.
# --------------------------------------------------------------------------

def test_a_rejected_row_stops_counting_toward_the_target(store, cfg):
    """A wave asks for n CANDIDATE rows. A rejected one produced nothing, so
    counting it kept the wave permanently short - and when a bug marked a
    whole wave rejected, replanning reported "already at target" and did
    nothing at all."""
    plan_wave(store, cfg, "synthesis", 4)
    store.conn.execute("UPDATE task SET state = 'rejected' WHERE rowid <= 2")
    assert plan_wave(store, cfg, "synthesis", 4) == 2
    assert store.task_counts() == {"rejected": 2, "pending": 4}


@pytest.mark.parametrize("state", ["gen_unroutable", "judge_unroutable", "judge_error"])
def test_a_parked_row_still_counts_toward_the_target(store, cfg, state):
    """Parked is not lost: re-opening brings the row back with whatever it
    already paid for. Planning a replacement as well would quietly double the
    wave every time the pool had a bad afternoon."""
    plan_wave(store, cfg, "synthesis", 4)
    store.conn.execute("UPDATE task SET state = ? WHERE rowid <= 2", (state,))
    assert plan_wave(store, cfg, "synthesis", 4) == 0


def test_an_accepted_row_still_counts_toward_the_target(store, cfg):
    plan_wave(store, cfg, "synthesis", 4)
    store.conn.execute("UPDATE task SET state = 'accepted'")
    assert plan_wave(store, cfg, "synthesis", 4) == 0


def test_replanning_a_rejected_seed_is_bounded_by_the_per_seed_cap(tmp_path, cfg):
    """The bound on "rejected rows are replaced": _candidate_seeds counts
    tasks per seed regardless of STATE, so a seed whose cap is spent is never
    offered again however many of its rows were rejected. A genuinely bad
    seed costs PER_SEED_CAP tasks, once, and then it is out of the pool."""
    with open_store(tmp_path, n_seeds=1) as store:
        assert plan_wave(store, cfg, "synthesis", PER_SEED_CAP) == PER_SEED_CAP
        store.conn.execute("UPDATE task SET state = 'rejected'")
        # The queue reads as empty and the target is unmet...
        assert plan_wave(store, cfg, "synthesis", PER_SEED_CAP) == 0
        # ...and it stays that way however often the operator re-runs it.
        assert plan_wave(store, cfg, "synthesis", 100) == 0
        assert store.conn.execute("SELECT COUNT(*) FROM task").fetchone()[0] == PER_SEED_CAP


def test_a_chunk_flagged_oversize_is_never_planned_against(tmp_path, cfg):
    """chunks.py writes `meta_json.oversize` on a chunk that is one
    paragraph the packer was forbidden to split and is over the token band -
    measured at 26,818 tokens on one real judgment. It was written from the
    first cut and read by nothing, so the planner would happily build a
    prompt around it: exactly the budget blowout chunking exists to prevent.
    Both directions - the ordinary chunk beside it is still selected."""
    import json as _json

    with open_store(tmp_path, n_seeds=0) as store:
        store.upsert_seeds([
            {"seed_id": "ok1", "source_id": SOURCE_ID, "text": "t", "token_count": 1200,
             "case_type": "bail", "code_era": "bns",
             "meta_json": _json.dumps({"kind": "chunk", "oversize": False})},
            {"seed_id": "big1", "source_id": SOURCE_ID, "text": "t", "token_count": 40003,
             "case_type": "bail", "code_era": "bns",
             "meta_json": _json.dumps({"kind": "chunk", "oversize": True})},
            {"seed_id": "plain", "source_id": SOURCE_ID, "text": "t", "token_count": 900,
             "case_type": "bail", "code_era": "bns"},  # no meta at all
        ])
        assert plan_wave(store, cfg, "synthesis", 12) > 0
        planned = {r[0] for r in store.conn.execute("SELECT DISTINCT seed_id FROM task")}
        assert "big1" not in planned
        assert {"ok1", "plain"} <= planned


def test_a_seed_whose_meta_is_not_json_is_still_planned_against(tmp_path, cfg):
    # The `json_valid` guard: meta_json is a free-text column, and a row
    # holding something that is not JSON must read as un-flagged rather than
    # failing the planner's whole query.
    with open_store(tmp_path, n_seeds=0) as store:
        store.upsert_seeds([
            {"seed_id": "weird", "source_id": SOURCE_ID, "text": "t", "token_count": 900,
             "case_type": "bail", "code_era": "bns", "meta_json": "not json at all"},
        ])
        assert plan_wave(store, cfg, "synthesis", 2) > 0


def test_reopen_returns_parked_rows_to_the_queue_that_owns_them(store, cfg):
    plan_wave(store, cfg, "synthesis", 3)
    ids = [r[0] for r in store.conn.execute("SELECT task_id FROM task ORDER BY rowid").fetchall()]
    store.set_task_state(ids[0], "gen_unroutable", "unroutable:generator")
    store.set_task_state(ids[1], "judge_unroutable", "judge-b-unroutable:x")
    store.set_task_state(ids[2], "rejected", "reject:citations")

    assert reopen_tasks(store, ["gen_unroutable", "judge_unroutable"]) == {
        "gen_unroutable": 1,
        "judge_unroutable": 1,
    }
    states = dict(
        store.conn.execute("SELECT task_id, state FROM task").fetchall()
    )
    assert states[ids[0]] == "pending"
    assert states[ids[1]] == "judging"
    # A rejected row is a decision, not a park: it is never re-opened.
    assert states[ids[2]] == "rejected"
    assert store.events("tasks_reopened")


def test_reopen_refuses_a_state_it_does_not_own(store, cfg):
    with pytest.raises(ValueError, match="rejected"):
        reopen_tasks(store, ["rejected"])


def test_reopen_cli_reports_what_it_re_opened(tmp_path, cfg, capsys):
    config_path = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    with open_store(tmp_path, n_seeds=4, db_path=paths.state_db) as store:
        plan_wave(store, cfg, "synthesis", 2)
        store.conn.execute("UPDATE task SET state = 'judge_unroutable'")

    assert tasks_main(["--config", config_path, "--reopen", "judge_unroutable"]) == 0
    out = capsys.readouterr().out
    assert "re-opened 2" in out
    assert "judge_unroutable -> judging" in out


def test_a_reopened_row_gets_its_attempt_budget_back(store, cfg):
    """R3-C1. The rows --reopen exists for parked AT the attempt cap: the
    motivating case is a wave that could not route at all (no key, everything
    cooling, the daily cap spent), and that park happens at attempts ==
    MAX_ATTEMPTS. Restoring only the state hands the row back already
    exhausted, so the first failure after the operator fixes the cause is
    terminal - and `rejected` is not re-openable."""
    plan_wave(store, cfg, "synthesis", 2)
    ids = [r[0] for r in store.conn.execute("SELECT task_id FROM task ORDER BY rowid").fetchall()]
    store.conn.execute("UPDATE task SET attempts = 8")
    store.set_task_state(ids[0], "gen_unroutable", "exhausted:unroutable:missing-key")
    store.set_task_state(ids[1], "judge_error", "judge-slot-a:boom")

    reopen_tasks(store, ["gen_unroutable", "judge_error"])

    rows = {
        r["task_id"]: r
        for r in store.conn.execute("SELECT task_id, state, attempts FROM task").fetchall()
    }
    assert (rows[ids[0]]["state"], rows[ids[0]]["attempts"]) == ("pending", 0)
    assert (rows[ids[1]]["state"], rows[ids[1]]["attempts"]) == ("judging", 0)


def test_reopen_covers_every_stream_unless_one_is_named(store, cfg):
    """The recovery command printed in the config TODO is bare `--reopen
    judge_unroutable`; when that silently meant `--stream synthesis` it left
    the curated_c2 and transition rows of the same wave parked."""
    plan_wave(store, cfg, "synthesis", 1)
    plan_wave(store, cfg, "curated_c2", 1)
    store.conn.execute("UPDATE task SET state = 'judge_unroutable'")

    assert reopen_tasks(store, ["judge_unroutable"]) == {"judge_unroutable": 2}
    assert store.task_counts() == {"judging": 2}

    store.conn.execute("UPDATE task SET state = 'judge_unroutable'")
    assert reopen_tasks(store, ["judge_unroutable"], stream="synthesis") == {
        "judge_unroutable": 1
    }
    assert store.task_counts() == {"judging": 1, "judge_unroutable": 1}


def test_reopen_refuses_a_planning_stream_rather_than_ignoring_it(tmp_path, cfg, capsys):
    """`--reopen judge_unroutable --stream transition` re-opened EVERY stream
    while naming one. --stream is the planner's; the filter is
    --reopen-stream, and the CLI now says so instead of acting on neither."""
    config_path = temp_config(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        tasks_main(
            ["--config", config_path, "--reopen", "judge_unroutable", "--stream", "transition"]
        )
    assert excinfo.value.code == 2
    assert "--reopen-stream" in capsys.readouterr().err


def test_reopen_and_plan_a_named_stream_in_one_command(tmp_path, cfg, capsys):
    """--stream is honoured by the PLANNER and ignored by the re-open, so a
    command that does BOTH is unambiguous: re-open every parked row, then plan
    this stream. `--reopen judge_unroutable --n 3 --stream transition` did
    exactly that until the round-4 guard refused it, and there is otherwise no
    way to re-open and plan a non-default stream in one command. The guard is
    for the case where --stream can only be read as a filter it does not
    apply: a re-open with nothing to plan."""
    config_path = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    with open_store(tmp_path, n_seeds=6, db_path=paths.state_db) as store:
        plan_wave(store, cfg, "synthesis", 2)
        store.conn.execute("UPDATE task SET state = 'judge_unroutable'")

    argv = ["--config", config_path, "--reopen", "judge_unroutable", "--n", "3",
            "--stream", "transition"]
    assert tasks_main(argv) == 0
    out = capsys.readouterr().out
    assert "re-opened 2" in out
    assert "stream=transition" in out

    with open_store(tmp_path, n_seeds=0, db_path=paths.state_db) as store:
        # The two synthesis rows came back to the judge queue and three
        # transition rows were planned - neither half swallowed the other.
        assert store.task_counts() == {"judging": 2, "pending": 3}
        streams = dict(
            store.conn.execute("SELECT stream, COUNT(*) FROM task GROUP BY stream").fetchall()
        )
        assert streams == {"synthesis": 2, "transition": 3}


def test_an_unfiltered_reopen_does_not_report_a_filter_it_did_not_have(tmp_path, cfg, capsys):
    """"STILL PARKED (not in --reopen-stream None)" told the operator their
    unfiltered command had a filter. Only a filter can leave a residue."""
    config_path = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    with open_store(tmp_path, n_seeds=6, db_path=paths.state_db) as store:
        plan_wave(store, cfg, "synthesis", 2)
        plan_wave(store, cfg, "transition", 1)
        store.conn.execute("UPDATE task SET state = 'judge_unroutable'")

    assert tasks_main(["--config", config_path, "--reopen", "judge_unroutable"]) == 0
    assert "STILL PARKED" not in capsys.readouterr().out

    # ...and a filtered one names the filter and the rows it left behind.
    with open_store(tmp_path, n_seeds=0, db_path=paths.state_db) as store:
        store.conn.execute("UPDATE task SET state = 'judge_unroutable'")
    assert (
        tasks_main(
            [
                "--config", config_path, "--reopen", "judge_unroutable",
                "--reopen-stream", "synthesis",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "STILL PARKED (not in --reopen-stream 'synthesis'): transition=1" in out


def test_reopen_cli_names_the_streams_it_touched(tmp_path, cfg, capsys):
    config_path = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    with open_store(tmp_path, n_seeds=6, db_path=paths.state_db) as store:
        plan_wave(store, cfg, "synthesis", 2)
        plan_wave(store, cfg, "transition", 1)
        store.conn.execute("UPDATE task SET state = 'gen_unroutable'")

    assert tasks_main(["--config", config_path, "--reopen", "gen_unroutable"]) == 0
    out = capsys.readouterr().out
    assert "re-opened 3" in out
    assert "synthesis" in out and "transition" in out


def test_rows_carry_pending_state_after_creation(store, cfg):
    plan_wave(store, cfg, "synthesis", 3)
    row = store.conn.execute("SELECT * FROM task LIMIT 1").fetchone()
    assert row["state"] == "pending"
    assert row["attempts"] == 0
    assert row["stream"] == "synthesis"
