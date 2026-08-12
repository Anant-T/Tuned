import hashlib

import pytest
from pipeline_fakes import build_cfg, open_store, seed_rows

from tuned.data import prompt_registry
from tuned.data.tasks import (
    CURATED_C2_MIX,
    PER_SEED_CAP,
    SYNTHESIS_MIX,
    allocate,
    default_mix,
    parse_mix,
    plan_rows,
    plan_wave,
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


def test_rows_carry_pending_state_after_creation(store, cfg):
    plan_wave(store, cfg, "synthesis", 3)
    row = store.conn.execute("SELECT * FROM task LIMIT 1").fetchone()
    assert row["state"] == "pending"
    assert row["attempts"] == 0
    assert row["stream"] == "synthesis"
