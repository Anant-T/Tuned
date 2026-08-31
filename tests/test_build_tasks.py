import hashlib
import json

import pytest
from pipeline_fakes import (
    SEED_TEXT,
    SOURCE_ID,
    STATUTE_SECTION_TEXT,
    add_transition_seeds,
    build_cfg,
    open_store,
    paths_for,
    seed_rows,
    temp_config,
)

from tuned.data import prompt_registry
from tuned.data.tasks import main as tasks_main
from tuned.data.generate import MAX_ATTEMPTS
from tuned.data.tasks import (
    CURATED_C2_MIX,
    FREE_PARK_DISPOSITIONS,
    PER_SEED_CAP,
    REPLY_RESERVE_TOKENS,
    REOPEN_STATES,
    SYNTHESIS_MIX,
    TERMINALLY_DEAD,
    allocate,
    default_mix,
    parse_mix,
    plan_rows,
    plan_wave,
    reopen_tasks,
    seed_token_budget,
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
    assert counts == {"irac_analysis": 40, "statute_qa": 25, "summarization": 35}


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


def test_planner_does_not_assign_statute_qa_to_an_ineligible_seed(tmp_path, cfg):
    distinct = (
        "Section 34. When a criminal act is done by several persons in "
        "furtherance of the common intention of all, each is liable."
    )
    with open_store(tmp_path, n_seeds=0) as store:
        store.upsert_seeds(
            [
                {
                    "seed_id": "eligible-statute",
                    "source_id": SOURCE_ID,
                    "native_id": "ok",
                    "case_type": "criminal",
                    "code_era": "ipc",
                    "text": SEED_TEXT,
                    "token_count": len(SEED_TEXT) // 4,
                    "meta_json": {"section_text": distinct},
                },
                {
                    "seed_id": "ineligible-blank",
                    "source_id": SOURCE_ID,
                    "native_id": "blank",
                    "case_type": "criminal",
                    "code_era": "ipc",
                    "text": SEED_TEXT,
                    "token_count": len(SEED_TEXT) // 4,
                    "meta_json": {"section_text": ""},
                },
                {
                    "seed_id": "ineligible-equal",
                    "source_id": SOURCE_ID,
                    "native_id": "equal",
                    "case_type": "criminal",
                    "code_era": "ipc",
                    "text": SEED_TEXT,
                    "token_count": len(SEED_TEXT) // 4,
                    "meta_json": {"section_text": SEED_TEXT},
                },
                {
                    "seed_id": "ineligible-missing",
                    "source_id": SOURCE_ID,
                    "native_id": "missing",
                    "case_type": "criminal",
                    "code_era": "ipc",
                    "text": SEED_TEXT,
                    "token_count": len(SEED_TEXT) // 4,
                    "meta_json": {},
                },
            ]
        )
        rows = plan_rows(
            store, cfg, "synthesis", 4, task_type_mix={"statute_qa": 1.0}
        )
        assert rows
        assert {row["seed_id"] for row in rows} == {"eligible-statute"}


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
    meta = {
        "section_text": (
            "Section 34. When a criminal act is done by several persons in "
            "furtherance of the common intention of all, each is liable."
        )
    }
    with open_store(tmp_path, n_seeds=1, meta=meta) as store:
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
    add_transition_seeds(store, 3)
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


def test_seed_token_budget_leaves_room_for_the_assistant_turn(cfg):
    """The budget is the train cap minus what the reply needs, never negative.

    Pinned as arithmetic rather than a literal so that moving
    train.main.max_seq_length moves the planner with it - the two numbers
    describe one row and drifting apart is what puts a doomed task in the
    queue.
    """
    assert seed_token_budget(cfg) == cfg.max_seq_length - REPLY_RESERVE_TOKENS

    class _Tiny:
        max_seq_length = 100

    assert seed_token_budget(_Tiny) == 0


def test_a_seed_too_long_to_leave_reply_room_is_never_planned_against(tmp_path, cfg):
    """The fourth seed-level exclusion, and the same shape as the other three.

    assemble.py drops a row whose RENDERED length exceeds max_seq_length -
    at the far end of the pipeline, after the teacher has already been paid
    for the generation. Measured 2026-08-26 over 1,368 real generations, the
    16 rows dropped that way averaged a 7,597-token seed against a 492-token
    trace, and every one came from PredEx or TathyaNyaya, whose seeds are
    never chunked. So the drop is silent where it costs a call, and BIASED -
    what it removes is the longest, most substantive cases.

    Both directions: the seed one token inside the budget is still selected,
    and so is the seed that records no length at all, which stays eligible
    on the same contract as the other exclusions ("a seed that declares
    NOTHING stays eligible").
    """
    budget = seed_token_budget(cfg)
    with open_store(tmp_path, n_seeds=0) as store:
        store.upsert_seeds([
            {"seed_id": "fits", "source_id": SOURCE_ID, "text": "t",
             "token_count": budget, "case_type": "bail", "code_era": "bns"},
            {"seed_id": "over", "source_id": SOURCE_ID, "text": "t",
             "token_count": budget + 1, "case_type": "bail", "code_era": "bns"},
            {"seed_id": "way_over", "source_id": SOURCE_ID, "text": "t",
             "token_count": 50_369, "case_type": "bail", "code_era": "bns"},
            {"seed_id": "unmeasured", "source_id": SOURCE_ID, "text": "t",
             "token_count": None, "case_type": "bail", "code_era": "bns"},
        ])
        assert plan_wave(store, cfg, "synthesis", 12) > 0
        planned = {r[0] for r in store.conn.execute("SELECT DISTINCT seed_id FROM task")}
        assert "over" not in planned
        assert "way_over" not in planned
        assert {"fits", "unmeasured"} <= planned


def test_a_seed_that_declares_a_stream_is_only_planned_into_that_stream(tmp_path, cfg):
    """The third seed-level exclusion, and the same shape as the other two: a
    property the SEED states about itself, honoured by the planner whoever
    called it.

    A seed whose meta carries scenario/provisions/answer key for one stream is
    meaningless in another - the wave would put its question without its
    grounding, and the stream-specific gate would skip the row it was written
    for. Both directions here, plus the seed that declares nothing.
    """
    import json as _json

    with open_store(tmp_path, n_seeds=0) as store:
        store.upsert_seeds([
            {"seed_id": "trans1", "source_id": SOURCE_ID, "text": "t", "token_count": 900,
             "case_type": "criminal", "code_era": "ipc",
             "meta_json": _json.dumps({"stream": "transition", "question": "which code?"})},
            {"seed_id": "plain1", "source_id": SOURCE_ID, "text": "t", "token_count": 900,
             "case_type": "bail", "code_era": "bns"},
        ])
        assert plan_wave(store, cfg, "synthesis", 6) > 0
        planned = {r[0] for r in store.conn.execute("SELECT DISTINCT seed_id FROM task")}
        assert "trans1" not in planned
        assert "plain1" in planned

        # ...and its own wave still draws it, or the exclusion above would be
        # satisfied by a planner that plans nothing.
        assert plan_wave(store, cfg, "transition", 2, task_type_mix={"transition": 1.0}) > 0
        own = {
            r[0] for r in store.conn.execute(
                "SELECT DISTINCT seed_id FROM task WHERE stream = 'transition'"
            )
        }
        assert "trans1" in own


def test_a_closed_world_stream_refuses_a_seed_that_declares_nothing(tmp_path, cfg):
    """THE TRANSITION MASSACRE, 2026-08-28: 2,063 of one 2,200-row wave died
    as `skip:slots` before a single teacher call.

    The exclusion above is ONE-DIRECTIONAL. `COALESCE(meta.stream, ?) = ?`
    keeps a transition seed out of a synthesis wave, but a seed that declares
    nothing satisfies it in EVERY wave - including transition's, whose
    build_slots needs scenario/old_section_text/new_section_text/savings_text
    and both dates. A generic corpus chunk carries none of them, so the row
    can only ever skip.

    "Declares nothing stays eligible for any wave" is the right contract for
    an open-world stream, where the task type is built out of the seed's TEXT.
    It is the wrong one for a stream whose task type is built out of the
    seed's META, because there the absence of a declaration is not neutrality
    - it is a guarantee the slots cannot render. Hence the stream, not the
    source id: `tuned/law-v1-transition-grid` stays transition.py's business.
    """
    import json as _json

    with open_store(tmp_path, n_seeds=0) as store:
        store.upsert_seeds([
            {"seed_id": "trans1", "source_id": SOURCE_ID, "text": "t", "token_count": 900,
             "case_type": "criminal", "code_era": "ipc",
             "meta_json": _json.dumps({"stream": "transition", "question": "which code?"})},
            {"seed_id": "plain1", "source_id": SOURCE_ID, "text": "t", "token_count": 900,
             "case_type": "bail", "code_era": "bns"},
        ])
        # More rows than the one eligible seed can carry, so an unfixed planner
        # HAS to reach for plain1 to fill the wave.
        plan_wave(store, cfg, "transition", 6, task_type_mix={"transition": 1.0})
        planned = {
            r[0] for r in store.conn.execute(
                "SELECT DISTINCT seed_id FROM task WHERE stream = 'transition'"
            )
        }
        assert "trans1" in planned
        assert "plain1" not in planned, (
            "a stream-less seed cannot render transition slots; planning it "
            "burns a wave slot and its per-seed cap on a guaranteed skip"
        )


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


def test_format_parked_reopens_to_pending_and_is_not_terminally_dead():
    assert REOPEN_STATES["format_parked"] == "pending"
    assert "format_parked" not in TERMINALLY_DEAD


def test_input_ineligible_reopens_to_pending_and_is_terminally_dead():
    assert REOPEN_STATES["input_ineligible"] == "pending"
    assert "input_ineligible" in TERMINALLY_DEAD


def test_input_ineligible_reopens_with_a_fresh_attempt_budget(tmp_path, cfg):
    with open_store(tmp_path, n_seeds=4) as store:
        assert plan_wave(store, cfg, "synthesis", 1) == 1
        task_id = store.conn.execute("SELECT task_id FROM task").fetchone()[0]
        store.conn.execute(
            "UPDATE task SET attempts = ? WHERE task_id = ?",
            (MAX_ATTEMPTS, task_id),
        )
        store.set_task_state(
            task_id, "input_ineligible", "input-ineligible:section_text"
        )
        skipped: dict[str, int] = {}
        assert reopen_tasks(store, ["input_ineligible"], skipped=skipped) == {
            "input_ineligible": 1
        }
        assert skipped == {}
        row = dict(store.conn.execute("SELECT * FROM task").fetchone())
        assert row["state"] == "pending"
        assert row["attempts"] == 0


def _plant_legacy_skip_slots(store, seed_id="seed000", task_type="statute_qa"):
    prompt_id = prompt_registry.pick_variant(task_type, seed_id, 0)
    store.create_tasks(
        [
            {
                "task_id": task_id_for(seed_id, task_type, prompt_id, 0),
                "seed_id": seed_id,
                "stream": "synthesis",
                "task_type": task_type,
                "prompt_id": prompt_id,
                "prompt_sha": prompt_registry.load(prompt_id).sha,
                "sample_ix": 0,
            }
        ]
    )
    task_id = store.conn.execute("SELECT task_id FROM task").fetchone()[0]
    store.set_task_state(task_id, "rejected", "skip:slots")
    return task_id


def test_legacy_skip_slots_zero_generation_does_not_block_the_per_seed_cap(tmp_path, cfg):
    with open_store(tmp_path, n_seeds=1, meta={"section_text": None}) as store:
        task_id = _plant_legacy_skip_slots(store)
        assert store.latest_generation(task_id) is None
        store.conn.execute(
            "UPDATE seed SET meta_json = ?",
            (json.dumps({"section_text": STATUTE_SECTION_TEXT}),),
        )
        store.conn.commit()
        assert plan_wave(
            store,
            cfg,
            "synthesis",
            PER_SEED_CAP,
            task_type_mix={"irac_analysis": 1.0},
        ) == PER_SEED_CAP
        row = dict(
            store.conn.execute(
                "SELECT * FROM task WHERE task_id = ?", (task_id,)
            ).fetchone()
        )
        assert row["state"] == "rejected"
        assert row["disposition"] == "skip:slots"
        assert store.latest_generation(task_id) is None


def test_legacy_skip_slots_generated_row_does_not_block_the_per_seed_cap(tmp_path, cfg):
    with open_store(tmp_path, n_seeds=1, meta={"section_text": None}) as store:
        task_id = _plant_legacy_skip_slots(store)
        store.record_generation(
            {
                "kind": "generation",
                "task_id": task_id,
                "attempt": 1,
                "provider": "cerebras",
                "model": "gpt-oss-120b",
                "think": "reasoning...",
                "answer": "Issue\nX\n\nConclusion\nY",
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "raw_path": str(tmp_path / "raw" / "gen.ndjson"),
                "raw_offset": 0,
            }
        )
        store.set_task_state(task_id, "rejected", "skip:slots")
        assert store.latest_generation(task_id) is not None
        store.conn.execute(
            "UPDATE seed SET meta_json = ?",
            (json.dumps({"section_text": STATUTE_SECTION_TEXT}),),
        )
        store.conn.commit()
        assert plan_wave(
            store,
            cfg,
            "synthesis",
            PER_SEED_CAP,
            task_type_mix={"irac_analysis": 1.0},
        ) == PER_SEED_CAP
        row = dict(
            store.conn.execute(
                "SELECT * FROM task WHERE task_id = ?", (task_id,)
            ).fetchone()
        )
        assert row["state"] == "rejected"
        assert row["disposition"] == "skip:slots"
        assert store.latest_generation(task_id) is not None


@pytest.mark.parametrize("task_type", ("irac_analysis", "transition"))
def test_non_statute_skip_slots_still_spends_the_per_seed_cap(tmp_path, cfg, task_type):
    with open_store(tmp_path, n_seeds=1) as store:
        task_id = _plant_legacy_skip_slots(store, task_type=task_type)
        added = plan_wave(
            store,
            cfg,
            "synthesis",
            PER_SEED_CAP,
            task_type_mix={"irac_analysis": 1.0},
        )
        assert added == PER_SEED_CAP - 1
        total = store.conn.execute("SELECT COUNT(*) FROM task").fetchone()[0]
        assert total == PER_SEED_CAP
        row = dict(
            store.conn.execute(
                "SELECT * FROM task WHERE task_id = ?", (task_id,)
            ).fetchone()
        )
        assert row["state"] == "rejected"
        assert row["disposition"] == "skip:slots"
        assert row["task_type"] == task_type


def test_input_ineligible_does_not_block_the_per_seed_cap(tmp_path, cfg):
    with open_store(tmp_path, n_seeds=1, meta={"section_text": None}) as store:
        assert plan_wave(
            store, cfg, "synthesis", 1, task_type_mix={"irac_analysis": 1.0}
        ) == 1
        task_id = store.conn.execute("SELECT task_id FROM task").fetchone()[0]
        store.set_task_state(
            task_id, "input_ineligible", "input-ineligible:section_text"
        )
        assert plan_wave(
            store,
            cfg,
            "synthesis",
            PER_SEED_CAP,
            task_type_mix={"irac_analysis": 1.0},
        ) == PER_SEED_CAP
        total = store.conn.execute("SELECT COUNT(*) FROM task").fetchone()[0]
        assert total == PER_SEED_CAP + 1


def test_exhausted_format_park_reopens_with_a_fresh_attempt_budget(tmp_path, cfg):
    with open_store(tmp_path, n_seeds=40) as store:
        assert plan_wave(store, cfg, "synthesis", 4) == 4
        ids = [r[0] for r in store.conn.execute("SELECT task_id FROM task ORDER BY rowid")]
        store.conn.execute(
            "UPDATE task SET attempts = ? WHERE task_id = ?", (MAX_ATTEMPTS, ids[0])
        )
        store.set_task_state(ids[0], "format_parked", "exhausted:format:irac_placement")
        store.set_task_state(ids[1], "rejected", "reject:citations")
        store.conn.commit()

        # A recoverable park holds its slot; a rejected legal failure does not.
        assert plan_wave(store, cfg, "synthesis", 4) == 1

        skipped: dict[str, int] = {}
        assert reopen_tasks(store, ["format_parked"], skipped=skipped) == {"format_parked": 1}
        assert skipped == {}
        row = dict(
            store.conn.execute("SELECT * FROM task WHERE task_id = ?", (ids[0],)).fetchone()
        )
        assert row["state"] == "pending"
        assert row["attempts"] == 0
        rejected = dict(
            store.conn.execute("SELECT * FROM task WHERE task_id = ?", (ids[1],)).fetchone()
        )
        assert rejected["state"] == "rejected"


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
        # ...and something the transition half of the command can be planned
        # against, since a closed-world stream draws only declared seeds.
        add_transition_seeds(store, 3)

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
        add_transition_seeds(store, 1)
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
        add_transition_seeds(store, 1)
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


# --------------------------------------------------------------------------
# Terminally-dead states, queue capacity, and the re-open budget.
# --------------------------------------------------------------------------

def test_cancelled_rows_do_not_occupy_queue_capacity(tmp_path, cfg):
    """N4. A stale-prompt cancellation must not make the queue look full.

    _existing_in_queue counted every state except `rejected`, so the 419 rows
    the 2026-08-18 purge moved to `stale_prompt` still held their slots even
    though no worker can ever claim one. `n` is a TARGET, so the next plan
    topped up to the difference instead of the number asked for: measured on
    the live store, `plan_wave(..., 500)` produced 281 - exactly 500 - 219
    already-counted rows in the unarmed queue - and an A/B arm of 100 against
    100 cancelled rows would have produced ZERO.

    This is a QUEUE-COUNTING bug, not a seed-supply one. The store had 60,103
    seeds with no task at all; seeds were never the binding constraint.
    """
    with open_store(tmp_path, n_seeds=40) as store:
        assert plan_wave(store, cfg, "synthesis", 10) == 10
        store.conn.execute("UPDATE task SET state = 'stale_prompt'")
        store.conn.commit()

        # The full ask, not the difference.
        assert plan_wave(store, cfg, "synthesis", 10) == 10
        live = store.conn.execute(
            "SELECT COUNT(*) FROM task WHERE state = 'pending'"
        ).fetchone()[0]
        assert live == 10


def test_rejected_and_stale_are_both_terminally_dead(tmp_path, cfg):
    """The two dead states behave identically for queue capacity, and the
    recoverable parks still hold their slots - a parked row keeps whatever it
    has already paid for, so planning a replacement would double the wave."""
    with open_store(tmp_path, n_seeds=40) as store:
        assert plan_wave(store, cfg, "synthesis", 6) == 6
        ids = [r[0] for r in store.conn.execute("SELECT task_id FROM task ORDER BY rowid")]
        store.set_task_state(ids[0], "rejected", "reject:citations")
        store.set_task_state(ids[1], "stale_prompt", "stale-prompt:x:a!=b")
        store.set_task_state(ids[2], "gen_unroutable", "unroutable:generator")

        # Two dead slots are refilled; the parked one is not.
        assert plan_wave(store, cfg, "synthesis", 6) == 2


def test_reopen_does_not_refund_attempts_a_cancellation_swept_up(tmp_path, cfg):
    """N6. The purge folded 6 genuinely-exhausted rows into `stale_prompt`.

    Those rows had spent three real, billed generations each; `stale_prompt` is
    a parking state, and under a blanket reset `--reopen stale_prompt` would
    have handed all three attempts back to every one of them. The rule is now
    the other way round: attempts are PRESERVED unless the disposition says the
    park cost nothing, and a row already at the cap is not re-opened at all.
    """
    with open_store(tmp_path, n_seeds=40) as store:
        assert plan_wave(store, cfg, "synthesis", 8) == 8
        ids = [r[0] for r in store.conn.execute("SELECT task_id FROM task ORDER BY rowid")]
        # Six exhausted rows swept into the cancellation...
        for task_id in ids[:6]:
            store.conn.execute(
                "UPDATE task SET attempts = ? WHERE task_id = ?", (MAX_ATTEMPTS, task_id)
            )
            store.set_task_state(task_id, "stale_prompt", "stale-prompt:p:a!=b")
        # ...and two that never got that far.
        for task_id in ids[6:]:
            store.conn.execute(
                "UPDATE task SET attempts = 1 WHERE task_id = ?", (task_id,)
            )
            store.set_task_state(task_id, "stale_prompt", "stale-prompt:p:a!=b")
        store.conn.commit()

        skipped: dict[str, int] = {}
        counts = reopen_tasks(store, ["stale_prompt"], skipped=skipped)

        assert counts == {"stale_prompt": 2}
        assert skipped == {"stale_prompt": 6}
        rows = {
            r["task_id"]: r
            for r in store.conn.execute("SELECT task_id, state, attempts FROM task")
        }
        # The exhausted six stay put, with their spend intact.
        for task_id in ids[:6]:
            assert (rows[task_id]["state"], rows[task_id]["attempts"]) == (
                "stale_prompt",
                MAX_ATTEMPTS,
            )
        # The two with budget left come back, and keep what they had spent.
        for task_id in ids[6:]:
            assert (rows[task_id]["state"], rows[task_id]["attempts"]) == ("pending", 1)


def test_reopen_reports_a_budget_skip_distinctly_from_an_empty_filter(tmp_path, cfg):
    """N5. A bare 0 cannot mean two different things.

    "the filter matched nothing" wants a different filter; "every match was out
    of budget" wants a re-plan. Both used to print the same line.
    """
    with open_store(tmp_path, n_seeds=40) as store:
        assert plan_wave(store, cfg, "synthesis", 1) == 1
        task_id = store.conn.execute("SELECT task_id FROM task").fetchone()[0]
        store.conn.execute("UPDATE task SET attempts = ?", (MAX_ATTEMPTS,))
        store.set_task_state(task_id, "gen_unroutable", "unroutable:no-reasoning-channel")
        store.conn.commit()

        exhausted_skip: dict[str, int] = {}
        assert reopen_tasks(store, ["gen_unroutable"], skipped=exhausted_skip) == {
            "gen_unroutable": 0
        }
        assert exhausted_skip == {"gen_unroutable": 1}

        # Same zero, nothing matched: the skip map stays empty.
        empty_skip: dict[str, int] = {}
        assert reopen_tasks(store, ["judge_error"], skipped=empty_skip) == {
            "judge_error": 0
        }
        assert empty_skip == {}


def test_off_teacher_reopens_to_the_generator_and_is_not_terminally_dead():
    """Back to the GENERATOR, not the judges: the recovery this park exists
    for is re-generating the row with the teacher the cut is defined over, not
    accepting the old teacher's answer after all.

    NOT terminally dead, for the same reason format_parked is not - the row is
    re-openable and keeps the judgements it already paid for, so counting it
    dead would have plan_wave top up a replacement as well and quietly double
    the wave.
    """
    assert REOPEN_STATES["off_teacher"] == "pending"
    assert "off_teacher" not in TERMINALLY_DEAD


def test_off_teacher_is_the_one_billed_park_that_gets_its_budget_back():
    """The default is to preserve, and this is the deliberate exception.

    The attempts were spent producing an answer a later RULING removed from
    the cut - a fact about the fleet, not about the answer - and the row
    cannot be replaced by any other route, task_id_for hashing (seed,
    task_type, prompt_id, sample_ix) so a re-plan is INSERT OR IGNORE'd back
    into the row it was meant to replace. Handed back at MAX_ATTEMPTS it is a
    park that is terminal in everything but name.
    """
    from tuned.data.verify import OFF_TEACHER_DISPOSITION

    assert OFF_TEACHER_DISPOSITION in FREE_PARK_DISPOSITIONS
    # ...and the other two entries stay what they were: parks that never
    # reached a provider at all.
    assert FREE_PARK_DISPOSITIONS == {
        "unroutable:generator", "exhausted:provider-fault", "verify:off-teacher",
    }


@pytest.mark.parametrize(
    "section_text",
    [
        None,                      # no key at all
        "",                        # present and empty
        "   ",                     # spaces - SQLite TRIM strips these
        "\t\n",                    # tabs/newlines - TRIM does NOT
        SEED_TEXT,                 # present, but not distinct from the seed
        STATUTE_SECTION_TEXT,      # the genuine article
    ],
)
def test_the_sql_statute_hint_never_hides_a_seed_the_real_predicate_accepts(
    tmp_path, section_text
):
    """_candidate_seeds answers the CHEAP half of the statute-QA test in the
    query that already visited the row, so the planner stops re-reading every
    candidate one point-lookup at a time.

    The direction is the whole safety argument. SQLite's TRIM strips spaces
    but not tabs or newlines, while the real predicate collapses all
    whitespace - so the hint is allowed to say "maybe" where
    statute_qa_section_eligible says no, and never the reverse. This asserts
    exactly that implication over every shape the column can hold.
    """
    from tuned.data.generate import seed_meta, statute_qa_section_eligible
    from tuned.data.tasks import _candidate_seeds

    meta = {} if section_text is None else {"section_text": section_text}
    store = open_store(tmp_path, n_seeds=1, meta=meta)
    if section_text is None:  # seed_rows always writes one; take it back out
        store.upsert_seeds([dict(seed_rows(1)[0], meta_json={})])
    try:
        candidates = _candidate_seeds(
            store, limit=10, sources=None, stream="synthesis", max_seed_tokens=10**6
        )
        assert len(candidates) == 1
        seed_id, _, maybe = candidates[0]

        seed = store.get_seed(seed_id)
        truth = statute_qa_section_eligible(
            seed.get("text") or "", seed_meta(seed).get("section_text")
        )
        assert not truth or maybe, (
            f"the hint hid a seed the real predicate accepts ({section_text!r})"
        )
        if section_text == STATUTE_SECTION_TEXT:
            assert truth and maybe  # the case that must survive both
    finally:
        store.close()


# --------------------------------------------------------------------------
# --variant: planning a wave on chosen templates.
#
# The default spread over every paraphrase is a randomised trial, and the
# 2026-08-31 read of it separated the personas that pay from the ones that do
# not (gen_irac_analysis_v4: 15.6 generations per accepted row, against 3.3
# for v1). Acting on that means planning the next wave on the winners. It must
# not mean DELETING the losers - task_id_for does not hash prompt_sha, so
# changing the pool re-maps every pending row's draw and parks the queue as
# stale_prompt. Hence an allowlist that binds new rows only.
# --------------------------------------------------------------------------

def test_a_variant_allowlist_pins_the_templates_a_wave_draws(store, cfg):
    rows = plan_rows(store, cfg, "synthesis", 12, variants=["gen_irac_analysis_v1"])
    irac = [r for r in rows if r["task_type"] == "irac_analysis"]
    assert irac, "the synthesis mix must plan some irac_analysis to test this"
    assert {r["prompt_id"] for r in irac} == {"gen_irac_analysis_v1"}
    for row in rows:
        assert row["prompt_sha"] == prompt_registry.load(row["prompt_id"]).sha
        assert row["task_id"] == task_id_for(
            row["seed_id"], row["task_type"], row["prompt_id"], row["sample_ix"]
        )


def test_the_allowlist_binds_only_the_task_types_it_names(store, cfg):
    """Restricting irac must not pin summarization to whatever it happens to
    list first - an operator narrows one task type at a time."""
    rows = plan_rows(store, cfg, "synthesis", 12, variants=["gen_irac_analysis_v1"])
    others = [r for r in rows if r["task_type"] != "irac_analysis"]
    assert others
    for row in others:
        assert row["prompt_id"] == prompt_registry.pick_variant(
            row["task_type"], row["seed_id"], row["sample_ix"]
        )


def test_no_allowlist_plans_exactly_what_it_planned_before(store, cfg):
    assert plan_rows(store, cfg, "synthesis", 9) == plan_rows(
        store, cfg, "synthesis", 9, variants=None
    )


def test_an_unknown_variant_fails_before_anything_is_written(store, cfg):
    with pytest.raises(KeyError):
        plan_rows(store, cfg, "synthesis", 4, variants=["gen_irac_analysis_v9"])
    assert store.task_counts() == {}


def test_the_wave_event_records_which_templates_were_allowed(store, cfg):
    """A wave planned on a subset is not reproducible from the target alone;
    the allowlist has to survive in the ledger next to it."""
    plan_wave(store, cfg, "synthesis", 6, variants=["gen_irac_analysis_v1"])
    detail = json.loads(store.events("wave_planned")[0]["detail_json"])
    assert detail["variants"] == ["gen_irac_analysis_v1"]


def test_an_unrestricted_wave_records_no_allowlist(store, cfg):
    plan_wave(store, cfg, "synthesis", 6)
    detail = json.loads(store.events("wave_planned")[0]["detail_json"])
    assert detail["variants"] is None


def test_cli_plans_a_wave_on_the_named_variants(tmp_path, cfg, capsys):
    config_path = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    with open_store(tmp_path, n_seeds=12, db_path=paths.state_db):
        pass
    assert tasks_main([
        "--config", config_path, "--stream", "synthesis", "--n", "8",
        "--variant", "gen_irac_analysis_v1", "--variant", "gen_irac_analysis_v2",
    ]) == 0
    with open_store(tmp_path, n_seeds=0, db_path=paths.state_db) as store:
        drawn = {
            r[0] for r in store.conn.execute(
                "SELECT prompt_id FROM task WHERE task_type = 'irac_analysis'"
            ).fetchall()
        }
    assert drawn and drawn <= {"gen_irac_analysis_v1", "gen_irac_analysis_v2"}


def test_cli_rejects_an_unknown_variant_by_name(tmp_path, cfg, capsys):
    config_path = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    with open_store(tmp_path, n_seeds=12, db_path=paths.state_db):
        pass
    with pytest.raises(SystemExit):
        tasks_main([
            "--config", config_path, "--stream", "synthesis", "--n", "8",
            "--variant", "gen_irac_analysis_v9",
        ])
    err = capsys.readouterr().err
    # Not merely "argparse refused something": the refusal has to name the id
    # and say what it should have been, or this test would pass just as well
    # against a build that has no --variant flag at all.
    assert "gen_irac_analysis_v9" in err and "not a generator template" in err


def test_cli_echoes_the_allowlist_it_planned_on(tmp_path, cfg, capsys):
    """The wave summary is what a CI log preserves; a run planned on two of
    four personas must not read identically to one planned on all four."""
    config_path = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    with open_store(tmp_path, n_seeds=12, db_path=paths.state_db):
        pass
    assert tasks_main([
        "--config", config_path, "--stream", "synthesis", "--n", "8",
        "--variant", "gen_irac_analysis_v1",
    ]) == 0
    assert "variants=gen_irac_analysis_v1" in capsys.readouterr().out


def test_cli_says_nothing_about_variants_when_it_used_them_all(tmp_path, cfg, capsys):
    config_path = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    with open_store(tmp_path, n_seeds=12, db_path=paths.state_db):
        pass
    assert tasks_main([
        "--config", config_path, "--stream", "synthesis", "--n", "8",
    ]) == 0
    assert "variants=" not in capsys.readouterr().out


def test_cli_adds_to_the_queue_it_finds_rather_than_to_the_one_it_was_told_about(
    tmp_path, cfg, capsys
):
    """`--n` is a TARGET, and the queue it is compared against keeps draining.

    Every wave so far has been sized by hand: read the live count out of a
    finished run's log, add the increment, pass the sum as `--n`. The reading
    and the dispatch are hours apart - the planner runs at a run boundary, and
    the boundary before it is where the count came from - so the queue has
    drained by hundreds of rows in between and the wave lands that much bigger
    than intended. Over-planning is not symmetric with under-planning here:
    generated rows cannot be dropped from the corpus afterwards.

    `--add` moves the arithmetic inside the job, where it is done against the
    queue that is actually there.
    """
    config_path = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    with open_store(tmp_path, n_seeds=12, db_path=paths.state_db):
        pass

    assert tasks_main(["--config", config_path, "--stream", "synthesis", "--n", "5"]) == 0
    capsys.readouterr()

    assert tasks_main(["--config", config_path, "--stream", "synthesis", "--add", "3"]) == 0
    out = capsys.readouterr().out
    assert "target=8" in out, "--add 3 onto a queue of 5 is a target of 8"
    assert "planned 3  collided 0" in out


def test_cli_refuses_add_and_n_together_because_they_are_two_answers(
    tmp_path, cfg, capsys
):
    """One says "make it this big", the other says "make it this much bigger".
    Silently letting one win is how a wave lands at the wrong size.

    The message is asserted, not just the exit: argparse already exits on an
    unknown flag, so a bare `raises(SystemExit)` here would pass before the
    flag exists and prove nothing.
    """
    config_path = temp_config(tmp_path)
    paths_for(tmp_path)
    with pytest.raises(SystemExit):
        tasks_main(["--config", config_path, "--stream", "synthesis", "--n", "5", "--add", "3"])
    err = capsys.readouterr().err
    assert "--add" in err and "--n" in err
    assert "unrecognized" not in err


def test_cli_counts_add_as_something_to_do(tmp_path, cfg, capsys):
    """The usage guard refuses a command that would plan nothing and re-open
    nothing. `--add` plans, so it has to satisfy that guard - otherwise the
    only way to use the new flag is to pass the old one alongside it."""
    config_path = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    with open_store(tmp_path, n_seeds=12, db_path=paths.state_db):
        pass
    assert tasks_main(["--config", config_path, "--stream", "synthesis", "--add", "2"]) == 0
    assert "planned 2" in capsys.readouterr().out
