import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pipeline_fakes import (
    CLEAN_ANSWER,
    CLEAN_THINK,
    NOVEL_ANSWER,
    NOVEL_WITH_INDEX,
    SEED_TEXT,
    FakeRouter,
    build_cfg,
    chat_response,
    open_store,
    paths_for,
    temp_config,
)

from tuned.data.citations import CitationIndex
from tuned.data.generate import run_workers
from tuned.data.store import Store
from tuned.data.tasks import plan_wave
from tuned.data.verify import content_for, latest_generations, live_leases
from tuned.data.verify import main as verify_main
from tuned.data.verify import rerun_gates

# The grounding text cites this one, so it is never "novel"; the index holds
# it as well, which is the realistic shape.
GROUNDED_CITATION = "(2008) 1 SCC 1"


@pytest.fixture
def cfg():
    return build_cfg()


@pytest.fixture
def paths(tmp_path):
    return paths_for(tmp_path)


@pytest.fixture
def index_path(tmp_path):
    CitationIndex.build([GROUNDED_CITATION, "2023 INSC 45"], tmp_path / "citations.txt")
    return str(tmp_path / "citations.txt")


def generated_store(
    tmp_path, paths, cfg, *, answer=CLEAN_ANSWER, reasoning=CLEAN_THINK, state="accepted",
    db_path=None,
):
    store = open_store(tmp_path, n_seeds=1, db_path=db_path)
    plan_wave(store, cfg, "synthesis", 1, task_type_mix={"irac_analysis": 1.0})
    router = FakeRouter(cfg, {"generator": [chat_response(answer, reasoning)]})
    asyncio.run(
        run_workers(
            store, cfg, router, paths=paths, streams=["synthesis"], n_workers=1, max_batches=1
        )
    )
    task_id = store.conn.execute("SELECT task_id FROM task").fetchone()[0]
    if state is not None:
        store.set_task_state(task_id, state)
    return store, task_id


def gate_detail(store, gen_id, gate):
    row = store.conn.execute(
        "SELECT detail_json FROM gate_result WHERE gen_id = ? AND gate = ?", (gen_id, gate)
    ).fetchone()
    return json.loads(row[0])


# --------------------------------------------------------------------------
# The demotion that the whole module exists for.
# --------------------------------------------------------------------------

def test_the_pilot_pass_cannot_see_a_novel_citation(tmp_path, paths, cfg):
    """Precondition for everything below: with citation_index=None the
    existence half is skipped, so this row is ACCEPTED carrying an invented
    authority. That is why verify.py is mandatory, not optional."""
    store, task_id = generated_store(tmp_path, paths, cfg, answer=NOVEL_ANSWER)
    with store:
        gen = store.latest_generation(task_id)
        assert store.gates_for(gen["gen_id"])["citations"] is True
        assert gate_detail(store, gen["gen_id"], "citations")["novel_skipped"] == "no-index"
        assert store.conn.execute("SELECT state FROM task").fetchone()[0] == "accepted"


def test_index_backed_rerun_demotes_a_novel_citation(tmp_path, paths, cfg, index_path):
    store, task_id = generated_store(tmp_path, paths, cfg, answer=NOVEL_ANSWER)
    with store:
        counts = rerun_gates(store, cfg, citation_index_path=index_path)
        assert counts["scanned"] == 1
        assert counts["regated"] == 1
        assert counts["demoted"] == 1
        assert counts["unverified"] == 0
        task = dict(store.conn.execute("SELECT * FROM task").fetchone())
        assert task["state"] == "rejected"
        assert task["disposition"] == "verify:citations"
        gen = store.latest_generation(task_id)
        detail = gate_detail(store, gen["gen_id"], "citations")
        assert detail["novel"] == [NOVEL_WITH_INDEX]
        assert "novel_skipped" not in detail
        event = json.loads(store.events("verify_demotion")[0]["detail_json"])
        assert event["gates"] == ["citations"]
        assert event["from_state"] == "accepted"
        assert event["content_source"] == "raw"


def test_a_clean_row_survives_the_rerun(tmp_path, paths, cfg, index_path):
    store, task_id = generated_store(tmp_path, paths, cfg)
    with store:
        counts = rerun_gates(store, cfg, citation_index_path=index_path)
        census = counts.pop("provenance")
        assert counts == {
            "scanned": 1, "regated": 1, "clean": 1, "demoted": 0, "soft_fail": 0,
            "diagnostic": 0, "missing_seed": 0, "slot_error": 0,
            "input_ineligible": 0, "held_by_worker": 0, "rebuilt_content": 0,
            "unverified": 0, "off_teacher": 0, "stale_prompt_sha": 0,
        }
        # The census is filled on EVERY run, armed or not - the whole point of
        # it is that a pass which demotes nothing still says who wrote the rows.
        assert list(census.values()) == [1]
        (state, teacher, _prompt_id, sha_state), = census
        assert (state, teacher, sha_state) == (
            "accepted", "bai/deepseek-v4-flash", "current",
        )
        assert store.conn.execute("SELECT state FROM task").fetchone()[0] == "accepted"
        gen = store.latest_generation(task_id)
        assert gate_detail(store, gen["gen_id"], "citations")["novel"] == []
        assert store.events("verify_demotion") == []


def test_a_grounded_citation_is_never_novel(tmp_path, paths, cfg):
    """The citation the teacher was HANDED must survive an index that does
    not contain it - grounding text is an allow-list in its own right."""
    empty_index = tmp_path / "empty.txt"
    CitationIndex.build([], empty_index)
    answer = CLEAN_ANSWER.replace(
        "is weighed with care.", f"is weighed with care, per {GROUNDED_CITATION}."
    )
    store, task_id = generated_store(tmp_path, paths, cfg, answer=answer)
    with store:
        counts = rerun_gates(store, cfg, citation_index_path=str(empty_index))
        assert counts["demoted"] == 0
        assert gate_detail(store, store.latest_generation(task_id)["gen_id"], "citations")[
            "novel"
        ] == []


# --------------------------------------------------------------------------
# What it re-scores, and what it refuses to touch.
# --------------------------------------------------------------------------

def test_the_original_bytes_are_read_back_from_the_raw_log(tmp_path, paths, cfg):
    store, task_id = generated_store(tmp_path, paths, cfg)
    with store:
        gen = store.latest_generation(task_id)
        content, how = content_for(cfg, gen)
        assert how == "raw"
        assert content.startswith(cfg.think_open)
        assert CLEAN_ANSWER in content


def test_content_falls_back_to_the_columns_when_the_raw_line_is_gone(tmp_path, paths, cfg):
    store, task_id = generated_store(tmp_path, paths, cfg)
    with store:
        gen = store.latest_generation(task_id)
        Path(gen["raw_path"]).unlink()
        content, how = content_for(cfg, gen)
        assert how == "rebuilt"
        assert gen["think"] in content and gen["answer"] in content
        counts = rerun_gates(store, cfg)
        assert counts["rebuilt_content"] == 1


def test_soft_failures_are_logged_but_never_demoted(tmp_path, paths, cfg, index_path):
    # A trace with no self-verification cue fails the diagnostic gate only.
    # Task 2 made that miss observable, not a hard/soft disposition. Verify
    # must still record and report it, and must not demote or churn the row.
    store, task_id = generated_store(
        tmp_path, paths, cfg, reasoning=CLEAN_THINK.replace("Let me check", "I note"),
        state="accepted",
    )
    with store:
        gen = store.latest_generation(task_id)
        assert store.gates_for(gen["gen_id"])["self_verification"] is False
        counts = rerun_gates(store, cfg, citation_index_path=index_path)
        assert counts["regated"] == 1
        assert counts["diagnostic"] == 1
        assert counts["soft_fail"] == 0
        assert counts["clean"] == 0
        assert counts["demoted"] == 0
        assert store.conn.execute("SELECT state FROM task").fetchone()[0] == "accepted"
        assert store.gates_for(gen["gen_id"])["self_verification"] is False
        event = json.loads(store.events("verify_diagnostic")[0]["detail_json"])
        assert event["gates"] == ["self_verification"]
        assert store.events("verify_soft_fail") == []
        assert store.events("verify_demotion") == []


def test_a_rejected_row_is_regated_but_not_re_demoted(tmp_path, paths, cfg, index_path):
    store, task_id = generated_store(
        tmp_path, paths, cfg, answer=NOVEL_ANSWER, state="rejected"
    )
    with store:
        counts = rerun_gates(store, cfg, citation_index_path=index_path)
        assert counts["regated"] == 1
        assert counts["demoted"] == 0
        assert store.events("verify_demotion") == []


def test_state_filter_limits_the_sweep(tmp_path, paths, cfg, index_path):
    store, task_id = generated_store(
        tmp_path, paths, cfg, answer=NOVEL_ANSWER, state="pending"
    )
    with store:
        assert rerun_gates(
            store, cfg, where_state="accepted", citation_index_path=index_path
        )["scanned"] == 0
        assert rerun_gates(
            store, cfg, where_state="pending", citation_index_path=index_path
        )["scanned"] == 1


def test_only_the_latest_attempt_is_scored(tmp_path, paths, cfg):
    store, task_id = generated_store(tmp_path, paths, cfg, answer=NOVEL_ANSWER, state="pending")
    with store:
        router = FakeRouter(cfg, {"generator": [chat_response(CLEAN_ANSWER, CLEAN_THINK)]})
        asyncio.run(
            run_workers(
                store, cfg, router, paths=paths, streams=["synthesis"], n_workers=1, max_batches=1
            )
        )
        assert store.conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0] == 2
        rows = latest_generations(store)
        assert len(rows) == 1
        assert rows[0]["attempt"] == 2
        assert rows[0]["stream"] == "synthesis"
        assert rows[0]["task_state"] == "judging"


def test_a_run_without_an_index_is_marked_unverified(tmp_path, paths, cfg):
    store, task_id = generated_store(tmp_path, paths, cfg)
    with store:
        counts = rerun_gates(store, cfg)
        assert counts["unverified"] == 1
        assert store.events("verify_no_index")
        assert gate_detail(store, store.latest_generation(task_id)["gen_id"], "citations")[
            "novel_skipped"
        ] == "no-index"


def test_live_leases_counts_only_held_tasks(tmp_path, paths, cfg):
    store, task_id = generated_store(tmp_path, paths, cfg, state="pending")
    with store:
        assert live_leases(store) == 0
        store.claim_tasks("gen-1", 1)
        assert live_leases(store) == 1
        # An expired lease is not a live one.
        stale = (datetime.now(UTC) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        store.conn.execute("UPDATE task SET claimed_at = ?", (stale,))
        assert live_leases(store) == 0


def test_cli_refuses_to_demote_under_a_live_fleet(tmp_path, paths, cfg, index_path, capsys):
    """rerun_gates writes task states without holding a lease of its own, so
    running it against live workers can overwrite a decision in flight."""
    config_path = temp_config(tmp_path)
    store, task_id = generated_store(
        tmp_path, paths, cfg, answer=NOVEL_ANSWER, state="judging", db_path=paths.state_db
    )
    with store:
        # A judge worker holding the row: a demotable state AND a live lease,
        # which is the collision this refusal exists for.
        store.claim_tasks("judge-1", 1, state_from="judging", state_to="judging_active")
    # The CLI opens its own handle on the same workdir.
    assert verify_main(["--config", config_path, "--index", index_path]) == 2
    out = capsys.readouterr().out
    assert "REFUSING" in out and "--force" in out
    with Store.open(paths.state_db) as reopened:
        assert reopened.conn.execute("SELECT state FROM task").fetchone()[0] == "judging_active"

    # --force means "overwrite the live holder if you must", so it overrides
    # the per-row re-check too - otherwise the flag would run the whole sweep
    # and quietly demote nothing.
    assert verify_main(["--config", config_path, "--index", index_path, "--force"]) == 0
    # `"demoted" in out` matched the printed LABEL and passed at zero
    # demotions, which is how the round-2 review found this test vacuous. The
    # state below is the load-bearing assertion; the count is asserted as a
    # NUMBER so the label can never stand in for it again.
    out = capsys.readouterr().out
    demoted = next(ln for ln in out.splitlines() if ln.startswith("demoted"))
    assert demoted.split()[1] == "1"
    with Store.open(paths.state_db) as reopened:
        assert reopened.conn.execute("SELECT state FROM task").fetchone()[0] == "rejected"


def test_a_missing_seed_is_counted_not_raised(tmp_path, paths, cfg, index_path):
    store, task_id = generated_store(tmp_path, paths, cfg)
    with store:
        store.conn.execute("PRAGMA foreign_keys=OFF")
        store.conn.execute("DELETE FROM seed")
        store.conn.execute("PRAGMA foreign_keys=ON")
        counts = rerun_gates(store, cfg, citation_index_path=index_path)
        assert counts["scanned"] == 1
        assert counts["missing_seed"] == 1
        assert counts["slot_error"] == 0
        assert counts["regated"] == 0


def test_an_unrenderable_row_is_counted_apart_from_a_missing_seed(tmp_path, paths, cfg):
    """A transition row generated before the dates became mandatory can no
    longer be re-gated (build_slots raises), and calling that "missing_seed"
    hid a class of rows that will never be verified from the one number an
    operator reads to decide the sweep was complete."""
    store, task_id = generated_store(tmp_path, paths, cfg)
    with store:
        store.conn.execute("UPDATE task SET stream = 'transition', task_type = 'transition'")
        counts = rerun_gates(store, cfg)
        assert counts["scanned"] == 1
        assert counts["slot_error"] == 1
        assert counts["missing_seed"] == 0
        assert counts["regated"] == 0
        event = json.loads(store.events("verify_skipped")[0]["detail_json"])
        assert event["reason"].startswith("SlotError")
        assert counts.get("input_ineligible", 0) == 0


def _legacy_ineligible_statute_qa(store, task_id, *, section_text=None):
    meta = {} if section_text is None else {"section_text": section_text}
    store.conn.execute(
        "UPDATE task SET task_type = 'statute_qa' WHERE task_id = ?",
        (task_id,),
    )
    store.conn.execute("UPDATE seed SET meta_json = ?", (json.dumps(meta),))
    store.conn.commit()


def test_legacy_ineligible_statute_qa_is_reported_not_regated(
    tmp_path, paths, cfg, index_path
):
    store, task_id = generated_store(tmp_path, paths, cfg)
    with store:
        gen_before = store.latest_generation(task_id)
        citations_before = gate_detail(store, gen_before["gen_id"], "citations")
        _legacy_ineligible_statute_qa(store, task_id)
        counts = rerun_gates(store, cfg, citation_index_path=index_path)
        assert counts["scanned"] == 1
        assert counts["input_ineligible"] == 1
        assert counts["slot_error"] == 0
        assert counts["regated"] == 0
        assert counts["demoted"] == 0
        assert counts["clean"] == 0
        task = dict(store.conn.execute("SELECT * FROM task").fetchone())
        assert task["state"] == "input_ineligible"
        assert task["disposition"] == "input-ineligible:section_text"
        event = json.loads(store.events("verify_input_ineligible")[0]["detail_json"])
        assert event["task_id"] == task_id
        assert event["from_state"] == "accepted"
        assert store.events("verify_demotion") == []
        citations_after = gate_detail(store, gen_before["gen_id"], "citations")
        assert citations_after == citations_before


def test_legacy_source_equal_statute_qa_is_not_treated_as_a_provision(
    tmp_path, paths, cfg, index_path
):
    store, task_id = generated_store(tmp_path, paths, cfg)
    with store:
        _legacy_ineligible_statute_qa(store, task_id, section_text=SEED_TEXT)
        counts = rerun_gates(store, cfg, citation_index_path=index_path)
        assert counts["input_ineligible"] == 1
        assert counts["regated"] == 0
        assert store.conn.execute("SELECT state FROM task").fetchone()[0] == (
            "input_ineligible"
        )


def test_legacy_ineligible_statute_qa_does_not_promote_a_rejected_row(
    tmp_path, paths, cfg, index_path
):
    store, task_id = generated_store(tmp_path, paths, cfg, state="rejected")
    with store:
        store.set_task_state(task_id, "rejected", "reject:citations")
        _legacy_ineligible_statute_qa(store, task_id)
        counts = rerun_gates(store, cfg, citation_index_path=index_path)
        assert counts["input_ineligible"] == 1
        assert counts["regated"] == 0
        assert counts["demoted"] == 0
        task = dict(store.conn.execute("SELECT * FROM task").fetchone())
        assert task["state"] == "rejected"
        assert task["disposition"] == "reject:citations"


def test_a_demotion_re_checks_the_lease_it_is_about_to_overwrite(tmp_path, paths, cfg, index_path):
    """This pass holds no lease of its own, and a sweep long enough to matter
    is long enough for a worker to claim a row after the opening check. The
    demotion is skipped and counted rather than racing the live holder."""
    store, task_id = generated_store(tmp_path, paths, cfg, answer=NOVEL_ANSWER, state="judging")
    with store:
        store.claim_tasks(
            "judge-1", 1, state_from="judging", state_to="judging_active"
        )
        counts = rerun_gates(store, cfg, citation_index_path=index_path)
        assert counts["regated"] == 1
        assert counts["demoted"] == 0
        assert counts["held_by_worker"] == 1
        assert store.conn.execute("SELECT state FROM task").fetchone()[0] == "judging_active"
        assert store.events("verify_demotion_deferred")
        # Once the lease expires the demotion goes through.
        stale = (datetime.now(UTC) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        store.conn.execute("UPDATE task SET claimed_at = ?", (stale,))
        assert rerun_gates(store, cfg, citation_index_path=index_path)["demoted"] == 1
        assert store.conn.execute("SELECT state FROM task").fetchone()[0] == "rejected"


def test_verify_shares_the_stores_lease_constants(tmp_path):
    """A second copy of the lease window is a fence that silently disagrees
    with the fencing."""
    from tuned.data import store as store_module
    from tuned.data import verify as verify_module

    assert verify_module.DEFAULT_LEASE_S is store_module.DEFAULT_LEASE_S


# --------------------------------------------------------------------------
# The cut: one teacher, at the prompt templates on disk.
#
# Verified in the live store on 2026-08-30: all 17 accepted rows and 73 of 80
# judging rows were written by RETIRED providers at retired prompt shas, while
# only 27 of 1,423 generations came from the configured generator. Nothing
# downstream can see any of that - decontaminate.generated_rows selects on
# `state = 'accepted'` and no other column - so a cut assembled then would
# have blended two teachers under one dataset card without a word.
# --------------------------------------------------------------------------

RETIRED_TEACHER = ("cerebras", "gpt-oss-120b")
RETIRED_SHA = "97185cd2068e"


def retire_teacher(store, provider=RETIRED_TEACHER[0], model=RETIRED_TEACHER[1]):
    store.conn.execute("UPDATE generation SET provider = ?, model = ?", (provider, model))
    store.conn.commit()


def retire_prompt_sha(store, sha=RETIRED_SHA):
    store.conn.execute("UPDATE task SET prompt_sha = ?", (sha,))
    store.conn.commit()


def census_of(counts):
    return {(teacher, sha): n for (_state, teacher, _pid, sha), n in counts["provenance"].items()}


def test_the_census_names_a_retired_teacher_without_being_asked(tmp_path, paths, cfg, index_path):
    """(a) of P1.1: the readout is unconditional.

    An operator running an ordinary verify pass must be able to answer "who
    wrote these rows" from its output alone - that question has no other
    reader in the whole chain.
    """
    store, _ = generated_store(tmp_path, paths, cfg)
    with store:
        retire_teacher(store)
        retire_prompt_sha(store)
        counts = rerun_gates(store, cfg, citation_index_path=index_path)
        assert census_of(counts) == {("cerebras/gpt-oss-120b", "stale"): 1}
        # ...and nothing was demoted for it. The census reports; the filter
        # rules, and it was not armed here.
        assert counts["off_teacher"] == 0 and counts["stale_prompt_sha"] == 0
        assert counts["regated"] == 1
        assert store.conn.execute("SELECT state FROM task").fetchone()[0] == "accepted"


def test_an_off_teacher_row_is_parked_not_rejected(tmp_path, paths, cfg, index_path):
    """Demote-only. The row is wrong about NOTHING - the seed, the law and the
    plan are all still good - so `rejected` would close a row that a change of
    ruling, or a regeneration, makes shippable again."""
    store, task_id = generated_store(tmp_path, paths, cfg)
    with store:
        retire_teacher(store)
        counts = rerun_gates(
            store, cfg, citation_index_path=index_path,
            generators=("bai/deepseek-v4-flash",),
        )
        assert counts["off_teacher"] == 1
        assert counts["demoted"] == 0
        state, disposition = store.conn.execute(
            "SELECT state, disposition FROM task"
        ).fetchone()
        assert (state, disposition) == ("off_teacher", "verify:off-teacher")
        event = json.loads(store.events("verify_off_cut")[-1]["detail_json"])
        assert event["reason"] == "off_teacher"
        assert event["teacher"] == "cerebras/gpt-oss-120b"


def test_an_off_cut_row_is_not_re_gated(tmp_path, paths, cfg, index_path):
    """Why the filter runs BEFORE the gates rather than after.

    A stale-sha row re-gated against today's template is scored against a
    question it was never asked; the gates read clean and the row stays
    accepted forever, which is how the live store came to hold 17 of them.
    """
    store, _ = generated_store(tmp_path, paths, cfg)
    with store:
        retire_prompt_sha(store)
        counts = rerun_gates(
            store, cfg, citation_index_path=index_path, require_current_prompt=True
        )
        assert counts["scanned"] == 1
        assert counts["regated"] == 0
        assert counts["clean"] == 0
        assert counts["stale_prompt_sha"] == 1
        state, disposition = store.conn.execute(
            "SELECT state, disposition FROM task"
        ).fetchone()
        # generate.py's own state, not a second name for one fact.
        assert (state, disposition) == ("stale_prompt", "verify:stale-prompt-sha")


def test_a_row_with_no_recorded_prompt_sha_is_not_current(tmp_path, paths, cfg, index_path):
    """The deliberate disagreement with generate.py's live guard.

    That guard reads `if planned_sha and planned_sha != live_sha` - a row it
    cannot prove stale gets its call. This pass is deciding whether an answer
    ALREADY BOUGHT belongs in a cut described as "one teacher at current
    prompts", and a row whose prompt generation was never recorded does not
    meet that description.
    """
    store, _ = generated_store(tmp_path, paths, cfg)
    with store:
        # task.prompt_sha is NOT NULL, so the reachable shape of "nothing was
        # recorded" is the empty string, not a null.
        store.conn.execute("UPDATE task SET prompt_sha = ''")
        store.conn.commit()
        counts = rerun_gates(
            store, cfg, citation_index_path=index_path, require_current_prompt=True
        )
        assert census_of(counts) == {("bai/deepseek-v4-flash", "unrecorded"): 1}
        assert counts["stale_prompt_sha"] == 1


def test_both_filters_are_off_unless_asked_for(tmp_path, paths, cfg, index_path):
    """An ad-hoc verify run stays a pure citation re-check. The cut is armed on
    the ship path (assemble_argvs) and nowhere else, so an operator re-gating
    between waves cannot demote the corpus by accident."""
    store, _ = generated_store(tmp_path, paths, cfg)
    with store:
        retire_teacher(store)
        retire_prompt_sha(store)
        counts = rerun_gates(store, cfg, citation_index_path=index_path)
        assert (counts["off_teacher"], counts["stale_prompt_sha"]) == (0, 0)
        assert store.conn.execute("SELECT state FROM task").fetchone()[0] == "accepted"


def test_the_cut_only_touches_states_it_may_demote(tmp_path, paths, cfg, index_path):
    """A rejected row is already out of the pool, and re-writing its state
    would overwrite the verdict that put it there."""
    store, _ = generated_store(tmp_path, paths, cfg, state="rejected")
    with store:
        retire_teacher(store)
        counts = rerun_gates(
            store, cfg, citation_index_path=index_path,
            generators=("bai/deepseek-v4-flash",),
        )
        assert counts["off_teacher"] == 0
        assert census_of(counts) == {("cerebras/gpt-oss-120b", "current"): 1}
        assert store.conn.execute("SELECT state FROM task").fetchone()[0] == "rejected"


def test_the_cut_defers_to_a_live_lease_like_every_other_demotion(
    tmp_path, paths, cfg, index_path
):
    """The third demotion path has to be behind the same fence as the other
    two - which is why there is one fence and not three copies of it."""
    store, _ = generated_store(tmp_path, paths, cfg, state="judging")
    with store:
        retire_teacher(store)
        store.claim_tasks("judge-1", 1, state_from="judging", state_to="judging_active")
        counts = rerun_gates(
            store, cfg, citation_index_path=index_path,
            generators=("bai/deepseek-v4-flash",),
        )
        assert counts["off_teacher"] == 0
        assert counts["held_by_worker"] == 1
        assert store.conn.execute("SELECT state FROM task").fetchone()[0] == "judging_active"


def test_an_off_teacher_row_comes_back_with_a_budget_to_regenerate_with(
    tmp_path, paths, cfg, index_path
):
    """The whole recovery path for decision 4, end to end.

    The ~90 rows the ruling removes must be REGENERATED, and their task_ids
    are hashes of (seed, task_type, prompt_id, sample_ix) - so a re-plan is
    INSERT OR IGNORE'd straight back into the existing row and cannot produce
    a substitute. Re-opening the row itself is the only route, and a row
    handed back at MAX_ATTEMPTS is one the generator will never claim.
    """
    from tuned.data.tasks import reopen_tasks

    store, task_id = generated_store(tmp_path, paths, cfg)
    with store:
        retire_teacher(store)
        store.conn.execute("UPDATE task SET attempts = 3")
        store.conn.commit()
        rerun_gates(
            store, cfg, citation_index_path=index_path,
            generators=("bai/deepseek-v4-flash",),
        )
        skipped: dict = {}
        assert reopen_tasks(store, ["off_teacher"], skipped=skipped) == {"off_teacher": 1}
        assert not skipped  # not "out of budget" - the park was a ruling, not a failure
        state, attempts = store.conn.execute(
            "SELECT state, attempts FROM task"
        ).fetchone()
        assert (state, attempts) == ("pending", 0)


def test_the_cli_pool_defaults_to_the_configured_generator(
    tmp_path, paths, cfg, index_path, capsys
):
    """--require-generator with no --generator means "whoever routing.generator
    says the teacher is today", never a second list that can drift from it."""
    config_path = temp_config(tmp_path)
    store, _ = generated_store(tmp_path, paths, cfg, db_path=paths.state_db)
    with store:
        retire_teacher(store)

    assert verify_main(
        ["--config", config_path, "--index", index_path, "--require-generator"]
    ) == 0
    out = capsys.readouterr().out
    assert f"cut: generator in {', '.join(cfg.routing.generator)}" in out
    assert "provenance (rows by state" in out and "cerebras/gpt-oss-120b" in out
    with Store.open(paths.state_db) as reopened:
        assert reopened.conn.execute("SELECT state FROM task").fetchone()[0] == "off_teacher"


def test_the_cli_pool_can_be_overridden_without_editing_the_config(
    tmp_path, paths, cfg, index_path
):
    """--generator implies the filter, so naming a pool cannot silently do
    nothing - and it is what re-admits a retired teacher for a one-off
    measurement without touching the shipped routing."""
    config_path = temp_config(tmp_path)
    store, _ = generated_store(tmp_path, paths, cfg, db_path=paths.state_db)
    with store:
        retire_teacher(store)

    assert verify_main(
        ["--config", config_path, "--index", index_path,
         "--generator", "cerebras/gpt-oss-120b"]
    ) == 0
    with Store.open(paths.state_db) as reopened:
        assert reopened.conn.execute("SELECT state FROM task").fetchone()[0] == "accepted"


def test_a_retired_prompt_id_reads_as_stale_rather_than_raising():
    """The bytes the teacher saw are gone whether the template was edited or
    deleted, and a census that raises on a retired id reports nothing at all."""
    from tuned.data.verify import prompt_sha_state

    assert prompt_sha_state({"prompt_id": "gen_no_such_template", "prompt_sha": "abc"}) == "stale"
    assert prompt_sha_state({"prompt_id": "gen_no_such_template"}) == "unrecorded"
