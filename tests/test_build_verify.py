import asyncio
import json
from pathlib import Path

import pytest
from pipeline_fakes import (
    CLEAN_ANSWER,
    CLEAN_THINK,
    NOVEL_ANSWER,
    NOVEL_WITH_INDEX,
    FakeRouter,
    build_cfg,
    chat_response,
    open_store,
    paths_for,
)

from tuned.data.citations import CitationIndex
from tuned.data.generate import run_workers
from tuned.data.tasks import plan_wave
from tuned.data.verify import content_for, latest_generations, rerun_gates

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
    tmp_path, paths, cfg, *, answer=CLEAN_ANSWER, reasoning=CLEAN_THINK, state="accepted"
):
    store = open_store(tmp_path, n_seeds=1)
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
        assert counts == {
            "scanned": 1, "regated": 1, "clean": 1, "demoted": 0, "soft_fail": 0,
            "missing_seed": 0, "rebuilt_content": 0, "unverified": 0,
        }
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
    # A trace with no self-verification cue fails a SOFT (regenerate) gate.
    # It is force-accepted here to stand for a row an earlier, looser pass
    # let through - exactly the case where verify.py must not churn.
    store, task_id = generated_store(
        tmp_path, paths, cfg, reasoning=CLEAN_THINK.replace("Let me check", "I note"),
        state="accepted",
    )
    with store:
        gen = store.latest_generation(task_id)
        assert store.gates_for(gen["gen_id"])["self_verification"] is False
        counts = rerun_gates(store, cfg, citation_index_path=index_path)
        assert counts["soft_fail"] == 1
        assert counts["demoted"] == 0
        assert store.conn.execute("SELECT state FROM task").fetchone()[0] == "accepted"
        assert store.events("verify_soft_fail")


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


def test_a_missing_seed_is_counted_not_raised(tmp_path, paths, cfg, index_path):
    store, task_id = generated_store(tmp_path, paths, cfg)
    with store:
        store.conn.execute("PRAGMA foreign_keys=OFF")
        store.conn.execute("DELETE FROM seed")
        store.conn.execute("PRAGMA foreign_keys=ON")
        counts = rerun_gates(store, cfg, citation_index_path=index_path)
        assert counts["scanned"] == 1
        assert counts["missing_seed"] == 1
        assert counts["regated"] == 0
