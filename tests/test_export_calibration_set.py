"""The calibration anchor set is the pipeline's first human-checkable output.

Every accept rate this project has reported until now is model-agreeing-with-
model. This exporter builds the labelling file for an external anchor, and
the one rule that keeps that anchor honest is that the labeller must never
see what the judge said - if they did, they would just agree with it, and
the anchor collapses back into another agreement measurement.
"""
import json
import sqlite3

from scripts.export_calibration_set import build_rows


def _make_fake_store():
    """An in-memory store shaped like the real one, with a judge rationale on
    every row - the assertions below prove build_rows never carries it out."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        create table seed (seed_id text primary key, text text);
        create table task (task_id text primary key, seed_id text, task_type text, state text);
        create table generation (
            gen_id integer primary key, task_id text, think text, answer text
        );
        create table judgement (
            gen_id integer, judge_slot text, grounding integer, rationale text
        );
        """
    )
    conn.executemany(
        "insert into seed (seed_id, text) values (?, ?)",
        [
            ("seed-1", "Facts of the first matter."),
            ("seed-2", "Facts of the second matter."),
        ],
    )
    conn.executemany(
        "insert into task (task_id, seed_id, task_type, state) values (?, ?, ?, ?)",
        [
            ("task-1", "seed-1", "issue_spotting", "done"),
            ("task-2", "seed-2", "statute_lookup", "done"),
            ("task-3", "seed-1", "issue_spotting", "done"),
        ],
    )
    conn.executemany(
        "insert into generation (gen_id, task_id, think, answer) values (?, ?, ?, ?)",
        [
            (1, "task-1", "reasoning for the ungrounded answer", "the ungrounded answer"),
            (2, "task-2", "reasoning for the tiebreak answer", "the tiebreak answer"),
            (3, "task-3", "reasoning for the clean answer", "the clean answer"),
        ],
    )
    conn.executemany(
        "insert into judgement (gen_id, judge_slot, grounding, rationale) values (?, ?, ?, ?)",
        [
            (1, "a", 2, "judge said this cites a repealed section"),
            (2, "tiebreak", 4, "arbiter waved this through on the tie"),
            (3, "a", 5, "judge said this is a clean accept"),
        ],
    )
    return conn


FAKE_STORE = _make_fake_store()


def test_export_withholds_judge_rationales_and_leaves_labels_empty():
    """The labeller must not see what the judge said, or they will agree with it.

    Anchoring on two rubric-independent bits keeps the labels valid across
    rubric rewrites - the 1-5 bands have already been rewritten once.
    """
    rows = build_rows(FAKE_STORE, limit=40)
    assert len(rows) <= 40
    for row in rows:
        assert set(row) >= {
            "gen_id", "task_type", "task_instruction", "seed_text",
            "answer", "asserts_false", "asserts_unsupported",
        }
        assert row["asserts_false"] is None
        assert row["asserts_unsupported"] is None
        blob = json.dumps(row)
        assert "rationale" not in blob
        assert "grounding" not in blob
