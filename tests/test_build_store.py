import json
import re
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from tuned.data.jsonl import append_ndjson, read_at
from tuned.data.store import _PRAGMAS, Store, utcnow

# The store's timestamp contract: fixed width, so lexicographic comparison is
# chronological comparison. Lease expiry is a string "<" inside SQL and breaks
# silently if this ever drifts (e.g. isoformat() dropping ".000000").
_TS_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")


def _ago(seconds: int) -> str:
    return (datetime.now(UTC) - timedelta(seconds=seconds)).strftime(_TS_FMT)


@pytest.fixture
def store(tmp_path):
    with Store.open(tmp_path / "state" / "law_v1.sqlite3") as s:
        yield s


def _seed_rows(n, source_id="ikanoon"):
    return [
        {
            "seed_id": f"sd{i}",
            "source_id": source_id,
            "text": f"judgment text {i}",
            "case_type": "bail",
            "code_era": "bns",
            "token_count": 100 + i,
        }
        for i in range(n)
    ]


def _task_rows(n, stream="analysis"):
    return [
        {
            "task_id": f"t{i}",
            "seed_id": f"sd{i}",
            "stream": stream,
            "task_type": "reason",
            "prompt_id": "p1",
            "prompt_sha": "deadbeef",
            "sample_ix": 0,
        }
        for i in range(n)
    ]


def _populate(store, n=4, stream="analysis"):
    store.upsert_source("ikanoon", "CC-BY-4.0", url="https://example.test")
    store.upsert_seeds(_seed_rows(n))
    store.create_tasks(_task_rows(n, stream=stream))
    return [f"t{i}" for i in range(n)]


def _gen_envelope(task_id, attempt=1, **extra):
    row = {
        "kind": "generation",
        "task_id": task_id,
        "attempt": attempt,
        "provider": "groq",
        "model": "qwen/qwen3.6-27b",
        "model_family": "qwen",
        "think": "reasoning...",
        "answer": "धारा 302 applies",
        "prompt_tokens": 100,
        "completion_tokens": 200,
        "finish_reason": "stop",
    }
    row.update(extra)
    return row


# --------------------------------------------------------------- 1. open/schema


def test_open_creates_parents_and_is_idempotent(tmp_path):
    db = tmp_path / "state" / "nested" / "law_v1.sqlite3"
    assert not db.parent.exists()
    s1 = Store.open(db)
    assert db.exists()
    s1.upsert_source("s1", "CC-BY-4.0")
    s1.close()

    s2 = Store.open(db)  # second open on an existing file must not raise
    s2.ensure_schema()  # and CREATE TABLE IF NOT EXISTS is re-runnable
    assert s2.conn.execute("SELECT COUNT(*) FROM source").fetchone()[0] == 1
    tables = {r[0] for r in s2.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {
        "source",
        "seed",
        "task",
        "generation",
        "gate_result",
        "judgement",
        "gold_label",
        "judge_threshold",
        "budget_ledger",
        "run_event",
    } <= tables
    s2.close()


def test_pragmas_are_active(store):
    assert store.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert store.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert store.conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL


def test_busy_timeout_is_armed_before_the_wal_switch():
    # Regression guard for an ordering bug found with 8 concurrent opens:
    # switching journal_mode takes a brief exclusive lock, so with the default
    # busy timeout of 0 a worker opening the DB at that instant dies with
    # "database is locked" instead of waiting. Threads are out of scope for
    # this suite, so pin the invariant that fixes it instead.
    assert _PRAGMAS.index("busy_timeout=5000") < _PRAGMAS.index("journal_mode=WAL")
    assert set(_PRAGMAS) == {
        "busy_timeout=5000",
        "journal_mode=WAL",
        "synchronous=NORMAL",
        "foreign_keys=ON",
    }


def test_context_manager_closes_connection(tmp_path):
    with Store.open(tmp_path / "s.sqlite3") as s:
        s.upsert_source("s1", "CC-BY-4.0")
    with pytest.raises(sqlite3.ProgrammingError):
        s.conn.execute("SELECT 1")


def test_utcnow_is_fixed_width_and_sorts_chronologically():
    now = utcnow()
    assert _TS_RE.match(now), now
    assert _ago(60) < now
    assert now < _ago(-60)


def test_foreign_keys_are_enforced(store):
    store.upsert_source("ikanoon", "CC-BY-4.0")
    with pytest.raises(sqlite3.IntegrityError):
        store.upsert_seeds([{"seed_id": "sd0", "source_id": "nope", "text": "x"}])


# ------------------------------------------------------------ 2. seeds & tasks


def test_seed_and_task_lifecycle(store):
    store.upsert_source("ikanoon", "CC-BY-4.0", url="https://example.test", version="2026-08")
    assert store.upsert_seeds(_seed_rows(4)) == 4
    assert store.seed_count() == 4
    assert store.seed_count("ikanoon") == 4
    assert store.seed_count("other") == 0
    assert store.get_seed("sd0")["text"] == "judgment text 0"
    assert store.get_seed("missing") is None

    src = store.conn.execute("SELECT * FROM source WHERE source_id='ikanoon'").fetchone()
    assert src["license"] == "CC-BY-4.0"
    assert _TS_RE.match(src["retrieved_at"])

    assert store.create_tasks(_task_rows(4)) == 4
    assert store.create_tasks(_task_rows(4)) == 0  # replanning the wave adds nothing
    assert store.task_counts() == {"pending": 4}

    claimed = store.claim_tasks("worker-A", 2)
    assert [r["task_id"] for r in claimed] == ["t0", "t1"]
    for row in claimed:
        assert row["state"] == "generating"
        assert row["attempts"] == 1
        assert row["claimed_by"] == "worker-A"
        assert _TS_RE.match(row["claimed_at"])
    assert store.task_counts() == {"generating": 2, "pending": 2}


def test_create_tasks_counts_only_new_rows(store):
    _populate(store, n=2)
    rows = _task_rows(4)  # t0/t1 exist, t2/t3 are new
    store.upsert_seeds(_seed_rows(4))
    assert store.create_tasks(rows) == 2
    assert store.task_counts() == {"pending": 4}


def test_upsert_seeds_replaces_without_destroying_dependent_tasks(store):
    _populate(store, n=2)
    store.claim_tasks("worker-A", 1)
    # INSERT OR REPLACE deletes+reinserts the parent row; with foreign_keys=ON
    # the child task rows must survive because the primary key is unchanged.
    assert store.upsert_seeds([{"seed_id": "sd0", "source_id": "ikanoon", "text": "revised"}]) == 1
    assert store.get_seed("sd0")["text"] == "revised"
    assert store.get_task("t0")["state"] == "generating"
    assert store.seed_count() == 2


def test_upsert_source_replaces_in_place(store):
    store.upsert_source("ikanoon", "CC-BY-4.0")
    store.upsert_source("ikanoon", "CC-BY-SA-4.0", url="https://new.test")
    rows = store.conn.execute("SELECT * FROM source").fetchall()
    assert len(rows) == 1
    assert rows[0]["license"] == "CC-BY-SA-4.0"
    assert rows[0]["url"] == "https://new.test"


def test_json_columns_accept_dicts_and_store_text(store):
    store.upsert_source("ikanoon", "CC-BY-4.0")
    store.upsert_seeds(
        [
            {
                "seed_id": "sd0",
                "source_id": "ikanoon",
                "text": "t",
                "meta_json": {"bench": "SC", "hi": "धारा"},
                "roles_json": ["accused", "state"],
                "answer_key_json": '{"already":"text"}',
            }
        ]
    )
    seed = store.get_seed("sd0")
    assert isinstance(seed["meta_json"], str)
    assert json.loads(seed["meta_json"]) == {"bench": "SC", "hi": "धारा"}
    assert json.loads(seed["roles_json"]) == ["accused", "state"]
    assert seed["answer_key_json"] == '{"already":"text"}'  # str passes through untouched


def test_a_failed_batch_leaves_no_partial_rows(store):
    # INSERT OR IGNORE does NOT swallow foreign-key violations, so one bad row
    # aborts the batch - and the transaction must take the good rows with it,
    # or a retried wave plan would be indexed twice.
    _populate(store, n=1)
    base = _task_rows(1)[0]
    with pytest.raises(sqlite3.IntegrityError):
        store.create_tasks(
            [
                dict(base, task_id="t-good", seed_id="sd0"),
                dict(base, task_id="t-bad", seed_id="no-such-seed"),
            ]
        )
    assert store.get_task("t-good") is None
    assert store.get_task("t-bad") is None
    assert store.task_counts() == {"pending": 1}  # only the original t0


def test_empty_batches_are_no_ops(store):
    assert store.upsert_seeds([]) == 0
    assert store.create_tasks([]) == 0
    assert store.claim_tasks("w", 0) == []
    assert store.claim_tasks("w", 5) == []


# ----------------------------------------------------------- 3. lease semantics


def test_stale_lease_is_recovered_but_a_live_lease_is_not_stolen(store):
    _populate(store, n=2)
    a_claim = store.claim_tasks("worker-A", 1)
    assert [r["task_id"] for r in a_claim] == ["t0"]

    # A fresh claim by B must take the OTHER task, never A's live lease.
    b_first = store.claim_tasks("worker-B", 2)
    assert [r["task_id"] for r in b_first] == ["t1"]
    assert store.get_task("t0")["claimed_by"] == "worker-A"

    # Backdate A's lease past its expiry: worker A is presumed dead.
    store.conn.execute(
        "UPDATE task SET claimed_at = ? WHERE task_id = 't0'", (_ago(3600),)
    )
    recovered = store.claim_tasks("worker-B", 2, lease_s=900)
    assert [r["task_id"] for r in recovered] == ["t0"]
    assert recovered[0]["claimed_by"] == "worker-B"
    assert recovered[0]["attempts"] == 2  # the dead attempt is still counted
    assert recovered[0]["state"] == "generating"


def test_generating_row_without_a_lease_stamp_is_reclaimable(store):
    _populate(store, n=1)
    store.claim_tasks("worker-A", 1)
    store.conn.execute("UPDATE task SET claimed_at = NULL WHERE task_id = 't0'")
    assert [r["task_id"] for r in store.claim_tasks("worker-B", 1)] == ["t0"]


def test_claim_filters_by_stream(store):
    store.upsert_source("ikanoon", "CC-BY-4.0")
    store.upsert_seeds(_seed_rows(2))
    store.create_tasks(
        [
            dict(_task_rows(1)[0], task_id="t-an", seed_id="sd0", stream="analysis"),
            dict(_task_rows(1)[0], task_id="t-dr", seed_id="sd1", stream="drafting"),
        ]
    )
    claimed = store.claim_tasks("worker-A", 5, stream="drafting")
    assert [r["task_id"] for r in claimed] == ["t-dr"]
    assert store.get_task("t-an")["state"] == "pending"


def test_two_handles_on_one_db_never_double_claim(tmp_path):
    db = tmp_path / "state" / "law_v1.sqlite3"
    a = Store.open(db)
    b = Store.open(db)
    try:
        _populate(a, n=4)
        # Both workers ask for 3 of the same 4 candidates; BEGIN IMMEDIATE
        # serialises the write transactions so the sets cannot overlap.
        got_a = {r["task_id"] for r in a.claim_tasks("worker-A", 3)}
        got_b = {r["task_id"] for r in b.claim_tasks("worker-B", 3)}
        assert len(got_a) == 3
        assert len(got_b) == 1  # only the leftover was still claimable
        assert got_a & got_b == set()
        assert a.task_counts() == {"generating": 4}
        for task_id in got_b:
            assert b.get_task(task_id)["claimed_by"] == "worker-B"
    finally:
        a.close()
        b.close()


def test_claim_runs_inside_an_immediate_transaction(store):
    # The double-claim test above cannot tell BEGIN IMMEDIATE from a deferred
    # BEGIN, because its two claims are sequential. Under genuine overlap the
    # deferred variant is not merely slower: measured with 6 threads, 5 of them
    # died with "database is locked" (SQLITE_BUSY_SNAPSHOT, which busy_timeout
    # cannot retry away) because they upgraded reader->writer mid-transaction.
    _populate(store, n=1)
    statements = []
    store.conn.set_trace_callback(statements.append)
    try:
        store.claim_tasks("worker-A", 1)
    finally:
        store.conn.set_trace_callback(None)
    assert statements[0].strip().upper().startswith("BEGIN IMMEDIATE")
    assert statements[-1].strip().upper().startswith("COMMIT")


def test_set_task_state_releases_the_lease(store):
    _populate(store, n=2)
    store.claim_tasks("worker-A", 1)
    before = store.get_task("t0")["updated_at"]

    store.set_task_state("t0", "accepted", disposition="gates_pass")
    row = store.get_task("t0")
    assert row["state"] == "accepted"
    assert row["claimed_by"] is None
    assert row["claimed_at"] is None
    assert row["disposition"] == "gates_pass"
    assert row["updated_at"] >= before

    # disposition=None keeps the diagnostic that caused the transition.
    store.set_task_state("t0", "rejected")
    assert store.get_task("t0")["disposition"] == "gates_pass"
    store.set_task_state("t0", "rejected", disposition="judge_fail")
    assert store.get_task("t0")["disposition"] == "judge_fail"


def test_set_task_state_to_generating_keeps_the_lease(store):
    _populate(store, n=1)
    store.claim_tasks("worker-A", 1)
    store.set_task_state("t0", "generating")
    assert store.get_task("t0")["claimed_by"] == "worker-A"


# ----------------------------------------- 5/8. generations, gates, judgements


def test_record_generation_rejects_duplicate_attempt(store):
    _populate(store, n=1)
    gen_id = store.record_generation(
        _gen_envelope("t0", 1, raw_path="raw/gen/a.ndjson", raw_offset=0)
    )
    assert gen_id > 0
    with pytest.raises(sqlite3.IntegrityError):
        store.record_generation(_gen_envelope("t0", 1, raw_path="raw/gen/a.ndjson", raw_offset=99))
    # a genuine retry (next attempt) is fine
    assert store.record_generation(
        _gen_envelope("t0", 2, raw_path="raw/gen/a.ndjson", raw_offset=99)
    ) != gen_id


def test_record_generation_requires_a_raw_pointer(store):
    _populate(store, n=1)
    with pytest.raises(sqlite3.IntegrityError):
        store.record_generation(_gen_envelope("t0", 1))  # no raw_path/raw_offset


def test_latest_generation_and_counts(store):
    _populate(store, n=3)
    store.record_generation(_gen_envelope("t0", 1, raw_path="a", raw_offset=0, answer="first"))
    store.record_generation(_gen_envelope("t0", 2, raw_path="a", raw_offset=10, answer="second"))
    store.record_generation(_gen_envelope("t1", 1, raw_path="a", raw_offset=20))
    latest = store.latest_generation("t0")
    assert latest["attempt"] == 2
    assert latest["answer"] == "second"
    assert store.latest_generation("t2") is None

    store.set_task_state("t0", "accepted")
    store.set_task_state("t1", "rejected", disposition="gate_fail")
    assert store.task_counts() == {"accepted": 1, "rejected": 1, "pending": 1}
    assert store.accepted_count() == 1
    assert store.accepted_count("analysis") == 1
    assert store.accepted_count("drafting") == 0


def test_gates_round_trip(store):
    _populate(store, n=1)
    gen_id = store.record_generation(_gen_envelope("t0", 1, raw_path="a", raw_offset=0))
    store.record_gates(
        gen_id, [("length", True, {"tokens": 900}), ("citation", False, {"missing": ["s302"]})]
    )
    assert store.gates_for(gen_id) == {"length": True, "citation": False}
    store.record_gates(gen_id, [("citation", True, None)])  # re-gating overwrites
    assert store.gates_for(gen_id) == {"length": True, "citation": True}
    detail = store.conn.execute(
        "SELECT detail_json FROM gate_result WHERE gen_id=? AND gate='length'", (gen_id,)
    ).fetchone()[0]
    assert json.loads(detail) == {"tokens": 900}
    store.record_gates(gen_id, [])  # no-op
    assert len(store.gates_for(gen_id)) == 2


def test_judgement_round_trip_and_replace(store):
    _populate(store, n=1)
    gen_id = store.record_generation(_gen_envelope("t0", 1, raw_path="a", raw_offset=0))
    store.record_judgement(
        gen_id, "j1", {"provider": "cerebras", "model": "m", "grounding": 4, "validity": 5}
    )
    store.record_judgement(gen_id, "j2", {"provider": "groq", "model": "m2", "grounding": 3})
    rows = store.judgements_for(gen_id)
    assert [r["judge_slot"] for r in rows] == ["j1", "j2"]
    assert rows[0]["grounding"] == 4
    store.record_judgement(gen_id, "j1", {"provider": "cerebras", "model": "m", "grounding": 2})
    assert store.judgements_for(gen_id)[0]["grounding"] == 2
    assert len(store.judgements_for(gen_id)) == 2


# ------------------------------------------------------------------- 6. budget


def test_usage_accumulates_and_is_zero_by_default(store):
    assert store.usage_today("groq", "m") == {
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "errors_429": 0,
    }
    store.record_usage("groq", "m", prompt_tokens=100, completion_tokens=200)
    store.record_usage("groq", "m", prompt_tokens=50, completion_tokens=25, is_429=True)
    assert store.usage_today("groq", "m") == {
        "requests": 2,
        "prompt_tokens": 150,
        "completion_tokens": 225,
        "errors_429": 1,
    }
    # a different model on the same provider keeps its own row
    assert store.usage_today("groq", "other")["requests"] == 0


def test_record_usage_is_one_atomic_statement(store):
    # A SELECT-then-UPDATE ledger loses increments silently under concurrency
    # (measured: 420 of 480 lost with 8 workers, no error raised). The whole
    # read-modify-write must happen inside ONE statement, under the write lock.
    statements = []
    store.conn.set_trace_callback(statements.append)
    try:
        store.record_usage("groq", "m", prompt_tokens=1, completion_tokens=1)
    finally:
        store.conn.set_trace_callback(None)
    assert len(statements) == 1, statements
    assert "ON CONFLICT" in statements[0].upper()


def test_reserve_budget_boundaries(store):
    day = "2026-08-11"
    store.record_usage("groq", "m", prompt_tokens=400, completion_tokens=200, day=day)
    store.record_usage("groq", "m", prompt_tokens=0, completion_tokens=0, day=day)
    # 600 tokens and 2 requests are on the ledger.
    assert store.reserve_budget("groq", "m", 400, limits={"tpd": 1000}, day=day) is True
    assert store.reserve_budget("groq", "m", 401, limits={"tpd": 1000}, day=day) is False
    assert store.reserve_budget("groq", "m", 0, limits={"rpd": 3}, day=day) is True
    assert store.reserve_budget("groq", "m", 0, limits={"rpd": 2}, day=day) is False
    assert store.reserve_budget("groq", "m", 10_000, limits={}, day=day) is True  # uncapped
    assert store.reserve_budget("groq", "m", 401, limits={"tpd": None}, day=day) is True
    both = {"tpd": 1000, "rpd": 99}
    assert store.reserve_budget("groq", "m", 401, limits=both, day=day) is False


def test_budget_rolls_over_at_the_day_boundary(store):
    store.record_usage("groq", "m", prompt_tokens=900, completion_tokens=100, day="2026-08-11")
    assert store.usage_today("groq", "m", day="2026-08-11")["prompt_tokens"] == 900
    assert store.usage_today("groq", "m", day="2026-08-12") == {
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "errors_429": 0,
    }
    assert store.reserve_budget("groq", "m", 500, limits={"tpd": 1000}, day="2026-08-11") is False
    assert store.reserve_budget("groq", "m", 500, limits={"tpd": 1000}, day="2026-08-12") is True


def test_record_usage_defaults_to_today_utc(store):
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    store.record_usage("groq", "m", prompt_tokens=5, completion_tokens=5)
    assert store.usage_today("groq", "m", day=today)["requests"] == 1
    assert store.usage_today("groq", "m")["requests"] == 1


# --------------------------------------------------------- 7. events/reconcile


def test_log_event_records_json_detail(store):
    store.log_event("wave_start", {"wave": 1, "n": 250})
    events = store.events("wave_start")
    assert len(events) == 1
    assert json.loads(events[0]["detail_json"]) == {"wave": 1, "n": 250}
    assert _TS_RE.match(events[0]["at"])


def test_reconcile_recovers_missing_rows_with_exact_offsets(store, tmp_path):
    _populate(store, n=3)
    raw = tmp_path / "raw" / "gen" / "2026-08-11" / "log.ndjson"

    envelopes = [_gen_envelope(f"t{i}", 1, answer=f"answer {i}") for i in range(3)]
    offsets = [append_ndjson(raw, env) for env in envelopes]
    judgement = {
        "kind": "judgement",
        "task_id": "t1",
        "attempt": 1,
        "judge_slot": "j1",
        "provider": "cerebras",
        "model": "judge-m",
        "grounding": 5,
        "validity": 4,
        "coverage": 3,
        "rationale": "grounded in the extract",
    }
    judgement_offset = append_ndjson(raw, judgement)

    # Only the first generation made it into the DB before the crash.
    store.record_generation(dict(envelopes[0], raw_path=str(raw), raw_offset=offsets[0]))

    assert store.reconcile_raw([raw]) == 3  # 2 generations + 1 judgement

    for i, env in enumerate(envelopes):
        row = store.latest_generation(f"t{i}")
        assert row is not None
        assert row["raw_path"] == str(raw)
        assert row["raw_offset"] == offsets[i]
        # the offset is a real seek target, not just a number we stored
        assert read_at(row["raw_path"], row["raw_offset"]) == env
        assert row["answer"] == f"answer {i}"
        assert row["provider"] == "groq"

    gen_t1 = store.latest_generation("t1")
    judged = store.judgements_for(gen_t1["gen_id"])
    assert len(judged) == 1
    assert judged[0]["judge_slot"] == "j1"
    assert judged[0]["grounding"] == 5
    assert judged[0]["raw_offset"] == judgement_offset
    assert read_at(judged[0]["raw_path"], judged[0]["raw_offset"]) == judgement

    # Idempotent: everything is indexed now, so a second sweep recovers nothing.
    assert store.reconcile_raw([raw]) == 0


def test_reconcile_skips_a_truncated_trailing_line(store, tmp_path):
    _populate(store, n=2)
    good = tmp_path / "good.ndjson"
    append_ndjson(good, _gen_envelope("t0", 1))
    crashed = tmp_path / "crashed.ndjson"
    offset = append_ndjson(crashed, _gen_envelope("t1", 1))
    with crashed.open("ab") as f:  # process died mid-record
        f.write(b'{"kind": "generation", "task_id": "t2", "att')

    assert store.reconcile_raw([good, crashed]) == 2
    assert store.latest_generation("t0") is not None
    assert store.latest_generation("t1")["raw_offset"] == offset
    bad = store.events("reconcile_bad_line")
    assert len(bad) == 1
    assert json.loads(bad[0]["detail_json"])["path"] == str(crashed)


def test_reconcile_offsets_survive_crlf_line_endings(store, tmp_path):
    _populate(store, n=2)
    raw = tmp_path / "crlf.ndjson"
    envelopes = [_gen_envelope("t0", 1), _gen_envelope("t1", 1)]
    raw.write_bytes(b"".join(json.dumps(e).encode("utf-8") + b"\r\n" for e in envelopes))

    assert store.reconcile_raw([raw]) == 2
    for i, env in enumerate(envelopes):
        row = store.latest_generation(f"t{i}")
        # A text-mode scan would collapse \r\n and put row 2 one byte early.
        assert read_at(raw, row["raw_offset"]) == env


def test_reconcile_handles_a_judgement_written_before_its_generation(store, tmp_path):
    _populate(store, n=1)
    raw = tmp_path / "out_of_order.ndjson"
    append_ndjson(
        raw,
        {"kind": "judgement", "task_id": "t0", "attempt": 1, "judge_slot": "j1", "grounding": 2},
    )
    append_ndjson(raw, _gen_envelope("t0", 1))
    assert store.reconcile_raw([raw]) == 2
    gen = store.latest_generation("t0")
    assert [r["judge_slot"] for r in store.judgements_for(gen["gen_id"])] == ["j1"]


def test_reconcile_logs_and_skips_unusable_records(store, tmp_path):
    _populate(store, n=1)
    raw = tmp_path / "junk.ndjson"
    append_ndjson(raw, _gen_envelope("nonexistent-task", 1))  # violates the task FK
    append_ndjson(raw, {"kind": "generation", "attempt": 1})  # no task_id
    append_ndjson(raw, {"kind": "judgement", "task_id": "t0", "attempt": 7})  # no such generation
    append_ndjson(raw, {"kind": "cost_report", "usd": 1.5})  # not an indexable kind
    raw.write_bytes(raw.read_bytes() + b"[1, 2, 3]\n")  # valid JSON, wrong shape

    assert store.reconcile_raw([raw, tmp_path / "does-not-exist.ndjson"]) == 0
    assert store.conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0] == 0
    kinds = [e["kind"] for e in store.events()]
    assert "reconcile_rejected" in kinds
    assert "reconcile_bad_record" in kinds
    assert "reconcile_orphan_judgement" in kinds
    assert "reconcile_unknown_kind" in kinds
    assert "reconcile_bad_line" in kinds
    assert "reconcile_missing_file" in kinds


def test_one_malformed_envelope_does_not_poison_the_sweep(store, tmp_path):
    # A field holding a nested object cannot be bound by the driver
    # (ProgrammingError, not IntegrityError). Before this was handled, a single
    # such record aborted reconcile AND rolled back every row recovered before
    # it - the crash-recovery path failing on exactly the garbage it exists for.
    _populate(store, n=2)
    raw = tmp_path / "mixed.ndjson"
    append_ndjson(raw, _gen_envelope("t0", 1, answer={"nested": "object"}))
    append_ndjson(raw, _gen_envelope("t1", 1, answer="perfectly fine"))

    assert store.reconcile_raw([raw]) == 1
    assert store.latest_generation("t0") is None
    assert store.latest_generation("t1")["answer"] == "perfectly fine"
    assert "reconcile_rejected" in [e["kind"] for e in store.events()]


def test_reconcile_does_not_overwrite_an_existing_row(store, tmp_path):
    _populate(store, n=1)
    raw = tmp_path / "log.ndjson"
    offset = append_ndjson(raw, _gen_envelope("t0", 1, answer="from the log"))
    # The DB already holds a richer row pointing somewhere else; reconcile
    # must leave it alone rather than rewriting it from the envelope.
    store.record_generation(
        _gen_envelope("t0", 1, raw_path="other.ndjson", raw_offset=4096, answer="from the db")
    )
    assert store.reconcile_raw([raw]) == 0
    row = store.latest_generation("t0")
    assert row["answer"] == "from the db"
    assert (row["raw_path"], row["raw_offset"]) != (str(raw), offset)
