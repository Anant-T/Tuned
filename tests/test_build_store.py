import json
import re
import sqlite3
import threading
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


def test_threads_sharing_one_handle_never_double_claim(tmp_path):
    # This suite avoids threads by policy. This test is the documented
    # exception, because the handle lock IS the subject under test and nothing
    # else can exercise it: the connection is opened check_same_thread=False,
    # so one Store can legitimately be shared by several worker threads.
    # Without a lock held across the whole transaction, their BEGIN/COMMIT
    # boundaries interleave on the single connection - the claim's
    # SELECT+UPDATE stops being indivisible and one thread's COMMIT publishes
    # another thread's half-built transaction. Review measured 4 genuine
    # double-claims this way. The two-handle test cannot catch it: separate
    # connections are serialised by SQLite itself, one connection is not.
    n_tasks, n_threads, batch = 400, 6, 7
    store = Store.open(tmp_path / "state" / "shared.sqlite3")
    try:
        store.upsert_source("ikanoon", "CC-BY-4.0")
        store.upsert_seeds(_seed_rows(n_tasks))
        store.create_tasks(_task_rows(n_tasks))

        claimed: list[str] = []
        errors: list[str] = []
        guard = threading.Lock()
        start = threading.Barrier(n_threads)

        def worker(wid):
            mine = []
            try:
                start.wait(timeout=10)
                while True:
                    rows = store.claim_tasks(f"worker-{wid}", batch)
                    if not rows:
                        break
                    mine.extend(r["task_id"] for r in rows)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                with guard:
                    errors.append(f"{type(exc).__name__}: {exc}")
            with guard:
                claimed.extend(mine)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert errors == []
        assert len(claimed) == n_tasks
        assert len(set(claimed)) == n_tasks  # zero double-claims
        assert store.task_counts() == {"generating": n_tasks}
    finally:
        store.close()


def test_set_task_state_fences_against_a_lost_lease(store):
    # A worker that stalled past its lease has already had its task reclaimed.
    # When it wakes up and reports, an unfenced UPDATE would clobber the live
    # holder and two workers would be generating one task with only one visible.
    _populate(store, n=1)
    store.claim_tasks("worker-A", 1)
    store.conn.execute("UPDATE task SET claimed_at = ? WHERE task_id = 't0'", (_ago(3600),))
    assert [r["task_id"] for r in store.claim_tasks("worker-B", 1)] == ["t0"]

    stale = store.set_task_state("t0", "accepted", disposition="from-A", expect_worker="worker-A")
    assert stale is False
    row = store.get_task("t0")
    assert row["state"] == "generating"  # untouched
    assert row["claimed_by"] == "worker-B"
    assert row["disposition"] is None

    # The live holder still succeeds.
    assert store.set_task_state("t0", "accepted", expect_worker="worker-B") is True
    assert store.get_task("t0")["state"] == "accepted"


def test_set_task_state_reports_whether_it_updated_anything(store):
    _populate(store, n=1)
    assert store.set_task_state("t0", "accepted") is True
    assert store.set_task_state("no-such-task", "accepted") is False


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


def test_latest_generations_come_back_in_ascending_gen_id_order(store):
    """The ORDER BY became DATASET-DEFINING in the assembly refactor and
    nothing defended it: decontaminate.store_items feeds these rows to dedupe,
    which keeps the FIRST row of a duplicate cluster and the first three rows
    of a case, so this order decides WHICH rows ship. Under DESC the pipeline
    yields the same count and a different set.

    The fixture is built so the two candidate orders DISAGREE: the tasks are
    recorded back to front, so gen_id order is the reverse of task_id order
    and a query with no ORDER BY at all cannot accidentally agree with it.
    """
    _populate(store, n=3)
    for task_id in ("t2", "t1", "t0"):
        store.record_generation(_gen_envelope(task_id, 1, raw_path="a", raw_offset=0))
    for task_id in ("t0", "t1", "t2"):
        store.set_task_state(task_id, "accepted")

    rows = store.latest_generations("accepted")
    assert [r["task_id"] for r in rows] == ["t2", "t1", "t0"]
    assert [r["gen_id"] for r in rows] == sorted(r["gen_id"] for r in rows)
    # ... and ascending gen_id is stable UNDER APPENDS, which is why it is the
    # one chosen: a later wave cannot displace a row that already shipped.
    store.record_generation(_gen_envelope("t0", 2, raw_path="a", raw_offset=99))
    store.set_task_state("t0", "accepted")
    after = store.latest_generations("accepted")
    assert [r["task_id"] for r in after] == ["t2", "t1", "t0"]


def test_an_empty_state_filter_is_a_state_and_not_the_absence_of_one(store):
    """`--state ''` shipped a REJECTED generation into the dataset: the clause
    was keyed on truthiness, so an empty string read as "no filter" and the
    assembly pass took every state there was."""
    _populate(store, n=2)
    store.record_generation(_gen_envelope("t0", 1, raw_path="a", raw_offset=0))
    store.record_generation(_gen_envelope("t1", 1, raw_path="a", raw_offset=10))
    store.set_task_state("t0", "accepted")
    store.set_task_state("t1", "rejected")

    assert len(store.latest_generations()) == 2           # None: every state
    assert len(store.latest_generations("accepted")) == 1
    assert store.latest_generations("") == []             # a state nothing holds


def test_the_task_states_the_workers_use_are_the_ones_the_store_lists():
    """TASK_STATES is what decontaminate.py validates --state against, so a
    state a worker can write but this tuple does not name would be refused at
    the CLI while sitting in the table."""
    from tuned.data import generate, judge, verify
    from tuned.data.store import TASK_STATES

    written = {
        generate.JUDGING_STATE, generate.PENDING_STATE, generate.REJECTED_STATE,
        generate.GEN_UNROUTABLE_STATE, judge.JUDGE_STATE_FROM, judge.JUDGE_STATE_TO,
        judge.ACCEPTED_STATE, judge.REJECTED_STATE, judge.SKIPPED_STATE,
        judge.ERROR_STATE, judge.UNROUTABLE_STATE, verify.REJECTED_STATE,
    }
    assert written <= set(TASK_STATES)
    assert len(TASK_STATES) == len(set(TASK_STATES))


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


def test_record_judgement_is_lease_fenced(store):
    """Every task-STATE write is fenced; a judgement write must be too. A
    worker that stalled past its lease has had its task legitimately
    re-claimed, and the live holder may already have bought its own slot and
    accepted the row - so the stale reply must not land on top of the scores
    that decision was actually made on."""
    _populate(store, n=1)
    gen_id = store.record_generation(_gen_envelope("t0", 1, raw_path="a", raw_offset=0))
    store.claim_tasks("worker-a", 1, stream="analysis")
    assert store.record_judgement(
        gen_id, "b", {"provider": "groq", "model": "m", "grounding": 5}, expect_worker="worker-a"
    ) is True

    # worker-a's lease expires and worker-b takes the task.
    store.conn.execute("UPDATE task SET claimed_at = ?", (_ago(3600),))
    store.claim_tasks("worker-b", 1, stream="analysis", state_from="generating", state_to="judging")
    assert store.record_judgement(
        gen_id, "b", {"provider": "groq", "model": "m", "grounding": 1}, expect_worker="worker-a"
    ) is False
    assert store.judgements_for(gen_id)[0]["grounding"] == 5  # untouched

    # The live holder still writes, and an unfenced write is unchanged.
    assert store.record_judgement(
        gen_id, "b", {"provider": "groq", "model": "m", "grounding": 2}, expect_worker="worker-b"
    ) is True
    assert store.record_judgement(gen_id, "b", {"provider": "groq", "model": "m", "grounding": 3})
    assert store.judgements_for(gen_id)[0]["grounding"] == 3


def test_record_judgement_fence_refuses_an_unknown_generation(store):
    _populate(store, n=1)
    gen_id = store.record_generation(_gen_envelope("t0", 1, raw_path="a", raw_offset=0))
    # Nobody holds the task at all: a fenced write has no lease to match.
    assert store.record_judgement(
        gen_id, "a", {"provider": "groq", "model": "m", "grounding": 4}, expect_worker="ghost"
    ) is False
    assert store.judgements_for(gen_id) == []


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


def test_judgement_binds_by_natural_key_not_a_stale_gen_id(store, tmp_path):
    # gen_id is an AUTOINCREMENT surrogate, so it is NOT stable across a
    # rebuild: recover the same logs into a fresh DB and the ids come out in a
    # different order. Here the judge envelope remembers a gen_id that now
    # belongs to t0's generation, while the answer it actually judged (t1) got
    # a different id. Trusting the surrogate silently attaches the judgement to
    # the wrong answer - corrupt training labels, not a crash.
    _populate(store, n=2)
    gen_t0 = store.record_generation(_gen_envelope("t0", 1, raw_path="a", raw_offset=0))
    gen_t1 = store.record_generation(_gen_envelope("t1", 1, raw_path="a", raw_offset=64))
    assert gen_t0 != gen_t1

    raw = tmp_path / "judge.ndjson"
    append_ndjson(
        raw,
        {
            "kind": "judgement",
            "task_id": "t1",
            "attempt": 1,
            "gen_id": gen_t0,  # stale surrogate from before the rebuild
            "judge_slot": "j1",
            "grounding": 5,
        },
    )

    assert store.reconcile_raw([raw]) == 1
    assert store.judgements_for(gen_t0) == []  # must NOT land on t0's answer
    assert [r["judge_slot"] for r in store.judgements_for(gen_t1)] == ["j1"]
    remapped = store.events("reconcile_gen_id_remapped")
    assert len(remapped) == 1
    detail = json.loads(remapped[0]["detail_json"])
    assert (detail["envelope_gen_id"], detail["resolved_gen_id"]) == (gen_t0, gen_t1)


def test_a_contradictory_envelope_gen_id_is_not_trusted(store, tmp_path):
    # No attempt, so the natural key cannot resolve and the surrogate is the
    # only lead - but it points at t0's row while the envelope says t1. Binding
    # it anyway is the misbind; refusing is correct.
    _populate(store, n=2)
    gen_t0 = store.record_generation(_gen_envelope("t0", 1, raw_path="a", raw_offset=0))
    raw = tmp_path / "judge.ndjson"
    append_ndjson(
        raw, {"kind": "judgement", "task_id": "t1", "gen_id": gen_t0, "judge_slot": "j1"}
    )

    assert store.reconcile_raw([raw]) == 0
    assert store.judgements_for(gen_t0) == []
    kinds = [e["kind"] for e in store.events()]
    assert "reconcile_gen_id_mismatch" in kinds
    assert "reconcile_orphan_judgement" in kinds


def test_a_matching_envelope_gen_id_is_still_usable(store, tmp_path):
    # The surrogate fallback must still work when nothing contradicts it.
    _populate(store, n=1)
    gen_id = store.record_generation(_gen_envelope("t0", 1, raw_path="a", raw_offset=0))
    raw = tmp_path / "judge.ndjson"
    append_ndjson(raw, {"kind": "judgement", "gen_id": gen_id, "judge_slot": "j1", "validity": 4})
    assert store.reconcile_raw([raw]) == 1
    assert [r["judge_slot"] for r in store.judgements_for(gen_id)] == ["j1"]


def test_reconcile_diagnostics_survive_a_failed_sweep(store, tmp_path, monkeypatch):
    # Diagnostics used to be log_event'd inside the sweep's own transaction, so
    # a rollback destroyed exactly the records explaining why it rolled back.
    _populate(store, n=1)
    raw = tmp_path / "log.ndjson"
    raw.write_bytes(b"{not json at all\n")
    append_ndjson(raw, _gen_envelope("t0", 1))

    def boom(*_args, **_kwargs):
        raise RuntimeError("disk fell over mid-sweep")

    monkeypatch.setattr(store, "_recover_generation", boom)
    with pytest.raises(RuntimeError, match="disk fell over"):
        store.reconcile_raw([raw])

    # The sweep's own writes rolled back ...
    assert store.conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0] == 0
    # ... but the diagnostic for the corrupt line outlived it.
    bad = store.events("reconcile_bad_line")
    assert len(bad) == 1
    assert json.loads(bad[0]["detail_json"])["offset"] == 0


def test_a_failure_on_a_later_file_keeps_earlier_files_recovered(store, tmp_path, monkeypatch):
    # Consequence of one transaction PER FILE rather than one per sweep: work
    # already committed is kept, and because reconcile is idempotent, re-running
    # finishes the job instead of redoing it.
    _populate(store, n=2)
    first, second = tmp_path / "one.ndjson", tmp_path / "two.ndjson"
    append_ndjson(first, _gen_envelope("t0", 1))
    append_ndjson(second, _gen_envelope("t1", 1))

    real = store._recover_generation

    def selective(rec, path, offset, diag):
        if rec.get("task_id") == "t1":
            raise RuntimeError("boom on the second file")
        return real(rec, path, offset, diag)

    monkeypatch.setattr(store, "_recover_generation", selective)
    with pytest.raises(RuntimeError, match="second file"):
        store.reconcile_raw([first, second])

    assert store.latest_generation("t0") is not None  # first file survived
    assert store.latest_generation("t1") is None

    monkeypatch.undo()
    assert store.reconcile_raw([first, second]) == 1  # re-run finishes the job
    assert store.latest_generation("t1") is not None


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


# --------------------------------------------------------- 9. artifact index


def test_record_artifact_indexes_one_downloaded_object(store):
    store.upsert_source("s3://bucket", "CC-BY-4.0", url="s3://bucket/")
    key = "data/pdf/year=2015/english/2015_1_1_20_EN.pdf"
    store.record_artifact(
        "s3://bucket",
        key,
        local_path="corpus/sc/2015.pdf",
        size_bytes=158371,
        sha256="ab" * 32,
        etag='"deadbeef"',
    )
    row = store.artifact("s3://bucket", key)
    assert row["local_path"] == "corpus/sc/2015.pdf"
    assert row["size_bytes"] == 158371
    assert row["sha256"] == "ab" * 32
    assert row["etag"] == '"deadbeef"'
    assert _TS_RE.match(row["fetched_at"])
    assert store.artifact("s3://bucket", "never/fetched.pdf") is None


def test_artifact_rows_need_a_registered_source(store):
    # The FK is what stops an acquisition run indexing bytes under a source
    # for which nobody ever recorded a licence.
    with pytest.raises(sqlite3.IntegrityError):
        store.record_artifact("s3://nope", "k", local_path="p", size_bytes=1, sha256="00")


def test_artifact_index_is_per_source_and_keyed_by_object_key(store):
    store.upsert_source("a", "CC-BY-4.0")
    store.upsert_source("b", "Apache-2.0")
    store.record_artifact("a", "k1", local_path="a/k1", size_bytes=10, sha256="aa")
    store.record_artifact("a", "k2", local_path="a/k2", size_bytes=20, sha256="bb")
    store.record_artifact("b", "k1", local_path="b/k1", size_bytes=30, sha256="cc")

    index = store.artifact_index("a")
    assert sorted(index) == ["k1", "k2"]
    # The same object_key under another source must not leak into this one -
    # two buckets partitioned by year both hold "year=2015/..." keys.
    assert index["k1"]["local_path"] == "a/k1"
    assert index["k2"]["size_bytes"] == 20
    assert store.artifact_count("a") == 2
    assert store.artifact_count("b") == 1
    assert store.artifact_count() == 3


def test_recording_the_same_object_again_updates_in_place(store):
    # Upstream replaced the object: the index must carry the new bytes, not
    # fork into two rows claiming the same key.
    store.upsert_source("a", "CC-BY-4.0")
    store.record_artifact("a", "k", local_path="a/k", size_bytes=10, sha256="aa")
    store.record_artifact("a", "k", local_path="a/k", size_bytes=11, sha256="bb")
    assert store.artifact_count("a") == 1
    row = store.artifact("a", "k")
    assert (row["size_bytes"], row["sha256"]) == (11, "bb")


# ------------------------------------------------------- 10. document index


def _doc(**over) -> dict:
    row = {
        "status": "ok",
        "text_path": "corpus/text/2015_1_1_20_EN.txt",
        "case_id": "C.A. 3221/2018",
        "citation": "[2015] 1 S.C.R. 1",
        "year": 2015,
        "pages": 21,
        "page_start": 1,
        "page_end": 20,
        "chars": 48000,
        "headnote_chars": 6100,
        "marker": "judgment_delivered_by",
        "sha256": "cd" * 32,
        "extract_version": 1,
        "meta": {"signals": ["HELD:"], "reportable": None},
    }
    row.update(over)
    return row


def test_record_document_indexes_one_extracted_judgment(store):
    store.upsert_source("s3://bucket", "CC-BY-4.0")
    key = "data/pdf/year=2015/english/2015_1_1_20_EN.pdf"
    store.record_document("s3://bucket", key, _doc())

    row = store.document("s3://bucket", key)
    assert row["status"] == "ok"
    assert row["reason"] is None
    assert row["text_path"] == "corpus/text/2015_1_1_20_EN.txt"
    assert (row["pages"], row["page_start"], row["page_end"]) == (21, 1, 20)
    assert (row["chars"], row["headnote_chars"]) == (48000, 6100)
    assert row["marker"] == "judgment_delivered_by"
    assert row["extract_version"] == 1
    # meta travels as JSON through the same _dumps path every other table uses
    assert json.loads(row["meta_json"])["signals"] == ["HELD:"]
    assert _TS_RE.match(row["extracted_at"])
    assert store.document("s3://bucket", "never/extracted.pdf") is None


def test_document_rows_need_a_registered_source(store):
    # Same FK reasoning as artifact: no text row may exist under a source
    # for which nobody recorded a licence.
    with pytest.raises(sqlite3.IntegrityError):
        store.record_document("s3://nope", "k", _doc())


def test_document_index_is_per_source_and_carries_the_resume_columns(store):
    store.upsert_source("a", "CC-BY-4.0")
    store.upsert_source("b", "CC-BY-4.0")
    store.record_document("a", "k1", _doc())
    store.record_document("a", "k2", _doc(status="quarantined", reason="no_judgment_start",
                                          text_path=None, chars=None))
    store.record_document("b", "k1", _doc(text_path="other/b.txt"))

    index = store.document_index("a")
    assert sorted(index) == ["k1", "k2"]
    # The three facts the resume decision reads, and nothing has to be
    # guessed from a NULL: status tells ok from quarantined, text_path says
    # where the bytes should be, extract_version says whether the rules that
    # produced them are still the rules.
    assert index["k1"]["status"] == "ok"
    assert index["k1"]["text_path"] == "corpus/text/2015_1_1_20_EN.txt"
    assert index["k1"]["extract_version"] == 1
    assert index["k2"]["status"] == "quarantined"
    assert index["k2"]["reason"] == "no_judgment_start"
    assert index["k2"]["text_path"] is None
    # Two sources partitioned by year both hold "year=2015/..." keys.
    assert store.document_index("b")["k1"]["text_path"] == "other/b.txt"


def test_document_index_does_not_carry_the_per_document_meta_blob(store):
    # ~40k rows are read in one go at the start of every extraction run. The
    # resume decision reads five columns; meta_json is the one column that is
    # unbounded per row, and carrying it would make resuming cost more than
    # it saves.
    store.upsert_source("a", "CC-BY-4.0")
    store.record_document("a", "k1", _doc(meta={"padding": "x" * 4096}))
    row = store.document_index("a")["k1"]
    assert "meta_json" not in row
    assert json.loads(store.document("a", "k1")["meta_json"])["padding"] == "x" * 4096


def test_re_extracting_a_document_replaces_its_row(store):
    # A rule change re-runs the extractor over a judgment already indexed:
    # the row must move, not fork into two claiming the same key.
    store.upsert_source("a", "CC-BY-4.0")
    store.record_document("a", "k", _doc(chars=100, extract_version=1))
    store.record_document("a", "k", _doc(status="quarantined", reason="headnote_residue",
                                         text_path=None, chars=None, extract_version=2))
    assert store.document_count("a") == 1
    row = store.document("a", "k")
    assert (row["status"], row["reason"], row["extract_version"]) == (
        "quarantined", "headnote_residue", 2,
    )
    assert row["text_path"] is None


def test_document_count_splits_by_status_and_by_source(store):
    store.upsert_source("a", "CC-BY-4.0")
    store.upsert_source("b", "CC-BY-4.0")
    store.record_document("a", "k1", _doc())
    store.record_document("a", "k2", _doc())
    store.record_document("a", "k3", _doc(status="quarantined", reason="body_too_short"))
    store.record_document("b", "k1", _doc())

    assert store.document_count() == 4
    assert store.document_count("a") == 3
    assert store.document_count("a", status="ok") == 2
    assert store.document_count("a", status="quarantined") == 1
    assert store.document_count("b", status="quarantined") == 0


def test_documents_reads_back_in_key_order_and_filters_by_status(store):
    # The manifest and --audit both walk this: key order is year order for
    # this corpus, so a sample taken along it is spread across the scope.
    store.upsert_source("a", "CC-BY-4.0")
    for key in ("year=2016/b.pdf", "year=2015/a.pdf", "year=2015/c.pdf"):
        store.record_document("a", key, _doc())
    store.record_document("a", "year=2015/q.pdf", _doc(status="quarantined", reason="no_text"))

    assert [row["object_key"] for row in store.documents("a", status="ok")] == [
        "year=2015/a.pdf", "year=2015/c.pdf", "year=2016/b.pdf",
    ]
    quarantined = store.documents("a", status="quarantined")
    assert [row["reason"] for row in quarantined] == ["no_text"]
    assert len(store.documents("a")) == 4


# ------------------------------------------------------- 11. chunk manifest


def _chunk_manifest_row(**over) -> dict:
    row = {
        "status": "ok", "reason": None, "tier": "packing", "why": "fallback",
        "chunk_count": 2, "seed_ids_json": ["c1", "c2"],
        "sha256": "ab" * 32, "extract_version": 5,
        "segment_version": 1, "chunk_version": 1, "roles_version": 1,
        "meta_json": {"degradation": {"from": "roles", "reason": "roles_backend_none"}},
    }
    row.update(over)
    return row


def test_record_chunk_manifest_indexes_one_chunked_document(store):
    store.upsert_source("a", "Public Domain")
    store.record_chunk_manifest("a", "k1", _chunk_manifest_row())

    row = store.chunk_manifest("a", "k1")
    assert row["status"] == "ok"
    assert row["tier"] == "packing"
    assert row["chunk_count"] == 2
    assert json.loads(row["seed_ids_json"]) == ["c1", "c2"]
    assert row["extract_version"] == 5
    assert _TS_RE.match(row["chunked_at"])
    assert store.chunk_manifest("a", "never-chunked") is None


def test_a_chunk_manifest_written_before_roles_version_existed_gains_the_column(tmp_path):
    # CREATE TABLE IF NOT EXISTS is a no-op against a database that already
    # has the table, so a column added to SCHEMA after the first shipped
    # shape never reaches an existing store - and the next record_ call
    # fails on a database that is otherwise perfectly good. Reproduced here
    # by creating the OLD table shape by hand and then opening the Store.
    db = tmp_path / "old.sqlite3"
    old = sqlite3.connect(db)
    old.executescript("""
      CREATE TABLE source (source_id TEXT PRIMARY KEY, license TEXT, url TEXT,
                           version TEXT, retrieved_at TEXT);
      CREATE TABLE chunk_manifest (
        source_id TEXT NOT NULL REFERENCES source(source_id),
        object_key TEXT NOT NULL, status TEXT NOT NULL, reason TEXT,
        tier TEXT, why TEXT, chunk_count INTEGER NOT NULL DEFAULT 0,
        seed_ids_json TEXT, sha256 TEXT, extract_version INTEGER,
        segment_version INTEGER, chunk_version INTEGER,
        meta_json TEXT, chunked_at TEXT,
        PRIMARY KEY (source_id, object_key));
    """)
    old.commit()
    old.close()

    with Store.open(db) as store:
        columns = {r["name"] for r in store.conn.execute("PRAGMA table_info(chunk_manifest)")}
        assert "roles_version" in columns
        store.upsert_source("a", "Public Domain")
        store.record_chunk_manifest("a", "k1", _chunk_manifest_row())
        assert store.chunk_manifest("a", "k1")["roles_version"] == 1
        store.ensure_schema()  # and re-running it must not try to add it twice


def test_chunk_manifest_rows_need_a_registered_source(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.record_chunk_manifest("nope", "k", _chunk_manifest_row())


def test_chunk_manifest_index_is_per_source_and_carries_the_resume_columns(store):
    store.upsert_source("a", "Public Domain")
    store.upsert_source("b", "Public Domain")
    store.record_chunk_manifest("a", "k1", _chunk_manifest_row())
    store.record_chunk_manifest("a", "k2", _chunk_manifest_row(status="empty", chunk_count=0,
                                                                seed_ids_json=[]))
    store.record_chunk_manifest("b", "k1", _chunk_manifest_row(sha256="cd" * 32))

    index = store.chunk_manifest_index("a")
    assert sorted(index) == ["k1", "k2"]
    # The columns the resume decision reads, unabridged - unlike
    # document_index this DOES carry seed_ids_json (the replace step needs
    # it on every run, not only the ones that find something stale).
    assert index["k1"]["sha256"] == "ab" * 32
    assert index["k1"]["extract_version"] == 5
    assert index["k1"]["segment_version"] == 1
    assert index["k1"]["chunk_version"] == 1
    assert index["k1"]["roles_version"] == 1
    assert json.loads(index["k1"]["seed_ids_json"]) == ["c1", "c2"]
    assert index["k2"]["status"] == "empty"
    assert store.chunk_manifest_index("b")["k1"]["sha256"] == "cd" * 32


def test_re_chunking_a_document_replaces_its_manifest_row(store):
    store.upsert_source("a", "Public Domain")
    store.record_chunk_manifest("a", "k", _chunk_manifest_row(chunk_count=2, seed_ids_json=["c1", "c2"]))
    store.record_chunk_manifest(
        "a", "k", _chunk_manifest_row(chunk_count=3, seed_ids_json=["c3", "c4", "c5"], sha256="ff" * 32)
    )
    assert store.chunk_manifest_count("a") == 1
    row = store.chunk_manifest("a", "k")
    assert row["chunk_count"] == 3
    assert json.loads(row["seed_ids_json"]) == ["c3", "c4", "c5"]
    assert row["sha256"] == "ff" * 32


def test_chunk_manifest_count_splits_by_source(store):
    store.upsert_source("a", "Public Domain")
    store.upsert_source("b", "Public Domain")
    store.record_chunk_manifest("a", "k1", _chunk_manifest_row())
    store.record_chunk_manifest("a", "k2", _chunk_manifest_row())
    store.record_chunk_manifest("b", "k1", _chunk_manifest_row())
    assert store.chunk_manifest_count() == 3
    assert store.chunk_manifest_count("a") == 2
    assert store.chunk_manifest_count("b") == 1


# ------------------------------------------------------- 12. delete_seeds


def test_delete_seeds_removes_the_named_rows_and_returns_the_count(store):
    store.upsert_source("a", "CC-BY-4.0")
    store.upsert_seeds(_seed_rows(3, source_id="a"))
    removed = store.delete_seeds(["sd0", "sd2"])
    assert removed == 2
    assert store.get_seed("sd0") is None
    assert store.get_seed("sd1") is not None
    assert store.get_seed("sd2") is None
    assert store.seed_count("a") == 1


def test_delete_seeds_ignores_unknown_ids_and_counts_only_rows_it_removed(store):
    store.upsert_source("a", "CC-BY-4.0")
    store.upsert_seeds(_seed_rows(2, source_id="a"))
    removed = store.delete_seeds(["sd0", "sd0", "never-existed"])
    assert removed == 1
    assert store.get_seed("sd0") is None


def test_delete_seeds_dedupes_before_the_statements_reach_sqlite(store):
    # The dedup's own docstring says it happens "before it reaches SQL", and
    # that sentence - not the return value - is what is observable. The
    # return value cannot see it: DELETE on an already-removed id matches
    # zero rows, so `removed` is 1 either way, which is why the test that
    # used to be named for deduplication could not have caught its removal.
    # set_trace_callback is a public sqlite3 API and counts the statements
    # the driver actually executed.
    store.upsert_source("a", "CC-BY-4.0")
    store.upsert_seeds(_seed_rows(3, source_id="a"))
    statements = []
    store.conn.set_trace_callback(statements.append)
    try:
        removed = store.delete_seeds(["sd0", "sd0", "sd1", "sd1", "sd0"])
    finally:
        store.conn.set_trace_callback(None)
    deletes = [s for s in statements if s.strip().upper().startswith("DELETE FROM SEED")]
    assert len(deletes) == 2  # two distinct ids, five ids in
    assert removed == 2
    assert store.get_seed("sd2") is not None


def test_delete_seeds_consumes_a_generator_exactly_once(store):
    # The same `list(dict.fromkeys(...))` also materializes the input, which
    # is what lets the count-before/count-after work at all: a generator
    # walked twice would be empty the second time.
    store.upsert_source("a", "CC-BY-4.0")
    store.upsert_seeds(_seed_rows(2, source_id="a"))
    assert store.delete_seeds(sid for sid in ["sd0", "sd1"]) == 2


def test_delete_seeds_of_an_empty_list_does_nothing(store):
    store.upsert_source("a", "CC-BY-4.0")
    store.upsert_seeds(_seed_rows(1, source_id="a"))
    assert store.delete_seeds([]) == 0
    assert store.seed_count("a") == 1


def test_delete_seeds_refuses_to_orphan_a_referencing_task(store):
    # A seed a task already points at must not vanish silently - that would
    # leave task.seed_id referencing nothing, and this store runs with
    # foreign_keys=ON specifically so that cannot happen unnoticed.
    _populate(store, n=1)
    with pytest.raises(sqlite3.IntegrityError):
        store.delete_seeds(["sd0"])


# ------------------------------------------------------- 13. seeds_by_source


def test_seeds_by_source_reads_back_in_seed_id_order(store):
    store.upsert_source("a", "CC-BY-4.0")
    store.upsert_source("b", "CC-BY-4.0")
    store.upsert_seeds([
        {"seed_id": "z1", "source_id": "a", "text": "t", "token_count": 1},
        {"seed_id": "a1", "source_id": "a", "text": "t", "token_count": 1},
        {"seed_id": "m1", "source_id": "b", "text": "t", "token_count": 1},
    ])
    rows = store.seeds_by_source("a")
    assert [r["seed_id"] for r in rows] == ["a1", "z1"]
    assert [r["seed_id"] for r in store.seeds_by_source("b")] == ["m1"]
    assert store.seeds_by_source("nonexistent-source") == []


def test_iter_seeds_by_source_yields_the_same_rows_in_the_same_order(store):
    store.upsert_source("a", "CC-BY-4.0")
    store.upsert_seeds(_seed_rows(7, source_id="a"))
    assert [r["seed_id"] for r in store.iter_seeds_by_source("a")] == [
        r["seed_id"] for r in store.seeds_by_source("a")
    ]
    assert list(store.iter_seeds_by_source("nonexistent-source")) == []


def test_iter_seeds_by_source_pages_and_survives_the_caller_deleting_as_it_walks(
    store, monkeypatch
):
    # chunks.py's seed driver deletes each parent row after replacing it, so
    # this walk must be robust to rows vanishing behind the cursor. Keyset
    # paging is; LIMIT/OFFSET would skip one unread row per delete, which is
    # why the page size is forced small enough here to cross a page boundary.
    store.upsert_source("a", "CC-BY-4.0")
    store.upsert_seeds(_seed_rows(9, source_id="a"))
    monkeypatch.setattr(type(store), "SEED_PAGE", 2)
    seen = []
    for row in store.iter_seeds_by_source("a"):
        seen.append(row["seed_id"])
        store.delete_seeds([row["seed_id"]])
    assert seen == [f"sd{i}" for i in range(9)]
    assert store.seed_count("a") == 0


# ------------------------------------------- gold labels and judge thresholds


def _judged(store, n=6):
    """n generations, each scored by two judges. Returns their gen_ids."""
    _populate(store, n=n)
    gen_ids = []
    for i in range(n):
        gen_id = store.record_generation(
            {**_gen_envelope(f"t{i}", attempt=1), "raw_path": "raw.ndjson", "raw_offset": i}
        )
        for slot, model in (("a", "judge/one"), ("b", "judge/two")):
            store.record_judgement(
                gen_id,
                slot,
                {"provider": "p", "model": model, "grounding": 5, "validity": 4, "coverage": 3},
            )
        gen_ids.append(gen_id)
    return gen_ids


def test_judged_generations_are_the_ones_a_judge_actually_scored(store):
    gen_ids = _judged(store, n=3)
    store.create_tasks(
        [
            {
                "task_id": "unjudged",
                "seed_id": "sd0",
                "stream": "analysis",
                "task_type": "reason",
                "prompt_id": "p1",
                "prompt_sha": "deadbeef",
                "sample_ix": 7,
            }
        ]
    )
    lonely = store.record_generation(
        {**_gen_envelope("unjudged"), "raw_path": "raw.ndjson", "raw_offset": 99}
    )
    rows = store.judged_generations()
    assert [row["gen_id"] for row in rows] == gen_ids
    assert lonely not in {row["gen_id"] for row in rows}
    # Joined to the task, because the calibration export stratifies on it.
    assert {row["stream"] for row in rows} == {"analysis"}
    assert rows[0]["task_type"] == "reason"


def test_judged_generations_filters_by_stream_and_reads_an_empty_filter_as_empty(store):
    _judged(store, n=2)
    assert len(store.judged_generations(["analysis"])) == 2
    assert store.judged_generations(["transition"]) == []
    # An empty list is "no streams", not "every stream": the caller asked for
    # nothing and getting everything back is the expensive direction to be
    # wrong in.
    assert store.judged_generations([]) == []


def test_judgements_by_gen_reads_many_at_once_and_matches_the_row_at_a_time_path(store):
    gen_ids = _judged(store, n=4)
    bulk = store.judgements_by_gen(gen_ids)
    assert set(bulk) == set(gen_ids)
    for gen_id in gen_ids:
        assert bulk[gen_id] == store.judgements_for(gen_id)
    # A generation nobody judged comes back as an empty list rather than
    # missing, so the caller's dict lookup cannot KeyError on it.
    assert store.judgements_by_gen([9999]) == {9999: []}


def test_gold_labels_round_trip_and_the_second_file_wins(store):
    gen_ids = _judged(store, n=3)
    assert store.upsert_gold_labels(
        [{"gen_id": gen_ids[0], "verdict": "accept", "fold": 0}]
    ) == 1
    assert store.gold_label_count() == 1
    row = store.gold_labels()[0]
    assert (row["verdict"], row["fold"]) == ("accept", 0)
    assert _TS_RE.match(row["labeled_at"])

    store.upsert_gold_labels([{"gen_id": gen_ids[0], "verdict": "reject", "fold": 5}])
    assert store.gold_label_count() == 1
    assert store.gold_labels()[0]["verdict"] == "reject"
    assert store.upsert_gold_labels([]) == 0


def test_a_gold_label_for_a_generation_that_does_not_exist_is_refused(store):
    # The label is irreplaceable operator hours attached to a specific
    # generation; a dangling one is hours nobody can trace back to an answer.
    with pytest.raises(sqlite3.IntegrityError):
        store.upsert_gold_labels([{"gen_id": 4242, "verdict": "accept", "fold": 0}])


def test_recording_thresholds_deactivates_the_previous_calibration(store):
    first = [
        {"calib_id": "c1", "model": "judge/one", "rule": "min_axis", "threshold": 4},
        {"calib_id": "c2", "model": "judge/two", "rule": "mean", "threshold": 4},
    ]
    assert store.record_judge_thresholds(first) == 2
    assert [row["model"] for row in store.judge_thresholds()] == ["judge/one", "judge/two"]
    assert all(row["active"] == 1 for row in store.judge_thresholds())

    store.record_judge_thresholds(
        [{"calib_id": "c3", "model": "judge/one", "rule": "both", "threshold": 5}]
    )
    live = store.judge_thresholds()
    assert [row["calib_id"] for row in live] == ["c3"]
    # Superseded rows are KEPT: a later report is only interpretable against
    # the fit it replaced.
    everything = store.judge_thresholds(active_only=False)
    assert {row["calib_id"] for row in everything} == {"c1", "c2", "c3"}
    assert {row["calib_id"] for row in everything if row["active"] == 0} == {"c1", "c2"}


def test_recording_no_thresholds_leaves_the_previous_calibration_alone(store):
    store.record_judge_thresholds([{"calib_id": "c1", "model": "m", "rule": "mean", "threshold": 4}])
    assert store.record_judge_thresholds([]) == 0
    assert [row["calib_id"] for row in store.judge_thresholds()] == ["c1"]
