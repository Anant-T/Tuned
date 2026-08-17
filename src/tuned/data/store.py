"""SQLite state store for the law_v1 generation pipeline - the only module that runs SQL.

DURABILITY RULE (the reason this module is shaped the way it is)
----------------------------------------------------------------
The database is a *derived index*, never the system of record.  Every paid
API result is appended to an immutable raw NDJSON log FIRST - via
``tuned.data.jsonl.append_ndjson``, which returns the byte offset the record
landed at - and only then indexed here as a row carrying
``(raw_path, raw_offset)``.  A crash between those two writes therefore
loses an index row, never a response somebody paid for:
``reconcile_raw()`` re-scans the raw logs and rebuilds whatever rows the DB
is missing, and ``jsonl.read_at(raw_path, raw_offset)`` seeks straight back
to the original envelope for any row in the DB.  Nothing may be written to
SQLite that is not already durable in a raw log.

Conventions
-----------
* Timestamps are fixed-width ISO-8601 UTC strings (``utcnow()``), so
  lexicographic order *is* chronological order - lease expiry is a plain
  string ``<`` inside SQL, with no parsing.
* ``*_json`` columns hold TEXT.  Writers may pass a dict/list and it gets
  serialised for them; readers get the TEXT back and decode it themselves.
  Nothing is silently decoded on the way out.
* The connection runs in explicit-transaction mode (``isolation_level=None``).
  Every multi-statement write goes through ``_write_txn()``, i.e.
  ``BEGIN IMMEDIATE`` - see ``claim_tasks`` for why that matters.
"""

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

_TS_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"

# How long a claim owns a task before another worker may recover it.  Public
# because it is a fact about the store's leases and other modules reason about
# it (verify.py refuses to write task states while any lease is live); a
# second copy elsewhere is a fence that silently disagrees with the fencing.
DEFAULT_LEASE_S = 900

# Every state a task row can hold, owned here because the task table is. A
# reader that takes a state from the operator (decontaminate.py's --state)
# validates against this, so a typo is a refusal rather than a zero-row read
# that still reports the generations as screened. generate.py and judge.py own
# the transitions; test_build_store pins their constants against this list, so
# a new state that is not added here fails there rather than here.
TASK_STATES = (
    "pending",
    "generating",
    "judging",
    "judging_active",
    "accepted",
    "rejected",
    "gen_unroutable",
    "judge_skipped",
    "judge_error",
    "judge_unroutable",
)

# ORDER IS LOAD-BEARING: busy_timeout must be armed BEFORE journal_mode=WAL.
# Switching the journal mode needs a brief exclusive lock, and with the default
# timeout of 0 a second worker opening the same DB at the same moment fails
# instantly with "database is locked" instead of waiting - which is precisely
# the restart-the-whole-fleet-at-once case this store exists to survive.
# (Reproduced with 8 concurrent opens; see the pragma-order regression test.)
_PRAGMAS = (
    "busy_timeout=5000",
    "journal_mode=WAL",
    "synchronous=NORMAL",
    "foreign_keys=ON",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS source (
  source_id TEXT PRIMARY KEY, license TEXT NOT NULL, url TEXT, version TEXT, retrieved_at TEXT);
CREATE TABLE IF NOT EXISTS seed (
  seed_id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL REFERENCES source(source_id),
  native_id TEXT, cnr TEXT, neutral_citation TEXT,
  court TEXT, decision_date TEXT, offence_date TEXT,
  case_type TEXT, code_era TEXT,
  text TEXT NOT NULL, token_count INTEGER,
  roles_json TEXT, answer_key_json TEXT, meta_json TEXT);
CREATE INDEX IF NOT EXISTS seed_by_source ON seed(source_id, case_type, code_era);
CREATE TABLE IF NOT EXISTS task (
  task_id TEXT PRIMARY KEY,
  seed_id TEXT NOT NULL REFERENCES seed(seed_id),
  stream TEXT NOT NULL, task_type TEXT NOT NULL,
  prompt_id TEXT NOT NULL, prompt_sha TEXT NOT NULL,
  sample_ix INTEGER NOT NULL, arm TEXT,
  state TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0,
  disposition TEXT, claimed_by TEXT, claimed_at TEXT,
  created_at TEXT, updated_at TEXT);
CREATE INDEX IF NOT EXISTS task_pending ON task(state, stream, claimed_at);
CREATE TABLE IF NOT EXISTS generation (
  gen_id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL REFERENCES task(task_id), attempt INTEGER NOT NULL,
  provider TEXT NOT NULL, model TEXT NOT NULL, model_family TEXT,
  params_json TEXT, raw_path TEXT NOT NULL, raw_offset INTEGER NOT NULL,
  think TEXT, answer TEXT,
  prompt_tokens INTEGER, completion_tokens INTEGER, think_tokens INTEGER, total_tokens INTEGER,
  latency_ms INTEGER, finish_reason TEXT, error TEXT, created_at TEXT,
  UNIQUE(task_id, attempt));
CREATE TABLE IF NOT EXISTS gate_result (
  gen_id INTEGER NOT NULL REFERENCES generation(gen_id),
  gate TEXT NOT NULL, passed INTEGER NOT NULL, detail_json TEXT,
  PRIMARY KEY (gen_id, gate));
CREATE TABLE IF NOT EXISTS judgement (
  gen_id INTEGER NOT NULL REFERENCES generation(gen_id),
  judge_slot TEXT NOT NULL, provider TEXT, model TEXT,
  grounding INTEGER, validity INTEGER, coverage INTEGER,
  rationale TEXT, raw_path TEXT, raw_offset INTEGER, created_at TEXT,
  PRIMARY KEY (gen_id, judge_slot));
CREATE TABLE IF NOT EXISTS gold_label (
  gen_id INTEGER PRIMARY KEY REFERENCES generation(gen_id),
  verdict TEXT NOT NULL, grounding INTEGER, validity INTEGER, coverage INTEGER,
  notes TEXT, labeled_at TEXT, fold INTEGER);
CREATE TABLE IF NOT EXISTS judge_threshold (
  calib_id TEXT PRIMARY KEY, judge_slot TEXT, model TEXT, rule TEXT,
  threshold INTEGER, precision REAL, recall REAL, n_gold INTEGER,
  fitted_at TEXT, active INTEGER);
CREATE TABLE IF NOT EXISTS budget_ledger (
  day TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
  requests INTEGER NOT NULL DEFAULT 0, prompt_tokens INTEGER NOT NULL DEFAULT 0,
  completion_tokens INTEGER NOT NULL DEFAULT 0, errors_429 INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (day, provider, model));
CREATE TABLE IF NOT EXISTS run_event (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, kind TEXT, detail_json TEXT);
CREATE TABLE IF NOT EXISTS artifact (
  source_id TEXT NOT NULL REFERENCES source(source_id),
  object_key TEXT NOT NULL, local_path TEXT NOT NULL,
  size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, etag TEXT, fetched_at TEXT,
  PRIMARY KEY (source_id, object_key));
CREATE TABLE IF NOT EXISTS document (
  source_id TEXT NOT NULL REFERENCES source(source_id),
  object_key TEXT NOT NULL, status TEXT NOT NULL, reason TEXT, text_path TEXT,
  case_id TEXT, citation TEXT, year INTEGER,
  pages INTEGER, page_start INTEGER, page_end INTEGER,
  chars INTEGER, headnote_chars INTEGER, marker TEXT,
  sha256 TEXT, extract_version INTEGER, meta_json TEXT, extracted_at TEXT,
  PRIMARY KEY (source_id, object_key));
CREATE INDEX IF NOT EXISTS document_by_status ON document(source_id, status);
CREATE TABLE IF NOT EXISTS chunk_manifest (
  source_id TEXT NOT NULL REFERENCES source(source_id),
  object_key TEXT NOT NULL, status TEXT NOT NULL, reason TEXT,
  tier TEXT, why TEXT, chunk_count INTEGER NOT NULL DEFAULT 0,
  seed_ids_json TEXT,
  sha256 TEXT, extract_version INTEGER,
  segment_version INTEGER, chunk_version INTEGER, roles_version INTEGER,
  meta_json TEXT, chunked_at TEXT,
  PRIMARY KEY (source_id, object_key));
"""

# Columns added to an existing table after its first shipped shape. CREATE
# TABLE IF NOT EXISTS is a no-op against a database that already has the
# table, so a column added to SCHEMA above would never reach one - and the
# next INSERT would fail on a database that is otherwise perfectly good.
# Kept as an explicit, idempotent list rather than a migration framework
# because there is exactly one entry and each is one ALTER.
_ADDED_COLUMNS = (("chunk_manifest", "roles_version", "INTEGER"),)

_SEED_COLS = (
    "seed_id", "source_id", "native_id", "cnr", "neutral_citation",
    "court", "decision_date", "offence_date", "case_type", "code_era",
    "text", "token_count", "roles_json", "answer_key_json", "meta_json",
)
_TASK_COLS = (
    "task_id", "seed_id", "stream", "task_type", "prompt_id", "prompt_sha",
    "sample_ix", "arm", "state", "attempts", "disposition",
    "claimed_by", "claimed_at", "created_at", "updated_at",
)
_GEN_COLS = (
    "task_id", "attempt", "provider", "model", "model_family", "params_json",
    "raw_path", "raw_offset", "think", "answer",
    "prompt_tokens", "completion_tokens", "think_tokens", "total_tokens",
    "latency_ms", "finish_reason", "error", "created_at",
)
_JUDGEMENT_COLS = (
    "gen_id", "judge_slot", "provider", "model",
    "grounding", "validity", "coverage", "rationale",
    "raw_path", "raw_offset", "created_at",
)
_GOLD_COLS = (
    "gen_id", "verdict", "grounding", "validity", "coverage",
    "notes", "labeled_at", "fold",
)
_THRESHOLD_COLS = (
    "calib_id", "judge_slot", "model", "rule", "threshold",
    "precision", "recall", "n_gold", "fitted_at", "active",
)
_DOCUMENT_COLS = (
    "source_id", "object_key", "status", "reason", "text_path",
    "case_id", "citation", "year",
    "pages", "page_start", "page_end",
    "chars", "headnote_chars", "marker",
    "sha256", "extract_version", "meta_json", "extracted_at",
)
# The five columns the extraction resume decision reads, and no more: the
# index is loaded whole at the start of every run (see document_index).
_DOCUMENT_RESUME_COLS = ("object_key", "status", "reason", "text_path", "extract_version")
_CHUNK_MANIFEST_COLS = (
    "source_id", "object_key", "status", "reason", "tier", "why",
    "chunk_count", "seed_ids_json", "sha256", "extract_version",
    "segment_version", "chunk_version", "roles_version", "meta_json", "chunked_at",
)
# The columns chunks.py's resume decision reads: whether this document/seed
# was chunked before, under which rules, and (seed_ids_json) what to delete
# before writing its replacement. Unlike _DOCUMENT_RESUME_COLS this DOES
# carry seed_ids_json - a few hundred bytes per row at most (a document
# backs a few dozen chunks, not the unbounded per-document blob meta_json
# is) - because the replace-not-duplicate rule needs it on every resumed run,
# not only on the runs that actually replace something.
_CHUNK_MANIFEST_RESUME_COLS = (
    "object_key", "status", "sha256", "extract_version",
    "segment_version", "chunk_version", "roles_version", "seed_ids_json",
)


# Errors that mean "this raw record is unusable" rather than "the database is
# broken": a constraint violation (unknown task_id, missing NOT NULL column) or
# a value the driver cannot bind (an envelope field holding a nested object).
# Both are skippable during reconcile. OperationalError/DatabaseError - disk
# full, I/O error, locked - deliberately stay unhandled: silently "recovering"
# through a failing disk would report success while losing data.
_UNUSABLE_RECORD = (sqlite3.IntegrityError, sqlite3.ProgrammingError, sqlite3.InterfaceError)


def utcnow() -> str:
    """Now as a fixed-width ISO-8601 UTC string; sorts lexicographically."""
    return datetime.now(UTC).strftime(_TS_FMT)


def utcday(day: str | None = None) -> str:
    """`day` if given, else today UTC as YYYY-MM-DD (the budget-ledger key)."""
    return day if day is not None else datetime.now(UTC).strftime("%Y-%m-%d")


def _lease_cutoff(lease_s: int) -> str:
    """Timestamp `lease_s` seconds ago - claims older than this are stale."""
    return (datetime.now(UTC) - timedelta(seconds=lease_s)).strftime(_TS_FMT)


def _dumps(value) -> str | None:
    """Serialise a *_json value; strings and None pass through untouched."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _pack(row: Mapping, cols: tuple[str, ...]) -> list:
    """Project a dict onto a fixed column tuple, serialising *_json values.

    Unknown keys are ignored on purpose: workers hand the same envelope dict
    to the raw log and to the DB, and the envelope carries extra routing keys
    ("kind", ...) that no column owns.
    """
    return [_dumps(row.get(c)) if c.endswith("_json") else row.get(c) for c in cols]


def _insert_sql(table: str, cols: tuple[str, ...], verb: str = "INSERT") -> str:
    return (
        f"{verb} INTO {table} ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' * len(cols))})"
    )


def _fill(row: dict, **defaults) -> dict:
    """Apply defaults for keys that are absent *or explicitly None*."""
    for key, value in defaults.items():
        if row.get(key) is None:
            row[key] = value
    return row


def _diag(sink: list, kind: str, detail: dict) -> None:
    """Buffer one reconcile diagnostic, stamped when it happened.

    Diagnostics are NOT written through log_event during a sweep: they would
    live inside the sweep's transaction and be rolled back by the very failure
    they explain. They are flushed separately once the sweep is over.
    """
    sink.append((utcnow(), kind, _dumps(detail)))


def _cap(limits: Mapping, key: str) -> float:
    """Read a budget cap; missing or None means unlimited."""
    value = limits.get(key)
    return float("inf") if value is None else value


class Store:
    """Typed, transaction-safe wrapper around the pipeline's SQLite file."""

    def __init__(self, conn: sqlite3.Connection, path: Path):
        self._conn = conn
        self.path = path
        # The connection is opened check_same_thread=False, so ONE handle can be
        # shared by several worker threads. SQLite serialises single statements
        # for us, but a transaction is not a statement: without this lock two
        # threads interleave their BEGIN/COMMIT on the same connection, so one
        # thread's claim SELECT+UPDATE stops being indivisible (measured: real
        # double-claims) and one thread's COMMIT publishes another's half-built
        # transaction. Every write path takes it; it is held for the WHOLE
        # transaction, not per statement.
        # RLock, not Lock: a nested _write_txn is a programming error, and an
        # RLock surfaces it as a loud "cannot start a transaction within a
        # transaction" instead of hanging the worker forever on a deadlock.
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- lifecycle

    @classmethod
    def open(cls, db_path: str | Path) -> "Store":
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: no implicit BEGIN from the driver, so this
        # module owns every transaction boundary explicitly (BEGIN IMMEDIATE).
        conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        for pragma in _PRAGMAS:
            conn.execute(f"PRAGMA {pragma}")
        store = cls(conn, db_path)
        store.ensure_schema()
        return store

    @property
    def conn(self) -> sqlite3.Connection:
        """Escape hatch for maintenance/repair scripts. Pipeline code uses the API."""
        return self._conn

    def ensure_schema(self) -> None:
        # executescript issues an implicit COMMIT first, so it must not run
        # while another thread holds an open transaction on this handle.
        with self._lock:
            self._conn.executescript(SCHEMA)
            for table, column, decl in _ADDED_COLUMNS:
                present = {
                    row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")
                }
                if column not in present:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @contextmanager
    def _write_txn(self) -> Iterator[sqlite3.Connection]:
        """BEGIN IMMEDIATE ... COMMIT, rolling back on any exception.

        The handle lock is held for the whole transaction so that concurrent
        threads sharing this Store serialise instead of interleaving their
        transaction boundaries. It is released as the exception unwinds.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    # Some errors make SQLite unwind the transaction itself; a
                    # "no transaction is active" here must not mask the real one.
                    pass
                raise
            self._conn.execute("COMMIT")

    def _write(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """One autocommit write, serialised against this handle's transactions.

        Without the lock a standalone INSERT/UPDATE issued from another thread
        gets swallowed into whatever transaction is currently open on this
        connection - committing at that transaction's time, or vanishing with
        its ROLLBACK. rowcount/lastrowid are captured on the returned cursor at
        execute time, so reading them after the lock is released is safe.
        """
        with self._lock:
            return self._conn.execute(sql, params)

    # ------------------------------------------------------------ sources/seeds

    def upsert_source(
        self, source_id: str, license: str, url: str | None = None, version: str | None = None
    ) -> None:
        self._write(
            "INSERT OR REPLACE INTO source (source_id, license, url, version, retrieved_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (source_id, license, url, version, utcnow()),
        )

    def upsert_seeds(self, rows: Iterable[dict]) -> int:
        """INSERT OR REPLACE seeds in one transaction; returns rows written.

        REPLACE is safe here even with foreign_keys=ON: the primary key is
        unchanged, so the implicit delete+insert nets to zero FK violations at
        statement end and dependent task rows survive (regression-tested).
        """
        payload = [_pack(dict(row), _SEED_COLS) for row in rows]
        if not payload:
            return 0
        before = self._conn.total_changes
        with self._write_txn() as conn:
            conn.executemany(_insert_sql("seed", _SEED_COLS, "INSERT OR REPLACE"), payload)
        return self._conn.total_changes - before

    def seed_count(self, source_id: str | None = None) -> int:
        if source_id is None:
            return self._conn.execute("SELECT COUNT(*) FROM seed").fetchone()[0]
        return self._conn.execute(
            "SELECT COUNT(*) FROM seed WHERE source_id = ?", (source_id,)
        ).fetchone()[0]

    def get_seed(self, seed_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM seed WHERE seed_id = ?", (seed_id,)).fetchone()
        return dict(row) if row is not None else None

    # How many seed rows iter_seeds_by_source holds at once. Whole
    # InJudgements judgments are the largest text in this store, and
    # DEFAULT_LIMITS puts 4,000 of them under one source_id.
    SEED_PAGE = 200

    def iter_seeds_by_source(self, source_id: str) -> Iterator[dict]:
        """seeds_by_source, one page at a time - same rows, same order.

        The streaming half of the same query, and the one chunks.py's seed
        driver uses: that driver reads WHOLE judgment text and the eager
        version holds every row of it in memory for the length of the pass.

        Paged by KEYSET (`seed_id > last seen`), not LIMIT/OFFSET, because
        the caller mutates the table as it walks it - chunk_seed_rows
        deletes each parent row after replacing it, and under OFFSET every
        such delete would silently skip the next unread row. Rows the caller
        INSERTS ahead of the cursor (a document's own chunks) are visited
        and recognised by their meta, exactly as they are today.
        """
        last = ""
        while True:
            page = self._conn.execute(
                "SELECT * FROM seed WHERE source_id = ? AND seed_id > ? "
                "ORDER BY seed_id LIMIT ?",
                (source_id, last, self.SEED_PAGE),
            ).fetchall()
            if not page:
                return
            for row in page:
                yield dict(row)
            last = page[-1]["seed_id"]

    def seeds_by_source(self, source_id: str) -> list[dict]:
        """Every seed row under one source, in seed_id order.

        seeds.py's InJudgements rows carry whole judgment text with no
        on-disk manifest of their own (unlike the document table), so
        finding "which whole rows still need chunking" has nowhere else to
        read from. seed_id order rather than insertion order, for the same
        reason document rows are walked in object_key order - a
        deterministic, machine-independent scan a resumed or `--limit`ed run
        can repeat.

        Materializes. Callers walking the whole source under memory
        pressure want iter_seeds_by_source above; this stays for the callers
        that genuinely want the list (tests, counts, ad-hoc inspection).
        """
        return [
            dict(row)
            for row in self._conn.execute(
                "SELECT * FROM seed WHERE source_id = ? ORDER BY seed_id", (source_id,)
            ).fetchall()
        ]

    # --------------------------------------------------------------- artifacts

    def record_artifact(
        self,
        source_id: str,
        object_key: str,
        *,
        local_path: str | Path,
        size_bytes: int,
        sha256: str,
        etag: str | None = None,
    ) -> None:
        """Index one acquired object - the corpus-phase twin of
        record_generation, and under the same durability rule: the bytes are
        already at `local_path` before this row claims they are.

        A crash between the two therefore costs an index row, never the
        download: acquire.py re-derives the row by hashing the file that is
        already on disk (its "adopt" path), exactly as reconcile_raw
        re-derives generation rows from the raw logs.

        INSERT OR REPLACE on (source_id, object_key): the SC bucket is a
        rolling release, so an object can genuinely change under a key we
        already hold, and re-acquiring must move the row rather than fork it.
        `etag` is not a content hash - it is the object's MD5 for a
        single-part upload but "<md5>-<parts>" for a multipart one - so
        nothing may verify content AGAINST it, and size_bytes/sha256 are what
        verification uses. Its INEQUALITY is still informative, though:
        recorded != listed means the object was re-uploaded under this key,
        which is why acquire.fetch_decision reads it.
        """
        self._write(
            "INSERT OR REPLACE INTO artifact "
            "(source_id, object_key, local_path, size_bytes, sha256, etag, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (source_id, object_key, str(local_path), int(size_bytes), sha256, etag, utcnow()),
        )

    def artifact(self, source_id: str, object_key: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM artifact WHERE source_id = ? AND object_key = ?",
            (source_id, object_key),
        ).fetchone()
        return dict(row) if row is not None else None

    def artifact_index(self, source_id: str) -> dict[str, dict]:
        """Every indexed object for one source, keyed by object_key.

        Read ONCE per acquisition run rather than per object: the resume
        decision is taken for each of ~100k keys, and a SELECT apiece would
        make restarting an interrupted sync cost more than the sync.
        """
        return {
            row["object_key"]: dict(row)
            for row in self._conn.execute(
                "SELECT * FROM artifact WHERE source_id = ?", (source_id,)
            ).fetchall()
        }

    def artifact_count(self, source_id: str | None = None) -> int:
        if source_id is None:
            return self._conn.execute("SELECT COUNT(*) FROM artifact").fetchone()[0]
        return self._conn.execute(
            "SELECT COUNT(*) FROM artifact WHERE source_id = ?", (source_id,)
        ).fetchone()[0]

    # --------------------------------------------------------------- documents

    def record_document(self, source_id: str, object_key: str, row: dict) -> None:
        """Index one extracted judgment - the text-phase twin of
        record_artifact, and under the same durability rule: the text file is
        already at `text_path` before this row claims it is.

        A crash between the two costs an index row, never the text: the next
        run sees a key with no row and extracts it again. That is cheaper
        here than acquire.py's adopt path (one PDF re-parsed, not one object
        re-downloaded), which is why there is no adopt here.

        A QUARANTINED document is a row too, with `status='quarantined'`, the
        reason, and no text_path. Recording the refusal is what makes an
        interrupted run resumable in both directions: a document nothing may
        emit must not be silently re-attempted forever, and it must not be
        invisible to the operator either.

        INSERT OR REPLACE on (source_id, object_key): re-extracting under new
        rules moves the row rather than forking it, and `extract_version` is
        what tells the two apart.
        """
        packed = dict(row)
        packed["source_id"] = source_id
        packed["object_key"] = object_key
        if "meta" in packed and packed.get("meta_json") is None:
            packed["meta_json"] = packed.pop("meta")
        self._write(
            _insert_sql("document", _DOCUMENT_COLS, "INSERT OR REPLACE"),
            _pack(_fill(packed, extracted_at=utcnow()), _DOCUMENT_COLS),
        )

    def document(self, source_id: str, object_key: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM document WHERE source_id = ? AND object_key = ?",
            (source_id, object_key),
        ).fetchone()
        return dict(row) if row is not None else None

    def document_index(self, source_id: str) -> dict[str, dict]:
        """Every indexed document for one source, keyed by object_key.

        Read ONCE per extraction run, exactly as artifact_index is - the
        resume decision is taken for tens of thousands of keys and a SELECT
        apiece would make restarting cost more than the work it skips.
        Unlike artifact_index this is NOT `SELECT *`: meta_json is unbounded
        per row and no resume decision reads it, so the whole-corpus read
        stays a few hundred bytes a row.
        """
        cols = ", ".join(_DOCUMENT_RESUME_COLS)
        return {
            row["object_key"]: dict(row)
            for row in self._conn.execute(
                f"SELECT {cols} FROM document WHERE source_id = ?", (source_id,)
            ).fetchall()
        }

    def documents(self, source_id: str, status: str | None = None) -> list[dict]:
        """Full document rows in object_key order, optionally one status only.

        Key order is year order for this corpus (the keys are
        `.../year=YYYY/english/...`), so a sample walked along it is spread
        across the scope rather than concentrated in whatever the extractor
        reached first.
        """
        sql = "SELECT * FROM document WHERE source_id = ?"
        params: tuple = (source_id,)
        if status is not None:
            sql += " AND status = ?"
            params += (status,)
        return [
            dict(row)
            for row in self._conn.execute(sql + " ORDER BY object_key", params).fetchall()
        ]

    def document_count(self, source_id: str | None = None, *, status: str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM document"
        clauses, params = [], []
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return self._conn.execute(sql, tuple(params)).fetchone()[0]

    # ---------------------------------------------------------- chunk manifest

    def record_chunk_manifest(self, source_id: str, object_key: str, row: dict) -> None:
        """Index one chunked document/seed - chunks.py's resume twin of
        record_document, over the same (source_id, object_key) identity
        convention `document` and `artifact` already use. `object_key` is
        whatever chunks.py's caller says identifies the thing that was
        chunked: extract.py's object_key for a document-table judgment, or
        the parent whole-text row's own seed_id for a seed-table one (there
        is no object_key there, and seed_id is already the stable,
        content-derived identity seeds.py gives it).

        INSERT OR REPLACE on that pair: re-chunking under new rules moves the
        row rather than forking it, exactly as record_document's own
        docstring reasons about `extract_version`. This module's caller must
        delete the OLD seed_ids_json rows (store.delete_seeds) before writing
        the replacement manifest row - this method only indexes the newest
        chunk set, it does not diff against what a previous row named.
        """
        packed = dict(row)
        packed["source_id"] = source_id
        packed["object_key"] = object_key
        self._write(
            _insert_sql("chunk_manifest", _CHUNK_MANIFEST_COLS, "INSERT OR REPLACE"),
            _pack(_fill(packed, chunked_at=utcnow()), _CHUNK_MANIFEST_COLS),
        )

    def chunk_manifest(self, source_id: str, object_key: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM chunk_manifest WHERE source_id = ? AND object_key = ?",
            (source_id, object_key),
        ).fetchone()
        return dict(row) if row is not None else None

    def chunk_manifest_index(self, source_id: str) -> dict[str, dict]:
        """Every indexed chunk-manifest row for one source, keyed by object_key.

        Read ONCE per chunking run, the same reason document_index is: the
        resume decision is taken once per document/seed in the source and a
        SELECT apiece would make restarting an interrupted run cost more than
        the work it skips. Carries seed_ids_json (unlike document_index's
        deliberate omission of meta_json) because every resumed run needs it
        the moment a document turns out stale, not only the runs that find one.
        """
        cols = ", ".join(_CHUNK_MANIFEST_RESUME_COLS)
        return {
            row["object_key"]: dict(row)
            for row in self._conn.execute(
                f"SELECT {cols} FROM chunk_manifest WHERE source_id = ?", (source_id,)
            ).fetchall()
        }

    def chunk_manifest_count(self, source_id: str | None = None) -> int:
        if source_id is None:
            return self._conn.execute("SELECT COUNT(*) FROM chunk_manifest").fetchone()[0]
        return self._conn.execute(
            "SELECT COUNT(*) FROM chunk_manifest WHERE source_id = ?", (source_id,)
        ).fetchone()[0]

    def delete_seeds(self, seed_ids: Iterable[str]) -> int:
        """Delete seed rows by id; returns how many rows were actually removed.

        The replace half of chunking's resume contract ("their chunks are
        replaced, never duplicated"): chunks.py calls this with a stale
        document's previous seed_ids (read off its chunk_manifest row)
        before writing the new ones, so a rule change that produces fewer or
        differently-bounded chunks does not leave the old boundaries behind
        as orphaned rows nothing points at any more.

        FK-enforced against task (foreign_keys=ON): deleting a seed some
        task row already references raises sqlite3.IntegrityError rather
        than silently orphaning that task. That is a real signal - chunking
        re-ran on a document a wave was already planned against - and not a
        case this method swallows.

        `dict.fromkeys` dedupes the input BEFORE IT REACHES SQL, and that
        sentence is the whole of what it buys: DELETE on an id an earlier
        statement in the same call already removed matches zero rows, so
        `total_changes` - and therefore the count this method returns - is
        identical either way. An earlier version of this docstring went on
        to call the dedup "a mutation this repository's own harness cannot
        kill". That was wrong, and wrong on the axis the sentence above
        names: what reaches the driver is observable through
        sqlite3.Connection.set_trace_callback, a public API, and removing
        the dedup takes a two-id call from 2 executed DELETEs to 4 with the
        return value unchanged. The test that asserts it counts statements
        rather than rows for exactly that reason.
        """
        ids = list(dict.fromkeys(seed_ids))
        if not ids:
            return 0
        before = self._conn.total_changes
        with self._write_txn() as conn:
            conn.executemany("DELETE FROM seed WHERE seed_id = ?", [(i,) for i in ids])
        return self._conn.total_changes - before

    # ------------------------------------------------------------------- tasks

    def create_tasks(self, rows: Iterable[dict]) -> int:
        """INSERT OR IGNORE by task_id; returns how many rows were NEW.

        Wave planning is re-runnable: replanning the same wave adds 0.
        """
        now = utcnow()
        payload = [
            _pack(
                _fill(dict(row), state="pending", attempts=0, created_at=now, updated_at=now),
                _TASK_COLS,
            )
            for row in rows
        ]
        if not payload:
            return 0
        before = self._conn.total_changes
        with self._write_txn() as conn:
            conn.executemany(_insert_sql("task", _TASK_COLS, "INSERT OR IGNORE"), payload)
        return self._conn.total_changes - before

    def claim_tasks(
        self,
        worker_id: str,
        n: int,
        *,
        stream: str | None = None,
        lease_s: int = DEFAULT_LEASE_S,
        state_from: str = "pending",
        state_to: str = "generating",
    ) -> list[dict]:
        """Lease up to `n` tasks to `worker_id`, recovering expired leases.

        Candidates are `state_from` tasks plus tasks stuck in `state_to` whose
        lease expired (a worker died mid-flight), oldest rowid first.

        The two state names are parameters because the pipeline has more than
        one queue over the same table: the generation worker leases
        pending -> generating, and the judge worker leases judging ->
        judging_active. The defaults are the generation queue, so every
        existing caller and test is unaffected.

        state_from and state_to MUST differ. If they were the same string the
        stale-lease clause would degenerate to "state = X OR state = X", i.e.
        every row another worker is holding right now becomes a candidate the
        moment it is claimed - the lease would stop fencing anything and two
        workers would run the same task with only one of them visible.
        """
        if state_from == state_to:
            raise ValueError(
                f"claim_tasks needs distinct states, got state_from == state_to == "
                f"{state_from!r}; a shared name disables lease fencing entirely"
            )
        if n <= 0:
            return []
        cutoff = _lease_cutoff(lease_s)
        # claimed_at IS NULL counts as stale: an in-flight row with no lease
        # stamp is unowned by construction and must not be stranded forever.
        clauses = [
            "(state = ? OR "
            "(state = ? AND (claimed_at IS NULL OR claimed_at < ?)))"
        ]
        params: list = [state_from, state_to, cutoff]
        if stream is not None:
            clauses.append("stream = ?")
            params.append(stream)
        select_sql = (
            f"SELECT task_id FROM task WHERE {' AND '.join(clauses)} ORDER BY rowid LIMIT ?"
        )
        params.append(n)

        now = utcnow()
        # BEGIN IMMEDIATE takes the write lock BEFORE the SELECT runs, so
        # "pick candidates" and "claim them" are one indivisible step. A second
        # worker starting a claim concurrently blocks at BEGIN (busy_timeout=
        # 5000ms) until this commits, and its own SELECT then sees those rows
        # as 'generating' with a fresh claimed_at - so it cannot re-claim them.
        # A deferred BEGIN would NOT be safe: both workers could read the same
        # candidate list as readers, and the loser would only discover the
        # conflict at UPDATE time (SQLITE_BUSY), after acting on those ids.
        with self._write_txn() as conn:
            ids = [r[0] for r in conn.execute(select_sql, params).fetchall()]
            if not ids:
                return []
            conn.executemany(
                "UPDATE task SET state = ?, claimed_by = ?, claimed_at = ?, "
                "attempts = attempts + 1, updated_at = ? WHERE task_id = ?",
                [(state_to, worker_id, now, now, task_id) for task_id in ids],
            )
            placeholders = ", ".join("?" * len(ids))
            rows = conn.execute(
                f"SELECT * FROM task WHERE task_id IN ({placeholders}) ORDER BY rowid", ids
            ).fetchall()
        return [dict(r) for r in rows]

    def set_task_state(
        self,
        task_id: str,
        state: str,
        disposition: str | None = None,
        *,
        expect_worker: str | None = None,
        reset_attempts: bool = False,
    ) -> bool:
        """Move a task to `state`, releasing its lease unless it stays 'generating'.

        Returns True if a row was actually updated.

        `reset_attempts` zeroes the claim counter in the SAME statement as the
        state move. Only the re-open path passes it: a task parked because
        nothing in the pool could serve it spent its whole budget discovering
        a fact about the FLEET, and handing it back to the queue still
        exhausted means the first ordinary failure after the operator fixes
        the cause is terminal. Default False, so every other caller is
        byte-for-byte unaffected.

        `expect_worker` is a lease fence. A worker that stalled past its lease
        (GC pause, hung socket) has already had its task legitimately reclaimed
        by someone else; when it finally comes back and reports its result, an
        unfenced UPDATE would clobber the live holder's row and two workers
        would be generating the same task with only one visible. Passing the
        worker id makes the UPDATE a no-op in that case and returns False, so
        the caller can detect the lost lease and drop its stale result.

        disposition=None leaves any existing disposition intact (a terminal
        state transition should not erase the diagnostic that caused it);
        pass a string to overwrite it. Keyword-only so it can never be
        confused positionally with `disposition`.
        """
        assignments = "state = ?, disposition = COALESCE(?, disposition), updated_at = ?"
        params: list = [state, disposition, utcnow()]
        if reset_attempts:
            assignments += ", attempts = 0"
        if state != "generating":
            assignments += ", claimed_by = NULL, claimed_at = NULL"
        where = "task_id = ?"
        params.append(task_id)
        if expect_worker is not None:
            # Evaluated against the pre-UPDATE row, so this still fences
            # correctly even though the same statement nulls claimed_by.
            where += " AND claimed_by = ?"
            params.append(expect_worker)
        cur = self._write(f"UPDATE task SET {assignments} WHERE {where}", tuple(params))
        return cur.rowcount > 0

    def get_task(self, task_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM task WHERE task_id = ?", (task_id,)).fetchone()
        return dict(row) if row is not None else None

    def accepted_count(self, stream: str | None = None) -> int:
        if stream is None:
            return self._conn.execute(
                "SELECT COUNT(*) FROM task WHERE state = 'accepted'"
            ).fetchone()[0]
        return self._conn.execute(
            "SELECT COUNT(*) FROM task WHERE state = 'accepted' AND stream = ?", (stream,)
        ).fetchone()[0]

    def task_counts(self) -> dict[str, int]:
        return {
            row["state"]: row["n"]
            for row in self._conn.execute(
                "SELECT state, COUNT(*) AS n FROM task GROUP BY state"
            ).fetchall()
        }

    # ------------------------------------------- generations, gates, judgements

    def record_generation(self, row: dict) -> int:
        """Index one generation; returns gen_id.

        Raises sqlite3.IntegrityError if (task_id, attempt) is already
        indexed - that duplicate is a bug in the worker, not a retry.
        """
        cur = self._write(
            _insert_sql("generation", _GEN_COLS),
            _pack(_fill(dict(row), created_at=utcnow()), _GEN_COLS),
        )
        return int(cur.lastrowid)

    def record_gates(self, gen_id: int, results: Iterable[tuple[str, bool, dict | None]]) -> None:
        payload = [
            (gen_id, gate, 1 if passed else 0, _dumps(detail)) for gate, passed, detail in results
        ]
        if not payload:
            return
        with self._write_txn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO gate_result (gen_id, gate, passed, detail_json) "
                "VALUES (?, ?, ?, ?)",
                payload,
            )

    def gates_for(self, gen_id: int) -> dict[str, bool]:
        return {
            row["gate"]: bool(row["passed"])
            for row in self._conn.execute(
                "SELECT gate, passed FROM gate_result WHERE gen_id = ?", (gen_id,)
            ).fetchall()
        }

    def record_judgement(
        self,
        gen_id: int,
        judge_slot: str,
        row: dict,
        *,
        expect_worker: str | None = None,
    ) -> bool:
        """Index one judge's scores for one generation.  Returns True if written.

        `expect_worker` is the SAME lease fence set_task_state takes, and it
        is here for the same reason.  This is an INSERT OR REPLACE on
        (gen_id, judge_slot), and a judge worker reads the recorded slots
        once and then awaits paid calls: a worker that stalls past its lease
        has already had its task re-claimed, and the live holder may have
        reused the stalled worker's slot A, bought its own slot B and
        accepted the row.  An unfenced late write from the stalled worker
        then replaces slot B with scores the accept was never made on - the
        row reads `accepted` carrying a judgement that contradicts it, and
        that pair is precisely what P5 calibration and gold labelling read.

        The check and the write are one BEGIN IMMEDIATE transaction, so the
        lease cannot move between them.  expect_worker=None writes
        unconditionally, which is what rebuild paths (reconcile_raw) need -
        they hold no lease and there is no live worker to lose to.
        """
        packed = dict(row)
        packed["gen_id"] = gen_id
        packed["judge_slot"] = judge_slot
        params = _pack(_fill(packed, created_at=utcnow()), _JUDGEMENT_COLS)
        sql = _insert_sql("judgement", _JUDGEMENT_COLS, "INSERT OR REPLACE")
        if expect_worker is None:
            self._write(sql, params)
            return True
        with self._write_txn() as conn:
            holder = conn.execute(
                "SELECT t.claimed_by FROM generation g JOIN task t ON t.task_id = g.task_id "
                "WHERE g.gen_id = ?",
                (gen_id,),
            ).fetchone()
            if holder is None or holder[0] != expect_worker:
                return False
            conn.execute(sql, params)
        return True

    def judgements_for(self, gen_id: int) -> list[dict]:
        return [
            dict(row)
            for row in self._conn.execute(
                "SELECT * FROM judgement WHERE gen_id = ? ORDER BY judge_slot", (gen_id,)
            ).fetchall()
        ]

    # ------------------------------------------------------- gold + calibration

    def judged_generations(self, streams: Iterable[str] | None = None) -> list[dict]:
        """Every generation that at least one judge has scored, oldest first.

        calibrate.py's population: a generation nobody judged has nothing to
        calibrate a judge against. Joined to its task row for the stream and
        prompt columns the export stratifies on, and ordered by gen_id for the
        same reason latest_generations is - the pilot export is a stratified
        prefix of this list, so its order decides WHICH rows the operator is
        asked to label, not merely the order they are printed in.
        """
        clause = ""
        params: tuple = ()
        if streams is not None:
            names = tuple(streams)
            if not names:
                return []
            clause = f"AND t.stream IN ({', '.join('?' * len(names))})"
            params = names
        return [
            dict(row)
            for row in self._conn.execute(
                "SELECT g.*, t.stream, t.seed_id, t.task_type, t.prompt_id, t.arm, "
                "       t.state AS task_state "
                "FROM generation g "
                "JOIN task t ON t.task_id = g.task_id "
                "WHERE EXISTS (SELECT 1 FROM judgement j WHERE j.gen_id = g.gen_id) "
                f"{clause} ORDER BY g.gen_id",
                params,
            ).fetchall()
        ]

    def judgements_by_gen(self, gen_ids: Iterable[int]) -> dict[int, list[dict]]:
        """gen_id -> its judgement rows, for many generations at once.

        judgements_for answers this one row at a time, which is right for a
        judge worker deciding one task. The calibration fit reads every
        judgement of every labelled generation at once, and a row-at-a-time
        read there is one query per gold label for a number that is a single
        pass over the table.
        """
        ids = list(dict.fromkeys(int(gen_id) for gen_id in gen_ids))
        out: dict[int, list[dict]] = {gen_id: [] for gen_id in ids}
        for start in range(0, len(ids), 500):  # SQLITE_MAX_VARIABLE_NUMBER
            chunk = ids[start : start + 500]
            rows = self._conn.execute(
                f"SELECT * FROM judgement WHERE gen_id IN ({', '.join('?' * len(chunk))}) "
                f"ORDER BY gen_id, judge_slot",
                chunk,
            ).fetchall()
            for row in rows:
                out[row["gen_id"]].append(dict(row))
        return out

    def upsert_gold_labels(self, rows: Iterable[dict]) -> int:
        """INSERT OR REPLACE human gold labels; returns rows written.

        REPLACE because the operator labels in sittings and may correct a
        verdict in a later one - the second file is the authority, and a
        pipeline that refused it would leave the mistake in the fit forever.
        """
        payload = [_pack(_fill(dict(row), labeled_at=utcnow()), _GOLD_COLS) for row in rows]
        if not payload:
            return 0
        before = self._conn.total_changes
        with self._write_txn() as conn:
            conn.executemany(_insert_sql("gold_label", _GOLD_COLS, "INSERT OR REPLACE"), payload)
        return self._conn.total_changes - before

    def gold_labels(self) -> list[dict]:
        return [
            dict(row)
            for row in self._conn.execute("SELECT * FROM gold_label ORDER BY gen_id").fetchall()
        ]

    def gold_label_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM gold_label").fetchone()[0])

    def record_judge_thresholds(self, rows: Iterable[dict]) -> int:
        """Replace the ACTIVE calibration with this one, in one transaction.

        Deactivating the previous rows and inserting the new ones has to be
        atomic: judge.thresholds_active counts active rows to decide whether
        its decisions are provisional, and a crash between the two writes
        would leave the fleet either double-calibrated (two rules for one
        model) or reading as provisional while a fitted calibration exists.
        Superseded rows are kept with active = 0 rather than deleted - the
        report of a later run is only interpretable against the fit it
        replaced.
        """
        payload = [_pack(_fill(dict(row), fitted_at=utcnow(), active=1), _THRESHOLD_COLS) for row in rows]
        if not payload:
            return 0
        before = self._conn.total_changes
        with self._write_txn() as conn:
            conn.execute("UPDATE judge_threshold SET active = 0 WHERE active = 1")
            conn.executemany(
                _insert_sql("judge_threshold", _THRESHOLD_COLS, "INSERT OR REPLACE"), payload
            )
        return self._conn.total_changes - before

    def judge_thresholds(self, *, active_only: bool = True) -> list[dict]:
        clause = "WHERE active = 1" if active_only else ""
        return [
            dict(row)
            for row in self._conn.execute(
                f"SELECT * FROM judge_threshold {clause} ORDER BY calib_id, model, judge_slot"
            ).fetchall()
        ]

    def latest_generation(self, task_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM generation WHERE task_id = ? ORDER BY attempt DESC LIMIT 1", (task_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def latest_generations(self, where_state: str | None = None) -> list[dict]:
        """The newest generation of every task, joined to its task row.

        Newest only: a task's disposition is about the answer it currently
        stands on, so a pass that reads superseded attempts would judge a task
        by a draft that was already replaced.

        Lives here rather than in the caller because two passes now read it -
        verify.py re-gates these rows and the assembly pass (decontaminate/
        dedupe) reads them as candidate dataset rows - and a second copy of
        this join is a second answer to "which generation is the row".

        `where_state=None` means EVERY state; `where_state=""` means the state
        whose name is the empty string, i.e. nothing. Truthiness here made
        `--state ''` silently disable the filter and ship rejected
        generations, which is not a reading anyone asked for.

        ORDER BY g.gen_id ASCENDING IS DATASET-DEFINING, not cosmetic. It was
        inert while verify.py was the only caller (rerun_gates re-gates each
        row independently), but decontaminate.store_items now feeds these rows
        into dedupe, which keeps the FIRST row of a duplicate cluster and the
        first three rows of a case - so this ORDER BY decides WHICH rows ship,
        not merely the order they ship in. Ascending gen_id is chosen because
        gen_id is a monotonic rowid that never changes, which makes the
        survivor set stable in two senses: identical between two runs over the
        same store, and unchanged by LATER waves - a generation recorded
        tomorrow sorts after everything that already shipped and cannot
        displace it. Under DESC the pipeline yields the same COUNT and a
        different SET.
        """
        clause = "WHERE t.state = ?" if where_state is not None else ""
        params: tuple = (where_state,) if where_state is not None else ()
        return [
            dict(row)
            for row in self._conn.execute(
                "SELECT g.*, t.stream, t.seed_id, t.task_type, t.prompt_id, t.prompt_sha, "
                "       t.sample_ix, t.arm, t.state AS task_state "
                "FROM generation g "
                "JOIN task t ON t.task_id = g.task_id "
                "JOIN (SELECT task_id, MAX(attempt) AS a FROM generation GROUP BY task_id) m "
                "  ON m.task_id = g.task_id AND m.a = g.attempt "
                f"{clause} ORDER BY g.gen_id",
                params,
            ).fetchall()
        ]

    # ------------------------------------------------------------------ budget

    def reserve_budget(
        self,
        provider: str,
        model: str,
        est_tokens: int,
        *,
        limits: Mapping,
        day: str | None = None,
    ) -> bool:
        """Advisory pre-flight check - True if this call still fits the daily caps.

        Deliberately does NOT increment anything: reservations would leak on
        every crash. record_usage() after the call is the source of truth, so
        the worst case is a small overshoot of in-flight requests, never a
        permanently poisoned ledger.
        """
        used = self.usage_today(provider, model, day=day)
        tokens = used["prompt_tokens"] + used["completion_tokens"]
        if tokens + est_tokens > _cap(limits, "tpd"):
            return False
        return used["requests"] + 1 <= _cap(limits, "rpd")

    def record_usage(
        self,
        provider: str,
        model: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        is_429: bool = False,
        day: str | None = None,
    ) -> None:
        """Accumulate one call into the daily ledger.

        Single-statement UPSERT: the read-modify-write happens inside SQLite
        under the write lock, so concurrent workers can never lose an
        increment the way a SELECT-then-UPDATE pair would.
        """
        self._write(
            "INSERT INTO budget_ledger "
            "(day, provider, model, requests, prompt_tokens, completion_tokens, errors_429) "
            "VALUES (?, ?, ?, 1, ?, ?, ?) "
            "ON CONFLICT(day, provider, model) DO UPDATE SET "
            "  requests = requests + 1, "
            "  prompt_tokens = prompt_tokens + excluded.prompt_tokens, "
            "  completion_tokens = completion_tokens + excluded.completion_tokens, "
            "  errors_429 = errors_429 + excluded.errors_429",
            (
                utcday(day),
                provider,
                model,
                int(prompt_tokens),
                int(completion_tokens),
                1 if is_429 else 0,
            ),
        )

    def usage_today(self, provider: str, model: str, day: str | None = None) -> dict:
        row = self._conn.execute(
            "SELECT requests, prompt_tokens, completion_tokens, errors_429 FROM budget_ledger "
            "WHERE day = ? AND provider = ? AND model = ?",
            (utcday(day), provider, model),
        ).fetchone()
        if row is None:
            return {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "errors_429": 0}
        return dict(row)

    # ------------------------------------------------------- events & reconcile

    def log_event(self, kind: str, detail: dict) -> None:
        self._write(
            "INSERT INTO run_event (at, kind, detail_json) VALUES (?, ?, ?)",
            (utcnow(), kind, _dumps(detail)),
        )

    def _flush_events(self, buffered: list[tuple[str, str, str | None]]) -> None:
        """Write diagnostics buffered during a sweep, in their own transaction."""
        if not buffered:
            return
        with self._write_txn() as conn:
            conn.executemany(
                "INSERT INTO run_event (at, kind, detail_json) VALUES (?, ?, ?)", buffered
            )

    def events(self, kind: str | None = None) -> list[dict]:
        if kind is None:
            rows = self._conn.execute("SELECT * FROM run_event ORDER BY event_id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM run_event WHERE kind = ? ORDER BY event_id", (kind,)
            ).fetchall()
        return [dict(r) for r in rows]

    def reconcile_raw(self, raw_paths: Iterable[Path]) -> int:
        """Rebuild DB rows from raw logs; returns how many rows were recovered.

        Crash-recovery path for the durability rule: any envelope that reached
        a raw log but never reached the DB is re-indexed here, pointed at the
        exact byte offset it occupies. Already-indexed records are left alone
        (the DB row may be richer than the envelope), so this is idempotent -
        a second run recovers 0. Corrupt or truncated lines are logged as
        run_events and skipped, never raised: a half-written last line is the
        expected shape of a crash, not an error.

        Transaction shape: ONE TRANSACTION PER FILE, not one for the whole
        sweep. Two reasons. The write lock is released between files, so a
        multi-GB recovery does not block every worker for its whole duration;
        and because the pass is idempotent, a failure part-way through keeps
        the files already recovered instead of discarding them - re-running
        finishes the job. The trade is that a failed sweep leaves recovery
        partially applied, which is exactly what idempotency makes safe.

        Diagnostics are buffered in memory and written AFTER the sweep, in
        their own transaction, so that a rollback cannot destroy the records
        explaining why it rolled back.
        """
        recovered = 0
        deferred: list[tuple[dict, Path, int]] = []
        diag: list[tuple[str, str, str | None]] = []
        try:
            for raw_path in raw_paths:
                path = Path(raw_path)
                if not path.exists():
                    _diag(diag, "reconcile_missing_file", {"path": str(path)})
                    continue
                with self._write_txn():
                    for offset, rec in self._scan_raw(path, diag):
                        kind = rec.get("kind")
                        if kind == "generation":
                            recovered += self._recover_generation(rec, path, offset, diag)
                        elif kind == "judgement":
                            gen_id = self._resolve_gen_id(rec, diag)
                            if gen_id is None:
                                # Its generation may live later in this file or
                                # in a file not scanned yet - retry after the
                                # sweep, so the result cannot depend on the
                                # order the caller passed the paths in.
                                deferred.append((rec, path, offset))
                            else:
                                recovered += self._recover_judgement(
                                    gen_id, rec, path, offset, diag
                                )
                        else:
                            _diag(
                                diag,
                                "reconcile_unknown_kind",
                                {"path": str(path), "offset": offset, "kind": kind},
                            )
            if deferred:
                with self._write_txn():
                    for rec, path, offset in deferred:
                        gen_id = self._resolve_gen_id(rec, diag)
                        if gen_id is None:
                            _diag(
                                diag,
                                "reconcile_orphan_judgement",
                                {
                                    "path": str(path),
                                    "offset": offset,
                                    "task_id": rec.get("task_id"),
                                    "attempt": rec.get("attempt"),
                                },
                            )
                            continue
                        recovered += self._recover_judgement(gen_id, rec, path, offset, diag)
        finally:
            # Runs on the failure path too, so the diagnostics outlive the
            # sweep that produced them. If this flush itself fails mid-crash
            # the original error is preserved as __context__.
            self._flush_events(diag)
        return recovered

    def _scan_raw(self, path: Path, diag: list) -> Iterator[tuple[int, dict]]:
        """Yield (byte offset of line start, record) for every parsable line.

        Read in BINARY and count bytes by hand: len(line) includes the line
        terminator, so the offsets stay exact even for a log that picked up
        CRLF endings. A text-mode read on Windows would collapse "\\r\\n" to
        "\\n" and skew every subsequent offset by one byte per preceding line,
        silently corrupting every raw_offset this function writes. The offsets
        produced here are exactly what jsonl.append_ndjson returned and what
        jsonl.read_at expects.
        """
        offset = 0
        with path.open("rb") as f:
            for line in f:
                start = offset
                offset += len(line)
                payload = line.strip()
                if not payload:
                    continue
                try:
                    rec = json.loads(payload)
                except ValueError as exc:  # JSONDecodeError / UnicodeDecodeError
                    _diag(
                        diag,
                        "reconcile_bad_line",
                        {"path": str(path), "offset": start, "error": str(exc)},
                    )
                    continue
                if not isinstance(rec, dict):
                    _diag(
                        diag,
                        "reconcile_bad_line",
                        {"path": str(path), "offset": start, "error": "record is not an object"},
                    )
                    continue
                yield start, rec

    def _resolve_gen_id(self, rec: Mapping, diag: list) -> int | None:
        """Map a raw envelope onto a gen_id - by NATURAL key first.

        gen_id is an AUTOINCREMENT surrogate, and surrogates are not stable
        across a rebuild: recover the same logs into a fresh database and the
        ids get handed out in a different order. An envelope's remembered
        gen_id can therefore address a DIFFERENT generation than the one it was
        written for, silently binding a judgement to the wrong answer - which
        would corrupt training labels rather than crash. (task_id, attempt) is
        the natural key and carries the UNIQUE constraint, so it always wins;
        the envelope's gen_id is only ever a fallback for envelopes that lack
        the natural key, and even then it must not contradict what it does
        carry.
        """
        task_id, attempt = rec.get("task_id"), rec.get("attempt")
        env_gen_id = rec.get("gen_id")

        if task_id is not None and attempt is not None:
            row = self._conn.execute(
                "SELECT gen_id FROM generation WHERE task_id = ? AND attempt = ?",
                (task_id, attempt),
            ).fetchone()
            if row is None:
                # The generation genuinely is not indexed yet. Falling back to
                # the envelope's gen_id here is what caused the misbind: it
                # would attach this judgement to whichever row now owns that id.
                return None
            gen_id = int(row[0])
            if env_gen_id is not None and env_gen_id != gen_id:
                _diag(
                    diag,
                    "reconcile_gen_id_remapped",
                    {
                        "task_id": task_id,
                        "attempt": attempt,
                        "envelope_gen_id": env_gen_id,
                        "resolved_gen_id": gen_id,
                    },
                )
            return gen_id

        if env_gen_id is None:
            return None
        row = self._conn.execute(
            "SELECT task_id, attempt FROM generation WHERE gen_id = ?", (env_gen_id,)
        ).fetchone()
        if row is None:
            return None
        # A partial natural key still has to agree, or the surrogate is stale.
        if (task_id is not None and row["task_id"] != task_id) or (
            attempt is not None and row["attempt"] != attempt
        ):
            _diag(
                diag,
                "reconcile_gen_id_mismatch",
                {
                    "envelope_gen_id": env_gen_id,
                    "envelope_task_id": task_id,
                    "envelope_attempt": attempt,
                    "row_task_id": row["task_id"],
                    "row_attempt": row["attempt"],
                },
            )
            return None
        return int(env_gen_id)

    def _recover_generation(self, rec: dict, path: Path, offset: int, diag: list) -> int:
        task_id, attempt = rec.get("task_id"), rec.get("attempt")
        if task_id is None or attempt is None:
            _diag(
                diag,
                "reconcile_bad_record",
                {"path": str(path), "offset": offset, "error": "generation lacks task_id/attempt"},
            )
            return 0
        if self._resolve_gen_id({"task_id": task_id, "attempt": attempt}, diag) is not None:
            return 0
        # raw_path/raw_offset always come from where the record ACTUALLY sits,
        # never from the envelope's own (possibly stale) copy of them.
        row = _fill(dict(rec), created_at=utcnow())
        row["raw_path"] = str(path)
        row["raw_offset"] = offset
        try:
            self._conn.execute(_insert_sql("generation", _GEN_COLS), _pack(row, _GEN_COLS))
        except _UNUSABLE_RECORD as exc:
            # Statement-level rollback only: the surrounding transaction stays
            # usable, so one bad envelope cannot poison the whole recovery.
            _diag(
                diag,
                "reconcile_rejected",
                {"path": str(path), "offset": offset, "kind": "generation", "error": str(exc)},
            )
            return 0
        return 1

    def _recover_judgement(
        self, gen_id: int, rec: dict, path: Path, offset: int, diag: list
    ) -> int:
        judge_slot = rec.get("judge_slot")
        if judge_slot is None:
            _diag(
                diag,
                "reconcile_bad_record",
                {"path": str(path), "offset": offset, "error": "judgement lacks judge_slot"},
            )
            return 0
        exists = self._conn.execute(
            "SELECT 1 FROM judgement WHERE gen_id = ? AND judge_slot = ?", (gen_id, judge_slot)
        ).fetchone()
        if exists is not None:
            return 0
        row = _fill(dict(rec), created_at=utcnow())
        row["gen_id"] = gen_id
        row["judge_slot"] = judge_slot
        row["raw_path"] = str(path)
        row["raw_offset"] = offset
        try:
            self._conn.execute(
                _insert_sql("judgement", _JUDGEMENT_COLS), _pack(row, _JUDGEMENT_COLS)
            )
        except _UNUSABLE_RECORD as exc:
            _diag(
                diag,
                "reconcile_rejected",
                {"path": str(path), "offset": offset, "kind": "judgement", "error": str(exc)},
            )
            return 0
        return 1
