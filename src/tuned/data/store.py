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
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

_TS_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"

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
"""

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


def _cap(limits: Mapping, key: str) -> float:
    """Read a budget cap; missing or None means unlimited."""
    value = limits.get(key)
    return float("inf") if value is None else value


class Store:
    """Typed, transaction-safe wrapper around the pipeline's SQLite file."""

    def __init__(self, conn: sqlite3.Connection, path: Path):
        self._conn = conn
        self.path = path

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
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @contextmanager
    def _write_txn(self) -> Iterator[sqlite3.Connection]:
        """BEGIN IMMEDIATE ... COMMIT, rolling back on any exception."""
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

    # ------------------------------------------------------------ sources/seeds

    def upsert_source(
        self, source_id: str, license: str, url: str | None = None, version: str | None = None
    ) -> None:
        self._conn.execute(
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
        self, worker_id: str, n: int, *, stream: str | None = None, lease_s: int = 900
    ) -> list[dict]:
        """Lease up to `n` tasks to `worker_id`, recovering expired leases.

        Candidates are pending tasks plus tasks stuck in 'generating' whose
        lease expired (a worker died mid-flight), oldest rowid first.
        """
        if n <= 0:
            return []
        cutoff = _lease_cutoff(lease_s)
        # claimed_at IS NULL counts as stale: a 'generating' row with no lease
        # stamp is unowned by construction and must not be stranded forever.
        clauses = [
            "(state = 'pending' OR "
            "(state = 'generating' AND (claimed_at IS NULL OR claimed_at < ?)))"
        ]
        params: list = [cutoff]
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
                "UPDATE task SET state = 'generating', claimed_by = ?, claimed_at = ?, "
                "attempts = attempts + 1, updated_at = ? WHERE task_id = ?",
                [(worker_id, now, now, task_id) for task_id in ids],
            )
            placeholders = ", ".join("?" * len(ids))
            rows = conn.execute(
                f"SELECT * FROM task WHERE task_id IN ({placeholders}) ORDER BY rowid", ids
            ).fetchall()
        return [dict(r) for r in rows]

    def set_task_state(self, task_id: str, state: str, disposition: str | None = None) -> None:
        """Move a task to `state`, releasing its lease unless it stays 'generating'.

        disposition=None leaves any existing disposition intact (a terminal
        state transition should not erase the diagnostic that caused it);
        pass a string to overwrite it.
        """
        if state == "generating":
            self._conn.execute(
                "UPDATE task SET state = ?, disposition = COALESCE(?, disposition), "
                "updated_at = ? WHERE task_id = ?",
                (state, disposition, utcnow(), task_id),
            )
        else:
            self._conn.execute(
                "UPDATE task SET state = ?, disposition = COALESCE(?, disposition), "
                "claimed_by = NULL, claimed_at = NULL, updated_at = ? WHERE task_id = ?",
                (state, disposition, utcnow(), task_id),
            )

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
        cur = self._conn.execute(
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

    def record_judgement(self, gen_id: int, judge_slot: str, row: dict) -> None:
        packed = dict(row)
        packed["gen_id"] = gen_id
        packed["judge_slot"] = judge_slot
        self._conn.execute(
            _insert_sql("judgement", _JUDGEMENT_COLS, "INSERT OR REPLACE"),
            _pack(_fill(packed, created_at=utcnow()), _JUDGEMENT_COLS),
        )

    def judgements_for(self, gen_id: int) -> list[dict]:
        return [
            dict(row)
            for row in self._conn.execute(
                "SELECT * FROM judgement WHERE gen_id = ? ORDER BY judge_slot", (gen_id,)
            ).fetchall()
        ]

    def latest_generation(self, task_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM generation WHERE task_id = ? ORDER BY attempt DESC LIMIT 1", (task_id,)
        ).fetchone()
        return dict(row) if row is not None else None

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
        self._conn.execute(
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
        # Single statement: safe to call inside an open _write_txn (it simply
        # joins that transaction) as well as standalone.
        self._conn.execute(
            "INSERT INTO run_event (at, kind, detail_json) VALUES (?, ?, ?)",
            (utcnow(), kind, _dumps(detail)),
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
        """
        recovered = 0
        deferred: list[tuple[dict, Path, int]] = []
        with self._write_txn():
            for raw_path in raw_paths:
                path = Path(raw_path)
                if not path.exists():
                    self.log_event("reconcile_missing_file", {"path": str(path)})
                    continue
                for offset, rec in self._scan_raw(path):
                    kind = rec.get("kind")
                    if kind == "generation":
                        recovered += self._recover_generation(rec, path, offset)
                    elif kind == "judgement":
                        gen_id = self._resolve_gen_id(rec)
                        if gen_id is None:
                            # Its generation may live later in this file or in
                            # a file not scanned yet - retry after the sweep.
                            deferred.append((rec, path, offset))
                        else:
                            recovered += self._recover_judgement(gen_id, rec, path, offset)
                    else:
                        self.log_event(
                            "reconcile_unknown_kind",
                            {"path": str(path), "offset": offset, "kind": kind},
                        )
            for rec, path, offset in deferred:
                gen_id = self._resolve_gen_id(rec)
                if gen_id is None:
                    self.log_event(
                        "reconcile_orphan_judgement",
                        {
                            "path": str(path),
                            "offset": offset,
                            "task_id": rec.get("task_id"),
                            "attempt": rec.get("attempt"),
                        },
                    )
                    continue
                recovered += self._recover_judgement(gen_id, rec, path, offset)
        return recovered

    def _scan_raw(self, path: Path) -> Iterator[tuple[int, dict]]:
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
                    self.log_event(
                        "reconcile_bad_line",
                        {"path": str(path), "offset": start, "error": str(exc)},
                    )
                    continue
                if not isinstance(rec, dict):
                    self.log_event(
                        "reconcile_bad_line",
                        {"path": str(path), "offset": start, "error": "record is not an object"},
                    )
                    continue
                yield start, rec

    def _resolve_gen_id(self, rec: Mapping) -> int | None:
        if rec.get("gen_id") is not None:
            row = self._conn.execute(
                "SELECT gen_id FROM generation WHERE gen_id = ?", (rec["gen_id"],)
            ).fetchone()
            if row is not None:
                return int(row[0])
        task_id, attempt = rec.get("task_id"), rec.get("attempt")
        if task_id is None or attempt is None:
            return None
        row = self._conn.execute(
            "SELECT gen_id FROM generation WHERE task_id = ? AND attempt = ?", (task_id, attempt)
        ).fetchone()
        return int(row[0]) if row is not None else None

    def _recover_generation(self, rec: dict, path: Path, offset: int) -> int:
        task_id, attempt = rec.get("task_id"), rec.get("attempt")
        if task_id is None or attempt is None:
            self.log_event(
                "reconcile_bad_record",
                {"path": str(path), "offset": offset, "error": "generation lacks task_id/attempt"},
            )
            return 0
        if self._resolve_gen_id({"task_id": task_id, "attempt": attempt}) is not None:
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
            self.log_event(
                "reconcile_rejected",
                {"path": str(path), "offset": offset, "kind": "generation", "error": str(exc)},
            )
            return 0
        return 1

    def _recover_judgement(self, gen_id: int, rec: dict, path: Path, offset: int) -> int:
        judge_slot = rec.get("judge_slot")
        if judge_slot is None:
            self.log_event(
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
            self.log_event(
                "reconcile_rejected",
                {"path": str(path), "offset": offset, "kind": "judgement", "error": str(exc)},
            )
            return 0
        return 1
