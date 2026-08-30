"""Read-only ship-gate readout for an audit-mode build.

Under `judge_mode: audit` most rows ship on gates alone; the dual-judged
sample is the only quality evidence for the batch. This prints that
evidence: task states, how many rows exited as audit-accepts, and the
sampled rows' dual-judge accept rate - the number the operator reads before
publishing the dataset (a low sample accept rate impeaches every
audit-accepted row, not just the sampled ones) - pooled AND broken out by
stream, over the window of judgements `since` names.

Works against any store copy - the live one, or a bundle snapshot pulled
from the baton repo. Opens it read-only; safe while a fleet is writing.

Usage: python data/scripts/audit_readout.py [path/to/law_v1.sqlite3] [--since TIMESTAMP]
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

AUDIT_DISPOSITION = "audit:gate-accept"
# Rows the hash SELECTED for judging that no judge could be reached for. They
# ship on their gates like an unsampled row, but the count is a fleet-health
# signal rather than a sampling choice: a rising number means the judge fleet
# is underwater and the sample is thinning.
AUDIT_UNJUDGED_DISPOSITION = "audit:gate-accept:unjudged"


def summarize(conn: sqlite3.Connection, *, since: str | None = None) -> dict:
    """Everything the ship-gate readout needs, pooled and per-stream.

    `since` restricts the dual-judged sample to judgements recorded at or
    after this UTC timestamp (store.utcnow's format: sortable ISO-ish text
    compares correctly as a string). `since=None` pools every judged row
    ever recorded, including verdicts from before audit mode shipped and
    from providers since retired - today's behaviour, unchanged.

    judgement.created_at is the ONLY column that dates a judgement: it is
    stamped by Store.record_judgement at the moment a judge slot's scores are
    written (store.py, record_judgement -> utcnow()). task.updated_at was
    considered and rejected - it moves on every task-state transition (claim,
    generation, judging_active, reconcile...), not only the one where a judge
    decided the row, so filtering on it would keep a task whose disposition
    dates from before the window but whose row was touched again afterwards.
    A task can carry two judgement rows (slot A, a tiebreak's slot B); its
    disposition is only decided once the required slots are in, so the
    LATEST of a task's judgements - MAX(created_at), not the earliest slot -
    is when that decision was actually made.
    """
    states = dict(
        conn.execute("SELECT state, COUNT(*) FROM task GROUP BY state ORDER BY state")
    )
    audit_accepts = conn.execute(
        "SELECT COUNT(*) FROM task WHERE disposition = ?", (AUDIT_DISPOSITION,)
    ).fetchone()[0]
    audit_unjudged = conn.execute(
        "SELECT COUNT(*) FROM task WHERE disposition = ?", (AUDIT_UNJUDGED_DISPOSITION,)
    ).fetchone()[0]

    # The sampled rows are the ones the judges actually decided: their
    # dispositions carry judge:accept / judge:reject*. Grouped by stream as
    # well as disposition in one pass, so the pooled figures below and the
    # per-stream breakdown can never disagree about which rows they counted.
    where = "t.disposition LIKE 'judge:%'"
    params: tuple = ()
    if since is not None:
        where += (
            " AND (SELECT MAX(j.created_at) FROM judgement j "
            "JOIN generation g ON g.gen_id = j.gen_id "
            "WHERE g.task_id = t.task_id) >= ?"
        )
        params = (since,)
    rows = conn.execute(
        f"SELECT t.stream, t.disposition, COUNT(*) FROM task t WHERE {where} "
        f"GROUP BY t.stream, t.disposition",
        params,
    ).fetchall()

    sampled: dict[str, int] = {}
    by_stream_raw: dict[str, dict[str, int]] = {}
    for stream, disposition, n in rows:
        sampled[disposition] = sampled.get(disposition, 0) + n
        by_stream_raw.setdefault(stream, {})
        by_stream_raw[stream][disposition] = by_stream_raw[stream].get(disposition, 0) + n

    accepted = sampled.get("judge:accept", 0)
    decided = sum(sampled.values())

    # Per-stratum SAMPLING fractions were considered and rejected here on
    # purpose - do not re-propose varying judge --audit-sample by stream. An
    # unsampled row is accepted unconditionally while a sampled row can be
    # rejected, so a fraction that differs by stream would make the corpus
    # MIX itself a function of sampling policy rather than of what the rows
    # actually are. What follows is a REPORTING split only: the same
    # hash-sampled population `sampled` above pools, broken out per stream so
    # an operator can see WHICH stream the pooled rate is hiding a collapse
    # in, rather than one blended number that a healthy stream can mask a
    # failing one inside of.
    by_stream: dict[str, dict] = {}
    for stream, dispositions in sorted(by_stream_raw.items()):
        stream_decided = sum(dispositions.values())
        stream_accepted = dispositions.get("judge:accept", 0)
        by_stream[stream] = {
            "decided": stream_decided,
            "accepted": stream_accepted,
            "accept_rate": (stream_accepted / stream_decided) if stream_decided else None,
        }

    return {
        "since": since,
        "states": states,
        "audit_accepts": audit_accepts,
        "audit_unjudged": audit_unjudged,
        "sampled_decided": decided,
        "sampled_accepted": accepted,
        "sample_accept_rate": (accepted / decided) if decided else None,
        "sampled_dispositions": sampled,
        "by_stream": by_stream,
    }


def format_summary(s: dict) -> list[str]:
    """The readout as lines. Shared with the Actions run report so the ship
    gate the operator reads and the summary CI prints are one formatter."""
    lines = [
        "task states: " + ", ".join(f"{k}={v}" for k, v in s["states"].items()),
        f"audit-accepted (gates only, not sampled): {s['audit_accepts']}",
    ]
    if s["audit_unjudged"]:
        lines.append(
            f"audit-accepted (sampled but NO judge reachable): {s['audit_unjudged']}"
            "  <- judge fleet is thinning the sample"
        )
    since = s.get("since")
    lines.append(f"dual-judged sample window: since={since}" if since else
                 "dual-judged sample window: all recorded judgements (no --since given)")
    rate = s["sample_accept_rate"]
    lines.append(
        f"dual-judged sample: {s['sampled_decided']} decided, "
        f"{s['sampled_accepted']} accepted"
        + (f" -> accept rate {100 * rate:.1f}%" if rate is not None
           else " (nothing sampled yet)")
    )
    by_stream = s.get("by_stream") or {}
    if by_stream:
        lines.append("  by stream:")
        for stream, d in sorted(by_stream.items()):
            stream_rate = d["accept_rate"]
            lines.append(
                f"    {stream}: {d['decided']} decided, {d['accepted']} accepted"
                + (f" -> accept rate {100 * stream_rate:.1f}%" if stream_rate is not None
                   else " (nothing sampled yet)")
            )
    return lines


def main(argv: list[str] | None = None) -> int:
    import argparse

    # argparse rather than hand-parsing: `--since` with no value behind it is
    # a usage error with a message, not an IndexError traceback, and this
    # script is the one an operator runs by hand under time pressure.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", nargs="?", default="data/build/state/law_v1.sqlite3")
    parser.add_argument(
        "--since", default=None, metavar="TIMESTAMP",
        help="restrict the dual-judged sample to judgements at or after this "
             "store.utcnow timestamp; the shipped window is "
             "routing.judge_mode_since in the build config",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    db = Path(args.db)
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        s = summarize(conn, since=args.since)
    finally:
        conn.close()
    for line in format_summary(s):
        print(line)
    for disposition, n in sorted(s["sampled_dispositions"].items()):
        print(f"  {disposition}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
