"""Read-only ship-gate readout for an audit-mode build.

Under `judge_mode: audit` most rows ship on gates alone; the dual-judged
sample is the only quality evidence for the batch. This prints that
evidence: task states, how many rows exited as audit-accepts, and the
sampled rows' dual-judge accept rate - the number the operator reads before
publishing the dataset (a low sample accept rate impeaches every
audit-accepted row, not just the sampled ones).

Works against any store copy - the live one, or a bundle snapshot pulled
from the baton repo. Opens it read-only; safe while a fleet is writing.

Usage: python data/scripts/audit_readout.py [path/to/law_v1.sqlite3]
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


def summarize(conn: sqlite3.Connection) -> dict:
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
    # dispositions carry judge:accept / judge:reject*. Rows decided before
    # audit mode shipped count too - they are the same dual treatment.
    sampled = dict(
        conn.execute(
            "SELECT disposition, COUNT(*) FROM task "
            "WHERE disposition LIKE 'judge:%' GROUP BY disposition"
        )
    )
    accepted = sampled.get("judge:accept", 0)
    decided = sum(sampled.values())
    return {
        "states": states,
        "audit_accepts": audit_accepts,
        "audit_unjudged": audit_unjudged,
        "sampled_decided": decided,
        "sampled_accepted": accepted,
        "sample_accept_rate": (accepted / decided) if decided else None,
        "sampled_dispositions": sampled,
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
    rate = s["sample_accept_rate"]
    lines.append(
        f"dual-judged sample: {s['sampled_decided']} decided, "
        f"{s['sampled_accepted']} accepted"
        + (f" -> accept rate {100 * rate:.1f}%" if rate is not None
           else " (nothing sampled yet)")
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    db = Path(args[0]) if args else Path("data/build/state/law_v1.sqlite3")
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        s = summarize(conn)
    finally:
        conn.close()
    for line in format_summary(s):
        print(line)
    for disposition, n in sorted(s["sampled_dispositions"].items()):
        print(f"  {disposition}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
