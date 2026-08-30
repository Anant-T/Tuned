"""Re-index raw NDJSON envelopes into the store: the merge half of remote work.

The raw logs are the system of record (store.py's durability rule); the
SQLite store is a derived index. Work that ran elsewhere - another machine, a
killed CI job - comes home as raw files, and this CLI folds them in via
store.reconcile_raw: idempotent, one transaction per file, judgements re-keyed
by natural key (task_id, attempt) rather than the non-portable gen_id
surrogate. It restores generation and judgement rows only; gate_result rows
and final task states are recomputed offline, for free, by
`python -m tuned.data.verify` - the step that follows this one.

Usage: python -m tuned.data.reconcile --config data/configs/data_law_v1.yaml \
           [--raw PATH ...]

Without --raw it sweeps every raw/{gen,judge}/*/*.ndjson under the config's
workdir. A missing path is a logged diagnostic (reconcile_missing_file), not
an error - the sweep's job is to fold in whatever is there.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence
from tuned.data.paths import DEFAULT_CONFIG


def default_raw_paths(root: Path) -> list[Path]:
    """Every raw log under the workdir, sorted for a stable sweep order.

    Order does not change the outcome (orphan judgements are deferred and
    retried after the sweep), but a stable listing keeps two operators' runs
    comparable line for line.
    """
    return sorted(
        p for kind in ("gen", "judge") for p in (root / "raw" / kind).glob("*/*.ndjson")
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from tuned.data.config import load_build_config
    from tuned.data.paths import build_paths
    from tuned.data.store import Store

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--raw", action="append", default=None, type=Path,
        help="raw NDJSON file to fold in (repeatable); default: sweep the workdir",
    )
    args = parser.parse_args(argv)

    cfg = load_build_config(args.config)
    paths = build_paths(cfg.build.workdir).ensure()
    raw_paths = args.raw if args.raw else default_raw_paths(paths.root)

    store = Store.open(paths.state_db)
    try:
        recovered = store.reconcile_raw(raw_paths)
        print(f"swept {len(raw_paths)} file(s), recovered {recovered} row(s)")
        print(
            "task states: "
            + ", ".join(f"{k}={v}" for k, v in sorted(store.task_counts().items()))
        )
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
