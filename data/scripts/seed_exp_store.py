"""Seed an isolated experiment store from the live control, read-only.

The exp_* arms before this one were seeded by hand. This copies the
`source` table and a deterministic per-source sample of `seed` rows out of
a live store into the arm's own store, so an arm can be stood up in one
command and the live database is never opened for write.

    python data/scripts/seed_exp_store.py \
        --config data/configs/data_law_v1_exp_<arm>.yaml \
        --from data/build/state/law_v1.sqlite3 --per-source 200 --seed 3407

An arm's config sets `build.workdir` to `data/build/exp_<arm>` and adds that
directory name to `paths.ISOLATED_WORKDIR_SIBLINGS`; without the second step
`is_live_control_workdir` treats the arm as the live control, which is the
deny-by-default answer and the safe one.

THE SAMPLE HAS NO LENGTH FILTER, on purpose. tasks._candidate_seeds refuses
a seed longer than seed_token_budget(cfg); the only way to prove that live
is to have oversize seeds in the store and zero tasks planned against them.
The per-source `oversize` count printed here is the numerator the report
needs.

Deterministic without an RNG: seed_ids are content-derived hashes, so
ORDER BY seed_id is already a stable pseudo-random order, and --seed picks
the starting offset inside it (wrapping at the source's end). The same
arguments produce the same rows on any machine.

Idempotent: Store.upsert_seeds is INSERT OR REPLACE on the primary key, so a
re-run rewrites the same rows and adds none.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

from tuned.data.config import load_build_config
from tuned.data.paths import build_paths, is_live_control_workdir
from tuned.data.store import Store
from tuned.data.tasks import seed_token_budget


def _open_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _sample(live: sqlite3.Connection, source_id: str, *, per_source: int, offset_seed: int):
    """`per_source` rows of one source, ORDER BY seed_id, starting at
    offset_seed mod count and wrapping - so a large offset never comes back
    short, and the same arguments always name the same rows."""
    total = live.execute(
        "SELECT COUNT(*) FROM seed WHERE source_id = ?", (source_id,)
    ).fetchone()[0]
    if total == 0:
        return []
    start = offset_seed % total
    take = min(per_source, total)
    head = live.execute(
        "SELECT * FROM seed WHERE source_id = ? ORDER BY seed_id LIMIT ? OFFSET ?",
        (source_id, take, start),
    ).fetchall()
    if len(head) < take:
        head += live.execute(
            "SELECT * FROM seed WHERE source_id = ? ORDER BY seed_id LIMIT ?",
            (source_id, take - len(head)),
        ).fetchall()
    return head


def seed_store(
    store: Store, live_db: Path, *, per_source: int, offset_seed: int, budget: int
) -> dict[str, dict]:
    """Copy `source` and a per-source seed sample from live_db into store.

    Returns {source_id: {"copied": n, "oversize": n_over_budget}}.
    """
    live = _open_ro(Path(live_db))
    try:
        report: dict[str, dict] = {}
        for src in live.execute("SELECT * FROM source ORDER BY source_id").fetchall():
            store.upsert_source(
                src["source_id"], src["license"], url=src["url"], version=src["version"]
            )
            rows = [dict(r) for r in _sample(
                live, src["source_id"], per_source=per_source, offset_seed=offset_seed
            )]
            store.upsert_seeds(rows)
            oversize = sum(1 for r in rows if (r.get("token_count") or 0) > budget)
            report[src["source_id"]] = {"copied": len(rows), "oversize": oversize}
        return report
    finally:
        live.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="the ARM's build config")
    parser.add_argument("--from", dest="live_db", required=True,
                        help="the live store to copy FROM (opened read-only)")
    parser.add_argument("--per-source", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3407, help="offset seed (see docstring)")
    args = parser.parse_args(argv)

    cfg = load_build_config(args.config, allow_unpinned=True)
    if is_live_control_workdir(cfg.build.workdir):
        print(
            f"refusing: {args.config} points at the live control workdir "
            f"{cfg.build.workdir!r}; an arm must have its own",
            file=sys.stderr,
        )
        return 2
    live_db = Path(args.live_db)
    if not live_db.is_file():
        print(f"no such live store: {live_db}", file=sys.stderr)
        return 2
    budget = seed_token_budget(cfg)
    paths = build_paths(cfg.build.workdir).ensure()
    with Store.open(paths.state_db) as store:
        report = seed_store(
            store, live_db, per_source=args.per_source, offset_seed=args.seed, budget=budget
        )
        total = store.seed_count()
    print(f"arm store: {paths.state_db}")
    print(f"seed budget (max_seq_length - reply reserve): {budget} tokens")
    for source_id, counts in report.items():
        print(f"  {source_id}: copied {counts['copied']}, over budget {counts['oversize']}")
    print(f"seeds in arm store: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
