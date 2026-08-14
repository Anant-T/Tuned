"""Offline gate re-runner - the mandatory second pass over the citation gate.

gates.py records a hard dependency and this module discharges it. A
GateContext built with citation_index=None runs only the SUSPECT half of the
citation gate; the EXISTENCE half is skipped and the stored detail says so
({"novel": None, "novel_skipped": "no-index"}). The pilot runs in exactly
that mode, because the 17M-row citation index does not exist yet when the
first waves go out. A gate row carrying novel_skipped therefore means "never
checked", never "passed", and no row may enter the dataset until this module
has re-run its gates WITH the real index.

Three properties matter:

* IT RE-SCORES THE ORIGINAL BYTES. The exact content the gates saw was
  written into the raw envelope at generation time, so the re-run reads it
  back with jsonl.read_at(raw_path, raw_offset) instead of re-assembling it
  from the think/answer columns. Re-assembly is close, not identical (leading
  prose before the think tag, a model's own spacing), and a verification pass
  that scores a slightly different string can demote a row for a difference
  it introduced itself. Re-assembly is the FALLBACK, used only when the raw
  line is unreadable, and it is recorded in the counts.

* IT USES THE SAME CONTEXT BUILDER as generation. generate.gate_context is
  the single definition of what a row's gates are run against - grounding
  text, dates, answer key, expectations - so the second pass measures the
  same thing the first one did, with only the index changed.

* IT ONLY EVER DEMOTES. A row that now fails a PERMANENT gate (citations,
  temporal, answer_key) is moved to rejected and the demotion is logged as a
  run_event. A row that now fails a soft gate is logged and left alone: soft
  failures are what regeneration is for, and re-opening an accepted row for
  a formatting wobble months later would churn the dataset for no gain.
  Nothing is ever promoted here - a row the judges rejected stays rejected.

RUN IT WHEN THE FLEET IS IDLE. A demotion is an unfenced task-state write
(there is no lease to hold - this pass never claimed anything), so a task a
judge worker is holding at that instant could have its state overwritten
mid-decision. Assembly-time is the intended moment; between waves is fine.
The CLI refuses to start while any lease is live, and every demotion
re-checks its OWN row's lease immediately before writing - the opening check
is a snapshot, and a sweep over tens of thousands of rows outlives it.

Run:  python -m tuned.data.verify --config configs/data_law_v1.yaml
      [--index data/build/corpus/citations.txt] [--state accepted]
"""

import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from tuned.data import gates
from tuned.data.generate import SlotError, build_prompt, gate_context
from tuned.data.jsonl import read_at
from tuned.data.store import DEFAULT_LEASE_S, _TS_FMT

REJECTED_STATE = "rejected"

# DEFAULT_LEASE_S and _TS_FMT are IMPORTED, never re-declared: this module
# decides whether a task is under a live lease, and a private copy of either
# constant is a fence that silently disagrees with the fencing it is meant to
# respect.
__all__ = ["DEFAULT_LEASE_S", "live_leases", "rerun_gates", "content_for", "main"]

# Failing one of these means the row is wrong about the law. gates.py owns
# the list; it is re-exported here so the demotion rule and the gate
# semantics can never drift apart.
PERMANENT_GATES = gates.PERMANENT_GATES


def live_leases(store, lease_s: int = DEFAULT_LEASE_S) -> int:
    """Tasks a worker is holding RIGHT NOW (claimed inside the lease window).

    This pass writes task states without holding a lease of its own - it
    never claimed anything - so running it against a live fleet can overwrite
    a decision a judge worker is in the middle of making. Counting the live
    leases is how the CLI refuses to do that by accident.
    """
    return int(
        store.conn.execute(
            "SELECT COUNT(*) FROM task WHERE claimed_at IS NOT NULL AND claimed_at >= ?",
            (_lease_cutoff(lease_s),),
        ).fetchone()[0]
    )


def _lease_cutoff(lease_s: int) -> str:
    return (datetime.now(UTC) - timedelta(seconds=lease_s)).strftime(_TS_FMT)


def lease_is_live(store, task_id: str, lease_s: int = DEFAULT_LEASE_S) -> bool:
    """Is THIS task under a live lease right now?

    live_leases() is checked once, at the top of the CLI. A sweep over tens of
    thousands of generations takes long enough for a worker to claim a row
    after that check passed, and rerun_gates demotes into judging_active
    without holding a lease of its own - so the row that matters is re-checked
    at the moment it would be overwritten, not at the moment the pass began.
    """
    row = store.conn.execute(
        "SELECT claimed_at FROM task WHERE task_id = ?", (task_id,)
    ).fetchone()
    return bool(row and row[0] and row[0] >= _lease_cutoff(lease_s))


def load_index(path):
    """The real CitationIndex, or None for a (pointless but harmless) dry run."""
    if path is None:
        return None
    from tuned.data.citations import CitationIndex

    return CitationIndex.load(path)


def latest_generations(store, *, where_state: str | None = None) -> list[dict]:
    """The newest generation of every task, joined to its task row.

    Newest only: a task's disposition is about the answer it currently
    stands on, and re-gating superseded attempts would demote a task for a
    draft that was already replaced.

    The join itself lives in store.py (all SQL does), and the assembly pass
    reads the same rows through the same method - so "which generation is the
    row" has one answer, not one per caller.
    """
    return store.latest_generations(where_state)


def content_for(cfg, gen: dict) -> tuple[str, str]:
    """(content, how) - the exact scored bytes, or the honest fallback.

    how is "raw" when the original envelope was readable and "rebuilt" when
    it was not, so a demotion made on re-assembled text is identifiable
    afterwards rather than indistinguishable from one made on the original.
    """
    raw_path, raw_offset = gen.get("raw_path"), gen.get("raw_offset")
    if raw_path and raw_offset is not None:
        try:
            record = read_at(raw_path, int(raw_offset))
        except (OSError, ValueError):
            record = None
        if isinstance(record, dict):
            content = record.get("content")
            if content:
                return str(content), "raw"
    think, answer = gen.get("think"), gen.get("answer") or ""
    if think:
        return f"{cfg.think_open}\n{think}\n{cfg.think_close}\n\n{answer}", "rebuilt"
    return answer, "rebuilt"


def rerun_gates(
    store,
    cfg,
    *,
    where_state: str | None = None,
    citation_index_path=None,
    demote_states: Sequence[str] = ("accepted", "judging", "judging_active"),
    respect_leases: bool = True,
) -> dict:
    """Re-gate stored generations with the real citation index; returns counts.

    `demote_states` is the set of task states a demotion may act on: a row
    that is already rejected needs no demoting, and one still pending has not
    been judged yet. Gate results are rewritten for every row scanned either
    way (record_gates is INSERT OR REPLACE), so the instrumentation is
    refreshed even where the disposition does not move.

    `respect_leases` skips (and counts) any demotion whose row a worker holds
    right now. It is the CLI's --force that turns this off, and turning it
    off means accepting that a judge worker's decision may be overwritten
    mid-flight.
    """
    index = load_index(citation_index_path)
    counts = {
        "scanned": 0,
        "regated": 0,
        "clean": 0,
        "demoted": 0,
        "soft_fail": 0,
        "missing_seed": 0,
        # Counted apart from missing_seed: a row whose SLOTS no longer render
        # (a transition row generated before the dates became mandatory) is a
        # row that will never be verified, and folding it into "missing seed"
        # hid that class inside a number that reads as "nothing to do here".
        "slot_error": 0,
        # Demotions skipped because a worker holds the row right now.
        "held_by_worker": 0,
        "rebuilt_content": 0,
        "unverified": 0,
    }
    if index is None:
        # Loud, because a verify pass without an index re-runs the SAME
        # skipped-existence check the pilot already ran, and reports "clean".
        store.log_event("verify_no_index", {"where_state": where_state})

    for gen in latest_generations(store, where_state=where_state):
        counts["scanned"] += 1
        seed = store.get_seed(gen["seed_id"])
        if seed is None:
            counts["missing_seed"] += 1
            continue
        try:
            bundle = build_prompt(cfg, gen, seed)
        except (SlotError, KeyError) as exc:  # unrenderable slots / retired prompt id
            counts["slot_error"] += 1
            store.log_event(
                "verify_skipped",
                {"task_id": gen["task_id"], "reason": f"{type(exc).__name__}: {exc}"[:200]},
            )
            continue

        content, how = content_for(cfg, gen)
        if how == "rebuilt":
            counts["rebuilt_content"] += 1
        ctx = gate_context(cfg, gen, seed, bundle.grounding, citation_index=index)
        results = gates.run_all(content, bundle.prompt_est_tokens, ctx)
        store.record_gates(int(gen["gen_id"]), [g.as_row() for g in results])
        counts["regated"] += 1
        if index is None:
            counts["unverified"] += 1

        disposition = gates.disposition(results)
        failed = [g.gate for g in results if not g.passed]
        if disposition is None:
            counts["clean"] += 1
            continue
        permanent = [gate for gate in failed if gate in PERMANENT_GATES]
        if not permanent:
            counts["soft_fail"] += 1
            store.log_event(
                "verify_soft_fail",
                {"task_id": gen["task_id"], "gen_id": int(gen["gen_id"]), "gates": failed},
            )
            continue
        if gen["task_state"] not in demote_states:
            continue
        if respect_leases and lease_is_live(store, gen["task_id"]):
            # Claimed since this pass started. The demotion is unfenced, so
            # writing it now would overwrite a decision a worker is making at
            # this instant; the next pass picks the row up.
            counts["held_by_worker"] += 1
            store.log_event(
                "verify_demotion_deferred",
                {
                    "task_id": gen["task_id"],
                    "gen_id": int(gen["gen_id"]),
                    "from_state": gen["task_state"],
                    "gates": permanent,
                },
            )
            continue
        counts["demoted"] += 1
        store.log_event(
            "verify_demotion",
            {
                "task_id": gen["task_id"],
                "gen_id": int(gen["gen_id"]),
                "from_state": gen["task_state"],
                "gates": permanent,
                "content_source": how,
                "detail": [
                    g.detail for g in results if g.gate == "citations" and not g.passed
                ],
            },
        )
        store.set_task_state(
            gen["task_id"], REJECTED_STATE, "verify:" + ",".join(permanent)
        )
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from tuned.data.config import load_build_config
    from tuned.data.paths import build_paths
    from tuned.data.store import Store

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data_law_v1.yaml")
    parser.add_argument("--index", default=None, help="path to the citation index")
    parser.add_argument("--state", default=None, help="only re-gate tasks in this state")
    parser.add_argument(
        "--force", action="store_true", help="run even while workers hold live leases"
    )
    args = parser.parse_args(argv)

    cfg = load_build_config(args.config)
    paths = build_paths(cfg.build.workdir).ensure()
    store = Store.open(paths.state_db)
    try:
        live = live_leases(store)
        if live and not args.force:
            print(
                f"REFUSING: {live} task(s) are under a live worker lease. This pass "
                f"writes task states without holding one, so it can overwrite a "
                f"decision a worker is making right now. Stop the fleet (or wait "
                f"{DEFAULT_LEASE_S}s for the leases to expire), or pass --force."
            )
            return 2
        counts = rerun_gates(
            store,
            cfg,
            where_state=args.state,
            citation_index_path=args.index,
            respect_leases=not args.force,
        )
        if args.index is None:
            print("WARNING: no --index given; the citation-existence half stays UNVERIFIED")
        for key, value in counts.items():
            print(f"{key:<18}{value:>8}")
        print("task states: " + ", ".join(f"{k}={v}" for k, v in sorted(store.task_counts().items())))
    finally:
        store.close()
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
