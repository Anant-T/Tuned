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

Run:  python -m tuned.data.verify --config configs/data_law_v1.yaml
      [--index data/build/corpus/citations.txt] [--state accepted]
"""

import sys
from collections.abc import Sequence

from tuned.data import gates
from tuned.data.generate import SlotError, build_prompt, gate_context
from tuned.data.jsonl import read_at

REJECTED_STATE = "rejected"

# Failing one of these means the row is wrong about the law. gates.py owns
# the list; it is re-exported here so the demotion rule and the gate
# semantics can never drift apart.
PERMANENT_GATES = gates.PERMANENT_GATES


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
    """
    clause = "WHERE t.state = ?" if where_state else ""
    params: tuple = (where_state,) if where_state else ()
    rows = store.conn.execute(
        "SELECT g.*, t.stream, t.seed_id, t.task_type, t.prompt_id, t.prompt_sha, "
        "       t.sample_ix, t.arm, t.state AS task_state "
        "FROM generation g "
        "JOIN task t ON t.task_id = g.task_id "
        "JOIN (SELECT task_id, MAX(attempt) AS a FROM generation GROUP BY task_id) m "
        "  ON m.task_id = g.task_id AND m.a = g.attempt "
        f"{clause} ORDER BY g.gen_id",
        params,
    ).fetchall()
    return [dict(row) for row in rows]


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
) -> dict:
    """Re-gate stored generations with the real citation index; returns counts.

    `demote_states` is the set of task states a demotion may act on: a row
    that is already rejected needs no demoting, and one still pending has not
    been judged yet. Gate results are rewritten for every row scanned either
    way (record_gates is INSERT OR REPLACE), so the instrumentation is
    refreshed even where the disposition does not move.
    """
    index = load_index(citation_index_path)
    counts = {
        "scanned": 0,
        "regated": 0,
        "clean": 0,
        "demoted": 0,
        "soft_fail": 0,
        "missing_seed": 0,
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
        except (SlotError, KeyError) as exc:  # unrenderable seed / retired prompt id
            counts["missing_seed"] += 1
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
    args = parser.parse_args(argv)

    cfg = load_build_config(args.config)
    paths = build_paths(cfg.build.workdir).ensure()
    store = Store.open(paths.state_db)
    try:
        counts = rerun_gates(
            store, cfg, where_state=args.state, citation_index_path=args.index
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
