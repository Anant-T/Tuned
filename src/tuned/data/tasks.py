"""Wave planner - turns seed rows into claimable `task` rows.

A wave is a batch of work the generation fleet can chew through unattended:
which seed, which task type, which prompt paraphrase, which sample index.
Nothing here calls a model and nothing here spends money; the planner is
pure bookkeeping over the store, and every decision it makes is
deterministic so a wave planned twice is the same wave.

THREE PROPERTIES, and each one exists because the pipeline runs for weeks
without supervision:

1. DETERMINISM. task_id = sha256(seed_id|task_type|prompt_id|sample_ix)[:16]
   and prompt_id = prompt_registry.pick_variant(task_type, seed_id,
   sample_ix), which is itself a hash. Nothing draws from an RNG, so the same
   store state plans the same rows in the same order on any machine, on any
   day, under any PYTHONHASHSEED.

2. `n` IS A TARGET, NOT AN INCREMENT. plan_wave(store, cfg, stream, 300)
   tops the (stream, arm) queue up TO 300 planned tasks; it does not add 300
   more every time it runs. That is what makes replanning after a crash safe:
   the operator re-runs the same command and gets whatever the interrupted
   run did not manage to insert, never a second wave nobody asked for.
   store.create_tasks is INSERT OR IGNORE on top of that, so even a
   task_id collision costs nothing. To plan a SECOND wave, ask for a bigger
   n (600) or label the new one with a different --arm.

3. PER-SEED CAP. A seed may back at most PER_SEED_CAP (4) tasks. Resampling
   the same seed beats reaching for more seeds (OpenThoughts), but only up to
   a point: past a handful of samples the teacher is re-answering a question
   it has effectively already answered, and the dataset gains duplication
   rather than coverage. Seeds are drawn fewest-samples-first, so a wave
   spreads across unused seeds before it ever resamples one.

`arm` labels an A/B cell ("unscripted"/"scripted") and is part of the queue
identity: an armed wave and an unarmed wave are counted separately, so
planning 300 unarmed tasks and then 100 "scripted" ones does not make the
unarmed wave look finished.

READ-ONLY SQL, DELIBERATELY. store.py is meant to be the only module that
runs SQL, but it exposes no seed-iteration or per-seed-count API and this
task's brief allows exactly one (unrelated) edit to it. The three SELECTs
below therefore go through the documented `store.conn` escape hatch. They
are strictly reads; every WRITE still goes through the store API
(create_tasks / log_event), so the transaction discipline that module owns
is untouched.

Plan:  python -m tuned.data.tasks --config configs/data_law_v1.yaml
       --stream synthesis --n 300 [--arm unscripted] [--mix irac_analysis=0.5,...]
"""

import hashlib
from collections.abc import Mapping, Sequence

from tuned.data import prompt_registry

# Max tasks any one seed may back, across every task type and wave.
PER_SEED_CAP = 4

TASK_ID_LEN = 16

# Default task-type mix per stream. Overridable per call (task_type_mix=) and
# from the CLI (--mix), which is where a wave that wants a different balance
# says so; these are the shipped defaults, not a closed vocabulary.
#
# synthesis: the pilot mix from the plan - IRAC analysis carries the stream,
# statute_qa is the construction workhorse, drafting and summarization keep
# the dataset from being one task in four costumes.
SYNTHESIS_MIX = {
    "irac_analysis": 0.40,
    "statute_qa": 0.25,
    "drafting": 0.18,
    "summarization": 0.17,
}

# curated_c2 is the teacher-REWRITE slice (PredEx-style rows re-expressed as
# reasoning + answer rather than synthesised from scratch), so its two useful
# shapes are "state what was decided" and "analyse it properly". Provisional
# until P8 has pass-rate data - it is a default, and --mix overrides it.
CURATED_C2_MIX = {"summarization": 0.60, "irac_analysis": 0.40}

# transition rows are the whole point of their own stream: one task type.
TRANSITION_MIX = {"transition": 1.0}

STREAM_MIX: dict[str, dict[str, float]] = {
    "synthesis": SYNTHESIS_MIX,
    "curated_c2": CURATED_C2_MIX,
    "transition": TRANSITION_MIX,
}

# Streams that reach a teacher at all. replay rows are copied, never
# generated, and never enter this table.
PLANNABLE_STREAMS = tuple(STREAM_MIX)


def task_id_for(seed_id: str, task_type: str, prompt_id: str, sample_ix: int) -> str:
    """The task's identity - and therefore what INSERT OR IGNORE dedupes on.

    All four fields are in it: the same seed asked the same question under a
    different paraphrase is a different task (the pilot compares paraphrases),
    and the same paraphrase at a different sample index is a different task
    (that is what resampling means).
    """
    key = f"{seed_id}|{task_type}|{prompt_id}|{sample_ix}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:TASK_ID_LEN]


def default_mix(stream: str) -> dict[str, float]:
    try:
        return dict(STREAM_MIX[stream])
    except KeyError:
        raise KeyError(
            f"no default task-type mix for stream {stream!r}; known streams: "
            f"{', '.join(PLANNABLE_STREAMS)} (pass task_type_mix= to plan another)"
        ) from None


def allocate(mix: Mapping[str, float], n: int) -> dict[str, int]:
    """Split `n` slots across task types by weight; the parts sum to exactly n.

    Largest-remainder, with ties broken by task-type NAME rather than by
    dict order - two operators writing the same mix in a different order must
    plan the identical wave, and Python dicts preserve insertion order.
    """
    if n <= 0:
        return {}
    weights = {k: float(v) for k, v in mix.items() if float(v) > 0}
    if not weights:
        raise ValueError(f"task-type mix has no positive weights: {dict(mix)!r}")
    total = sum(weights.values())
    exact = {k: n * w / total for k, w in weights.items()}
    counts = {k: int(v) for k, v in exact.items()}
    short = n - sum(counts.values())
    # -remainder first, then name: a stable, machine-independent order.
    order = sorted(exact, key=lambda k: (-(exact[k] - counts[k]), k))
    for i in range(short):
        counts[order[i % len(order)]] += 1
    return {k: c for k, c in sorted(counts.items()) if c > 0}


def _existing_in_queue(store, stream: str, arm: str | None) -> int:
    """How many tasks the (stream, arm) queue already holds.

    arm is matched exactly, NULL included: an unarmed wave and an armed wave
    are separate queues, so planning the A/B cells does not make the main
    wave look complete.
    """
    if arm is None:
        sql = "SELECT COUNT(*) FROM task WHERE stream = ? AND arm IS NULL"
        params: tuple = (stream,)
    else:
        sql = "SELECT COUNT(*) FROM task WHERE stream = ? AND arm = ?"
        params = (stream, arm)
    return int(store.conn.execute(sql, params).fetchone()[0])


def _candidate_seeds(store, *, limit: int, sources: Sequence[str] | None) -> list[tuple[str, int]]:
    """(seed_id, tasks already planned on it), fewest-first, under the cap.

    Ordered (n_tasks ASC, seed_id ASC): unused seeds before resampled ones,
    and a total order inside each tier so the choice never depends on SQLite's
    row layout.
    """
    clauses = ["COALESCE(t.n, 0) < ?"]
    params: list = [PER_SEED_CAP]
    if sources:
        clauses.append(f"s.source_id IN ({', '.join('?' * len(sources))})")
        params.extend(sources)
    params.append(limit)
    rows = store.conn.execute(
        "SELECT s.seed_id, COALESCE(t.n, 0) AS n_tasks FROM seed s "
        "LEFT JOIN (SELECT seed_id, COUNT(*) AS n FROM task GROUP BY seed_id) t "
        "  ON t.seed_id = s.seed_id "
        f"WHERE {' AND '.join(clauses)} "
        "ORDER BY n_tasks ASC, s.seed_id ASC LIMIT ?",
        params,
    ).fetchall()
    return [(row[0], int(row[1])) for row in rows]


def _sample_counts(store, seed_ids: Sequence[str]) -> dict[tuple[str, str], int]:
    """(seed_id, task_type) -> how many samples already exist.

    That count IS the next sample_ix, which is why it is read per pair rather
    than per seed: a seed already carrying two irac_analysis tasks starts its
    first statute_qa task at index 0, not at index 2.
    """
    counts: dict[tuple[str, str], int] = {}
    ids = list(seed_ids)
    # SQLITE_MAX_VARIABLE_NUMBER is 999 on older builds; chunk rather than
    # trust the host's limit.
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        rows = store.conn.execute(
            f"SELECT seed_id, task_type, COUNT(*) FROM task "
            f"WHERE seed_id IN ({', '.join('?' * len(chunk))}) "
            f"GROUP BY seed_id, task_type",
            chunk,
        ).fetchall()
        for seed_id, task_type, n in rows:
            counts[(seed_id, task_type)] = int(n)
    return counts


def plan_rows(
    store,
    cfg,
    stream: str,
    n: int,
    *,
    task_type_mix: Mapping[str, float] | None = None,
    arm: str | None = None,
    sources: Sequence[str] | None = None,
) -> list[dict]:
    """The rows plan_wave would insert - pure enough to test and to preview.

    Reads the store, writes nothing. `cfg` is accepted for signature parity
    with the other builders (and as the hook for future per-stream policy);
    no field is read off it today.
    """
    planned_already = _existing_in_queue(store, stream, arm)
    wanted = max(0, n - planned_already)
    if wanted == 0:
        return []

    mix = dict(task_type_mix) if task_type_mix else default_mix(stream)
    for task_type in mix:
        # Fail here, before anything is inserted, rather than at render time
        # when a worker has already claimed the row and is about to spend.
        prompt_registry.variants(task_type)
    quota = allocate(mix, wanted)

    candidates = _candidate_seeds(store, limit=wanted, sources=sources)
    if not candidates:
        return []
    pair_counts = _sample_counts(store, [seed_id for seed_id, _ in candidates])
    used = {seed_id: n_tasks for seed_id, n_tasks in candidates}
    order = [seed_id for seed_id, _ in candidates]

    rows: list[dict] = []
    cursor = 0
    for task_type in sorted(quota):
        for _ in range(quota[task_type]):
            seed_id = None
            # One pass over the ring: if every candidate is at the cap there
            # is nothing left to plan and the wave is simply short.
            for _probe in range(len(order)):
                candidate = order[cursor % len(order)]
                cursor += 1
                if used[candidate] < PER_SEED_CAP:
                    seed_id = candidate
                    break
            if seed_id is None:
                return rows
            used[seed_id] += 1
            sample_ix = pair_counts.get((seed_id, task_type), 0)
            pair_counts[(seed_id, task_type)] = sample_ix + 1
            prompt_id = prompt_registry.pick_variant(task_type, seed_id, sample_ix)
            rows.append(
                {
                    "task_id": task_id_for(seed_id, task_type, prompt_id, sample_ix),
                    "seed_id": seed_id,
                    "stream": stream,
                    "task_type": task_type,
                    "prompt_id": prompt_id,
                    "prompt_sha": prompt_registry.load(prompt_id).sha,
                    "sample_ix": sample_ix,
                    "arm": arm,
                }
            )
    return rows


def commit_rows(store, rows: Sequence[dict], *, stream: str, arm: str | None, target: int) -> int:
    """Insert the planned rows and record what happened. Returns rows created.

    A count below len(rows) means task_ids collided with rows already in the
    table - the crash-resume path working, not an error, so it is logged
    rather than raised.
    """
    created = store.create_tasks(rows)
    store.log_event(
        "wave_planned",
        {
            "stream": stream,
            "arm": arm,
            "target": target,
            "rows": len(rows),
            "created": created,
            "collided": len(rows) - created,
        },
    )
    return created


def plan_wave(
    store,
    cfg,
    stream: str,
    n: int,
    *,
    task_type_mix: Mapping[str, float] | None = None,
    arm: str | None = None,
    sources: Sequence[str] | None = None,
) -> int:
    """Top the (stream, arm) queue up to `n` tasks; returns rows NEWLY created.

    Re-running the same command adds 0 - see the module docstring on why `n`
    is a target rather than an increment.
    """
    rows = plan_rows(
        store, cfg, stream, n, task_type_mix=task_type_mix, arm=arm, sources=sources
    )
    return commit_rows(store, rows, stream=stream, arm=arm, target=n)


def parse_mix(spec: str) -> dict[str, float]:
    """"irac_analysis=0.5,statute_qa=0.5" -> {"irac_analysis": 0.5, ...}."""
    mix: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--mix entry must be task_type=weight, got {part!r}")
        name, _, weight = part.partition("=")
        mix[name.strip()] = float(weight)
    if not mix:
        raise ValueError(f"--mix parsed to nothing: {spec!r}")
    return mix


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from tuned.data.config import load_build_config
    from tuned.data.paths import build_paths
    from tuned.data.store import Store

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data_law_v1.yaml")
    parser.add_argument("--stream", default="synthesis")
    parser.add_argument("--n", type=int, required=True, help="target task count for the queue")
    parser.add_argument("--arm", default=None, help="A/B label, e.g. unscripted|scripted")
    parser.add_argument("--mix", default=None, help="task_type=weight,... (overrides the default)")
    parser.add_argument("--source", action="append", default=None, help="restrict to a source_id")
    args = parser.parse_args(argv)

    cfg = load_build_config(args.config)
    paths = build_paths(cfg.build.workdir).ensure()
    store = Store.open(paths.state_db)
    try:
        mix = parse_mix(args.mix) if args.mix else None
        rows = plan_rows(
            store, cfg, args.stream, args.n, task_type_mix=mix, arm=args.arm, sources=args.source
        )
        created = commit_rows(store, rows, stream=args.stream, arm=args.arm, target=args.n)
        by_type: dict[str, int] = {}
        for row in rows:
            by_type[row["task_type"]] = by_type.get(row["task_type"], 0) + 1
        print(f"stream={args.stream} arm={args.arm or '-'} target={args.n}")
        print(f"planned {created}  skipped {args.n - created}  (rows derived {len(rows)})")
        for task_type, count in sorted(by_type.items()):
            print(f"  {task_type:<18}{count:>6}")
        counts = store.task_counts()
        print("task states: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
