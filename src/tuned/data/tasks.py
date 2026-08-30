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

TERMINALLY DEAD ROWS DO NOT COUNT toward the target; recoverable parks do.
See _existing_in_queue and TERMINALLY_DEAD - the difference is whether the
row can still become a dataset row, and the bound on replacing dead ones is
the per-seed cap.
This module also owns the RE-OPEN path (reopen_tasks / --reopen): the
workers park a row whenever the failure is about the pool rather than the
answer, and parking is only survivable if something can un-park it.

READ-ONLY SQL, DELIBERATELY. store.py is meant to be the only module that
runs SQL, but it exposes no seed-iteration or per-seed-count API and this
task's brief allows exactly one (unrelated) edit to it. The three SELECTs
below therefore go through the documented `store.conn` escape hatch. They
are strictly reads; every WRITE still goes through the store API
(create_tasks / log_event), so the transaction discipline that module owns
is untouched.

Plan:  python -m tuned.data.tasks --config data/configs/data_law_v1.yaml
       --stream synthesis --n 300 [--arm unscripted] [--mix irac_analysis=0.5,...]
"""

import hashlib
from collections.abc import Mapping, Sequence

from tuned.data import prompt_registry
from tuned.data.paths import DEFAULT_CONFIG

# Max tasks any one seed may back, across every task type and wave.
PER_SEED_CAP = 4

# What the assistant turn needs, subtracted from the train config's
# max_seq_length to leave a seed budget the planner can honour.
#
# assemble.py already drops a row whose rendered length exceeds
# max_seq_length - but at the FAR END of the pipeline, after the teacher has
# been paid for the generation. Measured 2026-08-26 over 1,368 real
# generations (git 3406377^:docs/reports/2026-08-26-row-length-under-
# deepseek-traces.md - that one report has no prev_rep section):
# 16 rows were dropped that way, averaging a 7,597-token seed against a
# 492-token trace, and all 16 came from PredEx and TathyaNyaya - the two
# sources chunks.py does not chunk. The drop is therefore silent at the point
# it costs a call, and biased, since what it removes is the longest and most
# substantive cases.
#
# 3,500 = a 2,097-token reasoning trace (deepseek-v4-flash at the shipped
# `reasoning_effort: low`) + a ~1,331-token p99 answer + ~30 of chat template.
# It is a RESERVE, not a prediction: a shorter reply simply leaves the row
# further inside the cap, and assemble.py remains the backstop for a reply
# that overruns it.
REPLY_RESERVE_TOKENS = 3_500

TASK_ID_LEN = 16

# Default task-type mix per stream. Overridable per call (task_type_mix=) and
# from the CLI (--mix), which is where a wave that wants a different balance
# says so; these are the shipped defaults, not a closed vocabulary.
#
# synthesis: the pilot mix from the plan - IRAC analysis carries the stream,
# statute_qa is the construction workhorse, drafting and summarization keep
# the dataset from being one task in four costumes.
#
# drafting is PARKED at 0.0, not removed (2026-08-24). Every drafting prompt
# renders placeholder slots - document_kind / party_context / focus_issue /
# question are empty on all 60,603 seeds - so the teacher is asked to draft
# for "the party whose papers these are" against a judgment that already
# disposed of the matter. 4/20 accepts vs summarization's 13/20 (p=0.0095)
# and 66,666 tok per accepted row vs 18,028. Its 0.18 goes to summarization,
# the cheapest converter. Restore it when seeds carry the fields, or when the
# stream is retargeted as a downstream instrument (~14,225 eligible seeds).
SYNTHESIS_MIX = {
    "irac_analysis": 0.40,
    "statute_qa": 0.25,
    "drafting": 0.00,
    "summarization": 0.35,
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

# What `--stream` means when it is not passed. The flag itself defaults to
# None so the CLI can tell "not passed" from "passed synthesis" - --reopen
# ignores --stream, and it has to say so rather than act on it.
DEFAULT_STREAM = "synthesis"

# Parking state -> the queue state it belongs back in. These are the states a
# worker uses when the failure is about the POOL rather than about the answer
# (see reopen_tasks); each one goes back to the queue whose worker parked it,
# so a re-opened row resumes where it stopped instead of being generated
# again. `rejected` is absent on purpose - it is a decision, not a park.
REOPEN_STATES = {
    "gen_unroutable": "pending",
    "judge_unroutable": "judging",
    "judge_error": "judging",
    # Planned against a template that was edited afterwards. Re-openable
    # because restoring the template makes the row valid again, and free to
    # re-park if it was not: generate.py's guard fires before any render or
    # provider call.
    "stale_prompt": "pending",
    # Exhausted format/soft failures. Re-openable after a format or policy
    # change; not a legal reject.
    "format_parked": "pending",
    # Statute-QA planned without a genuine distinct provision. Re-openable
    # after the operator supplies section metadata; not a quality reject.
    "input_ineligible": "pending",
    # Generated by a teacher that is not in the shipped generator pool
    # (verify.py's --require-generator). Back to the GENERATOR, not to the
    # judges: the recovery this park exists for is re-generating the row with
    # the teacher the cut is defined over, not accepting the old one after all.
    "off_teacher": "pending",
}

# States a row can never leave under its own power, so they do NOT occupy
# queue capacity - see _existing_in_queue.
#
#   rejected      a decision: the gates or the judges said this is wrong.
#   stale_prompt  planned against template bytes that no longer exist; only a
#                 re-plan can produce a usable row for that seed.
#   input_ineligible  planned against statute input that is not a provision;
#                     a re-plan or a reopen after metadata is supplied can
#                     produce a usable row, so this must not fill the wave.
TERMINALLY_DEAD = frozenset({"rejected", "stale_prompt", "input_ineligible"})

# Parked dispositions whose ATTEMPTS WERE NEVER SPENT ON AN ANSWER, and which
# therefore get their budget back on re-open. Everything else keeps whatever it
# has already spent.
#
# THE DEFAULT IS TO PRESERVE, and that inversion is deliberate (2026-08-18,
# review round 3). Resetting by default is only right for the park this command
# was written for - a wave that could not route at all, or had no key, burnt
# its claims discovering a fact about the FLEET. Every other way a row can
# arrive here has a real billed generation behind it, and the fix round proved
# the cost: the stale-prompt cancellation swept up 6 rows that had genuinely
# exhausted three paid attempts, and under a blanket reset a `--reopen
# stale_prompt` would have handed all three back.
#
# THE JUDGE-SIDE PARKS ARE FREE, and for a reason worth writing down rather
# than inferring from the prefix: `attempts` counts GENERATION claims, and a
# judge park happens after a good generation already exists. The row goes back
# to `judging`, not to the generator, so the attempts it holds were spent
# producing the answer that is now waiting to be scored - not on discovering
# the judge pool was short. That is the original R3-C1 case and it still resets.
#
# `verify:off-teacher` IS ON THIS LIST, and it is the one entry whose row has a
# real billed answer behind it. Two facts put it here rather than under the
# preserve default. The attempts were spent producing an answer that a later
# RULING removed from the cut - a fact about the fleet the row was generated
# under, not about the answer. And the row cannot be replaced by any other
# route: task_id_for hashes (seed, task_type, prompt_id, sample_ix), so a
# re-plan at the same coordinates is INSERT OR IGNORE'd into the existing row.
# Handing it back at 3/3 attempts means the generator never claims it and the
# park is terminal in everything but name.
FREE_PARK_DISPOSITIONS = frozenset(
    {"unroutable:generator", "exhausted:provider-fault", "verify:off-teacher"}
)
FREE_PARK_PREFIXES = (
    "exhausted:unroutable:",
    "exhausted:format:",
    "reopened:from-",
    "judge-",
    "tiebreak:",
    "input-ineligible:",
)


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
    """How many LIVE tasks the (stream, arm) queue already holds.

    arm is matched exactly, NULL included: an unarmed wave and an armed wave
    are separate queues, so planning the A/B cells does not make the main
    wave look complete.

    A row counts only if it can still BECOME a dataset row. TERMINALLY_DEAD
    holds the states that cannot: a wave asks for n candidate rows, and one
    that produced nothing and never will leaves the wave permanently short of
    what the operator asked for. When a routing bug marked a whole wave
    rejected, re-running the plan reported "already at target" and did nothing
    at all - the wave could not replace rows it had lost.

    `stale_prompt` JOINED THAT SET on 2026-08-18 and it is not a cosmetic
    addition. The fix round cancelled 419 rows there, and while they still
    counted as live the queue was full of tasks no worker can ever claim:
    `plan_wave(..., 500)` topped up to 281 (500 - 219 already in the unarmed
    queue), and planning either 100-row A/B arm would have topped up to ZERO,
    because 100 stale rows already filled it. The seed pool was never the
    binding constraint - this count was.

    Everything else counts, INCLUDING the recoverable parking states. A parked
    row is re-openable and keeps whatever judgements it already paid for, so
    planning a replacement as well would quietly double the wave every time the
    pool had a bad afternoon.

    The bound on replacement is the per-seed cap, not this count:
    _candidate_seeds counts a seed's tasks regardless of state except
    `input_ineligible` and legacy ineligible statute-QA (`task_type =
    statute_qa` AND `disposition = skip:slots`). A seed whose
    PER_SEED_CAP tasks were all quality-rejected, or whose slots could
    not render for a non-statute reason, is never offered again. Other
    `skip:slots` rows still spend the cap so a generate-and-replan loop
    on empty source or missing transition dates cannot run forever.
    """
    placeholders = ", ".join("?" * len(TERMINALLY_DEAD))
    clauses = ["stream = ?", f"state NOT IN ({placeholders})"]
    params: list = [stream, *sorted(TERMINALLY_DEAD)]
    if arm is None:
        clauses.append("arm IS NULL")
    else:
        clauses.append("arm = ?")
        params.append(arm)
    return int(
        store.conn.execute(
            f"SELECT COUNT(*) FROM task WHERE {' AND '.join(clauses)}", params
        ).fetchone()[0]
    )


def seed_token_budget(cfg) -> int:
    """The longest seed that can still leave room for an assistant turn.

    Derived rather than configured: the seed budget and the training cap
    describe one row, and a literal here would drift out of step with
    train.main.max_seq_length the first time that moved.
    """
    return max(0, int(cfg.max_seq_length) - REPLY_RESERVE_TOKENS)


def _candidate_seeds(
    store, *, limit: int, sources: Sequence[str] | None, stream: str,
    max_seed_tokens: int,
) -> list[tuple[str, int]]:
    """(seed_id, tasks already planned on it), fewest-first, under the cap.

    Ordered (n_tasks ASC, seed_id ASC): unused seeds before resampled ones,
    and a total order inside each tier so the choice never depends on SQLite's
    row layout.

    A seed chunks.py flagged `oversize` is never offered. That flag means
    "this is one paragraph the packer was contractually forbidden to split,
    and it is over the token band" - measured at 26,818 tokens on one real
    judgment - so planning against it is exactly the prompt-budget blowout
    chunking exists to prevent. The flag was written from the first cut and
    read by nothing; this is where it is read. `json_valid` guards the
    extract so a row whose meta_json is not JSON (nothing this pipeline
    writes, but the column is free text) is treated as un-flagged rather
    than failing the whole query.

    A seed transition.py flagged `held_out` is never offered either, for a
    different reason with the same shape. Those are the eval reserve of the
    transition grid - cells kept back so the s.358 stream can be MEASURED,
    which only works while no teacher has ever answered them. transition.py
    already takes the reserve and the sample as disjoint prefixes of one
    order, so nothing here can put a reserved cell into the sample; this is
    the second half, and it is the half that matters, because a wave planned
    over every seed in the table would otherwise generate against the eval
    without anything in the pipeline noticing.

    A seed that DECLARES a stream is offered only to that stream's wave, and
    that is the third exclusion of the same shape - a property the seed states
    about itself, honoured here whoever called. Measured before the clause
    existed, on a store holding the transition grid: `plan_rows(store, cfg,
    "synthesis", 8)` - the CLI's own default, since --source defaults to None -
    planned eight tasks across drafting, irac_analysis, statute_qa and
    summarization on transition seeds. The transition QUESTION survived into
    them, because build_slots lets meta_json.question override the task-type
    default, so the teacher was asked which enactment governs with no provision
    block in front of it; and check_answer_key skipped the row entirely because
    ctx.stream was not "transition". A row carrying an answer key, ungraded
    against it.

    THE CLAUSE RATHER THAN A NARROWER CLI DEFAULT, decided by measurement:
    --source restricts by source_id and this module holds no map from a source
    to the stream it belongs to (the config's only source map is assemble's
    mix BUCKET map, which answers a different question). Excluding transition
    seeds by default would mean hard-coding another module's
    TRANSITION_SOURCE_ID here, and it would fix exactly one caller - the CLI -
    while plan_rows and plan_wave stayed open to every other.

    A seed that declares NOTHING stays eligible for any wave, which is every
    other builder's contract today: transition.py is the only writer of
    meta_json.stream on a seed row in the tree.

    A seed longer than `max_seed_tokens` is never offered, and that is the
    fourth exclusion of the same shape. It is the general case of the
    `oversize` clause above: that flag only exists on rows chunks.py has
    chunked, which is the InJudgements source alone, so PredEx and
    TathyaNyaya - 92.8% and 69.9% of whose seeds exceed the chunk band - can
    never carry it. Those two sources supplied every one of the 16 rows
    assemble.py dropped for length in the 2026-08-26 measurement. Chunking
    them is NOT the fix (seeds.predex_seed builds its text as facts + the
    court's reasoning, so a chunk is either an unanswerable fragment or the
    answer without its question); refusing to plan against them is.

    An unmeasured seed - token_count NULL - stays eligible, on the same
    contract as the clauses above. Nothing in the tree writes a NULL today;
    the COALESCE is there because the column permits one, and assemble.py is
    still the backstop.
    """
    clauses = [
        "COALESCE(t.n, 0) < ?",
        "COALESCE(CASE WHEN json_valid(s.meta_json) "
        "THEN json_extract(s.meta_json, '$.oversize') END, 0) = 0",
        "COALESCE(CASE WHEN json_valid(s.meta_json) "
        "THEN json_extract(s.meta_json, '$.held_out') END, 0) = 0",
        "COALESCE(CASE WHEN json_valid(s.meta_json) "
        "THEN json_extract(s.meta_json, '$.stream') END, ?) = ?",
        "COALESCE(s.token_count, 0) <= ?",
    ]
    params: list = [PER_SEED_CAP, stream, stream, max_seed_tokens]
    if sources:
        clauses.append(f"s.source_id IN ({', '.join('?' * len(sources))})")
        params.extend(sources)
    params.append(limit)
    rows = store.conn.execute(
        "SELECT s.seed_id, COALESCE(t.n, 0) AS n_tasks FROM seed s "
        "LEFT JOIN (SELECT seed_id, COUNT(*) AS n FROM task "
        "           WHERE state != 'input_ineligible' "
        "             AND NOT (task_type = 'statute_qa' "
        "                      AND COALESCE(disposition, '') = 'skip:slots') "
        "           GROUP BY seed_id) t "
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

    Reads the store, writes nothing. `cfg` supplies max_seq_length, from
    which seed_token_budget derives the longest seed worth planning against.
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

    candidates = _candidate_seeds(
        store, limit=wanted, sources=sources, stream=stream,
        max_seed_tokens=seed_token_budget(cfg),
    )
    if not candidates:
        return []
    pair_counts = _sample_counts(store, [seed_id for seed_id, _ in candidates])
    used = {seed_id: n_tasks for seed_id, n_tasks in candidates}
    order = [seed_id for seed_id, _ in candidates]
    statute_ok: set[str] | None = None
    if "statute_qa" in quota:
        from tuned.data.generate import seed_meta, statute_qa_section_eligible

        statute_ok = set()
        for seed_id, _ in candidates:
            seed = store.get_seed(seed_id)
            if seed is None:
                continue
            if statute_qa_section_eligible(
                seed.get("text") or "", seed_meta(seed).get("section_text")
            ):
                statute_ok.add(seed_id)

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
                if used[candidate] >= PER_SEED_CAP:
                    continue
                if task_type == "statute_qa" and statute_ok is not None and candidate not in statute_ok:
                    continue
                seed_id = candidate
                break
            if seed_id is None:
                if task_type == "statute_qa":
                    break
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


def reopen_tasks(
    store,
    states: Sequence[str],
    *,
    stream: str | None = None,
    skipped: dict[str, int] | None = None,
) -> dict[str, int]:
    """Return parked tasks to the queue that owns them. Counts per state.

    `skipped` is an optional out-parameter, filled with state -> how many
    rows were left behind because they had no attempt budget left. The
    same idiom as providers.eligible_refs(skipped=...), and it exists
    because a bare 0 is ambiguous: "the filter matched nothing" and "every
    match was out of budget" call for completely different next moves.

    `stream=None` means every stream, which is what the recovery command in
    the config's TODO block actually needs: a pool gap parks whatever was in
    flight, and that is rarely one stream.

    The workers park a row rather than close it whenever the failure is about
    the POOL and not about the answer: no judge left for slot B, no generator
    that can hold the prompt, no key. Parking is only survivable if something
    can un-park it, and until now "re-openable by hand" meant hand-written
    SQL against a live WAL database.

    Nothing here re-pays for work already done: a re-opened row keeps its
    generations and its recorded judgements, and judge.py reuses any slot the
    judgement table already holds (judge_slot_reused). Re-opening a row whose
    judge A answered costs exactly slot B - PROVIDED the re-judge scores the
    same generation, because store.judgements_for is keyed on gen_id. A row
    that judge.py parked in `gen_unroutable` from its regeneration path
    (judge.py's borderline branch) comes back to the GENERATOR, so the next
    pass produces a new gen_id and both slots are bought again. That is
    correct - they would be scoring a different answer - but it is not free.

    THE ATTEMPT BUDGET COMES BACK WITH THE ROW, UNLESS THE PARK ALREADY COST
    A CALL. The park this exists for happens AT the cap: a wave that could not
    route at all burnt three claims finding that out, and every one of them was
    spent on a fact about the fleet rather than on the answer. Restoring the
    state alone hands the row back exhausted, so the first ordinary failure
    afterwards - a 429 - is terminal, lands in `rejected` (which is
    deliberately not re-openable) and is then counted as a legal-quality
    reject. Zeroing `attempts` is the honest reading of "return it to the
    queue" for those.

    IT IS NOT THE HONEST READING FOR A BILLED PARK, and that distinction is
    new (2026-08-18, review round 2). `unroutable:no-reasoning-channel` is
    reached by making the call, paying for it, storing the generation and
    finding no trace in the reply. Zeroing attempts there made every reopen
    cycle buy one more call with NOTHING bounding the loop, and overwriting the
    disposition with `reopened:from-gen_unroutable` erased the only thing that
    distinguished an expensive park from a free one - so a fleet burning a call
    per cycle read exactly like one that had never been reached. Those rows now
    keep their attempts and their disposition, which bounds the loop at
    MAX_ATTEMPTS calls and leaves the cost legible. See
    BILLED_PARK_DISPOSITIONS.

    `rejected` is deliberately not re-openable. It is a decision - the gates
    or the judges said this example is wrong - and re-opening decisions is
    how a rejected row quietly re-enters the dataset.
    """
    unknown = [state for state in states if state not in REOPEN_STATES]
    if unknown:
        raise ValueError(
            f"cannot re-open {unknown}: re-openable states are "
            f"{sorted(REOPEN_STATES)} (a rejected row is a decision, not a park)"
        )
    from tuned.data.generate import MAX_ATTEMPTS

    if skipped is None:
        skipped = {}
    counts: dict[str, int] = {}
    for state in states:
        target = REOPEN_STATES[state]
        clauses = ["state = ?"]
        params: list = [state]
        if stream is not None:
            clauses.append("stream = ?")
            params.append(stream)
        rows = store.conn.execute(
            f"SELECT task_id, disposition, attempts FROM task "
            f"WHERE {' AND '.join(clauses)} ORDER BY rowid",
            params,
        ).fetchall()
        moved = 0
        exhausted = 0
        for task_id, disposition, attempts in rows:
            # WHOSE FAULT WERE THE ATTEMPTS? That is the whole question, and
            # the answer decides whether the budget comes back.
            #
            # The `reopened:` prefix is stripped first so that re-opening a row
            # twice without a claim in between cannot launder a spent park into
            # a free one and hand back the budget.
            settled = (disposition or "").removeprefix("reopened:")
            free = settled in FREE_PARK_DISPOSITIONS or settled.startswith(
                FREE_PARK_PREFIXES
            )
            # A row that has already spent its budget on real calls is not
            # re-openable: the next claim would buy attempt four. This is what
            # bounds the no-reasoning-channel cycle at MAX_ATTEMPTS calls, and
            # it is also what protects the genuinely-exhausted rows that a
            # bulk cancellation can sweep into a parking state.
            if not free and int(attempts or 0) >= MAX_ATTEMPTS:
                exhausted += 1
                continue
            # THE DISPOSITION IS PRESERVED on anything that cost a call.
            # Overwriting it with `reopened:from-<state>` erased the only thing
            # that told a free park from an expensive one, so a fleet burning a
            # call per cycle looked exactly like one never reached.
            new_disposition = (
                f"reopened:from-{state}" if free else f"reopened:{settled}"
            )
            # No fence: a parked row holds no lease (set_task_state released
            # it when it parked), so there is no live holder to lose to.
            if store.set_task_state(
                task_id, target, new_disposition, reset_attempts=free
            ):
                moved += 1
        # N5: a zero that means "all of them were out of budget" must not read
        # the same as a zero that means "the filter matched nothing". The
        # operator's next move differs entirely.
        if exhausted:
            skipped[state] = exhausted
        if moved:
            store.log_event(
                "tasks_reopened",
                {"from_state": state, "to_state": target, "stream": stream, "count": moved},
            )
        counts[state] = moved
    return counts


def parked_by_stream(
    store, states: Sequence[str], *, stream: str | None = None
) -> dict[str, int]:
    """stream -> how many rows are sitting in `states`. READ-only (see 3.).

    The CLI prints this before re-opening so the operator sees which streams
    a pool gap actually caught, and - when a filter is passed - what is being
    left behind.
    """
    clauses = [f"state IN ({', '.join('?' * len(states))})"]
    params: list = list(states)
    if stream is not None:
        clauses.append("stream = ?")
        params.append(stream)
    return {
        row[0]: row[1]
        for row in store.conn.execute(
            f"SELECT stream, COUNT(*) FROM task WHERE {' AND '.join(clauses)} "
            f"GROUP BY stream ORDER BY stream",
            params,
        ).fetchall()
    }


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
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    # No default, so that "was it passed?" is answerable: --stream is honoured
    # by the PLANNER and ignored by the re-open, and a --reopen-only command
    # that names one is asking for a filter it will not get.
    parser.add_argument("--stream", default=None, help=f"planning stream (default {DEFAULT_STREAM})")
    parser.add_argument("--n", type=int, default=None, help="target task count for the queue")
    parser.add_argument("--arm", default=None, help="A/B label, e.g. unscripted|scripted")
    parser.add_argument("--mix", default=None, help="task_type=weight,... (overrides the default)")
    parser.add_argument("--source", action="append", default=None, help="restrict to a source_id")
    parser.add_argument(
        "--reopen",
        action="append",
        default=None,
        metavar="STATE",
        help=(
            "return parked tasks to their queue (repeatable, or 'all'): "
            + ", ".join(sorted(REOPEN_STATES))
        ),
    )
    parser.add_argument(
        "--reopen-stream",
        default=None,
        metavar="STREAM",
        help=(
            "with --reopen, act on this stream only. The default is EVERY "
            "stream: a pool gap parks whatever was in flight, and the "
            "recovery command in the config TODO is written without a stream"
        ),
    )
    args = parser.parse_args(argv)
    if args.n is None and not args.reopen:
        parser.error("nothing to do: pass --n (plan a wave) or --reopen STATE")
    if args.reopen and args.stream is not None and args.n is None:
        # --stream is honoured by the PLANNER and ignored by the re-open, so
        # it is meaningful in this command exactly when this command also
        # plans - i.e. when --n is present. `--reopen X --n 3 --stream
        # transition` re-opens every stream and then plans 3 transition rows,
        # which is a real (and previously working) thing to want.
        #
        # Without --n nothing plans, so the only reading left is "filter the
        # re-open", which it does not do: the operator watched a command that
        # named a stream re-open every one of them. That is the same class of
        # mistake as --reopen defaulting to synthesis (round 3, I6), and it is
        # the only case this refuses.
        parser.error(
            "--stream is the planning stream and does not filter --reopen; "
            "use --reopen-stream to narrow the re-open, or add --n to plan a "
            "wave on this stream in the same command"
        )
    stream = args.stream or DEFAULT_STREAM

    cfg = load_build_config(args.config)
    paths = build_paths(cfg.build.workdir).ensure()
    store = Store.open(paths.state_db)
    try:
        if args.reopen:
            states = (
                sorted(REOPEN_STATES) if "all" in args.reopen else list(dict.fromkeys(args.reopen))
            )
            # Read the per-stream shape BEFORE the move, and read it again
            # unfiltered afterwards: with --reopen-stream the difference is
            # the residue this command deliberately left parked, which is
            # otherwise invisible until the wave comes up short.
            touched = parked_by_stream(store, states, stream=args.reopen_stream)
            out_of_budget: dict[str, int] = {}
            counts = reopen_tasks(
                store, states, stream=args.reopen_stream, skipped=out_of_budget
            )
            # Only a FILTER can leave a residue - an unfiltered re-open moves
            # every row in those states - so only a filtered run looks for
            # one. NOT a bug fix, and it is not claimed as one: with
            # stream=None every row moves, so the unfiltered branch was
            # already returning {} and printed nothing. What it fixes is the
            # LINE, which said "STILL PARKED (not in --reopen-stream None)"
            # whenever it could have printed at all - telling the operator
            # their unfiltered command had a filter.
            residue = (
                parked_by_stream(store, states) if args.reopen_stream is not None else {}
            )
            print(f"re-opened {sum(counts.values())}")
            for state in states:
                # A zero next to a non-zero skip count is a different message
                # from a bare zero: the rows were THERE and had nothing left to
                # spend, so re-running this command will never move them and
                # the operator wants a re-plan, not another --reopen.
                left = out_of_budget.get(state, 0)
                note = f"   ({left} out of attempt budget, not re-opened)" if left else ""
                print(f"  {state} -> {REOPEN_STATES[state]:<12}{counts[state]:>6}{note}")
            for name, count in sorted(touched.items()):
                print(f"  stream {name:<14}{count:>6}")
            if residue:
                left = ", ".join(f"{name}={count}" for name, count in sorted(residue.items()))
                print(
                    f"  STILL PARKED (not in --reopen-stream "
                    f"{args.reopen_stream!r}): {left}"
                )
            print(
                "task states: "
                + ", ".join(f"{k}={v}" for k, v in sorted(store.task_counts().items()))
            )
            if args.n is None:
                return 0
        mix = parse_mix(args.mix) if args.mix else None
        rows = plan_rows(
            store, cfg, stream, args.n, task_type_mix=mix, arm=args.arm, sources=args.source
        )
        created = commit_rows(store, rows, stream=stream, arm=args.arm, target=args.n)
        by_type: dict[str, int] = {}
        for row in rows:
            by_type[row["task_type"]] = by_type.get(row["task_type"], 0) + 1
        print(f"stream={stream} arm={args.arm or '-'} target={args.n}")
        if not rows:
            # "skipped N" reads as "N tasks were dropped"; the truth is that
            # the queue is already at (or past) the target, or every eligible
            # seed is at the per-seed cap.
            existing = _existing_in_queue(store, stream, args.arm)
            reason = (
                "already at target"
                if existing >= args.n
                else "no seeds under the per-seed cap"
            )
            print(f"planned 0  ({reason}: queue holds {existing}, target {args.n})")
        else:
            print(f"planned {created}  collided {len(rows) - created}  (rows derived {len(rows)})")
        for task_type, count in sorted(by_type.items()):
            print(f"  {task_type:<18}{count:>6}")
        counts = store.task_counts()
        print("task states: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
