"""Trim the pre-built streams so the assembled corpus lands on its profile.

The builders (replay.py, curated.py) produce FIXED pools - 4,320 replay
rows and 1,700 curated_c1 rows - sized for the finished v1.0-MVP corpus.
Until generation has produced its full share, feeding those pools whole to
decontaminate guarantees three red stats gates, and not for a reason that
waiting fixes:

  - MIX: replay is 73% of a corpus whose profile wants 42%.
  - EMPTY-THINK: the window is a fixed ~19% OF CORPUS SIZE, so a 4,000-row
    ship has a ~760-row no-think budget while the pools between them hold
    2,452 no-think rows. Both cannot fit. Feeding the target mix with the
    pools' own composition lands empty-think at 34.3%, still red.
  - TRACE: the exact complement of empty-think in this corpus - every row
    either carries _prov.reasoning or opens with the byte-exact empty
    block - so it is the same measurement, not a second problem.

This module sizes the corpus to whatever generation has actually produced
and selects the stream subset that hits both the mix and the empty-think
window. It writes NEW files and never touches the pools, so a later run
with more generated rows re-derives a bigger corpus from the same inputs.

The scarce resource is generated rows: grounded_synthesis is 30.1% of the
profile and can only come from the teacher, so N = generated_synthesis /
0.301 and every other count follows. One generated row buys ~2.05 corpus
rows at the default shape.

WHERE THE NO-THINK BUDGET COMES FROM is the one free choice, exposed as
--replay-nothink-share: the empty budget can be filled from replay's chat
slices (smoltalk_nothink/wildchat_prof/legal_qa_empty), from curated_c1's
raw legal rows (PredEx/aalap), or any blend. It defaults to replay's
as-built share, which preserves what the slice names say the design wants
- no-think trained on chit-chat, not on legal prediction. Lowering it
buys corpus (~2.88 rows per generated row at 0.0) by sourcing the budget
from raw legal rows instead, which is a TRAINING-DATA DESIGN CHANGE and
deliberately not the default.

Run:  python -m tuned.data.shape --config data/configs/data_law_v1.yaml
      [--profile v1.0-MVP] [--replay-nothink-share 0.15]
      [--empty-target 0.19] [--no-retention-correction]

Re-fit the retention table off the last completed chain (reads out/, writes
nothing):
      python -m tuned.data.shape --config ... --measure

Then point decontaminate at the output instead of the pools:
      python -m tuned.data.decontaminate --config ... \
          --in out/shaped_replay.jsonl --in out/shaped_curated_c1.jsonl
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from tuned.data.paths import DEFAULT_CONFIG

MANIFEST_FILENAME = "shape.json"
SHAPED_PREFIX = "shaped_"

# The empty-think gate's window is [0.18, 0.20]; the midpoint leaves 1pp of
# slack each way for rounding and for retention drift.
DEFAULT_EMPTY_TARGET = 0.19

# Rows requested here do not all survive: decontamination, dedupe and the
# over-length drop in assemble each take a cut, and the cut is PER SOURCE.
#
# EVERY FIGURE BELOW IS A READING, and each carries the n it was read from.
# `--measure` computes them off the chain's own artifacts (see
# retention_report). The seven file-based values were read on 2026-08-30 from
# the 2026-08-29 full-chain run in the build out/ dir - a chain that predates
# the drop record's `source` field, so its drops were attributed through
# `form`, which is exact for every file-based source (row_form falls back to
# `source` for a row with no task identity) and is why the generated streams
# were absent from that reading. `--measure` refuses that artifact rather than
# repeating the attribution.
#
# It said it would reproduce those seven numbers off the next completed chain,
# and on 2026-08-31 it did - all seven to three decimals, off a chain run on a
# store snapshot with pools shipped WHOLE (no shape step) rather than shaped.
# That is what qualified the same run's two generated readings below: the
# instrument was checked against seven known answers before it was believed on
# two unknown ones.
#
# The previous table was a set of round numbers that did not reproduce -
# PredEx sat at 0.900 against a measured 0.846, WildChat at 0.955 against
# 0.910 - while the prose beside it ("PredEx loses 15%") was right.
# Re-measure after any change to the decontamination corpora or the dedupe
# thresholds; a stale table shows up as a mix gate that misses by a point or
# two, never as silence.
MEASURED_RETENTION = {
    "open-thoughts/OpenThoughts-114k": 0.996,          # n=3,120  2026-08-30
    "L-NLProc/PredEx_Instruction-Tuning_Pred-Exp": 0.846,  # n=800  2026-08-30
    "HuggingFaceTB/smoltalk2:OpenHermes-2.5": 1.000,   # n=600    2026-08-30
    "opennyaiorg/aalap_instruction_dataset": 0.958,    # n=600    2026-08-30
    "GSMS-B/Indian-Legal-QA-BNS-BNSS-BSA": 0.983,      # n=300    2026-08-30
    "allenai/WildChat-4.8M": 0.910,                    # n=300    2026-08-30
    "169Pi/indian_law": 0.957,                         # n=300    2026-08-30
    # GENERATED rows are the ones this correction exists for: an accepted task
    # is not an assembled row, and sizing the corpus off the accepted COUNT
    # holds the numerator while the chain shrinks the denominator - which is
    # how the first shaped rehearsal shipped grounded_synthesis at 27.7%
    # against a 30.1% target with every stream pool individually on target.
    # Keyed by stream name because that is what a generated row carries as
    # _prov.source.
    #
    # Both were MEASURED on 2026-08-31, off the first chain to ship 50+
    # generated rows. synthesis replaces a 0.857 placeholder that was kept
    # (not deleted) precisely because deleting it fell back to a HIGHER
    # DEFAULT_RETENTION on no evidence - and the reading came in BELOW the
    # guess, so the guess was optimistic too. curated_c2 had no entry at all
    # and was therefore taking 0.95: its real retention is 0.817, so every
    # curated sizing before today was 16% optimistic.
    #
    # The loss is not spread across the chain. Of synthesis's 69 drops and
    # curated_c2's 90, ALL but two are decontamination and the largest single
    # reason is `case_id:iltur` (30 and 58) - generated rows whose seed case
    # is in the IL-TUR eval set. Dedupe took 2 rows and the length cut took 0.
    # So these figures move when the eval corpora move, not when the gates or
    # the templates do; re-measure after a decontamination corpus changes -
    # or after its POLICY does. Running decontaminate with
    # --no-case-id-from-text moves generated drops from 159 to 78 on this same
    # store, which would put both figures well above what is recorded here.
    #
    # READ THE DENOMINATORS. These are CHAIN retentions - shipped over rows
    # ENTERING decontaminate - while generated_counts multiplies them by
    # store.accepted_count(), which is the count BEFORE verify runs. The two
    # agree only where verify demotes nothing.
    #
    # curated_c2: they agree. It has no teacher cut, and the 4 rows its
    # citation-existence half rejects are inside the 491, so 0.817 is both the
    # chain retention and the store-accepted-to-shipped factor.
    #
    # synthesis: they do NOT agree today, and by 19%. 531 rows are accepted in
    # the store; the one-teacher cut demotes 84 of them (retired cerebras and
    # lightning gpt-oss-120b) and only 447 enter the chain. 531 x 0.846 = 449
    # against 378 that actually ship. The gap is a ONE-OFF: those 84 rows are
    # demoted for good by the first assembly run that arms the cut, and every
    # generation since 2026-08-28 is deepseek, so store-accepted converges on
    # entered and 0.846 becomes right. Until then, subtract 84 accepted (~71
    # effective) from any synthesis sizing by hand.
    "synthesis": 0.846,     # n=447 ENTERED  2026-08-31  (was a 0.857 placeholder)
    "curated_c2": 0.817,    # n=491 ENTERED  2026-08-31  (was taking the 0.95 default)
    # `transition` is ABSENT deliberately: it has put 5 rows through the chain,
    # far under RETENTION_MIN_N, so any number here would be invented and the
    # DEFAULT_RETENTION fallback is the honest answer until it has more.
}
DEFAULT_RETENTION = 0.95

# n below which a per-source figure is not reported. This table sizes the
# WHOLE corpus, and a retention read off a handful of rows is one drop away
# from a different number - so `--measure` prints the counts and withholds
# the ratio rather than offering a figure that reads like a measurement.
RETENTION_MIN_N = 50

SYNTHESIS_BUCKET = "grounded_synthesis"
CURATED_BUCKET = "curated"
REPLAY_BUCKET = "replay"


class ShapeError(Exception):
    """The shape cannot be computed - state the reason, write nothing."""


# --------------------------------------------------------------------------
# Row classification. Both facts are already on every row; nothing is
# re-derived from message text, which is what let the earlier per-slice
# accounting drift from what the gate actually measures.
# --------------------------------------------------------------------------

def row_source(row: dict) -> str:
    return ((row.get("_prov") or {}).get("source")) or ""


def is_trace(row: dict) -> bool:
    """Trace vs no-think, by the SAME field trace_count in stats.py reads."""
    return bool((row.get("_prov") or {}).get("reasoning"))


def classify(rows, assembly) -> dict:
    """{(bucket, trace): [rows]} plus a refusal on any unmapped source.

    An unmapped source is fatal here rather than merely red in stats: the
    shape arithmetic apportions by bucket, and a row belonging to no
    bucket would be silently shipped outside every target it was sized by.
    """
    groups: dict[tuple[str, bool], list] = {}
    unmapped: set[str] = set()
    for row in rows:
        source = row_source(row)
        bucket = assembly.stream_of(source)
        if bucket is None:
            unmapped.add(source or "<no _prov.source>")
            continue
        groups.setdefault((bucket, is_trace(row)), []).append(row)
    if unmapped:
        raise ShapeError(
            "these sources map to no assembly bucket, so no target sizes them: "
            + ", ".join(sorted(unmapped))
        )
    return groups


def _pool_retention(pools) -> dict:
    """Per-group retention, weighted by each source's share of its group.

    Retention is measured per SOURCE (PredEx loses 15%, OpenThoughts 0.4%)
    but the plan apportions per (bucket, trace) group, and a group blends
    sources - replay/nothink is smoltalk + legal_qa + wildchat. Weighting
    by the pool's own composition is right precisely because `select`
    keeps that composition when it trims.
    """
    out = {}
    for key, rows in pools.items():
        if not rows:
            continue
        out[key] = sum(
            MEASURED_RETENTION.get(row_source(r), DEFAULT_RETENTION) for r in rows
        ) / len(rows)
    return out


# --------------------------------------------------------------------------
# The plan.
# --------------------------------------------------------------------------

@dataclass
class ShapePlan:
    total: int
    generated_synthesis: int
    generated_curated: int
    demand: dict = field(default_factory=dict)      # (bucket, trace) -> rows to KEEP
    request: dict = field(default_factory=dict)     # (bucket, trace) -> rows to SELECT
    binding: str = ""

    @property
    def shares(self) -> dict:
        out: dict[str, float] = {}
        for (bucket, _), n in self.demand.items():
            out[bucket] = out.get(bucket, 0.0) + n / self.total
        out[SYNTHESIS_BUCKET] = out.get(SYNTHESIS_BUCKET, 0.0) + (
            self.generated_synthesis / self.total
        )
        out[CURATED_BUCKET] = out.get(CURATED_BUCKET, 0.0) + (
            self.generated_curated / self.total
        )
        return out

    @property
    def empty_share(self) -> float:
        empty = sum(n for (_, trace), n in self.demand.items() if not trace)
        return empty / self.total


def _demands(total, targets, *, generated_curated, empty_target, replay_nothink_share):
    """Rows each (bucket, trace) group must supply for a corpus of `total`.

    Negative demands are returned as-is; feasibility is the caller's test,
    so the search below sees "this size is too big" rather than a value
    silently clamped to zero that would report a shape it did not build.
    """
    replay = targets[REPLAY_BUCKET] * total
    curated = targets[CURATED_BUCKET] * total
    replay_nothink = replay_nothink_share * replay
    curated_nothink = empty_target * total - replay_nothink
    return {
        (REPLAY_BUCKET, False): replay_nothink,
        (REPLAY_BUCKET, True): replay - replay_nothink,
        (CURATED_BUCKET, False): curated_nothink,
        # generated curated_c2 rows all carry traces, so they come out of
        # the curated TRACE demand and only the remainder is raw c1.
        (CURATED_BUCKET, True): curated - generated_curated - curated_nothink,
    }


def _feasible(total, targets, *, pools, generated_curated, empty_target,
              replay_nothink_share, retention):
    """(ok, demand, request, reason) for one candidate corpus size."""
    demand = _demands(
        total, targets, generated_curated=generated_curated,
        empty_target=empty_target, replay_nothink_share=replay_nothink_share,
    )
    request = {}
    for key, want in demand.items():
        label = f"{key[0]}/{'trace' if key[1] else 'nothink'}"
        if want < -0.5:
            return False, demand, request, (
                f"{label} would need {round(want)} rows - the generated rows "
                f"already in that bucket overfill it"
            )
        keep = max(0, round(want))
        need = math.ceil(keep / retention.get(key, 1.0)) if keep else 0
        request[key] = need
        have = len(pools.get(key, ()))
        if need > have:
            return False, demand, request, (
                f"{label} needs {need} rows (to keep {keep} after losses) "
                f"but the pool holds {have}"
            )
    return True, demand, request, ""


def plan(pools, *, targets, generated_synthesis, generated_curated,
         empty_target=DEFAULT_EMPTY_TARGET, replay_nothink_share=None,
         retention=None, tolerance_pp=2.0) -> ShapePlan:
    """Largest corpus the pools and the generated rows can green-gate.

    THE SIZE IS NOT A FREE SEARCH. Generated rows cannot be dropped - the
    shaper trims stream files, while decontaminate reads every accepted
    generation out of the store - so shrinking the corpus raises
    grounded_synthesis ABOVE its target instead of lowering it. An earlier
    draft binary-searched down to zero and would have "succeeded" by
    proposing corpora whose synthesis share was double its target.

    So the search starts at the size that hits the target EXACTLY
    (generated / target) and walks down only as far as the mix tolerance
    allows. Down is the only useful direction: a short pool is short
    because the corpus wants more stream rows than exist, and a bigger
    corpus wants more still. Starting at the exact size rather than at the
    largest admissible one keeps the full tolerance as margin, which is
    what an unattended build needs - the alternative parks the gate at its
    edge, where rounding decides the verdict.
    """
    retention = dict(retention if retention is not None else {})
    if replay_nothink_share is None:
        nothink = len(pools.get((REPLAY_BUCKET, False), ()))
        trace = len(pools.get((REPLAY_BUCKET, True), ()))
        replay_nothink_share = nothink / (nothink + trace) if (nothink + trace) else 0.0

    target = targets[SYNTHESIS_BUCKET]
    if generated_synthesis <= 0:
        raise ShapeError(
            "no accepted grounded_synthesis rows: the profile wants "
            f"{target:.1%} of the corpus from the teacher and nothing has "
            "been generated, so there is no corpus to size"
        )

    tol = tolerance_pp / 100.0
    hi = round(generated_synthesis / target)
    lo = math.ceil(generated_synthesis / (target + tol))
    args = dict(
        targets=targets, pools=pools, generated_curated=generated_curated,
        empty_target=empty_target, replay_nothink_share=replay_nothink_share,
        retention=retention,
    )
    first_reason = ""
    for total in range(hi, lo - 1, -1):
        ok, demand, request, reason = _feasible(total, **args)
        if ok:
            return ShapePlan(
                total, generated_synthesis, generated_curated,
                demand={k: max(0, round(v)) for k, v in demand.items()},
                request=request,
                binding=(SYNTHESIS_BUCKET if total == hi else "pool"),
            )
        first_reason = first_reason or reason
    raise ShapeError(
        f"no corpus size between {lo} and {hi} rows works. At the largest, "
        f"{first_reason}. Either generate more grounded_synthesis (every row "
        f"buys ~{1/target:.1f} corpus rows), rebuild the short stream larger, "
        "or move the no-think budget with --replay-nothink-share."
    )


# --------------------------------------------------------------------------
# Selection.
# --------------------------------------------------------------------------

def _order_key(row: dict) -> str:
    prov = row.get("_prov") or {}
    raw = f"{prov.get('source')}|{prov.get('native_id')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def select(rows, n: int) -> list:
    """`n` rows, deterministic, preserving the per-source proportions.

    Proportional by largest remainder rather than a flat hash cut: the
    sub-mix inside a group is designed (600 smoltalk : 300 legal : 300
    wildchat), and a flat cut would let one source drift out of the shape
    the builder chose. Ordering is a stable content hash, so a re-run with
    a bigger corpus is a SUPERSET of the smaller one - a row never leaves
    the dataset because the corpus grew.
    """
    if n >= len(rows):
        return list(rows)
    if n <= 0:
        return []
    by_source: dict[str, list] = {}
    for row in rows:
        by_source.setdefault(row_source(row), []).append(row)

    exact = {s: n * len(g) / len(rows) for s, g in by_source.items()}
    take = {s: int(v) for s, v in exact.items()}
    remainder = n - sum(take.values())
    for source in sorted(by_source, key=lambda s: (-(exact[s] - take[s]), s)):
        if remainder <= 0:
            break
        if take[source] < len(by_source[source]):
            take[source] += 1
            remainder -= 1

    out = []
    for source in sorted(by_source):
        group = sorted(by_source[source], key=_order_key)
        out.extend(group[: take[source]])
    return out


def shape_streams(stream_rows: dict, plan_: ShapePlan, assembly) -> dict:
    """{stream_name: [selected rows]} for the plan's per-group requests."""
    chosen: dict[tuple[str, bool], list] = {}
    for key, want in plan_.request.items():
        pool = []
        for rows in stream_rows.values():
            pool.extend(
                r for r in rows
                if assembly.stream_of(row_source(r)) == key[0] and is_trace(r) == key[1]
            )
        chosen[key] = select(pool, want)

    keep = {id(r) for rows in chosen.values() for r in rows}
    return {
        name: [r for r in rows if id(r) in keep]
        for name, rows in stream_rows.items()
    }


# --------------------------------------------------------------------------
# The measurement the table above is made of.
# --------------------------------------------------------------------------

def retention_report(out_dir) -> dict:
    """Measured retention per `_prov.source`: shipped rows / rows that entered.

    READ OFF THE CHAIN'S OWN ARTIFACTS, never off the stream files. The chain
    may have been fed shaped files or a hand-picked subset, and generated rows
    come from the store rather than from any file in streams/ - so the stream
    pools are not what entered. `decontaminated.jsonl` plus
    `decontamination_drops.jsonl` IS what entered (kept + dropped == total, an
    identity decontamination.json states in its own counts), and
    law_v1_train + law_v1_eval is what shipped. Everything in between - dedupe,
    the per-case cap, split, the over-length drop in assemble - falls out of
    the difference, so a stage added later cannot be forgotten here.

    Returns {source: {"entered", "shipped", "retention", "reportable"}}.
    `reportable` is n >= RETENTION_MIN_N; the caller prints the counts for the
    rest WITHOUT a ratio, because a retention fitted on fifteen rows is not a
    measurement and this table sizes the whole corpus.
    """
    from tuned.data.assemble import EVAL_FILENAME, TRAIN_FILENAME
    from tuned.data.decontaminate import DROPS_FILENAME, OUT_FILENAME
    from tuned.data.jsonl import read_jsonl

    out_dir = Path(out_dir)
    entered: dict[str, int] = {}
    shipped: dict[str, int] = {}

    def tally(into, path, source_of):
        if not path.is_file():
            raise ShapeError(
                f"{path} is missing - retention is measured off a COMPLETED chain "
                f"(decontaminate through assemble), and a partial one would report "
                f"losses that are really stages that have not run yet"
            )
        for row in read_jsonl(path):
            key = source_of(row)
            into[key] = into.get(key, 0) + 1

    tally(entered, out_dir / OUT_FILENAME, row_source)
    # The drop record carries `source` beside `form` for exactly this: `form`
    # prefers a row's task_type, so a generated row's drop files under
    # `irac_analysis` while the row itself ships as `synthesis`.
    #
    # A drops file written before that field existed is REFUSED rather than
    # read through `form`. The fallback would be exact for every file-based
    # source and silently wrong for the generated streams - which are the ones
    # this correction exists for, and the ones whose figure nobody would think
    # to doubt. Reporting 1.000 for a source that lost 15% of its rows is how
    # the previous table survived as long as it did.
    def drop_source(record):
        if "source" not in record:
            raise ShapeError(
                f"{out_dir / DROPS_FILENAME} predates the drop record's `source` "
                f"field, so a drop cannot be attributed to the source that "
                f"shipped it. Reading it through `form` instead would be exact "
                f"for the file-based sources and silently wrong for the "
                f"generated streams. Re-run the chain from decontaminate."
            )
        return record.get("source") or ""

    tally(entered, out_dir / DROPS_FILENAME, drop_source)
    tally(shipped, out_dir / TRAIN_FILENAME, row_source)
    tally(shipped, out_dir / EVAL_FILENAME, row_source)

    report = {}
    for source in sorted(set(entered) | set(shipped)):
        n = entered.get(source, 0)
        kept = shipped.get(source, 0)
        report[source] = {
            "entered": n,
            "shipped": kept,
            "retention": round(kept / n, 3) if n else None,
            "reportable": n >= RETENTION_MIN_N,
        }
    return report


def print_retention(report: dict) -> None:
    print(f"{'source':<52}{'entered':>9}{'shipped':>9}{'retention':>11}")
    for source, row in sorted(report.items()):
        figure = (
            f"{row['retention']:.3f}" if row["reportable"]
            else f"n<{RETENTION_MIN_N}"
        )
        print(f"{source or '<no source>':<52}{row['entered']:>9}"
              f"{row['shipped']:>9}{figure:>11}")
    print(
        "  a withheld figure is not a zero: the source shipped too few rows for "
        "a ratio to mean anything. MEASURED_RETENTION sizes the whole corpus."
    )


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def generated_counts(store, assembly, *, correct=True) -> tuple[dict, dict]:
    """(effective, accepted) generated rows per bucket.

    AN ACCEPTED TASK IS NOT AN ASSEMBLED ROW. Decontamination, dedupe and
    the over-length cut take 14-19% of the generations, so the count that
    sizes the corpus is the accepted count TIMES its stream's retention.
    Sizing off the raw count instead holds the generated numerator fixed
    while the chain shrinks the denominator, which lands every share of
    the mix wrong at once - the failure mode the first shaped rehearsal
    hit, with all four stream pools individually on target.

    Generated rows reach decontaminate with _prov.source set to the task's
    STREAM, so one retention table serves both them and the stream files.

    The candidate stream names come from the assembly mapping rather than
    from the task table: a generation stream that is not mapped there
    cannot be assembled at all, so counting one here would size a corpus
    around rows that never arrive. Source-id keys simply count zero.
    """
    effective = {SYNTHESIS_BUCKET: 0.0, CURATED_BUCKET: 0.0}
    accepted = {SYNTHESIS_BUCKET: 0, CURATED_BUCKET: 0}
    for stream, bucket in assembly.source_streams.items():
        if bucket not in effective:
            continue
        count = store.accepted_count(stream)
        if not count:
            continue
        keep = MEASURED_RETENTION.get(stream, DEFAULT_RETENTION) if correct else 1.0
        accepted[bucket] += count
        effective[bucket] += count * keep
    return {k: round(v) for k, v in effective.items()}, accepted


def synthesis_band(pools, gc, *, targets, empty_target=DEFAULT_EMPTY_TARGET,
                   replay_nothink_share=None, retention=None, tolerance_pp=2.0,
                   step=25) -> tuple[int, int] | None:
    """The generated-synthesis counts that work at this generated-curated count.

    Refusal comes from BOTH sides - too few and the curated bucket overfills
    its share, too many and the corpus outgrows the shortest pool - so the
    answer is an interval, and how wide it is matters as much as where it is.
    A narrow one means the next wave has to land on a target it can miss.
    """
    feasible = []
    probe = max(step, gc // 100)
    for gs in range(probe, 3 * gc + 2000, probe):
        try:
            plan(pools, targets=targets, generated_synthesis=gs, generated_curated=gc,
                 empty_target=empty_target, replay_nothink_share=replay_nothink_share,
                 retention=retention, tolerance_pp=tolerance_pp)
            feasible.append(gs)
        except ShapeError:
            continue
    return (feasible[0], feasible[-1]) if feasible else None


def curated_ceiling(pools, *, targets, empty_target=DEFAULT_EMPTY_TARGET,
                    replay_nothink_share=None, retention=None, tolerance_pp=2.0,
                    max_curated=20000, step=25) -> int | None:
    """Largest generated-curated count for which SOME synthesis count works.

    THIS IS AN IRREVERSIBILITY BOUNDARY, not a tuning hint. `plan` refuses
    from both sides: too few generated synthesis rows and the curated bucket
    OVERFILLS its share, too many and the corpus outgrows the shortest stream
    pool. Between those the feasible synthesis window narrows as the generated
    curated count rises, and above this value it closes completely - no
    generation, of any stream, in any quantity, reopens it.

    And it cannot be walked back. The shaper trims STREAM files; decontaminate
    reads every accepted generation out of the store, so a generated row is in
    the corpus by existing. A build that crosses this line has produced a
    corpus that can never be assembled at this profile, and the only remedies
    left are rebuilding the short pool larger or changing the profile.

    Feasibility in the generated-curated count is treated as monotone - true
    for the squeeze described above - so this binary-searches it. The inner
    synthesis sweep is COARSE, and deliberately: it can understate the ceiling
    where the surviving window is narrower than one probe. Understating is the
    safe direction for a headroom report - it warns early rather than late -
    and it is the reason the sweep is not refined.
    """
    def works(gc: int) -> bool:
        # The feasible window sits a little above gc and scales with it; the
        # observed ratios run ~1.4x to ~2.7x, so this brackets it with room.
        # The probe scales too, which keeps this ~300 evaluations whatever the
        # magnitude - a fixed step is O(gc) per probe and turns the search into
        # tens of thousands of plans for no extra precision where it matters.
        probe = max(step, gc // 100)
        for gs in range(probe, 3 * gc + 2000, probe):
            try:
                plan(pools, targets=targets, generated_synthesis=gs,
                     generated_curated=gc, empty_target=empty_target,
                     replay_nothink_share=replay_nothink_share,
                     retention=retention, tolerance_pp=tolerance_pp)
                return True
            except ShapeError:
                continue
        return False

    if not works(step):
        return None
    # GALLOP, then bisect. Probing max_curated first would be the obvious
    # bracket and it is the expensive one: a probe costs O(gc) plans, so the
    # single largest probe dominates everything else the search does. Doubling
    # up from the bottom reaches the bracket in log2 probes that are each
    # cheaper than the one before it.
    lo = step
    hi = 2 * step
    while hi < max_curated and works(hi):
        lo, hi = hi, 2 * hi
    if hi >= max_curated:
        return max_curated if works(max_curated) else _bisect_ceiling(works, lo, max_curated, step)
    return _bisect_ceiling(works, lo, hi, step)


def _bisect_ceiling(works, lo: int, hi: int, step: int) -> int:
    """Largest feasible value in [lo, hi), given works(lo) and not works(hi)."""
    while lo + step < hi:
        mid = (lo + hi) // 2
        if works(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _print_headroom(pools, *, targets, profile, gen_synth, gen_curated, accepted,
                    empty_target, replay_nothink_share, tolerance_pp, retention) -> None:
    """How much generated-curated room is left before the corpus is stuck.

    Written because the number that matters is not in any report: `shape`
    says yes or no about TODAY's counts, and the queue keeps generating. The
    question an operator actually has - "how many more curated rows may this
    build produce before it can never be assembled" - had no command.
    """
    kw = dict(targets=targets, empty_target=empty_target,
              replay_nothink_share=replay_nothink_share,
              retention=retention, tolerance_pp=tolerance_pp)
    ceiling = curated_ceiling(pools, **kw)
    print(f"generated-curated headroom - profile {profile}")
    print(f"  pools                 {({f'{k[0]}/{"trace" if k[1] else "nothink"}': len(v) for k, v in sorted(pools.items())})}")
    raw_curated = accepted.get(CURATED_BUCKET, 0)
    print(f"  now                   {gen_curated} effective ({raw_curated} accepted), "
          f"synthesis {gen_synth} effective")
    if ceiling is None:
        print("  ceiling               NONE - these pools admit no corpus at this "
              "profile at any generated count. This is a POOL problem, not a "
              "generation one: rebuild the short stream or change the profile.")
        return
    print(f"  ceiling               {ceiling} effective generated-curated rows")
    left = ceiling - gen_curated
    # Effective counts are retention-corrected, so convert back before quoting
    # a number anyone can act on - the operator throttles ACCEPTED rows.
    factor = (gen_curated / raw_curated) if raw_curated else 1.0
    room = int(left / factor) if factor else left
    if left <= 0:
        print("  headroom              NONE - ALREADY PAST IT. No synthesis count "
              "assembles this corpus at this profile, and generated rows cannot "
              "be dropped.")
    else:
        print(f"  headroom              {left} effective (~{room} more accepted "
              f"curated rows)")
    band = synthesis_band(pools, gen_curated, **kw)
    print(f"  synthesis needed now  {f'{band[0]}..{band[1]} effective' if band else 'NONE - this curated count is already unshapeable'}")
    at_ceiling = synthesis_band(pools, ceiling, **kw)
    if at_ceiling:
        print(f"  ...and at the ceiling {at_ceiling[0]}..{at_ceiling[1]} effective "
              f"(window {at_ceiling[1] - at_ceiling[0]})")
    print()
    print("  Generated rows cannot be dropped: the shaper trims stream files, "
          "decontaminate reads every accepted generation. Past the ceiling the "
          "corpus is unassemblable at this profile PERMANENTLY. The remedy that "
          "keeps the work is rebuilding the binding pool larger, not generating "
          "less - see the REFUSED message, which names it.")


def main(argv=None) -> int:
    import argparse

    from tuned.data.config import load_build_config
    from tuned.data.jsonl import read_jsonl, write_jsonl
    from tuned.data.paths import build_paths
    from tuned.data.store import Store

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--profile", default="v1.0-MVP")
    p.add_argument("--in", dest="inputs", action="append", default=None,
                   help="stream JSONL to shape (repeatable; default every *.jsonl "
                        "in the streams dir)")
    p.add_argument("--out-dir", default=None, help="default the build out/ dir")
    p.add_argument("--empty-target", type=float, default=DEFAULT_EMPTY_TARGET,
                   help=f"no-think share to aim for (default {DEFAULT_EMPTY_TARGET})")
    p.add_argument("--replay-nothink-share", type=float, default=None,
                   help="share of the replay stream that is no-think; default "
                        "keeps the pool's as-built composition. Lowering it "
                        "sources the no-think budget from raw legal rows "
                        "instead of chat - a design change, not a knob")
    p.add_argument("--no-retention-correction", action="store_true",
                   help="request exactly the target counts, ignoring the measured "
                        "per-source losses in decontaminate/dedupe/assemble")
    p.add_argument("--headroom", action="store_true",
                   help="print how many more GENERATED CURATED rows the corpus can "
                        "still absorb before it becomes unassemblable at this "
                        "profile, and exit; reads the store, writes nothing")
    p.add_argument("--measure", action="store_true",
                   help="print measured per-source retention off the last completed "
                        "chain in out/ and exit; shapes nothing and writes nothing")
    args = p.parse_args(argv)

    cfg = load_build_config(args.config)
    paths = build_paths(cfg.build.workdir).ensure()
    out_dir = Path(args.out_dir) if args.out_dir else paths.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.measure:
        # Early, and before anything is read or written: this is the
        # instrument the MEASURED_RETENTION table is made of, and it must be
        # runnable against a build whose streams have moved on since.
        try:
            print_retention(retention_report(out_dir))
        except ShapeError as exc:
            print(f"REFUSED: {exc}")
            return 2
        return 0

    targets = cfg.assembly.targets(args.profile)
    inputs = ([Path(x) for x in args.inputs] if args.inputs
              else sorted(paths.streams_dir.glob("*.jsonl")))
    if not inputs:
        print(f"REFUSED: no stream files under {paths.streams_dir}")
        return 2
    stream_rows = {path.stem: list(read_jsonl(path)) for path in inputs}

    store = Store.open(paths.state_db)
    try:
        effective, accepted = generated_counts(
            store, cfg.assembly, correct=not args.no_retention_correction
        )
    finally:
        store.close()
    gen_synth, gen_curated = effective[SYNTHESIS_BUCKET], effective[CURATED_BUCKET]

    all_rows = [r for rows in stream_rows.values() for r in rows]
    try:
        pools = classify(all_rows, cfg.assembly)
        if args.headroom:
            _print_headroom(
                pools, targets=targets, profile=args.profile,
                gen_synth=gen_synth, gen_curated=gen_curated, accepted=accepted,
                empty_target=args.empty_target,
                replay_nothink_share=args.replay_nothink_share,
                tolerance_pp=cfg.assembly.gates.mix_tolerance_pp,
                retention=({} if args.no_retention_correction
                           else _pool_retention(pools)),
            )
            return 0
        plan_ = plan(
            pools, targets=targets,
            generated_synthesis=gen_synth, generated_curated=gen_curated,
            empty_target=args.empty_target,
            replay_nothink_share=args.replay_nothink_share,
            tolerance_pp=cfg.assembly.gates.mix_tolerance_pp,
            retention=({} if args.no_retention_correction
                       else _pool_retention(pools)),
        )
    except ShapeError as exc:
        print(f"REFUSED: {exc}")
        return 2

    shaped = shape_streams(stream_rows, plan_, cfg.assembly)

    print(f"profile {args.profile}: {targets}")
    print(f"generated: {accepted[SYNTHESIS_BUCKET]} grounded_synthesis, "
          f"{accepted[CURATED_BUCKET]} curated accepted "
          f"-> {gen_synth}/{gen_curated} expected to survive the chain")
    print(f"corpus sized to {plan_.total} rows (binding: {plan_.binding})")
    print(f"{'group':<28}{'pool':>8}{'select':>8}{'keep':>8}")
    for key in sorted(plan_.request, key=lambda k: (k[0], not k[1])):
        label = f"{key[0]}/{'trace' if key[1] else 'nothink'}"
        print(f"{label:<28}{len(pools.get(key, ())):>8}"
              f"{plan_.request[key]:>8}{plan_.demand[key]:>8}")
    print("  projected shares: "
          + ", ".join(f"{b} {s:.1%}" for b, s in sorted(plan_.shares.items())))
    print(f"  projected no-think: {plan_.empty_share:.1%}")

    written = {}
    for name, rows in shaped.items():
        out_path = out_dir / f"{SHAPED_PREFIX}{name}.jsonl"
        write_jsonl(out_path, rows)
        written[str(out_path)] = len(rows)
        print(f"wrote {len(rows)} of {len(stream_rows[name])} rows -> {out_path}")

    manifest = {
        "profile": args.profile,
        "targets": targets,
        "total": plan_.total,
        "binding": plan_.binding,
        "empty_target": args.empty_target,
        "replay_nothink_share": args.replay_nothink_share,
        "retention_correction": not args.no_retention_correction,
        "generated_accepted": accepted,
        "generated_effective": {SYNTHESIS_BUCKET: gen_synth,
                                CURATED_BUCKET: gen_curated},
        "groups": [
            {"bucket": k[0], "trace": k[1], "pool": len(pools.get(k, ())),
             "select": plan_.request[k], "keep": plan_.demand[k]}
            for k in sorted(plan_.request, key=lambda k: (k[0], not k[1]))
        ],
        "projected_shares": plan_.shares,
        "projected_nothink": plan_.empty_share,
        "outputs": written,
    }
    manifest_path = out_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"      manifest -> {manifest_path}")
    print("  feed these to decontaminate with --in, NOT the pools in streams/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
