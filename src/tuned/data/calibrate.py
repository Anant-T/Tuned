"""P5 - fitting judge thresholds to HUMAN gold labels.

GOLD LABELS ARE HUMAN-ONLY, AND THE RULE IS ABSOLUTE. This module renders a
file for the operator to fill in and ingests it afterwards; it never produces
a label, never asks a model for one, and imports nothing that could. A gold
label written by a model would calibrate the judges against a judge, and the
precision that came out would measure agreement rather than accuracy - which
is exactly the number the P5 gate exists to refuse. A test parses this
module's own imports and fails if a provider seam ever appears in it.

THE FLOW

    export_pilot   pick `calibration.pilot_export` judged generations,
                   stratified, deterministically - the same store gives the
                   same 180 rows on any machine.
    render_todo    write gold_todo.md: one readable block per generation, a
                   worked example at the top, and stable BEGIN/END markers so
                   the operator edits values and never structure.
    ingest         parse the completed file, assign folds, write gold_label.
    calibrate      fit each judge model over {thresholds} x {rules}, choose by
                   precision subject to recall >= min_recall, cross-validate
                   the choice, then measure the UNTOUCHED holdout.
    report         calibration_report.md, with kappa (inter-judge) and phi
                   (per-judge against gold) MEASURED, never estimated.

WHAT THE FOLDS ARE FOR, precisely. The candidate set is fixed, so pooling five
validation folds would give exactly the full-set number and the cross
validation would be decoration. What the folds actually buy here is SELECTION
STABILITY: the best candidate is chosen five times, each time on four fifths
of the labels, and `selection_agreement` reports how many of those five
choices agree with the one that ships. A rule that wins on 4/5 folds is a
rule; a rule that wins on 2/5 is a coin toss the holdout is about to expose.
The per-fold precision/recall of the shipped candidate are reported next to it
so the spread is visible rather than averaged away.

THE HOLDOUT IS TOUCHED ONCE. Fold `calibration.folds` (5, one past the CV
folds 0-4) never enters a selection. Its precision is the P5 gate: below
`calibration.min_precision` the judge is DISQUALIFIED and the next model in
routing.judge that is not already calibrated is named as its replacement -
but ONLY for a model that holds a routing.judge seat, because a replacement is
that seat's succession and `fits` also covers tiebreak-only models. The
swap is named, not performed - routing lives in the YAML, and a module that
edited it would be moving a fence nobody had read.

DISQUALIFICATION IS NOT A CRASH. A judge that fails the gate is recorded,
reported and left out of the active thresholds, and the run still writes the
calibration for the judges that passed. A build with one working judge and a
named swap is a build the operator can act on; an exception is a build with
nothing.

Run:  python -m tuned.data.calibrate --config data/configs/data_law_v1.yaml --export
      python -m tuned.data.calibrate --config data/configs/data_law_v1.yaml --ingest
      python -m tuned.data.calibrate --config data/configs/data_law_v1.yaml --fit
"""

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from tuned.data.config import CALIBRATION_RULES, JUDGE_SCORE_RANGE
from tuned.data.store import utcnow
from tuned.data.paths import DEFAULT_CONFIG

# Re-exported so calibrate.py and config.py cannot drift apart on the
# vocabulary: config validates `calibration.rules` against this tuple and this
# module fits exactly these three.
RULES = CALIBRATION_RULES

# The only two gold verdicts. A third ("borderline") is deliberately absent:
# precision and recall are defined against a binary truth, and a gold label
# the gate cannot use is a labelling hour spent for nothing. The operator's
# instruction says which way to round a borderline row and why.
ACCEPT, REJECT = "accept", "reject"
VERDICTS = (ACCEPT, REJECT)

# The three axes every judge scores, in the order the judgement table holds
# them. judge.JudgeScores has the same three; a test pins them together.
AXES = ("grounding", "validity", "coverage")

# Markers the operator's file is parsed on. HTML comments so the file still
# renders as markdown, and they carry the gen_id so a block that gets moved,
# duplicated or partially deleted is still attributable - a positional parse
# would silently re-attribute every label after the damage.
BEGIN = "<!-- gold:BEGIN {gid} -->"
END = "<!-- gold:END {gid} -->"
_BLOCK_RE = re.compile(
    r"<!--\s*gold:BEGIN\s+(?P<gid>\S+)\s*-->(?P<body>.*?)<!--\s*gold:END\s+(?P=gid)\s*-->",
    re.DOTALL,
)
# The worked example at the top of the file. Filled in, and skipped on ingest
# by this id: an operator who labels by copying the block above needs one to
# copy, and it must never enter the fit.
EXAMPLE_ID = "EXAMPLE"

# How much of the grounding text a block carries. The operator is scoring
# grounding faithfulness and needs the source, but a 180-block file with whole
# judgments in it is a file nobody opens twice. Blocks say when they truncated
# and name the seed, so the full text is one query away.
SOURCE_EXCERPT_CHARS = 4000

GOLD_TODO_NAME = "gold_todo.md"
CALIBRATION_REPORT_NAME = "calibration_report.md"


# --------------------------------------------------------------------------
# Deterministic stratified selection.
# --------------------------------------------------------------------------

def _rank(key: str) -> str:
    return hashlib.sha256(str(key).encode("utf-8")).hexdigest()


def _buckets(rows: Sequence[dict], *, strata, key) -> dict:
    out: dict = {}
    for row in rows:
        out.setdefault(strata(row), []).append(row)
    for bucket in out.values():
        bucket.sort(key=lambda row: _rank(key(row)))
    return out


def stratified_take(rows: Sequence[dict], n: int, *, strata, key) -> list[dict]:
    """`n` rows, spread as EVENLY over the strata as they allow.

    Round-robin, so a rare stratum is over-sampled relative to its share. That
    is what the pilot export wants and it is the opposite of what the folds
    want: the export exists to put the judges' DISAGREEMENTS in front of the
    operator, and a draw proportional to the population would be dominated by
    the rows both judges passed - a threshold fitted on those has never met
    the disagreement it is there to resolve. proportional_take is the other
    one; see assign_folds.

    No RNG anywhere: the same store must ask the operator to label the same
    180 rows however many times the export is run, or a re-export after a
    crash quietly asks for a different 6-9 hours.
    """
    buckets = _buckets(rows, strata=strata, key=key)
    order = sorted(buckets)
    out: list[dict] = []
    depth = max((len(b) for b in buckets.values()), default=0)
    for i in range(depth):
        for name in order:
            bucket = buckets[name]
            if i < len(bucket):
                out.append(bucket[i])
                if len(out) == n:
                    return out
    return out


def _largest_remainder(sizes: dict, n: int) -> dict:
    """Split `n` across strata in proportion to their sizes, exactly.

    Same largest-remainder allocation tasks.allocate uses for task types, and
    tied remainders break on the stratum NAME for the same reason: two runs
    over dicts built in a different order must allocate identically.
    """
    total = sum(sizes.values())
    if total <= 0 or n <= 0:
        return {name: 0 for name in sizes}
    exact = {name: n * size / total for name, size in sizes.items()}
    counts = {name: min(int(value), sizes[name]) for name, value in exact.items()}
    order = sorted(exact, key=lambda name: (-(exact[name] - counts[name]), name))
    i = 0
    while sum(counts.values()) < n and i < len(order) * n:
        name = order[i % len(order)]
        if counts[name] < sizes[name]:
            counts[name] += 1
        i += 1
    return counts


def proportional_take(rows: Sequence[dict], n: int, *, strata, key) -> list[dict]:
    """`n` rows whose stratum SHARES match the population's.

    What "stratification preserved across folds" means: a holdout drawn
    round-robin over accept/reject strata comes back at 50/50 whatever the
    labelled set looks like, and a holdout precision measured on a 50/50
    sample is not a precision anybody can read against a 67/33 corpus.
    """
    buckets = _buckets(rows, strata=strata, key=key)
    quota = _largest_remainder({name: len(bucket) for name, bucket in buckets.items()}, n)
    out: list[dict] = []
    for name in sorted(buckets):
        out.extend(buckets[name][: quota[name]])
    return out


def export_pilot(store, cfg, *, streams: Sequence[str] | None = None) -> list[dict]:
    """The generations the operator is asked to label.

    Stratified by (stream, task_type, the judges' own verdict pattern). The
    third one matters most and is the reason this is not a random draw: a
    sample drawn without it is dominated by the rows both judges passed, and a
    threshold fitted on those has never seen the disagreement it exists to
    resolve. Rows nobody judged are not in the population at all - there is
    nothing to calibrate against them.
    """
    _require_block(cfg)
    rows = store.judged_generations(streams)
    judgements = store.judgements_by_gen([row["gen_id"] for row in rows])
    for row in rows:
        row["judgements"] = judgements.get(row["gen_id"], [])
    return stratified_take(
        rows,
        cfg.calibration.pilot_export,
        strata=lambda row: (
            row.get("stream") or "",
            row.get("task_type") or "",
            _verdict_pattern(row["judgements"]),
        ),
        key=lambda row: row["gen_id"],
    )


def _verdict_pattern(judgements: Sequence[dict]) -> str:
    """A coarse label for what the judges did, used only for stratification.

    Deliberately coarse and computed from the SHIPPED thresholds, because
    fitted ones do not exist yet when the export runs - this is a sampling
    frame, not a decision.
    """
    axes = [_axes_of(row) for row in judgements]
    axes = [a for a in axes if a is not None]
    if not axes:
        return "unscored"
    mins = [min(a) for a in axes]
    if all(value >= 4 for value in mins):
        return "both-pass"
    if all(value <= 2 for value in mins):
        return "both-fail"
    return "split"


def _axes_of(row) -> tuple[int, int, int] | None:
    values = [row.get(axis) for axis in AXES]
    if any(value is None for value in values):
        return None
    return tuple(int(value) for value in values)


# --------------------------------------------------------------------------
# Rendering the operator's file.
# --------------------------------------------------------------------------

_INSTRUCTIONS = """\
Every label in this file is written by a person. Nothing in this pipeline may
fill one in, and nothing does: a gold label produced by a model would be
calibrating the judges against a judge, and the precision that came out would
measure agreement rather than accuracy.

Fill in the five fields inside each `gold:BEGIN`/`gold:END` block and change
nothing else - the markers carry the generation id, so blocks may be reordered
or labelled in any sitting, and a partially completed file can be ingested and
completed later.

  verdict    accept | reject. The question is NOT "is this well written" but
             "would I be content to see this in the dataset as a correct piece
             of Indian legal reasoning".
             A ROW YOU WOULD ARGUE ABOUT IS A REJECT. The threshold being
             fitted decides what ships unread, so the benefit of the doubt
             belongs to the reader of the dataset and not to the teacher.
  grounding  1-5. Is every fact and every citation in the answer supported by
             the materials shown above it?
  validity   1-5. Does the reasoning actually carry the conclusion it reaches?
  coverage   1-5. Does it deal with the question that was asked, whole?
  notes      free text, optional. One line on WHY, for the rows you rejected.

The three scores are what makes a disagreement diagnosable afterwards; the
verdict is what the thresholds are fitted against.
"""


def _excerpt(text: str | None, limit: int = SOURCE_EXCERPT_CHARS) -> str:
    body = (text or "").strip()
    if len(body) <= limit:
        return body or "(none recorded)"
    return f"{body[:limit]}\n\n[... truncated, {len(body) - limit} more characters ...]"


def _label_block(gid, *, verdict="", grounding="", validity="", coverage="", notes="") -> str:
    lines = [BEGIN.format(gid=gid), f"verdict: {verdict}"]
    lines += [f"{axis}: {value}" for axis, value in zip(AXES, (grounding, validity, coverage))]
    lines += [f"notes: {notes}", END.format(gid=gid)]
    return "\n".join(lines)


def render_gold_todo(rows: Sequence[dict], *, seeds: dict | None = None) -> str:
    """The operator's file. `seeds` maps seed_id -> seed row, for the source."""
    seeds = seeds or {}
    out = [
        f"# gold_todo.md - {len(rows)} pilot generations for human labelling",
        "",
        _INSTRUCTIONS,
        "",
        "## How to fill a block in (worked example - NOT a row to label)",
        "",
        "```",
        _label_block(
            EXAMPLE_ID,
            verdict=REJECT,
            grounding=2,
            validity=3,
            coverage=4,
            notes="names a section that is nowhere in the materials",
        ),
        "```",
        "",
    ]
    for i, row in enumerate(rows, start=1):
        seed = seeds.get(row.get("seed_id")) or {}
        judged = ", ".join(
            f"{j.get('judge_slot')}={'/'.join(str(j.get(axis)) for axis in AXES)}"
            for j in row.get("judgements", [])
        )
        out += [
            "---",
            "",
            f"## {i} of {len(rows)} - generation {row['gen_id']}",
            "",
            f"- stream: {row.get('stream')}   task type: {row.get('task_type')}"
            f"   prompt: {row.get('prompt_id')}",
            f"- judges (grounding/validity/coverage): {judged or 'none recorded'}",
            "",
            "### Materials the teacher was shown",
            "",
            _excerpt(seed.get("text")),
            "",
            "### Reasoning trace",
            "",
            _excerpt(row.get("think")),
            "",
            "### Answer",
            "",
            _excerpt(row.get("answer")),
            "",
            "### Your label",
            "",
            _label_block(row["gen_id"]),
            "",
        ]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# Parsing it back.
# --------------------------------------------------------------------------

class GoldParseError(ValueError):
    """The completed file could not be read as labels."""


def _parse_axis(raw: str, *, field_name: str, gid: str) -> int | None:
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise GoldParseError(
            f"block {gid}: {field_name} must be a whole score, got {raw!r}"
        ) from None
    low, high = JUDGE_SCORE_RANGE
    if not (low <= value <= high):
        raise GoldParseError(
            f"block {gid}: {field_name} {value} is outside the judge score range {low}-{high}"
        )
    return value


def parse_gold_todo(text: str) -> tuple[list[dict], list[str]]:
    """(labels, ids the operator has not filled in yet).

    An unlabelled block is NOT an error: the plan splits the labelling into
    two sittings of 60 and 120, so a half-finished file is the normal state
    and ingesting it has to be safe. A block with scores but no verdict is
    also unlabelled - the verdict is what the fit reads.
    """
    labels: list[dict] = []
    pending: list[str] = []
    seen: set[str] = set()
    for match in _BLOCK_RE.finditer(text or ""):
        gid = match.group("gid")
        if gid == EXAMPLE_ID:
            continue
        if gid in seen:
            raise GoldParseError(
                f"block {gid} appears twice; a duplicated block is two labels for one "
                f"generation and there is no rule for which one wins"
            )
        seen.add(gid)
        fields: dict[str, str] = {}
        for line in match.group("body").splitlines():
            name, sep, value = line.partition(":")
            if sep:
                fields[name.strip().lower()] = value.strip()
        verdict = fields.get("verdict", "").lower()
        if not verdict:
            pending.append(gid)
            continue
        if verdict not in VERDICTS:
            raise GoldParseError(
                f"block {gid}: verdict must be one of {list(VERDICTS)}, got {verdict!r}. "
                f"A row you would argue about is a reject."
            )
        try:
            gen_id = int(gid)
        except ValueError:
            raise GoldParseError(f"block {gid}: the marker id is not a generation id") from None
        labels.append(
            {
                "gen_id": gen_id,
                "verdict": verdict,
                **{
                    axis: _parse_axis(fields.get(axis, ""), field_name=axis, gid=gid)
                    for axis in AXES
                },
                "notes": fields.get("notes") or None,
            }
        )
    return labels, pending


# --------------------------------------------------------------------------
# Folds.
# --------------------------------------------------------------------------

def assign_folds(labels: Sequence[dict], *, folds: int, holdout: int, strata=None) -> list[dict]:
    """Fold numbers 0..folds-1, plus `folds` itself for the holdout.

    The holdout is taken FIRST, stratified, and the rest are dealt round-robin
    within each stratum - so every fold carries the same mix of streams and
    the same mix of accept/reject that the labelled set does. Fold membership
    is content-keyed on gen_id, so re-ingesting the same labels reproduces the
    same split exactly, which is what makes a holdout number comparable
    between two runs.
    """
    if holdout >= len(labels):
        raise ValueError(
            f"holdout {holdout} needs fewer than the {len(labels)} labels available; "
            f"a holdout that takes everything leaves nothing to fit on"
        )
    strata = strata or (lambda row: (row.get("verdict"), ""))
    rows = [dict(row) for row in labels]
    held = {
        row["gen_id"]
        for row in proportional_take(
            rows, holdout, strata=strata, key=lambda row: row["gen_id"]
        )
    }

    buckets: dict = {}
    for row in sorted(rows, key=lambda r: _rank(r["gen_id"])):
        if row["gen_id"] in held:
            row["fold"] = folds
            continue
        buckets.setdefault(strata(row), []).append(row)
    for bucket in buckets.values():
        for i, row in enumerate(bucket):
            row["fold"] = i % folds
    return sorted(rows, key=lambda row: row["gen_id"])


def ingest_gold(store, cfg, text: str) -> dict:
    """Parse, fold and write. Returns what happened, including what did not."""
    _require_block(cfg)
    labels, pending = parse_gold_todo(text)
    if not labels:
        return {"labelled": 0, "pending": len(pending), "written": 0, "folds": {}}
    foldable = assign_folds(
        labels, folds=cfg.calibration.folds, holdout=cfg.calibration.holdout
    )
    written = store.upsert_gold_labels(foldable)
    counts: dict[int, int] = {}
    for row in foldable:
        counts[row["fold"]] = counts.get(row["fold"], 0) + 1
    result = {
        "labelled": len(labels),
        "pending": len(pending),
        "pending_ids": pending,
        "written": written,
        "folds": counts,
        "accept": sum(1 for row in labels if row["verdict"] == ACCEPT),
        "reject": sum(1 for row in labels if row["verdict"] == REJECT),
    }
    store.log_event("gold_ingested", {k: v for k, v in result.items() if k != "pending_ids"})
    return result


# --------------------------------------------------------------------------
# The fit.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    rule: str
    threshold: int

    def __str__(self) -> str:
        return f"{self.rule}>={self.threshold}"


def decides_pass(candidate: Candidate, axes: Sequence[int]) -> bool:
    """Would this rule pass a judgement scoring `axes`?"""
    if candidate.rule not in RULES:
        raise ValueError(f"unknown rule {candidate.rule!r}; the rules are {list(RULES)}")
    low = min(axes)
    mean = sum(axes) / len(axes)
    if candidate.rule == "min_axis":
        return low >= candidate.threshold
    if candidate.rule == "mean":
        return mean >= candidate.threshold
    return low >= candidate.threshold and mean >= candidate.threshold


@dataclass(frozen=True)
class Counts:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        """0.0 when the rule passed nothing.

        Undefined, strictly - but a rule that passes nothing has no precision
        to offer and must not be allowed to win the maximisation by dividing
        by zero into 1.0. The safe direction is that it loses.
        """
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        """0.0 when there was nothing to recall - same reasoning, same
        direction: a fold with no accepted row cannot demonstrate recall, and
        a candidate that cannot demonstrate it does not clear the floor on
        the strength of an empty denominator."""
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def phi(self) -> float:
        """Matthews / phi correlation between the rule and the gold verdict.

        Reported rather than fitted on: it is the number that says whether a
        judge is tracking the operator at all, and a judge with high precision
        and phi near zero is one that passes almost nothing.
        """
        denominator = math.sqrt(
            (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)
        )
        if denominator == 0:
            return 0.0
        return (self.tp * self.tn - self.fp * self.fn) / denominator


def evaluate(candidate: Candidate, pairs: Sequence[tuple]) -> Counts:
    """`pairs` are (axes, gold_is_accept)."""
    tp = fp = tn = fn = 0
    for axes, accepted in pairs:
        passed = decides_pass(candidate, axes)
        if passed and accepted:
            tp += 1
        elif passed:
            fp += 1
        elif accepted:
            fn += 1
        else:
            tn += 1
    return Counts(tp=tp, fp=fp, tn=tn, fn=fn)


def candidates(cfg) -> list[Candidate]:
    return [
        Candidate(rule, threshold)
        for threshold in cfg.calibration.thresholds
        for rule in cfg.calibration.rules
    ]


def best_candidate(pairs: Sequence[tuple], cfg) -> Candidate | None:
    """Highest precision subject to recall >= min_recall, or None.

    None means NO candidate cleared the recall floor - which is a real
    outcome, not an error: a judge that cannot recall 60% of the rows the
    operator accepted has no threshold worth shipping, and inventing one by
    dropping the floor would be moving the gate to fit the result.

    Ties are broken deterministically and in the order that matters: higher
    recall, then the LOWER threshold (a rule that reaches the same precision
    more cheaply), then the order the config listed the rules in.
    """
    rule_order = {rule: i for i, rule in enumerate(cfg.calibration.rules)}
    scored = []
    for candidate in candidates(cfg):
        counts = evaluate(candidate, pairs)
        if counts.recall < cfg.calibration.min_recall:
            continue
        scored.append(
            (
                -counts.precision,
                -counts.recall,
                candidate.threshold,
                rule_order[candidate.rule],
                candidate,
            )
        )
    if not scored:
        return None
    return min(scored)[-1]


@dataclass
class JudgeFit:
    model: str
    n_gold: int
    candidate: Candidate | None = None
    fold_selections: dict = field(default_factory=dict)
    selection_agreement: int = 0
    fold_counts: dict = field(default_factory=dict)
    cv: Counts | None = None
    holdout: Counts | None = None
    disqualified: bool = False
    reason: str | None = None
    replacement: str | None = None
    # False for a model the gold set covers that holds no routing.judge seat -
    # a tiebreak-only model, since 2026-08-19. It is still fitted and still
    # reported; it just makes no claim on the judge bench. See calibrate().
    holds_judge_seat: bool = True

    @property
    def holdout_precision(self) -> float:
        return self.holdout.precision if self.holdout else 0.0


def fit_judge(model: str, rows: Sequence[dict], cfg) -> JudgeFit:
    """One judge model's threshold, cross-validated, then held out.

    `rows` are {"fold", "axes", "accepted"} - one per gold-labelled generation
    this model actually scored.
    """
    holdout_fold = cfg.calibration.folds
    fit_rows = [row for row in rows if row["fold"] != holdout_fold]
    holdout_rows = [row for row in rows if row["fold"] == holdout_fold]
    fit = JudgeFit(model=model, n_gold=len(rows))

    if not fit_rows:
        fit.disqualified = True
        fit.reason = "no gold-labelled judgement outside the holdout fold"
        return fit

    # Selection stability: choose five times, each on four fifths.
    for fold in sorted({row["fold"] for row in fit_rows}):
        trained = [row for row in fit_rows if row["fold"] != fold]
        chosen = best_candidate([(r["axes"], r["accepted"]) for r in trained], cfg)
        fit.fold_selections[fold] = str(chosen) if chosen else None

    shipped = best_candidate([(r["axes"], r["accepted"]) for r in fit_rows], cfg)
    if shipped is None:
        fit.disqualified = True
        fit.reason = (
            f"no rule in {[str(c) for c in candidates(cfg)]} reaches recall "
            f"{cfg.calibration.min_recall:.2f} on {len(fit_rows)} labelled rows"
        )
        return fit
    fit.candidate = shipped
    fit.selection_agreement = sum(
        1 for value in fit.fold_selections.values() if value == str(shipped)
    )
    fit.fold_counts = {
        fold: evaluate(shipped, [(r["axes"], r["accepted"]) for r in fit_rows if r["fold"] == fold])
        for fold in sorted({row["fold"] for row in fit_rows})
    }
    fit.cv = evaluate(shipped, [(r["axes"], r["accepted"]) for r in fit_rows])

    if not holdout_rows:
        fit.disqualified = True
        fit.reason = "no gold-labelled judgement in the holdout fold"
        return fit
    fit.holdout = evaluate(shipped, [(r["axes"], r["accepted"]) for r in holdout_rows])
    if fit.holdout.precision < cfg.calibration.min_precision:
        fit.disqualified = True
        fit.reason = (
            f"holdout precision {fit.holdout.precision:.3f} is below the "
            f"{cfg.calibration.min_precision:.2f} gate on {fit.holdout.n} held-out rows"
        )
    return fit


def cohens_kappa(pairs: Sequence[tuple[bool, bool]]) -> float:
    """Agreement between two binary raters, corrected for chance.

    0.0 when the two agree exactly as often as chance predicts, and 0.0 also
    when both raters are constant - there is no agreement beyond chance to
    measure when neither ever varies, and the alternative is a division by
    zero reported as perfect agreement.
    """
    n = len(pairs)
    if not n:
        return 0.0
    observed = sum(1 for a, b in pairs if a == b) / n
    a_yes = sum(1 for a, _ in pairs if a) / n
    b_yes = sum(1 for _, b in pairs if b) / n
    expected = a_yes * b_yes + (1 - a_yes) * (1 - b_yes)
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1 - expected)


# --------------------------------------------------------------------------
# The calibration.
# --------------------------------------------------------------------------

@dataclass
class Calibration:
    fits: list[JudgeFit] = field(default_factory=list)
    kappa: dict = field(default_factory=dict)
    n_gold: int = 0
    swaps: list[dict] = field(default_factory=list)
    unlabelled_models: list[str] = field(default_factory=list)
    # Set when the labelled store is smaller than calibration.pilot_export.
    # The fit must not run and no threshold row may be written.
    blocked: str | None = None

    @property
    def active(self) -> list[JudgeFit]:
        return [fit for fit in self.fits if not fit.disqualified and fit.candidate]


def _require_block(cfg) -> None:
    if cfg.calibration is None:
        raise ValueError(
            "this build config has no `calibration:` block, so there is no export size, "
            "no fold count, no holdout and no P5 gate. calibrate.py cannot invent them: "
            "min_recall and min_precision ARE the gate."
        )


def _routing_models(cfg, role: str = "judge") -> list[str]:
    return [ref.model for ref in cfg.routing_refs(role)]


def assign_replacements(calibration: "Calibration", routed: list[str]) -> None:
    """Name a successor for each disqualified JUDGE, and record the swaps.

    `routed` is routing.judge, in preference order. Mutates the fits in place
    and appends to `calibration.swaps`.

    A REPLACEMENT IS A JUDGE SEAT'S SUCCESSION, and only a model holding one
    may claim it. `calibration.fits` covers every model the gold set carries
    judgements from, which since 2026-08-19 includes TIEBREAK-ONLY seats.
    Without this fence a disqualified tiebreak model pops the first spare judge
    off the list, and the shipped report said exactly that: gemma, then
    tiebreak-only, was handed "named replacement: gpt-5-mini" while qwen - the
    actual slot-A judge, also disqualified - was told "NONE LEFT IN
    routing.judge". The one seat with a successor available was the one told it
    had none.

    A disqualified tiebreak-only model is still REPORTED disqualified: the
    measurement is real and the operator should see it. What it does not get is
    a claim on the judge bench; its own seat's succession is routing.tiebreak's
    business, and this module does not fit that pool.

    Extracted from calibrate() so the property can be tested at the unit rather
    than only through a whole store.
    """
    calibrated = {fit.model for fit in calibration.fits}
    judge_seats = set(routed)
    spare = [model for model in routed if model not in calibrated]
    for fit in calibration.fits:
        if not fit.disqualified:
            continue
        holds_a_judge_seat = fit.model in judge_seats
        replacement = (spare.pop(0) if spare else None) if holds_a_judge_seat else None
        fit.replacement = replacement
        fit.holds_judge_seat = holds_a_judge_seat
        calibration.swaps.append(
            {
                "model": fit.model,
                "replacement": replacement,
                "holds_judge_seat": holds_a_judge_seat,
                "reason": fit.reason,
                "holdout_precision": fit.holdout_precision,
            }
        )


def calibrate(store, cfg) -> Calibration:
    """Fit every judge model the gold set covers; name a swap for each failure."""
    _require_block(cfg)
    labels = {row["gen_id"]: row for row in store.gold_labels()}
    calibration = Calibration(n_gold=len(labels))
    if not labels:
        return calibration
    required = cfg.calibration.pilot_export
    if len(labels) < required:
        calibration.blocked = "insufficient-labels"
        return calibration

    judgements = store.judgements_by_gen(labels)
    by_model: dict[str, list[dict]] = {}
    for gen_id, rows in judgements.items():
        label = labels[gen_id]
        for row in rows:
            axes = _axes_of(row)
            if axes is None or row.get("model") is None:
                continue
            by_model.setdefault(row["model"], []).append(
                {
                    "fold": label.get("fold"),
                    "axes": axes,
                    "accepted": label["verdict"] == ACCEPT,
                    "gen_id": gen_id,
                    "slot": row.get("judge_slot"),
                }
            )

    for model in sorted(by_model):
        calibration.fits.append(fit_judge(model, by_model[model], cfg))

    routed = _routing_models(cfg)
    calibration.unlabelled_models = [model for model in routed if model not in by_model]

    # A named swap, never a performed one: routing lives in the YAML and a
    # module that rewrote it would be moving a fence nobody had read.
    #
    # A REPLACEMENT IS A JUDGE SEAT'S SUCCESSION, and only a model holding one
    # may claim it. `fits` covers every model the gold set carries judgements
    # from, which since 2026-08-19 includes TIEBREAK-ONLY seats; `routed` is
    # routing.judge alone. Without this fence a disqualified tiebreak model
    # popped a spare off the judge list and the shipped report said so: gemma,
    # then tiebreak-only, was handed "replacement gpt-5-mini" while qwen - the
    # actual slot-A judge, also disqualified - was told "NONE LEFT". The advice
    # was inverted for the only seat that had one to give.
    #
    # A disqualified tiebreak-only model is still REPORTED disqualified: the
    # measurement is real and the operator should see it. What it does not get
    # is a claim on the judge bench. Its own seat's succession is
    # routing.tiebreak's business and this module does not fit that pool.
    assign_replacements(calibration, routed)

    # Inter-judge agreement, over the generations two judges both scored,
    # under each judge's own fitted rule. Measured on the gold set because
    # that is the only place the two are comparable against a truth.
    by_gen: dict[int, list[tuple[str, tuple]]] = {}
    for model, rows in by_model.items():
        for row in rows:
            by_gen.setdefault(row["gen_id"], []).append((model, row["axes"]))
    fitted = {fit.model: fit.candidate for fit in calibration.fits if fit.candidate}
    pair_counts: dict[tuple[str, str], list] = {}
    for scored in by_gen.values():
        usable = [(model, axes) for model, axes in scored if fitted.get(model)]
        for i in range(len(usable)):
            for j in range(i + 1, len(usable)):
                (model_a, axes_a), (model_b, axes_b) = usable[i], usable[j]
                key = tuple(sorted((model_a, model_b)))
                first = axes_a if key[0] == model_a else axes_b
                second = axes_b if key[0] == model_a else axes_a
                pair_counts.setdefault(key, []).append(
                    (
                        decides_pass(fitted[key[0]], first),
                        decides_pass(fitted[key[1]], second),
                    )
                )
    calibration.kappa = {
        f"{a} vs {b}": {"kappa": cohens_kappa(pairs), "n": len(pairs)}
        for (a, b), pairs in sorted(pair_counts.items())
    }
    return calibration


def threshold_rows(calibration: Calibration, *, fitted_at: str | None = None) -> list[dict]:
    """The judge_threshold rows for the judges that PASSED.

    A disqualified judge writes no row on purpose: judge.thresholds_active
    counts active rows, and a row for a model the gate rejected would report
    the fleet as calibrated on a judge nobody may use.

    calib_id CARRIES THE TIMESTAMP, and that is not decoration. It is the
    table's primary key, so an id built from (model, rule, n_gold) alone makes
    a re-run INSERT OR REPLACE over its own predecessor - and the superseded
    row store.record_judge_thresholds had just marked active = 0 disappears
    instead of being kept. The whole point of keeping it is that a later
    report is only interpretable against the fit it replaced.
    """
    stamp = fitted_at or utcnow()
    rows = []
    for fit in calibration.active:
        calib_id = hashlib.sha256(
            f"{fit.model}|{fit.candidate}|{fit.n_gold}|{stamp}".encode("utf-8")
        ).hexdigest()[:16]
        rows.append(
            {
                "calib_id": calib_id,
                "judge_slot": None,
                "model": fit.model,
                "rule": fit.candidate.rule,
                "threshold": fit.candidate.threshold,
                "precision": fit.holdout.precision,
                "recall": fit.holdout.recall,
                "n_gold": fit.n_gold,
                "fitted_at": stamp,
            }
        )
    return rows


def calibration_report(calibration: Calibration, cfg) -> str:
    """calibration_report.md - every number in it measured on the gold set."""
    lines = [
        "# calibration_report.md",
        "",
        f"- gold labels: {calibration.n_gold}",
        f"- gate: precision >= {cfg.calibration.min_precision:.2f} on the holdout, "
        f"chosen to maximise precision subject to recall >= {cfg.calibration.min_recall:.2f}",
        f"- folds: 0-{cfg.calibration.folds - 1} for selection, "
        f"fold {cfg.calibration.folds} held out and read once",
    ]
    if calibration.blocked == "insufficient-labels":
        lines += [
            f"- BLOCKED: insufficient labels ({calibration.n_gold} < "
            f"{cfg.calibration.pilot_export} required by calibration.pilot_export); "
            f"no fit was run and no judge_threshold row was written",
            "",
        ]
        return "\n".join(lines) + "\n"
    lines += [
        "",
        "## Per judge",
        "",
    ]
    for fit in calibration.fits:
        lines += [
            f"### {fit.model}",
            "",
            f"- gold-labelled judgements: {fit.n_gold}",
            f"- fitted rule: {fit.candidate or 'NONE'}",
            f"- fold selections: {fit.fold_selections or '-'} "
            f"(agreement {fit.selection_agreement}/{len(fit.fold_selections)})",
        ]
        if fit.cv:
            lines.append(
                f"- cross-validation folds: precision {fit.cv.precision:.3f}, "
                f"recall {fit.cv.recall:.3f}, phi {fit.cv.phi:.3f}, n {fit.cv.n}"
            )
            lines.append(
                "- per fold precision/recall: "
                + ", ".join(
                    f"{fold}: {counts.precision:.2f}/{counts.recall:.2f}"
                    for fold, counts in sorted(fit.fold_counts.items())
                )
            )
        if fit.holdout:
            lines.append(
                f"- HOLDOUT: precision {fit.holdout.precision:.3f}, "
                f"recall {fit.holdout.recall:.3f}, phi {fit.holdout.phi:.3f}, "
                f"n {fit.holdout.n}"
            )
        if fit.disqualified:
            lines.append(f"- **DISQUALIFIED**: {fit.reason}")
            if fit.holds_judge_seat:
                lines.append(
                    f"- named replacement: {fit.replacement or 'NONE LEFT IN routing.judge'}"
                )
            else:
                lines.append(
                    "- no replacement named: this model holds no routing.judge seat, "
                    "so it has no judge bench to be replaced on; its own seat's "
                    "succession is routing.tiebreak's business"
                )
        lines.append("")

    lines += ["## Inter-judge agreement (Cohen's kappa)", ""]
    if calibration.kappa:
        for pair, value in calibration.kappa.items():
            lines.append(f"- {pair}: kappa {value['kappa']:.3f} over {value['n']} generations")
    else:
        lines.append("- no generation was scored by two fitted judges")
    lines.append("")

    if calibration.unlabelled_models:
        lines += [
            "## Judges the gold set does not cover",
            "",
            "These are in routing.judge and no gold-labelled generation carries a "
            "judgement from them, so nothing was fitted and nothing is claimed:",
            "",
        ]
        lines += [f"- {model}" for model in calibration.unlabelled_models]
        lines.append("")
    return "\n".join(lines)


def run_calibration(store, cfg) -> tuple[Calibration, str]:
    """Fit, write the active thresholds, return the report."""
    calibration = calibrate(store, cfg)
    if calibration.blocked:
        store.log_event(
            "judges_calibration_blocked",
            {
                "reason": calibration.blocked,
                "n_gold": calibration.n_gold,
                "required": cfg.calibration.pilot_export,
            },
        )
        return calibration, calibration_report(calibration, cfg)
    rows = threshold_rows(calibration)
    if rows:
        store.record_judge_thresholds(rows)
    store.log_event(
        "judges_calibrated",
        {
            "n_gold": calibration.n_gold,
            "active": [fit.model for fit in calibration.active],
            "disqualified": [swap["model"] for swap in calibration.swaps],
            "swaps": calibration.swaps,
            "kappa": calibration.kappa,
        },
    )
    return calibration, calibration_report(calibration, cfg)


def main(argv=None) -> int:
    import argparse
    from pathlib import Path

    from tuned.data.config import load_build_config
    from tuned.data.paths import build_paths
    from tuned.data.store import Store

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--export", action="store_true", help=f"write {GOLD_TODO_NAME}")
    parser.add_argument("--ingest", action="store_true", help=f"read {GOLD_TODO_NAME} back")
    parser.add_argument("--fit", action="store_true", help="fit thresholds and report")
    parser.add_argument("--stream", action="append", default=None)
    args = parser.parse_args(argv)
    if not (args.export or args.ingest or args.fit):
        parser.error("nothing to do: pass --export, --ingest or --fit")

    cfg = load_build_config(args.config)
    paths = build_paths(cfg.build.workdir).ensure()
    store = Store.open(paths.state_db)
    todo_path = Path(paths.gold_dir) / GOLD_TODO_NAME
    try:
        if args.export:
            rows = export_pilot(store, cfg, streams=args.stream)
            seeds = {
                row["seed_id"]: store.get_seed(row["seed_id"])
                for row in rows
                if row.get("seed_id")
            }
            todo_path.write_text(
                render_gold_todo(rows, seeds={k: v for k, v in seeds.items() if v}),
                encoding="utf-8",
            )
            print(f"exported {len(rows)} generations to {todo_path}")
            print("THESE LABELS ARE WRITTEN BY A PERSON. Nothing here fills one in.")
        if args.ingest:
            result = ingest_gold(store, cfg, todo_path.read_text(encoding="utf-8"))
            print(
                f"ingested {result['written']} labels "
                f"({result.get('accept', 0)} accept / {result.get('reject', 0)} reject); "
                f"{result['pending']} blocks still blank"
            )
            print("  folds: " + ", ".join(f"{k}={v}" for k, v in sorted(result["folds"].items())))
        if args.fit:
            calibration, report = run_calibration(store, cfg)
            report_path = Path(paths.gold_dir) / CALIBRATION_REPORT_NAME
            report_path.write_text(report, encoding="utf-8")
            print(f"gold labels {calibration.n_gold}; report -> {report_path}")
            for fit in calibration.fits:
                state = "DISQUALIFIED" if fit.disqualified else "active"
                print(
                    f"  {fit.model:<28} {str(fit.candidate or '-'):<14} "
                    f"holdout precision {fit.holdout_precision:.3f}  {state}"
                )
            for swap in calibration.swaps:
                if swap.get("holds_judge_seat", True):
                    print(
                        f"  SWAP {swap['model']} -> "
                        f"{swap['replacement'] or 'NOTHING LEFT'}"
                    )
                else:
                    print(
                        f"  SWAP {swap['model']} -> (no routing.judge seat; "
                        f"not a judge swap)"
                    )
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
