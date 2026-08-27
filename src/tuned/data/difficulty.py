"""Difficulty labels: one probe calibration, then a length proxy for everything.

PER-ROW PROBING DOES NOT EXIST HERE, AND THAT IS ARCHITECTURE RATHER THAN
ADVICE. Probing each row of an 18,000-row corpus was costed at 32M tokens and
65 days of free-tier quota - a build that never finishes is not a build with a
slow labelling step. So at most `difficulty.probe_sample` rows ever reach a
model, once, and every row in the corpus is labelled by a pure function of its
length afterwards. The constraint is enforced two ways: `label_rows` and
`label_corpus` take no probe and cannot reach one (a test walks their syntax
tree and fails on any name containing "probe"), and `probe_rows` refuses a
sample above the configured ceiling.

WHAT THE PROBE IS ACTUALLY FOR. It is not the label - the length proxy is the
label. The probe answers ONE question: does length point the way this module
assumes it points, on THIS corpus? Bands are cut at the target mix's quantiles
whatever the probe says; what the probe decides is whether those bands may be
used at all. If the short rows are not easier for a weak model than the long
ones, the proxy is not measuring difficulty here and `calibrate_bands` refuses
rather than labelling 18,000 rows off an assumption nobody checked. Both
directions are tested on injected probe outcomes.

HOW A PROBE OUTCOME IS GRADED, and its limit stated plainly. The shipped probe
prompt (prompts/probe_answer_v1.md) asks a small model for the answer itself
and tells it to say so plainly if it does not know. With no reference answer
there is nothing to mark it against, so the default grade is whether it
answered at all rather than whether it was right - `declined_to_answer`. That
is a weaker signal than correctness and it is not pretended otherwise; it is
also the only one this prompt supports. The transition stream is the exception
worth taking next: those rows carry deterministic answer keys, so
gates.check_answer_key could grade them exactly. `grade` is a parameter for
that reason.

WHERE THE LABELS LAND, and a measured correction to the brief. The brief says
tasks.py expects them. It does not. Measured:
`grep -c difficulty src/tuned/data/tasks.py` is 0, and every hit outside this
module is in config.py - the DifficultyCfg dataclass, its block loader and the
build.difficulty_target share check - all of it configuration plumbing, none of
it a read on a seed or task row. decontaminate.generated_rows does not put one
in `_prov` either, so nothing downstream carries it into the dataset yet.

So the label is written where the plan's provenance list puts it, on the seed
row's meta_json, and the fact that NO CONSUMER READS IT YET is recorded here
rather than disguised by a call that does not exist. Wiring it into `_prov` is
a one-line change in decontaminate.py and deliberately not made here: that
module owns the dataset row's provenance schema and this task does not.

Build:  python -m tuned.data.difficulty --config configs/data_law_v1.yaml
        [--probe] [--dry-run]
"""

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

# Easiest first. The order is load-bearing: bands are cut at the CUMULATIVE
# shares of this sequence, so reversing it would put the hard label on the
# short rows.
LABELS = ("easy", "medium", "hard")

# What a row's "length" is. token_count when the row carries one (every seed
# builder writes it, as a chars/4 estimate or a real count), and the same
# estimate over the text when it does not - never zero, because a row with no
# length lands in the easy band and a corpus of them would satisfy any mix.
CHARS_PER_TOKEN = 4

# The phrases the shipped probe prompt invites when the model does not know.
# Matched against the whitespace-normalised, lower-cased reply, so a phrase
# broken across a line still counts.
DECLINE_PHRASES = (
    "i do not know",
    "i don't know",
    "i am not sure",
    "i'm not sure",
    "cannot say",
    "can not say",
    "unable to say",
    "not able to answer",
    "cannot answer",
    "no idea",
)

PROBE_ROLE = "probe"
PROBE_PROMPT_ID = "probe_answer_v1"

# How much of a row the probe is shown. The probe is a 131k-context model, so
# this is not a context budget - it is a COST one. The calibration asks
# whether a weak model can answer a row at all, and the head of a row is
# enough to decide that; sending whole judgments would multiply the one paid
# pass this module is allowed by the length of the corpus's longest rows.
PROBE_QUESTION_CHARS = 4000


@dataclass(frozen=True)
class ProbeOutcome:
    """One calibration row: what it was, how long it was, did the probe answer."""

    row_id: str
    length: int
    solved: bool


@dataclass(frozen=True)
class Bands:
    """The two cut points, and the evidence they were allowed to be used on.

    `solve_rates` is per label over the calibration sample and is what the
    refusal reads. It is carried on the Bands rather than logged and dropped
    because the manifest has to be able to say WHY these bands were accepted
    months after the run.
    """

    easy_max: int
    medium_max: int
    n_probed: int
    solve_rates: dict
    target: dict

    def label(self, length: int) -> str:
        if length <= self.easy_max:
            return LABELS[0]
        if length <= self.medium_max:
            return LABELS[1]
        return LABELS[2]

    def as_dict(self) -> dict:
        return {
            "easy_max": self.easy_max,
            "medium_max": self.medium_max,
            "n_probed": self.n_probed,
            "solve_rates": dict(self.solve_rates),
            "target": dict(self.target),
        }


class ProxyRefused(ValueError):
    """The probe says length does not point the way the proxy assumes."""


# --------------------------------------------------------------------------
# Lengths and the sample.
# --------------------------------------------------------------------------

def row_length(row) -> int:
    """The row's length in the proxy's units. Never negative, never None."""
    count = row.get("token_count")
    if isinstance(count, (int, float)) and count > 0:
        return int(count)
    return len(str(row.get("text") or "")) // CHARS_PER_TOKEN


def _rank(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def probe_sample(rows: Sequence[dict], cfg, *, key=lambda row: row["seed_id"]) -> list[dict]:
    """The rows that may reach the probe: at most `difficulty.probe_sample`.

    Content-keyed, so the calibration sample is the same on any machine and a
    re-run after a crash re-probes the same rows instead of spending the
    quota again on a different thousand.

    Spread over the LENGTH RANGE rather than drawn flat: a sample that happens
    to miss the long tail cannot say whether the long tail is harder, which is
    the one question the probe is being paid to answer. So the rows are sorted
    by length, cut into as many strata as the sample has rows, and one row is
    taken from each - by sha within the stratum, so which one is still not a
    choice anybody made.
    """
    _require_block(cfg)
    limit = cfg.difficulty.probe_sample
    ordered = sorted(rows, key=lambda row: (row_length(row), _rank(key(row))))
    if len(ordered) <= limit:
        return ordered
    stride = len(ordered) / limit
    picked: list[dict] = []
    for i in range(limit):
        lo = int(i * stride)
        hi = max(lo + 1, int((i + 1) * stride))
        stratum = ordered[lo:hi]
        picked.append(min(stratum, key=lambda row: _rank(key(row))))
    return picked


# --------------------------------------------------------------------------
# Grading a probe reply.
# --------------------------------------------------------------------------

def declined_to_answer(text: str | None) -> bool:
    body = " ".join((text or "").split()).lower()
    if not body:
        return True
    return any(phrase in body for phrase in DECLINE_PHRASES)


def default_grade(row, text: str | None) -> bool:
    """Did the probe answer this row at all? See the module docstring on why
    this is the honest reading of the shipped probe prompt and not a claim
    about correctness."""
    return not declined_to_answer(text)


def probe_rows(rows: Sequence[dict], replies: Sequence, cfg, *, grade=default_grade,
               key=lambda row: row["seed_id"]) -> list[ProbeOutcome]:
    """Pair the sampled rows with what the probe said. PURE.

    `replies` is whatever the caller got back, one per row, in order - a live
    router's texts offline tests' fixtures. The refusal on an over-sized
    sample is here rather than at the call site because THIS is the function
    that could quietly become per-row probing: nothing else in the module
    takes probe replies at all.
    """
    _require_block(cfg)
    if len(replies) != len(rows):
        raise ValueError(
            f"{len(replies)} probe replies for {len(rows)} rows; they are paired by "
            f"position and a mismatch would attribute one row's answer to another"
        )
    if len(rows) > cfg.difficulty.probe_sample:
        raise ValueError(
            f"{len(rows)} rows were sent to the probe, above the "
            f"difficulty.probe_sample ceiling of {cfg.difficulty.probe_sample}. "
            f"Per-row probing was costed at 32M tokens / 65 days on this corpus; the "
            f"ceiling is the design, not a batch size."
        )
    return [
        ProbeOutcome(row_id=str(key(row)), length=row_length(row), solved=bool(grade(row, reply)))
        for row, reply in zip(rows, replies)
    ]


# --------------------------------------------------------------------------
# Calibration.
# --------------------------------------------------------------------------

# Slack for the ceil() in _quantile, and it is the SAME arithmetic
# config._SHARE_EPS exists for. The cut points are taken at CUMULATIVE target
# shares, and 0.34 + 0.50 is 0.8400000000000001 in binary floating point - so
# over 1,000 rows a bare ceil() asks for rank 841 where the share names 840
# and the medium/hard boundary moves one row for no reason anybody could find
# later. Nine places is far below any share an operator would write and far
# above the representation error; the same number, and the same reasoning, as
# config._SHARE_EPS and stats.gate_mix's rounding.
_RANK_EPS = 1e-9


def _quantile(values: Sequence[int], q: float) -> int:
    """Nearest-rank quantile: an actual observed length, never interpolated.

    A cut point between two observed lengths would be a threshold no row can
    sit on, and comparing `<=` against it makes the achieved mix depend on
    which side the rounding fell - which is exactly the drift check_mix would
    then report and nobody could explain.
    """
    if not values:
        return 0
    rank = max(1, min(len(values), math.ceil(q * len(values) - _RANK_EPS)))
    return values[rank - 1]


def target_shares(cfg) -> dict:
    """build.difficulty_target, as a share per label in LABELS order.

    Read from `build:` and not from the difficulty block - one definition, the
    same rule the eval fraction and the mix targets follow. A label the config
    does not mention is 0.0 rather than an error: a two-way easy/hard split is
    a legitimate target and config.py already grades the shares.
    """
    _require_block(cfg)
    return {label: float(cfg.build.difficulty_target.get(label, 0.0)) for label in LABELS}


def calibrate_bands(outcomes: Sequence[ProbeOutcome], cfg) -> Bands:
    """Length cut points at the target mix's quantiles, IF the probe allows it.

    The bands are the target's quantiles whatever the probe found - that is
    what makes the mix land where the config asked. What the probe decides is
    whether they may be used at all: if a weak model does not do better on the
    short rows than on the long ones, length is not tracking difficulty on
    this corpus and labelling 18,000 rows off it would be inventing a signal.
    The refusal carries the measured rates so the operator is arguing with a
    number rather than with this module.
    """
    _require_block(cfg)
    if not outcomes:
        raise ProxyRefused("no probe outcomes: there is nothing to calibrate the proxy on")
    target = target_shares(cfg)
    lengths = sorted(outcome.length for outcome in outcomes)
    easy_share = target[LABELS[0]]
    medium_share = target[LABELS[1]]
    easy_max = _quantile(lengths, easy_share)
    medium_max = max(easy_max, _quantile(lengths, easy_share + medium_share))

    provisional = Bands(
        easy_max=easy_max, medium_max=medium_max, n_probed=len(outcomes),
        solve_rates={}, target=target,
    )
    buckets: dict[str, list[bool]] = {label: [] for label in LABELS}
    for outcome in outcomes:
        buckets[provisional.label(outcome.length)].append(outcome.solved)
    rates = {
        label: (sum(values) / len(values) if values else None)
        for label, values in buckets.items()
    }

    short, long = rates[LABELS[0]], rates[LABELS[-1]]
    if short is None or long is None:
        raise ProxyRefused(
            f"the probe sample does not cover both ends of the band: solve rates {rates}. "
            f"A proxy checked at one end only has not been checked."
        )
    if short <= long:
        raise ProxyRefused(
            f"the probe solved {short:.1%} of the SHORT rows and {long:.1%} of the LONG "
            f"ones, so length is not tracking difficulty on this corpus and the proxy "
            f"would be inventing a signal. Measured over {len(outcomes)} probed rows; "
            f"per-band rates {rates}."
        )
    return Bands(
        easy_max=easy_max, medium_max=medium_max, n_probed=len(outcomes),
        solve_rates=rates, target=target,
    )


# --------------------------------------------------------------------------
# Labelling. NOTHING BELOW THIS LINE MAY REACH A MODEL.
# --------------------------------------------------------------------------

def label_rows(rows: Sequence[dict], bands: Bands) -> list[str]:
    """One label per row, from length alone. Pure, and no probe in sight."""
    return [bands.label(row_length(row)) for row in rows]


def measure_mix(labels: Sequence[str]) -> dict:
    total = len(labels)
    if not total:
        return {label: 0.0 for label in LABELS}
    counts = {label: 0 for label in LABELS}
    for label in labels:
        counts[label] += 1
    return {label: counts[label] / total for label in LABELS}


def mix_drift(mix: dict, cfg) -> dict:
    _require_block(cfg)
    target = target_shares(cfg)
    return {label: mix.get(label, 0.0) - target[label] for label in LABELS}


def check_mix(mix: dict, cfg) -> None:
    """Refuse bands that do not deliver the mix they were cut for.

    They can miss it, and the reason is worth naming: the cut points are
    observed lengths, so a corpus where thousands of rows share one length
    puts all of them on the same side of a boundary however the quantile
    falls. That is a fact about the corpus and not a bug to round away, so it
    is reported as a refusal with the drift rather than silently accepted.
    """
    _require_block(cfg)
    drift = mix_drift(mix, cfg)
    worst = max(drift, key=lambda label: abs(drift[label]))
    if abs(drift[worst]) > cfg.difficulty.mix_tolerance:
        raise ProxyRefused(
            f"the labelled mix misses build.difficulty_target by {drift[worst]:+.3f} on "
            f"{worst!r}, outside difficulty.mix_tolerance "
            f"({cfg.difficulty.mix_tolerance}). Measured mix {mix}, target "
            f"{target_shares(cfg)}. Length ties at a cut point are the usual cause and "
            f"they are a fact about the corpus, not a rounding to wave through."
        )


def label_corpus(rows: Sequence[dict], bands: Bands, cfg) -> tuple[list[dict], dict]:
    """(rows with a `difficulty`, the measured mix). Pure, and no probe."""
    labels = label_rows(rows, bands)
    mix = measure_mix(labels)
    check_mix(mix, cfg)
    return [{**row, "difficulty": label} for row, label in zip(rows, labels)], mix


# --------------------------------------------------------------------------
# The build.
# --------------------------------------------------------------------------

def _require_block(cfg) -> None:
    if getattr(cfg, "difficulty", None) is None:
        raise ValueError(
            "this build config has no `difficulty:` block, so there is no probe ceiling "
            "and no mix tolerance. difficulty.py will not guess at either: the ceiling "
            "is what keeps a 1,000-row calibration from becoming an 18,000-row one."
        )


def _seed_rows(store) -> list[dict]:
    return [
        dict(row)
        for row in store.conn.execute("SELECT * FROM seed ORDER BY seed_id").fetchall()
    ]


def apply_labels(store, labelled: Sequence[dict]) -> int:
    """Write `difficulty` onto each seed's meta_json. Returns rows written.

    meta_json rather than a column: no consumer reads the field yet (see the
    module docstring), and adding a column for a field nothing selects would
    be schema for a reader that does not exist. When one arrives, the value is
    already there under the name the plan's provenance list gives it.
    """
    rows = []
    for row in labelled:
        meta = row.get("meta_json")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except ValueError:
                meta = {}
        meta = dict(meta or {})
        meta["difficulty"] = row["difficulty"]
        rows.append({**row, "meta_json": meta})
    return store.upsert_seeds(rows)


def build_difficulty(store, cfg, *, outcomes=None, bands=None, dry_run=False) -> dict:
    """Calibrate (or accept pre-calibrated bands) and label every seed row.

    `outcomes` are probe results the CALLER obtained - this function never
    reaches a model, which is why the live probe lives in main() and the
    offline tests can drive the whole path with a fixture.
    """
    _require_block(cfg)
    rows = _seed_rows(store)
    if bands is None:
        if outcomes is None:
            raise ValueError(
                "build_difficulty needs either fitted bands or probe outcomes to fit "
                "them from; it does not probe, and a default band would be a threshold "
                "nobody measured"
            )
        bands = calibrate_bands(outcomes, cfg)

    labelled, mix = label_corpus(rows, bands, cfg)
    manifest = {
        "seeds": len(rows),
        "bands": bands.as_dict(),
        "mix": mix,
        "drift": mix_drift(mix, cfg),
        "tolerance": cfg.difficulty.mix_tolerance,
        "dry_run": dry_run,
    }
    if dry_run:
        manifest["written"] = 0
        return manifest
    manifest["written"] = apply_labels(store, labelled)
    store.log_event("difficulty_labelled", manifest)
    return manifest


def probe_questions(rows: Sequence[dict]) -> list[list[dict]]:
    """The probe's message lists, one per sampled row.

    A separate function so the live path is a loop over MESSAGES rather than a
    loop over the corpus - which is the shape the whole module exists to
    avoid, and the shape a reader has to be able to check at a glance.
    """
    from tuned.data import prompt_registry

    return [
        prompt_registry.render(
            PROBE_PROMPT_ID, question=str(row.get("text") or "")[:PROBE_QUESTION_CHARS]
        )
        for row in rows
    ]


def main(argv=None) -> int:  # pragma: no cover - the live probe path
    import argparse
    import asyncio

    from tuned.data.config import load_build_config
    from tuned.data.generate import make_router, usage_recorder
    from tuned.data.paths import build_paths
    from tuned.data.store import Store

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data_law_v1.yaml")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="run the probe calibration (spends quota, at most difficulty.probe_sample rows)",
    )
    parser.add_argument("--dry-run", action="store_true", help="measure, write nothing")
    args = parser.parse_args(argv)

    cfg = load_build_config(args.config)
    paths = build_paths(cfg.build.workdir).ensure()
    store = Store.open(paths.state_db)
    try:
        if not args.probe:
            raise SystemExit(
                "pass --probe: the bands come from one calibration pass and there is no "
                "stored calibration to relabel from yet"
            )
        rows = probe_sample(_seed_rows(store), cfg)
        router = make_router(store, cfg)

        async def drive():
            replies = []
            try:
                for messages in probe_questions(rows):
                    # No est_tokens -> est_tokens=0 -> a usd_cap gate sees
                    # est_cost 0.0 and a zero-cost call never trips
                    # `spent + 0.0 > cap` on its own (generate.budget_ok_for).
                    # Harmless while routing.probe holds a single free ref
                    # with no usd_cap declared, but this re-opens THE TRAP
                    # (see the openai/mistral provider blocks) the moment a
                    # priced ref reaches this role. Not changed here - a
                    # separate decision (found by review, 2026-08-27).
                    _, response = await router.complete(
                        PROBE_ROLE, messages, max_tokens=256,
                        on_attempt=usage_recorder(store),
                    )
                    replies.append(response.text)
            finally:
                await router.aclose()
            return replies

        outcomes = probe_rows(rows, asyncio.run(drive()), cfg)
        manifest = build_difficulty(store, cfg, outcomes=outcomes, dry_run=args.dry_run)
    finally:
        store.close()

    print(f"probed {manifest['bands']['n_probed']} rows; solve rates {manifest['bands']['solve_rates']}")
    print(f"bands easy<={manifest['bands']['easy_max']} medium<={manifest['bands']['medium_max']}")
    print(
        "mix " + ", ".join(f"{k}={v:.3f}" for k, v in manifest["mix"].items())
        + "  drift " + ", ".join(f"{k}={v:+.3f}" for k, v in manifest["drift"].items())
    )
    print(f"labelled {manifest['written']} of {manifest['seeds']} seed rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
