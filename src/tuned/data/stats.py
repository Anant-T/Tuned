"""The terminal builder gate: measure assemble.py's output, refuse if red.

Input is `out/law_v1_train.jsonl` and `out/law_v1_eval.jsonl`, verified against
the digests assemble.py recorded. Output is `out/stats.json` (machine-readable)
and `out/stats.md` (for a human and for push.py's dataset card), and an EXIT
CODE: non-zero if any gate is red. This is the only stage in the build whose
purpose is to say no.

PURE MEASUREMENT. Nothing here edits, filters or re-runs a decision. A gate
that could fix what it measures would stop being an instrument - and every
number below is one some earlier stage is responsible for, so a red gate names
the stage to go back to rather than a repair to apply.

THE THRESHOLDS ARE CONFIG, ALWAYS
----------------------------------
Every number this module compares against comes out of `assembly.gates` (and
the mix targets out of the profile), because a builder gate whose thresholds
live in its own source is a gate that gets edited to pass. The module knows
what to measure; the config knows what counts as good.

The profile is a CLI flag and is recorded in the report. `v1.0-MVP` is not a
weaker version of `v1.1-full`: the MVP cut is synthesis-light by construction
(it ships every zero-API row and whatever synthesis exists at the cut), so
grading it against 60/16/24 would fail a corpus that is exactly what it was
meant to be. The GATE always runs; the targets are data.

THE CHAIN IS A GATE, NOT A NOTE
--------------------------------
This module is the last thing that looks at the corpus before push.py writes a
card about it, so "were these rows decontaminated" has to be answerable HERE,
from the manifest chain alone. Each stage carries its predecessor's manifest
whole, so the walk is assemble -> split -> dedupe -> decontamination and every
link past the head must RECORD `"verified"`. Three states are red under
`gates.require_chain`, not one: the link is absent, the link is present with a
failed status, or the link is present and its `*_check` key was never written
at all. The third is the cheapest forgery of the three - stripping a key is
easier than corrupting a status - and tolerating it would let a card claim a
decontaminated dataset over rows nothing screened, which is the single most
expensive thing this build could ship. Only the head (`assemble`) may carry no
check: nothing upstream of it performed one.

WHAT EACH GATE IS FOR
----------------------
1. LENGTH - the p50/p90/p99 histogram the plan asked for and nothing had. The
   over-bucket count must be ZERO: assemble.py already dropped those rows, so a
   non-zero count here means assemble did not do what its manifest says.
2. MIX - source shares against the profile, +/- a tolerance in PERCENTAGE
   POINTS. A row's stream comes from an explicit config mapping; an unmapped
   source is RED and named, never bucketed by a guess.
3. TRACE - the >=80% reasoning share, a hard floor from the plan.
4. EMPTY-THINK - the share of rows opening with EXACTLY
   `replay.empty_think(open, close)`, by byte comparison rather than a regex: a
   pattern that tolerated whitespace would report a scaffold the model is not
   being taught.
5. DUP - a tripwire on the chain, not a re-deduplication. dedupe.py already
   ran; a duplicate share above the ceiling here means something after it
   reintroduced rows.
6. MARKUP - zero tolerance on `<|` in message content. Control tokens that
   survive into training teach the model to emit them, and the chat template
   is the only thing allowed to produce them.
7. LICENSE - every row carries a non-empty `_prov.license`, and the per-license
   counts are what push.py's card is built from.
8. CROSS-CODE - rows in a new-code shape whose provenance is a pre-transition
   corpus. REPORT-ONLY by default: the shape is a regex over prose, the
   false-positive direction costs real rows, and the number should be read
   once before it is armed.

Build:  python -m tuned.data.stats --config configs/data_law_v1.yaml
        [--profile v1.0-MVP] [--in-train PATH] [--in-eval PATH]
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tuned.data.assemble import MANIFEST_FILENAME as ASSEMBLE_MANIFEST_FILENAME
from tuned.data.assemble import token_length
from tuned.data.assemble import EVAL_FILENAME as ASSEMBLE_EVAL_FILENAME
from tuned.data.assemble import TRAIN_FILENAME as ASSEMBLE_TRAIN_FILENAME
# THE SAME REGEX curated.py filters with, imported rather than restated. A
# second copy would let the gate and the filter disagree about what a new-code
# mention looks like, and the gate is the one nobody would notice drifting.
from tuned.data.curated import _NEW_CODE_RE as NEW_CODE_RE
from tuned.data.decontaminate import item_key, row_answer, row_messages, row_prompt, row_prov
from tuned.data.replay import empty_think
from tuned.data.split import custody_of, custody_refusal

REPORT_FILENAME = "stats.json"
SUMMARY_FILENAME = "stats.md"

# 1  the first version. Eight measurements over assemble's output plus the
#    custody-chain walk, thresholds read from assembly.gates and mix targets
#    from the named profile, non-zero exit on any red gate.
STATS_VERSION = 1

GATE_LENGTH = "length"
GATE_MIX = "mix"
GATE_TRACE = "trace"
GATE_EMPTY_THINK = "empty_think"
GATE_DUP = "dup"
GATE_MARKUP = "markup"
GATE_LICENSE = "license"
GATE_CROSS_CODE = "cross_code"
GATE_CHAIN = "chain"
GATES = (
    GATE_CHAIN, GATE_LENGTH, GATE_MIX, GATE_TRACE, GATE_EMPTY_THINK,
    GATE_DUP, GATE_MARKUP, GATE_LICENSE, GATE_CROSS_CODE,
)

GREEN = "green"
RED = "red"
# Measured and published, but not gating - `report` is its own word because a
# green that never fails and a measurement nobody gates on are two different
# things, and one silence for both is how a disabled gate stops being visible.
REPORT = "report"

MARKUP = "<|"

# The chain, newest first: each stage's manifest carries the previous one under
# this key, and names the verification it performed under that one.
CHAIN = (
    ("assemble", None, None),
    ("split", "split", "split_check"),
    ("dedupe", "dedupe", "dedupe_check"),
    ("decontamination", "decontamination", "decontamination_check"),
)


@dataclass(frozen=True)
class Gate:
    name: str
    status: str
    summary: str
    detail: dict = field(default_factory=dict)

    @property
    def is_red(self) -> bool:
        return self.status == RED


# --------------------------------------------------------------------------
# Measurement.
# --------------------------------------------------------------------------

def percentile(values: Sequence[int], p: float) -> int:
    """Nearest-rank percentile: the ceil(p/100 * n)-th smallest, 1-indexed.

    Stated because there are half a dozen definitions and they disagree on
    small corpora - a linear-interpolation p99 over 40 rows is a number between
    two real rows, which is not what "the 99th percentile row is this long"
    means. Nearest-rank always names a row that exists.
    """
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def token_lengths(rows: Sequence[dict], tokenizer) -> list[int]:
    return [token_length(tokenizer, row.get("messages") or []) for row in rows]


def length_report(lengths: Sequence[int], limit: int) -> dict:
    return {
        "rows": len(lengths),
        "p50": percentile(lengths, 50),
        "p90": percentile(lengths, 90),
        "p99": percentile(lengths, 99),
        "max": max(lengths, default=0),
        "min": min(lengths, default=0),
        "limit": limit,
        "over_limit": sum(1 for length in lengths if length > limit),
    }


def stream_counts(rows: Sequence[dict], assembly) -> tuple[Counter, Counter]:
    """(rows per stream, rows per UNMAPPED source).

    Two counters rather than an "other" bucket: a source nobody mapped is a
    stream nobody sized, and folding it into a bucket is how a stream silently
    changes size.
    """
    counts: Counter = Counter()
    unmapped: Counter = Counter()
    for row in rows:
        source = row_prov(row).get("source")
        stream = assembly.stream_of(source)
        if stream is None:
            unmapped[str(source)] += 1
        else:
            counts[stream] += 1
    return counts, unmapped


def trace_count(rows: Sequence[dict]) -> int:
    return sum(1 for row in rows if bool(row_prov(row).get("reasoning")))


def empty_think_count(rows: Sequence[dict], think_open: str, think_close: str) -> int:
    """Rows whose assistant content STARTS WITH exactly the empty block.

    `startswith` on the byte-exact scaffold, never a regex: `<think>\\n</think>`
    and `<think>\\n\\n</think>` are different strings, the model is taught the
    second, and a pattern loose enough to accept both would report a scaffold
    that is not in the corpus.
    """
    scaffold = empty_think(think_open, think_close)
    return sum(1 for row in rows if row_answer(row).startswith(scaffold))


def duplicate_count(rows: Sequence[dict]) -> tuple[int, list[str]]:
    """(rows beyond the first of each content key, the worst offenders)."""
    keys = Counter(item_key(row_prompt(row), row_answer(row)) for row in rows)
    repeated = {key: n for key, n in keys.items() if n > 1}
    duplicates = sum(n - 1 for n in repeated.values())
    worst = [key for key, _ in sorted(repeated.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]
    return duplicates, worst


def markup_rows(rows: Sequence[dict]) -> list[int]:
    """Indexes of rows with `<|` anywhere in any message CONTENT.

    Content, not the rendered text: the chat template's own `<|im_start|>`
    markers are what the trainer needs, and a check over the rendered string
    would flag every row in the corpus.
    """
    return [
        index for index, row in enumerate(rows)
        if any(MARKUP in str(m.get("content") or "") for m in row_messages(row))
    ]


def license_counts(rows: Sequence[dict]) -> tuple[Counter, int]:
    counts: Counter = Counter()
    unlicensed = 0
    for row in rows:
        value = row_prov(row).get("license")
        if value and str(value).strip():
            counts[str(value)] += 1
        else:
            unlicensed += 1
    return counts, unlicensed


def is_old_code_source(source, old_code_sources: Sequence[str]) -> bool:
    """Same two-step lookup as the stream mapping: whole string, then dataset."""
    key = str(source or "")
    if not key:
        return False
    listed = set(old_code_sources)
    return key in listed or key.split(":", 1)[0] in listed


def cross_code_rows(rows: Sequence[dict], old_code_sources: Sequence[str]) -> list[int]:
    """Rows naming a new code whose provenance is a pre-transition corpus.

    Two provenance channels because two exist: `_prov.code_era == "ipc"`, which
    seeds.py sets on every seed it builds, and the configured list of corpora
    that predate the 2024 transition outright.
    """
    hits = []
    for index, row in enumerate(rows):
        prov = row_prov(row)
        old = str(prov.get("code_era") or "").lower() == "ipc" or is_old_code_source(
            prov.get("source"), old_code_sources
        )
        if old and NEW_CODE_RE.search(row_answer(row)):
            hits.append(index)
    return hits


def chain_links(manifest) -> list[dict]:
    """Walk assemble's manifest back to decontamination, naming each link.

    Every stage carries its predecessor's manifest WHOLE precisely so this walk
    exists: a summary would answer "how many rows" and not "was this the file
    that was screened".

    `check_key` travels with the link because the gate has to be able to NAME
    the record that is missing. `None` marks the head, the one link no stage
    upstream could have verified.
    """
    links = []
    node = manifest if isinstance(manifest, dict) else None
    for name, key, check_key in CHAIN:
        if key is None:
            present = isinstance(node, dict) and bool(node)
            links.append({"stage": name, "present": present, "check": None,
                          "check_key": None})
            continue
        parent = node
        node = (parent or {}).get(key) if isinstance(parent, dict) else None
        links.append({
            "stage": name,
            "present": isinstance(node, dict) and bool(node),
            "check": ((parent or {}).get(check_key) or {}).get("status")
            if isinstance(parent, dict) else None,
            "check_key": check_key,
        })
    return links


# --------------------------------------------------------------------------
# The gates. Every one reads its threshold from config.
# --------------------------------------------------------------------------

def gate_chain(links: Sequence[dict], *, required: bool) -> Gate:
    """Present, verified, and SAID SO - all three, for every link past the head.

    An unrecorded check is its own fault and not a tolerated `None`: a manifest
    carried forward with its `*_check` keys stripped is a chain nobody walked,
    and it is the easiest damage to inflict on the record. The head link is the
    only one exempt, because assemble is where the walk starts.
    """
    missing = [link["stage"] for link in links if not link["present"]]
    graded = [link for link in links if link["present"] and link["check_key"]]
    unrecorded = [link for link in graded if link["check"] is None]
    unverified = [link["stage"] for link in graded if link["check"] not in (None, "verified")]
    detail = {
        "links": list(links), "missing": missing, "unverified": unverified,
        "unrecorded": [link["stage"] for link in unrecorded],
    }
    if not missing and not unrecorded and not unverified:
        return Gate(GATE_CHAIN, GREEN,
                    "custody complete: assembled, split, deduped, decontaminated", detail)
    faults = []
    if missing:
        faults.append(f"absent: {', '.join(missing)}")
    if unrecorded:
        faults.append("verification NEVER RECORDED for " + ", ".join(
            f"{link['stage']} (no {link['check_key']})" for link in unrecorded
        ))
    if unverified:
        faults.append(f"unverified: {', '.join(unverified)}")
    return Gate(
        GATE_CHAIN, RED if required else REPORT,
        f"custody chain incomplete ({'; '.join(faults)})", detail,
    )


def gate_length(report: dict) -> Gate:
    """Reported always; RED only on an over-limit row.

    A non-zero count here is not a corpus problem, it is assemble.py claiming
    a drop it did not make - which is why the message points at that stage
    instead of at a threshold.
    """
    summary = (
        f"p50 {report['p50']} / p90 {report['p90']} / p99 {report['p99']} tokens, "
        f"max {report['max']}, limit {report['limit']}"
    )
    if report["over_limit"] == 0:
        return Gate(GATE_LENGTH, GREEN, summary, report)
    return Gate(
        GATE_LENGTH, RED,
        f"{summary} - {report['over_limit']} row(s) OVER the bucket, which assemble.py's "
        f"manifest says it dropped",
        report,
    )


def gate_mix(counts: Counter, unmapped: Counter, targets: dict, *, total: int,
             tolerance_pp: float, profile: str) -> Gate:
    shares = {stream: (counts.get(stream, 0) / total if total else 0.0) for stream in targets}
    # ROUNDED BEFORE THE COMPARISON, and that is not cosmetic: 62/100 against a
    # 0.60 target is 2.0000000000000018 percentage points in binary floating
    # point, so a bare `> 2.0` reds a corpus that is exactly on the tolerance.
    # Nine places is far below any threshold anyone would write and far above
    # the noise.
    misses = {
        stream: round((shares[stream] - target) * 100, 2)
        for stream, target in targets.items()
        if round(abs(shares[stream] - target) * 100, 9) > tolerance_pp
    }
    detail = {
        "profile": profile,
        "targets": dict(targets),
        "counts": dict(counts),
        "shares": {k: round(v, 4) for k, v in shares.items()},
        "tolerance_pp": tolerance_pp,
        "misses_pp": misses,
        "unmapped_sources": dict(unmapped),
    }
    body = ", ".join(
        f"{stream} {shares[stream]:.1%} (target {targets[stream]:.0%})"
        for stream in sorted(targets)
    )
    if unmapped:
        named = ", ".join(f"{s} x{n}" for s, n in sorted(unmapped.items()))
        return Gate(GATE_MIX, RED,
                    f"{len(unmapped)} source(s) map to no stream: {named}", detail)
    if misses:
        named = ", ".join(f"{stream} {delta:+}pp" for stream, delta in sorted(misses.items()))
        return Gate(GATE_MIX, RED,
                    f"{body} - outside +/-{tolerance_pp}pp: {named}", detail)
    return Gate(GATE_MIX, GREEN, f"{body}, all within +/-{tolerance_pp}pp", detail)


def gate_share(name: str, count: int, total: int, *, floor: float | None = None,
               ceiling: float | None = None, label: str) -> Gate:
    """One share against a floor, a ceiling, or a band - the three read alike."""
    share = count / total if total else 0.0
    detail = {"count": count, "total": total, "share": round(share, 4),
              "floor": floor, "ceiling": ceiling}
    bounds = []
    if floor is not None:
        bounds.append(f">= {floor:.1%}")
    if ceiling is not None:
        bounds.append(f"<= {ceiling:.1%}")
    body = f"{label} {share:.1%} of {total} ({' and '.join(bounds)})"
    # NOT rounded before the comparison, unlike gate_mix, and the difference is
    # measured rather than stylistic. Here the share is one DIVISION of two
    # integers, and IEEE division is correctly rounded: when count/total equals
    # the bound as a rational, `count / total` and `float("0.18")` are the same
    # double, bit for bit. Searched exhaustively over every total from 2 to
    # 4,000 for all four shipped bounds - no (count, total) produces a share
    # that differs from its bound in floating point while equalling it in
    # arithmetic, so a guard here could never change a verdict and would be a
    # branch nothing can reach. gate_mix subtracts and then scales by 100,
    # which is where the noise actually comes from (62/100 against 0.60 is
    # 2.0000000000000018 percentage points), and that is where the guard is.
    #
    # The plan's bounds are INCLUSIVE (">= 80%", "18-22%"), so a corpus sitting
    # exactly on one passes.
    if floor is not None and share < floor:
        return Gate(name, RED, f"{body} - BELOW the floor", detail)
    if ceiling is not None and share > ceiling:
        return Gate(name, RED, f"{body} - ABOVE the ceiling", detail)
    return Gate(name, GREEN, body, detail)


def gate_markup(offenders: Sequence[int], total: int, *, enabled: bool) -> Gate:
    detail = {"rows": len(offenders), "total": total, "first_rows": list(offenders[:5])}
    if not enabled:
        return Gate(GATE_MARKUP, REPORT,
                    f"{len(offenders)} row(s) carry '{MARKUP}' (gate disabled)", detail)
    if offenders:
        return Gate(GATE_MARKUP, RED,
                    f"{len(offenders)} row(s) carry '{MARKUP}' in message content - zero is "
                    f"the only acceptable count (rows {list(offenders[:5])})", detail)
    return Gate(GATE_MARKUP, GREEN, f"no '{MARKUP}' in any of {total} rows", detail)


def gate_license(counts: Counter, unlicensed: int, total: int, *, required: bool) -> Gate:
    detail = {"counts": dict(counts), "unlicensed": unlicensed, "total": total}
    named = ", ".join(f"{name} {n}" for name, n in sorted(counts.items())) or "none"
    if unlicensed and required:
        return Gate(GATE_LICENSE, RED,
                    f"{unlicensed} of {total} rows carry NO license - the dataset card "
                    f"cannot be written over them ({named})", detail)
    status = GREEN if not unlicensed else REPORT
    return Gate(GATE_LICENSE, status, f"{total} rows: {named}", detail)


def gate_cross_code(hits: Sequence[int], total: int, *, red: bool) -> Gate:
    detail = {"rows": len(hits), "total": total, "first_rows": list(hits[:5])}
    body = (
        f"{len(hits)} of {total} rows name a new code (BNS/BNSS/BSA) with pre-transition "
        f"provenance"
    )
    if hits and red:
        return Gate(GATE_CROSS_CODE, RED, body, detail)
    return Gate(GATE_CROSS_CODE, GREEN if not hits else REPORT,
                body + ("" if red else " [report-only]"), detail)


def measure(rows: Sequence[dict], *, cfg, tokenizer, profile: str,
            manifest: dict | None) -> tuple[list[Gate], dict]:
    """Every gate, in GATES order, plus the raw measurements behind them."""
    assembly = cfg.assembly
    gates_cfg = assembly.gates
    targets = assembly.targets(profile)
    total = len(rows)

    lengths = token_lengths(rows, tokenizer)
    length = length_report(lengths, cfg.max_seq_length)
    counts, unmapped = stream_counts(rows, assembly)
    traces = trace_count(rows)
    empties = empty_think_count(rows, cfg.think_open, cfg.think_close)
    duplicates, worst = duplicate_count(rows)
    offenders = markup_rows(rows)
    licenses, unlicensed = license_counts(rows)
    cross = cross_code_rows(rows, gates_cfg.old_code_sources)
    links = chain_links(manifest)

    gates = [
        gate_chain(links, required=gates_cfg.require_chain),
        gate_length(length),
        gate_mix(counts, unmapped, targets, total=total,
                 tolerance_pp=gates_cfg.mix_tolerance_pp, profile=profile),
        gate_share(GATE_TRACE, traces, total, floor=gates_cfg.trace_floor,
                   label="reasoning traces"),
        gate_share(GATE_EMPTY_THINK, empties, total, floor=gates_cfg.empty_think_min,
                   ceiling=gates_cfg.empty_think_max, label="byte-exact empty think"),
        gate_share(GATE_DUP, duplicates, total, ceiling=gates_cfg.dup_ceiling,
                   label="exact duplicates"),
        gate_markup(offenders, total, enabled=gates_cfg.markup),
        gate_license(licenses, unlicensed, total, required=gates_cfg.require_license),
        gate_cross_code(cross, total, red=gates_cfg.cross_code_red),
    ]
    measurements = {
        "rows": total,
        "length": length,
        "duplicate_keys": worst,
    }
    return gates, measurements


# --------------------------------------------------------------------------
# The report.
# --------------------------------------------------------------------------

def report_of(gates: Sequence[Gate], measurements: dict, *, profile: str, sides: dict,
              inputs: Sequence[str], custody: dict, tokenizer_id: dict) -> dict:
    from tuned.data.store import utcnow

    red = [g.name for g in gates if g.is_red]
    return {
        "stage": "stats",
        "stats_version": STATS_VERSION,
        "at": utcnow(),
        "inputs": list(inputs),
        # The profile is a CLI flag, so the report says which targets these
        # numbers were graded against - a number without its target is a
        # number nobody can act on.
        "profile": profile,
        "tokenizer": tokenizer_id,
        "sides": dict(sides),
        "measurements": measurements,
        "gates": {
            g.name: {"status": g.status, "summary": g.summary, "detail": g.detail}
            for g in gates
        },
        "red": red,
        "verdict": "red" if red else "green",
        "assemble_check": custody,
    }


_MARKS = {GREEN: "PASS", RED: "FAIL", REPORT: "note"}


def summary_of(report: dict) -> str:
    """The human-readable half - also what push.py's card is built from."""
    lines = [
        "# law_v1 builder gate",
        "",
        f"- verdict: **{report['verdict'].upper()}**",
        f"- profile: {report['profile']}",
        f"- rows: {report['measurements']['rows']} "
        f"(train {report['sides'].get('train', 0)}, eval {report['sides'].get('eval', 0)})",
        f"- tokenizer: {report['tokenizer'].get('repo')} @ "
        f"{report['tokenizer'].get('revision')}",
        "",
        "| gate | status | measurement |",
        "| --- | --- | --- |",
    ]
    for name in GATES:
        gate = report["gates"][name]
        lines.append(f"| {name} | {_MARKS[gate['status']]} | {gate['summary']} |")
    if report["red"]:
        lines += ["", f"RED: {', '.join(report['red'])}. Nothing downstream should publish "
                      f"this corpus until each is answered at the stage that produced it."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None, *, tokenizer=None) -> int:
    import argparse

    from tuned.data.assemble import load_tokenizer
    from tuned.data.config import load_build_config
    from tuned.data.decontaminate import write_manifest
    from tuned.data.jsonl import read_jsonl
    from tuned.data.paths import build_paths
    from tuned.data.store import Store

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data_law_v1.yaml")
    parser.add_argument("--profile", default=None,
                        help="mix profile to grade against (default assembly.default_profile)")
    parser.add_argument("--in-train", default=None,
                        help=f"default out/{ASSEMBLE_TRAIN_FILENAME}")
    parser.add_argument("--in-eval", default=None, help=f"default out/{ASSEMBLE_EVAL_FILENAME}")
    parser.add_argument("--report", default=None, help=f"default out/{REPORT_FILENAME}")
    args = parser.parse_args(argv)

    cfg = load_build_config(args.config)
    paths = build_paths(cfg.build.workdir).ensure()
    in_train = Path(args.in_train) if args.in_train else paths.out_dir / ASSEMBLE_TRAIN_FILENAME
    in_eval = Path(args.in_eval) if args.in_eval else paths.out_dir / ASSEMBLE_EVAL_FILENAME
    report_path = Path(args.report) if args.report else paths.out_dir / REPORT_FILENAME
    profile = args.profile or cfg.assembly.default_profile
    try:
        cfg.assembly.targets(profile)
    except KeyError as exc:
        print(f"stats REFUSES TO RUN: {exc.args[0]}")
        return 2

    missing = [str(p) for p in (in_train, in_eval) if not p.exists()]
    if missing:
        print(
            f"no such input: {', '.join(missing)}\n"
            f"  run: python -m tuned.data.assemble --config {args.config}"
        )
        return 2

    upstream, custody = custody_of(
        [in_train, in_eval], manifest_filename=ASSEMBLE_MANIFEST_FILENAME
    )
    if upstream is None:
        print(custody_refusal(
            custody, stage="stats",
            remedy=f"python -m tuned.data.assemble --config {args.config}",
        ))
        return 2

    if tokenizer is None:
        tokenizer = load_tokenizer(cfg)

    train_rows = list(read_jsonl(in_train))
    eval_rows = list(read_jsonl(in_eval))
    rows = train_rows + eval_rows
    gates, measurements = measure(
        rows, cfg=cfg, tokenizer=tokenizer, profile=profile, manifest=upstream
    )
    report = report_of(
        gates, measurements,
        profile=profile,
        sides={"train": len(train_rows), "eval": len(eval_rows)},
        inputs=[str(in_train), str(in_eval)],
        custody=custody,
        tokenizer_id=(upstream.get("tokenizer") or {}),
    )
    write_manifest(report_path, report)
    summary_path = report_path.parent / SUMMARY_FILENAME
    summary_path.write_text(summary_of(report), encoding="utf-8")
    store = Store.open(paths.state_db)
    try:
        store.log_event("stats", report)
    finally:
        store.close()

    print(summary_of(report))
    print(f"report -> {report_path}")
    print(f"       -> {summary_path}")
    if report["red"]:
        print(
            f"REFUSED: {len(report['red'])} gate(s) red - {', '.join(report['red'])}. "
            f"This corpus is not ready to push."
        )
        return 1
    return 0


if __name__ == "__main__":
    import os
    import sys

    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
