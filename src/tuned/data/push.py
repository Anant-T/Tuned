"""The last assembly-layer module: stats' green output -> an HF dataset repo.

Input is `out/law_v1_train.jsonl` and `out/law_v1_eval.jsonl` - the exact pair
stats.py graded - plus stats' own report (`out/stats.json`) and the manifest
chain that report was built from. Output is a private (by default) HuggingFace
dataset repo carrying those two files, a `README.md` dataset card and a
`build_manifest.json`, plus the repo's resulting revision (commit sha) printed
for `training/scripts/pin_dataset.py` to pick up. push.py's own job ends there - pinning
the revision into the training config is that script's job, not this one's.

THE TERMINAL GATE GATES THE PUSH
---------------------------------
stats.py is the last thing that looks at the corpus before a card gets written
about it, so push.py refuses to run unless ITS report exists, is green
(`red == []`), AND its custody chain is complete - three separate checks, not
one. The third is deliberately independent of `assembly.gates.require_chain`:
that toggle only controls whether stats.py REDS a broken chain or merely notes
it, because an operator mid-build may want to see the rest of the report
before deciding what to do about a hole. push.py is not mid-build - it is
about to write a card that CLAIMS a decontaminated dataset - so it treats an
incomplete chain as disqualifying regardless of how stats.py was configured to
grade it. None of these three is a `--force` flag; every one is a named
refusal that writes nothing.

A FOURTH CHECK, AT PUSH'S OWN BOUNDARY: stats.py graded two specific files: two
specific files. Nothing stops `assemble.py` from running again - a config
edit, a code change - after stats.py graded its output and before push.py
runs. stats' own report already carries the sha256 of exactly the bytes it
measured (`assemble_check.input_sha256`, the custody record produced when
stats.py verified ITS OWN input against assemble.py's manifest); push.py
re-hashes the files it is about to upload and compares. A mismatch is a
refusal, same as the rest of this chain: content-bound, never path-bound.

THE CARD IS MEASURED, NEVER INVENTED
--------------------------------------
Every number in the card comes out of stats' report, the decontamination
manifest, or the exact rows about to be uploaded - row counts and mix from
stats' `mix` gate, the license table from its `license` gate, source datasets
and their licenses read directly off the rows (config maps a source to a MIX
STREAM, never to a license - a source's real license is a fact about the data
and reading it twice is how the two copies drift), and the decontamination
statement from `decontamination.json` (which eval sets were screened, which
were waived, and the semantic layer's own per-script gaps - "screened" scripts
only, no per-script number invented for a script the manifest never measured).
If a number the card needs is missing from the inputs, rendering RAISES
(`CardDataMissing`) rather than writing a placeholder or a guess.

SECRETS
-------
`HF_TOKEN` is read from the environment via `providers.load_dotenv_keys` (the
worktree-root `.env`), the same seam every other API-key read in this build
uses. A missing token on a non-dry-run is a named refusal, never a crash and
never a `--force`. NOTHING in this module logs, prints or writes the token
value anywhere - not the refusal message, not the manifest, not the card.

THE LIVE PUSH IS A SEAM
-------------------------
`main(..., hub_client=...)` takes an injectable client exposing three methods
(`ensure_repo`, `current_manifest`, `upload` - see `RealHubClient`'s
docstring for the exact contract), the same shape acquire.py's `snapshot_fn`
and assemble.py's `tokenizer` parameters use. `RealHubClient` is the only
place `huggingface_hub` is imported, and it is imported lazily so the rest of
this module - and every `--dry-run` test - runs without the library
installed. A real-network test constructs `RealHubClient` directly and marks
itself `@pytest.mark.live`; everything else runs against a fake.

IDEMPOTENCE
-----------
A re-run against unchanged inputs is a no-op, not a duplicate upload: before
touching the repo, push.py asks the client for the repo's current
`build_manifest.json` (`current_manifest`) and compares outputs against the
manifest it just built locally. An exact match prints the existing revision
and returns without calling `ensure_repo` or `upload` at all. The comparison
is over `law_v1_train.jsonl`/`law_v1_eval.jsonl`/`README.md` - corpus bytes
AND the card that describes them - never `stats.json`, which embeds the
stats report's own `at` timestamp directly (it IS that report), so a
stats.py re-run over byte-identical rows would still change ITS bytes on
every push. README carries no such timestamp (the card states the stats
verdict, never the report's `at` - see THE CARD IS MEASURED below), so it is
a pure function of corpus + config + code: a stats-only re-run over an
unchanged corpus is still a no-op, and a genuinely different card (a new
`push.card_extra`, a card-rendering fix) is NOT silently swallowed - round
2's own regression, caught and closed (N-1). A malformed remote manifest
(data from outside this machine, not from this run) is a named refusal
(`RemoteManifestCorrupt`), never a crash.

Build:  python -m tuned.data.push --config data/configs/data_law_v1.yaml
        [--dry-run] [--report PATH] [--out DIR]
"""

import os
from collections.abc import Sequence
from pathlib import Path

from tuned.data.acquire import sha256_file
from tuned.data.assemble import EVAL_FILENAME, TRAIN_FILENAME
from tuned.data.assemble import MANIFEST_FILENAME as ASSEMBLE_MANIFEST_FILENAME
from tuned.data.decontaminate import EVAL_OK
from tuned.data.decontaminate import MANIFEST_FILENAME as DECON_MANIFEST_FILENAME
from tuned.data.decontaminate import write_manifest
from tuned.data.dedupe import MANIFEST_FILENAME as DEDUPE_MANIFEST_FILENAME
from tuned.data.jsonl import read_jsonl
from tuned.data.providers import load_dotenv_keys
from tuned.data.split import MANIFEST_FILENAME as SPLIT_MANIFEST_FILENAME
from tuned.data.split import read_manifest
from tuned.data.stats import GREEN
from tuned.data.stats import REPORT_FILENAME as STATS_REPORT_FILENAME

README_FILENAME = "README.md"
MANIFEST_FILENAME = "build_manifest.json"

# 1  the first version. Uploads the two assembled JSONL sides, a card measured
#    off stats' report and the decontamination manifest, stats' own report,
#    and a build_manifest.json carrying the full custody chain, every
#    reachable module version and the sha256 of every uploaded file; refuses
#    on a red or absent stats report, an incomplete custody chain
#    (independent of require_chain), a decontamination manifest that
#    disagrees with the chain's own record of it, bytes that no longer match
#    what stats.py measured, or a malformed remote manifest; idempotent
#    against the repo's own build_manifest.json (corpus + card bytes, never
#    stats.json's) through an injectable hub-client seam.
PUSH_VERSION = 1


class CardDataMissing(RuntimeError):
    """A number the card needs was not in the inputs. Refuse, never invent."""


# --------------------------------------------------------------------------
# The refusal checks. Every one returns a message, or None to proceed.
# --------------------------------------------------------------------------

def chain_faults(report: dict) -> list[str]:
    """The custody-chain complaints stats' own `chain` gate already computed.

    Read from the SAME detail `gate_chain` published in the report, not a
    re-walk of the manifest tree: two functions independently deciding "is
    the chain complete" from the same input could disagree, and the one
    nobody would notice drifting is this one. Deliberately blind to the
    gate's own status (RED vs REPORT) - see the module docstring for why
    push.py enforces this regardless of `assembly.gates.require_chain`.
    """
    detail = ((report.get("gates") or {}).get("chain") or {}).get("detail") or {}
    faults = []
    missing = detail.get("missing") or []
    unrecorded = detail.get("unrecorded") or []
    unverified = detail.get("unverified") or []
    if missing:
        faults.append(f"absent: {', '.join(missing)}")
    if unrecorded:
        faults.append("verification never recorded for " + ", ".join(unrecorded))
    if unverified:
        faults.append(f"unverified: {', '.join(unverified)}")
    return faults


def stats_refusal(report: dict | None, *, report_path: Path, config_path: str) -> str | None:
    """The three checks the module docstring names, in order. None means go."""
    remedy = f"python -m tuned.data.stats --config {config_path}"
    if report is None:
        return (
            f"push REFUSES TO RUN: no stats report at {report_path}.\n"
            f"  remedy: {remedy}\n"
            f"  nothing was uploaded; no repo carries a push stamp."
        )
    red = list(report.get("red") or [])
    if red or report.get("verdict") != GREEN:
        named = ", ".join(red) if red else "verdict is not green"
        return (
            f"push REFUSES TO RUN: stats' report is RED ({named}).\n"
            f"  report: {report_path}\n"
            f"  remedy: fix the stage that produced each red gate, then re-run: {remedy}\n"
            f"  nothing was uploaded; a red corpus does not get a dataset card."
        )
    faults = chain_faults(report)
    if faults:
        return (
            f"push REFUSES TO RUN: the custody chain is incomplete ({'; '.join(faults)}), "
            f"even though stats' own verdict is green.\n"
            f"  report: {report_path}\n"
            f"  remedy: re-run the stage(s) named above so every link records its own "
            f"verification, then re-run: {remedy}\n"
            f"  nothing was uploaded; a card cannot claim a decontaminated dataset over rows "
            f"nothing screened."
        )
    return None


def bytes_refusal(report: dict, *, train_path: Path, eval_path: Path) -> str | None:
    """Binds the bytes about to be uploaded to the bytes stats.py measured.

    `report['assemble_check']` is the custody record stats.py itself computed
    (split.custody_of, generalised) when it verified ITS OWN input against
    assemble.py's manifest - it already carries the sha256 of these two files
    under their own path strings. Re-hashing here and comparing is the same
    content-bound check every stage in this chain performs on its own input;
    push.py's input is the pair stats.py just graded, so this is that check at
    push.py's own boundary.
    """
    recorded = ((report.get("assemble_check") or {}).get("input_sha256")) or {}
    for path in (train_path, eval_path):
        expected = recorded.get(str(path))
        if not expected:
            return (
                f"push REFUSES TO RUN: stats' report carries no recorded digest for {path}.\n"
                f"  remedy: re-run stats.py against these exact files\n"
                f"  nothing was uploaded."
            )
        actual = sha256_file(path)
        if actual != expected:
            return (
                f"push REFUSES TO RUN: {path} does not match the bytes stats.py measured "
                f"(recorded {expected[:12]}..., found {actual[:12]}...) - something wrote "
                f"this file again after stats.py graded it.\n"
                f"  remedy: re-run stats.py against the current output, then push again\n"
                f"  nothing was uploaded."
            )
    return None


def decon_chain_faults(decon: dict | None, chain: dict | None) -> list[str]:
    """Cross-checks the standalone decontamination.json push.py reads for the
    card (`decon`) against the SAME record dedupe.py's custody chain already
    carries forward (`chain["split"]["dedupe"]["decontamination"]`) - four
    fields dedupe.manifest_of's upstream_summary already carries whole:
    `at`, `decon_version`, `counts`, and `eval_sets` (narrowed there to
    `status`/`allowed_missing`/`items` per key, which is why the comparison
    below narrows `decon`'s copy the same way before comparing).

    A post-hoc edit of decontamination.json AFTER the chain was built - the
    exact failure mode a disclosed hole silently vanishing needs - changes
    at least one of these four fields (a hole cleared changes its
    `eval_sets` entry), so this binds the file's identity the way every
    other link in this custody chain already is.

    NOT bound: `semantic_scripts` itself. dedupe.py's own summary never
    carried it forward (that is why build_manifest's own "decontamination"
    field exists at all - see build_manifest's comment), so there is nothing
    here to cross-check it against; an edit that touches ONLY
    `semantic_scripts` and none of the four fields above is a residual gap
    this check cannot see. Closing that completely needs dedupe.py to carry
    decontamination.json's own sha256 forward - an upstream change, out of
    this module's scope (task 15 round 2, N-3).
    """
    chain_decon = (((chain or {}).get("split") or {}).get("dedupe") or {}).get("decontamination")
    if chain_decon is None:
        # Nothing to cross-check against. Not this function's refusal to
        # make - an absent/broken chain link is stats_refusal's job (the
        # chain-completeness gate), which runs before this one ever could.
        return []
    decon = decon or {}
    faults = []
    if decon.get("at") != chain_decon.get("at"):
        faults.append("at")
    if decon.get("decon_version") != chain_decon.get("decon_version"):
        faults.append("decon_version")
    if decon.get("counts") != chain_decon.get("counts"):
        faults.append("counts")
    narrowed = {
        key: {"status": v.get("status"), "allowed_missing": v.get("allowed_missing"),
              "items": v.get("items")}
        for key, v in (decon.get("eval_sets") or {}).items()
    }
    if narrowed != (chain_decon.get("eval_sets") or {}):
        faults.append("eval_sets")
    return faults


# --------------------------------------------------------------------------
# Module versions, read straight off the files this chain writes.
# --------------------------------------------------------------------------

def module_versions(out_dir: Path, report: dict) -> dict[str, int | str | None]:
    """Every module version reachable from push.py's own inputs.

    NOT a real version for extract.py: `extract_version` lives per-document in
    the SQLite store (store.py's `document` table), and nothing in the
    file-based manifest chain push.py reads - decontamination.json,
    dedupe.json, split.json, assemble.json, stats.json - carries it forward.
    Row builders do not even agree on one `_prov` shape to read it off of
    (curated.py and replay.py write `license`; `decontaminate.generated_rows`
    does not - see `license_rows`, which has to account for that gap), but
    NONE of them writes extract_version either, so push.py that claimed one
    would be inventing it regardless.

    That absence is named here rather than left silent: `build_manifest.json`
    is the only provenance file that leaves the machine, and a reader on the
    Hub cannot otherwise tell "extraction was unversioned" from "extraction's
    version lives somewhere this file cannot reach." `version_faults` (the
    caller in `main()`) only refuses on `None`, so a string here changes
    nothing about the terminal gate.
    """
    decon = read_manifest(out_dir / DECON_MANIFEST_FILENAME)
    dedupe = read_manifest(out_dir / DEDUPE_MANIFEST_FILENAME)
    split = read_manifest(out_dir / SPLIT_MANIFEST_FILENAME)
    assemble = read_manifest(out_dir / ASSEMBLE_MANIFEST_FILENAME)
    return {
        "decontaminate": (decon or {}).get("decon_version"),
        "dedupe": (dedupe or {}).get("dedupe_version"),
        "split": (split or {}).get("split_version"),
        "assemble": (assemble or {}).get("assemble_version"),
        "stats": report.get("stats_version"),
        "push": PUSH_VERSION,
        "extract": "per-document in the build store (document.extract_version); "
                   "not reachable from the file chain",
    }


# --------------------------------------------------------------------------
# The card. Every value below comes from an argument; nothing is looked up
# again from disk, so a caller that wants "exactly what was rendered" can
# read it straight off this function's own inputs.
# --------------------------------------------------------------------------

def _gate_detail(report: dict, gate: str) -> dict:
    gates = report.get("gates")
    if not isinstance(gates, dict) or gate not in gates:
        raise CardDataMissing(f"stats' report carries no {gate!r} gate")
    detail = (gates[gate] or {}).get("detail")
    if not isinstance(detail, dict):
        raise CardDataMissing(f"stats' report's {gate!r} gate carries no detail")
    return detail


def license_rows(report: dict) -> list[tuple[str, int]]:
    """The Licenses table - every row stats measured, none dropped.

    `detail["unlicensed"]` sits in the SAME dict `detail["counts"]` comes
    from (`stats.gate_license`); rendering one and discarding the other is
    how a card under-reports its own corpus - the Source datasets table
    (`source_license_rows`) already renders these rows as `unknown`, so a
    Licenses table that silently sums short of `total` disagrees with a
    table on the same card. Live path: `decontaminate.generated_rows` writes
    `_prov` with no `license` key at all, so this is not hypothetical - it is
    the entire `grounded_synthesis` stream whenever `require_license: false`.
    """
    detail = _gate_detail(report, "license")
    counts = detail.get("counts")
    if not counts:
        raise CardDataMissing("stats' report carries no license counts")
    rows = dict(counts)
    unlicensed = detail.get("unlicensed") or 0
    if unlicensed:
        rows["unlicensed"] = unlicensed
    total = detail.get("total")
    measured = sum(rows.values())
    if total is not None and measured != total:
        raise CardDataMissing(
            f"license rows ({dict(sorted(rows.items()))}) sum to {measured}, not the "
            f"{total} rows stats' report measured - the license gate's counts and "
            f"unlicensed total do not add up to its own total"
        )
    return sorted(rows.items())


def mix_rows(report: dict) -> list[tuple[str, int, float, float]]:
    detail = _gate_detail(report, "mix")
    counts, shares, targets = detail.get("counts"), detail.get("shares"), detail.get("targets")
    if not targets:
        raise CardDataMissing("stats' report carries no mix targets")
    return [
        (stream, (counts or {}).get(stream, 0), (shares or {}).get(stream, 0.0), target)
        for stream, target in sorted(targets.items())
    ]


def source_license_rows(rows: Sequence[dict]) -> list[tuple[str, str, int]]:
    """(source, license, rows), measured off the rows about to ship.

    config.py maps a source to a MIX STREAM, never to a license - a source's
    real license is read off the rows themselves, the same field the license
    gate itself reads (`_prov.license`), so this table and that gate can never
    disagree about what a row's license is.
    """
    from collections import Counter

    counts: Counter = Counter()
    for row in rows:
        prov = row.get("_prov") or {}
        source = str(prov.get("source") or "unknown")
        license_ = str(prov.get("license") or "unknown")
        counts[(source, license_)] += 1
    if not counts:
        raise CardDataMissing("no rows to derive a source/license table from")
    return [(source, license_, n) for (source, license_), n in sorted(counts.items())]


def decon_sections(
    decon: dict | None,
) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """(screened eval sets, waived/hole eval sets, per-script semantic gaps).

    Exactly as decontamination.json records them: a set or a script the
    manifest is silent about produces nothing here rather than an invented
    "unknown" row. `decontaminate.refusals` already guarantees that by the
    time a decontamination.json exists at all, every eval set is either `ok`
    or explicitly `allowed_missing` - anything else would have refused the
    run before this file was ever written.
    """
    eval_sets = (decon or {}).get("eval_sets") or {}
    screened: list[tuple] = []
    holes: list[tuple] = []
    for key, entry in sorted(eval_sets.items()):
        status = entry.get("status")
        row = (key, status, bool(entry.get("allowed_missing")), entry.get("items"))
        (screened if status == EVAL_OK else holes).append(row)
    scripts = (decon or {}).get("semantic_scripts") or {}
    gaps = [
        (script, entry.get("control"), entry.get("eval_items"))
        for script, entry in sorted(scripts.items())
        # Mirrors decontaminate.py's own reader (decontaminate.py:3336): a
        # hole needs BOTH "not screened" AND at least one item it could have
        # caught. `not entry.get("screened")` alone also catches SCRIPT_NONE
        # (whose own manifest text ends "and not a hole" - no letters, so no
        # eval question in it to miss) and a script whose control PASSED but
        # held no eval items at all (nothing built, nothing to have missed) -
        # neither is a gap, and printing them under "Per-script gaps" states
        # a reason that contradicts the manifest.
        if not entry.get("screened") and (entry.get("eval_items") or entry.get("unscreened_rows"))
    ]
    return screened, holes, gaps


def render_card(
    *, report: dict, decon: dict | None, rows: Sequence[dict], push_cfg, versions: dict,
    extra: str | None = None,
) -> str:
    """The dataset card. Raises CardDataMissing rather than leave a gap."""
    rows_total = report.get("measurements", {}).get("rows")
    if rows_total is None:
        raise CardDataMissing("stats' report carries no row count")
    sides = report.get("sides")
    if not sides:
        raise CardDataMissing("stats' report carries no sides (train/eval split)")
    profile = report.get("profile")
    if profile is None:
        raise CardDataMissing("stats' report carries no profile")
    tokenizer = report.get("tokenizer")
    if not tokenizer:
        raise CardDataMissing("stats' report carries no tokenizer")
    licenses = license_rows(report)
    mix = mix_rows(report)
    sources = source_license_rows(rows)
    screened, holes, gaps = decon_sections(decon)
    decon_version = (decon or {}).get("decon_version")
    if decon_version is None:
        raise CardDataMissing("the decontamination manifest carries no decon_version")

    lines: list[str] = []
    lines.append(f"# {push_cfg.repo_id}")
    lines.append("")
    lines.append(
        "Indian-law supervised fine-tuning data for the law_v1 lane: reasoning-traced "
        "instruction/response pairs over IPC/CrPC/Evidence Act and the 2024 BNS/BNSS/BSA "
        "recodification, built by tuned's law_v1 data-curation pipeline."
    )
    lines.append("")
    lines.append("## Rows")
    lines.append("")
    lines.append(
        f"- total: {rows_total} (train {sides.get('train', 0)}, eval {sides.get('eval', 0)})"
    )
    lines.append(f"- mix profile graded: `{profile}`")
    lines.append(f"- tokenizer: `{tokenizer.get('repo')}` @ `{tokenizer.get('revision')}`")
    lines.append("")
    lines.append("## Mix")
    lines.append("")
    lines.append("| stream | rows | share | target |")
    lines.append("| --- | --- | --- | --- |")
    for stream, count, share, target in mix:
        lines.append(f"| {stream} | {count} | {share:.1%} | {target:.0%} |")
    lines.append("")
    lines.append("## Licenses")
    lines.append("")
    lines.append("| license | rows |")
    lines.append("| --- | --- |")
    for license_, count in licenses:
        lines.append(f"| {license_} | {count} |")
    lines.append("")
    lines.append("## Source datasets")
    lines.append("")
    lines.append("| source | license | rows |")
    lines.append("| --- | --- | --- |")
    for source, license_, count in sources:
        lines.append(f"| {source} | {license_} | {count} |")
    lines.append("")
    lines.append("## Decontamination")
    lines.append("")
    lines.append(
        f"Screened at decontaminate.py version {decon_version} against the eval sets below "
        f"BEFORE dedupe/split/assemble ran - deduping first can lose the clean twin of a "
        f"contaminated row, so this pipeline always screens first."
    )
    lines.append("")
    if screened:
        lines.append("Screened and clean:")
        for key, status, _allowed, items in screened:
            count = items if items is not None else "unknown"
            lines.append(f"- **{key}** - {count} item(s), status `{status}`")
    else:
        lines.append("No eval set is recorded as screened in this manifest.")
    lines.append("")
    if holes:
        lines.append("**Named holes** - eval sets waived rather than screened:")
        for key, status, allowed, _items in holes:
            waived = "waived" if allowed else "NOT waived - unresolved"
            lines.append(f"- **{key}** - status `{status}` ({waived})")
    else:
        lines.append("No eval set is recorded as waived.")
    lines.append("")
    if gaps:
        lines.append(
            "**Per-script gaps** in the semantic (paraphrase) layer - a script the embedding "
            "model has no discriminative power over is not screened by it (the exact-match "
            "layer still runs on every row regardless of script):"
        )
        for script, why, items in gaps:
            count = items if items is not None else 0
            lines.append(f"- **{script}** - not screened ({why}); {count} eval item(s)")
    lines.append("")
    lines.append("## The BNS Section 358 transition stream")
    lines.append("")
    lines.append(
        "A dedicated stream teaches the IPC-to-BNS recodification directly: which offence "
        "family governs depending on the offence date relative to the appointed day, and "
        "which procedural code (BNSS vs CrPC, BSA vs Evidence Act) applies at each stage from "
        "FIR through appeal. Answer keys are derived from statute text and the audited "
        "IPC-to-BNS mapping table, never model-generated."
    )
    lines.append("")
    lines.append("## Known risk: teacher legal error")
    lines.append("")
    lines.append(
        "Teacher-generated reasoning and answers can be legally wrong in a way none of the "
        "automated gates catch (citation existence, temporal validity, format, length and "
        "judge scores all pass a plausible-sounding wrong answer). The mitigation is human: "
        "an operator reads 50 random accepted examples before the corpus ships - the only "
        "legal-accuracy check in this pipeline - and this residual risk is accepted and "
        "disclosed here rather than assumed away."
    )
    if extra:
        lines.append("")
        lines.append(extra)
    lines.append("")
    lines.append("## Build provenance")
    lines.append("")
    for name in ("decontaminate", "dedupe", "split", "assemble", "stats", "push"):
        lines.append(f"- {name}: version {versions.get(name)}")
    # N-1 (round 2): NOT report['at'] - a timestamp is not a property of the
    # corpus, and rendering it here was the whole reason a stats-only re-run
    # over byte-identical rows changed README's bytes (see IDEMPOTENCE in
    # the module docstring). The timestamp still travels in
    # build_manifest.json's stats_report.at and in the uploaded stats.json
    # itself - dropped here, not lost.
    lines.append(f"- stats report verdict: {report.get('verdict')}")
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# build_manifest.json.
# --------------------------------------------------------------------------

def build_manifest(
    *, report: dict, decon: dict | None, push_cfg, versions: dict, outputs: Sequence[dict],
    chain: dict | None,
) -> dict:
    from tuned.data.store import utcnow

    return {
        "stage": "push",
        "push_version": PUSH_VERSION,
        "at": utcnow(),
        "repo_id": push_cfg.repo_id,
        "private": push_cfg.private,
        "profile": report.get("profile"),
        "counts": {
            "rows": (report.get("measurements") or {}).get("rows"),
            "train": (report.get("sides") or {}).get("train", 0),
            "eval": (report.get("sides") or {}).get("eval", 0),
        },
        "tokenizer": report.get("tokenizer"),
        # Every module version this run could reach - see module_versions.
        "module_versions": versions,
        # The custody chain, carried forward the way the rest of this pipeline
        # does: assemble.json["split"] is the WHOLE split.json, and
        # split.json["dedupe"] is the WHOLE dedupe.json (split.py's own
        # comment: "carried whole rather than summarised"). One level deeper,
        # dedupe.json["decontamination"] is NOT the whole decontamination.json
        # - dedupe.py deliberately narrows it to a custody-verification
        # summary (`at`/`decon_version`/`counts`/`eval_sets`/`semantic`, no
        # `semantic_scripts`) - so this field alone cannot reconstruct the
        # card's per-script gap claims. `decontamination` below closes that
        # gap directly from the same enriched decon manifest the card itself
        # reads, rather than reaching through dedupe.py's summary.
        "chain": chain,
        # A pointer back to the report that authorised this push (its own
        # `at`/`verdict`/`red`, not the whole thing - `chain` above already
        # carries most of the custody tree). `stats.json` itself now travels
        # in the uploaded set too (see main()), so this pointer's `at` can be
        # cross-checked against the file sitting right beside it in the repo.
        "stats_report": {
            "at": report.get("at"), "verdict": report.get("verdict"), "red": report.get("red"),
        },
        # The full detail behind the card's "Decontamination" section -
        # screened/waived eval sets AND per-script semantic gaps - read off
        # the SAME `decon` manifest render_card() does, so this can never
        # drift from what the card actually claims. `chain` above cannot
        # substitute for this: see the comment there.
        "decontamination": {
            "decon_version": (decon or {}).get("decon_version"),
            "eval_sets": {
                key: {
                    "status": v.get("status"), "allowed_missing": v.get("allowed_missing"),
                    "items": v.get("items"),
                }
                for key, v in ((decon or {}).get("eval_sets") or {}).items()
            },
            "semantic_scripts": (decon or {}).get("semantic_scripts") or {},
        },
        # sha256 of every uploaded file - the full audit record. NOT what
        # idempotence is measured against (see same_uploaded_bytes): README
        # and stats.json both embed the report's own timestamp, so listing
        # them here is provenance, not the instrument.
        "outputs": list(outputs),
    }


class RemoteManifestCorrupt(RuntimeError):
    """The repo's own build_manifest.json is not shaped like one this module
    wrote. That manifest comes from OUTSIDE this machine - refuse rather than
    let a malformed shape crash with a bare TypeError/AttributeError, the way
    every other input this module reads already refuses on."""


# The outputs idempotence is measured against: the corpus bytes AND the card
# that describes them. NOT stats.json - it embeds the stats report's own
# `at` timestamp directly (the report IS that JSON), so a stats.py re-run
# over byte-identical rows still changes ITS sha256, and comparing it would
# defeat "unchanged input set -> no-op" on every re-run, not just ones that
# changed the corpus. README.md does NOT have that problem any more:
# render_card no longer prints report['at'] (round 2, N-1) specifically so
# README is a pure function of corpus + config + code, which is what makes
# it safe to put back in this set - a genuinely changed card (a new
# push.card_extra, a card-rendering fix) now uploads instead of silently
# never reaching an already-pushed repo.
_CONTENT_OUTPUTS = frozenset({TRAIN_FILENAME, EVAL_FILENAME, README_FILENAME})


def same_uploaded_bytes(current: dict | None, manifest: dict) -> bool:
    """True when `current` (the repo's own last build_manifest.json) already
    carries the same train/eval/README bytes as `manifest` (the one this run
    just built) - stats.json is excluded, see _CONTENT_OUTPUTS. Raises
    RemoteManifestCorrupt if `current`'s shape cannot be read at all - see
    the class docstring for why that is a refusal, not a crash.
    """
    if current is None:
        return False
    if not isinstance(current, dict):
        raise RemoteManifestCorrupt(
            f"the repo's build_manifest.json is not a JSON object (found "
            f"{type(current).__name__})"
        )
    outputs = current.get("outputs")
    if not isinstance(outputs, list):
        raise RemoteManifestCorrupt(
            f"the repo's build_manifest.json's outputs is not a list (found "
            f"{type(outputs).__name__})"
        )
    for i, o in enumerate(outputs):
        if (
            not isinstance(o, dict)
            or not isinstance(o.get("path"), str)
            or not isinstance(o.get("sha256"), str)
        ):
            raise RemoteManifestCorrupt(
                f"the repo's build_manifest.json's outputs[{i}] is not a "
                f"{{path, sha256}} pair (found {o!r})"
            )

    def key(m: dict) -> list[tuple]:
        return sorted(
            (o.get("path"), o.get("sha256"))
            for o in (m.get("outputs") or [])
            if o.get("path") in _CONTENT_OUTPUTS
        )

    return key(current) == key(manifest)


# --------------------------------------------------------------------------
# The hub-client seam. Everything above this line runs with no dependency on
# huggingface_hub at all - only RealHubClient imports it, and only lazily.
# --------------------------------------------------------------------------

class RealHubClient:
    """The real seam. Duck-typed against three methods (no ABC: a fake in a
    test file only has to implement these, never import this class):

      ensure_repo(repo_id, *, private) -> None
          Create the dataset repo if it does not exist yet; a no-op on one
          that already does (`exist_ok=True`).
      current_manifest(repo_id) -> (dict | None, str | None)
          The repo's own build_manifest.json content and current revision, or
          (None, None) for a repo that does not exist yet.
      upload(repo_id, files: dict[str, Path], *, commit_message) -> str
          Upload every (path_in_repo -> local Path) pair in one commit-ish
          sequence and return the resulting revision (commit sha).

    The token is held only to construct the API client - no method here
    returns, prints or logs it.
    """

    def __init__(self, token: str, *, api=None):
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
        self._api = api

    def ensure_repo(self, repo_id: str, *, private: bool) -> None:
        self._api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    def current_manifest(self, repo_id: str) -> tuple[dict | None, str | None]:
        import json

        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

        try:
            info = self._api.dataset_info(repo_id)
        except RepositoryNotFoundError:
            return None, None
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=repo_id, filename=MANIFEST_FILENAME, repo_type="dataset")
        except EntryNotFoundError:
            return None, info.sha
        return json.loads(Path(path).read_text(encoding="utf-8")), info.sha

    def upload(self, repo_id: str, files: dict, *, commit_message: str) -> str:
        for name, path in sorted(files.items()):
            self._api.upload_file(
                path_or_fileobj=str(path), path_in_repo=name, repo_id=repo_id,
                repo_type="dataset", commit_message=commit_message,
            )
        return self._api.dataset_info(repo_id).sha


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None, *, hub_client=None) -> int:
    import argparse

    from tuned.data.config import load_build_config
    from tuned.data.paths import build_paths

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="data/configs/data_law_v1.yaml")
    parser.add_argument("--report", default=None, help=f"default out/{STATS_REPORT_FILENAME}")
    parser.add_argument("--out", default=None, help="where the card+manifest render (default out/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="render the card and manifest; make no network call")
    args = parser.parse_args(argv)

    cfg = load_build_config(args.config)
    if cfg.push is None:
        print(
            f"push REFUSES TO RUN: {args.config} has no `push:` block, so there is no "
            f"repo to write to.\n"
            f"  remedy: add push: {{repo_id: your/repo}} to the config\n"
            f"  nothing was uploaded."
        )
        return 2

    paths = build_paths(cfg.build.workdir).ensure()
    report_path = Path(args.report) if args.report else paths.out_dir / STATS_REPORT_FILENAME
    report = read_manifest(report_path)
    refusal = stats_refusal(report, report_path=report_path, config_path=args.config)
    if refusal:
        print(refusal)
        return 2

    inputs = report.get("inputs") or []
    if len(inputs) != 2:
        print(
            f"push REFUSES TO RUN: stats' report names {len(inputs)} input file(s), not the "
            f"(train, eval) pair push.py expects.\n"
            f"  report: {report_path}\n"
            f"  nothing was uploaded."
        )
        return 2
    train_path, eval_path = Path(inputs[0]), Path(inputs[1])
    missing = [str(p) for p in (train_path, eval_path) if not p.exists()]
    if missing:
        print(
            f"push REFUSES TO RUN: input(s) stats.py graded are gone: {', '.join(missing)}.\n"
            f"  remedy: python -m tuned.data.assemble --config {args.config}, then re-run "
            f"stats.py\n"
            f"  nothing was uploaded."
        )
        return 2

    refusal = bytes_refusal(report, train_path=train_path, eval_path=eval_path)
    if refusal:
        print(refusal)
        return 2

    decon = read_manifest(paths.out_dir / DECON_MANIFEST_FILENAME)
    # The whole custody chain, carried forward - assemble.json already nests
    # split.json, which nests dedupe.json, which nests decontamination.json.
    chain = read_manifest(paths.out_dir / ASSEMBLE_MANIFEST_FILENAME)
    versions = module_versions(paths.out_dir, report)
    version_faults = sorted(name for name, v in versions.items() if v is None and name != "push")
    if version_faults:
        print(
            f"push REFUSES TO RUN: the module-version record for {', '.join(version_faults)} "
            f"is unreadable in {paths.out_dir}, even though stats.py reported the custody "
            f"chain complete - an artifact went missing after the green report was written.\n"
            f"  nothing was uploaded."
        )
        return 2

    decon_faults = decon_chain_faults(decon, chain)
    if decon_faults:
        print(
            f"push REFUSES TO RUN: decontamination.json disagrees with the custody chain's "
            f"own record of it ({', '.join(decon_faults)}) - it was edited after the chain "
            f"was built.\n"
            f"  remedy: re-run the pipeline tail so the file and the chain agree again\n"
            f"  nothing was uploaded."
        )
        return 2

    extra = None
    if cfg.push.card_extra:
        extra_path = Path(cfg.push.card_extra)
        if not extra_path.exists():
            print(
                f"push REFUSES TO RUN: push.card_extra names {extra_path}, which does not "
                f"exist.\n"
                f"  nothing was uploaded."
            )
            return 2
        extra = extra_path.read_text(encoding="utf-8")

    rows = list(read_jsonl(train_path)) + list(read_jsonl(eval_path))
    try:
        card = render_card(
            report=report, decon=decon, rows=rows, push_cfg=cfg.push, versions=versions,
            extra=extra,
        )
    except CardDataMissing as exc:
        print(f"push REFUSES TO RUN: {exc}.\n  nothing was uploaded.")
        return 2

    out_dir = Path(args.out) if args.out else paths.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    readme_path = out_dir / README_FILENAME
    manifest_path = out_dir / MANIFEST_FILENAME
    readme_path.write_text(card, encoding="utf-8")

    outputs = [
        {"path": TRAIN_FILENAME, "rows": (report.get("sides") or {}).get("train", 0),
         "sha256": sha256_file(train_path)},
        {"path": EVAL_FILENAME, "rows": (report.get("sides") or {}).get("eval", 0),
         "sha256": sha256_file(eval_path)},
        {"path": README_FILENAME, "sha256": sha256_file(readme_path)},
        # stats.json travels with the repo now too, so the `stats_report`
        # pointer in build_manifest.json points at a file actually sitting
        # beside it - not one that only exists on the operator's disk.
        {"path": STATS_REPORT_FILENAME, "sha256": sha256_file(report_path)},
    ]
    manifest = build_manifest(
        report=report, decon=decon, push_cfg=cfg.push, versions=versions, outputs=outputs,
        chain=chain,
    )
    write_manifest(manifest_path, manifest)

    print(f"card -> {readme_path}")
    print(f"manifest -> {manifest_path}")

    if args.dry_run:
        print("DRY RUN: no network call was made; nothing was uploaded.")
        return 0

    load_dotenv_keys()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "push REFUSES TO RUN: HF_TOKEN is not set (checked the environment and the "
            "worktree-root .env via load_dotenv_keys). The token VALUE is never logged - "
            "only its presence is checked here.\n"
            "  nothing was uploaded."
        )
        return 2

    client = hub_client if hub_client is not None else RealHubClient(token)
    current, current_revision = client.current_manifest(cfg.push.repo_id)
    try:
        unchanged = same_uploaded_bytes(current, manifest)
    except RemoteManifestCorrupt as exc:
        print(
            f"push REFUSES TO RUN: {exc}.\n"
            f"  remedy: inspect {cfg.push.repo_id}'s build_manifest.json by hand, or delete "
            f"it and let this push recreate it\n"
            f"  nothing was uploaded."
        )
        return 2
    if unchanged:
        print(f"push: {cfg.push.repo_id} already carries these exact bytes - no-op.")
        print(f"revision: {current_revision}")
        return 0

    client.ensure_repo(cfg.push.repo_id, private=cfg.push.private)
    revision = client.upload(
        cfg.push.repo_id,
        {
            TRAIN_FILENAME: train_path, EVAL_FILENAME: eval_path,
            README_FILENAME: readme_path, MANIFEST_FILENAME: manifest_path,
            STATS_REPORT_FILENAME: report_path,
        },
        commit_message=f"push_version {PUSH_VERSION}: {report.get('sides')}",
    )
    print(f"pushed {cfg.push.repo_id}")
    print(f"revision: {revision}")
    return 0


if __name__ == "__main__":
    import sys

    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
