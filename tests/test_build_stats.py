"""stats.py - the terminal gate, and the whole tail end to end.

Every gate carries a corpus that passes it and a corpus that trips it. The
last section runs a synthetic corpus through decontaminated -> deduped ->
split -> assembled -> stats for real, once green and once per red gate.

Fixtures are structural shapes with filler prose; no real judgment or eval
text appears anywhere here.
"""

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from pipeline_fakes import paths_for, temp_config
from test_build_assemble import FakeTokenizer
from test_build_decontaminate import prose, row

from tuned.data.acquire import sha256_file
from tuned.data.assemble import main as assemble_main
from tuned.data.config import load_build_config
from tuned.data.dedupe import main as dedupe_main
from tuned.data.jsonl import read_jsonl, write_jsonl
from tuned.data.replay import empty_think
from tuned.data.split import main as split_main
from tuned.data.stats import (
    GATE_CHAIN,
    GATE_CROSS_CODE,
    GATE_DUP,
    GATE_EMPTY_THINK,
    GATE_LENGTH,
    GATE_LICENSE,
    GATE_MARKUP,
    GATE_MIX,
    GATE_TRACE,
    GATES,
    GREEN,
    RED,
    REPORT,
    REPORT_FILENAME,
    STATS_VERSION,
    SUMMARY_FILENAME,
    chain_links,
    cross_code_rows,
    duplicate_count,
    empty_think_count,
    gate_chain,
    gate_cross_code,
    gate_length,
    gate_license,
    gate_markup,
    gate_mix,
    gate_share,
    is_old_code_source,
    length_report,
    license_counts,
    markup_rows,
    measure,
    percentile,
    stream_counts,
    summary_of,
    trace_count,
)
from tuned.data.stats import main as stats_main

STATS_SRC = Path(__file__).parent.parent / "src" / "tuned" / "data" / "stats.py"

OPEN, CLOSE = "<think>", "</think>"
SYNTHESIS = "synthesis"
CURATED = "L-NLProc/PredEx_Instruction-Tuning_Pred-Exp"
REPLAY = "open-thoughts/OpenThoughts-114k"


def cfg_for(tmp_path):
    return load_build_config(temp_config(tmp_path), allow_unpinned=True)


def traced(seed: int, *, source=SYNTHESIS, license_="CC-BY-4.0", words=40, **prov) -> dict:
    r = row(prose(seed, 30), f"{OPEN}{prose(seed + 1, words)}{CLOSE}{prose(seed + 2, 25)}",
            reasoning=True, **prov)
    r["_prov"].update(source=source, license=license_)
    return r


def scaffolded(seed: int, *, source=REPLAY, license_="Apache-2.0", **prov) -> dict:
    r = row(prose(seed, 30), empty_think(OPEN, CLOSE) + prose(seed + 2, 25),
            reasoning=False, **prov)
    r["_prov"].update(source=source, license=license_)
    return r


# --------------------------------------------------------------------------
# The measurements.
# --------------------------------------------------------------------------

def test_percentile_names_a_row_that_exists():
    """Nearest-rank, not interpolation: a p99 between two real rows is not an
    answer to "the 99th percentile row is this long"."""
    values = [10, 20, 30, 40]
    assert percentile(values, 50) == 20
    assert percentile(values, 90) == 40
    assert percentile(values, 99) == 40
    assert all(percentile(values, p) in values for p in (1, 25, 50, 75, 99, 100))
    assert percentile([], 50) == 0
    assert percentile([7], 99) == 7
    # Order of the input does not decide it.
    assert percentile([40, 10, 30, 20], 50) == 20


def test_length_report_counts_over_the_limit_and_nothing_else():
    report = length_report([10, 20, 30, 8193], 8192)
    assert (report["p50"], report["max"], report["min"]) == (20, 8193, 10)
    assert report["over_limit"] == 1
    # AT the limit is inside it - the same boundary assemble.py drops on.
    assert length_report([8192], 8192)["over_limit"] == 0
    assert length_report([], 8192) == {
        "rows": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0, "min": 0,
        "limit": 8192, "over_limit": 0,
    }


def test_stream_counts_never_default_an_unmapped_source_into_a_bucket(tmp_path):
    cfg = cfg_for(tmp_path)
    rows = [traced(1), traced(2, source=CURATED), scaffolded(3),
            traced(4, source="nobody/mapped-this")]
    counts, unmapped = stream_counts(rows, cfg.assembly)
    assert counts == Counter({"grounded_synthesis": 1, "curated": 1, "replay": 1})
    assert unmapped == Counter({"nobody/mapped-this": 1})
    assert sum(counts.values()) + sum(unmapped.values()) == len(rows)


def test_the_empty_think_measure_is_a_byte_comparison_not_a_pattern():
    """`<think>\\n</think>` is not the scaffold the model is taught, and a
    regex loose enough to accept it would report one that is not there."""
    exact = scaffolded(1)
    one_newline = row("q", f"{OPEN}\n{CLOSE}answer")
    spaced = row("q", f"{OPEN}\n\n {CLOSE}answer")
    leading = row("q", f" {OPEN}\n\n{CLOSE}answer")
    assert empty_think_count([exact], OPEN, CLOSE) == 1
    assert empty_think_count([one_newline, spaced, leading], OPEN, CLOSE) == 0
    # STARTS WITH, so a real trace that happens to contain the empty block
    # later does not count.
    buried = row("q", f"{OPEN}the model wrote {OPEN}\n\n{CLOSE} in its trace{CLOSE}answer")
    assert empty_think_count([buried], OPEN, CLOSE) == 0


def test_the_trace_share_reads_provenance_not_content():
    """The two gates measure different things on purpose: _prov.reasoning is
    what the row builder ASSERTED, the scaffold is what the model SEES. A row
    that claims a trace and carries none is exactly what the pair catches."""
    liar = scaffolded(1)
    liar["_prov"]["reasoning"] = True
    assert trace_count([liar]) == 1
    assert empty_think_count([liar], OPEN, CLOSE) == 1


def test_duplicates_are_counted_beyond_the_first_of_each_key():
    a, b = traced(1), traced(2)
    assert duplicate_count([a, b])[0] == 0
    assert duplicate_count([a, b, a])[0] == 1
    assert duplicate_count([a, a, a])[0] == 2
    # The key is prompt AND answer: a shared prompt is not a duplicate row.
    shared = traced(3)
    shared["messages"][0]["content"] = a["messages"][0]["content"]
    assert duplicate_count([a, shared])[0] == 0
    assert duplicate_count([a, b, a, b])[1]  # the offenders are named


def test_markup_is_looked_for_in_content_not_in_the_rendered_template():
    """The template's own `<|im_start|>` is what the trainer needs; a check
    over the rendered string would flag every row in the corpus."""
    clean, dirty = traced(1), traced(2)
    dirty["messages"][1]["content"] += " <|im_end|>"
    prompt_side = traced(3)
    prompt_side["messages"][0]["content"] += " <|endoftext|>"
    assert markup_rows([clean]) == []
    assert markup_rows([clean, dirty, prompt_side]) == [1, 2]
    rendered = FakeTokenizer().apply_chat_template(clean["messages"])
    assert "<|" in rendered  # ...and the row is still clean


def test_license_counts_separate_the_absent_from_the_blank():
    rows = [traced(1), traced(2, license_="Apache-2.0"), traced(3, license_="   ")]
    del rows[0]["_prov"]["license"]
    counts, unlicensed = license_counts(rows)
    assert counts == Counter({"Apache-2.0": 1})
    assert unlicensed == 2


def test_cross_code_needs_both_a_new_code_shape_and_old_code_provenance():
    old_sources = ["169Pi/indian_law"]
    new_code = f"{OPEN}trace{CLOSE}Section 303 BNS applies here."
    old_code = f"{OPEN}trace{CLOSE}Section 302 IPC applies here."
    rows = [
        row("q", new_code, source="169Pi/indian_law"),          # 0: both -> hit
        row("q", old_code, source="169Pi/indian_law"),          # 1: old shape
        row("q", new_code, source=SYNTHESIS),                   # 2: modern corpus
        row("q", new_code, source=SYNTHESIS, code_era="ipc"),   # 3: era channel
        row("q", new_code, source=SYNTHESIS, code_era="bns"),   # 4: new era
    ]
    assert cross_code_rows(rows, old_sources) == [0, 3]
    # The regex is curated.py's, not a second copy of it.
    from tuned.data.curated import _NEW_CODE_RE

    from tuned.data.stats import NEW_CODE_RE

    assert NEW_CODE_RE is _NEW_CODE_RE


def test_old_code_sources_match_a_subset_by_its_dataset_half():
    assert is_old_code_source("169Pi/indian_law", ["169Pi/indian_law"])
    assert is_old_code_source("169Pi/indian_law:part2", ["169Pi/indian_law"])
    assert not is_old_code_source("169Pi/indian_law_v2", ["169Pi/indian_law"])
    assert not is_old_code_source(None, ["169Pi/indian_law"])
    assert not is_old_code_source("169Pi/indian_law", [])


# --------------------------------------------------------------------------
# The chain walk.
# --------------------------------------------------------------------------

def full_chain() -> dict:
    return {
        "stage": "assemble",
        "split_check": {"status": "verified"},
        "split": {
            "stage": "split",
            "dedupe_check": {"status": "verified"},
            "dedupe": {
                "stage": "dedupe",
                "decontamination_check": {"status": "verified"},
                "decontamination": {"decon_version": 4},
            },
        },
    }


def test_a_complete_chain_names_every_link_and_its_verification():
    links = chain_links(full_chain())
    assert [link["stage"] for link in links] == [
        "assemble", "split", "dedupe", "decontamination"
    ]
    assert all(link["present"] for link in links)
    assert gate_chain(links, required=True).status == GREEN


@pytest.mark.parametrize("break_at", ["split", "dedupe", "decontamination"])
def test_a_broken_chain_names_exactly_which_link_is_absent(break_at):
    manifest = full_chain()
    if break_at == "split":
        manifest["split"] = None
    elif break_at == "dedupe":
        manifest["split"]["dedupe"] = None
    else:
        manifest["split"]["dedupe"]["decontamination"] = None
    links = chain_links(manifest)
    gate = gate_chain(links, required=True)
    assert gate.status == RED
    assert break_at in gate.detail["missing"]
    # Everything ABOVE the break is still reported present, so the operator is
    # sent to one stage rather than told the whole chain failed.
    above = [link["stage"] for link in links if link["present"]]
    assert "assemble" in above


def test_a_present_but_unverified_link_is_red_too():
    """dedupe.py can ship with `decontamination: null` and a status saying
    why. A chain that reads the null and calls it a link would launder exactly
    the case the custody record exists to expose."""
    manifest = full_chain()
    manifest["split"]["dedupe"]["decontamination_check"] = {"status": "content_mismatch"}
    gate = gate_chain(chain_links(manifest), required=True)
    assert gate.status == RED and "decontamination" in gate.detail["unverified"]


def test_the_chain_gate_can_be_turned_down_to_a_note():
    gate = gate_chain(chain_links({"stage": "assemble"}), required=False)
    assert gate.status == REPORT and gate.detail["missing"]


# --------------------------------------------------------------------------
# The gates, at and around their thresholds.
# --------------------------------------------------------------------------

def test_the_length_gate_is_red_only_when_assemble_left_a_row_over_the_bucket():
    assert gate_length(length_report([100, 200], 8192)).status == GREEN
    red = gate_length(length_report([100, 9000], 8192))
    assert red.status == RED and "assemble.py" in red.summary


@pytest.mark.parametrize(
    "counts,status",
    [
        ({"grounded_synthesis": 60, "curated": 16, "replay": 24}, GREEN),   # exact
        ({"grounded_synthesis": 62, "curated": 15, "replay": 23}, GREEN),   # +2/-1/-1pp
        ({"grounded_synthesis": 63, "curated": 16, "replay": 21}, RED),     # +3pp
        ({"grounded_synthesis": 58, "curated": 16, "replay": 26}, GREEN),   # -2/+2pp
        ({"grounded_synthesis": 57, "curated": 16, "replay": 27}, RED),     # -3/+3pp
    ],
)
def test_the_mix_tolerance_is_percentage_points_at_the_boundary(counts, status):
    """Exactly 2pp is inside; 3pp is not. And 2pp is not 2% of 60% - a ratio
    reading would put the boundary at 1.2pp and pass nothing near it."""
    targets = {"grounded_synthesis": 0.60, "curated": 0.16, "replay": 0.24}
    gate = gate_mix(Counter(counts), Counter(), targets, total=sum(counts.values()),
                    tolerance_pp=2.0, profile="v1.1-full")
    assert gate.status == status


def test_an_unmapped_source_reds_the_mix_gate_even_when_the_shares_are_perfect():
    targets = {"grounded_synthesis": 0.60, "curated": 0.16, "replay": 0.24}
    counts = Counter({"grounded_synthesis": 60, "curated": 16, "replay": 24})
    gate = gate_mix(counts, Counter({"who/knows": 3}), targets, total=100,
                    tolerance_pp=2.0, profile="v1.1-full")
    assert gate.status == RED and "who/knows" in gate.summary


def test_the_mvp_profile_passes_a_corpus_the_full_profile_fails(tmp_path):
    """The gate always runs; the targets are data. An MVP corpus is
    synthesis-light BY DESIGN and grading it against 60/16/24 would fail a
    corpus that is exactly what it was meant to be."""
    cfg = cfg_for(tmp_path)
    counts = Counter({"grounded_synthesis": 30, "curated": 28, "replay": 42})
    for profile, status in (("v1.0-MVP", GREEN), ("v1.1-full", RED)):
        gate = gate_mix(counts, Counter(), cfg.assembly.targets(profile), total=100,
                        tolerance_pp=cfg.assembly.gates.mix_tolerance_pp, profile=profile)
        assert gate.status == status, profile
        assert gate.detail["profile"] == profile


@pytest.mark.parametrize(
    "count,status",
    [(80, GREEN), (81, GREEN), (79, RED), (100, GREEN), (0, RED)],
)
def test_the_trace_floor_is_inclusive(count, status):
    """80% exactly PASSES. `<` and not `<=`, because the plan's floor is
    ">= 80%" and a corpus sitting on it is a corpus that met it."""
    gate = gate_share(GATE_TRACE, count, 100, floor=0.80, label="reasoning traces")
    assert gate.status == status


@pytest.mark.parametrize(
    "count,status",
    [(17, RED), (18, GREEN), (20, GREEN), (22, GREEN), (23, RED)],
)
def test_the_empty_think_band_is_closed_at_both_ends(count, status):
    gate = gate_share(GATE_EMPTY_THINK, count, 100, floor=0.18, ceiling=0.22,
                      label="byte-exact empty think")
    assert gate.status == status


def test_the_empty_think_ceiling_and_the_trace_floor_cannot_both_be_satisfied_above_20pc():
    """A MEASURED interaction between two gates the brief specifies separately.

    On every shipped row builder a kept row is exactly one of two things: a
    reasoning row with a real trace (_prov.reasoning true) or a non-reasoning
    row carrying the byte-exact empty scaffold (_prov.reasoning false). Rows
    with neither - a bare answer from a teacher that returned no trace - are
    dropped by assemble.py as no_think_scaffold. So over the assembled corpus

        trace_share + empty_think_share == 1

    exactly, and the >= 80% floor caps the empty-think share at 20%. The
    configured band's upper half (20% to 22%) is therefore UNREACHABLE without
    failing the trace gate, and the effective band is 18-20%.

    Recorded as a test rather than as a comment because it is a fact about the
    thresholds an operator may want to change, and because a future builder
    that emits a third row shape would break the identity rather than the band.
    """
    corpus = [traced(i * 10) for i in range(78)] + [scaffolded(1000 + i) for i in range(22)]
    traces = trace_count(corpus)
    empties = empty_think_count(corpus, OPEN, CLOSE)
    assert traces + empties == len(corpus) == 100
    # 22% empty-think is inside the configured band...
    assert gate_share(GATE_EMPTY_THINK, empties, 100, floor=0.18, ceiling=0.22,
                      label="e").status == GREEN
    # ...and the same corpus fails the trace floor, which is the crossing point.
    assert gate_share(GATE_TRACE, traces, 100, floor=0.80, label="t").status == RED
    # 20% is the last share that satisfies both.
    at_20 = [traced(i * 10) for i in range(80)] + [scaffolded(1000 + i) for i in range(20)]
    assert gate_share(GATE_TRACE, trace_count(at_20), 100, floor=0.80, label="t").status == GREEN
    assert gate_share(GATE_EMPTY_THINK, empty_think_count(at_20, OPEN, CLOSE), 100,
                      floor=0.18, ceiling=0.22, label="e").status == GREEN


def test_the_float_guard_is_where_the_float_noise_is():
    """gate_mix rounds before comparing and gate_share does not, on purpose.

    A share is ONE division of two integers and IEEE division is correctly
    rounded, so when count/total equals its bound as a rational the two are the
    same double bit for bit - searched here over every total up to 4,000 for
    all four shipped bounds. gate_mix subtracts and then scales by 100, which
    is where the noise is: 62/100 against a 0.60 target is 2.0000000000000018
    percentage points, and an unguarded `> 2.0` reds a corpus sitting exactly
    on the tolerance.
    """
    for bound in (0.80, 0.18, 0.22, 0.005):
        for total in range(2, 4001):
            exact = bound * total
            count = round(exact)
            if abs(count - exact) < 1e-9:
                assert count / total == bound, (count, total, bound)
    assert abs(0.62 - 0.60) * 100 != 2.0
    assert round(abs(0.62 - 0.60) * 100, 9) == 2.0


@pytest.mark.parametrize("count,status", [(0, GREEN), (5, GREEN), (6, RED)])
def test_the_dup_ceiling_is_half_a_percent(count, status):
    gate = gate_share(GATE_DUP, count, 1000, ceiling=0.005, label="exact duplicates")
    assert gate.status == status


def test_the_markup_gate_has_no_tolerance_and_can_be_switched_off():
    assert gate_markup([], 100, enabled=True).status == GREEN
    assert gate_markup([3], 100, enabled=True).status == RED
    assert gate_markup([3], 100, enabled=False).status == REPORT


def test_the_license_gate_reds_on_one_row_and_publishes_the_counts():
    counts = Counter({"Apache-2.0": 99})
    assert gate_license(counts, 0, 99, required=True).status == GREEN
    red = gate_license(counts, 1, 100, required=True)
    assert red.status == RED and red.detail["unlicensed"] == 1
    assert gate_license(counts, 1, 100, required=False).status == REPORT
    assert "Apache-2.0 99" in gate_license(counts, 0, 99, required=True).summary


def test_cross_code_is_report_only_until_the_config_arms_it():
    assert gate_cross_code([], 100, red=False).status == GREEN
    note = gate_cross_code([1, 2], 100, red=False)
    assert note.status == REPORT and "report-only" in note.summary
    assert gate_cross_code([1, 2], 100, red=True).status == RED


def test_the_shipped_config_leaves_cross_code_unarmed(tmp_path):
    assert cfg_for(tmp_path).assembly.gates.cross_code_red is False


# --------------------------------------------------------------------------
# The report.
# --------------------------------------------------------------------------

def test_every_gate_appears_in_the_report_and_in_the_summary(tmp_path):
    cfg = cfg_for(tmp_path)
    rows = [traced(i * 10) for i in range(81)] + [scaffolded(1000 + i) for i in range(19)]
    gates, measurements = measure(rows, cfg=cfg, tokenizer=FakeTokenizer(),
                                  profile="v1.1-full", manifest=full_chain())
    assert {g.name for g in gates} == set(GATES)
    assert measurements["rows"] == 100
    from tuned.data.stats import report_of

    report = report_of(gates, measurements, profile="v1.1-full",
                       sides={"train": 90, "eval": 10}, inputs=["a", "b"],
                       custody={"status": "verified"}, tokenizer_id={"repo": "r"})
    text = summary_of(report)
    for name in GATES:
        assert f"| {name} |" in text
    assert "verdict" in text


# --------------------------------------------------------------------------
# The whole tail, end to end.
# --------------------------------------------------------------------------

# One token per whitespace word is nothing like a real tokenizer on this
# corpus, so the fake scales: the long row below is ~560 words and has to clear
# an 8192-token bucket that the ~100-word ordinary rows must stay under.
SCALE = 20


def e2e_corpus() -> list[dict]:
    """~100 rows across three streams, plus one that cannot fit the bucket.

    Composed to sit inside every band at once, which is not automatic:
    60/16/24 exactly on the mix, 19 byte-exact empty-think rows (19%, inside
    both the configured 18-22% band and the 18-20% the trace floor really
    allows), 81 reasoning rows, no duplicates, no markup, a license on every
    row. Six two-row cases carry CNRs so the split has siblings to keep
    together and dates to prefer; everything else is case-less.
    """
    rows: list[dict] = []
    for case in range(6):
        key = f"ESCR01{case:06d}{2001 + case}"
        rows += [traced(case * 100 + i, cnr=key) for i in range(2)]
    rows += [traced(2000 + i * 7) for i in range(48)]          # 60 synthesis so far
    rows += [traced(4000 + i * 7, source=CURATED, license_="Apache-2.0") for i in range(16)]
    rows += [scaffolded(6000 + i * 7) for i in range(19)]
    rows += [traced(8000 + i * 7, source=REPLAY, license_="Apache-2.0") for i in range(5)]
    # The row nothing can hold: dated 1900 so the split leaves it in train,
    # where dropping it cannot empty the eval side.
    rows.append(traced(9999, words=500, cnr="ESCR019999991900"))
    return rows


def run_pipeline(tmp_path, rows, *, profile=None, cfg_path=None):
    """decontaminated -> deduped -> split -> assembled -> stats, for real.

    Nothing here is a stand-in: every manifest in the chain stats.py walks was
    written by the stage that owns it.
    """
    cfg = cfg_path or temp_config(tmp_path)
    paths = paths_for(tmp_path)
    screened = paths.out_dir / "decontaminated.jsonl"
    written = write_jsonl(screened, rows)
    (paths.out_dir / "decontamination.json").write_text(
        json.dumps({
            "stage": "decontaminate",
            "decon_version": 4,
            "counts": {"kept": written},
            "output": {"path": str(screened), "rows": written,
                       "sha256": sha256_file(screened)},
            "thresholds": {"case_ids_from_text": True},
        }),
        encoding="utf-8",
    )
    tok = FakeTokenizer(SCALE)
    codes = {
        "dedupe": dedupe_main(["--config", cfg]),
        "split": split_main(["--config", cfg]),
        "assemble": assemble_main(["--config", cfg], tokenizer=tok),
    }
    argv = ["--config", cfg] + (["--profile", profile] if profile else [])
    codes["stats"] = stats_main(argv, tokenizer=tok)
    report = json.loads((paths.out_dir / REPORT_FILENAME).read_text(encoding="utf-8"))
    return codes, report, paths


def test_the_passing_variant_flows_all_the_way_through_and_exits_zero(tmp_path):
    codes, report, paths = run_pipeline(tmp_path, e2e_corpus())
    assert codes == {"dedupe": 0, "split": 0, "assemble": 0, "stats": 0}
    assert report["verdict"] == "green" and report["red"] == []

    # The numbers, stated so a drift shows up as a diff rather than as a pass.
    assert report["measurements"]["rows"] == 100          # 101 in, one over the bucket
    assert report["sides"] == {"train": 90, "eval": 10}
    assert report["gates"][GATE_MIX]["detail"]["counts"] == {
        "grounded_synthesis": 60, "curated": 16, "replay": 24
    }
    assert report["gates"][GATE_TRACE]["detail"]["count"] == 81
    assert report["gates"][GATE_EMPTY_THINK]["detail"]["count"] == 19
    assert report["gates"][GATE_DUP]["detail"]["count"] == 0
    assert report["gates"][GATE_LENGTH]["detail"]["over_limit"] == 0
    assert report["gates"][GATE_LENGTH]["detail"]["max"] <= 8192
    assert report["gates"][GATE_LICENSE]["detail"]["counts"] == {
        "Apache-2.0": 40, "CC-BY-4.0": 60
    }
    # Every link of the chain, written by the stage that owns it.
    assert report["gates"][GATE_CHAIN]["status"] == GREEN
    assert [link["stage"] for link in report["gates"][GATE_CHAIN]["detail"]["links"]] == [
        "assemble", "split", "dedupe", "decontamination"
    ]
    # And the split really did hold cases together.
    split_manifest = json.loads((paths.out_dir / "split.json").read_text(encoding="utf-8"))
    assert split_manifest["assignment"]["date_assigned_units"] == 5
    assert split_manifest["assignment"]["hash_assigned_units"] == 0
    assert split_manifest["assignment"]["date_boundary"] == "2002-00-00"
    assert (paths.out_dir / SUMMARY_FILENAME).read_text(encoding="utf-8").startswith("#")


def test_the_over_length_row_is_dropped_by_assemble_and_named(tmp_path):
    _codes, _report, paths = run_pipeline(tmp_path, e2e_corpus())
    drops = list(read_jsonl(paths.out_dir / "assemble_drops.jsonl"))
    assert [(d["side"], d["reason"]) for d in drops] == [("train", "over_length_bucket")]
    assert drops[0]["tokens"] > 8192
    # Dropped, not trimmed: the row is simply absent from the artifact.
    assert len(list(read_jsonl(paths.out_dir / "law_v1_train.jsonl"))) == 90


def _relabel(rows, source, n):
    out = [dict(r, _prov=dict(r["_prov"])) for r in rows]
    changed = 0
    for r in out:
        if changed >= n:
            break
        if r["_prov"]["source"] == SYNTHESIS and "cnr" not in r["_prov"]:
            r["_prov"]["source"] = source
            changed += 1
    assert changed == n
    return out


def red_mix(rows):
    return _relabel(rows, REPLAY, 20)


def red_trace(rows):
    out = [dict(r, _prov=dict(r["_prov"])) for r in rows]
    for r in out[:30]:
        r["_prov"]["reasoning"] = False
    return out


def red_empty_think(rows):
    """Ten more rows carrying the byte-exact scaffold, provenance untouched -
    so the empty-think share moves and the trace share does not."""
    out = [json.loads(json.dumps(r)) for r in rows]
    for r in out[20:30]:
        r["messages"][1]["content"] = empty_think(OPEN, CLOSE) + r["messages"][1]["content"][
            len(OPEN):
        ].split(CLOSE, 1)[1]
    return out


def red_markup(rows):
    out = [json.loads(json.dumps(r)) for r in rows]
    out[40]["messages"][1]["content"] += " <|im_end|>"
    return out


def red_license(rows):
    out = [json.loads(json.dumps(r)) for r in rows]
    del out[41]["_prov"]["license"]
    return out


@pytest.mark.parametrize(
    "mutate,gate",
    [
        (red_mix, GATE_MIX),
        (red_trace, GATE_TRACE),
        (red_empty_think, GATE_EMPTY_THINK),
        (red_markup, GATE_MARKUP),
        (red_license, GATE_LICENSE),
    ],
)
def test_each_red_variant_exits_non_zero_naming_exactly_its_own_gate(tmp_path, mutate, gate):
    codes, report, _paths = run_pipeline(tmp_path, mutate(e2e_corpus()))
    assert codes["stats"] == 1
    assert report["verdict"] == "red"
    assert report["red"] == [gate], report["gates"][gate]["summary"]


def append_to_the_artifact(paths, extra):
    """Put rows into assemble's output AFTER it ran, and re-stamp the digest.

    Two gates - length and dup - measure faults an intact pipeline cannot
    produce: assemble.py drops the over-length rows and dedupe.py removes the
    exact duplicates (measured: feeding the pipeline a duplicated row gets
    `drop[exact]: 1` out of dedupe, and stats sees a clean corpus). That is not
    a reason to leave either gate untested - it is the reason they exist. Both
    are tripwires on a stage that did not do what its manifest says, so both
    are reached by forging exactly that: the artifact a lying stage would have
    left, under a manifest that vouches for it.
    """
    from tuned.data.assemble import MANIFEST_FILENAME as ASSEMBLE_MANIFEST

    train = paths.out_dir / "law_v1_train.jsonl"
    write_jsonl(train, list(read_jsonl(train)) + list(extra))
    manifest = json.loads((paths.out_dir / ASSEMBLE_MANIFEST).read_text(encoding="utf-8"))
    manifest["outputs"][0]["sha256"] = sha256_file(train)
    (paths.out_dir / ASSEMBLE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return train


def test_the_length_gate_reds_when_assemble_did_not_do_what_it_claims(tmp_path):
    _codes, _report, paths = run_pipeline(tmp_path, e2e_corpus())
    append_to_the_artifact(paths, [traced(70_000, words=500)])
    assert stats_main(["--config", temp_config(tmp_path)],
                      tokenizer=FakeTokenizer(SCALE)) == 1
    report = json.loads((paths.out_dir / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["red"] == [GATE_LENGTH]
    assert report["gates"][GATE_LENGTH]["detail"]["over_limit"] == 1


def test_the_dup_gate_reds_on_rows_reintroduced_after_dedupe_ran(tmp_path):
    """A tripwire on the chain, not a re-deduplication.

    Fed a duplicated row, the pipeline removes it at dedupe and stats sees a
    clean corpus - so the only way to this gate is a row that came back
    afterwards, which is the only thing it is for.
    """
    _codes, report, paths = run_pipeline(tmp_path, e2e_corpus() + [e2e_corpus()[30]])
    assert report["red"] == [] and report["gates"][GATE_DUP]["detail"]["count"] == 0

    train = paths.out_dir / "law_v1_train.jsonl"
    reintroduced = list(read_jsonl(train))[:1]
    append_to_the_artifact(paths, reintroduced)
    assert stats_main(["--config", temp_config(tmp_path)],
                      tokenizer=FakeTokenizer(SCALE)) == 1
    report = json.loads((paths.out_dir / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["red"] == [GATE_DUP]
    assert report["gates"][GATE_DUP]["detail"]["count"] == 1


def test_a_broken_chain_reds_the_run_even_when_every_row_is_perfect(tmp_path):
    from tuned.data.assemble import MANIFEST_FILENAME as ASSEMBLE_MANIFEST

    _codes, _report, paths = run_pipeline(tmp_path, e2e_corpus())
    manifest = json.loads((paths.out_dir / ASSEMBLE_MANIFEST).read_text(encoding="utf-8"))
    manifest["split"]["dedupe"]["decontamination"] = None
    (paths.out_dir / ASSEMBLE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    assert stats_main(["--config", temp_config(tmp_path)], tokenizer=FakeTokenizer(SCALE)) == 1
    report = json.loads((paths.out_dir / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["red"] == [GATE_CHAIN]
    assert report["gates"][GATE_CHAIN]["detail"]["missing"] == ["decontamination"]


def test_the_cross_code_gate_fires_once_the_config_arms_it(tmp_path, monkeypatch):
    """Report-only by default and red on demand, over the same corpus - a
    toggle nothing can exercise is a toggle nobody has checked."""
    rows = e2e_corpus()
    offender = json.loads(json.dumps(rows[30]))
    offender["_prov"]["source"] = "169Pi/indian_law"
    offender["_prov"]["license"] = "Apache-2.0"
    offender["messages"][1]["content"] += " The BNS provision governs."
    rows = rows[:30] + [offender] + rows[31:]

    _codes, report, paths = run_pipeline(tmp_path, rows)
    assert report["red"] == [] and report["gates"][GATE_CROSS_CODE]["status"] == REPORT
    assert report["gates"][GATE_CROSS_CODE]["detail"]["rows"] == 1

    real_loader = load_build_config

    def armed(path, **kwargs):
        cfg = real_loader(path, **kwargs)
        gates = replace(cfg.assembly.gates, cross_code_red=True)
        return replace(cfg, assembly=replace(cfg.assembly, gates=gates))

    monkeypatch.setattr("tuned.data.config.load_build_config", armed)
    assert stats_main(["--config", temp_config(tmp_path)], tokenizer=FakeTokenizer(SCALE)) == 1
    report = json.loads((paths.out_dir / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["red"] == [GATE_CROSS_CODE]


def test_the_profile_flag_regrades_the_same_corpus_and_is_recorded(tmp_path):
    codes, report, _paths = run_pipeline(tmp_path, e2e_corpus(), profile="v1.0-MVP")
    assert report["profile"] == "v1.0-MVP"
    # 60/16/24 is nowhere near the MVP targets, so the same green corpus reds.
    assert codes["stats"] == 1 and report["red"] == [GATE_MIX]


def test_an_unknown_profile_refuses_rather_than_grading_against_the_default(tmp_path):
    _codes, _report, _paths = run_pipeline(tmp_path, e2e_corpus())
    assert stats_main(["--config", temp_config(tmp_path), "--profile", "v9"],
                      tokenizer=FakeTokenizer(SCALE)) == 2


# --------------------------------------------------------------------------
# The CLI's own contracts.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mutate,banner",
    [
        (lambda p: (p / "assemble.json").unlink(), "NO UPSTREAM MANIFEST"),
        (lambda p: (p / "assemble.json").write_text('{"outputs": []}', encoding="utf-8"),
         "NO OUTPUT DIGEST"),
        (lambda p: write_jsonl(p / "law_v1_eval.jsonl", []), "DESCRIBES DIFFERENT ROWS"),
    ],
)
def test_stats_refuses_rows_it_cannot_trace_to_assemble(tmp_path, capsys, mutate, banner):
    _codes, _report, paths = run_pipeline(tmp_path, e2e_corpus())
    (paths.out_dir / REPORT_FILENAME).unlink()
    mutate(paths.out_dir)
    assert stats_main(["--config", temp_config(tmp_path)], tokenizer=FakeTokenizer(SCALE)) == 2
    assert banner in capsys.readouterr().out
    assert not (paths.out_dir / REPORT_FILENAME).exists()


def test_a_missing_input_names_the_command_that_makes_it(tmp_path, capsys):
    cfg = temp_config(tmp_path)
    paths_for(tmp_path)
    assert stats_main(["--config", cfg], tokenizer=FakeTokenizer()) == 2
    assert "tuned.data.assemble" in capsys.readouterr().out


def test_the_run_is_logged_to_the_store(tmp_path):
    from tuned.data.store import Store

    _codes, _report, paths = run_pipeline(tmp_path, e2e_corpus())
    event = json.loads(Store.open(paths.state_db).events("stats")[0]["detail_json"])
    assert event["stage"] == "stats" and event["verdict"] == "green"


def test_the_report_names_the_tokenizer_that_measured_the_lengths(tmp_path):
    _codes, report, _paths = run_pipeline(tmp_path, e2e_corpus())
    assert report["tokenizer"]["repo"] == "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    assert report["stats_version"] == STATS_VERSION == 1


def test_no_threshold_is_written_into_this_module():
    """Every number the gates compare against comes from config. A builder
    gate whose thresholds live in its own source is a gate that gets edited to
    pass."""
    import re

    source = STATS_SRC.read_text(encoding="utf-8")
    body = source.split('"""', 2)[2]  # past the module docstring
    for banned in ("0.80", "0.18", "0.22", "0.005", "8192", "2.0", "60", "16", "24"):
        assert not re.search(rf"(?<![\w.]){re.escape(banned)}(?![\w.])\s*[)>=,]", body), banned


def test_cli_hard_exits_after_success():
    assert "os._exit(" in STATS_SRC.read_text(encoding="utf-8")


def test_the_version_ledger_describes_the_version_the_module_ships():
    import re

    source = STATS_SRC.read_text(encoding="utf-8")
    entries = [int(n) for n in re.findall(r"^# (\d+)  ", source, re.M)]
    assert entries == sorted(entries)
    assert entries[-1] == STATS_VERSION
    assert entries[0] == 1
