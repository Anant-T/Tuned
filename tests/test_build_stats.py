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


@pytest.mark.parametrize(
    "strip_at,check_key",
    [("assemble", "split_check"), ("split", "dedupe_check"),
     ("dedupe", "decontamination_check")],
)
def test_a_link_whose_verification_was_never_recorded_is_red_and_names_the_key(
    strip_at, check_key
):
    """The third state, and the cheapest forgery of the three.

    Corrupting a status is work; DELETING one is a keystroke, and every link is
    still `present` afterwards. Tolerating an absent check for anything but the
    head made a manifest with all three `*_check` keys stripped read "custody
    complete" - a chain nobody walked, vouched for by the gate that exists to
    say so.
    """
    manifest = full_chain()
    holder = {"assemble": manifest, "split": manifest["split"],
              "dedupe": manifest["split"]["dedupe"]}[strip_at]
    del holder[check_key]
    gate = gate_chain(chain_links(manifest), required=True)
    assert gate.status == RED
    assert gate.detail["missing"] == [] and gate.detail["unverified"] == []
    assert check_key in gate.summary
    assert "NEVER RECORDED" in gate.summary


def test_a_manifest_with_every_check_key_stripped_cannot_read_custody_complete():
    """The whole forgery at once, which is how it would actually arrive."""
    manifest = full_chain()
    del manifest["split_check"]
    del manifest["split"]["dedupe_check"]
    del manifest["split"]["dedupe"]["decontamination_check"]
    links = chain_links(manifest)
    assert all(link["present"] for link in links)  # ...every link still THERE
    gate = gate_chain(links, required=True)
    assert gate.status == RED
    assert gate.detail["unrecorded"] == ["split", "dedupe", "decontamination"]
    assert "custody complete" not in gate.summary


def test_the_head_link_is_the_only_one_allowed_to_carry_no_check():
    """assemble is where the walk starts, so nothing upstream verified it - and
    that exemption must not widen to the links that DO have a verifier."""
    links = chain_links(full_chain())
    assert links[0] == {"stage": "assemble", "present": True, "check": None,
                        "check_key": None}
    assert [link["check"] for link in links[1:]] == ["verified"] * 3
    assert gate_chain(links, required=True).status == GREEN


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
    [(17, RED), (18, GREEN), (20, GREEN), (21, RED)],
)
def test_the_empty_think_band_is_closed_at_both_ends(count, status):
    gate = gate_share(GATE_EMPTY_THINK, count, 100, floor=0.18, ceiling=0.20,
                      label="byte-exact empty think")
    assert gate.status == status


def test_the_dead_band_the_rejected_ceiling_left_open_and_the_shipped_one_closes():
    """Why `empty_think_max` was ruled from 0.22 down to 0.20.

    Given the identity assemble.py now ENFORCES - every kept row counted by
    exactly one of the two shares, pinned over the real pipeline below - a 22%
    empty-think corpus is a 78% traced one. Under the plan's 0.22 ceiling that
    corpus passed the empty-think band and failed the trace floor: two gates
    disagreeing about one corpus across a 20-22% dead band. Under the shipped
    0.20 they agree at every share.

    Stated over explicit (count, total) pairs rather than over a corpus built
    from helpers, because the arithmetic is the claim - a fixture would only
    restate how it was assembled.
    """
    for empties, traces in ((22, 78), (21, 79)):
        # The rejected ceiling: green on the band, red on the floor.
        assert gate_share(GATE_EMPTY_THINK, empties, 100, floor=0.18, ceiling=0.22,
                          label="e").status == GREEN
        assert gate_share(GATE_TRACE, traces, 100, floor=0.80, label="t").status == RED
        # The shipped ceiling: both say the same thing about the same corpus.
        assert gate_share(GATE_EMPTY_THINK, empties, 100, floor=0.18, ceiling=0.20,
                          label="e").status == RED
    # 20% is the last share that satisfies both, and both say so.
    assert gate_share(GATE_TRACE, 80, 100, floor=0.80, label="t").status == GREEN
    assert gate_share(GATE_EMPTY_THINK, 20, 100, floor=0.18, ceiling=0.20,
                      label="e").status == GREEN


def test_the_identity_the_ceiling_rests_on_is_enforced_and_not_assumed(tmp_path):
    """`trace + empty_think == emitted`, measured on the far side of the REAL
    assemble.main rather than over a corpus built to satisfy it.

    Three third shapes go in beside a conforming corpus - a real trace flagged
    `reasoning: False`, the empty scaffold on a row claiming a trace, and a
    one-newline block that is neither - and each is dropped by name. What is
    emitted then satisfies the identity, which is what makes the empty-think
    ceiling the trace floor's complement rather than an independent number.
    """
    from tuned.data.assemble import DROP_PROV_MISMATCH

    third_shapes = [traced(50_000), scaffolded(51_000), traced(52_000)]
    third_shapes[0]["_prov"]["reasoning"] = False               # trace, no claim
    third_shapes[1]["_prov"]["reasoning"] = True                # claim, no trace
    third_shapes[2]["messages"][1]["content"] = f"{OPEN}\n{CLOSE}an answer"  # neither
    third_shapes[2]["_prov"]["reasoning"] = False

    _codes, report, paths = run_pipeline(tmp_path, e2e_corpus() + third_shapes)
    drops = list(read_jsonl(paths.out_dir / "assemble_drops.jsonl"))
    assert sorted(d["reason"] for d in drops) == [
        "over_length_bucket", DROP_PROV_MISMATCH, DROP_PROV_MISMATCH, DROP_PROV_MISMATCH,
    ]
    emitted = list(read_jsonl(paths.out_dir / "law_v1_train.jsonl")) + list(
        read_jsonl(paths.out_dir / "law_v1_eval.jsonl")
    )
    assert len(emitted) == 100                                  # 104 in, four dropped
    assert trace_count(emitted) + empty_think_count(emitted, OPEN, CLOSE) == len(emitted)
    # ...and the report grades that same identity, off the same two gates.
    assert (report["gates"][GATE_TRACE]["detail"]["count"]
            + report["gates"][GATE_EMPTY_THINK]["detail"]["count"]
            == report["measurements"]["rows"] == 100)


def test_the_passing_corpus_loses_no_row_to_the_identity_rule(tmp_path):
    """The premise of every number in the end-to-end fixture, asserted rather
    than assumed: every shipped row builder keeps `_prov.reasoning` in step
    with its content, so the rule that drops the third shape drops NOTHING
    from a real corpus. The day that stops being true, this says so - instead
    of a mix share moving for a reason nobody can see.
    """
    from tuned.data.assemble import DROP_NO_SCAFFOLD, DROP_PROV_MISMATCH

    _codes, _report, paths = run_pipeline(tmp_path, e2e_corpus())
    drops = list(read_jsonl(paths.out_dir / "assemble_drops.jsonl"))
    by_reason = Counter(d["reason"] for d in drops)
    assert by_reason[DROP_PROV_MISMATCH] == 0
    assert by_reason[DROP_NO_SCAFFOLD] == 0
    assert by_reason == Counter({"over_length_bucket": 1})


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
    for bound in (0.80, 0.18, 0.20, 0.005):
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
    """101 rows across three streams, one of which cannot fit the bucket.

    Composed to sit inside every band at once, which is not automatic: of the
    100 that survive assemble, 60/16/24 exactly on the mix, 19 byte-exact
    empty-think rows (19%, inside the shipped 18-20% band) and 81 reasoning
    rows (81%, over the 80% floor) - which is the identity, 19 + 81 = 100 - no
    duplicates, no markup, a license on every row. Six two-row cases carry
    CNRs so the split has siblings to keep together and dates to prefer;
    everything else is case-less.

    Nothing is dropped here except the over-length row: dedupe drops ZERO on
    this corpus (the `drop[exact]: 1` belongs to the dup fixture below, which
    feeds it a repeated row on purpose) and assemble's identity rule drops
    zero, both pinned as their own tests.
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
    assignment = split_manifest["assignment"]
    assert assignment["date_assigned_units"] == 3
    assert assignment["hash_assigned_units"] == 4
    assert assignment["date_boundary"] == "2004-00-00"

    # THE HELD-OUT SET IS THE SAME CORPUS AS THE TRAINING SET, which is what
    # the per-source walk bought. Under the pooled walk every DATED atom went
    # to eval before a single date-less one did, and only the citation-bearing
    # sources carry dates - so the eval side was a sample of PredEx. Each
    # source now contributes its own tenth.
    assert {k: v["eval_rows"] for k, v in assignment["by_source"].items()} == {
        "L-NLProc/PredEx_Instruction-Tuning_Pred-Exp": 2,
        "open-thoughts/OpenThoughts-114k": 2,
        "synthesis": 6,
    }
    for source, side in assignment["by_source"].items():
        assert side["eval_rows"] >= side["eval_target_rows"], source
        assert side["eval_target_rows"] == round(side["rows"] * 0.1), source

    # ...and stats can now SEE it. The gates still grade train+eval pooled -
    # that is what the mix targets are sized for - so the held-out reading is
    # report-only and grades nothing.
    side = report["eval_side"]
    assert side["rows"] == 10 and side["grades_nothing"] is True
    assert side["gates"][GATE_TRACE]["detail"]["share"] == 0.8
    assert report["red"] == []  # an eval-side red cannot redden the build
    summary = (paths.out_dir / SUMMARY_FILENAME).read_text(encoding="utf-8")
    assert summary.startswith("#")
    assert "eval side (10 rows, report-only, grades nothing)" in summary
    assert "grounded_synthesis 60.0%/60.0%" in summary


def test_the_over_length_row_is_dropped_by_assemble_and_named(tmp_path):
    _codes, _report, paths = run_pipeline(tmp_path, e2e_corpus())
    drops = list(read_jsonl(paths.out_dir / "assemble_drops.jsonl"))
    assert [(d["side"], d["reason"]) for d in drops] == [("train", "over_length_bucket")]
    # The VALUE, not just `> 8192`. A number nothing pins is a number a report
    # can misquote and a fixture can drift out from under - which is what
    # happened: the round-1 report said 11,180.
    assert (drops[0]["tokens"], drops[0]["limit"]) == (11_120, 8192)
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


def _rescaffold(row_):
    """A traced row turned into a non-reasoning one - CONTENT AND FLAG
    TOGETHER. Moving one without the other is the third shape assemble.py now
    drops, so a fixture that did it would measure the drop rule rather than the
    gate it is aimed at."""
    row_["messages"][1]["content"] = empty_think(OPEN, CLOSE) + row_["messages"][1][
        "content"
    ][len(OPEN):].split(CLOSE, 1)[1]
    row_["_prov"]["reasoning"] = False


def _retrace(row_, seed):
    """The other direction: a scaffolded row given a real trace and the claim
    to go with it."""
    row_["messages"][1]["content"] = (
        f"{OPEN}{prose(seed, 40)}{CLOSE}"
        + row_["messages"][1]["content"][len(empty_think(OPEN, CLOSE)):]
    )
    row_["_prov"]["reasoning"] = True


def red_trace(rows):
    """Two traced rows converted to scaffolded ones: 79 traces, 21 empties.

    Under the enforced identity the trace floor cannot be missed alone - a
    corpus below 80% traces is above 20% empty-think by arithmetic - so this
    variant trips BOTH share gates, and that agreement is the ruling's point.
    """
    out = [json.loads(json.dumps(r)) for r in rows]
    for r in [r for r in out if r["_prov"]["reasoning"]][:2]:
        _rescaffold(r)
    return out


def red_empty_think(rows):
    """Two scaffolded rows converted to traced ones: 17 empties, 83 traces.

    BELOW the floor rather than above the ceiling, which is the only half of
    the band that moves on its own: 17% empty-think is 83% traced, and the
    trace floor is happy at 83%.
    """
    out = [json.loads(json.dumps(r)) for r in rows]
    for seed, r in enumerate([r for r in out if not r["_prov"]["reasoning"]][:2]):
        _retrace(r, 70_000 + seed)
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
    "mutate,red",
    [
        (red_mix, [GATE_MIX]),
        (red_trace, [GATE_TRACE, GATE_EMPTY_THINK]),
        (red_empty_think, [GATE_EMPTY_THINK]),
        (red_markup, [GATE_MARKUP]),
        (red_license, [GATE_LICENSE]),
    ],
)
def test_each_red_variant_exits_non_zero_naming_exactly_the_gates_it_trips(
    tmp_path, mutate, red
):
    """EXACT list equality, so a variant that tripped a gate it was not aimed
    at fails here rather than passing as a cascade.

    `red_trace` names two, and that is arithmetic rather than a cascade: the
    two share gates measure complementary halves of the same corpus, so a
    corpus under the 80% trace floor is over the 20% empty-think ceiling by
    the same rows. Which is exactly the coherence the ceiling was ruled down
    to the floor's complement to get - under the rejected 0.22 ceiling this
    corpus would have reported one red and left the other half green.
    """
    codes, report, _paths = run_pipeline(tmp_path, mutate(e2e_corpus()))
    assert codes["stats"] == 1
    assert report["verdict"] == "red"
    assert report["red"] == red, {g: report["gates"][g]["summary"] for g in report["red"]}
    # Nothing was dropped on the way in: each variant reds a GATE, and a
    # fixture that tripped assemble's identity rule instead would grade a
    # corpus smaller than the one it built.
    assert report["measurements"]["rows"] == 100


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
    assert report["stats_version"] == STATS_VERSION == 2


BANNED_THRESHOLDS = ("0.80", "0.18", "0.20", "0.005", "8192", "2.0", "60", "16", "24")


def banned_literals(source: str, banned=BANNED_THRESHOLDS) -> list[tuple[int, str]]:
    """(line, text) for every shipped threshold written as a literal IN CODE.

    An AST walk and not a regex over the text, for two reasons the old regex
    got wrong in both directions.

    It under-caught: it demanded the literal be followed by `)`, `>`, `=` or
    `,`, so `if share < 0.80:`, `floor = 0.80` and `return share >= 0.80` all
    sailed through - i.e. every form in which a threshold is actually hardcoded
    except the argument one. Only literals reached by the code are considered
    here, in any position.

    It would over-catch if simply widened: this module EXPLAINS its bounds in
    prose ("62/100 against a 0.60 target is 2.0000000000000018 percentage
    points", `float("0.18")`), and a comment about a number is not a number.
    Comments and strings are not in the AST at all, so they are legal by
    construction rather than by an exception.

    Two rules, because a threshold can be spelled two ways:

      * the literal is SPELLED like a bound - `0.80`, or `0.8` for the same
        value written shorter - anywhere in the code;
      * or the literal is an operand of a COMPARISON and equals a bound
        numerically. That is what catches a bare `share < 2` for the 2.0 mix
        tolerance while leaving `round(share, 2)` alone, which is the only
        legitimate use of that value in this module.

    The residual is stated rather than hidden: a bound whose value is a small
    integer (2.0) hardcoded OUTSIDE a comparison - `tolerance = 2` - is not
    caught, because it is indistinguishable from a rounding precision. Every
    place a threshold is USED is a comparison.
    """
    import ast

    tree = ast.parse(source)
    values = {float(b) for b in banned}
    compared = {
        id(operand)
        for node in ast.walk(tree) if isinstance(node, ast.Compare)
        for operand in (node.left, *node.comparators)
    }
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or isinstance(node.value, bool):
            continue
        if not isinstance(node.value, (int, float)):
            continue
        text = ast.get_source_segment(source, node) or repr(node.value)
        spelled = text in banned or ("." in text and float(node.value) in values)
        if spelled or (id(node) in compared and float(node.value) in values):
            hits.append((node.lineno, text))
    return hits


def test_no_threshold_is_written_into_this_module():
    """Every number the gates compare against comes from config. A builder
    gate whose thresholds live in its own source is a gate that gets edited to
    pass."""
    source = STATS_SRC.read_text(encoding="utf-8")
    assert banned_literals(source) == []


def test_the_banned_literal_guard_catches_the_forms_a_threshold_is_hardcoded_in():
    """Guards the guard, against the exact forms that used to escape it.

    The old regex caught one of these five and missed four, which made a test
    named "no threshold is written into this module" a test about where the
    punctuation fell.
    """
    escaped_before = [
        "if share < 0.80:\n    pass\n",                  # comparison, colon after
        "floor = 0.80\n",                                # assignment at end of line
        "def f(share):\n    return share >= 0.80\n",     # returned comparison
        "if report['max'] > 8192:\n    pass\n",          # hardcoded bucket
    ]
    caught_before = ["x = max(share, 0.80)\n"]           # the argument form
    for form in escaped_before + caught_before:
        assert banned_literals(form), form
    # Same value, shorter spelling, and the integer form of the tolerance in
    # the position a threshold is used in.
    assert banned_literals("cut = 0.8\n")
    assert banned_literals("if share < 2:\n    pass\n")
    # ...and the module's own arithmetic, prose and rounding stay legal, or the
    # guard would be traded for a different kind of noise.
    for legal in (
        "y = round(share, 2)\n",
        "z = round(v, 4)\nw = things[:5]\n",
        "rank = ceil(p / 100 * n)\n",
        "# 62/100 against a 0.60 target is 2.0000000000000018 percentage points\nq = 1\n",
        'd = "count/total and float(\\"0.18\\") are the same double"\n',
    ):
        assert banned_literals(legal) == [], legal


def test_the_guard_would_catch_the_mutant_it_exists_for():
    """The review's `R-STATS-HARDCODE-FLOOR` mutant died on a band test rather
    than here, which is what made this guard decorative. Planted in the REAL
    source, it now trips the guard that is named for it."""
    source = STATS_SRC.read_text(encoding="utf-8")
    mutant = source.replace(
        "if floor is not None and share < floor:",
        "if floor is not None and share < 0.80:",
    )
    assert mutant != source, "the line the mutant replaces has moved"
    assert [text for _line, text in banned_literals(mutant)] == ["0.80"]


def test_the_version_ledger_describes_the_version_the_module_ships():
    import re

    source = STATS_SRC.read_text(encoding="utf-8")
    entries = [int(n) for n in re.findall(r"^# (\d+)  ", source, re.M)]
    assert entries == sorted(entries)
    assert entries[-1] == STATS_VERSION
    assert entries[0] == 1
