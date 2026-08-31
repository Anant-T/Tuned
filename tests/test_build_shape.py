"""shape.py - sizing the corpus to what generation actually produced.

The arithmetic tests carry synthetic pools with the real profile numbers,
because the thing under test IS the arithmetic: which corpus size the
generated rows admit, and how the no-think budget is split. The CLI tests
run the real config against a temp workdir.
"""

import json

import pytest
from pipeline_fakes import open_store, paths_for, temp_config

from tuned.data.config import load_build_config
from tuned.data.jsonl import read_jsonl, write_jsonl
from tuned.data.shape import (
    CURATED_BUCKET,
    MANIFEST_FILENAME,
    REPLAY_BUCKET,
    SYNTHESIS_BUCKET,
    ShapeError,
    classify,
    curated_ceiling,
    is_trace,
    plan,
    select,
    shape_streams,
)
from tuned.data.shape import main as shape_main

MVP = {"grounded_synthesis": 0.301, "curated": 0.2796, "replay": 0.4194}

REPLAY_TRACE = "open-thoughts/OpenThoughts-114k"
REPLAY_NOTHINK = "HuggingFaceTB/smoltalk2:OpenHermes-2.5"
REPLAY_NOTHINK_2 = "allenai/WildChat-4.8M"
CURATED_NOTHINK = "L-NLProc/PredEx_Instruction-Tuning_Pred-Exp"
CURATED_TRACE = "169Pi/indian_law"


def row(source, *, reasoning, i):
    return {
        "messages": [{"role": "user", "content": f"q{i}"},
                     {"role": "assistant", "content": f"a{i}"}],
        "_prov": {"source": source, "native_id": f"{source}-{i}",
                  "license": "Apache-2.0", "reasoning": reasoning},
    }


def rows(source, n, *, reasoning, start=0):
    return [row(source, reasoning=reasoning, i=i) for i in range(start, start + n)]


def big_pools():
    """Pools far larger than any plan below will ask for."""
    return {
        (REPLAY_BUCKET, True): rows(REPLAY_TRACE, 8000, reasoning=True),
        (REPLAY_BUCKET, False): rows(REPLAY_NOTHINK, 4000, reasoning=False),
        (CURATED_BUCKET, False): rows(CURATED_NOTHINK, 4000, reasoning=False),
        (CURATED_BUCKET, True): rows(CURATED_TRACE, 4000, reasoning=True),
    }


# --------------------------------------------------------------------- plan

def test_the_corpus_is_sized_by_the_generated_synthesis_share():
    """grounded_synthesis can only come from the teacher, so it sets the size:
    301 generated rows at a 30.1% target is a 1,000-row corpus."""
    p = plan(big_pools(), targets=MVP, generated_synthesis=301,
             generated_curated=0, replay_nothink_share=0.273)
    assert p.total == 1000
    assert p.binding == SYNTHESIS_BUCKET


def test_the_planned_shares_land_on_the_profile():
    p = plan(big_pools(), targets=MVP, generated_synthesis=301,
             generated_curated=50, replay_nothink_share=0.273)
    for bucket, target in MVP.items():
        assert p.shares[bucket] == pytest.approx(target, abs=0.005), bucket


def test_the_planned_no_think_share_lands_in_the_gate_window():
    """18-20% is the gate; the default aim is the midpoint."""
    p = plan(big_pools(), targets=MVP, generated_synthesis=301,
             generated_curated=50, replay_nothink_share=0.273)
    assert 0.18 <= p.empty_share <= 0.20


def test_the_no_think_budget_moves_between_replay_and_curated():
    """--replay-nothink-share is the one free choice: the SAME budget, taken
    from chat rows or from raw legal rows. Total no-think does not move."""
    chat = plan(big_pools(), targets=MVP, generated_synthesis=301,
                generated_curated=50, replay_nothink_share=0.273)
    legal = plan(big_pools(), targets=MVP, generated_synthesis=301,
                 generated_curated=50, replay_nothink_share=0.0)
    assert chat.empty_share == pytest.approx(legal.empty_share, abs=0.005)
    assert legal.demand[(REPLAY_BUCKET, False)] == 0
    assert legal.demand[(CURATED_BUCKET, False)] > chat.demand[(CURATED_BUCKET, False)]
    assert legal.demand[(CURATED_BUCKET, True)] < chat.demand[(CURATED_BUCKET, True)]


def test_a_short_pool_shrinks_the_corpus_within_the_mix_tolerance():
    """The pools, not the teacher, bind here - but only inside the band where
    the generated share still passes the mix gate."""
    pools = big_pools()
    pools[(REPLAY_BUCKET, True)] = rows(REPLAY_TRACE, 300, reasoning=True)
    p = plan(pools, targets=MVP, generated_synthesis=301, generated_curated=50,
             replay_nothink_share=0.273)
    assert p.binding == "pool"
    assert p.total < 1000
    assert p.request[(REPLAY_BUCKET, True)] <= 300
    assert p.shares[SYNTHESIS_BUCKET] == pytest.approx(0.301, abs=0.02)


def test_a_pool_too_short_for_any_admissible_size_refuses():
    """Never silently ship a corpus whose synthesis share is double its
    target - that is what shrinking without a floor would produce."""
    pools = big_pools()
    pools[(REPLAY_BUCKET, True)] = rows(REPLAY_TRACE, 5, reasoning=True)
    with pytest.raises(ShapeError, match="replay/trace"):
        plan(pools, targets=MVP, generated_synthesis=301, generated_curated=50,
             replay_nothink_share=0.273)


def test_generated_curated_rows_come_out_of_the_curated_trace_demand():
    """curated_c2 rows carry traces, so each one displaces a raw 169Pi row
    rather than adding to the curated bucket."""
    none = plan(big_pools(), targets=MVP, generated_synthesis=301,
                generated_curated=0, replay_nothink_share=0.273)
    some = plan(big_pools(), targets=MVP, generated_synthesis=301,
                generated_curated=40, replay_nothink_share=0.273)
    assert none.total == some.total
    assert none.demand[(CURATED_BUCKET, True)] - some.demand[(CURATED_BUCKET, True)] == 40


def test_curated_generation_that_overfills_its_bucket_refuses_with_the_reason():
    with pytest.raises(ShapeError, match="overfill"):
        plan(big_pools(), targets=MVP, generated_synthesis=301,
             generated_curated=900, replay_nothink_share=0.273)


def test_no_generated_synthesis_refuses_rather_than_shipping_the_pools_whole():
    with pytest.raises(ShapeError, match="nothing has been generated"):
        plan(big_pools(), targets=MVP, generated_synthesis=0, generated_curated=10)


def test_retention_correction_requests_more_than_it_keeps():
    """Rows requested here lose ~15% (PredEx) before assemble counts them."""
    pools = big_pools()
    p = plan(pools, targets=MVP, generated_synthesis=301, generated_curated=50,
             replay_nothink_share=0.273,
             retention={(CURATED_BUCKET, False): 0.5})
    key = (CURATED_BUCKET, False)
    assert p.request[key] == pytest.approx(p.demand[key] * 2, abs=1)


def test_the_default_replay_nothink_share_is_the_pools_own_composition():
    """The default preserves what the slice names say the design wants -
    no-think trained on chat, not on legal prediction."""
    p = plan(big_pools(), targets=MVP, generated_synthesis=301, generated_curated=50)
    built = 4000 / (4000 + 8000)
    assert p.demand[(REPLAY_BUCKET, False)] / (
        p.demand[(REPLAY_BUCKET, False)] + p.demand[(REPLAY_BUCKET, True)]
    ) == pytest.approx(built, abs=0.01)


# ----------------------------------------------------------------- classify

def test_an_unmapped_source_refuses_rather_than_defaulting_to_a_bucket(tmp_path):
    """A row belonging to no bucket would be shipped outside every target
    that sized the corpus - fatal here, not merely red two stages later."""
    cfg = load_build_config(temp_config(tmp_path), allow_unpinned=True)
    with pytest.raises(ShapeError, match="no assembly bucket"):
        classify([row("nobody/mapped-this", reasoning=True, i=0)], cfg.assembly)


def test_trace_reads_the_same_field_the_stats_gate_reads():
    assert is_trace(row(REPLAY_TRACE, reasoning=True, i=0))
    assert not is_trace(row(REPLAY_NOTHINK, reasoning=False, i=0))
    assert not is_trace({"_prov": {}})


# ------------------------------------------------------------------- select

def test_select_preserves_the_per_source_proportions():
    """600 smoltalk : 300 wildchat is a designed sub-mix; a flat hash cut
    would let it drift."""
    pool = rows(REPLAY_NOTHINK, 600, reasoning=False) + \
        rows(REPLAY_NOTHINK_2, 300, reasoning=False)
    picked = select(pool, 300)
    assert len(picked) == 300
    by_source = {}
    for r in picked:
        by_source[r["_prov"]["source"]] = by_source.get(r["_prov"]["source"], 0) + 1
    assert by_source[REPLAY_NOTHINK] == 200
    assert by_source[REPLAY_NOTHINK_2] == 100


def test_select_is_deterministic_and_independent_of_input_order():
    """Two runs of the assemble chain over the same inputs must produce the
    same dataset, including after a stream file is rewritten in a new order."""
    pool = rows(REPLAY_NOTHINK, 500, reasoning=False)
    first = [r["_prov"]["native_id"] for r in select(pool, 50)]
    second = [r["_prov"]["native_id"] for r in select(list(reversed(pool)), 50)]
    assert first == second


def test_a_bigger_corpus_is_a_superset_of_a_smaller_one():
    """A row must never leave the dataset because the corpus grew - that
    would silently re-roll every earlier decontamination decision."""
    pool = rows(REPLAY_NOTHINK, 600, reasoning=False) + \
        rows(REPLAY_NOTHINK_2, 300, reasoning=False)
    small = {r["_prov"]["native_id"] for r in select(pool, 150)}
    big = {r["_prov"]["native_id"] for r in select(pool, 450)}
    assert small <= big


def test_select_returns_everything_when_asked_for_more_than_the_pool():
    pool = rows(REPLAY_NOTHINK, 10, reasoning=False)
    assert len(select(pool, 99)) == 10
    assert select(pool, 0) == []


# ------------------------------------------------------------ shape_streams

def test_shape_streams_splits_the_selection_back_across_its_source_files(tmp_path):
    cfg = load_build_config(temp_config(tmp_path), allow_unpinned=True)
    stream_rows = {
        "replay": rows(REPLAY_TRACE, 800, reasoning=True)
        + rows(REPLAY_NOTHINK, 400, reasoning=False),
        "curated_c1": rows(CURATED_NOTHINK, 400, reasoning=False)
        + rows(CURATED_TRACE, 200, reasoning=True),
    }
    pools = classify([r for rs in stream_rows.values() for r in rs], cfg.assembly)
    p = plan(pools, targets=MVP, generated_synthesis=90, generated_curated=10,
             replay_nothink_share=0.273)
    shaped = shape_streams(stream_rows, p, cfg.assembly)
    assert set(shaped) == {"replay", "curated_c1"}
    for name, out in shaped.items():
        assert len(out) < len(stream_rows[name])
    kept = sum(len(v) for v in shaped.values())
    assert kept == sum(p.request.values())


# ---------------------------------------------------------------------- CLI

def _write_streams(paths):
    write_jsonl(paths.streams_dir / "replay.jsonl",
                rows(REPLAY_TRACE, 800, reasoning=True)
                + rows(REPLAY_NOTHINK, 400, reasoning=False))
    write_jsonl(paths.streams_dir / "curated_c1.jsonl",
                rows(CURATED_NOTHINK, 400, reasoning=False)
                + rows(CURATED_TRACE, 200, reasoning=True))


def _accept(store, stream, n, *, start=0):
    for i in range(start, start + n):
        task_id = f"{stream}-{i}"
        store.create_tasks([{
            "task_id": task_id, "seed_id": "seed000", "stream": stream,
            "task_type": "irac_analysis", "prompt_id": "irac_analysis_v1",
            "prompt_sha": "abc", "sample_ix": 0,
        }])
        store.record_generation({
            "task_id": task_id, "attempt": 1, "provider": "p", "model": "m",
            "raw_path": "raw.jsonl", "raw_offset": 0, "think": "t", "answer": "a",
        })
        store.set_task_state(task_id, "accepted")


def test_generated_counts_discount_the_rows_the_chain_will_drop(tmp_path):
    """AN ACCEPTED TASK IS NOT AN ASSEMBLED ROW. Decontamination, dedupe and
    the length cut take 14-19% of the generations; sizing off the raw count
    holds the generated numerator while the chain shrinks the denominator,
    and lands every share of the mix wrong at once. That is exactly what the
    first shaped rehearsal did - grounded_synthesis 27.7% against a 30.1%
    target, with all four stream pools individually on target."""
    from tuned.data.shape import DEFAULT_RETENTION, MEASURED_RETENTION, generated_counts

    cfg = load_build_config(temp_config(tmp_path), allow_unpinned=True)
    paths = paths_for(tmp_path)
    store = open_store(tmp_path, n_seeds=1, db_path=paths.state_db)
    _accept(store, "synthesis", 100)
    _accept(store, "curated_c2", 100)
    effective, accepted = generated_counts(store, cfg.assembly)
    raw, _ = generated_counts(store, cfg.assembly, correct=False)
    store.close()

    assert accepted == {SYNTHESIS_BUCKET: 100, CURATED_BUCKET: 100}
    assert raw == {SYNTHESIS_BUCKET: 100, CURATED_BUCKET: 100}
    assert effective[SYNTHESIS_BUCKET] == round(100 * MEASURED_RETENTION["synthesis"])
    assert effective[CURATED_BUCKET] == round(100 * MEASURED_RETENTION["curated_c2"])
    # Both streams now carry a reading, and both readings are below the
    # default - so the discount is larger than the fallback ever applied.
    assert effective[CURATED_BUCKET] < round(100 * DEFAULT_RETENTION)
    assert effective[SYNTHESIS_BUCKET] < accepted[SYNTHESIS_BUCKET]


def test_the_cli_writes_shaped_streams_and_a_manifest(tmp_path, capsys):
    config = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    _write_streams(paths)
    store = open_store(tmp_path, n_seeds=1, db_path=paths.state_db)
    _accept(store, "synthesis", 90)
    _accept(store, "curated_c2", 10)
    store.close()

    assert shape_main(["--config", config, "--profile", "v1.0-MVP"]) == 0

    manifest = json.loads((paths.out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["profile"] == "v1.0-MVP"
    assert manifest["generated_accepted"] == {SYNTHESIS_BUCKET: 90, CURATED_BUCKET: 10}
    # Both are recorded: the accepted count is what an operator sees in the
    # store, the effective count is what sized the corpus, and a gap between
    # them is the single most confusing thing about this manifest.
    assert manifest["generated_effective"][SYNTHESIS_BUCKET] < 90
    assert manifest["projected_shares"][SYNTHESIS_BUCKET] == pytest.approx(0.301, abs=0.02)
    assert 0.18 <= manifest["projected_nothink"] <= 0.20

    for name in ("replay", "curated_c1"):
        out = paths.out_dir / f"shaped_{name}.jsonl"
        assert out.exists()
        assert 0 < len(list(read_jsonl(out))) < len(
            list(read_jsonl(paths.streams_dir / f"{name}.jsonl"))
        )


def test_the_cli_leaves_the_pools_untouched(tmp_path):
    """A later run with more generated rows must re-derive a bigger corpus
    from the same inputs, so the pools are read-only here."""
    config = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    _write_streams(paths)
    before = {p.name: p.read_bytes() for p in paths.streams_dir.glob("*.jsonl")}
    store = open_store(tmp_path, n_seeds=1, db_path=paths.state_db)
    _accept(store, "synthesis", 90)
    store.close()

    assert shape_main(["--config", config, "--profile", "v1.0-MVP"]) == 0
    after = {p.name: p.read_bytes() for p in paths.streams_dir.glob("*.jsonl")}
    assert before == after


def test_the_cli_refuses_when_nothing_has_been_generated(tmp_path, capsys):
    config = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    _write_streams(paths)
    open_store(tmp_path, n_seeds=1, db_path=paths.state_db).close()

    assert shape_main(["--config", config, "--profile", "v1.0-MVP"]) == 2
    assert "REFUSED" in capsys.readouterr().out
    assert not (paths.out_dir / MANIFEST_FILENAME).exists()


# --------------------------------------------------------------------------
# --measure: the instrument MEASURED_RETENTION is made of.
#
# The table used to be a set of round numbers with a comment promising a
# --measure that did not exist. PredEx sat at 0.900 against a real 0.846 and
# WildChat at 0.955 against 0.910, and this table sizes the WHOLE corpus - a
# wrong entry lands every share of the mix wrong at once.
# --------------------------------------------------------------------------

def _prov_row(source, i=0):
    return {"messages": [{"role": "user", "content": f"q{i}"}], "_prov": {"source": source}}


def _chain(out_dir, *, kept, drops, shipped):
    from tuned.data.assemble import EVAL_FILENAME, TRAIN_FILENAME
    from tuned.data.decontaminate import DROPS_FILENAME, OUT_FILENAME

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / OUT_FILENAME, [_prov_row(s, i) for i, s in enumerate(kept)])
    write_jsonl(out_dir / DROPS_FILENAME, [{"source": s, "reason": "x"} for s in drops])
    write_jsonl(out_dir / TRAIN_FILENAME, [_prov_row(s, i) for i, s in enumerate(shipped)])
    write_jsonl(out_dir / EVAL_FILENAME, [])


def test_retention_is_shipped_over_what_entered_not_over_what_survived(tmp_path):
    """kept + dropped IS what entered - the identity decontamination.json
    states in its own counts - so no stage between decontaminate and assemble
    has to be enumerated here, and one added later cannot be forgotten."""
    from tuned.data.shape import retention_report

    _chain(
        tmp_path / "out",
        kept=["predex"] * 77 + ["wildchat"] * 60,
        drops=["predex"] * 23,
        # 7 of the surviving predex rows and 6 wildchat rows are lost further
        # down the chain (dedupe, the per-case cap, the over-length drop).
        shipped=["predex"] * 70 + ["wildchat"] * 54,
    )
    report = retention_report(tmp_path / "out")
    assert report["predex"] == {
        "entered": 100, "shipped": 70, "retention": 0.7, "reportable": True,
    }
    assert report["wildchat"]["retention"] == 0.9


def test_a_figure_is_withheld_below_the_floor_and_the_counts_are_not(tmp_path, capsys):
    """A retention fitted on fifteen rows is not a measurement. It is also not
    a zero, which is why the counts are still printed beside the withheld
    ratio - the generated streams sit here today."""
    from tuned.data.shape import RETENTION_MIN_N, print_retention, retention_report

    _chain(tmp_path / "out", kept=["synthesis"] * 15, drops=[], shipped=["synthesis"] * 12)
    report = retention_report(tmp_path / "out")
    assert report["synthesis"]["reportable"] is False
    assert report["synthesis"]["entered"] == 15
    # ...and the ratio is still computed, so a caller that knows what it is
    # doing can read it; it is the PRINTED table that withholds it.
    assert report["synthesis"]["retention"] == 0.8
    print_retention(report)
    out = capsys.readouterr().out
    assert f"n<{RETENTION_MIN_N}" in out and "0.800" not in out
    assert "15" in out and "12" in out


def test_a_half_run_chain_refuses_rather_than_reporting_its_missing_stages(tmp_path):
    """Losses and stages-that-have-not-run are the same arithmetic and
    completely different facts."""
    from tuned.data.shape import retention_report

    _chain(tmp_path / "out", kept=["a"] * 60, drops=[], shipped=["a"] * 60)
    (tmp_path / "out" / "law_v1_train.jsonl").unlink()
    with pytest.raises(ShapeError, match="COMPLETED chain"):
        retention_report(tmp_path / "out")


def test_a_drops_file_without_source_refuses_instead_of_reading_form(tmp_path):
    """`form` prefers a row's task_type, so a generated row's drop files under
    `irac_analysis` while the row ships as `synthesis`. Falling back to it
    would be exact for every file-based source and silently wrong for exactly
    the streams the retention correction exists for."""
    from tuned.data.decontaminate import DROPS_FILENAME
    from tuned.data.shape import retention_report

    _chain(tmp_path / "out", kept=["a"] * 60, drops=[], shipped=["a"] * 60)
    write_jsonl(tmp_path / "out" / DROPS_FILENAME, [{"form": "irac_analysis", "reason": "x"}])
    with pytest.raises(ShapeError, match="predates the drop record"):
        retention_report(tmp_path / "out")


def test_measure_shapes_nothing_and_writes_nothing(tmp_path, capsys):
    config = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    _chain(paths.out_dir, kept=["predex"] * 60, drops=["predex"] * 40, shipped=["predex"] * 55)

    assert shape_main(["--config", config, "--measure"]) == 0
    assert "0.550" in capsys.readouterr().out
    assert not (paths.out_dir / MANIFEST_FILENAME).exists()
    assert not list(paths.out_dir.glob("shaped_*.jsonl"))


def test_the_table_holds_no_number_that_was_never_measured():
    """A stream that has not put RETENTION_MIN_N rows through the chain stays
    ABSENT rather than being filled in: a value there would be invented, and
    DEFAULT_RETENTION is the honest answer until it has. `transition` has
    shipped 3 rows of 5 and is still on the wrong side of that floor."""
    from tuned.data.shape import MEASURED_RETENTION

    assert "transition" not in MEASURED_RETENTION
    # The reading that made the point: the old table said 0.900.
    assert MEASURED_RETENTION["L-NLProc/PredEx_Instruction-Tuning_Pred-Exp"] == 0.846
    assert MEASURED_RETENTION["allenai/WildChat-4.8M"] == 0.910


def test_both_generated_streams_that_cleared_the_floor_carry_a_reading():
    """synthesis and curated_c2 were re-fit on 2026-08-31 off the first chain
    to ship 50+ generated rows (447 and 491). Both land BELOW the 0.95
    default, so the default was optimistic in exactly the place the correction
    exists for - and curated_c2, which had no entry at all, was the more
    optimistic of the two by 14%."""
    from tuned.data.shape import DEFAULT_RETENTION, MEASURED_RETENTION

    assert MEASURED_RETENTION["synthesis"] == 0.846
    assert MEASURED_RETENTION["curated_c2"] == 0.817
    assert MEASURED_RETENTION["synthesis"] < DEFAULT_RETENTION
    assert MEASURED_RETENTION["curated_c2"] < DEFAULT_RETENTION


def test_the_synthesis_reading_replaced_a_placeholder_that_flattered_it():
    """0.857 was never a reading - it was a guess kept in the table because
    deleting it fell back to a HIGHER number on no evidence. The measurement
    it was standing in for came in below it, which is the direction that
    matters: the guess was the optimistic one too."""
    from tuned.data.shape import MEASURED_RETENTION

    assert MEASURED_RETENTION["synthesis"] < 0.857


# --------------------------------------------------------------------------
# curated_ceiling: the point past which no amount of synthesis rescues the
# corpus. Generated rows CANNOT be dropped, so crossing it is irreversible.
# --------------------------------------------------------------------------

def _any_synthesis_works(pools, gc, *, targets, share=None):
    for gs in range(25, 6 * gc + 2000, 25):
        try:
            plan(pools, targets=targets, generated_synthesis=gs, generated_curated=gc,
                 replay_nothink_share=share, tolerance_pp=2.0)
            return True
        except ShapeError:
            pass
    return False


def test_curated_ceiling_is_where_no_synthesis_count_can_rescue_the_corpus():
    """The failure this exists to make visible: past the ceiling the corpus is
    unassemblable FOREVER, because shape trims stream files and decontaminate
    reads every accepted generation - so a curated row, once generated, cannot
    be taken back out.

    The short replay/nothink pool is what caps the corpus, and the corpus size
    is what caps how many generated curated rows fit inside the curated share.
    """
    pools = {
        (REPLAY_BUCKET, True): rows(REPLAY_TRACE, 8000, reasoning=True),
        (REPLAY_BUCKET, False): rows(REPLAY_NOTHINK, 200, reasoning=False),
        (CURATED_BUCKET, False): rows(CURATED_NOTHINK, 4000, reasoning=False),
        (CURATED_BUCKET, True): rows(CURATED_TRACE, 4000, reasoning=True),
    }
    ceiling = curated_ceiling(pools, targets=MVP, max_curated=4000)
    assert ceiling is not None and ceiling > 0
    # what it returns is really feasible ...
    assert _any_synthesis_works(pools, ceiling, targets=MVP)
    # ... and past it nothing is, which is the whole point. Probed well clear
    # of the boundary rather than one step past it: the sweep inside
    # curated_ceiling is deliberately coarse and errs LOW, so a value just
    # above what it returns may still be feasible - that is the documented
    # direction of the error, not a defect this test should encode.
    assert not _any_synthesis_works(pools, 2 * ceiling, targets=MVP)


def test_curated_ceiling_is_none_when_the_pools_admit_no_corpus_at_all():
    """A None here is the difference between "stop generating curated" and
    "these pools cannot build this profile" - opposite instructions."""
    starved = {
        (REPLAY_BUCKET, True): rows(REPLAY_TRACE, 5, reasoning=True),
        (REPLAY_BUCKET, False): rows(REPLAY_NOTHINK, 1, reasoning=False),
        (CURATED_BUCKET, False): rows(CURATED_NOTHINK, 1, reasoning=False),
        (CURATED_BUCKET, True): rows(CURATED_TRACE, 1, reasoning=True),
    }
    assert curated_ceiling(starved, targets=MVP) is None


def test_curated_ceiling_rises_when_the_binding_pool_is_larger():
    """It is the SHORT pool that sets the ceiling, so the remedy that keeps the
    generated work is rebuilding that pool rather than throttling generation."""
    def pools_with(nothink):
        return {
            (REPLAY_BUCKET, True): rows(REPLAY_TRACE, 6000, reasoning=True),
            (REPLAY_BUCKET, False): rows(REPLAY_NOTHINK, nothink, reasoning=False),
            (CURATED_BUCKET, False): rows(CURATED_NOTHINK, 6000, reasoning=False),
            (CURATED_BUCKET, True): rows(CURATED_TRACE, 6000, reasoning=True),
        }
    # Small numbers on purpose: the claim is directional, and the search costs
    # O(ceiling) plans, so proving it at 20k pools buys nothing but minutes.
    small = curated_ceiling(pools_with(200), targets=MVP, max_curated=3000)
    large = curated_ceiling(pools_with(600), targets=MVP, max_curated=3000)
    assert large > small


def test_headroom_reports_the_ceiling_and_shapes_nothing(tmp_path, capsys):
    """The number an operator actually needs and had no command for.

    `shape` answers yes/no about TODAY's counts while the queue keeps
    generating, so "how many more curated rows may this build produce before
    it can never be assembled" was unanswerable without writing a script.
    """
    config = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    _write_streams(paths)
    store = open_store(tmp_path, n_seeds=1, db_path=paths.state_db)
    _accept(store, "synthesis", 90)
    _accept(store, "curated_c2", 10)
    store.close()

    assert shape_main(["--config", config, "--profile", "v1.0-MVP", "--headroom"]) == 0
    out = capsys.readouterr().out
    assert "generated-curated headroom" in out
    assert "ceiling" in out and "headroom" in out
    # and it is a REPORT: nothing shaped, nothing written
    assert not (paths.out_dir / MANIFEST_FILENAME).exists()
    assert not list(paths.out_dir.glob("shaped_*.jsonl"))
