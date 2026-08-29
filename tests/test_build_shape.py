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
    from tuned.data.shape import MEASURED_RETENTION, generated_counts

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
