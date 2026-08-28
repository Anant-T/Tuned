"""difficulty.py - one probe calibration, then a length proxy for everything.

The architectural constraint is the thing under test here. Per-row probing was
costed at 32M tokens / 65 days, so it must not merely be absent today - it must
be UNREACHABLE from the labelling path, and that is checked structurally (the
syntax tree of the functions that see the corpus) as well as behaviourally (a
probe that explodes if called, driven through the whole labelling path).

Everything statistical runs on injected probe outcomes with known answers.
"""

import ast
import dataclasses
import json
from pathlib import Path

import pytest

from tuned.data import difficulty as D
from tuned.data.config import load_build_config
from tuned.data.store import Store

DATA_CONFIG = Path(__file__).parent.parent / "data" / "configs" / "data_law_v1.yaml"
DIFFICULTY_SRC = Path(D.__file__)


@pytest.fixture(scope="module")
def cfg():
    return load_build_config(DATA_CONFIG, allow_unpinned=True)


@pytest.fixture
def store(tmp_path):
    with Store.open(tmp_path / "state" / "law_v1.sqlite3") as s:
        yield s


def _corpus(n=1000, *, start=100, step=1):
    """n rows whose lengths are strictly increasing, so quantiles are exact."""
    return [
        {"seed_id": f"sd{i:05d}", "token_count": start + i * step, "text": "x" * 40}
        for i in range(n)
    ]


def _outcomes(rows, *, solved_of):
    """Probe outcomes over `rows`; `solved_of(length) -> bool`."""
    return [
        D.ProbeOutcome(
            row_id=row["seed_id"],
            length=D.row_length(row),
            solved=solved_of(D.row_length(row)),
        )
        for row in rows
    ]


def _declining_proxy(rows, cfg):
    """The honest shape: short rows solved, long rows not."""
    cut = sorted(D.row_length(row) for row in rows)[len(rows) // 2]
    return _outcomes(rows, solved_of=lambda length: length <= cut)


# --------------------------------------------------------------------------
# Lengths.
# --------------------------------------------------------------------------

def test_a_row_with_a_token_count_is_measured_by_it():
    assert D.row_length({"token_count": 812, "text": "short"}) == 812


def test_a_row_without_one_falls_back_to_the_same_estimate_the_builders_use():
    assert D.row_length({"text": "x" * 400}) == 100  # chars/4
    assert D.CHARS_PER_TOKEN == 4


@pytest.mark.parametrize("row", [{}, {"token_count": None}, {"token_count": 0}, {"text": None}])
def test_a_row_with_no_usable_length_is_zero_and_never_none(row):
    assert D.row_length(row) == 0


def test_a_nonsense_token_count_does_not_win_over_the_text():
    # A negative or zero count is not a length; falling through to the text is
    # the only reading that does not put the row in the easy band by accident.
    assert D.row_length({"token_count": -5, "text": "x" * 800}) == 200


# --------------------------------------------------------------------------
# The probe ceiling - the architectural constraint.
# --------------------------------------------------------------------------

def test_the_probe_sample_is_capped_at_the_configured_ceiling(cfg):
    rows = _corpus(18000)
    sample = D.probe_sample(rows, cfg)
    assert len(sample) == cfg.difficulty.probe_sample == 1000
    assert len({row["seed_id"] for row in sample}) == 1000


def test_a_corpus_smaller_than_the_ceiling_is_probed_whole(cfg):
    rows = _corpus(300)
    assert len(D.probe_sample(rows, cfg)) == 300


def test_the_probe_sample_is_the_same_rows_every_time(cfg):
    rows = _corpus(5000)
    first = [row["seed_id"] for row in D.probe_sample(rows, cfg)]
    second = [row["seed_id"] for row in D.probe_sample(list(reversed(rows)), cfg)]
    assert first == second


def test_the_probe_sample_spans_the_length_range(cfg):
    """A sample that misses the long tail cannot answer the one question the
    probe is paid to answer.

    Coverage is at STRATUM granularity, not row granularity: the corpus is cut
    into one stratum per sample slot and the row taken from each is the one
    with the lowest sha, so the globally shortest row need not be drawn. What
    must hold is that the draw reaches into the first and last strata - which
    is what "spans the range" can mean when which row is deliberately not a
    choice anybody made.
    """
    rows = _corpus(18000)
    sample = D.probe_sample(rows, cfg)
    lengths = sorted(D.row_length(row) for row in sample)
    everything = sorted(D.row_length(row) for row in rows)
    stride = len(everything) / cfg.difficulty.probe_sample
    assert lengths[0] <= everything[int(stride)]
    assert lengths[-1] >= everything[-int(stride) - 1]
    # And it is a spread, not a clump at one end.
    assert lengths[len(lengths) // 2] == pytest.approx(everything[len(everything) // 2], abs=stride)


def test_sending_more_rows_than_the_ceiling_to_the_probe_is_refused(cfg):
    rows = _corpus(cfg.difficulty.probe_sample + 1)
    with pytest.raises(ValueError, match="above the difficulty.probe_sample ceiling"):
        D.probe_rows(rows, ["answer"] * len(rows), cfg)


def test_exactly_the_ceiling_is_allowed(cfg):
    # The other direction, so the refusal above is a boundary and not a ban.
    rows = _corpus(cfg.difficulty.probe_sample)
    assert len(D.probe_rows(rows, ["answer"] * len(rows), cfg)) == cfg.difficulty.probe_sample


def test_replies_are_paired_by_position_and_a_mismatch_is_refused(cfg):
    rows = _corpus(5)
    with pytest.raises(ValueError, match="paired by position"):
        D.probe_rows(rows, ["a", "b"], cfg)


# --------------------------------------------------------------------------
# Grading a reply.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,declined",
    [
        ("Section 302 IPC applies.", False),
        ("I do not know.", True),
        ("I don't know which provision governs.", True),
        ("I am\n   not sure.", True),           # normalised across the line break
        ("I'm not sure, but possibly s.302.", True),
        ("", True),
        (None, True),
        ("   ", True),
        ("The answer is that no idea is needed here.", True),  # honest false positive
    ],
)
def test_the_decline_detector_reads_both_ways(text, declined):
    assert D.declined_to_answer(text) is declined
    assert D.default_grade({}, text) is not declined


def test_probe_outcomes_carry_the_row_the_length_and_the_grade(cfg):
    rows = _corpus(3)
    outcomes = D.probe_rows(rows, ["an answer", "I do not know", "another"], cfg)
    assert [o.solved for o in outcomes] == [True, False, True]
    assert [o.row_id for o in outcomes] == ["sd00000", "sd00001", "sd00002"]
    assert [o.length for o in outcomes] == [100, 101, 102]


def test_a_caller_may_supply_its_own_grader(cfg):
    # The seam the transition stream's answer keys would use.
    rows = _corpus(2)
    outcomes = D.probe_rows(rows, ["x", "y"], cfg, grade=lambda row, text: text == "x")
    assert [o.solved for o in outcomes] == [True, False]


# --------------------------------------------------------------------------
# Calibration: the proxy is checked, not assumed.
# --------------------------------------------------------------------------

def test_bands_are_cut_at_the_targets_quantiles(cfg):
    rows = _corpus(1000)
    bands = D.calibrate_bands(_declining_proxy(rows, cfg), cfg)
    lengths = sorted(D.row_length(row) for row in rows)
    assert bands.easy_max == lengths[340 - 1]
    assert bands.medium_max == lengths[840 - 1]
    assert bands.target == {"easy": 0.34, "medium": 0.50, "hard": 0.16}
    assert bands.n_probed == 1000


def test_the_cumulative_cut_is_not_moved_by_binary_floating_point():
    """0.34 + 0.50 is 0.8400000000000001, so a bare ceil() asks for rank 841
    where the share names 840 - the medium/hard boundary moving one row for a
    reason nobody could find later. Same arithmetic as config._SHARE_EPS."""
    lengths = list(range(1, 1001))
    assert D._quantile(lengths, 0.34 + 0.50) == 840
    assert D._quantile(lengths, 0.84) == 840
    # A share that genuinely falls between ranks still rounds UP - the epsilon
    # closes a representation gap, it does not change the rule.
    assert D._quantile(lengths, 0.8405) == 841
    assert D._quantile(lengths, 0.0) == 1        # never rank 0
    assert D._quantile(lengths, 1.0) == 1000     # never past the end
    assert D._quantile([], 0.5) == 0


def test_a_proxy_that_points_the_right_way_is_accepted(cfg):
    rows = _corpus(600)
    bands = D.calibrate_bands(_declining_proxy(rows, cfg), cfg)
    assert bands.solve_rates["easy"] > bands.solve_rates["hard"]


def test_an_inverted_proxy_is_refused_with_the_numbers(cfg):
    # The long rows are the ones the weak model answers. Length is then
    # tracking something, but not difficulty, and labelling 18,000 rows off it
    # would be inventing a signal.
    rows = _corpus(600)
    cut = sorted(D.row_length(row) for row in rows)[len(rows) // 2]
    inverted = _outcomes(rows, solved_of=lambda length: length > cut)
    with pytest.raises(D.ProxyRefused, match="not tracking difficulty"):
        D.calibrate_bands(inverted, cfg)


def test_a_flat_proxy_is_refused_too(cfg):
    # Equal solve rates are not evidence FOR the proxy, and `>` rather than
    # `>=` is what makes the refusal say so.
    rows = _corpus(600)
    flat = _outcomes(rows, solved_of=lambda length: True)
    with pytest.raises(D.ProxyRefused, match="not tracking difficulty"):
        D.calibrate_bands(flat, cfg)


def test_an_empty_probe_result_set_is_refused(cfg):
    with pytest.raises(D.ProxyRefused, match="nothing to calibrate"):
        D.calibrate_bands([], cfg)


def test_a_probe_that_only_covers_one_end_is_refused(cfg):
    # Every probed row lands in the easy band, so the hard band's rate is None
    # and the comparison has nothing to compare. A proxy checked at one end
    # only has not been checked.
    rows = [{"seed_id": f"sd{i}", "token_count": 10} for i in range(50)]
    outcomes = _outcomes(rows, solved_of=lambda length: True)
    with pytest.raises(D.ProxyRefused, match="does not cover both ends"):
        D.calibrate_bands(outcomes, cfg)


def test_the_refusal_carries_the_measured_rates_so_the_operator_argues_with_a_number(cfg):
    rows = _corpus(600)
    cut = sorted(D.row_length(row) for row in rows)[len(rows) // 2]
    inverted = _outcomes(rows, solved_of=lambda length: length > cut)
    with pytest.raises(D.ProxyRefused) as caught:
        D.calibrate_bands(inverted, cfg)
    message = str(caught.value)
    assert "600 probed rows" in message
    assert "per-band rates" in message
    assert "%" in message


# --------------------------------------------------------------------------
# Labelling, and the mix it has to deliver.
# --------------------------------------------------------------------------

def test_the_labels_hit_the_configured_mix_on_a_synthetic_corpus(cfg):
    rows = _corpus(5000)
    bands = D.calibrate_bands(_declining_proxy(D.probe_sample(rows, cfg), cfg), cfg)
    labelled, mix = D.label_corpus(rows, bands, cfg)
    target = D.target_shares(cfg)
    for label in D.LABELS:
        assert abs(mix[label] - target[label]) <= cfg.difficulty.mix_tolerance, (label, mix)
    assert len(labelled) == 5000
    assert {row["difficulty"] for row in labelled} == set(D.LABELS)


def test_the_bands_put_short_rows_easy_and_long_rows_hard():
    bands = D.Bands(easy_max=100, medium_max=200, n_probed=10, solve_rates={}, target={})
    assert bands.label(1) == "easy"
    assert bands.label(100) == "easy"
    assert bands.label(101) == "medium"
    assert bands.label(200) == "medium"
    assert bands.label(201) == "hard"


def test_a_mix_outside_tolerance_is_refused_rather_than_waved_through(cfg):
    # Every row the same length: the whole corpus lands on one side of the cut
    # whatever the quantile does. That is a fact about the corpus, and the
    # refusal names the drift rather than rounding it away.
    rows = [{"seed_id": f"sd{i}", "token_count": 500} for i in range(1000)]
    bands = D.Bands(easy_max=500, medium_max=600, n_probed=10, solve_rates={}, target={})
    with pytest.raises(D.ProxyRefused, match="misses build.difficulty_target"):
        D.label_corpus(rows, bands, cfg)


def test_a_mix_inside_tolerance_is_accepted(cfg):
    # The other direction of the same gate, at the boundary: drift exactly at
    # the tolerance passes, a hair beyond it does not.
    mix = {"easy": 0.34 + cfg.difficulty.mix_tolerance, "medium": 0.50, "hard": 0.16}
    D.check_mix(mix, cfg)
    mix["easy"] += 1e-6
    with pytest.raises(D.ProxyRefused):
        D.check_mix(mix, cfg)


def test_the_measured_mix_and_the_drift_agree_with_each_other(cfg):
    mix = D.measure_mix(["easy"] * 34 + ["medium"] * 50 + ["hard"] * 16)
    assert mix == {"easy": 0.34, "medium": 0.50, "hard": 0.16}
    assert D.mix_drift(mix, cfg) == {"easy": 0.0, "medium": 0.0, "hard": 0.0}
    assert D.measure_mix([]) == {"easy": 0.0, "medium": 0.0, "hard": 0.0}


def test_the_target_is_read_from_the_build_block_not_the_difficulty_one(cfg):
    assert D.target_shares(cfg) == cfg.build.difficulty_target
    raw = json.loads(json.dumps(cfg.build.difficulty_target))
    assert set(raw) == set(D.LABELS)


def test_a_target_that_omits_a_label_reads_as_zero_rather_than_failing(cfg):
    two_way = dataclasses.replace(cfg.build, difficulty_target={"easy": 0.5, "hard": 0.5})
    shares = D.target_shares(dataclasses.replace(cfg, build=two_way))
    assert shares == {"easy": 0.5, "medium": 0.0, "hard": 0.5}


# --------------------------------------------------------------------------
# The build over a store.
# --------------------------------------------------------------------------

def _seeded(store, n=2000):
    store.upsert_source("fixture/source", "CC0")
    store.upsert_seeds(
        [
            {
                "seed_id": f"sd{i:05d}",
                "source_id": "fixture/source",
                "text": "x" * 400,
                "token_count": 100 + i,
                "meta_json": {"estimator": "chars/4"},
            }
            for i in range(n)
        ]
    )


def test_build_difficulty_labels_every_seed_and_reports_the_shape(store, cfg):
    _seeded(store)
    rows = D._seed_rows(store)
    outcomes = _declining_proxy(D.probe_sample(rows, cfg), cfg)
    manifest = D.build_difficulty(store, cfg, outcomes=outcomes)

    assert manifest["seeds"] == 2000
    assert manifest["written"] == 2000
    assert manifest["bands"]["n_probed"] == cfg.difficulty.probe_sample
    for label in D.LABELS:
        assert abs(manifest["drift"][label]) <= cfg.difficulty.mix_tolerance

    labels = [
        json.loads(row["meta_json"])["difficulty"] for row in D._seed_rows(store)
    ]
    assert set(labels) == set(D.LABELS)
    assert D.measure_mix(labels) == manifest["mix"]
    assert [event["kind"] for event in store.events("difficulty_labelled")]


def test_labelling_preserves_everything_else_on_the_seed(store, cfg):
    _seeded(store, n=600)
    rows = D._seed_rows(store)
    D.build_difficulty(store, cfg, outcomes=_declining_proxy(D.probe_sample(rows, cfg), cfg))
    seed = store.get_seed("sd00000")
    meta = json.loads(seed["meta_json"])
    assert meta["estimator"] == "chars/4"   # the row's own metadata survives
    assert meta["difficulty"] in D.LABELS
    assert seed["token_count"] == 100
    assert seed["source_id"] == "fixture/source"


def test_a_dry_run_measures_and_writes_nothing(store, cfg):
    _seeded(store, n=600)
    rows = D._seed_rows(store)
    manifest = D.build_difficulty(
        store, cfg,
        outcomes=_declining_proxy(D.probe_sample(rows, cfg), cfg),
        dry_run=True,
    )
    assert manifest["written"] == 0
    assert manifest["seeds"] == 600
    assert "difficulty" not in json.loads(store.get_seed("sd00000")["meta_json"])


def test_relabelling_is_idempotent(store, cfg):
    _seeded(store, n=600)
    rows = D._seed_rows(store)
    outcomes = _declining_proxy(D.probe_sample(rows, cfg), cfg)
    first = D.build_difficulty(store, cfg, outcomes=outcomes)
    before = [json.loads(r["meta_json"])["difficulty"] for r in D._seed_rows(store)]
    second = D.build_difficulty(store, cfg, outcomes=outcomes)
    after = [json.loads(r["meta_json"])["difficulty"] for r in D._seed_rows(store)]
    assert before == after
    assert first["mix"] == second["mix"]
    assert store.seed_count() == 600


def test_a_seed_whose_meta_json_is_not_json_is_labelled_rather_than_crashing(store, cfg):
    _seeded(store, n=600)
    store.conn.execute("UPDATE seed SET meta_json = 'not json' WHERE seed_id = 'sd00000'")
    rows = D._seed_rows(store)
    D.build_difficulty(store, cfg, outcomes=_declining_proxy(D.probe_sample(rows, cfg), cfg))
    assert json.loads(store.get_seed("sd00000")["meta_json"])["difficulty"] in D.LABELS


def test_building_without_bands_or_outcomes_is_refused(store, cfg):
    _seeded(store, n=10)
    with pytest.raises(ValueError, match="does not probe"):
        D.build_difficulty(store, cfg)


def test_a_config_without_a_difficulty_block_is_refused(store, cfg):
    naked = dataclasses.replace(cfg, difficulty=None)
    _seeded(store, n=10)
    with pytest.raises(ValueError, match="ceiling is what keeps"):
        D.build_difficulty(store, naked, outcomes=[])
    with pytest.raises(ValueError, match="difficulty:"):
        D.probe_sample([], naked)


# --------------------------------------------------------------------------
# PER-ROW PROBING MUST NOT EXIST.
# --------------------------------------------------------------------------

def _function_named(name: str) -> ast.FunctionDef:
    tree = ast.parse(DIFFICULTY_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"difficulty.py has no function {name!r}")


@pytest.mark.parametrize("name", ["label_rows", "label_corpus", "measure_mix", "check_mix"])
def test_the_labelling_path_cannot_reach_a_probe(name):
    """The grep-able-absence rule, done structurally.

    select.py's disposal_nature test greps the module for a name that must
    appear nowhere. The analogue here is narrower and stronger: the functions
    that SEE THE CORPUS must contain no reference to a probe at all, so
    per-row probing cannot be added to them without this failing. A plain grep
    over the file could not say this - the module legitimately mentions the
    probe a dozen times, in the calibration half.
    """
    node = _function_named(name)
    mentions = [
        child.id if isinstance(child, ast.Name) else child.attr
        for child in ast.walk(node)
        if isinstance(child, (ast.Name, ast.Attribute))
    ]
    assert not [m for m in mentions if "probe" in m.lower()], (name, mentions)


def test_the_whole_labelling_path_runs_with_a_probe_that_explodes_if_touched(store, cfg):
    """The behavioural half: bands in hand, 5,000 rows labelled, nothing called.

    A structural check can be satisfied by a helper that hides the call one
    frame deeper, so the same claim is made again from the outside.
    """

    def detonate(*args, **kwargs):  # pragma: no cover - calling it is the failure
        raise AssertionError("the labelling path reached the probe")

    _seeded(store, n=5000)
    bands = D.Bands(
        easy_max=100 + 1700, medium_max=100 + 4200, n_probed=1000,
        solve_rates={"easy": 0.9, "medium": 0.6, "hard": 0.2},
        target=D.target_shares(cfg),
    )
    monkeyed = {"probe_rows": D.probe_rows, "probe_sample": D.probe_sample}
    try:
        D.probe_rows = detonate
        D.probe_sample = detonate
        manifest = D.build_difficulty(store, cfg, bands=bands)
    finally:
        for name, original in monkeyed.items():
            setattr(D, name, original)
    assert manifest["written"] == 5000
    assert manifest["bands"]["n_probed"] == 1000


def test_the_module_holds_exactly_one_seam_that_can_spend_quota():
    """Every path to a model is inside main(), and there is one of it.

    make_router/router.complete anywhere else would be a second way to spend,
    and the ceiling only binds the way that goes through probe_rows.
    """
    tree = ast.parse(DIFFICULTY_SRC.read_text(encoding="utf-8"))
    spenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr in ("complete", "aclose"):
                spenders.append(node.name)
            if isinstance(child, ast.Name) and child.id == "make_router":
                spenders.append(node.name)
    assert set(spenders) <= {"main", "drive"}, spenders


def test_the_costed_reason_is_written_down_where_the_next_reader_will_look():
    # The number is the argument. Without it the ceiling reads as a tunable.
    src = DIFFICULTY_SRC.read_text(encoding="utf-8")
    assert "32M tokens" in src
    assert "65 days" in src
