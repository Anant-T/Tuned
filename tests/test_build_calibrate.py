"""calibrate.py - judge thresholds fitted to human gold labels.

Everything statistical here is tested on synthetic label sets with KNOWN
answers: a judge built to hold out at precision 0.74 must be disqualified and
one at 0.76 must survive, a rule that only `mean` can express must be the rule
that wins, and the fold split must be reproducible and disjoint. Where a test
fills in a gold label it is standing in for the operator - that is a FIXTURE
writing labels, and the module under test is checked separately for having no
path that could write one itself.
"""

import ast
import dataclasses
import json
from pathlib import Path

import pytest

from tuned.data import calibrate as C
from tuned.data.config import load_build_config
from tuned.data.store import Store

DATA_CONFIG = Path(__file__).parent.parent / "configs" / "data_law_v1.yaml"
CALIBRATE_SRC = Path(C.__file__)


@pytest.fixture(scope="module")
def cfg():
    return load_build_config(DATA_CONFIG, allow_unpinned=True)


@pytest.fixture
def store(tmp_path):
    with Store.open(tmp_path / "state" / "law_v1.sqlite3") as s:
        yield s


PASSING = (5, 5, 5)
FAILING = (1, 1, 1)


def _fit_rows(spec, fold):
    """spec = list of (axes, accepted) repeated counts -> fit_judge rows."""
    return [
        {"fold": fold, "axes": axes, "accepted": accepted}
        for axes, accepted, count in spec
        for _ in range(count)
    ]


def _judge_rows(*, fit_spec, holdout_spec, folds=5):
    rows = []
    for fold in range(folds):
        rows += _fit_rows(fit_spec, fold)
    rows += _fit_rows(holdout_spec, folds)
    return rows


# --------------------------------------------------------------------------
# The rules.
# --------------------------------------------------------------------------

def test_the_rule_names_are_the_ones_config_validates_against():
    from tuned.data.config import CALIBRATION_RULES

    assert C.RULES is CALIBRATION_RULES


def test_the_axes_are_the_ones_the_judge_scores():
    from tuned.data.judge import JudgeScores

    assert C.AXES == ("grounding", "validity", "coverage")
    scored = JudgeScores(grounding=5, validity=4, coverage=3)
    assert [getattr(scored, axis) for axis in C.AXES] == [5, 4, 3]
    assert scored.min_axis == min(getattr(scored, axis) for axis in C.AXES)


@pytest.mark.parametrize(
    "rule,axes,threshold,expected",
    [
        ("min_axis", (4, 4, 4), 4, True),
        ("min_axis", (5, 5, 3), 4, False),
        ("mean", (5, 5, 3), 4, True),          # mean 4.33
        ("mean", (5, 3, 3), 4, False),          # mean 3.67
        ("both", (5, 5, 3), 4, False),          # mean clears, min does not
        ("both", (4, 4, 5), 4, True),
    ],
)
def test_each_rule_decides_what_it_says_it_decides(rule, axes, threshold, expected):
    assert C.decides_pass(C.Candidate(rule, threshold), axes) is expected


def test_an_unknown_rule_is_refused_rather_than_silently_failing_everything():
    with pytest.raises(ValueError, match="unknown rule"):
        C.decides_pass(C.Candidate("median", 4), (5, 5, 5))


def test_counts_report_zero_rather_than_dividing_by_zero():
    empty = C.Counts()
    assert (empty.precision, empty.recall, empty.phi) == (0.0, 0.0, 0.0)
    # A rule that passed nothing has no precision to win the maximisation with.
    assert C.Counts(tp=0, fp=0, tn=10, fn=5).precision == 0.0
    assert C.Counts(tp=3, fp=1, tn=0, fn=0).recall == 1.0
    perfect = C.Counts(tp=5, fp=0, tn=5, fn=0)
    assert perfect.precision == perfect.recall == perfect.phi == 1.0
    inverted = C.Counts(tp=0, fp=5, tn=0, fn=5)
    assert inverted.phi == -1.0


# --------------------------------------------------------------------------
# The sweep.
# --------------------------------------------------------------------------

def test_the_sweep_picks_the_rule_only_one_of_them_can_express(cfg):
    # (5,5,3) is accepted, (3,3,3) is not. No min_axis threshold separates
    # them at precision 1.0 - min_axis>=3 passes both - and only `mean` at 4
    # does, because 4.33 clears it and 3.0 does not.
    pairs = [((5, 5, 3), True)] * 20 + [((3, 3, 3), False)] * 20
    best = C.best_candidate(pairs, cfg)
    assert best == C.Candidate("mean", 4)
    counts = C.evaluate(best, pairs)
    assert (counts.precision, counts.recall) == (1.0, 1.0)


def test_the_sweep_refuses_to_buy_precision_below_the_recall_floor(cfg):
    # A rule that passes 3 of 20 accepted rows is perfectly precise and
    # recalls 0.15. The floor is what stops it winning.
    pairs = [((5, 5, 5), True)] * 3 + [((3, 3, 3), True)] * 17 + [((1, 1, 1), False)] * 20
    strict = C.evaluate(C.Candidate("min_axis", 4), pairs)
    assert strict.precision == 1.0 and strict.recall == pytest.approx(0.15)
    best = C.best_candidate(pairs, cfg)
    assert best == C.Candidate("min_axis", 3)  # the only one clearing recall 0.60
    assert C.evaluate(best, pairs).recall == 1.0


def test_no_candidate_clearing_the_floor_is_a_real_answer_not_an_error(cfg):
    pairs = [((1, 1, 1), True)] * 20 + [((5, 5, 5), False)] * 20
    assert C.best_candidate(pairs, cfg) is None


def test_ties_break_toward_the_cheaper_threshold_and_the_configured_order(cfg):
    # Every candidate separates these identically, so nothing distinguishes
    # them on precision or recall. The tie-break must still be deterministic.
    pairs = [((5, 5, 5), True)] * 10 + [((1, 1, 1), False)] * 10
    best = C.best_candidate(pairs, cfg)
    assert best == C.Candidate(cfg.calibration.rules[0], min(cfg.calibration.thresholds))
    reordered = dataclasses.replace(
        cfg.calibration, rules=tuple(reversed(cfg.calibration.rules))
    )
    other = C.best_candidate(pairs, dataclasses.replace(cfg, calibration=reordered))
    assert other == C.Candidate(cfg.calibration.rules[-1], min(cfg.calibration.thresholds))


# --------------------------------------------------------------------------
# The gate, both directions.
# --------------------------------------------------------------------------

def _holdout_at(precision_numerator, denominator=100):
    """A holdout the fitted rule passes `denominator` times, of which
    `precision_numerator` were accepted by the operator."""
    return [
        (PASSING, True, precision_numerator),
        (PASSING, False, denominator - precision_numerator),
        # Rows the rule fails, so the holdout is not purely positives.
        (FAILING, False, 20),
    ]


FIT_SPEC = [(PASSING, True, 14), (PASSING, False, 2), (FAILING, True, 1), (FAILING, False, 3)]


def test_a_judge_holding_out_at_0_74_is_disqualified(cfg):
    rows = _judge_rows(fit_spec=FIT_SPEC, holdout_spec=_holdout_at(74))
    fit = C.fit_judge("judge/under", rows, cfg)
    assert fit.candidate is not None
    assert fit.holdout.precision == pytest.approx(0.74)
    assert fit.disqualified is True
    assert "0.74" in fit.reason and "0.75" in fit.reason


def test_a_judge_holding_out_at_0_76_survives(cfg):
    rows = _judge_rows(fit_spec=FIT_SPEC, holdout_spec=_holdout_at(76))
    fit = C.fit_judge("judge/over", rows, cfg)
    assert fit.holdout.precision == pytest.approx(0.76)
    assert fit.disqualified is False
    assert fit.reason is None


def test_the_gate_is_read_on_the_holdout_and_not_on_the_fit(cfg):
    # The fit folds are excellent and the holdout is not. A gate read on the
    # fit would pass this judge, which is the whole failure mode a holdout
    # exists to catch.
    rows = _judge_rows(
        fit_spec=[(PASSING, True, 30), (FAILING, False, 10)],
        holdout_spec=_holdout_at(50),
    )
    fit = C.fit_judge("judge/overfit", rows, cfg)
    assert fit.cv.precision == 1.0
    assert fit.holdout.precision == pytest.approx(0.50)
    assert fit.disqualified is True


def test_a_judge_with_no_holdout_rows_is_disqualified_not_assumed_good(cfg):
    rows = _fit_rows(FIT_SPEC, 0) + _fit_rows(FIT_SPEC, 1)
    fit = C.fit_judge("judge/partial", rows, cfg)
    assert fit.disqualified is True
    assert "holdout fold" in fit.reason


def test_a_judge_that_cannot_reach_the_recall_floor_is_disqualified(cfg):
    rows = _judge_rows(
        fit_spec=[(FAILING, True, 18), (PASSING, True, 1), (PASSING, False, 1)],
        holdout_spec=_holdout_at(90),
    )
    fit = C.fit_judge("judge/deaf", rows, cfg)
    assert fit.candidate is None
    assert fit.disqualified is True
    assert "recall" in fit.reason


def test_selection_agreement_reports_a_stable_choice_as_stable(cfg):
    rows = _judge_rows(fit_spec=FIT_SPEC, holdout_spec=_holdout_at(90))
    fit = C.fit_judge("judge/steady", rows, cfg)
    assert len(fit.fold_selections) == cfg.calibration.folds
    assert fit.selection_agreement == cfg.calibration.folds
    assert set(fit.fold_counts) == set(range(cfg.calibration.folds))


def test_selection_agreement_reports_an_unstable_choice_as_unstable(cfg):
    # Fold 0 alone says `mean`; every other fold says min_axis. Leaving one
    # fold out at a time therefore changes the winner, and the agreement
    # count is what makes that visible instead of averaged away.
    rows = _fit_rows([((5, 5, 3), True, 30), ((3, 3, 3), False, 30)], 0)
    for fold in range(1, cfg.calibration.folds):
        rows += _fit_rows([(PASSING, True, 6), (FAILING, False, 6)], fold)
    rows += _fit_rows(_holdout_at(90), cfg.calibration.folds)
    fit = C.fit_judge("judge/wobbly", rows, cfg)
    assert fit.selection_agreement < cfg.calibration.folds
    assert len(set(fit.fold_selections.values())) > 1


# --------------------------------------------------------------------------
# Folds.
# --------------------------------------------------------------------------

def _labels(n, *, accept_every=3):
    return [
        {"gen_id": i, "verdict": C.ACCEPT if i % accept_every else C.REJECT}
        for i in range(1, n + 1)
    ]


def _bare_generations(store, n):
    """n generation rows with nothing judged - gold_label.gen_id is a FOREIGN
    KEY, so a label for a generation that does not exist is refused by the
    schema, which is the schema being right."""
    store.upsert_source("fixture/source", "CC0")
    store.upsert_seeds([{"seed_id": "seedbare", "source_id": "fixture/source", "text": "x" * 600}])
    store.create_tasks(
        [
            {
                "task_id": f"bare{i:04d}",
                "seed_id": "seedbare",
                "stream": "synthesis",
                "task_type": "irac_analysis",
                "prompt_id": "gen_irac_analysis_v1",
                "prompt_sha": "abc123abc123",
                "sample_ix": i,
            }
            for i in range(n)
        ]
    )
    return [
        store.record_generation(
            {
                "task_id": f"bare{i:04d}",
                "attempt": 0,
                "provider": "fixture",
                "model": "gen/model",
                "raw_path": "raw.ndjson",
                "raw_offset": i,
                "answer": f"answer {i}",
            }
        )
        for i in range(n)
    ]


def test_folds_are_disjoint_and_cover_every_label(cfg):
    rows = C.assign_folds(
        _labels(180), folds=cfg.calibration.folds, holdout=cfg.calibration.holdout
    )
    by_fold: dict[int, set] = {}
    for row in rows:
        by_fold.setdefault(row["fold"], set()).add(row["gen_id"])
    assert set(by_fold) == set(range(cfg.calibration.folds + 1))
    assert len(by_fold[cfg.calibration.folds]) == cfg.calibration.holdout
    seen = set()
    for members in by_fold.values():
        assert not (seen & members)
        seen |= members
    assert len(seen) == 180


def test_the_split_is_reproducible_and_order_independent(cfg):
    labels = _labels(180)
    first = C.assign_folds(labels, folds=5, holdout=40)
    second = C.assign_folds(list(reversed(labels)), folds=5, holdout=40)
    assert {row["gen_id"]: row["fold"] for row in first} == {
        row["gen_id"]: row["fold"] for row in second
    }


def test_stratification_survives_the_split(cfg):
    labels = _labels(180)
    rows = C.assign_folds(labels, folds=5, holdout=40)
    overall = sum(1 for row in labels if row["verdict"] == C.ACCEPT) / len(labels)
    for fold in range(6):
        members = [row for row in rows if row["fold"] == fold]
        share = sum(1 for row in members if row["verdict"] == C.ACCEPT) / len(members)
        assert abs(share - overall) < 0.10, (fold, share, overall)


def test_a_holdout_that_takes_everything_is_refused():
    with pytest.raises(ValueError, match="holdout"):
        C.assign_folds(_labels(10), folds=5, holdout=10)


# --------------------------------------------------------------------------
# The operator's file.
# --------------------------------------------------------------------------

def _rows_for_render(n=3):
    return [
        {
            "gen_id": i,
            "seed_id": f"seed{i}",
            "stream": "synthesis",
            "task_type": "irac_analysis",
            "prompt_id": "gen_irac_analysis_v1",
            "think": f"trace {i}",
            "answer": f"answer {i}",
            "judgements": [
                {"judge_slot": "a", "grounding": 5, "validity": 4, "coverage": 4},
                {"judge_slot": "b", "grounding": 3, "validity": 3, "coverage": 5},
            ],
        }
        for i in range(1, n + 1)
    ]


def test_the_rendered_file_carries_a_blank_block_per_generation():
    rows = _rows_for_render()
    text = C.render_gold_todo(rows, seeds={f"seed{i}": {"text": f"source {i}"} for i in (1, 2, 3)})
    for row in rows:
        assert C.BEGIN.format(gid=row["gen_id"]) in text
        assert C.END.format(gid=row["gen_id"]) in text
        assert f"source {row['gen_id']}" in text
        assert f"trace {row['gen_id']}" in text
        assert f"answer {row['gen_id']}" in text
    labels, pending = C.parse_gold_todo(text)
    assert labels == []
    assert pending == ["1", "2", "3"]


def test_the_worked_example_is_filled_in_and_never_ingested():
    text = C.render_gold_todo(_rows_for_render(1))
    assert C.BEGIN.format(gid=C.EXAMPLE_ID) in text
    labels, pending = C.parse_gold_todo(text)
    assert [row["gen_id"] for row in labels] == []
    assert C.EXAMPLE_ID not in pending


def test_a_completed_file_round_trips():
    text = C.render_gold_todo(_rows_for_render(2))
    # The operator's edit: values only, structure untouched.
    text = text.replace(
        "<!-- gold:BEGIN 1 -->\nverdict: \n" "grounding: \nvalidity: \ncoverage: \nnotes: ",
        "<!-- gold:BEGIN 1 -->\nverdict: accept\n"
        "grounding: 5\nvalidity: 4\ncoverage: 4\nnotes: fine",
    )
    labels, pending = C.parse_gold_todo(text)
    assert pending == ["2"]
    assert labels == [
        {
            "gen_id": 1,
            "verdict": "accept",
            "grounding": 5,
            "validity": 4,
            "coverage": 4,
            "notes": "fine",
        }
    ]


def test_blocks_are_read_by_their_marker_not_by_position():
    text = "\n".join(
        [
            "noise",
            C._label_block(7, verdict="reject", grounding=1, validity=2, coverage=3),
            "prose an operator typed in the middle",
            C._label_block(3, verdict="accept", grounding=5, validity=5, coverage=5),
        ]
    )
    labels, pending = C.parse_gold_todo(text)
    assert [row["gen_id"] for row in labels] == [7, 3]
    assert pending == []


@pytest.mark.parametrize(
    "block,match",
    [
        (C._label_block(1, verdict="maybe"), "verdict must be one of"),
        (C._label_block(1, verdict="accept", grounding="nine"), "whole score"),
        (C._label_block(1, verdict="accept", grounding=9), "score range"),
        (C._label_block(1, verdict="accept", grounding=0), "score range"),
        (C._label_block("notanid", verdict="accept"), "not a generation id"),
    ],
)
def test_a_malformed_label_is_refused_by_name(block, match):
    with pytest.raises(C.GoldParseError, match=match):
        C.parse_gold_todo(block)


def test_a_duplicated_block_is_refused_rather_than_silently_resolved():
    text = C._label_block(1, verdict="accept") + "\n" + C._label_block(1, verdict="reject")
    with pytest.raises(C.GoldParseError, match="twice"):
        C.parse_gold_todo(text)


def test_a_half_finished_file_ingests_the_half_that_is_done(store, cfg):
    ids = _bare_generations(store, 60)
    text = "\n".join(
        [C._label_block(g, verdict=C.ACCEPT if g % 2 else C.REJECT) for g in ids[:50]]
        + [C._label_block(g) for g in ids[50:]]
    )
    result = C.ingest_gold(store, cfg, text)
    assert result["labelled"] == 50
    assert result["pending"] == 10
    assert result["written"] == 50
    assert sum(result["folds"].values()) == 50
    assert result["folds"][cfg.calibration.folds] == cfg.calibration.holdout
    assert store.gold_label_count() == 50


def test_re_ingesting_a_corrected_file_replaces_the_label(store, cfg):
    ids = _bare_generations(store, 50)
    C.ingest_gold(store, cfg, "\n".join(C._label_block(g, verdict=C.ACCEPT) for g in ids))
    assert {row["verdict"] for row in store.gold_labels()} == {C.ACCEPT}
    C.ingest_gold(store, cfg, "\n".join(C._label_block(g, verdict=C.REJECT) for g in ids))
    assert store.gold_label_count() == 50
    assert {row["verdict"] for row in store.gold_labels()} == {C.REJECT}


# --------------------------------------------------------------------------
# End to end over a store.
# --------------------------------------------------------------------------

JUDGE_GOOD = "judge/good"
JUDGE_BAD = "judge/bad"


def _seed_store(store, *, n=180, bad_mode="passes_everything"):
    """A store of n judged generations and the operator's labels.

    THE LABELS HERE ARE THE FIXTURE STANDING IN FOR THE OPERATOR. The module
    under test never writes one; test_gold_labels_can_only_come_from_a_person
    is what pins that.

    The operator accepts two rows in three. JUDGE_GOOD passes exactly those,
    so its precision is 1.0. JUDGE_BAD is built two ways:

      passes_everything  it passes every row, so its precision IS the accept
                         share, 0.667 - under the 0.75 gate however many rows
                         it sees, which is the cleanest possible failure;
      nearly_good        it disagrees with the operator on every seventh row
                         in both directions, so it clears the gate while
                         still differing from JUDGE_GOOD - which is what
                         makes kappa a number rather than 1.0.
    """
    store.upsert_source("fixture/source", "CC0")
    store.upsert_seeds(
        [
            {
                "seed_id": f"seed{i:04d}",
                "source_id": "fixture/source",
                "text": f"materials for row {i}, long enough to excerpt " * 3,
                "meta_json": {},
            }
            for i in range(n)
        ]
    )
    store.create_tasks(
        [
            {
                "task_id": f"task{i:04d}",
                "seed_id": f"seed{i:04d}",
                "stream": "synthesis" if i % 2 else "transition",
                "task_type": "irac_analysis" if i % 2 else "transition",
                "prompt_id": "gen_irac_analysis_v1",
                "prompt_sha": "abc123abc123",
                "sample_ix": 0,
            }
            for i in range(n)
        ]
    )
    labels = []
    for i in range(n):
        gen_id = store.record_generation(
            {
                "task_id": f"task{i:04d}",
                "attempt": 0,
                "provider": "fixture",
                "model": "gen/model",
                "raw_path": "raw.ndjson",
                "raw_offset": i,
                "think": f"trace {i}",
                "answer": f"answer {i}",
            }
        )
        # The operator accepts two rows in three.
        accepted = i % 3 != 0
        bad_passes = True if bad_mode == "passes_everything" else (accepted != (i % 7 == 0))
        for model, passes in ((JUDGE_GOOD, accepted), (JUDGE_BAD, bad_passes)):
            axes = (5, 5, 5) if passes else (1, 1, 1)
            store.record_judgement(
                gen_id,
                "a" if model == JUDGE_GOOD else "b",
                {
                    "provider": "fixture",
                    "model": model,
                    "grounding": axes[0],
                    "validity": axes[1],
                    "coverage": axes[2],
                    "rationale": "",
                },
            )
        labels.append({"gen_id": gen_id, "verdict": C.ACCEPT if accepted else C.REJECT})
    return labels


def _operator_fills_in(todo: str, labels) -> str:
    """The fixture standing in for the operator: it edits the VALUES of each
    blank block in the rendered file and touches nothing else, which is what
    the file's own instructions ask a person to do."""
    verdicts = {row["gen_id"]: row["verdict"] for row in labels}
    out = todo
    for gen_id, verdict in verdicts.items():
        blank = C._label_block(gen_id)
        if blank not in out:
            continue
        out = out.replace(
            blank,
            C._label_block(gen_id, verdict=verdict, grounding=4, validity=4, coverage=4),
        )
    return out


def test_the_whole_flow_runs_on_synthetic_gold(store, cfg, tmp_path):
    labels = _seed_store(store)
    exported = C.export_pilot(store, cfg)
    assert len(exported) == cfg.calibration.pilot_export

    todo = C.render_gold_todo(
        exported,
        seeds={row["seed_id"]: store.get_seed(row["seed_id"]) for row in exported},
    )
    result = C.ingest_gold(store, cfg, _operator_fills_in(todo, labels))
    assert result["written"] == cfg.calibration.pilot_export
    assert result["folds"][cfg.calibration.folds] == cfg.calibration.holdout

    calibration, report = C.run_calibration(store, cfg)
    assert calibration.n_gold == cfg.calibration.pilot_export
    fits = {fit.model: fit for fit in calibration.fits}
    assert set(fits) == {JUDGE_GOOD, JUDGE_BAD}
    assert fits[JUDGE_GOOD].disqualified is False
    assert fits[JUDGE_BAD].disqualified is True
    assert fits[JUDGE_BAD].replacement in [ref.model for ref in cfg.routing_refs("judge")]

    # Only the judge that passed is written, and judge.py now reads the fleet
    # as calibrated rather than provisional.
    from tuned.data.judge import thresholds_active

    active = store.judge_thresholds()
    assert [row["model"] for row in active] == [JUDGE_GOOD]
    assert active[0]["rule"] in cfg.calibration.rules
    assert thresholds_active(store) == 1

    assert "# calibration_report.md" in report
    assert JUDGE_GOOD in report and JUDGE_BAD in report
    assert "DISQUALIFIED" in report
    assert "kappa" in report.lower()
    assert "named replacement" in report

    event = json.loads(store.events("judges_calibrated")[0]["detail_json"])
    assert event["active"] == [JUDGE_GOOD]
    assert event["disqualified"] == [JUDGE_BAD]


def test_a_second_calibration_supersedes_the_first_without_losing_it(store, cfg):
    labels = _seed_store(store, n=60)
    store.upsert_gold_labels(
        C.assign_folds(labels, folds=cfg.calibration.folds, holdout=12)
    )
    C.run_calibration(store, cfg)
    active_first = store.judge_thresholds(active_only=True)
    assert active_first
    C.run_calibration(store, cfg)
    active_second = store.judge_thresholds(active_only=True)
    # Exactly one live calibration per model, and the superseded rows are
    # still there to read the new one against. A calib_id built without the
    # timestamp would have REPLACED them and left only the current fit.
    assert len(active_second) == len(active_first)
    assert {row["calib_id"] for row in active_second}.isdisjoint(
        {row["calib_id"] for row in active_first}
    )
    assert len(store.judge_thresholds(active_only=False)) == 2 * len(active_first)
    assert all(row["active"] == 0 for row in store.judge_thresholds(active_only=False)
               if row["calib_id"] in {r["calib_id"] for r in active_first})


def test_the_export_is_the_same_180_rows_every_time(store, cfg):
    _seed_store(store)
    first = [row["gen_id"] for row in C.export_pilot(store, cfg)]
    second = [row["gen_id"] for row in C.export_pilot(store, cfg)]
    assert first == second
    assert len(set(first)) == len(first)


def test_the_export_spreads_over_streams_and_verdict_patterns(store, cfg):
    _seed_store(store)
    exported = C.export_pilot(store, cfg)
    streams = {row["stream"] for row in exported}
    assert streams == {"synthesis", "transition"}
    patterns = {C._verdict_pattern(row["judgements"]) for row in exported}
    # both-pass and both-fail at least: a draw that only saw agreement would
    # fit a threshold that has never met the disagreement it must resolve.
    assert len(patterns) >= 2


def test_a_generation_nobody_judged_is_not_in_the_population(store, cfg):
    _seed_store(store, n=20)
    store.create_tasks(
        [
            {
                "task_id": "lonely",
                "seed_id": "seed0000",
                "stream": "synthesis",
                "task_type": "irac_analysis",
                "prompt_id": "gen_irac_analysis_v1",
                "prompt_sha": "abc123abc123",
                "sample_ix": 9,
            }
        ]
    )
    lonely = store.record_generation(
        {
            "task_id": "lonely",
            "attempt": 0,
            "provider": "fixture",
            "model": "gen/model",
            "raw_path": "raw.ndjson",
            "raw_offset": 999,
            "answer": "unjudged",
        }
    )
    assert lonely not in {row["gen_id"] for row in store.judged_generations()}


def test_calibrating_with_no_gold_labels_reports_nothing_rather_than_claiming_anything(
    store, cfg
):
    _seed_store(store, n=20)
    calibration = C.calibrate(store, cfg)
    assert calibration.n_gold == 0
    assert calibration.fits == []
    assert C.threshold_rows(calibration) == []


def test_a_config_without_a_calibration_block_is_refused(store, cfg):
    naked = dataclasses.replace(cfg, calibration=None)
    with pytest.raises(ValueError, match="min_recall and min_precision ARE the gate"):
        C.export_pilot(store, naked)
    with pytest.raises(ValueError, match="calibration"):
        C.calibrate(store, naked)


def test_kappa_is_measured_and_reads_both_extremes():
    assert C.cohens_kappa([(True, True)] * 10 + [(False, False)] * 10) == 1.0
    assert C.cohens_kappa([(True, False)] * 10 + [(False, True)] * 10) == -1.0
    # Two raters who never vary agree perfectly by construction, and there is
    # no agreement BEYOND chance to report.
    assert C.cohens_kappa([(True, True)] * 20) == 0.0
    assert C.cohens_kappa([]) == 0.0
    mixed = [(True, True)] * 5 + [(True, False)] * 5 + [(False, True)] * 5 + [(False, False)] * 5
    assert C.cohens_kappa(mixed) == pytest.approx(0.0)


def test_kappa_is_reported_over_the_generations_two_fitted_judges_both_scored(store, cfg):
    labels = _seed_store(store, n=90, bad_mode="nearly_good")
    store.upsert_gold_labels(C.assign_folds(labels, folds=cfg.calibration.folds, holdout=20))
    calibration = C.calibrate(store, cfg)
    assert len(calibration.active) == 2
    key = f"{JUDGE_BAD} vs {JUDGE_GOOD}"
    assert key in calibration.kappa
    assert calibration.kappa[key]["n"] == 90
    # Strictly between: the two judges differ on every seventh row, so a
    # kappa of exactly 1.0 would mean the pairing collapsed a judge against
    # itself and a 0.0 would mean it was never computed.
    assert 0.0 < calibration.kappa[key]["kappa"] < 1.0


def test_a_judge_pair_is_only_reported_once_and_never_against_itself(store, cfg):
    labels = _seed_store(store, n=90, bad_mode="nearly_good")
    store.upsert_gold_labels(C.assign_folds(labels, folds=cfg.calibration.folds, holdout=20))
    calibration = C.calibrate(store, cfg)
    assert len(calibration.kappa) == 1
    (pair,) = calibration.kappa
    assert pair.split(" vs ")[0] != pair.split(" vs ")[1]


# --------------------------------------------------------------------------
# Gold labels are human-only.
# --------------------------------------------------------------------------

def test_gold_labels_can_only_come_from_a_person():
    """The module's own imports are parsed, not grepped for.

    A gold label written by a model would calibrate the judges against a
    judge. The rule is absolute, so it is enforced structurally: this module
    may import nothing that can reach a provider, and the allowlist is small
    enough to read.
    """
    tree = ast.parse(CALIBRATE_SRC.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    allowed = {
        "hashlib", "math", "re", "argparse", "pathlib",
        "collections.abc", "dataclasses",
        "tuned.data.config", "tuned.data.paths", "tuned.data.store",
    }
    assert imported <= allowed, sorted(imported - allowed)


def test_no_seam_in_this_module_could_produce_a_label():
    src = CALIBRATE_SRC.read_text(encoding="utf-8")
    for forbidden in ("providers", "Router", "httpx", "generate(", "async def", "await "):
        assert forbidden not in src, forbidden
    # And the shipped instructions say so to the person reading them.
    assert "written by a person" in C._INSTRUCTIONS


def test_the_operator_instructions_say_which_way_to_round_a_doubtful_row():
    # The threshold decides what ships unread, so the benefit of the doubt has
    # to run toward the reader of the dataset. If this sentence goes, the
    # labels it produced afterwards mean something else.
    assert "A ROW YOU WOULD ARGUE ABOUT IS A REJECT" in " ".join(C._INSTRUCTIONS.split())
