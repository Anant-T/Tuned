"""transition.py - the s.358 grid, its answer keys, and what cannot emit.

Fixtures here use the REPO'S OWN statute resources (ipc_bns_map.jsonl,
transition_provisions.jsonl) and never invent statutory text: a fixture that
made up a section number or a clause would be testing the module against a
law this build does not hold. Where a test needs a mapping the resource does
not contain, it copies real rows and changes a STRUCTURAL field - verified_by
to null, a side to null - which is exactly what the operator sheet does.
"""

import copy
import json
import random
from datetime import date, timedelta
from pathlib import Path

import pytest

from tuned.data import gates, generate, prompt_registry, tasks
from tuned.data import transition as T
from tuned.data.config import load_build_config
from tuned.data.statutes import (
    APPOINTED_DAY,
    CODE_KIND,
    OLD_CODES,
    Mapping,
    SectionRef,
    extract_sections,
    governing_family,
)
from tuned.data.store import Store

DATA_CONFIG = Path(__file__).parent.parent / "configs" / "data_law_v1.yaml"
TRANSITION_SRC = Path(T.__file__)


@pytest.fixture(scope="module")
def cfg():
    return load_build_config(DATA_CONFIG, allow_unpinned=True)


@pytest.fixture(scope="module")
def mapping():
    return Mapping.load()


@pytest.fixture(scope="module")
def provisions():
    return T.load_provisions()


@pytest.fixture(scope="module")
def grid(mapping, provisions):
    return T.build_grid(mapping, provisions=provisions)


@pytest.fixture(scope="module")
def selection(grid, cfg):
    cells, _ = grid
    return T.select_cells(cells, sample=cfg.transition.sample, reserve=cfg.transition.eval_reserve)


@pytest.fixture
def store(tmp_path):
    with Store.open(tmp_path / "state" / "law_v1.sqlite3") as s:
        yield s


def _seed_for_store(row: dict) -> dict:
    """A seed row as the store hands it back: *_json columns are TEXT."""
    out = dict(row)
    out["meta_json"] = json.dumps(row["meta_json"])
    out["answer_key_json"] = json.dumps(row["answer_key_json"])
    return out


def _stratified(cells, key, per_stratum=1):
    """One cell per value of `key`, in a deterministic order."""
    picked: dict = {}
    for cell in sorted(cells, key=lambda c: c.cell_id):
        picked.setdefault(key(cell), []).append(cell)
    return [c for bucket in picked.values() for c in bucket[:per_stratum]]


# --------------------------------------------------------------------------
# The shape of the grid, measured.
# --------------------------------------------------------------------------

def test_the_grid_is_the_verified_map_crossed_with_the_postures(grid, mapping):
    cells, refused = grid
    families = {cell.family.key for cell in cells}
    assert len(T.POSTURE_CELLS) == 80  # 5 date x 4 procedural x 4 question
    assert len(T.POSTURE_CELLS) == (
        len(T.DATE_POSTURES) * len(T.PROCEDURAL_POSTURES) * len(T.QUESTION_FORMS)
    )
    assert len(mapping.verified_rows()) == 154
    # 153, not 154: IPC 377 is verified and emits nothing at all, because no
    # posture of it can be keyed with certainty (see the judicial tests below).
    assert len(families) == 153
    # 154 x 80 minus the cells that cannot be built - the ones no gate stack
    # could accept AND the ones whose law this build cannot state - which is
    # the whole of the difference.
    ungateable = [entry for entry in refused if "cell" in entry]
    assert len(cells) == len(mapping.verified_rows()) * len(T.POSTURE_CELLS) - len(ungateable)
    assert len(cells) == 12144


def test_every_cell_id_is_unique_and_content_keyed(grid):
    cells, _ = grid
    ids = [cell.cell_id for cell in cells]
    assert len(set(ids)) == len(ids)
    for cell in cells[:50]:
        key = T.cell_key(
            cell.family, cell.date_posture, cell.procedural_posture, cell.question_form
        )
        assert cell.cell_id == T.cell_id_for(key)


def test_the_grid_is_identical_across_runs_and_under_a_shuffled_map(mapping, provisions):
    first, _ = T.build_grid(mapping, provisions=provisions)
    second, _ = T.build_grid(mapping, provisions=provisions)
    assert [c.cell_id for c in first] == [c.cell_id for c in second]

    shuffled = copy.deepcopy(mapping.rows)
    random.Random(20260817).shuffle(shuffled)
    third, _ = T.build_grid(Mapping(shuffled), provisions=provisions)
    # Same cells, same ORDER: family order is content-keyed, so the order the
    # rows happened to arrive in cannot move a cell.
    assert [c.cell_id for c in third] == [c.cell_id for c in first]


def test_the_sample_and_the_reserve_are_identical_across_runs(grid, cfg):
    cells, _ = grid
    a = T.select_cells(cells, sample=cfg.transition.sample, reserve=cfg.transition.eval_reserve)
    b = T.select_cells(
        list(reversed(cells)), sample=cfg.transition.sample, reserve=cfg.transition.eval_reserve
    )
    # Reversing the input reverses the family ORDER selection_order walks, so
    # the draws are not expected to be equal - what must hold is that the
    # same input gives the same draw, every time.
    c = T.select_cells(cells, sample=cfg.transition.sample, reserve=cfg.transition.eval_reserve)
    assert [x.cell_id for x in a.sample] == [x.cell_id for x in c.sample]
    assert [x.cell_id for x in a.reserve] == [x.cell_id for x in c.reserve]
    assert len(b.sample) == len(a.sample)


def test_the_draw_covers_every_family_and_every_posture_pair(selection, grid):
    cells, _ = grid
    per_family: dict[str, int] = {}
    for cell in selection.sample:
        per_family[cell.family.key] = per_family.get(cell.family.key, 0) + 1
    assert len(per_family) == len({c.family.key for c in cells}) == 153
    # A prefix of the coverage order gives every family within one cell of
    # every other - no family is left out of a 1,100-cell draw.
    assert max(per_family.values()) - min(per_family.values()) <= 1
    assert (min(per_family.values()), max(per_family.values())) == (7, 8)

    triples = {
        (c.date_posture.name, c.procedural_posture.name, c.question_form.name)
        for c in selection.sample
    }
    assert len(triples) == len(T.POSTURE_CELLS)
    pairs = {(c.date_posture.name, c.procedural_posture.name) for c in selection.sample}
    assert len(pairs) == len(T.DATE_POSTURES) * len(T.PROCEDURAL_POSTURES) == 20


def test_the_draw_is_stratified_and_not_merely_covering(selection):
    """Every posture pair PRESENT is the weak claim; the brief asks for a
    stratified sample, which is the even one.

    Found by mutation: collapsing coverage_stride to 1 leaves every coverage
    assertion above green - 1 is coprime with 80, so the sweep is still a
    permutation and all 20 pairs and all 80 triples still appear - while the
    per-pair counts go from 50-62 to 35-63. Presence was tested and EVENNESS
    was not, so the mutant lived in the gap between them. Measured ratio is
    1.24 with the real stride and 1.80 with the collapsed one.
    """
    counts: dict[tuple[str, str], int] = {}
    for cell in selection.sample:
        key = (cell.date_posture.name, cell.procedural_posture.name)
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 20
    assert max(counts.values()) / min(counts.values()) <= 1.5, counts


def test_the_reserve_is_disjoint_from_the_sample_by_construction(selection):
    reserve = {c.cell_id for c in selection.reserve}
    sample = {c.cell_id for c in selection.sample}
    assert len(reserve) == 150
    assert len(sample) == 1100
    assert not (reserve & sample)
    # And it is a PREFIX of the same order, which is what makes the line above
    # a property rather than a coincidence of these two numbers.
    assert [c.cell_id for c in selection.order[:150]] == [c.cell_id for c in selection.reserve]
    assert [c.cell_id for c in selection.order[150:1250]] == [c.cell_id for c in selection.sample]


def test_the_reserve_spreads_over_the_postures_too(selection):
    triples = {
        (c.date_posture.name, c.procedural_posture.name, c.question_form.name)
        for c in selection.reserve
    }
    assert len(triples) == len(T.POSTURE_CELLS)
    assert len({c.family.key for c in selection.reserve}) == 150


def test_a_draw_bigger_than_the_grid_is_refused(grid):
    cells, _ = grid
    with pytest.raises(ValueError, match="cannot supply"):
        T.select_cells(cells, sample=len(cells), reserve=1)


@pytest.mark.parametrize("n", list(range(3, 130)))
def test_the_coverage_stride_is_always_coprime_with_the_posture_count(n):
    from math import gcd

    stride = T.coverage_stride(n)
    assert 1 <= stride < n
    assert gcd(stride, n) == 1


def test_the_stride_makes_each_family_walk_every_posture_exactly_once(selection, grid):
    cells, _ = grid
    by_family: dict[str, list[str]] = {}
    for cell in selection.order:
        by_family.setdefault(cell.family.key, []).append(cell.cell_id)
    total = {c.family.key: 0 for c in cells}
    for cell in cells:
        total[cell.family.key] += 1
    for key, ids in by_family.items():
        assert len(set(ids)) == len(ids) == total[key]


# --------------------------------------------------------------------------
# require_verified: the 17 rows that cannot emit, both directions.
# --------------------------------------------------------------------------

def test_the_unverified_mapping_rows_emit_nothing(grid, mapping):
    cells, refused = grid
    unverified = {
        str(SectionRef(row["old_code"] or row["new_code"], row["old_section"] or row["new_section"]))
        for row in mapping.unverified_rows()
    }
    assert len(unverified) == 17
    assert not (unverified & {cell.family.key for cell in cells})

    # And the manifest says WHY, in statutes.py's own words.
    family_refusals = {entry["family"]: entry["reason"] for entry in refused if "cell" not in entry}
    assert set(family_refusals) == unverified
    for reason in family_refusals.values():
        assert "unverified" in reason and "verified_by is null" in reason


def test_flipping_one_verified_row_to_null_removes_exactly_its_cells(mapping, provisions):
    before, _ = T.build_grid(mapping, provisions=provisions)
    rows = copy.deepcopy(mapping.rows)
    victim = next(row for row in rows if row.get("verified_by") and row["kind"] == "one_to_one")
    key = str(SectionRef(victim["old_code"], victim["old_section"]))
    assert key in {cell.family.key for cell in before}

    victim["verified_by"] = None
    after, refused = T.build_grid(Mapping(rows), provisions=provisions)
    assert key not in {cell.family.key for cell in after}
    assert len(after) == len(before) - len(T.POSTURE_CELLS)
    reasons = {entry["family"]: entry["reason"] for entry in refused if "cell" not in entry}
    assert key in reasons
    assert "unverified" in reasons[key]


def test_signing_one_unverified_row_off_grows_the_grid_with_no_code_change(mapping, provisions):
    before, _ = T.build_grid(mapping, provisions=provisions)
    rows = copy.deepcopy(mapping.rows)
    victim = next(row for row in rows if not row.get("verified_by"))
    key = str(SectionRef(victim["old_code"], victim["old_section"]))

    victim["verified_by"] = "operator sign-off (fixture)"
    after, _ = T.build_grid(Mapping(rows), provisions=provisions)
    assert key in {cell.family.key for cell in after}
    assert len(after) == len(before) + len(T.POSTURE_CELLS)


def test_a_family_whose_identifying_side_is_ambiguous_is_refused(mapping, provisions):
    # Two verified rows landing on ONE new section is legal (IPC 499 and 500
    # both go to BNS 356) and Mapping.row() returns None rather than guessing.
    # A new_offence row - identified by its NEW side, because it has no old
    # one - that shares its new section with such a row can no longer name
    # itself, so it does not emit. The refusal is statutes.py's own, and it is
    # the only one this module needs: nothing here re-checks Mapping's
    # invariants on top of it.
    rows = copy.deepcopy(mapping.rows)
    ghost = next(row for row in rows if row["kind"] == "new_offence")
    collide = copy.deepcopy(ghost)
    collide["old_code"], collide["old_section"], collide["kind"] = "IPC", "999", "one_to_one"
    rows.append(collide)

    key = str(SectionRef(ghost["new_code"], ghost["new_section"]))
    before, _ = T.build_grid(mapping, provisions=provisions)
    assert key in {cell.family.key for cell in before}

    cells, refused = T.build_grid(Mapping(rows), provisions=provisions)
    assert key not in {cell.family.key for cell in cells}
    reasons = {entry["family"]: entry["reason"] for entry in refused if "cell" not in entry}
    assert reasons[key] == f"no mapping row for {key}"


# --------------------------------------------------------------------------
# Dates.
# --------------------------------------------------------------------------

def test_every_cell_dates_the_proceeding_no_earlier_than_the_conduct(grid):
    cells, _ = grid
    for cell in cells:
        assert cell.proceeding_started >= cell.offence_date, cell.coordinates


def test_no_cell_narrates_a_record_that_could_not_exist_yet(grid):
    """Ordering is not enough: the record has to be POSSIBLE.

    Measured before the floors landed: 2,400 cells dated the proceeding on the
    day of the conduct, 1,800 of them at a stage that cannot be reached that
    day, and one in seven of the drawn cells read "the appeal against the
    conviction was filed on 1 July 2024" for conduct on 1 July 2024. 1,571 of
    3,048 appeal cells filed the appeal inside six months.
    """
    cells, _ = grid
    for cell in cells:
        lag = (cell.proceeding_started - cell.offence_date).days
        assert lag >= cell.procedural_posture.min_lag_days, (cell.coordinates, lag)

    same_day = [c for c in cells if c.proceeding_started == c.offence_date]
    # Only the FIR stage, where an information recorded on the day of the
    # conduct is the ordinary case rather than an impossible record.
    assert {c.procedural_posture.name for c in same_day} == {"fir"}
    assert len(same_day) == 600
    appeals = [c for c in cells if c.procedural_posture.name == "appeal"]
    assert appeals
    assert min((c.proceeding_started - c.offence_date).days for c in appeals) == 180


def test_the_floors_never_push_a_posture_over_its_own_boundary(grid):
    """The lag moves the record, and the posture still means what it says.

    A floor wide enough to carry an appeal past the appointed day would turn
    every just_before cell into a straddling one and flip its procedural limb,
    which is the failure mode of fixing this the lazy way.
    """
    cells, _ = grid
    for cell in cells:
        if cell.date_posture.name in ("well_before", "just_before"):
            assert cell.proceeding_started < APPOINTED_DAY, cell.coordinates
        else:
            assert cell.proceeding_started >= APPOINTED_DAY, cell.coordinates


def test_the_on_appointed_day_question_is_still_about_the_offence_date(grid, provisions):
    """The conduct stays exactly on the boundary; only the record moves out."""
    cells, _ = grid
    on_day = [c for c in cells if c.date_posture.name == "on_appointed_day"]
    assert len(on_day) == 2400
    for cell in on_day:
        assert cell.offence_date == APPOINTED_DAY
        assert cell.proceeding_started == APPOINTED_DAY + timedelta(
            days=cell.procedural_posture.min_lag_days
        )
    # ...and the answer is unmoved: new on all three limbs, which is the edge
    # the posture exists for.
    for cell in on_day[:40]:
        assert T.answer_key_for(cell, provisions)["families_by_kind"] == {
            "substantive": "new", "procedural": "new", "evidence": "new"
        }


def test_the_appeal_scenario_reads_as_a_record_a_court_file_could_hold(grid, provisions):
    cell = next(
        c for c in grid[0]
        if c.date_posture.name == "on_appointed_day" and c.procedural_posture.name == "appeal"
    )
    scenario = T.render_cell(cell, provisions)["scenario"]
    assert T.pretty_date(APPOINTED_DAY) in scenario  # the conduct, on the boundary
    assert T.pretty_date(cell.proceeding_started) in scenario
    # The two dates in the prose are different ones - the conviction and the
    # appeal against it are no longer narrated as the day of the conduct.
    assert cell.proceeding_started != cell.offence_date


def test_the_date_postures_mean_what_they_are_named(grid):
    """Each posture, against the boundary IT is anchored to.

    "just_before" means the months before the boundary that decides the
    family's substantive answer, and for a family a court struck down that
    boundary is the JUDGMENT, not the appointed day. The proceeding is
    anchored to the appointed day whatever the family, because which
    procedural code governs turns on that day alone.
    """
    cells, _ = grid
    seen = {}
    for cell in cells:
        seen.setdefault(cell.date_posture.name, []).append(cell)

    def anchor(cell):
        event = T.JUDICIAL_INVALIDATIONS.get(cell.family.key)
        if event is not None and cell.date_posture.anchors_to_invalidation:
            assert cell.date_anchor == T.ANCHOR_INVALIDATION
            return event.decided_on
        assert cell.date_anchor == T.ANCHOR_APPOINTED_DAY
        return APPOINTED_DAY

    for cell in seen["well_before"]:
        assert cell.offence_date < APPOINTED_DAY and cell.proceeding_started < APPOINTED_DAY
    for cell in seen["just_before"]:
        assert cell.offence_date < anchor(cell)
        assert cell.proceeding_started < APPOINTED_DAY
    for cell in seen["on_appointed_day"]:
        assert cell.offence_date == APPOINTED_DAY
        assert cell.proceeding_started >= APPOINTED_DAY
    for cell in seen["just_after"]:
        assert cell.offence_date > anchor(cell)
        assert cell.proceeding_started > APPOINTED_DAY
    for cell in seen["straddling"]:
        assert cell.offence_date < APPOINTED_DAY <= cell.proceeding_started


def test_the_appointed_day_posture_is_new_on_both_axes(grid, provisions):
    # BNSS s.531(2)(a) saves what was pending IMMEDIATELY BEFORE the
    # commencement, and the codes came into force ON the appointed day - so a
    # matter dated that day is new on both axes, which is the edge the plan
    # named and the one a naive "<=" would get backwards.
    cells, _ = grid
    on_day = [c for c in cells if c.date_posture.name == "on_appointed_day"]
    assert on_day
    for cell in on_day[:40]:
        key = T.answer_key_for(cell, provisions)
        assert key["families_by_kind"] == {
            "substantive": "new", "procedural": "new", "evidence": "new"
        }


def test_the_dates_in_the_posture_prose_are_the_dates_in_the_columns(selection, provisions):
    for cell in selection.sample[:120]:
        row = T.seed_row(cell, provisions, held_out=False)
        scenario = row["meta_json"]["scenario"]
        assert T.pretty_date(cell.offence_date) in scenario
        assert T.pretty_date(cell.proceeding_started) in scenario
        assert row["offence_date"] == cell.offence_date.isoformat()
        assert row["meta_json"]["proceeding_started"] == cell.proceeding_started.isoformat()


def test_cell_dates_are_content_keyed_and_stable():
    dp = T.DATE_POSTURES[0]
    first = T.cell_dates("IPC 302|well_before|trial|charge_only", dp, APPOINTED_DAY)
    second = T.cell_dates("IPC 302|well_before|trial|charge_only", dp, APPOINTED_DAY)
    other = T.cell_dates("IPC 302|well_before|trial|savings_effect", dp, APPOINTED_DAY)
    assert first == second
    assert first != other
    lo, hi = dp.offence_span
    assert APPOINTED_DAY + timedelta(days=lo) <= first[0] <= APPOINTED_DAY + timedelta(days=hi)


# --------------------------------------------------------------------------
# Answer keys, recomputed independently.
# --------------------------------------------------------------------------

def _independent_key(cell: T.Cell, provisions) -> dict:
    """The key derived from statutes.py alone - no transition.py logic.

    Deliberately written the long way round, out of governing_family() and the
    mapping row's own two sides, so that a change in transition.py's
    derivation has to be justified against the decision table rather than
    against itself.

    The judicial timeline is re-derived here too, from the two dated constants
    and the offence date, WITHOUT calling transition.judicial_status: the
    branch it drives is the one that was wrong, so a test that asked the
    module the same question twice would agree with itself.
    """
    substantive = governing_family(
        "substantive",
        offence_date=cell.offence_date,
        proceeding_started=cell.proceeding_started,
    )
    procedural = governing_family(
        "procedural",
        offence_date=cell.offence_date,
        proceeding_started=cell.proceeding_started,
    )
    event = T.JUDICIAL_INVALIDATIONS.get(cell.family.key)
    if event is None:
        struck = False
    elif event.scope == T.SCOPE_CONDUCT_SCOPED:
        struck = None  # never decidable: the papers do not narrate consent
    else:
        struck = cell.offence_date > event.decided_on
        if not struck:
            struck = None  # conduct at or before the judgment: reach unknown

    old_ref, new_ref = cell.family.old_ref, cell.family.new_ref
    if struck is False:
        charge = old_ref if substantive == "old" else new_ref
        counterpart = new_ref if substantive == "old" else old_ref
    else:
        charge = counterpart = None
    engage = charge if charge is not None else counterpart
    if engage is None and struck is not False:
        engage = old_ref if old_ref is not None else new_ref

    expected = [engage] if engage is not None else []
    if "procedural" in cell.question_form.limbs:
        expected.append(SectionRef("BNSS", "531"))
    if "evidence" in cell.question_form.limbs:
        expected.append(SectionRef("BSA", "170"))
    if substantive == "old" or cell.question_form.name == "savings_effect":
        expected.append(SectionRef("BNS", "358"))

    forbidden = []
    if cell.question_form.forbids_counterpart and charge is not None and counterpart is not None:
        forbidden = [counterpart]

    if struck is None:
        consequence = "not_decidable_on_this_build"
    elif struck:
        consequence = "no_offence_lies"
    elif charge is not None:
        consequence = "old_liability_preserved" if substantive == "old" else "new_code_governs_directly"
    elif substantive == "old":
        consequence = "new_offence_cannot_reach_earlier_conduct"
    else:
        consequence = "repealed_without_successor"

    return {
        "governing_family": substantive,
        "expected_sections": [{"code": r.code, "number": r.number} for r in expected],
        "forbidden_sections": [{"code": r.code, "number": r.number} for r in forbidden],
        "requires_savings_mention": substantive == "old"
        or cell.question_form.name == "savings_effect",
        "requires_no_liability_statement": struck is True,
        "must_name_both_families": substantive == "old" and old_ref is not None,
        "savings_consequence": consequence,
        "procedural_family": procedural,
    }


def test_every_key_is_recomputable_from_the_statute_table_alone(grid, provisions):
    """A stratified sample covering every kind x date x procedural x question
    combination that exists, recomputed from statutes.py and compared."""
    cells, _ = grid
    strata = _stratified(
        cells,
        lambda c: (
            c.family.kind,
            c.date_posture.name,
            c.procedural_posture.name,
            c.question_form.name,
        ),
    )
    # 4 kinds x 80 posture triples, minus the (deleted x date posture)
    # combinations no deleted family can be built on: the two post-appointed-
    # day postures, and "just_before", which for IPC 497 is anchored to the
    # judgment and therefore refused while 124A/309 keep it.
    assert len(strata) == 4 * 80 - 16
    for cell in strata:
        key = T.answer_key_for(cell, provisions)
        want = _independent_key(cell, provisions)
        for field, value in want.items():
            if field == "procedural_family":
                assert key["families_by_kind"]["procedural"] == value, cell.coordinates
                continue
            assert key[field] == value, (field, cell.coordinates)


def test_the_edges_the_plan_named_are_all_in_the_grid(grid):
    cells, _ = grid
    kinds = {cell.family.kind for cell in cells}
    assert kinds == {"one_to_one", "changed", "new_offence", "deleted"}
    families = {cell.family.key for cell in cells}
    # No-counterpart new offences, named in the plan.
    for named in ("BNS 111", "BNS 113", "BNS 103(2)", "BNS 304", "BNS 69"):
        assert named in families
    # The two `deleted` families with no judicial event keep every posture
    # whose conduct predates the repeal.
    for named in ("IPC 124A", "IPC 309"):
        assert named in families
        assert {
            c.date_posture.name for c in cells if c.family.key == named
        } == {"well_before", "just_before", "straddling"}
    # IPC 497 keeps the postures whose conduct POST-DATES Joseph Shine, which
    # is a different set: "just_before" is anchored to the judgment for this
    # family and lands before it, so it goes; "just_after" is anchored there
    # too and lands after it, so it arrives.
    assert {c.date_posture.name for c in cells if c.family.key == "IPC 497"} == {
        "well_before", "just_after", "straddling"
    }
    # IPC 377 is verified, is in no refusal of the FAMILY kind, and still
    # emits nothing at all.
    assert "IPC 377" not in families


def test_a_new_offence_charged_against_earlier_conduct_names_the_section_it_rules_out(
    grid, provisions
):
    cells, _ = grid
    cell = next(
        c
        for c in cells
        if c.family.kind == "new_offence" and c.date_posture.name == "straddling"
    )
    key = T.answer_key_for(cell, provisions)
    assert key["governing_family"] == "old"
    assert key["charge"] is None  # nothing in the old codes to charge
    assert key["savings_consequence"] == "new_offence_cannot_reach_earlier_conduct"
    # The answer still has to NAME the new section, or it has ruled out
    # nothing; and it is never FORBIDDEN, because naming it is the answer.
    assert key["expected_sections"][0] == {
        "code": cell.family.new_ref.code, "number": cell.family.new_ref.number
    }
    assert key["forbidden_sections"] == []
    # No old-family section exists, so both-families is not demanded.
    assert key["must_name_both_families"] is False


def test_a_repealed_section_keeps_its_pre_repeal_postures_and_loses_the_others(grid, provisions):
    """A section REPEALED by the new code, with no court having touched it.

    The savings clause is the whole answer here: liability incurred before the
    appointed day survives the repeal, so the charge lies under the repealed
    section however long afterwards the matter is taken up.
    """
    cells, refused = grid
    deleted = [
        c for c in cells
        if c.family.kind == "deleted" and c.family.key not in T.JUDICIAL_INVALIDATIONS
    ]
    assert {c.family.key for c in deleted} == {"IPC 124A", "IPC 309"}
    for cell in deleted:
        key = T.answer_key_for(cell, provisions)
        assert key["governing_family"] == "old"
        assert key["judicial_status"] == T.STATUS_IN_FORCE
        assert key["charge"] == {
            "code": cell.family.old_ref.code, "number": cell.family.old_ref.number
        }
        assert key["counterpart"] is None
        assert key["savings_consequence"] == T.SAVINGS_PRESERVED
        assert key["requires_no_liability_statement"] is False

    ungateable = [entry for entry in refused if "cell" in entry]
    by_basis: dict[str, int] = {}
    for entry in ungateable:
        by_basis[entry["basis"]] = by_basis.get(entry["basis"], 0) + 1
    # The gate-stack half is unchanged in kind and smaller in count: the two
    # post-appointed-day postures of the three deleted families that build a
    # cell at all (IPC 377 never reaches this branch - it is refused earlier,
    # on the law).
    assert by_basis[T.REFUSAL_GATE_STACK] == 80 == 3 * 2 * 16 - 16
    assert by_basis[T.REFUSAL_LEGAL_CERTAINTY] == 96 == 80 + 16
    assert len(ungateable) == 176
    assert {entry["family"] for entry in ungateable} == {
        "IPC 124A", "IPC 309", "IPC 377", "IPC 497"
    }
    for entry in ungateable:
        if entry["basis"] == T.REFUSAL_GATE_STACK:
            assert "no suppression" in entry["reason"]
        else:
            assert "Joseph Shine" in entry["reason"] or "Navtej Singh Johar" in entry["reason"]


# --------------------------------------------------------------------------
# The second timeline: a section a court struck down.
# --------------------------------------------------------------------------

def test_the_invalidation_constants_are_grounded_in_the_audited_map(mapping):
    """A date this module keys an answer on may not come from recollection.

    The event, the case and the year are the audit sheet's; only the DAY is a
    constant here, and the constant carries the note saying so. This test is
    the tie: it reads the mapping row and asserts the constant's own claim
    about where it comes from is true of the file.
    """
    assert set(T.JUDICIAL_INVALIDATIONS) == {"IPC 497", "IPC 377"}
    for key, event in T.JUDICIAL_INVALIDATIONS.items():
        row = mapping.row(SectionRef(*key.split(" ", 1)))
        note = str(row.get("notes") or "")
        assert event.is_grounded_in(note), (key, note)
        assert event.case in note and str(event.year) in note
        assert row["kind"] == "deleted"
        # The source note says where the day came from and admits what it is.
        assert "operator-supplied constant" in event.source_note
        assert "ipc_bns_map.jsonl" in event.source_note
        assert event.scope in (T.SCOPE_SECTION_VOID, T.SCOPE_CONDUCT_SCOPED)
    assert T.JUDICIAL_INVALIDATIONS["IPC 497"].decided_on == date(2018, 9, 27)
    assert T.JUDICIAL_INVALIDATIONS["IPC 377"].decided_on == date(2018, 9, 6)
    assert T.JUDICIAL_INVALIDATIONS["IPC 497"].scope == T.SCOPE_SECTION_VOID
    assert T.JUDICIAL_INVALIDATIONS["IPC 377"].scope == T.SCOPE_CONDUCT_SCOPED


def test_a_constant_the_audit_sheet_does_not_carry_refuses_its_family(mapping, provisions):
    """The other direction: break the tie and the family stops emitting.

    Not a style check. The constant is the only thing that says WHEN the
    section stopped being chargeable, and a constant the sheet cannot
    corroborate is exactly the unstated recollection this stream must never
    key an answer on.
    """
    rows = copy.deepcopy(mapping.rows)
    victim = next(row for row in rows if row["old_section"] == "497")
    victim["notes"] = "Adultery. Repealed and not re-enacted."  # the case name goes
    cells, refused = T.build_grid(Mapping(rows), provisions=provisions)
    assert "IPC 497" not in {c.family.key for c in cells}
    reason = next(
        entry["reason"] for entry in refused
        if "cell" not in entry and entry["family"] == "IPC 497"
    )
    assert "Joseph Shine" in reason and "cannot be traced to the audit sheet" in reason


def test_a_note_recording_a_judgment_with_no_constant_refuses_its_family(mapping, provisions):
    """The operator writes a new note; the build refuses rather than keys it.

    This is the C1 failure mode generalised: the prompt told the teacher the
    section had been struck down while the key graded the answer as if it had
    not. A note nobody has dated cannot be keyed at all.
    """
    rows = copy.deepcopy(mapping.rows)
    victim = next(row for row in rows if row.get("verified_by") and row["kind"] == "one_to_one")
    key = str(SectionRef(victim["old_code"], victim["old_section"]))
    before, _ = T.build_grid(Mapping(copy.deepcopy(rows)), provisions=provisions)
    assert key in {c.family.key for c in before}

    victim["notes"] = (victim.get("notes") or "") + " Struck down in a later case."
    after, refused = T.build_grid(Mapping(rows), provisions=provisions)
    assert key not in {c.family.key for c in after}
    assert len(after) == len(before) - len(T.POSTURE_CELLS)
    reason = next(
        entry["reason"] for entry in refused
        if "cell" not in entry and entry["family"] == key
    )
    assert "records a judicial event" in reason and "struck down" in reason.lower()


def test_judicial_status_reads_the_timeline_and_not_the_appointed_day():
    """The three answers, at the boundary and on both sides of it."""
    shine = T.JUDICIAL_INVALIDATIONS["IPC 497"].decided_on
    assert T.judicial_status("IPC 302", date(2020, 1, 1)) == (T.STATUS_IN_FORCE, None)
    assert T.judicial_status("IPC 497", shine + timedelta(days=1))[0] == T.STATUS_VOID
    # The day itself and everything before it: refused, not guessed.
    for day in (shine, shine - timedelta(days=1), date(2001, 1, 1)):
        status, reason = T.judicial_status("IPC 497", day)
        assert status == T.STATUS_UNDECIDABLE
        assert "Joseph Shine" in reason and shine.isoformat() in reason
    # Conduct-scoped: undecidable on EVERY date, because the papers never say
    # whether the conduct was consensual.
    for day in (date(2001, 1, 1), date(2018, 9, 6), date(2023, 5, 5)):
        status, reason = T.judicial_status("IPC 377", day)
        assert status == T.STATUS_UNDECIDABLE
        assert "consent" in reason


def test_the_struck_down_family_keys_no_charge_lies_and_names_the_section(grid, provisions):
    """The C1 key, re-derived here from the timeline rather than read off the
    module: conduct after Joseph Shine, so no charge lies - and the answer
    must still NAME s.497, or it has ruled nothing out."""
    cells, _ = grid
    struck = [c for c in cells if c.family.key == "IPC 497"]
    assert len(struck) == 48
    shine = T.JUDICIAL_INVALIDATIONS["IPC 497"].decided_on
    for cell in struck:
        assert cell.offence_date > shine, cell.coordinates
        assert cell.offence_date < APPOINTED_DAY  # every surviving posture
        key = T.answer_key_for(cell, provisions)
        assert key["judicial_status"] == T.STATUS_VOID
        assert key["savings_consequence"] == T.SAVINGS_NO_OFFENCE_LIES
        assert key["charge"] is None, "nothing is chargeable under a void section"
        assert key["counterpart"] is None
        assert key["requires_no_liability_statement"] is True
        assert {"code": "IPC", "number": "497"} in key["expected_sections"]
        # ...and s.358 with it: the answer has to say what the savings clause
        # does here, which is preserve nothing.
        assert {"code": "BNS", "number": "358"} in key["expected_sections"]
        assert key["forbidden_sections"] == []
        assert key["judicial_event"]["case"] == "Joseph Shine"
        assert key["judicial_event"]["decided_on"] == shine.isoformat()


def test_the_postures_around_the_judgment_are_the_ones_the_grid_asks(grid, provisions):
    """just_before / just_after, anchored to the judgment for this family: one
    side is the question the grid keeps, the other is the one it refuses."""
    cells, refused = grid
    shine = T.JUDICIAL_INVALIDATIONS["IPC 497"].decided_on
    after = [
        c for c in cells
        if c.family.key == "IPC 497" and c.date_posture.name == "just_after"
    ]
    assert len(after) == 16
    for cell in after:
        assert cell.date_anchor == T.ANCHOR_INVALIDATION
        assert shine < cell.offence_date <= shine + timedelta(days=45)
        # The conduct sits days after the judgment and YEARS before the
        # appointed day, so the substantive limb is IPC-era and the question
        # is entirely about the judgment.
        assert cell.offence_date < APPOINTED_DAY
    refusals = [
        entry for entry in refused
        if entry.get("family") == "IPC 497" and "cell" in entry
        and "just_before" in entry["cell"]
    ]
    assert len(refusals) == 16
    for entry in refusals:
        assert entry["basis"] == T.REFUSAL_LEGAL_CERTAINTY
        assert "on or before 2018-09-27" in entry["reason"]


def test_the_read_down_family_emits_nothing_and_says_why(grid, mapping, provisions):
    """IPC 377 is verified, is not refused as a FAMILY, and still builds no
    cell: after Navtej Singh Johar the section's reach turns on consent, and
    the fact skeletons deliberately do not narrate it."""
    cells, refused = grid
    assert "IPC 377" in {
        str(SectionRef(row["old_code"], row["old_section"]))
        for row in mapping.verified_rows()
        if row["old_code"]
    }
    assert "IPC 377" not in {c.family.key for c in cells}
    assert "IPC 377" not in {e["family"] for e in refused if "cell" not in e}
    cell_refusals = [e for e in refused if e.get("family") == "IPC 377" and "cell" in e]
    assert len(cell_refusals) == 80 == len(T.POSTURE_CELLS)
    for entry in cell_refusals:
        assert entry["basis"] == T.REFUSAL_LEGAL_CERTAINTY
        assert "Navtej Singh Johar" in entry["reason"] and "consent" in entry["reason"]


def test_the_no_charge_lies_answer_passes_and_its_opposite_is_rejected(cfg, grid, provisions):
    """The whole of C1, at the gate: the legally correct answer PASSES and the
    answer the old key demanded - the charge lies under the struck-down
    section - is the one that fails."""
    cells, _ = grid
    cell = next(
        c for c in cells
        if c.family.key == "IPC 497" and c.question_form.name == "charge_only"
    )
    row = T.seed_row(cell, provisions, held_out=False)
    _, ctx = _ctx_for(cfg, row)

    correct = (
        "Conclusion: no charge lies under Section 497 of the Indian Penal Code. It was "
        "struck down before this conduct, so no liability was ever incurred under it and "
        "Section 358 BNS - which preserves liability incurred - preserves nothing here."
    )
    wrong = (
        "Conclusion: the charge lies under Section 497 of the Indian Penal Code, preserved "
        "by Section 358 BNS."
    )
    hedged = (
        "Conclusion: no charge lies under Section 497 IPC; the charge lies under Section "
        "497 IPC, preserved by Section 358 BNS."
    )
    good = gates.check_answer_key(correct, ctx, think="x")
    assert good.passed, good.detail
    assert gates.check_temporal(f"<think>x</think>\n{correct}", ctx).passed
    bad = gates.check_answer_key(wrong, ctx, think="x")
    assert not bad.passed
    assert bad.detail["liability_asserted"] == ["the charge lies under"]
    assert bad.detail["missing"] == [], "it cites everything the key asks for and is still wrong"
    # An answer that says both things is not a correct answer either.
    assert not gates.check_answer_key(hedged, ctx, think="x").passed


def test_only_charge_only_forbids_the_counterpart(grid, provisions):
    cells, _ = grid
    for cell in _stratified(cells, lambda c: (c.question_form.name, c.family.kind)):
        key = T.answer_key_for(cell, provisions)
        if key["forbidden_sections"]:
            assert cell.question_form.name == "charge_only"
            assert key["forbidden_sections"] == [key["counterpart"]]
            assert key["charge"] is not None
    # ...and it does forbid it whenever there is one to forbid: the other
    # direction, so the rule above is not vacuously satisfied by an empty
    # forbidden list everywhere.
    with_counterpart = [
        c
        for c in cells
        if c.question_form.name == "charge_only"
        and c.family.old_ref is not None
        and c.family.new_ref is not None
    ]
    assert with_counterpart
    for cell in with_counterpart[:60]:
        assert T.answer_key_for(cell, provisions)["forbidden_sections"]


def test_the_key_is_shaped_the_way_check_answer_key_reads_it(selection, provisions):
    cell = selection.sample[0]
    key = T.answer_key_for(cell, provisions)
    assert set(key) >= {
        "governing_family",
        "expected_sections",
        "forbidden_sections",
        "requires_savings_mention",
        "requires_no_liability_statement",
        "must_name_both_families",
    }
    assert json.loads(json.dumps(key)) == key  # store.upsert_seeds serialises it


# --------------------------------------------------------------------------
# The two gates, on the answer the key describes.
# --------------------------------------------------------------------------

def _ideal_answer(key: dict) -> str:
    cites = " ".join(
        f"Section {entry['number']} {entry['code']}" for entry in key["expected_sections"]
    )
    answer = f"Issue\nRule\nApplication\nConclusion: the position is {cites}."
    if key.get("requires_no_liability_statement"):
        # The whole of what the key asks for on a struck-down section: name it,
        # and say that no charge lies under it.
        answer += " No charge lies: the section was struck down before this conduct."
    return answer


def _ctx_for(cfg, row: dict):
    seed = _seed_for_store(row)
    task = {
        "task_type": T.TASK_TYPE,
        "seed_id": row["seed_id"],
        "stream": T.STREAM,
        "prompt_id": "gen_transition_v1",
    }
    bundle = generate.build_prompt(cfg, task, seed)
    return bundle, generate.gate_context(cfg, task, seed, bundle.grounding)


def test_the_answer_the_key_describes_passes_both_permanent_gates(cfg, selection, provisions):
    """check_answer_key and check_temporal read the same generation, and a key
    that satisfies one while the other rejects it is a key that burns seeds.

    This is the test that found the two real conflicts: an answer that merely
    used the word "savings" cleared gates._mentions_savings but not
    statutes._cites_savings_clause (287/1250 cells), and a repealed section
    cited under a new-family posture has no suppression at all (128 cells).
    """
    checked = 0
    for cell in selection.reserve + selection.sample:
        row = T.seed_row(cell, provisions, held_out=False)
        _, ctx = _ctx_for(cfg, row)
        key = row["answer_key_json"]
        answer = _ideal_answer(key)
        key_result = gates.check_answer_key(answer, ctx, think="reasoning")
        temporal = gates.check_temporal(f"<think>reasoning</think>\n{answer}", ctx)
        assert key_result.passed, (cell.coordinates, key_result.detail)
        assert temporal.passed, (cell.coordinates, temporal.detail)
        checked += 1
    assert checked == 1250


def test_an_answer_that_cites_the_wrong_family_is_caught(cfg, selection, provisions):
    # The other direction: the gates are not passing everything.
    cell = next(
        c for c in selection.sample
        if c.family.old_ref is not None and c.family.new_ref is not None
        and c.date_posture.name == "well_before"
    )
    row = T.seed_row(cell, provisions, held_out=False)
    _, ctx = _ctx_for(cfg, row)
    key = row["answer_key_json"]
    assert key["governing_family"] == "old"
    wrong = f"Conclusion: the charge lies under Section {cell.family.new_ref.number} BNS."
    assert not gates.check_answer_key(wrong, ctx, think="x").passed


def test_ungateable_reason_reads_both_directions(grid, provisions):
    cells, _ = grid
    good = T.answer_key_for(cells[0], provisions)
    assert T.ungateable_reason(good) is None
    # An old-family section demanded where the new family governs: no
    # suppression path exists, so the ideal answer is a permanent reject.
    bad = dict(good)
    bad["families_by_kind"] = {"substantive": "new", "procedural": "new", "evidence": "new"}
    bad["expected_sections"] = [{"code": "IPC", "number": "302"}]
    bad["requires_savings_mention"] = False
    assert "no suppression" in T.ungateable_reason(bad)
    # A NEW-family section where the old family governs is fine - but only
    # while the savings mention is required, which is what suppresses it.
    other = dict(good)
    other["families_by_kind"] = {"substantive": "old", "procedural": "old", "evidence": "old"}
    other["expected_sections"] = [{"code": "BNS", "number": "103"}]
    other["requires_savings_mention"] = True
    assert T.ungateable_reason(other) is None
    other["requires_savings_mention"] = False
    assert "no suppression" in T.ungateable_reason(other)


# --------------------------------------------------------------------------
# Prompts: skeletons, provisions, slots.
# --------------------------------------------------------------------------

def test_every_procedural_posture_has_a_skeleton_and_every_skeleton_a_posture():
    on_disk = {path.stem for path in T.TEMPLATES_DIR.glob("*.md")}
    assert on_disk == {pp.skeleton_id for pp in T.PROCEDURAL_POSTURES}


def test_skeleton_shas_are_pinned():
    """Same discipline as the prompt registry's golden hashes: editing a
    skeleton is a deliberate, reviewed act, and every cell built afterwards
    records the new sha so two runs are never silently compared across it."""
    assert {pp.skeleton_id: T.load_skeleton(pp.skeleton_id).sha for pp in T.PROCEDURAL_POSTURES} == {
        "posture_fir_v1": "ec7ebfd2a92f",
        "posture_chargesheet_v1": "a30be2610ddf",
        "posture_trial_v1": "8b3b6d58ff43",
        "posture_appeal_v1": "ce18143fb3aa",
    }


@pytest.mark.parametrize(
    "body,match",
    [
        ("<!-- posture -->\nx", "no <!-- papers --> block"),
        ("<!-- papers -->\nx", "no <!-- posture --> block"),
        ("<!-- posture -->\na\n<!-- papers -->\nb", "before"),
        ("stray\n<!-- papers -->\na\n<!-- posture -->\nb", "text before"),
        ("<!-- papers -->\n\n<!-- posture -->\nb", "empty <!-- papers --> block"),
        ("<!-- papers -->\na\n<!-- posture -->\n  ", "empty <!-- posture --> block"),
    ],
)
def test_a_malformed_skeleton_is_refused(body, match):
    with pytest.raises(ValueError, match=match):
        T._split_skeleton(body, "fixture_v1")


def test_an_unknown_skeleton_names_the_ones_that_exist():
    with pytest.raises(KeyError, match="posture_fir_v1"):
        T.load_skeleton("posture_nonexistent_v9")


def test_the_three_transition_provisions_load_and_say_they_are_not_quotations(provisions):
    assert set(provisions) == {"substantive", "procedural", "evidence"}
    assert provisions["substantive"].ref == SectionRef("BNS", "358")
    assert provisions["procedural"].ref == SectionRef("BNSS", "531")
    assert provisions["evidence"].ref == SectionRef("BSA", "170")
    for provision in provisions.values():
        # The repo holds no bare-act corpus, so nothing here claims to be one.
        assert provision.text_kind == "recorded-effect"
        assert "not a quotation" in provision.block()
        assert provision.derived_from


def test_a_provision_file_missing_a_limb_is_refused(tmp_path):
    rows = [
        json.loads(line)
        for line in T.TRANSITION_PROVISIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keep = [row for row in rows if row.get("kind") != "evidence"]
    path = tmp_path / "provisions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in keep) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence"):
        T.load_provisions(path)


@pytest.mark.parametrize("field", T._REQUIRED_PROVISION_KEYS)
def test_a_provision_row_missing_any_field_is_refused(tmp_path, field):
    rows = [
        json.loads(line)
        for line in T.TRANSITION_PROVISIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        if row.get("kind") == "substantive":
            row.pop(field, None)
    path = tmp_path / "provisions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        T.load_provisions(path)


def test_a_provision_block_carries_the_text_when_one_exists(tmp_path):
    # text_kind is read, not assumed: the day a bare-act corpus lands, the
    # same row carries the section as enacted and the label changes with it.
    rows = [
        json.loads(line)
        for line in T.TRANSITION_PROVISIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        if row.get("kind") == "substantive":
            row["text_kind"] = "verbatim"
    path = tmp_path / "provisions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    loaded = T.load_provisions(path)
    assert "Text as enacted" in loaded["substantive"].block()
    assert "not a quotation" not in loaded["substantive"].block()


def test_no_section_number_ever_reaches_the_scenario_or_the_papers(selection, provisions):
    """{scenario} is NOT part of grounding_text, so a section named there is a
    section the answer may cite without ever having been shown it. The papers
    stay clean for the same reason - they describe the conduct by reference to
    the provisions below rather than by number."""
    for cell in selection.sample[:200] + selection.reserve[:50]:
        row = T.seed_row(cell, provisions, held_out=False)
        assert extract_sections(row["meta_json"]["scenario"]) == []
        assert extract_sections(row["text"]) == []


def test_the_provision_blocks_parse_back_to_the_sections_the_key_names(selection, provisions):
    for cell in selection.sample[:150]:
        row = T.seed_row(cell, provisions, held_out=False)
        meta = row["meta_json"]
        cited = set(
            extract_sections(meta["old_section_text"])
            + extract_sections(meta["new_section_text"])
            + extract_sections(meta["savings_text"])
        )
        for entry in row["answer_key_json"]["expected_sections"]:
            ref = SectionRef(entry["code"], entry["number"])
            assert ref in cited, (cell.coordinates, str(ref), sorted(map(str, cited)))


def test_the_grounding_carries_every_expected_section(cfg, selection, provisions):
    # generate.grounding_text dedupes parts, and a dedupe that swallowed a
    # provision block would turn a correct citation into a PERMANENT reject at
    # check_citations. So the union is checked after the dedupe, not before.
    for cell in selection.sample[:80]:
        row = T.seed_row(cell, provisions, held_out=False)
        bundle, _ = _ctx_for(cfg, row)
        cited = {str(ref) for ref in extract_sections(bundle.grounding)}
        for entry in row["answer_key_json"]["expected_sections"]:
            assert f"{entry['code']} {entry['number']}" in cited


def test_every_transition_template_renders_from_a_cell(cfg, selection, provisions):
    row = T.seed_row(selection.sample[0], provisions, held_out=False)
    seed = _seed_for_store(row)
    for prompt_id in prompt_registry.variants("transition"):
        task = {
            "task_type": T.TASK_TYPE,
            "seed_id": row["seed_id"],
            "stream": T.STREAM,
            "prompt_id": prompt_id,
        }
        bundle = generate.build_prompt(cfg, task, seed)
        assert bundle.messages[-1]["role"] == "user"
        assert "{" not in bundle.messages[-1]["content"].replace("{{", "")
        assert bundle.prompt_est_tokens < cfg.build.length_band.total_max


def test_the_question_slot_carries_the_no_quotation_caution(selection, provisions):
    # In the QUESTION, which is not grounding: a caution inside a provision
    # block would be grounding text, and a trace echoing thirty characters of
    # it would trip the verbatim gate on a sentence this module wrote.
    row = T.seed_row(selection.sample[0], provisions, held_out=False)
    assert T.NO_QUOTATION_CAUTION in row["meta_json"]["question"]
    assert T.NO_QUOTATION_CAUTION not in row["meta_json"]["savings_text"]
    assert T.NO_QUOTATION_CAUTION not in row["text"]


def test_the_caution_forbids_the_answer_side_quote_too(selection, provisions):
    # The earlier wording said "do not quote words you have not been shown",
    # and the recorded effect HAS been shown - so it licensed exactly the
    # artefact it existed to prevent. It now says answer as well as reasoning,
    # in as many words.
    row = T.seed_row(selection.sample[0], provisions, held_out=False)
    caution = row["meta_json"]["question"]
    assert T.NO_QUOTATION_CAUTION in caution
    assert "in your reasoning or in your answer" in T.NO_QUOTATION_CAUTION
    assert "not the section's words" in T.NO_QUOTATION_CAUTION


def test_an_answer_quoting_the_build_paraphrase_as_the_statute_is_caught(
    cfg, selection, provisions
):
    """END TO END, on a real cell, through the whole gate stack.

    Measured at HEAD: this answer passed all nine gates and its disposition
    was clean, so a row presenting this build's paraphrase as the enacted
    words of s.358(2) would have entered the dataset. Only the trace side was
    caught, by verbatim_overlap.
    """
    cell = selection.sample[0]
    row = T.seed_row(cell, provisions, held_out=False)
    _, ctx = _ctx_for(cfg, row)
    key = row["answer_key_json"]

    faithful = _ideal_answer(key) + (
        " Section 358 of the Bharatiya Nyaya Sanhita preserves liabilities already "
        "incurred under the repealed Code, stated here in my own words."
    )
    quoted = _ideal_answer(key) + (
        ' Section 358(2) of the Bharatiya Nyaya Sanhita, 2023 provides: "The repeal of '
        'the Indian Penal Code, 1860 does not affect any right, privilege, obligation '
        'or liability acquired, accrued or incurred under it".'
    )
    assert gates.check_statutory_quotation(faithful, ctx).passed
    caught = gates.check_statutory_quotation(quoted, ctx)
    assert not caught.passed
    assert caught.detail["quotations"][0]["reproduces_grounding"] is True
    # The answer key and the temporal gate see nothing wrong with either, which
    # is precisely why the finding needed a gate of its own.
    assert gates.check_answer_key(quoted, ctx, think="x").passed


def test_the_seed_row_fills_every_slot_generate_requires(selection, provisions):
    row = T.seed_row(selection.sample[3], provisions, held_out=False)
    for name in ("scenario", "old_section_text", "new_section_text", "savings_text"):
        assert row["meta_json"][name].strip()
    assert row["text"].strip()
    assert row["offence_date"] and row["meta_json"]["proceeding_started"]


def test_the_stream_and_task_type_are_the_ones_the_rest_of_the_pipeline_uses():
    assert T.STREAM == gates.TRANSITION_STREAM
    assert T.TASK_TYPE in prompt_registry.task_types()
    assert set(tasks.TRANSITION_MIX) == {T.TASK_TYPE}
    assert T.STREAM in tasks.PLANNABLE_STREAMS


# --------------------------------------------------------------------------
# The build.
# --------------------------------------------------------------------------

def test_build_transition_writes_the_draw_and_reports_the_shape(store, cfg, mapping, provisions):
    manifest = T.build_transition(store, cfg, mapping=mapping, provisions=provisions)
    assert manifest["grid_cells"] == 12144
    assert manifest["families_emitting"] == 153
    assert manifest["families_refused"] == 17
    assert manifest["cells_refused"] == 176
    assert manifest["cells_refused_by_basis"] == {"legal-certainty": 96, "gate-stack": 80}
    # The family that passes every audit gate and still emits nothing is named
    # rather than left to be inferred from two counts that no longer subtract.
    assert manifest["families_emitting_nothing"] == ["IPC 377"]
    assert (manifest["sample"], manifest["reserve"]) == (1100, 150)
    assert manifest["written"] == 1250
    assert manifest["sample_families_covered"] == 153
    assert manifest["sample_posture_pairs"] == manifest["posture_pairs_total"] == 20
    assert store.seed_count(T.TRANSITION_SOURCE_ID) == 1250
    assert [event["kind"] for event in store.events("transition_grid_built")]


def test_build_transition_is_idempotent(store, cfg, mapping, provisions):
    T.build_transition(store, cfg, mapping=mapping, provisions=provisions)
    before = store.seed_count()
    T.build_transition(store, cfg, mapping=mapping, provisions=provisions)
    assert store.seed_count() == before == 1250


def test_a_dry_run_measures_and_writes_nothing(store, cfg, mapping, provisions):
    manifest = T.build_transition(
        store, cfg, mapping=mapping, provisions=provisions, dry_run=True
    )
    assert manifest["written"] == 0
    assert manifest["grid_cells"] == 12144
    assert store.seed_count() == 0


def test_a_config_without_a_transition_block_is_refused(store, cfg, mapping, provisions):
    import dataclasses

    naked = dataclasses.replace(cfg, transition=None)
    with pytest.raises(ValueError, match="transition.sample"):
        T.build_transition(store, naked, mapping=mapping, provisions=provisions)


# --------------------------------------------------------------------------
# The reserve can never be planned.
# --------------------------------------------------------------------------

def test_a_held_out_cell_is_never_offered_to_the_wave_planner(store, cfg, mapping, provisions):
    T.build_transition(store, cfg, mapping=mapping, provisions=provisions)
    rows = tasks.plan_rows(store, cfg, T.STREAM, 2000)
    planned = {row["seed_id"] for row in rows}
    held_out = {
        row[0]
        for row in store.conn.execute(
            "SELECT seed_id FROM seed WHERE json_extract(meta_json, '$.held_out') = 1"
        ).fetchall()
    }
    assert len(held_out) == 150
    assert not (planned & held_out)
    # The other direction: the SAMPLE is plannable, so the exclusion above is
    # not simply "nothing was planned".
    assert len(planned) > 1000
    assert planned <= {
        row[0]
        for row in store.conn.execute(
            "SELECT seed_id FROM seed WHERE json_extract(meta_json, '$.held_out') = 0"
        ).fetchall()
    }


def test_clearing_the_held_out_mark_makes_the_same_seed_plannable(store, cfg, mapping, provisions):
    T.build_transition(store, cfg, mapping=mapping, provisions=provisions)
    victim = store.conn.execute(
        "SELECT seed_id FROM seed WHERE json_extract(meta_json, '$.held_out') = 1 "
        "ORDER BY seed_id LIMIT 1"
    ).fetchone()[0]
    assert victim not in {row["seed_id"] for row in tasks.plan_rows(store, cfg, T.STREAM, 5000)}

    seed = store.get_seed(victim)
    meta = json.loads(seed["meta_json"])
    meta["held_out"] = False
    store.upsert_seeds([{**seed, "meta_json": meta}])
    assert victim in {row["seed_id"] for row in tasks.plan_rows(store, cfg, T.STREAM, 5000)}


def test_a_transition_seed_is_never_drawn_into_another_stream(store, cfg, mapping, provisions):
    """The CLI default is `--source` unset, so a synthesis wave sees the whole
    seed table - and the transition grid is now in it.

    Measured before the clause landed, on a store holding nothing else: eight
    tasks across drafting, irac_analysis, statute_qa and summarization, each on
    a transition seed. The transition QUESTION rode into them (meta_json.
    question overrides the task-type default), so the teacher was asked which
    enactment governs with no provision block in front of it, and
    check_answer_key skipped the row for not being on the transition stream -
    a row with an answer key, ungraded against it.
    """
    T.build_transition(store, cfg, mapping=mapping, provisions=provisions)
    for other in ("synthesis", "curated_c2"):
        assert tasks.plan_rows(store, cfg, other, 8, sources=None) == []
    # The other direction: its OWN wave still draws it, or the clause would be
    # satisfied by planning nothing at all.
    own = tasks.plan_rows(store, cfg, T.STREAM, 8)
    assert len(own) == 8
    assert {row["task_type"] for row in own} == {T.TASK_TYPE}


def test_a_seed_that_declares_no_stream_is_offered_to_every_wave(store, cfg):
    """The clause must be inert for every other builder: transition.py is the
    only writer of meta_json.stream, and a planner that treated a missing
    declaration as a refusal would empty every wave in the build."""
    store.upsert_source("fixture/source", "CC0")
    store.upsert_seeds(
        [
            {
                "seed_id": "streamless00001",
                "source_id": "fixture/source",
                "text": "x" * 600,
                "meta_json": {"estimator": "chars/4"},
            }
        ]
    )
    for stream in ("synthesis", "curated_c2", T.STREAM):
        planned = {row["seed_id"] for row in tasks.plan_rows(store, cfg, stream, 2)}
        assert "streamless00001" in planned, stream


def test_a_seed_with_no_held_out_key_is_still_plannable(store, cfg):
    # Every other source writes meta_json without the key, and treating a
    # missing flag as "held out" would empty every wave in the build.
    store.upsert_source("fixture/source", "CC0")
    store.upsert_seeds(
        [
            {
                "seed_id": "plainseed000001",
                "source_id": "fixture/source",
                "text": "x" * 600,
                "meta_json": {"estimator": "chars/4"},
            }
        ]
    )
    cfg = load_build_config(DATA_CONFIG, allow_unpinned=True)
    rows = tasks.plan_rows(store, cfg, "synthesis", 4)
    assert "plainseed000001" in {row["seed_id"] for row in rows}


# --------------------------------------------------------------------------
# What this module must never do.
# --------------------------------------------------------------------------

def test_nothing_in_this_module_reaches_a_model():
    """The stream's whole claim is that its answers are known in advance. A
    generated fact, a generated provision or a generated key would make the
    answer key a model's opinion, so the module is checked for the seam rather
    than trusted not to grow one."""
    src = TRANSITION_SRC.read_text(encoding="utf-8")
    for forbidden in ("providers", "Router", "httpx", "openai", "async def", "await "):
        assert forbidden not in src, forbidden


def test_the_module_never_reads_a_model_generated_field(selection, provisions):
    # Everything in a rendered cell traces to a template, the mapping resource
    # or the provisions resource - so the rendered slots are byte-reproducible
    # from those three inputs alone.
    cell = selection.sample[7]
    first = T.render_cell(cell, provisions)
    second = T.render_cell(cell, provisions)
    assert first == second
    skeleton = T.load_skeleton(cell.procedural_posture.skeleton_id)
    assert first["source"] == skeleton.papers
    assert first["savings_text"] == T.savings_block(provisions)
