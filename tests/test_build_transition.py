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
    assert len(families) == len(mapping.verified_rows()) == 154
    # 154 x 80 minus the cells no gate stack could accept (see the
    # ungateable test below), which is the whole of the difference.
    ungateable = [entry for entry in refused if "cell" in entry]
    assert len(cells) == len(families) * len(T.POSTURE_CELLS) - len(ungateable)
    assert len(cells) == 12192


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
    assert len(per_family) == len({c.family.key for c in cells}) == 154
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


def test_the_date_postures_mean_what_they_are_named(grid):
    cells, _ = grid
    seen = {}
    for cell in cells:
        seen.setdefault(cell.date_posture.name, []).append(cell)
    for cell in seen["well_before"] + seen["just_before"]:
        assert cell.offence_date < APPOINTED_DAY and cell.proceeding_started < APPOINTED_DAY
    for cell in seen["on_appointed_day"]:
        assert cell.offence_date == cell.proceeding_started == APPOINTED_DAY
    for cell in seen["just_after"]:
        assert cell.offence_date >= APPOINTED_DAY and cell.proceeding_started > APPOINTED_DAY
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
    old_ref, new_ref = cell.family.old_ref, cell.family.new_ref
    charge = old_ref if substantive == "old" else new_ref
    counterpart = new_ref if substantive == "old" else old_ref
    engage = charge if charge is not None else counterpart

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

    if charge is not None:
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
    # 4 kinds x 80 posture triples, minus the (deleted x post-appointed-day)
    # combinations build_grid excludes.
    assert len(strata) == 4 * 80 - 32
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
    # No-counterpart new offences and read-down/deleted sections, named in the
    # plan. IPC 377 and 497 are `deleted` and keep every pre-repeal posture.
    for named in ("BNS 111", "BNS 113", "BNS 103(2)", "BNS 304", "BNS 69"):
        assert named in families
    for named in ("IPC 377", "IPC 497"):
        assert named in families
        assert {
            c.date_posture.name for c in cells if c.family.key == named
        } == {"well_before", "just_before", "straddling"}


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
    cells, refused = grid
    deleted = [c for c in cells if c.family.kind == "deleted"]
    assert deleted
    for cell in deleted:
        key = T.answer_key_for(cell, provisions)
        assert key["governing_family"] == "old"
        assert key["charge"] == {
            "code": cell.family.old_ref.code, "number": cell.family.old_ref.number
        }
        assert key["counterpart"] is None
        assert key["savings_consequence"] == "old_liability_preserved"

    ungateable = [entry for entry in refused if "cell" in entry]
    assert len(ungateable) == 128 == 4 * 2 * 16
    assert {entry["family"] for entry in ungateable} == {
        "IPC 124A", "IPC 309", "IPC 377", "IPC 497"
    }
    for entry in ungateable:
        assert "no suppression" in entry["reason"]


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
    return f"Issue\nRule\nApplication\nConclusion: the position is {cites}."


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
    assert manifest["grid_cells"] == 12192
    assert manifest["families_emitting"] == 154
    assert manifest["families_refused"] == 17
    assert manifest["cells_refused"] == 128
    assert (manifest["sample"], manifest["reserve"]) == (1100, 150)
    assert manifest["written"] == 1250
    assert manifest["sample_families_covered"] == 154
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
    assert manifest["grid_cells"] == 12192
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
