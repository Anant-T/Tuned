"""The 50-example review packet the dataset card makes a ship prerequisite.

Two of these tests exist because the ad-hoc version of this screen got the
answer WRONG twice on 2026-08-31, in opposite directions - once inventing a
finding, once returning a clean bill of health from an instrument that could
not fire at all. Both are pinned here.
"""
import pytest

from tuned.data import review


# --- stratified_sample ------------------------------------------------------


def _rows(**counts):
    out = []
    for cell, n in counts.items():
        for i in range(n):
            out.append({"task_id": f"{cell}-{i:04d}", "cell": cell})
    return out


def test_a_tiny_cell_is_not_washed_out_by_a_proportional_draw():
    # transition had 5 accepted rows against irac_analysis' many hundreds; a
    # proportional draw of 50 returns it 0 or 1 times, and the stream with the
    # most legal risk goes unread.
    rows = _rows(irac=700, transition=5)
    got = review.stratified_sample(rows, n=50, floor=3, key=lambda r: r["cell"])
    picked = [r["cell"] for r in got]
    assert picked.count("transition") == 3


def test_the_floor_never_takes_more_of_a_cell_than_exists():
    rows = _rows(irac=40, drafting=2)
    got = review.stratified_sample(rows, n=20, floor=3, key=lambda r: r["cell"])
    assert [r["cell"] for r in got].count("drafting") == 2


def test_the_draw_is_the_requested_size_and_has_no_duplicates():
    rows = _rows(a=100, b=100, c=7)
    got = review.stratified_sample(rows, n=50, floor=3, key=lambda r: r["cell"])
    assert len(got) == 50
    assert len({r["task_id"] for r in got}) == 50


def test_the_draw_is_deterministic_for_a_given_salt():
    rows = _rows(a=100, b=60)
    first = review.stratified_sample(rows, n=30, floor=3, key=lambda r: r["cell"], salt="s")
    again = review.stratified_sample(rows, n=30, floor=3, key=lambda r: r["cell"], salt="s")
    assert [r["task_id"] for r in first] == [r["task_id"] for r in again]


def test_a_different_salt_draws_a_different_sample():
    rows = _rows(a=200)
    one = review.stratified_sample(rows, n=20, floor=0, key=lambda r: r["cell"], salt="x")
    two = review.stratified_sample(rows, n=20, floor=0, key=lambda r: r["cell"], salt="y")
    assert [r["task_id"] for r in one] != [r["task_id"] for r in two]


def test_asking_for_more_than_exists_returns_everything_once():
    rows = _rows(a=4, b=3)
    got = review.stratified_sample(rows, n=50, floor=3, key=lambda r: r["cell"])
    assert len(got) == 7


# --- unsourced_references ---------------------------------------------------


def test_a_section_the_source_never_names_is_reported():
    found = review.unsourced_references("The charge lies under Section 420.",
                                        "This appeal concerns Section 302.")
    assert found.sections == ("420",)


def test_a_section_the_source_does_name_is_not_reported():
    found = review.unsourced_references("The charge lies under Section 302.",
                                        "This appeal concerns Section 302.")
    assert found.sections == ()


def test_prose_after_a_section_number_is_not_read_as_the_section():
    # THE FIRST BUG. One permissive pattern on both sides read
    # "Section 29 contains" as section "29CON" and reported it missing from a
    # source that plainly contained it.
    found = review.unsourced_references("Section 29 contains the definition.",
                                        "Section 29 of the Act applies.")
    assert found.sections == ()


def test_a_suffixed_section_is_kept_whole():
    found = review.unsourced_references("punishable under Section 304B.", "nothing here")
    assert found.sections == ("304B",)


def test_the_source_may_space_the_suffix_the_answer_joins():
    # "Section 302 A" in a judgment and "Section 302A" in the answer are the
    # same section; the source side is matched loosely on purpose, because a
    # near-miss there only SUPPRESSES a flag.
    found = review.unsourced_references("under Section 302A.", "see Section 302 A of the Code")
    assert found.sections == ()


def test_a_bare_number_in_the_source_does_not_excuse_a_section_in_the_answer():
    # THE SECOND BUG. Making the source pattern prefix-optional let every page
    # and paragraph number in a 40-page judgment count as a section mention, so
    # the screen returned zero findings BY CONSTRUCTION - it could not fire.
    found = review.unsourced_references("The charge lies under Section 420.",
                                        "At page 420 the learned judge observed as follows.")
    assert found.sections == ("420",)


def test_a_reported_citation_absent_from_the_source_is_reported():
    found = review.unsourced_references("as held in (2019) 5 SCC 123", "no authority here")
    assert any("SCC" in c for c in found.citations)


def test_a_reported_citation_present_in_the_source_is_not_reported():
    found = review.unsourced_references("as held in (2019) 5 SCC 123",
                                        "following (2019) 5 SCC 123, the court held")
    assert found.citations == ()


def test_nothing_cited_is_an_empty_report_not_a_failure():
    found = review.unsourced_references("The appeal is allowed.", "some source text")
    assert found.sections == () and found.citations == ()
