"""segment.py - whole judgment text -> tier-selected, gapless segments.

Fixtures are structural shapes with invented prose (paragraph numbering,
lettered headings, footnote blocks) - no verbatim S.C.R./eval text anywhere.
"""

import subprocess

import pytest

from tuned.data.roles_infer import BACKEND_NONE, BACKEND_SUBPROCESS, RolesBridgeError
from tuned.data.segment import (
    FOOTNOTES_CITED_HEADING,
    FOOTNOTES_LABEL,
    MAX_PARA_STEP,
    MIN_TOC_HEADINGS,
    RESTART_COST,
    TIER_PACKING,
    TIER_ROLES,
    TIER_TOC,
    WHY_PACKING,
    WHY_ROLES,
    WHY_TOC,
    Segment,
    _normalize_segments,
    _packing_tier,
    _split_footnote_tail,
    _subdivide,
    _toc_segments,
    cited_footnotes,
    paragraph_offsets,
    paragraph_starts,
    parse_footnotes,
    resolve_footnotes,
    segment_document,
    toc_candidates,
)


def para(n: int, words: int = 40) -> str:
    return f"{n}. " + ("word " * words).strip() + "."


def judgment(n_paras: int = 5, *, heading: str = "JUDGMENT", words: int = 40) -> str:
    body = "\n\n".join(para(i, words) for i in range(1, n_paras + 1))
    return f"{heading}\n\nRAO, J.\n\n{body}\n"


def reconstruct(text: str, segments) -> str:
    return "".join(text[s.start : s.end] for s in segments)


# --------------------------------------------------------------------------
# Reconstruction is the invariant every test below leans on.
# --------------------------------------------------------------------------


def assert_gapless_partition(text: str, segments):
    assert reconstruct(text, segments) == text
    cursor = 0
    for seg in segments:
        assert seg.start == cursor, "segments must be contiguous with no gap"
        assert seg.end >= seg.start
        cursor = seg.end
    assert cursor == len(text)


# --------------------------------------------------------------------------
# Empty text.
# --------------------------------------------------------------------------


def test_empty_text_is_packing_tier_with_no_segments_and_recorded_degradation():
    result = segment_document("")
    assert result.tier == TIER_PACKING
    assert result.segments == ()
    assert result.degradation == {"from": "text", "reason": "empty_text"}


# --------------------------------------------------------------------------
# Paragraph detection - the workhorse's own correctness.
#
# The rule under test is NOT "strictly increasing". That rule shipped once and
# was wrong in a way its own tests could not see: it assumed a quoted number
# is never HIGHER than the number the surrounding prose already reached, and
# on real judgments it routinely is (a quoted paragraph of the judgment under
# appeal, a quoted statute section, a wrapped "Article 335"). When it is, the
# strictly-increasing counter spikes and every genuine later paragraph is
# rejected - splitting the paragraph at the citation AND swallowing the rest
# of the judgment. The four shapes below are the ones measured on the real
# staged corpus; every one of them has its own test here.
# --------------------------------------------------------------------------


def numbers(text) -> list[int]:
    return [n for _offset, n in paragraph_starts(text)]


def test_consecutive_numbered_paragraphs_are_all_accepted():
    text = judgment(5)
    assert numbers(text) == [1, 2, 3, 4, 5]


def test_the_close_paren_marker_form_is_also_recognised():
    # extract.py's own numbered-paragraph signal accepts both "1." and "1)" -
    # this module's marker must too, not only the period form every other
    # fixture in this file happens to use.
    text = "JUDGMENT\n\n1) First paragraph text here.\n\n2) Second paragraph text here.\n\n"
    assert numbers(text) == [1, 2]


def test_a_quoted_lower_paragraph_number_is_rejected_not_a_new_boundary():
    # The direction the old strictly-increasing rule got right, kept as a
    # test that actually REACHES the rule: the quoted number is line-initial
    # and unquoted, so _PARA_START's own `^` anchor matches it and only the
    # accept/reject rule can throw it out. (The version of this test that
    # shipped put its quoted number mid-line behind a quotation mark, where
    # the anchor rejected it before any rule was consulted - deleting the
    # whole filter left that test green.)
    text = (
        "JUDGMENT\n\nRAO, J.\n\n"
        "1. This appeal raises a narrow question.\n\n"
        "2. The parties were heard at some length on the second day.\n\n"
        "3. The report the appellant leans upon opens in these words:\n\n"
        "1. The onus rests where it always has, on the prosecution.\n\n"
        "4. We are not persuaded by that argument here.\n\n"
    )
    assert numbers(text) == [1, 2, 3, 4]
    result = segment_document(text)
    labels = [s.label for s in result.segments if s.label not in (None, FOOTNOTES_LABEL)]
    assert labels == ["1", "2", "3", "4"]
    assert_gapless_partition(text, result.segments)


def test_a_quoted_higher_paragraph_number_is_rejected_and_the_rest_survives():
    # H1, in miniature: a judgment at its own paragraph 3 quoting the court
    # below's paragraph 47. The old rule accepted 47 (it was "increasing"),
    # split paragraph 3 at the citation, and then rejected 4, 5 and 6 for
    # being smaller - which is how a real document lost 86% of itself into a
    # single chunk.
    text = (
        "JUDGMENT\n\nRAO, J.\n\n"
        "1. This appeal raises a narrow question.\n\n"
        "2. The parties were heard at some length on the second day.\n\n"
        "3. The High Court reasoned as follows:\n\n"
        "47. The society having failed to produce its register, the claim fails.\n\n"
        "4. That reasoning does not survive scrutiny on this record.\n\n"
        "5. The material points the other way on every disputed head.\n\n"
        "6. The challenge is allowed and the order below is set aside.\n\n"
    )
    assert numbers(text) == [1, 2, 3, 4, 5, 6]
    result = segment_document(text)
    labels = [s.label for s in result.segments if s.label not in (None, FOOTNOTES_LABEL)]
    assert labels == ["1", "2", "3", "4", "5", "6"]
    assert_gapless_partition(text, result.segments)


def test_three_consecutive_quoted_paragraphs_are_still_a_foreign_run():
    # The `2025_10` shape at full size: the quoted block is not one stray
    # number but THREE consecutive ones (122, 123, 124 on the real
    # document), which reads as numbering in its own right until you count
    # what entering and leaving it costs. Written with literals rather than
    # against RESTART_COST, so the constant cannot satisfy the test by
    # moving.
    text = (
        "JUDGMENT\n\nRAO, J.\n\n"
        "1. Notice was issued and the matter came to be admitted.\n\n"
        "2. The decree under challenge dealt with the claim under three heads:\n\n"
        "122. The first head was refused for want of any proof at all.\n\n"
        "123. The second head was allowed to the extent of one third.\n\n"
        "124. Interest was declined on both of those heads.\n\n"
        "3. We take up the correctness of that division first.\n\n"
        "4. The challenge succeeds on the second head alone.\n\n"
    )
    assert numbers(text) == [1, 2, 3, 4]


def test_a_paragraph_number_that_skips_two_is_still_this_document_s_own():
    # The other edge, in literals: real numbering skips (a mis-scanned
    # marker, a paragraph that never got one), and the measured distribution
    # over the staged corpus puts +2 nineteen times and +3 five times before
    # anything above that turns out to be a citation. A rule tight enough to
    # reject +3 would drop real paragraphs.
    text = (
        "JUDGMENT\n\n1. First.\n\n2. Second.\n\n4. Fourth, three never got a marker.\n\n"
        "7. Seventh, after two more the scan lost.\n\n8. Eighth.\n\n"
    )
    assert numbers(text) == [1, 2, 4, 7, 8]


def test_the_two_measured_constants_hold_their_measured_values():
    # Both are set from a measurement, not chosen, and every OTHER test here
    # reads them from the module - which makes those tests tautological
    # under a mutation that moves them. This is the one place their values
    # are pinned, so moving either forces a re-measurement rather than a
    # green suite. MAX_PARA_STEP: raw-candidate increments over the 15
    # staged documents are +1 x681, +2 x19, +3 x5, +4 x1, then citations.
    # RESTART_COST: the measured citation runs are 1, 1, 2, 3 and 3
    # candidates long and the measured genuine restarts are 7 and 21, and a
    # run must beat 2 * RESTART_COST to be accepted.
    assert (MAX_PARA_STEP, RESTART_COST) == (3, 2)


def test_a_quoted_statute_section_number_does_not_become_a_boundary():
    # The `2014_11` shape: a judgment at its own paragraph 6 setting out two
    # consecutive sections of a code (`178.`, `179.`) and then carrying on.
    # Two consecutive numbers are still a foreign run, not this document's
    # numbering.
    text = (
        "JUDGMENT\n\nRAO, J.\n\n"
        "1. Notice was issued and both matters were admitted.\n\n"
        "2. The narrow point is which forum could entertain the claim.\n\n"
        "3. The material provisions are extracted next.\n\n"
        "4. They read thus:\n\n"
        "178. Constitution of the appellate tribunal.- The tribunal shall sit\n"
        "in benches of two members nominated by its chairperson.\n\n"
        "179. Sittings and quorum.- A bench so constituted may take up any\n"
        "reference made to it under the preceding provision.\n\n"
        "5. Read together, those provisions answer the objection.\n\n"
        "6. Both matters are disposed of without any order on costs.\n\n"
    )
    assert numbers(text) == [1, 2, 3, 4, 5, 6]
    assert_gapless_partition(text, segment_document(text).segments)


def test_a_number_that_is_the_tail_of_a_wrapped_citation_is_not_a_boundary():
    # The `2010_1` shape, and the subtlest of the four: the number belongs to
    # "Article 335" on the line above and only LOOKS line-initial because the
    # line wrapped there. extract.py measured this exact shape on this exact
    # corpus before segment.py was written.
    text = (
        "JUDGMENT\n\nRAO, J.\n\n"
        "1. Notice was issued and the matter came to be admitted.\n\n"
        "2. The appellant relies on the guarantee contained in Article\n"
        "335. The provision speaks of the claims of all sections in the\n"
        "making of appointments to services and posts.\n\n"
        "3. That guarantee does not carry the appellant as far as claimed.\n\n"
        "4. The matter is disposed of.\n\n"
    )
    assert numbers(text) == [1, 2, 3, 4]
    assert_gapless_partition(text, segment_document(text).segments)


def test_a_long_quoted_numbered_list_does_not_swallow_the_paragraphs_after_it():
    # The `2010_3` shape: a post-mortem report's injury list, twenty-one
    # numbered items quoted inside the judgment's own paragraph 6. A run that
    # long IS this document's numbering for as long as it lasts, so its items
    # are accepted as boundaries (they are separate lines, and the packer
    # merges them straight back into one chunk) - what must NOT happen is the
    # thing the strictly-increasing rule did: leave the counter parked at 21
    # so that the judgment's own paragraphs 7, 8 and 9 are all rejected and
    # the rest of the document collapses into a single segment.
    listed = "".join(f"{i}. Incised wound of the {i}th described dimension.\n\n" for i in range(1, 22))
    text = (
        "JUDGMENT\n\nRAO, J.\n\n"
        "4. The court of first instance acted on the case as presented.\n\n"
        "5. That view was carried forward on the same footing in appeal.\n\n"
        "6. The post-mortem report records the following injuries:\n\n"
        + listed
        + "7. The medical evidence is therefore consistent with the charge.\n\n"
        "8. Those concurrent findings therefore stand undisturbed.\n\n"
        "9. The matter is disposed of.\n\n"
    )
    found = numbers(text)
    # the judgment's own tail survives - the defect this test exists for
    assert found[-3:] == [7, 8, 9]
    assert_gapless_partition(text, segment_document(text).segments)


def test_a_far_forward_jump_is_never_a_next_paragraph():
    # MAX_PARA_STEP, at its own edge and one past it. A real next paragraph
    # is `last + 1` and occasionally skips one or two (a mis-scanned marker);
    # `last + 169` is a citation. Both directions, so the constant is pinned
    # rather than merely present.
    step_ok = f"JUDGMENT\n\n1. First.\n\n{1 + MAX_PARA_STEP}. Still this document's own numbering.\n\n"
    step_over = f"JUDGMENT\n\n1. First.\n\n{2 + MAX_PARA_STEP}. A number from somewhere else.\n\n"
    assert numbers(step_ok) == [1, 1 + MAX_PARA_STEP]
    assert numbers(step_over) == [1]


def test_no_numbered_paragraphs_at_all_still_yields_one_segment():
    text = "ORDER\n\nThe appeal is dismissed with costs.\n"
    result = segment_document(text)
    assert result.tier == TIER_PACKING
    assert len(result.segments) == 1
    assert result.segments[0].label is None
    assert_gapless_partition(text, result.segments)


def test_a_repeated_paragraph_number_is_rejected_as_not_a_new_paragraph():
    text = "JUDGMENT\n\n1. First.\n\n1. Repeated marker, not a new paragraph.\n\n"
    assert numbers(text) == [1]


def test_a_number_that_goes_backward_after_the_first_match_is_rejected():
    # Still true, and still worth pinning: an isolated smaller number in the
    # middle of the judgment's own run is a quotation, not paragraph 1 again.
    text = (
        "JUDGMENT\n\n1. First.\n\n2. Second.\n\n"
        "1. A quoted marker, not a new paragraph.\n\n3. Third.\n\n4. Fourth.\n\n"
    )
    assert numbers(text) == [1, 2, 3, 4]


def test_a_second_opinion_restarting_its_numbering_is_picked_up_not_swallowed():
    # The genuine restart case the old rule documented as an accepted cost
    # ("that tail is swallowed into one large trailing segment"): a
    # concurring opinion numbering itself from 1 again. A restart long enough
    # to pay for itself is this document's numbering too.
    first = "".join(f"{i}. The majority's reasoning at step {i}.\n\n" for i in range(1, 9))
    second = "".join(f"{i}. The concurrence's reasoning at step {i}.\n\n" for i in range(1, 8))
    text = "JUDGMENT\n\nRAO, J.\n\n" + first + "BANUMATHI, J.\n\n" + second
    found = numbers(text)
    assert found == list(range(1, 9)) + list(range(1, 8))
    assert_gapless_partition(text, segment_document(text).segments)


def test_the_earliest_position_of_a_repeated_number_anchors_the_chain():
    # Determinism has to name WHICH occurrence wins, not only that one
    # does: a page-header artifact can repeat a paragraph number hundreds of
    # characters away, and the chain that continues from the LATER one moves
    # every boundary after it. Earliest-wins, asserted on offsets because
    # the numbers are identical either way.
    text = "JUDGMENT\n\n1. First.\n\n2. Second.\n\n2. A repeat of the marker.\n\n3. Third.\n\n"
    offsets = [offset for offset, _n in paragraph_starts(text)]
    assert offsets == [text.index("1. "), text.index("2. "), text.index("3. ")]


def test_a_deeply_indented_number_is_not_a_paragraph_start():
    # _PARA_START allows up to three leading spaces because that is what a
    # wrapped or slightly-inset paragraph carries; a block quotation is
    # indented further, and its numbering is not this document's. The
    # indented number here is one the document's own run does NOT contain,
    # so widening the allowance changes the accepted SET rather than only
    # which position a duplicate number resolves to.
    text = (
        "JUDGMENT\n\n1. First.\n\n2. Second.\n\n"
        "        3. Deep inside an indented block quotation.\n\n"
        "4. The real next paragraph.\n\n"
    )
    assert numbers(text) == [1, 2, 4]
    assert " " * 8 + "3." in text  # the indented line really is in the fixture


def test_a_short_document_whose_only_run_is_two_paragraphs_still_segments():
    # The restart cost must never make a SHORT document unsegmentable: two
    # paragraphs are two paragraphs, not a run too small to believe.
    text = "ORDER\n\n1. Notice was issued.\n\n2. The matter is disposed of.\n\n"
    assert numbers(text) == [1, 2]


# --------------------------------------------------------------------------
# Footnote tail.
# --------------------------------------------------------------------------


def test_footnote_tail_is_split_off_as_its_own_labelled_segment():
    text = judgment(3) + "\n[FOOTNOTES]\n1. (2019) 3 SCC 100.\n2. AIR 1985 SC 12.\n"
    body, footnote_start = _split_footnote_tail(text)
    assert footnote_start is not None
    assert text[footnote_start:].startswith("[FOOTNOTES]")
    assert body + text[footnote_start:] == text

    result = segment_document(text)
    assert result.segments[-1].label == FOOTNOTES_LABEL
    assert_gapless_partition(text, result.segments)


def test_no_footnote_marker_means_no_footnote_segment():
    text = judgment(3)
    body, footnote_start = _split_footnote_tail(text)
    assert footnote_start is None
    assert body == text
    result = segment_document(text)
    assert all(s.label != FOOTNOTES_LABEL for s in result.segments)


def test_footnote_marker_does_not_pollute_paragraph_numbering():
    # The footnote block restarts at "1." - without the split, that would
    # look like a backward (rejected) number after the body's higher-numbered
    # paragraphs, or worse, an accepted one if the body itself stayed low.
    text = judgment(2) + "\n[FOOTNOTES]\n1. Some Author, Some Book (2001).\n"
    result = segment_document(text)
    para_labels = [s.label for s in result.segments if s.label not in (None, FOOTNOTES_LABEL)]
    assert para_labels == ["1", "2"]
    assert result.segments[-1].label == FOOTNOTES_LABEL


# --------------------------------------------------------------------------
# Footnote resolution: parsing the tail, and rendering what a span cites.
# --------------------------------------------------------------------------


def test_the_tail_parses_in_the_shape_extract_py_actually_writes():
    # extract.py hoists the footnote LINES verbatim, and a typeset footnote
    # opens with its own number - `1 (2017) 9 SCC 499`, with or without the
    # period. This is the only shape in the built corpus.
    text = judgment(3) + "\n[FOOTNOTES]\n1 (2019) 3 SCC 100.\n2. AIR 1985 SC 12.\n"
    assert parse_footnotes(text) == {"1": "(2019) 3 SCC 100.", "2": "AIR 1985 SC 12."}


def test_the_tail_also_parses_the_bracketed_shape():
    text = judgment(3) + "\n[FOOTNOTES]\n[^3]: Salomon v A Salomon [1897] AC 22.\n"
    assert parse_footnotes(text) == {"3": "Salomon v A Salomon [1897] AC 22."}


def test_a_document_with_no_tail_parses_to_nothing():
    assert parse_footnotes(judgment(3)) == {}


def test_a_key_defined_twice_keeps_its_first_definition():
    # Two reprint pages each restarting their notes at 1; the earliest
    # citation in the document meant the first one.
    text = judgment(3) + "\n[FOOTNOTES]\n1 (2019) 3 SCC 100.\n1 AIR 1985 SC 12.\n"
    assert parse_footnotes(text) == {"1": "(2019) 3 SCC 100."}


def test_both_marker_shapes_are_found_and_ordered_by_position():
    notes = {"1": "first", "2": "second"}
    text = "As held in Salomon 2 and earlier in another judgment.[^1]"
    assert cited_footnotes(text, notes) == ["2", "1"]


def test_a_marker_no_note_answers_to_is_not_cited():
    assert cited_footnotes("As held in Salomon 9.[^7]", {"1": "first"}) == []


@pytest.mark.parametrize(
    "text",
    [
        "The question turns on Section 3 of the Act.",
        "Article 3 of the Constitution is engaged.",
        "The order was passed on 3 May 2018 by the Board.",
        "Recorded as Bruise 3 x 1 cm over the arm.",
        "Reported at (2001) 1 SCC 3, the Court held otherwise.",
    ],
)
def test_a_number_that_counts_something_is_not_read_as_a_marker(text):
    # The flattened-superscript rule has to live next to every other reason
    # a judgment puts a number after a word. A false positive appends a real
    # citation from this same judgment to a span that did not cite it.
    assert cited_footnotes(text, {"3": "Salomon v A Salomon [1897] AC 22."}) == []


def test_resolution_appends_and_only_appends():
    notes = {"3": "Salomon v A Salomon [1897] AC 22."}
    text = "The appellant relies on the rule in Salomon.[^3]"
    out = resolve_footnotes(text, notes)
    assert out.startswith(text)
    assert out == (
        f"{text}\n\n{FOOTNOTES_CITED_HEADING}\n"
        "[^3]: Salomon v A Salomon [1897] AC 22.\n"
    )


def test_a_span_that_cites_nothing_is_returned_unchanged():
    notes = {"3": "Salomon v A Salomon [1897] AC 22."}
    text = "That contention must be rejected."
    assert resolve_footnotes(text, notes) is text


# --------------------------------------------------------------------------
# ToC tier: validated, both directions.
# --------------------------------------------------------------------------


def _toc_judgment() -> str:
    return (
        "JUDGMENT\n\n"
        "A. Factual Matrix\n\n"
        "1. The appellant was convicted by the trial court.\n\n"
        "2. He appealed to the High Court, which affirmed the conviction.\n\n"
        "B. Issues For Determination\n\n"
        "3. Whether the conviction can stand on this record.\n\n"
        "C. Analysis\n\n"
        "4. We examine the evidence adduced at trial.\n\n"
        "5. The chain of circumstances is broken at a material link.\n\n"
    )


def test_a_validated_toc_is_used_and_reported_as_such():
    text = _toc_judgment()
    result = segment_document(text)
    assert result.tier == TIER_TOC
    assert result.why == WHY_TOC
    assert result.degradation is None
    labels = [s.label for s in result.segments if s.label]
    assert dict.fromkeys(labels) == dict.fromkeys(
        ["Factual Matrix", "Issues For Determination", "Analysis"]
    )
    assert_gapless_partition(text, result.segments)


def test_toc_sections_are_subdivided_at_this_document_s_own_paragraph_starts():
    # M2: the ToC tier decides BOUNDARIES, not chunk size. Its sections are
    # document-scale spans, so they are cut at the packing tier's own
    # paragraph starts before they leave this module - every heading is
    # still a boundary and still a label, and every paragraph inside the
    # section is a segment the packer can bin.
    text = _toc_judgment()
    result = segment_document(text)
    assert result.tier == TIER_TOC
    boundaries = {seg.start for seg in result.segments}
    # the packing tier's own offsets, read from the packing tier - NOT from
    # segment_document, which on this text returns the ToC tier whose labels
    # are headings, so paragraph_offsets would hand back heading positions
    # and the assertion below would pass without testing anything.
    para = paragraph_offsets(_packing_tier(text))
    assert len(para) == 5
    for offset in para:
        assert offset in boundaries
    for offset, _letter, _heading in toc_candidates(text):
        assert offset in boundaries


def test_no_tier_may_emit_a_segment_the_packing_tier_would_have_split():
    # The invariant the subdivision buys, stated directly: whichever tier
    # wins, its segment set is a REFINEMENT of the packing tier's, so it can
    # never hand chunks.py a span that packing would have broken up - which
    # is what "tier precedence is about boundaries, never about the band"
    # means operationally.
    long_section = "\n\n".join(para(i, words=200) for i in range(1, 10))
    text = (
        "JUDGMENT\n\nA. Factual Matrix\n\n"
        + long_section
        + "\n\nB. Issues For Determination\n\n"
        + "\n\n".join(para(i, words=200) for i in range(10, 14))
        + "\n\nC. Analysis\n\n"
        + "\n\n".join(para(i, words=200) for i in range(14, 18))
        + "\n"
    )
    toc = segment_document(text)
    assert toc.tier == TIER_TOC
    packing_only = _packing_tier(text)
    toc_bounds = {seg.start for seg in toc.segments} | {seg.end for seg in toc.segments}
    for seg in _normalize_segments(text, packing_only):
        assert seg.start in toc_bounds and seg.end in toc_bounds
    assert_gapless_partition(text, toc.segments)


def test_toc_candidates_are_read_in_document_order():
    text = _toc_judgment()
    candidates = toc_candidates(text)
    assert [c[1] for c in candidates] == ["A", "B", "C"]


def test_fewer_than_the_minimum_headings_falls_through_to_packing():
    text = (
        "JUDGMENT\n\nA. Facts\n\n1. Something happened.\n\nB. Analysis\n\n2. We decide it.\n\n"
    )
    assert len(toc_candidates(text)) < MIN_TOC_HEADINGS
    result = segment_document(text)
    assert result.tier == TIER_PACKING


def test_non_consecutive_letters_are_rejected():
    text = (
        "JUDGMENT\n\n"
        "A. Facts\n\n1. Para one.\n\n"
        "C. Skips B\n\n2. Para two.\n\n"
        "D. Continues\n\n3. Para three.\n\n"
    )
    result = segment_document(text)
    assert result.tier == TIER_PACKING


def test_a_hollow_section_with_no_paragraph_in_it_rejects_the_whole_toc():
    # B has no numbered paragraph inside its span - not validated against the
    # document, so the WHOLE candidate set is rejected, not just section B.
    text = (
        "JUDGMENT\n\n"
        "A. Facts\n\n1. Something happened at trial.\n\n"
        "B. Issues\n\n"
        "C. Analysis\n\n2. We decide the matter.\n\n"
    )
    packing_segments = segment_document(text, roles_backend=BACKEND_NONE)
    # Direct check of the validator itself, not only the tier it falls back
    # to: the hollow B section must be why, not some other candidate defect.
    assert _toc_segments(text, packing_segments.segments) is None
    assert packing_segments.tier == TIER_PACKING


def test_a_heading_needs_words_not_just_a_letter_and_a_capital():
    # The length floor in _TOC_HEADING is what separates a real heading from
    # extract.py's own margin letter, and it is the only thing that does: at
    # zero it matches "A. B", which is the shape of a garbled OCR margin
    # column, and three of those in a row would claim the whole document for
    # the ToC tier.
    text = (
        "JUDGMENT\n\n"
        "A. B\n\n1. Something happened at trial.\n\n"
        "B. C\n\n2. The High Court affirmed it.\n\n"
        "C. D\n\n3. We take a different view.\n\n"
    )
    assert toc_candidates(text) == []
    assert segment_document(text).tier == TIER_PACKING


def test_repeated_letter_is_rejected():
    text = (
        "JUDGMENT\n\n"
        "A. Facts\n\n1. Para one.\n\n"
        "A. Repeated\n\n2. Para two.\n\n"
        "B. Analysis\n\n3. Para three.\n\n"
    )
    result = segment_document(text)
    assert result.tier == TIER_PACKING


# --------------------------------------------------------------------------
# Roles tier: available, unavailable, and every failure kind degrades.
# --------------------------------------------------------------------------


def _fake_spawn(reply):
    import json

    def run(args, *, input, capture_output, text, timeout):
        class R:
            returncode = 0
            stdout = json.dumps(reply)
            stderr = ""

        return R()

    return run


def _fake_spawn_raises(exc):
    def run(*a, **k):
        raise exc

    return run


def test_roles_backend_none_degrades_with_that_exact_reason():
    result = segment_document(judgment(3), roles_backend=BACKEND_NONE)
    assert result.tier == TIER_PACKING
    assert result.why == WHY_PACKING
    assert result.degradation == {"from": "roles", "reason": "roles_backend_none"}


def test_roles_backend_available_and_producing_spans_is_used():
    text = judgment(3)
    spawn = _fake_spawn({"spans": [[0, 20, "FAC"], [20, 40, "ISSUE"]]})
    result = segment_document(text, roles_backend=BACKEND_SUBPROCESS, roles_spawn=spawn)
    assert result.tier == TIER_ROLES
    assert result.why == WHY_ROLES
    assert result.degradation is None
    assert_gapless_partition(text, result.segments)


def test_roles_backend_returning_no_spans_degrades_distinctly_from_a_crash():
    spawn = _fake_spawn({"spans": []})
    result = segment_document(judgment(3), roles_backend=BACKEND_SUBPROCESS, roles_spawn=spawn)
    assert result.tier == TIER_PACKING
    assert result.degradation == {"from": "roles", "reason": "no_role_spans"}


def test_roles_bridge_crash_degrades_with_the_bridge_error_kind_and_message():
    spawn = _fake_spawn_raises(FileNotFoundError("no interpreter"))
    result = segment_document(
        judgment(3), roles_backend=BACKEND_SUBPROCESS, roles_python_bin="/nope", roles_spawn=spawn
    )
    assert result.tier == TIER_PACKING
    assert result.degradation["from"] == "roles"
    assert result.degradation["reason"].startswith("spawn_failed:")


def test_roles_bridge_timeout_degrades_and_never_raises_out_of_segment_document():
    spawn = _fake_spawn_raises(subprocess.TimeoutExpired(cmd="worker", timeout=5))
    result = segment_document(judgment(3), roles_backend=BACKEND_SUBPROCESS, roles_spawn=spawn)
    assert result.tier == TIER_PACKING
    assert result.degradation["reason"].startswith("timeout:")


def test_a_missing_roles_backend_never_makes_the_whole_document_unusable():
    # The brief's own framing: --roles-backend none must leave the pipeline
    # fully functional on packing alone. This is that property, asserted.
    text = judgment(5)
    result = segment_document(text, roles_backend=BACKEND_NONE)
    assert result.segments
    assert_gapless_partition(text, result.segments)


def test_a_reversed_role_span_degrades_this_document_instead_of_ending_the_run():
    # M3: the "never fatally" contract, at the one place it had a hole. A
    # backend returning end < start used to reach Segment(), whose plain
    # ValueError is not a RolesBridgeError - so it travelled out of
    # segment_document, out of chunk_documents, and ended the whole pass
    # mid-loop after earlier documents had already been rewritten.
    text = judgment(3)
    spawn = _fake_spawn({"spans": [[10, 3, "FAC"]]})
    result = segment_document(text, roles_backend=BACKEND_SUBPROCESS, roles_spawn=spawn)
    assert result.tier == TIER_PACKING
    assert result.degradation["from"] == "roles"
    assert result.degradation["reason"].startswith("bad_output:")
    assert_gapless_partition(text, result.segments)


def test_a_negative_role_span_start_degrades_the_same_way():
    text = judgment(3)
    spawn = _fake_spawn({"spans": [[-5, 20, "FAC"]]})
    result = segment_document(text, roles_backend=BACKEND_SUBPROCESS, roles_spawn=spawn)
    assert result.tier == TIER_PACKING
    assert result.degradation["reason"].startswith("bad_output:")


def test_a_role_span_running_past_the_document_is_clipped_not_refused():
    # The other direction, and deliberately different: a model overshooting
    # the end of the last span is repairable, so it is repaired (clipped by
    # _normalize_segments) rather than costing the document its tier.
    text = judgment(3)
    spawn = _fake_spawn({"spans": [[0, 10_000, "FAC"]]})
    result = segment_document(text, roles_backend=BACKEND_SUBPROCESS, roles_spawn=spawn)
    assert result.tier == TIER_ROLES
    assert result.segments[-1].end == len(text)
    assert_gapless_partition(text, result.segments)


def test_toc_takes_priority_over_a_configured_and_available_roles_backend():
    # Tier priority is toc, then roles, then packing - a validated ToC never
    # even asks the roles bridge.
    def boom(*a, **k):
        raise AssertionError("roles bridge must not be consulted when ToC validates")

    text = _toc_judgment()
    result = segment_document(text, roles_backend=BACKEND_SUBPROCESS, roles_spawn=boom)
    assert result.tier == TIER_TOC


# --------------------------------------------------------------------------
# Normalization: gapless, ordered, overlap-clipped - regardless of tier.
# --------------------------------------------------------------------------


def test_normalize_fills_a_gap_between_segments():
    text = "0123456789"
    segments = [Segment(0, 3, "a"), Segment(6, 10, "b")]
    normalized = _normalize_segments(text, segments)
    assert_gapless_partition(text, normalized)
    assert [(s.start, s.end, s.label) for s in normalized] == [
        (0, 3, "a"),
        (3, 6, None),
        (6, 10, "b"),
    ]


def test_normalize_clips_an_overlap_first_segment_wins():
    text = "0123456789"
    segments = [Segment(0, 6, "a"), Segment(4, 10, "b")]
    normalized = _normalize_segments(text, segments)
    assert_gapless_partition(text, normalized)
    assert [(s.start, s.end, s.label) for s in normalized] == [(0, 6, "a"), (6, 10, "b")]


def test_normalize_sorts_by_start_not_by_end_a_nested_overlap_the_two_disagree_on():
    # A segment fully CONTAINED in another sorts differently depending on
    # which end of (start, end) is primary: by start, the containing
    # segment ("a", 0-10) comes first and swallows "b" (2-5) entirely; by
    # end, "b" (ending at 5) would sort BEFORE "a" (ending at 10) and
    # survive as its own segment instead - a real behavioural difference a
    # gapless-reconstruction check alone cannot see, only the LABELS can.
    text = "0123456789"
    segments = [Segment(0, 10, "a"), Segment(2, 5, "b")]
    normalized = _normalize_segments(text, segments)
    assert_gapless_partition(text, normalized)
    assert [(s.start, s.end, s.label) for s in normalized] == [(0, 10, "a")]


def test_normalize_handles_unsorted_input():
    text = "0123456789"
    segments = [Segment(6, 10, "b"), Segment(0, 3, "a")]
    normalized = _normalize_segments(text, segments)
    assert_gapless_partition(text, normalized)


def test_normalize_of_empty_segments_over_empty_text_is_empty():
    assert _normalize_segments("", []) == ()


def test_normalize_covers_leading_and_trailing_gaps():
    text = "0123456789"
    normalized = _normalize_segments(text, [Segment(3, 7, "mid")])
    assert_gapless_partition(text, normalized)
    assert normalized[0] == Segment(0, 3, None)
    assert normalized[-1] == Segment(7, 10, None)


def test_normalize_clips_a_span_running_past_the_end_of_the_document():
    # The untrusted-input half of the clip. Without it the segment's own
    # `end` names bytes the document does not have, and chunks.py turns that
    # end into the chunk's native_id AND its content-derived seed_id - an id
    # for a byte range nothing can ever re-read.
    text = "0123456789"
    normalized = _normalize_segments(text, [Segment(0, 10_000, "FAC")])
    assert_gapless_partition(text, normalized)
    assert normalized == (Segment(0, 10, "FAC"),)


def test_normalize_drops_a_span_that_starts_past_the_end_of_the_document():
    text = "0123456789"
    normalized = _normalize_segments(text, [Segment(0, 4, "a"), Segment(9000, 9500, "FAC")])
    assert_gapless_partition(text, normalized)
    assert [s.label for s in normalized] == ["a", None]
    assert normalized[-1].end == len(text)


# --------------------------------------------------------------------------
# Subdivision: a tier decides boundaries, never the token band.
# --------------------------------------------------------------------------


def test_subdivide_cuts_only_strictly_inside_a_segment_and_keeps_the_label():
    segments = (Segment(0, 10, "A"), Segment(10, 20, "B"))
    out = _subdivide(segments, [0, 4, 7, 10, 15, 20])
    assert [(s.start, s.end, s.label) for s in out] == [
        (0, 4, "A"), (4, 7, "A"), (7, 10, "A"),
        (10, 15, "B"), (15, 20, "B"),
    ]


def test_subdivide_with_no_offsets_inside_returns_the_segments_unchanged():
    segments = (Segment(0, 10, "A"), Segment(10, 20, "B"))
    assert _subdivide(segments, [0, 10, 20]) == segments
    assert _subdivide(segments, []) == segments


def test_subdivide_conserves_the_span_it_was_given():
    segments = (Segment(0, 10, "A"), Segment(10, 25, None))
    out = _subdivide(segments, [3, 3, 11, 24])
    assert out[0].start == 0 and out[-1].end == 25
    cursor = 0
    for seg in out:
        assert seg.start == cursor
        cursor = seg.end


def test_roles_spans_are_subdivided_at_paragraph_starts_too():
    # Not a ToC-only rule: a rhetorical role covers many paragraphs, so the
    # roles tier gets the same treatment - it says where a role begins, the
    # packing tier's paragraphs say where a chunk may.
    text = judgment(6, words=60)
    spawn = _fake_spawn({"spans": [[0, len(text) // 2, "FAC"], [len(text) // 2, len(text), "ANALYSIS"]]})
    result = segment_document(text, roles_backend=BACKEND_SUBPROCESS, roles_spawn=spawn)
    assert result.tier == TIER_ROLES
    assert_gapless_partition(text, result.segments)
    boundaries = {seg.start for seg in result.segments}
    for offset in paragraph_offsets(_packing_tier(text)):
        assert offset in boundaries
    assert set(s.label for s in result.segments) <= {"FAC", "ANALYSIS", None}


# --------------------------------------------------------------------------
# Segment itself.
# --------------------------------------------------------------------------


def test_segment_rejects_end_before_start():
    with pytest.raises(ValueError):
        Segment(10, 5, None)


def test_segment_allows_a_zero_length_span():
    Segment(5, 5, None)  # must not raise


def test_a_result_cannot_claim_a_tier_the_contract_does_not_name():
    from tuned.data import segment as seg

    with pytest.raises(ValueError, match="unknown tier"):
        seg.SegmentationResult(tier="ocr", why="fallback", segments=())


def test_each_named_tier_constructs():
    from tuned.data import segment as seg

    # Each tier is built the one way its degradation contract allows: packing
    # always carries the dict, the other two never do.
    for tier in seg.TIERS:
        degradation = {"from": "roles", "reason": "under test"}
        result = seg.SegmentationResult(
            tier=tier,
            why="under test",
            segments=(),
            degradation=degradation if tier == seg.TIER_PACKING else None,
        )
        assert result.tier == tier


def test_the_packing_tier_must_say_why_it_degraded():
    # chunks.py copies this dict straight into every chunk's meta_json, and it
    # is the only surviving record of why the roles tier did not carry the
    # document. The field defaults to None, so a packing return path that
    # forgets it degrades silently - the reader sees a fallback with no cause.
    from tuned.data import segment as seg

    with pytest.raises(ValueError, match="degradation"):
        seg.SegmentationResult(tier=seg.TIER_PACKING, why=seg.WHY_PACKING, segments=())


def test_a_tier_that_did_not_degrade_cannot_claim_a_degradation():
    # The inverse error, and the one that corrupts counts rather than losing
    # them: a toc/roles row carrying a degradation reports a fallback that
    # never happened.
    from tuned.data import segment as seg

    for tier, why in ((seg.TIER_TOC, seg.WHY_TOC), (seg.TIER_ROLES, seg.WHY_ROLES)):
        with pytest.raises(ValueError, match="degradation"):
            seg.SegmentationResult(
                tier=tier,
                why=why,
                segments=(),
                degradation={"from": "roles", "reason": "no_role_spans"},
            )
