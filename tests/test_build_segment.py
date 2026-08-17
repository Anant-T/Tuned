"""segment.py - whole judgment text -> tier-selected, gapless segments.

Fixtures are structural shapes with invented prose (paragraph numbering,
lettered headings, footnote blocks) - no verbatim S.C.R./eval text anywhere.
"""

import subprocess

import pytest

from tuned.data.roles_infer import BACKEND_NONE, BACKEND_SUBPROCESS, RolesBridgeError
from tuned.data.segment import (
    FOOTNOTES_LABEL,
    MIN_TOC_HEADINGS,
    TIER_PACKING,
    TIER_ROLES,
    TIER_TOC,
    WHY_PACKING,
    WHY_ROLES,
    WHY_TOC,
    Segment,
    _normalize_segments,
    _split_footnote_tail,
    _toc_segments,
    monotonic_paragraph_starts,
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
# Monotonic paragraph detection - the workhorse's own correctness.
# --------------------------------------------------------------------------


def test_consecutive_numbered_paragraphs_are_all_accepted():
    text = judgment(5)
    starts = monotonic_paragraph_starts(text)
    assert [n for _offset, n in starts] == [1, 2, 3, 4, 5]


def test_the_close_paren_marker_form_is_also_recognised():
    # extract.py's own numbered-paragraph signal accepts both "1." and "1)" -
    # this module's marker must too, not only the period form every other
    # fixture in this file happens to use.
    text = "JUDGMENT\n\n1) First paragraph text here.\n\n2) Second paragraph text here.\n\n"
    starts = monotonic_paragraph_starts(text)
    assert [n for _offset, n in starts] == [1, 2]


def test_a_quoted_earlier_paragraph_number_is_rejected_not_a_new_boundary():
    # The failure this module's monotonic rule exists for: a quotation from
    # an earlier report carries its OWN "15." that must not fracture the
    # paragraph doing the quoting.
    text = (
        "JUDGMENT\n\nRAO, J.\n\n"
        "1. This appeal raises a narrow question.\n\n"
        '2. In an earlier case this Court observed: "15. The onus lies on '
        'the prosecution to establish guilt beyond reasonable doubt."\n\n'
        "3. We are not persuaded by that argument here.\n\n"
    )
    starts = monotonic_paragraph_starts(text)
    assert [n for _offset, n in starts] == [1, 2, 3]
    result = segment_document(text)
    labels = [s.label for s in result.segments if s.label not in (None, FOOTNOTES_LABEL)]
    assert labels == ["1", "2", "3"]
    assert_gapless_partition(text, result.segments)


def test_a_number_that_goes_backward_after_the_first_match_is_rejected():
    text = "JUDGMENT\n\n1. First.\n\n5. Should be accepted (increasing).\n\n2. Should be rejected.\n\n"
    starts = monotonic_paragraph_starts(text)
    assert [n for _offset, n in starts] == [1, 5]


def test_no_numbered_paragraphs_at_all_still_yields_one_segment():
    text = "ORDER\n\nThe appeal is dismissed with costs.\n"
    result = segment_document(text)
    assert result.tier == TIER_PACKING
    assert len(result.segments) == 1
    assert result.segments[0].label is None
    assert_gapless_partition(text, result.segments)


def test_a_repeated_paragraph_number_is_rejected_as_not_strictly_greater():
    text = "JUDGMENT\n\n1. First.\n\n1. Repeated marker, not a new paragraph.\n\n"
    starts = monotonic_paragraph_starts(text)
    assert [n for _offset, n in starts] == [1]


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
    assert labels == ["Factual Matrix", "Issues For Determination", "Analysis"]
    assert_gapless_partition(text, result.segments)


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


# --------------------------------------------------------------------------
# Segment itself.
# --------------------------------------------------------------------------


def test_segment_rejects_end_before_start():
    with pytest.raises(ValueError):
        Segment(10, 5, None)


def test_segment_allows_a_zero_length_span():
    Segment(5, 5, None)  # must not raise
