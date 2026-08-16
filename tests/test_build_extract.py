"""extract.py - S.C.R. PDF -> clean judgment text.

Nothing here opens a PDF: the reader is an injected seam (a callable
returning one string per page), so every cleanup, boundary and quarantine
property is driven from text fixtures that model what pymupdf4llm emits.

The fixtures below are shaped on P0 CHECK 3's reading of these files: a
typeset S.C.R. reprint whose first pages are the publisher's editorial
headnote (issue summary, `HELD:` points, a Case Law Reference table) with
marginal print-alignment letters down the left margin, followed by the
court's own judgment.
"""

import re

import pytest

from tuned.data.extract import (
    FOOTNOTE_HEADING,
    FOOTNOTE_MAX_SHARE,
    MAX_STRIP_FRACTION,
    MIN_BODY_CHARS,
    MIN_DOC_CHARS,
    MIN_LATIN_RATIO,
    MIN_STOPWORD_RATE,
    PdfStructure,
    Q_MOJIBAKE_FONT,
    Q_SCANNED_ERA,
    Q_BODY_TOO_SHORT,
    Q_HEADNOTE_RESIDUE,
    Q_LOW_TEXT_QUALITY,
    Q_NO_JUDGMENT_START,
    Q_NO_TEXT,
    Q_STRIP_TOO_LARGE,
    RESIDUE_WINDOW,
    SEAM_ENUM_WINDOW,
    RUNNING_DIGIT_BLIND_CHARS,
    RUNNING_MAX_CHARS,
    _despace,
    _despace_pairs,
    _ENUM_ITEM,
    _MD_TABLE_ROW,
    _SIGNATURES,
    clean_pages,
    demote_markdown,
    extract_text,
    find_judgment_start,
    headnote_signals,
    latin_ratio,
    page_span_from_key,
    reportable_flag,
    seam_continues_an_enumeration,
    stopword_rate,
    structural_refusal,
)

# --------------------------------------------------------------------------
# A document shaped like the ones in the bucket.
# --------------------------------------------------------------------------

RUNNING_HEADER = "[2020] 7 S.C.R. 941"

# Front matter + headnote: every line of this is the publisher's editorial
# work - a Government work under s.2(k)/s.17(d) of the Copyright Act, outside
# the s.52(1)(q)(iv) judgment exemption - and none of it may reach the corpus.
HEADNOTE = """\
KALYANI SHARMA
v.
STATE OF MAHARASHTRA AND OTHERS
(Civil Appeal No. 3221 of 2018)
15 MARCH 2020
[R.F. NARIMAN, NAVIN SINHA AND B.R. GAVAI, JJ.]

Service Law - Promotion - Seniority of promotees inter se - Whether the
period of ad hoc officiation counts towards seniority - Bombay Engineering
Service (Recruitment) Rules, 1978, r.7.

HELD: 1. The seniority of a promotee is reckoned from the date of regular
appointment and not from the date on which ad hoc officiation began.
2. The High Court was in error in reading rule 7 as conferring a right to
count officiation towards seniority.

Case Law Reference:
(2011) 4 SCC 707       referred to       para 12
[1996] 2 S.C.R. 1      relied on         para 18
"""

# The court's own words. The first line is the boundary marker.
BODY = """\
The Judgment of the Court was delivered by
NAVIN SINHA, J.
1. The appellant was appointed an Assistant Engineer on 4 April 1994 and was
promoted on an ad hoc basis on 11 January 1999 pending regular selection.
2. The controversy in this appeal lies in a narrow compass and turns on the
construction of rule 7 of the 1978 Rules, which the High Court read as
conferring a right that its language does not carry.
3. We have heard learned counsel for the parties at some length and have
been taken through the record of the selection process.
"""

# Anything a test asserts must NOT survive the strip.
HEADNOTE_ONLY = ("HELD:", "Case Law Reference", "Service Law - Promotion", "KALYANI SHARMA")
# ... and anything that MUST.
BODY_ONLY = ("NAVIN SINHA, J.", "an Assistant Engineer on 4 April 1994", "rule 7 of the 1978 Rules")


def _paras(first: int, chars: int) -> str:
    """Numbered judgment paragraphs from `first`, ASCENDING as real ones do."""
    out: list[str] = []
    n = first
    while sum(len(part) for part in out) < chars:
        out.append(
            f"{n}. The submission advanced on behalf of the appellant in this behalf "
            f"proceeds on a reading of the record which, on examination, the record does "
            f"not bear out, and we say so for the reasons that follow below.\n"
        )
        n += 1
    return "".join(out)


def _pad(text: str, chars: int) -> str:
    """Grow a passage past a length floor without repeating a short line."""
    return text + _paras(4, max(0, chars - len(text)))


def scr_pages(*, headnote: str = HEADNOTE, body: str = BODY, pages: int = 6) -> list[str]:
    """The fixture document, split over `pages` pages with running furniture.

    The header, the printed page number and the marginal A-H letters are on
    every page, which is what the running-line pass exists to remove.
    """
    text = headnote + "\n" + _pad(body, 2600)
    lines = text.split("\n")
    per = max(1, -(-len(lines) // pages))
    out = []
    for i in range(pages):
        chunk = lines[i * per : (i + 1) * per]
        letter = "ABCDEFGH"[i % 8]
        out.append("\n".join([RUNNING_HEADER, letter, *chunk, str(941 + i)]))
    return out


# --------------------------------------------------------------------------
# The strip. This is the whole point of the module.
# --------------------------------------------------------------------------

def test_the_headnote_is_removed_and_the_courts_own_words_are_kept():
    result = extract_text(scr_pages())

    assert result.ok, result.reason
    # POSITIONAL, not just "absent": a mutant that emits the whole document
    # fails on the first assertion, and one that emits nothing fails on the
    # second and third - "not in" alone cannot tell those two apart.
    assert result.text.startswith("The Judgment of the Court was delivered by")
    for phrase in BODY_ONLY:
        assert phrase in result.text
    for phrase in HEADNOTE_ONLY:
        assert phrase not in result.text
    assert result.headnote_chars > len(HEADNOTE) - 200
    assert result.marker == "judgment_delivered_by"
    # The signals are recorded even though the text they name is gone: this
    # is what says the document HAD a headnote to remove.
    assert "held" in result.signals


def test_a_document_whose_boundary_cannot_be_found_emits_nothing():
    # The judgment IS in this document - only the marker line is gone. A
    # partial strip would be invisible downstream, so nothing is emitted.
    pages = scr_pages(body=BODY.split("\n", 1)[1])

    result = extract_text(pages)

    assert not result.ok
    assert result.reason == Q_NO_JUDGMENT_START
    assert result.text == ""
    # The fixture really does still carry the judgment - without this the
    # test would pass on a document that had nothing to lose.
    assert "an Assistant Engineer on 4 April 1994" in "\n".join(pages)


def test_a_cut_inside_the_headnote_is_caught_by_the_residue_check():
    # The dangerous shape: a marker-looking line EARLY in the editorial
    # front matter. Cutting there leaves the Case Law Reference table and
    # the HELD points in the "judgment" - a silent partial strip, which is
    # worse than a loud failure. The boundary is found AND VERIFIED.
    headnote = HEADNOTE.replace("HELD: 1.", "ORDER\n\nHELD: 1.")
    result = extract_text(scr_pages(headnote=headnote))

    assert not result.ok
    assert result.reason == Q_HEADNOTE_RESIDUE
    assert result.text == ""
    # And the premise: the early line really is taken as a boundary, so the
    # residue check is the only thing standing between it and the corpus.
    assert find_judgment_start(headnote).marker == "judgment_heading"


def test_the_residue_check_sees_the_sectioned_front_matter_too():
    # The newer S.C.R. volumes lay the front matter out as named sections
    # rather than one HELD block, so the early-cut hazard exists in two
    # shapes and the check has to see both. Second test on the same rule
    # deliberately: it is the only thing standing between a mis-cut and a
    # corpus that looks fine.
    front = (
        "KALYANI SHARMA v. STATE OF MAHARASHTRA AND OTHERS\n"
        "ORDER\n"
        "Issue for Consideration\n"
        "Whether the period of ad hoc officiation counts towards seniority.\n"
        "List of Acts\n"
        "Bombay Engineering Service (Recruitment) Rules, 1978.\n"
    )
    result = extract_text([front + _pad(BODY, 2000)])

    assert not result.ok
    assert result.reason == Q_HEADNOTE_RESIDUE
    assert result.text == ""
    # The premise: that "ORDER" really is taken as the boundary.
    assert find_judgment_start(front).marker == "judgment_heading"


def test_a_quoted_held_costs_nothing_when_the_headnote_it_matches_was_removed():
    # WHERE THE RECALL BOUNDARY SITS, half one. The residue check compares
    # the two halves of the document by signature NAME, not by occurrence: a
    # reprint whose front matter carried `HELD:` and whose judgment later
    # quotes an earlier report's `HELD:` is EMITTED, because that name is
    # already accounted for on the removed side. This is what keeps the
    # whole-document comparison from refusing the ordinary case, and it is
    # why the rule is a comparison and not a scan.
    quotation = "HELD: 1. The seniority of a promotee is reckoned from regular appointment.\n"
    # BOUNDED, so that a mutant widening the window cannot make this fixture
    # allocate its way out of failing (the harness note from Task 10: a test
    # parametrised by the value under mutation thrashes instead of dying).
    body = _paras(1, min(RESIDUE_WINDOW, 20_000) + 800) + quotation
    result = extract_text(
        [HEADNOTE + "\nThe Judgment of the Court was delivered by\nNAVIN SINHA, J.\n" + body]
    )

    assert result.ok, result.reason
    assert "HELD:" in result.text
    # ... and the two premises that make this test about the COMPARISON: the
    # quotation really is past the window, and the removed side really did
    # carry the same signature name.
    assert result.text.index("HELD:") > RESIDUE_WINDOW
    assert "held" in headnote_signals(HEADNOTE)


def test_the_same_quotation_in_a_document_that_had_nothing_removed_is_refused():
    # WHERE THE RECALL BOUNDARY SITS, half two - and the documented cost of
    # the rule, taken deliberately. This is the fixture above with its front
    # matter deleted: the cut is at offset 0, so nothing was removed, and the
    # `HELD:` in the body has no counterpart on the removed side. From
    # outside the document that is indistinguishable from a headnote the cut
    # went over the top of (which is exactly what
    # `test_a_marker_on_line_one_strips_nothing...` below is), and the two
    # errors are not symmetric: a false quarantine costs corpus size, which
    # is countable in the reason breakdown and recoverable by re-running,
    # while a false emit puts the publisher's copyrighted summary of the
    # answer into a published dataset where nothing downstream can find it.
    quotation = "HELD: 1. The seniority of a promotee is reckoned from regular appointment.\n"
    body = _paras(1, min(RESIDUE_WINDOW, 20_000) + 800) + quotation
    result = extract_text(["The Judgment of the Court was delivered by\nNAVIN SINHA, J.\n" + body])

    assert not result.ok
    assert result.reason == Q_HEADNOTE_RESIDUE
    assert result.text == ""
    # The premise: the phrase is past the window, so this refusal is the
    # whole-document comparison and not the seam check.
    assert body.index("HELD:") > RESIDUE_WINDOW


# --------------------------------------------------------------------------
# The residue guard, in every rendering the reader can hand it.
# --------------------------------------------------------------------------
#
# THE FINDING THIS SECTION EXISTS FOR. The guard reads the publisher's
# editorial furniture, and pymupdf4llm renders that furniture in whatever
# shape the typesetting suggests: a markdown TABLE for the Case Law Reference
# grid, letter-spaced capitals for a heading, a blockquote or a bullet for an
# indented block, a `#` heading for a large font, and a mid-line `HELD:`
# wherever the printed column happened to wrap. A guard that reads only the
# canonical rendering does not refuse the others - it EMITS them, headnote
# and all, and a contaminated document reads exactly like a clean judgment,
# so nothing downstream can ever find it again.
#
# The property is therefore not "the canonical shape is caught" but "every
# rendering of the same words reaches the same verdict". The renderings below
# are the same two pieces of furniture eight ways.

_HELD = "HELD: 1. The seniority of a promotee is reckoned from regular appointment."
_REF = "Case Law Reference:"
_REF_ROW = "(2011) 4 SCC 707       referred to       para 12"

FURNITURE = {
    "canonical": f"{_HELD}\n\n{_REF}\n{_REF_ROW}\n",
    "bold": f"**{_HELD}**\n\n**{_REF}**\n{_REF_ROW}\n",
    "md_heading": f"#### {_HELD}\n\n#### {_REF}\n{_REF_ROW}\n",
    "md_table": (
        f"{_HELD}\n\n"
        f"|{_REF}|||\n|---|---|---|\n|(2011) 4 SCC 707|referred to|para 12|\n"
    ),
    "letter_spaced": (
        "H E L D :  1. The seniority of a promotee is reckoned from regular\n"
        f"appointment.\n\nC A S E   L A W   R E F E R E N C E\n{_REF_ROW}\n"
    ),
    "mid_line_wrap": (
        f"Bombay Engineering Service (Recruitment) Rules, 1978, r.7. {_HELD}\n\n"
        f"{_REF}\n{_REF_ROW}\n"
    ),
    "blockquote": f"> {_HELD}\n>\n> {_REF}\n> {_REF_ROW}\n",
    "leading_bullet": f"- {_HELD}\n\n- {_REF}\n- {_REF_ROW}\n",
    # BOTH pieces hidden at once, which is the shape that matters: one blind
    # rendering is survivable because the other signature still fires, and
    # this is the document where neither does.
    "table_and_wrap": (
        f"Bombay Engineering Service (Recruitment) Rules, 1978, r.7. {_HELD}\n\n"
        f"|{_REF}|||\n|---|---|---|\n|(2011) 4 SCC 707|referred to|para 12|\n"
    ),
    # THE NINTH RENDERING AND THE TWO THAT ARE NOT RENDERINGS AT ALL, added
    # after the round-1 residual was read back as "exotic" and two of its six
    # shapes turned out to be ordinary law-report typography:
    #
    #   held_bare        the label with no punctuation after it;
    #   held_title_case  the label as the newer volumes set it;
    #   soft_hyphen      a heading the typesetter broke with a hyphen;
    #   numbered_heading a front-matter section printed as an ordered list;
    #   spaced_and_wrapped  a letter-spaced heading too wide for the column,
    #                    which the per-line de-spacing reads as two halves of
    #                    a heading and recognises as neither.
    "held_bare": (
        "HELD 1. The seniority of a promotee is reckoned from regular appointment.\n\n"
        f"{_REF}\n{_REF_ROW}\n"
    ),
    "held_title_case": (
        "Held: 1. The seniority of a promotee is reckoned from regular appointment.\n\n"
        f"{_REF}\n{_REF_ROW}\n"
    ),
    "soft_hyphen": f"{_HELD}\n\nCase Law Refer-\nence:\n{_REF_ROW}\n",
    "numbered_heading": f"1. {_HELD}\n\n2. {_REF}\n{_REF_ROW}\n",
    "spaced_and_wrapped": (
        "H E L D :  1. The seniority of a promotee is reckoned from regular\n"
        f"appointment.\n\nC A S E   L A W\nR E F E R E N C E :\n{_REF_ROW}\n"
    ),
}

# The same block with the furniture taken out and NOTHING else changed. It is
# the control that makes every test below non-vacuous: with this in place the
# identical early cut is emitted, so the furniture - and only the furniture -
# is what does the refusing.
NO_FURNITURE = (
    "The appeal turns on the construction of rule 7 of the 1978 Rules and on\n"
    "nothing else, as the parties were agreed before us at the hearing.\n"
)

_CAPTION = (
    "KALYANI SHARMA\nv.\nSTATE OF MAHARASHTRA AND OTHERS\n"
    "(Civil Appeal No. 3221 of 2018)\n15 MARCH 2020\n"
)


def _catchwords(chars: int) -> str:
    """Editorial catchword lines carrying no signature at all."""
    line = (
        "Bombay Engineering Service (Recruitment) Rules, 1978 - r.7 - Seniority of "
        "promotees inter se - Whether the period of ad hoc officiation counts towards "
        "seniority in the cadre - Appeal allowed.\n"
    )
    return line * max(1, -(-chars // len(line)))


@pytest.mark.parametrize("rendering", sorted(FURNITURE))
def test_the_editorial_furniture_is_recognised_in_every_rendering(rendering):
    # The guard has to see what the READER emits, not what the reporter
    # printed. Every rendering below is the same two pieces of furniture.
    assert headnote_signals(FURNITURE[rendering]) == ("held", "case_law_reference")
    # ... and the negative half, without which "sees everything" would pass
    # by seeing everything: ordinary editorial prose is not furniture.
    assert headnote_signals(NO_FURNITURE + _catchwords(400)) == ()


@pytest.mark.parametrize("rendering", sorted(FURNITURE))
def test_an_early_cut_is_refused_in_every_rendering_of_the_furniture(rendering):
    front = _CAPTION + "ORDER\n\n" + FURNITURE[rendering]
    result = extract_text([front + _pad(BODY, 2600)])

    assert not result.ok
    assert result.reason == Q_HEADNOTE_RESIDUE
    assert result.text == ""
    # PREMISE 1: the spurious "ORDER" really is taken as the boundary, so the
    # residue guard is the only thing standing between this document and the
    # corpus.
    assert find_judgment_start(front).marker == "judgment_heading"
    # PREMISE 2 - the control, and the whole finding. The SAME document with
    # the furniture replaced by ordinary editorial prose is EMITTED, at the
    # identical early cut. So a rendering the guard cannot read is not a near
    # miss: it is the publisher's headnote in the corpus.
    control = extract_text([_CAPTION + "ORDER\n\n" + NO_FURNITURE + _pad(BODY, 2600)])
    assert control.ok, control.reason
    assert control.text.startswith("ORDER")


# Each signature is a union of two patterns - one read on the demoted text
# and one on the same text with its spacing removed - and until this round
# only `held` and `case_law_reference` had a fixture for the second. The
# seven SECTIONED signatures the newer volumes actually use were pinned in
# their canonical rendering only, so the whole de-spaced half of each was
# deletable with the suite green. A union's branches each need the case only
# they can carry, and these are those cases.

_SECTIONED = {
    "list_of_acts": ("L I S T   O F   A C T S", "Code of Civil Procedure, 1908"),
    "list_of_keywords": ("L I S T   O F   K E Y W O R D S", "Seniority; promotion; rule 7"),
    "cases_referred_to": ("C A S E S   R E F E R R E D   T O :", "(2011) 4 SCC 707"),
    "issue_for_consideration": (
        "I S S U E   F O R   C O N S I D E R A T I O N",
        "Whether ad hoc officiation counts towards seniority.",
    ),
    "headnotes": ("H E A D N O T E S", "Service Law - Promotion - Seniority."),
    "case_arising_from": (
        "C A S E   A R I S I N G   F R O M",
        "Judgment and Order dated 12.03.2017 of the High Court of Bombay.",
    ),
    "appearances": (
        "A P P E A R A N C E S   F O R   P A R T I E S",
        "Ms. A. Shenoy, Sr. Adv. for the appellant.",
    ),
    "case_law_reference": ("C A S E   L A W   R E F E R E N C E", "(2011) 4 SCC 707"),
    "held": ("H E L D :", "1. The seniority is reckoned from regular appointment."),
}


@pytest.mark.parametrize("name", sorted(_SECTIONED))
def test_every_signature_is_recognised_in_the_letter_spaced_rendering(name):
    # THE CASE THE DE-SPACED HALF OF EACH SIGNATURE ALONE CAN CARRY. A
    # typeset heading is letter-spaced, and the demoted text of one carries no
    # adjacent "list" or "acts" for a plain pattern to find - so if this is
    # the rendering the volume uses, the de-spaced pattern is the ONLY thing
    # between that headnote and the corpus.
    heading, body = _SECTIONED[name]
    assert name in headnote_signals(f"{heading}\n{body}\n")
    # THE PREMISE that makes this a test of the de-spaced branch and not of
    # the signature as a whole: the plain branch cannot see this text at all.
    plain_only = tuple(
        n for n, pattern, _ in _SIGNATURES if pattern.search(f"{heading}\n{body}\n")
    )
    assert name not in plain_only
    # ... and the control, so that "recognised" is not "recognises anything":
    # the same words unspaced but not a heading are not a signature.
    assert name not in headnote_signals(f"The parties addressed us on the {body}\n")


@pytest.mark.parametrize("name", sorted(set(_SECTIONED) - {"held"}))
def test_a_letter_spaced_section_heading_is_still_one_when_the_volume_numbers_it(name):
    # THE CASE THE ENUMERATOR PREFIX OF THE DE-SPACED PATTERN ALONE CARRIES.
    # The two prefixes - one on the plain pattern, one on the de-spaced - each
    # covered for the other while a single fixture had both readings, so
    # either was deletable. This document has only the de-spaced reading: the
    # heading is letter-spaced, so nothing in it is adjacent enough for a
    # plain pattern, AND it is numbered, so a de-spaced pattern without the
    # prefix is anchored past the enumerator and sees nothing.
    heading, body = _SECTIONED[name]
    assert name in headnote_signals(f"2. {heading}\n{body}\n")
    plain_only = tuple(
        n for n, pattern, _ in _SIGNATURES if pattern.search(f"2. {heading}\n{body}\n")
    )
    assert name not in plain_only


def test_the_plain_form_carries_a_label_the_column_wrapped_after_a_word():
    # THE CASE THE UNANCHORED LIMB OF THE PLAIN `held` PATTERN ALONE CARRIES,
    # and the reason the two forms of the same limb are not the same rule. The
    # de-spaced limb refuses a letter immediately before the label - `WITHHELD:`
    # must not read as furniture, and de-spacing deletes the space that would
    # otherwise say so - so when the printed column happens to wrap after a
    # WORD rather than after punctuation, only the plain form, which still has
    # the space to anchor `\b` on, sees the reporter's label.
    line = (
        "Bombay Engineering Service (Recruitment) Rules, 1978, rule 7 seniority HELD: 1. The\n"
        "seniority of a promotee is reckoned from regular appointment.\n"
    )
    assert "held" in headnote_signals(line)
    assert not any(
        spaced.search(form)
        for name, _, spaced in _SIGNATURES
        if name == "held"
        for form in (_despace(line), _despace_pairs(line))
    )
    # ... and the negative that limb is shaped around: a word ENDING in the
    # label is not the label.
    assert headnote_signals("The consent was WITHHELD: the appeal fails.") == ()


def test_the_plain_form_carries_a_bare_label_with_its_text_on_the_same_line():
    # THE CASE THE PLAIN HALF OF `held` ALONE CAN CARRY, which is the other
    # side of the same union. Taking the spacing out of this line runs the
    # label into the sentence after it ("HELDTheseniority..."), and the
    # de-spaced pattern is anchored at both ends precisely so that a case name
    # like "HELDER AND ANOTHER" is not a signature - so only the plain form,
    # which has a word boundary to work with, sees this one.
    line = "HELD The seniority of a promotee is reckoned from regular appointment.\n"
    assert "held" in headnote_signals(line)
    spaced_only = tuple(n for n, _, spaced in _SIGNATURES if spaced.search(_despace(line)))
    assert "held" not in spaced_only
    assert "held" not in tuple(
        n for n, _, spaced in _SIGNATURES if spaced.search(_despace_pairs(line))
    )


def test_the_plain_form_carries_a_heading_the_column_wrapped_twice():
    # THE CASE THE PLAIN HALF OF `case_law_reference` ALONE CAN CARRY. The
    # de-spaced form reads one line, and the pair form reads two; a narrow
    # column can break a three-word heading over THREE lines, and the plain
    # pattern's `\s+` is the only thing that spans them - which is also why
    # that `\s+` is not the `[ \t]+` it looks like it could be.
    text = "Case\nLaw\nReference:\n(2011) 4 SCC 707       referred to       para 12\n"
    assert "case_law_reference" in headnote_signals(text)
    spaced_forms = (_despace(text), _despace_pairs(text))
    assert not any(
        spaced.search(form)
        for name, _, spaced in _SIGNATURES
        if name == "case_law_reference"
        for form in spaced_forms
    )
    # ... and the same heading NUMBERED, which is the case the enumerator
    # prefix of the PLAIN pattern alone carries: the de-spaced forms are
    # already out (the heading is broken over three lines), so if the plain
    # pattern is anchored past the enumerator nothing sees this headnote.
    numbered = "3. " + text
    assert "case_law_reference" in headnote_signals(numbered)
    assert not any(
        spaced.search(form)
        for name, _, spaced in _SIGNATURES
        if name == "case_law_reference"
        for form in (_despace(numbered), _despace_pairs(numbered))
    )


def test_a_marker_further_from_the_furniture_than_the_window_is_still_refused():
    # The window is measured in CHARACTERS and the front matter is measured
    # in PAGES: P0 puts the S.C.R. headnote at pages 1-3 of a routine
    # judgment, and a typeset law-report page is comfortably more than 1,500
    # characters - so RESIDUE_WINDOW covers well under half of it. A spurious
    # marker at the TOP of the front matter therefore has nothing to see
    # inside the window at all, and only the whole-document comparison can
    # catch it.
    catchwords = _catchwords(min(RESIDUE_WINDOW, 20_000) + 200)
    front = _CAPTION + "ORDER\n\n" + catchwords + "\n" + FURNITURE["canonical"]
    result = extract_text([front + _pad(BODY, 2600)])

    assert not result.ok
    assert result.reason == Q_HEADNOTE_RESIDUE
    assert result.text == ""
    # THE PREMISE that makes this a test of the comparison and not of the
    # window: past the cut, the window is empty.
    cut = front.index("ORDER")
    assert headnote_signals(front[cut:][:RESIDUE_WINDOW]) == ()
    assert front.index("HELD:") - cut > RESIDUE_WINDOW


def test_a_marker_on_line_one_strips_nothing_and_is_not_a_successful_strip():
    # The reporter puts "ORDER" at the TOP of the front matter, above the
    # headnote it introduces. The marker is found at offset 0, so the "strip"
    # removes nothing and the whole headnote is emitted as if it were the
    # court's own words - and `--audit` reports `headnote signals: none` on
    # it, which is indistinguishable from "this document had no headnote".
    front = (
        "ORDER\n\n" + _CAPTION + _catchwords(min(RESIDUE_WINDOW, 20_000) + 200) + "\n"
        + FURNITURE["canonical"]
    )
    result = extract_text([front + _pad(BODY, 2600)])

    assert not result.ok
    assert result.reason == Q_HEADNOTE_RESIDUE
    assert result.text == ""
    # THE PREMISE: nothing was removed, and there was nothing in the window
    # either - so neither the seam check nor the strip fraction can be what
    # refused this document.
    assert result.headnote_chars == 0
    assert find_judgment_start(front).offset == 0
    assert headnote_signals(front[:RESIDUE_WINDOW]) == ()


def test_a_cut_between_two_held_points_is_caught_though_the_names_match():
    # THE CASE THE COMPARISON CANNOT SEE. The marker fires in the MIDDLE of
    # the HELD block: `held` is on the removed side too, so the comparison is
    # satisfied by name while the second half of the publisher's holding sits
    # directly under the cut.
    #
    # This fixture repeats the LABEL on both sides, which is one rendering and
    # not the common one - a real headnote carries one `HELD:` and numbers the
    # points under it, which is the fixture two tests below. Three of the four
    # seam rules fire here at once, so this test pins the conclusion and NOT
    # any one rule; the branch-alone cases are the four tests that follow.
    front = (
        _CAPTION
        + "HELD: 1. The seniority of a promotee is reckoned from the date of regular\n"
        "appointment and not from the date on which ad hoc officiation began.\n"
        "ORDER\n\n"
        "HELD: 2. The High Court was in error in reading rule 7 as conferring a right\n"
        "to count officiation towards seniority.\n"
    )
    result = extract_text([front + _pad(BODY, 2600)])

    assert not result.ok
    assert result.reason == Q_HEADNOTE_RESIDUE
    assert result.text == ""
    # THE PREMISE: the comparison really is satisfied here - the same
    # signature name stands on both sides of the cut - so the comparison is
    # not what refused it.
    cut = front.index("ORDER")
    assert find_judgment_start(front).offset == cut
    assert "held" in headnote_signals(front[:cut])
    assert "held" in headnote_signals(front[cut:][:RESIDUE_WINDOW])


def test_the_seam_window_alone_carries_a_cut_that_lands_between_two_holdings():
    # BRANCH CASE 1 of the four-branch residue rule: the SEAM WINDOW, and the
    # only document in the suite that no other branch can refuse. The cut is a
    # properly set-off `ORDER` line between two unnumbered holdings, so it
    # begins a block (branch 3 silent) and continues no enumeration (branch 4
    # silent); `held` stands on both sides, so the comparison is satisfied
    # (branch 2 silent). What is left is the evidence AT THE SEAM.
    front = (
        _CAPTION
        + "HELD: The seniority of a promotee is reckoned from the date of regular\n"
        "appointment and not from the date on which ad hoc officiation began.\n"
        "\n"
        "ORDER\n"
        "\n"
        "HELD: The High Court was in error in reading rule 7 as conferring a right\n"
        "to count officiation towards seniority.\n"
    )
    document = front + _pad(BODY, 2600)
    result = extract_text([document])

    assert not result.ok
    assert result.reason == Q_HEADNOTE_RESIDUE
    assert result.text == ""
    # THE PREMISES, one per silenced branch. Branch 4 is asked THE RULE rather
    # than a line-anchored proxy for it: `_ENUM_ITEM` also reads the item
    # standing behind a label (`HELD: 1.`), which no such proxy can see, so a
    # proxy would keep passing on a fixture where branch 4 in fact fires.
    cut = front.index("ORDER")
    assert find_judgment_start(front).offset == cut
    assert "held" in headnote_signals(front[:cut])                  # comparison satisfied
    assert front[:cut].endswith("\n\n")                             # the cut begins a block
    assert not seam_continues_an_enumeration(document[:cut], document[cut:])
    # ... and the seam window is the one thing that does see it.
    assert "held" in headnote_signals(front[cut:][:RESIDUE_WINDOW])


def test_an_early_cut_in_a_one_held_headnote_is_refused_when_the_next_point_is_only_numbered():
    # THE CRITICAL. A real S.C.R. headnote carries ONE `HELD:` and numbers the
    # points under it, and that is all it takes to satisfy every condition the
    # first two branches need in order to stay silent: the only signature name
    # in the file stands on the removed side, and the seam window past the cut
    # is EMPTY because the publisher's second point does not repeat the label.
    #
    # The document is then emitted with the publisher's holding at the top of
    # it AND with `signals: ('held',)` - so the audit's first-run tell, which
    # reads `none` as the alarm, prints a healthy-looking line over
    # contaminated text. That is strictly worse than a rendering nothing
    # recognises, and it is what the third and fourth branches exist for.
    front = (
        _CAPTION
        + "HELD: 1. The seniority of a promotee is reckoned from the date of regular\n"
        "appointment and not from the date on which ad hoc officiation began.\n"
        "ORDER\n\n"
        "2. The High Court was in error in reading rule 7 as conferring a right to\n"
        "count officiation towards seniority.\n"
    )
    document = front + _pad(BODY, 2600)
    result = extract_text([document])

    assert not result.ok
    assert result.reason == Q_HEADNOTE_RESIDUE
    assert result.text == ""
    # THE PREMISES: both older branches really are satisfied by this document,
    # so neither of them can be what refused it.
    cut = front.index("ORDER")
    assert find_judgment_start(front).offset == cut
    assert headnote_signals(document[cut:][:RESIDUE_WINDOW]) == ()
    assert set(headnote_signals(document)) <= set(headnote_signals(front[:cut]))
    # THE CONTROL, and the whole finding: the identical document with the
    # publisher's LABEL taken off - same cut, same words, same numbering - is
    # emitted. So it is the furniture on the removed side, and nothing about
    # the shape of this fixture, that does the refusing.
    control = extract_text([front.replace("HELD: 1.", "1.") + _pad(BODY, 2600)])
    assert control.ok, control.reason
    assert control.text.startswith("ORDER")


def test_an_early_cut_on_a_wrapped_order_line_inside_the_holding_is_refused():
    # THE CRITICAL, second shape and the one that needs no numbering trick at
    # all: the printed column wrapped so that one line reads exactly `ORDER`,
    # deep inside the holding. Everything after it is the publisher's, and the
    # whole of it would have been emitted.
    front = (
        _CAPTION
        + "Service Law - Promotion - Seniority of promotees inter se - Whether the\n"
        "period of ad hoc officiation counts towards seniority.\n"
        "HELD: 1. The seniority of a promotee is reckoned from the date of regular\n"
        "appointment and not from the date on which ad hoc officiation began, and the\n"
        "rule must be read in that light and in no other, for the reasons which the\n"
        "ORDER\n"
        "2. The construction placed on rule 7 by the Full Bench does not lay down good\n"
        "law and the appeal is accordingly allowed with no order as to costs.\n"
        "3. The seniority list shall be redrawn within three months from today.\n"
    )
    document = front + _pad(BODY, 2600)
    result = extract_text([document])

    assert not result.ok
    assert result.reason == Q_HEADNOTE_RESIDUE
    assert result.text == ""
    # THE PREMISES, again: the seam window is empty and the comparison is
    # satisfied, so this refusal is neither of them.
    cut = front.index("ORDER\n")
    assert find_judgment_start(front).offset == cut
    assert headnote_signals(document[cut:][:RESIDUE_WINDOW]) == ()
    assert set(headnote_signals(document)) <= set(headnote_signals(front[:cut]))
    # THE CONTROL: label off, everything else identical, emitted.
    control = extract_text([front.replace("HELD: 1.", "1.") + _pad(BODY, 2600)])
    assert control.ok, control.reason
    assert control.text.startswith("ORDER")


def test_a_cut_that_does_not_begin_a_block_is_refused_when_furniture_was_removed():
    # BRANCH CASE 3: THE SEAM BREAK, alone. A typeset heading is set off from
    # the text above it; a wrapped line is not. So when the removed head
    # carried editorial furniture, the cut has to BEGIN a block - and this
    # document's does not, while carrying no enumeration anywhere for branch 4
    # to read and no signature past the cut for branch 1.
    front = (
        _CAPTION
        + "HELD: The seniority of a promotee is reckoned from the date of regular\n"
        "appointment and not from the date on which ad hoc officiation began, and the\n"
        "ORDER\n"
        "of the High Court proceeded on a reading of rule 7 that its language does not\n"
        "carry, as the parties were agreed before us at the hearing of this appeal.\n"
    )
    document = front + _pad(BODY, 2600)
    result = extract_text([document])

    assert not result.ok
    assert result.reason == Q_HEADNOTE_RESIDUE
    assert result.text == ""
    # THE PREMISES, one per silenced branch.
    cut = front.index("ORDER\n")
    assert find_judgment_start(front).offset == cut
    assert headnote_signals(document[cut:][:RESIDUE_WINDOW]) == ()           # branch 1 silent
    assert set(headnote_signals(document)) <= set(headnote_signals(front[:cut]))  # branch 2 silent
    assert not seam_continues_an_enumeration(document[:cut], document[cut:])  # branch 4 silent
    # THE CONTROL: the same cut in the same words with no furniture removed is
    # emitted, so the seam break only ever fires on a cut through furniture.
    control = extract_text([front.replace("HELD: ", "") + _pad(BODY, 2600)])
    assert control.ok, control.reason
    assert control.text.startswith("ORDER")


def test_a_cut_that_continues_the_removed_headnotes_numbering_is_refused():
    # BRANCH CASE 4: THE SEAM ENUMERATION, alone. Here the reporter really did
    # set `ORDER` off as its own block - between two numbered holding points -
    # so the seam break is satisfied and the cut still lands inside the
    # publisher's list. A judgment begins at ITS first paragraph; a body that
    # opens on the point after the one the headnote had reached is the
    # headnote continuing.
    # THREE points before the cut and not one, so that "the LAST item the
    # removed head reached" is a different number from "the first one it
    # had": a rule that compared against the first would pass this document
    # and the fixture would never say so.
    front = (
        _CAPTION
        + "HELD: 1. The seniority of a promotee is reckoned from the date of regular\n"
        "appointment and not from the date on which ad hoc officiation began.\n"
        "2. Rule 7 confers no right to count officiation towards seniority.\n"
        "3. The Full Bench decision does not lay down good law.\n"
        "\n"
        "ORDER\n"
        "\n"
        "4. The seniority list shall be redrawn within three months from today, and\n"
        "the appeal is allowed in those terms with no order as to costs.\n"
    )
    document = front + _pad(BODY, 2600)
    result = extract_text([document])

    assert not result.ok
    assert result.reason == Q_HEADNOTE_RESIDUE
    assert result.text == ""
    # THE PREMISES, one per silenced branch.
    cut = front.index("ORDER\n")
    assert find_judgment_start(front).offset == cut
    assert headnote_signals(document[cut:][:RESIDUE_WINDOW]) == ()           # branch 1 silent
    assert set(headnote_signals(document)) <= set(headnote_signals(front[:cut]))  # branch 2 silent
    assert front[:cut].endswith("\n\n")                                      # branch 3 silent
    # ... and the premise the three points exist for: the head reached 3, so
    # only a rule reading the LAST item can meet the body's 4.
    assert _ENUM_ITEM.findall(front[:cut]) == ["1", "2", "3"]
    # THE CONTROL: the same document whose body opens at ITS OWN first
    # paragraph instead of the headnote's next one is emitted.
    control = extract_text([front.replace("\n4. The seniority list", "\n1. The seniority list")
                            + _pad(BODY, 2600)])
    assert control.ok, control.reason


def test_a_headnote_that_has_paragraph_breaks_earlier_does_not_excuse_the_cut():
    # The block break is read at the END of the removed head and nowhere else,
    # which is the difference between "this headnote had paragraphs in it" and
    # "the cut began one". A `HELD` block runs to several points and has blank
    # lines between them; a rule that accepted any blank line anywhere would be
    # satisfied by the FIRST of them and never look at the seam again.
    front = (
        _CAPTION
        + "HELD: The seniority of a promotee is reckoned from regular appointment.\n"
        "\n"
        "The High Court read rule 7 as conferring a right that its language does not\n"
        "carry, and the construction placed on it by the Full Bench is therefore the\n"
        "ORDER\n"
        "under appeal, which cannot be sustained on any reading of the rule.\n"
    )
    document = front + _pad(BODY, 2600)
    result = extract_text([document])

    assert not result.ok
    assert result.reason == Q_HEADNOTE_RESIDUE
    assert result.text == ""
    # THE PREMISES: the head really does carry a paragraph break, and it is
    # really not at the seam - and no other branch can see this document.
    cut = front.index("ORDER\n")
    assert "\n\n" in front[:cut]
    assert not front[:cut].endswith("\n\n")
    assert headnote_signals(document[cut:][:RESIDUE_WINDOW]) == ()
    assert set(headnote_signals(document)) <= set(headnote_signals(front[:cut]))
    assert not seam_continues_an_enumeration(document[:cut], document[cut:])


def test_a_cut_after_a_single_holding_point_still_reads_the_number_behind_the_label():
    # The enumerator of the FIRST holding point sits behind the label that
    # opens the block (`HELD: 1.`) and not at the start of its line, so a rule
    # that only reads line-anchored numbers sees no enumeration at all in a
    # one-point headnote - which is the commonest headnote there is. The test
    # above has three points and cannot say this, because its second and third
    # are line-anchored and carry the comparison on their own.
    front = (
        _CAPTION
        + "HELD: 1. The seniority of a promotee is reckoned from the date of regular\n"
        "appointment and not from the date on which ad hoc officiation began.\n"
        "\n"
        "ORDER\n"
        "\n"
        "2. The High Court was in error in reading rule 7 as conferring a right to\n"
        "count officiation towards seniority.\n"
    )
    document = front + _pad(BODY, 2600)
    result = extract_text([document])

    assert not result.ok
    assert result.reason == Q_HEADNOTE_RESIDUE
    assert result.text == ""
    # THE PREMISES: every other branch is silent, and the ONLY enumerator in
    # the removed head is the one standing behind the label.
    cut = front.index("ORDER\n")
    assert headnote_signals(document[cut:][:RESIDUE_WINDOW]) == ()
    assert set(headnote_signals(document)) <= set(headnote_signals(front[:cut]))
    assert front[:cut].endswith("\n\n")
    assert re.search(r"^\s{0,3}\(?\d{1,3}[.)]\s", front[:cut], re.M) is None


def test_a_judgment_that_restarts_its_own_numbering_is_not_read_as_a_continuation():
    # THE RECALL SIDE of branch 4, and the reason it compares the NUMBERS
    # rather than demanding the body open at 1. The headnote here runs to
    # three points and the judgment opens at its own paragraph 1, which is
    # what a reprint looks like; a rule that refused every body not opening at
    # `1.` would also refuse every judgment whose first paragraph the reader
    # did not number.
    front = (
        _CAPTION
        + "HELD: 1. The seniority of a promotee is reckoned from regular appointment.\n"
        "2. The High Court was in error in reading rule 7.\n"
        "3. The seniority list shall be redrawn within three months.\n"
        "\n"
    )
    result = extract_text([front + _pad(BODY, 2600)])

    assert result.ok, result.reason
    assert result.text.startswith("The Judgment of the Court was delivered by")
    # THE PREMISE: the removed head really did end on point 3 and the body
    # really does open at 1, so branch 4 had something to compare.
    assert "3. The seniority list" in front
    assert re.search(r"^1\.", result.text, re.M) is not None


def test_a_body_numbered_past_the_headnotes_last_point_is_not_a_continuation():
    # THE OTHER RECALL SIDE of branch 4, and the side the suite did not have a
    # case for: the comparison is EXACT-SUCCESSOR, so it reads only the body
    # that counts up by ONE from the point the removed head reached. The test
    # above pins the body that counts DOWN (a judgment restarting at its own
    # `1.`) and says nothing about this one, so a rule loosened to refuse every
    # body whose first number merely EXCEEDS the head's last passes it.
    #
    # That loosening would refuse THIS document, which is a judgment whose
    # opening paragraphs the reader dropped, or which numbers from a base of
    # its own. Nothing inside the file separates it from a headnote continuing
    # except the GAP, and the gap is what the exact comparison reads.
    front = (
        _CAPTION
        + "HELD: 1. The seniority of a promotee is reckoned from regular appointment.\n"
        "2. The High Court was in error in reading rule 7.\n"
        "\n"
        "ORDER\n"
        "\n"
    )
    document = front + _paras(5, 2600)
    result = extract_text([document])

    assert result.ok, result.reason
    # THE PREMISES: every other branch is silent, the head really reached 2 and
    # the body really opens at 5 - so the GAP is the only thing emitting this.
    cut = front.index("ORDER\n")
    assert find_judgment_start(document).offset == cut
    assert headnote_signals(document[cut:][:RESIDUE_WINDOW]) == ()
    assert set(headnote_signals(document)) <= set(headnote_signals(document[:cut]))
    assert document[:cut].endswith("\n\n")
    assert _ENUM_ITEM.findall(document[:cut])[-1] == "2"
    assert _ENUM_ITEM.search(document[cut:][:SEAM_ENUM_WINDOW]).group(1) == "5"
    assert not seam_continues_an_enumeration(document[:cut], document[cut:])
    # THE CONTROL: close the gap to exactly one - same words, same cut, same
    # furniture removed - and the identical document is refused.
    control = extract_text([front + _paras(3, 2600)])
    assert not control.ok
    assert control.reason == Q_HEADNOTE_RESIDUE


def test_an_emitted_document_reports_every_signature_it_still_carries():
    # THE FIRST-RUN TELL, as a property rather than as advice. Because a
    # document is emitted only when the removed side accounts for every
    # signature the document carries, `signals` on an emitted document is the
    # whole document's signature set - so `headnote signals: none` in the
    # audit means "no editorial furniture anywhere in this file", and NOT
    # "the guard looked and shrugged". A blind guard therefore shows up as
    # `none` printed against a document that visibly has a headnote.
    emitted = extract_text(scr_pages())
    assert emitted.ok, emitted.reason
    cleaned, _ = clean_pages(scr_pages())
    assert set(emitted.signals) == set(headnote_signals("\n".join(cleaned)))
    assert emitted.signals != ()

    plain = extract_text([_pad(BODY, 2600)])
    assert plain.ok, plain.reason
    assert plain.signals == ()
    assert headnote_signals(_pad(BODY, 2600)) == ()


def test_a_boundary_that_would_discard_most_of_the_document_is_quarantined():
    front = _pad(HEADNOTE, 20000)
    body = _pad(BODY, 2000)
    result = extract_text([front + "\n" + body])

    assert not result.ok
    assert result.reason == Q_STRIP_TOO_LARGE
    # DISJOINT from the short-body rule: this body is comfortably over the
    # floor, so the quarantine can only be the strip fraction.
    assert len(body) > MIN_BODY_CHARS
    assert len(front) / (len(front) + len(body)) > MAX_STRIP_FRACTION


def test_a_body_too_short_to_be_a_judgment_is_quarantined():
    front = _pad(HEADNOTE, 4600)
    body = _pad("The Judgment of the Court was delivered by\nNAVIN SINHA, J.\n", 1300)
    result = extract_text([front + "\n" + body])

    assert not result.ok
    assert result.reason == Q_BODY_TOO_SHORT
    # DISJOINT from the strip-fraction rule: this document's cut is well
    # inside the allowance, so the quarantine can only be the body floor.
    assert len(front) / (len(front) + len(body)) < MAX_STRIP_FRACTION
    assert len(body) < MIN_BODY_CHARS


def test_a_scan_with_no_text_layer_is_quarantined_rather_than_read_as_empty():
    # v1 ships no OCR: a pre-1990-style scan under a 2010-2025 key has to
    # come back as a refusal, not as a zero-length judgment.
    result = extract_text(["", "   \n\n", ""])

    assert not result.ok
    assert result.reason == Q_NO_TEXT
    assert result.text == ""


def test_a_garbled_text_layer_is_quarantined_before_the_boundary_is_blamed():
    # A broken font map extracts as (cid:NN) soup. It is long enough to
    # pass the empty check and it carries a real marker, so "no text" and
    # "no judgment start" would both misdiagnose it.
    page = "The Judgment of the Court was delivered by\n" + _paras(1, 2000).replace(
        "the record", "the (cid:24)(cid:37)(cid:12)"
    )
    result = extract_text([page])

    assert not result.ok
    assert result.reason == Q_LOW_TEXT_QUALITY
    # Each of the other three readings of this document is ruled out by a
    # fact about the fixture, not by the reason string already asserted.
    # The last one matters most: a PARTIALLY broken font map leaves plenty
    # of real prose, so the letter ratio stays healthy and the (cid:NN)
    # count is the only thing that can see it.
    assert len(page) > MIN_DOC_CHARS
    assert find_judgment_start(page) is not None
    assert latin_ratio(page) > MIN_LATIN_RATIO


def test_a_regional_text_layer_is_quarantined():
    devanagari = "यह निर्णय हिंदी में है और यह अंग्रेजी उपसर्ग के अंतर्गत नहीं होना चाहिए। " * 40
    page = "The Judgment of the Court was delivered by\n" + devanagari
    result = extract_text([page])

    assert not result.ok
    assert result.reason == Q_LOW_TEXT_QUALITY
    # The OTHER limb: nothing here is a broken font map, so the letter
    # ratio is the only check that can refuse this document.
    assert "(cid:" not in page
    assert latin_ratio(page) < MIN_LATIN_RATIO


# Hindi typed in a legacy 8-bit Devanagari font (Kruti Dev and its
# relatives, which is what Indian courts typed Hindi in for twenty years).
# The glyph codes ARE Latin codepoints, so this is what the text layer of
# such a page extracts as - not Devanagari, not `(cid:NN)` soup, but fluent-
# looking ASCII with no English in it.
MANGLED_HINDI = (
    "vfHkfyf[kr fu.kZ; esa mYysf[kr rF; ,oa lk{; ds vk/kkj ij ;g fu"
    ""
    " fpr fd;k tkrk gS fd vihykFkhZ dks jkgr iznku dh tk;sA "
) * 40


def test_hindi_mangled_into_latin_letters_is_refused_though_every_letter_is_latin():
    # THE GAP THE OTHER TWO LIMBS CANNOT SEE, and the reason for a third.
    # The characters are Latin, so latin_ratio is happy; there is no broken
    # font map, so the (cid:NN) count is zero; and script detection has
    # nothing to detect because there is no Devanagari in the text layer at
    # all. What it has none of is ENGLISH.
    page = "The Judgment of the Court was delivered by\n" + MANGLED_HINDI
    result = extract_text([page])

    assert not result.ok
    assert result.reason == Q_LOW_TEXT_QUALITY
    # THE PREMISES, and they are the whole point: BOTH of the older limbs
    # pass this document. Without these two lines this test would pass on a
    # module that had never grown the third limb.
    assert "(cid:" not in page
    assert latin_ratio(page) > MIN_LATIN_RATIO
    assert stopword_rate(page) < MIN_STOPWORD_RATE


def test_real_judgment_prose_is_far_above_the_english_floor():
    # The other direction, without which the floor could be set anywhere. The
    # measured range over 15 real objects was 0.372-0.457 on the whole
    # document and on the emitted body alike, so the floor has better than
    # two-fold headroom under the worst real document.
    assert stopword_rate(HEADNOTE + BODY) > MIN_STOPWORD_RATE * 2
    assert extract_text(scr_pages()).ok


# ------------------------------------------- what the OBJECT is made of

def _structure(**over) -> PdfStructure:
    facts = {
        "fonts": (("ABCDEF+Helvetica", "WinAnsiEncoding"),),
        "image_filters": (),
        "image_only_pages": 0,
        "pages": 12,
    }
    facts.update(over)
    return PdfStructure(**facts)


@pytest.mark.parametrize(
    "structure",
    [
        # The ABBYY OCR layer's own font, which is the fingerprint of the
        # 2010-2017 volumes: the text is INVISIBLE glyphs laid under a
        # bitonal scan of the printed page.
        _structure(fonts=(("HiddenHorzOCR", "WinAnsiEncoding"),)),
        # ...the same thing seen from the image side, on a file whose fonts
        # the library reports differently.
        _structure(image_filters=("/JBIG2Decode",)),
        # ...and a file that says neither, but is half pages with a picture
        # on them and no text.
        _structure(image_only_pages=6, pages=12),
    ],
)
def test_a_scan_era_object_is_refused_on_its_structure(structure):
    # THE DECISION THIS STOPS BEING TAKEN BY ACCIDENT. These files READ
    # plausibly - 1,840-2,100 chars a page on the sampled objects, and the
    # module emitted all six of them - so nothing downstream would notice.
    # But citation-level accuracy is exactly where twenty-year-old OCR fails,
    # and the evidence is already in their running heads: `[201 O]` for
    # `[2010]`, `S.Q.R.` for `S.C.R.`. Whether OCR-era text is in v1 is an
    # operator's call; making it silently is not.
    assert structural_refusal(structure) == Q_SCANNED_ERA


def test_a_born_digital_object_with_a_photograph_in_it_is_not_a_scan():
    # MEASURED, and it is why the test is the scan STRUCTURE and not "has
    # images": one real 2018 object carries a DCTDecode photograph across 70
    # otherwise born-digital pages with subsetted embedded fonts. A gate that
    # read "an image" would have refused it.
    assert structural_refusal(
        _structure(
            fonts=(("IQIVMP+00bqoyjfrmmezrq,Bold", "WinAnsiEncoding"),),
            image_filters=("/DCTDecode",),
            image_only_pages=0,
            pages=70,
        )
    ) is None
    # ...and one image-only page in seventy is a scanned exhibit, not a
    # scanned judgment.
    assert structural_refusal(_structure(image_only_pages=1, pages=70)) is None


@pytest.mark.parametrize(
    "font",
    [
        "KrutiDev010",
        "ABCDEF+Kruti Dev 010",          # subsetted, as the reader reports it
        "DevLys 010",
        "Shree-Dev7-0714",
    ],
)
def test_a_legacy_devanagari_font_is_refused_though_its_text_reads_as_latin(font):
    # The structural half of the mangled-Hindi failure. A Devanagari-family
    # font declared with anything but an Identity encoding is an 8-bit glyph
    # mapping, and its text layer is nonsense however clean it looks - so the
    # refusal is available before a single page is converted.
    assert structural_refusal(_structure(fonts=((font, "WinAnsiEncoding"),))) == (
        Q_MOJIBAKE_FONT
    )


def test_every_font_family_is_written_in_the_form_the_gate_can_match():
    # A ROW OF THE TABLE NOTHING COULD REACH. The font name is folded before
    # the test and the family is not, so a family written with a hyphen can
    # never match anything: `"shree-dev"` sat in this tuple looking like
    # coverage of `Shree-Dev7-0714` while `"shreedev"` two entries earlier
    # was what actually matched it. The parametrised test above passed either
    # way - which is the shape of a fixture that guarantees a null - so the
    # form of the table is asserted here instead.
    from tuned.data.extract import DEVANAGARI_FONT_FAMILIES, IDENTITY_ENCODINGS, _folded

    for family in DEVANAGARI_FONT_FAMILIES:
        assert _folded(family) == family, family
    assert len(set(DEVANAGARI_FONT_FAMILIES)) == len(DEVANAGARI_FONT_FAMILIES)
    # The encodings are compared folded on BOTH sides, so those may carry the
    # hyphen they are really written with - and do.
    assert [_folded(name) for name in IDENTITY_ENCODINGS] != list(IDENTITY_ENCODINGS)


def test_a_devanagari_font_with_a_unicode_encoding_is_not_mojibake():
    # The other direction, and it is what keeps the rule from refusing every
    # bilingual judgment: an Identity-H Devanagari font extracts as real
    # Devanagari, which the letter-ratio check handles on its own terms.
    assert structural_refusal(_structure(fonts=(("Mangal", "Identity-H"),))) is None
    assert structural_refusal(_structure(fonts=(("Nirmala UI", "Identity-V"),))) is None


def test_an_undecodable_font_is_reported_before_the_era_it_shares_a_file_with():
    # ORDER, on a document that is both. Re-OCR fixes a scan; nothing fixes a
    # font this module cannot decode - so the reason the operator reads is
    # the one that says what to do.
    both = _structure(
        fonts=(("HiddenHorzOCR", "WinAnsiEncoding"), ("KrutiDev010", "WinAnsiEncoding")),
        image_filters=("/JBIG2Decode",),
    )
    assert structural_refusal(both) == Q_MOJIBAKE_FONT
    # BOTH really do fire on this object - which is what makes this a test of
    # the order rather than of either rule.
    assert structural_refusal(_structure(fonts=(("HiddenHorzOCR", "x"),))) == Q_SCANNED_ERA
    assert structural_refusal(_structure(fonts=(("KrutiDev010", "x"),))) == Q_MOJIBAKE_FONT


def test_an_ordinary_born_digital_object_is_admitted():
    # Without this the gate could refuse everything and every test above
    # would still pass.
    assert structural_refusal(_structure()) is None
    assert structural_refusal(PdfStructure()) is None


def test_a_page_break_in_the_middle_of_a_sentence_does_not_become_a_paragraph_break():
    # Judgments run on across pages. A blank line at every page boundary
    # announces a paragraph break in the middle of a sentence, and the
    # Tier-3 packer downstream reads blank lines.
    pages = [
        "The Judgment of the Court was delivered by\nNAVIN SINHA, J.\n"
        + _paras(1, 1600)
        + "9. The question is whether the delay stands explained on this record, and",
        "we turn to the evidence bearing on it, which begins with the deposition.\n",
    ]
    result = extract_text(pages)

    assert result.ok, result.reason
    assert "explained on this record, and\nwe turn to the evidence" in result.text


def test_a_cut_that_is_both_too_large_and_too_short_is_reported_as_the_short_body():
    # DESIGN DECISION 6, which nothing enforced. When both rules hold there
    # is nothing to recover either way, and the ORDER decides which word the
    # operator reads on the quarantine breakdown. `strip_too_large` has to
    # keep its alarming meaning - MOST OF A DOCUMENT THAT STILL HAD PLENTY OF
    # TEXT IN IT was thrown away, i.e. a boundary bug - and if it fired first
    # it would also fire on every one-line dismissal order in the corpus and
    # stop meaning anything at all.
    front = _pad(HEADNOTE, 12000)
    body = _pad("The Judgment of the Court was delivered by\nNAVIN SINHA, J.\n", 900)
    result = extract_text([front + "\n" + body])

    assert not result.ok
    assert result.reason == Q_BODY_TOO_SHORT
    # BOTH rules really do fire on this document - which is what makes this a
    # test of the order and not of either rule. (The two tests above are the
    # disjoint cases, and neither can see the order at all.)
    assert len(body.strip()) < MIN_BODY_CHARS
    assert len(front) / (len(front) + len(body)) > MAX_STRIP_FRACTION


def test_the_emitted_text_is_plain_and_the_paragraph_anchor_survives():
    # THE TIER-1 SEGMENTATION CONTRACT, and it was pinned only inside
    # find_judgment_start's own per-line demotion - not on what the pipeline
    # EMITS. pymupdf4llm hands over markdown, `**1.**` does not match the
    # `^\d+\.` paragraph regex that P0 found to be the only corpus-wide
    # segmentation signal, and a corpus of undemoted text is a corpus
    # segment.py cannot read. The regression would surface a stage later, as
    # a segmenter that mysteriously finds no paragraphs.
    page = (
        "**The Judgment of the Court was delivered by**\n"
        "## NAVIN SINHA, J.\n"
        "-----\n"
        + "".join(
            f"**{n}.** The submission advanced on behalf of the appellant proceeds on a "
            f"reading of the record which, on examination, the record does not bear out.\n"
            for n in range(1, 16)
        )
    )
    result = extract_text([page])

    assert result.ok, result.reason
    assert "**" not in result.text
    assert "#" not in result.text
    assert "-----" not in result.text
    # THE POINT, which absence alone does not make: the downstream anchor
    # matches the EMITTED text and does not match what the reader handed
    # over, so the demotion is what puts it there.
    assert re.search(r"^\d+\.", result.text, re.M) is not None
    assert re.search(r"^\d+\.", page, re.M) is None


def test_the_emitted_text_carries_no_blank_run_the_packer_would_read_as_a_gap():
    # normalise_whitespace, which nothing pinned at the pipeline level. The
    # Tier-3 packer downstream reads blank lines as structure, and the page
    # joiner two rules up exists for exactly that reason - so leaving the
    # reader's own blank runs in place would undo it from the other end.
    page = (
        "The Judgment of the Court was delivered by\nNAVIN SINHA, J.   \n\n\n\n"
        + _paras(1, 2000)
        + "\n\n\n\n"
    )
    result = extract_text([page])

    assert result.ok, result.reason
    assert "\n\n\n" not in result.text
    assert "   \n" not in result.text
    # One trailing newline, not none and not four: a file every consumer can
    # concatenate without inventing a paragraph break.
    assert result.text.endswith("\n")
    assert not result.text.endswith("\n\n")
    # THE PREMISE: the input really does carry the runs and the trailing
    # spaces, so this is a test of the pass and not of the fixture.
    assert "\n\n\n" in page and "   \n" in page


def test_a_judgment_with_no_headnote_at_all_is_kept_whole():
    # Not every object in the bucket need be a reprint; a plain judgment has
    # nothing to strip and must not be quarantined for it.
    result = extract_text([_pad(BODY, 2600)])

    assert result.ok
    assert result.headnote_chars == 0
    assert result.signals == ()
    assert result.text.startswith("The Judgment of the Court was delivered by")


# --------------------------------------------------------------------------
# Finding the boundary.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "line,marker",
    [
        ("The Judgment of the Court was delivered by", "judgment_delivered_by"),
        ("The following Judgment of the Court was delivered by", "judgment_delivered_by"),
        ("The Judgment and Order of the Court was delivered by", "judgment_delivered_by"),
        ("**The Judgment of the Court was delivered by**", "judgment_delivered_by"),
        # THE SECOND ALTERNATION OF `_DELIVERED_BY`, which nothing reached:
        # the reporters put the verb before the noun as often as after it, and
        # with no case here the whole limb was deletable with the suite green.
        ("NAVIN SINHA, J. delivered the following judgment", "judgment_delivered_by"),
        ("The Court delivered the following order", "judgment_delivered_by"),
        ("R.F. NARIMAN, J. delivered the following Judgment", "judgment_delivered_by"),
        # THE TWO THE 2014 VOLUMES USE AND THIS LIST DID NOT HAVE, each of
        # which quarantined a real judgment as `no_judgment_start`. The first
        # carries no "judgment" token at all; the second is plural AND takes
        # "were", so a pattern that widened only the noun still misses it.
        ("The order of the Court was delivered by", "judgment_delivered_by"),
        ("The Judgments of the Court were delivered by", "judgment_delivered_by"),
        ("The Orders of the Bench were delivered by", "judgment_delivered_by"),
        ("J U D G M E N T", "judgment_heading"),
        ("JUDGMENT", "judgment_heading"),
        ("## JUDGMENT", "judgment_heading"),
        ("O R D E R", "judgment_heading"),
        # The heading behind an inline tag and behind a margin letter - see
        # the two dedicated tests for what each of those cost.
        ("**<u>Judgment</u>**", "judgment_heading"),
        ("C JUDGMENT", "judgment_heading"),
    ],
)
def test_the_markers_the_reprints_actually_use(line, marker):
    found = find_judgment_start(f"HELD: something editorial\n{line}\n1. The appellant says.")
    assert found is not None
    assert found.marker == marker
    # The marker line itself is kept: it is the anchor a reader (and the
    # audit) uses to see the seam, and it is the court's text, not the
    # publisher's.
    assert found.offset == len("HELD: something editorial\n")


@pytest.mark.parametrize(
    "line",
    [
        "From the Judgment and Order dated 12.02.2018 of the High Court of Bombay",
        "Order XLI Rule 27 of the Code of Civil Procedure",
        "the judgment of the court below was delivered without reasons, we are told",
        "Case Arising From Judgment/Order dated 03.05.2019",
        # The widened noun and verb must not have widened the SENTENCE: what
        # holds `_DELIVERED_BY` together is that the line ENDS on "delivered
        # by", and prose about an order of the court does not.
        "The order of the Court was delivered by post to the parties on 3 May",
        "The orders of the Court were pronounced in open court that morning",
    ],
)
def test_prose_that_merely_mentions_a_judgment_is_not_a_boundary(line):
    # Every one of these appears in the editorial front matter of real
    # reprints. Matching one would cut INSIDE the headnote.
    assert find_judgment_start(line) is None


def test_the_first_marker_wins_so_a_later_opinion_does_not_become_the_start():
    # A concurring or dissenting opinion later in the same file carries its
    # own heading; taking the last match would drop the majority judgment.
    text = (
        "HELD: editorial\n"
        "The Judgment of the Court was delivered by\nA, J.\n1. Majority reasoning.\n"
        "J U D G M E N T\nB, J.\n1. I agree, but add this.\n"
    )
    found = find_judgment_start(text)
    assert found.marker == "judgment_delivered_by"
    assert text[found.offset :].startswith("The Judgment of the Court")


def test_headnote_signals_name_the_editorial_furniture_and_not_ordinary_prose():
    assert "held" in headnote_signals("HELD: 1. The appeal is allowed.")
    assert "case_law_reference" in headnote_signals("Case Law Reference:\n(2011) 4 SCC 707")
    assert "list_of_acts" in headnote_signals("List of Acts\nCode of Civil Procedure, 1908")
    assert "issue_for_consideration" in headnote_signals("Issue for Consideration\nWhether...")
    # Prose the COURT writes, which must not read as editorial furniture -
    # otherwise every judgment quoting an earlier holding is quarantined.
    assert headnote_signals("This Court held that the appeal must fail.") == ()
    assert headnote_signals("The submission is that the cases referred by counsel.") == ()
    # And the reason HELD is matched CASE-SENSITIVELY. The text is hard
    # wrapped, so a line boundary lands wherever the column ended: "...the
    # High Court / held: that the suit was barred" puts an ordinary verb at
    # the start of a line, and a case-blind rule would read the judgment's
    # own quotation of an order as the reporter's headnote and refuse the
    # document. Prose is never in capitals; the reprint's HELD always is.
    assert headnote_signals("held: that the suit was barred by limitation.") == ()
    # WHERE THAT LINE MOVED, and why. The assertion here used to be that
    # title-case `Held:` is prose too, which was over-broad: the rationale
    # above is about what a COLUMN WRAP can leave at the start of a line, and
    # a wrap cannot capitalise a word. A capital at a line start means the
    # source had one - a sentence opening or a heading - and `Held:` opening a
    # line is the label the newer volumes set. So it is furniture, and the
    # discriminator that survives is the one the rationale actually supports:
    # LOWER CASE stays out, and title case is LINE-ANCHORED, so the verb
    # inside a sentence is untouched wherever the wrap happens to fall.
    assert headnote_signals("Held: that the suit was barred by limitation.") == ("held",)
    assert headnote_signals("The High Court Held: that the suit was barred.") == ()
    # The two other renderings of the same label that no pattern read: a bare
    # all-caps HELD with no punctuation at all, and the same heading with the
    # spacing the typesetter put in it.
    assert headnote_signals("HELD\nThe seniority is reckoned from regular appointment.") == ("held",)
    assert headnote_signals("H E L D\nThe seniority is reckoned from appointment.") == ("held",)
    # ... and the negative that keeps `HELD` from meaning "any capital word":
    # an all-caps line is not a signature unless it is THIS label.
    assert headnote_signals("HELDER AND ANOTHER v. STATE OF MAHARASHTRA") == ()


# --------------------------------------------------------------------------
# Cleanup: signature stamps, running furniture, footnotes.
# --------------------------------------------------------------------------

def test_a_digital_signature_side_stamp_is_removed_but_dated_prose_survives():
    page = (
        "1. The offence is alleged to have occurred on 12.03.2019 at about 21:30:00 hours.\n"
        "Digitally signed by\n"
        "ARJUN BISHT\n"
        "Date: 2023.05.11\n"
        "16:44:12 IST\n"
        "Reason:\n"
        "Signature Not Verified\n"
        "2. The trial court convicted the appellant.\n"
    )
    cleaned, stats = clean_pages([page])
    text = cleaned[0]

    assert "ARJUN BISHT" not in text
    assert "Digitally signed" not in text
    assert "Signature Not Verified" not in text
    assert "16:44:12" not in text
    # The stamp's shapes appear in ordinary judgment prose too: a date and a
    # time inside a sentence are facts of the case, not furniture.
    assert "occurred on 12.03.2019 at about 21:30:00 hours" in text
    assert "2. The trial court convicted the appellant." in text
    # All six lines of the block, not "most of them": a stamp half removed
    # leaves a signer's name loose in the middle of a judgment.
    assert stats["signature"] == 6


def test_the_signer_name_is_dropped_only_after_a_bare_stamp_header():
    # The rotated block sometimes extracts as one line and sometimes as
    # several. Dropping "the line after" unconditionally would eat a line of
    # the judgment in the one-line case.
    one_line = (
        "Digitally signed by ARJUN BISHT Date: 2023.05.11 16:44:12 IST Signature Not Verified\n"
        "1. This sentence is the first paragraph of the judgment.\n"
    )
    cleaned, _ = clean_pages([one_line])
    assert "ARJUN BISHT" not in cleaned[0]
    assert "1. This sentence is the first paragraph of the judgment." in cleaned[0]


def test_running_headers_page_numbers_and_watermarks_go_but_body_lines_stay():
    pages = []
    for i in range(6):
        pages.append(
            f"{RUNNING_HEADER}\n"
            "www.example-watermark.in\n"
            f"{i + 1}. Paragraph {i + 1} of the judgment is on this page and on no other page at all.\n"
            + ("A short repeated aside.\n" if i < 2 else "")
            + f"{941 + i}\n"
        )
    cleaned, stats = clean_pages(pages)
    joined = "\n".join(cleaned)

    assert RUNNING_HEADER not in joined
    assert "www.example-watermark.in" not in joined
    assert "\n941\n" not in joined and not joined.startswith("941\n")
    # THE TRAP: these body lines differ from each other only in a digit, so
    # the digit-blind key that catches the printed page number reads all six
    # as one repeated line. If it is applied to lines of prose, the body of
    # every judgment whose pages open with a numbered paragraph is deleted
    # as furniture - and the deletion is silent. They are deliberately SHORT
    # enough to be running-line candidates (under RUNNING_MAX_CHARS), so the
    # length cap cannot be what saves them and the digit-blind guard is the
    # only thing standing between them and deletion.
    for i in range(6):
        line = f"{i + 1}. Paragraph {i + 1} of the judgment is on this page and on no other page at all."
        assert len(line) <= RUNNING_MAX_CHARS
        assert len(line) > RUNNING_DIGIT_BLIND_CHARS
        assert line in joined
    # 2 of 6 pages is below the threshold: a line has to be furniture on
    # MOST pages before it is treated as furniture at all.
    assert "A short repeated aside." in joined
    assert stats["running"] >= 6


def _body_line(i: int) -> str:
    """A body line that differs from its neighbours only in a digit.

    LONG on purpose: digit-blinding is what folds a printed page number onto
    one key, and applied to prose it folds a whole judgment's paragraphs onto
    one key too - so the module allows it only on lines short enough to BE a
    head or a foot, and a fixture whose "body" is short is testing the length
    cap rather than the rule it means to.
    """
    line = f"{i + 1}. The submission advanced for the appellant on this page alone."
    assert RUNNING_DIGIT_BLIND_CHARS < len(line) <= RUNNING_MAX_CHARS
    return line


def test_a_head_on_alternate_pages_is_furniture_though_it_is_on_only_half_of_them():
    # RUNNING_FRACTION, and the old 0.6 was STRUCTURALLY UNREACHABLE. A law
    # report is printed recto/verso: the left-hand page carries the volume
    # head and the right-hand page carries the case name, so neither can
    # appear on more than about half the pages. A threshold above 0.5
    # therefore refuses BOTH by construction - and the pass reported success
    # by removing nothing. Measured: running heads survived in all ten
    # emitted bodies, up to 31 in one document.
    verso = "1096 SUPREME COURT REPORTS [2010] 10 S.C.R."
    recto = "KALYANI SHARMA v. STATE OF MAHARASHTRA AND ORS. 1097"
    pages = []
    for i in range(6):
        head = verso if i % 2 == 0 else recto
        pages.append(f"{head}\n{_body_line(i)}\n")
    cleaned, stats = clean_pages(pages)
    joined = "\n".join(cleaned)

    assert "SUPREME COURT REPORTS" not in joined
    assert "KALYANI SHARMA" not in joined
    assert stats["running"] == 6
    # THE PREMISE: each head really is on HALF the pages and no more, so a
    # threshold set for "most pages" could never have reached either.
    assert sum(1 for page in pages if verso in page) == 3
    assert sum(1 for page in pages if recto in page) == 3
    assert 3 / len(pages) == 0.5
    # ... and the body lines, which differ only in a digit, are still there.
    for i in range(6):
        assert _body_line(i) in joined


def test_a_running_head_the_scanner_spelled_three_ways_is_still_one_line():
    # WHAT THE FRACTION ALONE COULD NOT FIX. The 2010-2017 volumes are scans
    # with an OCR text layer, and the one line that repeats on every left-hand
    # page arrives under several spellings of the same thing: the year's `0`
    # read as the letter `O`, the closing bracket read as `)`, a space the
    # scanner put inside the year. Digit-blinding alone leaves those as three
    # different keys, none of which reaches any threshold - so the head
    # survives on a document where it is on every other page.
    spellings = [
        "326 SUPREME COURT REPORTS [201 O] 1 S.C.R.",
        "328 SUPREME COURT REPORTS [2010] 1 S.C.R.",
        "330 SUPREME COURT REPORTS (2010) 1 S.C.R.",
    ]
    pages = [f"{spellings[i % 3]}\n{_body_line(i)}\n" for i in range(6)]
    cleaned, stats = clean_pages(pages)
    joined = "\n".join(cleaned)

    assert "SUPREME COURT REPORTS" not in joined
    assert stats["running"] == 6
    # THE PREMISE: no two of the three spellings are the same string, and
    # each is on only a third of the pages - so nothing but folding them onto
    # one key can have carried any of them over a threshold.
    assert len(set(spellings)) == 3
    for spelling in spellings:
        assert sum(1 for page in pages if spelling in page) == 2
    for i in range(6):
        assert _body_line(i) in joined


def test_a_bare_paragraph_number_is_never_furniture_however_many_pages_carry_one():
    # THE PRICE OF THE HARD KEY, guarded rather than paid. Once the key drops
    # punctuation, `48.` and the printed page number `923` are both `#` - and
    # printed page numbers are on every page, so the merged key clears any
    # threshold and takes the paragraph anchor with it. That anchor is the one
    # corpus-wide Tier-1 segmentation signal there is, and its deletion would
    # be silent.
    pages = [f"{12 + i}.\n{_body_line(i)}\n{941 + i}\n" for i in range(6)]
    cleaned, stats = clean_pages(pages)
    joined = "\n".join(cleaned)

    for i in range(6):
        assert f"\n{12 + i}.\n" in f"\n{joined}\n"
    # THE PREMISE and the control in one: the printed page numbers, which
    # differ from the enumerators only in the punctuation the key drops, DO
    # go - so the pass really did fire on this document and the enumerators
    # survived it rather than surviving a pass that did nothing.
    assert stats["running"] == 6
    for i in range(6):
        assert f"\n{941 + i}\n" not in f"\n{joined}\n"


def test_a_numbers_only_line_is_not_folded_onto_the_printed_page_number():
    # The other half of the same guard, and a measured one: the reader broke
    # `(1974)` out of a citation onto its own line, and a key that folded its
    # bracket away would have put it under `#` with every page number in the
    # volume - which is enough of them to cross any threshold. A line with no
    # letters in it has no spelling for a scanner to disagree about, so it
    # keeps the punctuation that tells it from a page number.
    pages = [f"{_body_line(i)}\n" + ("1974)\n" if i == 2 else "") + f"{941 + i}\n"
             for i in range(6)]
    cleaned, stats = clean_pages(pages)
    joined = "\n".join(cleaned)

    assert "1974)" in joined
    assert stats["running"] == 6
    for i in range(6):
        assert f"\n{941 + i}\n" not in f"\n{joined}\n"


_AWARD_TABLE = (
    "The sums worked out under each head were as follows:",
    "Loss of dependency",
    "1250000",
    "Funeral expenses",
    "25000",
    "Loss of consortium",
    "40000",
)


def _award_pages(*, table_at_foot: bool) -> list[str]:
    """Six pages, a printed page number at the head and foot of each.

    One page also carries an award table whose figures the reader put on
    their own lines - which is what it does with a grid. `table_at_foot`
    puts that table where the page's foot number would have been, so the
    last of its figures sits at the page edge instead of inside the page.
    """
    pages = []
    for i in range(6):
        lines = [f"{600 + i}", _body_line(i)]
        if i != 3:
            lines.append(f"{600 + i}")
        elif table_at_foot:
            lines.extend(_AWARD_TABLE)
        else:
            lines.extend([*_AWARD_TABLE, f"{600 + i}"])
        pages.append("\n".join(lines) + "\n")
    return pages


def test_a_figure_in_a_table_is_not_furniture_though_it_shares_the_page_numbers_key():
    # WHAT THE DIGIT SKELETON CANNOT SEE. `\d+` matches a whole run, so every
    # bare integer in the document has the same key: `1250000`, `25000` and
    # `1095` are all `#`. The printed page number alone puts `#` over the
    # threshold on any document long enough for this pass to run at all - so
    # from there EVERY numbers-only line in the document is furniture by key,
    # and an award table loses its figures while their labels stay behind.
    # That is the silent corruption of emitted text this pass exists to
    # prevent, arriving through the pass itself.
    cleaned, stats = clean_pages(_award_pages(table_at_foot=False))
    joined = "\n".join(cleaned)

    for figure in ("1250000", "25000", "40000"):
        assert f"\n{figure}\n" in f"\n{joined}\n"
    for label in ("Loss of dependency", "Funeral expenses", "Loss of consortium"):
        assert label in joined
    # THE PREMISE, and the reason this is not a test of a pass that did
    # nothing: the printed page numbers - head and foot, twelve of them, the
    # same shape and the same key as the three figures - all go.
    assert stats["running"] == 12
    for i in range(6):
        assert f"\n{600 + i}\n" not in f"\n{joined}\n"


def test_the_same_figure_at_the_foot_of_the_page_is_still_furniture():
    # THE OTHER SIDE of the rule above, without which it would be "a number
    # is never furniture" and the printed page numbers would survive in every
    # body. Position is the whole of the evidence: the identical line, one
    # place further down the same page, is a page number as far as anything
    # here can tell - so it goes, while the two figures above it stay.
    cleaned, stats = clean_pages(_award_pages(table_at_foot=True))
    joined = "\n".join(cleaned)

    assert "\n40000\n" not in f"\n{joined}\n"
    assert "\n1250000\n" in f"\n{joined}\n" and "\n25000\n" in f"\n{joined}\n"
    # THE PREMISE: the two documents differ in nothing but where the table
    # sits, `40000` is the same string in both, and the eleven printed page
    # numbers still go - so the twelfth removal here is the figure.
    assert _AWARD_TABLE[-1] == "40000"
    assert stats["running"] == 12
    for i in range(6):
        assert f"\n{600 + i}\n" not in f"\n{joined}\n"


def _closing_line(i: int) -> str:
    """A second body line per page, unique and past the digit-blind cap."""
    line = f"{i + 1}. That figure was not disputed by either side in this appeal."
    assert RUNNING_DIGIT_BLIND_CHARS < len(line) <= RUNNING_MAX_CHARS
    return line


def test_a_table_running_across_pages_cannot_vote_its_own_figures_into_furniture():
    # THE COUNTING SIDE of the same rule, and the reason a key is registered
    # only where it may be USED. A schedule of damages spanning six pages
    # puts a bare figure on every page; if those figures could be counted
    # where they can never be deleted, they would carry `#` over the
    # threshold themselves - and then the one cell that happens to fall at a
    # page edge is deleted as a printed page number, by a key its own table
    # armed. Nothing here is furniture and nothing is removed.
    figures = [str(125000 + i * 1000) for i in range(6)]
    pages = []
    for i in range(6):
        cell = figures[i]
        # The last page is the one that matters: its figure sits where a
        # printed page number would.
        rows = [_body_line(i), cell, _closing_line(i)] if i < 5 else [
            _body_line(i), _closing_line(i), cell
        ]
        pages.append("\n".join(rows) + "\n")
    cleaned, stats = clean_pages(pages)
    joined = "\n".join(cleaned)

    assert stats["running"] == 0
    for figure in figures:
        assert f"\n{figure}\n" in f"\n{joined}\n"
    # THE PREMISE: the six figures really are ONE key - they differ only in
    # digits, and the digit-blind key is what the printed page number needs -
    # so counting them where they sit would have been six pages' worth, past
    # the threshold for a six-page document.
    from tuned.data.extract import _running_key

    assert len({_running_key(f) for f in figures}) == 1
    assert 6 >= max(2, int(6 * 0.4 + 0.999999))


def test_a_page_number_in_brackets_does_not_take_the_citation_year_with_it():
    # THE CLASS THIS RULE COVERS is "a line with numbers and no words", not
    # "a line of bare digits". The letterless key keeps punctuation, so
    # `(1974)` broken out of a citation is safe from a volume that prints
    # `1095` at the foot - but not from one that prints `(1095)`, where both
    # are `(#)` and the page number is on every page. Position is then the
    # only thing left, and it is enough.
    pages = [f"{_body_line(i)}\n" + ("(1974)\n" if i == 2 else "") + f"({1095 + i})\n"
             for i in range(6)]
    cleaned, stats = clean_pages(pages)
    joined = "\n".join(cleaned)

    assert "(1974)" in joined
    # THE PREMISE and the control: the bracketed page numbers share that key
    # exactly and they all go, so the citation year survived the pass rather
    # than survived a pass that did nothing. (Four digits because a bracketed
    # THREE-digit number is the bare-enumerator shape and is excluded from
    # furniture before any of this - and S.C.R. page numbers do reach four.)
    from tuned.data.extract import _running_key

    assert _running_key("(1974)") == _running_key("(1095)") == "(#)"
    assert stats["running"] == 6
    for i in range(6):
        assert f"\n({1095 + i})\n" not in f"\n{joined}\n"


def test_the_scr_margin_letters_go_and_a_lettered_heading_stays():
    # P0 found "A B C D E F G H" running down the left margin of these
    # reprints - and lettered section headings in the same alphabet.
    page = "A\nB.\nA. FACTUAL MATRIX\n1. The appellant filed a writ petition.\nH\n"
    cleaned, stats = clean_pages([page])

    assert "A. FACTUAL MATRIX" in cleaned[0]
    assert "1. The appellant filed a writ petition." in cleaned[0]
    assert [line for line in cleaned[0].split("\n") if line.strip() in {"A", "B.", "H"}] == []
    assert stats["margin_letter"] == 3


@pytest.mark.parametrize(
    "line,signal",
    [
        ("C Held:", "held"),
        ("A Case Law Reference:", "case_law_reference"),
        ("D Cases referred to:", "cases_referred_to"),
    ],
)
def test_an_inlined_margin_letter_does_not_hide_a_signature(line, signal):
    # THE FAILURE THIS ANSWERS, and it was measured rather than imagined: on a
    # reader that lays the page out in columns the print-alignment letter
    # arrives INSIDE the line instead of on one of its own, and every
    # `^`-anchored signature then reads `C Held:` as prose. One real document
    # printed `headnote signals: none` over exactly that text - i.e. this
    # module's own first-run alarm, firing, and reading like a clean
    # judgment. A reader upgrade is how that arrives silently.
    assert signal in headnote_signals(line)
    # THE PREMISE: without the letter the guard already saw it, so the letter
    # is the whole of what was hiding it.
    assert signal in headnote_signals(line[2:])


def test_the_margin_letter_is_taken_off_the_matching_form_and_never_off_the_text():
    # `A` is an English word. Stripping a leading capital from the emitted
    # text would eat it out of every sentence that begins "A person who..." -
    # so the letter comes off the forms the guard MATCHES against and off
    # nothing else. The cost of being wrong is then nil: a matching form that
    # reads "person who..." simply fails to match, as it did before.
    page = "A person who commits an offence under this section shall be liable.\n"
    cleaned, stats = clean_pages([page])

    assert "A person who commits an offence" in cleaned[0]
    assert stats["margin_letter"] == 0
    assert headnote_signals(page) == ()


_PAGE_FILLER = (
    "The submission was pressed with some vigour, and we have considered it against "
    "the material on the record, which we now set out at the length the argument "
    "deserves before turning to the authorities relied upon on either side. " * 3
)


def test_a_footnote_at_the_foot_of_a_page_stops_splitting_a_sentence():
    pages = [
        f"12. The question is whether the delay stands explained on this record. {_PAGE_FILLER}\n"
        "13. We turn to the evidence bearing on it, which begins with the deposition of\n"
        "1 State of Punjab v. Gurmit Singh, (1996) 2 SCC 384, at paragraph 21.\n",
        "PW-1, who spoke of the events of that night.\n",
    ]
    cleaned, stats = clean_pages(pages)

    assert stats["footnote_pages"] == 1
    # The sentence now runs on across the page break instead of being cut in
    # half by the citation.
    assert cleaned[0].rstrip().endswith("the deposition of")
    # The footnote is not discarded - it is carried out WITH THE PAGE INDEX
    # it came off, which is what lets the strip drop the ones belonging to
    # the headnote and keep the ones belonging to the judgment.
    assert stats["footnote_lines"][0][0] == 0
    assert "Gurmit Singh" in stats["footnote_lines"][0][1]


@pytest.mark.parametrize(
    "tail",
    [
        # The next numbered paragraph, which happens to end the page.
        "14. The appellant then submitted that the evidence was insufficient.\n",
        # An enumerated sub-list inside a paragraph: low number, but nothing
        # about it says "reference".
        "1. that the delay was explained on the record before the trial court.\n",
    ],
)
def test_a_numbered_paragraph_at_the_foot_of_a_page_is_never_taken_for_a_footnote(tail):
    page = (
        f"12. The question is whether the delay stands explained. {_PAGE_FILLER}\n"
        "13. We turn to the evidence bearing on it, and record our conclusion.\n" + tail
    )
    cleaned, stats = clean_pages([page])

    assert stats["footnote_pages"] == 0
    assert cleaned[0].rstrip().endswith(tail.strip())


def test_a_paragraph_below_the_footnote_block_keeps_the_whole_block_in_place():
    # Reading order is not always print order: pymupdf4llm can put the foot
    # of the page before the last text block on it. Taking everything from
    # the first footnote marker to the end of the page would then carry a
    # numbered PARAGRAPH out of the body with it - and nothing downstream
    # would ever see that it had gone.
    page = (
        f"12. The question is whether the delay stands explained. {_PAGE_FILLER}\n"
        "1 State of Punjab v. Gurmit Singh, (1996) 2 SCC 384, at paragraph 21.\n"
        "13. We turn to the evidence bearing on it, and record our conclusion.\n"
    )
    cleaned, stats = clean_pages([page])

    assert stats["footnote_pages"] == 0
    assert "13. We turn to the evidence" in cleaned[0]
    assert "Gurmit Singh" in cleaned[0]


def test_a_reference_block_that_is_most_of_the_page_is_left_where_it_is():
    # One of the two guards decision 4 names, and neither of them had a test.
    # The failure both exist to avoid is taking a numbered PARAGRAPH out of
    # the body: a "footnote block" that is most of the page is not a footnote
    # block - it is a page of citations, or the reading order coming back
    # scrambled - and moving it takes the page with it.
    head = "12. The question is whether the delay stands explained on this record.\n"
    block = (
        "1 State of Punjab v. Gurmit Singh, (1996) 2 SCC 384, at paragraph 21, where "
        "the point was considered at length and the earlier authorities reviewed.\n"
        "2 Kesavananda Bharati v. State of Kerala, (1973) 4 SCC 225, at paragraph 316.\n"
    )
    cleaned, stats = clean_pages([head + block])

    assert stats["footnote_pages"] == 0
    assert "Gurmit Singh" in cleaned[0]
    # THE PREMISE and the disjointness in one: the block satisfies every
    # OTHER condition - it reads as references, and its numbers are below the
    # paragraph above it - so the share guard is the only thing that can be
    # holding it, and the SAME block on a page long enough to make it a
    # minority is moved.
    assert len(block) > len(head + block) * FOOTNOTE_MAX_SHARE
    long_page = head + _PAGE_FILLER + _PAGE_FILLER + "\n" + block
    _, long_stats = clean_pages([long_page])
    assert len(block) < len(long_page) * FOOTNOTE_MAX_SHARE
    assert long_stats["footnote_pages"] == 1


def test_a_footnote_block_of_several_notes_is_moved_whole():
    # FOOTNOTE_WINDOW is how far up from the foot of the page the block's
    # FIRST marker is looked for, and it had no test. A window that reaches
    # only the last line moves the last note and leaves the ones above it
    # interleaved with the judgment - which is the same corruption the pass
    # exists to fix, only quieter, because the page still looks tidy.
    page = (
        f"12. The question is whether the delay stands explained. {_PAGE_FILLER}\n"
        "13. We turn to the evidence bearing on it, which begins with the deposition of\n"
        "1 State of Punjab v. Gurmit Singh, (1996) 2 SCC 384, at paragraph 21.\n"
        "2 Sharma v. State of Maharashtra, (2004) 2 SCC 1, at paragraph 9.\n"
        "3 Kesavananda Bharati v. State of Kerala, (1973) 4 SCC 225.\n"
    )
    cleaned, stats = clean_pages([page])

    assert stats["footnote_pages"] == 1
    # ALL THREE, not just the one at the foot.
    assert len(stats["footnote_lines"]) == 3
    for name in ("Gurmit Singh", "Sharma v. State", "Kesavananda"):
        assert name not in cleaned[0]
    # ... and the body ends at the sentence the notes interrupted, which is
    # what says the block was taken from the right place.
    assert cleaned[0].rstrip().endswith("the deposition of")


def test_a_number_on_an_earlier_page_cannot_make_this_page_a_footnote_block():
    # THE MISREADING, and it shipped: "paragraph numbering ascends through a
    # judgment, so the highest number seen ANYWHERE above is a safe mark to
    # compare against". Three things that are not paragraphs inflate it - a
    # wrapped sentence whose next line opens on the number of the ARTICLE it
    # was naming (`...of Article` / `335. ...`), a quoted statute, a quoted
    # paragraph of the judgment under appeal - and
    # from then on EVERY number below the inflated mark is footnote-eligible
    # for the rest of the file. Measured: paragraph 48 of one real judgment
    # was carried to the foot of the file under `[FOOTNOTES]`.
    inflating = (
        "39. The scheme is to be read alongside the duty of preserving the\n"
        "standard of the service, which is the caution sounded by Article\n"
        "335. The relaxations that it permits are meant for candidates of the\n"
        "categories it names and for nobody besides them.\n"
    )
    later = (
        f"{_PAGE_FILLER}\n"
        "48. Mere reference to the decision in Sharma v. State, (2000) 2 SCC 1,\n"
        "does not re-validate the reasoning of the Division Bench in this case.\n"
    )
    cleaned, stats = clean_pages([inflating, later])

    assert stats["footnote_pages"] == 0
    assert "48. Mere reference" in cleaned[1]
    # THE PREMISE, stated as a fact about the FIXTURE rather than as a second
    # copy of the rule: the wrapped sentence really does read as a paragraph
    # number, so the mark really would have been 335 had it been threaded,
    # and the paragraph at the foot of the later page really is below it.
    from tuned.data.extract import _PARA_LINE

    assert _PARA_LINE.match("335. The relaxations that it permits are meant for candidates")
    assert 48 < 335
    # ... and the pass is not simply switched off: the SAME page numbers past
    # a real note, and the note goes.
    within = (
        f"{_PAGE_FILLER}\n"
        "48. Mere reference to the earlier decision does not re-validate the\n"
        "reasoning of the Division Bench in this case, as we have explained.\n"
        "1 Sharma v. State of Maharashtra, (2000) 2 SCC 1, at paragraph 34.\n"
    )
    kept, within_stats = clean_pages([inflating, within])
    assert within_stats["footnote_pages"] == 1
    assert "Sharma v. State of Maharashtra" not in kept[1]
    assert "48. Mere reference" in kept[1]


_SEVERED_HEAD = (
    "72. Similarly, the Division Bench in Mehta v. Union of India, AIR 1955\n"
    "Bom 113, took the view that the earlier ruling cannot serve as a useful\n"
    "guide on this question. The observations relied upon read as follows:\n"
    f"{_PAGE_FILLER}{_PAGE_FILLER}\n"
)
_SEVERED_BLOCK = (
    "8. Where a decree is one for the payment of money, as Mehta v. Union,\n"
    "(1889) 13 Bom 241, explains, and an appeal against it is lodged by the\n"
    "party who has been directed to pay, the court below stays execution so\n"
    "far as it directs payment, and on an application by that party it\n"
)


@pytest.mark.parametrize(
    "tail",
    [
        # The measured one: the page ran out in the middle of a clause and
        # the last word is an ordinary lower-case word.
        "may do so once the party liable brings the sum into the registry, unless\n",
        # The other shape a wrapped legal sentence ends on, which the
        # lower-case limb cannot see: a citation, and then the comma that
        # says the sentence has not finished.
        "may do so on the terms settled in Mehta v. Union, (2005) 4 SCC 1,\n",
    ],
)
def test_a_quoted_paragraph_the_page_break_severed_is_not_taken_for_a_footnote(tail):
    # The failure PER-PAGE NUMBERING CANNOT SEE, and the reason the two rules
    # are not one: here the inflating number is a real paragraph (`72.`) on
    # the SAME page, and the block under it is a quoted paragraph of an old
    # report whose sentence runs on to the next page. Measured: seven lines
    # of a judgment moved to the foot of the file, cut mid-clause on a
    # lower-case word (the fixture's wording is invented, the shape is the
    # one that was measured).
    severed = _SEVERED_BLOCK + tail
    page = _SEVERED_HEAD + severed
    cleaned, stats = clean_pages([page])

    assert stats["footnote_pages"] == 0
    assert "Where a decree is one for the payment of money"in cleaned[0]
    # THE PREMISE, and the disjointness with every other condition: the block
    # satisfies all of them - its number is below the paragraph above it ON
    # THIS PAGE, it reads as a reference, and it is a small minority of the
    # page - so the trailing-off guard is the only thing that can be holding
    # it. The proof is the SAME block, finished, which goes.
    assert 8 < 72
    assert len(severed) < len(page) * FOOTNOTE_MAX_SHARE
    finished = _SEVERED_BLOCK + "may do so on terms it thinks fit to impose.\n"
    moved, moved_stats = clean_pages([_SEVERED_HEAD + finished])
    assert moved_stats["footnote_pages"] == 1
    assert "Where a decree is one for the payment of money"not in moved[0]


def test_a_document_too_short_for_most_pages_to_mean_anything_keeps_its_repeated_lines():
    # RUNNING_MIN_PAGES, untested. "Furniture on MOST pages" says nothing
    # over two pages, and the deletion it licenses is silent: a two-page
    # judgment whose pages share a short line - a continued heading, a party
    # name - would lose it with no record anywhere that it had been there.
    short = [
        f"{RUNNING_HEADER}\n1. The appellant filed a writ petition in the High Court.\n",
        f"{RUNNING_HEADER}\n2. The respondent answered it on affidavit.\n",
    ]
    cleaned, stats = clean_pages(short)

    assert RUNNING_HEADER in "\n".join(cleaned)
    assert stats["running"] == 0
    # THE CONTROL: the same line over enough pages IS furniture and does go.
    # Without it this test would pass on a pass that removed nothing at all.
    longer = short + [
        f"{RUNNING_HEADER}\n3. The High Court heard them at some length.\n",
        f"{RUNNING_HEADER}\n4. We now record our conclusion on the point.\n",
    ]
    cleaned_longer, longer_stats = clean_pages(longer)
    assert RUNNING_HEADER not in "\n".join(cleaned_longer)
    assert longer_stats["running"] == 4


def test_footnotes_from_the_headnote_pages_do_not_survive_the_strip():
    # Footnotes are moved to the END of the document, which carries them
    # ACROSS the headnote boundary - so the editorial front matter's own
    # numbered references would come back after being stripped, and the
    # licensing problem would return by another road.
    pages = [
        HEADNOTE + "\n1 Editorial reference, (2011) 4 SCC 707, noted by the reporter.\n",
        "The Judgment of the Court was delivered by\nNAVIN SINHA, J.\n"
        + _pad(BODY.split("\n", 2)[2], 1400)
        + "\n2 Boundary page note, (2004) 2 SCC 1, on the same page as the cut.\n",
        _paras(9, 1400) + "\n3 Kesavananda Bharati v. State of Kerala, (1973) 4 SCC 225.\n",
    ]
    cleaned, stats = clean_pages(pages)
    result = extract_text(pages)

    # THE PREMISE, without which this test asserts nothing: all three notes
    # really are detected and moved out of their pages. A fixture in which
    # nothing is moved would pass every assertion below while the page
    # filter did no work at all.
    assert stats["footnote_pages"] == 3
    assert result.ok, result.reason
    assert result.boundary_page == 1

    assert "Editorial reference" not in result.text
    assert "Kesavananda Bharati" in result.text
    assert result.text.count(FOOTNOTE_HEADING) == 1
    assert result.footnotes == 1
    # The documented cost of the rule: the boundary page carries the tail of
    # the headnote AND the head of the judgment, so its notes are dropped
    # too. Losing one body footnote is the cheaper error.
    assert "Boundary page note" not in result.text


# --------------------------------------------------------------------------
# Metadata captured while the page is open.
# --------------------------------------------------------------------------

def test_the_reportable_flag_is_captured_and_non_reportable_is_not_read_as_reportable():
    assert reportable_flag("REPORTABLE\nIN THE SUPREME COURT OF INDIA") == "REPORTABLE"
    assert reportable_flag("NON-REPORTABLE\nIN THE SUPREME COURT") == "NON-REPORTABLE"
    assert reportable_flag("NONREPORTABLE\nIN THE SUPREME COURT") == "NON-REPORTABLE"
    assert reportable_flag("[2020] 7 S.C.R. 941\nKALYANI SHARMA") is None
    # P0 found the flag in 2 of 70 documents: it is metadata, and a
    # selection built on it would select nothing. Reading it out of the
    # BODY - where "reportable" appears in prose - would invent it.
    assert reportable_flag("\n".join(["x"] * 40) + "\nREPORTABLE") is None


def test_the_page_span_comes_out_of_the_object_key():
    span = page_span_from_key("data/pdf/year=2020/english/2020_7_941_960_EN.pdf")
    assert (span.year, span.volume, span.start, span.end) == (2020, 7, 941, 960)
    assert span.pages == 20
    assert page_span_from_key("data/pdf/year=2020/english/2020_7_941_EN.pdf") is None
    assert page_span_from_key("data/pdf/year=2020/regional/2020_hi_7_941_960.pdf") is None
    assert page_span_from_key("metadata/parquet/year=2020/part-0.parquet") is None
    # _EN is REQUIRED, not decoration. v1 is english-only (the prefix is the
    # language ground truth), and a regional object that happens to share
    # the numeric convention must not be read as one of ours.
    assert page_span_from_key("data/pdf/year=2020/regional/2020_7_941_960_HI.pdf") is None
    # An inverted span is a key nobody can read - `pages` would be negative
    # and the citation's page anchor would point backwards - so it does not
    # parse. The price, stated here so that it is found by reading rather
    # than by a hole in the corpus: that object drops out of the prefix index
    # too, and if the metadata carries no pdf_key column it can then only
    # ever join as `unmatched`.
    assert page_span_from_key("data/pdf/year=2020/english/2020_7_960_941_EN.pdf") is None


def test_markdown_decoration_is_demoted_without_eating_the_filename_underscores():
    assert demote_markdown("**HELD:** 1. The appeal") == "HELD: 1. The appeal"
    assert demote_markdown("# J U D G M E N T") == "J U D G M E N T"
    assert demote_markdown("*emphasis* here") == "emphasis here"
    # The S.C.R. object stem is all underscores and must survive intact,
    # which an italic rule written as a bare "_" strip would not leave it.
    assert demote_markdown("2020_7_941_960_EN.pdf") == "2020_7_941_960_EN.pdf"
    assert demote_markdown("-----") == ""


@pytest.mark.parametrize(
    "row,alternative",
    [
        # A leading bar with NOTHING to close it - the shape a reader emits
        # when the table's right edge ran off the detected region. One bar, so
        # the two-bar alternative cannot see it.
        ("|Case Law Reference", "leading_bar"),
        # Two bars and no leading one - a row the reader indented past the
        # three columns the leading-bar alternative allows, or one it never
        # opened. No bar in column 0, so that alternative cannot see it.
        ("     Case Law | Reference | para 12", "two_bars"),
    ],
)
def test_a_table_row_is_demoted_in_both_shapes_the_reader_can_emit(row, alternative):
    # `_MD_TABLE_ROW` is a union of two alternatives and the `md_table`
    # fixture satisfies BOTH, so either could be deleted with the suite green.
    # These are the rows only one of them can carry - and a bar left in
    # column 0 is the difference between refusing a document and publishing
    # the reporter's Case Law Reference table.
    assert "|" not in demote_markdown(row)
    assert "Case Law" in demote_markdown(row)
    # THE PREMISE: this row has the shape only ONE alternative can read, and
    # it is stated as a fact about the ROW rather than as a second copy of the
    # rule - a test that re-implements the pattern it is testing passes when
    # the two copies drift together and says nothing when they drift apart.
    assert _MD_TABLE_ROW.search(row) is not None
    if alternative == "leading_bar":
        assert row.lstrip(" \t").startswith("|") and row.count("|") == 1
    else:
        assert not row.lstrip(" \t").startswith("|") and row.count("|") == 2
    # ... and the whole point of the rule, which "no bars left" does not make
    # on its own: the furniture under the bar is visible to the guard again.
    assert headnote_signals(demote_markdown(row)) == ("case_law_reference",)


# ------------------------------------------------ the reader's inline HTML

@pytest.mark.parametrize(
    "line",
    [
        # The four this corpus actually produced, in the counts measured on
        # the shipped read: 878 `<u>`, 164 `<br>`, 164 `<sup>`, 10 `<mark>`.
        # (The smoke's 886/222 was the same corpus read with the layout
        # engine on - see READER_LAYOUT.)
        "<u>Case Law Reference</u>",
        "<mark>Case Law Reference</mark>",
        "<sup>Case Law Reference</sup>",
        # ...and the same class of thing under other names, with attributes,
        # in the other case, and behind the markdown that wraps it.
        '<span class="hdr">Case Law Reference</span>',
        "<B>Case Law Reference</B>",
        "**<u>Case Law Reference</u>**",
    ],
)
def test_an_inline_tag_does_not_hide_the_editorial_furniture(line):
    # THE MISREADING: "the reader emits markdown, so demoting markdown is
    # enough". It emits HTML too, and a tag sits in front of the first word
    # of a line exactly the way a table bar does - so a `^`-anchored
    # signature reads `<u>Case Law Reference</u>` as prose. Measured: the
    # guard was blind to `case_law_reference` in five of fifteen real
    # documents for this reason alone.
    assert headnote_signals(line) == ("case_law_reference",)
    # THE PREMISE: the tag really is in column 0, so nothing but the strip
    # can be what let the anchor through.
    assert line.startswith("<") or line.startswith("**<")


def test_the_line_break_tag_becomes_a_space_and_does_not_glue_two_words():
    # `<br>` is the one tag that carries meaning: it is the line break
    # INSIDE a table cell, so deleting it outright welds the words on either
    # side of it together - `Sub<br>Inspectors` -> `SubInspectors`, in real
    # emitted text. A newline instead would tear the `|cell|cell|` row in
    # half before the table rule can read it, which is why it is a space.
    assert demote_markdown("Sub<br>Inspectors") == "Sub Inspectors"
    row = "|2.<br>|Selection of candidates as Sub<br>Inspectors|168|"
    demoted = demote_markdown(row)
    assert "SubInspectors" not in demoted
    assert "Sub Inspectors" in demoted
    assert "|" not in demoted


def test_the_tags_do_not_survive_into_the_emitted_judgment():
    # They leaked into six of ten emitted training texts, 326 `<u>` in one.
    pages = scr_pages(
        body="<u>" + BODY.replace("NAVIN SINHA, J.", "<u>NAVIN SINHA, J.</u>") + "</u>"
    )
    result = extract_text(pages)

    assert result.ok, result.reason
    assert "<u>" not in result.text and "</u>" not in result.text
    assert "NAVIN SINHA, J." in result.text


def test_a_heading_with_a_footnote_dagger_after_it_is_still_a_heading():
    # THE MISREADING the `$` anchor made: the 2023+ volumes print
    # `Headnotes †`, the dagger footnoting the editor's name, and a heading
    # anchored at the end of the line reads that as prose. It cost the
    # `headnotes` signature on every 2023+ document measured.
    assert "headnotes" in headnote_signals("**Headnotes** <sup>**†**</sup>")
    assert "headnotes" in headnote_signals("Headnotes †")
    assert "cases_referred_to" in headnote_signals("Cases Referred To: †")
    # ...and the anchor still does its job: only NON-WORD characters are
    # tolerated, so a heading followed by a WORD is the sentence the anchor
    # was put there to exclude. `Headnotes prepared by: <editor>` is a real
    # line of these volumes - at the FOOT of the document, where reading it
    # as furniture would put a signature past the cut that the removed head
    # cannot account for, and quarantine the judgment.
    assert "headnotes" not in headnote_signals("†Headnotes prepared by: A. Editor")
    assert "cases_referred_to" not in headnote_signals("Cases referred to in argument were")


# --------------------------------------------------------------------------
# The join, the resume decision, and the run.
# --------------------------------------------------------------------------

import json
import os
import sys
import types
from pathlib import Path

from pipeline_fakes import temp_config

from tuned.data.acquire import SC_LICENSE, SC_SOURCE_ID
from tuned.data.extract import (
    AUDIT_REMOVED_CHARS,
    EXTRACT_VERSION,
    PART_SUFFIX,
    ROUTE_AMBIGUOUS,
    ROUTE_PDF_KEY,
    ROUTE_SCR_PREFIX,
    ROUTE_UNMATCHED,
    STATUS_OK,
    STATUS_QUARANTINED,
    ExtractionError,
    audit_report,
    extract_corpus,
    extract_decision,
    main,
    pdf_index,
    read_pdf_pages,
    reader_fingerprint,
    resolve_pdf,
    audit_sample,
    spread,
    text_path_for,
    write_manifest,
    write_text,
)
from tuned.data.store import Store

EXTRACT_SRC = Path(__file__).parent.parent / "src" / "tuned" / "data" / "extract.py"


class FakeReader:
    """Stands in for pymupdf4llm: path -> pages, and it remembers the order."""

    def __init__(self, docs: dict, *, fail: tuple = ()):
        self.docs = {str(k): v for k, v in docs.items()}
        self.fail = {str(f) for f in fail}
        self.read: list[str] = []

    def __call__(self, path):
        path = str(path)
        self.read.append(path)
        if path in self.fail:
            raise ExtractionError(f"cannot parse {path}")
        return self.docs[path]


class DocumentIndexFailsAt:
    """Store proxy whose Nth record_document raises - the process dying
    between the text landing and the row that points at it."""

    def __init__(self, store, at: int = 1):
        self._store = store
        self._at = at
        self.calls = 0

    def __getattr__(self, name):
        attr = getattr(self._store, name)
        if name != "record_document":
            return attr

        def recording(*args, **kwargs):
            self.calls += 1
            if self.calls >= self._at:
                raise RuntimeError("index write died")
            return attr(*args, **kwargs)

        return recording


def _key(year=2015, volume=1, start=1, end=20) -> str:
    return f"data/pdf/year={year}/english/{year}_{volume}_{start}_{end}_EN.pdf"


def _selection(key=None, **over) -> dict:
    row = {
        "case_id": "C.A. 3221/2018",
        "title": "Kalyani Sharma v. State of Maharashtra",
        "citation": "[2015] 1 S.C.R. 1",
        "year": 2015,
        "court": "Supreme Court of India",
        "coram": 3,
        "case_type": "civil",
        "signals": ["citation"],
        "priority": 4,
        "scr_prefix": "2015_1_1_",
        "pdf_key": key,
        "source_id": SC_SOURCE_ID,
    }
    row.update(over)
    return row


def _corpus(tmp_path, keys, *, store=None):
    """A store whose artifact index holds `keys`, with a PDF on disk for each."""
    store = store or Store.open(tmp_path / "state" / "law_v1.sqlite3")
    store.upsert_source(SC_SOURCE_ID, SC_LICENSE)
    paths = {}
    for key in keys:
        local = tmp_path / "corpus" / "sc" / key
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(b"%PDF-1.7 not really")
        store.record_artifact(
            SC_SOURCE_ID, key, local_path=local, size_bytes=local.stat().st_size, sha256="aa"
        )
        paths[key] = str(local)
    return store, paths


@pytest.fixture
def store(tmp_path):
    with Store.open(tmp_path / "state" / "law_v1.sqlite3") as s:
        yield s


# ------------------------------------------------------------------ the join

def test_the_join_prefers_the_metadata_link_and_falls_back_to_the_citation_prefix(
    tmp_path, store
):
    linked, inferred = _key(start=1), _key(start=101, end=140)
    _corpus(tmp_path, [linked, inferred], store=store)
    index = pdf_index(store)

    # Route 1: the metadata carried a PDF link and acquire fetched it.
    assert resolve_pdf(_selection(linked), index) == (linked, ROUTE_PDF_KEY)
    # Route 2: no link, but the S.C.R. citation addresses the filename -
    # which is the only join available if the metadata has no link column.
    by_citation = _selection(None, scr_prefix="2015_1_101_")
    assert resolve_pdf(by_citation, index) == (inferred, ROUTE_SCR_PREFIX)
    # And the prefix really is doing the work: it picks the OTHER file.
    assert inferred != linked


def test_a_link_to_an_object_that_was_never_fetched_falls_back_to_the_prefix(
    tmp_path, store
):
    # A --limit'ed or interrupted acquire leaves rows whose metadata link
    # points at an object that is not on disk. Failing there would strand
    # every such judgment even when the citation prefix finds it.
    present = _key(start=101, end=140)
    _corpus(tmp_path, [present], store=store)
    index = pdf_index(store)

    row = _selection(
        "data/pdf/year=2015/english/2015_1_9999_9999_EN.pdf", scr_prefix="2015_1_101_"
    )
    assert resolve_pdf(row, index) == (present, ROUTE_SCR_PREFIX)


def test_a_row_with_no_usable_join_is_reported_not_guessed(tmp_path, store):
    _corpus(tmp_path, [_key()], store=store)
    index = pdf_index(store)

    assert resolve_pdf(_selection(None, scr_prefix=None), index) == (None, ROUTE_UNMATCHED)
    assert resolve_pdf(_selection(None, scr_prefix="2015_9_9_"), index) == (
        None,
        ROUTE_UNMATCHED,
    )


def test_an_ambiguous_prefix_resolves_to_nothing(tmp_path, store):
    # Two objects addressed by one citation prefix: extracting "whichever
    # sorts first" would attach a judgment to the wrong citation, and the
    # error would be invisible from here on.
    a, b = _key(start=1, end=20), _key(start=1, end=40)
    _corpus(tmp_path, [a, b], store=store)

    assert resolve_pdf(_selection(None), pdf_index(store)) == (None, ROUTE_AMBIGUOUS)


# ------------------------------------------------------- the resume decision

def test_a_document_already_extracted_at_this_version_is_skipped(tmp_path):
    text = tmp_path / "a.txt"
    text.write_text("body", encoding="utf-8")
    row = {"status": STATUS_OK, "text_path": str(text), "extract_version": EXTRACT_VERSION}
    assert extract_decision(row, text) == "skip"
    assert extract_decision(row, text, force=True) == "extract"


def test_a_document_with_no_row_is_extracted(tmp_path):
    assert extract_decision(None, tmp_path / "absent.txt") == "extract"


def test_a_document_whose_text_file_vanished_is_extracted_again(tmp_path):
    # The crash window: the row landed, the file did not (or was deleted).
    row = {"status": STATUS_OK, "text_path": "gone.txt", "extract_version": EXTRACT_VERSION}
    assert extract_decision(row, tmp_path / "gone.txt") == "extract"


def test_a_document_extracted_under_older_rules_is_extracted_again(tmp_path):
    text = tmp_path / "a.txt"
    text.write_text("body", encoding="utf-8")
    row = {"status": STATUS_OK, "text_path": str(text), "extract_version": EXTRACT_VERSION - 1}
    assert extract_decision(row, text) == "extract"


def test_a_quarantined_document_is_not_re_attempted_at_the_same_version(tmp_path):
    # Deterministic rules over unchanged bytes give the same refusal, so
    # re-reading it every run would spend the whole corpus's time on the
    # documents that cannot be used.
    row = {"status": STATUS_QUARANTINED, "text_path": None, "extract_version": EXTRACT_VERSION}
    assert extract_decision(row, tmp_path / "never.txt") == "skip"
    assert extract_decision(row, tmp_path / "never.txt", force=True) == "extract"
    older = dict(row, extract_version=EXTRACT_VERSION - 1)
    assert extract_decision(older, tmp_path / "never.txt") == "extract"


# --------------------------------------------------------------- durability

def test_the_text_file_is_whole_or_absent_never_a_prefix(tmp_path, monkeypatch):
    # THE FAILURE HAS TO HAPPEN DURING THE WRITE. Killing the rename instead
    # is a test that cannot observe what it is about: after `os.replace` the
    # only moment a prefix could have been visible at `path` has already
    # passed, so a writer that streams straight into `path` passes it - and
    # then the cleanup (`part.unlink()`, where `part` IS `path` under that
    # mutation) deletes the evidence and both assertions come out true.
    dest = tmp_path / "out" / "a.txt"
    part = dest.with_name(dest.name + PART_SUFFIX)
    text = "The Judgment of the Court was delivered by\n" * 200
    payload = text.encode("utf-8")
    at_failure = {}

    def dying_fsync(fd):
        # The process dies with the buffer half flushed: what is on disk at
        # this instant is a strict PREFIX of a judgment.
        os.ftruncate(fd, len(payload) // 3)
        at_failure["part_size"] = part.stat().st_size if part.exists() else None
        at_failure["dest_exists"] = dest.exists()
        raise OSError("killed part of the way through the write")

    monkeypatch.setattr(os, "fsync", dying_fsync)
    with pytest.raises(OSError):
        write_text(dest, text)

    # THE PREMISE, without which the assertions below are about nothing: a
    # prefix of the judgment really was on disk at the moment of failure.
    assert at_failure["part_size"] is not None
    assert 0 < at_failure["part_size"] < len(payload)
    # THE POINT: it was never under the name a reader looks at. This is the
    # assertion a non-atomic writer fails, and it can only be made while the
    # failure is still in flight.
    assert at_failure["dest_exists"] is False
    assert not dest.exists()
    # ... and the partial does not survive its own failure, where a later
    # run could find it beside the real name.
    assert list(dest.parent.glob("*" + PART_SUFFIX)) == []


def test_the_index_row_is_written_after_the_text_so_a_crash_costs_no_work(tmp_path, store):
    key = _key()
    _, paths = _corpus(tmp_path, [key], store=store)
    reader = FakeReader({paths[key]: scr_pages()})
    proxy = DocumentIndexFailsAt(store, at=1)

    with pytest.raises(RuntimeError):
        extract_corpus(
            proxy,
            [_selection(key)],
            index=pdf_index(store),
            text_root=tmp_path / "text",
            reader=reader,
        )

    # The text is durable even though nothing points at it yet.
    dest = text_path_for(tmp_path / "text", key)
    assert dest.read_text(encoding="utf-8").startswith("The Judgment of the Court")
    assert store.document(SC_SOURCE_ID, key) is None
    # The next run finds a key with no row and does the work again - which
    # is why there is no adopt path here.
    assert extract_decision(store.document_index(SC_SOURCE_ID).get(key), dest) == "extract"


def test_a_document_that_becomes_quarantined_loses_its_stale_text(tmp_path, store):
    key = _key()
    _, paths = _corpus(tmp_path, [key], store=store)
    dest = text_path_for(tmp_path / "text", key)

    good = FakeReader({paths[key]: scr_pages()})
    extract_corpus(
        store,
        [_selection(key)],
        index=pdf_index(store),
        text_root=tmp_path / "text",
        reader=good,
    )
    assert dest.exists()

    # Same object, rules that now refuse it (here: the marker line gone).
    unsegmentable = FakeReader({paths[key]: scr_pages(body=BODY.split("\n", 1)[1])})
    stats = extract_corpus(
        store,
        [_selection(key)],
        index=pdf_index(store),
        text_root=tmp_path / "text",
        reader=unsegmentable,
        force=True,
    )

    assert stats["quarantined"] == 1
    assert store.document(SC_SOURCE_ID, key)["status"] == STATUS_QUARANTINED
    # THE POINT: a consumer that globs the text tree instead of reading the
    # manifest must not find a file the index says does not exist.
    assert not dest.exists()


# --------------------------------------------------------------------- runs

def test_extraction_follows_the_selection_file_order(tmp_path, store):
    # Task 11's contract: selection.jsonl is a STRATIFIED SPREAD, not the
    # top N by priority, and an interrupted extraction is meant to fail over
    # the whole corpus shape. Re-sorting here would throw that away.
    keys = [_key(start=1, end=20), _key(start=101, end=140), _key(start=201, end=260)]
    _, paths = _corpus(tmp_path, keys, store=store)
    rows = [
        _selection(keys[0], priority=1, case_type="civil"),
        _selection(keys[1], priority=7, case_type="criminal"),
        _selection(keys[2], priority=4, case_type="constitutional"),
    ]
    reader = FakeReader({path: scr_pages() for path in paths.values()})

    extract_corpus(
        store, rows, index=pdf_index(store), text_root=tmp_path / "text", reader=reader
    )

    assert reader.read == [paths[k] for k in keys]
    # The fixture's file order really does disagree with priority order, so
    # a "strongest first" sort would be visible above.
    assert [row["priority"] for row in rows] != sorted(
        (row["priority"] for row in rows), reverse=True
    )


def test_limit_counts_work_and_not_rows_examined(tmp_path, store):
    keys = [_key(start=s, end=s + 19) for s in (1, 101, 201, 301)]
    _, paths = _corpus(tmp_path, keys, store=store)
    rows = [_selection(k) for k in keys]
    reader = FakeReader({path: scr_pages() for path in paths.values()})

    first = extract_corpus(
        store,
        rows,
        index=pdf_index(store),
        text_root=tmp_path / "text",
        reader=reader,
        limit=2,
    )
    assert first["extracted"] == 2

    # A resumed run must ADVANCE. Counting examined rows would spend the
    # whole cap re-deciding the two already done, forever.
    second = extract_corpus(
        store,
        rows,
        index=pdf_index(store),
        text_root=tmp_path / "text",
        reader=reader,
        limit=2,
    )
    assert second["extracted"] == 2
    assert second["skipped"] == 2
    assert store.document_count(SC_SOURCE_ID, status=STATUS_OK) == 4


def test_one_unreadable_pdf_does_not_cost_the_rest_of_the_run(tmp_path, store):
    keys = [_key(start=s, end=s + 19) for s in (1, 101, 201)]
    _, paths = _corpus(tmp_path, keys, store=store)
    reader = FakeReader({path: scr_pages() for path in paths.values()}, fail=(paths[keys[1]],))

    stats = extract_corpus(
        store,
        [_selection(k) for k in keys],
        index=pdf_index(store),
        text_root=tmp_path / "text",
        reader=reader,
    )

    assert stats["failed"] == 1
    assert stats["extracted"] == 2
    assert keys[1] in stats["failures"][0]["key"]
    assert store.document_count(SC_SOURCE_ID, status=STATUS_OK) == 2


def test_enough_failures_stop_the_run_because_the_fault_is_not_in_the_objects(
    tmp_path, store
):
    keys = [_key(start=s, end=s + 19) for s in (1, 101, 201, 301)]
    _, paths = _corpus(tmp_path, keys, store=store)
    reader = FakeReader({}, fail=tuple(paths.values()))

    with pytest.raises(ExtractionError, match="stopping after"):
        extract_corpus(
            store,
            [_selection(k) for k in keys],
            index=pdf_index(store),
            text_root=tmp_path / "text",
            reader=reader,
            max_failures=2,
        )
    assert len(reader.read) == 2


def test_the_run_reports_a_quarantine_breakdown_by_reason(tmp_path, store):
    ok_key, blind_key, scan_key = (_key(start=s, end=s + 19) for s in (1, 101, 201))
    _, paths = _corpus(tmp_path, [ok_key, blind_key, scan_key], store=store)
    reader = FakeReader(
        {
            paths[ok_key]: scr_pages(),
            paths[blind_key]: scr_pages(body=BODY.split("\n", 1)[1]),
            paths[scan_key]: ["", "", ""],
        }
    )

    stats = extract_corpus(
        store,
        [_selection(k) for k in (ok_key, blind_key, scan_key)],
        index=pdf_index(store),
        text_root=tmp_path / "text",
        reader=reader,
    )

    assert stats["extracted"] == 1
    assert stats["quarantined"] == 2
    assert stats["reasons"] == {Q_NO_JUDGMENT_START: 1, Q_NO_TEXT: 1}
    assert store.document(SC_SOURCE_ID, scan_key)["reason"] == Q_NO_TEXT


def test_the_recorded_document_carries_the_span_and_the_selection_identity(tmp_path, store):
    key = _key(year=2020, volume=7, start=941, end=960)
    _, paths = _corpus(tmp_path, [key], store=store)
    reader = FakeReader({paths[key]: scr_pages()})

    extract_corpus(
        store,
        [_selection(key, year=2020, scr_prefix="2020_7_941_")],
        index=pdf_index(store),
        text_root=tmp_path / "text",
        reader=reader,
    )

    row = store.document(SC_SOURCE_ID, key)
    assert (row["page_start"], row["page_end"]) == (941, 960)
    # The PDF's own page count is a DIFFERENT number from the printed span,
    # and keeping both is what makes their disagreement visible.
    assert row["pages"] == 6
    assert row["citation"] == "[2015] 1 S.C.R. 1"
    assert row["case_id"] == "C.A. 3221/2018"
    assert row["chars"] > 0
    assert json.loads(row["meta_json"])["signals"] == ["held", "case_law_reference"]


def test_every_document_row_records_which_reader_made_it(tmp_path, store):
    # EXTRACT_VERSION ALONE UNDER-GUARDS THE CORPUS, and the smoke proved it:
    # the same rules over the same PDFs gave DIFFERENT verdicts on the two
    # lanes the installed library can dispatch to - two 2025 objects flipped
    # between `ok` and quarantined. So the text on disk is a function of the
    # reader as well as of this file, and a row that does not name its reader
    # cannot be re-derived: the audit's byte-for-byte re-check would blame
    # the rules for a library upgrade.
    good, bad = _key(start=1), _key(start=101, end=140)
    _, paths = _corpus(tmp_path, [good, bad], store=store)
    reader = FakeReader({paths[good]: scr_pages(), paths[bad]: ["too short"]})

    extract_corpus(
        store,
        [_selection(good), _selection(bad, scr_prefix="2015_1_101_")],
        index=pdf_index(store),
        text_root=tmp_path / "text",
        reader=reader,
    )

    # ON EVERY ROW, refused as well as emitted: a quarantine is a verdict the
    # reader produced too, and re-opening it after an upgrade is exactly the
    # case this exists for.
    for key, status in ((good, STATUS_OK), (bad, STATUS_QUARANTINED)):
        row = store.document(SC_SOURCE_ID, key)
        assert row["status"] == status
        assert json.loads(row["meta_json"])["reader"]["lane"].endswith("FakeReader")
        # ...and the layout-engine state with it. This module sets that
        # switch for its own reader only, so an injected one records that
        # nothing was pinned rather than claiming a state it did not set.
        assert json.loads(row["meta_json"])["reader"]["layout"] is None
    # ...and an injected reader is NAMED rather than allowed to claim the
    # pinned lane, which is the misreading that would make the field a lie
    # in every test-shaped run.
    from tuned.data.extract import READER_LANE, READER_LAYOUT

    assert reader_fingerprint(reader)["lane"] != READER_LANE
    assert reader_fingerprint(read_pdf_pages) == {
        "lane": READER_LANE,
        "pymupdf4llm": reader_fingerprint(read_pdf_pages)["pymupdf4llm"],
        "layout": READER_LAYOUT,
    }
    assert reader_fingerprint(read_pdf_pages)["lane"] == "pymupdf4llm.helpers.pymupdf_rag"
    # THREE INPUTS TO THE BYTES, not two. The layout switch is process-global
    # and moves what the pinned lane extracts - measured, emitted text
    # differs on 2 of the 15 real objects with it toggled - so a row that
    # names only the lane and the version cannot re-derive its own text.
    assert reader_fingerprint(read_pdf_pages)["layout"] == "off"


class StructuredReader(FakeReader):
    """A reader that can also report what the PDF is made of."""

    def __init__(self, docs, structures, **over):
        super().__init__(docs, **over)
        self._structures = {str(k): v for k, v in structures.items()}
        self.structure = self._structure

    def _structure(self, path):
        return self._structures[str(path)]


def test_a_scan_era_object_is_refused_without_being_read_at_all(tmp_path, store):
    # STRUCTURE BEFORE TEXT, and the saving is the point: measured over the
    # 15 real objects the structural read costs 0.003-0.53 s against
    # 0.8-21.4 s to convert the same document - 21x to 523x cheaper on every
    # one of them - so a scan-era object is refused for a fraction of what
    # reading it would cost. (The absolute figures are the machine's; the
    # ratio is the claim, and an earlier run at half these times gives the
    # same one.)
    key = _key()
    _, paths = _corpus(tmp_path, [key], store=store)
    reader = StructuredReader(
        {paths[key]: scr_pages()},
        {paths[key]: PdfStructure(fonts=(("HiddenHorzOCR", "WinAnsiEncoding"),), pages=6)},
    )

    stats = extract_corpus(
        store,
        [_selection(key)],
        index=pdf_index(store),
        text_root=tmp_path / "text",
        reader=reader,
    )

    assert stats["quarantined"] == 1
    assert stats["reasons"] == {Q_SCANNED_ERA: 1}
    assert stats["structure_probed"] == 1
    # NEVER READ. The premise of the ordering, and a fixture in which the
    # document was unreadable could not tell the two apart - this one reads
    # perfectly well and is refused anyway.
    assert reader.read == []
    assert extract_text(scr_pages()).ok
    row = store.document(SC_SOURCE_ID, key)
    assert row["status"] == STATUS_QUARANTINED
    assert row["text_path"] is None
    # ...and the structure is on the row either way, so the operator can see
    # WHAT was refused rather than only that something was.
    assert json.loads(row["meta_json"])["structure"]["fonts"] == ["HiddenHorzOCR"]


def test_the_scan_era_gate_can_be_opened_deliberately(tmp_path, store):
    # "Which years are in v1" is an operator's decision. The gate makes the
    # decision explicit; it does not take it. Without this the quarantine
    # would be a wall rather than a question.
    key = _key()
    _, paths = _corpus(tmp_path, [key], store=store)
    reader = StructuredReader(
        {paths[key]: scr_pages()},
        {paths[key]: PdfStructure(image_filters=("/JBIG2Decode",), pages=6)},
    )

    stats = extract_corpus(
        store,
        [_selection(key)],
        index=pdf_index(store),
        text_root=tmp_path / "text",
        reader=reader,
        allow_scanned_era=True,
    )

    assert stats["extracted"] == 1
    assert reader.read == [paths[key]]
    # ...and the knob is for the ERA only. A text layer this module cannot
    # decode is not a matter of preference, so the same flag does not admit
    # it.
    mojibake = _key(start=201, end=240)
    _corpus(tmp_path, [mojibake], store=store)
    local = str(tmp_path / "corpus" / "sc" / mojibake)
    second = StructuredReader(
        {local: scr_pages()},
        {local: PdfStructure(fonts=(("KrutiDev010", "WinAnsiEncoding"),), pages=6)},
    )
    again = extract_corpus(
        store,
        [_selection(mojibake, scr_prefix="2015_1_201_")],
        index=pdf_index(store),
        text_root=tmp_path / "text",
        reader=second,
        allow_scanned_era=True,
    )
    assert again["reasons"] == {Q_MOJIBAKE_FONT: 1}


def test_a_reader_that_cannot_report_structure_leaves_the_gates_unarmed_and_says_so(
    tmp_path, store, capsys
):
    # THE SILENCE THIS BREAKS. A run with no scan-era refusals because it
    # never looked reads exactly like a corpus with no scans in it, and the
    # difference matters more than most of what the run prints.
    key = _key()
    _, paths = _corpus(tmp_path, [key], store=store)
    reader = FakeReader({paths[key]: scr_pages()})

    stats = extract_corpus(
        store,
        [_selection(key)],
        index=pdf_index(store),
        text_root=tmp_path / "text",
        reader=reader,
    )

    assert stats["structure_probed"] == 0
    assert stats["extracted"] == 1
    assert Q_SCANNED_ERA not in stats["reasons"]
    # THE PREMISE: this reader really has no probe, and one that has is
    # counted - so the zero is about the reader and not about the corpus.
    assert getattr(reader, "structure", None) is None
    assert json.loads(store.document(SC_SOURCE_ID, key)["meta_json"])["structure"] is None


def test_a_second_selection_row_on_one_pdf_does_not_re_record_the_document(tmp_path, store):
    # Two rows, two case ids, one PDF. The manifest keeps the FIRST row, so
    # the index has to as well: re-recording from the later row would leave
    # the document row and the manifest row describing one judgment under two
    # identities, and nothing downstream reads object_key for identity, so
    # nothing could ever see it. --force is where it bites, because that is
    # the mode in which the second row is not skipped by the resume rule.
    key = _key()
    _, paths = _corpus(tmp_path, [key], store=store)
    rows = [_selection(key), _selection(key, case_id="C.A. 3221-A/2018")]
    reader = FakeReader({paths[key]: scr_pages()})

    stats = extract_corpus(
        store, rows, index=pdf_index(store), text_root=tmp_path / "text", reader=reader,
        force=True,
    )

    assert stats["extracted"] == 1
    assert stats["duplicate_rows"] == 1
    assert store.document(SC_SOURCE_ID, key)["case_id"] == rows[0]["case_id"]
    # ... and the PDF was read once, not twice.
    assert len(reader.read) == 1
    # THE PREMISES: both rows really do resolve to the one PDF, and they
    # really do disagree about which judgment it is.
    assert [resolve_pdf(row, pdf_index(store))[0] for row in rows] == [key, key]
    assert rows[0]["case_id"] != rows[1]["case_id"]


def test_the_run_counts_the_documents_carrying_a_reportable_flag(tmp_path, store):
    # A NON-NULL flag on this corpus is a warning rather than a fact about
    # the judgment: P0 found it in 2 of 70 objects, and it belongs to the
    # COURT-RELEASED PDF and not to the S.C.R. reprint - so a document that
    # has one may not be a reprint at all, which makes it exactly the
    # document whose boundary rules may not apply. That caveat existed only
    # in a report until it was counted.
    flagged, plain = _key(start=1, end=20), _key(start=101, end=140)
    _, paths = _corpus(tmp_path, [flagged, plain], store=store)
    pages = scr_pages()
    reader = FakeReader(
        {paths[flagged]: ["REPORTABLE\n" + pages[0], *pages[1:]], paths[plain]: pages}
    )

    stats = extract_corpus(
        store, [_selection(k) for k in (flagged, plain)], index=pdf_index(store),
        text_root=tmp_path / "text", reader=reader,
    )

    assert stats["reportable"] == 1
    # THE PREMISE: both documents were extracted, so the count is the flag
    # and not the extraction.
    assert stats["extracted"] == 2


# ----------------------------------------------------------------- manifest

def test_the_manifest_joins_the_selection_row_to_the_extraction_facts(tmp_path, store):
    ok_key, bad_key = _key(start=1, end=20), _key(start=101, end=140)
    _, paths = _corpus(tmp_path, [ok_key, bad_key], store=store)
    rows = [_selection(ok_key), _selection(bad_key, case_id="C.A. 99/2019")]
    reader = FakeReader(
        {paths[ok_key]: scr_pages(), paths[bad_key]: scr_pages(body=BODY.split("\n", 1)[1])}
    )
    extract_corpus(
        store, rows, index=pdf_index(store), text_root=tmp_path / "text", reader=reader
    )

    out = tmp_path / "extraction.jsonl"
    written = write_manifest(store, rows, out, index=pdf_index(store))

    manifest = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert written == 1
    # Only what was EMITTED: a quarantined judgment has no text to point at,
    # and listing it would hand the segmenter a path that is not there.
    assert [row["case_id"] for row in manifest] == ["C.A. 3221/2018"]
    row = manifest[0]
    assert row["case_type"] == "civil"  # from the selection row
    assert row["priority"] == 4
    assert row["page_start"] == 1  # ... and from the extraction
    assert row["chars"] > 0
    assert Path(row["text_path"]).read_text(encoding="utf-8").startswith("The Judgment")
    assert row["doc_id"] == "2015_1_1_20_EN"
    # ...and WHICH READER MADE IT. Downstream reads this file and not the
    # document table, so a corpus assembled across a library upgrade would
    # otherwise be indistinguishable from one built in a single pass - and
    # the smoke measured the two lanes disagreeing about whole documents.
    assert row["reader_lane"].endswith("FakeReader")
    assert "reader_version" in row
    # ALL THREE inputs to the bytes, not two: the layout-engine state is
    # process-global and moves what the pinned lane extracts (emitted text
    # differs on 2 of the 15 real objects with it toggled), so the column
    # that lets a manifest be re-derived has to carry it too. None here
    # because this module pins that switch for its own reader only.
    assert "reader_layout" in row and row["reader_layout"] is None


def test_two_selection_rows_landing_on_one_pdf_are_written_to_the_manifest_once(
    tmp_path, store
):
    # The metadata can carry the same judgment twice, and two citations can
    # address one object. Emitting both would put that judgment in the
    # corpus twice under two case ids, and nothing downstream reads
    # object_key for identity - so this is the only place it can be caught.
    key = _key()
    _, paths = _corpus(tmp_path, [key], store=store)
    rows = [_selection(key), _selection(key, case_id="C.A. 3221-A/2018")]
    reader = FakeReader({paths[key]: scr_pages()})
    extract_corpus(
        store, rows, index=pdf_index(store), text_root=tmp_path / "text", reader=reader
    )

    out = tmp_path / "extraction.jsonl"
    written = write_manifest(store, rows, out, index=pdf_index(store))

    assert written == 1
    # THE PREMISE: both rows really do resolve to the one PDF, so the
    # deduplication is what makes this 1 and not the join failing.
    assert [resolve_pdf(row, pdf_index(store))[0] for row in rows] == [key, key]
    # ... and the drop is reported rather than swallowed.
    events = [e for e in store.events("manifest_duplicate_documents")]
    assert len(events) == 1
    assert json.loads(events[0]["detail_json"])["count"] == 1


# -------------------------------------------------------------------- audit

def test_the_audit_prints_the_seam_on_both_sides_of_the_cut(tmp_path, store):
    key = _key()
    _, paths = _corpus(tmp_path, [key], store=store)
    reader = FakeReader({paths[key]: scr_pages()})
    extract_corpus(
        store,
        [_selection(key)],
        index=pdf_index(store),
        text_root=tmp_path / "text",
        reader=reader,
    )

    report = audit_report(store, 1, index=pdf_index(store), reader=reader)

    # The operator's whole job here is to check the boundary, so the report
    # has to show the last of what was thrown away next to the first of what
    # was kept. Either half alone proves nothing.
    assert "Case Law Reference" in report
    assert "The Judgment of the Court was delivered by" in report
    assert "judgment_delivered_by" in report
    assert key in report

    # ... and the removed side opens where a LINE does. It is a window cut by
    # character count, so 400 characters back lands wherever it lands, and
    # the operator is being asked to read this text and judge it.
    removed = (
        report.split("LAST OF WHAT WAS REMOVED")[1]
        .split("\n", 1)[1]
        .split("FIRST OF WHAT WAS KEPT")[0]
    )
    opening = removed.strip().splitlines()[0].strip()
    cleaned, _ = clean_pages(scr_pages())
    joined = "\n".join(cleaned)
    assert any(line.strip() == opening for line in joined.split("\n"))
    # THE PREMISE: the window really is truncated here, and the raw cut
    # really would have opened mid-word - so the snap is what put a whole
    # line at the top and not the fixture.
    result = extract_text(scr_pages())
    assert result.headnote_chars > AUDIT_REMOVED_CHARS
    assert joined[result.headnote_chars - AUDIT_REMOVED_CHARS - 1] != "\n"


def test_the_audit_opens_with_the_check_that_settles_whether_the_guard_can_read_at_all(
    tmp_path, store
):
    # The cheapest check available on run one, and the only one that can find
    # a guard that is blind to this reporter's typesetting - because a
    # contaminated document reads like a clean judgment everywhere else. It
    # has to be impossible to miss, so it is the first thing in the report,
    # above the sample.
    key = _key()
    _, paths = _corpus(tmp_path, [key], store=store)
    # A judgment with no headnote at all - the document that legitimately
    # prints `none`, and therefore the one the tell has to be read against.
    reader = FakeReader({paths[key]: [_pad(BODY, 2600)]})
    extract_corpus(
        store, [_selection(key)], index=pdf_index(store), text_root=tmp_path / "text",
        reader=reader,
    )

    report = audit_report(store, 1, index=pdf_index(store), reader=reader)

    assert "READ THIS FIRST" in report
    # It is the HEAD of the report, above the first document.
    assert report.index("READ THIS FIRST") < report.index(key)
    # ... and it quotes the line the operator will actually read, verbatim -
    # once in the tell and once against this document. A tell that named a
    # string the report does not print would be worse than no tell: the
    # operator would scan for it, not find it, and conclude nothing is wrong.
    assert report.count("headnote signals: none") == 2
    # ...and the header names the reader beside the rules, for the reason
    # every row now carries it: this report's verdicts are a function of both,
    # and an operator reading a re-run against a different library needs to
    # see which one changed.
    assert f"extract_version {EXTRACT_VERSION}" in report
    assert reader_fingerprint(reader)["lane"] in report


def test_the_audit_says_when_the_rules_no_longer_reproduce_what_the_corpus_holds(
    tmp_path, store
):
    # `--audit` re-extracts rather than reading back from disk, and the
    # docstring's second reason for that - "re-running is also a check that
    # the rules still produce what the corpus holds" - was a claim with no
    # code under it: nothing compared anything to the row it already had.
    key = _key()
    _, paths = _corpus(tmp_path, [key], store=store)
    reader = FakeReader({paths[key]: scr_pages()})
    extract_corpus(
        store, [_selection(key)], index=pdf_index(store), text_root=tmp_path / "text",
        reader=reader,
    )

    fresh = audit_report(store, 1, index=pdf_index(store), reader=reader)
    assert "DIFFERS FROM THE STORED TEXT" not in fresh
    assert "NO LONGER AGREE" not in fresh

    # The corpus now holds text these rules do not produce - which is what a
    # rule change without a version bump leaves behind.
    row = store.document(SC_SOURCE_ID, key)
    store.record_document(SC_SOURCE_ID, key, dict(row, sha256="0" * 64, chars=11))
    stale = audit_report(store, 1, index=pdf_index(store), reader=reader)
    assert "DIFFERS FROM THE STORED TEXT" in stale

    # ... and a row whose STATUS no longer reproduces is the louder half of
    # the same check.
    store.record_document(SC_SOURCE_ID, key, dict(row, status=STATUS_QUARANTINED,
                                                 reason=Q_NO_JUDGMENT_START))
    flipped = audit_report(store, 1, index=pdf_index(store), reader=reader)
    assert "NO LONGER AGREE" in flipped


def test_the_audit_sample_walks_the_corpus_instead_of_reading_the_first_n():
    # Object keys sort by year, so a sample taken along them is a walk
    # across 2010-2025. Taking the head of the list would show the operator
    # sixteen years of corpus through one year of it - and the first year is
    # also the year most likely to have been extracted before a rule change.
    assert spread(list(range(100)), 4) == [0, 25, 50, 75]
    assert spread(list(range(100)), 1) == [0]
    assert spread(list(range(10)), 0) == []
    assert spread([1, 2], 5) == [1, 2]


def test_the_audit_shows_quarantined_documents_and_why(tmp_path, store):
    ok_key, bad_key = _key(start=1, end=20), _key(start=101, end=140)
    _, paths = _corpus(tmp_path, [ok_key, bad_key], store=store)
    reader = FakeReader(
        {paths[ok_key]: scr_pages(), paths[bad_key]: scr_pages(body=BODY.split("\n", 1)[1])}
    )
    extract_corpus(
        store,
        [_selection(ok_key), _selection(bad_key)],
        index=pdf_index(store),
        text_root=tmp_path / "text",
        reader=reader,
    )

    report = audit_report(store, 2, index=pdf_index(store), reader=reader)

    assert Q_NO_JUDGMENT_START in report
    assert bad_key in report
    # The refusals are the reason the audit exists, so they are sampled
    # ALONGSIDE the successes rather than being crowded out by them.
    assert ok_key in report


def test_the_audit_of_a_run_that_refused_everything_still_shows_the_refusals(tmp_path, store):
    # "Half of them refusals" must not become "none of them" on the run whose
    # audit matters most. Half of an odd sample rounds down, so a corpus with
    # no successes to make up the other half printed an audit with nothing in
    # it at all - at exactly the moment the operator needs to see what was
    # refused and why.
    keys = [_key(start=1, end=20), _key(start=101, end=140)]
    _, paths = _corpus(tmp_path, keys, store=store)
    unsegmentable = scr_pages(body=BODY.split("\n", 1)[1])
    reader = FakeReader({paths[k]: unsegmentable for k in keys})
    extract_corpus(
        store, [_selection(k) for k in keys], index=pdf_index(store),
        text_root=tmp_path / "text", reader=reader,
    )

    report = audit_report(store, 1, index=pdf_index(store), reader=reader)

    assert Q_NO_JUDGMENT_START in report
    assert sum(key in report for key in keys) == 1
    # THE PREMISE: there really is nothing emitted to sample instead.
    assert store.document_count(SC_SOURCE_ID, status=STATUS_OK) == 0


def test_a_healthy_corpus_still_spends_half_the_audit_on_the_refusals(tmp_path, store):
    # THE OTHER HALF OF THE FILL RULE, and the case that half had none of.
    # `max(n - emitted, n // 2)` has two terms: the first rescues the audit of
    # a run that refused everything (the test above), and the SECOND is the
    # ordinary run - where there are more than enough successes to fill the
    # sample and the refusals would be crowded out to nothing by an even walk
    # over a table they are a minority of. With no case here, `n // 2` was
    # deletable with the whole suite green.
    emitted = [_key(start=1 + 50 * i, end=40 + 50 * i) for i in range(8)]
    refused = [_key(start=901, end=940), _key(start=951, end=990)]
    _, paths = _corpus(tmp_path, emitted + refused, store=store)
    unsegmentable = scr_pages(body=BODY.split("\n", 1)[1])
    reader = FakeReader(
        {paths[k]: scr_pages() for k in emitted} | {paths[k]: unsegmentable for k in refused}
    )
    extract_corpus(
        store, [_selection(k) for k in emitted + refused], index=pdf_index(store),
        text_root=tmp_path / "text", reader=reader,
    )

    picked = audit_sample(store, 4, source_id=SC_SOURCE_ID)

    assert len(picked) == 4
    assert sum(row["status"] == STATUS_QUARANTINED for row in picked) == 2
    # THE PREMISE that makes this the SECOND term and not the first: there are
    # plenty of emitted documents, so `n - emitted` is negative and only the
    # half rule can be what reserved the two places.
    assert store.document_count(SC_SOURCE_ID, status=STATUS_OK) == len(emitted)
    assert 4 - store.document_count(SC_SOURCE_ID, status=STATUS_OK) < 0


# --------------------------------------------------------------- the reader

def _install_reader(monkeypatch, lane_to_markdown, *, shim=None, with_lane=True):
    """A fake pymupdf4llm whose PINNED LANE carries `lane_to_markdown`.

    The package-level `to_markdown` is deliberately a DIFFERENT function -
    the dispatch shim the installed 1.28.2 really has - so that "which of the
    two ran" is observable. A test that put the same callable in both places
    would pass whichever one the module called, which is the whole question.
    """
    package = types.ModuleType("pymupdf4llm")
    package.to_markdown = shim or (lambda *a, **k: pytest.fail("the shim was called"))
    package.layout_calls = []
    package.use_layout = package.layout_calls.append
    helpers = types.ModuleType("pymupdf4llm.helpers")
    monkeypatch.setitem(sys.modules, "pymupdf4llm", package)
    monkeypatch.setitem(sys.modules, "pymupdf4llm.helpers", helpers)
    if with_lane:
        lane = types.ModuleType("pymupdf4llm.helpers.pymupdf_rag")
        lane.to_markdown = lane_to_markdown
        helpers.pymupdf_rag = lane
        monkeypatch.setitem(sys.modules, "pymupdf4llm.helpers.pymupdf_rag", lane)
    else:
        monkeypatch.setitem(sys.modules, "pymupdf4llm.helpers.pymupdf_rag", None)
    return package


def test_the_reader_pins_the_options_that_decide_what_the_text_contains(monkeypatch):
    # THE READER'S DEFAULTS ARE NOT THIS REPO'S DECISIONS. pymupdf4llm crops
    # 50 points off the top and bottom of every page unless told otherwise,
    # and that band is where the running head, the printed page number, the
    # page-tail footnotes and the REPORTABLE line live - so three passes
    # above would silently receive nothing to do, and the audit would report
    # that as a clean document.
    seen = {}

    def to_markdown(path, *, page_chunks=False, margins=(0, 50, 0, 50),
                    table_strategy="lines_strict", write_images=False,
                    embed_images=False, force_text=True, show_progress=True):
        seen.update(
            path=path, page_chunks=page_chunks, margins=margins,
            table_strategy=table_strategy, show_progress=show_progress,
        )
        return [{"text": "page one"}, {"text": "page two"}]

    _install_reader(monkeypatch, to_markdown)

    assert read_pdf_pages("a.pdf") == ["page one", "page two"]
    assert seen["margins"] == 0
    assert seen["page_chunks"] is True
    assert seen["table_strategy"] == "lines_strict"
    assert seen["show_progress"] is False
    # THE PREMISE, without which "margins == 0" is a test of nothing: the
    # library's own default is a CROP, so passing nothing is a decision too.
    import inspect

    assert inspect.signature(to_markdown).parameters["margins"].default == (0, 50, 0, 50)


def test_the_reader_is_the_pinned_lane_and_never_the_package_level_dispatch_shim(
    monkeypatch,
):
    # WHAT `pymupdf4llm.to_markdown` IS in 1.28.2: not the reader, but a
    # `(*args, **kwargs)` shim that forwards to one of two implementations,
    # defaulting to the one that drops `margins` and `table_strategy`, turns
    # OCR on, and was measured reporting NO HEADNOTE on a document that has
    # one. Calling the package-level name is therefore not "calling the
    # library", it is letting the library choose the corpus.
    called = []

    def shim(*args, **kwargs):
        called.append("shim")
        return [{"text": "the wrong lane"}]

    def lane(path, *, page_chunks=False, margins=(0, 50, 0, 50),
             table_strategy="lines_strict", show_progress=True):
        called.append("lane")
        return [{"text": "the pinned lane"}]

    package = _install_reader(monkeypatch, lane, shim=shim)

    assert read_pdf_pages("a.pdf") == ["the pinned lane"]
    assert called == ["lane"]
    # THE PREMISE: the shim really is reachable and really would have
    # answered - so the import path is the only thing that chose.
    assert package.to_markdown("a.pdf") == [{"text": "the wrong lane"}]
    assert called == ["lane", "shim"]
    # ...and the layout switch is thrown on the way, which is not a tidiness
    # measure: it is a process-global mutation of pymupdf that decides what
    # the PINNED LANE extracts. See the test below.
    assert package.layout_calls == [False]


def test_the_layout_state_the_row_records_is_the_one_the_reader_throws(monkeypatch):
    # THE FIELD AND THE SWITCH ARE ONE FACT. `use_layout` was documented here
    # as belt and braces - "so that anything else in the process gets the
    # lane this module chose" - and that is not what it does. It mutates
    # `pymupdf` process-wide and changes what the pinned legacy lane itself
    # extracts: measured over the 15 real objects, raw pages differ on 6 and
    # the difference survives into the EMITTED text on 2 (2022_large
    # 129,282 -> 129,143 chars, 2025_large 128,900 -> 128,921), because the
    # table detector sees a different page. A row that recorded `layout: off`
    # while the reader threw `True` would name a state that did not make its
    # bytes, so the recorded value and the thrown switch read the same
    # constant - and this fails if either one is hard-coded past it.
    from tuned.data import extract

    def lane(path, *, page_chunks=False, margins=0, table_strategy="lines_strict"):
        return ["only page"]

    package = _install_reader(monkeypatch, lane)

    assert read_pdf_pages("a.pdf") == ["only page"]
    assert package.layout_calls == [False]
    assert reader_fingerprint(read_pdf_pages)["layout"] == extract.READER_LAYOUT == "off"

    monkeypatch.setattr(extract, "READER_LAYOUT", "on")
    package.layout_calls.clear()

    assert read_pdf_pages("a.pdf") == ["only page"]
    assert package.layout_calls == [True]
    assert reader_fingerprint(read_pdf_pages)["layout"] == "on"


def test_a_library_without_the_pinned_lane_is_refused_rather_than_dispatched(monkeypatch):
    # The refusal that must NOT become a fallback. If the lane is gone, the
    # only thing left is the path this module rejected on measured evidence,
    # so "use whatever is there" would silently rebuild the corpus under the
    # reader that reports no headnote on a document that has one.
    def lane(path, *, page_chunks=False, margins=0, table_strategy="lines_strict"):
        return []  # pragma: no cover - never reached, the lane is not installed

    _install_reader(monkeypatch, lane, with_lane=False)
    with pytest.raises(ExtractionError, match="pymupdf4llm.helpers.pymupdf_rag"):
        read_pdf_pages("a.pdf")


def test_an_ocr_switch_is_turned_off_where_the_reader_declares_one(monkeypatch):
    # v1 ships no OCR: `no_text` is a quarantine and not an invitation. The
    # pinned lane has no OCR knob today, but the lane the shim prefers
    # defaults `use_ocr=True`, and a version of the pinned one could grow the
    # same default without saying so.
    seen = {}

    def lane(path, *, page_chunks=False, margins=0, table_strategy="lines_strict",
             use_ocr=True, **kwargs):
        seen.update(use_ocr=use_ocr, **kwargs)
        return ["only page"]

    _install_reader(monkeypatch, lane)
    assert read_pdf_pages("a.pdf") == ["only page"]
    assert seen["use_ocr"] is False
    # THE PREMISE: the reader's own default is ON, so passing nothing would
    # have OCR'd the corpus.
    import inspect

    assert inspect.signature(lane).parameters["use_ocr"].default is True
    # ...and `force_ocr`, which this reader does NOT declare, is not smuggled
    # in on the catch-all: a switch that is only accepted is not a switch
    # that is honoured, and a row saying "OCR off" would then be a claim
    # nothing checked.
    assert "force_ocr" not in seen


def test_a_reader_that_cannot_take_the_pinned_options_is_refused_not_silently_defaulted(
    monkeypatch,
):
    def lane(path, *, page_chunks=False, table_strategy="lines_strict"):
        return []

    _install_reader(monkeypatch, lane)
    with pytest.raises(ExtractionError, match="margins"):
        read_pdf_pages("a.pdf")


def test_a_reader_missing_only_a_cosmetic_option_still_runs(monkeypatch):
    # The line between the two: an option that changes the CORPUS is refused,
    # an option that changes the LOG is dropped. Stopping a run over a
    # progress bar would be its own kind of wrong.
    def lane(path, *, page_chunks=False, margins=(0, 50, 0, 50),
             table_strategy="lines_strict"):
        return ["only page"]

    _install_reader(monkeypatch, lane)
    assert read_pdf_pages("a.pdf") == ["only page"]


def test_a_reader_that_only_swallows_kwargs_is_refused_like_one_that_cannot_take_them(
    monkeypatch,
):
    # THE ESCAPE HATCH THAT DEFEATED THE CHECK IT WAS PAIRED WITH. A reader
    # whose signature ends in `**kwargs` ACCEPTS `margins=0` and then does
    # nothing with it, so the library's 50-point crop stays in force on every
    # page while the option check reports that the behaviour is pinned - which
    # is precisely the silent default this check exists to prevent, arriving
    # by a different door. Accepting a name is not honouring it.
    seen = {}

    def loose(path, *, page_chunks=False, **kwargs):
        seen.update(kwargs)
        # What the library actually does with what it swallowed: nothing.
        # The crop default is still what produced this page.
        return ["page one"]

    _install_reader(monkeypatch, loose)
    with pytest.raises(ExtractionError, match="margins"):
        read_pdf_pages("a.pdf")
    # THE PREMISES. The reader really would have run and really would have
    # taken the options without complaint - so nothing but this check stands
    # between a renamed option and a silently cropped corpus.
    assert seen == {}
    assert loose("a.pdf", margins=0, table_strategy="x") == ["page one"]
    assert seen == {"margins": 0, "table_strategy": "x"}
    # ... and the error names the option whose default decides the corpus.
    assert "**kwargs" in _error_text(loose)


def _error_text(to_markdown) -> str:
    from tuned.data.extract import _reader_options

    try:
        _reader_options(to_markdown)
    except ExtractionError as exc:
        return str(exc)
    return ""


def test_a_reader_with_kwargs_beside_the_required_options_may_use_them_for_the_rest(
    monkeypatch,
):
    # The other side of the same line, so that the rule above is "**kwargs is
    # not evidence" and not "**kwargs is fatal": a reader that names the three
    # corpus-deciding options explicitly is honouring them, and the cosmetic
    # ones may ride in on the catch-all.
    seen = {}

    def modern(path, *, page_chunks=False, margins=(0, 50, 0, 50),
               table_strategy="lines_strict", **kwargs):
        seen.update(page_chunks=page_chunks, margins=margins, **kwargs)
        return ["only page"]

    _install_reader(monkeypatch, modern)
    assert read_pdf_pages("a.pdf") == ["only page"]
    assert seen["margins"] == 0
    assert seen["show_progress"] is False


# ---------------------------------------------------------------------- CLI

def test_cli_hard_exits_after_success():
    assert "os._exit(" in EXTRACT_SRC.read_text(encoding="utf-8")


def test_the_version_ledger_describes_the_version_the_module_ships():
    # `extract_version` is the third resume input: rows written under older
    # rules are re-extracted, rows at this version are left alone. So a rule
    # change without a bump leaves a corpus the rules no longer produce - and
    # a bump without a ledger entry leaves nobody able to say what the stale
    # rows are stale FOR, which is the question `--force` or a targeted
    # re-run turns on. Cross-checked against the comment rather than pinned
    # to a literal: two independent statements of the same fact, and this
    # fails when they drift apart in either direction.
    source = EXTRACT_SRC.read_text(encoding="utf-8")
    entries = [int(n) for n in re.findall(r"^#   (\d+)  ", source, re.M)]
    assert entries == sorted(entries)
    assert entries[-1] == EXTRACT_VERSION
    assert entries[0] == 2  # version 1 predates the ledger and is not described


def _write_selection(paths_obj, rows):
    from tuned.data.jsonl import write_jsonl
    from tuned.data.select import SELECTION_FILENAME

    out = paths_obj.corpus_dir / SELECTION_FILENAME
    write_jsonl(out, rows)
    return out


def _build_paths(config):
    from tuned.data.config import load_build_config
    from tuned.data.paths import build_paths

    return build_paths(load_build_config(config, allow_unpinned=True).build.workdir).ensure()


def test_cli_extracts_the_selection_into_the_build_corpus(tmp_path, capsys):
    config = temp_config(tmp_path)
    paths = _build_paths(config)
    key = _key()
    store, local = _corpus(tmp_path, [key], store=Store.open(paths.state_db))
    store.close()
    _write_selection(paths, [_selection(key)])
    reader = FakeReader({local[key]: scr_pages()})

    assert main(["--config", config], reader=reader) == 0

    out = capsys.readouterr().out
    assert "documents indexed -> 1" in out
    text = paths.corpus_dir / "text" / key.replace(".pdf", ".txt")
    body = text.read_text(encoding="utf-8")
    assert body.startswith("The Judgment of the Court")
    assert "HELD:" not in body
    manifest = (paths.corpus_dir / "extraction.jsonl").read_text(encoding="utf-8")
    assert json.loads(manifest.splitlines()[0])["object_key"] == key
    # The run names its reader, and says out loud when the structural gates
    # could not run - a corpus with no scan-era refusals because nothing
    # looked reads exactly like a corpus with no scans in it.
    assert "reader    " in out
    assert "STRUCTURAL GATES DID NOT RUN" in out


def test_the_cli_is_quiet_about_the_gates_when_the_reader_can_arm_them(tmp_path, capsys):
    # The other side of the line above: the warning has to be about THIS
    # reader, or it is noise that trains the operator to skip it.
    config = temp_config(tmp_path)
    paths = _build_paths(config)
    key = _key()
    store, local = _corpus(tmp_path, [key], store=Store.open(paths.state_db))
    store.close()
    _write_selection(paths, [_selection(key)])
    reader = StructuredReader({local[key]: scr_pages()}, {local[key]: PdfStructure(pages=6)})

    assert main(["--config", config], reader=reader) == 0

    out = capsys.readouterr().out
    assert "STRUCTURAL GATES DID NOT RUN" not in out
    assert "documents indexed -> 1" in out


def test_cli_says_so_when_it_refused_more_than_a_quarter_of_what_it_read(tmp_path, capsys):
    # First-run signal #2, and it had no test. The refusal set is MEANT to be
    # non-empty - that is the design - so the number alone tells the operator
    # nothing; the RATE is what says the fault is in the rules or in the
    # source rather than in the individual documents.
    config = temp_config(tmp_path)
    paths = _build_paths(config)
    keys = [_key(start=s, end=s + 19) for s in (1, 101, 201, 301)]
    store, local = _corpus(tmp_path, keys, store=Store.open(paths.state_db))
    store.close()
    _write_selection(paths, [_selection(k) for k in keys])
    unsegmentable = scr_pages(body=BODY.split("\n", 1)[1])
    docs = {local[keys[0]]: scr_pages()}
    docs.update({local[k]: unsegmentable for k in keys[1:]})

    assert main(["--config", config], reader=FakeReader(docs)) == 0

    out = capsys.readouterr().out
    assert "HIGH QUARANTINE RATE" in out
    # The rate, not the count: three of four.
    assert "75.0%" in out


def test_cli_refuses_a_run_in_which_nothing_could_be_joined_to_a_pdf(tmp_path, capsys):
    # Same reasoning as select.py's NOTHING SELECTED backstop: a full
    # selection that joins to zero PDFs is a wrong assumption about keys,
    # not a corpus, and exiting 0 would report an empty extraction as done.
    config = temp_config(tmp_path)
    paths = _build_paths(config)
    store, _ = _corpus(tmp_path, [_key()], store=Store.open(paths.state_db))
    store.close()
    _write_selection(paths, [_selection(None, scr_prefix="1999_9_9_")])

    code = main(["--config", config], reader=FakeReader({}))
    out = capsys.readouterr().out
    assert code == 1
    assert "NOTHING JOINED" in out


def test_cli_says_what_to_run_when_the_selection_is_missing(tmp_path, capsys):
    config = temp_config(tmp_path)
    _build_paths(config)

    code = main(["--config", config], reader=FakeReader({}))
    out = capsys.readouterr().out
    assert code == 2
    assert "tuned.data.select" in out
