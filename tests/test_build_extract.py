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

import pytest

from tuned.data.extract import (
    FOOTNOTE_HEADING,
    MAX_STRIP_FRACTION,
    MIN_BODY_CHARS,
    MIN_DOC_CHARS,
    MIN_LATIN_RATIO,
    Q_BODY_TOO_SHORT,
    Q_HEADNOTE_RESIDUE,
    Q_LOW_TEXT_QUALITY,
    Q_NO_JUDGMENT_START,
    Q_NO_TEXT,
    Q_STRIP_TOO_LARGE,
    RESIDUE_WINDOW,
    RUNNING_DIGIT_BLIND_CHARS,
    RUNNING_MAX_CHARS,
    clean_pages,
    demote_markdown,
    extract_text,
    find_judgment_start,
    headnote_signals,
    latin_ratio,
    page_span_from_key,
    reportable_flag,
)

# --------------------------------------------------------------------------
# A document shaped like the ones in the bucket.
# --------------------------------------------------------------------------

RUNNING_HEADER = "[2020] 7 S.C.R. 941"

# Front matter + headnote: every line of this is the publisher's editorial
# work under EBC v. Modak, and none of it may reach the corpus.
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


def test_an_editorial_phrase_far_past_the_seam_is_not_treated_as_residue():
    # The window is finite on purpose. A headnote leaks AT the seam; a
    # judgment quoting an earlier report's "HELD:" a page later is ordinary
    # legal writing, and refusing those documents would cost more than the
    # check is worth.
    quotation = "HELD: 1. The seniority of a promotee is reckoned from regular appointment.\n"
    # BOUNDED, so that a mutant widening the window cannot make this fixture
    # allocate its way out of failing (the harness note from Task 10: a test
    # parametrised by the value under mutation thrashes instead of dying).
    body = _paras(1, min(RESIDUE_WINDOW, 20_000) + 800) + quotation
    result = extract_text(["The Judgment of the Court was delivered by\nNAVIN SINHA, J.\n" + body])

    assert result.ok, result.reason
    assert "HELD:" in result.text
    # ... and the premise that makes this test about the WINDOW: the phrase
    # really is past it.
    assert result.text.index("HELD:") > RESIDUE_WINDOW


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
        ("J U D G M E N T", "judgment_heading"),
        ("JUDGMENT", "judgment_heading"),
        ("## JUDGMENT", "judgment_heading"),
        ("O R D E R", "judgment_heading"),
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
    assert headnote_signals("Held: that the suit was barred by limitation.") == ()


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


def test_the_scr_margin_letters_go_and_a_lettered_heading_stays():
    # P0 found "A B C D E F G H" running down the left margin of these
    # reprints - and lettered section headings in the same alphabet.
    page = "A\nB.\nA. FACTUAL MATRIX\n1. The appellant filed a writ petition.\nH\n"
    cleaned, stats = clean_pages([page])

    assert "A. FACTUAL MATRIX" in cleaned[0]
    assert "1. The appellant filed a writ petition." in cleaned[0]
    assert [line for line in cleaned[0].split("\n") if line.strip() in {"A", "B.", "H"}] == []
    assert stats["margin_letter"] == 3


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

    assert stats["footnote"] == 1
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

    assert stats["footnote"] == 0
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

    assert stats["footnote"] == 0
    assert "13. We turn to the evidence" in cleaned[0]
    assert "Gurmit Singh" in cleaned[0]


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
    assert stats["footnote"] == 3
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


def test_markdown_decoration_is_demoted_without_eating_the_filename_underscores():
    assert demote_markdown("**HELD:** 1. The appeal") == "HELD: 1. The appeal"
    assert demote_markdown("# J U D G M E N T") == "J U D G M E N T"
    assert demote_markdown("*emphasis* here") == "emphasis here"
    # The S.C.R. object stem is all underscores and must survive intact,
    # which an italic rule written as a bare "_" strip would not leave it.
    assert demote_markdown("2020_7_941_960_EN.pdf") == "2020_7_941_960_EN.pdf"
    assert demote_markdown("-----") == ""


# --------------------------------------------------------------------------
# The join, the resume decision, and the run.
# --------------------------------------------------------------------------

import json
import os
from pathlib import Path

from pipeline_fakes import temp_config

from tuned.data.acquire import SC_LICENSE, SC_SOURCE_ID
from tuned.data.extract import (
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
    resolve_pdf,
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
    dest = tmp_path / "out" / "a.txt"

    def dying_replace(src, dst):
        raise OSError("killed between the write and the rename")

    monkeypatch.setattr(os, "replace", dying_replace)
    with pytest.raises(OSError):
        write_text(dest, "the judgment text")

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


# ---------------------------------------------------------------------- CLI

def test_cli_hard_exits_after_success():
    assert "os._exit(" in EXTRACT_SRC.read_text(encoding="utf-8")


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
