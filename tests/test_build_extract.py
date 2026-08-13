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
    Q_BODY_TOO_SHORT,
    Q_HEADNOTE_RESIDUE,
    Q_LOW_TEXT_QUALITY,
    Q_NO_JUDGMENT_START,
    Q_NO_TEXT,
    Q_STRIP_TOO_LARGE,
    clean_pages,
    demote_markdown,
    extract_text,
    find_judgment_start,
    headnote_signals,
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
    soup = "(cid:24)(cid:37)(cid:12)(cid:3)(cid:55)(cid:82) " * 200
    page = "The Judgment of the Court was delivered by\n" + soup
    result = extract_text([page])

    assert not result.ok
    assert result.reason == Q_LOW_TEXT_QUALITY
    # Both of the other refusals would be a misdiagnosis of this document,
    # and each is ruled out by a fact about the fixture rather than by the
    # reason string this test already asserts.
    assert len(page) > MIN_DOC_CHARS
    assert find_judgment_start(page) is not None


def test_a_regional_text_layer_is_quarantined():
    devanagari = "यह निर्णय हिंदी में है और यह अंग्रेजी उपसर्ग के अंतर्गत नहीं होना चाहिए। " * 40
    result = extract_text(["The Judgment of the Court was delivered by\n" + devanagari])

    assert not result.ok
    assert result.reason == Q_LOW_TEXT_QUALITY


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
    assert stats["signature"] >= 5


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
            f"{i + 1}. Paragraph {i + 1} of the judgment, which appears on one page only "
            f"and differs from its neighbours in nothing but the number it carries.\n"
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
    # as furniture - and the deletion is silent.
    for i in range(6):
        assert f"Paragraph {i + 1} of the judgment" in joined
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


def test_markdown_decoration_is_demoted_without_eating_the_filename_underscores():
    assert demote_markdown("**HELD:** 1. The appeal") == "HELD: 1. The appeal"
    assert demote_markdown("# J U D G M E N T") == "J U D G M E N T"
    assert demote_markdown("*emphasis* here") == "emphasis here"
    # The S.C.R. object stem is all underscores and must survive intact,
    # which an italic rule written as a bare "_" strip would not leave it.
    assert demote_markdown("2020_7_941_960_EN.pdf") == "2020_7_941_960_EN.pdf"
    assert demote_markdown("-----") == ""
