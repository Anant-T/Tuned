import ast
import time
from pathlib import Path

import pytest

from tuned.data.citations import (
    CITATION_PATTERNS,
    CitationIndex,
    citations_from_row,
    extract_citations,
    normalize,
    novel_citations,
    suspect_citations,
    suspect_key,
)

CITATIONS_SRC = Path(__file__).parent.parent / "src" / "tuned" / "data" / "citations.py"


# --------------------------------------------------------------------------
# normalize: one canonical spelling per citation, whatever the input form.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        # SC neutral - leading zeros and case
        ("2023 INSC 45", "2023 INSC 45"),
        ("2023 INSC 0045", "2023 INSC 45"),
        ("2023 insc 45", "2023 INSC 45"),
        ("  2023   INSC   45  ", "2023 INSC 45"),
        # HC neutral - court code case, leading zeros, bench suffix separator
        ("2023:DHC:2720", "2023:DHC:2720"),
        ("2023:dhc:02720", "2023:DHC:2720"),
        ("2023:DHC:2720-DB", "2023:DHC:2720:DB"),
        ("2023:dhc:2720:db", "2023:DHC:2720:DB"),
        ("2024:KER:12345", "2024:KER:12345"),
        # SCC - spacing and case
        ("(2008) 1 SCC 1", "(2008) 1 SCC 1"),
        ("(2008)   1   SCC   1", "(2008) 1 SCC 1"),
        ("( 2008 ) 1 scc 1", "(2008) 1 SCC 1"),
        ("2008 1 SCC 1", "(2008) 1 SCC 1"),
        # AIR - court token case and dots (the AIR token itself is
        # case-SENSITIVE, so "clean air 2019 ..." is not a citation)
        ("AIR 1973 SC 1461", "AIR 1973 SC 1461"),
        ("AIR  1973  s.c.  1461", "AIR 1973 SC 1461"),
        ("AIR 1973 Del 1461", "AIR 1973 DEL 1461"),
        # SCR - parenthesised and bare year both canonicalise with parens
        ("(1974) 2 SCR 348", "(1974) 2 SCR 348"),
        ("1974 2 SCR 348", "(1974) 2 SCR 348"),
        ("(1974) 02 scr 0348", "(1974) 2 SCR 348"),
    ],
)
def test_normalize_known_formats(raw, expected):
    assert normalize(raw) == expected


def test_normalize_equivalence_classes():
    assert normalize("2023 INSC 45") == normalize("2023 INSC 0045")
    assert normalize("2023:DHC:2720") == normalize("2023:dhc:02720")
    assert normalize("(2008) 1 SCC 1") == normalize("2008  1  scc  1")
    assert normalize("(1974) 2 SCR 348") == normalize("1974 2 SCR 348")
    # different cases must NOT collide
    assert normalize("2023 INSC 45") != normalize("2023 INSC 46")
    assert normalize("2023:DHC:2720") != normalize("2023:BHC:2720")


def test_normalize_unknown_format_is_an_opaque_key():
    assert normalize("Kesavananda  Bharati v.  State of Kerala") == "KESAVANANDA BHARATI V. STATE OF KERALA"
    assert normalize("") == ""
    assert normalize("   ") == ""


@pytest.mark.parametrize(
    "raw",
    [
        "2023 INSC 0045",
        "2023:dhc:02720",
        "2008 1 SCC 1",
        "AIR 1973 s.c. 1461",
        "1974 2 SCR 348",
        "Some Unknown  Case",
    ],
)
def test_normalize_is_idempotent(raw):
    once = normalize(raw)
    assert normalize(once) == once


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def test_pattern_keys():
    # the five briefed formats, plus the two the review added because their
    # absence made fabrications in those formats invisible to the gate
    assert {"insc", "hc_neutral", "scc", "air", "scr"} <= set(CITATION_PATTERNS)
    assert set(CITATION_PATTERNS) == {
        "insc", "hc_neutral", "scc_online", "scc", "air", "scr", "crilj"
    }


def test_extract_all_formats_in_order_deduped():
    text = (
        "The Bench in 2023 INSC 45 followed (2008) 1 SCC 1 and AIR 1973 SC 1461, "
        "as the Delhi High Court held in 2023:DHC:2720; see also (1974) 2 SCR 348 "
        "and again (2008)  1  scc  1."
    )
    assert extract_citations(text) == [
        "2023 INSC 45",
        "(2008) 1 SCC 1",
        "AIR 1973 SC 1461",
        "2023:DHC:2720",
        "(1974) 2 SCR 348",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "2023 INSC",
        "INSC 45",
        "the year 2023 and the number 45",
        "invoice 12345 dated 2023",
        "para 1461 of AIR",
        "2023:DHC",
        "DHC:2720",
        "the ratio was 2:1 in 2023",
        "(2008) 1 report at page 1",
    ],
)
def test_extract_negatives(text):
    assert extract_citations(text) == []


def test_extract_never_carves_a_citation_out_of_a_longer_number():
    # "2023 INSC 45" must NOT be extracted from "2023 INSC 4512".
    assert extract_citations("relied on 2023 INSC 4512") == ["2023 INSC 4512"]
    assert extract_citations("2023 INSC 45 and 2023 INSC 4512") == ["2023 INSC 45", "2023 INSC 4512"]
    # a year glued to a preceding digit is not a citation at all
    assert extract_citations("12023 INSC 45") == []
    assert extract_citations("(2008) 1 SCC 12") == ["(2008) 1 SCC 12"]
    assert extract_citations("2023:DHC:27201") == ["2023:DHC:27201"]
    assert extract_citations("x2023:DHC:2720") == []


def test_extract_empty_text():
    assert extract_citations("") == []
    assert extract_citations(None) == []


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------

def test_index_build_writes_sorted_lf_text(tmp_path):
    out = tmp_path / "corpus" / "citation_index.txt"
    index = CitationIndex.build(["2023 INSC 45", "AIR 1973 SC 1461", "(2008) 1 SCC 1"], out)
    raw = out.read_bytes()
    assert b"\r\n" not in raw  # portable LF, even when built on Windows
    lines = raw.decode("utf-8").splitlines()
    assert lines == sorted(lines)
    assert set(lines) == {"2023 INSC 45", "AIR 1973 SC 1461", "(2008) 1 SCC 1"}
    assert len(index) == 3
    assert out.parent.is_dir()


def test_index_build_normalizes_and_dedupes(tmp_path):
    out = tmp_path / "index.txt"
    index = CitationIndex.build(
        ["2023 INSC 0045", "2023 INSC 45", "  2023 insc 45 ", "", "2023:dhc:02720"], out
    )
    assert len(index) == 2
    assert out.read_text(encoding="utf-8").splitlines() == ["2023 INSC 45", "2023:DHC:2720"]


def test_index_round_trip_and_contains_normalizes_the_query(tmp_path):
    out = tmp_path / "index.txt"
    CitationIndex.build(["2023 INSC 45", "(1974) 2 SCR 348"], out)
    loaded = CitationIndex.load(out)
    assert len(loaded) == 2
    assert loaded.contains("2023 INSC 45")
    assert loaded.contains("2023 INSC 0045")  # raw input is normalized first
    assert loaded.contains("1974 2 SCR 348")
    assert "  2023   insc   45 " in loaded
    assert not loaded.contains("2023 INSC 46")
    assert not loaded.contains("")


def test_index_leaves_no_tmp_file(tmp_path):
    out = tmp_path / "index.txt"
    CitationIndex.build(["2023 INSC 45"], out)
    assert list(tmp_path.iterdir()) == [out]


# --------------------------------------------------------------------------
# novel_citations - the rejection gate primitive
# --------------------------------------------------------------------------

def _index(tmp_path, citations):
    return CitationIndex.build(citations, tmp_path / "index.txt")


def test_novel_citations_in_index_is_allowed(tmp_path):
    index = _index(tmp_path, ["2023 INSC 45"])
    assert novel_citations("As held in 2023 INSC 0045, ...", "", index) == []


def test_novel_citations_only_in_context_is_allowed(tmp_path):
    index = _index(tmp_path, ["2023 INSC 45"])
    text = "The extract relies on (2008) 1 SCC 1."
    context = "Source judgment reported at (2008)  1  SCC  1."
    assert novel_citations(text, context, index) == []


def test_novel_citations_nowhere_is_returned(tmp_path):
    index = _index(tmp_path, ["2023 INSC 45"])
    text = "See 2023 INSC 45, (2008) 1 SCC 1 and AIR 1973 SC 1461."
    context = "Grounding passage mentioning (2008) 1 SCC 1 only."
    assert novel_citations(text, context, index) == ["AIR 1973 SC 1461"]


def test_novel_citations_preserves_order_and_dedupes(tmp_path):
    index = _index(tmp_path, [])
    text = "First 2023:DHC:2720, then AIR 1973 SC 1461, then 2023:dhc:2720 again."
    assert novel_citations(text, "", index) == ["2023:DHC:2720", "AIR 1973 SC 1461"]


def test_novel_citations_clean_text(tmp_path):
    index = _index(tmp_path, ["2023 INSC 45"])
    assert novel_citations("No authority is cited at all.", "", index) == []


# --------------------------------------------------------------------------
# corpus ingestion - offline only
# --------------------------------------------------------------------------

def test_citations_from_row_reads_only_citation_columns():
    row = {
        "neutral_citation": "2023 INSC 0045",
        "law_report_citation": "(2008) 1 SCC 1",
        "headnote_text": "AIR 1973 SC 1461 - copyrighted editorial matter, never indexed",
    }
    assert citations_from_row(row) == ["2023 INSC 45", "(2008) 1 SCC 1"]


def test_citations_from_row_handles_missing_and_unparseable_values():
    assert citations_from_row({}) == []
    assert citations_from_row({"neutral_citation": None, "law_report_citation": "  "}) == []
    # an unrecognised reporter still becomes an opaque key rather than vanishing
    assert citations_from_row({"law_report_citation": "1998 Cri LJ  1234"}) == ["1998 CRI LJ 1234"]


def test_citations_from_row_splits_multiple_citations_in_one_cell():
    row = {"law_report_citation": "(2008) 1 SCC 1 : AIR 2008 SC 12"}
    assert citations_from_row(row) == ["(2008) 1 SCC 1", "AIR 2008 SC 12"]


def test_headnote_column_is_declared_forbidden_and_never_ingested():
    import tuned.data.citations as citations

    assert "headnote_text" in citations._FORBIDDEN_COLUMNS
    assert "headnote_text" not in citations._CITATION_COLUMNS


# --------------------------------------------------------------------------
# Review fix 1: recall holes. A citation this module cannot parse is a
# citation the index is never asked about, so an unmatched fabrication used
# to pass the hard gate in silence.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        # volume in parentheses
        ("2008 (1) SCC 77", "(2008) 1 SCC 77"),
        ("2008 ( 1 ) SCC 77", "(2008) 1 SCC 77"),
        ("1974 (2) SCR 348", "(1974) 2 SCR 348"),
        # dotted reporter tokens
        ("(2008) 1 S.C.C. 55", "(2008) 1 SCC 55"),
        ("(2008) 1 S. C. C. 55", "(2008) 1 SCC 55"),
        ("1974 (2) S.C.R. 348", "(1974) 2 SCR 348"),
        # long AIR court token
        ("AIR 2019 SUPREME COURT 9999", "AIR 2019 SC 9999"),
        ("AIR 2019 Supreme Court 9999", "AIR 2019 SC 9999"),
        ("AIR 2003 ALLAHABAD 12", "AIR 2003 ALLAHABAD 12"),
        # SCC OnLine
        ("2019 SCC OnLine SC 4321", "2019 SCC ONLINE SC 4321"),
        ("2019 SCC Online Del 12", "2019 SCC ONLINE DEL 12"),
        ("(2019) SCC OnLine Bom 7", "2019 SCC ONLINE BOM 7"),
        # Criminal Law Journal
        ("1980 Cri LJ 1440", "1980 CRI LJ 1440"),
        ("1980 CriLJ 1440", "1980 CRI LJ 1440"),
        ("1999 Cr.L.J. 12", "1999 CRI LJ 12"),
        ("2003 (2) Cri LJ 45", "2003 2 CRI LJ 45"),
    ],
)
def test_normalize_formats_added_by_review(raw, expected):
    assert normalize(raw) == expected
    assert extract_citations(f"as held in {raw}, the appeal fails") == [expected]


def test_review_added_formats_share_keys_with_their_plain_spellings():
    assert normalize("2008 (1) SCC 77") == normalize("(2008) 1 SCC 77")
    assert normalize("(2008) 1 S.C.C. 55") == normalize("(2008) 1 SCC 55")
    assert normalize("AIR 2019 SUPREME COURT 9999") == normalize("AIR 2019 SC 9999")
    assert normalize("1980 CrLJ 1440") == normalize("1980 Cri LJ 1440")
    assert normalize("1974 (2) S.C.R. 348") == normalize("(1974) 2 SCR 348")


# Six fabricated authorities in the formats that used to slip through. Every
# one must now be caught by ONE of the two channels: extracted (and killed on
# an index miss) or surfaced as citation-shaped-but-unparseable.
FABRICATIONS = [
    "2008 (1) SCC 77",
    "(2008) 1 S.C.C. 55",
    "AIR 2019 SUPREME COURT 9999",
    "2019 SCC OnLine SC 4321",
    "1980 Cri LJ 1440",
    "2011 (2) KLT 123",
]


@pytest.mark.parametrize("fabrication", FABRICATIONS)
def test_every_fabrication_is_caught_by_one_of_the_two_channels(tmp_path, fabrication):
    index = CitationIndex.build(["2023 INSC 45"], tmp_path / "index.txt")
    text = f"The proposition is settled by {fabrication}."
    caught = novel_citations(text, "", index) + suspect_citations(text)
    assert caught, f"{fabrication!r} passed the gate in silence"


def test_fabrications_in_the_index_or_context_are_still_allowed(tmp_path):
    index = CitationIndex.build(["2008 (1) SCC 77"], tmp_path / "index.txt")
    # same case, other spelling - normalization makes it one key
    assert novel_citations("see (2008) 1 SCC 77", "", index) == []
    assert novel_citations("see 1980 Cri LJ 1440", "source: 1980 CrLJ 1440", index) == []


@pytest.mark.parametrize(
    "text,expected",
    [
        ("reported at 2011 (2) KLT 123 today", ["2011 (2) KLT 123"]),
        ("see 2005 (3) MhLJ 45", ["2005 (3) MHLJ 45"]),
        ("(2003) 2 Bom CR 456 was followed", ["(2003) 2 BOM CR 456"]),
        ("2019 ALL MR 234 is on all fours", ["2019 ALL MR 234"]),
    ],
)
def test_suspect_citations_surfaces_unmodelled_reporters(text, expected):
    assert suspect_citations(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "In 2023 the court awarded 5 crore to the family",
        "In 2023, 45 IPC cases were filed",
        "the appeal of 2023 was decided in 45 days",
        "Kesavananda Bharati v. State of Kerala",
        "Criminal Appeal No. 1234 of 2023 decided on 15.03.2023",
        # already parsed by a known pattern - not a suspect
        "(2008) 1 SCC 1 and AIR 1973 SC 1461",
        "2023 INSC 45",
        "para 12 of 2019 SCC OnLine SC 4321",
        "2023:DHC:2720",
    ],
)
def test_suspect_citations_negatives(text):
    assert suspect_citations(text) == []


def test_suspect_citations_order_and_dedup():
    text = "see 2011 (2) KLT 123, then 2005 (3) MhLJ 45, then 2011 (2) KLT 123 again"
    assert suspect_citations(text) == ["2011 (2) KLT 123", "2005 (3) MHLJ 45"]


# --------------------------------------------------------------------------
# Review fix N1: the widened AIR court token must not eat prose. A phantom
# key ("AIR 1973 AT PAGE 1461") fails the existence gate and rejects a good
# example, so this is a false-REJECT bug, not a cosmetic one.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "reported in AIR 1973 at page 1461",
        "clean air 2019 standards mandate 45 units",  # AIR is case-sensitive
        "the air 2019 quality index fell 45 points",
        "see AIR 1973 see also 1461",
        "AIR 1973 of the report 1461",
    ],
)
def test_air_pattern_does_not_eat_prose(text):
    assert extract_citations(text) == []


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("AIR 1973 SC 1461", "AIR 1973 SC 1461"),
        ("AIR 2019 SUPREME COURT 9999", "AIR 2019 SC 9999"),
        ("AIR 2019 Supreme Court 9999", "AIR 2019 SC 9999"),
        ("AIR 2019 Supreme Court of India 9999", "AIR 2019 SC 9999"),
        ("AIR 2003 ALLAHABAD 12", "AIR 2003 ALLAHABAD 12"),
        ("AIR 2003 Allahabad High Court 12", "AIR 2003 ALLAHABAD HIGH COURT 12"),
    ],
)
def test_air_recognised_court_names_still_extract(raw, expected):
    assert normalize(raw) == expected
    assert extract_citations(f"followed in {raw}, at para 4") == [expected]


# --------------------------------------------------------------------------
# Review fix N3: citations wrap across indented lines.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("(2008) 1 SCC\n          77", ["(2008) 1 SCC 77"]),
        ("AIR\n   1973 SC 1461", ["AIR 1973 SC 1461"]),
        ("2023\n        INSC 45", ["2023 INSC 45"]),
        ("1980 Cri LJ\n        1440", ["1980 CRI LJ 1440"]),
    ],
)
def test_wrapped_citations_still_extract(text, expected):
    assert extract_citations(text) == expected


def test_extraction_stays_linear_on_degenerate_whitespace():
    text = "(2008) 1 SCC" + " " * 16000 + "77"
    start = time.perf_counter()
    extract_citations(text)
    suspect_citations(text)
    assert time.perf_counter() - start < 1.0


def test_no_module_level_dataset_import():
    """Importing this module must never touch the network - datasets is
    imported lazily inside the streaming helper, as in replay.py."""
    tree = ast.parse(CITATIONS_SRC.read_text(encoding="utf-8"))
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = set()
    for node in top_level:
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif node.module:
            names.add(node.module.split(".")[0])
    assert "datasets" not in names
    assert names <= {"os", "re", "collections", "pathlib"}


# --------------------------------------------------------------------------
# suspect_key - the comparison form for unmodelled reporters.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "a, b",
    [
        # The two pairs measured in the 2026-08-18 pilot, each of which burned
        # a seed on a PERMANENT gate.
        ("2015 (4) KLT 163", "2015(4) KLT 163"),
        ("(2006) 7 SCALE 28", "2006 (7) SCALE 28"),
        # The same two variations stated on their own.
        ("2011 (2) KLT 123", "2011(2) KLT 123"),
        ("(1914) A.C. 676", "1914 AC 676"),
    ],
)
def test_suspect_key_folds_bracket_and_space_variation(a, b):
    assert suspect_key(a) == suspect_key(b)
    # ...and the fold is to tokens, not to a blob.
    assert " " in suspect_key(a)


@pytest.mark.parametrize(
    "a, b",
    [
        # The volume/page digits shift, which a concatenating key would fold.
        ("2015 (4) KLT 163", "2015 (41) KLT 63"),
        ("2015 (4) KLT 163", "2016 (4) KLT 163"),
        ("2015 (4) KLT 163", "2015 (4) MhLJ 163"),
    ],
)
def test_suspect_key_keeps_different_citations_apart(a, b):
    assert suspect_key(a) != suspect_key(b)


def test_suspect_key_is_idempotent_and_total():
    assert suspect_key("") == ""
    assert suspect_key(None) == ""
    once = suspect_key("2015 (4) KLT 163")
    assert suspect_key(once) == once
