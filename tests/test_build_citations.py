import ast
from pathlib import Path

import pytest

from tuned.data.citations import (
    CITATION_PATTERNS,
    CitationIndex,
    citations_from_row,
    extract_citations,
    normalize,
    novel_citations,
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
        # AIR - court token case and dots
        ("AIR 1973 SC 1461", "AIR 1973 SC 1461"),
        ("air  1973  s.c.  1461", "AIR 1973 SC 1461"),
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
        "air 1973 s.c. 1461",
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
    assert set(CITATION_PATTERNS) == {"insc", "hc_neutral", "scc", "air", "scr"}


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
