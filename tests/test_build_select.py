"""select.py - which judgments enter the corpus.

Offline throughout: the metadata rows are dicts, the landmark list is a
frozenset of normalised titles, and the one test that reads real parquet
writes the file itself.
"""

from pathlib import Path

import pytest
from pipeline_fakes import temp_config

from tuned.data.acquire import SC_SOURCE_ID
from tuned.data.select import (
    CITATION,
    CONSTITUTION_BENCH,
    CORAM,
    LANDMARK,
    SELECTION_FIELDS,
    SIGNAL_ORDER,
    case_type_of,
    citation_of,
    coram_size,
    english_available,
    judges_of,
    landmark_key,
    landmark_keys,
    main,
    metadata_rows,
    pdf_key_of,
    scr_prefix,
    select_corpus,
    select_judgment,
    selection_row,
    stratified_take,
    year_of,
)
from tuned.data.store import Store

SELECT_SRC = Path(__file__).parent.parent / "src" / "tuned" / "data" / "select.py"

FIVE_JUDGES = (
    "Hon'ble Mr. Justice A.K. Sikri, Hon'ble Mr. Justice S. Abdul Nazeer, "
    "Justice R.K. Agrawal, Justice A.M. Khanwilkar and Justice Ashok Bhushan"
)


def _meta_row(**overrides) -> dict:
    row = {
        "case_id": "Civil Appeal No. 7138 of 2010",
        "title": "Government of India & Ors vs Isro Drivers Association",
        "citation": "[2020] 7 S.C.R. 941",
        "judge": "Justice Sanjay Kishan Kaul, Justice Ajay Rastogi",
        "author_judge": "Justice Sanjay Kishan Kaul",
        "available_languages": "en",
        "court": "Supreme Court of India",
        "year": 2020,
    }
    row.update(overrides)
    return row


@pytest.fixture
def store(tmp_path):
    with Store.open(tmp_path / "state" / "law_v1.sqlite3") as s:
        yield s


# --------------------------------------------------------------------------
# The two constraints that overturned an earlier assumption.
# --------------------------------------------------------------------------

def test_the_corrupt_upstream_outcome_field_is_never_read():
    # Upstream repo issue #29, unresolved, 32 documented contradictions:
    # the field cannot be used as an outcome label, and operative outcomes
    # come from the judgment's own conclusion later in the pipeline. The
    # plan asks for this as a grep, and a grep is only honest if the name
    # appears nowhere in the module - comments and docstrings included.
    assert "disposal_nature" not in SELECT_SRC.read_text(encoding="utf-8")


def test_the_line_one_reportable_flag_is_not_a_selection_signal():
    # P0 raw-substring-searched all 70 sampled PDFs and found it in 2. The
    # bucket ships the S.C.R. typeset reprint, which does not carry the
    # flag; building selection on it would silently select nothing. It stays
    # worth capturing opportunistically during EXTRACTION - a different
    # module - so this pins the absence here only.
    assert "REPORTABLE" not in SELECT_SRC.read_text(encoding="utf-8")


def test_a_corrupt_upstream_column_cannot_leak_through_the_projection():
    # Not reading the field is not enough on its own: a projection that
    # copied the row would carry the value downstream anyway.
    row = _meta_row()
    row["disposal_nature"] = "Appeal allowed"
    row["raw_html"] = "<html>...</html>"
    out = selection_row(row, select_judgment(row))
    assert set(out) == set(SELECTION_FIELDS)
    assert "Appeal allowed" not in str(out)
    assert "<html>" not in str(out)


# --------------------------------------------------------------------------
# Signal 1: the court's own significance filter.
# --------------------------------------------------------------------------

def test_citation_is_read_off_the_row():
    assert citation_of(_meta_row()) == "[2020] 7 S.C.R. 941"


@pytest.mark.parametrize("value", [None, "", "   ", "-", "NA", "n/a", "None", "NULL"])
def test_a_null_shaped_citation_is_no_citation(value):
    # Parquet exported from scraped data spells "absent" several ways; a
    # literal "NA" read as a citation would select the whole corpus.
    assert citation_of(_meta_row(citation=value)) is None


def test_whitespace_inside_a_citation_is_collapsed_not_dropped():
    assert citation_of(_meta_row(citation="  (2008)   1  SCC   1 ")) == "(2008) 1 SCC 1"


# --------------------------------------------------------------------------
# Signal 2: coram.
# --------------------------------------------------------------------------

def test_a_constitution_bench_is_counted_off_the_judge_string():
    assert coram_size(_meta_row(judge=FIVE_JUDGES, author_judge=None)) == 5
    assert CONSTITUTION_BENCH == 5


def test_the_author_is_not_counted_twice_however_it_is_spelled():
    # "A.K. Sikri" in the bench string and "A K SIKRI" as the author is one
    # judge; counting it twice turns a 4-judge bench into a Constitution one.
    row = _meta_row(judge=FIVE_JUDGES, author_judge="A K SIKRI")
    assert coram_size(row) == 5


def test_an_author_absent_from_the_bench_string_is_still_a_judge():
    row = _meta_row(judge="Justice A.K. Sikri", author_judge="Justice Ashok Bhushan")
    assert coram_size(row) == 2


def test_a_list_valued_judge_field_is_read_as_a_list():
    # Surname-first entries carry their own commas, so a list stringified
    # and then split on commas reads as four judges rather than two.
    row = _meta_row(judge=["Sikri, A.K.", "Bhushan, Ashok"], author_judge=None)
    assert coram_size(row) == 2


def test_judges_are_returned_readable_not_normalised():
    row = _meta_row(judge="Hon'ble Mr. Justice A.K. Sikri", author_judge=None)
    assert judges_of(row) == ("Hon'ble Mr. Justice A.K. Sikri",)


@pytest.mark.parametrize("value", [None, "", "  ", "Justice"])
def test_an_empty_bench_is_a_coram_of_zero(value):
    assert coram_size(_meta_row(judge=value, author_judge=None)) == 0


# --------------------------------------------------------------------------
# Signal 3: the pre-computed landmark list.
# --------------------------------------------------------------------------

def test_landmark_key_survives_the_ways_a_case_name_is_written():
    a = landmark_key("Kesavananda Bharati vs. State of Kerala & Ors.")
    b = landmark_key("KESAVANANDA BHARATI  v  STATE OF KERALA and Another")
    assert a == b
    assert a != landmark_key("Kesavananda Bharati vs State of Karnataka")


def test_landmark_keys_are_built_off_the_injudgements_title_column():
    rows = [{"Titles": "Maneka Gandhi vs Union Of India"}, {"Titles": ""}, {"other": "x"}]
    keys = landmark_keys(rows)
    assert keys == {landmark_key("Maneka Gandhi v. Union of India")}


# --------------------------------------------------------------------------
# Filters that are metadata questions, not extraction problems.
# --------------------------------------------------------------------------

def test_english_availability_is_read_off_the_language_field():
    assert english_available(_meta_row(available_languages="en")) is True
    assert english_available(_meta_row(available_languages=["hi", "en"])) is True
    assert english_available(_meta_row(available_languages="hi,mr")) is False
    assert english_available(_meta_row(available_languages="english")) is True


def test_a_row_that_does_not_say_what_language_it_is_gets_the_benefit_of_the_doubt():
    # The english/ vs regional/ split in the PDF prefix has already filtered
    # these; an absent column must not empty the corpus.
    assert english_available(_meta_row(available_languages=None)) is None
    assert select_judgment(_meta_row(available_languages=None)).selected is True


def test_a_regional_only_judgment_is_filtered_out_here_not_downstream():
    decision = select_judgment(_meta_row(available_languages="hi"))
    assert decision.selected is False
    assert decision.reason == "not_english"


def test_year_is_read_from_the_partition_or_from_a_date():
    assert year_of({"year": 2015}) == 2015
    assert year_of({"decision_date": "2015-03-12"}) == 2015
    assert year_of({"date": "12-03-2015"}) == 2015
    assert year_of({"title": "no date here"}) is None


def test_the_scope_is_the_years_acquire_downloads():
    assert select_judgment(_meta_row(year=2009)).reason == "out_of_scope_year"
    assert select_judgment(_meta_row(year=2026)).reason == "out_of_scope_year"
    assert select_judgment(_meta_row(year=2010)).selected is True


# --------------------------------------------------------------------------
# The decision itself.
# --------------------------------------------------------------------------

def test_a_reported_judgment_is_selected_on_the_citation_alone():
    decision = select_judgment(_meta_row(judge=None, author_judge=None))
    assert decision.selected is True
    assert decision.signals == (CITATION,)


def test_an_unreported_two_judge_bench_nobody_cites_is_not_selected():
    decision = select_judgment(_meta_row(citation=None))
    assert decision.selected is False
    assert decision.reason == "no_significance_signal"


def test_a_constitution_bench_is_selected_without_a_citation():
    decision = select_judgment(_meta_row(citation=None, judge=FIVE_JUDGES, author_judge=None))
    assert decision.selected is True
    assert decision.signals == (CORAM,)


def test_a_landmark_is_selected_without_a_citation_or_a_large_bench():
    row = _meta_row(citation=None)
    landmarks = frozenset({landmark_key(row["title"])})
    decision = select_judgment(row, landmarks=landmarks)
    assert decision.selected is True
    assert decision.signals == (LANDMARK,)


def test_signals_come_back_in_the_order_the_research_ranked_them():
    row = _meta_row(judge=FIVE_JUDGES, author_judge=None)
    decision = select_judgment(row, landmarks=frozenset({landmark_key(row["title"])}))
    assert decision.signals == SIGNAL_ORDER == (CITATION, CORAM, LANDMARK)


def test_the_citation_outranks_a_bench_and_a_landmark_together():
    # Priority only decides what an interrupted or capped run keeps, but it
    # has to encode the ranking the research actually found: the court's own
    # reportable-decisions filter is the primary signal.
    reported = select_judgment(_meta_row(judge=None, author_judge=None))
    row = _meta_row(citation=None, judge=FIVE_JUDGES, author_judge=None)
    both_others = select_judgment(row, landmarks=frozenset({landmark_key(row["title"])}))
    assert both_others.signals == (CORAM, LANDMARK)
    assert reported.priority > both_others.priority


def test_the_case_type_strata_come_off_the_title_and_the_case_id():
    assert case_type_of(_meta_row(case_id="Criminal Appeal No. 55 of 2015", title="X v State")) == "criminal"
    assert case_type_of(_meta_row(title="Writ Petition under Article 32", case_id="WP 1/2015")) == "constitutional"
    assert case_type_of(_meta_row(title="Dispute over an arbitration clause")) == "commercial"
    assert case_type_of(_meta_row()) == "civil"


def test_the_case_type_never_filters_anything_out():
    # Stratification, not selection: every stratum is kept.
    rows = [
        _meta_row(case_id=f"Criminal Appeal No. {i}", title="X v State") for i in range(3)
    ] + [_meta_row(case_id=f"Civil Appeal No. {i}") for i in range(3)]
    chosen, stats = select_corpus(rows)
    assert stats["selected"] == 6
    assert stats["by_stratum"] == {"criminal": 3, "civil": 3}


# --------------------------------------------------------------------------
# Links to the PDF side (hints for extraction, never a filter).
# --------------------------------------------------------------------------

def test_the_scr_citation_gives_the_pdf_filename_prefix():
    # P0: english PDFs are named {year}_{volume}_{startpage}_{endpage}_EN.pdf
    # and that is S.C.R. pagination, so the citation addresses the file.
    assert scr_prefix("[2020] 7 S.C.R. 941") == "2020_7_941_"
    assert scr_prefix("2020 7 SCR 941") == "2020_7_941_"
    # The bracket is NOT what tells the reporters apart - only the name is,
    # so the negative case has to differ in the name alone.
    assert scr_prefix("(2020) 7 SCR 941") == "2020_7_941_"
    assert scr_prefix("(2008) 1 SCC 1") is None
    assert scr_prefix("[2008] 1 SCC 1") is None
    assert scr_prefix(None) is None


def test_a_pdf_link_is_reduced_to_a_bucket_key():
    assert pdf_key_of({"pdf_link": "https://indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com/data/pdf/year=2020/english/2020_7_941_960_EN.pdf"}) == (
        "data/pdf/year=2020/english/2020_7_941_960_EN.pdf"
    )
    assert pdf_key_of({"raw_file_path": "/data/pdf/year=2020/english/x.pdf"}) == (
        "data/pdf/year=2020/english/x.pdf"
    )
    assert pdf_key_of({"pdf_link": "not-a-pdf.html"}) is None
    assert pdf_key_of({}) is None


def test_a_row_with_no_usable_pdf_link_is_still_selected():
    # The metadata schema is not verified offline. Making the link mandatory
    # would let one wrong field name select nothing at all.
    decision = select_judgment(_meta_row())
    assert decision.selected is True
    assert selection_row(_meta_row(), decision)["pdf_key"] is None


# --------------------------------------------------------------------------
# Whole-corpus assembly.
# --------------------------------------------------------------------------

def test_select_corpus_counts_what_it_kept_and_why_it_dropped_the_rest():
    rows = [
        _meta_row(),                                        # citation
        _meta_row(citation=None),                           # nothing
        _meta_row(year=2005),                               # out of scope
        _meta_row(available_languages="hi"),                # regional
        _meta_row(citation=None, judge=FIVE_JUDGES, author_judge=None),  # coram
    ]
    chosen, stats = select_corpus(rows)
    assert stats["total"] == 5
    assert stats["selected"] == 2
    assert stats["rejects"] == {
        "no_significance_signal": 1,
        "out_of_scope_year": 1,
        "not_english": 1,
    }
    assert stats["by_signal"] == {CITATION: 1, CORAM: 1}
    assert len(chosen) == 2


def test_the_degraded_selection_says_so_instead_of_looking_complete():
    # The landmark list is gated. A run without it is a WEAKER selection,
    # and silently returning the same shape is how that gets forgotten.
    _, without = select_corpus([_meta_row()])
    assert without["degraded"] is True
    assert without["landmarks"] is None
    assert without["landmark_matches"] == 0

    _, with_list = select_corpus([_meta_row()], landmarks=frozenset({"a title"}))
    assert with_list["degraded"] is False
    assert with_list["landmarks"] == 1


def test_the_landmark_match_rate_is_reported_because_the_join_is_by_title():
    rows = [_meta_row(title="Maneka Gandhi vs Union Of India"), _meta_row(title="Other v Case")]
    landmarks = landmark_keys([{"Titles": "Maneka Gandhi v. Union of India"}])
    _, stats = select_corpus(rows, landmarks=landmarks)
    # A title join can silently match nothing; the rate is the only way a
    # real run finds that out.
    assert stats["landmark_matches"] == 1
    assert stats["by_signal"][LANDMARK] == 1


def test_field_coverage_reports_which_column_names_actually_matched():
    # The parquet schema is not verified offline. If "judge" turns out to be
    # called something else, coverage says so on the first real run instead
    # of the corpus quietly having no Constitution Benches in it.
    rows = [_meta_row(), _meta_row(judge=None, author_judge=None), _meta_row(citation=None)]
    _, stats = select_corpus(rows)
    assert stats["field_coverage"]["citation"] == 2
    assert stats["field_coverage"]["judge"] == 2
    assert stats["field_coverage"]["available_languages"] == 3


def test_a_capped_run_takes_from_every_stratum_not_just_the_first():
    rows = [_meta_row(case_id=f"Civil Appeal {i}") for i in range(10)]
    rows += [_meta_row(case_id=f"Criminal Appeal {i}", title="X v State") for i in range(2)]
    chosen, stats = select_corpus(rows, limit=4)
    assert stats["matched"] == 12
    assert stats["selected"] == 4
    assert stats["by_stratum"] == {"civil": 2, "criminal": 2}
    assert len(chosen) == 4


def test_stratified_take_prefers_the_stronger_signals_within_a_stratum():
    weak = {"case_type": "civil", "priority": 4, "case_id": "weak"}
    strong = {"case_type": "civil", "priority": 6, "case_id": "strong"}
    assert [r["case_id"] for r in stratified_take([weak, strong], 1)] == ["strong"]


def test_stratified_take_returns_everything_when_the_cap_is_not_binding():
    rows = [{"case_type": "civil", "priority": 4, "case_id": str(i)} for i in range(3)]
    assert len(stratified_take(rows, 10)) == 3


def test_a_selection_row_carries_what_extraction_needs():
    row = _meta_row(judge=FIVE_JUDGES, author_judge=None)
    out = selection_row(row, select_judgment(row))
    assert out["case_id"] == row["case_id"]
    assert out["citation"] == row["citation"]
    assert out["year"] == 2020
    assert out["coram"] == 5
    assert out["signals"] == [CITATION, CORAM]
    assert out["scr_prefix"] == "2020_7_941_"
    assert out["source_id"] == SC_SOURCE_ID


# --------------------------------------------------------------------------
# Reading what acquire indexed.
# --------------------------------------------------------------------------

def test_metadata_rows_reads_the_parquet_acquire_recorded(store, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")

    store.upsert_source(SC_SOURCE_ID, "CC-BY-4.0")
    for year in (2015, 2005):
        path = tmp_path / f"{year}.parquet"
        pq.write_table(
            pa.table({"case_id": [f"CA {year}"], "citation": [f"[{year}] 1 S.C.R. 1"]}), path
        )
        store.record_artifact(
            SC_SOURCE_ID,
            f"metadata/parquet/year={year}/part-0.parquet",
            local_path=path,
            size_bytes=path.stat().st_size,
            sha256="x",
        )

    rows = list(metadata_rows(store, (2015,)))
    # The out-of-scope partition is not even opened, and the year the row
    # sits in is carried across from the key.
    assert [r["case_id"] for r in rows] == ["CA 2015"]
    assert rows[0]["year"] == 2015


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def test_cli_writes_the_selection_and_records_the_run(tmp_path, capsys):
    from tuned.data.config import load_build_config
    from tuned.data.jsonl import read_jsonl
    from tuned.data.paths import build_paths

    config = temp_config(tmp_path)
    rows = [_meta_row(case_id=f"Civil Appeal {i}") for i in range(3)] + [_meta_row(citation=None)]

    assert main(["--config", config], rows=rows, landmarks=frozenset()) == 0

    cfg = load_build_config(config, allow_unpinned=True)
    paths = build_paths(cfg.build.workdir)
    written = list(read_jsonl(paths.corpus_dir / "selection.jsonl"))
    assert len(written) == 3
    assert set(written[0]) == set(SELECTION_FIELDS)

    with Store.open(paths.state_db) as opened:
        events = opened.events("corpus_selection")
        assert len(events) == 1
        assert '"selected": 3' in events[0]["detail_json"]
    out = capsys.readouterr().out
    assert "no_significance_signal" in out
    # The pair with the next test is what makes the degraded banner mean
    # something: it is absent exactly when the landmark list was there.
    assert "DEGRADED" not in out


def test_cli_says_loudly_when_it_ran_without_the_landmark_list(tmp_path, capsys):
    code = main(["--config", temp_config(tmp_path)], rows=[_meta_row()], landmarks=None)
    out = capsys.readouterr().out
    assert code == 0
    assert "DEGRADED" in out
    assert "opennyaiorg/InJudgements_dataset" in out


def test_cli_hard_exits_after_success():
    assert "os._exit(" in SELECT_SRC.read_text(encoding="utf-8")
