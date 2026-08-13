"""select.py - which judgments enter the corpus.

Offline throughout: the metadata rows are dicts, the landmark list is a
frozenset of normalised titles, and the one test that reads real parquet
writes the file itself.
"""

from pathlib import Path

import pytest
from pipeline_fakes import temp_config

from tuned.data.acquire import HF_SOURCES, SC_SOURCE_ID
from tuned.data.select import (
    CITATION,
    CONSTITUTION_BENCH,
    CORAM,
    LANDMARK,
    LANDMARKS_NO_TITLE_COLUMN,
    LANDMARKS_NOT_ACQUIRED,
    LANDMARKS_OK,
    SELECTION_FIELDS,
    SIGNAL_ORDER,
    case_type_of,
    citation_of,
    coram_size,
    english_available,
    judges_of,
    landmark_key,
    landmark_keys,
    landmark_set,
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


def test_field_coverage_counts_rows_where_the_signal_resolved():
    # The parquet schema is not verified offline. If "judge" turns out to be
    # called something else, coverage says so on the first real run instead
    # of the corpus quietly having no Constitution Benches in it.
    rows = [_meta_row(), _meta_row(judge=None, author_judge=None), _meta_row(citation=None)]
    _, stats = select_corpus(rows)
    assert stats["total"] == 3
    assert stats["field_coverage"]["citation"] == 2
    assert stats["field_coverage"]["judge"] == 2
    assert stats["field_coverage"]["language"] == 3


def test_coverage_counts_the_candidate_that_resolved_not_the_first_name_in_the_list():
    # THE CASE COVERAGE EXISTS FOR. Every signal reads through an ordered
    # candidate list, so counting the literal first name reports on candidate
    # #0 alone: "the fallback matched" and "nothing matched" both read 0, and
    # an operator reading citation=0.0 would conclude the exact opposite of
    # the truth. This row uses candidates #1 and #2 - the names P0 records
    # for this corpus - and BOTH signals fire on it.
    rows = [
        {
            "case_id": "Civil Appeal No. 7138 of 2010",
            "title": "Government of India vs Isro Drivers Association",
            "law_report_citation": "[2020] 7 S.C.R. 941",
            "coram_members": FIVE_JUDGES,
            "language_codes": "en",
            "year": 2020,
        }
    ]
    _, stats = select_corpus(rows)
    assert stats["selected"] == 1
    assert stats["by_signal"] == {CITATION: 1, CORAM: 1}

    # The number the whole citation-is-universal question turns on.
    assert stats["field_coverage"]["citation"] / stats["total"] == 1.0
    assert stats["field_coverage"]["judge"] == 1
    assert stats["field_coverage"]["language"] == 1
    # And the map that actually answers "did our names match?".
    assert stats["resolved_fields"]["citation"] == "law_report_citation"
    assert stats["resolved_fields"]["judge"] == "coram_members"
    assert stats["resolved_fields"]["language"] == "language_codes"


def test_a_signal_no_candidate_resolved_names_no_winner():
    rows = [{"mystery_column": "x", "year": 2015, "citation": "[2015] 1 S.C.R. 1"}]
    _, stats = select_corpus(rows)
    assert stats["field_coverage"]["citation"] == 1
    assert stats["resolved_fields"]["citation"] == "citation"
    # Nothing carried a bench, and the report says which name won rather than
    # leaving "no Constitution Benches" and "no such column" indistinguishable.
    assert stats["field_coverage"]["judge"] == 0
    assert stats["resolved_fields"]["judge"] is None


def test_the_uncapped_selection_is_ordered_too_not_left_in_parquet_order():
    # The weights exist so that a capped OR INTERRUPTED run keeps the
    # reported judgments before it keeps anything else. Extraction is the
    # next task, is costed at 4-6 days, WILL be interrupted, and consumes
    # this file top-down - and the documented default command has no --limit,
    # so ordering only under a cap means the default run does none.
    rows = [
        _meta_row(case_id="Civil Appeal weak-1", judge=None, author_judge=None),
        _meta_row(case_id="Civil Appeal weak-2", judge=None, author_judge=None),
        _meta_row(case_id="Civil Appeal STRONG", judge=FIVE_JUDGES, author_judge=None),
    ]
    chosen, stats = select_corpus(rows)
    assert stats["selected"] == 3
    assert [row["priority"] for row in chosen] == [6, 4, 4]
    assert [row["case_id"] for row in chosen] == [
        "Civil Appeal STRONG",
        "Civil Appeal weak-1",
        "Civil Appeal weak-2",
    ]


def test_a_capped_run_is_a_prefix_of_the_uncapped_one():
    # One ordering, not two: what a cap changes is where the file stops, so
    # an interrupted uncapped run and a capped run keep the same judgments.
    rows = [_meta_row(case_id=f"Civil Appeal {i}", judge=None, author_judge=None) for i in range(4)]
    rows += [_meta_row(case_id=f"Criminal Appeal {i}", title="X v State") for i in range(4)]
    rows += [_meta_row(case_id="Civil Appeal CB", judge=FIVE_JUDGES, author_judge=None)]

    full, _ = select_corpus(rows)
    for cap in (1, 3, 5):
        capped, stats = select_corpus(rows, limit=cap)
        assert stats["selected"] == cap
        assert [r["case_id"] for r in capped] == [r["case_id"] for r in full[:cap]]


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

def test_a_snapshot_with_no_title_column_is_not_a_missing_access_grant(store, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")

    source = HF_SOURCES["injudgements"]
    # Never acquired: that IS the access grant, and saying so is right.
    assert landmark_set(store) == (None, LANDMARKS_NOT_ACQUIRED)

    store.upsert_source(source.source_id, "Apache-2.0")
    untitled = tmp_path / "untitled.parquet"
    pq.write_table(pa.table({"Text": ["a judgment"], "Labels": ["x"]}), untitled)
    store.record_artifact(
        source.source_id,
        "data/train-0.parquet",
        local_path=untitled,
        size_bytes=untitled.stat().st_size,
        sha256="x",
    )
    # On disk, but carrying no column this module recognises as a title. The
    # answer is a column name in a file the operator already has, and
    # reporting it as the missing grant sends them to the one place it is not.
    assert landmark_set(store) == (None, LANDMARKS_NO_TITLE_COLUMN)

    titled = tmp_path / "titled.parquet"
    pq.write_table(pa.table({"Titles": ["Maneka Gandhi vs Union Of India"]}), titled)
    store.record_artifact(
        source.source_id,
        "data/train-1.parquet",
        local_path=titled,
        size_bytes=titled.stat().st_size,
        sha256="y",
    )
    keys, why = landmark_set(store)
    assert why == LANDMARKS_OK
    assert keys == landmark_keys([{"Titles": "Maneka Gandhi v. Union of India"}])


def test_the_cli_does_not_send_an_operator_who_has_the_grant_to_go_and_get_it(tmp_path, capsys):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    from tuned.data.config import load_build_config
    from tuned.data.paths import build_paths

    config = temp_config(tmp_path)
    paths = build_paths(load_build_config(config, allow_unpinned=True).build.workdir).ensure()
    untitled = tmp_path / "untitled.parquet"
    pq.write_table(pa.table({"Text": ["a judgment"]}), untitled)
    with Store.open(paths.state_db) as opened:
        opened.upsert_source(HF_SOURCES["injudgements"].source_id, "Apache-2.0")
        opened.record_artifact(
            HF_SOURCES["injudgements"].source_id,
            "data/train-0.parquet",
            local_path=untitled,
            size_bytes=untitled.stat().st_size,
            sha256="x",
        )

    assert main(["--config", config], rows=[_meta_row()]) == 0
    out = capsys.readouterr().out
    assert "NO TITLE COLUMN" in out
    # The grant page is the wrong answer here and must not be printed.
    assert HF_SOURCES["injudgements"].url not in out


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
    # The coverage block names the column that answered, and says so loudly
    # where nothing did - a bare count is the thing that misled.
    assert "<- citation" in out
    assert "NO CANDIDATE MATCHED" in out  # nothing here carries a pdf link
    # The pair with the next three tests is what makes each banner mean
    # something: the missing-list one is absent exactly when the list was
    # there, and the join warning is absent when the join worked.
    assert "NO LANDMARK LIST" not in out
    assert "NONE matched" not in out
    assert "--no-landmarks" not in out


def test_the_default_command_writes_a_file_extraction_can_stop_half_way_through(tmp_path):
    from tuned.data.config import load_build_config
    from tuned.data.jsonl import read_jsonl
    from tuned.data.paths import build_paths

    config = temp_config(tmp_path)
    rows = [_meta_row(case_id=f"Civil Appeal {i}", judge=None, author_judge=None) for i in range(4)]
    rows.append(_meta_row(case_id="Civil Appeal CB", judge=FIVE_JUDGES, author_judge=None))

    # No --limit: the documented default, and the run whose ordering used to
    # be whatever order the parquet happened to list.
    assert main(["--config", config], rows=rows, landmarks=frozenset()) == 0

    paths = build_paths(load_build_config(config, allow_unpinned=True).build.workdir)
    written = list(read_jsonl(paths.corpus_dir / "selection.jsonl"))
    assert len(written) == 5
    assert written[0]["case_id"] == "Civil Appeal CB"
    assert [row["priority"] for row in written] == [6, 4, 4, 4, 4]


def test_cli_says_when_it_ran_without_the_landmark_list_without_overstating_it(tmp_path, capsys):
    code = main(["--config", temp_config(tmp_path)], rows=[_meta_row()], landmarks=None)
    out = capsys.readouterr().out
    assert code == 0
    assert "NO LANDMARK LIST" in out
    assert "opennyaiorg/InJudgements_dataset" in out
    # InJudgements is ~1,600 SC judgments spread over 1950-2017, so its
    # absence costs a few hundred priority bumps in an 8-of-16-year window.
    # It is not a blocker on P7, and the banner must not say the selection is
    # materially weaker than it is.
    assert "DEGRADED" not in out


def test_cli_says_when_the_third_signal_was_switched_off_on_purpose(tmp_path, capsys):
    # A THIRD way the list can be absent, and the only one that is nobody's
    # problem. Falling through to silence would make a deliberately weakened
    # run look like a complete one.
    code = main(["--config", temp_config(tmp_path), "--no-landmarks"], rows=[_meta_row()])
    out = capsys.readouterr().out
    assert code == 0
    assert "--no-landmarks" in out
    assert HF_SOURCES["injudgements"].url not in out


def test_cli_says_when_the_title_join_matched_nothing(tmp_path, capsys):
    # A DIFFERENT failure from a missing list, and the one the normalised
    # title join can produce silently and completely.
    code = main(
        ["--config", temp_config(tmp_path)],
        rows=[_meta_row()],
        landmarks=frozenset({"a title no metadata row carries"}),
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "NONE matched" in out
    assert "NO LANDMARK LIST" not in out


def test_cli_fails_when_a_non_empty_read_selected_nothing(tmp_path, capsys):
    # The parquet column names are the one thing that cannot be checked
    # offline, and a wrong one produces exactly this: rows read, none kept.
    # Exiting 0 would hand extraction an empty corpus and look successful.
    rows = [{"mystery_column": "x", "year": 2015} for _ in range(3)]
    code = main(["--config", temp_config(tmp_path)], rows=rows, landmarks=frozenset())
    assert code == 1
    assert "NOTHING SELECTED" in capsys.readouterr().out


def test_a_limit_of_zero_is_a_typo_not_a_diagnosis_of_the_schema(tmp_path, capsys):
    # `--limit 0` selects nothing from a perfectly good read, which the
    # empty-selection backstop would report as "NOTHING SELECTED ... check
    # the field coverage against the parquet's real column names" - sending
    # an operator who mistyped a cap to audit a schema that is fine.
    with pytest.raises(SystemExit):
        main(
            ["--config", temp_config(tmp_path), "--limit", "0"],
            rows=[_meta_row()],
            landmarks=frozenset(),
        )
    assert "NOTHING SELECTED" not in capsys.readouterr().out


def test_cli_hard_exits_after_success():
    assert "os._exit(" in SELECT_SRC.read_text(encoding="utf-8")
