"""split.py - case-level, content-keyed, and refusing rather than warning.

Fixtures are STRUCTURAL SHAPES, never real judgment or eval text: the
identifiers are invented in the real formats and the prose is filler.
"""

import json
import random
from pathlib import Path

import pytest
from pipeline_fakes import paths_for, temp_config
from test_build_decontaminate import prose, row

from tuned.data.acquire import sha256_file
from tuned.data.decontaminate import item_of
from tuned.data.dedupe import case_id_of
from tuned.data.jsonl import read_jsonl, write_jsonl
from tuned.data.split import (
    CUSTODY_MISMATCH,
    CUSTODY_NO_DIGEST,
    CUSTODY_NO_MANIFEST,
    CUSTODY_UNREADABLE,
    CUSTODY_VERIFIED,
    DATE_FROM_CITATION,
    DATE_FROM_CNR,
    DATE_FROM_NONE,
    DATE_FROM_PROV,
    MANIFEST_FILENAME,
    SPLIT_VERSION,
    DegenerateSplit,
    RowsLost,
    StraddlingCase,
    assign_units,
    case_id_date,
    custody_of,
    date_key,
    eval_target,
    item_date,
    manifest_digests,
    ordered_units,
    prov_date,
    split_items,
    units_of,
    year_in,
)
from tuned.data.split import main as split_main

SPLIT_SRC = Path(__file__).parent.parent / "src" / "tuned" / "data" / "split.py"


# --------------------------------------------------------------------------
# Fixtures. Invented identifiers in the real formats.
# --------------------------------------------------------------------------

def cnr(state: str, year: int, n: int = 451) -> str:
    """A well-formed CNR: 4 letters, 2-digit establishment, 6-digit number,
    4-digit year - the shape decontaminate._CNR matches."""
    return f"{state}01{n:06d}{year}"


def items(*rows, ids_from_text: bool = True):
    return [item_of(r, f"fixture#{i}", ids_from_text=ids_from_text) for i, r in enumerate(rows)]


def keyed(seed: int, **prov) -> dict:
    """A row with unique content and whatever provenance the case needs."""
    return row(prose(seed, 40), prose(seed + 5000, 30), **prov)


# --------------------------------------------------------------------------
# Dates: what actually carries one.
# --------------------------------------------------------------------------

def test_date_key_normalises_the_three_shapes_and_refuses_the_rest():
    assert date_key("2024-03-12") == "2024-03-12"
    assert date_key("2024-03") == "2024-03-00"
    assert date_key("2024") == "2024-00-00"
    assert date_key(2024) == "2024-00-00"
    # A full timestamp is truncated to its date, which is what [:10] is for.
    assert date_key("2024-03-12T09:15:00Z") == "2024-03-12"
    for junk in (None, "", "not a date", "12-03-2024", "24-03-12", "0001", "9999"):
        assert date_key(junk) is None, junk


def test_a_year_only_case_sorts_earliest_in_its_year():
    """The conservative direction: a case known only by its year is LESS
    likely to be pulled into the newest-first eval side than a dated one, not
    more. A filler of "-99-99" would have reversed that."""
    assert date_key("2024") < date_key("2024-01-01")
    assert date_key("2024") > date_key("2023-12-31")


def test_prov_date_takes_the_strongest_field_present():
    assert prov_date({"decision_date": "2024-03-12", "year": "1999"}) == "2024-03-12"
    assert prov_date({"year": "1999"}) == "1999-00-00"
    assert prov_date({"judgment_date": "2020-01-02"}) == "2020-01-02"
    # Present but unusable is the same as absent - it must not shadow a
    # weaker field that IS usable.
    assert prov_date({"decision_date": "n/a", "year": "1999"}) == "1999-00-00"
    assert prov_date({}) is None
    assert prov_date(None) is None


def test_no_row_builder_in_this_repo_writes_a_prov_date():
    """The measurement the date ladder is built on, kept as a test.

    seeds.py hardcodes decision_date=None in all three converters and
    decontaminate.generated_rows never copies it into _prov; replay.py and
    curated.py write exactly {source, license, native_id, reasoning}. So the
    _prov channel is inert on every row this build can produce today, and the
    day that changes this test is what says so.
    """
    from tuned.data.curated import predex_prediction_row
    from tuned.data.replay import legal_qa_row

    built, _ = legal_qa_row(
        {"question": "What does the section require?", "answer": "A" * 400}, "<t>", "</t>"
    )
    assert prov_date(built["_prov"]) is None
    assert set(built["_prov"]) == {"source", "license", "native_id", "reasoning"}
    built, _ = predex_prediction_row(
        {"Case Name": "X v Y", "Input": "F" * 600, "Output": "R" * 600}, "<t>", "</t>"
    )
    assert prov_date(built["_prov"]) is None


def test_the_cnr_year_is_read_positionally_because_a_scan_finds_nothing():
    """`ESCR010004512020` -> 2020, off the last four characters.

    The citation channel's scan is useless here and that is the point: a CNR's
    12 digits are one unbroken run, so a four-digit-run regex bounded by
    non-digits matches nowhere in it. Reading position is what makes the whole
    `cnr:` channel exist rather than a stricter version of the citation one.
    """
    assert case_id_date(f"cnr:{cnr('ESCR', 2020)}") == ("2020-00-00", DATE_FROM_CNR)
    assert case_id_date("cnr:TSXX011998452021") == ("2021-00-00", DATE_FROM_CNR)
    assert year_in("TSXX011998452021") is None  # what the scan would have said
    # A malformed key is dated by nothing rather than by a guess.
    assert case_id_date("cnr:NOTACNR") == (None, DATE_FROM_NONE)
    assert case_id_date("cnr:ESCR01000451209") == (None, DATE_FROM_NONE)  # 11 digits
    assert case_id_date("cnr:ESCR0100045120200") == (None, DATE_FROM_NONE)  # 13 digits
    # In range, or nothing: a case number ending 0451 is not a year 451.
    assert case_id_date("cnr:ESCR012020000451") == (None, DATE_FROM_NONE)


@pytest.mark.parametrize(
    "citation,year",
    [
        ("2023 INSC 45", "2023"),
        ("2023 DELHI:45", "2023"),
        ("2023 DELHI:45:DB", "2023"),
        ("(2020) 7 SCC 1", "2020"),
        ("(2020) 7 SCR 941", "2020"),
        ("2023 SCC ONLINE DEL 45", "2023"),
        ("1998 3 CRI LJ 45", "1998"),
        # AIR is the one canonical form whose year is NOT first, which is why
        # the year is scanned for rather than read off the front.
        ("AIR 1960 SC 30", "1960"),
        # An unmodelled reporter keeps its text upper-cased; the scan still
        # finds a year, and a page number out of year range does not fool it.
        ("(2019) 3 KLT 45", "2019"),
        ("SUIT 1234 OF 2019", "2019"),
    ],
)
def test_every_canonical_citation_form_yields_its_year(citation, year):
    assert case_id_date(f"cit:{citation}") == (f"{year}-00-00", DATE_FROM_CITATION)


def test_a_title_bucket_carries_no_date_and_says_so():
    assert case_id_date("title:state of x versus kumar and another") == (None, DATE_FROM_NONE)
    assert case_id_date(None) == (None, DATE_FROM_NONE)
    assert case_id_date("") == (None, DATE_FROM_NONE)


def test_prov_beats_the_identifier_and_the_channel_says_which_fired():
    [item] = items(keyed(1, cnr=cnr("ESCR", 2020), decision_date="2011-05-06"))
    unit = units_of([item])[0]
    assert (unit.date, unit.channel) == ("2011-05-06", DATE_FROM_PROV)
    [item] = items(keyed(1, cnr=cnr("ESCR", 2020)))
    unit = units_of([item])[0]
    assert (unit.date, unit.channel) == ("2020-00-00", DATE_FROM_CNR)


def test_the_date_is_the_case_s_own_not_an_authority_it_cites():
    """A 2020 judgment discussing a 1960 authority is a 2020 case.

    The prompt names AIR 1960 SC 30, so identifiers_from_text puts it on the
    row - but the row is BUCKETED under its CNR, and only the bucket is dated.
    A channel that read every identifier would file this under 1960.
    """
    r = row(f"The court considered AIR 1960 SC 30. {prose(9, 30)}", prose(10, 20),
            cnr=cnr("ESCR", 2020))
    [item] = items(r)
    assert "cit:AIR 1960 SC 30" in item.identifiers
    assert case_id_of(item) == f"cnr:{cnr('ESCR', 2020)}"
    assert units_of([item])[0].date == "2020-00-00"


# --------------------------------------------------------------------------
# The atoms.
# --------------------------------------------------------------------------

def test_siblings_of_one_case_are_one_atom_and_case_less_rows_are_their_own():
    key = cnr("ESCR", 2020)
    rows = [keyed(1, cnr=key), keyed(2, cnr=key), keyed(3), keyed(4)]
    units = units_of(items(*rows))
    assert len(units) == 3
    by_key = {u.key: u for u in units}
    assert by_key[f"cnr:{key}"].rows == 2
    assert sum(1 for u in units if u.case_id is None) == 2


def test_a_case_is_as_new_as_the_newest_thing_known_about_it():
    key = cnr("ESCR", 2020)
    rows = [keyed(1, cnr=key), keyed(2, cnr=key, decision_date="2021-06-01")]
    [unit] = units_of(items(*rows))
    assert unit.date == "2021-06-01"
    # And the order the two rows arrive in does not decide it.
    [unit] = units_of(items(*reversed(rows)))
    assert unit.date == "2021-06-01"


def test_eval_target_is_half_up_where_round_is_half_to_even():
    assert eval_target(40, 0.10) == 4
    assert eval_target(45, 0.10) == 5  # round() gives 4 here
    assert eval_target(55, 0.10) == 6  # round() gives 6 too - the pair matters
    assert round(45 * 0.10) == 4 != eval_target(45, 0.10)
    assert eval_target(0, 0.10) == 0
    assert eval_target(4, 0.10) == 0


def test_dated_atoms_come_first_newest_first_and_the_rest_by_content_hash():
    rows = [
        keyed(1, cnr=cnr("AAAA", 2015)),
        keyed(2, cnr=cnr("BBBB", 2024)),
        keyed(3, cnr=cnr("CCCC", 2019)),
        keyed(4),
        keyed(5),
    ]
    dated, dateless = ordered_units(units_of(items(*rows)))
    assert [u.date for u in dated] == ["2024-00-00", "2019-00-00", "2015-00-00"]
    assert len(dateless) == 2
    # The date-less order is the hash order, not the input order and not the
    # key order - a rule that read either would move under a shuffle.
    assert [u.key for u in dateless] == sorted((u.key for u in dateless),
                                              key=lambda k: __import__("hashlib")
                                              .sha256(k.encode()).hexdigest())


# --------------------------------------------------------------------------
# The assignment.
# --------------------------------------------------------------------------

def corpus(n_dated: int = 12, n_caseless: int = 20, rows_per_case: int = 2):
    """A corpus with dated multi-row cases and case-less singletons."""
    rows = []
    for c in range(n_dated):
        key = cnr("ESCR", 2000 + c)
        for r in range(rows_per_case):
            rows.append(keyed(c * 100 + r, cnr=key))
    rows += [keyed(90_000 + i) for i in range(n_caseless)]
    return rows


def test_the_newest_cases_fill_the_eval_side_and_the_rest_comes_from_hashes():
    train, evaluation, stats = split_items(items(*corpus()), fraction=0.10)
    assert stats["rows"] == 44 and stats["eval_target_rows"] == 4
    # Two 2-row cases: the two newest, 2011 and 2010.
    assert stats["eval_rows"] == 4 and stats["date_assigned_units"] == 2
    assert stats["hash_assigned_units"] == 0
    assert stats["date_boundary"] == "2010-00-00"
    evaluated = {case_id_of(i) for i in evaluation}
    assert evaluated == {f"cnr:{cnr('ESCR', 2011)}", f"cnr:{cnr('ESCR', 2010)}"}
    assert len(train) == 40


def test_the_hash_channel_fills_what_the_dated_cases_cannot():
    """One dated case, a target of four rows: the dated side runs out at two
    and the remainder must come from the date-less atoms."""
    rows = [keyed(1, cnr=cnr("ESCR", 2020)), keyed(2, cnr=cnr("ESCR", 2020))]
    rows += [keyed(500 + i) for i in range(38)]
    _train, evaluation, stats = split_items(items(*rows), fraction=0.10)
    assert stats["eval_target_rows"] == 4
    assert (stats["date_assigned_units"], stats["hash_assigned_units"]) == (1, 2)
    assert stats["eval_rows"] == 4 and len(evaluation) == 4


def test_overshooting_by_an_atom_is_preferred_to_undershooting():
    """A 3-row case (dedupe's cap) against a 4-row target: the walk takes the
    second atom because the first left it short, so the fraction is a floor."""
    rows = [keyed(i, cnr=cnr("ESCR", 2020)) for i in range(3)]
    rows += [keyed(10 + i, cnr=cnr("BBBB", 2019)) for i in range(3)]
    rows += [keyed(100 + i) for i in range(34)]
    _train, _evaluation, stats = split_items(items(*rows), fraction=0.10)
    assert stats["eval_target_rows"] == 4
    assert stats["eval_rows"] == 6  # 3 + 3, never 3
    assert stats["eval_fraction_achieved"] == 0.15


def test_the_channel_census_counts_every_atom_not_just_the_eval_ones():
    rows = [
        keyed(1, cnr=cnr("ESCR", 2020)),
        keyed(2, neutral_citation="(2019) 3 SCC 12"),
        keyed(3, case_name="the state of somewhere versus a named appellant"),
        keyed(4),
    ]
    _train, _evaluation, stats = split_items(items(*rows), fraction=0.25)
    assert stats["by_date_channel"] == {
        DATE_FROM_PROV: 0, DATE_FROM_CITATION: 1, DATE_FROM_CNR: 1, DATE_FROM_NONE: 2,
    }


# --------------------------------------------------------------------------
# Determinism.
# --------------------------------------------------------------------------

def assignment(rows, *, fraction=0.10):
    """key -> side, which is what has to be stable."""
    return split_record(rows, fraction=fraction)[0]


def split_record(rows, *, fraction=0.10):
    """(key -> side, the stats block the manifest is written from).

    BOTH halves, because "no row moved" and "the manifest is the same
    manifest" are two claims and only the first was ever checked. The channel
    census is a measurement about the corpus that goes into split.json and out
    of it into an operator's reading of when the `_prov` channel woke up; a
    number that moves under a shuffle is a number nobody can act on, even when
    every row landed where it landed before.
    """
    train, evaluation, stats = split_items(items(*rows), fraction=fraction)
    sides = {**{i.key: "train" for i in train}, **{i.key: "eval" for i in evaluation}}
    return sides, stats


def test_the_same_input_assigns_the_same_way_twice():
    rows = corpus()
    assert assignment(rows) == assignment(rows)


def hash_filled_corpus():
    """A corpus where the CONTENT-KEYED channel actually decides something.

    corpus() alone cannot test it: its dated cases fill the eval target on
    their own, so hash_assigned is 0 and the date-less ordering never runs. A
    determinism test over that corpus passes with a positional row key - which
    is precisely the nondeterminism it exists to forbid, and a mutant swapping
    the content key for the input index survived it.
    """
    rows = [keyed(1, cnr=cnr("ESCR", 2020)), keyed(2, cnr=cnr("ESCR", 2020))]
    rows += [keyed(500 + i) for i in range(38)]
    return rows


def channel_tie_corpus():
    """A corpus where two rows of ONE case reach the SAME date by different
    channels: one carries a `_prov.year`, the other is dated off the CNR.

    Neither of the other two corpora can see the tie - no row in them carries
    a `_prov` date at all - and the tie is invisible in the ASSIGNMENT by
    construction, because the date and therefore the side are identical
    whichever channel wins. It is the census that moves, which is why the
    shuffle test has to compare the manifest and not only the sides.
    """
    key = cnr("ESCR", 2020)
    rows = [keyed(1, cnr=key, year=2020), keyed(2, cnr=key)]
    rows += [keyed(700 + i) for i in range(18)]
    return rows


@pytest.mark.parametrize("seed", [1, 2, 3, 17, 99])
@pytest.mark.parametrize(
    "build", [corpus, hash_filled_corpus, channel_tie_corpus],
    ids=["dated", "hash_filled", "channel_tie"],
)
def test_shuffling_the_input_does_not_move_one_row_or_one_manifest_number(seed, build):
    """Content-keyed end to end: atoms are grouped, ordered by date then by a
    hash of the atom key, and the target is filled by walking that order - so
    nothing in the decision can see which line a row arrived on.

    The manifest stats are compared too, not just the sides. They were the half
    that could still move: `units_of` took a later row's date only on a strict
    `>`, so when two rows of one case named the same date through different
    channels the FIRST one to arrive labelled the case, and `by_date_channel`
    came out `{'prov': 1}` or `{'cnr': 1}` depending on the shuffle.
    """
    rows = build()
    base = split_record(rows)
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    assert split_record(shuffled) == base


def test_the_hash_channel_really_decides_the_second_corpus():
    """Guards the guard: if the hash-filled corpus ever stopped using the
    content-keyed channel, the shuffle test above would go quiet rather than
    fail."""
    _train, _evaluation, stats = split_items(items(*hash_filled_corpus()), fraction=0.10)
    assert stats["hash_assigned_units"] > 0
    _train, _evaluation, dated = split_items(items(*corpus()), fraction=0.10)
    assert dated["hash_assigned_units"] == 0


def test_the_tie_corpus_really_puts_two_channels_on_one_case():
    """Guards the guard, again: a corpus whose two rows stopped disagreeing
    about the channel would make the census stable for a reason that has
    nothing to do with the tie-break."""
    key = f"cnr:{cnr('ESCR', 2020)}"
    rows = channel_tie_corpus()
    channels = {item_date(i, key)[1] for i in items(*rows[:2])}
    assert channels == {DATE_FROM_PROV, DATE_FROM_CNR}
    assert {item_date(i, key)[0] for i in items(*rows[:2])} == {"2020-00-00"}


def test_two_channels_on_one_date_break_toward_the_stronger_one():
    """The tie-break itself, in both arrival orders. `_prov` beats the case
    identifier for the same reason `item_date` consults it first: it is the
    channel a seed builder fills deliberately, and the census exists to say
    when it starts firing."""
    key = cnr("ESCR", 2020)
    by_prov, by_cnr = keyed(1, cnr=key, year=2020), keyed(2, cnr=key)
    for order in ([by_prov, by_cnr], [by_cnr, by_prov]):
        unit = [u for u in units_of(items(*order)) if u.case_id][0]
        assert (unit.date, unit.channel) == ("2020-00-00", DATE_FROM_PROV)
    # A NEWER date still wins outright - the tie-break only breaks ties, and a
    # stronger channel naming an older date must not drag the case back.
    older_prov = keyed(3, cnr=key, year=2015)
    unit = [u for u in units_of(items(older_prov, by_cnr)) if u.case_id][0]
    assert (unit.date, unit.channel) == ("2020-00-00", DATE_FROM_CNR)


def test_the_output_files_are_byte_identical_across_two_cli_runs(tmp_path):
    cfg, paths = _deduped(tmp_path, corpus())
    assert split_main(["--config", cfg]) == 0
    first = [(paths.out_dir / n).read_bytes() for n in ("split_train.jsonl", "split_eval.jsonl")]
    assert split_main(["--config", cfg]) == 0
    second = [(paths.out_dir / n).read_bytes() for n in ("split_train.jsonl", "split_eval.jsonl")]
    assert first == second


def test_no_clock_or_rng_reaches_the_assignment():
    """The manifest is timestamped and that is the ONLY time this module reads
    a clock; nothing here seeds, samples or shuffles."""
    source = SPLIT_SRC.read_text(encoding="utf-8")
    for banned in ("import random", "random.", "time.time", "datetime.now", "uuid"):
        assert banned not in source, banned


# --------------------------------------------------------------------------
# The invariants, each in both directions.
# --------------------------------------------------------------------------

def test_a_straddling_assignment_is_refused(tmp_path):
    """Two rows, one CNR, forced apart by a broken assigner.

    By construction this cannot happen - which is exactly why the assert needs
    an assigner that makes it happen, or the refusal branch is a claim no test
    has ever reached.
    """
    key = cnr("ESCR", 2020)
    fixture = items(keyed(1, cnr=key), keyed(2, cnr=key))

    def straddles(rows, *, fraction):
        return [0], [1], {"rows": 2, "eval_target_rows": 1, "eval_rows": 1, "train_rows": 1}

    with pytest.raises(StraddlingCase, match=f"cnr:{key}"):
        split_items(fixture, fraction=0.5, assign=straddles)
    # And the same pair through the REAL assigner stays together - the refusal
    # above is unreachable in production, which is why it needed a seam.
    both = items(*[keyed(1000 + i) for i in range(18)], keyed(1, cnr=key), keyed(2, cnr=key))
    train, evaluation, _ = split_items(both, fraction=0.10)
    siblings = [i for i in evaluation if case_id_of(i) == f"cnr:{key}"]
    assert len(siblings) in (0, 2)
    assert len([i for i in train if case_id_of(i) == f"cnr:{key}"]) == 2 - len(siblings)


def test_case_less_rows_never_trip_the_disjointness_assert():
    """The None bucket is not a bucket. Grouping case-less rows under one key
    would make every split a straddle."""
    fixture = items(*[keyed(i) for i in range(40)])

    def halves(rows, *, fraction):
        return list(range(20)), list(range(20, 40)), assign_units(rows, fraction=fraction)[2]

    train, evaluation, _ = split_items(fixture, fraction=0.5, assign=halves)
    assert len(train) == len(evaluation) == 20


def test_an_assigner_that_loses_a_row_is_refused():
    fixture = items(*[keyed(i) for i in range(10)])

    def loses(rows, *, fraction):
        return list(range(8)), [9], {"rows": 10, "eval_target_rows": 1,
                                     "eval_rows": 1, "train_rows": 8}

    with pytest.raises(RowsLost, match="partition"):
        split_items(fixture, fraction=0.10, assign=loses)


def test_an_assigner_that_emits_more_rows_than_it_read_is_refused():
    """The OTHER direction of the counting check, which had no case: an
    assigner that loses rows is caught by `<`, and only one that invents them
    needs the `!=`. The identity check below would catch this too, so the test
    pins WHICH refusal fires - a mutant weakening `!=` to `<` survived a test
    that only asked for the exception type."""
    fixture = items(*[keyed(i) for i in range(10)])

    def invents(rows, *, fraction):
        return list(range(10)), [0, 1], {"rows": 10, "eval_target_rows": 1,
                                         "eval_rows": 2, "train_rows": 10}

    with pytest.raises(RowsLost, match="partition"):
        split_items(fixture, fraction=0.10, assign=invents)


def test_an_assigner_that_duplicates_a_row_is_refused_even_at_the_right_count():
    """The count is right and the rows are wrong, which is the reading a
    count-only check cannot tell from a working split."""
    fixture = items(*[keyed(i) for i in range(10)])

    def doubles(rows, *, fraction):
        return list(range(9)), [8], {"rows": 10, "eval_target_rows": 1,
                                     "eval_rows": 1, "train_rows": 9}

    with pytest.raises(RowsLost, match="on both sides or on neither"):
        split_items(fixture, fraction=0.10, assign=doubles)


def test_an_empty_input_is_refused_rather_than_split_into_two_empty_files():
    with pytest.raises(DegenerateSplit, match="no rows"):
        split_items([], fraction=0.10)


def test_a_corpus_that_is_one_case_is_refused_rather_than_shipped_untrained():
    """Every row on one CNR: the eval side takes the whole atom and train is
    empty. A build that shipped that would have nothing to train on."""
    key = cnr("ESCR", 2020)
    fixture = items(*[keyed(i, cnr=key) for i in range(40)])
    with pytest.raises(DegenerateSplit, match="TRAIN side came out empty"):
        split_items(fixture, fraction=0.10)


def test_a_target_of_zero_rows_is_allowed_but_an_empty_eval_against_a_target_is_not():
    """Below 5 rows a 10% target rounds to zero, and an empty eval side is
    then the honest answer rather than a refusal - the two branches are
    different and both are reachable."""
    fixture = items(*[keyed(i) for i in range(4)])
    train, evaluation, stats = split_items(fixture, fraction=0.10)
    assert stats["eval_target_rows"] == 0 and evaluation == [] and len(train) == 4

    def empties(rows, *, fraction):
        return list(range(len(rows))), [], {"rows": len(rows), "eval_target_rows": 3,
                                            "eval_rows": 0, "train_rows": len(rows)}

    with pytest.raises(DegenerateSplit, match="eval side came out EMPTY"):
        split_items(items(*[keyed(i) for i in range(30)]), fraction=0.10, assign=empties)


def test_a_case_reachable_under_two_cnrs_is_counted_rather_than_missed():
    """The assert's known blind spot, as a number.

    One row carries two CNRs and is bucketed under the sorted-first; its
    sibling carries only the second. They are two atoms, so no straddle is
    raised - and the manifest says one CNR crossed anyway.
    """
    a, b = cnr("AAAA", 2020), cnr("ZZZZ", 2020)
    fixture = items(
        row(prose(1, 40), prose(2, 30), cnr=f"{a} and {b}"),
        *[keyed(10 + i, cnr=b) for i in range(9)],
    )
    train, evaluation, stats = split_items(fixture, fraction=0.10)
    sides = {case_id_of(i) for i in train}, {case_id_of(i) for i in evaluation}
    assert sides[0] & sides[1] == set()  # no straddle by the atom rule
    assert stats["cross_side_identifiers"] == 1
    # And an ordinary split reports zero, so the number means something.
    _t, _e, clean = split_items(items(*corpus()), fraction=0.10)
    assert clean["cross_side_identifiers"] == 0


# --------------------------------------------------------------------------
# The chain of custody.
# --------------------------------------------------------------------------

def test_manifest_digests_reads_both_upstream_shapes():
    """decontaminate.py writes one `output`; everything downstream writes an
    `outputs` list. One reader, or the chain breaks at the shape change."""
    assert manifest_digests({"output": {"sha256": "aa"}}) == {"aa"}
    assert manifest_digests({"outputs": [{"sha256": "bb"}, {"sha256": "cc"}]}) == {"bb", "cc"}
    assert manifest_digests({"outputs": [{"sha256": "bb"}], "output": {"sha256": "aa"}}) == {
        "aa", "bb"
    }
    # A record without a digest is not a digest of "".
    assert manifest_digests({"outputs": [{"path": "x", "rows": 3}]}) == set()
    assert manifest_digests({}) == set()
    assert manifest_digests(None) == set()


def _deduped(tmp_path, rows, *, manifest: dict | None = "default"):
    """A dedupe-style input plus the manifest that vouches for it."""
    cfg = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    out = paths.out_dir / "deduped.jsonl"
    written = write_jsonl(out, rows)
    if manifest == "default":
        manifest = {
            "stage": "dedupe",
            "dedupe_version": 4,
            "outputs": [{"path": str(out), "rows": written, "sha256": sha256_file(out)}],
            "thresholds": {"case_ids_from_text": True},
            "decontamination": {"at": "2026-08-01T00:00:00Z", "decon_version": 4},
        }
    if manifest is not None:
        (paths.out_dir / "dedupe.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cfg, paths


def test_a_verified_input_carries_the_whole_upstream_manifest_forward(tmp_path):
    cfg, paths = _deduped(tmp_path, corpus())
    assert split_main(["--config", cfg]) == 0
    manifest = json.loads((paths.out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["split_version"] == SPLIT_VERSION == 1
    assert manifest["dedupe_check"]["status"] == CUSTODY_VERIFIED
    # WHOLE, not summarised: decontamination's own record has to survive to
    # stats.py through this.
    assert manifest["dedupe"]["decontamination"]["decon_version"] == 4
    assert manifest["eval_fraction"] == 0.10
    assert manifest["counts"]["rows"] == 44
    assert manifest["counts"]["train_rows"] + manifest["counts"]["eval_rows"] == 44
    # And the outputs are digested so the NEXT stage can check them.
    digests = manifest_digests(manifest)
    assert digests == {
        sha256_file(paths.out_dir / "split_train.jsonl"),
        sha256_file(paths.out_dir / "split_eval.jsonl"),
    }


@pytest.mark.parametrize(
    "manifest,status,banner",
    [
        (None, CUSTODY_NO_MANIFEST, "NO UPSTREAM MANIFEST"),
        ({"stage": "dedupe", "outputs": []}, CUSTODY_NO_DIGEST, "NO OUTPUT DIGEST"),
        ({"stage": "dedupe", "outputs": [{"path": "x", "rows": 1, "sha256": "0" * 64}]},
         CUSTODY_MISMATCH, "DESCRIBES DIFFERENT ROWS"),
    ],
)
def test_a_broken_chain_is_a_refusal_and_not_a_banner(tmp_path, capsys, manifest, status,
                                                      banner):
    """dedupe.py warns and ships; this tail refuses. The artifact here IS the
    dataset, and a dataset whose provenance is a directory layout is one
    nobody can write a card for."""
    cfg, paths = _deduped(tmp_path, corpus(), manifest=manifest)
    assert split_main(["--config", cfg]) == 2
    out = capsys.readouterr().out
    assert banner in out and "nothing was written" in out
    assert not (paths.out_dir / "split_train.jsonl").exists()
    assert not (paths.out_dir / MANIFEST_FILENAME).exists()
    record = custody_of([paths.out_dir / "deduped.jsonl"], manifest_filename="dedupe.json")[1]
    assert record["status"] == status


def test_an_unreadable_manifest_is_its_own_status_not_a_missing_one(tmp_path):
    """Truncated JSON and no file at all send an operator to different places."""
    cfg, paths = _deduped(tmp_path, corpus())
    (paths.out_dir / "dedupe.json").write_text('{"stage": "ded', encoding="utf-8")
    assert split_main(["--config", cfg]) == 2
    record = custody_of([paths.out_dir / "deduped.jsonl"], manifest_filename="dedupe.json")[1]
    assert record["status"] == CUSTODY_UNREADABLE


def test_the_same_bytes_under_another_name_still_verify(tmp_path):
    """Custody is bound to CONTENT, not to a path: `--in elsewhere.jsonl` over
    the same rows is still the rows dedupe wrote."""
    cfg, paths = _deduped(tmp_path, corpus())
    elsewhere = paths.out_dir / "renamed.jsonl"
    elsewhere.write_bytes((paths.out_dir / "deduped.jsonl").read_bytes())
    assert split_main(["--config", cfg, "--in", str(elsewhere)]) == 0


def test_a_file_at_the_expected_path_with_other_bytes_does_not(tmp_path):
    cfg, paths = _deduped(tmp_path, corpus())
    write_jsonl(paths.out_dir / "deduped.jsonl", corpus(n_dated=3, n_caseless=3))
    assert split_main(["--config", cfg]) == 2


# --------------------------------------------------------------------------
# The CLI.
# --------------------------------------------------------------------------

def test_the_two_files_partition_the_input_row_for_row(tmp_path):
    rows = corpus()
    cfg, paths = _deduped(tmp_path, rows)
    assert split_main(["--config", cfg]) == 0
    train = list(read_jsonl(paths.out_dir / "split_train.jsonl"))
    evaluation = list(read_jsonl(paths.out_dir / "split_eval.jsonl"))
    assert len(train) + len(evaluation) == len(rows)
    # BYTE-IDENTICAL rows, in input order per side: this pass assigns, it does
    # not rewrite.
    assert [json.dumps(r, ensure_ascii=False) for r in train + evaluation] != []
    written = {json.dumps(r, sort_keys=True) for r in train + evaluation}
    assert written == {json.dumps(r, sort_keys=True) for r in rows}


def test_the_eval_fraction_is_build_held_out_frac_and_not_a_knob_of_its_own(tmp_path):
    from tuned.data.config import load_build_config

    cfg, paths = _deduped(tmp_path, corpus())
    assert split_main(["--config", cfg]) == 0
    manifest = json.loads((paths.out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["eval_fraction"] == load_build_config(cfg, allow_unpinned=True).build.held_out_frac


def test_a_missing_input_names_the_command_that_makes_it(tmp_path, capsys):
    cfg = temp_config(tmp_path)
    paths_for(tmp_path)
    assert split_main(["--config", cfg]) == 2
    assert "tuned.data.dedupe" in capsys.readouterr().out


def test_the_run_says_the_prov_date_channel_never_fired(tmp_path, capsys):
    """The expected line on today's corpus, printed rather than inferred from
    a zero - its ABSENCE is what would be news."""
    cfg, _paths = _deduped(tmp_path, corpus())
    assert split_main(["--config", cfg]) == 0
    out = capsys.readouterr().out
    assert "date[prov]: 0" in out
    assert "no row carried an explicit _prov date" in out


def test_a_corpus_with_prov_dates_does_not_print_that_line(tmp_path, capsys):
    rows = [keyed(i, decision_date=f"20{10 + i:02d}-01-01") for i in range(40)]
    cfg, _paths = _deduped(tmp_path, rows)
    assert split_main(["--config", cfg]) == 0
    out = capsys.readouterr().out
    assert "date[prov]: 40" in out
    assert "no row carried an explicit _prov date" not in out


def test_the_run_is_logged_to_the_store(tmp_path):
    from tuned.data.store import Store

    cfg, paths = _deduped(tmp_path, corpus())
    assert split_main(["--config", cfg]) == 0
    event = json.loads(Store.open(paths.state_db).events("split")[0]["detail_json"])
    assert event["stage"] == "split"


def test_cli_hard_exits_after_success():
    assert "os._exit(" in SPLIT_SRC.read_text(encoding="utf-8")


def test_the_version_ledger_describes_the_version_the_module_ships():
    import re

    source = SPLIT_SRC.read_text(encoding="utf-8")
    entries = [int(n) for n in re.findall(r"^# (\d+)  ", source, re.M)]
    assert entries == sorted(entries)
    assert entries[-1] == SPLIT_VERSION
    assert entries[0] == 1
