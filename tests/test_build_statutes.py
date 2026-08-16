import copy
import itertools
import json
import pickle
import time
from datetime import date

import pytest

from tuned.data.config import load_build_config
from tuned.data.jsonl import write_jsonl
from tuned.data.statutes import (
    APPOINTED_DAY,
    CODE_KIND,
    FLAG_NEW_FOR_OLD,
    FLAG_OLD_FOR_NEW,
    IPC_BNS_MAP_PATH,
    NEW_CODES,
    OLD_CODES,
    RESOURCES_DIR,
    Mapping,
    SectionRef,
    SectionRegistry,
    cross_code_flags,
    cross_code_review,
    extract_sections,
    governing_family,
    normalize_number,
    resolve_code,
    statute_pattern,
)

BEFORE = date(2024, 6, 30)
ON = date(2024, 7, 1)
AFTER = date(2024, 8, 15)
ERR = "ValueError"


def test_appointed_day_constant():
    assert APPOINTED_DAY == date(2024, 7, 1)


def test_config_appointed_day_matches_the_module_constant():
    cfg = load_build_config("configs/data_law_v1.yaml", allow_unpinned=True)
    assert date.fromisoformat(cfg.build.appointed_day) == APPOINTED_DAY


def test_code_tables():
    assert OLD_CODES == {"IPC", "CRPC", "IEA"}
    assert NEW_CODES == {"BNS", "BNSS", "BSA"}
    assert CODE_KIND == {
        "IPC": "substantive",
        "BNS": "substantive",
        "CRPC": "procedural",
        "BNSS": "procedural",
        "IEA": "evidence",
        "BSA": "evidence",
    }


# --------------------------------------------------------------------------
# THE DECISION TABLE. Every cell of kind x offence x proceeding, written out
# by hand - this table IS the legal specification, so it is never derived
# from the code under test.
#
#   substantive : BNS s.358(2) + General Clauses Act s.6 - offence date only.
#   procedural  : BNSS s.531(2)(a) - pending on the appointed day or not.
#   evidence    : BSA s.170 - same pending-proceeding rule.
#   ON the appointed day counts as NEW on both axes.
# --------------------------------------------------------------------------

DECISION_TABLE = [
    # kind, offence_date, proceeding_started, expected
    # -- substantive: the proceeding date is irrelevant, always ------------
    ("substantive", BEFORE, BEFORE, "old"),
    ("substantive", BEFORE, ON, "old"),
    ("substantive", BEFORE, AFTER, "old"),  # IPC applies forever to a pre-transition offence
    ("substantive", BEFORE, None, "old"),
    ("substantive", ON, BEFORE, "new"),
    ("substantive", ON, ON, "new"),
    ("substantive", ON, AFTER, "new"),
    ("substantive", ON, None, "new"),
    ("substantive", AFTER, BEFORE, "new"),
    ("substantive", AFTER, ON, "new"),
    ("substantive", AFTER, AFTER, "new"),
    ("substantive", AFTER, None, "new"),
    ("substantive", None, BEFORE, ERR),
    ("substantive", None, ON, ERR),
    ("substantive", None, AFTER, ERR),
    ("substantive", None, None, ERR),
    # -- procedural: the proceeding date decides; offence date only as a
    #    documented best-effort fallback when no proceeding date is known --
    ("procedural", BEFORE, BEFORE, "old"),
    ("procedural", BEFORE, ON, "new"),
    ("procedural", BEFORE, AFTER, "new"),  # THE SPLIT CELL: 2023 offence, 2024-08 FIR -> BNSS
    ("procedural", BEFORE, None, "old"),
    ("procedural", ON, BEFORE, "old"),
    ("procedural", ON, ON, "new"),
    ("procedural", ON, AFTER, "new"),
    ("procedural", ON, None, "new"),
    ("procedural", AFTER, BEFORE, "old"),
    ("procedural", AFTER, ON, "new"),
    ("procedural", AFTER, AFTER, "new"),
    ("procedural", AFTER, None, "new"),
    ("procedural", None, BEFORE, "old"),
    ("procedural", None, ON, "new"),
    ("procedural", None, AFTER, "new"),
    ("procedural", None, None, ERR),
    # -- evidence: BSA s.170, identical pending-proceeding rule ------------
    ("evidence", BEFORE, BEFORE, "old"),
    ("evidence", BEFORE, ON, "new"),
    ("evidence", BEFORE, AFTER, "new"),
    ("evidence", BEFORE, None, "old"),
    ("evidence", ON, BEFORE, "old"),
    ("evidence", ON, ON, "new"),
    ("evidence", ON, AFTER, "new"),
    ("evidence", ON, None, "new"),
    ("evidence", AFTER, BEFORE, "old"),
    ("evidence", AFTER, ON, "new"),
    ("evidence", AFTER, AFTER, "new"),
    ("evidence", AFTER, None, "new"),
    ("evidence", None, BEFORE, "old"),
    ("evidence", None, ON, "new"),
    ("evidence", None, AFTER, "new"),
    ("evidence", None, None, ERR),
]


def test_decision_table_covers_every_cell():
    kinds = ("substantive", "procedural", "evidence")
    dates = (BEFORE, ON, AFTER, None)
    assert {(k, o, p) for k, o, p, _ in DECISION_TABLE} == set(itertools.product(kinds, dates, dates))
    assert len(DECISION_TABLE) == 48


@pytest.mark.parametrize("kind,offence,proceeding,expected", DECISION_TABLE)
def test_governing_family_decision_table(kind, offence, proceeding, expected):
    if expected == ERR:
        with pytest.raises(ValueError):
            governing_family(kind, offence_date=offence, proceeding_started=proceeding)
    else:
        assert governing_family(kind, offence_date=offence, proceeding_started=proceeding) == expected


def test_substantive_ignores_the_proceeding_date_entirely():
    for proceeding in (BEFORE, ON, AFTER, None):
        assert governing_family("substantive", offence_date=BEFORE, proceeding_started=proceeding) == "old"
        assert governing_family("substantive", offence_date=AFTER, proceeding_started=proceeding) == "new"


def test_the_split_cell():
    """2023 offence, FIR registered 2024-08: investigation under the BNSS
    (procedural, fresh proceeding) while the charge stays under the IPC
    (substantive, pre-transition offence)."""
    offence, proceeding = date(2023, 5, 1), date(2024, 8, 1)
    assert governing_family("substantive", offence_date=offence, proceeding_started=proceeding) == "old"
    assert governing_family("procedural", offence_date=offence, proceeding_started=proceeding) == "new"
    assert governing_family("evidence", offence_date=offence, proceeding_started=proceeding) == "new"


def test_governing_family_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown kind"):
        governing_family("penal", offence_date=BEFORE, proceeding_started=BEFORE)


def test_governing_family_substantive_error_names_the_offence_date():
    with pytest.raises(ValueError, match="offence_date"):
        governing_family("substantive", offence_date=None, proceeding_started=AFTER)


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("Section 302 IPC", [("IPC", "302")]),
        ("S. 302 of the IPC", [("IPC", "302")]),
        ("Sec 103(2) BNS", [("BNS", "103(2)")]),
        ("§302 IPC", [("IPC", "302")]),
        ("§§ 302 IPC", [("IPC", "302")]),
        ("Section 173 CrPC", [("CRPC", "173")]),
        ("section 173 Cr.P.C.", [("CRPC", "173")]),
        ("Section 173 Cr. P. C.", [("CRPC", "173")]),
        ("u/s 420 IPC", [("IPC", "420")]),
        ("u/s 420 I.P.C.", [("IPC", "420")]),
        ("Sections 302, IPC", [("IPC", "302")]),
        ("Section 302 (IPC)", [("IPC", "302")]),
        ("under section 302 of the Indian Penal Code, 1860", [("IPC", "302")]),
        ("Section 65B of the Evidence Act", [("IEA", "65B")]),
        ("section 65 of the Indian Evidence Act", [("IEA", "65")]),
        ("Sec. 63 BSA", [("BSA", "63")]),
        ("Section 173(8) BNSS", [("BNSS", "173(8)")]),
        ("section 3(5) of the Bharatiya Nyaya Sanhita", [("BNS", "3(5)")]),
        ("s. 304B IPC", [("IPC", "304B")]),
        ("Section 376(2)(N) IPC", [("IPC", "376(2)(n)")]),
        # lists: the code is stated once, at the end
        ("Sections 302, 307 and 34 IPC", [("IPC", "302"), ("IPC", "307"), ("IPC", "34")]),
        ("u/s 302/34 IPC", [("IPC", "302"), ("IPC", "34")]),
        ("S. 302 r/w 149 of the IPC", [("IPC", "302"), ("IPC", "149")]),
        # marker-less lead-ins
        ("punishable under 302 IPC", [("IPC", "302")]),
        ("convicted under 302 and 34 IPC", [("IPC", "302"), ("IPC", "34")]),
        ("charged with 420 IPC", [("IPC", "420")]),
        ("read with 34 of the IPC", [("IPC", "34")]),
    ],
)
def test_extract_sections_alias_matrix(text, expected):
    assert [(r.code, r.number) for r in extract_sections(text)] == expected


def test_extract_sections_case_and_dedup_and_order():
    text = "First Section 103 BNS, then s. 302 ipc, then SECTION 103 bns again."
    assert [str(r) for r in extract_sections(text)] == ["BNS 103", "IPC 302"]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "The 302 bus route runs past the court.",
        "the number 302 and the IPC are mentioned far apart in this sentence",
        "Section 302 and the IPC are different things",
        "Section 302 of the Motor Vehicles Act",
        "Section 302 IPCX",  # trailing-letter guard: IPCX is not the IPC
        "Section 302 SIPC",
        # a bare number with no citation lead-in is not a section
        "In 2023, 45 IPC cases were filed in the district",
        "Chapter 5 IPC deals with abetment",
        "the Indian Penal Code, 1860 (IPC)",  # 1860 must not read as section 860
        "under the Indian Penal Code, 1860",
    ],
)
def test_extract_sections_negatives(text):
    assert extract_sections(text) == []


def test_statute_pattern_groups():
    m = statute_pattern().search("charged u/s 302 of the IPC today")
    assert m is not None
    assert m.group("number") == "302"
    assert resolve_code(m.group("code")) == "IPC"


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("IPC", "IPC"),
        ("I.P.C.", "IPC"),
        ("ipc", "IPC"),
        ("CrPC", "CRPC"),
        ("Cr.P.C.", "CRPC"),
        ("CRPC", "CRPC"),
        ("Cr. P. C.", "CRPC"),
        ("Evidence Act", "IEA"),
        ("Indian Evidence Act", "IEA"),
        ("IEA", "IEA"),
        ("BNS", "BNS"),
        ("BNSS", "BNSS"),
        ("BSA", "BSA"),
        ("Indian Penal Code, 1860", "IPC"),
        ("Motor Vehicles Act", None),
    ],
)
def test_resolve_code_aliases(alias, expected):
    assert resolve_code(alias) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("302", "302"),
        ("304b", "304B"),
        ("103 (2)", "103(2)"),
        (" 3 ( 5 ) ", "3(5)"),
        ("376(2)(N)", "376(2)(n)"),
        (302, "302"),  # external section lists ship integers
        (None, ""),
    ],
)
def test_normalize_number(raw, expected):
    assert normalize_number(raw) == expected


def test_base_number_keeps_the_letter_suffix():
    assert SectionRef("BNS", "103(2)").base_number == "103"
    assert SectionRef("IPC", "304B").base_number == "304B"  # 304B is NOT 304
    assert SectionRef("IPC", "302").base_number == "302"


# --------------------------------------------------------------------------
# cross_code_flags - the temporal gate primitive
# --------------------------------------------------------------------------

OLD_OFFENCE = {"offence_date": date(2023, 5, 1), "proceeding_started": None}
NEW_OFFENCE = {"offence_date": date(2024, 9, 1), "proceeding_started": None}


def test_flag_new_code_cited_for_pre_transition_offence():
    assert cross_code_flags("The accused is liable under Section 103 BNS.", kind_dates=OLD_OFFENCE) == [
        FLAG_NEW_FOR_OLD
    ]


def test_savings_clause_discussion_is_not_flagged():
    text = (
        "Although Section 103 BNS now covers murder, section 358 BNS saves the "
        "repealed IPC for offences committed before 1 July 2024."
    )
    assert cross_code_flags(text, kind_dates=OLD_OFFENCE) == []


def test_savings_clause_detected_as_a_section_reference_too():
    text = "Section 103 BNS is irrelevant here: see s. 358 BNS."
    assert cross_code_flags(text, kind_dates=OLD_OFFENCE) == []


def test_flag_old_code_cited_for_post_transition_offence():
    assert cross_code_flags("The charge is under Section 302 IPC.", kind_dates=NEW_OFFENCE) == [
        FLAG_OLD_FOR_NEW
    ]


def test_clean_texts():
    assert cross_code_flags("The charge is under Section 302 IPC.", kind_dates=OLD_OFFENCE) == []
    assert cross_code_flags("The charge is under Section 103 BNS.", kind_dates=NEW_OFFENCE) == []
    assert cross_code_flags("No statute is cited here.", kind_dates=OLD_OFFENCE) == []


def test_split_cell_text_is_clean_but_a_bns_charge_in_it_is_flagged():
    dates = {"offence_date": date(2023, 5, 1), "proceeding_started": date(2024, 8, 1)}
    clean = (
        "For the 2023 offence the charge remains under Section 302 IPC, while the "
        "investigation registered in August 2024 runs under Section 173 BNSS."
    )
    assert cross_code_flags(clean, kind_dates=dates) == []
    dirty = clean + " The accused is also charged under Section 103 BNS."
    assert cross_code_flags(dirty, kind_dates=dates) == [FLAG_NEW_FOR_OLD]


def test_pending_proceeding_keeps_the_old_procedural_code():
    dates = {"offence_date": date(2023, 5, 1), "proceeding_started": date(2024, 2, 1)}
    assert cross_code_flags("The appeal continues under Section 374 CrPC.", kind_dates=dates) == []
    assert cross_code_flags("The appeal is filed under Section 415 BNSS.", kind_dates=dates) == [
        FLAG_NEW_FOR_OLD
    ]


def test_flags_are_deduped_and_undecidable_citations_are_skipped():
    text = "Sections 103 and 105 BNS both apply."
    assert cross_code_flags(text, kind_dates=OLD_OFFENCE) == [FLAG_NEW_FOR_OLD]
    # no dates at all: a substantive citation cannot be judged, so nothing is guessed
    assert cross_code_flags(text, kind_dates={"offence_date": None, "proceeding_started": None}) == []


# --------------------------------------------------------------------------
# Review fix 2: the s.358 suppression is code-aware. IPC 358, CrPC 358 and
# BNSS 358 are real, routinely cited sections - only BNS 358 is the savings
# clause, and a bare "section 358" attributed to another code must not disarm
# a genuine flag.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "The accused is liable under Section 103 BNS. Compensation was sought under Section 358 CrPC.",
        "Section 103 BNS applies. See also Section 358 IPC on grave provocation.",
        "Section 103 BNS applies; Section 358 BNSS governs the appeal.",
    ],
)
def test_section_358_of_another_code_does_not_suppress(text):
    assert cross_code_flags(text, kind_dates=OLD_OFFENCE) == [FLAG_NEW_FOR_OLD]


@pytest.mark.parametrize(
    "text",
    [
        "Section 103 BNS now covers murder, but section 358 BNS saves the repealed IPC.",
        "Section 103 BNS is irrelevant here: see s. 358 BNS.",
        "Section 103 BNS applies, though the section 358 savings clause preserves the IPC.",
        "Section 103 BNS, but section 358 of the Bharatiya Nyaya Sanhita saves the IPC.",
    ],
)
def test_genuine_savings_clause_still_suppresses(text):
    assert cross_code_flags(text, kind_dates=OLD_OFFENCE) == []


# --------------------------------------------------------------------------
# Review fix 3: "with" is only a citation lead-in in fixed legal collocations.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "with 20 IPC provisions dropped",
        "compared with 511 IPC sections",
        "dealt with 45 CrPC applications",
        "with 358 BNS sections renumbered",
        "the file was tagged with 34 IPC entries",
        "we begin with 302 IPC-era jurisprudence",
        "dispensed with 173 BNSS formalities",
        "told us 302 IPC applies",  # the u-s abbreviation needs its slash or dot
    ],
)
def test_bare_with_and_bare_us_are_not_citation_lead_ins(text):
    assert extract_sections(text) == []


def test_with_false_positive_cannot_disarm_the_savings_check():
    text = "Section 103 BNS applies, with 358 BNS sections renumbered by the new code."
    assert cross_code_flags(text, kind_dates=OLD_OFFENCE) == [FLAG_NEW_FOR_OLD]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("read with 34 IPC", [("IPC", "34")]),
        ("charged with 420 IPC", [("IPC", "420")]),
        ("punishable with 302 IPC", [("IPC", "302")]),
        ("along with 149 IPC", [("IPC", "149")]),
        ("u/s 420 IPC", [("IPC", "420")]),
        ("u/ss 420 and 34 IPC", [("IPC", "420"), ("IPC", "34")]),
        # the canonical charge-sheet form: the code is stated once, at the end
        ("convicted under Section 302 read with Section 34 IPC", [("IPC", "302"), ("IPC", "34")]),
    ],
)
def test_legal_collocations_still_extract(text, expected):
    assert [(r.code, r.number) for r in extract_sections(text)] == expected


# --------------------------------------------------------------------------
# Review fix 4: bounded whitespace runs - degenerate model output must not
# stall the gate (this input took ~78s before the fix).
# --------------------------------------------------------------------------

def test_extraction_stays_linear_on_degenerate_whitespace():
    for text in (
        "Section 302" + " " * 16000 + "IPC",
        "u/s 302, 307" + " " * 16000 + "IPC",
        "Section 302" + " " * 65536 + "IPC",
    ):
        start = time.perf_counter()
        extract_sections(text)
        assert time.perf_counter() - start < 1.0


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Section 302 of the\n            IPC", [("IPC", "302")]),
        (
            "u/s 302,\n    307 and 34\n    IPC",
            [("IPC", "302"), ("IPC", "307"), ("IPC", "34")],
        ),
        ("charged under Section 103(2)\n\n        of the Bharatiya Nyaya Sanhita", [("BNS", "103(2)")]),
        ("read with\n        Section 34\n        IPC", [("IPC", "34")]),
    ],
)
def test_wrapped_and_indented_citations_still_extract(text, expected):
    assert [(r.code, r.number) for r in extract_sections(text)] == expected


# --------------------------------------------------------------------------
# Review fix 5: undecidable must be distinguishable from clean.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kind_dates",
    [
        {"offence_date": None, "proceeding_started": None},
        {},
        {"offence_date": None, "proceeding_started": date(2024, 8, 1)},
    ],
)
def test_undecidable_substantive_citations_are_reported_not_swallowed(kind_dates):
    flags, undecidable = cross_code_review("Section 103 BNS applies.", kind_dates=kind_dates)
    assert flags == []
    assert undecidable == [SectionRef("BNS", "103")]


def test_decidable_text_reports_nothing_undecidable():
    flags, undecidable = cross_code_review("Section 302 IPC applies.", kind_dates=OLD_OFFENCE)
    assert (flags, undecidable) == ([], [])


def test_undecidable_and_flagged_can_coexist():
    # no offence date: the substantive cite is undecidable, while the
    # procedural one is decided by the proceeding date and is wrong
    dates = {"offence_date": None, "proceeding_started": date(2024, 8, 1)}
    flags, undecidable = cross_code_review(
        "Charged under Section 302 IPC; the appeal lies under Section 374 CrPC.", kind_dates=dates
    )
    assert flags == [FLAG_OLD_FOR_NEW]
    assert undecidable == [SectionRef("IPC", "302")]


def test_cross_code_flags_is_the_flags_channel_of_the_review():
    text = "Section 103 BNS applies."
    assert cross_code_flags(text, kind_dates=OLD_OFFENCE) == cross_code_review(
        text, kind_dates=OLD_OFFENCE
    )[0]


@pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy, lambda f: pickle.loads(pickle.dumps(f))])
def test_code_flags_survive_copy_and_pickle(clone):
    """A str subclass with a 2-argument __new__ needs __getnewargs__, or every
    copy/deepcopy/pickle of a gate result blows up."""
    flags, _ = cross_code_review("liable under Section 103 BNS", kind_dates=OLD_OFFENCE)
    cloned = clone(flags[0])
    assert cloned == FLAG_NEW_FOR_OLD
    assert cloned.ref == SectionRef("BNS", "103")
    assert isinstance(cloned, type(flags[0]))


def test_flags_name_the_offending_section():
    flags, _ = cross_code_review(
        "The accused is liable under Section 103 BNS.", kind_dates=OLD_OFFENCE
    )
    assert flags == [FLAG_NEW_FOR_OLD]  # still a plain string to every caller
    assert isinstance(flags[0], str)
    assert flags[0].ref == SectionRef("BNS", "103")


# --------------------------------------------------------------------------
# packaged resources + Mapping + SectionRegistry
# --------------------------------------------------------------------------

def test_resources_file_ships_with_the_package():
    from importlib.resources import files

    traversable = files("tuned.data").joinpath("resources").joinpath("ipc_bns_map.jsonl")
    assert traversable.is_file()
    assert IPC_BNS_MAP_PATH.is_file()
    assert RESOURCES_DIR.is_dir()
    assert IPC_BNS_MAP_PATH.read_text(encoding="utf-8").strip()


def test_mapping_loads_the_starter_rows():
    mapping = Mapping.load()
    assert len(mapping) >= 25
    kinds = {row["kind"] for row in mapping}
    assert kinds == {"one_to_one", "changed", "new_offence", "deleted"}


def test_the_audited_map_is_mostly_verified_and_the_remainder_is_named():
    """The 2026-08-16 source audit stamped every row where two independent
    sources - at least one official, current-edition bare acts preferred -
    agree on every cell. The 17 unverified rows are exactly the operator
    decision sheet (kind-only questions; every BNS number two-source
    agreed). These pins MOVE when the operator rules on a sheet row:
    update them with the ruling, never loosen them to inequalities."""
    mapping = Mapping.load()
    assert len(mapping) == 171
    assert len(mapping.verified_rows()) == 154
    assert len(mapping.unverified_rows()) == 17
    assert len(mapping.verified_rows()) + len(mapping.unverified_rows()) == len(mapping)


@pytest.mark.parametrize(
    "old,new",
    [
        (("IPC", "302"), ("BNS", "103")),
        (("IPC", "304B"), ("BNS", "80")),
        (("IPC", "306"), ("BNS", "108")),
        (("IPC", "307"), ("BNS", "109")),
        (("IPC", "375"), ("BNS", "63")),
        (("IPC", "376"), ("BNS", "64")),
        (("IPC", "379"), ("BNS", "303")),
        (("IPC", "392"), ("BNS", "309")),
        (("IPC", "406"), ("BNS", "316")),
        (("IPC", "420"), ("BNS", "318")),
        (("IPC", "498A"), ("BNS", "85")),
        (("IPC", "499"), ("BNS", "356")),
        (("IPC", "500"), ("BNS", "356")),
        (("IPC", "34"), ("BNS", "3(5)")),
        (("IPC", "120B"), ("BNS", "61")),
        # 124A -> 152 was DELETED here by the source audit: all four sources
        # treat sedition as repealed with no counterpart and BNS 152 as a new
        # offence - the old case encoded the starter map's one structural error.
    ],
)
def test_counterpart_one_to_one_and_changed(old, new):
    assert Mapping.load().counterpart(SectionRef(*old)) == SectionRef(*new)


def test_counterpart_deleted_sections_have_none():
    mapping = Mapping.load()
    for number in ("377", "497"):
        ref = SectionRef("IPC", number)
        assert mapping.counterpart(ref) is None
        assert mapping.row(ref)["kind"] == "deleted"


def test_counterpart_new_offences_have_no_old_side():
    mapping = Mapping.load()
    for number in ("111", "113", "103(2)", "304"):
        ref = SectionRef("BNS", number)
        assert mapping.counterpart(ref) is None
        assert mapping.row(ref)["kind"] == "new_offence"
        assert mapping.row(ref)["old_section"] is None


def test_counterpart_reverse_lookup_and_ambiguity():
    mapping = Mapping.load()
    assert mapping.counterpart(SectionRef("BNS", "103")) == SectionRef("IPC", "302")
    # IPC 499 and IPC 500 both land on BNS 356 - never guess which one
    assert mapping.counterpart(SectionRef("BNS", "356")) is None
    assert mapping.row(SectionRef("BNS", "356")) is None


def test_counterpart_unmapped():
    mapping = Mapping.load()
    assert mapping.counterpart(SectionRef("IPC", "999")) is None
    assert mapping.row(SectionRef("IPC", "999")) is None


def test_require_verified_passes_audited_rows_and_refuses_the_sheet():
    """IPC 302 carries a two-source stamp and flows. IPC 375 ships a kind
    at low confidence and sits on the operator decision sheet, so it must
    refuse until a human rules - rape is exactly the section where a token
    comparison is not an instrument and the audit said so out loud."""
    mapping = Mapping.load()
    assert mapping.require_verified(SectionRef("IPC", "302"))["new_section"] == "103"
    with pytest.raises(ValueError, match="unverified"):
        mapping.require_verified(SectionRef("IPC", "375"))
    with pytest.raises(ValueError, match="no mapping row"):
        mapping.require_verified(SectionRef("IPC", "999"))


def test_require_verified_error_names_the_row():
    # 375 rather than 420: the audit verified 420, and the row this error
    # names must be one that actually refuses.
    mapping = Mapping.load()
    with pytest.raises(ValueError) as exc:
        mapping.require_verified(SectionRef("IPC", "375"))
    message = str(exc.value)
    assert "IPC 375" in message and "BNS 63" in message


def test_require_verified_accepts_an_audited_row(tmp_path):
    path = tmp_path / "map.jsonl"
    write_jsonl(
        path,
        [
            {
                "old_code": "IPC",
                "old_section": "302",
                "new_code": "BNS",
                "new_section": "103",
                "kind": "one_to_one",
                "verified_by": "operator:bare-act-audit-2026-08",
                "notes": "",
            }
        ],
    )
    mapping = Mapping.load(path)
    assert mapping.require_verified(SectionRef("IPC", "302"))["verified_by"].startswith("operator:")
    assert len(mapping.verified_rows()) == 1


def test_mapping_skips_comment_rows(tmp_path):
    path = tmp_path / "map.jsonl"
    path.write_text(
        json.dumps({"_comment": "header"})
        + "\n"
        + json.dumps(
            {
                "old_code": "IPC",
                "old_section": "302",
                "new_code": "BNS",
                "new_section": "103",
                "kind": "one_to_one",
                "verified_by": None,
                "notes": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert len(Mapping.load(path)) == 1


@pytest.mark.parametrize(
    "row,match",
    [
        ({"old_code": "IPC", "old_section": "302", "new_code": "BNS", "new_section": "103"}, "missing keys"),
        (
            {"old_code": "IPC", "old_section": "302", "new_code": "BNS", "new_section": "103",
             "kind": "renamed", "verified_by": None},
            "unknown kind",
        ),
        (
            {"old_code": "IPC", "old_section": "302", "new_code": "BNS", "new_section": "103",
             "kind": "new_offence", "verified_by": None},
            "new_offence",
        ),
        (
            {"old_code": "IPC", "old_section": "377", "new_code": "BNS", "new_section": "103",
             "kind": "deleted", "verified_by": None},
            "deleted",
        ),
        (
            {"old_code": "IPC", "old_section": "302", "new_code": None, "new_section": None,
             "kind": "one_to_one", "verified_by": None},
            "both sides",
        ),
        (
            {"old_code": "BNS", "old_section": "103", "new_code": "BNS", "new_section": "103",
             "kind": "one_to_one", "verified_by": None},
            "not an old code",
        ),
        (
            {"old_code": "IPC", "old_section": "302", "new_code": "IPC", "new_section": "103",
             "kind": "one_to_one", "verified_by": None},
            "not a new code",
        ),
    ],
)
def test_mapping_rejects_malformed_rows(row, match):
    with pytest.raises(ValueError, match=match):
        Mapping([row])


def test_mapping_rejects_duplicate_old_sections():
    row = {
        "old_code": "IPC", "old_section": "302", "new_code": "BNS", "new_section": "103",
        "kind": "one_to_one", "verified_by": None, "notes": "",
    }
    with pytest.raises(ValueError, match="duplicate"):
        Mapping([row, dict(row)])


def test_section_registry_membership_and_base_number_fallback():
    registry = SectionRegistry.load()
    assert registry.contains(SectionRef("IPC", "302"))
    assert registry.contains(SectionRef("BNS", "103"))
    assert registry.contains(SectionRef("bns", "103"))
    # base-number fallback: the registry knows BNS 103, so 103(2) resolves
    assert registry.contains(SectionRef("BNS", "103(2)"))
    assert SectionRef("IPC", "304B") in registry
    # a subsection in the file also registers its parent section
    assert registry.contains(SectionRef("BNS", "3(5)"))
    assert registry.contains(SectionRef("BNS", "3"))
    assert not registry.contains(SectionRef("IPC", "9999"))
    assert not registry.contains(SectionRef("MVA", "185"))
    assert len(registry) > 0


def test_section_registry_accepts_plain_code_section_rows(tmp_path):
    path = tmp_path / "sections.jsonl"
    write_jsonl(path, [{"code": "BNSS", "section": "173"}, {"code": "BSA", "section": 61}])
    registry = SectionRegistry.load(path)
    assert registry.contains(SectionRef("BNSS", "173"))
    assert registry.contains(SectionRef("BSA", "61"))  # integer section numbers survive
    assert not registry.contains(SectionRef("BNSS", "9999"))


def test_section_registry_extends_with_extra_files(tmp_path):
    path = tmp_path / "extra.jsonl"
    write_jsonl(path, [{"code": "CRPC", "section": "173"}])
    registry = SectionRegistry.load(IPC_BNS_MAP_PATH, path)
    assert registry.contains(SectionRef("IPC", "302"))
    assert registry.contains(SectionRef("CRPC", "173"))
