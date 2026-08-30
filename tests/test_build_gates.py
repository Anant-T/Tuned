import json
import random
from datetime import date

import pytest

from tuned.data.citations import CitationIndex
from tuned.data.config import LengthBand, load_build_config
from tuned.data.gates import (
    BANNED_META,
    DEFAULT_MAX_RUN,
    DIAGNOSTIC_GATES,
    GATE_ORDER,
    IRAC_ANSWER_TASK_TYPES,
    IRAC_REQUIRED,
    MAX_UNGROUNDED_REFS,
    PERMANENT_GATES,
    RANGE_SPAN_MAX,
    SHINGLE_STEP,
    STATUTORY_FAMILIES,
    VERIFICATION_CUES,
    _norm_ws,
    GateContext,
    GateResult,
    check_answer_key,
    check_banned_meta,
    check_citations,
    check_irac_placement,
    check_length_band,
    check_self_verification,
    check_statutory_grounding,
    check_temporal,
    check_think_format,
    check_statutory_quotation,
    check_verbatim_overlap,
    disposition,
    find_verbatim_run,
    grounded_refs,
    irac_headings,
    run_all,
    split_think,
    statutory_refs,
)
from tuned.data.statutes import APPOINTED_DAY

# Offence dates either side of the appointed day (2024-07-01).
BEFORE = date(2023, 5, 4)
AFTER = date(2024, 9, 1)

# The grounding chunk the teacher is shown. Cites one real reporter citation
# and carries a holding worth quoting.
#
# IT ALSO NAMES THE CHARGE, and that sentence is load-bearing rather than
# scene-setting: GOOD_ANSWER opens its Rule limb with "Section 302 IPC governs
# the charge", and under check_statutory_grounding an answer that names a
# provision the materials never carried is not a valid example - it is the
# GATE-1 defect. Without this line the end-to-end "fully valid" fixture
# asserts that a row citing a section from nowhere passes every gate. Appended
# rather than woven in, so SOURCE_RUN_35 / SOURCE_RUN_LONG keep slicing the
# same prose.
SOURCE = (
    "In Anwar Ali v. State, (2008) 1 SCC 1, the Supreme Court held that a "
    "conviction may rest on circumstantial evidence only where the chain of "
    "circumstances is so complete as to exclude every hypothesis except the "
    "guilt of the accused. The trial court had relied on the recovery of a "
    "blood-stained weapon at the instance of the accused and on the testimony "
    "of a single eye witness who deposed four days after the incident. "
    "The appellant stands convicted under Section 302 IPC. "
    # Extended 2026-08-28 so SOURCE_RUN_LONG can slice DEFAULT_MAX_RUN (500)
    # contiguous characters from offset 80. Deliberately inert prose: no
    # citation, no section number, no quotation marks - nothing that would
    # widen grounded_refs or the citation index the other fixtures pin.
    "The High Court had earlier declined to interfere, observing that the "
    "appreciation of the recovered articles and of the witness statements was "
    "a matter squarely within the province of the court of first instance, "
    "and that no perversity had been shown in the way that court weighed them."
)

INDEX_ENTRIES = ["(2008) 1 SCC 1", "2023 INSC 45", "AIR 1973 SC 1461"]

# Deliberately small so band violations are easy to aim at one at a time.
BAND = LengthBand(total_max=4000, total_min=100, think_min=50, think_max=800, answer_min=20)

_CFG = None


def _cfg():
    global _CFG
    if _CFG is None:
        _CFG = load_build_config("data/configs/data_law_v1.yaml", allow_unpinned=True)
    return _CFG


def _tags():
    cfg = _cfg()
    return cfg.think_open, cfg.think_close


@pytest.fixture
def index(tmp_path):
    return CitationIndex.build(INDEX_ENTRIES, tmp_path / "citation_index.txt")


def _ctx(**over):
    think_open, think_close = _tags()
    base = dict(
        think_open=think_open,
        think_close=think_close,
        band=BAND,
        citation_index=None,
        source_text=SOURCE,
        offence_date=BEFORE,
        proceeding_started=None,
        stream="synthesis",
        expect_reasoning=True,
        answer_key=None,
    )
    base.update(over)
    return GateContext(**base)


def _content(think: str, answer: str) -> str:
    think_open, think_close = _tags()
    return f"{think_open}{think}{think_close}{answer}"


def _empty_think_content(answer: str) -> str:
    think_open, think_close = _tags()
    return f"{think_open}\n\n{think_close}{answer}"


def _norm(text: str) -> str:
    return " ".join(text.split())


# A >=30-char run lifted straight out of the source, whitespace-normalized.
SOURCE_RUN_35 = _norm(SOURCE)[80:115]
# A run at the gate's own threshold, derived from it so the two cannot drift.
SOURCE_RUN_LONG = _norm(SOURCE)[80 : 80 + DEFAULT_MAX_RUN]


# --------------------------------------------------------------------------
# A fully valid synthetic example, used for the end-to-end run_all pass.
# --------------------------------------------------------------------------

GOOD_THINK = (
    "Let me check what the prosecution actually proved. There is no direct "
    "testimony tying this man to the killing, so everything runs through "
    "inference and each link has to stand on its own feet rather than borrow "
    "credibility from the others.\n\n"
    "The weapon recovery matters only if the discovery statement was proved "
    "in the manner the evidence statute demands and the officer who effected "
    "it was examined. Absent that foundation the seizure adds nothing and "
    "cannot be counted as a link at all.\n\n"
    "A lone witness can sustain a conviction where the deposition is of "
    "sterling quality, but quality is tested against the surrounding facts, "
    "never assumed. This man surfaced well after the event, which is a reason "
    "to look harder, not a reason to throw the account away outright.\n\n"
    "Wait, I should pin down which code applies before going further. The "
    "killing predates the appointed day, so the charge stays with the old "
    "penal statute; the passage of time never converts it into an offence "
    "under the new one. Steps taken afterwards are a separate question and do "
    "not touch the charging provision.\n\n"
    "Let me verify the authority I am leaning on: (2008) 1 SCC 1 is the "
    "decision on when an inferential case suffices, and that is exactly the "
    "standard this dispute turns on.\n\n"
    "One more point on sentence: a lesser alternative is open only where "
    "proof supports it, and nothing here points that way, so I should not "
    "invent a fallback nobody argued.\n\n"
    "Double-check on dates: incident, arrest and committal all fall before "
    "the transition, so every substantive question stays with the old code, "
    "and I can write this up without hedging on that point.\n\n"
    "It is worth separating what was actually established from what the "
    "prosecutor merely asserted in argument. The panchnama was signed by two "
    "neighbours who were never called, the seizure memo carries a date that "
    "nobody spoke to, and the investigating officer retired before the trial "
    "opened. None of that is fatal by itself, yet together it leaves the "
    "second limb of the case resting on paper rather than on sworn "
    "testimony.\n\n"
    "To confirm the shape of the answer before drafting: I will name the "
    "charging provision, set out the standard that governs an inferential "
    "case, walk the two contested links, and then say plainly what follows "
    "for the conviction. I should resist the urge to catalogue every minor "
    "discrepancy in the deposition, because the case turns on the two links "
    "and padding the answer with small points would blur that."
)

GOOD_ANSWER = (
    "## Issue\n"
    "Whether a conviction can be sustained where the prosecution case rests "
    "wholly on inference, a disputed weapon seizure and one witness who came "
    "forward only after several days.\n\n"
    "## Rule\n"
    "Section 302 IPC governs the charge, the offence predating the appointed "
    "day. On the sufficiency of an inferential case the Supreme Court in "
    "(2008) 1 SCC 1 held that a conviction may rest on circumstantial "
    "evidence only where the chain of circumstances is so complete as to "
    "exclude every hypothesis except the guilt of the accused.\n\n"
    "## Application\n"
    "The seizure was not spoken to by the officer who made it, so it cannot "
    "be treated as an established link. The lone deposition, delayed and "
    "uncorroborated, does not reach the sterling standard. What is left is a "
    "chain with two broken links, which leaves open a hypothesis other than "
    "guilt.\n\n"
    "## Conclusion\n"
    "The conviction cannot be sustained on this record and the accused is "
    "entitled to the benefit of the doubt."
)

def _good_content():
    return _content(GOOD_THINK, GOOD_ANSWER)


# A trace that reasons ABOUT the forbidden successor provision and correctly
# rules it out - legitimate in the trace, and absent from the answer. The
# savings reference is what keeps the temporal gate quiet.
TRANSITION_THINK = GOOD_THINK + (
    "\n\nOne last note on the successor provision: Section 103 BNS would be "
    "the modern analogue of this charge, but Section 358 BNS preserves the "
    "liability incurred under the repealed code, so it is not the charging "
    "provision here."
)


# --------------------------------------------------------------------------
# Module constants.
# --------------------------------------------------------------------------

def test_gate_order_and_permanent_gates():
    assert GATE_ORDER == (
        "think_format",
        "length_band",
        "citations",
        "statutory_grounding",
        "temporal",
        "self_verification",
        "irac_placement",
        "verbatim_overlap",
        "statutory_quotation",
        "banned_meta",
        "prompt_echo",
        "answer_key",
    )
    assert PERMANENT_GATES == frozenset({"citations", "temporal", "answer_key"})
    # statutory_quotation is deliberately NOT permanent: the same sentence
    # without the quotation marks is a true statement of the recorded effect,
    # so rewriting the prose does make it right, which is this module's own
    # definition of a regenerate.
    assert "statutory_quotation" not in PERMANENT_GATES
    # statutory_grounding is not permanent either, and for the neighbouring
    # reason: a section the materials never carried is UNSUPPORTED, not false.
    # A fresh answer over the same materials can be supported, so the seed is
    # not burned. citations is permanent because an authority that does not
    # exist is false however it is rewritten.
    assert "statutory_grounding" not in PERMANENT_GATES
    assert PERMANENT_GATES <= set(GATE_ORDER)


def test_appointed_day_is_the_statutes_constant():
    # The fixture dates only mean anything relative to the real appointed day.
    assert BEFORE < APPOINTED_DAY <= AFTER


def test_gate_result_as_row_matches_the_store_row_shape():
    result = GateResult("citations", False, {"novel": ["(2019) 4 SCC 999"]})
    gate, passed, detail = result.as_row()
    assert (gate, passed) == ("citations", False)
    assert json.loads(json.dumps(detail)) == detail


# --------------------------------------------------------------------------
# split_think
# --------------------------------------------------------------------------

def test_split_think_well_formed():
    think, answer = split_think(_content("  reasoning  ", "the answer"), *_tags())
    assert think == "  reasoning  "
    assert answer == "the answer"


def test_split_think_no_tags_returns_none_and_the_whole_content():
    think, answer = split_think("no tags at all", *_tags())
    assert think is None
    assert answer == "no tags at all"


def test_split_think_empty_block():
    think, answer = split_think(_empty_think_content("answer"), *_tags())
    assert think == "\n\n"
    assert answer == "answer"


def test_split_think_second_open_tag_is_not_well_formed():
    think_open, think_close = _tags()
    content = f"{think_open}a{think_open}b{think_close}answer"
    assert split_think(content, think_open, think_close)[0] is None


def test_split_think_is_not_fooled_by_a_tag_inside_the_answer():
    think_open, think_close = _tags()
    for answer in (
        f"answer with a stray {think_close}",
        f"answer with a stray {think_open}",
        f"answer {think_open}second block{think_close}",
    ):
        think, whole = split_think(_content("trace", answer), think_open, think_close)
        assert think is None, answer
        assert whole == _content("trace", answer)


def test_split_think_close_before_open_is_not_well_formed():
    think_open, think_close = _tags()
    assert split_think(f"{think_close}trace{think_open}", think_open, think_close)[0] is None


def test_split_think_keeps_a_preamble_inside_the_answer():
    think_open, think_close = _tags()
    content = f"Sure, here it is.{think_open}trace{think_close}Conclusion: acquit."
    think, answer = split_think(content, think_open, think_close)
    assert think == "trace"
    assert "Sure, here it is." in answer
    assert "Conclusion: acquit." in answer


def test_split_think_missing_tag_config_never_raises():
    assert split_think("whatever", "", "</think>") == (None, "whatever")
    assert split_think(None, "<think>", "</think>") == (None, "")


# --------------------------------------------------------------------------
# check_think_format
# --------------------------------------------------------------------------

def test_think_format_valid_reasoning_row():
    result = check_think_format(_content("a real trace", "answer"), _ctx())
    assert result.gate == "think_format"
    assert result.passed
    assert result.detail["open_count"] == 1 and result.detail["close_count"] == 1
    assert result.detail["prefix_chars"] == 0


def test_think_format_empty_trace_when_reasoning_expected_fails():
    result = check_think_format(_content("   \n  ", "answer"), _ctx())
    assert not result.passed
    assert result.detail["reason"] == "empty-trace"


def test_think_format_empty_block_byte_exact_passes():
    result = check_think_format(_empty_think_content("answer"), _ctx(expect_reasoning=False))
    assert result.passed


@pytest.mark.parametrize("inner", ["", "\n", "\n\n\n", " \n\n ", "\r\n\r\n", "\t"])
def test_think_format_empty_block_whitespace_variants_fail(inner):
    result = check_think_format(_content(inner, "answer"), _ctx(expect_reasoning=False))
    assert not result.passed
    assert result.detail["reason"] == "empty-block-not-byte-exact"


def test_think_format_real_trace_where_none_was_expected_fails():
    result = check_think_format(_content("surprise trace", "answer"), _ctx(expect_reasoning=False))
    assert not result.passed
    assert result.detail["reason"] == "empty-block-not-byte-exact"


def test_think_format_nested_and_doubled_tags_fail():
    think_open, think_close = _tags()
    ctx = _ctx()
    nested = f"{think_open}outer {think_open}inner{think_close} rest{think_close}answer"
    doubled = f"{think_open}a{think_close}mid{think_open}b{think_close}answer"
    for content in (nested, doubled):
        result = check_think_format(content, ctx)
        assert not result.passed
        assert result.detail["reason"] == "not-exactly-one-pair"


def test_think_format_no_tags_at_all_fails():
    result = check_think_format("just prose", _ctx())
    assert not result.passed
    assert result.detail == {
        "open_count": 0,
        "close_count": 0,
        "expect_reasoning": True,
        "reason": "not-exactly-one-pair",
    }


def test_think_format_close_before_open_fails():
    think_open, think_close = _tags()
    result = check_think_format(f"{think_close}trace{think_open}", _ctx())
    assert not result.passed
    assert result.detail["reason"] == "close-before-open"


def test_think_format_without_configured_tags_fails_fast():
    # Empty tag strings match at every offset; the guard keeps the scan out of
    # its quadratic path and there is nothing to verify anyway.
    result = check_think_format("a" * 20000, _ctx(think_open="", think_close=""))
    assert not result.passed
    assert result.detail == {"reason": "no-think-tags-configured"}


def test_think_format_records_a_preamble_without_failing():
    think_open, think_close = _tags()
    result = check_think_format(f"Here goes.{think_open}trace{think_close}answer", _ctx())
    assert result.passed
    assert result.detail["prefix_chars"] == len("Here goes.")


# --------------------------------------------------------------------------
# check_length_band
# --------------------------------------------------------------------------

def test_length_band_pass_carries_every_number():
    result = check_length_band(100, 200, 300, _ctx())
    assert result.passed
    assert result.detail["prompt_est"] == 100
    assert result.detail["think_est"] == 200
    assert result.detail["answer_est"] == 300
    assert result.detail["total_est"] == 600
    assert result.detail["total_max"] == BAND.total_max
    assert result.detail["violations"] == []


def test_length_band_total_max():
    result = check_length_band(3000, 700, 400, _ctx())
    assert not result.passed
    assert result.detail["violations"] == ["total>total_max"]


def test_length_band_total_min():
    result = check_length_band(5, 50, 20, _ctx())
    assert not result.passed
    assert "total<total_min" in result.detail["violations"]


def test_length_band_think_max():
    result = check_length_band(100, 900, 300, _ctx())
    assert not result.passed
    assert result.detail["violations"] == ["think>think_max"]


def test_length_band_think_min_only_when_reasoning_expected():
    short = check_length_band(100, 10, 300, _ctx())
    assert not short.passed
    assert short.detail["violations"] == ["think<think_min"]
    assert short.detail["think_min_applies"] is True

    empty_think_row = check_length_band(100, 0, 300, _ctx(expect_reasoning=False))
    assert empty_think_row.passed
    assert empty_think_row.detail["think_min_applies"] is False


def test_length_band_answer_min():
    result = check_length_band(100, 200, 5, _ctx())
    assert not result.passed
    assert result.detail["violations"] == ["answer<answer_min"]


def test_length_band_reports_every_violation_at_once():
    result = check_length_band(0, 0, 0, _ctx())
    assert not result.passed
    assert set(result.detail["violations"]) == {
        "total<total_min",
        "think<think_min",
        "answer<answer_min",
    }


def test_length_band_coerces_junk_counts_instead_of_raising():
    result = check_length_band(None, "200", 300, _ctx())
    assert result.detail["prompt_est"] == 0
    assert result.detail["think_est"] == 200


def test_length_band_survives_an_infinite_count():
    result = check_length_band(float("inf"), 200, 300, _ctx())
    assert result.detail["prompt_est"] == 0
    assert json.loads(json.dumps(result.detail)) == result.detail


def test_length_band_against_the_real_config_band():
    band = _cfg().build.length_band
    ctx = _ctx(band=band)
    assert check_length_band(400, 900, 400, ctx).passed
    assert not check_length_band(400, 100, 400, ctx).passed  # under think_min
    assert not check_length_band(400, 900, 8000, ctx).passed  # over total_max


# --------------------------------------------------------------------------
# check_citations
# --------------------------------------------------------------------------

def test_citations_without_an_index_still_runs_the_suspect_channel():
    # The suspect verdict never depended on the index, so an invented cite in
    # an unmodelled reporter is rejected during the pilot rather than paid for
    # twice.
    result = check_citations("Relying on 2011 (2) KLT 123, the court agreed.", _ctx())
    assert not result.passed
    assert result.detail["suspect"] == ["2011 (2) KLT 123"]
    assert result.detail["novel"] is None
    assert result.detail["novel_skipped"] == "no-index"


def test_citations_without_an_index_passes_but_records_the_unchecked_half():
    # A fabricated citation in a MODELLED format is invisible until the index
    # exists: this pass means "suspect-clean", not "citations verified", and
    # verify.py must re-run the gate before the row is promoted.
    result = check_citations("Following (2019) 4 SCC 999 the appeal fails.", _ctx())
    assert result.passed
    assert result.detail == {"novel": None, "novel_skipped": "no-index", "suspect": []}


def test_citations_fabricated_citation_fails_and_is_a_permanent_reject(index):
    ctx = _ctx(citation_index=index)
    result = check_citations("The point is settled by (2019) 4 SCC 999.", ctx)
    assert not result.passed
    assert result.detail["novel"] == ["(2019) 4 SCC 999"]
    assert disposition([result]) == "reject"


def test_citations_indexed_citation_passes(index):
    result = check_citations("See AIR 1973 SC 1461 on this.", _ctx(citation_index=index))
    assert result.passed
    assert result.detail == {"novel": [], "suspect": []}


def test_citations_grounded_but_unindexed_citation_passes(index):
    source = SOURCE + " Followed in 2024:KER:5555."
    ctx = _ctx(citation_index=index, source_text=source)
    result = check_citations("As held in 2024:KER:5555, the rule stands.", ctx)
    assert result.passed


def test_citations_suspect_channel_fails_when_absent_from_the_source(index):
    ctx = _ctx(citation_index=index)
    result = check_citations("Relying on 2011 (2) KLT 123, the court agreed.", ctx)
    assert not result.passed
    assert result.detail["novel"] == []
    assert result.detail["suspect"] == ["2011 (2) KLT 123"]


def test_citations_same_suspect_in_source_and_output_passes(index):
    ctx = _ctx(citation_index=index, source_text=SOURCE + " See 2011 (2) KLT 123.")
    result = check_citations("Relying on 2011 (2) KLT 123, the court agreed.", ctx)
    assert result.passed
    assert result.detail["suspect"] == []


def test_citations_reads_the_whole_content_including_the_trace(index):
    ctx = _ctx(citation_index=index)
    content = _content("privately relying on (2019) 4 SCC 999", "a clean answer")
    assert not check_citations(content, ctx).passed


# The two pairs that cost a seed each in the pilot. citations is a PERMANENT
# gate, so both of these went straight to `rejected` with their attempts
# unspent - one at attempts=1 - over punctuation.
@pytest.mark.parametrize(
    "grounding, written",
    [
        # A space before the bracket, and a court marker the suspect pattern
        # does not reach.
        ("Cooperative Bank Ltd. vs. Santosh, 2015(4) KLT 163(LB). However,",
         "the rule in 2015 (4) KLT 163 governs"),
        # The year re-rendered into the standard bracketed form - a teacher
        # doing its job.
        ("Jyothi Ademma v. Plant Engineer, Nellore, [2006 (7) SCALE 28 ] held",
         "as held in (2006) 7 SCALE 28"),
    ],
)
def test_citations_a_reformatted_suspect_is_the_same_citation(grounding, written):
    """RE-TYPING A CITATION IS NOT INVENTING ONE (2026-08-18).

    The suspect diff compared raw strings, so a citation carried in by the
    grounding and written back in standard form read as novel. Both pairs below
    are the same case and both burned a seed permanently. The fix folds the
    punctuation they differ by; see citations.suspect_key.
    """
    ctx = _ctx(source_text=grounding)
    result = check_citations(written, ctx)
    assert result.passed, result.detail
    assert result.detail["suspect"] == []


def test_citations_a_reporter_absent_from_the_grounding_is_still_rejected():
    """The folding must not buy the fabrication its cover.

    The pilot's third citation reject was real: '(1955) I LLJ 688' attached to
    Shivnandan Sharma v. Punjab National Bank. suspect_key only ever equates
    two strings that are BOTH present, so an invention has nothing to fold
    against.

    THE GROUNDING CARRIES A DIFFERENT REPORTER ON PURPOSE (review round 2,
    M-1). It first used prose with no citation in it at all, which made the
    test vacuous: with an EMPTY grounding-key set every key is absent, so the
    assertion held no matter what suspect_key returned - including the empty
    string. A grounding carrying a real, near-miss LLJ citation is what makes
    the folding actually run and actually have to discriminate: '1950 2 LLJ
    921' against '1955 I LLJ 688', same reporter, different case.
    """
    ctx = _ctx(
        source_text="Following Bharat Bank v. Employees, (1950) 2 LLJ 921, the "
                    "tribunal proceeded to hear the reference."
    )
    result = check_citations("the point is settled by (1955) I LLJ 688", ctx)
    assert not result.passed
    assert result.detail["suspect"] == ["(1955) I LLJ 688"]
    assert disposition([result]) == "reject"


def test_citations_folding_does_not_merge_two_different_cases():
    """The key joins tokens rather than concatenating them, because '2015 (4)
    KLT 163' and '2015 (41) KLT 63' would otherwise fold together - a permanent
    gate failing OPEN, which is the direction that lets a fabrication in."""
    ctx = _ctx(source_text="Following 2015 (41) KLT 63 the appeal was allowed.")
    assert not check_citations("relying on 2015 (4) KLT 163", ctx).passed


# --------------------------------------------------------------------------
# check_statutory_grounding
# --------------------------------------------------------------------------

# Materials able to express every misreading the tests below exist to reject.
# Nothing here is decoration:
#   Section 319       a reference the answer may legitimately make.
#   Entry 56          the NEIGHBOUR of the fabricated Entry 54, so a numeric
#                     mismatch is a mismatch and not merely an absence.
#   "then aged 54"    the digits 54 in prose, so a digit-anchored check has
#                     something to be wrong about. NOT "54 years": a period
#                     is refused by _REF_NOT_A_PERIOD, which would shield a
#                     keyword-less mutant from the test that exists to kill
#                     it (measured - that mutant survived until this line
#                     changed).
#   Sections 62 to 65 a range whose interior no keyword names.
#   Section 14(1)     a subsection-only mention of a section.
#   397. ... hurt.—   enacted text naming its own section with no keyword,
#                     the marginal note wrapping across a line as bare-act
#                     text really does.
STAT_SOURCE = (
    "The charge-sheet was laid under Section 319 of the Code, and the "
    "competence question turns on Entry 56 of List II. The complainant, then "
    "aged 54, deposed on affidavit. Sections 62 to 65 were pressed, "
    "as was Section 14(1) of the Act.\n"
    "397. Robbery, or dacoity, with attempt to cause death or grievous\n"
    "hurt.—If, at the time of committing robbery, the offender uses any "
    "deadly weapon, the imprisonment shall not be less than seven years."
)


def _stat_ctx(**over):
    over.setdefault("source_text", STAT_SOURCE)
    return _ctx(**over)


def test_statutory_grounding_a_reference_the_materials_never_carried_fails():
    # The measured GATE-1 defect: s.313 appears nowhere in what the generator
    # was shown. Regenerate, never reject - the seed is not the problem.
    result = check_statutory_grounding(
        "The accused must be examined under Section 313 of the Code.", _stat_ctx()
    )
    assert not result.passed
    assert [(u["family"], u["number"]) for u in result.detail["ungrounded"]] == [
        ("section", "313")
    ]
    assert result.detail["ungrounded"][0]["as_written"] == "Section 313"
    assert "statutory_grounding" not in PERMANENT_GATES


def test_statutory_grounding_a_reference_the_materials_carried_passes():
    result = check_statutory_grounding(
        "Section 319 of the Code empowers the court to summon the absent "
        "accused, and Entry 56 of List II settles competence.",
        _stat_ctx(),
    )
    assert result.passed
    assert result.detail["ungrounded"] == []
    assert result.detail["answer_refs"] == 2


def test_statutory_grounding_a_neighbouring_number_is_not_the_same_reference():
    """Entry 54 against materials that cite Entry 56 - measured 17 times in
    gen 395, and the reason the number is compared and not merely the keyword.
    A check that asked 'does the answer cite SOME entry the materials cite'
    would pass this."""
    assert check_statutory_grounding("Entry 56 of List II governs.", _stat_ctx()).passed
    result = check_statutory_grounding("Entry 54 of List II governs.", _stat_ctx())
    assert not result.passed
    assert result.detail["ungrounded"][0]["number"] == "54"


def test_statutory_grounding_a_bare_number_in_prose_does_not_ground_a_reference():
    """The materials say the complainant was 54 years of age. Those digits are
    not Entry 54, and a check anchored on the digits rather than on the
    reference keyword would call the fabrication grounded."""
    assert ("entry", "54") not in grounded_refs(STAT_SOURCE)
    assert ("section", "54") not in grounded_refs(STAT_SOURCE)
    assert not check_statutory_grounding("Entry 54 of List II.", _stat_ctx()).passed


def test_statutory_grounding_the_keyword_is_required_on_both_sides():
    # No marker, no reference: the digits alone are never a citation, in the
    # answer or in the materials.
    assert statutory_refs("In 2023 some 302 appeals were filed; 54 succeeded.") == []
    assert check_statutory_grounding(
        "In 2023 some 302 appeals were filed.", _stat_ctx()
    ).passed


def test_statutory_grounding_ignores_a_reference_that_appears_only_in_the_trace():
    """A trace exists to reach for a provision and put it down again - gen 367
    was ACCEPTED after naming Article 15 in its trace and then refusing it in
    terms. Scoring the trace would reject that discipline for practising it.

    The fixture has to be able to express the misreading, so the fabricated
    reference is in the trace and the answer is otherwise clean.
    """
    think = (
        " Let me check whether Section 313 of the Code is engaged. It is not "
        "in the materials, so I will not cite it. "
    )
    answer = "\nIssue\nWhether the summons lies.\n\nConclusion\nSection 319 answers it."
    content = _content(think, answer)
    ctx = _stat_ctx()

    by_gate = {result.gate: result for result in run_all(content, 100, ctx)}
    assert by_gate["statutory_grounding"].passed
    # ... and the mutant that reads the whole content instead of the answer
    # really would have failed it, so the test is not vacuous.
    assert not check_statutory_grounding(content, ctx).passed


def test_statutory_grounding_is_not_evaluated_when_the_content_did_not_parse():
    """think is None means split_think could not parse the content, so
    `answer` is the WHOLE generation, trace included. Scoring it would score
    the trace - the very thing the test above pins. check_answer_key refuses
    the same shape for the same reason; check_think_format has already failed
    the row, so nothing is lost by declining."""
    whole = "<think>Section 313 might apply.</think> Section 319 applies."
    result = check_statutory_grounding(whole, _stat_ctx(), think=None)
    assert result.passed
    assert result.detail == {"skipped": "unparsed-format"}


@pytest.mark.parametrize(
    "materials",
    [
        "",
        "   \n\t ",
        # THE STATE THAT ACTUALLY OCCURS, and the one a blank-string test could
        # never reach: a full narrative judgment that names no provision at
        # all. 80 of the pilot's 508 groundings are like this (91 before the
        # materials-side widenings) and a blank one never occurred once, so
        # keying the skip on the source string rather than on the allow-list
        # tested a case that does not happen and missed the case that does.
        "The appellant, aged 54, was convicted by the Sessions Judge and the "
        "High Court dismissed the appeal on 12 March 2019 without reasons.",
    ],
)
def test_statutory_grounding_skips_when_the_allow_list_is_empty(materials):
    # An empty allow-list can only reject, never inform: every reference the
    # answer carries is ungrounded by construction, so the gate would be
    # scoring the shape of the grounding rather than the answer.
    result = check_statutory_grounding("Section 319 applies.", _stat_ctx(source_text=materials))
    assert result.passed
    assert result.detail == {"skipped": "no-material-references"}
    assert grounded_refs(materials) == set()


def test_statutory_grounding_still_scores_a_row_with_one_material_reference():
    # The skip is keyed on emptiness, not on scarcity - a single key is enough
    # to make the answer's references answerable. Pins that the widened skip
    # did not become "skip whenever the grounding is thin".
    ctx = _stat_ctx(source_text="The charge was laid under Section 319 of the Code.")
    assert check_statutory_grounding("Section 319 applies.", ctx).passed
    assert not check_statutory_grounding("Section 313 applies.", ctx).passed


def test_statutory_grounding_reads_enacted_headings_in_the_materials():
    """Bare-act text numbers its sections without ever writing "Section", and
    the pilot's gen 412 proved the cost: its grounding IS bare-act text, so a
    keyword-only scan found no sections there and read three correct citations
    as inventions. The marginal-note dash is what separates a heading from a
    judgment's numbered paragraph."""
    assert ("section", "397") in grounded_refs(STAT_SOURCE)
    assert check_statutory_grounding("Section 397 is attracted.", _stat_ctx()).passed
    # A numbered paragraph is not a section heading - no dash, no reference.
    paragraphs = "58. In our view the High Court erred in reversing the acquittal."
    assert ("section", "58") not in grounded_refs(paragraphs)


def test_statutory_grounding_grounds_the_interior_of_a_range():
    # "Sections 62 to 65" shows the teacher s.63 as surely as naming it does.
    assert ("section", "63") in grounded_refs(STAT_SOURCE)
    assert check_statutory_grounding("Section 63 is engaged.", _stat_ctx()).passed
    # Bounded, so a malformed range cannot spray the allow-list: the interior
    # of a span wider than RANGE_SPAN_MAX is not handed over, only its ends.
    wide = f"Sections 1 to {RANGE_SPAN_MAX + 200} of the Act."
    interior = str(RANGE_SPAN_MAX + 100)
    assert ("section", "1") in grounded_refs(wide)
    assert ("section", interior) not in grounded_refs(wide)


def test_statutory_grounding_drops_the_subsection_but_never_the_letter_suffix():
    """The materials mention only s.14(1); an answer entitled to cite s.14 is
    entitled to say which limb applies, and materials that print a subsection
    have plainly shown the section. The LETTER suffix is a different section
    (statutes.py: IPC 304B is not IPC 304) and is never folded away."""
    assert check_statutory_grounding("Section 14 applies.", _stat_ctx()).passed
    assert check_statutory_grounding("Section 14(3) applies.", _stat_ctx()).passed
    for suffixed in ("Section 14A", "Section 14-A"):
        result = check_statutory_grounding(f"{suffixed} applies.", _stat_ctx())
        assert not result.passed
        assert result.detail["ungrounded"][0]["number"] == "14A"
    # The identity survives the fold in the other direction too: materials
    # naming only the suffixed section do not ground the bare one.
    ctx = _stat_ctx(source_text="Section 14A of the Act was invoked.")
    assert not check_statutory_grounding("Section 14 applies.", ctx).passed


@pytest.mark.parametrize(
    "written",
    [
        # U+202F NARROW NO-BREAK SPACE, the separator gpt-oss-120b put in
        # all 434 references it wrote across the 46 judged answers. Python's
        # \s matches it, and _norm_ws folds it, so the detail reads plainly.
        "Section 319",
        "Section 319",
        "section 319",
        "Sec. 319",
        "S. 319",
        "§319",
        "§§ 319",
        "Sections 319 and 397",
        "Sections 319/397",
    ],
)
def test_statutory_grounding_reads_the_forms_the_generator_actually_writes(written):
    assert check_statutory_grounding(f"{written} of the Code applies.", _stat_ctx()).passed


def test_statutory_grounding_folds_the_dashes_a_generation_mixes():
    # Measured: the answer writes "Section 32-A" with U+2011 where the
    # materials print U+002D. Same section, and NFKC plus the dash class is
    # what makes them one key.
    ctx = _stat_ctx(source_text="Section 32-A of the Act was invoked.")
    assert check_statutory_grounding("Section 32‑A applies.", ctx).passed
    assert check_statutory_grounding("Section 32A applies.", ctx).passed
    # ... and 32 alone is a different section.
    assert not check_statutory_grounding("Section 32 applies.", ctx).passed


def test_statutory_grounding_never_splits_a_letter_suffix_off_at_the_dash():
    """The dash is a RANGE separator between digits and a suffix joiner before
    a letter. Without that fence "Section 120-B" splits into 120 and B, and
    IPC 120B would ground against materials that only ever named s.120."""
    assert [(f, n) for f, n, _ in statutory_refs("Section 120-B")] == [
        ("section", "120B")
    ]
    assert [(f, n) for f, n, _ in statutory_refs("Sections 62-65")] == [
        ("section", "62"),
        ("section", "65"),
    ]


def test_statutory_grounding_a_period_is_not_a_reference():
    # "an order 30 days later" is a period, not Order 30.
    assert statutory_refs("The court passed an order 30 days later.") == []
    assert check_statutory_grounding(
        "The court passed an order 30 days later.", _stat_ctx()
    ).passed


def test_statutory_grounding_an_initial_is_not_a_section():
    # "S." only marks a section when a NUMBER follows it, or every Indian
    # name beginning with an initial becomes a citation.
    assert statutory_refs("Per S. Vaidhyanathan J., the appeal fails.") == []


def test_statutory_grounding_families_are_the_five_the_defects_named():
    # Clause/Part/Paragraph/Schedule were measured and dropped: they are limbs
    # of a drafted instrument or paragraphs of the judgment in the materials,
    # and adding them false-failed an ACCEPTED row on "Paragraph 102".
    assert [family for family, _ in STATUTORY_FAMILIES] == [
        "section",
        "article",
        "entry",
        "order",
        "rule",
    ]
    assert statutory_refs("Clause 9(1) of this deed and Part II of Schedule 3.") == []


def test_statutory_grounding_reports_every_ungrounded_reference_once():
    result = check_statutory_grounding(
        "Order 12 Rule 6 is invoked, and Rule 6 again, and Article 136.",
        _stat_ctx(),
    )
    assert not result.passed
    assert [(u["family"], u["number"]) for u in result.detail["ungrounded"]] == [
        ("article", "136"),
        ("order", "12"),
        ("rule", "6"),
    ]
    assert result.detail["ungrounded_count"] == 3
    assert result.detail["tolerated"] == MAX_UNGROUNDED_REFS == 0


def test_statutory_grounding_detail_survives_json():
    result = check_statutory_grounding("Section 313 applies.", _stat_ctx())
    assert json.loads(json.dumps(result.detail)) == result.detail


def test_statutory_grounding_never_raises_on_empty_or_junk():
    assert check_statutory_grounding("", _stat_ctx()).passed
    assert check_statutory_grounding(None, _stat_ctx()).passed
    assert statutory_refs("") == []
    assert grounded_refs("") == set()


# --------------------------------------------------------------------------
# check_statutory_grounding - the 508-row review round.
#
# Every test below rejects a misreading that a full-population sweep caught the
# first version making. The gen ids are the rows that proved it.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("materials", "answer", "gens"),
    [
        # u/s: 25 of 508 groundings, 5 false fails.
        ("The accused was convicted u/s 302 and 201 IPC by the Sessions Judge.",
         "Section 302 IPC and s. 201 IPC are made out.", "120/133/445/449/458"),
        # the "No." infix.
        ("When Act No.32 of 1994 amended Schedule III, Exemption Entry No.8 "
         "did not include the word 'like'.",
         "Only the nouns in the brackets of Entry 8 are exempt.", "227"),
        # single-letter R. / O.: 14 of 508 groundings.
        ("The application under section 26-A of the Act read with R. 2 was "
         "refused, and O. 43 was not invoked.",
         "Section 26-A and Rule 2 require every partner to sign. Order 43 too.",
         "263/267/275"),
    ],
)
def test_statutory_grounding_reads_the_spellings_the_materials_use(materials, answer, gens):
    """The materials vocabulary is WIDER than the answer vocabulary, and every
    widening is one-directional: it can only add an allow-list key, so it can
    only turn a rejection into a pass. Each row here is a generation that
    repeated its materials correctly and was rejected for it."""
    result = check_statutory_grounding(answer, _stat_ctx(source_text=materials))
    assert result.passed, (gens, result.detail["ungrounded"])


def test_statutory_grounding_the_wider_vocabulary_is_materials_only():
    # The single-letter and "No." forms are deliberately NOT read in an answer:
    # they are shapes this generator has never written, and admitting them
    # there would risk reading an initial as a rule for no measured gain. The
    # cost is recorded on statutory_refs and pinned here.
    assert statutory_refs("under O. 12 R. 6 CPC") == []
    assert statutory_refs("convicted u/s 313 of the Code") == []
    assert statutory_refs("see Entry No. 8 of the Schedule") == []
    # Each answer-side blank needs its materials-side counterpart asserted
    # HERE, or the widening is unpinned: a mutant deleting the extension makes
    # the fixture's allow-list empty, the gate takes the no-material-references
    # skip, and a result.passed assertion elsewhere stays green for a reason
    # that has nothing to do with the extension. Measured - the
    # _REF_NUMBER_INFIX mutant survived the whole suite until this line.
    assert ("section", "313") in grounded_refs("convicted u/s 313 of the Code")
    assert ("entry", "8") in grounded_refs("see Entry No. 8 of the Schedule")
    assert ("rule", "6") in grounded_refs("under O. 12 R. 6 CPC")
    assert ("order", "12") in grounded_refs("under O. 12 R. 6 CPC")


def test_statutory_grounding_the_us_form_needs_its_slash():
    """"before us." is the pronoun, and the materials gap tolerates a wrap, so
    a slashless alternative read "the revenue is in appeal before us.\\n\\n6. A
    perusal..." as section 6 - a key the materials never granted, on the side
    where a bogus key is a silent false PASS. 23 of 508 groundings carry the
    phrase; 9 had a number close enough behind it to be swallowed."""
    assert grounded_refs("the revenue is in appeal before us.\n\n6. A perusal of the notification") == set()
    assert ("section", "302") in grounded_refs("convicted u/s 302 IPC")
    assert ("section", "302") in grounded_refs("convicted u / s 302 IPC")
    assert ("section", "5") in grounded_refs("charged u/ss 5 and 6")


def test_statutory_grounding_folds_the_pdf_soft_hyphen_artifact():
    """The extractor writes U+00AC where the PDF had a soft hyphen, so gens
    419/423/430 are grounded in "Section 260¬A" while the answer writes
    "Section 260-A". Both are s.260A, and 260 alone is a different section."""
    ctx = _stat_ctx(source_text="The appellant filed an appeal under Section 260¬A of the Act.")
    assert check_statutory_grounding("Section 260-A is the route.", ctx).passed
    assert check_statutory_grounding("Section 260A is the route.", ctx).passed
    assert not check_statutory_grounding("Section 260 is the route.", ctx).passed


def test_statutory_grounding_keyword_and_number_must_share_a_line():
    """A markdown heading followed by a numbered list is not a citation.
    "**Rule**\\n1. Section 15(1) of the Act provides" fired on five clean rows
    (127, 265, 299, 306, 383) because \\s spans newlines.

    The genuine catch is written inline and survives - that is the whole point
    of the fence, so both halves are asserted here."""
    assert statutory_refs("**Rule**\n1. Section 15(1) of the Act provides") == [
        ("section", "15", "Section 15(1)")
    ]
    assert statutory_refs("APPLICATION TO RESTORE TRIBUNAL ORDER\n\n1.  The Applicant") == []
    assert [(f, n) for f, n, _ in statutory_refs("Under Order XII, Rule 6 of the CPC")] == [
        ("rule", "6")
    ]


def test_statutory_grounding_materials_may_wrap_where_an_answer_may_not():
    """The asymmetry is the cost function. Grounding text comes out of a PDF
    and wraps mid-citation, and on that side a missed spelling is a FALSE FAIL;
    on the answer side it is only a miss."""
    assert ("section", "12A") in grounded_refs("was invoked under Section\n12-A of the Act")
    assert statutory_refs("was invoked under Section\n12-A of the Act") == []


def test_statutory_grounding_a_date_is_not_a_reference():
    """Gen 39 wrote "interim status-quo order 1 June 1990" and gens 410/415
    wrote "Section 6A - 01 January 1966"; the first reads as Order 1 and the
    second lets the dash split "6A - 01" into a range. A month name after the
    number settles both."""
    assert statutory_refs("an interim status-quo order 1 June 1990 was passed") == []
    assert [(f, n) for f, n, _ in statutory_refs("the dates in Section 6A - 01 January 1966")] == [
        ("section", "6A")
    ]
    # ... and the guard is about the MONTH, not about the digits: a real
    # reference followed by ordinary prose is untouched.
    assert [(f, n) for f, n, _ in statutory_refs("Section 6A applies")] == [("section", "6A")]


def test_statutory_grounding_a_subdivision_is_not_the_provision_it_divides():
    """"sub-rule 2" is not Rule 2 and "sub-section 2(a)" is not Section 2. Gen
    90 cites "sub-rule 2 of Rule 7" against materials reading "sub-rule (2) of
    Rule 7", and gen 372 cites "sub-section 2 (a)" against "clause (a) of
    sub-section (2)" - both the answer repeating its materials exactly.

    THIS IS WHAT KEPT THE ORDER AND RULE FAMILIES. Gen 90's was the only
    rule/order false fire to survive the same-line fence, and its twin is in
    the SECTION family, which cannot be dropped - so dropping order and rule
    would have left the defect standing and taken two judged catches with it.
    """
    assert [(f, n) for f, n, _ in statutory_refs("under sub-rule 2 of Rule 7")] == [
        ("rule", "7")
    ]
    assert [(f, n) for f, n, _ in statutory_refs("under sub‑section 2 (a) and sub-section 3")] == []
    assert statutory_refs("subsection 2 of the Act") == []
    # The un-prefixed reference in the same sentence is untouched.
    assert [(f, n) for f, n, _ in statutory_refs("Section 20 and sub-section 2")] == [
        ("section", "20")
    ]


def test_statutory_grounding_or_is_a_list_separator():
    """Materials reading "not by way of writ under Article 226 or 32" name
    BOTH articles, and gen 424's answer cites "Articles 226/32". Without "or"
    in the separator the 32 read as an invention."""
    materials = "relief lies under Article 227 but not by way of writ under Article 226 or 32."
    assert ("article", "32") in grounded_refs(materials)
    assert ("article", "226") in grounded_refs(materials)
    assert check_statutory_grounding("Articles 226/32 are unavailable.", _stat_ctx(
        source_text=materials)).passed


def test_statutory_grounding_enacted_headings_ground_three_families():
    """Enacted text numbers rules and orders exactly as it numbers sections and
    the layout does not say which. Gen 280's grounding prints the assembly's
    RULES as "57. Suspension of rules.-", so emitting the number under
    `section` alone left the answer's "Rule 57" ungrounded."""
    keys = grounded_refs("x\n57. Suspension of rules.— Any member may move that")
    assert ("rule", "57") in keys
    assert ("section", "57") in keys
    assert ("order", "57") in keys
    # ... and nothing else: the channel does not invent an article or an entry.
    assert ("article", "57") not in keys
    assert ("entry", "57") not in keys


def test_statutory_grounding_enacted_headings_tolerate_a_footnote_number():
    """The extractor leaves the footnote marker welded to the heading: gen
    280's grounding reads "16 57. Suspension of rules.-", where 16 is the
    footnote and 57 is the rule the answer cites."""
    keys = grounded_refs("x\n16 57. Suspension of rules.— Any member may move that")
    assert ("rule", "57") in keys
    assert check_statutory_grounding("Rule 57 lets the Speaker suspend it.", _stat_ctx(
        source_text="x\n16 57. Suspension of rules.— Any member may move that")).passed


@pytest.mark.parametrize(
    "materials",
    [
        # NO MARGINAL-NOTE DASH. A judgment's numbered paragraph must not
        # register, or every paragraph number in the materials grounds a
        # provision. This is the mutant that survived the first round: an
        # ASCII hyphen is NOT the dash, and nothing pinned it.
        "x\n58. In our view the High Court erred in reversing the acquittal.",
        "x\n58. Suspension of rules - any member may move that a rule be suspended.",
        "x\n58. Suspension of rules -- any member may move that a rule be suspended.",
    ],
)
def test_statutory_grounding_a_numbered_paragraph_is_not_an_enacted_heading(materials):
    assert ("section", "58") not in grounded_refs(materials)
    assert ("rule", "58") not in grounded_refs(materials)


def test_statutory_grounding_an_enacted_heading_must_open_its_line():
    """The ^ anchor is what separates a heading from a sentence that happens to
    contain a number, a dot and a dash. Without it "...decided in 1998. Suspension
    of rules - see..." would ground a provision out of running prose."""
    inline = "The question was decided long ago. 58. Suspension of rules.— see below."
    assert ("section", "58") in grounded_refs("x\n58. Suspension of rules.— see below.")
    assert ("section", "58") not in grounded_refs(inline)


def test_statutory_grounding_failure_is_a_regenerate_not_a_burnt_seed():
    results = [
        GateResult(gate, gate != "statutory_grounding", {}) for gate in GATE_ORDER
    ]
    assert disposition(results) == "regenerate"


# --------------------------------------------------------------------------
# check_temporal
# --------------------------------------------------------------------------

def test_temporal_new_code_for_a_pre_transition_offence_fails():
    result = check_temporal("The accused is liable under Section 103 BNS.", _ctx())
    assert not result.passed
    assert result.detail["flags"][0]["flag"] == "bns-cited-for-old-offence"
    assert result.detail["flags"][0]["ref"] == "BNS 103"
    assert result.detail["flags"][0]["code"] == "BNS"
    assert disposition([result]) == "reject"


def test_temporal_savings_clause_reference_rescues_the_same_text():
    text = (
        "Section 103 BNS replaced the old provision, but Section 358 BNS "
        "preserves the liability, so the charge remains under the repealed code."
    )
    assert check_temporal(text, _ctx()).passed


def test_temporal_old_code_for_a_pre_transition_offence_passes():
    result = check_temporal("The charge is under Section 302 IPC.", _ctx())
    assert result.passed
    assert result.detail["flags"] == []
    assert result.detail["undecidable"] == []


def test_temporal_old_code_for_a_post_transition_offence_fails():
    result = check_temporal("The charge is under Section 302 IPC.", _ctx(offence_date=AFTER))
    assert not result.passed
    assert result.detail["flags"][0]["flag"] == "ipc-cited-for-new-offence"


def test_temporal_split_families_in_one_sentence_pass():
    # 2023 offence, FIR registered after the appointed day: IPC charge, BNSS
    # procedure, both correct at once.
    ctx = _ctx(offence_date=BEFORE, proceeding_started=AFTER)
    text = "Charged under Section 302 IPC and investigated under Section 173 BNSS."
    assert check_temporal(text, ctx).passed


def test_temporal_undecidable_fails_on_the_transition_stream():
    ctx = _ctx(stream="transition", offence_date=None, proceeding_started=None)
    result = check_temporal("The charge is under Section 302 IPC.", ctx)
    assert not result.passed
    assert result.detail["undecidable"] == ["IPC 302"]
    assert result.detail["undecidable_fatal"] is True
    assert disposition([result]) == "reject"


def test_temporal_undecidable_passes_elsewhere_but_is_recorded():
    ctx = _ctx(offence_date=None, proceeding_started=None)
    result = check_temporal("The charge is under Section 302 IPC.", ctx)
    assert result.passed
    assert result.detail["undecidable"] == ["IPC 302"]
    assert result.detail["undecidable_fatal"] is False
    assert result.detail["stream"] == "synthesis"


def test_temporal_detail_is_json_safe_even_with_codeflag_and_sectionref():
    result = check_temporal("Liable under Section 103 BNS.", _ctx())
    assert json.loads(json.dumps(result.detail)) == result.detail


def test_temporal_empty_content_never_raises():
    assert check_temporal("", _ctx()).passed


# --------------------------------------------------------------------------
# check_self_verification
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cue", VERIFICATION_CUES)
def test_self_verification_every_cue_is_detected(cue):
    result = check_self_verification(f"Some reasoning. {cue} the section number.", _ctx())
    assert result.passed
    assert cue in result.detail["cues"]


def test_self_verification_is_case_insensitive_and_survives_line_breaks():
    result = check_self_verification("Wait,\n     the date is wrong.", _ctx())
    assert result.passed
    assert result.detail["cues"] == ["wait,"]


def test_self_verification_without_a_cue_fails():
    result = check_self_verification("A confident march through the elements.", _ctx())
    assert not result.passed
    assert result.detail["cues"] == []


def test_self_verification_skipped_when_no_reasoning_expected():
    result = check_self_verification("", _ctx(expect_reasoning=False))
    assert result.passed
    assert result.detail == {"skipped": "no-reasoning-expected"}


def test_self_verification_none_trace_fails_without_raising():
    assert not check_self_verification(None, _ctx()).passed


# --------------------------------------------------------------------------
# check_irac_placement
# --------------------------------------------------------------------------

def test_irac_headings_tolerates_markdown_forms():
    text = "## Issue\nfoo\n**Rule:** bar\n1. Application -\nbaz\n### Conclusions\n"
    assert irac_headings(text) == {"issue", "rule", "application", "conclusion"}


def test_irac_headings_ignores_prose_that_merely_starts_with_the_word():
    assert irac_headings("Issues of fact remain open here.\nRule of thumb applies.") == set()


def test_irac_placement_full_four_in_the_answer_passes():
    result = check_irac_placement("clean trace", GOOD_ANSWER, _ctx())
    assert result.passed
    assert result.detail["answer_headings"] == ["application", "conclusion", "issue", "rule"]
    assert result.detail["think_headings"] == []


def test_irac_placement_issue_plus_conclusion_is_enough():
    answer = "Issue: whether the appeal lies.\n\nConclusion: it does not."
    assert check_irac_placement("clean trace", answer, _ctx()).passed
    assert tuple(IRAC_REQUIRED) == ("issue", "conclusion")


def test_irac_placement_missing_conclusion_fails():
    answer = "Issue: whether the appeal lies.\n\nRule: the limitation period is 90 days."
    result = check_irac_placement("clean trace", answer, _ctx())
    assert not result.passed
    assert result.detail["missing_in_answer"] == ["conclusion"]


def test_irac_placement_heading_inside_the_trace_fails_even_with_a_good_answer():
    think = "First pass.\n\nIssue: whether the appeal lies.\n\nMore thinking."
    result = check_irac_placement(think, GOOD_ANSWER, _ctx())
    assert not result.passed
    assert result.detail["think_headings"] == ["issue"]
    assert disposition([result]) == "regenerate"


def test_irac_headings_accept_a_full_stop_terminator():
    assert irac_headings("**Conclusion.**\nThe appeal fails.") == {"conclusion"}
    # Same terminator on the tripwire side: a scripted trace cannot dodge the
    # gate by punctuating its headings with a full stop.
    result = check_irac_placement("Rule.\nThe test is settled.", GOOD_ANSWER, _ctx())
    assert not result.passed
    assert result.detail["think_headings"] == ["rule"]


def test_irac_placement_skipped_on_an_unparsed_row():
    # think is None => split_think failed => `answer` is the whole generation,
    # so neither half of this gate is measuring what it claims to.
    result = check_irac_placement(None, "no headings anywhere", _ctx())
    assert result.passed
    assert result.detail == {"skipped": "unparsed-format"}


def test_irac_placement_skipped_on_the_replay_stream():
    result = check_irac_placement("Issue: anything", "no headings", _ctx(stream="replay"))
    assert result.passed
    assert result.detail == {"skipped": "stream-replay"}


def test_irac_placement_skipped_when_no_reasoning_expected():
    result = check_irac_placement(None, "no headings", _ctx(expect_reasoning=False))
    assert result.passed
    assert result.detail == {"skipped": "no-reasoning-expected"}


# A deliverable-shaped answer with no line-initial IRAC headings. Long enough
# that a caller using the live length band still clears answer_min.
NO_IRAC_ANSWER = (
    "The petition should be settled as a revision. The facts as recorded do not "
    "make out the charge, and the delay in the deposition is unexplained. The "
    "client should be advised to press that gap rather than to negotiate around "
    "it. Service, limitation and the verification clause must all be attended to "
    "in the instrument itself, not left as an afterthought in the reasoning."
)


def test_drafting_answer_without_irac_headings_passes_placement():
    result = check_irac_placement("clean trace", NO_IRAC_ANSWER, _ctx(task_type="drafting"))
    assert result.passed
    assert result.detail["missing_in_answer"] == []
    assert result.detail["required"] == []


def test_summarization_answer_without_irac_headings_passes_placement():
    result = check_irac_placement(
        "clean trace", NO_IRAC_ANSWER, _ctx(task_type="summarization")
    )
    assert result.passed
    assert result.detail["missing_in_answer"] == []
    assert result.detail["required"] == []


@pytest.mark.parametrize(
    "task_type",
    ("irac_analysis", "statute_qa", "transition", "drafting", "summarization"),
)
def test_think_side_irac_still_fails_for_every_task_type(task_type):
    think = "First pass.\n\nIssue: whether the appeal lies.\n\nMore thinking."
    result = check_irac_placement(think, GOOD_ANSWER, _ctx(task_type=task_type))
    assert not result.passed
    assert result.detail["think_headings"] == ["issue"]
    assert disposition([result]) == "regenerate"


@pytest.mark.parametrize("task_type", ("irac_analysis", "statute_qa", "transition"))
def test_analysis_statute_and_transition_answers_still_require_irac(task_type):
    result = check_irac_placement("clean trace", NO_IRAC_ANSWER, _ctx(task_type=task_type))
    assert not result.passed
    assert result.detail["missing_in_answer"] == ["issue", "conclusion"]
    assert result.detail["required"] == ["issue", "conclusion"]
    assert task_type in IRAC_ANSWER_TASK_TYPES


# --------------------------------------------------------------------------
# prompt_echo — conservative instruction-packet restatement on stored think.
# --------------------------------------------------------------------------

_ECHO_OPENING = (
    "We need to produce a legal analysis in first person with 450-700 words "
    "of deliberation before writing the answer."
)
_ECHO_SPAN = (
    "The facts are thin. Never write as though the matter had been handed to "
    "you as a text, and keep quotation for the answer."
)
_NORMAL_THINK = (
    "I start from the facts. The complaint alleges theft and the recovery is "
    "said to complete the chain. Let me check the dates against the "
    "charge-sheet, because a chronology settled quickly is a chronology "
    "settled badly. The source of the obligation is the provision both "
    "sides invoke, not a passage handed to me as homework."
)


def test_known_instruction_restatement_fails_prompt_echo():
    assert "prompt_echo" in GATE_ORDER
    for think in (_ECHO_OPENING, _ECHO_SPAN):
        results = run_all(_content(think, GOOD_ANSWER), 200, _ctx())
        echo = next(result for result in results if result.gate == "prompt_echo")
        assert not echo.passed, think
        assert disposition([echo]) == "regenerate"


def test_ordinary_legal_reasoning_does_not_fail_prompt_echo():
    results = run_all(_content(_NORMAL_THINK, GOOD_ANSWER), 200, _ctx())
    echo = next(result for result in results if result.gate == "prompt_echo")
    assert echo.passed
    assert echo.detail.get("skipped") is None


_DRAFTING_PROCESS = (
    "I need to write the plaint from the pleaded facts, naming the parties "
    "and the relief the papers actually support."
)
_INSTRUCTION_RESTATE = (
    "We need to produce a response/reasoning that follows the packet before "
    "turning to the pleaded facts."
)


def test_first_person_drafting_process_does_not_fail_prompt_echo():
    results = run_all(_content(_DRAFTING_PROCESS, GOOD_ANSWER), 200, _ctx())
    echo = next(result for result in results if result.gate == "prompt_echo")
    assert echo.passed
    assert echo.detail.get("restated_opening") is False


def test_we_need_to_produce_response_reasoning_fails_prompt_echo():
    results = run_all(_content(_INSTRUCTION_RESTATE, GOOD_ANSWER), 200, _ctx())
    echo = next(result for result in results if result.gate == "prompt_echo")
    assert not echo.passed
    assert echo.detail["restated_opening"] is True
    assert disposition([echo]) == "regenerate"


# --------------------------------------------------------------------------
# check_statutory_quotation
# --------------------------------------------------------------------------

# What a transition prompt actually shows the teacher: the provision's
# identity, its marginal note, and the effect this build RECORDS for it,
# labelled as not a quotation. There is no bare-act corpus behind it.
RECORDED_EFFECT = (
    "Section 358 of the Bharatiya Nyaya Sanhita, 2023 (Repeal and savings).\n"
    "Operative effect as recorded in this build's statute table (not a quotation): "
    "The repeal of the Indian Penal Code, 1860 does not affect any right, privilege, "
    "obligation or liability acquired, accrued or incurred under it."
)


def _quotation_ctx(**over):
    over.setdefault("source_text", RECORDED_EFFECT)
    return _ctx(stream="transition", **over)


def test_statutory_quotation_the_paraphrase_quoted_as_the_sections_words_fails():
    """The artefact the whole "not a quotation" label exists to prevent.

    Measured before this gate: this exact answer passed all nine gates and
    entered the dataset presenting the build's own paraphrase as the enacted
    words of s.358(2).
    """
    answer = (
        'Section 358(2) of the Bharatiya Nyaya Sanhita, 2023 provides: "The repeal of '
        'the Indian Penal Code, 1860 does not affect any right, privilege, obligation '
        'or liability acquired, accrued or incurred under it". The charge therefore '
        "stands under the old Code."
    )
    result = check_statutory_quotation(answer, _quotation_ctx())
    assert not result.passed
    hit = result.detail["quotations"][0]
    assert "provides" in hit["attribution"]
    # Recorded, not part of the verdict: these words WERE the build's own.
    assert hit["reproduces_grounding"] is True
    assert disposition([result]) == "regenerate"


def test_statutory_quotation_an_unquoted_faithful_restatement_passes():
    """The other direction, and the one that matters for yield: saying the
    same thing in the answer's own words is exactly what the prompt asks
    for."""
    answer = (
        "Section 358(2) of the Bharatiya Nyaya Sanhita preserves rights, obligations "
        "and liabilities already accrued under the repealed Indian Penal Code, so the "
        "charge stands under the old Code however long afterwards it is taken up."
    )
    result = check_statutory_quotation(answer, _quotation_ctx())
    assert result.passed
    assert result.detail["quotations"] == []


def test_statutory_quotation_invented_words_are_refused_too():
    # Nothing in this repository could check a quotation against the Act, so
    # words from nowhere are refused on the same footing as the paraphrase.
    answer = 'Section 531 BNSS reads: "every proceeding shall abate upon commencement".'
    result = check_statutory_quotation(answer, _quotation_ctx())
    assert not result.passed
    assert result.detail["quotations"][0]["reproduces_grounding"] is False


@pytest.mark.parametrize(
    "lead",
    [
        'Section 358(2) of the BNS provides: ',
        'Section 358 BNS states: ',
        'Section 358 BNS reads, in terms, ',
        'The provision is in these words: ',
        'The Sanhita says: ',
        'That sub-section runs ',
    ],
)
def test_statutory_quotation_reads_the_attribution_shapes(lead):
    answer = lead + '"the repeal does not affect any liability already incurred".'
    assert not check_statutory_quotation(answer, _quotation_ctx()).passed


def test_statutory_quotation_a_quote_attributed_to_someone_else_passes():
    """The sentence break is the discriminator, and it is doing work: a
    quotation attributed to a witness is not a statement about what a section
    says, and rejecting it would be a gate firing on the wrong artefact."""
    answer = (
        'Section 302 IPC governs the charge. The informant said "he struck him twice '
        'with the rod" and the witness confirmed it.'
    )
    assert check_statutory_quotation(answer, _quotation_ctx()).passed


def test_statutory_quotation_a_quotation_attributed_to_nothing_passes():
    """No section named anywhere before the quote, so nothing is being passed
    off as a section's words - and the gate must neither fail it nor raise.

    Found by mutation: removing the "is a section named at all?" guard left
    every assertion above green, because every other fixture here names one.
    The gate does not merely mis-fire without the guard - `max()` over an
    empty sequence raises, and a gate that raises takes the whole run with it.
    """
    answer = (
        'The papers record that the accused "stands charged and has pleaded not guilty", '
        "and nothing in this point turns on how that was put."
    )
    result = check_statutory_quotation(answer, _quotation_ctx())
    assert result.passed
    assert result.detail["quotations"] == []


def test_statutory_quotation_a_short_quoted_phrase_is_not_an_attribution():
    # A word or two in quotes is a term of art being named, not a passage
    # being passed off as enacted text.
    answer = 'Section 358 BNS turns on whether a liability was "incurred" before the day.'
    assert check_statutory_quotation(answer, _quotation_ctx()).passed


def test_statutory_quotation_skipped_off_the_transition_stream():
    """Every other stream is grounded in judgment text, where quoting a
    holding in the answer is legitimate and useful."""
    answer = 'The Court held, in Anwar Ali: "the chain of circumstances must be complete".'
    result = check_statutory_quotation(answer, _ctx(stream="synthesis"))
    assert result.passed
    assert result.detail == {"skipped": "not-transition"}


def test_statutory_quotation_catches_the_trace_too():
    # Same reasoning as check_temporal: a fabricated statutory quotation is no
    # better inside the reasoning, and the caution now forbids both.
    content = _content(
        'let me check - section 358 BNS provides: "the repeal does not affect any '
        'liability already incurred" - so the old Code stands.',
        "Conclusion: the old Code governs the charge.",
    )
    assert not check_statutory_quotation(content, _quotation_ctx()).passed


def test_statutory_quotation_survives_reflowed_whitespace():
    answer = (
        'Section 358(2)\n   of the BNS\n   provides:\n   "the repeal does not affect '
        'any liability already incurred".'
    )
    assert not check_statutory_quotation(answer, _quotation_ctx()).passed


def test_statutory_quotation_reads_curly_quotation_marks():
    answer = (
        "Section 358(2) of the BNS provides: “the repeal does not affect any "
        "liability already incurred”."
    )
    assert not check_statutory_quotation(answer, _quotation_ctx()).passed


# --------------------------------------------------------------------------
# check_verbatim_overlap / find_verbatim_run
# --------------------------------------------------------------------------

def test_verbatim_overlap_naming_the_case_and_the_statute_is_not_transcription():
    """THE RE-AUDIT OF DEFAULT_MAX_RUN, 30 -> 120 on 2026-08-18.

    This test used to assert the opposite - that a 35-character shared run
    FAILS - and that threshold made this the pilot's worst gate: 151/221
    generations (68%), rising to 58/58 on third attempts. 30 characters is five
    or six words of Indian legal English, so what it actually matched was the
    names of things. All 120 distinct matched runs were of this shape:
    ' High Court of Madhya Pradesh ', ' Central Excise and Salt Act, ',
    ' S. Govinda Menon v. Union of ', ' Section 22 Hindu Succession A'. A trace
    cannot reason about a case without naming it.

    Measured over the 55 pilot traces whose grounding could be recovered, the
    longest run shared with the source ran p25 34, p50 54, p75 76, p90 130,
    max 335 characters, and the failure rate by threshold went 30:82%, 40:65%,
    50:55%, 60:42%, 80:24%, 100:16%, 120:13%, 150:11%. 120 clears the median
    incidental overlap by better than 2x while still catching the six genuine
    copies, which measured 167-335 characters.

    RAISED AGAIN 120 -> 500 on 2026-08-28, by the same method on the current
    generator. 120 was fitted on gpt-oss pilot traces; bai/deepseek-v4-flash
    (the SOLE routing.generator since 2026-08-28) quotes the sentence under
    analysis inside otherwise self-authored deliberation, and its longest
    shared run measures p50 127 over 1,086 stored generations - the old
    threshold sat AT the median incidental overlap, the exact position the
    30 -> 120 re-audit condemned. The deepseek failure-rate curve ran 120:52%,
    200:29%, 300:15%, 400:7%, 500:4%, 600:3%, flattening at 500; the failing
    traces' copied coverage was p50 2.1% of the trace (max 19%), i.e. quotes,
    not the transcription this gate exists to catch. Evidence beside
    DEFAULT_MAX_RUN in gates.py and in
    prev_rep.md 2.5 (verbatim-overlap-drafting-drift).
    """
    think = f"Working through it: {SOURCE_RUN_35} and so the link holds."
    result = check_verbatim_overlap(think, _ctx())
    assert result.passed
    assert result.detail["match"] is None


def test_verbatim_overlap_a_run_at_the_threshold_is_still_transcription():
    think = f"Working through it: {SOURCE_RUN_LONG} and so the link holds."
    result = check_verbatim_overlap(think, _ctx())
    assert not result.passed
    assert result.detail["match"] in _norm(SOURCE)
    assert len(result.detail["match"]) <= 80
    assert disposition([result]) == "regenerate"


def test_verbatim_overlap_threshold_is_pinned_to_the_measured_sweep():
    """A bare pin, so moving the number is a deliberate act with a re-audit
    attached rather than a tuning nudge. The evidence is in gates.py beside the
    constant and in the test above."""
    assert DEFAULT_MAX_RUN == 500


def test_verbatim_overlap_same_run_in_the_answer_only_passes():
    # The gate never sees the answer: quoting a holding there is legitimate.
    result = check_verbatim_overlap("a paraphrased trace of my own words", _ctx())
    assert result.passed
    content = _content("a paraphrased trace of my own words", SOURCE_RUN_LONG)
    assert check_verbatim_overlap(split_think(content, *_tags())[0], _ctx()).passed


def test_verbatim_overlap_survives_reflowed_whitespace():
    reflowed = SOURCE_RUN_LONG.replace(" ", "\n   ")
    assert not check_verbatim_overlap(f"prelude {reflowed} coda", _ctx()).passed


def test_verbatim_overlap_clean_paraphrase_passes():
    think = "The inference chain here is thin and the seizure was never proved."
    result = check_verbatim_overlap(think, _ctx())
    assert result.passed
    assert result.detail["match"] is None
    assert result.detail["max_run"] == DEFAULT_MAX_RUN


def test_verbatim_overlap_short_or_missing_trace_passes():
    assert check_verbatim_overlap(None, _ctx()).passed
    assert check_verbatim_overlap("tiny", _ctx()).passed
    assert check_verbatim_overlap(_norm(SOURCE), _ctx(source_text="")).passed


# Filler for the stride tests: long enough to slice a run of any threshold
# this gate is likely to carry, and sharing nothing with SOURCE.
_ALIGNMENT_TEXT = (
    "the accused walked to Fort Kochi that morning and waited by the ferry "
    "until the tide turned, saying nothing to anyone who passed him there. "
) * 5


@pytest.mark.parametrize("offset", range(SHINGLE_STEP))
def test_find_verbatim_run_catches_an_exactly_max_run_copy_at_every_alignment(offset):
    # The shingle stride must not depend on where the shared run happens to
    # sit: a run of EXACTLY max_run chars is the tightest case, and step-10
    # anchoring of max_run-length shingles would miss most of these offsets.
    # Derived from the threshold rather than typed out, so raising
    # DEFAULT_MAX_RUN cannot silently stop testing the tightest case.
    run = _ALIGNMENT_TEXT[:DEFAULT_MAX_RUN]
    assert len(run) == DEFAULT_MAX_RUN
    source = "q" * offset + run + " padding that shares nothing else at all"
    text = "unrelated opening words here " + run + " unrelated closing words"
    assert find_verbatim_run(text, source, DEFAULT_MAX_RUN) == run


def test_find_verbatim_run_ignores_a_shorter_shared_run():
    run = _ALIGNMENT_TEXT[: DEFAULT_MAX_RUN - 1]
    assert len(run) == DEFAULT_MAX_RUN - 1
    source = "zzz" + run + "!!! nothing else in common"
    text = "yyy" + run + "??? nothing else in common at all"
    assert find_verbatim_run(text, source, DEFAULT_MAX_RUN) is None


def test_find_verbatim_run_edge_inputs():
    assert find_verbatim_run("", SOURCE, 30) is None
    assert find_verbatim_run(SOURCE, "", 30) is None
    assert find_verbatim_run(SOURCE, SOURCE, 0) is None
    assert find_verbatim_run("abcdef", "xxabcdefxx", 6) == "abcdef"


def _naive_verbatim_run(text: str, source: str, max_run: int) -> str | None:
    """Reference implementation: every window, no shingles, no stride."""
    for i in range(0, len(text) - max_run + 1):
        window = text[i : i + max_run]
        if window in source:
            return window
    return None


def test_find_verbatim_run_agrees_with_a_naive_scan_on_random_strings():
    # The stride is the one place this module could silently under-report, so
    # it is checked against the O(n*m) reference over random strings - half of
    # them carrying a deliberately spliced shared run of 25-35 chars at an
    # arbitrary alignment.
    rng = random.Random(20260812)
    for _ in range(200):
        source = "".join(rng.choice("abc ") for _ in range(150))
        text = "".join(rng.choice("abc ") for _ in range(150))
        if rng.random() < 0.5:
            length = rng.randrange(25, 36)
            start = rng.randrange(0, len(source) - length)
            at = rng.randrange(0, len(text) - length)
            text = text[:at] + source[start : start + length] + text[at + length :]
        for max_run in (12, 25, 30, 35):
            found = find_verbatim_run(text, source, max_run)
            expected = _naive_verbatim_run(text, source, max_run)
            assert (found is None) == (expected is None), (max_run, source, text)
            if found is not None:
                assert len(found) == max_run
                assert found in source and found in text


def test_verbatim_overlap_custom_max_run():
    think = "shares the chain of circumstances phrase"
    assert not check_verbatim_overlap(think, _ctx(), max_run=12).passed
    assert check_verbatim_overlap(think, _ctx(), max_run=120).passed


# --------------------------------------------------------------------------
# check_banned_meta
# --------------------------------------------------------------------------

@pytest.mark.parametrize("phrase", BANNED_META)
def test_banned_meta_every_phrase_fails(phrase):
    result = check_banned_meta(f"Reasoning onward, {phrase} otherwise.", _ctx())
    assert not result.passed
    assert phrase in result.detail["hits"]
    assert disposition([result]) == "regenerate"


def test_banned_meta_is_case_insensitive_and_survives_line_breaks():
    result = check_banned_meta("The Provided\n    Text says otherwise.", _ctx())
    assert not result.passed
    assert result.detail["hits"] == ["the provided text"]


def test_banned_meta_clean_trace_passes():
    result = check_banned_meta(GOOD_THINK, _ctx())
    assert result.passed
    assert result.detail == {"hits": []}


def test_banned_meta_skipped_when_no_reasoning_expected():
    result = check_banned_meta("the provided text", _ctx(expect_reasoning=False))
    assert result.passed
    assert result.detail == {"skipped": "no-reasoning-expected"}


def test_banned_meta_none_trace_passes():
    assert check_banned_meta(None, _ctx()).passed


# --------------------------------------------------------------------------
# check_answer_key
# --------------------------------------------------------------------------

def _key(**over):
    key = {
        "governing_family": "old",
        "expected_sections": [{"code": "IPC", "number": "302"}],
        "forbidden_sections": [{"code": "BNS", "number": "103"}],
        "requires_savings_mention": False,
        "must_name_both_families": False,
    }
    key.update(over)
    return key


def _transition_ctx(**over):
    over.setdefault("answer_key", _key())
    return _ctx(stream="transition", **over)


def test_answer_key_skipped_off_the_transition_stream():
    result = check_answer_key("anything", _ctx(answer_key=_key()))
    assert result.passed
    assert result.detail == {"skipped": "not-transition"}


def test_answer_key_skipped_when_the_transition_row_carries_no_key():
    result = check_answer_key("anything", _ctx(stream="transition"))
    assert result.passed
    assert result.detail == {"skipped": "no-answer-key"}


def test_answer_key_expected_section_present_passes():
    result = check_answer_key("The charge stands under Section 302 IPC.", _transition_ctx())
    assert result.passed
    assert result.detail["missing"] == []
    assert result.detail["cited"] == ["IPC 302"]
    assert result.detail["governing_family"] == "old"


def test_answer_key_expected_section_missing_fails():
    result = check_answer_key("The charge stands under Section 304 IPC.", _transition_ctx())
    assert not result.passed
    assert result.detail["missing"] == ["IPC 302"]
    assert disposition([result]) == "reject"


def test_answer_key_forbidden_section_present_fails():
    answer = "Charged under Section 302 IPC, now renumbered as Section 103 BNS."
    result = check_answer_key(answer, _transition_ctx())
    assert not result.passed
    assert result.detail["forbidden_present"] == ["BNS 103"]


def test_answer_key_matches_at_base_number_granularity():
    ctx = _transition_ctx(
        answer_key=_key(expected_sections=[{"code": "BNS", "number": "103"}],
                        forbidden_sections=[]),
    )
    result = check_answer_key("Punishable under Section 103(2) BNS.", ctx)
    assert result.passed
    # ... and the same tolerance makes a forbidden section impossible to
    # dodge by adding a subsection.
    forbidding = _transition_ctx(
        answer_key=_key(expected_sections=[], forbidden_sections=[{"code": "BNS", "number": "103"}]),
    )
    assert not check_answer_key("Punishable under Section 103(2) BNS.", forbidding).passed


def test_answer_key_never_strips_a_letter_suffix():
    ctx = _transition_ctx(
        answer_key=_key(expected_sections=[{"code": "IPC", "number": "304B"}],
                        forbidden_sections=[]),
    )
    assert not check_answer_key("Charged under Section 304 IPC.", ctx).passed
    assert check_answer_key("Charged under Section 304B IPC.", ctx).passed


def test_answer_key_pinned_subsection_is_not_satisfied_by_a_sibling():
    # BNS 103(1) is murder and BNS 103(2) is mob lynching. A key that pins the
    # subsection means it; the superset direction (bare key, subsectioned
    # cite) is the one that stays tolerant.
    ctx = _transition_ctx(
        answer_key=_key(expected_sections=[{"code": "BNS", "number": "103(2)"}],
                        forbidden_sections=[]),
    )
    assert check_answer_key("Punishable under Section 103(2) BNS.", ctx).passed
    assert not check_answer_key("Punishable under Section 103(1) BNS.", ctx).passed
    assert not check_answer_key("Punishable under Section 103 BNS.", ctx).passed


def test_answer_key_pinned_forbidden_subsection_does_not_fire_on_a_sibling():
    ctx = _transition_ctx(
        answer_key=_key(expected_sections=[],
                        forbidden_sections=[{"code": "BNS", "number": "103(2)"}]),
    )
    assert check_answer_key("Punishable under Section 103(1) BNS.", ctx).passed
    assert not check_answer_key("Punishable under Section 103(2) BNS.", ctx).passed


def test_answer_key_unresolvable_code_is_malformed_not_a_silent_no_op():
    # A typo'd forbidden code must not sit there never firing.
    entry = {"code": "BNSX", "number": "103"}
    ctx = _transition_ctx(answer_key=_key(forbidden_sections=[entry]))
    result = check_answer_key("Charged under Section 103 BNS.", ctx)
    assert not result.passed
    assert result.detail["malformed_key_entries"] == [repr(entry)]


def test_answer_key_non_dict_key_fails_without_raising():
    ctx = _ctx(stream="transition", answer_key=["IPC 302"])
    result = check_answer_key("Charged under Section 302 IPC.", ctx)
    assert not result.passed
    assert "IPC 302" in result.detail["malformed_key"]


def test_answer_key_skipped_on_an_unparsed_row():
    ctx = _transition_ctx()
    assert check_answer_key("Charged under Section 302 IPC.", ctx, think=None).passed
    assert check_answer_key("anything", ctx, think=None).detail == {
        "skipped": "unparsed-format"
    }


def test_answer_key_governing_family_is_coerced_for_json():
    ctx = _transition_ctx(answer_key=_key(governing_family=BEFORE))
    result = check_answer_key("Charged under Section 302 IPC.", ctx)
    assert result.detail["governing_family"] == "2023-05-04"
    assert json.loads(json.dumps(result.detail)) == result.detail


def test_answer_key_savings_mention_required_and_present_as_a_section():
    ctx = _transition_ctx(answer_key=_key(requires_savings_mention=True, forbidden_sections=[]))
    answer = "Section 302 IPC still governs; Section 358 BNS preserves the liability."
    result = check_answer_key(answer, ctx)
    assert result.passed
    assert result.detail["savings_ok"] is True


@pytest.mark.parametrize(
    "phrase",
    ["the savings clause preserves it", "see §358 of the new code", "see section 358"],
)
def test_answer_key_savings_mention_accepts_the_phrase_forms(phrase):
    ctx = _transition_ctx(answer_key=_key(requires_savings_mention=True))
    result = check_answer_key(f"Section 302 IPC still governs; {phrase}.", ctx)
    assert result.passed, result.detail


def test_answer_key_savings_mention_required_and_absent_fails():
    ctx = _transition_ctx(answer_key=_key(requires_savings_mention=True))
    result = check_answer_key("Section 302 IPC still governs the charge.", ctx)
    assert not result.passed
    assert result.detail["savings_required"] is True
    assert result.detail["savings_ok"] is False


def test_answer_key_can_require_that_no_charge_lies():
    """The key's one word-reading field, both directions.

    On a section a court struck down, the right answer and the wrong answer
    cite the SAME section - the citations are identical and only the assertion
    differs - so this is the only field that can tell them apart.
    """
    ctx = _transition_ctx(
        answer_key=_key(requires_no_liability_statement=True, forbidden_sections=[]),
    )
    correct = (
        "No charge lies under Section 302 IPC: it was struck down before this conduct."
    )
    result = check_answer_key(correct, ctx)
    assert result.passed, result.detail
    assert result.detail["no_liability_required"] is True
    assert result.detail["no_liability_cues"]
    assert result.detail["liability_asserted"] == []

    wrong = "The charge lies under Section 302 IPC and the sentence follows from it."
    failed = check_answer_key(wrong, ctx)
    assert not failed.passed
    assert failed.detail["missing"] == []  # it cites everything, and is still wrong
    assert failed.detail["liability_asserted"] == ["the charge lies under"]
    assert disposition([failed]) == "reject"


def test_answer_key_no_liability_is_not_consulted_unless_the_key_asks():
    """The affirmative vocabulary must be inert everywhere else: on an
    ordinary cell "the charge lies under Section 302 IPC" is the RIGHT
    answer, and a gate that rejected it would be the same error one code
    change to the left."""
    ctx = _transition_ctx(answer_key=_key(forbidden_sections=[]))
    result = check_answer_key("The charge lies under Section 302 IPC.", ctx)
    assert result.passed
    assert result.detail["no_liability_required"] is False
    assert result.detail["no_liability_ok"] is True
    assert result.detail["liability_asserted"] == []


def test_answer_key_no_liability_needs_the_denial_not_merely_the_absence():
    # Saying nothing either way is not the answer: the key asks the model to
    # say that no charge lies, and silence would pass on any answer at all.
    ctx = _transition_ctx(
        answer_key=_key(requires_no_liability_statement=True, forbidden_sections=[]),
    )
    result = check_answer_key("Section 302 IPC is the provision in issue.", ctx)
    assert not result.passed
    assert result.detail["no_liability_cues"] == []


def test_answer_key_a_negated_assertion_is_not_an_assertion():
    # A permanent gate that fires on a correct answer awkwardly phrased is the
    # failure this whole field exists to remove, so the negation window is
    # tested with the cue INSIDE the negated clause.
    ctx = _transition_ctx(
        answer_key=_key(requires_no_liability_statement=True, forbidden_sections=[]),
    )
    answer = (
        "It is not the case that the charge lies under Section 302 IPC - the section was "
        "struck down, so no offence lies."
    )
    result = check_answer_key(answer, ctx)
    assert result.passed, result.detail
    assert result.detail["liability_asserted"] == []


def test_answer_key_an_answer_that_says_both_things_is_still_rejected():
    # ...and the clause break is what keeps the negation from reaching across
    # into the clause that contradicts it.
    ctx = _transition_ctx(
        answer_key=_key(requires_no_liability_statement=True, forbidden_sections=[]),
    )
    answer = (
        "No charge lies under Section 302 IPC; the charge lies under Section 302 IPC and "
        "the sentence follows."
    )
    result = check_answer_key(answer, ctx)
    assert not result.passed
    assert result.detail["no_liability_cues"]  # it DID deny...
    assert result.detail["liability_asserted"] == ["the charge lies under"]  # ...and assert


def _no_liability_ctx():
    return _transition_ctx(
        answer_key=_key(requires_no_liability_statement=True, forbidden_sections=[]),
    )


# Every one of these ASSERTS a live charge under a section that is not
# chargeable, and every one carries a denial cue as well - so the denial limb
# cannot be what fails them. What fails them is the assertion limb, which is
# the thing under test.
ASSERTS_THROUGH_A_COMMA = [
    # The reviewer's measured shape: a concessive clause hands the sentence to
    # a main clause that asserts, and round 1's rule let the "no" reach across.
    "Although it is no longer in force, the accused stands charged under Section 302 "
    "IPC, and the savings clause preserves that liability.",
    # ...and their second: two views, the second of which asserts.
    "On one view no offence is made out, on another the charge lies under Section 302 "
    "IPC and the savings clause would preserve it.",
    # More concessives, because "although" is not the only one.
    "Though no offence survives the repeal, the accused stands charged under Section "
    "302 IPC.",
    "While no offence survives today, the charge lies under Section 302 IPC.",
    "Granted that the provision is void, the accused stands charged under Section 302 "
    "IPC.",
    # A bare comma with no concessive at all: still a new clause, still an
    # assertion.
    "No offence is made out on these dates, the charge lies under Section 302 IPC.",
    # A concessive AND a complement clause: the complement exception must not
    # rescue an assertion that a concessive already handed over.
    "Although it is no longer in force, it is said that the charge lies under Section "
    "302 IPC.",
]

# Every one of these DENIES, in a shape where the negation genuinely reaches
# the cue. A permanent gate that rejected any of them would be doing the harm
# this field exists to prevent.
DENIES_THROUGH_A_COMMA = [
    # The reviewer's measured correct shape: commas around an interrupter.
    "It is not, on any view, the case that the charge lies under Section 302 IPC.",
    # Interrupters can nest and can carry their own verb.
    "It is not, on any view, and never has been, the case that the charge lies under "
    "Section 302 IPC.",
    "It cannot be said, on these facts, that the charge lies under Section 302 IPC.",
    # A negation with no comma at all between it and the cue.
    "It is not the case that the charge lies under Section 302 IPC.",
    "The question is not whether the charge lies under Section 302 IPC but whether any "
    "offence existed at all.",
]


@pytest.mark.parametrize("answer", ASSERTS_THROUGH_A_COMMA)
def test_answer_key_a_comma_does_not_carry_the_negation_into_the_next_clause(answer):
    """THE round-1 hole. `_CLAUSE_BREAK_RE` was `[.;:!?]\\s` - no comma - so a
    concessive clause excused the clause that contradicted it, and

        "Although it is no longer in force, the accused stands charged under
         s.497, and s.358 preserves that liability"

    passed all ten gates with a clean disposition and entered the dataset. It
    is the semicolon hedge this module already pinned, repunctuated.
    """
    result = check_answer_key(answer, _no_liability_ctx())
    assert not result.passed, result.detail
    assert result.detail["liability_asserted"], result.detail
    # ...and it is the ASSERTION limb failing, not the denial limb: each of
    # these does carry a denial cue, so a test that passed for want of one
    # would be testing nothing.
    assert result.detail["no_liability_cues"], result.detail


@pytest.mark.parametrize("answer", DENIES_THROUGH_A_COMMA)
def test_answer_key_a_negation_that_really_governs_is_not_broken_by_a_comma(answer):
    """The other direction, and the constraint that makes the fix non-trivial:
    simply adding "," to the break set fixes the shapes above and starts
    PERMANENTLY rejecting these, which is the failure the field exists to
    remove. The complement clause is what tells them apart - "it is not ...
    the case THAT the charge lies" negates exactly what follows it.
    """
    result = check_answer_key(answer + " No charge lies.", _no_liability_ctx())
    assert result.passed, result.detail
    assert result.detail["liability_asserted"] == []


def test_answer_key_a_distant_concessive_does_not_defeat_a_genuine_denial():
    """The CONCESSIVE_LOOKBACK bound, which nothing else in this file
    exercised - the same shape NEGATION_WINDOW was in until mutation found it
    (review round 3, survivor S7: the unbounded mutant is STRICTER, not
    looser, so every existing DENIES fixture stayed green under it).

    A concessive opening a long recital - 160 characters of comma-joined
    procedural narration, no hard break - has spent its force by the time the
    negator arrives; reading it as governing would over-reject a correct
    denial. The bound says a concessive only defeats the complement rescue
    from nearby. The denial must be the PARENTHETICAL shape (a comma between
    negator and cue) - that comma is what routes the check through the
    concessive lookback at all; without it the negation governs directly and
    the bound is never consulted.
    """
    answer = (
        "Although the chargesheet, the committal record, the deposition summaries and "
        "the exhibits all travelled with the appeal paperbook in their original form, "
        "it is not, on any view, the case that the charge lies under Section 302 IPC. "
        "No charge lies."
    )
    result = check_answer_key(answer, _no_liability_ctx())
    assert result.passed, result.detail
    assert result.detail["liability_asserted"] == []


def test_answer_key_a_distant_negation_does_not_reach_the_assertion():
    """The NEGATION_WINDOW bound, which nothing else in this file exercised.

    Found by mutation: unbounding the window (scope = everything before the
    cue) left every other assertion green, because every other fixture puts
    its negator within a few words of the cue. The sentence below is the case
    the bound exists for - a "no" that belongs to a different part of the
    sentence entirely, sixty-odd characters away, with no hard break between
    to stop it. It denies nothing about the charge, and it must not excuse the
    assertion that follows it.
    """
    ctx = _no_liability_ctx()
    far = (
        "There is no dispute that the dates are as stated and that the parties are "
        "correctly described in the papers now before this court, and the charge lies "
        "under Section 302 IPC."
    )
    result = check_answer_key(far, ctx)
    assert not result.passed
    assert result.detail["liability_asserted"] == ["the charge lies under"]


def test_answer_key_denial_cues_match_at_word_boundaries():
    """"void" is a substring of "avoid", so an answer that denies nothing
    satisfied the limb that exists to require a denial."""
    ctx = _no_liability_ctx()
    result = check_answer_key(
        "To avoid doubt, the provision engaged is Section 302 IPC.", ctx
    )
    assert not result.passed
    assert result.detail["no_liability_cues"] == []
    # The word itself still counts, so the boundary did not cost the cue.
    kept = check_answer_key(
        "The section was void when this conduct occurred, so Section 302 IPC reaches "
        "nothing.",
        ctx,
    )
    assert kept.detail["no_liability_cues"] == ["void"]
    assert kept.passed


def test_answer_key_both_families_required_and_named():
    ctx = _transition_ctx(
        answer_key=_key(must_name_both_families=True, forbidden_sections=[]),
    )
    answer = "Charged under Section 302 IPC; the successor provision is Section 103 BNS."
    result = check_answer_key(answer, ctx)
    assert result.passed
    assert result.detail["families"] == ["new", "old"]


def test_answer_key_both_families_required_but_only_one_named_fails():
    ctx = _transition_ctx(answer_key=_key(must_name_both_families=True))
    result = check_answer_key("Charged under Section 302 IPC alone.", ctx)
    assert not result.passed
    assert result.detail["families"] == ["old"]
    assert result.detail["both_families_required"] is True


def test_answer_key_malformed_entry_fails_loudly():
    ctx = _transition_ctx(answer_key=_key(expected_sections=[{"code": "IPC"}, "302 IPC"]))
    result = check_answer_key("Charged under Section 302 IPC.", ctx)
    assert not result.passed
    assert len(result.detail["malformed_key_entries"]) == 2


def test_answer_key_resolves_spelled_out_code_aliases():
    ctx = _transition_ctx(
        answer_key=_key(expected_sections=[{"code": "Indian Penal Code", "number": 302}]),
    )
    result = check_answer_key("Charged under Section 302 IPC.", ctx)
    assert result.passed
    assert result.detail["expected"] == ["IPC 302"]


def test_answer_key_detail_is_json_safe():
    result = check_answer_key("Charged under Section 103 BNS.", _transition_ctx())
    assert json.loads(json.dumps(result.detail)) == result.detail


# --------------------------------------------------------------------------
# run_all / disposition
# --------------------------------------------------------------------------

def test_run_all_returns_every_gate_in_order_even_when_the_first_fails():
    results = run_all("no tags anywhere", 100, _ctx())
    assert [r.gate for r in results] == list(GATE_ORDER)
    assert not results[0].passed


def test_run_all_details_are_all_json_safe(index):
    ctx = _ctx(citation_index=index, stream="transition", answer_key=_key())
    for content in (_good_content(), "no tags anywhere", _empty_think_content("short")):
        for result in run_all(content, 200, ctx):
            assert json.loads(json.dumps(result.detail)) == result.detail


def test_run_all_on_a_fully_valid_example_passes_every_gate(index):
    band = _cfg().build.length_band
    ctx = _ctx(citation_index=index, band=band)
    results = run_all(_good_content(), 400, ctx)
    failed = [(r.gate, r.detail) for r in results if not r.passed]
    assert failed == []
    assert disposition(results) is None


def test_run_all_on_a_valid_empty_think_row_passes_every_gate(index):
    ctx = _ctx(citation_index=index, expect_reasoning=False)
    results = run_all(_empty_think_content(GOOD_ANSWER), 100, ctx)
    assert [(r.gate, r.detail) for r in results if not r.passed] == []
    # The reasoning-only gates report themselves as skipped, not as passes
    # that were actually evaluated.
    skipped = {r.gate for r in results if r.detail.get("skipped")}
    assert {"self_verification", "irac_placement", "banned_meta"} <= skipped


def test_run_all_fabricated_citation_is_a_permanent_reject(index):
    ctx = _ctx(citation_index=index, band=_cfg().build.length_band)
    content = _content(GOOD_THINK, GOOD_ANSWER.replace("(2008) 1 SCC 1", "(2019) 4 SCC 999"))
    results = run_all(content, 400, ctx)
    failed = {r.gate for r in results if not r.passed}
    assert failed == {"citations"}
    assert disposition(results) == "reject"


def test_run_all_format_failure_is_retryable(index):
    ctx = _ctx(citation_index=index, band=_cfg().build.length_band)
    content = _good_content() + _tags()[1]  # a stray close tag in the answer
    results = run_all(content, 400, ctx)
    failed = {r.gate for r in results if not r.passed}
    assert "think_format" in failed
    assert not (failed & PERMANENT_GATES)
    assert disposition(results) == "regenerate"


def test_run_all_derives_think_and_answer_estimates_from_the_split():
    results = run_all(_content("t" * 400, "a" * 200), 50, _ctx())
    band_detail = results[1].detail
    assert band_detail["think_est"] == 100
    assert band_detail["answer_est"] == 50
    assert band_detail["prompt_est"] == 50


def test_run_all_answer_key_gate_sees_a_preamble_before_the_think_block():
    # split_think keeps prose written before the block inside the answer, so a
    # forbidden section cited there cannot hide from the answer-key gate.
    think_open, think_close = _tags()
    ctx = _ctx(stream="transition", answer_key=_key())
    content = f"Quick note: Section 103 BNS applies.{think_open}trace{think_close}Section 302 IPC."
    results = {r.gate: r for r in run_all(content, 100, ctx)}
    assert not results["answer_key"].passed
    assert results["answer_key"].detail["forbidden_present"] == ["BNS 103"]


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        " " * 4000,
        "\r\n\r\n",
        "<think>" * 3,
        "</think>",
        "\x00 stray null and a §§ 302/34 IPC list",
        "Ünïcödé — em-dash, ellipsis…, and (2008)  1  S.C.C.  1",
        "<think>" + "a" * 5000 + "</think>" + "b" * 5000,
    ],
)
def test_gates_never_raise_on_weird_content(index, content):
    for ctx in (
        _ctx(citation_index=index),
        _ctx(citation_index=index, expect_reasoning=False),
        _ctx(citation_index=index, stream="transition", answer_key=_key()),
        _ctx(citation_index=index, stream="replay", offence_date=None),
    ):
        results = run_all(content, 100, ctx)
        assert [r.gate for r in results] == list(GATE_ORDER)
        for result in results:
            assert json.loads(json.dumps(result.detail)) == result.detail
        assert disposition(results) in (None, "reject", "regenerate")


def test_a_malformed_row_is_regenerated_not_permanently_rejected(index):
    # Identical words, one stray close tag apart. The trace rules OUT the
    # forbidden section, which is exactly the reasoning a transition example
    # should contain - scoring the trace as the answer would escalate a
    # formatting retry into a permanent reject and burn the seed.
    ctx = _ctx(citation_index=index, stream="transition", answer_key=_key())

    well_formed = _content(TRANSITION_THINK, GOOD_ANSWER)
    clean = run_all(well_formed, 100, ctx)
    assert [(r.gate, r.detail) for r in clean if not r.passed] == []
    assert disposition(clean) is None

    malformed = well_formed + _tags()[1]
    results = {r.gate: r for r in run_all(malformed, 100, ctx)}
    assert not results["think_format"].passed
    assert results["answer_key"].detail == {"skipped": "unparsed-format"}
    assert results["irac_placement"].detail == {"skipped": "unparsed-format"}
    failed = {gate for gate, r in results.items() if not r.passed}
    assert not (failed & PERMANENT_GATES)
    assert disposition(list(results.values())) == "regenerate"


def test_disposition_mapping():
    passing = [GateResult(gate, True, {}) for gate in GATE_ORDER]
    assert disposition(passing) is None

    for gate in PERMANENT_GATES:
        mixed = [GateResult(g, g != gate, {}) for g in GATE_ORDER]
        assert disposition(mixed) == "reject", gate

    for gate in set(GATE_ORDER) - PERMANENT_GATES - DIAGNOSTIC_GATES:
        mixed = [GateResult(g, g != gate, {}) for g in GATE_ORDER]
        assert disposition(mixed) == "regenerate", gate

    for gate in DIAGNOSTIC_GATES:
        mixed = [GateResult(g, g != gate, {}) for g in GATE_ORDER]
        assert disposition(mixed) is None, gate


def test_self_verification_only_failure_is_stored_but_does_not_decide_disposition():
    results = [GateResult(gate, gate != "self_verification", {}) for gate in GATE_ORDER]
    failed = [result for result in results if not result.passed]
    assert [result.gate for result in failed] == ["self_verification"]
    assert disposition(results) is None


def test_self_verification_does_not_appear_in_disposition_when_another_gate_fails():
    results = [
        GateResult("self_verification", False, {"cues": []}),
        GateResult("irac_placement", False, {"missing_in_answer": ["conclusion"]}),
    ]
    assert disposition(results) == "regenerate"


def test_disposition_permanent_beats_retryable_when_both_fail():
    results = [
        GateResult("think_format", False, {}),
        GateResult("citations", False, {}),
    ]
    assert disposition(results) == "reject"


def test_disposition_of_an_empty_result_list_is_clean():
    assert disposition([]) is None


# --------------------------------------------------------------------------
# The 2026-08-18 cue widening, and its re-audit record.
# --------------------------------------------------------------------------
# Fixtures below are REAL SPANS from the re-pilot's logged traces, trimmed.
# The run is not in the repo (data/ is gitignored) so the counterfactual cannot
# be replayed in-test; these are the spans the counterfactual was computed from,
# frequency-ranked, and the frequencies are recorded beside them.
# The cues added on 2026-08-18, named here so the "did the old list miss
# it?" half of the re-audit cannot silently drift out of step with gates.py.
_WIDENING_ADDITIONS = frozenset({
    "actually", "wait:", "let's check", "let's verify", "let's confirm",
    "let's make sure", "let's see", "let me think", "let me see",
    "let me work", "not sure", "unsure", "unclear", "re-read",
    "re-check", "cross-check", "re-derive",
})

WIDENING_EVIDENCE = (
    # (span, how many of the 89 cue-less traces this shape recovered)
    ("Instrument likely is a revision petition? Actually we are senior drafting "
     "a response", 17),
    ("--- Now, let's verify the substance and ensure that the instrument", 5),
    ("Let's check the four headings: 1", 5),
    ("Let me think through the operative part carefully: Paragraph", 7),
    ("Actually Section 4 deals with licensing, not sure", 3),
    ("previous year is AY 1969-70? Wait: If transfer on 15 May 1968", 2),
    ("Let me re-derive the relief from the provisions: Section", 1),
)


@pytest.mark.parametrize("span, _frequency", WIDENING_EVIDENCE)
def test_the_widened_cues_recover_the_spans_they_were_measured_on(span, _frequency):
    """Each of these was a trace verifying itself in words the old vocabulary
    could not see. Together they are the 2 -> 16 clean-row counterfactual."""
    ctx = _ctx()
    assert check_self_verification(span, ctx).passed, span


@pytest.mark.parametrize("span, _frequency", WIDENING_EVIDENCE)
def test_the_old_vocabulary_missed_every_one_of_them(span, _frequency):
    """The other half of the re-audit: if a span here starts passing under the
    PRE-widening list, it was never evidence for the widening and this table is
    lying about what it bought."""
    old = tuple(c for c in VERIFICATION_CUES if c not in _WIDENING_ADDITIONS)
    text = _norm_ws(span).lower()
    assert not [c for c in old if c.lower() in text], span


def test_the_widened_cues_admit_no_trace_that_never_doubts():
    """THE SAFETY PROPERTY. A gate that exists to reject confident
    hallucination must not be widened into passing traces that never doubt.

    These are the shapes the widening was measured against and deliberately
    left out - `verify`/`confirm`/`ensure` as bare stems, and `uncertain` -
    because in a DRAFTING deliverable they are the instrument, not doubt.
    Measured: "ensure" appears in 45 of the 89 cue-less traces, and "uncertain"
    passed a trace carrying no doubt-marking language of any kind.
    """
    ctx = _ctx()
    for span in (
        "We must ensure essential averments are present and the recitals complete.",
        "The petition must be verified by the petitioner and served on the respondent.",
        "We need to include verification clause, annexures, limitation and service.",
        "The fourth heading covers verification, annexures, limitation, service.",
        "Provisions, indispensable averments, gaps, uncertainties are listed below.",
        "I start with what actually has to be decided for my client.",
    ):
        assert not check_self_verification(span, ctx).passed, span


def test_actually_counts_only_where_it_opens_a_sentence():
    """Position is the meaning for this one: sentence-initial "Actually" is a
    model changing its mind, mid-sentence "actually" is an adverb. The bare
    form passed 'what actually has to be decided' - a real fixture in
    test_build_verify - which is a trace stating its task, not doubting it."""
    ctx = _ctx()
    assert check_self_verification("The section applies? Actually it does not.", ctx).passed
    assert not check_self_verification(
        "I start with what actually has to be decided here.", ctx
    ).passed


def test_a_cue_is_not_matched_inside_a_longer_word():
    """'wait' lives inside 'awaiting' and 'waited'. The pre-existing cues are
    PREFIXES ('re-examin' must keep matching 're-examining'), so the anchor is
    a word boundary on the LEFT only - both ends would break them."""
    ctx = _ctx()
    assert not check_self_verification("The parties awaited the order.", ctx).passed
    assert not check_self_verification("Counsel is awaiting instructions.", ctx).passed
    assert check_self_verification("Wait: the date is wrong.", ctx).passed
    assert check_self_verification("I will re-examining that step.", ctx).passed
