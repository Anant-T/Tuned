import json
import random
from datetime import date

import pytest

from tuned.data.citations import CitationIndex
from tuned.data.config import LengthBand, load_build_config
from tuned.data.gates import (
    BANNED_META,
    DEFAULT_MAX_RUN,
    GATE_ORDER,
    IRAC_REQUIRED,
    PERMANENT_GATES,
    SHINGLE_STEP,
    VERIFICATION_CUES,
    GateContext,
    GateResult,
    check_answer_key,
    check_banned_meta,
    check_citations,
    check_irac_placement,
    check_length_band,
    check_self_verification,
    check_temporal,
    check_think_format,
    check_verbatim_overlap,
    disposition,
    find_verbatim_run,
    irac_headings,
    run_all,
    split_think,
)
from tuned.data.statutes import APPOINTED_DAY

# Offence dates either side of the appointed day (2024-07-01).
BEFORE = date(2023, 5, 4)
AFTER = date(2024, 9, 1)

# The grounding chunk the teacher is shown. Cites one real reporter citation
# and carries a holding worth quoting.
SOURCE = (
    "In Anwar Ali v. State, (2008) 1 SCC 1, the Supreme Court held that a "
    "conviction may rest on circumstantial evidence only where the chain of "
    "circumstances is so complete as to exclude every hypothesis except the "
    "guilt of the accused. The trial court had relied on the recovery of a "
    "blood-stained weapon at the instance of the accused and on the testimony "
    "of a single eye witness who deposed four days after the incident."
)

INDEX_ENTRIES = ["(2008) 1 SCC 1", "2023 INSC 45", "AIR 1973 SC 1461"]

# Deliberately small so band violations are easy to aim at one at a time.
BAND = LengthBand(total_max=4000, total_min=100, think_min=50, think_max=800, answer_min=20)

_CFG = None


def _cfg():
    global _CFG
    if _CFG is None:
        _CFG = load_build_config("configs/data_law_v1.yaml", allow_unpinned=True)
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


# --------------------------------------------------------------------------
# Module constants.
# --------------------------------------------------------------------------

def test_gate_order_and_permanent_gates():
    assert GATE_ORDER == (
        "think_format",
        "length_band",
        "citations",
        "temporal",
        "self_verification",
        "irac_placement",
        "verbatim_overlap",
        "banned_meta",
        "answer_key",
    )
    assert PERMANENT_GATES == frozenset({"citations", "temporal", "answer_key"})
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


def test_length_band_against_the_real_config_band():
    band = _cfg().build.length_band
    ctx = _ctx(band=band)
    assert check_length_band(400, 900, 400, ctx).passed
    assert not check_length_band(400, 100, 400, ctx).passed  # under think_min
    assert not check_length_band(400, 900, 8000, ctx).passed  # over total_max


# --------------------------------------------------------------------------
# check_citations
# --------------------------------------------------------------------------

def test_citations_skipped_without_an_index():
    result = check_citations("cites (2019) 4 SCC 999", _ctx())
    assert result.passed
    assert result.detail == {"skipped": "no-index"}


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


def test_irac_placement_skipped_on_the_replay_stream():
    result = check_irac_placement("Issue: anything", "no headings", _ctx(stream="replay"))
    assert result.passed
    assert result.detail == {"skipped": "stream-replay"}


def test_irac_placement_skipped_when_no_reasoning_expected():
    result = check_irac_placement(None, "no headings", _ctx(expect_reasoning=False))
    assert result.passed
    assert result.detail == {"skipped": "no-reasoning-expected"}


# --------------------------------------------------------------------------
# check_verbatim_overlap / find_verbatim_run
# --------------------------------------------------------------------------

def test_verbatim_overlap_35_char_source_run_in_the_trace_fails():
    think = f"Working through it: {SOURCE_RUN_35} and so the link holds."
    result = check_verbatim_overlap(think, _ctx())
    assert not result.passed
    assert result.detail["match"] in _norm(SOURCE)
    assert len(result.detail["match"]) <= 80
    assert disposition([result]) == "regenerate"


def test_verbatim_overlap_same_run_in_the_answer_only_passes():
    # The gate never sees the answer: quoting a holding there is legitimate.
    result = check_verbatim_overlap("a paraphrased trace of my own words", _ctx())
    assert result.passed
    content = _content("a paraphrased trace of my own words", SOURCE_RUN_35)
    assert check_verbatim_overlap(split_think(content, *_tags())[0], _ctx()).passed


def test_verbatim_overlap_survives_reflowed_whitespace():
    reflowed = SOURCE_RUN_35.replace(" ", "\n   ")
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


@pytest.mark.parametrize("offset", range(SHINGLE_STEP))
def test_find_verbatim_run_catches_an_exactly_max_run_copy_at_every_alignment(offset):
    # The shingle stride must not depend on where the shared run happens to
    # sit: a run of EXACTLY max_run chars is the tightest case, and step-10
    # anchoring of max_run-length shingles would miss most of these offsets.
    run = "the accused walked to Fort Koc"
    assert len(run) == DEFAULT_MAX_RUN
    source = "q" * offset + run + " padding that shares nothing else at all"
    text = "unrelated opening words here " + run + " unrelated closing words"
    assert find_verbatim_run(text, source, DEFAULT_MAX_RUN) == run


def test_find_verbatim_run_ignores_a_shorter_shared_run():
    run = "the accused walked to Fort Ko"  # 29 chars
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


def test_disposition_mapping():
    passing = [GateResult(gate, True, {}) for gate in GATE_ORDER]
    assert disposition(passing) is None

    for gate in PERMANENT_GATES:
        mixed = [GateResult(g, g != gate, {}) for g in GATE_ORDER]
        assert disposition(mixed) == "reject", gate

    for gate in set(GATE_ORDER) - PERMANENT_GATES:
        mixed = [GateResult(g, g != gate, {}) for g in GATE_ORDER]
        assert disposition(mixed) == "regenerate", gate


def test_disposition_permanent_beats_retryable_when_both_fail():
    results = [
        GateResult("think_format", False, {}),
        GateResult("citations", False, {}),
    ]
    assert disposition(results) == "reject"


def test_disposition_of_an_empty_result_list_is_clean():
    assert disposition([]) is None
