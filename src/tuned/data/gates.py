"""Deterministic rejection gates for the law_v1 curation pipeline.

Every teacher generation passes through these pure functions BEFORE any
judge sees it - they are cheap, they never call a model, and they catch the
failure modes that no amount of judging can repair. Two dispositions come
out the other side (see `disposition`):

  reject      a citation that does not exist, a section from the wrong code
              family, or an answer that contradicts a known-good answer key.
              These are content errors: the example is wrong about the law,
              and rewriting the prose cannot make it right. Never repaired,
              never regenerated - the seed is burned. PERMANENT_GATES.
  regenerate  format, length, missing self-verification, scripted IRAC in the
              think trace, verbatim copying, meta-references to "the provided
              text". The teacher was asked the right question and answered it
              badly; ask again (<=2 attempts).

Nothing here re-implements the hard primitives. Citation existence lives in
citations.py (`novel_citations` + the `suspect_citations` second channel),
the IPC/CrPC/IEA -> BNS/BNSS/BSA transition rules live in statutes.py
(`cross_code_review`, `extract_sections`), the byte-exact empty-think block
lives in replay.py (`empty_think`). This module only composes them and
records WHY something failed.

MANDATORY FOLLOW-UP, and it is not optional: a GateContext built with
citation_index=None runs only the SUSPECT half of the citation gate (the
unmodelled-reporter channel, which needs no index). The existence half is
skipped and the detail says so - {"novel": None, "novel_skipped": "no-index"}.
A row that passed the gates in that mode has NEVER had its citations checked
against the corpus. verify.py (not built yet - this is a recorded dependency)
MUST re-run run_all with the real index before any such row is promoted into
the dataset, and a stored gate_result carrying novel_skipped must be read as
"unverified", never as "passed".

Instrumentation is the second job, and it shapes the API: `run_all` never
short-circuits. Per-gate pass rates over the pilot are how the prompt gets
fixed, so every gate runs on every generation and returns a GateResult even
when it does not apply (passed=True, detail={"skipped": ...}). The result
list is a fixed length in a fixed order (GATE_ORDER), which is also what
makes store.gate_result rows comparable across runs. Every detail dict is
json.dumps-able - store.record_gates serialises it verbatim.

Length checks here are COARSE: callers pass estimated tokens (chars//4, or
real API usage numbers when the provider reported them). Exact tokenizer
enforcement happens at assembly, against the pinned tokenizer.
"""

import re
from dataclasses import dataclass
from datetime import date

from tuned.data.citations import (
    CitationIndex,
    novel_citations,
    suspect_citations,
    suspect_key,
)
from tuned.data.config import LengthBand
from tuned.data.replay import empty_think
from tuned.data.statutes import (
    NEW_CODES,
    OLD_CODES,
    SectionRef,
    cross_code_review,
    extract_sections,
    normalize_number,
    resolve_code,
    statute_pattern,
)

# The two task.stream values that change gate behaviour.
TRANSITION_STREAM = "transition"
REPLAY_STREAM = "replay"

GATE_ORDER = (
    "think_format",
    "length_band",
    "citations",
    "temporal",
    "self_verification",
    "irac_placement",
    "verbatim_overlap",
    "statutory_quotation",
    "banned_meta",
    "answer_key",
)

# Failing one of these means the example states something false about the
# law. Reject-never-repair.
PERMANENT_GATES = frozenset({"citations", "temporal", "answer_key"})

# A trace that never doubts itself is a trace that teaches confident
# hallucination. Extendable - callers may append, the list is not a closed
# vocabulary. Matched case-insensitively against the whitespace-normalized
# trace, so a cue split across a line break still counts.
VERIFICATION_CUES = (
    "let me check",
    "let me verify",
    "double-check",
    "wait,",
    "actually,",
    "on second thought",
    "re-examin",
    "to confirm",
    "am I sure",
    "sanity check",
    "let me reconsider",
    "verify this",
)

# Phrases that leak the synthesis harness into the trace. The student model
# is never shown "the provided text" at inference time, so a trace that
# reasons ABOUT a passage teaches it to hallucinate one.
BANNED_META = (
    "the source says",
    "the provided text",
    "based on the provided",
    "according to the passage",
    "the document indicates",
    "as given above",
    "the excerpt",
    "the material provided",
    "in the given text",
)

IRAC_SECTIONS = ("issue", "rule", "application", "conclusion")
# Issue + Conclusion is the floor; the full four trivially satisfies it.
IRAC_REQUIRED = ("issue", "conclusion")

# A line-initial IRAC heading, tolerant of markdown (`## Issue`, `**Issue:**`,
# `1. Issue`, `- Rule -`, `**Conclusion.**`) and of the plural forms. The
# alternation is built FROM IRAC_SECTIONS so the two can never drift. The
# trailing terminator is what keeps ordinary prose out: the heading word must
# be followed by a colon, a full stop, an emphasis/dash marker, or the end of
# the line, so "Issues of fact remain open" (line-initial, but running
# straight into prose) is not a heading. \r is tolerated so CRLF generations
# behave like LF ones.
_IRAC_HEADING_RE = re.compile(
    r"^[ \t]{0,3}(?:#{1,6}[ \t]*)?(?:[-*+•][ \t]+|\d{1,2}[.)][ \t]*)?"
    r"(?:\*{1,3}|_{1,3})?[ \t]*"
    r"(?P<word>" + "|".join(IRAC_SECTIONS) + r")(?:s|\(s\))?"
    r"[ \t\r]*(?:[:*_.–—-]|$)",
    re.IGNORECASE | re.MULTILINE,
)

# BNS s.358 is the repeal-and-savings clause. statutes._cites_savings_clause
# is deliberately NOT reused here: there it SUPPRESSES a temporal flag (and
# is code-aware so "s.358 CrPC" cannot disarm the gate), whereas here it is a
# positive requirement that the answer explain why the old code still bites -
# a different question with a wider accepted vocabulary, spelled out by the
# answer-key contract.
_SAVINGS_RE = re.compile(r"\bsaving|§\s{0,2}358\b|\bsection\s{1,4}358\b", re.IGNORECASE)

# THE ANSWER "no charge lies", as something a key can require and this gate can
# read. A section a court struck down before the conduct is chargeable on no
# date, so on those cells the correct answer NAMES the section (or it has ruled
# out nothing) and then says no charge lies under it. Citations alone cannot
# tell that answer from its opposite - both name the same section - so the key
# needs these two vocabularies and this gate needs both limbs:
#
#   the DENIAL must be present, and
#   an affirmative attribution of the charge must be absent.
#
# The first limb alone already rejects the wrong answer, which asserts the
# charge and never denies it; the second catches the answer that hedges by
# saying both. Neither is a style rule - on these cells they are the whole of
# the difference between the right answer and a false statement of law.
#
# THE COST, stated because a closed vocabulary always has one: an answer that
# is correct and avoids all fourteen cues - "s.497 ceased to have effect in
# 2018 and therefore reaches nothing done afterwards; s.358 is engaged and
# there is nothing for it to save" - fails this limb and is a PERMANENT
# reject. That is inherent to matching words rather than meaning. It is
# mitigated by the prompt, which hands the teacher the vocabulary in the
# provision block ("Struck down in Joseph Shine v. Union of India (2018) and
# not re-enacted"), and the list is deliberately NOT widened speculatively:
# every cue added is also a cue an assertion can wear.
NO_LIABILITY_CUES = (
    "no charge",
    "no offence",
    "not an offence",
    "no prosecution",
    "cannot be charged",
    "cannot be prosecuted",
    "no liability",
    "struck down",
    "read down",
    "unconstitutional",
    "void",
    "no longer in force",
    "nothing to preserve",
    "preserves nothing",
)

# Affirmative attributions of a charge to a section. Each carries its own
# subject ("the charge lies under", not "charge lies under") so that the
# DENIAL - "no charge lies under s.497" - does not contain one as a substring;
# a negation window catches the rest ("it is not the case that the charge lies
# under ..."), because a permanent gate that fires on a correct answer
# awkwardly phrased is the same failure this whole finding is about.
LIABILITY_CUES = (
    "the charge lies under",
    "the charge would lie under",
    "a charge lies under",
    "charged under section",
    "stands charged under",
    "is chargeable under",
    "remains chargeable",
    "is punishable under",
    "remains punishable",
    "continues to be punishable",
    "is liable under",
    "may be prosecuted under",
    "liability is preserved",
    "liability was preserved",
)

# How far back a negation may sit and still govern the cue, and where it stops
# governing. Three rules, and each one is here because a measured phrasing
# needed it - see _negated_at, where they are applied.
#
#   1. a HARD break ends the negation's reach: "no charge lies under s.497;
#      the charge lies under s.497" is not a denial, it is an answer that says
#      both things.
#   2. a COMMA ends it too - "Although it is no longer in force, the accused
#      stands charged under s.497" asserts the charge, and the concessive
#      clause before the comma does not take that back. Round 1 shipped rule 1
#      without rule 2, and that sentence passed every gate and entered the
#      dataset: the pinned semicolon hedge with different punctuation.
#   3. EXCEPT where the cue sits inside the negation's own complement clause -
#      "it is not, on any view, the case THAT the charge lies under s.497"
#      negates exactly what follows it, and rejecting that would be a
#      permanent gate firing on a correct answer, which is the harm this whole
#      field exists to remove. The complementizer is the signal; a concessive
#      lead before the negator overrides it, because "although the section is
#      not in force, it is said that the charge lies" is an assertion wearing
#      a complement.
NEGATION_WINDOW = 48
CONCESSIVE_LOOKBACK = 120
_NEGATOR_RE = re.compile(
    r"\b(?:no|not|never|nor|neither|cannot|without|nothing)\b|n't\b", re.IGNORECASE
)
_CLAUSE_BREAK_RE = re.compile(r"[.;:!?]\s")
_CONCESSIVE_RE = re.compile(
    r"\b(?:although|though|even\s{1,4}if|while|whilst|whereas|albeit|"
    r"granted\s{1,4}that|admittedly|notwithstanding|on\s{1,4}one\s{1,4}view|"
    r"that\s{1,4}said)\b",
    re.IGNORECASE,
)
_COMPLEMENT_RE = re.compile(r"\bthat\b", re.IGNORECASE)


def _cue_pattern(cue: str) -> re.Pattern:
    """A cue, matched at word boundaries.

    Plain `in` was the first cut and it was wrong in the direction that
    matters: "void" is a substring of "avoid", so "To avoid doubt, the
    provision engaged is s.497" satisfied the limb that exists to REQUIRE a
    denial, on an answer that denies nothing.
    """
    return re.compile(r"\b" + re.escape(cue) + r"\b", re.IGNORECASE)

# Shingle stride for the verbatim scan; see find_verbatim_run.
SHINGLE_STEP = 10

# HOW LONG A SHARED RUN HAS TO BE BEFORE IT IS TRANSCRIPTION.
#
# RAISED 30 -> 120 on 2026-08-18, and the old value is why this gate fired on
# 151 of 221 pilot generations (68%, and 58/58 = 100% on third attempts).
#
# 30 characters is five or six words of Indian legal English, which is not a
# quotation - it is a case name, a court, or the title of an Act. The gate's
# own docstring says it exists to catch a trace that "copies the source ... it
# is transcription", and at 30 it was not measuring that. Every one of the 120
# distinct runs it matched was of this kind:
#
#     ' High Court of Madhya Pradesh '
#     ' Central Excise and Salt Act, '
#     ' S. Govinda Menon v. Union of '
#     ' Section 22 Hindu Succession A'
#
# A trace CANNOT reason about a case without naming it, so at 30 the gate was
# rejecting the act of thinking about the matter at all.
#
# THE RE-AUDIT, measured over the 55 pilot drafting traces whose grounding
# could be recovered from the stored prompts. Longest run shared between trace
# and source, by percentile: p0 19, p25 34, p50 54, p75 76, p90 130, p100 335.
# Failure rate by candidate threshold:
#
#     max_run    30    40    50    60    80   100   120   150   200
#     traces     82%   65%   55%   42%   24%   16%   13%   11%    9%
#
# 120 sits where the curve flattens: it clears the median incidental overlap
# (54) by better than 2x, and the runs it still catches are the six genuine
# copies measured at 167, 201, 226, 246, 261 and 335 characters - five of them
# on attempts 2 and 3, i.e. exactly the inflated traces the retired effort
# ladder was producing. Anything below 100 is still mostly matching proper
# nouns; anything above 150 starts giving up real copies.
#
# NOT NORMALISED FOR LENGTH, and that is a known residual rather than an
# oversight: a longer trace is monotonically likelier to contain a shared run
# of any fixed size, which is most of why this gate read 57% at attempt 1 and
# 100% at attempt 3 on the same corpus. Retiring the effort ladder removes the
# cause rather than the symptom, so a length-normalised variant is left
# unwritten until there is evidence it is still needed.
#
# SECOND CONSUMER, and moving this number moved it too: check_statutory_
# quotation's `reproduces_grounding` field calls find_verbatim_run with this
# same default to decide whether a quoted span was the build's own paraphrase
# or came from nowhere. That field is DIAGNOSTIC ONLY - it is recorded and is
# explicitly not part of the verdict, both before and after this change - so
# nothing about which rows pass moved with it. What did move is the label: a
# quoted span now has to share 120 characters with the grounding, not 30,
# before it is called a reproduction, so the flag will read `false` more often
# on the transition stream and must not be read as "fabrication rose". Verify
# against raw text before drawing anything from that field.
DEFAULT_MAX_RUN = 120

# A QUOTATION ATTRIBUTED TO A SECTION, which on the transition stream is a
# thing no row may carry. This repository holds no bare-act corpus: what a
# transition prompt shows the teacher is each provision's identity, its
# marginal note, and the OPERATIVE EFFECT this build's statute table records
# for it, labelled in the teacher's own words as not a quotation. So a span in
# quotation marks attributed to a section is either the build's paraphrase
# passed off as enacted words or words from nowhere at all, and nothing in the
# repository could tell the two apart. The form is refused outright.
#
# Measured before this gate existed: an answer reading `Section 358(2) ...
# provides: "The repeal of the Indian Penal Code, 1860 does not affect any
# right, privilege, obligation or liability ..."` passed all nine gates clean.
# The identical lift inside the TRACE was caught by verbatim_overlap, so the
# hole was the answer side exactly.
#
# Three shapes make an attribution, and all three are needed:
#   a quoted span,
#   a section named before it - by number ("Section 358(2) of the BNS") or by
#   word ("the provision", "the Sanhita"),
#   and no sentence break between the two.
# The last is what keeps "Section 302 IPC governs. The informant said 'he
# struck him'" out of it: that quote is attributed to a witness, not to a
# section, and the full stop says so.
# DOUBLE QUOTES ONLY, and it is a measured judgement rather than an oversight.
# A single-quoted attribution (`s.358 BNS provides: 'the repeal does not
# affect any liability'`) escapes this gate. Widening the class to `'` closes
# that and costs a false positive on ordinary possessives, which this domain
# is full of: measured, `Section 302 IPC is engaged: the accused's plea and
# the informant's statement are on the record` then reads "s plea and the
# informant" as a quoted span attributed to s.302 - a clean answer sent back
# for rewriting. The escape is a teacher choosing an unusual quotation mark;
# the cost lands on ordinary prose, so the narrow class stays until a
# possessive-aware form is worth writing.
ATTRIBUTION_WINDOW = 160
_QUOTE_RE = re.compile(r"[\"“]([^\"“”]{12,}?)[\"”]")
_SECTION_SUBJECT_RE = re.compile(
    r"\b(?:sections?|sub-?sections?|clauses?|provisions?|sanhita|adhiniyam"
    r"|penal\s{1,4}code|criminal\s{1,4}procedure|evidence\s{1,4}act"
    r"|the\s{1,4}act|the\s{1,4}code)\b",
    re.IGNORECASE,
)
_SENTENCE_BREAK_RE = re.compile(r"[.!?]\s")


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    detail: dict

    def as_row(self) -> tuple[str, bool, dict]:
        """The (gate, passed, detail_json) triple store.record_gates eats."""
        return (self.gate, self.passed, self.detail)


@dataclass(frozen=True)
class GateContext:
    think_open: str
    think_close: str
    band: LengthBand
    # None = the citation gate reports a skipped pass. The PILOT runs before
    # the 17.1M-row index exists; assembly re-runs the gate WITH the index
    # via verify.py, so a skipped-pass here is provisional, never final.
    citation_index: CitationIndex | None
    # The grounding chunk / seed text the teacher was shown. Doubles as the
    # allow-list for citations and as the corpus the verbatim scan diffs
    # the think trace against.
    source_text: str
    offence_date: date | None
    proceeding_started: date | None
    stream: str
    expect_reasoning: bool
    answer_key: dict | None = None

    @property
    def kind_dates(self) -> dict:
        """cross_code_review's date bundle, built once from the context."""
        return {
            "offence_date": self.offence_date,
            "proceeding_started": self.proceeding_started,
        }


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------

def _norm_ws(text: str | None) -> str:
    """Collapse every whitespace run to one space. Applied to both sides of
    every phrase/substring comparison, so a phrase broken across a line break
    ("the provided\n    text") reads the same as an inline one."""
    return " ".join((text or "").split())


def _as_int(value) -> int:
    """Estimated-token counts arrive from callers (chars//4, or a provider's
    usage block, which occasionally ships None, and float("inf") on a bad
    division). A gate never raises."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _est_tokens(text: str | None) -> int:
    return len(text or "") // 4


def _find_all(haystack: str, needle: str) -> list[int]:
    out: list[int] = []
    start = haystack.find(needle)
    while start != -1:
        out.append(start)
        start = haystack.find(needle, start + 1)
    return out


def _tag_positions(text: str, think_open: str, think_close: str) -> tuple[list[int], list[int]]:
    """Occurrences of each tag. Opens that fall INSIDE a close tag are not
    counted - with the real tags "<think>" is not a substring of "</think>",
    but the config owns those strings and a future pair where it is (say
    "think>" / "</think>") must not read as two opens."""
    closes = _find_all(text, think_close)
    close_spans = [(i, i + len(think_close)) for i in closes]
    opens = [
        i
        for i in _find_all(text, think_open)
        if not any(start <= i < end for start, end in close_spans)
    ]
    return opens, closes


def split_think(content: str, think_open: str, think_close: str) -> tuple[str | None, str]:
    """(think, answer) for a generation. think is None when the content does
    not carry exactly one well-formed, correctly ordered tag pair; the answer
    is then the whole content, so no downstream gate is handed a blank.

    The answer is EVERYTHING OUTSIDE the think block, not merely the tail. A
    model that opens with "Sure - here is the analysis." before the think tag
    would otherwise park prose where no answer-side gate can see it, and
    check_answer_key's forbidden_sections would miss a wrong section cited in
    that preamble. (check_think_format records the preamble separately; it is
    a formatting smell, not a failure on its own.)
    """
    text = content or ""
    if not think_open or not think_close:
        return None, text

    opens, closes = _tag_positions(text, think_open, think_close)
    if len(opens) != 1 or len(closes) != 1:
        return None, text
    open_at, close_at = opens[0], closes[0]
    if close_at < open_at + len(think_open):
        return None, text

    think = text[open_at + len(think_open) : close_at]
    prefix = text[:open_at]
    suffix = text[close_at + len(think_close) :]
    answer = f"{prefix}\n{suffix}" if prefix.strip() else suffix
    return think, answer


def irac_headings(text: str | None) -> set[str]:
    """Line-initial IRAC headings present in `text`, lower-cased and
    singularised."""
    return {m.group("word").lower() for m in _IRAC_HEADING_RE.finditer(text or "")}


def find_verbatim_run(text: str, source: str, max_run: int = DEFAULT_MAX_RUN) -> str | None:
    """The first >=max_run-char run of `source` reproduced verbatim in `text`,
    or None. Both sides must already be whitespace-normalized.

    Exactness under a stride, since this is easy to get subtly wrong: source
    shingles are taken every SHINGLE_STEP characters, but they are
    (max_run - step + 1) chars long, NOT max_run. That length is what makes
    the stride lossless. A shared run of exactly max_run chars sitting at
    source offset s admits anchor start positions in [s, s + max_run - anchor_len]
    - an interval of exactly `step` positions, which therefore always contains
    a multiple of step. Anchoring max_run-length shingles at a stride of 10
    (the obvious implementation) would instead miss every shared run of length
    max_run..max_run+8 that happens to be misaligned - i.e. it would find the
    long copies and quietly wave the borderline ones through.

    An anchor hit is only a candidate (the anchor is shorter than max_run), so
    each one is verified with a real substring test over the <=step windows of
    `text` that contain it. False positives are impossible; the completeness
    argument above says false negatives are too.
    """
    if max_run <= 0 or len(text) < max_run or len(source) < max_run:
        return None

    step = min(SHINGLE_STEP, max_run)
    anchor_len = max_run - step + 1
    anchors = {
        source[i : i + anchor_len] for i in range(0, len(source) - anchor_len + 1, step)
    }

    last_start = len(text) - max_run
    for j in range(0, len(text) - anchor_len + 1):
        if text[j : j + anchor_len] not in anchors:
            continue
        for a in range(max(0, j + anchor_len - max_run), min(j, last_start) + 1):
            window = text[a : a + max_run]
            if window in source:
                return window
    return None


# --------------------------------------------------------------------------
# The gates. Each returns a GateResult and never raises.
# --------------------------------------------------------------------------

def check_think_format(content: str, ctx: GateContext) -> GateResult:
    """Exactly one correctly ordered tag pair, nothing stray or nested
    anywhere, and the trace matches what the row promised: non-empty when
    reasoning is expected, the byte-exact empty block otherwise.

    Counting tags over the WHOLE content is what covers "no tags inside the
    answer": a second open or close anywhere - before the block, after it, or
    nested inside it - makes a count != 1 and fails here.
    """
    text = content or ""
    if not ctx.think_open or not ctx.think_close:
        # An empty tag string matches at every offset, which turns the scan
        # quadratic on a long generation; there is also nothing to verify.
        return GateResult(
            "think_format", False, {"reason": "no-think-tags-configured"}
        )

    opens, closes = _tag_positions(text, ctx.think_open, ctx.think_close)
    detail: dict = {
        "open_count": len(opens),
        "close_count": len(closes),
        "expect_reasoning": ctx.expect_reasoning,
    }

    if len(opens) != 1 or len(closes) != 1:
        detail["reason"] = "not-exactly-one-pair"
        return GateResult("think_format", False, detail)
    open_at, close_at = opens[0], closes[0]
    if close_at < open_at + len(ctx.think_open):
        detail["reason"] = "close-before-open"
        return GateResult("think_format", False, detail)

    think = text[open_at + len(ctx.think_open) : close_at]
    block = text[open_at : close_at + len(ctx.think_close)]
    detail["think_chars"] = len(think)
    # Recorded, never fatal: prose before the block is a smell, and
    # split_think keeps it inside the answer so the content gates still see it.
    detail["prefix_chars"] = open_at

    if ctx.expect_reasoning:
        if not think.strip():
            detail["reason"] = "empty-trace"
            return GateResult("think_format", False, detail)
    elif block != empty_think(ctx.think_open, ctx.think_close):
        # Byte-exact or nothing: the trainer's empty-think rows are literally
        # "<think>\n\n</think>", and a whitespace variant is a different token
        # sequence teaching a different habit.
        detail["reason"] = "empty-block-not-byte-exact"
        return GateResult("think_format", False, detail)

    return GateResult("think_format", True, detail)


def check_length_band(
    prompt_est: int, think_est: int, answer_est: int, ctx: GateContext
) -> GateResult:
    """Coarse length band over ESTIMATED tokens. think_min only applies when a
    trace was expected - an empty-think row has a 0-token trace by design."""
    band = ctx.band
    prompt, think, answer = _as_int(prompt_est), _as_int(think_est), _as_int(answer_est)
    total = prompt + think + answer

    violations: list[str] = []
    if total > band.total_max:
        violations.append("total>total_max")
    if total < band.total_min:
        violations.append("total<total_min")
    if think > band.think_max:
        violations.append("think>think_max")
    if ctx.expect_reasoning and think < band.think_min:
        violations.append("think<think_min")
    if answer < band.answer_min:
        violations.append("answer<answer_min")

    detail = {
        "prompt_est": prompt,
        "think_est": think,
        "answer_est": answer,
        "total_est": total,
        "total_max": band.total_max,
        "total_min": band.total_min,
        "think_min": band.think_min,
        "think_max": band.think_max,
        "answer_min": band.answer_min,
        "think_min_applies": ctx.expect_reasoning,
        "violations": violations,
    }
    return GateResult("length_band", not violations, detail)


def check_citations(content: str, ctx: GateContext) -> GateResult:
    """Reject-on-unknown, over both of citations.py's channels.

    novel_citations covers the formats the index models. suspect_citations
    covers the ones it does not - a citation-SHAPED string no pattern parsed
    is a string the index is never asked about, so an invention in an
    unmodelled reporter would otherwise sail through untouched. Suspects
    carried IN by the grounding text are not the model's invention, so the
    output's suspects are diffed against the source's.

    Without an index the SUSPECT channel still runs - its verdict never
    depended on the index, and rejecting an invented KLT cite during the pilot
    saves the teacher spend a later verify.py pass cannot refund. Only the
    existence half is skipped, and it says so in the detail. READ THE MODULE
    DOCSTRING: a pass carrying novel_skipped means "not yet checked", and
    verify.py must re-run this gate with the real index before the row is
    promoted.

    THE SUSPECT DIFF IS KEYED, NOT LITERAL (2026-08-18). It used to compare the
    two suspect lists as raw strings, which made re-typing a citation into
    standard form indistinguishable from inventing one: the pilot burned two
    seeds permanently on '2015 (4) KLT 163' against a grounding that read
    '2015(4) KLT 163(LB)', and '(2006) 7 SCALE 28' against '[2006 (7) SCALE
    28 ]'. citations.suspect_key folds exactly the punctuation those pairs
    differ by. The REPORTED value stays the string the model actually wrote, so
    the detail still reads as evidence.

    SCOPE IS THE WHOLE CONTENT, TRACE INCLUDED, and it is left that way
    deliberately. It looks inconsistent with check_answer_key, which refuses to
    score an unparsed trace - but that refusal is about answer-key MATCHING
    (a trace saying "the successor WOULD be s.103 BNS, but not here" would trip
    forbidden_sections on a hypothetical), whereas a fabricated citation is a
    fabricated citation wherever it sits. check_temporal and
    check_statutory_quotation both state the same whole-content rule for the
    same class of error, and test_citations_reads_the_whole_content_including_
    the_trace pins it by name. FLAGGED, NOT CHANGED: the residual exposure is
    that a teacher musing about a half-remembered reporter in its private
    reasoning earns a PERMANENT reject. No pilot row hit it - all three
    firings were citations in the ANSWER - so there is nothing measured to act
    on, and narrowing a permanent gate on an unmeasured hunch is the wrong
    direction to be wrong in.
    """
    text = content or ""
    source = ctx.source_text or ""
    grounded_keys = {suspect_key(c) for c in suspect_citations(source)}
    suspects = [c for c in suspect_citations(text) if suspect_key(c) not in grounded_keys]

    if ctx.citation_index is None:
        detail = {"novel": None, "novel_skipped": "no-index", "suspect": suspects}
        return GateResult("citations", not suspects, detail)

    novel = novel_citations(text, source, ctx.citation_index)
    detail = {"novel": novel, "suspect": suspects}
    return GateResult("citations", not novel and not suspects, detail)


def check_temporal(content: str, ctx: GateContext) -> GateResult:
    """Cross-code family check over the whole content (a wrong-family section
    is just as wrong inside the trace as inside the answer).

    The undecidable channel is where the policy lives. On the TRANSITION
    stream the dates are always known - the whole stream is built from them -
    so a section whose family cannot be decided means the example text lost
    the dates that make it answerable: fail. On every other stream the
    upstream corpus rows rarely carry an offence date, and failing there would
    reject most of the corpus for a property it never claimed to have: pass,
    but record what could not be decided so the pilot can count it.
    """
    flags, undecidable = cross_code_review(content or "", kind_dates=ctx.kind_dates)

    flag_details = []
    for flag in flags:
        ref = getattr(flag, "ref", None)
        entry = {"flag": str(flag)}
        if isinstance(ref, SectionRef):
            entry.update({"ref": str(ref), "code": ref.code, "number": ref.number})
        flag_details.append(entry)

    undecidable_fatal = bool(undecidable) and ctx.stream == TRANSITION_STREAM
    detail = {
        "flags": flag_details,
        "undecidable": [str(ref) for ref in undecidable],
        "undecidable_fatal": undecidable_fatal,
        "stream": ctx.stream,
    }
    return GateResult("temporal", not flags and not undecidable_fatal, detail)


def check_self_verification(think: str | None, ctx: GateContext) -> GateResult:
    """>=1 self-verification cue in the trace. Only meaningful where a trace
    was expected."""
    if not ctx.expect_reasoning:
        return GateResult("self_verification", True, {"skipped": "no-reasoning-expected"})

    text = _norm_ws(think).lower()
    hits = [cue for cue in VERIFICATION_CUES if cue.lower() in text]
    return GateResult("self_verification", bool(hits), {"cues": hits})


def check_irac_placement(think: str | None, answer: str, ctx: GateContext) -> GateResult:
    """IRAC belongs in the ANSWER; an IRAC heading inside the trace is the
    MSLR failure mode - a model that scripts its "reasoning" as a template it
    fills in afterwards, which is exactly the habit this dataset must not
    teach. Replay rows are general-domain and carry no IRAC contract.
    """
    if not ctx.expect_reasoning:
        return GateResult("irac_placement", True, {"skipped": "no-reasoning-expected"})
    if ctx.stream == REPLAY_STREAM:
        return GateResult("irac_placement", True, {"skipped": "stream-replay"})
    if think is None:
        # split_think could not parse the content, so `answer` is the whole
        # generation and the think/answer boundary this gate is ABOUT does not
        # exist: the tripwire cannot fire and the heading requirement would be
        # scored against the wrong text. check_think_format has already failed
        # (a None think implies it), so the row is retried regardless - say
        # "not evaluated" rather than banking a meaningless pass.
        return GateResult("irac_placement", True, {"skipped": "unparsed-format"})

    in_answer = irac_headings(answer)
    in_think = irac_headings(think)
    missing = [name for name in IRAC_REQUIRED if name not in in_answer]

    detail = {
        "answer_headings": sorted(in_answer),
        "think_headings": sorted(in_think),
        "missing_in_answer": missing,
        "required": list(IRAC_REQUIRED),
    }
    return GateResult("irac_placement", not missing and not in_think, detail)


def check_verbatim_overlap(
    think: str | None, ctx: GateContext, max_run: int = DEFAULT_MAX_RUN
) -> GateResult:
    """No long verbatim run of the grounding text inside the TRACE. The answer
    may quote freely - a holding is legitimately reproduced there - but a
    trace that copies the source is not reasoning, it is transcription, and it
    teaches the student to expect a passage it will never be given."""
    trace = _norm_ws(think)
    source = _norm_ws(ctx.source_text)
    match = find_verbatim_run(trace, source, max_run)

    detail = {
        "max_run": max_run,
        "think_chars": len(trace),
        "source_chars": len(source),
        "match": match[:80] if match else None,
    }
    return GateResult("verbatim_overlap", match is None, detail)


def check_statutory_quotation(content: str, ctx: GateContext) -> GateResult:
    """No quoted span attributed to a section. TRANSITION STREAM ONLY.

    Every other stream is grounded in judgment text, where a quoted holding is
    a legitimate and useful thing for an answer to carry; check_verbatim_
    overlap already keeps that out of the TRACE and nothing keeps it out of the
    answer, deliberately. The transition stream is the one where the grounding
    is not statute text at all, so the same act means something different: see
    ATTRIBUTION_WINDOW.

    WHOLE CONTENT, trace included, on the same reasoning as check_temporal - a
    fabricated statutory quotation is no better inside the reasoning, and the
    prompt's caution now forbids both.

    NOT a permanent gate, and that is this repository's own rule rather than a
    softening: PERMANENT means "the example is wrong about the law and
    rewriting the prose cannot make it right". Here rewriting the prose is
    exactly what makes it right - the same sentence without the quotation marks
    and the attribution is a true statement of the recorded effect - so this is
    a regenerate. The row cannot reach the dataset either way: only a clean
    disposition promotes.

    `reproduces_grounding` is recorded and NOT part of the verdict. It says
    whether the quoted words were the build's own paraphrase or came from
    nowhere, which is worth counting over a pilot; both are refused, because
    with no bare-act corpus neither can be verified as the section's words.

    IT READS DEFAULT_MAX_RUN, so it moved on 2026-08-18 when that constant went
    30 -> 120 for check_verbatim_overlap's sake. The VERDICT is unaffected -
    this field has never been part of it - but the threshold for calling a
    quotation a reproduction is now four times longer, so the flag turns
    `false` more readily and a drop in it means nothing about fabrication. See
    the note beside DEFAULT_MAX_RUN.
    """
    if ctx.stream != TRANSITION_STREAM:
        return GateResult("statutory_quotation", True, {"skipped": "not-transition"})

    text = _norm_ws(content)
    source = _norm_ws(ctx.source_text)
    hits: list[dict] = []
    for match in _QUOTE_RE.finditer(text):
        window = text[max(0, match.start() - ATTRIBUTION_WINDOW) : match.start()]
        ends = [m.end() for m in _SECTION_SUBJECT_RE.finditer(window)]
        ends += [m.end() for m in statute_pattern().finditer(window)]
        if not ends:
            continue
        between = window[max(ends) :]
        if _SENTENCE_BREAK_RE.search(between):
            continue
        quoted = match.group(1)
        hits.append(
            {
                "quoted": quoted[:80],
                "attribution": (between.strip() or window[-40:].strip())[:60],
                "reproduces_grounding": find_verbatim_run(quoted, source, DEFAULT_MAX_RUN)
                is not None,
            }
        )
    detail = {"quotations": hits, "window": ATTRIBUTION_WINDOW}
    return GateResult("statutory_quotation", not hits, detail)


def check_banned_meta(think: str | None, ctx: GateContext) -> GateResult:
    """No harness leakage in the trace."""
    if not ctx.expect_reasoning:
        return GateResult("banned_meta", True, {"skipped": "no-reasoning-expected"})

    text = _norm_ws(think).lower()
    hits = [phrase for phrase in BANNED_META if phrase in text]
    return GateResult("banned_meta", not hits, {"hits": hits})


def _wanted_sections(entries) -> tuple[list[tuple[str, str]], list[str]]:
    """Answer-key section entries -> normalized (code, number) pairs, plus
    whatever could not be parsed."""
    wanted: list[tuple[str, str]] = []
    malformed: list[str] = []
    if entries is None:
        return wanted, malformed
    if not isinstance(entries, (list, tuple)):
        return wanted, [repr(entries)]

    for entry in entries:
        if not isinstance(entry, dict):
            malformed.append(repr(entry))
            continue
        # An unresolvable code is MALFORMED, not a code named literally: a
        # typo'd forbidden entry ("BNSS " for "BNS") would otherwise resolve to
        # a string extract_sections can never emit, so the entry would sit
        # there never firing - a silent false negative on a permanent gate.
        code = resolve_code(entry.get("code"))
        number = normalize_number(entry.get("number"))
        if not code or not number:
            malformed.append(repr(entry))
            continue
        wanted.append((code, number))
    return wanted, malformed


def _ref_matches(ref: SectionRef, code: str, number: str) -> bool:
    """Does a cited section satisfy an answer-key entry?

    The key decides the granularity. A key that names a bare section ("103")
    is satisfied by any subsection of it ("103(2)") - the key did not care.
    A key that PINS a subsection ("103(2)") requires that exact number,
    because sibling subsections are different offences: BNS 103(1) is murder
    and BNS 103(2) is mob lynching, and treating one as the other is exactly
    the error this gate is supposed to catch. Letter suffixes are never
    stripped either way (SectionRef.base_number guarantees it), so IPC 304B
    never matches IPC 304.
    """
    if ref.code != code:
        return False
    if "(" in number:
        return ref.number == number
    return ref.base_number == number


def _mentions_savings(answer: str, refs: list[SectionRef]) -> bool:
    if any(ref.code == "BNS" and ref.base_number == "358" for ref in refs):
        return True
    return bool(_SAVINGS_RE.search(answer or ""))


_DENIAL_PATTERNS = tuple((cue, _cue_pattern(cue)) for cue in NO_LIABILITY_CUES)
_ASSERTION_PATTERNS = tuple((cue, _cue_pattern(cue)) for cue in LIABILITY_CUES)


def _denies_liability(answer: str) -> list[str]:
    """Cues by which the answer says no charge lies. Whitespace-normalized, so
    a phrase broken across a line break still counts, and matched at word
    boundaries, so "avoid" is not "void"."""
    text = _norm_ws(answer)
    return [cue for cue, pattern in _DENIAL_PATTERNS if pattern.search(text)]


def _negated_at(text: str, at: int) -> bool:
    """Does a negation govern the liability cue starting at `at`?

    Scope: the NEGATION_WINDOW characters before the cue, cut back to the last
    HARD break in them; the closest negator in what remains is the one that
    governs. Then the three rules in the constants above, in order:

      no comma between that negator and the cue        -> it governs
      a concessive lead before that negator            -> it does not
      a "that"-complement between negator and cue      -> it governs
      otherwise (a bare comma, i.e. a new clause)      -> it does not

    The order is the whole design. A comma is a handover by default, because
    the measured failure was a concessive clause handing the sentence to a main
    clause that asserts the charge; the complement exception is narrower than
    the handover rule, so a sentence that is BOTH ("although the section is not
    in force, it is said that the charge lies") is read as the assertion it is.

    THE RESIDUAL, measured rather than guessed, because a rule about English
    written in regular expressions has one and it should be written down:

      "Nothing said here is a concession that, no matter the view taken, the
       charge lies under s.497"        -> read as an ASSERTION. It denies. The
       nearest negator ("no matter") is an intensifier inside the complement,
       and its own tail carries no second complement to save it.
      "It is not disputed, and no one has suggested otherwise, that the charge
       lies under s.497"               -> read as a DENIAL. It asserts: "not
       disputed that X" means X. No vocabulary of cues can see a negation of a
       negation, and this is the class of evasion cue-matching cannot close.

    Both are double-embedded, but the class REACHES ORDINARY LEGAL REGISTER -
    "It is not disputed that the charge lies under s.497" and "It is not denied
    that ..." pass clean (measured, review round 3) and are phrasings a
    one-line declarative answer can produce. The exposure is bounded by the 9
    affected cells per 1,250 drawn, not by the phrasing being exotic; the
    first shape fails safe only in the sense that it burns a seed rather than
    teaching a falsehood. Which negator governs when several are in scope - the
    nearest, as here, or the first - is NOT pinned by any test, because no
    phrasing in the suite has two negators in one scope where the choice
    changes the verdict; across six probe sentences (four here, two more in
    review round 3) every shape where the choice matters favours "first",
    3-0, and none favours "nearest" - recorded, still too little to turn a design
    on.
    """
    scope = text[max(0, at - NEGATION_WINDOW) : at]
    breaks = list(_CLAUSE_BREAK_RE.finditer(scope))
    if breaks:
        scope = scope[breaks[-1].end() :]
    negators = list(_NEGATOR_RE.finditer(scope))
    if not negators:
        return False

    scope_at = at - len(scope)
    negator = negators[-1]
    tail = text[scope_at + negator.end() : at]
    if "," not in tail:
        return True

    lead_end = scope_at + negator.start()
    lead = text[max(0, lead_end - CONCESSIVE_LOOKBACK) : lead_end]
    lead_breaks = list(_CLAUSE_BREAK_RE.finditer(lead))
    if lead_breaks:
        lead = lead[lead_breaks[-1].end() :]
    if _CONCESSIVE_RE.search(lead):
        return False
    return bool(_COMPLEMENT_RE.search(tail))


def _asserts_liability(answer: str) -> list[str]:
    """Cues by which the answer says a charge DOES lie, negated ones excluded.

    Every occurrence of every cue is examined, so one negated use does not
    excuse a later bare one.
    """
    text = _norm_ws(answer)
    hits: list[str] = []
    for cue, pattern in _ASSERTION_PATTERNS:
        for match in pattern.finditer(text):
            if not _negated_at(text, match.start()):
                hits.append(cue)
                break
    return hits


def check_answer_key(
    answer: str, ctx: GateContext, *, think: str | None = ""
) -> GateResult:
    """Transition-stream answers are checked against a known-good key: the
    stream exists precisely because the old/new-code answer is decidable in
    advance, and a generation that gets it wrong is wrong about the law
    (PERMANENT). Every other stream skips.

    `think` is the split_think verdict, NOT text this gate reads: None means
    the content did not parse, so `answer` is the whole generation, trace
    included. Judging that against the key would score the model's private
    reasoning as if it were its answer - and a trace that says "the successor
    provision WOULD be s.103 BNS, but it does not apply here" would then trip
    forbidden_sections and escalate a formatting retry into a permanent
    reject. A None think implies check_think_format has already failed, so the
    row can never be promoted on this pass; skipping here costs no safety and
    the retry gets fully gated. Callers that did not split pass nothing.

    A malformed key still FAILS rather than skipping: that one is an operator
    bug in a hand-authored key, and a key nobody can parse means the row was
    never really checked.

    THE KEY CAN ALSO REQUIRE THAT NO CHARGE LIES. `requires_no_liability_
    statement` is the one field here that reads the answer's words rather than
    its citations, and it exists because on a section a court struck down the
    right answer and the wrong answer cite the SAME section: one says the
    charge lies under it, the other says nothing does. Citations cannot tell
    them apart, so before this field the only answer the key could express on
    those cells was the false one. See NO_LIABILITY_CUES / LIABILITY_CUES.

    governing_family is recorded, not enforced: what the answer may cite is
    already pinned by expected/forbidden sections, and demanding a single
    family would contradict must_name_both_families (the transition answers
    that name both codes in one sentence are the good ones). Whether a cited
    section belongs to the era is check_temporal's job.
    """
    if ctx.stream != TRANSITION_STREAM:
        return GateResult("answer_key", True, {"skipped": "not-transition"})
    key = ctx.answer_key
    if not key:
        return GateResult("answer_key", True, {"skipped": "no-answer-key"})
    if think is None:
        return GateResult("answer_key", True, {"skipped": "unparsed-format"})
    if not isinstance(key, dict):
        return GateResult(
            "answer_key", False, {"malformed_key": repr(key)[:200]}
        )

    text = answer or ""
    refs = extract_sections(text)
    expected, malformed = _wanted_sections(key.get("expected_sections"))
    forbidden, malformed_forbidden = _wanted_sections(key.get("forbidden_sections"))
    malformed = malformed + malformed_forbidden

    missing = [
        f"{code} {number}"
        for code, number in expected
        if not any(_ref_matches(ref, code, number) for ref in refs)
    ]
    present_forbidden = [
        f"{code} {number}"
        for code, number in forbidden
        if any(_ref_matches(ref, code, number) for ref in refs)
    ]

    savings_required = bool(key.get("requires_savings_mention"))
    savings_ok = _mentions_savings(text, refs) if savings_required else True

    # The key may require the answer to be that NO charge lies - see
    # NO_LIABILITY_CUES. Both limbs, and only where the key asked: on an
    # ordinary cell "the charge lies under IPC s.302" is the RIGHT answer, so
    # the affirmative vocabulary must not be consulted there at all.
    no_liability_required = bool(key.get("requires_no_liability_statement"))
    denials = _denies_liability(text) if no_liability_required else []
    assertions = _asserts_liability(text) if no_liability_required else []
    no_liability_ok = (bool(denials) and not assertions) if no_liability_required else True

    both_required = bool(key.get("must_name_both_families"))
    families = sorted(
        {"old" for ref in refs if ref.code in OLD_CODES}
        | {"new" for ref in refs if ref.code in NEW_CODES}
    )
    both_ok = ({"old", "new"} <= set(families)) if both_required else True

    family = key.get("governing_family")
    detail = {
        # Coerced: the key is hand-authored and detail must survive json.dumps.
        "governing_family": None if family is None else str(family),
        "cited": [str(ref) for ref in refs],
        "expected": [f"{code} {number}" for code, number in expected],
        "missing": missing,
        "forbidden_present": present_forbidden,
        "savings_required": savings_required,
        "savings_ok": savings_ok,
        "no_liability_required": no_liability_required,
        "no_liability_ok": no_liability_ok,
        "no_liability_cues": denials,
        "liability_asserted": assertions,
        "both_families_required": both_required,
        "families": families,
        "malformed_key_entries": malformed,
    }
    passed = (
        not missing
        and not present_forbidden
        and savings_ok
        and no_liability_ok
        and both_ok
        and not malformed
    )
    return GateResult("answer_key", passed, detail)


# --------------------------------------------------------------------------
# Composition.
# --------------------------------------------------------------------------

def run_all(content: str, prompt_est_tokens: int, ctx: GateContext) -> list[GateResult]:
    """Every gate, always, in GATE_ORDER - no short-circuit.

    Per-gate pass rates over the pilot are the instrumentation that tells the
    operator whether the PROMPT is broken or the MODEL is, so a run that
    stopped at the first failure would answer the wrong question. The result
    list is therefore a fixed length whatever happens, and gates that do not
    apply to this row report a skipped pass.

    Splitting happens once here; think/answer token counts are chars//4
    estimates (callers with real provider usage numbers can call
    check_length_band themselves).
    """
    text = content or ""
    think, answer = split_think(text, ctx.think_open, ctx.think_close)
    return [
        check_think_format(text, ctx),
        check_length_band(prompt_est_tokens, _est_tokens(think), _est_tokens(answer), ctx),
        check_citations(text, ctx),
        check_temporal(text, ctx),
        check_self_verification(think, ctx),
        check_irac_placement(think, answer, ctx),
        check_verbatim_overlap(think, ctx),
        check_statutory_quotation(text, ctx),
        check_banned_meta(think, ctx),
        check_answer_key(answer, ctx, think=think),
    ]


def disposition(results: list[GateResult]) -> str | None:
    """None = clean. "reject" = wrong about the law, burn the seed.
    "regenerate" = badly written, ask again."""
    failed = [result.gate for result in results if not result.passed]
    if not failed:
        return None
    return "reject" if any(gate in PERMANENT_GATES for gate in failed) else "regenerate"
