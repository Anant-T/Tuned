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

from tuned.data.citations import CitationIndex, novel_citations, suspect_citations
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
)

# Stream vocabulary (task.stream in the store). Two of them change gate
# behaviour and are therefore named constants rather than literals.
STREAMS = ("synthesis", "curated_c2", "transition", "replay")
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
# `1. Issue`, `- Rule -`) and of the plural forms. The trailing terminator is
# what keeps ordinary prose out: the heading word must be followed by a
# colon, an emphasis/dash marker, or the end of the line, so "Issues of fact
# remain open" (line-initial, but running straight into prose) is not a
# heading. \r is tolerated so CRLF generations behave like LF ones.
_IRAC_HEADING_RE = re.compile(
    r"^[ \t]{0,3}(?:#{1,6}[ \t]*)?(?:[-*+•][ \t]+|\d{1,2}[.)][ \t]*)?"
    r"(?:\*{1,3}|_{1,3})?[ \t]*"
    r"(?P<word>issue|rule|application|conclusion)(?:s|\(s\))?"
    r"[ \t\r]*(?::|\*|_|-|–|—|$)",
    re.IGNORECASE | re.MULTILINE,
)

# BNS s.358 is the repeal-and-savings clause. statutes._cites_savings_clause
# is deliberately NOT reused here: there it SUPPRESSES a temporal flag (and
# is code-aware so "s.358 CrPC" cannot disarm the gate), whereas here it is a
# positive requirement that the answer explain why the old code still bites -
# a different question with a wider accepted vocabulary, spelled out by the
# answer-key contract.
_SAVINGS_RE = re.compile(r"\bsaving|§\s{0,2}358\b|\bsection\s{1,4}358\b", re.IGNORECASE)

# Shingle stride for the verbatim scan; see find_verbatim_run.
SHINGLE_STEP = 10
DEFAULT_MAX_RUN = 30


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
    usage block, which occasionally ships None). A gate never raises."""
    try:
        return int(value)
    except (TypeError, ValueError):
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
    """
    if ctx.citation_index is None:
        return GateResult("citations", True, {"skipped": "no-index"})

    text = content or ""
    source = ctx.source_text or ""
    novel = novel_citations(text, source, ctx.citation_index)
    grounded_suspects = set(suspect_citations(source))
    suspects = [c for c in suspect_citations(text) if c not in grounded_suspects]

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
        raw_code = entry.get("code")
        code = resolve_code(raw_code) or str(raw_code or "").strip().upper()
        number = normalize_number(entry.get("number"))
        if not code or not number:
            malformed.append(repr(entry))
            continue
        wanted.append((code, number))
    return wanted, malformed


def _ref_matches(ref: SectionRef, code: str, number: str) -> bool:
    """Section identity at BASE-number granularity: BNS 103 and BNS 103(2) are
    the same section for gate purposes, so an answer key may name either. The
    letter suffix is never stripped (SectionRef.base_number guarantees that),
    so IPC 304B never matches IPC 304."""
    if ref.code != code:
        return False
    return ref.number == number or ref.base_number == number.split("(", 1)[0]


def _mentions_savings(answer: str, refs: list[SectionRef]) -> bool:
    if any(ref.code == "BNS" and ref.base_number == "358" for ref in refs):
        return True
    return bool(_SAVINGS_RE.search(answer or ""))


def check_answer_key(answer: str, ctx: GateContext) -> GateResult:
    """Transition-stream answers are checked against a known-good key: the
    stream exists precisely because the old/new-code answer is decidable in
    advance, and a generation that gets it wrong is wrong about the law
    (PERMANENT). Every other stream skips.

    A malformed key entry FAILS rather than being skipped. It is an operator
    bug, but silently skipping it means the transition example is never
    actually checked - and shipping an unchecked transition row is the exact
    outcome this gate exists to prevent, so it fails loudly and shows up in
    the per-gate counts immediately.

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

    both_required = bool(key.get("must_name_both_families"))
    families = sorted(
        {"old" for ref in refs if ref.code in OLD_CODES}
        | {"new" for ref in refs if ref.code in NEW_CODES}
    )
    both_ok = ({"old", "new"} <= set(families)) if both_required else True

    detail = {
        "governing_family": key.get("governing_family"),
        "cited": [str(ref) for ref in refs],
        "expected": [f"{code} {number}" for code, number in expected],
        "missing": missing,
        "forbidden_present": present_forbidden,
        "savings_required": savings_required,
        "savings_ok": savings_ok,
        "both_families_required": both_required,
        "families": families,
        "malformed_key_entries": malformed,
    }
    passed = not missing and not present_forbidden and savings_ok and both_ok and not malformed
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
        check_banned_meta(think, ctx),
        check_answer_key(answer, ctx),
    ]


def disposition(results: list[GateResult]) -> str | None:
    """None = clean. "reject" = wrong about the law, burn the seed.
    "regenerate" = badly written, ask again."""
    failed = [result.gate for result in results if not result.passed]
    if not failed:
        return None
    return "reject" if any(gate in PERMANENT_GATES for gate in failed) else "regenerate"
