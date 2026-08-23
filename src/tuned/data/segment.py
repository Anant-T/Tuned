"""One judgment's cleaned text -> the segments chunks.py packs into chunks.

Three tiers, tried in this order, and exactly this order because it is the
order P0 measured coverage in (see the module-level facts the task brief
carries, restated here so the priority is not just asserted):

  TOC       lettered section headings (`A. FACTUAL MATRIX`, `B. ISSUES`...).
            Measured at <=20% coverage across every year sampled, so it is
            never assumed - a document only gets this tier when its
            candidate headings are validated AGAINST THE DOCUMENT: strictly
            consecutive A, B, C... with no gaps or repeats, AND every
            inter-heading span independently confirmed to contain at least
            one real numbered paragraph (see _toc_segments). Three capital
            letters that happen to start a line are not a table of contents;
            a table of contents whose sections are hollow is not validated.
  ROLES     OpenNyAI's rhetorical-role model, reached through roles_infer.py's
            subprocess bridge. Tried only when a backend other than "none" is
            configured, and its failure - bridge unavailable, timeout,
            crash, or a clean run that simply returns no spans - degrades to
            packing rather than raising. The degradation is always recorded
            (see SegmentationResult.degradation), because "the roles tier
            silently did nothing" is exactly the failure mode a resumable,
            unsupervised, weeks-long pipeline cannot afford.
  PACKING   whole numbered paragraphs, monotonically increasing. Measured
            viable on 90-100% of documents - the workhorse, not a fallback of
            last resort - and it is the one tier that ALWAYS produces at
            least one segment for non-empty text, which is what lets
            `--roles-backend none` leave the whole pipeline functional.

WHICH NUMBERS ARE THIS DOCUMENT'S OWN. The raw signal P0 measured was
`^\\s*\\d{1,3}\\.\\s+` with no further filtering, and it needs one: a
judgment routinely quotes numbered material that is not its own - an earlier
report's paragraphs, the paragraphs of the judgment under appeal, a statute's
sections, a post-mortem report's injury list - and an un-filtered regex reads
those as boundaries inside whatever paragraph is doing the quoting.

The first cut of this module filtered them with a STRICTLY-INCREASING rule,
on the premise that "a quoted number is never higher than where the
surrounding prose already got to". That premise is false, and measurably so:
on the 15 staged PDFs it fails on five documents (a quoted paragraph `184.`
of the judgment under appeal at the quoting judgment's own paragraph 15, a
quoted section `178.` of a code, an "Article 335" whose number wrapped onto
the next line, a twenty-one item injury list, an isolated quoted `22.`).
When it fails it does BOTH harms at once: the citation becomes a boundary
AND, because the counter is now parked at the foreign number, every genuine
later paragraph is rejected - one document collapsed to a single chunk
holding 86% of itself. The counter-evidence had already been measured on
these same objects by extract.py's own `split_footnotes` (see its docstring,
which names three of these five shapes); it was not carried across.

WHAT REPLACES IT is a rule about RUNS rather than a running maximum. The
accepted boundaries are the highest-scoring chain of candidates, where

  CONTINUE  a candidate whose number exceeds SOME EARLIER ACCEPTED number by
            at most MAX_PARA_STEP scores +1. "Some earlier", not "the
            immediately preceding candidate", is the whole point: the
            document's own paragraph 16 continues its own paragraph 15 even
            when a quoted `122.` sits between them on the page.
  RESTART   any other jump scores +1 - RESTART_COST. A document really does
            restart its numbering (a concurring opinion beginning at 1
            again, an ORDER after a JUDGMENT), so this must be possible -
            but a restart has to pay for itself, which a two- or three-line
            citation cannot and a real second opinion easily can.

A quoted foreign run pays a restart to be entered AND another to be left, so
it is accepted only when it is long enough to be worth reading as numbering
in its own right (a twenty-one item injury list is; `178.`/`179.` is not).
When it IS accepted the items become boundaries between lines of a list,
which the packer merges straight back into one chunk - the harm the old rule
took is the one it could not undo, and that is the asymmetry this rule is
built around. Measured against the shipped rule on the same 15 documents:
94 -> 145 chunks, 9 -> 3 oversize, 75.5% -> 84.8% in band, worst chunk
26,818 -> 3,906 tokens.

Under-segmenting is still the safe direction by this module's contract - a
single oversize segment is what chunks.py's "never split inside a paragraph,
emit oversize alone and flag it" rule exists for - and nothing here ever
drops a byte: every tier's output is normalized into a gapless partition
below.

FOOTNOTES are cut off before paragraph-scanning rather than left to be
mis-numbered by it: extract.py appends any trailing footnote block after a
literal `[FOOTNOTES]` line, and a footnote marker restarting at "1." would
otherwise look exactly like this module's idea of a new paragraph. The tail
becomes its own segment (label "footnotes"), never dropped, never merged
into the numbered body.

TIER PRECEDENCE IS ABOUT BOUNDARIES, NEVER ABOUT THE TOKEN BAND. A ToC
section and a rhetorical role are both DOCUMENT-SCALE spans - one section of
a real judgment ran to 23,560 tokens - so a tier that emitted them as
segments would hand chunks.py material it is contractually forbidden to
split, and the tier with priority would produce WORSE chunks than the tier it
outranked. So the ToC and roles tiers contribute their boundaries and then
their spans are SUBDIVIDED at the packing tier's own paragraph starts
(_subdivide), carrying the section heading or role label onto every piece.
The tiers still decide where a chunk may begin; they never decide that a
chunk may be out of band. Whatever a tier claims, the resulting segment set
is a refinement of the packing tier's, so no tier can produce an oversize
chunk that packing would have avoided.

EVERY TIER'S OUTPUT IS NORMALIZED (_normalize_segments) before it leaves this
module: sorted, clipped against overlaps - forward against len(text) as well
as backward against the cursor - and gap-filled so the result is always a
GAPLESS, ORDERED partition of the whole document. That is what turns "never
truncated, never silently" from a rule about the packing tier into an
invariant the offset-audit and chunks.py's packer can both rely on
regardless of which tier produced the segments - including a future real
roles backend, whose span output this module does not otherwise control.
"""

import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass

from tuned.data import roles_infer
from tuned.data.extract import FOOTNOTE_HEADING

# Bump when a rule here changes what segments a document produces - recorded
# in every chunk's meta (chunks.py folds this in) so a document chunked under
# an older rule is recognisably stale even though nothing about EXTRACT_VERSION
# moved.
#
#   1  first cut: monotonic numbered-paragraph packing, ToC validated against
#      the document's own paragraph positions, OpenNyAI via the subprocess
#      bridge with a packing degrade on any failure.
#   2  the strictly-increasing filter replaced by the run-scoring rule above
#      (its premise was false on 5 of the 15 staged documents), and the ToC
#      and roles tiers subdivided at paragraph starts instead of emitting
#      document-scale spans. Both move where a chunk's boundaries fall.
SEGMENT_VERSION = 2

TIER_TOC = "toc"
TIER_ROLES = "roles"
TIER_PACKING = "packing"
TIERS = (TIER_TOC, TIER_ROLES, TIER_PACKING)

WHY_TOC = "toc validated"
WHY_ROLES = "roles available"
WHY_PACKING = "fallback"

FOOTNOTES_LABEL = "footnotes"

# A table of contents this short is indistinguishable from noise: P0 never
# saw fewer than three real sections in a document that had a genuine one at
# all (FACTUAL MATRIX / ISSUES / ANALYSIS at minimum), and two capital
# letters is exactly what a garbled margin-letter pass could still produce.
MIN_TOC_HEADINGS = 3


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    label: str | None

    def __post_init__(self):
        if self.end < self.start:
            raise ValueError(f"segment end {self.end} precedes start {self.start}")


@dataclass(frozen=True)
class SegmentationResult:
    tier: str
    why: str
    segments: tuple[Segment, ...]
    # None on TIER_TOC/TIER_ROLES (nothing degraded). Always a dict on
    # TIER_PACKING, naming why the roles tier specifically did not carry this
    # document - "roles_backend_none" when it was never configured, a
    # RolesBridgeError's `kind: message` when it was configured and failed,
    # or "no_role_spans" when it ran cleanly and had nothing to say. ToC's
    # own near-universal non-viability is not reported here: P0 already
    # measured that as the expected case, not a degradation of one.
    degradation: dict | None = None

    def __post_init__(self):
        # TIERS is the whole tier vocabulary. A result carrying anything else
        # would be counted under a tier no downstream report knows about.
        if self.tier not in TIERS:
            raise ValueError(f"unknown tier {self.tier!r}")


# --------------------------------------------------------------------------
# Footnote tail.
# --------------------------------------------------------------------------

_FOOTNOTE_MARK = re.compile(r"\n(" + re.escape(FOOTNOTE_HEADING) + r")\n")


def _split_footnote_tail(text: str) -> tuple[str, int | None]:
    """(body, footnote_start) - footnote_start is None if there is no tail.

    body == text[:footnote_start] exactly (when present), so body + the tail
    reconstructs `text` byte for byte; nothing between the two is discarded.
    """
    match = _FOOTNOTE_MARK.search(text)
    if match is None:
        if text.startswith(FOOTNOTE_HEADING + "\n"):
            return "", 0
        return text, None
    return text[: match.start(1)], match.start(1)


# --------------------------------------------------------------------------
# Tier: packing (numbered paragraphs).
# --------------------------------------------------------------------------

_PARA_START = re.compile(r"^[ \t]{0,3}(\d{1,3})[.)][ \t]+(?=\S)", re.M)

# _PARA_START's own \d{1,3}: the largest number it can ever match.
_MAX_PARA_NUMBER = 999

# How far a real next paragraph may step. Measured over every raw candidate
# on the 15 staged documents: +1 occurs 681 times, +2 nineteen times, +3
# five times, +4 once, and everything above that (+6, +7, +8, +10, +11, +22,
# +31, +135, +169, +296) is a citation, a statute section or a wrapped
# "Article NNN". Three is where the document's own numbering stops and
# somebody else's begins.
MAX_PARA_STEP = 3

# What a restart costs, in boundaries. A foreign numbered run pays this
# twice - once to be entered, once to be left - so it has to be longer than
# 2 * RESTART_COST to be worth accepting, while a real second opinion or a
# post-JUDGMENT ORDER pays it once against everything it then contributes.
# At 2, the measured citation runs (1, 1, 2, 3 and 3 candidates long) are all
# rejected and the measured genuine restarts (7 and 21 candidates) are all
# kept, with the nearest miss two candidates clear of the line either way.
RESTART_COST = 2


def paragraph_starts(text: str) -> list[tuple[int, int]]:
    """[(char_offset, paragraph_number), ...] - this document's own numbering.

    The highest-scoring chain of _PARA_START candidates under the CONTINUE /
    RESTART scoring the module docstring sets out. Ties break by the chain
    that spans the most of the document and then by the earliest start, so
    the result is a pure function of `text` - nothing here reads run state.

    Exposed (not a leading-underscore helper) because two other things read
    it: the ToC tier's validation (a heading's section is only real when it
    contains one of THESE positions, not any digit-dot-space match) and
    _subdivide, which cuts every tier's spans at exactly these offsets so no
    tier can escape the token band.
    """
    candidates = [(m.start(), int(m.group(1))) for m in _PARA_START.finditer(text)]
    if not candidates:
        return []

    # best_at[v] = (score, index) of the best chain ending on the NUMBER v so
    # far; best_any = the best chain ending anywhere so far. Both only ever
    # hold candidates already passed, because this loop runs in document
    # order - which is what makes "some earlier accepted number" cheap.
    best_at: list[tuple[int, int]] = [(0, -1)] * (_MAX_PARA_NUMBER + 1)
    best_any: tuple[int, int] = (0, -1)
    score = [0] * len(candidates)
    pred = [-1] * len(candidates)
    span_start = [0] * len(candidates)

    for j, (offset, number) in enumerate(candidates):
        best, source = 1, -1  # a chain that begins here, owing nothing
        for value in range(max(1, number - MAX_PARA_STEP), number):
            cand_score, cand_index = best_at[value]
            if cand_score and cand_score + 1 > best:
                best, source = cand_score + 1, cand_index
        any_score, any_index = best_any
        if any_score and any_score + 1 - RESTART_COST > best:
            best, source = any_score + 1 - RESTART_COST, any_index
        score[j], pred[j] = best, source
        span_start[j] = span_start[source] if source != -1 else offset
        if best > best_at[number][0]:
            best_at[number] = (best, j)
        if best > best_any[0]:
            best_any = (best, j)

    end = max(
        range(len(candidates)),
        key=lambda j: (score[j], candidates[j][0] - span_start[j], -span_start[j]),
    )
    chain: list[tuple[int, int]] = []
    while end != -1:
        chain.append(candidates[end])
        end = pred[end]
    chain.reverse()
    return chain


def _packing_segments(body: str) -> tuple[Segment, ...]:
    """Segments over `body` alone (the footnote tail is handled by the caller)."""
    if not body:
        return ()
    starts = paragraph_starts(body)
    if not starts:
        # No numbered-paragraph signal at all (a short order, a garbled
        # scan): the whole body is one segment. This is the branch that
        # makes packing NEVER refuse a non-empty document.
        return (Segment(0, len(body), None),)
    segments: list[Segment] = []
    if starts[0][0] > 0:
        segments.append(Segment(0, starts[0][0], None))
    for i, (offset, number) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(body)
        segments.append(Segment(offset, end, str(number)))
    return tuple(segments)


def _packing_tier(text: str) -> tuple[Segment, ...]:
    body, footnote_start = _split_footnote_tail(text)
    segments = list(_packing_segments(body))
    if footnote_start is not None:
        segments.append(Segment(footnote_start, len(text), FOOTNOTES_LABEL))
    return tuple(segments)


# --------------------------------------------------------------------------
# Tier: ToC (lettered section headings), validated against the document.
# --------------------------------------------------------------------------

# One capital letter, a period, a short Title/CAPS phrase, nothing after it
# on the line. Deliberately narrower than a bare `^[A-Z]\\.` - a heading
# carries WORDS, which is what tells it apart from extract.py's own margin
# letter (`^[A-H]\\.?$`, nothing else on the line) rather than colliding
# with the furniture that module already strips.
_TOC_HEADING = re.compile(r"^[ \t]{0,3}([A-Z])\.[ \t]+([A-Z][A-Za-z0-9 ,'&/\-]{2,80})[ \t]*$", re.M)


def toc_candidates(text: str) -> list[tuple[int, str, str]]:
    """[(offset, letter, heading text), ...] in document order. Exposed for
    the same reason paragraph_starts is - it is a signal on its own, not
    only an internal step of the validated tier."""
    return [(m.start(), m.group(1), m.group(2).strip()) for m in _TOC_HEADING.finditer(text)]


def _toc_segments(text: str, packing: Sequence[Segment]) -> tuple[Segment, ...] | None:
    """Validated ToC segments, or None when the candidates do not hold up.

    Two independent checks, both required:

      CONSECUTIVE   the candidate letters read A, B, C... in that exact
                    order with no gap and no repeat. Anything else is closer
                    to noise (a stray heading style, a garbled OCR margin
                    letter that slipped past extract.py's own stripping)
                    than to a real table of contents.
      NON-HOLLOW    every span between one heading and the next contains at
                    least one packing-tier paragraph start. A ToC whose
                    sections are hollow of the corpus-wide numbered-paragraph
                    signal is not validated against this document - it is
                    just three capitalised lines.
    """
    candidates = toc_candidates(text)
    if len(candidates) < MIN_TOC_HEADINGS:
        return None
    expected = ord("A")
    for _offset, letter, _heading in candidates:
        if ord(letter) != expected:
            return None
        expected += 1

    bounds = [c[0] for c in candidates] + [len(text)]
    para_starts = paragraph_offsets(packing)
    for i in range(len(candidates)):
        lo, hi = bounds[i], bounds[i + 1]
        if not any(lo <= p < hi for p in para_starts):
            return None

    segments: list[Segment] = []
    if candidates[0][0] > 0:
        segments.append(Segment(0, candidates[0][0], None))
    for i, (offset, _letter, heading) in enumerate(candidates):
        segments.append(Segment(offset, bounds[i + 1], heading))
    return tuple(segments)


# --------------------------------------------------------------------------
# Subdivision: a tier decides boundaries, never the token band.
# --------------------------------------------------------------------------


def paragraph_offsets(segments: Sequence[Segment]) -> list[int]:
    """The packing tier's own paragraph-start offsets, read off its segments.

    One derivation, three readers (ToC validation, _subdivide, and the tests
    that check the two agree) - the alternative is re-running the chain
    scoring per reader and hoping the three copies stay in step.
    """
    return [seg.start for seg in segments if seg.label not in (None, FOOTNOTES_LABEL)]


def _subdivide(segments: Sequence[Segment], offsets: Sequence[int]) -> tuple[Segment, ...]:
    """Cut each segment at the offsets strictly inside it, label unchanged.

    The result is a REFINEMENT of the input: same total span, same labels,
    every original boundary still a boundary. That is what lets a ToC
    section or a rhetorical role stay a real segmentation decision while
    chunks.py still sees paragraph-sized material to pack - and it is why no
    tier can emit an oversize chunk that the packing tier would have
    avoided, since every tier's segment set ends up a refinement of packing's.

    `offsets` must be ascending and `segments` must already be an ordered
    partition - both are, because the only caller passes _normalize_segments'
    output and paragraph_offsets read off it. Stated rather than defended,
    so a future caller knows what it owes rather than finding out from a
    non-monotonic segment list.
    """
    out: list[Segment] = []
    for seg in segments:
        cursor = seg.start
        for offset in offsets:
            if seg.start < offset < seg.end:
                out.append(Segment(cursor, offset, seg.label))
                cursor = offset
        out.append(Segment(cursor, seg.end, seg.label))
    return tuple(out)


# --------------------------------------------------------------------------
# Normalization: every tier's output becomes a gapless, ordered partition.
# --------------------------------------------------------------------------


def _normalize_segments(text: str, segments: Sequence[Segment]) -> tuple[Segment, ...]:
    """Sort, clip overlaps and fill gaps so the result always covers
    [0, len(text)) exactly once. Applied uniformly to every tier's raw
    output so "no byte silently dropped" is an invariant of this module
    rather than a promise each tier has to keep separately - load-bearing
    for a roles backend whose span output this file does not itself produce.

    Overlap resolution is FIRST-SEGMENT-WINS after the (start, end) sort: a
    later segment's start is clipped forward to the previous one's end. That
    is a deterministic, order-stable rule (ties broken by end, then by
    dict/list order being irrelevant post-sort) rather than a judgement
    about which segment is "more right".

    Clipping is BOTH ways. Backward against the cursor is the overlap rule
    above; forward against len(text) is the untrusted-input half, and it is
    not hypothetical - a span running past the end of the document would
    otherwise reach chunks.py as a chunk whose recorded `end`, `native_id`
    and content-derived `seed_id` all name bytes the document does not have,
    while `text[start:end]` silently returned fewer. A span starting past
    the end collapses to nothing at all and is dropped.
    """
    limit = len(text)
    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    out: list[Segment] = []
    cursor = 0
    for seg in ordered:
        start = min(max(seg.start, cursor), limit)
        end = min(max(seg.end, start), limit)
        if start > cursor:
            out.append(Segment(cursor, start, None))
        if end > start:
            out.append(Segment(start, end, seg.label))
        cursor = max(cursor, end)
    if cursor < limit:
        out.append(Segment(cursor, limit, None))
    return tuple(out)


# --------------------------------------------------------------------------
# The entry point.
# --------------------------------------------------------------------------


def segment_document(
    text: str,
    *,
    roles_backend: str = roles_infer.BACKEND_NONE,
    roles_python_bin: str | None = None,
    roles_timeout: float = roles_infer.DEFAULT_TIMEOUT_S,
    roles_spawn=subprocess.run,
) -> SegmentationResult:
    """One document's segments, tier-selected and normalized.

    `roles_backend` defaults to BACKEND_NONE, which never imports
    roles_infer's subprocess machinery beyond the module import itself and
    never spawns anything - the roles tier is opt-in per run, not per
    document, and its absence is exactly as functional as its presence,
    only recorded differently.
    """
    if not text:
        return SegmentationResult(
            tier=TIER_PACKING,
            why=WHY_PACKING,
            segments=(),
            degradation={"from": "text", "reason": "empty_text"},
        )

    packing = _packing_tier(text)
    offsets = paragraph_offsets(packing)

    toc = _toc_segments(text, packing)
    if toc is not None:
        return SegmentationResult(
            tier=TIER_TOC,
            why=WHY_TOC,
            segments=_subdivide(_normalize_segments(text, toc), offsets),
            degradation=None,
        )

    roles_reason = "roles_backend_none"
    if roles_backend != roles_infer.BACKEND_NONE:
        try:
            result = roles_infer.infer_roles(
                text,
                backend=roles_backend,
                python_bin=roles_python_bin,
                timeout=roles_timeout,
                spawn=roles_spawn,
            )
        except roles_infer.RolesBridgeError as exc:
            roles_reason = f"{exc.kind}: {exc}"
        else:
            if result.spans:
                role_segments = tuple(Segment(s.start, s.end, s.label) for s in result.spans)
                return SegmentationResult(
                    tier=TIER_ROLES,
                    why=WHY_ROLES,
                    segments=_subdivide(_normalize_segments(text, role_segments), offsets),
                    degradation=None,
                )
            roles_reason = "no_role_spans"

    return SegmentationResult(
        tier=TIER_PACKING,
        why=WHY_PACKING,
        segments=_normalize_segments(text, packing),
        degradation={"from": "roles", "reason": roles_reason},
    )
