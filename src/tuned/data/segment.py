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

WHY MONOTONIC. The raw signal P0 measured was `^\\s*\\d{1,3}\\.\\s+` with no
further filtering. This module accepts a match as a new paragraph boundary
only when its number is STRICTLY GREATER than the last accepted one. That is
a deliberate narrowing, not a re-measurement of P0's number: a judgment
routinely quotes an earlier report's own numbered paragraphs ("in XYZ, this
Court held: '15. The appellant contended...'"), and an un-filtered regex
reads that quoted "15." as a boundary inside whatever paragraph is doing the
quoting - splitting a real paragraph in half on a citation instead of the
text the court actually wrote at that point. Requiring strictly-increasing
numbers rejects exactly that case (a quoted number is never higher than
where the surrounding prose already got to) at the cost of under-segmenting
the rarer case of a document that restarts its own numbering partway through
(a second "ORDER" section numbered from 1 after a "JUDGMENT" section that
reached paragraph 40, say): that tail is swallowed into one large trailing
segment. Under-segmenting is safe by this module's own contract - a single
oversize segment is exactly what chunks.py's "never split inside a
paragraph, emit oversize alone and flag it" rule exists for - while
over-segmenting on a quotation is not: it would silently misrepresent a
citation as a chunk boundary. The real-data run this task's report carries
measures how often each of these actually happens on the 15 staged PDFs.

FOOTNOTES are cut off before paragraph-scanning rather than left to be
mis-numbered by it: extract.py appends any trailing footnote block after a
literal `[FOOTNOTES]` line, and a footnote marker restarting at "1." would
otherwise look exactly like this module's idea of a new paragraph. The tail
becomes its own segment (label "footnotes"), never dropped, never merged
into the numbered body.

EVERY TIER'S OUTPUT IS NORMALIZED (_normalize_segments) before it leaves this
module: sorted, clipped against overlaps, and gap-filled so the result is
always a GAPLESS, ORDERED partition of the whole document. That is what
turns "never truncated, never silently" from a rule about the packing tier
into an invariant the offset-audit and chunks.py's packer can both rely on
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
SEGMENT_VERSION = 1

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


def monotonic_paragraph_starts(text: str) -> list[tuple[int, int]]:
    """[(char_offset, paragraph_number), ...], strictly increasing in number.

    Exposed (not a leading-underscore helper) because the ToC tier's own
    validation reads it too - a heading's section is only real when it
    contains one of THESE positions, not any digit-dot-space match.
    """
    starts: list[tuple[int, int]] = []
    last: int | None = None
    for match in _PARA_START.finditer(text):
        number = int(match.group(1))
        if last is None or number > last:
            starts.append((match.start(), number))
            last = number
    return starts


def _packing_segments(body: str) -> tuple[Segment, ...]:
    """Segments over `body` alone (the footnote tail is handled by the caller)."""
    if not body:
        return ()
    starts = monotonic_paragraph_starts(body)
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
    the same reason monotonic_paragraph_starts is - it is a signal on its
    own, not only an internal step of the validated tier."""
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
    para_starts = [seg.start for seg in packing if seg.label not in (None, FOOTNOTES_LABEL)]
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
    """
    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    out: list[Segment] = []
    cursor = 0
    for seg in ordered:
        start = max(seg.start, cursor)
        end = max(seg.end, start)
        if start > cursor:
            out.append(Segment(cursor, start, None))
        if end > start:
            out.append(Segment(start, end, seg.label))
        cursor = max(cursor, end)
    if cursor < len(text):
        out.append(Segment(cursor, len(text), None))
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

    toc = _toc_segments(text, packing)
    if toc is not None:
        return SegmentationResult(
            tier=TIER_TOC, why=WHY_TOC, segments=_normalize_segments(text, toc), degradation=None
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
                    segments=_normalize_segments(text, role_segments),
                    degradation=None,
                )
            roles_reason = "no_role_spans"

    return SegmentationResult(
        tier=TIER_PACKING,
        why=WHY_PACKING,
        segments=_normalize_segments(text, packing),
        degradation={"from": "roles", "reason": roles_reason},
    )
