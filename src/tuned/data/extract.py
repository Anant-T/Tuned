"""Turn the selected S.C.R. PDFs into text a segmenter can work on.

Input is `corpus/selection.jsonl` (select.py) joined to the PDFs acquire.py
landed; output is one plain-text file per judgment under `corpus/text/`,
a `document` row per judgment - INCLUDING the ones this module refuses to
emit - and `corpus/extraction.jsonl`, the manifest downstream reads.

THE HEADNOTE IS THE WHOLE PROBLEM
---------------------------------
P0 CHECK 3 established that the objects under `data/pdf/year=YYYY/english/`
are not the court's plain judgment release: they are the typeset SUPREME
COURT REPORTS reprint, and every one of them opens with the publisher's
editorial headnote - a legal-issue summary, enumerated `HELD:` points and a
Case Law Reference table - before the court's own words begin. That front
matter is the reporter's copyright under *Eastern Book Company v. D.B.
Modak*, not the court's uncopyrightable judgment, and it is also a SUMMARY
OF THE ANSWER: a headnote that survives into a training example both
infringes and leaks.

So the boundary between headnote and judgment is where this module is most
likely to be wrong, and the design follows from that:

    THE BOUNDARY IS VERIFIED, NOT MERELY FOUND.

Finding a marker is not the same as having found the right one. A cut is
accepted only if all four of these hold, and a document that fails any of
them is QUARANTINED - recorded with its reason and emitting no text at all -
rather than emitted in a maybe-clean state:

  * a marker was found                        (else no_judgment_start)
  * no editorial signature survives the cut   (else headnote_residue)
  * the cut did not discard most of the file  (else strip_too_large)
  * what is left is the size of a judgment    (else body_too_short)

The residue check is the load-bearing one. Cutting EARLY - at a marker-ish
line inside the front matter - is the failure that would otherwise be
silent, because the result still looks like a judgment; the residue check
turns it into a refusal. Cutting LATE is caught by the strip fraction. A
document nobody can segment confidently is worth far less than the cost of
the one that quietly poisons a dataset, so the refusal set is meant to be
non-empty and is reported as a rate.

RESUMABILITY IS THE DESIGN
--------------------------
This runs for hours over tens of thousands of documents and WILL be
interrupted, so it follows acquire.py's pattern rather than inventing a
second one: one three-outcome decision per object (extract / skip / redo),
taken against a document index read once per run, and the house durability
rule - bytes on disk BEFORE the row that points at them. A crash between the
two costs an index row and no work: the next run finds a key with no row and
extracts it again. Unlike acquire.py there is no "adopt", because re-parsing
one PDF is cheaper than hashing it to prove it is the same one.

A quarantine is a row too. Recording the refusal is what stops an
interrupted run re-attempting the same unsegmentable document forever, and
what lets `--audit` show the operator what was refused and why.

`extract_version` is the third resume input: the text on disk is a function
of these rules, so changing them has to invalidate the rows they produced.
Bump it when a cleanup or boundary rule changes; the next run re-extracts
what is stale and leaves the rest alone.

THE READER
----------
pymupdf4llm, per the plan (CPU-only layout module, handles multi-column and
header/footer, AGPL-3.0 accepted for offline dataset prep). It is behind ONE
function - `read_pdf_pages` - so the MIT fallback (pypdf + pdfplumber) is a
swap of that function and nothing else. Everything above it is pure text
over `list[str]`, one string per page, which is also why the whole module is
testable without a PDF.

Build:  python -m tuned.data.extract --config configs/data_law_v1.yaml
        [--limit N] [--force] [--audit N] [--selection PATH]
"""

import hashlib
import os
import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tuned.data.acquire import SC_SOURCE_ID, local_path_for
from tuned.data.select import SELECTION_FILENAME

# Bump when a cleanup or boundary rule changes: rows written under an older
# version are re-extracted, rows at this version are left alone.
EXTRACT_VERSION = 1

TEXT_DIRNAME = "text"
EXTRACTION_FILENAME = "extraction.jsonl"
PART_SUFFIX = ".part"

STATUS_OK = "ok"
STATUS_QUARANTINED = "quarantined"

# Why a document was refused. Each names the check that fired, because the
# operator's next move differs: no_text wants OCR (out of scope for v1),
# no_judgment_start wants a marker added, headnote_residue wants the
# boundary rules looked at.
Q_NO_TEXT = "no_text"
Q_LOW_TEXT_QUALITY = "low_text_quality"
Q_NO_JUDGMENT_START = "no_judgment_start"
Q_HEADNOTE_RESIDUE = "headnote_residue"
Q_STRIP_TOO_LARGE = "strip_too_large"
Q_BODY_TOO_SHORT = "body_too_short"
QUARANTINE_REASONS = (
    Q_NO_TEXT,
    Q_LOW_TEXT_QUALITY,
    Q_NO_JUDGMENT_START,
    Q_HEADNOTE_RESIDUE,
    Q_STRIP_TOO_LARGE,
    Q_BODY_TOO_SHORT,
)

MARKER_DELIVERED_BY = "judgment_delivered_by"
MARKER_HEADING = "judgment_heading"

# A text layer at all (a 2010-2025 object below this is a scan, and v1 ships
# no OCR).
MIN_DOC_CHARS = 400
# The smallest thing that can be a judgment. A one-line dismissal order is
# real, but it is not worth a training example and it is indistinguishable
# from a mis-cut.
MIN_BODY_CHARS = 1500
# Above this the "headnote" is most of the file, which no reprint's is - the
# marker matched something late instead.
MAX_STRIP_FRACTION = 0.8
# Latin letters as a share of non-space characters. English legal prose runs
# ~0.75-0.80; a (cid:NN) soup or a Devanagari page falls far below.
MIN_LATIN_RATIO = 0.5
# How far past the cut an editorial signature still means "cut too early".
# A headnote leaks at the SEAM; an incidental "HELD:" deep in a quotation is
# a different thing and is not this check's business.
RESIDUE_WINDOW = 3000

PAGE_SEPARATOR = "\n\n"


class ExtractionError(RuntimeError):
    """Something could not be extracted. Actionable by construction."""


# --------------------------------------------------------------------------
# Markdown demotion.
# --------------------------------------------------------------------------

# pymupdf4llm emits MARKDOWN: `**bold**` for emphasised runs, `#` headings
# for detected headers, `-----` rules. Downstream is a numbered-paragraph
# regex (P0's one corpus-wide Tier-1 signal), and `**1.**` does not match
# `^\d+\.`, so the decoration is removed here rather than fought with
# everywhere it could turn up.
_MD_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.M)
_MD_RULE = re.compile(r"^[ \t]{0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$", re.M)
_MD_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", re.DOTALL)
_MD_ITALIC_STAR = re.compile(r"(?<!\*)\*(?=\S)([^*\n]+?)(?<=\S)\*(?!\*)")
# Underscore emphasis ONLY when the underscores are not inside a word: the
# S.C.R. object stem is `2020_7_941_960_EN` and a bare "_" strip would eat it.
_MD_ITALIC_UNDER = re.compile(r"(?<![A-Za-z0-9_])_(?=\S)([^_\n]+?)(?<=\S)_(?![A-Za-z0-9_])")


def demote_markdown(text: str) -> str:
    """Markdown decoration out, characters in."""
    text = _MD_RULE.sub("", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_ITALIC_STAR.sub(r"\1", text)
    text = _MD_ITALIC_UNDER.sub(r"\1", text)
    return text


def _flat(line: str) -> str:
    """One line with its whitespace collapsed - the matching form."""
    return " ".join(line.split())


def _squash(line: str) -> str:
    """One line with ALL spacing removed, casefolded.

    Typeset headings are letter-spaced (`J U D G M E N T`), so the only
    reliable way to recognise one is to take the spacing out entirely.
    """
    return re.sub(r"\s+", "", line).casefold()


# --------------------------------------------------------------------------
# Per-page cleanup: signature stamps and margin letters.
# --------------------------------------------------------------------------

# P0 ranked the digital-signature side-stamp the #1 nuisance in these files.
# It is a rotated text block, so it lands in the flow as either one line or
# six, and both shapes are handled.
_SIGNATURE_LINES = (
    re.compile(r"digitally\s+signed\s+by", re.I),
    re.compile(r"^signature\s+not\s+verified\.?$", re.I),
    re.compile(r"^reason\s*:.{0,60}$", re.I),
    re.compile(r"^date\s*:\s*\d{4}[.\-/]\d{2}[.\-/]\d{2}", re.I),
    re.compile(r"^\d{4}[.\-/]\d{2}[.\-/]\d{2}\s+\d{2}:\d{2}:\d{2}"),
    re.compile(r"^\d{2}:\d{2}:\d{2}\s*(?:IST|UTC|GMT)?$", re.I),
)
# The bare stamp header, whose NEXT line is the signer's name. Only this
# shape licenses eating a following line: in the one-line form the next line
# is the judgment.
_SIGNATURE_HEADER = re.compile(r"^digitally\s+signed\s+by\s*:?$", re.I)
# The print-alignment letters P0 found running down the left margin of the
# reprints. A lettered SECTION heading ("A. FACTUAL MATRIX") carries text
# and is not this.
_MARGIN_LETTER = re.compile(r"^[A-H]\.?$")


def _is_signature(line: str) -> bool:
    flat = _flat(line)
    return any(pattern.search(flat) for pattern in _SIGNATURE_LINES)


def clean_page(text: str) -> tuple[str, dict]:
    """One page with its signature stamp and margin letters removed."""
    stats = {"signature": 0, "margin_letter": 0}
    out: list[str] = []
    drop_next = False
    for line in text.split("\n"):
        flat = _flat(line)
        if drop_next:
            drop_next = False
            if flat:
                # The signer's name, on the line after a bare stamp header.
                stats["signature"] += 1
                continue
        if _is_signature(line):
            stats["signature"] += 1
            drop_next = bool(_SIGNATURE_HEADER.match(flat))
            continue
        if _MARGIN_LETTER.match(flat):
            stats["margin_letter"] += 1
            continue
        out.append(line)
    return "\n".join(out), stats


# --------------------------------------------------------------------------
# Corpus-level cleanup: running headers, footers, watermarks.
# --------------------------------------------------------------------------

# A line has to be furniture on MOST pages before it is treated as furniture
# anywhere, and a document has to be long enough for "most pages" to mean
# something.
RUNNING_MIN_PAGES = 4
RUNNING_FRACTION = 0.6
RUNNING_MAX_CHARS = 100

_DIGITS = re.compile(r"\d+")
# Digit-blind matching exists for ONE thing: the printed page number, which
# is the only part of a running head that changes from page to page. Applied
# to a whole line of prose it is actively dangerous - "1. Paragraph 1 of the
# judgment..." and "2. Paragraph 2 of the judgment..." collapse to the same
# key, and a document whose every page opens with a numbered paragraph would
# have its body deleted as furniture. So it is allowed only on lines short
# enough to BE a head or a foot.
RUNNING_DIGIT_BLIND_CHARS = 60


def _running_key(line: str) -> str:
    """Match key for a repeated line: case folded, digits collapsed if short."""
    flat = _flat(line).casefold()
    skeleton = _DIGITS.sub("#", flat)
    return skeleton if len(skeleton) <= RUNNING_DIGIT_BLIND_CHARS else flat


def running_lines(
    pages: Sequence[str],
    *,
    min_pages: int = RUNNING_MIN_PAGES,
    fraction: float = RUNNING_FRACTION,
    max_chars: int = RUNNING_MAX_CHARS,
) -> frozenset[str]:
    """Keys of the lines that are page furniture: heads, feet, watermarks."""
    if len(pages) < min_pages:
        return frozenset()
    seen: dict[str, int] = {}
    for page in pages:
        keys = {
            _running_key(line)
            for line in page.split("\n")
            if line.strip() and len(_flat(line)) <= max_chars
        }
        for key in keys:
            seen[key] = seen.get(key, 0) + 1
    threshold = max(2, int(len(pages) * fraction + 0.999999))
    return frozenset(key for key, count in seen.items() if key and count >= threshold)


# --------------------------------------------------------------------------
# Corpus-level cleanup: footnotes.
# --------------------------------------------------------------------------

# A numbered judgment paragraph. Also, unavoidably, the shape of a footnote
# marker - which is why the two are told apart by the NUMBER and not by the
# pattern.
_PARA_LINE = re.compile(r"^\s{0,3}(\d{1,3})[.)]\s+\S")
_FOOTNOTE_LINE = re.compile(r"^\s{0,3}(\d{1,3})[.)]?\s+\S")
# What a footnote in a judgment nearly always is: a reference. Requiring it
# costs recall on the discursive ones and buys the guarantee that matters -
# a numbered PARAGRAPH is never moved out of the body.
_FOOTNOTE_HINT = re.compile(
    r"\(\d{4}\)|\[\d{4}\]|\bS\.?\s?C\.?\s?C\.?\b|\bS\.?\s?C\.?\s?R\.?\b|\bAIR\b|"
    r"\bibid\b|\bsupra\b|\bop\.\s?cit\b|\bv\.\s|\bvs\.?\s",
    re.I,
)
FOOTNOTE_WINDOW = 8
FOOTNOTE_MAX_SHARE = 0.25
FOOTNOTE_HEADING = "[FOOTNOTES]"


def split_footnotes(page: str, para_high: int) -> tuple[str, list[str], int]:
    """Split a page's footnote tail off its body.

    Returns (body, footnote lines, new paragraph high-water mark).

    A footnote marker and a numbered judgment paragraph are the same
    PATTERN, so they are told apart by the NUMBER: a marker counts as a
    footnote only when it is below the highest paragraph number that appears
    STRICTLY ABOVE it. Paragraph numbering ascends through a judgment, so a
    real paragraph never satisfies that (it is the number above's successor)
    while a footnote restarting at 1 always does - and a page that has
    established no paragraph numbering above the line yields nothing at all.

    Two further conditions, both there because the failure to avoid is
    taking a numbered PARAGRAPH out of the body: the block must read like a
    reference (_FOOTNOTE_HINT), and it must not be most of the page. Both
    cost recall on discursive footnotes, which is the right direction -
    leaving a footnote interleaved is a nuisance, deleting a paragraph is
    silent corruption of the one segmentation signal that works corpus-wide.
    """
    lines = page.split("\n")
    # highs[i] = the largest paragraph number strictly above line i.
    highs: list[int] = []
    for line in lines:
        highs.append(para_high)
        match = _PARA_LINE.match(line)
        if match:
            para_high = max(para_high, int(match.group(1)))

    last = len(lines) - 1
    while last >= 0 and not lines[last].strip():
        last -= 1
    if last < 0:
        return page, [], para_high

    start = None
    for i in range(max(0, last - FOOTNOTE_WINDOW + 1), last + 1):
        match = _FOOTNOTE_LINE.match(lines[i])
        if match and 0 < int(match.group(1)) < highs[i]:
            start = i
            break
    if start is None:
        return page, [], para_high
    for i in range(start, last + 1):
        # Nothing between the block's first marker and the foot of the page
        # may be the NEXT paragraph - that would make this a paragraph break,
        # not a footnote block.
        match = _PARA_LINE.match(lines[i])
        if match and int(match.group(1)) >= highs[i]:
            return page, [], para_high

    block = [line for line in lines[start:] if line.strip()]
    text = "\n".join(block)
    if not block or not _FOOTNOTE_HINT.search(text):
        return page, [], para_high
    if len(text) > max(1, len(page)) * FOOTNOTE_MAX_SHARE:
        # A "footnote block" that is most of the page is something else.
        return page, [], para_high
    return "\n".join(lines[:start]), block, para_high


@dataclass(frozen=True)
class CleanedPages:
    pages: tuple[str, ...]
    footnotes: tuple[tuple[int, str], ...]  # (page index, line)
    stats: dict


def clean_pages(pages: Sequence[str]) -> tuple[list[str], dict]:
    """Every cleanup pass, in the order P0 ranked the nuisances.

    Returns (pages, stats); `stats["footnote_lines"]` carries the moved
    lines with the page they came off, because whether a footnote survives
    depends on which side of the headnote boundary its page falls.
    """
    demoted = [demote_markdown(page or "") for page in pages]
    stats = {"signature": 0, "margin_letter": 0, "running": 0, "footnote": 0}

    staged: list[str] = []
    for page in demoted:
        cleaned, counts = clean_page(page)
        stats["signature"] += counts["signature"]
        stats["margin_letter"] += counts["margin_letter"]
        staged.append(cleaned)

    furniture = running_lines(staged)
    if furniture:
        deduped = []
        for page in staged:
            kept = []
            for line in page.split("\n"):
                if line.strip() and _running_key(line) in furniture:
                    stats["running"] += 1
                    continue
                kept.append(line)
            deduped.append("\n".join(kept))
        staged = deduped

    out: list[str] = []
    footnotes: list[tuple[int, str]] = []
    para_high = 0
    for index, page in enumerate(staged):
        body, moved, para_high = split_footnotes(page, para_high)
        if moved:
            stats["footnote"] += 1
            footnotes.extend((index, line) for line in moved)
        out.append(body)
    stats["footnote_lines"] = tuple(footnotes)
    return out, stats


# --------------------------------------------------------------------------
# The boundary.
# --------------------------------------------------------------------------

# "The Judgment of the Court was delivered by", and the variants the
# reporters use. Matched on the SQUASHED line, so spacing and punctuation
# cannot break it.
_DELIVERED_BY = re.compile(
    r"(?:following)?judgment(?:andorder|andopinion|&order)?ofthe(?:court|bench)"
    r"was(?:being)?delivered(?:by|by:)$|"
    r"delivered(?:the)?following(?:judgment|order)$"
)
# A heading, alone on its line. `J U D G M E N T` squashes to `judgment`.
_HEADING_WORDS = frozenset(
    {
        "judgment", "judgement", "order", "judgmentandorder", "judgement&order",
        "judgment&order", "judgmentandorder:", "judgment:", "order:",
    }
)
# The author line - "NAVIN SINHA, J." - which is a boundary hint and NEVER a
# boundary: it is also how a coram is printed in the editorial front matter,
# so cutting on it would cut inside the headnote. Reported on a quarantine
# so the operator can see what a rule change would buy.
_AUTHOR_LINE = re.compile(r"^[A-Z][A-Z.\s'()\-]{2,60},\s*(?:J\.?|JJ\.?|C\.?J\.?I?\.?)\s*:?$")


@dataclass(frozen=True)
class Boundary:
    offset: int
    marker: str
    line: str


def find_judgment_start(text: str) -> Boundary | None:
    """Where the court's own words begin, or None if that cannot be said.

    The FIRST marker wins: a concurring or dissenting opinion later in the
    same file carries its own heading, and taking a later match would drop
    the majority judgment.

    The marker line is KEPT (the offset points at its start, not past it).
    It is the court's text, not the reporter's, and it is the anchor the
    audit prints the seam around.
    """
    offset = 0
    for line in text.split("\n"):
        # Demoted PER LINE rather than over the whole text, because the
        # offset has to stay in the caller's coordinates. extract_text has
        # already demoted the page; this is what makes the matcher correct
        # for a caller that has not.
        plain = demote_markdown(line)
        flat = _flat(plain)
        squashed = _squash(plain)
        if squashed:
            if _DELIVERED_BY.search(squashed):
                return Boundary(offset, MARKER_DELIVERED_BY, flat)
            if squashed in _HEADING_WORDS:
                return Boundary(offset, MARKER_HEADING, flat)
        offset += len(line) + 1
    return None


def author_line_offset(text: str) -> int | None:
    """First line that looks like an authoring judge. Diagnostic only."""
    offset = 0
    for line in text.split("\n"):
        if _AUTHOR_LINE.match(_flat(line)):
            return offset
        offset += len(line) + 1
    return None


# --------------------------------------------------------------------------
# Editorial signatures.
# --------------------------------------------------------------------------

# The furniture of an S.C.R. headnote, in both the older typeset shape and
# the sectioned shape the newer volumes use. Anchored at a line start and
# (for HELD) case-sensitive, because the COURT writes "held" in prose all
# day - "this Court held that..." must not read as editorial furniture, or
# every judgment discussing a precedent is quarantined.
_SIGNATURES = (
    ("held", re.compile(r"^HELD\s*[:.]", re.M)),
    ("case_law_reference", re.compile(r"^\s*case\s+law\s+(?:reference|referred|cited)", re.I | re.M)),
    ("cases_referred_to", re.compile(r"^\s*cases?\s+referred\s+to\s*:?\s*$", re.I | re.M)),
    ("list_of_acts", re.compile(r"^\s*list\s+of\s+acts", re.I | re.M)),
    ("list_of_keywords", re.compile(r"^\s*list\s+of\s+keywords", re.I | re.M)),
    ("issue_for_consideration", re.compile(r"^\s*issues?\s+for\s+consideration", re.I | re.M)),
    ("headnotes", re.compile(r"^\s*headnotes?\s*:?\s*$", re.I | re.M)),
    ("case_arising_from", re.compile(r"^\s*case\s+arising\s+from", re.I | re.M)),
    ("appearances", re.compile(r"^\s*appearances?\s+for\s+parties", re.I | re.M)),
)


def headnote_signals(text: str) -> tuple[str, ...]:
    """Which pieces of editorial headnote furniture this text carries."""
    return tuple(name for name, pattern in _SIGNATURES if pattern.search(text))


# --------------------------------------------------------------------------
# Quality.
# --------------------------------------------------------------------------

_CID = re.compile(r"\(cid:\d+\)")
CID_LIMIT = 10


def latin_ratio(text: str) -> float:
    """Latin letters as a share of non-space characters (0.0 for nothing)."""
    letters = 0
    total = 0
    for char in text:
        if char.isspace():
            continue
        total += 1
        if char.isalpha() and "LATIN" in unicodedata.name(char, ""):
            letters += 1
    return letters / total if total else 0.0


# --------------------------------------------------------------------------
# The page span, off the object key.
# --------------------------------------------------------------------------

# P0 CHECK 2: `{year}_{volume}_{startpage}_{endpage}_EN.pdf`, and CHECK 3
# identified that pagination as the S.C.R.'s own.
_KEY_SPAN = re.compile(r"(?:^|/)(\d{4})_(\d{1,3})_(\d{1,6})_(\d{1,6})_EN\.pdf$", re.I)


@dataclass(frozen=True)
class PageSpan:
    year: int
    volume: int
    start: int
    end: int

    @property
    def pages(self) -> int:
        """Printed S.C.R. pages this judgment occupies."""
        return self.end - self.start + 1


def page_span_from_key(key: str) -> PageSpan | None:
    """The S.C.R. page span the object key encodes, or None.

    Worth recording (the brief left it to judgement): it is free here - the
    PDFs are on disk by definition, which is exactly why it could not be
    done at selection time - it gives downstream the citation's own page
    anchor, and printed span against PDF page count is an integrity signal
    that costs nothing to carry. It is NOT a filter: the convention is read
    off 70 sampled filenames, and a document whose key does not parse is
    extracted with no span rather than refused.
    """
    match = _KEY_SPAN.search(key.replace("\\", "/"))
    if not match:
        return None
    year, volume, start, end = (int(part) for part in match.groups())
    if end < start:
        return None
    return PageSpan(year, volume, start, end)


# --------------------------------------------------------------------------
# REPORTABLE.
# --------------------------------------------------------------------------

# Line 1 of the COURT-RELEASED judgment carries this flag. P0 raw-searched
# 70 objects of this bucket and found it in 2, so it is dead as a selection
# signal and select.py correctly ignores it - it is captured here because
# the page is open anyway, and it is worth nothing more than metadata.
REPORTABLE_LINES = 12
_NON_REPORTABLE = re.compile(r"^non[\s\-_]?reportable\b", re.I)
_REPORTABLE = re.compile(r"^reportable\b", re.I)


def reportable_flag(page_one: str) -> str | None:
    """"REPORTABLE" / "NON-REPORTABLE" / None, from the head of page one."""
    for line in page_one.split("\n")[:REPORTABLE_LINES]:
        flat = _flat(line)
        # NON- first: a substring search for "REPORTABLE" reads
        # "NON-REPORTABLE" as its own opposite.
        if _NON_REPORTABLE.match(flat):
            return "NON-REPORTABLE"
        if _REPORTABLE.match(flat):
            return "REPORTABLE"
    return None


# --------------------------------------------------------------------------
# Putting one document together.
# --------------------------------------------------------------------------

_BLANK_RUN = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.M)


def normalise_whitespace(text: str) -> str:
    text = _TRAILING_SPACE.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    return _BLANK_RUN.sub("\n\n", text).strip() + "\n"


@dataclass(frozen=True)
class Extraction:
    ok: bool
    reason: str | None = None
    text: str = ""
    chars: int = 0
    headnote_chars: int = 0
    marker: str | None = None
    pages: int = 0
    boundary_page: int | None = None
    signals: tuple[str, ...] = ()
    reportable: str | None = None
    footnotes: int = 0
    author_hint: float | None = None
    cleanup: dict = field(default_factory=dict)

    def meta(self) -> dict:
        return {
            "signals": list(self.signals),
            "reportable": self.reportable,
            "footnotes": self.footnotes,
            "boundary_page": self.boundary_page,
            "cleanup": self.cleanup,
            "author_hint": self.author_hint,
        }


def _page_of(offsets: Sequence[int], offset: int) -> int:
    page = 0
    for index, start in enumerate(offsets):
        if start <= offset:
            page = index
        else:
            break
    return page


def extract_text(pages: Sequence[str]) -> Extraction:
    """One document's pages -> the court's own words, or a refusal.

    The checks run in the order that makes the REASON the useful one: an
    unreadable text layer is diagnosed as such rather than blamed on a
    missing marker, and a boundary that discarded the judgment is reported
    as a bad cut rather than as a thin document.
    """
    n_pages = len(pages)
    reportable = reportable_flag(demote_markdown(pages[0])) if pages else None
    cleaned, stats = clean_pages(pages)
    footnote_lines = stats.pop("footnote_lines", ())

    offsets: list[int] = []
    cursor = 0
    for page in cleaned:
        offsets.append(cursor)
        cursor += len(page) + len(PAGE_SEPARATOR)
    joined = PAGE_SEPARATOR.join(cleaned)

    def no(reason: str, **over) -> Extraction:
        return Extraction(
            False,
            reason,
            pages=n_pages,
            reportable=reportable,
            signals=headnote_signals(joined),
            cleanup=stats,
            **over,
        )

    if len(joined.strip()) < MIN_DOC_CHARS:
        return no(Q_NO_TEXT)
    if len(_CID.findall(joined)) > CID_LIMIT or latin_ratio(joined) < MIN_LATIN_RATIO:
        return no(Q_LOW_TEXT_QUALITY)

    boundary = find_judgment_start(joined)
    if boundary is None:
        hint = author_line_offset(joined)
        return no(
            Q_NO_JUDGMENT_START,
            author_hint=None if hint is None else round(hint / len(joined), 4),
        )

    body = joined[boundary.offset :]
    residue = headnote_signals(body[:RESIDUE_WINDOW])
    if residue:
        # The cut landed INSIDE the front matter. Emitting this document
        # would ship the publisher's summary of the answer.
        return no(Q_HEADNOTE_RESIDUE, marker=boundary.marker, headnote_chars=boundary.offset)
    # SHORT BEFORE LARGE-STRIP, deliberately. When both hold there is
    # nothing to recover either way, and `strip_too_large` should keep its
    # alarming meaning: most of a document that still had plenty of text in
    # it was thrown away, which is a boundary bug and not a thin judgment.
    if len(body.strip()) < MIN_BODY_CHARS:
        return no(Q_BODY_TOO_SHORT, marker=boundary.marker, headnote_chars=boundary.offset)
    if boundary.offset / len(joined) > MAX_STRIP_FRACTION:
        return no(Q_STRIP_TOO_LARGE, marker=boundary.marker, headnote_chars=boundary.offset)

    boundary_page = _page_of(offsets, boundary.offset)
    # Footnotes are appended at the end of the document, so they travel
    # ACROSS the boundary unless they are filtered by the page they came
    # off. Strictly after the boundary page: the boundary page carries both
    # the tail of the headnote and the head of the judgment, and an
    # editorial reference is the one thing that must not come back.
    kept = [line for page, line in footnote_lines if page > boundary_page]
    if kept:
        body = body.rstrip() + "\n\n" + FOOTNOTE_HEADING + "\n" + "\n".join(kept)

    text = normalise_whitespace(body)
    return Extraction(
        True,
        None,
        text=text,
        chars=len(text),
        headnote_chars=boundary.offset,
        marker=boundary.marker,
        pages=n_pages,
        boundary_page=boundary_page,
        signals=headnote_signals(joined[: boundary.offset]),
        reportable=reportable,
        footnotes=len(kept),
        cleanup=stats,
    )


# --------------------------------------------------------------------------
# The reader seam. Nothing above this line imports a PDF library.
# --------------------------------------------------------------------------

def read_pdf_pages(path: str | Path) -> list[str]:
    """One string per page, via pymupdf4llm.

    The MIT fallback the plan names (pypdf + pdfplumber) is a replacement
    for THIS FUNCTION and nothing else - everything above it is text.
    """
    try:
        import pymupdf4llm
    except ImportError as exc:
        raise ExtractionError(
            "pymupdf4llm is needed to read the judgment PDFs and is not installed - "
            "run: pip install -e .[build]"
        ) from exc
    chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    return [chunk["text"] if isinstance(chunk, dict) else str(chunk) for chunk in chunks]
