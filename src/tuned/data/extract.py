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
Case Law Reference table - before the court's own words begin. A smoke over
15 real bucket objects spread across 2010-2025 found the headnote on 15 of
15, so this is a property of the bucket and not of a sample.

WHY THAT FRONT MATTER MAY NOT BE SHIPPED, stated as the operative rule
rather than the famous case. The S.C.R. headnote is prepared by the
Supreme Court's own Editorial Section - the 2023+ volumes print the
editor's name under it ("Headnotes prepared by: ...") - which makes it a
GOVERNMENT WORK under s.2(k) of the Copyright Act with copyright vesting in
the Government under s.17(d). The s.52(1)(q)(iv) exemption that puts
judgments in the public domain covers "any judgment or order of a court",
and the reporter's summary of one is not the judgment. *Eastern Book
Company v. D.B. Modak* is BACKGROUND and not the authority here: it is
about a PRIVATE reporter's copyright in its own headnotes and it is cited
for why editorial matter attracts copyright at all, not for who owns this
particular headnote.

And it is also a SUMMARY OF THE ANSWER: a headnote that survives into a
training example both infringes and leaks.

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
silent, because the result still looks like a judgment. Cutting LATE is
caught by the strip fraction. A document nobody can segment confidently is
worth far less than the cost of the one that quietly poisons a dataset, so
the refusal set is meant to be non-empty and is reported as a rate.

WHAT THE RESIDUE CHECK ACTUALLY GUARANTEES, stated narrowly because the
wider claim that used to stand here was false. It is a union of FOUR rules,
and an early cut is refused when it leaves the evidence any one of them
reads:

  1  SEAM      a signature within RESIDUE_WINDOW past the cut;
  2  COMPARE   a signature name the removed head cannot account for,
                anywhere in the file (including at its END, which no
                forward window can reach);
  3  BLOCK     the cut did not BEGIN a block of text, while furniture was
                removed - a heading is set off, a wrapped line is not;
  4  ENUM      the body opens on the item after the last one the removed
                head reached - the two sides read as one list.

Rules 1 and 2 read the SIGNATURES and 3 and 4 read the CUT, and that split
is the answer to a demonstrated failure: a real headnote carries ONE `HELD:`
with numbered points under it, so a cut inside that block satisfies both
signature rules at once - the name is accounted for on the removed side and
nothing repeats the label past the cut - and the document is emitted with
the publisher's holding on top of it.

The residual is what is left when all four are silent: an editorial
continuation none of the four rules can read - unnumbered, LETTERED (`(a)`),
ROMAN (`(ii)`), or numbered NON-CONSECUTIVELY, or whose next number falls
past SEAM_ENUM_WINDOW - running on past a properly set-off cut with no
signature anywhere after it. Rule 4 is EXACT-SUCCESSOR, so it reads only the
one continuation that counts up by one, and the width of that residual is
the deliberate price: a rule refusing every body whose first number merely
EXCEEDS the head's last would also refuse the judgment whose opening
paragraphs the reader dropped, or which numbers from a different base. The
whole family is still EMITTED, and `signals` will name the furniture that
WAS removed, so the first-run tell below does not fire on any of it. It is
this module's sharpest known residual.

The guard also reads the furniture in every rendering the PDF reader can
produce (a table bar, a blockquote marker, a bullet, a heading hash,
letter-spaced capitals - across a line wrap or not - a mid-line wrap, a
soft-hyphenated word, an ordered-list heading). That is the answer to the
same discovery: a guard that reads one rendering does not REFUSE the
documents it cannot read - it EMITS them, headnote and all. So on an emitted
document `signals` is the signature set of the whole CLEANED document, and

    `--audit` printing `headnote signals: none` against a document that
    visibly HAS a headnote means the guard is blind to this reporter's
    typesetting - and then every `ok` document in that run is suspect.

That is the cheapest check available on run one and it is printed at the top
of every audit. It is a check on RECOGNITION and not on placement: it says
nothing about the residual two paragraphs up, where the furniture IS
recognised and `signals` reads healthy.

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
import json
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
#
#   2  the headnote guard reads the furniture in every rendering the reader
#      can emit, and compares the two halves of the document instead of only
#      looking forward from the cut. Both change which documents are emitted
#      AND (through demotion of table bars, blockquote markers and bullets)
#      the text of the ones that are, so every version-1 row is stale.
#   3  the residue check reads the CUT as well as the signatures (a cut that
#      begins no block, or that continues the removed head's numbering, is
#      refused), and the signature set grew four renderings - the bare and
#      title-case `HELD` labels, the soft-hyphenated heading, the numbered
#      section heading and the letter-spaced heading that wrapped. Both move
#      the emitted SET; neither moves the text of a document that is still
#      emitted, so a version-2 row is stale in its verdict and not in its
#      bytes. Re-extraction is still the only way to find that out.
#   4  FIRST CONTACT WITH THE REAL OBJECTS, and every version-3 row is stale
#      in its verdict AND in its bytes. The reader is pinned to one lane
#      (versions 1-3 could not read a PDF at all, and the lane the library
#      would have chosen gives different verdicts); the demoter strips the
#      inline HTML the reader really emits, which is what was hiding
#      `Case Law Reference` and the `Judgment` heading; footnote splitting
#      no longer carries a paragraph high-water mark across pages, which is
#      what was moving real paragraphs to the foot of the file; the running-
#      head pass fires at all; two marker phrasings and one `$`-anchored
#      heading are recognised; and two structural quarantines are new. The
#      row also now records WHICH READER made it, because the verdict is a
#      function of that too.
EXTRACT_VERSION = 4

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
# The two that read the OBJECT rather than its text - see PdfStructure.
# scanned_era wants the OCR-era decision taken deliberately; mojibake_font
# wants a document whose text layer is undecodable kept out of the corpus
# even though it reads as clean Latin.
Q_SCANNED_ERA = "scanned_era"
Q_MOJIBAKE_FONT = "mojibake_font"
QUARANTINE_REASONS = (
    Q_NO_TEXT,
    Q_LOW_TEXT_QUALITY,
    Q_NO_JUDGMENT_START,
    Q_HEADNOTE_RESIDUE,
    Q_STRIP_TOO_LARGE,
    Q_BODY_TOO_SHORT,
    Q_SCANNED_ERA,
    Q_MOJIBAKE_FONT,
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
# How far into the body an opening enumerator still counts as "this is where
# the body starts". Past this the first number on a line is a quotation or a
# statutory clause, not the seam.
SEAM_ENUM_WINDOW = 1200

# A page break is joined with a plain newline, NOT a blank line. A judgment
# runs on across pages, so a blank line there would announce a paragraph
# break in the middle of a sentence at every page boundary - and the Tier-3
# packer downstream reads blank lines. Inside a page the text is already
# hard-wrapped, so a page break now looks exactly like the line breaks
# around it.
PAGE_SEPARATOR = "\n"


class ExtractionError(RuntimeError):
    """Something could not be extracted. Actionable by construction."""


# --------------------------------------------------------------------------
# Markdown demotion.
# --------------------------------------------------------------------------

# pymupdf4llm emits MARKDOWN: `**bold**` for emphasised runs, `#` headings
# for detected headers, `-----` rules, `|cell|cell|` for detected TABLES,
# `>` for indented blocks and `-` for lists. Downstream is a numbered-
# paragraph regex (P0's one corpus-wide Tier-1 signal), and `**1.**` does not
# match `^\d+\.`, so the decoration is removed here rather than fought with
# everywhere it could turn up.
#
# It is also, and more importantly, what the HEADNOTE GUARD reads. Every
# editorial signature below is anchored at a line start, so a `|` or a `>` in
# front of "Case Law Reference" is the difference between refusing a document
# and publishing the reporter's headnote. Demotion is therefore not
# cosmetic: it is the guard's eyesight, and it must strip every decoration
# that can sit BEFORE the first word of a line.
_MD_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.M)
_MD_RULE = re.compile(r"^[ \t]{0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$", re.M)
_MD_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", re.DOTALL)
_MD_ITALIC_STAR = re.compile(r"(?<!\*)\*(?=\S)([^*\n]+?)(?<=\S)\*(?!\*)")
# Underscore emphasis ONLY when the underscores are not inside a word: the
# S.C.R. object stem is `2020_7_941_960_EN` and a bare "_" strip would eat it.
_MD_ITALIC_UNDER = re.compile(r"(?<![A-Za-z0-9_])_(?=\S)([^_\n]+?)(?<=\S)_(?![A-Za-z0-9_])")
# A table's own decoration. The S.C.R. headnote's Case Law Reference block IS
# a grid, so it is exactly what the reader's table detection turns into
# `|pipes|` - and a `|` in column 0 is enough to hide the single most
# recognisable piece of front matter in the corpus from a `^`-anchored rule.
# The separator row (`|---|---|`) carries no text and goes; a content row
# keeps its cells and loses its bars. A line with two or more bars is treated
# as a row as well, because which of the two shapes this reader emits is a
# property of a library version nobody here has run.
_MD_TABLE_RULE = re.compile(r"^[ \t]{0,3}\|[ \t:|-]*$", re.M)
_MD_TABLE_ROW = re.compile(r"^(?:[ \t]{0,3}\|.*|.*\|.*\|.*)$", re.M)
# Blockquote and list markers, for the same reason and nothing more: both sit
# in front of the first word of a line.
_MD_QUOTE = re.compile(r"^[ \t]{0,3}(?:>[ \t]?)+", re.M)
_MD_BULLET = re.compile(r"^[ \t]{0,3}[-*+][ \t]+", re.M)

# THE READER DOES NOT EMIT ONLY MARKDOWN. Measured over 15 real objects:
# 886 `<u>`, 222 `<br>`, 164 `<sup>` and 10 `<mark>`, and the damage was not
# cosmetic. `<u>Case Law Reference</u>` defeats a `^`-anchored signature in
# five documents; `<u>Judgment</u>` squashes to `<u>judgment</u>`, which is
# not in _HEADING_WORDS, and quarantined ALL THREE 2025 documents as
# `no_judgment_start` on a heading that is plainly there. So an inline tag
# sits in front of the first word of a line exactly the way a table bar
# does, and it is stripped for the same reason and in the same place.
#
# The list is the tags an inline typesetting run can arrive as - the four
# above are the ones this corpus produced, the rest are the same class of
# thing and cost nothing to name. BLOCK tags are deliberately absent: this
# is a decoration stripper, not an HTML parser, and a reader that started
# emitting `<div>`/`<table>` would be a reader change to look at rather than
# to absorb silently.
_HTML_INLINE_TAG = re.compile(
    r"</?(?:u|b|i|em|strong|sup|sub|mark|small|span|font)\b[^>]*>", re.I
)
# `<br>` is the one that carries meaning: it is a LINE BREAK inside a cell,
# so deleting it glues two words together (`Sub<br>Inspectors` ->
# `SubInspectors`, measured) while turning it into a newline would tear a
# `|cell|cell|` row in half before _MD_TABLE_ROW can read it. A space is the
# rendering that loses neither.
_HTML_BREAK = re.compile(r"<br\b[^>]*>", re.I)


def demote_markdown(text: str) -> str:
    """Markdown AND inline-HTML decoration out, characters in.

    Order matters three times: the tags come off first because a `|` row and
    a `^` anchor are both read past them; a quoted rule (`> ---`) is only a
    rule once the quote marker is gone; and a rule (`- - -`) is only
    distinguishable from a bullet before the bullet is stripped.
    """
    text = _HTML_BREAK.sub(" ", text)
    text = _HTML_INLINE_TAG.sub("", text)
    text = _MD_TABLE_RULE.sub("", text)
    text = _MD_TABLE_ROW.sub(lambda m: m.group(0).replace("|", " "), text)
    text = _MD_QUOTE.sub("", text)
    text = _MD_RULE.sub("", text)
    text = _MD_BULLET.sub("", text)
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


def _despace(text: str) -> str:
    """Every line with its spacing removed, one line per line, CASE KEPT.

    The same answer `_squash` gives the boundary matcher, given to the
    headnote guard - which needs it for the same reason (`H E L D :` and
    `C A S E  L A W  R E F E R E N C E` are how a letter-spaced heading
    extracts) and needs the CASE with it, because case is the only thing
    that tells the reporter's `HELD` from the court's `held`.
    """
    return "\n".join("".join(line.split()) for line in text.split("\n"))


def _despace_pairs(text: str) -> str:
    """`_despace`, with every line also joined to the one after it.

    A letter-spaced heading is set at whatever width the printed column was,
    so `C A S E   L A W   R E F E R E N C E` arrives as TWO lines whenever the
    column was narrower than the heading - and the per-line form then reads
    two halves of a heading and recognises neither. Joining consecutive PAIRS
    lets a heading cross one wrap while keeping the `^` anchor that stops
    "list of acts" matching a sentence about one: a signature wholly inside
    one line still begins a pair. (`$`-anchored signatures are read off the
    per-line form; a pair has the next line stuck to its end by design.)

    The LAST line has no successor and is not carried here: a lone `l[-1]`
    entry would only repeat what `_despace` gives the same caller, which is
    why mutating that tail away changed nothing and it is gone.
    """
    lines = ["".join(line.split()) for line in text.split("\n")]
    return "\n".join(a + b for a, b in zip(lines, lines[1:]))


# A word broken across a line by the typesetter's soft hyphen. `Case Law
# Refer-\nence` is the same heading as `Case Law Reference`, and it is only
# ever joined for MATCHING - the emitted text keeps what the reader gave it,
# because guessing which hyphens were the author's is a different problem.
_SOFT_HYPHEN = re.compile(r"-[ \t]*\n[ \t]*")


def _dehyphen(text: str) -> str:
    """Every soft-hyphen line break closed up, for the matching form only."""
    return _SOFT_HYPHEN.sub("", text)


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
#
# MEASURED, on the pinned reader, over 15 objects spread across 2010-2025:
# 8 to 562 own-line margin letters per document on the twelve pre-2023
# objects and ZERO on the three 2023+ ones, which set their headnote in a
# labelled layout with no print-alignment column. So the own-line model is
# right for this reader, and `dropped: 0 margin` on a 2023+ document is the
# document and not the rule.
_MARGIN_LETTER = re.compile(r"^[A-H]\.?$")
# The same letter INLINED into the line beside it, which is how it arrives
# on a reader that lays the page out in columns rather than emitting the
# margin as its own block. Measured on the layout reader this module now
# refuses: one document's `headnote signals` printed EMPTY over text that
# plainly reads `C Held:` - the audit's own first-run alarm, firing for real
# and reading like a healthy document. A reader upgrade is exactly how that
# would arrive silently, so the guard reads past the letter.
#
# It is taken off the MATCHING forms and never off the emitted text. `A` is
# an English word: dropping the letter from a line that really begins "A
# person who..." would eat it, while a matching form that reads "person
# who..." simply fails to match, as it did before.
_INLINE_MARGIN_LETTER = re.compile(r"^[A-H][ \t]+(?=\S)", re.M)


def _unmargin(text: str) -> str:
    """Every line with an inlined print-alignment letter off its front."""
    return _INLINE_MARGIN_LETTER.sub("", text)


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

# A line has to be furniture on ENOUGH pages before it is treated as
# furniture anywhere, and a document has to be long enough for "enough pages"
# to mean something.
#
# 0.4, AND THE OLD 0.6 WAS STRUCTURALLY UNREACHABLE. A law report is printed
# recto/verso: the left-hand page carries `NNN SUPREME COURT REPORTS [YYYY] V
# S.C.R.` and the right-hand page carries the case name, so NEITHER head can
# appear on more than about half the pages and a threshold above 0.5 refuses
# both by construction. Measured before the change: running heads survived in
# ALL TEN emitted bodies, up to 31 in one, and `dropped: N running` read 0 on
# nine of fifteen documents - a pass that never fired, reporting success.
RUNNING_MIN_PAGES = 4
RUNNING_FRACTION = 0.4
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
# The rest of the key, and it exists because the fraction alone did not fix
# the pass. The 2010-2017 volumes are scans with an OCR text layer, and the
# ONE line that repeats on every left-hand page arrives under several
# spellings of the same thing: `[2010]`, `[2010)`, `[201 O]`, `(201 O]`,
# `[201 OJ]` - the year's `0` read as the letter `O`, the bracket read as
# `J`, the space the OCR put inside the year. Digit-blinding alone leaves
# those as five different keys, none of which reaches any threshold. So the
# key drops everything that is not a letter or the digit placeholder, folds
# `O` onto the placeholder, and collapses a run of placeholders to one - at
# which point `[2010]` and `[201 O]` are the same key and the head is
# furniture again. (Case is folded by upper-casing here rather than
# case-folding: the key is compared only against other keys.)
_RUNNING_NOISE = re.compile(r"[^A-Z#]+")
_RUNNING_PLACEHOLDER_RUN = re.compile(r"#+")
# ...with ONE line that is never furniture however often it repeats. A bare
# paragraph enumerator (`48.`) is the Tier-1 segmentation signal P0 found,
# and the hard key above cannot tell it from a printed page number (`923`)
# once the punctuation is gone - both are `#`. Deleting a page number is
# free; deleting the paragraph anchor is the corruption this module exists
# to avoid, so the shape is excluded from furniture rather than ranked
# against a threshold.
_BARE_ENUMERATOR = re.compile(r"^\(?\d{1,3}[.)]$")
# ...and the hard key is for a line with WORDS in it. A line with no letters
# has no spelling for a scanner to disagree about, and folding its
# punctuation away would put `1974)` - the tail of a citation the reader
# broke onto its own line, measured - under the same key as every printed
# page number in the volume, which is enough of them to cross any threshold.
# So a numbers-only line keeps the plain digit skeleton it always had.
_HAS_LETTER = re.compile(r"[A-Za-z]")


def _running_key(line: str) -> str:
    """Match key for a repeated line, or "" for a line that is never furniture.

    Case folded, and on a line short enough to BE a head or a foot also
    digit-blind - plus, where the line has words in it, letter-`O`-blind and
    stripped of the punctuation the scanners disagree about.
    """
    flat = _flat(line)
    if _BARE_ENUMERATOR.match(flat):
        return ""
    skeleton = _DIGITS.sub("#", flat)
    if len(skeleton) > RUNNING_DIGIT_BLIND_CHARS:
        return flat.casefold()
    if not _HAS_LETTER.search(skeleton):
        return skeleton
    hard = _RUNNING_NOISE.sub("", skeleton.upper().replace("O", "#"))
    return _RUNNING_PLACEHOLDER_RUN.sub("#", hard)


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
# A block that trails off in mid-sentence. The last line of a note ENDS
# something - a citation's year or page number, a full stop - while the last
# line of a paragraph the page break interrupted ends on a lower-case word or
# a comma and continues overleaf. See split_footnotes.
_UNFINISHED_TAIL = re.compile(r"[a-z,]$")


def split_footnotes(page: str) -> tuple[str, list[str]]:
    """Split a page's footnote tail off its body.

    Returns (body, footnote lines).

    A footnote marker and a numbered judgment paragraph are the same
    PATTERN, so they are told apart by the NUMBER: a marker counts as a
    footnote only when it is below the highest paragraph number that appears
    STRICTLY ABOVE IT ON THIS PAGE. Paragraph numbering ascends through a
    judgment, so a real paragraph never satisfies that (it is the number
    above's successor) while a footnote restarting at 1 always does - and a
    page that has established no paragraph numbering above the line yields
    nothing at all.

    ON THIS PAGE is the correction, and it is the whole of the first of two
    fixes. This mark used to be threaded across the document and never
    decreased, which made "below the highest paragraph number" mean "below
    the highest number ANYWHERE ABOVE, on any page" - so once one line
    inflated it, every number under it became footnote-eligible for the rest
    of the file. Measured on 15 real objects, three things inflated it and
    none of them was a paragraph: a wrapped sentence whose next line opened
    `335. The several...` (the number belongs to `Article 335` on the line
    above), a quoted statute (`178. Place of inquiry or trial.-`) and a
    quoted paragraph of the judgment under appeal (`122. The details...`).
    Paragraph 48 of one judgment and half a sentence of another were carried
    to the foot of the file under `[FOOTNOTES]` - the silent corruption this
    function's own docstring names. Per page, the mark cannot travel: a
    quoted `8.` is only a footnote if THIS page numbered past 8 above it.

    Three further conditions, all there because the failure to avoid is
    taking a numbered PARAGRAPH out of the body: the block must read like a
    reference (_FOOTNOTE_HINT), it must not be most of the page, and it must
    not TRAIL OFF - a block whose last line ends on a lower-case word or a
    comma is a sentence the page break interrupted, not a note. That last
    one is the second fix, and it is not redundant with the first: the case
    it caught (a quoted paragraph `8.` under a real paragraph `72.` on the
    SAME page, severed at "...lodging the amount in Court, unless") is one
    per-page numbering cannot see. All three cost recall on discursive
    footnotes, which is the right direction - leaving a footnote interleaved
    is a nuisance, deleting a paragraph is silent corruption of the one
    segmentation signal that works corpus-wide.
    """
    lines = page.split("\n")
    # highs[i] = the largest paragraph number strictly above line i, ON THIS
    # PAGE. It starts at zero for every page, which is what stops one bad
    # line poisoning the rest of the document.
    para_high = 0
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
        return page, []

    start = None
    for i in range(max(0, last - FOOTNOTE_WINDOW + 1), last + 1):
        match = _FOOTNOTE_LINE.match(lines[i])
        if match and 0 < int(match.group(1)) < highs[i]:
            start = i
            break
    if start is None:
        return page, []
    for i in range(start, last + 1):
        # Nothing between the block's first marker and the foot of the page
        # may be the NEXT paragraph - that would make this a paragraph break,
        # not a footnote block.
        match = _PARA_LINE.match(lines[i])
        if match and int(match.group(1)) >= highs[i]:
            return page, []

    block = [line for line in lines[start:] if line.strip()]
    text = "\n".join(block)
    if not block or not _FOOTNOTE_HINT.search(text):
        return page, []
    if len(text) > max(1, len(page)) * FOOTNOTE_MAX_SHARE:
        # A "footnote block" that is most of the page is something else.
        return page, []
    if _UNFINISHED_TAIL.search(block[-1].strip()):
        # The sentence runs on to the next page: this is the body, cut by
        # the page break.
        return page, []
    return "\n".join(lines[:start]), block


def clean_pages(pages: Sequence[str]) -> tuple[list[str], dict]:
    """Every cleanup pass, in the order P0 ranked the nuisances.

    Returns (pages, stats); `stats["footnote_lines"]` carries the moved
    lines with the page they came off, because whether a footnote survives
    depends on which side of the headnote boundary its page falls.
    """
    demoted = [demote_markdown(page or "") for page in pages]
    stats = {"signature": 0, "margin_letter": 0, "running": 0, "footnote_pages": 0}

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
    for index, page in enumerate(staged):
        body, moved = split_footnotes(page)
        if moved:
            stats["footnote_pages"] += 1
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
#
# TWO OF THOSE VARIANTS WERE MISSING, both found in the 2014 volumes and
# both quarantined honestly as `no_judgment_start` until now:
#
#   "The order of the Court was delivered by"       - no "judgment" token
#   "The Judgments of the Court were delivered by"  - plural, and "were"
#
# The second is how a reprint announces separate opinions, so the token that
# has to move is the VERB as well as the noun. What holds the pattern
# together is the rest of it - `of the Court/Bench ... delivered by` at the
# end of the line - which is why widening the noun costs nothing: prose
# about an order of the court does not end on "delivered by".
_DELIVERED_BY = re.compile(
    r"(?:following)?(?:judgments?|orders?)(?:andorder|andopinion|&order)?"
    r"ofthe(?:court|bench)(?:was|were)(?:being)?delivered(?:by|by:)$|"
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
            # The heading is an EXACT match against a small set, so a margin
            # letter inlined in front of it (`C JUDGMENT`) is not a near
            # miss - it is a miss. _DELIVERED_BY needs no such help: it is
            # searched, not matched, and anchored at the line's END.
            if squashed in _HEADING_WORDS or _squash(_unmargin(flat)) in _HEADING_WORDS:
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
# the sectioned shape the newer volumes use. Each is matched TWICE - once on
# the demoted text and once on the same text with its spacing removed - and
# the second form is not a nicety: a letter-spaced heading is how these
# reprints print a heading, and a guard that cannot read one is a guard that
# EMITS the headnote under it.
#
# `HELD` is case-sensitive and NOT anchored to a line start. Casing is the
# discriminator that matters - the COURT writes "held" in prose all day, and
# "this Court held that..." must never read as editorial furniture or every
# judgment discussing a precedent is quarantined - while POSITION is not a
# discriminator at all: the text is hard-wrapped at whatever width the column
# was, so the reporter's `HELD:` lands mid-line whenever the sentence before
# it did not end at the margin. The `^` was buying nothing and hiding that
# case. Everything else stays line-anchored: those are headings, and an
# unanchored "list of acts" would match a sentence about one.
# The newer volumes NUMBER their front-matter sections, and the reader puts
# the enumerator in front of the heading exactly as printed - so a bare `^`
# reads `1. Issue for Consideration` as prose. An ordered-list heading is a
# heading.
_ENUM_HEADING = r"[ \t]*(?:\(?\d{1,3}[.)][ \t]*)?"
_ENUM_HEADING_SQUASHED = r"(?:\(?\d{1,3}[.)])?"
# What a `$`-anchored heading is allowed to have AFTER it. The 2023+ volumes
# print `Headnotes †` - the dagger footnoting the editor's name at the foot
# of the page - and a bare `$` reads that line as prose, which cost the
# `headnotes` signature on every 2023+ document measured. Only NON-WORD
# characters are tolerated: a heading followed by a word is a sentence, and
# that is the whole reason these two are anchored at the end at all.
_HEADING_TAIL = r"[^\w\n]*$"
_SIGNATURES = (
    # `HELD` is the reporter's label in three shapes the reprints actually
    # use: `HELD:` anywhere on a line (the column wrap), a bare all-caps
    # `HELD` opening a line with no punctuation at all, and the title-case
    # `Held:` the volumes set as often as the shouted one. Only the first is
    # unanchored - all-caps mid-line is unambiguous - and the title-case form
    # is line-anchored AND requires the colon, because that is exactly what
    # separates the reporter's label from the ordinary verb a column wrap can
    # leave at the start of a line ("...the High Court / held: that ...",
    # which is lower case and stays out).
    #
    # `Held:` is read on the DE-SPACED form only, and deliberately: taking
    # the spacing out of a line cannot break a pattern that has no spacing in
    # it, so a plain limb beside it would be a branch with no case of its own
    # (it was there, mutation showed it was dead, and it is gone). The
    # de-spaced BARE limb is anchored at both ends for the opposite reason -
    # de-spacing `HELDER AND ANOTHER v. STATE` leaves a line that opens with
    # those four letters and is a case name.
    ("held",
     re.compile(r"\bHELD\s*[:.]|^[ \t]*HELD\b", re.M),
     re.compile(r"(?<![A-Za-z])HELD[:.]|^HELD$|^Held:", re.M)),
    ("case_law_reference",
     re.compile(rf"^{_ENUM_HEADING}case\s+law\s+(?:reference|referred|cited)", re.I | re.M),
     re.compile(rf"^{_ENUM_HEADING_SQUASHED}caselaw(?:reference|referred|cited)", re.I | re.M)),
    ("cases_referred_to",
     re.compile(rf"^{_ENUM_HEADING}cases?\s+referred\s+to\s*:?{_HEADING_TAIL}", re.I | re.M),
     re.compile(rf"^{_ENUM_HEADING_SQUASHED}cases?referredto:?{_HEADING_TAIL}", re.I | re.M)),
    ("list_of_acts",
     re.compile(rf"^{_ENUM_HEADING}list\s+of\s+acts", re.I | re.M),
     re.compile(rf"^{_ENUM_HEADING_SQUASHED}listofacts", re.I | re.M)),
    ("list_of_keywords",
     re.compile(rf"^{_ENUM_HEADING}list\s+of\s+keywords", re.I | re.M),
     re.compile(rf"^{_ENUM_HEADING_SQUASHED}listofkeywords", re.I | re.M)),
    ("issue_for_consideration",
     re.compile(rf"^{_ENUM_HEADING}issues?\s+for\s+consideration", re.I | re.M),
     re.compile(rf"^{_ENUM_HEADING_SQUASHED}issues?forconsideration", re.I | re.M)),
    ("headnotes",
     re.compile(rf"^{_ENUM_HEADING}headnotes?\s*:?{_HEADING_TAIL}", re.I | re.M),
     re.compile(rf"^{_ENUM_HEADING_SQUASHED}headnotes?:?{_HEADING_TAIL}", re.I | re.M)),
    ("case_arising_from",
     re.compile(rf"^{_ENUM_HEADING}case\s+arising\s+from", re.I | re.M),
     re.compile(rf"^{_ENUM_HEADING_SQUASHED}casearisingfrom", re.I | re.M)),
    ("appearances",
     re.compile(rf"^{_ENUM_HEADING}appearances?\s+for\s+parties", re.I | re.M),
     re.compile(rf"^{_ENUM_HEADING_SQUASHED}appearances?forparties", re.I | re.M)),
)


# --------------------------------------------------------------------------
# The seam. Where the cut sits INSIDE what it removed.
# --------------------------------------------------------------------------
#
# THE FAILURE THESE TWO RULES EXIST FOR, which the signature rules cannot
# see. A real S.C.R. headnote carries ONE `HELD:` and numbers the points
# under it, so a marker-ish line inside that block satisfies every condition
# the two signature rules need in order to stay silent: the only signature
# name in the file is on the REMOVED side (so the comparison is satisfied by
# name), and nothing repeats the label past the cut (so the window is empty).
# The document is then emitted with the publisher's holding at the top of it
# AND with `signals` naming the furniture that WAS removed - so the audit's
# first-run tell, which reads `none` as the alarm, prints a healthy line over
# contaminated text. That is worse than a rendering nothing recognises.
#
# Both rules therefore read the CUT rather than the signatures, and both are
# armed only when the removed head carried furniture: where nothing editorial
# was removed there is no block for the cut to be inside, and a document with
# no headnote must not pay for this.

# The removed head ends on a blank line - the evidence that the block the
# last signature opened had CLOSED before the cut. A typeset heading is set
# off from the text above it; a line that the printed column merely wrapped
# onto is not, and that is the whole difference between `O R D E R` the
# court's heading and `ORDER` the fourteenth word of the publisher's holding.
_SEAM_BLOCK_BREAK = re.compile(r"\n[ \t]*\n[ \t]*\Z")
# An enumerated item at a line start, optionally behind the label that opens
# the block (`HELD: 1.`). The headnote's holding points and the judgment's own
# paragraphs are both written this way, which is exactly why a body opening on
# the number AFTER the one the headnote reached is the headnote continuing.
_ENUM_ITEM = re.compile(r"^[ \t]{0,3}(?:[A-Za-z][A-Za-z ]{0,23}:[ \t]*)?\(?(\d{1,3})[.)][ \t]", re.M)


def seam_splits_a_block(head: str) -> bool:
    """True when the cut did not BEGIN a block of text."""
    return _SEAM_BLOCK_BREAK.search(head) is None


def seam_continues_an_enumeration(head: str, body: str) -> bool:
    """True when the body opens on the item after the removed head's last one.

    Compares the NUMBERS rather than demanding the body open at `1.`: a
    judgment whose first paragraph the reader did not number would fail that,
    and this rule is meant to cost recall only where the two sides of the cut
    read as one list.
    """
    items = _ENUM_ITEM.findall(head)
    opening = _ENUM_ITEM.search(body[:SEAM_ENUM_WINDOW])
    if not items or opening is None:
        return False
    return int(opening.group(1)) == int(items[-1]) + 1


def headnote_signals(text: str) -> tuple[str, ...]:
    """Which pieces of editorial headnote furniture this text carries.

    Demotes FIRST, so the guard sees exactly what the boundary matcher sees.
    Everything the reader can put in front of the first word of a line - a
    table bar, a blockquote marker, a bullet, a heading hash, a bold run - is
    decoration, and a guard that reads one rendering of the furniture refuses
    one rendering of the headnote and publishes the other eight.

    Each signature is then read against up to SIX forms of the same text,
    one per way the typesetting can break a heading the eye reads whole: as
    demoted; with soft-hyphen line breaks closed up (`Case Law Refer-/ence`);
    with an inlined print-alignment letter off the front of the line (`C
    Held:`); and each of those with spacing removed line by line (the
    letter-spaced heading) and across one wrap (the letter-spaced heading the
    printed column was too narrow to hold).
    """
    plain = demote_markdown(text)
    plains = (plain,)
    for variant in (_dehyphen(plain), _unmargin(plain)):
        if variant not in plains:
            plains += (variant,)
    spaced_forms = tuple(
        form for source in plains for form in (_despace(source), _despace_pairs(source))
    )
    return tuple(
        name
        for name, pattern, spaced in _SIGNATURES
        if any(pattern.search(form) for form in plains)
        or any(spaced.search(form) for form in spaced_forms)
    )


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


# THE GAP latin_ratio CANNOT SEE. A Devanagari page set in a legacy
# non-Unicode font (Kruti Dev, Shree-Dev, DevLys - the ones Indian courts
# typed Hindi in for twenty years) does not extract as Devanagari at all: the
# glyph codes are Latin codepoints, so the text layer comes out as fluent-
# looking ASCII nonsense - `vfHkfyf[kr fu.kZ; esa mYysf[kr` - whose
# latin_ratio is 0.80, comfortably ABOVE the floor. Script detection cannot
# catch it because there is no script to detect.
#
# What it has none of is ENGLISH. Measured over the same 15 objects, real
# judgment text runs 0.372-0.457 stopwords per word (whole document and
# emitted body alike) and the mangled sample above runs 0.000, so the floor
# sits at 0.15: under half the worst real document and far above anything
# that is not English prose. The list is closed-class words only - no legal
# vocabulary - so it says "this is English", not "this is a judgment".
MIN_STOPWORD_RATE = 0.15
_WORD = re.compile(r"[A-Za-z]+")
_STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were be
    been being it its this that these those he she his her they them their which who
    whom not no nor so than then there here when where while shall may can will would
    should could has have had do does did any all such other same more most""".split()
)


def stopword_rate(text: str) -> float:
    """English closed-class words as a share of words (0.0 for nothing)."""
    words = _WORD.findall(text)
    if not words:
        return 0.0
    return sum(1 for word in words if word.lower() in _STOPWORDS) / len(words)


# --------------------------------------------------------------------------
# What the OBJECT is, as opposed to what its text says.
# --------------------------------------------------------------------------
#
# Two refusals that no amount of reading the text can reach, because the
# evidence for both is in the PDF's structure and both produce text that
# READS fine.
#
#   SCANNED ERA   The 2010-2017 volumes are not born-digital. They are JBIG2
#                 bitonal scans carrying an ABBYY OCR text layer, which the
#                 `HiddenHorzOCR` font is the fingerprint of. The text comes
#                 out plausible - 1,840-2,100 chars/page on the sampled
#                 2010 and 2014 objects, and this module emitted all six of
#                 them - but citation-level accuracy is exactly where
#                 twenty-year-old OCR fails, and the evidence is already
#                 visible in the running heads: `[201 O]` for `[2010]`,
#                 `S.Q.R.` for `S.C.R.`, `1f` for `11`. Whether OCR-era text
#                 belongs in v1 is a decision, and a decision that is taken
#                 by accident is the thing this quarantine exists to
#                 prevent. Measured: fires on 6 of 15, silent on the other
#                 9 - including a 2018 object that carries a DCTDecode
#                 photograph and is otherwise born-digital, which is why the
#                 test is the JBIG2/OCR-font structure and not "has images".
#   MOJIBAKE      see MIN_STOPWORD_RATE above for the failure; this is the
#                 structural half of it. A Devanagari-family font declared
#                 with anything other than an Identity encoding is a legacy
#                 8-bit glyph mapping, and its text layer is nonsense however
#                 clean it looks.
SCAN_OCR_FONTS = ("hiddenhorzocr",)
SCAN_IMAGE_FILTERS = ("jbig2decode",)
# Half the pages carrying an image and no text is a scan whatever its fonts
# and filters say. A born-digital judgment with one photographed exhibit is
# not, which is why this is a share and not a count.
SCAN_IMAGE_ONLY_SHARE = 0.5
# Below this a page with an image on it has no text worth the name - a page
# number and a running head come to more than this, so it is not a page the
# OCR merely did badly on, it is a page with no text layer.
MIN_IMAGE_PAGE_TEXT = 50
DEVANAGARI_FONT_FAMILIES = (
    "krutidev", "kruti", "devlys", "shreedev", "shree-dev", "chanakya", "shusha",
    "agra", "mangal", "kokila", "aparajita", "utsaah", "sanskrit", "devanagari",
)
IDENTITY_ENCODINGS = ("identity-h", "identity-v")


@dataclass(frozen=True)
class PdfStructure:
    """What the PDF is made of. Raw facts; the reading of them is below.

    Deliberately dumb: `pdf_structure` (past the reader seam, and therefore
    unverifiable offline) only reports what the library says, and every
    judgement about what those facts MEAN is made by `structural_refusal`,
    which is pure text and fully tested.
    """

    fonts: tuple[tuple[str, str], ...] = ()      # (base font name, encoding)
    image_filters: tuple[str, ...] = ()
    image_only_pages: int = 0
    pages: int = 0

    def digest(self) -> dict:
        """The compact form that goes on the document row."""
        return {
            "fonts": sorted({name for name, _enc in self.fonts})[:12],
            "image_filters": sorted(set(self.image_filters)),
            "image_only_pages": self.image_only_pages,
            "pages": self.pages,
        }


def _folded(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").casefold())


def structural_refusal(structure: PdfStructure) -> str | None:
    """The quarantine this object's STRUCTURE calls for, or None.

    Mojibake first: a document can be both a scan and mangled, and the
    unreadable text layer is the more actionable of the two - re-OCR fixes a
    scan, nothing fixes a font this module cannot decode.
    """
    identities = tuple(_folded(name) for name in IDENTITY_ENCODINGS)
    fonts = tuple(_folded(name) for name, _encoding in structure.fonts)
    for (name, encoding), folded in zip(structure.fonts, fonts):
        if any(family in folded for family in DEVANAGARI_FONT_FAMILIES):
            if _folded(encoding) not in identities:
                return Q_MOJIBAKE_FONT
    if any(marker in font for font in fonts for marker in SCAN_OCR_FONTS):
        return Q_SCANNED_ERA
    if any(marker in _folded(name) for name in structure.image_filters
           for marker in SCAN_IMAGE_FILTERS):
        return Q_SCANNED_ERA
    if structure.pages and structure.image_only_pages >= structure.pages * SCAN_IMAGE_ONLY_SHARE:
        return Q_SCANNED_ERA
    return None


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
        # Reading "NON-REPORTABLE" as its own opposite is the trap here, and
        # THREE things independently prevent it: this ordering, the `^` in
        # _REPORTABLE, and `.match` rather than `.search`. Any one of them
        # suffices, which is why no single-point mutation of the three can
        # fail the test that names the trap - only removing all three does
        # (verified: each alone SURVIVES, the three together are CAUGHT).
        # Recorded rather than thinned out: the cost is two lines, and the
        # comment now names the real mechanism rather than one third of it.
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
    reportable = reportable_flag(demote_markdown(pages[0] or "")) if pages else None
    cleaned, stats = clean_pages(pages)
    # Popped, not carried: the moved lines are TEXT, and `stats` becomes the
    # document row's meta_json. Only the count belongs in the database.
    footnote_lines = stats.pop("footnote_lines", ())

    offsets: list[int] = []
    cursor = 0
    for page in cleaned:
        offsets.append(cursor)
        cursor += len(page) + len(PAGE_SEPARATOR)
    joined = PAGE_SEPARATOR.join(cleaned)
    # Every signature the document carries ANYWHERE. It is what a refusal
    # reports, and it is one half of the residue comparison below.
    doc_signals = headnote_signals(joined)

    def no(reason: str, **over) -> Extraction:
        return Extraction(
            False,
            reason,
            pages=n_pages,
            reportable=reportable,
            signals=doc_signals,
            cleanup=stats,
            **over,
        )

    if len(joined.strip()) < MIN_DOC_CHARS:
        return no(Q_NO_TEXT)
    if (
        len(_CID.findall(joined)) > CID_LIMIT
        or latin_ratio(joined) < MIN_LATIN_RATIO
        # The third limb reads a different failure from the other two: they
        # ask whether the characters are Latin, and this asks whether the
        # LATIN IS ENGLISH. A legacy Devanagari font extracts as fluent-
        # looking ASCII and sails past both of the others.
        or stopword_rate(joined) < MIN_STOPWORD_RATE
    ):
        return no(Q_LOW_TEXT_QUALITY)

    boundary = find_judgment_start(joined)
    if boundary is None:
        hint = author_line_offset(joined)
        return no(
            Q_NO_JUDGMENT_START,
            author_hint=None if hint is None else round(hint / len(joined), 4),
        )

    body = joined[boundary.offset :]
    boundary_page = _page_of(offsets, boundary.offset)
    # Footnotes are appended at the end of the document, so they travel
    # ACROSS the boundary unless they are filtered by the page they came
    # off. Strictly after the boundary page: the boundary page carries both
    # the tail of the headnote and the head of the judgment, and an
    # editorial reference is the one thing that must not come back. Decided
    # here rather than after the checks so that what is kept is part of what
    # the residue check reads - text appended after the guard has run is
    # text the guard never saw.
    kept = [line for page, line in footnote_lines if page > boundary_page]
    removed_signals = headnote_signals(joined[: boundary.offset])

    # THE RESIDUE CHECK, in two directions, because there are two ways to be
    # wrong about a cut and only one of them looks forward.
    #
    #   NEAR:    any signature within RESIDUE_WINDOW past the cut. A headnote
    #            leaks at the SEAM, and this fires on the evidence regardless
    #            of what was removed.
    #   BEYOND:  any signature the document carries that the REMOVED HEAD
    #            does not. The window is measured in characters and the front
    #            matter is measured in pages (P0: pages 1-3 of a routine
    #            judgment), so a marker at the top of the front matter has
    #            nothing to see inside the window - and a marker on line one
    #            removes nothing at all, which the window cannot distinguish
    #            from a document that never had a headnote.
    #
    # BEYOND compares by signature NAME, not by occurrence, and that is what
    # keeps it from refusing the ordinary case: a reprint whose front matter
    # carried `HELD:` and whose judgment later quotes an earlier report's
    # `HELD:` is emitted, because the name is accounted for on the removed
    # side. What it refuses is a signature with NO counterpart in what was
    # thrown away - which is indistinguishable, from outside the document,
    # from a headnote the cut went over the top of. It also catches the one
    # thing no forward window ever could: editorial matter at the END of a
    # file, which this module has no pass to strip.
    residue = sorted(
        set(headnote_signals(body[:RESIDUE_WINDOW]))
        | (
            (set(doc_signals) | set(headnote_signals("\n".join(kept))))
            - set(removed_signals)
        )
    )
    # AND THE TWO THAT READ THE CUT INSTEAD OF THE SIGNATURES, because a
    # headnote with ONE `HELD:` and numbered points under it satisfies both
    # rules above while the cut sits in the middle of it (see the seam rules).
    # Armed only when furniture was actually removed.
    #
    #   BLOCK:   the cut did not begin a block. A heading is set off; a
    #            wrapped line is not.
    #   ENUM:    the body opens on the item after the last one the removed
    #            head reached, i.e. the two sides read as one list.
    head = joined[: boundary.offset]
    split_seam = bool(removed_signals) and (
        seam_splits_a_block(head) or seam_continues_an_enumeration(head, body)
    )
    if residue or split_seam:
        # The cut landed INSIDE the front matter, or went over the top of it.
        # Emitting this document would ship the publisher's summary of the
        # answer.
        return no(Q_HEADNOTE_RESIDUE, marker=boundary.marker, headnote_chars=boundary.offset)
    # SHORT BEFORE LARGE-STRIP, deliberately. When both hold there is
    # nothing to recover either way, and `strip_too_large` should keep its
    # alarming meaning: most of a document that still had plenty of text in
    # it was thrown away, which is a boundary bug and not a thin judgment.
    if len(body.strip()) < MIN_BODY_CHARS:
        return no(Q_BODY_TOO_SHORT, marker=boundary.marker, headnote_chars=boundary.offset)
    if boundary.offset / len(joined) > MAX_STRIP_FRACTION:
        return no(Q_STRIP_TOO_LARGE, marker=boundary.marker, headnote_chars=boundary.offset)

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
        # Signals of what was REMOVED - "this document had a headnote and it
        # is gone". A quarantine reports the signals of the WHOLE document
        # instead, because there the question is what is in the file at all.
        #
        # On an EMITTED document the two are the same set, and that is a
        # property of the check above rather than a coincidence: nothing is
        # emitted while it still carries a signature the removed head did
        # not. So `signals: none` on an emitted document means the file
        # carries no editorial furniture anywhere - which is what makes it
        # readable as a first-run alarm (see audit_report).
        signals=removed_signals,
        reportable=reportable,
        footnotes=len(kept),
        cleanup=stats,
    )


# --------------------------------------------------------------------------
# The reader seam. Nothing above this line imports a PDF library.
# --------------------------------------------------------------------------

# EVERY option that decides what the text CONTAINS is passed explicitly, so
# that the corpus is a property of this repository and not of whatever the
# resolver installed. Each of these has a library default, and each default
# quietly decides something this module then has to live with:
#
#   margins=0          the library's default crops 50 points off the top and
#                      bottom of EVERY page. That band is where the running
#                      head, the printed page number, the page-tail FOOTNOTES
#                      and the REPORTABLE line live - i.e. the default
#                      deletes the input to three passes below, and
#                      `dropped: N running` would read 0 for a reason that
#                      has nothing to do with the document.
#   table_strategy     the Case Law Reference block is a GRID, and this
#                      decides whether it arrives as `|pipes|` or as plain
#                      lines. demote_markdown handles both, but which one
#                      shows up is not left to a version number.
#   write_images /     no image, and no base64 of an image, may ever land in
#   embed_images       a judgment's text.
#   force_text         text drawn over a figure is still the court's words.
#   show_progress      a progress bar per document, over tens of thousands of
#                      documents, is noise in a log nobody can then read.
#
# VERIFIED against pymupdf4llm 1.28.2 - see READER_LANE below for what that
# verification found and why the lane is now pinned by name.
READER_OPTIONS = {
    "page_chunks": True,
    "margins": 0,
    "table_strategy": "lines_strict",
    "write_images": False,
    "embed_images": False,
    "force_text": True,
    "show_progress": False,
}
# The ones whose absence changes the CORPUS silently. A reader that cannot
# take these is refused; the rest degrade to their defaults, because losing a
# progress bar is not worth stopping a run over.
READER_REQUIRED = ("page_chunks", "margins", "table_strategy")
# OCR IS OFF, and v1 ships no OCR by design (`no_text` is a quarantine, not
# an invitation). The pinned lane below has no OCR knob at all, so there is
# nothing to turn off there - but the OTHER lane defaults `use_ocr=True`,
# and a future version of the pinned one could grow the same default. These
# are therefore passed only where they are EXPLICIT parameters, on the same
# principle as READER_REQUIRED: a value that rides in on `**kwargs` is
# accepted, not honoured, and would report a pinning that had not happened.
READER_OCR_OFF = {"use_ocr": False, "force_ocr": False}

# THE LANE, pinned by import path rather than taken from the package.
#
# `pymupdf4llm.to_markdown` in 1.28.2 is not a function, it is a
# `(*args, **kwargs)` dispatch shim that forwards to one of two completely
# different implementations. _reader_options refuses it - correctly, and
# that refusal is why every document raised until this was pinned - but
# "refuse the shim" is only half an answer, because the shim's DEFAULT
# destination is the layout path, and that path:
#
#   * silently drops `margins` and `table_strategy` (the two options above
#     whose absence changes the corpus);
#   * defaults `use_ocr=True`, which makes `no_text` unmeasurable and
#     contradicts the no-OCR-in-v1 decision;
#   * deletes text it classifies as a formula, on legal documents, upstream
#     and unfixed;
#   * ran ~2.3x slower on the same objects (measured), with an open
#     Windows-only crash in its ONNX dependency;
#   * and - measured - emitted `ok` with `signals: none` on a document whose
#     text plainly reads `C Held:`, i.e. it fired this module's own
#     first-run alarm and read as healthy.
#
# So the legacy implementation is named explicitly. It honours all three
# required options, has no OCR, and is what every measurement in this file
# was taken against.
READER_LANE = "pymupdf4llm.helpers.pymupdf_rag"


def _pinned_to_markdown():
    """The pinned `to_markdown`, or an actionable refusal.

    `use_layout(False)` is belt and braces: it puts the package-level shim
    on the same path this function returns, so anything else in the process
    that calls `pymupdf4llm.to_markdown` gets the lane this module chose
    rather than the one the library prefers.
    """
    try:
        import pymupdf4llm
    except ImportError as exc:
        raise ExtractionError(
            "pymupdf4llm is needed to read the judgment PDFs and is not installed - "
            "run: pip install -e .[build]"
        ) from exc
    try:
        from pymupdf4llm.helpers.pymupdf_rag import to_markdown
    except ImportError as exc:
        raise ExtractionError(
            f"the installed pymupdf4llm no longer provides {READER_LANE}.to_markdown, "
            f"and this module pins that lane by name rather than accepting whatever "
            f"`pymupdf4llm.to_markdown` dispatches to: the alternative path drops "
            f"`margins` and `table_strategy`, turns OCR ON by default, and was measured "
            f"reporting no headnote on a document that has one. Re-pin READER_LANE "
            f"against the installed version deliberately - do not fall back."
        ) from exc
    use_layout = getattr(pymupdf4llm, "use_layout", None)
    if callable(use_layout):
        use_layout(False)
    return to_markdown


def reader_fingerprint(reader) -> dict:
    """Which reader produced a document's text - recorded on every row.

    EXTRACT_VERSION alone under-guards the corpus and the smoke proved it:
    the same rules over the same PDFs gave DIFFERENT verdicts on the two
    lanes (two 2025 objects flipped between `ok` and quarantined), so the
    text on disk is a function of the library as well as of this file. A row
    that does not say which reader made it cannot be re-derived, and the
    audit's byte-for-byte re-check would blame the rules for a library
    upgrade.
    """
    if reader is read_pdf_pages:
        version = None
        try:
            from importlib.metadata import version as _version

            version = _version("pymupdf4llm")
        except Exception:  # pragma: no cover - absent library, reported as None
            version = None
        return {"lane": READER_LANE, "pymupdf4llm": version}
    # An injected reader (a test's, or the MIT fallback the plan names) is
    # named for what it is. The point is that the row says which one ran.
    name = getattr(reader, "__qualname__", type(reader).__qualname__)
    module = getattr(reader, "__module__", type(reader).__module__)
    return {"lane": f"{module}.{name}", "pymupdf4llm": None}


def _reader_options(to_markdown) -> dict:
    """READER_OPTIONS this `to_markdown` can actually take, or an error.

    A `**kwargs` in the signature does NOT satisfy the requirement, and that
    is the whole point of the check rather than an oversight of it: a reader
    that swallows `**kwargs` accepts `margins=0` without applying it, so a
    renamed option would leave the library's 50-point CROP in force on every
    page while this function reported that it had pinned the behaviour. An
    option that is only *accepted* is not an option that is *honoured*, and
    only an explicit parameter is evidence of the latter.
    """
    import inspect

    params = inspect.signature(to_markdown).parameters
    var_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    missing = [name for name in READER_REQUIRED if name not in params]
    if missing:
        raise ExtractionError(
            f"the installed pymupdf4llm does not accept {', '.join(missing)}, and this "
            f"module pins the reader's behaviour rather than inheriting it - `margins` "
            f"above all, because the library's default crops the running heads and the "
            f"page-tail footnotes away before this module can see them"
            + (
                " (its signature takes **kwargs, which accepts these names without "
                "promising to honour them - that is not the same thing and is not "
                "accepted here)"
                if var_keyword
                else ""
            )
            + f". Check the installed version against READER_OPTIONS in this file."
        )
    # The non-required ones may ride in on **kwargs: losing a progress bar to
    # a renamed option is not worth stopping a run over, which is the same
    # reason they are not in READER_REQUIRED.
    options = {
        name: value
        for name, value in READER_OPTIONS.items()
        if name in params or var_keyword
    }
    # ...and the OCR switches never do. See READER_OCR_OFF.
    options.update(
        {name: value for name, value in READER_OCR_OFF.items() if name in params}
    )
    return options


def read_pdf_pages(path: str | Path) -> list[str]:
    """One string per page, via the pinned pymupdf4llm lane.

    The MIT fallback the plan names (pypdf + pdfplumber) is a replacement
    for THIS FUNCTION and nothing else - everything above it is text.
    """
    to_markdown = _pinned_to_markdown()
    chunks = to_markdown(str(path), **_reader_options(to_markdown))
    return [chunk["text"] if isinstance(chunk, dict) else str(chunk) for chunk in chunks]


def pdf_structure(path: str | Path) -> PdfStructure:
    """What the PDF is made of, for the two structural quarantines.

    Past the reader seam and therefore not verifiable offline, so it makes
    no judgements at all: it reports the fonts, the image filters and the
    count of pages that carry an image and no text, and `structural_refusal`
    - which is pure - decides what that means. Measured at 0.03-0.33 s per
    document against 28.9 s for the same document's markdown conversion, so
    it is free relative to the read it accompanies.
    """
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - ships with pymupdf4llm
        raise ExtractionError(
            "pymupdf is needed to read the judgment PDFs' structure and is not "
            "installed - run: pip install -e .[build]"
        ) from exc
    fonts: dict[tuple[str, str], None] = {}
    filters: dict[str, None] = {}
    image_only = 0
    with pymupdf.open(str(path)) as doc:
        for page in doc:
            for entry in page.get_fonts(full=True):
                # (xref, ext, type, basefont, refname, encoding, referencer)
                fonts.setdefault((str(entry[3]), str(entry[5])), None)
            images = page.get_images(full=True)
            for image in images:
                try:
                    filters.setdefault(str(doc.xref_get_key(image[0], "Filter")[1]), None)
                except Exception:  # pragma: no cover - a filter we cannot read
                    continue
            if images and len(page.get_text().strip()) < MIN_IMAGE_PAGE_TEXT:
                image_only += 1
        return PdfStructure(
            fonts=tuple(fonts),
            image_filters=tuple(filters),
            image_only_pages=image_only,
            pages=doc.page_count,
        )


# The reader seam carries the structural probe with it, because the facts
# are a property of the object the reader opens: a reader that can report
# them hangs `pdf_structure` off itself, and one that cannot (a test's, the
# MIT fallback before it grows one) leaves the structural gates unarmed -
# which extract_corpus counts and the CLI prints, so that "unarmed" is a
# line in the log rather than a silence.
read_pdf_pages.structure = pdf_structure


def structure_of(reader, path: str | Path) -> PdfStructure | None:
    """This reader's structural facts for `path`, or None if it has none."""
    probe = getattr(reader, "structure", None)
    return None if probe is None else probe(path)


# --------------------------------------------------------------------------
# Joining the selection to the PDFs on disk.
# --------------------------------------------------------------------------

ROUTE_PDF_KEY = "pdf_key"
ROUTE_SCR_PREFIX = "scr_prefix"
ROUTE_AMBIGUOUS = "ambiguous"
ROUTE_UNMATCHED = "unmatched"
JOIN_ROUTES = (ROUTE_PDF_KEY, ROUTE_SCR_PREFIX, ROUTE_AMBIGUOUS, ROUTE_UNMATCHED)


@dataclass(frozen=True)
class PdfIndex:
    """The acquired PDFs, addressed both ways the selection can address them."""

    by_key: dict[str, str]
    by_prefix: dict[str, tuple[str, ...]]

    def __len__(self) -> int:
        return len(self.by_key)


def pdf_index(store, source_id: str = SC_SOURCE_ID) -> PdfIndex:
    """Build the join index from the artifact table, in one read.

    Driven off the artifact index rather than a directory walk, for select.py's
    reason: a PDF can only be read out of a file the store says is complete.
    """
    by_key: dict[str, str] = {}
    by_prefix: dict[str, list[str]] = {}
    for key, artifact in store.artifact_index(source_id).items():
        if not key.lower().endswith(".pdf"):
            continue
        by_key[key] = artifact["local_path"]
        span = page_span_from_key(key)
        if span is not None:
            by_prefix.setdefault(f"{span.year}_{span.volume}_{span.start}_", []).append(key)
    return PdfIndex(by_key, {k: tuple(sorted(v)) for k, v in by_prefix.items()})


def resolve_pdf(row: dict, index: PdfIndex) -> tuple[str | None, str]:
    """Which acquired PDF is this selection row's, and by which route.

    `pdf_key` first (the metadata's own link, when the schema carries one),
    then `scr_prefix` - select.py's inference that "[2020] 7 S.C.R. 941"
    addresses `2020_7_941_*`, which is the ONLY join available if no link
    column exists. Both routes are counted at run level, because which one
    carried the corpus is a fact no offline test can settle.

    A `pdf_key` naming an object that is not indexed falls THROUGH to the
    prefix rather than failing: an interrupted or `--limit`ed acquire leaves
    exactly that state, and stranding those rows would be a self-inflicted
    hole in the corpus.

    An ambiguous prefix resolves to nothing. Taking "whichever sorts first"
    would attach a judgment to another judgment's citation, and nothing
    downstream could see it.
    """
    key = row.get("pdf_key")
    if key and key in index.by_key:
        return key, ROUTE_PDF_KEY
    prefix = row.get("scr_prefix")
    if prefix:
        matches = index.by_prefix.get(prefix, ())
        if len(matches) == 1:
            return matches[0], ROUTE_SCR_PREFIX
        if len(matches) > 1:
            return None, ROUTE_AMBIGUOUS
    return None, ROUTE_UNMATCHED


# --------------------------------------------------------------------------
# Where the text goes, and the resume decision.
# --------------------------------------------------------------------------

def text_path_for(root: str | Path, object_key: str) -> Path:
    """`data/pdf/year=2015/english/x_EN.pdf` -> `<root>/.../x_EN.txt`.

    The bucket layout is mirrored (acquire.local_path_for does the refusing:
    the key comes off a remote listing and must not choose where this
    process writes), so the object key alone finds the text again.
    """
    return local_path_for(root, object_key).with_suffix(".txt")


def extract_decision(indexed: dict | None, text_path: str | Path, *, force: bool = False) -> str:
    """"extract" or "skip" for one document - the whole resume policy.

    Four ways to be out of date and one way to be current:

      * no row at all - including the crash window where the text landed and
        the row did not, which is why there is no "adopt";
      * a row written under different rules (`extract_version`);
      * a row that says `ok` and points at text that is not there;
      * `--force`, which re-reads everything.

    A QUARANTINE at the current version is a SKIP: the rules are
    deterministic and the bytes have not changed, so re-reading it would
    spend the run's time on the documents that cannot be used. Changing the
    rules is what re-opens them, and that is what the version is for.
    """
    if force or indexed is None:
        return "extract"
    if indexed.get("extract_version") != EXTRACT_VERSION:
        return "extract"
    if indexed.get("status") == STATUS_OK and not Path(text_path).exists():
        return "extract"
    return "skip"


def write_text(path: str | Path, text: str) -> tuple[int, str]:
    """Write one judgment durably; returns (chars, sha256).

    Same rule as acquire.download_object and for the same reason: written to
    a sibling `.part` and renamed, so `path` is either absent or the whole
    document. A reader that finds a prefix of a judgment has no way to know.
    """
    path = Path(path)
    part = path.with_name(path.name + PART_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = text.encode("utf-8")
    try:
        with part.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(part, path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return len(text), hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# The run.
# --------------------------------------------------------------------------

DEFAULT_MAX_FAILURES = 25
# Above this share of refusals the run is not a corpus with some bad
# documents in it - something about the rules or the source is wrong.
QUARANTINE_ALARM = 0.25


def extract_corpus(
    store,
    rows: Iterable[dict],
    *,
    index: PdfIndex,
    text_root: str | Path,
    reader=read_pdf_pages,
    source_id: str = SC_SOURCE_ID,
    limit: int | None = None,
    force: bool = False,
    max_failures: int = DEFAULT_MAX_FAILURES,
    allow_scanned_era: bool = False,
) -> dict:
    """Extract the selection, in the order it is given, resumably.

    THE ORDER IS THE SELECTION FILE'S and is never re-sorted. Task 11 made
    selection.jsonl a stratified SPREAD rather than the top N by priority
    precisely so that an interrupted extraction fails over the whole corpus
    shape instead of over one case type; re-ranking here would undo it.

    `limit` caps WORK - documents extracted, quarantined or failed - not
    rows examined, so a resumed run advances instead of spending the cap on
    the documents it already has (acquire.py's lesson, same words).

    A per-document failure is counted and the run continues; `max_failures`
    stops a run whose fault is not in the documents. A failure to write the
    INDEX is deliberately not caught - that is the database, not a PDF.

    `allow_scanned_era` re-admits the OCR-era volumes. It is the one knob on
    a structural quarantine and it exists because "which years are in v1" is
    an operator's decision: the gate makes the decision explicit, it does
    not make it here. The mojibake gate has no such knob - a text layer this
    module cannot decode is not a matter of preference.
    """
    text_root = Path(text_root)
    fingerprint = reader_fingerprint(reader)
    stats: dict = {
        "considered": 0,
        "extracted": 0,
        "quarantined": 0,
        "skipped": 0,
        "failed": 0,
        "duplicate_rows": 0,
        "reportable": 0,
        "chars": 0,
        "routes": dict.fromkeys(JOIN_ROUTES, 0),
        "reasons": {},
        "failures": [],
        "reader": fingerprint,
        # Whether the structural gates could run at all. A reader with no
        # structural probe leaves scanned_era and mojibake_font unarmed, and
        # that has to be visible: a run reporting zero scan-era refusals
        # because it never looked reads exactly like a clean corpus.
        "structure_probed": 0,
    }
    indexed = store.document_index(source_id)
    seen: set[str] = set()

    for row in rows:
        if limit is not None and (
            stats["extracted"] + stats["quarantined"] + stats["failed"] >= limit
        ):
            break
        stats["considered"] += 1
        key, route = resolve_pdf(row, index)
        stats["routes"][route] += 1
        if key is None:
            continue
        if key in seen:
            # ONE OBJECT, ONE ROW, and the FIRST row wins - the same rule the
            # manifest applies, applied in the same direction. Two selection
            # rows can land on one PDF, and if this one re-recorded from the
            # later row then `--force` would leave the document row and the
            # manifest row describing one judgment under two case ids, with
            # nothing downstream reading object_key for identity to notice.
            # It also saves re-reading the same PDF.
            stats["duplicate_rows"] += 1
            continue
        seen.add(key)

        try:
            dest = text_path_for(text_root, key)
            if extract_decision(indexed.get(key), dest, force=force) == "skip":
                stats["skipped"] += 1
                continue
            local = index.by_key[key]
            # STRUCTURE BEFORE TEXT, and it is the cheaper read by two
            # orders of magnitude: a scan-era object is refused without
            # spending 29 seconds converting it to markdown first.
            structure = structure_of(reader, local)
            refusal = None
            if structure is not None:
                stats["structure_probed"] += 1
                refusal = structural_refusal(structure)
                if refusal == Q_SCANNED_ERA and allow_scanned_era:
                    refusal = None
            if refusal is not None:
                result = Extraction(False, refusal, pages=structure.pages)
            else:
                result = extract_text(reader(local))
        except Exception as exc:
            stats["failed"] += 1
            detail = {"key": key, "error": f"{type(exc).__name__}: {exc}"}
            stats["failures"].append(detail)
            store.log_event("extract_failed", {"source_id": source_id, **detail})
            if stats["failed"] >= max_failures:
                raise ExtractionError(
                    f"extract: stopping after {stats['failed']} failures "
                    f"(last {key!r}) - at this rate the fault is not in the documents"
                ) from exc
            continue

        if result.reportable is not None:
            # Counted because a NON-NULL flag on this corpus is a warning,
            # not a fact about the judgment: P0 found REPORTABLE in 2 of 70
            # of these objects, and the flag belongs to the COURT-RELEASED
            # PDF rather than to the S.C.R. reprint. So a document that
            # carries one is better read as "this object may not be a
            # reprint" - i.e. exactly the object whose boundary rules may not
            # apply. Un-counted, that caveat was in a report nobody reads at
            # runtime.
            stats["reportable"] += 1
        span = page_span_from_key(key)
        record = {
            "status": STATUS_OK if result.ok else STATUS_QUARANTINED,
            "reason": result.reason,
            "case_id": row.get("case_id"),
            "citation": row.get("citation"),
            "year": row.get("year"),
            "pages": result.pages,
            "page_start": span.start if span else None,
            "page_end": span.end if span else None,
            "marker": result.marker,
            "extract_version": EXTRACT_VERSION,
            # The reader goes on EVERY row, emitted or refused: the verdict
            # is a function of the lane as much as of these rules, and a row
            # that does not name its reader cannot be re-derived.
            "meta": {
                **result.meta(),
                "reader": fingerprint,
                "structure": None if structure is None else structure.digest(),
            },
        }
        if result.ok:
            chars, digest = write_text(dest, result.text)
            record.update(text_path=str(dest), chars=chars, headnote_chars=result.headnote_chars,
                          sha256=digest)
            stats["extracted"] += 1
            stats["chars"] += chars
        else:
            # BEFORE the row, and in this order on purpose. A document that
            # was emitted under older rules and is refused under these ones
            # must not leave readable text behind: a consumer that globs the
            # text tree instead of reading the manifest would pick up a
            # judgment the index says does not exist. Unlinking first is
            # also the self-healing order - a crash here leaves a row saying
            # `ok` and no file, which the next run re-extracts.
            dest.unlink(missing_ok=True)
            record.update(text_path=None, headnote_chars=result.headnote_chars)
            stats["quarantined"] += 1
            stats["reasons"][result.reason] = stats["reasons"].get(result.reason, 0) + 1

        # The text is durable from here; the row may now claim it exists.
        store.record_document(source_id, key, record)
        indexed[key] = {
            "object_key": key,
            "status": record["status"],
            "reason": record["reason"],
            "text_path": record["text_path"],
            "extract_version": EXTRACT_VERSION,
        }
    return stats


# --------------------------------------------------------------------------
# The manifest.
# --------------------------------------------------------------------------

MANIFEST_FIELDS = (
    "doc_id", "case_id", "title", "citation", "year", "court", "coram",
    "case_type", "priority", "object_key", "text_path", "chars", "pages",
    "page_start", "page_end", "marker", "reportable", "source_id",
    # WHICH READER MADE THIS TEXT. Downstream reads the manifest and not the
    # document table, so a corpus assembled across a library upgrade would
    # otherwise be indistinguishable from one that was not.
    "reader_lane", "reader_version",
)


def manifest_rows(
    store, rows: Iterable[dict], *, index: PdfIndex, source_id: str, dropped: list | None = None
) -> Iterator[dict]:
    """One row per EMITTED judgment: the selection row plus what came out.

    Derived state, regenerable from (selection.jsonl + the document table) at
    any time, which is why an interrupted run losing it costs nothing. Only
    documents with text: listing a quarantined judgment would hand the
    segmenter a path that is not there.

    ONE ROW PER OBJECT. Two selection rows can land on the same PDF - the
    metadata can carry a judgment twice, and two citations can address one
    object - and the corpus would then hold that judgment twice under two
    case ids. Nothing downstream reads object_key for identity, so this is
    the only place it can be caught.

    WHICH row wins is the FIRST one in the order `rows` arrives in, and that
    is all this guarantees: it is deduplication, not selection. The dropped
    keys are logged as `manifest_duplicate_documents` rather than discarded
    silently, because a judgment addressed by two citations is a fact about
    the metadata and the operator is the one who can say which citation is
    the one to keep.
    """
    documents = {row["object_key"]: row for row in store.documents(source_id, status=STATUS_OK)}
    seen: set[str] = set()
    for row in rows:
        key, _ = resolve_pdf(row, index)
        document = documents.get(key) if key else None
        if document is None:
            continue
        if key in seen:
            if dropped is not None:
                dropped.append(key)
            continue
        seen.add(key)
        meta = json.loads(document["meta_json"] or "{}")
        merged = {
            **{field: row.get(field) for field in MANIFEST_FIELDS},
            "doc_id": Path(key).stem,
            "object_key": key,
            "text_path": document["text_path"],
            "chars": document["chars"],
            "pages": document["pages"],
            "page_start": document["page_start"],
            "page_end": document["page_end"],
            "marker": document["marker"],
            "reportable": meta.get("reportable"),
            "reader_lane": (meta.get("reader") or {}).get("lane"),
            "reader_version": (meta.get("reader") or {}).get("pymupdf4llm"),
            "source_id": source_id,
        }
        yield merged


def write_manifest(
    store, rows: Iterable[dict], path: str | Path, *, index: PdfIndex,
    source_id: str = SC_SOURCE_ID,
) -> int:
    from tuned.data.jsonl import write_jsonl

    dropped: list[str] = []
    written = write_jsonl(
        path, manifest_rows(store, rows, index=index, source_id=source_id, dropped=dropped)
    )
    if dropped:
        # Not silent: two selection rows on one judgment is a fact about the
        # metadata, and the operator should see it rather than wonder why
        # the manifest is shorter than the emitted count.
        store.log_event(
            "manifest_duplicate_documents",
            {"count": len(dropped), "keys": sorted(set(dropped))[:20]},
        )
    return written


# --------------------------------------------------------------------------
# --audit: the operator's only window into extraction quality.
# --------------------------------------------------------------------------

AUDIT_REMOVED_CHARS = 400
AUDIT_KEPT_CHARS = 600


def spread(items: Sequence, n: int) -> list:
    """`n` items evenly spaced along `items` - deterministic, not random.

    Along object_key order, which is year order for this corpus, so the
    sample is a walk across the scope rather than a look at whatever the
    extractor happened to reach first.
    """
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def audit_sample(store, n: int, *, source_id: str = SC_SOURCE_ID) -> list[dict]:
    """Documents to read, half of them refusals where there are any.

    The refusals are the reason the audit exists, and they are a minority by
    construction, so an even sample over the whole table would rarely show
    one.
    """
    quarantined = store.documents(source_id, status=STATUS_QUARANTINED)
    emitted = store.documents(source_id, status=STATUS_OK)
    # Half, or as many as it takes to fill the sample when there are not
    # enough emitted documents to fill it with. The second half of that
    # matters most in the worst case there is: a run that refused everything
    # would otherwise print an audit with nothing in it, at exactly the
    # moment the operator most needs to see what was refused.
    want_bad = min(len(quarantined), max(n - len(emitted), n // 2)) if quarantined else 0
    picked = spread(emitted, n - want_bad) + spread(quarantined, want_bad)
    return sorted(picked, key=lambda row: row["object_key"])


def _indent(text: str, prefix: str = "      ") -> str:
    return "\n".join(prefix + line for line in text.strip().splitlines()) or prefix + "(nothing)"


def _from_line_start(excerpt: str, truncated: bool) -> str:
    """An excerpt that begins where a line does.

    A window cut by character count opens mid-word, and the operator is
    being asked to JUDGE this text - the first line has to be readable.
    """
    if not truncated or "\n" not in excerpt:
        return excerpt
    return excerpt.split("\n", 1)[1]


# The one check that costs nothing on run one and settles the largest
# residual in the module: whether the guard can read THIS reporter's
# typesetting at all. It is printed at the top of every audit because a
# blind guard is silent everywhere else - a contaminated document reads like
# a clean judgment, and the line the operator would otherwise scroll past
# says exactly what a clean document says.
AUDIT_TELL = (
    "  READ THIS FIRST: an emitted document lists every editorial signature the guard\n"
    "  RECOGNISED in it, so `headnote signals: none` means the guard found no editorial\n"
    "  furniture anywhere in this file after cleanup (running heads, signature stamps and\n"
    "  pre-boundary footnotes are removed before it looks). If you can see a headnote on a\n"
    "  document that printed `none`, the guard cannot read this reporter's typesetting -\n"
    "  and then every `ok` document in this run is suspect, not just this one.\n"
    "  It settles RECOGNITION and not placement: a document that names the furniture it\n"
    "  removed can still have been cut inside that furniture, which the four residue\n"
    "  rules narrow but do not close. Stop and re-read the boundary rules."
)


def audit_report(
    store, n: int, *, index: PdfIndex, reader=read_pdf_pages, source_id: str = SC_SOURCE_ID
) -> str:
    """Re-read `n` sampled documents and show the seam, both sides of it.

    RE-EXTRACTED rather than read back from disk, for two reasons: the text
    that was thrown away is not kept anywhere (the store holds the judgment,
    not the publisher's headnote), and re-running is a CHECK - the sampled
    document is re-extracted under today's rules and compared, byte for
    byte, against the row the corpus holds.
    """
    fingerprint = reader_fingerprint(reader)
    lane = fingerprint["lane"]
    if fingerprint["pymupdf4llm"]:
        lane += f" {fingerprint['pymupdf4llm']}"
    lines = [
        f"AUDIT of {store.document_count(source_id, status=STATUS_OK)} emitted and "
        f"{store.document_count(source_id, status=STATUS_QUARANTINED)} quarantined documents"
        f"  (extract_version {EXTRACT_VERSION}, reader {lane})",
        AUDIT_TELL,
    ]
    for row in audit_sample(store, n, source_id=source_id):
        key = row["object_key"]
        lines.append("")
        lines.append(f"--- {key}")
        local = index.by_key.get(key)
        head = f"    {row['citation'] or '?'}   {row['case_id'] or '?'}   "
        if row["page_start"] is not None:
            head += f"S.C.R. pp. {row['page_start']}-{row['page_end']}   "
        head += f"{row['pages']} PDF pages"
        lines.append(head)
        if local is None:
            lines.append("    THE PDF IS NO LONGER INDEXED - cannot re-read it")
            continue
        try:
            pages = reader(local)
        except Exception as exc:
            lines.append(f"    UNREADABLE NOW: {type(exc).__name__}: {exc}")
            continue
        cleaned, stats = clean_pages(pages)
        joined = PAGE_SEPARATOR.join(cleaned)
        result = extract_text(pages)
        boundary = find_judgment_start(joined)

        # The other half of "re-running is also a check": what the rules
        # produce TODAY against what the corpus was built from. A row that no
        # longer reproduces is not a document to read, it is a version bump
        # that has not been run - and the audit is the only place anything
        # compares the two.
        now = STATUS_OK if result.ok else STATUS_QUARANTINED
        if now != row["status"]:
            lines.append(
                f"    !! THE RULES NO LONGER AGREE WITH THE CORPUS: the store says "
                f"{row['status']}{' (' + row['reason'] + ')' if row['reason'] else ''}, "
                f"re-extracting now gives {now}"
                f"{' (' + result.reason + ')' if result.reason else ''} - re-run with a "
                f"bumped extract_version"
            )
        elif result.ok and row["sha256"]:
            digest = hashlib.sha256(result.text.encode("utf-8")).hexdigest()
            if digest != row["sha256"]:
                lines.append(
                    f"    !! DIFFERS FROM THE STORED TEXT: {row['chars']:,} chars stored, "
                    f"{len(result.text):,} chars now - the text on disk was produced by "
                    f"rules this run no longer has"
                )

        if not result.ok:
            lines.append(f"    QUARANTINED {result.reason}")
            lines.append(f"    headnote signals: {', '.join(result.signals) or 'none'}")
            if result.author_hint is not None:
                lines.append(
                    f"    an author line ('NAME, J.') sits at {result.author_hint:.1%} of the "
                    f"document - a hint, deliberately not a boundary"
                )
            lines.append("    FIRST OF WHAT IS THERE:")
            lines.append(_indent(joined[:AUDIT_KEPT_CHARS]))
            continue

        share = result.headnote_chars / max(1, len(joined))
        lines.append(
            f"    boundary: {result.marker} on page {result.boundary_page + 1} of "
            f"{result.pages}, {result.headnote_chars:,} chars removed ({share:.1%})"
        )
        lines.append(
            f"    headnote signals: {', '.join(result.signals) or 'none'}   "
            f"reportable: {result.reportable or '-'}   "
            f"dropped: {stats['signature']} signature / {stats['margin_letter']} margin / "
            f"{stats['running']} running lines, footnotes off "
            f"{stats['footnote_pages']} pages"
        )
        lines.append("    LAST OF WHAT WAS REMOVED (must be the reporter's editorial matter):")
        start = max(0, boundary.offset - AUDIT_REMOVED_CHARS)
        opens_mid_line = start > 0 and joined[start - 1] != "\n"
        lines.append(
            _indent(_from_line_start(joined[start: boundary.offset], truncated=opens_mid_line))
        )
        lines.append("    FIRST OF WHAT WAS KEPT (must be the court's own words):")
        lines.append(_indent(result.text[:AUDIT_KEPT_CHARS]))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def _fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"  # pragma: no cover - loop returns first


def main(argv: Sequence[str] | None = None, *, reader=None) -> int:
    import argparse

    from tuned.data.config import load_build_config
    from tuned.data.jsonl import read_jsonl
    from tuned.data.paths import build_paths
    from tuned.data.store import Store

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data_law_v1.yaml")
    parser.add_argument("--selection", default=None, help=f"default corpus/{SELECTION_FILENAME}")
    parser.add_argument("--out", default=None, help=f"default corpus/{EXTRACTION_FILENAME}")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after N documents extracted, quarantined or FAILED; documents "
        "already done are skipped without spending the cap, so a resumed run advances",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-extract documents the index already has, at this version",
    )
    parser.add_argument("--max-failures", type=int, default=DEFAULT_MAX_FAILURES)
    parser.add_argument(
        "--allow-scanned-era",
        action="store_true",
        help="admit the OCR-era volumes (JBIG2 scans with an ABBYY text layer, "
        "2010-2017) instead of quarantining them as scanned_era. The structure is "
        "recorded on the row either way; this decides whether it refuses",
    )
    parser.add_argument(
        "--audit",
        type=int,
        default=0,
        metavar="N",
        help="after the pass, re-read N documents and print the boundary in context "
        "(half of them refusals). A fully-resumed run does no work and goes straight "
        "to the audit, which is how an existing extraction is audited",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error(f"--limit must be at least 1, got {args.limit}")

    cfg = load_build_config(args.config)
    paths = build_paths(cfg.build.workdir).ensure()
    selection = Path(args.selection) if args.selection else paths.corpus_dir / SELECTION_FILENAME
    out_path = Path(args.out) if args.out else paths.corpus_dir / EXTRACTION_FILENAME
    text_root = paths.corpus_dir / TEXT_DIRNAME
    if not selection.exists():
        print(
            f"no selection at {selection}\n"
            f"  run: python -m tuned.data.select --config {args.config}\n"
            f"  (and before that, python -m tuned.data.acquire --kind all)"
        )
        return 2

    store = Store.open(paths.state_db)
    code = 0
    try:
        index = pdf_index(store)
        print(
            f"selection {selection}  ->  {text_root}  "
            f"({len(index)} PDFs indexed, extract_version {EXTRACT_VERSION})"
        )
        stats = extract_corpus(
            store,
            read_jsonl(selection),
            index=index,
            text_root=text_root,
            reader=reader if reader is not None else read_pdf_pages,
            limit=args.limit,
            force=args.force,
            max_failures=args.max_failures,
            allow_scanned_era=args.allow_scanned_era,
        )
        joined = stats["routes"][ROUTE_PDF_KEY] + stats["routes"][ROUTE_SCR_PREFIX]
        print(
            "  join      "
            + "  ".join(f"{route} {stats['routes'][route]}" for route in JOIN_ROUTES)
        )
        print(
            f"  extracted {stats['extracted']}  quarantined {stats['quarantined']}  "
            f"skipped {stats['skipped']}  failed {stats['failed']}  "
            f"{_fmt_bytes(stats['chars'])} of text"
        )
        for reason, count in sorted(stats["reasons"].items()):
            print(f"    quarantine[{reason}]: {count}")
        reader_meta = stats["reader"]
        print(
            f"  reader    {reader_meta['lane']}"
            f"  (pymupdf4llm {reader_meta['pymupdf4llm'] or 'n/a'})"
        )
        if not stats["structure_probed"]:
            print(
                "    STRUCTURAL GATES DID NOT RUN: this reader reports no PDF "
                "structure, so nothing was checked for scan-era (JBIG2/OCR-layer) "
                "or undecodable-font objects. A run with no scanned_era refusals "
                "reads the same either way - this line is the difference."
            )
        if stats["duplicate_rows"]:
            print(
                f"    {stats['duplicate_rows']} selection rows named a PDF another row had "
                f"already taken - first row wins, in the index and in the manifest alike"
            )
        if stats["reportable"]:
            print(
                f"    {stats['reportable']} documents carry a REPORTABLE / NON-REPORTABLE "
                f"flag. That flag belongs to the COURT-RELEASED judgment PDF, not to the "
                f"S.C.R. reprint (P0 found it in 2 of 70), so read a non-null flag as "
                f"'this object may not be a reprint' - which is exactly the object whose "
                f"boundary rules may not apply. Put these in the --audit sample."
            )
        decided = stats["extracted"] + stats["quarantined"]
        if decided and stats["quarantined"] / decided > QUARANTINE_ALARM:
            print(
                f"  HIGH QUARANTINE RATE {stats['quarantined'] / decided:.1%} - read "
                f"`--audit 20` before trusting this run. The refusals are supposed to be "
                f"a minority; at this rate the boundary rules and the source are what to "
                f"check, not the individual documents."
            )
        for failure in stats["failures"]:
            print(f"    failed {failure['key']}: {failure['error']}")
        if stats["failed"]:
            code = max(code, 1)

        written = write_manifest(store, read_jsonl(selection), out_path, index=index)
        print(f"wrote {written} rows -> {out_path}")
        store.log_event(
            "corpus_extraction",
            {
                **{k: v for k, v in stats.items() if k != "failures"},
                "failures": len(stats["failures"]),
                "selection": str(selection),
                "out_path": str(out_path),
                "extract_version": EXTRACT_VERSION,
                "manifest_rows": written,
            },
        )
        # STANDING totals, not this pass's: a resumed run does no work and
        # would otherwise print zeros over a finished corpus, and the split
        # is what says how large the refusal set has grown.
        print(
            f"documents indexed -> {store.document_count(SC_SOURCE_ID)}"
            f" ({store.document_count(SC_SOURCE_ID, status=STATUS_OK)} emitted /"
            f" {store.document_count(SC_SOURCE_ID, status=STATUS_QUARANTINED)} quarantined)"
            f"  ({paths.state_db})"
        )
        if stats["considered"] and not joined:
            # Every selected judgment failing to find a PDF is not a corpus,
            # it is a wrong assumption about keys - either acquire has not
            # run, or the metadata link column and the S.C.R. filename
            # inference are both wrong. Exiting 0 would report an empty
            # extraction as a finished one.
            print(
                "  NOTHING JOINED: not one selected judgment matched an acquired PDF. "
                "Check that `acquire --kind pdf` has run, and compare the join counts "
                "above against the pdf_key coverage select.py printed."
            )
            code = max(code, 1)
        if args.audit:
            print()
            print(audit_report(store, args.audit, index=index,
                               reader=reader if reader is not None else read_pdf_pages))
    finally:
        store.close()
    return code


if __name__ == "__main__":
    import sys

    exit_code = main()
    # Same reasoning as acquire.py/select.py: pymupdf/pyarrow can leave
    # non-daemon threads that wedge interpreter shutdown after all output is
    # written. Skip shutdown entirely.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
