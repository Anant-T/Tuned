"""Drop every candidate row that overlaps an eval set - and REFUSE to run blind.

Input is the assembly stream (the JSONL streams replay.py/curated.py wrote,
plus the accepted generations in the store); output is
`out/decontaminated.jsonl`, a drop log with a machine-readable reason per
dropped row, and a manifest that says what this pass could and could not see.

THE ONE THING THIS MODULE EXISTS TO PREVENT
-------------------------------------------
The project's headline number is a blind pairwise judge run plus
BhashaBench-Legal as a forgetting guard. A leaked eval question does not make
that number noisy - it makes it wrong in the FLATTERING direction, and
nothing downstream can detect it. So the failure to design against is not an
imprecise matcher; it is

    THE MATCHER NEVER RAN AND THE RUN STILL EXITED 0.

Hence the rule that outranks every other decision here: **an eval set that is
absent, unreadable or empty is a REFUSAL, not a warning.** Nothing is
written - no output file, no "decontaminated" stamp - and the CLI exits 2
naming the set and the command that fixes it. An operator who genuinely wants
to proceed says so per set (`--allow-missing-eval bbl`, the fleet's
`--allow-pool-gaps` idiom), and that override is written INTO THE MANIFEST,
not merely printed, so stats.py and the dataset card carry it forward. A
dataset built without BBL screening must be able to say so about itself years
later.

THE ASYMMETRY THAT DECIDES EVERY JUDGEMENT CALL HERE
----------------------------------------------------
A false negative (an eval row survives into training) is invisible,
permanent, and inflates the metric the project is judged by. A false positive
costs one row out of ~18,000. These are not comparable, so every borderline
call in this module goes to DROPPING - and every drop carries a
machine-readable reason so the yield price stays auditable rather than
mysterious. (dedupe.py is NOT in this regime; see its docstring.)

THE LEVELS, and the case each one alone must carry
--------------------------------------------------
1. TEXT CONTAINMENT - the share of an eval item's 13-grams that appear in the
   row is >= CONTAINMENT (0.5). Catches the eval question quoted, verbatim or
   near-verbatim, inside a much longer training row. Applies to items of 38
   tokens or more; below that the 13-gram window cannot deliver "near".
2. NARROW-WINDOW CONTAINMENT - the same rule at a SMALLER window, for items
   between 13 and 37 tokens. See THE WINDOW IS LENGTH-AWARE below: this is
   the band BhashaBench-Legal lives in, and at a fixed 13 the rule there is an
   exact-match rule wearing a near-match label.
3. SHORT-ITEM CONTAINMENT - an eval item shorter than 13 tokens has no
   13-grams at all. It is matched WHOLE, and nothing less than the whole of it
   counts. Items below SHORT_MIN_TOKENS (5) are matchable by nothing here and
   are COUNTED as `unmatchable`, per set, because a 3-token question would
   otherwise match half the corpus - and a set whose items are ALL unmatchable
   is a refusal, not a clean run.
4. CASE-IDENTIFIER OVERLAP - CNR, neutral/reporter citation, or normalised
   case title shared between a row and an eval item. This is the level that
   catches SAME JUDGMENT, DIFFERENT QUESTION, which no n-gram method can ever
   see (PredEx and IL-TUR CJPE draw on the same appellate pools), and it is
   the cheapest and most certain of the four.

THE WINDOW IS LENGTH-AWARE, and the measurement that forced it
---------------------------------------------------------------
One substituted token destroys every gram covering it - up to n of them. An
item of L tokens has G = L - n + 1 grams, so after one central edit its
containment in a row carrying the rest is (G - n)/G, which reaches 0.5 only
when L >= 3n - 1. At n = 13 that is L >= 38 tokens. Measured:

    item tokens  |  13     25     29     35     38     50     150
    C after 1 edit  0.000  0.000  0.235  0.435  0.500  0.658  0.906

So at a fixed 13 the rule is EXACT MATCH below ~38 tokens, whatever the
threshold says - and BBL is 24,365 MCQ questions, most of them shorter than
that. The window is therefore the largest one that survives a single central
edit at this item's length:

    window_for(L) = min(13, (L + 1) // 3)        for L >= 13
                  = L      (whole sequence)      for 5 <= L < 13
                  = nothing                      below 5

which puts C after one edit at or above 0.5 at EVERY length from 13 up, and
leaves items of 38 tokens and more exactly where they were.

WHAT IT COSTS, because it is a real cost and run one has to read it. The
weakest evidence the rule will accept is one contiguous run of ceil((L + w -
1)/2) tokens shared with the row, where w is the window at that length. All
measured, not derived from the approximation:

    item tokens      13    16    20    25    30    38+
    run needed @13   13    14    16    19    21    unchanged
    run needed now    8    10    13    16    20    unchanged

So the exposure widens by at most 5 tokens, and only for items under 38. (The
closed form (2L - 1)/3 quoted here before is the approximation you get by
substituting w = (L + 1)/3 and it is wrong by up to a token in both
directions - at L = 25 it says 16.3 where the rule actually needs 16, and at
L = 30 it says 19.667 where the rule needs 20.)

THE FALSE-POSITIVE CLASS THIS BUYS, named from measurement rather than from
first principles: it is QUESTION BOILERPLATE AT 13-16 TOKENS, not statutory
quotation. Three realistic MCQ stems - "under which section of the indian
penal code is criminal breach of trust punishable" (14 tokens), "what is the
limitation period prescribed for filing an appeal against a decree" (13) -
score C = 0.800 against unrelated rows that merely share ordinary legal
phrasing, where at a fixed 13 they score 0.000. The stem, not the subject
matter, is what collides.

That is why the narrow level is counted separately: if `narrow` drops dominate
run one, the calibrated relaxation is to raise the divisor (3 -> 2.5 costs the
one-edit guarantee below ~20 tokens and buys two tokens of evidence back). Do
not change it before reading that count.

The band edges, because a reader deciding whether to move the divisor needs
them: at L = 13 the bar is 8 contiguous tokens, from L = 17 it is 11 or more,
and the sharpest discontinuity in the whole design is at 12/13 - where an item
goes from needing 100% of itself (the `short` rule matches whole) to needing
62% - and not at the 37/38 window cap.

WHAT IS STILL NOT DELIVERED, stated rather than implied:

* Below 38 tokens the guarantee is ONE substituted token ANYWHERE, and after
  that placement decides. It is not the cliff this docstring used to claim.
  Measured over every placement at L = 20 (window 7, 14 grams):

      1 edit    20 of 20 placements drop      C 0.500 - 0.929
      2 edits   63 of 190 placements drop     C 0.000 - 0.857
      3 edits  140 of 1,140 placements drop   C 0.000 - 0.786

  The survivors are the edits within ~5 tokens of an END of the item, where a
  token is covered by fewer windows and so destroys fewer grams: two edits at
  positions 0 and 1 leave C = 0.857, two in the middle leave 0.429. So the
  module is MORE sensitive than it was documented to be, in the direction that
  matters. Above 38 the window stops shrinking and the slope is gentle
  everywhere (a 100-token item scores 0.85 after one edit and 0.26 after
  five). Measured in
  test_the_narrow_band_tolerates_exactly_one_edit_and_no_more.
* A 5% word-level paraphrase measures C ~ 0.33 at 400 tokens and above, ~0.26
  at 100 and ~0.07 at 40 - i.e. it is never caught by any exact rule at this
  threshold, and SHORTER items are worse, not better. That is what the
  semantic layer is for. It is OPTIONAL, and what it delivers is now measured
  rather than assumed: at the shipped threshold it catches a verbatim or
  lightly reworded question inside a row of any length, and it does NOT catch
  a heavy paraphrase without also collapsing questions about the same section.
  See SEMANTIC_THRESHOLD for the table.
* THE SEMANTIC LAYER IS LATIN-SCRIPT ONLY on the shipped embedding model, and
  the manifest says so per script rather than implying otherwise.
  `potion-base-8M` cannot separate a rewording from an unrelated sentence in
  Devanagari at all: measured on this module's own control constants, the
  Hindi item scores 0.990 against its own rewording and 0.962 against a
  cricket report - a gap of 0.028 - where the English pair is 0.955 against
  0.089, a gap of 0.866. Its Hindi half is not weak, it is inverted
  drops 4 of 4 clean Hindi rows at every threshold from 0.6 to 0.95. 7,318 of
  BhashaBench-Legal's 24,365 questions are Hindi, and every one of them is
  screened by the exact stack alone. See dominant_script.
* A MIXED-SCRIPT ROW IS SPLIT, NOT ROUTED WHOLE, and that is what keeps the
  sentence above from being a lie by omission. Cross-script words inside a
  probe text dilute its embedding even when the probe reached the right index,
  so an English eval question quoted verbatim in a Hindi row was KEPT five
  times out of five while the manifest recorded no hole in the English screen.
  Each window is now probed with one script's words at a time and every script
  it could not read is counted per row. See script_partition.

WHERE THIS DEPARTS FROM THE PLAN, and the measurement that forced it
--------------------------------------------------------------------
The plan (and the brief) specify "13-gram Jaccard >= 0.8". Jaccard cannot
express the leak this module is for. A 30-token eval question sitting
verbatim inside a 2,500-token training row scores

    Jaccard = |A n B| / |A u B| ~ 18 / 2,500 ~ 0.007

- three orders of magnitude below 0.8 - while its containment in the row is
  1.0. (Measured on the fixture in
  test_the_jaccard_rule_the_brief_asks_for_cannot_see_a_verbatim_leak.) A Jaccard threshold only fires when the two texts are
  the SAME LENGTH and nearly identical, i.e. when the training row IS the
  eval item. That case is real but rare; the common one is quotation inside a
  longer row, and Jaccard is blind to it.

So the rule is CONTAINMENT, at 0.5, and Jaccard is still computed and
recorded on every drop as a diagnostic. Jaccard is NOT kept as a second rule
because it would be a branch with no case of its own: J <= C always
(|A n B|/|A u B| <= |A n B|/|B|), so {J >= 0.8} is a strict subset of
{C >= 0.5} and the branch could never fire alone. Pinned by
test_the_jaccard_rule_the_brief_asks_for_is_subsumed_by_containment.

Containment at 0.5 (rather than "one shared 13-gram", the GPT-3/LLaMA rule)
is what keeps the spec's standing exception: a shared STATUTE QUOTATION alone
is not contamination - statutes are the domain being taught. One quoted
provision inside a 200-token eval item is ~10% of its grams and survives;
the question itself reproduced is 100% and does not.

THE CANDIDATE STEP CONTRIBUTES ZERO FALSE NEGATIVES
---------------------------------------------------
Candidates come from an inverted index over eval-item 13-gram hashes: any row
with containment > 0 necessarily shares at least one 13-gram with the item,
so the index is EXACT and the expensive containment/Jaccard arithmetic runs
only on candidates. MinHashLSH is deliberately NOT used here - LSH is an
approximation whose error direction is MISSED PAIRS, and missed pairs are the
catastrophic direction. The exactness is asserted against brute force in
test_the_gram_index_finds_every_pair_brute_force_finds, not merely claimed.

Gram hashes are a rolling polynomial over crc32 token hashes, NOT Python's
`hash()`: `hash()` of a str is salted per process, so an index built with it
would give a different candidate set (and therefore a different dataset) on
every run.

WHAT IL-TUR IS SCREENED AGAINST, and the arithmetic behind it
--------------------------------------------------------------
IL-TUR is ~488,000 rows across 8 heterogeneous configs, and `bail` alone is
353,698 of them (datasets-server, 2026-08-14). The eval index costs, measured:
~220 bytes per DISTINCT GRAM (319 MB over 1.46M grams at 20,300 items), and an
item of L tokens contributes about L - w + 1 grams, which is ~L for the
judgment-length items IL-TUR is full of. So even at a conservative 200 tokens
an item, the whole repo is

    488,000 x ~190 grams x ~220 B  ~  20 GB of index,

of which `bail` is three quarters. That does not fit on the machine this
build runs on, and a screen that OOMs screens nothing at all - which is this
module's one forbidden outcome, arrived at from the other side.

THE DECISION: the eval surface is the TEST-TYPE SPLITS OF EVERY CONFIG
(`test`, `test_specific`, `expert`), with `train_all`, `fold_N`, `dev` and
`val` excluded. Those are the splits the IL-TUR tasks are scored on, so they
are what a leak into training would flatter; the training splits of an eval
benchmark are material we could legitimately have learned from anyway.

Two things follow, and both are recorded rather than implied:

* Selection matches the SPLIT NAME as a path component, never a config name.
  The config names were not verified against the Hub and a guessed one would
  silently exclude a whole task. `manifest["eval_sets"][key]["selection"]`
  lists every object read and every object left out WITH THE REASON, and an
  EXCLUDED name beats an included one in the same key.
* If NO object names a screened split, that is a REFUSAL (`no_screened_split`),
  decided before a single row is read. It used to read the whole repo instead,
  on the argument that over-screening is safe - which stopped being true the
  moment the eval surface became a few thousand rows of a 488,000-row repo,
  since "read everything" is the ~20 GB of index this section exists to avoid.
  A set with NO filter (aibe) is untouched by that rung: reading everything is
  its normal path rather than a fallback from anything.
* No per-config+split row count has been read for IL-TUR, so its floor and its
  shortfall line are OFF and the run says so. BBL (17,047 + 7,318 = 24,365
  across two configs of `test`) and aibe (1,157, single `train` split, which
  IS the set) are verified and carry both. Getting IL-TUR's per-split counts
  is a first-run item.

WHAT THIS MODULE CANNOT SEE (read before trusting a green run)
--------------------------------------------------------------
* A PARAPHRASED eval question. n-grams are defeated by rewriting; the
  semantic layer (semhash) is the intended answer and is optional - its
  status is recorded in the manifest either way, and `--require-semantic`
  turns its absence into a refusal. Its status is PER SCRIPT
  (`semantic_scripts`), because a layer that ran over the English two thirds
  of BBL and not the Hindi third is not the same screen as one that ran.
* An eval item under 5 tokens (counted PER SET - and a set whose items
  are all under it is refused, because a screen that compared against
  nothing is not a screen).
* A row whose case identity nothing records. `case_identifier_coverage` in
  the manifest is that instrument: if it reads 0, the case-identifier level
  did not run, and the run says so loudly rather than reporting clean. NOTHING
  in the pipeline
  populates seed.cnr or seed.neutral_citation today (grep-verified), so on
  the first real build this level rests entirely on identifiers found in the
  row text and on `_prov`.

Build:  python -m tuned.data.decontaminate --config configs/data_law_v1.yaml
        [--in PATH]... [--out PATH] [--allow-missing-eval KEY]...
        [--no-generated] [--no-case-id-from-text] [--require-semantic]
"""

import csv
import hashlib
import json
import math
import os
import re
import unicodedata
import zlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tuned.data.acquire import HF_SOURCES
from tuned.data.citations import extract_citations
from tuned.data.select import landmark_key

OUT_FILENAME = "decontaminated.jsonl"
DROPS_FILENAME = "decontamination_drops.jsonl"
MANIFEST_FILENAME = "decontamination.json"

# Bump when a rule changes: the manifest records it, so a dataset card can say
# which decontamination rules produced the corpus it describes.
#
# 2: the manifest gained an `output` block (path, rows, sha256) and dedupe.py
#    binds its chain of custody to that DIGEST rather than to the directory
#    the manifest happens to sit in.
# 3: the eval sets carry per-config+split expectations and a `selection`
#    record (what was screened against, what was left out, and why) in place
#    of a single `split` string; the semantic layer probes a row in windows at
#    a measured threshold and both are recorded in `thresholds`;
#    `identifier_drops` counts every identifier a drop matched rather than the
#    sorted-first one; and the tokeniser keeps combining marks inside a word,
#    which moves Devanagari items between levels. The rules moved AND the
#    manifest changed shape.
# 4: the semantic layer is PER SCRIPT - one index per script, every window
#    SPLIT into the scripts it carries rather than routed whole, screening only
#    the scripts whose own control half passes, with `semantic_scripts`
#    recording every script it could not screen and why; every semantic drop
#    now names the eval set and item it matched and the score, where it used to
#    record `*`/`semhash`; the probe window widened from 20 to 30 words so that
#    every 21-word span of a row lies wholly inside one window at every offset;
#    and a filtered eval set whose filter selects nothing is a REFUSAL rather
#    than a silent read-everything fallback.
DECON_VERSION = 4

# The n-gram window AT ITS WIDEST. 13 tokens is long enough that ordinary
# legal phrasing does not collide by accident; window_for narrows it for items
# too short to tolerate an edit at 13.
NGRAM = 13
# Share of an eval item's grams that must appear in the row. See the docstring
# for why this is containment and not Jaccard, and why it is not "one shared
# gram".
CONTAINMENT = 0.5
# An eval item shorter than NGRAM is matched whole; below this it is matchable
# by nothing (a 3-token question would drop half the corpus) and is counted.
SHORT_MIN_TOKENS = 5
# A case title shorter than this is not a usable identifier ("state v kumar").
TITLE_MIN_TOKENS = 4

LEVEL_TEXT = "text"
LEVEL_NARROW = "narrow"
LEVEL_SHORT = "short"
LEVEL_CASE_ID = "case_id"
LEVEL_SEMANTIC = "semantic"
# The order they are reported in: cheapest and most certain first. The
# semantic layer is not in NGRAM_LEVELS because it is not part of the exact
# screen - it only ever adds drops on top of it.
NGRAM_LEVELS = (LEVEL_CASE_ID, LEVEL_TEXT, LEVEL_NARROW, LEVEL_SHORT)
LEVELS = NGRAM_LEVELS + (LEVEL_SEMANTIC,)


# --------------------------------------------------------------------------
# Text primitives. dedupe.py imports these - ONE definition of what a row's
# text is, so the two passes cannot disagree about it.
# --------------------------------------------------------------------------

# A WORD IS ITS LETTERS PLUS ITS COMBINING MARKS, and getting that wrong is
# not a cosmetic bug in this module - it moves an item between levels.
#
# `[a-z0-9]+` drops every Devanagari word outright (a Hindi BhashaBench
# question would compare as a handful of digits). `[^\W_]+` reads the letters
# but SPLITS AT EVERY MATRA: Devanagari vowel signs and the virama are
# categories Mn/Mc, which Python's `\w` does not include, so `भारतीय` came out
# as 3 tokens and `दंड` as 2. Measured on a 16-word Hindi question: 29 tokens
# instead of 16.
#
# That inflation is not a rounding error here. The window design guarantees
# survival of exactly one TOKEN edit (window_for), so one edited Hindi WORD
# was 2-3 token edits and the guarantee evaporated - measured 3 of 16 word
# positions missed against 0 of 18 on the English equivalent. It also made a
# two-word Hindi stock phrase (`अपील खारिज`, 5 tokens inflated) clear the
# 5-token floor and become matchable, where English "appeal dismissed" (2
# tokens) correctly cannot match anything. And it miscalibrated the
# item-length histogram at ~2x on the Hindi half of BBL - 7,318 of its 24,365
# questions.
#
# The class is derived from unicodedata rather than hand-listed, over the BMP
# ranges that carry Indic and diacritic marks: U+0300-U+1DFF (combining
# diacriticals, every mainline Indic block, Combining Diacritical Marks
# Extended, VEDIC EXTENSIONS at U+1CD0-U+1CFF and the Supplement at
# U+1DC0-U+1DFF) plus U+A8E0-U+A8FF for Devanagari Extended, which sits far
# outside the first range. It is NOT "every Indic block" - the comment here
# used to say so while stopping at U+1AFF, which left U+1CDA (Vedic tone
# marker) splitting a word in a Sanskrit citation. Measured cold at import:
# 1.0-1.1 ms over 152 ranges. (The comment here said "under a millisecond" for
# the narrower scan, which measured 3.65 ms on the reviewer's machine and
# 1.4 ms on this one - it is a small number, not an unmeasured one.)
def _combining_marks() -> str:
    """The body of a regex character class holding every combining mark."""
    import unicodedata

    ranges: list[list[int]] = []
    for code in [*range(0x0300, 0x1E00), *range(0xA8E0, 0xA900)]:
        if unicodedata.category(chr(code)).startswith("M"):
            if ranges and code == ranges[-1][1] + 1:
                ranges[-1][1] = code
            else:
                ranges.append([code, code])
    return "".join(
        chr(lo) if lo == hi else f"{chr(lo)}-{chr(hi)}" for lo, hi in ranges
    )


# A token STARTS with a letter or digit and may carry marks after it: a lone
# matra following punctuation is not a word, and requiring the first character
# to be alphanumeric is what keeps that out without a second rule.
_TOKEN = re.compile(rf"[^\W_](?:[^\W_]|[{_combining_marks()}])*", re.UNICODE)

# ZERO-WIDTH NON-JOINER and ZERO-WIDTH JOINER, STRIPPED rather than added to
# the class above, and the difference between those two decisions is the whole
# point. They are category Cf, so `\w` excludes them and a word carrying one
# used to split there: `प्रतिवादी` written with a ZWNJ read as two tokens, and a
# three-word phrase spelled with them read as five - the same inflation as the
# matra bug one category over, with the same two consequences (a one-word edit
# stops being a one-token edit, and a short phrase clears the 5-token floor and
# becomes falsely matchable).
#
# ADDING them to the class would fix the split and leave the two spellings of
# the same word unequal, which is the fault that matters here: an eval item and
# a training row that differ only in an invisible joiner would compare as
# different words. STRIPPING makes them equal, and equal is what a screen needs.
_ZERO_WIDTH = str.maketrans({"‌": None, "‍": None})
_MASK = (1 << 64) - 1
_BASE = 1_000_003


def tokens(text: str) -> tuple[str, ...]:
    """Case-folded word tokens IN ANY SCRIPT - the unit both passes compare in.

    Punctuation, markup and whitespace are dropped, so a row that differs
    from an eval item only in typesetting still matches it. Digits are KEPT:
    section numbers and years are most of what distinguishes one legal
    question from another. COMBINING MARKS STAY INSIDE THE WORD THEY MODIFY
    and ZERO-WIDTH JOINERS ARE REMOVED FIRST, so the joiner and joiner-less
    spellings of a word are the same token - see _TOKEN and _ZERO_WIDTH for
    the measurements that forced both.
    """
    return tuple(_TOKEN.findall((text or "").lower().translate(_ZERO_WIDTH)))


def gram_hashes(toks: Sequence[str], n: int = NGRAM) -> frozenset[int]:
    """Stable 64-bit hashes of every n-token window.

    STABLE is the load-bearing word. Python's `hash()` for str is salted per
    process (PYTHONHASHSEED), so an index keyed on it selects a different
    candidate set - and therefore a different dataset - on every run.
    crc32 is deterministic across runs, machines and Python versions; the
    rolling polynomial over it is what keeps this one multiply per token
    rather than n.

    A 64-bit collision can only ever ADD to a posting count, never remove
    from one, so the error direction is the safe one - it can cost a row, not
    hide a leak. It is NOT rejected downstream, though: the posting count IS
    the intersection size the arithmetic divides, so a collision inflates
    containment directly rather than proposing a candidate something later
    throws out. (The comment here used to claim the latter, which would have
    been a stronger guarantee than the code gives.)
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if len(toks) < n:
        return frozenset()
    hs = [zlib.crc32(t.encode("utf-8")) for t in toks]
    power = pow(_BASE, n, 1 << 64)
    window = 0
    for value in hs[:n]:
        window = (window * _BASE + value) & _MASK
    out = [window]
    for i in range(n, len(hs)):
        window = (window * _BASE - hs[i - n] * power + hs[i]) & _MASK
        out.append(window)
    return frozenset(out)


def window_for(length: int, *, n: int = NGRAM, short_min: int = SHORT_MIN_TOKENS) -> int:
    """The n-gram window an eval item of `length` tokens is screened at.

    0 means NOTHING here can match it (the item is under the floor).

    THE DERIVATION, not a tuning knob. One substituted token destroys every
    gram covering it - n of them, centrally. An item of L tokens has
    G = L - n + 1 grams, so a row carrying the rest of it holds (G - n)/G of
    them, which is >= CONTAINMENT (0.5) exactly when G >= 2n, i.e. when

        L >= 3n - 1.

    Turned round, the widest window that still tolerates one edit at length L
    is (L + 1) // 3. Above 3*NGRAM - 1 = 38 tokens that bound exceeds NGRAM
    and the cap takes over, so long items are screened exactly as before; the
    cap is what keeps a 1,200-token IL-TUR judgment from being gramed at 400,
    where three edits anywhere would take its containment to zero.

    Below NGRAM an item has no grams at all, so it is matched WHOLE - the
    window is its own length and nothing short of all of it counts. Below
    `short_min` it is matchable by nothing: `unmatchable`, counted per set,
    and a set that is entirely unmatchable is a refusal.
    """
    if length < short_min:
        return 0
    if length < n:
        return length
    return min(n, (length + 1) // 3)


def level_for(length: int, *, n: int = NGRAM, short_min: int = SHORT_MIN_TOKENS) -> str | None:
    """Which text level screens an item of this length, or None for nothing.

    Three bands, and each carries a case the others cannot:
    `text` a 150-token judgment summary quoted at 55% of its grams;
    `narrow` a 20-token MCQ question with ONE word changed (containment 0.000
    at a fixed 13, and not a whole-sequence match either);
    `short` a 9-token question reproduced entire.
    """
    window = window_for(length, n=n, short_min=short_min)
    if not window:
        return None
    if window >= n:
        return LEVEL_TEXT
    if length < n:
        return LEVEL_SHORT
    return LEVEL_NARROW


def jaccard_from(shared: int, n_a: int, n_b: int) -> float:
    """|a n b| / |a u b| from the three counts.

    The counted form is the definition because the index already knows how
    many grams a candidate shares (it counted the postings), and re-deriving
    it with a set intersection would be both slower and a second answer to
    the same question.
    """
    if not n_a or not n_b:
        return 0.0
    return shared / (n_a + n_b - shared)


def containment_from(shared: int, n_part: int) -> float:
    """Share of `part` that the other side also holds."""
    return (shared / n_part) if n_part else 0.0


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    """Set form of jaccard_from - two texts too short to gram are not
    'identical', so the empty case reads 0.0 rather than raising."""
    return jaccard_from(len(a & b), len(a), len(b))


def containment(part: frozenset[int], whole: frozenset[int]) -> float:
    """Share of `part` that `whole` also holds.

    Asymmetric on purpose: `part` is the eval item and `whole` is the
    training row, and the question is "how much of the eval item is in this
    row", not "how similar are they".
    """
    return containment_from(len(part & whole), len(part))


# --------------------------------------------------------------------------
# The row contract.
#
# An assembly row is {"messages": [...], "_prov": {...}} - replay.py and
# curated.py's shape, and the shape store rows are lifted into below. The row
# dict is never mutated: everything this pass derives lives on the Item beside
# it, so the bytes written out are the bytes read in.
# --------------------------------------------------------------------------

def row_messages(row) -> list[dict]:
    messages = row.get("messages")
    return [m for m in messages if isinstance(m, dict)] if isinstance(messages, list) else []


def row_prompt(row) -> str:
    """Everything shown to the model - the question and its grounding."""
    return "\n".join(
        str(m.get("content") or "") for m in row_messages(row) if m.get("role") != "assistant"
    )


def row_answer(row) -> str:
    return "\n".join(
        str(m.get("content") or "") for m in row_messages(row) if m.get("role") == "assistant"
    )


def row_prov(row) -> dict:
    prov = row.get("_prov")
    return prov if isinstance(prov, dict) else {}


def row_form(row) -> str:
    """The QUESTION FORM this row is an instance of.

    Two rows built on the same judgment in two different forms are two
    examples, not one duplicate - which is why dedupe.py's prompt rule and
    the per-case cap both read this. Falls back to the stream/source so rows
    that carry no task identity are still grouped by something real.
    """
    prov = row_prov(row)
    for key in ("form", "task_type", "prompt_id", "stream", "source"):
        value = prov.get(key)
        if value:
            return str(value)
    return ""


# CNR: 16 characters - 4-letter state/court code, 2-digit establishment,
# 6-digit case number, 4-digit year (ESCR010004512020). Written with or
# without separators; fenced on both sides so it cannot be carved out of a
# longer alphanumeric run.
_CNR = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{4}[-]?\d{2}[-]?\d{6}[-]?\d{4})(?![A-Za-z0-9])")

_ID_FIELDS = {
    "cnr": ("cnr", "cnr_number", "case_number_cnr"),
    "cit": ("neutral_citation", "citation", "law_report_citation", "case_citation"),
    "title": ("case_name", "case_title", "title", "name", "native_id"),
}


def cnr_keys(text: str) -> list[str]:
    """Normalised CNRs in `text` (separators dropped, upper-cased)."""
    return [m.group(1).replace("-", "").upper() for m in _CNR.finditer(text or "")]


def title_key(text: str) -> str | None:
    """A case name as a join key, or None if it is too generic to be one.

    select.landmark_key is the normalisation - one definition of "these two
    strings name the same case" for the whole build. The length floor is what
    keeps 'state v kumar' from joining half the criminal docket.
    """
    key = landmark_key(text or "")
    return key if len(key.split()) >= TITLE_MIN_TOKENS else None


def identifiers_from_fields(fields) -> set[str]:
    """Namespaced identifiers from explicit metadata (never from prose)."""
    out: set[str] = set()
    for namespace, names in _ID_FIELDS.items():
        for name in names:
            value = fields.get(name)
            if not value:
                continue
            text = str(value)
            if namespace == "cnr":
                out.update(f"cnr:{key}" for key in cnr_keys(text))
            elif namespace == "cit":
                out.update(f"cit:{key}" for key in extract_citations(text))
            else:
                key = title_key(text)
                if key:
                    out.add(f"title:{key}")
    return out


def identifiers_from_text(text: str) -> set[str]:
    """Identifiers a passage NAMES: CNRs and citations, never titles.

    Titles are not extractable from prose without inventing a parser, and a
    guessed one would generate identifiers no other side can match. The
    citations found here are the authorities the passage cites AS WELL AS its
    own - which is deliberate on the eval side and the reason the row side of
    this channel is separately counted and switchable (see
    `case_ids_from_text`): if one landmark citation turns out to account for
    a large share of the drops, that is the citation graph, not
    contamination, and the manifest's `top_identifiers` is where it shows up.
    """
    out = {f"cnr:{key}" for key in cnr_keys(text)}
    out.update(f"cit:{key}" for key in extract_citations(text or ""))
    return out


@dataclass(frozen=True)
class Item:
    """One candidate row plus everything derived from it, computed once.

    `row` is untouched: the output file is the input rows minus the drops,
    byte for byte.
    """

    row: dict
    origin: str
    key: str
    prompt: str
    answer: str
    form: str
    identifiers: frozenset[str]

    @property
    def text(self) -> str:
        return f"{self.prompt}\n{self.answer}"


def item_key(prompt: str, answer: str) -> str:
    """Stable content id for a row - the same bytes always get the same key."""
    payload = f"{prompt}\n\x00\n{answer}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def item_of(row: dict, origin: str, *, ids_from_text: bool = True) -> Item:
    prompt, answer = row_prompt(row), row_answer(row)
    prov = row_prov(row)
    identifiers = identifiers_from_fields(prov)
    if ids_from_text:
        # The PROMPT only. An identifier in the answer is an authority the
        # model cited, not the case the row is about.
        identifiers |= identifiers_from_text(prompt)
    return Item(
        row=row,
        origin=origin,
        key=item_key(prompt, answer),
        prompt=prompt,
        answer=answer,
        form=row_form(row),
        identifiers=frozenset(identifiers),
    )


# --------------------------------------------------------------------------
# The eval corpora.
# --------------------------------------------------------------------------

# When the per-config+split row counts below were read off the Hub's
# datasets-server. An edit to one of them is a decision, not a typo-fix.
EVAL_COUNTS_VERIFIED_AT = "2026-08-14"


@dataclass(frozen=True)
class EvalPart:
    """One config+split of an eval set, and the rows the Hub reported for it.

    `config` is descriptive - the label the datasets-server gives that view of
    the repo - because selection matches on the SPLIT name in a file's path,
    which is the only thing a snapshot on disk actually carries. `rows` is None
    where the count has not been read, and a None anywhere in a set's parts
    turns that set's floor and shortfall instrument off rather than
    denominating them against a guess.
    """

    config: str
    split: str
    rows: int | None = None


@dataclass(frozen=True)
class EvalSet:
    """One eval corpus: acquire.py owns WHERE it comes from, this owns WHY.

    repo_id/license/source_id are read off acquire.HF_SOURCES rather than
    repeated here - two spellings of a repo id is how the acquire command
    this module prints stops matching the source id it then looks up.

    `include_splits`/`exclude_splits` are the EVAL SURFACE: which splits of the
    repo this dataset is screened against. They are matched as NAME COMPONENTS
    of an object key, so `test_specific-00000-of-00003.parquet` is selected by
    "test" and `train_all-...` is excluded by "train".
    """

    key: str
    why: str
    include_splits: tuple[str, ...] = ()
    exclude_splits: tuple[str, ...] = ()
    parts: tuple[EvalPart, ...] = ()
    # Why this surface and not the whole repo. Printed and recorded, because a
    # subset that is not named narrows the guarantee silently.
    selection_note: str = ""

    @property
    def expect_rows(self) -> int | None:
        """Rows expected FOR THE SPLITS THIS SET IS SCREENED AGAINST.

        Not the whole repo: `rows` is counted after the split filter, so a
        whole-repo expectation makes a correct, complete download read short on
        every run - and the SHORT line the CLI prints below is what an
        operator reads to decide whether a config or a shard is missing.
        """
        counts = [part.rows for part in self.parts]
        return sum(counts) if counts and None not in counts else None

    @property
    def source(self):
        return HF_SOURCES[self.key]

    @property
    def repo_id(self) -> str:
        return self.source.repo_id

    @property
    def license(self) -> str:
        return self.source.license

    @property
    def source_id(self) -> str:
        return self.source.source_id

    @property
    def url(self) -> str:
        return self.source.url


# The three eval corpora the charter is judged against. Their repo ids live in
# acquire.HF_SOURCES and were checked against the Hub on 2026-08-14 - one of
# the three was wrong at that check (opennyaiorg/aibe -> aibe_dataset), which
# is exactly what a wrong id does here: `not_acquired`, a REFUSAL with the id
# printed, never a quiet skip.
#
# THE ROW COUNTS ARE PER CONFIG AND SPLIT, and that is the whole point of the
# shape: `rows` is counted AFTER the split filter, so an expectation that
# describes the whole repo makes a correct, complete download print a
# shortfall on every run - which is the SHORT line the CLI prints per set.
EVAL_SETS = {
    "bbl": EvalSet(
        key="bbl",
        why="the project's headline forgetting guard (24,365 questions)",
        # TWO CONFIGS, both split `test`, and the expectation encodes both: an
        # English-only download reads 17,047 against 24,365 and says so loudly,
        # which is precisely the "one config landed" failure a whole-repo
        # number cannot express. There is no train split to exclude.
        include_splits=("test",),
        parts=(
            EvalPart("english", "test", 17_047),
            EvalPart("hindi", "test", 7_318),
        ),
        selection_note="both configs, split test - the whole set is the eval surface",
    ),
    "iltur": EvalSet(
        key="iltur",
        why="8 Indian legal tasks; CJPE shares appellate pools with PredEx",
        # THE EVAL SURFACE IS THE TEST-TYPE SPLITS OF EVERY CONFIG, and the
        # arithmetic is in the module docstring under WHAT IL-TUR IS SCREENED
        # AGAINST. Short version: the repo is ~488,000 rows across 8 configs
        # (bail alone is 353,698), the index costs ~220 bytes per distinct
        # gram and an item of L tokens contributes about L of them, so the
        # whole repo is tens of gigabytes of index - it does not fit, and a
        # screen that OOMs on the operator's machine screens nothing.
        #
        # Matched on the SPLIT NAME, never on a config name: the config names
        # were not verified and a guessed one would silently exclude a config.
        # `train_all` and `fold_N` are training material; `test_specific` and
        # `expert` are what the tasks are scored on.
        include_splits=("test", "expert"),
        exclude_splits=("train", "fold", "dev", "val", "validation"),
        # No verified counts: the datasets-server was read for the repo total
        # and for `bail`, not per split. The floor and the shortfall line are
        # therefore OFF for this set and say so, rather than being denominated
        # against a number nobody has read. First-run item.
        selection_note=(
            "test-type splits of every config (test/expert); train_all, fold_N, dev and "
            "val are EXCLUDED - see the module docstring for the memory arithmetic"
        ),
    ),
    "aibe": EvalSet(
        key="aibe",
        why="the bar-exam MCQ set Aalap measured against",
        # ONE SPLIT, named `train`, and it IS the eval set - so there is no
        # filter at all here and the complete shape must read `ok` with a
        # shortfall of zero. Filtering for a `test` split would empty it.
        parts=(EvalPart("default", "train", 1_157),),
        selection_note="single split `train`, which is the whole set",
    ),
}

EVAL_OK = "ok"
EVAL_NOT_ACQUIRED = "not_acquired"
EVAL_NO_FILES = "no_files"
EVAL_NO_TEXT_COLUMN = "no_text_column"
EVAL_EMPTY = "empty"
EVAL_UNREADABLE = "unreadable"
# The files are there and readable in principle, but the library that reads
# them is not installed. A different remedy from every other rung, and the one
# EVERY operator hits on the first real run: HF snapshots are usually parquet.
EVAL_NO_READER = "no_reader"
# Loaded, and NOTHING in it can match anything. An eval set the screen
# compared against nothing is not a screened eval set, whatever the row count
# says, and "cannot reach an eval set must not report clean" is not satisfied
# by reaching it in name only.
EVAL_UNMATCHABLE = "unmatchable"
# Loaded, and far smaller than the set is documented to be - a fragment, a
# single shard, or a repo id that resolved to something else.
EVAL_TOO_FEW = "too_few_rows"
# The set has a SPLIT FILTER and no object on disk names a screened split.
#
# This used to be a silent fallback to "read everything", justified as
# "over-screening is safe". It is not safe here and the arithmetic says why:
# IL-TUR is ~488,000 rows across 8 heterogeneous configs and the eval surface
# chosen for it is a few thousand of them, so reading everything is ~20 GB of
# index on a machine that does not have it - the OOM this module's own
# docstring rejects, arrived at from the other side. Worse, the fallback
# OVERWROTE `record["excluded"] = []`, so the manifest read "selected all,
# excluded nothing" while ~488,000 training rows became the eval surface, and
# its designated tell - `surplus` - is structurally 0 for the one set with
# fallback risk, because IL-TUR has no verified row count.
#
# So a filtered set whose filter selects nothing is a REFUSAL, the same regime
# as every other "loaded but useless" state, and the remedy names the layout it
# saw against the layout it expected. A set with NO filter (aibe, whose single
# `train` split IS the set) never reaches this: reading everything is its
# normal path, not a fallback.
EVAL_NO_SPLIT = "no_screened_split"

# The floor is a HUNDREDTH of the expected count, not a half. The counts are
# verified now (EVAL_COUNTS_VERIFIED_AT), but a shard that has not finished
# downloading is a legitimate mid-pull state and the FILE LAYOUT that decides
# which of them are read is not verified at all - so a floor that refuses a
# CORRECT download would push the operator
# straight to `--allow-missing-eval bbl`, which is the exact outcome this
# module exists to prevent - so the refusal is sized to catch a set that is
# obviously not the set, and the SHORTFALL against expect_rows is printed and
# recorded at every level below it, where an operator can act on it without
# being tempted to waive anything.
EVAL_MIN_SHARE = 0.01

# Every candidate column an eval row might carry its question in, tried in
# order. Same discipline as select.py's schema handling: the real column names
# cannot be checked offline, so each read is a short ordered list and the
# WINNER is reported - "no candidate matched" and "the fallback matched" must
# never be the same number on the one instrument built to tell them apart.
_EVAL_TEXT_FIELDS = (
    "question", "query", "prompt", "input", "text", "instruction",
    "question_text", "stem", "context", "passage", "document",
)
# Read after the question and appended to it: the options and the reference
# answer are part of the item a row could have memorised.
_EVAL_EXTRA_FIELDS = ("options", "choices", "answer", "correct_answer", "target", "output")

_READABLE_SUFFIXES = (".jsonl", ".ndjson", ".json", ".csv", ".tsv", ".parquet")


@dataclass(frozen=True)
class EvalItem:
    set_key: str
    item_id: str
    text: str
    identifiers: frozenset[str]


@dataclass
class EvalCorpus:
    spec: EvalSet
    status: str
    items: list[EvalItem] = field(default_factory=list)
    files: int = 0
    rows: int = 0
    text_field: str | None = None
    detail: str = ""
    allowed_missing: bool = False
    # Items nothing in this module can match (under SHORT_MIN_TOKENS). Counted
    # HERE and not only in the index, because a set that is entirely
    # unmatchable has to be refused before anything is written.
    unmatchable: int = 0
    # Which objects were read and which were left out, with the reason. An
    # undocumented subset narrows the guarantee silently, so the selection is
    # recorded rather than implied by a row count.
    selection: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == EVAL_OK

    @property
    def shortfall(self) -> int:
        """How many rows short of the expectation FOR THE SPLITS SELECTED.

        `rows` counts what the split filter kept, so the expectation it is
        compared against has to describe the same population - otherwise a
        correct, complete download reads short on every run and the banner
        that says so is noise.
        """
        expect = self.spec.expect_rows
        return max(0, expect - self.rows) if expect else 0

    @property
    def surplus(self) -> int:
        """Rows ABOVE the expectation for the splits this set is screened against.

        The tell that the filter selected MORE than the eval surface: a shard
        counted twice, a config that is not in `parts`, or an object whose name
        happens to carry a screened split. It used to be described as the
        fallback's tell, which it never was and now could not be - the fallback
        is a refusal (EVAL_NO_SPLIT) and IL-TUR, the only set with fallback
        risk, has no verified count for this to be denominated against at all.
        It fires for BBL and aibe, whose counts are verified, and it is OFF and
        says so where they are not.
        """
        expect = self.spec.expect_rows
        return max(0, self.rows - expect) if expect else 0


def read_rows(path: Path) -> Iterator[dict]:
    """Rows out of one snapshot file, dispatched on suffix.

    jsonl/json/csv/tsv are read here in pure Python so the load path is
    exercised offline. Parquet - what HF snapshots usually ship - is behind a
    lazy pyarrow import, and it HAS executed: the test round-trips a real
    pyarrow file where the [build] extra is present and keeps the ImportError
    path (EVAL_NO_READER, a refusal with `pip install -e .[build]` as the
    remedy) where it is not.
    """
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        yield record
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else payload.get("data", [])
        yield from (r for r in records if isinstance(r, dict))
    elif suffix in (".csv", ".tsv"):
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ",")
    elif suffix == ".parquet":
        import pyarrow.parquet as pq

        yield from pq.read_table(path).to_pylist()


def eval_item_texts(record: dict) -> tuple[list[str], str | None]:
    """The screenable texts of one eval row, and the column its question came from.

    The question and the options/answer are SEPARATE items, never one
    concatenated blob, because containment divides by the eval item's own
    length: a 29-token question with 20 tokens of options appended scores
    17/37 = 0.46 for a row that quotes the whole question verbatim - under the
    threshold, and a leak the screen would have reported clean. Measured on
    the fixture in test_a_question_is_screened_separately_from_its_options
    (0.89 concatenated vs 1.0 apart, and worse for shorter questions).

    Sub-floor leftovers (an answer key of "b") are not returned: a 1-token
    string is not evidence of leakage, and treating it as an unscreenable
    item would bury the count of questions that genuinely are.
    """
    question, winner = "", None
    for name in _EVAL_TEXT_FIELDS:
        value = record.get(name)
        if value:
            question, winner = str(value), name
            break
    if winner is None:
        return [], None
    extras = []
    for name in _EVAL_EXTRA_FIELDS:
        value = record.get(name)
        if value:
            extras.append(" ".join(str(v) for v in value) if isinstance(value, list) else str(value))
    texts = [question]
    rest = "\n".join(extras)
    if len(tokens(rest)) >= SHORT_MIN_TOKENS:
        texts.append(rest)
    return texts, winner


_NAME_PART = re.compile(r"[^a-z0-9]+")


def _name_parts(key: str) -> set[str]:
    """The alphanumeric runs of an object key - `data/test-00000.parquet` ->
    {data, test, 00000, parquet}, and `latest.parquet` -> {latest, parquet}."""
    return set(_NAME_PART.split(key.lower()))


def select_split_files(spec: EvalSet, paths: Sequence[tuple[str, Path]]):
    """The files of this set that are its eval surface, and the record of it.

    A NAME COMPONENT, not a substring: `latest.parquet` contains "test", and a
    bare substring test would select it and silently screen against the wrong
    file.

    EXCLUDE BEATS INCLUDE, so `train_test-00000.parquet` is excluded: a name
    carrying both is ambiguous and the safe reading of an ambiguous name is the
    one that does not put a training split into the eval surface.

    A filter that selects nothing is NOT a fallback to reading everything - see
    EVAL_NO_SPLIT for the arithmetic. `no_screened_split` says so and the
    exclusion record is left intact, because "selected all, excluded nothing"
    is exactly the sentence the manifest must never be able to write about a
    layout nobody recognised.
    """
    include, exclude = set(spec.include_splits), set(spec.exclude_splits)
    record = {
        "include_splits": list(spec.include_splits),
        "exclude_splits": list(spec.exclude_splits),
        "note": spec.selection_note,
        "selected": [],
        "excluded": [],
        "no_screened_split": False,
    }
    if not include and not exclude:
        # No filter at all: the whole repo IS the eval surface (aibe). Reading
        # everything here is the normal path and not a fallback from anything.
        record["selected"] = [key for key, _ in paths]
        return list(paths), record
    selected = []
    for key, path in paths:
        parts = _name_parts(key)
        if exclude & parts:
            record["excluded"].append({"key": key, "why": sorted(exclude & parts)[0]})
        elif include and not (include & parts):
            record["excluded"].append({"key": key, "why": "no screened split in the name"})
        else:
            selected.append((key, path))
            record["selected"].append(key)
    if not selected:
        record["no_screened_split"] = True
        return [], record
    return selected, record


def eval_corpus(store, spec: EvalSet, *, reader=read_rows) -> EvalCorpus:
    """Load one eval set from what acquire.py landed, or say exactly why not.

    Read through the store's artifact index rather than a directory walk, for
    select.py's reason: an item can only come out of a file the store says is
    complete. Every way this comes back short is a DIFFERENT status, because
    they send the operator to different places - `not_acquired` is an acquire
    run (or an access grant), `no_text_column` is a column name in a file
    already on disk.
    """
    index = store.artifact_index(spec.source_id)
    if not index:
        return EvalCorpus(spec, EVAL_NOT_ACQUIRED, detail="nothing indexed under this source id")
    paths = [
        (key, Path(row["local_path"]))
        for key, row in sorted(index.items())
        if Path(key).suffix.lower() in _READABLE_SUFFIXES
    ]
    if not paths:
        return EvalCorpus(
            spec, EVAL_NO_FILES, detail=f"{len(index)} objects, none of {_READABLE_SUFFIXES}"
        )
    seen = sorted({part for key, _ in paths for part in _name_parts(key)})
    paths, selection = select_split_files(spec, paths)
    if selection["no_screened_split"]:
        # BEFORE a single row is read. The banner this replaced printed after
        # eval_corpus had materialised every row and both indexes were built -
        # i.e. after the ~20 GB the docstring says does not fit.
        return EvalCorpus(
            spec, EVAL_NO_SPLIT, files=0, selection=selection,
            detail=(
                f"{len(selection['excluded'])} objects, none naming any of "
                f"{list(spec.include_splits)} as a path component. The names carry "
                f"{seen[:12]}"
            ),
        )

    items: list[EvalItem] = []
    rows = 0
    winners: dict[str, int] = {}
    for key, path in paths:
        try:
            records = list(reader(path))
        except ImportError as exc:
            # NOT "unreadable". The file is fine and the repo id is fine; the
            # reader is missing, and sending an operator to re-download a
            # correct snapshot - or worse, to doubt a repo id that is the one
            # genuinely uncertain thing here - is the wrong instruction. HF
            # snapshots are usually parquet, so this is what the first real
            # run hits, on all three sets at once.
            return EvalCorpus(
                spec, EVAL_NO_READER, files=len(paths), selection=selection,
                detail=f"{key}: {exc}",
            )
        except Exception as exc:  # a corrupt or unreadable snapshot file
            return EvalCorpus(
                spec, EVAL_UNREADABLE, files=len(paths), selection=selection,
                detail=f"{key}: {type(exc).__name__}: {exc}",
            )
        for i, record in enumerate(records):
            rows += 1
            texts, winner = eval_item_texts(record)
            if winner is None or not any(t.strip() for t in texts):
                continue
            winners[winner] = winners.get(winner, 0) + 1
            identifiers = identifiers_from_fields(record) | identifiers_from_text(
                "\n".join(texts)
            )
            for part, text in enumerate(texts):
                items.append(
                    EvalItem(
                        spec.key, f"{key}#{i}" + ("" if part == 0 else f"/{part}"),
                        text, frozenset(identifiers),
                    )
                )
    if rows and not items:
        return EvalCorpus(
            spec, EVAL_NO_TEXT_COLUMN, files=len(paths), rows=rows, selection=selection,
            detail=f"{rows} rows, none carrying any of {_EVAL_TEXT_FIELDS}",
        )
    if not items:
        return EvalCorpus(spec, EVAL_EMPTY, files=len(paths), rows=rows,
                          selection=selection, detail="0 rows")
    corpus = EvalCorpus(
        spec, EVAL_OK, items=items, files=len(paths), rows=rows, selection=selection,
        text_field=max(winners, key=winners.get),
        unmatchable=sum(1 for item in items if not window_for(len(tokens(item.text)))),
    )
    floor = math.ceil((spec.expect_rows or 0) * EVAL_MIN_SHARE)
    # ROWS, not items: a BBL row with options produces two items, so an item
    # floor compares one population against another and moves with a column
    # list rather than with the download. The counts in EvalPart are rows.
    if floor and rows < floor:
        corpus.status = EVAL_TOO_FEW
        corpus.detail = (
            f"{rows} rows, and the splits this set is screened against hold "
            f"{spec.expect_rows} (verified {EVAL_COUNTS_VERIFIED_AT}). "
            f"Under {floor} ({EVAL_MIN_SHARE:.0%}) this is a fragment, a single shard, or "
            f"a different dataset"
        )
    elif corpus.unmatchable == len(items):
        # The one thing that must not be got wrong, one layer in from a
        # missing set: the screen ran against a corpus none of whose items it
        # can match, and the run would otherwise be stamped decontaminated.
        corpus.status = EVAL_UNMATCHABLE
        corpus.detail = (
            f"all {len(items)} items are under {SHORT_MIN_TOKENS} tokens, so NOTHING here "
            f"can match any of them - this set was compared against nothing"
        )
    return corpus


def eval_corpora(store, *, allow_missing: Iterable[str] = (), reader=read_rows,
                 keys: Iterable[str] | None = None) -> dict[str, EvalCorpus]:
    allowed = set(allow_missing)
    out: dict[str, EvalCorpus] = {}
    for key in (sorted(EVAL_SETS) if keys is None else list(keys)):
        corpus = eval_corpus(store, EVAL_SETS[key], reader=reader)
        corpus.allowed_missing = not corpus.ok and key in allowed
        out[key] = corpus
    return out


def _acquire_remedy(key: str, spec: EvalSet) -> str:
    return (
        f"python -m tuned.data.acquire --kind hf --hf-source {key}\n"
        f"               (all three eval sets report gated=auto, so accept the terms\n"
        f"                at {spec.url} and set HF_TOKEN;\n"
        f"                the repo id itself was verified against the Hub 2026-08-14)"
    )


def _remedy(key: str, corpus: EvalCorpus) -> str:
    """What to DO about this status. They send the operator to different
    places, which is the whole reason each way of coming back short is its own
    status rather than one warning."""
    spec = corpus.spec
    if corpus.status == EVAL_NO_READER:
        return (
            "pip install -e .[build]\n"
            "               (the snapshot is fine and so is the repo id - the reader for\n"
            "                these files is not installed. HF snapshots are usually\n"
            "                parquet, so this is the first-run case, not a corrupt file)"
        )
    if corpus.status == EVAL_UNREADABLE:
        return (
            f"the file itself is corrupt - delete it and re-run\n"
            f"               python -m tuned.data.acquire --kind hf --hf-source {key}"
        )
    if corpus.status == EVAL_NO_TEXT_COLUMN:
        return (
            "add this set's real question column to _EVAL_TEXT_FIELDS in this module\n"
            "               (the file is on disk and readable; nothing in it was\n"
            "                recognised as a question)"
        )
    if corpus.status == EVAL_UNMATCHABLE:
        return (
            "read the column that was chosen - an item under 5 tokens is an answer\n"
            "               key or a label, not a question, so _EVAL_TEXT_FIELDS is\n"
            "               probably matching the wrong column"
        )
    if corpus.status == EVAL_NO_SPLIT:
        return (
            f"check the layout under {spec.url} against this set's split filter\n"
            f"               (expected a path component from {list(spec.include_splits)};\n"
            f"                the layout it actually saw is printed above.\n"
            f"                Fix include_splits in EVAL_SETS for the real layout - this is\n"
            f"                NOT read-everything-instead: {spec.key} is ~488,000 rows across\n"
            f"                8 configs and the whole repo is tens of GB of index, so an\n"
            f"                unrecognised layout means the eval surface is unknown, not wide)"
        )
    if corpus.status == EVAL_TOO_FEW:
        return (
            f"check the configs and the shard count under {spec.url}\n"
            f"               ({spec.expect_rows} is the sum of "
            f"{[(p.config, p.split, p.rows) for p in spec.parts]}, read off the\n"
            f"                datasets-server on {EVAL_COUNTS_VERIFIED_AT}; if a config is\n"
            f"                simply missing from the snapshot, acquire it rather than\n"
            f"                editing EvalSet.parts. Do NOT waive the set to get past this:\n"
            f"                a waived BBL is the failure this module exists to prevent)"
        )
    return _acquire_remedy(key, spec)


def refusals(corpora: dict[str, EvalCorpus]) -> list[str]:
    """One actionable refusal per eval set that could not be read and was not
    explicitly waived. Empty means the pass may run."""
    out = []
    for key in sorted(corpora):
        corpus = corpora[key]
        if corpus.ok or corpus.allowed_missing:
            continue
        spec = corpus.spec
        out.append(
            f"eval set {key!r} ({spec.repo_id}) is {corpus.status}: {corpus.detail}.\n"
            f"    it guards: {spec.why}\n"
            f"    fix it:    {_remedy(key, corpus)}\n"
            f"    override:  --allow-missing-eval {key}  (recorded in the manifest)"
        )
    return out


REFUSAL_HEADER = (
    "REFUSING TO DECONTAMINATE: an eval set this dataset is measured against cannot be read,\n"
    "and a pass that cannot reach an eval set must not report clean. A leak into training\n"
    "inflates the headline number in the flattering direction and nothing downstream can\n"
    "detect it."
)


# --------------------------------------------------------------------------
# The index: exact candidate generation over the eval side.
# --------------------------------------------------------------------------

class EvalIndex:
    """Inverted index over eval-item n-grams, ONE TABLE PER WINDOW.

    EXACT: containment > 0 requires at least one shared gram AT THAT ITEM'S
    WINDOW, so every pair the arithmetic could drop is generated here. Nothing
    approximate (LSH, embeddings) stands between an eval item and its
    candidate row - the error direction of an approximation is missed pairs,
    which is the one direction this module may not have.

    The window is a property of the ITEM (see window_for), so the index is
    keyed by it and a row is gramed once per window the index actually holds.
    A corpus of BBL-length questions occupies a handful of windows; the walk
    costs one pass over the row per window and nothing per item.
    """

    def __init__(self, items: Iterable[EvalItem], *, n: int = NGRAM,
                 short_min: int = SHORT_MIN_TOKENS):
        self.n = n
        self.short_min = short_min
        self.items: list[EvalItem] = []
        # COUNTS, not the sets. The posting walk already yields the exact
        # intersection size, so the only thing the arithmetic still needs from
        # an eval item is how many grams it had - and holding ~4,000 grams per
        # IL-TUR judgment as a Python set costs ~60 bytes each, which is the
        # difference between an index that fits and one that does not.
        self.gram_counts: list[int] = []
        self.windows: list[int] = []
        self.token_counts: list[int] = []
        # window -> {gram -> item indexes}
        self.by_gram: dict[int, dict[int, list[int]]] = {}
        self.unmatchable: list[str] = []
        self.by_identifier: dict[str, list[int]] = {}
        for ix, item in enumerate(items):
            toks = tokens(item.text)
            self.items.append(item)
            self.token_counts.append(len(toks))
            window = window_for(len(toks), n=n, short_min=short_min)
            self.windows.append(window)
            grams = gram_hashes(toks, window) if window else frozenset()
            self.gram_counts.append(len(grams))
            if window:
                table = self.by_gram.setdefault(window, {})
                for gram in grams:
                    table.setdefault(gram, []).append(ix)
            else:
                # Too short for any rule here. NOT silently ignored: it is
                # a hole in the screen and the manifest counts it, per set.
                self.unmatchable.append(item.item_id)
            for identifier in item.identifiers:
                self.by_identifier.setdefault(identifier, []).append(ix)

    def __len__(self) -> int:
        return len(self.items)

    def query(self, toks: Sequence[str]) -> dict[int, frozenset[int]]:
        """The row's grams at every window this index holds.

        Built once per row and handed to both `candidates` and the Jaccard
        diagnostic, so a row is never gramed twice at the same window.
        """
        return {window: gram_hashes(toks, window) for window in self.by_gram}

    def candidates(self, query: dict[int, frozenset[int]]) -> dict[int, int]:
        """Every eval item sharing >=1 gram with the row -> how many it shares.

        The count IS the intersection size (grams are a set on both sides), so
        containment and Jaccard both fall out of the posting walk and no
        candidate is ever re-intersected. That matters at the scale this runs
        at: one boilerplate gram shared with 5,000 eval items would otherwise
        cost 5,000 set intersections per row.
        """
        found: dict[int, int] = {}
        for window, grams in query.items():
            table = self.by_gram.get(window)
            if not table:
                continue
            for gram in grams:
                for ix in table.get(gram, ()):
                    found[ix] = found.get(ix, 0) + 1
        return found

    def level_of(self, ix: int) -> str | None:
        return level_for(self.token_counts[ix], n=self.n, short_min=self.short_min)

    def length_report(self) -> dict[str, dict]:
        """PER SET: how many items each level screens, and the token spread.

        A pooled count cannot tell an operator WHICH set is blind, and the
        window calibration above is an argument until this table is read
        against a real download - which is why it is in the manifest and on
        the manifest rather than in a comment. `unmatchable` is the
        hole: items nothing here can match.
        """
        out: dict[str, dict] = {}
        for ix, item in enumerate(self.items):
            entry = out.setdefault(
                item.set_key,
                {LEVEL_TEXT: 0, LEVEL_NARROW: 0, LEVEL_SHORT: 0, "unmatchable": 0, "_t": []},
            )
            entry[self.level_of(ix) or "unmatchable"] += 1
            entry["_t"].append(self.token_counts[ix])
        for entry in out.values():
            spread = sorted(entry.pop("_t"))
            entry["min_tokens"] = spread[0]
            entry["median_tokens"] = spread[len(spread) // 2]
            entry["max_tokens"] = spread[-1]
        return dict(sorted(out.items()))

    def identifier_candidates(self, identifiers: Iterable[str]) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        for identifier in sorted(identifiers):
            for ix in self.by_identifier.get(identifier, ()):
                out.append((identifier, ix))
        return out


# --------------------------------------------------------------------------
# The decision.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Hit:
    level: str
    eval_set: str
    item_id: str
    detail: dict


def hits_for(item: Item, index: EvalIndex, *, threshold: float = CONTAINMENT) -> list[Hit]:
    """Every level that fires on this row, in LEVELS order.

    All levels are evaluated rather than short-circuited: the first is the
    drop reason, and the rest are what says whether a level is carrying cases
    of its own on real data or is dead weight the others subsume.
    """
    found: list[Hit] = []
    toks = tokens(item.text)
    query = index.query(toks)

    # ONE hit for the level, whatever it matched. A row that shares three
    # identifiers with four eval items is one contaminated row, and appending a
    # Hit per (identifier, item) pair would inflate `by_level["case_id"]` -
    # the instrument that says whether this branch is carrying cases of its
    # own on real data - by the size of the citation graph.
    #
    # EVERY MATCHING IDENTIFIER IS BLAMED, though, and that is a different
    # question from which one the drop is filed under. `top_identifiers` exists
    # to answer "which citation is costing rows", asked in order to decide
    # whether to pass --no-case-id-from-text; that lever removes a whole
    # channel, so the operator has to see every identifier that would have
    # caused this drop, including the redundant ones. Blaming only the
    # sorted-first hid exactly that redundancy - and since "cit:" sorts before
    # "cnr:", a row matched on both its own CNR and a landmark citation always
    # named the citation.
    matching = index.identifier_candidates(item.identifiers)
    if matching:
        identifier, ix = matching[0]
        hit = index.items[ix]
        found.append(
            Hit(
                LEVEL_CASE_ID, hit.set_key, hit.item_id,
                {"identifier": identifier,
                 "identifiers": sorted({name for name, _ in matching})},
            )
        )

    # The best hit PER LEVEL, not one overall: the levels are a union and the
    # counts are the instrument that says whether each branch is carrying
    # cases of its own on real data. One pooled "best" would hide a level that
    # never fires behind one that always does.
    best: dict[str, tuple[float, int, int]] = {}
    for ix, shared in sorted(index.candidates(query).items()):
        share = containment_from(shared, index.gram_counts[ix])
        if share < threshold:
            continue
        level = index.level_of(ix)
        if level is not None and (level not in best or share > best[level][0]):
            best[level] = (share, ix, shared)
    for level in (LEVEL_TEXT, LEVEL_NARROW, LEVEL_SHORT):
        if level not in best:
            continue
        share, ix, shared = best[level]
        hit = index.items[ix]
        window = index.windows[ix]
        found.append(
            Hit(
                level, hit.set_key, hit.item_id,
                # Jaccard rides along as a diagnostic: it is the number the
                # plan asked for as a rule, and recording it is how the first
                # real run shows how far below 0.8 a genuine leak scores. The
                # window and the item's length ride along because the level a
                # drop is filed under is a fact about the ITEM's length, and
                # the first run has to be able to read the two together.
                {"containment": round(share, 4),
                 "jaccard": round(
                     jaccard_from(shared, index.gram_counts[ix], len(query.get(window, ()))), 4
                 ),
                 "window": window,
                 "eval_tokens": index.token_counts[ix]},
            )
        )

    return sorted(found, key=lambda h: NGRAM_LEVELS.index(h.level))


def decontaminate_items(
    items: Iterable[Item],
    index: EvalIndex,
    *,
    threshold: float = CONTAINMENT,
    semantic=None,
) -> tuple[list[Item], list[dict], dict]:
    """Split candidate rows into (kept, drop records, stats).

    Order is the input's and is never re-sorted: two runs over the same input
    produce the same output bytes, which is what stops the train/test boundary
    downstream from moving between runs.
    """
    kept: list[Item] = []
    drops: list[dict] = []
    stats = {
        "total": 0,
        "kept": 0,
        "dropped": 0,
        "empty_text": 0,
        "with_identifier": 0,
        "by_reason": {},
        "by_level": dict.fromkeys(LEVELS, 0),
        "by_eval_set": {},
        "identifier_drops": {},
    }
    for item in items:
        stats["total"] += 1
        if item.identifiers:
            stats["with_identifier"] += 1
        found = hits_for(item, index, threshold=threshold)
        if not tokens(item.text):
            # A row that tokenises to NOTHING can match nothing, so it would
            # leave this pass looking screened while never having been
            # compared to anything. Counted, and printed by the CLI.
            stats["empty_text"] += 1
        if semantic is not None:
            found = found + list(semantic(item))
        if not found:
            kept.append(item)
            continue
        first = found[0]
        reason = f"{first.level}:{first.eval_set}"
        stats["dropped"] += 1
        stats["by_reason"][reason] = stats["by_reason"].get(reason, 0) + 1
        for hit in found:
            stats["by_level"][hit.level] = stats["by_level"].get(hit.level, 0) + 1
            stats["by_eval_set"][hit.eval_set] = stats["by_eval_set"].get(hit.eval_set, 0) + 1
            if hit.level == LEVEL_CASE_ID:
                # Every identifier that matched, not just the one the drop is
                # filed under: this is a count of ROW-MATCHES per identifier
                # over CASE-IDENTIFIER hits, so it sums to at least
                # `by_level["case_id"]` - NOT to `dropped`, since a row dropped
                # on text or on the semantic layer contributes nothing to it.
                # See hits_for for why the lever this feeds needs all of them.
                for key in hit.detail["identifiers"]:
                    stats["identifier_drops"][key] = stats["identifier_drops"].get(key, 0) + 1
        drops.append(
            {
                "key": item.key,
                "origin": item.origin,
                "reason": reason,
                "form": item.form,
                "hits": [
                    {"level": h.level, "eval_set": h.eval_set, "item_id": h.item_id, **h.detail}
                    for h in found
                ],
            }
        )
    stats["kept"] = len(kept)
    return kept, drops, stats


# --------------------------------------------------------------------------
# The semantic layer (optional, and recorded either way).
# --------------------------------------------------------------------------

SEMANTIC_UNAVAILABLE = "semhash-not-installed"
# The seam answered in a shape this code cannot read, or answered wrongly on a
# control whose answer is not in doubt. NOT "ran": see semantic_control.
SEMANTIC_UNUSABLE = "semhash-control-failed"
# Installed, but the embedding index could not be BUILT - semhash fetches a
# model on first use, so this is what an air-gapped machine with the extra
# installed reads. Its own rung because its remedy is the opposite of
# `control-failed`'s: get network or pre-warm the HF cache, do not go looking
# for API drift. The ledger used to say "control-failed means the API drifted",
# which was false on exactly the machine this module was written on.
SEMANTIC_NO_MODEL = "semhash-model-unavailable"
SEMANTIC_NO_ITEMS = "no-eval-items-to-compare"
# There were eval items, and not one of them is in a script whose control
# passed - a Hindi-only partial pull of BBL is 7,318 items and clears every
# floor above this line. The missing rung: `ran` has to mean "this layer could
# have found a reworded eval question", and a layer that built no index at all
# could not have. Distinct from `no-eval-items-to-compare` (nothing downloaded)
# and from `semhash-control-failed` (the seam has no power anywhere), because
# the remedy is a different one again: a multilingual embedding model.
SEMANTIC_NO_SCREENABLE_ITEMS = "no-eval-items-in-a-screenable-script"
SEMANTIC_RAN = "ran"

# THE OPERATING POINT, chosen by measurement against the installed library
# (cache-only) and not by taking semhash's default. Re-measured at the probe
# geometry below, with every leak placed at the WORST alignment the geometry
# admits and every sibling at the best - i.e. each column is the pessimistic
# reading of both error directions at once. The fixtures are in the repo and
# the table is re-runnable: test_the_semantic_threshold_table_reproduces.
#
#   threshold           0.6   0.65   0.7   0.75   0.8   0.85   0.9   0.95
#   verbatim leaks      5/5    5/5   5/5    5/5   5/5    5/5   4/5    0/5
#   reworded leaks      5/5    5/5   5/5    5/5   5/5    4/5   2/5    0/5
#   siblings dropped    3/5    2/5   1/5    0/5   0/5    0/5   0/5    0/5
#   SAME-STEM siblings  5/5    5/5   5/5    3/5   2/5    0/5   0/5    0/5
#   clean rows dropped  0/4    0/4   0/4    0/4   0/4    0/4   0/4    0/4
#
# "leaks" are an eval question quoted verbatim or lightly reworded inside a
# 300-word row. "Siblings" are a DIFFERENT question about the same section or
# article - the statute-quotation exception one band over. "Same-stem
# siblings" are the harder and more honest version of that class: the SAME
# question stem carrying a different offence ("...criminal misappropriation of
# property by a public servant..." against "...criminal breach of trust by a
# public servant..."). They are the false positive that costs real yield,
# because this corpus is about the same statutes the eval sets are.
#
# THERE IS NO CLEAN SEPARATION, and pretending otherwise is what the previous
# table did. Measured on these fixtures the reworded leaks span 0.819-0.915
# and the same-stem siblings span 0.714-0.847, so the two distributions
# OVERLAP on [0.819, 0.847]: any threshold in that band both misses a leak and
# drops a sibling. 0.8 sits below the overlap and therefore buys every leak at
# a MEASURED, NAMED price - 2 of 5 same-stem siblings. Their exact containment
# against the eval item is 0.06-0.38 across the five, well under the 0.5 the
# exact stack requires, so these are rows it rightly keeps. That price is the
# right one
# under this module's asymmetry (a false negative is invisible, permanent and
# flatters the headline number; a false positive costs one row of ~18,000),
# and the per-Hit provenance recorded below is what makes the real rate
# readable on run one instead of arguable now.
SEMANTIC_THRESHOLD = 0.8
# THE PROBE GEOMETRY. The row is probed in WINDOWS as well as whole: see
# SemanticFilter for the measurement showing a whole-row embedding cannot see
# a verbatim eval question inside a 288-word row at ANY threshold that does
# not also drop clean rows.
#
# THE SIZE IS SET BY AN ALIGNMENT GUARANTEE, not by eval-question length. A
# span of L words placed at offset d is held by the window starting at the
# largest multiple of the stride at or below d, which holds min(L, size - (d
# mod stride)) of it - so the WORST placement leaves size - stride + 1 words
# in the best window. At the previous 20/10 that was 11 of a 20-word question
# and the gap was live. Measured on the repo's own five reworded leaks, each
# placed at the worst offset ITS geometry admits:
#
#     20/10   0.890 0.825 0.890 0.866 0.735   -> 4 of 5 caught at 0.8
#     30/10   0.897 0.892 0.909 0.915 0.819   -> 5 of 5
#
# The fifth is a genuine rewording of a BhashaBench-shaped question and 20/10
# lost it by 0.065. Re-run by test_the_probe_geometry_closes_the_alignment_gap.
#
# So the size is chosen to make the guarantee hold instead:
#
#     SEMANTIC_PROBE_WORDS - SEMANTIC_PROBE_STRIDE + 1  >=  a BBL question
#
# 30 - 10 + 1 = 21 words, and BBL's MCQ stems are ~20. EVERY span of 21 words
# or fewer now sits ENTIRELY inside at least one probe, at every offset, and
# the alignment gap is closed rather than documented.
#
# It is also CHEAPER, which is the part that is easy to disbelieve: the number
# of windows is about (row - size)/stride, so widening the window at a fixed
# stride REMOVES windows. Measured on a 300-word row: 29 probes at 30/10
# against 30 at 20/10. Narrowing the stride is the expensive lever (58 probes
# at 20/5) and it does NOT close the gap - only widening the window does.
SEMANTIC_PROBE_WORDS = 30
SEMANTIC_PROBE_STRIDE = 10
# The longest span of a row guaranteed to lie WHOLLY inside one probe window,
# at every offset. Derived from the two constants above and pinned against
# both a literal and the behaviour of probe_texts, because a geometry that is
# only pinned by an expression derived from itself is invariant to the change
# it exists to catch.
SEMANTIC_COVERED_SPAN = SEMANTIC_PROBE_WORDS - SEMANTIC_PROBE_STRIDE + 1


def probe_texts(text: str, *, size: int = SEMANTIC_PROBE_WORDS,
                stride: int = SEMANTIC_PROBE_STRIDE) -> list[str]:
    """The row, plus every `size`-word window of it at `stride`.

    THE WHOLE ROW IS KEPT because the eval side is not all short: an IL-TUR
    judgment item is hundreds of words and only the whole row is comparable to
    it. The windows are for the other end, which is most of the eval corpus.

    At the shipped size and stride every span of SEMANTIC_COVERED_SPAN words or
    fewer lies WHOLLY inside at least one returned window, whatever its offset
    - including a span at the very end of the row, which is what the tail
    anchor below is for. That guarantee is the geometry's whole point and it is
    asserted from the returned windows themselves, never from the loop index.
    """
    words = (text or "").split()
    if not words:
        return []
    if len(words) <= size:
        return [text]
    starts = list(range(0, len(words) - size + 1, stride))
    # The tail, anchored at the end. Without it the last (len - size) % stride
    # words of every row sit in no window at all and are screened only by the
    # whole-row probe, which is the comparison this design has just established
    # cannot see anything.
    if starts[-1] != len(words) - size:
        starts.append(len(words) - size)
    return [text] + [" ".join(words[i : i + size]) for i in starts]


def probe_windows(row_words: int, *, size: int = SEMANTIC_PROBE_WORDS,
                  stride: int = SEMANTIC_PROBE_STRIDE) -> list[tuple[int, int]]:
    """The [start, end) word ranges probe_texts ACTUALLY returns for a row of
    `row_words` words - read back off the probes, not re-derived from `stride`.

    Everything that reasons about the geometry (the control's placement, the
    coverage assertions) goes through here, so an instrument can never agree
    with the code by sharing its arithmetic. A window whose start is computed
    twice is a window that is pinned zero times.
    """
    marked = " ".join(f"w{i}" for i in range(row_words))
    out = []
    for probe in probe_texts(marked, size=size, stride=stride)[1:]:
        indexes = [int(word[1:]) for word in probe.split()]
        out.append((indexes[0], indexes[-1] + 1))
    return out


def worst_alignment_offset(target_words: int, row_words: int, *,
                           size: int = SEMANTIC_PROBE_WORDS,
                           stride: int = SEMANTIC_PROBE_STRIDE) -> int:
    """Where to put a `target_words` span so the probe windows hold it WORST.

    THIS IS WHY THE CONTROL IS NOT ALIGNED TO THE STRIDE. Its padding used to
    be `(filler * 2)[: 4 * SEMANTIC_PROBE_STRIDE]`, i.e. derived from the very
    constant it existed to guard: every stride change re-aligned the control's
    paraphrase back onto a window boundary, so the control passed for every
    geometry and was invariant to the mutation it was watching for.

    Reading the placement off the windows instead inverts that. A geometry that
    degrades makes some offset genuinely bad, this finds it, and the control is
    run THERE - so the control fails when the geometry stops delivering, and a
    healthy geometry (where every offset is equally good) is free to put it
    anywhere. Ties go to the LATEST offset, which parks the control in the tail
    region - the part of the row a missing tail anchor drops on the floor.
    """
    windows = probe_windows(row_words, size=size, stride=stride)
    best_offset, best_hold = 0, None
    for offset in range(row_words - target_words + 1):
        hold = max(
            (max(0, min(end, offset + target_words) - max(start, offset))
             for start, end in windows),
            default=0,
        )
        if best_hold is None or hold <= best_hold:
            best_hold, best_offset = hold, offset
    return best_offset


# --------------------------------------------------------------------------
# WHICH SCRIPT A PROBE IS IN, and why this layer has to know.
#
# The embedding model is potion-base-8M, and it has no discriminative power
# outside Latin script. Measured cache-only against the installed library:
#
#     cos(Hindi item, its own REWORDING)      = 0.990   <- must be far apart
#     cos(Hindi item, an unrelated CRICKET report) = 0.962   <- and they are not
#     cos(English item, its own REWORDING)    = 0.955
#     cos(English item, an unrelated SOURDOUGH recipe) = 0.089
#
# 0.028 of separation in Devanagari against 0.866 in Latin, on the four
# strings this module defines as its own control. Every one of those numbers
# is re-run by test_the_control_cosines_are_what_this_module_says_they_are.
#
# End to end that is not a degradation, it is an inversion: an index holding
# ONE Hindi eval question drops 4 of 4 clean Hindi rows at EVERY threshold
# from 0.6 to 0.95, where the same shape in English drops 0 of 4 at all of
# them. A
# 300-word English row carrying as few as TEN quoted Devanagari words - a
# quoted FIR, a line of statute, ordinary in this corpus - drops at 0.8 against
# an index that holds Hindi, and is kept against one that does not. And a
# Telugu row with no ASCII in it at all embeds to the ZERO VECTOR (every
# character is [UNK]; one that happens to carry a section number does not, and
# is then driven entirely by the digits), so a Telugu index flags a verbatim
# leak and an unrelated row identically.
#
# 7,318 of BhashaBench-Legal's 24,365 questions are Hindi, so the index WILL
# hold Devanagari on the first real run.
#
# THE RULE: this layer screens a script only when its own control half passes
# in that script, and says so per script in the manifest when it does not. With
# this model that means Hindi semantic screening is OFF and the manifest
# records an honest, named hole - carried like a waiver rather than like a
# clean bill of health. The exact stack, which is tokeniser-correct in
# Devanagari since round 2, still carries those rows, and it is the guarantee
# that matters. A future multilingual model turns Hindi back on by PASSING THE
# CONTROL, not by anybody asserting that it should be on.
#
# AND THE SECOND HALF OF THE RULE, which the first half is worthless without: a
# probe is SPLIT into the scripts it carries, never routed whole. Routing whole
# left the minority script's words INSIDE the majority script's probe text,
# where they diluted the embedding without ever being screened themselves - so
# the layer lost English leaks to Hindi neighbours while recording that English
# was fully screened. See script_partition for the table.
# --------------------------------------------------------------------------

SCRIPT_LATIN = "latin"
SCRIPT_DEVANAGARI = "devanagari"
# Coarse on purpose - one block per WORD, no morphological analysis. The blocks
# are named rather than lumped into "other" so the manifest can say WHICH script
# went unscreened; an operator reading `other: 4,000 rows` cannot act on it.
_SCRIPT_BLOCKS = (
    (0x0041, 0x024F, SCRIPT_LATIN),      # Latin, Latin-1 Supplement, Extended-A/B
    (0x0370, 0x03FF, "greek"),
    (0x0400, 0x04FF, "cyrillic"),
    (0x0590, 0x05FF, "hebrew"),
    (0x0600, 0x06FF, "arabic"),
    (0x0900, 0x097F, SCRIPT_DEVANAGARI),
    (0xA8E0, 0xA8FF, SCRIPT_DEVANAGARI),  # Devanagari Extended
    (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "odia"),
    (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
    (0x0D80, 0x0DFF, "sinhala"),
    (0x0E00, 0x0E7F, "thai"),
    # Both scheduled Indian languages, and both used to land in `none` - the
    # bucket that says "there was nothing here to screen". Meitei (Manipuri) and
    # Santali are written in these, so a row in either read as letterless.
    (0x1C50, 0x1C7F, "ol_chiki"),
    (0xABC0, 0xABFF, "meitei_mayek"),
    (0x4E00, 0x9FFF, "han"),
)
# A probe with no LETTERS at all - a section-number table, a citation list, an
# emoji. Nothing that short of a letter can carry an eval question, so this is
# benign rather than a hole, and the banner says so.
SCRIPT_NONE = "none"
# Letters in a block this table does not name. NOT the same fact as `none` and
# that distinction is the whole reason for the table: `none` means "there was
# nothing here to screen", `unlisted` means "there was something here and this
# module cannot say what script it is in", which is an unscreened hole.
SCRIPT_UNLISTED = "unlisted"


def script_of(char: str) -> str | None:
    """The script one character belongs to, or None if it is not a LETTER.

    EVIDENCE IS LETTERS (Unicode general category L*), and nothing else. The
    version this replaced counted whatever fell inside a block range, which made
    the routing asymmetric in two directions at once: ASCII digits and ASCII
    punctuation (`_ [ ] | ~ °` all sit inside 0x41-0x24F) counted as Latin
    evidence while their Devanagari counterparts counted as Devanagari, and a
    Devanagari digit or a bare combining mark counted as script evidence where
    an ASCII digit counted as none. A section-number table read as a Latin
    probe; `abcd भारती` read 4-5 for Devanagari because the two matras counted.

    A letter in no listed block is SCRIPT_UNLISTED rather than None: it is
    something to screen that this module cannot name, which is a hole, where a
    digit is not.
    """
    if not unicodedata.category(char).startswith("L"):
        return None
    code = ord(char)
    for low, high, name in _SCRIPT_BLOCKS:
        if low <= code <= high:
            return name
    return SCRIPT_UNLISTED


def dominant_script(text: str) -> str:
    """The script MOST OF this text is written in.

    PLURALITY of LETTERS, not majority, and ties go to the alphabetically first
    script - deterministic, cheap, and coarse. This is what routes a WORD into
    a script partition (see script_partition) and what routes the whole-row
    probe; the window probes are partitioned per script rather than routed
    whole, because routing a mixed window whole is what let a handful of
    Devanagari words hide an English leak inside it.

    SCRIPT_NONE for a text with no letters at all - a section-number table, a
    citation list. That is not a script and it is not a hole.
    """
    counts: dict[str, int] = {}
    for char in text or "":
        name = script_of(char)
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return SCRIPT_NONE
    return min(sorted(counts), key=lambda name: -counts[name])


# The fewest words a script partition must carry before it is embedded at all.
# DERIVED, not chosen: SHORT_MIN_TOKENS is already the length under which this
# module refuses to treat an eval item as screenable, so a partition holding
# fewer tokens than that cannot contain any indexable eval item, in any script.
# Set it higher and a short eval question stops being findable inside a
# code-switched row; set it lower and nothing is gained, because there is
# nothing shorter to find.
SCRIPT_PARTITION_FLOOR = SHORT_MIN_TOKENS

# HOW MUCH OF AN EVAL ITEM A MINORITY-SCRIPT PARTITION HAS TO BE before that
# partition is indexed as a findable eval item in its own right. A SHARE, not a
# token floor, and the two are rules about different populations:
# SHORT_MIN_TOKENS is a rule about ITEMS - the length under which a whole eval
# item is not screenable, and every item it admits appears WHOLE on the exact
# stack. A minority-script partition is a FRAGMENT of an item, matched at
# cosine >= 0.8 with no exact-stack backstop under it at all, and a floor
# cannot tell a five-token fragment that is the question from a five-token
# fragment that is the statute's name.
#
# MEASURED, on the residue a Hindi eval item actually contributes. Every
# minority-script Latin residue of a 25-word Hindi question, as a share of the
# item's script-bearing words:
#
#   BARE CITATIONS (must not be indexed)          share
#     Indian Penal Code 1860 302                  0.107
#     Code of Criminal Procedure 1973             0.138
#     Negotiable Instruments Act 1881             0.107
#     Limitation Act 1963                         0.074
#     Constitution of India                       0.107
#     IPC 302                                     0.038
#     section 138 NI Act                          0.107
#   ENGLISH CLAUSES OF THE QUESTION (must be)     share
#     criminal breach of trust by a public servant  0.242   <- round 4's P08
#     ...under which section...is...punishable      0.359
#     ...maximum term of imprisonment...cheque      0.324
#     ...entitled to maintenance under the code     0.286
#     ...compoundable between complainant/accused   0.286
#
# The two populations separate on [0.138, 0.242) and 0.2 sits inside it with
# margin either way. Re-run by test_the_residue_share_table_reproduces.
#
# WHAT IT BUYS, measured end to end: without it, a Hindi item naming its
# statute in English puts `Indian Penal Code 1860 302` in the LATIN index as an
# eval item, and every Hindi row citing that act meets it at cosine 1.000 with
# exact containment 0.000 - a drop record indistinguishable from a verbatim
# leak, unbounded across BhashaBench's 7,318 Hindi questions.
#
# THE ITEM'S OWN DOMINANT SCRIPT IS ALWAYS INDEXED, whatever its share. The
# rule is about residues, and an item whose scripts are all minorities of each
# other (six scripts at 0.16 apiece) must not fall out of every index at once.
SCRIPT_RESIDUE_MIN_SHARE = 0.2


def script_share(text: str, script: str) -> float:
    """What share of `text`'s SCRIPT-BEARING words are written in `script`.

    Script-neutral words are excluded from both ends. They go into EVERY
    partition by design (see script_partition), so counting them would let a
    citation's `302 1860 (2)` inflate the weight of the very residue they are
    shared by - `Indian Penal Code 1860 302` is three Latin words wearing five,
    and it is the three that say how much of the item this is.

    0.0 for a text with no script-bearing words at all, which is the answer
    that keeps it out of every index rather than dividing by zero.
    """
    words = [w for w in (text or "").split() if dominant_script(w) != SCRIPT_NONE]
    if not words:
        return 0.0
    return sum(dominant_script(w) == script for w in words) / len(words)


def script_partition(text: str, *, floor: int = SCRIPT_PARTITION_FLOOR) -> dict[str, str]:
    """{script: the words of `text` written in it, in order, with the digits}.

    WHY THE PROBE IS SPLIT RATHER THAN ROUTED WHOLE. The embedding is over the
    whole probe text, so cross-script words inside a probe dilute it even when
    the probe routes to the right index. Measured cache-only against the shipped
    model, a reworded eval question sitting in a 30-word window scores 0.948
    against its eval item and 0.893 with two Devanagari words beside it, 0.729
    with three and 0.706 with four - and an ENGLISH eval question quoted
    verbatim inside a Hindi-dominant row scored 0.40-0.59 and was KEPT, five
    times out of five, while the row's Devanagari half was recorded as the only
    unscreened thing about it. The screen said `latin: screened, unscreened
    rows 0` over rows whose English leak it had just missed.

    Splitting the probe collapses that: the same five verbatim leaks score
    1.000 against the Latin index once the Devanagari words are not in the
    probe text, and the same five reworded leaks that a plain layer misses at
    three interleaved Hindi words are all caught.

    SCRIPT-NEUTRAL WORDS GO IN EVERY PARTITION. `302`, `1973`, `(2)` are not
    evidence of any script and they are most of what distinguishes one section
    from another - dropping them from the Latin partition would throw away the
    numbers this module's tokeniser was fixed to keep. The consequence is that
    a MONOLINGUAL text's partition is the text itself, word for word, so the
    threshold grid and the sibling overlap are unchanged by this split: it only
    does anything to a probe that was mixed.

    THE FLOOR IS SHORT_MIN_TOKENS, counted in `tokens` over the partition -
    below it there is no eval item to find. A partition under the floor is
    dropped and is NOT recorded as a hole, for the same reason the exact stack
    does not count a 3-token eval item as an item.
    """
    words = (text or "").split()
    scripts: dict[str, int] = {}
    for word in words:
        name = dominant_script(word)
        if name != SCRIPT_NONE:
            scripts[name] = scripts.get(name, 0) + 1
    out: dict[str, str] = {}
    for script in sorted(scripts):
        part = " ".join(
            word for word in words if dominant_script(word) in (script, SCRIPT_NONE)
        )
        if len(tokens(part)) >= floor:
            out[script] = part
    return out


# The negative half of the control. Nothing in an Indian-law eval set is
# semantically near this, so a seam that flags it flags everything - which is
# the drift direction that would empty the corpus rather than pass it.
SEMANTIC_CONTROL_NEGATIVE = (
    "the sourdough starter doubled overnight so I shaped the loaf and baked it at "
    "two hundred and thirty degrees with steam for the first fifteen minutes"
)
# The positive half. NOT an exact copy: an exact copy is recognised by any
# seam with no power at all, and the shipped control passed at every threshold
# from 0.3 to 0.95 while catching 0 of 2 paraphrases at the one it ran at.
# This is a NEAR-PARAPHRASE - same case, same section, reordered and reworded -
# EMBEDDED IN A ROW, because that is the shape the layer meets in production
# and the shape a whole-row seam fails on.
SEMANTIC_CONTROL_ITEM = (
    "the appellant was convicted under section 302 of the penal code and sentenced to "
    "imprisonment for life by the court of sessions"
)
_SEMANTIC_CONTROL_PARAPHRASE = (
    "the accused was convicted under section 302 of the penal code and sentenced to "
    "life imprisonment by the sessions court"
)
_SEMANTIC_CONTROL_FILLER = (
    "having heard learned counsel for the parties and having perused the material placed "
    "on the record we are of the considered view that the concurrent findings recorded "
    "by the courts below do not call for any interference in exercise of the appellate "
    "jurisdiction vested in this court under the constitution"
).split()

# The Devanagari half. Same three parts, same shape, in the script that is 30%
# of BhashaBench-Legal - and it FAILS against potion-base-8M, which is the
# point of having it. Measured cache-only at every threshold from 0.6 to 0.95:
# the positive is flagged and so is the cricket report, so the half never
# passes and Hindi screening is off and says so. This is not a placeholder for
# a control that will one day be written; it is the instrument that decides.
_SEMANTIC_CONTROL_ITEM_DEVANAGARI = (
    "भारतीय दंड संहिता की किस धारा के अंतर्गत लोक सेवक द्वारा किए गए आपराधिक न्यासभंग के लिए "
    "आजीवन कारावास का दंड निर्धारित किया गया है"
)
_SEMANTIC_CONTROL_PARAPHRASE_DEVANAGARI = (
    "भारतीय दंड संहिता के किस प्रावधान के अंतर्गत लोक सेवक द्वारा किया गया आपराधिक न्यासभंग "
    "आजीवन कारावास से दंडनीय है"
)
# A cricket report. The Devanagari counterpart of the sourdough loaf, and the
# half this model cannot tell from a judgment: it scores 0.962 against the item,
# where that item's own rewording scores 0.990.
_SEMANTIC_CONTROL_NEGATIVE_DEVANAGARI = (
    "कल के मुकाबले में भारतीय टीम ने शानदार बल्लेबाजी करते हुए तीन विकेट से जीत हासिल की और "
    "कप्तान ने नाबाद शतक लगाकर दर्शकों का दिल जीत लिया"
)
_SEMANTIC_CONTROL_FILLER_DEVANAGARI = (
    "उभय पक्षों के विद्वान अधिवक्ताओं को सुना गया और अभिलेख पर उपलब्ध सामग्री का अवलोकन किया "
    "गया अभियोजन साक्षियों के बयानों तथा विचारण के दौरान प्रदर्शित दस्तावेजों पर विचार करने के "
    "पश्चात हम इस निष्कर्ष पर पहुंचे हैं कि नीचे के न्यायालयों द्वारा अभिलिखित समवर्ती "
    "निष्कर्षों में हस्तक्षेप का कोई आधार नहीं बनता है विचारण न्यायालय ने जो कारण दिए हैं वे "
    "अभिलेख पर उपलब्ध साक्ष्य से समर्थित हैं तथा प्रथम अपील न्यायालय ने अपने समक्ष उठाए गए "
    "प्रत्येक आधार पर विचार किया है"
).split()

# The control row is this long. NOT a multiple of the stride away from the
# window size - (137 - 30) % 10 = 7 - so the tail anchor is live in the
# control, and a row long enough that the worst-alignment scan has somewhere
# genuinely bad to find when the geometry degrades.
SEMANTIC_CONTROL_ROW_WORDS = 137


@dataclass(frozen=True)
class ScriptControl:
    """One script's two-sided control: a rewording that must flag, an unrelated
    row that must not.

    Both halves run against a ONE-ITEM index of `item`, through the same
    SemanticFilter the corpus goes through, so the control exercises the
    routing as well as the seam.
    """

    script: str
    item: str
    paraphrase: str
    negative: str
    filler: tuple[str, ...]

    @property
    def row(self) -> str:
        """The paraphrase inside a row, at the WORST alignment this geometry
        admits - see worst_alignment_offset for why that is not the stride."""
        words = self.paraphrase.split()
        pad = list(self.filler)
        while len(pad) < SEMANTIC_CONTROL_ROW_WORDS:
            pad = pad + list(self.filler)
        pad = pad[: SEMANTIC_CONTROL_ROW_WORDS - len(words)]
        at = worst_alignment_offset(len(words), SEMANTIC_CONTROL_ROW_WORDS)
        return " ".join([*pad[:at], *words, *pad[at:]])


SEMANTIC_CONTROLS = (
    ScriptControl(
        script=SCRIPT_LATIN,
        item=SEMANTIC_CONTROL_ITEM,
        paraphrase=_SEMANTIC_CONTROL_PARAPHRASE,
        negative=SEMANTIC_CONTROL_NEGATIVE,
        filler=tuple(_SEMANTIC_CONTROL_FILLER),
    ),
    ScriptControl(
        script=SCRIPT_DEVANAGARI,
        item=_SEMANTIC_CONTROL_ITEM_DEVANAGARI,
        paraphrase=_SEMANTIC_CONTROL_PARAPHRASE_DEVANAGARI,
        negative=_SEMANTIC_CONTROL_NEGATIVE_DEVANAGARI,
        filler=tuple(_SEMANTIC_CONTROL_FILLER_DEVANAGARI),
    ),
)


class SemanticSeamError(RuntimeError):
    """semhash answered in a shape or a direction this code cannot use.

    Raised rather than defaulted around. A permissive `getattr(result,
    "selected", <something>)` on an API whose attribute name is the one thing
    that could be wrong is exactly the assertion the brief forbids: under a
    result object that names its survivors anything else, the default decides
    the answer and the run records a semantic layer that never compared
    anything.
    """


class SemanticModelError(RuntimeError):
    """The embedding index could not be built.

    A DIFFERENT rung from SemanticSeamError and deliberately not a subclass of
    it: semhash downloads a model the first time it runs, so this is what an
    installed-but-air-gapped machine reads, and its remedy ("get network, or
    pre-warm the HF cache, then re-run") has nothing to do with the API drift
    a seam error reports.
    """


def semhash_available() -> bool:
    try:  # pragma: no cover - depends on the environment, not on logic
        import semhash  # noqa: F401
    except ImportError:
        return False
    return True


def semhash_index(records: Sequence[str]):
    """`SemHash.from_records`, with construction failure split BY ERROR KIND.

    Both seams build their index through here so that "the model could not be
    fetched" and "the API drifted" cannot arrive at the caller as the same
    exception - they are different statuses with different remedies.

    THE SPLIT IS THE ERROR, NOT THE CALL SITE, and that distinction cost this
    module a round. Everything raised here used to become SemanticModelError,
    whose remedy text says in as many words "This is NOT API drift" - so a
    renamed `from_records` or a renamed keyword, which is construction-side API
    drift and nothing else, sent the operator to pre-warm a cache that was
    already warm. A TypeError or an AttributeError from a constructor is the
    signature this module is written against having moved; anything else
    (OSError and its huggingface_hub descendants, connection and cache errors)
    is the model.
    """
    from semhash import SemHash  # local import: absence is a status, not a crash

    try:
        return SemHash.from_records(records=list(records))
    except (TypeError, AttributeError) as exc:
        raise SemanticSeamError(
            f"semhash's constructor is not the one this module is written against "
            f"({type(exc).__name__}: {exc}). It is documented as "
            f"`SemHash.from_records(records=[...])`; if the installed version renamed the "
            f"method or the keyword, fix semhash_index. THIS IS API DRIFT - the model and "
            f"the cache are not the problem, so pre-warming the cache will not help."
        ) from exc
    except Exception as exc:
        raise SemanticModelError(
            f"semhash could not build its embedding index ({type(exc).__name__}: {exc}). "
            f"It downloads an embedding model on first use, so this is usually a machine "
            f"with the [build] extra installed and no network: get network once, or "
            f"pre-warm the HuggingFace cache, and re-run. This is NOT API drift."
        ) from exc


def selected_records(result) -> list:
    """`result.selected`, or a named error naming what came back instead.

    There is no default because the attribute name is the one thing about this
    API that could be wrong, and a default turns being wrong about it into a
    silent clean bill of health. (Both seams have now executed against the
    real library - `DeduplicationResult.selected` is the name it uses - so
    this stands as drift protection rather than as a live unknown.)
    """
    try:
        selected = result.selected
    except AttributeError as exc:
        raise SemanticSeamError(
            f"semhash returned a {type(result).__name__} with no `.selected` "
            f"(it carries {sorted(vars(result))[:8] if hasattr(result, '__dict__') else 'no __dict__'}). "
            f"The API this module is written against is documented as "
            f"`SemHash.from_records(...).deduplicate(...).selected`; if the installed "
            f"version renames it, fix selected_records - do NOT default it, because a "
            f"default here reads as 'nothing was semantically similar'."
        ) from exc
    if selected is None:
        return []
    return list(selected)


def duplicate_provenance(result) -> tuple[str | None, float | None, str | None]:
    """(the indexed text matched, the score, the probe that matched it) - BEST EFFORT.

    THE STRONGEST EVIDENCE IN THE RESULT, not the first dropped probe: a row
    can lose several windows at once and the one that matters to an operator
    reading the drop log is the closest of them.

    `DeduplicationResult.filtered` carries a DuplicateRecord per dropped probe
    with the records it matched and their scores, and this pass used to throw
    all of it away: a semantic Hit recorded `eval_set="*"`, `item_id="semhash"`
    and nothing else, so on run one four thousand false positives and four
    thousand real leaks would have read identically in `by_level`.

    DELIBERATELY UNABLE TO RAISE, and that is the difference between this and
    selected_records. The DECISION is `len(selected) < len(probes)` and it
    stays there; this only labels a decision already made. A default in
    selected_records reads as "nothing was contaminated" and has to be an
    error; a default here reads as "we do not know which item", which is what
    the old code said about every single drop.

    IT NORMALISES TWO SHAPES, because the installed library emits two. semhash
    0.4.1 hands back a `DuplicateRecord` whose `record` and whose duplicates are
    plain strings for a NEAR duplicate and single-key dicts (`{"text": ...}`)
    for an EXACT one - but only when `deduplicate` was called with exactly one
    record, which is what a row shorter than one probe window, or one script's
    share of a mixed window, is. Unnormalised, the dict reached
    `item_by_text.get(...)` and this function's caller died on `TypeError:
    unhashable type: 'dict'` - a crash raised BY the value returned from the
    one function in this module documented as unable to raise, on the case the
    module exists for (a verbatim leak). Both shapes are read; anything else
    degrades to None, which reads as "we do not know which item".

    TIES ARE BROKEN BY TEXT, not by the order the index returned them in. Two
    eval items can sit at the same distance from a probe - the same question in
    two eval sets, a question and its own restatement - and `by_eval_set` in the
    manifest is a counter, not a diagnostic aside, so which of them gets the
    blame may not depend on an approximate index's traversal order.
    """
    best_text, best_score, best_probe = None, None, None
    try:
        filtered = list(result.filtered)
    except (AttributeError, TypeError):
        return None, None, None
    for record in filtered:
        probe = _provenance_text(getattr(record, "record", None))
        for pair in getattr(record, "duplicates", None) or ():
            try:
                text, score = _provenance_text(pair[0]), float(pair[1])
            except (TypeError, ValueError, IndexError):
                continue
            if text is None:
                continue
            if best_score is None or (score, text) > (best_score, best_text or ""):
                best_text, best_score, best_probe = text, score, probe
    return best_text, best_score, best_probe


def _provenance_text(value) -> str | None:
    """A semhash record as a string, whichever of its two shapes it arrived in.

    A single-key mapping is unwrapped (the key is the column name semhash gave
    the strings it was handed, `text` today, and this does not depend on it
    being called that); a string is itself; anything else is unknown and reads
    as None rather than as a wrong answer.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and len(value) == 1:
        inner = next(iter(value.values()))
        return inner if isinstance(inner, str) else None
    return None


class SemanticFilter:
    """Flags rows semantically close to an eval item, PER SCRIPT.

    THE ROW IS PROBED IN WINDOWS, and that is not an optimisation - it is the
    same correction containment made to Jaccard one layer up. Cosine
    similarity between a 15-word eval question and a 1,800-word row is
    dominated by the row's own length, exactly as Jaccard was. Measured
    against the installed library, an eval question quoted VERBATIM inside a
    288-word row:

        whole row only   flagged at 0.3 and 0.4 - and so was a clean row -
                         and at NOTHING from 0.5 up
        windowed         flagged at 0.5 through 0.8, clean row not flagged
                         above 0.5

    So a whole-row seam has no operating point at all: every threshold either
    sees nothing or drops everything. Cost of the windows, measured against a
    20,000-item index at the previous 20/10 geometry: 85 ms/row whole, 198
    ms/row windowed, i.e. ~25 min against ~59 min over an 18,000-row corpus.
    Widening the window to 30 did not move that: the query count is one per
    window and a wider window at the same stride produces exactly one FEWER
    (9 against 10 at 100 words, 29/30 at 300, 129/130 at 1,300). The stride is
    the cost lever, and it was not touched.

    THERE IS ONE INDEX PER SCRIPT, and every probe is SPLIT into the scripts it
    carries rather than routed whole - see script_partition for the measurement
    that forced the split and dominant_script for the one that forced the
    per-script indexes. `screened` is the set of scripts whose control half
    passed; a partition in any other script is counted as UNSCREENED, per
    script and per row, and reaches the manifest as a named hole rather than as
    silence. That count now includes the Devanagari share of an
    English-dominant row, which the routing-whole version could not see at all.

    THE WHOLE-ROW PROBE OF A ROW THAT HAS WINDOWS IS NOT SPLIT, and that
    boundary is measured. (A row with NO windows is a different case: there the
    whole-row probe IS the window and is split like one - see `route`.) Its
    residue would be every English word of a 300-word Hindi judgment strung
    together, and this embedding model is order-insensitive: a Hindi row merely
    MENTIONING the six English terms of art of an eval question scored 0.979
    against it whether they sat four words apart or a hundred and fifty, and
    whether or not they were in the question's order. That is the whole-row
    seam's own defect (no operating point at any threshold) arriving through a
    side door. A WINDOW's residue keeps its locality - the same row's window
    residues score 0.936 at four words apart, 0.790 at twelve and 0.485 at
    twenty-five - so the split is applied where locality survives it, and the
    whole-row probe keeps the dominant-script routing it always had.

    This layer only ever ADDS drops; everything the no-false-negative
    guarantee rests on is the pure-Python screen above.
    """

    def __init__(self, eval_items: Sequence, *, threshold: float = SEMANTIC_THRESHOLD,
                 screened: Iterable[str] | None = None):
        self.threshold = threshold
        # BOTH SIDES ARE PARTITIONED THE SAME WAY. An eval item is indexed under
        # every script it carries a screenable share of, as that script's words
        # only - so a code-switched eval question is compared against a
        # code-switched row in one script at a time, rather than each side
        # diluting the other. A monolingual item indexes as itself.
        self.items_by_script: dict[str, list[tuple[str, EvalItem]]] = {}
        # Minority-script partitions that were a residue rather than a share of
        # their item, per script. Counted, not silently dropped: this is the
        # decision that keeps every row citing a statute out of the drop log,
        # and a decision nothing can read is how this module has been wrong
        # before.
        self.residue_items: dict[str, int] = {}
        # The indexed texts that are a FRAGMENT of their item rather than the
        # whole of it. Read by `match` so a Hit can say so - see the residue
        # provenance there.
        self.residue_texts: set[str] = set()
        for entry in eval_items:
            item = (
                entry if isinstance(entry, EvalItem)
                else EvalItem("*", "semhash", str(entry), frozenset())
            )
            parts = script_partition(item.text)
            if not parts:
                # Under the floor in every script, or letterless. The exact
                # stack counts these as unmatchable; here they are simply not
                # indexed, and an item too short to embed is not a hole.
                continue
            dominant = dominant_script(item.text)
            for script, part in parts.items():
                if (script != dominant
                        and script_share(item.text, script) < SCRIPT_RESIDUE_MIN_SHARE):
                    # A RESIDUE, not a share of the item. Indexing it makes the
                    # statute's own name an eval item - see
                    # SCRIPT_RESIDUE_MIN_SHARE for the measurement.
                    self.residue_items[script] = self.residue_items.get(script, 0) + 1
                    continue
                if part.split() != item.text.split():
                    self.residue_texts.add(part)
                self.items_by_script.setdefault(script, []).append((part, item))
        self.screened = (
            frozenset(self.items_by_script) if screened is None else frozenset(screened)
        )
        self.indexes: dict[str, object] = {}
        self.item_by_text: dict[str, EvalItem] = {}
        for script in sorted(self.items_by_script):
            if script not in self.screened:
                continue
            group = self.items_by_script[script]
            self.indexes[script] = semhash_index([part for part, _ in group])
            for part, item in group:
                self.item_by_text.setdefault(part, item)
        # Counted, not merely skipped: `probes` is how much of the corpus this
        # layer never looked at, and `rows` is how many rows carried any of it.
        self.unscreened_probes: dict[str, int] = {}
        self.unscreened_rows: dict[str, int] = {}
        # Rows in which EVERY probe went unscreened, against rows where only
        # part did. The banner used to call any row with one unscreened window
        # "compared by the exact stack ONLY", which overstates a row whose other
        # windows were screened in full.
        self.wholly_unscreened_rows: dict[str, int] = {}
        # Probes with no letters in them at all - a table of section numbers, a
        # citation list. Counted SEPARATELY from the unscreened ones and never
        # as a hole, because there is no eval question in them to miss; counted
        # rather than dropped on the floor, because "no letterless windows" and
        # "letterless windows, ignored" must not read the same in the manifest.
        self.letterless_probes = 0
        self.letterless_rows = 0

    def screening_scripts(self) -> frozenset[str]:
        """The scripts this layer can actually find a leak in.

        Control passed AND an index was built AND that index holds items - one
        definition of `screened`, read by the manifest, the banner and the
        ladder in main() alike. The three used to be conflated: the manifest
        wrote `screened: true` for a script whose control passed over ZERO eval
        items, which is a screen that cannot find anything, and a run whose eval
        items were all Devanagari recorded `ran` having built no index at all.
        """
        return frozenset(self.indexes)

    def script_report(self) -> dict[str, dict]:
        """What this layer screened and what it did not, by script."""
        out: dict[str, dict] = {}
        seen = (set(self.items_by_script) | set(self.unscreened_probes)
                | set(self.residue_items))
        if self.letterless_probes:
            out[SCRIPT_NONE] = {
                "screened": False, "eval_items": 0, "residue_items": 0,
                # NOT `unscreened`: the banner reads those as holes and this is
                # not one. Its own pair of names, so the count is readable and
                # the hole count stays honest.
                "unscreened_probes": 0, "unscreened_rows": 0,
                "wholly_unscreened_rows": 0,
                "letterless_probes": self.letterless_probes,
                "letterless_rows": self.letterless_rows,
            }
        for script in sorted(seen):
            out[script] = {
                "screened": script in self.indexes,
                # A CODE-SWITCHED ITEM IS COUNTED ONCE PER SCRIPT IT IS INDEXED
                # IN, so these do not sum to the corpus - the question "how many
                # eval items could this script's screen have found" is per
                # script, and one item is findable in two of them.
                "eval_items": len(self.items_by_script.get(script, ())),
                # ... and the partitions in this script that were a residue of
                # their item rather than a share of it, and so are findable by
                # nobody. See SCRIPT_RESIDUE_MIN_SHARE.
                "residue_items": self.residue_items.get(script, 0),
                "unscreened_probes": self.unscreened_probes.get(script, 0),
                "unscreened_rows": self.unscreened_rows.get(script, 0),
                "wholly_unscreened_rows": self.wholly_unscreened_rows.get(script, 0),
            }
        return out

    def route(self, text: str) -> dict[str, list[str]]:
        """{script: the texts of `text` that go to that script's index}.

        Every WINDOW is split into its scripts. The whole-row probe is routed
        whole, by dominant script - but ONLY when the row is long enough to
        have windows of its own, which is the one place that exemption was ever
        measured.

        WHEN THE ROW IS NO LONGER THAN ONE WINDOW, THE WHOLE-ROW PROBE IS A
        WINDOW, and it gets a window's treatment. `probe_texts` returns
        `[text]` for a row of `SEMANTIC_PROBE_WORDS` words or fewer, and
        splitting `probes[1:]` alone left every such row routed whole by
        `dominant_script` - the exact path the split exists to remove, still
        live below window length. Measured against the shipped model, a
        reworded eval question with four Devanagari words in front of it, in
        rows of 18 to 40 words:

            row words   18 19 20 ... 29 30 | 31 32 ... 40
            probes       1  1  1       1  1 |  3  3      3
            caught      0/5      ...     0/5 | 5/5 ...  5/5

        Nothing in that band is caught, and the same rows with the four
        Devanagari words REMOVED are caught 5/5 from 20 words up - so it is the
        dilution, not the length. The exact stack does not carry them either
        (13-gram containment 0.000; 0.188 at the item's own window, against the
        0.5 it requires), and the manifest recorded `latin: screened true,
        unscreened 0` over every one of them. Re-run by
        test_a_row_no_longer_than_one_window_is_split_like_the_window_it_is.

        The whole-row exemption survives above that boundary because that is
        where its own justifying measurement was taken: a 300-word row's Latin
        residue loses the locality a window's residue keeps (see the class
        docstring's 0.979 table).

        A PROBE WITH NO PARTITION AT ALL is routed whole rather than dropped -
        a letterless window (a run of section numbers) or a probe too short to
        hold an eval item in any script. `script_partition` returns `{}` for
        both, and returning nothing for them would put a window in no bucket at
        all: `SCRIPT_NONE` is reachable from a WINDOW only through this branch,
        because a post-split partition never carries a `none` key.

        Separated from `match` so the routing can be asserted without a model
        behind it.
        """
        probes = probe_texts(text)
        if not probes:
            return {}
        windows = probes[1:]
        by_script: dict[str, list[str]] = {}
        if windows:
            by_script[dominant_script(probes[0])] = [probes[0]]
        else:
            windows = [probes[0]]
        for probe in windows:
            parts = script_partition(probe)
            if not parts:
                by_script.setdefault(dominant_script(probe), []).append(probe)
                continue
            for script, part in parts.items():
                by_script.setdefault(script, []).append(part)
        return by_script

    def match(self, text: str) -> Hit | None:
        """The eval item a window of this text is a semantic duplicate of.

        Each script's share of the row is compared only against that script's
        index. A share whose script is not screened is counted and skipped -
        never compared against another script's index, and never left inside
        another script's probe text, which are the two ways this layer used to
        lose a leak silently.
        """
        by_script = self.route(text)
        if not by_script:
            return None
        hit = None
        screened_here = 0
        letterless = False
        for script in sorted(by_script):
            group = by_script[script]
            index = self.indexes.get(script)
            if index is None:
                if script in self.screened:
                    # Screenable, and this corpus simply holds no eval item in
                    # it. Nothing to compare against is not a hole in the
                    # screen, so it is not counted as one.
                    continue
                if script == SCRIPT_NONE:
                    # A letterless probe - a table of section numbers. There is
                    # no eval question in it to miss, so it is counted under its
                    # own name and never as an unscreened hole.
                    self.letterless_probes += len(group)
                    letterless = True
                    continue
                self.unscreened_probes[script] = (
                    self.unscreened_probes.get(script, 0) + len(group)
                )
                self.unscreened_rows[script] = self.unscreened_rows.get(script, 0) + 1
                continue
            screened_here += 1
            if hit is not None:
                # Already flagged by an earlier script. The remaining queries
                # cannot change the verdict and this layer's cost is per query.
                continue
            result = index.deduplicate(records=group, threshold=self.threshold)
            kept = selected_records(result)
            if len(kept) >= len(group):
                continue
            matched_text, score, probe = duplicate_provenance(result)
            item = self.item_by_text.get(matched_text) if matched_text else None
            detail = {"threshold": self.threshold, "script": script}
            if score is not None:
                detail["score"] = round(score, 4)
            if probe is not None:
                detail["probe_words"] = len(probe.split())
            # WHICH SIDES OF THIS MATCH WERE FRAGMENTS. A terms-of-art false
            # positive and a buried verbatim leak were byte-identical shapes in
            # the drop log - same level, same score band, same fields - and
            # "the per-Hit provenance makes the real rate readable on run one"
            # was the whole argument for shipping the 0.8 operating point. Both
            # are recorded whether true or false: a missing key reads as an
            # older run, a `false` reads as a whole-text match.
            detail["item_residue"] = bool(
                matched_text is not None and matched_text in self.residue_texts
            )
            detail["probe_residue"] = bool(
                probe is not None and probe not in set(probe_texts(text))
            )
            hit = Hit(
                LEVEL_SEMANTIC,
                item.set_key if item else "*",
                item.item_id if item else "semhash",
                detail,
            )
        if letterless:
            self.letterless_rows += 1
        if not screened_here:
            for script in by_script:
                if script in self.unscreened_rows:
                    self.wholly_unscreened_rows[script] = (
                        self.wholly_unscreened_rows.get(script, 0) + 1
                    )
        return hit

    def matches(self, text: str) -> bool:
        """Is any window of this text a semantic duplicate of an eval item?"""
        return self.match(text) is not None

    def __call__(self, item: Item):
        hit = self.match(item.text)
        return (hit,) if hit is not None else ()


def semantic_controls(*, threshold: float = SEMANTIC_THRESHOLD) -> dict[str, str]:
    """{script: "" if its control passed, else why it did not}.

    `semantic: "ran"` in the manifest has to mean "this layer can find a
    reworded eval question inside a row", not "the call did not raise". The
    two ways it can be wrong point in opposite directions and both are silent:
    a result object whose survivors are named something else makes every row
    look like a duplicate (drops everything), and the reverse default makes
    every row look clean (drops nothing, which is this module's catastrophic
    direction).

    THE POSITIVE HALF IS A NEAR-PARAPHRASE INSIDE A ROW, and that is the whole
    design of this function. The control it replaced fed the seam an exact
    copy of an eval item, which passed at every threshold from 0.3 to 0.95 -
    including the one it shipped at, where the seam caught 0 of 2 paraphrases.
    A control that a power-less seam passes certifies nothing. This one fails
    for an exact-match seam, fails for a whole-row seam (measured: a verbatim
    leak in a 288-word row is invisible to one), fails when the probe geometry
    stops covering the row (the paraphrase is placed at the worst alignment the
    geometry admits, not at a stride boundary), and has a CEILING: measured
    cache-only at the shipped geometry it passes at 0.7 through 0.85 and fails
    at 0.9 and 0.95, where a rewording is no longer visible. The dedupe control
    makes no ceiling claim and does not need to; this one is the reason the
    threshold can be called measured rather than assumed.

    IT IS PER SCRIPT, and that is what turns a model's blindness into a
    recorded hole instead of four thousand silent false positives. Each half
    runs against its OWN one-item index - the eval texts are whatever was
    downloaded, and a control has to be a question whose answer is known before
    the run - and a script whose half fails is simply not screened.
    """
    out: dict[str, str] = {}
    for control in SEMANTIC_CONTROLS:
        seam = SemanticFilter(
            [control.item], threshold=threshold, screened=[control.script],
        )
        if not seam.matches(control.row):
            out[control.script] = (
                f"the semantic layer did not find a REWORDED copy of its {control.script} "
                f"control item inside a row at threshold {threshold}. A layer that cannot "
                f"recognise a rewording cannot recognise the paraphrase it exists for, and "
                f"recording it as having run would put a screen in the manifest that never "
                f"screened anything. (An exact-match seam, a whole-row seam, a probe geometry "
                f"that no longer covers the row and a too-high threshold all land here - the "
                f"semantic_detail beside this says which was asked for.)"
            )
        elif seam.matches(control.negative):
            out[control.script] = (
                f"the semantic layer flagged {control.script} text with nothing to do with "
                f"Indian law, so in that script it flags everything - every row written in it "
                f"would be dropped as contaminated."
            )
        else:
            out[control.script] = ""
    return out


def screened_scripts(results: dict[str, str]) -> list[str]:
    """The scripts this run may screen, from every script's control result.

    Raises unless AT LEAST ONE script's control passed. The ladder: a script
    whose half fails is not screened and is recorded as such; a run where NO
    half passes has no working seam at all and is `semhash-control-failed`, the
    same rung as a drifted API. dedupe.py's control is single-script by
    construction (it compares this corpus against itself, and rule 2 already
    scopes it) and keeps its own shape.

    The ladder lives HERE and not in main() so there is one of it. The version
    this replaced was a `semantic_control()` that main() had stopped calling -
    it re-derived the same decision inline - so inverting the function's
    condition changed no behaviour anywhere and the mutation survived the whole
    suite.
    """
    passed = sorted(script for script, why in results.items() if not why)
    if not passed:
        raise SemanticSeamError(" / ".join(why for why in results.values() if why))
    return passed


# --------------------------------------------------------------------------
# Loading the candidate rows.
# --------------------------------------------------------------------------

def stream_items(paths: Iterable[Path], *, ids_from_text: bool = True) -> Iterator[Item]:
    """Rows out of the assembly stream files, in file then line order."""
    from tuned.data.jsonl import read_jsonl

    for path in paths:
        for i, row in enumerate(read_jsonl(path)):
            yield item_of(row, f"{Path(path).name}#{i}", ids_from_text=ids_from_text)


def stream_paths(streams_dir: Path) -> list[Path]:
    """Every stream file, in sorted order.

    Sorted, not glob order: the input order is the survivor order downstream
    (dedupe keeps the first row of a duplicate cluster), so a filesystem's
    idea of directory order would decide which rows ship.
    """
    return sorted(p for p in Path(streams_dir).glob("*.jsonl") if p.is_file())


def generated_rows(store, cfg=None, *, state: str = "accepted") -> Iterator[dict]:
    """The accepted generations, as assembly rows.

    THE PROMPT IS THE GROUNDING, NOT THE RENDERED TEMPLATE. Every generated
    row is rendered from the same handful of prompt templates, so a text
    comparison over the rendered prompt would find every pair of rows ~80%
    identical - the template, not the content. What distinguishes these rows
    is the seed's grounding text and the model's answer, so that is what this
    pass compares. The rendered row is assemble.py's business and is built
    later, after this decision.

    A row the seed no longer backs still comes through (with an empty
    grounding) rather than being skipped: skipping it would let a row into the
    dataset that this pass never screened.
    """
    think_open = getattr(cfg, "think_open", "<think>")
    think_close = getattr(cfg, "think_close", "</think>")
    for gen in store.latest_generations(state):
        seed = store.get_seed(gen["seed_id"]) or {}
        think, answer = gen.get("think") or "", gen.get("answer") or ""
        content = f"{think_open}\n{think}\n{think_close}\n\n{answer}" if think else answer
        scores = [
            value
            for judgement in store.judgements_for(gen["gen_id"])
            for value in (judgement["grounding"], judgement["validity"], judgement["coverage"])
            if value is not None
        ]
        yield {
            "messages": [
                {"role": "user", "content": seed.get("text") or ""},
                {"role": "assistant", "content": content},
            ],
            "_prov": {
                "source": gen.get("stream"),
                "native_id": seed.get("native_id"),
                "cnr": seed.get("cnr"),
                "neutral_citation": seed.get("neutral_citation"),
                "task_type": gen.get("task_type"),
                "prompt_id": gen.get("prompt_id"),
                "seed_id": gen.get("seed_id"),
                "gen_id": gen.get("gen_id"),
                "score": round(sum(scores) / len(scores), 3) if scores else None,
                "reasoning": bool(think),
            },
        }


def store_items(store, cfg=None, *, state: str = "accepted",
                ids_from_text: bool = True) -> Iterator[Item]:
    for row in generated_rows(store, cfg, state=state):
        yield item_of(row, f"store:{row['_prov']['gen_id']}", ids_from_text=ids_from_text)


# --------------------------------------------------------------------------
# The manifest: what this pass could and could not see.
# --------------------------------------------------------------------------

def output_record(path: Path, rows: int) -> dict:
    """What this pass wrote, identified by CONTENT.

    The chain of custody downstream cannot be a directory: dedupe.py used to
    read the manifest beside its input and inherit whatever it found there, so
    an --in from anywhere else adopted a manifest describing OTHER ROWS and
    shipped never-screened rows under a decontamination stamp. A digest of the
    bytes actually written is the only claim that cannot be inherited by a
    file that was not screened.
    """
    from tuned.data.acquire import sha256_file

    return {"path": str(path), "rows": rows, "sha256": sha256_file(path)}


def semantic_script_record(layer, controls: dict[str, str] | None) -> dict:
    """PER SCRIPT: which the semantic layer screened, which it could not, why.

    An honest hole, carried like a waiver. `potion-base-8M` has no
    discriminative power over Devanagari (see dominant_script), so on the first
    real run this block is where the manifest says that Hindi - 7,318 of BBL's
    24,365 questions - was screened by the exact stack alone. A dataset card
    that claims a paraphrase screen has to be able to say which scripts it
    covered, and `semantic: ran` alone cannot.

    `screened` MEANS ONE THING HERE: the control passed AND an index was built
    AND it holds at least one eval item. Two definitions used to be in play and
    the manifest's was the weaker of them, so a script with zero eval items read
    `screened: true` - a screen that cannot find anything, recorded as one that
    can. Without a layer at all (the control itself failed, or semhash is not
    installed) each script still gets its block, with `index: false`: an empty
    `semantic_scripts` object was the same shape for "not installed", "no eval
    items" and "the control failed", which is three facts wearing one silence.
    """
    report = layer.script_report() if layer is not None else {}
    out: dict[str, dict] = {}
    for script, why in sorted((controls or {}).items()):
        out[script] = {
            "control": "passed" if not why else why,
            "control_passed": not why,
            "screened": False,
            "index": False,
            "eval_items": 0,
            "residue_items": 0,
            "unscreened_probes": 0,
            "unscreened_rows": 0,
            "wholly_unscreened_rows": 0,
        }
    for script, entry in report.items():
        record = out.setdefault(
            script,
            {"control": "no control half for this script", "control_passed": True,
             "screened": False, "index": False},
        )
        index_built = bool(entry["screened"])
        record.update(entry)
        record["index"] = index_built
        # ONE definition, in one place, and BOTH clauses have a case:
        # `control_passed` is false for a script the seam has no power in, and
        # `index_built` is false for a script whose control passed over no eval
        # items at all. "Items > 0" is not a third clause because it cannot be
        # false when index_built is true - SemanticFilter only ever builds an
        # index for a script it has items for, which is asserted directly
        # rather than restated here as a branch with no case.
        record["screened"] = bool(record["control_passed"] and index_built)
    if SCRIPT_NONE in out:
        # Named, counted and explicitly NOT a hole. A probe with no letters in
        # it is a table of section numbers or a citation list; there is no eval
        # question in it to miss, and the banner does not report it as one.
        out[SCRIPT_NONE]["control"] = (
            "no letters in these probes at all, so there is no eval question in them "
            "to miss - counted rather than screened, and not a hole"
        )
        out[SCRIPT_NONE]["control_passed"] = False
    return dict(sorted(out.items()))


def manifest_of(stats: dict, corpora: dict[str, EvalCorpus], index: EvalIndex, *,
                inputs: Sequence[str], semantic: str, semantic_detail: str = "",
                output: dict | None = None, generations: dict | None = None,
                threshold: float = CONTAINMENT, ids_from_text: bool = True,
                semantic_layer=None, semantic_controls: dict[str, str] | None = None,
                top: int = 20) -> dict:
    """The record that has to outlive this run.

    Every waived eval set, every hole in the screen and every threshold is in
    here, because "was this dataset screened against BBL?" has to be
    answerable from the dataset itself years later - a printed warning is not
    an answer.
    """
    from tuned.data.store import utcnow

    coverage = stats["with_identifier"] / stats["total"] if stats["total"] else 0.0
    lengths = index.length_report()
    return {
        "stage": "decontaminate",
        "decon_version": DECON_VERSION,
        "at": utcnow(),
        "inputs": list(inputs),
        # The rows this pass produced, by content. dedupe.py verifies its own
        # input against this and refuses to inherit the record otherwise.
        "output": output,
        "thresholds": {
            "ngram": index.n,
            "containment": threshold,
            "short_min_tokens": index.short_min,
            # Below this many tokens an item is screened at a NARROWER window
            # than `ngram` - see window_for. The number an operator reading a
            # `narrow` drop count needs beside it.
            "full_ngram_from_tokens": 3 * index.n - 1,
            "title_min_tokens": TITLE_MIN_TOKENS,
            "case_ids_from_text": ids_from_text,
            # The semantic layer's operating point and probe geometry, recorded
            # whether or not it ran: `semantic: ran` at one threshold is not the
            # same screen as `ran` at another, and the dataset card has to be
            # able to say which.
            "semantic": SEMANTIC_THRESHOLD,
            "semantic_probe_words": SEMANTIC_PROBE_WORDS,
            "semantic_probe_stride": SEMANTIC_PROBE_STRIDE,
        },
        "counts": {k: stats[k] for k in ("total", "kept", "dropped", "empty_text")},
        "by_reason": dict(sorted(stats["by_reason"].items())),
        "by_level": dict(sorted(stats["by_level"].items())),
        "by_eval_set": dict(sorted(stats["by_eval_set"].items())),
        "eval_sets": {
            key: {
                "repo_id": corpus.spec.repo_id,
                "license": corpus.spec.license,
                "status": corpus.status,
                "allowed_missing": corpus.allowed_missing,
                "files": corpus.files,
                "rows": corpus.rows,
                # What was screened against and what was left out, named. A
                # subset that is not written down narrows the guarantee
                # silently, and this is the record that outlives the run.
                "selection": corpus.selection,
                "expect_parts": [
                    {"config": part.config, "split": part.split, "rows": part.rows}
                    for part in corpus.spec.parts
                ],
                "expect_rows": corpus.spec.expect_rows,
                "expect_verified_at": EVAL_COUNTS_VERIFIED_AT if corpus.spec.parts else None,
                "row_shortfall": corpus.shortfall,
                # `row_shortfall: 0` is otherwise two different facts wearing
                # one number - "the download is complete" and "there is no
                # verified count to compare it against, so the instrument is
                # off". This says which.
                "row_shortfall_measured": corpus.spec.expect_rows is not None,
                "row_surplus": corpus.surplus,
                "items": len(corpus.items),
                "text_field": corpus.text_field,
                "detail": corpus.detail,
                # The token-length histogram, by the level that screens each
                # band. Read this against the window calibration on run one.
                "item_tokens": lengths.get(key),
            }
            for key, corpus in sorted(corpora.items())
        },
        # The holes, named. An eval item too short for any rule is not
        # screened against, and a corpus whose rows carry no case identifier
        # never met the case-identifier level.
        # Whether the accepted generations were screened at all. A run that
        # only looked at the stream files must not be indistinguishable, years
        # later, from one that looked at everything - and neither must a run
        # that asked for them and got none back.
        "generations_screened": bool((generations or {}).get("screened")),
        "generations": generations or {"screened": False, "state": None, "read": 0},
        # PER SET. A pooled number cannot tell the operator which set is
        # blind, and the refusal above needs the per-set count anyway.
        "unmatchable_eval_items": {
            key: entry["unmatchable"] for key, entry in sorted(lengths.items())
        },
        "unmatchable_eval_items_total": len(index.unmatchable),
        "case_identifier_coverage": round(coverage, 4),
        "case_identifier_level_inert": not stats["with_identifier"] or not index.by_identifier,
        "eval_identifiers": len(index.by_identifier),
        # "ran" here means OBSERVED WORKING - semantic_control passed on a pair
        # whose answer is not in doubt - not merely "called". See
        # semantic_control for the two silent directions it pins.
        "semantic": semantic,
        "semantic_detail": semantic_detail,
        # WHICH SCRIPTS the semantic layer could actually screen. `ran` over a
        # corpus 30% of which is in a script the model is blind to is not the
        # same screen as `ran`, and this is where the difference is written
        # down. See semantic_script_record.
        "semantic_scripts": semantic_script_record(semantic_layer, semantic_controls),
        # ROW-MATCHES per identifier, over CASE-IDENTIFIER drops only, and
        # TRUNCATED to the top `top`. Both halves of that sentence were wrong
        # here before: it claimed to sum to at least `dropped`, which is false
        # twice over - a row dropped on text or on the semantic layer carries
        # no identifier and contributes nothing (11 drops summed to 1 on a real
        # fixture), and the slice below discards the tail (30 identifiers
        # summed to 20). What IS true is that every case-identifier hit blames
        # at least one identifier, so the TOTAL is at least
        # `by_level["case_id"]` - and the total is recorded beside the slice so
        # the truncation is visible rather than inferred.
        #
        # A row that shares two identifiers with the eval side is blamed on
        # both, because --no-case-id-from-text removes a whole channel and the
        # operator deciding whether to pass it has to see every identifier that
        # would have cost this row.
        "top_identifiers": sorted(
            stats["identifier_drops"].items(), key=lambda kv: (-kv[1], kv[0])
        )[:top],
        "identifier_drops_total": sum(stats["identifier_drops"].values()),
        "identifier_drops_identifiers": len(stats["identifier_drops"]),
        "top_identifiers_shown": min(top, len(stats["identifier_drops"])),
    }


def write_manifest(path: Path, manifest: dict) -> None:
    """Durably, same rule as everything else here: written to a sibling .tmp
    and renamed, so the manifest is either absent or whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None, *, reader=read_rows) -> int:
    import argparse

    from tuned.data.config import load_build_config
    from tuned.data.jsonl import write_jsonl
    from tuned.data.paths import build_paths
    from tuned.data.store import TASK_STATES, Store

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data_law_v1.yaml")
    parser.add_argument("--in", dest="inputs", action="append", default=None,
                        help="stream JSONL to screen (repeatable; default every *.jsonl "
                             "in the streams dir)")
    parser.add_argument("--out", default=None, help=f"default out/{OUT_FILENAME}")
    parser.add_argument("--allow-missing-eval", action="append", default=[], metavar="KEY",
                        choices=sorted(EVAL_SETS),
                        help="run without this eval set - RECORDED IN THE MANIFEST")
    parser.add_argument("--no-generated", action="store_true",
                        help="screen the stream files only, not the accepted generations")
    parser.add_argument("--no-case-id-from-text", action="store_true",
                        help="take a row's case identifiers from _prov only, not from the "
                             "citations its grounding names")
    parser.add_argument("--require-semantic", action="store_true",
                        help="refuse to run when semhash is unavailable")
    parser.add_argument("--state", default="accepted", choices=TASK_STATES,
                        help="task state to read (default accepted). Validated: a typo'd "
                             "state used to read zero generations and still record them "
                             "as screened, and --state '' used to disable the filter "
                             "entirely and ship REJECTED generations")
    args = parser.parse_args(argv)

    cfg = load_build_config(args.config)
    paths = build_paths(cfg.build.workdir).ensure()
    out_path = Path(args.out) if args.out else paths.out_dir / OUT_FILENAME
    ids_from_text = not args.no_case_id_from_text

    store = Store.open(paths.state_db)
    try:
        corpora = eval_corpora(store, allow_missing=args.allow_missing_eval, reader=reader)
        blocked = refusals(corpora)
        if blocked:
            print(REFUSAL_HEADER)
            for line in blocked:
                print(f"  {line}")
            print("  nothing was written; no output carries a decontaminated stamp.")
            return 2

        semantic_fn, semantic_status, semantic_detail = None, SEMANTIC_UNAVAILABLE, ""
        semantic_controls_ran: dict[str, str] = {}
        eval_items = [item for c in corpora.values() for item in c.items]
        if semhash_available():
            if not eval_items:
                # Every set waived: there is nothing for this layer to compare
                # against, which is not the same fact as "it did not run" and
                # must not be recorded as "ran".
                semantic_status = SEMANTIC_NO_ITEMS
            else:
                try:
                    # The controls FIRST, on one-item indexes: they cost a
                    # second and they prove the model, the API shape and the
                    # seam's power PER SCRIPT before the 20,000-item index is
                    # built. A script whose control fails is not screened.
                    semantic_controls_ran = semantic_controls()
                    seam = SemanticFilter(
                        eval_items, screened=screened_scripts(semantic_controls_ran)
                    )
                except SemanticModelError as exc:
                    # Its own rung. The remedy is network or a warm cache, and
                    # sending an operator to look for API drift instead is the
                    # wrong instruction on the commonest failure this layer has.
                    semantic_status, semantic_detail = SEMANTIC_NO_MODEL, str(exc)
                except Exception as exc:
                    # NOT recorded as "ran". The seam is installed and
                    # answered; it answered in a way that means nothing was
                    # compared.
                    #
                    # Broad on purpose, and only around the control plus
                    # construction: this layer's absence is a STATUS, not a
                    # crash that kills a decontamination run the refusal ladder
                    # has already cleared, and semhash is entitled to raise
                    # anything it likes from inside a model it loaded lazily.
                    # Once the control passes the shape is confirmed and the
                    # per-row path is not wrapped.
                    semantic_status = SEMANTIC_UNUSABLE
                    semantic_detail = (
                        str(exc) if isinstance(exc, SemanticSeamError)
                        else f"{type(exc).__name__}: {exc}"
                    )
                else:
                    if seam.screening_scripts():
                        semantic_fn, semantic_status = seam, SEMANTIC_RAN
                    else:
                        # Items, a working control in SOME script, and no index:
                        # every eval item is in a script this model is blind to.
                        # The layer is kept for the manifest's per-script record
                        # and is NOT recorded as having run.
                        semantic_fn = seam
                        semantic_status = SEMANTIC_NO_SCREENABLE_ITEMS
                        semantic_detail = (
                            f"the control passed for "
                            f"{', '.join(sorted(screened_scripts(semantic_controls_ran)))} "
                            f"and every eval item is in another script, so no index was "
                            f"built and nothing could have been found. See "
                            f"semantic_scripts for the per-script counts; a multilingual "
                            f"embedding model is what turns this on."
                        )
        if args.require_semantic and semantic_status != SEMANTIC_RAN:
            print(
                f"REFUSING TO DECONTAMINATE: --require-semantic was passed and the semantic "
                f"layer is {semantic_status}.\n"
                f"  {semantic_detail or 'run: pip install -e .[build]'}"
            )
            return 2

        inputs = (
            [Path(p) for p in args.inputs] if args.inputs else stream_paths(paths.streams_dir)
        )
        missing = [str(p) for p in inputs if not Path(p).exists()]
        if missing:
            print(f"no such input: {', '.join(missing)}")
            return 2

        items = list(stream_items(inputs, ids_from_text=ids_from_text))
        input_names = [str(p) for p in inputs]
        generations = {"screened": False, "state": None, "read": 0}
        if not args.no_generated:
            generated = list(store_items(store, cfg, state=args.state, ids_from_text=ids_from_text))
            print(f"read {len(items)} stream rows and {len(generated)} {args.state} generations")
            items += generated
            input_names.append(f"store:{args.state}")
            # What ACTUALLY happened, not what was asked for: `true` used to
            # mean only that --no-generated was absent, so a run that read
            # nothing at all from the store recorded the generations as
            # screened.
            generations = {
                "screened": bool(generated), "state": args.state, "read": len(generated),
            }
            if not generated:
                print(
                    f"    NO GENERATIONS IN STATE {args.state!r} - the store side of this run "
                    f"screened NOTHING, and the manifest records generations_screened: false. "
                    f"If tasks have been accepted, check --state."
                )
        else:
            print(f"read {len(items)} stream rows (generations NOT screened: --no-generated)")

        index = EvalIndex([item for c in corpora.values() for item in c.items])
        manifest_bands = index.length_report()
        for key in sorted(corpora):
            corpus = corpora[key]
            state = corpus.status if corpus.ok else f"{corpus.status} (WAIVED)"
            print(f"  eval {key:<6} {len(corpus.items):>7} items  {state}"
                  f"{'  <- ' + corpus.text_field if corpus.text_field else ''}")
            bands = manifest_bands.get(key)
            if bands:
                # The token-length histogram, printed where the operator is
                # already looking: which level each set is actually screened
                # by is what decides whether the window calibration holds.
                print(
                    f"    tokens: median {bands['median_tokens']}"
                    f" (min {bands['min_tokens']}, max {bands['max_tokens']})"
                    f"  screened by: {LEVEL_TEXT} {bands[LEVEL_TEXT]},"
                    f" {LEVEL_NARROW} {bands[LEVEL_NARROW]},"
                    f" {LEVEL_SHORT} {bands[LEVEL_SHORT]},"
                    f" unmatchable {bands['unmatchable']}"
                )
            if corpus.shortfall:
                # Not a refusal below the floor, but never silent either: the
                # verified count is the only handle on "this is a fragment of
                # the set" that does not require the network. Denominated
                # against the splits actually selected, so a complete download
                # reads zero here and a one-config BBL reads 7,318.
                print(
                    f"    {corpus.rows} rows, {corpus.spec.expect_rows} expected across"
                    f" {len(corpus.spec.parts)} config/split"
                    f" ({EVAL_COUNTS_VERIFIED_AT}) - {corpus.shortfall} SHORT."
                    f" A config or a shard is missing; check before trusting this screen"
                )
            if corpus.surplus:
                print(
                    f"    {corpus.rows} rows against {corpus.spec.expect_rows} expected"
                    f" - {corpus.surplus} MORE than the eval surface. The split filter"
                    f" selected more than it should have; check `selection` in the manifest"
                )
            excluded = corpus.selection.get("excluded") or []
            if excluded:
                # PARTIAL exclusion, which only the manifest recorded before:
                # the count and the reasons are what say whether the filter cut
                # the training splits or cut a whole task by accident.
                whys: dict[str, int] = {}
                for entry in excluded:
                    whys[entry["why"]] = whys.get(entry["why"], 0) + 1
                print(
                    f"    {len(excluded)} of {len(excluded) + len(corpus.selection['selected'])}"
                    f" objects left out of the eval surface"
                    f" ({', '.join(f'{why} {n}' for why, n in sorted(whys.items()))})"
                    f" - `selection` in the manifest names every one"
                )
            if corpus.ok and corpus.spec.expect_rows is None:
                print(
                    f"    no verified row count for this set, so the floor and the shortfall"
                    f" line are OFF here - {corpus.spec.selection_note or 'whole set'}"
                )
            if corpus.ok and len(corpus.items) < corpus.rows:
                # Some rows of a set that IS loaded carried none of the
                # candidate column names. Screening against 10% of BBL while
                # the status reads `ok` is the shape this module exists to
                # refuse, so the shortfall is printed next to the status.
                print(
                    f"    {corpus.rows - len(corpus.items)} of {corpus.rows} rows carried none"
                    f" of {_EVAL_TEXT_FIELDS[:4]}... - THOSE QUESTIONS WERE NOT SCREENED"
                    f" AGAINST. Add the real column name to _EVAL_TEXT_FIELDS."
                )

        kept, drops, stats = decontaminate_items(items, index, semantic=semantic_fn)
        # Written FIRST, so the manifest can carry the digest of the bytes
        # that actually landed rather than of the bytes this pass intended.
        written = write_jsonl(out_path, [item.row for item in kept])
        manifest = manifest_of(
            stats, corpora, index, inputs=input_names, semantic=semantic_status,
            semantic_detail=semantic_detail, ids_from_text=ids_from_text,
            output=output_record(out_path, written), generations=generations,
            semantic_layer=semantic_fn, semantic_controls=semantic_controls_ran,
        )
        write_jsonl(out_path.parent / DROPS_FILENAME, drops)
        write_manifest(out_path.parent / MANIFEST_FILENAME, manifest)
        store.log_event("decontamination", manifest)

        print(f"  screened {stats['total']}  kept {stats['kept']}  dropped {stats['dropped']}")
        for reason, count in sorted(stats["by_reason"].items()):
            print(f"    drop[{reason}]: {count}")
        print(
            f"  case identifiers: {stats['with_identifier']}/{stats['total']} rows carried one"
            f" ({manifest['case_identifier_coverage']:.1%}), eval side "
            f"{manifest['eval_identifiers']}"
        )
        if manifest["case_identifier_level_inert"]:
            # The Task-11 lesson: an instrument that reads healthy in exactly
            # the case it exists for. A level with nothing to match on did not
            # run, and must not be mistaken for a level that found nothing.
            print(
                "    THE CASE-IDENTIFIER LEVEL DID NOT RUN - one side carried no identifiers"
                " at all. That is the level that catches SAME JUDGMENT, DIFFERENT QUESTION,"
                " which no n-gram method can see. Nothing in the pipeline populates seed.cnr"
                " or seed.neutral_citation today; populate them, or accept that this run was"
                " screened on text alone (the manifest says which)."
            )
        for key, count in sorted(manifest["unmatchable_eval_items"].items()):
            if count:
                print(
                    f"    {key}: {count} eval items are under {SHORT_MIN_TOKENS} tokens and "
                    f"NOTHING here can match them"
                )
        if stats["empty_text"]:
            print(
                f"    {stats['empty_text']} candidate rows carried NO USABLE TEXT - they can "
                f"match nothing, so they pass this screen by being unreadable, not by being clean"
            )
        if not manifest["generations_screened"]:
            print(
                "    THE ACCEPTED GENERATIONS WERE NOT SCREENED (--no-generated). They are the "
                "rows built FROM the seed pools the eval sets draw on, so this run has not "
                "screened the corpus - recorded in the manifest as generations_screened: false"
            )
        if manifest["top_identifiers"]:
            print("  identifiers that cost the most rows (read this before trusting the yield):")
            for identifier, count in manifest["top_identifiers"][:5]:
                print(f"    {identifier:<48} {count}")
        for script, entry in manifest["semantic_scripts"].items():
            if entry["screened"] or not (entry["eval_items"] or entry["unscreened_rows"]):
                continue
            # A hole with a name and a size. The alternative - and what shipped
            # before this - was a run in which the Hindi third of BBL produced
            # thousands of drops that read exactly like real leaks.
            #
            # THE TWO ROW COUNTS ARE DIFFERENT FACTS. A row with one unscreened
            # Devanagari window and thirty screened English ones was screened,
            # with a hole in it; a row that is entirely in this script was not
            # screened at all. Printing the first number as the second is the
            # overstatement direction, and this instrument's whole job is to
            # not overstate in either.
            whole = entry.get("wholly_unscreened_rows", 0)
            print(
                f"    SEMANTIC SCREENING IS OFF FOR {script.upper()}:"
                f" {entry['eval_items']} eval items and {entry['unscreened_rows']} candidate"
                f" rows carried text this layer could not read"
                f" ({whole} of those rows were compared by the exact stack ONLY;"
                f" the rest were screened in another script)."
            )
            print(f"      {entry['control']}")
        if semantic_status != SEMANTIC_RAN:
            print(
                f"  semantic layer did NOT run ({semantic_status}) - paraphrased eval questions"
                f" are not screened. Recorded in the manifest."
            )
            if semantic_detail:
                print(f"    {semantic_detail}")
        for key in sorted(corpora):
            if corpora[key].allowed_missing:
                print(f"  WAIVED: {key} was not screened against ({corpora[key].status}) - "
                      f"this is recorded in {MANIFEST_FILENAME} and travels to the dataset card")
        print(f"wrote {written} rows -> {out_path}")
        print(f"      {len(drops)} drops -> {out_path.parent / DROPS_FILENAME}")
        print(f"      manifest -> {out_path.parent / MANIFEST_FILENAME}")
        if stats["total"] and not written:
            print(
                "  EVERYTHING WAS DROPPED from a non-empty input - that is a matcher fault, "
                "not a corpus: check the eval item counts above against the drop reasons."
            )
            return 1
        if not stats["total"]:
            print(
                "  NOTHING READ: no candidate rows at all. Exiting 0 here would stamp an "
                "empty dataset decontaminated - check --in and whether any task is accepted."
            )
            return 1
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    import os
    import sys

    exit_code = main()
    # Same reasoning as select.py/extract.py: pyarrow/hf-xet can leave
    # non-daemon threads that wedge interpreter shutdown after all output is
    # written. Skip shutdown entirely.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
