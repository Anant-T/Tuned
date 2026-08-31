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

Build:  python -m tuned.data.decontaminate --config data/configs/data_law_v1.yaml
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

from tuned.data.acquire import HF_SOURCES, rebase_under_corpus
from tuned.data.citations import extract_citations
# answer_without_preamble lives in gates: the length band has to measure the
# answer THAT WILL SHIP, so the gate and this assembler must cut in exactly
# one place. Re-exported here because that is where callers found it first.
from tuned.data.gates import (  # noqa: F401
    PREAMBLE_MIN_CHARS,
    TRIMMED_MIN_CHARS,
    answer_without_preamble,
)
from tuned.data.select import landmark_key
from tuned.data.paths import DEFAULT_CONFIG

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
# 5: the exact index MEASURES ITSELF - grams and an estimated footprint, per
#    eval set and in total - and `--max-index-grams` turns a memory-kill during
#    construction into a named refusal that says which set crossed the line.
#    Every drop record also carries `source` beside `form`: `form` prefers a
#    row's task_type, so a generated row's drop filed under `irac_analysis`
#    could not be attributed to the stream shape.py sizes by, which made the
#    retention of the generated streams unmeasurable from this file.
#    No rule changed; the manifest gained the numbers that decide whether the
#    index has to be sharded.
DECON_VERSION = 5

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

# WHAT ONE GRAM COSTS IN THE INVERTED INDEX. Measured 2026-08-30 with
# tracemalloc over three synthetic corpora (200 items x 2k words, 400 x 2k,
# 100 x 8k): 176.5-176.8 bytes per gram, flat in both item count and item
# length. It is an UPPER bound for real text, where a repeated gram appends to
# an existing posting list instead of opening a new dict entry - so an estimate
# built on it errs toward refusing early, which is the right direction for a
# guard against a memory kill. It does NOT cover the eval items' own text,
# which decontaminate_items holds alongside the index.
INDEX_BYTES_PER_GRAM = 177

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


def token_hashes(toks: Sequence[str]) -> list[int]:
    """The per-token crc32 pass, on its own.

    Split out of gram_hashes because it does not depend on n and the caller
    that dominates this module does: EvalIndex.query grams ONE row at EVERY
    window the index holds - a dozen of them for a real eval mix - and fused,
    each of those windows re-encoded and re-CRC'd the row's whole token list.
    Roughly an order of magnitude of the same work for the same numbers.
    Nothing about the result changes; gram_hashes is still the composition.
    """
    return [zlib.crc32(t.encode("utf-8")) for t in toks]


def grams_from(hs: Sequence[int], n: int) -> frozenset[int]:
    """The rolling n-window, over tokens that are already hashed.

    The subtract-and-shift keeps this one multiply per token rather than n.
    See gram_hashes for why the hashes underneath have to be crc32.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if len(hs) < n:
        return frozenset()
    power = pow(_BASE, n, 1 << 64)
    window = 0
    for value in hs[:n]:
        window = (window * _BASE + value) & _MASK
    out = [window]
    for i in range(n, len(hs)):
        window = (window * _BASE - hs[i - n] * power + hs[i]) & _MASK
        out.append(window)
    return frozenset(out)


def gram_hashes(toks: Sequence[str], n: int = NGRAM) -> frozenset[int]:
    """Stable 64-bit hashes of every n-token window.

    STABLE is the load-bearing word. Python's `hash()` for str is salted per
    process (PYTHONHASHSEED), so an index keyed on it selects a different
    candidate set - and therefore a different dataset - on every run.
    crc32 is deterministic across runs, machines and Python versions.

    A 64-bit collision can only ever ADD to a posting count, never remove
    from one, so the error direction is the safe one - it can cost a row, not
    hide a leak. It is NOT rejected downstream, though: the posting count IS
    the intersection size the arithmetic divides, so a collision inflates
    containment directly rather than proposing a candidate something later
    throws out. (The comment here used to claim the latter, which would have
    been a stronger guarantee than the code gives.)

    This is the one-window form and stays the module's public spelling. A
    caller that grams the SAME tokens at several windows should hash once
    with token_hashes and window with grams_from.
    """
    return grams_from(token_hashes(toks), n)


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
# An assembly row is {"messages": [...], "_prov": {...}} - the shape
# replay.assembly_row builds (curated.py imports it), and the shape store rows
# are lifted into below. The row
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
# The index: exact candidate generation over the eval side.
# --------------------------------------------------------------------------

class EvalIndexTooLarge(RuntimeError):
    """The exact index would not fit. Refuse by name rather than by OOM kill.

    A memory kill inside EvalIndex is exit 137 with no output at all: no
    manifest, no counts, nothing that says WHICH eval set grew. The eval
    surface is chosen by split NAME across whole IL-TUR configs, so one
    upstream re-release can multiply it without a line of this repo changing.
    This carries the gram count, the estimate and the set that crossed the
    line, so the next decision - raise the cap, shard the index, narrow the
    selection - is made on numbers instead of on a second 90-minute run.
    """


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

    # EvalItem is QUOTED: this signature is evaluated at import time, and
    # eval_sets imports this module. A bare annotation here would make the
    # two import each other.
    def __init__(self, items: Iterable["EvalItem"], *, n: int = NGRAM,
                 short_min: int = SHORT_MIN_TOKENS, max_grams: int | None = None):
        self.n = n
        self.short_min = short_min
        # Grams are the index's size in the only unit that predicts its
        # footprint, and they are counted per set because a pooled total
        # cannot say which set to shard.
        self.total_grams = 0
        self.grams_by_set: dict[str, int] = {}
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
            self.total_grams += len(grams)
            self.grams_by_set[item.set_key] = (
                self.grams_by_set.get(item.set_key, 0) + len(grams)
            )
            if max_grams is not None and self.total_grams > max_grams:
                # Checked as it grows, not at the end: the point is to beat
                # the allocation that would have killed the process.
                raise EvalIndexTooLarge(
                    f"the exact eval index passed {max_grams:,} grams while reading "
                    f"{item.set_key!r} (item {ix + 1}): {self.total_grams:,} grams, "
                    f"~{self.total_grams * INDEX_BYTES_PER_GRAM / 1024 ** 3:.1f} GiB "
                    f"estimated at {INDEX_BYTES_PER_GRAM} bytes/gram.\n"
                    f"  per set so far: "
                    f"{', '.join(f'{k}={v:,}' for k, v in sorted(self.grams_by_set.items()))}\n"
                    f"  raise --max-index-grams if the machine has the memory, or shard "
                    f"the index by eval set. Screening nothing is not an option here: "
                    f"the whole module exists to make that impossible."
                )
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

    @property
    def bytes_estimate(self) -> int:
        """Estimated resident size of the postings. See INDEX_BYTES_PER_GRAM.

        An ESTIMATE, and named one: it is the measured per-gram cost times the
        grams actually indexed, which is an upper bound for real text and
        excludes the item text held beside it.
        """
        return self.total_grams * INDEX_BYTES_PER_GRAM

    def query(self, toks: Sequence[str]) -> dict[int, frozenset[int]]:
        """The row's grams at every window this index holds.

        Built once per row and handed to both `candidates` and the Jaccard
        diagnostic, so a row is never gramed twice at the same window.

        The crc32 pass is hoisted out of the loop: the token hashes do not
        depend on the window, and this comprehension runs once per window per
        row - it was the module's hottest line, re-encoding every token of
        every row about a dozen times to produce identical numbers.
        """
        hs = token_hashes(toks)
        return {window: grams_from(hs, window) for window in self.by_gram}

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
                # BESIDE `form`, not instead of it. `form` is the question
                # form (row_form prefers task_type), which is what dedupe's
                # prompt rule groups by; `source` is what shape.py's retention
                # table is keyed on, and for a generated row the two differ -
                # `irac_analysis` vs `synthesis`. One of them has to be here
                # or the retention of the generated streams cannot be measured
                # from the artifacts at all.
                "source": row_prov(item.row).get("source") or "",
                "hits": [
                    {"level": h.level, "eval_set": h.eval_set, "item_id": h.item_id, **h.detail}
                    for h in found
                ],
            }
        )
    stats["kept"] = len(kept)
    return kept, drops, stats


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
    # THE LICENCE TRAVELS WITH THE GROUNDING. A generated row is the model's
    # words over one seed's text, and it is that seed's source whose terms
    # the dataset card has to state - the same stamp replay.assembly_row
    # puts on every curated and replay row. This builds its 12-key _prov as a
    # literal rather than calling that 4-key builder and merging eight more
    # keys onto it: the merge would be more code, not less.
    # Omitting it (until 2026-08-29) left every synthesis row
    # licence-less, which stats' `license` gate refuses: "the dataset card
    # cannot be written over them". Read once; there are a handful of
    # sources and tens of thousands of rows.
    licenses = store.source_licenses()
    for gen in store.latest_generations(state):
        seed = store.get_seed(gen["seed_id"]) or {}
        think, answer = gen.get("think") or "", gen.get("answer") or ""
        answer, preamble_dropped = answer_without_preamble(answer, gen.get("task_type"))
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
                "license": licenses.get(seed.get("source_id")),
                "native_id": seed.get("native_id"),
                "cnr": seed.get("cnr"),
                "neutral_citation": seed.get("neutral_citation"),
                "task_type": gen.get("task_type"),
                "prompt_id": gen.get("prompt_id"),
                # THE TEACHER TRAVELS WITH THE ROW, for the same reason the
                # licence does: the store holds it and the file chain drops
                # it, so by the time push.py writes the manifest there is no
                # other channel left that can say which model wrote which row.
                # `prompt_sha` beside it because "which teacher" and "which
                # version of the question" are one provenance fact, not two -
                # verify.py demotes on either. Spelled provider/model, the
                # way config.py spells a routing pool entry.
                "teacher": f"{gen.get('provider')}/{gen.get('model')}",
                "prompt_sha": gen.get("prompt_sha"),
                "seed_id": gen.get("seed_id"),
                "gen_id": gen.get("gen_id"),
                "score": round(sum(scores) / len(scores), 3) if scores else None,
                "reasoning": bool(think),
                # How many characters of second deliberation were cut
                # off the front of this answer. 0 means it opened with
                # its structure, which is what the templates ask for; a
                # non-zero value makes the trim auditable against the
                # raw generation the store still holds.
                "answer_preamble_dropped": preamble_dropped,
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

    `eval_items` IS PER SCRIPT AND DOES NOT SUM TO THE CORPUS. A code-switched
    eval question is indexed in every script it carries a screenable share of,
    so one item is counted under two of them - the question the field answers
    is "how many eval items could THIS script's screen have found", and that is
    two different ones. `residue_items` beside it is the partitions this layer
    refused to index because they were a residue of their item rather than a
    share of it (see SCRIPT_RESIDUE_MIN_SHARE); those are findable by nobody
    and are counted so the decision is readable rather than silent.

    `screened` MEANS ONE THING HERE: the control passed AND an index was built
    AND it holds at least one eval item. Two definitions used to be in play and
    the manifest's was the weaker of them, so a script with zero eval items read
    `screened: true` - a screen that cannot find anything, recorded as one that
    can. Without a layer at all (the control itself failed, or semhash is not
    installed) each script still gets its block, with `index: false`: an empty
    `semantic_scripts` object was the same shape for "not installed", "no eval
    items" and "the control failed", which is three facts wearing one silence.
    """
    # Deferred on purpose: eval_sets and semantic both import this module's
    # primitives, so importing them at module scope here would be circular.
    # This module is the base layer; they sit above it.
    from tuned.data.semantic import SCRIPT_NONE

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


def manifest_of(stats: dict, corpora: "dict[str, EvalCorpus]", index: EvalIndex, *,
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
    # Deferred on purpose: eval_sets and semantic both import this module's
    # primitives, so importing them at module scope here would be circular.
    # This module is the base layer; they sit above it.
    from tuned.data.eval_sets import EVAL_COUNTS_VERIFIED_AT
    from tuned.data.semantic import (
        SEMANTIC_PROBE_STRIDE,
        SEMANTIC_PROBE_WORDS,
        SEMANTIC_THRESHOLD,
    )

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
                # WHAT THIS SET COSTS THE INDEX. The eval surface is selected
                # by split NAME across whole configs, so an upstream
                # re-release can multiply one set without anything here
                # changing - and until these were written down, the only
                # symptom would have been exit 137.
                "index_grams": index.grams_by_set.get(key, 0),
                "index_bytes_estimate": (
                    index.grams_by_set.get(key, 0) * INDEX_BYTES_PER_GRAM
                ),
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
        # The total the operator sets --max-index-grams from, beside the
        # constant the estimate is built on so a later reader can re-derive it
        # rather than trust it.
        "eval_index": {
            "grams": index.total_grams,
            "bytes_estimate": index.bytes_estimate,
            "bytes_per_gram": INDEX_BYTES_PER_GRAM,
        },
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

def main(argv: Sequence[str] | None = None, *, reader=None) -> int:
    # Deferred on purpose: eval_sets and semantic both import this module's
    # primitives, so importing them at module scope here would be circular.
    # This module is the base layer; they sit above it.
    from tuned.data.eval_sets import (
        EVAL_COUNTS_VERIFIED_AT,
        EVAL_SETS,
        REFUSAL_HEADER,
        _EVAL_TEXT_FIELDS,
        eval_corpora,
        read_rows,
        refusals,
    )
    from tuned.data.semantic import (
        SEMANTIC_NO_ITEMS,
        SEMANTIC_NO_MODEL,
        SEMANTIC_NO_SCREENABLE_ITEMS,
        SEMANTIC_RAN,
        SEMANTIC_UNAVAILABLE,
        SEMANTIC_UNUSABLE,
        SemanticFilter,
        SemanticModelError,
        SemanticSeamError,
        screened_scripts,
        semantic_controls,
        semhash_available,
    )

    import argparse

    from tuned.data.config import load_build_config
    from tuned.data.jsonl import write_jsonl
    from tuned.data.paths import build_paths
    from tuned.data.store import TASK_STATES, Store

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
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
    parser.add_argument(
        "--max-index-grams", type=int, default=None, metavar="N",
        help=(
            "refuse rather than build an exact index larger than N grams. "
            "OFF by default: no measured peak for this chain exists yet, and a "
            "guessed ceiling that fires on today's eval surface would block the "
            "ship path for a number nobody has read. Set it from the "
            "`eval_index` block this run writes into decontamination.json."
        ),
    )
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
        corpora = eval_corpora(
            store, allow_missing=args.allow_missing_eval, reader=reader or read_rows,
            corpus_dir=paths.corpus_dir,
        )
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

        try:
            index = EvalIndex(
                [item for c in corpora.values() for item in c.items],
                max_grams=args.max_index_grams,
            )
        except EvalIndexTooLarge as exc:
            print(f"decontaminate REFUSES TO RUN: {exc}\n  nothing was written.")
            return 2
        print(
            f"eval index: {index.total_grams:,} grams, "
            f"~{index.bytes_estimate / 1024 ** 3:.2f} GiB estimated"
            + (f" (cap {args.max_index_grams:,})" if args.max_index_grams else "")
        )
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
