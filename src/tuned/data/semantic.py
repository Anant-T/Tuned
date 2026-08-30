"""The optional semantic layer of the decontamination pass.

Lifted verbatim out of decontaminate.py on 2026-08-30. It is optional by
design and recorded either way, it shares nothing with the exact n-gram
stack except Item and Hit, and it is a third of the module it used to live
in - so reading or changing the n-gram rules meant paging through 1,162
lines that have no bearing on them.

It is also the code with the widest gap between how much is written and how
much has run: the only full-chain manifest on disk records
`semantic: semhash-not-installed`, so CI is the first machine to execute it.
That is an argument for isolating it, not for treating it as dead.

The dependency runs one way on purpose: this module imports decontaminate's
primitives and eval_sets' EvalItem; decontaminate imports this one only
inside the functions that need it (manifest_of, semantic_script_record,
main). Reversing that is a circular import.
"""

import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from tuned.data.decontaminate import (
    LEVEL_SEMANTIC,
    SHORT_MIN_TOKENS,
    Hit,
    Item,
    tokens,
)
from tuned.data.eval_sets import EvalItem


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
    the probe routes to the right index. THE DILUTION TABLE, measured
    cache-only against the shipped model on this repo's own five reworded
    leaks, each at the worst alignment its geometry admits, with k Devanagari
    words interleaved and the window routed WHOLE - best window score per k:

        k (Hindi words)     0      1      2      3      4
        best window       0.915  0.899  0.860  0.731  0.713

    Three interleaved words is where a leak stops clearing 0.8. (An earlier
    version of this table read 0.948/0.893/0.729/0.706 from a fixture no longer
    in the repo and did not reproduce; these numbers come from
    test_the_dilution_cosine_table_reproduces, which re-runs them.) The same
    dilution kept an ENGLISH eval question quoted verbatim inside a
    Hindi-dominant row, five times out of five, while the row's Devanagari half
    was recorded as the only unscreened thing about it - the screen said
    `latin: screened, unscreened rows 0` over rows whose English leak it had
    just missed.

    Splitting the probe collapses that, and the collapse DEPENDS ON PLACEMENT,
    which neither the claim this replaces nor its correction said. Measured
    over the five verbatim leaks at four placements each: at arbitrary offsets
    all 20 score exactly 1.000 against the Latin index once the Devanagari
    words are not in the probe text; at the WORST alignments the geometry
    admits, 10 of the 20 are 1.000 and the rest run down to 0.9487, because a
    question straddling a window boundary is only ever held in part. Every one
    of the 40 is caught at 0.8 either way. (The ledger has now been wrong about
    this cell twice - first "the same five score 1.000", then "19 of 20, the
    twentieth 0.9487". Neither reproduces; re-run by
    test_a_verbatim_leak_in_a_hindi_row_scores_by_where_it_sits.)

    The same five reworded leaks that a plain layer misses at three interleaved
    Hindi words are all caught.

    IT IS NOT CHEAPER, and the ledger used to say it was. The Latin query count
    per row is UNCHANGED, plus or minus one - measured on a 50/50 Hinglish row,
    29 Latin queries at 300 words before the split and 29 after, 27 and 27 at
    275, 9 and 9 at 100. What the split ADDS is one Devanagari partition per
    window (28, 26, 8), counted as unscreened and never queried. Re-run by
    test_splitting_the_probe_does_not_change_the_query_count.

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
        probe = provenance_text(getattr(record, "record", None))
        for pair in getattr(record, "duplicates", None) or ():
            try:
                text, score = provenance_text(pair[0]), float(pair[1])
            except (TypeError, ValueError, IndexError):
                continue
            if text is None:
                continue
            if best_score is None or (score, text) > (best_score, best_text or ""):
                best_text, best_score, best_probe = text, score, probe
    return best_text, best_score, best_probe


def provenance_text(value) -> str | None:
    """A semhash record as a string, whichever of its two shapes it arrived in.

    PUBLIC because dedupe.py reads the same two shapes off the same library.
    It used to coerce with `str(record)` instead, which stringifies a dict to
    `"{'text': ...}"` - so under the dict shape every `budget[text]` read 0 and
    every row in the corpus flagged as a duplicate of itself, in a function
    whose docstring cites the fail-loud rule as its protection.

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
        #
        # THIS COUNTS WINDOWS, and until `route` gained its no-partition branch
        # it could not: `SCRIPT_NONE` was reachable only from `probes[0]`, so a
        # LETTERED row's letterless windows were counted nowhere and the
        # counter was structurally 1 or 0 - a number whose own comment said the
        # opposite of what it measured, and an assertion of `> 0` that could
        # not separate 1 from 9. A 174-word Latin row with a run of section
        # numbers through the middle of it has nine.
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
            # Word-level comparison, not string membership: the short-row split
            # rebuilds probes[0] as " ".join(words), so a raw text with a
            # newline differs from its own whole-row probe by whitespace alone
            # and a membership test misreads every pipeline-built short row as
            # a residue.
            detail["probe_residue"] = bool(
                probe is not None
                and all(probe.split() != p.split() for p in probe_texts(text))
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


