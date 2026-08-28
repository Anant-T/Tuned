"""Contract tests for the prompt templates and prompt_registry.

Two things are under test and they are different in kind.

The registry is ordinary code: load/render/variants/pick_variant, tested the
way any loader is tested.

The TEMPLATES are the dataset's design written down, and the tests below are
the only mechanism that keeps a design decision from being edited away by
accident. Each property here was paid for by evidence recorded in the plan:
scripted/structured think traces cost -33.8% quality AND increase fabrication
(MSLR), so no template may hand the teacher a think-block outline; a trace
that reasons about "the provided text" teaches the student to hallucinate a
passage it will never be given at inference, so every template must forbid
that register WITHOUT naming the strings gates.check_banned_meta matches (see
test_generator_never_enumerates_the_banned_meta_phrases - the enumeration was
measured to CAUSE the failure it was meant to prevent); a trace that never
doubts itself teaches confident hallucination, so every template must ask for
a self-verification move AND hand over the vocabulary gates.VERIFICATION_CUES
actually recognises; and IRAC belongs to the answer, never to the trace, which
is what gates.check_irac_placement enforces from the other side.

The golden-sha test makes a prompt edit deliberate rather than incidental:
every generation ever made carries its prompt_sha (store.task.prompt_sha), so
an unnoticed prompt change silently makes two runs incomparable.
"""

import difflib
import hashlib
import itertools
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from tuned.data import prompt_registry as reg
from tuned.data.gates import (
    BANNED_META,
    INSTRUCTION_ECHO_SPANS,
    VERIFICATION_CUES,
    _norm_ws,
)

# --------------------------------------------------------------------------
# Golden hashes. A failure here is not a bug report, it is a question: did you
# mean to change this prompt? If yes, regenerate the block and say so in the
# commit message - every pilot metric computed under the old sha describes a
# different prompt.
#
#   .venv/Scripts/python.exe -c "from tuned.data.prompt_registry import all_ids, load; [print(f'    {i!r}: {load(i).sha!r},') for i in all_ids()]"
# --------------------------------------------------------------------------
#
# RE-PINNED 2026-08-18 (pilot gate-rejection fix round). Every gen_* template
# changed: the banned-meta enumeration was replaced by a descriptive rule, the
# self-verification cue vocabulary was handed over explicitly, the trace-length
# floor was raised off think_min, and the IRAC prohibition was sharpened to the
# line-initial shape the gate actually matches. No metric computed under the
# old shas describes these prompts.
#
# RE-PINNED 2026-08-27 (generator reasoning ceiling), then REVERTED
# 2026-08-28. The 2026-08-27 edit changed every gen_* template on one line -
# the clause licensing the reasoning to run as long as it liked was deleted,
# and the 450-700 band was given an explicit upper stop - and re-pinned the
# shas below to match. It is proven harmful to gpt-oss (paired A/B, 4
# pre-registered fails,
# docs/reports/2026-08-27-gptoss-floor-under-the-prompt-ceiling.md) and
# at-best-wash for deepseek (clean rerun,
# docs/reports/2026-08-28-deepseek-prompt-era-rerun.md: +4.97pp for pre-edit,
# full-gate clean 16.5% vs 8.5%, 17% cheaper per passing row) - helping
# nobody on any measurement, so the operator reverted it. All fourteen
# gen_* templates are back to their pre-2026-08-27 bytes, so the shas below
# are the 2026-08-18 pins restored, not a new pin.
#
# RE-PINNED 2026-08-28 (anti-rehearsal clause shipped). The three-arm
# clause/cap A/B (docs/reports/2026-08-28-deepseek-clause-and-cap-ab.md)
# measured the clause's own irac_placement target FAIL (-4.46pp vs a >=15pp
# bar) but a strong showing on everything else - length_band +16.05pp
# (42.20% -> 58.25%), think p50 -18.7%, full-gate clean 11.0% -> 14.6%, no
# metric worse - and the operator shipped it on that totality as a
# think-length lever, not an irac-placement fix. Six templates changed by
# exactly one clause each, inserted after "never opens a line with one of
# those four words" and before the word-count sentence:
# gen_irac_analysis_v1-v4 and gen_summarization_v1-v2. The other eight
# gen_* and all three non-gen templates are untouched; their shas below are
# unchanged from the 2026-08-18/2026-08-18-restored pins.
EXPECTED_SHAS = {
    'gen_drafting_v1': '48534e3010f5',
    'gen_drafting_v2': '618b240ab03e',
    'gen_irac_analysis_v1': 'f2b4a76489cb',
    'gen_irac_analysis_v2': 'a5e62bd4bb3f',
    'gen_irac_analysis_v3': 'c4922e9d298c',
    'gen_irac_analysis_v4': '78f0e8944ae1',
    'gen_statute_qa_v1': '94e43b22bf48',
    'gen_statute_qa_v2': '4d04338ba007',
    'gen_statute_qa_v3': '5888a6c4461d',
    'gen_statute_qa_v4': '713a9060835e',
    'gen_summarization_v1': '52fdcf8dbd04',
    'gen_summarization_v2': '3b9eefc64d33',
    'gen_transition_v1': '113813116cfb',
    'gen_transition_v2': '2f28a53e5259',
    'judge_pointwise_v1': 'cd552205602e',
    'judge_tiebreak_v1': 'a34456f4918b',
    'probe_answer_v1': '8370e47920ee',
}

GEN_IDS = tuple(i for i in reg.all_ids() if i.startswith("gen_"))
JUDGE_IDS = ("judge_pointwise_v1", "judge_tiebreak_v1")

# The variant budget from the brief: the two task types that carry most of the
# mix get four paraphrases, the rest get two.
EXPECTED_VARIANT_COUNTS = {
    "irac_analysis": 4,
    "statute_qa": 4,
    "drafting": 2,
    "summarization": 2,
    "transition": 2,
}

# Per-task-type slot contract. The task layer fills these; a template that
# grows or loses one silently would make render() raise (or quietly ignore a
# value) at generation time, mid-wave.
COMMON_SLOTS = {"source", "question"}
EXPECTED_SLOTS = {
    "irac_analysis": COMMON_SLOTS | {"focus_issue"},
    "statute_qa": COMMON_SLOTS | {"section_text"},
    "drafting": COMMON_SLOTS | {"document_kind", "party_context"},
    "summarization": set(COMMON_SLOTS),
    "transition": COMMON_SLOTS
    | {"old_section_text", "new_section_text", "savings_text", "scenario"},
}

JUDGE_SLOTS = {"source", "candidate_think", "candidate_answer"}

# Harness vocabulary. A template that tells the teacher to "rewrite the
# passage" has already lost: the student is never shown a passage, so the
# trace it learns from must not be about one. These words are permitted ONLY
# inside a prohibition - see _prohibition_line.
HARNESS_VERBS = (
    "rewrite",
    "paraphrase",
    "the passage",
    "the document",
    "based on the provided text",
)
PROHIBITION_MARKERS = ("never", "do not", "avoid")

GROUNDING_SENTENCE = (
    "You must not rely on any statutory provision, case name, or authority "
    "that does not appear in the materials above."
)

# The anti-MSLR clause, standardised across every generator template so it can
# be asserted rather than eyeballed.
IRAC_PLACEMENT_CLAUSE = "never inside your reasoning"

# The SHAPE half of the same clause, added 2026-08-18. Naming the channel was
# not enough: 38 pilot rows carried all four headings in the answer AND in the
# trace, so gates.check_irac_placement rejected a correct answer for what the
# reasoning looked like. gates._IRAC_HEADING_RE matches a LINE-INITIAL heading
# word, so that is the shape the template now prohibits by name.
IRAC_LINE_START_CLAUSE = "never opens a line with one of those four words"

ANSWER_LENGTH_CLAUSE = "250 to 450 words"

# The soft floor on the trace. gates.check_length_band enforces think_min=500
# ESTIMATED tokens on every reasoning row, so a well-behaved teacher that
# answers a short seed in three crisp sentences fails the band and burns a
# regeneration. Behavioural, never structural: it says think properly, not
# think in N parts.
#
# RAISED 2026-08-18 off "a few hundred words". The band is in chars//4, so
# think_min=500 is 2,000 characters; "a few hundred words" is ~300 words ~1,900
# chars ~475 est tokens - the instruction targeted a trace BELOW the floor it
# had to clear. Measured: every one of the 20 attempt-1 gpt-oss length_band
# violations was `think<think_min`, on a measured attempt-1 mean of 2,411 chars
# (603 est tok) sitting barely over it. 450-700 words is ~2,900-4,500 chars,
# ~725-1,125 est tokens: clear of think_min 500 and clear of think_max 3,000.
# The band VALUES are untouched - this moves the instruction onto the band, not
# the band onto the instruction.
#
# CEILING ADDED 2026-08-27, REVERTED 2026-08-28. The 2026-08-27 edit deleted
# the licence clause from all fourteen templates and trimmed this constant to
# "450 to 700 words of deliberation" to match. It is proven harmful to
# gpt-oss (docs/reports/2026-08-27-gptoss-floor-under-the-prompt-ceiling.md,
# 4 pre-registered fails) and at-best-wash for deepseek
# (docs/reports/2026-08-28-deepseek-prompt-era-rerun.md: +4.97pp for the
# pre-edit prompts, clean full-gate 16.5% vs 8.5%, 17% cheaper per passing
# row) - so the operator reverted it. The licence clause is back in every
# template and this constant is back to its full 2026-08-18 form.
REASONING_FLOOR_CLAUSE = "450 to 700 words of deliberation is normal"

# A "defuser" is the clause that stops an instruction from reading as a
# script. Two families, defending different failure modes, and both are
# accepted-phrasing lists rather than free text so the property cannot drift
# back out of a template during an edit.
#
#   ORDER: attaches to a multi-item enumeration in the reasoning instruction
#   ("what relief is sought, which provision founds it, ...") and says the
#   items are not a sequence to walk. Without it the enumeration IS a think
#   outline, which is the MSLR failure the whole design avoids.
#
#   RITUAL: attaches to the self-check and says the double-check is a move,
#   not a section - otherwise the teacher appends a tidy "Verification"
#   paragraph, which is a scripted trace wearing a cue's clothing.
ORDER_DEFUSERS = ("in whatever order", "in no fixed order", "as they arise")
RITUAL_DEFUSERS = (
    "not a heading",
    "not as a heading",
    "never a heading",
    "never as a heading",
    "not a ritual",
    "not as a ritual",
    "never as a ritual",
    "as a closing ritual",
    "as a closing formality",
    "not as a set-piece",
    "not a section of it",
    "not a part of the advice",
    "saving it for the end",
)

# What an enumeration ITEM looks like: an interrogative opening a list item,
# i.e. one introduced by the colon/dash that opens the list or by the comma
# between items. Anchoring on that punctuation is what separates a checklist
# ("...: what relief is sought, which provision founds it, what has to be
# averred...") from ordinary prose that merely uses the same words ("any
# instinct about what must by now have replaced what"). Three items is the
# threshold; two reads as a sentence.
_ENUM_ITEM_RE = re.compile(
    r"[:,—–-]\s+(?:and\s+|or\s+|on\s+|by\s+|to\s+|for\s+|of\s+)?"
    r"(?:what|which|how|where)\b",
    re.IGNORECASE,
)
_ENUM_THRESHOLD = 3

# Numbered-step heuristic, deliberately simple and slightly over-broad: any
# line that OPENS with "Step 3", "3." or "3)" is treated as a think-block
# outline. Templates here are written as prose paragraphs and never number
# anything, so the false-positive risk is theoretical and the failure mode it
# guards (a helpfully-added "1. Identify the issue 2. State the rule" list) is
# exactly the MSLR failure.
_NUMBERED_OUTLINE_RE = re.compile(r"^[ \t]*(?:step[ \t]*)?\d+[.)]", re.IGNORECASE | re.MULTILINE)

# Anything a judge must not be told. It scores what is in front of it; a hint
# that a known-good outcome exists turns the rubric into a matching exercise.
JUDGE_LEAKS = ("correct answer", "gold", "expected outcome")

# The two prompt roots the grounding-band test below reads directly (not
# through the registry, since it inspects source text rather than rendered
# output).
PROMPTS = reg.PROMPTS_DIR
HARMONY = Path(__file__).parent.parent / "src" / "tuned" / "data" / "prompts_harmony"


def _slot_values(prompt_id: str) -> dict:
    """Inert dummies - no harness verb, no brace, no IRAC heading word - so a
    content assertion below can never be satisfied or broken by the filler."""
    return {name: f"[[{name}]]" for name in reg.slots(prompt_id)}


def _rendered(prompt_id: str) -> str:
    messages = reg.render(prompt_id, **_slot_values(prompt_id))
    return "\n".join(m["content"] for m in messages)


def _prohibition_line(line: str) -> bool:
    lowered = line.lower()
    return any(marker in lowered for marker in PROHIBITION_MARKERS)


def _template_text(prompt_id: str) -> str:
    template = reg.load(prompt_id)
    return f"{template.system or ''}\n{template.user}"


# --------------------------------------------------------------------------
# Golden hashes + file hygiene.
# --------------------------------------------------------------------------

def test_every_template_on_disk_is_pinned():
    assert set(reg.all_ids()) == set(EXPECTED_SHAS), (
        "templates on disk and EXPECTED_SHAS disagree - a template was added "
        "or removed without pinning it"
    )


@pytest.mark.parametrize("prompt_id", sorted(EXPECTED_SHAS))
def test_template_sha_is_pinned(prompt_id):
    assert reg.load(prompt_id).sha == EXPECTED_SHAS[prompt_id]


@pytest.mark.parametrize("prompt_id", sorted(EXPECTED_SHAS))
def test_sha_is_twelve_hex_of_the_raw_file(prompt_id):
    raw = (reg.PROMPTS_DIR / f"{prompt_id}.md").read_bytes()
    sha = reg.load(prompt_id).sha
    assert sha == hashlib.sha256(raw).hexdigest()[: reg.SHA_LEN]
    assert len(sha) == 12 and set(sha) <= set("0123456789abcdef")


@pytest.mark.parametrize("prompt_id", sorted(EXPECTED_SHAS))
def test_templates_are_lf_only(prompt_id):
    # The sha is over RAW bytes, so a CRLF checkout would change every pinned
    # hash. .gitattributes pins the repo to LF; this asserts it held.
    assert b"\r" not in (reg.PROMPTS_DIR / f"{prompt_id}.md").read_bytes()


@pytest.mark.parametrize("prompt_id", sorted(EXPECTED_SHAS))
def test_blocks_parse(prompt_id):
    template = reg.load(prompt_id)
    assert template.prompt_id == prompt_id
    assert template.user.strip()
    assert template.system is None or template.system.strip()
    assert reg.SYSTEM_MARK not in template.user
    assert reg.USER_MARK not in template.user


# --------------------------------------------------------------------------
# Generator templates: the properties the evidence bought.
# --------------------------------------------------------------------------

def test_variant_budget():
    assert dict(
        (task_type, len(reg.variants(task_type))) for task_type in reg.task_types()
    ) == EXPECTED_VARIANT_COUNTS


@pytest.mark.parametrize("task_type", sorted(EXPECTED_SLOTS))
def test_every_variant_of_a_task_type_takes_the_same_slots(task_type):
    for prompt_id in reg.variants(task_type):
        assert set(reg.slots(prompt_id)) == EXPECTED_SLOTS[task_type], prompt_id


@pytest.mark.parametrize("prompt_id", GEN_IDS)
def test_generator_renders_with_no_unfilled_slot(prompt_id):
    rendered = _rendered(prompt_id)
    # Generator templates carry no literal braces at all, so any brace left in
    # the rendered text is an unfilled (or mis-escaped) slot.
    assert "{" not in rendered and "}" not in rendered
    for name in reg.slots(prompt_id):
        assert f"[[{name}]]" in rendered


@pytest.mark.parametrize("prompt_id", GEN_IDS)
def test_generator_adopts_a_role_and_puts_the_materials_before_the_ask(prompt_id):
    template = reg.load(prompt_id)
    assert "You are" in template.user
    # "the materials above" is only true if the chunk is above it.
    assert template.user.index("{source}") < template.user.index("{question}")
    assert template.user.index("{source}") < template.user.index(GROUNDING_SENTENCE)


@pytest.mark.parametrize("prompt_id", GEN_IDS)
def test_generator_never_enumerates_the_banned_meta_phrases(prompt_id):
    """A template may NOT name the strings gates.check_banned_meta matches.

    INVERTED 2026-08-18, and the inversion is the whole finding. This test used
    to REQUIRE >=6 of BANNED_META in every template, on the theory that naming
    what is forbidden forbids it best. Measured over the pilot's 221 gated
    generations it does the opposite: the teacher restates the rule in its
    private trace ("We must not use the phrase \"the source says\"") and
    check_banned_meta - which matches those literals against the trace - fires
    on a row that is OBEYING the instruction. 115/221 failures, and the hit
    histogram tracks the templates' own list order ('the source says' 89,
    'the provided text' 58, 'the excerpt' 23, ... 'in the given text' 9), which
    is the tell that these are echoes and not native harness leakage. Attempt-1
    rate 11/60; attempts 2-3, where the trace is 5x longer and has more room
    for compliance narration, 51/59 and 53/58.

    The register is still forbidden - test_generator_uses_harness_verbs_only_
    inside_a_prohibition and the "in your own words" test both still bind - but
    it is now forbidden by description. BANNED_META stays what it always should
    have been: a private checklist the teacher never reads.
    """
    lowered = _rendered(prompt_id).lower()
    present = [phrase for phrase in BANNED_META if phrase in lowered]
    assert not present, (
        f"{prompt_id} names {present} verbatim - the teacher will echo them "
        f"into its trace and check_banned_meta will reject a compliant row"
    )


@pytest.mark.parametrize("prompt_id", GEN_IDS)
def test_generator_asks_for_first_person_deliberation(prompt_id):
    lowered = _rendered(prompt_id).lower()
    assert "first person" in lowered and "present tense" in lowered
    # Genuine uncertainty, not a hedge: the word has to appear somewhere.
    assert any(word in lowered for word in ("uncertain", "doubt", "unsure", "arguable"))


@pytest.mark.parametrize("prompt_id", GEN_IDS)
def test_generator_uses_harness_verbs_only_inside_a_prohibition(prompt_id):
    offenders = [
        (verb, line)
        for line in _rendered(prompt_id).splitlines()
        for verb in HARNESS_VERBS
        if verb in line.lower() and not _prohibition_line(line)
    ]
    assert not offenders, offenders


@pytest.mark.parametrize("prompt_id", GEN_IDS)
def test_generator_states_the_grounding_constraint(prompt_id):
    rendered = _rendered(prompt_id)
    assert GROUNDING_SENTENCE in rendered
    assert "make explicit" in rendered


@pytest.mark.parametrize("prompt_id", GEN_IDS)
def test_generator_self_check_is_behavioural_and_gate_visible(prompt_id):
    """The template must hand over the cue VOCABULARY, not just ask for the move.

    STRENGTHENED 2026-08-18 from ">=1 cue" to ">=4". gates.VERIFICATION_CUES is
    a closed twelve-phrase list and check_self_verification passes only on a
    literal match, but the templates described the behaviour ("double-check the
    part that carries the weight") and never said which words buy the pass.
    Measured attempt-1 failure rate 51/60 = 85%, the worst single-attempt gate
    in the pilot; of the rows that DID pass, the cue was 'double-check' 39
    times - i.e. the gate was passing on templates' own word rather than on any
    verification the teacher chose. Traces that failed were verifying
    themselves in words the list does not carry ("that judgment is not binding
    on Supreme Court").

    THE ASYMMETRY WITH test_generator_never_enumerates_the_banned_meta_phrases
    IS DELIBERATE and is why one test now forbids naming gate strings while
    this one requires it. check_banned_meta fires on PRESENCE, so a teacher
    that quotes the rule back fails it; check_self_verification fires on
    ABSENCE, so a teacher that quotes these back SATISFIES it. Echoing is fatal
    in one direction and harmless in the other. The ritual defuser (asserted
    separately) is what keeps the offered vocabulary from becoming a set-piece.
    """
    # MIRROR THE GATE EXACTLY. check_self_verification does
    # `_norm_ws(think).lower()` and tests `cue.lower() in text`, so BOTH sides
    # are lowered there. This matcher lowered only the haystack, which made
    # 'am I sure' - the one cue carrying a capital - impossible to match, and
    # the count it reported was therefore one short of the truth on every
    # template. Whitespace is normalised for the same reason: a cue broken
    # across a line wrap counts for the gate and must count here.
    text = _norm_ws(_rendered(prompt_id)).lower()
    cues = [cue for cue in VERIFICATION_CUES if cue.lower() in text]
    assert len(cues) >= 5, (
        f"{prompt_id} offers only {cues} of gates.VERIFICATION_CUES - the "
        f"teacher is being asked to self-verify without being told which "
        f"words check_self_verification can see"
    )


def test_the_cue_matcher_here_mirrors_the_gate():
    """The bug this pins is a CASE mismatch, and it hid a real shortfall.

    gates.check_self_verification lowers BOTH sides - `_norm_ws(think).lower()`
    then `cue.lower() in text`. The matcher above lowered only the haystack,
    so 'am I sure', the one cue in VERIFICATION_CUES carrying a capital, could
    never match and every template was reported one cue short of the truth.
    With the shortfall hidden, the >=4 contract had no slack at all.
    """
    capitalised = [cue for cue in VERIFICATION_CUES if cue != cue.lower()]
    assert capitalised, "if no cue carries a capital this test guards nothing"

    probe = "I paused there. Am I sure of this reading?"
    # What the gate sees.
    assert [c for c in VERIFICATION_CUES if c.lower() in _norm_ws(probe).lower()]
    # What a matcher that lowers only one side sees: nothing.
    assert [c for c in capitalised if c in probe.lower()] == []


# An inventory of short quoted phrases inside a REASONING instruction is the
# enumeration shape the MSLR evidence forbids - a list the teacher fills in
# rather than a way of thinking. _ENUM_ITEM_RE cannot see it (it scores
# enumerations of INTERROGATIVES), so this is the guard for the other shape.
_QUOTED_SPAN_RE = re.compile(r'"[^"\n]{1,40}"')
_MAX_QUOTED_SPANS = 2


@pytest.mark.parametrize("prompt_id", GEN_IDS)
def test_generator_hands_over_the_cues_as_prose_not_an_inventory(prompt_id):
    """ADDED 2026-08-18 (round 2). The round-1 fix handed the self-check
    vocabulary over as a quoted list of five phrases - the right vocabulary in
    the wrong shape, and each template carried ten quote characters because of
    it. The cues now arrive as speech spread over two sentences, which the gate
    matches identically and which reads as thinking rather than as a checklist.
    """
    spans = _QUOTED_SPAN_RE.findall(_rendered(prompt_id))
    assert len(spans) <= _MAX_QUOTED_SPANS, (
        f"{prompt_id} carries {len(spans)} short quoted spans {spans[:6]} - "
        f"that is an inventory, not prose"
    )


@pytest.mark.parametrize("prompt_id", GEN_IDS)
def test_generator_states_the_irac_answer_contract(prompt_id):
    rendered = _rendered(prompt_id)
    for heading in ("Issue", "Rule", "Application", "Conclusion"):
        assert heading in rendered, heading
    # IRAC in the answer only - the other half of gates.check_irac_placement.
    assert IRAC_PLACEMENT_CLAUSE in rendered
    # ...and in the line-initial shape the gate's regex actually matches.
    assert IRAC_LINE_START_CLAUSE in rendered
    assert ANSWER_LENGTH_CLAUSE in rendered
    assert REASONING_FLOOR_CLAUSE in rendered


def test_every_instruction_echo_span_still_occurs_in_a_live_template():
    """A stale echo span fails SILENTLY, so the suite has to hold it.

    gates.INSTRUCTION_ECHO_SPANS is matched against GENERATED traces to catch
    a teacher parroting its own instructions back. Nothing matches it against
    the prompts, so a span whose wording has been edited out of every template
    can never fire: check_prompt_echo simply stops detecting that class, with
    no error and a green suite. The 2026-08-27 ceiling edit was exactly how
    that happens - it deleted " is normal" from all fourteen templates, and
    the span read "450 to 700 words of deliberation is normal" (that edit was
    reverted 2026-08-28; see EXPECTED_SHAS above for why).

    check_prompt_echo compares against _norm_ws(think).lower(), so both sides
    are normalised the same way here.

    The contract is "at least one live template", deliberately not "every
    template". Coverage today is uneven and legitimately so:

        14 - never write as though the matter had been handed to you as a text
        14 - 450 to 700 words of deliberation is normal
        14 - let me check this, or actually, that does not follow, ...
         2 - those headings belong to the answer and never inside your reasoning
         1 - work it out before you commit to anything (gen_irac_analysis_v1)

    A span that matches NOTHING is a bug in the span or a prompt edit that was
    not carried through. It is never a reason to delete the span or to loosen
    this test.
    """
    live = {i: _norm_ws(reg.load(i).user).lower() for i in reg.all_ids()}
    dead = sorted(
        span
        for span in INSTRUCTION_ECHO_SPANS
        if not any(span in user for user in live.values())
    )
    assert not dead, (
        f"instruction-echo spans matching no live template: {dead}. "
        f"check_prompt_echo can never fire on these - repoint the span at the "
        f"wording the templates actually use now."
    )


def _reasoning_paragraphs(prompt_id: str) -> list[str]:
    """Every paragraph EXCEPT the answer contract.

    The answer contract is excluded because its four-part structure is the one
    structure this design wants: IRAC in the answer is mandatory, so its
    "under the first ..., under the third ..." enumeration is a feature. It is
    identified by the placement clause it must already carry, which every
    generator is separately asserted to have."""
    return [
        block
        for block in reg.load(prompt_id).user.split("\n\n")
        if IRAC_PLACEMENT_CLAUSE not in block
    ]


@pytest.mark.parametrize("prompt_id", GEN_IDS)
def test_generator_self_check_carries_a_ritual_defuser(prompt_id):
    lowered = _rendered(prompt_id).lower()
    assert any(phrase in lowered for phrase in RITUAL_DEFUSERS), (
        f"{prompt_id} asks for a double-check without saying it is a move rather "
        f"than a section - that is how a cue becomes a scripted heading"
    )


@pytest.mark.parametrize("prompt_id", GEN_IDS)
def test_generator_enumeration_carries_an_order_defuser(prompt_id):
    """A multi-item enumeration inside a REASONING instruction must say it is
    not a sequence. Paragraphs without such an enumeration are unaffected."""
    for paragraph in _reasoning_paragraphs(prompt_id):
        items = len(_ENUM_ITEM_RE.findall(paragraph))
        if items < _ENUM_THRESHOLD:
            continue
        assert any(phrase in paragraph.lower() for phrase in ORDER_DEFUSERS), (
            f"{prompt_id} lists {items} things to work through and never says "
            f"the order is not fixed - that reads as a think outline: "
            f"{paragraph[:120]!r}"
        )


@pytest.mark.parametrize("prompt_id", GEN_IDS)
def test_generator_forbids_verbatim_copying_into_the_trace(prompt_id):
    lowered = _rendered(prompt_id).lower()
    assert "in your own words" in lowered


@pytest.mark.parametrize("prompt_id", GEN_IDS)
def test_generator_gives_no_numbered_step_outline(prompt_id):
    hits = [m.group(0) for m in _NUMBERED_OUTLINE_RE.finditer(_rendered(prompt_id))]
    assert not hits, f"{prompt_id} numbers something: {hits}"
    assert not re.search(r"\bstep\s*\d", _rendered(prompt_id), re.IGNORECASE)


def _word_count(text: str) -> int:
    """Words, not whitespace-separated tokens: the templates set their clauses
    off with spaced em dashes, which str.split would count as words."""
    return sum(1 for token in text.split() if any(ch.isalnum() for ch in token))


@pytest.mark.parametrize("prompt_id", GEN_IDS)
def test_generator_user_block_is_the_intended_size(prompt_id):
    # Long enough to carry every property above, short enough that the
    # instruction does not swamp the chunk it wraps. That rationale used to be
    # stated as "the teacher's context is 8k on Cerebras"; the 2026-08-19 probe
    # put it at 131k, so the ceiling now rests on READABILITY and on not
    # drowning the chunk, not on a window. The target is the
    # brief's ~250-450; the ceiling carries a small tolerance so that adding
    # one clause to the longest template is not a forced rewrite.
    #
    # 470 -> 500 on 2026-08-18: the fix round added four clauses to every
    # template at once (the cue vocabulary being the expensive one), which the
    # old tolerance was never sized for, and the two longest templates sat 2
    # and 5 words under it. The context rationale is unaffected at this size -
    # measured pilot prompt_est ran 1,445-2,799 est tokens and the net change
    # is +22 words (~30 tokens), against a probed 131k window whose cliff sits
    # at 104,858 routing tokens - two orders of magnitude clear. generate.py
    # still routes a seed away when it does not fit; nothing this build
    # produces gets near it.
    #
    # 500 -> 600 on 2026-08-28: the anti-rehearsal clause shipped into six
    # templates (see EXPECTED_SHAS above) adds ~85 words each, and five of
    # the six landed over the old 500-word ceiling (gen_irac_analysis_v1/v2/
    # v3 and gen_summarization_v1/v2; v4 landed exactly at 500). Measured
    # post-ship max is gen_summarization_v1 at 574 words. The context
    # rationale is unaffected at this size for the same reason as the prior
    # bump - still two orders of magnitude clear of the routing cliff.
    words = _word_count(reg.load(prompt_id).user)
    assert 250 <= words <= 600, f"{prompt_id} user block is {words} words"


@pytest.mark.parametrize("task_type", sorted(EXPECTED_VARIANT_COUNTS))
def test_variants_are_real_paraphrases(task_type):
    """A one-word swap is not a paraphrase. The standardised clauses (the
    grounding sentence, the self-check vocabulary, the IRAC answer contract)
    are the only text variants may hold in common; persona, framing and order
    must differ.

    The banned-meta ENUMERATION used to be named here as one of those shared
    blocks. It no longer exists - 2026-08-18 replaced it with a descriptive
    rule - and removing it is what dropped max pairwise similarity from 0.42
    to 0.30.
    """
    for a, b in itertools.combinations(reg.variants(task_type), 2):
        ratio = difflib.SequenceMatcher(None, _template_text(a), _template_text(b)).ratio()
        assert ratio < 0.8, f"{a} vs {b} are {ratio:.2f} similar"


# --------------------------------------------------------------------------
# Judge templates.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("prompt_id", JUDGE_IDS)
def test_judge_sees_the_source_and_both_halves_of_the_candidate(prompt_id):
    assert set(reg.slots(prompt_id)) == JUDGE_SLOTS
    rendered = _rendered(prompt_id)
    for name in JUDGE_SLOTS:
        assert f"[[{name}]]" in rendered


@pytest.mark.parametrize("prompt_id", JUDGE_IDS)
def test_judge_scores_exactly_the_three_axes(prompt_id):
    rendered = _rendered(prompt_id)
    for axis in ("grounding_faithfulness", "reasoning_validity", "issue_coverage"):
        assert axis in rendered, axis
    assert "1 to 5" in rendered


@pytest.mark.parametrize("prompt_id", JUDGE_IDS)
def test_judge_output_contract_is_strict_json(prompt_id):
    """The exemplar the judge is asked to imitate must itself parse. A judge
    shown `{"grounding": n}` is being shown invalid JSON and asked to emit
    valid JSON, and the worker parses what comes back with json.loads."""
    rendered = _rendered(prompt_id)
    assert "JSON" in rendered
    # Escaped in the file as {{...}}; a single brace pair after rendering.
    assert "{{" not in rendered
    line = next(line for line in rendered.splitlines() if '"grounding"' in line).strip()
    payload = json.loads(line)

    assert set(payload) == {"grounding", "validity", "coverage", "rationale"}
    for axis in ("grounding", "validity", "coverage"):
        assert isinstance(payload[axis], int) and 1 <= payload[axis] <= 5, axis
    assert isinstance(payload["rationale"], str) and payload["rationale"].strip()
    assert len(payload["rationale"].split()) <= 80
    assert "80 words" in rendered

    # A concrete exemplar anchors; mixed values plus an explicit disclaimer are
    # what keep it a shape rather than a suggested score.
    assert len({payload["grounding"], payload["validity"], payload["coverage"]}) > 1
    assert "illustration" in rendered.lower()


@pytest.mark.parametrize("prompt_id", JUDGE_IDS)
def test_judge_is_never_told_an_outcome(prompt_id):
    lowered = _rendered(prompt_id).lower()
    for leak in JUDGE_LEAKS:
        assert leak not in lowered, leak


@pytest.mark.parametrize("prompt_id", JUDGE_IDS)
def test_judge_does_not_score_style_or_length(prompt_id):
    lowered = _rendered(prompt_id).lower()
    assert "verbosity" in lowered
    assert "not your axes" in lowered
    # Hesitation and self-correction are what the dataset is FOR; a judge that
    # marks them down would select exactly the confident traces we reject.
    assert "deliberation" in lowered


def test_tiebreak_arbitrates_blind():
    system = reg.load("judge_tiebreak_v1").system.lower()
    assert "disagreement" in system or "disagreed" in system
    assert "not shown" in system
    rendered = _rendered("judge_tiebreak_v1").lower()
    assert "split the difference" in rendered


def test_grounding_bands_are_mutually_exclusive():
    """Band 2 must not claim the territory band 3 describes.

    Band 2 read "provision, case or rule that is not in the materials" while
    band 3 read "at least one proposition of substance rests on nothing
    given" — the same event. Band 2 is the only band that hard-fails
    (judge_policy.FAIL_MAX = 2), so the collision funnelled 41 of 101
    judgements into a failing band. Band 2 must now require a MISSTATEMENT,
    not mere absence.
    """
    for root in (PROMPTS, HARMONY):
        text = (root / "judge_pointwise_v1.md").read_text(encoding="utf-8")
        line = next(ln for ln in text.splitlines() if ln.startswith("grounding_faithfulness"))
        band2 = line.split("2:")[1].split("1:")[0]
        assert "misstates" in band2 or "contradict" in band2, (
            "band 2 must turn on misstatement, not absence"
        )
        assert "or rule that is not in the materials" not in band2, (
            "band 2 still swallows band 3's territory"
        )


# --------------------------------------------------------------------------
# Probe template.
# --------------------------------------------------------------------------

def test_probe_is_minimal():
    assert set(reg.slots("probe_answer_v1")) == {"question"}
    template = reg.load("probe_answer_v1")
    assert len(template.user.split()) < 80
    messages = reg.render("probe_answer_v1", question="Is X an offence?")
    assert messages[-1]["content"].startswith("Is X an offence?")


# --------------------------------------------------------------------------
# Registry API.
# --------------------------------------------------------------------------

def test_render_returns_system_then_user():
    messages = reg.render("gen_irac_analysis_v1", **_slot_values("gen_irac_analysis_v1"))
    assert [m["role"] for m in messages] == ["system", "user"]
    assert all(m["content"].strip() for m in messages)


def test_render_raises_on_a_missing_slot():
    with pytest.raises(KeyError) as exc:
        reg.render("gen_irac_analysis_v1", source="s", question="q")
    assert "focus_issue" in str(exc.value)


def test_render_ignores_extra_slots():
    # The task layer hands the same context dict to templates that use
    # different subsets of it.
    messages = reg.render(
        "gen_summarization_v1", source="s", question="q", focus_issue="unused"
    )
    assert "unused" not in messages[-1]["content"]


def test_load_unknown_id_raises_keyerror_listing_known_ids():
    with pytest.raises(KeyError) as exc:
        reg.load("gen_nonexistent_v9")
    message = str(exc.value)
    assert "gen_irac_analysis_v1" in message and "judge_pointwise_v1" in message


def test_variants_unknown_task_type_raises_keyerror_listing_known_types():
    with pytest.raises(KeyError) as exc:
        reg.variants("appellate_brief")
    assert "irac_analysis" in str(exc.value)


def test_variants_are_ordered_by_version_number():
    assert reg.variants("irac_analysis") == (
        "gen_irac_analysis_v1",
        "gen_irac_analysis_v2",
        "gen_irac_analysis_v3",
        "gen_irac_analysis_v4",
    )


def test_all_ids_is_sorted_and_covers_the_generators():
    ids = reg.all_ids()
    assert list(ids) == sorted(ids)
    assert set(GEN_IDS) <= set(ids)


# --------------------------------------------------------------------------
# pick_variant.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("task_type", sorted(EXPECTED_VARIANT_COUNTS))
def test_pick_variant_is_deterministic_and_matches_the_documented_hash(task_type):
    pool = reg.variants(task_type)
    for seed_id in ("seed-a", "0f3c9a11", "L-NLProc/PredEx:42"):
        for sample_ix in range(4):
            picked = reg.pick_variant(task_type, seed_id, sample_ix)
            assert picked == reg.pick_variant(task_type, seed_id, sample_ix)
            digest = hashlib.sha256(f"{seed_id}:{sample_ix}".encode("utf-8")).hexdigest()
            assert picked == pool[int(digest, 16) % len(pool)]


@pytest.mark.parametrize("task_type", sorted(EXPECTED_VARIANT_COUNTS))
def test_pick_variant_uses_the_whole_pool(task_type):
    counts = Counter(
        reg.pick_variant(task_type, f"seed{i:05d}", i % 4) for i in range(2000)
    )
    assert set(counts) == set(reg.variants(task_type))
    # Not a distribution test, just a smell test: nothing may be starved.
    assert min(counts.values()) > 2000 // (4 * len(reg.variants(task_type)))


def test_pick_variant_separates_samples_of_one_seed():
    """3-4 samples per seed is the OpenThoughts resample strategy; if every
    sample of a seed drew the same paraphrase the resampling would be
    measuring temperature alone."""
    picks = {
        reg.pick_variant("irac_analysis", "one-seed", ix) for ix in range(8)
    }
    assert len(picks) > 1


# --------------------------------------------------------------------------
# Recovery overlay isolation. Live bytes stay the control.
# --------------------------------------------------------------------------

_OVERLAY_DIR = Path(__file__).parent.parent / "src" / "tuned" / "data" / "prompts_harmony"


def test_live_prompt_files_are_untouched_when_the_overlay_is_armed():
    live_before = {
        prompt_id: (reg.PROMPTS_DIR / f"{prompt_id}.md").read_bytes()
        for prompt_id in EXPECTED_SHAS
    }
    try:
        reg.set_overlay(_OVERLAY_DIR)
        overlaid = reg.load("gen_irac_analysis_v1")
        assert overlaid.sha != EXPECTED_SHAS["gen_irac_analysis_v1"]
        live_after = {
            prompt_id: (reg.PROMPTS_DIR / f"{prompt_id}.md").read_bytes()
            for prompt_id in EXPECTED_SHAS
        }
        assert live_before == live_after
    finally:
        reg.set_overlay(None)
    assert reg.load("gen_irac_analysis_v1").sha == EXPECTED_SHAS["gen_irac_analysis_v1"]


def test_live_judge_sha_stays_put_when_recovery_overlay_is_armed():
    live_bytes = {
        prompt_id: (reg.PROMPTS_DIR / f"{prompt_id}.md").read_bytes()
        for prompt_id in JUDGE_IDS
    }
    try:
        reg.set_overlay(_OVERLAY_DIR)
        for prompt_id in JUDGE_IDS:
            overlaid = reg.load(prompt_id)
            assert overlaid.sha != EXPECTED_SHAS[prompt_id]
            assert (reg.PROMPTS_DIR / f"{prompt_id}.md").read_bytes() == live_bytes[prompt_id]
    finally:
        reg.set_overlay(None)
    for prompt_id in JUDGE_IDS:
        assert reg.load(prompt_id).sha == EXPECTED_SHAS[prompt_id]


# Overlay drift. An ACCIDENTAL base edit is already caught above by
# test_template_sha_is_pinned - the pin stops matching and forces a look.
# What these tests catch is narrower: a DELIBERATE base edit, where the
# author moves the sha pin on purpose and forgets to carry the same change
# into the overlay copy. The overlay is sixteen near-copies of prompts/ and
# must stay copies: prompt_sha is sha256 of RAW FILE BYTES, so a generated
# overlay would have no bytes to hash and the exp_harmony rows would stop
# being comparable.

_PACKET_MARKERS = ("450 to 700 words", "Let me check this, or actually")

# RE-PINNED 2026-08-24 (task 1, judge calibration): judge_tiebreak_v1 overlay
# had its worked exemplar (validity 2) restored - the earlier removal is what
# saturated the tiebreak arbiter to 18/18 accepts. judge_pointwise_v1 is
# unaffected and keeps dropping its example verdict.
_EXPECTED_OVERLAY_SHAS = {
    "gen_drafting_v1": "609834efa759",
    "gen_drafting_v2": "ea66b7bba577",
    "gen_irac_analysis_v1": "088c0442f674",
    "gen_irac_analysis_v2": "5fa4ce5dba19",
    "gen_irac_analysis_v3": "d3635ee18266",
    "gen_irac_analysis_v4": "5bc40d3c1bef",
    "gen_statute_qa_v1": "bf49860e80dc",
    "gen_statute_qa_v2": "aaaaec660f01",
    "gen_statute_qa_v3": "9d2859618af8",
    "gen_statute_qa_v4": "598deaeafd23",
    "gen_summarization_v1": "42ee72ab542c",
    "gen_summarization_v2": "84e8a00c5425",
    "gen_transition_v1": "717e0c99aea7",
    "gen_transition_v2": "73a97936afa1",
    "judge_pointwise_v1": "dec02ad95f7b",
    "judge_tiebreak_v1": "09100c3f704f",
}


def test_every_overlay_file_has_a_pinned_sha():
    on_disk = {path.stem for path in _OVERLAY_DIR.glob("*.md")}
    assert on_disk == set(_EXPECTED_OVERLAY_SHAS)


def test_every_base_gen_prompt_has_an_overlay_counterpart():
    # test_every_overlay_file_has_a_pinned_sha (above) only walks the overlay
    # side: it would catch a stray addition to prompts_harmony/, but not a
    # stray addition to prompts/. And prompt_registry._template_path() falls
    # back to the base file whenever the overlay has none for that id -
    # silently, no error, no test failure at the registry level. So a new
    # gen_*.md dropped into prompts/ with no overlay counterpart would ship
    # rendered under exp_harmony with the recovery-packet markers still in
    # it, and every drift test in this section (all keyed off
    # _EXPECTED_OVERLAY_SHAS) would keep passing regardless. This pins the
    # base and overlay gen_* id sets equal so that gap cannot open unnoticed.
    base_gen_ids = {p.stem for p in reg.PROMPTS_DIR.glob("gen_*.md")}
    overlay_gen_ids = {p for p in _EXPECTED_OVERLAY_SHAS if p.startswith("gen_")}
    assert base_gen_ids == overlay_gen_ids


def test_overlay_bytes_are_pinned_like_the_live_bytes_are():
    for prompt_id, expected in sorted(_EXPECTED_OVERLAY_SHAS.items()):
        raw = (_OVERLAY_DIR / f"{prompt_id}.md").read_bytes()
        actual = hashlib.sha256(raw).hexdigest()[:12]
        assert actual == expected, (
            f"{prompt_id} overlay changed: {actual} != {expected}. An overlay "
            f"edit is deliberate or it is drift - move the pin on purpose."
        )


def test_generator_overlays_differ_from_their_base_in_exactly_two_lines():
    ids = [p for p in _EXPECTED_OVERLAY_SHAS if p.startswith("gen_")]
    assert len(ids) == 14
    for prompt_id in sorted(ids):
        base = (reg.PROMPTS_DIR / f"{prompt_id}.md").read_text(encoding="utf-8").splitlines()
        over = (_OVERLAY_DIR / f"{prompt_id}.md").read_text(encoding="utf-8").splitlines()
        assert len(base) == len(over), f"{prompt_id}: overlay changed line count"
        changed = [i for i, (b, o) in enumerate(zip(base, over)) if b != o]
        assert len(changed) == 2, (
            f"{prompt_id}: {len(changed)} lines differ from the base, expected 2 "
            f"(the two packet-marker lines). This only counts changed lines - it "
            f"does not by itself prove the packet markers are gone; "
            f"test_the_packet_is_in_every_base_and_in_no_generator_overlay checks "
            f"that."
        )


def test_the_packet_is_in_every_base_and_in_no_generator_overlay():
    ids = [p for p in _EXPECTED_OVERLAY_SHAS if p.startswith("gen_")]
    for prompt_id in sorted(ids):
        base = (reg.PROMPTS_DIR / f"{prompt_id}.md").read_text(encoding="utf-8")
        over = (_OVERLAY_DIR / f"{prompt_id}.md").read_text(encoding="utf-8")
        for marker in _PACKET_MARKERS:
            assert marker in base, f"{prompt_id}: base lost {marker!r}"
            assert marker not in over, (
                f"{prompt_id}: overlay carries {marker!r} - stripping the packet "
                f"is the whole reason this overlay exists"
            )


def test_tiebreak_templates_carry_a_low_score_anchor():
    """Both tiebreak overlays must show the model a failing exemplar.

    Removing the worked example that contained `"validity": 2` saturated
    mistral-large to 18/18 accepts at validity 5.00, against 2.75 for the
    same model in the same seat on the frozen store. An arbiter that has
    never seen a low score does not produce one.
    """
    for root in (reg.PROMPTS_DIR, _OVERLAY_DIR):
        text = (root / "judge_tiebreak_v1.md").read_text(encoding="utf-8")
        scores = [int(n) for n in re.findall(r'"(?:grounding|validity|coverage)":\s*(\d)', text)]
        assert scores, f"{root.name}: tiebreak template has no exemplar verdict at all"
        assert min(scores) <= 2, (
            f"{root.name}: lowest exemplar score is {min(scores)}; "
            "the arbiter needs a failing anchor or it saturates upward"
        )


def test_judge_overlays_drop_the_example_verdict():
    for prompt_id in ("judge_pointwise_v1",):
        base = (reg.PROMPTS_DIR / f"{prompt_id}.md").read_text(encoding="utf-8")
        over = (_OVERLAY_DIR / f"{prompt_id}.md").read_text(encoding="utf-8")
        assert '"grounding": 4' in base
        assert '"grounding": 4' not in over
        assert "no example verdict" in over
        assert len(over.splitlines()) == len(base.splitlines()) - 2
