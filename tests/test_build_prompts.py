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
passage it will never be given at inference, so every template must enumerate
the banned meta-phrases gates.check_banned_meta rejects; a trace that never
doubts itself teaches confident hallucination, so every template must ask for
a self-verification move in words that gates.VERIFICATION_CUES can see; and
IRAC belongs to the answer, never to the trace, which is what
gates.check_irac_placement enforces from the other side.

The golden-sha test makes a prompt edit deliberate rather than incidental:
every generation ever made carries its prompt_sha (store.task.prompt_sha), so
an unnoticed prompt change silently makes two runs incomparable.
"""

import difflib
import hashlib
import itertools
import re
from collections import Counter

import pytest

from tuned.data import prompt_registry as reg
from tuned.data.gates import BANNED_META, VERIFICATION_CUES

# --------------------------------------------------------------------------
# Golden hashes. A failure here is not a bug report, it is a question: did you
# mean to change this prompt? If yes, regenerate the block and say so in the
# commit message - every pilot metric computed under the old sha describes a
# different prompt.
#
#   .venv/Scripts/python.exe -c "from tuned.data.prompt_registry import all_ids, load; [print(f'    {i!r}: {load(i).sha!r},') for i in all_ids()]"
# --------------------------------------------------------------------------
EXPECTED_SHAS = {
    'gen_drafting_v1': '5553e7e82c7d',
    'gen_drafting_v2': '827e1ab90466',
    'gen_irac_analysis_v1': '143a35a01d3f',
    'gen_irac_analysis_v2': 'cb8461054db0',
    'gen_irac_analysis_v3': '72ab826bc18e',
    'gen_irac_analysis_v4': '6245b3c46c63',
    'gen_statute_qa_v1': '014706e18c12',
    'gen_statute_qa_v2': 'b3dcabe4740a',
    'gen_statute_qa_v3': '9c63591e0f26',
    'gen_statute_qa_v4': '6a4c1b6d06af',
    'gen_summarization_v1': '2b39b9000d03',
    'gen_summarization_v2': '8353d0e11c5b',
    'gen_transition_v1': '0792acf9f34f',
    'gen_transition_v2': '8daf00a7f855',
    'judge_pointwise_v1': '3825edfc4d4a',
    'judge_tiebreak_v1': 'b539ef67f22d',
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

ANSWER_LENGTH_CLAUSE = "250 to 450 words"

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
def test_generator_enumerates_the_banned_meta_phrases(prompt_id):
    lowered = _rendered(prompt_id).lower()
    present = [phrase for phrase in BANNED_META if phrase in lowered]
    assert len(present) >= 6, f"{prompt_id} enumerates only {present}"


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
    lowered = _rendered(prompt_id).lower()
    cues = [cue for cue in VERIFICATION_CUES if cue in lowered]
    assert cues, (
        f"{prompt_id} asks for no self-verification move in words "
        f"gates.check_self_verification can recognise"
    )


@pytest.mark.parametrize("prompt_id", GEN_IDS)
def test_generator_states_the_irac_answer_contract(prompt_id):
    rendered = _rendered(prompt_id)
    for heading in ("Issue", "Rule", "Application", "Conclusion"):
        assert heading in rendered, heading
    # IRAC in the answer only - the other half of gates.check_irac_placement.
    assert IRAC_PLACEMENT_CLAUSE in rendered
    assert ANSWER_LENGTH_CLAUSE in rendered


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
    # instruction does not swamp the chunk it wraps (the teacher's context is
    # 8k on Cerebras and the chunk has to fit beside it). The target is the
    # brief's ~250-450; the ceiling carries a small tolerance so that adding
    # one clause to the longest template is not a forced rewrite.
    words = _word_count(reg.load(prompt_id).user)
    assert 250 <= words <= 470, f"{prompt_id} user block is {words} words"


@pytest.mark.parametrize("task_type", sorted(EXPECTED_VARIANT_COUNTS))
def test_variants_are_real_paraphrases(task_type):
    """A one-word swap is not a paraphrase. The shared blocks (the banned-meta
    enumeration, the grounding sentence) are the only text variants may hold
    in common; persona, framing and order must differ."""
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
    rendered = _rendered(prompt_id)
    assert "JSON" in rendered
    line = next(line for line in rendered.splitlines() if '"grounding"' in line)
    for key in ('"grounding"', '"validity"', '"coverage"', '"rationale"'):
        assert key in line, key
    # Escaped in the file as {{...}}; a single brace pair after rendering.
    assert line.strip().startswith("{") and line.strip().endswith("}")
    assert "{{" not in rendered
    assert "80 words" in rendered


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
