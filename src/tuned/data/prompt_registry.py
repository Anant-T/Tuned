"""Loader for the teacher/judge prompt templates in `prompts/`.

The templates are the dataset's design, written down. Everything the pilot
learns about WHY a generation failed is only interpretable against the exact
bytes the teacher was shown, so a prompt is identified by two things and
both are recorded on every task row (store.task.prompt_id / prompt_sha):

  prompt_id   the file stem - "gen_irac_analysis_v2", "judge_pointwise_v1".
  sha         sha256 of the RAW FILE BYTES, first 12 hex. Editing a comma in
              a template gives every generation made afterwards a different
              sha, so two runs are never silently compared across a prompt
              change. tests/test_build_prompts.py pins every sha, which is
              what makes a prompt edit a deliberate, reviewed act.

Raw bytes, not normalized text: .gitattributes pins the whole repo to LF
("* text=auto eol=lf"), so a checkout on Windows and a checkout on Kaggle
hash identically. If that line ever goes, the golden-sha test fails loudly
on the next platform rather than drifting quietly.

File format - an optional system block, then a user block:

    <!-- system -->
    ...system message...
    <!-- user -->
    ...user message...

`{slot}` placeholders are str.format fields, so literal braces in a template
(the judge templates' JSON contract) are escaped as `{{`/`}}`. render()
fills them and returns the OpenAI-style message list providers.py sends.

Variant selection is deterministic, never random: pick_variant hashes
(seed_id, sample_ix). Re-planning a wave after a crash therefore re-derives
the SAME prompt for the same task, so a task_id that was already generated
under one paraphrase is never quietly re-run under another - the resumed row
stays comparable with the one it replaces. Difficulty and area
randomization belong to the task layer; templates carry neither.

CONTRACTS THE CALLERS MUST HONOUR. None of these can be enforced from here,
and each one silently corrupts a different downstream number if it is missed:

  1. GROUNDING TEXT IS THE UNION OF EVERY GROUNDING SLOT. gates.GateContext
     .source_text - and the judge's own {source} slot - must be the
     concatenation of EVERY slot the generator was shown as material:
     {source} plus, as applicable, {section_text}, {old_section_text},
     {new_section_text}, {savings_text}. Pass {source} alone and a teacher
     that correctly cites the very section it was handed is scored as having
     invented it: check_citations sees a citation absent from source_text,
     which is a PERMANENT reject (the seed is burned, never retried). The
     statute_qa and transition streams are the ones this destroys, and they
     are exactly the streams whose whole point is citing the provision.

     ONE ASYMMETRY, ADDED DELIBERATELY (generate.judge_source_text): on the
     TRANSITION stream the judge's {source} also carries {scenario}, because
     which enactment governs is decided by the dates and a judge that cannot
     see them is scoring an axis it has no access to. GateContext.source_text
     does NOT get it: that string is the citation allow-list, and a scenario
     mentioning a section number would quietly authorise citing it. Grounding
     and judge-source are therefore the same string everywhere except there.
     Duplicate slot values are also collapsed - the pilot's statute_qa
     fallback fills {source} and {section_text} from the same seed text, and
     concatenating both would double every judge prompt for no added
     grounding.

  2. THE JUDGE PROMPT IS THE LONGEST PROMPT IN THE PIPELINE. It carries the
     same materials the generator saw PLUS the candidate's trace and answer,
     so it can be twice the generator's prompt. Two judge-pool models are
     8k-context (cerebras/zai-glm-4.7, cerebras/gemma-4-31b); the judge
     worker must budget-check materials+candidate before dispatch and route
     anything that does not fit to a 32k+ judge (groq/qwen/qwen3.6-27b,
     mistral/mistral-small-latest). Silently truncating a judge prompt scores
     a partially-read answer, which is worse than not judging it.

  3. THE TRANSITION SLOT SPLIT IS SEMANTIC. {source} is the papers (facts,
     parties, what has happened); {scenario} is the posture and the DATES -
     when the offence is alleged, when the proceeding started, what stage it
     has reached. The templates label both, but nothing here enforces the
     split: put the dates only in {source} and the prompts still read
     correctly while the "which date decides it" instruction loses its
     anchor, and check_temporal's undecidable channel (fatal on this stream)
     is what will report it, one wave later.
"""

import hashlib
import re
import string
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SYSTEM_MARK = "<!-- system -->"
USER_MARK = "<!-- user -->"

SHA_LEN = 12

# gen_<task_type>_v<k>. The task type is whatever sits between the prefix and
# the version suffix, so adding a task type means adding files, never editing
# this module.
_GEN_RE = re.compile(r"^gen_(?P<task_type>[a-z0-9_]+?)_v(?P<version>\d+)$")


def _prompts_dir() -> Path:
    """Same resources idiom as statutes.py: importlib.resources first, so an
    installed wheel works, with a filesystem fallback for loaders that cannot
    hand back a real path."""
    try:
        from importlib.resources import files

        return Path(str(files("tuned.data").joinpath("prompts")))
    except Exception:  # pragma: no cover - non-filesystem loader fallback
        return Path(__file__).resolve().parent / "prompts"


PROMPTS_DIR = _prompts_dir()


@dataclass(frozen=True)
class Template:
    prompt_id: str
    system: str | None
    user: str
    sha: str


def _split_blocks(text: str, prompt_id: str) -> tuple[str | None, str]:
    """(system, user). The user block is mandatory; the system block is not.

    Anything before the first marker is a hard error rather than a silently
    dropped preamble: a template whose first line drifted above the marker
    would otherwise lose that instruction with no diagnostic at all.
    """
    user_at = text.find(USER_MARK)
    if user_at == -1:
        raise ValueError(f"template {prompt_id!r} has no {USER_MARK} block")

    system_at = text.find(SYSTEM_MARK)
    if system_at == -1:
        head = text[:user_at]
        if head.strip():
            raise ValueError(
                f"template {prompt_id!r} has text before its {USER_MARK} marker "
                f"but no {SYSTEM_MARK} block: {head.strip()[:60]!r}"
            )
        return None, text[user_at + len(USER_MARK) :].strip()

    if system_at > user_at:
        raise ValueError(
            f"template {prompt_id!r} puts {SYSTEM_MARK} after {USER_MARK}"
        )
    head = text[:system_at]
    if head.strip():
        raise ValueError(
            f"template {prompt_id!r} has text before its {SYSTEM_MARK} marker: "
            f"{head.strip()[:60]!r}"
        )
    system = text[system_at + len(SYSTEM_MARK) : user_at].strip()
    user = text[user_at + len(USER_MARK) :].strip()
    if not system:
        raise ValueError(f"template {prompt_id!r} has an empty {SYSTEM_MARK} block")
    if not user:
        raise ValueError(f"template {prompt_id!r} has an empty {USER_MARK} block")
    return system, user


@lru_cache(maxsize=1)
def all_ids() -> tuple[str, ...]:
    """Every template id on disk, sorted. Cached: the directory is package
    data, fixed for the life of the process."""
    return tuple(sorted(path.stem for path in PROMPTS_DIR.glob("*.md")))


@lru_cache(maxsize=None)
def load(prompt_id: str) -> Template:
    path = PROMPTS_DIR / f"{prompt_id}.md"
    try:
        raw = path.read_bytes()
    except OSError:
        raise KeyError(
            f"no prompt template {prompt_id!r} in {PROMPTS_DIR}; known ids: "
            f"{', '.join(all_ids())}"
        ) from None
    system, user = _split_blocks(raw.decode("utf-8"), prompt_id)
    sha = hashlib.sha256(raw).hexdigest()[:SHA_LEN]
    return Template(prompt_id=prompt_id, system=system, user=user, sha=sha)


def slots(prompt_id: str) -> frozenset[str]:
    """The `{slot}` names a template requires, across both blocks."""
    template = load(prompt_id)
    text = f"{template.system or ''}\n{template.user}"
    return frozenset(
        name for _, name, _, _ in string.Formatter().parse(text) if name
    )


def render(prompt_id: str, **slot_values) -> list[dict]:
    """The message list for `prompt_id` with its slots filled.

    A missing slot raises KeyError naming every one that is missing: a
    half-filled prompt would otherwise reach a teacher with a literal
    "{focus_issue}" in it and burn real tokens on a malformed task. Extra
    keyword arguments are ignored - the task layer may hand the same context
    dict to templates that use different subsets of it.
    """
    template = load(prompt_id)
    required = slots(prompt_id)
    missing = sorted(required - set(slot_values))
    if missing:
        raise KeyError(
            f"template {prompt_id!r} needs slot(s) {missing}; "
            f"it takes {sorted(required)}"
        )
    fill = {name: slot_values[name] for name in required}

    messages: list[dict] = []
    if template.system is not None:
        messages.append({"role": "system", "content": template.system.format(**fill)})
    messages.append({"role": "user", "content": template.user.format(**fill)})
    return messages


@lru_cache(maxsize=1)
def _variants_by_task() -> dict[str, tuple[str, ...]]:
    """task_type -> its generator ids, ordered by version NUMBER.

    Numeric order, not lexicographic: with ten variants "gen_x_v10" sorts
    before "gen_x_v2" as a string, which would silently re-map every
    pick_variant answer the day a tenth paraphrase is added.
    """
    found: dict[str, list[tuple[int, str]]] = {}
    for prompt_id in all_ids():
        match = _GEN_RE.match(prompt_id)
        if match:
            found.setdefault(match.group("task_type"), []).append(
                (int(match.group("version")), prompt_id)
            )
    return {
        task_type: tuple(pid for _, pid in sorted(pairs))
        for task_type, pairs in sorted(found.items())
    }


def task_types() -> tuple[str, ...]:
    return tuple(_variants_by_task())


def variants(task_type: str) -> tuple[str, ...]:
    try:
        return _variants_by_task()[task_type]
    except KeyError:
        raise KeyError(
            f"no generator templates for task type {task_type!r}; known types: "
            f"{', '.join(task_types())}"
        ) from None


def pick_variant(task_type: str, seed_id: str, sample_ix: int) -> str:
    """Which paraphrase this (seed, sample) draws. Deterministic and stable
    across processes - a plain hash() would not be, PYTHONHASHSEED randomizes
    str hashing per run, and a wave replanned tomorrow must reproduce today's
    assignment exactly.

    The whole 256-bit digest feeds the modulo (not a truncated prefix) so the
    spread stays even for any variant count.
    """
    pool = variants(task_type)
    digest = hashlib.sha256(f"{seed_id}:{sample_ix}".encode("utf-8")).hexdigest()
    return pool[int(digest, 16) % len(pool)]
