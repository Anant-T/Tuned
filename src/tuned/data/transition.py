"""The s.358 transition stream - statute-grounded cells with deterministic keys.

This is the only stream in the dataset whose right answer is known BEFORE the
teacher is asked. Every other stream is graded by gates and judges on how the
answer reads; a transition cell is graded against an answer key computed here,
out of statutes.py's decision table, at the moment the cell is built. That is
what makes it the dataset's transition-accuracy eval as well as one of its
training streams (see RESERVE below), and it is why nothing in this module may
ever ask a model for anything.

THE GRID
--------
One cell per

    verified mapping row  x  date posture  x  procedural posture  x  question form

and the four factors are owned in four different places on purpose. There are
TWO boundaries in the dates, not one - see JUDICIAL_INVALIDATIONS: the
appointed day decides which ENACTMENT governs, and a judgment date decides
whether the section named on the old side of it was in force at all. A key
that answered the first question without asking the second demanded that a
struck-down section be named as the section the charge lies under, which made
"no charge lies" - the correct answer - a permanent reject.

The four factors:

* the FAMILIES are resources/ipc_bns_map.jsonl, through Mapping.require_verified
  and nothing else. The 17 rows whose `verified_by` is still null cannot emit a
  single cell, and they are not skipped silently - build_grid records each one
  in the manifest with the refusal statutes.py itself raised. When the operator
  signs a row off the grid grows and no code changes. A row whose audited NOTE
  records a judicial event this module has no dated constant for is refused the
  same way, for the same reason: the build cannot say whether the section was
  in force on the day of the conduct, so it does not guess;
* the DATE POSTURES and PROCEDURAL POSTURES are here, because they are the two
  axes the two savings rules turn on and their intersection is the whole point
  of the stream;
* the QUESTION FORMS are here for the same reason, and they are what makes the
  answer key's teeth fair - see `charge_only`;
* the FACT SKELETONS are files under transition_templates/, versioned and
  content-hashed exactly like prompts/, so a skeleton edit gives every cell
  built afterwards a different sha and two runs are never silently compared
  across it.

NO MODEL-GENERATED FACTS, ANYWHERE. The papers are a template with the dates
filled in; the provisions are the mapping's own audited section numbers and
marginal notes; the savings blocks are resources/transition_provisions.jsonl.
A fact skeleton is deliberately abstract about the conduct ("the conduct
described in the provisions set out below") because 154 verified rows include
general clauses as well as offences, and a template that narrated an offence
would be narrating the wrong thing for those.

WHAT THE PROMPTS ACTUALLY CARRY, AND WHAT THEY DO NOT
-----------------------------------------------------
This repository holds no bare-act corpus. The only statute resources are the
IPC->BNS mapping (section numbers, kinds, marginal notes) and the three
repeal-and-savings provisions in transition_provisions.jsonl, which record each
provision's OPERATIVE EFFECT and say on every row that they are not
quotations. So a provision block here identifies its section and gives its
marginal note; it never claims to reproduce the section's words, and the
question slot tells the teacher not to quote words it has not been shown. The
plan asked for "the statute texts" and the honest state is one step short of
that; the fix is a bare-act corpus (P7), and transition_provisions.jsonl reads
`text_kind` so it can carry the real text the day one exists.

SECTION NUMBERS LIVE IN THE GROUNDING SLOTS ONLY. The papers and the posture
carry dates and stages and no section number at all. That is not tidiness:
{source} and the provision slots are the citation allow-list (see
generate.grounding_text), while {scenario} deliberately is NOT, so a section
named in the posture would be a section the answer may cite without ever
having been shown it. A test asserts extract_sections() finds nothing in any
rendered posture.

THE ANSWER KEY
--------------
gates.check_answer_key is the consumer and this module conforms to it rather
than forking it: `expected_sections`, `forbidden_sections`,
`requires_savings_mention`, `must_name_both_families` and `governing_family`
are exactly the keys it reads. Everything else in the dict
(`families_by_kind`, `procedural_rule`, `savings_consequence`, the grid
coordinates) is metadata for the eval and the report, which that gate ignores.

Three things about the teeth are worth stating because all three were got
wrong first:

1. `forbidden_sections` is populated on ONE question form, `charge_only`, and
   only when a counterpart exists. The other three forms invite the answer to
   contrast the two provisions - "IPC s.302 governs, not BNS s.103" is the
   BEST answer on this stream - and a forbidden entry would make that
   contrast a PERMANENT reject. charge_only earns its teeth by asking, in the
   question itself, for the governing provision and not the corresponding one.

2. `requires_savings_mention` is set whenever the OLD family governs the
   charge, and that is not a style rule: it is the same condition
   statutes._cites_savings_clause uses to suppress a cross-code flag. A
   correct answer on a pre-appointed-day cell has to cite BNS s.358 or BNSS
   s.531 or BSA s.170 - all new-family sections - and without the savings
   mention check_temporal would reject the correct answer. The two gates
   agree because the key makes them agree.

3. `requires_no_liability_statement` is the only field in the key that reads
   the answer's WORDS rather than its citations, and it is set only where the
   section was void when the conduct occurred. It has to exist: on those cells
   the right answer and the wrong answer cite the same section - "no charge
   lies under IPC s.497" and "the charge lies under IPC s.497" are the same
   citation set - so a key made of citations alone could express only the
   false one, and did.

THE RESERVE
-----------
`transition.eval_reserve` cells are taken from the FRONT of one deterministic
order and the `transition.sample` training cells from immediately behind them,
so the two sets are disjoint by construction rather than by a filter somebody
could forget. Reserved cells are written to the store as seeds carrying
meta_json.held_out = true, and tasks._candidate_seeds refuses a held-out seed
the same way it refuses an oversize one - so the eval can never be planned as
a teacher generation. Both halves are asserted.

Build:  python -m tuned.data.transition --config configs/data_law_v1.yaml
        [--dry-run]
"""

import hashlib
import re
from dataclasses import dataclass
from datetime import date, timedelta
from math import gcd
from pathlib import Path

from tuned.data.jsonl import read_jsonl
from tuned.data.statutes import (
    APPOINTED_DAY,
    CODE_KIND,
    OLD_CODES,
    RESOURCES_DIR,
    Mapping,
    SectionRef,
    governing_family,
)

# The stream and task type these seeds are planned under - gates.
# TRANSITION_STREAM and tasks.TRANSITION_MIX's only key. Stated as literals
# here rather than imported so this module does not pull the gate layer in;
# a test pins all three together.
STREAM = "transition"
TASK_TYPE = "transition"

# source.source_id for every grid cell. Not an upstream dataset: these rows
# are constructed from the repo's own statute resources, which is what the
# license string records.
TRANSITION_SOURCE_ID = "tuned/law-v1-transition-grid"
TRANSITION_LICENSE = "repo-internal: constructed from src/tuned/data/resources"

CELL_ID_LEN = 16

TRANSITION_PROVISIONS_PATH = RESOURCES_DIR / "transition_provisions.jsonl"

# Statute short titles, for the provision blocks. The long forms are what
# statutes._CODE already matches ("Indian Penal Code" and the three Sanhita
# names are in its alternation), so a rendered block is parsed back into the
# same SectionRef the key names - which is what makes the answer-key gate
# and the citation allow-list agree about a block this module wrote.
CODE_TITLES = {
    "IPC": "Indian Penal Code, 1860",
    "CRPC": "Code of Criminal Procedure, 1973",
    "IEA": "Indian Evidence Act, 1872",
    "BNS": "Bharatiya Nyaya Sanhita, 2023",
    "BNSS": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "BSA": "Bharatiya Sakshya Adhiniyam, 2023",
}

_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


# --------------------------------------------------------------------------
# The fact-skeleton registry - prompt_registry's idiom, its own directory.
# --------------------------------------------------------------------------

PAPERS_MARK = "<!-- papers -->"
POSTURE_MARK = "<!-- posture -->"
SHA_LEN = 12


def _templates_dir() -> Path:
    """Same resources idiom as statutes.py and prompt_registry.py."""
    try:
        from importlib.resources import files

        return Path(str(files("tuned.data").joinpath("transition_templates")))
    except Exception:  # pragma: no cover - non-filesystem loader fallback
        return Path(__file__).resolve().parent / "transition_templates"


TEMPLATES_DIR = _templates_dir()


@dataclass(frozen=True)
class Skeleton:
    """One procedural posture's fact skeleton.

    `sha` is sha256 of the RAW FILE BYTES, first 12 hex - the same identity
    prompt_registry gives a teacher prompt, and for the same reason: it is
    recorded on every seed this skeleton builds, so editing a comma in a
    skeleton makes every cell built afterwards visibly different from the ones
    built before rather than silently comparable with them.
    """

    skeleton_id: str
    papers: str
    posture: str
    sha: str


def _split_skeleton(text: str, skeleton_id: str) -> tuple[str, str]:
    papers_at = text.find(PAPERS_MARK)
    posture_at = text.find(POSTURE_MARK)
    if papers_at == -1:
        raise ValueError(f"skeleton {skeleton_id!r} has no {PAPERS_MARK} block")
    if posture_at == -1:
        raise ValueError(f"skeleton {skeleton_id!r} has no {POSTURE_MARK} block")
    if posture_at < papers_at:
        raise ValueError(f"skeleton {skeleton_id!r} puts {POSTURE_MARK} before {PAPERS_MARK}")
    head = text[:papers_at]
    if head.strip():
        raise ValueError(
            f"skeleton {skeleton_id!r} has text before its {PAPERS_MARK} marker: "
            f"{head.strip()[:60]!r}"
        )
    papers = text[papers_at + len(PAPERS_MARK) : posture_at].strip()
    posture = text[posture_at + len(POSTURE_MARK) :].strip()
    if not papers:
        raise ValueError(f"skeleton {skeleton_id!r} has an empty {PAPERS_MARK} block")
    if not posture:
        raise ValueError(f"skeleton {skeleton_id!r} has an empty {POSTURE_MARK} block")
    return papers, posture


def load_skeleton(skeleton_id: str) -> Skeleton:
    path = TEMPLATES_DIR / f"{skeleton_id}.md"
    try:
        raw = path.read_bytes()
    except OSError:
        known = ", ".join(sorted(p.stem for p in TEMPLATES_DIR.glob("*.md")))
        raise KeyError(
            f"no transition skeleton {skeleton_id!r} in {TEMPLATES_DIR}; known ids: {known}"
        ) from None
    papers, posture = _split_skeleton(raw.decode("utf-8"), skeleton_id)
    return Skeleton(
        skeleton_id=skeleton_id,
        papers=papers,
        posture=posture,
        sha=hashlib.sha256(raw).hexdigest()[:SHA_LEN],
    )


# --------------------------------------------------------------------------
# The three transition provisions.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Provision:
    code: str
    section: str
    title: str
    kind: str
    text_kind: str
    effect: str
    derived_from: str

    @property
    def ref(self) -> SectionRef:
        return SectionRef(self.code, self.section)

    def block(self) -> str:
        """The prompt block for this provision.

        `text_kind` decides the label, and it is the label that keeps the
        prompt honest: a recorded-effect block says so in the words the
        teacher reads, so a teacher that quotes it is quoting the build's
        statement of the rule and not pretending to quote the Act.
        """
        lead = (
            "Text as enacted"
            if self.text_kind == "verbatim"
            else "Operative effect as recorded in this build's statute table (not a quotation)"
        )
        return (
            f"Section {self.section} of the {CODE_TITLES[self.code]} ({self.title}).\n"
            f"{lead}: {self.effect}"
        )


_REQUIRED_PROVISION_KEYS = (
    "code", "section", "title", "kind", "text_kind", "effect", "derived_from",
)

# Which provision governs which of statutes.CODE_KIND's three kinds. The
# mapping is the whole reason the file has three rows, so it is checked rather
# than assumed: a file that lost its evidence row would otherwise build cells
# whose evidence limb has no provision behind it.
PROVISION_KINDS = ("substantive", "procedural", "evidence")


def load_provisions(path: str | Path = TRANSITION_PROVISIONS_PATH) -> dict[str, Provision]:
    """kind -> Provision, refusing anything the stream cannot stand on.

    Every field is required and none is defaulted. An `effect` this module
    filled in for a missing row would be exactly the invented statutory
    statement the whole stream exists not to teach.
    """
    found: dict[str, Provision] = {}
    for i, row in enumerate(read_jsonl(path)):
        if "_comment" in row:
            continue
        missing = [key for key in _REQUIRED_PROVISION_KEYS if not row.get(key)]
        if missing:
            raise ValueError(f"transition provision row {i} is missing {missing}: {row}")
        kind = row["kind"]
        if kind not in PROVISION_KINDS:
            raise ValueError(
                f"transition provision row {i} has kind {kind!r}, not one of {PROVISION_KINDS}"
            )
        if kind in found:
            raise ValueError(f"transition provision row {i} is a second {kind!r} provision: {row}")
        if row["code"] not in CODE_TITLES:
            raise ValueError(f"transition provision row {i} has unknown code {row['code']!r}")
        found[kind] = Provision(**{key: row[key] for key in _REQUIRED_PROVISION_KEYS})
    absent = [kind for kind in PROVISION_KINDS if kind not in found]
    if absent:
        raise ValueError(
            f"{path} carries no {absent} provision. The transition stream asks which "
            f"enactment governs each of the three limbs, so a missing limb is a question "
            f"the prompts would put without the provision that answers it."
        )
    return found


def savings_block(provisions: dict[str, Provision]) -> str:
    """The {savings_text} slot: all three provisions, in limb order.

    One slot, three provisions, because the template has one savings slot and
    the question has three limbs - a cell shown only the substantive savings
    clause would be asked about procedure and evidence with nothing in front
    of it to decide them on.
    """
    return "\n\n".join(provisions[kind].block() for kind in PROVISION_KINDS)


# --------------------------------------------------------------------------
# The postures.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DatePosture:
    """A pair of day-offset spans, one for the conduct and one for the record.

    The spans are ordered so that the proceeding can never predate the offence,
    and the PROCEDURAL POSTURE then pushes it further out by the shortest time
    in which the record it narrates could exist (ProceduralPosture.
    min_lag_days) - an impossible record would teach the model to reason about
    one. The spans are NOT disjoint and this docstring used to say they were:
    on_appointed_day puts both dates on the same day, which is a real record at
    the FIR stage and an impossible one at every other, and it was the claim
    that was wrong rather than the design. Within a span the exact day is
    derived from the cell's own content key, so two cells in the same posture
    are not the same paper twice, and the derivation is a hash rather than an
    RNG so a grid rebuilt next year is the same grid.

    `anchors_to_invalidation` names WHICH BOUNDARY the offence span is measured
    from. For every ordinary family that is the appointed day, and the two
    spans are both offsets from it. For a family a court struck down the
    operative boundary is the JUDGMENT, years earlier, and "the months
    immediately before" / "the days immediately after" are only the legally
    interesting question when they are the months and days around THAT date.
    The proceeding span stays anchored to the appointed day either way,
    because which procedural code governs turns on the appointed day for every
    family alike. A cell records which anchor it was built on
    (`Cell.date_anchor`) so the eval never has to infer it.
    """

    name: str
    offence_span: tuple[int, int]
    proceeding_span: tuple[int, int]
    note: str
    anchors_to_invalidation: bool = False


DATE_POSTURES = (
    DatePosture(
        "well_before", (-1800, -900), (-800, -120),
        "conduct and proceeding both years before the appointed day",
    ),
    DatePosture(
        "just_before", (-330, -190), (-180, -8),
        "conduct in the months immediately before the boundary that decides this family's "
        "substantive answer, proceeding in the months immediately before the appointed day",
        anchors_to_invalidation=True,
    ),
    DatePosture(
        "on_appointed_day", (0, 0), (0, 0),
        "conduct ON the appointed day, and the record at the earliest its stage allows - "
        "new on both axes, because the codes came into force that day and the procedural "
        "saving reaches only what was pending IMMEDIATELY BEFORE it",
    ),
    DatePosture(
        "just_after", (1, 45), (50, 120),
        "conduct in the days immediately after the boundary that decides this family's "
        "substantive answer, proceeding after the appointed day",
        anchors_to_invalidation=True,
    ),
    DatePosture(
        "straddling", (-900, -60), (3, 240),
        "conduct before the appointed day, proceeding begun after it - the intersection "
        "the stream exists for: the charge stays under the old code while the proceeding "
        "runs under the new one",
    ),
)


@dataclass(frozen=True)
class ProceduralPosture:
    """A stage, the limb of the procedural saving it engages, and the shortest
    time in which the record its skeleton narrates could exist.

    `min_lag_days` belongs HERE, next to skeleton_id, because it is a fact
    about what the skeleton says rather than about the dates: the appeal
    skeleton narrates a conviction appealed against, and a conviction on the
    day of the conduct is a record no court file can hold. The two are edited
    together, and a skeleton edit is already a reviewed act (its sha is
    pinned).

    The floors are deliberately the FASTEST a real file could move, not the
    typical: this is a minimum, the drawn date is used whenever it is later,
    and a generous floor would quietly flatten the date distribution the
    postures exist to spread.
    """

    name: str
    skeleton_id: str
    limb: str
    min_lag_days: int


PROCEDURAL_POSTURES = (
    # An information recorded and an investigation opened on the day of the
    # conduct is the ordinary case, not an impossible one. Zero is a measured
    # value here, not an unset field.
    ProceduralPosture("fir", "posture_fir_v1", "investigation", 0),
    # The papers are a FINAL REPORT already before the court with cognizance
    # being taken on it. The date the skeleton narrates is the investigation's
    # start, which is the only lever it gives; three weeks is what keeps the
    # record from reading as conduct, investigation, report and cognizance all
    # on one day.
    ProceduralPosture("chargesheet", "posture_chargesheet_v1", "inquiry", 21),
    # Investigation, report, cognizance, charge framed, plea taken, part-heard.
    ProceduralPosture("trial", "posture_trial_v1", "trial", 90),
    # All of the above, a concluded trial, a conviction, and an appeal filed
    # against it. Six months is fast for that and it is possible; the same day
    # is not. It is also the widest floor the postures can carry - just_before
    # holds conduct and proceeding within 330 days of the appointed day, and a
    # longer floor would push that posture's record past the boundary it is
    # named for.
    ProceduralPosture("appeal", "posture_appeal_v1", "appeal", 180),
)


@dataclass(frozen=True)
class QuestionForm:
    """What the cell asks, and therefore what its key may demand.

    `forbids_counterpart` is the one flag with teeth: see the module
    docstring. It is true only on charge_only, whose question asks for the
    governing provision and expressly not the corresponding one, so a
    forbidden entry is grading the answer against the question that was put.
    """

    name: str
    ask: str
    limbs: tuple[str, ...]
    forbids_counterpart: bool


# The instruction every form carries. It is in the QUESTION slot, which is
# NOT part of grounding_text: a caution in a provision block would be
# grounding text, and a trace that echoed thirty characters of it would trip
# the verbatim gate on a sentence this module wrote rather than on anything
# the teacher copied.
# It says "the ANSWER too" in as many words, because the version that did not
# was read - correctly - as licensing exactly the artefact it exists to
# prevent: the recorded effect HAS been shown, so "do not quote words you have
# not been shown" left quoting it as enacted text permitted, and an answer that
# did so passed every gate. gates.check_statutory_quotation is the other half;
# a caution the gates do not enforce is a request, and a gate the prompt does
# not warn about is a trap.
NO_QUOTATION_CAUTION = (
    "You have been given each provision's identity and the effect this build's statute "
    "table records for it, not the section as enacted. Do not quote words you have not "
    "been shown, and do not name any provision that is not before you. The recorded "
    "effect is this build's statement of the rule and not the section's words, so do not "
    "present it - or any words at all - as text quoted from a section: no quotation marks "
    "attributed to a provision, in your reasoning or in your answer. State the effect in "
    "your own words and name the section you take it from."
)

QUESTION_FORMS = (
    QuestionForm(
        "charge_only",
        "State the enactment and the section under which the charge lies, and nothing "
        "else. Name the governing provision. Do NOT name the corresponding provision of "
        "the other enactment - the repealing and savings provision is not that "
        "corresponding provision and you may cite it.",
        ("substantive",),
        True,
    ),
    QuestionForm(
        "limb_by_limb",
        "Take the three questions separately: which enactment the accused stands charged "
        "under, which governs the conduct of this proceeding, and which governs the "
        "reception of evidence in it. Say for each which enactment governs and which "
        "date decides it.",
        ("substantive", "procedural", "evidence"),
        False,
    ),
    QuestionForm(
        "savings_effect",
        "Say what the repealing and savings provision preserves on these dates, and what "
        "follows from that for the accused. Name the provision you are working from.",
        ("substantive",),
        False,
    ),
    QuestionForm(
        "procedural_divergence",
        "Say which enactment governs the conduct of this proceeding at the stage it has "
        "reached, and whether that answer is the same as the enactment under which the "
        "charge lies. If the two differ, say so expressly and say why they differ.",
        ("substantive", "procedural"),
        False,
    ),
)

# Every (date, procedural, question) triple, in a fixed order. The grid is
# this tuple crossed with the verified families.
POSTURE_CELLS = tuple(
    (dp, pp, qf) for dp in DATE_POSTURES for pp in PROCEDURAL_POSTURES for qf in QUESTION_FORMS
)


# --------------------------------------------------------------------------
# The SECOND timeline: judicial invalidation.
# --------------------------------------------------------------------------
#
# statutes.py owns one boundary, the appointed day, and it decides which
# ENACTMENT governs. It does not and cannot decide whether the section named on
# either side of that boundary was in force at all, and for four `deleted`
# families that is the question the answer turns on. A section a court has
# struck down is not chargeable on any date after the judgment, so the savings
# clause has nothing to preserve: BNS s.358 saves liability INCURRED under the
# repealed Code, and no liability is incurred under a void section.
#
# The boundary is therefore the JUDGMENT date, not the appointed day, and it
# is years earlier. Keying these cells off the appointed day alone produced
# the error this block exists to remove: an answer key that demanded the
# struck-down section be named as the section the charge lies under, so that
# "no charge lies" - the correct answer - was a PERMANENT reject.
#
# WHERE THESE FACTS COME FROM, because a date recalled rather than read is the
# same failure one layer down. The event, the case and the year are recorded in
# the audited statute table (ipc_bns_map.jsonl `notes`), and every constant
# below is checked against that note at build time - a constant whose case name
# and year the audit sheet does not carry refuses its family rather than
# keying it. The DAY of each judgment is not in that file and is pinned here as
# a named constant with its source note. No judgment text is reproduced
# anywhere in this build, and nothing here is inferred from anything else.

# What the judicial event did to the section, as the audited note records it.
# Two shapes, because the two events in this build are not the same shape of
# fact and keying them the same way is what would produce a wrong answer.
SCOPE_SECTION_VOID = "section_void"
SCOPE_CONDUCT_SCOPED = "conduct_scoped"

# What this build can say about a family on one offence date.
STATUS_IN_FORCE = "in_force"
STATUS_VOID = "void"
STATUS_UNDECIDABLE = "undecidable"


@dataclass(frozen=True)
class JudicialEvent:
    """One judgment, as a date and a scope this module may key an answer on.

    `case` is the fragment the AUDITED NOTE must carry, not a citation: it is
    what ties the constant to the operator's sheet, and `is_grounded_in` is
    where that tie is enforced.
    """

    family: str
    case: str
    decided_on: date
    scope: str
    source_note: str

    @property
    def year(self) -> int:
        return self.decided_on.year

    def is_grounded_in(self, note: str) -> bool:
        text = note or ""
        return self.case.lower() in text.lower() and str(self.year) in text


JUDICIAL_INVALIDATIONS: dict[str, JudicialEvent] = {
    "IPC 497": JudicialEvent(
        family="IPC 497",
        case="Joseph Shine",
        decided_on=date(2018, 9, 27),
        scope=SCOPE_SECTION_VOID,
        source_note=(
            "The event, the case and the year are the audited statute table's own: "
            "ipc_bns_map.jsonl records this row as 'Adultery. Struck down in Joseph Shine "
            "v. Union of India (2018) and not re-enacted.' The DAY is not in that file. "
            "2018-09-27 is the operator-supplied constant for that judgment and is pinned "
            "here so that no date this module keys an answer on comes from unstated "
            "recollection; build time checks the case and the year back against the note. "
            "No text of the judgment is reproduced anywhere in this build. OPERATOR QUEUE: "
            "carry the status onto the mapping row itself (a judicial_status field) when "
            "the audit sheet is next signed - a constant in code is the honest place for "
            "it only until the sheet can hold it."
        ),
    ),
    "IPC 377": JudicialEvent(
        family="IPC 377",
        case="Navtej Singh Johar",
        decided_on=date(2018, 9, 6),
        scope=SCOPE_CONDUCT_SCOPED,
        source_note=(
            "The event, the case and the year are the audited statute table's own: "
            "ipc_bns_map.jsonl records this row as 'Unnatural offences. Read down for "
            "consenting adults by Navtej Singh Johar (2018) and carried into no BNS "
            "section.' The DAY is not in that file. 2018-09-06 is the operator-supplied "
            "constant, pinned for the same reason as the row above. The SCOPE is what "
            "matters here and it is why this family keys nothing: the note records that "
            "the section was read down for a class of conduct defined by CONSENT, and the "
            "fact skeletons describe the conduct only as 'the conduct described in the "
            "provisions set out below'. Which answer is correct therefore turns on a fact "
            "the papers deliberately do not carry. No text of the judgment is reproduced "
            "anywhere in this build."
        ),
    ),
}

# A note that records a judicial event this module has no constant for. The
# check is deliberately WIDE and its failure mode is a refusal, not a guess:
# the C1 error was a family whose own prompt told the teacher the section had
# been struck down while the key graded the answer as if it had not, so a new
# note the operator writes must stop the family emitting rather than key it.
_JUDICIAL_MARKER_RE = re.compile(
    r"struck\s+down|read\s+down|declared\s+void|unconstitutional|abeyance", re.IGNORECASE
)


def judicial_status(family_key: str, offence_date: date) -> tuple[str, str | None]:
    """(status, why it cannot be keyed) for one family on one offence date.

    Three answers and no fourth. IN_FORCE is every family the audit sheet
    records no judicial event for. VOID is a section a court struck down
    before the conduct: no charge lies, and that is a fact the key states.
    UNDECIDABLE is everything this build cannot settle from its own sources,
    and it is a REFUSAL - the cell is not built, the reason is named in the
    manifest, and no answer key is ever written on a guess.

    Two things land in UNDECIDABLE, both deliberately:

    * conduct-scoped read-downs (IPC 377), on EVERY date. After the judgment
      the section reaches non-consensual conduct and not consensual conduct,
      and the papers do not say which this was; before the judgment, whether a
      prosecution already on foot survives a declaration that the section was
      always inconsistent with the Constitution is a further question the
      audited note does not answer. Either way the correct answer depends on
      something this build does not carry.
    * conduct that PREDATES a striking down. Whether a prosecution for earlier
      conduct survives the declaration is a question of the declaration's
      reach backwards, and the audited note records the striking down and
      nothing about that. The day of the judgment itself is treated as
      undecidable for the same reason.
    """
    event = JUDICIAL_INVALIDATIONS.get(family_key)
    if event is None:
        return STATUS_IN_FORCE, None
    if event.scope == SCOPE_CONDUCT_SCOPED:
        return STATUS_UNDECIDABLE, (
            f"{family_key} was read down in {event.case} ({event.year}) for a class of "
            f"conduct defined by consent, and the fact skeletons describe the conduct only "
            f"as the conduct the provisions set out - so whether a charge lies on these "
            f"papers turns on a fact this build does not narrate. The key would have to "
            f"guess, and on this stream a guess IS the wrong answer"
        )
    if offence_date > event.decided_on:
        return STATUS_VOID, None
    return STATUS_UNDECIDABLE, (
        f"the conduct is dated on or before {event.decided_on.isoformat()}, when "
        f"{event.case} struck {family_key} down. Whether a prosecution for conduct that "
        f"early survives the declaration is a question of how far back it reaches, and "
        f"this build's statute table records the striking down and nothing about that, "
        f"so the cell is refused rather than keyed either way"
    )


# --------------------------------------------------------------------------
# Families.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Family:
    """One verified mapping row, as the grid sees it."""

    key: str
    kind: str
    old_ref: SectionRef | None
    new_ref: SectionRef | None
    note: str

    @property
    def identifying_ref(self) -> SectionRef:
        return self.old_ref if self.old_ref is not None else self.new_ref


def _ref_of(row: dict, side: str) -> SectionRef | None:
    code, number = row.get(f"{side}_code"), row.get(f"{side}_section")
    if not code or not number:
        return None
    return SectionRef(str(code).upper(), str(number))


def families(mapping: Mapping) -> tuple[list[Family], list[dict]]:
    """(families that may emit, refusals with the reason each was refused).

    THE ONLY GATE IS require_verified, called on every row - including the
    ones this module could perfectly well have read straight off `mapping.
    rows`. Reading the verified rows directly would produce the same list
    today and would stop being the same list the moment the refusal changed;
    routing every family through the pipeline's own refusal is what makes
    "the 17 unverified rows cannot emit" a property of statutes.py rather
    than a habit of this module.
    """
    ok: list[Family] = []
    refused: list[dict] = []
    for row in mapping.rows:
        old_ref, new_ref = _ref_of(row, "old"), _ref_of(row, "new")
        ref = old_ref if old_ref is not None else new_ref
        if ref is None:
            refused.append({"family": None, "kind": row.get("kind"), "reason": "row has no side"})
            continue
        try:
            # The refusal this raises covers BOTH ways a family can fail to
            # name itself: an unsigned audit row, and a row whose identifying
            # side is ambiguous - two new_offence rows on one new section make
            # Mapping.row() return None rather than guess, and "no mapping row
            # for BNS 48" is that refusal. Nothing is added here on top of it,
            # because a second check on Mapping's own invariants (old sides are
            # unique or construction raises) could not fire.
            mapping.require_verified(ref)
        except ValueError as exc:
            refused.append({"family": str(ref), "kind": row.get("kind"), "reason": str(exc)})
            continue
        note = str(row.get("notes") or "").strip()
        # THE SECOND GATE, and it is the same shape as the first: a family
        # whose status this build cannot state does not emit, and the reason is
        # the manifest's rather than a comment's. Both directions are refusals
        # because both are the same mistake - a key written on a fact nobody
        # can point at.
        event = JUDICIAL_INVALIDATIONS.get(str(ref))
        if event is not None and not event.is_grounded_in(note):
            refused.append({
                "family": str(ref),
                "kind": row.get("kind"),
                "reason": (
                    f"the judicial-invalidation constant for {ref} names {event.case} "
                    f"({event.year}) and this row's audited note does not, so the date the "
                    f"key would turn on cannot be traced to the audit sheet: {note!r}"
                ),
            })
            continue
        marker = None if event is not None else _JUDICIAL_MARKER_RE.search(note)
        if marker is not None:
            refused.append({
                "family": str(ref),
                "kind": row.get("kind"),
                "reason": (
                    f"this row's audited note records a judicial event ({marker.group(0)!r}) "
                    f"and no invalidation constant gives its date, so the build cannot say "
                    f"whether {ref} was in force on the day of the conduct. Add the date as a "
                    f"named constant with its source note, or the family cannot be keyed"
                ),
            })
            continue
        ok.append(
            Family(
                key=str(ref),
                kind=row["kind"],
                old_ref=old_ref,
                new_ref=new_ref,
                note=str(row.get("notes") or "").strip(),
            )
        )
    return ok, refused


# --------------------------------------------------------------------------
# Cells.
# --------------------------------------------------------------------------

ANCHOR_APPOINTED_DAY = "appointed_day"
ANCHOR_INVALIDATION = "invalidation"


@dataclass(frozen=True)
class Cell:
    cell_id: str
    family: Family
    date_posture: DatePosture
    procedural_posture: ProceduralPosture
    question_form: QuestionForm
    offence_date: date
    proceeding_started: date
    date_anchor: str = ANCHOR_APPOINTED_DAY

    @property
    def coordinates(self) -> dict:
        return {
            "family": self.family.key,
            "mapping_kind": self.family.kind,
            "date_posture": self.date_posture.name,
            "procedural_posture": self.procedural_posture.name,
            "question_form": self.question_form.name,
            # WHICH boundary this cell's conduct date was measured from. A
            # posture named "just_before" means one thing around the appointed
            # day and another around a judgment, and the eval reports the
            # coordinate: it should not have to re-derive which was meant.
            "date_anchor": self.date_anchor,
        }


def cell_key(family: Family, dp: DatePosture, pp: ProceduralPosture, qf: QuestionForm) -> str:
    return f"{family.key}|{dp.name}|{pp.name}|{qf.name}"


def cell_id_for(key: str) -> str:
    return hashlib.sha256(f"{TRANSITION_SOURCE_ID}:{key}".encode("utf-8")).hexdigest()[:CELL_ID_LEN]


def _digest(key: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest(), 16)


def _in_span(key: str, salt: str, span: tuple[int, int]) -> int:
    lo, hi = span
    return lo + _digest(key, salt) % (hi - lo + 1)


def cell_dates(
    key: str,
    dp: DatePosture,
    appointed_day: date,
    *,
    offence_anchor: date | None = None,
    min_lag_days: int = 0,
) -> tuple[date, date]:
    """(offence_date, proceeding_started) for one cell.

    Content-keyed, never random: the same key gives the same pair on any
    machine under any PYTHONHASHSEED, which is what lets a grid be rebuilt
    and compared with the one before it.

    `offence_anchor` is the boundary the CONDUCT is placed around and defaults
    to the appointed day. The proceeding is always placed around the appointed
    day, because which procedural code governs turns on that day and on
    nothing else - see DatePosture.

    `min_lag_days` is the procedural posture's floor and it moves the
    PROCEEDING only, never the conduct. That direction is the whole point on
    the on_appointed_day posture: the question there is about the OFFENCE date
    sitting exactly on the boundary, so the conduct stays where it is and the
    record moves out to where it could exist. The proceeding stays on or after
    the appointed day either way, so the posture is still new on both axes.
    """
    anchor = appointed_day if offence_anchor is None else offence_anchor
    offence = anchor + timedelta(days=_in_span(key, "offence", dp.offence_span))
    proceeding = appointed_day + timedelta(days=_in_span(key, "proceeding", dp.proceeding_span))
    return offence, max(proceeding, offence + timedelta(days=min_lag_days))


def offence_anchor_for(family: Family, dp: DatePosture, appointed_day: date) -> tuple[date, str]:
    """(the day this cell's conduct is placed around, the name of that anchor)."""
    event = JUDICIAL_INVALIDATIONS.get(family.key)
    if event is not None and dp.anchors_to_invalidation:
        return event.decided_on, ANCHOR_INVALIDATION
    return appointed_day, ANCHOR_APPOINTED_DAY


def build_grid(
    mapping: Mapping,
    *,
    appointed_day: date = APPOINTED_DAY,
    provisions: dict[str, Provision] | None = None,
) -> tuple[list[Cell], list[dict]]:
    """Every cell the verified map can emit, plus the refusal list.

    Two kinds of thing land in `refused`, and the manifest prints both: a
    FAMILY the audit sheet has not signed off (statutes.require_verified's own
    message), and a CELL whose ideal answer no gate stack could pass
    (ungateable_reason). Neither is dropped silently - a grid that lost a
    quarter of itself looks exactly like a healthy one without that list.

    Family order is content-keyed (sha of the family key) rather than by
    section number, so the reserve and the sample are not accidentally a slice
    of the low-numbered IPC sections.
    """
    provisions = load_provisions() if provisions is None else provisions
    ok, refused = families(mapping)
    ok = sorted(ok, key=lambda f: (hashlib.sha256(f.key.encode("utf-8")).hexdigest(), f.key))
    cells: list[Cell] = []
    for family in ok:
        for dp, pp, qf in POSTURE_CELLS:
            key = cell_key(family, dp, pp, qf)
            anchor, anchor_name = offence_anchor_for(family, dp, appointed_day)
            offence, proceeding = cell_dates(
                key, dp, appointed_day, offence_anchor=anchor, min_lag_days=pp.min_lag_days
            )
            cell = Cell(
                cell_id=cell_id_for(key),
                family=family,
                date_posture=dp,
                procedural_posture=pp,
                question_form=qf,
                offence_date=offence,
                proceeding_started=proceeding,
                date_anchor=anchor_name,
            )
            answer_key = answer_key_for(cell, provisions)
            reason = ungateable_reason(answer_key)
            if reason is not None:
                refused.append({
                    "family": family.key,
                    "cell": key,
                    "reason": reason,
                    # WHY the cell could not be built, in two words the manifest
                    # can count. They are different failures: one is a gate
                    # stack that cannot accept the ideal answer, the other is a
                    # law this build cannot state with certainty.
                    "basis": (
                        REFUSAL_LEGAL_CERTAINTY
                        if answer_key.get("undecidable_reason")
                        else REFUSAL_GATE_STACK
                    ),
                })
                continue
            cells.append(cell)
    return cells, refused


# --------------------------------------------------------------------------
# Selection: one order, two disjoint prefixes.
# --------------------------------------------------------------------------

def coverage_stride(n: int) -> int:
    """A step coprime with `n`, so stepping it visits every posture.

    Starting near the golden section keeps consecutive families far apart in
    posture space; coprimality is what makes the sweep a permutation rather
    than a short cycle, and it is checked rather than assumed because the
    posture count changes the moment a question form is added.
    """
    if n <= 2:
        return 1
    for candidate in range(max(1, round(n * 0.61803)), n):
        if gcd(candidate, n) == 1:
            return candidate
    return 1


@dataclass(frozen=True)
class Selection:
    reserve: list[Cell]
    sample: list[Cell]
    order: list[Cell]


def selection_order(cells: list[Cell]) -> list[Cell]:
    """The grid in coverage order: every family's j-th pick before any
    family's (j+1)-th, and each family starting at a different posture.

    Two properties fall out, and both are asserted rather than hoped for:

    * taking the first K cells gives every family floor(K/F) or ceil(K/F)
      cells, so no family is left out of a draw bigger than the family count;
    * within one j-sweep a full family's postures are (f + j*stride) mod P for
      every f, so when at least P families are full EVERY posture triple
      appears in that sweep. With 150 full families and 80 triples that is a
      guarantee, not a hash coincidence.

    A family may be SHORT of the full posture count - build_grid excludes
    ungateable cells, which is what happens to the four `deleted` families on
    the two post-appointed-day date postures. A short family is rotated
    through its own length rather than walked in file order, so its draw
    spreads over the postures it does have instead of always starting at the
    same one.

    And because both the reserve and the sample are PREFIXES of this one
    order, they are disjoint by construction.
    """
    by_family: dict[str, list[Cell]] = {}
    for cell in cells:
        by_family.setdefault(cell.family.key, []).append(cell)
    order_keys = list(dict.fromkeys(cell.family.key for cell in cells))
    strides = {key: coverage_stride(len(bucket)) for key, bucket in by_family.items()}

    out: list[Cell] = []
    depth = max((len(v) for v in by_family.values()), default=0)
    for j in range(depth):
        for f, key in enumerate(order_keys):
            bucket = by_family[key]
            if j >= len(bucket):
                continue
            out.append(bucket[(f + j * strides[key]) % len(bucket)])
    return out


def select_cells(cells: list[Cell], *, sample: int, reserve: int) -> Selection:
    """`reserve` eval cells, then `sample` training cells, off one order."""
    order = selection_order(cells)
    if sample + reserve > len(order):
        raise ValueError(
            f"the grid holds {len(order)} cells, which cannot supply {reserve} reserved "
            f"plus {sample} sampled. Either the mapping lost verified rows or the config "
            f"is asking for more of the grid than exists."
        )
    return Selection(
        reserve=order[:reserve], sample=order[reserve : reserve + sample], order=order
    )


# --------------------------------------------------------------------------
# The answer key.
# --------------------------------------------------------------------------

# What the savings clause does on this cell. A closed vocabulary, because the
# eval reports it and a free-text field would be four spellings of two things.
SAVINGS_PRESERVED = "old_liability_preserved"
SAVINGS_NO_RETROSPECTIVE_OFFENCE = "new_offence_cannot_reach_earlier_conduct"
SAVINGS_REPEALED_WITHOUT_SUCCESSOR = "repealed_without_successor"
SAVINGS_NOT_ENGAGED = "new_code_governs_directly"
# The section was VOID when the conduct occurred, so no liability was ever
# incurred under it and s.358 - which preserves liability incurred - has
# nothing to preserve. This is the one the four `deleted` families were
# missing, and the one whose absence made "no charge lies" a permanent reject.
SAVINGS_NO_OFFENCE_LIES = "no_offence_lies"
# Nothing is asserted: the cell is refused, and this value exists so that a key
# built for a refused cell cannot be mistaken for a key that decided something.
SAVINGS_NOT_DECIDABLE = "not_decidable_on_this_build"

# Why a cell could not be built. Counted separately in the manifest.
REFUSAL_GATE_STACK = "gate-stack"
REFUSAL_LEGAL_CERTAINTY = "legal-certainty"


def _entry(ref: SectionRef) -> dict:
    return {"code": ref.code, "number": ref.number}


def answer_key_for(cell: Cell, provisions: dict[str, Provision]) -> dict:
    """The cell's key, computed from statutes.py's decision table alone.

    Derivable independently: given the mapping row and the two dates, nothing
    here needs the prompt, the skeleton or the store. tests recompute a
    stratified sample of these straight out of governing_family() and compare.
    """
    dates = {"offence_date": cell.offence_date, "proceeding_started": cell.proceeding_started}
    by_kind = {kind: governing_family(kind, **dates) for kind in PROVISION_KINDS}
    substantive = by_kind["substantive"]

    # THE SECOND TIMELINE, before the first one is applied to anything. Which
    # ENACTMENT governs is decided by the appointed day; whether the section
    # named on the old side of it was in force at all is decided by the
    # judgment date, and a key that answered the first question without asking
    # the second is the C1 error.
    status, undecidable = judicial_status(cell.family.key, cell.offence_date)

    old_ref, new_ref = cell.family.old_ref, cell.family.new_ref
    if status == STATUS_IN_FORCE:
        charge_ref = old_ref if substantive == "old" else new_ref
        counterpart = new_ref if substantive == "old" else old_ref
    else:
        # Nothing is chargeable under a section that was void when the conduct
        # occurred, and nothing is chargeable under a section whose reach this
        # build cannot state. There is no counterpart either: these four
        # families have a null new side, which is what `deleted` means.
        charge_ref = None
        counterpart = None
    # The section the answer must ENGAGE with. Where the governing family has
    # a section it is that one; where it has none - a new offence charged
    # against earlier conduct, a repealed section charged against later
    # conduct, a section a court struck down before the conduct - it is the
    # section the answer has to name in order to rule it out. Expecting
    # nothing there would leave the cell with no teeth at all.
    engage_ref = charge_ref if charge_ref is not None else counterpart
    if engage_ref is None and status != STATUS_IN_FORCE:
        engage_ref = cell.family.identifying_ref

    # See the module docstring, and note what MEASUREMENT put here: an answer
    # that merely uses the word "savings" satisfies gates._mentions_savings but
    # NOT statutes._cites_savings_clause, which wants BNS s.358 cited or the
    # words "section 358". The two are different tests, and on 287 of the first
    # 1,250 cells built that difference rejected the ideal answer at
    # check_temporal while check_answer_key passed it. Expecting s.358 itself
    # wherever the savings mention is required closes the gap by construction:
    # a compliant answer now necessarily carries the citation the suppression
    # looks for.
    requires_savings = substantive == "old" or cell.question_form.name == "savings_effect"

    expected: list[SectionRef] = []
    if engage_ref is not None:
        expected.append(engage_ref)
    limbs = cell.question_form.limbs
    if "procedural" in limbs:
        expected.append(provisions["procedural"].ref)
    if "evidence" in limbs:
        expected.append(provisions["evidence"].ref)
    if requires_savings:
        expected.append(provisions["substantive"].ref)

    forbidden: list[SectionRef] = []
    if cell.question_form.forbids_counterpart and charge_ref is not None and counterpart is not None:
        forbidden.append(counterpart)

    if status == STATUS_UNDECIDABLE:
        consequence = SAVINGS_NOT_DECIDABLE
    elif status == STATUS_VOID:
        # s.358 preserves liability INCURRED under the repealed Code. A void
        # section incurs none, so the savings clause is engaged and preserves
        # nothing - which is the answer, and why the answer still has to cite
        # s.358 to give it.
        consequence = SAVINGS_NO_OFFENCE_LIES
    elif charge_ref is not None:
        consequence = SAVINGS_PRESERVED if substantive == "old" else SAVINGS_NOT_ENGAGED
    elif substantive == "old":
        consequence = SAVINGS_NO_RETROSPECTIVE_OFFENCE
    else:
        consequence = SAVINGS_REPEALED_WITHOUT_SUCCESSOR

    # Demanded exactly where a correct answer necessarily cites both: the old
    # family's own section, and BNS s.358, which the line above has just put
    # in `expected` for every old-governing cell. So this is satisfiable by
    # construction and never asks for a citation the question did not invite.
    both_families = substantive == "old" and old_ref is not None

    procedural_rule = {
        **_entry(provisions["procedural"].ref),
        "effect": (
            "the Code of Criminal Procedure, 1973 continues to govern this proceeding"
            if by_kind["procedural"] == "old"
            else "the Bharatiya Nagarik Suraksha Sanhita, 2023 governs this proceeding"
        ),
    }
    evidence_rule = {
        **_entry(provisions["evidence"].ref),
        "effect": (
            "evidence continues to be received under the Indian Evidence Act, 1872"
            if by_kind["evidence"] == "old"
            else "evidence is received under the Bharatiya Sakshya Adhiniyam, 2023"
        ),
    }

    event = JUDICIAL_INVALIDATIONS.get(cell.family.key)
    return {
        # --- read by gates.check_answer_key ---
        "governing_family": substantive,
        "expected_sections": [_entry(ref) for ref in expected],
        "forbidden_sections": [_entry(ref) for ref in forbidden],
        "requires_savings_mention": requires_savings,
        "must_name_both_families": both_families,
        # THE PASSING ANSWER IS "no charge lies". Set exactly where the section
        # was void when the conduct occurred: the answer must SAY that no
        # charge lies (it still has to name the section, or it has ruled out
        # nothing), and an answer that names it as the section the charge lies
        # under fails. Without this the key could only ask for citations, and
        # the only answer it could express on these cells was the wrong one.
        "requires_no_liability_statement": status == STATUS_VOID,
        # --- metadata: the eval and the report read these, the gate does not ---
        "families_by_kind": dict(by_kind),
        "charge": None if charge_ref is None else _entry(charge_ref),
        "counterpart": None if counterpart is None else _entry(counterpart),
        "savings_consequence": consequence,
        "judicial_status": status,
        "judicial_event": None if event is None else {
            "case": event.case,
            "decided_on": event.decided_on.isoformat(),
            "scope": event.scope,
        },
        # Set only on a cell that must NOT be built. build_grid reads it, and
        # its presence is what makes the refusal a named legal refusal rather
        # than a gate-stack one.
        "undecidable_reason": undecidable,
        "procedural_rule": procedural_rule,
        "evidence_rule": evidence_rule,
        "appointed_day": APPOINTED_DAY.isoformat(),
        **cell.coordinates,
    }


def ungateable_reason(key: dict) -> str | None:
    """Why this cell must not be built, or None. TWO bases, in order.

    FIRST, the law: a cell whose correct answer this build cannot state with
    certainty is refused with the reason judicial_status gave, and no key is
    written on a guess. That check comes first because it is prior - a cell
    whose answer nobody knows is not worth asking whether the gates could
    grade it.

    SECOND, the gate stack. A cell is only worth building if an answer that
    satisfies its key also passes the rest of the gate stack. check_answer_key
    and check_temporal read the same generation, and one of them demands
    citations the other can reject:

    * a NEW-family section cited where the OLD family governs is suppressed by
      statutes._cites_savings_clause, and every such key expects BNS s.358, so
      the suppression always fires. Gateable.
    * an OLD-family section cited where the NEW family governs has NO
      suppression path in cross_code_review at all. The key would be demanding
      a citation that a PERMANENT gate rejects, so the ideal answer is a
      permanent reject and the cell burns its seed on the way to teaching
      nothing.

    The second case is not hypothetical and it is not a corner: it is every
    `deleted` family (IPC 124A/309/377/497) on a date posture whose offence
    falls on or after the appointed day. The only correct answer there has to
    NAME the repealed section in order to say it was repealed, and naming it
    is what check_temporal flags. Those cells are excluded from the grid and
    counted in the manifest; the same four families keep every posture whose
    offence predates the repeal AND whose answer this build can state.
    """
    undecidable = key.get("undecidable_reason")
    if undecidable:
        return str(undecidable)
    savings_mentioned = bool(key.get("requires_savings_mention"))
    by_kind = key.get("families_by_kind") or {}
    for entry in key.get("expected_sections") or []:
        code = entry["code"]
        kind = CODE_KIND.get(code)
        if kind is None:  # pragma: no cover - CODE_TITLES and CODE_KIND agree
            continue
        cited = "old" if code in OLD_CODES else "new"
        governing = by_kind.get(kind)
        if governing is None or cited == governing:
            continue
        if cited == "new" and savings_mentioned:
            continue
        return (
            f"the key expects {code} {entry['number']} while the "
            f"{governing!r} family governs {kind} questions on these dates, and "
            f"check_temporal has no suppression for that direction"
        )
    return None


# --------------------------------------------------------------------------
# Rendering.
# --------------------------------------------------------------------------

def pretty_date(day: date) -> str:
    return f"{day.day} {_MONTHS[day.month - 1]} {day.year}"


def _provision_block(ref: SectionRef, body: str) -> str:
    """"Section 302 of the Indian Penal Code, 1860." - NUMBER FIRST.

    Not a house style: statutes._STATUTE_RE matches `<marker> <number> <of
    the> <code>` and nothing else, so "Indian Penal Code, 1860 - section 302"
    parses to no section at all. A block written that way is a provision the
    pipeline's own extractor cannot see in its own grounding, which is how a
    citation allow-list ends up empty while reading perfectly well to a human.
    A test parses every block back and compares it against the answer key.
    """
    return f"Section {ref.number} of the {CODE_TITLES[ref.code]}.\n{body}"


_KIND_PHRASE = {
    "one_to_one": (
        "a one-to-one carry-over of the provision above: no source consulted for this "
        "build's statute table flags a change of ingredients, scope or punishment"
    ),
    "changed": (
        "a CHANGED correspondence: sources consulted for this build's statute table flag "
        "altered ingredients, scope or punishment, so the two provisions are not "
        "interchangeable and neither stands in for the other"
    ),
}


def old_block(family: Family) -> str:
    if family.old_ref is None:
        return (
            "The repealed codes contain no counterpart to the provision set out below. "
            "This build's statute table records it as an offence created by the new "
            "enactment with no corresponding section in the Indian Penal Code, 1860."
        )
    note = family.note or "no marginal note recorded"
    return _provision_block(
        family.old_ref,
        f"Marginal note recorded in this build's statute table: {note}",
    )


def new_block(family: Family) -> str:
    if family.new_ref is None:
        return (
            "The new enactments contain no counterpart to the provision set out above. "
            "This build's statute table records that section as repealed without "
            "re-enactment: nothing in the new code carries it forward, and no provision "
            "of the new code is its successor."
        )
    if family.old_ref is None:
        note = family.note or "no marginal note recorded"
        return _provision_block(
            family.new_ref,
            f"Marginal note recorded in this build's statute table: {note}",
        )
    return _provision_block(
        family.new_ref,
        f"The correspondence recorded in this build's statute table is "
        f"{_KIND_PHRASE.get(family.kind, 'unclassified')}.",
    )


def render_cell(cell: Cell, provisions: dict[str, Provision]) -> dict:
    """The prompt slots for one cell: {source} and the four meta slots.

    generate.build_slots reads scenario / old_section_text / new_section_text
    / savings_text off meta_json and refuses a transition task that is missing
    any of them, unspent - so every one of them is filled here or the cell is
    not written at all.
    """
    skeleton = load_skeleton(cell.procedural_posture.skeleton_id)
    scenario = skeleton.posture.format(
        offence_date=pretty_date(cell.offence_date),
        proceeding_date=pretty_date(cell.proceeding_started),
    )
    question = f"{cell.question_form.ask}\n\n{NO_QUOTATION_CAUTION}"
    return {
        "source": skeleton.papers,
        "scenario": scenario,
        "old_section_text": old_block(cell.family),
        "new_section_text": new_block(cell.family),
        "savings_text": savings_block(provisions),
        "question": question,
        "skeleton_id": skeleton.skeleton_id,
        "skeleton_sha": skeleton.sha,
    }


def seed_row(cell: Cell, provisions: dict[str, Provision], *, held_out: bool) -> dict:
    """One store.seed row. `held_out` is the eval reserve's mark."""
    slots = render_cell(cell, provisions)
    key = answer_key_for(cell, provisions)
    return {
        "seed_id": cell.cell_id,
        "source_id": TRANSITION_SOURCE_ID,
        "native_id": cell_key(
            cell.family, cell.date_posture, cell.procedural_posture, cell.question_form
        ),
        "court": None,
        "decision_date": None,
        "offence_date": cell.offence_date.isoformat(),
        "case_type": "criminal",
        "code_era": "ipc" if key["governing_family"] == "old" else "bns",
        "text": slots["source"],
        "token_count": len(slots["source"]) // 4,
        "answer_key_json": key,
        "meta_json": {
            "estimator": "chars/4",
            "stream": STREAM,
            "task_type": TASK_TYPE,
            # generate.build_slots' four required transition slots.
            "scenario": slots["scenario"],
            "old_section_text": slots["old_section_text"],
            "new_section_text": slots["new_section_text"],
            "savings_text": slots["savings_text"],
            "question": slots["question"],
            # gate_context reads this one; the seed table has no column for it.
            "proceeding_started": cell.proceeding_started.isoformat(),
            "skeleton_id": slots["skeleton_id"],
            "skeleton_sha": slots["skeleton_sha"],
            # THE RESERVE MARK. tasks._candidate_seeds refuses a held-out seed,
            # so an eval cell can never be planned as a teacher generation.
            "held_out": held_out,
            **cell.coordinates,
        },
    }


# --------------------------------------------------------------------------
# Build.
# --------------------------------------------------------------------------

def build_transition(
    store,
    cfg,
    *,
    mapping: Mapping | None = None,
    provisions: dict[str, Provision] | None = None,
    dry_run: bool = False,
) -> dict:
    """Write the sampled and reserved cells as seed rows; return the manifest.

    The manifest is the instrument: it carries the grid's measured shape, the
    per-family coverage of the draw, and every family that could NOT emit with
    the refusal statutes.py raised for it. A build that quietly lost half the
    grid to an un-signed audit sheet looks exactly like a healthy one without
    that list.
    """
    if cfg.transition is None:
        raise ValueError(
            "this build config has no `transition:` block, so the grid has no sample "
            "size and no eval reserve. transition.sample and transition.eval_reserve "
            "are the only two numbers this module cannot derive."
        )
    mapping = Mapping.load() if mapping is None else mapping
    provisions = load_provisions() if provisions is None else provisions

    cells, refused = build_grid(mapping, provisions=provisions)
    selection = select_cells(
        cells, sample=cfg.transition.sample, reserve=cfg.transition.eval_reserve
    )

    reserve_ids = {cell.cell_id for cell in selection.reserve}
    sample_ids = {cell.cell_id for cell in selection.sample}
    overlap = reserve_ids & sample_ids
    if overlap:  # pragma: no cover - a prefix split cannot overlap; the assert is the point
        raise AssertionError(
            f"{len(overlap)} cells are in BOTH the eval reserve and the training sample; "
            f"the two are prefixes of one order and must be disjoint by construction"
        )

    rows = [seed_row(cell, provisions, held_out=True) for cell in selection.reserve]
    rows += [seed_row(cell, provisions, held_out=False) for cell in selection.sample]

    per_family: dict[str, int] = {}
    for cell in selection.sample:
        per_family[cell.family.key] = per_family.get(cell.family.key, 0) + 1
    posture_pairs = {
        (cell.date_posture.name, cell.procedural_posture.name) for cell in selection.sample
    }

    family_refusals = [entry for entry in refused if "cell" not in entry]
    cell_refusals = [entry for entry in refused if "cell" in entry]
    refusals_by_basis: dict[str, int] = {}
    for entry in cell_refusals:
        basis = entry.get("basis", REFUSAL_GATE_STACK)
        refusals_by_basis[basis] = refusals_by_basis.get(basis, 0) + 1
    emitting = {cell.family.key for cell in cells}
    # A family can pass every audit gate and still emit NOTHING - IPC 377 does,
    # because no posture of it can be keyed with certainty. It is in neither
    # `families_emitting` nor `families_refused`, so without this line it is
    # visible only by subtracting two numbers nobody subtracts.
    silent = sorted(
        {entry["family"] for entry in cell_refusals if entry["family"] not in emitting}
    )
    manifest = {
        "grid_cells": len(cells),
        "families_emitting": len(emitting),
        "families_refused": len(family_refusals),
        "refusals": family_refusals,
        "cells_refused": len(cell_refusals),
        "cells_refused_by_basis": refusals_by_basis,
        "families_emitting_nothing": silent,
        "cell_refusal_families": sorted({entry["family"] for entry in cell_refusals}),
        "posture_cells": len(POSTURE_CELLS),
        "sample": len(selection.sample),
        "reserve": len(selection.reserve),
        "sample_families_covered": len(per_family),
        "sample_per_family_min": min(per_family.values()) if per_family else 0,
        "sample_per_family_max": max(per_family.values()) if per_family else 0,
        "sample_posture_pairs": len(posture_pairs),
        "posture_pairs_total": len(DATE_POSTURES) * len(PROCEDURAL_POSTURES),
        "appointed_day": APPOINTED_DAY.isoformat(),
        "dry_run": dry_run,
    }
    if dry_run:
        manifest["written"] = 0
        return manifest

    store.upsert_source(TRANSITION_SOURCE_ID, TRANSITION_LICENSE, url=None)
    manifest["written"] = store.upsert_seeds(rows)
    store.log_event("transition_grid_built", manifest)
    return manifest


def main(argv=None) -> int:
    import argparse

    from tuned.data.config import load_build_config
    from tuned.data.paths import build_paths
    from tuned.data.store import Store

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data_law_v1.yaml")
    parser.add_argument(
        "--dry-run", action="store_true", help="measure the grid, write nothing"
    )
    args = parser.parse_args(argv)

    cfg = load_build_config(args.config)
    paths = build_paths(cfg.build.workdir).ensure()
    store = Store.open(paths.state_db)
    try:
        manifest = build_transition(store, cfg, dry_run=args.dry_run)
    finally:
        store.close()

    print(
        f"grid {manifest['grid_cells']} cells "
        f"= {manifest['families_emitting']} families x {manifest['posture_cells']} postures"
    )
    print(
        f"  refused families {manifest['families_refused']} "
        f"(unverified mapping rows cannot emit)"
    )
    for refusal in manifest["refusals"]:
        print(f"    {refusal['family']:<12} {str(refusal['kind']):<12} {refusal['reason'][:90]}")
    print(
        f"  refused cells {manifest['cells_refused']} across "
        f"{manifest['cell_refusal_families']} "
        f"({manifest['cells_refused_by_basis']})"
    )
    if manifest["families_emitting_nothing"]:
        print(
            f"    EMIT NOTHING AT ALL: {manifest['families_emitting_nothing']} "
            f"(verified, but no posture of them can be keyed with certainty)"
        )
    print(
        f"  reserve {manifest['reserve']}  sample {manifest['sample']}  "
        f"written {manifest['written']}"
    )
    print(
        f"  sample covers {manifest['sample_families_covered']}/{manifest['families_emitting']} "
        f"families ({manifest['sample_per_family_min']}-{manifest['sample_per_family_max']} "
        f"cells each) and {manifest['sample_posture_pairs']}/{manifest['posture_pairs_total']} "
        f"date x procedural posture pairs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
