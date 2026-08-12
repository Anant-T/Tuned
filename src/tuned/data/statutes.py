"""IPC/CrPC/IEA -> BNS/BNSS/BSA transition logic for the law_v1 pipeline.

India replaced its three criminal codes with effect from the appointed day,
1 July 2024. Which family a given answer may cite is NOT a style question -
it is decided by statute, and citing the wrong family is a hard rejection.
cross_code_flags() is that gate's primitive.

The two rules, and only these two:

  SUBSTANTIVE offences (IPC <-> BNS) follow the date of the OFFENCE.
    BNS s.358(2) preserves the effect of s.6 of the General Clauses Act,
    1897: repeal does not affect any right, obligation, liability, penalty
    or punishment incurred under the repealed enactment, and any
    investigation/proceeding about it may continue as if the repeal had not
    happened. So an offence committed before 2024-07-01 is charged and
    punished under the IPC FOREVER - the passage of time never converts it
    into a BNS offence.

  PROCEDURAL (CrPC <-> BNSS) and EVIDENCE (IEA <-> BSA) follow whether the
  PROCEEDING was already pending on the appointed day.
    BNSS s.531(2)(a): an appeal, application, trial, inquiry or investigation
    pending immediately before the commencement is disposed of / continued /
    held under the CrPC as it stood. BSA s.170 saves pending proceedings the
    same way. Anything STARTED on or after the appointed day runs under the
    new code.

Their intersection is the case worth getting right: an FIR registered in
August 2024 for a 2023 offence is investigated under the BNSS while the
charges stay under the IPC. Both statements are correct in the same
sentence, and a model that "corrects" either one is wrong.

On the appointed day itself counts as NEW on both axes: the codes came into
force ON 2024-07-01, and s.531(2)(a) saves only what was pending
*immediately before* that date.
"""

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from tuned.data.jsonl import read_jsonl

OLD_CODES = {"IPC", "CRPC", "IEA"}
NEW_CODES = {"BNS", "BNSS", "BSA"}
CODE_KIND = {
    "IPC": "substantive",
    "BNS": "substantive",
    "CRPC": "procedural",
    "BNSS": "procedural",
    "IEA": "evidence",
    "BSA": "evidence",
}
KINDS = frozenset(CODE_KIND.values())

# Single source of truth. configs/data_law_v1.yaml build.appointed_day must
# equal this; the gate layer asserts it at load.
APPOINTED_DAY = date(2024, 7, 1)

# Flag vocabulary of cross_code_flags(). The two strings are DIRECTION
# labels, named after the commonest (substantive) case: "bns-cited-for-old-
# offence" is emitted whenever a new-family code is cited where the old
# family governs, and "ipc-cited-for-new-offence" for the reverse - including
# when the codes involved are CrPC/BNSS or IEA/BSA.
FLAG_NEW_FOR_OLD = "bns-cited-for-old-offence"
FLAG_OLD_FOR_NEW = "ipc-cited-for-new-offence"

MAPPING_KINDS = frozenset({"one_to_one", "changed", "new_offence", "deleted"})


def _resources_dir() -> Path:
    try:
        from importlib.resources import files

        return Path(str(files("tuned.data").joinpath("resources")))
    except Exception:  # pragma: no cover - non-filesystem loader fallback
        return Path(__file__).resolve().parent / "resources"


RESOURCES_DIR = _resources_dir()
IPC_BNS_MAP_PATH = RESOURCES_DIR / "ipc_bns_map.jsonl"


@dataclass(frozen=True)
class SectionRef:
    code: str
    number: str

    @property
    def base_number(self) -> str:
        """Section number without its subsection: 103(2) -> 103. The letter
        suffix is part of the section identity and is NEVER stripped - IPC
        304B is not IPC 304."""
        return self.number.split("(", 1)[0]

    def __str__(self) -> str:
        return f"{self.code} {self.number}"


# --------------------------------------------------------------------------
# The decision table.
# --------------------------------------------------------------------------

def governing_family(
    kind: str, *, offence_date: date | None, proceeding_started: date | None
) -> str:
    """Which code family governs a question of `kind` ("substantive" |
    "procedural" | "evidence"). Returns "old" or "new"."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}, must be one of {sorted(KINDS)}")

    if kind == "substantive":
        # BNS s.358(2) + General Clauses Act s.6: liability incurred under the
        # repealed IPC survives the repeal. The proceeding date is irrelevant
        # here - a 2019 murder tried in 2030 is still an IPC s.302 murder.
        if offence_date is None:
            raise ValueError(
                "substantive family needs an offence_date: the charge follows "
                "the date of the offence, not the date of the proceeding"
            )
        return "old" if offence_date < APPOINTED_DAY else "new"

    # BNSS s.531(2)(a) (procedural) and BSA s.170 (evidence): what was pending
    # immediately before the appointed day continues under the old code;
    # anything started on/after it runs under the new one - EVEN IF the
    # offence predates the transition.
    reference = proceeding_started if proceeding_started is not None else offence_date
    if reference is None:
        raise ValueError(
            f"{kind} family needs proceeding_started (or an offence_date to "
            "fall back on) to decide whether the proceeding was pending on "
            "the appointed day"
        )
    # Best-effort fallback: with no proceeding date recorded, the offence date
    # is the only available proxy for when the matter started. It is a proxy,
    # not the rule - callers that can supply proceeding_started must.
    return "old" if reference < APPOINTED_DAY else "new"


# --------------------------------------------------------------------------
# Section extraction.
# --------------------------------------------------------------------------

_CODE_ALIASES = {
    "IPC": "IPC",
    "INDIANPENALCODE": "IPC",
    "PENALCODE": "IPC",
    "CRPC": "CRPC",
    "CODEOFCRIMINALPROCEDURE": "CRPC",
    "CRIMINALPROCEDURECODE": "CRPC",
    "IEA": "IEA",
    "EVIDENCEACT": "IEA",
    "INDIANEVIDENCEACT": "IEA",
    "BNS": "BNS",
    "BHARATIYANYAYASANHITA": "BNS",
    "BNSS": "BNSS",
    "BHARATIYANAGARIKSURAKSHASANHITA": "BNSS",
    "BSA": "BSA",
    "BHARATIYASAKSHYAADHINIYAM": "BSA",
}


def resolve_code(alias: str) -> str | None:
    """Alias -> canonical code. Everything that is not a letter is dropped
    first, so "Cr.P.C.", "CrPC", "Cr. P. C." and "cr p c" are one key, and
    "Indian Penal Code, 1860" folds onto INDIANPENALCODE."""
    key = re.sub(r"[^A-Za-z]", "", alias or "").upper()
    return _CODE_ALIASES.get(key)


# Longest-first: BNSS before BNS, the spelled-out names before nothing else
# can eat their prefix.
_CODE = (
    r"\b(?P<code>"
    r"B\.?N\.?S\.?S\.?|B\.?N\.?S\.?|I\.?P\.?C\.?|Cr\.?\s*P\.?\s*C\.?|I\.?E\.?A\.?|B\.?S\.?A\.?"
    r"|Indian\s+Penal\s+Code|Penal\s+Code"
    r"|Code\s+of\s+Criminal\s+Procedure|Criminal\s+Procedure\s+Code"
    r"|Indian\s+Evidence\s+Act|Evidence\s+Act"
    r"|Bharatiya\s+Nyaya\s+Sanhita|Bharatiya\s+Nagarik\s+Suraksha\s+Sanhita"
    r"|Bharatiya\s+Sakshya\s+Adhiniyam"
    r")(?![A-Za-z])"
)

# "Section", "Sections", "Sec.", "S.", "SS.", "u/s", "u/ss", "§", "§§", plus
# the two marker-less lead-ins Indian pleadings use constantly ("punishable
# under 302 IPC", "read with 34 IPC").
# The single-letter form REQUIRES its dot ("S. 302") - a bare "s" is too
# common in prose to use as a citation marker - and a bare number REQUIRES
# one of these lead-ins, so "In 2023, 45 IPC cases were filed" and
# "Chapter 5 IPC" yield nothing.
_PREFIX = r"(?:§{1,2}|\bu/?ss?\.?|\bsec(?:tion|t)?s?\.?|\bss?\.|\bunder|\b(?:read\s+)?with)"

# Subsection contents are digits, one/two letters, or a roman numeral - never
# three arbitrary letters, otherwise "Section 302 (IPC)" parses as number
# "302(IPC)".
_SUB = r"(?:\s*\(\s*(?:\d{1,3}[a-z]?|[a-z]{1,2}|[ivx]{1,4})\s*\)){0,3}"
# (?<!\d) so a section number is never carved out of a longer number: the
# "1860" in "the Indian Penal Code, 1860 (IPC)" must not read as section 860.
_NUM_BARE = r"(?<!\d)\d{1,3}[A-Za-z]{0,2}" + _SUB
_NUM = r"(?<!\d)(?P<number>\d{1,3}[A-Za-z]{0,2}" + _SUB + r")"

# Between number and code: an optional comma, an optional "of/under/in", an
# optional "the", an optional opening bracket - "302 IPC", "302 of the IPC",
# "302, IPC", "302 (IPC)".
_JOIN = r"(?:\s*,)?\s*(?:of\s+|under\s+|in\s+)?(?:the\s+)?[(\[]?\s*"

_STATUTE_RE = re.compile(_PREFIX + r"\s*" + _NUM + _JOIN + _CODE, re.IGNORECASE)

# Indian pleadings cite lists constantly - "u/s 302, 307 and 34 IPC",
# "Sections 302/34 IPC", "S. 302 r/w 149 IPC" - and the code appears only
# once, at the end. Without this the FIRST numbers of every such list would
# be invisible to the temporal gate.
_SEP = r"(?:\s*(?:,|/|&|and|r/?w|read\s+with)\s*)"
_SECTION_LIST_RE = re.compile(
    _PREFIX + r"\s*(?P<numbers>" + _NUM_BARE + r"(?:" + _SEP + _NUM_BARE + r"){1,9})" + _JOIN + _CODE,
    re.IGNORECASE,
)
_SEP_SPLIT_RE = re.compile(_SEP, re.IGNORECASE)


def statute_pattern() -> re.Pattern:
    """Single "<section number> <code>" citation, groups (number, code)."""
    return _STATUTE_RE


def normalize_number(raw: str) -> str:
    """304b -> 304B, "103 (2)" -> 103(2), "376(2)(N)" -> 376(2)(n).
    Section-level letter suffixes are upper case, subsection letters and
    roman numerals lower case - the convention the statutes are printed in.
    Non-string input is coerced: external section lists happily ship 302 as
    an integer."""
    s = re.sub(r"\s+", "", "" if raw is None else str(raw))
    s = re.sub(r"\(([^)]*)\)", lambda m: "(" + m.group(1).lower() + ")", s)
    head, sep, rest = s.partition("(")
    return head.upper() + sep + rest


def extract_sections(text: str) -> list[SectionRef]:
    """Every section citation in `text`, code alias-resolved and upper-cased,
    number normalized, in order of first appearance, deduped."""
    if not text:
        return []
    hits: list[tuple[int, int, SectionRef]] = []
    seq = 0

    for m in _SECTION_LIST_RE.finditer(text):
        code = resolve_code(m.group("code"))
        if code is None:
            continue
        for number in _SEP_SPLIT_RE.split(m.group("numbers")):
            if not number.strip():
                continue
            hits.append((m.start(), seq, SectionRef(code, normalize_number(number))))
            seq += 1

    for m in _STATUTE_RE.finditer(text):
        code = resolve_code(m.group("code"))
        if code is None:
            continue
        hits.append((m.start(), seq, SectionRef(code, normalize_number(m.group("number")))))
        seq += 1

    hits.sort(key=lambda t: (t[0], t[1]))
    out: list[SectionRef] = []
    seen: set[SectionRef] = set()
    for _start, _seq, ref in hits:
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


_SAVINGS_PHRASE_RE = re.compile(r"\bsection\s+358\b", re.IGNORECASE)


def _cites_savings_clause(text: str, refs: list[SectionRef]) -> bool:
    """BNS s.358 is the repeal-and-savings clause itself. Text that discusses
    it is explaining WHY the IPC still applies to a pre-transition offence,
    so its BNS reference is legitimate, not a temporal error."""
    if any(ref.code == "BNS" and ref.base_number == "358" for ref in refs):
        return True
    return bool(_SAVINGS_PHRASE_RE.search(text))


def cross_code_flags(text: str, *, kind_dates: dict) -> list[str]:
    """THE TEMPORAL GATE PRIMITIVE. Every cited section is checked against the
    family that governs ITS OWN kind on these dates; [] means clean.

    Per-kind evaluation is the whole point: for a 2023 offence with an FIR
    registered in August 2024, "IPC s.302 read with BNSS s.173" is correct on
    both halves and must not be flagged, while "BNS s.103" in the same text
    must be.

    kind_dates = {"offence_date": date|None, "proceeding_started": date|None}.
    A citation whose family cannot be decided on the dates given (e.g. a
    substantive cite with no offence_date) is skipped, not guessed at.
    """
    offence_date = kind_dates.get("offence_date")
    proceeding_started = kind_dates.get("proceeding_started")

    refs = extract_sections(text)
    savings = _cites_savings_clause(text, refs)

    flags: list[str] = []
    for ref in refs:
        kind = CODE_KIND.get(ref.code)
        if kind is None:
            continue
        try:
            expected = governing_family(
                kind, offence_date=offence_date, proceeding_started=proceeding_started
            )
        except ValueError:
            continue
        cited = "old" if ref.code in OLD_CODES else "new"
        if cited == expected:
            continue
        if cited == "new":
            if savings:
                continue
            flag = FLAG_NEW_FOR_OLD
        else:
            flag = FLAG_OLD_FOR_NEW
        if flag not in flags:
            flags.append(flag)
    return flags


# --------------------------------------------------------------------------
# Section existence + old<->new mapping.
# --------------------------------------------------------------------------

def _data_rows(path: str | Path) -> list[dict]:
    """JSONL rows minus the "_comment" header rows. The resources files carry
    their documentation as a leading JSON object so the file stays valid
    JSONL for every other tool that reads it."""
    return [row for row in read_jsonl(path) if "_comment" not in row]


class SectionRegistry:
    """Does (code, number) name a real section? Seeded from the committed
    mapping resource and extendable at build time with fuller section lists
    (GSMS-B / Zenodo dumps) via load()/add()."""

    def __init__(self, sections: dict[str, set[str]] | None = None):
        self._sections: dict[str, set[str]] = {}
        for code, numbers in (sections or {}).items():
            for number in numbers:
                self.add(code, number)

    def add(self, code: str, number: str) -> None:
        code = (code or "").upper()
        number = normalize_number(number)
        if not code or not number:
            return
        bucket = self._sections.setdefault(code, set())
        bucket.add(number)
        # A section that has a subsection necessarily exists as a section, so
        # "3(5)" also registers "3".
        bucket.add(number.split("(", 1)[0])

    def add_rows(self, rows: list[dict]) -> "SectionRegistry":
        """Accepts mapping-schema rows (old/new pairs) and plain
        {"code", "section"} rows."""
        for row in rows:
            if row.get("code") and row.get("section"):
                self.add(row["code"], row["section"])
                continue
            if row.get("old_code") and row.get("old_section"):
                self.add(row["old_code"], row["old_section"])
            if row.get("new_code") and row.get("new_section"):
                self.add(row["new_code"], row["new_section"])
        return self

    @classmethod
    def load(cls, *paths: str | Path) -> "SectionRegistry":
        registry = cls()
        for path in paths or (IPC_BNS_MAP_PATH,):
            registry.add_rows(_data_rows(path))
        return registry

    def contains(self, ref: SectionRef) -> bool:
        numbers = self._sections.get((ref.code or "").upper())
        if not numbers:
            return False
        number = normalize_number(ref.number)
        return number in numbers or number.split("(", 1)[0] in numbers

    def __contains__(self, ref: SectionRef) -> bool:
        return self.contains(ref)

    def __len__(self) -> int:
        return sum(len(numbers) for numbers in self._sections.values())


class Mapping:
    """Old-code -> new-code section mapping, loaded from JSONL.

    Rows are old->new only. counterpart() also answers the reverse question
    (given a BNS section, which IPC section did it come from) but ONLY when
    the answer is unambiguous: IPC 499 and IPC 500 both map to BNS 356, so
    BNS 356 has no single counterpart and returns None rather than guessing.
    """

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self._by_old: dict[tuple[str, str], dict] = {}
        self._by_new: dict[tuple[str, str], list[dict]] = {}
        for i, row in enumerate(rows):
            self._validate(i, row)
            if row.get("old_code") and row.get("old_section"):
                key = self._key(row["old_code"], row["old_section"])
                if key in self._by_old:
                    raise ValueError(f"duplicate mapping row for {key[0]} {key[1]}: {row}")
                self._by_old[key] = row
            if row.get("new_code") and row.get("new_section"):
                self._by_new.setdefault(self._key(row["new_code"], row["new_section"]), []).append(row)

    @staticmethod
    def _key(code: str, number: str) -> tuple[str, str]:
        return (code.upper(), normalize_number(number))

    @staticmethod
    def _validate(i: int, row: dict) -> None:
        missing = {"old_code", "old_section", "new_code", "new_section", "kind", "verified_by"} - set(row)
        if missing:
            raise ValueError(f"mapping row {i} missing keys {sorted(missing)}: {row}")
        kind = row["kind"]
        if kind not in MAPPING_KINDS:
            raise ValueError(f"mapping row {i} has unknown kind {kind!r}: {row}")
        has_old = bool(row["old_code"]) and bool(row["old_section"])
        has_new = bool(row["new_code"]) and bool(row["new_section"])
        if kind == "new_offence" and (has_old or not has_new):
            raise ValueError(f"mapping row {i}: new_offence needs a null old side and a new side: {row}")
        if kind == "deleted" and (has_new or not has_old):
            raise ValueError(f"mapping row {i}: deleted needs an old side and a null new side: {row}")
        if kind in ("one_to_one", "changed") and not (has_old and has_new):
            raise ValueError(f"mapping row {i}: {kind} needs both sides populated: {row}")
        if has_old and row["old_code"].upper() not in OLD_CODES:
            raise ValueError(f"mapping row {i}: old_code {row['old_code']!r} is not an old code: {row}")
        if has_new and row["new_code"].upper() not in NEW_CODES:
            raise ValueError(f"mapping row {i}: new_code {row['new_code']!r} is not a new code: {row}")
        if row["verified_by"] is not None and not isinstance(row["verified_by"], str):
            raise ValueError(f"mapping row {i}: verified_by must be a string or null: {row}")

    @classmethod
    def load(cls, path: str | Path = IPC_BNS_MAP_PATH) -> "Mapping":
        return cls(_data_rows(path))

    def row(self, ref: SectionRef) -> dict | None:
        key = self._key(ref.code, ref.number)
        row = self._by_old.get(key)
        if row is not None:
            return row
        candidates = self._by_new.get(key, [])
        return candidates[0] if len(candidates) == 1 else None

    def counterpart(self, ref: SectionRef) -> SectionRef | None:
        key = self._key(ref.code, ref.number)
        row = self._by_old.get(key)
        if row is not None:
            if not row.get("new_code") or not row.get("new_section"):
                return None  # deleted, or otherwise unmapped
            return SectionRef(row["new_code"].upper(), normalize_number(row["new_section"]))
        candidates = self._by_new.get(key, [])
        if len(candidates) != 1:
            return None  # unmapped, or ambiguous - never guess
        row = candidates[0]
        if not row.get("old_code") or not row.get("old_section"):
            return None  # new_offence: nothing on the old side
        return SectionRef(row["old_code"].upper(), normalize_number(row["old_section"]))

    def verified_rows(self) -> list[dict]:
        return [row for row in self.rows if row.get("verified_by")]

    def unverified_rows(self) -> list[dict]:
        return [row for row in self.rows if not row.get("verified_by")]

    def require_verified(self, ref: SectionRef) -> dict:
        """The pipeline refuses to state an old<->new equivalence that no
        operator has signed off on - an unaudited mapping row is exactly the
        kind of confident-sounding legal error this dataset must not teach."""
        row = self.row(ref)
        if row is None:
            raise ValueError(f"no mapping row for {ref}")
        if not row.get("verified_by"):
            raise ValueError(
                f"mapping row {row.get('old_code')} {row.get('old_section')} -> "
                f"{row.get('new_code')} {row.get('new_section')} (kind={row.get('kind')}) "
                "is unverified: verified_by is null, an operator audit must sign it off first"
            )
        return row

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)
