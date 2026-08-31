"""The 50-example legal-accuracy review packet.

The dataset card makes a human read of 50 accepted examples a ship
prerequisite - "the only legal-accuracy check in this pipeline" - because every
automated gate scores FORM. A row can satisfy twelve gates and a judge and
still state the opposite of the right answer; one accepted transition row does
exactly that. This module is the part of that read a machine can do: draw a
sample that does not lose the small streams, and order the reading by where an
invented authority would show up.

Nothing here decides whether an answer is right. It decides what to read first.
"""
from __future__ import annotations

import hashlib
import re
from typing import Callable, Iterable, NamedTuple, Sequence

DEFAULT_SAMPLE = 50
DEFAULT_FLOOR = 3

# Section references. The PREFIX is case-insensitive; the SUFFIX deliberately
# is not. With a global re.I, "[A-Z]{1,3}" happily reads the "of" in
# "Section 302 of the Code" as a suffix and invents section "302OF".
#
# \b before the prefix matters too: without it the "s." alternative matches
# inside "cases. 5 SCC 123" and reports a section 5 nobody cited.
_PREFIX = r"\b(?i:sections?|sec\.?|s\.|u/s)\s*"
# Strict - what the ANSWER claims. A suffix must be attached to the number, so
# "Section 29 contains" is section 29 and not section "29CON".
_SECTION_STRICT = re.compile(_PREFIX + r"(\d+(?:-?[A-Z]{1,3})?)\b")
# Loose - what the SOURCE offers. Split so both readings are credited: a source
# saying "Section 302 A" answers an answer saying "302A", and one saying
# "Section 420 IPC" still answers a bare "420". Permissiveness here only ever
# SUPPRESSES a flag, which is the safe direction for a screen whose findings
# cost a human's attention.
_SECTION_LOOSE = re.compile(_PREFIX + r"(\d+)(\s*-?\s*[A-Z]{1,3})?\b")

# Reported citations: (2019) 5 SCC 123 / AIR 1978 SC 597 / 2021 SCC OnLine Del 1
_CITE = re.compile(
    r"(?:\(?(?:1[89]|20)\d{2}\)?\s*)?\(?\d{0,3}\)?\s*"
    r"(?:SCC(?:\s+OnLine)?|AIR|SCR|Cri\.?\s?L\.?J\.?|CriLJ|ILR)\s+[A-Za-z]{0,4}\s*\d+",
    re.I,
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text).upper()


class Unsourced(NamedTuple):
    """References the answer makes that its source never made.

    NOT a list of errors. Settled background authority is cited from memory by
    every competent lawyer, and on the transition stream naming the savings
    provisions is the task. This is where an invented authority WOULD appear,
    which is a reading order, not a verdict.
    """

    sections: tuple[str, ...]
    citations: tuple[str, ...]


def unsourced_references(answer: str, source: str) -> Unsourced:
    answer, source = answer or "", source or ""

    have: set[str] = set()
    for number, suffix in _SECTION_LOOSE.findall(source):
        have.add(_norm(number))
        if suffix:
            have.add(_norm(number + suffix))
    sections = {_norm(m) for m in _SECTION_STRICT.findall(answer)} - have

    cited = {_norm(m) for m in _CITE.findall(source)}
    citations = {_norm(m) for m in _CITE.findall(answer)} - cited

    return Unsourced(tuple(sorted(sections)), tuple(sorted(citations)))


def _digest(salt: str, value: str) -> str:
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()


def stratified_sample(
    rows: Sequence[dict],
    *,
    n: int = DEFAULT_SAMPLE,
    floor: int = DEFAULT_FLOOR,
    key: Callable[[dict], str],
    ident: Callable[[dict], str] = lambda row: row["task_id"],
    salt: str = "",
) -> list[dict]:
    """`n` rows, with at least `floor` from every cell that has that many.

    A proportional draw is the wrong instrument here. `transition` had 5
    accepted rows against `irac_analysis`' several hundred, so proportionality
    returns it zero or one times - and it is the stream carrying the most legal
    risk and the only one with ground truth to check against. The floor buys
    that coverage out of the largest cells, which can spare it.

    Deterministic in `salt`: the same salt re-draws the same 50 rows, so a
    re-render after the corpus grows is comparable to the last read rather than
    a fresh sample nobody has looked at.
    """
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(key(row), []).append(row)
    for cell, members in grouped.items():
        members.sort(key=lambda row: _digest(salt, ident(row)))

    total = sum(len(members) for members in grouped.values())
    order = sorted(grouped)
    if n >= total:
        return [row for cell in order for row in grouped[cell]]

    take = {cell: min(floor, len(grouped[cell])) for cell in order}
    # More cells than the sample can floor: spend what there is on the biggest,
    # rather than returning more rows than were asked for.
    while sum(take.values()) > n:
        biggest = max(order, key=lambda cell: (take[cell], len(grouped[cell])))
        take[biggest] -= 1

    spare = {cell: len(grouped[cell]) - take[cell] for cell in order}
    remaining = n - sum(take.values())
    pool = sum(spare.values())
    if remaining and pool:
        # Largest remainder, so the shares add up to exactly `remaining` and
        # the rounding does not quietly favour whichever cell sorts first.
        exact = {cell: remaining * spare[cell] / pool for cell in order}
        whole = {cell: int(exact[cell]) for cell in order}
        for cell in sorted(order, key=lambda c: (-(exact[c] - whole[c]), c)):
            if sum(whole.values()) >= remaining:
                break
            whole[cell] += 1
        for cell in order:
            take[cell] += whole[cell]

    return [row for cell in order for row in grouped[cell][: take[cell]]]
