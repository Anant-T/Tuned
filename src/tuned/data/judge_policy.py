"""Pure dual-judge decision policy.

No store, no providers, no calibration. judge.py and eval_matched.py both
apply this matrix so a later edit cannot silently fork the two paths.
"""

from collections.abc import Sequence

PASS, BORDERLINE, FAIL = "pass", "borderline", "fail"
PASS_MIN = 4
FAIL_MAX = 2


def slot_verdict(grounding: int, validity: int, coverage: int) -> str:
    score = min(int(grounding), int(validity), int(coverage))
    if score >= PASS_MIN:
        return PASS
    if score <= FAIL_MAX:
        return FAIL
    return BORDERLINE


def decide(verdicts: Sequence[str], *, already_regenerated: bool) -> str:
    """What to do with a set of judge verdicts. Pure; the whole matrix.

    Two judges:
      pass  + pass              -> accept
      fail  + fail              -> reject
      exactly one pass          -> tiebreak
      no pass, some borderline  -> ONE regeneration, then reject

    Three (a tiebreak was run): the tiebreak decides. Its own borderline
    goes down the same one-regeneration path.
    """
    if not verdicts:
        raise ValueError("decide() needs at least one verdict")
    if len(verdicts) >= 3:
        final = verdicts[2]
        if final == PASS:
            return "accept"
        if final == FAIL:
            return "reject"
        return "reject" if already_regenerated else "regenerate"

    passes = sum(1 for v in verdicts if v == PASS)
    if passes == len(verdicts):
        return "accept"
    if all(v == FAIL for v in verdicts):
        return "reject"
    if passes:
        return "tiebreak"
    return "reject" if already_regenerated else "regenerate"
