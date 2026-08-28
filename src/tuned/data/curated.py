"""Build the ~1,700-row curated C1 stream: filtered/reformatted rows pulled
directly from public datasets, needing zero teacher-model generation.

Same row contract as replay.py: JSONL rows {"messages": [user, assistant],
"_prov": {...}}, assistant content is think_open + trace + think_close +
answer for reasoning rows, or the byte-exact empty block
(think_open + "\n\n" + think_close) + answer for non-reasoning rows.

Reuses replay.py's shared quality-filter primitives (is_refusal, has_emoji,
has_markup, sha256_hex, empty_think) by import - they were already public
there, so no edit to replay.py was needed. The row-building and combined-
reject-reason helpers below are curated.py's own (small, mechanical, and
not the "filter helpers" the brief's import rule is about) since this
module's per-slice filters differ enough from replay.py's (per-field
length bands, a code-era contamination regex, per-task licensing) that
sharing replay's private combinators would be a worse fit than replay's
own five converters share with each other.

Per-source converters are pure: given a raw row plus the model's think
tags, each returns (row, None) on acceptance or (None, reason) on
rejection, so build_curated can tally per-slice reject reasons exactly
like build_replay does. Cross-row state (exact dedup on
sha256(user content), counting toward the per-slice target) lives in
build_curated, not the converters.

datasets/pyarrow imports are lazy (inside the streaming helpers, never at
module import time).

Build:  python -m tuned.data.curated --config data/configs/data_law_v1.yaml
        [--out PATH] [--counts 800,600,300]
"""

import re
from pathlib import Path

from tuned.data.replay import empty_think, has_emoji, has_markup, is_refusal, sha256_hex

SLICE_ORDER = ("predex_prediction", "aalap_safe", "pi169_audited")
DEFAULT_COUNTS = (800, 600, 300)

# Per-task licensing for opennyaiorg/aalap_instruction_dataset (HF card:
# top-level license "other", size 21,178 train / 1,094 test, gated="auto";
# confirmed columns input_text/system_prompt/user_prompt/output_text/task/
# combined_input_prompt via the HF datasets-server dataset_info metadata,
# which is readable without a token even though the actual rows are
# gated). The card's own per-task table:
#
#   Issue Generation                 CC0-1.0
#   Argument Generation              CC0-1.0
#   Event Timeline                   CC0-1.0
#   Combine Event Timeline           CC0-1.0
#   Legalbench                       Other          <- unestablished, EXCLUDED
#   Statute Ingredients              CC0-1.0
#   Summary Generation               CC0-1.0
#   Legal Open ORCA                  MIT
#   Contract Clause Generation       cc-by-nc-4.0   <- known-NC, EXCLUDED (brief)
#   Legal NIv2 MCQ                   Apache 2.0
#   Constitution General Knowledge   Apache 2.0
#   Incomplete Instructions          CC0-1.0
#   General Aalap                    CC0-1.0
#
# Every task NOT in this dict is excluded by construction (allowlist
# lookup fails closed) - this covers both tasks with a known-NC license
# (contract_clause_generation) and any task string this table doesn't
# recognise at all (unknown -> exclude, per the brief).
#
# Keys are the VERIFIED runtime task FAMILIES, not the card's Title Case:
# the flagged follow-up above fired on 2026-08-29 exactly as written -
# every row rejected "task_not_allowlisted", shortfall raised at 0/600.
# A 3,000-row authenticated stream showed the real `task` values are
# snake_case with a `___variant` suffix (e.g. argument_generation___
# petitioner, incomplete_instructions___opennyai_legal_tasks, and the
# dataset's own spelling general_alap), so the lookup takes the prefix
# before "___" and this table maps each family to the card's license.
AALAP_SAFE_TASKS = {
    "issue_generation": "CC0-1.0",
    "argument_generation": "CC0-1.0",
    "event_timeline": "CC0-1.0",
    "combine_event_timeline": "CC0-1.0",
    "statute_ingredients": "CC0-1.0",
    "summary_generation": "CC0-1.0",
    "legal_open_orca": "MIT",
    "legal_niv2_mcq": "Apache-2.0",
    "constitution_general_knowledge": "Apache-2.0",
    "incomplete_instructions": "CC0-1.0",
    "general_alap": "CC0-1.0",
}

# 169Pi/indian_law predates the 2024 IPC->BNS/BNSS/BSA transition (Apache-
# 2.0, ~50M synthetic tokens); any BNS/BNSS/BSA mention in its `response`
# is new-code contamination from an old-code corpus, not a real citation.
_NEW_CODE_RE = re.compile(r"\bBNSS?\b|\bBSA\b")


def _row(user: str, assistant_content: str, *, source: str, license_: str, native_id: str | None, reasoning: bool) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant_content},
        ],
        "_prov": {
            "source": source,
            "license": license_,
            "native_id": native_id,
            "reasoning": reasoning,
        },
    }


def _quality_reject(*texts: str) -> str | None:
    """Shared tail-end checks every converter below runs on its extracted
    user/answer/trace text. Returns a reject reason, or None if clean. No
    ASCII-ratio check (unlike replay.py's _common_reject): every slice here
    is Indian-legal text where Devanagari terms are legitimate content."""
    if any(not t for t in texts):
        return "empty"
    if any(is_refusal(t) for t in texts):
        return "refusal"
    if any(has_emoji(t) for t in texts):
        return "emoji"
    if has_markup(*texts):
        return "markup"
    return None


# --------------------------------------------------------------------------
# Per-source converters.
# --------------------------------------------------------------------------

def predex_prediction_row(raw: dict, think_open: str, think_close: str) -> tuple[dict | None, str | None]:
    """L-NLProc/PredEx_Instruction-Tuning_Pred-Exp - Input (case facts, as
    the dataset frames it) is the user turn, Output (the REAL court
    reasoning/decision explanation) is the visible answer behind an EMPTY
    think block. These are non-reasoning rows by design: real judicial
    text is the answer, never a fabricated first-person think trace over
    it - that synthesis is the teacher's job elsewhere in the pipeline.
    """
    user = (raw.get("Input") or "").strip()
    answer = (raw.get("Output") or "").strip()
    if not (user and answer):
        return None, "empty"

    # Same 24,000-char proxy replay.py's ot_row uses for the 8192-token
    # assembly-time gate - PredEx Output ranges up to ~196k chars per the
    # HF card, and this pipeline drops overlong rows rather than truncate.
    if len(user) + len(answer) >= 24_000:
        return None, "too_long"

    reason = _quality_reject(user, answer)
    if reason:
        return None, reason

    content = empty_think(think_open, think_close) + answer
    if has_markup(user, content):
        return None, "markup"

    return _row(
        user, content,
        source="L-NLProc/PredEx_Instruction-Tuning_Pred-Exp",
        license_="Apache-2.0",
        native_id=((raw.get("Case Name") or "").strip() or None),
        reasoning=False,
    ), None


def aalap_safe_row(raw: dict, think_open: str, think_close: str) -> tuple[dict | None, str | None]:
    """opennyaiorg/aalap_instruction_dataset - per-task mixed licensing.
    Only tasks on AALAP_SAFE_TASKS are admitted; everything else (the
    known-NC Contract Clause Generation task, the unestablished-license
    Legalbench task, and any task string this table doesn't recognise at
    all) is rejected by the allowlist lookup failing closed.

    user = combined_input_prompt (the dataset's own ready-to-use
    instruction framing - inferred from the field name to be
    system_prompt + user_prompt + input_text combined, not independently
    confirmed since the dataset is gated; "as the dataset frames it" per
    the brief's own PredEx language), falling back to user_prompt if that
    field is empty or the inference turns out wrong.
    Empty-think formatting unless the row carries a genuine reasoning
    field (the confirmed real schema - input_text/system_prompt/
    user_prompt/output_text/task/combined_input_prompt - has none, so this
    branch is defensive/forward-compatible rather than expected to fire).
    """
    # The runtime task string carries a ___variant suffix; the family
    # prefix is what the license table is keyed on (see AALAP_SAFE_TASKS).
    task = (raw.get("task") or "").split("___", 1)[0]
    license_ = AALAP_SAFE_TASKS.get(task)
    if license_ is None:
        return None, "task_not_allowlisted"

    user = (raw.get("combined_input_prompt") or raw.get("user_prompt") or "").strip()
    answer = (raw.get("output_text") or "").strip()
    if not (user and answer):
        return None, "empty"

    reason = _quality_reject(user, answer)
    if reason:
        return None, reason

    trace = (raw.get("reasoning") or "").strip()
    if trace:
        content = f"{think_open}{trace}{think_close}{answer}"
        reasoning = True
    else:
        content = empty_think(think_open, think_close) + answer
        reasoning = False

    if has_markup(user, content):
        return None, "markup"

    return _row(
        user, content,
        source="opennyaiorg/aalap_instruction_dataset",
        license_=license_,
        native_id=None,
        reasoning=reasoning,
    ), None


def pi169_audited_row(raw: dict, think_open: str, think_close: str) -> tuple[dict | None, str | None]:
    """169Pi/indian_law (Apache-2.0, prompt/complex_cot/response) - REAL
    reasoning rows: complex_cot is the think trace, response is the
    answer. "Audited" slice: tighter length bands than the replay
    reasoning slices, plus a hard drop on any BNS/BNSS/BSA mention in the
    response (old-code dataset; a new-code claim there is code-era
    contamination, not a legitimate citation).
    """
    user = (raw.get("prompt") or "").strip()
    think = (raw.get("complex_cot") or "").strip()
    answer = (raw.get("response") or "").strip()
    if not (user and think and answer):
        return None, "empty"

    if not (400 <= len(think) <= 8_000):
        return None, "think_length"
    if not (100 <= len(answer) <= 4_000):
        return None, "answer_length"

    if _NEW_CODE_RE.search(answer):
        return None, "new_code_contamination"

    reason = _quality_reject(user, think, answer)
    if reason:
        return None, reason

    content = f"{think_open}{think}{think_close}{answer}"
    if has_markup(user, content):
        return None, "markup"

    return _row(
        user, content,
        source="169Pi/indian_law",
        license_="Apache-2.0",
        native_id=None,
        reasoning=True,
    ), None


_CONVERTERS = {
    "predex_prediction": predex_prediction_row,
    "aalap_safe": aalap_safe_row,
    "pi169_audited": pi169_audited_row,
}


# --------------------------------------------------------------------------
# Assembly.
# --------------------------------------------------------------------------

def build_curated(cfg, counts=DEFAULT_COUNTS, rows_by_source: dict | None = None, out_path=None) -> dict:
    """Iterate each source until its target count is met, applying the
    slice's converter + per-slice exact dedup on sha256(user content).
    Raises RuntimeError (naming the slice and shortfall) if a source
    exhausts before its count is met. Returns per-slice accept/reject-
    reason counts. Mirrors build_replay exactly."""
    from tuned.data.jsonl import write_jsonl
    from tuned.data.paths import build_paths

    think_open, think_close = cfg.think_open, cfg.think_close

    if out_path is None:
        out_path = build_paths(cfg.build.workdir).streams_dir / "curated_c1.jsonl"
    out_path = Path(out_path)

    if rows_by_source is None:
        rows_by_source = _real_rows_by_source()

    stats: dict = {}
    all_rows: list[dict] = []

    for slice_name, n in zip(SLICE_ORDER, counts):
        if n <= 0:
            stats[slice_name] = {"accepted": 0, "target": n, "rejects": {}}
            continue

        converter = _CONVERTERS[slice_name]
        source_iter = iter(rows_by_source[slice_name])
        seen: set[str] = set()
        accepted = 0
        rejects: dict[str, int] = {}

        for raw in source_iter:
            if accepted >= n:
                break
            row, reason = converter(raw, think_open, think_close)
            if row is None:
                rejects[reason] = rejects.get(reason, 0) + 1
                continue
            key = sha256_hex(row["messages"][0]["content"])
            if key in seen:
                rejects["duplicate"] = rejects.get("duplicate", 0) + 1
                continue
            seen.add(key)
            all_rows.append(row)
            accepted += 1

        if accepted < n:
            raise RuntimeError(
                f"curated build: source {slice_name!r} exhausted at {accepted} of {n} rows "
                f"(short by {n - accepted})"
            )

        stats[slice_name] = {"accepted": accepted, "target": n, "rejects": rejects}

    written = write_jsonl(out_path, all_rows)
    stats["total"] = written
    stats["out_path"] = str(out_path)
    return stats


def parse_counts(s: str) -> tuple[int, ...]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != len(SLICE_ORDER):
        raise ValueError(
            f"--counts must have {len(SLICE_ORDER)} comma-separated ints "
            f"({','.join(SLICE_ORDER)}), got {len(parts)}: {s!r}"
        )
    return tuple(int(p) for p in parts)


def _stream(**load_kwargs):
    from datasets import load_dataset

    ds = load_dataset(streaming=True, **load_kwargs)
    for row in ds:
        yield row


def _real_rows_by_source() -> dict:
    import os

    hf_token = os.environ.get("HF_TOKEN")
    return {
        "predex_prediction": _stream(path="L-NLProc/PredEx_Instruction-Tuning_Pred-Exp", split="train"),
        # gated="auto" on HF - needs an authorized HF_TOKEN with the
        # dataset's terms accepted.
        "aalap_safe": _stream(path="opennyaiorg/aalap_instruction_dataset", split="train", token=hf_token),
        "pi169_audited": _stream(path="169Pi/indian_law", split="train"),
    }


if __name__ == "__main__":
    import argparse
    import os
    import sys

    from tuned.data.config import load_build_config

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="data/configs/data_law_v1.yaml")
    p.add_argument("--out", default=None)
    p.add_argument("--counts", default=None, help="predex,aalap,pi169")
    args = p.parse_args()

    cfg = load_build_config(args.config)
    counts = parse_counts(args.counts) if args.counts else DEFAULT_COUNTS

    stats = build_curated(cfg, counts, out_path=args.out)

    print(f"{'slice':<20}{'accepted':>10}{'target':>10}")
    for name, n in zip(SLICE_ORDER, counts):
        s = stats.get(name, {"accepted": 0, "rejects": {}})
        print(f"{name:<20}{s['accepted']:>10}{n:>10}")
        for reason, cnt in sorted(s.get("rejects", {}).items()):
            print(f"    reject[{reason}]: {cnt}")
    print(f"wrote {stats['total']} rows -> {stats['out_path']}")

    # Same reasoning as smoke.py/replay.py: abandoned streaming iterators
    # can leave non-daemon datasets/hf-xet threads that wedge interpreter
    # shutdown after all output is written. Skip shutdown entirely.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
