"""Build the ~4,320-row general-replay stream: filtered/reformatted rows
pulled from public datasets, needing zero teacher-model generation.

Output contract matches smoke.py's format_example semantics: JSONL rows
{"messages": [user, assistant]} where assistant content is
think_open + trace + think_close + answer for reasoning rows, or the
byte-exact empty block (think_open + "\n\n" + think_close) + answer for
non-reasoning rows. Every row additionally carries a "_prov" block
(source/license/native_id/reasoning) that the downstream assembler strips
before the row reaches the trainer - it stays here so provenance survives
the curation pipeline for audit.

Per-source converters are pure: given a raw row (whatever schema the
upstream HF dataset happens to use) plus the model's think tags, each
returns (row, None) on acceptance or (None, reason) on rejection, so
build_replay can tally per-slice reject reasons without re-deriving them.
Cross-row state (exact dedup on sha256(user content), counting toward the
per-slice target) lives in build_replay, not the converters.

datasets/pyarrow imports are lazy (inside the streaming helpers, never at
module import time) so `import tuned.data.replay` never touches the
network - the same discipline smoke.py uses.

Build:  python -m tuned.data.replay --config data/configs/data_law_v1.yaml
        [--out PATH] [--counts 2520,600,600,300,300]
"""

import hashlib
import re
from pathlib import Path
from tuned.data.paths import DEFAULT_CONFIG

SLICE_ORDER = ("ot_reasoning", "nemotron_reasoning", "smoltalk_nothink", "legal_qa_empty", "wildchat_prof")
DEFAULT_COUNTS = (2520, 600, 600, 300, 300)

_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")

_REFUSAL_PATTERNS = (
    "i can't",
    "i cannot",
    "i'm sorry",
    "i am sorry",
    "as an ai",
    "i'm unable",
    "against my guidelines",
)

_ADVISORY_RE = re.compile(r"\b(what|how|should|explain|draft|write a)\b", re.IGNORECASE)


# --------------------------------------------------------------------------
# Pure quality-filter primitives (unit-tested directly).
# --------------------------------------------------------------------------

def is_refusal(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(p in t for p in _REFUSAL_PATTERNS)


def has_emoji(text: str) -> bool:
    return bool(_EMOJI_RE.search(text or ""))


def is_single_turn(messages: list[dict]) -> bool:
    """messages is our normalized [{"role": "user"|"assistant"|..., "content": str}]
    shape (converters map whatever the source calls its turns into this
    before calling here). Exactly one user + one assistant turn required;
    any other role present (e.g. "system") is ignored."""
    users = sum(1 for m in messages if m.get("role") == "user")
    assistants = sum(1 for m in messages if m.get("role") == "assistant")
    return users == 1 and assistants == 1


def has_markup(*texts: str) -> bool:
    return any("<|" in (t or "") for t in texts)


def is_advisory_question(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return bool(_ADVISORY_RE.search(t))


def is_mostly_ascii(text: str, threshold: float = 0.10) -> bool:
    t = text or ""
    if not t:
        return True
    non_ascii = sum(1 for c in t if ord(c) > 127)
    return (non_ascii / len(t)) <= threshold


def answer_length_ok(text: str, lo: int = 200, hi: int = 1600) -> bool:
    return lo <= len(text or "") <= hi


def sha256_hex(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def empty_think(think_open: str, think_close: str) -> str:
    """The byte-exact empty reasoning block prepended to non-reasoning
    answers, so the model sees the same <think>...</think> scaffold shape
    regardless of whether this row actually reasoned."""
    return f"{think_open}\n\n{think_close}"


def assembly_row(user: str, assistant_content: str, *, source: str, license_: str, native_id: str | None, reasoning: bool) -> dict:
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


def _common_reject(*texts: str, ascii_check: bool = True) -> str | None:
    """Shared tail-end checks every converter runs on its extracted
    user/answer/trace text. Returns a reject reason, or None if clean."""
    if any(not t for t in texts):
        return "empty"
    if any(is_refusal(t) for t in texts):
        return "refusal"
    if any(has_emoji(t) for t in texts):
        return "emoji"
    if has_markup(*texts):
        return "markup"
    if ascii_check and any(not is_mostly_ascii(t) for t in texts):
        return "non_ascii"
    return None


# --------------------------------------------------------------------------
# Per-source converters.
# --------------------------------------------------------------------------

def ot_row(raw: dict, think_open: str, think_close: str) -> tuple[dict | None, str | None]:
    """open-thoughts/OpenThoughts-114k - same conversations->problem/reasoning/
    solution parsing as smoke.py, plus a total-chars<24000 proxy for the
    8192-token assembly-time gate."""
    conversations = raw.get("conversations") or []
    messages = [
        {"role": "user" if t.get("from") == "user" else "assistant", "content": (t.get("value") or "").strip()}
        for t in conversations
        if t.get("from") in ("user", "assistant")
    ]
    if not is_single_turn(messages):
        return None, "not_single_turn"

    user = next(m["content"] for m in messages if m["role"] == "user")
    assistant_raw = next(m["content"] for m in messages if m["role"] == "assistant")

    if "<|end_of_thought|>" not in assistant_raw:
        return None, "no_end_of_thought"

    parts = assistant_raw.split("<|end_of_thought|>", 1)
    reasoning = parts[0].replace("<|begin_of_thought|>", "").strip()
    solution = parts[1].replace("<|begin_of_solution|>", "").replace("<|end_of_solution|>", "").strip()

    if not (user and reasoning and solution):
        return None, "empty"

    if len(user) + len(reasoning) + len(solution) >= 24_000:
        return None, "too_long"

    reason = _common_reject(user, reasoning, solution)
    if reason:
        return None, reason

    content = f"{think_open}{reasoning}{think_close}{solution}"
    if has_markup(user, content):
        return None, "markup"
    return assembly_row(user, content, source="open-thoughts/OpenThoughts-114k", license_="Apache-2.0", native_id=None, reasoning=True), None


def nemotron_row(raw: dict, think_open: str, think_close: str) -> tuple[dict | None, str | None]:
    """nvidia/Nemotron-Post-Training-Dataset-v2, chat split.

    Schema (verified at runtime - see task-5-report.md): uuid, license,
    generator, version, category, reasoning, messages ([{role, content}],
    role in {system, user, assistant}). reasoning is the literal string
    "on"/"off"; license is the literal string "CC BY 4.0" for the clean
    subset (space-separated, no hyphen) - anything else (incl. the
    CC-BY-SA StackOverflow-derived rows and the ODC-BY WildChat-derived
    rows folded into this dataset) is allowlisted OUT here, not guessed
    at. Reasoning-on assistant content is a single <think>...</think>
    block followed by the answer.
    """
    lic = re.sub(r"[\s-]+", " ", str(raw.get("license") or "").strip().lower())
    if lic != "cc by 4.0":
        return None, "license"

    if str(raw.get("reasoning") or "").strip().lower() != "on":
        return None, "not_reasoning_on"

    messages = [
        {"role": m.get("role"), "content": (m.get("content") or "").strip()}
        for m in (raw.get("messages") or [])
        if m.get("role") in ("user", "assistant")
    ]
    if not is_single_turn(messages):
        return None, "not_single_turn"

    user = next(m["content"] for m in messages if m["role"] == "user")
    assistant_raw = next(m["content"] for m in messages if m["role"] == "assistant")

    if "<think>" not in assistant_raw or "</think>" not in assistant_raw:
        return None, "no_think_markers"

    head, _, tail = assistant_raw.partition("</think>")
    trace = head.split("<think>", 1)[-1].strip()
    answer = tail.strip()

    if not (user and trace and answer):
        return None, "empty"

    reason = _common_reject(user, trace, answer)
    if reason:
        return None, reason

    content = f"{think_open}{trace}{think_close}{answer}"
    if has_markup(user, content):
        return None, "markup"
    return assembly_row(
        user, content,
        source="nvidia/Nemotron-Post-Training-Dataset-v2",
        license_="CC-BY-4.0",
        native_id=raw.get("uuid"),
        reasoning=True,
    ), None


def smoltalk_row(raw: dict, think_open: str, think_close: str) -> tuple[dict | None, str | None]:
    """HuggingFaceTB/smoltalk2, SFT config, *_no_think splits. Rows are
    already {"messages": [{"role","content"}, ...]} - empty-think slice."""
    messages = [
        {"role": m.get("role"), "content": (m.get("content") or "").strip()}
        for m in (raw.get("messages") or [])
        if m.get("role") in ("user", "assistant")
    ]
    if not is_single_turn(messages):
        return None, "not_single_turn"

    user = next(m["content"] for m in messages if m["role"] == "user")
    answer = next(m["content"] for m in messages if m["role"] == "assistant")

    if not answer_length_ok(answer):
        return None, "length"

    reason = _common_reject(user, answer)
    if reason:
        return None, reason

    content = empty_think(think_open, think_close) + answer
    if has_markup(user, content):
        return None, "markup"
    sub = raw.get("source") or "smoltalk2"
    return assembly_row(
        user, content,
        source=f"HuggingFaceTB/smoltalk2:{sub}",
        license_="Apache-2.0",
        native_id=None,
        reasoning=False,
    ), None


def legal_qa_row(raw: dict, think_open: str, think_close: str) -> tuple[dict | None, str | None]:
    """GSMS-B/Indian-Legal-QA-BNS-BNSS-BSA - flat question/answer rows, not
    a conversation list, so no is_single_turn check (trivially single-turn
    by construction). The anti-topic-shortcut slice: keep the ASCII filter
    OFF - Indian statute names and Devanagari terms are legitimate here."""
    user = (raw.get("question") or "").strip()
    answer = (raw.get("answer") or "").strip()

    if not (user and answer):
        return None, "empty"

    if not answer_length_ok(answer):
        return None, "length"

    reason = _common_reject(user, answer, ascii_check=False)
    if reason:
        return None, reason

    content = empty_think(think_open, think_close) + answer
    if has_markup(user, content):
        return None, "markup"
    return assembly_row(
        user, content,
        source="GSMS-B/Indian-Legal-QA-BNS-BNSS-BSA",
        license_="Apache-2.0",
        native_id=raw.get("chunk_id"),
        reasoning=False,
    ), None


def wildchat_row(raw: dict, think_open: str, think_close: str) -> tuple[dict | None, str | None]:
    """allenai/WildChat-4.8M - English, single-turn, advisory/professional
    register only. conversation entries carry role/content plus a lot of
    per-turn metadata we ignore; toxic/redacted are top-level booleans."""
    if raw.get("language") != "English":
        return None, "language"
    if raw.get("toxic") or raw.get("redacted"):
        return None, "flagged"

    messages = [
        {"role": m.get("role"), "content": (m.get("content") or "").strip()}
        for m in (raw.get("conversation") or [])
        if m.get("role") in ("user", "assistant")
    ]
    if not is_single_turn(messages):
        return None, "not_single_turn"

    user = next(m["content"] for m in messages if m["role"] == "user")
    answer = next(m["content"] for m in messages if m["role"] == "assistant")

    if not is_advisory_question(user):
        return None, "not_advisory"

    if not answer_length_ok(answer):
        return None, "length"

    reason = _common_reject(user, answer)
    if reason:
        return None, reason

    content = empty_think(think_open, think_close) + answer
    if has_markup(user, content):
        return None, "markup"
    return assembly_row(
        user, content,
        source="allenai/WildChat-4.8M",
        license_="ODC-BY",
        native_id=raw.get("conversation_hash"),
        reasoning=False,
    ), None


_CONVERTERS = {
    "ot_reasoning": ot_row,
    "nemotron_reasoning": nemotron_row,
    "smoltalk_nothink": smoltalk_row,
    "legal_qa_empty": legal_qa_row,
    "wildchat_prof": wildchat_row,
}


# --------------------------------------------------------------------------
# Assembly.
# --------------------------------------------------------------------------

def build_replay(cfg, counts=DEFAULT_COUNTS, rows_by_source: dict | None = None, out_path=None) -> dict:
    """Iterate each source until its target count is met, applying the
    slice's converter + per-slice exact dedup on sha256(user content).
    Raises RuntimeError (naming the slice and shortfall) if a source
    exhausts before its count is met. Returns per-slice accept/reject-
    reason counts."""
    from tuned.data.jsonl import write_jsonl
    from tuned.data.paths import build_paths

    think_open, think_close = cfg.think_open, cfg.think_close

    if out_path is None:
        out_path = build_paths(cfg.build.workdir).streams_dir / "replay.jsonl"
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
                f"replay build: source {slice_name!r} exhausted at {accepted} of {n} rows "
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
        "ot_reasoning": _stream(path="open-thoughts/OpenThoughts-114k", split="train"),
        "nemotron_reasoning": _stream(
            path="nvidia/Nemotron-Post-Training-Dataset-v2", split="chat", token=hf_token
        ),
        "smoltalk_nothink": _chain(
            _stream(path="HuggingFaceTB/smoltalk2", name="SFT", split="smoltalk_smollm3_smol_magpie_ultra_no_think"),
            _stream(path="HuggingFaceTB/smoltalk2", name="SFT", split="OpenHermes_2.5_no_think"),
        ),
        "legal_qa_empty": _stream(path="GSMS-B/Indian-Legal-QA-BNS-BNSS-BSA", split="train"),
        "wildchat_prof": _stream(path="allenai/WildChat-4.8M", split="train"),
    }


def _chain(*iterators):
    for it in iterators:
        yield from it


if __name__ == "__main__":
    import argparse
    import os
    import sys

    from tuned.data.config import load_build_config

    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--out", default=None)
    p.add_argument("--counts", default=None, help="ot,nemotron,smoltalk,legal,wildchat")
    args = p.parse_args()

    cfg = load_build_config(args.config)
    counts = parse_counts(args.counts) if args.counts else DEFAULT_COUNTS

    stats = build_replay(cfg, counts, out_path=args.out)

    print(f"{'slice':<20}{'accepted':>10}{'target':>10}")
    for name, n in zip(SLICE_ORDER, counts):
        s = stats.get(name, {"accepted": 0, "rejects": {}})
        print(f"{name:<20}{s['accepted']:>10}{n:>10}")
        for reason, cnt in sorted(s.get("rejects", {}).items()):
            print(f"    reject[{reason}]: {cnt}")
    print(f"wrote {stats['total']} rows -> {stats['out_path']}")

    # Same reasoning as smoke.py: abandoned streaming iterators can leave
    # non-daemon datasets/hf-xet threads that wedge interpreter shutdown
    # after all output is written. Skip shutdown entirely.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
