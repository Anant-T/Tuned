"""Build the seq-ceiling probe dataset by concatenating smoke examples.

The training pipeline does not pack sequences, so a VRAM probe at seq N only
measures N if examples actually tokenize past N - otherwise the probe is a
false green. Rows from an existing smoke jsonl are concatenated into
multi-turn conversations until each exceeds a character threshold of
chars_per_token * target_tokens (>= 6 chars/token is conservative for
English/reasoning text; overshoot is free because the trainer truncates at
max_length, undershoot is the failure mode). One probe file serves any lane
with max_seq_length <= target_tokens.

Build:  python -m tuned.data.probe --src data/smoke_v1.jsonl --out data/probe_6k.jsonl
"""

import json
from pathlib import Path


def message_chars(example: dict) -> int:
    return sum(len(m.get("content", "")) for m in example.get("messages", []))


def build_probe(
    out_path: str | Path,
    src_path: str | Path,
    n: int = 8,
    target_tokens: int = 6144,
    chars_per_token: int = 6,
) -> int:
    """Concatenate smoke rows into n long multi-turn examples, each with
    >= chars_per_token * target_tokens characters of message content."""
    min_chars = chars_per_token * target_tokens
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with Path(src_path).open(encoding="utf-8") as src, out_path.open(
        "w", encoding="utf-8"
    ) as out:
        messages: list[dict] = []
        chars = 0
        for line in src:
            if written >= n:
                break
            row = json.loads(line)
            messages.extend(row["messages"])
            chars += message_chars(row)
            if chars >= min_chars:
                out.write(json.dumps({"messages": messages}) + "\n")
                written += 1
                messages, chars = [], 0
    if written < n:
        raise RuntimeError(
            f"only built {written} of {n} probe examples - source {src_path} "
            f"too small for {min_chars} chars each"
        )
    return written


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/smoke_v1.jsonl")
    p.add_argument("--out", default="data/probe_6k.jsonl")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--target-tokens", type=int, default=6144)
    args = p.parse_args()

    count = build_probe(args.out, args.src, n=args.n, target_tokens=args.target_tokens)
    print(f"wrote {count} probe examples to {args.out}")
