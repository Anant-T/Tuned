"""Build the ~1k-example smoke dataset from OpenThoughts-114k (Apache-2.0).

Smoke-run purpose is plumbing validation, so assistant content is plain
"reasoning + solution" text; thinking-template tag fidelity is handled in the
main data pipeline, not here.
"""

import json
from pathlib import Path


def format_example(problem: str, reasoning: str, solution: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": f"{reasoning}\n\n{solution}"},
        ]
    }


def _stream_openthoughts():
    from datasets import load_dataset

    ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train", streaming=True)
    for row in ds:
        yield row


def build_smoke(out_path: str | Path, n: int = 1000, rows=None) -> int:
    if rows is None:
        rows = _stream_openthoughts()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            if written >= n:
                break

            conversations = row.get("conversations") or []

            # Find first user message and first assistant message by role
            user_msg = None
            assistant_msg = None
            for turn in conversations:
                from_role = turn.get("from") or ""
                if from_role == "user" and user_msg is None:
                    user_msg = (turn.get("value") or "").strip()
                elif from_role == "assistant" and assistant_msg is None:
                    assistant_msg = (turn.get("value") or "").strip()

            # Skip if either message is missing
            if not user_msg or not assistant_msg:
                skipped += 1
                continue

            problem = user_msg

            # Completeness guard: rows without a full thought block are skipped.
            if "<|end_of_thought|>" not in assistant_msg:
                skipped += 1
                continue

            # Split assistant message at the end-of-thought marker and strip all markup tags
            parts = assistant_msg.split("<|end_of_thought|>", 1)
            reasoning = parts[0].replace("<|begin_of_thought|>", "").strip()
            solution = parts[1].strip()

            # Strip solution markers
            solution = solution.replace("<|begin_of_solution|>", "").replace("<|end_of_solution|>", "").strip()

            if not (problem and reasoning and solution):
                skipped += 1
                continue
            f.write(json.dumps(format_example(problem, reasoning, solution)) + "\n")
            written += 1

    if written < n:
        raise RuntimeError(f"only wrote {written} of {n}")
    print(f"wrote {written}, skipped {skipped}")
    return written


if __name__ == "__main__":
    count = build_smoke("data/smoke_v1.jsonl")
    print(f"wrote {count} examples to data/smoke_v1.jsonl")
