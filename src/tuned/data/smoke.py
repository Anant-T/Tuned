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
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            if written >= n:
                break
            # Try new dataset field names first (conversations structure)
            conversations = row.get("conversations") or []
            if conversations and len(conversations) >= 2:
                problem = (conversations[0].get("value") or "").strip()
                assistant_msg = (conversations[1].get("value") or "").strip()
                # Split assistant message at the end-of-thought marker if it exists
                if "<|end_of_thought|>" in assistant_msg:
                    parts = assistant_msg.split("<|end_of_thought|>", 1)
                    reasoning = parts[0].replace("<|begin_of_thought|>", "").strip()
                    solution = parts[1].strip()
                else:
                    # If no markers, treat whole message as reasoning
                    reasoning = assistant_msg
                    solution = ""
            else:
                # Fallback to old field names for test compatibility
                problem = (row.get("problem") or "").strip()
                reasoning = (row.get("deepseek_reasoning") or "").strip()
                solution = (row.get("deepseek_solution") or "").strip()

            if not (problem and reasoning and solution):
                continue
            f.write(json.dumps(format_example(problem, reasoning, solution)) + "\n")
            written += 1
    return written


if __name__ == "__main__":
    count = build_smoke("data/smoke_v1.jsonl")
    print(f"wrote {count} examples to data/smoke_v1.jsonl")
