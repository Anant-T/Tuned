"""Build the ~1k-example smoke dataset from OpenThoughts-114k (Apache-2.0).

Assistant content is wrapped in the base model's reasoning scaffold
(config data.think_open / data.think_close), e.g. [THINK]trace[/THINK]solution
for Ministral-3 Reasoning, so training matches the model's native template.
"""

import json
from pathlib import Path


def format_example(
    problem: str,
    reasoning: str,
    solution: str,
    think_open: str = "",
    think_close: str = "",
) -> dict:
    if think_open or think_close:
        content = f"{think_open}{reasoning}{think_close}{solution}"
    else:
        content = f"{reasoning}\n\n{solution}"
    return {
        "messages": [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": content},
        ]
    }


def _stream_openthoughts():
    from datasets import load_dataset

    ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train", streaming=True)
    for row in ds:
        yield row


def build_smoke(out_path: str | Path, n: int = 1000, rows=None, think_open: str = "", think_close: str = "") -> int:
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
            f.write(
                json.dumps(
                    format_example(problem, reasoning, solution, think_open, think_close)
                ) + "\n"
            )
            written += 1

    if written < n:
        raise RuntimeError(f"only wrote {written} of {n}")
    print(f"wrote {written}, skipped {skipped}")
    return written


if __name__ == "__main__":
    import argparse

    from tuned.train.config import load_config

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/law_v1.yaml")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    out = args.out or cfg.train.smoke.dataset
    count = build_smoke(
        out,
        think_open=cfg.data.think_open,
        think_close=cfg.data.think_close,
    )
    print(f"wrote {count} examples to {out}")
