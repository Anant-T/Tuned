"""Build the final SFT artifacts from split.py's two sides.

Input is `out/split_train.jsonl` and `out/split_eval.jsonl`, verified against
the digests split.py recorded for them. Output is `out/law_v1_train.jsonl` and
`out/law_v1_eval.jsonl` - messages format, UNPACKED - plus a manifest carrying
split's own manifest (and through it dedupe's and decontamination's) forward.

UNPACKED, AND THAT IS A DECISION WITH A MEASUREMENT BEHIND IT
--------------------------------------------------------------
The 2026-08-09 perf audit REJECTED packing on this stack: it forfeits SDPA's
is_causal fast path and enable_gqa, costs a 64 MiB block-diagonal mask, and
turns one 8192-token attention into an 8192-square one over ~2,500 segments.
`gradient_accumulation_steps: 6` is the substitute and it lives on the trainer
side. The builder never packs, and nothing here should be argued from the
plan's earlier packing framing.

THIS PASS DOES NOT REWRITE ROWS
--------------------------------
A row is emitted BYTE-IDENTICAL or dropped with a logged reason. There is no
third outcome and in particular no repair: the empty-think scaffold is
`replay.empty_think`, `think_open + "\\n\\n" + think_close`, and stats.py
measures it with a byte comparison downstream, so a helpful whitespace fix here
would move a number there while looking like a cleanup. Every drop is counted
by reason in the manifest, so the corpus that does not ship is as visible as
the corpus that does.

THE TWO SHAPES THIS ACCEPTS, and which one production sends
------------------------------------------------------------
1. A BUILT ROW - `{"messages": [user, assistant], "_prov": {...}}`. It is
   verified against format_example's shape and passed through untouched.
2. RAW FIELDS - `{"problem", "reasoning", "solution"}` - rendered through
   `smoke.format_example`, which is the one definition of how a trace and an
   answer become one assistant turn.

Production currently sends only shape 1, and that is worth saying plainly
because the brief for this module expected otherwise: the accepted-generation
export already exists and already emits messages.
`decontaminate.generated_rows` turns a store generation into a built row, and
`decontaminate.store_items` feeds it into the screen - so a generation is a
messages row by the time it is screened, long before this pass. Shape 2 is the
seam for a future exporter that does not, tested on fixtures, and it is dead
code against today's pipeline rather than a live path.

WHAT "CONFORMS" MEANS, AND WHY IT IS NOT BYTE-EQUALITY WITH format_example
---------------------------------------------------------------------------
format_example builds `think_open + reasoning + think_close + solution` with
no separators. `generated_rows` builds
`think_open + "\\n" + think + "\\n" + think_close + "\\n\\n" + answer`, and
replay/curated build `empty_think(...) + answer`. All three are the same
SHAPE - an opened and closed reasoning block, then an answer - and they differ
in whitespace, so a byte-equality check against format_example's output would
reject the entire generated stream. The check is structural: one user turn and
one assistant turn in that order, the assistant content OPENING with the think
tag, closing it, and leaving a non-empty answer behind it.

A GENERATION WITH NO TRACE IS DROPPED, NOT SCAFFOLDED. `generated_rows` emits
a bare answer with no think tags at all when the teacher returned no reasoning
(`content = ... if think else answer`). Adding the empty scaffold here would be
mutating `messages`, which this pass does not do, so such a row is dropped as
`no_think_scaffold` and counted. The count is printed, because it is the
number that says how much of the generated stream arrived without a trace.

DROP, NEVER TRUNCATE, ABOVE THE LENGTH BUCKET
----------------------------------------------
The bucket is the TRAINER's `train.main.max_seq_length`, resolved through
config.py the same way the think tags are - a builder carrying its own 8192
could pass a corpus the trainer then truncates. Length is measured on the
chat-template-rendered text under the pinned tokenizer, which is what sft.py
feeds the trainer. A row over the bucket is dropped and counted; it is never
trimmed, and no attempt is made to shorten a trace to fit.

Build:  python -m tuned.data.assemble --config configs/data_law_v1.yaml
        [--in-train PATH] [--in-eval PATH] [--out-train PATH] [--out-eval PATH]
"""

from collections.abc import Sequence
from pathlib import Path

from tuned.data.smoke import format_example
from tuned.data.split import MANIFEST_FILENAME as SPLIT_MANIFEST_FILENAME
from tuned.data.split import (
    custody_of,
    custody_refusal,
    output_records,
)
from tuned.data.split import EVAL_FILENAME as SPLIT_EVAL_FILENAME
from tuned.data.split import TRAIN_FILENAME as SPLIT_TRAIN_FILENAME

TRAIN_FILENAME = "law_v1_train.jsonl"
EVAL_FILENAME = "law_v1_eval.jsonl"
MANIFEST_FILENAME = "assemble.json"

# 1  the first version. Built rows pass through byte-identical, raw
#    problem/reasoning/solution fields render through smoke.format_example,
#    the structural shape is verified rather than assumed, and a row over the
#    trainer's max_seq_length is dropped and counted rather than truncated.
ASSEMBLE_VERSION = 1

DROP_NO_CONTENT = "no_messages_or_fields"
DROP_TURNS = "not_one_user_one_assistant"
DROP_EMPTY_USER = "empty_user"
DROP_NO_SCAFFOLD = "no_think_scaffold"
DROP_UNCLOSED_SCAFFOLD = "unclosed_think_scaffold"
DROP_EMPTY_ANSWER = "empty_answer"
DROP_TOO_LONG = "over_length_bucket"
DROP_REASONS = (
    DROP_NO_CONTENT,
    DROP_TURNS,
    DROP_EMPTY_USER,
    DROP_NO_SCAFFOLD,
    DROP_UNCLOSED_SCAFFOLD,
    DROP_EMPTY_ANSWER,
    DROP_TOO_LONG,
)

RAW_FIELDS = ("problem", "reasoning", "solution")


# --------------------------------------------------------------------------
# Shape.
# --------------------------------------------------------------------------

def conforms(row, think_open: str, think_close: str) -> str | None:
    """A drop reason, or None if the row is what the trainer expects.

    Structural, not byte-equality with format_example - three shipped builders
    produce the same shape with three different whitespace layouts (see the
    module docstring), and only one of them is format_example itself.
    """
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        return DROP_TURNS
    user, assistant = messages
    if not isinstance(user, dict) or not isinstance(assistant, dict):
        return DROP_TURNS
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        return DROP_TURNS
    prompt = user.get("content")
    content = assistant.get("content")
    if not isinstance(prompt, str) or not isinstance(content, str):
        return DROP_TURNS
    if not prompt.strip():
        return DROP_EMPTY_USER
    if not content.startswith(think_open):
        # Includes the generated row whose teacher returned no trace: it is a
        # bare answer, and scaffolding it here would be a rewrite.
        return DROP_NO_SCAFFOLD
    tail = content[len(think_open):]
    if think_close not in tail:
        return DROP_UNCLOSED_SCAFFOLD
    if not tail.split(think_close, 1)[1].strip():
        return DROP_EMPTY_ANSWER
    return None


def raw_fields(row) -> tuple[str, str, str] | None:
    """(problem, reasoning, solution) if the row arrived unbuilt, else None."""
    if not all(isinstance(row.get(f), str) for f in RAW_FIELDS):
        return None
    return tuple(row[f] for f in RAW_FIELDS)


def built_row(row, *, think_open: str, think_close: str) -> tuple[dict | None, str | None]:
    """(row to emit, drop reason) - one of the two is always None.

    A built row is returned AS IT CAME IN, the same object, so the bytes
    written are the bytes read. A raw row is rendered through
    format_example and keeps whatever `_prov` it arrived with, because stats.py
    and push.py both read provenance and a row that loses it at the last stage
    is a row nobody can license.
    """
    if isinstance(row.get("messages"), list):
        reason = conforms(row, think_open, think_close)
        return (None, reason) if reason else (row, None)
    fields = raw_fields(row)
    if fields is None:
        return None, DROP_NO_CONTENT
    problem, reasoning, solution = fields
    built = format_example(problem, reasoning, solution, think_open, think_close)
    if "_prov" in row:
        built["_prov"] = row["_prov"]
    reason = conforms(built, think_open, think_close)
    return (None, reason) if reason else (built, None)


# --------------------------------------------------------------------------
# Length, under the pinned tokenizer.
# --------------------------------------------------------------------------

def rendered(tokenizer, messages) -> str:
    """The exact string sft.py hands the trainer for this row.

    `add_generation_prompt=False` because this is a completed conversation, not
    a prompt to continue - the same call sft.py:432 makes.
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def token_length(tokenizer, messages) -> int:
    """Tokens in the rendered row.

    `add_special_tokens=False` because the template has ALREADY emitted the
    control tokens as text; letting the tokenizer add its own on top would
    count a BOS this row will never carry and shift every length by a constant
    the gate would then be calibrated against.
    """
    return len(tokenizer.encode(rendered(tokenizer, messages), add_special_tokens=False))


def load_tokenizer(cfg):
    """The pinned tokenizer, from the reference the train config owns."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg.model_repo, revision=cfg.model_revision)


# --------------------------------------------------------------------------
# The pass.
# --------------------------------------------------------------------------

def assemble_rows(rows, *, tokenizer, think_open: str, think_close: str,
                  max_tokens: int) -> tuple[list[dict], list[dict], dict]:
    """(rows to write, drops, stats). Nothing is rewritten and nothing is trimmed."""
    kept: list[dict] = []
    drops: list[dict] = []
    lengths: list[int] = []
    total = 0
    for index, row in enumerate(rows):
        total += 1
        built, reason = built_row(row, think_open=think_open, think_close=think_close)
        if built is None:
            drops.append({"index": index, "reason": reason, "prov": row.get("_prov")})
            continue
        length = token_length(tokenizer, built["messages"])
        if length > max_tokens:
            # DROPPED, never truncated: a trimmed trace teaches an answer its
            # reasoning does not reach, and a trimmed answer teaches one that
            # stops mid-sentence.
            drops.append({
                "index": index, "reason": DROP_TOO_LONG, "tokens": length,
                "limit": max_tokens, "prov": row.get("_prov"),
            })
            continue
        lengths.append(length)
        kept.append(built)
    stats = {
        "total": total,
        "kept": len(kept),
        "dropped": len(drops),
        "by_reason": {
            reason: sum(1 for d in drops if d["reason"] == reason)
            for reason in DROP_REASONS
            if any(d["reason"] == reason for d in drops)
        },
        "max_tokens_kept": max(lengths, default=0),
        "limit": max_tokens,
    }
    return kept, drops, stats


def manifest_of(sides: dict, *, inputs: Sequence[str], outputs: Sequence[dict],
                upstream: dict | None, custody: dict, tokenizer_id: dict,
                max_tokens: int) -> dict:
    from tuned.data.store import utcnow

    return {
        "stage": "assemble",
        "assemble_version": ASSEMBLE_VERSION,
        "at": utcnow(),
        "inputs": list(inputs),
        "outputs": list(outputs),
        # WHICH tokenizer measured the lengths. A gate whose instrument is not
        # recorded is a number nobody can reproduce.
        "tokenizer": tokenizer_id,
        "max_tokens": max_tokens,
        "packing": False,
        "counts": {side: dict(stats) for side, stats in sides.items()},
        "split": upstream,
        "split_check": custody,
    }


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None, *, tokenizer=None) -> int:
    import argparse

    from tuned.data.config import load_build_config
    from tuned.data.decontaminate import write_manifest
    from tuned.data.jsonl import read_jsonl, write_jsonl
    from tuned.data.paths import build_paths
    from tuned.data.store import Store

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data_law_v1.yaml")
    parser.add_argument("--in-train", default=None, help=f"default out/{SPLIT_TRAIN_FILENAME}")
    parser.add_argument("--in-eval", default=None, help=f"default out/{SPLIT_EVAL_FILENAME}")
    parser.add_argument("--out-train", default=None, help=f"default out/{TRAIN_FILENAME}")
    parser.add_argument("--out-eval", default=None, help=f"default out/{EVAL_FILENAME}")
    parser.add_argument("--drops", default=None, help="default out/assemble_drops.jsonl")
    args = parser.parse_args(argv)

    cfg = load_build_config(args.config)
    paths = build_paths(cfg.build.workdir).ensure()
    in_train = Path(args.in_train) if args.in_train else paths.out_dir / SPLIT_TRAIN_FILENAME
    in_eval = Path(args.in_eval) if args.in_eval else paths.out_dir / SPLIT_EVAL_FILENAME
    out_train = Path(args.out_train) if args.out_train else paths.out_dir / TRAIN_FILENAME
    out_eval = Path(args.out_eval) if args.out_eval else paths.out_dir / EVAL_FILENAME
    drops_path = Path(args.drops) if args.drops else paths.out_dir / "assemble_drops.jsonl"

    missing = [str(p) for p in (in_train, in_eval) if not p.exists()]
    if missing:
        print(
            f"no such input: {', '.join(missing)}\n"
            f"  run: python -m tuned.data.split --config {args.config}"
        )
        return 2

    upstream, custody = custody_of(
        [in_train, in_eval], manifest_filename=SPLIT_MANIFEST_FILENAME
    )
    if upstream is None:
        print(custody_refusal(
            custody, stage="assemble",
            remedy=f"python -m tuned.data.split --config {args.config}",
        ))
        return 2

    if tokenizer is None:
        tokenizer = load_tokenizer(cfg)

    sides, all_drops, written = {}, [], {}
    for side, in_path, out_path in (("train", in_train, out_train),
                                    ("eval", in_eval, out_eval)):
        kept, drops, stats = assemble_rows(
            read_jsonl(in_path),
            tokenizer=tokenizer,
            think_open=cfg.think_open,
            think_close=cfg.think_close,
            max_tokens=cfg.max_seq_length,
        )
        written[side] = write_jsonl(out_path, kept)
        sides[side] = stats
        all_drops += [dict(d, side=side) for d in drops]
    write_jsonl(drops_path, all_drops)

    manifest = manifest_of(
        sides,
        inputs=[str(in_train), str(in_eval)],
        outputs=output_records([(out_train, written["train"]), (out_eval, written["eval"])]),
        upstream=upstream,
        custody=custody,
        tokenizer_id={"repo": cfg.model_repo, "revision": cfg.model_revision},
        max_tokens=cfg.max_seq_length,
    )
    write_manifest(out_train.parent / MANIFEST_FILENAME, manifest)
    store = Store.open(paths.state_db)
    try:
        store.log_event("assemble", manifest)
    finally:
        store.close()

    for side in ("train", "eval"):
        stats = sides[side]
        print(f"{side}: read {stats['total']}, kept {stats['kept']}, "
              f"dropped {stats['dropped']} (longest kept {stats['max_tokens_kept']} tokens)")
        for reason, count in sorted(stats["by_reason"].items()):
            print(f"    drop[{reason}]: {count}")
    scaffold_drops = sum(
        s["by_reason"].get(DROP_NO_SCAFFOLD, 0) for s in sides.values()
    )
    if scaffold_drops:
        print(
            f"  {scaffold_drops} row(s) carried NO reasoning scaffold at all and were "
            f"dropped rather than repaired - a generation whose teacher returned no trace "
            f"arrives as a bare answer, and adding the empty block here would be a rewrite"
        )
    print(f"wrote {written['train']} rows -> {out_train}")
    print(f"      {written['eval']} rows -> {out_eval}")
    print(f"      {len(all_drops)} drops -> {drops_path}")
    print(f"      manifest -> {out_train.parent / MANIFEST_FILENAME}")
    print(f"  UNPACKED, per the 2026-08-09 perf audit; the trainer's ga=6 is the substitute")

    if sides["train"]["total"] and not written["train"]:
        print("  EVERY TRAIN ROW WAS DROPPED from a non-empty input - that is a rule fault, "
              "not a corpus.")
        return 1
    if sides["eval"]["total"] and not written["eval"]:
        print("  EVERY EVAL ROW WAS DROPPED, so the held-out set split.py built is gone and "
              "there is nothing to score against.")
        return 1
    if not sides["train"]["total"]:
        print("  NOTHING READ on the train side: the input is empty. Exiting 0 here would "
              "report an empty dataset as assembled.")
        return 1
    return 0


if __name__ == "__main__":
    import os
    import sys

    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
