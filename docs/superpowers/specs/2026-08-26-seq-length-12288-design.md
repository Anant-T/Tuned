# Raise the training sequence cap to 12,288

**Date:** 2026-08-26
**Status:** approved, not yet implemented
**Lane:** `configs/law_v1_8b_ddp.yaml` — Qwen3-8B, 2x T4 DDP, qualified 2026-08-08 at seq 8192

## Summary

Raise `max_seq_length` from 8192 to 12,288 in both the `smoke` and `main` blocks of the production lane, and requalify with a PROBE run against real 12,288-token rows.

This recovers **zero training rows today**. It is accepted anyway as future-proofing: it converts a latent mid-run DDP hang into a pre-flight gate. The reasoning for both halves of that sentence is below.

## The measurement that shapes this decision

Length distribution of every built row in the pipeline, at a conservative 3.5 chars/token:

| stream | rows | p50 | p90 | p99 | p100 |
|---|---|---|---|---|---|
| `data/build/streams/curated_c1.jsonl` | 1,100 | 2,438 | 4,094 | 6,102 | **6,725** |
| `data/build/streams/replay.jsonl` | 4,320 | 3,042 | 5,934 | 6,766 | **7,610** |

**0 of 5,420 rows exceed 8192 tokens.** The longest example in the corpus sits ~580 tokens under the current cap.

Source judgments are genuinely long — `data/build/corpus/extraction.jsonl` carries 5,471 documents, of which 41.4% exceed 8192 tokens, p90 = 22,901 tokens (44 pages), p99 = 73,413 (136 pages), p100 = 616,310 (1,076 pages). None of that length reaches the trainer: `chunks.py` sets `MAX_CHUNK_TOKENS = 1500`, so training rows are built from chunks, not whole documents.

Two consequences follow, and both are load-bearing:

1. **The 8192 cap is not what bounds row length.** The chunk budget is. Raising the cap without raising `MAX_CHUNK_TOKENS` cannot produce a longer row.
2. **The rows that *are* dropped are dropped by a char proxy, not by the cap.** `curated.py:145` and `replay.py:168` both reject at `>= 24_000` chars as a stand-in for the 8192-token gate — that is 2.93 chars/token, against a real ratio nearer 3.5–4.2 for Indian legal English on the Qwen3 tokenizer. The proxy is 17–30% stricter than the gate it approximates. Recalibrating it is **out of scope here**.

## Why raise the cap anyway

At `bs=1` the collator pads to the longest row *in the batch*, which is the row itself. `max_seq_length` is therefore a cap, not a shape. Raising it to 12,288 allocates ~6 MB of additional RoPE cache and nothing else, because no row is long enough to use the headroom. Tokens per optimizer step stay at ~30k real tokens, throughput is unchanged, and `max_steps` is unchanged.

The cost of a row that *actually* reaches 12,288 is roughly +1.8 GiB: checkpoint-boundary hidden states are `36 layers x seq x 4096 x 2 B` = 2.25 GiB at 8192, so +1.13 GiB; plus ~0.3 GiB of per-block recompute transient and ~0.35 GiB of cross-entropy transient. Measured headroom is ~2.3 GiB (peaks 12.98/13.18 GiB against a 14.56 GiB cap, real-GiB accounting). These are estimates from layer arithmetic — PROBE settles them.

The risk this retires: today a row that grows past 8192 is dropped by `assemble.py` and nothing bad happens. Under a raised cap that row is *trained*, at up to 12,288, at step N of a MAIN run. If it does not fit, one rank OOMs and the other hangs at the NCCL barrier until the watchdog kills the session — a quota-week lost to a failure that had no pre-flight signal. Proving the ceiling survivable now makes that impossible later.

## Changes

### Config — `configs/law_v1_8b_ddp.yaml`

- `train.smoke.max_seq_length: 8192` -> `12288`
- `train.main.max_seq_length: 8192` -> `12288`

Both move. They are read independently, and a smoke block qualifying a different ceiling than main is precisely the divergence that surfaces mid-MAIN.

Unchanged: `gradient_accumulation_steps` (2 smoke / 6 main), `max_steps: 0` sentinel, `save_steps`, LoRA block, optimizer block, `max_grad_norm: 0.3`.

The OOM fallback ladder comment gains a rung: `standard-quant repo -> UNSLOTH_CE_LOSS_N_CHUNKS 32 -> seq 8192 -> seq 6144`.

### Notebook — `notebooks/kaggle_smoke.ipynb`

- `UNSLOTH_CE_LOSS_N_CHUNKS` `16` -> `32`, spent pre-emptively. The CE transient scales with sequence length; ~0.7 GiB at 8192/16-chunks becomes ~1.05 GiB at 12,288, and 32 chunks returns it to ~0.5 GiB. Must remain in the parent env, before `import unsloth`, so the torchrun children inherit it.
- `PROBE_SEQ = 12288` for the requalification run.
- Reference-numbers section updated with the new PROBE peaks once measured.

Deliberately **not** spent: the standard-quant repo rung. Dynamic 4-bit is an accuracy advantage and its −1.31 GiB is the reserve to hold back.

### Code — `src/tuned/train/sft.py`

**`_ReservedCeiling` sampling — a real hole this change opens.** The callback checks steps 1–3 then every 25th, justified in its docstring by "the full memory shape exists by step 1-2 at bs=1 with a fixed bucket". That premise holds at 8192-with-drop, where every probe row truncates to the same length. It is false once the cap exceeds the longest row: the peak-memory step becomes whichever step carries the longest row, and it has a 1-in-25 chance of being sampled.

Change the default `every=25` -> `every=1`. `torch.cuda.max_memory_reserved()` is a stats-counter read with no CUDA sync — free against a ~74 s step. Update the docstring to state the new rationale (variable bucket, every step checked).

**Ladder strings.** `check_vram_reserved` (`sft.py:140`) and the `_ReservedCeiling` raise (`sft.py:581`) both end their remediation ladder at "seq 6144". Both become `... -> seq 8192 -> seq 6144`, so an operator reading the abort mid-session is not told to skip a rung.

The 13.5 GiB abort line itself does **not** move. A PROBE that trips it is the gate working.

### Probe dataset — `data/probe_long.jsonl`

Rebuild at the new target: `python -m tuned.data.probe --config configs/law_v1_8b_ddp.yaml --target-tokens 12288`

This is the single largest false-green risk in the change. `probe.py`'s own docstring states it: a probe at seq N that is not fed rows tokenizing past N measures nothing. The existing `probe_long.jsonl` is built for 8192 and would produce a green PROBE that proves the old ceiling, not the new one.

## Gate plan

**One session, clearing PROBE and SAVETEST together.** Not the full four-gate ladder, and not two sessions.

PROBE is the only run that measures anything new — it is the only one fed rows that reach 12,288. SMOKE runs `data/smoke_v1.jsonl`, whose rows top out well under 8192; at a 12,288 cap it exercises a byte-identical memory shape to the qualified 8192 SMOKE and would spend ~1.2 h of quota to reproduce known numbers. RESUME exercises scheduler and scaler restoration, neither of which this change touches.

SAVETEST proves the checkpoint push path, which does not depend on row length — so it can ride along on the probe dataset instead of costing a second model load. Merge the two `ARGS` entries into one:

```
"PROBE": ["--max-steps", "2", "--save-steps", "1",
          "--dataset", "data/probe_long.jsonl"]
         + (["--max-seq-length", str(PROBE_SEQ)] if PROBE_SEQ else []),
```

`--no-hub` is dropped so the run pushes. This additionally proves a checkpoint push succeeds *at the new sequence length*, which the split plan never tested.

**Precondition: the staged-model input must be attached.** `stage_model.ipynb`'s output mounted under `/kaggle/input/.../qwen3-8b-staged/` is auto-detected via `TUNED_MODEL_PATH` and skips the ~7 GB hub download. Wall clock here is dominated by model load, not by stepping — 2 steps at ga=2 is ~4 micro-batches, roughly 4 minutes.

| gate | green means |
|---|---|
| ceiling (`PROBE_SEQ=12288`) | finite loss; `peak_vram_reserved_gb_dev0/1` below 13.5 GiB with >= 1 GiB margin on the worst rank; `label_coverage=` nonzero. `eos_in_labels=` near-zero is the expected artifact of truncating probe rows (`sft.py:150`), not a failure. |
| save path (same run) | `last-checkpoint/` visible in the checkpoint repo. |

**Rejected: probing at `ga=1`.** Halving the micro-batches would save ~2 minutes and cost measurement fidelity. Under `ga=2`, micro-batch 2's forward runs while micro-batch 1's LoRA gradients are still resident; under `ga=1` with `set_to_none` they are freed first, so the measured ceiling comes in ~0.15 GiB low — about 15% of the margin being gated on. Keep `ga=2`; do not add a `--gradient-accumulation-steps` override for this.

Before any MAIN run: the 2-step `--no-hub` probe on the real dataset, confirming `post_filter_rows=` is **unchanged**. `config.py:987` feeds `train.main.max_seq_length` into `assemble.py`'s drop gate, so this change does relax that cap to 12,288 — but the 24,000-char proxy in `curated.py` clips first, so the corpus is expected to be identical. If `post_filter_rows` moves, `max_steps` must be re-derived and committed before MAIN; `check_resume_schedule` freezes it, and a mid-stream change breaks resume.

## Non-risks

Stated explicitly so they are not re-investigated:

- **RoPE / context extension.** Qwen3-8B's `max_position_embeddings` is 40,960. 12,288 is well inside native context: no `rope_scaling`, no YaRN, no positional degradation.
- **`group_by_length`.** A reordering only; at `bs=1` it changes which row lands in which step, never how many rows share a batch. Peak memory is unaffected.
- **`eos_in_labels`.** Can only improve under a higher cap, since strictly fewer rows truncate.
- **Throughput and quota.** Real tokens per optimizer step are set by actual row lengths, not the cap. ~30k tokens/step and the `--time-budget-s 37800` envelope are unchanged.
- **The `approx_tokens_per_sec` log line** (`sft.py:641`) multiplies by `max_seq_length` and is labelled an upper bound. It becomes a looser upper bound. Cosmetic; do not "fix" it by changing the multiplier without changing the label.

## Acceptance criteria

1. Both `max_seq_length` fields read `12288`; no other config field changed.
2. `data/probe_long.jsonl` rebuilt at `--target-tokens 12288`.
3. Merged gate green at `PROBE_SEQ=12288` in a single session: finite loss, worst-rank reserved peak below 13.5 GiB with >= 1 GiB margin, `label_coverage=` nonzero, and `last-checkpoint/` visible in the Hub repo.
4. The `PROBE` `ARGS` entry carries `--save-steps 1` and no longer carries `--no-hub`; `SAVETEST` is retired as a separate mode.
5. `_ReservedCeiling` checks every step; docstring reflects the variable-bucket rationale.
6. Both remediation-ladder strings include the `seq 8192` rung.
7. `post_filter_rows=` verified unchanged on the real dataset before MAIN.
8. Existing test suite green.

## Out of scope

- **Recalibrating the 24,000-char proxy** in `curated.py:145` / `replay.py:168`. This is the change that would actually recover dropped rows, at zero VRAM cost. Tracked separately.
- **Raising `MAX_CHUNK_TOKENS`** above 1,500. The only change that would make training rows longer, and a corpus-design decision with teacher-cost and citation-context consequences, not a memory decision.
- **Seq 16,384.** Needs ~+3.0 GiB against ~2.3 GiB of headroom, forcing the standard-quant rung and its accuracy cost, and leaving under 1 GiB of margin — below this lane's own green bar.
