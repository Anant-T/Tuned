# deepseek takes judge slot B: it answers, and that is all this measured

2026-08-27. `bai/deepseek-v4-flash` now holds a seat in `routing.judge` and
`routing.tiebreak`, with `thinking: {type: disabled}` and `temperature: 0.2`
scoped to those two roles.

**Pass line, fixed before the run: at least 9 of 10 calls return a parseable
three-axis verdict. Measured: 9/10. PASS — by exactly one call.** A
confirmatory second batch of 10 returned 10/10, so the shipped configuration
stands at **20 of 21 across every call made** (19/20 excluding the sanity call).

**This measured whether deepseek *answers* as a judge. It did not measure
whether it judges *well*.** No calibration against gold labels was run; see §6.

---

## 1. Why the seat needed filling

`routing.judge` wants two independent judges per row. As of 2026-08-27 it had
one.

| ref | role | status 2026-08-27 |
|---|---|---|
| `groq/qwen/qwen3.6-27b` | judge, slot A | working |
| `cerebras/gemma-4-31b` | judge, slot B | **HTTP 402 `payment_required`** |
| `cerebras/gpt-oss-120b` | lead generator | **HTTP 402 `payment_required`** |
| `openai/gpt-5-mini` | judge, position 3 | working, **PAID** |

Both cerebras 402s were verified the same day by direct probe. That is the
whole problem: with gemma dead, the next ref the judge pool reaches for slot B
is a paid model, and `routing.family_separation` is not a general guard against
that. Separation excludes the *generator's* family. The two `openai` refs are
declared family `gpt-oss` by deliberate design, so separation hides them on a
gpt-oss row and **on no other row**. With generation failing over to
`bai/deepseek-v4-flash`, separation excludes only `{deepseek}` and both paid
refs become reachable.

The only thing standing between that and real spend today is the `usd_cap: 0.0`
fence added to both `openai/gpt-5-*` blocks earlier the same day. A fence is
not a judge. Meanwhile the rows themselves go nowhere: the 2026-08-23 live
drain stalled with **34 `judge_error`** on exactly this pool gap.

`bai/deepseek-v4-flash` is free, and it is a family no other judge in the pool
holds — which is what makes it a real second opinion rather than a second vote
from the same weights.

## 2. The change

In the `bai` provider block:

```yaml
roles: [generator, judge, tiebreak]
role_params:
  judge: {thinking: {type: disabled}, temperature: 0.2}
  tiebreak: {thinking: {type: disabled}, temperature: 0.2}
```

`temperature: 0.2` is the repo's judge convention (`build_payload` docstring,
`providers.py:955-959`); it was added after the probe, for the reason in §6.
Every measurement below was taken at the model's generator default of 0.7.

and the ref appended to `routing.judge` (position 3) and `routing.tiebreak`
(position 4) — after every free ref, before every paid one.

Two mechanisms have to hold together, and they are not the same mechanism:

* **The reply budget must stay small.** A judge call arrives with 1,024 tokens
  because a verdict is short. `_bai_request_hook` raises a *generator's* budget
  to the model's 16,384 ceiling, and leaves a `JUDGING_ROLES` budget alone —
  that is Task 1, committed as `ca8316f`. Raising it 16x would not give the
  answer more room; it would give the model sixteen times as much room to
  think instead of judge.
* **The model must not spend the 1,024 it keeps.** Reasoning is billed against
  `max_tokens` here and emitted **first**, so a call that deliberates too long
  returns a well-formed **HTTP 200 with empty `content` and
  `finish_reason: "length"`**. `thinking: disabled` is what addresses that, and
  it is role-scoped because the generator wants the reasoning it is being paid
  for.

## 3. Offline: the payload carries both

`tests/test_build_providers.py::test_shipped_bai_judge_payload_disables_thinking_and_keeps_the_small_budget`
builds the payload through `ChatClient.build_payload` on the config **as
shipped** — not by reading the YAML, because `role_params` is the middle of
three merge layers and `req.max_tokens` is applied after all of them, so the
key has to survive the merge to reach the wire.

| role | `thinking` | `max_tokens` | `temperature` |
|---|---|---|---|
| `judge` | `{"type": "disabled"}` | 1024 | 0.2 |
| `tiebreak` | `{"type": "disabled"}` | 1024 | 0.2 |
| `generator` | *absent* | 16384 | 0.7 |

The generator row is asserted in the same test: deepseek is still
`routing.generator` ref 2, and a `thinking: disabled` that leaked onto the
generator would silently strip the reasoning traces this corpus is made of.

Full suite: **3,579 passed, 19 skipped.**

## 4. Live: 21 real calls

Built through the shipped config and the real provider layer — `cfg.model_for`
to the same `ChatClient` the Router builds, then `build_payload`, then the
shipped bai quirk hooks. No payload was assembled by hand.

* **Candidate:** `gen_id 64` from `data/build/exp_deepseek/state/law_v1.sqlite3`,
  opened `mode=ro`. Stream `synthesis`. The judge prompt was rendered by
  `judge_messages` — the same renderer `judge_slot` calls — at **6,903 real
  prompt tokens** (the router's conservative estimate for it is 8,498).
* **`max_retries=1` deliberately**, so one probe call is exactly one HTTP
  request. The shipped client retries; retrying here would smooth the very
  distribution being measured.
* **Paced 10 s apart.** b.ai's limit is a request-counted bucket of 10/min that
  rejected calls also consume.

### Per-call record — pre-registered batch (n=10)

| # | status | `finish_reason` | `reasoning_tokens` | content | parsed | axes |
|---|---|---|---|---|---|---|
| 1 | 200 | stop | 0 | yes | yes | 4,5,5 |
| 2 | 200 | stop | **637** | yes | yes | 5,5,5 |
| 3 | 200 | stop | 0 | yes | yes | 5,5,5 |
| 4 | 200 | stop | 0 | yes | yes | 4,4,5 |
| 5 | 200 | stop | 0 | yes | yes | 5,5,5 |
| 6 | 200 | stop | 0 | yes | yes | 5,5,5 |
| 7 | 200 | stop | 0 | yes | yes | 5,5,5 |
| 8 | 200 | stop | *not reported* | yes | yes | 5,5,5 |
| 9 | 200 | stop | **748** | yes | yes | 5,5,5 |
| 10 | 200 | **length** | **1024** | **empty** | **no** | — |

**9/10. The pass line was 9/10.** Call 10 is the predicted failure mode,
exactly: HTTP 200, the entire reply budget consumed by reasoning, no content.

### Confirmatory batch (n=10) and sanity call (n=1)

10/10 and 1/1, all `finish_reason: stop`, no non-zero reasoning tokens in
either. Latency across all 21: min 1.8 s, median 2.8 s, max 10.1 s — the slow
calls are the ones that thought.

## 5. The reasoning distribution, against the ~90% claim

`thinking: {type: disabled}` was recorded on 2026-08-25 as **~90% effective**.
Across the 21 shipped-config calls:

| | n |
|---|---|
| `reasoning_tokens` reported as 0 | 14 |
| `reasoning_tokens` reported non-zero | 3 (637, 748, **1024**) |
| `completion_tokens_details` **absent from the response** | 4 |
| truncated empty (`finish_reason: length`) | **1 / 21 = 4.8%** |

Two things to take from this rather than the 90% figure.

**The four absent rows are "not reported", not zero.** The provider omits
`completion_tokens_details` on some 200s. Counting them as zeroes would
manufacture a cleaner number than was measured. Of the 17 calls where it *was*
measurable, reasoning was suppressed on **14 — about 82%**.

**Suppression is probabilistic, and the failure is bimodal.** When the model
thinks anyway it does not think a little: the observed non-zero values are 637,
748, and 1,024 against a 1,024-token budget. There is no useful middle. The
per-call truncation rate of 1/21 has a 95% interval of roughly **0.1% to 24%** —
n=21 constrains this loosely, and the true rate could be several times the
observed one.

### A knob that does not help: `reasoning_effort: minimal`

The judge payload inherits `reasoning_effort: low` from the model's params. A
2026-08-25 note recorded that `thinking: disabled` and
`reasoning_effort: minimal` "both work", so a **diagnostic arm** (not shipped)
ran the identical call with `minimal` substituted:

| arm | parsed | truncated | non-zero reasoning |
|---|---|---|---|
| shipped (`thinking: disabled` + `low`) | 20/21 | 1 | 637, 748, 1024 |
| diagnostic (`thinking: disabled` + `minimal`) | **7/10** | **3** | 478, 1024, 1024, 1024 |

`minimal` measured **worse**, not better. At n=10 that is not a conclusive
ranking, but it is conclusive that `minimal` is not the fix for the 4.8%, and
the 2026-08-25 "both work" note should be read as "neither knob is
deterministic". **No effort key was added to the config.**

### The arm that was never tried, because config cannot express it

Both arms above carry a `reasoning_effort` key, and every arm reachable from
this config always will. The merge is a **dict overlay** — `model.params <
role_params[role] < req.params` — so a role layer can set or override a key but
can never **unset** one. `reasoning_effort: low` sits in the model's `params`
for the generator's benefit, and there is no way to spell "thinking disabled,
and no `reasoning_effort` key at all" in YAML.

That untried arm is worth naming because the two keys pull against each other:
`reasoning_effort: low` asks for *some* reasoning while `thinking: disabled`
asks for *none*, and a provider resolving that conflict inconsistently is one
plausible account of the ~82% suppression measured above. **This is a
hypothesis and nothing here tests it** — the shipped pairing was not compared
against a payload with the key absent, because producing one requires a code
change rather than a config change. It is recorded so the next person tuning
this does not assume the space was exhausted.

### What a truncation actually costs — traced, not assumed

The `_bai_response_hook` marks the truncated-empty reply `retryable=True`, which
reads like a free re-ask. It is not one:

1. The hook raises from inside `_to_response`, which sits **outside** the scope
   of `ChatClient.complete`'s retry loop — that `except` catches
   `httpx.HTTPError` only. There is **no in-client re-ask of the same ref**.
2. `Router.complete` treats a retryable error as **fail-over**: `report_failure`,
   then the next eligible ref.
3. `judge_slot`'s `for attempt in (1, 2)` retries an **unparsable reply**. On a
   `ProviderError` it returns immediately.

So on a gpt-oss row where qwen holds slot A, a truncated deepseek reply leaves
gemma (402) and the two openai refs (family `gpt-oss`, excluded by separation)
— i.e. **nothing**. The aggregate error carries no `skipped` set and is not
`context_exceeded`, so `outcome.unroutable` is `False` and the row **re-queues
rather than parking**; the next sweep re-asks it. The circuit breaker needs 4
*consecutive* failures, so an independent ~5% rate will not cool the ref.

The cost of the 4.8% is therefore a re-queued row, not a lost one — but it is
also not free, and it is not the "it just retries" that `retryable=True`
suggests at a glance.

## 6. What this does NOT establish

### It does not fill slot B for the rows that are stalling today

`cerebras/gpt-oss-120b` is 402 (§1). It is the lead generator, so with it dead
the free generator **is** `bai/deepseek-v4-flash` — and on a deepseek-generated
row `family_separation` excludes the deepseek **judge** as well. What is left
for slot B on that row is `cerebras/gemma-4-31b` (402) and the two paid
`openai` refs behind the `usd_cap: 0.0` fence. **Slot B on a deepseek row is
still empty after this change.**

That is not a corner case, it is the case both motivating failures came from:
the 2026-08-23 live drain's 34 `judge_error` and the validation wave's 8 lost
rows were both on deepseek-generated waves. A reader deciding whether the build
can now be run should read this as: **it cannot, not on the current fleet.**

What this change actually buys is slot B for **gpt-oss and lightning** rows.
It becomes load-bearing the moment cerebras is funded — which restores
gpt-oss generation, and on a gpt-oss row separation excludes `{gpt-oss}`,
leaving qwen in slot A and deepseek in slot B with both paid refs excluded
along with the generator. It is equally load-bearing if generation moves to any
other non-deepseek family. Until one of those happens, the seat is correct and
idle.

### Judging quality

**Judging quality is entirely unmeasured.** No gold-labelled calibration was
run against this model, no `judge_threshold` row exists for it, and its
verdicts will enter the build under the same provisional thresholds any
uncalibrated judge gets. This branch carries prior judge-calibration work
precisely because "returns a verdict" and "returns a good verdict" are
different claims; only the first is made here.

Three specific gaps:

**The probed candidate was deepseek's own generation.** `gen_id 64` was
produced by `bai/deepseek-v4-flash`. In production `family_separation` forbids
that pairing outright. The probe made it anyway because the question was
mechanical — does a verdict come back — but it means the *scores* in §4 are
self-assessment and carry no evidence about accuracy whatsoever.

**The verdicts were not self-consistent, and the config was corrected for it.**
Twenty verdicts on one unchanging candidate: `(5,5,5)` x12, `(4,3,3)` x3,
`(4,5,5)` x2, `(4,4,4)`, `(4,4,5)`, `(5,4,5)`. A two-point swing on two axes
between calls on identical input.

This is not merely untidy. Under a `min_axis 4` rule the `(4,3,3)` draws
**reject** the candidate the `(5,5,5)` draws **accept** — so **3 of the 20
draws flipped the decision**, with nothing about the candidate changing.

The cause was sampling: every probe call inherited **`temperature: 0.7`,
`top_p: 0.95`** from the model's generator `params`, because the `role_params`
block carried only `thinking`. `temperature: 0.2` for a judge is this repo's
convention, stated in `build_payload`'s own docstring
(`providers.py:955-959`) and already live in `mistral-large-latest`'s tiebreak
block. **Both `role_params` blocks now pin `temperature: 0.2`.**

**Every number in §4 and §5 was measured at 0.7 and has not been re-measured
at 0.2.** The 20/21 answer-rate is a property of the configuration that was
probed, not of the one that ships. Re-measuring at 0.2 is a separate,
properly-designed experiment — it was deliberately not bolted onto this run,
and no measured value in this report was changed. Lower temperature is expected
to tighten the verdict spread; its effect on the truncation rate is unknown in
either direction.

**One prompt, one shape, one stream.** 6,903 tokens on `synthesis`. The same
store holds rows rendering to 14k-19k judge prompts, and nothing was measured
there, on the `transition` or `statute_qa` streams, or on the **tiebreak**
prompt — which is larger than the judge prompt and whose seat this change also
filled. The tiebreak `role_params` and routing entry are asserted offline only.

## 7. Reproducing

`data/build/exp_deepseek/out/probe_judge.py` (scratch, gitignored). Reads the
exp store `mode=ro`. The live control store
`data/build/state/law_v1.sqlite3` was never opened at all —
`size=554532864 mtime=1787309490` before and after.
