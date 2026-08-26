# b.ai DeepSeek qualification — measured 2026-08-25

All numbers below are live-API measurements taken on 2026-08-25, not vendor claims.
b.ai publishes no pricing, quota, or rate-limit documentation, so every limit here is
inferred from observed behaviour and is a **floor**, not a ceiling.

## What the provider is

| | |
|---|---|
| Base URL | `https://api.b.ai` |
| OpenAI shape | `POST /v1/chat/completions` (verified) |
| Anthropic shape | `POST /v1/messages` (documented, not exercised) |
| Model list | `GET /v1/models` — 42 models |
| Auth | `Authorization: Bearer sk-…` (verified); `x-api-key` also documented |

An aggregator fronting other vendors' models, not a first-party lab.

## Access matrix on the no-deposit tier

Probed every candidate individually; the trailing 429s in the first sweep were my own
burst tripping the limiter, so each was re-probed sequentially with backoff.

| Model | Verdict |
|---|---|
| `deepseek-v4-flash` | **free** |
| `deepseek-v4-flash-vision-exp` | **free** |
| `mimo-v2.5`, `hy3` | **free** |
| `deepseek-v4-pro` | 403 `access_denied` — "Deposit required to unlock premium models" |
| all `gpt-5.x`, all `claude-*`, `gemini-*`, `glm-*`, `kimi-*`, `minimax-*` | 403, same message |
| `qwen3.8-27b` | 400 `credit insufficient balance: balance=0 required=3204` |

Account balance is zero, so only the four free-tier models are reachable. **`deepseek-v4-pro`
is not free** — the free DeepSeek here is flash only.

## Rate limiting

No documented limits, and **no rate-limit headers on any response** — nothing matching
`x-ratelimit-*`, no `Retry-After`. A 429 arrives with an **empty body**. That matters for
integration: `providers.parse_quota_headers` will find nothing here, so limits must be
declared statically in config rather than learned from the wire.

### Corrected model: a request-counted bucket of 10

The first pass reported "concurrency 4 is the safe operating point". That was a description of
one workload, not of the limiter. Isolating the variables:

**It is not a concurrency cap.** Same 24 requests, same tokens, opposite concurrency:

| Dispatch | Admitted |
|---|---|
| 24 fired **sequentially**, back to back (in-flight = 1) | 10/24 |
| 24 fired **simultaneously** (barrier-synchronised) | 12/24 |

A 17× difference in overlap moved the admitted count by two. The cap is not on in-flight
requests.

**It counts requests, not tokens.** Two workloads at opposite extremes:

| Workload | Request rate | Token rate | Result |
|---|---|---|---|
| tiny 16-token replies | ~41/min | ~650 tok/min | **throttled** at 10 |
| ~1000-token replies | ~4.5/min | ~4,700 tok/min | **clean**, 10/10 |

7× the tokens passes; 9× the requests throttles.

**The bucket holds 10.** Every saturating test admitted exactly ten: 24 back-to-back → 10,
24/min held for 60 s → 10, a 30-request burst → 10, and 15 invalid requests → 10. Recovery
from a fully drained bucket to first success took **40.5 s**.

**Rejected requests still consume it.** After 15 deliberately-invalid requests (10 × 400,
5 × 429), the very next *valid* request returned 429. A 400 costs a token bucket slot, so
validation failures must be rate-limited like any other call.

So the operating limit is **~10 requests/minute**, and it is indifferent to how large those
requests are. The earlier sustained drain (20 calls in 216 s = 5.6/min, zero 429s) was clean
because it sat under that ceiling, not because concurrency 4 is special — at 316 tok/s it
still yielded ≈1.14M tokens/hour. Long, token-heavy calls are the efficient shape here;
many small calls are the wasteful one. No daily cap surfaced across this session.

## The empty-content cliff — the operationally important finding

`deepseek-v4-flash` is a reasoning model that emits `reasoning_content` as a field separate
from `content`, with `completion_tokens_details.reasoning_tokens` in usage. Reasoning is
billed against the same `max_tokens` budget and is emitted **first**, so a budget that runs
out during reasoning returns a well-formed HTTP 200 whose `content` is an empty string.

| `max_tokens` | empty `content` |
|---|---|
| 256 | 3/3 |
| 512 | 3/3 |
| 1024 | 2/3 |
| 2048 | 1/3 |
| 4096 | 0/3 short prompt — but **10/20** on the real synthesis prompt |
| 6144 | 1/4 |
| 8192 | 1/4 |
| 12288 | 0/4 |

The cliff is not at a fixed budget; it scales with prompt difficulty. Reasoning took 66–82%
of output tokens throughout — but note those shares are **censored measurements**, taken at
budgets that were themselves truncating the reasoning. Measured at 16384 where it completes,
natural reasoning is ~6,300 tokens, so at `max_tokens=4096` the model is being cut off
mid-reasoning roughly half the time, which is exactly why half those calls returned empty.

**Detector:** empty `content` ⟺ `finish_reason == "length"`, with no exceptions observed.
Truncation is therefore a free and reliable validity check — no response with
`finish_reason == "length"` should ever be trusted.

This is the same failure mode as the gpt-5-mini judge waste recorded on 2026-08-23, and here
it is avoidable.

## Reasoning control — re-measured, and the first result was wrong

**Correction (same day).** The first pass tested every knob at `max_tokens=512` and concluded
that only `thinking` worked and the rest were "silently ignored". That inference was invalid:
512 was the ceiling, and natural reasoning on this prompt runs ~6,300 tokens, so *any* knob
that reduced reasoning to anything above 512 would still report exactly 512 and look
identical to no effect. The test could not distinguish "ignored" from "reduced but still over
budget". Re-measured at `max_tokens=16384`, where reasoning completes naturally.

Design: arms **interleaved and randomised** in one time window (sequential blocks confound the
arm with server-load drift), `temperature=0`, and a **negative control** — an invented
parameter `zzz_not_a_real_param` that cannot possibly do anything. The control is what makes
the rest readable: an arm that does not separate from the control has no demonstrated effect.

n=8/arm interleaved, reasoning tokens:

| Arm | mean | sd | vs control |
|---|---|---|---|
| `NEGCTRL_fake` (invented param) | 6,831 | 1,253 | — |
| baseline (no knob) | 6,300 | 2,903 | −532, se 1,130 → **not separable** ✅ design validated |
| `reasoning_effort: "minimal"` | **1,537** | 506 | **−5,294, se 517 → separable (~10σ)** |

Second interleaved run, n=6/arm:

| Arm | mean | sd | vs control |
|---|---|---|---|
| `thinking: {"type": "adaptive"}` | 4,702 | 1,632 | −288, se 1,652 → not separable |

Corrected verdicts:

| Parameter | Verdict |
|---|---|
| `thinking: {"type": "disabled"}` | **works, deterministically** — reasoning exactly 0, n=4, sd 0 |
| `reasoning_effort: "minimal"` | **works** — ~4.4× less reasoning, and much tighter variance |
| `thinking: {"type": "adaptive"}` | no detectable effect at n=6 (not proven inert — underpowered) |
| `thinking: {"type": "enabled", budget_tokens: N}` | **advisory, not a cap** — at N=1024, mean 2,934 but one sample ran 6,583 |
| unknown parameters | genuinely ignored — the control is statistically identical to baseline |
| invalid `thinking.type` | **400, validated** — 5/5 |

The `thinking` object *is* parsed and validated. An invalid type returns
`unknown variant ..., expected one of `adaptive`, `enabled`, `disabled`` — which is also how
the undocumented `adaptive` mode was discovered.

Baseline reasoning is enormous and enormously variable: mean ~6,300–7,000 tokens, observed
range **298–10,426** at `temperature=0`. Temperature 0 does not make this model's reasoning
length reproducible, and that variance is why small-n comparisons here are worthless.

With thinking disabled at `max_tokens=2048`: 0/4 empty, 0/4 truncated, 6,967 content chars
average — **longer** content than the thinking-on runs produced, and 4 calls in 20.8 s versus
59.3 s. Disabling reasoning costs nothing in output length and is ~3–4× faster.

### Resolved: `"minimal"` is not a legal value, and that is the intermittent 400

The intermittent 400 on `reasoning_effort: "minimal"` (2/8, then 2/10 on a later run) is now
explained. The error message states it outright:

> `'reasoning_effort' must be one of: 'low', 'medium', 'high', 'xhigh', 'max'`

`"minimal"` is **not in the enum**. The strict upstream rejects it; the lenient one accepts it
(and measurably reduced reasoning, which is why the earlier contrast was significant). So the
~20% failure rate is not flakiness — it is one upstream enforcing the schema and the other not.

Two consequences:

* **Use `reasoning_effort: "low"`, never `"minimal"`.** `"low"` is inside the documented enum
  and therefore valid on both upstreams. Anything outside the enum buys a guaranteed ~20%
  hard-failure rate on a status your `_ABORT_STATUSES` treats as fatal.
* **The enum reveals a `"max"` level** — plus `medium` and `xhigh` — none of which appear in
  b.ai's published docs. The full ladder is `low | medium | high | xhigh | max`.

Separately, **`thinking: {"type": "disabled"}` is not honoured by every upstream**: in a 10-call
run one response came back with 165 reasoning tokens despite thinking being disabled. Disabled
is reliable in aggregate but not guaranteed per call, so a caller that depends on zero
reasoning must check `reasoning_tokens` on the response rather than assume it.

## Reasoning at maximum — possible, but only with streaming

The effort ladder on the synthesis prompt, `max_tokens=32768`, `temperature=0`, n=4:

| Arm | reasoning tokens | latency | outcome |
|---|---|---|---|
| `effort_low` | 2,097 | 33.4 s | 4/4 |
| `effort_high` | 4,300 (sd 3,429) | 54.9 s | 4/4 — not separable from default |
| baseline (no knob) | 6,576 | 73.3 s | 4/4 |
| `effort_max` | 9,584 *(biased, see below)* | 99.3 s | **2/4 — two died** |

The default already reasons hard, sitting between `high` and `max`; `high` is *below* default and
statistically indistinguishable from it.

**The two `effort_max` failures were HTTP 524 — a Cloudflare origin timeout, not a rejection.**
Surviving calls averaged 99.3 s, i.e. exactly Cloudflare's ~100 s proxy limit. Non-streaming
max-effort calls are therefore a coin flip: the ones that finish under 100 s return, the rest die.

That also means **the 9,584 figure above is survivorship-biased** — only the fastest calls
survived to be measured, so it is the short tail of the distribution, not the distribution.
Re-measured over SSE, where nothing times out:

| `effort_max` via streaming | value |
|---|---|
| success rate | **4/4** |
| reasoning tokens | **12,573 mean** (7,245 / 12,018 / 13,958 / 17,070) |
| wall time | 83–172 s (one ran 172.4 s with no 524) |
| time to first byte | ~1.0 s |
| content produced | 5,544–7,775 chars — **no longer than baseline's 5,638** |

So maximum effort roughly **doubles reasoning** (12,573 vs 6,576) and produces **no more answer**.
It buys a richer trace, not a better or longer conclusion.

Practical consequences:

* **Streaming is mandatory above ~100 s of generation.** SSE keeps bytes flowing so 524 never
  fires; TTFB is ~1 s regardless of how long the full generation runs.
* **`max_tokens` must be ≥ 32768** at max effort — reasoning alone reached 17,070.
* **`providers.ChatClient` cannot stream today.** There is no SSE path in it
  (`_flatten_chunk_text` handles Mistral's content-block shape, not `text/event-stream`).
  Adopting max-effort reasoning means adding delta accumulation, separate `reasoning_content`
  assembly, usage capture from the final chunk, and retry semantics for partial streams.
* Because the limiter counts requests, none of this costs throughput — only wall-clock,
  worker count, and token consumption.

## Backend heterogeneity behind one model ID

The same invalid request produced **three different error message formats**, and the byte
offset in the deserialisation error differed between calls (`column 125` vs `column 140`) —
meaning one upstream wraps the forwarded payload with ~15 more characters than another.
b.ai is fronting **multiple upstream backends behind the single `deepseek-v4-flash` id**, and
they do not behave identically.

Quantified two independent ways, which agree:

| Signal | Variant A | Variant B |
|---|---|---|
| validation-error shape (n=20) | 16 × raw, `column 125` | 4 × wrapped (`column 140` / other) |
| `system_fingerprint` on 200s (n=133) | 104 × `a26a7955944dc5c60445bff77fac9c8e` | 29 × **absent** |

Both land at roughly **80/20**, so `system_fingerprint` presence is a usable per-call
discriminator for which upstream served a request — worth logging on every call. All response
ids share one UUID shape, so the id is not a discriminator.

The practical consequences: per-call behaviour is not guaranteed stable for identical inputs,
~20% of traffic goes somewhere with different parameter handling, and a 400 should therefore
be **retried once on a fresh connection** before being treated as a malformed-payload abort.

## Quality gate — IPC to BNS mapping

Six mappings with known ground truth, `temperature=0`, `max_tokens=4096`:

| Question | Expected | Got | |
|---|---|---|---|
| IPC 302 punishment for murder | BNS 103 | 103 | PASS |
| IPC 420 cheating | BNS 318(4) | **316** | **FAIL** |
| IPC 376 rape | BNS 64 | 64 | PASS |
| IPC 498A cruelty by husband | BNS 85 | 85 | PASS |
| IPC 304B dowry death | BNS 80 | 80 | PASS |
| IPC 307 attempt to murder | BNS 109 | 109 | PASS |

**5/6.** The failure is a substantive confusion, not a near-miss: BNS 316 is criminal breach
of trust (IPC 406–409), a different offence entirely. Separately, in the first smoke test at a
smaller budget the model said IPC 302 maps to BNS **101** — the *definition* of murder rather
than the punishment section. Both errors land squarely in the pending IPC-BNS audit's
territory, so this model must not be treated as authoritative on transition mappings.

## Recommended use in this project

The pool's existing families are gpt-oss, gemma, qwen, and mistral. **deepseek is a new
family**, which is the strongest argument for adding it: `routing.family_separation` requires
every generator to be judged by a different family, and gpt-oss currently sits on both sides
of that line.

**Primary recommendation — judge / tiebreak slot, thinking disabled.** Free, a new family,
~333 calls/hour, and with `thinking: {"type": "disabled"}` it produces zero empty replies at
`max_tokens=2048` while running ~3× faster. That is a direct answer to the slot-B pool gap
that stalled the live drain, and it avoids the reasoning-budget waste that made gpt-5-mini
expensive for the same job.

**Secondary — supplementary generator, not the primary teacher.** 1.14M tok/hour free is
real capacity, but 5/6 on a basic statutory mapping is below what the bulk of the corpus
should be built from, and Mistral's ~1B tok/month remains the pivotal pool. Reasonable as a
diversity source for a minority of examples, gated behind the citation set-difference check.

**Do not** plan around `deepseek-v4-pro` — it is deposit-gated.

## Integration notes

Add to `configs/data_law_v1.yaml` as a provider with `base_url: https://api.b.ai/v1`,
`api_key_env: BAI_API_KEY`, `quirks: [bai]`, and a new `bai` entry in `providers.QUIRKS`
whose response hook rejects `finish_reason == "length"` as invalid rather than returning
empty content, and whose request hook sets reasoning per role:

* **judge / tiebreak → `thinking: {"type": "disabled"}`.** The only knob that is exactly
  deterministic (reasoning 0, sd 0). A judge does not need a visible trace, and this removes
  the empty-reply failure entirely while running ~3–4× faster.
* **generator → `reasoning_effort: "low"`.** Keeps the trace (which the ≥80%
  reasoning-trace floor needs) while cutting reasoning to 2,097 tokens, so a much smaller
  `max_output` becomes safe. **Not `"minimal"`** — see the resolution section above: it is
  outside the `low|medium|high|xhigh|max` enum and buys a ~20% hard-failure rate on a status
  `_ABORT_STATUSES` treats as fatal. `low` is what shipped in `configs/data_law_v1.yaml`, and
  `docs/reports/2026-08-26-row-length-under-deepseek-traces.md` later confirmed it keeps the
  templated row inside the 8192 training cap (1.5% over, against 85% at baseline).

Suggested `limits`, all measured rather than documented:
`{rpm: 8, tpm: null, max_context: 800000, max_output: 12288}`.

* **`rpm: 8`** — the bucket admits exactly 10/minute and counts *rejected* requests too, so
  leave headroom. `TokenBucket` should be configured per-request, not per-token.
* **`tpm`** is not the binding constraint and should not be the throttle; token volume passed
  freely at 7× the rate that request count throttled at.
* **`max_context: 800000`** — the ceiling is between **844,701 (accepted)** and **900,000
  (`Input token exceed the limit`)**. This supersedes the 18,752 "floor" reported in the first
  pass, which was never a context measurement at all — it was the rate limiter interfering.
  800000 is a safe declaration well inside the measured boundary.
* **`max_output: 12288`** sized for unconstrained reasoning (mean 6,300, observed max 10,426);
  with `reasoning_effort: "low"` on the generator a smaller budget suffices, though the shipped
  config keeps 16384 as a REPLY ceiling because the bai request hook raises a smaller caller
  budget up to it — reasoning is billed here and emitted first.

Because the limiter counts requests and ignores size, **the cheap call and the expensive call
cost the same**. Batch aggressively: one 200k-token call beats twenty 10k-token calls.
`_ABORT_STATUSES` treating 400 as fatal also needs revisiting — see the intermittent 400 on
`reasoning_effort` and the upstream split below.

Two environment gotchas:

1. **The key in `.env` is quote-wrapped and lowercase with a stray space**
   (`beyond_api_key ="sk-…"`). The literal quotes are sent as part of the token and produce
   `401 Invalid token`. Rename to `BAI_API_KEY=sk-…` unquoted so `load_dotenv_keys` and
   `api_key_env` resolve it.
2. **Cloudflare rejects the default `urllib` user-agent** with `403 error code: 1010` before
   the request reaches the API. Any client must send a normal UA. `requests`/`httpx` set one
   already; raw `urllib` does not.

## Head to head against the current generator, `cerebras/gpt-oss-120b`

Measured live on the same prompts, using the exact production params from
`configs/data_law_v1.yaml` (`temperature 0.7, top_p 0.95, reasoning_effort medium,
max_tokens 4096`). Lightning was **not** benchmarked — it is pay-per-token.

| | `cerebras/gpt-oss-120b` | `b.ai deepseek-v4-flash` |
|---|---|---|
| cost | free | free |
| latency / call | **2.1 s** | 33 s (low) → 73 s (default) → 83–172 s (max) |
| reasoning tokens | 668 | 2,097 (low) → 6,576 (default) → 12,573 (max) |
| trace field | `reasoning` | `reasoning_content` |
| content produced | 8,034 chars | ~5,600–6,800 chars |
| `max_output` | **4,096 (hard cap)** | 32,768+ |
| `max_context` | 131,072 | ~800,000 |
| rate limit | rpm 5, tpm 30k, **tpd 1M** | 10 req/min, no daily cap observed |
| **daily capacity** | **~364 examples/day** | ~600/hour, no daily cap seen |
| BNS knowledge (ungrounded) | **0/10** | 7/10 (max) / 9/10 (thinking off) |
| family | gpt-oss — *already on both sides* | **deepseek — new** |

Two findings dominate everything else.

**1. The daily cap, not the speed, is what binds.** gpt-oss-120b is 35× faster per call, but
`tpd: 1000000` against a measured ~2,745 tokens per generated example allows **~364
examples/day**. A 15–20k example corpus is therefore a **41–55 day** run on the free generator.
The same corpus through deepseek is rate-bound at 600 calls/hour — roughly **25–33 hours** of
wall clock — at the cost of ~12 concurrent workers and much longer per-call latency. Speed per
call and throughput per day point in opposite directions here.

**2. The current generator does not know the BNS.** On the ungrounded mapping gate it scored
**0/10**, echoing IPC numbers back as BNS sections (302→302, 420→420, 498A→498A). Direct
questioning confirms it is not a formatting artifact:

* claims the BNS has **447 sections** (it has 358)
* claims **§103 is giving false evidence / perjury** (it is punishment for murder)
* claims the BNS **capped murder at 10 years RI and eliminated the death penalty** — pure
  fabrication; §103 retains death or imprisonment for life

This is fluent, confident invention about the precise statute the corpus is about.

**How much that matters is bounded by grounding, but not eliminated.** `prompt_registry`
supplies `source_text` and `check_citations` rejects any citation absent from it, so invented
*section numbers* are already caught. What is not caught is fabricated *content* about sections
that genuinely appear in the source: "§103 caps murder at ten years" cites a real, in-source
section and passes the citation gate while poisoning the reasoning trace — and traces are what
the ≥80% reasoning-trace floor is made of.

**Caveat on scope.** This gate measures *parametric* knowledge with no source text supplied,
which is not how the production path calls the model. It is a proxy for how much a teacher can
be trusted when it reasons *beyond* the provided excerpt, not a measurement of grounded
generation quality. Deepseek is materially better (7–9/10) but is not authoritative either — it
missed IPC 420→318(4), answering 316.

## Billing, from the provider console — two things `usage` does not tell you

The rate limiter counts requests (above), but *consumption* is metered in tokens, and the
provider's own console disagrees with the API response in two ways that matter.

**1. Rejected oversized requests are billed in full.** The two largest console entries are
**1,192,441** and **1,619,229** input tokens. Neither corresponds to any accepted call — no
successful response reported above 844,701 — so they are the 900k and 1M context probes, both
of which the API refused with `Input token exceed the limit`. A request that returns 400
without producing a single output token still costs its entire input. Probing context limits
is therefore the most expensive thing you can do here, and any production path that risks
oversized inputs should pre-count locally (`context_estimate`) rather than let the API reject
them.

**2. Console input counts exceed `usage.prompt_tokens` on large inputs.** They agree exactly
at small sizes (31,805 / 63,678 / 198,755 all match) but diverge as inputs grow — the 128k
probe reported `prompt_tokens=127,201` and was billed **158,945** (×1.25). The mechanism is
not established, and the ratio is not constant across the sampled rows, so **do not budget
from `usage.prompt_tokens` alone** at large context. Log both and reconcile against the
console periodically.

Total for the context-ceiling work: ≈**4.58M tokens** across nine calls. Balance remained
zero and calls kept succeeding, so nothing was charged — but the accounting exists, which
means a quota plausibly exists behind it even though none was hit.

## Caveats

No published terms, quotas, or rate limits, so free-tier access can change without notice and
without a deprecation path. Balance is zero, meaning there is no paid fallback if the free
allowance is withdrawn mid-run. A daily cap was not reached but was also not disproven.
