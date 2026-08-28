# The two recorded follow-ups, measured: verbatim_overlap and the gen_drafting mandate

2026-08-28. Both investigations are read-only — every store was opened `mode=ro`, no
template, gate, or config was edited, and no teacher was called. The evidence is the
~2,900 generations the build has already paid for, plus the gate's own matcher re-run
offline.

The two follow-ups turn out to be the same story told twice: **a gpt-oss-era
calibration meeting the deepseek-era generator.** `verbatim_overlap`'s 120-char
threshold was fitted on gpt-oss pilot traces and sits at deepseek's *median*; the
drafting templates carry a four-heading answer mandate whose removal the harmony lane
already measured (to 0% failure) but which persists in the base prompts the deepseek
lane reads.

---

# Part 1 — verbatim_overlap is a generator-calibration failure, not a summarization problem

## 1. The frame was wrong: it fails both task types, and only under deepseek

Fail rates at the current threshold (`DEFAULT_MAX_RUN = 120`), every store, every
generation with a gate row:

| generator | irac_analysis | summarization |
|---|---|---|
| deepseek-v4-flash (11 arms, pooled n=1,086) | 41.5–59.6% by arm | 44.2–63.6% by arm |
| gpt-oss-120b (6 post-08-18 arms, n=473) | 0–6.5% | 0–6.9% |

The closeout's "verbatim_overlap ~60% of summarization" undersold it: irac_analysis
fails at essentially the same rate. This is a property of the generator, not the task.

Two false trails, closed:

- **The LIVE store's gpt-oss numbers are stale-threshold artifacts.** LIVE (and its
  `exp_dialect` copy) carry 257 gate rows recorded at `max_run: 30` — the pre-08-18
  threshold — which is why LIVE shows gpt-oss drafting at 44.9%. Every experiment
  store is pure `max_run: 120`. Cross-era comparisons must partition on the recorded
  `max_run`, and the 41–64% deepseek figures above are all at 120.
- **The knobs don't move it.** The 11 deepseek arms span the ceiling edit, the cap
  arm, the clause arm, both prompt-era reruns and all four irac arms; the rate sits
  in the same 41–64% band throughout. (The f2only run's own noise channel already
  showed ±14pp pool drift on identical templates, which covers the spread.)

## 2. The distribution: 120 sits at deepseek's median

Longest run shared between `_norm_ws(think)` and `_norm_ws(seed.text)` — grounding
for these two task types is exactly the seed text, so the offline scan reproduces the
gate bit-for-bit. Binary search per trace over `find_verbatim_run`:

| population | p25 | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| deepseek, both types (n=1,086) | 63 | **127** | 218 | 355 | 936 | 1,340 |
| gpt-oss, both types (n=473) | 0 | **37** | 48 | 75 | 203 | 353 |

Failure rate by candidate threshold, deepseek:

    max_run    30    40    60    80   100   120   150   200   300   500   800
    fails      99%   94%   77%   67%   60%   52%   42%   29%   15%    4%    2%

The 2026-08-18 re-audit set 120 "where the curve flattens" — for gpt-oss pilot
traces, whose curve read 24% at 80 and 13% at 120. Deepseek's curve does not flatten
until ~500. By the re-audit's own method applied to the current generator, 120 is not
a calibrated value; it is the median of the incidental-overlap distribution, which is
exactly the position the audit condemned 30 for.

## 3. What the failures actually are: sentence-length quotes, not transcription

Copied coverage — total characters inside disjoint ≥120-char shared runs, as a share
of the trace — over all 567 failing deepseek traces:

    p25 1.2%    p50 2.1%    p75 3.7%    p90 5.9%    max 19.1%

The median failing trace has **2% of its text** in long shared runs. The matches are
the operative sentences the trace is analyzing ("the appellant has wrongly asserted
that the respondent was aware of the conditio…", "The order passed by the learned
Division Bench of the High Court of Punjab & Har…"). This is a reasoning style that
quotes the passage under discussion — the thing a lawyer's deliberation does — inside
an otherwise self-authored trace. Wholesale transcription (the behavior the gate's
docstring names) would read as tens of percent, and no trace reaches it.

The prompt already forbids the habit ("do not carry sentences over from the
instructions into your reasoning", every template) and the repair hint already names
it. Deepseek does it anyway — consistent with the qualification finding that this
provider's trace behavior resists instruction.

## 4. The rate is strongly length-inflated

Fail rate by think-length quartile, deepseek, both task types:

| think chars | n | fail |
|---|---|---|
| 726–6,777 | 271 | 17.0% |
| 6,786–11,664 | 272 | 41.9% |
| 11,698–19,083 | 271 | 63.8% |
| 19,116–59,851 | 272 | **82.7%** |

The gate is an existence test — one ≥120 run anywhere fails the trace — so at
constant quoting propensity the fail probability compounds with length. A 20k-char
trace fails 5× as often as a 7k one. Two consequences:

- **F2 pushes this gate up on summarization.** F2 lengthens summarization deliberation
  +23.8% at p50 (measured, f2only), and the post-F2 arms show summarization verbatim
  at 61–63.2% against ~50% in the pooled pre-F2 arms.
- The pending summarization `length_band` follow-up and this gate are coupled:
  anything that shortens traces moves both; anything that licenses longer traces
  (a wider band) buys more verbatim failures at the current threshold.

## 5. Retries are a coin-flip, so the gate burns calls without selecting

Across consecutive attempts on the same task (deepseek arms): P(pass N+1 | fail N) =
43.0% (172/400), P(pass N+1 | pass N) = 63.0% — and 37% of passes regress to
failure on the next attempt. The gate is not filtering a stable per-row trait; it is
re-sampling a ~50/50 property per call. Rows mostly wash through eventually; the cost
is paid in extra generations, which is exactly what "gates not judges are the
ceiling" measured.

## 6. The prize, quantified post-F2

In the two post-F2 arms (`exp_irac_fix` + `exp_irac_f2only`), summarization: clean
21.5% (17/79), verbatim failing 62.0%, and verbatim the **sole** failed gate on
13.9pp — fixing this one gate lifts the summarization clean ceiling 21.5% → 35.4%,
a +65% relative gain. Over all 1,086 pooled deepseek generations the solo share is
4.8pp, and it grows as the other gates heal: verbatim_overlap is the frontier the
closeout said it was, and more so after F2.

## 7. Levers, in the order the evidence ranks them

1. **Re-fit the threshold by the 2026-08-18 method, on deepseek traces.** The
   audit's own logic ("at 30 the gate was rejecting the act of thinking about the
   matter at all") applies verbatim at 120-vs-deepseek: it rejects the act of quoting
   the sentence under analysis. The measured curve puts the flattening at ~400–500
   (8%→4%). This is a one-constant change with a doctrine question attached — a
   ~400-char run is a two-to-three-sentence quote, and admitting it into the trace is
   a data-quality call the operator must own, not a tuning detail. Note the recorded
   side-effect: `check_statutory_quotation.reproduces_grounding` reads the same
   constant (diagnostic only, but its meaning shifts again, as the 30→120 note
   already warns).
2. **A/B a targeted anti-quoting clause.** "Trace not steerable" was proven for
   *length*; the anti-rehearsal clause and F2 both proved trace *style* is steerable.
   The current one-sentence prohibition has never been the treatment in an arm. Cheap
   (bai is free), but any arm must run back-to-back with an untreated noise channel —
   this gate drifted ±14pp between pools an hour apart on identical templates.
3. **A coverage-based gate** (share of trace inside long runs, which is what
   "transcription" actually means) would separate quoting from copying cleanly — the
   measured failing population sits at 2% coverage, real transcription would not —
   but it is a gate redesign, not a re-fit, and nothing above requires it.

Doing nothing is also priced now: ~half of every deepseek generation call is spent
re-rolling a coin this gate flips, on both task types, forever.

---

# Part 2 — gen_drafting: the drift is real, the fix is already proven, and the arm must wait for the unpark

## 1. The drift, textually

`prompts/gen_drafting_v1.md:21` and `_v2.md:21` both mandate "the settled work under
four headings, each on its own line — Issue, Rule, Application, Conclusion", while
`gates.IRAC_ANSWER_TASK_TYPES` (`irac_analysis`, `statute_qa`, `transition`)
deliberately stopped requiring headings of drafting answers. Same contradiction the
summarization templates carried until `ebde9a7`: the answer mandate seeds the
scaffolding the think-side tripwire then fails. Both drafting shas are byte-identical
across every overlay ever built (`48534e3010f5` / `618b240ab03e`) — no arm has ever
treated them.

## 2. The mechanism is already measured — twice, and once on drafting itself

The harmony lane's `prompts_harmony/gen_drafting_v1.md:21` already says the opposite:
"produce the {document_kind} itself… **Do not set the answer out as Issue, Rule,
Application, Conclusion.**" The stores hold the before/after, gpt-oss both sides:

| drafting under | arms | n | irac_placement fail | think-rehearsal |
|---|---|---|---|---|
| four-heading mandate | LIVE pilot (cerebras) | 272 | 29.4% | 27.9% |
| four-heading mandate | exp_harmony | 18 | 38.9% | 33.3% |
| genre-form answer (harmony rewrite) | exp_s1, exp_measure, exp_recovery, exp_hybrid | 141 | **0.0%** | **0.0%** |

(irac_placement is unaffected by the max_run threshold change, so the LIVE rows are
usable here; mistral-small's 89% is already recorded in the config and retired with
the model.) Together with F2's −58.64pp/−64.83pp on summarization, the
mandate→rehearsal mechanism is now measured on two task types and two generators,
with the same answer both times.

## 3. Why no arm can run today, and why running one would be wrong anyway

There is not a single deepseek drafting generation in any store — and not by
accident: drafting is **parked at weight 0.00** in `SYNTHESIS_MIX` (2026-08-24), for
a prior, independent defect: all 60,603 seeds render placeholder slots
(`document_kind` / `party_context` empty), so the task asks the teacher to draft "for
the party whose papers these are" against a judgment that already disposed of the
matter. Its exit numbers were 4/20 accepts and 66,666 tokens per accepted row.

A drift arm run now would therefore measure a malformed task nothing plans, on
prompts whose slots are placeholders — a number with no referent. The drift
measurement is downstream of the unpark precondition (seeds carrying the fields, or
the stream retargeted), not of another prompt campaign.

## 4. Disposition

- **The templates stay untouched**, per the closeout's own rule (no edit without an
  arm) — and nothing needs them before the unpark.
- **The arm design is settled now**, so the unpark inherits it: ctl = current base
  templates, fix = the F2-pattern genre-form rewrite of the answer contract (the
  harmony drafting text is the ported precedent; the deepseek-era length sentences
  and anti-rehearsal clause carry over verbatim), back-to-back, untreated task type
  as the noise channel, pre-registered on irac_placement fail and think-rehearsal
  rate. Prediction on two prior measurements: rehearsal collapses to ~0.
- **The docstring risk is already fenced**: chunks.py no longer claims PredEx/
  TathyaNyaya are "short excerpts", so nobody will chunk them to fix a length tail;
  the analogous trap here would be editing the drafting templates casually because
  the fix is "already proven" — proven on gpt-oss and on summarization, not on
  deepseek drafting, which is exactly what the arm exists to close.

---

## Method note

All figures from read-only SQL over the 21 build stores plus offline re-runs of
`gates.find_verbatim_run`/`gates._norm_ws` against `seed.text` (the exact grounding
for these two task types — `GROUNDING_SLOTS` renders only `{source}` for them, so
the offline scan reproduces the gate's corpus byte-for-byte). Longest-run values by
binary search over `max_run`; coverage by mask-and-repeat at 120. Live store
fingerprint untouched; no writes anywhere.
