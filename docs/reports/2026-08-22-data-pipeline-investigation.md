# Cursor report — law_v1 data pipeline

**Date:** 20 Aug 2026 (diagnosis) · 21 Aug 2026 (hybrid, dialect, Harmony+s1, live drain)  
**Workspace:** `C:\Users\Anant\Desktop\projects\tuned`  
**Scope:** What `.claude/worktrees` is, why synthesis reject/yield looks catastrophic, why teacher format is not stable, annotated traces, and the isolated hybrid-yield experiment against that baseline.

Source of numbers: worktree SQLite store  
`tuned/.claude/worktrees/law-v1-data-pipeline/data/build/state/law_v1.sqlite3`  
(1,060 planned synthesis tasks. 20 Aug snapshot was 1,386 gens / 6 accept / read-only. **21 Aug addendum §14:** 1,396 gens / 15 accept; judges are writing.)

Experiment store (own sqlite, own logs):  
`tuned/.claude/worktrees/law-v1-data-pipeline/data/build/exp_hybrid/`

This file is a write-up of a Cursor investigation plus the probe that followed. It is not a training-lane change on `main`. The live synthesis wave has **not** been promoted.

---

## 1. What `.claude/worktrees` is

`.claude/worktrees` is Claude Code’s isolated git checkout folder. The whole `.claude/` tree is gitignored as agent scratch.

There is **one** live worktree:

| Checkout | Path | Branch | HEAD |
|---|---|---|---|
| This Cursor workspace | `C:\Users\Anant\Desktop\projects\tuned` | `main` | `c3c3651` (training lane) |
| Linked worktree | `tuned/.claude/worktrees/law-v1-data-pipeline` | `worktree-law-v1-data-pipeline` | `46f0311` |

It shares the same `.git` object store. Merge-base with `main` is still `c3c3651`. The branch has **179 commits**, has **never been pushed**, and has **not been merged**. Working tree is clean.

It is the **law_v1 dataset-curation pipeline**: target 18,000 rows (MVP 10,300), mix 60% grounded synthesis / 16% curated / 24% replay, ≥80% reasoning-trace floor, then Hub push for the Qwen3-8B DDP lane.

Runtime sitting in that checkout (not in git): `data/build/` (job DB ~528 MB, streams, raw logs), `pulled/law_v1_corpus/` (AWS PDFs 2018–2024), `.venv`, `.env`, and an SDD ledger under `.superpowers/sdd/lets-plan-about-data-sparkling-axolotl/`.

Replay (4,320) and curated C1 (1,100) are already on disk. They do not go through the teacher ritual. The dying stream is **synthesis**.

---

## 2. The reject rate is the wrong denominator

Of **1,060** planned synthesis tasks:

| State | Count | Meaning |
|---|---|---|
| `stale_prompt` | 419 | Templates changed; never generated |
| `rejected` | 392 | Terminal |
| pending / judging / generating | 243 | In flight |
| `accepted` | 6 | `judge:accept` |

Yield of rows that were actually attempted: **6 / 641 = 0.9%**.

Of the 392 rejects:

- **359 (91.6%)** — `exhausted:regenerate:*` after 3 identical retries. Format gates, not law.
- **24** — `judge:reject`.
- **9** — permanent citation gate (`reject:citations`).

`temporal` and `answer_key` have **never fired**. This is not a hallucination-of-statutes problem.

Almost every generation **did think**: **1,327 / 1,386** have a non-empty think block. Only 59 are missing tags. “It thought” is the default. The 6 accepts are not “the ones that reasoned.”

---

## 3. What a passing row actually is

The product is not “a good Indian-law answer.” It is a chat example the Qwen3-8B trainer can pack at 8192, with a reasoning trace the student will imitate.

Empirical spec from the 6 accepts (all `gpt-oss-120b`, think **562–826** tokens):

1. **Grounding** — a real judgment/statute chunk. Every section, article, and case name in the answer must appear in that chunk.
2. **Think** — first-person deliberation, **500–3000** estimated tokens, containing at least one **exact cue** (`let me check`, `let me think`, `actually`, `am I sure`, `double-check`, …). No IRAC headings. No 120-character copy of the chunk. No “the excerpt” / “the provided text”.
3. **Answer** — `Issue` / `Rule` / `Application` / `Conclusion` on their own lines. Quotation allowed here. ~250–450 words.
4. **Judges** — two families, min(grounding, validity, coverage) **≥ 4** on a 1–5 scale. Thresholds are still provisional (`judge_threshold` is empty; 46 gold labels).

Cues actually present in the 6 accepts: `actually`, `double-check`, `unsure`.  
**“Let me think” would help it pass.** That phrase is on the allow-list. The rejects are the opposite: `cues=[]`.

---

## 4. Gate failures across 1,386 generations

Gates never short-circuit. Fail counts:

| Gate | Fails | What it checks | What failed in this run |
|---|---|---|---|
| `self_verification` | 919 | Think must contain ≥1 ritual cue | Every fail is `cues=[]`. gpt-oss restates the prompt instead of using doubt language. |
| `length_band` | 561 | think 500–3000, answer ≥120, packed total ≤8192 | 439 `think` below min (failed think_est p50=420). 76 above max. 30 empty answers. |
| `irac_placement` | 317 | Issue+Conclusion in the **answer**; none of those headings in **think** | 260 fails are complete answers that also scripted IRAC inside think. |
| `verbatim_overlap` | 274 | No long copy of the grounding chunk inside think | 170 fails under the old **30-char** run (legal boilerplate). 104 under today’s 120. |
| `banned_meta` | 223 | No “the excerpt”, “the provided text”, “the source says” | Model talks about the prompt packet. |
| `statutory_grounding` | 89 | Section/article numbers in the answer must appear in the materials | Mostly Article 14/21 and extra IPC sections the chunk never showed. |
| `think_format` | 59 | Exactly one think-tag pair | All 59 are not-exactly-one-pair. magistral-small often emits no tags. |
| `citations` | 11 | No invented reporter cites | Only permanent law gate that has fired. 9 tasks burned. |

**165 / 1,386** generations pass every gate. The rest fail 1–5 of them **even with a think block**.  
**562 of 946** traces that were already ≥500 think tokens still had **zero** cues.

Yield by task type (attempted rows only, stale excluded):

| Task type | Attempted | Rejected | Rate | Accepted |
|---|---|---|---|---|
| drafting | 142 | 116 | 82% | 2 |
| irac_analysis | 249 | 155 | 62% | 4 |
| summarization | 105 | 61 | 58% | 0 |
| statute_qa | 145 | 60 | 41% | 0 |

Drafting is the worst because IRAC-on-a-pleading is an unnatural contract. Statute QA is the most salvageable and still has **0 accepts** (43 sitting in `judging`).

---

## 5. The 6 vs everyone else

The 6 survived because thinking **also** matched a stack of extra rules at once (AND, not OR):

- Think 562–826 tokens (above the 500 floor).
- A cue somewhere: `actually` / `double-check` / `unsure`.
- IRAC headings only in the answer.
- No packet-talk (“the excerpt”, “the provided text”).
- Two judges, min score ≥ 4.

gpt-oss’s default think habit is instruction restatement:

> “We need to produce a reasoning in first person, present tense…”

That is not deliberation. It is the model repeating the prompt. That one habit trips four gates at once: no cue, think too short, IRAC leaked into think, packet-talk.

The 6 often still **start** with that same line. They slipped through because later they also hit a cue, stayed in band, and kept headings out of think.

Of rows that reached a **judge decision**: **6 accept, 24 reject**. Kill axis is **validity** (slot A mean 2.63, slot B mean 2.88) against a provisional pass of min-axis ≥ 4. Grounding is much higher. `judge_threshold` has 0 rows.

**109** tasks are still in `judging`. Only 4 of those 109 have any judgement recorded. The judge fleet is not draining. 54 of 109 still carry a leftover `regenerate:*` disposition.

---

## 6. Accepted vs rejected examples (annotated)

Real traces from the store. **PASS** = what the gates require. **FAIL** = what burns the row. Unmarked lines are the shared habit (prompt restatement) on both sides.

Highlighted side-by-side in the Cursor canvas: `canvases/law-v1-rejection-audit.canvas.tsx` (green / red diff lines).

### 6.1 Same opening, one cue — accept vs burn

**Accepted think** — task `63de3214c67d333e` · `irac_analysis` · `gen_irac_analysis_v2` · gpt-oss-120b · **562** think tokens · `judge:accept`

```
The user wants advice with reasoning first (450-700 words) then advice under
headings Issue, Rule, Application, Conclusion.
Must not refer to text as source. Must use only Shanbaug, Gian Kaur, P. Rathinam,
and IPC sections.
We need to think through the issue: Article 21 and withdrawal of life support
for a person in PVS.
PASS  We need to double-check: Section 302 (murder) or 304 (culpable homicide)
      for active euthanasia; Section 306 for abetment of suicide.
Common Cause v. Union is not in the materials, so we cannot rely on it.
PASS  cue hit: "double-check". think_tokens=562 (above 500). no IRAC headings
      in think.
```

**Accepted answer** (headings live here, not in think):

```
I begin by laying out the factual and legal landscape... Shanbaug and Gian Kaur.
PASS  **Application**
Because Shanbaug leaves passive euthanasia untouched, any unilateral withdrawal
could be culpable homicide. Obtain a court order.
PASS  **Conclusion**
The Constitution does not currently empower you to discontinue life-support
without a court's blessing.
```

**Rejected think (no cue)** — task `43b870ab53eabc80` · `summarization` · `gen_summarization_v1` · lightning gpt-oss-120b · **704** think tokens · `exhausted:regenerate:self_verification` after 3 attempts

```
FAIL  We need to produce reasoning first (internal) then headnote with four
      headings. 450-700 words, then headnote 250-450 words.
Facts: Selection for 4010 posts; final result 16 March 2015; High Court
excluded whitener/blade candidates.
Supreme Court: Rule 15(h) sets minimum marks; whitener/blade is not a listed
disqualification. Accommodate over and above.
FAIL  think_tokens=704 (longer than the accept) but cues=[]. never says
      actually / double-check / let me think. 3 attempts, then burned.
```

The rejected row is **on-topic and longer**. It dies because the think never contains a ritual cue. Length is not the difference.

### 6.2 IRAC scripted inside think — reject

**Rejected think** — task `0ddd8daf119e3b3f` · `summarization` · `gen_summarization_v1` · **717** think tokens · `exhausted:regenerate:self_verification,irac_placement`

```
FAIL  We need to produce a headnote with Issue, Rule, Application, Conclusion.
FAIL  Thus the issue: Whether the appellate court should interfere with the
      trial court's finding on desertion.
FAIL  Rule: For judicial separation under Section 10(1)(a) HMA, desertion must
      be continuous for two years.
FAIL  Application: Key witness vague; independent witnesses do not corroborate;
      two-year desertion not proven.
FAIL  Conclusion: Appeal dismissed; trial court stands.
FAIL  IRAC headings belong in the ANSWER only. Scripting them in think is
      irac_placement, and there is still no cue.
```

That think is a draft of the answer. The gate treats that as fake reasoning.

---

## 7. Teacher format is unfixed

The pipeline asks for a dialect in English, then checks after the fact. It does **not** pin the reply shape.

There is no JSON schema, no grammar, no constrained decoding, no few-shot of a passing think **on the live wave**. The 21 Aug probe added a frozen prior turn behind the exp config only (§10). The generator call is still:

- messages
- `temperature: 0.7`
- `top_p: 0.95`
- `reasoning_effort: medium`
- `max_tokens`

Then gates grep the result.

That is why “same type of example, same prompt” still yields different shapes:

1. **It is not the same prompt.** Four templates per task type (`v1`…`v4`), hashed from `(seed_id, sample_ix)`. `{source}` is a different judgment chunk every time. Templates also moved mid-run — 419 `stale_prompt` rows because `planned_sha != live_sha`.
2. **Temperature 0.7 is a sampler.** Format-following (whether this draw says `actually,`, puts IRAC in think, or narrates “the excerpt”) is stochastic.
3. **Retries re-roll the same request.** The reasoning-effort ladder was retired 18 Aug after it made traces 5× longer and dirtier. Attempt 2/3 is another sample from the same distribution, which is why 359 rows burned three times on the same style failures.
4. **Think channel ≠ answer channel.** gpt-oss returns hidden `reasoning_content` plus the visible answer. The prompt can describe the answer (Issue / Rule / Application / Conclusion). It cannot lock what the model mutters in the reasoning stream.
5. **The serving stack is not one model.** Cerebras `gpt-oss-120b`, Lightning’s host of the same family, and an earlier Mistral slice (different think wrapping).

The gates **are** the format spec. They run after generation. That is why yield is 6 rows, not a stable teacher style.

---

## 8. What this is not

- Not a bad-seed problem. 60,603 seeds are in the store.
- Not a citation-hallucination problem. Permanent law gates are almost silent.
- Not “the model didn’t think.” 96% of generations have a think block.
- Not “the model said let me think.” That phrase is a **pass** cue. Failures are missing those phrases.

The teacher is producing on-topic Indian-law answers. The pipeline is refusing the **shape** of the reasoning, then the judges are refusing the **validity** of the ones that look right. Replay and curated C1 already meet the trainer contract because they are not generated under this think ritual.

---

## 9. Levers (what the diagnosis implied)

If the goal is synthesis yield, the lever is not “get better AWS judgments.” What the 20 Aug evidence actually named:

- Stop burning seeds after three identical format misses (`exhausted:regenerate` is 91.6% of rejects).
- Few-shot a passing think dialect instead of grepping cues the teacher never produces.
- Modest style retarget (`think_min` 500→400; drafting need not wear Issue/Conclusion headings). Do **not** loosen `citations` / `temporal` / `answer_key` / `statutory_grounding`.
- Drain judging and fit `judge_threshold` only after format yield is real. Thresholds are still empty (46 gold labels).
- Leave temperature 0.7; do not revive `EFFORT_LADDER_RETIRED`.

Those were implemented on 21 Aug as an **isolated probe**, not as a live-wave change. Results are §10–§12.

Replay 4,320 + curated 1,100 already exist and are the easy 24% + 16% of the mix. The 60% synthesis stream is still what is stalled on the live store.

---

## 10. What ran (21 Aug 2026) — live store frozen

Work landed only on branch `worktree-law-v1-data-pipeline`. The live sqlite was opened `mode=ro` only. No `--reopen`, no reject migrate, no judge drain on that file.

Experiment config: `configs/data_law_v1_exp_hybrid.yaml` (`extends` the shipped yaml).

| Knob | Live (control) | Experiment |
|---|---|---|
| workdir | `data/build` | `data/build/exp_hybrid` |
| `think_min` | 500 | 400 |
| few-shot prior turn | off | on (`src/tuned/data/fewshot/`) |
| drafting answer IRAC | required | skip headings; think-side IRAC still fails |
| drafting templates | `prompts/gen_drafting_v1/v2` (SHAs unchanged) | overlay `prompts_exp/` (pleading body) |
| format exhaustion | was `rejected` | now `format_parked` (code in worktree; **not** migrated on live) |

Frozen baseline file: `data/build/exp_hybrid/baseline.md`. Live task states after the probe are still the control snapshot: `accepted=6 generating=8 judging=109 pending=126 rejected=392 stale_prompt=419`.

---

## 11. Measurement A — offline re-gate (free)

Same 1,386 stored generations, experimental `GateContext` (`think_min=400`, drafting skip). Few-shot and overlay cannot be scored on old traces.

| | Stored | Re-gate |
|---|---|---|
| all-gates-pass | **165 / 1,386 (11.9%)** | **223 / 1,386 (16.1%)** |
| `length_band` fails | 561 | 375 (−186) |
| `verbatim_overlap` fails | 274 | 138 (−136) |
| `self_verification` fails | 919 | 854 (−65) |
| `irac_placement` fails | 317 | 313 (−4) |

67 stored fails became passes. 9 stored “all-pass” rows failed on first evaluation of `statutory_grounding` (that gate was missing from their stored `gate_result` set). **0 True→False flips** on any law gate that had actually been stored. `citations` / `temporal` / `answer_key` did not move.

Report: `data/build/exp_hybrid/re_gate.md`. Because all-gates-pass **rose**, the 50-row paid probe was allowed to run.

---

## 12. Measurement B — 50-row generate (paid)

50 new synthesis tasks in the exp store only (`arm=exp_hybrid`): 20 irac_analysis, 13 statute_qa, 9 drafting, 8 summarization. Teacher still Cerebras/Lightning `gpt-oss-120b`, T=0.7. 114 generations (retries included).

**Primary metric: 25 / 50 latest generations pass every gate = 50%.**

Baseline was 11.9%. Plan bars: promote candidate ≥20/50; stop &lt;10/50; 10–19 report and wait.

| Task type | n | All-gates-pass | Rate |
|---|---|---|---|
| statute_qa | 13 | 9 | 69% |
| irac_analysis | 20 | 10 | 50% |
| summarization | 8 | 3 | 38% |
| drafting | 9 | 3 | 33% |
| **all** | **50** | **25** | **50%** |

Exp task states after the probe: `judging=25` (the 25 that cleared gates), `format_parked=22`, `rejected=3`.

The 22 parks are still mostly `self_verification` (no cue), then length / banned_meta / IRAC. The 3 rejects are permanent `citations` (+ statutory_grounding on two). That is the intended split: style exhaustion no longer looks like “wrong about the law.”

Judges were **not** the go/no-go. The 25 judging rows have not been drained; `judge_threshold` is still empty.

---

## 13. Decision — promote candidate, live not promoted

25/50 beats the 20/50 bar. That is **not** the same as shipping it onto the live wave.

Still not done, on purpose:

- Shipped `configs/data_law_v1.yaml` still has `think_min: 500` and no few-shot.
- Live `EXPECTED_SHAS` / `prompts/` unchanged.
- Live `exhausted:regenerate:*` rows still sit in `rejected` (359). The `format_parked` state exists in code only for new runs.
- Live `stale_prompt` (419) and `judging` (109) untouched. Eight live tasks are still `generating`.

If promoting: merge exp knobs into the shipped yaml, move `prompts_exp` → `prompts` and re-pin SHAs, disposition-guarded migrate of `exhausted:regenerate:%` → `format_parked`, then `--reopen stale_prompt format_parked --reset-attempts` on live, then drain judges on the current qwen/gemma pool (thresholds still provisional).

If not: leave live as the control. Next conversation is a different teacher or more few-shot on drafting/summarization, still in `exp_hybrid`.

---

## 14. Addendum — 21 Aug 2026 (Harmony+s1, live drain, $2 OpenAI cap)

This supersedes the “live sqlite frozen / 6 accept / 109 judging” snapshot in §10. Training lane on `main` is unchanged.

### Live Chat Completions store

| State | 20 Aug | 21 Aug ~16:08 IST |
|---|---:|---:|
| accepted (`judge:accept`) | 6 | **15** |
| judging | 109 | **76** (+1 `judging_active`) |
| rejected | 392 | **414** (`judge:reject` 24 → 39) |
| pending | 126 | 127 |
| generating | 8 | 8 |
| stale_prompt | 419 | 419 |
| generations | 1,386 | 1,396 |

Attempted yield **15 / ~641 ≈ 2.3%**. Among judged format-passers **15 / (15+39) = 28%**. Validity still kills. OpenAI ledger on this file: **0 requests** (live yaml still declares gpt-5-* as `family: gpt-oss`, so family separation skips them on gpt-oss-120b rows). A judge worker is draining Qwen + Gemma + Mistral.

> **Correction (23 Aug 2026).** That drain has since **stalled, on routing rather
> than on validity.** The 76 (+1) judging rows are now 43 `judging` + 34
> `judge_error`, with **zero** new accepts or rejects — accepted is still 15 and
> rejected still 414. All 34 carry one disposition: `judge-slot-b: role 'judge':
> no eligible model (skipped: cooling, family-excluded)` (235 `judge_route_error`
> events). Slot B had nobody left after gemma + gpt-oss family exclusion.
> Reopening those tasks before the pool has an eligible slot-B member will just
> re-park them.

### Harmony+s1 isolated store (`exp_harmony`)

Official [openai/harmony](https://github.com/openai/harmony) render/parse + simplescaling/s1 `" Wait"` continue. Generator **cerebras/gpt-oss-120b only**. Overlay strips the 450–700 packet. Live prompt SHAs unchanged.

| Wave | n | Token-1 | Format | Judge |
|---|---:|---|---|---|
| First probe | 12 | 12/12 | 11/12 (92%) | **3 accept / 9 reject** |
| Expanded (same arm) | 36 total | first wave held | **27/36 (75%)** | 3 accept, 18 reject, **15 still judging** |

Wave 2 generate (Cerebras): claimed=21, gen-ok=21, gated-out=14, errors=0, ~108k tokens that run; day ledger gpt-oss **150 requests / ~407k tokens**. Format killers on the new exhausts: `irac_placement`, `self_verification`, `statutory_grounding` — not missing cues from the prefill.

Judge scores (40 judgement rows): mean grounding **3.60**, validity **3.17**, coverage **4.00**. Validity hist: v1=2, v2=13, v3=11, v4=4, v5=10. Gemma still the harsh slot vs Qwen.

### OpenAI $2 hard cap

Operator: OpenAI may judge, **$2 TOTAL** across `gpt-5-mini` + `gpt-5-nano`, hard (no tpd probe-grant overshoot). Prices in yaml: mini $0.25/$2.00 per 1M in/out, nano $0.05/$0.40.

- Live yaml: `usd_cap: 2.0` but family still `gpt-oss` → key unused on live gpt-oss rows.
- Exp yaml: family **`gpt-5`** so Harmony rows can be graded if Qwen/Gemma cannot serve.
- Spend so far: **$0** on both stores. *(True when this addendum was written;
  superseded the same evening — see the correction below.)*

> **Correction (23 Aug 2026).** Between **16:11 and 18:44 IST on 21 Aug**, after
> this addendum's snapshot, `exp_harmony` ran **124 `openai/gpt-5-mini` judge
> requests — 377,537 prompt / 122,607 completion tokens ≈ $0.34** of the $2 cap.
> The live store is still genuinely at 0 OpenAI requests, for the `family:
> gpt-oss` reason given above.
>
> **~78% of that spend bought nothing.** 95 of the store's 96
> `judge_parse_error` events are empty replies (`no object found: ''`), all from
> `gpt-5-mini`; only **27 usable judgements** came out of 124 requests. Cause:
> judge calls send `max_tokens=1024` (`DEFAULT_JUDGE_REPLY_TOKENS`), the
> `openai` quirk in `providers.py` renames it `max_completion_tokens`, and the
> gpt-5 family bills *reasoning* tokens against that same budget while the exp
> yaml leaves `params: {}` — no `reasoning_effort`. Mean completion was **989
> tokens/request**, i.e. essentially every call spent its whole reply budget
> thinking and returned empty content. **Before spending the remaining ~$1.66:**
> set `reasoning_effort: minimal` on both gpt-5 refs, and/or raise the reply
> allowance for that ref alone.
>
> The 1 non-empty parse failure is a separate alias gap: the model emitted
> `ground_faithfulness`, the alias table accepts `grounding_faithfulness`.
>
> One caveat for the accept counts above: `exp_harmony`'s accept lift (3 → 8) is
> confounded by judge composition — gpt-5-mini's mean grounding is **4.48** vs
> gemma's **3.12** on the same store. Validity remains the kill axis for every
> judge (2.86–3.74).

Closed-API **generations** (OpenAI/Gemini as teacher) stay out of the training mix (spec line 14 / ToS).

### Tests

`test_build_{generate,judge,providers,harmony,config}`: **459 passed**, including `test_openai_usd_cap_is_hard_and_shared_across_models`.

### Do not promote

No live yaml Harmony flag, no live prompt SHA bump. Token-1 is solved on Completions. Accept rate among format-passers is still ~20–28%. Fit `judge_threshold` (46 gold, 0 rows) before any promote.

---

## 15. Foundation recovery work — 22 Aug 2026

### 15.1 Purpose and isolation

This section records the separate **law-v1 foundation** implementation. It does not promote the Harmony/hybrid experiments, reopen the live store, run a recovery generation, or prepare a Qwen3-8B training dataset.

The target recovery checkout was unsafe to edit because it contains substantial uncommitted Harmony/recovery work. A clean linked worktree was created instead:

| Checkout | Path | Branch | Role |
|---|---|---|---|
| Existing recovery/Harmony worktree | `tuned/.claude/worktrees/law-v1-data-pipeline` | `worktree-law-v1-data-pipeline` | Preserved unchanged; contains dirty recovery/Harmony implementation and live/experiment artifacts |
| Foundation worktree | `tuned/.claude/worktrees/law-v1-foundation` | `law-v1-foundation` | Clean branch for the store-free foundation contracts below |

The foundation branch starts from `46f0311362a25b1cc4f2da2eb7a66ad764af761b`. Its Python 3.12.13 environment is `.claude/worktrees/law-v1-foundation/.venv/`. A clean-worktree baseline completed:

```text
3108 passed, 4 skipped in 957.21s
```

The first baseline invocation failed because its requested pytest `--basetemp` parent did not exist; that was an environment/scratch-path setup error, not a code failure. Re-running against the ignored `.superpowers/sdd/.../pytest-baseline` directory produced the passing result above.

All foundation changes are intentionally **uncommitted**. No control SQLite database, `data/build` artifact, provider endpoint, recovery YAML, live prompt, or main/training-lane file was changed.

### 15.2 Data and label decisions

The following operator decisions govern this branch:

1. **Statute scope:** English BNS, BNSS, and BSA only for v1. The official Gazette Act PDFs are the canonical enactment evidence; official Gazette commencement notices are the canonical in-force evidence.
2. **India Code role:** India Code is not used as a scraped provision source in this v1 slice. It remains a later, pinned reading-copy option for old codes (IPC/CrPC/IEA), where a Gazette-only historical text could be stale.
3. **Distribution posture:** do not emit a raw bare-Act corpus or publish statute text. Any later local dataset row must pair official text with original legal reasoning.
4. **Language:** Hindi provision text is omitted from v1.
5. **Labels:** all current labels are user-attested **Fable 5 model-generated references**, not human gold. They may support later model-agreement comparisons only; they cannot calibrate judges, activate thresholds, or prove legal correctness.

The last point supersedes contradictory prose in:

`tuned/.claude/worktrees/law-v1-data-pipeline/data/build/gold/gold_todo.md`

That file says labels were written by a person, and the existing live calibration code was designed around human labels. The live control database was not opened, migrated, relabelled, or deleted. The implementation below only prevents *newly attested model-reference* records from entering the human-gold path.

### 15.3 Completed, review-gated foundation contracts

The execution ledger is:

`tuned/.claude/worktrees/law-v1-foundation/.superpowers/sdd/law-v1-foundation_8ed15962/progress.md`

All five tasks completed with focused red/green tests and an independent code review. Important review findings were fixed and re-reviewed; minor hardening suggestions were recorded but did not block acceptance.

#### Task 1 — Fable 5 reference-label quarantine

Files:

- `src/tuned/data/reference_labels.py` (new)
- `src/tuned/data/store.py` (modified)
- `src/tuned/data/calibrate.py` (modified)
- `tests/test_build_reference_labels.py` (new)

Reason: the control schema has no provenance field on `gold_label`, and `Store.open()` can mutate a database through schema setup. Adding a column/table or reading the live control file would violate custody. The branch therefore uses a caller-attested in-memory `ModelReferenceLabel` type rather than a schema migration.

Behavior:

- `parse_model_reference_todo(...)` reuses the existing block grammar but ignores the file’s human-authorship prose; the caller must supply `LabelAttestation(kind="model_reference", model="fable-5")`.
- Reference labels are joinable only by `gen_id`, and comparison rows state `claim="model_agreement_reference"` rather than legal accuracy.
- `Store.upsert_gold_labels(...)` rejects typed references, mappings marked `kind`/`label_kind="model_reference"`, and both nested-attestation serializations (`dataclasses.asdict(...)` and `label.__dict__`) **before** SQL packing/writing.
- `calibrate.ingest_gold(..., attested_kind="human")` refuses non-human attestation before folds/writes. CLI `--ingest` now requires `--attest-human-gold`; `--export` and `--fit` are unchanged.
- The module has no provider, network, fold-assignment, calibration, or threshold-write seam.

Verification:

```text
156 passed
```

The review initially found the nested-attestation serialization bypass. It was fixed in `reference_labels.py` and pinned by a test proving both serialized forms leave `gold_label` empty.

#### Task 2 — Gazette registry and distinct statute-QA attachment

Files:

- `src/tuned/data/gazette.py` (new)
- `src/tuned/data/resources/gazette_manifest_v1.jsonl` (new identity/provenance manifest)
- `src/tuned/data/seeds.py` (modified with `attach_statute_qa_section`)
- `tests/test_build_gazette.py` (new)

Reason: the live store has `section_text` on **0** seeds, so statute-QA rows cannot meet the already-existing rule that provision text must be distinct from the judgment/source text. Passing a judgment chunk or transition mapping note as `section_text` would fabricate statutory grounding.

Behavior:

- Parses fixture-supplied English Gazette Act text for BNS (Act 45/2023), BNSS (Act 46/2023), and BSA (Act 47/2023).
- Requires Act/section identity, official eGazette document URL, Gazette publication identifier, retrieval time, SHA-256 source digest, source kind, and optional in-force/as-of fields.
- Refuses malformed documents, missing sections, duplicate/conflicting provision records, incomplete bodies, blank text, source-equal text, unofficial URLs/IDs, and Act-mismatched commencement notices.
- Normalizes attached `section_text` and preserves source URL/ID/retrieval/digest/provenance in seed metadata.
- Uses no network in parser/tests and does not fetch or write real Act bytes yet.

Verification:

```text
318 passed
```

Review correction: the BSA commencement placeholder initially held only the eGazette host plus `S.O. 849(E)`. It was removed rather than guessed. The BSA in-force record remains unset until a real Gazette Extra document URL and identifier are independently verified. Commencement parsing now also requires evidence for the requested Act, so an Act 45 notice cannot stamp Act 47.

#### Task 3 — Store-free Stage-1 publish gate

Files:

- `src/tuned/data/stage1_publish.py` (new)
- `tests/test_build_stage1_publish.py` (new)

Reason: later isolated assembly must refuse unsafe rows before publication, but the current generation-time citation code can skip a citation check when no index is provided. The gate cannot use `Store.open()` because that would risk mutating/read-opening the control database.

`gate_rows(...)` is a pure in-memory gate. It:

1. Requires non-blank source/license and a lower-case SHA-256 `digest` or `source_digest`.
2. Requires official `gazette_act` provenance whenever `_prov.section_text` is present; commencement evidence cannot carry provision text.
3. Requires a caller-supplied citation index and runs both novel-citation and keyed suspect-citation checks over the full assistant content against the user grounding.
4. Requires a caller-supplied evaluation-ID set and rejects case-ID intersections sourced from row provenance and the user prompt. Answer-only identifiers are intentionally not treated as overlap.
5. Returns the original kept objects plus first-fault refusal records. `None` citation index/evaluation IDs refuse the batch; empty index/set remain valid and are still evaluated.

Verification:

```text
140 passed
```

No Store, SQLite, config, `data/build`, network, provider, assembly, or CLI path is imported.

#### Task 4 — Store-free 80-pair recovery-cohort contract

Files:

- `src/tuned/data/recovery_cohort.py` (new)
- `tests/test_build_recovery_cohort.py` (new)

Reason: committed `transition.py` already handles the 1,100 transition training cells and 150 held-out reserve. What was missing was a reviewable pair-selection contract for the later recovery experiment. The dirty worktree’s recovery evaluator was not copied.

`select_recovery_cohort(...)`:

- uses the committed `tasks.task_id_for(seed_id, task_type, prompt_id, sample_ix)` as pair identity; `arm` is metadata, never identity;
- selects exactly 20 records each for `irac_analysis`, `statute_qa`, `drafting`, and `summarization` (80 total), ordered by `task_id`;
- refuses held-out/oversize candidates, transition-stream candidates, unsupported strata, missing source/license, task-ID mismatches, and already-present treatment IDs in defined first-fault order;
- fails closed with `underfilled_stratum:<task_type>` and never returns a partial cohort;
- performs no Store/config/path/data-build/manifest/provider/network operation.

Verification:

```text
76 passed
```

#### Task 5 — Hash-addressable in-memory manifest serializer

Files:

- `src/tuned/data/recovery_manifest.py` (new)
- `tests/test_build_recovery_manifest.py` (new)

Reason: an eventual recovery experiment needs an auditable cohort record, but actual control candidates, official Act bytes/section extraction, isolated experiment storage, and provider authorization are not ready. A serializer can document caller-supplied in-memory selection without falsely claiming that a control cohort is frozen.

The document declares:

```text
schema: recovery_cohort_manifest_v1
kind: selected_cohort_document
claim: in_memory_selection_record
```

Behavior:

- Always reruns Task 4’s selector; it accepts no pre-made list of 80 IDs.
- Snapshots only task identity, stream, source/license, and optional `arm`. It never serializes `section_text`, source text, messages, `_prov`, `meta_json`, generated text, model, prompt SHA, or other candidate payload.
- Hashes offered candidate metadata in encounter order and sorted existing treatment IDs. Caller-provided required hashes must match; valid extra hashes are retained for future real-source provenance.
- Canonically serializes JSON, recomputes/document-validates `document_hash`, validates 80/20 stratum counts and task IDs, and rejects tampering.
- Reads/writes only an explicit absolute caller path. It does not create parents or choose a default path, and refuses paths under `data/build` or with an `exp_recovery` component.

Verification:

```text
47 passed
```

Review correction: an otherwise eligible candidate missing `stream` originally raised a raw `KeyError`. `_snapshot_pair` now raises the auditable token `pair_field_missing:stream`, with a dedicated test and scoped re-review.

### 15.4 What remains blocked — and why

The foundation branch has contracts, parsers, gates, cohort selection, and manifest serialization. It does **not** have a valid live recovery dataset or a training-ready Stage-1 corpus.

| Blocker | Reason | Required next action |
|---|---|---|
| Official Gazette source bytes and section extraction | `gazette_manifest_v1.jsonl` intentionally holds identities/provenance only; no real Act bodies/digests were fetched or attached to an isolated candidate corpus | Acquire named official BNS/BNSS/BSA Gazette PDF bytes, record retrieval/digest, parse sections, and validate real provision attachment |
| BSA commencement evidence | Unverified host-level placeholder was removed; claiming in-force status without an Extra document would be false | Verify the exact official Gazette Extra URL/ID before adding it |
| Control candidate export | Task 4 accepts caller-supplied rows only; reading `data/build/state/law_v1.sqlite3` was deliberately forbidden | Authorize a read-only, fingerprinted control-candidate export into an isolated workflow |
| Isolated recovery workdir/config | No safe committed `exp_recovery` paths/config contract exists in this clean branch; the dirty worktree implementation was not copied | Define and test a distinct isolated workdir/config after the recovery changes are saved/committed or otherwise made available |
| Live cohort freeze | Task 5 produces `in_memory_selection_record`, not `frozen_control_cohort` | Combine verified source data, allowed control export, and isolated storage; then write the real manifest |
| Provider generation/judging | Cohort/provenance gates are incomplete; provider calls would be premature | Re-authorize spend and run only after the real frozen manifest validates |
| Qwen3-8B Stage 1 corpus | Existing inventory is not an 8–10k law-majority, provenance-complete corpus | Assemble only after real rows pass custody, license, citation, case-ID, decontamination, and mixture gates |

### 15.5 Explicit non-actions

- No `generate`, `judge`, `calibrate`, `--reopen`, `--fit`, or provider call was run.
- No `data/build/state/law_v1.sqlite3`, `data/build/gold`, `data/build/exp_*`, or live YAML/prompt file was modified.
- No existing Fable-labelled `gold_label` row was changed. The new fence blocks future model-reference ingestion but does not claim to repair unknown historical DB provenance.
- No transition cell was presented as a verbatim statute provision; the committed transition resource remains mapping/effect data, not a bare-Act corpus.
- No recovery manifest with actual control IDs was written. The serializer tests use inline synthetic candidates and temporary absolute paths only.
- No branch commit, push, merge, or training run occurred.

