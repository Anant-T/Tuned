# law_v1 synthesis error reduction

**Date:** 21 Aug 2026  
**Workspace:** `C:\Users\Anant\Desktop\projects\tuned`  
**Related:** `2026-08-22-data-pipeline-investigation.md` (diagnosis + probes) · canvas `tuned-status.canvas.tsx`

> **Correction (23 Aug 2026).** Every `$0 spent` figure below was true at this
> note's 16:08 IST snapshot and went stale three minutes later. Between
> **16:11 and 18:44 IST on 21 Aug** the `exp_harmony` store spent **≈ $0.34**
> of the $2 cap on `openai/gpt-5-mini` (124 requests, 377,537 prompt /
> 122,607 completion tokens), and roughly **78% of it bought nothing** — 95 of
> 96 `judge_parse_error` events are empty replies and only 27 usable
> judgements came back. Read the cause and the fix before spending more:
> `2026-08-22-data-pipeline-investigation.md` § "OpenAI $2 hard cap". The live
> store's ledger does still show genuinely 0 OpenAI requests.

Working note for Cursor. Live synthesis sqlite is **no longer frozen** (judges are draining). Harmony work stays in `data/build/exp_harmony/`.

## Current snapshot (21 Aug, ~16:08 IST)

| Store | Accept | Judging | Reject | Notes |
|---|---:|---:|---:|---|
| Live Chat Completions | **15** | 76 + 1 active | 414 | 1,396 gens. OpenAI $0. |
| Harmony+s1 `exp_harmony` | **3** | 15 | 18 | 36 tasks, 75 gens, all `gpt-oss-120b`. |

- First Harmony+s1 12-task probe: token-1 12/12, format 11/12, **3 accept / 9 reject**.
- Expanded wave: format 27/36 (75%). Validity still kills (mean 3.17, need ≥ 4).
- OpenAI judging: **$2 hard cap** (mini+nano one wallet). Exp yaml family `gpt-5` so it can grade gpt-oss rows. Live yaml still lumps OpenAI as `gpt-oss`. **$0 spent.** *(True at 16:08 IST only — $0.34 by 18:44; see the correction above.)*
- Generation: free Cerebras `gpt-oss-120b` first. No OpenAI/Gemini teachers (ToS).
- Tests: 459 passed (`test_build_{generate,judge,providers,harmony,config}`).

Do **not** promote Harmony onto live yaml.

---

---

## 1. What is going on

The training lane on `main` is fine. The stall is **teacher synthesis** on branch `worktree-law-v1-data-pipeline`.

Product: chat rows for Qwen3-8B at packed 8192, with a reasoning trace the student will imitate.

Two kill layers, stacked:

1. **Format gates** (post-hoc grep of the hidden think channel).
2. **Judges** (min-axis ≥ 4, uncalibrated; `judge_threshold` is empty).

Replay 4,320 and curated C1 1,100 already exist. They never enter this ritual. The dying stream is synthesis (target ~60% of the mix).

---

## 2. The 6 live accepts vs the hybrid 25

The 6 are **`judge:accept`**. The 25 are **format-pass only** (state `judging`, 0 judgement rows). They are not the same product.

### Fair format yield (the report mixed denominators)

`2026-08-22-data-pipeline-investigation.md` §12 scored the probe as *latest gen per task* (25/50 = 50%) against a live baseline of *all generations* (165/1,386 = 11.9%). Aligned:

| Metric | Live control | Hybrid probe (`exp_hybrid`) |
|---|---|---|
| All generations | 11.9% (165/1,386) | 21.9% (25/114) |
| Latest gen per task | **27%** (139/515) | **50%** (25/50) |

Live latest format-passers: **109 stuck in judging**, 24 `judge:reject`, **6 accept**. Among format-passers that already have a judge decision, accept = 6/(6+24) = **20%**. If the 25 meet that bench, expect ~5 accepted rows, not 25 training examples.

### Think dialect: the 25 are clones of the 6

| Signal | Live 6 | Hybrid 25 |
|---|---|---|
| Think opening | 5/6 “We need to produce…” | 24/25 “We need to produce…” |
| Few-shot dialect copied (`I start from` / `Let me check` / `am I sure`) | 0 | **0** |
| Cues that actually fire | `actually`, `double-check`, `unsure` | same, plus `to confirm` / `re-derive` / `re-examin` |
| Mean think tokens | 682 (min 562) | 613 (**8 of 25 are 401–485**) |
| Teacher | cerebras `gpt-oss-120b` | 23 cerebras + 2 lightning, T=0.7 |
| Judges | gemma tiebreak 5/5/5 | **none** |

`think_min` 500→400 is load-bearing for the 20/50 promote bar: without those 8 short traces, the probe is **17/50**, below 20/50, still above live 27% latest-task.

Statute QA looking “new” in the 25 is a judge-queue fact. Live already has **43** statute_qa latest format-passers sitting in `judging`.

Side-by-side canvas: [six vs hybrid 25](C:\Users\Anant\.cursor\projects\c-Users-Anant-Desktop-projects-tuned\canvases\law-v1-six-vs-hybrid25.canvas.tsx).

---

## 3. Why prior-turn few-shot could not fix this

Wired correctly in `src/tuned/data/fewshot.py`: `[system, few-shot user, few-shot assistant-with-<think>, live user]`.

It still failed because of the **teacher**, not the file layout:

1. **gpt-oss CoT is unsupervised.** Output follows system instructions; the analysis channel often does not ([OpenAI gpt-oss](https://openai.com/index/introducing-gpt-oss/), [Harmony CoT cookbook](https://developers.openai.com/cookbook/articles/gpt-oss/handle-raw-cot)).
2. **Echo of Prompt.** Hidden CoT often *starts* by copying the instruction packet. Hao et al. 2026 measure EOP ~0.86 on gpt-oss ([arxiv 2602.06600](https://arxiv.org/pdf/2602.06600)). A long “450–700 words, first person, Issue/Rule/…” block is more echoable than the case facts → “We need to produce a reasoning in first person…”.
3. **Harmony drops prior-turn analysis after `final`.** A completed few-shot think is disposable scratch, not a style template. 0/25 “I start from” is the spec working.
4. **Gates grep cues, not dialect.** `self_verification` is a substring allow-list. No gate rejects “We need to produce…”. A later `actually` / `double-check` saves the row — that is how the 6 and the 25 survived.
5. **Cerebras JSON schema constrains `final`, not think**, and `raw` reasoning is incompatible with json_schema ([Cerebras structured outputs](https://inference-docs.cerebras.ai/capabilities/structured-outputs)). Do not schema-lock the dialect probe.

Anti-echo sentences (“do not restate these instructions”) become the next echo. XML `<think>` on gpt-oss fights Harmony. Loosening `citations` / `statutory_grounding` / `temporal` is off the table.

---

## 4. How to cut remaining errors (ranked)

Keep law gates. Change **opening control** of the analysis channel.

| Rank | Lever | Why | Status |
|---|---|---|---|
| 1 | **Shrink the dialect packet.** Case first. Delete “450–700 words of deliberation” and the cue checklist. | Removes the text EOP copies. | **In `prompts_dialect/`** |
| 2 | **Same-turn think excerpt in the live user message** (not a prior assistant turn). | Stays in this turn’s echo window. Harmony will not drop it. | **In `prompts_dialect/`** |
| 3 | **Harmony Completions analysis prefill** (not Chat Completions `assistant.reasoning`). | Only hosted way to lock token-1 of analysis on Cerebras. Chat Completions ignored the reasoning field. | **Wired. Token-1 12/12 on Harmony+s1.** |
| 4 | **s1 continue** `" Wait"` when the continuation has no cue. | Salvage. Must not be `Wait,` / `am I sure` (those are `VERIFICATION_CUES`). | **In exp yaml. First wave 13 continue events, 0 errors.** |
| 5 | Drain judging and fit `judge_threshold`. | Format is no longer the only bottleneck. ~20–28% accept among judged format-passers. | **Live 76 left; Harmony 15 left. OpenAI $2 cap, $0 spent.** *(superseded — $0.34 spent by 18:44 IST, see correction)* |
| 6 | JSON schema on **visible IRAC / citations only**, `reasoning_format=parsed`. | None on think dialect; can hide reasoning if mis-set. | After dialect yield is real |
| 7 | Self-host vLLM + Harmony analysis prefill, or a teacher with think-prefill (Qwen3). | Highest control. Days. | Fallback |

Do **not** retry prior-turn few-shot. Do **not** revive `EFFORT_LADDER_RETIRED`. Do **not** drop temperature to greedy.

---

## 5. Dialect probe (this session)

Worktree config (live yaml / SHAs / store untouched):

`configs/data_law_v1_exp_dialect.yaml`

| Knob | Hybrid (done) | Dialect (this) |
|---|---|---|
| workdir | `data/build/exp_hybrid` | `data/build/exp_dialect` |
| fewshot prior turn | on | **off** |
| prompt overlay | `prompts_exp` (drafting IRAC skip only) | **`prompts_dialect`** (packet shrink + same-turn excerpt, all 12 synthesis templates) |
| `reasoning_prefill` | off | `"I start from the facts. Let me check "` |
| `think_min` | 400 | 400 |
| drafting skip IRAC answer | on | on |

Primary metric is **token 1 of `generation.think`**, not “a cue appeared later”:

- **Win:** think starts `I start from` / `Let me check` / restating **case facts**.
- **Loss:** think starts `We need to produce` / `The user wants` / `450-700`.
- Law gates (`citations`, `statutory_grounding`, `temporal`) must not regress vs hybrid.
- `self_verification` still required; do not locally prepend the prefill onto stored think (that would fake the gate).

N = 12 new synthesis tasks, arm `exp_dialect`, mix 4 irac / 3 statute_qa / 3 drafting / 2 summarization. Teacher Cerebras `gpt-oss-120b`, T=0.7. Kill prefill after 2 traces if the API 400s or the prefill appears only in visible content.

### Commands (from the worktree, never `--config configs/data_law_v1.yaml`)

```powershell
cd .claude\worktrees\law-v1-data-pipeline
# store bootstrap + plan 12  (see scripts in data/build/exp_dialect/)
.venv\Scripts\python.exe -m tuned.data.generate --config configs/data_law_v1_exp_dialect.yaml --stream synthesis --n-workers 1 --max-batches 40 --arm exp_dialect
.venv\Scripts\python.exe data/build/exp_dialect/score_probe.py
```

Pytest for the wiring: `tests/test_build_dialect.py` (19 passed with `test_build_fewshot.py`; live SHAs unchanged).

---

## 6. Probe results (Cerebras `gpt-oss-120b` only)

Ran 21 Aug 2026 from the worktree with `CEREBRAS_API_KEY`. Generator list pinned to `[cerebras/gpt-oss-120b]` — no Lightning, no OpenAI as teacher. `--arm exp_dialect` so the copied live pending queue was not claimed.

| | Hybrid 21 Aug | Dialect 21 Aug |
|---|---|---|
| Teacher | cerebras/lightning gpt-oss-120b | **cerebras/gpt-oss-120b only** |
| n (latest / task) | 50 | 12 |
| All-gates-pass | **25 / 50 (50%)** | **3 / 12 (25%)** |
| Think starts `I start from` / `Let me check` | 0 | **0** |
| Think starts `We need to produce` / `The user wants` | 24/25 | **12 / 12** |
| Tokens billed | (prior run) | 108,317 (~$0.05 on Cerebras) |
| Permanent reject (`citations`) | 3 | 1 |
| `format_parked` | 22 | 8 |

The 3 format-passes (2 drafting, 1 statute_qa) are the same species as the live 6: restated opening, a cue later, gates grepped it. **Prefill did not land in `generation.think`.** Cerebras accepted the request (no 400s, 0 errors) and still opened the analysis channel with instruction echo. Same-turn few-shot did not transfer either.

Fail gates on the 9 non-passes: `self_verification` 9, `length_band` 6, `banned_meta` 3, plus 1 `citations` / 1 `statutory_grounding` / 1 `irac_placement`.

**Decision:** do not promote the Chat Completions dialect overlay. Packet-shrink + user-turn example + `assistant.reasoning` prefill does not bind gpt-oss's hidden think channel on Cerebras Chat Completions. Harmony Completions (below) does bind the channel.

Live store at that probe: accepted=6, judging=109, rejected=392, stale_prompt=419. **Superseded** — see the current snapshot at the top of this file and `2026-08-22-data-pipeline-investigation.md` §14.

---

## 7. What this is not

- Not a bad-seed problem (60,603 seeds).
- Not “the model didn’t think” (96% of live gens have a think block).
- Not citation hallucination (permanent law gates are almost silent).
- Not “say let me think” as a fail — that phrase is a **pass** cue. Failures are missing those phrases **and** opening on the instruction packet.
- Not a promotion onto the live wave. Promote only after dialect token-1 wins **and** judges drain without validity collapse.

---

## 8. Pointers

- Live store: `data/build/state/law_v1.sqlite3` (worktree; **judges writing 21 Aug**)
- Harmony store: `data/build/exp_harmony/state/law_v1.sqlite3`
- Hybrid store: `data/build/exp_hybrid/state/law_v1.sqlite3`
- Dialect store: `data/build/exp_dialect/state/law_v1.sqlite3`
- Gates: `src/tuned/data/gates.py` (`VERIFICATION_CUES`, `BANNED_META`)
- Generate path: `src/tuned/data/generate.py` `build_prompt` → Harmony Completions when `build.harmony_completions`
- Harmony renderer: `src/tuned/data/harmony.py` (`python -m tuned.data.harmony`)
- OpenAI cap: `budget_ok_for` in `generate.py` (`usd_cap` / `usd_per_1m_*` on openai models)
- Keys: worktree `.env`. Never commit.

---

## 9. Harmony Completions probe (21 Aug 2026)

Cerebras `/v1/completions` with a Harmony prompt that ends mid analysis-message. Teacher `gpt-oss-120b`, `CEREBRAS_API_KEY`. Live sqlite not written. Tests: `tests/test_build_harmony.py` (9 passed).

| Call | Packet | Effort | max_tokens | Token-1 | Analysis | Final |
|---|---|---|---|---|---|---|
| 1 | 450–700 in | medium | 80 | **LOSS** — continuation was `450-700 words deliberation. Then produce final with headings.` | 61 chars, then channel switch | legal prose starting “I begin by laying out the facts” (wrong channel) |
| 2 | **stripped** | medium | 80 | **WIN** — `302 IPC elements: causing death…` | stayed in analysis | none yet (hit length) |
| 3 | stripped + grounded prefill | medium | 80 | **WIN** — `1) causation, 2) intention…` | stayed in analysis | none yet |
| 4 | stripped | medium | 2048 | **WIN** | ~157 tok, ends `Now produce headings.` | IRAC (`**Issue**` …), ~750 words, invented medical facts (smoke had no grounding) |
| 5 | stripped | **high** | 2048 | **WIN** — case facts | 2048 tok, never left analysis | empty (`finish=length`) |

What this means:

1. Completions honors Harmony special tokens. Chat Completions `assistant.reasoning` did not.
2. The 450–700 sentence is still fatal on this wire: analysis becomes a 61-character instruction echo and the real think lands in `final`. Every live `gen_*.md` still contains that sentence.
3. With the packet gone, token-1 is case law, not “We need to produce…”. That is the dialect win Chat Completions never got.
4. Medium effort analysis is too short for `think_min` 400 (~157 tokens). High effort fills 2048 tokens of analysis and never emits final. Isolated yaml uses `think_min` 200.
5. Prefill must not include a verification cue, or `self_verification` is faked. Cue-free prefill is `"I start from the facts. "`.
6. Do not promote onto live yaml.

### 12-row on real seeds (same day)

Isolated store `data/build/exp_harmony/state/law_v1.sqlite3`. Overlay `src/tuned/data/prompts_harmony/` (packet stripped; live SHAs unchanged). Config `configs/data_law_v1_exp_harmony.yaml`. Prefill cue-free. Generator cerebras/gpt-oss-120b only. Live sqlite not written.

12 synthesis tasks planned (5 irac / 3 statute_qa / 2 drafting / 2 summarization). 9 produced at least one generation; 3 parked `gen_unroutable` (2 cooling, 1 provider-fault). 23 paid generations. Cerebras usage on that db showed ~75.6k tokens.

| Metric | Hybrid 50 | Dialect 12 (Chat Completions) | Harmony 12 (Completions) |
|---|---|---|---|
| Think starts `I start from the facts` | 0 | 0 | **9 / 9 latest** |
| Think starts `We need to produce` | 24/25 | 12/12 | **0 / 9** |
| Latest all-gates-pass | 25/50 (50%) | 3/12 (25%) | **3 / 9 (33%)**; 3/12 planned |
| `self_verification` on latest fails | (cues later) | 9/12 | **6 / 9** (honest: no cue in prefill) |

The 3 format-passes (1 drafting, 1 irac, 1 statute_qa) are in `judging`. Fail gates on the 6 latest fails: `self_verification` 6, plus 1 `irac_placement`, 1 `banned_meta`, 1 `length_band`. Continuations are case facts (`The client is…`, `Need to decide issue…`), not the instruction packet. One continuation still opens `The user only gave sections…`.

**Decision:** Harmony Completions + packet-stripped overlay is the first hosted lever that binds token-1. Do not promote yet: `self_verification` is now the real format killer, 3 rows never generated, and judges have not scored the 3 format-passers. Next lever on this wire is a second-call s1 continue (` Wait, am I sure?`) only on traces whose continuation has no cue — not putting a cue back in the prefill.

```powershell
cd .claude\worktrees\law-v1-data-pipeline
.venv\Scripts\python.exe data\build\exp_harmony\bootstrap.py   # wipes the exp db
.venv\Scripts\python.exe -m tuned.data.generate --config configs/data_law_v1_exp_harmony.yaml --stream synthesis --n-workers 1 --max-batches 50
.venv\Scripts\python.exe data\build\exp_harmony\score_probe.py
```
