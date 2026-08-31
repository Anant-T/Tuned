# agent_0

Operating file for the unattended law_v1 data-build session. **Read this first
after every autocompact, at the start of every session, when starting a new
task, and after any subagent returns.** It exists because a compact wiped the
working state once already.

---

## 0. State of play (2026-08-31, ~06:25Z)

**The pipeline works end to end. What is left is throughput and one prompt
decision - no breakage anywhere.**

- **Build is running and healthy.** Run #9 (`33349462956`, started 02:03Z),
  11 batches, **1 error, zero cooling parks, zero unroutable**. Measured:
  271 generation calls/hr, **~44 accepted rows/hr** (~23 of them synthesis).
- **The assembly chain went GREEN for the first time** - `CHAIN RC=0`, all nine
  gates passing at `v1.0-MVP`. The historically-RED gates were never
  miscalibrated; the old runs just never ran `shape` (F16).
- **`transition` is finished** - 2,200 of 2,200 tasks terminal, 5 accepted. It
  converted at 4.3% while holding a third of an account-level rate bucket, so
  its ending is a gain, not a loss (F22).
- **Fixed tonight:** the claim loop treated the fleet's in-flight budget as a
  per-stream quota, so `transition` ending silently cut concurrency 36 -> 24.
  Restored and made invariant to how many streams have work: +17% now, +68%
  once `curated_c2` drains (F23). Suite **3,755 / 19 skipped**.
- **Draining the whole pending queue** costs ~20,356 generation calls and
  yields ~2,994 more accepted rows: **~2.4 days** with the fix, ~3.1 without.
  That lands synthesis at **~2,360** accepted against the ~3,207 MVP wants -
  so Step 6 needs roughly **1,500-1,600** more synthesis tasks, not the ~5,000
  F21 estimated off a pooled accept rate (per-variant rates are much higher).
- **There is a hard ceiling at ~10,021 rows**, binding on the replay/nothink
  pool. Both profiles cap there but `v1.1-full` costs ~2x the synthesis to
  reach it - **on current pools MVP dominates** (F19, F21).

**Do not, without deciding first:**
- dispatch `data-assemble` (it would fail at `shape` and burn a runner - F20);
- edit any prompt template in place (parks the live queue as `stale_prompt`;
  ADD a new `prompt_id` instead - F15);
- delete a template file (678 pending tasks are stamped with `v4` - F24);
- reopen the 2,063 `skip:slots` transition rows (they would re-die - F18);
- widen `curated_c2` (it raises the bar it is already above - F16).

**Shipped since 03:40Z** (all on `origin/main`, suite **3,796 / 19**):
- `c61311a` **the variant allowlist** - `--variant` pins a wave to chosen
  templates, binding new rows only, so Step 6 can stop buying the expensive
  personas without deleting a template and parking the live queue (F27).
- `fb611f5` **the answer's second deliberation is no longer shipped** - 490 of
  733 IRAC-contract rows opened with 2,400-4,800 chars of first-person
  reasoning before the first heading, which no gate can see (`answer_max` does
  not exist). Trimmed at assembly for zero generations: median answer **801 ->
  379 words**, rows over the templates' own 450-word spec **94% -> 33%** (F28).
- **The 50-example review packet** the dataset card requires before shipping is
  built and pre-screened; 12 of 50 rows cite an authority their source never
  names (F29). The human read itself is still outstanding.
- `4505840` **the review packet is reproducible** - `data/scripts/review_packet.py`
  plus `src/tuned/data/review.py` behind 15 tests, read-only against any store
  copy, deterministic in `--salt` so a re-render after the corpus grows returns
  the same 50 rows (F29).
- `0227bdd` **the length band now weighs the answer that ships** - F28 had left
  the gate and the assembler disagreeing, and 110 rows (13.7% of the accepted
  corpus) were refused on characters the corpus would never hold (F32).
- The cron was **measured, not changed**: fires are late, never lost (F26).

**The card claimed a citation check that has run on zero rows** - 1,875 of
1,875 accepted rows carry `novel_skipped: no-index`, so the existence half has
never run, and the card listed it among the gates a row ships on. Corrected
tonight; the gate itself cannot be armed without a citation index, whose
coverage is 4.7% (F34).

**One legal error is in the corpus.** Accepted row `412b8d1c5430` puts an appeal
under the BNSS where its own answer key says the CrPC continues to govern - it
cites all four required sections, so a permanent gate passed it. `check_answer_key`
checks citations, not conclusions, and `families_by_kind` is never read (F30).
1 row of 718; the blind spot behind it is systematic (19.9% of gate-passing
transition generations). Not fixed unattended - `transition` is a finished
stream, so the gate would guard ~0 future rows.

**Next run is `33363831595`** (dispatched 06:20Z on `0227bdd`, pending behind
the 02:03Z run which ends ~07:20Z). It is the first run carrying ALL of
tonight's fixes.

**What the next session should pick up**, in order: read the F23 prediction
(`claimed=36`) off run `33363831595` once it starts - it is the first run on
post-fix code; then the persona decision below; then Step 6, which `--variant`
now unblocks. Do not dispatch `data-assemble` (F20).

**The one decision waiting:** two prompt personas are burning half the fleet.
`gen_irac_analysis_v4` (examiner writing a model answer) spends **15.6
generations per accepted row** against `v1`'s 3.3; `gen_summarization_v1` (law
reporter settling a headnote) fails `irac_placement` at 27% where `v2`
(advocate's letter to a client) fails at 4% - **on the identical instruction
sentence**, randomised seeds, z = 5.7. Genre beats instruction. Adding two
variants in the proven speech/letter genre is worth 33% fewer calls and 72%
more rows on the same tasks (F24). It needs a 10-row A/B first, which cannot
run while CI holds the account-level rate bucket.

---

## 1. Agent prompt (verbatim, so the role is re-assumable)

> you are now agent_0 you have to reach the goal of getting everything done and
> ready, use can use cheaper subagents for parallel tasks and and you can also
> use deepseek with websearch on/off for tasks. fix things, always recheck your
> reasoning and outputs. create a agent_0.md in tuned that keep the
> tasks, summary, agent propmt and the autocompact is set to 200k. take all the
> decisions yourself for all the task and choose the best and always preffer
> free deepseek or cheap subagents for easier task that you geniunely think are
> easy and can be done by cheaper models to save on usage without degrading any
> quality. i am going to sleep with charger on and no sleep time so autoaccept
> all the best decision and also do small tests of 5-10 example always before
> doing anything lengthy and use only free judges and generator.

Follow-ups:

> use superpowers after every autocompact or a cheaper subagent task or startig
> a new task altogether and auto load agent_0.md that i asked you to mke after
> every auto compact or session and always update task list to latest tasks.also alwasy reason things and blockers to resolve them to get max efficiencey and best quality fixes.



Autocompact was raised **200k -> 260k** on 2026-08-31.

## 2. Session protocol

1. **Read this file** before anything else.
2. **Invoke the relevant superpowers skill** and announce it: `systematic-debugging`
   for anything broken, `brainstorming` before planning, `test-driven-development`
   for new behaviour, `verification-before-completion` before claiming done.
3. **Re-verify subagent claims** before acting. Five hypotheses have already been
   killed by testing rather than trusting (section 5).
4. **Interrogate every constant before relying on it.** The operator had to ask
   "why 300 s?" and the honest answer was "nobody chose it." Ask that question
   unprompted of every number and default in the path being changed, and record
   the answer in section 6.
5. Prefer free/cheap models for genuinely easy subagent work; **and free judges and generator
   only**, no paid refs ever.
6. Test on 5-10 examples before anything lengthy.

## 3. Hard constraints - do not violate

- **No AI attribution or watermarks anywhere, ever** (global rule). Not in
  commits, PRs, docs, or any artifact.
- **Never run a writer against `data/build` while the cron holds the baton.**
  Read via `sqlite3 "file:...?mode=ro"` / `uri=True`, or work on a scratch copy.
- **Exactly ONE generating host may exist** - the generator's rate bucket is
  account-level. Cancel the remote worker before generating locally.check and verify first.
- **Never `seed-push` against a remote that owns the baton.**
- `.env` is gitignored and holds provider keys. **No step may echo the
  environment** - CI logs on a public repo are public.
- Free fleet only: bai + groq+mistral (+ cerebras, which answers 402). No OpenRouter, no
  paid refs.

## 4. Task list (live)

| # | Task | Status |
|---|---|---|
| 0 | `agent_0.md` at repo root | **DONE** 2026-08-31 |
| - | Cancel run #8 (pre-guard code, claiming nothing) | **DONE** - cancel accepted 01:38Z |
| 1 | Stop the shredder: wire `TRANSIENT_SKIPS` into `generate.py` + refund attempt + back off | **DONE** `3a71494` |
| 1b | Make `cooldown_s`/`breaker_threshold` configurable; set `routing.cooldown_s: 60` | **DONE** `bc244e1` |
| 2 | Live 5-10 row test on a scratch DB copy; then force a breaker trip | **DONE** - both halves green |
| 3 | Push, dispatch `data-worker`, watch the armed reopen recover ~5,190 rows | **DONE** - run #9 generating |
| 4 | Measure the breaker trip rate; report (do not act) | **DONE** - 0 trips in 180 gens; 4.72 gen/min, ~23.6 accepted/hr (F17) |
| 5 | Root-cause the `transition` stream's 99% reject rate | **DONE** - 96% was the planner bug (`708d455`); the rest is a 95.7% `statutory_quotation` compliance gap (F18) |
| 6 | Widen the queue, sized to measured yield | **READY TO FIRE, deliberately not dispatched** - command and allowlist decided (F33) - synthesis-only, **~1,500-1,600 tasks** (re-sized on per-variant rates, F24); blocked on the variant decision, not on capacity |
| 8 | Fleet claim budget: restore the designed 36 in flight and make it invariant to drained streams | **DONE** - 3 tests, suite 3,755/19 (F23) |
| 9 | Author `gen_irac_analysis_v5` + `gen_summarization_v3` in the proven speech/letter genre; 10-row A/B; then plan Step 6 on the winners | **DROPPED tonight, and it is not a loss** - the A/B needs a free rate bucket, and the fleet never has one (F33). `--variant` takes the same prize by pinning the personas that already won |
| 10 | Variant allowlist so a wave can be planned on the templates that earned it | **DONE** `c61311a` (F27); reachable from the operator path only since F33 |
| 15 | Carry the allowlist through `--phase plan` and `data-plan.yml` | **DONE** - 3 tests (F33) |
| 11 | Stop shipping the answer's second deliberation | **DONE** `fb611f5` (F28) |
| 12 | Build the 50-example review packet the card requires, and pre-screen it | **DONE** `4505840` - reproducible CLI, 4/50 flagged (the first 12/50 was a screen bug, F29) |
| 13 | The legal read of those 50 examples | **OPEN - human task.** The packet only prepares it. The 5 transition rows are already read: 1 is wrong (F30). **Re-rendered 2026-08-31 11:2xZ against the SWEPT store** (935 accepted, post-`verify` so none of the 84 off-teacher rows can waste a reader's attention): 21 curated_c2/irac, 18 synthesis/irac, 5 summarization, 3 drafting, 3 transition; 3 of 50 cite an authority the source never names - read those first. `data/build/out/review_packet.html`, gitignored |
| 16 | Stop the card claiming a citation check that never ran | **DONE** (F34) |
| 17 | Measure the generated-curated ceiling; ship `shape --headroom` | **DONE** (F35) |
| 18 | Decide the ceiling remedy | **DONE** - hand throttle shipped, then REPLACED same day by a measured guard (`be25afd`): `STREAMS` lists all three, `served_streams` drops curated_c2 within 150 effective of the ceiling and on any ceiling it cannot measure (F35) |
| 19 | Re-open curated_c2 when synthesis nears ~1,100 accepted | **CLOSED - obsolete.** The re-open was the hand throttle's expiry; the guard now decides per run, so there is no date to remember. Live: serving, 401 effective against a 2,050 ceiling, 1,499 of headroom (F38) |
| 20 | Plan the next wave on `gen_irac_analysis_v1,gen_irac_analysis_v3,gen_summarization_v2` | **OPEN, SIZED at ~780 synthesis tasks on v1+v3** (~1,279 at the pooled yield). Without it the drained queue lands 418 effective rows BELOW the band and the corpus cannot be assembled at all (F37, F41). **Do NOT dispatch yet:** the claim is FIFO by rowid, so a wave planned now is worked LAST and the dispatch costs up to ~4 h of idle fleet. Trigger is queue depth (< ~1,000 pending synthesis), not the clock (F43). **Command rehearsed and corrected (F45): `--n` counts the LIVE arm-NULL queue, so it is `<live> + 780` - the stream total over-plans 2.3x**. ETA now MEASURED, not guessed: the queue drains ~863 pending/run at ~315.6 min/cycle, so 4,072 pending is ~25 h and the whole queue is dry around 2026-09-01 14:00Z. The <~1,000-synthesis trigger arrives sooner - roughly two runs out, ~2026-09-01 00:00Z - but the per-stream split has to be read on the baton at dispatch time, not extrapolated (F50) |
| 21 | Verify the ceiling guard's first live run | **OPEN** - `33375922778` was EVICTED, not verified: it was stamped at `dc2182d`, which predates the guard and carries the REJECTED hand throttle. Watch the replacement instead; expect `ceiling guard: serving every stream - ~400 effective ... ceiling of 2050`, then curated_c2 claims - which DO resume despite every curated row sitting behind 3,162 synthesis rows by rowid, because claiming is per-stream (F38, F39, F43). Pre-registered on the swept working copy: `ceiling guard: serving every stream - 398 effective generated-curated rows against a ceiling of 2050` (the live figure will be a little higher; 2050 is a function of the static replay/curated_c1 pools and should match exactly). **The guard cannot bind on this queue** - draining every pending curated_c2 task lands ~1,241 effective against a throttle point of 1,900, so what is being verified is that it MEASURES and serves, not that it throttles. Log route CLOSED until the run completes: GitHub serves no log blob for an in_progress job (404) and the step summary renders only at step end, so the read lands when the run's log is archived. Its JOB started 12:34:43Z (there is no runner lag - the earlier 58 min was the fence, see F50's correction), so the 315-minute window runs to ~17:49Z and the archived log lands ~18:00Z (F49, F50. DEAD END, do not re-attempt: the baton cannot answer this early. curated_c2 shares synthesis's `gen_*` templates (`CURATED_C2_MIX` is summarization+irac_analysis, there is no curated prompt file) and the raw gen record has no stream or task_id field, so `prompt_id` cannot tell the two streams apart - a tail read of raw/gen showing 87 synthesis-shaped prompts proves nothing either way). Robust either way now: the 12:17Z cron is stamped at `699af2c`, which carries `_run_log`, so if it ever displaces this run the replacement puts the same line on the baton within 15 minutes instead |
| 22 | Filter IL-TUR-contaminated seeds at PLAN time | **CLOSED - wrong fix.** The drops are co-citation, not contamination: 0 of 88 match the eval item's own case. A seed filter would implement the over-firing and cost 9.6% of the pool for no integrity gain (F40) |
| 27 | **OPERATOR ACTION, DEADLINE ~2026-09-02: squash the baton's git history** | **OPEN, hard stop.** 55.00 GB of a 100 GB ACCOUNT-WIDE free private quota is already used (baton 38.83 + 16.17 in ten checkpoint repos) and it grows 14.8 GB/day - every checkpoint is a fresh ~565 MB LFS blob; wall ~2026-09-03, and a squash needs up to 36 h to reflect. `super_squash_history` takes it back to ~1 GB, is irreversible, and must run when NO worker holds the baton - end-of-job after the final push, or cancel-squash-dispatch by hand. Recurs ~weekly (F48). **EXACT FIGURES 2026-08-31 13:25Z** from HF `usedStorage` (not derived): baton 42.07 GB, seven checkpoint repos 16.16 GB, **total 58.23 GB of 100** - ~3.2 GB above what F48 recorded this morning, so the 14.8 GB/day on file is probably LOW and the wall probably EARLIER than 09-03. Rate now being measured over a timed interval rather than derived. **A squash is not the only lever**: six of the seven checkpoint repos are retired lanes holding 6.84 GB - `-rslora` 0.54 + `-rslora32` 1.59 + `-alpha64` 1.06 (arms retired in 15c3eb9 / 2b3ac29 / c3c3651, both questions recorded FULLY CLOSED) and `qwen-ckpt` 0.78 + `-ckpt-ddp` 1.30 + `-ckpt-manual` 1.57 (pre-8B lanes, superseded when the repo was stripped to Qwen3-8B DDP on 2026-08-08). Only `qwen8b-ckpt-ddp` (8.68 GiB) is live. That is ~half a squash's worth of headroom for a delete that needs no quiet window - but it is irreversible and may be the only copy of those weights, so it is an OPERATOR call, listed not recommended. **RATE NOW TIMED, AND IT IS 2.8x THE FIGURE ON FILE: 58.24 GB at 13:24:19Z -> 59.97 GB at 14:24:23Z = +1.73 GB in 1.00 h = 41.5 GB/day.** Headroom 40.0 GB, so the wall is **~2026-09-01 14:20Z, about 24 h out** - and a squash needs up to 36 h to reflect, so IT IS ALREADY LATE. Mechanism: the DB is NOT the main cost. `raw/gen/<day>/gen.ndjson` is append-only and ~250 MB by mid-day, and the 900 s cadence re-uploads it WHOLE as a fresh LFS blob 4x/hour = ~1.0 GB/h against the hourly DB's 0.565 GB/h. It grows all day, so the rate ACCELERATES until the UTC date rolls the file over (F52) |
| 28 | Decide whether 15 dual-judged rows/run is enough quality evidence | **OPEN, OPERATOR DECISION** - the audit sample is sized right (5.09%) but delivers 15 judged rows and 0 rejections per run, with 25% of the sample shipping unjudged. Cause is family exhaustion, not budget: after groq's two models hit their daily cap there is only ONE free family left for a deepseek row and a dual judgement needs two. mistral's idle 5,000k is NOT the fix - it is the reserved tiebreak family (F51) |
| 26 | OPERATOR DECISION: retire the pending v2/v4 irac tasks and replan them on v1/v3 | **OPEN, sized at ~182 accepted rows (~37% of the shortfall) for zero extra fleet time.** Needs a cancel/park command that does not exist yet, plus a park state that frees queue capacity honestly. F37's "no safe window" objection is superseded - `--phase plan` shows the pattern (F46) |
| 25 | OPERATOR DECISION: turn off the row-side case_id channel (`--no-case-id-from-text`) | **OPEN** - recovers 81 generated rows (~9%) with exact containment untouched; deferred because it is an eval-integrity call and the rows are recoverable retroactively, so waiting costs nothing (F40) |
| 23 | Re-fit `synthesis` retention after the teacher purge lands, or teach `generated_counts` to skip rows the cut will take | **CLOSED - neither is needed.** The shipping chain runs `verify` immediately before `shape` over one DB and verify writes the demotion back, so production sizing already reads a post-demotion store. Only an ad-hoc `--headroom` run outside the chain sees the inflated count, and the guard reads the curated bucket, which has no teacher cut (F44) |
| 24 | Citation-existence half | **CLOSED** - index exists, is on the baton, is armed live, costs 8 rows of 943, and all 8 were already dropped by the chain (F39) |
| 14 | Enforce `families_by_kind` per limb in `check_answer_key` | **OPEN, deliberately not done unattended** - only worth it if `transition` is replanned (F30) |
| 7 | Prove the assembly chain end to end | **DONE** - `CHAIN RC=0`, stats **GREEN** at v1.0-MVP (F16) |

**The one open blocker:** `curated_c2` is over-generated against synthesis and
generated rows cannot be dropped, so `shape` refuses at the true composition.
Structural requirement is 1.46:1 synthesis:curated; the queue delivers 1.12:1.
Nothing is broken - the corpus is short, and only in one bucket.

Re-measured 2026-08-31T03:40Z against per-variant rates (F24), the pending queue
now projects to **synthesis 2,360 / curated_c2 1,322** at full drain - a 1.79:1
ratio, which is INSIDE the feasible window. The shortfall is no longer the
ratio; it is the absolute synthesis count against the MVP's ~3,207.

Full plan: `~/.claude/plans/mossy-sauteeing-riddle.md` (not in the repo).

## 5. Findings log

### Root cause of the stall (confirmed)

The build has produced nothing since **2026-08-29**, burning a ~5h16m runner
every cron fire. `CLAIMABLE_STATES` is `pending/generating/judging/judging_active`
and the queue holds **none**:

```
run #7 (2026-08-30T16:42Z, 5h16m, "success"):
  task states: accepted=650, format_parked=559, gen_unroutable=5190,
               input_ineligible=42, judge_skipped=5, rejected=2582, stale_prompt=502
  [gen] batch 1: claimed=0 gen-ok=0 err=0 tokens=0
```

`gen_unroutable` = `exhausted:unroutable:cooling` **5,180** + `exhausted:provider-fault` 10.

`routing.generator` is a one-element list (`bai/deepseek-v4-flash`). The breaker
trips after 4 consecutive failures and cools the ref 300 s
(`providers.py:1774-1775`, `1850-1855`). With one ref, cooling empties the pool;
`eligible_refs` yields nothing with reason `"cooling"`; `generate.py:2005-2012`
parks the task **without checking why nothing was eligible**. No call is made, so
all three attempts (`MAX_ATTEMPTS = 3`) burn in milliseconds.

**Tasks destroyed ~= cooldown x claim rate.** The park timestamps confirm it -
the whole crater is one two-hour window:

```
2026-08-29T11  4,554     2026-08-29T10  634
2026-08-29T09      1     2026-08-28T20    1
```

The fix already exists in the codebase and is simply unwired:
`providers.py:1161` defines `TRANSIENT_SKIPS = frozenset({"cooling", "over-budget"})`,
and `eligible_refs(skipped=...)` exists precisely to distinguish transient from
structural. `grep TRANSIENT_SKIPS generate.py` returns nothing.

### The recovery is armed, untested, and pointed at the shredder

`run_worker` re-opens `REOPEN_ON_EMPTY = ("gen_unroutable", "format_parked",
"off_teacher")` on an empty queue (`actions_worker.py:710-738`). It is in
`origin/main` but **has never executed**: it landed in `cb92c60` at
`2026-08-30T21:38Z`, and runs #5-#8 all check out sha `17af942b`. Run #7's log
goes straight from `task states:` to `[gen] started` with no guard line.

**The next cron fire is the first to run it and will re-open 5,190 rows into a
still-broken generator.** Step 1 must land first.

### Verified working - this is a recovery, not a rebuild

- Live probe of `bai/deepseek-v4-flash`: **HTTP 200 in 1.6 s**; key clean.
- 650 accepted rows, incl. 337 synthesis and 310 curated_c2 - up from 17 accepted
  synthesis on Aug 29; ~33% of terminal synthesis outcomes, inside the 20-44%
  forecast.
- **The 5,180 recover for free**: `FREE_PARK_PREFIXES` (`tasks.py:208-215`)
  contains `"exhausted:unroutable:"`, exactly their disposition, so
  `--reopen gen_unroutable` restores state *and* attempt budget.

### Refuted hypotheses - do NOT retry these

1. **"`loaded 0 key(s) from .env` means credentials are missing."** No - the
   `.env` *file* is absent in CI by design; the 5 Actions secrets are passed as
   env vars by `data-worker.yml`.
2. **"`_task_counts` fails because a WAL DB cannot open `mode=ro` without -shm."**
   Tested against the real 637 MB baton DB: `mode=ro` returned counts fine. The
   snapshot is `journal_mode: delete`.
3. **"`Store.open` converts to WAL and *then* `mode=ro` fails."** Explicitly
   simulated the sequence; counts still returned.
4. **"`--reopen` is not `action=append`, so only the last state is used."**
   It *is* `action="append"` (`tasks.py:793-802`). The multi-state call is well formed.
5. **"The auto-reopen ran and failed silently under `check=False`."** It never
   ran at all - the guard postdates every run. There is no second bug.

### Step 2: the fix is proven on real rows and a real breaker (2026-08-31)

Both halves ran against a **scratch copy** of the baton DB
(`scratchpad/livetest/`), never `data/build`, with run #8 already cancelled so
exactly one generating host existed.

*Merit half* - 5 reopened synthesis rows, live on deepseek:

```
batch 1: claimed=5 gen-ok=5 gated-out=4 err=0 tokens=34169 [regenerate=4]
```

1 row to `judging`, 4 `regenerate` (gated on merit), **zero `gen_unroutable`**.

*Breaker half* - the REAL Router, tripped with 4 real `report_failure` calls,
then 5 batches x 5 claims through a wholly-cooling pool:

```
after trip: cooldown_remaining=60.0s          <- config change live, was 300
batch 1..5: claimed=5 gen-ok=0 err=5
  generator pool transiently empty - waiting 60s   (x5)
gen_unroutable: 0
pending attempts: {0: 5185, 1: 4}
```

25 claims through a dead pool destroyed **nothing**; pre-fix that is ~8 rows.
The 4 rows at `attempts=1` are exactly the ones genuinely gated out in the
merit half. The re-open also confirmed the free-park refund: all 5,190 came
back at `attempts=0`.

### Step 5: the transition stream was never a gate problem (2026-08-31)

`transition`: 2,150 rejected / 26 format_parked / 21 gen_unroutable / **3
accepted**. But **2,063 of the 2,150 rejects are `skip:slots`** - refused by
`build_slots` *before any teacher call*. Partitioned by seed source, the split
is exact:

```
s3://indian-supreme-court-judgments     skip:slots 1415 / 1415
L-NLProc/PredEx_...                     skip:slots  367 /  367
L-NLProc/TathyaNyaya-...                skip:slots  281 /  281
tuned/law-v1-transition-grid            skip:slots    0 /  137
```

Root cause in `tasks.py:_candidate_seeds`: the seed-stream clause
`COALESCE(json_extract(meta_json,'$.stream'), ?) = ?` is **one-directional**.
It keeps a transition seed out of a synthesis wave, but a seed declaring
nothing satisfies `COALESCE(NULL,'transition')='transition'` and flows *into*
the transition wave - where the slots it does not carry cannot render.
Verified in the schema: grid seeds carry all four text slots plus both dates;
the generic seeds carry none of them.

Why the grid did not simply fill the wave: there are **1,100 usable grid seeds**
(1,250 minus 150 `held_out`) and only 137 were ever used - and those 137 are
*exactly* the alphabetically-first 137 (`first137_match=True`). A wave takes a
seed_id-ordered prefix of the never-used pool, so only the grid seeds inside
that prefix are reached. Capacity was never the constraint.

Fixed in `708d455` with `tasks.CLOSED_WORLD_STREAMS = {"transition"}`: a
closed-world stream requires the declaration, an open-world one only requires
that it does not name someone else. One existing test asserted the opposite -
`test_a_seed_that_declares_no_stream_is_offered_to_every_wave` - but its stated
reason ("would empty every wave in the build") is about the open-world streams
and still holds for them; including transition in its loop was
over-generalisation that production disproved. Narrowed, not deleted.

Cost accounting: `skip:slots` is refused before the call, so **no tokens were
burned**. What was lost is 2,063 wave slots and a per-seed cap on 2,063 seeds
(transition skips spend the cap; only `statute_qa` is exempt in that subquery).

**Consequence for Step 6:** transition can now draw at most 1,100 seeds x
`PER_SEED_CAP` 4 = 4,400 tasks, minus the 137 spent. A 2,200-row wave therefore
fills, but at ~2 samples per scenario - lower diversity than the plan assumed.
Size the transition mix against 1,100 distinct scenarios, not 2,200.

### Step 3: THE BUILD IS UNSTALLED (2026-08-31T02:19Z)

Run #9 `33349462956`, dispatched on `main` at 02:03Z after run #8 was cancelled.
GitHub will not serve logs for an in-progress run, but **the worker pushes its
own logs to the baton**, which is the better instrument anyway:

```
logs/33349462956/gen.log
  batch 1: claimed=36 gen-ok=36 gated-out=33 err=0 tokens=290411 [regenerate=31 reject=2]
  batch 2: claimed=36 gen-ok=36 gated-out=34 err=0 tokens=253579 [regenerate=30 reject=4]
logs/33349462956/judge.log
  judge_mode=audit audit_sample=0.05
  judge batch 73:  claimed=3 decided=3 accepted=3 [audit-accept=3]
  judge batch 138: claimed=2 decided=2 accepted=2 [audit-accept=2]
```

The baton held **zero claimable rows** before this run (run #7 measured
`pending=0`, and run #8 claimed nothing and changed nothing), so `claimed=36`
is itself the proof that the armed re-open fired. Also on the baton:
`raw/gen/2026-08-31/gen.ndjson` (5.78 MB), and a commit titled
**`periodic checkpoint (raw+streams)` at 02:18:49Z - the first one in the
repo's entire history**, because every previous run generated nothing for the
raw+streams cadence to upload.

`loaded 0 key(s) from .env` appears again and is again correct: the file is
absent in CI by design and the 5 secrets arrive as env vars (refuted
hypothesis 1).

### Step 4 (preliminary): the breaker is not the constraint - the gates are

Zero `err`, zero breaker trips, zero cooldowns across 72 generations. The 14
minutes from worker start to first push produced 2 batches of 36, so roughly
**5 gen/min, ~300/hour**.

What IS striking is `gated-out` 33 and 34 of 36 - a ~92% first-attempt gate
failure. Those are `regenerate` (retryable, and the row keeps its place), not
rejects; only 2 and 4 were rejects. But with `MAX_ATTEMPTS = 3` a row that
never passes lands in `format_parked`, and the baton already holds 559 of
those. Gate pass rate, not routing and not judging, is now the ceiling -
consistent with the 2026-08-28 finding that judge accept was 82.9%.
Measure over a full run before acting; do not tune gates on two batches.

### The corpus arithmetic - what "ready" actually requires

`stats` MEASURES the emitted mix; nothing downsamples to hit it. So the
profile's three shares pin three absolute counts, and replay's 4,320 rows are
what pin the total:

```
v1.0-MVP  grounded_synthesis 0.3010  curated 0.2796  replay 0.4194  (+/-2pp)
x 10,300 =            3,100          2,880          4,320
```

Those are exactly the design counts - replay was built to 4,320 to put the
total at 10,300. With replay fixed, the total must land in [9,832, 10,816],
which sets the real floors:

| bucket | accepted now | floor to pass the gate | gap |
|---|---|---|---|
| grounded_synthesis (synthesis 337 + transition 3) | **340** | 2,763 | **~2,423** |
| curated (curated_c2 310 + curated_c1 1,700) | ~2,010 | 2,552 | ~542 |
| replay | 4,320 | - | 0 |

So the gate is reachable, not structurally impossible - but it needs ~2,400
more grounded_synthesis rows, and that single number is the ship date. The
only other lever is trimming replay, which shrinks the whole corpus
proportionally; that is a product decision, recorded here rather than taken.

### Step 6 sizing, from MEASURED yield (2026-08-31)

Task-level terminal yield on the baton, counting only graded outcomes
(`accepted + rejected + format_parked`; `stale_prompt` and `input_ineligible`
are artefacts, not quality outcomes):

| stream | graded terminal | accepted | yield |
|---|---|---|---|
| synthesis | 1,020 | 337 | **33.0%** |
| curated_c2 | 592 | 310 | **52.4%** |

Against the gaps in the corpus table above:

- **grounded_synthesis** needs ~2,760 more accepted to reach 3,100. The 3,401
  synthesis tasks now queued yield ~1,122, so the current queue closes less
  than half the gap. Closing the rest needs roughly **5,000 further synthesis
  tasks**. Seed headroom is ample (61,853 seeds x `PER_SEED_CAP` 4, ~5k used).
- **curated_c2 needs nothing more.** 1,768 queued x 52.4% = ~926 accepted, so
  curated lands near 2,936 against a 2,880 target and a 3,086 ceiling. Planning
  more would push it OUT of band. Do not widen curated.
- **transition stays where it is.** With `708d455` its wave can draw the 1,100
  usable grid seeds, but the stream converts at 3/137 = 2.2% even on
  well-formed seeds, so widening it now would buy permanent rejects. It is not
  needed for MVP: synthesis has the headroom to carry grounded_synthesis alone.
  Fix the quotation problem first (see below).

**Route: `data-plan.yml` only.** It is the sole sanctioned path to the remote
queue - the worker never plans, and `--phase seed-push` refuses once the remote
owns the baton. `--plan-n` is a TARGET for the stream, not an increment, and it
counts every non-terminally-dead task, so the number to pass is
`current_counted + wanted`, not `wanted`.

**Deliberately NOT dispatched tonight.** Two reasons, both about the fence:
`data-plan` shares the `data-build` concurrency group, which holds one running
plus one pending run, and a second pending trigger REPLACES the one already
waiting - so a plan dispatched now would be silently displaced by the 05:17Z
cron. And there is no urgency: 5,190 queued rows at ~300 gen/hour is well over
a day of work. Dispatch it when the queue is closer to drained, sized on yield
measured after the fix rather than before it.

### The transition stream has a SECOND defect, unfixed

Separate from the planner bug: of the 137 well-formed grid-seeded rows, 3 were
accepted and 87 rejected, and `statutory_quotation` fires on 85 of those 87
(97.7%). The prompt hands the teacher the build's "operative effect" prose for
the savings clause and tells it explicitly not to present that as quoted
section text; the model quotes it anyway. On this stream `temporal` and
`answer_key` are PERMANENT gates, so a format slip that would merely
`regenerate` on synthesis becomes an irreversible `reject` here - which is why
the same gates that synthesis survives convert transition at 2%.

Not fixed tonight, and deliberately so: it is a generator-behaviour problem,
the stream is not on the MVP critical path, and prompt levers on this teacher
have a recorded history of not working (the 2026-08-27 paired A/B on trace
length). Fix it before any transition wave is widened, not after.

### RETRACTED: the "trace budget" blocker was already solved by `shape.py`

I recorded a P0 blocker here claiming the trace and empty-think gates were RED
by construction. **That was wrong, and the correction matters more than the
finding.** The measurement was right; the conclusion was not.

What I measured and still stands:

| stream | rows | `_prov.reasoning` true | false (empty-think) |
|---|---|---|---|
| `streams/replay.jsonl` | 4,320 | 3,120 | 1,200 |
| `streams/curated_c1.jsonl` | 1,700 | 300 | 1,400 |

What I missed: `src/tuned/data/shape.py` exists for exactly this, and its
docstring states the problem in the same terms before solving it - *"the pools
between them hold 2,452 no-think rows. Both cannot fit... Feeding the target
mix with the pools' own composition lands empty-think at 34.3%, still red"*,
and *"TRACE: the exact complement of empty-think... the same measurement, not a
second problem."*

The builders produce **deliberately oversized fixed pools**. `shape` runs first
in the assembly chain (`actions_worker.assemble_argvs`:
`verify -> shape --profile -> decontaminate -> dedupe -> split -> assemble ->
stats`) and trims them to a subset that hits both the mix and the empty-think
window. It writes new `shaped_*.jsonl` files and never touches the pools, so a
later run with more generated rows re-derives a bigger corpus from the same
inputs.

**The consequence is the opposite of what I wrote.** The corpus is sized off
the scarce resource: `N = generated_synthesis / 0.301`, because
grounded_synthesis can only come from the teacher. So the gates can be green at
essentially any size, and generation volume buys corpus SIZE rather than
greenness. At today's 340 accepted grounded_synthesis that is ~1,130 rows; the
10,300 in the config is the finished target, not a threshold to clear before
anything can ship.

**The real open decision is `--replay-nothink-share`**, and it is a genuine
one: the no-think budget can be filled from replay's chat slices
(smoltalk_nothink / wildchat_prof / legal_qa_empty) or from curated_c1's raw
legal rows (PredEx / aalap). The default preserves the design's intent -
no-think trained on chit-chat, not on legal prediction - and yields ~2.05
corpus rows per generated row; sourcing it from raw legal rows instead yields
~2.88 but is, in shape.py's own words, "a TRAINING-DATA DESIGN CHANGE and
deliberately not the default." That is an operator call about what the model
learns, not a gate to be cleared, and it is left unmade.

Lesson for this file: `prev_rep.md` not mentioning empty_think meant nobody had
written a REPORT about it, not that nobody had solved it. Read the module that
owns a concern before declaring the concern unowned.

### F13. Gate forensics: `length_band` is NOT a lever, and the lever was already pulled

Subagent forensics (2026-08-31) ranked `length_band` the #1 blocking gate for
synthesis (43.9%) and #2 for curated_c2 (42.7%) and called it "a genuinely
miscalibrated global length band". **Re-verified, and that reading is wrong.**

`gate_result.detail_json` records the band *as it stood at gate time*, so the
question "is this measurement current?" is answerable rather than assumed:

    band as recorded (think_max, total_max, total_min, answer_min)
      (4000, 8192, 300, 120) -> 1350 failures
      (3000, 8192, 300, 120) ->  562 failures

**Every recorded failure was graded under think_max 3000 or 4000. None under the
live 4500.** Replaying `check_length_band`'s arithmetic over all 4,246 verdicts
with total_max held at 8192:

    think_max | overall | synthesis | curated_c2 | transition
         3000 |   47.5% |     49.8% |      48.0% |      14.8%
         4000 |   55.7% |     57.4% |      57.3% |      24.9%
         4500 |   58.3% |     60.1% |      59.2% |      29.2%   <- live now
         5000 |   59.6% |     61.7% |      59.7% |      32.1%
         6000 |   60.2% |     62.4% |      60.0% |      34.0%
         8192 |   60.3% |     62.5% |      60.0% |      34.0%   <- ceiling

The shipped 3000->4500 move already banked **+10.8pp**; everything above 4500 is
worth **+2.0pp total** and saturates by 6000. The lever is spent.

It is spent because the band is not arbitrary: `total_max` is **8192, the
training sequence length** (`drop-never-truncate >8192`), and `total` is
prompt+think+answer. Of the 1,687 rows that still fail at an *infinite*
think_max, **1,200 fail `total>total_max`** - they do not fit in the model's
context and no gate setting changes that. Trace length is not steerable by
prompt either (paired A/B 2026-08-27, all three levers exhausted).

Corroboration from the other direction: `ebde9a7` bought its irac_placement fix
at a cost of **summarization length_band -15.91pp**, logged there as a known
follow-up. Length is the binding constraint, not a mistuned knob.

Second-order finding, not chased: 469 rows (24.5% of failures, 454 of them
synthesis) fail `think<think_min` at think_min=500 - the *opposite* direction.
Synthesis fails this band at both ends.

### F14. `irac_placement` is a deepseek compliance gap; the 08-28 precedent does not transfer

Same forensics: irac_placement fails 63.8% on curated_c2/irac_analysis and 53.6%
on synthesis/irac_analysis, but only 5.2% on statute_qa - a 15x spread that has
the exact shape of the 2026-08-28 `irac_placement` drift. It is not that.

- **Model, not task.** Same task, same prompt population: deepseek 63.8-67.8%,
  gpt-oss 26.8%. All four irac_analysis prompt versions fail 43-78%, so no single
  bad template edit is responsible.
- **The 08-28 fix removed a contradiction, it did not restate a rule.** `ebde9a7`
  dropped a headed-IRAC *answer* mandate that `gates.IRAC_ANSWER_TASK_TYPES` had
  already stopped requiring for summarization. For irac_analysis the heading
  mandate is correct and the gate does require it - there is no contradiction to
  remove. The template already says the rule explicitly ("headings ... never
  inside your reasoning"); deepseek writes them into `think` anyway.
- The gate is firing correctly. Sampled failures (gen_id 1438, 166) contain real
  substantive conclusions under labels inside `think` - the exact "hidden first
  draft" the gate targets, not false positives.

**No cheap win. Generator swap is foreclosed by the 2026-08-28 allocation
directive.** Logged, not acted on.

### F15. A template fix must ship as a NEW prompt_id, never as an edit

`task_id_for` hashes `seed|task_type|prompt_id|sample_ix` - **not** `prompt_sha`
(`tasks.py:239-248`), and `prompt_sha` is written once at plan time
(`tasks.py:584`) with no re-stamp anywhere. `generate.py:1701` parks a row the
moment planned_sha != live_sha. So editing `gen_irac_analysis_v4.md` parks every
task already planned against it as `stale_prompt`, permanently - `--reopen
stale_prompt` returns it to `pending` where it re-parks on the next claim.

Adding `..._v5` as a new prompt_id instead yields new task_ids, plans cleanly,
and leaves v4's pending rows generating against an unchanged v4 file. That is the
only safe shape for any future template fix, and it is why the four existing
versions exist.


### F16. THE CHAIN IS GREEN - and the corpus is short on synthesis, not broken

**First end-to-end assembly run in the repo's history. `CHAIN RC=0`, verdict
GREEN at profile v1.0-MVP**, all nine gates passing on 716 rows:

    chain PASS | length PASS p50 3532 / p99 7039, max 8088 (limit 8192)
    mix   PASS | curated 28.1% (28%), grounded_synthesis 28.9% (30%), replay 43.0% (42%)
    trace PASS 80.0% (>=80%) | empty_think PASS 20.0% (>=18, <=20)
    dup PASS 0.0% | markup PASS | license PASS | cross_code PASS 0

This retires the standing risk "three gates are red / `build_manifest.json` has
never existed". The three gates were never miscalibrated - **the pre-2026-08-29
runs simply never ran `shape`**, and shipping pools sized for the FINISHED
corpus into a half-generated one guarantees a replay-dominated mix.
`assemble_argvs` inserts shape only when `streams` is passed, and CI does pass
it (read off the bundle's own `streams/` dir), so CI is on the fixed path.
`build_manifest.json` is written by `push.py`, which runs only after stats goes
green - so it has never existed because the chain never got that far, not
because anything is missing.

Two local-only gaps, neither a CI defect, both checked rather than assumed:
- decontaminate needs the eval parquets (bbl/iltur/aibe) under `corpus/hf/`.
  `stage_bundle` deliberately keeps them off the baton ("ONE file, never the
  corpus dir - 1.9 GB"), and `run_assemble` re-fetches them on the runner from
  `EVAL_SETS` before the chain. Only my scratch tree lacked them.
- "semantic layer did NOT run (semhash-not-installed)" is my local env; semhash
  is in the `[build]` extra that `data-assemble.yml` installs.

**The one real blocker: `curated_c2` is over-generated relative to synthesis.**
At the true composition shape REFUSES - *"curated/trace would need -136 rows -
the generated rows already in that bucket overfill it"*. Generated rows cannot
be dropped (shape trims stream files; decontaminate reads every accepted
generation), so a smaller corpus raises the synthesis share instead of lowering
it. I got the GREEN above only by simulating curated_c2 down to 120 in the
scratch DB. Measured requirement, by asking `shape.plan` directly:

    gen_curated |  min gen_synth | ratio | corpus
            309 |            455 |  1.47 |   1512
            800 |           1170 |  1.46 |   3887
           1377 |           2015 |  1.46 |   6694
           2000 |           2925 |  1.46 |   9718

**The required ratio is a structural 1.46:1**, flat across the whole range - it
is set by the profile targets and the curated_c1 no-think pool, not by volume.

**And the queue does not deliver it.** The reopened pool run #9 is eating is
63.8% synthesis / 35.4% curated_c2 (3,665 / 2,037), but the two streams accept
at different rates - synthesis 33.0%, curated_c2 52.4% - so accepted rows arrive
at only **1.12:1**:

    at queue exhaustion:  synthesis 337 + 3665x0.330 = 1546
                          curated_c2 310 + 2037x0.524 = 1377   (1.12)
    shape needs 2015 synthesis at that curated count -> SHORT BY 469

So draining the current queue does not make the corpus shippable. This is the
measurement Step 6 was waiting for, and it confirms the earlier "do not widen
curated_c2" call - widening it raises the bar it is already under.

**Step 6, now sized:** the next `data-plan` dispatch is **synthesis-only**, for
roughly **+500 accepted synthesis rows** (~1,500 planned tasks at the measured
33% accept rate), `--mix` explicit because statute_qa silently under-fills.

**Not dispatched tonight, deliberately - and the timing rule, corrected.**
`data-plan.yml` shares the `data-build` concurrency group, which is exactly what
stops a planner from stealing the baton from a running worker. My first reading
was that a dispatch now would "very likely be discarded"; that is too strong and
contradicts the cron's own rationale, which picks */4 over */3 precisely so that
"nothing is ever displaced; ... the thing displaced could be an operator's
data-assemble dispatch". The real rule is narrower:

The group holds one running plus one pending, and a new trigger REPLACES the
pending one, so **a dispatch runs only if it is the LAST trigger to arrive
before the running job ends**. Cron is `17 */4` (00:17 / 04:17 / 08:17Z). Run #9
was dispatched manually at 02:03Z, off-phase, and ends ~07:28Z. So the window in
which a data-plan dispatch actually survives is **after 04:17Z and before
07:28Z** - it replaces the 04:17 pending run and starts when #9 finishes. The
cost is one skipped worker cycle (~49 min of generation before the 08:17 cron).

It is still not worth firing tonight, for two reasons that outlast the timing:
the queue is not ~35h of work but closer to **60h** - 5,749 tasks where the
first batches show ~31 of 36 going to `regenerate`, so most tasks are claimed
about three times - and planning a wave now LOCKS IN the current ~33% yield
under the present templates. Given F14/F18 (both live gates are generator
non-compliance, not thresholds), the next wave is worth more after a template
decision than before one. Fire it when the queue actually nears exhaustion.


**Accept rates re-measured with explicit exclusions (2026-08-31 03:15Z).** The
earlier 33.0% for synthesis was slightly optimistic; excluding `gen_unroutable`
(a shredder artifact, not a merit outcome) and `skip:slots` (the planner bug -
no answer was ever bought) the terminal rates are:

    stream       accepted  rejected  parked  ineligible   accept
    curated_c2        310        13     269           0    52.4%
    synthesis         337       419     264          42    31.7%
    transition          3        87      26           0     2.6%

curated_c2 accepts at **1.65x** synthesis's rate, which is the whole reason the
1.12:1 arrival ratio sits below the 1.47:1 the window needs. Note the shape of
the difference too: curated_c2 barely gets *rejected* (13) and mostly *parks*
(269), while synthesis carries 419 rejects - permanent gates and judge calls.
At 31.7% the queue's endpoint is synth ~1,499 rather than 1,546, so the
shortfall is **~+520 accepted synthesis**, not ~+470. Conclusions unchanged.

### F17. Live throughput, measured on run #9

    5 gen batches, claimed=36 each: gen-ok 35-36, err 0-1, cooling 0, unroutable 0
    gated-out 32-34 of 36 -> ~3 rows reach judging per batch
    judge: audit mode, claimed=3 decided=3 accepted=3 [audit-accept=3]

180 generations in 38 min = **4.72 generations/min**, ~**23.6 accepted rows/hour**
across all streams. The shredder fix is holding under real load: zero cooling
parks in 180 generations. Gates remain the ceiling (~8% of generations survive),
exactly as the 2026-08-28 finding said - not routing, not judging.


### F18. Step 5 answered: transition died of the planner bug, and its remainder is a compliance gap

The stream reads as "2,150 rejected against 3 accepted", which looks like a
catastrophic gate problem. It is two different things, and neither is that.

**96% of it never bought an answer.** Of 2,150 rejected transition tasks, only
**87 have a generation at all**; **2,063 carry disposition `skip:slots`** -
`build_prompt` raised SlotError, "the task cannot be rendered from what the seed
carries", before any teacher was called. Joining task -> seed partitions it
perfectly, with no overlap:

    transition skip:slots      -> seed declares nothing : 2,063  (100%)
    transition that generated  -> seed declares transition : 117  (100%)

**That is the planner bug Step 5 already fixed** (`708d455`): `_candidate_seeds`
offered `transition` - a CLOSED-WORLD stream, whose task is built from the
seed's META - to seeds that declare no stream and therefore carry no transition
metadata. `tasks.py:137` had already written down the symptom ("every one died
`skip:slots` before a teacher was called") without the cause being closed.

**These 2,063 are correctly dead. Do not reopen them** - their seeds cannot
render a transition prompt, so they would re-die on the next claim, and
`rejected` is TERMINALLY_DEAD for exactly this reason. The remedy is to re-plan
transition against the 1,250-seed grid, which the shipped fix now makes the only
thing the planner can do.

**On the real seeds the yield is still 2.6% (3 of 117), and the blocker is
`statutory_quotation` at 95.7%** (200 of 209 verdicts):

    statutory_quotation  95.7% | irac_placement 76.1% | length_band 75.1%
    prompt_echo          40.2% | answer_key     29.7% | temporal      19.6%

The gate is not miscalibrated and is not misscoped - it is **transition-only by
design**, because that stream's grounding "is not statute text at all", so a
quoted span attributed to a section cannot be verified against any bare-act
corpus and is refused whether it was paraphrased or invented. It is a
`regenerate`, not a permanent gate, and the prompt already forbids the act.
deepseek quotes statute anyway, in 96% of attempts.

**So this is F14 again, in a second place**: a negative constraint the template
states plainly and the sole generator does not honour. Together they explain the
~8% gate survival rate directly - gates are the ceiling, as the 2026-08-28
finding said, and what they are catching is generator non-compliance rather than
mistuned thresholds.

**Consequence for Step 6, and it is the useful one:** `source_streams` maps
`transition: grounded_synthesis` (`data_law_v1.yaml:161`), so transition rows
compete for the *same* bucket as synthesis and could in principle have closed the
469-row gap. At 2.6% they cannot - re-planning the whole 1,250-seed grid buys
roughly 30 accepted rows. **The shortfall has to be closed by synthesis**, which
is what Step 6 already says. Transition stays a moat feature, not a volume
source.


### F22. The 36 -> 24 batch drop is the `transition` stream ending, not the queue draining

`claim_tasks(worker_id, n_workers, stream=...)` is called **once per stream**
(`generate.py:2210`), `--n-workers` defaults to **12** (`actions_worker.py:1056`),
and `STREAMS` has three entries. So the batch size is
`12 x (number of streams with claimable work)` and nothing else:

    36 = 3 streams x 12      batches 1-8
    26 = transition running out mid-batch
    24 = 2 streams x 12      batches 10, 11 - stable

Baton DB at 03:03Z confirms it. `transition`: **pending=0, generating=0**,
2,177 rejected / 17 format_parked / 5 accepted / 1 gen_unroutable = 2,200 =
the whole planned stream. It is spent. `synthesis` still has 3,602 pending,
`curated_c2` 1,978.

**This is good news, not bad.** transition converted at 4.3% this run and was
holding a third of an account-level rate bucket to do it. That third is now
serving streams that convert at ~41%.

**But it exposed a real throughput defect** (see F23): the claim cap is
per-stream, so a stream ending *reduces in-flight work* with no compensation.
The fleet went from 36 calls in flight to 24 against an unchanged rpm-8 bucket.


### F19. The corpus has a hard ceiling at ~10,021 rows, and the pools were built for it

Asking `shape.plan` across the whole (synthesis x curated) grid instead of one
point turns the "we are short on synthesis" story into something more precise,
and corrects my own first reading of it.

**The constraint is two-sided.** For every synthesis count there is a *window*
of feasible curated_c2 counts, roughly `gs/2.18 <= gc <= gs/1.47`, about 300-400
rows wide. Too many generated curated rows overfill the curated/trace bucket;
too few leave the curated/trace POOL (only 300 rows) unable to fill it. So
"freeze curated_c2" is wrong - it breaks the lower bound as synthesis grows:

     gen_synth |  min gc |  max gc | corpus       POOLS ON HAND
           455 |      10 |     310 |   1512         curated/nothink 1400
          1000 |     360 |     680 |   3322         curated/trace    300
          1546 |     710 |    1050 |   5136         replay/trace    3120
          2015 |    1010 |    1380 |   6694         replay/nothink  1200
          3000 |    1640 |    2050 |   9967
          3212 |       - |    2060 |  10021  <- CEILING
          3218 |  infeasible at any curated count

**Bisected ceiling: gen_synth 3,212 -> 10,021 rows.** One row more and it
refuses: *"replay/nothink needs 1,281 rows (to keep 1,246 after losses) but the
pool holds 1,200."* The binding pool is **replay/nothink**.

**The `--replay-nothink-share` knob cannot lift it** - I tested, rather than
assuming, because the refusal names that lever and agent_0 had it logged as an
unmade operator decision. The pool's as-built share is 0.278 and that is already
optimal; moving it down shifts demand onto replay/trace, which binds sooner:

    as-built 0.278 -> 3,204 synth / 10,021 rows      0.15 -> 2,784 / 8,704
        0.24 -> 3,120 / 9,749                        0.05 -> 2,364 / 7,404

**This is a design point, not a defect.** The replay pool is 4,320 rows and the
replay target is 41.9%; 4,320 / 0.419 = 10,310. The pools were sized for a
~10.3k-row corpus and they deliver almost exactly that. Nothing is
mis-provisioned.

What it does mean, concretely:
- **The real synthesis target is ~3,212 accepted rows, not the ~3,617 the plan
  carried.** Generating past ~3,212 does not buy a bigger corpus - it makes the
  corpus *unshippable*, because generated rows cannot be dropped and the mix
  then cannot be hit within +/-2pp at any size.
- A corpus beyond ~10k - the 15-20k the 2026-08-07 dataset spec asks for -
  requires **rebuilding the replay and curated pools larger**, which is
  downloaded public data and costs no generator time at all. That work is
  independent of everything the fleet is doing and is the single cheapest way
  to raise the ceiling.
- The queue's endpoint (synth 1,546 / curated 1,377) sits *above* the window's
  top for that synthesis count (1,050), which is the same shortfall F16
  measured, seen from the other side.

Suite green at the time of writing: **3,752 passed, 19 skipped**.


### F20. Operational: do NOT dispatch `data-assemble` yet, and the one command that says when

`shape` refuses at today's true composition, and `run_assemble` stops the chain
on the first non-zero rc - so a `data-assemble` dispatch right now burns a runner
and produces nothing. That refusal is the system working: it declines to ship a
corpus whose mix it cannot hit, rather than shipping a replay-dominated one,
which is exactly what the unshaped pre-08-29 runs did.

**No new tooling is needed to know when it is ready** - `shape` already answers
it, and its refusal states the shortfall in rows:

    python -m tuned.data.shape --config data/configs/data_law_v1.yaml --profile v1.0-MVP

- Prints a plan and writes `out/shaped_*.jsonl` -> the chain will run; dispatch
  `data-assemble`.
- `REFUSED: ... curated/trace would need -N rows` -> curated_c2 is above the
  window for the synthesis on hand; **more synthesis** (not less curated).
- `REFUSED: ... curated/trace needs N rows ... pool holds 300` -> the opposite
  end; curated_c2 is below the window.
- `REFUSED: ... replay/nothink needs N ... pool holds 1200` -> the F19 ceiling;
  no amount of generation helps, the pools must be rebuilt larger.

It reads the live store, so run it against a **read-only or scratch copy** while
a worker holds the baton, never against `data/build` directly.

Rebuilding the replay/curated pools larger (the F19 remedy) also writes into
`streams/`, which the baton owns - so that work waits for a window with no
worker running, or it risks a BATON STOLEN.


### F21. The profile decision, costed - and the two different "how many rows" numbers

`agent_0`/memory both carry "which profile ships is an UNMADE DECISION". It can
now be costed against the pools actually on hand, and the answer is lopsided:

    profile      | targets (synth/cur/replay) | ceiling gen_synth | max corpus
    v1.0-MVP     | 0.301 / 0.28  / 0.419      |             3,207 |     10,021
    v1.1-full    | 0.600 / 0.16  / 0.240      |             6,289 |     10,147

**Both profiles cap at the same ~10.1k corpus, and v1.1-full costs nearly twice
the generated synthesis to reach it.** The binding pool just moves: replay/nothink
under MVP, the curated pool under full. So v1.1-full's nominal 18,000-row target
is not reachable with these pools either - it buys ~130 more rows for ~3,000
more accepted synthesis rows, which at the measured rate is weeks of generator
time. **On current pools v1.0-MVP dominates**; v1.1-full is only worth choosing
for the higher teacher-generated PROPORTION (60% vs 30% of the corpus), and that
argument should be made on training grounds, not on size.

**Two numbers, and they are not the same target** - I had been quoting the first
without naming it:

- **~+470 accepted synthesis** brings the counts inside the feasible window, so
  the chain STOPS REFUSING and a corpus ships. That corpus is ~6,700 rows.
- **~+2,870 accepted synthesis** (337 -> 3,207) reaches the MVP ceiling and the
  full ~10,021-row corpus.

At the measured rate - ~90 tasks completed/hour, 63.8% of the queue synthesis,
33.0% terminal accept - synthesis accrues at roughly **19 accepted rows/hour**,
so the ceiling is **~6-7 days of continuous cron generation**, and the shippable
window is ~1 day away.

The queue cannot get there on its own: its 3,665 synthesis tasks yield ~1,209
accepted, against the 2,870 needed. **Step 6 therefore needs ~5,000 more
synthesis tasks, not the ~1,515 sized for the window alone** - a `data-plan`
target of roughly `current_counted + 5,000`, synthesis only, `--mix` explicit.
Still gated on the template decision in F14/F18, since planning locks the wave
to today's prompt_ids.


### F23. FIXED: the claim budget belonged to the streams, not to the fleet

The bound the design chose is `n_workers * len(streams)` - sized against the
rate bucket and the 900 s lease, and written into the comment above the loop.
The loop did not enforce it. It asked each stream for `n_workers` and kept
whatever came back, so the real in-flight count was
`n_workers * (streams that still HAVE work)`. F22's drained stream therefore
cut the fleet by a third with nothing wrong and nothing logged.

The rate bucket is a `TokenBucket` per (provider, model): capacity `rpm`=8,
refill 8/60 = one request per 7.5 s, and it **starts full**. So a batch of N
admits 8 immediately and the rest at 7.5 s apiece, then waits on the gather
barrier for the slowest call. Measured against run #9 - 11 batches, 5.3 min
apart, 271 calls/hour - the model that reproduces it is:

    batch_wall = 7.5 * (N - 8) + T_tail,  T_tail ~ 198 s

    N = 12  (1 stream, after curated_c2 drains)   189 calls/hr
    N = 24  (2 streams, TODAY)                    272 calls/hr   <- measured 271
    N = 36  (the designed bound)                  318 calls/hr
    N = 48  (3 streams x 16, the comment's own)   347 calls/hr

The fix restores the designed 36 and, more importantly, makes it **invariant to
how many streams still have work** - so when `curated_c2` drains in ~1.5 days
the fleet stays at 36 instead of falling to 12, which is where the bigger half
of the win is (+68%, not +17%).

Implementation (`generate.py`, the claim loop): keep the per-stream ask as a
FAIRNESS FLOOR, then spend the unspent remainder on whoever can use it.

    budget = n_workers * len(streams)
    for stream in streams:                       # floor: nobody gets starved
        claimed.extend(store.claim_tasks(worker_id, n_workers, stream=stream))
    for stream in streams:                       # top-up: budget is the fleet's
        if len(claimed) >= budget: break
        claimed.extend(store.claim_tasks(worker_id, budget - len(claimed), stream=stream))

The ceiling never rises above the sized bound, and the top-up only ever asks
streams the caller passed in - so `--stream synthesis` still means only
synthesis. Three tests: a drained stream must not shrink the batch; a
one-task stream must still get its task (no pre-emption); and with every
stream full the top-up must find nothing to do. Suite **3,755 / 19 skipped**.

**Falsifiable prediction for the next run** (starts ~07:25Z, when run #9's
315 minutes are up and the 04:17Z cron's queued job takes the group): with
`transition` empty, the first pass claims 12 synthesis + 0 transition + 12
curated_c2 = 24, and the top-up brings it to the budget of 36 - allocated
`synthesis` first, so **`claimed=36` with synthesis holding 24 of it.** If the
log still says `claimed=24`, the fix did not reach the runner.

**Not done, deliberately:** raising `--n-workers` past 12. 48 in flight is
blessed by the comment's own lease arithmetic and would add a further ~9%, but
it is a tuning decision on top of a bug fix, and the two should not ship in one
commit.

#### CONFIRMED 2026-08-31T08:20Z, on run 33363831595

Read off `logs/33363831595/gen.log` on the baton - the in-progress job's log is
404 from both `gh run view --log` and the jobs API, but the worker checkpoints
`logs/` to the HF state repo every 15 min, which is a read-only path that works
while the run is live. Worth remembering: that is how to watch a run without
waiting ~5 h for it to finish.

    batch 1..16: claimed=36 gen-ok=36 err=0 lost-lease=0

**`claimed=36` on all 16 batches.** The prediction's failure mode was
`claimed=24`, so the fix reached the runner and the budget is now invariant to
how many streams have work. `err=0` and `lost-lease=0` throughout: no breaker
trips, no expired leases.

Two things the log says that F23 did not ask about:

**Gates are the whole cost.** `gated-out` runs 22-34 of every 36, with
`regenerate=` almost equal to it and `reject=` at 0-1. So ~78% of generations
are regenerated, which is the same fact F36 measures per variant - and it is
why the variant choice is worth 3.6x rather than a few points.

**The judge budget is fine, checked rather than assumed.**
`groq/qwen/qwen3.6-27b left=72k` and falling ~10k per 16 batches, so it
exhausts in ~7 h. That is a non-event twice over: the judge pool continues
`cerebras/gemma-4-31b` (left=873k) then `bai/deepseek-v4-flash` (uncapped),
and in `judge_mode: audit` a sampled row the fleet cannot serve ships as
`audit:gate-accept:unjudged` at the attempt cap instead of parking
(`judge.py`, the MAX_JUDGE_ATTEMPTS branch). So groq running dry costs some
audit evidence, never a stall. No action.

**Throughput, measured: ~75 gate-passing rows/hour. Two wrong answers came
first, and how they were wrong is the finding.**

`gen.log` carries no timestamps, so I read the batch counter and paired it
with the wall-clock time *I fetched it*. That gave 16 batches at "08:15Z" and
18 at "09:11Z" - 2 batches in 56 minutes - and I wrote down ~17 rows/hour and
called my own earlier "looks faster than 23.6/hour" impression refuted. Both
numbers were junk, in opposite directions, for the same reason: **my fetch
time is not the log's time.**

The log's real time is recoverable and exact. The baton is a git repo, so
every checkpoint push is a commit with a timestamp, and the pushes land on a
clean 15:00 cadence. Reading `gen.log` at each revision gives the whole
series rather than two smeared points:

    07:34:24Z   batch  2      08:34:37Z   batch 13
    07:49:26Z   batch  5      08:49:39Z   batch 16
    08:04:28Z   batch  8      09:04:41Z   batch 18
    08:19:35Z   batch 11

16 batches in 90.3 min = **5.64 min/batch**, steady (2-3 per checkpoint, no
drift). At claimed=36 that is **383 generations/hour**, and at the measured
19.7% gate-pass over the same window (113/575) that is **~75 gate-passing
rows/hour** - three times the 23.6 on record, so the original impression was
right and my "correction" was the worse of the two errors.

The denominators were also wrong, both ways. The run *triggered* at 06:20:57Z
but its job *started* 07:18:59Z - two seconds after the previous job ended at
07:18:57Z. That gap is the whole story of the cadence: the `data-build`
concurrency group hands off back to back, and the 4 h cron against a 5h15m32s
job means one is always waiting. Setup costs ~4 min (job start 07:18:59,
batch 1 begins ~07:23), so a run is ~5h12m of generating, ~390 gate-passing
rows, and the fleet runs **continuously at ~1,800 rows/day**.

Rule, since I got this wrong twice: **read the rate from the baton's commit
timestamps, never from when I happened to fetch.**

Cumulative for the run at 09:04:41Z: **647 generations, 127 past the gates =
19.6%**, 4.39M tokens, 6,782 tokens/generation. That is the pooled figure for
the CURRENT three-stream mix; it should move once the throttle leaves
synthesis + transition, and again on the F36 variants.


### F24. CORRECTION to F14 - `irac_placement` is not a compliance gap, it is GENRE

F14 called `irac_placement` "deepseek not honouring a rule the template states
plainly". That was wrong, and the evidence to refute it was already in the
store. **The prompt variant rotation is a randomised controlled trial that has
already run**, at 165-1,027 generations per arm: `pick_variant` is
`sha256(seed_id:sample_ix) % len(pool)`, so seeds are assigned to variants by
hash - balanced by construction, no seed-composition confound.

First, the failure is not the one F14 assumed. Of 2,381 failing
`irac_placement` gate results:

    2,162   IRAC headings found INSIDE the trace        (the MSLR tripwire)
      128   headings missing from the answer
       91   both

**94.6% is the trace tripwire.** The model is writing `Issue: / Rule: /
Application: / Conclusion:` in its own reasoning.

Now the decisive part. `gen_summarization_v1` and `v2` carry the anti-heading
clause **as the same sentence, word for word** - "Issue, Rule, Application and
Conclusion are not words your reasoning may put at the head of a line either".
Their failure rates:

    gen_summarization_v1   50/187 = 27%     persona: law reporter settling a HEADNOTE
    gen_summarization_v2    7/165 =  4%     persona: advocate writing a LETTER to a client

z = 5.7, p < 1e-7. Same instruction, same model, same gate, randomised seeds,
**7x apart**. The instruction is not being ignored - it is being outvoted by
the genre the persona evokes. The same split runs through irac_analysis:

    prompt_id                gens  acc%   gens/accepted-row  irac_fail%  persona
    gen_irac_analysis_v1      770  80.6%        3.3             43%      a judge writing judgment
    gen_irac_analysis_v3      726  83.5%        4.2             50%      senior advocate, ALOUD to a junior
    gen_irac_analysis_v2      784  62.3%        7.2             57%      advocate advising a client
    gen_irac_analysis_v4     1027  44.6%       15.6             78%      examiner writing a MODEL ANSWER

`v4` also leads on `banned_meta` (26% vs 10-14%) and `prompt_echo` (27% vs
6-17%). Of course it does: a model answer written to be marked *is* a labelled
IRAC artefact with commentary about marking. The persona asks for exactly what
three gates forbid.

**The rule: a persona whose output genre is a formally structured DOCUMENT
(headnote, model answer) leaks that structure into the trace. A persona whose
genre is SPEECH or a LETTER does not.** This is the same mechanism as the
2026-08-28 harmony genre-form fix, which is now twice-confirmed.

#### What it costs, in the only currency that matters

`gens/accepted-row` is fleet time. `v4` spends **15.6 generations per accepted
row against `v1`'s 3.3** - 4.7x. Over the pending pool:

    the two DOCUMENT-genre variants (irac v4, summ v1):
        2,221 tasks  ->   9,767 calls  ->    816 accepted rows
    the same 2,221 tasks at the SPEECH/LETTER variants' measured rates:
                     ->   6,553 calls  ->  1,401 accepted rows

**33% fewer calls and 72% more rows, for the same tasks.** That is a bigger
prize than every throughput lever in this file combined, and it is free -
it costs one line in the Step 6 plan command.

#### The mechanism is CLEAR, and adding a variant is safe - verified

Retiring a variant would mean deleting its template, and the file must stay:
678 pending `v4` tasks are stamped with that `prompt_id` and `generate.py:1701`
checks its sha. So the move is ADDITIVE. Three things checked before saying it
is safe:

1. **`pick_variant` runs at PLAN time only** - the single call site is
   `tasks.py:576`. `generate.py` reads `task["prompt_id"]` and loads that
   template. **Adding a file cannot touch a pending row.**
2. **Growing the pool DOES re-map future planning** - `pool[digest % len(pool)]`
   goes 4 -> 5, so the `pick_variant` docstring's guarantee ("a wave replanned
   tomorrow must reproduce today's assignment exactly") stops holding across
   the change, and `commit_rows`' INSERT-OR-IGNORE crash-resume idempotence
   does not span it. Bounded, not dangerous: `_existing_in_queue` makes
   `--plan-n` a whole-stream target and `PER_SEED_CAP` is 4, so a replan at an
   unchanged N still creates nothing.
3. **Two tests force the change to be acknowledged**, by design - the sha table
   at `tests/test_build_prompts.py:114` (the same tripwire F15 relies on) and
   the exact-tuple assertion at `:843`. Adding `v5` is: the template, one sha
   line, one tuple entry.

So the blocker is NOT mechanism. It is that I cannot validate a new prompt's
CONTENT without a live call, and the account-level rate bucket is held by CI.
Nothing is on the critical path either: the pending queue holds ~2.4 days.

**Ready for the operator, one decision, ~15 minutes:** author
`gen_irac_analysis_v5` (senior-advocate-aloud, the `v3` genre) and
`gen_summarization_v3` (letter, the `v2` genre), add the two test entries,
run a 10-row A/B against `v4`/`v1` on a scratch DB with the worker paused,
then plan Step 6 on the winners. Adding two prose-genre variants dilutes `v4`
from 25% to 17% of new tasks even if they only perform averagely.


### F25. The diagnostic gate is not warning us about the shipped corpus

`self_verification` is the one gate in `DIAGNOSTIC_GATES` (`gates.py:111`):
recorded, never enforced. Its raw rate looks alarming - it fails **55-56% of
all summarization generations** - and the obvious reading is that half the
corpus teaches a trace that never doubts itself, which is the exact habit
`gates.py:118` says the dataset must not teach.

That reading is wrong, and the right denominator says so. Scored on **the
generation that actually WON each accepted task** (its highest attempt, which
is the text that ships):

    stream      task_type        accepted   no cue      %
    curated_c2  irac_analysis         329       32    9.7%
    synthesis   irac_analysis         251       18    7.2%
    synthesis   summarization          52        7   13.5%
    synthesis   statute_qa             35        0    0.0%
    synthesis   drafting               20        1    5.0%
    transition  transition              5        0    0.0%
    TOTAL                             692       58    8.4%

**91.6% of shipped rows carry a verification cue.** The enforced gates are
cleaning up the diagnostic one for free: a generation too lazy to doubt itself
is usually also failing `length_band` or `irac_placement`, so it is regenerated
for those, and the replacement carries a cue.

So the standing question - should `self_verification` be promoted out of
`DIAGNOSTIC_GATES`? - answers itself: **no.** Enforcing it would buy at most
8.4pp of cue coverage and pay for it in regenerations, and `gens/accepted-row`
is the scarcest thing this build has (F24). Same trap as the 2026-08-31 "12.2%
clean" reading: the raw diagnostic rate is over ALL generations and measures
the discard pile, not the corpus.


### F26. The cron is late, not lossy - do NOT tune the schedule on jitter

Two scheduled fires (00:17Z, 04:17Z) produced no run, which looks like GitHub
dropping them. It is not. Measured intervals between consecutive worker starts
over 38 hours: **9.62, 4.21, 8.51, 6.49, 4.76, 4.43 h - mean 6.34 h against a
6 h period.** Nothing is being lost; fires arrive with heavy jitter and the
long gaps are paid back by the short ones.

**Confirmed live while this was being written:** the 04:17Z fire arrived at
**05:42:37Z - 85 minutes late** - and entered the queue as `pending` behind the
02:03Z dispatch, which is precisely the behaviour described above. It also
carries sha `fb611f5`, so it is the first run to execute both of tonight's
fixes, and the first place to check the F23 prediction of `claimed=36`.

That matters because the obvious "fix" is wrong. A run lasts **5.45 h**, and
the workflows share concurrency group `data-build` with
`cancel-in-progress: false`, so exactly one runs and one waits. At `*/4` the
mean interval (~4.2 h) is already below the run length, so the pending slot
self-fills and the extra fires are discarded on arrival - the schedule would be
tighter on paper and identical in practice. **No cron change made.** Tuning a
6 h period against +/-3 h of jitter is fitting noise.

### F27. A wave can now be planned on chosen templates (`c61311a`)

F24 established that `gen_irac_analysis_v4` costs 15.6 generations per accepted
row against `v1`'s 3.3, and Step 6 wants to stop buying the expensive personas.
There was no way to express that:

- `--arm` labels an A/B cell; it does **not** pin a template. `pick_variant`
  still runs unconditionally over the whole pool.
- Deleting the losing template files is **unsafe**: `task_id_for` does not hash
  `prompt_sha`, so removing a variant re-maps the draw for every row already
  pending and parks the queue as `stale_prompt` - which never re-opens, because
  re-stamping the sha re-parks it instantly.

So the mechanism has to bind **new rows only**. `prompt_registry.group_variants`
validates an operator's allowlist against the registry at plan time (a typo
fails before a single row is inserted, naming the legal ids), and
`pick_variant(..., allow=...)` **narrows without reordering** - a surviving
variant keeps its registry position, so the same (seed, sample) that drew `v1`
out of the full pool still draws `v1`. `plan_rows`/`plan_wave`/`commit_rows`
thread it through, the `wave_planned` event records it, and `--variant` is
repeatable on the CLI. Naming some task types narrows **only those**; every
other task type keeps the full paraphrase rotation.

Ten tests, incl. the pin, the no-op equivalence `plan_rows(...) ==
plan_rows(..., variants=None)`, and that an unknown id fails before any write.
Suite **3,773 / 19**.

### F28. FIXED: the shipped answer contained a SECOND deliberation (`fb611f5`)

Every `irac_analysis` template asks for a **250-450 word** answer and says the
reasoning "takes as long as it needs to take" - in the trace. The corpus did
not obey. Of 733 accepted rows whose task type owes an IRAC answer, **490 open
with 2,400-4,800 characters of first-person deliberation before the first IRAC
heading**: 65% of all answer words sat in front of the answer.

It is not a summary of the trace and not a lead-in. Against the `think` block
it shares a median **57% of content words but only 5.4% of 5-word runs** - the
model reasons a second time, in fresh wording, inside the field the student is
trained to emit. No gate can see it: `check_length_band` has `total_max`,
`think_max` and `answer_min` but **no `answer_max`** - answer length is bounded
only from below - and `check_irac_placement` only requires the headings to be
present in the answer and absent from the think, never that the answer *opens*
with them.

Distribution is bimodal, which is why a threshold is honest here: **234 rows at
exactly 0 characters of preamble, 13 between 1 and 1,200, 486 above 1,201.**
`PREAMBLE_MIN_CHARS = 1000` sits in the empty middle; no plausible value in the
gap changes the outcome.

Fixed at **assembly**, not at generation. `decontaminate.generated_rows` is the
one seam where `think` + `answer` become the shipped turn, so trimming there
costs **zero generations**; gating it would have re-run every offending row.
`answer_without_preamble` cuts from the first IRAC heading, and only for
`IRAC_ANSWER_TASK_TYPES` - a summary or a drafted notice is prose by genre, and
a line of one that happens to begin "Issue" is not a contract to act on.
`TRIMMED_MIN_CHARS = 480` is **not arbitrary**: `answer_min` is 120 tokens and
`gates._est_tokens` is `len // 4`, so 480 characters is exactly the gate floor -
the trim can never push a row under the bar it already cleared.

Measured on the live store: **488 of 733 rows trimmed, median answer 801 -> 379
words** (inside the templates' own spec), rows over 450 words **94% -> 33%**.
Safety checks before shipping: **0%** of trimmed answers back-reference the cut
text ("as noted above"); **3.3%** fall under 200 words but none under the gate
floor; **15.9%** lose a section number that appeared *only* in the preamble -
the residual risk, and the reason the packet now screens the trimmed text.
Worth roughly **125 rescued generations** as a side effect: 1,591 rows blew
`total_max`, and 250 of those had it as their only violation.

The trace is untouched, the full generation stays in the build store, each row
carries `answer_preamble_dropped`, and the dataset card discloses it under
**Answer normalisation**. Suite **3,781 / 19**.

**Verified through the real assembly chain**, same store, only the code
changed - so this is a controlled before/after, not an estimate:

| | before | after |
|---|---|---|
| `CHAIN RC` | 0 | **0** (all nine gates still PASS) |
| length p50 | 3,532 tok | **3,224 tok** (-8.7%) |
| length p90 | 5,542 tok | **5,311 tok** |
| length p99 / max | 7,039 / 8,088 | **unchanged** |
| rows | 716 | **718** |

p99 and max not moving is the right shape: the longest rows are long because
of their *source*, which the trim does not touch. The +2 rows are the rescue
effect arriving in the corpus. At the ~10,021-row ceiling a 308-token median
saving is roughly **3M tokens an epoch** that were a duplicated deliberation.

### F29. The 50-example review packet exists, and 12 rows carry an authority the source never names

The dataset card names a human read of 50 accepted examples as a ship
prerequisite - "the only legal-accuracy check in this pipeline" - and it had
never been scheduled. `data/build/out/review_packet.html` (gitignored, nothing
uploaded) is a stratified draw with a **floor of 3 per cell**, so `transition`
(5 accepted rows in total) and `statute_qa` survive a draw that a proportional
sample would have washed out. Each card shows the answer **as it will ship**
(post-trim), with the cut preamble moved into the trace rather than deleted, and
collapsible source, trace and judge, plus a verdict control.

A mechanical pre-screen orders the reading. Legal correctness needs a human, but
two failure modes do not: an answer citing a **section** or a **reported
citation** the source never mentions. **4 of those 50 rows** carry one, three of
them `transition`.

**CORRECTION.** This was first reported as **12 of 50**, and 12 was wrong. Eight
were false positives from a THIRD instance of the same bug: with `re.I`, the
source-side pattern read "Section 482 of the Code" as section `482OF`, so the
bare `482` never entered the source set and the answer's clean citation looked
unsourced. Re-run on the identical 50 rows, old screen 12, new screen 4, and the
4 are a strict subset of the 12. Transition's 3/3 is **expected by construction**, though not for the reason I
first wrote: the task is not offence-mapping (IPC 302 -> BNS 103) but
**which-enactment-governs**, and it turns on the three repeal-and-savings
provisions (BNS 358, BNSS 531, BSA 170) plus s.6 of the General Clauses Act.
Those are cited from law, never from the source judgment, so they can never be
"sourced". The packet flags them as "verify the limbs", and F30 is what came of
actually doing that.

**Three screen bugs, each of which produced a confidently wrong number.** The
lesson is not that regexes are hard; it is that this screen produced a clean,
plausible, quotable figure on all three occasions, and nothing but re-deriving
it caught the error. It is now in `src/tuned/data/review.py` behind 15 tests,
and each bug below is pinned by one:
1. One permissive pattern on both sides read "Section 29 contains" as section
   `29CON`. Now strict on the answer (the suffix must be attached to the
   number), permissive on the source (a near-miss there only *suppresses* a
   flag, so permissiveness is the safe direction).
2. Making the source pattern prefix-optional let every page number in a 40-page
   judgment count as a section mention - the screen returned **0 findings by
   construction**. A separate probe (0 sections extracted from 50 answers)
   caught it. A clean "nothing found" is worthless until the instrument is shown
   able to find something.
3. Case-insensitivity on the SUFFIX turned "Section 482 of" into `482OF`, which
   inflated the finding count 3x. Fixed by scoping the flag - the prefix is
   case-insensitive, the suffix is not - and by crediting the bare number in the
   source set alongside the suffixed form, so "Section 420 IPC" still answers a
   bare `420`.

**Now reproducible.** `data/scripts/review_packet.py --state S --out P` rebuilds
it from any store copy, read-only. The draw is deterministic in `--salt`, so
re-rendering after the corpus grows returns the SAME 50 rows and stays
comparable to the last read instead of silently becoming a fresh sample.

### F30. An accepted row states the OPPOSITE conclusion to its own answer key

Working the review packet turned up the thing the packet exists to find, and it
was findable **mechanically**, because the transition seeds carry ground truth
in `seed.answer_key_json`.

Accepted row `412b8d1c5430` (`gen_transition_v1`, offence 2021-12-27, appointed
day 2024-07-01) concludes:

> 2. The conduct of this appeal is governed by the Bharatiya Nagarik Suraksha
> Sanhita, 2023.

Its own key says `families_by_kind.procedural = "old"` and
`procedural_rule.effect = "the Code of Criminal Procedure, 1973 continues to
govern this proceeding"`. The answer's reasoning is wrong on the facts too: it
argues the appeal, **filed 25 March 2023**, "was not pending immediately before
the commencement" - but commencement is 1 July 2024, so it plainly was, and
BNSS s.531(2)(a) saves it to the CrPC. The other four accepted transition rows
are correct on all three limbs (read individually, not sampled).

**Why every gate passed it.** `check_answer_key` is PERMANENT and it ran - but
it checks *citations, not conclusions*: expected sections present, forbidden
absent, savings mentioned, both families named. This answer cites IPC 468,
BNSS 531, BSA 170 and BNS 358 exactly as required, so it passes. The docstring
is explicit that `governing_family` is "recorded, not enforced", with a sound
reason: a single global family would contradict `must_name_both_families`.

**But `families_by_kind` is never read at all** - not enforced, not even in the
gate's `detail`. That field does not have the problem the docstring describes:
it is *per limb*, and a transition conclusion states exactly those three limbs.
It is the one field that encodes the answer, and nothing consults it.

Scale, measured across every generation that PASSED `answer_key` and whose key
says `procedural = "old"`: **29 of 146 (19.9%)** put the procedural limb under
the new code in their conclusion region. Most died on other gates
(`rejected`/`format_parked`), so the corpus damage is **1 row of 718** - but the
blind spot is systematic, not a one-off. One caveat on my own instrument: row
`9fbcc98d9a4b` counts in the 29 because its heading is `Conclusion -` rather
than `**Conclusion**`, so the fallback window caught Application text; its
actual conclusion declines to decide two limbs (under-answering, not inversion).

**Not fixed tonight, deliberately.** `transition` is a finished stream (2,200 of
2,200 terminal, 5 accepted), so a new permanent gate would apply to ~0 future
rows unless the stream is replanned - and a correctness gate on a permanent
path is not a thing to write unattended at 06:00 and merge without an operator
reading it. Recorded with the evidence instead. **If transition is ever
replanned, enforce `families_by_kind` per limb first**; that is the cheapest
real legal-accuracy gate available anywhere in this pipeline, because the ground
truth is already sitting in the seed.

**The general lesson, which is the point of the packet:** a row can be
well-formed, cite every required authority, satisfy twelve gates and a judge,
and still state the opposite of the right answer. Eleven of the twelve gates
score form. This is the concrete instance that argues the human read is not
ceremony.

### F31. "Has an answer key" is exactly "can run", and no other stream has ground truth

Chasing F30 outward: **only `transition` seeds carry an answer key** - 1,250 of
61,853 seeds, all on one 22-field schema. Every other task type has **zero**
(irac_analysis, summarization, statute_qa, drafting, across both synthesis and
curated_c2). So there is no second free legal-accuracy gate hiding anywhere in
this pipeline; F30's fix opportunity is transition-only, which is what
`check_answer_key`'s docstring already implies ("the stream exists precisely
because the old/new-code answer is decidable in advance").

The two properties turn out to be the same property. `transition.py:render_cell`
writes `answer_key_json` **and** the four slots `build_slots` requires
(`scenario`, `old_section_text`, `new_section_text`, `savings_text`), which for
transition have **no fallbacks**. So a seed either is a purpose-built cell -
keyed and renderable - or it is an ordinary case chunk: unkeyed, and a
guaranteed `SlotError`. Measured, the correlation is exact: of 2,200 transition
tasks, the **2,063 unkeyed ones are precisely the 2,063 `skip:slots`**, and all
137 keyed ones actually ran (114 rejected, 17 format_parked, 5 accepted, 1
unroutable).

That is the same defect `708d455` already fixed and whose message already
records ~1,100 grid seeds left unplanned - this only supplies the mechanism and
the exact equivalence, which is worth having because **"does the seed have an
answer key" is a one-column test for "could this task ever have run"**, and it
is far cheaper than re-deriving slot renderability.

Still-useful operational fact: **1,113 of the 1,250 cells have never been
used**, and the planner now restricts transition to declared seeds, so a replan
would have 8x the material that produced the 5 accepted rows.

**Recommendation: do not replan transition yet.** Conversion among tasks that
could actually run is 5/137 = **3.6%**, and F22's judgment stands - at that rate
it is the worst possible claim on an account-level rate bucket, and the corpus
is short on *synthesis*, not transition. The rejects are dominated by
`length_band` and `irac_placement`, both of which have moved since these rows
were generated (`think_max` 3000 -> 4500; F24's genre result). Re-measure the
rate on a **10-row probe** after the genre variants land; replan only if it
clears the synthesis streams' opportunity cost.

### F32. The length band was weighing text the corpus would never hold (`gates.py`)

F28 trimmed the answer's second deliberation **at assembly** - and in doing so
left the gate and the assembler disagreeing about what a row *is*. From that
commit until this one, `check_length_band` weighed several thousand characters
of preamble that `decontaminate` would delete before the row ever reached the
training bucket, and refused rows on their size.

Measured on the live store: of 2,431 `length_band` failures, **250 name
`total>total_max` as their ONLY violation**, and **110 of those sit inside the
band once the answer is measured as it ships** (109 `irac_analysis`, 1
`statute_qa`).

**Two corrections to my own arithmetic, both downward. Neither number I first
reached was right:**

1. I said **172** before including `prompt_est`. `total` is prompt + think +
   answer; dropping the prompt inflated it 56%. The length figure is **110**.
2. I then called 110 "a 13.7% loss", which is **wrong** and the more misleading
   error. Those 110 rows had their *length verdict* changed - most of them also
   failed OTHER gates, so they were never one measurement away from the corpus.
   The number that matters is rows for which `length_band` was the **sole**
   blocker:

| | rows |
|---|---|
| generations whose only failing gate is `length_band` | 221 |
| ...that pass on today's caps alone (rejected under a superseded `think_max`) | 6 |
| ...that pass on today's caps **and** the shipped-answer measure | **41** |
| **attributable to this fix** | **35** (4.3% of the 805 accepted) |

So the honest claim is **35 rows, ~4.3%**, not 110 or 13.7%. Still worth having
for zero generations, and still a real inconsistency to close - but a third of
what I first wrote, and the third time tonight that a plausible first number
did not survive being re-derived.

The fix is to measure once, in one place. `answer_without_preamble` and its two
constants **moved from `decontaminate.py` into `gates.py`**, beside the
`_IRAC_HEADING_RE` they use - `decontaminate` already imported from `gates`, so
the reverse import would have been a cycle - and `decontaminate` re-exports the
name for callers that found it there first. `run_all` now passes
`_est_tokens(answer_without_preamble(answer, ctx.task_type)[0])`.

Why this is strictly more correct, not merely more permissive:
- `total_max` guards the **8192 training bucket**, and the row that enters that
  bucket is the trimmed one. Measuring the raw text was conservative by
  accident, never by design.
- `think_max` is untouched: the trim never touches the trace, and the 1,325
  rows failing `total>total_max,think>think_max` together stay rejected - they
  are genuinely too long.
- `answer_min` now applies to the shipped answer, which is the stricter reading
  and the right one: today a row can clear the floor on the strength of a
  preamble that gets deleted. It **cannot newly fail**, because
  `TRIMMED_MIN_CHARS` is `answer_min * 4` by construction. Verified on the
  store: of 488 trimmed accepted rows the shortest is **215 est tokens against
  a 120 floor**, and **0** would flip to failing.
- A genre without an IRAC contract (`summarization`, `drafting`) is untrimmed
  and so measured exactly as before.

**This recovers nothing already rejected.** Those rows are `rejected`, and
reopening regenerates rather than re-gates, so recovering them would pay for the
calls a second time; the value here is forward-looking - the same loss stops
recurring.

**A re-gate would be worth 41 rows for zero API calls**, and that is now a
measured number rather than a hunch: 41 generations sitting in the store pass
every gate under today's configuration, 35 of them because of this fix and 6
because `think_max` moved 3000 -> 4500 underneath them. At ~44 accepted rows/hr
that is roughly an hour of generation, recoverable without touching the rate
bucket. It needs a CLI that re-runs `run_all` over stored generations and
promotes the passes - not written tonight, because it is a new write path into
the store and the operator should see it before it runs.

Five tests, one per claim above. Suite **3,801 / 19**; card discloses it.

### F52. THE STORAGE BURN IS 41.5 GB/DAY, AND THE RAW LOG IS MOST OF IT

F48 put the burn at 14.8 GB/day and the wall at ~2026-09-03. Both were
DERIVED. Timed with a stopwatch over exactly one hour, against HF's own
`usedStorage`:

    13:24:19Z   58.24 GB
    14:24:23Z   59.97 GB     +1.73 GB in 1.00 h  =  41.5 GB/day
    headroom 40.0 GB      ->  wall ~2026-09-01 14:20Z (0.96 days)

All 1.73 GB landed on `tantan01/tuned-law-state`; no other repo moved.

#### The database is not the problem - the append-only raw log is

Per hour the worker pushes ~4 raw checkpoints (`--push-every` 900) and ~1
with the database. That is ~0.565 GB/h of DB against **~1.0 GB/h of raw**,
because `raw/gen/<day>/gen.ndjson` is APPEND-ONLY and LFS versions whole
files: at ~250 MB by mid-day, each of the four pushes stores another 250 MB
blob, and every one is kept. The file grows all day, so the hourly cost
RISES until the UTC date rolls it over and a fresh file starts near zero.
Integrated over a day that is ~20+ GB from raw alone - which is why the
derived figure, built from DB pushes, came out ~3x low.

#### Consequence: the squash is already late

`super_squash_history` reclaims the ~38 GB of baton history but takes up to
36 h to reflect in the quota. 36 h from this measurement is ~2026-09-02
02:00Z; the wall is ~2026-09-01 14:20Z. **There is no longer a schedule on
which the squash alone arrives in time**, so it should be run at the first
safe moment rather than planned for tomorrow.

#### Two mitigations that buy time, neither sufficient alone

- **Delete the six retired checkpoint repos: +6.84 GB**, no baton window
  needed, no effect on a running job (task 27). Buys ~4 hours.
- **Raise `--push-every` 900 -> 3600**: cuts raw from ~1.0 to ~0.25 GB/h, so
  ~41.5 -> ~23.5 GB/day. Buys ~0.7 days. It is a one-flag, fully reversible
  change and it TRADES DURABILITY: a run that dies loses up to an hour of
  generation instead of 15 minutes. Not taken unilaterally for that reason.

The real fix, for after the deadline: rotate the raw log hourly rather than
daily (`raw/gen/<day>/<hour>/`). Each push would then re-upload at most the
current hour's file (~40 MB) instead of the whole day's, cutting the dominant
term ~5x permanently. It changes the baton layout that `RESTORE_SUBS` and
`reconcile_raw` read, so it is a tested change, not an overnight one.

### F51. THE AUDIT SAMPLE DIES WHEN THE SECOND FAMILY DOES, NOT WHEN THE BUDGET DOES

Run `33363831595` is the first COMPLETE run to read end to end. Its judge
side accepted 393 rows and rejected none. The outcome tags split exactly:

    audit-accept            373    gate-only, never sampled (by design)
    accept                   15    actually dual-judged
    audit-accept-unjudged     5    hash-SELECTED, and shipped without a judge

So the sample rate is right - 20 of 393 is 5.09% against `--audit-sample
0.05` - but the delivered evidence is **15 judged rows, 15 accepts, 0
rejects**. A quarter of the sample leaked.

`AUDIT_UNJUDGED_DISPOSITION`'s comment is explicit that this beats the
alternative ("what it must not do is end in judge_error, which would make a
row's survival depend on whether the hash picked it"), so the leak is a
designed concession, not a bug. What was never measured is how big it gets.

#### The cause is family exhaustion, and the correlation is exact

    groq/openai/gpt-oss-20b  left=0k at 09:38:44Z   (spent 205.2k)
    groq/qwen/qwen3.6-27b    left=0k at 11:35:29Z   (spent 200.6k)

    unjudged at  09:36:22Z   <- gpt-oss on its last tokens
                 11:40:36Z                    11:51:43Z    |  every one after qwen ran out
                 12:15:25Z    |
                 12:28:32Z   /

Every generation is deepseek since the sole-generator ruling, so
family_separation excludes deepseek from judging its own row. That leaves
exactly three free families - qwen (groq), gpt-oss (groq), gemma (cerebras) -
and a dual judgement needs TWO OF THEM. After 11:35Z only gemma remained, so
the second slot could not be filled **at any budget**: cerebras still had
784k of 1,000k unspent and spent only 99.8k all run.

**The binding constraint is not tokens, it is distinct free families.** Both
survivors of a pairing must differ, so every dual judgement burns at least
one groq call, and groq's per-model ~200k/day is the ceiling on the whole
audit instrument. For the last hour of a 5h15m run the instrument was dead.

#### SHARPENED 2026-08-31 13:40Z: it is not a leak, it is most of the day

The next run, `33381288057`, opened its FIRST batch with both groq models
already at zero:

    bai/deepseek-v4-flash         spent=29780.3k  left=-
    cerebras/gemma-4-31b          spent=  235.6k  left=764k
    groq/qwen/qwen3.6-27b         spent=  207.2k  left=0k
    groq/openai/gpt-oss-20b       spent=  213.2k  left=0k
    mistral/mistral-large-latest  spent=    0.0k  left=5000k

The ledger carries ACROSS runs, because the bucket is a daily one (reset
boundary assumed UTC midnight - not verified). So this entire 5h15m run has
no second family and every hash-sampled row in it ships unjudged. Confirmed
live: `audit-accept-unjudged` appears in its judge stream within the hour,
against 09:36Z for the previous run.

The day's shape, then:

    00:00Z ---- both groq models funded ---- 09:38Z (gpt-oss) 11:35Z (qwen)
      |<---- ~1.5 runs WITH dual judging ---->|<-- ~3.1 runs with NONE -->|

**~12.4 of every 24 hours, and roughly 3 of every 4.6 runs, produce zero
dual-judged rows.** The daily total (~15-40) still matches the config's own
"~35-40 rows/UTC-day" estimate, so the VOLUME was known. What was not is the
DISTRIBUTION, and that is what breaks the warrant:

> "the sample's accept rate is the quality evidence for the whole batch"

holds only if the sample is representative. A sha256 hash-sample is. A hash
sample intersected with "was generated before ~11:35Z" is NOT - and because
claiming is FIFO by rowid per stream (F43), what the fleet generates early in
a UTC day is systematically different work from what it generates late.
The evidence covers the front of the queue and nothing else.

#### The idle 5,000k of mistral is NOT the fix - I checked before proposing it

`mistral/mistral-large-latest` reports `spent=0.0k left=5000k` in every
single batch line, which looks exactly like free judge capacity going to
waste. It is not. The config gives it `roles: [tiebreak]` and `rpm: 2`, and
says why at length: with the judges' families excluded, mistral is the ONE
family left for a contested row, so promoting it to judge would let it be
spent in slot A or B and then be excluded from the tiebreak seat it exists to
fill. At rpm 2 it could serve ~630 calls in a 315-minute run - ample for
tiebreaks, nowhere near 393 rows. The config refuted this before I did.

#### The arithmetic for whoever decides, so the call is cheap

Measured on the completed run, not modelled:

    groq spend    qwen +83.2k, gpt-oss +61.7k  = 144.9k for 15 dual judgements
    per judgement 9.7k groq  +  6.7k cerebras  (gemma already carries one slot)
    daily groq    ~400k (200k per model)       -> ~41 judgements/day
    daily accepts ~1,794                       -> SUSTAINABLE SAMPLE = 2.3%

`--audit-sample 0.05` is over-subscribed ~2.2x against the groq budget. That
is the whole mechanism: the sample spends at 5% until the bucket is dry a
third of the way into the day, then delivers nothing.

Three options, priced:

- **(a) Accept it.** Free. But then say so in the model card: the accept-rate
  evidence describes rows generated before ~11:35Z, not the corpus.
- **(b) Lower `--audit-sample` to ~0.02.** One flag in data-worker.yml, no
  prompt sha, instantly reversible. The budget then spans the UTC day and the
  sample becomes REPRESENTATIVE. It buys no extra evidence - still ~40
  rows/day - it fixes the distribution. Recommended if anything is done.
- **(c) Add a free judge family.** Nothing is available: bai is excluded by
  family_separation (it generates every row), groq is the exhausted one,
  cerebras already carries a slot, mistral is the reserved tiebreak. This
  needs a NEW free provider, not a re-ordering - I checked the ordering
  hypothesis and it is already optimal.

Reordering the judge list does NOT help, which is worth stating because it
looks like it should: gemma is second in `routing.judge` and already takes a
slot on every judgement, so there is no second groq call to eliminate.

#### Not acted on

Judge routing is a quality decision and the free-fleet ruling is a hard
constraint, so this is an operator call, not an agent one. Recorded because
it changes what the evidence is WORTH: "the sample's accept rate is the
quality evidence for the whole batch" currently rests on 15 rows per run, and
the shortfall is not random - it is concentrated in the back half of every
run, after groq runs dry.

### F50. WHAT A COMPLETE RUN ACTUALLY PRODUCES (AND A 58-MINUTE GAP I FIRST MISREAD)

Run `33363831595` (sha `0227bdd`, no ceiling guard) is the first run to go
the distance instead of being cancelled or evicted. Its readout:

    accepted        946 -> 1,339    (+393)
    pending       4,935 -> 4,072    (-863)
    format_parked   451 ->   886    (+435)
    rejected      2,620 -> 2,635     (+15)

    56 gen batches, 2,016 claimed, 2,007 gen-ok, 9 err (0.45%)
    deepseek tokens 14,150.2k -> 27,391.4k  = 13,241k spent

**393 accepted from 2,007 generations is 19.6%.** F37 measured 19.6% clean
on a different instrument (this run's own gate_result rows, before any of
this log existed). Two independent readings agreeing exactly
is the strongest confirmation the yield model has had. (Count the batch
lines with care: `_finish` tails gen.log into its own report, so the last 20
appear TWICE in the archived log and a naive grep reads 76 batches, not 56.)

#### WATCH ITEM: err ran 4x higher in the next run, and it is not our doing

Run `33381288057` opened at 9 err in its first 14 batches (504 claimed,
**1.8%**) against the completed run's 9 in 2,016 (**0.45%**). Under the old
rate the expectation was 2.3, so this is not noise (Poisson p ~ 0.0009).

What it is NOT: a regression from tonight's commits. `git diff 0227bdd..c84fb1f`
over `src/tuned/data/` and `data/configs/` touches ONLY `push.py` and
`shape.py` - not `generate.py`, not `tasks.py`, not routing. The generation
path is identical in both runs, and both served curated_c2 (the `dcb1d8d`
hand throttle postdates `0227bdd` and `be25afd` reverted it). The difference
is which TASKS are being worked: the queue is FIFO by rowid per stream, so
the fleet has drained into a later slice with different seeds.

Not diagnosable from outside the store, for the reason above - and low
impact: `gen-ok` holds at 34-36 of 36 and `gated-out` at 25-29, so
throughput is untouched. **Threshold: if err exceeds ~5% sustained, or
`gen-ok` drops below ~32/36, read the `worker_task_error` events in the
store.** Below that it is not worth a 565 MB download.

#### `err` in the batch line is NOT a failure rate

It reads like one and it is not. `generate.py:491` increments `stats.errors`
on `result.error is not None OR result.skipped is not None`, so a task that
was SKIPPED without a call - `skip:slots` is the big one, it killed 2,063
transition tasks - lands in the same counter as a provider failure. Only the
other site (`generate.py:2267`, a raised exception) is attributable, and it
persists to `store.log_event("worker_task_error", ...)` inside the 565 MB
database, which is not on any cheap baton surface.

Nor does the raw log help: 368 rows of `raw/gen` from the current run carry
ZERO `error` fields, because only answered calls are written there.

Practical effect: the previous run's 0.45% and a later run's 1.4% are not
comparable as failure rates, and neither is worth chasing from outside the
store. Read `worker_task_error` events if the question ever matters.

#### CORRECTED: the 58-minute gap was the fence, not a runner queue

The first reading of this was WRONG and is left here because the wrong
version was committed. Run `33363831595` was created 06:20:57Z and its first
log line is 07:19:21Z, which I attributed to ~58 minutes waiting for a hosted
runner. It was not. The job timings say otherwise:

    33349462956   job 02:03:25Z -> 07:18:57Z
    33363831595   job 07:18:59Z  (2 s after its predecessor)  -> 12:34:34Z
    33381288057   job 12:34:43Z  (9 s after its predecessor)

**Job start follows fence release within seconds, every time.** The 58
minutes was `concurrency: data-build` holding the run behind `33349462956`,
which is the fence doing precisely its job. There is no hidden runner lag,
and `gh run view --json jobs` exposes the real `startedAt` - it is only the
RUN-level `startedAt` that lies by echoing `createdAt`.

What this corrects downstream:

- **The cycle is ~315.6 min, not ~374.** Two consecutive runs measured
  315.5 and 315.6 minutes of job time with ~2-9 s of dead air between them.
  The fleet chains back to back and the cadence comment's ~322-327 min
  estimate is if anything pessimistic.
- **Throughput is ~1,794 accepted rows/day, not ~1,500** (393 per 315.6 min).
- The 4,072 pending tasks are ~4.7 runs = **~25 hours**, not ~29.

#### Displacement is real, and it was observed - but I caused it

`33361492672` is the `17 */4` cron delivered 05:42:37Z (~85 min late, which
IS a real and separate finding). It never ran: it was cancelled 06:20:59Z,
two seconds after my own `workflow_dispatch` created `33363831595`. A second
pending trigger replaces the one already waiting, exactly as the workflow
comment warns - only here the operator dispatch displaced the cron, not the
other way round.

With a 315.6 min run against a 240 min cron the pending slot is always
occupied when the next cron fires, so a cron run is displaced roughly once
per cycle. That is harmless when the displaced run is an identical cron, and
it is why the machine stays saturated. `*/4` stays: denser makes displacement
more likely, not less, and the only thing worth protecting is an operator
dispatch - which should be issued when no cron is pending, not defended by
changing the period.

#### Throughput, and what it means for the queue

At +393 accepted per run and ~315.6 min per cycle, the fleet banks **~1,794
accepted rows/day**. The 4,072 pending tasks are therefore ~4.7 runs, or
about **25 hours**, of work - which is when the queue runs dry, not when the
corpus is done.

### F49. A 5h16m JOB PUBLISHES NOTHING UNTIL IT ENDS, INCLUDING THE GUARD'S OWN LINE

Verifying F35's ceiling guard meant reading one line that the worker prints in
its first minute. There is no way to read it before the job ends:

    gh api repos/Anant-T/Tuned/actions/jobs/99412108279/logs
      -> BlobNotFound / HTTP 404      (job in_progress for 5 h at the time)

GitHub writes the log archive when the job COMPLETES; nothing is served
before that. The Actions job summary is no better - `GITHUB_STEP_SUMMARY`
renders when the STEP finishes, and `run_worker` is a single step that runs
the whole 5h16m deadline. So `_report`, which exists precisely to make a line
visible "without opening the run", has the same blind spot for anything
decided before the end.

The cost landed on this exact question. `be25afd` (the guard) is an ancestor
of `c84fb1f` but NOT of `0227bdd`, so run `33381288057` is the first job that
carries it - and its single most consequential decision, which streams it may
serve, is unreadable for the 5h16m it takes to make it.

#### Fixed for every run after this one

`_run_log(root, line)` prints the line AND appends it to
`logs/<GITHUB_RUN_ID>/worker.log`. That path is not decorative:
`stage_bundle` copies `logs/` on EVERY push including the DB-less ones, so
the line rides the fast `--push-every` cadence (900 s default) rather than
the hourly database one. **A worker-level decision becomes readable off the
baton ~15 minutes in instead of ~5h16m.**

Four lines routed through it, all of them decisions rather than progress:

- both `ceiling guard:` branches (which streams this run serves);
- `no claimable work: ... - trying gen_unroutable, ...` (the queue was found
  dead and the recovery was reached for);
- `re-open found work: N claimable - continuing`.

Deliberately NOT the `QUEUE EMPTY` report. That one already goes through
`_report` into the job summary, and a run that ends on it ends in minutes -
so its log IS published promptly. There is no blind spot there to fix.

Two tests, written before the change and watched fail:
`test_the_ceiling_guards_decision_is_left_where_the_baton_will_carry_it` and
`test_the_reopen_decision_is_left_on_the_baton_too`. The writer is
best-effort (`OSError` is caught and reported) - a run must not die because
it could not narrate itself.

**It does not help task 21.** Run `33381288057` was already queued at
`c84fb1f` before this landed, so that verification still waits on the
archived log. The next cron run is the first to write worker.log.

### F48. PRIVATE STORAGE RUNS OUT ON ~2026-09-03, AND THE FIX NEEDS 36 HOURS' NOTICE

The baton is a PRIVATE Hugging Face dataset repo on a FREE account, and every
checkpoint uploads the ~565 MB VACUUMed database as a brand-new LFS blob. Git
keeps every one of them. Measured just now, against the live repo:

    tantan01/tuned-law-state   used_storage   38.83 GB   (82 commits)
    first commit  2026-08-28 20:25Z   last  2026-08-31 11:20Z   span 2.62 days
    growth                             14.8 GB/day

**The quota is ACCOUNT-WIDE, not per-repo,** and ten private model repos hold
the rest of it - so the headroom is smaller than the baton alone suggests:

    baton (dataset)                                       38.83 GB
    tuned-law-v1-qwen8b-ckpt-ddp        (the live lane)    9.32 GB
    tuned-law-v1-qwen-ckpt-manual                          1.56 GB
    tuned-law-v1-qwen8b-ckpt-ddp-rslora32                  1.59 GB
    tuned-law-v1-qwen-ckpt-ddp                             1.30 GB
    tuned-law-v1-qwen8b-ckpt-ddp-alpha64                   1.07 GB
    tuned-law-v1-qwen-ckpt                                 0.79 GB
    tuned-law-v1-qwen8b-ckpt-ddp-rslora                    0.54 GB
    three more at 0.00 GB                                  0.00 GB
    ------------------------------------------------------------
    TOTAL PRIVATE                                         55.00 GB
    free-account PRIVATE quota                           100    GB   (HF doc)
    headroom                                              45    GB  =  3.0 days
    wall                                             ~2026-09-03

The account is `isPro: false, canPay: false`, so there is no overage to spill
into: past the quota the Hub refuses uploads. Every push then fails, the worker
prints `CHECKPOINTS FAILING`, `final_push_ok` goes false and the job exits 1 -
and **each subsequent run loses up to a full 5h15m of generation**, because the
answers only exist on the runner. The queue holds ~27 h of work and the top-up
wave extends it, so the build IS still running on 09-04. This is a hard stop,
not a warning.

The code already knew half of it - `run_worker`'s two-cadence comment costs the
DB push at "~56 GB/day at 15-minute pushes, against ~29 with the database
hourly", which is why the DB cadence is hourly. Nobody carried that forward to
"and the account holds 100 GB".

#### The fix, and the reason it cannot wait for the wall

`HfApi.super_squash_history(repo_id, repo_type="dataset")` collapses the history
to one commit and drops the old LFS blobs. The working tree is only ~700 MB, so
one squash should take 38.8 GB back to ~1 GB - about 6.5 more days each time.
It is a RECURRING maintenance action, roughly weekly, not a one-off.

Two constraints make the timing tight:

- **The quota takes up to 36 hours to reflect a squash** (HF's own doc). So the
  squash has to land by ~2026-09-02 to clear before the 09-04 wall.
- **It is irreversible, and it breaks a live holder.** `Bundle.push` passes
  `parent_commit=self.head`; after a squash that parent no longer exists, so
  every push from a running worker 412s and the run loses everything since its
  last successful checkpoint.

**The safe moment is end-of-job, immediately after the final push.** That is the
last baton interaction a run makes (`run_worker` returns straight after
`_report`), and the `data-build` concurrency group guarantees the next run has
not pulled yet. Doing it there is a small change to the supervisor; doing it by
hand means cancelling the queued run first, squashing, then dispatching.

**Not done tonight.** It is a destructive, irreversible operation on the single
source of truth for the whole build, it is four days out rather than four hours,
and the choice between "automate it in the supervisor" and "run it by hand
between dispatches" is exactly the kind of call that should be made awake.

**A second lever, operator-only: 6.85 GB sits in retired experiment repos.**
`-ddp-rslora` (0.54), `-ddp-rslora32` (1.59) and `-ddp-alpha64` (1.07) are the
three adapter-scale A/B arms whose questions memory records as CLOSED and whose
code was retired in `15c3eb9` / `2b3ac29` / `c3c3651`; `qwen-ckpt-manual` (1.56),
`qwen-ckpt-ddp` (1.30) and `qwen-ckpt` (0.79) predate the 2026-08-08 strip to the
Qwen3-8B DDP lane. Deleting them buys ~0.46 days - useful margin around a squash,
not a substitute for one, and it is irreversible, so it is the operator's call
and no part of it was done here. `tuned-law-v1-qwen8b-ckpt-ddp` (9.32 GB) is the
LIVE lane and must not be touched.

**The cheap partial mitigation, if the wall ever gets close:** raise
`--db-every`. The database is nearly all of the growth, and the cost of a
DB-less checkpoint is bounded and already documented - up to `--db-every` of
already-paid answers return to `pending` on a crash.

### F47. PLAN DECISION 1 (ALIGN `default_profile`) - TRIED, MEASURED, REVERTED

The plan's Decision 1 was "align `assembly.default_profile` to what CI grades":
the config says `v1.1-full` while `actions_worker.PROFILE` is `v1.0-MVP`. It was
recorded and never executed, so it was executed tonight, test-first. The test
failed for the right reason (`'v1.1-full' == 'v1.0-MVP'`), the one-line config
change turned it green - **and the full suite went from 3,828 passing to 33
failures.**

**The failures are the point.** 33 gate tests build their fixture corpus at
60 / 16 / 24 and grade it with no `--profile`, so the fixture's proportions and
the default profile are ONE decision expressed in two places. Flip the default
and every one of them reports `mix FAIL curated 16.0% (target 28%),
grounded_synthesis 60.0% (target 30%), replay 24.0% (target 42%)` - the gate
working correctly against targets the fixture was never built for.

Reverted, both sides, rather than rewriting 33 gate fixtures unattended.

**And the footgun it was meant to close is smaller than it looked.** Every
`stats` report names its profile on the third line (`- profile: v1.0-MVP`), and
the supervisor passes `--profile` explicitly to both `shape` and `stats`, so a
misgraded ad-hoc reading announces itself. That is not nothing, but it does not
buy 33 fixture edits made while nobody is awake to review them.

**The two-step for whoever takes it awake**, in this order:

1. Make the gate fixtures derive their proportions from
   `cfg.assembly.targets()` instead of hardcoding 60 / 16 / 24. They then test
   what they mean - "a corpus at the graded profile's proportions passes mix" -
   and stop encoding a profile choice by accident.
2. Then flip `default_profile` to `v1.0-MVP`, with the cross-file test that
   pins it to `actions_worker.PROFILE` (the same drift guard
   `test_the_supervisor_default_sample_is_the_judges_own_constant` already
   applies to `DEFAULT_AUDIT_SAMPLE`).

Step 1 has a trap worth naming before anyone starts it: a fixture that builds
its corpus FROM `cfg.assembly.targets()` and is then graded AGAINST those same
targets is tautological - it passes whatever the lookup returns and can no
longer catch a targets bug. The hardcoded 60 / 16 / 24 is at least a concrete
second opinion. So step 1 is "derive the fixture, keep an independent assertion
on the numbers", not a blind substitution, and it is NOT obviously worth doing
on its own merits.

### F46. THE VARIANT EFFECT, MEASURED PER TASK: v1 64%, v4 28%

F36 and F37 measured the variant effect per GENERATION. Sizing a wave needs it
per TASK, because a task may spend up to three attempts. Read off the whole
synthesis irac population (arm IS NULL), forward-valid denominator - the same
`accepted / (accepted + rejected + format_parked)` F41 settled on:

    prompt_id                 acc   rej  park   yield/task   pending
    gen_irac_analysis_v1      146    46    35      64.3%        350
    gen_irac_analysis_v3      105    24    58      56.1%        322
    gen_irac_analysis_v2       80    47    64      41.9%        361
    gen_irac_analysis_v4       60    50   108      27.5%        355
    ALL irac                  391   167   265      47.5%

n = 187-227 per arm, and the ordering is F36's and F37's exactly - v1 > v3 > v2
> v4 - now on a third independent instrument. **v1 is 2.3x v4 per task.**

**This confirms F41's sizing rather than moving it.** F41 costed the top-up two
ways: ~1,279 tasks at the pooled 38.6%, or ~780 on v1+v3. The v1/v3 mean here is
~60.5%, and 780 x 0.605 = 472 accepted against the 494 the shortfall asks for.
The two numbers were derived from different instruments and they agree.

#### The unclaimed prize in the queue, sized - and NOT taken

716 of the pending irac tasks are on the two worst templates (v2 361, v4 355).
Worked as they stand they return `361 x 0.419 + 355 x 0.275 = ~249` accepted.
The same 716 tasks on v1/v3 would return `~431`. **The swap is worth ~182
accepted synthesis rows - about 37% of the whole shortfall - for the same fleet
time**, and slightly less token spend, since the better templates retry less.

It is not taken tonight, and the reason is not the one F37 gave. **F37 said
there is "no safe window" because a worker holds the baton continuously. That is
now wrong**: `--phase plan` established the pattern - pull, reconcile, wait out
the leases, refuse if a host is still live, act, push - inside the `data-build`
concurrency group. A park phase could be exactly as safe. The real reasons are:

- **No such command exists.** `tuned.data.tasks` plans and re-opens; there is no
  cancel/park path, so this is new code plus a new CI phase.
- **It destroys queued work**, and it needs a state to destroy it INTO: a park
  that is not in `TERMINALLY_DEAD` still occupies queue capacity against
  `--plan-n`, and none of the three dead states honestly means "the operator
  retired this template".
- **It is an optimisation, not a blocker.** The top-up alone closes the
  shortfall; this would close a third of it again, sooner. Worth doing awake,
  not asleep.

### F45. `--plan-n` IS MEASURED AGAINST THE LIVE QUEUE, AND THE OBVIOUS READING OVER-PLANS BY 2.3x

The top-up is one dispatch, so it was worth rehearsing rather than typing. Run
on a scratch copy of the swept working store, asking for the ~780-task wave the
way the docs read - "`--n` is a target for the whole stream, not an increment",
stream total 4,970, so `--n 5750`:

    stream=synthesis target=5750 variants=gen_irac_analysis_v1,gen_irac_analysis_v3,gen_summarization_v2
    planned 1805  collided 0
      irac_analysis 1354   summarization 451

**1,805 tasks, not 780.** The target is not measured against the stream. It is
measured against `_existing_in_queue` (`tasks.py:284`), which counts rows that
can still BECOME a dataset row - and it filters two ways at once:

    synthesis, every arm                        4,970
    synthesis, arm IS NULL                      4,570   (400 rows sit in an A/B arm)
    minus TERMINALLY_DEAD                        -625   (rejected 364, stale_prompt 244,
                                                         input_ineligible 17)
    = the number --n is compared against        3,945
    5750 - 3945                                 1,805   <- what it actually planned

Both filters bite. `arm IS NULL` is deliberate (an armed wave is a separate
queue), and TERMINALLY_DEAD is deliberate (a wave that lost rows must be able to
replace them). Together they mean **the stream total is never the right input**,
and here the naive reading over-plans by 2.3x.

#### The command, with the number read at dispatch time

    n = <live> + 780,  where <live> is, on the BATON, at the moment of dispatch:

    SELECT COUNT(*) FROM task
     WHERE stream = 'synthesis' AND arm IS NULL
       AND state NOT IN ('rejected', 'stale_prompt', 'input_ineligible');

    data-plan:  stream = synthesis
                n      = <live + 780>
                mix    = irac_analysis=0.75,summarization=0.25
                variants = gen_irac_analysis_v1,gen_irac_analysis_v3,gen_summarization_v2

Do NOT reuse 4,725 (3,945 + 780): that live count was read off a scratch copy
and the queue moves every run.

#### What the rehearsal confirmed working

- **The allowlist binds exactly.** All 1,805 new rows drew from the three named
  templates and nothing else - v1 691 / v3 663 / summarization_v2 451 - and the
  irac split is the sha-modulo over a two-element pool, not a preference.
- **The mix binds exactly.** 1,354 / 451 is 0.75 / 0.25 to the row.
- **All new rows are `pending`, `sample_ix 0`** - fresh seeds, no re-draw of an
  existing (seed, task_type) pair, so nothing collides (`collided 0`).
- **The dispatch path carries it.** `data-plan.yml` exposes a `variants` input
  and `actions_worker.py:1199-1206` splits it on commas into repeated
  `--variant`, so the comma-separated string works from the browser form.

### F44. THE 19% SYNTHESIS OVERSTATEMENT NEVER REACHES THE SHIPPING PATH

F39b found `generated_counts` multiplying a PRE-verify `accepted_count()` by a
retention measured on rows that had already passed verify, and task 23 proposed
either re-fitting after the teacher purge or teaching the counter to skip the
rows the cut will take. Measured before building either: **neither is needed.**

**The chain already orders it correctly.** `assemble_argvs` builds
`verify --require-generator --require-current-prompt --state accepted` ->
`shape` -> `decontaminate` -> `dedupe` -> `split` -> `assemble` -> `stats`, run
as sequential subprocesses over one state DB. `verify` writes the demotion back
(`verify.py:377`, `store.set_task_state(task_id, off_teacher, ...)`) - it is not
a stream filter - so by the time `shape` calls `accepted_count()` the 84 rows
are already out of `accepted`. Production sizing reads a post-demotion store and
needs no correction.

**The exposure is one command, outside the chain.** Only an ad-hoc
`shape --headroom` / `plan` against a store no assembly has swept sees the
inflated figure. That is exactly how it was found tonight, and it is why the
number in F41 is right: F41 sized off the working copy AFTER a local chain run
(accepted synthesis 443, `off_teacher` 84), not off the raw baton.

**And it cannot reach the ceiling guard.** `ceiling_state` takes
`effective[CURATED_BUCKET]` only, and the curated bucket has no teacher cut -
`off_teacher` is 84 rows, all synthesis. The irreversible side of the band is
untouched by this.

**Live confirmation, 07:24Z baton snapshot vs the swept working copy:**

    baton (no assembly has ever swept it)   accepted synthesis 409   off_teacher   0
    working copy, after a local chain run   accepted synthesis 443   off_teacher  84

So `off_teacher` is still 0 on the live store, and will stay 0 until the first
assemble dispatch - at which point it becomes 84 permanently and the two
denominators converge. The set is closed: every generation since the 2026-08-28
sole-generator ruling is deepseek, which is in the pool.

**Not built, deliberately.** A `generated_counts` that re-derives verify's
teacher predicate would duplicate `latest_generations` + `teacher_of` in a
second place, to track a fixed 84-row quantity that the shipping path already
handles and that self-clears on the next assemble. That duplication is the F24
drift failure mode, offered as a fix. Recorded in the retention comment instead.

### F43. THE TOP-UP CAN WAIT: THE CLAIM IS FIFO, SO A WAVE PLANNED NOW IS WORKED LAST

F41 sized the gap at ~780 tasks and F42 put it on the critical path, which
makes "plan it tonight" tempting. It is the wrong move, on two measurements.

**Claim order is strict insertion order.** `Store.claim_tasks`
(`store.py:905-975`) selects `... WHERE (state = ? OR (state = ? AND lease
expired)) ORDER BY rowid LIMIT ?`. There is no priority column, no stream
weighting and no shuffle - the queue is FIFO by rowid. `task_id` is a TEXT
hash, so it does not alias rowid, and rowid is therefore plain insertion
order. Two consequences:

- Tasks planned tonight sit BEHIND every pending task IN THEIR OWN STREAM -
  3,162 of them for synthesis. **The queue is one FIFO per stream, not one
  FIFO**: `generate.py:2228` runs `for stream in streams: claim_tasks(...,
  stream=stream)`, giving each `--stream` a guaranteed `n_workers` slice per
  pass before a top-up walks them in order for the remainder. So the 1,592
  pending curated_c2 rows are NOT in front of a new synthesis task - but all
  3,162 pending synthesis rows are, and the queue as a whole is ~27 h of
  fleet work.

  Measured, because the single-FIFO reading is tempting and wrong in the
  other direction too: ordered by rowid, the first 3,162 pending rows are
  synthesis and every curated_c2 row is behind them. Under one shared FIFO
  curated_c2 would never be claimed at all until synthesis drained - which
  would have made task 21 look like a guard failure when it is nothing of
  the kind. Per-stream claiming is what makes "serving every stream" mean
  what it says.
- They also sit behind the RE-OPENED rows. A re-open is an UPDATE, not an
  insert, so a `gen_unroutable` row keeps its original low rowid and jumps
  ahead of anything planned later. `REOPEN_ON_EMPTY` therefore feeds the fleet
  before a fresh wave ever gets claimed.

**Planning now moves no task one place forward in the queue.** It only adds
rows to the tail.

**And the dispatch is not free.** `data-plan` shares the `data-build`
concurrency group, so dispatching it cancels the QUEUED worker run (the
correction recorded under F37 - it evicts the queued worker, not itself). The
cron is 4-hourly and a run takes ~5h15m, so the group always holds one waiting
run. Evict it and the fleet idles from the plan run's end until the next cron
fire: up to ~4 hours of lost generation, spent to reorder nothing.

**So the trigger for task 20 is queue depth, not the clock.** Plan when pending
synthesis is under roughly one fleet-day (~1,000 tasks at today's rate), or
fold a plan step into `data-worker.yml` so it costs no separate concurrency
slot. Waiting carries no idle-fleet risk: ~27 h of queue with `REOPEN_ON_EMPTY`
behind it cannot run dry before the next attended window.

### F42. THE SHIPPING GATE IS RED ON THREE COUNTS AND THEY ARE ALL ONE CAUSE

`stats` returns 1 on RED and the chain breaks there, so a RED corpus means
`push.py` never runs and nothing reaches the hub. It has been carried as a
standing risk ("mix gate likely RED at v1.0-MVP") without anyone reading the
report. Run on the fully-armed corpus (6,592 rows, teacher cut + citation half
armed, pools shipped WHOLE):

    chain        PASS   custody complete
    length       PASS   p50 2617 / p90 5655 / p99 7175, max 8096 of 8192
    mix          FAIL   replay 64.9% (target 42), grounded_synthesis 5.7% (30),
                        curated 29.4% (28) - off by +22.94pp and -24.38pp
    trace        FAIL   63.3% carry reasoning traces, floor 80%
    empty_think  FAIL   36.7% byte-exact empty think, window [18%, 20%]
    dup          PASS   0.0%
    markup       PASS   no '<|' in any row
    license      PASS   Apache-2.0 5588, MIT 315, ODC-BY 273, CC-BY-4.0 252, CC0 161
    cross_code   PASS   0 rows name BNS/BNSS/BSA with pre-transition provenance

**The three failures are one fact wearing three hats.** The corpus is
replay-dominated because the pools shipped whole, and `replay/nothink` is 1,200
rows of the 1,200-row binding pool - so replay floods the mix (+22.94pp), those
rows carry no trace (63.3% against an 80% floor), and they are byte-exact empty
think (36.7% against a 20% ceiling). One cause, three gates. `assemble_argvs`
says exactly this in its docstring; this is the reading that confirms it.

**What matters is what is NOT red.** Every gate that is independent of corpus
SIZING passes: custody, length, duplicates, markup, licence, cross-code. There
is no second blocker hiding behind the expected one - which is the only thing
this run could have told us that the docstring did not.

So the shipping path is gated on exactly one thing: getting the corpus inside
the band so `shape` runs. `shape` trims `replay/nothink` against
`DEFAULT_EMPTY_TARGET = 0.19`, which is the midpoint of the empty_think
window, and solves the mix simultaneously - the three gates resolve together or
not at all. That makes F41's ~780-task top-up the whole of the remaining
shipping work, and F38's retention re-fit load-bearing for it: `shape` aims at
these targets THROUGH the retention table, so a table that was 16% optimistic
on curated was aiming the shaper at the wrong point.

**One known non-blocker, re-confirmed live.** `markup` PASSes while all 575
aalap rows carry Llama-2 `<s> [INST] <<SYS>>` markup, because the gate only
tests for the ChatML `<|` prefix. Already recorded; still true; still not a
gate failure.

### F41. THE QUEUE AS PLANNED DOES NOT REACH THE BAND, AND THE TOP-UP IS ~780 TASKS

The ceiling guard stops the corpus becoming UNASSEMBLABLE. It says nothing
about whether the corpus becomes ASSEMBLABLE, and those are different
questions: below the band is recoverable (add synthesis), above the ceiling is
not. Having fixed the second, here is the first, measured.

**The band's floor rises with curated, and slower than 1:1.** Read off
`synthesis_band` on the live pools at v1.0-MVP:

    curated eff    synthesis band (effective)    midpoint ratio
        398          600 ..  1050  (window 450)      2.07x
        600          900 ..  1375  (window 475)      1.90x
        800         1175 ..  1675  (window 500)      1.78x
       1000         1475 ..  2000  (window 525)      1.74x
       1242         1825 ..  2375  (window 550)      1.69x
       2050         3000 ..  3200  (window 200)      1.51x

The corpus needs roughly **1.5-2.1 effective synthesis rows per effective
curated row**, and the window WIDENS as both grow - so the target is easier to
hit later, not harder, right up until it collapses to 200 at the ceiling.

**The queue lands at 1.13, not 1.7.** Projecting the pending queue at measured
per-task yields:

    synthesis   443 accepted + 3,162 pending x 38.6%  = 1,664 accepted = 1,407 eff
    curated_c2  487 accepted + 1,592 pending x 64.8%  = 1,519 accepted = 1,241 eff
                                                         ratio 1.13, band wants 1.69

So at drain the corpus sits **418 effective rows BELOW the floor** - still
outside, still on the low-synthesis side, exactly where F35 found it and for
the same reason.

**Watch the yield denominator - it is the same trap as F39b.** Synthesis's
all-terminal yield is 24.8%, and using it oversizes the fix by 3x. 635 of its
terminal tasks are not merit outcomes at all: `stale_prompt` 502 (template
re-stamps), `off_teacher` 84 (the retired-provider purge), `input_ineligible`
42. None of those recur for a NEW task. The forward-valid figure is bounded:

    merit only      accepted/(accepted+rejected)              50.7%
    + gate parks    accepted/(acc+rej+format_parked)          38.6%   <- use this
    all terminal    accepted/everything                       24.8%   <- do NOT

38.6% is the honest one: `format_parked` is a real gate failure, and while
`REOPEN_ON_EMPTY` re-opens those rows they re-park unless a gate threshold
moves.

#### The fix, sized

    shortfall            418 effective
                       = 494 accepted synthesis
                       = ~1,279 more synthesis tasks at the pooled 38.6%
                       = ~780 tasks planned on v1+v3 only (F37: 3.11 vs 5.10
                         generations per clean row, a 1.64x)

**~780 tasks.** That is the whole gap, and F37's variant lever pays for a third
of it by itself. This is what task 20 should plan, and the two findings
compose: the wave that fixes the mix is the same wave that fixes the yield.

#### And it settles the throttle argument with an independent number

Corpus size at v1.0-MVP for each option, from `plan` on the live pools:

    stop curated now (the hand throttle)   curated  398 + synth 1044 ->  3,326 rows
    drain the queue + the top-up           curated 1242 + synth 1825 ->  6,063 rows
    ...at the top of the band              curated 1242 + synth 2375 ->  7,421 rows
    curated at the ceiling                 curated 2050 + synth 3100 -> 10,021 rows

The throttle would have capped the corpus at **3,326 rows against 6,063** for
the same fleet - a third confirmation, from a direction F35 did not use, that
throttling curated was the wrong instrument. The right one was always to plan
more synthesis, because synthesis is the side that can still be added.

**The real ceiling is the pools, not the fleet.** Even at the irreversibility
ceiling the corpus tops out near 10,021 rows, against a dataset spec of
15-20k. The binding pool is `replay/nothink` at 1,200. So the last lever on
corpus SIZE is not generation at all - it is rebuilding that pool larger,
which is exactly the remedy `--headroom`'s refusal message names.

### F40. THE case_id CHANNEL IS DELETING 9.4% OF THE CORPUS FOR CO-CITATION, NOT CONTAMINATION

This started as "filter the contaminated seeds at plan time" (task 22) and
ended somewhere else. The premise was right and the fix was wrong.

**Step 1 - the loss is seed-side and deterministic.** Of the 88 `case_id:iltur`
drops on generated rows, the matched citation is in the SEED's materials in
100% of cases (96% seed-only, 4% seed and answer). Not one is
model-introduced. Three independent counts agree, which is what a
seed-deterministic loss looks like:

    9.6%  of the seed pool (61,853) cites an IL-TUR identifier
    9.7%  of the 7,330 seeds already planned on
    9.4%  of accepted generated rows dropped for case_id

**Step 2 - so what is actually matching?** An eval item's identifiers come from
two channels: `identifiers_from_fields` (its own case identity) and
`identifiers_from_text` (every authority the passage cites). Classifying all
88 drops against those two channels:

    the eval item's OWN case (real overlap)                    0    0.0%
    only an authority the eval item CITES (citation graph)    97   77.0%
    unresolvable (multi-part ids, two unindexed files)        29   23.0%

**Zero.** Not one drop is our seed discussing the case an IL-TUR item is about.
Every classified match is a shared citation - our seed cites case X, an eval
item also cites case X, neither is case X. Indian judgments cite the same
landmark authorities constantly, so on judgment-derived seeds this fires
almost at random.

`decontaminate.py` predicted this exactly - "if one landmark citation turns
out to account for a large share of the drops, that is the citation graph, not
contamination" - and its tell never fired because the tell was the wrong shape.
It watches for ONE landmark dominating `top_identifiers`; the reality is 114
distinct citations with the largest at 3.2%. Diffuse, not concentrated, and
therefore invisible to the check written for it.

**Step 3 - the switch already exists, and turning it off is safe.** The same
docstring says the row side of this channel is "separately counted and
switchable" for this reason: `--no-case-id-from-text`. Measured on the scratch
chain, both runs on the same store:

                            baseline   --no-case-id-from-text
      total drops               321        161
      on GENERATED streams      159         78
      on file sources           162         83
      channel: case_id          179          0
      channel: narrow            61         72
      channel: short             69         75
      channel: text              12         14

The containment channels **go UP**, 142 to 161. That is the whole safety
argument in one line: 19 of the 179 case_id drops really did overlap by TEXT,
and exact n-gram containment - which is the primary defence and is untouched -
catches every one of them. The other 160 were co-citation and nothing else.

Net prize: **81 generated rows recovered (~9%), plus 79 file-source rows**, with
contamination protection intact.

#### Not flipped tonight, and the reason is not caution

Three reasons, in order of weight:

1. **Nothing is lost by waiting.** `decontaminate` re-reads every accepted
   generation out of the store on every run. The 81 rows are not destroyed,
   only excluded from today's assembly - a flip next week recovers them
   retroactively, plus everything accumulated since. This is the opposite of
   F35's ceiling, where the damage was permanent and waiting was the expensive
   option. Same discipline, opposite conclusion, because the reversibility is
   opposite.
2. **It is an eval-integrity decision, and those belong to the operator.**
   Every number this project will ever publish rests on the eval sets being
   clean. The evidence above is, I think, complete enough to decide in one
   read - but "I measured it and it looked fine" is not the standard for
   loosening a contamination guard unattended.
3. Nothing is blocked on it. ~27 h of queue remains.

**Task 22 is CLOSED as the wrong fix.** A plan-time seed filter would have
implemented the over-firing rather than removing it: shrinking the seed pool
by 9.6% to avoid generating rows that a mis-aimed channel deletes. It would
have worked, in the sense that the waste would have stopped - and it would
have cost pool and hidden the defect. The measurement that made it look
attractive is the same one that killed it.

#### If it is flipped, two things follow

- `--no-case-id-from-text` must be added to `assemble_argvs`' decontaminate
  step, not just run by hand, or the next cron run silently reverts it.
- **`MEASURED_RETENTION` must be re-measured.** Its generated figures (0.846,
  0.817) are readings against the CURRENT policy, and this change moves
  generated drops 159 -> 78. The table's own comment says re-measure when the
  decontamination corpora change; this is the same trigger by a different
  route.

### F39. THE CITATION-EXISTENCE HALF COSTS 0.85%, AND IT HAS BEEN ARMED ALL ALONG

Standing risk, carried for days as "no citation index - verify existence-half
UNVERIFIED". Every clause of it is wrong, and the last one is wrong in the
expensive direction.

    the index exists            76,238 citations, 1.25 MB, built 2026-08-29
    it is ON THE BATON          corpus/citation_index.txt, 1 of 26 files
    the live chain arms it      actions_worker reads root/corpus/citation_index.txt
                                and passes it to verify; the "no index in the
                                bundle" branch has never been taken
    arming it costs             8 rows of 943 - 0.85%

Measured by re-gating the full accepted population with the real index:
`regated 943, clean 847, demoted 8, unverified 0`. Four synthesis and four
curated_c2, all `accepted -> rejected`.

**Why the 4.7%-coverage worry never materialised.** The index covers 4.7% of
the citations that appear, and the reasonable fear was that arming a gate
which fails on ANY novel citation would delete most of the corpus.
`novel_citations(text, source, index)` takes the GROUNDING as well as the
index, so a citation the materials already carry is not novel whatever the
index knows. The model overwhelmingly cites what it was handed. What the gate
catches is the narrow thing it is for: a citation introduced from the model's
own memory that also exists nowhere in the corpus.

So `citation_index=None` in `generate.py` is not a gap - it is the pilot mode
the docstring describes, and the "MANDATORY FOLLOW-UP" it records (verify must
re-run with the real index before promotion) was implemented and is live.

**And the 8 rows were free.** All eight had already been dropped by
decontamination or dedupe - 0 of 8 appear in the assembled corpus. The
existence half has, so far, rejected nothing the chain was not rejecting
anyway.

**But do not read those 8 as caught fabrications.** Every one of them failed on
an SCC or AIR citation - `(1999) 3 SCC 231`, `AIR 1960 SC 980`, `(2013) 1 SCC
641`, `AIR 1954 SC 569` - and the index is **INSC 37,995 + SCR 30,601 + 7,642
other, with ZERO SCC and ZERO AIR**. So an SCC or AIR citation the model
introduces beyond its grounding is flagged novel BY CONSTRUCTION, whether it is
genuine or invented. The gate cannot tell those two apart in the two reporters
the corpus actually cites most.

The correct summary is narrower than "arming it is safe":

- The prediction it would reject ~95% of genuine citations was wrong by two
  orders of magnitude, and wrong for a specific reason worth keeping - it
  reasoned from index COVERAGE and did not account for `novel_citations`
  exempting anything the grounding already carries.
- Arming it is nearly free (8 rows, all already dropped) and should stay
  armed: it is a live tripwire on citations invented outside the materials.
- It is NOT evidence of citation soundness, and its pass rate must never be
  quoted as such - the same caution already recorded for the 99.2% `citations`
  pass rate, for the same reason. Extending the index to SCC and AIR is what
  would turn it into a real guarantee.

### F39b. THE RETENTION TABLE AND ITS CALLER USE DIFFERENT DENOMINATORS

Caught while checking F38's own numbers, and it partly undercuts them.

`--measure` reports a CHAIN retention: shipped over rows ENTERING
decontaminate. `generated_counts` multiplies that figure by
`store.accepted_count(stream)` - the count BEFORE verify runs. Those are the
same population only where verify demotes nothing.

    curated_c2   491 accepted -> 491 entered -> 401 shipped   0.817 is CORRECT
    synthesis    531 accepted -> 447 entered -> 378 shipped   0.846 OVERSTATES by 19%

The gap is exactly F38b's one-teacher cut. `531 x 0.846 = 449` against 378
that actually ship, and the difference is the 84 retired-provider rows that
never reach the chain.

It is a ONE-OFF, which is why the fix is a note rather than a number: those 84
are demoted for good by the first assembly run that arms the cut, and every
generation since 2026-08-28 is deepseek, so store-accepted converges on
entered and 0.846 becomes right. **Until that run lands, subtract 84 accepted
(~71 effective) from any synthesis sizing by hand.** Recorded in the table
itself so the next reader meets it at the number, not in this file.

Worth stating plainly: this is the same class of error as the throttle's, and
the third time on this project that two different denominators were quietly
multiplied together. The seven file-based sources are immune - they are not in
the store, so entered is what the loader shipped - which is precisely why it
took a generated-stream reading to expose it.

### F38. THE TWO GENERATED RETENTION FIGURES WERE A GUESS AND A DEFAULT. BOTH ARE NOW READINGS, AND BOTH WERE OPTIMISTIC

`MEASURED_RETENTION` sizes the whole corpus - an accepted task is not an
assembled row, and holding the accepted COUNT as the numerator while the chain
shrinks the denominator is what shipped grounded_synthesis at 27.7% against a
30.1% target with every stream pool individually on target. Its seven
file-based figures were readings. Its two GENERATED figures were not:

- `synthesis: 0.857` was labelled in the source, in capitals, `A PLACEHOLDER,
  NOT A READING` - kept rather than deleted only because deleting it fell back
  to a HIGHER number on no evidence at all.
- `curated_c2` had no entry, so it was silently taking `DEFAULT_RETENTION`
  0.95 - a figure nobody ever measured for it.

Both are now measured, off the first chain to ship 50+ generated rows:

    stream        entered  shipped  retention   was
    synthesis         447      378      0.846   0.857  (placeholder)
    curated_c2        491      401      0.817   0.95   (default)
    transition          5        3       n<50   absent (still absent)

**Both readings came in BELOW the number they replaced**, which is the
direction that matters: every generated sizing before today was optimistic,
curated_c2's by 16%.

#### The instrument was checked before it was believed

The table's own comment predicted `--measure` "will reproduce these seven
numbers off the next completed chain." It did - all seven, to three decimals
(0.996 / 0.846 / 1.000 / 0.958 / 0.983 / 0.910 / 0.957), off a chain run on a
store snapshot with pools shipped WHOLE rather than shaped. Seven known
answers reproduced exactly is what qualifies the two unknown ones; without that
check this is just a number from a different-looking run.

#### The loss is one decontamination rule, not a spread

Of synthesis's 69 drops and curated_c2's 90, **all but two are
decontamination** - dedupe took 2 rows and the length cut took 0 - and the
single largest reason is `case_id:iltur`:

    curated_c2   case_id:iltur   58      synthesis   case_id:iltur   30
    curated_c2   narrow:bbl      11      synthesis   narrow:bbl      15
    curated_c2   short:bbl       10      synthesis   short:bbl       14

These are generated rows whose SEED CASE is in the IL-TUR eval set. So:

1. The figures move when the eval corpora move, not when the gates or the
   templates do. Re-measure after a decontamination corpus changes, not after
   a prompt edit.
2. **88 of 938 accepted generated rows (9.4%) were burned on seeds the chain
   was always going to delete.** That is fleet time spent generating rows that
   cannot ship. A planner-side seed filter against the IL-TUR case-id set
   would recover it, and it changes no prompt sha - but it only helps rows
   planned AFTER it, since pending tasks are already stamped to their seeds.
   Logged as task 22, not done tonight (see F37 on why nothing writes to the
   store while a worker holds the baton).

#### What it does to the one-way door

The guard compares an EFFECTIVE count against an effective ceiling, and the
ceiling is invariant to these two values - it is set by the pools, and it
stayed at 2,050. Only the numerator moved:

    effective generated-curated   466 -> 401   (491 accepted x 0.817, not x 0.95)
    headroom to the guard's trip point       1,499 effective rows
    the door in ACCEPTED terms    ~2,158 -> ~2,509 accepted curated rows

So the door is **further away than F35 said**, by ~350 accepted rows. This is
the guard being more accurate, not more permissive: the same code, reading a
measured retention instead of a default nobody fitted. It is also the third
correction in this file that ran in the safe direction only by luck, which is
the argument for measuring rather than defaulting.

### F38b. THE ONE-TEACHER CUT TAKES 84 ACCEPTED SYNTHESIS ROWS

Measured on the same chain, and it belongs beside F35's arithmetic. The
assembly chain arms `verify --require-generator --require-current-prompt`, and
that cut demotes **531 accepted synthesis rows to 447** - 84 rows, 16%, every
one of them from a RETIRED provider (`cerebras/gpt-oss-120b` and
`lightning/lightning-ai/gpt-oss-120b`) under the 2026-08-28 sole-generator
ruling. `curated_c2` is untouched at 491, so the ceiling guard is unaffected.

The consequence: **wherever this file quotes accepted synthesis, the number
that reaches the corpus is 16% lower.** F35's band table reads 531 on the
synthesis axis; the assembling corpus sees 447. Combined with F38's retention
that is 447 x 0.846 = 378 effective rows from 531 accepted - a 29% total
haircut between "accepted" and "in the training set", and the single most
common way this project has mis-sized itself.

### F37. F36's VARIANT EFFECT IS ONE GATE, AND THE GATE IS RIGHT

F36 measured that the prompt variant is worth 3.6x the fleet's time and left
the mechanism open. It is `irac_placement`, almost entirely. Measured on this
run's 428 generations (2026-08-31 07:15Z onward), independent of F36's sample:

    gate                  failed  seen   fail%     sole blocker on
    irac_placement           247   428   57.7%     66 generations
    length_band              152   428   35.5%     13
    banned_meta              100   428   23.4%     20
    prompt_echo               94   428   22.0%      9
    statutory_grounding       49   428   11.4%      7
    self_verification         31   428    7.2%      9
    verbatim_overlap          14   428    3.3%      -

`irac_placement` fails more than the next two gates combined and is the sole
thing standing between 66 generations (15% of the run) and a clean row.

#### It is not drift. I checked, because F24 was.

F24's `irac_placement` blow-up WAS template/gate drift, so drift is the first
hypothesis, not the last. It does not hold here, on four checks:

1. **The failure mode is the trace, not the answer.** Of 247 failures, **225
   are leak-only** - the ANSWER carries its headings correctly and the THINK
   trace also runs under IRAC labels. Only 11 are a missing heading.
2. **The regex cannot be firing on prose.** `_IRAC_HEADING_RE` is line-initial
   and requires a terminator (`:`, `.`, emphasis, or end of line), so "the
   issue is settled, this is the rule" - which the templates themselves
   suggest writing - does not match.
3. **The matched text is real.** Sampled four failures: `'    Issue: Whether
   the issuing bank can refuse...'`, `'**Rule:**'`, `'Conclusion: Conviction
   of Devender upheld...'`. These are full labelled IRAC run-throughs inside
   `<think>`, not false positives.
4. **The template already forbids it, explicitly and at length.** v4 says the
   headings "belong to the model answer and never inside your reasoning, which
   runs as continuous prose and never opens a line with one of those four
   words", and spends a further three sentences pre-empting the exact habit.

So the gate is doing precisely its job - this is the MSLR pathology, a model
scripting its reasoning as a template it fills in afterwards - and the model
ignores the prohibition anyway. That is the same result as the trace-length
A/B and the genre finding: **an explicit format prohibition does not survive
contact with the model's habit.** Do not spend another round on wording.

#### Retries do not rescue it either

    irac_placement pass        attempt 1  49.7%   attempt 2  37.9%   attempt 3  36.7%
    ALL gates clean            attempt 1  21.5%   attempt 2  18.2%   attempt 3  18.3%

The rate FALLS with attempt. That is survivorship, not decay - attempt 2 only
ever re-rolls tasks that failed attempt 1, which are the harder seeds and the
worse variants - but the practical reading stands: a retry is worth ~18%, not
the 21.5% of a fresh task, and it costs a whole generation. Retrying is not
the lever, and MAX_ATTEMPTS is not mis-set.

#### The lever is allocation, and here is its price

Same run, irac_analysis only. The assignment is `sha256(seed_id:sample_ix) %
len(pool)`, so this is a randomised trial, not an observational split:

    variant   clean  gens  clean%  gens/row  ktok/row
    v1           27    73   37.0%      2.70      15.9
    v3           27    95   28.4%      3.52      23.5
    v2           20   138   14.5%      6.90      47.7
    v4           10   122    8.2%     12.20      82.5
    ALL          84   428   19.6%      5.10      33.9

**v4 costs 4.5x v1 per clean row in generations and 5.2x in tokens.** The
ordering replicates F36's (v1 > v3 > v2 > v4) on fresh data, which is what a
real effect looks like.

The pending queue does not know any of this - it was planned before the
measurement and is split almost evenly:

    pending irac_analysis:  v2 405   v1 402   v4 397   v3 360

At that mix the fleet pays 5.10 generations per clean row. On v1+v3 only it
would pay ~3.0 - **about 1.7x the corpus for the same fleet**, or ~128
gate-passing rows/hour against today's ~75.

#### The other three gates are right too - I checked all four

A gate fix is FREE and a template fix is expensive: gate thresholds live in
config and change no prompt sha, while editing a template re-stamps it and
parks every pending task on it. So an over-firing gate would have been the
cheapest win available. There is none. All four checked:

- **`banned_meta` (100 failures)** - every one is the single phrase "the
  excerpt". No template mentions the word (`grep` = 0 across all four irac
  templates), so the model volunteers it: it refers to the material it was
  handed as a document. At inference there is no excerpt, which is precisely
  what the gate exists to stop. Correct.
- **`prompt_echo` (94)** - 69 are echoed instruction spans, 29 a restated
  opening. One echoed span is `"never write as though the matter had been
  handed to you as a text"` - the model reciting the template's own
  prohibition back into the answer. Correct, and a second instance of the F37
  rule: stating a prohibition is not enforcing it.
- **`length_band` (152)** - `think>think_max` 138, `total>total_max` 86,
  `think<think_min` 8. **Not unsatisfiable**: prompt + think_min + answer_min
  exceeds total_max on 0 of 428, and a think at the 4,500 cap fails to fit on
  only 7%. prompt_est runs p50 1,527 / p90 3,229 / max 4,669 against 8,192, so
  the budget is not being eaten by the prompt.
- **`irac_placement`** - four separate checks above.

#### think_max: measured, positive, and deliberately NOT taken

Since trace length is not steerable by prompt, the only lever on the 138
`think>think_max` failures is the cap. Raising it saturates, because
`total<=8192` takes over:

    think_max   passes length_band   as SOLE blocker -> clean rate
       4500          276 / 428        (baseline)        19.6%
       5000          300              +3                20.3%
       5500          316              +6                21.0%
       6000          328              +8                21.5%
       6500          333              +9                21.7%
       7000          333              (no further gain)

So 4500 -> 6000 is worth **+9.5% relative throughput**. Not taken, for three
reasons. It is the *second* lever, and the variant allocation above is worth
+70% for no quality cost at all - spending the quality budget on the smaller
one first is backwards. `think_max` has already ratcheted 3,000 -> 4,000 ->
4,500 chasing yield, and each raise buys volume with trace discipline. And
longer traces cost tokens per epoch against a corpus already budgeted at
~37-50M tokens per epoch on a quota-week.

Recorded rather than done, so the next person to want throughput finds the
number already measured and does not re-derive it: **the cap is worth ~+9.5%
if it is ever the right thing to spend.** It changes no prompt sha, so it can
be taken at any time - and `REOPEN_ON_EMPTY` re-opens `format_parked` rows, so
previously parked generations would get a second chance under the wider band.

#### What NOT to do about it tonight

The stamped tasks cannot be re-pointed: `task_id_for` hashes the prompt_id, so
a task IS its variant. The three tempting actions are all wrong right now:

- **Editing a template** re-stamps its prompt_sha and parks every pending task
  on it as `stale_prompt`, permanently (there is no re-stamp; re-opening
  re-parks instantly).
- **Parking the v2/v4 tasks** means writing task state to the store, and the
  baton is held by a running worker continuously - run N+1 starts 2 seconds
  after run N ends. There is no safe window, and a writer against `data/build`
  during a run is the rule this session already broke once.
- **Dispatching `data-plan` now** costs a worker cycle. CORRECTED later the
  same day: the original claim here was that it "would evict itself" because
  the cron claims the waiting slot. That is backwards. The group holds one
  running plus one waiting, a new pending trigger REPLACES the waiting one, and
  the queued run starts when the current one ends - which on a 315-minute run
  is ~40 minutes BEFORE the next cron fires, as the workflow's own comment says
  it is designed to do. So a `data-plan` dispatch evicts the QUEUED WORKER and
  then runs normally. Planning is therefore possible at any time; it just costs
  one ~5.25 h worker cycle. Not worth it while ~27 h of queue remains - and
  worth it as soon as the top-up in F41 is the thing standing between the
  corpus and the band.

And it is not urgent: 3,162 synthesis plus 1,592 curated_c2 tasks at ~2.36
generations each is ~27 hours of fleet work. The recommendation is for the
NEXT wave, whenever it is planned - `--plan-variant gen_irac_analysis_v1
--plan-variant gen_irac_analysis_v3` (both, not v1 alone: prompt diversity is
worth more than the 0.8 generations/row v1 saves over v3).

### F36. THE PROMPT VARIANT IS WORTH 3.6x THE FLEET'S TIME, AND IT REPLICATES ON DEEPSEEK

`pick_variant` is `sha256(seed_id:sample_ix) % len(pool)`, so seeds are
RANDOMLY assigned to prompt paraphrases and the rotation is a free randomised
trial. Read on the live snapshot, deepseek generations only, terminal tasks:

    prompt_id                 tasks   accepted   accept%   gens/accepted row
    gen_irac_analysis_v1        127        111     87.4%        3.35
    gen_irac_analysis_v3         93         65     69.9%        5.57
    gen_irac_analysis_v2         81         53     65.4%        6.70
    gen_irac_analysis_v4         94         44     46.8%       11.95
    gen_summarization_v2         28         27     96.4%        2.26
    gen_summarization_v1         22         16     72.7%        4.88

**v1 buys 3.6x the rows per generation that v4 does.** Generations ARE the
fleet's time, so this is the largest lever on the binding resource, and it is
already paid for.

#### The two confounds, both checked

1. **Retired generator.** The pooled table mixes cerebras/gpt-oss (1,135 gens),
   lightning (146) and mistral (115) with deepseek (1,800). The figures above
   are deepseek ONLY - `EXISTS (... g.provider='bai')` - because the pooled
   profile belongs to a generator that no longer runs.
2. **Gate era.** A ranking is worthless if the arms ran under different
   `think_max` settings. They did not - the arms are contemporaneous to the
   SECOND:

       gen_irac_analysis_v1   372 gens   2026-08-28T21:27:42 .. 08-31T05:03:41
       gen_irac_analysis_v2   355        2026-08-28T21:27:56 .. 08-31T05:03:23
       gen_irac_analysis_v3   362        2026-08-28T21:27:52 .. 08-31T05:04:04
       gen_irac_analysis_v4   526        2026-08-28T21:28:04 .. 08-31T05:03:30

   All four start within 22 seconds and end within 41. Same period, same
   teacher, same thresholds, random assignment. This is a clean RCT.

Note v4 spent the MOST generations (526) to produce the FEWEST accepted rows
(44). It is not merely a worse template; it is actively the most expensive one.

#### What can and cannot be done about it

**The 3,453 pending synthesis tasks are already stamped.** `task_id_for` hashes
`prompt_id`, so a task IS its variant, and `tuned.data.tasks` has no park or
cancel - only `--reopen`, which un-parks. Deleting a bad template would park
its pending tasks as `stale_prompt`, which is TERMINALLY_DEAD and irreversible.
**So the queue runs as planned**; expected yield at the rates above is ~2,630
accepted, against ~3,612 needed for MVP.

**The lever is the NEXT wave, and the flag for it shipped this session (F33).**
The top-up needed is ~600-700 accepted rows, and planned on the good variants
that is a much smaller wave than it would otherwise be:

    Actions -> data-plan -> variants:
      gen_irac_analysis_v1,gen_irac_analysis_v3,gen_summarization_v2

**v1+v3, not v1 alone** - deliberately. Pinning to the single best template
maximises yield and minimises PROMPT DIVERSITY: every irac row would then come
from one paraphrase, which is a training-data risk the accept rate does not
price. v3 is a different persona at a good rate (senior advocate arguing aloud
vs judge writing judgment), so v1+v3 keeps two genres and drops only the two
bad ones. Cost is ~9pp of yield against v1 alone.

**Do NOT dispatch it yet.** `data-plan` shares the `data-build` concurrency
group, so dispatching now would displace the queued worker run, and the queue
already holds 3,453 pending tasks - more than the frozen-curated band admits
(F35). Plan the top-up when synthesis nears drained, at the same time as the
curated_c2 re-open.

**Cheap follow-up, safe to do any time:** ADD a second summarization variant in
the proven genre. Adding a FILE does not change any existing template's sha, so
no pending task is parked; EDITING one parks every pending task on it. That is
the whole reason the rule is "add, never delete or edit".

### F35. THE CURATED CEILING IS A ONE-WAY DOOR, AND IT IS NOW GUARDED BY MEASUREMENT

This is the most important thing in this file. It is not a throughput problem
and the door does not open again.

**Rewritten twice on 2026-08-31. Both corrections are kept, because the way
each one was wrong is more useful than the numbers.**

1. *"~11 accepted rows from a wall it would hit in days."* The ceiling number
   was right; the framing was not. That figure measured the distance to where
   damage becomes PERMANENT and read it as the distance to trouble. The corpus
   had already left the feasible band - `shape` refuses from both sides, and it
   was refusing then.
2. *"So throttle curated_c2 off `STREAMS`."* The asymmetry behind that was
   sound and the arithmetic under it was not: it multiplied PENDING tasks by a
   JUDGED-accept rate, which is the wrong denominator by ~32 points, and it
   never priced the throttle. Freezing curated caps the corpus at 3,655 rows
   where draining it reaches 8,430.

The standing answer is neither: the run now MEASURES its distance to the
ceiling and stops curated_c2 on the reading. See the DECISION section.

**`shape` refuses from BOTH sides.** Too few generated synthesis rows and the
curated bucket overfills its share of the mix; too many and the corpus outgrows
the shortest stream pool. Between them is a window, and the window NARROWS as
the generated-curated count rises - until it closes:

    effective gen_curated   feasible gen_synthesis   corpus   window
              371                 550 ..  1025        1,827    475
            1,200               1,800 ..  2,300       5,980    500
            1,600               2,350 ..  2,900       7,807    550
            2,000               3,000 ..  3,200       9,801    200
            2,050               3,000 ..  3,200       9,801    200
            2,100                    INFEASIBLE          -       -

**Ceiling: ~2,050-2,065 effective generated-curated rows.** Past it no synthesis
count, in any quantity, assembles this corpus at `v1.0-MVP`. And it cannot be
undone: `shape` trims STREAM files, while `decontaminate` reads every accepted
generation out of the store, so **a generated row is in the corpus by
existing.**

#### Where the build is heading, unattended

**The 96.3% in the first version of this block is the accept rate among JUDGED
rows. It is not the yield of a pending task, and using it as one is the error
that produced the throttle.** A pending task must clear the format gates
before a judge ever sees it, and `format_parked` is where most of the losses
are. Both denominators, measured 2026-08-31 over 2,360 curated_c2 tasks:

    judged accept rate      491 / 508  = 96.7%   <- what the first version used
    per-task yield          491 / 756  = 64.9%   <- accepted / everything decided

`format_parked` sits between them and is not cleanly terminal: `REOPEN_ON_EMPTY`
re-opens those rows when the queue drains, so some fraction of them eventually
pass. The landing is therefore bracketed, not known:

    at 96.7%   2,277 accepted = 2,163 effective   -> BREACHES the 2,050 ceiling
    at 64.9%   1,691 accepted = 1,607 effective   -> 443 effective clear

**So the concern was real and the arithmetic was not.** One end of that range
does breach a door that never reopens, driven by a recovery path the worker
runs BY ITSELF. Nothing warns: `shape` is not run by the worker, only by
`data-assemble`, so the failure would first surface as a `REFUSED` days later,
by which time the rows that caused it are unremovable.

What follows from a range this wide is not a better forecast. It is that the
build must stop on a MEASUREMENT rather than on either estimate - see the
decision below, which is what actually shipped.

Timeline, as first written: "the queue holds ~2.4 days ... days away, not
hours. No emergency tonight." **That was wrong, and it is the reason this
finding was nearly left as a note.** It measured the distance to the CEILING
and read it as the distance to trouble. The ceiling is only where the damage
becomes permanent - the corpus left the feasible BAND well before it, and is
outside it now. The headroom figure above is likewise headroom to the ceiling,
not to feasibility. Both numbers stand; the reassurance drawn from them did
not.

#### What binds it, measured

    pools: curated/nothink 1400   curated/trace 300
           replay/nothink  1200   replay/trace 3120

At gc=2,066 the refusals name the two walls exactly:

    gs=2,600  "curated/trace would need -286 rows - the generated rows
               already in that bucket overfill it"
    gs=3,100  "replay/nothink needs 1233 rows (to keep 1200 after losses)
               but the pool holds 1200"

**`replay/nothink` at 1,200 rows is the binding constraint** - it caps the
corpus at ~10,021 (F19's ceiling, now explained rather than just observed),
and the corpus size is what caps how many generated curated rows fit inside
the curated share.

**`--replay-nothink-share` does NOT rescue it.** The refusal message suggests
that lever, so I tested it at 0.0 / 0.1 / 0.2 / 0.3 / 0.4 / 0.5 / 0.6 / 0.7:
INFEASIBLE at every value. It moves the no-think budget between streams; it
cannot conjure rows into a pool that holds 1,200.

#### What the band actually requires (this is the operative table)

`synthesis_band` inverted - for a realised accepted-synthesis count, the
accepted curated_c2 counts that admit ANY corpus at all:

    accepted curated_c2    accepted synthesis that admits it   max corpus
              491  (today)           817 .. 1,342                    3,655
              700                  1,138 .. 1,721                    5,082
              900                  1,459 .. 2,071
            1,100                  1,809 .. 2,421                    6,465
            1,300                  2,130 .. 2,742
            1,526  (drain @64.9%)  2,480 .. 3,151                    8,430
            1,800                  2,917 .. 3,617                    9,691
            2,000                  3,238 .. 3,734                   10,021
            2,158  (the ceiling)   3,501 .. 3,734                   10,021

Re-measured 2026-08-31 09:30Z on live counts, and stated the other way round
from the first version because curated is the axis with the one-way door on
it. Two readings matter. First, **the band RISES with curated** - more curated
rows admit a bigger corpus, which is why freezing the count is not a free
safety measure. Second, **full-MVP synthesis (3,617) is reachable only if
curated lands in ~1,800..2,158**; frozen at 491 the corpus tops out at 3,655
rows.

Read the top row: at today's 491 accepted curated_c2 the corpus needs
817..1,342 accepted synthesis, and the store holds **531**. That is the refusal
the chain already reports - it is a present fact, not a future risk - and it is
on the LOW side: synthesis is short, not curated long. The ratio the corpus
wants is roughly **two accepted synthesis rows per accepted curated row**, and
the pair is currently at ~1.08.

The projected pair is what the throttle got wrong. Synthesis has 3,162 pending
and curated_c2 1,592; at their measured per-task yields those land near (1,526
curated, ~2,000 synthesis), which is inside the band and admits ~8,430 rows.
The first version of this section projected (2,149 curated, 1,550 synthesis) by
crediting curated with a 96.3% judged-accept rate and synthesis with a 33%
terminal rate - two different denominators on the two axes, which is exactly
how a pair lands outside a region it is actually inside.

#### Verified end to end, not just computed

The band above is arithmetic over `shape.plan`; this is `plan` itself, run
against the LIVE baton snapshot rather than the local `data/build` copy (which
holds a stale 17 synthesis / 0 curated and will happily "succeed" on a 50-row
corpus - check which store a number came from before believing it):

    accepted    {grounded_synthesis: 414, curated: 391}
    effective   {grounded_synthesis: 355, curated: 371}
    REFUSED: no corpus size between 1106 and 1179 rows works. At the largest,
      curated/trace would need -128 rows - the generated rows already in that
      bucket overfill it. Either generate more grounded_synthesis (every row
      buys ~3.3 corpus rows), rebuild the short stream larger, or move the
      no-think budget with --replay-nothink-share.

Three things this settles: the refusal is REAL and current, it is on the
OVERFILL side exactly as the band predicts, and `plan`'s own first suggested
remedy is "generate more grounded_synthesis" - which is what the throttle
redirects the entire fleet to do.

414, not 409: `grounded_synthesis` is synthesis (409) PLUS transition (5). The
band table's left column is that sum. Transition is drained, so all further
growth is synthesis.

#### The design agrees, and says the same number

The config is internally consistent and was never sized for this. Taking the
built pools as given: replay.py makes 4,320 rows and the profile puts replay at
41.94%, so the intended corpus is **10,300** rows. Then

    grounded_synthesis  0.3010 -> 3,100   (= the 2,000-row core + 1,100 transition)
    curated             0.2796 -> 2,880   (= curated.py's 1,700 + 1,180 GENERATED)
    replay              0.4194 -> 4,320   (= the pool exactly)

Every one of those lands on a number the config states in prose elsewhere, so
this is the design, not a coincidence. **The design budget for curated_c2 is
1,180 corpus rows**, and the stream was planned at 2,360. It has been over
budget since it was planned; the ceiling is just where over-budget becomes
permanent.

#### DECISION, as it ended up: a measured ceiling guard, 2026-08-31

**This was a hand throttle for about four hours, and then it was not. The
round trip is the finding.**

What shipped first: `STREAMS = ("synthesis", "transition")`, curated_c2 simply
never claimed. Taken on the asymmetry - stopping is reversible because pending
rows cost nothing, draining is not because generated rows cannot be dropped.
The asymmetry argument was sound. The number it was applied to was not, and
the price of applying it was never checked.

**The price, measured afterwards.** Freezing curated at 491 accepted does not
merely pause a stream; it caps the corpus, because the curated count sets how
large a corpus the fixed shares admit:

    curated accepted     max synthesis (accepted)    MAX CORPUS
        491 (frozen)                      1,342          3,655
      1,526 (drained at 64.9%)            3,151          8,430
      2,000                               3,734         10,021

**The throttle cost ~4,800 rows of assembled corpus** to prevent a breach only
the pessimistic end of the yield range could cause. Neither hardcoded answer
was right: serving curated_c2 unconditionally risks a permanent door, holding
it unconditionally pays 4,800 rows for that safety.

**What shipped instead (`be25afd`): the decision became a measurement**, taken
once per run after the baton is restored and before anything is claimed.

    ceiling_state(config, root)   -> (effective generated-curated, ceiling)
                                     ACCEPTED rows, retention-corrected, for
                                     the profile the chain actually assembles
    served_streams(STREAMS, ...)  -> drops curated_c2 within
                                     CEILING_MARGIN_EFFECTIVE (150) of it

`STREAMS` lists all three again. On live counts the guard serves curated_c2 now
(466 effective) and will hold it at 1,900 effective - about 2,000 accepted -
against the 2,050 ceiling. It costs 2.9 s per run. Both ends of the yield range
are now safe without either being predicted, which is the point: **a margin
does not have to be forecast if the build stops on the reading.**

It **fails closed.** An unmeasurable ceiling - no store, no stream files, a
config that will not load, a cold checkout before the baton is unpacked - holds
curated_c2 rather than serving it. The asymmetry that justified the hand
throttle is exactly right for the UNKNOWN case; it was only wrong as a
substitute for looking.

Three things stopped being restated while they were load-bearing:

- the served stream list now reaches `child_argvs` and `_finish` as an
  argument. The module constant no longer knows what a given run decided, and a
  report reading the constant would describe the wrong run.
- `PROFILE` is one constant. A ceiling is a property of a profile - the shares
  decide which pool binds - so a guard measuring `v1.0-MVP` while the chain
  assembled something else would clear a stream on the wrong number.
- `_task_counts` still reads the WHOLE store while `_claimable_in(db, streams)`
  answers "is there anything THIS run serves". Keeping both matters: with only
  the whole-store count, unserved rows read as work forever, the queue-empty
  guard never fires, `REOPEN_ON_EMPTY` never runs, and every run sits its full
  ~5 h claiming nothing - the exact stall this session opened by fixing.
  Holding a stream would otherwise silently disarm the recovery.

`transition` STAYS although it is drained (0 pending, 2,177 rejected).
`budget = n_workers * len(streams)` and the top-up walks the tuple in order, so
a drained stream donates its floor to synthesis. Dropping it would cut the
fleet from 2xN to N calls in flight.

#### The low side still binds, and the guard does not touch it

None of the above changes the fact that **the corpus is unassemblable TODAY,
and on the LOW side**: at 491 accepted curated the feasible accepted-synthesis
band is 817..1,342, and synthesis stands at 531. The remedy is to generate
synthesis, which is what the fleet is doing.

That deadline is SOFT, and its asymmetry is the opposite of the ceiling's:
underfilling curated costs assembly DELAY and is fixed by generating more
curated rows - which the guard now allows automatically. Only the HIGH side is
one-way. So there is no number on this side worth taking a risk to hit.

`shape --headroom` prints the live band, so the condition is checkable in one
command and does not depend on this table staying fresh. The 391 rows already
accepted are inside the window for a synthesis landing of 636-1,196, so
nothing generated so far is wasted.

The remaining alternative, if synthesis does reach full MVP and more curated is
wanted than the ceiling allows: **rebuild `replay/nothink` larger.**
`replay.py` takes `--counts` and its sources are public datasets, so the pool
is not fixed by anything but the build that made it - and a larger short pool
raises the corpus ceiling and the curated ceiling with it (pinned by
`test_curated_ceiling_rises_when_the_binding_pool_is_larger`). It costs a
window with no worker running, because `streams/` is baton-owned. It is the
right move only AFTER synthesis has earned it; doing it now would be growing
the corpus past its design to accommodate rows the design never asked for.

#### The instrument, so this number is never a stale doc again

    python -m tuned.data.shape --config <cfg> --profile v1.0-MVP --headroom

New (`--headroom`), read-only, writes nothing. Prints the ceiling, today's
count, the headroom in ACCEPTED rows (converted back through the retention
factor, because accepted rows are what an operator throttles), the synthesis
band needed now, and the band at the ceiling with its width. `curated_ceiling`
and `synthesis_band` are public and tested, including the case where the pools
admit no corpus at all - which is a POOL problem and the opposite instruction
to "generate less".

The search galloping-then-bisects and its inner sweep is deliberately coarse,
so it can UNDERSTATE the ceiling by up to one probe. That is the safe direction
for a warning and it is why the report says 2,050 where a fine scan says 2,065.

#### What this corrects

The state of play said the shortfall was "~850 synthesis rows" and F16 put it
at 469. Both measured the RATIO (1.46:1) and neither measured the CEILING, so
both described a corpus that gets better as it grows. It does not: past
gen_curated ~2,050 it stops existing. The real requirement is a MATCHED PAIR -
and at full drain the pair is (2,157 curated, ~1,810 synthesis), which is
outside the feasible region on both axes at once.

### F34. The dataset card claimed a citation check that has run on ZERO rows

The card told its reader, twice, that rows ship having passed "citation
existence". Measured on the live store:

    accepted rows with a `citations` gate result   1,875
      existence half SKIPPED (novel_skipped)       1,875   (100%, all "no-index")
      existence half actually checked                  0

Not "mostly skipped". **Never run, on any row that has ever shipped.**

This is not an inference. `check_citations` returns
`{"novel": None, "novel_skipped": "no-index", "suspect": [...]}` whenever
`ctx.citation_index is None`, and passes on the suspect channel alone. And
`gates.py`'s module docstring already made the reading rule mandatory, in
capitals, before I found this: *"a stored gate_result carrying novel_skipped
must be read as 'unverified', never as 'passed'."* The card was reading it as
passed.

It matters more here than the count suggests. Citation existence is the one
claim a reader of an INDIAN-LAW corpus checks first, and F29 independently
found 4 of 50 sampled rows citing an authority their source never names. A card
that claims the check ran is worse than a card that stays silent, because it
tells a reader not to look.

**Fixed in the prose, not in the pipeline, and that is the right scope.** The
truthful version now names both halves, states the condition on the strong one,
and does not count a skipped half as a check survived:

- the SUSPECT half - citation-shaped strings in reporter formats the index does
  not model, diffed against the grounding so a cite carried IN is not counted
  as an invention - needs no index and always runs. Verified against
  `citations.suspect_citations` before writing it onto a public card;
- the EXISTENCE half runs only where an index is present, and where it is not
  the row is `novel_skipped: no-index` and reads as unverified;
- and a limitation that survives even WITH an index: existence is not aptness.
  The strongest thing the gate can say is that an authority is real, never that
  it supports the proposition it was cited for. The "Known risk" section said
  citation existence was among the gates a wrong answer sails through, which
  implied the gate had run; it now says what is actually true.

**Why not just arm the gate.** Building the index is not a prose fix and cannot
be validated unattended: the 2026-08-31 audit put its coverage at 4.7% of
citations, and arming an index that thin turns a permanent `citations` reject
into the default outcome for correct citations. The card change is honest at
any coverage; arming the gate needs a coverage measurement first.

Card prose only - no `PUSH_VERSION` bump, following the P1.7 precedent in
`push.py` (README text changes nothing in `build_manifest.json`'s shape).

#### Two more claims audited in the same pass, both corrected

Having found one, I read every sentence of the rendered card against the code
rather than stopping at the finding:

1. **The transition section rendered unconditionally.** "A dedicated stream
   teaches the IPC-to-BNS recodification directly" is a headline feature claim,
   and the stream ended with **5 accepted rows** (F22). At ~5,000 rows that is
   0.1% - and the reader most harmed by the overstatement is exactly the one
   choosing this dataset BECAUSE of the transition. The section is now counted
   from the shipped rows (`_prov.task_type`, over train + eval, so the number
   is exact rather than sampled) and **disappears entirely at zero**. It also
   now says the thing F30 proves: an answer key constrains which sections an
   answer must cite, not which conclusion it reaches.
2. **"each row carries the number of characters cut from it"** was wrong about
   four fifths of the corpus. `answer_preamble_dropped` is written in
   `decontaminate.py:1196`, in the GENERATED-row branch only; curated and
   replay rows have no such field and are never trimmed. Now "each generated
   row".

**Left alone, deliberately:** the card says "an operator reads 50 random
accepted examples before the corpus ships". The pipeline does not enforce that,
and Task 13 is still open - but `push.py` runs only at the end of an
operator-dispatched `data-assemble`, so there IS a human at the ship, and F20
already says not to dispatch it yet. Recorded rather than rewritten: turning it
into an enforced attestation would change the ship path, and that is the
operator's call, not one to take while they are asleep.

### F33. The allowlist existed and could not be reached, and Step 6 is now one dispatch

`c61311a` put `--variant` on `python -m tuned.data.tasks`. That module runs on
the operator's laptop, which **must not touch the baton** - `--phase seed-push`
refuses once the remote owns it, and the only sanctioned path from an operator
to the remote queue is `data-plan.yml -> actions_worker.py --phase plan`.

`run_plan` built the planner argv by hand:

    argv = [... "--stream", ..., "--n", ..., "--mix", args.plan_mix]

No variant. So the feature that F24 called "a bigger prize than every
throughput lever in this file combined, and it is free - it costs one line in
the Step 6 plan command" **had no line to be written on.** Shipped tonight:
`--plan-variant` (repeatable AND comma-separated, because `workflow_dispatch`
has no list input type) plus a `variants` input on the workflow.

Two details are load-bearing rather than cosmetic:

- **Blank means every template, not no template.** An operator leaving the box
  empty sends `""`; forwarding that would hand the planner an allowlist naming
  nothing. Same shape as the empty-skip-set guard Step 1 needed in
  `generate.py`, and pinned by its own test.
- **The ids are split here and validated there.** `tuned.data.tasks` checks
  each against the registry and exits with a usage error on a typo, so a
  mistyped persona fails before a row exists rather than quietly shrinking the
  pool. Splitting in `run_plan` keeps that check where it already is.

#### The persona decision, taken

Read straight off F24's randomised trial - no new evidence needed, and none
obtainable, since the A/B that Task 9 wanted requires a rate bucket the fleet
holds around the clock:

| task type | allow | why | reject |
|---|---|---|---|
| `irac_analysis` | `v1`, `v3` | 3.3 and 4.2 gens/accepted-row; judge-writing-judgment and advocate-speaking-aloud | `v2` (7.2), `v4` (**15.6**, and top of `banned_meta` and `prompt_echo`) |
| `summarization` | `v2` | 4% `irac_placement` failure | `v1` (27% - **the identical instruction sentence**, z = 5.7) |

`statute_qa` and `drafting` stay out of the mix: the first has zero eligible
seeds and the second is parked, and the shipped `SYNTHESIS_MIX` sends a quarter
of the wave to `statute_qa` silently. That is why `--plan-mix` has no default.

#### The command

Dispatch `data-plan` with:

    stream    synthesis
    n         <live synthesis tasks now> + 1600
    mix       irac_analysis=0.55,summarization=0.45
    variants  gen_irac_analysis_v1,gen_irac_analysis_v3,gen_summarization_v2

**`n` is a TARGET, not an increment,** and it is measured against LIVE tasks -
everything except `rejected` / `stale_prompt` / `input_ineligible`. At the
06:00Z snapshot synthesis held **4,003 live** (3,453 pending, 409 accepted,
124 `format_parked`, 12 generating, 5 `judge_skipped`), so the target was
**5,603**. That number decays: accepted rows keep counting, rejected ones stop,
so live falls at roughly the rejection rate and a target fixed today inserts
MORE the later it is fired. Erring high is safe here - the corpus is short on
synthesis in particular, and more synthesis moves the `shape` ratio the right
way. Re-read the count first if it is easy; do not block on it.

**Why 1,600.** F24 measured the speech/letter variants at 2,221 tasks ->
1,401 accepted rows, i.e. **0.63 accepted rows per task**. The gap to the MVP's
~3,207 accepted synthesis is ~850, which is 1,347 tasks; 1,600 carries ~19%
margin for the rate moving under a narrowed pool.

#### Why it was NOT dispatched tonight

Not caution - arithmetic. The `data-build` group holds one running run and one
waiting run, and a new pending trigger **replaces** the waiting one. So a
`data-plan` dispatch now would cancel the queued worker, run ~20 min against a
565 MB baton round trip, and leave the machine idle until the 08:17Z cron:
**~55 min of lost generation, about 40 accepted rows.**

Against that, it buys nothing for about 2.4 days, because that is what the
pending queue already holds. There is no free window either - runs are ~5h16m
against a 4h cron, so the fleet is saturated by design. The right moment is
when the queue is actually running down, and the whole cost of finding that
moment is one `gh run list`.

**The deadline is real but distant:** if nobody fires it, the queue drains
around 2026-09-02 and the worker starts re-opening parked rows instead of doing
new work.

## 6. Constants interrogated

| Constant | Verdict |
|---|---|
| `cooldown_s = 300.0` (`providers.py:1774`) | **Nobody chose it.** Generic `Router.__init__` default; no config key; all three `make_router` sites bare. deepseek's 429s come from a per-minute bucket (**301 of 2,151 requests on 08-29**, 14%) that refills in ~60 s. -> Step 1b |
| `breaker_threshold = 4` | Also a default, but justified: `ChatClient.complete` absorbs 6 jittered retries honouring `Retry-After` under a 120 s sleep budget and 300 s deadline before **one** failure is counted. Four of those is a real outage. **Leave at 4.** |
| `MAX_ATTEMPTS = 3` | Fine, but `claim_tasks` bumps `attempts` *at claim time, before routing*. So returning a cooling row to `pending` without refunding only lengthens the fuse from 1 claim to 3. **The refund is mandatory, not optional.** |
| `routing.generator` = 1 ref | **Deliberate - do not widen.** `yaml:890-909` is an operator directive of 2026-08-28: the sole-generator flip is "AN ALLOCATION DECISION, NOT A YIELD DECISION" (cerebras metered, ~$4.63, directed to judging). Widening also breaks the single-teacher cut that `off_teacher`/`--require-generator` protects, and the gates (`think_max 4500`) are fitted on deepseek. |
| `rpm: 8` (bai) | Chosen with evidence (`yaml:353`: a valid call came back 429 at 10). Next lever if trips stay frequent. |
| cron `0 */4` (`data-worker.yml`) | **Measured, left alone.** Fires are late, not dropped: mean interval 6.34 h over 38 h on the old 6 h period. A run is 5.45 h and the concurrency group holds one running + one waiting, so at `*/4` the pending slot already self-fills. Changing it would be fitting +/-3 h of jitter. (F26) |
| `PREAMBLE_MIN_CHARS = 1000` (`decontaminate.py:1214`) | **Chosen from a bimodal histogram**, not picked: 234 rows at exactly 0 preamble, 13 between 1 and 1,200, 486 above 1,201. Any value in the gap gives the same answer. (F28) |
| `TRIMMED_MIN_CHARS = 480` (`decontaminate.py:1220`) | **Derived, not chosen.** `answer_min` is 120 tokens and `gates._est_tokens` is `len // 4`, so 480 chars *is* the gate floor - the trim cannot push a row under a bar it already cleared. If either moves, this must move with it. |
| `DEFAULT_AUDIT_SAMPLE = 0.05` | Reasoned, not arbitrary (`judge.py:181-189`). **Open question:** is 5% enough evidence at ~35-40 judged rows/day? Out of scope tonight. |

## 7. Decision log

1. **Ship profile `v1.0-MVP`.** CI already hardcodes it (`actions_worker.py:94`)
   while `assembly.default_profile` says `v1.1-full` (`yaml:119`). MVP needs
   ~3,617 accepted synthesis rows vs ~12,600. **The "align config" half was
   tried on 2026-08-31 and REVERTED** - 33 gate fixtures encode the default
   profile in their own proportions, so the flip needs a fixture change first
   (F47). The SHIP decision stands; only the config default is unchanged.
2. **Cancelled run #8** (2026-08-31T01:38Z). Pre-guard sha, claiming nothing,
   holding the `data-build` concurrency group; holds no leases (claimed=0), and
   leases expire on a 900 s clock regardless. Cancelling loses nothing.
3. **Cooldown 300 s -> 60 s, but secondary.** Throughput fix, not the bug fix;
   alone it would only shrink the crater 5x and leave the shredder armed.
4. **Free fleet only.** cerebras answers 402 so it is free in practice; leave it
   routed. No `usd_cap` exists - if that account is topped up, judging would
   spend silently. Noted, not changed.
5. **Cheap subagents** for mechanical work (test writing, transition forensics,
   doc drafting); the main session keeps root-causing and verification.
6. **Dispatched a worker on `0227bdd` at 06:20Z, replacing the queued
   `fb611f5` fire.** The concurrency group holds one running plus one waiting,
   so a dispatch REPLACES the waiting run rather than adding to it: the
   scheduled run went `cancelled` at 06:20:59Z having never started, so nothing
   was lost, and the next ~5.5h of generation gets F32's 13.7% instead of
   waiting a further cycle for it. Confirmed the cancellation was the
   concurrency swap and not a failure before recording this.

## 8. Open risks

- **PRIVATE STORAGE RUNS OUT ~2026-09-03.** 55.00 GB of a 100 GB ACCOUNT-WIDE free
  quota (baton 38.83 + 16.17 in checkpoint repos), +14.8 GB/day, and past it the Hub
  refuses uploads - every run then loses its whole 5h15m of generation. Squash the
  baton by ~09-01 (36 h to reflect). (F48)
- **The breaker will trip again** - 14% of calls refused on 08-29. Steps 1/1b
  make that a bounded pause, not a massacre, but throughput is the next
  constraint. Measure in Step 4; the lever after that is `rpm`, not a second ref.
- **Corpus is the long pole.** ~337 accepted synthesis vs ~3,617 for MVP.
  Tonight makes the pipeline *run*; it cannot finish the corpus.
- **`--replay-nothink-share` is an unmade operator decision** - it sets whether
  the no-think budget comes from chit-chat or from raw legal rows, and with it
  whether a generated row buys ~2.05 or ~2.88 corpus rows. Not a gate.
- **`stats` returns 1 on RED and breaks the chain**, so a RED corpus means
  `push.py` never runs and nothing reaches the hub. `build_manifest.json` has
  never existed. Shipping is gated on `mix`/`trace`/`empty_think`, all downstream
  of synthesis volume.
- The `tests` workflow failed its first three runs (2026-08-30) and passed #4-#6.
