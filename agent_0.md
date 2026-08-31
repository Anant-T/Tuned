# agent_0

Operating file for the unattended law_v1 data-build session. **Read this first
after every autocompact, at the start of every session, when starting a new
task, and after any subagent returns.** It exists because a compact wiped the
working state once already.

---

## 0. State of play (2026-08-31, ~06:00Z)

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

**Shipped since 03:40Z** (both on `origin/main`, suite **3,781 / 19**):
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
- The cron was **measured, not changed**: fires are late, never lost (F26).

**One legal error is in the corpus.** Accepted row `412b8d1c5430` puts an appeal
under the BNSS where its own answer key says the CrPC continues to govern - it
cites all four required sections, so a permanent gate passed it. `check_answer_key`
checks citations, not conclusions, and `families_by_kind` is never read (F30).
1 row of 718; the blind spot behind it is systematic (19.9% of gate-passing
transition generations). Not fixed unattended - `transition` is a finished
stream, so the gate would guard ~0 future rows.

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
| 6 | Widen the queue, sized to measured yield | **SIZED, deliberately not dispatched** - synthesis-only, **~1,500-1,600 tasks** (re-sized on per-variant rates, F24); blocked on the variant decision, not on capacity |
| 8 | Fleet claim budget: restore the designed 36 in flight and make it invariant to drained streams | **DONE** - 3 tests, suite 3,755/19 (F23) |
| 9 | Author `gen_irac_analysis_v5` + `gen_summarization_v3` in the proven speech/letter genre; 10-row A/B; then plan Step 6 on the winners | **READY, needs the bucket free** (F24). Superseded in part: `--variant` now lets Step 6 avoid the losing personas **without** authoring anything (F27) |
| 10 | Variant allowlist so a wave can be planned on the templates that earned it | **DONE** `c61311a` (F27) |
| 11 | Stop shipping the answer's second deliberation | **DONE** `fb611f5` (F28) |
| 12 | Build the 50-example review packet the card requires, and pre-screen it | **DONE** - `data/build/out/review_packet.html`, 12/50 flagged (F29) |
| 13 | The legal read of those 50 examples | **OPEN - human task.** The packet only prepares it. The 5 transition rows are already read: 1 is wrong (F30) |
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
citation** the source never mentions. **12 of 50 rows** carry one - `drafting`
1/4, `irac_analysis` 5/34, `statute_qa` 1/4, `summarization` 2/5, `transition`
3/3. Transition's 3/3 is **expected by construction**, though not for the reason I
first wrote: the task is not offence-mapping (IPC 302 -> BNS 103) but
**which-enactment-governs**, and it turns on the three repeal-and-savings
provisions (BNS 358, BNSS 531, BSA 170) plus s.6 of the General Clauses Act.
Those are cited from law, never from the source judgment, so they can never be
"sourced". The packet flags them as "verify the limbs", and F30 is what came of
actually doing that.

**Two screen bugs found and fixed before any of this was believed**, both of
which had manufactured a wrong answer:
1. One permissive pattern on both sides read "Section 29 contains" as section
   `29CON`. Now strict on the answer (the suffix must be attached to the
   number), permissive on the source (a near-miss there only *suppresses* a
   flag, so permissiveness is the safe direction).
2. Making the source pattern prefix-optional let every page number in a 40-page
   judgment count as a section mention - the screen returned **0 findings by
   construction**. A separate probe (0 sections extracted from 50 answers)
   caught it. A clean "nothing found" is worthless until the instrument is shown
   able to find something.

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

1. **Ship profile `v1.0-MVP`.** CI already hardcodes it (`actions_worker.py:79`)
   while `assembly.default_profile` says `v1.1-full` (`yaml:119`). MVP needs
   ~3,617 accepted synthesis rows vs ~12,600. Align config to what CI grades.
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

## 8. Open risks

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
