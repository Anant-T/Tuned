# agent_0

Operating file for the unattended law_v1 data-build session. **Read this first
after every autocompact, at the start of every session, when starting a new
task, and after any subagent returns.** It exists because a compact wiped the
working state once already.

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
| 4 | Measure the breaker trip rate; report (do not act) | IN PROGRESS - 0 trips so far |
| 5 | Root-cause the `transition` stream's 99% reject rate | **DONE** - planner bug, fixed `708d455` |
| 6 | Widen the queue, sized to measured yield | TODO - arithmetic below |

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

## 6. Constants interrogated

| Constant | Verdict |
|---|---|
| `cooldown_s = 300.0` (`providers.py:1774`) | **Nobody chose it.** Generic `Router.__init__` default; no config key; all three `make_router` sites bare. deepseek's 429s come from a per-minute bucket (**301 of 2,151 requests on 08-29**, 14%) that refills in ~60 s. -> Step 1b |
| `breaker_threshold = 4` | Also a default, but justified: `ChatClient.complete` absorbs 6 jittered retries honouring `Retry-After` under a 120 s sleep budget and 300 s deadline before **one** failure is counted. Four of those is a real outage. **Leave at 4.** |
| `MAX_ATTEMPTS = 3` | Fine, but `claim_tasks` bumps `attempts` *at claim time, before routing*. So returning a cooling row to `pending` without refunding only lengthens the fuse from 1 claim to 3. **The refund is mandatory, not optional.** |
| `routing.generator` = 1 ref | **Deliberate - do not widen.** `yaml:890-909` is an operator directive of 2026-08-28: the sole-generator flip is "AN ALLOCATION DECISION, NOT A YIELD DECISION" (cerebras metered, ~$4.63, directed to judging). Widening also breaks the single-teacher cut that `off_teacher`/`--require-generator` protects, and the gates (`think_max 4500`) are fitted on deepseek. |
| `rpm: 8` (bai) | Chosen with evidence (`yaml:353`: a valid call came back 429 at 10). Next lever if trips stay frequent. |
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
