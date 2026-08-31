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
| 1 | Stop the shredder: wire `TRANSIENT_SKIPS` into `generate.py` + refund attempt + back off | TODO |
| 1b | Make `cooldown_s`/`breaker_threshold` configurable; set `routing.cooldown_s: 60` | TODO |
| 2 | Live 5-10 row test on a scratch DB copy; then force a breaker trip | TODO |
| 3 | Push, dispatch `data-worker`, watch the armed reopen recover ~5,190 rows | TODO |
| 4 | Measure the breaker trip rate; report (do not act) | TODO |
| 5 | Root-cause the `transition` stream's 99% reject rate (subagent) | TODO |
| 6 | Widen the queue, sized to measured yield | TODO |

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
- **`stats` returns 1 on RED and breaks the chain**, so a RED corpus means
  `push.py` never runs and nothing reaches the hub. `build_manifest.json` has
  never existed. Shipping is gated on `mix`/`trace`/`empty_think`, all downstream
  of synthesis volume.
- The `tests` workflow failed its first three runs (2026-08-30) and passed #4-#6.
