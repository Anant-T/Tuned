# gpt-oss under the prompt ceiling: the arms ran, the provider did not

2026-08-27. Both arms of the gpt-oss floor A/B were built, paired, fenced and
run back to back in one sitting. **Neither arm produced a single generation.**
`cerebras/gpt-oss-120b` — the generator this measurement exists to grade, and
the generator Task 1 put back in the lead of `routing.generator` the same day —
answers **HTTP 402 `payment_required`** to every call.

**The five pre-registered measurements are 0 PASS / 0 FAIL / 5 NOT MEASURED.**
The question this report was commissioned to answer is still open, and this
report does not answer it in either direction.

---

## 1. What this was supposed to measure

On 2026-08-27, commit `286fd3a` gave the reasoning band a ceiling in all
fourteen generator templates. Doing so deleted a sentence added on 2026-08-18
for the **opposite** purpose:

> …your own working beforehand runs as long as the problem deserves and is
> never a retelling of the materials. Work the point through fully — 450 to 700
> words of deliberation **is normal for a matter of any substance**.

replaced by, in fourteen different surface forms:

> …your own working beforehand is never a retelling of the materials. Work the
> point through fully — 450 to 700 words of deliberation, **and 700 is a
> ceiling you do not cross**.

That sentence existed to push gpt-oss traces **up**, because gpt-oss's measured
failure mode is the floor, not the roof. Counted over the 1,281 gpt-oss
generations already in the live store — **context only, not this report's
control**:

| | value |
|---|---|
| median `think_est` | 620 tokens |
| `think < think_min` | 381/1,281 = **29.7%** |
| `think > think_max` (3000) | 65/1,281 = 5.1% |

The edit was measured on **deepseek** (`docs/reports/2026-08-27-generator-prompt-length-fix.md`),
where it came out inert. It has never been measured on gpt-oss. Task 1 then made
gpt-oss the lead generator again. That combination — the generator whose
dominant failure is `think < think_min`, reading templates that just lost the
sentence written to raise it — is what these two arms exist to grade.

## 2. What actually happened

Every call in both arms failed. The provider's own words, returned to a bare
one-shot probe of 16 max tokens at temperature 0 sent three minutes after the
run, through `httpx` (the client `providers.ChatClient` uses):

```
2026-08-27T12:42:54Z  cerebras/gpt-oss-120b   HTTP 402
2026-08-27T12:42:54Z  cerebras/gemma-4-31b    HTTP 402
{"message":"Payment required to access this resource. Visit your billing tab.",
 "type":"payment_required_error","param":"quota","code":"payment_required"}
```

That is the same status the pipeline saw, from a payload this repo did not
build, on **both** cerebras models. It is an account state, not a routing bug,
not a config error and not a transient: two independent observations three
minutes apart agree, and the two models fail identically.

The cerebras block in `configs/data_law_v1.yaml` records the account as
"header-verified 2026-08-15 on a funded account ($5 credit)". That credit is
now spent.

### Both arms, as run

| | control | treatment |
|---|---|---|
| tasks planned | 40 | 40 |
| call attempts (`run_event.generation_error`) | 90 | 90 |
| generations recorded | **0** | **0** |
| ledger | `cerebras/gpt-oss-120b` req=6, tok=0+0 | `cerebras/gpt-oss-120b` req=6, tok=0+0 |
| task states | `gen_unroutable` 30, `pending` 10 | `gen_unroutable` 30, `pending` 10 |

| status | attempts (both arms) | error |
|---|---|---|
| 402 | 12 | `cerebras/gpt-oss-120b: HTTP 402 (provider unusable, failing over)` |
| — | 168 | `role 'generator': no eligible model (skipped: cooling)` |

The twelve 402s cooled the single generator ref; every later attempt found the
pool empty and parked the task. No tokens were sent, nothing was spent, and
nothing was judged.

## 3. The five pass lines, unmoved

The pass lines and the population were fixed before the run and are reproduced
here exactly as registered. Nothing about the outcome moved them.

| # | measurement | pass line | control (pre-edit) | treatment (shipped) | verdict |
|---|---|---|---|---|---|
| 1 | `think < think_min` breach rate | **treatment <= control + 5pp** | n=0 | **n=0** | **NOT MEASURED** |
| 2 | median trace words | **treatment >= 400 (absolute)** | n=0 | **n=0** | **NOT MEASURED** |
| 3 | `length_band` pass rate | **treatment >= control − 5pp** | n=0 | **n=0** | **NOT MEASURED** |
| 4 | `self_verification` pass rate | **treatment >= control − 5pp** | n=0 | **n=0** | **NOT MEASURED** |
| 5 | all generations `gpt-oss-120b`; `$0` ledgered | **hard** | 0 generations; openai requests 0 | **0 generations; openai requests 0** | **NOT MEASURED** |

**0 PASS / 0 FAIL / 5 NOT MEASURED.**

Measurement 5 is **NOT MEASURED and not PASS**, and the distinction is the
whole point of writing it down. An empty population satisfies "every generation
was `gpt-oss-120b`" vacuously and "nothing was ledgered against openai"
trivially. Reporting a green line there would be the same species of tidy story
this project has already had to retract twice this week. `report_ab.py` renders
NOT MEASURED whenever either arm's population is empty, so the vacuous pass is
not available to a future run either.

## 4. What the run does establish

Everything except the finding. The apparatus is built, verified and paired, and
the numbers below come from the arm stores rather than from intent.

**The control really reads the pre-edit templates.** This is the failure mode
that would have been hardest to catch: `prompt_registry._template_path` falls
back to the packaged `prompts/` directory for any id the overlay does not carry,
so a control missing a template reads the **edited** one and produces a perfect
null that looks like good news.

- 14 files extracted with `git show f499372:<path>`, byte-identical to the git
  blobs.
- **14/14** contain `is normal for a matter of any substance`.
- **0/14** contain any of the fourteen ceiling phrasings.
- All 14 overlay SHA-12s differ from their shipped counterparts.
- **40/40** control tasks carry a `prompt_sha` matching a pre-edit overlay file;
  **0/40** match a shipped file. The treatment is the exact mirror: 0/40 overlay,
  40/40 shipped. `prompt_sha` is stamped at plan time from the bytes actually
  loaded, so this is the store's own evidence, not an inference from the config.

The overlay carries **only the 14 generator templates**. `prompt_registry`
does not require the judge and probe templates to be present — `all_ids()`
enumerates the package directory and `_template_path` falls through per id — so
`judge_pointwise_v1`, `judge_tiebreak_v1` and `probe_answer_v1` were left out
rather than copied. They were never touched by `286fd3a`, so a copy would have
been identical anyway.

**The arms are a matched pair.** Identical seed pools (600 vs 600), identical
task plans on `(task_id, seed_id, arm, task_type, prompt_id)` (40 vs 40), and
`prompt_sha` differing on 40/40 — which is the intended difference and the only
one. The two config files are line-for-line identical below their headers once
`workdir:` and `prompt_overlay:` are removed; a test asserts it.

**Both arms ran in one sitting.** The control's last call attempt was
`2026-08-27T12:40:04.355Z`; the treatment's first was `2026-08-27T12:40:05.464Z`.
**Inter-arm gap: 1 second.** They ran from a single script with no operator step
between them, which is the direct lesson of the deepseek A/B — its arms ran
13h41m apart, and because several upstreams sit behind one model id with no
provider-side identifier recorded anywhere, every pooled between-arm number
from that run is uninterpretable. That specific confound is not available here.

**The single-ref generator fence held, and can be seen holding.** The live
`routing.generator` list is `[cerebras/gpt-oss-120b, bai/deepseek-v4-flash,
lightning/lightning-ai/gpt-oss-120b]`. Under the live config, twelve 402s would
have failed over to deepseek and this would have quietly become a **deepseek**
A/B — a different model family with roughly 3.6× the median trace length,
answering a question nobody asked, under a report titled gpt-oss. Both arms pin
the single ref, so the failure surfaced as 30 `gen_unroutable` tasks instead.
The arm is unmeasured, which is the correct outcome; it is not mismeasured.

**The `$0` fence holds on both arm configs, verified both ways.** With the
shipped `usd_cap: 0.0` plus prices, `budget_ok('openai', 'gpt-5-mini', 100000)`
and `…'gpt-5-nano'…` both return `False` on a fresh ledger. With the two price
keys stripped and the cap left in place, both return `True` — the cap alone
blocks nothing, exactly as `generate._usd_per_1m` returning `0.0` for a missing
price predicts. Zero openai requests on either ledger.

**The live control store was never opened for write.** Opened read-only through
`file:…?mode=ro` URIs throughout, and fingerprinted before and after:

```
before   554532864  1787309490  sha256 2ea51e4c996273fbee6d79ee1d632b6677c8752d50cb9f45258370f07fcc8f48
after    554532864  1787309490  sha256 2ea51e4c996273fbee6d79ee1d632b6677c8752d50cb9f45258370f07fcc8f48
```

Size, mtime and full content hash identical.

## 5. What this means for the prompt edit

**Nothing. That is the finding, and it should not be dressed up as anything
else.**

The edit is not vindicated by this run and it is not condemned by it. The
specific worry that prompted the measurement — that removing the sentence which
existed to raise gpt-oss's traces would push a generator already breaching
`think < think_min` at 29.7% further below the floor — is exactly as live now as
it was this morning. No result here bears on it in either direction, and no
partial reading of a 0/0 table should be offered as though it did.

Three things follow, and only these three:

1. **The edit survives on the same ground it stood on this morning, and no
   narrower or broader.** It is measured inert on deepseek and it removed a real
   internal contradiction (a paragraph that told the model to run "as long as
   the problem deserves" and then handed it a 700-word band). Nothing on this
   branch has ever measured a **benefit** from it, and this run adds nothing.
2. **The gpt-oss risk is unretired.** Whatever is decided about the demotion or
   the ceiling should be decided knowing that the combination now in production
   — gpt-oss leading, ceiling-carrying templates — has never been observed.
3. **In practice, production is not running that combination anyway**, because
   the lead ref is 402. See below.

## 6. The thing this run did find, which nobody asked for

`cerebras/gemma-4-31b` returns the same 402. It is **judge slot B** in the live
`routing.judge` and third in `routing.tiebreak`.

This is a reading of the shipped config against the measured 402, not a judged
run — no judge was dispatched in either arm — but it is a short reading:

- Generation: `routing.generator` ref 1 is 402, so the live config fails over to
  ref 2, `bai/deepseek-v4-flash`. **Production is currently generating on
  deepseek regardless of what Task 1 decided this morning.**
- Judging a deepseek row: `family_separation` excludes only `{deepseek}`. The
  judge list is `[groq/qwen, cerebras/gemma (402), openai/gpt-5-mini,
  openai/gpt-5-nano]`. Slot A is qwen. Slot B's first candidate is dead, so slot
  B's next candidate is **`openai/gpt-5-mini`** — a paid model.

That is precisely the slot-B pool gap the 2026-08-23 live drain stalled in with
34 `judge_error`, and precisely the failover Task 1's `usd_cap: 0.0` + prices
fence was written against — described in the config as a thing that *could*
happen. It is the state of the account today. The fence added this morning is
the only thing standing between a live drain and a paid judge, and it is doing
that job now rather than hypothetically.

**This is the operator item that should outrank the prompt question.** Two
independent things are needed before either can move: cerebras funding (or a
replacement free provider), and a decision about judge slot B while gemma is
down.

## 7. How to finish the measurement

Nothing needs rebuilding. Both arm stores hold the same 600 seeds and the same
40 paired tasks, and the parked tasks reopen in place:

```bash
for arm in ctl new; do
  ./.venv/Scripts/python.exe -m tuned.data.tasks \
      --config configs/data_law_v1_exp_gptoss_$arm.yaml --reopen gen_unroutable
done
# then both arms back to back, control first, in ONE sitting:
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tuned.data.generate \
    --config configs/data_law_v1_exp_gptoss_ctl.yaml --n-workers 3 --max-batches 30
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m tuned.data.generate \
    --config configs/data_law_v1_exp_gptoss_new.yaml --n-workers 3 --max-batches 30
./.venv/Scripts/python.exe data/build/exp_gptoss_new/out/report_ab.py
```

Preconditions, all of which must hold before the re-run is worth starting:

- `cerebras/gpt-oss-120b` answers 200 to a one-shot probe. Probe **first**; the
  arms burn 180 pointless call attempts otherwise, as they did here.
- Do not substitute a provider to get around the 402. The only other ref serving
  a 120b is `lightning/lightning-ai/gpt-oss-120b`, which is **paid** and records
  a different model id — it would trip the arm's own hard measurement 5 and
  spend money on a run whose entire framing is a `$0` fence. It was not done.
- Re-run both arms. Do not pair a fresh treatment against a stale control:
  several upstreams can sit behind one model id and none of them is recorded,
  which is what made the deepseek A/B's between-arm numbers uninterpretable.
- The five pass lines above do not move.

## 8. Files

| | |
|---|---|
| control config | `configs/data_law_v1_exp_gptoss_ctl.yaml` |
| treatment config | `configs/data_law_v1_exp_gptoss_new.yaml` |
| pre-edit templates | `data/build/exp_gptoss_ctl/prompts_preedit/` (14 files, from `f499372`) |
| measurement script | `data/build/exp_gptoss_new/out/report_ab.py` (uncommitted; `data/` is gitignored) |
| generated numbers | `data/build/exp_gptoss_new/out/report_body.md` |
| fence verification | `data/build/exp_gptoss_{ctl,new}/out/fence_check_arms.txt` |
| provider probe | `data/build/exp_gptoss_ctl/out/cerebras_probe.txt` |
| run timeline | `data/build/exp_gptoss_ctl/out/run_timeline.txt` |
| live-store fingerprints | `data/build/exp_gptoss_ctl/out/live_stat_{before,after}.txt`, `live_sha_before.txt`, `live_after_full.txt` |
