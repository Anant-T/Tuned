"""Offline doubles for the pipeline-worker tests (tasks/generate/judge/verify).

Not collected by pytest (the name does not match test_*), imported by the
four worker test modules from the tests directory itself.

Nothing here dials out, sleeps, or touches httpx: FakeRouter implements the
two methods the workers actually use - pick() and complete() - over the REAL
build config, so routing preference order, model families and context limits
are the shipped ones and a test that asserts "the mistral judge was excluded
on context length" is asserting something true about the real pool.

That is also why the fixtures below ADD models rather than assume them: the
pool changes (glm was retired archived on 2026-08-18), and a rule that needs
a shape the config no longer carries has to supply it or it stops measuring
anything. See cfg_with_extra_judge.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from tuned.data.config import BuildConfig, ModelCfg, ModelRef, ProviderCfg, load_build_config
from tuned.data.providers import TRANSIENT_SKIPS, ChatResponse, ProviderError
from tuned.data.store import Store

DATA_CONFIG = Path(__file__).parent.parent / "configs" / "data_law_v1.yaml"

_CFG: BuildConfig | None = None


def build_cfg() -> BuildConfig:
    global _CFG
    if _CFG is None:
        _CFG = load_build_config(DATA_CONFIG, allow_unpinned=True)
    return _CFG


# --------------------------------------------------------------------------
# Content that passes (or fails, on purpose) the real gates.
# --------------------------------------------------------------------------

SEED_TEXT = (
    "In Anwar Ali v. State, (2008) 1 SCC 1, the Supreme Court held that a conviction may "
    "rest on circumstantial evidence only where the chain of circumstances excludes every "
    "hypothesis except guilt. The trial court relied on a recovery at the instance of the "
    "accused and on one eye witness who deposed four days later."
)

# Distinct from SEED_TEXT so default test seeds remain statute-QA eligible
# after the section_text=source fallback is removed.
STATUTE_SECTION_TEXT = (
    "Section 34. When a criminal act is done by several persons in furtherance "
    "of the common intention of all, each of such persons is liable for that act "
    "in the same manner as if it were done by him alone."
)

# ~3,000 chars of trace: over think_min (500 est tokens), under think_max,
# carrying a VERIFICATION_CUES phrase and no IRAC heading, and sharing no
# 30-char run with SEED_TEXT (the verbatim gate).
CLEAN_THINK = (
    "I start with what actually has to be decided for my client, and not with what is easiest "
    "to say. The complaint alleges one thing; the papers before me establish rather less than "
    "that, and the gap between the two is where this matter will be won or lost. So I take the "
    "ingredients one at a time and ask what each of them needs before it can be made out, and "
    "whether anything here supplies it. "
) * 4 + (
    "Let me check that against the dates, because a chronology settled quickly is a chronology "
    "settled badly. If the sequence runs the way I have assumed, the inference is available; if "
    "it does not, the inference collapses and with it most of the case against my client. I am "
    "not certain how a court would treat the delay, and I say so rather than write around it. "
) * 4

CLEAN_ANSWER = (
    "Issue\nWhether the material now on record can sustain the case advanced against the client.\n\n"
    "Rule\nA conviction resting on circumstance requires a chain complete enough to exclude every "
    "hypothesis but guilt, and testimony delayed without explanation is weighed with care.\n\n"
    "Application\nHere the chain has a link missing at its centre, and the delay in the deposition "
    "is unexplained on these papers, so the material falls short of that standard.\n\n"
    "Conclusion\nThe case as it stands is defensible, and the client should be advised to press "
    "the gap rather than to negotiate around it."
)

# A citation-SHAPED string in an unmodelled reporter: the suspect channel
# catches it with or without an index, and citations is a PERMANENT gate.
FABRICATED_SUSPECT = "(2019) 3 KLT 45"
FABRICATED_ANSWER = CLEAN_ANSWER.replace(
    "is weighed with care.",
    f"is weighed with care, as held in {FABRICATED_SUSPECT}.",
)

# A citation the index models but does not contain, and that the grounding
# text does not carry: invisible without an index, novel with one.
NOVEL_WITH_INDEX = "(2011) 4 SCC 707"
NOVEL_ANSWER = CLEAN_ANSWER.replace(
    "is weighed with care.",
    f"is weighed with care: see {NOVEL_WITH_INDEX}.",
)


def chat_response(
    text: str = CLEAN_ANSWER,
    reasoning: str | None = CLEAN_THINK,
    *,
    prompt_tokens: int = 900,
    completion_tokens: int = 800,
    finish_reason: str = "stop",
    status: int = 200,
) -> ChatResponse:
    return ChatResponse(
        text=text,
        reasoning=reasoning,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        latency_ms=42,
        status=status,
        raw={"usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}},
    )


def judge_reply(grounding=5, validity=5, coverage=5, rationale="Well grounded.", *,
                wrapper: str = "bare", names: str = "short") -> ChatResponse:
    """A judge response with the scores requested, in one of the shapes the
    real judges emit: bare JSON, fenced JSON, or JSON buried in prose; under
    the short axis names or the rubric's long ones."""
    keys = (
        ("grounding", "validity", "coverage")
        if names == "short"
        else ("grounding_faithfulness", "reasoning_validity", "issue_coverage")
    )
    body = (
        f'{{"{keys[0]}": {grounding}, "{keys[1]}": {validity}, "{keys[2]}": {coverage}, '
        f'"rationale": "{rationale}"}}'
    )
    if wrapper == "fenced":
        body = f"```json\n{body}\n```"
    elif wrapper == "prose":
        body = f"Here is my assessment of the work.\n\n{body}\n\nThat is my verdict."
    return chat_response(text=body, reasoning=None, prompt_tokens=1200, completion_tokens=60)


# --------------------------------------------------------------------------
# The router double.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Picked:
    ref: ModelRef
    provider_cfg: ProviderCfg
    model_cfg: ModelCfg


class FakeRouter:
    """providers.Router's surface, minus the transport.

    `script` maps a role to the replies it should hand back, in order. An
    entry may be a ChatResponse, an Exception (raised), or a callable taking
    (ref, messages) and returning a ChatResponse. The LAST entry repeats
    forever, so a one-entry script serves a whole batch.

    Family exclusion and the routing preference order are honoured for real:
    complete() walks cfg.routing_refs(role) and takes the first ref whose
    family is not excluded, exactly as Router.eligible does, and raises
    ProviderError when nothing is left. That is what makes the family
    separation and context-routing assertions meaningful.

    EVERY skip reason the real Router can report is modelled, not just
    family exclusion: `missing_keys` (provider names with no key in the
    environment), `cooling` and `over_budget` (ref strings, "provider/model").
    The resulting ProviderError carries the same `skipped` set and the same
    `retryable` classification the real Router computes, because "why was
    nothing eligible" is what decides whether a worker re-queues the row or
    parks it - and a double that only ever says "family-excluded" cannot
    exercise that decision at all.
    """

    def __init__(
        self,
        cfg: BuildConfig,
        script: dict | None = None,
        *,
        missing_keys: set[str] | None = None,
        cooling: set[str] | None = None,
        over_budget: set[str] | None = None,
    ):
        self.cfg = cfg
        self.script = {role: list(items) for role, items in (script or {}).items()}
        self.missing_keys = set(missing_keys or ())
        self.cooling = set(cooling or ())
        self.over_budget = set(over_budget or ())
        self.calls: list[dict] = []
        self.closed = False

    # -- selection

    def _skip_reason(self, ref: ModelRef, model: ModelCfg, exclude_families) -> str | None:
        """Same order the real Router.eligible checks in."""
        label = f"{ref.provider}/{ref.model}"
        if model.family in set(exclude_families or ()):
            return "family-excluded"
        if label in self.cooling:
            return "cooling"
        if ref.provider in self.missing_keys:
            return "missing-key"
        if label in self.over_budget:
            return "over-budget"
        return None

    def _eligible(self, role: str, exclude_families, skipped: set | None = None) -> Picked | None:
        for ref in self.cfg.routing_refs(role):
            provider, model = self.cfg.model_for(ref)
            reason = self._skip_reason(ref, model, exclude_families)
            if reason is not None:
                if skipped is not None:
                    skipped.add(reason)
                continue
            return Picked(ref=ref, provider_cfg=provider, model_cfg=model)
        return None

    def pick(self, role, *, est_tokens=0, exclude_families=frozenset(), skipped=None):
        return self._eligible(role, exclude_families, skipped)

    def _next(self, role: str):
        items = self.script.get(role)
        if not items:
            return chat_response()
        return items.pop(0) if len(items) > 1 else items[0]

    async def complete(
        self,
        role,
        messages,
        *,
        params=None,
        params_for_ref=None,
        max_tokens=None,
        est_tokens=0,
        exclude_families=frozenset(),
        on_attempt=None,
        prompt=None,
    ):
        skipped: set[str] = set()
        picked = self._eligible(role, exclude_families, skipped)
        # Same resolution order as the real Router: the per-ref hook is MERGED
        # over the call-wide params key by key, resolved against the ref
        # actually about to be called. It does not replace them - a double
        # that drops `params` whenever a hook is present disagrees with the
        # real Router about the thing the hook exists to do.
        sent = dict(params or {})
        if params_for_ref is not None and picked is not None:
            sent.update(params_for_ref(picked.ref, picked.model_cfg))
        self.calls.append(
            {
                "role": role,
                "messages": [dict(m) for m in messages],
                "params": sent,
                "max_tokens": max_tokens,
                "est_tokens": est_tokens,
                "exclude_families": frozenset(exclude_families or ()),
                "ref": picked.ref if picked else None,
                "prompt": prompt,
            }
        )
        if picked is None:
            # Nothing in the pool can take this call. The real Router reports
            # WHY, and classifies the error retryable only when some reason
            # lifts on its own - which is what tells a worker whether to park
            # the row or hand it back to the queue.
            reasons = ", ".join(sorted(skipped)) if skipped else "role list is empty"
            raise ProviderError(
                f"role {role!r}: no eligible model (skipped: {reasons})",
                retryable=bool(skipped & TRANSIENT_SKIPS),
                skipped=frozenset(skipped),
            )
        item = self._next(role)
        if isinstance(item, BaseException):
            # Report the attempt the way the client would before failing.
            if on_attempt is not None and getattr(item, "status", None) is not None:
                on_attempt(picked.ref, item.status, None)
            raise item
        if callable(item):
            item = item(picked.ref, messages)
        if on_attempt is not None:
            on_attempt(picked.ref, item.status, (item.raw or {}).get("usage"))
        return picked.ref, item

    async def aclose(self):
        self.closed = True

    # -- assertions helpers

    def calls_for(self, role: str) -> list[dict]:
        return [c for c in self.calls if c["role"] == role]


# --------------------------------------------------------------------------
# Store scaffolding.
# --------------------------------------------------------------------------

SOURCE_ID = "test/seeds"


def seed_rows(n: int, *, text: str = SEED_TEXT, meta=None, case_type="criminal") -> list[dict]:
    payload = {"section_text": STATUTE_SECTION_TEXT}
    if meta is not None:
        payload.update(meta)
    return [
        {
            "seed_id": f"seed{i:03d}",
            "source_id": SOURCE_ID,
            "native_id": f"case-{i}",
            "case_type": case_type,
            "code_era": "ipc",
            "text": text,
            "token_count": len(text) // 4,
            "meta_json": payload,
        }
        for i in range(n)
    ]


def open_store(tmp_path, *, n_seeds: int = 4, db_path=None, **seed_kwargs) -> Store:
    """A seeded store. `db_path` puts it where a CLI under test will look
    (build_paths(workdir).state_db) instead of beside tmp_path."""
    store = Store.open(db_path or (tmp_path / "state" / "law_v1.sqlite3"))
    store.upsert_source(SOURCE_ID, "Apache-2.0", url="https://example.test")
    if n_seeds:
        store.upsert_seeds(seed_rows(n_seeds, **seed_kwargs))
    return store


def paths_for(tmp_path):
    from tuned.data.paths import build_paths

    return build_paths(tmp_path / "build").ensure()


def cfg_without_the_paid_judges(cfg: BuildConfig) -> BuildConfig:
    """The shipped config minus the openai backstop - the pool before 2026-08-15.

    A whole family of rules here is ABOUT a judge pool that runs out: slot B
    parking a row that has already paid for slot A, an under-sized judge being
    fatal rather than a warning, `--allow-pool-gaps` having short rows left to
    run. The shipped pool no longer runs out - that is what closing the gap
    means - so those rules need a pool that does, and this is the one they were
    written against. It is not hypothetical either: it is exactly what an
    operator who has not funded OPENAI_API_KEY is running.

    Routing only, on purpose. What empties the slot is the ref not being in the
    list; leaving the provider block reachable through `cfg.model_for` keeps
    every walk the same shape it has in production.
    """
    from dataclasses import replace

    def drop(refs):
        return tuple(r for r in refs if not r.startswith("openai/"))

    patched = replace(
        cfg,
        routing=replace(
            cfg.routing, judge=drop(cfg.routing.judge), tiebreak=drop(cfg.routing.tiebreak)
        ),
    )
    # The prefix trap the config comment warns about: "groq/openai/gpt-oss-20b"
    # is a GROQ ref whose MODEL id starts with "openai/". Dropping it here would
    # quietly empty the tiebreak pool and make every rule below pass for a
    # reason nobody chose.
    assert "groq/openai/gpt-oss-20b" in patched.routing.tiebreak
    assert len(patched.routing.judge) == len(cfg.routing.judge) - 2
    return patched


def cfg_with_fourth_judge_family(cfg: BuildConfig, *, max_context: int = 131072) -> BuildConfig:
    """The shipped config plus the 32k+ fourth-family judge it is missing.

    This is the operator-side half of the round-2 judge-pool fix, in the one
    shape a test can hold: a large model in a family that is neither the
    generator's nor either existing judge's, so slot B and the tiebreak have
    somewhere to go on a long row. Used to prove that widening the pool is
    what un-parks a row, rather than any weakening of family separation.

    It lives behind GROQ_API_KEY on purpose: the fourth family is the model
    the operator is still sourcing, keys arrive piecemeal, and "the new judge
    is configured but its key has not landed" is the shape of the R3-C3
    preflight hole. `max_context` is a parameter because 16k candidates are
    common and a 16k judge cannot hold the longest row the length gate passes.
    """
    from dataclasses import replace

    extra = ModelCfg(
        id="fourth-judge",
        family="fourth",
        roles=("judge", "tiebreak"),
        limits={"rpm": 30, "tpm": 8000, "max_context": max_context, "max_output": 8192},
        params={"temperature": 0.2},
    )
    providers = tuple(
        replace(p, models=p.models + (extra,)) if p.name == "groq" else p for p in cfg.providers
    )
    return replace(
        cfg,
        providers=providers,
        routing=replace(
            cfg.routing,
            judge=cfg.routing.judge + ("groq/fourth-judge",),
            tiebreak=cfg.routing.tiebreak + ("groq/fourth-judge",),
        ),
    )


class StealsTheLease:
    """Store proxy that hands the task to another worker at a chosen call.

    Everything else is delegated untouched, so the pass under test runs
    against the real store: the point is WHEN the lease moves, not what the
    store does about it. `when` narrows the steal to one call of a repeated
    method (`log_event`, say) by inspecting its arguments.
    """

    def __init__(self, store, at: str, thief: str = "thief-worker", when=None):
        self._store = store
        self._at = at
        self._thief = thief
        self._when = when
        self.stolen = False

    def __getattr__(self, name):
        attr = getattr(self._store, name)
        if name != self._at or self.stolen:
            return attr

        def steal(*args, **kwargs):
            if self._when is None or self._when(*args, **kwargs):
                self.stolen = True
                self._store.conn.execute("UPDATE task SET claimed_by = ?", (self._thief,))
            return attr(*args, **kwargs)

        return steal


def cfg_with_split_pools(cfg: BuildConfig, *, judge_context: int, tiebreak_context: int):
    """A pool whose JUDGE role is complete and whose TIEBREAK role is not.

    One fourth-family model in the judge list only, so every judge slot fills,
    and one fifth-family model in the tiebreak list only at a size the caller
    chooses. That is what lets a test put a threshold between what the judge
    prompt requires and what the (slightly longer) tiebreak prompt requires -
    the two are sized separately, and nothing else in the config can tell them
    apart.

    EVERY PRE-EXISTING TIEBREAK IS NARROWED OUT, which is what makes
    ``tiebreak_context`` decisive rather than merely present. Until 2026-08-19
    that happened by accident: the shipped tiebreak pool was gpt-oss (removed
    by family separation on a gpt-oss row) plus gemma at a stale 8192 (removed
    on length), so the caller's model was the only one left. The gemma probe
    put it at 131k, it started satisfying the check on its own, and both
    callers of this helper went green while measuring nothing. The exclusion
    is now explicit and survives the next window correction.
    """
    from dataclasses import replace

    def model(model_id, family, role, max_context):
        return ModelCfg(
            id=model_id,
            family=family,
            roles=(role,),
            limits={"rpm": 30, "tpm": 8000, "max_context": max_context, "max_output": 8192},
            params={"temperature": 0.2},
        )

    def narrow(m):
        if "tiebreak" not in m.roles:
            return m
        return replace(m, limits={**m.limits, "max_context": 1024})

    extra = (
        model("fourth-judge", "fourth", "judge", judge_context),
        model("fifth-tiebreak", "fifth", "tiebreak", tiebreak_context),
    )
    providers = tuple(
        replace(
            p,
            models=tuple(narrow(m) for m in p.models) + (extra if p.name == "groq" else ()),
        )
        for p in cfg.providers
    )
    return replace(
        cfg,
        providers=providers,
        routing=replace(
            cfg.routing,
            judge=cfg.routing.judge + ("groq/fourth-judge",),
            tiebreak=cfg.routing.tiebreak + ("groq/fifth-tiebreak",),
        ),
    )


@contextmanager
def judge_prompt_overlay_with_pinned_tiebreak_gap(*, tiebreak_extra_chars: int = 4000):
    """Arms a temporary prompt overlay in which judge_tiebreak_v1 renders
    ``tiebreak_extra_chars`` of inert filler LARGER than judge_pointwise_v1,
    and judge_pointwise_v1 is otherwise byte-identical to the shipped
    template.

    Several preflight/pool_gaps tests need the tiebreak prompt to size larger
    than the pointwise judge prompt - that is what lets one synthetic model
    sit between the two thresholds and prove the plumbing sizes the tiebreak
    slot from its OWN prompt rather than reusing the judge's. Until Task 2 of
    the 2026-08-24 judge-calibration plan that gap was a four-token ACCIDENT
    of the shipped prose (judge_tiebreak_v1 happened to render a few tokens
    longer than judge_pointwise_v1). Splitting the grounding rubric's bands
    lengthened judge_pointwise_v1 and inverted that ordering PERMANENTLY:
    providers.required_context is needed_tokens * CONTEXT_SAFETY_MARGIN with
    no rounding to absorb a content edit, so nothing about the next prompt
    edit restores it either way by accident.

    This overlay makes the direction and size of the gap an explicit,
    test-controlled fact instead of prose the two templates happen to carry:
    the default padding is ~1,000 tokens, dwarfing anything a plausible rubric
    edit could move either template by. Everything else stays real - the
    renderer, the token estimator, and the per-generator-window narrowing in
    generate.judge_tokens_for_generator_window all run unmodified, reading
    these bytes exactly as they would read the shipped ones.
    """
    import tempfile

    from tuned.data import prompt_registry as reg

    with tempfile.TemporaryDirectory() as tmp:
        overlay = Path(tmp)
        for prompt_id in ("judge_pointwise_v1", "judge_tiebreak_v1"):
            text = (reg.PROMPTS_DIR / f"{prompt_id}.md").read_text(encoding="utf-8")
            if prompt_id == "judge_tiebreak_v1":
                filler = " Filler." * (tiebreak_extra_chars // 8)
                text = text.rstrip("\n") + "\n" + filler + "\n"
            (overlay / f"{prompt_id}.md").write_text(text, encoding="utf-8", newline="\n")
        reg.set_overlay(overlay)
        try:
            yield overlay
        finally:
            reg.set_overlay(None)


def cfg_with_extra_judge(
    cfg: BuildConfig, *, provider: str, family: str, model_id: str, max_context: int
) -> BuildConfig:
    """The shipped config plus one judge, in a family and behind a key of the
    caller's choosing.

    THE SHAPE THIS SUPPLIES used to be in the config: cerebras/zai-glm-4.7, a
    keyed 8k judge in a third family. It was retired on 2026-08-18 (archived
    upstream, 404 on every call), and two properties here are only expressible
    with one in the pool -

      * "this gap is CONTEXT-shaped and that one is KEY-shaped" needs a family
        that serves the short rows and not the long ones;
      * a slot B that can be filled AT ALL needs a second KEYED judge family,
        because slot B excludes slot A's.

    Without one, every gap in a partially-keyed pool classifies unservable and
    the distinction the rules are about disappears - which reads as the tests
    passing while measuring one case instead of two.
    """
    from dataclasses import replace

    extra = ModelCfg(
        id=model_id,
        family=family,
        roles=("judge",),
        limits={"rpm": 30, "tpm": 8000, "max_context": max_context, "max_output": 4096},
        params={"temperature": 0.2},
    )
    providers = tuple(
        replace(p, models=p.models + (extra,)) if p.name == provider else p for p in cfg.providers
    )
    assert any(model_id in (m.id for m in p.models) for p in providers), provider
    return replace(
        cfg,
        providers=providers,
        routing=replace(cfg.routing, judge=cfg.routing.judge + (f"{provider}/{model_id}",)),
    )


# The family the two-generator fixture supplies. Named, not literal, because
# tests assert gap tuples and refusal strings that carry it.
SECOND_GENERATOR_FAMILY = "secondgen"
SECOND_GENERATOR_REF = "cerebras/second-generator"
# The window the fixture's second generator declares. Mirrors what the departed
# mistral-small judge carried, so the pool shape the older tests were written
# against is preserved.
#
# ITS VALUE IS NOT LOAD-BEARING, and the previous comment here claimed it was.
# It said tests about "each generator family is checked at the size its OWN
# window permits" assert nothing without it - and the review disproved that by
# setting it to 131072 and watching all 3101 tests stay green. Measured: the
# per-family judge sizer returns the FLAT worst case (23,729) for every window
# at or above 16,384, and only narrows below that - 12,000 gives 22,151 and
# 8,192 gives 19,104. At 32,000 it narrows nothing and never did.
#
# What actually exercises the narrowing is _narrow_generator's 8192 in the
# generate and providers suites. The curve itself is pinned by
# test_the_judge_sizer_only_narrows_below_the_flat_worst_case, so a change to
# the sizing rule is caught even though this constant does not catch it.
SECOND_GENERATOR_CONTEXT = 32000


def cfg_with_two_generator_families(cfg: BuildConfig) -> BuildConfig:
    """The shipped config with a SECOND generator family, SYNTHESISED here.

    A large body of pool/preflight tests is ABOUT the algorithm that walks
    generator families - each family checked at its own window, a gap reported
    per family, the router's key filter applied per family - and those
    properties are only observable when more than one family exists.

    IT USED TO BORROW ITS SECOND FAMILY FROM THE SHIPPED CONFIG, promoting
    mistral-small-latest to generator, and that broke twice: on 2026-08-18 when
    mistral was demoted to judge-only, and on 2026-08-19 when mistral-small
    left the build after human calibration disqualified it. The second break
    was the loud kind only by luck - `patch()` matched on model id with no
    `assert matched`, so it silently no-opped while appending an unresolvable
    ref to routing.generator.

    So it synthesises its own, the way every other fixture in this file already
    does. The model is:

      * its OWN family, so family separation has something to remove;
      * in routing.judge as well as routing.generator, because the properties
        this supports are mostly about a generation whose family must be
        excluded from judging itself - that was mistral-small's real role here;
      * under the CEREBRAS provider, which is the generator's own. Not
        cosmetic: under groq it shared a key with the qwen judge, so every test
        that withholds GROQ_API_KEY to make a JUDGE family
        key-shaped-unavailable also deleted this generator family and collapsed
        its own premise. Its key now travels with the generator it stands
        beside, and `keys` covers it because cerebras is in the shipped config.
        Tests needing an UNKEYED generator family have their own helper
        (_with_extra_generator) and always did.
    """
    from dataclasses import replace

    extra = ModelCfg(
        id="second-generator",
        family=SECOND_GENERATOR_FAMILY,
        roles=("generator", "judge"),
        limits={
            "rpm": 30,
            "tpm": 8000,
            "tpd": 200000,
            "max_context": SECOND_GENERATOR_CONTEXT,
            "max_output": 8192,
        },
        params={},
        role_params={
            "generator": {"temperature": 0.7, "top_p": 0.95},
            "judge": {"temperature": 0.2},
        },
    )
    providers = tuple(
        replace(p, models=p.models + (extra,)) if p.name == "cerebras" else p
        for p in cfg.providers
    )
    assert any(
        m.id == "second-generator" for p in providers for m in p.models
    ), "the cerebras provider is gone - this fixture has nothing to attach to"
    return replace(
        cfg,
        providers=providers,
        routing=replace(
            cfg.routing,
            generator=tuple(cfg.routing.generator) + (SECOND_GENERATOR_REF,),
            judge=tuple(cfg.routing.judge) + (SECOND_GENERATOR_REF,),
        ),
    )


# Every free judge PROMOTED into routing.judge after the two-family era: gemma
# on 2026-08-19 (it took mistral-small's seat when calibration removed that
# model), bai/deepseek-v4-flash on 2026-08-27 (it took the slot-B seat
# gemma's HTTP 402 had emptied), and groq/openai/gpt-oss-20b, same day (it
# took the slot-B seat a DEEPSEEK generation leaves empty, since separation
# excludes deepseek from judging its own rows). All three are free, all three
# sit ahead of the paid backstops, and any ONE of them alone is enough to keep
# the pool from running out.
PROMOTED_JUDGES = (
    "cerebras/gemma-4-31b", "bai/deepseek-v4-flash", "groq/openai/gpt-oss-20b",
)


def cfg_without_the_promoted_judge(cfg: BuildConfig) -> BuildConfig:
    """The judge pool as it was before the free judges were promoted into it.

    A family of rules is ABOUT a judge pool that RUNS OUT: slot B parking a row
    that has already paid for slot A, and the re-open that must not re-buy it.
    Those need a pool with one usable family besides the generator's, and until
    2026-08-19 the shipped config supplied one by coincidence.

    Every promotion since then has widened it, so this drops ALL of them rather
    than the one that happened to be newest when it was written - see
    PROMOTED_JUDGES. Leaving a later arrival behind does NOT fail quietly, and
    it is worth being exact about that: the pool would still have a spare
    family, slot B would fill, and the rules about slot B running out go RED -
    measured, on the deepseek promotion, at 14 tests across two modules.

    The cost is misdirection, not a false green. Those failures surface at
    assertions about parking, re-queueing and pool gaps - several layers from
    the cause - so they read as a regression in judge.py or the Router when
    what actually happened is that this fixture stopped constructing the
    condition its own name promises. Widening it here is what keeps the next
    promotion from being debugged in the wrong file.

    Composed with cfg_without_the_paid_judges this restores the shape those
    rules were written against, by ROUTING only: the dropped models keep their
    tiebreak seats and their provider blocks, so every other walk has the shape
    it has in production.
    """
    from dataclasses import replace

    judge = tuple(r for r in cfg.routing.judge if r not in PROMOTED_JUDGES)
    assert len(judge) == len(cfg.routing.judge) - len(PROMOTED_JUDGES), (
        f"not all of {list(PROMOTED_JUDGES)} are in routing.judge - this "
        "fixture describes a pool that no longer exists"
    )
    return replace(cfg, routing=replace(cfg.routing, judge=judge))


# The FREE tiebreak refs that survive separation on a gpt-oss row judged by
# qwen and gemma - i.e. the ones whose family is in none of {gpt-oss, qwen,
# gemma}. mistral-large-latest was given that seat on 2026-08-19;
# bai/deepseek-v4-flash joined routing.tiebreak on 2026-08-27 with the judge
# seat. The other free refs in that list (groq/openai/gpt-oss-20b,
# cerebras/gemma-4-31b) are excluded by separation on such a row already.
FREE_TIEBREAKS = ("mistral/mistral-large-latest", "bai/deepseek-v4-flash")


def cfg_without_the_free_tiebreak(cfg: BuildConfig) -> BuildConfig:
    """A pool with NO tiebreak left for a gpt-oss row.

    On the shipped config there is one: a gpt-oss generation is judged by qwen
    and gemma, the tiebreak excludes {gpt-oss, qwen, gemma}, and
    mistral-large-latest is the family that survives - which is the whole
    point of putting it in that seat on 2026-08-19.

    Rules about what happens when NOTHING survives (judge.py's park-loudly
    path, tiebreak_unroutable_two_judge_decision -> reject) therefore need the
    condition constructed. Dropping every free survivor from routing.tiebreak
    is the smallest way to do it and leaves the judge slots untouched - EVERY
    one, not just mistral, because a survivor left behind keeps the seat filled
    and the rules that need it empty go RED rather than quietly passing. See
    FREE_TIEBREAKS, and the note there about where those failures surface: at
    the park-loudly and pool-gap assertions, which points the reader at judge.py
    instead of at this fixture.
    """
    from dataclasses import replace

    tiebreak = tuple(r for r in cfg.routing.tiebreak if r not in FREE_TIEBREAKS)
    assert len(tiebreak) == len(cfg.routing.tiebreak) - len(FREE_TIEBREAKS), (
        f"not all of {list(FREE_TIEBREAKS)} are in routing.tiebreak - this "
        "fixture describes a pool that no longer exists"
    )
    return replace(cfg, routing=replace(cfg.routing, tiebreak=tiebreak))


def cfg_with_context(cfg: BuildConfig, *, family: str, role: str, max_context: int):
    """The shipped config with one (family, role)'s context window rewritten.

    Refuses a (family, role) it did not find, which is the same discipline the
    bound on _divert_point exists for: silently returning the config unchanged
    made the caller assert against a window it believed it had narrowed, and a
    fixture that no-ops when the pool moves under it hands back a green that
    measured the shipped config instead of the one the test described.
    """
    from dataclasses import replace

    matched = 0

    def patch(model):
        nonlocal matched
        if model.family != family or role not in model.roles:
            return model
        matched += 1
        return replace(model, limits={**model.limits, "max_context": max_context})

    patched = replace(
        cfg,
        providers=tuple(
            replace(p, models=tuple(patch(m) for m in p.models)) for p in cfg.providers
        ),
    )
    assert matched, f"no model in family {family!r} serves role {role!r} - nothing was narrowed"
    return patched


def temp_config(tmp_path, *, two_generator_families: bool = False) -> str:
    """The real build config with its workdir redirected into tmp_path.

    For the CLI tests: `main()` resolves its own paths from the config, and
    the shipped one points at the operator's live data/build.

    `two_generator_families` is the file-level twin of
    cfg_with_two_generator_families. It is the same idea applied to the config
    FILE rather than to a loaded BuildConfig: the judge pool's only remaining
    hole is reachable ONLY from a mistral generation - that is what removes
    mistral from the judge pool and empties slot B - and mistral stopped being a
    generator on 2026-08-18.

    IT HAS NO CALLERS, and the correction is worth recording because the wrong
    version of this comment was believed. An earlier draft said "one CLI test
    genuinely needs it"; that was checked on 2026-08-27 and is false at HEAD and
    was false at c3e01a9 - every use of the NAME in the suite is of
    cfg_with_two_generator_families, the BuildConfig-level twin, which is heavily
    used and is not this. So this branch is dead code today. It is kept rather
    than deleted because the scenario it builds is real and awkward to
    reconstruct, and a CLI test that needs a two-family config file is a
    plausible near-term need. Anyone who deletes it is not breaking a test; and
    anyone who cites it as load-bearing should re-run the grep first.
    """
    raw = DATA_CONFIG.read_text(encoding="utf-8")
    redirected = raw.replace("workdir: data/build", f"workdir: {(tmp_path / 'build').as_posix()}")
    assert redirected != raw
    if two_generator_families:
        # The file-level twin of cfg_with_two_generator_families, and it
        # synthesises the same family for the same reason: it used to promote
        # the shipped mistral-small block, which no longer exists.
        #
        # THE ANCHOR IS A WHOLE MODEL, down to its params line. Anchoring on
        # the first lines alone inserts the new block between that model's
        # `roles` and its `limits`, which silently hands the new model's limits
        # to it and leaves the real one with none.
        anchor = (
            "      - id: gemma-4-31b\n"
            "        family: gemma\n"
            "        roles: [judge, tiebreak]\n"
            "        limits: {rpm: 5, tpm: 30000, tpd: 1000000, "
            "max_context: 131072, max_output: 4096}\n"
            "        params: {temperature: 0.2}\n"
        )
        injected = anchor + (
            "      - id: second-generator\n"
            f"        family: {SECOND_GENERATOR_FAMILY}\n"
            "        roles: [generator, judge]\n"
            "        limits: {rpm: 30, tpm: 8000, tpd: 200000, "
            f"max_context: {SECOND_GENERATOR_CONTEXT}, max_output: 8192}}\n"
            "        params: {}\n"
            "        role_params:\n"
            "          generator: {temperature: 0.7, top_p: 0.95}\n"
            "          judge: {temperature: 0.2}\n"
        )
        for old, new in (
            (anchor, injected),
            # Anchor matches routing.generator's CURRENT order - gpt-oss
            # reclaimed the lead slot from deepseek on 2026-08-27 (Task 1).
            # This branch has no callers today (see the docstring above), so
            # nothing exercises a stale anchor here and it can go quietly out
            # of sync with the live config, which is exactly what happened to
            # this same anchor once already. Keep it matched on every
            # reorder even though nothing currently runs it.
            ("  generator: [cerebras/gpt-oss-120b, bai/deepseek-v4-flash,\n"
             "              lightning/lightning-ai/gpt-oss-120b]",
             "  generator: [cerebras/gpt-oss-120b, bai/deepseek-v4-flash,\n"
             f"              lightning/lightning-ai/gpt-oss-120b, {SECOND_GENERATOR_REF}]"),
            ("  judge: [groq/qwen/qwen3.6-27b, cerebras/gemma-4-31b,",
             f"  judge: [groq/qwen/qwen3.6-27b, cerebras/gemma-4-31b, {SECOND_GENERATOR_REF},"),
        ):
            assert redirected.count(old) == 1, old
            redirected = redirected.replace(old, new)
    path = tmp_path / "cfg.yaml"
    path.write_text(redirected, encoding="utf-8")
    return str(path)


# Long enough that prompt + max_tokens needs 10,910 tokens of declared window
# (measured: 4,632 estimated prompt tokens + 4,096 max_output, times
# CONTEXT_SAFETY_MARGIN), while prompt + trace + answer still fit the
# 8192-token length band - so a test can move the routing without the length
# gate firing and masking the result.
#
# 59 REPEATS, NOT 60 - the seed itself must also clear
# tasks.seed_token_budget(cfg) (max_seq_length 8192 minus the 3,500-token
# reply reserve = 4,692), or plan_wave silently plans nothing against it and
# every test below reads as "no task was ever claimed" with no other clue why.
# At 60 repeats the seed's own token_count (len(text)//4 = 4,710) OVERSHOT
# that ceiling by 18 tokens, so plan_wave planned zero tasks against it; at 59
# it is 4,631, comfortably under, and still far above what the window check
# below needs.
#
# IT IS NO LONGER LONG FOR THE SHIPPED POOL, and that is the point of saying
# the number out loud. This constant was sized against a cerebras max_context
# of 8192; the 2026-08-19 probes put the real window at 131,072, so on the
# shipped config nothing excludes it and any test that still expected a divert
# was measuring a config that no longer exists. Tests that need "too long for
# the generator" must now SHRINK THE WINDOW rather than grow the text -
# cfg_with_context(cfg, family="gpt-oss", role="generator", max_context=8192)
# reproduces the old exclusion exactly, and refuses if the pool moves under
# it. Growing the text instead would need ~420,000 characters, which the
# length band would reject long before the router saw it.
LONG_SEED_TEXT = (SEED_TEXT + " ") * 59

# The window that makes LONG_SEED_TEXT too long, as a name rather than a
# literal sprinkled through the suite. Any value below 11,008 works; 8192 is
# used because it is the window the pilot actually ran against and the tests
# that quote it are describing that history.
NARROW_GENERATOR_CONTEXT = 8192

# A seed no model in the SHIPPED pool can hold, for the tests that are about
# the row rather than about the pool. Sized against the real widest window
# rather than narrowed by fixture, because "no generator can hold this" is
# the thing those tests assert and a fixture would make it true by
# construction.
#
# The arithmetic, so the next window change can redo it: exclusion needs
# required_context(est + max_output) > the widest declared generator window,
# which is bai's 800,000 since 2026-08-25 (was cerebras/lightning's 131,072
# before bai joined - the old 90,000-word text cleared THAT but left
# deepseek, bai's family, comfortably eligible). That needs est > 636,000 at
# CONTEXT_SAFETY_MARGIN 1.25 and max_output 4,000. Measured on the real
# rendered prompt rather than assumed from a chars/token constant - template
# overhead pushes the estimate above a flat chars/4 - 550,000 "word "
# repeats measures context_est_tokens=688,224, comfortably past 636,000, and
# still costs a test nothing but memory.
OVERSIZE_SEED_TEXT = "word " * 550_000

# Everything a transition seed must carry: the four template slots AND the
# two dates, without which check_temporal is fatally undecidable on this
# stream. The scenario names dates and posture only - no section numbers, so
# that keeping it out of the citation allow-list costs nothing.
TRANSITION_META = {
    "scenario": (
        "The offence is alleged to have been committed on 12 March 2024. The first "
        "information was recorded on 20 March 2024 and the charge-sheet was filed on "
        "4 September 2024, after the appointed day of 1 July 2024. The matter is at "
        "the stage of framing of charge."
    ),
    "old_section_text": (
        "Whoever commits theft shall be punished with imprisonment of either "
        "description for a term which may extend to three years, or with fine, or with both."
    ),
    "new_section_text": (
        "Whoever commits theft shall be punished with imprisonment of either "
        "description for a term which may extend to three years, or with fine, or with "
        "both, and in a case of second or subsequent conviction, with rigorous imprisonment."
    ),
    "savings_text": (
        "The repeal shall not affect any investigation, inquiry, trial or proceeding "
        "pending immediately before the appointed day, which shall be continued as if "
        "this enactment had not come into force."
    ),
    "offence_date": "2024-03-12",
    "proceeding_started": "2024-09-04",
}
