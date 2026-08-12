"""Offline doubles for the pipeline-worker tests (tasks/generate/judge/verify).

Not collected by pytest (the name does not match test_*), imported by the
four worker test modules from the tests directory itself.

Nothing here dials out, sleeps, or touches httpx: FakeRouter implements the
two methods the workers actually use - pick() and complete() - over the REAL
build config, so routing preference order, model families and context limits
are the shipped ones and a test that asserts "the glm judge was excluded" is
asserting something true about the real pool.
"""

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
    ):
        skipped: set[str] = set()
        picked = self._eligible(role, exclude_families, skipped)
        # Same resolution order as the real Router: per-ref params win, and
        # they are resolved against the ref actually about to be called.
        sent = (
            dict(params_for_ref(picked.ref, picked.model_cfg))
            if params_for_ref is not None and picked is not None
            else dict(params or {})
        )
        self.calls.append(
            {
                "role": role,
                "messages": [dict(m) for m in messages],
                "params": sent,
                "max_tokens": max_tokens,
                "est_tokens": est_tokens,
                "exclude_families": frozenset(exclude_families or ()),
                "ref": picked.ref if picked else None,
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
    return [
        {
            "seed_id": f"seed{i:03d}",
            "source_id": SOURCE_ID,
            "native_id": f"case-{i}",
            "case_type": case_type,
            "code_era": "ipc",
            "text": text,
            "token_count": len(text) // 4,
            "meta_json": meta,
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


def temp_config(tmp_path) -> str:
    """The real build config with its workdir redirected into tmp_path.

    For the CLI tests: `main()` resolves its own paths from the config, and
    the shipped one points at the operator's live data/build.
    """
    raw = DATA_CONFIG.read_text(encoding="utf-8")
    redirected = raw.replace("workdir: data/build", f"workdir: {(tmp_path / 'build').as_posix()}")
    assert redirected != raw
    path = tmp_path / "cfg.yaml"
    path.write_text(redirected, encoding="utf-8")
    return str(path)


# Long enough that prompt + max_tokens passes 8k (so the cerebras generator
# and the glm judge cannot hold it) while the prompt + trace + answer still
# fit the 8192-token length band - the routing has to move without the
# length gate firing and masking the result.
LONG_SEED_TEXT = (SEED_TEXT + " ") * 60

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
