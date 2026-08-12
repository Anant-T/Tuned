"""Normalize public reasoning-seed datasets into the store's `seed` table.

Zero teacher-model generation happens here: this module only reshapes rows
from a handful of public Indian-legal-reasoning datasets into the schema
`store.py` owns (see its `_SEED_COLS`) so later pipeline phases can draw
prompts/facts/reasoning off them for teacher generation. Contrast with
`curated.py`, which turns rows from a (mostly disjoint) set of public
datasets directly into training rows - this module never writes a stream
file, only seed rows.

Per-source converters are pure: given a raw row (whatever schema the
upstream HF dataset happens to use) they return a seed dict, or None if the
row fails a quality filter. Cross-row state (in-run dedup on the derived
seed_id, per-source counts against the requested limit) lives in
load_seeds, not the converters - the same split the sibling `replay.py`
uses between its row converters and `build_replay`.

seed_id is sha256(source_id + ":" + native_id-or-text)[:16] - deterministic
over the same input, so re-running load_seeds against the same raw rows is
an idempotent INSERT OR REPLACE (store.upsert_seeds), never a growing table.

datasets/pyarrow imports are lazy (inside the streaming helpers, never at
module import time), matching replay.py/smoke.py's discipline.

Build:  python -m tuned.data.seeds --config configs/data_law_v1.yaml
        [--limit-per-source N]
"""

import hashlib
import re

from tuned.data.replay import has_markup

# --------------------------------------------------------------------------
# Source identity - the same string is embedded in every seed row this
# source produces (source.source_id / seed.source_id, FK-linked) and used
# as the DB registration key in _SOURCE_INFO below. Defined once so the two
# never drift apart.
# --------------------------------------------------------------------------

PREDEX_SOURCE_ID = "L-NLProc/PredEx_Instruction-Tuning_Pred-Exp"
TATHYANYAYA_SOURCE_ID = "L-NLProc/TathyaNyaya-and-FactLegalLlama-NyayaFacts-Datasets"
INJUDGEMENTS_SOURCE_ID = "opennyaiorg/InJudgements_dataset"

SOURCE_ORDER = ("predex", "tathyanyaya", "injudgements")
DEFAULT_LIMITS = {"predex": 12178, "tathyanyaya": 8000, "injudgements": 4000}

# case_type keyword heuristic, checked most-distinctive-first so a
# constitutional writ that happens to mention "accused" in passing doesn't
# get swallowed by the (much broader) criminal bucket. Unmatched text
# defaults to "civil" per the brief.
_CASE_TYPE_PATTERNS = (
    ("constitutional", re.compile(
        r"\b(constitution|fundamental right|article\s+\d+|writ petition|"
        r"habeas corpus|mandamus|certiorari|quo warranto)\b", re.IGNORECASE,
    )),
    ("criminal", re.compile(
        r"\b(indian penal code|\bipc\b|murder|homicide|dacoity|kidnap\w*|rape|"
        r"assault|accused|convict\w*|acquit\w*|\bfir\b|criminal (?:appeal|trial)|"
        r"sentence|bail)\b", re.IGNORECASE,
    )),
    ("commercial", re.compile(
        r"\b(contract|company|arbitration|commercial|insolvency|trademark|"
        r"patent|shareholder|partnership deed|\bgst\b|goods and services tax)\b",
        re.IGNORECASE,
    )),
)


def classify_case_type(text: str) -> str:
    t = text or ""
    for case_type, pattern in _CASE_TYPE_PATTERNS:
        if pattern.search(t):
            return case_type
    return "civil"


def seed_id_for(source_id: str, native_key: str) -> str:
    """Stable id for one (source, native identifier) pair - sha256, truncated
    to 16 hex chars. Deterministic so re-normalizing the same raw row on a
    later run produces the same primary key (idempotent upsert)."""
    return hashlib.sha256(f"{source_id}:{native_key}".encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Per-source converters.
# --------------------------------------------------------------------------

def predex_seed(raw: dict) -> dict | None:
    """L-NLProc/PredEx_Instruction-Tuning_Pred-Exp (Apache-2.0, 12,178 rows).

    Verified real header (partial CSV fetch, train_pred_exp.csv):
    `Case Name,Input,Output,Label,Count,Decision_Count,text`. Input is the
    case facts/procedural context; Output is the real court reasoning and
    decision explanation. text here is built explicitly from Input+Output
    per the brief ("case facts + the real court reasoning"), NOT taken from
    the dataset's own precomputed `text` column, whose exact construction
    is undocumented.
    """
    facts = (raw.get("Input") or "").strip()
    reasoning = (raw.get("Output") or "").strip()
    if not facts or not reasoning:
        return None
    text = f"{facts}\n\n{reasoning}"
    if len(text) < 500:
        return None
    if has_markup(text):
        return None

    native_id = (raw.get("Case Name") or "").strip() or None
    return {
        "seed_id": seed_id_for(PREDEX_SOURCE_ID, native_id or text),
        "source_id": PREDEX_SOURCE_ID,
        "native_id": native_id,
        "court": None,
        "decision_date": None,
        "offence_date": None,
        "case_type": classify_case_type(text),
        "code_era": "ipc",
        "text": text,
        "token_count": len(text) // 4,
        "meta_json": {"estimator": "chars/4", "label": raw.get("Label")},
    }


def tathyanyaya_seed(raw: dict) -> dict | None:
    """L-NLProc/TathyaNyaya-and-FactLegalLlama-NyayaFacts-Datasets
    (Apache-2.0, ~25.4k rows). Same treatment as predex_seed.

    Verified real header (partial CSV fetch, train_single.csv):
    `Case Name,text,label,Reasoning` - lowercase `text`/`label` for the two
    PredEx-analogous columns, `Reasoning` (capitalized) for the extracted
    judicial-reasoning excerpt, mirroring PredEx's Input/Output split.
    """
    case_text = (raw.get("text") or "").strip()
    reasoning = (raw.get("Reasoning") or "").strip()
    if not case_text or not reasoning:
        return None
    text = f"{case_text}\n\n{reasoning}"
    if len(text) < 500:
        return None
    if has_markup(text):
        return None

    native_id = (raw.get("Case Name") or "").strip() or None
    return {
        "seed_id": seed_id_for(TATHYANYAYA_SOURCE_ID, native_id or text),
        "source_id": TATHYANYAYA_SOURCE_ID,
        "native_id": native_id,
        "court": None,
        "decision_date": None,
        "offence_date": None,
        "case_type": classify_case_type(text),
        "code_era": "ipc",
        "text": text,
        "token_count": len(text) // 4,
        "meta_json": {"estimator": "chars/4", "label": raw.get("label")},
    }


def injudgements_seed(raw: dict) -> dict | None:
    """opennyaiorg/InJudgements_dataset (Apache-2.0, ~12k full-text
    judgments, 1950-2017; gated="auto" on HF - the dataset card's
    dataset_info.features were readable without a token, the parquet rows
    were not). Confirmed columns: Titles, Court_Name, Cites, Cited_by,
    Doc_url, Text, Doc_size, Case_Type, Court_Type, Court_Name_Normalized.

    Text is kept WHOLE here (no chunking) - segment.py chunks it later by
    segment. token_count is a chars/4 estimate, noted as such in meta_json
    since a full judgment is far too long to count exactly here.
    """
    text = (raw.get("Text") or "").strip()
    if len(text) < 500:
        return None
    if has_markup(text):
        return None

    native_id = (raw.get("Doc_url") or "").strip() or None
    case_type = (raw.get("Case_Type") or "").strip().lower() or None
    court = (raw.get("Court_Name_Normalized") or raw.get("Court_Name") or "").strip() or None
    return {
        "seed_id": seed_id_for(INJUDGEMENTS_SOURCE_ID, native_id or text),
        "source_id": INJUDGEMENTS_SOURCE_ID,
        "native_id": native_id,
        "court": court,
        "decision_date": None,
        "offence_date": None,
        "case_type": case_type,
        "code_era": "ipc",
        "text": text,
        "token_count": len(text) // 4,
        "meta_json": {
            "estimator": "chars/4",
            "cites": raw.get("Cites"),
            "cited_by": raw.get("Cited_by"),
        },
    }


_CONVERTERS = {
    "predex": predex_seed,
    "tathyanyaya": tathyanyaya_seed,
    "injudgements": injudgements_seed,
}

# (source_id, license, url) per routing key - registered via upsert_source
# before that key's seeds are upserted (seed.source_id is FK-enforced).
_SOURCE_INFO = {
    "predex": (PREDEX_SOURCE_ID, "Apache-2.0", f"https://huggingface.co/datasets/{PREDEX_SOURCE_ID}"),
    "tathyanyaya": (TATHYANYAYA_SOURCE_ID, "Apache-2.0", f"https://huggingface.co/datasets/{TATHYANYAYA_SOURCE_ID}"),
    "injudgements": (INJUDGEMENTS_SOURCE_ID, "Apache-2.0", f"https://huggingface.co/datasets/{INJUDGEMENTS_SOURCE_ID}"),
}


# --------------------------------------------------------------------------
# Assembly.
# --------------------------------------------------------------------------

def load_seeds(store, cfg, sources: dict, limits: dict) -> dict:
    """Normalize each injected raw iterable into store.seed via the
    matching converter, up to that source's limit.

    `cfg` is accepted for signature parity with build_replay/build_curated
    and as a hook for future per-source config (e.g. license overrides);
    load_seeds itself does not read any field off it today.

    A source key absent from `sources` is skipped entirely (no source
    registration, no lookup of its limit) - mirrors build_replay's
    zero-count skip, so a controller can hand this a partial dict without
    needing a real iterator for sources it isn't touching this run.

    Returns per-source {"accepted", "limit", "rejected", "written"}. Rows
    are deduped within a single call by seed_id (stable across runs), so a
    genuinely repeated raw row is counted once; store.upsert_seeds itself
    makes a second full call idempotent (store.seed_count() unchanged).
    """
    stats: dict = {}
    for key in SOURCE_ORDER:
        if key not in sources:
            continue

        source_id, license_, url = _SOURCE_INFO[key]
        store.upsert_source(source_id, license_, url=url)

        converter = _CONVERTERS[key]
        limit = limits.get(key, DEFAULT_LIMITS[key])
        seen: set[str] = set()
        rows: list[dict] = []
        accepted = 0
        rejected = 0

        for raw in sources[key]:
            if accepted >= limit:
                break
            seed = converter(raw)
            if seed is None:
                rejected += 1
                continue
            if seed["seed_id"] in seen:
                rejected += 1
                continue
            seen.add(seed["seed_id"])
            rows.append(seed)
            accepted += 1

        written = store.upsert_seeds(rows)
        stats[key] = {"accepted": accepted, "limit": limit, "rejected": rejected, "written": written}
    return stats


def _stream(**load_kwargs):
    from datasets import load_dataset

    ds = load_dataset(streaming=True, **load_kwargs)
    for row in ds:
        yield row


def _real_sources() -> dict:
    import os

    hf_token = os.environ.get("HF_TOKEN")
    return {
        "predex": _stream(path=PREDEX_SOURCE_ID, split="train"),
        "tathyanyaya": _stream(path=TATHYANYAYA_SOURCE_ID, split="train"),
        # gated="auto" on HF - needs an authorized HF_TOKEN with the
        # dataset's terms accepted.
        "injudgements": _stream(path=INJUDGEMENTS_SOURCE_ID, split="train", token=hf_token),
    }


if __name__ == "__main__":
    import argparse
    import os
    import sys

    from tuned.data.config import load_build_config
    from tuned.data.paths import build_paths
    from tuned.data.store import Store

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data_law_v1.yaml")
    p.add_argument("--limit-per-source", type=int, default=None)
    args = p.parse_args()

    cfg = load_build_config(args.config)
    paths = build_paths(cfg.build.workdir).ensure()
    store = Store.open(paths.state_db)

    limits = dict(DEFAULT_LIMITS)
    if args.limit_per_source is not None:
        limits = {k: args.limit_per_source for k in DEFAULT_LIMITS}

    stats = load_seeds(store, cfg, _real_sources(), limits)

    print(f"{'source':<16}{'accepted':>10}{'limit':>10}{'rejected':>10}")
    for key in SOURCE_ORDER:
        s = stats.get(key, {"accepted": 0, "limit": limits.get(key, 0), "rejected": 0})
        print(f"{key:<16}{s['accepted']:>10}{s['limit']:>10}{s['rejected']:>10}")
    print(f"seed_count total -> {store.seed_count()} ({paths.state_db})")

    store.close()

    # Same reasoning as smoke.py/replay.py: abandoned streaming iterators can
    # leave non-daemon datasets/hf-xet threads that wedge interpreter
    # shutdown after all output is written. Skip shutdown entirely.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
