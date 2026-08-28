import ast
from pathlib import Path

import pytest

from tuned.data.seeds import (
    DEFAULT_LIMITS,
    INJUDGEMENTS_SOURCE_ID,
    PREDEX_SOURCE_ID,
    SOURCE_ORDER,
    TATHYANYAYA_SOURCE_ID,
    classify_case_type,
    injudgements_seed,
    load_seeds,
    predex_seed,
    seed_id_for,
    tathyanyaya_seed,
)
from tuned.data.store import Store

SEEDS_SRC = Path(__file__).parent.parent / "src" / "tuned" / "data" / "seeds.py"


# --------------------------------------------------------------------------
# classify_case_type
# --------------------------------------------------------------------------

def test_classify_case_type_criminal():
    assert classify_case_type("The accused was convicted under the IPC and denied bail.") == "criminal"


def test_classify_case_type_constitutional():
    assert classify_case_type("This writ petition challenges Article 21 as a fundamental right.") == "constitutional"


def test_classify_case_type_commercial():
    assert classify_case_type("The arbitration clause in the shareholder contract was disputed.") == "commercial"


def test_classify_case_type_default_civil():
    assert classify_case_type("The parties disagreed over the boundary of the property.") == "civil"


def test_classify_case_type_empty_defaults_civil():
    assert classify_case_type("") == "civil"


def test_classify_case_type_constitutional_beats_criminal_keywords():
    # "accused" (criminal) appears, but the constitutional pattern (writ
    # petition) is checked first and should win.
    text = "In this writ petition the accused's fundamental right to a fair trial under Article 21 was raised."
    assert classify_case_type(text) == "constitutional"


# --------------------------------------------------------------------------
# seed_id_for
# --------------------------------------------------------------------------

def test_seed_id_for_deterministic_and_distinct():
    a = seed_id_for("src", "native-1")
    b = seed_id_for("src", "native-1")
    c = seed_id_for("src", "native-2")
    assert a == b
    assert a != c
    assert len(a) == 16


# --------------------------------------------------------------------------
# predex_seed
# --------------------------------------------------------------------------

def _predex_raw(name="Kamlesh Vs. Union of India", facts=None, output=None, label=1):
    if facts is None:
        facts = ("The appellant was provisionally appointed and later challenged the "
                  "discontinuation of her service before the Tribunal and the High Court. " * 6)
    if output is None:
        output = ("The Court held that prolonged service alone does not entitle an "
                   "employee to regularization absent a sanctioned vacancy, and dismissed "
                   "the appeal accordingly, affirming the High Court's reasoning in full. " * 4)
    return {
        "Case Name": name,
        "Input": facts,
        "Output": output,
        "Label": label,
        "Count": 12,
        "Decision_Count": 3,
        "text": "unused precomputed column",
    }


def test_predex_seed_accept():
    seed = predex_seed(_predex_raw())
    assert seed is not None
    assert seed["source_id"] == PREDEX_SOURCE_ID
    assert seed["native_id"] == "Kamlesh Vs. Union of India"
    assert seed["code_era"] == "ipc"
    assert seed["case_type"] == "civil"  # no criminal/constitutional/commercial keywords in the fixture text
    assert seed["text"].startswith("The appellant was provisionally appointed")
    assert "held that prolonged service" in seed["text"]
    assert seed["token_count"] == len(seed["text"]) // 4
    assert seed["meta_json"] == {"estimator": "chars/4", "label": 1}
    assert seed["seed_id"] == seed_id_for(PREDEX_SOURCE_ID, "Kamlesh Vs. Union of India")


def test_predex_seed_reject_empty_facts():
    assert predex_seed(_predex_raw(facts="")) is None


def test_predex_seed_reject_empty_output():
    assert predex_seed(_predex_raw(output="")) is None


def test_predex_seed_reject_short_text():
    assert predex_seed(_predex_raw(facts="short", output="also short")) is None


def test_predex_seed_reject_markup():
    raw = _predex_raw(output=_predex_raw()["Output"] + " <|im_start|>")
    assert predex_seed(raw) is None


def test_predex_seed_native_id_none_when_case_name_missing():
    raw = _predex_raw()
    del raw["Case Name"]
    seed = predex_seed(raw)
    assert seed is not None
    assert seed["native_id"] is None
    # falls back to hashing the full text, still deterministic
    assert seed["seed_id"] == seed_id_for(PREDEX_SOURCE_ID, seed["text"])


# --------------------------------------------------------------------------
# tathyanyaya_seed
# --------------------------------------------------------------------------

def _tathyanyaya_raw(name="Pandurang Chandrakant Mhatre Vs. State", text=None, reasoning=None, label=0):
    if text is None:
        text = ("Nineteen persons were arraigned before the Trial Court for offences "
                 "under Sections 147, 148 and 302 read with Section 149 of the IPC. " * 6)
    if reasoning is None:
        reasoning = ("The Court re-appreciated the ocular evidence and held that the "
                     "common object of the unlawful assembly was clearly established " * 5)
    return {"Case Name": name, "text": text, "label": label, "Reasoning": reasoning}


def test_tathyanyaya_seed_accept():
    seed = tathyanyaya_seed(_tathyanyaya_raw())
    assert seed is not None
    assert seed["source_id"] == TATHYANYAYA_SOURCE_ID
    assert seed["native_id"] == "Pandurang Chandrakant Mhatre Vs. State"
    assert seed["code_era"] == "ipc"
    assert "re-appreciated the ocular evidence" in seed["text"]
    assert seed["meta_json"] == {"estimator": "chars/4", "label": 0}


def test_tathyanyaya_seed_reject_empty_reasoning():
    assert tathyanyaya_seed(_tathyanyaya_raw(reasoning="")) is None


def test_tathyanyaya_seed_reject_short_text():
    assert tathyanyaya_seed(_tathyanyaya_raw(text="x", reasoning="y")) is None


def test_tathyanyaya_seed_reject_markup():
    raw = _tathyanyaya_raw()
    raw["Reasoning"] += " <|end|>"
    assert tathyanyaya_seed(raw) is None


# --------------------------------------------------------------------------
# injudgements_seed
# --------------------------------------------------------------------------

def _injudgements_raw(text=None, case_type="Criminal", court_norm="Supreme Court of India"):
    if text is None:
        text = "This is the full text of a reported judgment spanning several pages of analysis. " * 10
    return {
        "Titles": "State v. Respondent",
        "Court_Name": "Supreme Court",
        "Cites": 42,
        "Cited_by": 7,
        "Doc_url": "https://indiankanoon.org/doc/12345/",
        "Text": text,
        "Doc_size": len(text),
        "Case_Type": case_type,
        "Court_Type": "SC",
        "Court_Name_Normalized": court_norm,
    }


def test_injudgements_seed_accept():
    seed = injudgements_seed(_injudgements_raw())
    assert seed is not None
    assert seed["source_id"] == INJUDGEMENTS_SOURCE_ID
    assert seed["native_id"] == "https://indiankanoon.org/doc/12345/"
    assert seed["court"] == "Supreme Court of India"
    assert seed["case_type"] == "criminal"
    assert seed["code_era"] == "ipc"
    assert seed["token_count"] == len(seed["text"]) // 4
    assert seed["meta_json"] == {"estimator": "chars/4", "cites": 42, "cited_by": 7}


def test_injudgements_seed_court_falls_back_to_raw_name():
    raw = _injudgements_raw(court_norm="")
    seed = injudgements_seed(raw)
    assert seed["court"] == "Supreme Court"


def test_injudgements_seed_reject_short_text():
    assert injudgements_seed(_injudgements_raw(text="too short")) is None


def test_injudgements_seed_reject_markup():
    raw = _injudgements_raw(text=_injudgements_raw()["Text"] + " <|im_start|>")
    assert injudgements_seed(raw) is None


# --------------------------------------------------------------------------
# load_seeds end-to-end.
# --------------------------------------------------------------------------

def _predex_rows(n):
    return [_predex_raw(name=f"Case {i}", facts=_predex_raw()["Input"] + str(i)) for i in range(n)]


def _tathyanyaya_rows(n):
    return [_tathyanyaya_raw(name=f"Matter {i}", text=_tathyanyaya_raw()["text"] + str(i)) for i in range(n)]


def _injudgements_rows(n):
    rows = []
    for i in range(n):
        raw = _injudgements_raw()
        raw["Doc_url"] = f"https://indiankanoon.org/doc/{i}/"
        rows.append(raw)
    return rows


@pytest.fixture
def store(tmp_path):
    with Store.open(tmp_path / "state" / "law_v1.sqlite3") as s:
        yield s


def test_load_seeds_end_to_end_counts_and_sources(store):
    sources = {
        "predex": iter(_predex_rows(5)),
        "tathyanyaya": iter(_tathyanyaya_rows(3)),
        "injudgements": iter(_injudgements_rows(2)),
    }
    stats = load_seeds(store, cfg=None, sources=sources, limits={"predex": 5, "tathyanyaya": 3, "injudgements": 2})

    assert stats["predex"]["accepted"] == 5
    assert stats["tathyanyaya"]["accepted"] == 3
    assert stats["injudgements"]["accepted"] == 2
    assert store.seed_count() == 10
    assert store.seed_count("predex source not a real key") == 0

    for source_id in (PREDEX_SOURCE_ID, TATHYANYAYA_SOURCE_ID, INJUDGEMENTS_SOURCE_ID):
        row = store.conn.execute(
            "SELECT license FROM source WHERE source_id = ?", (source_id,)
        ).fetchone()
        assert row is not None, f"{source_id} was never registered"
        assert row["license"] == "Apache-2.0"


def test_load_seeds_respects_limit(store):
    sources = {"predex": iter(_predex_rows(10)), "tathyanyaya": iter([]), "injudgements": iter([])}
    stats = load_seeds(store, cfg=None, sources=sources, limits={"predex": 4, "tathyanyaya": 0, "injudgements": 0})
    assert stats["predex"]["accepted"] == 4
    assert store.seed_count(PREDEX_SOURCE_ID) == 4


def test_load_seeds_uses_default_limits_when_key_absent_from_limits_dict(store):
    sources = {"predex": iter(_predex_rows(3)), "tathyanyaya": iter([]), "injudgements": iter([])}
    stats = load_seeds(store, cfg=None, sources=sources, limits={})
    assert stats["predex"]["accepted"] == 3
    assert stats["predex"]["limit"] == DEFAULT_LIMITS["predex"]


def test_load_seeds_skips_source_key_absent_from_sources(store):
    # deliberately omits tathyanyaya/injudgements - a 0-count/absent source
    # must never be looked up or registered.
    sources = {"predex": iter(_predex_rows(2))}
    stats = load_seeds(store, cfg=None, sources=sources, limits={"predex": 2})
    assert "tathyanyaya" not in stats
    assert "injudgements" not in stats
    row = store.conn.execute(
        "SELECT 1 FROM source WHERE source_id = ?", (TATHYANYAYA_SOURCE_ID,)
    ).fetchone()
    assert row is None


def test_load_seeds_idempotent_second_run_adds_no_new_seeds(store):
    def fresh_sources():
        return {
            "predex": iter(_predex_rows(4)),
            "tathyanyaya": iter(_tathyanyaya_rows(2)),
            "injudgements": iter(_injudgements_rows(1)),
        }

    limits = {"predex": 4, "tathyanyaya": 2, "injudgements": 1}
    load_seeds(store, cfg=None, sources=fresh_sources(), limits=limits)
    first_count = store.seed_count()
    assert first_count == 7

    load_seeds(store, cfg=None, sources=fresh_sources(), limits=limits)
    assert store.seed_count() == first_count


def test_load_seeds_dedups_repeated_native_id_within_one_call(store):
    dup_raw = _predex_raw(name="Same Case Name Every Time")
    sources = {"predex": iter([dup_raw, dup_raw, dup_raw]), "tathyanyaya": iter([]), "injudgements": iter([])}
    stats = load_seeds(store, cfg=None, sources=sources, limits={"predex": 10, "tathyanyaya": 0, "injudgements": 0})
    assert stats["predex"]["accepted"] == 1
    assert stats["predex"]["rejected"] == 2
    assert store.seed_count(PREDEX_SOURCE_ID) == 1


# --------------------------------------------------------------------------
# Module-import / CLI hygiene.
# --------------------------------------------------------------------------

def test_cli_hard_exits_after_success():
    text = SEEDS_SRC.read_text(encoding="utf-8")
    assert "os._exit(0)" in text


def test_module_import_never_touches_datasets_at_top_level():
    tree = ast.parse(SEEDS_SRC.read_text(encoding="utf-8"))
    banned = {"datasets", "pyarrow", "huggingface_hub"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = {alias.name.split(".")[0] for alias in node.names}
            assert not (names & banned), f"top-level import of {names & banned}"
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in banned, f"top-level `from {node.module} import ...`"


def test_module_importable_without_error():
    import importlib

    import tuned.data.seeds as seeds_mod
    importlib.reload(seeds_mod)
    assert hasattr(seeds_mod, "load_seeds")


def test_source_order_matches_default_limits_keys():
    assert set(SOURCE_ORDER) == set(DEFAULT_LIMITS.keys())
