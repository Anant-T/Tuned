import ast
import json
from pathlib import Path

import pytest

from tuned.data.config import load_build_config
from tuned.data.curated import (
    AALAP_SAFE_TASKS,
    DEFAULT_COUNTS,
    SLICE_ORDER,
    aalap_safe_row,
    build_curated,
    parse_counts,
    pi169_audited_row,
    predex_prediction_row,
)

CURATED_SRC = Path(__file__).parent.parent / "src" / "tuned" / "data" / "curated.py"


def _real_think_tags():
    cfg = load_build_config("data/configs/data_law_v1.yaml", allow_unpinned=True)
    return cfg.think_open, cfg.think_close


def test_empty_think_byte_exact_against_real_config_and_row():
    think_open, think_close = _real_think_tags()
    row, reason = predex_prediction_row(_predex_raw(), think_open, think_close)
    assert reason is None
    assert row["messages"][1]["content"].startswith("<think>\n\n</think>")


# --------------------------------------------------------------------------
# predex_prediction_row
# --------------------------------------------------------------------------

def _predex_raw(user=None, answer=None, name="Kamlesh Vs. Union of India"):
    if user is None:
        user = ("The appellant was provisionally appointed and later challenged the "
                 "discontinuation of her service before the Tribunal. " * 4)
    if answer is None:
        answer = ("The Court held that prolonged service alone does not entitle an "
                   "employee to regularization absent a sanctioned vacancy. " * 4)
    return {"Case Name": name, "Input": user, "Output": answer}


def test_predex_prediction_row_accept():
    row, reason = predex_prediction_row(_predex_raw(), "<think>", "</think>")
    assert reason is None
    assert row["messages"][0]["content"] == _predex_raw()["Input"].strip()
    assert row["messages"][1]["content"] == "<think>\n\n</think>" + _predex_raw()["Output"].strip()
    assert row["_prov"] == {
        "source": "L-NLProc/PredEx_Instruction-Tuning_Pred-Exp",
        "license": "Apache-2.0",
        "native_id": "Kamlesh Vs. Union of India",
        "reasoning": False,
    }


def test_predex_prediction_row_reject_empty():
    row, reason = predex_prediction_row(_predex_raw(user=""), "<think>", "</think>")
    assert row is None
    assert reason == "empty"


def test_predex_prediction_row_reject_too_long():
    row, reason = predex_prediction_row(_predex_raw(answer="x" * 25_000), "<think>", "</think>")
    assert row is None
    assert reason == "too_long"


def test_predex_prediction_row_reject_refusal():
    row, reason = predex_prediction_row(
        _predex_raw(answer="I'm sorry, I cannot help with that request. " * 5), "<think>", "</think>"
    )
    assert row is None
    assert reason == "refusal"


def test_predex_prediction_row_reject_emoji():
    row, reason = predex_prediction_row(
        _predex_raw(answer=_predex_raw()["Output"] + " \U0001F600"), "<think>", "</think>"
    )
    assert row is None
    assert reason == "emoji"


def test_predex_prediction_row_reject_markup():
    row, reason = predex_prediction_row(
        _predex_raw(answer=_predex_raw()["Output"] + " <|im_start|>"), "<think>", "</think>"
    )
    assert row is None
    assert reason == "markup"


def test_predex_prediction_row_native_id_none_when_case_name_missing():
    raw = _predex_raw()
    del raw["Case Name"]
    row, reason = predex_prediction_row(raw, "<think>", "</think>")
    assert reason is None
    assert row["_prov"]["native_id"] is None


# --------------------------------------------------------------------------
# aalap_safe_row
# --------------------------------------------------------------------------

def _aalap_raw(task="issue_generation", combined=None, user_prompt=None, output=None, reasoning=None):
    if combined is None:
        combined = "Given the following case excerpt, identify the key legal issues raised. " * 5
    if user_prompt is None:
        user_prompt = "Identify the key legal issues raised in the case excerpt. " * 5
    if output is None:
        output = "The key issues are jurisdiction, limitation, and the maintainability of the suit. " * 4
    raw = {
        "task": task,
        "input_text": "case excerpt text",
        "system_prompt": "You are a legal assistant.",
        "user_prompt": user_prompt,
        "combined_input_prompt": combined,
        "output_text": output,
    }
    if reasoning is not None:
        raw["reasoning"] = reasoning
    return raw


def test_aalap_safe_row_accept_uses_combined_input_prompt():
    row, reason = aalap_safe_row(_aalap_raw(), "<think>", "</think>")
    assert reason is None
    assert row["messages"][0]["content"] == _aalap_raw()["combined_input_prompt"].strip()
    assert row["messages"][1]["content"].startswith("<think>\n\n</think>")
    assert row["_prov"] == {
        "source": "opennyaiorg/aalap_instruction_dataset",
        "license": "CC0-1.0",
        "native_id": None,
        "reasoning": False,
    }


def test_aalap_safe_row_falls_back_to_user_prompt_when_combined_missing():
    raw = _aalap_raw(combined="")
    row, reason = aalap_safe_row(raw, "<think>", "</think>")
    assert reason is None
    assert row["messages"][0]["content"] == raw["user_prompt"].strip()


@pytest.mark.parametrize("task,expected_license", sorted(AALAP_SAFE_TASKS.items()))
def test_aalap_safe_row_accepts_every_allowlisted_task(task, expected_license):
    row, reason = aalap_safe_row(_aalap_raw(task=task), "<think>", "</think>")
    assert reason is None, f"task {task!r} unexpectedly rejected: {reason}"
    assert row["_prov"]["license"] == expected_license


def test_aalap_safe_row_rejects_known_nc_task():
    # contract_clause_generation is documented cc-by-nc-4.0 on the HF card -
    # must never be admitted regardless of content quality. Real runtime
    # value, ___variant suffix included (verified 2026-08-29).
    row, reason = aalap_safe_row(
        _aalap_raw(task="contract_clause_generation___generation"), "<think>", "</think>"
    )
    assert row is None
    assert reason == "task_not_allowlisted"


def test_aalap_safe_row_rejects_unestablished_license_task():
    # legalbench is licensed "Other" on the card - not an established
    # non-NC license, so per the brief's "unknown -> exclude" rule it is
    # excluded even though it isn't the one known-NC task.
    row, reason = aalap_safe_row(
        _aalap_raw(task="legalbench___cuad_renewal_term"), "<think>", "</think>"
    )
    assert row is None
    assert reason == "task_not_allowlisted"


def test_aalap_safe_row_maps_the_variant_suffix_to_its_family():
    # The runtime task strings carry ___variant suffixes the card's table
    # never mentioned; the license is keyed on the family prefix.
    row, reason = aalap_safe_row(
        _aalap_raw(task="argument_generation___petitioner"), "<think>", "</think>"
    )
    assert reason is None
    assert row["_prov"]["license"] == "CC0-1.0"


def test_aalap_safe_row_rejects_unrecognized_task_string():
    row, reason = aalap_safe_row(_aalap_raw(task="Some Future Task Nobody Documented"), "<think>", "</think>")
    assert row is None
    assert reason == "task_not_allowlisted"


def test_aalap_safe_row_rejects_missing_task_field():
    raw = _aalap_raw()
    del raw["task"]
    row, reason = aalap_safe_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "task_not_allowlisted"


def test_aalap_safe_row_uses_real_reasoning_field_when_present():
    trace = "First, the limitation period is computed from the date of the impugned order. " * 3
    row, reason = aalap_safe_row(_aalap_raw(reasoning=trace), "<think>", "</think>")
    assert reason is None
    assert row["messages"][1]["content"] == f"<think>{trace.strip()}</think>{_aalap_raw()['output_text'].strip()}"
    assert row["_prov"]["reasoning"] is True


def test_aalap_safe_row_reject_empty_output():
    row, reason = aalap_safe_row(_aalap_raw(output=""), "<think>", "</think>")
    assert row is None
    assert reason == "empty"


def test_aalap_safe_row_reject_markup():
    raw = _aalap_raw(output=_aalap_raw()["output_text"] + " <|im_start|>")
    row, reason = aalap_safe_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "markup"


# --------------------------------------------------------------------------
# pi169_audited_row
# --------------------------------------------------------------------------

def _pi169_raw(prompt=None, think=None, response=None):
    if prompt is None:
        prompt = "What is the punishment for theft under the Indian Penal Code?"
    if think is None:
        think = ("Theft is defined under Section 378 IPC as dishonestly taking movable "
                  "property out of the possession of another without consent. " * 8)[:2000]
    if response is None:
        response = ("Under Section 379 IPC, whoever commits theft shall be punished with "
                    "imprisonment of either description for a term which may extend to "
                    "three years, or with fine, or with both. " * 2)
    return {"prompt": prompt, "complex_cot": think, "response": response}


def test_pi169_audited_row_accept():
    row, reason = pi169_audited_row(_pi169_raw(), "<think>", "</think>")
    assert reason is None
    raw = _pi169_raw()
    assert row["messages"][0]["content"] == raw["prompt"].strip()
    assert row["messages"][1]["content"] == f"<think>{raw['complex_cot'].strip()}</think>{raw['response'].strip()}"
    assert row["_prov"] == {
        "source": "169Pi/indian_law",
        "license": "Apache-2.0",
        "native_id": None,
        "reasoning": True,
    }


def test_pi169_audited_row_reject_think_too_short():
    row, reason = pi169_audited_row(_pi169_raw(think="too short a trace"), "<think>", "</think>")
    assert row is None
    assert reason == "think_length"


def test_pi169_audited_row_reject_think_too_long():
    row, reason = pi169_audited_row(_pi169_raw(think="x" * 8_001), "<think>", "</think>")
    assert row is None
    assert reason == "think_length"


def test_pi169_audited_row_reject_answer_too_short():
    row, reason = pi169_audited_row(_pi169_raw(response="Too short."), "<think>", "</think>")
    assert row is None
    assert reason == "answer_length"


def test_pi169_audited_row_reject_answer_too_long():
    row, reason = pi169_audited_row(_pi169_raw(response="x" * 4_001), "<think>", "</think>")
    assert row is None
    assert reason == "answer_length"


@pytest.mark.parametrize("term", ["BNS", "BNSS", "BSA"])
def test_pi169_audited_row_reject_new_code_contamination(term):
    raw = _pi169_raw(response=_pi169_raw()["response"] + f" This is also punishable under the {term}.")
    row, reason = pi169_audited_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "new_code_contamination"


def test_pi169_audited_row_allows_ipc_mentions():
    # sanity: the drop is specific to BNS/BNSS/BSA, not any code mention -
    # this dataset's own content is IPC-era and must not be rejected for it.
    raw = _pi169_raw()
    assert "IPC" in raw["response"]
    row, reason = pi169_audited_row(raw, "<think>", "</think>")
    assert reason is None


def test_pi169_audited_row_reject_refusal():
    raw = _pi169_raw(response="I'm sorry, I cannot help with that request. " * 3)
    row, reason = pi169_audited_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "refusal"


def test_pi169_audited_row_reject_markup():
    raw = _pi169_raw(response=_pi169_raw()["response"] + " <|im_start|>")
    row, reason = pi169_audited_row(raw, "<think>", "</think>")
    assert row is None
    assert reason == "markup"


def test_pi169_audited_row_reject_empty():
    row, reason = pi169_audited_row(_pi169_raw(prompt=""), "<think>", "</think>")
    assert row is None
    assert reason == "empty"


# --------------------------------------------------------------------------
# build_curated end-to-end.
# --------------------------------------------------------------------------

def _synthetic_sources(n_each=6):
    predex = [_predex_raw(name=f"case {i}", user=_predex_raw()["Input"] + str(i)) for i in range(n_each)]
    aalap = [_aalap_raw(combined=_aalap_raw()["combined_input_prompt"] + str(i)) for i in range(n_each)]
    pi169 = [_pi169_raw(prompt=_pi169_raw()["prompt"] + f" (variant {i})") for i in range(n_each)]
    return {
        "predex_prediction": iter(predex),
        "aalap_safe": iter(aalap),
        "pi169_audited": iter(pi169),
    }


class _FakeBuildCfg:
    def __init__(self, workdir):
        self.workdir = workdir


class _FakeCfg:
    def __init__(self, workdir, think_open="<think>", think_close="</think>"):
        self.think_open = think_open
        self.think_close = think_close
        self.build = _FakeBuildCfg(workdir)


def test_build_curated_writes_expected_counts_and_prov(tmp_path):
    cfg = _FakeCfg(str(tmp_path / "workdir"))
    counts = (2, 2, 2)
    out_path = tmp_path / "curated_c1.jsonl"
    stats = build_curated(cfg, counts, rows_by_source=_synthetic_sources(), out_path=out_path)

    assert stats["total"] == 6
    for name, n in zip(SLICE_ORDER, counts):
        assert stats[name]["accepted"] == n

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 6
    rows = [json.loads(line) for line in lines]

    reasoning_by_source = {"169Pi/indian_law": True}
    for row in rows:
        prov = row["_prov"]
        assert set(prov.keys()) == {"source", "license", "native_id", "reasoning"}
        expected_reasoning = reasoning_by_source.get(prov["source"], False)
        assert prov["reasoning"] is expected_reasoning
        assert "<|" not in row["messages"][0]["content"]
        assert "<|" not in row["messages"][1]["content"]


def test_build_curated_raises_on_shortfall_names_source(tmp_path):
    cfg = _FakeCfg(str(tmp_path / "workdir"))
    sources = _synthetic_sources(n_each=1)  # aalap_safe will run out at 1 < 2
    counts = (1, 2, 1)
    with pytest.raises(RuntimeError, match="aalap_safe"):
        build_curated(cfg, counts, rows_by_source=sources, out_path=tmp_path / "curated_c1.jsonl")


def test_build_curated_dedup_within_slice(tmp_path):
    cfg = _FakeCfg(str(tmp_path / "workdir"))
    dup = _pi169_raw()
    pi169 = [dup, dup, dup, dup, _pi169_raw(prompt="a genuinely different question here")]
    sources = {
        "predex_prediction": iter([]),
        "aalap_safe": iter([]),
        "pi169_audited": iter(pi169),
    }
    counts = (0, 0, 2)
    stats = build_curated(cfg, counts, rows_by_source=sources, out_path=tmp_path / "curated_c1.jsonl")
    assert stats["pi169_audited"]["accepted"] == 2
    assert stats["pi169_audited"]["rejects"].get("duplicate") == 3


def test_build_curated_skips_zero_count_slices_without_touching_source(tmp_path):
    cfg = _FakeCfg(str(tmp_path / "workdir"))
    sources = {"predex_prediction": iter(_synthetic_sources()["predex_prediction"])}
    counts = (1, 0, 0)
    stats = build_curated(cfg, counts, rows_by_source=sources, out_path=tmp_path / "curated_c1.jsonl")
    assert stats["aalap_safe"]["accepted"] == 0
    assert stats["pi169_audited"]["accepted"] == 0
    assert stats["total"] == 1


def test_build_curated_default_out_path_uses_build_paths(tmp_path):
    cfg = _FakeCfg(str(tmp_path / "workdir"))
    stats = build_curated(cfg, (1, 1, 1), rows_by_source=_synthetic_sources())
    expected = tmp_path / "workdir" / "streams" / "curated_c1.jsonl"
    assert Path(stats["out_path"]) == expected
    assert expected.exists()


# --------------------------------------------------------------------------
# --counts parsing.
# --------------------------------------------------------------------------

def test_parse_counts_valid():
    assert parse_counts("800,600,300") == DEFAULT_COUNTS


def test_parse_counts_wrong_length():
    with pytest.raises(ValueError, match="3 comma-separated"):
        parse_counts("1,2")


def test_parse_counts_non_integer():
    with pytest.raises(ValueError):
        parse_counts("a,b,c")


# --------------------------------------------------------------------------
# Module-import / CLI hygiene.
# --------------------------------------------------------------------------

def test_cli_hard_exits_after_success():
    text = CURATED_SRC.read_text(encoding="utf-8")
    assert "os._exit(0)" in text


def test_module_import_never_touches_datasets_at_top_level():
    tree = ast.parse(CURATED_SRC.read_text(encoding="utf-8"))
    banned = {"datasets", "pyarrow", "huggingface_hub"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = {alias.name.split(".")[0] for alias in node.names}
            assert not (names & banned), f"top-level import of {names & banned}"
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in banned, f"top-level `from {node.module} import ...`"


def test_module_importable_without_error():
    import importlib

    import tuned.data.curated as curated_mod
    importlib.reload(curated_mod)
    assert hasattr(curated_mod, "build_curated")


def test_aalap_safe_tasks_excludes_known_nc_and_unestablished_tasks():
    assert "contract_clause_generation" not in AALAP_SAFE_TASKS
    assert "legalbench" not in AALAP_SAFE_TASKS
    assert "cc-by-nc-4.0" not in AALAP_SAFE_TASKS.values()
