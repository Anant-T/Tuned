"""push.py - stats' green output -> an HF dataset repo, through a fake.

Fixtures are structural shapes with filler prose; no real judgment or eval
text appears anywhere here. The e2e corpus and the real dedupe/split/assemble/
stats tail are test_build_stats.py's - reused rather than rebuilt, per the
task-15 brief.
"""

import json
import os
from pathlib import Path

import pytest
from pipeline_fakes import paths_for, temp_config
from test_build_config import _base_doc, _write
from test_build_decontaminate import prose, row
from test_build_assemble import FakeTokenizer
from test_build_stats import SCALE, e2e_corpus, run_pipeline, traced

from tuned.data.acquire import sha256_file
from tuned.data.assemble import main as assemble_main
from tuned.data.config import PushCfg, load_build_config
from tuned.data.decontaminate import SCRIPT_NONE
from tuned.data.dedupe import main as dedupe_main
from tuned.data.jsonl import read_jsonl, write_jsonl
from tuned.data.push import (
    EVAL_FILENAME,
    MANIFEST_FILENAME,
    PUSH_VERSION,
    README_FILENAME,
    TRAIN_FILENAME,
    CardDataMissing,
    RemoteManifestCorrupt,
    build_manifest,
    bytes_refusal,
    chain_faults,
    decon_chain_faults,
    decon_sections,
    license_rows,
    mix_rows,
    module_versions,
    render_card,
    same_uploaded_bytes,
    source_license_rows,
    stats_refusal,
)
from tuned.data.push import main as push_main
from tuned.data.split import main as split_main
from tuned.data.stats import REPORT_FILENAME, SUMMARY_FILENAME
from tuned.data.stats import main as stats_main

PUSH_SRC = Path(__file__).parent.parent / "src" / "tuned" / "data" / "push.py"
DATA_CONFIG = Path(__file__).parent.parent / "data" / "configs" / "data_law_v1.yaml"


# --------------------------------------------------------------------------
# The hub-client double.
# --------------------------------------------------------------------------

class FakeHubClient:
    """In-memory double for push.RealHubClient's three-method contract.

    `existing_manifest`/`existing_revision` seed a repo that already exists,
    the way a real second push would find one.
    """

    def __init__(self, *, existing_manifest=None, existing_revision=None):
        self._manifest = existing_manifest
        self._revision = existing_revision
        self.ensure_calls = 0
        self.upload_calls = 0
        self.ensure_args = None
        self.uploaded_files = None
        self.commit_messages: list[str] = []

    def ensure_repo(self, repo_id, *, private):
        self.ensure_calls += 1
        self.ensure_args = (repo_id, private)

    def current_manifest(self, repo_id):
        return self._manifest, self._revision

    def upload(self, repo_id, files, *, commit_message):
        self.upload_calls += 1
        self.uploaded_files = dict(files)
        self.commit_messages.append(commit_message)
        self._revision = f"rev{self.upload_calls}"
        self._manifest = json.loads(Path(files[MANIFEST_FILENAME]).read_text(encoding="utf-8"))
        return self._revision


# --------------------------------------------------------------------------
# Enriching the e2e fixture's decontamination.json.
# --------------------------------------------------------------------------

EVAL_SETS = {
    "bbl": {"status": "ok", "allowed_missing": False, "items": 500},
    "il_tur": {"status": "ok", "allowed_missing": False, "items": 300},
    "aibe": {"status": "not_acquired", "allowed_missing": True, "items": 0},
}
SEMANTIC_SCRIPTS = {
    "latin": {"control": "passed", "screened": True, "eval_items": 700},
    "devanagari": {
        "control": "no discriminative power over this script",
        "screened": False,
        "eval_items": 100,
    },
}


def enrich_decon(paths, *, eval_sets=EVAL_SETS, semantic_scripts=SEMANTIC_SCRIPTS):
    """Fills in the eval-set/semantic-script detail run_pipeline's minimal
    decontamination.json stub does not carry, WITHOUT touching the
    output/counts/thresholds keys dedupe.py already verified its input
    against - those stay exactly as run_pipeline wrote them.

    ALSO patches the same eval_sets detail into the custody chain's own
    embedded copy (assemble.json -> split -> dedupe -> decontamination),
    narrowed the way dedupe.manifest_of's upstream_summary narrows it
    (status/allowed_missing/items only). A REAL decontaminate.py run writes
    eval_sets into decontamination.json before dedupe.py ever reads it, so
    the two would already agree; this two-step stub-then-enrich shortcut is
    what's out of step with that, not push.py's N-3 cross-check
    (decon_chain_faults) - patch the fixture to match reality rather than
    let every test in this file silently exercise a manifest that could
    never happen with the real pipeline."""
    decon_path = paths.out_dir / "decontamination.json"
    decon = json.loads(decon_path.read_text(encoding="utf-8"))
    decon["eval_sets"] = eval_sets
    decon["semantic_scripts"] = semantic_scripts
    decon_path.write_text(json.dumps(decon), encoding="utf-8")

    assemble_path = paths.out_dir / "assemble.json"
    assemble = json.loads(assemble_path.read_text(encoding="utf-8"))
    assemble["split"]["dedupe"]["decontamination"]["eval_sets"] = {
        key: {"status": v.get("status"), "allowed_missing": v.get("allowed_missing"),
              "items": v.get("items")}
        for key, v in eval_sets.items()
    }
    assemble_path.write_text(json.dumps(assemble), encoding="utf-8")
    return decon


def green_pipeline(tmp_path, *, require_chain=True):
    """run_pipeline's e2e corpus, enriched, through push's own default paths."""
    cfg_path = temp_config(tmp_path)
    if not require_chain:
        text = Path(cfg_path).read_text(encoding="utf-8")
        patched = text.replace("require_chain: true", "require_chain: false")
        assert patched != text
        Path(cfg_path).write_text(patched, encoding="utf-8")
    codes, report, paths = run_pipeline(tmp_path, e2e_corpus(), cfg_path=cfg_path)
    assert codes == {"dedupe": 0, "split": 0, "assemble": 0, "stats": 0}
    enrich_decon(paths)
    return cfg_path, report, paths


# --------------------------------------------------------------------------
# config.py: the push: block.
# --------------------------------------------------------------------------

def test_push_is_none_when_the_block_is_absent(tmp_path):
    doc = _base_doc()
    assert "push" not in doc
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.push is None


def test_the_shipped_config_carries_a_real_push_target():
    cfg = load_build_config(DATA_CONFIG, allow_unpinned=True)
    assert cfg.push == PushCfg(repo_id="tantan01/tuned-law-v1-data", private=True, card_extra=None)


def test_recovery_push_target_is_not_the_live_dataset_repo():
    recovery = Path(__file__).parent.parent / "configs" / "data_law_v1_exp_recovery.yaml"
    harmony = Path(__file__).parent.parent / "configs" / "data_law_v1_exp_harmony.yaml"
    rec = load_build_config(recovery, allow_unpinned=True)
    live = load_build_config(DATA_CONFIG, allow_unpinned=True)
    har = load_build_config(harmony, allow_unpinned=True)
    assert rec.push.repo_id != "tantan01/tuned-law-v1-data"
    assert rec.push.repo_id != live.push.repo_id
    assert live.push.repo_id == "tantan01/tuned-law-v1-data"
    assert har.push.repo_id == "tantan01/tuned-law-v1-data"


def test_the_train_configs_pin_target_matches_pushs_own_repo():
    """M5: closes the loop task 15 exists to make real. pin_dataset.py reads
    hub.dataset_repo from the SAME train config data_law_v1.yaml points at
    (build.train_config) - it must name the exact repo push.py writes to, or
    the pin script has nothing real to pin."""
    from tuned.train.config import load_config

    cfg = load_build_config(DATA_CONFIG, allow_unpinned=True)
    train_cfg = load_config(
        Path(__file__).parent.parent / "training" / "configs" / "law_v1_8b_ddp.yaml", allow_unpinned=True,
    )
    assert train_cfg.hub.dataset_repo == cfg.push.repo_id


def test_repo_id_is_required_and_named(tmp_path):
    doc = _base_doc()
    doc["push"] = {"private": True}
    with pytest.raises(ValueError, match="push.repo_id is missing"):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)


def test_repo_id_must_be_a_non_empty_string(tmp_path):
    doc = _base_doc()
    doc["push"] = {"repo_id": "   "}
    with pytest.raises(ValueError, match="non-empty string"):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)


def test_repo_id_rejects_leading_or_trailing_whitespace(tmp_path):
    doc = _base_doc()
    doc["push"] = {"repo_id": "  x/y  "}
    with pytest.raises(ValueError, match="whitespace"):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)


def test_repo_id_must_be_namespace_slash_name(tmp_path):
    doc = _base_doc()
    doc["push"] = {"repo_id": "notarepo"}
    with pytest.raises(ValueError, match="namespace/name"):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)

    doc["push"] = {"repo_id": "x/y/z"}
    with pytest.raises(ValueError, match="namespace/name"):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)

    doc["push"] = {"repo_id": "/y"}
    with pytest.raises(ValueError, match="namespace/name"):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)


def test_repo_id_rejects_control_characters_without_echoing_the_value(tmp_path):
    doc = _base_doc()
    suspect = "x/y\nevil-log-injection"
    doc["push"] = {"repo_id": suspect}
    with pytest.raises(ValueError) as exc_info:
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    message = str(exc_info.value)
    assert "non-printable" in message or "control" in message
    assert suspect not in message  # named by shape, never echoed whole


def test_private_defaults_true_and_is_strict_about_yaml_booleans(tmp_path):
    doc = _base_doc()
    doc["push"] = {"repo_id": "x/y"}
    cfg = load_build_config(_write(tmp_path, doc), allow_unpinned=True)
    assert cfg.push.private is True

    doc["push"] = {"repo_id": "x/y", "private": "false"}
    with pytest.raises(ValueError, match="push.private must be a YAML boolean"):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)


def test_card_extra_must_be_a_string_when_present(tmp_path):
    doc = _base_doc()
    doc["push"] = {"repo_id": "x/y", "card_extra": 7}
    with pytest.raises(ValueError, match="push.card_extra must be a string"):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)


def test_push_block_must_be_a_mapping(tmp_path):
    doc = _base_doc()
    doc["push"] = ["repo_id"]
    with pytest.raises(ValueError, match="must be a block of keys"):
        load_build_config(_write(tmp_path, doc), allow_unpinned=True)


# --------------------------------------------------------------------------
# chain_faults / stats_refusal - pure, both directions.
# --------------------------------------------------------------------------

def _report(*, red=(), verdict="green", chain_detail=None):
    return {
        "red": list(red),
        "verdict": verdict,
        "gates": {"chain": {"detail": chain_detail or {"missing": [], "unrecorded": [], "unverified": []}}},
    }


def test_chain_faults_is_empty_on_a_complete_chain():
    assert chain_faults(_report()) == []


def test_chain_faults_names_each_kind_of_break():
    detail = {"missing": ["decontamination"], "unrecorded": ["dedupe"], "unverified": ["split"]}
    faults = chain_faults(_report(chain_detail=detail))
    assert any("absent: decontamination" in f for f in faults)
    assert any("dedupe" in f and "never recorded" in f for f in faults)
    assert any("unverified: split" in f for f in faults)


def test_stats_refusal_none_on_a_green_complete_report():
    assert stats_refusal(_report(), report_path=Path("x"), config_path="c.yaml") is None


# --------------------------------------------------------------------------
# decon_chain_faults - N-3 (round 2): the standalone decontamination.json
# push.py reads for the card, cross-checked against dedupe.py's own custody
# summary of it embedded in the chain.
# --------------------------------------------------------------------------

def _chain_with_decon(decon_summary):
    return {"split": {"dedupe": {"decontamination": decon_summary}}}


def test_decon_chain_faults_is_empty_when_consistent():
    summary = {"at": "t0", "decon_version": 4, "counts": {"kept": 100},
               "eval_sets": {"bbl": {"status": "ok", "allowed_missing": False, "items": 500}}}
    decon = {"at": "t0", "decon_version": 4, "counts": {"kept": 100},
             # The standalone file carries MORE fields per eval set than the
             # chain's narrowed summary - that is not a divergence.
             "eval_sets": {"bbl": {"status": "ok", "allowed_missing": False, "items": 500,
                                   "repo_id": "org/bbl", "license": "CC-BY-4.0"}}}
    assert decon_chain_faults(decon, _chain_with_decon(summary)) == []


def test_decon_chain_faults_names_each_kind_of_divergence():
    summary = {"at": "t0", "decon_version": 4, "counts": {"kept": 100},
               "eval_sets": {"bbl": {"status": "ok", "allowed_missing": False, "items": 500}}}
    chain = _chain_with_decon(summary)

    edited_at = {"at": "LATER", "decon_version": 4, "counts": {"kept": 100},
                 "eval_sets": summary["eval_sets"]}
    assert decon_chain_faults(edited_at, chain) == ["at"]

    edited_version = {"at": "t0", "decon_version": 99, "counts": {"kept": 100},
                      "eval_sets": summary["eval_sets"]}
    assert decon_chain_faults(edited_version, chain) == ["decon_version"]

    edited_counts = {"at": "t0", "decon_version": 4, "counts": {"kept": 1},
                     "eval_sets": summary["eval_sets"]}
    assert decon_chain_faults(edited_counts, chain) == ["counts"]

    # The exact failure this check exists for: a disclosed hole erased.
    edited_eval_sets = {"at": "t0", "decon_version": 4, "counts": {"kept": 100},
                        "eval_sets": {"bbl": {"status": "ok", "allowed_missing": True,
                                              "items": 500}}}
    assert decon_chain_faults(edited_eval_sets, chain) == ["eval_sets"]


def test_decon_chain_faults_empty_when_the_chain_carries_no_decon_link():
    """Not this function's refusal to make - an absent/broken chain link is
    stats_refusal's job (the chain-completeness gate), which runs first."""
    assert decon_chain_faults({"decon_version": 4}, {}) == []
    assert decon_chain_faults({"decon_version": 4}, None) == []
    assert decon_chain_faults(None, _chain_with_decon(None)) == []


def test_decon_chain_faults_does_not_cross_check_semantic_scripts():
    """The documented residual: dedupe's own summary never carried
    semantic_scripts forward, so an edit that touches ONLY that field is
    invisible to this check - closing it needs an upstream change (dedupe.py
    carrying decontamination.json's own sha256), out of task 15's scope."""
    summary = {"at": "t0", "decon_version": 4, "counts": {}, "eval_sets": {}}
    chain = _chain_with_decon(summary)
    decon = {"at": "t0", "decon_version": 4, "counts": {}, "eval_sets": {},
             "semantic_scripts": {"devanagari": {"screened": True}}}  # edited post-hoc
    assert decon_chain_faults(decon, chain) == []


def test_stats_refusal_on_a_missing_report():
    msg = stats_refusal(None, report_path=Path("out/stats.json"), config_path="c.yaml")
    assert msg is not None and "no stats report" in msg


def test_stats_refusal_on_a_red_report():
    msg = stats_refusal(_report(red=["license"], verdict="red"), report_path=Path("x"),
                        config_path="c.yaml")
    assert msg is not None and "license" in msg and "RED" in msg


def test_stats_refusal_on_an_incomplete_chain_even_when_green():
    detail = {"missing": ["decontamination"], "unrecorded": [], "unverified": []}
    msg = stats_refusal(_report(chain_detail=detail), report_path=Path("x"), config_path="c.yaml")
    assert msg is not None and "custody chain is incomplete" in msg and "decontamination" in msg


def test_stats_refusal_on_a_non_green_verdict_even_with_no_red_gates():
    """R1: `if red or report.get('verdict') != GREEN` is two clauses, not one
    - a report with red == [] but a verdict that is not literally "green"
    must still refuse, independent of whether anything is named in `red`."""
    msg = stats_refusal(_report(red=[], verdict="yellow"), report_path=Path("x"),
                        config_path="c.yaml")
    assert msg is not None and "RED" in msg and "verdict is not green" in msg


# --------------------------------------------------------------------------
# bytes_refusal - pure, both directions.
# --------------------------------------------------------------------------

def test_bytes_refusal_none_when_the_digests_match(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text('{"a": 1}\n', encoding="utf-8")
    eval_ = tmp_path / "eval.jsonl"
    eval_.write_text('{"b": 2}\n', encoding="utf-8")
    report = {"assemble_check": {"input_sha256": {
        str(train): sha256_file(train), str(eval_): sha256_file(eval_),
    }}}
    assert bytes_refusal(report, train_path=train, eval_path=eval_) is None


def test_bytes_refusal_when_a_file_was_rewritten_after_stats_graded_it(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text('{"a": 1}\n', encoding="utf-8")
    eval_ = tmp_path / "eval.jsonl"
    eval_.write_text('{"b": 2}\n', encoding="utf-8")
    report = {"assemble_check": {"input_sha256": {
        str(train): sha256_file(train), str(eval_): sha256_file(eval_),
    }}}
    train.write_text('{"a": 999}\n', encoding="utf-8")  # rewritten
    msg = bytes_refusal(report, train_path=train, eval_path=eval_)
    assert msg is not None and "does not match the bytes stats.py measured" in msg


def test_bytes_refusal_when_the_report_never_recorded_a_digest(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    eval_ = tmp_path / "eval.jsonl"
    eval_.write_text("{}\n", encoding="utf-8")
    msg = bytes_refusal({"assemble_check": {}}, train_path=train, eval_path=eval_)
    assert msg is not None and "no recorded digest" in msg


# --------------------------------------------------------------------------
# Card building blocks - pure, both directions.
# --------------------------------------------------------------------------

def test_license_rows_reads_stats_license_counts():
    report = {"gates": {"license": {"detail": {"counts": {"Apache-2.0": 40, "CC-BY-4.0": 60}}}}}
    assert license_rows(report) == [("Apache-2.0", 40), ("CC-BY-4.0", 60)]


def test_license_rows_refuses_when_the_gate_is_absent():
    with pytest.raises(CardDataMissing, match="license"):
        license_rows({"gates": {}})


def test_license_rows_refuses_when_counts_is_empty():
    with pytest.raises(CardDataMissing, match="license counts"):
        license_rows({"gates": {"license": {"detail": {"counts": {}}}}})


def test_license_rows_renders_an_explicit_unlicensed_row_when_non_zero():
    """I2: `detail["unlicensed"]` sits in the same dict `detail["counts"]`
    comes from - a card that renders one and drops the other under-reports
    its own corpus (Licenses summing to 90 under `total: 100`, while Source
    datasets says `unknown | 10` on the very same card)."""
    report = {"gates": {"license": {"detail": {
        "counts": {"Apache-2.0": 40, "CC-BY-4.0": 50}, "unlicensed": 10, "total": 100,
    }}}}
    rows = license_rows(report)
    assert rows == [("Apache-2.0", 40), ("CC-BY-4.0", 50), ("unlicensed", 10)]
    assert sum(count for _license, count in rows) == 100


def test_license_rows_omits_the_unlicensed_row_when_zero():
    report = {"gates": {"license": {"detail": {
        "counts": {"Apache-2.0": 40, "CC-BY-4.0": 60}, "unlicensed": 0, "total": 100,
    }}}}
    rows = license_rows(report)
    assert rows == [("Apache-2.0", 40), ("CC-BY-4.0", 60)]
    assert "unlicensed" not in dict(rows)


def test_license_rows_refuses_when_counts_and_unlicensed_do_not_sum_to_total():
    """The table must sum to the total stats' report measured - enforced
    here, not left for a human to notice a card that quietly under-reports."""
    report = {"gates": {"license": {"detail": {
        "counts": {"Apache-2.0": 40, "CC-BY-4.0": 50}, "unlicensed": 5, "total": 100,
    }}}}
    with pytest.raises(CardDataMissing, match="sum to 95, not the 100"):
        license_rows(report)


def test_mix_rows_reads_counts_shares_and_targets():
    report = {"gates": {"mix": {"detail": {
        "counts": {"replay": 24}, "shares": {"replay": 0.24}, "targets": {"replay": 0.24},
    }}}}
    assert mix_rows(report) == [("replay", 24, 0.24, 0.24)]


def test_mix_rows_refuses_without_targets():
    with pytest.raises(CardDataMissing, match="mix targets"):
        mix_rows({"gates": {"mix": {"detail": {"counts": {}, "shares": {}, "targets": {}}}}})


def test_source_license_rows_groups_by_source_and_license():
    rows = [
        {"_prov": {"source": "s1", "license": "MIT"}},
        {"_prov": {"source": "s1", "license": "MIT"}},
        {"_prov": {"source": "s2", "license": "Apache-2.0"}},
    ]
    assert source_license_rows(rows) == [("s1", "MIT", 2), ("s2", "Apache-2.0", 1)]


def test_source_license_rows_refuses_on_no_rows():
    with pytest.raises(CardDataMissing, match="no rows"):
        source_license_rows([])


def test_decon_sections_separates_screened_waived_and_script_gaps():
    decon = {"eval_sets": EVAL_SETS, "semantic_scripts": SEMANTIC_SCRIPTS}
    screened, holes, gaps = decon_sections(decon)
    assert [s[0] for s in screened] == ["bbl", "il_tur"]
    assert [h[0] for h in holes] == ["aibe"]
    assert holes[0][2] is True  # allowed_missing
    assert [g[0] for g in gaps] == ["devanagari"]


def test_decon_sections_on_an_empty_manifest_names_nothing():
    assert decon_sections(None) == ([], [], [])
    assert decon_sections({}) == ([], [], [])


def test_decon_sections_gap_list_excludes_script_none_and_control_passed_no_index():
    """I4: mirrors decontaminate.py:3336's own reader - a hole needs BOTH
    "not screened" AND at least one item (eval_items or unscreened_rows) it
    could have caught. SCRIPT_NONE's own manifest text ends "and not a hole",
    and a script whose control PASSED but held no eval items was never a
    hole either. A manifest with one true hole, one SCRIPT_NONE, and one
    control-passed-no-index script must render exactly one gap line."""
    scripts = {
        "devanagari": {  # the true hole: unscreened, with items it could have caught
            "control": "no discriminative power over this script",
            "screened": False, "eval_items": 100, "unscreened_rows": 0,
        },
        SCRIPT_NONE: {  # counted, explicitly not a hole
            "control": "no letters in these probes at all, so there is no eval question "
                      "in them to miss - counted rather than screened, and not a hole",
            "screened": False, "eval_items": 0, "unscreened_rows": 0,
        },
        "gujarati": {  # control passed, no index built - nothing to have missed
            "control": "passed", "screened": False, "eval_items": 0, "unscreened_rows": 0,
        },
    }
    _screened, _holes, gaps = decon_sections({"semantic_scripts": scripts})
    assert [g[0] for g in gaps] == ["devanagari"]


# --------------------------------------------------------------------------
# render_card - content pinned, no placeholder survives.
# --------------------------------------------------------------------------

def _card_fixture():
    report = {
        "measurements": {"rows": 100},
        "sides": {"train": 90, "eval": 10},
        "profile": "v1.1-full",
        "tokenizer": {"repo": "tantan01/qwen3-8b", "revision": "deadbeef"},
        "at": "2026-08-17T00:00:00Z",
        "verdict": "green",
        "gates": {
            "license": {"detail": {"counts": {"Apache-2.0": 40, "CC-BY-4.0": 60}}},
            "mix": {"detail": {
                "counts": {"grounded_synthesis": 60, "curated": 16, "replay": 24},
                "shares": {"grounded_synthesis": 0.60, "curated": 0.16, "replay": 0.24},
                "targets": {"grounded_synthesis": 0.60, "curated": 0.16, "replay": 0.24},
            }},
        },
    }
    decon = {"decon_version": 4, "eval_sets": EVAL_SETS, "semantic_scripts": SEMANTIC_SCRIPTS}
    rows = [row("q", "a", source="synthesis", license="CC-BY-4.0") for _ in range(60)]
    push_cfg = PushCfg(repo_id="tantan01/tuned-law-v1-data", private=True)
    versions = {"decontaminate": 4, "dedupe": 4, "split": 1, "assemble": 2, "stats": 1, "push": 1}
    return report, decon, rows, push_cfg, versions


def test_render_card_has_no_surviving_placeholder():
    import re

    report, decon, rows, push_cfg, versions = _card_fixture()
    card = render_card(report=report, decon=decon, rows=rows, push_cfg=push_cfg, versions=versions)
    assert re.search(r"\{[a-zA-Z_]+\}", card) is None


def test_render_card_states_measured_numbers():
    report, decon, rows, push_cfg, versions = _card_fixture()
    card = render_card(report=report, decon=decon, rows=rows, push_cfg=push_cfg, versions=versions)
    assert "total: 100 (train 90, eval 10)" in card
    assert "| Apache-2.0 | 40 |" in card
    assert "| CC-BY-4.0 | 60 |" in card
    assert "| grounded_synthesis | 60 | 60.0% | 60% |" in card


def test_render_card_names_screened_sets_waived_holes_and_script_gaps():
    report, decon, rows, push_cfg, versions = _card_fixture()
    card = render_card(report=report, decon=decon, rows=rows, push_cfg=push_cfg, versions=versions)
    assert "**bbl**" in card and "**il_tur**" in card
    assert "**aibe**" in card and "waived" in card
    assert "**devanagari**" in card and "not screened" in card
    # "latin" IS screened (screened: True in the fixture), so it must never
    # appear as a per-script GAP - only entries decon_sections excludes ever
    # reach the "not screened" loop.
    assert "**latin**" not in card


def test_render_card_refuses_without_a_decon_version():
    report, _decon, rows, push_cfg, versions = _card_fixture()
    with pytest.raises(CardDataMissing, match="decon_version"):
        render_card(report=report, decon={}, rows=rows, push_cfg=push_cfg, versions=versions)


def _delete_rows(report):
    del report["measurements"]["rows"]


def _delete_sides(report):
    del report["sides"]


def _delete_profile(report):
    del report["profile"]


def _delete_tokenizer(report):
    del report["tokenizer"]


def _delete_license_gate(report):
    del report["gates"]["license"]


def _delete_mix_gate(report):
    report["gates"]["mix"]["detail"]["targets"] = {}


@pytest.mark.parametrize("mutate, match", [
    (_delete_rows, "row count"),
    (_delete_sides, "sides"),
    (_delete_profile, "profile"),
    (_delete_tokenizer, "tokenizer"),
    (_delete_license_gate, "license"),
    (_delete_mix_gate, "mix targets"),
])
def test_render_card_raises_named_for_every_missing_report_field(mutate, match):
    """I3: render_card's own docstring promises every value either comes
    from a real input or raises CardDataMissing naming the path - no third
    behavior (a report missing `sides` must never render "train 0, eval 0"
    beside a real total). Parametrized over every field the card prints:
    delete its source, assert the named raise."""
    report, decon, rows, push_cfg, versions = _card_fixture()
    mutate(report)
    with pytest.raises(CardDataMissing, match=match):
        render_card(report=report, decon=decon, rows=rows, push_cfg=push_cfg, versions=versions)


def test_render_card_raises_named_when_there_are_no_rows():
    report, decon, _rows, push_cfg, versions = _card_fixture()
    with pytest.raises(CardDataMissing, match="no rows"):
        render_card(report=report, decon=decon, rows=[], push_cfg=push_cfg, versions=versions)


def test_render_card_raises_named_when_the_decon_manifest_is_absent():
    report, _decon, rows, push_cfg, versions = _card_fixture()
    with pytest.raises(CardDataMissing, match="decon_version"):
        render_card(report=report, decon=None, rows=rows, push_cfg=push_cfg, versions=versions)


def test_render_card_appends_card_extra_when_given():
    report, decon, rows, push_cfg, versions = _card_fixture()
    card = render_card(report=report, decon=decon, rows=rows, push_cfg=push_cfg, versions=versions,
                       extra="EXTRA MARKER TEXT")
    assert "EXTRA MARKER TEXT" in card


# --------------------------------------------------------------------------
# build_manifest / same_uploaded_bytes.
# --------------------------------------------------------------------------

def test_build_manifest_carries_versions_counts_and_outputs():
    report, decon, _rows, push_cfg, versions = _card_fixture()
    outputs = [{"path": "law_v1_train.jsonl", "rows": 90, "sha256": "a" * 64}]
    # The innermost link (dedupe.manifest_of's upstream_summary) carries NO
    # "stage" key at all - measured on a real chain: keys are exactly
    # {at, counts, decon_version, eval_sets, semantic}. Modelling "stage"
    # here would describe a shape the real pipeline never produces.
    chain = {"stage": "assemble", "assemble_version": 2,
             "split": {"stage": "split", "split_version": 1,
                       "dedupe": {"stage": "dedupe", "dedupe_version": 4,
                                  "decontamination": {"decon_version": 4, "at": "t0",
                                                       "counts": {}, "eval_sets": {},
                                                       "semantic": "ran"}}}}
    manifest = build_manifest(report=report, decon=decon, push_cfg=push_cfg, versions=versions,
                              outputs=outputs, chain=chain)
    assert manifest["push_version"] == PUSH_VERSION
    assert manifest["repo_id"] == "tantan01/tuned-law-v1-data"
    assert manifest["module_versions"] == versions
    assert manifest["counts"] == {"rows": 100, "train": 90, "eval": 10}
    assert manifest["outputs"] == outputs
    assert manifest["decontamination"]["eval_sets"]["aibe"]["allowed_missing"] is True
    # I1: the whole chain, carried forward byte-for-byte - not a pointer.
    assert manifest["chain"] == chain
    assert manifest["chain"]["split"]["dedupe"]["decontamination"]["decon_version"] == 4


def test_same_uploaded_bytes_compares_outputs_only():
    a = {"outputs": [{"path": TRAIN_FILENAME, "sha256": "1"}, {"path": EVAL_FILENAME, "sha256": "2"}],
        "at": "t1"}
    b = {"outputs": [{"path": EVAL_FILENAME, "sha256": "2"}, {"path": TRAIN_FILENAME, "sha256": "1"}],
        "at": "t2"}
    assert same_uploaded_bytes(a, b) is True  # order-independent, timestamp-blind
    c = {"outputs": [{"path": TRAIN_FILENAME, "sha256": "DIFFERENT"},
                     {"path": EVAL_FILENAME, "sha256": "2"}]}
    assert same_uploaded_bytes(a, c) is False
    # N-2 (round 2, mutant N10): TRAIN differing was already pinned above by
    # `c` - nothing varied EVAL alone, so dropping EVAL_FILENAME from
    # _CONTENT_OUTPUTS left the whole suite green. An eval-only corpus
    # change is exactly the half a leak would be planted in (R2's own
    # lesson, one layer up).
    d = {"outputs": [{"path": TRAIN_FILENAME, "sha256": "1"},
                     {"path": EVAL_FILENAME, "sha256": "DIFFERENT"}]}
    assert same_uploaded_bytes(a, d) is False
    assert same_uploaded_bytes(None, a) is False


def test_same_uploaded_bytes_ignores_only_stats_json_bytes():
    """M2 + N-1 (round 2): stats.json embeds the stats report's own `at`
    timestamp directly (it IS that report), so its bytes differ on every
    re-run even over byte-identical train/eval data - excluded, or a
    stats-only re-run would never be a no-op. README.md does NOT get the
    same exemption any more: render_card no longer prints report['at'], so a
    README difference is a REAL card difference (a card_extra change, a
    rendering fix) that must trigger a re-upload, not be silently dropped
    (round 2's N-1 - the bug excluding README used to cause)."""
    a = {"outputs": [
        {"path": TRAIN_FILENAME, "sha256": "1"}, {"path": EVAL_FILENAME, "sha256": "2"},
        {"path": README_FILENAME, "sha256": "readme-v1"},
        {"path": REPORT_FILENAME, "sha256": "stats-v1"},
    ]}
    # stats.json alone differs (a stats-only re-run over the same corpus) ->
    # still a no-op.
    b = {"outputs": [
        {"path": TRAIN_FILENAME, "sha256": "1"}, {"path": EVAL_FILENAME, "sha256": "2"},
        {"path": README_FILENAME, "sha256": "readme-v1"},
        {"path": REPORT_FILENAME, "sha256": "stats-v2-different-timestamp"},
    ]}
    assert same_uploaded_bytes(a, b) is True
    # README alone differs (a genuinely different card over the SAME corpus)
    # -> NOT a no-op; this is the N-1 regression this test now pins shut.
    c = {"outputs": [
        {"path": TRAIN_FILENAME, "sha256": "1"}, {"path": EVAL_FILENAME, "sha256": "2"},
        {"path": README_FILENAME, "sha256": "readme-v2-different-content"},
        {"path": REPORT_FILENAME, "sha256": "stats-v1"},
    ]}
    assert same_uploaded_bytes(a, c) is False


def test_same_uploaded_bytes_raises_named_on_a_non_dict_remote_manifest():
    """M3: a corrupt remote manifest is a named refusal, not a crash."""
    with pytest.raises(RemoteManifestCorrupt, match="not a JSON object"):
        same_uploaded_bytes(["not", "a", "dict"], {"outputs": []})


def test_same_uploaded_bytes_raises_named_when_outputs_is_not_a_list():
    with pytest.raises(RemoteManifestCorrupt, match="outputs is not a list"):
        same_uploaded_bytes({"outputs": "nope"}, {"outputs": []})


def test_same_uploaded_bytes_raises_named_on_a_malformed_outputs_entry():
    with pytest.raises(RemoteManifestCorrupt, match=r"outputs\[0\]"):
        same_uploaded_bytes({"outputs": [{"path": "x"}]}, {"outputs": []})  # no sha256
    with pytest.raises(RemoteManifestCorrupt, match=r"outputs\[0\]"):
        same_uploaded_bytes({"outputs": ["not-a-dict"]}, {"outputs": []})
    with pytest.raises(RemoteManifestCorrupt, match=r"outputs\[0\]"):
        same_uploaded_bytes({"outputs": [{"path": None, "sha256": "1"}]}, {"outputs": []})


# --------------------------------------------------------------------------
# End to end: the real tail, then push --dry-run.
# --------------------------------------------------------------------------

def test_dry_run_renders_a_green_card_and_manifest_with_measured_numbers(tmp_path):
    cfg_path, report, paths = green_pipeline(tmp_path)
    assert push_main(["--config", cfg_path, "--dry-run"]) == 0

    readme = (paths.out_dir / README_FILENAME).read_text(encoding="utf-8")
    assert "total: 100 (train 90, eval 10)" in readme
    assert "| Apache-2.0 | 40 |" in readme
    assert "| CC-BY-4.0 | 60 |" in readme
    assert "| grounded_synthesis | 60 | 60.0% | 60% |" in readme
    assert "| curated | 16 | 16.0% | 16% |" in readme
    assert "| replay | 24 | 24.0% | 24% |" in readme
    assert "**bbl**" in readme and "**aibe**" in readme and "**devanagari**" in readme

    manifest = json.loads((paths.out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["push_version"] == PUSH_VERSION == 1
    assert manifest["counts"] == {"rows": 100, "train": 90, "eval": 10}
    assert manifest["module_versions"] == {
        "decontaminate": 4, "dedupe": 4, "split": 1, "assemble": 2, "stats": 1, "push": 1,
        "extract": "per-document in the build store (document.extract_version); "
                  "not reachable from the file chain",
    }
    paths_out = {o["path"] for o in manifest["outputs"]}
    assert paths_out == {TRAIN_FILENAME, EVAL_FILENAME, README_FILENAME, REPORT_FILENAME}
    # I1: the custody chain carried forward, not a 3-field pointer.
    assert manifest["chain"]["stage"] == "assemble"
    assert manifest["chain"]["split"]["dedupe"]["decontamination"]["decon_version"] == 4
    # A reader of the published set alone can reconstruct the card's
    # screened/waived/per-script-gap claims straight off build_manifest.json:
    assert manifest["decontamination"]["eval_sets"]["bbl"]["status"] == "ok"
    assert manifest["decontamination"]["eval_sets"]["aibe"]["allowed_missing"] is True
    assert manifest["decontamination"]["semantic_scripts"]["devanagari"]["screened"] is False


def test_dry_run_refuses_when_stats_never_ran(tmp_path):
    cfg_path = temp_config(tmp_path)
    paths = paths_for(tmp_path)
    assert not (paths.out_dir / REPORT_FILENAME).exists()
    code = push_main(["--config", cfg_path, "--dry-run"])
    assert code == 2


def test_dry_run_refuses_on_a_red_report(tmp_path, capsys):
    cfg_path, _report, paths = green_pipeline(tmp_path)
    report_path = paths.out_dir / REPORT_FILENAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["red"] = ["license"]
    report["verdict"] = "red"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    code = push_main(["--config", cfg_path, "--dry-run"])
    assert code == 2
    out = capsys.readouterr().out
    assert "RED" in out and "license" in out


def test_dry_run_refuses_when_the_chain_is_incomplete_even_though_stats_is_green(tmp_path):
    from tuned.data.assemble import MANIFEST_FILENAME as ASSEMBLE_MANIFEST

    cfg_path, _report, paths = green_pipeline(tmp_path, require_chain=False)
    manifest_path = paths.out_dir / ASSEMBLE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split"]["dedupe"]["decontamination"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert stats_main(["--config", cfg_path], tokenizer=FakeTokenizer(SCALE)) == 0  # still green

    report = json.loads((paths.out_dir / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["red"] == [] and report["verdict"] == "green"  # the trap this test is for

    code = push_main(["--config", cfg_path, "--dry-run"])
    assert code == 2  # push refuses anyway


@pytest.mark.parametrize("filename", [TRAIN_FILENAME, EVAL_FILENAME])
def test_dry_run_refuses_when_a_file_was_rewritten_after_stats_graded_it(tmp_path, filename):
    """R2: bytes_refusal's loop covers BOTH sides. The eval side is the half
    of the corpus a leak would be planted in - before this test, nothing
    tampered it, so a regression that dropped it from the loop would go
    unnoticed."""
    cfg_path, _report, paths = green_pipeline(tmp_path)
    target = paths.out_dir / filename
    rows = list(read_jsonl(target))
    rows[0]["messages"][1]["content"] += " tampered"
    write_jsonl(target, rows)

    code = push_main(["--config", cfg_path, "--dry-run"])
    assert code == 2


def test_dry_run_refuses_when_a_chain_manifest_went_missing_after_a_green_report(tmp_path):
    from tuned.data.dedupe import MANIFEST_FILENAME as DEDUPE_MANIFEST

    cfg_path, _report, paths = green_pipeline(tmp_path)
    (paths.out_dir / DEDUPE_MANIFEST).unlink()
    code = push_main(["--config", cfg_path, "--dry-run"])
    assert code == 2


def test_dry_run_refuses_when_decontamination_json_itself_went_missing(tmp_path, capsys):
    """The literal chain-link case: the file the card's eval-set/hole/gap
    detail is read from is gone, even though stats' own report - which only
    walks the NESTED tree embedded inside assemble.json, not this standalone
    file - still reports the chain complete."""
    from tuned.data.decontaminate import MANIFEST_FILENAME as DECON_MANIFEST

    cfg_path, _report, paths = green_pipeline(tmp_path)
    (paths.out_dir / DECON_MANIFEST).unlink()
    code = push_main(["--config", cfg_path, "--dry-run"])
    assert code == 2
    out = capsys.readouterr().out
    assert "decontaminate" in out


def test_dry_run_refuses_when_decontamination_json_was_edited_after_the_chain_was_built(
    tmp_path, capsys,
):
    """N-3 (round 2): decontamination.json disagreeing with the custody
    chain's own record of it (dedupe.py's upstream_summary) is a named
    refusal - the exact failure mode a disclosed hole silently vanishing
    needs. Edits ONLY eval_sets (waiving a previously-unwaived hole), which
    the cross-check's `eval_sets` field catches directly."""
    from tuned.data.decontaminate import MANIFEST_FILENAME as DECON_MANIFEST

    cfg_path, _report, paths = green_pipeline(tmp_path)
    decon_path = paths.out_dir / DECON_MANIFEST
    decon = json.loads(decon_path.read_text(encoding="utf-8"))
    decon["eval_sets"]["aibe"]["allowed_missing"] = False  # edited post-hoc
    decon_path.write_text(json.dumps(decon), encoding="utf-8")

    code = push_main(["--config", cfg_path, "--dry-run"])
    assert code == 2
    out = capsys.readouterr().out
    assert "disagrees with the custody chain" in out and "eval_sets" in out


def test_module_versions_reads_every_reachable_stage(tmp_path):
    cfg_path, report, paths = green_pipeline(tmp_path)
    versions = module_versions(paths.out_dir, report)
    assert versions == {
        "decontaminate": 4, "dedupe": 4, "split": 1, "assemble": 2, "stats": 1, "push": 1,
        # M1: extract.py's absence named, not silent - see module_versions'
        # own docstring for why a string here changes nothing about the gate.
        "extract": "per-document in the build store (document.extract_version); "
                  "not reachable from the file chain",
    }


def test_module_versions_reads_stats_version_off_the_report_not_a_constant(tmp_path):
    """R9 (review mutation harness): `versions['stats']` must come from
    `report['stats_version']`, not a hardcoded literal. STATS_VERSION
    happens to be 1 today, which makes a hardcoded-1 mutant tautological
    against every OTHER fixture in this file (both are 1 either way) - only
    a report naming a version that is NOT 1 can tell the two apart."""
    cfg_path, report, paths = green_pipeline(tmp_path)
    fake_report = dict(report, stats_version=99)
    versions = module_versions(paths.out_dir, fake_report)
    assert versions["stats"] == 99


def test_dry_run_needs_no_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg_path, _report, _paths = green_pipeline(tmp_path)
    assert push_main(["--config", cfg_path, "--dry-run"]) == 0


def test_a_config_with_no_push_block_refuses_by_name(tmp_path):
    cfg_path, _report, _paths = green_pipeline(tmp_path)
    text = Path(cfg_path).read_text(encoding="utf-8")
    patched = text.replace(
        "push:\n  repo_id: tantan01/tuned-law-v1-data\n  private: true\n", ""
    )
    assert patched != text
    Path(cfg_path).write_text(patched, encoding="utf-8")
    code = push_main(["--config", cfg_path, "--dry-run"])
    assert code == 2


def test_dry_run_refuses_when_the_report_does_not_name_exactly_two_inputs(tmp_path):
    """R12: `if len(inputs) != 2` has no test - a stats report naming three
    (or zero) input files must refuse by name rather than IndexError on
    `inputs[1]`."""
    cfg_path, _report, paths = green_pipeline(tmp_path)
    report_path = paths.out_dir / REPORT_FILENAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["inputs"] = list(report["inputs"]) + ["extra.jsonl"]
    report_path.write_text(json.dumps(report), encoding="utf-8")

    code = push_main(["--config", cfg_path, "--dry-run"])
    assert code == 2


def test_dry_run_refuses_when_push_card_extra_names_a_missing_file(tmp_path):
    """R11: `if not extra_path.exists()` has no coverage at main() level."""
    cfg_path, _report, paths = green_pipeline(tmp_path)
    text = Path(cfg_path).read_text(encoding="utf-8")
    patched = text.replace(
        "push:\n  repo_id: tantan01/tuned-law-v1-data\n  private: true\n",
        "push:\n  repo_id: tantan01/tuned-law-v1-data\n  private: true\n"
        "  card_extra: nope/absent.md\n",
    )
    assert patched != text
    Path(cfg_path).write_text(patched, encoding="utf-8")

    code = push_main(["--config", cfg_path, "--dry-run"])
    assert code == 2


def test_dry_run_appends_card_extra_content_when_the_file_exists(tmp_path):
    """R11's other half: the read-and-append path, also uncovered at main()
    level before this."""
    cfg_path, _report, paths = green_pipeline(tmp_path)
    extra_path = paths.out_dir / "extra.md"
    extra_path.write_text("EXTRA MARKER TEXT FROM CARD_EXTRA", encoding="utf-8")
    text = Path(cfg_path).read_text(encoding="utf-8")
    patched = text.replace(
        "push:\n  repo_id: tantan01/tuned-law-v1-data\n  private: true\n",
        f"push:\n  repo_id: tantan01/tuned-law-v1-data\n  private: true\n"
        f"  card_extra: {extra_path.as_posix()}\n",
    )
    assert patched != text
    Path(cfg_path).write_text(patched, encoding="utf-8")

    code = push_main(["--config", cfg_path, "--dry-run"])
    assert code == 0
    readme = (paths.out_dir / README_FILENAME).read_text(encoding="utf-8")
    assert "EXTRA MARKER TEXT FROM CARD_EXTRA" in readme


# --------------------------------------------------------------------------
# The live path, through the fake - upload, idempotence, and the missing
# token refusal.
# --------------------------------------------------------------------------

def test_missing_token_refuses_on_a_non_dry_run(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("tuned.data.push.load_dotenv_keys", lambda *a, **kw: 0)
    cfg_path, _report, _paths = green_pipeline(tmp_path)
    code = push_main(["--config", cfg_path], hub_client=FakeHubClient())
    assert code == 2


def test_a_live_push_uploads_through_the_fake_and_prints_the_revision(tmp_path, monkeypatch, capsys):
    # M6: stub load_dotenv_keys so this test's HF_TOKEN monkeypatch is the
    # only source of the token - load_dotenv_keys does os.environ.setdefault
    # from inside the SUT, which monkeypatch cannot track or unwind.
    monkeypatch.setattr("tuned.data.push.load_dotenv_keys", lambda *a, **kw: 0)
    monkeypatch.setenv("HF_TOKEN", "sk-test-dummy")
    cfg_path, _report, paths = green_pipeline(tmp_path)
    from tuned.data.assemble import MANIFEST_FILENAME as ASSEMBLE_MANIFEST

    on_disk_chain = json.loads((paths.out_dir / ASSEMBLE_MANIFEST).read_text(encoding="utf-8"))
    client = FakeHubClient()
    code = push_main(["--config", cfg_path], hub_client=client)
    assert code == 0
    assert client.ensure_calls == 1 and client.ensure_args == ("tantan01/tuned-law-v1-data", True)
    assert client.upload_calls == 1
    # I1: stats.json now travels with the repo too, so the published set can
    # corroborate the card's claims without reaching back to local disk.
    assert set(client.uploaded_files) == {TRAIN_FILENAME, EVAL_FILENAME, README_FILENAME,
                                          MANIFEST_FILENAME, REPORT_FILENAME}
    out = capsys.readouterr().out
    assert "pushed tantan01/tuned-law-v1-data" in out
    assert "revision: rev1" in out

    manifest = json.loads((paths.out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["chain"] == on_disk_chain  # I1: carried forward byte-for-byte
    assert manifest["chain"]["split"]["dedupe"]["decontamination"]["decon_version"] == 4
    # The per-script gap claim the card makes is reconstructable too - not
    # from `chain` (dedupe.py's own summary drops semantic_scripts), but from
    # `decontamination` below, read off the same enriched manifest the card is.
    assert manifest["decontamination"]["semantic_scripts"]["devanagari"]["screened"] is False


def test_a_second_push_of_unchanged_bytes_is_a_no_op(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("tuned.data.push.load_dotenv_keys", lambda *a, **kw: 0)
    monkeypatch.setenv("HF_TOKEN", "sk-test-dummy")
    cfg_path, _report, paths = green_pipeline(tmp_path)
    client = FakeHubClient()
    assert push_main(["--config", cfg_path], hub_client=client) == 0
    assert client.upload_calls == 1

    capsys.readouterr()
    code = push_main(["--config", cfg_path], hub_client=client)
    assert code == 0
    assert client.upload_calls == 1  # NOT uploaded again
    assert client.ensure_calls == 1  # NOT re-created either
    out = capsys.readouterr().out
    assert "no-op" in out and "rev1" in out


def test_a_stats_only_rerun_over_identical_bytes_is_still_a_no_op(tmp_path, monkeypatch):
    """M2: idempotence must survive a stats.py re-run over byte-identical
    data. Re-running ONLY stats.py regenerates stats.json (which embeds the
    report's own `at` directly - it IS that report) with a fresh timestamp
    even though train.jsonl/eval.jsonl are untouched, and stats.json is
    excluded from the idempotence key for exactly that reason;
    same_uploaded_bytes must still call this a no-op, derived from the
    corpus + card bytes, not the report's clock. This is HALF the
    idempotence contract - see the paired test right below for the other
    half (a genuinely different card must NOT be swallowed the same way)."""
    monkeypatch.setattr("tuned.data.push.load_dotenv_keys", lambda *a, **kw: 0)
    monkeypatch.setenv("HF_TOKEN", "sk-test-dummy")
    cfg_path, _report, paths = green_pipeline(tmp_path)
    client = FakeHubClient()
    assert push_main(["--config", cfg_path], hub_client=client) == 0
    assert client.upload_calls == 1

    train_before = (paths.out_dir / TRAIN_FILENAME).read_bytes()
    eval_before = (paths.out_dir / EVAL_FILENAME).read_bytes()
    readme_before = (paths.out_dir / README_FILENAME).read_bytes()
    stats_before = (paths.out_dir / REPORT_FILENAME).read_text(encoding="utf-8")

    assert stats_main(["--config", cfg_path], tokenizer=FakeTokenizer(SCALE)) == 0
    assert (paths.out_dir / TRAIN_FILENAME).read_bytes() == train_before
    assert (paths.out_dir / EVAL_FILENAME).read_bytes() == eval_before
    stats_after = (paths.out_dir / REPORT_FILENAME).read_text(encoding="utf-8")
    assert stats_after != stats_before  # the trap: a fresh `at`, same corpus

    code = push_main(["--config", cfg_path], hub_client=client)
    assert code == 0
    # The re-rendered LOCAL card is also unchanged - render_card no longer
    # prints report['at'], so nothing in it varies with the re-run either.
    assert (paths.out_dir / README_FILENAME).read_bytes() == readme_before
    assert client.upload_calls == 1  # NOT uploaded again - the corpus never changed


def test_a_card_only_change_triggers_an_upload_not_a_stale_no_op(tmp_path, monkeypatch):
    """N-1 (round 2): the OTHER idempotence direction, paired with the test
    above. Excluding README from the idempotence key (the original M2 fix)
    made a card that legitimately changed over an unchanged corpus
    unreachable on an already-pushed repo, while push claimed "already
    carries these exact bytes" - false about README, one of the four files
    that push. A push.card_extra addition over an untouched corpus is the
    concrete case the round-2 review reproduced; it must upload again."""
    monkeypatch.setattr("tuned.data.push.load_dotenv_keys", lambda *a, **kw: 0)
    monkeypatch.setenv("HF_TOKEN", "sk-test-dummy")
    cfg_path, _report, paths = green_pipeline(tmp_path)
    client = FakeHubClient()
    assert push_main(["--config", cfg_path], hub_client=client) == 0
    assert client.upload_calls == 1

    train_before = (paths.out_dir / TRAIN_FILENAME).read_bytes()
    eval_before = (paths.out_dir / EVAL_FILENAME).read_bytes()

    extra_path = paths.out_dir / "extra.md"
    extra_path.write_text("NEW-DISCLOSURE-MARKER", encoding="utf-8")
    text = Path(cfg_path).read_text(encoding="utf-8")
    patched = text.replace(
        "push:\n  repo_id: tantan01/tuned-law-v1-data\n  private: true\n",
        f"push:\n  repo_id: tantan01/tuned-law-v1-data\n  private: true\n"
        f"  card_extra: {extra_path.as_posix()}\n",
    )
    assert patched != text
    Path(cfg_path).write_text(patched, encoding="utf-8")

    code = push_main(["--config", cfg_path], hub_client=client)
    assert code == 0
    assert (paths.out_dir / TRAIN_FILENAME).read_bytes() == train_before
    assert (paths.out_dir / EVAL_FILENAME).read_bytes() == eval_before  # corpus untouched
    assert client.upload_calls == 2  # the card change reached the repo, not swallowed
    uploaded_readme = client.uploaded_files[README_FILENAME].read_text(encoding="utf-8")
    assert "NEW-DISCLOSURE-MARKER" in uploaded_readme


def test_a_corrupt_remote_manifest_is_a_named_refusal_not_a_crash(tmp_path, monkeypatch, capsys):
    """M3: data from outside this machine (the repo's own previous
    build_manifest.json) must refuse by name, never crash with a bare
    TypeError/AttributeError."""
    monkeypatch.setattr("tuned.data.push.load_dotenv_keys", lambda *a, **kw: 0)
    monkeypatch.setenv("HF_TOKEN", "sk-test-dummy")
    cfg_path, _report, _paths = green_pipeline(tmp_path)
    client = FakeHubClient(existing_manifest={"outputs": "not-a-list"}, existing_revision="rev0")

    code = push_main(["--config", cfg_path], hub_client=client)
    assert code == 2
    out = capsys.readouterr().out
    assert "REFUSES TO RUN" in out and "outputs is not a list" in out
    assert client.upload_calls == 0


def test_changed_bytes_are_not_treated_as_a_no_op(tmp_path, monkeypatch):
    """Idempotence is content-bound: a re-run over a DIFFERENT green corpus
    uploads again rather than reusing the stale no-op path.

    The extra row must be one that actually SURVIVES assemble.py (a real
    <think> scaffold via `traced()`, not a bare `row()` with no scaffold at
    all, which assemble.py's own no_think_scaffold filter drops) - now that
    M2 measures idempotence off train/eval bytes alone, a row that never
    reaches the shipped files would make this test pass for the wrong
    reason: the "different" corpus would ship byte-identical either way.
    """
    monkeypatch.setattr("tuned.data.push.load_dotenv_keys", lambda *a, **kw: 0)
    monkeypatch.setenv("HF_TOKEN", "sk-test-dummy")
    cfg_path, _report, paths = green_pipeline(tmp_path)
    client = FakeHubClient()
    assert push_main(["--config", cfg_path], hub_client=client) == 0
    assert client.upload_calls == 1

    train_before = (paths.out_dir / TRAIN_FILENAME).read_bytes()
    eval_before = (paths.out_dir / EVAL_FILENAME).read_bytes()

    extra_rows = e2e_corpus() + [traced(50_000, source="synthesis", license_="CC-BY-4.0")]
    codes, report, _paths = run_pipeline(tmp_path, extra_rows, cfg_path=cfg_path)
    assert codes == {"dedupe": 0, "split": 0, "assemble": 0, "stats": 0}
    assert report["verdict"] == "green" and report["red"] == []
    # The trap this test is for: the corpus really did change on disk.
    assert (paths.out_dir / TRAIN_FILENAME).read_bytes() != train_before or \
        (paths.out_dir / EVAL_FILENAME).read_bytes() != eval_before
    enrich_decon(paths)
    assert push_main(["--config", cfg_path], hub_client=client) == 0
    assert client.upload_calls == 2


@pytest.mark.live
def test_a_real_push_round_trips_through_the_real_hub(tmp_path):
    """Skipped by default: touches the real Hub and needs a real HF_TOKEN with
    write access to a scratch namespace. Not part of the default suite run."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN not set - this test touches the real HuggingFace Hub")
    pytest.skip("real-Hub round trip is an operator-run smoke test, not CI")


# --------------------------------------------------------------------------
# Conventions.
# --------------------------------------------------------------------------

def test_cli_hard_exits_after_success():
    assert "os._exit(" in PUSH_SRC.read_text(encoding="utf-8")


def test_the_version_ledger_describes_the_version_the_module_ships():
    import re

    source = PUSH_SRC.read_text(encoding="utf-8")
    entries = [int(n) for n in re.findall(r"^# (\d+)  ", source, re.M)]
    assert entries == sorted(entries)
    assert entries[-1] == PUSH_VERSION
    assert entries[0] == 1


def test_the_repo_id_is_never_hardcoded_in_the_module():
    """repo_id comes from cfg.push.repo_id, never from module source - a
    module that hardcoded the production repo id could not be pointed at a
    scratch namespace for a test or a dry run."""
    source = PUSH_SRC.read_text(encoding="utf-8")
    assert "tantan01" not in source


def test_the_hub_client_seam_is_the_only_place_that_imports_the_library():
    """Everything else in the module loads with no huggingface_hub install -
    every --dry-run test above already proved this by running without it."""
    source = PUSH_SRC.read_text(encoding="utf-8")
    lines = source.splitlines()
    top_level_imports = [
        ln for ln in lines[: lines.index("class RealHubClient:")]
        if ln.startswith("import huggingface_hub") or ln.startswith("from huggingface_hub")
    ]
    assert top_level_imports == []
