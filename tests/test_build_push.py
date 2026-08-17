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
from test_build_stats import SCALE, e2e_corpus, run_pipeline

from tuned.data.acquire import sha256_file
from tuned.data.assemble import main as assemble_main
from tuned.data.config import PushCfg, load_build_config
from tuned.data.dedupe import main as dedupe_main
from tuned.data.jsonl import read_jsonl, write_jsonl
from tuned.data.push import (
    EVAL_FILENAME,
    MANIFEST_FILENAME,
    PUSH_VERSION,
    README_FILENAME,
    TRAIN_FILENAME,
    CardDataMissing,
    build_manifest,
    bytes_refusal,
    chain_faults,
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
DATA_CONFIG = Path(__file__).parent.parent / "configs" / "data_law_v1.yaml"


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
    against - those stay exactly as run_pipeline wrote them."""
    decon_path = paths.out_dir / "decontamination.json"
    decon = json.loads(decon_path.read_text(encoding="utf-8"))
    decon["eval_sets"] = eval_sets
    decon["semantic_scripts"] = semantic_scripts
    decon_path.write_text(json.dumps(decon), encoding="utf-8")
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
    manifest = build_manifest(report=report, decon=decon, push_cfg=push_cfg, versions=versions,
                              outputs=outputs)
    assert manifest["push_version"] == PUSH_VERSION
    assert manifest["repo_id"] == "tantan01/tuned-law-v1-data"
    assert manifest["module_versions"] == versions
    assert manifest["counts"] == {"rows": 100, "train": 90, "eval": 10}
    assert manifest["outputs"] == outputs
    assert manifest["decontamination"]["eval_sets"]["aibe"]["allowed_missing"] is True


def test_same_uploaded_bytes_compares_outputs_only():
    a = {"outputs": [{"path": "x", "sha256": "1"}, {"path": "y", "sha256": "2"}], "at": "t1"}
    b = {"outputs": [{"path": "y", "sha256": "2"}, {"path": "x", "sha256": "1"}], "at": "t2"}
    assert same_uploaded_bytes(a, b) is True  # order-independent, timestamp-blind
    c = {"outputs": [{"path": "x", "sha256": "DIFFERENT"}, {"path": "y", "sha256": "2"}]}
    assert same_uploaded_bytes(a, c) is False
    assert same_uploaded_bytes(None, a) is False


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
    }
    paths_out = {o["path"] for o in manifest["outputs"]}
    assert paths_out == {TRAIN_FILENAME, EVAL_FILENAME, README_FILENAME}


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


def test_dry_run_refuses_when_a_file_was_rewritten_after_stats_graded_it(tmp_path):
    cfg_path, _report, paths = green_pipeline(tmp_path)
    train = paths.out_dir / TRAIN_FILENAME
    rows = list(read_jsonl(train))
    rows[0]["messages"][1]["content"] += " tampered"
    write_jsonl(train, rows)

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


def test_module_versions_reads_every_reachable_stage(tmp_path):
    cfg_path, report, paths = green_pipeline(tmp_path)
    versions = module_versions(paths.out_dir, report)
    assert versions == {
        "decontaminate": 4, "dedupe": 4, "split": 1, "assemble": 2, "stats": 1, "push": 1,
    }


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
    monkeypatch.setenv("HF_TOKEN", "sk-test-dummy")
    cfg_path, _report, paths = green_pipeline(tmp_path)
    client = FakeHubClient()
    code = push_main(["--config", cfg_path], hub_client=client)
    assert code == 0
    assert client.ensure_calls == 1 and client.ensure_args == ("tantan01/tuned-law-v1-data", True)
    assert client.upload_calls == 1
    assert set(client.uploaded_files) == {TRAIN_FILENAME, EVAL_FILENAME, README_FILENAME,
                                          MANIFEST_FILENAME}
    out = capsys.readouterr().out
    assert "pushed tantan01/tuned-law-v1-data" in out
    assert "revision: rev1" in out


def test_a_second_push_of_unchanged_bytes_is_a_no_op(tmp_path, monkeypatch, capsys):
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


def test_changed_bytes_are_not_treated_as_a_no_op(tmp_path, monkeypatch):
    """Idempotence is content-bound: a re-run over a DIFFERENT green corpus
    uploads again rather than reusing the stale no-op path."""
    monkeypatch.setenv("HF_TOKEN", "sk-test-dummy")
    cfg_path, _report, paths = green_pipeline(tmp_path)
    client = FakeHubClient()
    assert push_main(["--config", cfg_path], hub_client=client) == 0
    assert client.upload_calls == 1

    extra_rows = e2e_corpus() + [
        row(prose(50_000, 30), prose(50_001, 25), source="synthesis", license="CC-BY-4.0",
            reasoning=True)
    ]
    run_pipeline(tmp_path, extra_rows, cfg_path=cfg_path)
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
