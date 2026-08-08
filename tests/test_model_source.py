"""resolve_model_source: the staged-snapshot override for the model load.

The notebook's pre-download cell verifies the staged REVISION.txt against the
config pin BEFORE setting TUNED_MODEL_PATH, so by the time sft.py sees the
override the revision question is settled - a local path carries none.
"""

import pytest

from tuned.train.sft import resolve_model_source


def test_no_override_passes_repo_and_revision_through():
    assert resolve_model_source("unsloth/x", "abc123", None) == ("unsloth/x", "abc123")
    assert resolve_model_source("unsloth/x", "abc123", "") == ("unsloth/x", "abc123")


def test_override_must_look_like_a_snapshot(tmp_path):
    with pytest.raises(SystemExit):
        resolve_model_source("unsloth/x", "abc123", str(tmp_path))


def test_override_returns_path_and_drops_revision(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    name, rev = resolve_model_source("unsloth/x", "abc123", str(tmp_path))
    assert name == str(tmp_path)
    assert rev is None
