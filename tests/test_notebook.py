import json
from pathlib import Path

NB = Path(__file__).parent.parent / "notebooks" / "kaggle_smoke.ipynb"


def test_notebook_is_valid_and_complete():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    sources = ["".join(c["source"]) for c in nb["cells"]]
    joined = "\n".join(sources)
    # the operator's mode switch exists and defaults to the cheap gate
    assert 'MODE = "SAVETEST"' in joined
    # single-GPU pin and scratch cache - never /kaggle/working
    assert 'CUDA_VISIBLE_DEVICES"] = "0"' in joined
    assert "/tmp/hf_cache" in joined
    assert "/kaggle/working/hf_cache" not in joined
    # secrets come from Kaggle, never hardcoded
    assert "UserSecretsClient" in joined
    assert "hf_" not in joined.replace("hf_cache", "").replace("hf_transfer", "").replace("HF_", "")
