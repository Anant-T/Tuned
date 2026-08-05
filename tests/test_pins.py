"""Dependency-pin and pre-quantized-load tripwires.

The Kaggle install cell runs `uv pip install --system -e .[dev,train]`, which
re-resolves against live PyPI every run (uv.lock is gitignored and unused), so
any training dep left unpinned in pyproject.toml can drift between otherwise
identical runs.
"""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent
PYPROJECT = ROOT / "pyproject.toml"
SFT = ROOT / "src" / "tuned" / "train" / "sft.py"


def test_training_deps_that_gate_the_load_path_are_pinned():
    train = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]["train"]
    pins = dict(
        re.fullmatch(r"([A-Za-z0-9_-]+)==(.+)", spec).groups()
        for spec in train
        if "==" in spec
    )
    # peft: the unsloth#5677 mitigation depends on peft's re.fullmatch semantics
    # in _check_target_module_exists (verified in 0.20.0).
    assert "peft" in pins, "peft must be ==-pinned in [train]"
    # bitsandbytes: a broken/incompatible bnb flips unsloth's
    # ALLOW_PREQUANTIZED_MODELS to False, which silently strips the
    # -unsloth-bnb-4bit suffix AND drops the pinned model revision.
    assert "bitsandbytes" in pins, "bitsandbytes must be ==-pinned in [train]"


def test_sft_guards_prequantized_load_before_model_download():
    src = SFT.read_text(encoding="utf-8")
    # unsloth 2026.8.3 loader.py strips the -unsloth-bnb-4bit suffix and drops
    # revision= whenever ALLOW_PREQUANTIZED_MODELS is False (bitsandbytes native
    # kernels failed to load). That turns a doomed run into a 28 GB fp16
    # re-download inside the 45-min watchdog. Fail fast instead.
    guard = src.find("ALLOW_PREQUANTIZED_MODELS")
    load = src.find("FastModel.from_pretrained")
    assert guard != -1, "sft.py must check unsloth ALLOW_PREQUANTIZED_MODELS"
    assert load != -1
    assert guard < load, "the ALLOW_PREQUANTIZED_MODELS guard must run before the model load"
