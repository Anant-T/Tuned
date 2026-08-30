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
    pins = _pins("train")
    # peft: the unsloth#5677 mitigation depends on peft's re.fullmatch semantics
    # in _check_target_module_exists (verified in 0.20.0).
    assert "peft" in pins, "peft must be ==-pinned in [train]"
    # bitsandbytes: a broken/incompatible bnb flips unsloth's
    # ALLOW_PREQUANTIZED_MODELS to False, which silently strips the
    # -unsloth-bnb-4bit suffix AND drops the pinned model revision.
    assert "bitsandbytes" in pins, "bitsandbytes must be ==-pinned in [train]"


def _pins(extra: str) -> dict[str, str]:
    specs = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ][extra]
    return dict(
        re.fullmatch(r"([A-Za-z0-9_-]+)==(.+)", spec).groups()
        for spec in specs
        if "==" in spec
    )


def test_the_build_extra_pins_the_tokenizer_the_trainer_uses():
    # assemble.py measures every row's token_length against max_seq_length with
    # a tokenizer it loads itself, and stats.py re-measures the same rows; the
    # trainer then trains on what survived. A transformers version skew between
    # the two extras means the builder's ruler is not the trainer's ruler, so
    # the corpus is filtered against a length nobody trains at.
    #
    # This test also guards the blocker it was written for: transformers was
    # ABSENT from [build] entirely, so `tuned.data.assemble` raised
    # ModuleNotFoundError on the CI ship path and no dataset was ever produced.
    build, train = _pins("build"), _pins("train")
    assert "transformers" in build, (
        "transformers must be pinned in [build] - assemble.py:load_tokenizer "
        "imports it, and without it the data-assemble chain dies at step 5 of 6"
    )
    assert "transformers" in train
    assert build["transformers"] == train["transformers"], (
        "the [build] tokenizer pin must equal the [train] pin - a different "
        f"version measures a different token_length ({build['transformers']} "
        f"vs {train['transformers']})"
    )


def _names(extra: str) -> set[str]:
    specs = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ][extra]
    return {re.match(r"[A-Za-z0-9_.-]+", spec).group(0).lower() for spec in specs}


def test_the_workers_extra_is_a_subset_of_the_full_build_extra():
    """The cron installs [build-worker]; the assemble job installs [build] and
    runs the worker's modules too (reconcile, and generate's prompt machinery
    through the gates).

    Two hand-kept lists drift, and the direction that drifts silently is the
    dangerous one: a dependency added to [build] for a module the worker also
    imports leaves the unattended job with a ModuleNotFoundError hours later,
    on a host nobody is watching. So the subset relation is asserted rather
    than remembered.
    """
    worker, build = _names("build-worker"), _names("build")
    assert worker <= build, (
        f"[build-worker] must stay a subset of [build]; extra: {sorted(worker - build)}"
    )
    assert worker, "the worker still has to install something"


def test_boto3_is_only_in_the_extra_named_for_it():
    """acquire.py imports boto3 lazily inside the S3 seam, so nothing else in
    the tree can reach it - and neither CI job takes that path. In [build] it
    cost every unattended run a botocore resolve for code it never ran."""
    assert "boto3" in _names("acquire-s3")
    for extra in ("build", "build-worker", "dev", "train"):
        assert "boto3" not in _names(extra), f"boto3 leaked back into [{extra}]"


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
