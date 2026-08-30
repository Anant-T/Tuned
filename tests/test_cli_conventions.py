"""The conventions every data CLI shares, in one table instead of 19 copies.

Each convention used to be hand-copied into the test module of whichever CLI
someone remembered to copy it into: 12 copies of the hard-exit check, 4 of the
top-level-import guard, 3 of the importability check. Copies drift, and these
had:

  * Five copies asserted the literal `os._exit(0)` and seven asserted
    `os._exit(`. The strict form is not a stricter test, it is a WRONG one -
    nine of the seventeen CLIs exit with the code `main()` returned
    (`os._exit(exit_code)`, `os._exit(code)`), so the strict form could never
    have been copied onto them. `os._exit(` is the form that means the thing
    the convention is about: shutdown is skipped.
  * acquire's banned set carried boto3/botocore and the other three did not,
    which is correct - it is the only S3 reader - but nothing said so.

A copied test also only covers what it was copied onto. Five CLIs that hard-exit
had no test saying so (citations, decontaminate, dedupe, generate, judge), and
citations.py and smoke.py had no import guard at all.

The check is BIDIRECTIONAL, which is the point of listing the modules that must
NOT hard-exit. `os._exit` skips interpreter shutdown outright: no atexit
handlers, no buffered-writer flush, no context-manager unwinding above the
call. That is correct for a CLI whose pyarrow/hf-xet machinery leaves
non-daemon threads that wedge shutdown after all output is written (the
2026-08-08 Kaggle hang on a finished child), and it is a silent data-loss bug
in a CLI that does not need it. Growing one by copy-paste is as real a
regression as losing one, so both directions fail here.

Honest accounting: this file is slightly LONGER than the 19 bodies it replaces.
It is bought for the five modules that gained coverage, the two that gained a
guard, and one edit point - not for line count.
"""

import ast
import importlib
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src" / "tuned"


def _src(module: str) -> Path:
    return (SRC / Path(*module.split("."))).with_suffix(".py")


# Skips interpreter shutdown because its own work can leave non-daemon threads
# (pyarrow / hf-xet / an abandoned streaming iterator) that hang the process
# AFTER every byte of output is written. Adding a module here means the CLI is
# known to hang without it; removing one means that is no longer true.
HARD_EXIT = (
    "data.acquire",
    "data.assemble",
    "data.chunks",
    "data.citations",
    "data.curated",
    "data.decontaminate",
    "data.dedupe",
    "data.extract",
    "data.generate",
    "data.judge",
    "data.push",
    "data.replay",
    "data.seeds",
    "data.select",
    "data.smoke",
    "data.split",
    "data.stats",
)

# Runs interpreter shutdown normally, and must keep doing so. `data.verify` and
# `data.tasks` sweep the store but hold it in sqlite, not pyarrow; `train.sft`
# is the one that would actually break - it runs under torchrun, where skipping
# shutdown strands rank coordination and any in-flight checkpoint write.
NO_HARD_EXIT = (
    "data.calibrate",
    "data.difficulty",
    "data.eval_matched",
    "data.probe",
    "data.providers",
    "data.reconcile",
    "data.roles_infer",
    "data.shape",
    "data.tasks",
    "data.transition",
    "data.verify",
    "train.sft",
)

_HEAVY = frozenset({"datasets", "pyarrow", "huggingface_hub"})

# Importing the module must not pull a heavy client in at module level - these
# CLIs are imported by tests and by other modules that never call their loading
# paths, and a top-level import is what turns `import tuned.data.replay` into a
# network-touching call. Verified by AST, not sys.modules: the test venv has
# datasets installed for other reasons, so absence-from-sys.modules proves
# nothing.
IMPORT_GUARDS = {
    "data.acquire": _HEAVY | {"boto3", "botocore"},  # the only S3 reader
    "data.citations": _HEAVY,
    "data.curated": _HEAVY,
    "data.replay": _HEAVY,
    "data.seeds": _HEAVY,
    "data.smoke": _HEAVY,
}

# Importing the module raises nothing and the CLI's entry point is really on it
# - catches a module that only imports because nothing ever imported it.
IMPORTABLE = {
    "tuned.data.curated": "build_curated",
    "tuned.data.replay": "build_replay",
    "tuned.data.seeds": "load_seeds",
}


@pytest.mark.parametrize("module", HARD_EXIT)
def test_the_long_running_clis_skip_interpreter_shutdown(module):
    text = _src(module).read_text(encoding="utf-8")
    assert "os._exit(" in text, (
        f"{module} lost its hard exit. If that is deliberate, move it to "
        f"NO_HARD_EXIT with the reason it no longer hangs on shutdown."
    )


@pytest.mark.parametrize("module", NO_HARD_EXIT)
def test_the_short_clis_do_not_grow_a_hard_exit_by_copy_paste(module):
    text = _src(module).read_text(encoding="utf-8")
    assert "os._exit(" not in text, (
        f"{module} grew an os._exit. It skips atexit handlers and buffered "
        f"flushes; add it to HARD_EXIT only with the hang it fixes."
    )


@pytest.mark.parametrize("module", sorted(IMPORT_GUARDS))
def test_module_import_never_touches_the_heavy_clients(module):
    banned = IMPORT_GUARDS[module]
    tree = ast.parse(_src(module).read_text(encoding="utf-8"))
    for node in tree.body:  # module-level statements only, not nested in defs
        if isinstance(node, ast.Import):
            names = {alias.name.split(".")[0] for alias in node.names}
            assert not (names & banned), f"{module}: top-level import of {names & banned}"
        if isinstance(node, ast.ImportFrom) and node.module:
            head = node.module.split(".")[0]
            assert head not in banned, f"{module}: top-level `from {node.module} import ...`"


@pytest.mark.parametrize("module,attr", sorted(IMPORTABLE.items()))
def test_module_importable_without_error(module, attr):
    mod = importlib.import_module(module)
    importlib.reload(mod)
    assert hasattr(mod, attr)


def test_every_cli_module_in_the_table_exists():
    """The table is a list of strings; a renamed module would otherwise make
    every check above vacuously pass on a file that is no longer there."""
    for module in (*HARD_EXIT, *NO_HARD_EXIT, *IMPORT_GUARDS):
        assert _src(module).exists(), module


def test_the_two_hard_exit_lists_are_disjoint_and_cover_every_cli():
    """Every module with a `__main__` block is in exactly one column - the
    guarantee that makes the bidirectional check total rather than a sample."""
    assert not set(HARD_EXIT) & set(NO_HARD_EXIT)
    listed = {f"{m}.py" for m in (*HARD_EXIT, *NO_HARD_EXIT)}
    found = set()
    for path in sorted(SRC.rglob("*.py")):
        if 'if __name__ == "__main__":' in path.read_text(encoding="utf-8"):
            found.add(f"{path.parent.name}.{path.name}")
    assert found == listed, f"unlisted CLI modules: {found ^ listed}"
