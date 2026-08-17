"""The subprocess bridge to OpenNyAI's rhetorical-role model.

WHY A SUBPROCESS AT ALL. OpenNyAI's rhetorical-role classifier is the only
semantic-segmentation option this plan found (P0), and it is not installable
into this repository's own environment: it pins a dependency stack (spaCy
model weights, a specific torch range) that is documented as py>=3.13
incompatible - a constraint about the PACKAGE, not about what Python happens
to run this file. Rather than let one optional tier decide which Python the
whole pipeline runs under, the model lives in its own interpreter (wherever
an operator has one set up with `opennyai` installed) and this module talks
to it over stdin/stdout, one document at a time.

THE PROTOCOL is the whole of what this file guarantees, and it is
deliberately small: spawn `python_bin -m tuned.data.roles_infer --worker`,
write one line of JSON to its stdin (`{"text": ...}`), read one line of JSON
back from its stdout (`{"spans": [[start, end, label], ...]}` or
`{"error": "..."}`), under a timeout. THIS FILE IS BOTH ENDS of that pipe -
the bridge that spawns (`infer_roles`) and the worker that gets spawned
(`_run_worker`, reached via `--worker`) - so the wire format only has to
agree with itself.

`--roles-backend none` (BACKEND_NONE, the default) never spawns anything:
`infer_roles(text, backend=BACKEND_NONE)` returns an empty RolesResult
immediately. That is what "leaves the pipeline fully functional on packing
alone" means at this module's boundary - segment.py never has to ask whether
a subprocess bridge exists before it can chunk a single document.

WHAT THIS MODULE DOES NOT DO: decide what happens when the bridge fails.
`infer_roles` under BACKEND_SUBPROCESS either returns a real RolesResult or
raises RolesBridgeError - spawn failure, timeout and a crashing/malformed
worker are three distinct RolesBridgeError.kind values, not three ways of
silently returning no roles. Collapsing "opennyai is not installed over
there" into an empty result here would make that indistinguishable from "this
document genuinely has no roles", and only the CALLER (segment.py) has the
context to decide that a failure here means "degrade to packing, and record
why" rather than "stop the run". So this module never swallows an error into
an empty success; segment.py is where the degradation policy lives.

Live-model behaviour (spawning a REAL opennyai-equipped interpreter and
checking the roles it returns) is `@pytest.mark.live` and skipped by default,
because no such interpreter exists in this repository's own environment. The
protocol itself - spawn, feed, parse, timeout, crash - is exercised with a
FAKE subprocess (an injectable `spawn` callable), so it is fully tested
without needing OpenNyAI, or even a second real Python, to be present.
"""

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass

# Bump when the wire format or the worker's own behaviour changes in a way
# that could move what a document's role spans are, recorded in every chunk
# that used this tier (segment.py folds it into the chunk's meta).
#
#   1  first cut: one JSON line in (`{"text": ...}`), one JSON line out
#      (`{"spans": [[start, end, label], ...]}` or `{"error": ...}`).
ROLES_VERSION = 1

BACKEND_NONE = "none"
BACKEND_SUBPROCESS = "opennyai-subprocess"
BACKENDS = (BACKEND_NONE, BACKEND_SUBPROCESS)

# Generous by design: the model is a spaCy pipeline over a whole judgment,
# not an API call, and a timeout that fires under ordinary load would make
# "the model is slow today" indistinguishable from "the bridge is broken" -
# exactly the ambiguity RolesBridgeError.kind exists to keep apart.
DEFAULT_TIMEOUT_S = 300.0

_WORKER_ARGS = ("--worker",)


@dataclass(frozen=True)
class RoleSpan:
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class RolesResult:
    spans: tuple[RoleSpan, ...] = ()

    def labels(self) -> tuple[str, ...]:
        """The distinct role labels present, in first-seen order - what
        segment.py folds into a chunk's roles_json."""
        seen: dict[str, None] = {}
        for span in self.spans:
            seen.setdefault(span.label, None)
        return tuple(seen)


class RolesBridgeError(RuntimeError):
    """The subprocess bridge could not produce a result. Actionable by kind.

    `kind` names WHICH of the protocol's failure modes fired - spawn, feed,
    timeout, crash or parse - because the caller's response differs: a
    missing interpreter is a configuration problem worth surfacing loudly on
    every document, while a timeout on one unusually long judgment is not
    evidence about the next one.
    """

    def __init__(self, message: str, *, kind: str):
        super().__init__(message)
        self.kind = kind


# --------------------------------------------------------------------------
# The bridge (parent side): spawn, feed, parse, timeout, crash.
# --------------------------------------------------------------------------


def _python_bin_for(python_bin: str | None) -> str:
    if python_bin:
        return python_bin
    import sys

    return sys.executable


def infer_roles(
    text: str,
    *,
    backend: str = BACKEND_NONE,
    python_bin: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    spawn=subprocess.run,
) -> RolesResult:
    """Rhetorical-role spans for `text`, or an empty result under BACKEND_NONE.

    `spawn` defaults to `subprocess.run` and is the whole of what a test
    injects to fake the subprocess: anything with that signature (`spawn(
    args, input=..., capture_output=True, text=True, timeout=...)` returning
    an object with `.returncode`/`.stdout`/`.stderr`) stands in for a real
    interpreter without one ever being started.

    Raises RolesBridgeError - never returns a result papering over a
    failure - for every way the OTHER interpreter can fail to answer:
    missing binary (`kind="spawn_failed"`), running past `timeout`
    (`kind="timeout"`), a non-zero exit (`kind="crashed"`), or an exit-0
    reply that is not the one line of JSON this protocol promises
    (`kind="bad_output"`).
    """
    if backend == BACKEND_NONE:
        return RolesResult()
    if backend != BACKEND_SUBPROCESS:
        raise ValueError(f"unknown roles backend {backend!r}; known: {BACKENDS}")

    args = [_python_bin_for(python_bin), "-m", "tuned.data.roles_infer", *_WORKER_ARGS]
    payload = json.dumps({"text": text})
    try:
        completed = spawn(args, input=payload, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise RolesBridgeError(
            f"could not spawn the roles-backend interpreter {args[0]!r}: {exc}",
            kind="spawn_failed",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RolesBridgeError(
            f"roles worker exceeded its {timeout}s timeout", kind="timeout"
        ) from exc

    if completed.returncode != 0:
        raise RolesBridgeError(
            f"roles worker exited {completed.returncode}: "
            f"{(completed.stderr or '').strip()[:500]}",
            kind="crashed",
        )
    try:
        reply = json.loads((completed.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RolesBridgeError(
            f"roles worker produced no parsable JSON on stdout: "
            f"{(completed.stdout or '')[:200]!r}",
            kind="bad_output",
        ) from exc
    if not isinstance(reply, dict) or "spans" not in reply:
        raise RolesBridgeError(
            f"roles worker reply carries no 'spans' field: {reply!r}", kind="bad_output"
        )
    try:
        spans = tuple(
            RoleSpan(start=int(s), end=int(e), label=str(label)) for s, e, label in reply["spans"]
        )
    except (TypeError, ValueError) as exc:
        raise RolesBridgeError(
            f"roles worker 'spans' field is not a list of (start, end, label): {exc}",
            kind="bad_output",
        ) from exc
    return RolesResult(spans=spans)


# --------------------------------------------------------------------------
# The worker (child side): what runs inside the OTHER interpreter.
# --------------------------------------------------------------------------


def _opennyai_spans(text: str) -> list[tuple[int, int, str]]:  # pragma: no cover - needs opennyai
    """The one call into the real model. Never reached without opennyai
    installed, and not exercised by any test in this repository - see the
    `@pytest.mark.live` tests for what would cover it on a machine that has
    the package.
    """
    import opennyai  # noqa: F401

    raise NotImplementedError(
        "opennyai is importable but this bridge has not been wired to its real "
        "pipeline API yet - only its absence is handled below"
    )


def _run_worker(stdin, stdout) -> int:
    """Read one request, write one reply, return the process exit code.

    A plain function over injected streams rather than the real
    sys.stdin/sys.stdout, so the worker's OWN logic (opennyai present vs.
    absent, malformed request) is unit-testable in-process without spawning
    anything - the spawn/feed/timeout/crash PROTOCOL is what needs a real or
    fake subprocess; this half is pure.
    """
    try:
        request = json.loads(stdin.read())
        text = request["text"]
    except Exception as exc:
        stdout.write(json.dumps({"error": f"bad request: {type(exc).__name__}: {exc}"}))
        return 1
    try:
        spans = _opennyai_spans(text)
    except ImportError as exc:
        # THE case this bridge exists to make non-fatal: opennyai is not
        # installed in whichever interpreter `python_bin` pointed at. Exit
        # non-zero (this is not a spans result) so `infer_roles` reports it
        # as kind="crashed" with the real reason in stderr, rather than a
        # silent empty roles list segment.py could mistake for "this
        # document has no rhetorical structure".
        import sys

        print(f"opennyai is not importable in this interpreter: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        stdout.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    stdout.write(json.dumps({"spans": [[s.start, s.end, s.label] for s in spans]}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--worker",
        action="store_true",
        required=True,
        help="run as the subprocess bridge's worker (read one JSON request off "
        "stdin, write one JSON reply to stdout) - this is what infer_roles spawns, "
        "not a command an operator runs by hand",
    )
    parser.parse_args(argv)
    return _run_worker(sys.stdin, sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
