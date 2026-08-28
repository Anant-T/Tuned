"""roles_infer.py - the subprocess bridge to OpenNyAI, both ends.

No real OpenNyAI model runs anywhere in this file (it is not installed in
this repository's environment, and this module's own contract is that its
absence is never fatal). The PROTOCOL - spawn, feed, parse, timeout, crash -
is exercised with a fake `spawn` callable standing in for subprocess.run,
plus one real-subprocess test that spawns this module's own worker mode
against THIS interpreter (which genuinely lacks opennyai) to prove the wiring
end to end without needing the real package.
"""

import json
import subprocess
import sys
from io import StringIO

import pytest

from tuned.data.roles_infer import (
    BACKEND_NONE,
    BACKEND_SUBPROCESS,
    RoleSpan,
    RolesBridgeError,
    RolesResult,
    _run_worker,
    infer_roles,
)

# --------------------------------------------------------------------------
# A fake subprocess.run - the double the protocol tests are built on.
# --------------------------------------------------------------------------


class FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def fake_spawn(reply=None, *, returncode=0, stdout=None, stderr="", raises=None):
    """A `spawn=` double: records the call, returns/raises what it is told to."""
    calls = []

    def run(args, *, input, capture_output, text, timeout):
        calls.append({"args": args, "input": input, "timeout": timeout})
        if raises is not None:
            raise raises
        body = stdout if stdout is not None else json.dumps(reply if reply is not None else {})
        return FakeCompleted(returncode=returncode, stdout=body, stderr=stderr)

    run.calls = calls
    return run


# --------------------------------------------------------------------------
# BACKEND_NONE: no spawn, ever.
# --------------------------------------------------------------------------


def test_backend_none_never_spawns_and_returns_empty():
    def boom(*a, **k):
        raise AssertionError("BACKEND_NONE must never call spawn")

    result = infer_roles("some text", backend=BACKEND_NONE, spawn=boom)
    assert result == RolesResult()
    assert result.spans == ()
    assert result.labels() == ()


def test_unknown_backend_is_refused_before_anything_spawns():
    def boom(*a, **k):
        raise AssertionError("an unknown backend must not spawn")

    with pytest.raises(ValueError, match="unknown roles backend"):
        infer_roles("x", backend="not-a-real-backend", spawn=boom)


# --------------------------------------------------------------------------
# BACKEND_SUBPROCESS: the happy path.
# --------------------------------------------------------------------------


def test_subprocess_backend_parses_a_clean_reply():
    spawn = fake_spawn(reply={"spans": [[0, 10, "FAC"], [10, 25, "ISSUE"]]})
    result = infer_roles("judgment text here", backend=BACKEND_SUBPROCESS, spawn=spawn)
    assert result.spans == (RoleSpan(0, 10, "FAC"), RoleSpan(10, 25, "ISSUE"))
    assert result.labels() == ("FAC", "ISSUE")
    # The wire format: one JSON line of {"text": ...} in.
    sent = json.loads(spawn.calls[0]["input"])
    assert sent == {"text": "judgment text here"}


def test_subprocess_backend_can_return_no_spans_without_erroring():
    spawn = fake_spawn(reply={"spans": []})
    result = infer_roles("x", backend=BACKEND_SUBPROCESS, spawn=spawn)
    assert result.spans == ()


def test_labels_deduplicates_in_first_seen_order():
    result = RolesResult(
        spans=(RoleSpan(0, 5, "FAC"), RoleSpan(5, 9, "ISSUE"), RoleSpan(9, 12, "FAC"))
    )
    assert result.labels() == ("FAC", "ISSUE")


def test_python_bin_defaults_to_this_interpreter():
    spawn = fake_spawn(reply={"spans": []})
    infer_roles("x", backend=BACKEND_SUBPROCESS, spawn=spawn)
    assert spawn.calls[0]["args"][0] == sys.executable


def test_python_bin_override_is_honoured():
    spawn = fake_spawn(reply={"spans": []})
    infer_roles("x", backend=BACKEND_SUBPROCESS, python_bin="/opt/py312/bin/python", spawn=spawn)
    assert spawn.calls[0]["args"][0] == "/opt/py312/bin/python"


def test_worker_module_and_flag_are_on_the_command_line():
    spawn = fake_spawn(reply={"spans": []})
    infer_roles("x", backend=BACKEND_SUBPROCESS, spawn=spawn)
    args = spawn.calls[0]["args"]
    assert args[1:] == ["-m", "tuned.data.roles_infer", "--worker"]


def test_timeout_is_forwarded_to_spawn():
    spawn = fake_spawn(reply={"spans": []})
    infer_roles("x", backend=BACKEND_SUBPROCESS, timeout=42.5, spawn=spawn)
    assert spawn.calls[0]["timeout"] == 42.5


# --------------------------------------------------------------------------
# BACKEND_SUBPROCESS: every failure mode, each its own RolesBridgeError.kind.
# --------------------------------------------------------------------------


def test_missing_interpreter_is_spawn_failed():
    spawn = fake_spawn(raises=FileNotFoundError("no such file"))
    with pytest.raises(RolesBridgeError) as excinfo:
        infer_roles("x", backend=BACKEND_SUBPROCESS, python_bin="/does/not/exist", spawn=spawn)
    assert excinfo.value.kind == "spawn_failed"


def test_timeout_is_reported_as_timeout_not_a_crash():
    spawn = fake_spawn(raises=subprocess.TimeoutExpired(cmd="worker", timeout=5))
    with pytest.raises(RolesBridgeError) as excinfo:
        infer_roles("x", backend=BACKEND_SUBPROCESS, timeout=5, spawn=spawn)
    assert excinfo.value.kind == "timeout"


def test_nonzero_exit_is_crashed_and_carries_stderr():
    spawn = fake_spawn(returncode=1, stdout="", stderr="Traceback: something exploded")
    with pytest.raises(RolesBridgeError) as excinfo:
        infer_roles("x", backend=BACKEND_SUBPROCESS, spawn=spawn)
    assert excinfo.value.kind == "crashed"
    assert "something exploded" in str(excinfo.value)


def test_exit_zero_but_unparsable_stdout_is_bad_output():
    spawn = fake_spawn(returncode=0, stdout="not json at all")
    with pytest.raises(RolesBridgeError) as excinfo:
        infer_roles("x", backend=BACKEND_SUBPROCESS, spawn=spawn)
    assert excinfo.value.kind == "bad_output"


def test_exit_zero_empty_stdout_is_bad_output_not_a_silent_empty_result():
    # The distinction this whole module exists to preserve: "produced
    # nothing parsable" must never look like "produced zero spans".
    spawn = fake_spawn(returncode=0, stdout="")
    with pytest.raises(RolesBridgeError) as excinfo:
        infer_roles("x", backend=BACKEND_SUBPROCESS, spawn=spawn)
    assert excinfo.value.kind == "bad_output"


def test_reply_missing_the_spans_field_is_bad_output():
    spawn = fake_spawn(reply={"unexpected": "shape"})
    with pytest.raises(RolesBridgeError) as excinfo:
        infer_roles("x", backend=BACKEND_SUBPROCESS, spawn=spawn)
    assert excinfo.value.kind == "bad_output"


def test_spans_field_with_the_wrong_shape_is_bad_output():
    spawn = fake_spawn(reply={"spans": [["not", "a", "triple", "extra"]]})
    with pytest.raises(RolesBridgeError) as excinfo:
        infer_roles("x", backend=BACKEND_SUBPROCESS, spawn=spawn)
    assert excinfo.value.kind == "bad_output"


def test_a_reversed_span_is_bad_output_not_a_result_the_caller_has_to_survive():
    # Span CONTENT, which the shape check above does not reach: [10, 3] is a
    # well-typed triple and a malformed interval. segment.py turns each span
    # into a Segment, and Segment's own ValueError is not a RolesBridgeError
    # - so an unvalidated one travelled out of segment_document and ended
    # the whole chunking pass on one bad reply.
    spawn = fake_spawn(reply={"spans": [[0, 5, "FAC"], [10, 3, "ANALYSIS"]]})
    with pytest.raises(RolesBridgeError) as excinfo:
        infer_roles("x" * 40, backend=BACKEND_SUBPROCESS, spawn=spawn)
    assert excinfo.value.kind == "bad_output"
    assert "not an interval" in str(excinfo.value)


def test_a_negative_span_start_is_bad_output():
    spawn = fake_spawn(reply={"spans": [[-1, 5, "FAC"]]})
    with pytest.raises(RolesBridgeError) as excinfo:
        infer_roles("x" * 40, backend=BACKEND_SUBPROCESS, spawn=spawn)
    assert excinfo.value.kind == "bad_output"


def test_a_zero_length_span_is_accepted_as_an_interval():
    # The edge the validation must NOT reject: start == end is a degenerate
    # but well-formed interval, and Segment allows it.
    spawn = fake_spawn(reply={"spans": [[5, 5, "FAC"]]})
    assert infer_roles("x" * 40, backend=BACKEND_SUBPROCESS, spawn=spawn).spans[0].end == 5


def test_a_span_running_past_the_text_is_returned_for_the_caller_to_clip():
    # Deliberately NOT an error: segment._normalize_segments clips it
    # forward to len(text), because a model overshooting its last span is a
    # nuisance to repair rather than a reply to throw away. The two halves
    # of M3 are split on purpose and this pins which half owns which case.
    spawn = fake_spawn(reply={"spans": [[0, 10_000, "FAC"]]})
    result = infer_roles("x" * 40, backend=BACKEND_SUBPROCESS, spawn=spawn)
    assert result.spans[0].end == 10_000


def test_worker_writes_multiple_lines_only_the_last_is_read_as_the_reply():
    # A worker that logs progress to stdout before its final JSON line must
    # not be mistaken for a broken one - only the LAST line is the protocol.
    spawn = fake_spawn(stdout="loading model...\n" + json.dumps({"spans": []}))
    result = infer_roles("x", backend=BACKEND_SUBPROCESS, spawn=spawn)
    assert result.spans == ()


# --------------------------------------------------------------------------
# The worker side, in-process (no subprocess needed to exercise its logic).
# --------------------------------------------------------------------------


def test_worker_reports_missing_opennyai_as_a_nonzero_exit_not_empty_spans():
    stdin = StringIO(json.dumps({"text": "hello"}))
    stdout = StringIO()
    code = _run_worker(stdin, stdout)
    # opennyai is not installed in this environment - this is the real path,
    # not a mock of it.
    assert code == 3
    assert stdout.getvalue() == ""


def test_worker_reports_a_bad_request_as_a_nonzero_exit_with_an_error_body():
    stdin = StringIO("not json")
    stdout = StringIO()
    code = _run_worker(stdin, stdout)
    assert code == 1
    body = json.loads(stdout.getvalue())
    assert "error" in body


def test_worker_reports_a_request_missing_text_as_a_bad_request():
    stdin = StringIO(json.dumps({"nope": "no text field"}))
    stdout = StringIO()
    code = _run_worker(stdin, stdout)
    assert code == 1
    assert "error" in json.loads(stdout.getvalue())


# --------------------------------------------------------------------------
# End to end, through a REAL subprocess (this module spawning itself).
# --------------------------------------------------------------------------


def test_real_subprocess_wiring_reports_opennyai_missing_as_crashed():
    """Spawns `sys.executable -m tuned.data.roles_infer --worker` for real -
    no fake `spawn`, no fake opennyai. Proves the argv/stdin/stdout wiring
    this module claims works, without needing the real package: this
    interpreter genuinely lacks opennyai, so the worker's own ImportError
    path is what answers."""
    with pytest.raises(RolesBridgeError) as excinfo:
        infer_roles("hello world", backend=BACKEND_SUBPROCESS, python_bin=sys.executable, timeout=60)
    assert excinfo.value.kind == "crashed"
    assert "opennyai" in str(excinfo.value)


@pytest.mark.live
def test_real_opennyai_model_produces_role_spans():  # pragma: no cover - needs opennyai
    """Skipped everywhere this repository runs. Documents what the live
    check WOULD assert on a machine with a pre-3.13 interpreter carrying a
    real opennyai install: real judgment text in, at least one recognisable
    rhetorical-role span out, each within the text's own bounds."""
    pytest.skip("no opennyai-equipped interpreter is configured for this repo")
