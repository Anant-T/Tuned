"""Test-suite guards that must not depend on which machine runs the suite.

THE SEMANTIC LAYER IS OPT-IN, EVERYWHERE
----------------------------------------
`import semhash` is made to fail for EVERY test in this suite, so no test runs
the real embedding seam - and therefore fetches a model - unless it says so.
The [build] extra may or may not be installed on any given machine, and a suite
whose drop counts depend on that is not a suite: with semhash present the seam
runs inside every CLI test, downloads a model and flags rows the exact stack
did not.

It lives here rather than in the two modules that own the seam because autouse
fixtures do not cross module boundaries and six test modules already import
`tuned.data.decontaminate` - test_build_citations, test_build_statutes and
test_build_store among them. A copy per module leaves the next module to import
it unguarded by default, which is the wrong way round for a guard.

`sys.modules[name] = None` is the documented way to make an import raise
ImportError, which is precisely the state `semhash_available()` exists to
report. `install_fake_semhash` overrides the entry for the tests that pin the
seam's shape, and the cache-only measurement tests delete it and import the
real library through their own helper.

THE HERMETICITY GUARD
---------------------
The opt-in above is a guard, and deleting it left the suite GREEN on an
air-gapped box while making 38 outbound HTTP attempts on a networked one - the
tests would be reaching for a model over the network and passing either way,
which is the same class of fault as an instrument that reads healthy in the
case it exists to catch.

So network use is made to FAIL rather than to depend on the machine: every
test in the modules below runs with the socket layer refusing to connect. Four
entry points are patched, and all four are load-bearing - `create_connection`
and `getaddrinfo` are what the HTTP stack in `requests`/`urllib3` reaches for,
and `socket.connect`/`connect_ex` are what a raw socket uses, so patching only
some of them leaves a live path out. A test that genuinely wants the network
marks itself `@pytest.mark.live` (nothing does today; the marker exists so that
adding one is a visible decision).

WHERE THE GUARD STOPS, stated rather than implied: it patches THIS
interpreter. A test that spawns a subprocess (`_cli_bytes` does, to compare
bytes under two PYTHONHASHSEEDs) is outside monkeypatch's reach entirely, and
that run is made hermetic a different way - a `semhash.py` on its PYTHONPATH
that raises ImportError, which removes the only thing in these modules that
reaches the network at all.

The socket guard is deliberately not global. Other modules in this suite are
not part of this task's blast radius, and a guard that turns unrelated failures
into socket errors would obscure more than it protects.
"""

import socket
import sys

import pytest

_GUARDED_MODULES = frozenset(
    {"test_build_decontaminate", "test_build_dedupe", "test_build_acquire"}
)

# The socket entry points that are refused. Named rather than inlined so the
# hermeticity test can assert that EACH of them is closed: three of the four
# could be deleted with the suite still green, because only one path was ever
# exercised.
_REFUSED_SOCKET_HOOKS = (
    (socket.socket, "connect"),
    (socket.socket, "connect_ex"),
    (socket, "create_connection"),
    (socket, "getaddrinfo"),
)


class OutboundNetworkAttempt(RuntimeError):
    """A test in a network-free module tried to open a connection."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: this test is allowed to use the network (nothing is, today)"
    )


@pytest.fixture(autouse=True)
def _reset_prompt_overlay():
    """Harmony/recovery configs arm a process-global prompt overlay.

    Without a reset, loading an experiment yaml leaks overlay SHAs into later
    live prompt tests (42 contract failures when harmony runs first).
    """
    from tuned.data.prompt_registry import set_overlay

    set_overlay(None)
    yield
    set_overlay(None)


@pytest.fixture(autouse=True)
def _the_semantic_layer_is_opt_in(monkeypatch):
    """No test runs the REAL semhash unless it says so. See the module
    docstring - this is suite-wide on purpose."""
    monkeypatch.setitem(sys.modules, "semhash", None)


@pytest.fixture(autouse=True)
def _no_outbound_network(request, monkeypatch):
    module = getattr(request.module, "__name__", "")
    if module not in _GUARDED_MODULES or request.node.get_closest_marker("live"):
        return

    def refuse(*args, **kwargs):
        raise OutboundNetworkAttempt(
            f"{module} tried to open a network connection. These tests must be "
            f"hermetic: the semantic seam is opt-in through install_fake_semhash, and "
            f"a test that reaches the Hub passes or fails on which machine it runs on. "
            f"If a test genuinely needs the network, mark it @pytest.mark.live."
        )

    for target, name in _REFUSED_SOCKET_HOOKS:
        monkeypatch.setattr(target, name, refuse)
