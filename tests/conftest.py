"""Test-suite guards that must not depend on which machine runs the suite.

THE HERMETICITY GUARD
---------------------
`test_build_decontaminate.py` and `test_build_dedupe.py` each carry an autouse
fixture that makes `import semhash` fail, so the real embedding seam never runs
inside an ordinary test. Delete both fixtures and the suite stays GREEN on an
air-gapped box while making 38 outbound HTTP attempts on a networked one - the
tests would be reaching for a model over the network and passing either way,
which is the same class of fault as an instrument that reads healthy in the
case it exists to catch.

So network use is made to FAIL rather than to depend on the machine: every
test in those modules runs with the socket layer refusing to connect. A test
that genuinely wants the network marks itself `@pytest.mark.live` (nothing
does today; the marker exists so that adding one is a visible decision).

The guard is deliberately not global. Other modules in this suite are not
part of this task's blast radius, and a guard that turns unrelated failures
into socket errors would obscure more than it protects.
"""

import socket

import pytest

_GUARDED_MODULES = frozenset(
    {"test_build_decontaminate", "test_build_dedupe", "test_build_acquire"}
)


class OutboundNetworkAttempt(RuntimeError):
    """A test in a network-free module tried to open a connection."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: this test is allowed to use the network (nothing is, today)"
    )


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

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
