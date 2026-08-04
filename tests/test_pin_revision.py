import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from pin_revision import resolve_revision, write_pin


class FakeApi:
    def model_info(self, repo):
        class Info:
            sha = "abc123def456"
        return Info()


def test_resolve_revision_returns_sha():
    assert resolve_revision("any/repo", api=FakeApi()) == "abc123def456"


def test_write_pin_replaces_null(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("model:\n  repo: r\n  revision: null\n", encoding="utf-8")
    write_pin(cfg, "abc123def456")
    assert "revision: abc123def456" in cfg.read_text(encoding="utf-8")
