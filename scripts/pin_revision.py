"""Resolve the current commit hash of the base-model repo and pin it in the config.

Usage: python scripts/pin_revision.py [config_path]
Requires: pip install huggingface_hub (present in the [train] extra; also fine standalone).
"""

import re
import sys
from pathlib import Path

import yaml


def resolve_revision(repo: str, api=None) -> str:
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    return api.model_info(repo).sha


def write_pin(config_path: Path, revision: str) -> None:
    text = config_path.read_text(encoding="utf-8")
    new = re.sub(r"^(\s*revision:)\s*\S.*$", rf"\1 {revision}", text, count=1, flags=re.M)
    if new == text:
        raise SystemExit("no revision line found")
    config_path.write_text(new, encoding="utf-8")


if __name__ == "__main__":
    cfg_path = Path(sys.argv[1] if len(sys.argv) > 1 else "configs/law_v1.yaml")
    repo = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["model"]["repo"]
    sha = resolve_revision(repo)
    write_pin(cfg_path, sha)
    print(f"pinned {repo} @ {sha}")
