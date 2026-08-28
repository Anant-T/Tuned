"""Resolve the current commit hash of the private dataset repo and pin it in the training config.

Usage: python scripts/pin_dataset.py [config_path]
Requires: pip install huggingface_hub (present in the [train] extra; also fine standalone).
"""

import re
import sys
from pathlib import Path

import yaml


def resolve_dataset_revision(repo: str, api=None) -> str:
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    return api.dataset_info(repo).sha


def rewrite_dataset_revision(text: str, sha: str) -> str:
    new = re.sub(r"^(\s*dataset_revision:)\s*\S.*$", rf"\1 {sha}", text, count=1, flags=re.M)
    if new != text:
        return new
    # No existing dataset_revision line - insert one right under `hub:`.
    new = re.sub(r"^(hub:\s*\n)", rf"\1  dataset_revision: {sha}\n", text, count=1, flags=re.M)
    if new == text:
        raise SystemExit("no hub: section found")
    return new


def write_pin(config_path: Path, sha: str) -> None:
    text = config_path.read_text(encoding="utf-8")
    new = rewrite_dataset_revision(text, sha)
    config_path.write_text(new, encoding="utf-8")


if __name__ == "__main__":
    cfg_path = Path(sys.argv[1] if len(sys.argv) > 1 else "training/configs/law_v1_8b_ddp.yaml")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    repo = (raw.get("hub") or {}).get("dataset_repo")
    if not repo:
        raise SystemExit("hub.dataset_repo is not set - nothing to pin")
    sha = resolve_dataset_revision(repo)
    write_pin(cfg_path, sha)
    print(f"pinned {repo} @ {sha}")
