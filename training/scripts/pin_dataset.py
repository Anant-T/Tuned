"""Pin the private dataset repo's revision AND the corpus digest in the training config.

Usage: python training/scripts/pin_dataset.py [config_path]
Requires: pip install huggingface_hub (present in the [train] extra; also fine standalone).

Two pins, one pass, because the training lane needs both and they must agree:

  hub.dataset_revision - which commit of the dataset repo to fetch.
  hub.dataset_sha256   - which bytes that fetch must produce.

The digest is not belt-and-braces. A main run is one epoch spread over ~3
Kaggle sessions, and correct resumption depends on `skip_first_batches`
replaying the same LengthGroupedSampler permutation - which is a function of
the corpus FILE. A rebuilt corpus between two sessions retrains some rows and
never trains others, with loss and grad_norm both green. push.py already
records the digest it uploaded in build_manifest.json, so this reads it from
the repo rather than recomputing it from a local copy that may not be what
shipped.
"""

import json
import re
import sys
from pathlib import Path

import yaml

MANIFEST_FILENAME = "build_manifest.json"
TRAIN_FILENAME = "law_v1_train.jsonl"


def resolve_dataset_revision(repo: str, api=None) -> str:
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    return api.dataset_info(repo).sha


def resolve_dataset_sha256(
    repo: str, revision: str | None = None, download=None
) -> str | None:
    """Read the shipped corpus digest out of the repo's own build manifest.

    push.py writes one `outputs[]` entry per uploaded file, each carrying the
    sha256 it computed at upload time. Returns None when the manifest or the
    entry is absent, so a repo predating the manifest still pins its revision.
    """
    if download is None:  # pragma: no cover - exercised by the fake in tests
        from huggingface_hub import hf_hub_download as download
    try:
        path = download(
            repo_id=repo, filename=MANIFEST_FILENAME,
            revision=revision, repo_type="dataset",
        )
    except Exception:
        return None
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for entry in manifest.get("outputs") or []:
        if entry.get("path") == TRAIN_FILENAME:
            return entry.get("sha256")
    return None


def rewrite_pin(text: str, field: str, value: str) -> str:
    """Set `hub.<field>: value`, inserting the key under `hub:` if absent.

    One function for both pins: the two used to be one hardcoded rewrite for
    dataset_revision, and adding the digest by copy-paste would have been a
    second regex to keep in step with the first.
    """
    new = re.sub(rf"^(\s*{field}:)\s*\S.*$", rf"\1 {value}", text, count=1, flags=re.M)
    if new != text:
        return new
    new = re.sub(r"^(hub:\s*\n)", rf"\1  {field}: {value}\n", text, count=1, flags=re.M)
    if new == text:
        raise SystemExit("no hub: section found")
    return new


def rewrite_dataset_revision(text: str, sha: str) -> str:
    return rewrite_pin(text, "dataset_revision", sha)


def write_pin(config_path: Path, sha: str, digest: str | None = None) -> None:
    text = config_path.read_text(encoding="utf-8")
    text = rewrite_pin(text, "dataset_revision", sha)
    if digest:
        text = rewrite_pin(text, "dataset_sha256", digest)
    config_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    cfg_path = Path(sys.argv[1] if len(sys.argv) > 1 else "training/configs/law_v1_8b_ddp.yaml")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    repo = (raw.get("hub") or {}).get("dataset_repo")
    if not repo:
        raise SystemExit("hub.dataset_repo is not set - nothing to pin")
    sha = resolve_dataset_revision(repo)
    digest = resolve_dataset_sha256(repo, sha)
    write_pin(cfg_path, sha, digest)
    print(f"pinned {repo} @ {sha}")
    if digest:
        print(f"corpus {TRAIN_FILENAME} sha256={digest}")
    else:
        print(
            f"WARNING: no {TRAIN_FILENAME} entry in {MANIFEST_FILENAME} - "
            "hub.dataset_sha256 left as-is; a main run will refuse until it is set"
        )
