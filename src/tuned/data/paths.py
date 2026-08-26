"""Single source of truth for the law_v1 data-build workspace layout.

Everything the curation pipeline reads or writes lives under one root
(default data/build, covered by the repo-wide /data/ gitignore rule).
This module owns the directory tree; no other module should hardcode
a path under the build workdir.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# Frozen live control. Isolated experiment siblings (exp_recovery,
# exp_harmony) are allowed next to these names, not under them.
LIVE_WORKDIR = Path("data/build")
LIVE_STATE_DB = LIVE_WORKDIR / "state" / "law_v1.sqlite3"
ISOLATED_WORKDIR_SIBLINGS = frozenset(
    {"exp_recovery", "exp_harmony", "exp_s1", "exp_measure", "exp_deepseek"}
)


def package_repo_root() -> Path:
    """Repo root for this checkout (src/tuned/data/ -> three parents)."""
    return Path(__file__).resolve().parents[3]


def resolve_config_workdir(
    workdir: str | Path, *, repo_root: Path | None = None
) -> Path:
    """Resolve a config-derived workdir against the repository, not cwd.

    Absolute paths (pytest tmp dirs, operator overrides) are left as given
    so unrelated runtime paths stay unchanged. Relative paths join the
    package repo root, then resolve so `..`, `.`, and native separators
    collapse to one location.
    """
    path = Path(workdir)
    if path.is_absolute():
        return path
    return ((repo_root or package_repo_root()) / path).resolve()


def workdir_key(path: Path) -> str:
    """Case-normalized, resolved key for comparing workdirs on any OS."""
    return os.path.normcase(os.path.normpath(str(Path(path).resolve())))


def is_live_control_workdir(
    workdir: str | Path, *, repo_root: Path | None = None
) -> bool:
    """True when `workdir` is the live control root or a control subtree.

    Isolated experiment directories under data/build (exp_recovery,
    exp_harmony) are not live control.
    """
    work = resolve_config_workdir(workdir, repo_root=repo_root)
    live = resolve_config_workdir(LIVE_WORKDIR, repo_root=repo_root)
    work_key = workdir_key(work)
    live_key = workdir_key(live)
    if work_key == live_key:
        return True
    if work_key == workdir_key(live / "state" / "law_v1.sqlite3"):
        return True
    prefix = live_key.rstrip("\\/") + os.sep
    if not work_key.startswith(prefix):
        return False
    first = work_key[len(prefix) :].split(os.sep)[0]
    if not first:
        return True
    return first not in ISOLATED_WORKDIR_SIBLINGS


@dataclass(frozen=True)
class BuildPaths:
    root: Path

    @property
    def state_db(self) -> Path:
        return self.root / "state" / "law_v1.sqlite3"

    @property
    def corpus_dir(self) -> Path:
        return self.root / "corpus"

    @property
    def gold_dir(self) -> Path:
        return self.root / "gold"

    @property
    def streams_dir(self) -> Path:
        return self.root / "streams"

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def raw_gen_dir(self, day: str) -> Path:
        d = self.root / "raw" / "gen" / day
        d.mkdir(parents=True, exist_ok=True)
        return d

    def raw_judge_dir(self, day: str) -> Path:
        d = self.root / "raw" / "judge" / day
        d.mkdir(parents=True, exist_ok=True)
        return d

    def ensure(self) -> "BuildPaths":
        for d in (
            self.state_db.parent,
            self.root / "raw" / "gen",
            self.root / "raw" / "judge",
            self.corpus_dir,
            self.gold_dir,
            self.streams_dir,
            self.out_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self


def build_paths(workdir: str | Path) -> BuildPaths:
    path = Path(workdir)
    if path.is_absolute():
        return BuildPaths(root=path)
    return BuildPaths(root=resolve_config_workdir(path))
