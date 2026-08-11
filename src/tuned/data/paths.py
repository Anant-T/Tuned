"""Single source of truth for the law_v1 data-build workspace layout.

Everything the curation pipeline reads or writes lives under one root
(default data/build, covered by the repo-wide /data/ gitignore rule).
This module owns the directory tree; no other module should hardcode
a path under the build workdir.
"""

from dataclasses import dataclass
from pathlib import Path


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
    return BuildPaths(root=Path(workdir))
