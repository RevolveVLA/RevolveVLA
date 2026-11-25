"""meta/revolve_dataset.json: the record of one build of this component. Pure Python.

  rec = read(root)        # Record, or None when root holds no readable record
  rec.ok                  # the one success predicate: a convert or merge build with no problems
  rec.write(root)

A dataset directory is this component's when it holds a readable record; it is a finished dataset (a resume target,
a merge input) when the record is ok."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .layout import SUMMARY

KINDS = ("convert", "merge")


@dataclass
class Record:
    kind: str
    inputs: dict                # the resume key: a finished dataset with equal inputs is kept
    contract: str | None = None     # the contract file it was built with (repo-relative when in the repo)
    prompt: str | None = None
    episodes: int = 0
    frames: int = 0
    features: dict = field(default_factory=dict)        # {key: shape}, LeRobot's bookkeeping columns left out
    per_episode: list = field(default_factory=list)
    checks: dict = field(default_factory=dict)          # {check: "ok" | "failed"}, the checks of its kind
    warnings: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    frame_tolerance: int | None = None                  # convert: checks.FRAME_TOLERANCE when it was built
    action_late_fails: bool | None = None               # convert: checks.ACTION_LATE_FAILS when it was built

    @property
    def ok(self) -> bool:
        return self.kind in KINDS and self.problems == []

    def write(self, root: Path) -> None:
        doc = {k: v for k, v in asdict(self).items()
               if v is not None or k not in ("frame_tolerance", "action_late_fails")}
        (Path(root) / SUMMARY).write_text(json.dumps(doc, indent=1))


def read(root) -> Record | None:
    try:
        doc = json.loads((Path(root) / SUMMARY).read_text())
    except (OSError, ValueError):
        return None
    names = {f.name for f in fields(Record)}
    if not isinstance(doc, dict) or not isinstance(doc.get("kind"), str) or not isinstance(doc.get("inputs"), dict):
        return None
    rec = Record(**{k: v for k, v in doc.items() if k in names})
    return rec if isinstance(rec.problems, list) else None
