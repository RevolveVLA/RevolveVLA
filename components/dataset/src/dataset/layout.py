"""Where things are in a dataset of this component, and its progress line. Pure Python.

  <D>/                          LeRobotDataset(REPO_ID, root=<D>)
  <D>/meta/rosetta_contract.yaml    CONTRACT: the robot's Rosetta contract the dataset was built with
  <D>/meta/revolve_dataset.json     SUMMARY: the build record (record.py)
  <D>.partial/                  the build directory (PARTIAL_MARK inside), kept after a failure"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ID = "revolve/dataset"     # LeRobot needs a repo id; never pushed (the dataset directory is the root)
CONTRACT = Path("meta") / "rosetta_contract.yaml"
SUMMARY = Path("meta") / "revolve_dataset.json"
PARTIAL_MARK = ".revolve_dataset_build"


def partial(out: Path) -> Path:
    return out.parent / f"{out.name}.partial"


def progress(**d):
    print("PROGRESS " + json.dumps(d), flush=True)
