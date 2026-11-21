"""The Rosetta converter: recorded episodes (bag/ + episode.json) -> one LeRobot v3.0 dataset by components/dataset as it is
(launch.convert, the call the manager's dataset step makes through turn_dataset): no conversion logic here.

  converter: {plugin: rosetta, format: lerobot, turn: 0}      # format: the only one; turn: the dataset's `turn` label
The dataset component takes fps, cameras and features from the robot's Rosetta contract, so there are no such params.
Result.paths: dataset (the output dir), job (the container job dir: stdout.log, result.json)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Sequence

from datagen.core.interfaces import Converter, Result
from datagen.core.registry import register

from ... import hostmounts
from .. import generate_module

LAUNCH = generate_module.COMPONENT.parent / "dataset" / "launch.py"
FORMATS = ("lerobot",)


def load_launch():
    """components/dataset/launch.py by path, as the manager loads it (its name `launch` is the datagen one's too)."""
    name = "dataset_launch"
    if name not in sys.modules:
        s = importlib.util.spec_from_file_location(name, LAUNCH)
        sys.modules[name] = m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
    hostmounts.apply()                           # an `out` outside the repo: the launchers must know it before a job starts
    return sys.modules[name]


@register("converter", "rosetta")
class RosettaConverter(Converter):
    def __init__(self, spec, params, launch=None):
        """launch: the dataset launcher module (convert(task, episodes, out, turn)), injected by tests."""
        if unknown := sorted(set(params) - {"format", "turn"}):
            raise ValueError(f"converter rosetta: unknown params {unknown} (known: format, turn)")
        if params.get("format", "lerobot") not in FORMATS:
            raise ValueError(f"converter rosetta: format {params['format']!r} is not one of {list(FORMATS)}")
        self.spec, self.turn, self.launch = spec, params.get("turn", 0), launch

    def convert(self, episodes: Sequence[Path], out: Path) -> Result:
        if not episodes:
            return Result.fail("no recorded episodes to convert")
        generate_module.use_robot(self.spec.robot)
        res = (self.launch or load_launch()).convert(self.spec.task, [str(Path(e).resolve()) for e in episodes],
                                                     str(Path(out).resolve()), self.turn)
        paths = {"dataset": str(out), "job": str(res["dir"])}
        if res["exit"] != 0:
            return Result(False, f"dataset convert exit {res['exit']} (job {paths['job']})", {}, paths)
        return Result(True, "", {"episodes": len(episodes)}, paths)
