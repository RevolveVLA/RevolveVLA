"""The Rosetta converter: the run's manifest (<spec out>/manifest.json, DATAGEN-v2.md 8.4) -> one LeRobot v3.0 dataset of its
passed episodes by components/dataset as it is (launch.convert, the manager's dataset call): no conversion logic here.

  converter: {plugin: rosetta, format: lerobot, turn: 0}      # format: the only one; turn: the dataset's `turn` label
The dataset component takes fps, cameras and features from the robot's Rosetta contract, so there are no such params.
The pipeline does not call it yet (no contract for the new bags: DATAGEN-v2.md 7).
Result.paths: dataset (the output dir), job (the container job dir: stdout.log, result.json)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Sequence

from datagen.core.interfaces import Converter, Result
from datagen.core.registry import register

LAUNCH = Path(__file__).resolve().parents[5] / "dataset" / "launch.py"
FORMATS = ("lerobot",)


def load_launch():
    """components/dataset/launch.py by path, as the manager loads it (its name `launch` is the simulator one's too)."""
    name = "dataset_launch"
    if name not in sys.modules:
        s = importlib.util.spec_from_file_location(name, LAUNCH)
        sys.modules[name] = m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
    return sys.modules[name]


@register("converter", "rosetta")
class RosettaConverter(Converter):
    def __init__(self, spec, params, launch=None):
        """launch: the dataset launcher module (convert(manifests, out, turn)), injected by tests."""
        if unknown := sorted(set(params) - {"format", "turn"}):
            raise ValueError(f"converter rosetta: unknown params {unknown} (known: format, turn)")
        if params.get("format", "lerobot") not in FORMATS:
            raise ValueError(f"converter rosetta: format {params['format']!r} is not one of {list(FORMATS)}")
        self.spec, self.turn, self.launch = spec, params.get("turn", 0), launch

    def convert(self, episodes: Sequence[Path], out: Path) -> Result:
        """Result.realised: the episodes the dataset holds, from the job's result.json."""
        if not episodes:
            return Result.fail("no recorded episodes to convert")
        run = Path(self.spec.out) if Path(self.spec.out).is_absolute() else Path.cwd() / self.spec.out   # as cli.py
        res = (self.launch or load_launch()).convert([str(run / "manifest.json")], str(Path(out).resolve()), self.turn)
        paths = {"dataset": str(out), "job": str(res["dir"])}
        if res["exit"] != 0:
            return Result(False, f"dataset convert exit {res['exit']} (job {paths['job']})", {}, paths)
        n = ((res.get("result") or {}).get("progress") or {}).get("episodes")
        if not isinstance(n, int) or isinstance(n, bool):
            return Result(False, f"dataset convert gave no episode count (job {paths['job']})", {}, paths)
        return Result(True, "", {"episodes": n}, paths)
