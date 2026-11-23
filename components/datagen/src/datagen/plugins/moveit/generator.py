"""generator moveit: the scripted MoveIt expert (DATAGEN-v2.md 5, 12) as a `Generator`: it runs in the client container, a live
ROS 2 client of one sim, and does the task of the scene instance through the controllers' interfaces only. `prepare` starts
the planner once per sim launch (off the episode's clock); `drive` runs one episode.

Params (generator block of the spec): `repo` (the repo as the container sees it, default the one `revolve_common` is in)."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

from datagen.core.errors import DatagenError
from datagen.core.interfaces import Generator
from datagen.core.registry import register

from .experts import EXPERTS

EXPERT = "moveit_ompl_pilz_v1"


@register("generator", "moveit")
class MoveItGenerator(Generator):
    def __init__(self, spec, params):
        if spec is not None and spec.task not in EXPERTS:
            raise DatagenError(f"the moveit generator has an expert for {sorted(EXPERTS)}, not for task {spec.task!r}")
        self.repo = Path(params["repo"]) if "repo" in params else None
        self._expert = None

    def _repo(self) -> Path:
        from revolve_common import specs
        return self.repo or specs.REPO_SPECS.parent

    def prepare(self, client, instance, task, robot) -> None:
        from .setup import build
        self._expert = EXPERTS[task](build(client, instance, task, robot, self._repo()))

    def drive(self, client, instance, task, robot):
        if self._expert is None:
            self.prepare(client, instance, task, robot)
        try:
            notes = self._expert.run()
        except Exception:                       # a failed step ends the try (the evaluator times it out), not the scene
            traceback.print_exc(file=sys.stderr)
            notes = f"error: {traceback.format_exc(limit=-1).strip().splitlines()[-1]}"
        return {"expert": EXPERT, "notes": notes}
