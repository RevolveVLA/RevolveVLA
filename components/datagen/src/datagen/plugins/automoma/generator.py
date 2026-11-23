"""generator automoma: AutoMoMa's whole-body plan (cuRobo, in the datagen image) streamed to the sim (DATAGEN-v2.md 5) as a `Generator`.

`plan` (host, per scene instance, before its sim starts) runs the planner on the instance (plan.py) and leaves `plan.json` in the
scene folder: the 30 Hz base / arm / hand series, or why there is none (a failed plan rejects the scene; the planner is
deterministic, so a stored failure is not retried). `drive` (client container) streams it: the arm and hand as joint
trajectory points, the base as a twist tracking the planned path with the ground-truth pose. The verdict is the evaluator's.

Params (generator block of the spec): `repo` (the repo as the client container sees it, default the one `revolve_common` is in),
`plan_timeout_s` (900)."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from datagen.core.errors import DatagenError
from datagen.core.interfaces import Generator, Result
from datagen.core.naming import container_prefix
from datagen.core.registry import register

from . import launch, problem

EXPERT = "automoma_curobo_v1"
PARAMS = ("repo", "plan_timeout_s")


def read_plan(folder: Path) -> dict | None:
    try:
        return json.loads((Path(folder) / "plan.json").read_text())
    except FileNotFoundError:
        return None


@register("generator", "automoma")
class AutoMoma(Generator):
    def __init__(self, spec, params):
        if unknown := sorted(set(params) - set(PARAMS)):
            raise DatagenError(f"generator automoma: unknown params {unknown} (known: {list(PARAMS)})")
        if spec is not None:
            try:
                problem.check_supported(spec.task, spec.robot)
            except problem.NotSupported as e:
                raise DatagenError(str(e)) from None
        self.spec, self.params = spec, dict(params)
        self.io = None

    def plan(self, instance, task, robot) -> Result:
        folder = Path(instance).parent
        done = read_plan(folder)
        if done is None:
            seed = folder.name.removeprefix("scene_")
            log = Path(self.spec.out) / "logs" / f"scene_{seed}.plan.log"
            try:
                launch.run(container_prefix(self.spec.out), instance, task, robot, folder, log,
                           float(self.params.get("plan_timeout_s", 900)))
            except launch.PlanError as e:
                return Result.fail(str(e))
            done = read_plan(folder)
        return Result() if done and done["ok"] else Result.fail(done["reason"] if done else "the plan run left no plan.json")

    def prepare(self, client, instance, task, robot) -> None:
        from datagen.plugins.moveit import setup
        from datagen.plugins.moveit.robot_io import RobotIO
        from revolve_common import specs
        repo = Path(self.params["repo"]) if "repo" in self.params else specs.REPO_SPECS.parent
        kw = setup.load(instance, task, robot, repo)
        self.io = RobotIO(client, robot, kw["world"], kw["case"], (kw["handle"].model, kw["articulation"][0]))

    def drive(self, client, instance, task, robot):
        from .execute import Streamer
        if self.io is None:
            self.prepare(client, instance, task, robot)
        plan = read_plan(Path(instance).parent)
        if not plan or not plan["ok"]:
            return {"expert": EXPERT, "notes": "error: no plan"}
        self.io.begin_episode()
        try:
            Streamer(self.io, plan).run()
            notes = json.dumps(plan["report"]["events"])
        except Exception:                           # a failed step ends the try (the evaluator times it out), not the scene
            traceback.print_exc(file=sys.stderr)
            notes = f"error: {traceback.format_exc(limit=-1).strip().splitlines()[-1]}"
        return {"expert": EXPERT, "notes": notes}

