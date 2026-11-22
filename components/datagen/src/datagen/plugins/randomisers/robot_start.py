"""robot_start: where the robot starts, in the scene instance's `robot_start` (DATAGEN-v2.md 8.1).

Dims (world frame, SI):
  x_m, y_m, yaw_rad   absolute base pose [m, m, rad]
  dx_m, dy_m, dyaw_rad  the same as an offset from the base scene's robot_start (use one form per axis)
  arm                 choices among the robot.yaml `poses` names
A dim a spec does not vary keeps the base scene's value.

Hard limits: a box around the base scene's robot_start, +-REACH_M in x and y (offsets: +-REACH_M), a full turn
around its yaw (offsets: +-pi), the robot's pose names for `arm`. The box is a sane bound, not the room: whether a start
is free of the scene's geometry is the `collision_free` validator's job."""
from __future__ import annotations

import math

from datagen.core.interfaces import Randomiser
from datagen.core.registry import register
from datagen.core.spec import Choices, DatagenSpec, Range, SpecError

from datagen.scene import base

REACH_M = 1.5
ABSOLUTE, OFFSET = ("x_m", "y_m", "yaw_rad"), ("dx_m", "dy_m", "dyaw_rad")
DIGITS = 6                                  # the instance holds (and reports) the start rounded to 1 micro-unit


@register("randomiser", "robot_start")
class RobotStart(Randomiser):
    scene_field = "robot_start"

    def limits(self, spec: DatagenSpec):
        varied = spec.randomisers.get("robot_start", {})
        for a, o in zip(ABSOLUTE, OFFSET):
            if a in varied and o in varied:
                raise SpecError(f"randomisers.robot_start: {a} and {o} both set (absolute or offset, not both)")
        x, y, yaw = base.read_scene(spec)["robot_start"]["base"]
        lim = {"x_m": Range(x - REACH_M, x + REACH_M), "y_m": Range(y - REACH_M, y + REACH_M),
               "yaw_rad": Range(yaw - math.pi, yaw + math.pi), "dx_m": Range(-REACH_M, REACH_M),
               "dy_m": Range(-REACH_M, REACH_M), "dyaw_rad": Range(-math.pi, math.pi),
               "arm": Choices(base.arm_poses(spec))}
        return {f"robot_start.{k}": v for k, v in lim.items()}

    def realise(self, values, context):
        """-> the instance's `robot_start`: {"base": [x, y, yaw], "arm": pose name | joints}. context["base"]: the
        base scene's robot_start."""
        start = context["base"]
        pose = [values[a] if a in values else v + values.get(o, 0.0)
                for v, a, o in zip(start["base"], ABSOLUTE, OFFSET)]
        return {"base": [round(v, DIGITS) for v in pose], "arm": values.get("arm", start["arm"])}

    def measure(self, realised, base):
        """Every dim form of the realised start: the absolute ones, the offsets from `base`, and `arm`."""
        pose = realised["base"]
        return {**dict(zip(ABSOLUTE, pose)), **{o: round(v - b, DIGITS) for o, v, b in zip(OFFSET, pose, base["base"])},
                "arm": realised["arm"]}
