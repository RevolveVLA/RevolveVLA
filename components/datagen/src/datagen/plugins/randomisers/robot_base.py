"""robot_base: the base start, as offsets from the task's start-box centre (object frame) and from facing the object.

Hard limits: the planner's start box (box_m 0.40, so +-0.20 m) and yaw window (+-30 deg) of generator.yaml, shrunk by
the per-attempt jitter (offset_m 0.03, offset_deg 3), so every attempt stays inside them (mobile_panda's blocks)."""
import ast
import math
from pathlib import Path

from datagen.core.interfaces import Randomiser
from datagen.core.registry import register
from datagen.core.spec import Range

BASE_XY_M = 0.17
BASE_YAW_DEG = 27.0
FACING_YAW_OBJ = math.pi        # the base faces the object at this yaw in the object frame
ROBOTS = Path(__file__).resolve().parents[5] / "simulator" / "src" / "sim" / "robots"


def box_centre_obj(implementation: str, robots: Path = ROBOTS) -> dict:
    """{task: (x, y)}: robots/<implementation>/planner.py BOX_CENTRE_OBJ, read as a literal without importing the
    module (it needs the datagen image; sampling and realising run on the host)."""
    tree = ast.parse((robots / implementation / "planner.py").read_text())
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "BOX_CENTRE_OBJ" for t in n.targets):
            return ast.literal_eval(n.value)
    raise KeyError(f"robots/{implementation}/planner.py has no BOX_CENTRE_OBJ")


@register("randomiser", "robot_base")
class RobotBase(Randomiser):
    def limits(self):
        return {"robot_base.x_m": Range(-BASE_XY_M, BASE_XY_M), "robot_base.y_m": Range(-BASE_XY_M, BASE_XY_M),
                "robot_base.yaw_deg": Range(-BASE_YAW_DEG, BASE_YAW_DEG)}

    def realise(self, values, context):
        """-> {"offset": (dx, dy, dyaw_rad), "start_pose": (x, y, yaw)} in the object frame; a dim that is not varied
        is 0. context: `box_centre` (x, y) of the task (box_centre_obj(robot)[task]), `facing` (default
        FACING_YAW_OBJ)."""
        dx, dy, dyaw = values.get("x_m", 0.0), values.get("y_m", 0.0), math.radians(values.get("yaw_deg", 0.0))
        (cx, cy), facing = context["box_centre"], context.get("facing", FACING_YAW_OBJ)
        return {"offset": (dx, dy, dyaw), "start_pose": (cx + dx, cy + dy, facing + dyaw)}
