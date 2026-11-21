"""robot_base: the base start, as offsets from the task's start-box centre (object frame) and from facing the object.

Hard limits: the task's start design for the spec's robot (task.yaml starts.<robot>, method box): the start box and yaw
window shrunk by the per-attempt jitter, so every attempt stays inside them (specs.BoxStarts.base_limits)."""
import math

from datagen.core.errors import DatagenError
from datagen.core.interfaces import Randomiser
from datagen.core.registry import register
from datagen.core.spec import Range

FACING_YAW_OBJ = math.pi        # the base faces the object at this yaw in the object frame


class RobotBaseError(DatagenError):
    pass


def box_design(task: str, robot: str):
    """The `method: box` start design of the robot for the task (specs.BoxStarts); an error when it has none."""
    from datagen.specs_bridge import starts            # reads the task spec (revolve_common); not for realise
    design = starts(task, robot)
    if design is None or design.method != "box":
        raise RobotBaseError(f"robot_base: {robot} on {task} has "
                             + ("no start design" if design is None else f"a {design.method} start design")
                             + f" (task.yaml starts.{robot}); it needs `method: box`")
    return design


@register("randomiser", "robot_base")
class RobotBase(Randomiser):
    def limits(self, spec):
        lim = box_design(spec.task, spec.robot).base_limits()
        return {f"robot_base.{dim}": Range(-v, v) for dim, v in lim.items()}

    def realise(self, values, context):
        """-> {"offset": (dx, dy, dyaw_rad), "start_pose": (x, y, yaw)} in the object frame; a dim that is not varied
        is 0. context: `box_centre` (x, y) of the task (box_design(task, robot).centre), `facing` (default
        FACING_YAW_OBJ)."""
        dx, dy, dyaw = values.get("x_m", 0.0), values.get("y_m", 0.0), math.radians(values.get("yaw_deg", 0.0))
        (cx, cy), facing = context["box_centre"], context.get("facing", FACING_YAW_OBJ)
        return {"offset": (dx, dy, dyaw), "start_pose": (cx + dx, cy + dy, facing + dyaw)}
