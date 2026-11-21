"""robot_base: the base start, as offsets from the task's start-box centre (object frame) and from facing the object.

Hard limits: the task's start design for the spec's robot (task.yaml starts.<robot>, method box; another method is a
StartDesignError naming its randomiser): the start box and yaw window shrunk by the per-attempt jitter, so every
attempt stays inside them (specs.BoxStarts.base_limits)."""
import math

from datagen.core.interfaces import Randomiser
from datagen.core.registry import register
from datagen.core.spec import Range

from .design import design_for

FACING_YAW_OBJ = math.pi        # the base faces the object at this yaw in the object frame


def box_design(spec):
    """The `method: box` start design of the spec's robot on its task (specs.BoxStarts)."""
    return design_for(spec, "box")


@register("randomiser", "robot_base")
class RobotBase(Randomiser):
    def limits(self, spec):
        lim = box_design(spec).base_limits()
        return {f"robot_base.{dim}": Range(-v, v) for dim, v in lim.items()}

    def realise(self, values, context):
        """-> {"offset": (dx, dy, dyaw_rad), "start_pose": (x, y, yaw)} in the object frame; a dim that is not varied
        is 0. context: `box_centre` (x, y) of the task (box_design(spec).centre), `facing` (default
        FACING_YAW_OBJ)."""
        dx, dy, dyaw = values.get("x_m", 0.0), values.get("y_m", 0.0), math.radians(values.get("yaw_deg", 0.0))
        (cx, cy), facing = context["box_centre"], context.get("facing", FACING_YAW_OBJ)
        return {"offset": (dx, dy, dyaw), "start_pose": (cx + dx, cy + dy, facing + dyaw)}
