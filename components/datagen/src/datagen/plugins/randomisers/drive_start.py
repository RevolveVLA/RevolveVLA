"""drive_start: a `method: drive` start design's per-seed draws: the reach sample (grasp_m, d_m, yaw_deg as reach_start)
and the far start the robot drives in from: back_m behind the reach base, side_m sideways, heading_deg from facing
the object.

Hard limits: the design's ranges (back_m, side_m) and +-heading_half_deg. A dim a spec does not vary keeps the planner's
seeded draw (start_design.drive_sample)."""
from datagen.core.registry import register
from datagen.core.spec import Range

from .reach_start import REACH_DIMS, ReachStart, picked, reach_limits

DRIVE_DIMS = ("back_m", "side_m", "heading_deg")      # start_design.DRIVE_DIMS


@register("randomiser", "drive_start")
class DriveStart(ReachStart):
    method, name = "drive", "drive_start"

    def dims(self, design) -> dict:
        return {**reach_limits(design), "back_m": Range(*design.back_m), "side_m": Range(*design.side_m),
                "heading_deg": Range(-design.heading_half_deg, design.heading_half_deg)}

    def realise(self, values, context):
        """{"reach": {...}, "drive": {...}} of the varied dims; a group none of whose dims is varied is left out."""
        drive = picked(values, DRIVE_DIMS)
        return {**super().realise(values, context), **({"drive": drive} if drive else {})}
