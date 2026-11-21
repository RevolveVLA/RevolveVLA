"""reach_start: the seed's reach sample of a `method: reach` start design: where along the task's grasp axis the
grasp is (grasp_m), the reach distance of the base (d_m) and the base turn about the arm's first joint (yaw_deg).

Hard limits: the design's own ranges (grasp_m, d_m) and +-yaw_half_deg. A dim a spec does not vary keeps the
planner's seeded draw (start_design.reach_sample). The per-attempt jitter stays the planner's."""
from datagen.core.interfaces import Randomiser
from datagen.core.registry import register
from datagen.core.spec import Range

from .design import design_for

REACH_DIMS = ("grasp_m", "d_m", "yaw_deg")      # start_design.REACH_DIMS


def reach_limits(design) -> dict:
    return {"grasp_m": Range(*design.grasp_m), "d_m": Range(*design.d_m),
            "yaw_deg": Range(-design.yaw_half_deg, design.yaw_half_deg)}


def picked(values, dims) -> dict:
    return {d: values[d] for d in dims if d in values}


@register("randomiser", "reach_start")
class ReachStart(Randomiser):
    method, name = "reach", "reach_start"      # the design's method, this randomiser's registered name

    def limits(self, spec):
        return {f"{self.name}.{d}": lim for d, lim in self.dims(design_for(spec, self.method)).items()}

    def dims(self, design) -> dict:
        return reach_limits(design)

    def realise(self, values, context):
        """{"reach": {dim: v}} of the varied dims (the planner's layout keys); {} when none is varied."""
        reach = picked(values, REACH_DIMS)
        return {"reach": reach} if reach else {}
