"""grasp: which grasp every attempt of the seed plans, in one of two modes chosen by the dims a spec varies. The task's
`planner.grasp` (task.yaml) holds what is allowed:

  points  `id` (a subset of the task's defined grasp points 0..n-1):   grasp: {id: {choices: [0, 2, 4]}}
  range   `along_m` inside the task's `range` along the grasp axis, plus `depth_m` / `roll_deg` where the task has a
          depth_range / roll_range_deg:                                grasp: {along_m: [-0.1, 0.1]}

Both modes at once is an error, as is a grasp spec next to reach_start / drive_start `grasp_m` (two sources for the one
grasp). Every start design takes it: a box design plans the id among its grasps or the value; a reach / drive design
plans the one grasp (a point's value or the range value) instead of drawing its own grasp_m."""
from datagen.core.interfaces import Randomiser
from datagen.core.registry import register
from datagen.core.spec import Choices, Range, SpecError

from .design import any_design

RANGE_DIMS = ("along_m", "depth_m", "roll_deg")         # start_design.GRASP_DIMS
START_RANDOMISERS = ("reach_start", "drive_start")


def check_modes(spec):
    """SpecError for a spec that mixes the two modes, varies depth / roll without along_m, or names the grasp twice."""
    dims = spec.randomisers.get("grasp", {})
    if not dims:
        return
    if "id" in dims and any(d in dims for d in RANGE_DIMS):
        raise SpecError(f"randomisers.grasp: id (points mode) and {[d for d in RANGE_DIMS if d in dims]} (range mode) "
                        "are two modes: vary one of them")
    if "along_m" not in dims and "id" not in dims:
        raise SpecError(f"randomisers.grasp: {sorted(dims)} needs along_m (range mode) or id (points mode)")
    for r in START_RANDOMISERS:
        if "grasp_m" in spec.randomisers.get(r, {}):
            raise SpecError(f"randomisers: grasp and {r}.grasp_m both give the grasp: vary one of them")


@register("randomiser", "grasp")
class Grasp(Randomiser):
    def limits(self, spec):
        """id: the task's points; along_m (range mode) from its range, depth_m / roll_deg where it has their ranges."""
        from datagen.specs_bridge import grasp
        any_design(spec, "grasp")
        check_modes(spec)
        g = grasp(spec.task, spec.robot)
        out = {"grasp.id": Choices(tuple(range(len(g.points))))}
        for dim, bounds in zip(RANGE_DIMS, (g.range, g.depth_range, g.roll_range_deg)):
            if bounds is not None:
                out[f"grasp.{dim}"] = Range(*bounds)
        return out

    def realise(self, values, context):
        """{"grasp_id": id} (points) or {"grasp": {along_m[, depth_m, roll_deg]}} (range); {} when none is varied (the
        planner then cycles the task's grasps over the attempts)."""
        if "id" in values:
            return {"grasp_id": values["id"]}
        value = {d: values[d] for d in RANGE_DIMS if d in values}
        return {"grasp": value} if value else {}
