"""The planner's grasp of a run, from the task's `planner.grasp` (revolve_common.specs.Grasp) as the planner's task dict
holds it: {axis, points, range?, depth, roll_deg, depth_range?, roll_range_deg?}.

By default the run plans every defined point (the grasp ids 0..n-1). A layout (ROUND_LAYOUT, start_design.py) may name
one grasp instead: `grasp_id` (a defined point) or `grasp` {along_m[, depth_m, roll_deg]} (a value inside the task's
range). Either becomes `given` a value; `narrowed` makes it the task's only grasp (id 0), which every start method
plans the same way. Standard library only."""
from collections.abc import Mapping

_EPS = 1e-12


def fixed(grasp: Mapping, along: float) -> dict:
    """The grasp at `along` with the task's own depth and roll."""
    return {"along": along, "depth": grasp["depth"], "roll_deg": grasp["roll_deg"]}


def _inside(name, v, bounds):
    if bounds is None:
        raise ValueError(f"layout grasp.{name}: the task's grasp has no {name} range to take {v!r} from")
    if not bounds[0] - _EPS <= v <= bounds[1] + _EPS:
        raise ValueError(f"layout grasp.{name}: {v!r} is outside the task's [{bounds[0]}, {bounds[1]}]")
    return float(v)


def given(layout: Mapping | None, grasp: Mapping, by_id: bool = False) -> dict | None:
    """{along, depth, roll_deg} of the grasp the layout gives by value, None when it gives none. `grasp` (value mode):
    each value checked against the task's range, depth and roll_deg left out keep the task's. `grasp_id` is a value
    only with by_id (the reach / drive designs plan one grasp by value; a box design plans the id)."""
    layout = layout or {}
    if (value := layout.get("grasp")) is not None:
        out = fixed(grasp, _inside("along_m", value["along_m"], grasp.get("range")))
        for key, name, bounds in (("depth", "depth_m", "depth_range"), ("roll_deg", "roll_deg", "roll_range_deg")):
            if name in value:
                out[key] = _inside(name, value[name], grasp.get(bounds))
        return out
    if by_id and (gid := layout.get("grasp_id")) is not None:
        return fixed(grasp, grasp["points"][gid])
    return None


def narrowed(grasp: Mapping, value: Mapping) -> dict:
    """The task's grasp block with `value` as its one grasp (the planner builds its assets from it)."""
    return {**grasp, "points": [value["along"]], "depth": value["depth"], "roll_deg": value["roll_deg"]}
