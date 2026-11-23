"""How far apart two splits are: the closest pair of configs (sampled values) or of realised starts, and the per-seed
rejection that enforces a split's `min_distance {realised: true}` while it is generated.

A realised start is `Result.realised` of the generator: the dims it can report (e.g. `robot_base.x_m`, `grasp.id`) in the
same units as the sampled dims, so both are compared with `samplers.distance` (continuous dims normalised by their spec
range, choices 0 / 1) restricted to the dims the start reports."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .interfaces import Result
from .samplers import distance
from .spec import Choices, Dim, Range, SplitSpec

REASON = "min_distance (realised)"
TOLERANCE = 1e-9                                 # a distance exactly at `value` passes despite float error
Point = tuple[int, Mapping[str, Any]]            # (seed, values)


def varied(dims: Mapping[str, Dim], values: Mapping[str, Any]) -> dict[str, Dim]:
    """The dims of `dims` that can differ (a range wider than a point, more than one choice) and `values` has."""
    return {k: d for k, d in dims.items() if k in values and (d.hi > d.lo if isinstance(d, Range) else len(d.values) > 1)}


def closest(a: Sequence[Point], b: Sequence[Point], dims: Mapping[str, Dim], own: bool = False):
    """(distance, seed in a, seed in b) of the closest pair, None without a comparable pair. `own`: a and b are the
    same list (a point is not its own neighbour)."""
    best = None
    for i, (sa, va) in enumerate(a):
        cmp = varied(dims, va)
        for j, (sb, vb) in enumerate(b):
            if (own and i == j) or not cmp or not cmp.keys() <= vb.keys():
                continue
            cand = (distance(va, vb, cmp), sa, sb)
            best = min(best, cand) if best else cand
    return best


def realised_points(entries: Sequence[Mapping]) -> list[Point]:
    """[(seed, realised)] of manifest `used` entries."""
    return [(e["config"]["seed"], e["generate"].get("realised", {})) for e in entries]


def too_close(split: SplitSpec, dims: Mapping[str, Dim], res: Result, data: Mapping) -> str | None:
    """REASON when the split asks for a realised min_distance and `res.realised` is closer than `value` to a realised
    start of the `from` split's used configs; None otherwise (also when there is nothing to compare)."""
    md = split.min_distance
    if md is None or not md.realised or not res.ok:
        return None
    theirs = realised_points(data["splits"].get(md.source, {}).get("used", []))
    pair = closest([(-1, res.realised)], theirs, dims)
    return REASON if pair and pair[0] < md.value - TOLERANCE else None
