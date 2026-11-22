"""The robot's footprint on the floor and its clearance from the scene's boxes.

Footprint = a rectangle in the base frame (+x forward) extruded over [z_min, z_max]: the chassis box plus the folded
arm's overhang. Chassis 0.50 x 0.50 m centred on base_footprint (specs/robots/common/omni_base.urdf.xacro defaults);
overhang ahead of it: the folded Panda reaches 0.11 m past the chassis at door height, the folded XL's jaws hang 0.51 m
ahead of base_footprint (DATAGEN-v2.md 12). The same rectangle is used for every arm pose of the robot.

Distance = the gap between the footprint and the top-down hull of every box whose z range meets the footprint's (0 when
they overlap), so a counter top above the robot is not an obstacle and a thin floor slab is not either."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .sdf_boxes import Box

Point = tuple[float, float]


@dataclass(frozen=True)
class Footprint:
    x: tuple[float, float] = (-0.25, 0.25)       # rear, front [m] from base_footprint
    y: tuple[float, float] = (-0.25, 0.25)
    z: tuple[float, float] = (0.02, 1.0)         # the heights it blocks [m]

    def polygon(self, x: float, y: float, yaw: float) -> list[Point]:
        """Its corners in the world for a base at (x, y, yaw)."""
        c, s = math.cos(yaw), math.sin(yaw)
        corners = ((self.x[0], self.y[0]), (self.x[1], self.y[0]), (self.x[1], self.y[1]), (self.x[0], self.y[1]))
        return [(x + c * px - s * py, y + s * px + c * py) for px, py in corners]


FOOTPRINTS = {"mobile_panda": Footprint(x=(-0.25, 0.36)), "mobile_so101_xl": Footprint(x=(-0.25, 0.51))}


def footprint_of(robot: str) -> Footprint:
    """The robot's footprint (a robot without an entry: the bare chassis)."""
    return FOOTPRINTS.get(robot, Footprint())


def hull(points: Sequence[Point]) -> list[Point]:
    """Convex hull, counter-clockwise (monotone chain); fewer than 3 points for a degenerate set."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and cross(h[-2], h[-1], p) <= 0:
                h.pop()
            h.append(p)
        return h[:-1]
    return half(pts) + half(pts[::-1])


def _edges(poly):
    return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]


def _to_segment(p: Point, a: Point, b: Point) -> float:
    ax, ay, dx, dy = a[0], a[1], b[0] - a[0], b[1] - a[1]
    t = 0.0 if dx == dy == 0 else min(max(((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy), 0.0), 1.0)
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def _separated(a, b) -> bool:
    """True when some edge normal of a or b separates the two convex polygons (SAT)."""
    for poly in (a, b):
        for p, q in _edges(poly):
            nx, ny = q[1] - p[1], p[0] - q[0]
            if nx == ny == 0:
                continue
            pa, pb = [nx * v[0] + ny * v[1] for v in a], [nx * v[0] + ny * v[1] for v in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return True
    return False


def polygon_distance(a: Sequence[Point], b: Sequence[Point]) -> float:
    """Gap between two convex polygons [m]; 0 when they touch or overlap."""
    if len(a) >= 3 and len(b) >= 3 and not _separated(a, b):
        return 0.0
    return min([_to_segment(p, *e) for p in a for e in _edges(b)] + [_to_segment(p, *e) for p in b for e in _edges(a)])


class Clearance:
    """`reject(x, y, yaw)` -> None, or why the footprint at that base pose is closer than `min_m` to a scene box."""

    def __init__(self, boxes: Sequence[Box], footprint: Footprint, min_m: float):
        self.footprint, self.min_m = footprint, min_m
        z0, z1 = footprint.z
        self.hulls = []
        for b in boxes:
            c = b.corners()
            if c[:, 2].max() > z0 and c[:, 2].min() < z1:
                self.hulls.append((b.name, hull([(float(px), float(py)) for px, py in c[:, :2]])))

    def nearest(self, x: float, y: float, yaw: float) -> tuple[float, str | None]:
        """(gap [m], box name) of the closest box; (inf, None) when none is within min_m (the gap is exact below min_m;
        a farther box may be skipped)."""
        poly = self.footprint.polygon(x, y, yaw)
        centre = np.mean(poly, axis=0)
        r = max(math.hypot(px - centre[0], py - centre[1]) for px, py in poly)
        best = (math.inf, None)
        for name, h in self.hulls:
            hc = np.mean(h, axis=0)
            hr = max(math.hypot(px - hc[0], py - hc[1]) for px, py in h)
            if math.hypot(*(hc - centre)) - r - hr > min(best[0], self.min_m):
                continue                                    # bounding circles apart: cannot beat the best or min_m
            best = min(best, (polygon_distance(poly, h), name))
        return best

    def reject(self, x: float, y: float, yaw: float) -> str | None:
        gap, name = self.nearest(x, y, yaw)
        if gap >= self.min_m:
            return None
        return (f"collision: footprint overlaps {name}" if gap == 0 else
                f"collision: footprint {gap:.3f} m from {name} (min {self.min_m} m)")
