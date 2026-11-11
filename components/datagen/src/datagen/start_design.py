"""Start design of a round with a `starts` block (RCA items 1-5, 8, 12): clustered plan starts.

Pure sampling logic: the standard library and numpy only, so the host imports it (plan.py uses it in
the planner image, the start-state scripts and the tests on the host).

All positions and yaws are in the object (planner) frame of frames.py, where the base faces the
object at yaw FACING_YAW_OBJ. A round's cluster centres come from its first seed and lie in the box
`box_m` and the yaw window `yaw_half_deg`, both shrunk by the largest offset. Attempt a of a plan run
belongs to cluster a mod clusters and plans grasp cluster mod 5; its start is the cluster centre plus
a base offset of 1 cm to `offset_m` and a yaw offset of 1 deg to `offset_deg`, drawn from the plan
run's seed. Centres and offsets come from separate seeded generators, never from numpy's or
random's global state (plan.py seeds numpy globally for AutoMoMa).

A block with `mode: reach` (ReachSpec) draws one (grasp, reach, yaw) sample per plan seed instead: the grasp
point along the task's grasp axis, the reach distance d of Robot.base_for_reach (base facing the object) and a
base turn about the arm's first joint axis (the first joint turns back by the same angle, so the arm keeps its
pose: the only base freedom a 5-DoF arm has at a fixed grasp, besides d). Attempt 0 is the sample; the other
attempts jitter d by up to jitter_m and the turn by up to jitter_deg, never the grasp.
"""
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

FACING_YAW_OBJ = math.pi
# The box centre per task and the nominal arm are the robot's: robots/<robot>/planner.py BOX_CENTRE_OBJ and
# robot.yaml arm.ready (plan.py passes them in).
GRASPS = 5

# D0 core.ROUND_BLOCKS["starts"]: kind, low, high, low excluded (checked again here: the
# environment is a trust boundary).
_TABLE = {"clusters": ("int", 1, None, False), "per_cluster": ("int", 1, None, False),
          "box_m": ("number", 0, 0.40, True), "yaw_half_deg": ("number", 0, 30, True),
          "offset_m": ("number", 0.01, 0.03, False), "offset_deg": ("number", 1, 3, False)}
_EPS = 1e-12


@dataclass(frozen=True)
class StartsSpec:
    clusters: int
    per_cluster: int
    box_m: float
    yaw_half_deg: float
    offset_m: float
    offset_deg: float
    nominal_arm: str


@dataclass(frozen=True)
class Start:
    attempt_id: int
    cluster: int
    grasp_id: int
    x: float
    y: float
    yaw: float          # facing + delta, not wrapped
    dx: float
    dy: float
    dyaw: float


@dataclass(frozen=True)
class ReachSpec:
    grasp_m: tuple[float, float]    # TCP along the task's grasp axis (model frame), low / high
    d_m: tuple[float, float]        # reach distance (Robot.base_for_reach d), low / high
    yaw_half_deg: float             # base turn about the arm's first joint axis, +- this
    jitter_m: float                 # attempts after the first: d +- this
    jitter_deg: float               # and the turn +- this

    @property
    def window_half_deg(self):
        return self.yaw_half_deg + self.jitter_deg


@dataclass(frozen=True)
class ReachSample:
    grasp: float                    # m along the grasp axis
    d: float                        # m
    dyaw: float                     # rad, the base turn (+ = counter-clockwise)


_REACH_TABLE = {"yaw_half_deg": ("number", 0, 30, False), "jitter_m": ("number", 0, 0.03, False),
                "jitter_deg": ("number", 0, 3, False)}


def _check_value(key, v, table=_TABLE):
    kind, low, high, low_excluded = table[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (kind == "int" and not isinstance(v, int)):
        raise ValueError(f"starts.{key}: {v!r} is not {'an int' if kind == 'int' else 'a number'}")
    if not math.isfinite(v):
        raise ValueError(f"starts.{key}: {v!r} is not finite")
    if (v <= low if low_excluded else v < low) or (high is not None and v > high):
        raise ValueError(f"starts.{key}: {v!r} is out of range")


def _check_feasible(spec):
    if spec.box_m < 2 * spec.offset_m - _EPS or spec.yaw_half_deg < spec.offset_deg - _EPS:
        raise ValueError(f"starts: no start fits: box_m {spec.box_m} < 2 offset_m {spec.offset_m} or "
                         f"yaw_half_deg {spec.yaw_half_deg} < offset_deg {spec.offset_deg}")


def _range(key, v, positive=False):
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        raise ValueError(f"starts.{key}: {v!r} is not a [low, high] pair")
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError(f"starts.{key}: {x!r} is not a finite number")
    if v[0] > v[1] or (positive and v[0] <= 0):
        raise ValueError(f"starts.{key}: {v!r} is not low <= high" + (" with low > 0" if positive else ""))
    return float(v[0]), float(v[1])


def _check_keys(block, fields, optional, what):
    if not set(fields) <= set(block) <= set(fields) | set(optional):
        raise ValueError(f"starts ({what}): sub-keys {sorted(block)}, want {fields}"
                         + (f" (optional {sorted(optional)})" if optional else ""))


def parse_reach(block: Mapping) -> ReachSpec:
    """The checked ReachSpec of a `mode: reach` starts block."""
    _check_keys(block, ["mode", *ReachSpec.__dataclass_fields__], {}, "mode reach")
    for key in _REACH_TABLE:
        _check_value(key, block[key], _REACH_TABLE)
    return ReachSpec(_range("grasp_m", block["grasp_m"]), _range("d_m", block["d_m"], positive=True),
                     float(block["yaw_half_deg"]), float(block["jitter_m"]), float(block["jitter_deg"]))


def parse_starts(block: Mapping) -> StartsSpec | ReachSpec:
    """The checked spec of a starts block: a ReachSpec for `mode: reach`, else a StartsSpec (D0 types and ranges,
    then feasibility)."""
    if not isinstance(block, Mapping):
        raise ValueError(f"starts: {type(block).__name__} is not a mapping")
    if "mode" in block:
        if block["mode"] != "reach":
            raise ValueError(f"starts.mode: {block['mode']!r}, want 'reach' (or no mode: clustered starts)")
        return parse_reach(block)
    fields = list(StartsSpec.__dataclass_fields__)
    if set(block) != set(fields):
        raise ValueError(f"starts: sub-keys {sorted(block)}, want {fields}")
    for key in _TABLE:
        _check_value(key, block[key])
    if not isinstance(block["nominal_arm"], str) or block["nominal_arm"] != "ready":
        raise ValueError(f"starts.nominal_arm: {block['nominal_arm']!r}, want 'ready'")
    spec = StartsSpec(**{k: block[k] for k in fields})
    _check_feasible(spec)
    return spec


def _no_constant(name):
    raise ValueError(f"starts: {name} is not JSON")


def starts_from_env(environ: Mapping[str, str]) -> StartsSpec | ReachSpec | None:
    """The spec of ROUND_STARTS (D0 round_key_env), None without it. Reads no other key."""
    try:
        text = environ["ROUND_STARTS"]
    except KeyError:
        return None
    try:
        block = json.loads(text, parse_constant=_no_constant)
    except (TypeError, json.JSONDecodeError) as e:
        raise ValueError(f"ROUND_STARTS is not JSON: {e}") from None
    if not isinstance(block, dict):
        raise ValueError("ROUND_STARTS is not a JSON object")
    return parse_starts(block)


def wrap_rad(a: float) -> float:
    """a in (-pi, pi]."""
    r = math.fmod(a + math.pi, 2 * math.pi)
    r = r + 2 * math.pi if r <= 0 else r
    return r - math.pi


def yaw_dev_deg(yaw: float, facing: float = FACING_YAW_OBJ) -> float:
    """yaw - facing [deg] in (-180, 180]."""
    r = math.fmod(math.degrees(yaw - facing) + 180.0, 360.0)
    r = r + 360.0 if r <= 0 else r
    return r - 180.0


def in_yaw_window(yaw: float, half_deg: float, facing: float = FACING_YAW_OBJ) -> bool:
    return abs(yaw_dev_deg(yaw, facing)) <= half_deg + 1e-9


def in_box(x: float, y: float, centre: tuple[float, float], box_m: float) -> bool:
    return abs(x - centre[0]) <= box_m / 2 + _EPS and abs(y - centre[1]) <= box_m / 2 + _EPS


def yaw_window_rad(spec: StartsSpec | ReachSpec, facing: float = FACING_YAW_OBJ) -> tuple[float, float]:
    """(low, high) base yaw [rad] of the start and the goal IK (a reach block: the turn plus its jitter)."""
    h = math.radians(spec.window_half_deg if isinstance(spec, ReachSpec) else spec.yaw_half_deg)
    return facing - h, facing + h


def box_limits(spec: StartsSpec, centre: tuple[float, float]):
    """((x low, x high), (y low, y high)) [m] of the start base."""
    h = spec.box_m / 2
    return (centre[0] - h, centre[0] + h), (centre[1] - h, centre[1] + h)


def _nonneg_int(v, name):
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError(f"{name} {v!r} is not an int >= 0")


def grasp_of_cluster(cluster: int) -> int:
    _nonneg_int(cluster, "cluster")
    return cluster % GRASPS


def cluster_of_attempt(attempt_id: int, clusters: int) -> int:
    _nonneg_int(attempt_id, "attempt_id")
    if isinstance(clusters, bool) or not isinstance(clusters, int) or clusters < 1:
        raise ValueError(f"clusters {clusters!r} is not an int >= 1")
    return attempt_id % clusters


def _rng(seed, stream):
    return np.random.default_rng([int(seed), stream])      # 0: centres, 1: offsets


def cluster_centres(seed: int, spec: StartsSpec, box_centre: tuple[float, float],
                    facing: float = FACING_YAW_OBJ) -> list[tuple[float, float, float]]:
    """spec.clusters centres (x, y, yaw), uniform in the box and the window shrunk by the offsets."""
    _check_feasible(spec)
    h = max(spec.box_m / 2 - spec.offset_m, 0.0)
    w = math.radians(max(spec.yaw_half_deg - spec.offset_deg, 0.0))
    u = _rng(seed, 0).uniform(-1.0, 1.0, size=(spec.clusters, 3))
    return [(float(box_centre[0] + h * a), float(box_centre[1] + h * b), float(facing + w * c)) for a, b, c in u]


def start_offsets(seed: int, attempts: int, spec: StartsSpec) -> list[tuple[float, float, float]]:
    """One (dx, dy, dyaw) per attempt: a base distance in [0.01, offset_m] m in any direction and a
    yaw of 1 to offset_deg deg with either sign."""
    _nonneg_int(attempts, "attempts")
    u = _rng(seed, 1).random(size=(attempts, 4))
    out = []
    for r, t, m, s in u:
        dist = 0.01 + (spec.offset_m - 0.01) * r
        ang = 2 * math.pi * t
        dyaw = math.radians(1.0 + (spec.offset_deg - 1.0) * m)
        out.append((dist * math.cos(ang), dist * math.sin(ang), dyaw if s < 0.5 else -dyaw))
    return out


def plan_starts(centre_seed: int, plan_seed: int, attempts: int, spec: StartsSpec,
                box_centre: tuple[float, float], facing: float = FACING_YAW_OBJ) -> list[Start]:
    """The starts of one plan run: centres from the round's first seed, offsets from the run's seed."""
    centres = cluster_centres(centre_seed, spec, box_centre, facing)
    out = []
    for a, (dx, dy, dyaw) in enumerate(start_offsets(plan_seed, attempts, spec)):
        c = cluster_of_attempt(a, spec.clusters)
        cx, cy, cyaw = centres[c]
        out.append(Start(a, c, grasp_of_cluster(c), cx + dx, cy + dy, cyaw + dyaw, dx, dy, dyaw))
    return out


REQUEST_FILE = "plan_request.json"     # a mint round's plan dir: stages.Round.generate writes core.PlanRequest


def read_plan_request(path, seed: int, attempts: int, spec: StartsSpec):
    """(clusters, centre_seeds) of a plan dir's plan_request.json, None without the file. ValueError
    when it is not a request of this plan run: seed, one cluster in 0..clusters-1 per attempt, one
    centre seed per cluster."""
    try:
        text = open(path).read()
    except FileNotFoundError:
        return None
    r = json.loads(text, parse_constant=_no_constant)

    def ints(x, n, high):
        return (isinstance(x, list) and len(x) == n
                and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= high for v in x))

    if not (isinstance(r, dict) and r.get("seed") == seed and ints(r.get("clusters"), attempts, spec.clusters - 1)
            and ints(r.get("centre_seeds"), spec.clusters, 2**32 - 1)):
        raise ValueError(f"{path}: not a plan request of seed {seed} with {attempts} clusters in "
                         f"0..{spec.clusters - 1} and {spec.clusters} centre seeds: {text[:200]}")
    return tuple(r["clusters"]), tuple(r["centre_seeds"])


def request_centres(centre_seeds, spec: StartsSpec, box_centre: tuple[float, float],
                    facing: float = FACING_YAW_OBJ) -> list[tuple[float, float, float]]:
    """The centre of cluster c: entry c of cluster_centres(centre_seeds[c], ...)."""
    drawn = {s: cluster_centres(s, spec, box_centre, facing) for s in set(centre_seeds)}
    return [drawn[s][c] for c, s in enumerate(centre_seeds)]


def request_starts(clusters, centre_seeds, plan_seed: int, spec: StartsSpec, box_centre: tuple[float, float],
                   facing: float = FACING_YAW_OBJ) -> list[Start]:
    """The starts of a requested plan run: attempt a in cluster clusters[a], from that cluster's
    request centre, offsets from the run's seed. clusters a mod C and one centre seed give plan_starts."""
    centres = request_centres(centre_seeds, spec, box_centre, facing)
    out = []
    for a, (dx, dy, dyaw) in enumerate(start_offsets(plan_seed, len(clusters), spec)):
        c = clusters[a]
        cx, cy, cyaw = centres[c]
        out.append(Start(a, c, grasp_of_cluster(c), cx + dx, cy + dy, cyaw + dyaw, dx, dy, dyaw))
    return out


def reach_sample(seed: int, spec: ReachSpec) -> ReachSample:
    """The seed's (grasp, d, turn): uniform in the block's ranges, one seeded generator each."""
    _nonneg_int(seed, "seed")
    h = math.radians(spec.yaw_half_deg)
    return ReachSample(float(_rng(seed, 0).uniform(*spec.grasp_m)), float(_rng(seed, 1).uniform(*spec.d_m)),
                       float(_rng(seed, 2).uniform(-h, h)))


def reach_attempts(seed: int, attempts: int, spec: ReachSpec) -> list[tuple[float, float]]:
    """(d, turn [rad]) per attempt: attempt 0 the seed's sample, the others jittered by up to jitter_m / jitter_deg."""
    _nonneg_int(attempts, "attempts")
    s = reach_sample(seed, spec)
    u = _rng(seed, 3).uniform(-1.0, 1.0, size=(max(attempts - 1, 0), 2))
    jit = [(0.0, 0.0)] + [(spec.jitter_m * a, math.radians(spec.jitter_deg) * b) for a, b in u]
    return [(s.d + dd, s.dyaw + dy) for dd, dy in jit[:attempts]]


def turn_about(base, pivot_xy, dyaw: float) -> tuple[float, float, float]:
    """Base (x, y, yaw) turned by dyaw about the point at base-frame pivot_xy (that point stays put)."""
    x, y, yaw = (float(v) for v in base)
    px, py = (float(v) for v in pivot_xy)

    def r(a):
        return px * math.cos(a) - py * math.sin(a), px * math.sin(a) + py * math.cos(a)
    (ax, ay), (bx, by) = r(yaw), r(yaw + dyaw)
    return x + ax - bx, y + ay - by, yaw + dyaw


def reach_starts(seed: int, attempts: int, spec: ReachSpec, base_for_reach, pivot_xy) -> list[Start]:
    """The starts of one plan run of a reach block: base_for_reach(d) is the base (x, y, facing) that puts the
    grasp at reach d (Robot.base_for_reach), turned about pivot_xy (the arm's first joint axis, base frame).
    All on grasp 0 (the plan narrows the task's grasps to the sample's), cluster 0; dx = d, dyaw = the turn."""
    out = []
    for a, (d, dyaw) in enumerate(reach_attempts(seed, attempts, spec)):
        x, y, yaw = turn_about(base_for_reach(d), pivot_xy, dyaw)
        out.append(Start(a, 0, 0, x, y, yaw, d, 0.0, dyaw))
    return out


def ik_seed(start: Start, arm: tuple[float, ...]) -> tuple[float, ...]:
    """Start IK seed: base_x, base_y, base_yaw of the start, then the nominal arm (the robot's READY)."""
    return (start.x, start.y, start.yaw) + tuple(arm)
