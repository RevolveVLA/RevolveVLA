"""Pure base motion of the MoveIt expert: the ramped, constant-twist segments the omni base is driven with (DATAGEN-v2.md
5.2: ramp the twist over ~0.5 s, the controller has no limiter). Nothing here knows ROS; `RobotIO` publishes them."""
from __future__ import annotations

import math

import numpy as np

RAMP_S = 0.5            # the twist ramps up and down over this long [s]
RATE_HZ = 50.0          # cmd_vel publishing rate; the base stops 0.5 s after the last message


def ramp(t: float, total: float, ramp_s: float = RAMP_S) -> float:
    """The trapezoid's factor (0..1) at time t of a segment that lasts `total`."""
    return max(0.0, min(1.0, t / ramp_s, (total - t) / ramp_s))


def profile(amount: float, rate: float, ramp_s: float = RAMP_S) -> tuple[float, float]:
    """-> (peak rate, duration [s]) of a segment covering `amount` (m or rad) at most at `rate`: a trapezoid, or a triangle
    with a lower peak when `amount` is too small to reach `rate` (it would overshoot)."""
    if amount <= 0.0:
        return 0.0, 0.0
    if amount < rate * ramp_s:
        return amount / ramp_s, 2 * ramp_s
    return rate, amount / rate + ramp_s


def rotation(yaw: float, yaw_goal: float, rate: float) -> tuple[tuple, float, float]:
    """Turn from `yaw` to `yaw_goal` by the short way -> (unit body twist, amount [rad], rate)."""
    d = math.atan2(math.sin(yaw_goal - yaw), math.cos(yaw_goal - yaw))
    return (0.0, 0.0, math.copysign(1.0, d)), abs(d), rate


def translation(pose, goal_xy, rate: float) -> tuple[tuple, float, float]:
    """Straight to `goal_xy` from the base pose (x, y, yaw), the base keeping its heading -> (unit body twist, distance
    [m], rate). The omni base strafes: the direction is the goal's in the body frame."""
    x, y, yaw = pose
    d = np.asarray(goal_xy, dtype=float) - (x, y)
    c, s = math.cos(yaw), math.sin(yaw)
    body = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    dist = float(np.linalg.norm(body))
    return ((*(body / dist), 0.0) if dist > 0 else (0.0, 0.0, 0.0)), dist, rate


def orbit(hinge_in_base, ccw: bool) -> tuple:
    """The constant body twist (vx, vy, wz per unit rad/s) that turns the whole robot rigidly about a vertical line through
    `hinge_in_base` (x, y in the base frame): the robot pivots with a door it holds."""
    w = 1.0 if ccw else -1.0
    return (w * hinge_in_base[1], -w * hinge_in_base[0], w)
