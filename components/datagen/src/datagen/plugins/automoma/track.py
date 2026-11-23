"""Following a base reference with a velocity-commanded omni base (numpy only): the feedforward twist of the planned path plus
a proportional correction from the measured pose, in the robot frame, with a slew limit (the omni rollers kick under a step in
twist, DATAGEN-v2.md 5.2)."""
from __future__ import annotations

import math

import numpy as np


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def twist(ref, ref_next, pose, dt: float, gain: float, v_max: float, w_max: float) -> np.ndarray:
    """The body twist (vx, vy, wz) that moves the base from `pose` onto the path: ref (x, y, yaw) is where it should be now,
    ref_next dt later (the world velocity is their difference), all in the world; clamped to v_max [m/s] and w_max [rad/s]."""
    ref, ref_next, pose = np.asarray(ref, float), np.asarray(ref_next, float), np.asarray(pose, float)
    v = np.array([*(ref_next[:2] - ref[:2]) / dt, wrap(ref_next[2] - ref[2]) / dt])
    e = np.array([*(ref[:2] - pose[:2]), wrap(ref[2] - pose[2])])
    v = v + gain * e
    c, s = math.cos(pose[2]), math.sin(pose[2])
    body = np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]])
    n = np.linalg.norm(body[:2])
    if n > v_max:
        body[:2] *= v_max / n
    body[2] = max(-w_max, min(w_max, body[2]))
    return body


def slew(prev, new, accel: float, dt: float) -> np.ndarray:
    """`new` limited to a change of accel * dt from `prev` per component."""
    step = accel * dt
    prev, new = np.asarray(prev, float), np.asarray(new, float)
    return prev + np.clip(new - prev, -step, step)
