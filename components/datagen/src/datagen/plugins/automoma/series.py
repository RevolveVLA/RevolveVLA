"""A plan as a 30 Hz command series (numpy only; runs in the datagen image and in the client container).

The planned motion is a few timed segments of whole-body rows (world base x, y, yaw + the arm's joints): the drive-in, a smooth
move onto the grasp, the fingers closing, the pull. `build` puts them on one clock and samples it at the action rate: the rows
the executor streams (base reference for cmd_vel, arm and hand references for the controllers' single-point trajectories)."""
from __future__ import annotations

import numpy as np

RATE_HZ = 30


def smooth(t, knots, rows):
    """Rows at times t: a cubic Hermite spline through (knots, rows) with finite-difference tangents (Catmull-Rom for uneven
    knots), clamped at the ends. rows (n, d), knots (n,) increasing."""
    knots, rows, t = np.asarray(knots, float), np.asarray(rows, float), np.asarray(t, float)
    if len(knots) == 1:
        return np.tile(rows[0], (len(t), 1))
    m = np.gradient(rows, knots, axis=0)
    m[0] = m[-1] = 0.0                                          # a segment starts and ends at rest
    i = np.clip(np.searchsorted(knots, t, side="right") - 1, 0, len(knots) - 2)
    h = (knots[i + 1] - knots[i])[:, None]
    s = np.clip((t - knots[i])[:, None] / h, 0.0, 1.0)
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    return h00 * rows[i] + h10 * h * m[i] + h01 * rows[i + 1] + h11 * h * m[i + 1]


def pull_duration(rows, vel_limits, slowdown: float, minimum: float) -> float:
    """The pull's time: `slowdown` x the time its rows take at the joints' velocity limits, at least `minimum` [s]."""
    d = np.abs(np.diff(np.asarray(rows, float), axis=0))
    hint = float(np.maximum(np.linalg.norm(d[:, :2], axis=1) / vel_limits[0], (d[:, 2:] / vel_limits[2:]).max(1)).sum())
    return max(minimum, slowdown * hint)


def build(drive, drive_dt: float, pull, hand, timing: dict, pull_s: float, rate: int = RATE_HZ) -> dict:
    """The series. drive (n, 10) rows at drive_dt (the last one is the staging pose), pull (m, 10) rows (the first one is the
    grasp pose; all rows world base + 7 arm joints), hand = (open, squeeze) finger references, timing = planner.yaml `timing`,
    pull_s = the pull's duration -> {rate_hz, base (N, 3), arm (N, 7), hand (N,), events {t_stage, t_close, t_pull, t_end}}."""
    drive, pull = np.asarray(drive, float), np.asarray(pull, float)
    td = timing["drive_slowdown"] * drive_dt * np.arange(len(drive))
    t_stage = td[-1]
    t_close = t_stage + timing["approach_s"] + timing["settle_s"]
    t_pull = t_close + timing["close_s"] + timing["hold_s"]
    tp = t_pull + pull_s * np.linspace(0.0, 1.0, len(pull))
    t_end = tp[-1] + timing["end_hold_s"]
    ts = np.arange(int(round(t_end * rate)) + 1) / rate
    q = np.empty((len(ts), drive.shape[1]))
    rows = [(ts < t_stage, lambda t: smooth(t, td, drive)),
            ((ts >= t_stage) & (ts < t_stage + timing["approach_s"]),
             lambda t: drive[-1] + (pull[0] - drive[-1]) * ease((t - t_stage) / timing["approach_s"])[:, None]),
            ((ts >= t_stage + timing["approach_s"]) & (ts < t_pull), lambda t: np.tile(pull[0], (len(t), 1))),
            ((ts >= t_pull) & (ts < tp[-1]), lambda t: smooth(t, tp, pull)),
            (ts >= tp[-1], lambda t: np.tile(pull[-1], (len(t), 1)))]
    for mask, f in rows:
        if mask.any():
            q[mask] = f(ts[mask])
    hand_ref = np.interp(ts, [0.0, t_close, t_close + timing["close_s"], t_end], [hand[0], hand[0], hand[1], hand[1]])
    return {"rate_hz": rate, "base": q[:, :3], "arm": q[:, 3:], "hand": hand_ref,
            "events": {"t_stage": t_stage, "t_close": t_close, "t_pull": t_pull, "t_end": t_end}}


def ease(s):
    """Smoothstep: 0 -> 1 with zero slope at both ends."""
    return 3 * s**2 - 2 * s**3
