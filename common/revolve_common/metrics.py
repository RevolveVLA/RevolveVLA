"""Metric registry: pure functions over a state dict, and the condition evaluator.

Each metric reads named entries of one state (the producer, e.g. the simulator, builds the state at a
phase end, with arrays already cut to the condition's window). The logic is ported from
the earlier pipeline's simulator phase checks (phase replay, phases, hold check, validation) and task
validators; units are SI except the *_deg / *_cm metrics.

State keys:
  tcp_pose, grasp_pose        4x4 world poses; grasp_segment optional [[x,y,z],[x,y,z]] (policy runs)
  fingers                     per-finger openings [m]; grasp_half_width_m
  contact_left_finger_handle, contact_right_finger_handle   bools over the window
  handle_in_tcp, handle_in_tcp_ref   handle origin in the TCP frame now ((3,) or (n, 3): last row) / at the grasp end
  joints                      {joint name: position [rad or m]}
  targets (n, k), dt, change_tol   commanded base/arm targets over the phase
  link_speeds, contact_depths, other_robot_contacts, commands, gz_log_problems, spawn_clearance_m
  t                           sim time [s] (series only, for hold_s)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Metric:
    fn: Callable
    unit: str
    params: tuple = field(default_factory=tuple)


REGISTRY: dict[str, Metric] = {}


def metric(name, unit, params=()):
    def deco(fn):
        REGISTRY[name] = Metric(fn, unit, tuple(params))
        return fn
    return deco


def rot_deg(Ra, Rb):
    Ra, Rb = np.asarray(Ra, float), np.asarray(Rb, float)
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Ra.T @ Rb) - 1) / 2))))


@metric("tcp_position_error_m", "m")
def tcp_position_error_m(s):
    """TCP vs the plan's grasp pose; a policy run: vs the task's grasp segment (phase_replay_system end_of)."""
    p = np.asarray(s["tcp_pose"], float)[:3, 3]
    seg = s.get("grasp_segment")
    if seg is not None:
        a, b = np.asarray(seg, float)
        u = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
        return float(np.linalg.norm(p - (a + u * (b - a))))
    return float(np.linalg.norm(p - np.asarray(s["grasp_pose"], float)[:3, 3]))


@metric("tcp_rotation_error_deg", "deg")
def tcp_rotation_error_deg(s):
    return rot_deg(np.asarray(s["tcp_pose"])[:3, :3], np.asarray(s["grasp_pose"])[:3, :3])


@metric("grasp_width_error_m", "m")
def grasp_width_error_m(s):
    """Largest per-finger deviation from the handle half-width (grasp_width: all |f - half| <= tol)."""
    f = np.asarray(s["fingers"], float)
    return float(np.max(np.abs(f - s["grasp_half_width_m"])))


@metric("finger_contacts", "fingers")
def finger_contacts(s):
    """Fingers (0-2) that touched the handle in the window (both_fingers_on_handle: == 2)."""
    return int(bool(np.any(s["contact_left_finger_handle"]))) + int(bool(np.any(s["contact_right_finger_handle"])))


@metric("handle_slip_m", "m")
def handle_slip_m(s):
    """Handle in the TCP frame now vs at the grasp end (hold_check: handle[r] - handle[g])."""
    h = np.asarray(s["handle_in_tcp"], float)
    h = h[-1] if h.ndim == 2 else h
    return float(np.linalg.norm(h - np.asarray(s["handle_in_tcp_ref"], float)))


@metric("joint_deg", "deg", params=("joint",))
def joint_deg(s, joint):
    return math.degrees(float(s["joints"][joint]))


@metric("joint_cm", "cm", params=("joint",))
def joint_cm(s, joint):
    return float(s["joints"][joint]) * 100


@metric("targets_static_s", "s")
def targets_static_s(s):
    """Time the base/arm targets stayed static before the phase end (hold_duration)."""
    t = np.asarray(s["targets"], float)
    moved = np.flatnonzero(np.any(np.abs(np.diff(t, axis=0)) > s["change_tol"], axis=1)) + 1
    lm = int(moved[-1]) if len(moved) else None
    return (len(t) - (lm + 1 if lm is not None else 0)) * s["dt"]


@metric("peak_body_speed_mps", "m/s")
def peak_body_speed_mps(s):
    v = np.asarray(s["link_speeds"], float)
    return math.inf if not np.all(np.isfinite(v)) else (float(v.max()) if v.size else 0.0)


@metric("penetration_m", "m")
def penetration_m(s):
    d = np.asarray(s.get("contact_depths", []), float)
    return float(d.max()) if d.size else 0.0


@metric("other_robot_contacts", "contacts")
def other_robot_contacts(s):
    return int(np.sum(s["other_robot_contacts"]))


@metric("non_finite_commands", "commands")
def non_finite_commands(s):
    return int(np.sum(~np.isfinite(np.asarray(s["commands"], float))))


@metric("gz_log_problem_lines", "lines")
def gz_log_problem_lines(s):
    return len(s["gz_log_problems"])


@metric("spawn_clearance_m", "m")
def spawn_clearance_m(s):
    return float(s["spawn_clearance_m"])


# ------------------------------------------------------------------ evaluation
CMP = {"lt": lambda v, x: v < x, "le": lambda v, x: v <= x, "gt": lambda v, x: v > x,
       "ge": lambda v, x: v >= x, "eq": lambda v, x: v == x}
EPS = 1e-9


def value(cond, state):
    m = REGISTRY[cond.metric]
    kw = {p: getattr(cond, p) for p in m.params}
    return m.fn(state, **kw)


def applies(cond, offline=False, policy=False):
    return not ((cond.only == "offline" and not offline) or (cond.only == "plan" and policy))


def evaluate(cond, states, offline=False, policy=False):
    """One condition on a state (dict) or a time series (list of dicts with "t"). With hold_s, every
    sample in the last hold_s must pass; the reported value is the worst one. -> dict or None (n/a)."""
    if not applies(cond, offline, policy):
        return None
    op = cond.op
    lim = cond.offline if offline and cond.offline is not None else cond.threshold
    series = states if isinstance(states, list) else [states]
    out = dict(metric=cond.metric, op=op, threshold=lim, unit=REGISTRY[cond.metric].unit)
    if cond.hold_s is not None:
        t_end = series[-1]["t"]
        if t_end - series[0]["t"] < cond.hold_s - EPS:
            return dict(out, pass_=False, value=None, note="hold window before the series start")
        window = [s for s in series if s["t"] >= t_end - cond.hold_s - EPS]
    else:
        window = series[-1:]
    vals = [value(cond, s) for s in window]
    oks = [CMP[op](v, lim) for v in vals]
    if op in ("ge", "gt"):
        worst = min(vals)
    elif op in ("le", "lt"):
        worst = max(vals)
    else:
        worst = next((v for v, ok in zip(vals, oks) if not ok), vals[-1])
    return dict(out, pass_=all(oks), value=worst)


def evaluate_all(conds, states, offline=False, policy=False):
    """-> (pass, [results]) over the conditions that apply."""
    res = [r for r in (evaluate(c, states, offline, policy) for c in conds) if r is not None]
    return all(r["pass_"] for r in res), res
