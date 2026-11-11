"""The policy's view of a mobile manipulator (every robot): what observation.state and action hold, and how the
base part is made from base poses. One definition for the dataset (the simulator's bag writer), the live
closed loop (closed_loop_system publishes the measured twist and integrates the commanded one), the ROS relay and
the oracle policy. Numpy and the standard library only.

  observation.state = base twist (measured) + the arm joints + the raw gripper joints
  action            = base twist (commanded) + the arm joints (absolute) + the gripper opening per contact link [m]
  base twist        = vx, vy [m/s], wz [rad/s] in the robot (base_link) frame. No world-frame base pose in either.

From base poses (x, y [m], yaw [rad] in the world, as the state log's base joints), per 1 ms step k:
  measured_twist  the achieved pose's backward difference over the last 1/fps (lag_steps of dt), so a state
                  sample at a frame tick covers the frame period before it;
  command_twist   the commanded pose's forward difference over the next 1/fps (linear between steps), so the
                  action of a frame, held for one frame period and integrated, moves the command to where the
                  next frame's command starts.
The world displacement is turned into the robot frame at the mid yaw of the interval. integrate() is the inverse
the closed loop runs every step (mid heading), exact up to O((wz dt)^2).

The Rosetta contract carries the layout: observation.state = [an Odometry source (its twist), a JointState source
(positions)], action = [a Twist source, a JointState source (positions)]; contract_layout() reads it back as
policy names (BASE_TWIST + joint names), which specs.py checks against the robot.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BASE_TWIST = ("base_vx", "base_vy", "base_wz")          # policy names of the base twist entries
JS, ODOM, TWIST, CI = ("sensor_msgs/msg/JointState", "nav_msgs/msg/Odometry", "geometry_msgs/msg/Twist",
                       "sensor_msgs/msg/CompressedImage")
# select field -> policy name, per message type (Odometry's twist is in its child frame: the robot's)
_TWIST_FIELDS = {TWIST: ("linear.x", "linear.y", "angular.z"),
                 ODOM: ("twist.twist.linear.x", "twist.twist.linear.y", "twist.twist.angular.z")}


def lag_steps(fps: float, dt: float) -> int:
    """Steps of dt in one frame period (33 at 30 fps and 1 ms)."""
    return max(1, round(1.0 / (fps * dt)))


def body_twist(b0, b1, span: float) -> np.ndarray:
    """(..., 3) twist [vx, vy, wz] in the robot frame that takes base pose b0 to b1 (..., 3: x, y, yaw) in span s."""
    b0, b1 = np.asarray(b0, float), np.asarray(b1, float)
    d = b1 - b0
    mid = b0[..., 2] + d[..., 2] / 2
    c, s = np.cos(mid), np.sin(mid)
    return np.stack([(c * d[..., 0] + s * d[..., 1]) / span, (-s * d[..., 0] + c * d[..., 1]) / span,
                     d[..., 2] / span], -1)


def measured_twist(base, fps: float, dt: float) -> np.ndarray:
    """(n, 3) per step k: base[max(k - L, 0)] -> base[k] over L dt (L = lag_steps; 0 at a still start)."""
    base = np.asarray(base, float)
    L = lag_steps(fps, dt)
    k = np.arange(len(base))
    return body_twist(base[np.maximum(k - L, 0)], base, L * dt)


def measured_twist_at(base, k: int, fps: float, dt: float) -> np.ndarray:
    """measured_twist(base, fps, dt)[k] from rows 0..k only (the live closed loop's)."""
    L = lag_steps(fps, dt)
    return body_twist(base[max(k - L, 0)], base[k], L * dt)


def command_twist(base_cmd, fps: float, dt: float) -> np.ndarray:
    """(n, 3) per step k: base_cmd[k] -> the command at t_k + 1/fps (linear between steps, held after the last)."""
    base_cmd = np.asarray(base_cmd, float)
    t = np.arange(len(base_cmd)) * dt
    ahead = np.stack([np.interp(t + 1.0 / fps, t, base_cmd[:, i]) for i in range(3)], -1)
    return body_twist(base_cmd, ahead, 1.0 / fps)


def world_velocity(yaw: float, twist) -> np.ndarray:
    """(3,) world-frame base velocity [xd, yd, yawd] of a robot-frame twist at heading yaw."""
    vx, vy, wz = (float(v) for v in twist)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * vx - s * vy, s * vx + c * vy, wz])


def integrate(base, twist, dt: float) -> np.ndarray:
    """Base pose (3,) after dt at the robot-frame twist, turned at the mid heading."""
    base = np.asarray(base, float)
    return base + dt * world_velocity(base[2] + float(twist[2]) * dt / 2, twist)


def policy_state(base_twist, arm, grip_joints) -> np.ndarray:
    """(..., 3 + dof + k) observation.state rows."""
    return np.concatenate([np.asarray(base_twist, float), np.asarray(arm, float), np.asarray(grip_joints, float)], -1)


def policy_action(base_twist, arm, grip_opening) -> np.ndarray:
    """(..., 3 + dof + 1) action rows."""
    g = np.asarray(grip_opening, float)[..., None]
    return np.concatenate([np.asarray(base_twist, float), np.asarray(arm, float), g], -1)


def episode_actions(log, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """(rows, actions) of a state log (state_log.npz arrays) on the 1 / fps grid from its start, in the policy
    format: row k = the step at or before k x int(1e9 / fps) ns (Rosetta's porter tick with `hold`), the
    commanded twist of that step, the arm targets and the gripper opening target."""
    dt = float(log["control_dt"])
    dt_ns, tick_ns = round(dt * 1e9), int(1e9 / fps)
    n = (len(log["time_s"]) - 1) * dt_ns // tick_ns + 1
    rows = np.arange(n) * tick_ns // dt_ns
    a = policy_action(command_twist(log["action_base"], fps, dt), log["action_arm"], log["action_grip"])
    return rows, a[rows]


# ------------------------------------------------------------------ the contract
@dataclass(frozen=True)
class Source:
    topic: str
    type: str
    fields: tuple[str, ...]          # the contract's select
    names: tuple[str, ...]           # policy names (BASE_TWIST entries or joint names)


def _source(src: dict) -> Source:
    ch, sel = src["channel"], tuple(src.get("select") or ())
    ty = ch["type"]
    if ty == JS:
        bad = [f for f in sel if not f.startswith("position.")]
        if bad:
            raise ValueError(f"{ch['topic']}: JointState select {bad}: positions only")
        names = tuple(f.removeprefix("position.") for f in sel)
    elif ty in _TWIST_FIELDS:
        if sel != _TWIST_FIELDS[ty]:
            raise ValueError(f"{ch['topic']}: {ty} select {list(sel)}, want {list(_TWIST_FIELDS[ty])}")
        names = BASE_TWIST
    elif ty == CI:
        names = ()
    else:
        raise ValueError(f"{ch['topic']}: message type {ty} is not one of {[JS, ODOM, TWIST, CI]}")
    return Source(ch["topic"], ty, sel, names)


def sources(entry) -> list[Source]:
    """A contract key's sources (one mapping or a list of them), in contract order."""
    return [_source(s) for s in (entry if isinstance(entry, list) else [entry])]


def contract_layout(contract: dict) -> dict[str, list[Source]]:
    """{key: [Source]} of every observation and action key; the numeric keys' names concatenate in order."""
    return {k: sources(v) for sec in ("observations", "actions") for k, v in (contract.get(sec) or {}).items()}


def names(srcs: list[Source]) -> list[str]:
    return [n for s in srcs for n in s.names]


def action_sources(contract: dict) -> list[Source]:
    (entry,) = contract["actions"].values()
    return sources(entry)


def state_sources(contract: dict) -> list[Source]:
    return sources(contract["observations"]["observation.state"])
