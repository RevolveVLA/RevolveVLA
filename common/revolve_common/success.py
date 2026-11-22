"""Eval success of a policy rollout from its state log (state_log.npz + live.json), next to the phase verdict.

The phase gates are not an outcome measure: `handle_slip_m` is the door link's origin (the hinge axis) in the TCP
frame, so a hand turning with the door fails a good opening, and a closed gripper hooking the bar with one finger
passes the contact check. Success here (the task's `success:` block, specs.Success):
  - the door is at or above joint_deg for hold_s, and
  - at least held_fraction of its travel from 0 to joint_deg was made with the handle held: per 1 ms step the gripper
    target closed (<= grip_closed_max_m), both fingers touched the handle within contact_s, each finger open
    half_width_m +- width_tol_m, and the bar centre (door angle applied to the grasp pose) within bar_max_m of the
    TCP across the bar axis.
The hinge point is read from the log (the TCP pose and handle_in_tcp give it in the world frame); the bar centre is
the grasp pose's x, y at bar_z_m, turned about the hinge by the door angle.
"""
from __future__ import annotations

import numpy as np


def rotations(t):
    """tcp_pose rows (x y z qw qx qy qz) -> (n, 3, 3)."""
    w, x, y, z = t[:, 3], t[:, 4], t[:, 5], t[:, 6]
    r = np.empty((len(t), 3, 3))
    r[:, 0] = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], 1)
    r[:, 1] = np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], 1)
    r[:, 2] = np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1)
    return r


def within_last(x, w):
    """x (bool, n): any True in the last w steps (this one included)."""
    c = np.concatenate([[0], np.cumsum(x.astype(np.int64))])
    i = np.arange(len(x))
    return (c[i + 1] - c[np.clip(i - w + 1, 0, None)]) > 0


def run_length(m):
    """m (bool, n): the number of consecutive True steps ending at each step."""
    i = np.arange(len(m))
    return i - np.maximum.accumulate(np.where(m, -1, i))


def score(log, grasp_pose, spec, grip_closed_max_m):
    """-> dict(passed, opened, held_fraction, door_max_deg); opened = the door at or above joint_deg for hold_s (C1),
    passed = opened and held (C2). log: the state_log.npz arrays (door_pos, action_grip, tcp_pose,
    handle_in_tcp, robot_joint_names, robot_joint_pos, contact_left/right_finger_handle, control_dt); grasp_pose:
    live.json grasp_pose_world (4 x 4); spec: specs.Success."""
    q, grip, dt = log["door_pos"], log["action_grip"], float(log["control_dt"])
    tcp = log["tcp_pose"]
    rot, pos = rotations(tcp), tcp[:, :3]
    hinge = np.median(pos + np.einsum("nij,nj->ni", rot, log["handle_in_tcp"]), axis=0)
    rel = np.asarray(grasp_pose, float)[:2, 3] - hinge[:2]
    c, s = np.cos(q), np.sin(q)
    bar = np.stack([hinge[0] + c * rel[0] - s * rel[1], hinge[1] + s * rel[0] + c * rel[1], np.full(len(q), spec.bar_z_m)], 1)
    in_tcp = np.einsum("nji,nj->ni", rot, bar - pos)                 # the bar centre in the TCP frame
    axis = rot[:, 2, :]                                              # the bar's axis (world z) in the TCP frame
    across = in_tcp - np.einsum("ni,ni->n", in_tcp, axis)[:, None] * axis

    n_touch = max(1, round(spec.contact_s / dt))
    touch = within_last(log["contact_left_finger_handle"], n_touch) & within_last(log["contact_right_finger_handle"], n_touch)
    fingers = [i for i, n in enumerate(log["robot_joint_names"]) if "finger" in str(n)]
    open_ok = np.all(np.abs(log["robot_joint_pos"][:, fingers] - spec.half_width_m) < spec.width_tol_m, axis=1)
    held = (grip <= grip_closed_max_m) & touch & open_ok & (np.linalg.norm(across, axis=1) < spec.bar_max_m)

    goal = np.radians(spec.joint_deg)
    above = q >= goal
    opened = bool((run_length(above) >= round(spec.hold_s / dt)).any())
    end = int(np.argmax(above)) if above.any() else int(np.argmax(q))
    gain = np.clip(np.diff(q, prepend=q[0]), 0, None)[:end + 1]
    fraction = float(gain[held[:end + 1]].sum() / (gain.sum() or 1e-12))
    return dict(passed=opened and fraction >= spec.held_fraction, opened=opened, held_fraction=round(fraction, 3),
                door_max_deg=round(float(np.degrees(q.max())), 2))
