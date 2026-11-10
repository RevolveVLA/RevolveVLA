#!/usr/bin/env python3
"""The can-on-plate motion plan of the Panda-only demo (fixed-base Panda + Hand on a table; no Gazebo
dependency). The Panda's kinematics come from the mobile Panda robot (robots/mobile_panda/robot.py).

Plans Cartesian straight-line TCP segments with min-jerk timing and solves IK
for every 10 ms sample with damped least squares, seeded by the previous
sample so the joint path is continuous.

  panda_plan.py     print the plan and the worst IK error
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the sim modules
import robots  # noqa: E402
from robots.base import min_jerk, rot, rotvec, tf  # noqa: E402, F401
from robots.mobile_panda.robot import JOINT_TF, LOWER, UPPER  # noqa: E402

PANDA = robots.load("mobile_panda")
WORLD = "can_on_plate"
BASE = np.array([0.0, 0.0, 0.40])           # panda pose in the world
DT = 0.01                                    # setpoint period [s]
READY = PANDA.ready
TCP_TF = PANDA.tcp_tf

# Scene (must match the world file)
# The can lies on its side with its axis along world x; the fingers close
# along world y, across its diameter, level with its centre of mass.
CAN_XY, CAN_H, CAN_R = np.array([0.55, -0.20]), 0.10, 0.03
PLATE_XY, PLATE_TOP, PLATE_R = np.array([0.50, 0.22]), 0.415, 0.11
TABLE_Z = 0.40
GRASP_DZ = 0.0             # TCP height relative to the can's axis
GRIP_OPEN, GRIP_CLOSED = 0.04, 0.0
CARRY_S = float(__import__("os").environ.get("CARRY_S", 3.5))   # carry duration [s]


def fk(q):
    """TCP pose in the world plus joint axes/origins for the Jacobian."""
    T = tf(p=BASE)
    axes, origins = [], []
    for Tj, qi in zip(JOINT_TF, q):
        T = T @ Tj
        axes.append(T[:3, 2].copy())
        origins.append(T[:3, 3].copy())
        T = T @ tf(rot("z", qi))
    T = T @ TCP_TF
    return T, np.array(axes), np.array(origins)


def ik(p_goal, R_goal, q, iters=30, tol=1e-4):
    q = q.copy()
    for _ in range(iters):
        T, z, o = fk(q)
        e = np.concatenate([p_goal - T[:3, 3], rotvec(R_goal @ T[:3, :3].T)])
        if np.linalg.norm(e) < tol:
            break
        J = np.vstack([np.cross(z, T[:3, 3] - o).T, z.T])
        lam = 1e-3
        Jp = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(6))
        null = (np.eye(7) - Jp @ J) @ (0.2 * (READY - q))
        q = np.clip(q + Jp @ e + null, LOWER + 1e-3, UPPER - 1e-3)
    return q, np.linalg.norm(e[:3]), np.linalg.norm(e[3:])


def plan():
    """Returns (times, joint setpoints, gripper setpoints, events)."""
    T_ready, _, _ = fk(READY)
    R_down = T_ready[:3, :3]          # ready pose already points the hand down
    can_c = np.array([*CAN_XY, TABLE_Z + CAN_R])
    place_c = np.array([*PLATE_XY, PLATE_TOP + CAN_R + 0.004])

    grasp = can_c + [0, 0, GRASP_DZ]
    above_can = grasp + [0, 0, 0.14]
    place = place_c + [0, 0, GRASP_DZ]
    above_plate = place + [0, 0, 0.15]

    # (target TCP position or None to hold, duration [s], gripper target, label)
    segments = [
        (T_ready[:3, 3], 1.0, GRIP_OPEN, "settle at ready"),
        (above_can, 3.0, GRIP_OPEN, "move above can"),
        (grasp, 2.0, GRIP_OPEN, "descend to can"),
        (None, 1.5, GRIP_CLOSED, "close gripper"),
        (above_can, 2.0, GRIP_CLOSED, "lift can"),
        (above_plate, CARRY_S, GRIP_CLOSED, "carry to plate"),
        (place, 2.5, GRIP_CLOSED, "lower onto plate"),
        (None, 1.5, GRIP_OPEN, "release"),
        (above_plate, 2.0, GRIP_OPEN, "retreat"),
        (T_ready[:3, 3], 3.0, GRIP_OPEN, "return to ready"),
        (None, 1.5, GRIP_OPEN, "done"),
    ]

    q, p = READY.copy(), T_ready[:3, 3].copy()
    qs, grips, events, worst = [], [], [], (0.0, 0.0)
    for target, dur, grip, label in segments:
        events.append((len(qs) * DT, label))
        n = max(2, int(round(dur / DT)))
        goal = p if target is None else np.asarray(target, float)
        g0 = grips[-1] if grips else grip
        for s in min_jerk(n):
            q, ep, er = ik(p + s * (goal - p), R_down, q)
            worst = (max(worst[0], ep), max(worst[1], er))
            qs.append(q.copy())
            # ramp the gripper too, so the fingers meet the can slowly
            grips.append(g0 + s * (grip - g0))
        p = goal
    return np.arange(len(qs)) * DT, np.array(qs), np.array(grips), events, worst


if __name__ == "__main__":
    import time
    t0 = time.time()
    ts, qs, grips, events, worst = plan()
    print(f"Planned {len(ts)} setpoints ({ts[-1]:.1f} s) in {time.time()-t0:.1f} s; "
          f"worst IK error {worst[0]*1000:.2f} mm / {math.degrees(worst[1]):.3f} deg")
    for t, label in events:
        print(f"  t={t:5.1f}s  {label}")
