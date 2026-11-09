#!/usr/bin/env python3
"""Panda kinematics and the can-on-plate motion plan (no Gazebo dependency).

Plans Cartesian straight-line TCP segments with min-jerk timing and solves IK
for every 10 ms sample with damped least squares, seeded by the previous
sample so the joint path is continuous.

  panda_plan.py     print the plan and the worst IK error
"""
import math

import numpy as np

WORLD = "can_on_plate"
BASE = np.array([0.0, 0.0, 0.40])           # panda pose in the world
DT = 0.01                                    # setpoint period [s]

# franka_description robots/fer/kinematics.yaml: (xyz, rpy) per joint
JOINTS = [
    ((0, 0, 0.333), (0, 0, 0)),
    ((0, 0, 0), (-math.pi / 2, 0, 0)),
    ((0, -0.316, 0), (math.pi / 2, 0, 0)),
    ((0.0825, 0, 0), (math.pi / 2, 0, 0)),
    ((-0.0825, 0.384, 0), (-math.pi / 2, 0, 0)),
    ((0, 0, 0), (math.pi / 2, 0, 0)),
    ((0.088, 0, 0), (math.pi / 2, 0, 0)),
]
LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
FLANGE_Z, HAND_YAW, TCP_Z = 0.107, -math.pi / 4, 0.1034

# Joint viscous damping [N m s/rad]. Written into the SDF by gen_panda_sdf.py,
# where MuJoCo integrates it implicitly (stable at any inertia, unlike an
# explicit D gain), and fed forward by the controller so tracking doesn't lag.
ARM_DAMPING = [75.0, 100.0, 25.0, 35.0, 2.5, 2.5, 0.5]

# Link masses and centres of mass (link frame), from franka_description
# robots/fer/inertials.yaml and end_effectors/franka_hand/inertials.yaml.
LINK_MASS_COM = [
    (2.7907, (0.00033, -0.02204, -0.04762)),
    (2.5420, (0.00038, -0.09211, 0.01908)),
    (2.2513, (0.05152, 0.01696, -0.02971)),
    (2.2037, (-0.05113, 0.05825, 0.01698)),
    (2.2855, (-0.00005, 0.03730, -0.09280)),
    (1.353, (0.06572, -0.00371, 0.00153)),
    (0.35973, (0.00089, -0.00044, 0.05491)),
]
# hand (0.6544 kg) + both fingers (2 x 0.0291 kg), in the hand frame
HAND_MASS_COM = [(0.6544, (-0.0000376, 0.0119128, 0.020726)),
                 (0.0582, (0.0, 0.0, 0.080))]
GRAVITY = 9.81

# Scene (must match the world file)
# The can lies on its side with its axis along world x; the fingers close
# along world y, across its diameter, level with its centre of mass.
CAN_XY, CAN_H, CAN_R = np.array([0.55, -0.20]), 0.10, 0.03
PLATE_XY, PLATE_TOP, PLATE_R = np.array([0.50, 0.22]), 0.415, 0.11
TABLE_Z = 0.40
GRASP_DZ = 0.0             # TCP height relative to the can's axis
GRIP_OPEN, GRIP_CLOSED = 0.04, 0.0
CARRY_S = float(__import__("os").environ.get("CARRY_S", 3.5))   # carry duration [s]


def rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def tf(R=np.eye(3), p=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


JOINT_TF = [tf(rot("z", r[2]) @ rot("y", r[1]) @ rot("x", r[0]), p)
            for p, r in JOINTS]
TCP_TF = tf(p=(0, 0, FLANGE_Z)) @ tf(rot("z", HAND_YAW)) @ tf(p=(0, 0, TCP_Z))


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


def gravity_torques(q):
    """Joint torques that hold the arm (incl. hand + fingers) against gravity."""
    T = tf(p=BASE)
    axes, origins, coms = [], [], []
    for Tj, qi, (m, c) in zip(JOINT_TF, q, LINK_MASS_COM):
        T = T @ Tj
        axes.append(T[:3, 2].copy())
        origins.append(T[:3, 3].copy())
        T = T @ tf(rot("z", qi))
        coms.append((m, (T @ np.array([*c, 1.0]))[:3]))
    Th = T @ tf(p=(0, 0, FLANGE_Z)) @ tf(rot("z", HAND_YAW))
    coms += [(m, (Th @ np.array([*c, 1.0]))[:3]) for m, c in HAND_MASS_COM]
    tau = np.zeros(7)
    for i in range(7):
        for k, (m, c) in enumerate(coms):
            if k >= i:     # bodies outboard of joint i
                tau[i] += m * GRAVITY * np.cross(axes[i], c - origins[i])[2]
    return tau


def rotvec(R):
    """Rotation matrix -> axis*angle."""
    a = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if a < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * a / (2 * math.sin(a))


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


def min_jerk(n):
    s = np.linspace(0.0, 1.0, n)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


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
