"""Scripted runs of the mobile SO-101 in the kitchen (no Gazebo needed), from the shared skills.py moves.

EPISODES:
- drawer: from the room centre, drive to the counter (the bar straight ahead, in the arm's plane), reach
  the bar with the tool horizontal and the jaws closing vertically (fixed jaw below), close, pull the
  drawer by backing the base off along the drawer axis with the arm held, hold, release, back off, arm to
  READY.
(No fridge episode and no skill-test SCENARIOS: not ported to this robot.)
Distances below are for scale 1 and grow with robot.yaml `scale` (dist()).
smoke_plan: spawn smoke + drawer-bar reach: settle 1 s, rest 10 s at READY (base and joint drift), gripper
close / open, drive 1 m in front of the counter, to the drawer reach pose (grasp_pose), TCP to the
pre-grasp 5 cm in front of the bar and to the grasp pose (jaws open), back out, arm to READY.
"""
import math

import numpy as np

import skills as sk
from robots.base import Plan, hand_rot

REACH_D = 0.36           # TCP ahead of the shoulder_lift axis at the grasp [m] (robot.base_for_reach)
FAR, PRE = 0.10, 0.05    # approach: joint move to FAR out along the bar normal, Cartesian to PRE, then in
PULL, PULL_S = 0.20, 4.0  # base back-off with the bar grasped [m, s]
HOLD_S = 1.5             # hold before the release (the task's open phase holds 1.0 s)
BACK = 0.15              # base back-off after the release [m]
DRAWER_MIN = 0.15
SMOKE_EE_TOL = 0.01      # TCP tolerance of the smoke's reach checks [m]


def dist(robot, d):
    return d * robot.scale


def grasp_pose(robot, S):
    """(TCP target, orientation, base pose) of the drawer bar grasp."""
    dr = S["drawer"]
    n = np.asarray(dr["normal"], float)
    p = np.array(dr["bar"], float)
    R = hand_rot(-n, np.array([0.0, 0.0, -1.0]))
    return p, R, robot.base_for_reach(p, sk.heading(n), d=dist(robot, REACH_D))


def drawer(S):
    sc = sk.Scenario("episode_drawer")
    P, rb = sc.P, sc.P.robot
    dr = S["drawer"]
    n, ax = np.asarray(dr["normal"], float), np.asarray(dr["axis"], float)
    p, R, base = grasp_pose(rb, S)
    far, pre = dist(rb, FAR), dist(rb, PRE)
    sk.drive(P, base, "drive counter_reach")
    P.hold(0.3, "base settle")
    P.arm_pose(p + far * n, R, 4.0, label="arm bar far")
    P.arm_pose(p + pre * n, R, 1.5, label="arm bar pre", cartesian=True)
    P.arm_pose(p, R, 2.5, label="arm bar in", cartesian=True)
    P.gripper(rb.gripper.closed_m, 1.5, label="grip close")
    P.hold(0.5, "grip settle")
    sc.check("grasped", 0.5, obj="bar")
    b0 = P.b.copy()
    P.base_path(lambda s: [b0[0] + s * PULL * ax[0], b0[1] + s * PULL * ax[1], b0[2]], PULL_S,
                label="pull drawer", info=dict(dist=PULL))
    sc.check("drawer_pulled", HOLD_S, obj="bar", want=PULL)
    P.gripper(rb.gripper.open_m, 1.5, label="release")
    sk.drive(P, (P.b[0] + BACK * n[0], P.b[1] + BACK * n[1], P.b[2]), "drive counter_back")
    P.arm_joints(rb.ready, 2.5, label="arm ready")
    sc.check("drawer_open", 1.0, want=DRAWER_MIN)
    P.hold(0.5, "done")
    sc.info = dict(grasp=dict(grasp=p, R=R, base=base), budget=sk.BUDGET["drawer"])
    return sc


EPISODES = {"drawer": drawer}
SCENARIOS = {}


def smoke_plan(robot, T):
    """-> (Plan, checks, targets); T (smoke_system.targets) is not used: the targets are the drawer grasp's."""
    gr = robot.gripper
    S = sk.scene()
    P = Plan(robot, sk.BASE0)
    checks = []

    def check(kind, dur, **info):
        P.hold(dur, f"check {kind} {info.get('name', '')}", info)
        checks.append((len(P.events) - 1, kind, info))

    P.hold(1.0, "settle")
    P.hold(10.0, "rest")
    P.gripper(gr.closed_m, 1.5, label="gripper close")
    check("grip", 0.5, name="closed_1", want=gr.closed_m)
    P.gripper(gr.open_m, 1.5, label="gripper open")
    check("grip", 0.5, name="open_1", want=gr.open_m)
    n = np.asarray(S["drawer"]["normal"], float)
    p, R, base = grasp_pose(robot, S)
    one_m = base + np.array([*(0.6 * n[:2]), 0.0])
    P.drive(one_m, label="drive counter_1m")
    check("base", 2.0, name="counter_1m", goal=one_m.tolist())
    P.drive(base, label="drive counter_reach")
    check("base", 1.0, name="counter_reach", goal=base.tolist())
    far = dist(robot, FAR)
    pre = p + dist(robot, PRE) * n
    P.arm_pose(p + far * n, R, 4.0, label="arm counter far")
    P.arm_pose(pre, R, 2.0, label="arm counter pre_grasp", cartesian=True)
    check("ee", 2.0, name="counter_pre_grasp", p=pre.tolist(), R=R.tolist(), tol=SMOKE_EE_TOL)
    P.arm_pose(p, R, 2.0, label="arm counter grasp", cartesian=True)
    check("ee", 2.0, name="counter_grasp", p=p.tolist(), R=R.tolist(), tol=SMOKE_EE_TOL)
    P.arm_pose(p + far * n, R, 2.0, label="arm counter out", cartesian=True)
    P.arm_joints(robot.ready, 3.0, label="arm ready")
    P.hold(1.0, "done")
    targets = dict(counter=dict(handle=p.tolist()), fridge=dict(handle=[math.nan] * 3))
    return P, checks, targets
