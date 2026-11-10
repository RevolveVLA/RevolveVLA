"""Scripted runs of the mobile Panda in the kitchen (no Gazebo needed): its grasp skills (tuned to its
reach: base ~0.7 m from panda_link0, handles ~45 deg off the arm's forward axis), the skill-test
scenarios (SCENARIOS), the scripted episodes (EPISODES) and the robot-smoke plan (smoke_plan). Built from
the shared skills.py moves; targets come from skills.scene() (the models' manifest.json files), so a
different fridge / counter asset with a manifest re-runs the same plans. The numbers tuned to this robot on
the partnet kitchen (bearings, episode poses) are below (PARTNET); the asset set's own in kitchen_assets.py.

Episodes:
- fridge: from the room centre, drive to the fridge, pinch the top door handle, turn the whole robot
  about the hinge with the door (to 95 deg, door swings east; arm IK keeps the TCP on the door's arc),
  hold, release; drive in front of the open fridge, pinch the cup on the rack from the west side (hand
  horizontal: a top-down hand doesn't fit under the fridge ceiling), lift 3 cm, pull it out, back off,
  drive to the table (slower with the cup), move it over the table, lower, release.
- drawer: from the room centre, drive to the counter, pinch the drawer bar, pull it 19 cm, release,
  back off.
Skill tests (SCENARIOS, $SKILL of the skill system):
  fridge_handle   grasp the top door handle, pull the door 8 deg on its arc, release
  drawer_pull     grasp the drawer bar (hand rolled 90 deg), pull it 15 cm, release
  cup_pick_place  cup on the table: pick, lift 15 cm, hold 2 s, place 18 cm west, release
"""
import math

import numpy as np

import kitchen_assets as ka
import skills as sk
from robots.base import Plan, hand_rot, rot

BEARING = math.radians(45)     # handle/bar ~45 deg off the arm's forward axis
CUP_GRASP_Z = 0.050            # TCP (finger pad centre) above the cup bottom (COM 0.044)
CUP_SIDE_GRASP_Z = 0.058       # side grasp: hand body (+-3 cm) clears the shelf under the cup

# Tuned to this robot's reach on the partnet kitchen (kitchen_assets.py, $KITCHEN_ASSETS=partnet)
PARTNET = dict(
    # handle on the door's west edge: reach it from the west
    handle_bearing=math.radians(-45),
    drawer_bearing=math.radians(45),
    # 10 cm: the joint move home brushed the drawer front (14.8 -> 13.4 cm)
    drawer_release_back=0.20,
    episode=dict(
        # door to 95 deg: at 64 its edge leaves < 1 cm for the forearm at the cup
        door_goal=math.radians(95.0), swing_dur=8.0, swing_shift=(0.0, 0.0),
        # the open door blocks the east side: reach the cup from the west;
        # hand x up (the wrist above the hand, clear of the rack's front lip)
        cup_yaw=math.radians(100.0), cup_bearing=math.radians(-45), cup_d=0.73,
        cup_delta=math.radians(20.0), cup_roll=1, cup_grasp_z=0.084,
        cup_stage=(-0.15, -0.20),
        back_off=(-0.25, -0.45),
        place_xy=(0.895, -0.30), table_yaw=0.0, table_d=0.70,
        drawer_back=0.3,
    ),
)
HANDLE_BEARING = PARTNET["handle_bearing"]
DRAWER_BEARING = PARTNET["drawer_bearing"]
DRAWER_RELEASE_BACK = PARTNET["drawer_release_back"]


# ------------------------------------------------------------------ skills
def grasp_fridge_handle(P, S, z=ka.HANDLE_Z, bearing=HANDLE_BEARING, d=0.7, depth=0.0, drive=True):
    """Drive to the reach pose and pinch the top door handle (door closed).
    depth: TCP offset behind the handle centre towards the door [m]."""
    f = S["fridge"]
    n = f["normal"]
    p = np.array([f["handle"][0], f["handle"][1], z]) - depth * n
    hd = sk.heading(n)
    R = hand_rot(-n, rot("z", hd) @ np.array([0.0, 1.0, 0.0]))
    base = P.robot.base_for_reach(p - 0.06 * (-n), hd, bearing, d)
    if drive:
        P.drive(base, label="drive fridge_reach")
        P.hold(0.5, "base settle")
    sk.approach(P, p, R, label="handle")
    sk.close(P)
    return dict(grasp=p, R=R, base=base)


def grasp_drawer_bar(P, S, bearing=DRAWER_BEARING, d=0.7, drive=True):
    """Drive to the reach pose and pinch the drawer bar (hand rolled 90 deg,
    fingers close vertically across the bar's 25 mm height)."""
    dr = S["drawer"]
    n = dr["normal"]
    p = np.array(dr["bar"], float)
    hd = sk.heading(n)
    R = hand_rot(-n, np.array([0.0, 0.0, -1.0]))
    base = P.robot.base_for_reach(p + 0.06 * n, hd, bearing, d)
    if drive:
        P.drive(base, label="drive counter_reach")
        P.hold(0.5, "base settle")
    sk.approach(P, p, R, label="bar")
    sk.close(P)
    return dict(grasp=p, R=R, base=base)


def table_base(robot, S, xy, d=0.55):
    """Base pose north of the table, facing south, `d` from link0 to xy."""
    t = S["table"]
    return robot.base_for_reach(np.array([xy[0], xy[1], t["top"]]), -math.pi / 2, 0.0, d)


def pick(P, S, cup_xyz, lift=0.15, drive=True, base=None):
    """Top-down pinch of an upright cup (bottom at cup_xyz), then lift."""
    base = table_base(P.robot, S, cup_xyz[:2]) if base is None else base
    if drive:
        P.drive(base, label="drive table_reach")
        P.hold(0.5, "base settle")
    R = P.robot.fk(P.robot.ready, base)[0][:3, :3]        # hand down, as at READY
    p = np.asarray(cup_xyz, float) + [0, 0, CUP_GRASP_Z]
    sk.approach(P, p, R, far=0.14, pre=0.07, t_in=2.0, label="cup")
    sk.close(P)
    P.arm_pose(p + [0, 0, lift], R, 2.5, label="arm lift", cartesian=True)
    return dict(grasp=p, R=R, base=base, lifted=p + [0, 0, lift])


def side_grasp(yaw, roll=1):
    """Horizontal hand along `yaw`, fingers closing horizontally; roll +1: hand x up, -1: down."""
    u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    return u, hand_rot(u, roll * np.array([u[1], -u[0], 0.0]))


def pick_side(P, S, cup_xyz, yaw, bearing=BEARING, d=0.7, delta=0.0, roll=1, grasp_z=CUP_SIDE_GRASP_Z,
              lift=0.03, out=0.30, far=0.35, stage=None, drive=True):
    """Horizontal pinch of an upright cup (bottom at cup_xyz) with the hand
    pointing along `yaw` (e.g. into a fridge shelf, where a top-down hand
    doesn't fit), lift it `lift` and pull it `out` m back along the hand axis.
    The base faces yaw - delta; the joint-space move ends `far` m out (outside
    the fridge), the rest is Cartesian. stage: base xy offset at which the
    joint-space move is made (the base then drives in with the arm fixed),
    when the arm would sweep through furniture at the reach pose."""
    u, R = side_grasp(yaw, roll)
    p = np.asarray(cup_xyz, float) + [0, 0, grasp_z]
    base = P.robot.base_for_reach(p, yaw - delta, bearing, d)
    st = np.zeros(3) if stage is None else np.array([stage[0], stage[1], 0.0])
    if drive:
        P.drive(base + st, label="drive cup_reach")
        P.hold(0.5, "base settle")
    P.arm_pose(p - far * u + st, R, 3.0, label="arm cup far")
    if stage is not None:
        P.drive(base, 2.0, label="drive cup_in")
    P.arm_pose(p - 0.07 * u, R, 2.5, label="arm cup pre", cartesian=True)
    P.arm_pose(p, R, 1.5, label="arm cup in", cartesian=True)
    sk.close(P)
    P.arm_pose(p + [0, 0, lift], R, 1.5, label="arm lift", cartesian=True)
    P.arm_pose(p + [0, 0, lift] - out * u, R, 2.5, label="arm extract", cartesian=True)
    return dict(grasp=p, R=R, base=base, lifted=p + [0, 0, lift], out=p + [0, 0, lift] - out * u)


def place_side(P, S, place_xyz, yaw, bearing=BEARING, d=0.7, delta=0.0, roll=1, grasp_z=CUP_SIDE_GRASP_Z,
               clearance=0.003, drive=True):
    """Counterpart of pick_side: drive to the reach pose of place_xyz (cup
    bottom) with the cup held, move it forward to above the place point at
    carry height, lower it, release and back out along the hand axis. Put
    place_xyz within ~12 cm of an edge the hand points across: link6 hangs
    near the hand's height and must stay outside the table."""
    u, R = side_grasp(yaw, roll)
    p = np.asarray(place_xyz, float) + [0, 0, grasp_z + clearance]
    base = P.robot.base_for_reach(p, yaw - delta, bearing, d)
    if drive:
        P.drive(base, label="drive place_reach")
        P.hold(0.3, "base settle")
    z = P.tcp()[2, 3]
    P.arm_pose(np.array([p[0], p[1], z]), R, 2.5, label="arm forward", cartesian=True)
    P.arm_pose(p, R, 2.5, label="arm lower", cartesian=True)
    P.hold(0.3, "place settle")
    sk.release(P, back=0.12)
    return dict(place=p, base=base)


def place(P, S, grasp, place_xyz, clearance=0.003, carry=3.0, grasp_z=CUP_GRASP_Z, R=None):
    """Carry the held cup above place_xyz (cup bottom), lower it, release, retreat.
    R: hand orientation (default the grasp's; pass the current one if the base turned)."""
    R = grasp["R"] if R is None else R
    p = np.asarray(place_xyz, float) + [0, 0, grasp_z + clearance]
    above = np.array([p[0], p[1], P.tcp()[2, 3]])
    P.arm_pose(above, R, carry, label="arm carry", cartesian=True)
    P.arm_pose(p, R, 2.5, label="arm lower", cartesian=True)
    P.hold(0.3, "place settle")
    sk.release(P, back=0.12)
    return dict(place=p)


# ------------------------------------------------- skill-test scenarios
DOOR_PULL = math.radians(8.0)           # "a few degrees"
DRAWER_PULL = 0.15                      # >= 10 cm required
CUP_START = (1.50, -0.25)               # cup bottom on the table top (xy)
CUP_PLACE = (1.32, -0.25)


def _finish(sc):
    sc.P.hold(0.5, "done")
    return sc


def fridge_handle(S):
    sc = sk.Scenario("fridge_handle")
    P = sc.P
    g = grasp_fridge_handle(P, S)
    sc.check("grasped", 0.5, obj="handle")
    sk.pull_door(P, S, g, DOOR_PULL, 3.0)
    sc.check("door_pulled", 1.0, obj="handle", want=DOOR_PULL)
    # back out 20 cm before the joint move home: from 10 cm the hand swept into the handle
    sk.release(P, back=0.20, dur=2.5)
    P.arm_joints(P.robot.ready, 3.0, label="arm ready")
    sc.check("door_released", 1.0, want=DOOR_PULL)
    sc.info = dict(grasp=g)
    return _finish(sc)


def drawer_pull(S):
    sc = sk.Scenario("drawer_pull")
    P = sc.P
    g = grasp_drawer_bar(P, S)
    sc.check("grasped", 0.5, obj="bar")
    sk.pull_drawer(P, S, g, DRAWER_PULL, 3.0)
    sc.check("drawer_pulled", 0.5, obj="bar", want=DRAWER_PULL)
    sk.release(P, back=DRAWER_RELEASE_BACK)
    P.arm_joints(P.robot.ready, 3.0, label="arm ready")
    sc.check("drawer_released", 1.0, want=DRAWER_PULL)
    sc.info = dict(grasp=g)
    return _finish(sc)


def cup_pick_place(S):
    sc = sk.Scenario("cup_pick_place")
    P = sc.P
    top = S["table"]["top"]
    start = np.array([*CUP_START, top])
    base = table_base(P.robot, S, start[:2])
    # sideways first: the straight line from the spawn pose clips the table's corner
    P.drive((P.b[0], base[1], base[2]), label="drive table_side")
    g = pick(P, S, start)
    sc.check("cup_lifted", 2.0, obj="cup", z0=float(top))
    pl = place(P, S, g, np.array([*CUP_PLACE, top]))
    P.arm_joints(P.robot.ready, 3.0, label="arm ready")
    sc.check("cup_placed", 2.0, place=[*CUP_PLACE, top])
    sc.info = dict(grasp=g, place=pl, cup_start=start)
    return _finish(sc)


SCENARIOS = {"fridge_handle": fridge_handle, "drawer_pull": drawer_pull, "cup_pick_place": cup_pick_place}


# ------------------------------------------------- scripted episodes
# Episode constants (PARTNET above): the cup side grasp must reach the shelf and the table with
# joint margin; swing shift and back-offs keep the chassis clear of the furniture and the open door.
EP = PARTNET["episode"]
V_CARRY, W_CARRY = 0.35, 0.6         # base peak speeds while carrying the cup
CUP_FRIDGE = ka.CUP_FRIDGE           # cup bottom on the fridge shelf (worlds/kitchen_partnet_skills.sdf)
DOOR_GOAL = EP["door_goal"]          # swing target (>= 60 deg required)
DOOR_MIN = math.radians(60.0)
SWING_DUR = EP["swing_dur"]
CUP_YAW = EP["cup_yaw"]              # hand direction into the fridge
CUP_BEARING, CUP_D, CUP_DELTA, CUP_ROLL = EP["cup_bearing"], EP["cup_d"], EP["cup_delta"], EP["cup_roll"]
EP_CUP_GRASP_Z = EP["cup_grasp_z"]   # TCP above the cup bottom (hand-x-down: link7 hangs below the hand)
CUP_STAGE = EP["cup_stage"]          # arm to the pre-grasp pose here first (at the reach pose it sweeps the fridge wall)
SWING_SHIFT = EP["swing_shift"]      # base offset at the end of the swing
BACK_OFF = EP["back_off"]            # base move after the cup is out (clear of the open door's edge)
PLACE_XY = EP["place_xy"]            # cup bottom on the table (10 cm inside its west edge)
TABLE_YAW = EP["table_yaw"]          # hand direction at the table (east, across the west edge)
TABLE_D = EP["table_d"]
DRAWER_BACK = EP["drawer_back"]      # base back-off along the drawer normal before arm home
DRAWER_GOAL, DRAWER_MIN = 0.19, 0.15


def fridge(S):
    sc = sk.Scenario("episode_fridge")
    P, rb = sc.P, sc.P.robot
    # 1. door: drive, pinch the handle, swing about the hinge, hold, release
    f = S["fridge"]
    hd = sk.heading(f["normal"])
    p = np.array([f["handle"][0], f["handle"][1], ka.HANDLE_Z])
    sk.drive(P, rb.base_for_reach(p + 0.06 * f["normal"], hd, HANDLE_BEARING, 0.7), "drive fridge_reach")
    P.hold(0.3, "base settle")
    g = grasp_fridge_handle(P, S, bearing=HANDLE_BEARING, drive=False)
    sc.check("grasped", 0.5, obj="handle")
    sk.swing_door(P, S, g, 0.0, DOOR_GOAL, SWING_DUR, shift=SWING_SHIFT)
    sc.check("door_open", 1.0, obj="handle", want=DOOR_MIN)
    sk.release(P, back=0.20, dur=2.0, label="release handle")
    P.arm_joints(rb.ready, 2.5, label="arm ready")
    # 2. cup: in front of the open fridge, side pinch, lift, pull out
    cup = np.array(CUP_FRIDGE)
    base = rb.base_for_reach(cup + [0, 0, EP_CUP_GRASP_Z], CUP_YAW - CUP_DELTA, CUP_BEARING, CUP_D)
    sk.drive(P, (base[0], P.b[1], base[2]), "drive fridge_side")      # east first, clear of the door
    sk.drive(P, base + [CUP_STAGE[0], CUP_STAGE[1], 0], "drive cup_reach")
    P.hold(0.3, "base settle")
    c = pick_side(P, S, cup, CUP_YAW, CUP_BEARING, CUP_D, CUP_DELTA, CUP_ROLL, EP_CUP_GRASP_Z,
                  stage=CUP_STAGE, drive=False)
    sk.span(sc, "door_stays_open", "release handle", "arm extract", want=DOOR_MIN)
    # 3. table: back off, drive (base turned, arm pose kept), lower, release
    # gentler with the cup in the hand: at 0.6 m/s peak the cup swung 6.6 deg in the fingers (try 2)
    sk.drive(P, (P.b[0] + BACK_OFF[0], P.b[1] + BACK_OFF[1], P.b[2]), "drive fridge_back", v=V_CARRY)
    sc.check("cup_out", 0.3, obj="cup")
    top = S["table"]["top"]
    place_ = np.array([*PLACE_XY, top])
    tb = rb.base_for_reach(place_ + [0, 0, EP_CUP_GRASP_Z], TABLE_YAW - CUP_DELTA, CUP_BEARING, TABLE_D)
    sk.drive(P, tb, "drive table_place", v=V_CARRY, w=W_CARRY)
    P.hold(0.3, "base settle")
    place_side(P, S, place_, TABLE_YAW, CUP_BEARING, TABLE_D, CUP_DELTA, CUP_ROLL, EP_CUP_GRASP_Z, drive=False)
    sc.check("cup_placed", 2.0, place=place_.tolist())
    P.hold(0.5, "done")
    sc.info = dict(grasp=g, cup=c, place=place_, budget=sk.BUDGET["fridge"])
    return sc


def drawer(S):
    sc = sk.Scenario("episode_drawer")
    P, rb = sc.P, sc.P.robot
    dr = S["drawer"]
    p = np.array(dr["bar"], float)
    sk.drive(P, rb.base_for_reach(p + 0.06 * dr["normal"], sk.heading(dr["normal"]), DRAWER_BEARING, 0.7),
             "drive counter_reach")
    P.hold(0.3, "base settle")
    g = grasp_drawer_bar(P, S, bearing=DRAWER_BEARING, drive=False)
    sc.check("grasped", 0.5, obj="bar")
    sk.pull_drawer(P, S, g, DRAWER_GOAL, 3.0)
    sc.check("drawer_pulled", 0.5, obj="bar", want=DRAWER_GOAL)
    sk.release(P, back=0.20, dur=2.0)
    # back the base off before homing the arm: from in place the hand brushed the drawer front (try 1)
    n = np.asarray(dr["normal"], float)
    sk.drive(P, (P.b[0] + DRAWER_BACK * n[0], P.b[1] + DRAWER_BACK * n[1], P.b[2]), "drive counter_back")
    P.arm_joints(rb.ready, 2.5, label="arm ready")
    sc.check("drawer_open", 1.0, want=DRAWER_MIN)
    P.hold(0.5, "done")
    sc.info = dict(grasp=g, budget=sk.BUDGET["drawer"])
    return sc


EPISODES = {"fridge": fridge, "drawer": drawer}


# ------------------------------------------------- robot smoke
STANDOFF = 0.08                            # TCP in front of a handle's front face [m]


def smoke_plan(robot, T):
    """The robot-smoke plan (smoke_system.Smoke): settle 1 s, rest 10 s (drift check), gripper close/open,
    drive 1 m in front of the fridge, then to a reach pose and move the TCP near the top door handle
    (z 1.15), back to READY; same for the kitchen_counter drawer bar and above the table, where the gripper
    closes and opens again. T: the smoke targets (smoke_system.targets). -> (Plan, checks, T)."""
    gr = robot.gripper
    P = Plan(robot, sk.BASE0)
    checks = []         # (event index of the hold, kind, info)

    def check(kind, dur, **info):
        P.hold(dur, f"check {kind} {info.get('name', '')}", info)
        checks.append((len(P.events) - 1, kind, info))

    P.hold(1.0, "settle")
    P.hold(10.0, "rest")
    P.gripper(gr.closed_m, 1.5, label="gripper close")
    check("grip", 0.5, name="closed_1", want=gr.closed_m)
    P.gripper(gr.open_m, 1.5, label="gripper open")
    check("grip", 0.5, name="open_1", want=gr.open_m)

    def visit(name, t, closing_arm, bearing, label):
        hd = sk.heading(t["normal"])
        one_m = np.array([*(t["face"][:2] + 1.0 * t["normal"][:2]), hd])
        P.drive(one_m, label=f"drive {name}_1m")
        check("base", 2.0, name=f"{name}_1m", goal=one_m.tolist())
        pre = t["front"] + STANDOFF * t["normal"]
        far = t["front"] + (STANDOFF + 0.07) * t["normal"]
        base = robot.base_for_reach(pre, hd, bearing, 0.7)
        P.drive(base, label=f"drive {name}_reach")
        check("base", 1.0, name=f"{name}_reach", goal=base.tolist())
        closing = rot("z", hd) @ np.array(closing_arm, float)
        R = hand_rot(-t["normal"], closing)
        P.arm_pose(far, R, 4.0, label=f"arm {name} far")
        P.arm_pose(pre, R, 2.0, label=f"arm {name} {label}", cartesian=True)
        check("ee", 2.0, name=f"{name}_{label}", p=pre.tolist(), R=R.tolist())
        P.arm_joints(robot.ready, 4.0, label="arm ready")

    visit("fridge", T["fridge"], (0, 1, 0), math.radians(45), "handle")
    visit("counter", T["counter"], (0, 0, -1), math.radians(45), "drawer_bar")

    t = T["table"]
    hd = sk.heading(t["normal"])
    one_m = np.array([t["centre"][0], t["face"][1] + 1.0, hd])
    P.drive(one_m, label="drive table_1m")
    check("base", 2.0, name="table_1m", goal=one_m.tolist())
    above = np.array([t["centre"][0], t["face"][1] - 0.18, t["top"] + 0.20])
    base = robot.base_for_reach(above, hd, 0.0, 0.55)
    P.drive(base, label="drive table_reach")
    check("base", 1.0, name="table_reach", goal=base.tolist())
    R = robot.fk(robot.ready, base)[0][:3, :3]          # hand down, as at READY
    P.arm_pose(above, R, 3.0, label="arm table above")
    check("ee", 1.0, name="table_above", p=above.tolist(), R=R.tolist())
    P.gripper(gr.closed_m, 1.5, label="gripper close")
    check("grip", 0.5, name="closed_2", want=gr.closed_m)
    P.gripper(gr.open_m, 1.5, label="gripper open")
    check("grip", 0.5, name="open_2", want=gr.open_m)
    P.arm_joints(robot.ready, 3.0, label="arm ready")
    P.hold(1.0, "done")
    return P, checks, T
