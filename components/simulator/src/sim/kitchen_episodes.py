"""Kitchen episodes for the mobile Panda: chains of the skills in
scripts/mobile_panda_skills.py (no Gazebo needed).

episode(name) returns a mobile_panda_skills.Scenario: one Plan (base and arm
moves, absolute targets per step) plus check windows (holds) and spans
(action ranges between two events, e.g. "door stays open during the cup
pick"). Targets come from scene() (the models' manifest.json files), so a
different fridge / counter asset with a manifest re-runs the same episode.

- fridge: from the room centre, drive to the fridge, pinch the top door
  handle, turn the whole robot about the hinge with the door (to 95 deg, door
  swings east; arm IK keeps the TCP on the door's arc), hold, release; drive
  in front of the open fridge, pinch the cup on the rack from the west side
  (hand horizontal: a top-down hand doesn't fit under the fridge ceiling), lift 3 cm,
  pull it out, back off, drive to the table (slower with the cup), move it
  over the table, lower, release.
- drawer: from the room centre, drive to the counter, pinch the drawer bar,
  pull it 19 cm, release, back off.
The asset-set numbers live in scripts/kitchen_assets.py (EPISODE).
"""
import math

import numpy as np

import kitchen_assets as ka
import mobile_panda as mp
import mobile_panda_skills as sk

# Asset-set constants (scripts/kitchen_assets.py): the cup side grasp must reach the shelf
# and the table with joint margin (scan: task4_reach_scan.py); swing shift and back-offs
# keep the chassis clear of the furniture and the open door (task4_plan_check.py).
EP = ka.EPISODE
V_DRIVE, W_DRIVE = EP["v_drive"], 1.0  # base peak speeds for the episodes' drives [m/s, rad/s]
V_CARRY, W_CARRY = 0.35, 0.6         # ... while carrying the cup
CUP_FRIDGE = ka.CUP_FRIDGE           # cup bottom on the fridge shelf (worlds/kitchen_partnet_skills.sdf)
DOOR_GOAL = EP["door_goal"]          # swing target (>= 60 deg required)
DOOR_MIN = math.radians(60.0)
SWING_DUR = EP["swing_dur"]
CUP_YAW = EP["cup_yaw"]              # hand direction into the fridge
CUP_BEARING, CUP_D, CUP_DELTA, CUP_ROLL = EP["cup_bearing"], EP["cup_d"], EP["cup_delta"], EP["cup_roll"]
CUP_GRASP_Z = EP["cup_grasp_z"]      # TCP above the cup bottom (hand-x-down: link7 hangs below the hand)
CUP_STAGE = EP["cup_stage"]          # arm to the pre-grasp pose here first (at the reach pose it sweeps the fridge wall)
SWING_SHIFT = EP["swing_shift"]      # base offset at the end of the swing
BACK_OFF = EP["back_off"]            # base move after the cup is out (clear of the open door's edge)
PLACE_XY = EP["place_xy"]            # cup bottom on the table (10 cm inside its west edge)
TABLE_YAW = EP["table_yaw"]          # hand direction at the table (east, across the west edge)
TABLE_D = EP["table_d"]
DRAWER_BACK = EP["drawer_back"]      # base back-off along the drawer normal before arm home
DRAWER_GOAL, DRAWER_MIN = 0.19, 0.15
BUDGET = {"fridge": 90.0, "drawer": 45.0}


def drive(P, goal, label="drive", v=None, w=None):
    """Min-jerk drive with the episodes' peak speeds (peak = 1.875 x mean)."""
    v, w = V_DRIVE if v is None else v, W_DRIVE if w is None else w
    goal = np.array(goal, float)
    d = goal - P.b
    d[2] = mp.wrap(d[2])
    dur = max(1.5, 1.875 * max(np.linalg.norm(d[:2]) / v, abs(d[2]) / w))
    P.drive(goal, dur, label=label)


def span(sc, kind, start, end, **info):
    """Check over the actions from event `start` to the end of event `end` (labels)."""
    sc.spans = getattr(sc, "spans", [])
    sc.spans.append(dict(kind=kind, start=start, end=end, info=info))


def fridge(S):
    sc = sk.Scenario("episode_fridge")
    P = sc.P
    # 1. door: drive, pinch the handle, swing about the hinge, hold, release
    f = S["fridge"]
    hd = sk.heading(f["normal"])
    p = np.array([f["handle"][0], f["handle"][1], ka.HANDLE_Z])
    drive(P, mp.base_for_reach(p + 0.06 * f["normal"], hd, ka.HANDLE_BEARING, 0.7), "drive fridge_reach")
    P.hold(0.3, "base settle")
    g = sk.grasp_fridge_handle(P, S, bearing=ka.HANDLE_BEARING, drive=False)
    sc.check("grasped", 0.5, obj="handle")
    sk.swing_door(P, S, g, 0.0, DOOR_GOAL, SWING_DUR, shift=SWING_SHIFT)
    sc.check("door_open", 1.0, obj="handle", want=DOOR_MIN)
    sk.release(P, back=0.20, dur=2.0, label="release handle")
    P.arm_joints(mp.READY, 2.5, label="arm ready")
    # 2. cup: in front of the open fridge, side pinch, lift, pull out
    cup = np.array(CUP_FRIDGE)
    base = mp.base_for_reach(cup + [0, 0, CUP_GRASP_Z], CUP_YAW - CUP_DELTA, CUP_BEARING, CUP_D)
    drive(P, (base[0], P.b[1], base[2]), "drive fridge_side")      # east first, clear of the door
    drive(P, base + [CUP_STAGE[0], CUP_STAGE[1], 0], "drive cup_reach")
    P.hold(0.3, "base settle")
    c = sk.pick_side(P, S, cup, CUP_YAW, CUP_BEARING, CUP_D, CUP_DELTA, CUP_ROLL, CUP_GRASP_Z,
                     stage=CUP_STAGE, drive=False)
    span(sc, "door_stays_open", "release handle", "arm extract", want=DOOR_MIN)
    # 3. table: back off, drive (base turned, arm pose kept), lower, release
    # gentler with the cup in the hand: at 0.6 m/s peak the cup swung 6.6 deg in the fingers (try 2)
    drive(P, (P.b[0] + BACK_OFF[0], P.b[1] + BACK_OFF[1], P.b[2]), "drive fridge_back", v=V_CARRY)
    sc.check("cup_out", 0.3, obj="cup")
    top = S["table"]["top"]
    place = np.array([*PLACE_XY, top])
    tb = mp.base_for_reach(place + [0, 0, CUP_GRASP_Z], TABLE_YAW - CUP_DELTA, CUP_BEARING, TABLE_D)
    drive(P, tb, "drive table_place", v=V_CARRY, w=W_CARRY)
    P.hold(0.3, "base settle")
    sk.place_side(P, S, place, TABLE_YAW, CUP_BEARING, TABLE_D, CUP_DELTA, CUP_ROLL, CUP_GRASP_Z, drive=False)
    sc.check("cup_placed", 2.0, place=place.tolist())
    P.hold(0.5, "done")
    sc.info = dict(grasp=g, cup=c, place=place, budget=BUDGET["fridge"])
    return sc


def drawer(S):
    sc = sk.Scenario("episode_drawer")
    P = sc.P
    dr = S["drawer"]
    p = np.array(dr["bar"], float)
    drive(P, mp.base_for_reach(p + 0.06 * dr["normal"], sk.heading(dr["normal"]), ka.DRAWER_BEARING, 0.7),
          "drive counter_reach")
    P.hold(0.3, "base settle")
    g = sk.grasp_drawer_bar(P, S, bearing=ka.DRAWER_BEARING, drive=False)
    sc.check("grasped", 0.5, obj="bar")
    sk.pull_drawer(P, S, g, DRAWER_GOAL, 3.0)
    sc.check("drawer_pulled", 0.5, obj="bar", want=DRAWER_GOAL)
    sk.release(P, back=0.20, dur=2.0)
    # back the base off before homing the arm: from in place the hand brushed the drawer front (try 1)
    n = np.asarray(dr["normal"], float)
    drive(P, (P.b[0] + DRAWER_BACK * n[0], P.b[1] + DRAWER_BACK * n[1], P.b[2]), "drive counter_back")
    P.arm_joints(mp.READY, 2.5, label="arm ready")
    sc.check("drawer_open", 1.0, want=DRAWER_MIN)
    P.hold(0.5, "done")
    sc.info = dict(grasp=g, budget=BUDGET["drawer"])
    return sc


EPISODES = {"fridge": fridge, "drawer": drawer}


def episode(name, S=None):
    return EPISODES[name](S or sk.scene())
