"""Reusable contact skills for the mobile Panda in the kitchen (no Gazebo needed).

Each skill appends moves to a mobile_panda.Plan (base first, then arm) and
returns a dict of the world-frame targets it used, so episodes can chain
skills and checks can refer to them. Scene geometry comes from the models'
manifest.json files (scene()), with the model poses of worlds/kitchen_partnet_robot.sdf
(and worlds/kitchen_partnet_skills.sdf).

Skills:
- grasp_fridge_handle / pull_door: pinch the top door handle across its 50 mm
  width (fingers towards the door), then move the TCP on the arc of the door
  around its hinge (hand turns with the door).
- grasp_drawer_bar / pull_drawer: pinch the 25 mm high bar vertically (hand
  rolled 90 deg), then pull straight out along the drawer axis.
- pick / place: top-down pinch across the cup's diameter at its centre of mass.
- release: open the gripper and back the TCP out along the hand's -z.
All contact phases are effort-controlled (Controller: P + gravity comp + SDF
damping, fingers force-capped); nothing uses velocity commands.
"""
import json
import math
import os

import numpy as np

import kitchen_assets as ka
import mobile_panda as mp
import panda_plan as pp

# Model poses (x, y, yaw) of the asset set ($KITCHEN_ASSETS); must match the world files
POSES = ka.POSES
TABLE_Z0 = 0.002
BEARING = math.radians(45)     # handle/bar ~45 deg off the arm's forward axis (README)
CUP_H, CUP_R = 0.089, 0.03
CUP_GRASP_Z = 0.050            # TCP (finger pad centre) above the cup bottom (COM 0.044)
CUP_SIDE_GRASP_Z = 0.058       # side grasp: hand body (+-3 cm) clears the shelf under the cup


def model_dir(name):
    for d in os.environ.get("GZ_SIM_RESOURCE_PATH", "").split(":") + [
            os.path.join(os.path.dirname(__file__), "..", "assets", "generated")]:
        if d and os.path.isfile(os.path.join(d, name, "manifest.json")):
            return os.path.join(d, name)
    raise RuntimeError(f"model {name} not found on GZ_SIM_RESOURCE_PATH")


def model_tf(name):
    x, y, yaw = POSES[name]
    return pp.tf(pp.rot("z", yaw), (x, y, TABLE_Z0 if name == "table_01" else 0.0))


def to_world(name, p):
    return (model_tf(name) @ np.array([*p, 1.0]))[:3]


def heading(normal):
    """Base yaw facing a furniture front whose outward normal is `normal`."""
    return math.atan2(-normal[1], -normal[0])


def scene():
    """World-frame geometry used by the skills."""
    man = {}
    for n in POSES:
        with open(os.path.join(model_dir(n), "manifest.json")) as f:
            man[n] = json.load(f)
    S = {}
    # Fridge top door: handle box (45 deep x 50 wide x 550 tall) on the door front
    fm, R_f = man["fridge"], model_tf("fridge")[:3, :3]
    h = next(x for x in fm["handles"] if x["name"] == "top_door_handle")
    j = fm["joints"]["top_door_hinge"]
    S["fridge"] = dict(handle=to_world("fridge", h["center_model_frame"]), handle_size=h["size"],
                       normal=R_f[:, 0], hinge=to_world("fridge", j["origin_model_frame"]),
                       axis=R_f @ np.array(j["axis_model_frame"]), joint="top_door_hinge",
                       limits=(j["lower"], j["upper"]))
    # Drawer bar (20 deep x 25 high x 220 long), 35 mm in front of the drawer face
    cm, R_c = man["kitchen_counter"], model_tf("kitchen_counter")[:3, :3]
    h = next(x for x in cm["handles"] if x["name"] == "drawer2_top_handle_bar")
    j = cm["joints"]["middle_slide_top"]
    S["drawer"] = dict(bar=to_world("kitchen_counter", h["center_model_frame"]), bar_size=h["size"],
                       normal=R_c[:, 0], axis=R_c @ np.array(j["axis_model_frame"]),
                       joint="middle_slide_top", limits=(j["lower"], j["upper"]))
    lo, hi = man["table_01"]["aabb_model_frame"]
    S["table"] = dict(top=hi[2] + TABLE_Z0, centre=to_world("table_01", (0, 0, 0)),
                      north=to_world("table_01", (0, hi[1], 0))[1], half=(hi[0], hi[1]))
    return S


def _rot_about(axis, a):
    u = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


# ------------------------------------------------------------------ skills
def approach(P, p, R, far=0.15, pre=0.06, t_far=4.0, t_in=2.5, label="target"):
    """Joint-space move to `far` m out along the hand's -z, Cartesian to `pre`, then in to p."""
    z = R[:, 2]
    P.arm_pose(p - far * z, R, t_far, label=f"arm {label} far")
    P.arm_pose(p - pre * z, R, 1.5, label=f"arm {label} pre", cartesian=True)
    P.arm_pose(p, R, t_in, label=f"arm {label} in", cartesian=True)


def close(P, dur=1.5, force=mp.GRIP_FORCE, settle=0.5, label="grip close"):
    P.gripper(mp.GRIP_CLOSED, dur, force=force, label=label)
    P.hold(settle, "grip settle")


def release(P, back=0.10, dur=1.5, label="release"):
    """Open, then back the TCP out along the hand's -z."""
    P.gripper(mp.GRIP_OPEN, 1.5, label=label)
    T = P.tcp()
    P.arm_pose(T[:3, 3] - back * T[:3, 2], T[:3, :3], dur, label="arm retreat", cartesian=True)


def grasp_fridge_handle(P, S, z=ka.HANDLE_Z, bearing=ka.HANDLE_BEARING, d=0.7, depth=0.0, drive=True):
    """Drive to the reach pose and pinch the top door handle (door closed).
    depth: TCP offset behind the handle centre towards the door [m]."""
    f = S["fridge"]
    n = f["normal"]
    p = np.array([f["handle"][0], f["handle"][1], z]) - depth * n
    hd = heading(n)
    R = mp.hand_rot(-n, pp.rot("z", hd) @ np.array([0.0, 1.0, 0.0]))
    base = mp.base_for_reach(p - 0.06 * (-n), hd, bearing, d)
    if drive:
        P.drive(base, label="drive fridge_reach")
        P.hold(0.5, "base settle")
    approach(P, p, R, label="handle")
    close(P)
    return dict(grasp=p, R=R, base=base)


def door_tcp(S, p0, R0, angle):
    """TCP pose that keeps a grasp (p0, R0 taken at door angle 0) at door angle `angle`."""
    f = S["fridge"]
    Rz = _rot_about(f["axis"], angle)
    return f["hinge"] + Rz @ (p0 - f["hinge"]), Rz @ R0


def pull_door(P, S, grasp, angle, dur, a0=0.0, label="pull door"):
    """Move the grasped handle on the door's arc from a0 to `angle` [rad]."""
    P.arm_path(lambda s: door_tcp(S, grasp["grasp"], grasp["R"], a0 + s * (angle - a0)),
               dur, label=label, info=dict(angle=angle))
    return door_tcp(S, grasp["grasp"], grasp["R"], angle)


def swing_door(P, S, grasp, a0, a1, dur, shift=(0.0, 0.0), label="swing door"):
    """Turn the whole robot about the door hinge from door angle a0 to a1 [rad]
    with the handle grasped: the base turns about the hinge with the door, so
    the arm joints barely move and the grasp only carries the door's
    damping/friction/inertia. shift: world xy offset of the base reached at
    the end (to keep the chassis clear of nearby furniture); the arm then
    keeps the TCP on the door's arc by IK."""
    f = S["fridge"]
    h, sgn = np.asarray(f["hinge"][:2], float), float(np.sign(f["axis"][2]))
    b0, shift = P.b.copy(), np.asarray(shift, float)

    def base(s):
        a = sgn * (a1 - a0) * s
        R2 = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        return [*(h + R2 @ (b0[:2] - h) + s * shift), b0[2] + a]
    if not shift.any():
        P.base_path(base, dur, label=label, info=dict(angle=a1))
    else:
        P.arm_path(lambda s: door_tcp(S, grasp["grasp"], grasp["R"], a0 + s * (a1 - a0)), dur,
                   label=label, info=dict(angle=a1), base=base)
    return base(1.0)


def grasp_drawer_bar(P, S, bearing=ka.DRAWER_BEARING, d=0.7, drive=True):
    """Drive to the reach pose and pinch the drawer bar (hand rolled 90 deg,
    fingers close vertically across the bar's 25 mm height)."""
    dr = S["drawer"]
    n = dr["normal"]
    p = np.array(dr["bar"], float)
    hd = heading(n)
    R = mp.hand_rot(-n, np.array([0.0, 0.0, -1.0]))
    base = mp.base_for_reach(p + 0.06 * n, hd, bearing, d)
    if drive:
        P.drive(base, label="drive counter_reach")
        P.hold(0.5, "base settle")
    approach(P, p, R, label="bar")
    close(P)
    return dict(grasp=p, R=R, base=base)


def pull_drawer(P, S, grasp, dist, dur, label="pull drawer"):
    """Pull the grasped bar `dist` m out along the drawer axis."""
    p = grasp["grasp"] + dist * np.asarray(S["drawer"]["axis"])
    P.arm_pose(p, grasp["R"], dur, label=label, cartesian=True, info=dict(dist=dist))
    return p


def table_base(S, xy, d=0.55):
    """Base pose north of the table, facing south, `d` from link0 to xy."""
    t = S["table"]
    return mp.base_for_reach(np.array([xy[0], xy[1], t["top"]]), -math.pi / 2, 0.0, d)


def pick(P, S, cup_xyz, lift=0.15, drive=True, base=None):
    """Top-down pinch of an upright cup (bottom at cup_xyz), then lift."""
    base = table_base(S, cup_xyz[:2]) if base is None else base
    if drive:
        P.drive(base, label="drive table_reach")
        P.hold(0.5, "base settle")
    R = mp.fk(mp.READY, base)[0][:3, :3]        # hand down, as at READY
    p = np.asarray(cup_xyz, float) + [0, 0, CUP_GRASP_Z]
    approach(P, p, R, far=0.14, pre=0.07, t_in=2.0, label="cup")
    close(P)
    P.arm_pose(p + [0, 0, lift], R, 2.5, label="arm lift", cartesian=True)
    return dict(grasp=p, R=R, base=base, lifted=p + [0, 0, lift])


def side_grasp(yaw, roll=1):
    """Horizontal hand along `yaw`, fingers closing horizontally; roll +1: hand x up, -1: down."""
    u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    return u, mp.hand_rot(u, roll * np.array([u[1], -u[0], 0.0]))


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
    base = mp.base_for_reach(p, yaw - delta, bearing, d)
    st = np.zeros(3) if stage is None else np.array([stage[0], stage[1], 0.0])
    if drive:
        P.drive(base + st, label="drive cup_reach")
        P.hold(0.5, "base settle")
    P.arm_pose(p - far * u + st, R, 3.0, label="arm cup far")
    if stage is not None:
        P.drive(base, 2.0, label="drive cup_in")
    P.arm_pose(p - 0.07 * u, R, 2.5, label="arm cup pre", cartesian=True)
    P.arm_pose(p, R, 1.5, label="arm cup in", cartesian=True)
    close(P)
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
    base = mp.base_for_reach(p, yaw - delta, bearing, d)
    if drive:
        P.drive(base, label="drive place_reach")
        P.hold(0.3, "base settle")
    z = P.tcp()[2, 3]
    P.arm_pose(np.array([p[0], p[1], z]), R, 2.5, label="arm forward", cartesian=True)
    P.arm_pose(p, R, 2.5, label="arm lower", cartesian=True)
    P.hold(0.3, "place settle")
    release(P, back=0.12)
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
    release(P, back=0.12)
    return dict(place=p)


# ------------------------------------------------- scripted test scenarios
BASE0 = (0.0, 0.0, 0.0)                 # spawn pose (room centre, facing east)
DOOR_PULL = math.radians(8.0)           # "a few degrees"
DRAWER_PULL = 0.15                      # >= 10 cm required
CUP_START = (1.50, -0.25)               # cup bottom on the table top (xy)
CUP_PLACE = (1.32, -0.25)


class Scenario:
    """A Plan plus check windows: (event index, kind, info); a check is
    evaluated over the event's duration (a hold) and at its end."""

    def __init__(self, name, base0=BASE0):
        self.name, self.P, self.checks, self.info = name, mp.Plan(base0), [], {}
        self.P.hold(1.0, "settle")

    def check(self, kind, dur, **info):
        self.P.hold(dur, f"check {kind}", info)
        self.checks.append((len(self.P.events) - 1, kind, info))


def scenario(name, S=None):
    S = S or scene()
    sc = Scenario(name)
    P = sc.P
    if name == "fridge_handle":
        g = grasp_fridge_handle(P, S)
        sc.check("grasped", 0.5, obj="handle")
        pull_door(P, S, g, DOOR_PULL, 3.0)
        sc.check("door_pulled", 1.0, obj="handle", want=DOOR_PULL)
        # back out 20 cm before the joint move home: from 10 cm the hand swept into the handle
        release(P, back=0.20, dur=2.5)
        P.arm_joints(mp.READY, 3.0, label="arm ready")
        sc.check("door_released", 1.0, want=DOOR_PULL)
        sc.info = dict(grasp=g)
    elif name == "drawer_pull":
        g = grasp_drawer_bar(P, S)
        sc.check("grasped", 0.5, obj="bar")
        pull_drawer(P, S, g, DRAWER_PULL, 3.0)
        sc.check("drawer_pulled", 0.5, obj="bar", want=DRAWER_PULL)
        release(P, back=ka.DRAWER_RELEASE_BACK)
        P.arm_joints(mp.READY, 3.0, label="arm ready")
        sc.check("drawer_released", 1.0, want=DRAWER_PULL)
        sc.info = dict(grasp=g)
    elif name == "cup_pick_place":
        top = S["table"]["top"]
        start = np.array([*CUP_START, top])
        base = table_base(S, start[:2])
        # sideways first: the straight line from the spawn pose clips the table's corner
        P.drive((P.b[0], base[1], base[2]), label="drive table_side")
        g = pick(P, S, start)
        sc.check("cup_lifted", 2.0, obj="cup", z0=float(top))
        pl = place(P, S, g, np.array([*CUP_PLACE, top]))
        P.arm_joints(mp.READY, 3.0, label="arm ready")
        sc.check("cup_placed", 2.0, place=[*CUP_PLACE, top])
        sc.info = dict(grasp=g, place=pl, cup_start=start)
    else:
        raise ValueError(f"unknown scenario {name}")
    P.hold(0.5, "done")
    return sc
