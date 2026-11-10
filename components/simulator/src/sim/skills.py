"""Robot-independent building blocks of the scripted kitchen runs (no Gazebo needed): the scene geometry
from the models' manifest.json files (scene(), with the model poses of the kitchen worlds), Scenario (a
Plan plus check windows and spans), the generic moves (approach / close / release, TCP arcs about a door
hinge, straight drawer pulls, min-jerk drives) and the dispatch to the robot's own scripted runs
(robots/<robot>/scripted.py: SCENARIOS for the skill tests, EPISODES for the scripted episodes).

Every move appends to a robots.base.Plan of the robot (P.robot); all contact phases are effort-controlled
(Controller: P + gravity comp + SDF damping, gripper force-capped); nothing uses velocity commands.
"""
import json
import math
import os

import numpy as np

import kitchen_assets as ka
import robots
from robots.base import Plan, tf, rot, wrap

# Model poses (x, y, yaw) of the asset set ($KITCHEN_ASSETS); must match the world files
POSES = ka.POSES
TABLE_Z0 = 0.002
CUP_H, CUP_R = 0.089, 0.03
BASE0 = (0.0, 0.0, 0.0)                 # spawn pose (room centre, facing east)
V_DRIVE, W_DRIVE = ka.V_DRIVE, 1.0             # base peak speeds of the episodes' drives [m/s, rad/s]
BUDGET = {"fridge": 90.0, "drawer": 45.0}       # scripted episodes: sim time by which the last check ends


def model_dir(name):
    for d in os.environ.get("GZ_SIM_RESOURCE_PATH", "").split(":") + [
            os.path.join(os.path.dirname(__file__), "..", "assets", "generated")]:
        if d and os.path.isfile(os.path.join(d, name, "manifest.json")):
            return os.path.join(d, name)
    raise RuntimeError(f"model {name} not found on GZ_SIM_RESOURCE_PATH")


def model_tf(name):
    x, y, yaw = POSES[name]
    return tf(rot("z", yaw), (x, y, TABLE_Z0 if name == "table_01" else 0.0))


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


# ------------------------------------------------------------------ moves
def approach(P, p, R, far=0.15, pre=0.06, t_far=4.0, t_in=2.5, label="target"):
    """Joint-space move to `far` m out along the hand's -z, Cartesian to `pre`, then in to p."""
    z = R[:, 2]
    P.arm_pose(p - far * z, R, t_far, label=f"arm {label} far")
    P.arm_pose(p - pre * z, R, 1.5, label=f"arm {label} pre", cartesian=True)
    P.arm_pose(p, R, t_in, label=f"arm {label} in", cartesian=True)


def close(P, dur=1.5, force=None, settle=0.5, label="grip close"):
    """Close the gripper (force: the squeeze, default the robot's) and let the grasp settle."""
    P.gripper(P.robot.gripper.closed_m, dur, force=force, label=label)
    P.hold(settle, "grip settle")


def release(P, back=0.10, dur=1.5, label="release"):
    """Open, then back the TCP out along the hand's -z."""
    P.gripper(P.robot.gripper.open_m, 1.5, label=label)
    T = P.tcp()
    P.arm_pose(T[:3, 3] - back * T[:3, 2], T[:3, :3], dur, label="arm retreat", cartesian=True)


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


def pull_drawer(P, S, grasp, dist, dur, label="pull drawer"):
    """Pull the grasped bar `dist` m out along the drawer axis."""
    p = grasp["grasp"] + dist * np.asarray(S["drawer"]["axis"])
    P.arm_pose(p, grasp["R"], dur, label=label, cartesian=True, info=dict(dist=dist))
    return p


def drive(P, goal, label="drive", v=None, w=None):
    """Min-jerk drive with the episodes' peak speeds (peak = 1.875 x mean)."""
    v, w = V_DRIVE if v is None else v, W_DRIVE if w is None else w
    goal = np.array(goal, float)
    d = goal - P.b
    d[2] = wrap(d[2])
    dur = max(1.5, 1.875 * max(np.linalg.norm(d[:2]) / v, abs(d[2]) / w))
    P.drive(goal, dur, label=label)


# ------------------------------------------------- scripted runs
class Scenario:
    """A Plan plus check windows: (event index, kind, info); a check is
    evaluated over the event's duration (a hold) and at its end."""

    def __init__(self, name, robot=None, base0=BASE0):
        self.name, self.checks, self.info = name, [], {}
        self.P = Plan(robot or robots.current(), base0)
        self.P.hold(1.0, "settle")

    def check(self, kind, dur, **info):
        self.P.hold(dur, f"check {kind}", info)
        self.checks.append((len(self.P.events) - 1, kind, info))


def span(sc, kind, start, end, **info):
    """Check over the actions from event `start` to the end of event `end` (labels)."""
    sc.spans = getattr(sc, "spans", [])
    sc.spans.append(dict(kind=kind, start=start, end=end, info=info))


def _scripted(table, name):
    rb = robots.current()
    runs = getattr(rb.scripted, table)
    if name not in runs:
        raise SystemExit(f"robot {rb.name} has no scripted {table[:-1].lower()} {name!r} (has {sorted(runs)})")
    return runs[name]


def episode(name, S=None):
    """The robot's scripted episode `name` (robots/<robot>/scripted.py EPISODES)."""
    return _scripted("EPISODES", name)(S or scene())


def scenario(name, S=None):
    """The robot's skill-test scenario `name` (robots/<robot>/scripted.py SCENARIOS)."""
    return _scripted("SCENARIOS", name)(S or scene())
