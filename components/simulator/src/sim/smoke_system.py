"""gz-sim Python system: smoke test of the robot ($REVOLVE_ROBOT) in the kitchen.

Loaded by worlds/kitchen_partnet_robot.sdf; runs every 1 ms step. The whole run is
one ActionSeq planned at configure time by the robot's smoke_plan (robots/<robot>/scripted.py: rest,
gripper close / open, drives and TCP reaches to the kitchen's handles) and applied step by step.
Checks: rest drift, base pose at each stop, TCP pose at each target (tolerance: the check's `tol`, else
EE_TOL), gripper opening, robot contacts (none expected: nothing is grasped), clearance from
the robot's collision shapes to the furniture's boxes, peak body speed and
contact penetration. Progress on /robot/status; the verdict line is
"DONE PASS" / "DONE FAIL", numbers in $KITCHEN_TEST_OUT (JSON); the actions are
saved next to it (smoke_actions.npz). kitchen_test.py drives the run.
"""
import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
from gz.msgs.contacts_pb2 import Contacts
from gz.msgs.stringmsg_pb2 import StringMsg
from gz.sim import Link, Model, World
from gz.transport import Node

import kitchen_assets as ka
import robots
from robots.base import CONTROL_DT, Controller, rot, rotvec, tf, wrap

WORLD = "kitchen"
# Model poses (x, y, yaw), must match worlds/kitchen_partnet_robot.sdf (scripts/kitchen_assets.py)
POSES = ka.POSES
TABLE_Z0 = 0.002
BASE0 = (0.0, 0.0, 0.0)                    # room centre, facing east
HANDLE_Z = ka.HANDLE_Z                     # fridge handle target height
BASE_TOL, YAW_TOL = 0.03, math.radians(3)
REST_BASE_TOL, REST_ARM_TOL = 0.01, math.radians(1)
EE_TOL, EE_ROT_TOL = 0.02, math.radians(5)
MAX_SPEED, MAX_PEN = 5.0, 0.01
STATUS = "/robot/status"


# ------------------------------------------------------------------ helpers
def rpy_mat(r, p, y):
    return rot("z", y) @ rot("y", p) @ rot("x", r)


def pose_mat(text):
    v = [float(x) for x in (text or "0 0 0 0 0 0").split()]
    return tf(rpy_mat(*v[3:]), v[:3])


def gz_mat(pose):
    q, p = pose.rot(), pose.pos()
    w, x, y, z = q.w(), q.x(), q.y(), q.z()
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return tf(R, (p.x(), p.y(), p.z()))


def model_dir(name):
    for d in os.environ.get("GZ_SIM_RESOURCE_PATH", "").split(":"):
        if os.path.isfile(os.path.join(d, name, "model.sdf")):
            return os.path.join(d, name)
    raise RuntimeError(f"model {name} not found on GZ_SIM_RESOURCE_PATH")


def collisions(name):
    """{link: [(kind, pose in link, dims)]} for box / capsule collisions."""
    out = {}
    root = ET.parse(os.path.join(model_dir(name), "model.sdf")).getroot()
    for link in root.iter("link"):
        for c in link.findall("collision"):
            g = c.find("geometry")
            T = pose_mat(c.findtext("pose"))
            if g.find("box") is not None:
                dims = [float(x) for x in g.find("box/size").text.split()]
                out.setdefault(link.get("name"), []).append(("box", T, dims))
            elif g.find("capsule") is not None:
                dims = [float(g.findtext("capsule/radius")), float(g.findtext("capsule/length"))]
                out.setdefault(link.get("name"), []).append(("capsule", T, dims))
    return out


def contact_links(name):
    """The links of a model with a contact sensor, in model order (a robot's: every collision link)."""
    root = ET.parse(os.path.join(model_dir(name), "model.sdf")).getroot()
    return [ln.get("name") for ln in root.iter("link") if ln.find("sensor[@type='contact']") is not None]


def sample_points(kind, T, dims):
    """Points (link frame) and radii approximating a collision shape."""
    if kind == "capsule":
        r, L = dims
        z = np.linspace(-L / 2, L / 2, 7)
        P = np.column_stack([0 * z, 0 * z, z])
        rad = np.full(len(z), r)
    else:
        g = [np.linspace(-d / 2, d / 2, 3) for d in dims]
        P = np.array(np.meshgrid(*g)).reshape(3, -1).T
        rad = np.zeros(len(P))
    return (T[:3, :3] @ P.T).T + T[:3, 3], rad


def model_tf(name):
    x, y, yaw = POSES[name]
    return tf(rot("z", yaw), (x, y, TABLE_Z0 if name == "table_01" else 0.0))


def to_world(name, p):
    return (model_tf(name) @ np.array([*p, 1.0]))[:3]


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


# ----------------------------------------------------------------- the plan
def targets():
    """EE targets and base poses from the models' manifests (model frame)."""
    man = {}
    for n in POSES:
        with open(os.path.join(model_dir(n), "manifest.json")) as f:
            man[n] = json.load(f)
    T = {}
    # Fridge top door handle: front face = handle centre + half depth along the
    # fridge's front normal (model +x).
    h = next(x for x in man["fridge"]["handles"] if x["name"] == "top_door_handle")
    c = np.array(h["center_model_frame"])
    n_f = model_tf("fridge")[:3, 0]
    front = to_world("fridge", (c[0] + h["size"][0] / 2, c[1], HANDLE_Z))
    T["fridge"] = dict(handle=to_world("fridge", c).tolist(), front=front, normal=n_f,
                       face=to_world("fridge", (man["fridge"]["aabb_model_frame"][1][0], c[1], 0)))
    # Drawer bar
    h = next(x for x in man["kitchen_counter"]["handles"] if x["name"] == "drawer2_top_handle_bar")
    c = np.array(h["center_model_frame"])
    n_c = model_tf("kitchen_counter")[:3, 0]
    front = to_world("kitchen_counter", (c[0] + h["size"][0] / 2, c[1], c[2]))
    T["counter"] = dict(handle=to_world("kitchen_counter", c).tolist(), front=front, normal=n_c,
                        face=to_world("kitchen_counter",
                                      (man["kitchen_counter"]["aabb_model_frame"][1][0], c[1], 0)))
    # Table: north edge, top height
    lo, hi = man["table_01"]["aabb_model_frame"]
    T["table"] = dict(top=hi[2] + TABLE_Z0, face=to_world("table_01", (0, hi[1], 0)),
                      normal=np.array([0.0, 1.0, 0.0]), centre=to_world("table_01", (0, 0, 0)))
    return T


# ------------------------------------------------------------------ system
class Smoke:
    def configure(self, entity, sdf, ecm, event_mgr):
        world = World(entity)
        self.robot = robots.current()
        self.model = Model(world.model_by_name(ecm, self.robot.model))
        self.ctl = Controller(self.robot, self.model, ecm)
        self.ctl.reset(ecm, BASE0)
        self.plan, self.checks, self.T = self.robot.scripted.smoke_plan(self.robot, targets())
        self.seq = self.plan.actions()
        self.out = os.environ.get("KITCHEN_TEST_OUT", "robot_smoke.json")
        self.seq.save(os.path.join(os.path.dirname(os.path.abspath(self.out)), "smoke_actions.npz"))
        ev = self.plan.events
        self.check_at = {}   # control step at the end of each check hold -> check
        for i, kind, info in self.checks:
            end = ev[i + 1][0] if i + 1 < len(ev) else self.plan.t
            self.check_at[int(round(end / CONTROL_DT)) - 1] = (kind, info)
        self.ev_steps = [(int(round(t / CONTROL_DT)), lab) for t, lab, _ in ev]
        self.rest = (int(round(ev[1][0] / CONTROL_DT)), int(round(ev[2][0] / CONTROL_DT)))

        # collision shapes: robot (sampled points) and furniture (boxes)
        rob = collisions(self.robot.model)
        self.robot_links = contact_links(self.robot.model)
        self.rob = []
        for ln in self.robot_links:
            for kind, Tc, dims in rob.get(ln, []):
                self.rob.append((Link(self.model.link_by_name(ecm, ln)), ln,
                                 *sample_points(kind, Tc, dims)))
        self.furn = []          # (model, link name, Link, box poses in link [n,4,4], half sizes [n,3])
        for mn in POSES:
            m = Model(world.model_by_name(ecm, mn))
            for ln, cols in collisions(mn).items():
                boxes = [(Tc, np.array(d) / 2) for k, Tc, d in cols if k == "box"]
                if boxes:
                    self.furn.append((mn, ln, Link(m.link_by_name(ecm, ln)),
                                      np.array([b[0] for b in boxes]), np.array([b[1] for b in boxes])))
        self.links = {}
        for e in world.models(ecm):
            md = Model(e)
            for le in md.links(ecm):
                link = Link(le)
                link.enable_velocity_checks(ecm, True)
                self.links[f"{md.name(ecm)}::{link.name(ecm)}"] = link
        self.hand = self.ctl.hand
        self.results, self.robot_contacts = [], []
        self.vmax, self.nan = (0.0, "", 0.0), False
        self.pen = {"max": 0.0, "where": "", "t": 0.0, "msgs": 0}
        self.clear_min = {}     # phase kind -> (d, robot link, furniture, t)
        self.rest0 = None
        self.rest_dev = [0.0, 0.0]
        self.q_err_max = 0.0
        self.finished, self.t, self.ev_i = False, 0.0, 0
        self.node = Node()
        self.pub = self.node.advertise(STATUS, StringMsg)
        topics = [f"/world/{WORLD}/model/{self.robot.model}/link/{ln}/sensor/contact/contact"
                  for ln in self.robot_links]
        topics += [f"/world/{WORLD}/model/{m}/link/{m}/sensor/contact/contact"
                   for m in ("cup", "table_01")]
        for tp in topics:
            if not self.node.subscribe(Contacts, tp, self.on_contacts):
                self.say(f"could not subscribe to {tp}")
        self.say(f"planned {len(self.seq)} steps ({self.plan.t:.1f} s), {len(self.checks)} checks, "
                 f"worst IK {self.plan.ik_worst[0]*1000:.2f} mm / "
                 f"{math.degrees(self.plan.ik_worst[1]):.2f} deg; handle "
                 f"{np.round(self.T['fridge']['handle'], 3)}, bar {np.round(self.T['counter']['handle'], 3)}")

    def say(self, text):
        print(f"[robot] {text}", flush=True)
        msg = StringMsg()
        msg.data = text
        self.pub.publish(msg)

    def on_contacts(self, msg):
        self.pen["msgs"] += 1
        for c in msg.contact:
            names = (c.collision1.name, c.collision2.name)
            depth = max(c.depth) if len(c.depth) else 0.0
            if depth > self.pen["max"]:
                self.pen.update(max=depth, t=self.t, where=" / ".join(names))
            if any(self.robot.model in n for n in names) and len(self.robot_contacts) < 50:
                self.robot_contacts.append([round(self.t, 3), self.phase(), *names, depth])

    def phase(self):
        return self.ev_steps[self.ev_i][1] if self.ev_steps else ""

    def pre_update(self, info, ecm):
        if info.paused:
            return
        k = int(round(info.sim_time.total_seconds() / CONTROL_DT))
        self.ctl.apply(ecm, self.seq[k])

    # -------------------------------------------------------------- metrics
    def tcp(self, ecm):
        return gz_mat(self.hand.world_pose(ecm)) @ tf(p=(0, 0, self.robot.tcp_z))

    def clearance(self, ecm, per_link=False):
        """Min distance robot shapes -> furniture boxes, per furniture model:
        {model: (d, robot link, furniture link)}; per_link: {model: {robot link: d}}."""
        P, rad, who = [], [], []
        for link, ln, pts, r in self.rob:
            Tl = gz_mat(link.world_pose(ecm))
            P.append((Tl[:3, :3] @ pts.T).T + Tl[:3, 3])
            rad.append(r)
            who += [ln] * len(r)
        P, rad, who = np.vstack(P), np.concatenate(rad), np.array(who)
        best, links = {}, {}
        for mn, ln, link, Tc, h in self.furn:
            Tb = gz_mat(link.world_pose(ecm)) @ Tc                       # [n,4,4]
            loc = np.einsum("nij,pi->npj", Tb[:, :3, :3], P) - np.einsum(
                "nij,ni->nj", Tb[:, :3, :3], Tb[:, :3, 3])[:, None, :]    # [n,p,3]
            q = np.abs(loc) - h[:, None, :]
            d = (np.linalg.norm(np.maximum(q, 0), axis=2) + np.minimum(q.max(axis=2), 0)).min(0) - rad
            i = int(np.argmin(d))
            if mn not in best or d[i] < best[mn][0]:
                best[mn] = (float(d[i]), str(who[i]), ln)
            if per_link:
                lk = links.setdefault(mn, {})
                for rl in self.robot_links:
                    m = float(d[who == rl].min()) if np.any(who == rl) else math.inf
                    lk[rl] = round(min(lk.get(rl, math.inf), m), 4)
        return (best, links) if per_link else best

    def post_update(self, info, ecm):
        if info.paused or self.finished:
            return
        t = self.t = info.sim_time.total_seconds()
        k = int(round(t / CONTROL_DT)) - 1          # action applied this step
        while self.ev_i + 1 < len(self.ev_steps) and k >= self.ev_steps[self.ev_i + 1][0]:
            self.ev_i += 1
            self.say(f"t={t:6.2f}s {self.ev_steps[self.ev_i][1]}")
        st = self.ctl.state(ecm)
        a = self.seq[k]
        if self.ev_i >= 1:
            self.q_err_max = max(self.q_err_max, float(np.max(np.abs(st["arm"] - a.arm))))
        if self.rest[0] <= k < self.rest[1]:
            if self.rest0 is None:
                self.rest0 = (st["base"].copy(), st["arm"].copy())
            self.rest_dev[0] = max(self.rest_dev[0], float(np.linalg.norm(st["base"][:2] - self.rest0[0][:2])))
            self.rest_dev[1] = max(self.rest_dev[1], float(np.max(np.abs(st["arm"] - self.rest0[1]))),
                                   abs(wrap(st["base"][2] - self.rest0[0][2])))
        if k % 5 == 0:
            for lname, link in self.links.items():
                v, w = link.world_linear_velocity(ecm), link.world_angular_velocity(ecm)
                com = link.world_inertial_pose(ecm)
                if v is None or w is None or com is None:
                    continue
                s = (v + w.cross(com.pos() - link.world_pose(ecm).pos())).length()
                if math.isnan(s):
                    self.nan = True
                elif s > self.vmax[0]:
                    self.vmax = (s, lname, t)
        if k % 100 == 0 and self.ev_i >= 1:
            kind = self.phase().split()[0]
            for mn, (d, rl, fl) in self.clearance(ecm).items():
                key = f"{kind}:{mn}"
                if key not in self.clear_min or d < self.clear_min[key][0]:
                    self.clear_min[key] = (round(d, 4), rl, fl, round(t, 2))
        if k in self.check_at:
            self.evaluate(*self.check_at[k], st, a, ecm, t)
        if k >= len(self.seq) - 1:
            self.finished = True
            self.report()

    def evaluate(self, kind, info, st, a, ecm, t):
        r = {"kind": kind, "name": info.get("name"), "t": round(t, 3)}
        if kind == "base":
            Tb = gz_mat(self.ctl.base_link.world_pose(ecm))
            goal = np.array(info["goal"])
            r.update(goal=goal.round(4).tolist(), pose=[Tb[0, 3], Tb[1, 3], yaw_of(Tb)],
                     err_xy=float(np.linalg.norm(Tb[:2, 3] - goal[:2])),
                     err_yaw=abs(wrap(yaw_of(Tb) - goal[2])))
            r["pass"] = r["err_xy"] <= BASE_TOL and r["err_yaw"] <= YAW_TOL
            msg = f"err {r['err_xy']*100:.2f} cm / {math.degrees(r['err_yaw']):.2f} deg"
        elif kind == "ee":
            Tt = self.tcp(ecm)
            p, R = np.array(info["p"]), np.array(info["R"])
            r.update(target=p.round(4).tolist(), tcp=Tt[:3, 3].round(4).tolist(),
                     err_p=float(np.linalg.norm(Tt[:3, 3] - p)),
                     err_rot=float(np.linalg.norm(rotvec(R @ Tt[:3, :3].T))),
                     q=st["arm"].round(3).tolist())
            r["clearance"], r["clearance_per_link"] = self.clearance(ecm, per_link=True)
            r["tol"] = info.get("tol", EE_TOL)
            r["pass"] = r["err_p"] <= r["tol"] and r["err_rot"] <= EE_ROT_TOL
            if info["name"].startswith("fridge"):
                r["pass"] &= ka.HANDLE_BAND[0] <= Tt[2, 3] <= ka.HANDLE_BAND[1]
            cl = ", ".join(f"{m} {d*100:.1f} cm ({rl}/{fl})" for m, (d, rl, fl) in r["clearance"].items())
            msg = f"TCP err {r['err_p']*1000:.1f} mm / {math.degrees(r['err_rot']):.2f} deg, " \
                  f"z {Tt[2, 3]:.3f}; clearance {cl}"
        else:
            f = st["fingers"]
            r.update(fingers_mm=(f * 1000).round(2).tolist(), want_mm=info["want"] * 1000)
            r["pass"] = bool(np.all(np.abs(f - info["want"]) <= 0.003))
            msg = f"fingers {f[0]*1000:.1f} / {f[1]*1000:.1f} mm (want {info['want']*1000:.0f})"
        self.results.append(r)
        self.say(f"t={t:6.2f}s CHECK {kind} {info.get('name')}: {'PASS' if r['pass'] else 'FAIL'} {msg}")

    def report(self):
        checks = []

        def check(name, ok, value, limit):
            checks.append({"name": name, "pass": bool(ok), "value": value, "limit": limit})

        check("rest 10 s: base drift (cm)", self.rest_dev[0] <= REST_BASE_TOL,
              round(self.rest_dev[0] * 100, 4), REST_BASE_TOL * 100)
        check("rest 10 s: arm joint / base yaw drift (deg)", self.rest_dev[1] <= REST_ARM_TOL,
              round(math.degrees(self.rest_dev[1]), 4), round(math.degrees(REST_ARM_TOL), 2))
        for r in self.results:
            if r["kind"] == "base":
                check(f"base at {r['name']} (cm, deg)", r["pass"],
                      [round(r["err_xy"] * 100, 2), round(math.degrees(r["err_yaw"]), 2)],
                      [BASE_TOL * 100, round(math.degrees(YAW_TOL), 1)])
            elif r["kind"] == "ee":
                check(f"TCP at {r['name']} (mm, deg, z m)", r["pass"],
                      [round(r["err_p"] * 1000, 1), round(math.degrees(r["err_rot"]), 2), r["tcp"][2]],
                      [r["tol"] * 1000, round(math.degrees(EE_ROT_TOL), 1),
                       "%.2f-%.2f" % ka.HANDLE_BAND if r["name"].startswith("fridge") else "-"])
            else:
                check(f"gripper {r['name']} (mm per finger)", r["pass"], r["fingers_mm"], r["want_mm"])
        drive_contacts = [c for c in self.robot_contacts if c[1].startswith("drive")]
        check("robot contacts while driving", not drive_contacts, drive_contacts[:5], [])
        check("robot contacts at any time (nothing is grasped)", not self.robot_contacts,
              self.robot_contacts[:5], [])
        check("physics: peak body speed (m/s)", self.vmax[0] < MAX_SPEED and not self.nan,
              [round(self.vmax[0], 3), self.vmax[1], round(self.vmax[2], 2)], MAX_SPEED)
        check("physics: max contact penetration (cm)",
              self.pen["max"] < MAX_PEN and self.pen["msgs"] > 0,
              round(self.pen["max"] * 100, 4), MAX_PEN * 100)
        n_res = sum(1 for r in self.results)
        check("all planned checks ran", n_res == len(self.checks), n_res, len(self.checks))
        check("controller: non-finite joint commands (zeroed)", not self.ctl.bad,
              self.ctl.bad[:3], [])
        ok = all(c["pass"] for c in checks)
        data = {"pass": ok, "checks": checks, "results": self.results,
                "clearance_min_by_phase_cm": {k: [round(v[0] * 100, 2), *v[1:]]
                                              for k, v in sorted(self.clear_min.items())},
                "peak_speed": {"value": self.vmax[0], "link": self.vmax[1], "t": self.vmax[2]},
                "penetration": self.pen, "robot_contacts": self.robot_contacts,
                "arm_tracking_err_max_deg": math.degrees(self.q_err_max),
                "plan": {"duration_s": self.plan.t, "steps": len(self.seq),
                         "ik_worst_mm_deg": [self.plan.ik_worst[0] * 1000,
                                             math.degrees(self.plan.ik_worst[1])],
                         "events": [(round(t, 2), lab) for t, lab, _ in self.plan.events]},
                "mount_z": float(self.robot.mount[2]), "mount_x": float(self.robot.mount[0])}
        with open(self.out, "w") as f:
            json.dump(data, f, indent=1, default=float)
        self.say(f"DONE {'PASS' if ok else 'FAIL'}: {sum(c['pass'] for c in checks)}/{len(checks)} "
                 f"checks, summary {self.out}")


def get_system():
    return Smoke()
