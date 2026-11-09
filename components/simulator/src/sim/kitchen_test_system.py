"""gz-sim Python system: scripted test of the articulated kitchen scene.

Loaded by gz-sim-python-system-loader-system (worlds/kitchen_partnet.sdf) and runs
every 1 ms step. Timeline (sim time):
  0-1 s    settle
  1-11 s   rest, everything closed
  11-15 s  effort opens the fridge top door and the live drawer to their limits
  15-25 s  released, rest open
  25-29 s  effort closes them again
  29-35 s  rest closed
Measures joint positions/velocities per phase, the cup pose (on its fridge
shelf the whole time), every link's peak speed and the contact penetration
reported by the contact sensors. Progress goes to /kitchen_test/status; the
verdict line is "DONE PASS" or "DONE FAIL" and the numbers are written as JSON
to $KITCHEN_TEST_OUT. scripts/kitchen_test.py drives the run.
"""
import json
import math
import os
import xml.etree.ElementTree as ET

from gz.msgs.contacts_pb2 import Contacts
from gz.msgs.stringmsg_pb2 import StringMsg
from gz.sim import Joint, Link, Model, World
from gz.transport import Node

import kitchen_assets as ka

# DOOR2: a second fridge door, if the asset set has one (None for the one-door PartNet fridge)
DOOR, DOOR2, DRAWER = "top_door_hinge", ka.DOOR2, "middle_slide_top"
# Efforts: door 2 N*m (~1.5 rad/s against damping 1.0 + frictionloss 0.5),
# drawer 5 N (~0.3 m/s against damping 10 + frictionloss 2).
EFFORT = {DOOR: 2.0, DRAWER: 5.0}
T_SETTLE, T_OPEN, T_RELEASE, T_CLOSE, T_RELEASE2, T_END = 1.0, 11.0, 15.0, 25.0, 29.0, 35.0
LIMIT_TOL = {DOOR: math.radians(10), DOOR2: math.radians(10), DRAWER: 0.03}
DRIFT_TOL = {DOOR: math.radians(10), DOOR2: math.radians(10), DRAWER: 0.02}
# "Stopped / no jitter": peak |qd| in a rest window (rad/s, m/s)
REST_VEL_TOL = {DOOR: 0.05, DOOR2: 0.05, DRAWER: 0.01}
CUP_MOVE_TOL, MAX_SPEED, MAX_PENETRATION = 0.05, 5.0, 0.01
# Fridge interior around the cup, fridge model frame (scripts/kitchen_assets.py)
FRIDGE_INSIDE = ka.FRIDGE_INSIDE


def _vec(v):
    return [v.x(), v.y(), v.z()]


def _manifest(name):
    for d in os.environ.get("GZ_SIM_RESOURCE_PATH", "").split(":"):
        p = os.path.join(d, name)
        if os.path.isfile(os.path.join(p, "manifest.json")):
            with open(os.path.join(p, "manifest.json")) as f:
                return json.load(f), p
    raise RuntimeError(f"model {name} not found on GZ_SIM_RESOURCE_PATH")


class Window:
    """min/max of q and peak |qd| of one joint over a time window."""

    def __init__(self):
        self.qmin = self.qmax = self.q_end = None
        self.vmax = self.v_end = 0.0

    def add(self, q, v):
        self.qmin = q if self.qmin is None else min(self.qmin, q)
        self.qmax = q if self.qmax is None else max(self.qmax, q)
        self.q_end, self.v_end = q, v
        self.vmax = max(self.vmax, abs(v))

    def as_dict(self):
        return {"qmin": self.qmin, "qmax": self.qmax, "q_end": self.q_end, "v_end": self.v_end,
                "vmax": self.vmax}


class KitchenTest:
    def configure(self, entity, sdf, ecm, event_mgr):
        from gz.math import Pose3d, Quaterniond, Vector3d
        self.Vector3d = Vector3d
        world = World(entity)
        self.out = os.environ.get("KITCHEN_TEST_OUT", "kitchen_test.json")
        fridge = Model(world.model_by_name(ecm, "fridge"))
        counter = Model(world.model_by_name(ecm, "kitchen_counter"))
        self.joints = {DOOR: Joint(fridge.joint_by_name(ecm, DOOR)),
                       DRAWER: Joint(counter.joint_by_name(ecm, DRAWER))}
        if DOOR2:
            self.joints = {DOOR: self.joints[DOOR], DOOR2: Joint(fridge.joint_by_name(ecm, DOOR2)),
                           DRAWER: self.joints[DRAWER]}
        self.limits = {}
        for name, model in ((DOOR, "fridge"), (DOOR2, "fridge"), (DRAWER, "kitchen_counter")):
            if name is None:
                continue
            jm = _manifest(model)[0]["joints"][name]
            self.limits[name] = (jm["lower"], jm["upper"])
        for j in self.joints.values():
            j.enable_position_check(ecm)
            j.enable_velocity_check(ecm)
        # Fridge model frame from its body link (body link frame != model frame)
        _, fdir = _manifest("fridge")
        pose = [float(x) for x in ET.parse(os.path.join(fdir, "model.sdf")).find(
            ".//link[@name='body']/pose").text.split()]
        self.fridge_body_in_model = Pose3d(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5])
        self.fridge_body = Link(fridge.link_by_name(ecm, "body"))
        self.cup = Link(Model(world.model_by_name(ecm, "cup")).canonical_link(ecm))
        self.links = {}
        for m in world.models(ecm):
            model = Model(m)
            for e in model.links(ecm):
                link = Link(e)
                link.enable_velocity_checks(ecm, True)
                self.links[f"{model.name(ecm)}::{link.name(ecm)}"] = link
        self.windows = {}
        self.vmax = (0.0, "", 0.0)
        self.nan = False
        self.cup_p0 = None
        self.cup_dmax = 0.0
        self.cup_outside = []
        self.pen = {"max": 0.0, "where": "", "t": 0.0, "msgs": 0}
        self.t = 0.0
        self.finished = False
        self.node = Node()
        self.pub = self.node.advertise("/kitchen_test/status", StringMsg)
        # The Contact system ignores <topic> and publishes on the scoped sensor name
        for m, l in (("cup", "cup"), ("table_01", "table_01")):
            topic = f"/world/kitchen/model/{m}/link/{l}/sensor/contact/contact"
            if not self.node.subscribe(Contacts, topic, self.on_contacts):
                self.say(f"could not subscribe to {topic}")
        self.say(f"configured: {len(self.links)} links, limits "
                 + ", ".join(f"{k} {v}" for k, v in self.limits.items()))

    def say(self, text):
        print(f"[kitchen_test] {text}", flush=True)
        msg = StringMsg()
        msg.data = text
        self.pub.publish(msg)

    def on_contacts(self, msg):
        self.pen["msgs"] += 1
        for c in msg.contact:
            for d in c.depth:
                if d > self.pen["max"]:
                    self.pen.update(max=d, t=self.t,
                                    where=f"{c.collision1.name} / {c.collision2.name}")

    def phase(self, t):
        if t < T_SETTLE:
            return "settle"
        if t < T_OPEN:
            return "rest_closed"
        if t < T_RELEASE:
            return "opening"
        if t < T_CLOSE:
            return "rest_open"
        if t < T_RELEASE2:
            return "closing"
        return "rest_closed_again"

    def cup_in_fridge_frame(self, ecm):
        model_pose = self.fridge_body.world_pose(ecm) * self.fridge_body_in_model.inverse()
        p = self.cup.world_pose(ecm).pos()
        return _vec(model_pose.rot().rotate_vector_reverse(p - model_pose.pos())), _vec(p)

    def pre_update(self, info, ecm):
        if info.paused or self.finished:
            return
        t = info.sim_time.total_seconds()
        ph = self.phase(t)
        sign = 1.0 if ph == "opening" else -1.0 if ph == "closing" else 0.0
        if sign:
            for name, f in EFFORT.items():
                self.joints[name].set_force(ecm, [sign * f])

    def post_update(self, info, ecm):
        if info.paused or self.finished:
            return
        t = self.t = info.sim_time.total_seconds()
        step = int(round(t * 1000))
        ph = self.phase(t)
        if ph != "settle":
            for name, j in self.joints.items():
                q, v = (j.position(ecm) or [0.0])[0], (j.velocity(ecm) or [0.0])[0]
                self.windows.setdefault(ph, {}).setdefault(name, Window()).add(q, v)
                # rest windows exclude the first 0.5 s after a release
                if ph.startswith("rest") and ph != "rest_closed" and t - self.phase_start(ph) >= 0.5:
                    self.windows.setdefault(ph + "_settled", {}).setdefault(name, Window()).add(q, v)
        if step % 5 == 0:
            for lname, link in self.links.items():
                v, w = link.world_linear_velocity(ecm), link.world_angular_velocity(ecm)
                com = link.world_inertial_pose(ecm)
                if v is None or w is None or com is None:
                    continue
                r = com.pos() - link.world_pose(ecm).pos()
                vc = v + w.cross(r)
                s = vc.length()
                if math.isnan(s):
                    self.nan = True
                elif s > self.vmax[0]:
                    self.vmax = (s, lname, t)
        if ph != "settle" and step % 10 == 0:
            pf, pw = self.cup_in_fridge_frame(ecm)
            if self.cup_p0 is None:
                self.cup_p0 = pw
            self.cup_dmax = max(self.cup_dmax, math.dist(pw, self.cup_p0))
            inside = all(lo <= x <= hi for x, (lo, hi) in zip(pf, FRIDGE_INSIDE))
            if not inside and len(self.cup_outside) < 5:
                self.cup_outside.append([round(t, 3)] + [round(x, 4) for x in pf])
            self.cup_last = (pw, pf)
        if step % 1000 == 0:
            q = {n: round((j.position(ecm) or [0.0])[0], 4) for n, j in self.joints.items()}
            self.say(f"t={t:5.1f}s {ph:<18} door {math.degrees(q[DOOR]):6.1f} deg  drawer "
                     f"{q[DRAWER]*1000:6.1f} mm  cup moved {self.cup_dmax*1000:.1f} mm  "
                     f"vmax {self.vmax[0]:.2f} m/s  pen {self.pen['max']*1000:.2f} mm")
        if t >= T_END:
            self.finished = True
            self.report()

    def phase_start(self, ph):
        return {"rest_open": T_RELEASE, "rest_closed_again": T_RELEASE2}[ph]

    def report(self):
        W = {ph: {n: w.as_dict() for n, w in d.items()} for ph, d in self.windows.items()}
        checks = []

        def check(name, ok, value, limit):
            checks.append({"name": name, "pass": bool(ok), "value": value, "limit": limit})

        def unit(n, x):
            return round(math.degrees(x), 2) if n != DRAWER else round(x * 1000, 1)

        u = {DOOR: "deg", DOOR2: "deg", DRAWER: "mm"}
        for n in (DOOR, DRAWER):
            lo, hi = self.limits[n]
            op, ro = W["opening"][n], W["rest_open_settled"][n]
            err = op["q_end"] - hi
            check(f"{n}: opens to upper limit ({u[n]}, end of push - limit)",
                  abs(err) <= LIMIT_TOL[n], unit(n, err), unit(n, LIMIT_TOL[n]))
            over = op["qmax"] - hi
            check(f"{n}: overshoot past upper limit ({u[n]})",
                  over <= LIMIT_TOL[n], unit(n, over), unit(n, LIMIT_TOL[n]))
            check(f"{n}: stopped at limit, |qd| at end of push", abs(op["v_end"]) <= REST_VEL_TOL[n],
                  round(op["v_end"], 5), REST_VEL_TOL[n])
            drift = ro["qmax"] - ro["qmin"]
            check(f"{n}: rest open drift over {T_CLOSE - T_RELEASE - 0.5:.1f} s ({u[n]})",
                  drift <= DRIFT_TOL[n], unit(n, drift), unit(n, DRIFT_TOL[n]))
            check(f"{n}: rest open peak |qd| (jitter)", ro["vmax"] <= REST_VEL_TOL[n],
                  round(ro["vmax"], 5), REST_VEL_TOL[n])
            cl = W["closing"][n]
            check(f"{n}: closes to lower limit ({u[n]})", abs(cl["q_end"] - lo) <= LIMIT_TOL[n],
                  unit(n, cl["q_end"] - lo), unit(n, LIMIT_TOL[n]))
        for n in (n for n in (DOOR, DOOR2, DRAWER) if n is not None):
            rc = W["rest_closed"][n]
            drift = rc["qmax"] - rc["qmin"]
            check(f"{n}: rest closed drift over {T_OPEN - T_SETTLE:.0f} s ({u[n]})",
                  drift <= DRIFT_TOL[n], unit(n, drift), unit(n, DRIFT_TOL[n]))
            check(f"{n}: rest closed peak |qd| (jitter)", rc["vmax"] <= REST_VEL_TOL[n],
                  round(rc["vmax"], 5), REST_VEL_TOL[n])
            rc2 = W["rest_closed_again_settled"][n]
            drift = rc2["qmax"] - rc2["qmin"]
            check(f"{n}: rest closed again drift over {T_END - T_RELEASE2 - 0.5:.1f} s ({u[n]})",
                  drift <= DRIFT_TOL[n], unit(n, drift), unit(n, DRIFT_TOL[n]))
        check(f"cup: max displacement over {T_END - T_SETTLE:.0f} s, door closed/opening/open/"
              "closing (cm)", self.cup_dmax < CUP_MOVE_TOL, round(self.cup_dmax * 100, 3),
              CUP_MOVE_TOL * 100)
        check("cup: inside the fridge the whole time (samples outside)", not self.cup_outside,
              self.cup_outside, [])
        check("physics: peak body speed (m/s)", self.vmax[0] < MAX_SPEED and not self.nan,
              round(self.vmax[0], 3), MAX_SPEED)
        check("physics: max contact penetration (cm)",
              self.pen["max"] < MAX_PENETRATION and self.pen["msgs"] > 0,
              round(self.pen["max"] * 100, 4), MAX_PENETRATION * 100)
        ok = all(c["pass"] for c in checks)
        data = {"pass": ok, "checks": checks, "windows": W,
                "peak_speed": {"value": self.vmax[0], "link": self.vmax[1], "t": self.vmax[2]},
                "penetration": self.pen,
                "cup_final_world": self.cup_last[0], "cup_final_fridge_frame": self.cup_last[1],
                "efforts": EFFORT, "limits": self.limits,
                "timeline": {"settle": T_SETTLE, "open": T_OPEN, "release": T_RELEASE,
                             "close": T_CLOSE, "release2": T_RELEASE2, "end": T_END}}
        with open(self.out, "w") as f:
            json.dump(data, f, indent=1)
        self.say(f"DONE {'PASS' if ok else 'FAIL'}: {sum(c['pass'] for c in checks)}/{len(checks)} "
                 f"checks, summary {self.out}")


def get_system():
    return KitchenTest()
