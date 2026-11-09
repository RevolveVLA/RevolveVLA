"""gz-sim Python system: mobile Panda contact-skill tests in the kitchen.

Loaded by worlds/kitchen_partnet_skills.sdf; runs every 1 ms step. $SKILL picks the
scenario (plans in scripts/mobile_panda_skills.py):
  fridge_handle   grasp the top door handle, pull the door 8 deg on its arc, release
  drawer_pull     grasp the drawer bar (hand rolled 90 deg), pull it 15 cm, release
  cup_pick_place  cup on the table: pick, lift 15 cm, hold 2 s, place 18 cm west, release
  reset_test      cup_pick_place, saving the scene mid-carry and restoring it 1.5 s later
                  (scripts/scene_state.py); checks the restored state and that the rerun
                  from it reaches the same state as the original
  replay          replay $REPLAY_DIR/actions.npz (a recorded run) from the same initial
                  state and compare with $REPLAY_DIR/final_state.npz
Every run saves its applied actions (actions.npz, one Action per step), the
initial and final scene state (.npz) next to $KITCHEN_TEST_OUT, and checks
physics sanity (peak body speed, contact penetration, robot contacts other
than the fingers on the grasped object). Verdict "DONE PASS/FAIL" on
/mobile_panda_skill/status; scripts/kitchen_test.py drives the run.
"""
import json
import math
import os
import sys
import time

import numpy as np
from gz.msgs.contacts_pb2 import Contacts
from gz.msgs.stringmsg_pb2 import StringMsg
from gz.sim import Link, Model, World
from gz.transport import Node

import mobile_panda as mp
import mobile_panda_skills as sk
import mobile_panda_smoke_system as smoke
import panda_plan as pp
import scene_state as ss

# Contact callbacks run on a transport thread that needs the GIL, which the sim
# thread holds for every step. With the default 5 ms switch interval they get ~200
# messages/s, so a sim near real time (~1100/s) builds a backlog that outruns the
# end-of-run wait and finger contacts go missing. A 0.1 ms interval keeps up.
sys.setswitchinterval(1e-4)

WORLD = "kitchen"
SKILL = os.environ.get("SKILL", "drawer_pull")
MAX_SPEED, MAX_PEN = 5.0, 0.01
SLIP_TOL = 0.01                  # grasped object vs TCP drift [m]
RESTORE_TOL = (0.001, math.radians(0.1))
RESTORE_VEL_TOL = (0.005, 0.05)  # "near equal" [m/s, rad/s]
REPLAY_TOL = 0.01
OBJ = {"handle": ("fridge", "top_door", "top_door_handle"),
       "bar": ("kitchen_counter", "drawer2_top", "drawer2_top_handle_bar"),
       "cup": ("cup", "cup", "cup_collision")}
FINGERS = ("panda_leftfinger", "panda_rightfinger")
# Opt-in fast mode (`--step <s>`): the world's physics step is
# KITCHEN_STEP, an integer multiple R of the 1 ms action grid. Step n then only takes
# the values R*m - 1, so the action sequence is resampled by sim time and every
# step-index event is moved to the first step at or after it (grid()).
STEP = float(os.environ.get("KITCHEN_STEP") or mp.CONTROL_DT)
R = int(round(STEP / mp.CONTROL_DT))
assert R >= 1 and abs(R * mp.CONTROL_DT - STEP) < 1e-9, f"KITCHEN_STEP {STEP} is not a multiple of 1 ms"
PROFILE = bool(os.environ.get("KITCHEN_PROFILE"))


def grid(k):
    """First physics step index >= k (identity for the default 1 ms step)."""
    return R * -(-(k + 1) // R) - 1


class SkillSystem:
    clearance = smoke.Smoke.clearance

    def configure(self, entity, sdf, ecm, event_mgr):
        world = World(entity)
        self.model = Model(world.model_by_name(ecm, mp.MODEL))
        self.ctl = mp.Controller(self.model, ecm)
        if os.environ.get("KITCHEN_FINGER_LAW", "dt") == "dt":
            self.ctl.dt = STEP
        self.ctl.reset(ecm, sk.BASE0)
        self.state = ss.SceneState(world, ecm)
        self.out = os.environ.get("KITCHEN_TEST_OUT", "skill.json")
        self.dir = os.path.dirname(os.path.abspath(self.out))
        self.S = sk.scene()
        self.ref = None
        skill = SKILL
        if SKILL == "replay":
            rd = os.environ["REPLAY_DIR"]
            self.ref = ss.load(os.path.join(rd, "final_state.npz"))
            skill = str(self.ref["skill"])
        self.skill = skill
        base_skill = "cup_pick_place" if skill == "reset_test" else skill
        self.sc = self.make_scenario(base_skill)
        self.seq = self.sc.P.actions()
        if SKILL == "replay":
            self.seq = mp.ActionSeq.load(os.path.join(rd, "actions.npz"))
        n = len(self.seq)
        # reset_test: save mid-carry, restore 1.5 s later, rerun from there
        self.k_save = self.k_restore = None
        if skill == "reset_test":
            ev = {lab: t for t, lab, _ in self.sc.P.events}
            self.k_save = int(round((ev["arm carry"] + 1.2) / mp.CONTROL_DT))
            self.k_restore = self.k_save + 1500
        self.n_total = n + (self.k_restore - self.k_save if self.k_save else 0)
        self.snaps = {}
        self.applied = np.zeros(self.n_total, dtype=np.int64)
        self.cup_start = None
        if base_skill == "cup_pick_place":
            self.cup_start = np.eye(4)
            self.cup_start[:3, 3] = self.sc.info["cup_start"] + [0, 0, 0.0005]

        # windows: action index range -> check
        ev = self.sc.P.events
        self.windows = []
        for i, kind, info in self.sc.checks:
            t0 = ev[i][0]
            t1 = ev[i + 1][0] if i + 1 < len(ev) else self.sc.P.t
            self.windows.append(dict(k0=grid(int(round(t0 / mp.CONTROL_DT))),
                                     k1=grid(int(round(t1 / mp.CONTROL_DT)) - 1), kind=kind, info=info))
        self.ev_steps = [(int(round(t / mp.CONTROL_DT)), lab) for t, lab, _ in ev]

        # collision shapes for the clearance report (as in the smoke test)
        rob = smoke.collisions(mp.MODEL)
        self.rob = []
        for ln in smoke.ROBOT_LINKS:
            for kind, Tc, dims in rob.get(ln, []):
                self.rob.append((Link(self.model.link_by_name(ecm, ln)), ln,
                                 *smoke.sample_points(kind, Tc, dims)))
        self.furn = []
        for mn in sk.POSES:
            m = Model(world.model_by_name(ecm, mn))
            for ln, cols in smoke.collisions(mn).items():
                boxes = [(Tc, np.array(d) / 2) for k, Tc, d in cols if k == "box"]
                if boxes:
                    self.furn.append((mn, ln, Link(m.link_by_name(ecm, ln)),
                                      np.array([b[0] for b in boxes]), np.array([b[1] for b in boxes])))
        self.obj_links = {}
        for key, (mn, ln, _) in OBJ.items():
            self.obj_links[key] = Link(Model(world.model_by_name(ecm, mn)).link_by_name(ecm, ln))
        self.cup = self.obj_links["cup"]
        self.jidx = {}
        self.results, self.robot_contacts, self.other_contacts = [], [], []
        self.obj_contact = {}           # (object key, finger) -> last t in contact
        self.contact_first = {}
        self.last_stamp, self.ending = 0.0, None
        self.vmax, self.nan = (0.0, "", 0.0), False
        self.pen = {"max": 0.0, "where": "", "t": 0.0, "msgs": 0}
        self.win, self.win_stats = None, {}
        self.grasp_rel = {}
        self.extra = {}
        self.finished, self.t, self.ev_i, self.a = False, 0.0, 0, 0
        self.prof = dict(steps=0, pre_s=0.0, post_s=0.0, cb_s=0.0, wall_first=None, sim_first=None)
        self.node = Node()
        self.pub = self.node.advertise("/mobile_panda_skill/status", StringMsg)
        topics = [f"/world/{WORLD}/model/{mp.MODEL}/link/{ln}/sensor/contact/contact"
                  for ln in smoke.ROBOT_LINKS]
        topics += [f"/world/{WORLD}/model/{m}/link/{m}/sensor/contact/contact"
                   for m in ("cup", "table_01")]
        for tp in topics:
            if not self.node.subscribe(Contacts, tp, self.on_contacts):
                self.say(f"could not subscribe to {tp}")
        self.say(f"skill {SKILL} ({skill}): {n} actions ({n * mp.CONTROL_DT:.1f} s), "
                 f"{len(self.windows)} checks, worst IK {self.sc.P.ik_worst[0]*1000:.2f} mm / "
                 f"{math.degrees(self.sc.P.ik_worst[1]):.2f} deg"
                 + (f"; save at step {self.k_save}, restore at {self.k_restore}" if self.k_save else "")
                 + (f"; physics step {STEP * 1000:g} ms" if R > 1 else ""))

    def make_scenario(self, name):
        return sk.scenario(name, self.S)

    # ------------------------------------------------------------ plumbing
    def say(self, text):
        print(f"[skill] {text}", flush=True)
        msg = StringMsg()
        msg.data = text
        self.pub.publish(msg)

    def on_contacts(self, msg):
        if PROFILE:
            c0 = time.perf_counter()
            self._on_contacts(msg)
            self.prof["cb_s"] += time.perf_counter() - c0
        else:
            self._on_contacts(msg)

    def _on_contacts(self, msg):
        # Runs on a transport thread and can lag the sim by seconds: use the
        # message's sim-time stamp, not the latest step time.
        self.pen["msgs"] += 1
        st = msg.header.stamp
        t = st.sec + st.nsec * 1e-9
        self.last_stamp = max(self.last_stamp, t)
        for c in msg.contact:
            names = (c.collision1.name, c.collision2.name)
            depth = max(c.depth) if len(c.depth) else 0.0
            if depth > self.pen["max"]:
                self.pen.update(max=depth, t=t, where=" / ".join(names))
            rob = [n for n in names if n.startswith(mp.MODEL + "::")]
            if not rob:
                continue
            other = [n for n in names if not n.startswith(mp.MODEL + "::")]
            link = rob[0].split("::")[1]
            hit = None
            for key, (mn, ln, cn) in OBJ.items():
                if other and other[0].startswith(f"{mn}::{ln}::{cn}"):
                    hit = key
            if hit and link in FINGERS:
                self.obj_contact.setdefault((hit, link), []).append(t)
                self.contact_first.setdefault(f"{hit}/{link}", round(t, 3))
            elif len(self.other_contacts) < 50:
                self.other_contacts.append([round(t, 3), self.phase_at(t), *names, depth])

    def phase_at(self, t):
        a = self.action_index(int(round(t / mp.CONTROL_DT)) - 1)
        lab = ""
        for k, label in self.ev_steps:
            if k <= a:
                lab = label
        return lab

    def phase(self):
        return self.ev_steps[self.ev_i][1] if self.ev_steps else ""

    def action_index(self, n):
        """Action applied at physics step n (0-based)."""
        if self.k_restore is not None and n > self.k_restore:
            return n - (self.k_restore - self.k_save)
        return n

    def pre_update(self, info, ecm):
        if PROFILE and not info.paused:
            c0 = time.perf_counter()
            self._pre_update(info, ecm)
            p = self.prof
            p["pre_s"] += time.perf_counter() - c0
            p["steps"] += 1
            if p["wall_first"] is None:
                p["wall_first"], p["sim_first"] = time.time(), info.sim_time.total_seconds()
            if "wall5" not in p and info.sim_time.total_seconds() >= 5.0:   # past the slow start
                p["wall5"], p["sim5"] = time.time(), info.sim_time.total_seconds()
        else:
            self._pre_update(info, ecm)

    def _pre_update(self, info, ecm):
        if info.paused:
            return
        n = int(round(info.sim_time.total_seconds() / mp.CONTROL_DT)) - 1
        self.state.after_restore(ecm)
        if n == grid(0) and self.cup_start is not None:      # cup on the table for this test
            self.state.set_model_pose(ecm, "cup", self.cup_start)
        meas = None
        if self.k_restore is not None and n == self.k_restore + 1:
            self.state.restore(ecm, self.snaps["save"])
            # the ECM still holds the pre-restore state this step: control from the snapshot
            meas = robot_state(self.snaps["save"])
            self.say(f"t={info.sim_time.total_seconds():6.2f}s RESTORE state of step {self.k_save}")
        a = min(self.action_index(n), len(self.seq) - 1)    # after the end: hold the last action
        if n < self.n_total:
            self.applied[n] = a
        self.ctl.apply(ecm, self.seq[a], meas)

    # -------------------------------------------------------------- metrics
    def tcp(self, ecm):
        return smoke.gz_mat(self.ctl.hand.world_pose(ecm)) @ pp.tf(p=(0, 0, pp.TCP_Z))

    def joint(self, snap, name):
        names = [str(x) for x in snap["joint_names"]]
        return float(snap["joint_pos"][names.index(name)])

    def in_contact(self, key, t0, t1):
        """Both fingers touched the object during [t0, t1] (sim time)."""
        return all(any(t0 <= t <= t1 for t in self.obj_contact.get((key, f), [])) for f in FINGERS)

    def post_update(self, info, ecm):
        if PROFILE and not (info.paused or self.finished):
            c0 = time.perf_counter()
            self._post_update(info, ecm)
            self.prof["post_s"] += time.perf_counter() - c0
        else:
            self._post_update(info, ecm)

    def _post_update(self, info, ecm):
        if info.paused or self.finished:
            return
        if self.ending is not None:     # wait (sim keeps running) for lagging contact messages
            if self.last_stamp >= self.ending or time.time() - self.ending_wall > 60:
                self.finished = True
                self.report()
            return
        t = self.t = info.sim_time.total_seconds()
        n = int(round(t / mp.CONTROL_DT)) - 1
        a = self.a = self.action_index(n)
        while self.ev_i + 1 < len(self.ev_steps) and a >= self.ev_steps[self.ev_i + 1][0]:
            self.ev_i += 1
            jd = dict(self.state.joints)
            door = (jd["fridge::top_door_hinge"].position(ecm) or [0.0])[0]
            drawer = (jd["kitchen_counter::middle_slide_top"].position(ecm) or [0.0])[0]
            cup = smoke.gz_mat(self.cup.world_pose(ecm))[:3, 3]
            self.say(f"t={t:6.2f}s {self.ev_steps[self.ev_i][1]:<18} door {math.degrees(door):5.2f} deg, "
                     f"drawer {drawer*100:5.2f} cm, cup {np.round(cup, 3)}, TCP {np.round(self.tcp(ecm)[:3, 3], 3)}")
        if self.ev_i + 0 < len(self.ev_steps) and a < self.ev_steps[self.ev_i][0]:
            self.ev_i = max(i for i, (k, _) in enumerate(self.ev_steps) if k <= a)
        if (n % 5 == 0) if R == 1 else ((n + 1) // R) % max(1, 5 // R) == 0:
            for lname, link in self.state.links:
                v, w = link.world_linear_velocity(ecm), link.world_angular_velocity(ecm)
                com = link.world_inertial_pose(ecm)
                if v is None or w is None or com is None:
                    continue
                s = (v + w.cross(com.pos() - link.world_pose(ecm).pos())).length()
                if math.isnan(s):
                    self.nan = True
                elif s > self.vmax[0]:
                    self.vmax = (s, lname, t)
        if n == grid(0):
            self.snaps["initial"] = self.state.capture(ecm, step=n)
        if self.k_save is not None:
            self.reset_hooks(n, a, ecm)
        for w in self.windows:
            if w["k0"] <= a <= w["k1"]:
                self.window_step(w, a, ecm, t)
        if n >= self.n_total - 1:
            self.snaps["final"] = self.state.capture(ecm, step=n, skill=self.skill)
            self.ending, self.ending_wall = t, time.time()
            if PROFILE:
                p = self.prof
                wall = self.ending_wall - p["wall_first"]
                self.extra["profile"] = dict(
                    step_s=STEP, steps=p["steps"], sim_s=round(t - p["sim_first"], 3), wall_s=round(wall, 2),
                    rtf=round((t - p["sim_first"]) / wall, 3),
                    rtf_after_5s=round((t - p["sim5"]) / (self.ending_wall - p["wall5"]), 3),
                    python_pre_ms_per_step=round(p["pre_s"] / p["steps"] * 1000, 4),
                    python_post_ms_per_step=round(p["post_s"] / p["steps"] * 1000, 4),
                    python_share=round((p["pre_s"] + p["post_s"]) / wall, 3),
                    contact_cb_s=round(p["cb_s"], 2), contact_msgs=self.pen["msgs"],
                    wall_ms_per_step=round(wall / p["steps"] * 1000, 4))
                self.say(f"profile {json.dumps(self.extra['profile'])}")

    def reset_hooks(self, n, a, ecm):
        if n == self.k_save:
            self.snaps["save"] = self.state.capture(ecm, step=n, action=a)
            ss.save(os.path.join(self.dir, "saved_state.npz"), self.snaps["save"])
            self.say(f"t={self.t:6.2f}s SAVE scene state (step {n})")
        elif n == self.k_save + 1:
            self.snaps["save+1"] = self.state.capture(ecm, step=n)
        elif n == self.k_restore:
            self.snaps["orig_end"] = self.state.capture(ecm, step=n)
        elif n == self.k_restore + 1:
            d = ss.diff(self.snaps["save+1"], self.state.capture(ecm, step=n))
            self.extra["restore_vs_saved"] = d
            ok = (d["link_pos_m"][0] <= RESTORE_TOL[0] and d["link_rot_rad"][0] <= RESTORE_TOL[1]
                  and d["link_vel_mps"][0] <= RESTORE_VEL_TOL[0]
                  and d["link_angvel_radps"][0] <= RESTORE_VEL_TOL[1])
            self.results.append(dict(kind="restore", name="restored state vs saved (1 step later)",
                                     pass_=ok, value=_fmt(d)))
            self.say(f"t={self.t:6.2f}s CHECK restore: {'PASS' if ok else 'FAIL'} {_fmt(d)}")
        elif n == 2 * self.k_restore - self.k_save:
            d = ss.diff(self.snaps["orig_end"], self.state.capture(ecm, step=n))
            self.extra["rerun_from_restore_vs_original"] = d
            ok = d["link_pos_m"][0] <= REPLAY_TOL
            self.results.append(dict(kind="restore", name="rerun from restored state vs original (1.5 s)",
                                     pass_=ok, value=_fmt(d)))
            self.say(f"t={self.t:6.2f}s CHECK rerun from restore: {'PASS' if ok else 'FAIL'} {_fmt(d)}")

    def window_step(self, w, a, ecm, t):
        cup = smoke.gz_mat(self.cup.world_pose(ecm))
        st = self.win_stats.setdefault(id(w), dict(cup_zmin=math.inf, cup_vmax=0.0, tilt_max=0.0,
                                                   t0=t))
        if a == w["k0"]:
            st.update(cup_zmin=math.inf, cup_vmax=0.0, tilt_max=0.0, t0=t)
            if w["info"].get("obj"):        # object pose in the hand at the window start
                st["rel0"] = np.linalg.inv(self.tcp(ecm)) @ smoke.gz_mat(
                    self.obj_links[w["info"]["obj"]].world_pose(ecm))
        st["cup_zmin"] = min(st["cup_zmin"], cup[2, 3])
        v = self.cup.world_linear_velocity(ecm)
        st["cup_vmax"] = max(st["cup_vmax"], v.length() if v is not None else 0.0)
        st["tilt_max"] = max(st["tilt_max"], math.degrees(math.acos(max(-1.0, min(1.0, cup[2, 2])))))
        if a == w["k1"]:
            self.evaluate(w, st, ecm, t)

    def evaluate(self, w, st, ecm, t):
        kind, info = w["kind"], w["info"]
        snap = self.state.capture(ecm)
        fingers = self.ctl.state(ecm)["fingers"]
        Tt = self.tcp(ecm)
        r = dict(kind=kind, name=kind, t=round(t, 3), fingers_mm=(fingers * 1000).round(1).tolist())
        obj = info.get("obj")
        if obj:
            rel = np.linalg.inv(Tt) @ smoke.gz_mat(self.obj_links[obj].world_pose(ecm))
            if kind == "grasped":
                self.grasp_rel[obj] = rel
            elif kind == "cup_lifted":      # drift in the hand over the hold
                self.grasp_rel[obj] = st["rel0"]
            slip = float(np.linalg.norm(rel[:3, 3] - self.grasp_rel[obj][:3, 3]))
            contact = True       # evaluated in report() (contact messages lag)
            r.update(slip_mm=round(slip * 1000, 2), contact_window=[obj, st["t0"], t])
        ok = True
        if kind == "grasped":
            # pinched across the handle box's width (model y) / the bar's height (the smaller
            # of its y/z sizes, whichever axis the manifest gives the bar's length on)
            bs = self.S["drawer"]["bar_size"]
            width = {"handle": self.S["fridge"]["handle_size"][1], "bar": min(bs[1], bs[2])}[obj] / 2
            ok = contact and bool(np.all(np.abs(fingers - width) < 0.006))
            msg = f"fingers {fingers[0]*1000:.1f}/{fingers[1]*1000:.1f} mm (object half-width " \
                  f"{width*1000:.1f})"
        elif kind in ("door_pulled", "door_released"):
            ang = self.joint(snap, "fridge::top_door_hinge")
            r["door_deg"] = round(math.degrees(ang), 2)
            ok = ang >= math.radians(3.0)
            msg = f"door {math.degrees(ang):.2f} deg (planned {math.degrees(info['want']):.1f})"
            if kind == "door_pulled":
                best, per = self.clearance(ecm, per_link=True)
                arm = {k: v for k, v in per["fridge"].items() if k not in FINGERS + ("panda_hand",)}
                r["clearance_fridge_cm"] = {k: round(v * 100, 1) for k, v in per["fridge"].items()}
                ok &= contact and slip <= SLIP_TOL and min(arm.values()) > 0.01
                msg += f", slip {slip*1000:.1f} mm, clearance to fridge (cm): " + \
                       ", ".join(f"{k.replace('panda_', '')} {v*100:.1f}" for k, v in per["fridge"].items())
        elif kind in ("drawer_pulled", "drawer_released"):
            x = self.joint(snap, "kitchen_counter::middle_slide_top")
            r["drawer_cm"] = round(x * 100, 2)
            ok = x >= 0.10
            msg = f"drawer {x*100:.1f} cm (planned {info['want']*100:.0f})"
            if kind == "drawer_pulled":
                best = self.clearance(ecm)
                r["clearance_counter_min_cm"] = round(best["kitchen_counter"][0] * 100, 2)
                ok &= contact and slip <= SLIP_TOL
                msg += f", slip {slip*1000:.1f} mm"
        elif kind == "cup_lifted":
            lift = st["cup_zmin"] - info["z0"]
            r.update(lift_min_cm=round(lift * 100, 2))
            ok = lift >= 0.10 and slip <= SLIP_TOL
            msg = f"cup lifted >= {lift*100:.1f} cm over the 2 s hold, slip in hand {slip*1000:.1f} mm"
        elif kind == "cup_placed":
            p = np.array(info["place"])
            dxy = float(np.linalg.norm(snap_cup(self, ecm)[:2] - p[:2]))
            dz = float(snap_cup(self, ecm)[2] - p[2])
            r.update(tilt_max_deg=round(st["tilt_max"], 2), speed_max_cms=round(st["cup_vmax"] * 100, 3),
                     place_err_cm=round(dxy * 100, 2), z_above_top_mm=round(dz * 1000, 2))
            ok = st["tilt_max"] <= 15 and st["cup_vmax"] < 0.02 and dxy < 0.03 and abs(dz) < 0.01
            msg = f"tilt max {st['tilt_max']:.2f} deg, speed max {st['cup_vmax']*100:.2f} cm/s over 2 s, " \
                  f"{dxy*100:.1f} cm from the place point, bottom {dz*1000:.1f} mm above the top"
        r["pass_"] = bool(ok)
        r["value"] = msg
        self.results.append(r)
        self.say(f"t={t:6.2f}s CHECK {kind}: {'PASS' if ok else 'FAIL'} {msg}")

    def report(self):
        checks = []

        def check(name, ok, value, limit):
            checks.append({"name": name, "pass": bool(ok), "value": value, "limit": limit})

        for r in self.results:
            if "contact_window" in r:
                obj, t0, t1 = r["contact_window"]
                r["contact_both_fingers"] = c = self.in_contact(obj, t0, t1)
                r["pass_"] = r["pass_"] and c
                r["value"] += f", both fingers on the {obj} {c}"
            check(r["name"], r["pass_"], r["value"], "")
        n_win = sum(1 for r in self.results if r["kind"] != "restore")
        if SKILL != "replay":
            check("all planned checks ran", n_win >= len(self.windows), n_win, len(self.windows))
        if SKILL == "replay":
            d = ss.diff(self.ref, self.snaps["final"])
            self.extra["replay_vs_recorded"] = d
            check("replay final state vs recorded run (max link position diff, m)",
                  d["link_pos_m"][0] <= REPLAY_TOL, _fmt(d), REPLAY_TOL)
        if self.k_save is not None:
            check("reset: restore + rerun checks ran", sum(r["kind"] == "restore" for r in self.results) == 2,
                  sum(r["kind"] == "restore" for r in self.results), 2)
        check("robot contacts other than fingers on the grasped object", not self.other_contacts,
              self.other_contacts[:5], [])
        check("physics: peak body speed (m/s)", self.vmax[0] < MAX_SPEED and not self.nan,
              [round(self.vmax[0], 3), self.vmax[1], round(self.vmax[2], 2)], MAX_SPEED)
        check("physics: max contact penetration (cm)", self.pen["max"] < MAX_PEN and self.pen["msgs"] > 0,
              [round(self.pen["max"] * 100, 4), self.pen["where"], round(self.pen["t"], 2)], MAX_PEN * 100)
        check("controller: non-finite joint commands (zeroed)", not self.ctl.bad, self.ctl.bad[:3], [])
        ok = all(c["pass"] for c in checks)
        idx = self.applied
        ss.save(os.path.join(self.dir, "initial_state.npz"), self.snaps["initial"])
        ss.save(os.path.join(self.dir, "final_state.npz"), self.snaps["final"])
        mp.ActionSeq(**{k: getattr(self.seq, k)[idx] for k in mp.ActionSeq.FIELDS}).save(
            os.path.join(self.dir, "actions.npz"))
        data = {"pass": ok, "skill": SKILL, "checks": checks, "results": self.results, "extra": self.extra,
                "peak_speed": {"value": self.vmax[0], "link": self.vmax[1], "t": self.vmax[2]},
                "penetration": self.pen, "finger_contact_first_t": self.contact_first, "other_robot_contacts": self.other_contacts,
                "final_joints": {str(n): float(v) for n, v in zip(self.snaps["final"]["joint_names"],
                                                                 self.snaps["final"]["joint_pos"])},
                "plan": {"duration_s": self.sc.P.t, "steps": len(self.seq), "physics_steps": self.n_total,
                         "ik_worst_mm_deg": [self.sc.P.ik_worst[0] * 1000, math.degrees(self.sc.P.ik_worst[1])],
                         "events": [(round(t, 2), lab) for t, lab, _ in self.sc.P.events]}}
        with open(self.out, "w") as f:
            json.dump(data, f, indent=1, default=_json)
        self.say(f"DONE {'PASS' if ok else 'FAIL'}: {sum(c['pass'] for c in checks)}/{len(checks)} "
                 f"checks, summary {self.out}")


def robot_state(snap):
    """Controller.state()-like dict from a scene snapshot."""
    names = [str(x) for x in snap["joint_names"]]

    def get(f, joints):
        return np.array([snap[f][names.index(f"{mp.MODEL}::{j}")] for j in joints])
    arm = [f"panda_joint{i}" for i in range(1, 8)]
    fingers = ["panda_finger_joint_left", "panda_finger_joint_right"]
    return dict(base=get("joint_pos", mp.BASE_JOINTS), base_vel=get("joint_vel", mp.BASE_JOINTS),
                arm=get("joint_pos", arm), arm_vel=get("joint_vel", arm),
                fingers=get("joint_pos", fingers), fingers_vel=get("joint_vel", fingers))


def snap_cup(sys_, ecm):
    return smoke.gz_mat(sys_.cup.world_pose(ecm))[:3, 3]


def _fmt(d):
    return {k: ([round(v[0], 6), v[1]] if isinstance(v, list) else v) for k, v in d.items()}


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def get_system():
    return SkillSystem()
