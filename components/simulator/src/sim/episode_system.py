"""gz-sim Python system: the robot's scripted kitchen episodes (robots/<robot>/scripted.py EPISODES).

Loaded by a copy of worlds/kitchen_partnet_skills.sdf that a scripted episode (`cli.py episode`)
writes (module and camera swapped). $EPISODE picks the episode (fridge,
drawer). Same controller, logging and physics checks as the skill tests
(skill_system.SkillSystem: peak body speed, contact penetration,
robot contacts other than the gripper's contact links on the grasped object, their contact
and slip of the grasped object), plus the episode checks:
  door_open        hinge angle >= want for the whole hold (handle still grasped), handle slip
  door_stays_open  hinge angle >= want from the handle release until the cup is out
  cup_out          cup outside the fridge's bounding box, lifted, in both fingers
  cup_placed       on the table: tilt <= 15 deg, speed < 2 cm/s for 2 s (SkillSystem),
                   and no finger touching it (released)
  drawer_open      drawer >= want for the whole hold after the release
  sim time         the episode's last check ends within the budget
Verdict "DONE PASS/FAIL" on /robot_skill/status; kitchen_test.py
drives the run and records the MP4 (the only output the episode needs; the
summary JSON/TXT is written for the checks).
"""
import json
import math
import os
import time

import numpy as np

import skill_system as sks
import skills as sk
import smoke_system as smoke

EPISODE = os.environ.get("EPISODE", "drawer")
NEW_KINDS = ("door_open", "cup_out", "drawer_open")


class EpisodeSystem(sks.SkillSystem):
    def make_scenario(self, name):
        return sk.episode(EPISODE, self.S)

    def configure(self, entity, sdf, ecm, event_mgr):
        super().configure(entity, sdf, ecm, event_mgr)
        self.skill = f"episode_{EPISODE}"
        start = {lab: k for k, lab in self.ev_steps}
        ends = {}
        for i, (k, lab) in enumerate(self.ev_steps):
            ends[lab] = self.ev_steps[i + 1][0] - 1 if i + 1 < len(self.ev_steps) else len(self.seq) - 1
        self.spans = [dict(sp, k0=sks.grid(start[sp["start"]]), k1=sks.grid(ends[sp["end"]]), vmin=math.inf)
                      for sp in getattr(self.sc, "spans", [])]
        self.ep_stats = {}
        self.track = dict(door_max=(0.0, 0.0), drawer_max=(0.0, 0.0))
        self.k_lift = start.get("arm lift")
        if self.k_lift is not None:
            self.k_lift = sks.grid(self.k_lift)
        with open(os.path.join(sk.model_dir("fridge"), "manifest.json")) as f:
            lo, hi = json.load(f)["aabb_model_frame"]
        c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        w = np.array([sk.to_world("fridge", p) for p in c])
        self.fridge_box = (w.min(0), w.max(0))
        self.wall0 = time.time()
        self.say(f"episode {EPISODE}: plan {self.sc.P.t:.1f} s, budget {self.sc.info['budget']:.0f} s, "
                 f"{len(self.spans)} spans")

    def joints_now(self, ecm):
        jd = dict(self.state.joints)
        door = (jd["fridge::top_door_hinge"].position(ecm) or [0.0])[0]
        drawer = (jd["kitchen_counter::middle_slide_top"].position(ecm) or [0.0])[0]
        return door, drawer

    def window_step(self, w, a, ecm, t):
        door, drawer = self.joints_now(ecm)
        es = self.ep_stats.setdefault(id(w), {})
        if a == w["k0"] or not es:
            es.update(door_min=math.inf, drawer_min=math.inf)
        es["door_min"] = min(es["door_min"], door)
        es["drawer_min"] = min(es["drawer_min"], drawer)
        super().window_step(w, a, ecm, t)

    def _post_update(self, info, ecm):
        busy = not (info.paused or self.finished or self.ending is not None)
        super()._post_update(info, ecm)
        if not busy:
            return
        t = info.sim_time.total_seconds()
        a = self.a
        door, drawer = self.joints_now(ecm)
        if door > self.track["door_max"][0]:
            self.track["door_max"] = (door, t)
        if drawer > self.track["drawer_max"][0]:
            self.track["drawer_max"] = (drawer, t)
        for sp in self.spans:
            if sp["k0"] <= a <= sp["k1"]:
                sp["vmin"] = min(sp["vmin"], door)
                if a == sp["k1"]:
                    ok = sp["vmin"] >= sp["info"]["want"]
                    msg = (f"door min {math.degrees(sp['vmin']):.2f} deg from '{sp['start']}' to the end of "
                           f"'{sp['end']}' (want >= {math.degrees(sp['info']['want']):.0f})")
                    self.results.append(dict(kind=sp["kind"], name=sp["kind"], t=round(t, 3), pass_=ok,
                                             value=msg, door_min_deg=round(math.degrees(sp["vmin"]), 2)))
                    self.say(f"t={t:6.2f}s CHECK {sp['kind']}: {'PASS' if ok else 'FAIL'} {msg}")
        if self.k_lift is not None and a == self.k_lift:        # cup in the hand at the lift start
            self.grasp_rel["cup"] = np.linalg.inv(self.tcp(ecm)) @ smoke.gz_mat(self.cup.world_pose(ecm))
        if a >= len(self.seq) - 1 and "end" not in self.track:
            cup = smoke.gz_mat(self.cup.world_pose(ecm))
            v = self.cup.world_linear_velocity(ecm)
            self.track["end"] = dict(t=round(t, 3), door_deg=round(math.degrees(door), 2),
                                     drawer_cm=round(drawer * 100, 2), cup_pos=cup[:3, 3].round(4).tolist(),
                                     cup_tilt_deg=round(math.degrees(math.acos(min(1.0, cup[2, 2]))), 2),
                                     cup_speed_cms=round((v.length() if v is not None else 0.0) * 100, 3))

    def evaluate(self, w, st, ecm, t):
        kind, info = w["kind"], w["info"]
        if kind not in NEW_KINDS:
            return super().evaluate(w, st, ecm, t)
        es = self.ep_stats[id(w)]
        Tt = self.tcp(ecm)
        fingers = self.ctl.state(ecm)["fingers"]
        r = dict(kind=kind, name=kind, t=round(t, 3), fingers_mm=(fingers * 1000).round(1).tolist())
        obj, slip = info.get("obj"), 0.0
        if obj:
            rel = np.linalg.inv(Tt) @ smoke.gz_mat(self.obj_links[obj].world_pose(ecm))
            ref = self.grasp_rel.get(obj, st.get("rel0", rel))
            slip = float(np.linalg.norm(rel[:3, 3] - ref[:3, 3]))
            r.update(slip_mm=round(slip * 1000, 2), contact_window=[obj, st["t0"], t])
        if kind == "door_open":
            r["door_min_deg"] = round(math.degrees(es["door_min"]), 2)
            ok = es["door_min"] >= info["want"] and slip <= sks.SLIP_TOL
            msg = (f"door >= {math.degrees(es['door_min']):.2f} deg over the {t - st['t0']:.1f} s hold "
                   f"(want >= {math.degrees(info['want']):.0f}), handle slip in the hand since the grasp "
                   f"{slip*1000:.1f} mm")
        elif kind == "cup_out":
            c = smoke.gz_mat(self.cup.world_pose(ecm))[:3, 3]
            lo, hi = self.fridge_box
            inside = bool(np.all(c >= lo - sk.CUP_R) and np.all(c <= hi + sk.CUP_R))
            tilt = math.degrees(math.acos(min(1.0, smoke.gz_mat(self.cup.world_pose(ecm))[2, 2])))
            ok = not inside and slip <= sks.SLIP_TOL and tilt <= 15
            r.update(cup_pos=c.round(4).tolist(), outside_fridge=not inside, tilt_deg=round(tilt, 2))
            msg = (f"cup at {np.round(c, 3).tolist()}, outside the fridge box {not inside}, tilt {tilt:.1f} deg, "
                   f"slip in the hand since the lift {slip*1000:.1f} mm")
        else:   # drawer_open
            r["drawer_min_cm"] = round(es["drawer_min"] * 100, 2)
            ok = es["drawer_min"] >= info["want"]
            msg = (f"drawer >= {es['drawer_min']*100:.2f} cm over the {t - st['t0']:.1f} s hold after the "
                   f"release (want >= {info['want']*100:.0f})")
        r["pass_"], r["value"] = bool(ok), msg
        self.results.append(r)
        self.say(f"t={t:6.2f}s CHECK {kind}: {'PASS' if ok else 'FAIL'} {msg}")

    def report(self):
        for r in self.results:
            if r["kind"] == "cup_placed" and "released" not in r:
                t1 = r["t"]
                t0 = t1 - 2.0
                touch = [f for f in self.fingers
                         if any(t0 <= x <= t1 for x in self.obj_contact.get(("cup", f), []))]
                r["released"] = not touch
                r["pass_"] = r["pass_"] and not touch
                r["value"] += f", fingers off the cup {not touch}"
        t_last = max((r["t"] for r in self.results if "t" in r), default=math.inf)
        budget = self.sc.info["budget"]
        ok = t_last <= budget
        self.results.append(dict(kind="budget", name="sim time", pass_=ok,
                                 value=f"last check ends at {t_last:.2f} s sim (budget {budget:.0f} s), "
                                       f"plan {self.sc.P.t:.1f} s"))
        self.extra["episode"] = dict(
            name=EPISODE, plan_s=self.sc.P.t, last_check_s=t_last, budget_s=budget,
            wall_s_to_end=round(time.time() - self.wall0, 1),
            door_max_deg=[round(math.degrees(self.track["door_max"][0]), 2), round(self.track["door_max"][1], 2)],
            drawer_max_cm=[round(self.track["drawer_max"][0] * 100, 2), round(self.track["drawer_max"][1], 2)],
            end=self.track.get("end"))
        self.say(f"episode metrics {json.dumps(self.extra['episode'])}")
        super().report()


def get_system():
    return EpisodeSystem()
