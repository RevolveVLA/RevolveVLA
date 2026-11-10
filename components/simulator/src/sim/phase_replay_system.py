"""gz-sim Python system: the live (online) phase validator (cli.py replay / episode / restore-check).
Ported from the earlier pipeline's phase_replay_system.py. The gates are the task spec's
phase conditions (live.json "task" -> specs/tasks/<t>/task.yaml), evaluated with
revolve_common.metrics on metric states built from the live sim; nothing is hardcoded here. Each
phase's metric states are kept in phase_live.json ("metric_states") so a verdict can be re-evaluated.

ReplaySystem (automoma_replay_system: the replay of $REPLAY_DIR/actions.npz, the episode's checks,
state_log.npz, summary.json) plus, while the run happens (phases.py has the rules):
- the phase boundaries from the fed actions, step by step (phases.Boundaries);
- at every phase start the whole scene state (scene_state.SceneState.capture: robot base, arm and
  fingers, the door / drawer joints, the cup and every other non-static body) in
  phase_states/<phase>_start.npz, with the action index, the sim time and the action of that step,
  and the same one step later in <phase>_start_next.npz (the reference for a restore check);
- the gates of the phase that ends there, on the live sim state; its verdict waits until the
  contact messages (which lag the sim by seconds) are stamped past the phase end, then
  "LIVE t=... PHASE <name> PASS/FAIL" is published on /robot_skill/status;
- per step: the peak link speed, the deepest contact and the robot contacts other than the
  gripper's contact links on the handle, for the first-violation time of each global check.
Writes $REPLAY_DIR/phase_live.json before the DONE line; live_validate.py finalize adds the gz
log check and the offline verdict of the same run -> phase_report.json.

Restore mode (live.json "restore"): actions.npz starts at a saved phase start (action index
`action_index` of the source run). Step 0 restores the snapshot (the controller acts on the
snapshot's robot state in that step), and the state after step 0 is compared with the source
run's state one step after the boundary (<phase>_start_next.npz) and with the boundary snapshot
itself (restore_check.json, with the rest of the run against the source state_log.npz).

Policy mode (live.json "policy", from a policy_episode.py episode): the phase boundaries are
phases.PolicyBoundaries (hold_check.py's windows on the gripper target and the object joint), the
open phase has no hold_duration gate (a policy's targets never stand still), handle_in_tcp is logged
per step and the closed-loop success check (closed_loop_system's, hold_check.py) runs at the end, so
summary.json / verdict.json can be compared with the rollout's.
"""
import json
import math

import numpy as np

from revolve_common import metrics as M

import automoma_replay_system as ars
import phases as ph
import scene_state as ss
import skill_system as sks
import smoke_system as smoke
import spec
from robots.base import CONTROL_DT, robot_state

DT = CONTROL_DT
SLIP_EVERY = 5                  # steps between handle slip samples in the open phase


class PhaseReplaySystem(ars.ReplaySystem):
    def configure(self, entity, sdf, ecm, event_mgr):
        self.L = json.loads((ars.REPLAY_DIR / "live.json").read_text())
        self._meas = None
        super().configure(entity, sdf, ecm, event_mgr)
        L, n = self.L, self.n_total
        self.r = spec.task(L["task"])
        T = self.r.task
        if tuple(p.name for p in T.phases) != ph.PHASES:
            raise RuntimeError(f"task {T.name} phases {[p.name for p in T.phases]}: the boundary rules know {ph.PHASES}")
        self.conds = {p.name: p.pass_.all for p in T.phases}
        self.timed = {c.metric: c for c in T.sanity.timed}
        self.live = T.live
        si = spec.hold(self.r)[3]
        self.slip_cond = spec.cond(self.r, "open", "handle_slip_m")
        self.hold_s = spec.open_cond(self.r).hold_s
        self.R = L.get("restore")
        self.offset = self.R["action_index"] if self.R else 0
        first = self.R["phase"] if self.R else "approach"
        self.pol = L.get("policy")
        bp = T.boundaries.policy
        self.bounds = (ph.PolicyBoundaries(dict(closed_max_m=bp.grip_closed_max_m, settle_s=bp.grasp_settle_s,
                                                moved_eps=si(bp.joint_moved_eps)), DT, first)
                       if self.pol else ph.Boundaries(first, T.boundaries.change_tol))
        self.mstates = {}
        if self.pol:
            self.log["handle_in_tcp"] = np.zeros((n, 3))
        self.order = list(ph.PHASES[ph.PHASES.index(first):])
        self.starts = {first: 0}
        self.state_dir = ars.REPLAY_DIR / "phase_states"
        self.state_dir.mkdir(exist_ok=True)
        self.state_files, self.next_due = {}, {}
        self.ps = dict(peak_body_speed_mps=np.zeros(n + 1), penetration_m=np.zeros(n + 1),
                       other_robot_contacts=np.zeros(n + 1, np.int64))
        self.first = {}                 # timed sanity metric -> first violating step
        self.state_checks, self.live_windows = {}, {}
        self.pending, self.decided = [], {}
        self.live_grasp_rel = None
        if self.R and "grasp_rel" in self.R:
            self.live_grasp_rel = np.array(self.R["grasp_rel"])
        self.slip = dict(max=0.0, first=None)
        self.restore = None
        if self.R:
            self.restore_snap = ss.load(ars.REPLAY_DIR / self.R["snapshot"])
            self.restore_next = ss.load(ars.REPLAY_DIR / self.R["snapshot_next"])
            self.state_files[first] = dict(file=self.R["snapshot"], step=0, action_index=self.offset,
                                           sim_time_s=0.0, source_sim_time_s=round(self.offset * DT, 3))
        self.obj = L["obj"]
        self.say(f"LIVE feed {L['feed']}, task {T.name}, phases from {first}"
                 + (f" (restored at source action {self.offset})" if self.R else "")
                 + ", gates " + "; ".join(f"{p}: " + ", ".join(f"{c.metric} {c.op} {c.threshold}" for c in cs)
                                          for p, cs in self.conds.items()))

    # ------------------------------------------------------------------ helpers
    def measured(self, ecm):
        return self._meas if self._meas is not None else self.ctl.state(ecm)

    def handle_rel(self, ecm):
        return np.linalg.inv(self.tcp(ecm)) @ smoke.gz_mat(self.obj_links[self.obj].world_pose(ecm))

    def snapshot(self, ecm, name, n, **extra):
        a = self.seq[min(n, len(self.seq) - 1)]
        snap = self.state.capture(ecm, step=n, action_index=n + self.offset, sim_time_s=n * DT,
                                  source_sim_time_s=(n + self.offset) * DT, feed=self.L["feed"],
                                  action_base=a.base, action_arm=a.arm, action_grip=a.grip,
                                  action_grip_force=a.grip_force, **extra)
        ss.save(self.state_dir / f"{name}.npz", snap)
        return snap

    def save_phase_state(self, phase, n, ecm):
        extra = {} if self.live_grasp_rel is None else dict(grasp_rel=self.live_grasp_rel)
        self.snapshot(ecm, f"{phase}_start", n, phase=phase, **extra)
        self.state_files[phase] = dict(file=f"phase_states/{phase}_start.npz", step=n,
                                       action_index=n + self.offset, sim_time_s=round(n * DT, 3),
                                       source_sim_time_s=round((n + self.offset) * DT, 3))
        self.next_due[n + 1] = phase

    # ------------------------------------------------------------- the stream
    def pre_update(self, info, ecm):
        if info.paused:
            return
        n = int(round(info.sim_time.total_seconds() / DT)) - 1
        if self.R and n == 1:
            self.state.after_restore(ecm)
            self.check_restore(ecm)
        if 0 <= n < self.n_total:
            if n in self.next_due:
                ph_ = self.next_due.pop(n)
                self.snapshot(ecm, f"{ph_}_start_next", n, phase=ph_)
            if n == 1 and not self.R:            # the robot's spawn reset takes effect in step 0
                self.save_phase_state("approach", 1, ecm)
            a = self.seq[n]
            new = self.bounds.step(n, a.base, a.arm, a.grip, q=(self.door.position(ecm) or [0.0])[0])
            if new:
                self.boundary(new, n, ecm, info.sim_time.total_seconds())
        if self.R and n == 0:
            self.state.restore(ecm, self.restore_snap)
            self._meas = robot_state(self.robot, self.restore_snap)
        super().pre_update(info, ecm)
        self._meas = None
        if self.pol and 0 <= n < self.n_total:
            self.log["handle_in_tcp"][n] = self.handle_rel(ecm)[:3, 3]

    def boundary(self, new, n, ecm, t):
        old = self.order[self.order.index(new) - 1]
        self.starts[new] = n
        self.end_of(old, n, ecm)
        self.save_phase_state(new, n, ecm)
        self.pending.append(old)
        self.say(f"LIVE t={t:6.3f}s boundary {old} -> {new} at step {n} (source action {n + self.offset})")

    def window_steps(self, p, s, e):
        """The finger-contact window of phase p (its finger_contacts window_s, ending at the phase end)."""
        c = next((c for c in self.conds[p] if c.metric == "finger_contacts"), None)
        if c is None:
            return None
        w = c.window_s if c.window_s is not None else (e - s) * DT
        return max(s, e - int(round(w / DT))), e

    def judge(self, p, metric_name, c, state, t_fail):
        """One task condition on a metric state -> a phase check (None: does not apply live)."""
        r = M.evaluate(c, state, offline=False, policy=bool(self.pol))
        if r is None:
            return None
        self.mstates.setdefault(p, {})[metric_name] = state
        return ph.from_result(r, t_fail=None if r["pass_"] else t_fail)

    def end_of(self, p, e, ecm):
        """State gates of phase p, which ends before step e (the ECM holds the state after step e - 1): every
        condition of the task's phase p but finger_contacts (decided once the contact stamps pass the end)."""
        s, out = self.starts[p], []
        Tt = self.tcp(ecm)
        te = round(e * DT, 3)
        if p == "grasp":
            self.live_grasp_rel = self.handle_rel(ecm)
        if p == "open" and self.live_grasp_rel is not None:
            slip = float(np.linalg.norm(self.handle_rel(ecm)[:3, 3] - self.live_grasp_rel[:3, 3]))
            self.slip["max"] = max(self.slip["max"], slip * 1000)
        for c in self.conds[p]:
            m = c.metric
            if m == "finger_contacts":
                continue
            if m in ("tcp_position_error_m", "tcp_rotation_error_deg"):
                st = dict(tcp_pose=Tt, grasp_pose=np.array(self.L["grasp_pose_world"]))
                seg = (self.pol or {}).get("grasp_segment_world")
                if seg and m == "tcp_position_error_m":   # policy: any grasp height of the task (policy_episode.py)
                    st["grasp_segment"] = seg
                chk = self.judge(p, m, c, st, te)
            elif m == "grasp_width_error_m":
                chk = self.judge(p, m, c, dict(fingers=self.ctl.state(ecm)["fingers"],
                                               grasp_half_width_m=self.ep["thresholds"]["grasp_half_width_m"]), te)
            elif m in ("joint_deg", "joint_cm"):
                k0 = max(s, e - int(round((c.hold_s or 0.0) / DT)) - 1)
                series = [dict(t=k * DT, joints={c.joint: float(self.log["door_pos"][k])}) for k in range(k0, e)]
                bad = [x["t"] for x in series if ph.violates(c, M.value(c, x))]
                chk = self.judge(p, m, c, series, round(bad[0] + DT, 3) if bad else te)
            elif m == "handle_slip_m":
                ref = self.live_grasp_rel[:3, 3] if self.live_grasp_rel is not None else np.full(3, math.inf)
                chk = self.judge(p, m, c, dict(handle_in_tcp=self.handle_rel(ecm)[:3, 3], handle_in_tcp_ref=ref),
                                 self.slip["first"] if self.slip["first"] is not None else te)
            elif m == "targets_static_s":
                tg = np.hstack([self.seq.base[s:e], self.seq.arm[s:e]])
                moved = np.flatnonzero(np.any(np.abs(np.diff(tg, axis=0)) > self.bounds.tol, axis=1)) + 1
                tg = tg[moved[-1] - 1:] if len(moved) else tg     # from the last motion on: the same value, a small state
                chk = self.judge(p, m, c, dict(targets=tg, dt=DT, change_tol=self.bounds.tol), te)
            elif m == "spawn_clearance_m":     # offline only (replay.py precheck): not a live gate
                chk = self.judge(p, m, c, dict(spawn_clearance_m=self.ep.get("clearance", {}).get(
                    "settle", {}).get("min_m", math.inf)), te)
            else:
                chk = ph.check(m, False, None, c.op, c.threshold, "", t_fail=te,
                               note="metric not measured by the live validator")
            if chk is not None:
                out.append(chk)
        w = self.window_steps(p, s, e)
        if w is not None:
            self.live_windows[p] = w
        self.state_checks[p] = out

    # ------------------------------------------------------------- per step
    def on_contacts(self, msg):
        super().on_contacts(msg)
        if not hasattr(self, "ps"):
            return
        st = msg.header.stamp
        k = min(max(int(round((st.sec + st.nsec * 1e-9) / DT)) - 1, 0), self.n_total)
        depth, other = 0.0, 0
        for c in msg.contact:                    # as SkillSystem._on_contacts classifies them
            names = (c.collision1.name, c.collision2.name)
            if len(c.depth):
                depth = max(depth, max(c.depth))
            rob = [s for s in names if s.startswith(self.robot.model + "::")]
            if not rob:
                continue
            oth = [s for s in names if not s.startswith(self.robot.model + "::")]
            link = rob[0].split("::")[1]
            hit = bool(oth) and any(oth[0].startswith(f"{mn}::{ln}::{cn}") for mn, ln, cn in sks.OBJ.values())
            other += not (hit and link in self.fingers)
        P = self.ps
        P["penetration_m"][k] = max(P["penetration_m"][k], depth)
        P["other_robot_contacts"][k] += other
        for m, v in (("penetration_m", depth), ("other_robot_contacts", other)):
            c = self.timed.get(m)
            if c is not None and ph.violates(c, M.value(c, {ph.SANITY_KEY[m]: [v]})) \
                    and k < self.first.get(m, math.inf):
                self.first[m] = k

    def _post_update(self, info, ecm):
        busy = not (info.paused or self.finished or self.ending is not None)
        if busy:     # SkillSystem keeps only the run's max speed: let it find this step's max
            old, nan0 = self.vmax, self.nan
            self.vmax = (0.0, "", 0.0)
        super()._post_update(info, ecm)
        if not busy:
            return
        step_max = self.vmax
        self.vmax = step_max if step_max[0] > old[0] else old
        t = info.sim_time.total_seconds()
        n = int(round(t / DT)) - 1
        if 0 <= n <= self.n_total:
            v = self.ps["peak_body_speed_mps"][n] = step_max[0] if not (self.nan and not nan0) else math.inf
            c = self.timed.get("peak_body_speed_mps")
            if c is not None and ph.violates(c, M.value(c, {"link_speeds": [v]})) and c.metric not in self.first:
                self.first[c.metric] = n
        if self.bounds.phase == "open" and self.live_grasp_rel is not None and n % SLIP_EVERY == 0:
            slip = float(np.linalg.norm(self.handle_rel(ecm)[:3, 3] - self.live_grasp_rel[:3, 3]))
            self.slip["max"] = max(self.slip["max"], slip * 1000)
            if self.slip_cond is not None and self.slip["first"] is None and ph.violates(self.slip_cond, slip):
                self.slip["first"] = round(t, 3)
        while self.pending and self.last_stamp >= self.end_step(self.pending[0]) * DT + self.live.contact_wait_s:
            p = self.pending.pop(0)
            self.decided[p] = self.decide(p, live_t=t)

    # ------------------------------------------------------------- verdicts
    def end_step(self, p):
        i = self.order.index(p)
        return self.starts[self.order[i + 1]] if i + 1 < len(self.order) and self.order[i + 1] in self.starts \
            else self.n_total

    def decide(self, p, live_t=None):
        s, e = self.starts[p], self.end_step(p)
        checks = list(self.state_checks.get(p, []))
        if p in self.live_windows:
            a, b = self.live_windows[p]
            t0, t1 = (a + 1) * DT - DT / 2, b * DT + DT / 2
            on = [any(t0 <= x <= t1 for x in self.obj_contact.get((self.obj, f), [])) for f in self.fingers]
            c = next(c for c in self.conds[p] if c.metric == "finger_contacts")
            chk = self.judge(p, c.metric, c, dict(contact_left_finger_handle=[on[0]], contact_right_finger_handle=[on[1]],
                                                 window_s=[round(a * DT, 3), round(b * DT, 3)]), round(b * DT, 3))
            if chk is not None:
                chk["unit"] += f" on the handle over {round(a * DT, 3)}-{round(b * DT, 3)} s ([left, right] {on})"
                checks.append(chk)
        checks += ph.global_checks(p, s, e if p != self.order[-1] else self.n_total + 1, self.first, self.ps,
                                   list(self.timed.values()), DT)
        extra = dict(decided_sim_t=round(live_t, 3) if live_t is not None else None,
                     contact_stamp_at_decision=round(self.last_stamp, 3))
        if p in self.live_windows:
            extra["contact_window_s"] = [round(x * DT, 3) for x in self.live_windows[p]]
        sf = self.state_files.get(p)
        r = ph.phase_result(p, s, e, checks, DT, sf["file"] if sf else None, extra)
        if sf:
            r["start_state_info"] = sf
        if live_t is not None:
            bad = "; ".join(f"{c['name']} {c['value']}" for c in checks if not c["pass"])
            self.say(f"LIVE t={live_t:6.3f}s PHASE {p} {r['status'].upper()}"
                     + (f" (first failure {r['first_failure_t']} s: {bad})" if bad else "")
                     + f" [phase {r['start_t']}-{r['end_t']} s, contacts stamped to {self.last_stamp:.3f} s]")
        return r

    def check_restore(self, ecm):
        cap = self.state.capture(ecm, step=1)
        ss.save(self.state_dir / "restored_state.npz", cap)
        d_next, d_snap = ss.diff(self.restore_next, cap), ss.diff(self.restore_snap, cap)
        tol_m, tol_deg = self.live.restore_pos_max_m, self.live.restore_rot_max_deg
        ok = d_next["link_pos_m"][0] <= tol_m and math.degrees(d_next["link_rot_rad"][0]) <= tol_deg
        self.restore = dict(phase=self.R["phase"], source_action_index=self.offset,
                            vs_source_next_step=d_next, vs_boundary_snapshot=d_snap,
                            tolerance=dict(link_pos_m=tol_m, link_rot_deg=tol_deg), pass_=bool(ok),
                            max_link_rot_deg_vs_next=math.degrees(d_next["link_rot_rad"][0]),
                            max_link_rot_deg_vs_boundary=math.degrees(d_snap["link_rot_rad"][0]))
        self.say(f"LIVE restore check {'PASS' if ok else 'FAIL'}: vs the source state at the same instant "
                 f"{d_next['link_pos_m'][0] * 1000:.3f} mm ({d_next['link_pos_m'][1]}), "
                 f"{math.degrees(d_next['link_rot_rad'][0]):.4f} deg; vs the boundary snapshot "
                 f"{d_snap['link_pos_m'][0] * 1000:.3f} mm, {math.degrees(d_snap['link_rot_rad'][0]):.4f} deg")

    def report(self):
        for p in self.order:             # never-reached phases have no start
            if p in self.starts and p not in self.decided and p not in self.pending:
                self.pending.append(p)
        for p in list(self.pending):
            self.decided[p] = self.decide(p, live_t=self.t)
        self.pending = []
        final = {p: self.decide(p) for p in self.order if p in self.starts}
        phases = {}
        for p in self.order:
            if p not in final:
                phases[p] = dict(status="not_reached", failed_checks=["not_reached"], first_failure_t=None,
                                 checks=[], start_state=None)
                continue
            r = final[p]
            live = self.decided.get(p, r)
            r["live_status"], r["decided_sim_t"] = live["status"], live["decided_sim_t"]
            r["contact_stamp_at_decision"] = live["contact_stamp_at_decision"]
            r["live_matches_final"] = live["status"] == r["status"] and \
                live["failed_checks"] == r["failed_checks"]
            phases[p] = r
        untimed = []
        for c in self.r.task.sanity.untimed:
            if c.metric == "non_finite_commands":       # the gz log check is added by live_validate finalize
                r = M.evaluate(c, dict(commands=[math.nan] * len(self.ctl.bad)))
                untimed.append(ph.from_result(r, what="zeroed controller commands (untimed: no regeneration)"))
        out = dict(feed=self.L["feed"], label=self.L["label"], restored_from=self.R, steps=self.n_total,
                   boundaries={p: dict(step=s, action_index=s + self.offset, t=round(s * DT, 3))
                               for p, s in self.starts.items()},
                   phases=phases, global_checks_first_violation={k: round(v * DT, 3) for k, v in self.first.items()},
                   global_untimed=untimed, handle_slip_max_sampled_mm=round(self.slip["max"], 2),
                   task=self.r.task.name,
                   conditions={p: [c.model_dump(exclude_none=True) for c in cs] for p, cs in self.conds.items()},
                   sanity_timed=[c.model_dump(exclude_none=True) for c in self.timed.values()],
                   metric_states=self.mstates)
        if self.restore is not None:
            out["restore_check"] = self.restore
        (ars.REPLAY_DIR / "phase_live.json").write_text(json.dumps(out, indent=1, default=_json))
        if self.pol:
            self.policy_success()
        super().report()

    def policy_success(self):
        """closed_loop_system.report's success check on this run: hold_check.evaluate at every release of the
        gripper target after the object joint reached the threshold (and at the end while closed)."""
        import hold_check              # the closed-loop eval's module (policy mode only)
        S, L, g, n = self.pol["success"], self.log, self.seq.grip[:self.n_total], self.n_total
        hp = dict(S["check"], threshold=S["joint_min"], hold_s=S["hold_s"])
        lim, want, q = hp["closed_max_m"], S["joint_min"], L["door_pos"]
        qmax = np.maximum.accumulate(np.append(q[1:], q[-1]))        # the joint after step k
        cands = [k for k in range(1, n) if g[k - 1] <= lim < g[k] and qmax[k] >= want]
        if n >= 1 and g[n - 1] <= lim and qmax[n - 1] >= want:
            cands.append(n - 1)
        touched = lambda t0, t1: self.in_contact(self.ep["hold"]["obj"], t0, t1)
        res = None
        for r in cands:
            res = hold_check.evaluate(r, q, g, self.robot.gripper.openings(L["robot_joint_pos"][:, self.robot.grip_cols]),
                                      L["handle_in_tcp"], touched, hp, DT)
            if res["pass"]:
                break
        ok = bool(res and res["pass"])
        H = self.ep["hold"]
        unit = 180 / math.pi if H["unit"] == "deg" else 100.0
        k = int(np.argmax(q))
        msg = (f"{H['part']} >= {want * unit:.2f} {H['unit']} over the {S['hold_s']} s before a release, slip "
               f"{res['slip_mm'] if res else None} mm: "
               + ("pass" if ok else (f"not reached (max {q[k] * unit:.2f} {H['unit']} at t={(k + 1) * DT:.2f}s)"
                                     if not res else "fail " + ", ".join(x for x, v in res["parts"].items() if not v["pass"]))))
        self.results.append(dict(kind="closed_loop_success", name="closed loop success", pass_=ok, value=msg,
                                 t=round((res["release_step"] + 1) * DT if ok else n * DT, 3)))
        self.extra["closed_loop"] = dict(success=ok, hold_check=res, releases_checked=cands,
                                         joint_max=[float(q[k]), (k + 1) * DT])


def _json(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def get_system():
    return PhaseReplaySystem()
