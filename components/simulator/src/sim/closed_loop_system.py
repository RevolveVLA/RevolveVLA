"""gz-sim Python system: the closed loop (a policy rollout). The robot follows whole-body targets streamed
over gz-transport (no ROS in the 1 kHz loop) while the live phase validator (phase_replay_system, policy
mode: phases.PolicyBoundaries, the task.yaml gates) judges every phase and saves its start state.
Ported from the earlier pipeline's closed_loop_system.py; the base class is now
PhaseReplaySystem (was automoma_replay_system.ReplaySystem), so phase verdicts, phase_states/ and
phase_live.json come out of the rollout itself instead of a post-hoc replay (legacy policy_episode.py).

Loaded by the world copy cli.py rollout writes (module_name swapped). $CL_DIR (= $REPLAY_DIR) holds
closed_loop.json (cli.py rollout_spec: spawn, the task object's joint and handle, thresholds, limits,
budget) and live.json (policy mode; "restore" for a retry from a saved phase start state, restored in
step 0 with the spawn = that state's commanded action). Everything else is the base classes'
(controller, physics and contact checks, per-step state_log.npz, DONE line).

  in   /closed_loop/action  gz.msgs.Actuators, position = the robot's action (robot.yaml joints.action):
       base x, y, yaw (world frame, yaw unwrapped), the arm joints, gripper (opening per contact link, m);
       sent at the contract rate (30 Hz sim time). The newest one is the target until the next one: between
       ticks, on a stall and after the source stops.
  out  /closed_loop/state   gz.msgs.Actuators every 1 ms step, header stamp n x dt: position and velocity
       of the robot joints at the start of step n (state_log robot_joint_names order).

Target -> Action: clamped to the joint and base limits; feed-forward velocity = (new - previous
target) / max(their sim spacing, 1 / fps), clamped to the velocity limits, zeroed once the target is
older than 2 ticks; grip force = the robot's gripper.force_n (every plan's). Before the first action the
spawn is the target. Non-finite or wrong-length messages are counted and dropped.

Success (hold_check.py): the expert's hold check on plan-free windows, evaluated at every release of the
gripper target once the object joint has reached the threshold; the first passing release is the success.
It only ends the run (after_success_s later, the release phase): the verdict is the phase report, and the
final hold check (closed_loop_success in summary.json) is PhaseReplaySystem.policy_success's. The run ends
budget_s of sim after the first action, after_success_s after a success (if set), on a physics abort (NaN,
peak body speed over the limit) or wait_s without any action.
Writes state_log.npz (the replay arrays; action_* = the applied targets; action_index = the received action
in effect, -1 before the first; rx_step, rx_values = every accepted message; handle_in_tcp = the handle
link's origin in the TCP frame) and actions.npz (the applied Actions, so a closed-loop run replays with
automoma_replay_system). DONE PASS = success and every physics and contact check passes (the expert's
"accepted").
"""
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from gz.msgs.actuators_pb2 import Actuators

import hold_check
import phase_replay_system as prs
import phases as ph
import robots
import skill_system as sks
import smoke_system as smoke
from robots.base import CONTROL_DT, ActionSeq

CL_DIR = Path(os.environ.get("CL_DIR", "."))
ACTION_TOPIC, STATE_TOPIC = "/closed_loop/action", "/closed_loop/state"
DT = CONTROL_DT


class ClosedLoopSystem(prs.PhaseReplaySystem):
    def make_scenario(self, name):
        ep = self.ep = json.loads((CL_DIR / "closed_loop.json").read_text())
        th = ep["thresholds"]
        sks.SLIP_TOL, sks.MAX_SPEED, sks.MAX_PEN = (th["handle_slip_max_m"], th["peak_body_speed_max_mps"],
                                                    th["penetration_max_m"])
        n = int(round((ep["wait_s"] + ep["budget_s"]) / DT)) + 1
        sp, rb = ep["spawn"], robots.current()
        self.n_action, self.n_ff = len(rb.action_names), 3 + rb.dof     # whole-body action, its base + arm part
        rows = lambda v: np.tile(np.asarray(v, float), (n, 1))
        # rows are overwritten step by step with the applied targets (pre_update)
        seq = ActionSeq(base=rows(sp["base"]), base_vel=np.zeros((n, 3)), arm=rows(sp["arm"]),
                        arm_vel=np.zeros((n, rb.dof)), grip=np.full(n, sp["grip"]),
                        grip_force=np.full(n, rb.gripper.force_n))
        P = SimpleNamespace(events=[(0.0, "closed loop", {})], t=n * DT, ik_worst=(0.0, 0.0), actions=lambda: seq)
        return SimpleNamespace(P=P, checks=[], info=dict(budget=n * DT), spans=[])

    def configure(self, entity, sdf, ecm, event_mgr):
        super().configure(entity, sdf, ecm, event_mgr)
        ep, n = self.ep, self.n_total
        lim = ep["limits"]
        self.fps = float(ep["fps"])
        self.lo, self.hi, self.vlim = (np.array(lim[k], float) for k in ("lower", "upper", "velocity"))
        self.log["action_index"] = np.full(n, -1, np.int32)
        self.inbox, self.rejected = None, 0     # newest message from the transport thread: (count, values)
        self.rx_count, self.taken = 0, 0
        self.rx_step, self.rx_values = [], []
        self.target = None                      # (values, step it took effect, feed-forward)
        self.first_step = self.n_end = self.end_reason = None
        H = ep["success"]
        self.want, self.hold_s = H["joint_min"], H["hold_s"]
        self.hp = dict(H["check"], threshold=H["joint_min"], hold_s=H["hold_s"])
        self.log["handle_in_tcp"] = np.zeros((n, 3))
        self.run = dict(success_t=None, success=None, last=None, releases=[], max=(-math.inf, 0.0))
        self.state_pub = self.node.advertise(STATE_TOPIC, Actuators)
        if not self.node.subscribe(Actuators, ACTION_TOPIC, self.on_action):
            self.say(f"could not subscribe to {ACTION_TOPIC}")
        self.say(f"closed loop: spawn {np.round(ep['spawn']['base'], 3).tolist()}, budget {ep['budget_s']} s from "
                 f"the first action, success {ep['success']}, after success {ep['after_success_s']} s")

    # -------------------------------------------------------------- input
    def on_action(self, msg):
        v = np.array(msg.position, float)
        if v.shape != (self.n_action,) or not np.all(np.isfinite(v)):
            self.rejected += 1
            return
        self.rx_count += 1
        self.inbox = (self.rx_count, v)

    def set_target(self, n):
        """Row min(n, last) of self.seq = the Action for step n from the newest accepted message."""
        msg = self.inbox
        if msg is not None and msg[0] != self.taken and self.ending is None:
            self.taken = msg[0]
            v = np.clip(msg[1], self.lo, self.hi)
            k = self.n_ff
            if self.target is None:
                ff = np.zeros(k)
                self.first_step = n
                self.sc.info["budget"] = (n + 1) * DT + self.ep["budget_s"]
            else:
                spacing = max((n - self.target[1]) * DT, 1.0 / self.fps)
                ff = np.clip((v[:k] - self.target[0][:k]) / spacing, -self.vlim, self.vlim)
            self.target = (v, n, ff)
            self.rx_step.append(n)
            self.rx_values.append(v)
        if self.target is None:
            return
        v, n0, ff = self.target
        k = self.n_ff
        if (n - n0) * DT > 2.0 / self.fps + 1e-9:
            ff = np.zeros(k)
        s, r = self.seq, min(n, len(self.seq) - 1)
        s.base[r], s.base_vel[r], s.arm[r], s.arm_vel[r], s.grip[r] = v[:3], ff[:3], v[3:k], ff[3:k], v[k]
        if n < self.n_total:
            self.log["action_index"][n] = len(self.rx_step) - 1

    def pre_update(self, info, ecm):
        if info.paused:
            return
        n = int(round(info.sim_time.total_seconds() / DT)) - 1
        self.set_target(n)
        super().pre_update(info, ecm)           # applies row n, logs the state at the start of step n
        if n < self.n_total:
            self.log["handle_in_tcp"][n] = self.handle_in_tcp(ecm)[:3, 3]
            msg = Actuators()
            ns = n * 1_000_000
            msg.header.stamp.sec, msg.header.stamp.nsec = ns // 10**9, ns % 10**9
            msg.position.extend(self.log["robot_joint_pos"][n])
            msg.velocity.extend(self.log["robot_joint_vel"][n])
            self.state_pub.publish(msg)

    # ------------------------------------------------------------ success
    def handle_in_tcp(self, ecm):
        return np.linalg.inv(self.tcp(ecm)) @ smoke.gz_mat(self.obj_links[self.ep["hold"]["obj"]].world_pose(ecm))

    def touched(self, t0, t1):
        return self.in_contact(self.ep["hold"]["obj"], t0, t1)

    def hold_eval(self, r):
        L, g = self.log, self.seq.grip
        rb = self.robot
        return hold_check.evaluate(r, L["door_pos"], g, rb.gripper.openings(L["robot_joint_pos"][:, rb.grip_cols]),
                                   L["handle_in_tcp"], self.touched, self.hp, DT)

    def _post_update(self, info, ecm):
        busy = not (info.paused or self.finished or self.ending is not None)
        super()._post_update(info, ecm)
        if not busy or self.ending is not None:
            return
        t = info.sim_time.total_seconds()
        n = int(round(t / DT)) - 1
        q = (self.door.position(ecm) or [0.0])[0]
        R = self.run
        if q > R["max"][0]:
            R["max"] = (q, t)
        g, lim = self.seq.grip, self.hp["closed_max_m"]
        if (R["success_t"] is None and R["max"][0] >= self.want and 0 < n < self.n_total
                and g[n - 1] <= lim < g[n]):   # a release: the hold check at step n
            R["releases"].append(n)
            R["last"] = self.hold_eval(n)
            if R["last"]["pass"]:
                R["success_t"], R["success"] = t, R["last"]
                self.say(f"t={t:6.2f}s SUCCESS hold check at the release, slip {R['last']['slip_mm']} mm, hold min "
                         f"{R['last']['hold_min']:.4f}")
        after = self.ep["after_success_s"]
        if self.nan or self.vmax[0] >= sks.MAX_SPEED:
            reason = "abort"
        elif R["success_t"] is not None and after is not None and t >= R["success_t"] + after - 1e-9:
            reason = "success"
        elif self.first_step is not None and n >= self.first_step + round(self.ep["budget_s"] / DT):
            reason = "budget"
        elif self.first_step is None and t >= self.ep["wait_s"]:
            reason = "no_action"
        else:
            return
        self.end_reason, self.n_end = reason, n + 1
        self.unfinished(n + 1, t, ecm)
        self.snaps["final"] = self.state.capture(ecm, step=n, skill=self.skill)
        self.ending, self.ending_wall = t, time.time()

    def unfinished(self, e, t, ecm=None):
        """The phase still running when the rollout ends fails (a policy that never closes, pulls or releases):
        its state gates on the final state (when the ECM is at hand) + a failing phase_ended check. The last
        phase (release) ends with the stream."""
        cur = self.bounds.phase
        if cur == self.order[-1] or any(c["name"] == "phase_ended" for c in self.state_checks.get(cur, [])):
            return
        if ecm is not None:
            self.end_of(cur, e, ecm)
        nxt = self.order[self.order.index(cur) + 1]
        self.state_checks.setdefault(cur, []).append(ph.check(
            "phase_ended", False, f"no {cur} -> {nxt} boundary before the {self.end_reason} end", "eq", True,
            t_fail=round(t, 3), note="policy rollout: the phase did not end within the rollout budget"))

    def report(self):
        n = self.n_end or self.n_total
        self.end_reason = self.end_reason or "steps"
        self.unfinished(n, n * DT)
        gaps = np.diff(self.rx_step) * DT if len(self.rx_step) > 1 else np.zeros(1)
        t0 = (self.first_step + 1) * DT if self.first_step is not None else None
        self.extra["rollout"] = dict(
            end_reason=self.end_reason, steps=n, sim_s=round(n * DT, 3), first_action_t=t0,
            last_action_t=(self.rx_step[-1] + 1) * DT if self.rx_step else None,
            live_success_t=self.run["success_t"], releases_seen=self.run["releases"],
            joint_max=[self.run["max"][0], self.run["max"][1]], joint_final=float(self.log["door_pos"][n - 1]),
            actions_accepted=len(self.rx_step), actions_received=self.rx_count, actions_rejected=self.rejected,
            largest_action_gap_s=round(float(gaps.max()), 4), fps=self.fps)
        self.say(f"closed loop {json.dumps(self.extra['rollout'], default=float)}")
        # truncate to the steps that ran; actions.npz and state_log.npz then hold the applied targets
        self.n_total = n
        self.applied = np.arange(n)
        self.log = {k: v[:n] for k, v in self.log.items()}
        self.log["rx_step"] = np.array(self.rx_step, np.int64)
        self.log["rx_values"] = np.array(self.rx_values, float).reshape(-1, self.n_action)
        super().report()                        # phase verdicts, policy_success, state_log, actions, summary, DONE


def get_system():
    return ClosedLoopSystem()
