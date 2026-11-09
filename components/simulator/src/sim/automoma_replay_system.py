"""gz-sim Python system: replay one AutoMoMa plan in the kitchen and log every 1 ms step.

Ported from the earlier pipeline's simulator plugin (unchanged but for a scripted episode's spans).
Loaded by the replay world copy that cli.py writes (module_name swapped).
$REPLAY_DIR holds actions.npz and episode.json from replay.py. The robot starts at the
first action (joint reset at configure). Everything but the input, the spawn and the log is
the scripted episode system (mobile_panda_episode_system, reused read-only):
Controller.apply every step, check windows from the episode's events, the grasped /
door_open or drawer_open / sim-time checks, physics and contact checks, "DONE PASS/FAIL" on
/mobile_panda_skill/status. Thresholds and the task object's joint and handle (door_* arrays,
handle contacts) come from episode.json (the plugin data).

Writes $REPLAY_DIR/state_log.npz before the DONE line: one row per physics step n (0-based),
the state at the start of the step (time_s = n * dt), the Action and joint efforts applied
during it, and the contacts the step produced.

A recovery replay ($REPLAY_RECOVERY_SEED set, bands from $ROUND_RECOVERY) adds seeded DART-style
noise to the executed base and arm targets (recovery_noise). The action_* arrays stay the clean
expert targets of actions.npz; the log adds noise_base (N, 3), noise_arm (N, 7) and recovery_seed,
and the executed target of step n is action_*[n] + noise_*[n]. Without the seed nothing changes.
"""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import mobile_panda as mp
import mobile_panda_episode_system as eps
import mobile_panda_skill_system as sks
import panda_plan as pp

REPLAY_DIR = Path(os.environ.get("REPLAY_DIR", "."))
RECOVERY_SEED = os.environ.get("REPLAY_RECOVERY_SEED")
# The noise spans the first NOISE_SPAN of the approach ("approach in" to "grip close"): the open-loop
# plan then pulls the robot back to its own targets before the fingers reach the handle, so the
# grasp and the pull stay the nominal ones. Each ramp takes NOISE_RAMP of that span.
NOISE_SPAN, NOISE_RAMP = 0.6, 0.3


def recovery_noise(n, k0, k1, block, seed):
    """Target noise of a recovery replay from its seed alone (check and record agree): one offset of
    the base (x, y, yaw) and of the 7 arm joints, flat in the middle of steps [k0, k0 + NOISE_SPAN
    (k1 - k0)) with raised-cosine ramps in and out, zero elsewhere. Peaks (RCA item 29): planar base
    U[0.01, base_m] m in a uniform direction, yaw +-U[1, yaw_deg] deg, each joint U[-arm_rad, arm_rad].
    -> noise_base (n, 3) [m, m, rad], noise_arm (n, 7) [rad]."""
    rng = np.random.default_rng(seed)
    r, phi = rng.uniform(0.01, block["base_m"]), rng.uniform(0.0, 2 * np.pi)
    yaw = rng.choice((-1.0, 1.0)) * np.radians(rng.uniform(1.0, block["yaw_deg"]))
    arm = rng.uniform(-block["arm_rad"], block["arm_rad"], 7)
    w = int(NOISE_SPAN * (k1 - k0))
    m = int(NOISE_RAMP * w)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(m) / m)
    prof = np.zeros(n)
    prof[k0:k0 + m], prof[k0 + m:k0 + w - m], prof[k0 + w - m:k0 + w] = ramp, 1.0, ramp[::-1]
    return prof[:, None] * np.array([r * np.cos(phi), r * np.sin(phi), yaw]), prof[:, None] * arm


class _ForceTap:
    """Joint stand-in that keeps the force Controller.apply last commanded (for the log)."""

    def __init__(self, joint, buf, i):
        self._j, self._buf, self._i = joint, buf, i

    def __getattr__(self, name):
        return getattr(self._j, name)

    def set_force(self, ecm, f):
        self._buf[self._i] = f[0]
        self._j.set_force(ecm, f)


def _pose7(pose):
    p, q = pose.pos(), pose.rot()
    return p.x(), p.y(), p.z(), q.w(), q.x(), q.y(), q.z()


class ReplaySystem(eps.EpisodeSystem):
    def make_scenario(self, name):
        ep = self.ep = json.loads((REPLAY_DIR / "episode.json").read_text())
        th = ep["thresholds"]
        sks.SLIP_TOL, sks.MAX_SPEED, sks.MAX_PEN = (th["handle_slip_max_m"], th["peak_body_speed_max_mps"],
                                                    th["penetration_max_m"])
        seq = mp.ActionSeq.load(REPLAY_DIR / "actions.npz")
        P = SimpleNamespace(events=[tuple(e) for e in ep["events"]], t=len(seq) * mp.CONTROL_DT,
                            ik_worst=(0.0, 0.0), actions=lambda: seq)
        return SimpleNamespace(P=P, checks=[tuple(c) for c in ep["checks"]], info=dict(budget=ep["budget_s"]),
                               spans=ep.get("spans", []))     # a scripted episode's spans (live_validate.py)

    def configure(self, entity, sdf, ecm, event_mgr):
        super().configure(entity, sdf, ecm, event_mgr)
        c, n = self.ctl, self.n_total
        a = self.seq[0]                 # spawn at the first action, not SkillSystem's BASE0 / READY
        c.reset(ecm, a.base, a.arm, a.grip)
        self.force = np.zeros(12)
        taps = [_ForceTap(j, self.force, i) for i, j in enumerate(c.base + c.arm + c.fingers)]
        c.base, c.arm, c.fingers = taps[:3], taps[3:10], taps[10:]
        self.door = dict(self.state.joints)[self.ep["object_joint"]]
        z = lambda *s: np.zeros((n, *s))
        self.log = dict(robot_joint_pos=z(12), robot_joint_vel=z(12), robot_joint_effort=z(12),
                        door_pos=z(), door_vel=z(), base_link_pose=z(7), tcp_pose=z(7))
        self.contact_msgs = []          # (step, robot link, left on handle, right on handle, other, depth)
        self.wall = [None, None]        # wall time at the first and the last logged step
        self.noise = None               # (noise_base, noise_arm) of a recovery replay
        if RECOVERY_SEED is not None:
            start = {lab: k for k, lab in self.ev_steps}
            self.noise = recovery_noise(len(self.seq), start["approach in"], start["grip close"],
                                        json.loads(os.environ["ROUND_RECOVERY"]), int(RECOVERY_SEED))

    def pre_update(self, info, ecm):
        # SkillSystem.pre_update minus the cup / restore hooks (unused here), with the
        # controller's state read once and logged
        if info.paused:
            return
        n = int(round(info.sim_time.total_seconds() / mp.CONTROL_DT)) - 1
        a = min(n, len(self.seq) - 1)
        m = self.measured(ecm)
        act = self.seq[a]
        if self.noise is not None:      # the executed target; the log keeps the clean one
            act = mp.Action(act.base + self.noise[0][a], act.base_vel, act.arm + self.noise[1][a], act.arm_vel,
                            act.grip, act.grip_force)
        self.ctl.apply(ecm, act, m)
        if n >= self.n_total:
            return
        self.applied[n] = a
        L = self.log
        L["robot_joint_pos"][n] = (*m["base"], *m["arm"], *m["fingers"])
        L["robot_joint_vel"][n] = (*m["base_vel"], *m["arm_vel"], *m["fingers_vel"])
        L["robot_joint_effort"][n] = self.force
        L["door_pos"][n] = (self.door.position(ecm) or [0.0])[0]
        L["door_vel"][n] = (self.door.velocity(ecm) or [0.0])[0]
        L["base_link_pose"][n] = _pose7(self.ctl.base_link.world_pose(ecm))
        h = _pose7(self.ctl.hand.world_pose(ecm))
        w, x, y, zq = h[3:]
        hz = np.array([2 * (x * zq + w * y), 2 * (y * zq - w * x), 1 - 2 * (x * x + y * y)])
        L["tcp_pose"][n] = (*(np.array(h[:3]) + pp.TCP_Z * hz), *h[3:])
        self.wall[0 if n == 0 else 1] = time.time()

    def measured(self, ecm):
        """The robot state the controller acts on and the log records (a subclass restoring a
        snapshot in this step returns the snapshot's: the ECM still holds the old state)."""
        return self.ctl.state(ecm)

    def on_contacts(self, msg):
        super().on_contacts(msg)
        st = msg.header.stamp
        k = int(round((st.sec + st.nsec * 1e-9) / mp.CONTROL_DT)) - 1
        link, left, right, other, depth = None, False, False, 0, 0.0
        for c in msg.contact:
            names = (c.collision1.name, c.collision2.name)
            rob = [s for s in names if s.startswith(mp.MODEL + "::")]
            if not rob:
                continue
            link = rob[0].split("::")[1]
            oth = [s for s in names if not s.startswith(mp.MODEL + "::")]
            on_handle = bool(oth) and oth[0].startswith(self.ep["object_handle"])
            left |= on_handle and link == "panda_leftfinger"
            right |= on_handle and link == "panda_rightfinger"
            other += not (on_handle and link in sks.FINGERS)
            depth = max(depth, max(c.depth) if len(c.depth) else 0.0)
        if link is not None:
            self.contact_msgs.append((k, link, left, right, other, depth))

    def save_log(self):
        n, ep = self.n_total, self.ep
        starts = np.array([k for k, _ in self.ev_steps])
        event = np.searchsorted(starts, np.arange(n), side="right") - 1
        sub_idx = np.array([ep["subtasks"].index(s) for s in ep["event_subtask"]])
        # a contact sensor message stands for the steps until the next one (the Contact system
        # publishes only while a contact exists, so no message = no contact)
        hold = max(1, int(round(1.0 / (ep["contact_rate_hz"] * mp.CONTROL_DT))))
        cl, cr = np.zeros(n, bool), np.zeros(n, bool)
        co, cd = np.zeros(n, np.int16), np.zeros(n, np.float32)
        per_link = {}
        for k, link, left, right, other, depth in sorted(self.contact_msgs, key=lambda m: m[0]):
            per_link.setdefault(link, []).append((k, left, right, other, depth))
        for msgs in per_link.values():
            L, R, O, D = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, np.int16), np.zeros(n, np.float32)
            for k, left, right, other, depth in msgs:
                s = slice(max(k, 0), max(min(k + hold, n), 0))
                L[s], R[s], O[s], D[s] = left, right, other, depth
            cl |= L
            cr |= R
            co += O
            cd = np.maximum(cd, D)
        seq = self.seq
        recovery = {} if self.noise is None else dict(
            noise_base=self.noise[0][:n], noise_arm=self.noise[1][:n], recovery_seed=np.int64(RECOVERY_SEED))
        np.savez_compressed(
            REPLAY_DIR / "state_log.npz", **self.log,
            time_s=np.arange(n) * mp.CONTROL_DT, control_dt=mp.CONTROL_DT,
            robot_joint_names=np.array(self.ctl.names), door_joint=self.ep["object_joint"],
            action_base=seq.base[:n], action_base_vel=seq.base_vel[:n], action_arm=seq.arm[:n],
            action_arm_vel=seq.arm_vel[:n], action_grip=seq.grip[:n], action_grip_force=seq.grip_force[:n],
            event=event.astype(np.int16), event_labels=np.array([lab for _, lab in self.ev_steps]),
            event_start_step=starts, subtask=sub_idx[event].astype(np.int8),
            subtask_labels=np.array(ep["subtasks"]),
            contact_left_finger_handle=cl, contact_right_finger_handle=cr, contact_other_robot=co,
            contact_robot_max_depth_m=cd, plan_run=ep["plan_run"], attempt_id=ep["attempt_id"], **recovery)

    def report(self):
        t0 = time.time()
        self.save_log()
        wall = (self.wall[1] - self.wall[0]) if None not in self.wall else None
        sim = self.n_total * mp.CONTROL_DT
        self.extra["replay"] = dict(
            plan_run=self.ep["plan_run"], attempt_id=self.ep["attempt_id"], steps=self.n_total, sim_s=sim,
            loop_wall_s=round(wall, 2) if wall else None, rtf=round(sim / wall, 3) if wall else None,
            real_time_factor_cap=float(os.environ.get("REPLAY_RTF", "1.0")),
            contact_msgs=len(self.contact_msgs), state_log_write_s=round(time.time() - t0, 2))
        super().report()


def get_system():
    return ReplaySystem()
