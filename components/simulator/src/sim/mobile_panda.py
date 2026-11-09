"""Mobile Panda: Panda + Franka Hand on a virtual holonomic base.

The base is three joints from the world: prismatic base_x, prismatic base_y,
revolute base_yaw, then a box chassis (base_link) with panda_link0 fixed on
top at (MOUNT_X, 0, MOUNT_Z). scripts/gen_panda_sdf.py --mobile writes the
model (models/mobile_panda) from these constants.

Everything is commanded with efforts (no velocity commands, see README notes):
- base: P on (x, y, yaw) + feed-forward of the SDF joint damping, clamped in
  the controller (not by an SDF <effort> limit);
- arm and fingers: the can demo's laws (pick_place_system.arm_torques /
  finger_forces).

A robot command is one Action per 1 ms control step: base target (x, y, yaw)
and velocity, arm joint target and velocity, gripper opening target and max
finger force. Plan builds an ActionSeq from moves (drive, arm joints, IK
end-effector pose, gripper); an ActionSeq can be saved/loaded and replayed
step by step. Kinematics and Plan don't need Gazebo.
"""
import math
from dataclasses import dataclass

import numpy as np

import panda_plan as pp

MODEL = "mobile_panda"
MOUNT_Z = 0.60                       # panda_link0 above the floor [m]
MOUNT_X = 0.12                       # panda_link0 ahead of the base centre [m]
CHASSIS_SIZE = (0.50, 0.50, MOUNT_Z - 0.005)   # box, 5 mm above the floor
CHASSIS_MASS, VIRTUAL_MASS = 40.0, 1.0
BASE_JOINTS = ("base_x", "base_y", "base_yaw")
BASE_DAMPING = np.array([1000.0, 1000.0, 150.0])   # SDF <damping> [N s/m, N m s/rad]
BASE_KP = np.array([6000.0, 6000.0, 800.0])
BASE_F_MAX = np.array([1000.0, 1000.0, 300.0])
BASE_V_MAX, BASE_W_MAX = 0.4, 0.6    # planning limits (peak of min-jerk) [m/s, rad/s]
CONTROL_DT = 0.001                   # one Action per physics step [s]
PLAN_DT = 0.01                       # plan samples, upsampled to CONTROL_DT
GRIP_OPEN, GRIP_CLOSED = 0.04, 0.0   # per-finger opening [m]
GRIP_FORCE = 12.0                    # max squeeze per finger [N] (can demo)
READY = pp.READY
FINGER_MASS, FINGER_DAMPING = 0.0291, 2.0     # finger link mass [kg], SDF <damping> [N s/m]
FAST_EXPLICIT_MAX_DT = 0.006


def finger_forces_dt(fq, fv, grip, fmax, dt):
    """pick_place_system.finger_forces for a physics step dt > 1 ms (fast mode).

    The explicit common-mode PD (pair centring, 3000 N/m + 15 N s/m on a 29 g
    finger, anti-symmetric mode) is unstable above ~3 ms (free fingers threw the
    cup out at 4 ms). Up to FAST_EXPLICIT_MAX_DT the law stays explicit with the
    centring D dropped (the SDF damping is implicit) and the squeeze D cut where
    D*dt/m > 0.9 (6 ms): stable for the free fingers, and as stiff as the default
    against a held handle. Above, the centring force is the implicit
    (backward-Euler) PD of the anti-symmetric mode with the finger mass and SDF
    damping: stable at any dt, but it assumes free fingers, so it is softer than
    the default when the fingers hold something heavy (8 and 10 ms lose the
    handle). The squeeze stays explicit (it must reach the force cap against a
    blocked finger), its D capped for dt.
    """
    import pick_place_system as pps
    m, c = FINGER_MASS, FINGER_DAMPING
    if dt <= FAST_EXPLICIT_MAX_DT + 1e-9:
        off = (fq[0] - fq[1]) / 2
        f_center = -pps.CENTER_KP * off
        kd = pps.FINGER_KD if pps.FINGER_KD * dt / m <= 0.9 else 0.4 * m / dt
        sq = [max(-fmax, min(fmax, pps.FINGER_KP * (grip - q) - kd * v)) for q, v in zip(fq, fv)]
        return [sq[0] + f_center, sq[1] - f_center]
    kd = min(pps.FINGER_KD, 0.5 * m / dt)
    sq = [max(-fmax, min(fmax, pps.FINGER_KP * (grip - q) - kd * v)) for q, v in zip(fq, fv)]
    off, off_v = (fq[0] - fq[1]) / 2, (fv[0] - fv[1]) / 2
    K, D = pps.FINGER_KP + pps.CENTER_KP, kd + pps.CENTER_KD
    v1 = (m * off_v - dt * K * off) / (m + dt * dt * K + dt * (D + c))   # anti mode velocity next step
    f_anti = -K * (off + dt * v1) - D * v1
    f_center = f_anti - (-pps.FINGER_KP * off - kd * off_v)              # minus the squeeze's share
    return [sq[0] + f_center, sq[1] - f_center]


def _gravity_consts():
    Rj = np.array([T[:3, :3] for T in pp.JOINT_TF])
    pj = np.array([T[:3, 3] for T in pp.JOINT_TF])
    Th = pp.tf(p=(0, 0, pp.FLANGE_Z)) @ pp.tf(pp.rot("z", pp.HAND_YAW))
    hm = np.array([m for m, _ in pp.HAND_MASS_COM])
    hc = np.array([Th[:3, :3] @ np.array(c) + Th[:3, 3] for _, c in pp.HAND_MASS_COM])
    lm = np.array([m for m, _ in pp.LINK_MASS_COM])
    lc = np.array([c for _, c in pp.LINK_MASS_COM], dtype=float)
    return Rj, pj, lm, lc, hm, hc


_GRAV = None


def gravity_torques_fast(q):
    """pp.gravity_torques with fewer temporaries (~10x faster, equal to ~1e-14 N m).
    Fast mode only, so the default 1 ms path stays bit-identical."""
    global _GRAV
    if _GRAV is None:
        _GRAV = _gravity_consts()
    Rj, pj, lm, lc, hm, hc = _GRAV
    R, p = np.eye(3), np.array(pp.BASE, dtype=float)
    ax, org, mc = np.empty((7, 3)), np.empty((7, 3)), np.empty((8, 3))
    ms = np.empty(8)
    for i in range(7):
        p = p + R @ pj[i]
        R = R @ Rj[i]
        ax[i], org[i] = R[:, 2], p
        c, s_ = math.cos(q[i]), math.sin(q[i])
        R = R @ np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        mc[i], ms[i] = lm[i] * (R @ lc[i] + p), lm[i]
    hw = (R @ hc.T).T + p                      # hand + fingers, outboard of every joint
    mc[7], ms[7] = hm @ hw, hm.sum()
    Sm = np.cumsum(ms[::-1])[::-1][:7]          # mass outboard of joint i
    Smc = np.cumsum(mc[::-1], axis=0)[::-1][:7]
    d = Smc - Sm[:, None] * org                 # sum m (c - o_i)
    return pp.GRAVITY * (ax[:, 0] * d[:, 1] - ax[:, 1] * d[:, 0])


# ---------------------------------------------------------------- kinematics
def base_tf(base):
    """World pose of base_link for base = (x, y, yaw)."""
    return pp.tf(pp.rot("z", base[2]), (base[0], base[1], 0.0))


LINK0_IN_BASE = pp.tf(p=(MOUNT_X, 0.0, MOUNT_Z))


def link0_tf(base):
    return base_tf(base) @ LINK0_IN_BASE


def fk(q, base):
    """TCP pose in the world, joint axes and origins (world) for the Jacobian."""
    T = link0_tf(base)
    axes, origins = [], []
    for Tj, qi in zip(pp.JOINT_TF, q):
        T = T @ Tj
        axes.append(T[:3, 2].copy())
        origins.append(T[:3, 3].copy())
        T = T @ pp.tf(pp.rot("z", qi))
    return T @ pp.TCP_TF, np.array(axes), np.array(origins)


def ik(p_goal, R_goal, q, base, iters=300, margin=0.05):
    """Damped least squares with a step limit and a weak pull towards READY.
    Returns (q, position error [m], rotation error [rad])."""
    q = np.array(q, float)
    lo, hi = pp.LOWER + margin, pp.UPPER - margin
    for _ in range(iters):
        T, z, o = fk(q, base)
        e = np.concatenate([p_goal - T[:3, 3], 0.5 * pp.rotvec(R_goal @ T[:3, :3].T)])
        if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-4:
            break
        J = np.vstack([np.cross(z, T[:3, 3] - o).T, 0.5 * z.T])
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(6))
        dq = Jp @ e + (np.eye(7) - Jp @ J) @ (0.05 * (READY - q))
        n = np.linalg.norm(dq)
        if n > 0.2:
            dq *= 0.2 / n
        q = np.clip(q + dq, lo, hi)
    T, _, _ = fk(q, base)
    return q, np.linalg.norm(p_goal - T[:3, 3]), np.linalg.norm(pp.rotvec(R_goal @ T[:3, :3].T))


# Seeds with the hand pointing forward (q6 ~ 3), rolled 0 / +-90 deg: IK from
# READY (hand down) alone tends to end in a flipped wrist for horizontal grasps.
FORWARD_SEEDS = [np.array([0.0, q2, 0.0, q4, 0.0, q6, q7])
                 for q2, q4 in ((-0.3, -2.2), (0.3, -1.8)) for q6 in (2.6, 3.1)
                 for q7 in (0.785, 0.785 - math.pi / 2, 0.785 + math.pi / 2)]


def ik_best(p_goal, R_goal, base, seeds=None, tol=(1e-3, 1e-2)):
    """IK from several seeds; the converged solution closest to READY."""
    rng = np.random.default_rng(0)
    seeds = list(seeds or []) + [READY] + FORWARD_SEEDS + [
        np.clip(READY + rng.normal(0, 0.5, 7), pp.LOWER, pp.UPPER) for _ in range(8)]
    best = None
    for s in seeds:
        q, ep, er = ik(p_goal, R_goal, s, base)
        ok = ep < tol[0] and er < tol[1]
        # prefer joint-limit margin (up to 0.6 rad), then closeness to READY
        margin = np.min(np.minimum(q - pp.LOWER, pp.UPPER - q))
        key = (not ok, -min(margin, 0.6) + 0.05 * np.linalg.norm(q - READY) if ok else ep + er)
        if best is None or key < best[0]:
            best = (key, q, ep, er)
    return best[1], best[2], best[3]


def base_for_reach(p, heading, bearing=0.0, d=0.7):
    """Base pose (x, y, yaw) facing `heading` that puts the TCP target p at
    horizontal distance d from panda_link0, `bearing` [rad] left of the arm's
    forward axis. Horizontal grasps need |bearing| ~ 45 deg: straight ahead the
    hand can't point forward within joint 6's limit (q6 would be ~4.5 rad)."""
    a = heading + bearing
    l0 = np.asarray(p[:2], float) - d * np.array([math.cos(a), math.sin(a)])
    b = l0 - MOUNT_X * np.array([math.cos(heading), math.sin(heading)])
    return np.array([b[0], b[1], heading])


def hand_rot(approach, closing):
    """TCP orientation: hand z along the approach direction, fingers close along closing."""
    z = np.asarray(approach, float) / np.linalg.norm(approach)
    y = np.asarray(closing, float) / np.linalg.norm(closing)
    return np.column_stack([np.cross(y, z), y, z])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------- actions
@dataclass
class Action:
    """One control-step command."""
    base: np.ndarray        # (x, y, yaw) target
    base_vel: np.ndarray    # feed-forward velocity
    arm: np.ndarray         # 7 joint targets
    arm_vel: np.ndarray
    grip: float             # per-finger opening target [m]
    grip_force: float       # max squeeze per finger [N]


class ActionSeq:
    """Actions at CONTROL_DT, stored as arrays (replayable, saveable)."""
    FIELDS = ("base", "base_vel", "arm", "arm_vel", "grip", "grip_force")

    def __init__(self, **arrays):
        for k in self.FIELDS:
            setattr(self, k, np.asarray(arrays[k], float))

    def __len__(self):
        return len(self.grip)

    def __getitem__(self, k):
        k = min(max(int(k), 0), len(self) - 1)
        return Action(self.base[k], self.base_vel[k], self.arm[k], self.arm_vel[k],
                      float(self.grip[k]), float(self.grip_force[k]))

    def save(self, path):
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.FIELDS})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(**{k: d[k] for k in cls.FIELDS})


class Plan:
    """Builds an ActionSeq from moves. Each move starts from the last commanded
    state; the base stays where it was during arm moves (base, then arm)."""

    def __init__(self, base, q=READY, grip=GRIP_OPEN, grip_force=GRIP_FORCE):
        self.b = np.array(base, float)
        self.q = np.array(q, float)
        self.g, self.f = float(grip), float(grip_force)
        self.rows = []          # (base[3], q[7], grip, force) at PLAN_DT
        self.events = []        # (t, label, info)
        self.ik_worst = (0.0, 0.0)

    @property
    def t(self):
        return len(self.rows) * PLAN_DT

    def _push(self, n, label, info, fn):
        self.events.append((self.t, label, info or {}))
        for s in pp.min_jerk(max(2, int(round(n / PLAN_DT)))):
            fn(s)
            self.rows.append((self.b.copy(), self.q.copy(), self.g, self.f))

    def hold(self, dur, label="hold", info=None):
        self._push(dur, label, info, lambda s: None)

    def drive(self, goal, dur=None, label="drive", info=None):
        b0 = self.b.copy()
        goal = np.array(goal, float)
        goal[2] = b0[2] + wrap(goal[2] - b0[2])       # shortest turn
        d = goal - b0
        if dur is None:   # min-jerk peak speed = 1.875 * mean speed
            dur = max(3.0, 1.875 * max(np.linalg.norm(d[:2]) / BASE_V_MAX,
                                       abs(d[2]) / BASE_W_MAX))

        def step(s):
            self.b = b0 + s * d
        self._push(dur, label, dict(info or {}, goal=goal.tolist()), step)

    def base_path(self, path, dur, label="base_path", info=None):
        """Base along path(s) -> (x, y, yaw) for s in [0, 1] (min-jerk timing),
        arm joints held (e.g. the whole robot turning about a door hinge)."""
        def step(s):
            self.b = np.array(path(s), float)
        self._push(dur, label, info, step)

    def arm_joints(self, q_goal, dur, label="arm", info=None):
        q0, q_goal = self.q.copy(), np.array(q_goal, float)

        def step(s):
            self.q = q0 + s * (q_goal - q0)
        self._push(dur, label, info, step)

    def arm_pose(self, p, R, dur, label="arm_pose", cartesian=False, info=None):
        """TCP to (p, R) in the world. cartesian=False: IK at the goal, then
        joint-space min-jerk (large moves). cartesian=True: straight line with
        interpolated orientation, IK per sample seeded by the previous one."""
        p, R = np.asarray(p, float), np.asarray(R, float)
        info = dict(info or {}, p=p.tolist(), R=R.tolist())
        if not cartesian:
            q_goal, ep, er = ik_best(p, R, self.b, seeds=[self.q])
            self.ik_worst = (max(self.ik_worst[0], ep), max(self.ik_worst[1], er))
            info.update(ik_err=[ep, er], q_goal=q_goal.tolist())
            return self.arm_joints(q_goal, dur, label, info)
        T0, _, _ = fk(self.q, self.b)
        p0, R0 = T0[:3, 3].copy(), T0[:3, :3].copy()
        w = pp.rotvec(R @ R0.T)

        def step(s):
            a = np.linalg.norm(w) * s
            Rs = (_axis_angle(w, a) if a > 1e-9 else np.eye(3)) @ R0
            self.q, ep, er = ik(p0 + s * (p - p0), Rs, self.q, self.b, iters=50)
            self.ik_worst = (max(self.ik_worst[0], ep), max(self.ik_worst[1], er))
        self._push(dur, label, info, step)

    def arm_path(self, path, dur, label="arm_path", info=None, base=None):
        """TCP along path(s) -> (p, R) for s in [0, 1] (min-jerk timing),
        IK per sample seeded by the previous one (e.g. an arc around a hinge).
        base: optional base path(s) -> (x, y, yaw), moved together with the arm."""
        def step(s):
            if base is not None:
                self.b = np.array(base(s), float)
            p, R = path(s)
            self.q, ep, er = ik(np.asarray(p, float), np.asarray(R, float), self.q, self.b, iters=50)
            self.ik_worst = (max(self.ik_worst[0], ep), max(self.ik_worst[1], er))
        self._push(dur, label, info, step)

    def tcp(self):
        """Commanded TCP pose (world) at the end of the plan so far."""
        return fk(self.q, self.b)[0]

    def gripper(self, width, dur=1.5, force=None, label="gripper", info=None):
        """width: per-finger opening target [m] (GRIP_OPEN / GRIP_CLOSED)."""
        g0 = self.g
        self.f = GRIP_FORCE if force is None else float(force)

        def step(s):
            self.g = g0 + s * (width - g0)
        self._push(dur, label, info, step)

    def actions(self):
        """Upsample to CONTROL_DT (linear) and differentiate for feed-forward."""
        n = len(self.rows)
        B = np.array([r[0] for r in self.rows])
        Q = np.array([r[1] for r in self.rows])
        G = np.array([r[2] for r in self.rows])
        F = np.array([r[3] for r in self.rows])
        k = int(round(PLAN_DT / CONTROL_DT))
        x = np.arange(n * k) / k
        x = np.minimum(x, n - 1)

        def up(A):
            A = A.reshape(n, -1)
            return np.column_stack([np.interp(x, np.arange(n), A[:, i]) for i in range(A.shape[1])])
        Bd = np.gradient(B, PLAN_DT, axis=0)
        Qd = np.gradient(Q, PLAN_DT, axis=0)
        return ActionSeq(base=up(B), base_vel=up(Bd), arm=up(Q), arm_vel=up(Qd),
                         grip=up(G)[:, 0], grip_force=np.repeat(F, k))


def _axis_angle(w, a):
    u = w / np.linalg.norm(w)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


# ---------------------------------------------------------------- controller
class Controller:
    """In-loop effort controller for the mobile Panda (call apply() every step
    from a gz-sim system's pre_update)."""

    def __init__(self, model, ecm):
        from gz.sim import Joint, Link
        import pick_place_system as pps
        self._arm_torques, self._finger_forces = pps.arm_torques, pps.finger_forces
        jn = lambda n: Joint(model.joint_by_name(ecm, n))
        self.base = [jn(n) for n in BASE_JOINTS]
        self.arm = [jn(f"panda_joint{i}") for i in range(1, 8)]
        self.fingers = [jn(f"panda_finger_joint_{s}") for s in ("left", "right")]
        for j in self.base + self.arm + self.fingers:
            j.enable_position_check(ecm)
            j.enable_velocity_check(ecm)
        self.names = list(BASE_JOINTS) + [f"panda_joint{i}" for i in range(1, 8)] + [
            "panda_finger_joint_left", "panda_finger_joint_right"]
        self.bad = []           # non-finite commands (zeroed instead): (joint, value, state)
        self.dt = CONTROL_DT    # physics step; > CONTROL_DT selects finger_forces_dt
        self.hand = Link(model.link_by_name(ecm, "panda_hand"))
        self.base_link = Link(model.link_by_name(ecm, "base_link"))

    def reset(self, ecm, base, q=READY, grip=GRIP_OPEN):
        for j, v in zip(self.base + self.arm + self.fingers,
                        list(base) + list(q) + [grip, grip]):
            j.reset_position(ecm, [float(v)])

    @staticmethod
    def _read(joints, ecm):
        return (np.array([(j.position(ecm) or [0.0])[0] for j in joints]),
                np.array([(j.velocity(ecm) or [0.0])[0] for j in joints]))

    def state(self, ecm):
        """Joint-space state (base pose = base joint values: model at the origin)."""
        b, bv = self._read(self.base, ecm)
        q, qv = self._read(self.arm, ecm)
        f, fv = self._read(self.fingers, ecm)
        return dict(base=b, base_vel=bv, arm=q, arm_vel=qv, fingers=f, fingers_vel=fv)

    def apply(self, ecm, a, meas=None):
        """meas: measured state to use instead of the ECM's (a dict as from
        state(); e.g. a snapshot being restored in this same step)."""
        m = meas or self.state(ecm)
        b, q, fq, fv = m["base"], m["arm"], m["fingers"], m["fingers_vel"]
        e = a.base - b
        e[2] = wrap(e[2])
        fb = np.clip(BASE_KP * e + BASE_DAMPING * a.base_vel, -BASE_F_MAX, BASE_F_MAX)
        ff = (self._finger_forces(fq, fv, a.grip, a.grip_force) if self.dt <= CONTROL_DT
              else finger_forces_dt(fq, fv, a.grip, a.grip_force, self.dt))
        if self.dt <= CONTROL_DT:
            tau = self._arm_torques(q, a.arm, a.arm_vel)
        else:
            import pick_place_system as pps
            tau = np.clip(pps.ARM_KP * (a.arm - q) + pps.ARM_DAMPING * a.arm_vel + gravity_torques_fast(q),
                          -pps.ARM_TAU_MAX, pps.ARM_TAU_MAX)
        u = np.concatenate([fb, tau, ff])
        # The plugin writes force commands to MuJoCo ctrl unchecked
        for i in np.flatnonzero(~np.isfinite(u)):
            if len(self.bad) < 20:
                self.bad.append((self.names[i], float(u[i]), [*b, *q, *fq]))
            u[i] = 0.0
        for j, ui in zip(self.base + self.arm + self.fingers, u):
            j.set_force(ecm, [float(ui)])
