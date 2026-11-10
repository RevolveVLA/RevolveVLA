"""Mobile Panda: the Franka Panda (FER) + Franka Hand on the shared virtual holonomic base
(specs/robots/mobile_panda/robot.yaml; robots/base.py has the base, Plan and Controller).

Kinematics, masses and limits from franka_description (Apache-2.0): robots/fer/kinematics.yaml,
inertials.yaml, end_effectors/franka_hand/inertials.yaml. Laws (ported unchanged from the can-on-plate
demo, can_demo/): arm torques = P on the target + gravity compensation + feed-forward of the SDF joint
damping (MuJoCo integrates it implicitly; an explicit D gain went unstable near q2 = 0); fingers
force-controlled, each squeezing with a capped PD, a stiff common-mode PD keeping the pair centred on
the hand like the Franka hand's mechanically coupled fingers.
"""
import math

import numpy as np

from robots.base import CONTROL_DT, Gripper, Robot, rot, rotvec, tf

# franka_description robots/fer/kinematics.yaml: (xyz, rpy) per joint
JOINTS = [
    ((0, 0, 0.333), (0, 0, 0)),
    ((0, 0, 0), (-math.pi / 2, 0, 0)),
    ((0, -0.316, 0), (math.pi / 2, 0, 0)),
    ((0.0825, 0, 0), (math.pi / 2, 0, 0)),
    ((-0.0825, 0.384, 0), (-math.pi / 2, 0, 0)),
    ((0, 0, 0), (math.pi / 2, 0, 0)),
    ((0.088, 0, 0), (math.pi / 2, 0, 0)),
]
LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
FLANGE_Z, HAND_YAW = 0.107, -math.pi / 4

# Link masses and centres of mass (link frame), from franka_description
# robots/fer/inertials.yaml and end_effectors/franka_hand/inertials.yaml.
LINK_MASS_COM = [
    (2.7907, (0.00033, -0.02204, -0.04762)),
    (2.5420, (0.00038, -0.09211, 0.01908)),
    (2.2513, (0.05152, 0.01696, -0.02971)),
    (2.2037, (-0.05113, 0.05825, 0.01698)),
    (2.2855, (-0.00005, 0.03730, -0.09280)),
    (1.353, (0.06572, -0.00371, 0.00153)),
    (0.35973, (0.00089, -0.00044, 0.05491)),
]
# hand (0.6544 kg) + both fingers (2 x 0.0291 kg), in the hand frame
HAND_MASS_COM = [(0.6544, (-0.0000376, 0.0119128, 0.020726)),
                 (0.0582, (0.0, 0.0, 0.080))]
GRAVITY = 9.81
# gravity_torques() chains the arm from this point (the can demo's arm base): the torques do not depend
# on it, but the float operations do, and the 1 ms law has to stay bit-identical to the recorded runs.
GRAVITY_ORIGIN = np.array([0.0, 0.0, 0.40])

JOINT_TF = [tf(rot("z", r[2]) @ rot("y", r[1]) @ rot("x", r[0]), p)
            for p, r in JOINTS]

FINGER_KP, FINGER_KD = 1000.0, 5.0
CENTER_KP, CENTER_KD = 2000.0, 10.0   # common-mode (pair stays centred)
FINGER_MASS, FINGER_DAMPING = 0.0291, 2.0     # finger link mass [kg], SDF <damping> [N s/m]
FAST_EXPLICIT_MAX_DT = 0.006


def tcp_tf(tcp_z):
    """The TCP in the panda_link7 frame: flange, hand yaw, TCP offset along the hand z."""
    return tf(p=(0, 0, FLANGE_Z)) @ tf(rot("z", HAND_YAW)) @ tf(p=(0, 0, tcp_z))


def gravity_torques(q):
    """Joint torques that hold the arm (incl. hand + fingers) against gravity."""
    T = tf(p=GRAVITY_ORIGIN)
    axes, origins, coms = [], [], []
    for Tj, qi, (m, c) in zip(JOINT_TF, q, LINK_MASS_COM):
        T = T @ Tj
        axes.append(T[:3, 2].copy())
        origins.append(T[:3, 3].copy())
        T = T @ tf(rot("z", qi))
        coms.append((m, (T @ np.array([*c, 1.0]))[:3]))
    Th = T @ tf(p=(0, 0, FLANGE_Z)) @ tf(rot("z", HAND_YAW))
    coms += [(m, (Th @ np.array([*c, 1.0]))[:3]) for m, c in HAND_MASS_COM]
    tau = np.zeros(7)
    for i in range(7):
        for k, (m, c) in enumerate(coms):
            if k >= i:     # bodies outboard of joint i
                tau[i] += m * GRAVITY * np.cross(axes[i], c - origins[i])[2]
    return tau


def _gravity_consts():
    Rj = np.array([T[:3, :3] for T in JOINT_TF])
    pj = np.array([T[:3, 3] for T in JOINT_TF])
    Th = tf(p=(0, 0, FLANGE_Z)) @ tf(rot("z", HAND_YAW))
    hm = np.array([m for m, _ in HAND_MASS_COM])
    hc = np.array([Th[:3, :3] @ np.array(c) + Th[:3, 3] for _, c in HAND_MASS_COM])
    lm = np.array([m for m, _ in LINK_MASS_COM])
    lc = np.array([c for _, c in LINK_MASS_COM], dtype=float)
    return Rj, pj, lm, lc, hm, hc


_GRAV = None


def gravity_torques_fast(q):
    """gravity_torques with fewer temporaries (~10x faster, equal to ~1e-14 N m).
    Fast mode only, so the default 1 ms path stays bit-identical."""
    global _GRAV
    if _GRAV is None:
        _GRAV = _gravity_consts()
    Rj, pj, lm, lc, hm, hc = _GRAV
    R, p = np.eye(3), np.array(GRAVITY_ORIGIN, dtype=float)
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
    return GRAVITY * (ax[:, 0] * d[:, 1] - ax[:, 1] * d[:, 0])


def finger_forces(fq, fv, grip, fmax):
    """Coupled-finger forces (left, right) for per-finger opening target grip."""
    # Left opens along +y_hand, right along -y_hand: (qL - qR)/2 is how far
    # the pair has shifted off-centre along the hand's y axis.
    off, off_v = (fq[0] - fq[1]) / 2, (fv[0] - fv[1]) / 2
    f_center = -CENTER_KP * off - CENTER_KD * off_v
    out = []
    for qf, vf, sign in zip(fq, fv, (1, -1)):
        squeeze = FINGER_KP * (grip - qf) - FINGER_KD * vf
        squeeze = max(-fmax, min(fmax, squeeze))
        out.append(squeeze + sign * f_center)
    return out


def finger_forces_dt(fq, fv, grip, fmax, dt):
    """finger_forces for a physics step dt > 1 ms (fast mode).

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
    m, c = FINGER_MASS, FINGER_DAMPING
    if dt <= FAST_EXPLICIT_MAX_DT + 1e-9:
        off = (fq[0] - fq[1]) / 2
        f_center = -CENTER_KP * off
        kd = FINGER_KD if FINGER_KD * dt / m <= 0.9 else 0.4 * m / dt
        sq = [max(-fmax, min(fmax, FINGER_KP * (grip - q) - kd * v)) for q, v in zip(fq, fv)]
        return [sq[0] + f_center, sq[1] - f_center]
    kd = min(FINGER_KD, 0.5 * m / dt)
    sq = [max(-fmax, min(fmax, FINGER_KP * (grip - q) - kd * v)) for q, v in zip(fq, fv)]
    off, off_v = (fq[0] - fq[1]) / 2, (fv[0] - fv[1]) / 2
    K, D = FINGER_KP + CENTER_KP, kd + CENTER_KD
    v1 = (m * off_v - dt * K * off) / (m + dt * dt * K + dt * (D + c))   # anti mode velocity next step
    f_anti = -K * (off + dt * v1) - D * v1
    f_center = f_anti - (-FINGER_KP * off - kd * off_v)              # minus the squeeze's share
    return [sq[0] + f_center, sq[1] - f_center]


class ParallelGripper(Gripper):
    """Franka Hand: two prismatic fingers, joint value = the finger's opening [m]."""

    def joint_targets(self, opening):
        return {j: opening for j in self.joints}

    def openings(self, q):
        return np.asarray(q)

    def efforts(self, q, v, opening, force, dt):
        return (finger_forces(q, v, opening, force) if dt <= CONTROL_DT
                else finger_forces_dt(q, v, opening, force, dt))


class MobilePanda(Robot):
    def __init__(self, spec):
        super().__init__(spec)
        self.lower, self.upper = LOWER, UPPER
        self.tcp_tf = tcp_tf(self.tcp_z)
        # Seeds with the hand pointing forward (q6 ~ 3), rolled 0 / +-90 deg: IK from
        # READY (hand down) alone tends to end in a flipped wrist for horizontal grasps.
        self.forward_seeds = [np.array([0.0, q2, 0.0, q4, 0.0, q6, q7])
                              for q2, q4 in ((-0.3, -2.2), (0.3, -1.8)) for q6 in (2.6, 3.1)
                              for q7 in (0.785, 0.785 - math.pi / 2, 0.785 + math.pi / 2)]

    def make_gripper(self, spec):
        return ParallelGripper(spec)

    def fk(self, q, base):
        T = self.link0_tf(base)
        axes, origins = [], []
        for Tj, qi in zip(JOINT_TF, q):
            T = T @ Tj
            axes.append(T[:3, 2].copy())
            origins.append(T[:3, 3].copy())
            T = T @ tf(rot("z", qi))
        return T @ self.tcp_tf, np.array(axes), np.array(origins)

    def ik(self, p_goal, R_goal, q, base, iters=300, margin=0.05):
        """Damped least squares with a step limit and a weak pull towards READY."""
        READY = self.ready
        q = np.array(q, float)
        lo, hi = LOWER + margin, UPPER - margin
        for _ in range(iters):
            T, z, o = self.fk(q, base)
            e = np.concatenate([p_goal - T[:3, 3], 0.5 * rotvec(R_goal @ T[:3, :3].T)])
            if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-4:
                break
            J = np.vstack([np.cross(z, T[:3, 3] - o).T, 0.5 * z.T])
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(6))
            dq = Jp @ e + (np.eye(7) - Jp @ J) @ (0.05 * (READY - q))
            n = np.linalg.norm(dq)
            if n > 0.2:
                dq *= 0.2 / n
            q = np.clip(q + dq, lo, hi)
        T, _, _ = self.fk(q, base)
        return q, np.linalg.norm(p_goal - T[:3, 3]), np.linalg.norm(rotvec(R_goal @ T[:3, :3].T))

    def ik_best(self, p_goal, R_goal, base, seeds=None, tol=(1e-3, 1e-2)):
        """IK from several seeds; the converged solution closest to READY."""
        READY = self.ready
        rng = np.random.default_rng(0)
        seeds = list(seeds or []) + [READY] + self.forward_seeds + [
            np.clip(READY + rng.normal(0, 0.5, 7), LOWER, UPPER) for _ in range(8)]
        best = None
        for s in seeds:
            q, ep, er = self.ik(p_goal, R_goal, s, base)
            ok = ep < tol[0] and er < tol[1]
            # prefer joint-limit margin (up to 0.6 rad), then closeness to READY
            margin = np.min(np.minimum(q - LOWER, UPPER - q))
            key = (not ok, -min(margin, 0.6) + 0.05 * np.linalg.norm(q - READY) if ok else ep + er)
            if best is None or key < best[0]:
                best = (key, q, ep, er)
        return best[1], best[2], best[3]

    def arm_torques(self, q, q_des, qd_des, dt=CONTROL_DT):
        g = gravity_torques(q) if dt <= CONTROL_DT else gravity_torques_fast(q)
        tau = self.arm_kp * (q_des - q) + self.arm_damping * qd_des + g
        return np.clip(tau, -self.arm_tau_max, self.arm_tau_max)


def make(spec):
    return MobilePanda(spec)
