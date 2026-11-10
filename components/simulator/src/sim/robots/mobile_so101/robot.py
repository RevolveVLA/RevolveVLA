"""Mobile SO-101: the SO-101 arm (TheRobotStudio SO-ARM100, so101_new_calib.urdf = robot.yaml
`description`) on the shared virtual holonomic base, with a taller chassis box so the arm reaches the
kitchen drawer bar (specs/robots/mobile_so101/robot.yaml; robots/base.py has the base, Plan, Controller).

The arm's base (so101_base_link, the URDF's base_link renamed) sits on the chassis at the robot.yaml
mount. Arm: 5 revolute joints (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll),
torque-controlled (P + gravity compensation from the URDF masses + feed-forward of the SDF joint
damping). Gripper: one revolute jaw (joint `gripper`) against the fixed jaw of gripper_link,
torque-controlled with a cap (the squeeze). Opening per contact link = pad gap / 2 (jaw angle <-> gap by
gap()); contact links (fixed jaw, moving jaw) = (gripper_link, the moving jaw link).

Size: robot.yaml `scale` s (default 1) scales the URDF on load (joint, visual and inertial origins x s, masses
x s^3, inertias x s^5) and the built-in geometry here and in model.py / planner.py / scripted.py (TCP offset,
jaw pads, collision boxes, approach distances); robot.yaml values are absolute. The TCP stays TCP_ABOVE_PAD
above the fixed pad face at every scale (it is set by the bar, not by the arm).

TCP (so101_tcp, between the jaw tips, 1 mm above a 14 mm bar resting on the fixed jaw): z along the
approach (out of the jaws), y along the closing direction (from the moving jaw towards the fixed jaw), as
robots.base.hand_rot. The hand link so101_hand is a massless frame link placed robot.yaml
tcp.hand_offset_z behind the TCP along its z (TCP = hand origin + offset * hand z, as every robot).
"""
import math
import xml.etree.ElementTree as ET

import numpy as np

from robots.base import CONTROL_DT, Gripper, Robot, rot, rotvec, tf

ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
JAW = "gripper"                      # the moving jaw's joint (URDF name)
ARM_LINKS = ("shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link")
JAW_LINK = "moving_jaw_so101_v1_link"
# URDF names -> model names (the URDF's base_link clashes with the mobile base's)
RENAME = {"base_link": "so101_base_link"}

# TCP in the gripper_link frame: x 0 = 8 mm above the fixed jaw's pad face (x -0.008: Menagerie
# fixed_jaw_box4), z -0.092 = the middle of the jaw tips' pads; axes see the module doc.
TCP_R = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
TCP_IN_GRIPPER = tf(TCP_R, (0.0, 0.0, -0.092))
FIXED_PAD_X = -0.008                 # fixed jaw pad face (gripper_link x)
TCP_ABOVE_PAD = 0.008                # TCP x above the fixed pad face: half the 14 mm bar + 1 mm

# Rotor inertia stand-in [kg m^2] added (isotropically) to each moving link's inertia in the SDF:
# the gz MuJoCo plugin drops <armature>, and the bare URDF inertias (2e-6..1.6e-4) make the 1 ms
# torque loop stiff. The MJCFs' sts3215 armature is 0.028 about the joint axis. At scale s they grow as s^5.
INERTIA_ADD = {"shoulder_link": 0.01, "upper_arm_link": 0.01, "lower_arm_link": 0.01,
               "wrist_link": 0.01, "gripper_link": 0.005, JAW_LINK: 0.001}
JAW_KP, JAW_DAMPING = 3.0, 0.08      # [N m/rad], SDF <damping> [N m s/rad] (robot.yaml gripper kp / damping)
JAW_R = 0.079                        # jaw axis -> pad [m]: squeeze force [N] -> jaw torque cap


# ---------------------------------------------------------------- the URDF
def _vec(s):
    return [float(x) for x in (s or "0 0 0").split()]


def rpy_mat(r, p, y):
    return rot("z", y) @ rot("y", p) @ rot("x", r)


def tcp_in_gripper(scale=1.0):
    return TCP_IN_GRIPPER if scale == 1.0 else tf(TCP_R, (FIXED_PAD_X * scale + TCP_ABOVE_PAD, 0.0, -0.092 * scale))


def load_urdf(path, scale=1.0):
    """{links: {name: dict(mass, com, inertia, visuals)}, joints: {name: dict(...)}} (URDF names), scaled by
    `scale` (origins x s, masses x s^3, inertias x s^5; visuals get mesh_scale s)."""
    root = ET.parse(path).getroot()
    links, joints = {}, {}
    for ln in root.findall("link"):
        i = ln.find("inertial")
        d = dict(visuals=[])
        if i is not None:
            o, it = i.find("origin"), i.find("inertia")
            d.update(mass=float(i.find("mass").get("value")), com=_vec(o.get("xyz")), com_rpy=_vec(o.get("rpy")),
                     inertia={k: float(it.get(k)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")})
        for v in ln.findall("visual"):
            o = v.find("origin")
            m = v.find("geometry/mesh")
            if m is not None:
                mat = v.find("material")
                d["visuals"].append(dict(xyz=_vec(o.get("xyz")), rpy=_vec(o.get("rpy")), mesh=m.get("filename"),
                                         material=mat.get("name") if mat is not None else None))
        links[ln.get("name")] = d
        if scale != 1.0:
            for v in d["visuals"]:
                v.update(xyz=[x * scale for x in v["xyz"]], mesh_scale=scale)
            if "mass" in d:
                d.update(mass=d["mass"] * scale ** 3, com=[x * scale for x in d["com"]],
                         inertia={k: x * scale ** 5 for k, x in d["inertia"].items()})
    for j in root.findall("joint"):
        o, lim = j.find("origin"), j.find("limit")
        joints[j.get("name")] = dict(type=j.get("type"), parent=j.find("parent").get("link"),
                                     child=j.find("child").get("link"), xyz=_vec(o.get("xyz")),
                                     rpy=_vec(o.get("rpy")), axis=_vec(j.find("axis").get("xyz")),
                                     lower=float(lim.get("lower")) if lim is not None else None,
                                     upper=float(lim.get("upper")) if lim is not None else None)
        if scale != 1.0:
            joints[j.get("name")]["xyz"] = [x * scale for x in joints[j.get("name")]["xyz"]]
    return dict(links=links, joints=joints)


def urdf_path(spec):
    import spec as sim_spec
    return sim_spec.path(spec.description)


# ---------------------------------------------------------------- the gripper
class JawGripper(Gripper):
    """One revolute jaw against the fixed jaw: joint value = the jaw angle [rad]."""

    def __init__(self, spec, jaw_tf, jaw_lower):
        super().__init__(spec)
        self.jaw_tf = jaw_tf
        self.s = spec.scale
        self.kp = JAW_KP if spec.gripper.kp is None else spec.gripper.kp
        self._qj = np.linspace(jaw_lower, 1.7, 400)       # gap() is monotonic up to ~1.73 rad
        self._gj = self.gap(self._qj)

    def gap(self, q_jaw):
        """Jaw pad gap [m] at jaw angle q_jaw: the moving jaw's tip pad face (Menagerie moving_jaw_box2,
        jaw frame (-0.0123, -0.076), x scale) to the fixed pad face, along gripper_link x."""
        s = self.s
        x = self.jaw_tf[0, 3] - s * 0.0123 * np.cos(q_jaw) + s * 0.076 * np.sin(q_jaw)
        return x - s * FIXED_PAD_X

    def jaw_for_opening(self, g):
        """Jaw angle for a per-finger opening g (gap 2 g), clipped to the joint range (closed: the stop)."""
        return float(np.interp(2.0 * g, self._gj, self._qj))

    def joint_targets(self, opening):
        return {JAW: self.jaw_for_opening(opening)}

    def openings(self, q):
        o = self.gap(np.asarray(q)[..., 0]) / 2
        return np.stack([o, o], axis=-1)

    def efforts(self, q, v, opening, force, dt):
        """Jaw torque to the opening target, capped at the squeeze force [N] at the pads (the SDF damping
        is the D term)."""
        cap = force * JAW_R * self.s
        return [float(np.clip(self.kp * (self.jaw_for_opening(opening) - q[0]), -cap, cap))]


# ---------------------------------------------------------------- the robot
class MobileSo101(Robot):
    def __init__(self, spec):
        U = self.urdf = load_urdf(urdf_path(spec), spec.scale)
        self.tcp_in_gripper = tcp_in_gripper(spec.scale)
        self.joint_tf = [tf(rpy_mat(*U["joints"][n]["rpy"]), U["joints"][n]["xyz"]) for n in ARM]
        self.jaw_tf = tf(rpy_mat(*U["joints"][JAW]["rpy"]), U["joints"][JAW]["xyz"])
        self.jaw_lower = U["joints"][JAW]["lower"]
        super().__init__(spec)
        self.lower = np.array([U["joints"][n]["lower"] for n in ARM])
        self.upper = np.array([U["joints"][n]["upper"] for n in ARM])
        # link masses and centres of mass (link frame) of the links after each arm joint; the moving jaw
        # is taken at jaw angle 0 (12 g) and lumped with gripper_link
        self.link_mass_com = [(U["links"][ln]["mass"], np.array(U["links"][ln]["com"])) for ln in ARM_LINKS]
        jl = U["links"][JAW_LINK]
        self.jaw_mass_com = (jl["mass"], (self.jaw_tf @ np.array([*jl["com"], 1.0]))[:3])
        # the arm's plane (lift, elbow and wrist_flex axes are parallel) at pan 0, in the base frame
        T0 = self.fk(np.zeros(5), (0.0, 0.0, 0.0))
        self.plane_y = float(T0[0][1, 3])                # TCP lateral offset at pan 0 (the TCP is on the roll axis)
        self.lift_x = float(T0[2][1][0])                 # shoulder_lift axis ahead of the base centre

    def make_gripper(self, spec):
        return JawGripper(spec, self.jaw_tf, self.jaw_lower)

    def fk(self, q, base):
        T = self.link0_tf(base)
        axes, origins = [], []
        for Tj, qi in zip(self.joint_tf, q):
            T = T @ Tj
            axes.append(T[:3, 2].copy())
            origins.append(T[:3, 3].copy())
            T = T @ tf(rot("z", qi))
        return T @ self.tcp_in_gripper, np.array(axes), np.array(origins)

    def gravity_torques(self, q, base=(0.0, 0.0, 0.0)):
        """Arm joint torques that hold the links (incl. the jaw) against gravity (base yaw irrelevant)."""
        T = self.link0_tf(base)
        axes, origins, coms = [], [], []
        for Tj, qi, (m, c) in zip(self.joint_tf, q, self.link_mass_com):
            T = T @ Tj
            axes.append(T[:3, 2].copy())
            origins.append(T[:3, 3].copy())
            T = T @ tf(rot("z", qi))
            coms.append((m, (T @ np.array([*c, 1.0]))[:3]))
        coms.append((self.jaw_mass_com[0], (T @ np.array([*self.jaw_mass_com[1], 1.0]))[:3]))
        tau = np.zeros(len(ARM))
        for i in range(len(ARM)):
            for k, (m, c) in enumerate(coms):
                if k >= i:     # bodies outboard of joint i
                    tau[i] += m * 9.81 * np.cross(axes[i], c - origins[i])[2]
        return tau

    def ik(self, p_goal, R_goal, q, base, iters=300, margin=0.02):
        """Damped least squares on the 6-D pose error with the 5 arm joints (a least-squares fit: exact
        only where the pose is reachable, e.g. targets in the arm's vertical plane)."""
        q = np.array(q, float)
        lo, hi = self.lower + margin, self.upper - margin
        for _ in range(iters):
            T, z, o = self.fk(q, base)
            e = np.concatenate([p_goal - T[:3, 3], 0.5 * rotvec(R_goal @ T[:3, :3].T)])
            if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-4:
                break
            J = np.vstack([np.cross(z, T[:3, 3] - o).T, 0.5 * z.T])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            n = np.linalg.norm(dq)
            if n > 0.2:
                dq *= 0.2 / n
            q = np.clip(q + dq, lo, hi)
        T, _, _ = self.fk(q, base)
        return q, np.linalg.norm(p_goal - T[:3, 3]), np.linalg.norm(rotvec(R_goal @ T[:3, :3].T))

    def ik_best(self, p_goal, R_goal, base, seeds=None, tol=(1e-3, 1e-2)):
        """IK from several seeds; the converged solution closest to READY."""
        READY, LOWER, UPPER = self.ready, self.lower, self.upper
        rng = np.random.default_rng(0)
        seeds = list(seeds or []) + [READY, np.zeros(5)] + [
            np.clip(READY + rng.normal(0, 0.5, 5), LOWER, UPPER) for _ in range(8)]
        best = None
        for s in seeds:
            q, ep, er = self.ik(p_goal, R_goal, s, base)
            ok = ep < tol[0] and er < tol[1]
            margin = np.min(np.minimum(q - LOWER, UPPER - q))
            key = (not ok, -min(margin, 0.4) + 0.05 * np.linalg.norm(q - READY) if ok else ep + er)
            if best is None or key < best[0]:
                best = (key, q, ep, er)
        return best[1], best[2], best[3]

    def reach_origin(self):
        """The TCP reaches straight ahead in the arm's plane: d is measured from the shoulder_lift axis,
        PLANE_Y to the side (bearing is 0: a horizontal grasp straight ahead is the SO-101's natural one)."""
        return np.array([self.lift_x, self.plane_y])

    def arm_torques(self, q, q_des, qd_des, dt=CONTROL_DT):
        tau = self.arm_kp * (q_des - q) + self.arm_damping * qd_des + self.gravity_torques(q)
        return np.clip(tau, -self.arm_tau_max, self.arm_tau_max)


def make(spec):
    return MobileSo101(spec)
