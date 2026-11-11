"""The robot interface every robot under robots/<robot>/ implements, and the parts all robots share.

A robot is an arm with a gripper on the shared virtual holonomic base: three joints from the world
(prismatic base_x, prismatic base_y, revolute base_yaw), then a box chassis (base_link) with the arm's
root link fixed on top at the robot.yaml mount. Everything is commanded with efforts (no velocity
commands): the base with P on (x, y, yaw) + feed-forward of the SDF joint damping, clamped here; the
arm and the gripper with the robot's own laws (Robot.arm_torques, Gripper.efforts).

A robot command is one Action per 1 ms control step: base target (x, y, yaw) and velocity, arm joint
target and velocity, gripper opening target and max squeeze force. The gripper speaks "opening per
contact link [m]" (gap / 2) everywhere outside the robot: commands, task thresholds, metrics; the robot
maps it to its joints (Gripper.joint_targets) and its joint values back (Gripper.openings). Plan builds
an ActionSeq from moves (drive, arm joints, IK end-effector pose, gripper); an ActionSeq can be saved,
loaded and replayed step by step. Kinematics and Plan don't need Gazebo; Controller runs in a gz system.
"""
import math
from dataclasses import dataclass

import numpy as np

CONTROL_DT = 0.001                   # one Action per physics step [s]
PLAN_DT = 0.01                       # plan samples, upsampled to CONTROL_DT
VIRTUAL_MASS = 1.0                   # base_x_link / base_y_link mass [kg]
BASE_LINK = "base_link"              # the chassis link (every robot's base chain)


# ------------------------------------------------------------------ frames
def rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def tf(R=np.eye(3), p=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def rotvec(R):
    """Rotation matrix -> axis*angle."""
    a = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if a < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * a / (2 * math.sin(a))


def min_jerk(n):
    s = np.linspace(0.0, 1.0, n)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def _axis_angle(w, a):
    u = w / np.linalg.norm(w)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def base_tf(base):
    """World pose of base_link for base = (x, y, yaw)."""
    return tf(rot("z", base[2]), (base[0], base[1], 0.0))


def hand_rot(approach, closing):
    """TCP orientation: hand z along the approach direction, the gripper closes along closing."""
    z = np.asarray(approach, float) / np.linalg.norm(approach)
    y = np.asarray(closing, float) / np.linalg.norm(closing)
    return np.column_stack([np.cross(y, z), y, z])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------ the robot
class Gripper:
    """The gripper of a robot: its actuated joints (state order) and the two contact links whose
    contacts with a handle count as the two "finger" contacts of the metrics."""

    def __init__(self, spec):
        g = spec.gripper
        self.joints = tuple(spec.joints.gripper)
        self.contact_links = tuple(g.contact_links)
        self.hand_links = tuple(g.hand_links)
        self.open_m, self.closed_m, self.force_n = g.open_m, g.closed_m, g.force_n

    def joint_targets(self, opening):
        """{gripper joint: position} for a per-contact-link opening [m]."""
        raise NotImplementedError

    def openings(self, q):
        """Raw gripper joint values (..., k) -> per-contact-link openings (..., 2) [m]."""
        raise NotImplementedError

    def efforts(self, q, v, opening, force, dt):
        """Joint efforts (k,) to the opening target with at most `force` [N] of squeeze."""
        raise NotImplementedError


class Robot:
    """One robot: its spec (specs/robots/<robot>/robot.yaml) and its laws. Subclasses add the kinematics
    (fk, ik, ik_best), gravity compensation and the arm law, and build their Gripper."""

    def __init__(self, spec):
        self.spec = spec
        self.name, self.model = spec.name, spec.model
        self.implementation, self.scale = spec.implementation, spec.scale     # code folder robots/<implementation>/
        self.base_joints = tuple(spec.joints.base)
        self.arm_joints = tuple(spec.joints.arm)
        self.dof = len(self.arm_joints)
        self.mount = np.array(spec.mount.xyz, float)
        sx, sy = spec.chassis.size_xy
        self.chassis_size = (sx, sy, spec.mount.xyz[2] - 0.005)     # box, 5 mm above the floor
        self.chassis_mass = spec.chassis.mass
        b = spec.base
        self.base_kp, self.base_damping, self.base_f_max = (np.array(v, float) for v in (b.kp, b.damping, b.f_max))
        self.base_v_max, self.base_w_max = b.v_max, b.w_max      # planning limits (peak of min-jerk)
        self.base_limits = np.array(b.limits, float)
        a = spec.arm
        self.ready = np.array(a.ready, float)
        self.arm_kp, self.arm_damping, self.arm_tau_max = (np.array(v, float) for v in (a.kp, a.damping, a.tau_max))
        self.root_link, self.hand_link = spec.links.root, spec.links.hand
        self.on_chassis = set(spec.links.on_chassis)
        self.tcp_link, self.tcp_z = spec.tcp.link, spec.tcp.hand_offset_z
        self.link0_in_base = tf(p=(self.mount[0], self.mount[1], self.mount[2]))
        self.state_names, self.action_names = list(spec.joints.state), list(spec.joints.action)
        self.gripper = self.make_gripper(spec)
        n = 3 + self.dof
        self.grip_cols = slice(n, n + len(self.gripper.joints))   # gripper columns of the state log

    def make_gripper(self, spec):
        raise NotImplementedError

    def link0_tf(self, base):
        return base_tf(base) @ self.link0_in_base

    def fk(self, q, base):
        """TCP pose in the world, joint axes and origins (world) for the Jacobian."""
        raise NotImplementedError

    def ik(self, p_goal, R_goal, q, base, iters=300):
        """-> (q, position error [m], rotation error [rad])."""
        raise NotImplementedError

    def ik_best(self, p_goal, R_goal, base, seeds=None, tol=(1e-3, 1e-2)):
        raise NotImplementedError

    def arm_torques(self, q, q_des, qd_des, dt):
        """Capped arm joint torques (P + damping feed-forward + gravity) for physics step dt."""
        raise NotImplementedError

    @property
    def scripted(self):
        """robots/<implementation>/scripted.py: smoke_plan, SCENARIOS (skill tests), EPISODES (scripted episodes)."""
        import importlib
        return importlib.import_module(f"robots.{self.implementation}.scripted")

    @property
    def planner(self):
        """robots/<implementation>/planner.py: the cuRobo planner robot (datagen only; imports cuRobo and gen_assets)."""
        import importlib
        return importlib.import_module(f"robots.{self.implementation}.planner")

    def reach_origin(self):
        """Base-frame (x, y) the reach distance d of base_for_reach is measured from."""
        return np.array([self.mount[0], 0.0])

    def first_joint_xy(self):
        """Base-frame (x, y) of the arm's first joint axis (vertical: turning the base about it and the first
        joint back by the same angle leaves the rest of the arm where it was)."""
        return self.fk(self.ready, (0.0, 0.0, 0.0))[2][0][:2].copy()

    def base_for_reach(self, p, heading, bearing=None, d=None):
        """Base pose (x, y, yaw) facing `heading` that puts the TCP target p at horizontal distance d
        from the reach origin, `bearing` [rad] left of the arm's forward axis (defaults: robot.yaml reach)."""
        bearing = self.spec.reach.bearing_rad if bearing is None else bearing
        d = self.spec.reach.d_m if d is None else d
        a = heading + bearing
        l0 = np.asarray(p[:2], float) - d * np.array([math.cos(a), math.sin(a)])
        ox, oy = self.reach_origin()
        c, s = math.cos(heading), math.sin(heading)
        b = l0 - np.array([ox * c - oy * s, ox * s + oy * c])
        return np.array([b[0], b[1], heading])


# ------------------------------------------------------------------- actions
@dataclass
class Action:
    """One control-step command."""
    base: np.ndarray        # (x, y, yaw) target
    base_vel: np.ndarray    # feed-forward velocity
    arm: np.ndarray         # arm joint targets
    arm_vel: np.ndarray
    grip: float             # per-contact-link opening target [m]
    grip_force: float       # max squeeze [N]


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

    def __init__(self, robot, base, q=None, grip=None, grip_force=None):
        self.robot = robot
        self.b = np.array(base, float)
        self.q = np.array(robot.ready if q is None else q, float)
        self.g = float(robot.gripper.open_m if grip is None else grip)
        self.f = float(robot.gripper.force_n if grip_force is None else grip_force)
        self.rows = []          # (base[3], q[dof], grip, force) at PLAN_DT
        self.events = []        # (t, label, info)
        self.ik_worst = (0.0, 0.0)

    @property
    def t(self):
        return len(self.rows) * PLAN_DT

    def _push(self, n, label, info, fn):
        self.events.append((self.t, label, info or {}))
        for s in min_jerk(max(2, int(round(n / PLAN_DT)))):
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
            dur = max(3.0, 1.875 * max(np.linalg.norm(d[:2]) / self.robot.base_v_max,
                                       abs(d[2]) / self.robot.base_w_max))

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
        rb = self.robot
        p, R = np.asarray(p, float), np.asarray(R, float)
        info = dict(info or {}, p=p.tolist(), R=R.tolist())
        if not cartesian:
            q_goal, ep, er = rb.ik_best(p, R, self.b, seeds=[self.q])
            self.ik_worst = (max(self.ik_worst[0], ep), max(self.ik_worst[1], er))
            info.update(ik_err=[ep, er], q_goal=q_goal.tolist())
            return self.arm_joints(q_goal, dur, label, info)
        T0, _, _ = rb.fk(self.q, self.b)
        p0, R0 = T0[:3, 3].copy(), T0[:3, :3].copy()
        w = rotvec(R @ R0.T)

        def step(s):
            a = np.linalg.norm(w) * s
            Rs = (_axis_angle(w, a) if a > 1e-9 else np.eye(3)) @ R0
            self.q, ep, er = rb.ik(p0 + s * (p - p0), Rs, self.q, self.b, iters=50)
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
            self.q, ep, er = self.robot.ik(np.asarray(p, float), np.asarray(R, float), self.q, self.b, iters=50)
            self.ik_worst = (max(self.ik_worst[0], ep), max(self.ik_worst[1], er))
        self._push(dur, label, info, step)

    def tcp(self):
        """Commanded TCP pose (world) at the end of the plan so far."""
        return self.robot.fk(self.q, self.b)[0]

    def gripper(self, width, dur=1.5, force=None, label="gripper", info=None):
        """width: per-contact-link opening target [m] (gripper.open_m / closed_m); force: squeeze [N]."""
        g0 = self.g
        self.f = self.robot.gripper.force_n if force is None else float(force)

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


# ---------------------------------------------------------------- controller
class Controller:
    """In-loop effort controller of a robot's gz model (call apply() every step from a gz-sim
    system's pre_update)."""

    def __init__(self, robot, model, ecm):
        from gz.sim import Joint, Link
        self.robot = robot
        jn = lambda n: Joint(model.joint_by_name(ecm, n))
        self.base = [jn(n) for n in robot.base_joints]
        self.arm = [jn(n) for n in robot.arm_joints]
        self.grip = [jn(n) for n in robot.gripper.joints]
        for j in self.base + self.arm + self.grip:
            j.enable_position_check(ecm)
            j.enable_velocity_check(ecm)
        self.names = list(robot.base_joints) + list(robot.arm_joints) + list(robot.gripper.joints)
        self.bad = []           # non-finite commands (zeroed instead): (joint, value, state)
        self.dt = CONTROL_DT    # physics step; > CONTROL_DT selects the robot's fast-mode laws
        self.hand = Link(model.link_by_name(ecm, robot.hand_link))
        self.base_link = Link(model.link_by_name(ecm, BASE_LINK))

    def reset(self, ecm, base, q=None, grip=None):
        R = self.robot
        q = R.ready if q is None else q
        grip = R.gripper.open_m if grip is None else grip
        for j, v in zip(self.base + self.arm + self.grip,
                        list(base) + list(q) + list(R.gripper.joint_targets(grip).values())):
            j.reset_position(ecm, [float(v)])

    @staticmethod
    def _read(joints, ecm):
        return (np.array([(j.position(ecm) or [0.0])[0] for j in joints]),
                np.array([(j.velocity(ecm) or [0.0])[0] for j in joints]))

    def state(self, ecm):
        """Joint-space state (base pose = base joint values: model at the origin). grip_q / grip_qv: the
        raw gripper joints; fingers: the per-contact-link openings [m] (the metrics')."""
        b, bv = self._read(self.base, ecm)
        q, qv = self._read(self.arm, ecm)
        g, gv = self._read(self.grip, ecm)
        return dict(base=b, base_vel=bv, arm=q, arm_vel=qv, grip_q=g, grip_qv=gv,
                    fingers=self.robot.gripper.openings(g))

    def apply(self, ecm, a, meas=None):
        """meas: measured state to use instead of the ECM's (a dict as from
        state(); e.g. a snapshot being restored in this same step)."""
        R = self.robot
        m = meas or self.state(ecm)
        b, q = m["base"], m["arm"]
        e = a.base - b
        e[2] = wrap(e[2])
        fb = np.clip(R.base_kp * e + R.base_damping * a.base_vel, -R.base_f_max, R.base_f_max)
        ff = R.gripper.efforts(m["grip_q"], m["grip_qv"], a.grip, a.grip_force, self.dt)
        tau = R.arm_torques(q, a.arm, a.arm_vel, self.dt)
        u = np.concatenate([fb, tau, ff])
        # The plugin writes force commands to MuJoCo ctrl unchecked
        for i in np.flatnonzero(~np.isfinite(u)):
            if len(self.bad) < 20:
                self.bad.append((self.names[i], float(u[i]), [*b, *q, *m["grip_q"]]))
            u[i] = 0.0
        for j, ui in zip(self.base + self.arm + self.grip, u):
            j.set_force(ecm, [float(ui)])


def robot_state(robot, snap):
    """Controller.state()-like dict from a scene snapshot (scene_state)."""
    names = [str(x) for x in snap["joint_names"]]

    def get(f, joints):
        return np.array([snap[f][names.index(f"{robot.model}::{j}")] for j in joints])
    g = get("joint_pos", robot.gripper.joints)
    return dict(base=get("joint_pos", robot.base_joints), base_vel=get("joint_vel", robot.base_joints),
                arm=get("joint_pos", robot.arm_joints), arm_vel=get("joint_vel", robot.arm_joints),
                grip_q=g, grip_qv=get("joint_vel", robot.gripper.joints), fingers=robot.gripper.openings(g))


# ---------------------------------------------------------------- the model
def sdf_num(*v):
    return " ".join(f"{x:.6g}" for x in v)


def base_sdf(robot, root_inner):
    """SDF of the shared base chain: world -> base_x -> base_y -> base_yaw -> base_link (box chassis,
    contact sensor) -> the arm's root link (robot.yaml links.root, at the mount; root_inner = its
    inertial and visual elements) by the fixed joint base_mount."""
    sx, sy, sz = robot.chassis_size
    m, mv = robot.chassis_mass, VIRTUAL_MASS
    box_i = (m * (sy**2 + sz**2) / 12, m * (sx**2 + sz**2) / 12, m * (sx**2 + sy**2) / 12)
    lim = robot.base_limits
    out = []
    chain = [("world", "base_x_link", "base_x", "prismatic", "1 0 0", lim[0], robot.base_damping[0]),
             ("base_x_link", "base_y_link", "base_y", "prismatic", "0 1 0", lim[1], robot.base_damping[1]),
             ("base_y_link", BASE_LINK, "base_yaw", "revolute", "0 0 1", lim[2], robot.base_damping[2])]
    for parent, child, joint, jtype, axis, lm, damping in chain:
        if child != BASE_LINK:
            out.append(f"""
    <link name="{child}">
      <inertial><mass>{mv}</mass>
        <inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
    </link>""")
        out.append(f"""
    <joint name="{joint}" type="{jtype}">
      <parent>{parent}</parent>
      <child>{child}</child>
      <axis>
        <xyz>{axis}</xyz>
        <limit><lower>{-lm:.6g}</lower><upper>{lm:.6g}</upper><effort>-1</effort></limit>
        <dynamics><damping>{damping}</damping></dynamics>
      </axis>
    </joint>""")
    zc = 0.005 + sz / 2
    out.append(f"""
    <link name="{BASE_LINK}">
      <inertial>
        <pose>0 0 {zc:.6g} 0 0 0</pose>
        <mass>{m}</mass>
        <inertia><ixx>{box_i[0]:.6g}</ixx><iyy>{box_i[1]:.6g}</iyy><izz>{box_i[2]:.6g}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <collision name="base_box">
        <pose>0 0 {zc:.6g} 0 0 0</pose>
        <geometry><box><size>{sdf_num(sx, sy, sz)}</size></box></geometry>
      </collision>
      <visual name="base_box_visual">
        <pose>0 0 {zc:.6g} 0 0 0</pose>
        <geometry><box><size>{sdf_num(sx, sy, sz)}</size></box></geometry>
        <material><ambient>0.22 0.24 0.27 1</ambient><diffuse>0.25 0.27 0.3 1</diffuse></material>
      </visual>
      <visual name="base_front_marker">
        <pose>{sx / 2 - 0.02:.6g} 0 {0.005 + sz * 0.5:.6g} 0 0 0</pose>
        <geometry><box><size>0.045 0.30 0.05</size></box></geometry>
        <material><ambient>0.9 0.55 0.1 1</ambient><diffuse>0.9 0.55 0.1 1</diffuse></material>
      </visual>
      <sensor name="contact" type="contact">
        <update_rate>200</update_rate>
        <contact><collision>base_box</collision></contact>
      </sensor>
    </link>
    <link name="{robot.root_link}">
      <pose relative_to="{BASE_LINK}">{sdf_num(*robot.mount)} 0 0 0</pose>
      {root_inner}
    </link>
    <joint name="base_mount" type="fixed">
      <parent>{BASE_LINK}</parent>
      <child>{robot.root_link}</child>
    </joint>""")
    return "".join(out)
