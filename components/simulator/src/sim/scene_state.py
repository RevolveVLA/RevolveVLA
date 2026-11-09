"""Save / restore the whole sim state from a gz-sim Python system.

gz-sim's own reset only goes back to t = 0, so this restores a snapshot through
the regular command components, applied by the Physics system before the next
step:
- every non-fixed joint: Joint.reset_position / reset_velocity (robot base,
  arm, fingers, fridge doors, drawers);
- every free model (no joints, not static: cup, table): Model.set_world_pose_cmd
  and Link.set_linear_velocity / set_angular_velocity on its canonical link.
The robot controller is stateless (efforts from the measured state and the
current Action), so the caller only has to rewind its action index.

Limitations (see README):
- MuJoCo's internal state is not restored: qacc_warmstart (solver warm start),
  noslip/constraint forces of the previous step, ctrl history. The first step
  after a restore therefore differs slightly from the original run.
- gz-sim applies a WorldPoseCmd on two consecutive steps (it removes it one
  iteration late), which would drag a moving free body back by one step;
  after_restore() overwrites the second application with the body's current
  pose, making it a no-op.
- The MuJoCo plugin writes the world-frame angular velocity command straight
  into the free joint's qvel[3:6], which MuJoCo reads in the body frame;
  restore() pre-rotates the command so the body gets the saved world rate.
A snapshot is a dict of numpy arrays (save()/load() as .npz).
"""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
from gz.math import Pose3d, Quaterniond, Vector3d
from gz.sim import Joint, Link, Model

SKIP_MODELS = ("ground_plane", "walls", "scene_camera")


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def mat_quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _tf(pos, quat):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = quat_mat(quat), pos
    return T


def _vec(v):
    return np.array([v.x(), v.y(), v.z()]) if v is not None else np.zeros(3)


def _sdf_link_pose(model_name, link_name):
    """Pose of a link in its model frame, from <link><pose> in model.sdf."""
    for d in os.environ.get("GZ_SIM_RESOURCE_PATH", "").split(":"):
        f = os.path.join(d, model_name, "model.sdf")
        if os.path.isfile(f):
            for link in ET.parse(f).getroot().iter("link"):
                if link.get("name") == link_name:
                    v = [float(x) for x in (link.findtext("pose") or "0 0 0 0 0 0").split()]
                    r, p, y = v[3:]
                    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                                              math.sin(p), math.cos(y), math.sin(y))
                    T = np.eye(4)
                    T[:3, :3] = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                                          [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                                          [-sp, cp * sr, cp * cr]])
                    T[:3, 3] = v[:3]
                    return T
    return np.eye(4)


class SceneState:
    def __init__(self, world, ecm):
        self.links, self.joints, self.free = [], [], []
        for e in world.models(ecm):
            m = Model(e)
            name = m.name(ecm)
            if name in SKIP_MODELS or m.static(ecm):
                continue
            for le in m.links(ecm):
                link = Link(le)
                link.enable_velocity_checks(ecm, True)
                self.links.append((f"{name}::{link.name(ecm)}", link))
            joints = m.joints(ecm)
            for je in joints:
                j = Joint(je)
                j.enable_position_check(ecm)
                j.enable_velocity_check(ecm)
                self.joints.append((f"{name}::{j.name(ecm)}", j))
            if not joints:
                cl = Link(m.canonical_link(ecm))
                self.free.append((name, m, cl, np.linalg.inv(_sdf_link_pose(name, cl.name(ecm)))))
        self._followup = False

    def capture(self, ecm, **extra):
        P, Q, V, W = [], [], [], []
        for _, link in self.links:
            ps = link.world_pose(ecm)
            P.append([ps.pos().x(), ps.pos().y(), ps.pos().z()])
            q = ps.rot()
            Q.append([q.w(), q.x(), q.y(), q.z()])
            V.append(_vec(link.world_linear_velocity(ecm)))
            W.append(_vec(link.world_angular_velocity(ecm)))
        jn, jp, jv = [], [], []
        for n, j in self.joints:
            p, v = j.position(ecm), j.velocity(ecm)
            if p:            # fixed joints have no position
                jn.append(n)
                jp.append(p[0])
                jv.append(v[0] if v else 0.0)
        snap = dict(link_names=np.array([n for n, _ in self.links]), link_pos=np.array(P),
                    link_quat=np.array(Q), link_vel=np.array(V), link_angvel=np.array(W),
                    joint_names=np.array(jn), joint_pos=np.array(jp), joint_vel=np.array(jv))
        snap.update({k: np.asarray(v) for k, v in extra.items()})
        return snap

    def restore(self, ecm, snap):
        """Queue the snapshot; it takes effect in the next physics step."""
        jidx = {str(n): i for i, n in enumerate(snap["joint_names"])}
        for n, j in self.joints:
            if n in jidx:
                i = jidx[n]
                j.reset_position(ecm, [float(snap["joint_pos"][i])])
                j.reset_velocity(ecm, [float(snap["joint_vel"][i])])
        lidx = {str(n): i for i, n in enumerate(snap["link_names"])}
        for name, m, cl, T_ml_inv in self.free:
            i = lidx[f"{name}::{cl.name(ecm)}"]
            T_link = _tf(snap["link_pos"][i], snap["link_quat"][i])
            m.set_world_pose_cmd(ecm, _pose(T_link @ T_ml_inv))
            # Physics maps a link velocity command to the world with the link's
            # current rotation; the plugin then writes it into qvel unrotated.
            cur = cl.world_pose(ecm).rot()
            R_cur = quat_mat([cur.w(), cur.x(), cur.y(), cur.z()])
            R_new = T_link[:3, :3]
            cl.set_linear_velocity(ecm, Vector3d(*(R_cur.T @ snap["link_vel"][i])))
            cl.set_angular_velocity(ecm, Vector3d(*(R_cur.T @ R_new.T @ snap["link_angvel"][i])))
        self._followup = True

    def set_model_pose(self, ecm, name, T):
        """Teleport a free model (e.g. the cup) to world pose T (4x4)."""
        for n, m, _, _ in self.free:
            if n == name:
                m.set_world_pose_cmd(ecm, _pose(T))
                self._followup = True
                return
        raise KeyError(name)

    def after_restore(self, ecm):
        """Call in the pre_update right after restore()/set_model_pose(): turns
        gz-sim's repeated WorldPoseCmd into a no-op (current pose)."""
        if not self._followup:
            return
        self._followup = False
        for _, m, cl, T_ml_inv in self.free:
            ps = cl.world_pose(ecm)
            q = ps.rot()
            T = _tf([ps.pos().x(), ps.pos().y(), ps.pos().z()], [q.w(), q.x(), q.y(), q.z()])
            m.set_world_pose_cmd(ecm, _pose(T @ T_ml_inv))


def _pose(T):
    q = mat_quat(T[:3, :3])
    return Pose3d(Vector3d(*T[:3, 3]), Quaterniond(*q))


def diff(a, b):
    """Max differences between two snapshots (same scene)."""
    out = {}
    la = {str(n): i for i, n in enumerate(a["link_names"])}
    ib = [(la[str(n)], i) for i, n in enumerate(b["link_names"]) if str(n) in la]
    ia, ibb = np.array([x[0] for x in ib]), np.array([x[1] for x in ib])
    names = [str(b["link_names"][i]) for i in ibb]
    dp = np.linalg.norm(a["link_pos"][ia] - b["link_pos"][ibb], axis=1)
    dots = np.abs(np.sum(a["link_quat"][ia] * b["link_quat"][ibb], axis=1))
    dr = 2 * np.arccos(np.clip(dots, -1, 1))
    dv = np.linalg.norm(a["link_vel"][ia] - b["link_vel"][ibb], axis=1)
    dw = np.linalg.norm(a["link_angvel"][ia] - b["link_angvel"][ibb], axis=1)
    for key, d in (("pos_m", dp), ("rot_rad", dr), ("vel_mps", dv), ("angvel_radps", dw)):
        k = int(np.argmax(d))
        out[f"link_{key}"] = [float(d[k]), names[k]]
    ja = {str(n): i for i, n in enumerate(a["joint_names"])}
    jb = [(ja[str(n)], i) for i, n in enumerate(b["joint_names"]) if str(n) in ja]
    if jb:
        x, y = np.array([j[0] for j in jb]), np.array([j[1] for j in jb])
        jn = [str(b["joint_names"][i]) for i in y]
        for key, f in (("pos", "joint_pos"), ("vel", "joint_vel")):
            d = np.abs(a[f][x] - b[f][y])
            k = int(np.argmax(d))
            out[f"joint_{key}"] = [float(d[k]), jn[k]]
    out["bit_identical"] = bool(
        all(np.array_equal(a[k], b[k]) for k in ("link_pos", "link_quat", "link_vel", "link_angvel",
                                                  "joint_pos", "joint_vel")))
    return out


def save(path, snap):
    np.savez_compressed(path, **snap)


def load(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}
