"""Minimal SDF kinematics in numpy (no torch, no gz): frame poses from <pose relative_to>,
joint motion about <axis><xyz>, box collisions (a model file's mesh collisions as their AABB
boxes). Enough for the generated gz models (models/mobile_panda, assets/generated*/*) and the
world file; not a general SDF parser.

SDF defaults used: a link pose is relative to the model frame, a joint pose to its child
link, a frame pose to its attached_to; the axis is expressed in the joint frame.
"""
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def rot(axis, a):
    """Rotation of angle a about a unit 3-vector axis."""
    u = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def tf(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    """4x4 from xyz and fixed-axis roll, pitch, yaw (R = Rz Ry Rx, as SDF and URDF)."""
    T = np.eye(4)
    T[:3, :3] = rot((0, 0, 1), rpy[2]) @ rot((0, 1, 0), rpy[1]) @ rot((1, 0, 0), rpy[0])
    T[:3, 3] = xyz
    return T


def pose_of(el):
    """(relative_to or None, 4x4) of an element's <pose>; identity if absent."""
    p = el.find("pose")
    if p is None:
        return None, np.eye(4)
    v = [float(x) for x in p.text.split()]
    return p.get("relative_to"), tf(v[:3], v[3:])


def mesh_box(model_dir, mesh, T):
    """(4x4, size) of a mesh collision's box: the AABB of its OBJ vertices, scaled, in the
    collision frame T. A hull's AABB is never smaller than the hull."""
    path = Path(model_dir) / mesh.find("uri").text.split("://", 1)[-1].split("/", 1)[1]   # model://<name>/<file>
    sc = mesh.find("scale")
    v = np.array([[float(x) for x in ln.split()[1:4]]
                  for ln in path.read_text().splitlines() if ln.startswith("v ")])
    v = v * (np.array([float(x) for x in sc.text.split()]) if sc is not None else 1.0)
    lo, hi = v.min(0), v.max(0)
    return T @ tf((lo + hi) / 2), (hi - lo).tolist()


class Model:
    def __init__(self, src):
        m = src if isinstance(src, ET.Element) else ET.parse(src).getroot().find("model")
        mdir = None if isinstance(src, ET.Element) else Path(src).parent
        self.name = m.get("name")
        self.local = {}            # frame name -> (relative_to, 4x4)
        self.joints = {}           # joint name -> dict(type, parent, child, axis, lower, upper)
        self.parent_joint = {}     # child link -> joint name
        self.boxes = {}            # link -> [(collision name, 4x4 in link, size)]
        self.meshes = []           # mesh collisions of an inline model (not in boxes: callers must reject them)
        self.attached = {}         # <frame> name -> link
        for link in m.findall("link"):
            rel, T = pose_of(link)
            name = link.get("name")
            self.local[name] = (rel or "__model__", T)
            self.boxes[name] = []
            for c in link.findall("collision"):
                box = c.find("geometry/box/size")
                if box is not None:
                    self.boxes[name].append(
                        (c.get("name"), pose_of(c)[1], [float(x) for x in box.text.split()]))
                elif c.find("geometry/mesh") is not None and mdir is not None:
                    self.boxes[name].append((c.get("name"), *mesh_box(mdir, c.find("geometry/mesh"), pose_of(c)[1])))
                elif c.find("geometry/mesh") is not None:
                    self.meshes.append(f"{name}::{c.get('name')}")
        for j in m.findall("joint"):
            rel, T = pose_of(j)
            name, child = j.get("name"), j.find("child").text
            self.local[name] = (rel or child, T)
            ax = j.find("axis")
            lim = ax.find("limit") if ax is not None else None
            self.joints[name] = dict(
                type=j.get("type"), parent=j.find("parent").text, child=child,
                axis=[float(x) for x in ax.find("xyz").text.split()] if ax is not None else None,
                lower=float(lim.find("lower").text) if lim is not None else None,
                upper=float(lim.find("upper").text) if lim is not None else None,
                velocity=float(lim.find("velocity").text)
                if lim is not None and lim.find("velocity") is not None else None)
            self.parent_joint[child] = name
        for f in m.findall("frame"):
            rel, T = pose_of(f)
            self.local[f.get("name")] = (rel or f.get("attached_to"), T)
            self.attached[f.get("name")] = f.get("attached_to")

    def zero(self, name):
        """Model-frame pose of a link, joint or frame at zero joint positions."""
        if name in ("__model__", "world"):
            return np.eye(4)
        rel, T = self.local[name]
        return self.zero(rel) @ T

    def pose(self, name, q):
        """Model-frame pose of a link or frame for joint positions q (dict, missing = 0)."""
        if name in ("__model__", "world"):
            return np.eye(4)
        if name in self.attached:                # <frame> rigidly on its link
            link = self.attached[name]
            return self.pose(link, q) @ np.linalg.inv(self.zero(link)) @ self.zero(name)
        jn = self.parent_joint.get(name)
        if jn is None:
            return self.zero(name)
        j = self.joints[jn]
        P0, J0 = self.zero(j["parent"]), self.zero(jn)
        M = np.eye(4)
        if j["type"] == "revolute":
            M[:3, :3] = rot(j["axis"], q.get(jn, 0.0))
        elif j["type"] == "prismatic":
            M[:3, 3] = np.asarray(j["axis"], float) * q.get(jn, 0.0)
        return (self.pose(j["parent"], q) @ np.linalg.inv(P0) @ J0 @ M
                @ np.linalg.inv(J0) @ self.zero(name))


def world_entities(path):
    """World file: [(name, uri or None, 4x4 pose, inline Model or None)] for every
    <include> and top-level <model>."""
    w = ET.parse(path).getroot().find("world")
    out = []
    for inc in w.findall("include"):
        name = inc.find("name").text
        out.append((name, inc.find("uri").text, pose_of(inc)[1], None))
    for m in w.findall("model"):
        out.append((m.get("name"), None, pose_of(m)[1], Model(m)))
    return out


def yaw_of(T):
    return math.atan2(T[1, 0], T[0, 0])


def angle_between(Ra, Rb):
    """Rotation angle of Ra^T Rb [rad]."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))
