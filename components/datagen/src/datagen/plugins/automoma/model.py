"""The task object of a scene model (numpy + ElementTree): the links' poses at the closed door, their collision boxes and
the task joint, all in the model frame, as the planner's object URDF and grasp need them. The task joint's parent link must be
fixed to the world (a fridge body, a cabinet) and the joint frame must be its child link's frame."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from datagen.plugins.moveit.geometry import rpy_matrix, sdf_pose


def rot(axis, a) -> np.ndarray:
    """3x3 rotation of angle a about an axis."""
    u = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def tf(xyz=(0, 0, 0), rpy=(0, 0, 0)) -> np.ndarray:
    return rpy_matrix(*xyz, *rpy)


@dataclass
class Joint:
    name: str
    type: str                    # revolute | prismatic
    parent: str
    child: str
    axis: np.ndarray             # in the joint frame (= the child link's frame)
    lower: float
    upper: float


@dataclass
class Model:
    links: dict = field(default_factory=dict)       # link -> 4x4 in the model frame at the closed door
    boxes: dict = field(default_factory=dict)       # link -> [(collision name, 4x4 in the link, size)]
    joint: Joint | None = None                      # the task joint
    parent_of: dict = field(default_factory=dict)   # child link -> (joint name, parent link) of every joint

    @classmethod
    def read(cls, sdf, joint: str) -> Model:
        m = ET.parse(sdf).getroot().find("model")
        out = cls()
        for lk in m.findall("link"):
            out.links[lk.get("name")] = sdf_pose(lk)
            out.boxes[lk.get("name")] = [(c.get("name"), sdf_pose(c), [float(v) for v in c.find("geometry/box/size").text.split()])
                                         for c in lk.findall("collision") if c.find("geometry/box/size") is not None]
        out.parent_of = {j.findtext("child"): (j.get("name"), j.findtext("parent")) for j in m.findall("joint")}
        j = next(j for j in m.findall("joint") if j.get("name") == joint)
        lim = j.find("axis/limit")
        out.joint = Joint(joint, j.get("type"), j.findtext("parent"), j.findtext("child"),
                          np.array([float(v) for v in j.findtext("axis/xyz").split()]), float(lim.findtext("lower")),
                          float(lim.findtext("upper")))
        if not np.allclose(sdf_pose(j), np.eye(4)):
            raise ValueError(f"{sdf}: joint {joint} must sit at its child link's frame")
        if out.parent_of[out.joint.parent][1] != "world":
            raise ValueError(f"{sdf}: the parent link of joint {joint} must be fixed to the world")
        return out

    def door_pose(self, q: float) -> np.ndarray:
        """The task joint's child link in the model frame at joint position q."""
        j, T0 = self.joint, self.links[self.joint.child]
        M = np.eye(4)
        if j.type == "revolute":
            M[:3, :3] = rot(j.axis, q)
        else:
            M[:3, 3] = j.axis * q
        return T0 @ M
