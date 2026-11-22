"""Pure geometry of the MoveIt expert (numpy only): SDF poses, quaternions, the frames a grasp is made in."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import NamedTuple

import numpy as np


def rpy_matrix(x, y, z, roll, pitch, yaw) -> np.ndarray:
    """SDF <pose> x y z roll pitch yaw -> 4x4 (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw),
                              math.sin(yaw))
    T = np.eye(4)
    T[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    T[:3, 3] = (x, y, z)
    return T


def sdf_pose(element) -> np.ndarray:
    """The 4x4 of an SDF element's <pose> (identity without one)."""
    return rpy_matrix(*map(float, element.findtext("pose", "0 0 0 0 0 0").split()))


def quat(R) -> tuple[float, float, float, float]:
    """Rotation matrix -> (x, y, z, w)."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return x, y, z, w


def heading(v) -> float:
    """The angle of a vector's xy part about +z."""
    return math.atan2(v[1], v[0])


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Box(NamedTuple):
    link: str
    name: str
    T: np.ndarray          # world frame of the collision
    size: list


class Joint(NamedTuple):
    child: str             # the moving link
    T: np.ndarray          # world frame of the joint
    axis: np.ndarray       # world frame


class ModelGeometry(NamedTuple):
    boxes: list            # [Box]: the collision boxes (the planning scene)
    joints: dict           # {joint name: Joint} (joints with an axis)


def model_geometry(pose, sdf) -> ModelGeometry:
    """World frames of a scene model with its joints at 0. `pose` = the scene's [x, y, z, yaw] of the model, `sdf` = the
    model's SDF file."""
    x, y, z, yaw = pose
    T_m = rpy_matrix(x, y, z, 0, 0, yaw)
    m = ET.parse(sdf).getroot().find("model")
    links, boxes, joints = {}, [], {}
    for lk in m.findall("link"):
        links[lk.get("name")] = T_l = T_m @ sdf_pose(lk)
        for c in lk.findall("collision"):
            b = c.find("geometry/box/size")
            if b is not None:
                boxes.append(Box(lk.get("name"), c.get("name"), T_l @ sdf_pose(c), [float(v) for v in b.text.split()]))
    for j in (j for j in m.findall("joint") if j.find("axis/xyz") is not None):
        T_j = links[j.findtext("child")] @ sdf_pose(j)          # an SDF joint pose is in its child link's frame
        axis = T_j[:3, :3] @ np.array([float(v) for v in j.findtext("axis/xyz").split()])
        joints[j.get("name")] = Joint(j.findtext("child"), T_j, axis)
    return ModelGeometry(boxes, joints)


def tcp_frame(T_h, along) -> np.ndarray:
    """The TCP's rotation for a grasp of a bar: +z into the handle (its -x), x along the bar, which is the handle's
    `along` axis (0..2: z for the fridge's vertical bar, y for the drawer's horizontal one), y = z x x."""
    x = T_h[:3, along]
    return np.column_stack([x, np.cross(-T_h[:3, 0], x), -T_h[:3, 0]])


def grasp_pose(T_h, R_tcp, offset=(0.0, 0.0, 0.0)) -> np.ndarray:
    """World 4x4 of the TCP at the handle centre + `offset` (handle frame) with the rotation R_tcp."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R_tcp, T_h[:3, 3] + T_h[:3, :3] @ np.asarray(offset)
    return T


def back_off(T, back: float) -> np.ndarray:
    """T moved `back` m against its approach axis (z): the pre-grasp of a grasp."""
    out = T.copy()
    out[:3, 3] -= back * T[:3, 2]
    return out


def stand_off(T_h, dist: float, side: float, yaw_off: float) -> tuple[float, float, float]:
    """A base pose [x, y, yaw] relative to a handle: `dist` m out of its face (handle +x), `side` m along its y, the base
    facing the handle turned by `yaw_off` rad (clockwise seen from above). The frame is the handle's at the closed door."""
    p = T_h[:3, 3] + dist * T_h[:3, 0] + side * T_h[:3, 1]
    return float(p[0]), float(p[1]), wrap(heading(-T_h[:3, 0]) - yaw_off)


def rotation_about_z(point, angle: float) -> np.ndarray:
    """World motion of a body turning by `angle` (rad, about +z) around the vertical line through `point`."""
    c, s = math.cos(angle), math.sin(angle)
    R = np.eye(4)
    R[:2, :2] = [[c, -s], [s, c]]
    P = np.eye(4)
    P[:3, 3] = point
    return P @ R @ np.linalg.inv(P)
