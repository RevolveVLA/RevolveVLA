"""AutoMoMa planner frame -> world frame (numpy only: the gz image has no torch).

AutoMoMa plans in an object-centred frame: root = inverse(object world pose) with the
translation z zeroed (automoma/planning/planner.py _init_root_pose), so for an object
rotated only about z, world = (x, y, 0, yaw of the object) * planner.

obj_pose is the task's `world_pose` [x, y, z, yaw] (specs/tasks/<name>/task.yaml).
Base yaw is not wrapped, on either side: gz base_yaw accepts +-8 pi and the controller
wraps the yaw error itself.
"""
import math

import numpy as np


def object_tf(obj_pose):
    """4x4 planner frame -> world."""
    x, y, _, yaw = obj_pose
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def base_to_world(base, obj_pose):
    """(..., 3) planner base (x, y, yaw) -> world base (x, y, yaw) = gz base_x, base_y,
    base_yaw joint values (the robot model sits at the world origin)."""
    b = np.asarray(base, float)
    T = object_tf(obj_pose)
    xy = b[..., :2] @ T[:2, :2].T + T[:2, 3]
    return np.concatenate([xy, b[..., 2:3] + obj_pose[3]], axis=-1)


def pose_to_world(T, obj_pose):
    """(..., 4, 4) planner-frame pose -> world."""
    return object_tf(obj_pose) @ np.asarray(T, float)


def quat_wxyz_to_R(q):
    """(..., 4) unit quaternion w, x, y, z (cuRobo, AutoMoMa order) -> (..., 3, 3)."""
    w, x, y, z = np.moveaxis(np.asarray(q, float), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def pose7_to_tf(p):
    """(..., 7) x, y, z, qw, qx, qy, qz -> (..., 4, 4)."""
    p = np.asarray(p, float)
    T = np.zeros(p.shape[:-1] + (4, 4))
    T[..., :3, :3] = quat_wxyz_to_R(p[..., 3:])
    T[..., :3, 3] = p[..., :3]
    T[..., 3, 3] = 1.0
    return T
