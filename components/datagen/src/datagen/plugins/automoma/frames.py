"""AutoMoMa planner frame <-> world frame (numpy only).

AutoMoMa plans in an object-centred frame: the object's world pose with its translation z zeroed, so for an object rotated
only about z, world = (x, y, 0, yaw of the object) * planner. Base yaw is not wrapped, on either side.
"""
import math

import numpy as np


def object_tf(obj_pose):
    """4x4 planner frame -> world; obj_pose = [x, y, z, yaw] of the object in the world."""
    x, y, _, yaw = obj_pose
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def base_to_world(base, obj_pose):
    """(..., 3) planner base (x, y, yaw) -> world base (x, y, yaw)."""
    b = np.asarray(base, float)
    T = object_tf(obj_pose)
    xy = b[..., :2] @ T[:2, :2].T + T[:2, 3]
    return np.concatenate([xy, b[..., 2:3] + obj_pose[3]], axis=-1)


def base_from_world(base, obj_pose):
    """(..., 3) world base (x, y, yaw) -> planner base."""
    b = np.asarray(base, float)
    T = np.linalg.inv(object_tf(obj_pose))
    xy = b[..., :2] @ T[:2, :2].T + T[:2, 3]
    return np.concatenate([xy, b[..., 2:3] - obj_pose[3]], axis=-1)
