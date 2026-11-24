"""The robot's joints and base pose as `LiveSource` wants them (callables), read-only, from the topics a client container sees
(/joint_states and /model/<robot>/pose): SimClient keeps only /evaluator/*. The ROS message types are the node's (lazy imports)."""
from __future__ import annotations

import math
import threading


def ros_types():
    from sensor_msgs.msg import JointState
    from tf2_msgs.msg import TFMessage
    return JointState, TFMessage


class RosState:
    """Subscribes on `node` (a SimClient's); `joints()` -> {name: rad} | None, `base()` -> (x, y, yaw) | None (until the first message)."""

    def __init__(self, node, robot: str, types=None):
        joint_state, tf_message = types or ros_types()
        self.robot, self._lock, self._joints, self._pose = robot, threading.Lock(), None, None
        node.create_subscription(tf_message, f"/model/{robot}/pose", self._on_pose, 50)
        node.create_subscription(joint_state, "/joint_states", self._on_joints, 50)

    def _on_pose(self, m) -> None:
        mine = [t for t in m.transforms if t.child_frame_id == self.robot] or m.transforms[-1:]
        t = mine[-1].transform
        q = t.rotation
        with self._lock:
            self._pose = (t.translation.x, t.translation.y,
                          math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))

    def _on_joints(self, m) -> None:
        with self._lock:
            self._joints = dict(zip(m.name, m.position))

    def joints(self):
        with self._lock:
            return self._joints and dict(self._joints)

    def base(self):
        with self._lock:
            return self._pose
