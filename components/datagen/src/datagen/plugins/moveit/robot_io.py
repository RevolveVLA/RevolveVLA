"""The robot side of the MoveIt expert on a live sim (rclpy; imported inside the client container only): ground-truth base
pose, door / slide joint, ramped base segments on cmd_vel, joint-trajectory goals to the arm and hand controllers, all in
sim time and through the controllers' interfaces (DATAGEN-v2.md 5). It adds its subscriptions to the node of the
`SimClient`, which is already spun by that client's executor."""
from __future__ import annotations

import math
import threading
import time

import numpy as np

from . import motion
from .geometry import rpy_matrix


class EpisodeOver(Exception):
    """The evaluator ended the episode (its result is out): nothing left to command."""


class RobotIO:
    def __init__(self, client, robot: str, world: str, case: dict, joint_topic: tuple):
        """`case`: {arm_controller, arm_joints, hand_controller, hand_joint}; `joint_topic` = (model, joint) of the door /
        drawer (its position is `articulation`)."""
        import rclpy
        from control_msgs.action import FollowJointTrajectory
        from geometry_msgs.msg import TwistStamped
        from rclpy.action import ActionClient
        from sensor_msgs.msg import JointState
        from tf2_msgs.msg import TFMessage
        self.client, self.node, self.case = client, client.node, case
        self._rclpy, self._twist_type, self._goal_type = rclpy, TwistStamped, FollowJointTrajectory
        self._lock = threading.Lock()
        self._pose, self._joints, self.articulation = None, {}, None
        self._seen = set(client.results)           # results already there: the episode being driven is a new one
        model, joint = joint_topic
        self.node.create_subscription(TFMessage, f"/model/{robot}/pose", self._on_pose, 50)
        self.node.create_subscription(JointState, "/joint_states", self._on_joints, 50)
        self.node.create_subscription(JointState, f"/world/{world}/model/{model}/joint_state",
                                      lambda m: self._on_articulation(joint, m), 10)
        self.cmd = self.node.create_publisher(TwistStamped, "/omni_wheel_drive_controller/cmd_vel", 10)
        self.arm = ActionClient(self.node, FollowJointTrajectory, f'/{case["arm_controller"]}/follow_joint_trajectory')
        self.hand = ActionClient(self.node, FollowJointTrajectory, f'/{case["hand_controller"]}/follow_joint_trajectory')

    # --- subscriptions
    def _on_pose(self, m) -> None:
        t = m.transforms[-1].transform
        q = t.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        with self._lock:
            self._pose = (t.translation.x, t.translation.y, t.translation.z, yaw)

    def _on_joints(self, m) -> None:
        with self._lock:
            self._joints = dict(zip(m.name, m.position))

    def _on_articulation(self, joint, m) -> None:
        if joint in m.name:
            self.articulation = m.position[m.name.index(joint)]

    # --- state
    def begin_episode(self) -> None:
        """Results of earlier episodes no longer end this one."""
        self._seen = set(self.client.results)

    def check(self) -> None:
        """EpisodeOver once the evaluator published the result of the episode being driven."""
        if set(self.client.results) - self._seen:
            raise EpisodeOver()

    def phase(self):
        s = self.client.state()
        return s["phase"] if s else None

    def base(self):
        """-> (4x4 world pose of base_footprint, (x, y, yaw)) from the gz ground truth."""
        self.wait_for(lambda: self._pose is not None, 10)
        with self._lock:
            x, y, z, yaw = self._pose
        return rpy_matrix(x, y, z, 0, 0, yaw), (x, y, yaw)

    def joints(self) -> dict:
        with self._lock:
            return dict(self._joints)

    def point(self, frame: str, in_frame: str) -> np.ndarray:
        """`frame`'s origin in `in_frame` (tf of the sim's robot_state_publisher), homogeneous."""
        tr = self.client.tf.lookup_transform(in_frame, frame, self._rclpy.time.Time(),
                                             timeout=self._rclpy.duration.Duration(seconds=2)).transform.translation
        return np.array([tr.x, tr.y, tr.z, 1.0])

    # --- waiting, in sim time
    def wait(self, sim_s: float, wall_s: float = 120.0) -> None:
        w0 = time.time()
        while self.client.now() == 0.0 and time.time() - w0 < wall_s:      # no /clock yet
            time.sleep(0.005)
        t0 = self.client.now()
        while self.client.now() - t0 < sim_s and time.time() - w0 < wall_s:
            self.check()
            time.sleep(0.005)

    def wait_for(self, pred, wall_s: float) -> bool:
        w0 = time.time()
        while not pred() and time.time() - w0 < wall_s:
            time.sleep(0.02)
        return pred()

    # --- base
    def twist(self, v) -> None:
        m = self._twist_type()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.header.frame_id = "base_footprint"
        m.twist.linear.x, m.twist.linear.y, m.twist.angular.z = (float(c) for c in v)
        self.cmd.publish(m)

    def segment(self, direction, amount: float, rate: float) -> None:
        """A ramped constant body twist `direction` x peak rate covering `amount` (m or rad), then zero twists."""
        peak, total = motion.profile(amount, rate)
        t0 = self.client.now()
        try:
            while (t := self.client.now() - t0) < total:
                self.check()
                self.twist(motion.ramp(t, total) * peak * np.asarray(direction))
                self.wait(1.0 / motion.RATE_HZ)
            for _ in range(5):
                self.twist((0.0, 0.0, 0.0))
                self.wait(1.0 / motion.RATE_HZ)
        except EpisodeOver:
            self.twist((0.0, 0.0, 0.0))
            raise

    def drive_to(self, goal_xy, yaw_goal: float, v: float, w: float) -> None:
        """Open loop to a world pose: rotate to `yaw_goal` (w rad/s), then translate straight to `goal_xy` (v m/s)."""
        self.segment(*motion.rotation(self.base()[1][2], yaw_goal, w))
        self.segment(*motion.translation(self.base()[1], goal_xy, v))

    # --- arm and hand
    def goal(self, client, joints, q, duration: float) -> int | None:
        """A single-point FollowJointTrajectory goal; blocks until its result -> error_code (0 = success)."""
        g = self._goal_type.Goal()
        g.trajectory.joint_names = list(joints)
        from trajectory_msgs.msg import JointTrajectoryPoint
        p = JointTrajectoryPoint()
        p.positions = [float(x) for x in q]
        p.time_from_start.sec, p.time_from_start.nanosec = int(duration), int((duration % 1) * 1e9)
        g.trajectory.points = [p]
        if not client.wait_for_server(timeout_sec=30):
            raise RuntimeError("no trajectory action server")
        h = client.send_goal_async(g)
        if not self.wait_for(h.done, 30):
            raise RuntimeError("trajectory goal not accepted")
        r = h.result().get_result_async()
        self.wait_for(r.done, 120)
        return r.result().result.error_code if r.done() else None

    def move_arm(self, q, duration: float):
        return self.goal(self.arm, self.case["arm_joints"], q, duration)

    def move_hand(self, q: float, duration: float):
        return self.goal(self.hand, [self.case["hand_joint"]], [q], duration)
