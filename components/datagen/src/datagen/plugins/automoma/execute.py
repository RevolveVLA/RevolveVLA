"""Streams a plan to the sim (rclpy; imported inside the client container only): at the action rate, in sim time, the arm and
the hand as single-point joint trajectories on the controllers' topics (each replaces the active one) and the base as a body
twist that tracks the planned world path with ground-truth pose feedback (DATAGEN-v2.md 5). The helpers of the moveit plugin do
the robot side (poses, cmd_vel, the EpisodeOver stop)."""
from __future__ import annotations

import numpy as np

from datagen.plugins.moveit.robot_io import EpisodeOver, RobotIO

from . import track

LEAD_S = 0.1            # each point asks the controller to reach the reference this far ahead (the contract test's value)
GAIN = 2.0              # 1/s: the base's proportional correction
V_MAX, W_MAX = 0.4, 0.6


class Streamer:
    def __init__(self, io: RobotIO, plan: dict):
        from trajectory_msgs.msg import JointTrajectory
        self.io, self.plan, self.accel = io, plan, plan["base_accel"]
        self.msg = JointTrajectory
        c = io.case
        self.arm = io.node.create_publisher(JointTrajectory, f'/{c["arm_controller"]}/joint_trajectory', 10)
        self.hand = io.node.create_publisher(JointTrajectory, f'/{c["hand_controller"]}/joint_trajectory', 10)
        self.rate = plan["rate_hz"]
        self.base, self.q, self.grip = (np.asarray(plan[k], float) for k in ("base", "arm", "hand"))

    def point(self, pub, joints, q) -> None:
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectoryPoint
        m = self.msg()
        m.joint_names = list(joints)
        p = JointTrajectoryPoint()
        p.positions = [float(x) for x in q]
        p.time_from_start = Duration(sec=int(LEAD_S), nanosec=int((LEAD_S % 1) * 1e9))
        m.points = [p]
        pub.publish(m)

    def run(self) -> None:
        """Replays the series from the call on, then holds its end until the evaluator's verdict. EpisodeOver ends it early;
        the base is stopped either way."""
        io = self.io
        try:
            self.stream()
            io.wait_for(lambda: io.check() or False, 15)
        except EpisodeOver:
            pass
        finally:
            for _ in range(5):
                io.twist((0.0, 0.0, 0.0))

    def stream(self) -> None:
        io, dt, n = self.io, 1.0 / self.rate, len(self.base)
        lead = int(round(LEAD_S * self.rate))
        io.wait_for(lambda: self.arm.get_subscription_count() and self.hand.get_subscription_count(), 30)
        t0, prev = io.client.now(), np.zeros(3)
        for i in range(n):
            io.wait(max(0.0, t0 + i * dt - io.client.now()))
            j = min(i + lead, n - 1)
            self.point(self.arm, io.case["arm_joints"], self.q[j])
            self.point(self.hand, [io.case["hand_joint"]], [self.grip[j]])
            cmd = track.twist(self.base[i], self.base[min(i + 1, n - 1)], io.base()[1], dt, GAIN, V_MAX, W_MAX)
            prev = track.slew(prev, cmd, self.accel, dt)
            io.twist(prev)
