"""What a served checkpoint's observation and action vectors mean on a robot of the new simulator (plain numpy, no ROS).

A checkpoint trained on the earlier simulator's `mobile_panda` data reads the state `[base twist (vx, vy, wz), arm joints,
finger joints]` (12) and the cameras `camera1..3` (wrist, overhead left, overhead right) and writes the action `[base twist,
arm joint targets, gripper target]` (11): the base twist goes to the omni controller's `cmd_vel`, the arm and gripper targets
to the joint trajectory controllers (single-point trajectories at 30 Hz, DATAGEN-v2.md 5.2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

TWIST = 3


@dataclass(frozen=True)
class Contract:
    robot: str
    cameras: Mapping[str, str]                  # checkpoint camera name -> the sim's camera (topic /<camera>/image/compressed)
    arm_controller: str
    arm_joints: Sequence[str]
    hand_controller: str
    hand_joint: str                             # the actuated finger: the gripper target
    finger_joints: Sequence[str]                # the state's finger entries (a missing one repeats the first)
    odom: str                                   # nav_msgs/Odometry: its twist is the state's base twist
    arm_limits: Sequence[tuple[float, float]]   # the actions are clipped to these (a checkpoint is not trusted with a limit)
    hand_limits: tuple[float, float]
    twist_limits: tuple[float, float, float]    # |vx|, |vy|, |wz|

    @property
    def state_dim(self) -> int:
        return TWIST + len(self.arm_joints) + len(self.finger_joints)

    @property
    def action_dim(self) -> int:
        return TWIST + len(self.arm_joints) + 1

    @property
    def joints(self) -> dict[str, list[str]]:
        """Joint names by trajectory controller, as the streamer publishes them."""
        return {self.arm_controller: list(self.arm_joints), self.hand_controller: [self.hand_joint]}

    @property
    def joint_names(self) -> tuple[str, ...]:
        return (*self.arm_joints, *self.finger_joints)

    def state(self, twist: Sequence[float], joints: Mapping[str, float]) -> np.ndarray:
        """The state vector from the odometry twist and the joint states by name (the finger joints that did not report
        take the first one's value)."""
        fingers = [joints.get(j, joints[self.finger_joints[0]]) for j in self.finger_joints]
        return np.array([*twist, *(joints[j] for j in self.arm_joints), *fingers], dtype=np.float32)

    def action(self, a: Sequence[float]) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """One action vector as (positions by controller, base twist), limited; ValueError on a wrong size or a non-finite value."""
        a = np.asarray(a, dtype=np.float64)
        if a.shape != (self.action_dim,) or not np.isfinite(a).all():
            raise ValueError(f"action of shape {a.shape} (want {self.action_dim}) or with non-finite values: {a}")
        lim = np.array(self.twist_limits)
        arm = np.clip(a[TWIST:-1], *np.array(self.arm_limits).T)
        return ({self.arm_controller: arm, self.hand_controller: np.clip(a[-1:], *self.hand_limits)},
                np.clip(a[:TWIST], -lim, lim))


# Franka Emika Panda's joint ranges; the hand: 0.04 open, a goal past closed squeezes (DATAGEN-v2.md 5.2)
MOBILE_PANDA = Contract(
    robot="mobile_panda",
    cameras={"camera1": "wrist_cam", "camera2": "overhead_left_cam", "camera3": "overhead_right_cam"},
    arm_controller="panda_arm_controller", arm_joints=tuple(f"panda_joint{i}" for i in range(1, 8)),
    hand_controller="panda_hand_controller", hand_joint="panda_finger_joint1",
    finger_joints=("panda_finger_joint1", "panda_finger_joint2"), odom="/omni_wheel_drive_controller/odom",
    arm_limits=((-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698), (-2.8973, 2.8973),
                (-0.0175, 3.7525), (-2.8973, 2.8973)),
    hand_limits=(-0.04, 0.04), twist_limits=(0.3, 0.3, 0.6))

CONTRACTS = {c.robot: c for c in (MOBILE_PANDA,)}
