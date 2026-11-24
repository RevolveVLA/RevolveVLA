"""Warm reset: put the robot at a start other than the scene's without relaunching the sim. After `/simulator/reset` (the
launch instance's start) the base is driven to the target pose on cmd_vel, closed loop on the gz ground truth, and the arm to
the target joints with one joint_trajectory, all in sim time through the controllers' interfaces (no teleport, no gz call).
It runs before `/evaluator/start`, so none of it is recorded. Deltas too large for a straight, unobstructed move are refused.

The motion is pure (`run`, over a `Plant`: tests drive it with a simulated robot); `RosPlant` is the rclpy side and
is only imported in a client container. `SimClient.reset_to` ties both to the reset service."""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict, dataclass, replace
from typing import Protocol, Sequence

CMD_VEL = "/omni_wheel_drive_controller/cmd_vel"
POLL_S = 0.02                       # control period [sim s]
RAMP_S = 0.5                        # the base command ramps up over this long (the controller has no limiter)
SETTLE_S = 0.4                      # zero twists before the final measurement
ARM_RATE = 0.5                      # rad/s of the slowest joint's move
ARM_MIN_S = 2.0
MISS_EXIT = 79                      # exit code of a client whose warm reset missed (the host falls back to a relaunch)


@dataclass(frozen=True)
class ResetTarget:
    """The start to reach: base pose [x, y, yaw] in the world, arm joints, and where to command the arm."""
    robot: str
    base: tuple[float, float, float]
    arm: tuple[float, ...]
    arm_joints: tuple[str, ...]
    arm_controller: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> ResetTarget:
        d = json.loads(text)
        return cls(d["robot"], tuple(d["base"]), tuple(d["arm"]), tuple(d["arm_joints"]), d["arm_controller"])


@dataclass(frozen=True)
class Tolerance:
    base_m: float = 0.03
    yaw_rad: float = 0.03
    arm_rad: float = 0.03
    max_base_m: float = 0.3         # larger deltas are refused: a straight line is only known to be free for small ones
    max_yaw_rad: float = 0.6
    max_arm_rad: float = 1.0
    v_max: float = 0.2              # m/s
    w_max: float = 0.4              # rad/s


@dataclass(frozen=True)
class ResetReport:
    ok: bool
    base_err_m: float | None = None
    yaw_err_rad: float | None = None
    arm_err_rad: float | None = None
    wall_s: float = 0.0
    sim_s: float = 0.0
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def with_times(self, wall_s: float, sim_s: float) -> ResetReport:
        return replace(self, wall_s=round(wall_s, 3), sim_s=round(sim_s, 3))


class Plant(Protocol):
    """The robot as the motion sees it. `pose` / `joints` are None until a message arrived since `forget()`."""

    def forget(self) -> None: ...
    def pose(self) -> tuple[float, float, float] | None: ...
    def joints(self) -> dict[str, float] | None: ...
    def twist(self, v: Sequence[float]) -> None: ...
    def command_arm(self, q: Sequence[float], duration_s: float) -> None: ...
    def now(self) -> float: ...
    def sleep(self, sim_s: float) -> None: ...


def angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def deltas(pose, joints, target: ResetTarget) -> tuple[float, float, float]:
    """-> (base distance [m], |yaw error| [rad], largest arm joint error [rad]) from the current state to the target."""
    return (math.hypot(target.base[0] - pose[0], target.base[1] - pose[1]), abs(angle(target.base[2] - pose[2])),
            max(abs(joints[j] - q) for j, q in zip(target.arm_joints, target.arm)))


def refusal(d: tuple[float, float, float], tol: Tolerance) -> str | None:
    """Why a move of these deltas is not attempted, else None."""
    for what, got, limit, unit in (("base", d[0], tol.max_base_m, "m"), ("yaw", d[1], tol.max_yaw_rad, "rad"),
                                   ("arm", d[2], tol.max_arm_rad, "rad")):
        if got > limit:
            return f"{what} delta {got:.3f} {unit} is over the {limit} {unit} a warm reset takes"
    return None


def base_command(pose, target: ResetTarget, tol: Tolerance, elapsed_s: float) -> tuple[float, float, float]:
    """The body twist (vx, vy, wz) towards the target: proportional, clipped to the rates, never below a creep speed while
    outside half the tolerance (the omni base strafes), ramped up over RAMP_S."""
    ex, ey = target.base[0] - pose[0], target.base[1] - pose[1]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    bx, by = c * ex + s * ey, -s * ex + c * ey                      # the error in the body frame
    dist, yaw_err = math.hypot(bx, by), angle(target.base[2] - pose[2])
    up = min(1.0, elapsed_s / RAMP_S)
    v = 0.0 if dist <= tol.base_m / 2 else min(tol.v_max, max(0.03, 1.5 * dist))
    w = 0.0 if abs(yaw_err) <= tol.yaw_rad / 2 else min(tol.w_max, max(0.05, 1.5 * abs(yaw_err)))
    return ((up * v * bx / dist, up * v * by / dist) if v else (0.0, 0.0)) + (up * math.copysign(w, yaw_err),)


def wait_for(plant: Plant, read, timeout_s: float):
    t0 = plant.now()
    while (got := read()) is None:
        if plant.now() - t0 > timeout_s:
            return None
        plant.sleep(POLL_S)
    return got


def run(plant: Plant, target: ResetTarget, tol: Tolerance = Tolerance(), timeout_s: float = 30.0) -> ResetReport:
    """Drive the plant (just reset, nothing started) to `target` -> a report without wall times (`SimClient.reset_to` adds them)."""
    plant.forget()
    pose, joints = wait_for(plant, plant.pose, 10.0), wait_for(plant, plant.joints, 10.0)
    if pose is None or joints is None:
        return ResetReport(False, reason="no robot pose / joint states after the reset")
    if why := refusal(deltas(pose, joints, target), tol):
        return ResetReport(False, *deltas(pose, joints, target), reason=why)
    t0 = plant.now()
    plant.command_arm(target.arm, max(ARM_MIN_S, deltas(pose, joints, target)[2] / ARM_RATE))
    while (t := plant.now() - t0) < timeout_s:
        pose, joints = plant.pose(), plant.joints()
        base_d, yaw_d, arm_d = deltas(pose, joints, target)
        if base_d <= tol.base_m / 2 and yaw_d <= tol.yaw_rad / 2 and arm_d <= tol.arm_rad:
            break
        plant.twist(base_command(pose, target, tol, t))
        plant.sleep(POLL_S)
    plant.twist((0.0, 0.0, 0.0))
    plant.sleep(SETTLE_S)
    pose, joints = plant.pose(), plant.joints()
    base_d, yaw_d, arm_d = deltas(pose, joints, target)
    ok = base_d <= tol.base_m and yaw_d <= tol.yaw_rad and arm_d <= tol.arm_rad
    why = None if ok else (f"missed the target after {plant.now() - t0:.1f} s: base {base_d:.3f} m, yaw {yaw_d:.3f} rad, "
                           f"arm {arm_d:.3f} rad")
    return ResetReport(ok, round(base_d, 4), round(yaw_d, 4), round(arm_d, 4), sim_s=round(plant.now() - t0, 3), reason=why)



class RosPlant:
    """`Plant` over the sim's topics, on the node of a `SimClient` (rclpy: client containers only)."""

    def __init__(self, client, target: ResetTarget):
        from geometry_msgs.msg import TwistStamped
        from sensor_msgs.msg import JointState
        from tf2_msgs.msg import TFMessage
        from trajectory_msgs.msg import JointTrajectory
        self.client, self.node, self._lock = client, client.node, threading.Lock()
        robot, self._names = target.robot, target.arm_joints
        self._pose = self._joints = None
        self._twist_type, self._traj_type = TwistStamped, JointTrajectory
        self.node.create_subscription(TFMessage, f"/model/{robot}/pose", self._on_pose, 50)
        self.node.create_subscription(JointState, "/joint_states", self._on_joints, 50)
        self.cmd = self.node.create_publisher(TwistStamped, CMD_VEL, 10)
        self.arm = self.node.create_publisher(JointTrajectory, f"/{target.arm_controller}/joint_trajectory", 10)

    def _on_pose(self, m) -> None:
        t = m.transforms[-1].transform
        q = t.rotation
        with self._lock:
            self._pose = (t.translation.x, t.translation.y,
                          math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))

    def _on_joints(self, m) -> None:
        with self._lock:
            self._joints = dict(zip(m.name, m.position))

    def forget(self) -> None:
        with self._lock:
            self._pose = self._joints = None

    def pose(self):
        with self._lock:
            return self._pose

    def joints(self):
        with self._lock:
            return self._joints and dict(self._joints)

    def twist(self, v) -> None:
        m = self._twist_type()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.header.frame_id = "base_footprint"
        m.twist.linear.x, m.twist.linear.y, m.twist.angular.z = (float(c) for c in v)
        self.cmd.publish(m)

    def command_arm(self, q, duration_s: float) -> None:
        """One point, `duration_s` away: the controller interpolates from where the arm is (a zero stamp: start now)."""
        self.client.wait_for(lambda: self.arm.get_subscription_count() > 0 and self.cmd.get_subscription_count() > 0, 30.0)
        from trajectory_msgs.msg import JointTrajectoryPoint
        m = self._traj_type()
        m.joint_names = list(self._names)
        p = JointTrajectoryPoint(positions=[float(x) for x in q])
        p.time_from_start.sec, p.time_from_start.nanosec = int(duration_s), int((duration_s % 1) * 1e9)
        m.points = [p]
        self.arm.publish(m)

    def now(self) -> float:
        return self.client.now()

    def sleep(self, sim_s: float) -> None:
        t0, w0 = self.now(), time.monotonic()
        while self.now() - t0 < sim_s and time.monotonic() - w0 < 20 * sim_s + 1:
            time.sleep(0.005)
