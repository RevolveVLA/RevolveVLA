"""The oracle's actions: the 30 Hz action series of a datagen episode bag (DATAGEN-v2.md 5.3, 7.4; the `action_series` of the
simulator's check_contract.py): per joint trajectory controller its `controller_state` reference positions (the last state at
or before each tick) and the base twist (the last `cmd_vel` within the controller's 0.5 s timeout, else zero).

Plain numpy, plus `read_bag` (rosbag2_py, imported there): it runs inside the simulator image's client container."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ACTION_HZ = 30.0
CMD_VEL_TIMEOUT_S = 0.5             # the omni controller's cmd_vel_timeout: no command for longer = a zero twist
CMD_VEL = "/omni_wheel_drive_controller/cmd_vel"


@dataclass(frozen=True)
class ActionSeries:
    """`ticks` actions at ACTION_HZ: `joints` = {controller: joint names}, `positions` = {controller: (ticks, n joints)},
    `base` = (ticks, 3) twist (x, y, yaw rate) in the robot frame."""
    joints: Mapping[str, Sequence[str]]
    positions: Mapping[str, np.ndarray]
    base: np.ndarray

    @property
    def ticks(self) -> int:
        return len(self.base)

    def at(self, k: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """The action of tick k; past the end the last arm / hand references are held and the base stops."""
        last = self.ticks - 1
        return ({c: p[min(k, last)] for c, p in self.positions.items()},
                self.base[k] if k <= last else np.zeros(3))


def stamp(m) -> float:
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def hold(stamps: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """The last value at or before each grid time (the first one before the first stamp)."""
    return values[np.clip(np.searchsorted(stamps, grid, side="right") - 1, 0, len(stamps) - 1)]


def from_messages(states: Mapping[str, Sequence], cmds: Sequence) -> ActionSeries:
    """The series over the span every controller has states (`states` = {controller: [JointTrajectoryControllerState]},
    `cmds` = [TwistStamped]; time-ordered)."""
    if not states or not all(states.values()):
        raise ValueError("no controller_state messages in the bag")
    t0, t1 = max(stamp(ms[0]) for ms in states.values()), min(stamp(ms[-1]) for ms in states.values())
    grid = np.arange(t0, t1, 1.0 / ACTION_HZ)
    if not len(grid):
        raise ValueError(f"the controller states span {t1 - t0:.3f} s: no action tick")
    positions = {c: hold(np.array([stamp(m) for m in ms]), np.array([m.reference.positions for m in ms]), grid)
                 for c, ms in states.items()}
    base = np.zeros((len(grid), 3))
    if cmds:
        ct = np.array([stamp(m) for m in cmds])
        cv = np.array([[m.twist.linear.x, m.twist.linear.y, m.twist.angular.z] for m in cmds])
        i = np.searchsorted(ct, grid, side="right") - 1
        ok = (i >= 0) & (grid - ct[np.clip(i, 0, None)] <= CMD_VEL_TIMEOUT_S)
        base[ok] = cv[i[ok]]
    return ActionSeries({c: list(ms[0].joint_names) for c, ms in states.items()}, positions, base)


def read_bag(path) -> ActionSeries:
    """The action series of an episode bag (rosbag2 mcap directory)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(Path(path)), storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    wanted = [t for t in types if t.endswith("/controller_state") or t == CMD_VEL]
    msgs = {t: [] for t in wanted}
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic in msgs:
            msgs[topic].append(deserialize_message(data, get_message(types[topic])))
    return from_messages({t.split("/")[1]: ms for t, ms in msgs.items() if t != CMD_VEL}, msgs.get(CMD_VEL, []))
