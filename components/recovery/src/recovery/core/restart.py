"""Where a failed episode is restarted: the time (the failed phase's start, or a little before the failure) and the robot
state the bag has then, as a scene instance's `robot_start` (base pose + arm joints). The gripper and the scene's joints
cannot be set at a start, so a state in which a scene joint has moved (the door opened) is `unreproducible`, with the value."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from revolve_common.supervision import BagSource

DIGITS = 6


@dataclass(frozen=True)
class Restart:
    """`t`: episode seconds in the failed episode. `reason`: why the state cannot be reproduced (None: it can)."""
    t: float
    base: tuple[float, float, float] | None = None
    arm: tuple[float, ...] | None = None
    object_joint: float | None = None           # the largest scene joint position (absolute) at t
    reason: str | None = None

    @property
    def reproducible(self) -> bool:
        return self.reason is None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Restart:
        return cls(d["t"], d["base"] and tuple(d["base"]), d["arm"] and tuple(d["arm"]), d["object_joint"], d["reason"])


def restart_time(restart_from: str, back_s: float, t_fail: float, phase_t_start: float) -> float:
    if restart_from == "phase_start":
        return phase_t_start
    return round(max(phase_t_start, t_fail - back_s), 3)


def restart_at(source: BagSource, t: float, arm_joints: Sequence[str], max_object_joint: float) -> Restart:
    """The restart of episode time `t` from the bag (the last messages at or before it; at the very start of the episode, the first state)."""
    state = source.state_at(t)
    if t >= 0 and (first := source.first_state_t) is not None and first > t:
        state = source.state_at(first)
    if state.base is None or any(j not in state.joints for j in arm_joints):
        return Restart(t, reason=f"no robot state in the bag at t={t:.2f} s")
    name, value = max(state.objects.items(), key=lambda kv: abs(kv[1]), default=(None, 0.0))
    base = tuple(round(v, DIGITS) for v in state.base)
    arm = tuple(round(state.joints[j], DIGITS) for j in arm_joints)
    why = f"{name} = {value:.3f} is beyond {max_object_joint}" if abs(value) > max_object_joint else None
    return Restart(t, base, arm, round(abs(value), DIGITS), why)
