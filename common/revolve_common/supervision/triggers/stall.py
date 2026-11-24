"""stall: the robot stands still. Over the last `window_s` the peak-to-peak of every arm joint (the gripper's are
excluded) and, with `include_base`, of the base x, y and yaw is below `min_motion` [rad / m], and the phase is not done.
`joints`: `arm` (the `arm_joints` given to `build`, else every joint that is not a gripper's), `all`, or a name list."""
from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np

from ..core import Tick, TriggerSpec, is_gripper
from . import register


class Spec(TriggerSpec):
    window_s: float
    min_motion: float
    joints: Literal["arm", "all"] | list[str] = "arm"
    include_base: bool = True


@register("stall")
class Stall:
    Spec = Spec

    def __init__(self, params, task, *, arm_joints=None, **_):
        self.p, self.task, self.arm = params, task, arm_joints
        self.phases = tuple(params.phases) if params.phases else None
        self.reset()

    def reset(self) -> None:
        self.samples, self.channels = deque(), None

    def sample(self, tick: Tick) -> np.ndarray | None:
        p, joints = self.p, tick.joints or {}
        names = sorted(joints) if p.joints == "all" else sorted(p.joints) if isinstance(p.joints, list) else \
            sorted(j for j in joints if not is_gripper(j) and (self.arm is None or j in self.arm))
        base = list(tick.base) if p.include_base and tick.base is not None else []
        if p.include_base and tick.base is None or any(n not in joints for n in names) or not (names or base):
            return None
        if self.channels is None:
            self.channels = names
        return np.array([joints[n] for n in self.channels] + base) if names == self.channels else None

    def check(self, tick: Tick) -> str | None:
        x = self.sample(tick)
        if x is None:
            return None
        self.samples.append((tick.t, x))
        while len(self.samples) > 1 and self.samples[1][0] <= tick.t - self.p.window_s:
            self.samples.popleft()
        if tick.t - self.samples[0][0] < self.p.window_s - 1e-9 or self.task.is_done(tick.phase, tick.values):
            return None
        data = np.array([s for _, s in self.samples])
        if self.p.include_base:
            data[:, -1] = np.unwrap(data[:, -1])
        motion = float(np.ptp(data, axis=0).max())
        if motion < self.p.min_motion:
            return f"stalled: moved {motion:.4f} < {self.p.min_motion:g} in {self.p.window_s:g} s"
        return None
