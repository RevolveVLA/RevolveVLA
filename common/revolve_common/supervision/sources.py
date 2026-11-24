"""Where ticks come from: `LiveSource` (the running sim, via a SimClient) and `BagSource` (a recorded episode). No
rclpy / rosbag2 imports here: the live robot state and the bag reader are injected."""
from __future__ import annotations

import bisect
import json
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Mapping, NamedTuple

from .core import Tick

STATE, RESULT, JOINTS = "/evaluator/state", "/evaluator/result", "/joint_states"
OBJECT_JOINTS = re.compile(r"^/world/[^/]+/model/([^/]+)/joint_state$")


class LiveSource:
    """`tick()` -> the Tick of the newest /evaluator/state of a running episode, or None (no new state since the last
    call, or no episode running). Tick.t counts from the first state of the episode it sees (poll
    faster than 10 Hz). `joints` / `base`: callables for the robot's current joint positions {name: rad} and base
    (x, y, yaw), or None; SimClient has no such accessors yet (it keeps /evaluator/* only), so the caller supplies
    them."""

    def __init__(self, client, joints: Callable[[], Mapping[str, float] | None] | None = None,
                 base: Callable[[], tuple[float, float, float] | None] | None = None):
        self.client, self.joints, self.base = client, joints, base
        self._last, self._episode, self._t0 = None, None, 0.0

    def tick(self) -> Tick | None:
        s = self.client.state()
        if s is None or s is self._last or s.get("episode") is None:
            return None
        self._last = s
        if s["episode"] != self._episode:
            self._episode, self._t0 = s["episode"], s["t"]
        return Tick(s["t"] - self._t0, s.get("phase"), s.get("values") or {},
                    self.joints() if self.joints else None, self.base() if self.base else None)


class BagMessage(NamedTuple):
    """One decoded bag message. `t`: sim time [s] (header stamp). `data` by topic:
    /evaluator/state and /evaluator/result the JSON str; /joint_states and a scene model's joint_state
    `{"name": [...], "position": [...]}`; /model/<robot>/pose the model's pose `(x, y, yaw)`."""
    topic: str
    t: float
    data: object


@dataclass(frozen=True)
class RobotState:
    """The world at one sim time. `joints`: every robot joint (arm and gripper); `objects`: scene joints keyed
    `<model>::<joint>`."""
    t: float
    base: tuple[float, float, float] | None
    joints: Mapping[str, float]
    objects: Mapping[str, float]


class _Series:
    def __init__(self, items):
        items = sorted(items, key=lambda x: x[0])
        self.t, self.v = [t for t, _ in items], [v for _, v in items]

    def at(self, t: float):
        """The last value at or before t (None before the first)."""
        i = bisect.bisect_right(self.t, t) - 1
        return self.v[i] if i >= 0 else None


class BagSource:
    """The ticks (and robot states) of one recorded episode. `read_messages(bag, wanted)` yields the BagMessages of the
    bag whose topic satisfies `wanted(topic)` (the real one uses rosbag2_py, in the simulator image; tests pass a fake).
    `result`: the episode's `episode_<n>.json` (default: the one in the bag); its `sim_time_start` is the time zero of
    `Tick.t`, else the first state of the episode. A failed result becomes the last tick (phase = its failed phase)."""

    def __init__(self, read_messages: Callable[[str, Callable[[str], bool]], Iterable[BagMessage]], bag: str,
                 robot: str, result: Mapping | None = None):
        pose, rows = f"/model/{robot}/pose", {"state": [], "joints": [], "pose": [], "result": []}
        objects: dict[str, list] = {}
        for m in read_messages(bag, lambda t: t in (STATE, RESULT, JOINTS, pose) or bool(OBJECT_JOINTS.match(t))):
            if m.topic == STATE:
                rows["state"].append((m.t, json.loads(m.data)))
            elif m.topic == RESULT:
                rows["result"].append((m.t, json.loads(m.data)))
            elif m.topic == JOINTS:
                rows["joints"].append((m.t, dict(zip(m.data["name"], m.data["position"]))))
            elif m.topic == pose:
                rows["pose"].append((m.t, tuple(m.data)))
            elif obj := OBJECT_JOINTS.match(m.topic):
                objects.setdefault(obj[1], []).append((m.t, {f"{obj[1]}::{n}": p for n, p in
                                                             zip(m.data["name"], m.data["position"])}))
        self.states = [(t, s) for t, s in sorted(rows["state"], key=lambda x: x[0]) if s.get("episode") is not None]
        self.result = result or (rows["result"][-1][1] if rows["result"] else None)
        self._joints, self._pose = _Series(rows["joints"]), _Series(rows["pose"])
        self._objects = {k: _Series(v) for k, v in objects.items()}
        self.t0 = (self.result or {}).get("sim_time_start", self.states[0][1]["t"] if self.states else 0.0)

    def state_at(self, t: float) -> RobotState:
        """The last messages at or before `t` (episode time), no interpolation."""
        a = self.t0 + t
        objs: dict[str, float] = {}
        for series in self._objects.values():
            objs.update(series.at(a) or {})
        return RobotState(t, self._pose.at(a), self._joints.at(a) or {}, objs)

    def ticks(self) -> Iterator[Tick]:
        last = 0.0
        for _, s in self.states:
            last = s["t"] - self.t0
            r = self.state_at(last)
            yield Tick(last, s.get("phase"), s.get("values") or {}, r.joints or None, r.base)
        r = self.result
        if r and not r.get("passed", True):
            end = max(last, r.get("t_fail") or 0.0)
            now = self.state_at(end)
            values = self.states[-1][1].get("values") or {} if self.states else {}
            yield Tick(end, r.get("failed_phase"), values, now.joints or None, now.base, r)
