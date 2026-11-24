"""Phase supervision: "is this phase failing, and why?", the same code live (client container, fed by /evaluator/state)
and offline (from a recorded bag). A `Tick` is one observation; a `Trigger` looks at ticks and says why a phase is
failing; a `Supervisor` runs an ordered list of triggers and latches the first firing. Pure python + numpy.

Triggers are specs `{plugin: <name>, phases: [<phase>, ...]?, ...}` (`phase: <p>` = `phases: [<p>]`), built with
`build(specs, task)`; the phase names and value keys are checked against the task. Plugins: `triggers/`.
"""
from __future__ import annotations

import operator
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Protocol, Sequence

from pydantic import ValidationError, model_validator

from ..specs import Strict, SpecError
from . import triggers as registry

OPS = {"lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge, "eq": operator.eq,
       "ne": operator.ne}
GRIPPER_HINTS = ("finger", "gripper", "jaw")


def compare(value, op: str, threshold: float) -> bool:
    """`value <op> threshold`; a missing or non-finite (None) value compares false."""
    return value is not None and OPS[op](value, threshold)


def is_gripper(joint: str) -> bool:
    return any(h in joint.lower() for h in GRIPPER_HINTS)


@dataclass(frozen=True)
class Tick:
    """One observation. `t`: sim s since the episode start. `phase`: the active phase (None between episodes; on the
    last tick of a failed recorded episode, its failed phase). `values`: /evaluator/state `values`. `joints`: robot
    joint name -> position; `base`: (x, y, yaw) in the world. `result`: the episode's evaluator result, only on the
    last tick of a recorded episode."""
    t: float
    phase: str | None = None
    values: Mapping[str, float | None] = field(default_factory=dict)
    joints: Mapping[str, float] | None = None
    base: tuple[float, float, float] | None = None
    result: Mapping | None = None


@dataclass(frozen=True)
class Firing:
    trigger: str
    reason: str
    phase: str
    t: float


class Trigger(Protocol):
    name: str
    phases: tuple[str, ...] | None          # None = every phase

    def reset(self) -> None: ...
    def check(self, tick: Tick) -> str | None: ...      # the reason it fires, or None


class Held:
    """Debounce: `update(t, cond)` is true once `cond` held on every tick for `seconds` of tick time."""

    def __init__(self, seconds: float = 0.0):
        self.seconds, self.since = seconds, None

    def reset(self) -> None:
        self.since = None

    def update(self, t: float, cond: bool) -> bool:
        if not cond:
            self.since = None
            return False
        if self.since is None:
            self.since = t
        return t - self.since >= self.seconds - 1e-9


# ------------------------------------------------------------------ the task, as far as supervision needs it
@dataclass(frozen=True)
class PhaseInfo:
    name: str
    timeout_s: float | None
    done: tuple[tuple[str, str, float], ...]        # (value key, op, threshold) of every done condition


@dataclass(frozen=True)
class TaskInfo:
    phases: Mapping[str, PhaseInfo]
    values: frozenset[str]                          # the /evaluator/state value keys of the task
    sanity: frozenset[str] | None = None            # sanity check names; None = unknown (any `sanity.<k>` is taken)

    @classmethod
    def of(cls, task) -> TaskInfo:
        """From a `specs.Task`, or a `specs.Resolved` (adds the merged sanity keys)."""
        sanity = None
        if hasattr(task, "task"):
            task, sanity = task.task, frozenset(task.sanity)
        phases, values = {}, set()
        for p in task.phases:
            keys = [f"{p.name}.done[{i}]" for i in range(len(p.done))]
            values.update(keys, (f"{p.name}.while[{i}]" for i in range(len(p.while_))))
            phases[p.name] = PhaseInfo(p.name, p.timeout_s, tuple((k, c.op, c.value) for k, c in zip(keys, p.done)))
        return cls(phases, frozenset(values), sanity)

    def has_value(self, key: str) -> bool:
        return key in self.values or (key.startswith("sanity.") and (self.sanity is None or key[7:] in self.sanity))

    def is_done(self, phase: str, values: Mapping) -> bool:
        """Every done condition of `phase` holds right now (hold times ignored)."""
        info = self.phases.get(phase)
        return info is not None and all(compare(values.get(k), op, v) for k, op, v in info.done)


# ------------------------------------------------------------------ specs
class TriggerSpec(Strict):
    """The keys every trigger spec has; a plugin's `Spec` adds its own."""
    plugin: str
    phases: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _phase(cls, v):
        if isinstance(v, dict) and "phase" in v:
            v = {k: x for k, x in v.items() if k != "phase"} | {"phases": [v["phase"]]}
        return v


def parse(model, spec, path: str):
    """`model.model_validate(spec)`; the errors as a SpecError naming the yaml path."""
    try:
        return model.model_validate(spec)
    except ValidationError as e:
        raise SpecError("; ".join(f"{path}{''.join(f'.{x}' if isinstance(x, str) else f'[{x}]' for x in err['loc'])}: "
                                  f"{err['msg']}" for err in e.errors())) from None


def build(specs: Sequence[Mapping], task, *, arm_joints: Iterable[str] | None = None, hz: float = 10.0,
          path: str = "triggers") -> Supervisor:
    """A Supervisor of the trigger specs (priority = list order), checked against `task` (a specs.Task / Resolved).
    `arm_joints`: the robot's arm joint names (stall's `joints: arm`; default: every joint that is not a gripper's)."""
    info = TaskInfo.of(task)
    built = []
    for i, spec in enumerate(specs):
        at = f"{path}[{i}]"
        if not isinstance(spec, Mapping) or "plugin" not in spec:
            raise SpecError(f"{at}: a trigger is a mapping with a `plugin` key, got {spec!r}")
        cls = registry.get(spec["plugin"], at)
        params = parse(cls.Spec, spec, at)
        unknown = [p for p in params.phases or () if p not in info.phases]
        if unknown:
            raise SpecError(f"{at}.phases: unknown phase {unknown} (phases: {', '.join(info.phases)})")
        built.append(cls(params, info, arm_joints=None if arm_joints is None else frozenset(arm_joints), path=at))
    return Supervisor(built, hz)


class Supervisor:
    """Ordered (list order = priority), latched: the first trigger to fire wins and every later `check` returns that
    `Firing`. Triggers see only ticks of their phases; all are reset when the phase changes. `hz`: the nominal tick
    rate (`period` = how often a live loop should check). `history`: how often each trigger fired (a tick may fire
    several; the first wins)."""

    def __init__(self, triggers: Sequence[Trigger], hz: float = 10.0):
        self.triggers, self.hz = list(triggers), hz
        self.history: Counter = Counter()
        self.start()

    @property
    def period(self) -> float:
        return 1.0 / self.hz

    @property
    def firing(self) -> Firing | None:
        return self._firing

    def start(self) -> None:
        """A new episode: unlatch, reset every trigger and the history."""
        self._firing, self._phase, self._t = None, None, None
        self.history.clear()
        for tr in self.triggers:
            tr.reset()

    def check(self, tick: Tick) -> Firing | None:
        if self._firing or tick.phase is None or (self._t is not None and tick.t < self._t):
            return self._firing
        if tick.phase != self._phase:
            self._phase = tick.phase
            for tr in self.triggers:
                tr.reset()
        self._t = tick.t
        fired = [(tr, why) for tr in self.triggers if tr.phases is None or tick.phase in tr.phases
                 if (why := tr.check(tick)) is not None]
        self.history.update(tr.name for tr, _ in fired)
        if fired:
            self._firing = Firing(fired[0][0].name, fired[0][1], tick.phase, tick.t)
        return self._firing

    def scan(self, ticks: Iterable[Tick]) -> Firing | None:
        """`start()`, then `check` the ticks until one fires (offline: a recorded episode)."""
        self.start()
        for tick in ticks:
            if self.check(tick):
                break
        return self._firing
