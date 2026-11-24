"""phase_time: fires once the active phase has run `fraction` of its `timeout_s` (task.yaml), or `after_s` seconds,
i.e. before the evaluator's own timeout. The phase clock starts at the first tick the trigger sees in the phase."""
from __future__ import annotations

from pydantic import model_validator

from ...specs import SpecError
from ..core import Tick, TriggerSpec
from . import register


class Spec(TriggerSpec):
    fraction: float | None = None
    after_s: float | None = None

    @model_validator(mode="after")
    def _one(self):
        if (self.fraction is None) == (self.after_s is None):
            raise ValueError("exactly one of fraction, after_s")
        if (self.after_s if self.fraction is None else self.fraction) <= 0:
            raise ValueError("fraction / after_s must be > 0")
        return self


@register("phase_time")
class PhaseTime:
    Spec = Spec

    def __init__(self, params, task, *, path="phase_time", **_):
        self.p, self.task, self.phases = params, task, tuple(params.phases) if params.phases else None
        bare = [n for n in self.phases or () if task.phases[n].timeout_s is None]
        if params.fraction is not None and bare:
            raise SpecError(f"{path}.fraction: phase {bare} has no timeout_s")
        self.t0 = None

    def reset(self) -> None:
        self.t0 = None

    def limit(self, phase: str) -> float | None:
        if self.p.after_s is not None:
            return self.p.after_s
        info = self.task.phases.get(phase)
        return None if info is None or info.timeout_s is None else self.p.fraction * info.timeout_s

    def check(self, tick: Tick) -> str | None:
        if self.t0 is None:
            self.t0 = tick.t
        limit = self.limit(tick.phase)
        if limit is not None and tick.t - self.t0 >= limit:
            return f"{tick.phase} running {tick.t - self.t0:.1f} s >= {limit:.1f} s"
        return None
