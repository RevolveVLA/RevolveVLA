"""condition: a /evaluator/state value compared with a threshold, for `for_s` seconds.
`{plugin: condition, value: "open.while[1]", op: gt, threshold: 0.01, for_s: 0.2}`; value = `<phase>.done[i]`,
`<phase>.while[i]` or `sanity.<check>`."""
from __future__ import annotations

from typing import Literal

from ...specs import SpecError
from ..core import Held, Tick, TriggerSpec, compare
from . import register


class Spec(TriggerSpec):
    value: str
    op: Literal["lt", "le", "gt", "ge", "eq", "ne"]
    threshold: float
    for_s: float = 0.0


@register("condition")
class Condition:
    Spec = Spec

    def __init__(self, params, task, *, path="condition", **_):
        if not task.has_value(params.value):
            raise SpecError(f"{path}.value: unknown value {params.value!r} (values: {', '.join(sorted(task.values))}, "
                            f"sanity.<check>)")
        self.p, self.phases = params, tuple(params.phases) if params.phases else None
        self.held = Held(params.for_s)

    def reset(self) -> None:
        self.held.reset()

    def check(self, tick: Tick) -> str | None:
        p, v = self.p, tick.values.get(self.p.value)
        if self.held.update(tick.t, compare(v, p.op, p.threshold)):
            return f"{p.value}={v:.4g} {p.op} {p.threshold:g}" + (f" for {p.for_s:g} s" if p.for_s else "")
        return None
