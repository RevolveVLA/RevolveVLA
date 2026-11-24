"""verdict: fires when the episode's evaluator result says it failed. Only recorded episodes carry a result (on the
last tick); live, the evaluator ends the episode instead. Reason: `<failed_phase>:<reason>:<check>`."""
from __future__ import annotations

from ..core import Tick, TriggerSpec
from . import register


@register("verdict")
class Verdict:
    Spec = TriggerSpec

    def __init__(self, params, task, **_):
        self.phases = tuple(params.phases) if params.phases else None

    def reset(self) -> None:
        pass

    def check(self, tick: Tick) -> str | None:
        r = tick.result
        if not r or r.get("passed", True):
            return None
        return ":".join(str(x) for x in (r.get("failed_phase"), r.get("reason"), r.get("check")) if x)
