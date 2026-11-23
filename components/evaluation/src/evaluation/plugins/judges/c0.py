"""C0: the simulator's phase verdict (every phase of the rollout passed), as the rollout record says."""
from __future__ import annotations

from ...core.errors import EvalError
from ...core.interfaces import Judge, Verdict
from ...core.registry import REGISTRY


@REGISTRY.register("judge", "c0")
class C0(Judge):
    def __init__(self, spec, params):
        if params:
            raise EvalError(f"judge c0: takes no parameters, got {sorted(params)}")

    def score(self, record, rollout_dir, task):
        return Verdict(bool(record["passed"]), {"criterion": "c0"})
