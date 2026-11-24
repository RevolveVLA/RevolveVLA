"""policy frozen: the oracle's replay held at its first action (the robot never moves: `stall` and `phase_time` fire). No params."""
from __future__ import annotations

import numpy as np
from evaluation.core.errors import EvalError
from evaluation.core.registry import REGISTRY
from evaluation.plugins.policies.oracle import Oracle


def perturb(series, params):
    """The series with every action as the first one and no base twist."""
    return type(series)(series.joints, {c: np.repeat(p[:1], len(p), axis=0) for c, p in series.positions.items()},
                        np.zeros_like(series.base))


@REGISTRY.register("policy", "frozen")
class Frozen(Oracle):
    name = "frozen"

    def __init__(self, spec, params):
        if params:
            raise EvalError(f"policy frozen: takes no parameters, got {sorted(params)}")

    def perturbation(self, item):
        return {"name": "frozen"}
