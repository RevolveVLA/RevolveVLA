"""SmolVLA served from the checkpoint (policy server container + Rosetta's policy runner)."""
from __future__ import annotations

from ...core.errors import EvalError
from ...core.interfaces import Policy
from ...core.registry import REGISTRY


@REGISTRY.register("policy", "smolvla")
class SmolVLA(Policy):
    name = "smolvla"

    def __init__(self, spec, params):
        if params:
            raise EvalError(f"policy smolvla: takes no parameters, got {sorted(params)}")
