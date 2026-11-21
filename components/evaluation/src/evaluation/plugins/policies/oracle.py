"""The expert's own actions replayed through the same ROS 2 path (no checkpoint, no policy server)."""
from __future__ import annotations

from ...core.errors import EvalError
from ...core.interfaces import Policy
from ...core.registry import REGISTRY


@REGISTRY.register("policy", "oracle")
class Oracle(Policy):
    name = "oracle"
    needs_ckpt = False
    client_only = True

    def __init__(self, spec, params):
        if params:
            raise EvalError(f"policy oracle: takes no parameters, got {sorted(params)}")
