"""The expert's own actions replayed through the same ROS 2 path as a policy's (no checkpoint, no policy server): the 30 Hz
action series of a passed episode datagen recorded on the same scene instance (the training set's `item.episodes`)."""
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

    def client_args(self, item):
        bags = [e["bag"] for e in item.episodes if e.get("passed") and e.get("bag")]
        if not bags:
            raise EvalError(f"policy oracle: no passed datagen episode with a bag for seed {item.config.seed} "
                            f"(scene {item.scene}): the oracle replays one, use a `train` set of a recorded run")
        return ["--bag", str(bags[0])]
