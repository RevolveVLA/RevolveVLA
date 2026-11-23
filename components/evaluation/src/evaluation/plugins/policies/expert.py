"""The training spec's datagen generator as a closed-loop policy (the stack proof): the generator that made the training
episodes (e.g. `moveit`) runs in the policy client container on the sim's domain and drives the episode from the scene
instance, through the same controller interfaces a policy uses. Nothing is replayed: no bag, no checkpoint."""
from __future__ import annotations

import json

from ...core.errors import EvalError
from ...core.interfaces import Policy
from ...core.registry import REGISTRY


@REGISTRY.register("policy", "expert")
class Expert(Policy):
    name = "expert"
    needs_ckpt = False
    client_only = True

    def __init__(self, spec, params):
        if params:
            raise EvalError(f"policy expert: takes no parameters, got {sorted(params)}")
        self.spec = spec

    def client_args(self, item):
        return ["--spec", json.dumps(self.spec.train.raw), "--instance", str(item.scene)]
