"""policy noisy_expert: the oracle's replay with Gaussian noise on every action (arm references in rad, base twist in m/s and
rad/s), seeded by the config's seed: a policy that drifts off the expert's path and fails. Params: `sigma` (default 0.05)."""
from __future__ import annotations

import numpy as np
from evaluation.core.errors import EvalError
from evaluation.core.registry import REGISTRY
from evaluation.plugins.policies.oracle import Oracle


def perturb(series, params):
    rng = np.random.default_rng(params["seed"])
    sigma = params["sigma"]
    return type(series)(series.joints, {c: p + rng.normal(0.0, sigma, p.shape) for c, p in series.positions.items()},
                        series.base + rng.normal(0.0, sigma, series.base.shape))


@REGISTRY.register("policy", "noisy_expert")
class NoisyExpert(Oracle):
    name = "noisy_expert"

    def __init__(self, spec, params):
        if unknown := sorted(set(params) - {"sigma"}):
            raise EvalError(f"policy noisy_expert: unknown parameters {unknown} (known: ['sigma'])")
        self.sigma = float(params.get("sigma", 0.05))
        if self.sigma <= 0:
            raise EvalError(f"policy noisy_expert: sigma must be > 0, got {self.sigma}")

    def perturbation(self, item):
        return {"name": "noisy_expert", "sigma": self.sigma, "seed": item.config.seed}
