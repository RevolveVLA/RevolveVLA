"""Sampling strategies, registered by name. `random`: independent uniform draws per seed (PRD 02 adds stratified,
sobol, lhs and the shared min_distance / redraw handling)."""
from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from .interfaces import Config, Sampler
from .registry import register
from .spec import Choices, Dim, SplitSpec


def draw(rng: random.Random, dim: Dim) -> Any:
    return rng.choice(dim.values) if isinstance(dim, Choices) else rng.uniform(dim.lo, dim.hi)


@register("sampler", "random")
class RandomSampler(Sampler):
    def sample(self, split: SplitSpec, dims: Mapping[str, Dim], seeds: Sequence[int], base_seed: int) -> list[Config]:
        out = []
        for seed in seeds:
            rng = random.Random(f"{base_seed}:{split.name}:{seed}")     # str seeds are hashed deterministically
            values = {k: draw(rng, d) for k, d in dims.items()} | dict(split.sampler.overrides)
            out.append(Config(split.name, seed, (), values))
        return out
