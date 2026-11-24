"""Which failures get a recovery job: per phase, the reproducible failures spread over the eval configs (round robin over
configs, each config's failures in run order, the configs in a seeded order), enough of them for the phase's quota of
episodes (`per_failure` each); the phases are then interleaved, so a stopped run has a balanced mixture. A function of the
failures, the quota and the seed only."""
from __future__ import annotations

import math
import random
from itertools import zip_longest
from typing import Mapping, Sequence

from .failures import Failure


def config_of(failure: Failure) -> str:
    """The eval config of a failure: its set and scene instance (the rollouts of one config share it)."""
    return f"{failure.set}:{failure.scene}"


def spread(failures: Sequence[Failure], seed: str) -> list[Failure]:
    """`failures` round robin over their configs (the config order shuffled by `seed`)."""
    by_config: dict[str, list[Failure]] = {}
    for f in failures:
        by_config.setdefault(config_of(f), []).append(f)
    configs = sorted(by_config)
    random.Random(seed).shuffle(configs)
    return [f for row in zip_longest(*(by_config[c] for c in configs)) for f in row if f is not None]


def limit(quota: Mapping[str, int], phase: str, per_failure: int) -> int:
    """The jobs a phase's quota of episodes needs."""
    return math.ceil(quota[phase] / per_failure)


class Admission:
    """`select` for failures that arrive one batch at a time (a coupled evaluation): each reproducible failure of a phase with jobs
    left is picked, in arrival order (the evaluation's), not spread over its configs. `taken`: the phases of the jobs picked so far."""

    def __init__(self, quota: Mapping[str, int], per_failure: int, taken: Sequence[str] = ()):
        self.left = {p: limit(quota, p, per_failure) - sum(t == p for t in taken) for p in quota}

    def pick(self, failures: Sequence[Failure]) -> list[Failure]:
        chosen = []
        for f in failures:
            if f.restart.reproducible and self.left.get(f.phase, 0) > 0:
                self.left[f.phase] -= 1
                chosen.append(f)
        return chosen


def select(failures: Sequence[Failure], quota: Mapping[str, int], per_failure: int, seed: int) -> list[Failure]:
    """The failures to recover, in run order."""
    chosen = {}
    for phase in quota:
        mine = [f for f in failures if f.phase == phase and f.restart.reproducible]
        chosen[phase] = spread(mine, f"{seed}:{phase}")[:limit(quota, phase, per_failure)]
    return [f for row in zip_longest(*chosen.values()) for f in row if f is not None]
