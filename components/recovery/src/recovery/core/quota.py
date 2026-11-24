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


def select(failures: Sequence[Failure], quota: Mapping[str, int], per_failure: int, seed: int) -> list[Failure]:
    """The failures to recover, in run order."""
    chosen = {}
    for phase, episodes in quota.items():
        mine = [f for f in failures if f.phase == phase and f.restart.reproducible]
        chosen[phase] = spread(mine, f"{seed}:{phase}")[:math.ceil(episodes / per_failure)]
    return [f for row in zip_longest(*chosen.values()) for f in row if f is not None]
