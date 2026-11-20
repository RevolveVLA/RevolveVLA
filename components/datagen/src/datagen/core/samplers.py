"""Sampling strategies, registered by name: random, stratified, sobol, lhs.

Every strategy proposes, per seed, a point in the unit cube of the varied dims (+ a cell for stratified); the shared
`sample` maps it to values, applies the split's overrides (fixed dims, never drawn) and redraws a refused draw
(reject callback, min_distance) from the seed's own stream, up to MAX_TRIES. The per-seed stream is seeded by
(base seed, split, seed), so random and stratified draws do not depend on the other seeds of the pool; sobol and lhs
points are indexed by the seed's position in the pool (the sequence itself is seeded by (base seed, split)).
numpy / scipy are not imported here except lazily inside sobol / lhs, so the core stays import-light."""
from __future__ import annotations

import math
import random
from itertools import product
from typing import Any, Mapping, Sequence

from .errors import DatagenError
from .interfaces import Config, Sampler
from .registry import register
from .spec import Choices, Dim, Range, SplitSpec

MAX_TRIES = 100


class SamplingError(DatagenError):
    pass


def value(dim: Dim, u: float) -> Any:
    """The value at unit coordinate `u` in [0, 1): uniform in a range, equal shares per choice."""
    if isinstance(dim, Choices):
        return dim.values[min(int(u * len(dim.values)), len(dim.values) - 1)]
    return dim.lo + u * (dim.hi - dim.lo)


def distance(a: Mapping[str, Any], b: Mapping[str, Any], dims: Mapping[str, Dim]) -> float:
    """Euclidean over the continuous dims, each normalised by its range (choices and zero-width dims ignored)."""
    return math.sqrt(sum(((a[k] - b[k]) / (d.hi - d.lo)) ** 2 for k, d in dims.items()
                         if isinstance(d, Range) and d.hi > d.lo and k in a and k in b))


def cell_order(sizes: Sequence[int]) -> list[tuple[int, ...]]:
    """Every cell (product of `sizes` options per dim), greedily ordered so every prefix uses each dim's options as
    evenly as possible."""
    left, out = list(product(*[range(n) for n in sizes])), []
    counts = [[0] * n for n in sizes]
    while left:
        best = min(left, key=lambda c: (sum(counts[j][o] for j, o in enumerate(c)), c))
        left.remove(best)
        out.append(best)
        for j, o in enumerate(best):
            counts[j][o] += 1
    return out


def cell_sizes(split: SplitSpec, free: Mapping[str, Dim]) -> list[int]:
    """Cells per free dim of a stratified split: its `bins` for a continuous dim (default 1), one per choice."""
    return [len(d.values) if isinstance(d, Choices) else split.sampler.bins.get(k, 1) for k, d in free.items()]


class _Sampler(Sampler):
    def __init__(self):
        self.redrawn = []

    def sample(self, split, dims, seeds, base_seed, reject=None, avoid=()):
        self.redrawn = []
        fixed = dict(split.sampler.overrides)
        free = {k: d for k, d in dims.items() if k not in fixed}
        rngs = [random.Random(f"{base_seed}:{split.name}:{s}") for s in seeds]
        md = split.min_distance
        own = md is not None and md.source == split.name
        out: list[Config] = []
        for seed, rng, (cell, u) in zip(seeds, rngs, self._starts(split, free, rngs, base_seed)):
            for _ in range(MAX_TRIES):
                config = Config(split.name, seed, cell, {k: value(d, x) for (k, d), x in zip(free.items(), u)} | fixed)
                reason = None
                if md is not None and any(distance(config.values, o.values, dims) < md.value
                                          for o in (out if own else avoid)):
                    reason = "min_distance"
                elif reject is not None:
                    reason = reject(config)
                if reason is None:
                    break
                self.redrawn.append({"seed": seed, "cell": list(cell), "values": dict(config.values),
                                     "reason": str(reason)})
                u = self._redraw(rng, cell, free, split)
            else:
                raise SamplingError(f"{split.name}: seed {seed} (cell {list(cell)}): all {MAX_TRIES} draws were "
                                    f"rejected, last: {reason}")
            out.append(config)
        return out

    def _starts(self, split: SplitSpec, free: Mapping[str, Dim], rngs: Sequence[random.Random], base_seed: int):
        """[(cell, unit point)] per seed: the first draw."""
        return [((), self._redraw(rng, (), free, split)) for rng in rngs]

    def _redraw(self, rng: random.Random, cell: tuple, free: Mapping[str, Dim], split: SplitSpec) -> list[float]:
        return [rng.random() for _ in free]


@register("sampler", "random")
class RandomSampler(_Sampler):
    """Independent uniform draw per seed."""


@register("sampler", "stratified")
class StratifiedSampler(_Sampler):
    """Cells = `bins` per continuous dim (default 1) x every choice of a choices dim. The pool's seeds take the cells
    in `cell_order`, each uniform in its cell (a choices dim is its cell's choice), so every prefix is balanced."""

    def _starts(self, split, free, rngs, base_seed):
        order = cell_order(cell_sizes(split, free))
        cells = [order[i % len(order)] for i in range(len(rngs))]
        return [(cell, self._redraw(rng, cell, free, split)) for cell, rng in zip(cells, rngs)]

    def _redraw(self, rng, cell, free, split):
        sizes = cell_sizes(split, free)
        return [(c + (rng.random() if isinstance(d, Range) else 0.5)) / n for c, n, d in zip(cell, sizes, free.values())]


class _QmcSampler(_Sampler):
    """Low-discrepancy points over the whole pool, one per seed in pool order; redraws are uniform."""

    def _starts(self, split, free, rngs, base_seed):
        if not free:
            return [((), [])] * len(rngs)
        rng = random.Random(f"{base_seed}:{split.name}:{type(self).__name__}")
        return [((), list(map(float, p))) for p in self._points(split, len(free), len(rngs), rng.getrandbits(32))]

    def _points(self, split: SplitSpec, d: int, n: int, seed: int):
        raise NotImplementedError


@register("sampler", "sobol")
class SobolSampler(_QmcSampler):
    """Scrambled Sobol sequence (scipy.stats.qmc): every prefix of the pool is well spread."""

    def _points(self, split, d, n, seed):
        from scipy.stats import qmc
        return qmc.Sobol(d, scramble=True, seed=seed).random_base2(max(0, math.ceil(math.log2(n))))[:n]


@register("sampler", "lhs")
class LatinHypercubeSampler(_QmcSampler):
    """Latin hypercube (scipy.stats.qmc): one design over the first `target` seeds (the picks), one over the rest
    (the replacements), so the used set is a hypercube while no replacement is needed."""

    def _points(self, split, d, n, seed):
        import numpy as np
        from scipy.stats import qmc
        sizes = [m for m in (min(split.target, n), n - split.target) if m > 0]
        return np.vstack([qmc.LatinHypercube(d, seed=seed + i).random(m) for i, m in enumerate(sizes)])


def summary(configs: Sequence[Config], dims: Mapping[str, Dim], bins: Mapping[str, int] = {},
            others: Sequence[Config] = ()) -> dict:
    """{count, cells: {cell: n}, dims: {k: {min, max, bin_counts | counts}}, min_distance (own pairs), min_distance_to
    (nearest of `others`)}; the two distances are None without a pair."""
    out: dict[str, Any] = {"count": len(configs), "cells": {}, "dims": {}}
    for c in configs:
        key = ",".join(map(str, c.cell))
        out["cells"][key] = out["cells"].get(key, 0) + 1
    for k, d in dims.items():
        vals = [c.values[k] for c in configs]
        if isinstance(d, Choices):
            out["dims"][k] = {"counts": {v: vals.count(v) for v in d.values}}
            continue
        out["dims"][k] = {"min": min(vals, default=None), "max": max(vals, default=None)}
        if n := bins.get(k):
            w = (d.hi - d.lo) / n or 1.0
            idx = [min(int((v - d.lo) / w), n - 1) for v in vals]
            out["dims"][k]["bin_counts"] = [idx.count(i) for i in range(n)]
    def nearest(a, b):
        return min((distance(x.values, y.values, dims) for x in a for y in b if x is not y), default=None)
    out["min_distance"], out["min_distance_to"] = nearest(configs, configs), nearest(configs, others)
    return out
