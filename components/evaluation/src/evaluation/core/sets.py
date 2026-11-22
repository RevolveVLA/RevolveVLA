"""What the eval set plugins share: the training run they are measured against and the in-memory DatagenSpec that lets
datagen's sampler, validators, instancer and checks work on an eval set (zero datagen changes).

The training run = the training spec + its datagen state (+ manifest.json episodes), read by the `training` plugin
(`spec.training`). Without a state only the training splits' sampled configs are known: they are re-sampled
(deterministic in seed, split and pool) and the first `target` of a split stand in for its used configs ("sampled-only":
no scene instances)."""
from __future__ import annotations

import dataclasses
from itertools import zip_longest
from pathlib import Path
from typing import Mapping, Sequence

from datagen.core import registry, stats
from datagen.core.interfaces import Config
from datagen.core.spec import DatagenSpec, Dim, SplitSpec

from .errors import EvalError
from .registry import REGISTRY
from .spec import EvalSpec, SetSpec


def repo_root(spec: EvalSpec, out: Path) -> Path:
    """What the relative paths of the spec (`out`, `train_manifest`) are relative to: the CLI's root, recovered from the
    resolved `out`."""
    p = Path(spec.out)
    return Path.cwd() if p.is_absolute() else out.parents[len(p.parts) - 1]


class Training:
    def __init__(self, spec: DatagenSpec, path: Path, data: Mapping | None = None, episodes: Mapping | None = None):
        """data: the training state (None when it does not exist: sampled-only); episodes: its recorded ones by seed."""
        self.spec, self.path, self.data, self._sampled = spec, path, data, {}
        self.episodes = episodes or {}

    @classmethod
    def load(cls, spec: EvalSpec, root: Path) -> Training:
        run = REGISTRY.create("training", spec.training.plugin, spec, spec.training.params).load(root)
        return cls(spec.train, run.path, run.data, run.episodes)

    @property
    def sampled_only(self) -> bool:
        return self.data is None

    def need_manifest(self, why: str):
        if self.sampled_only:
            raise EvalError(f"{why} needs the training state {self.path}: run datagen for {self.spec.out} first")

    @property
    def base(self) -> Path:
        """What the scene paths of the state are relative to: the training run's `out`."""
        return self.path.parent

    def rec(self, name: str) -> Mapping:
        """The state block of training split `name` ({sampled, used: [{config, generate}], rejected})."""
        if self.data is not None:
            return stats.rec(self.data, name)
        configs = self._resample(name)
        used = [{"config": c.to_dict(), "generate": {"ok": True}} for c in configs[:self.spec.split(name).target]]
        return {"sampled": [c.to_dict() for c in configs], "used": used, "rejected": []}

    def _resample(self, name: str) -> list[Config]:
        if name not in self._sampled:
            s, md = self.spec.split(name), self.spec.split(name).min_distance
            avoid = self._resample(md.source) if md and md.source != name else []
            self._sampled[name] = registry.create("sampler", s.sampler.strategy).sample(
                s, self.spec.dims(), s.pool.seeds(), self.spec.seed, avoid=avoid)
        return self._sampled[name]

    def configs(self, name: str) -> list[Config]:
        return [Config.from_dict(e["config"]) for e in self.rec(name)["used"]]

    def touched(self) -> set[int]:
        """Every seed the splits that were trained on (`record: true`) own or touched."""
        seeds = set()
        for s in (s for s in self.spec.splits if s.record):
            seeds |= set(s.pool.seeds())
            if self.data is not None:
                seeds |= stats.seeds(stats.rec(self.data, s.name))
        return seeds

    def usable(self, name: str) -> list[Mapping]:
        """The used entries of split `name` that produced a scene (an accepted one: datagen v2 has no record step)."""
        return [e for e in self.rec(name)["used"] if e["generate"]["ok"]]


def derive(spec: EvalSpec, s: SetSpec) -> DatagenSpec:
    """The training spec (task, robot, randomisers = the training dims, validators, seed) with the set as a
    `record: false` split beside the training split it keeps `min_distance` from: what the instancer, the validators
    and datagen's checks take. A set's `min_distance.realised` is dropped: the instance IS the realised start, so the
    sampled distance is the realised one."""
    md = s.min_distance
    if md and md.source == s.name:
        raise EvalError(f"sets.{s.name}: the set has the name of the training split it measures against: rename it")
    source = [dataclasses.replace(spec.train.split(md.source), min_distance=None)] if md else []
    mine = SplitSpec(s.name, s.count, s.sampler, s.pool, md and dataclasses.replace(md, realised=False), record=False)
    return dataclasses.replace(spec.train, splits=(*source, mine), converter=None, checks=())


def sample_dims(spec: EvalSpec, s: SetSpec) -> dict[str, Dim]:
    """The dims a set draws from: the training ranges, narrowed where the set has `randomisers`."""
    dims = spec.train.dims()
    dims.update({f"{r}.{k}": d for r, sub in s.randomisers.items() for k, d in sub.items()})
    return dims


def spread(entries: Sequence[Mapping], count: int) -> list[Mapping]:
    """The first `count` entries taken round-robin over their stratification cells (each cell in manifest order)."""
    cells: dict[tuple, list] = {}
    for e in entries:
        cells.setdefault(tuple(e["config"]["cell"]), []).append(e)
    return [e for row in zip_longest(*cells.values()) for e in row if e is not None][:count]
