"""The checks of an eval set, as datagen `Check`s over a manifest-shaped view `{"splits": {<training split>: block,
<set>: block}}` (stats.rec). Distances are datagen's own (`SampledMinDistance`, `RealisedMinDistance`, measured in the
TRAINING dims); `InRange` and `SeedsDisjoint` are the eval-specific ones. Each returns a datagen `Result`."""
from __future__ import annotations

from typing import Mapping

from datagen.core import stats
from datagen.core.checks import RealisedMinDistance, SampledMinDistance
from datagen.core.interfaces import Check, Result
from datagen.core.spec import DatagenSpec, Dim, contains

from .sets import Training
from .spec import EvalSpec


class InRange(Check):
    """Every value of every used config lies inside the TRAINING range of its dim."""
    name = "in_range"

    def __init__(self, split: str, dims: Mapping[str, Dim]):
        self.split, self.dims = split, dims

    def run(self, manifest):
        bad = [f"seed {e['config']['seed']} {k}={v}" for e in stats.rec(manifest, self.split)["used"]
               for k, v in e["config"]["values"].items() if k in self.dims and not contains(self.dims[k], v)]
        return Result(not bad, "outside the training range: " + ", ".join(bad) if bad else "")


class SeedsDisjoint(Check):
    """No seed the set touched (pool, accepted, rejected) is one of `taken` ({owner: seeds}: the trained splits, the
    other eval sets)."""
    name = "seeds_disjoint"

    def __init__(self, split: str, taken: Mapping[str, set[int]]):
        self.split, self.taken = split, taken

    def run(self, manifest):
        mine = stats.seeds(stats.rec(manifest, self.split))
        shared = [f"{who} {sorted(mine & seeds)}" for who, seeds in self.taken.items() if mine & seeds]
        return Result(not shared, "seeds also used by " + ", ".join(shared) if shared else "")


def set_checks(spec: EvalSpec, derived: DatagenSpec, name: str, training: Training) -> list[Check]:
    """The checks of held-out set `name` (`derived` = sets.derive): range, seeds, and the distances it asks for."""
    taken = {"training": training.touched()}
    taken |= {f"set {n!r}": set(s.pool.seeds()) for n, s in spec.sets.items() if n != name and s.pool}
    checks = [InRange(name, derived.dims()), SeedsDisjoint(name, taken)]
    return checks + [c(derived, {}) for c in (SampledMinDistance, RealisedMinDistance) if c.applies(derived)]


def run_checks(checks, view: Mapping, training: Training) -> dict[str, Result]:
    """{check name: Result}, plus a note when the training configs are only the re-sampled ones."""
    out = {c.name: c.run(view) for c in checks}
    if training.sampled_only:
        out["training_configs"] = Result(True, f"sampled-only: no manifest {training.path}, distances are measured "
                                               "to the re-sampled training configs (no realised starts)")
    return out
