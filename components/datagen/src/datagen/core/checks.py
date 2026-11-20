"""Whole-run checks over the manifest, registered by name. A spec lists them (`checks: [{plugin: coverage}, ...]`);
without a list `default_checks` selects every built-in one that applies to the spec. Each reads numbers from `stats`
and returns one verdict line; a check never changes the manifest."""
from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping

from . import stats
from .separation import TOLERANCE
from .interfaces import Check, Result
from .registry import register
from .spec import DatagenSpec, PluginRef, SplitSpec

BUILTIN: dict[str, type[Check]] = {}        # registration order = run / report order


def _builtin(name: str):
    def deco(cls):
        cls.name = name
        BUILTIN[name] = register("check", name)(cls)
        return cls
    return deco


def default_checks(spec: DatagenSpec) -> tuple[PluginRef, ...]:
    return tuple(PluginRef(n) for n, c in BUILTIN.items() if c.applies(spec))


class _Check(Check):
    def __init__(self, spec: DatagenSpec, params: Mapping[str, Any] = {}):
        self.spec, self.params = spec, params

    @classmethod
    def applies(cls, spec: DatagenSpec) -> bool:
        return True


@_builtin("target")
class TargetReached(_Check):
    """Every split has its target number of accepted (and, when it records, recorded) episodes."""

    def run(self, manifest):
        short = []
        for s in self.spec.splits:
            p = stats.progress(s, stats.rec(manifest, s.name))
            if p["done"] < s.target:
                short.append(f"{s.name} {p['done']}/{s.target}")
        return Result(not short, ", ".join(short))


@_builtin("split_disjoint")
class SplitsDisjoint(_Check):
    """No seed (pool, accepted or rejected) belongs to two splits."""

    @classmethod
    def applies(cls, spec):
        return len(spec.splits) > 1

    def run(self, manifest):
        touched = {s.name: stats.seeds(stats.rec(manifest, s.name)) for s in self.spec.splits}
        shared = [f"{a}/{b} {sorted(touched[a] & touched[b])}" for a, b in combinations(touched, 2)
                  if touched[a] & touched[b]]
        return Result(not shared, "seeds in two splits: " + ", ".join(shared) if shared else "")


class _MinDistance(_Check):
    kind = ""           # "sampled" | "realised": which closest pair is judged

    @classmethod
    def splits(cls, spec) -> list[SplitSpec]:
        return [s for s in spec.splits if s.min_distance and (s.min_distance.realised or cls.kind == "sampled")]

    @classmethod
    def applies(cls, spec):
        return bool(cls.splits(spec))

    def run(self, manifest):
        bad = []
        for s in self.splits(self.spec):
            d = stats.distances(self.spec, s, manifest)
            pair = d[self.kind]
            if self.kind == "realised" and d["unreported"]:
                bad.append(f"{s.name}: no realised start for seeds {d['unreported']}")
            elif pair and pair[0] < d["value"] - TOLERANCE:
                bad.append(f"{s.name} seed {pair[1]} - {d['from']} seed {pair[2]}: {pair[0]:.3f} < {d['value']}")
        return Result(not bad, "; ".join(bad))


@_builtin("min_distance")
class SampledMinDistance(_MinDistance):
    """The used configs keep their split's `min_distance` (sampled values) from the `from` split's used ones."""
    kind = "sampled"


@_builtin("realised_min_distance")
class RealisedMinDistance(_MinDistance):
    """Same on the realised starts (what the generator ended up with), for splits with `min_distance.realised`."""
    kind = "realised"


@_builtin("coverage")
class CellCoverage(_Check):
    """Every stratification cell is filled (no exhausted cell; no empty cell while target >= cells), by the sampled
    configs and by the realised starts."""

    @classmethod
    def applies(cls, spec):
        return any(s.sampler.strategy == "stratified" for s in spec.splits)

    def run(self, manifest):
        bad = []
        for s in self.spec.splits:
            c = stats.cells(self.spec, s, stats.rec(manifest, s.name))
            if c is None:
                continue
            full = s.target >= len(c["rows"])
            for row in c["rows"]:
                if row["exhausted"]:
                    bad.append(f"{s.name} cell {row['cell']} exhausted")
                elif full and row["used"] == 0:
                    bad.append(f"{s.name} cell {row['cell']} empty")
                elif full and row["realised"] == 0:
                    bad.append(f"{s.name} cell {row['cell']} empty when realised")
        return Result(not bad, ", ".join(bad))


@_builtin("repeat_stats")
class RepeatStats(_Check):
    """The N-of-N repeat replays (`generator.checks` > 1) are not flaky: at most `max_rate` (default 0.5) of the seeds
    the generator tried were rejected by a repeat check."""

    @classmethod
    def applies(cls, spec):
        return spec.generator.params.get("checks", 1) > 1

    def run(self, manifest):
        limit = self.params.get("max_rate", 0.5)
        bad = []
        for s in self.spec.splits:
            r = stats.repeats(self.spec, stats.rec(manifest, s.name))
            if r["rate"] > limit:
                bad.append(f"{s.name} {r['flaky']}/{r['tried']} seeds failed a repeat check (> {limit})")
        return Result(not bad, ", ".join(bad))
