"""DatagenSpec: the datagen section of a task yaml, as frozen dataclasses, loaded and validated here.

Every error names the yaml path of the offending key. Plugin refs ({plugin: name, ...free-form params}) are the only
free-form blocks: their params belong to the plugin. `hash` fingerprints the settings (not `out`, not `run`)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import DatagenError


class SpecError(DatagenError, ValueError):
    pass


@dataclass(frozen=True)
class Range:
    lo: float
    hi: float


@dataclass(frozen=True)
class Choices:
    values: tuple


Dim = Range | Choices


@dataclass(frozen=True)
class PluginRef:
    plugin: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SamplerSpec:
    strategy: str = "random"
    bins: Mapping[str, int] = field(default_factory=dict)        # dotted dim -> bins per dim (stratified)
    overrides: Mapping[str, Any] = field(default_factory=dict)   # dotted dim -> fixed value


@dataclass(frozen=True)
class PoolSpec:
    """Inclusive seed ranges: the initial pool, then the replacement pool (draws that replace rejected ones)."""
    initial: tuple[int, int]
    replacement: tuple[int, int] | None = None

    def ranges(self) -> list[tuple[int, int]]:
        return [self.initial] + ([self.replacement] if self.replacement else [])

    def seeds(self) -> list[int]:
        return [s for lo, hi in self.ranges() for s in range(lo, hi + 1)]


@dataclass(frozen=True)
class MinDistance:
    source: str         # the split (yaml key `from`) the configs must stay `value` away from, in normalised dims
    value: float
    realised: bool = False


@dataclass(frozen=True)
class SplitSpec:
    name: str
    target: int
    sampler: SamplerSpec
    pool: PoolSpec
    min_distance: MinDistance | None = None
    record: bool = True


@dataclass(frozen=True)
class RunSpec:
    """How the run executes, not what it makes (so it is not in the hash): seeds in parallel, headless replays."""
    workers: int = 1
    headless: bool = True


@dataclass(frozen=True)
class DatagenSpec:
    task: str
    robot: str
    out: str
    generator: PluginRef
    splits: tuple[SplitSpec, ...]
    recorder: PluginRef | None = None
    converter: PluginRef | None = None
    randomisers: Mapping[str, Mapping[str, Dim]] = field(default_factory=dict)
    validators: tuple[PluginRef, ...] = ()
    checks: tuple[PluginRef, ...] = ()
    seed: int = 0
    run: RunSpec = RunSpec()
    scene: str | None = None            # the base scene the instances copy (default: the task's scene)

    def dims(self) -> dict[str, Dim]:
        """{`<randomiser>.<dim>`: Dim} of every varied dimension."""
        return {f"{r}.{d}": dim for r, dims in self.randomisers.items() for d, dim in dims.items()}

    def split(self, name: str) -> SplitSpec:
        return next(s for s in self.splits if s.name == name)

    @property
    def hash(self) -> str:
        """Stable across yaml formatting / key order; changes with any setting except `out` and `run`."""
        d = asdict(self)
        d.pop("out"), d.pop("run")
        if d["scene"] is None:          # keeps the hash of a spec without `scene` as it was
            d.pop("scene")
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ parsing
def _fail(at: str, msg: str):
    raise SpecError(f"{at}: {msg}")


def _mapping(v, at, required=(), optional=(), free=False):
    """`v` as a dict with exactly the allowed keys (any keys when `free`)."""
    if not isinstance(v, dict):
        _fail(at, f"expected a mapping, got {type(v).__name__}")
    for k in v:
        if not free and k not in (*required, *optional):
            _fail(f"{at}.{k}", f"unknown key (allowed: {', '.join((*required, *optional))})")
    for k in required:
        if k not in v:
            _fail(at, f"missing key {k!r}")
    return v


def _num(v, at):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        _fail(at, f"expected a number, got {v!r}")
    return v


def _int(v, at, minimum=None):
    if isinstance(v, bool) or not isinstance(v, int):
        _fail(at, f"expected an integer, got {v!r}")
    if minimum is not None and v < minimum:
        _fail(at, f"must be >= {minimum}, got {v}")
    return v


def _str(v, at):
    if not isinstance(v, str) or not v:
        _fail(at, f"expected a non-empty string, got {v!r}")
    return v


def _seed_range(v, at):
    if not isinstance(v, list) or len(v) != 2:
        _fail(at, f"expected [first, last] seeds, got {v!r}")
    lo, hi = _int(v[0], f"{at}[0]", 0), _int(v[1], f"{at}[1]", 0)
    if lo > hi:
        _fail(at, f"first seed {lo} > last seed {hi}")
    return lo, hi


def _plugin(v, at):
    if not isinstance(v, dict) or "plugin" not in v:
        _fail(at, "expected a mapping with a `plugin` key")
    return PluginRef(_str(v["plugin"], f"{at}.plugin"), {k: x for k, x in v.items() if k != "plugin"})


def _plugins(v, at):
    if not isinstance(v, list):
        _fail(at, "expected a list of {plugin: ...}")
    return tuple(_plugin(x, f"{at}[{i}]") for i, x in enumerate(v))


def _dim(v, at) -> Dim:
    if isinstance(v, list):
        if len(v) != 2:
            _fail(at, f"a range is [lo, hi], got {v!r}")
        lo, hi = _num(v[0], f"{at}[0]"), _num(v[1], f"{at}[1]")
        if lo > hi:
            _fail(at, f"lo {lo} > hi {hi}")
        return Range(lo, hi)
    d = _mapping(v, at, required=("choices",))
    if not isinstance(d["choices"], list) or not d["choices"]:
        _fail(f"{at}.choices", "expected a non-empty list")
    return Choices(tuple(d["choices"]))


def contains(dim: Dim, value) -> bool:
    return dim.lo <= value <= dim.hi if isinstance(dim, Range) else value in dim.values


def _sampler(v, at, dims):
    d = _mapping(v, at, optional=("strategy", "bins", "overrides"))
    bins, overrides = d.get("bins", {}), d.get("overrides", {})
    _mapping(bins, f"{at}.bins", free=True)
    _mapping(overrides, f"{at}.overrides", free=True)
    for k, n in bins.items():
        if not isinstance(dims.get(k), Range):
            _fail(f"{at}.bins.{k}", f"not a continuous dim of `randomisers` (known: {sorted(dims)})")
        _int(n, f"{at}.bins.{k}", 1)
    for k, x in overrides.items():
        if k not in dims:
            _fail(f"{at}.overrides.{k}", f"not a dim of `randomisers` (known: {sorted(dims)})")
        if not contains(dims[k], x):
            _fail(f"{at}.overrides.{k}", f"{x!r} is outside {dims[k]}")
    return SamplerSpec(_str(d.get("strategy", "random"), f"{at}.strategy"), dict(bins), dict(overrides))


def _split(name, v, at, dims):
    d = _mapping(v, at, required=("target", "pool"),
                 optional=("sampler", "min_distance", "record"))
    target = _int(d["target"], f"{at}.target", 1)
    p = _mapping(d["pool"], f"{at}.pool", required=("initial",), optional=("replacement",))
    pool = PoolSpec(_seed_range(p["initial"], f"{at}.pool.initial"),
                    _seed_range(p["replacement"], f"{at}.pool.replacement") if "replacement" in p else None)
    if len(pool.seeds()) < target:
        _fail(f"{at}.pool", f"{len(pool.seeds())} seeds cannot reach target {target}")
    md = None
    if "min_distance" in d:
        m = _mapping(d["min_distance"], f"{at}.min_distance", required=("from", "value"), optional=("realised",))
        value = _num(m["value"], f"{at}.min_distance.value")
        if value <= 0:
            _fail(f"{at}.min_distance.value", f"must be > 0, got {value}")
        realised = m.get("realised", False)
        if not isinstance(realised, bool):
            _fail(f"{at}.min_distance.realised", f"expected true/false, got {realised!r}")
        md = MinDistance(_str(m["from"], f"{at}.min_distance.from"), value, realised)
    record = d.get("record", True)
    if not isinstance(record, bool):
        _fail(f"{at}.record", f"expected true/false, got {record!r}")
    return SplitSpec(name, target, _sampler(d.get("sampler", {}), f"{at}.sampler", dims), pool, md, record)


def _run(v, at) -> RunSpec:
    d = _mapping(v, at, optional=("workers", "headless"))
    headless = d.get("headless", True)
    if not isinstance(headless, bool):
        _fail(f"{at}.headless", f"expected true/false, got {headless!r}")
    return RunSpec(_int(d.get("workers", 1), f"{at}.workers", 1), headless)


def _check_pools_and_sources(splits, at):
    taken = []     # (lo, hi, split)
    for s in splits:
        for lo, hi in s.pool.ranges():
            for olo, ohi, other in taken:
                if lo <= ohi and olo <= hi:
                    _fail(f"{at}.splits.{s.name}.pool", f"seeds [{lo}, {hi}] overlap [{olo}, {ohi}] of split {other!r}")
            taken.append((lo, hi, s.name))
    earlier = []
    for s in splits:
        earlier.append(s.name)
        if s.min_distance and s.min_distance.source not in earlier:
            _fail(f"{at}.splits.{s.name}.min_distance.from",
                  f"{s.min_distance.source!r} is not this split or an earlier one (splits: {[x.name for x in splits]})")


def parse(raw: Any, source: str = "<spec>") -> DatagenSpec:
    d = _mapping(raw, source, required=("task", "robot", "out", "generator", "splits"),
                 optional=("recorder", "converter", "randomisers", "validators", "checks", "seed", "run", "scene"))
    rand = {}
    for r, dims in _mapping(d.get("randomisers", {}), f"{source}.randomisers", free=True).items():
        at = f"{source}.randomisers.{r}"
        rand[r] = {k: _dim(x, f"{at}.{k}") for k, x in _mapping(dims, at, free=True).items()}
    spec_dims = {f"{r}.{k}": x for r, dims in rand.items() for k, x in dims.items()}
    raw_splits = _mapping(d["splits"], f"{source}.splits", free=True)
    if not raw_splits:
        _fail(f"{source}.splits", "needs at least one split")
    splits = tuple(_split(n, v, f"{source}.splits.{n}", spec_dims) for n, v in raw_splits.items())
    _check_pools_and_sources(splits, source)
    return DatagenSpec(
        task=_str(d["task"], f"{source}.task"), robot=_str(d["robot"], f"{source}.robot"),
        out=_str(d["out"], f"{source}.out"), generator=_plugin(d["generator"], f"{source}.generator"),
        splits=splits,
        recorder=_plugin(d["recorder"], f"{source}.recorder") if "recorder" in d else None,
        converter=_plugin(d["converter"], f"{source}.converter") if "converter" in d else None,
        randomisers=rand, validators=_plugins(d.get("validators", []), f"{source}.validators"),
        checks=_plugins(d.get("checks", []), f"{source}.checks"), seed=_int(d.get("seed", 0), f"{source}.seed"),
        run=_run(d.get("run", {}), f"{source}.run"),
        scene=_str(d["scene"], f"{source}.scene") if "scene" in d else None)


def load(path: str | Path) -> DatagenSpec:
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text())
    except (OSError, yaml.YAMLError) as e:
        raise SpecError(f"{p}: cannot read: {e}") from e
    return parse(raw, str(p))


def check_limits(spec: DatagenSpec, limits: Mapping[str, Dim]):
    """The spec's dims must lie within the randomisers' hard limits ({`<randomiser>.<dim>`: Dim})."""
    for key, dim in spec.dims().items():
        r, _, name = key.partition(".")
        at = f"randomisers.{r}.{name}"
        if key not in limits:
            _fail(at, f"unknown dim (the plugin has: {sorted(k for k in limits if k.startswith(r + '.'))})")
        lim = limits[key]
        if isinstance(dim, Range):
            ok = isinstance(lim, Range) and lim.lo <= dim.lo and dim.hi <= lim.hi
        else:
            ok = all(contains(lim, v) for v in dim.values)
        if not ok:
            _fail(at, f"{dim} is outside the hard limits {lim}")
