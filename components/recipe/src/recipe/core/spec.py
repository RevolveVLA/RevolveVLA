"""RecipeSpec: the recipe yaml as frozen dataclasses, validated here (datagen's spec helpers: every error names the yaml path of the
offending key). `sources` are plugin refs named by the recipe; `take` says how much of each (`all` | n | {phase: n}); `splits` are
fractions grouped by scene. `hash` fingerprints the settings (not `out`)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml
from datagen.core.spec import PluginRef, SpecError, _fail, _int, _mapping, _num, _plugin, _str

UNITS = ("episodes", "frames")
ON_SHORT = ("warn", "error")
ALL = "all"
ANY_PHASE = None            # the key of a `take` that is not per phase


@dataclass(frozen=True)
class RecipeSpec:
    sources: Mapping[str, PluginRef]
    take: Mapping[str, Mapping[str | None, int | str]]     # source -> {phase | None: n | "all"}
    out: str
    seed: int = 0
    unit: str = "episodes"
    fps: int = 30                                       # frames per second of an episode (unit: frames)
    splits: Mapping[str, float] = field(default_factory=lambda: {"train": 1.0})
    exclude: tuple[str, ...] = ()
    on_short: str = "warn"
    output: PluginRef = PluginRef("selection")
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def hash(self) -> str:
        """Stable across yaml formatting / key order; changes with any setting except `out`."""
        d = asdict(self)
        d.pop("out"), d.pop("raw")
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _amount(v, at):
    return ALL if v == ALL else _int(v, at, 0)


def _take(v, at) -> dict:
    if v == ALL or isinstance(v, int) and not isinstance(v, bool):
        return {ANY_PHASE: _amount(v, at)}
    per_phase = _mapping(v, at, free=True)
    if not per_phase:
        _fail(at, "expected all, a number, or {phase: number}")
    return {_str(p, f"{at}.{p}"): _amount(n, f"{at}.{p}") for p, n in per_phase.items()}


def _splits(v, at) -> dict[str, float]:
    s = _mapping(v, at, free=True)
    if not s:
        _fail(at, "needs at least one split")
    out = {name: float(_num(f, f"{at}.{name}")) for name, f in s.items()}
    for name, f in out.items():
        if f <= 0:
            _fail(f"{at}.{name}", f"must be > 0, got {f}")
    if abs(sum(out.values()) - 1) > 1e-6:
        _fail(at, f"the fractions must sum to 1, got {sum(out.values())}")
    return out


def parse(raw: Any, source: str = "<spec>") -> RecipeSpec:
    d = _mapping(raw, source, required=("sources", "take", "out"),
                 optional=("seed", "unit", "fps", "splits", "exclude", "on_short", "output"))
    sources = {n: _plugin(v, f"{source}.sources.{n}") for n, v in _mapping(d["sources"], f"{source}.sources", free=True).items()}
    if not sources:
        _fail(f"{source}.sources", "needs at least one source")
    take = {n: _take(v, f"{source}.take.{n}") for n, v in _mapping(d["take"], f"{source}.take", free=True).items()}
    for n in take:
        if n not in sources:
            _fail(f"{source}.take.{n}", f"not a source (sources: {sorted(sources)})")
    for n in sources:
        if n not in take:
            _fail(f"{source}.take", f"no amount for source {n!r} (use 0 to leave it out)")
    unit, on_short = d.get("unit", "episodes"), d.get("on_short", "warn")
    if unit not in UNITS:
        _fail(f"{source}.unit", f"expected one of {UNITS}, got {unit!r}")
    if on_short not in ON_SHORT:
        _fail(f"{source}.on_short", f"expected one of {ON_SHORT}, got {on_short!r}")
    exclude = d.get("exclude", [])
    if not isinstance(exclude, list):
        _fail(f"{source}.exclude", "expected a list of directories")
    return RecipeSpec(
        sources=sources, take=take, out=_str(d["out"], f"{source}.out"), seed=_int(d.get("seed", 0), f"{source}.seed"),
        unit=unit, fps=_int(d.get("fps", 30), f"{source}.fps", 1), splits=_splits(d.get("splits", {"train": 1.0}), f"{source}.splits"),
        exclude=tuple(_str(p, f"{source}.exclude[{i}]") for i, p in enumerate(exclude)), on_short=on_short,
        output=_plugin(d["output"], f"{source}.output") if "output" in d else PluginRef("selection"), raw=raw)


def load(path: str | Path) -> RecipeSpec:
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text())
    except (OSError, yaml.YAMLError) as e:
        raise SpecError(f"{p}: cannot read: {e}") from e
    return parse(raw, str(p))
