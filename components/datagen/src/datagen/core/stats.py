"""What a manifest says, per split, as plain data: progress, stratification cells (sampled vs realised), the distance
to the `min_distance` source split, repeat-check failures. The checks judge these numbers, the report prints them."""
from __future__ import annotations

import math
from typing import Any, Mapping

from . import samplers, separation
from .interfaces import Config
from .spec import DatagenSpec, Range, SplitSpec

EMPTY: Mapping[str, Any] = {"sampled": [], "used": [], "rejected": []}


def rec(data: Mapping, name: str) -> Mapping:
    return data["splits"].get(name, EMPTY)


def seeds(r: Mapping) -> set[int]:
    """Every seed the split touched: pool, accepted, rejected."""
    return {c["seed"] for c in r["sampled"]} | {e["config"]["seed"] for e in r["used"]} | {e["seed"] for e in r["rejected"]}


def progress(split: SplitSpec, r: Mapping) -> dict:
    """target, used, rejected; done = what counts towards the target (the accepted scenes)."""
    return {"target": split.target, "used": len(r["used"]), "rejected": len(r["rejected"]), "done": len(r["used"])}


def distances(spec: DatagenSpec, split: SplitSpec, data: Mapping) -> dict:
    """Closest (distance, seed, source seed) of the split's used configs to the `from` split's used ones, sampled and
    realised (None: no pair); `unreported` = used seeds with no realised start."""
    md, dims = split.min_distance, spec.dims()
    own = md.source == split.name
    mine, theirs = rec(data, split.name)["used"], rec(data, md.source)["used"]
    sampled = [(e["config"]["seed"], e["config"]["values"]) for e in mine]
    sampled_src = [(e["config"]["seed"], e["config"]["values"]) for e in theirs]
    real, real_src = separation.realised_points(mine), separation.realised_points(theirs)
    return {"from": md.source, "value": md.value, "realised_required": md.realised,
            "sampled": separation.closest(sampled, sampled if own else sampled_src, dims, own),
            "realised": separation.closest(real, real if own else real_src, dims, own),
            "unreported": [s for s, r in real if not separation.varied(dims, r)]}


def cells(spec: DatagenSpec, split: SplitSpec, r: Mapping) -> dict | None:
    """The stratification cells of a stratified split with more than one: {bins, rows: [{cell, pool, rejected, used,
    realised, exhausted}]}. A cell is exhausted when it got fewer used than picks and has no unused config left;
    `realised` (None without realised starts) is the cell each used start landed in by its realised values."""
    free = {k: d for k, d in spec.dims().items() if k not in split.sampler.overrides}
    sizes = samplers.cell_sizes(split, free)
    if split.sampler.strategy != "stratified" or math.prod(sizes) <= 1:
        return None
    sampled = [Config.from_dict(c) for c in r["sampled"]]
    used = [Config.from_dict(e["config"]) for e in r["used"]]
    rejected = {e["seed"] for e in r["rejected"]}
    bins = {k: n for (k, d), n in zip(free.items(), sizes) if isinstance(d, Range)}
    real = [e["generate"].get("realised", {}) for e in r["used"]]
    reported = any(separation.varied(free, x) for x in real)
    got = samplers.summary(used, free, bins)["cells"]
    landed = samplers.summary([_landed(c, free, sizes, x) for c, x in zip(used, real)], free, bins)["cells"]
    done = {c.seed for c in used} | rejected
    rows = []
    for cell in samplers.cell_order(sizes):
        pool = [c for c in sampled if c.cell == cell]
        picks = sum(c.cell == cell for c in sampled[:split.target])
        key = ",".join(map(str, cell))
        rows.append({"cell": key, "pool": len(pool), "rejected": sum(c.seed in rejected for c in pool),
                     "used": got.get(key, 0), "realised": landed.get(key, 0) if reported else None,
                     "exhausted": got.get(key, 0) < picks and all(c.seed in done for c in pool)})
    return {"bins": bins, "rows": rows}


def _landed(config: Config, free, sizes, realised: Mapping) -> Config:
    """`config` moved to the cell its realised values fall in (a dim it does not report keeps its sampled cell)."""
    cell = tuple(min(max(int((realised[k] - d.lo) / ((d.hi - d.lo) / n or 1.0)), 0), n - 1)
                 if isinstance(d, Range) and k in realised else c
                 for (k, d), n, c in zip(free.items(), sizes, config.cell))
    return Config(config.split, config.seed, cell, config.values | {k: v for k, v in realised.items() if k in free})


def repeats(spec: DatagenSpec, r: Mapping) -> dict:
    """Seeds tried by the generator and how many were rejected by the N-of-N repeat check (reason `repeat check k/N`)."""
    tried = len(r["used"]) + sum(e["step"] != "validate" for e in r["rejected"])
    flaky = sum("repeat check" in e["reason"] for e in r["rejected"])
    return {"n": spec.generator.params.get("checks", 1), "tried": tried, "flaky": flaky,
            "rate": flaky / tried if tried else 0.0}
