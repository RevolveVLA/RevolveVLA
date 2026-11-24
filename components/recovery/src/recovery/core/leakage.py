"""The leakage guard of a recorder: the scenes it trains on must not be scenes the policy is scored on. Of the eval spec's sets,
`run` are mined (live: rolled out; phases: their failures are recovered), `reserved` are kept unseen (default: its held-out sets
not in `run`; plus, always, the pools the training spec keeps aside: its splits with `record: false`). The seeds (pool ranges) of the
two must not overlap. The same guard serves the live recorder (from the eval spec) and phase recovery (from the evaluation run's
results.json, which keeps the sets' kinds and pools: `Sets`)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from datagen.core.spec import SpecError


@dataclass(frozen=True)
class Sets:
    """What the guard needs of an evaluation: per set its kind and seed ranges, and the pools kept aside (label, ranges)."""
    kinds: Mapping[str, str]
    pools: Mapping[str, list]
    aside: Sequence = ()


def of_spec(eval_spec) -> Sets:
    d = eval_spec.leakage()
    return Sets({n: s["kind"] for n, s in d["sets"].items()}, {n: s["pool"] for n, s in d["sets"].items()}, d["aside"])


def of_run(run: Path) -> Sets:
    """The sets of a finished (or running) evaluation run, as its results.json recorded them."""
    try:
        d = json.loads((Path(run) / "results.json").read_text())["leakage"]
    except (OSError, ValueError, KeyError):
        raise SpecError(f"{run}: no recorded set kinds and pools (results.json 'leakage'): the run is not an evaluation run, or "
                        "predates them; phase recovery will not guess which sets are held-out") from None
    return Sets({n: s["kind"] for n, s in d["sets"].items()}, {n: s["pool"] for n, s in d["sets"].items()}, d["aside"])


def sets_of(sets, run: Sequence[str] | None, reserved: Sequence[str] | None, required: bool = True) -> tuple[list[str], list[str]]:
    """(sets to mine, sets to keep unseen) with the defaults filled in; SpecError for an unknown set name, and (`required`) when
    every set is held-out. `sets`: an eval spec or `Sets`."""
    sets = sets if isinstance(sets, Sets) else of_spec(sets)
    known = list(sets.kinds)
    for key, names in (("sets", run), ("heldout", reserved)):
        if bad := [n for n in names or () if n not in known]:
            raise SpecError(f"from.{key}: unknown eval set {bad} (sets: {', '.join(known)})")
    run = list(run) if run is not None else [n for n in known if sets.kinds[n] != "heldout"]
    if required and not run:
        raise SpecError("from.sets: the eval spec has no set that is not held-out: name the sets to roll out")
    reserved = list(reserved) if reserved is not None else [n for n in known if sets.kinds[n] == "heldout" and n not in run]
    return run, reserved


def check(sets, run: Sequence[str], reserved: Sequence[str]) -> None:
    """SpecError when a set to mine is reserved, or its seeds overlap a reserved set's or an aside pool of the training spec."""
    sets = sets if isinstance(sets, Sets) else of_spec(sets)
    if both := [n for n in run if n in reserved]:
        raise SpecError(f"from.sets: {both} are also in from.heldout: they must stay unseen")
    kept = [(f"held-out set {n!r}", sets.pools[n]) for n in reserved] + [tuple(a) for a in sets.aside]
    for n in run:
        for lo, hi in sets.pools[n]:
            for what, ranges in kept:
                if any(lo <= b and a <= hi for a, b in ranges):
                    raise SpecError(f"from.sets: set {n!r} (seeds [{lo}, {hi}]) overlaps {what} {ranges}: a recorder "
                                    "would train on scenes the policy is scored on")
