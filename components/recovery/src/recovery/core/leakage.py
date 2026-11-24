"""The leakage guard of a live run: the scenes a live recorder trains on must not be scenes the policy is scored on. Of the
eval spec's sets, `run` are rolled out, `reserved` are kept unseen (default: its held-out sets not in `run`; plus, always, the
pools the training spec keeps aside: its splits with `record: false`). The seeds (pool ranges) of the two must not overlap."""
from __future__ import annotations

from typing import Sequence

from datagen.core.spec import SpecError


def pool_of(eval_spec, name: str) -> list[tuple[int, int]]:
    """The seed ranges a set draws its configs from (a `train` set: the training split's pool)."""
    s = eval_spec.sets[name]
    if s.kind == "train":
        return next(p.pool for p in eval_spec.train.splits if p.name == s.source).ranges()
    return s.pool.ranges()


def sets_of(eval_spec, run: Sequence[str] | None, reserved: Sequence[str] | None) -> tuple[list[str], list[str]]:
    """(sets to roll out, sets to keep unseen) with the defaults filled in; SpecError for an unknown set name."""
    known = list(eval_spec.sets)
    for key, names in (("sets", run), ("heldout", reserved)):
        if bad := [n for n in names or () if n not in known]:
            raise SpecError(f"from.{key}: unknown eval set {bad} (sets: {', '.join(known)})")
    run = list(run) if run is not None else [n for n in known if eval_spec.sets[n].kind != "heldout"]
    if not run:
        raise SpecError("from.sets: the eval spec has no set that is not held-out: name the sets to roll out")
    reserved = list(reserved) if reserved is not None else [n for n in known if eval_spec.sets[n].kind == "heldout" and n not in run]
    return run, reserved


def check(eval_spec, run: Sequence[str], reserved: Sequence[str]) -> None:
    """SpecError when a set to roll out is reserved, or its seeds overlap a reserved set's or an aside pool of the training spec."""
    if both := [n for n in run if n in reserved]:
        raise SpecError(f"from.sets: {both} are also in from.heldout: they must stay unseen")
    aside = [(f"training split {s.name!r} (kept aside)", s.pool.ranges()) for s in eval_spec.train.splits if not s.record]
    kept = [(f"held-out set {n!r}", pool_of(eval_spec, n)) for n in reserved] + aside
    for n in run:
        for lo, hi in pool_of(eval_spec, n):
            for what, ranges in kept:
                if any(lo <= b and a <= hi for a, b in ranges):
                    raise SpecError(f"from.sets: set {n!r} (seeds [{lo}, {hi}]) overlaps {what} {ranges}: the live recorder "
                                    "would train on scenes the policy is scored on")
