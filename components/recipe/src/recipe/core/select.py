"""`take`: how much of each source goes into the recipe. Per source `all`, a number, or a number per phase; counted in the recipe's
unit (episodes, or frames: the seconds of the segment or the episode x fps). The order is balanced over scenes (one item of each
scene in turn, scenes and the items of a scene shuffled by the recipe's seed) so a few scenes cannot fill a quota; the first ones
in that order are taken until the amount is reached (frames: until it is reached or passed)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from .items import Item
from .spec import ALL, ANY_PHASE


@dataclass(frozen=True)
class Row:
    """What one `take` entry asked for and got. `asked`: the amount or `all`; `phase`: None when not asked per phase."""
    source: str
    kind: str
    phase: str | None
    asked: int | str
    got: int
    available: int

    @property
    def short(self) -> bool:
        return self.asked != ALL and self.got < self.asked


def weight(item: Item, unit: str, fps: int) -> int:
    return item.frames(fps) if unit == "frames" else 1


def spread(items: Sequence[Item], rng: random.Random) -> list[Item]:
    """The items in a balanced order: round robin over the scenes."""
    by_scene: dict[str, list[Item]] = {}
    for it in sorted(items, key=sort_key):
        by_scene.setdefault(it.scene, []).append(it)
    scenes = list(by_scene.values())
    rng.shuffle(scenes)
    for group in scenes:
        rng.shuffle(group)
    ordered = []
    while scenes:
        ordered += [g.pop() for g in scenes]
        scenes = [g for g in scenes if g]
    return ordered


def sort_key(it: Item) -> tuple:
    return (it.source, it.phase or "", it.scene, it.result, it.segment or ())


def pick(items: Sequence[Item], amount: int | str, unit: str, fps: int, rng: random.Random) -> list[Item]:
    ordered = spread(items, rng)
    if amount == ALL:
        return ordered
    taken, total = [], 0
    for it in ordered:
        if total >= amount:
            break
        taken.append(it)
        total += weight(it, unit, fps)
    return taken


def select(items: Mapping[str, Sequence[Item]], kinds: Mapping[str, str], take: Mapping[str, Mapping[str | None, int | str]],
           unit: str, fps: int, seed: int) -> tuple[list[Item], list[Row]]:
    """The taken items of every source (`items`: its usable ones) and one Row per `take` entry."""
    taken, rows = [], []
    for source, wanted in take.items():
        for phase, amount in wanted.items():
            mine = [i for i in items.get(source, ()) if phase is ANY_PHASE or i.phase == phase]
            got = pick(mine, amount, unit, fps, random.Random(f"{seed}:{source}:{phase}"))
            taken += got
            rows.append(Row(source, kinds[source], phase, amount, sum(weight(i, unit, fps) for i in got),
                            sum(weight(i, unit, fps) for i in mine)))
    return taken, rows
