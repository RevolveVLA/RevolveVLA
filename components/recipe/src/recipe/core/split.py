"""`splits`: the taken items divided into train / val / ... by fractions of the recipe's unit, grouped by scene: one scene (all the
items of every source that start from it) goes to one split only. Scenes are shuffled by the recipe's seed, then each goes to the
split furthest below its share."""
from __future__ import annotations

import random
from typing import Mapping, Sequence

from .items import Item
from .select import sort_key, weight


def assign(items: Sequence[Item], fractions: Mapping[str, float], unit: str, fps: int, seed: int) -> dict[str, list[Item]]:
    by_scene: dict[str, list[Item]] = {}
    for it in sorted(items, key=sort_key):
        by_scene.setdefault(it.scene, []).append(it)
    scenes = sorted(by_scene)
    random.Random(f"{seed}:split").shuffle(scenes)
    total = sum(weight(i, unit, fps) for i in items)
    given = dict.fromkeys(fractions, 0)
    out: dict[str, list[Item]] = {name: [] for name in fractions}
    for scene in scenes:
        name = max(fractions, key=lambda n: fractions[n] * total - given[n])
        out[name] += by_scene[scene]
        given[name] += sum(weight(i, unit, fps) for i in by_scene[scene])
    return {n: sorted(v, key=sort_key) for n, v in out.items()}
