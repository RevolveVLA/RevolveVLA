"""The leakage guard: items that come from anything the recipe must not train on are refused. `exclude:` lists directories (a
held-out set's folder: every `scene.yaml` under it names an excluded scene instance, `scene_<seed>` its seed) or scene files.
An item is refused when its scene or its episode folder is under an excluded path, or its config seed is an excluded seed."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .errors import RecipeError
from .files import resolve, scene_seed
from .items import Item


class Exclusion:
    def __init__(self, paths: Iterable[str], seeds: Iterable[int] = ()):
        self.paths, self.seeds = tuple(paths), frozenset(seeds)

    @classmethod
    def load(cls, entries: Iterable[str], root: str | Path) -> Exclusion:
        paths, seeds = [], set()
        for e in entries:
            p = Path(resolve(root, e))
            scenes = [p] if p.is_file() else sorted(p.rglob("scene.yaml")) if p.is_dir() else []
            if not scenes:
                raise RecipeError(f"exclude {e}: no scene.yaml there (a guard that excludes nothing is refused)")
            paths.append(str(p))
            seeds |= {s for s in map(scene_seed, scenes) if s is not None}
        return cls(paths, seeds)

    def refuses(self, item: Item) -> bool:
        under = lambda path: any(path == p or path.startswith(p.rstrip("/") + "/") for p in self.paths)
        return under(item.scene) or under(item.episode_dir) or item.config_seed in self.seeds
