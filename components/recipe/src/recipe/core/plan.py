"""The plan of a recipe: what every source has (minus what the leakage guard refuses), the taken items per `take` entry, and the
split they went to. `to_dict` is the selection.json an output writes."""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Sequence

from .interfaces import Source
from .items import Item
from .leakage import Exclusion
from .select import Row, select, weight
from .spec import RecipeSpec
from .split import assign


@dataclass
class Inventory:
    """What the sources hold: usable items per source, how many the leakage guard refused, why others were left out."""
    items: dict[str, list[Item]] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)
    refused: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, Mapping[str, int]] = field(default_factory=dict)

    @classmethod
    def read(cls, sources: Mapping[str, Source], exclusion: Exclusion) -> Inventory:
        inv = cls()
        for name, source in sources.items():
            found = source.read()
            inv.kinds[name], inv.skipped[name] = source.kind, dict(found.skipped)
            inv.items[name] = [i for i in found.items if not exclusion.refuses(i)]
            inv.refused[name] = len(found.items) - len(inv.items[name])
        return inv

    def counts(self) -> dict[tuple[str, str | None], int]:
        """Usable items per (source, phase)."""
        return dict(Counter((n, i.phase) for n, items in self.items.items() for i in items))


@dataclass
class Plan:
    spec: RecipeSpec
    inventory: Inventory
    rows: list[Row]
    splits: dict[str, list[Item]]

    @classmethod
    def build(cls, spec: RecipeSpec, inventory: Inventory) -> Plan:
        taken, rows = select(inventory.items, inventory.kinds, spec.take, spec.unit, spec.fps, spec.seed)
        return cls(spec, inventory, rows, assign(taken, spec.splits, spec.unit, spec.fps, spec.seed))

    def got(self, row: Row) -> dict[str, int]:
        """What `row` got in each split, in the recipe's unit."""
        s = self.spec
        return {name: sum(weight(i, s.unit, s.fps) for i in items if i.source == row.source and row.phase in (None, i.phase))
                for name, items in self.splits.items()}

    @property
    def short(self) -> list[Row]:
        return [r for r in self.rows if r.short]

    def problems(self, exclusion: Exclusion) -> list[str]:
        """Broken invariants of the selection (none unless a source or the guard misbehaves)."""
        found, owner, seen = [], {}, set()
        for name, items in self.splits.items():
            for i in items:
                if owner.setdefault(i.scene, name) != name:
                    found.append(f"scene {i.scene} is in splits {owner[i.scene]} and {name}")
                if exclusion.refuses(i):
                    found.append(f"{i.result} is excluded but selected")
                if not os.path.isdir(i.bag):
                    found.append(f"{i.bag}: no such bag")
                if (key := (i.result, i.segment)) in seen:
                    found.append(f"{i.result} {i.segment or ''} is selected twice")
                seen.add(key)
        return found

    def to_dict(self, created: str | None = None) -> dict:
        s = self.spec
        return {"recipe": {"hash": s.hash, "seed": s.seed, "unit": s.unit, "fps": s.fps, "splits": dict(s.splits),
                           "on_short": s.on_short},
                "created": created or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "counts": [{**asdict(r), "splits": self.got(r)} for r in self.rows],
                "refused": dict(self.inventory.refused), "skipped": {n: dict(c) for n, c in self.inventory.skipped.items() if c},
                "splits": {n: [i.to_dict(s.fps) for i in items] for n, items in self.splits.items()}}
