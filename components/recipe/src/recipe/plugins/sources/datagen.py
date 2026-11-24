"""source datagen: the episodes of a datagen run's `manifest.json` (DATAGEN-v2.md 8.4; recovery's has the same entries).
Params: `manifest` (the file), `passed` (true default | false | any: by the evaluator's verdict). One item per episode, phase
None. Entries without a bag are left out (`no_bag`)."""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ...core.files import duration, read_json, resolve
from ...core.interfaces import Source
from ...core.items import Item, Read
from ...core.params import Params
from ...core.registry import REGISTRY


@REGISTRY.register("source", "datagen")
class DatagenSource(Source):
    kind = "datagen"

    def __init__(self, name: str, params: Mapping[str, Any], root):
        p = Params(f"sources.{name}", params, ("manifest", "passed"))
        self.name, self.manifest, self.want = name, p.path("manifest", root), p.passed()

    def read(self) -> Read:
        data = read_json(self.manifest, "a datagen manifest")
        base, items, skipped = Path(self.manifest).parent, [], Counter()
        for e in data["episodes"]:
            if self.want is not None and bool(e["passed"]) != self.want:
                skipped["not_wanted"] += 1
                continue
            result, bag = resolve(base, e["result"]), e.get("bag") and resolve(base, e["bag"])
            if not bag or not os.path.isdir(bag):
                skipped["no_bag"] += 1
                continue
            items.append(Item(self.kind, self.name, None, resolve(base, e["scene"]), e["seed"], os.path.dirname(result), result,
                              bag, None, bool(e["passed"]), e.get("task", data.get("task")), e.get("robot", data.get("robot")),
                              duration(result), {"episode": e["episode"], **e.get("datagen", {})}))
        return Read(items, skipped)
