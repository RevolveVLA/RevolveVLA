"""source rollouts: the tries of an evaluation run (`<run>/results.json`): one item per try that has a verdict and a bag.
Params: `run` (the eval run's out), `set` (a set name or list of them; default all), `passed` (true default | false | any: by the
evaluator's verdict of the try). Phase None; `labels` hold set, rollout, try and failed_phase. Tries with a stack error are left
out (`error`), tries without a bag `no_bag`."""
from __future__ import annotations

import os
from collections import Counter
from typing import Any, Mapping

from datagen.core.spec import SpecError

from ...core.files import duration, read_json, resolve
from ...core.interfaces import Source
from ...core.items import Item, Read
from ...core.params import Params
from ...core.registry import REGISTRY


@REGISTRY.register("source", "rollouts")
class RolloutsSource(Source):
    kind = "rollout"

    def __init__(self, name: str, params: Mapping[str, Any], root):
        p = Params(f"sources.{name}", params, ("run", "set", "passed"))
        sets = p.params.get("set", [])
        sets = [sets] if isinstance(sets, str) else sets
        if not isinstance(sets, list) or not all(isinstance(s, str) for s in sets):
            raise SpecError(f"sources.{name}.set: expected a set name or a list of them, got {p.params['set']!r}")
        self.name, self.root, self.run = name, root, p.path("run", root)
        self.sets, self.want = set(sets), p.passed()

    def read(self) -> Read:
        data = read_json(os.path.join(self.run, "results.json"), "an evaluation run (no results.json)")
        items, skipped = [], Counter()
        for set_name, block in sorted(data["sets"].items()):
            if self.sets and set_name not in self.sets:
                continue
            for key, entry in sorted(block.get("rollouts", {}).items()):
                for i, t in enumerate(entry["tries"]):
                    rec = t["record"]
                    if rec.get("error"):
                        skipped["error"] += 1
                    elif self.want is not None and bool(rec["passed"]) != self.want:
                        skipped["not_wanted"] += 1
                    elif not (rec.get("result_file") and rec.get("bag") and os.path.isdir(resolve(self.root, rec["bag"]))):
                        skipped["no_bag"] += 1
                    else:
                        result = resolve(self.root, rec["result_file"])
                        items.append(Item(self.kind, self.name, None, resolve(self.root, rec["scene"]), entry["seed"],
                                          os.path.dirname(result), result, resolve(self.root, rec["bag"]), None,
                                          bool(rec["passed"]), None, None, rec.get("duration_s") or duration(result),
                                          {"set": set_name, "rollout": key, "try": i, "failed_phase": rec.get("failed_phase"),
                                           "policy": rec.get("policy")}))
        return Read(items, skipped)
