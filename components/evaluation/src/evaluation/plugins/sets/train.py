"""In-distribution set: `count` of the configs the policy was trained on (spread over the stratification cells), started
from the training run's own accepted check dirs (the training manifest's `generate.paths.check`)."""
from __future__ import annotations

from pathlib import Path

from datagen.core.interfaces import Config

from ...core import sets
from ...core.errors import EvalError
from ...core.interfaces import Built, EvalItem, EvalSet
from ...core.registry import REGISTRY


@REGISTRY.register("set", "train")
class TrainSet(EvalSet):
    def __init__(self, spec):
        self.spec = spec

    def build(self, set_spec, out, workers):
        root = sets.repo_root(self.spec, out)
        training = sets.Training.load(self.spec, root)
        training.need_manifest(f"set {set_spec.name!r} (from: {set_spec.source})")
        entries = sets.spread(training.usable(set_spec.source), set_spec.count)
        if len(entries) < set_spec.count:
            raise EvalError(f"set {set_spec.name!r}: training split {set_spec.source!r} has {len(entries)} usable "
                            f"configs in {training.path}, {set_spec.count} wanted")
        items = []
        for e in entries:
            check = e["generate"].get("paths", {}).get("check")
            start = Path(check) if check and Path(check).is_absolute() else root / (check or "")
            if not check or not start.is_dir():
                raise EvalError(f"set {set_spec.name!r}: no start dir for training seed {e['config']['seed']} "
                                f"({check or 'generate.paths.check missing'}) in {training.path}")
            items.append(EvalItem(Config.from_dict(e["config"]), start))
        return Built(items)
