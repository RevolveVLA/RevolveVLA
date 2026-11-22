"""In-distribution set: `count` of the configs the policy was trained on (spread over the stratification cells), started
from the training run's own scene instances (the training state's `generate.paths.scene`) with the episodes datagen recorded
on them."""
from __future__ import annotations

from datagen.core.interfaces import Config

from ...core import sets
from ...core.errors import EvalError
from ...core.interfaces import Built, EvalItem, EvalSet
from ...core.registry import REGISTRY


@REGISTRY.register("set", "train")
class TrainSet(EvalSet):
    def __init__(self, spec):
        self.spec = spec

    def build(self, set_spec, out, workers, abort=None):
        training = sets.Training.load(self.spec, sets.repo_root(self.spec, out))
        training.need_manifest(f"set {set_spec.name!r} (from: {set_spec.source})")
        entries = sets.spread(training.usable(set_spec.source), set_spec.count)
        if len(entries) < set_spec.count:
            raise EvalError(f"set {set_spec.name!r}: training split {set_spec.source!r} has {len(entries)} usable "
                            f"configs in {training.path}, {set_spec.count} wanted")
        items = []
        for e in entries:
            seed, rel = e["config"]["seed"], e["generate"].get("paths", {}).get("scene")
            scene = training.base / rel if rel else None
            if not scene or not scene.is_file():
                raise EvalError(f"set {set_spec.name!r}: no scene instance for training seed {seed} "
                                f"({scene or 'generate.paths.scene missing'}) in {training.path}")
            items.append(EvalItem(Config.from_dict(e["config"]), scene, tuple(training.episodes.get(seed, ()))))
        return Built(items)
