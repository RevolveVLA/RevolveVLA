"""Training source `datagen` (the default): the training spec's datagen state, `<train.out>/state.json` or the spec's
`train_manifest` (datagen v2 keeps manifest.json for DATAGEN-v2.md 8.4). A run without it is sampled-only (core/sets.py)."""
from __future__ import annotations

from pathlib import Path

from datagen.core.state import State as Manifest        # datagen v2: the resumable state is state.json

from ...core.interfaces import TrainingRun, TrainingSource
from ...core.registry import REGISTRY


@REGISTRY.register("training", "datagen")
class DatagenTraining(TrainingSource):
    def __init__(self, spec, params):
        self.spec = spec

    def load(self, root: Path) -> TrainingRun:
        path = Path(self.spec.train_manifest)
        path = path if path.is_absolute() else root / path
        # Manifest.open refuses a manifest made by another training spec (ManifestError)
        return TrainingRun(path, Manifest.open(path, self.spec.train, persist=False).data if path.exists() else None)
