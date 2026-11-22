"""Training source `datagen` (the default): the training spec's datagen v2 run, `<train.out>/state.json` (or the spec's
`train_manifest`: sampled / used configs + the scene path of each) and, beside it, `manifest.json` (DATAGEN-v2.md 8.4: the
recorded episodes). A run without a state is sampled-only (core/sets.py)."""
from __future__ import annotations

import json
from pathlib import Path

from datagen.core.state import State

from ...core.errors import EvalError
from ...core.interfaces import TrainingRun, TrainingSource
from ...core.registry import REGISTRY

PATHS = ("scene", "result", "bag")          # the manifest episode paths (relative to the run's out)


@REGISTRY.register("training", "datagen")
class DatagenTraining(TrainingSource):
    def __init__(self, spec, params):
        self.spec = spec

    def load(self, root: Path) -> TrainingRun:
        path = Path(self.spec.train_manifest)
        path = path if path.is_absolute() else root / path
        if not path.exists():
            return TrainingRun(path, None)
        # State.open refuses a state made by another training spec (StateError)
        return TrainingRun(path, State.open(path, self.spec.train, persist=False).data, self._episodes(path.parent))

    @staticmethod
    def _episodes(base: Path) -> dict[int, list[dict]]:
        """The manifest.json episodes by seed, their paths made absolute (a missing bag stays None)."""
        path = base / "manifest.json"
        if not path.exists():
            return {}
        try:
            episodes = json.loads(path.read_text())["episodes"]
        except (OSError, ValueError, KeyError) as e:
            raise EvalError(f"training manifest {path}: cannot read its episodes: {e}") from e
        found: dict[int, list[dict]] = {}
        for e in episodes:
            found.setdefault(e["seed"], []).append({**e, **{k: str(base / e[k]) for k in PATHS if e.get(k)}})
        return found
