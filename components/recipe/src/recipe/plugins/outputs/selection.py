"""output selection: `<out>/selection.json` (the plan: recipe settings, asked vs got per row and split, the refused and left-out
counts, and the selected items per split, each with its bag, result, scene, phase, segment and frames). Paths are absolute. The
dataset component's exporter reads it; a LeRobot output would be another plugin."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from datagen.core import manifest

from ...core.interfaces import Output
from ...core.params import Params
from ...core.registry import REGISTRY

NAME = "selection.json"


@REGISTRY.register("output", "selection")
class SelectionOutput(Output):
    def __init__(self, params: Mapping[str, Any]):
        Params("output", params)

    def write(self, out: Path, selection: Mapping[str, Any]) -> list[Path]:
        manifest.write(Path(out) / NAME, dict(selection))
        return [Path(out) / NAME]
