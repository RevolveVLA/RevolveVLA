"""The training spec's task (task.yaml, robot-free), for what the eval needs from it: the judge's `success:` rule."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from revolve_common.specs import Task


def load_task(name: str, root: Path) -> Task | None:
    """specs/tasks/<name>/task.yaml under `root` (the repo root), None when there is no such file. (revolve_common.specs
    pulls numpy: imported here, not at import time, so the CLI's import stays light.)"""
    from revolve_common import specs
    path = root / "specs" / "tasks" / name / "task.yaml"
    return specs.parse(specs.Task, path) if path.is_file() else None
