"""Paths and run files the sources share."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .errors import RecipeError


def resolve(base: str | Path, path: str | Path) -> str:
    """`path` (absolute, or relative to `base`) as a normalised absolute string."""
    return os.path.normpath(os.path.join(os.path.abspath(base), path))


def read_json(path: str | Path, what: str) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise RecipeError(f"{path}: not {what} ({e})") from None


def duration(result: str | Path) -> float | None:
    """`duration_s` of an episode result file (None when the file or the key is missing)."""
    try:
        d = json.loads(Path(result).read_text()).get("duration_s")
    except (OSError, ValueError, AttributeError):
        return None
    return float(d) if isinstance(d, (int, float)) and not isinstance(d, bool) else None


def scene_seed(scene: str | Path) -> int | None:
    """The sampling seed in the name of a scene instance's folder (`<...>/scene_<seed>/scene.yaml`)."""
    m = re.fullmatch(r"scene_(\d+)", Path(scene).parent.name)
    return int(m.group(1)) if m else None
