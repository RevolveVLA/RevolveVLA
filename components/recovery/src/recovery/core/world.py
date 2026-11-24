"""Which scene instances one sim can serve: the same world (models, assets), differing at most in `robot_start`."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def world_key(scene) -> str | None:
    """The scene instance without its `robot_start` (hashed); None when the file cannot be read."""
    try:
        d = yaml.safe_load(Path(scene).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return None
    d.pop("robot_start", None)
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def same_world(a, b) -> bool:
    key = world_key(a)
    return key is not None and key == world_key(b)
