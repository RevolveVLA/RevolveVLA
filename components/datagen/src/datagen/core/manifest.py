"""<out>/manifest.json as DATAGEN-v2.md 8.4 specifies it: one entry per episode attempted, accepted or not, merged over all
scene instances (so over all workers). It is derived, never read back: the scene logs (core/episodes.py) and the sim's
`episode_<n>.json` are the sources, the pipeline's resumable state is state.json.

manifest = {task, robot, created, episodes: [{seed, scene, robot, task, episode, result, bag, passed, datagen: {expert,
attempt, wall_s}}], counts: {scenes, attempted, passed}}. Paths are relative to the manifest's directory; `passed` is
copied from the result file, nothing else decides acceptance."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from . import episodes


def entries(out: Path, scene_dir: Path, task: str, robot: str) -> list[dict]:
    """The manifest entries of one scene instance, in try order (a try whose result file is gone is left out)."""
    def rel(p: Path) -> str:
        return os.path.relpath(p, out)
    log, found = episodes.read(scene_dir), []
    for t in log["tries"]:
        result, bag = scene_dir / f"episode_{t['episode']}.json", scene_dir / f"episode_{t['episode']}_bag"
        if result.is_file():
            found.append({"seed": log["seed"], "scene": rel(scene_dir / "scene.yaml"), "robot": robot, "task": task,
                          "episode": t["episode"], "result": rel(result), "bag": rel(bag) if bag.is_dir() else None,
                          "passed": bool(json.loads(result.read_text())["passed"]),
                          "datagen": {"expert": t["expert"], "attempt": t["attempt"], "wall_s": t["wall_s"]}})
    return found


def build(out: Path, task: str, robot: str, scenes: Sequence[Path], created: str | None = None) -> dict:
    """The manifest of the scene dirs `scenes` (in this order: the caller makes it deterministic)."""
    found = [e for d in scenes for e in entries(out, d, task, robot)]
    return {"task": task, "robot": robot, "created": created or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "episodes": found,
            "counts": {"scenes": len({e["seed"] for e in found}), "attempted": len(found),
                       "passed": sum(e["passed"] for e in found)}}


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def created(path: Path) -> str | None:
    """`created` of the manifest already at `path` (a resumed run keeps it)."""
    try:
        return json.loads(path.read_text())["created"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None
