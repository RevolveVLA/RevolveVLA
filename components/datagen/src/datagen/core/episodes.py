"""The episodes of one scene instance: the per-scene log `<scene dir>/datagen.json` and the loop that fills it.

The scene dir is the sim's `--out` (DATAGEN-v2.md 8.3): `scene.yaml` (the instance), `episode_<n>.json` + `episode_<n>_bag/`
(the sim) and `datagen.json` (ours): {seed, tries: [{attempt, episode, passed, reason, wall_s, expert, notes}], faults:
[kind]}. A try is one episode the evaluator ruled on (a failed one stays: it is data); `faults` are the infrastructure
relaunches, which are not tries. The loop runs wherever a client of the sim runs (a client container, or the CPU fake);
it is resumable from the log alone, and `finished` decides from it whether a sim is needed at all."""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

from .errors import DatagenError
from .interfaces import EpisodeClient, Generator, Result
from .spec import EpisodeSpec

LOG = "datagen.json"


class SceneError(DatagenError):
    """A scene instance that cannot be run (not a failed episode): infrastructure retries used up, a crashed generator."""


def read(scene_dir: Path) -> dict:
    try:
        return json.loads((Path(scene_dir) / LOG).read_text())
    except FileNotFoundError:
        return {"tries": [], "faults": []}


def write(scene_dir: Path, log: Mapping[str, Any]) -> None:
    path = Path(scene_dir) / LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(log, indent=1))
    os.replace(tmp, path)


def add_faults(scene_dir: Path, faults: list[str]) -> None:
    """Append the infrastructure relaunches of a run of the scene to its log."""
    if faults:
        log = read(scene_dir)
        write(scene_dir, {**log, "faults": [*log["faults"], *faults]})


def accepted(log: Mapping) -> int:
    return sum(t["passed"] for t in log["tries"])


def finished(log: Mapping, episodes: EpisodeSpec) -> bool:
    """The scene has its accepted episodes, or no tries left."""
    return accepted(log) >= episodes.per_scene or len(log["tries"]) >= episodes.attempts


def verdict(log: Mapping, episodes: EpisodeSpec) -> Result:
    """ok when the scene has `per_scene` accepted episodes; else why not (the last failed try's reason)."""
    if accepted(log) >= episodes.per_scene:
        return Result()
    last = log["tries"][-1]["reason"] if log["tries"] else "no episode"
    return Result.fail(f"{accepted(log)}/{episodes.per_scene} accepted in {len(log['tries'])} tries (last: {last})")


def clean_partial(scene_dir: Path) -> None:
    """Remove the bags of episodes that never got their `episode_<n>.json` (a killed sim): by the completion rule (4.3)
    they are incomplete, and the sim numbers its next episode after the highest result file, so it would reuse them."""
    for bag in Path(scene_dir).glob("episode_*_bag"):
        n = re.fullmatch(r"episode_(\d+)_bag", bag.name)
        if n and not (bag.parent / f"episode_{n[1]}.json").exists():
            shutil.rmtree(bag, ignore_errors=True)


def run_episodes(client: EpisodeClient, generator: Generator, task: str, robot: str, instance: Path, seed: int,
                 episodes: EpisodeSpec, clock=time.monotonic) -> dict:
    """The client loop of DATAGEN-v2.md 4.4 on a freshly launched sim of `instance`: ready, then tries (the first needs
    no reset, later ones reset the sim) until the scene is `finished`. -> the log."""
    scene_dir = Path(instance).parent
    clean_partial(scene_dir)
    log = {**read(scene_dir), "seed": seed}
    if finished(log, episodes):
        return log
    client.wait_ready()
    first = True
    while not finished(log, episodes):
        if not first:
            client.reset()
        first, t0 = False, clock()
        n = client.start_episode()
        notes = generator.drive(client, instance, task, robot)
        client.wait_result(n)
        result = client.wait_files(n)
        log["tries"].append({"attempt": len(log["tries"]), "episode": n, "passed": bool(result["passed"]),
                             "reason": result.get("reason"), "wall_s": round(clock() - t0, 1),
                             "expert": notes["expert"], "notes": notes.get("notes", "")})
        write(scene_dir, log)
    return log
