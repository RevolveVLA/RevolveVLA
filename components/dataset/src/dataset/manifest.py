"""The producers' manifests -> the episodes to convert, and the fail-closed input checks (DATASET-v2.md sections 6
and 8.3). Pure Python (unit-tested on the host): nothing here reads a bag's messages. Problems start with `inputs:`.

  plan = read(manifest_args, split=None, turn=0)     # passed entries only, in order; plan.problems
  problems = check_inputs(plan.episodes, contract_topics)   # results and bags, before any porting

A manifest is the schema datagen writes (components/datagen/src/datagen/core/manifest.py): {task, robot, created,
episodes: [{seed, scene, robot, task, episode, result, bag, passed, <producer block>}], counts}. Entry paths are
relative to the manifest's directory, or absolute. Nothing is re-judged (`passed` is the evaluator's), no folder is
scanned, no producer-private file is read."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import labels as L

BAG_METADATA = "metadata.yaml"
_UNREADABLE = object()       # a result file that could not be read (its problem is already listed)


@dataclass
class Episode:
    where: str              # "<manifest> entry <i>": how problems name the episode
    bag: Path
    result: Path
    labels: dict
    duration_s: float | None = None     # the evaluator's, filled by check_inputs


@dataclass
class Plan:
    task: str | None = None
    robot: str | None = None
    episodes: list[Episode] = field(default_factory=list)
    digests: list[str] = field(default_factory=list)     # sha256 of each manifest file, in the given order
    problems: list[str] = field(default_factory=list)


def _path(base: Path, p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else base / p


def read(manifests, split=None, turn=0) -> Plan:
    """manifests: paths as given on the command line (each becomes the `manifest` label). -> the passed entries of
    split `split` (all splits if None), manifests in the given order, each in its own order."""
    plan = Plan()
    for m in manifests:
        try:
            raw = Path(m).read_bytes()
            doc = json.loads(raw)
        except (OSError, ValueError) as e:
            plan.problems.append(f"inputs: {m}: unreadable manifest ({e})")
            continue
        plan.digests.append(hashlib.sha256(raw).hexdigest())
        if not isinstance(doc, dict) or not isinstance(doc.get("episodes"), list):
            plan.problems.append(f"inputs: {m}: not a manifest (no `episodes` list)")
            continue
        for k in ("task", "robot"):
            v = doc.get(k)
            if not isinstance(v, str) or not v:
                plan.problems.append(f"inputs: {m}: no `{k}`")
            elif getattr(plan, k) is None:
                setattr(plan, k, v)
            elif getattr(plan, k) != v:
                plan.problems.append(f"inputs: {m}: {k} {v!r} != {getattr(plan, k)!r} of the first manifest")
        base = Path(m).parent
        for i, e in enumerate(doc["episodes"]):
            where = f"{m} entry {i}"
            if not isinstance(e, dict):
                plan.problems.append(f"inputs: {where}: not a mapping")
                continue
            if e.get("passed") is not True:
                if not isinstance(e.get("passed"), bool):
                    plan.problems.append(f"inputs: {where}: `passed` is {e.get('passed')!r}, not true or false")
                continue
            where += f" (scene {e.get('scene')}, episode {e.get('episode')})"
            bad = [f"{k} {e.get(k)!r} != the manifest's {doc.get(k)!r}" for k in ("task", "robot")
                   if e.get(k) != doc.get(k)]
            if bad:
                plan.problems.append(f"inputs: {where}: {'; '.join(bad)}")
                continue
            try:
                lab = L.of_entry(e, turn, str(m))
            except L.LabelError as err:
                plan.problems.append(f"inputs: {where}: labels: {err}")
                continue
            if split is not None and lab["split"] != split:
                continue
            paths = {k: e.get(k) for k in ("bag", "result")}
            if not all(isinstance(v, str) and v for v in paths.values()):
                plan.problems.append(f"inputs: {where}: no `bag` or `result` path ({paths})")
                continue
            plan.episodes.append(Episode(where, _path(base, paths["bag"]), _path(base, paths["result"]), lab))
    if not plan.problems and not plan.episodes:
        plan.problems.append(f"inputs: no passed episodes in {list(map(str, manifests))}"
                             + (f" for split {split!r}" if split is not None else ""))
    return plan


def bag_message_counts(bag: Path) -> dict:
    """{topic: message count} from a rosbag2 directory's metadata.yaml."""
    info = yaml.safe_load((bag / BAG_METADATA).read_text())["rosbag2_bagfile_information"]
    return {t["topic_metadata"]["name"]: int(t["message_count"]) for t in info.get("topics_with_message_count") or []}


def check_inputs(episodes, contract_topics, task, robot) -> list[str]:
    """Before porting: every episode's result exists, says passed and is this task and robot, with a duration; its
    bag exists and holds every contract topic with at least one message. Fills Episode.duration_s."""
    problems = []
    for ep in episodes:
        try:
            r = json.loads(ep.result.read_text())
        except (OSError, ValueError) as e:
            problems.append(f"inputs: {ep.where}: result {ep.result}: unreadable ({e})")
            r = _UNREADABLE
        if isinstance(r, dict):
            if r.get("passed") is not True:
                problems.append(f"inputs: {ep.where}: result {ep.result} is not passed (passed: {r.get('passed')!r})")
            for k, want in (("task", task), ("robot", robot)):
                if k in r and r[k] != want:
                    problems.append(f"inputs: {ep.where}: result {k} {r[k]!r} != {want!r}")
            d = r.get("duration_s")
            if isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d) or d < 0:
                problems.append(f"inputs: {ep.where}: result {ep.result} has no duration_s ({d!r})")
            else:
                ep.duration_s = float(d)
        elif r is not _UNREADABLE:
            problems.append(f"inputs: {ep.where}: result {ep.result} is not a JSON object")
        if not (ep.bag / BAG_METADATA).is_file():
            problems.append(f"inputs: {ep.where}: bag {ep.bag}: no {BAG_METADATA}")
            continue
        try:
            counts = bag_message_counts(ep.bag)
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as e:
            problems.append(f"inputs: {ep.where}: bag {ep.bag}: unreadable {BAG_METADATA} ({e!r})")
            continue
        missing = [t for t in contract_topics if counts.get(t, 0) == 0]
        if missing:
            problems.append(f"inputs: {ep.where}: bag {ep.bag} lacks contract topics (absent or no messages): "
                            f"{missing}")
    return problems
