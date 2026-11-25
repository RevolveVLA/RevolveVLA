"""Episode labels: columns `labels/<name>` of a dataset's meta/episodes parquet (one row per episode, beside LeRobot's
own columns), so `LeRobotDataset(...).meta.episodes` carries them (DATASET-v2.md section 6). Training filters with
`select(meta.episodes, source="recovery")` -> episode indices for LeRobot's `--dataset.episodes`.

  source    demo (the manifest entry has a `datagen` block) | recovery (a `recovery` block)
  seed      entry.seed
  split     entry.split if the producer writes one, else <split> of a scene path <split>/<scene dir>/<file>
  scene     entry.scene, as written in the manifest
  episode   entry.episode (the simulator's episode number in its run dir)
  expert    datagen.expert (recovery: its `expert`, else "")
  attempt   datagen.attempt (recovery: its `attempt`, else -1)
  turn      --turn K
  manifest  the manifest's path as given on the command line

Fixed types, never null: LeRobot's metadata parquet writer keeps the first row's schema. Pure Python."""
from __future__ import annotations

PREFIX = "labels/"
TYPES = {"source": str, "seed": int, "split": str, "scene": str, "episode": int, "expert": str, "attempt": int,
         "turn": int, "manifest": str}
NAMES = tuple(TYPES)
SOURCES = {"datagen": "demo", "recovery": "recovery"}      # the entry's producer block -> source


class LabelError(ValueError):
    pass


def _typed(name, v):
    t = TYPES[name]
    if isinstance(v, bool) or not isinstance(v, t):      # bool is an int to Python, never a label value
        raise LabelError(f"{name}: {v!r} is not a {t.__name__}")
    return v


def split_of(entry: dict) -> str:
    """The entry's split: its `split` field, else the first component of its relative `scene` path, which must be
    `<split>/<scene dir>/<file>` (at least three components: `scene_7/scene.yaml` names no split)."""
    if "split" in entry:
        return _typed("split", entry["split"])
    scene = entry.get("scene")
    parts = scene.split("/") if isinstance(scene, str) and not scene.startswith("/") else []
    if len(parts) < 3 or not all(parts):
        raise LabelError(f"split: no `split` field and no split directory in scene {scene!r}")
    return parts[0]


def of_entry(entry: dict, turn: int, manifest: str) -> dict:
    """One manifest entry -> its labels; LabelError when the entry cannot be labelled (fail closed)."""
    blocks = [b for b in SOURCES if b in entry]
    if len(blocks) != 1:
        raise LabelError(f"source: the entry needs exactly one of the blocks {sorted(SOURCES)}, has {blocks}")
    block = entry[blocks[0]]
    if not isinstance(block, dict):
        raise LabelError(f"source: the `{blocks[0]}` block is not a mapping")
    if blocks[0] == "datagen":
        expert, attempt = block.get("expert"), block.get("attempt")
    else:
        expert, attempt = block.get("expert", ""), block.get("attempt", -1)
    d = dict(source=SOURCES[blocks[0]], seed=entry.get("seed"), split=split_of(entry), scene=entry.get("scene"),
             episode=entry.get("episode"), expert=expert, attempt=attempt, turn=turn, manifest=manifest)
    return {k: _typed(k, d[k]) for k in NAMES}


def columns(labels: dict) -> dict:
    return {PREFIX + k: labels[k] for k in NAMES}


def of_row(row) -> dict:
    """A meta/episodes row (dict-like) -> its labels."""
    return {k: row[PREFIX + k] for k in NAMES}


def select(episodes, **want) -> list[int]:
    """Episode indices whose labels equal every `want` (e.g. source="recovery", split="train").
    episodes: LeRobotDatasetMetadata.episodes (a datasets.Dataset) or an iterable of row dicts."""
    bad = set(want) - set(NAMES)
    if bad:
        raise ValueError(f"unknown labels {sorted(bad)}; known: {NAMES}")
    return [int(r["episode_index"]) for r in episodes if all(r[PREFIX + k] == v for k, v in want.items())]
