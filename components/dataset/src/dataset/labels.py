"""Episode labels: columns `labels/<name>` of the LeRobot dataset's meta/episodes parquet (one row per
episode, beside LeRobot's own `stats/...`, `data/...`), so `LeRobotDataset(...).meta.episodes` carries them
and the training side filters with `select()` -> episode indices for LeRobot's `--dataset.episodes`.

  source        demo | recovery
  phase         recovery: the phase the suffix restarts from; demo: "" (full episode)
  suffix        recovery: true (suffix-only, from the phase's saved start state); demo: false
  seed          demo: the datagen seed; recovery: the failed rollout's seed (-1 when unknown)
  turn          the experiment turn the episode was generated in
  from_rollout  recovery: the rollout dir it was recovered from (repo-relative); demo: ""
  episode_dir   the source episode dir (repo-relative)

Fixed types (str/bool/int, never null): LeRobot's metadata parquet writer keeps the first row's schema.
Pure Python: no LeRobot import (unit-tested on the host)."""
from __future__ import annotations

import json
from pathlib import Path

PREFIX = "labels/"
NAMES = ("source", "phase", "suffix", "seed", "turn", "from_rollout", "episode_dir")
SOURCES = ("demo", "recovery")


def rel(p, repo=None) -> str:
    """Container (/repo/...) or host repo path -> repo-relative; anything else unchanged."""
    s = str(p)
    for root in filter(None, (repo and str(repo).rstrip("/"), "/repo")):
        if s.startswith(root + "/"):
            return s[len(root) + 1:]
    return s


def of_episode(ep_dir, turn: int, repo=None) -> dict:
    """Labels of one episode dir from its episode.json: `recovery` block (components/recovery) or
    `datagen` block (components/datagen)."""
    e = json.loads((Path(ep_dir) / "episode.json").read_text())
    if "recovery" in e:
        r = e["recovery"]
        seed = r.get("rollout_seed")
        d = dict(source="recovery", phase=str(r["phase"]), suffix=bool(r.get("suffix", True)),
                 seed=-1 if seed is None else int(seed), from_rollout=rel(r.get("from_rollout") or "", repo))
    elif "datagen" in e:
        d = dict(source="demo", phase="", suffix=False, seed=int(e["datagen"]["seed"]), from_rollout="")
    else:
        raise ValueError(f"{ep_dir}/episode.json: neither a datagen nor a recovery episode")
    return dict(d, turn=int(turn), episode_dir=rel(ep_dir, repo))


def columns(labels: dict) -> dict:
    return {PREFIX + k: labels[k] for k in NAMES}


def of_row(row) -> dict:
    """A meta/episodes row (dict-like) -> its labels."""
    return {k: row[PREFIX + k] for k in NAMES}


def select(episodes, **want) -> list[int]:
    """Episode indices whose labels equal every `want` (e.g. source="recovery", phase="open").
    episodes: LeRobotDatasetMetadata.episodes (a datasets.Dataset) or an iterable of row dicts."""
    bad = set(want) - set(NAMES)
    if bad:
        raise ValueError(f"unknown labels {sorted(bad)}; known: {NAMES}")
    return [int(r["episode_index"]) for r in episodes if all(r[PREFIX + k] == v for k, v in want.items())]
