"""One LeRobot dataset, loaded the way training loads it, offline, in the dataset image:

  summary(root) -> {"info": meta/info.json, "lengths": frames per episode in episode order, "tasks": meta/tasks'
                    strings, "task_indices": the distinct task_index values of the frames, "labels": each episode's
                    labels (None without labels/* columns)}

LeRobotDataset(REPO_ID, root=<dir>) must load and decode its last frame (images included); checks.problems reads the
rest."""
from __future__ import annotations

import json
from pathlib import Path

from . import labels as L
from .layout import REPO_ID


def summary(root) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(REPO_ID, root=root)
    episodes = ds.meta.episodes
    rows = sorted(episodes, key=lambda r: int(r["episode_index"]))
    lengths = [int(r["length"]) for r in rows]
    if len(ds) != sum(lengths):
        raise RuntimeError(f"{root}: {len(ds)} frames in the dataset, {sum(lengths)} in its episodes")
    if len(ds):
        ds[len(ds) - 1]         # decodes the last frame, its video frames too
    has = all(L.PREFIX + k in episodes.column_names for k in L.NAMES)
    return dict(info=json.loads((Path(root) / "meta" / "info.json").read_text()), lengths=lengths,
                tasks=[str(t) for t in ds.meta.tasks.index],
                task_indices=sorted({int(i) for i in ds.hf_dataset.unique("task_index")}),
                labels=[L.of_row(r) for r in rows] if has else None)
