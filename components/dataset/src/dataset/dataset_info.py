"""Load check of one LeRobot dataset, in revolve-dataset, offline:

  python3 dataset_info.py <dataset dir> [<repo id>]

LeRobotDataset(repo_id, root=<dataset dir>) must load and decode its last frame. The last stdout
line is {"info": meta/info.json, "lengths": frames per episode in episode order, "tasks": the task
strings, "labels": each episode's labels (labels.py) in episode order}; checks.dataset_problems checks it.
Ported from the earlier pipeline's dataset_info.py (+ labels, importable as summary())."""
import json
import sys
from pathlib import Path

import labels as L


def summary(root, repo_id="revolve/dataset") -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id, root=root)
    episodes = ds.meta.episodes
    rows = sorted(episodes, key=lambda r: int(r["episode_index"]))
    lengths = [int(r["length"]) for r in rows]
    if len(ds) != sum(lengths):
        raise SystemExit(f"{len(ds)} frames in the dataset, {sum(lengths)} in its episodes")
    ds[len(ds) - 1]         # decodes the last frame's images
    has = all(L.PREFIX + k in episodes.column_names for k in L.NAMES)
    return dict(info=json.loads((Path(root) / "meta" / "info.json").read_text()), lengths=lengths,
                tasks=[str(t) for t in ds.meta.tasks.index], labels=[L.of_row(r) for r in rows] if has else None)


if __name__ == "__main__":
    print(json.dumps(summary(*sys.argv[1:])))
