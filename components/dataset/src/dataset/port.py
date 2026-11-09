"""Episode dirs (each with its MCAP bag) -> one LeRobot v3.0 dataset through Rosetta's rosetta_port, unchanged
(runs in revolve-dataset with /opt/rosetta/install sourced; entry.py convert):

  port(episode_dirs, out, contract, turn)

The bags are staged as <tmp>/raw/ep_<i>/ (symlinks to bag files; rosetta_port takes every bag under
--raw-dir in sorted path order = the given order) and rosetta_port's own CLI (main) runs on them with the
legacy arguments (no --push-to-hub, default codec), writing
<out>.partial/<repo id>/. Ported from the earlier pipeline's port_episode.py: LeRobotDatasetMetadata.save_episode
is wrapped, here to add each episode's labels (labels.py) to its episode_metadata (they become
meta/episodes columns) and to print PROGRESS {"episodes_converted", "episodes_total"}. rosetta_port skips a
bag that fails and goes on: any skipped bag fails the port (labels would shift)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import labels as L

REPO_ID = "revolve/dataset"   # LeRobot needs a repo id; never pushed (the dataset dir is the root)


def stage(episode_dirs, raw: Path) -> None:
    for i, ep in enumerate(episode_dirs):
        bag = Path(ep) / "bag"
        if not (bag / "metadata.yaml").is_file():
            raise SystemExit(f"{ep}: no bag (bag/metadata.yaml)")
        d = raw / f"ep_{i:06d}"
        d.mkdir(parents=True)
        for f in bag.iterdir():
            (d / f.name).symlink_to(f.resolve())


def port(episode_dirs, out, contract, turn: int, repo=None) -> Path:
    """-> the finished dataset dir <out>.partial/<repo id> (entry.py checks it, then moves it to out)."""
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from rosetta_ros.offline import port as rosetta_port

    episode_dirs = [Path(e) for e in episode_dirs]
    labels = [L.of_episode(e, turn, repo) for e in episode_dirs]
    out = Path(out)
    work = out.parent / f"{out.name}.partial"
    shutil.rmtree(work, ignore_errors=True)        # a half-written port: LeRobot refuses an existing dataset
    raw = work / "raw"
    stage(episode_dirs, raw)
    n = len(episode_dirs)
    orig = LeRobotDatasetMetadata.save_episode
    seen = []

    def save_episode(self, episode_index, episode_length, episode_tasks, episode_stats, episode_metadata):
        meta = dict(episode_metadata, **L.columns(labels[episode_index]))
        orig(self, episode_index, episode_length, episode_tasks, episode_stats, meta)
        seen.append(episode_index)
        print("PROGRESS " + json.dumps({"stage": "port", "episodes_converted": len(seen), "episodes_total": n}), flush=True)

    LeRobotDatasetMetadata.save_episode = save_episode
    argv = sys.argv
    sys.argv = ["rosetta_port", "--raw-dir", str(raw), "--contract", str(contract), "--repo-id", REPO_ID,
                "--root", str(work / "lerobot")]
    try:
        rosetta_port.main()
    finally:
        sys.argv = argv
        LeRobotDatasetMetadata.save_episode = orig
    if seen != list(range(n)):
        raise SystemExit(f"port: {len(seen)} of {n} episodes saved (rosetta_port skipped a failed bag; see its log)")
    shutil.rmtree(raw)
    return work / "lerobot" / REPO_ID
