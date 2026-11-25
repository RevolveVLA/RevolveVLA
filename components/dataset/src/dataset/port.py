"""The passed episodes' bags -> one LeRobot v3.0 dataset through Rosetta's porter (rosetta_ros.offline.port), unchanged.
In the dataset image, with Rosetta's install sourced.

  built, frames, problems = port(bags, labels, contract_path, work, names)

The bags are staged as <work>/raw/ep_<i>/ (symlinks to each bag's files: the porter takes every bag under its raw dir
in sorted path order, so the staging order is the manifest order). LeRobotDatasetMetadata.save_episode is wrapped to
add each episode's labels to its meta/episodes row and to print PROGRESS after each saved episode. The porter only
logs what it does to a bad stream and carries on, so its log is read for problems (LogProblems): a failed episode, an
action topic missing from a bag (zero-filled), and a message it drops because it cannot decode it or it lacks the
contract's timeline (that stream then reads zeros until a good message arrives)."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from . import labels as L
from .layout import REPO_ID, progress


def stage(bags, raw: Path) -> None:
    for i, bag in enumerate(bags):
        d = raw / f"ep_{i:06d}"
        d.mkdir(parents=True)
        for f in sorted(Path(bag).iterdir()):
            (d / f.name).symlink_to(f.resolve())


class LogProblems(logging.Handler):
    """Rosetta's porter log -> problem strings ("port: <episode name>: <log line>").

    Its lines at the pinned commit (rosetta_ros/offline/port.py, offline/bag_frames.py, ingest.py): before each
    episode `"<i> / <n> episodes processed"` (INFO, names the episode the next lines belong to); the problems below.
    Not problems: the recovery notices (`Decode recovered ...`, `Timeline '...' recovered ...`), the warmup lines,
    the contract-mismatch warning (the bags embed no contract), progress and frame counts."""

    EPISODE = re.compile(r"^(\d+) / \d+ episodes processed$")
    PATTERNS = (re.compile(r"^Decode failed for '.+' on .+; dropping message$", re.S),       # ingest.py
                re.compile(r"^Message on '.+' \(.+\) is missing its '.+' timeline; dropping message$", re.S),
                re.compile(r"^Topic .+ not in bag, .+ will be zero-filled$"),                  # bag_frames.py
                re.compile(r"^\s*-> FAILED .+", re.S))                                         # port.py

    def __init__(self, names):
        super().__init__(logging.INFO)
        self.names, self.out, self.current = list(names), [], None

    def emit(self, record):
        msg = record.getMessage()
        if m := self.EPISODE.match(msg.strip()):
            self.current = int(m.group(1))
            return
        if record.levelno >= logging.WARNING and any(p.match(msg) for p in self.PATTERNS):
            i = self.current
            name = self.names[i] if i is not None and 0 <= i < len(self.names) else "an episode"
            self.out.append(f"port: {name}: {msg.strip()}")


def port(bags, labels, contract_path, work: Path, names=None):
    """-> (the dataset directory, frames per saved episode in order, problems)."""
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from rosetta_ros.offline import port as rosetta_port

    raw, root = work / "raw", work / "lerobot"
    stage(bags, raw)
    n, saved = len(bags), []
    orig = LeRobotDatasetMetadata.save_episode

    def save_episode(self, episode_index, episode_length, episode_tasks, episode_stats, episode_metadata, *a, **k):
        meta = dict(episode_metadata, **L.columns(labels[episode_index]))
        orig(self, episode_index, episode_length, episode_tasks, episode_stats, meta, *a, **k)
        saved.append(int(episode_length))
        progress(stage="port", episodes_converted=len(saved), episodes_total=n)

    logging.basicConfig(level=logging.INFO, format="%(message)s")     # the porter's log, as its own CLI prints it
    log = logging.getLogger()
    log.setLevel(min(log.level or logging.INFO, logging.INFO))
    handler = LogProblems(names or [str(b) for b in bags])
    log.addHandler(handler)
    LeRobotDatasetMetadata.save_episode = save_episode
    problems = []
    try:
        rosetta_port.port(raw_dir=raw, repo_id=REPO_ID, contract_path=Path(contract_path), root=root,
                          writer_opts={"push_to_hub": False})
    except Exception as e:      # every bag failed, or the writer did
        problems.append(f"port: {type(e).__name__}: {e}")
    finally:
        LeRobotDatasetMetadata.save_episode = orig
        log.removeHandler(handler)
    problems += handler.out
    if len(saved) != n:
        problems.append(f"port: {len(saved)} of {n} episodes saved (the porter skips a failed episode)")
    return root / REPO_ID, saved, problems
