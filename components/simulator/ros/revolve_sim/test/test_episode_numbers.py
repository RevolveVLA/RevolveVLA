"""`revolve_sim.episode.next_episode`: one past the highest n of the run dir's episode_<n>.json, episode_<n>_gz.log
and episode_<n>_bag* entries, so a relaunch into the same dir overwrites nothing (host only, no ROS)."""
import sys
from pathlib import Path

import pytest
from host_cases import snapshot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revolve_sim.episode import next_episode  # noqa: E402


@pytest.mark.parametrize("files, dirs, expected", [
    ([], [], 0),
    (["episode_0.json", "episode_0_gz.log"], ["episode_0_bag"], 1),     # one complete episode
    (["episode_0.json"], ["episode_0_bag", "episode_1_bag"], 2),        # a run killed mid-episode: its bag counts
    (["episode_4_gz.log"], ["episode_2_bag(1)"], 5),                    # a gz log alone; a renamed bag copy
    (["episode_9.json", "episode_10.json", "episode_07.json"], [], 11),  # by value, not string order
    (["episode_x.json", "episode_3.json.tmp", "episode3.json", "my_episode_3.json", "episode_3_gz.json",
      "episode_3_rosbag", "episode_-3.json", "notes.txt"], ["episode_3"], 0),    # other names are not episodes
])
def test_next_episode(tmp_path, files, dirs, expected):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in files:
        (run_dir / name).write_text("x")
    for name in dirs:
        (run_dir / name).mkdir()
    (run_dir / "sub").mkdir()
    (run_dir / "sub" / "episode_99.json").write_text("x")      # only run_dir's own entries count
    before = snapshot(run_dir)
    assert next_episode(run_dir) == expected
    assert snapshot(run_dir) == before                          # nothing created or changed


def test_missing_run_dir_is_zero_and_not_created(tmp_path):
    assert next_episode(tmp_path / "no_such_run") == 0
    assert not (tmp_path / "no_such_run").exists()
