# dataset
MCAP bags -> LeRobot v3.0 dataset with Rosetta's `rosetta_port` (ThirdParty/rosetta), and merging datasets
(previous turn + this turn's episodes) with LeRobot 0.6.1's `aggregate_datasets`. Episode labels ride along
as meta/episodes columns.

- Build: `./revolve build dataset [--clean]` -> image `revolve-dataset` (deps only: ros:jazzy-ros-base, rosdep
  keys of Rosetta's packages, `lerobot[dataset]==0.6.1` + CPU torch 2.11, revolve_common deps; 4.16 GB disk, 0.96 GB content, as automoma-lerobot) and
  Rosetta colcon-built inside it into ThirdParty/rosetta/install (git-ignored, ~12 MB, BUILD_OK = the
  submodule sha), mounted read-only at /opt/rosetta/install at runtime. Patches: ThirdParty/patches/rosetta/
  (none).
- Host API (launch.py, docker SDK, CPU, offline: HF_HUB_OFFLINE, no token, never pushed):
  `convert(task, episode_dirs, out, turn)`, `merge(task, datasets, out)`,
  `turn_dataset(task, turn_dir, turn, episode_dirs, prev_dataset=None)` -> turn_dir/dataset (turn 0) or
  turn_dir/dataset_new merged after prev_dataset into turn_dir/dataset. Each returns the job's
  {exit, wall_s, result, dir}; the job runs under revolve_common.wrapper (GET /status, /result, POST /cancel;
  status.json/result.json in the job dir, default <out>/../jobs/dataset_<kind>_<name>/).
- Entrypoint (in the image, src/dataset/entry.py): `convert --task T --turn K --out DIR EPISODE_DIR...`,
  `merge --task T --out DIR DATASET...`. Progress (progress.py): `episodes_converted`, `episodes_total`,
  `stage` (port | merge | done | failed); the final progress carries dataset, episodes, frames, problems.
- Inputs: episode dirs with `bag/` (MCAP) and `episode.json` (datagen: `datagen` block; recovery: `recovery`
  block), the task spec (prompt) and the robot's Rosetta contract (specs/robots/<r>/contract.yaml).
- Output: `<out>/` = a LeRobot v3.0 dataset (load with `LeRobotDataset("revolve/dataset", root=<out>)`) +
  meta/rosetta_contract.yaml (Rosetta's sidecar) + meta/revolve_dataset.json (inputs, lengths, labels,
  features, problems). Built in `<out>.partial/` and moved to `<out>` only after the checks pass: frames per
  episode vs the episode's sim duration (+-1, legacy rule), feature keys/shapes vs the contract, the single
  task string == task.yaml prompt, labels read back. Same inputs again: kept (resume).
- Labels (src/dataset/labels.py), columns `labels/<name>` of meta/episodes: source (demo | recovery), phase,
  suffix, seed, turn, from_rollout, episode_dir. Training-side filter:
  `labels.select(LeRobotDataset(...).meta.episodes, source="recovery")` -> indices for `--dataset.episodes`.
- Tests: `./revolve test dataset` (unit tests on stub data + real conversions; fixtures in runs/test_cache/).
