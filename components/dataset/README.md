# dataset

Turns the simulator's episode recordings (rosbag2 bags) into LeRobot v3.0 datasets for training. It sits between
datagen, which records the episodes and lists them in a manifest, and training, which reads the dataset; the manager
runs it as a job. `convert` builds one dataset from manifests and `merge` joins finished datasets.

## Inputs

| Input | From | Format |
|---|---|---|
| Producer manifest | datagen (or recovery) | `manifest.json`: task, robot and one entry per episode, `passed` set by the simulator's evaluator |
| Episode bag | the simulator, listed in the manifest | rosbag2 MCAP directory; path relative to the manifest or absolute |
| Evaluator result | the simulator, listed in the manifest | `episode_<n>.json` with `passed: true` and `duration_s` |
| Task prompt | `specs/tasks/<task>/task.yaml` | the one task string every frame carries |
| Robot contract | `specs/robots/<robot>/contract.yaml` | Rosetta contract (Rosetta: the bag-to-LeRobot library): topics, features, 30 fps |
| Datasets to merge | earlier `convert` or `merge` outputs | directories with `meta/revolve_dataset.json` and no problems |

Only entries with `passed: true` are used, in manifest order; nothing is re-judged and no folder is scanned. The
split label is the entry's `split` field, else the first part of a `<split>/<scene dir>/<file>` scene path.

## Outputs

| Output | Where | Read by |
|---|---|---|
| LeRobot v3.0 dataset | `<out>/` (`LeRobotDataset("revolve/dataset", root=<out>)`) | training |
| Episode labels | `labels/<name>` columns of `<out>/meta/episodes` (source, seed, split, scene, episode, expert, attempt, turn, manifest) | training (`labels.select`) |
| Contract used | `<out>/meta/rosetta_contract.yaml` | `merge` |
| Build record | `<out>/meta/revolve_dataset.json`: inputs, per-episode frames and labels, checks, warnings, problems | `merge`, resume, people |
| Failed build | `<out>.partial/`, kept for inspection | people |
| Result line | last stdout line: `{"dataset", "episodes", "frames", "problems"}`, plus `"warnings"` | the manager |

## Run

```
./revolve build dataset [--clean]       # the revolve-dataset image, and Rosetta built into ThirdParty/rosetta/install
./revolve dataset convert --manifest M [--manifest M2 ...] [--split S] --turn K --out D    # manifests -> dataset at D
./revolve dataset merge --out D DATASET [DATASET ...]                                     # datasets -> one, in order
python components/dataset/mock_manager.py [--job-dir J] convert|merge ...   # the same job the way the manager runs it
```

Both commands run in the `revolve-dataset` image (Ubuntu 26.04, ROS 2 Lyrical, Python 3.14, LeRobot 0.6.1, CPU torch
2.11.0, numpy 2.3.5), on CPU, with no network. The dataset is built in `D.partial/` and moved to `D` only when every
check passes. Running the same inputs again keeps the finished dataset.

Any problem fails the whole job: a result that is not passed, a bag missing a contract topic or unreadable, a message
Rosetta drops, frames per episode more than 6 away from floor(duration_s x 30) or not what the bag's timing gives, the
wrong features, task or labels. An action stream that starts after the first frame only gives a warning for
now. `D` must be empty, missing or this component's dataset, and must not hold an input.

Exit codes: 0 the dataset is at `D`; 1 a problem, listed on stderr and in the result line (nothing is left at `D`
once the `D` check has passed); 2 a bad command line.

## Test

```
./revolve test dataset -- -k unit       # host only, about 3 s: manifests, labels, checks, the out rule, launch, the CLI's failures
./revolve test dataset                  # plus real conversions in the image, about 10 min
```

The full run needs `./revolve build dataset` and Docker. It converts real bags of all three robots, splits, merges,
resumes, refuses broken inputs and runs the HTTP job. Its bags are recorded on the first run with the simulator image,
the scene assets and a GPU (about 5 min more), then reused from `runs/test_cache/dataset/`. Results go to
`runs/dataset_tests/<time>/`.

## Files

- `src/dataset/cli.py`: the entrypoint, convert and merge.
- `src/dataset/checks.py`: the dataset checks, with the frame tolerance.
- `launch.py`: the docker command and the manager's calls (`convert`, `merge`, `start`, `wait`).
- `build.sh`: the image and the Rosetta build; local Rosetta fixes go in `ThirdParty/patches/rosetta/`.

Contract: [DATASET-v2.md](DATASET-v2.md).
