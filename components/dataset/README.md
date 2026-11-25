# dataset
The simulator's episode bags -> LeRobot v3.0 training datasets (contract: [DATASET-v2.md](DATASET-v2.md)). `convert` takes
producer manifests (datagen's `manifest.json`), ports the passed episodes in manifest order through Rosetta's
`rosetta_port` and the robot's Rosetta contract (`specs/robots/<robot>/contract.yaml`), and labels each episode;
`merge` joins finished datasets with LeRobot's `aggregate_datasets`. Acceptance is the evaluator's `passed`: nothing
is re-judged, no folder is scanned, and any problem fails the whole job.

## Run
```
./revolve build dataset [--clean]       # the image (deps only) + Rosetta colcon-built into ThirdParty/rosetta/install
./revolve dataset convert --manifest M [--manifest M2 ...] [--split S] --turn K --out D
./revolve dataset merge --out D DATASET [DATASET ...]
python components/dataset/mock_manager.py [--job-dir J] convert|merge ...   # stand-in for the manager: the same job under the HTTP wrapper
```
Both run in the `revolve-dataset` image (env `REVOLVE_DATASET_IMAGE`), CPU only, offline (no network, `HF_HUB_OFFLINE=1`,
never pushed). `D` is built in `D.partial/` and moved to `D` only after every check passed. The same inputs and `D` again:
the finished dataset is kept (`"resumed": true`); a finished dataset of other inputs is removed first. `D` must be empty,
missing or a dataset of this component, `D.partial` empty, missing or a build directory of this component, and neither
may hold an input; else nothing is touched.
Exit codes: 0 ok, 1 a problem (listed; past the out check nothing is at `D`, and `D.partial/` is kept after a build), 2 a
refused command line. The last stdout line is `{"dataset", "episodes", "frames", "problems"}`, plus `"warnings"` when
there are any (also on stderr); progress lines are `PROGRESS {json}` (stage checks | port | merge | done | failed,
episodes_converted, episodes_total).

## Image
Ubuntu 26.04 + ROS 2 Lyrical (Python 3.14), like the simulator image; rosbag2 + MCAP; Rosetta's rosdep keys; from pip
(`requirements.txt`, generated from `requirements.in` + `overrides.txt`): LeRobot 0.6.1 (`lerobot[dataset]`; < 0.6.2, which
drops the async inference Rosetta's policy runner uses), CPU torch 2.11.0, torchvision 0.26.0, torchcodec 0.11.1, numpy
2.3.5 (overriding lerobot 0.6.1's numpy < 2.3, which has no cp314 wheel; 2.3.5 is Ubuntu's python3-numpy), opencv-python-headless
4.13. Local Rosetta fixes only as `ThirdParty/patches/rosetta/*.patch` (none).

## Inputs and outputs
Inputs: the manifests' passed entries (`bag`, `result`; paths relative to the manifest or absolute), the task's prompt
(`specs/tasks/<task>/task.yaml`), the robot's contract. Fails closed on a missing or unpassed result, a bag missing a
contract topic (or with no messages on it) or one rosbag2 cannot read, unknown task / robot, an entry it cannot label
(the split is the entry's `split`, else the first part of a `<split>/<scene dir>/<file>` scene path), and a message
Rosetta drops (undecodable, or without the contract's timeline: it zero-fills that stream). Merge inputs must be
finished builds of this component with no problems; the contract is theirs.
`D/`: a LeRobot v3.0 dataset (`LeRobotDataset("revolve/dataset", root=D)`), `meta/rosetta_contract.yaml` (the contract used),
`meta/revolve_dataset.json` (inputs, per-episode frames and labels, features, checks, problems). Labels: columns
`labels/{source, seed, split, scene, episode, expert, attempt, turn, manifest}` of `meta/episodes`
(`src/dataset/labels.py`, `select()` for training). Checks: frames within 6 of floor(duration_s x 30) and equal to the
bag's ticks minus warmup, features vs the contract, one task == the prompt, labels read back, episode and frame counts;
merge also needs byte-identical contracts and the same fps, features and task. An action stream whose first message
comes after the first frame (Rosetta zero-fills it until then) is a warning, recorded per episode as
`action_start_frames`; `checks.ACTION_LATE_FAILS` turns it into a problem.

## Tests
```
./revolve test dataset -- -k unit       # host only: manifests, labels, checks, the CLI's fail-closed paths, launch
./revolve test dataset                  # + real conversions in the image of every robot's simulator bags, a merge, resume, the HTTP job
```
The real tests need `./revolve build dataset`; their bags are recorded on the first run (`tests/bags.py`: the simulator
image, the scene assets and a GPU, about 5 min) into `runs/test_cache/dataset/` and reused after. Those episodes fail the
task, so their manifest lists them as passed: a test fixture only. Outputs: `runs/dataset_tests/<time>/`.
