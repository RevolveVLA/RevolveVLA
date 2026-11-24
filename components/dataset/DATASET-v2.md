# Dataset v2: converting the new simulator's episode bags into LeRobot datasets

This document is the complete hand-over for building **dataset v2**, the component that turns the simulator's
recorded episodes into LeRobot v3.0 training datasets, on top of the rewritten simulator. It is self-contained:
everything the dataset component may rely on is specified here, with the exact names, paths, topics, message types and
decisions as they are on this branch. If something you need is not here, it is not part of the contract: ask for it
rather than reaching around the interfaces (see [Roles and boundaries](#3-roles-and-boundaries)).

Contents

1. [Purpose and status](#1-purpose-and-status)
2. [What changed, and why the old component cannot convert the new bags](#2-what-changed-and-why-the-old-component-cannot-convert-the-new-bags)
3. [Roles and boundaries](#3-roles-and-boundaries)
4. [The image: ROS 2 Lyrical and Rosetta built against it](#4-the-image-ros-2-lyrical-and-rosetta-built-against-it)
5. [The contract](#5-the-contract)
6. [Input: the producer's manifest](#6-input-the-producers-manifest)
7. [CLI and job wrapper](#7-cli-and-job-wrapper)
8. [Output, checks and result](#8-output-checks-and-result)
9. [Merge](#9-merge)
10. [The bags you read](#10-the-bags-you-read)
11. [Getting real test bags](#11-getting-real-test-bags)
12. [Caveats and lessons learned](#12-caveats-and-lessons-learned)
13. [Replacing the old dataset code](#13-replacing-the-old-dataset-code)
14. [Definition of done for dataset v2](#14-definition-of-done-for-dataset-v2)
15. [File map](#15-file-map)

---

## 1. Purpose and status

The dataset component converts **accepted episodes** (episode bags the simulator recorded and its evaluator passed)
into one **LeRobot v3.0 dataset** per call, through **Rosetta** (`ThirdParty/rosetta`, its offline porter
`rosetta_port`) and the robot's **Rosetta contract** (`specs/robots/<r>/contract.yaml`). It also merges finished
datasets (LeRobot's `aggregate_datasets`). Training reads the result with `LeRobotDataset`; a policy runner
(Rosetta's `policy_runner_node`) uses the same contract live.

Status of the pipeline on this branch:

| Part | Status |
|---|---|
| Simulator (`components/simulator`) | Rewritten and tested. Records one bag per episode with every topic the contract needs, including `/task/prompt`. |
| Robot contracts (`specs/robots/<r>/contract.yaml`) | **Written and checked**: `./revolve validate` (against the robot and the simulator's topic list) and the simulator suite (Rosetta's own loader and decoders on real bags). |
| Datagen (`components/datagen`, `components/datagen/DATAGEN-v2.md`) | Datagen v2: demonstrations on the new simulator, writes `manifest.json`. Its arm/gripper commands must stream (section 5.4). |
| Dataset (`components/dataset`) | **Old code, broken** (section 2). It is yours to replace with v2 (section 13). |
| Recovery, training, evaluation, manager | Not adapted yet. Recovery and evaluation will write the same manifest schema (section 6). |

## 2. What changed, and why the old component cannot convert the new bags

| Old (what `components/dataset/src/dataset/` expects) | New |
|---|---|
| Image `ros:jazzy-ros-base` (Ubuntu 24.04, Python 3.12); Rosetta colcon-built inside it into `ThirdParty/rosetta/install`. | ROS 2 **Lyrical** on Ubuntu 26.04 (Python 3.14), like the simulator (decision DS8, section 4). The existing `ThirdParty/rosetta/install` is a Jazzy / Python 3.12 build: rebuild it in the new image. |
| Input: a list of episode dirs, each with `bag/` (MCAP) and `episode.json` (`steps`, `control_dt`, a `datagen` or `recovery` block). | Input: the producer's **`manifest.json`** (section 6). Per episode the simulator writes `episode_<n>.json` (the evaluator's result: `duration_s`, `passed`, phases) and a bag directory `episode_<n>_bag/`. There is no `episode.json`, no `bag/` subdirectory, no `steps x control_dt`. |
| Episode length check: `steps x control_dt`. | `duration_s` of the result JSON (section 8.3). |
| `specs/robots/<r>/contract.yaml` in the old format, generated from the old robot specs; the old schema's `robot.contract` field. | The old contracts were deleted with the old robot specs. The **new contracts are hand-written in Rosetta's own format**, next to the robot's other interfaces (decision DS2, section 5). `revolve_common.specs` has no `contract` field: the path is `specs/robots/<robot>/contract.yaml`. |
| The robot and its topics: the old custom simulator (gz Python systems, a custom torque controller). | Upstream ROS 2 stack: `gz_ros2_control`, `joint_trajectory_controller` for arm and gripper, `omni_wheel_drive_controller` for the base, ros_gz bridges, image_transport JPEG cameras (section 10). |
| Action = whatever the old replay wrote. | Action = the streamed controller commands `/<jtc>/joint_trajectory` + `cmd_vel` (decision DS3, section 5.4). |
| `launch.py` imports `components/simulator/launch.py` (the old simulator HTTP job). | That file is gone: the old launcher fails at import. |
| `revolve_common.specs.load(task).robot` and `r.robot.contract` (old schema). | `specs.resolve(root, task, robot=...)`; tasks no longer name a robot. |

The old code's ideas carry over well (labels as `meta/episodes` columns, build in `<out>.partial/`, checks before the
move, resume on identical inputs, the merge through `aggregate_datasets`), but its inputs, image and contract do not.

## 3. Roles and boundaries

* **Simulator**: the only recorder. Writes per episode `episode_<n>.json` (written last: once it exists, the
  episode's files are complete) and `episode_<n>_bag/`. Never changed by the dataset component.
* **Producers** (datagen now; recovery and evaluation later): run episodes against the simulator and write a
  **`manifest.json`** listing every attempted episode with `passed` copied from the evaluator's result (section 6).
* **Dataset component** (you): given manifests, converts the **passed** episodes, in manifest order, into one LeRobot
  dataset with labels; or merges given datasets. Two separate operations with explicit inputs (decision DS9):
  * `convert`: manifests -> one dataset;
  * `merge`: datasets -> one dataset.

Rules:

1. **Never re-judge.** Acceptance is the evaluator's `passed`, as copied into the manifest. Do not filter by your own
   criteria, re-run checks on the physics, or read the producer's private files (`datagen.json`, `state.json`, ...).
2. **No folder scanning.** The manifest lists the episodes. Do not glob for bags.
3. **No turn logic.** What to merge, when to retrain, and which turn an episode belongs to are the manager's / the
   training's business. You get `--turn K` as a label value and explicit dataset paths to merge.
4. **Fail closed.** A passed entry whose bag or result is missing, or whose bag lacks a contract topic, fails the
   conversion with a problem list. No silent skips (a skipped episode would also shift every label after it).
5. **Never modify inputs**: bags, results, manifests and the contract stay as they are.
6. **Offline.** Never push to the Hugging Face Hub, never download at run time.
7. **The contract is the only robot interface.** Features, fps, cameras, units and the task channel come from
   `specs/robots/<r>/contract.yaml` through Rosetta. Do not hard-code topics or joint names in the converter.

## 4. The image: ROS 2 Lyrical and Rosetta built against it

Decision DS8: **match the ROS version across the project**. The dataset image moves to **ROS 2 Lyrical on Ubuntu
26.04**, like the simulator image (`components/simulator/Dockerfile`), and so will any later ROS-using component
(policy runner / serve, evaluation).

What this means:

* **Base**: `ubuntu:26.04` + the ROS 2 apt source + `ros-lyrical-ros-base` + rosbag2 with the MCAP storage plugin
  (`ros-lyrical-rosbag2-storage-mcap`; the bags are MCAP with zstd chunks) + the message packages of the contract's
  types (`sensor_msgs`, `nav_msgs`, `geometry_msgs`, `trajectory_msgs`, `std_msgs`; also `tf2_msgs`,
  `control_msgs` if you read diagnostics). Copy the apt-source lines from the simulator Dockerfile.
* **OpenCV for Python** (`cv2`): Rosetta decodes `sensor_msgs/CompressedImage` with OpenCV and fails without it.
  The simulator image has `cv2` 4.10 on Python 3.14.
* **ffmpeg** (LeRobot's video encoding, torchcodec).
* **Rosetta is colcon-built against Lyrical**, not taken from its pixi environment (Rosetta upstream targets Jazzy via
  RoboStack; `ThirdParty/rosetta/pixi.toml`). Keep the existing pattern: dependencies only in the image (rosdep over
  `rosetta/`, `robots/ros/`, `policies/lerobot/` through a build-time bind mount), and `./revolve build dataset`
  colcon-builds Rosetta inside the image into `ThirdParty/rosetta/install` (git-ignored, mounted read-only at runtime at
  the path it was built at). Packages up to `lerobot_rosetta`: `rosetta`, `rosetta_interfaces` (messages, needs
  rosidl), `rosetta_ros` (`rosetta_port`), `lerobot_robot_rosetta`, `lerobot_rosetta`. The `ros2` robot interface
  is registered through a Python entry point (`rosetta.robot_interfaces`), so Rosetta must be installed (colcon or
  pip), not just put on `PYTHONPATH`.
* **Local fixes to Rosetta only as patches** in `ThirdParty/patches/rosetta/*.patch`, applied to a copy of the
  sources at build time (the old `build.sh` already does this); never edit the submodule.
* **LeRobot 0.6.1 + torch on the image's Python (3.14) must be verified**: pin as Rosetta does
  (`lerobot<0.6.2`: 0.6.2 removes the async inference Rosetta's policy runner uses; the old image used
  `lerobot[dataset]==0.6.1`, CPU `torch==2.11.0`, `torchvision==0.26.0`, `torchcodec==0.11.1`). Check that wheels
  exist for cp314 and that `import lerobot, torch` and a `LeRobotDataset` load work in the image; if a pin has no
  cp314 wheel, choose the nearest working one and record why.
* **Keep a test that ports a real simulator-recorded bag** (section 11) in `./revolve test dataset`.
* CPU only. `HF_HUB_OFFLINE=1`, no token, `HF_HOME` inside the container.

What has been verified on this branch (in the simulator image, not yet in a dataset image): Rosetta's two pure-Python
packages `rosetta` and `rosetta_ros` install with `pip install --no-deps --no-build-isolation` from the submodule on
Lyrical / Python 3.14 (OpenCV, numpy, PyYAML, rclpy and the message packages from apt), `load_contract` validates all
three robot contracts, and Rosetta's decoders decode every contract source of real bags at the contract's widths and
image size (`components/simulator/ros/revolve_sim/test/check_rosetta.py`). `rosetta_interfaces`, the LeRobot writer,
LeRobot itself and torch on Lyrical are **not** verified yet.

## 5. The contract

### 5.1 Where the contracts are

| Robot | Contract | State / action width |
|---|---|---|
| `mobile_panda` | `specs/robots/mobile_panda/contract.yaml` | 11 / 11 |
| `mobile_so101` | `specs/robots/mobile_so101/contract.yaml` | 9 / 9 |
| `mobile_so101_xl` | `specs/robots/mobile_so101_xl/contract.yaml` | 9 / 9 (same joints and controllers as `mobile_so101`, the arm at scale 2) |

Hand-written in Rosetta's format (`ThirdParty/rosetta/docs/reference/contract.md`, message types and operators in
`docs/reference/message-types.md`, the porter in `docs/reference/rosetta-port.md`). The dataset component and the
policy consume them; nothing generates them.

### 5.2 mobile_panda's contract, in full

```yaml
# mobile_panda's Rosetta contract (ThirdParty/rosetta, docs/reference/contract.md): how the dataset component turns
# the simulator's episode bags into LeRobot frames, and how a policy runner feeds and commands the robot. Every topic
# is one of the simulator's recorded topics (components/simulator/ros/revolve_sim/config/record.yaml); `./revolve
# validate` checks it against this robot. Format and choices: components/dataset/DATASET-v2.md.
# Timelines: observations on `header` (the sensor's sim-time stamp); actions on `receive` (the bag's sim-time receive
# stamp), because a streamed JointTrajectory may carry a zero stamp ("start now"), which the header timeline drops.
robot_type: mobile_panda
robot_interface: ros2
fps: 30

x-image: &image
  type: sensor_msgs/msg/CompressedImage
  qos: {reliability: best_effort, history: keep_last, depth: 10}   # matches any publisher

observations:
  observation.images.wrist:
    channel: {topic: /wrist_cam/image/compressed, <<: *image}
    align: {strategy: hold, timeline: header}
    apply: [resize: [480, 480]]                                     # the native size
  observation.images.overhead_left:
    channel: {topic: /overhead_left_cam/image/compressed, <<: *image}
    align: {strategy: hold, timeline: header}
    apply: [resize: [480, 480]]
  observation.images.overhead_right:
    channel: {topic: /overhead_right_cam/image/compressed, <<: *image}
    align: {strategy: hold, timeline: header}
    apply: [resize: [480, 480]]
  # 11 wide: base twist (wheel odometry, robot frame) [m/s, m/s, rad/s], arm [rad], finger [m]
  observation.state:
    - channel: {topic: /omni_wheel_drive_controller/odom, type: nav_msgs/msg/Odometry}
      align: {strategy: hold, timeline: header}
      select: [twist.twist.linear.x, twist.twist.linear.y, twist.twist.angular.z]
    - channel: {topic: /joint_states, type: sensor_msgs/msg/JointState}
      align: {strategy: hold, timeline: header}
      select: [position.panda_joint1, position.panda_joint2, position.panda_joint3, position.panda_joint4,
               position.panda_joint5, position.panda_joint6, position.panda_joint7, position.panda_finger_joint1]

actions:
  # 11 wide, the state's units per dimension: base twist, arm targets, finger target (single-point trajectories)
  action:
    # zeros = stop; the controller also stops after its 0.5 s cmd_vel_timeout
    - channel: {topic: /omni_wheel_drive_controller/cmd_vel, type: geometry_msgs/msg/TwistStamped, safety: zeros}
      align: {strategy: hold, timeline: receive}
      select: [twist.linear.x, twist.linear.y, twist.angular.z]
    # none: the trajectory controller holds its last target by itself; zeros would drive the arm to the all-zero
    # pose, and hold sends zeros before the first command (Rosetta's warning)
    - channel: {topic: /panda_arm_controller/joint_trajectory, type: trajectory_msgs/msg/JointTrajectory,
                safety: none}
      align: {strategy: hold, timeline: receive}
      select: [position.panda_joint1, position.panda_joint2, position.panda_joint3, position.panda_joint4,
               position.panda_joint5, position.panda_joint6, position.panda_joint7]
    - channel: {topic: /panda_hand_controller/joint_trajectory, type: trajectory_msgs/msg/JointTrajectory,
                safety: none}
      align: {strategy: hold, timeline: receive}
      select: [position.panda_finger_joint1]

tasks:
  task:
    channel: {topic: /task/prompt, type: std_msgs/msg/String,
              qos: {reliability: reliable, durability: transient_local, history: keep_last, depth: 1}}
```

The SO-101 contracts differ only in `robot_type`, the joint names (arm `shoulder_pan`, `shoulder_lift`,
`elbow_flex`, `wrist_flex`, `wrist_roll`; gripper `gripper` [rad, the moving jaw]) and the controllers
(`arm_controller`, `gripper_controller`).

### 5.3 The decisions behind it

| Decision | What the contract says |
|---|---|
| DS2 location and checks | Hand-written per robot at `specs/robots/<r>/contract.yaml`. `./revolve validate` (`revolve_common.specs.check_contract`, no Rosetta needed) checks: `robot_type` = the robot, `robot_interface: ros2`, `fps: 30`, exactly the three image keys, a `tasks` channel on `/task/prompt`, every topic is one the simulator records (`record.yaml` filled in for the robot: its JTC and omni controller names, its cameras), every joint selected from `/joint_states` or a `joint_trajectory` topic is a controller joint, each `/<jtc>/joint_trajectory` action selects exactly its controller's joints in controller order, every JTC has an action, and the `/joint_states` state joints equal the action joints in order. |
| DS3 action topics | Arm and gripper: each controller's `/<jtc>/joint_trajectory` (`trajectory_msgs/JointTrajectory`, Rosetta decodes the first point and encodes a single point). Base: `/omni_wheel_drive_controller/cmd_vel` (`geometry_msgs/TwistStamped`). One topic per channel, used both ways: decoded when porting, encoded by the policy. |
| DS4 state and action | `observation.state` = base twist `twist.twist.{linear.x, linear.y, angular.z}` from `/omni_wheel_drive_controller/odom` (wheel odometry, robot frame) + the arm joint positions (controller order) + the gripper joint position (Panda `panda_finger_joint1` [m, per finger, 0 closed .. 0.04 open]; SO-101 `gripper` [rad]) from `/joint_states`. `action` = cmd_vel `twist.{linear.x, linear.y, angular.z}` + arm targets + gripper target, the same units per dimension. No ground truth, contacts or evaluator values in `observation.*` (they may only appear in record-only sections, `info` / `complementary_data`, none today). fps 30, `align: hold`. |
| DS5 images | Robot-agnostic keys `observation.images.{wrist, overhead_left, overhead_right}`, mapped to `/wrist_cam/image/compressed`, `/overhead_left_cam/image/compressed`, `/overhead_right_cam/image/compressed` (JPEG), `resize: [480, 480]` (the native size; Rosetta requires a fixed-size operator on image keys). Stored as LeRobot video with the porter's default codec. The contract stays a subset of the recorded topics: no new camera or depth without changing the robot xacro and `record.yaml`. |
| DS7 prompt | A `tasks` channel on `/task/prompt` (`std_msgs/String`, transient local), published by the simulator from `task.yaml`'s `prompt`. |
| Timelines (this branch) | Observations on `header` (the sensor's sim-time stamp). **Actions on `receive`** (the bag's receive time, sim time): a streamed `JointTrajectory` may carry a zero header stamp (the controller's "start now"; datagen's streamer and the simulator's contract test send zero stamps), and Rosetta's header timeline drops every message stamped (0, 0), which would leave the arm and gripper action zero-filled. Measured: the bag's receive time trails the header stamps by 1-5 ms on the 1 kHz streams. |
| Safety (live only) | Base `zeros` (zero twist = stop; the controller also stops after its 0.5 s `cmd_vel_timeout`). Arm and gripper `none`: the trajectory controller holds its last target by itself; `zeros` would drive the arm to the all-zero joint pose (Rosetta's own warning), and `hold` sends zeros before the first command (which Rosetta's LeRobot plugin triggers at every episode start). |

### 5.4 What it means for the producers (DS3)

Because a Rosetta action channel is one topic, **datagen must stream single-point targets on `/<jtc>/joint_trajectory`
at 30 Hz or faster, exactly like the policy, and `cmd_vel` continuously** (zero twists while the base stands: the
porter holds the last `cmd_vel` at any age). MoveIt may still plan; the expert time-samples the plan and streams it.
**FollowJointTrajectory goals leave no action in the bag** and are not acceptable for demonstrations. The JTCs'
`/<jtc>/controller_state` (1 kHz, `reference` = the setpoint they track) stays recorded as a diagnostic only. This
is in `components/datagen/DATAGEN-v2.md` (its "Change since the first version" note and sections 5 and 7.4). For
you: a passed episode whose bag lacks a contract topic (for example a `joint_trajectory` topic because the expert
used FollowJointTrajectory) fails the conversion (section 8.3); never fill it from `controller_state`.

### 5.5 Frame layout and feature names

Rosetta concatenates multi-source keys in list order and prefixes feature names with the first topic segment at
which the sources' topics differ. For `mobile_panda` (as `load_contract` resolves it):

| Key | Width | Feature names (in order) |
|---|---|---|
| `observation.images.wrist`, `.overhead_left`, `.overhead_right` | 480 x 480 x 3 (video) | |
| `observation.state` | 11 (float64) | `omni_wheel_drive_controller.twist.twist.linear.x`, `...linear.y`, `...angular.z`, `joint_states.position.panda_joint1` .. `panda_joint7`, `joint_states.position.panda_finger_joint1` |
| `action` | 11 (float64) | `omni_wheel_drive_controller.twist.linear.x`, `...linear.y`, `...angular.z`, `panda_arm_controller.position.panda_joint1` .. `panda_joint7`, `panda_hand_controller.position.panda_finger_joint1` |

Plus the porter's `is_first`, `is_last`, `is_terminal` and LeRobot's `timestamp`, `frame_index`, `episode_index`,
`index`, `task_index`. SO-101 robots: 9 = 3 + 5 + 1 with `arm_controller` / `gripper_controller` prefixes.

## 6. Input: the producer's manifest

Decision DS6. The input of `convert` is one or more **`manifest.json`** files in the schema datagen writes
(`components/datagen/DATAGEN-v2.md` section 8.4; `components/datagen/src/datagen/core/manifest.py`):

```json
{
 "task": "open_fridge_door",
 "robot": "mobile_panda",
 "created": "2026-11-22T10:15:00Z",
 "episodes": [
  {"seed": 7, "scene": "train/scene_7/scene.yaml", "robot": "mobile_panda", "task": "open_fridge_door", "episode": 0,
   "result": "train/scene_7/episode_0.json", "bag": "train/scene_7/episode_0_bag", "passed": false,
   "datagen": {"expert": "moveit_ompl_pilz_v1", "attempt": 0, "wall_s": 41.2}},
  {"seed": 7, "scene": "train/scene_7/scene.yaml", "robot": "mobile_panda", "task": "open_fridge_door", "episode": 1,
   "result": "train/scene_7/episode_1.json", "bag": "train/scene_7/episode_1_bag", "passed": true,
   "datagen": {"expert": "moveit_ompl_pilz_v1", "attempt": 1, "wall_s": 38.7}}
 ],
 "counts": {"scenes": 1, "attempted": 2, "passed": 1}
}
```

Rules:

* **Only `passed: true` entries**, **in manifest order** (several manifests: in the order given, each in its own
  order). Paths are relative to the manifest's directory (absolute paths are allowed: accept both).
* **Never re-judge** (`passed` is the evaluator's, copied by the producer), never scan folders, never read
  `datagen.json` / `state.json`.
* Per passed entry you read exactly: `entry.bag` (the rosbag2 directory) and `entry.result` (the evaluator's result
  JSON: `duration_s` for the frame check, `passed` must be true there too, `task` / `robot`).
* Every manifest must have the same `task` and `robot`, and every entry the manifest's `task` / `robot`. The robot
  selects the contract (`specs/robots/<robot>/contract.yaml`), the task the prompt (`specs/tasks/<task>/task.yaml`).
* `--split S`: keep only the entries of split S. The split of an entry is its `split` field if the producer writes
  one, else the first path component of `entry.scene` (datagen's layout `<split>/scene_<seed>/`). Ask the datagen
  owner to add an explicit `split` field; until then derive it as described and record it in the labels.
* **Fail closed**: a passed entry with a missing bag directory (`metadata.yaml`) or result file, a result that is not
  `passed`, an unknown robot/task, or a bag lacking a contract topic fails the whole conversion with a problem list
  (section 8.4). No silent skips.

Labels (one row per episode in `meta/episodes`, columns `labels/<name>`, fixed types, never null: LeRobot's metadata
parquet writer keeps the first row's schema):

| Label | Type | From |
|---|---|---|
| `source` | str | `demo` for an entry with a `datagen` block, `recovery` for one with a `recovery` block (recovery's manifests, later); anything else fails closed |
| `seed` | int | `entry.seed` |
| `split` | str | as above |
| `scene` | str | `entry.scene` (as written in the manifest) |
| `episode` | int | `entry.episode` (the simulator's episode number in its run dir) |
| `expert` | str | `entry.datagen.expert` (recovery: its equivalent, or `""`) |
| `attempt` | int | `entry.datagen.attempt` (recovery: its equivalent, or -1) |
| `turn` | int | `--turn K` |
| `manifest` | str | the manifest's path as given on the command line |

Recovery and evaluation will write manifests with the same schema (an entry block named after the producer instead
of `datagen`); keep the label extraction in one small, unit-tested function.

## 7. CLI and job wrapper

Decision DS10 (with DS9's arguments):

```
./revolve dataset convert --manifest M [--manifest M2 ...] [--split S] --turn K --out D
./revolve dataset merge --out D DATASET [DATASET ...]
```

* `./revolve dataset ...` runs the job in the dataset image (docker, CPU, no network needed): the CLI is how you test
  the component in isolation, without the manager. Add the subcommand to `revolve` (next to `build` / `test` /
  `sim`), mounting the repo read-only at `/repo`, `runs/` (and any `--out` / manifest location outside the repo,
  at its own path) writable, and `ThirdParty/rosetta/install` read-only at the path it was built at.
* `convert`: one dataset from the passed entries of the manifests, in order. `merge`: LeRobot's `aggregate_datasets`
  of the given datasets in the given order (section 9).
* The **HTTP job wrapper stays the outer contract** for the future manager: `python -m revolve_common.wrapper
  --job-dir DIR [--parser mod:fn] -- <your entrypoint> ...` (job.json in; status.json, result.json, stdout.log out;
  GET /status, GET /result, POST /cancel). Print progress as `PROGRESS {json}` lines (e.g. `stage` port | checks |
  merge | done | failed, `episodes_converted`, `episodes_total`). Keep `launch.py`-style host helpers if useful, but
  the CLI is the tested path.
* Exit 0 only when the dataset is at `<out>` and every check passed; any problem -> non-zero exit and nothing at
  `<out>` (section 8).

## 8. Output, checks and result

### 8.1 Layout

```
<out>/                          LeRobot v3.0 dataset (LeRobotDataset(<repo id>, root=<out>) loads it)
  meta/info.json                fps 30, robot_type, features (from the contract)
  meta/stats.json
  meta/tasks.parquet            one task: the prompt
  meta/episodes/...parquet      per episode: LeRobot's columns + labels/<name> (section 6)
  meta/rosetta_contract.yaml    the contract used (the porter's copy of --contract)
  meta/revolve_dataset.json     this component's summary (below)
  data/chunk-*/file-*.parquet
  videos/observation.images.<cam>/chunk-*/file-*.mp4
```

Built in **`<out>.partial/`** and moved to `<out>` only after every check passed. Use a fixed, never-pushed repo id
(the old code used `revolve/dataset`).

### 8.2 meta/revolve_dataset.json

```json
{
 "kind": "convert",
 "inputs": {"manifests": ["runs/fridge/datagen/manifest.json"], "split": "train", "turn": 0},
 "contract": "specs/robots/mobile_panda/contract.yaml",
 "episodes": 2, "frames": 1203,
 "per_episode": [{"bag": "...", "result": "...", "duration_s": 20.004, "frames": 601, "labels": {"source": "demo", "...": "..."}}],
 "features": {"observation.state": [11], "action": [11], "observation.images.wrist": [480, 480, 3], "...": "..."},
 "checks": {"frames": "ok", "features": "ok", "task": "ok", "labels": "ok", "episodes": "ok"},
 "problems": []
}
```

The exact shape is yours; it must hold the inputs (for resume), per-episode lengths and labels, the features and the
check results.

### 8.3 Checks (all before the move)

* **Frames per episode** vs the evaluator's duration: decision DS6 says `floor(duration_s x 30)` +- 1. **Measured on
  this branch, Rosetta's port does not meet +- 1**: the bag starts 16-83 ms after the episode start (the recorder's
  record service call) and ends 38-109 ms after its end (the result is published, then 0.1 s wall, then the recorder
  stops), and the porter skips the ticks before every observation stream has a sample (1-3 ticks, the cameras come
  last). Expected frames from six real bags: -3, 0, +1, +2, +1, +2 relative to `floor(duration_s x 30)`
  (the -3 on a heavily loaded host). Proposal (pending
  the project owner's confirmation, record it in your PR): `abs(frames - floor(duration_s x 30)) <= 6` (0.2 s), plus
  an exact internal check that the episode's frame count equals the porter's tick count over the bag's span minus its
  warmup ticks.
* **Feature keys and shapes** vs the contract (images `(480, 480, 3)`, `observation.state` and `action` the contract's
  widths; the porter's and LeRobot's bookkeeping keys allowed).
* **Task string == task.yaml `prompt`** for every frame, and exactly one task in `meta/tasks`.
* **Labels read back** from `meta/episodes` equal the expected labels, in order.
* **Episode count** == the number of passed manifest entries converted (after `--split`).
* fps 30, `codebase_version` v3.0, sum of episode lengths == `total_frames`.
* Fail closed on the inputs (section 6), before porting: every passed entry's result and bag exist, the result
  says `passed: true`, and the bag holds every contract topic (read `metadata.yaml`'s `topics_with_message_count`:
  a topic with zero messages counts as missing). Rosetta itself only zero-fills a missing action topic with a
  warning, and skips a failed episode and carries on: the converter must turn both into problems.

### 8.4 Result JSON

The last line / `result.json` content:

```json
{"dataset": "runs/fridge/dataset", "episodes": 2, "frames": 1203, "problems": []}
```

Any problem -> non-zero exit, `problems` lists them (one string each, naming the episode and the check), nothing at
`<out>` (keep `<out>.partial/` for inspection).

### 8.5 Resume

Same inputs + same `<out>`: an existing finished dataset whose `meta/revolve_dataset.json` lists exactly these inputs
is kept (exit 0, `"resumed": true`). Anything else: rebuild from scratch in `<out>.partial/` (LeRobot's writer refuses
an existing directory: remove a stale `.partial` first).

## 9. Merge

Decision DS9: `merge --out D DATASET...` = LeRobot 0.6.1 `aggregate_datasets` in the given order (it copies each
source's `meta/episodes` rows whole, so the `labels/*` columns come along; it shifts indices and aggregates stats).

* **Fail closed unless** every input has a **byte-identical `meta/rosetta_contract.yaml`**, the same fps, the same
  features (keys, shapes, dtypes) and the same task.
* Labels are preserved (checked after the merge: the merged `meta/episodes` labels == the inputs' labels
  concatenated in order; lengths likewise).
* Copy `meta/rosetta_contract.yaml` from the first input; write `meta/revolve_dataset.json` (kind `merge`, inputs).
* Same resume, `.partial`, checks and result JSON as `convert`.
* What to merge (e.g. the previous turn's dataset + this turn's) is decided by the caller, never here.

## 10. The bags you read

### 10.1 Layout (per producer run dir, the simulator's `--out`)

```
<run dir>/
  episode_<n>.json            the evaluator's result, written LAST: once it exists, the episode's files are complete
  episode_<n>_gz.log          the gz server log of the episode (not for you)
  episode_<n>_bag/            one rosbag2 directory per episode
    metadata.yaml
    0_episode_<n>_bag_<date-time>.mcap
```

Bags: MCAP, zstd chunk compression (storage preset `zstd_fast`), **sim time** throughout (receive times and header
stamps). Sim time restarts at 1 s after every simulator reset, so times are comparable within one bag only. Read
them with `rosbag2_py` (what Rosetta's porter uses) or any MCAP reader.

### 10.2 Topics (`components/simulator/ros/revolve_sim/config/record.yaml`)

| Topic | Type | Rate | For the contract |
|---|---|---|---|
| `/<camera>/image/compressed` (`wrist_cam`, `overhead_left_cam`, `overhead_right_cam`) | `sensor_msgs/CompressedImage` | 30 Hz | Images. JPEG, `format` `"rgb8; jpeg compressed bgr8"`, 480 x 480. Only with `--record images` (datagen's setting; the simulator's own tests mostly record `no-images`). |
| `/joint_states` | `sensor_msgs/JointState` | 1 kHz | State (arm + gripper positions; also wheels and other joints, unused). |
| `/omni_wheel_drive_controller/odom` | `nav_msgs/Odometry` | 1 kHz | State (base twist, robot frame). |
| `/<jtc>/joint_trajectory` | `trajectory_msgs/JointTrajectory` | as streamed (>= 30 Hz) | Action (first point's positions). Header stamp may be zero. |
| `/omni_wheel_drive_controller/cmd_vel` | `geometry_msgs/TwistStamped` | as streamed | Action. |
| `/task/prompt` | `std_msgs/String` | once per bag | Task (transient local). The same string is in `metadata.yaml` under `custom_data` `lerobot.operator_prompt`. |
| `/<jtc>/controller_state` | `control_msgs/JointTrajectoryControllerState` | 1 kHz | Diagnostic only. |
| `/clock`, `/tf`, `/tf_static`, `/robot_description` | | 1 kHz / ~1.1 kHz / once / once | Not in the contract. `/tf_static` and `/robot_description` are in every bag once. |
| `/model/<m>/pose`, `/world/<w>/model/<m>/joint_state`, contacts, `/<camera>/camera_info`, `/evaluator/state`, `/evaluator/result` | | | Ground truth / evaluator: never in `observation.*` (DS4). |

Topics that only carry messages on demand (contacts, `joint_trajectory`, `cmd_vel`) are absent from a bag in which
they never had a publisher. For a passed demonstration, the contract's action topics must be present with messages
(section 8.3).

Measured (simulator contract test, 20 s episodes, `mobile_panda`, images): ~602 frames per camera, ~20 000
`/joint_states` and odometry messages, the streamed topics at 30 Hz while streaming, `/task/prompt` once. Bag sizes:
`--record images` ~6.8 MB/s (~135 MB per 20 s episode), `no-images` ~0.76 MB/s (images dominate).

### 10.3 Timing details that matter for porting

* Rosetta's ticks start at the bag's first message and are 1/30 s apart; ticks before every observation source has a
  sample are skipped (the cameras' first frames arrive 20-60 ms after the bag start).
* `/task/prompt` is delivered when the recording subscribes, i.e. at about the same time as the first camera frame:
  in one measured bag the prompt arrived 1 ms after the last observation stream's first message. The porter gives
  frames before the first task message the bag metadata's `lerobot.operator_prompt`, which the simulator writes with
  the same prompt, so every frame gets the prompt. Keep the task check anyway (section 8.3).
* Observations use header stamps: the cameras' stamps trail the bag time by 5-40 ms (render time), the 1 kHz streams
  by 1-5 ms.

## 11. Getting real test bags

* **The simulator's contract test** records two `mobile_panda` episodes with images and every contract topic:
  `./revolve test simulator -- -k "contract and not rosetta_contracts"` (needs `./revolve build simulator`, the scene
  assets from `./revolve assets` and a GPU; ~80 s). It deletes its run dir `runs/contract_<pid>/` when it passes and
  keeps it when it fails. To keep the bags, run the same flow by hand (below) or copy the run dir before the test's
  last line.
* **By hand**, with any scripted client that streams like the policy (the simplest is the contract test's
  `components/simulator/ros/revolve_sim/test/check_contract.py`, run in a client container as `client()` in
  `test_contract.py` does):

  ```
  ./revolve sim --scene kitchen_partnet --robot mobile_panda --task open_fridge_door --no-autostart \
      --record images --out runs/dataset_bags --name ds-sim --domain-id 120
  # client on ROS_DOMAIN_ID=120: wait for /evaluator/state "ready", call /evaluator/start, stream
  # /panda_arm_controller/joint_trajectory, /panda_hand_controller/joint_trajectory and cmd_vel at 30 Hz,
  # wait for runs/dataset_bags/episode_<n>.json
  docker rm -f ds-sim
  ```

  Such episodes fail the task (`passed: false`); for converter tests write a small test manifest that lists them as
  passed (test fixture only, never in production).
* **Datagen v2** produces real passed demonstrations with a manifest:
  `./revolve datagen specs/datagen/smoke_v2.yaml` (one scene instance, one accepted episode; `components/datagen/
  README.md`). Use it once the datagen expert streams its arm/gripper commands (section 5.4); bags made with
  FollowJointTrajectory goals lack the action topics and must fail your conversion.
* Never commit bags, datasets or renders (they show licence-bound assets, section 12). Cache fixtures under `runs/`
  (git-ignored), generated on the first test run.

## 12. Caveats and lessons learned

* **`controller_state` is a 1 kHz diagnostic**, not the action. Do not build the action from it, and do not decode it
  with Rosetta (Rosetta has no decoder for `control_msgs/JointTrajectoryControllerState`; registering one would be
  custom code for a diagnostic).
* **JPEG decode needs OpenCV** in the image (`cv2`). Without it every image fails to decode and the episode fails
  with "no frames emitted".
* **Header stamps of zero are dropped** by Rosetta's `header` timeline; that is why the contract's actions use
  `receive` (section 5.3). If you ever change a timeline, re-run the simulator's Rosetta test.
* **`hold` holds at any age**: a stale `cmd_vel` stays the action until the next message. The producers stream
  continuously (section 5.4); do not add your own timeout logic.
* **The prompt**: the task string per frame comes from `/task/prompt`, else from the bag metadata (section 10.3).
  Check it equals `task.yaml`'s prompt; never substitute another prompt.
* **Labels** must have fixed types and no nulls (LeRobot keeps the first row's parquet schema).
* **Rosetta's porter skips failed episodes** and logs `Completed: <ok>/<total>`: compare the converted count with the
  input count and fail on any difference (the old `port.py` wrapped `LeRobotDatasetMetadata.save_episode` to count
  episodes and attach labels; that technique still works with Rosetta at the pinned commit).
* **Rosetta takes every bag under `--raw-dir`, in sorted path order**: stage the passed bags yourself (e.g. symlinks
  `raw/ep_000000/` -> each bag's files, in manifest order), never point it at a producer's run dir.
* **No contract is embedded in the bags**: the simulator records with `ros2 bag record`, not Rosetta's recorder, so
  the porter's "different contract" check has nothing to compare and `--contract` (the robot's file) is the only
  contract. Pass it explicitly; the porter copies it to `meta/rosetta_contract.yaml`.
* **Licence-bound assets**: `assets/` (PartNet-Mobility, ReplicaCAD: non-commercial) is git-ignored. Camera images in
  bags and datasets show those assets: never commit or publish datasets, videos or frames made from them.
* **Hub never pushed**: no `--push-to-hub`, `HF_HUB_OFFLINE=1`, no token in the image or the command.
* Never commit `runs/`, `cache/`, `.venv/`, `assets/` or `ThirdParty/*/install/`.
* Sim time restarts at every reset: never compare times across bags.

## 13. Replacing the old dataset code

The old `components/dataset/` (Dockerfile on `ros:jazzy-ros-base`, `build.sh`, `launch.py`, `src/dataset/{entry,
port, checks, labels, merge, dataset_info, progress}.py`, tests) is yours to replace. When v2 lands, delete what it
replaces (the project keeps no dead code); reuse parts deliberately (labels as columns, `.partial` + move, resume,
`aggregate_datasets` merge, `dataset_info.summary`). Update `components/dataset/README.md` to describe v2 and drop the
pointer to this file only if this file moves.

What is gone and why old imports break:

| Gone | Replaced by |
|---|---|
| `components/simulator/launch.py` (the old simulator HTTP job; `launch.py` here imports it) | `./revolve sim` (`components/simulator/README.md`). |
| The old robot spec field `contract` and the generated contracts | Hand-written `specs/robots/<r>/contract.yaml` (section 5). |
| `episode.json` with `steps`, `control_dt`, `datagen` / `recovery` blocks; `bag/` subdirectories | The evaluator's `episode_<n>.json` + `episode_<n>_bag/`, listed by the producer's manifest (sections 6 and 10). |
| The Jazzy image and the Jazzy `ThirdParty/rosetta/install` | A Lyrical image and a Lyrical Rosetta build (section 4). |
| `turn_dataset()` (turn logic in the dataset launcher) | Nothing here: turn logic is the manager's (DS9). |

Still usable: `revolve_common.wrapper`, `revolve_common.job`, `revolve_common.specs` (`specs.resolve`,
`specs.check_contract`, `specs.CONTRACT_FILE`), the `ThirdParty/rosetta` submodule (pinned; patches only in
`ThirdParty/patches/rosetta/`).

Rebase tips: take the simulator, `common/`, `specs/` and `components/datagen/` from main as they are; keep only your
dataset changes. Submodules: `git submodule update --init` (never `--recursive` or `--remote`). After the rebase:
`./revolve build simulator`, `./revolve validate`, `./revolve test simulator -- -k "contract or rosetta"` (the
interface you build on), then your own build and tests.

## 14. Definition of done for dataset v2

- [ ] `components/dataset/` contains only v2: Lyrical image (dependencies only), `build.sh` (Rosetta colcon-built
      against Lyrical into `ThirdParty/rosetta/install`, patches only from `ThirdParty/patches/rosetta/`), the
      converter, tests, README; the old code is deleted.
- [ ] LeRobot 0.6.1 + torch verified on the image's Python (versions recorded in the Dockerfile and README).
- [ ] `./revolve dataset convert --manifest M [--manifest M2 ...] [--split S] --turn K --out D` and `./revolve
      dataset merge --out D DATASET...` work in the image, offline, CPU only; the HTTP job wrapper runs the same
      entrypoint.
- [ ] Inputs: only passed manifest entries, in manifest order; fail closed on a missing bag / result, a non-passed
      result, a bag lacking a contract topic; no folder scanning; nothing re-judged.
- [ ] Output: a LeRobot v3.0 dataset at `<out>` (built in `<out>.partial/`), `meta/rosetta_contract.yaml`,
      `meta/revolve_dataset.json`, labels in `meta/episodes`; `LeRobotDataset` loads it and decodes its last frame.
- [ ] Checks: frames vs `duration_s` (tolerance as agreed, section 8.3), features vs the contract, task == prompt,
      labels read back, episode count; result JSON `{dataset, episodes, frames, problems}`; non-zero exit and nothing
      at `<out>` on any problem.
- [ ] Merge: fails closed unless contracts are byte-identical and fps / features / task match; labels preserved.
- [ ] Resume: same inputs + out keeps the finished dataset.
- [ ] Tests (`./revolve test dataset`): unit tests (manifest parsing, labels, checks, split, fail-closed cases) and
      at least one real conversion of a simulator-recorded bag (section 11) per supported robot you have bags for,
      plus a merge.
- [ ] `./revolve validate` and `./revolve test simulator -- -k "contract or rosetta"` still pass.
- [ ] No bags, datasets, renders, `runs/` content, tokens or Hub pushes.

## 15. File map

| Path | What |
|---|---|
| `specs/robots/<r>/contract.yaml` | The robot's Rosetta contract (section 5). |
| `common/revolve_common/specs.py` | `check_contract` (run by `./revolve validate`), spec loaders (`resolve` for the task prompt). |
| `common/tests/test_contracts.py` | The contract checks' unit tests. |
| `components/simulator/ros/revolve_sim/config/record.yaml` | The simulator's recorded topics (what a contract may name). |
| `components/simulator/ros/revolve_sim/revolve_sim/evaluator.py` | Publishes `/task/prompt`; writes `episode_<n>.json`; drives the recorder. |
| `components/simulator/ros/revolve_sim/launch/sim.launch.py` | Starts the recorder (topics, `--custom-data` prompt). |
| `components/simulator/ros/revolve_sim/test/check_rosetta.py`, `test_rosetta_contracts.py`, `test_contract.py` | Rosetta's loader on every contract; the contract checked on real bags (topics, stamps, decoders, prompt). |
| `components/simulator/README.md` | The simulator: run, outputs, evaluator, reset, recording. |
| `components/datagen/DATAGEN-v2.md`, `components/datagen/src/datagen/core/manifest.py` | The producer side: commanding (streaming), the manifest. |
| `components/dataset/` | The old component (to replace). |
| `ThirdParty/rosetta/docs/reference/{contract,message-types,rosetta-port}.md` | Rosetta's contract format, codecs and porter. |
| `ThirdParty/patches/rosetta/` | Local Rosetta patches (none today). |
| `common/revolve_common/wrapper.py`, `job.py` | The HTTP job wrapper and job files. |
