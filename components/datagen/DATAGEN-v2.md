# Datagen v2: building the data generator against the new simulator

This document is the complete hand-over for building **datagen v2**, the component that produces expert
demonstrations, on top of the rewritten simulator. It is self-contained: everything datagen may rely on is
specified here, with the exact names, flags, topics, paths and message types as they run on this branch. If
something you need is not here, it is not part of the contract. Ask for a new simulator flag or topic rather than
reaching into the simulator (see [Roles and boundaries](#3-roles-and-boundaries)).

Contents

1. [Purpose and status](#1-purpose-and-status)
2. [What changed since the old datagen](#2-what-changed-since-the-old-datagen)
3. [Roles and boundaries](#3-roles-and-boundaries)
4. [The simulator contract](#4-the-simulator-contract)
5. [Commanding the robot](#5-commanding-the-robot)
6. [Observations and privileged information](#6-observations-and-privileged-information)
7. [Recording: what the simulator writes](#7-recording-what-the-simulator-writes)
8. [Datagen inputs and outputs](#8-datagen-inputs-and-outputs)
9. [Specs primer](#9-specs-primer)
10. [Robots](#10-robots)
11. [Build, run, test](#11-build-run-test)
12. [Caveats and lessons learned](#12-caveats-and-lessons-learned)
13. [Rebasing an existing datagen branch](#13-rebasing-an-existing-datagen-branch)
14. [Definition of done for datagen v2](#14-definition-of-done-for-datagen-v2)
15. [File map](#15-file-map)

---

## 1. Purpose and status

Datagen turns a **task** (for example `open_fridge_door`), a **robot** (for example `mobile_panda`) and a set of
**scene instances** (the scene with a randomised robot start, one per seed) into recorded expert episodes that the
evaluator accepted. Downstream, the dataset component converts the episode bags into training data. Recovery and
evaluation use the same simulator interface and the same bags.

Status of the pipeline on this branch:

| Part | Status |
|---|---|
| Simulator (`components/simulator`) | Rewritten and tested: world from YAML, robot, controllers, evaluator, reset, recording. |
| Specs (`specs/`, `common/revolve_common/specs.py`) | New scene / robot / task / sanity formats, validated by `./revolve validate`. |
| Datagen (`components/datagen`) | **Old code, broken.** It is yours to replace with v2 (section 13). |
| Manager (`components/manager`, `revolve run/resume/status/stop`) | **Broken and removed from the CLI for now.** Datagen v2 is tested in isolation with its own `mock_manager.py`. |
| Recovery, dataset, training, evaluation | Untouched, not yet adapted to the new simulator. |

## 2. What changed since the old datagen

The old flow is gone. In short:

| Old | New |
|---|---|
| AutoMoMa + cuRobo planned offline from static files (the old plugin dicts from `specs_bridge.py`). | Datagen is a live ROS 2 client of a running simulator. It reads the simulator's topics and commands the robot through the controllers' action topics while the episode runs. |
| A gz-internal replay system (`automoma_replay_system.py`, `phase_replay_system.py`, ...) replayed the plan inside the simulator, checked it offline and recorded the bag. | No replay and no plan files inside the simulator. The simulator records every episode itself, whoever drives it. Acceptance is the evaluator's `passed`. |
| Simulator = gz-sim with custom Python systems, a custom robot model generator and a custom torque controller, started as an HTTP job (`components/simulator/launch.py`). | Simulator = ROS 2 **Lyrical** on Ubuntu 26.04 + Gazebo **Kura** (gz-sim 11, gz-physics main) with the **MuJoCo** physics engine + **gz_ros2_control** + upstream `ros2_controllers` + MoveIt 2. Started with `./revolve sim`. |
| Robots as generated models + `contract.yaml`; tasks with start blocks and planner fields. | Robots as xacro URDF + `controllers.yaml` + a tiny `robot.yaml`; the scene is the single source of truth for the robot start; tasks are ordered, state-driven phases. |
| The manager scheduled datagen through the HTTP job wrapper. | The manager is broken for now. `mock_manager.py` (yours) launches simulators and generators directly. |

## 3. Roles and boundaries

### 3.1 The pieces

```
                 randomizer config ──> randomizer ──> scene instances (scene_<seed>/scene.yaml)
                                                              │
mock_manager.py --workers N ──┬── pair 1: revolve sim (container, ROS_DOMAIN_ID=D1) <──ROS 2──> generator 1
                              ├── pair 2: revolve sim (container, ROS_DOMAIN_ID=D2) <──ROS 2──> generator 2
                              └── ...                                  │
                                                     episode results + bags in scene_<seed>/ (written by the sim)
                                                     manifest.json (written by datagen)
```

* **Simulator**: one container per sim. It owns the world, physics, controllers, evaluator, reset and the
  **recording**. It is started only through the `./revolve sim` CLI (section 4.1).
* **Generator** (datagen proper): a separate container acting as a ROS 2 client of exactly one running sim, on that
  sim's `ROS_DOMAIN_ID`, with host networking. Given a scene instance (+ robot + task), it **does the task**: it
  plans and commands the robot live. It never randomises anything.
* **Randomizer**: a separate tool that lives in the datagen folder, with its own config YAML. It writes concrete
  per-seed scene instances (section 8.1). It runs before the sims and never touches a running sim.
* **mock_manager.py**: a minimal driver for testing datagen in isolation. It runs the randomizer, launches
  `revolve sim` as a subprocess per pair, starts the generators, collects the per-pair outputs and merges the
  manifest. It is not the real manager and is not used by anything else.

### 3.2 Rules

1. **Strict 1:1 pairing.** One generator process drives exactly one sim (one domain). `--workers N` means N
   (sim, generator) pairs; the scene instances are split across pairs; each pair writes its own `scene_<seed>/`
   folders; the manifests are merged at the end. No planner or state is shared across sims (how you use GPU memory
   for planners is up to you).
2. **Real-time, sim time.** The simulator runs at real-time factor 1 (1 ms physics step). All timing in the
   generator must use **sim time** (`use_sim_time: true`, `/clock`). Faster than real time is a possible later
   flag; never assume it.
3. **The simulator is the only recorder.** Datagen never records bags, never rewrites or moves the simulator's files.
   It writes only its own `manifest.json` (and logs).
4. **No controller bypass.** During an episode, datagen commands only through the controllers' interfaces (section
   5). Never teleport, `set_pose`, or call gz services or topics directly. Never reach into the simulator container
   (`docker exec`), its gz transport or its files.
5. **No randomisation inside the generator.** Variation comes only from the scene instances.
6. **No extra acceptance.** An episode is accepted if and only if the evaluator's result has `passed: true`. If you
   need an extra check, it belongs in `task.yaml` (a phase condition or a sanity check), not in datagen.
7. **New needs become new simulator CLI flags or topics**, added to the simulator and to this document, never
   workarounds inside the simulator from the outside.

## 4. The simulator contract

### 4.1 Starting a sim: `./revolve sim`

```
./revolve sim --scene S --robot R [--task T [--no-autostart] [--record images|no-images] [--out DIR]]
              [--gui] [--name N] [--domain-id D]
```

| Flag | Meaning |
|---|---|
| `--scene S` | A scene name (`specs/scenes/<S>/scene.yaml`) **or a path to a scene.yaml file**: a scene instance (section 8.1). A file inside the repo is used through the repo mount; a file outside the repo is mounted read-only at its own path. With `--task`, the scene's `name` must be the task's `scene`. |
| `--robot R` | `specs/robots/<R>/` (section 10). |
| `--task T` | `specs/tasks/<T>/task.yaml`: starts the evaluator and the recorder. Without it, there are no episodes. |
| `--no-autostart` | With `--task` (refused without it, exit 1): episodes start only on `/evaluator/start`. **Datagen always passes it.** (Without it, the first episode starts by itself as soon as the evaluator has every input.) |
| `--record images\|no-images` | With `--task` (refused without it, exit 1): the episode bags with the three cameras (JPEG) or without them. Default `images`. **Datagen uses `images`.** The simulator's own tests use `no-images` (size). |
| `--out DIR` | With `--task` (refused without it, exit 1): episode results and bags go into DIR (a host directory, created if missing, mounted writable into the container **at the same absolute path**, so every path in a result is a valid host path). Default: `runs/sim/<time>_<task>/` (the time in UTC, the container's clock). Episode numbers continue after the highest n of any `episode_<n>` file already in DIR (a result, a gz log copy or a bag, also a bag that a killed run left without its result), so a relaunch into the same DIR never overwrites anything. |
| `--name N` | The container name (default `revolve-sim`). Must be unique per running sim. It is also the sim's `GZ_PARTITION`. |
| `--domain-id D` | The sim's `ROS_DOMAIN_ID` (0..232). Default: derived from the name (1..101), skipping the domains of other running containers. **Datagen passes it explicitly**, distinct per pair, and starts its generator with the same value. |
| `--gui` | Also the gz GUI client on the host's X display (needs `DISPLAY` and `XAUTHORITY`). Costs real-time factor; never for datagen runs. |

The command runs in the foreground until the sim exits. It prints
`revolve sim: ROS_DOMAIN_ID=<D> (from the host: export ROS_DOMAIN_ID=<D>)` on stderr at start.

Examples:

```
# a datagen pair's sim: a scene instance, recorded with images into its scene folder
./revolve sim --scene runs/fridge_exp/datagen/scene_7/scene.yaml --robot mobile_panda --task open_fridge_door \
    --no-autostart --record images --out runs/fridge_exp/datagen/scene_7 --name dg-sim-0 --domain-id 120

# the plain scene, interactive look
./revolve sim --scene kitchen_partnet --robot mobile_so101_xl --task open_drawer --no-autostart --gui
```

Container facts that matter to a client:

* Host networking, one `ROS_DOMAIN_ID` per sim, and `GZ_PARTITION=<name>`: several sims on one host do not see
  each other, as long as names and domains differ.
* ROS 2 in the simulator uses Fast DDS over **UDPv4 only** (`FASTDDS_BUILTIN_TRANSPORTS=UDPv4`). Use the same in
  your client container: no shared-memory transport, which a killed container would leak into the host's
  `/dev/shm`.
* The repo is mounted read-only at `/repo` inside the sim; `runs/` and `--out` are writable.
* The sim needs the GPU (camera rendering through EGL). `revolve sim` runs a GPU health check first and refuses to
  start on a GPU in a broken state ("GPU requires reset": reboot).

### 4.2 Readiness

A sim is ready for an episode when **`/evaluator/state` has `"ready": true`**. Before that, `/evaluator/start` is
refused. Bring-up (container start to ready) is typically 10-20 s on the reference laptop.

Do not infer readiness from anything else (controller states, topic presence). If `ready` does not come within a
generous timeout (we use 180 s), tear the sim down and relaunch.

### 4.3 Episodes

| Interface | Type | Semantics |
|---|---|---|
| `/evaluator/start` | `std_srvs/srv/Trigger` | Starts an episode at the current sim time. `success: true`, `message: "episode <n>"` (n = the episode number, also the file index). Refused with `success: false` when inputs are not ready (`"inputs not ready: ..."`) or an episode is running (`"episode <n> is running"`). |
| `/evaluator/state` | `std_msgs/msg/String`, JSON, 10 Hz | `{"t": <sim time s>, "ready": bool, "resets": int, "episode": int or null, "phase": str or null, "values": {key: float or null}}`. `episode`/`phase` are null between episodes. `values` holds every condition's current value (also between episodes; null when it cannot be computed, e.g. a `slip` condition outside its phase or before the inputs are live), keyed `<phase>.done[i]`, `<phase>.while[i]`, `sanity.<check>`. `resets` counts the world resets the evaluator has seen. |
| `/evaluator/result` | `std_msgs/msg/String`, JSON, **transient local** (depth 1) | Published once when an episode ends (the schema below). A late subscriber gets the last result. |

The evaluator runs at 100 Hz on sim time. An episode ends when the last phase passes (`passed: true`) or when a phase
fails: a `while` condition is violated, the phase's `timeout_s` or the task's `budget_s` runs out, a timed sanity
check is violated (fails the active phase), or an untimed sanity check is violated (fails as phase `global`). A
world reset during an episode ends it with `reason: "reset"`.

Result JSON (also written to `<out>/episode_<n>.json`):

```json
{
 "task": "open_fridge_door", "scene": "kitchen_partnet", "robot": "mobile_panda", "episode": 0,
 "sim_time_start": 3.91,
 "passed": false, "failed_phase": "approach", "reason": "timeout_s",
 "t_fail": 20.001, "check": null, "condition": null, "value": 20.001, "duration_s": 20.001,
 "phases": [
  {"name": "approach", "status": "failed", "t_start": 0.0, "t_end": 20.001, "reason": "timeout_s",
   "last": {"approach.done[0]": 0.888, "approach.done[1]": 1.573, "approach.done[2]": 1.573}},
  {"name": "grasp", "status": "not_reached", "t_start": null, "t_end": null, "last": {}},
  {"name": "open", "status": "not_reached", "t_start": null, "t_end": null, "last": {}}
 ],
 "sanity_last": {"sanity.body_speed": 0.0002, "sanity.penetration": 0.0, "sanity.robot_contacts": 0.0,
                 "sanity.finite_cmds": 0.0, "sanity.gz_log": 0.0},
 "rate_hz": 98.95, "tick_ms": {"mean": 1.05, "max": 2.67},
 "contacts": [],
 "gz_log": "/abs/path/to/out/episode_0_gz.log",
 "result_file": "/abs/path/to/out/episode_0.json",
 "bag": "/abs/path/to/out/episode_0_bag"
}
```

| Field | Meaning |
|---|---|
| `passed` | **The acceptance criterion.** |
| `failed_phase` | The phase that failed (`global` for an untimed sanity check), null if passed. |
| `reason` | `timeout_s`, `budget_s`, `while`, `sanity`, `reset`; null if passed. |
| `check` / `condition` / `value` | The failing condition's key (e.g. `open.while[0]`, `sanity.robot_contacts`), its readable form, its value. |
| `t_fail`, `duration_s` | Seconds since the episode start (sim time). |
| `phases[]` | Per phase: `status` (`passed`, `failed`, `aborted`, `not_reached`), `t_start`/`t_end` (s since episode start), `last` (the last values of its conditions). |
| `sanity_last` | The last sanity values. |
| `contacts` | Robot contacts at the end (not with the ground), for debugging a contact failure. |
| `rate_hz`, `tick_ms` | Evaluator rate and cost (a health signal; expect ~99-100 Hz). |
| `sim_time_start` | Sim time at the start. Sim time restarts at 1 s on every reset, so this is not unique across episodes. |
| `result_file`, `bag`, `gz_log` | Absolute paths (host paths with `--out`). `bag` is a rosbag2 directory; null with recording off. `gz_log` is a copy of the episode's part of the gz server's console log, from its start to its end (the log itself is inside the container); empty if gz wrote nothing meanwhile, null if gz wrote no log. |

**Completion rule:** `/evaluator/result` is published first (so that it is inside the bag); then the bag is closed
(its `metadata.yaml` written); then `episode_<n>.json` is written. **Once `episode_<n>.json` exists, the episode's
files are complete.** Wait for that file before you read the bag, reference it in the manifest as final, or tear
the sim down. This typically takes well under a second after the result message.

### 4.4 Reset: `/simulator/reset` and the per-episode loop

`/simulator/reset` (`std_srvs/srv/Trigger`, node `/simulator`, always available) puts the sim back to the scene
instance's start: every model and the robot to their initial poses and joint positions, sim time back to 1 s,
controllers restarted on the reset state, the arm and gripper driven back to their start values by their
controllers (no teleport) and settled, the evaluator re-armed. It returns `success: true` with a message like
`"reset in 1.74 s, sim time 2.607"`, or `success: false` with the reason (a step failed or timed out; it waits at
most 20 s per step). Typical wall time 1.7-2.5 s.

The client loop for one sim (one scene instance):

```
launch:  revolve sim --scene <instance> --robot R --task T --no-autostart --record images \
             --out <scene dir> --name <unique> --domain-id <D>       (subprocess)
wait:    /evaluator/state "ready": true                             (timeout -> relaunch)
for each wanted episode on this scene instance:
    r = resets in the latest /evaluator/state
    if not the first episode of this sim:
        call /simulator/reset                                       (success false -> relaunch)
        wait for an /evaluator/state with resets > r and ready: true
    call /evaluator/start  -> "episode <n>"
    plan and command the robot (section 5), in sim time
    wait for /evaluator/result with episode == n
    wait for <scene dir>/episode_<n>.json                           (the files are complete)
    record the episode in the manifest (passed or not)
teardown: docker rm -f <name>
```

Notes:

* The first episode after launch needs no reset (the sim starts at the instance's start).
* You may call `/simulator/reset` mid-episode to abort it; the episode ends with `reason: "reset"`, its files are
  written as usual, and its bag can end with up to ~0.2 s of messages after the jump back in sim time.
* Episodes of the same scene instance reset the sim. **A different scene instance is a new sim**: tear down and
  relaunch with the new `--scene` (the robot start is part of the world).
* Relaunch also when the reset fails, readiness times out, the sim exits, or you see the clock fault (section 12).
* Teardown is `docker rm -f <name>` (this is also what kills the `revolve sim` subprocess). It does not leak
  resources with the UDPv4 transport.

### 4.5 Your client container

* ROS 2 **Lyrical** `rclpy` and the message packages of the topics you use: `std_msgs`, `std_srvs`, `sensor_msgs`,
  `geometry_msgs`, `trajectory_msgs`, `control_msgs`, `tf2_msgs`, `nav_msgs`, `ros_gz_interfaces` (built here from
  `ThirdParty/ros_gz`, installed in `ThirdParty/ros_gz_ws/install` by `./revolve build simulator`), plus MoveIt 2
  (`moveit_py`) if you plan with it. Use the same ROS distribution as the simulator: Lyrical is built for
  Ubuntu 26.04 (Python 3.14). Check your planner stack (torch, cuRobo) against that, or keep the planner in its own
  process and talk to it locally.
* The simplest working setup, used by the simulator's own contract test: a container from the simulator image
  (`revolve-simulator:latest`) with `--network host`, `-e ROS_DOMAIN_ID=<D>`, `-e FASTDDS_BUILTIN_TRANSPORTS=UDPv4`,
  `-e PYTHONPATH=/repo/common`, the repo read-only at `/repo`, `ThirdParty/gz_kura/install` at `/opt/gz_ws/install`
  and `ThirdParty/ros_gz_ws/install` at `/opt/ros_gz_ws/install` (both read-only), and `--out` at its own path. See
  `client()` in `components/simulator/ros/revolve_sim/test/test_contract.py`. Your own image (with your planner) can
  build on the same base.
* Every node: `use_sim_time: true`. Subscriptions to `/evaluator/result` must be transient local to get a result
  published before you subscribed.
* rclpy: spin in an executor thread (a `MultiThreadedExecutor` in a background thread), do not use `spin_once` in a
  loop (it drops camera frames and service replies under load).

## 5. Commanding the robot

### 5.1 Interfaces

| What | Interface | Type |
|---|---|---|
| Base | `/omni_wheel_drive_controller/cmd_vel` | `geometry_msgs/msg/TwistStamped` (only this type), **robot frame** (`base_footprint`): `linear.x` forward, `linear.y` left, `angular.z` counter-clockwise (the other twist fields are not used by an omni base). |
| Arm | `/<arm controller>/follow_joint_trajectory` (action) or `/<arm controller>/joint_trajectory` (topic) | `control_msgs/action/FollowJointTrajectory`, `trajectory_msgs/msg/JointTrajectory` |
| Gripper | `/<hand controller>/follow_joint_trajectory` or `/<hand controller>/joint_trajectory` | same types |

Controller names per robot:

| Robot | Arm controller | Hand controller | Arm joints | Hand joint |
|---|---|---|---|---|
| `mobile_panda` | `panda_arm_controller` | `panda_hand_controller` | `panda_joint1` .. `panda_joint7` | `panda_finger_joint1` [m per finger; 0 closed, 0.04 open] |
| `mobile_so101`, `mobile_so101_xl` | `arm_controller` | `gripper_controller` | `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll` | `gripper` [rad, moving jaw; -0.1745 closed limit, 1.745 open] |

All arm and gripper controllers are `joint_trajectory_controller` on **effort** command interfaces with PID gains
(no gravity-compensation plugin). The base is `omni_wheel_drive_controller` (4 omni wheels, X layout).

### 5.2 Rules and semantics

* **Ramp base twists** over ~0.5 s. The controller has no limiter; a step in twist kicks the yaw (up to ~9 deg seen).
* The base stops if no `cmd_vel` arrives for 0.5 s (`cmd_vel_timeout`). Publish continuously (we use 50 Hz) while
  moving, then a few zero twists.
* **FollowJointTrajectory** (e.g. what MoveIt executes): the JTC interpolates the whole trajectory. The result's
  `error_code` 0 means success.
* **Streaming** (what a policy does at inference): publish single-point `JointTrajectory` messages on
  `/<controller>/joint_trajectory` at the action rate (30 Hz) with a short `time_from_start` (the contract test uses
  0.1 s) and a zero header stamp (start now). Each message replaces the active trajectory.
* **Gripper**: grasping is by friction, with no weld. Command a goal *past* closed so the effort controller squeezes
  at its clamp: Panda `panda_finger_joint1` -0.04 (~12 N per finger, `u_clamp` 12); SO-101 / SO-101 XL `gripper`
  about -0.3 (the XL's clamp gives ~12 N). The joint stops where the object is; the controller keeps squeezing.
* Never send NaN/Inf: the `finite_cmds` sanity check fails the episode (it watches `cmd_vel` and the
  `joint_trajectory` topics; FollowJointTrajectory goals are not seen by it, but are still your responsibility).
* MoveIt plans the arm only. Command the hand directly (FollowJointTrajectory or streamed points), not with
  MoveIt's GripperCommand: the hand controllers are JTCs.

### 5.3 How the action is recorded (independent of how you command)

The recorded action is defined on the simulator side, so it is the same whether you used MoveIt + FollowJointTrajectory
or streamed points:

* **arm and gripper**: each JTC's `/<controller>/controller_state` (`control_msgs/msg/JointTrajectoryControllerState`)
  `reference.positions`, the setpoint the controller is tracking at that instant (published every control cycle,
  1 kHz);
* **base**: the commanded twist on `/omni_wheel_drive_controller/cmd_vel` (zero when none arrived within 0.5 s).

The dataset samples both at **30 Hz** (section 7.4). At inference, the policy streams single-point
`joint_trajectory` messages and `cmd_vel` at 30 Hz, i.e. exactly the recorded action format. Consequence for the
expert: smooth, continuous references make learnable actions. Avoid large jumps in the reference (e.g. a 0 s
`time_from_start` to a far point) and long pauses that are not part of the task.

## 6. Observations and privileged information

Datagen is a **privileged expert**: it may use ground truth that the policy will not have. Use these sources (and
only these):

| Need | Source |
|---|---|
| Robot model | `/robot_description` (`std_msgs/String`, transient local) from the sim. Never re-run xacro yourself; the sim's URDF is the one that runs (initial joint values, controllers file baked in). |
| Planning (MoveIt 2) | `specs/robots/<r>/moveit/` (SRDF, kinematics, joint limits, Pilz limits, MoveIt controllers, moveit_py options) + `revolve_sim.moveit.moveit_params(repo, robot, urdf=...)` (`components/simulator/ros/revolve_sim/revolve_sim/moveit.py`), or your own planner config. MoveIt plans in `base_footprint` (no virtual joint). |
| Robot joint state | `/joint_states` (`sensor_msgs/JointState`, joint_state_broadcaster, 1 kHz) |
| Robot link poses | `/tf` (robot_state_publisher at 100 Hz + the omni odometry `odom -> base_footprint` at 1 kHz), `/tf_static` |
| Robot base pose in the world (ground truth) | `/model/<robot>/pose` (`tf2_msgs/TFMessage`, 100 Hz; `header.frame_id` = the world name, `child_frame_id` = the robot model name) |
| Scene object poses (ground truth) | `/model/<m>/pose` for every scene model `m` (100 Hz): the model in the world (`frame_id` = world name, `child_frame_id` = `m`) and each link in the model (`frame_id` = `m`, `child_frame_id` = `m/<link>`) |
| Scene articulations | `/world/<world>/model/<m>/joint_state` (`sensor_msgs/JointState`, 100 Hz) for each articulated model, e.g. `fridge` (`top_door_hinge`, ...) and `kitchen_counter` (`middle_slide_top`, ...) |
| Contacts | `/world/<world>/model/<robot>/link/<link>/sensor/contact/contact` (`ros_gz_interfaces/Contacts`) for every robot link with a collision except the omni rollers. Published **only while in contact**. Use the header stamps, not arrival time. Collision names are the gz names after sdformat's fixed-joint lumping (e.g. `<link>_collision[_k]`). |
| Cameras | `/<camera>/image` (raw, `sensor_msgs/Image`), `/<camera>/image/compressed` (JPEG), `/<camera>/camera_info`, 30 Hz, 480 x 480; cameras `wrist_cam`, `overhead_left_cam`, `overhead_right_cam`. Use the same topics the policy will use for any input the policy also has. |
| Collision geometry of the scene | Files: the scene instance + the model SDFs in the scene's `assets` search path, through `revolve_common.specs` (`specs.load(<instance>)`, `specs.scene_models(scene, repo)` -> `ModelInfo.sdf` per model). The assets are read-only. Note the fridge body link frame is flipped (z down); take model-frame values. Prefer live ground-truth poses for anything that moves. |
| Task semantics | `task.yaml` through `revolve_common.specs` (`specs.resolve(repo / "specs", task, robot=...)`): targets (e.g. `handle: fridge::top_door::top_door_handle`), phases and thresholds. |
| Evaluator progress | `/evaluator/state` values (e.g. `approach.done[0]` = the current TCP-to-handle distance). Useful for closed-loop experts and debugging. |

The world name is the scene's `name` (`kitchen_partnet`). Scene model names are the keys of the scene's `models:`
(`walls`, `kitchen_counter`, `fridge`, `table_01`, `cup`).

## 7. Recording: what the simulator writes

### 7.1 Layout

With `--out DIR`, per episode n:

```
DIR/
  episode_<n>.json            the result (section 4.3), written last
  episode_<n>_gz.log          the episode's part of the gz server's console log (a copy)
  episode_<n>_bag/            rosbag2 directory, one recording per episode
    metadata.yaml
    0_episode_<n>_bag_<date-time>.mcap
```

The bags are mcap with zstd chunk compression (storage preset `zstd_fast`), sim time stamps. Any mcap or rosbag2 reader
handles them (`ros2 bag info`, `rosbag2_py`, the `rosbags` and `mcap` Python libraries). Datagen must not move or
rewrite them; reference them from the manifest.

### 7.2 The frozen topic list

The list lives in one file, `components/simulator/ros/revolve_sim/config/record.yaml`, filled in per scene and robot.
It is the contract with the dataset converter: it changes only together with its consumers.

| Topic | Type | Rate | Notes |
|---|---|---|---|
| `/clock` | `rosgraph_msgs/Clock` | 1 kHz | |
| `/robot_description` | `std_msgs/String` | once per bag | The URDF (transient local). |
| `/tf` | `tf2_msgs/TFMessage` | ~1.1 kHz | robot_state_publisher (100 Hz) + odometry `odom -> base_footprint` (1 kHz) |
| `/tf_static` | `tf2_msgs/TFMessage` | once per bag | Transient local; in every bag. |
| `/joint_states` | `sensor_msgs/JointState` | 1 kHz | |
| `/<jtc>/controller_state` | `control_msgs/JointTrajectoryControllerState` | 1 kHz | Per JTC: `reference` = the arm / gripper action. |
| `/<jtc>/joint_trajectory` | `trajectory_msgs/JointTrajectory` | as commanded | Streamed commands. |
| `/omni_wheel_drive_controller/cmd_vel` | `geometry_msgs/TwistStamped` | as commanded | The base action. |
| `/omni_wheel_drive_controller/odom` | `nav_msgs/Odometry` | 1 kHz | Wheel odometry. |
| `/model/<robot>/pose` | `tf2_msgs/TFMessage` | 100 Hz | Ground truth. |
| `/model/<m>/pose` (each scene model) | `tf2_msgs/TFMessage` | 100 Hz | Ground truth. |
| `/world/<world>/model/<m>/joint_state` (each articulated model) | `sensor_msgs/JointState` | 100 Hz | |
| `/world/<world>/model/<robot>/link/<l>/sensor/contact/contact` | `ros_gz_interfaces/Contacts` | only in contact | |
| `/<camera>/camera_info` | `sensor_msgs/CameraInfo` | 30 Hz | |
| `/evaluator/state` | `std_msgs/String` (JSON) | 10 Hz | |
| `/evaluator/result` | `std_msgs/String` (JSON) | once, at the end | Recorded live only (never the previous episode's). |
| `/<camera>/image/compressed` | `sensor_msgs/CompressedImage` | 30 Hz | `--record images` only. JPEG (`format` `"rgb8; jpeg compressed bgr8"`), from `compressed_image_transport` through ros_gz_image's image_transport publisher. Raw images are never recorded. |

Topics that carry messages only on demand (contacts, `joint_trajectory`, `cmd_vel`) are absent from a bag in which
they never had a publisher; treat "absent" as "no messages". Every other topic is in every bag.

`controller_state`, `joint_states`, `odom` and `/tf` run at the controller rate (1 kHz) because the upstream
controllers publish every cycle; the dataset downsamples.

### 7.3 Sizes (measured, 20 s episodes)

| Recording | Robot | Size per episode | Per second |
|---|---|---|---|
| `--record images` | mobile_panda | 133-139 MB | ~6.8 MB/s |
| `--record no-images` | mobile_panda | 15.3 MB | ~0.76 MB/s |
| `--record no-images` | mobile_so101_xl | 14.7 MB | ~0.73 MB/s |

The images dominate. Plan disk space for the images case: ~0.4 GB per minute of recorded episodes.

### 7.4 Extracting the 30 Hz action

```python
import numpy as np, rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

def read(bag, topics):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    out = {t: [] for t in topics}
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic in out:
            out[topic].append(deserialize_message(data, get_message(types[topic])))
    return out

stamp = lambda m: m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
msgs = read(bag, ["/panda_arm_controller/controller_state", "/panda_hand_controller/controller_state",
                  "/omni_wheel_drive_controller/cmd_vel"])
arm, hand = msgs["/panda_arm_controller/controller_state"], msgs["/panda_hand_controller/controller_state"]
t0, t1 = max(stamp(arm[0]), stamp(hand[0])), min(stamp(arm[-1]), stamp(hand[-1]))
grid = np.arange(t0, t1, 1 / 30)
def hold(ms, f):                       # the last value at or before each tick
    st = np.array([stamp(m) for m in ms]); v = np.array([f(m) for m in ms])
    return v[np.clip(np.searchsorted(st, grid, side="right") - 1, 0, len(st) - 1)]
a_arm = hold(arm, lambda m: m.reference.positions)
a_hand = hold(hand, lambda m: m.reference.positions)
cmd = msgs["/omni_wheel_drive_controller/cmd_vel"]
a_base = np.zeros((len(grid), 3))      # zero unless a cmd_vel arrived within the controller's 0.5 s timeout
if cmd:
    ct = np.array([stamp(m) for m in cmd])
    cv = np.array([[m.twist.linear.x, m.twist.linear.y, m.twist.angular.z] for m in cmd])
    i = np.searchsorted(ct, grid, side="right") - 1
    ok = (i >= 0) & (grid - ct[np.clip(i, 0, None)] <= 0.5)
    a_base[ok] = cv[i[ok]]
```

The same code, with checks, is `action_series()` in
`components/simulator/ros/revolve_sim/test/check_contract.py`. The dataset component will convert bags with a
Rosetta contract written against the topic list above.

## 8. Datagen inputs and outputs

### 8.1 Randomizer and scene instances

The randomizer is its own tool in `components/datagen/` with its own config YAML. It writes one concrete
**scene instance** per seed: a copy of the scene's `scene.yaml` with randomised fields. First version: **only
`robot_start`** (base `[x, y, yaw]` and optionally `arm`), everything else unchanged.

Rules for a scene instance (enforced by the loader):

* The file is named `scene.yaml` and lives **outside** any `specs/` directory (e.g. `runs/<exp>/datagen/scene_<seed>/scene.yaml`).
* Its `name` stays the base scene's name (`kitchen_partnet`): it is the world name, and the task's `scene`.
* `assets` stay as in the base scene (paths relative to the repo).
* `robot_start.arm`: a pose name of the robot's `robot.yaml` (`ready`, `folded`) or a joint list.
* It must pass `./revolve validate <path/to/scene.yaml> --robot <robot>` (with `--robot`, `robot_start.arm` is checked against that robot; `./revolve sim` runs the same checks and refuses an invalid instance before it starts docker, exit 1).
* The start must be collision-free and leave the robot room to work: check it against the scene geometry
  (section 6) before writing it. A robot spawned in collision fails the episode on a sanity check.

A suggested randomizer config (the format is yours to define):

```yaml
scene: kitchen_partnet           # the base scene (specs/scenes/<scene>/scene.yaml)
robot: mobile_panda              # for collision checks against the robot footprint
seeds: [0, 99]                   # inclusive range
robot_start:
  base:                          # uniform ranges in the world frame [m, m, rad]
    x: [-1.85, -1.65]
    y: [1.00, 1.20]
    yaw: [1.47, 1.67]
  arm: [folded]                  # choices
out: runs/<exp>/datagen          # writes <out>/scene_<seed>/scene.yaml
```

The scene instance can live in the same `scene_<seed>/` folder that the sim writes its episodes into (`--out`); the
evaluator numbers its episodes from the `episode_<n>` files there and ignores everything else. The simulator's contract test does exactly this.

### 8.2 Datagen job input

The job YAML that `mock_manager.py` (and later the manager) hands datagen:

```yaml
task: open_fridge_door
robot: mobile_panda
episodes:                         # one entry per scene instance
  - {seed: 0, scene: runs/fridge_exp/datagen/scene_0/scene.yaml}
  - {seed: 1, scene: runs/fridge_exp/datagen/scene_1/scene.yaml}
episodes_per_scene: 1             # accepted episodes wanted per scene instance
out: runs/fridge_exp/datagen      # scene_<seed>/ folders + manifest.json
```

`mock_manager.py --workers N` splits `episodes` across N (sim, generator) pairs.

### 8.3 Output layout

```
runs/<exp>/datagen/
  manifest.json                   written by datagen (merged across workers)
  scene_<seed>/                   = the sim's --out for that scene instance
    scene.yaml                    the scene instance (randomizer)
    episode_<n>.json              results (simulator)
    episode_<n>_bag/              bags (simulator)
    episode_<n>_gz.log            gz server log copies (simulator)
```

### 8.4 manifest.json

One entry per episode attempted, accepted or not. Paths relative to the manifest's directory (or absolute, but be
consistent):

```json
{
 "task": "open_fridge_door",
 "robot": "mobile_panda",
 "created": "2026-11-22T10:15:00Z",
 "episodes": [
  {"seed": 7, "scene": "scene_7/scene.yaml", "robot": "mobile_panda", "task": "open_fridge_door", "episode": 0,
   "result": "scene_7/episode_0.json", "bag": "scene_7/episode_0_bag", "passed": false,
   "datagen": {"expert": "moveit_ompl_pilz_v1", "attempt": 0, "wall_s": 41.2}},
  {"seed": 7, "scene": "scene_7/scene.yaml", "robot": "mobile_panda", "task": "open_fridge_door", "episode": 1,
   "result": "scene_7/episode_1.json", "bag": "scene_7/episode_1_bag", "passed": true,
   "datagen": {"expert": "moveit_ompl_pilz_v1", "attempt": 1, "wall_s": 38.7}}
 ],
 "counts": {"scenes": 1, "attempted": 2, "passed": 1}
}
```

* `passed` is copied from the result JSON. Nothing else decides acceptance.
* Retries: datagen may retry a scene instance (reset, new attempt) until it has `episodes_per_scene` passed
  episodes or a retry limit you choose. Failed episodes **stay on disk** and in the manifest with `passed: false`
  (they are useful for recovery).
* A relaunch into the same `scene_<seed>/` continues the episode numbering, so nothing is overwritten.

## 9. Specs primer

Formats and checks: `common/revolve_common/specs.py`. Validate everything: `./revolve validate` (all specs);
`./revolve validate <file>`; `./revolve validate --robot <r>` (also checks every task against that robot);
`--no-assets` for names and syntax only. Values are SI (m, rad, s).

**Scene** (`specs/scenes/<s>/scene.yaml`): one fixed scene; the simulator generates the world SDF from it (ground,
lights, 1 ms MuJoCo physics are simulator defaults).

```yaml
name: kitchen_partnet
assets: [assets/generated_partnet, assets/generated]    # model:// search path, repo-relative, first match wins
models:                                                  # <entity>: {model, pose [x, y, z, yaw], joints}
  fridge: {model: fridge, pose: [-1.5, 2.68, 0.0, -1.5707963267948966]}
  kitchen_counter: {model: kitchen_counter, pose: [-2.625, 0.29, 0.0, 0.0]}
robot_start: {base: [-1.75, 1.1, 1.5707963267948966], arm: folded}   # the only place the start lives
```

(Initial scene joint values other than 0 are rejected by the simulator for now.)

**Robot** (`specs/robots/<r>/`): `robot.urdf.xacro` (upstream arm description + the shared omni base + the
`<ros2_control>` block for gz_ros2_control + camera and contact sensors), `controllers.yaml` (upstream controllers
only), `moveit/`, and a small `robot.yaml`:

```yaml
name: mobile_panda
description: robot.urdf.xacro
controllers: controllers.yaml
poses: {ready: [...], folded: [...]}            # named arm poses (arm joint order)
frames: {tcp: panda_tcp, hand: panda_hand, base: base_footprint}   # robot::<alias> entities
groups: {fingers: [panda_leftfinger, panda_rightfinger]}
```

**Task** (`specs/tasks/<t>/task.yaml`): ordered, state-driven phases. Each phase has `done` conditions (all true,
each for its `for_s`, -> next phase), optional `while` invariants (a violation fails the phase) and an optional
`timeout_s`; the episode has a `budget_s`. A condition is a metric on named entities with one comparison
(`lt`/`le`/`eq`/`ge`/`gt`) and an optional `for_s`:

```yaml
name: open_fridge_door
scene: kitchen_partnet
prompt: open the fridge door
targets: {handle: fridge::top_door::top_door_handle}      # <model>::<link>::<collision>
budget_s: 45.0
phases:
  - name: approach
    timeout_s: 20.0
    done:
      - {metric: distance, from: robot::tcp, to: handle, le: 0.015}
      - {metric: angle, from: robot::tcp, to: handle, axes: [z, -x], le: 0.131}
      - {metric: angle, from: robot::tcp, to: handle, axes: [x, z], le: 0.131}
  - name: grasp
    timeout_s: 5.0
    done: [{metric: contacts, from: robot::fingers, to: handle, eq: 2, for_s: 0.5}]
  - name: open
    timeout_s: 15.0
    done: [{metric: joint, of: fridge::top_door_hinge, ge: 1.047, for_s: 1.0}]
    while:
      - {metric: contacts, from: robot::fingers, to: handle, eq: 2}
      - {metric: slip, from: robot::tcp, to: handle, le: 0.013}
```

Metrics (`specs.METRICS`): `joint`, `distance` (a frame/link by its origin, a collision by its geometry; a mesh
collision by its origin), `angle` (`axes: [<from axis>, <to axis>]`, signed x/y/z, 0..pi), `contacts` (links of
`from` touching `to`), `gripper_opening`, `slip` (drift since the phase start), `peak_body_speed`, `penetration`,
`non_finite_commands`, `gz_log_problem_lines`. Entities: `robot`, `robot::<alias>`, `<model>`,
`<model>::<link|joint>`, `<model>::<link>::<collision>`, a target alias, and `any` / `any_except_targets` (contacts
`to` only).

**Sanity** (`specs/tasks/sanity.yaml`): the checks every task gets (only checks, keyed): `body_speed` (< 5 m/s),
`penetration` (< 0.01 m), `robot_contacts` (the robot touching anything but the targets: 0), `finite_cmds`
(untimed), `gz_log` (untimed). Timed checks fail the active phase; untimed ones fail the episode as `global`. A task
overrides or adds by key under `sanity:`, removes with `null`. Note `robot_contacts`: **any** robot contact with
something other than the task targets fails the episode (the ground excepted), so the expert must keep the robot
clear of the counter, the fridge body and the door edges.

## 10. Robots

| Robot | Arm | Base | TCP frame | Notes |
|---|---|---|---|---|
| `mobile_panda` | moveit_resources Panda (7 DoF) + hand | shared omni base | `panda_tcp` = `panda_hand` + 0.1034 m along z | Task: `open_fridge_door`. |
| `mobile_so101` | SO-101 (5 DoF), scale 1 | shared omni base | `so101_tcp` | No task yet. |
| `mobile_so101_xl` | SO-101 at scale 2, chassis top 0.60 m | shared omni base | `so101_tcp` | Task: `open_drawer`. |

TCP conventions (all robots): **+z = approach** (out of the gripper), **fingers close along ±y**. Panda: symmetric
fingers (mimic). SO-101: `so101_tcp` lies between the jaw tips, +y points from the moving jaw to the fixed jaw, and
the TCP is 8 mm off the fixed jaw's tip pad at any scale. Hand frames: `panda_hand`, `gripper_link`. Base frame:
`base_footprint` (on the floor). Finger groups (contacts): Panda `panda_leftfinger`, `panda_rightfinger`; SO-101
`gripper_link` (fixed jaw), `moving_jaw_so101_v1_link`.

Named arm poses: `ready`, `folded` (the scene start uses `folded`, tucked for driving). MoveIt groups: Panda
`panda_arm` (+ `hand`), SO-101 `arm` (tip `gripper_link`, plan for the pose link `so101_tcp`) and `gripper`. The
SO-101 uses the `pick_ik` kinematics plugin (5 DoF: a full 6-DoF pose target is usually unreachable; see section 12).

## 11. Build, run, test

```
./revolve assets                      # one-time: fetch and convert the scene assets into assets/ (licence-bound,
                                      # git-ignored; assets_pipeline/README.md)
./revolve build simulator [--clean]   # the simulator image (deps only) + Kura, ros_gz, gz_ros2_control, revolve_sim
                                      # into ThirdParty/*/install (rebuild after changing components/simulator/ros)
./revolve sim --scene kitchen_partnet --robot mobile_panda --task open_fridge_door --no-autostart
./revolve test simulator              # T1 bring-up, T2 control, MoveIt smoke, T3 evaluator, T4 positive episodes,
                                      # reset, the datagen contract; -- -m "not slow" skips T4; -- -k contract
./revolve test common                 # specs and the shared library
./revolve validate                    # all specs
```

* Host: Docker with the NVIDIA runtime. The reference machine is a laptop with an 8 GB RTX 5060 (sm_120). A sim uses
  ~0.4 GB of GPU memory (EGL rendering of three 480 x 480 cameras at 30 Hz) and a few CPU cores (1 kHz physics +
  controllers, real time). Measure before running N pairs on one host; on the reference laptop we run one sim at a
  time. CPU load beyond what the host can sustain shows as a real-time factor below 1 and an evaluator rate below
  100 Hz (the result's `rate_hz`).
* The Python of `./revolve` is the repo's `.venv` (see `common/README.md`), without the host's ROS `PYTHONPATH`.
* Timings (reference laptop): bring-up to `ready` 10-20 s; reset 1.7-2.5 s; a successful scripted expert episode
  ~20-25 s sim time (approach ~12 s, grasp ~3 s, open ~4-5 s); an episode can last at most the task's `budget_s`
  (45 s for both tasks).
* The reference expert clients (MoveIt 2 through moveit_py + scripted base segments), each a full positive
  episode, are the simulator's T4 test scripts:
  `components/simulator/ros/revolve_sim/test/check_open_door.py` (mobile_panda, fridge) and
  `components/simulator/ros/revolve_sim/test/check_open_drawer.py` (mobile_so101_xl, drawer). The minimal datagen
  -style client is `check_contract.py` next to them.

## 12. Caveats and lessons learned

Physics and control:

* **Ramp base twists** (~0.5 s). Step changes kick the yaw.
* **The base creeps under sustained force** (MuJoCo omni rollers): ~1 mm/s at 50 N, 26-34 mm in 2 s at 150 N. Pushing
  or pulling hard with the arm while the base "stands" moves the base; re-measure the base pose (ground truth)
  instead of assuming it stayed. The stall push is ~364 N along x.
* After a teleport the base controller does not drive for ~2.5 s (only relevant to tests; datagen never teleports).
* **Parallel grippers do not self-centre.** The Panda hand is symmetric (mimic) and stiff: get the TCP onto the
  handle centre within ~1 mm before closing (small Pilz LIN corrections measured from tf + ground truth), or the
  grasp loses one finger. The SO-101's fixed jaw does not move at all: place the fixed jaw *into* the object by a
  few mm (the moving jaw alone gives one contact). On `mobile_so101_xl` + drawer, aim the TCP **3 mm in front of the
  bar centre** (at scale 2 a bar centred on the TCP sits only ~2.6 mm on the fixed tip pad and slipped off in 2 of 7
  runs).
* **Folded arms stick out ahead of the chassis**: the folded Panda reaches ~0.11 m in front of the chassis at door
  height (it touches the door before the chassis does); the folded XL jaws hang ~0.51 m ahead of `base_footprint`.
  Stop the drive-in early enough, or unfold first, or `robot_contacts` fails the episode.
* **A pattern that worked** (both T4 scripts): open-loop ramped drive to a base pose found by an IK search (rotate,
  then translate); MoveIt OMPL to a pre-grasp a few cm back along the approach; Pilz LIN onto the handle; small LINs
  until the TCP is within ~1 mm of the target; squeeze; then open. Fridge: Pilz CIRC about the hinge for the first
  part, then the base orbits the hinge with one constant body twist at <= 0.12 rad/s (0.2 rad/s caused finger
  contact gaps). Drawer: the base backs off along the slide (0.20 m at 0.05 m/s) with the arm holding.
* **Panda grasp orientation**: `open_fridge_door`'s `axes: [x, z] le 0.131` requires the hand's x up; the flipped
  hand (x down) has more joint margin but is rejected by the task. Horizontal grasps work best ~45 deg off the arm's
  forward axis.
* **SO-101 (5 DoF)**: use `pick_ik`. A horizontal approach with the jaws closing vertically is reachable only in the
  vertical plane through the shoulder-pan axis: pick the grasp point on the bar in front of the shoulder after the
  drive. Pilz LIN can exceed joint acceleration limits near a stretched elbow. OMPL warns "Cannot find planning
  configuration for group 'arm' using planner 'RRTConnectkConfigDefault'" (benign; it uses RRTConnect).
* Effort JTCs with integrators: never change controller gains live (`ros2 param set` made the arm violently
  unstable). Relaunch instead.

MoveIt / moveit_py:

* `moveit_py` segfaults at interpreter shutdown: end your process with `os._exit(0)` after flushing outputs.
* MoveItPy needs live `/joint_states` on sim time (`use_sim_time: true`) or the planning scene monitor fails to
  configure.
* A named plan-request parameter block must not be called like a pipeline (`ompl:` collides: "Planning plugin name
  is empty"); the shared `moveit_py.yaml` names them `ompl_rrtc`, `pilz_lin`, `pilz_circ`.
* After `RobotState.set_from_ik`, call `update()` before collision checks.
* Pilz CIRC: `path_constraints` named `center` with a position constraint point; start and goal must be equidistant
  from it.
* Build the planning scene from the scene's collision boxes in the base frame (from the ground-truth base pose); remove
  the boxes you intend to touch for the final LIN.

Timing and the evaluator:

* **Sim time restarts at 1 s after every reset.** Compare times within an episode only. Re-create anything that caches
  time (a tf2 buffer: `Buffer.clear()` also drops the static transforms, so make a new Buffer and listener).
* Contacts count for 50 ms after their message in the evaluator; use message stamps for your own contact logic.
* The evaluator's distance to a mesh collision is to the mesh's origin; handles are boxes (exact).
* `non_finite_commands` watches only the topics, not action goals.

Infrastructure:

* **The rare clock fault**: in ~2 % of launches, the controller manager inside gz never receives `/clock` and logs
  "No clock received, using time argument instead!" once a second; the robot then does not behave. One or two such
  lines at bring-up are a benign race; three or more = a broken sim: fail fast and relaunch (the simulator tests do
  this). The cause is not pinned yet.
* Host networking: everything on the host shares the network. Isolation comes from per-container `ROS_DOMAIN_ID`
  and `GZ_PARTITION` (= the container name). Always pass distinct `--name` and `--domain-id` per pair; the host's
  default domain 0 may be used by unrelated containers.
* Fast DDS: UDPv4 only in the sim; use the same in clients.
* A freshly created publisher loses its first messages before discovery completes: wait for a subscriber
  (`get_subscription_count()`) before relying on a one-shot message.
* GPU: a GPU in the "requires reset" state hangs sims; `revolve sim` refuses to start then (reboot the host;
  `REVOLVE_GPU_PREFLIGHT=0` skips the check).
* Known benign log lines per launch: one MuJoCo "Nan, Inf or huge value in CTRL ... Time = 0.0000" warning; one "does
  not support velocity limits" warning per joint; OBJ loader material warnings; Ogre2 camera visibility-mask
  warnings; gz "Found both JointVelocityReset and JointVelocityCmd components" per ros2_control joint at load and
  after each reset; rosbag2 "No input serialization format specified"; per reset "Moved backwards in time"
  (robot_state_publisher), "Detected jump back in time" (gz Contact) and a few TF_OLD_DATA lines from the evaluator.
* **Licence-bound assets**: `assets/` (PartNet-Mobility, ReplicaCAD: non-commercial) is git-ignored and made by
  `./revolve assets`. Never commit assets, never bake them into an image, and do not publish renders of the PartNet
  assets (camera images in bags included). Treat the asset files as untrusted downloads: read them, never execute
  anything from them.
* Never commit `runs/`, `cache/`, `.venv/`, `assets/` or `ThirdParty/*/install/`.

## 13. Rebasing an existing datagen branch

Expect a **large rebase**: the simulator, the specs and the CLI were rewritten underneath the old datagen. The old
datagen code in `components/datagen/` is yours to replace. When your v2 lands, delete what it replaces (the project
keeps no dead code): `generate.py`, `launch.py`, `src/datagen/*` (AutoMoMa / cuRobo planner entry points,
`specs_bridge.py`, `generator.yaml`, ...), and its tests, unless you reuse parts deliberately.

What moved or was deleted, and why your old imports break:

| Gone | Replaced by |
|---|---|
| `components/simulator/src/sim/` (robot model generator, `spec.py`, `sdf.py`, `replay.py`, `bag.py`, `capture.py`, `validation.py`, the gz Python systems: `automoma_replay_system.py`, `phase_replay_system.py`, `episode_system.py`, `scene_state.py`, `hold_check.py`, ...), `src/worlds/*.sdf`, `src/egl/` | `components/simulator/ros/revolve_sim/` (`world.py`: scene -> world SDF, `robot.py`, `evaluator.py`, `episode.py`, `metrics.py`, `reset.py`, `moveit.py`, `launch/sim.launch.py`, `config/`). Nothing there is a library for datagen except `moveit.py` and the specs. |
| `components/simulator/launch.py` (the simulator HTTP job: `replay` jobs), `components/simulator/ros/Dockerfile` | `./revolve sim` (section 4.1). |
| `common/revolve_common/metrics.py` (+ its test) | The evaluator's `revolve_sim/metrics.py` (internal) and the metric registry in `specs.METRICS`. |
| The old schemas in `common/revolve_common/specs.py` (experiments, task start blocks, planner fields `detailed`/`skip`/`floor`, contracts) | The new `specs.py` (section 9). |
| `specs/experiments/*.yaml`, `specs/robots/*/contract.yaml` | Nothing yet (the manager is broken). Your job YAML (section 8.2) carries task, robot, scenes. |
| `ThirdParty/franka_description` | `moveit_resources_panda_description` (ROS binaries). |
| `components/simulator/src/sim/robots/`, `model.py`, the custom controller, `gen_camera_model.py`, old robot YAMLs | `specs/robots/<r>/` (xacro + controllers.yaml + robot.yaml + moveit/). |
| `revolve run / resume / status / stop` | Removed with the manager for now. |
| `components/simulator/assets_pipeline/` | `assets_pipeline/` at the repo root (`./revolve assets`, its own image). |

Still there and usable: `revolve_common.job`, `revolve_common.wrapper`, `revolve_common.gpu`,
`revolve_common.containers`, `revolve_common.policy_io`, `revolve_common.specs`; the `ThirdParty/automoma`,
`ThirdParty/curobo` and `ThirdParty/rosetta` submodules; `./revolve build datagen` (the old image: CUDA 12.8,
Python 3.11, no ROS 2: section 4.5 explains what the v2 client needs).

Rebase tips: take the simulator, `common/` and `specs/` from this branch as they are (do not resolve conflicts in
favour of the old versions); keep only your datagen changes; then port datagen to the contract here. Submodules:
`git submodule update --init` (never `--recursive` or `--remote`). After the rebase: `./revolve build simulator`,
`./revolve test simulator -- -k contract` (the interface you build on), `./revolve validate`.

## 14. Definition of done for datagen v2

- [ ] `components/datagen/` contains only v2 code: generator, randomizer (+ its config), `mock_manager.py`, tests,
      README; the old datagen code is deleted.
- [ ] The randomizer writes scene instances that pass `./revolve validate <instance>` and start collision-free.
- [ ] `mock_manager.py --workers N` (at least N = 1; N = 2 where the host allows) runs the randomizer, N
      (sim, generator) pairs with distinct `--name` / `--domain-id`, and merges one valid `manifest.json` (schema in
      section 8.4, every path existing).
- [ ] Every episode goes through the contract only: `--no-autostart`, readiness via `/evaluator/state`,
      `/evaluator/start`, `/simulator/reset` between episodes, relaunch per scene instance, `docker rm -f` teardown,
      no teleports, no bags written by datagen.
- [ ] Episodes pass the evaluator (`passed: true`) for both tasks (`open_fridge_door` on `mobile_panda`,
      `open_drawer` on `mobile_so101_xl`) at a success rate you report; failed attempts are kept with
      `passed: false`.
- [ ] Every bag has the frozen topics (section 7.2), cameras JPEG at ~30 Hz, `/tf_static`, and a 30 Hz action series
      can be extracted from it (section 7.4).
- [ ] All timing in sim time; works at real-time factor 1.
- [ ] Tests: `./revolve test datagen` runs datagen's unit tests and one end-to-end `mock_manager.py` run on a real
      sim; `./revolve test simulator -- -k contract` still passes.
- [ ] No assets, renders of PartNet assets, `runs/` content or credentials committed.

## 15. File map

| Path | What |
|---|---|
| `revolve` | The CLI (`assets`, `build`, `test`, `sim`, `validate`). |
| `components/simulator/README.md` | Simulator details (launch, evaluator, reset, tests). |
| `components/simulator/ros/revolve_sim/config/record.yaml` | The frozen recording topic list. |
| `components/simulator/ros/revolve_sim/config/bridge.yaml` | The gz <-> ROS bridge entries (the topics in section 6). |
| `components/simulator/ros/revolve_sim/launch/sim.launch.py` | What `revolve sim` runs inside the container. |
| `components/simulator/ros/revolve_sim/revolve_sim/evaluator.py`, `reset.py` | The evaluator and the reset service. |
| `components/simulator/ros/revolve_sim/revolve_sim/moveit.py` | MoveIt 2 parameters for a robot. |
| `components/simulator/ros/revolve_sim/test/test_contract.py`, `check_contract.py` | The contract test: a datagen-like client. |
| `components/simulator/ros/revolve_sim/test/check_open_door.py`, `check_open_drawer.py` | Reference expert clients (MoveIt). |
| `components/simulator/ros/revolve_sim/test/check_evaluator.py`, `check_control.py` | Client helpers (EvalProbe, Probe: ramped drive, goals). |
| `common/revolve_common/specs.py` | Spec schemas, metric registry, loader (scene instances included). |
| `specs/scenes/`, `specs/robots/`, `specs/tasks/` | Scenes, robots (+ MoveIt configs), tasks + `sanity.yaml`. |
| `assets_pipeline/` | The one-time asset download and conversion. |
