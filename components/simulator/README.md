# simulator
Gazebo Kura (gz-sim 11 / gz-physics 10 main) + MuJoCo physics on an upstream ROS 2 Lyrical stack (ros_gz,
gz_ros2_control, ros2_controllers, MoveIt 2). The world, the robot and the task come from the YAML specs:
`specs/scenes/<s>/scene.yaml`, `specs/robots/<r>/`, `specs/tasks/<t>/task.yaml` + `specs/tasks/sanity.yaml`
(format and checks: `common/revolve_common/specs.py`, `./revolve validate`).

Status: the world, the robot and its controllers run (`ros/revolve_sim`); with `--task`, the evaluator scores
episodes live (phases, metrics, sanity checks) and records them; `/simulator/reset` puts everything back to the
scene's start for the next episode.

## Run
```
./revolve sim --scene kitchen_partnet --robot mobile_panda [--task open_fridge_door [--no-autostart]
              [--record images|no-images] [--out DIR]] [--gui] [--name N] [--domain-id D]
./revolve sim --scene kitchen_partnet --robot mobile_so101        # or mobile_so101_xl
./revolve sim --scene runs/exp/scene_7/scene.yaml --robot mobile_panda --task open_fridge_door   # a scene instance
./revolve validate runs/exp/scene_7/scene.yaml --robot mobile_panda   # check an instance before running it
```
`./revolve sim` first checks the scene, the robot and the task as `./revolve validate` does (with the asset files),
and refuses what the launch would refuse before it starts docker (exit 1, the errors on stderr). It then runs in the
foreground until the sim stops. It prints
`revolve sim: ROS_DOMAIN_ID=<D> (from the host: export ROS_DOMAIN_ID=<D>)` on stderr first; export that to use ROS 2
from the host. Stop it with `docker rm -f <name>`. The datagen contract (how a client drives episodes,
what is recorded): `components/datagen/DATAGEN-v2.md`.

The sim runs the installed copy of `ros/revolve_sim` in `ThirdParty/ros_gz_ws/install`, not the source tree: after
editing `revolve_sim` (evaluator, launch, world, reset, configs), run `./revolve build simulator` again, or the sim
runs the old code without a warning. The tests' check scripts run from the source tree, so a stale install can make
new checks run against an old evaluator.

### Inputs
| Input | Type | Default | Constraint |
|---|---|---|---|
| `--scene S` | a scene name, or a path ending in `.yaml` / `.yml` | required | A name: `specs/scenes/<S>/scene.yaml`. A path: a scene instance, a file named `scene.yaml` outside `specs/` (e.g. one a randomizer wrote per seed, with another `robot_start`); any other file is refused. An instance keeps the name of the scene it was made from, and its `assets` are relative to the repo. A file inside the repo is used through the repo mount, one outside is mounted read-only at its own path. With `--task`, its `name` must be the task's `scene`. |
| `--robot R` | a robot name | required | `specs/robots/<R>/`. The scene's `robot_start.arm` must be one of its poses or a list of exactly its arm joint count, else it is refused (`robot_start.arm` in the message). |
| `--task T` | a task name | none: no evaluator, no recorder, no episodes | `specs/tasks/<T>/task.yaml`, on the scene's name, checked against the scene (an instance too) and the robot. |
| `--no-autostart` | flag | off: the first episode starts once every input is live | Needs `--task` (refused without it, exit 1). Episodes then start only on `/evaluator/start`. |
| `--record` | `images` or `no-images` | `images` | Needs `--task` (refused without it, exit 1). The episode bags with the three cameras (JPEG) or without them. |
| `--out DIR` | a host directory | `runs/sim/<time>_<task>/`, the time in UTC (the container's clock) | Needs `--task` (refused without it, exit 1). Created if missing, mounted writable at the same absolute path, so the paths in a result are host paths. |
| `--gui` | flag | off | Needs `DISPLAY` and `XAUTHORITY` (the host's X / Xwayland). Costs real-time factor. |
| `--name N` | a container name | `revolve-sim` | Unique per running sim (a clash: docker exits 125). It is also the sim's `GZ_PARTITION`. |
| `--domain-id D` | integer 0..232 | from the name: 1..101, skipping the domains of the other running containers | The sim's `ROS_DOMAIN_ID`. A value another container uses gives a warning only. On Linux, ROS 2 documents 0..101 and 215..232 as safe: in between, the DDS ports can fall into the kernel's ephemeral port range. |
| scene, robot, task, sanity files | YAML (`specs.py`) | | Unknown keys, duplicate keys and NaN / Inf numbers are errors. The scene name, its model names and the robot name become ROS 2 topic names: letters, digits and `_`, not starting with a digit. A model joint's initial value must be 0 for now (gz-sim has no upstream way to start a joint away from its SDF zero): `./revolve validate` and `./revolve sim` refuse other values. |
| assets | `assets/` (git-ignored) | | The scene's `assets` search path, made once by `./revolve assets`. |
| builds | `ThirdParty/gz_kura/install`, `ThirdParty/ros_gz_ws/install` | | From `./revolve build simulator` (refused without them). |
| `REVOLVE_SIM_IMAGE` | env, an image tag | `revolve-simulator:latest` | The image `./revolve sim` runs and `./revolve build simulator` builds. |
| `REVOLVE_GPU_PREFLIGHT` | env | `1` | `0` turns off the host GPU check; on, a GPU in a broken state ("GPU requires reset") is refused with exit 1. |

`./revolve validate <file>` checks a scene instance on its own: the schema, its models in the asset search path and
the initial joint values. Add `--robot R` to also check its `robot_start` against robot R, as `./revolve sim` does.

### Outputs
| Output | Where | What |
|---|---|---|
| episode result | `<run dir>/episode_<n>.json` and `/evaluator/result` | One per episode (Evaluator below). The file is written last: once it exists, the episode's files are complete. |
| episode bag | `<run dir>/episode_<n>_bag/` | rosbag2, mcap with zstd chunks, sim time; closed (with its `metadata.yaml`) before the result file. Null `bag` in the result if the recorder failed. |
| gz log copy | `<run dir>/episode_<n>_gz.log` | The episode's part of the gz server's console log, from its start to its end (the log itself is in the container, gone after `docker rm -f`). Empty if gz wrote nothing meanwhile. |
| episode numbers | | Continue after the highest n of any `episode_<n>` file in the run dir, a bag without a result too (a run killed mid-episode), so a relaunch into the same dir overwrites nothing. An existing bag dir is never recorded into. |
| ground truth | `/model/<robot>/pose`, `/model/<m>/pose`, `/world/<w>/model/<m>/joint_state` | tf2_msgs/TFMessage (the model in the world, a link as `<m>/<link>` in its model), sensor_msgs/JointState for each articulated model; 100 Hz. |
| robot state | `/joint_states`, `/tf`, `/tf_static`, `/robot_description` | joint_state_broadcaster; robot_state_publisher at 100 Hz + the omni odometry; the URDF (transient local). |
| contacts | `/world/<w>/model/<robot>/link/<l>/sensor/contact/contact` | ros_gz_interfaces/Contacts, every robot link with a collision except the omni rollers, only while in contact. |
| cameras | `/<camera>/image`, `/<camera>/image/compressed`, `/<camera>/camera_info` | 30 Hz, 480 x 480, cameras `wrist_cam`, `overhead_left_cam`, `overhead_right_cam`. Raw images are 0.7 MB over UDP: a reliable subscriber got every frame, a best-effort one lost 2.5 % (585 of 600). The JPEG topic is the lighter input. |
| controllers | `/<jtc>/follow_joint_trajectory` (action), `/<jtc>/joint_trajectory`, `/<jtc>/controller_state`, `/omni_wheel_drive_controller/cmd_vel`, `.../odom` | Commands in, state out (Run notes below). |
| evaluator | `/evaluator/state`, `/evaluator/result`, `/evaluator/start` (std_srvs/Trigger) | With `--task` (Evaluator below). |
| task prompt | `/task/prompt` | With `--task`: std_msgs/String, transient local, task.yaml's `prompt`, published once by the evaluator (so every episode bag has it once; the Rosetta contracts' `tasks` channel, `components/dataset/DATASET-v2.md`); also in each bag's `metadata.yaml` as `custom_data` `lerobot.operator_prompt` (the recorder's `--custom-data`: Rosetta's porter takes it as the task of the frames before the first `/task/prompt` message). |
| reset | `/simulator/reset` (std_srvs/Trigger) | Always (Reset below). |
| exit code | | Runs until stopped. 1: a refused argument, a missing scene file or build, a broken GPU, or a launch error such as a refused spec (one line on stderr, or the launch's `[ERROR] [launch]` line). 2: a usage error (unknown flag or value). 125: the container name is taken. 137: stopped by `docker rm -f`. |

One container (`--name`) on the GPU and the host network, isolated from other ROS 2 / gz instances on the host by
its own `ROS_DOMAIN_ID` and `GZ_PARTITION`, the repo mounted read-only at `/repo` (`runs/` writable). ROS 2 in the
container uses Fast DDS over UDPv4 only (`FASTDDS_BUILTIN_TRANSPORTS=UDPv4`): no shared-memory segments, which a
killed container (`docker rm -f`) would leave in the host's /dev/shm (`--ipc host`); clients on the host network
reach it over UDP. Robots (`specs/robots/`): mobile_panda (the moveit_resources Panda), mobile_so101 (the SO-101
arm) and mobile_so101_xl (the same arm at scale 2 on a lower chassis), all on the shared omni base
(`common/omni_base.urdf.xacro`); `common/so101.urdf.xacro` is the SO-101 macro (ported from ThirdParty/so_arm100
so101_new_calib.urdf, `scale` arg, box collisions, its wrist camera and contact sensors), `common/sensors.urdf.xacro`
the camera and contact sensor macros. The container runs `ros2 launch revolve_sim sim.launch.py`, the upstream
ros_gz / gz_ros2_control layout:
- `revolve_sim/world.py`: scene.yaml -> world SDF (`python3 -m revolve_sim.world <scene>` prints it): the models as
  `model://` includes at their poses (search path = the scene's `assets`), a JointStatePublisher on every
  articulated model, a PosePublisher on every model (`/model/<m>/pose`: the model in the world, its links in the
  model), ground plane, sun, ambient light, 1 ms MuJoCo physics, the Sensors (ogre2), Contact,
  SceneBroadcaster and UserCommands systems, and the robot. The world is named after the scene.
- the robot: `specs/robots/<r>/robot.urdf.xacro` (xacro args `initial_positions` = the scene's `robot_start.arm`
  pose, `controllers` = the robot's controllers.yaml), included in the world SDF (its URDF file) at
  `robot_start.base`, so it is part of the world's initial state (a gz world reset removes entities spawned
  later), robot_state_publisher, one controller spawner for every controller in controllers.yaml (gz_ros2_control
  loads them into its controller manager, which runs every physics step). Sim time starts at 1 s (`gz sim
  --initial-sim-time 1`: ROS time 0 means "no clock"), also after a reset.
- `config/bridge.yaml` (ros_gz_bridge, filled in for the world, robot, cameras, contact links, articulated models):
  `/clock`, the ground-truth poses and scene joint states, the contacts, `/<camera>/camera_info`; the service
  `/world/<w>/control` (ros_gz_interfaces/srv/ControlWorld, used by the reset); camera images through
  ros_gz_image: `/<camera>/image` (and `/<camera>/image/compressed` through image_transport).
- Commands: `/omni_wheel_drive_controller/cmd_vel` (geometry_msgs/TwistStamped, robot frame; ramp changes over
  ~0.5 s, the controller has no limiter), FollowJointTrajectory on `/<arm controller>/follow_joint_trajectory`
  and the hand controller (mobile_panda: panda_arm_controller, panda_hand_controller; mobile_so101 and
  mobile_so101_xl: arm_controller, gripper_controller).
- MoveIt 2: a robot's MoveIt config lives with its description in `specs/robots/<r>/moveit/` (mobile_panda: the
  moveit_resources Panda SRDF groups + the omni base's disabled collisions, no virtual joint so MoveIt plans in
  the base frame; moveit_controllers.yaml: arm and hand as FollowJointTrajectory; moveit_py.yaml: the planning
  scene monitor, the OMPL and Pilz pipelines, named plan request parameters, shared by every robot:
  `common/moveit_py.yaml`). The SO-101 robots: `common/so101.srdf.xacro` (groups `arm`, `gripper`, states, disabled
  collisions) + `common/omni_base.srdf.xacro` in the robot's SRDF, and the shared `common/so101_moveit/` files
  (kinematics: pick_ik, as the arm has 5 DoF; joint limits; Pilz Cartesian limits; MoveIt controllers) linked into
  the robot's moveit/. `revolve_sim.moveit.moveit_params(repo, robot)` builds the parameters (kinematics, joint
  limits, Pilz limits from the robot's moveit/ if there, else with the pipeline configs from
  moveit_resources_panda_moveit_config unchanged) for `MoveItPy(config_dict=...)` or a move_group node. Nothing in
  the sim launch uses MoveIt: a client (the MoveIt smoke and T4 tests) runs it next to the sim.
- Contact sensors on every robot link with a collision except the omni rollers (the robot xacro; gz link and
  collision names after sdformat's fixed-joint lumping). robot_state_publisher publishes the moving links' tf at
  100 Hz.
- Env: `GZ_MUJOCO_NOSLIP_ITERATIONS=10` (grasps and the omni base hold better), `GZ_SIM_RESOURCE_PATH` = the assets.
- `--gui`: a `gz sim -g` client on the host's X display (DISPLAY + XAUTHORITY); it costs RTF.

## Evaluator (`--task`)
`revolve_sim/evaluator.py` (node `/evaluator`, sim time, 100 Hz) evaluates the task (`specs.resolve` on the scene
the sim runs, an instance too: the task's phases + the merged `tasks/sanity.yaml` checks) with
`revolve_sim/episode.py` (phase rules) and `revolve_sim/metrics.py` (the metrics of `specs.METRICS`, numpy).
Inputs, all upstream: robot links = ground-truth base pose x robot_state_publisher tf; scene links = the
PosePublishers; scene joints = the JointStatePublishers; contacts = the robot's contact sensors (depth = MuJoCo
penetration; the ground is never a contact partner, but counts for penetration); commands = non-finite messages on
the controllers' command topics (`<omni>/cmd_vel`, `<jtc>/joint_trajectory`; FollowJointTrajectory goals are not
seen); gz log = `[error]` / `[critical]` lines of the gz server's `~/.gz/sim/log/<time>/server_console.log` in the
container. Each tick takes the poses and contacts at one common stamp (the oldest of the newest gz pose and tf data;
tf interpolated, gz poses from a 2 s history, contacts within 50 ms before it): under load the node's callback
queues fall behind by tens of ms, and mixing inputs of different ages showed slip, speed and lost contacts that
were only these offsets. Distance to a collision is to its box / sphere / cylinder / capsule (a mesh: to its
origin). An episode counts only the non-finite commands and gz log lines since its start.
- Episodes: the first starts once every input is live (autostart), or on `/evaluator/start` (std_srvs/Trigger,
  reply `episode <n>`) with `--no-autostart` (`ros2 service call /evaluator/start std_srvs/srv/Trigger`; one at a
  time, again after the previous one ended; refused with `inputs not ready: ...` or `episode <n> is running`).
  Phases: all `done` true (each for its `for_s`) -> next phase; a `while` false, the phase's `timeout_s` or the
  episode's `budget_s` -> the phase fails; a timed sanity check fails the active phase, an untimed one the episode
  as phase `global`. A world reset (`/simulator/reset`) during an episode ends it with `reason: reset`; the active
  phase then has status `failed` and is the result's `failed_phase`, like a real failure, so count resets by their
  `reason`.
- `/evaluator/state` (std_msgs/String JSON, 10 Hz): `t`, `ready`, `resets` (the world resets it has seen),
  `episode`, `phase`, `values`: every condition's current value, keyed `<phase>.done[i]`, `<phase>.while[i]`,
  `sanity.<check>`, also between episodes. A value that cannot be computed is null: before the inputs are live, and
  a `slip` condition outside its phase (slip is the drift since its phase started).
- `/evaluator/result` (std_msgs/String JSON, transient local) and `<run dir>/episode_<n>.json`: `task`, `scene`,
  `robot`, `episode`, `sim_time_start`, `passed`, `failed_phase`, `reason` (`timeout_s`, `budget_s`, `while`,
  `sanity`, `reset`), `check`, `condition`, `value`, `t_fail` (s since the episode start), `duration_s`, `phases`
  [{`name`, `status` (passed / failed / aborted / not_reached), `t_start`, `t_end`, `reason`, `last` (the last
  values)}], `sanity_last`, `contacts` (at the end), `rate_hz`, `tick_ms`, `gz_log` (the copy in the run dir),
  `result_file`, `bag`. The result is published first (so it is in the bag), then the bag is closed, then the file
  is written.
- Bags: one per episode, `<run dir>/episode_<n>_bag/`. The topics: `config/record.yaml`, the frozen list (clock,
  robot_description, tf, tf_static, joint states, each JTC's controller_state and joint_trajectory, the omni
  cmd_vel and odom, the ground-truth poses, the scene joint states, the contacts, camera_info, the evaluator's
  state and result, `/task/prompt`; with `--record images` also `/<camera>/image/compressed`, JPEG from
  compressed_image_transport through ros_gz_image's image_transport publisher, 30 Hz). The launch starts `ros2 bag
  record` paused into a scratch bag in the container; the evaluator stops it and, per episode, calls its `record`
  service with the episode's bag path and `stop` at the end: each recording subscribes anew, so the transient-local
  /tf_static, /robot_description and /task/prompt are in every bag (a QoS override takes /evaluator/result live
  only). Topics with messages only on demand (contacts, commands) are missing from a bag that saw no publisher.
- Launch args: `autostart`, `run_dir`, `record` (images, no-images, none) (`ros2 launch revolve_sim sim.launch.py ...
  task:=<t>`).

## Reset (client contract)
Episodes back to back on one sim (datagen, recovery, evaluation) go through one service, `/simulator/reset`
(std_srvs/Trigger, `revolve_sim/reset.py`, node `/simulator`, always running). It returns once the sim is back at
the scene's start; `success: false` + `message` if a step failed or timed out (then relaunch the sim):
1. the active controllers are deactivated (`/controller_manager/switch_controller`);
2. the gz world reset, `/world/<w>/control` with `reset: {all: true}` (the request ROS 2 Simulation Interfaces
   `ResetSimulation` sends; `ros_gz_sim`'s implementation lives in its `gzserver` node only, not in `gz sim`):
   every model back to its initial pose and joint positions, the robot included, sim time back to 1 s. The
   gz_ros2_control plugin keeps its controller manager across it (`ThirdParty/patches/gz_ros2_control`);
3. once `/clock` has jumped back (+ 50 ms), the controllers are activated again, so they latch onto the reset
   state; the omni base controller gets one zero twist;
4. each joint trajectory controller drives its joints to their ros2_control initial values (the scene's
   `robot_start.arm`, the gripper's start) over 1 s; the reset waits until they stay within 1 deg / 1 mm for 0.5 s;
5. with a task: it waits until the evaluator has seen the reset (`resets` in `/evaluator/state` goes up). A running
   episode ends with `reason: reset` (its result JSON and bag are written as usual; the bag can end with up to
   ~0.2 s of messages after the jump back in sim time). The evaluator drops every input for 0.5 s of sim time,
   then is ready again once its inputs are live.

Client sequence per episode: read `resets` from `/evaluator/state`, call `/simulator/reset`, wait for an
`/evaluator/state` with `resets` above that value and `ready: true`, call `/evaluator/start`, command the robot,
wait for `/evaluator/result`, then wait for `episode_<n>.json` before reading the bag or tearing the sim down.
Sim time restarts at 1 s on every reset: compare times within an episode only (an `/evaluator/state` `t` is not
unique across episodes). A new scene instance is a new sim (relaunch). Typical reset: ~1.7 s wall.

Known log lines: one MuJoCo "Nan, Inf or huge value in CTRL ... Time = 0.0000" warning at load; one "does not
support velocity limits" warning per joint with a URDF velocity limit (MuJoCo has none; ros2_control needs them);
gz "Found both JointVelocityReset and JointVelocityCmd components" once per ros2_control joint at load and after
each reset; per reset, "Moved backwards in time" (robot_state_publisher), "Detected jump back in time" (gz
Contact) and a few TF_OLD_DATA lines from the evaluator before it notices the jump.

## Build and test
```
./revolve build simulator [--clean]   # image revolve-simulator (deps only), then Kura, ros_gz, gz_ros2_control,
                                      # revolve_sim
./revolve test simulator              # T1 bring-up, T2 control (every robot), MoveIt smoke (SO-101 robots), T3
                                      # evaluator and T4 (MoveIt) for open_fridge_door (mobile_panda) and open_drawer
                                      # (mobile_so101_xl) on a launched sim, the reset loop, the datagen contract,
                                      # the --out dir, a scene instance launch + the host-only tests
                                      # (ros/revolve_sim/test); -- -m "not slow" skips T4
```
- `Dockerfile`: dependencies only, Ubuntu 26.04 + ROS 2 Lyrical binaries (ros2_control, ros2_controllers, MoveIt 2 +
  moveit_py + the moveit_resources Panda + pick_ik, xacro, robot_state_publisher, rosbag2, launch_testing) + Kura's
  apt build deps. Never the Lyrical ros-gz / gz-ros2-control binaries (built on Gazebo Jetty). Every bash in the
  image sources `/opt/revolve_env.sh`: ROS, then the ros_gz workspace, then Kura first on PATH / LD_LIBRARY_PATH /
  CMAKE_PREFIX_PATH (the binaries pull Jetty sdformat / gz-math / gz-tools vendors into /opt/ros/lyrical). Apt
  python3-pydantic, so `revolve_common.specs` imports in the image (PYTHONPATH /repo/common).
- `build.sh`, with colcon inside the image: (1) `ThirdParty/gz_kura/fetch.sh` (kura.repos pins + ThirdParty/patches)
  and Kura into `ThirdParty/gz_kura/install`, mounted at `/opt/gz_ws`; (2) the submodules ThirdParty/ros_gz,
  ThirdParty/gz_ros2_control (+ demos; ThirdParty/patches/gz_ros2_control applied in its working tree first) and
  ThirdParty/gz_vendors/* (gz_*_vendor, all satisfied by the Kura install) and `ros/revolve_sim` (ament_python,
  installed by copy: rebuild after changing it) into `ThirdParty/ros_gz_ws/install`, mounted at
  `/opt/ros_gz_ws/install`. Both git-ignored (build/, log/, .ccache/ too), mounted at runtime at the paths they are
  built at. Rebuilds are incremental; missing submodules are initialised, checked-out ones are left alone.
- Assets (PartNet-Mobility, ReplicaCAD: non-commercial) are read in place from `assets/` (git-ignored, repo root)
  through the repo mount; never copied into git or the image. They are made once by `./revolve assets`
  (`assets_pipeline/`, its own image; see its README).
- Tests (`ros/revolve_sim/test`, host pytest): each module starts a fresh `revolve sim` container and runs a
  check script in it (`check_bringup.py`, `check_control.py`, `check_moveit.py`, `check_evaluator.py`,
  `check_open_door.py`, `check_open_drawer.py`, `check_reset.py`, rclpy; each prints one RESULT JSON line). The
  check scripts read an episode's result file only once it exists (the evaluator writes it after closing the bag).
  A module runs once per robot of its `ROBOTS` (T1 / T2: `robot_cases.ROBOTS`, with the per-robot T2 targets in
  `robot_cases.CONTROL`; the MoveIt smoke: the robots and TCP pose targets of `robot_cases.MOVEIT`; T3 / T4 / reset:
  each robot's task, push and T4 script in `robot_cases.TASKS`). T1 measures each camera's render rate on
  `/<camera>/camera_info` over 5 s of sim time and checks the size and content of a few raw images (raw-image
  delivery to a fresh subscriber can stall for 1-2 s under host load). Reset (`test_reset.py`): 5 resets between 6
  episodes in one sim (moving episodes reset mid-episode, idle ones after their result), each checked against the
  start (base 1 cm / 1 deg, arm 1 deg, scene joints, controllers active, evaluator ready, a T2 arm move after it,
  one result + bag per episode, memory / processes stable); T4 ends with one reset from the opened door / drawer.
  Contract (`test_contract.py`, `check_contract.py`): the datagen interface as datagen uses it: a scene instance in
  `--out`, the sim from the CLI with `--record images --out --domain-id`, the client in its own container (host
  network, the sim's domain), two episodes (ramped twist, FollowJointTrajectory goals, streamed joint_trajectory
  points) with a reset between, then the files: two results + two closed bags with every frozen topic, cameras JPEG
  at ~30 Hz, /tf_static, and a 30 Hz action series from each bag; then the robot's Rosetta contract against both bags
  with Rosetta's own loader and decoders (`check_rosetta.py`, in a throwaway container from the simulator image with
  Rosetta's two pure-Python packages pip-installed from `ThirdParty/rosetta` into its /tmp: every contract topic with
  messages and the contract's type, no zero header stamp on a header-timeline source, decodable at the contract's
  widths and image size, `/task/prompt` once and in the bag metadata). `test_rosetta_contracts.py` (no sim): every
  robot's contract loads with Rosetta's loader and has the agreed streams. `test_out_dir.py`: a relaunch into an
  `--out` that a killed episode left (a bag without a result): the next episode number, its own closed bag, host
  paths, the gz log copy, every condition in `/evaluator/state`. `test_launch_refusals.py`: a valid scene instance starts, and
  the launch itself (the host check bypassed) refuses a `robot_start.arm` that does not fit the robot. The other
  modules record `--record no-images`. No sim: `test_episode_unit.py` (phase rules, metrics),
  `test_episode_numbers.py`, `test_cli.py` (`./revolve sim` refusals before docker and `./revolve validate`, with a
  fake docker; the cases that pass the checks need the assets and the builds).
  A launch whose controller manager logs "No clock received" once a second (it never got /clock; rare, cause not
  pinned) fails at once with that message (conftest.py; no retry).
  `REVOLVE_TEST_ROBOT=<r>[,<r>]` keeps only those. Logs: `runs/test_simulator/<time>/<module>/<robot>/` or
  `$REVOLVE_TEST_LOGS`.
