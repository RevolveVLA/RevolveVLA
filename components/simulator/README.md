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
./revolve sim --scene kitchen_partnet --robot mobile_panda [--task open_fridge_door [--no-autostart]] [--gui]
./revolve sim --scene kitchen_partnet --robot mobile_so101        # or mobile_so101_xl
```
Robots (`specs/robots/`): mobile_panda (the moveit_resources Panda), mobile_so101 (the SO-101 arm) and
mobile_so101_xl (the same arm at scale 2 on a lower chassis), all on the shared omni base
(`common/omni_base.urdf.xacro`); `common/so101.urdf.xacro` is the SO-101 macro (ported from ThirdParty/so_arm100
so101_new_calib.urdf, `scale` arg, box collisions, its wrist camera and contact sensors), `common/sensors.urdf.xacro`
the camera and contact sensor macros.
One container (`--name`, default `revolve-sim`) on the GPU and the host network, isolated from other ROS 2 / gz
instances on the host: its own `ROS_DOMAIN_ID` (`--domain-id`, default derived from the container name, 1..101,
skipping the domains of the other running containers; printed at start: export it to use ROS 2 from the host) and
its own `GZ_PARTITION` (the container name), the repo mounted read-only at `/repo` (`runs/` writable). ROS 2 in
the container uses Fast DDS over UDPv4 only (`FASTDDS_BUILTIN_TRANSPORTS=UDPv4`): no shared-memory segments, which
a killed container (`docker rm -f`) would leave in the host's /dev/shm (`--ipc host`); clients on the host network
reach it over UDP as before. It runs `ros2 launch revolve_sim
sim.launch.py`, the upstream ros_gz / gz_ros2_control layout:
- `revolve_sim/world.py`: scene.yaml -> world SDF (`python3 -m revolve_sim.world <scene>` prints it): the models as
  `model://` includes at their poses (search path = the scene's `assets`), a JointStatePublisher on every
  articulated model, a PosePublisher on every model (`/model/<m>/pose`: the model in the world, its links in the
  model), ground plane, sun, ambient light, 1 ms MuJoCo physics, the Sensors (ogre2), Contact,
  SceneBroadcaster and UserCommands systems, and the robot. The world is named after the scene. Initial joint
  positions other than 0 are rejected for now (gz-sim has no upstream way to start a joint away from its SDF zero).
- the robot: `specs/robots/<r>/robot.urdf.xacro` (xacro args `initial_positions` = the scene's `robot_start.arm`
  pose, `controllers` = the robot's controllers.yaml), included in the world SDF (its URDF file) at
  `robot_start.base`, so it is part of the world's initial state (a gz world reset removes entities spawned
  later), robot_state_publisher, one controller spawner for every controller in controllers.yaml (gz_ros2_control
  loads them into its controller manager, which runs every physics step). Sim time starts at 1 s (`gz sim
  --initial-sim-time 1`: ROS time 0 means "no clock"), also after a reset.
- `config/bridge.yaml` (ros_gz_bridge, filled in for the world, robot, cameras, contact links, articulated models):
  `/clock`, `/model/<robot>/pose` and `/model/<m>/pose` (ground truth, tf2_msgs/TFMessage; child_frame_id
  `<m>/<link>` for a link),
  `/world/<w>/model/<m>/joint_state`, `/world/<w>/model/<robot>/link/<l>/sensor/contact/contact`
  (ros_gz_interfaces/Contacts), `/<camera>/camera_info`; the service `/world/<w>/control`
  (ros_gz_interfaces/srv/ControlWorld, used by the reset); camera images through ros_gz_image: `/<camera>/image`.
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
`revolve_sim/evaluator.py` (node `/evaluator`, sim time, 100 Hz) evaluates the task (`specs.resolve`: the task's
phases + the merged `tasks/sanity.yaml` checks) with `revolve_sim/episode.py` (phase rules) and
`revolve_sim/metrics.py` (the metrics of `specs.METRICS`, numpy). Inputs, all upstream: robot links = ground-truth
base pose x robot_state_publisher tf; scene links = the PosePublishers; scene joints = the JointStatePublishers;
contacts = the robot's contact sensors (depth = MuJoCo penetration; the ground is never a contact partner, but
counts for penetration); commands = non-finite messages on the controllers' command topics (`<omni>/cmd_vel`,
`<jtc>/joint_trajectory`; FollowJointTrajectory goals are not seen); gz log = `[error]` / `[critical]` lines of
the gz server's `~/.gz/sim/log/<time>/server_console.log`. Each tick takes the poses and contacts at one common
stamp (the oldest of the newest gz pose and tf data; tf interpolated, gz poses from a 2 s history, contacts within
50 ms before it): under load the node's callback queues fall behind by tens of ms, and mixing inputs of different
ages showed slip, speed and lost contacts that were only these offsets. Distance to a collision is to its box / sphere /
cylinder / capsule (a mesh: to its origin). An episode counts only the non-finite commands and gz log lines since its
start.
- Episodes: the first starts once every input is live (autostart), or on `/evaluator/start` (std_srvs/Trigger)
  with `--no-autostart` (`ros2 service call /evaluator/start std_srvs/srv/Trigger`; one at a time, again after
  the previous one ended). Phases: all `done` true (each for its `for_s`) -> next phase; a `while` false, the
  phase's `timeout_s` or the episode's `budget_s` -> the phase fails; a timed sanity check fails the active
  phase, an untimed one the episode as phase `global`.
- `/evaluator/state` (std_msgs/String JSON, 10 Hz): `t`, `ready`, `resets` (the world resets it has seen),
  `episode`, `phase`, `values` (every condition's current value, keyed `<phase>.done[i]`, `<phase>.while[i]`,
  `sanity.<check>`; also between episodes).
- `/evaluator/result` (std_msgs/String JSON, transient local) and `runs/sim/<time>_<task>/episode_<n>.json`:
  `passed`, `failed_phase`, `reason` (`timeout_s`, `budget_s`, `while`, `sanity`, `reset`), `check`, `condition`,
  `value`,
  `t_fail` (s since the episode start), `duration_s`, `phases` [{`name`, `status` (passed / failed / aborted /
  not_reached), `t_start`, `t_end`, `reason`, `last` (the last values)}], `sanity_last`, `contacts` (at the end),
  `rate_hz`, `tick_ms`, `sim_time_start`, `gz_log`, `bag`.
- Bag: `ros2 bag record` (mcap, sim time) of the bridged topics (not the camera images), tf, joint states, the
  command topics and the evaluator's, started paused by the launch; the evaluator resumes it for each episode,
  then pauses and splits it: episode n = `runs/sim/<time>_<task>/bag/<n>_bag_<time>.mcap`.
- Launch args: `autostart`, `run_dir`, `bag` (`ros2 launch revolve_sim sim.launch.py ... task:=<t>`).

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
   episode ends with `reason: reset` (its result JSON and bag are written as usual); the recorder is paused and split,
   so the next episode gets the next bag file. The evaluator drops every input for 0.5 s of sim time, then is ready
   again once its inputs are live.

Client sequence per episode: read `resets` from `/evaluator/state`, call `/simulator/reset`, wait for an
`/evaluator/state` with `resets` above that value and `ready: true`, call `/evaluator/start`, command the robot,
wait for `/evaluator/result`. Sim time restarts at 1 s on every reset: compare times within an episode only (an
`/evaluator/state` `t` is not unique across episodes). A new scene instance is a new sim (relaunch). Typical reset:
~1.7 s wall.

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
                                      # (mobile_so101_xl) on a launched sim + the evaluator unit tests
                                      # (ros/revolve_sim/test); -- -m "not slow" skips T4
```
- `Dockerfile`: dependencies only, Ubuntu 26.04 + ROS 2 Lyrical binaries (ros2_control, ros2_controllers, MoveIt 2 +
  moveit_py + the moveit_resources Panda + pick_ik, xacro, robot_state_publisher, rosbag2, launch_testing) + Kura's apt build
  deps. Never the Lyrical ros-gz / gz-ros2-control binaries (built on Gazebo Jetty). Every bash in the image sources
  `/opt/revolve_env.sh`: ROS, then the ros_gz workspace, then Kura first on PATH / LD_LIBRARY_PATH /
  CMAKE_PREFIX_PATH (the binaries pull Jetty sdformat / gz-math / gz-tools vendors into /opt/ros/lyrical). Apt
  python3-pydantic, so `revolve_common.specs` imports in the image (PYTHONPATH /repo/common).
- `build.sh`, with colcon inside the image: (1) `ThirdParty/gz_kura/fetch.sh` (kura.repos pins + ThirdParty/patches)
  and Kura into `ThirdParty/gz_kura/install`, mounted at `/opt/gz_ws`; (2) the submodules ThirdParty/ros_gz,
  ThirdParty/gz_ros2_control (+ demos; ThirdParty/patches/gz_ros2_control applied in its working tree first) and
  ThirdParty/gz_vendors/* (gz_*_vendor, all satisfied by the Kura install)
  and `ros/revolve_sim` (ament_python; rebuild after changing it) into `ThirdParty/ros_gz_ws/install`, mounted at `/opt/ros_gz_ws/install`. Both git-ignored (build/, log/,
  .ccache/ too), mounted at runtime at the paths they are built at. Rebuilds are incremental; missing submodules are
  initialised, checked-out ones are left alone.
- Assets (PartNet-Mobility, ReplicaCAD: non-commercial) are read in place from `assets/` (git-ignored, repo root)
  through the repo mount; never copied into git or the image. They are made once by `./revolve assets`
  (`assets_pipeline/`, its own image; see its README).
- Tests (`ros/revolve_sim/test`, host pytest): each module starts a fresh `revolve sim` container and runs a
  check script in it (`check_bringup.py`, `check_control.py`, `check_moveit.py`, `check_evaluator.py`,
  `check_open_door.py`, `check_open_drawer.py`, `check_reset.py`, rclpy;
  each prints one RESULT JSON line); `test_episode_unit.py` needs no sim. A module runs once per robot of its
  `ROBOTS` (T1 / T2: `robot_cases.ROBOTS`, with the per-robot T2 targets in `robot_cases.CONTROL`; the MoveIt
  smoke: the robots and TCP pose targets of `robot_cases.MOVEIT`; T3 / T4 / reset: each robot's task, push and T4
  script in `robot_cases.TASKS`). Reset (`test_reset.py`): 5 resets between 6 episodes in one sim (moving episodes
  reset mid-episode, idle ones after their result), each checked against the start (base 1 cm / 1 deg, arm 1 deg,
  scene joints, controllers active, evaluator ready, a T2 arm move after it, one result + bag per episode, memory /
  processes stable); T4 ends with one reset from the opened door / drawer.
  A launch whose controller manager logs "No clock received" once a second (it never got /clock; rare, cause not
  pinned) fails at once with that message (conftest.py; no retry).
  `REVOLVE_TEST_ROBOT=<r>[,<r>]` keeps only those. Logs: `runs/test_simulator/<time>/<module>/<robot>/` or
  `$REVOLVE_TEST_LOGS`.
