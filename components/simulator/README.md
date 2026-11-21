# simulator
Gazebo Kura (gz-sim 11 / gz-physics 10 main) + MuJoCo physics for the PartNet kitchen and the robots
(arms on a virtual holonomic base: mobile_panda, mobile_so101). Ported from the earlier pipeline's gz demo
(scripts, worlds) and simulator plugin (replay, live phase validator, save/restore), with every threshold
read from specs/tasks/<t>/task.yaml.

Robots: the robot is `$REVOLVE_ROBOT` (the experiment's `robot:`; the CLI and the launchers pass it to every
job). Each robot has `specs/robots/<robot>/robot.yaml` + `contract.yaml` and `src/sim/robots/<robot>/` with
the same files (`robot.py` kinematics and laws, `model.py` gz model from robot.yaml `description`,
`scripted.py` smoke plan and scripted episodes, `planner.py` the cuRobo planner robot that datagen and
recovery build their planner assets from) behind `src/sim/robots/base.py` (shared base chain, Plan,
Controller). Everything else in `src/sim/` is robot-independent. `src/sim/can_demo/` is the fixed-arm
can-on-plate demo (Panda only, not a pipeline task).

## Build and test
```
./revolve build simulator [--clean]   # image revolve-simulator (deps only), then Kura, ros_gz, gz_ros2_control
./revolve test simulator              # pytest tests/ (all sim jobs in containers, run concurrently)
```
- `Dockerfile`: dependencies only, Ubuntu 26.04 + ROS 2 Lyrical binaries (ros2_control, ros2_controllers, MoveIt 2 +
  moveit_py + the moveit_resources Panda, xacro, robot_state_publisher, rosbag2, launch_testing) + Kura's apt build
  deps. Never the Lyrical ros-gz / gz-ros2-control binaries (built on Gazebo Jetty). Every bash in the image sources
  `/opt/revolve_env.sh`: ROS, then the ros_gz workspace, then Kura first on PATH / LD_LIBRARY_PATH /
  CMAKE_PREFIX_PATH (the binaries pull Jetty sdformat / gz-math / gz-tools vendors into /opt/ros/lyrical).
- `build.sh`, with colcon inside the image: (1) `ThirdParty/gz_kura/fetch.sh` (kura.repos pins + ThirdParty/patches)
  and Kura into `ThirdParty/gz_kura/install`, mounted at `/opt/gz_ws`; (2) the submodules ThirdParty/ros_gz,
  ThirdParty/gz_ros2_control (+ demos) and ThirdParty/gz_vendors/* (gz_*_vendor, all satisfied by the Kura install)
  into `ThirdParty/ros_gz_ws/install`, mounted at `/opt/ros_gz_ws/install`. Both git-ignored (build/, log/,
  .ccache/ too), mounted at runtime at the paths they are built at. Rebuilds are incremental; missing submodules are
  initialised, checked-out ones are left alone.
- Assets (PartNet-Mobility, ReplicaCAD: non-commercial) are read in place from `assets/` (git-ignored, repo root)
  through the repo mount; never copied into git or the image. They are made once by `./revolve assets`
  (`assets_pipeline/`, its own image; see its README).

## Entrypoints (`src/sim/cli.py`, run under `revolve_common.wrapper`)
| Command | What | Was |
|---|---|---|
| `scene-check` | kitchen without robot: door/drawer limits, rest drift/jitter, cup on its shelf | `kitchen_test` |
| `robot-smoke` | the robot's spawn + smoke test in the kitchen (its `scripted.smoke_plan`) | `kitchen_robot_test` |
| `can-demo` | Panda can-on-plate (`src/sim/can_demo/`) | `headless` can demo |
| `episode --name fridge\|drawer` | scripted episode, fed through the live phase validator: its own checks + per-phase verdicts + phase states | `episode <name>` |
| `replay --plan RUN --attempt A` | AutoMoMa plan attempt: offline verdict + per-phase verdicts + `phase_states/` + `state_log.npz` + MCAP `bag/` | `replay` + `live_validate` + bag write |
| `restore-check --source D --phase P` | restore `D/phase_states/P_start.npz`, run on, compare with the source | `live_validate --restore` |
| `rollout --start D0 --out D [--restore D1:PHASE]` | closed-loop policy rollout (image revolve-simulator-ros): `closed_loop_system` (rebased on the live phase validator, policy mode: `phases.PolicyBoundaries`) follows the actions `ros_relay.py` brings from ROS 2; per-phase verdicts + phase states; a phase still running at the end fails (`phase_ended`) | `gz_closed_loop.sh` + `policy_episode.py` + `live_validate` |
Exit code = verdict. `--video` renders the scene camera to `<out>/video.mp4` (GPU). The rollout's host side
(serve, Rosetta's runner, networks) is components/evaluation.

Host side: `launch.py` (docker SDK) starts a job container with the repo read-only at /repo, runs/
read-write, the Kura install read-only, the wrapper as the command (GET /status, /result, POST /cancel
on the container's IP; no published ports), and writes job.json; the wrapper writes status.json,
result.json and stdout.log into the job dir.

## Outputs of a phase run (`episode`, `replay`, `restore-check`)
`phase_report.json` (per phase: status, checks named by metric with value/threshold/t_fail, start state;
`conditions` = the task.yaml gates used; `metric_states` = the states they were judged on; `result`
= first failed phase), `phase_states/<phase>_start.npz` (+ `_start_next.npz`), `state_log.npz`,
`summary.json` (the ported gz checks), `verdict.json` (replay: the offline verdict), `bag/`, `bag.json`,
`bag_check.json` (replay), `restore_check.json` (restore).

## Layout
`src/sim/`: the gz Python systems and helpers (flat: gz loads systems by module name from
GZ_SIM_SYSTEM_PLUGIN_PATH); `spec.py` is the bridge to specs/ (replaces the earlier pipeline's plugin loader and the
task validator.py files). `src/worlds/`: the world files. `src/egl/`: the NVIDIA EGL vendor file for GPU
rendering. Generated robot models go to `runs/models/` (with the training cameras) and `runs/models/plain/`.
