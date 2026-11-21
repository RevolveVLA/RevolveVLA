# simulator
Gazebo Kura (gz-sim 11 / gz-physics 10 main) + MuJoCo physics on an upstream ROS 2 Lyrical stack (ros_gz,
gz_ros2_control, ros2_controllers, MoveIt 2). The world, the robot and the task come from the YAML specs:
`specs/scenes/<s>/scene.yaml`, `specs/robots/<r>/`, `specs/tasks/<t>/task.yaml` + `specs/tasks/sanity.yaml`
(format and checks: `common/revolve_common/specs.py`, `./revolve validate`).

Status: only the dependency image and the third-party build are here. The earlier gz Python systems and their
launcher (`src/sim/`, `src/worlds/`, `launch.py`, the old tests) are removed. Next: the ROS 2 package
`ros/revolve_sim` (the world generated from scene.yaml, the robot spawn and controllers, the bridge, the live
phase / metric evaluator) with `./revolve sim --scene <s> --robot <r> [--task <t>] [--gui]` and its tests.

## Build and test
```
./revolve build simulator [--clean]   # image revolve-simulator (deps only), then Kura, ros_gz, gz_ros2_control
```
- `Dockerfile`: dependencies only, Ubuntu 26.04 + ROS 2 Lyrical binaries (ros2_control, ros2_controllers, MoveIt 2 +
  moveit_py + the moveit_resources Panda, xacro, robot_state_publisher, rosbag2, launch_testing) + Kura's apt build
  deps. Never the Lyrical ros-gz / gz-ros2-control binaries (built on Gazebo Jetty). Every bash in the image sources
  `/opt/revolve_env.sh`: ROS, then the ros_gz workspace, then Kura first on PATH / LD_LIBRARY_PATH /
  CMAKE_PREFIX_PATH (the binaries pull Jetty sdformat / gz-math / gz-tools vendors into /opt/ros/lyrical). Apt
  python3-pydantic, so `revolve_common.specs` imports in the image (PYTHONPATH /repo/common).
- `build.sh`, with colcon inside the image: (1) `ThirdParty/gz_kura/fetch.sh` (kura.repos pins + ThirdParty/patches)
  and Kura into `ThirdParty/gz_kura/install`, mounted at `/opt/gz_ws`; (2) the submodules ThirdParty/ros_gz,
  ThirdParty/gz_ros2_control (+ demos) and ThirdParty/gz_vendors/* (gz_*_vendor, all satisfied by the Kura install)
  into `ThirdParty/ros_gz_ws/install`, mounted at `/opt/ros_gz_ws/install`. Both git-ignored (build/, log/,
  .ccache/ too), mounted at runtime at the paths they are built at. Rebuilds are incremental; missing submodules are
  initialised, checked-out ones are left alone.
- Assets (PartNet-Mobility, ReplicaCAD: non-commercial) are read in place from `assets/` (git-ignored, repo root)
  through the repo mount; never copied into git or the image. They are made once by `./revolve assets`
  (`assets_pipeline/`, its own image; see its README).
