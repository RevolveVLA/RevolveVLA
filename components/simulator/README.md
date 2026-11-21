# simulator
Gazebo Kura (gz-sim 11 / gz-physics 10 main) + MuJoCo physics on an upstream ROS 2 Lyrical stack (ros_gz,
gz_ros2_control, ros2_controllers, MoveIt 2). The world, the robot and the task come from the YAML specs:
`specs/scenes/<s>/scene.yaml`, `specs/robots/<r>/`, `specs/tasks/<t>/task.yaml` + `specs/tasks/sanity.yaml`
(format and checks: `common/revolve_common/specs.py`, `./revolve validate`).

Status: the world, the robot and its controllers run (`ros/revolve_sim`); the live phase / metric evaluator is
next (`./revolve sim --task` only checks that the task fits the scene and robot so far).

## Run
```
./revolve sim --scene kitchen_partnet --robot mobile_panda [--task open_fridge_door] [--gui]
```
One container (`--name`, default `revolve-sim`) on the GPU and the host network (ROS 2 from the host works;
`ROS_DOMAIN_ID` is passed through), the repo mounted read-only at `/repo`. It runs `ros2 launch revolve_sim
sim.launch.py`, the upstream ros_gz / gz_ros2_control layout:
- `revolve_sim/world.py`: scene.yaml -> world SDF (`python3 -m revolve_sim.world <scene>` prints it): the models as
  `model://` includes at their poses (search path = the scene's `assets`), a JointStatePublisher on every
  articulated model, ground plane, sun, ambient light, 1 ms MuJoCo physics, the Sensors (ogre2), Contact,
  SceneBroadcaster and UserCommands systems. The world is named after the scene. Initial joint positions other
  than 0 are rejected for now (gz-sim has no upstream way to start a joint away from its SDF zero).
- the robot: `specs/robots/<r>/robot.urdf.xacro` (xacro args `initial_positions` = the scene's `robot_start.arm`
  pose, `controllers` = the robot's controllers.yaml), robot_state_publisher, `ros_gz_sim create` at
  `robot_start.base`, one controller spawner for every controller in controllers.yaml (gz_ros2_control loads them
  into its controller manager, which runs every physics step).
- `config/bridge.yaml` (ros_gz_bridge, filled in for the world, robot, cameras, contact links, articulated models):
  `/clock`, `/model/<robot>/pose` and `/world/<w>/dynamic_pose/info` (ground truth, tf2_msgs/TFMessage),
  `/world/<w>/model/<m>/joint_state`, `/world/<w>/model/<robot>/link/<l>/sensor/contact/contact`
  (ros_gz_interfaces/Contacts), `/<camera>/camera_info`; camera images through ros_gz_image: `/<camera>/image`.
- Commands: `/omni_wheel_drive_controller/cmd_vel` (geometry_msgs/TwistStamped, robot frame; ramp changes over
  ~0.5 s, the controller has no limiter), FollowJointTrajectory on `/<arm controller>/follow_joint_trajectory`
  and the hand controller (mobile_panda: panda_arm_controller, panda_hand_controller).
- Env: `GZ_MUJOCO_NOSLIP_ITERATIONS=10` (grasps and the omni base hold better), `GZ_SIM_RESOURCE_PATH` = the assets.
- `--gui`: a `gz sim -g` client on the host's X display (DISPLAY + XAUTHORITY); it costs RTF.

Known log lines: one MuJoCo "Nan, Inf or huge value in CTRL ... Time = 0.0000" warning at load; one "does not
support velocity limits" warning per joint with a URDF velocity limit (MuJoCo has none; ros2_control needs them).

## Build and test
```
./revolve build simulator [--clean]   # image revolve-simulator (deps only), then Kura, ros_gz, gz_ros2_control,
                                      # revolve_sim
./revolve test simulator              # T1 bring-up + T2 control on a launched sim (ros/revolve_sim/test)
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
  and `ros/revolve_sim` (ament_python; rebuild after changing it) into `ThirdParty/ros_gz_ws/install`, mounted at `/opt/ros_gz_ws/install`. Both git-ignored (build/, log/,
  .ccache/ too), mounted at runtime at the paths they are built at. Rebuilds are incremental; missing submodules are
  initialised, checked-out ones are left alone.
- Assets (PartNet-Mobility, ReplicaCAD: non-commercial) are read in place from `assets/` (git-ignored, repo root)
  through the repo mount; never copied into git or the image. They are made once by `./revolve assets`
  (`assets_pipeline/`, its own image; see its README).
- Tests (`ros/revolve_sim/test`, host pytest): each module starts a fresh `revolve sim` container and runs a
  check script in it (`check_bringup.py`, `check_control.py`, rclpy; each prints one RESULT JSON line). Logs:
  `runs/test_simulator/<time>/` or `$REVOLVE_TEST_LOGS`.
