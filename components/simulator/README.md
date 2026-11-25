# simulator

The simulator runs one world (a scene, a robot and, optionally, a task) in a Docker container on the GPU. It is
Gazebo Kura with MuJoCo physics on ROS 2 Lyrical, with ros_gz, gz_ros2_control and MoveIt 2. Every other stage of
the loop runs on top of it: datagen and evaluation drive episodes through its ROS 2 interfaces, and with `--task`
the simulator scores each episode and records it to a bag.

## Inputs

| Input | Comes from | Format |
|---|---|---|
| Scene | `specs/scenes/<scene>/scene.yaml`, or a scene instance (a copy of the scene with another robot start, written by datagen) | `--scene <name>`, or `--scene <path>/scene.yaml` outside `specs/` |
| Robot | `specs/robots/<robot>/`: `mobile_panda`, `mobile_so101`, `mobile_so101_xl` | `--robot <name>` |
| Task (optional) | `specs/tasks/<task>/task.yaml` plus the shared checks in `specs/tasks/sanity.yaml` | `--task <name>`. Without it there is no scoring and no recording |
| Episode options | command line, all need `--task` | `--record images\|no-images` (default `images`), `--out DIR`, `--no-autostart` |
| Container options | command line | `--name N` (default `revolve-sim`), `--domain-id D` (0 to 232, default picked from the name), `--gui` (needs `DISPLAY` and `XAUTHORITY`) |
| Robot commands | a client on the sim's ROS domain | `/<arm controller>/joint_trajectory` or `follow_joint_trajectory`, `/omni_wheel_drive_controller/cmd_vel` |
| Scene assets | `assets/`, made once by `./revolve assets` | SDF models, git-ignored |
| Builds | `./revolve build simulator` | `ThirdParty/gz_kura/install`, `ThirdParty/ros_gz_ws/install` |
| Environment | your shell | `REVOLVE_SIM_IMAGE` (default `revolve-simulator:latest`), `REVOLVE_GPU_PREFLIGHT=0` skips the GPU check |

## Outputs

| Output | Where it lands | Read by |
|---|---|---|
| Episode result | `<out>/episode_<n>.json`, also on `/evaluator/result`. Written last, so the episode is complete once it exists | datagen, evaluation |
| Episode bag | `<out>/episode_<n>_bag/`: rosbag2, MCAP, the topics in `config/record.yaml`, camera JPEGs with `--record images` | dataset (through Rosetta) |
| gz server log | `<out>/episode_<n>_gz.log`, the episode's part of the log | you, when debugging |
| Live scoring | `/evaluator/state` (JSON, 10 Hz) and the `/evaluator/start` service | datagen and evaluation clients |
| Reset | the `/simulator/reset` service puts the world, robot and controllers back at the scene start (about 1.7 s) | datagen and evaluation clients |
| Sensors and state | three 480 x 480 cameras at 30 Hz, `/joint_states`, `/tf`, contacts, ground-truth poses, `/task/prompt` (with `--task`) | clients, the bags |

`<out>` defaults to `runs/sim/<time>_<task>/`. Episode numbers continue after the highest one already in `<out>`,
so a relaunch into the same folder overwrites nothing.

## Run

```
./revolve build simulator             # the image, then Kura, ros_gz, gz_ros2_control and revolve_sim
./revolve validate                    # check every spec; the same checks run before each sim starts
./revolve sim --scene kitchen_partnet --robot mobile_panda                           # world and robot only
./revolve sim --scene kitchen_partnet --robot mobile_panda --task open_fridge_door   # plus scoring and bags
./revolve sim --scene runs/exp/scene_7/scene.yaml --robot mobile_panda --task open_fridge_door --out runs/exp/scene_7
docker rm -f revolve-sim              # stop it
```

The sim runs in the foreground. Its first line on stderr is `ROS_DOMAIN_ID=<D>`: export that to reach it with
ROS 2 from the host. Exit codes: 1 for a refused input, a missing build, a broken GPU or a launch error; 2 for a
usage error; 125 when the container name is taken; 137 after `docker rm -f`.

The sim runs the installed copy of `ros/revolve_sim`. After you edit it, run `./revolve build simulator` again, or
the sim keeps running the old code without a warning.

## Test

```
./revolve test simulator                     # the full suite: needs the build, the assets and a GPU
./revolve test simulator -- -m "not slow"    # skip the long MoveIt episodes
REVOLVE_TEST_ROBOT=mobile_panda ./revolve test simulator    # one robot only
```

Each test module starts its own sim container and runs a check script in it. They cover bring-up, control of
every robot, a MoveIt smoke test, scoring, full MoveIt episodes for both tasks, the reset loop, the datagen
contract and each robot's Rosetta contract on real bags. `test_cli.py`, `test_episode_unit.py` and
`test_episode_numbers.py` need no Docker. `test_rosetta_contracts.py` needs the simulator image but no GPU. Logs
go to `runs/test_simulator/<time>/`.

## Files

- `ros/revolve_sim/launch/sim.launch.py`: what the container runs.
- `ros/revolve_sim/revolve_sim/world.py`: turns `scene.yaml` into the world SDF.
- `ros/revolve_sim/revolve_sim/evaluator.py`, `episode.py`, `metrics.py`: the scoring.
- `ros/revolve_sim/revolve_sim/reset.py`: the `/simulator/reset` service.
- `ros/revolve_sim/config/record.yaml`: the topics every bag records.
- `Dockerfile`, `build.sh`: the dependency image (Ubuntu 26.04, ROS 2 Lyrical) and the source build.

How a client drives episodes and what each bag holds: [DATAGEN-v2.md](../datagen/DATAGEN-v2.md), sections 4 to 7.
