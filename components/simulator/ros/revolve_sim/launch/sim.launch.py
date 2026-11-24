# The simulator: the world generated from a scene spec, the robot from a robot spec, the upstream ros_gz /
# gz_ros2_control layout (as in gz_ros2_control_demos): ros_gz_sim gz_sim.launch.py + robot_state_publisher +
# controller spawners + ros_gz_bridge / ros_gz_image, and the reset service (revolve_sim/reset.py).
#   ros2 launch revolve_sim sim.launch.py scene:=<s> robot:=<r> [task:=<t> [autostart:=false] [run_dir:=DIR]
#       [record:=images|no-images|none]] [gui:=true] [repo:=/repo]
# The robot starts at the scene's robot_start: its URDF is part of the world (at the base pose), so a gz world reset
# restores it; the arm through the ros2_control initial values (xacro arg initial_positions).
# Sim time starts at 1 s (ROS time 0 means no clock), also after a reset. With a task: the evaluator
# (revolve_sim/evaluator.py) and a rosbag2 recorder of the topics in config/record.yaml (the cameras only with
# record:=images; idle until the evaluator records an episode into its own bag), both into run_dir (default
# <repo>/runs/sim/<time>_<task>). Needs revolve_common on PYTHONPATH (the repo's common/).
import os
import tempfile
import time
from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction,
                            SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from revolve_common import specs
from revolve_sim.robot import controller_types, describe, robot_dir
from revolve_sim.world import make_world

SIM_TIME = {'use_sim_time': True}
JTC = 'joint_trajectory_controller/JointTrajectoryController'
OMNI = 'omni_wheel_drive_controller/OmniWheelDriveController'


def fill(entries, world, robot, lists):
    """Config entries (bridge.yaml, record.yaml) with {world} and {robot} filled in, each entry with a key of lists
    ({camera}, ...) repeated for each of its values."""
    out = []
    for e in entries:
        text = yaml.safe_dump(e)
        key, values = next(((k, v) for k, v in lists.items() if k in text), (None, [None]))
        for v in values:
            t = text.replace(key, v) if key else text
            out.append(yaml.safe_load(t.replace('{world}', world).replace('{robot}', robot)))
    return out


def setup(context):
    arg = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    repo = Path(arg('repo')).resolve()
    tmp = Path(tempfile.mkdtemp(prefix='revolve_sim_'))
    scene = specs.load(specs.scene_file(arg('scene'), repo / 'specs'))
    robot, urdf, cams, contacts = describe(repo, arg('robot'), arm=scene.robot_start.arm)
    (tmp / 'robot.urdf').write_text(urdf)
    world_sdf, scene, resource_paths, models = make_world(arg('scene'), repo, robot=(robot.name, tmp / 'robot.urdf'))
    task = arg('task')
    if task:    # the task fits this scene (a scene instance keeps its scene's name) and robot
        specs.resolve(repo / 'specs', task, robot=robot.name, scene=scene)
    controllers = robot_dir(repo, robot.name) / robot.controllers

    (tmp / f'{scene.name}.sdf').write_text(world_sdf)
    ctrl_types = controller_types(controllers)
    ctrl_names = list(ctrl_types)
    lists = {'{camera}': cams, '{contact_link}': contacts, '{model}': models, '{scene_model}': list(scene.models),
             '{jtc}': [n for n, t in ctrl_types.items() if t == JTC], '{omni}': [n for n, t in ctrl_types.items()
                                                                               if t == OMNI]}
    bridge_yaml = tmp / 'bridge.yaml'
    share = Path(FindPackageShare('revolve_sim').perform(context))
    bridge_yaml.write_text(yaml.safe_dump(fill(yaml.safe_load((share / 'config' / 'bridge.yaml').read_text()),
                                               scene.name, robot.name, lists)))

    gz_args = f'-s -r -v 3 --headless-rendering --initial-sim-time 1 {tmp / scene.name}.sdf'
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([PathJoinSubstitution([FindPackageShare('ros_gz_sim'), 'launch',
                                                             'gz_sim.launch.py'])]),
        launch_arguments=[('gz_args', gz_args), ('on_exit_shutdown', 'true')])
    # tf of the moving links at 100 Hz (default 20 Hz), the evaluator's rate
    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
               parameters=[{'robot_description': urdf, 'publish_frequency': 100.0}, SIM_TIME])
    # one spawner for every controller in controllers.yaml; the omni controller's odom tf goes to /tf
    spawner = Node(package='controller_manager', executable='spawner', output='screen',
                   arguments=[*ctrl_names, '--param-file', str(controllers),
                              '--controller-ros-args', '-r ~/tf_odometry:=/tf'])
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
                  parameters=[{'config_file': str(bridge_yaml)}, SIM_TIME])
    images = Node(package='ros_gz_image', executable='image_bridge', output='screen',
                  arguments=[f'/{c}/image' for c in cams], parameters=[SIM_TIME])
    reset = Node(package='revolve_sim', executable='reset', output='screen',
                 parameters=[{'world': scene.name}, SIM_TIME])
    actions = [
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', ':'.join(
            [*resource_paths, *filter(None, [os.environ.get('GZ_SIM_RESOURCE_PATH')])])),
        SetEnvironmentVariable('GZ_MUJOCO_NOSLIP_ITERATIONS', '10'),   # grasps + omni-wheel holding
        gz, rsp, spawner, bridge, images, reset,
    ]
    if task:
        run_dir = Path(arg('run_dir') or repo / 'runs' / 'sim' / f'{time.strftime("%Y%m%d_%H%M%S")}_{task}')
        run_dir.mkdir(parents=True, exist_ok=True)
        record = arg('record')
        if record not in ('images', 'no-images', 'none'):
            raise ValueError(f'record:={record}: images, no-images or none')
        bag = record != 'none'
        actions.append(Node(package='revolve_sim', executable='evaluator', output='screen', parameters=[{
            'repo': str(repo), 'scene': arg('scene'), 'robot': robot.name, 'task': task, 'run_dir': str(run_dir),
            'autostart': arg('autostart') == 'true', 'bag': bag}, SIM_TIME]))
        if bag:   # paused into a scratch bag in the container; the evaluator stops it and records each episode
            rec = yaml.safe_load((share / 'config' / 'record.yaml').read_text())
            topics = fill(rec['topics'] + (rec['images'] if record == 'images' else []), scene.name, robot.name,
                          lists)
            (tmp / 'record_qos.yaml').write_text(yaml.safe_dump(rec['qos_overrides']))
            # the prompt also in each bag's metadata, under the key Rosetta's porter reads as the task of the frames
            # before the first /task/prompt message
            prompt = specs.load(specs.spec_path(repo / 'specs', 'tasks', task), assets=False).prompt
            # mcap with zstd chunk compression (the 1 kHz controller / odometry / tf streams shrink ~4.5x)
            actions.append(ExecuteProcess(cmd=['ros2', 'bag', 'record', '--start-paused', '--use-sim-time',
                                               '--disable-keyboard-controls', '-o', str(tmp / 'idle_bag'),
                                               '--storage-preset-profile', 'zstd_fast',
                                               '--qos-profile-overrides-path', str(tmp / 'record_qos.yaml'),
                                               '--custom-data', f'lerobot.operator_prompt={prompt}',
                                               '--topics', *topics], output='screen'))
    if arg('gui') == 'true':
        actions.append(ExecuteProcess(cmd=['gz', 'sim', '-g', '-v', '2'], output='screen', additional_env={
            '__NV_PRIME_RENDER_OFFLOAD': '1', '__GLX_VENDOR_LIBRARY_NAME': 'nvidia'}))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('scene', description='specs/scenes/<scene>, or a scene.yaml path (a scene '
                              'instance: named like its scene, assets relative to the repo)'),
        DeclareLaunchArgument('robot', description='specs/robots/<robot>'),
        DeclareLaunchArgument('task', default_value='', description='specs/tasks/<task> (checked against the '
                              'scene and robot)'),
        DeclareLaunchArgument('autostart', default_value='true', description='with a task: start the first episode '
                              'as soon as the evaluator has every input (false: on the ~/start service)'),
        DeclareLaunchArgument('run_dir', default_value='', description='with a task: the episode results + bags '
                              '(default <repo>/runs/sim/<time>_<task>)'),
        DeclareLaunchArgument('record', default_value='images', description='with a task: record each episode '
                              '(rosbag2) with the camera images (images), without them (no-images) or not (none)'),
        DeclareLaunchArgument('gui', default_value='false', description='also start the gz GUI client'),
        DeclareLaunchArgument('repo', default_value=os.environ.get('REVOLVE_REPO', '/repo')),
        OpaqueFunction(function=setup),
    ])
