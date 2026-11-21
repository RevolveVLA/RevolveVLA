# The simulator: the world generated from a scene spec, the robot from a robot spec, the upstream ros_gz /
# gz_ros2_control layout (as in gz_ros2_control_demos): ros_gz_sim gz_sim.launch.py + robot_state_publisher +
# ros_gz_sim create + controller spawners + ros_gz_bridge / ros_gz_image.
#   ros2 launch revolve_sim sim.launch.py scene:=<s> robot:=<r> [task:=<t>] [gui:=true] [repo:=/repo]
# The robot starts at the scene's robot_start: the base pose through create, the arm through the ros2_control
# initial values (xacro arg initial_positions). Needs revolve_common on PYTHONPATH (the repo's common/).
import os
import tempfile
from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction,
                            RegisterEventHandler, SetEnvironmentVariable)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from revolve_common import specs
from revolve_sim.robot import describe, robot_dir
from revolve_sim.world import make_world

SIM_TIME = {'use_sim_time': True}


def bridge_config(template, world, robot, cams, contacts, models):
    out = []
    for e in yaml.safe_load(Path(template).read_text()):
        text = yaml.safe_dump(e)
        for key, values in (('{camera}', cams), ('{contact_link}', contacts), ('{model}', models)):
            if key in text:
                break
        else:
            key, values = None, [None]
        for v in values:
            t = text.replace(key, v) if key else text
            out.append(yaml.safe_load(t.replace('{world}', world).replace('{robot}', robot)))
    return out


def setup(context):
    arg = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    repo = Path(arg('repo')).resolve()
    world_sdf, scene, resource_paths, models = make_world(arg('scene'), repo)
    robot, urdf, cams, contacts = describe(repo, arg('robot'), arm=scene.robot_start.arm)
    if arg('task'):
        specs.resolve(repo / 'specs', arg('task'), robot=robot.name)   # the task fits this scene and robot
    controllers = robot_dir(repo, robot.name) / robot.controllers

    tmp = Path(tempfile.mkdtemp(prefix='revolve_sim_'))
    (tmp / f'{scene.name}.sdf').write_text(world_sdf)
    bridge_yaml = tmp / 'bridge.yaml'
    share = Path(FindPackageShare('revolve_sim').perform(context))
    bridge_yaml.write_text(yaml.safe_dump(bridge_config(share / 'config' / 'bridge.yaml', scene.name, robot.name,
                                                        cams, contacts, models)))
    ctrl_names = [k for k, v in yaml.safe_load(controllers.read_text())['controller_manager']['ros__parameters']
                  .items() if isinstance(v, dict) and 'type' in v]

    gz_args = f'-s -r -v 3 --headless-rendering {tmp / scene.name}.sdf'
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([PathJoinSubstitution([FindPackageShare('ros_gz_sim'), 'launch',
                                                             'gz_sim.launch.py'])]),
        launch_arguments=[('gz_args', gz_args), ('on_exit_shutdown', 'true')])
    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
               parameters=[{'robot_description': urdf}, SIM_TIME])
    x, y, yaw = scene.robot_start.base
    spawn = Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-world', scene.name, '-topic', 'robot_description', '-name', robot.name,
                            '-x', str(x), '-y', str(y), '-z', '0.0', '-Y', str(yaw)])
    # one spawner for every controller in controllers.yaml; the omni controller's odom tf goes to /tf
    spawner = Node(package='controller_manager', executable='spawner', output='screen',
                   arguments=[*ctrl_names, '--param-file', str(controllers),
                              '--controller-ros-args', '-r ~/tf_odometry:=/tf'])
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
                  parameters=[{'config_file': str(bridge_yaml)}, SIM_TIME])
    images = Node(package='ros_gz_image', executable='image_bridge', output='screen',
                  arguments=[f'/{c}/image' for c in cams], parameters=[SIM_TIME])
    actions = [
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', ':'.join(
            [*resource_paths, *filter(None, [os.environ.get('GZ_SIM_RESOURCE_PATH')])])),
        SetEnvironmentVariable('GZ_MUJOCO_NOSLIP_ITERATIONS', '10'),   # grasps + omni-wheel holding
        gz, rsp, spawn, bridge, images,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[spawner])),
    ]
    if arg('gui') == 'true':
        actions.append(ExecuteProcess(cmd=['gz', 'sim', '-g', '-v', '2'], output='screen', additional_env={
            '__NV_PRIME_RENDER_OFFLOAD': '1', '__GLX_VENDOR_LIBRARY_NAME': 'nvidia'}))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('scene', description='specs/scenes/<scene> (or a scene.yaml path)'),
        DeclareLaunchArgument('robot', description='specs/robots/<robot>'),
        DeclareLaunchArgument('task', default_value='', description='specs/tasks/<task> (checked against the '
                              'scene and robot)'),
        DeclareLaunchArgument('gui', default_value='false', description='also start the gz GUI client'),
        DeclareLaunchArgument('repo', default_value=os.environ.get('REVOLVE_REPO', '/repo')),
        OpaqueFunction(function=setup),
    ])
