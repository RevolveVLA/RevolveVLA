"""A robot spec -> its URDF (xacro), camera sensors, contact-sensor links and command topics."""
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from revolve_common import specs


def robot_dir(repo, name):
    return Path(repo).resolve() / 'specs' / 'robots' / name


def describe(repo, name, arm=None):
    """(Robot spec, URDF text, [camera names], [links with a contact sensor]). arm: a pose name or joint list for
    the ros2_control initial positions (default: the description's own default); specs.SpecError if it does not fit
    the robot (xacro would index the list by joint: extra values would become the gripper's start; `revolve sim`
    refuses such a scene before docker, specs.check_run)."""
    d = robot_dir(repo, name)
    robot = specs.load(d / 'robot.yaml')
    if arm is not None and (errs := specs.arm_errors(arm, robot)):
        raise specs.SpecError('\n'.join(errs))
    import xacro
    mappings = {'controllers': str(d / robot.controllers)}
    if arm is not None:
        q = robot.poses[arm] if isinstance(arm, str) else arm
        mappings['initial_positions'] = ' '.join(map(str, q))
    urdf = xacro.process_file(str(d / robot.description), mappings=mappings).toxml()
    cams, contacts = [], []
    for g in ET.fromstring(urdf).findall('gazebo'):
        for s in g.findall('sensor'):
            if s.get('type') == 'camera':
                cams.append(s.get('name'))
            elif s.get('type') == 'contact':
                contacts.append(g.get('reference'))
    return robot, urdf, cams, contacts


# the command topic (and its message type) of each controller type that takes commands on a topic
CMD_TOPICS = {'omni_wheel_drive_controller/OmniWheelDriveController': ('cmd_vel', 'geometry_msgs/msg/TwistStamped'),
              'joint_trajectory_controller/JointTrajectoryController':
                  ('joint_trajectory', 'trajectory_msgs/msg/JointTrajectory')}


def controller_types(controllers_yaml):
    """{controller name: type} of a ros2_control controllers.yaml."""
    params = yaml.safe_load(Path(controllers_yaml).read_text())['controller_manager']['ros__parameters']
    return {name: v['type'] for name, v in params.items() if isinstance(v, dict) and 'type' in v}


def command_topics(controllers_yaml):
    """[(topic, message type)] of the controllers in a ros2_control controllers.yaml."""
    return [(f'/{name}/{CMD_TOPICS[t][0]}', CMD_TOPICS[t][1]) for name, t in controller_types(controllers_yaml).items()
            if t in CMD_TOPICS]
