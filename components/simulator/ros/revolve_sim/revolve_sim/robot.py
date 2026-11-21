"""A robot spec -> its URDF (xacro), camera sensors and contact-sensor links."""
import xml.etree.ElementTree as ET
from pathlib import Path

from revolve_common import specs


def robot_dir(repo, name):
    return Path(repo).resolve() / 'specs' / 'robots' / name


def describe(repo, name, arm=None):
    """(Robot spec, URDF text, [camera names], [links with a contact sensor]). arm: a pose name or joint list for
    the ros2_control initial positions (default: the description's own default)."""
    d = robot_dir(repo, name)
    robot = specs.load(d / 'robot.yaml')
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
