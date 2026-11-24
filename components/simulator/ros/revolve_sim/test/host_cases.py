"""Helpers of the tests that start no sim from the folder's conftest: scene instance files, the robots' poses from
their specs, directory snapshots."""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[5]
SCENE_SPEC = REPO / "specs" / "scenes" / "kitchen_partnet" / "scene.yaml"


def pose(robot_name, name):
    """A pose of the robot's spec (specs/robots/<robot>/robot.yaml): its arm joint values."""
    return yaml.safe_load((REPO / "specs" / "robots" / robot_name / "robot.yaml").read_text())["poses"][name]


def write_instance(path, arm=None, joints=None):
    """kitchen_partnet's scene.yaml written at path (a scene instance), with robot_start.arm = arm and the fridge's
    joints = joints if given."""
    data = yaml.safe_load(SCENE_SPEC.read_text())
    if arm is not None:
        data["robot_start"]["arm"] = arm
    if joints is not None:
        data["models"]["fridge"]["joints"] = joints
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def snapshot(root):
    """Every path under root, with its type and mtime: equal snapshots mean nothing was created or written."""
    return sorted((str(p.relative_to(root)), p.is_dir(), p.stat().st_mtime_ns) for p in root.rglob("*"))
