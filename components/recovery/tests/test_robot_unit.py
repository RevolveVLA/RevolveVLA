"""core/robot.py: the arm controller + joints from the robot's controllers.yaml."""
import pytest

from datagen.core.spec import SpecError
from recovery.core.robot import Arm, arm_of
from revolve_common import specs


def test_arms_of_the_repo_robots():
    assert arm_of("mobile_panda", specs.REPO_SPECS) == Arm("panda_arm_controller", tuple(f"panda_joint{i}" for i in range(1, 8)))
    assert arm_of("mobile_so101_xl", specs.REPO_SPECS) == Arm(
        "arm_controller", ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"))


def test_unknown_robot_is_a_spec_error():
    with pytest.raises(SpecError, match="robot 'nope': no "):
        arm_of("nope", specs.REPO_SPECS)


def test_no_controller_with_the_arm_joints(tmp_path):
    robot = tmp_path / "robots" / "r"
    robot.mkdir(parents=True)
    (robot / "robot.yaml").write_text("name: r\ndescription: r.xacro\ncontrollers: c.yaml\nposes: {a: [0.0, 0.0, 0.0]}\n")
    (robot / "c.yaml").write_text("hand:\n  ros__parameters:\n    joints: [g]\nodd: 1\n")
    with pytest.raises(SpecError, match="no controller in c.yaml with the 3 arm joints"):
        arm_of("r", tmp_path)
