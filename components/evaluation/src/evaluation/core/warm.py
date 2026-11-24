"""The start a warm reset has to reach for a scene instance (revolve_common.warm_reset): its `robot_start` (base pose, arm
pose name or joints) and where the robot's arm is commanded (the moveit generator's controller table: robot.yaml does not
carry controller names)."""
from __future__ import annotations

from pathlib import Path

import yaml
from datagen.plugins.moveit.setup import CASES
from revolve_common import specs
from revolve_common.sim_session import REPO
from revolve_common.warm_reset import ResetTarget


def reset_target(scene: Path, robot: str, repo: Path = REPO) -> ResetTarget | None:
    """The target of the instance `scene` for `robot`; None when the robot has no controller table or the instance cannot be read."""
    case = CASES.get(robot)
    try:
        start = (yaml.safe_load(Path(scene).read_text()) or {})["robot_start"]
        poses = specs.read_yaml(specs.spec_path(repo / "specs", "robots", robot))["poses"]
        arm = poses[start["arm"]] if isinstance(start["arm"], str) else start["arm"]
        return case and ResetTarget(robot, tuple(start["base"]), tuple(arm), tuple(case["arm_joints"]), case["arm_controller"])
    except (OSError, KeyError, TypeError, yaml.YAMLError, specs.SpecError):
        return None
