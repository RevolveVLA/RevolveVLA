"""The arm of a robot as its controllers.yaml gives it: the joint trajectory controller whose joints are the arm's
(robot.yaml `poses` has `arm_dof` joints per pose) and its joint names in that order."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datagen.core.spec import SpecError
from revolve_common import specs


@dataclass(frozen=True)
class Arm:
    controller: str
    joints: tuple[str, ...]


def arm_of(robot: str, specs_root: Path) -> Arm:
    try:
        path = specs.spec_path(specs_root, "robots", robot)
        spec = specs.parse(specs.Robot, path)
        controllers = specs.read_yaml(path.parent / spec.controllers)
    except specs.SpecError as e:
        raise SpecError(str(e)) from None
    for name, block in controllers.items():
        joints = block.get("ros__parameters", {}).get("joints") if isinstance(block, dict) else None
        if isinstance(joints, list) and len(joints) == spec.arm_dof:
            return Arm(name, tuple(joints))
    raise SpecError(f"robot {robot}: no controller in {spec.controllers} with the {spec.arm_dof} arm joints of its poses")
