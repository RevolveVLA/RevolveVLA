"""What one plan run needs, built from the v2 specs (the scene instance, the task, the robot) and planner.yaml: the robot's start
(the instance's `robot_start`), the task object and its handle, the goal of the joint, the scene's models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from datagen.scene.sdf_boxes import scene_boxes
from revolve_common import specs

from . import frames
from .model import Model

PLANNER = Path(__file__).with_name("planner.yaml")


class NotSupported(ValueError):
    """The robot or task has no planner data (planner.yaml)."""


@dataclass
class Problem:
    robot: str
    task: str
    scene: str
    object: str                 # the task object's scene entity (fridge)
    handle: specs.Entity
    world_pose: list            # the object's [x, y, z, yaw] in the world
    start_base: list            # world x, y, yaw of the instance's robot_start
    start_base_obj: list        # the same in the planner (object) frame
    ready: list
    start_arm: list
    folded: list
    tcp: str
    arm_dof: int
    joint: str                  # the task joint (of the object)
    goal: float                 # the joint position the pull ends at [rad | m]
    boxes: list                 # [(name, 4x4 world, size)]: the scene's collision boxes but the task joint's moving link
    model_file: Path
    cfg: dict                   # planner.yaml robots.<robot>
    planner: dict               # planner.yaml (whole)
    grasp_along: list
    door_spheres: dict
    yaw_window: float
    floor: dict

    def object_model(self) -> Model:
        return Model.read(self.model_file, self.joint)


def task_joint(task: specs.Task) -> tuple[specs.Entity, float]:
    """(the joint entity, its lowest `ge` bound) of the task's `joint` phase conditions: what the task asks the joint to reach."""
    found = [(specs.parse_entity(c.params["of"]), c.value) for ph in task.phases for c in ph.done
             if c.metric == "joint" and c.op == "ge"]
    if not found:
        raise NotSupported(f"task {task.name}: no phase condition `joint ge` to plan for")
    return min(found, key=lambda f: f[1])


def check_supported(task: str, robot: str) -> None:
    """NotSupported unless planner.yaml has the robot and the task."""
    data = yaml.safe_load(PLANNER.read_text())
    if robot not in data["robots"] or task not in data["tasks"]:
        raise NotSupported(f"the automoma generator plans for robots {sorted(data['robots'])} and tasks "
                           f"{sorted(data['tasks'])}, not for robot {robot!r} + task {task!r}")


def build(instance, task: str, robot: str, repo: Path) -> Problem:
    """The plan inputs of the scene instance `instance` (its scene.yaml)."""
    check_supported(task, robot)
    data = yaml.safe_load(PLANNER.read_text())
    res = specs.resolve(Path(repo) / "specs", task, robot=robot)
    scene = specs.load(instance)
    handle = specs.parse_entity(res.task.targets["handle"])
    joint, threshold = task_joint(res.task)
    t = data["tasks"][task]
    poses = {k: list(map(float, v)) for k, v in res.robot.poses.items()}
    start = scene.robot_start
    moving = f"{handle.model}::{handle.name}::"
    boxes = [(b.name, b.pose, list(b.size)) for b in scene_boxes(scene, repo) if not b.name.startswith(moving)]
    pose = list(scene.models[handle.model].pose)
    sdf = specs.find_model(scene, scene.models[handle.model].model, repo)
    p = Problem(robot=robot, task=task, scene=scene.name, object=handle.model, handle=handle, world_pose=pose,
                start_base=list(start.base), start_base_obj=frames.base_from_world(start.base, pose).tolist(),
                ready=poses["ready"], start_arm=poses[start.arm] if isinstance(start.arm, str) else list(start.arm),
                folded=poses["folded"], tcp=res.robot.frames["tcp"], arm_dof=res.robot.arm_dof,
                joint=joint.name, goal=threshold + t["goal_margin"], boxes=boxes, model_file=sdf,
                cfg=data["robots"][robot], planner=data, grasp_along=list(t["grasp_along"]),
                door_spheres=t["door_spheres"], yaw_window=t["yaw_window"], floor=data["floor"])
    return p
