"""What an expert needs of one (scene instance, task, robot), assembled once per sim launch (`Generator.prepare`): the
specs (task, robot, the instance's scene), the handle's geometry, the MoveIt planner and the robot side. The controller
names per robot are the only table here (robot.yaml does not carry them)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from revolve_common import specs

from .geometry import ModelGeometry, model_geometry

CASES = {       # controllers.yaml names, the hand's joint, its open / squeezing goals (DATAGEN-v2.md 5.1, 5.2), MoveIt group
    "mobile_panda": {"arm_controller": "panda_arm_controller", "hand_controller": "panda_hand_controller",
                     "arm_joints": [f"panda_joint{i}" for i in range(1, 8)], "hand_joint": "panda_finger_joint1",
                     "hand_open": 0.04, "hand_squeeze": -0.04, "group": "panda_arm"},
    "mobile_so101_xl": {"arm_controller": "arm_controller", "hand_controller": "gripper_controller",
                        "arm_joints": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                        "hand_joint": "gripper", "hand_open": 0.0, "hand_squeeze": -0.3, "group": "arm"},
}


@dataclass
class Setup:
    task: str
    robot: specs.Robot
    case: dict
    world: str
    handle: object            # the parsed task target `handle` (specs.Entity)
    geometry: ModelGeometry   # of the handle's model, joints at 0
    articulation: tuple       # (the joint that moves the handle's link, its descriptor)
    io: object                # RobotIO
    planner: object           # ArmPlanner

    @property
    def handle_box(self):
        """The handle bar's collision box (world frame at the closed door / drawer)."""
        return next(b for b in self.geometry.boxes if b.link == self.handle.name and b.name == self.handle.collision)

    @property
    def T_handle(self):
        return self.handle_box.T

    @property
    def boxes(self):
        return self.geometry.boxes


def robot_description(client, timeout_s: float = 30.0) -> str:
    """The sim's /robot_description (transient local String): the URDF that runs."""
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from std_msgs.msg import String
    got = []
    sub = client.node.create_subscription(String, "/robot_description", lambda m: got.append(m.data),
                                          QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    try:
        if not client.wait_for(lambda: bool(got), timeout_s):
            raise RuntimeError("no /robot_description")
    finally:
        client.node.destroy_subscription(sub)
    return got[0]


def load(instance, task: str, robot: str, repo: Path) -> dict:
    """The spec part of a Setup (no ROS): its fields but `io` and `planner`."""
    if robot not in CASES:
        raise ValueError(f"the moveit expert has no controller table for robot {robot!r} (known: {sorted(CASES)})")
    res = specs.resolve(repo / "specs", task, robot=robot)
    scene = specs.load(instance)                    # the instance: model poses (the robot start is the only variation)
    models, _ = specs.scene_models(scene, repo)
    handle = specs.parse_entity(res.task.targets["handle"])
    geometry = model_geometry(scene.models[handle.model].pose, models[handle.model].sdf)
    joint = next(j for j, v in geometry.joints.items() if v.child == handle.name)
    return dict(task=task, robot=res.robot, case=CASES[robot], world=scene.name, handle=handle, geometry=geometry,
                articulation=(joint, geometry.joints[joint]))


def build(client, instance, task: str, robot: str, repo: Path) -> Setup:
    from .planner import ArmPlanner
    from .robot_io import RobotIO
    kw = load(instance, task, robot, repo)
    io = RobotIO(client, robot, kw["world"], kw["case"], (kw["handle"].model, kw["articulation"][0]))
    frames = kw["robot"].frames
    planner = ArmPlanner(repo, robot, robot_description(client), kw["case"]["group"], frames["tcp"], frames["base"],
                         f"datagen_moveit_{robot}")
    return Setup(io=io, planner=planner, **kw)
