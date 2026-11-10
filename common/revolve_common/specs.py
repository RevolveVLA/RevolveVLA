"""Spec schemas (pydantic v2) and the loader.

specs/ layout: experiments/<exp>.yaml -> robots/<robot>/robot.yaml + tasks/<task>/task.yaml (robot-free, with
optional per-robot settings) -> scenes/<scene>/scene.yaml. Unknown keys are errors everywhere except the
blocks ported unchanged from the old plugin.yaml files (planner, card, the robot's planner/camera fields),
which are typed loosely.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from . import metrics

OPS = ("lt", "le", "gt", "ge", "eq")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ task
class Condition(Strict):
    """One metric compared with one threshold. hold_s: must hold at every sample of the last hold_s
    of the phase. window_s: the window the producer aggregates the metric over (it ends at the
    phase end). offline: the threshold the offline (expert replay) verdict uses instead. only: the
    condition applies to offline verdicts only, or to plan-driven runs only (not policy rollouts)."""
    metric: str
    lt: float | None = None
    le: float | None = None
    gt: float | None = None
    ge: float | None = None
    eq: float | bool | None = None
    joint: str | None = None
    hold_s: float | None = Field(default=None, ge=0)
    window_s: float | None = Field(default=None, gt=0)
    offline: float | None = None
    only: Literal["offline", "plan"] | None = None

    @model_validator(mode="after")
    def _one_op(self):
        set_ops = [o for o in OPS if getattr(self, o) is not None]
        if len(set_ops) != 1:
            raise ValueError(f"metric {self.metric}: exactly one of {OPS} is needed, got {set_ops or 'none'}")
        if self.metric not in metrics.REGISTRY:
            raise ValueError(f"metric: unknown metric {self.metric!r} (known: {sorted(metrics.REGISTRY)})")
        need = metrics.REGISTRY[self.metric].params
        if "joint" in need and not self.joint:
            raise ValueError(f"metric {self.metric}: needs joint")
        return self

    @property
    def op(self) -> str:
        return next(o for o in OPS if getattr(self, o) is not None)

    @property
    def threshold(self):
        return getattr(self, self.op)


class AllOf(Strict):
    all: list[Condition] = []


def _as_all(v):
    return v if v is None or (isinstance(v, dict) and "all" in v) else {"all": [v]}


class Phase(Strict):
    name: str
    end: Literal["gripper_close", "arm_motion", "gripper_open", "stream_end"]
    pass_: AllOf = Field(alias="pass")
    save_state: bool = True
    timeout_s: float | None = Field(default=None, gt=0)

    _norm = field_validator("pass_", mode="before")(_as_all)


class Hold(Strict):
    """validator.py HOLD: the gripped part's key, the part held open, its unit."""
    grip_part: str
    part: str
    unit: Literal["deg", "cm"]


class TaskObject(Strict):
    name: str
    gz_model: str
    world_pose: list[float] = Field(min_length=4, max_length=4)
    joint: str
    handle: str
    joint_topic: str
    goal: float                         # old goal_angle: rad (revolute) or m (prismatic)
    hold: Hold
    grasp_half_width_axes: list[int] = Field(min_length=1)   # half of min(handle size[a]) (validator.py)


class PolicyBoundaries(Strict):
    grip_closed_max_m: float = Field(gt=0)
    grasp_settle_s: float = Field(gt=0)
    joint_moved_eps: float = Field(gt=0)       # in object.hold.unit


class Boundaries(Strict):
    change_tol: float = Field(gt=0)
    policy: PolicyBoundaries


class Sanity(Strict):
    timed: list[Condition] = []                # fail the phase that holds the first violation
    untimed: list[Condition] = []              # fail the episode as "global" (no regeneration)


class Replay(Strict):
    standoff_m: float
    settle_s: float
    approach_s: float
    spawn_clearance_min_m: float
    retreat_m: float
    retreat_s: float
    hold_check: str
    pull_slowdown: float
    pull_min_s: float


class Rollout(Strict):
    budget_s: float = Field(gt=0)
    wait_s: float = Field(ge=0)
    after_success_s: float | None = None
    real_time_factor: float = Field(gt=0)


class Live(Strict):
    contact_wait_s: float = Field(ge=0)
    restore_pos_max_m: float = Field(gt=0)
    restore_rot_max_deg: float = Field(gt=0)


class TaskRobot(Strict):
    """A task's settings for one robot (task.yaml `robots: {<robot>: ...}`): merged over the task's own
    planner block (key by key) and replay block when the task is resolved for that robot."""
    planner: dict[str, Any] = {}
    replay: dict[str, float | str] = {}


class Task(Strict):
    name: str
    scene: str
    prompt: str
    card: dict[str, str] = {}
    object: TaskObject
    planner: dict[str, Any]
    replay: Replay
    budget_s: float = Field(gt=0)
    boundaries: Boundaries
    phases: list[Phase] = Field(min_length=1)
    sanity: Sanity
    rollout: Rollout
    live: Live
    robots: dict[str, TaskRobot] = {}       # per-robot settings (reach-dependent numbers), optional

    @model_validator(mode="after")
    def _phases(self):
        names = [p.name for p in self.phases]
        if len(set(names)) != len(names):
            raise ValueError(f"phases: duplicate names {names}")
        for p in self.phases:
            for c in p.pass_.all:
                if c.joint is not None and c.joint != self.object.joint:
                    raise ValueError(f"phases.{p.name}: joint {c.joint!r} is not the task joint {self.object.joint!r}")
        return self

    def phase(self, name):
        return next(p for p in self.phases if p.name == name)

    def for_robot(self, robot):
        """This task with its `robots.<robot>` settings merged in (the task as that robot runs it)."""
        o = self.robots.get(robot)
        if o is None:
            return self
        return self.model_copy(update=dict(planner={**self.planner, **o.planner},
                                           replay=self.replay.model_copy(update=o.replay)))


# ------------------------------------------------------------------ robot, scene
class Joints(Strict):
    base: list[str] = Field(min_length=3, max_length=3)   # the shared base chain: x, y [m] (world), yaw [rad]
    arm: list[str] = Field(min_length=1)
    gripper: list[str] = Field(min_length=1)               # actuated gripper joints, raw values in the state

    @computed_field
    @property
    def state(self) -> list[str]:
        """Order of the state log's robot_joint_* columns / observation.state: base, arm, gripper joints."""
        return [*self.base, *self.arm, *self.gripper]

    @computed_field
    @property
    def action(self) -> list[str]:
        """Order of the whole-body action: base, arm, `gripper` = the opening per contact link [m]."""
        return [*self.base, *self.arm, "gripper"]


class Camera(Strict):
    name: str
    link: str
    pose: list[float] = Field(min_length=6, max_length=6)
    rate: float
    width: int
    height: int
    hfov: float
    clip: list[float] = Field(min_length=2, max_length=2)


class Links(Strict):
    root: str                        # the arm's first link, fixed on the chassis (base_link) at `mount`
    hand: str                        # TCP = hand origin + tcp.hand_offset_z * hand z, hand orientation
    on_chassis: list[str]            # links mounted on the chassis (no clearance check against it)


class Tcp(Strict):
    link: str                        # the TCP frame in the model and the planner robot
    hand_offset_z: float


class Mount(Strict):
    xyz: list[float] = Field(min_length=3, max_length=3)   # links.root in base_link [m]


class Chassis(Strict):
    size_xy: list[float] = Field(min_length=2, max_length=2)   # box [m], from z 0.005 up to the mount
    mass: float = Field(gt=0)


class Base(Strict):
    """The base joints' control (P + damping feed-forward, clamped) and SDF values; v_max / w_max: the
    planning limits (peak of a min-jerk drive); limits: +- joint ranges [m, m, rad]."""
    kp: list[float] = Field(min_length=3, max_length=3)
    damping: list[float] = Field(min_length=3, max_length=3)    # SDF <damping> [N s/m, N m s/rad]
    f_max: list[float] = Field(min_length=3, max_length=3)
    v_max: float = Field(gt=0)
    w_max: float = Field(gt=0)
    limits: list[float] = Field(min_length=3, max_length=3)


class Arm(Strict):
    ready: list[float]               # the rest / IK-preference configuration [rad]
    kp: list[float]
    damping: list[float]             # SDF <damping> [N m s/rad], fed forward by the controller
    tau_max: list[float]
    # rotor inertia stand-in [kg m^2] added to each arm joint's child link (the gz MuJoCo plugin drops
    # <armature>); unset: the implementation's own
    armature: list[float] | None = None


class GripperSpec(Strict):
    contact_links: list[str] = Field(min_length=2, max_length=2)   # their handle contacts = finger_contacts
    hand_links: list[str]            # the links that may touch the grasped object
    open_m: float                    # opening per contact link [m] (gap / 2)
    closed_m: float
    force_n: float = Field(gt=0)     # max squeeze [N]
    # gripper joint position gain and SDF <damping>; unset: the implementation's own values
    kp: float | None = Field(None, gt=0)
    damping: float | None = Field(None, gt=0)
    armature: float | None = Field(None, ge=0)   # as arm.armature, for the gripper joint's link


class Reach(Strict):
    d_m: float = Field(gt=0)         # base_for_reach default distance from the robot's reach origin
    bearing_rad: float = 0.0


class Robot(Strict):
    name: str                        # spec name (specs/robots/<name>/)
    # simulator code folder robots/<implementation>/ (default: the spec name): robots that share one
    # implementation and differ only in their spec data
    implementation: str | None = None
    # uniform size factor the implementation applies to the description and to its own built-in geometry
    # (lengths x s, masses x s^3, inertias x s^5); the values in this file are absolute (not scaled)
    scale: float = Field(1.0, gt=0)
    model: str                       # gz model name (joint scope prefix "<model>::")
    description: str                 # the robot's model source (dir or URDF), repo-relative
    joints: Joints
    links: Links
    tcp: Tcp
    mount: Mount
    chassis: Chassis
    base: Base
    arm: Arm
    gripper: GripperSpec
    reach: Reach
    contract: str                    # Rosetta contract, relative to the robot spec dir
    camera_model: str
    gz_sdf: str
    cameras: list[Camera] = Field(min_length=1)
    planner: dict[str, Any]          # ported cuRobo fields (spheres, weights, limits)

    @model_validator(mode="after")
    def _sizes(self):
        if self.implementation is None:
            self.implementation = self.name
        n = len(self.joints.arm)
        for k in ("ready", "kp", "damping", "tau_max", "armature"):
            if getattr(self.arm, k) is not None and len(getattr(self.arm, k)) != n:
                raise ValueError(f"arm.{k}: {len(getattr(self.arm, k))} values for {n} arm joints")
        return self


class Scene(Strict):
    name: str
    world: str
    assets: list[str]
    skip: list[str] = []
    detailed: list[str] = []
    floor: dict[str, Any]


# ------------------------------------------------------------------ experiment
class SeedRange(Strict):
    lo: int = Field(ge=0)
    hi: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def _pair(cls, v):
        if isinstance(v, (list, tuple)):
            if len(v) != 2:
                raise ValueError("a seed range is [first, last] (inclusive)")
            return {"lo": v[0], "hi": v[1]}
        return v

    @model_validator(mode="after")
    def _order(self):
        if self.hi < self.lo:
            raise ValueError(f"seed range [{self.lo}, {self.hi}]: last < first")
        return self

    def seeds(self):
        return list(range(self.lo, self.hi + 1))


class Seeds(Strict):
    train: SeedRange
    probe: SeedRange
    heldout: SeedRange

    @model_validator(mode="after")
    def _disjoint(self):
        sets = {k: set(getattr(self, k).seeds()) for k in ("train", "probe", "heldout")}
        for a, b in (("train", "probe"), ("train", "heldout"), ("probe", "heldout")):
            common = sets[a] & sets[b]
            if common:
                raise ValueError(f"seeds.{a} and seeds.{b} overlap: {sorted(common)[:10]}")
        return self


class Datagen(Strict):
    episodes_per_seed: int = Field(ge=1)
    workers: int = Field(ge=1)


class Recovery(Strict):
    episodes_per_failure: int = Field(ge=1)
    phases: list[str] = Field(min_length=1)


class Training(Strict):
    policy: str
    base: str
    mode: Literal["finetune_last", "from_base", "finetune_recovery_only"] = "finetune_last"
    steps: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    backend: Literal["local", "gpu_machine", "modal"]
    backends: dict[str, dict[str, Any]] = {}

    @model_validator(mode="after")
    def _backend(self):
        if self.backend != "local" and self.backend not in self.backends:
            raise ValueError(f"training.backends: no settings for backend {self.backend!r}")
        return self


class Eval(Strict):
    rollouts_per_seed: int = Field(ge=1)
    retries: int = Field(ge=0)


class Experiment(Strict):
    name: str
    robot: str
    task: str
    max_turns: int = Field(ge=1)
    seeds: Seeds
    datagen: Datagen
    recovery: Recovery
    training: Training
    eval: Eval
    stall_timeout_s: float = Field(gt=0)


class Resolved(BaseModel):
    experiment: Experiment | None = None
    task: Task
    robot: Robot
    scene: Scene


# ------------------------------------------------------------------ loader
ROBOT_ENV = "REVOLVE_ROBOT"          # the robot of task-level loads (the CLI sets it from the experiment)
KINDS = {"experiments": Experiment, "tasks": Task, "robots": Robot, "scenes": Scene}
FILES = {"tasks": "task.yaml", "robots": "robot.yaml", "scenes": "scene.yaml"}


class SpecError(ValueError):
    pass


def robot_env():
    """{REVOLVE_ROBOT: its value} when set: what a launcher passes on to a job's container."""
    v = os.environ.get(ROBOT_ENV)
    return {ROBOT_ENV: v} if v else {}


def read_yaml(path):
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise SpecError(f"{path}: not a YAML mapping")
    return data


def find_root(path):
    """The specs/ dir that holds path (the nearest parent with an experiments/ or tasks/ child)."""
    p = Path(path).resolve()
    for d in [p, *p.parents]:
        if d.is_dir() and d.name == "specs":
            return d
    raise SpecError(f"{path}: not under a specs/ directory")


def kind_of(path, root):
    rel = Path(path).resolve().relative_to(root)
    if rel.parts and rel.parts[0] in KINDS:
        return rel.parts[0]
    raise SpecError(f"{path}: not under specs/{{{','.join(KINDS)}}}/")


def parse(model, path):
    from pydantic import ValidationError
    try:
        return model.model_validate(read_yaml(path))
    except ValidationError as e:
        lines = [f"{path}: {model.__name__} spec invalid"]
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            lines.append(f"  {loc}: {err['msg']}")
        raise SpecError("\n".join(lines)) from None


def spec_path(root, kind, name):
    p = Path(root) / kind / name / FILES[kind]
    if not p.is_file():
        raise SpecError(f"{kind[:-1]} {name!r}: no {p}")
    return p


def resolve_task(task, root, robot_name):
    """(the task as robot_name runs it, the robot, the scene); the robot's contract must name the robot's
    state and action joints in its observation.state / action selects."""
    for r in task.robots:
        if not (Path(root) / "robots" / r / FILES["robots"]).is_file():
            raise SpecError(f"task {task.name}: robots.{r}: no robot {r!r}")
    robot = parse(Robot, spec_path(root, "robots", robot_name))
    scene = parse(Scene, spec_path(root, "scenes", task.scene))
    contract = Path(root) / "robots" / robot.name / robot.contract
    if not contract.is_file():
        raise SpecError(f"robot {robot.name}: contract {contract} missing")
    c = read_yaml(contract)
    sel = {k: [x.removeprefix("position.") for x in v["select"]]
           for k, v in {**c.get("observations", {}), **c.get("actions", {})}.items() if "select" in v}
    for key, want in (("observation.state", robot.joints.state), ("action", robot.joints.action)):
        if sel.get(key) != want:
            raise SpecError(f"robot {robot.name}: contract {key} select {sel.get(key)} != the robot's joints {want}")
    return task.for_robot(robot.name), robot, scene


def load(path, robot=None):
    """Load and validate any spec file; experiments and tasks are resolved to their robot and scene.
    The robot: an experiment's `robot:`; for a task (robot-free) the `robot` argument, else $REVOLVE_ROBOT.
    -> Resolved (experiment/task) or the parsed Robot/Scene."""
    path = Path(path)
    root = find_root(path)
    kind = kind_of(path, root)
    spec = parse(KINDS[kind], path)
    if kind == "experiments":
        task = parse(Task, spec_path(root, "tasks", spec.task))
        bad = [p for p in spec.recovery.phases if p not in [x.name for x in task.phases]]
        if bad:
            raise SpecError(f"{path}: recovery.phases {bad} are not phases of task {task.name}")
        if spec.name != path.stem:
            raise SpecError(f"{path}: name {spec.name!r} != the file name {path.stem!r}")
        if robot is not None and robot != spec.robot:
            raise SpecError(f"{path}: robot {spec.robot!r}, not {robot!r}")
        task, rb, scene = resolve_task(task, root, spec.robot)
        return Resolved(experiment=spec, task=task, robot=rb, scene=scene)
    if spec.name != path.parent.name:
        raise SpecError(f"{path}: name {spec.name!r} != its folder {path.parent.name!r}")
    if kind == "tasks":
        robot = robot or os.environ.get(ROBOT_ENV)
        if not robot:
            raise SpecError(f"{path}: a task is robot-free: name the robot (an experiment's `robot:`, ${ROBOT_ENV})")
        task, rb, scene = resolve_task(spec, root, robot)
        return Resolved(task=task, robot=rb, scene=scene)
    return spec


def load_experiment(path) -> Resolved:
    r = load(path)
    if not isinstance(r, Resolved) or r.experiment is None:
        raise SpecError(f"{path}: not an experiment spec")
    return r
