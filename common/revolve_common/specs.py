"""Spec schemas (pydantic v2) and the loader.

specs/ layout: experiments/<exp>.yaml -> robots/<robot>/robot.yaml + tasks/<task>/task.yaml (robot-free, with
optional per-robot settings) -> scenes/<scene>/scene.yaml. Unknown keys are errors everywhere except the
blocks ported unchanged from the old plugin.yaml files (planner, card, the robot's planner/camera fields),
which are typed loosely.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import (BaseModel, ConfigDict, Field, StrictFloat, StrictInt, computed_field, field_validator,
                      model_validator)

from . import metrics, policy_io

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


class Success(Strict):
    """Eval success of a policy rollout, judged from its state log (success.py): the object joint at or above
    joint_deg for hold_s, with the handle held for at least held_fraction of the travel from 0 to joint_deg. Held:
    gripper closed, both fingers on the handle within contact_s, each finger open half_width_m +- width_tol_m, the
    handle bar's centre (at height bar_z_m, the grasp pose turned with the joint) within bar_max_m of the TCP across
    the bar."""
    joint_deg: float = Field(gt=0)
    hold_s: float = Field(gt=0)
    held_fraction: float = Field(gt=0, le=1)
    contact_s: float = Field(gt=0)
    half_width_m: float = Field(gt=0)
    width_tol_m: float = Field(gt=0)
    bar_max_m: float = Field(gt=0)
    bar_z_m: float


class Live(Strict):
    contact_wait_s: float = Field(ge=0)
    restore_pos_max_m: float = Field(gt=0)
    restore_rot_max_deg: float = Field(gt=0)


# ---- start design: where a robot's base starts for a task and what may vary (task.yaml `starts:`, one block per robot)
# Numbers keep their yaml type (30 stays 30, not 30.0): the planner takes the block as ROUND_STARTS JSON text.
Num = StrictInt | StrictFloat
Pair = Annotated[list[Num], Field(min_length=2, max_length=2)]       # [low, high]


def _ordered(name, pair, positive=False):
    if not all(math.isfinite(x) for x in pair) or pair[0] > pair[1] or (positive and pair[0] <= 0):
        raise ValueError(f"{name}: {pair} is not low <= high" + (" with low > 0" if positive else ""))


class BoxStarts(Strict):
    """method box: clustered starts, `clusters` x `per_cluster` attempts in a box_m square around `centre` (object
    frame [m]) and a +-yaw_half_deg window about facing the object, each attempt jittered by offset_m / offset_deg."""
    method: Literal["box"]
    centre: Pair
    clusters: StrictInt = Field(ge=1)
    per_cluster: StrictInt = Field(ge=1)
    box_m: Num = Field(gt=0, le=0.40)
    yaw_half_deg: Num = Field(gt=0, le=30)
    offset_m: Num = Field(ge=0.01, le=0.03)
    offset_deg: Num = Field(ge=1, le=3)
    nominal_arm: Literal["ready"] = "ready"

    @model_validator(mode="after")
    def _fits(self):
        if not all(math.isfinite(x) for x in self.centre):
            raise ValueError(f"centre: {self.centre} is not finite")
        if self.box_m < 2 * self.offset_m or self.yaw_half_deg < self.offset_deg:
            raise ValueError(f"no start fits: box_m {self.box_m} < 2 offset_m {self.offset_m} or "
                             f"yaw_half_deg {self.yaw_half_deg} < offset_deg {self.offset_deg}")
        return self

    def block(self) -> dict:
        """The planner's ROUND_STARTS block (start_design.parse_starts)."""
        return self.model_dump(exclude={"method", "centre", "variants"}, exclude_none=True)

    def base_limits(self) -> dict:
        """How far a start may be from the box centre / from facing the object so that every jittered attempt stays
        inside the box and the window: +- {"x_m": h, "y_m": h, "yaw_deg": w}."""
        h = round(self.box_m / 2 - self.offset_m, 9)
        return {"x_m": h, "y_m": h, "yaw_deg": round(self.yaw_half_deg - self.offset_deg, 9)}


class ReachStarts(Strict):
    """method reach: per seed one grasp point along the bar (grasp_m, task frame), the reach distance d_m and a base
    turn of +-yaw_half_deg about the arm's first joint; the attempts of a seed jitter d and the turn."""
    method: Literal["reach"]
    grasp_m: Pair
    d_m: Pair
    yaw_half_deg: Num = Field(ge=0, le=30)
    bearing_deg: Num | None = Field(None, ge=-90, le=90)     # default: the robot's reach.bearing_rad
    jitter_m: Num = Field(ge=0, le=0.03)
    jitter_deg: Num = Field(ge=0, le=3)

    @model_validator(mode="after")
    def _ranges(self):
        _ordered("grasp_m", self.grasp_m)
        _ordered("d_m", self.d_m, positive=True)
        return self

    def block(self) -> dict:
        return {"mode": self.method, **self.model_dump(exclude={"method", "variants"}, exclude_none=True)}


class DriveStarts(ReachStarts):
    """method drive: a reach start plus a far start the robot drives in from with its arm folded: back_m behind the
    reach base, side_m sideways, the heading +-heading_half_deg from facing the object."""
    method: Literal["drive"]
    back_m: Pair
    side_m: Pair
    heading_half_deg: Num = Field(ge=0, le=45)
    slowdown: Num = Field(ge=1, le=10)                       # the replay runs the drive-in at slowdown x its time
    sequence: Literal["simultaneous", "drive_then_arm"] | None = None

    @model_validator(mode="after")
    def _drive_ranges(self):
        _ordered("back_m", self.back_m, positive=True)
        _ordered("side_m", self.side_m)
        return self


def _by_method(*types):
    return Annotated[Union[types], Field(discriminator="method")]


StartDesign = BoxStarts | ReachStarts | DriveStarts
_Variants = dict[str, _by_method(BoxStarts, ReachStarts, DriveStarts)]


class BoxEntry(BoxStarts):
    variants: _Variants = {}       # named alternatives (an experiment's datagen.variant), each replacing the block


class ReachEntry(ReachStarts):
    variants: _Variants = {}


class DriveEntry(DriveStarts):
    variants: _Variants = {}


class Grasp(Strict):
    """planner.grasp: how the TCP takes the handle, along `axis` of the object model frame. `points` are the defined
    grasps (ids 0..n-1, the planner's grasp set); `range` is the optional continuous interval along the axis a spec
    may draw a grasp from instead. depth (the TCP behind the handle centre, towards the door) and roll_deg are fixed
    unless depth_range / roll_range_deg allow a draw too."""
    axis: Literal["x", "y", "z"]
    points: list[Num] = Field(min_length=1)
    range: Pair | None = None
    depth: Num
    roll_deg: Num
    depth_range: Pair | None = None
    roll_range_deg: Pair | None = None

    @model_validator(mode="after")
    def _ordered_and_holding(self):
        for name, pair, fixed in (("range", self.range, None), ("depth_range", self.depth_range, self.depth),
                                  ("roll_range_deg", self.roll_range_deg, self.roll_deg)):
            if pair is None:
                continue
            _ordered(f"grasp.{name}", pair)
            if fixed is not None and not pair[0] <= fixed <= pair[1]:
                raise ValueError(f"grasp.{name} {pair} does not hold the fixed value {fixed}")
        return self


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
    success: Success | None = None          # eval success of policy rollouts (absent: only the phase verdict)
    robots: dict[str, TaskRobot] = {}       # per-robot settings (reach-dependent numbers), optional
    # per robot: where its base starts and what may vary; a robot without a block (or null) has no start design:
    # AutoMoMa's whole-body IK picks the base
    starts: dict[str, _by_method(BoxEntry, ReachEntry, DriveEntry) | None] = {}

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

    @model_validator(mode="after")
    def _grasp(self):
        Grasp.model_validate(self.planner.get("grasp"))
        return self

    @property
    def grasp(self) -> Grasp:
        """planner.grasp, checked (as the robot's override merges it when the task is resolved for the robot)."""
        return Grasp.model_validate(self.planner["grasp"])

    def phase(self, name):
        return next(p for p in self.phases if p.name == name)

    def starts_for(self, robot, variant=None) -> StartDesign | None:
        """The start design of `robot` (None: none), or of its named variant (an error when it has none)."""
        entry = self.starts.get(robot)
        if variant is None:
            return entry
        if entry is None or variant not in entry.variants:
            raise SpecError(f"datagen variant {variant!r}: task {self.name} has no starts.{robot}.variants.{variant}")
        return entry.variants[variant]

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

    @computed_field
    @property
    def policy_state(self) -> list[str]:
        """observation.state (the contract): the base twist in the robot frame, arm, gripper joints (policy_io)."""
        return [*policy_io.BASE_TWIST, *self.arm, *self.gripper]

    @computed_field
    @property
    def policy_action(self) -> list[str]:
        """action (the contract): the commanded base twist in the robot frame, arm targets, gripper opening."""
        return [*policy_io.BASE_TWIST, *self.arm, "gripper"]


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
    ready: list[float]               # the IK-preference configuration [rad]
    # the fully folded rest pose [rad] (in the joint limits, clear of the chassis): a `drive` start's arm
    folded: list[float] | None = None
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
        for k in ("ready", "folded", "kp", "damping", "tau_max", "armature"):
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
    # optional: a named variant of the task's start design (task.yaml starts.<robot>.variants),
    # passed to every job as $REVOLVE_DATAGEN_VARIANT; default: the task's own blocks
    variant: str | None = None


class Recovery(Strict):
    episodes_per_failure: int = Field(ge=1)
    phases: list[str] = Field(min_length=1)


class Training(Strict):
    policy: str
    base: str
    mode: Literal["finetune_last", "from_base", "finetune_recovery_only"] = "finetune_last"
    steps: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    save_every: int | None = Field(default=None, ge=1)    # also save every N steps -> turn_<k>/ckpt_<step>/
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
VARIANT_ENV = "REVOLVE_DATAGEN_VARIANT"   # an experiment's datagen.variant (unset: none)
KINDS = {"experiments": Experiment, "tasks": Task, "robots": Robot, "scenes": Scene}
FILES = {"tasks": "task.yaml", "robots": "robot.yaml", "scenes": "scene.yaml"}


class SpecError(ValueError):
    pass


def robot_env():
    """{REVOLVE_ROBOT: its value} (and {REVOLVE_DATAGEN_VARIANT: its value}) when set: what a launcher passes on to
    a job's container."""
    return {k: os.environ[k] for k in (ROBOT_ENV, VARIANT_ENV) if os.environ.get(k)}


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
    """(the task as robot_name runs it, the robot, the scene); the robot's contract must carry the robot's
    policy state and action (policy_io: base twist + joints) in its observation.state / action sources."""
    for key, names in (("robots", task.robots), ("starts", task.starts)):
        for r in names:
            if not (Path(root) / "robots" / r / FILES["robots"]).is_file():
                raise SpecError(f"task {task.name}: {key}.{r}: no robot {r!r}")
    robot = parse(Robot, spec_path(root, "robots", robot_name))
    entry = task.starts.get(robot_name)
    for d in ([] if entry is None else [entry, *entry.variants.values()]):
        if d.method == "drive" and robot.arm.folded is None:
            raise SpecError(f"task {task.name}: starts.{robot_name} drives in, but robot {robot_name} "
                            "has no arm.folded")
    scene = parse(Scene, spec_path(root, "scenes", task.scene))
    contract = Path(root) / "robots" / robot.name / robot.contract
    if not contract.is_file():
        raise SpecError(f"robot {robot.name}: contract {contract} missing")
    try:
        lay = policy_io.contract_layout(read_yaml(contract))
    except (ValueError, KeyError, TypeError) as e:
        raise SpecError(f"robot {robot.name}: contract {contract}: {e}") from None
    for key, want in (("observation.state", robot.joints.policy_state), ("action", robot.joints.policy_action)):
        got = policy_io.names(lay[key]) if key in lay else None
        if got != want:
            raise SpecError(f"robot {robot.name}: contract {key} {got} != the robot's policy {key} {want}")
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
