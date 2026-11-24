"""Spec schemas (pydantic v2), the metric registry (names + parameters) and the loader.

specs/ layout (every name = its folder name):
  scenes/<s>/scene.yaml     one fixed scene: the models (model:// names in the asset search path) and the robot start
  robots/<r>/robot.yaml     the robot: description (xacro), controllers, named poses, frame and link-group aliases
  robots/<r>/contract.yaml  the robot's Rosetta contract (dataset / policy I/O; check_contract)
  tasks/<t>/task.yaml       the task on a scene: prompt, targets, ordered phases, budget, sanity overrides
  tasks/sanity.yaml         the sanity checks every task gets (a keyed map of checks, nothing else)

A condition is one metric with its parameters, one comparison and an optional hold time:
  {metric: distance, from: robot::tcp, to: handle, le: 0.015}
  {metric: contacts, from: robot::fingers, to: handle, eq: 2, for_s: 0.5}     # true for the last 0.5 s
Values are SI: m, rad, s, m/s. Entities (the metric parameters that name things):
  robot                       the whole robot
  robot::<alias>              a robot.yaml frame (a link) or group (links); checked when a task meets a robot
  <model>                     a whole scene model
  <model>::<link|joint>       a link or joint of a scene model (from its SDF)
  <model>::<link>::<collision>  one collision of that link (e.g. a handle)
  <target>                    a task target alias (targets: {handle: fridge::top_door::top_door_handle})
  any, any_except_targets     contacts `to` only: every body except the robot itself and the ground (and the targets)
Unknown keys, duplicate keys and non-finite numbers (.nan, .inf) are errors everywhere. The scene name, the scene's
model names and the robot name are parts of ROS 2 topic names: letters, digits and _, not starting with a digit.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

OPS = ("lt", "le", "eq", "ge", "gt")
AXES = ("x", "-x", "y", "-y", "z", "-z")
SANITY_FILE = "sanity.yaml"
GLOBAL_PHASE = "global"          # where an untimed sanity failure is attributed (not a phase name)

# ------------------------------------------------------------------ metric registry
# Entity kinds a parameter accepts. A robot frame is one link; a group is several links (robot::<group>, robot,
# a whole model); `any` covers any / any_except_targets.
JOINT, LINK, COLLISION, FRAME, GROUP, ANY = "joint", "link", "collision", "frame", "group", "any"
POINT = frozenset({LINK, COLLISION, FRAME})       # one body: a pose (a collision: its geometry, for distance)
BODIES = POINT | {GROUP}
ANY_NAMES = ("any", "any_except_targets")


@dataclass(frozen=True)
class Param:
    type: Literal["entity", "axes", "patterns"]
    kinds: frozenset = frozenset()   # entity: the entity kinds it takes
    default: Any = None              # None: required


@dataclass(frozen=True)
class Metric:
    unit: str
    doc: str
    params: dict[str, Param] = field(default_factory=dict)


def _e(kinds, default=None):
    return Param("entity", frozenset(kinds), default)


METRICS: dict[str, Metric] = {
    "joint": Metric("rad | m", "position of a scene joint", {"of": _e({JOINT})}),
    "distance": Metric("m", "distance between two bodies: a frame/link by its origin, a collision by its geometry "
                       "(0 inside it)", {"from": _e(POINT), "to": _e(POINT)}),
    "angle": Metric("rad", "angle (0..pi) between axis axes[0] of `from` and axis axes[1] of `to` (signed axes "
                    "x, -x, y, -y, z, -z of each frame; a collision's frame is its pose)",
                    {"from": _e(POINT), "to": _e(POINT), "axes": Param("axes")}),
    "contacts": Metric("links", "number of links of `from` in contact with `to`",
                       {"from": _e(BODIES), "to": _e(BODIES | {ANY})}),
    "gripper_opening": Metric("m", "distance between the origins of the two links of a robot group",
                              {"of": _e({GROUP})}),
    "slip": Metric("m", "drift of `from` relative to `to` (position in `to`'s frame) since the phase start",
                   {"from": _e(POINT), "to": _e(POINT)}),
    "peak_body_speed": Metric("m/s", "highest link speed of `of` (default: the robot)", {"of": _e(BODIES, "robot")}),
    "penetration": Metric("m", "deepest contact involving `of` (default: the robot)", {"of": _e(BODIES, "robot")}),
    "non_finite_commands": Metric("commands", "commands with a NaN or Inf value sent to the robot"),
    "gz_log_problem_lines": Metric("lines", "gz error lines, except those matching an `ignore` regex",
                                   {"ignore": Param("patterns", default=())}),
}


class SpecError(ValueError):
    pass


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)   # every number finite


# ------------------------------------------------------------------ conditions
class Condition(Strict):
    """Parsed from {metric, <params>, <op>: value, for_s?}. for_s: the comparison held for the last for_s seconds."""
    metric: str
    params: dict[str, Any] = {}
    op: Literal["lt", "le", "eq", "ge", "gt"]
    value: float
    for_s: float | None = Field(default=None, gt=0)

    EXTRA_KEYS: ClassVar[tuple[str, ...]] = ()

    @model_validator(mode="before")
    @classmethod
    def _from_yaml(cls, v):
        if not isinstance(v, dict):
            raise ValueError(f"a condition is a mapping {{metric: ..., <op>: value}}, got {v!r}")
        v = dict(v)
        name = v.pop("metric", None)
        if name not in METRICS:
            raise ValueError(f"metric: unknown metric {name!r} (known: {', '.join(METRICS)})")
        m = METRICS[name]
        ops = [o for o in OPS if o in v]
        if len(ops) != 1:
            unknown = sorted(set(v) - {*m.params, *OPS, "for_s", *cls.EXTRA_KEYS})
            raise ValueError(f"metric {name}: exactly one of {', '.join(OPS)} is needed, got {ops or 'none'}"
                             + (f" (unknown key(s) {unknown})" if unknown else ""))
        value = v.pop(ops[0])
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"metric {name}: {ops[0]} needs a number, got {value!r}")
        try:
            finite = math.isfinite(value)
        except OverflowError:        # an int too large for a float
            finite = False
        if not finite:
            raise ValueError(f"metric {name}: {ops[0]} needs a finite number, got {value!r}")
        out = {"metric": name, "op": ops[0], "value": value}
        for k in ("for_s", *cls.EXTRA_KEYS):
            if k in v:
                out[k] = v.pop(k)
        params = {}
        for k, p in m.params.items():
            if k in v:
                params[k] = _param(name, k, p, v.pop(k))
            elif p.default is None:
                raise ValueError(f"metric {name}: missing parameter {k!r} ({_describe(p)})")
            else:
                params[k] = p.default
        if v:
            allowed = [*m.params, *OPS, "for_s", *cls.EXTRA_KEYS]
            raise ValueError(f"metric {name}: unknown key(s) {sorted(v)} (allowed: {', '.join(allowed)})")
        out["params"] = params
        return out

    def entities(self):
        """{param: entity string} of the entity parameters."""
        return {k: self.params[k] for k, p in METRICS[self.metric].params.items() if p.type == "entity"}


class SanityCheck(Condition):
    """A sanity check: a condition, timed (default: fails the phase active at the first violation) or untimed
    (`timed: false`: fails the episode as phase `global`)."""
    timed: bool = True

    EXTRA_KEYS: ClassVar[tuple[str, ...]] = ("timed",)


def _describe(p):
    return {"entity": "an entity", "axes": f"[from axis, to axis], each one of {', '.join(AXES)}",
            "patterns": "a list of regexes"}[p.type]


def _param(metric, key, p, v):
    if p.type == "entity":
        if not isinstance(v, str):
            raise ValueError(f"metric {metric}: {key} must be an entity name, got {v!r}")
        parse_entity(v)
        return v
    if p.type == "axes":
        if not (isinstance(v, list) and len(v) == 2 and all(a in AXES for a in v)):
            raise ValueError(f"metric {metric}: {key} must be {_describe(p)}, got {v!r}")
        return tuple(v)
    if not (isinstance(v, list) and all(isinstance(s, str) for s in v)):
        raise ValueError(f"metric {metric}: {key} must be {_describe(p)}, got {v!r}")
    for s in v:
        try:
            re.compile(s)
        except re.error as e:
            raise ValueError(f"metric {metric}: {key}: bad regex {s!r}: {e}") from None
    return tuple(v)


# ------------------------------------------------------------------ entities
NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")
TOPIC_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")   # a ROS 2 topic name token (no - or ., no leading digit)
RESERVED = ("robot", *ANY_NAMES)


@dataclass(frozen=True)
class Entity:
    text: str
    model: str | None = None         # a scene model, or None for robot / any
    name: str | None = None          # robot alias, or a model's link/joint
    collision: str | None = None
    robot: bool = False
    any: bool = False


def parse_entity(text):
    """Syntax only: -> Entity (a bare name is a target alias or a whole model; resolved by check_entity)."""
    if text in ANY_NAMES:
        return Entity(text, any=True)
    parts = text.split("::")
    if not all(NAME.match(p) for p in parts) or len(parts) > 3:
        raise ValueError(f"entity {text!r}: not one of robot, robot::<alias>, <model>, <model>::<link|joint>, "
                         f"<model>::<link>::<collision>, a target alias, {', '.join(ANY_NAMES)}")
    if parts[0] == "robot":
        if len(parts) == 3:
            raise ValueError(f"entity {text!r}: robot entities are robot or robot::<alias>")
        return Entity(text, robot=True, name=parts[1] if len(parts) == 2 else None)
    return Entity(text, model=parts[0], name=parts[1] if len(parts) > 1 else None,
                  collision=parts[2] if len(parts) == 3 else None)


@dataclass
class ModelInfo:
    """A scene model's names, from its SDF (None: assets not checked)."""
    sdf: Path | None = None
    links: dict[str, set[str]] = field(default_factory=dict)       # link -> its collision names
    joints: dict[str, tuple[str, float | None, float | None]] = field(default_factory=dict)  # type, lower, upper


def check_entity(text, kinds, models, targets, robot=None):
    """Errors (strings) for entity `text` in a parameter taking `kinds`. models: {entity: ModelInfo} of the scene
    (empty ModelInfo: assets not checked, link/joint names not checked); targets: the task's; robot: a Robot to
    check robot::<alias> against (None: later, see check_robot_refs)."""
    e = parse_entity(text)
    if not e.robot and not e.any and e.name is None and e.model in targets:
        e = parse_entity(targets[e.model])
    if e.any:
        return [] if ANY in kinds else [f"{text}: {', '.join(ANY_NAMES)} only work as contacts `to`"]
    if e.robot:
        if e.name is None:
            return [] if GROUP in kinds else [f"{text}: the whole robot is a link group; {_kinds(kinds)} needed"]
        if robot is None:
            return []
        kind = FRAME if e.name in robot.frames else GROUP if e.name in robot.groups else None
        if kind is None:
            return [f"{text}: robot {robot.name} has no frame or group {e.name!r} (frames: "
                    f"{', '.join(robot.frames) or 'none'}; groups: {', '.join(robot.groups) or 'none'})"]
        return [] if kind in kinds else [f"{text}: robot {kind} {e.name!r}, but {_kinds(kinds)} needed"]
    if e.model not in models:
        what = "a target alias or scene model" if e.name is None else "a scene model"
        return [f"{text}: {e.model!r} is not {what} (models: {', '.join(models)}"
                + (f"; targets: {', '.join(targets)})" if targets else ")")]
    info = models[e.model]
    if e.name is None:
        return [] if GROUP in kinds else [f"{text}: a whole model is a link group; {_kinds(kinds)} needed"]
    if info.sdf is None:                 # assets not checked: syntax + model only
        kind = COLLISION if e.collision else None
        return [] if kind is None or kind in kinds else [f"{text}: a collision, but {_kinds(kinds)} needed"]
    if e.collision is not None:
        if e.name not in info.links:
            return [f"{text}: model {e.model} has no link {e.name!r} (links: {', '.join(info.links)})"]
        if e.collision not in info.links[e.name]:
            return [f"{text}: link {e.model}::{e.name} has no collision {e.collision!r} "
                    f"(collisions: {', '.join(sorted(info.links[e.name]))})"]
        return [] if COLLISION in kinds else [f"{text}: a collision, but {_kinds(kinds)} needed"]
    found = [k for k, names in ((LINK, info.links), (JOINT, info.joints)) if e.name in names]
    if not found:
        return [f"{text}: model {e.model} ({info.sdf}) has no link or joint {e.name!r} (links: "
                f"{', '.join(info.links)}; joints: {', '.join(info.joints)})"]
    return [] if set(found) & kinds else [f"{text}: a {found[0]}, but {_kinds(kinds)} needed"]


def _kinds(kinds):
    return "a " + " or ".join(sorted(kinds - {ANY}) + (list(ANY_NAMES) if ANY in kinds else []))


# ------------------------------------------------------------------ scene
def _name(v):
    if not NAME.match(v) or v in RESERVED:
        raise ValueError(f"{v!r}: names are letters, digits, _ . - and not one of {', '.join(RESERVED)}")
    return v


def _topic_name(v):
    """The scene name, a scene model's entity name and the robot name are parts of ROS 2 topic names
    (/world/<scene>/..., /model/<model>/pose, /model/<robot>/pose)."""
    if not TOPIC_NAME.match(v):
        raise ValueError(f"{v!r}: this name is part of ROS 2 topic names: letters, digits and _, not starting "
                         f"with a digit")
    return v


class SceneModel(Strict):
    model: str                                   # model:// name (the folder in the asset search path)
    pose: list[float] = Field(default=[0.0, 0.0, 0.0, 0.0], min_length=4, max_length=4)   # x, y, z [m], yaw [rad]
    joints: dict[str, float] = {}                # initial joint positions [rad | m]

    @field_validator("model")
    @classmethod
    def _uri(cls, v):
        v = v.removeprefix("model://")
        if not NAME.match(v):
            raise ValueError(f"model {v!r}: a model:// name (a folder in the asset search path)")
        return v


class RobotStart(Strict):
    base: list[float] = Field(min_length=3, max_length=3)   # x, y [m], yaw [rad] in the world
    arm: str | list[float]                                  # a robot.yaml pose name, or joint positions [rad]


class Scene(Strict):
    name: str
    assets: list[str] = Field(min_length=1)      # search path for model:// names, repo-relative; first match wins
    models: dict[str, SceneModel] = Field(min_length=1)
    robot_start: RobotStart

    @field_validator("name")
    @classmethod
    def _scene_name(cls, v):
        return _topic_name(v)

    @field_validator("models")
    @classmethod
    def _names(cls, v):
        for k in v:
            _topic_name(_name(k))
        return v


def find_model(scene, name, repo):
    """The SDF of model://name: the first search dir with <name>/model.config (its <sdf>) or <name>/model.sdf."""
    for d in scene.assets:
        base = (Path(repo) / d / name)
        cfg = base / "model.config"
        if cfg.is_file():
            try:
                sdf = ET.parse(cfg).getroot().findtext("sdf")
            except ET.ParseError as e:
                raise SpecError(f"{cfg}: {e}") from None
            if sdf and (base / sdf.strip()).is_file():
                return base / sdf.strip()
        if (base / "model.sdf").is_file():
            return base / "model.sdf"
    return None


def read_model(sdf):
    """ModelInfo of an SDF file: the top-level model's links (with their collisions) and joints."""
    try:
        root = ET.parse(sdf).getroot()
    except ET.ParseError as e:
        raise SpecError(f"{sdf}: {e}") from None
    model = root.find("model") if root.tag == "sdf" else root
    if model is None or model.tag != "model":
        raise SpecError(f"{sdf}: no <model>")
    info = ModelInfo(sdf=Path(sdf))
    for link in model.findall("link"):
        info.links[link.get("name")] = {c.get("name") for c in link.findall("collision")}
    for j in model.findall("joint"):
        lo, hi = (j.findtext(f"axis/limit/{k}") for k in ("lower", "upper"))
        info.joints[j.get("name")] = (j.get("type"), float(lo) if lo else None, float(hi) if hi else None)
    return info


def scene_models(scene, repo, assets=True):
    """({entity: ModelInfo}, errors): resolves every model:// name and checks the initial joint positions.
    assets=False: no file access (empty ModelInfo each)."""
    models, errors = {}, []
    for ent, m in scene.models.items():
        if not assets:
            models[ent] = ModelInfo()
            continue
        sdf = find_model(scene, m.model, repo)
        if sdf is None:
            errors.append(f"models.{ent}.model: model://{m.model} is in none of {scene.assets} (under {repo})")
            models[ent] = ModelInfo()
            continue
        info = models[ent] = read_model(sdf)
        for j, q in m.joints.items():
            if j not in info.joints:
                errors.append(f"models.{ent}.joints.{j}: {sdf} has no joint {j!r} "
                              f"(joints: {', '.join(info.joints) or 'none'})")
                continue
            typ, lo, hi = info.joints[j]
            if typ == "fixed":
                errors.append(f"models.{ent}.joints.{j}: a fixed joint")
            elif (lo is not None and q < lo) or (hi is not None and q > hi):
                errors.append(f"models.{ent}.joints.{j}: {q} is outside the joint limits [{lo}, {hi}]")
    return models, errors


def check_sim_support(scene):
    """Errors for what a valid scene may hold but the simulator cannot start yet (`./revolve validate` and the
    simulator's world generator both refuse it): a nonzero initial joint position (gz-sim has no upstream way to
    start a joint away from its SDF zero; only position-holding controllers)."""
    return [f"models.{ent}.joints {nonzero}: nonzero initial joint positions are not supported by the simulator yet"
            for ent, m in scene.models.items() if (nonzero := {j: q for j, q in m.joints.items() if q != 0.0})]


# ------------------------------------------------------------------ robot
class Robot(Strict):
    name: str
    description: str                 # the robot's xacro (URDF), relative to the robot's folder
    controllers: str                 # ros2_control controllers yaml, relative to the robot's folder
    poses: dict[str, list[float]] = Field(min_length=1)   # named arm poses [rad], arm joint order
    frames: dict[str, str] = {}      # alias -> link (robot::<alias>, e.g. tcp, hand, base)
    groups: dict[str, list[str]] = {}   # alias -> links (e.g. fingers)

    @field_validator("name")
    @classmethod
    def _robot_name(cls, v):
        return _topic_name(v)

    @model_validator(mode="after")
    def _check(self):
        sizes = {len(p) for p in self.poses.values()}
        if len(sizes) != 1 or 0 in sizes:
            raise ValueError(f"poses: every pose needs the same, nonzero number of joints, got "
                             f"{ {k: len(p) for k, p in self.poses.items()} }")
        for k in (*self.frames, *self.groups):
            _name(k)
        both = set(self.frames) & set(self.groups)
        if both:
            raise ValueError(f"{sorted(both)}: both a frame and a group")
        for k, links in self.groups.items():
            if not links:
                raise ValueError(f"groups.{k}: no links")
        return self

    @property
    def arm_dof(self):
        return len(next(iter(self.poses.values())))


def check_robot_files(robot, folder):
    """Errors: the description and controllers files exist; if xacro is importable, the aliased links exist."""
    errors = []
    for k in ("description", "controllers"):
        if not (Path(folder) / getattr(robot, k)).is_file():
            errors.append(f"{k}: no file {Path(folder) / getattr(robot, k)}")
    if errors:
        return errors
    try:
        import xacro
    except ImportError:              # no xacro (e.g. the host venv): the simulator image checks the links
        return errors
    try:
        urdf = xacro.process_file(str(Path(folder) / robot.description)).toxml()
        links = {lk.get("name") for lk in ET.fromstring(urdf).findall("link")}
    except Exception as e:           # xacro raises many types
        return [f"description: xacro failed: {e}"]
    for k, link in robot.frames.items():
        if link not in links:
            errors.append(f"frames.{k}: no link {link!r} in the description")
    for k, ls in robot.groups.items():
        errors += [f"groups.{k}: no link {lk!r} in the description" for lk in ls if lk not in links]
    return errors


# ------------------------------------------------------------------ robot contract (Rosetta)
# specs/robots/<r>/contract.yaml: the robot's Rosetta contract (ThirdParty/rosetta docs/reference/contract.md), hand
# written; the dataset component and a policy runner load it with Rosetta itself. Checked here without Rosetta: its
# topics against the simulator's recorded topics, its joints against the robot's controllers, the fixed image keys
# and frame rate (components/dataset/DATASET-v2.md).
CONTRACT_FILE = "contract.yaml"
CONTRACT_FPS = 30
CONTRACT_IMAGES = ("observation.images.wrist", "observation.images.overhead_left", "observation.images.overhead_right")
CONTRACT_TASK_TOPIC = "/task/prompt"
CONTRACT_SECTIONS = ("observations", "actions", "rewards", "signals", "info", "complementary_data")
RECORD_YAML = Path("components") / "simulator" / "ros" / "revolve_sim" / "config" / "record.yaml"
JTC = "joint_trajectory_controller/JointTrajectoryController"
OMNI = "omni_wheel_drive_controller/OmniWheelDriveController"
JOINT_MSGS = ("sensor_msgs/msg/JointState", "trajectory_msgs/msg/JointTrajectory")


def robot_controllers(robot, folder):
    """{controller: (type, [joints])} of the robot's controllers.yaml (JTC joints; omni base wheel_names)."""
    data = yaml.safe_load((Path(folder) / robot.controllers).read_text())
    cm = data["controller_manager"]["ros__parameters"]
    out = {}
    for name, v in cm.items():
        if isinstance(v, dict) and "type" in v:
            p = (data.get(name) or {}).get("ros__parameters", {})
            out[name] = (v["type"], list(p.get("joints") or p.get("wheel_names") or []))
    return out


def robot_cameras(robot, folder):
    """The robot's camera sensor names from its description, or None without xacro (e.g. the host venv)."""
    try:
        import xacro
    except ImportError:
        return None
    urdf = xacro.process_file(str(Path(folder) / robot.description)).toxml()
    return [s.get("name") for g in ET.fromstring(urdf).findall("gazebo") for s in g.findall("sensor")
            if s.get("type") == "camera"]


def recorded_topics(record, robot, controllers, cameras=None):
    """record.yaml's topics (+ images) as regexes, filled in for this robot: {robot}, {jtc}, {omni}, {camera} (any one
    name when cameras is None); the scene's templates ({world}, {scene_model}, {model}, {contact_link}) match any."""
    fill = {"robot": [robot.name], "jtc": [n for n, (t, _) in controllers.items() if t == JTC],
            "omni": [n for n, (t, _) in controllers.items() if t == OMNI], "camera": cameras}
    rec = yaml.safe_load(Path(record).read_text())
    out = []
    for entry in rec["topics"] + rec.get("images", []):
        parts = re.split(r"\{(\w+)\}", entry)          # literal, name, literal, ...
        rx = "".join(re.escape(p) if i % 2 == 0 else
                     ("(" + "|".join(map(re.escape, fill[p])) + ")" if fill.get(p) is not None else "[^/]+")
                     for i, p in enumerate(parts))
        out.append(re.compile(rx))
    return out


def _sources(data):
    """[(where, source)] of every frame source of a contract (a key holds one source or a list)."""
    out = []
    for sec in CONTRACT_SECTIONS:
        for key, spec in (data.get(sec) or {}).items():
            srcs = spec if isinstance(spec, list) else [spec]
            out += [(f"{sec}.{key}" + (f"[{i}]" if isinstance(spec, list) else ""), s) for i, s in enumerate(srcs)]
    return out


def _joints(select):
    """A JointState / JointTrajectory select -> joint names (`<field>.<joint>`, or a bare joint)."""
    return [s.split(".", 1)[1] if "." in s else s for s in select or []]


def check_contract(robot, folder, record=None, cameras=False):
    """Errors of the robot's contract.yaml: it exists and parses; robot_type = the robot, robot_interface ros2, fps
    CONTRACT_FPS; the image keys are exactly CONTRACT_IMAGES; a `tasks` channel on CONTRACT_TASK_TOPIC; every topic is
    one the simulator records (record: record.yaml, default the repo's of the specs/ dir holding folder); every
    joint selected from /joint_states or a joint_trajectory topic is a controller joint; each /<jtc>/joint_trajectory selects exactly that
    controller's joints in order, every JTC has one, and /joint_states selects the same joints in the same order (the
    state mirrors the action). cameras: the robot's camera names (False: from the description if xacro is importable,
    None: any name)."""
    folder = Path(folder)
    path = folder / CONTRACT_FILE
    if not path.is_file():
        return [f"contract: no file {path}"]
    try:
        data = {k: v for k, v in read_yaml(path).items() if not str(k).startswith("x-")}
    except SpecError as e:
        return [f"contract: {e}"]
    record = Path(record) if record else find_root(folder).parent / RECORD_YAML
    controllers = robot_controllers(robot, folder)
    if cameras is False:
        cameras = robot_cameras(robot, folder)
    errors = []
    for k, want in (("robot_type", robot.name), ("robot_interface", "ros2"), ("fps", CONTRACT_FPS)):
        if data.get(k) != want:
            errors.append(f"contract: {k} {data.get(k)!r}, want {want!r}")
    try:
        images = sorted(k for k in data.get("observations") or {} if str(k).startswith("observation.images."))
        if images != sorted(CONTRACT_IMAGES):
            errors.append(f"contract: image keys {images}, want exactly {sorted(CONTRACT_IMAGES)}")
        sources = _sources(data)
        channels = [(w, s["channel"]) for w, s in sources] + \
            [(f"tasks.{k}", t["channel"]) for k, t in (data.get("tasks") or {}).items()] + \
            [(f"adjunct[{i}]", a["channel"]) for i, a in enumerate(data.get("adjunct") or [])]
        if not any(ch["topic"] == CONTRACT_TASK_TOPIC for w, ch in channels if w.startswith("tasks.")):
            errors.append(f"contract: no tasks channel on {CONTRACT_TASK_TOPIC}")
        topics = recorded_topics(record, robot, controllers, cameras)
        errors += [f"contract: {w}: topic {ch['topic']} is not recorded by the simulator ({record.name})"
                   for w, ch in channels if not any(rx.fullmatch(ch["topic"]) for rx in topics)]
        robot_joints = {j for _, js in controllers.values() for j in js}
        state, action, acted = [], [], set()
        for w, s in sources:
            ch = s["channel"]
            if ch["type"] not in JOINT_MSGS or not (ch["topic"] == "/joint_states" or
                                                    ch["topic"].endswith("/joint_trajectory")):
                continue                            # e.g. a scene's joint states (record-only)
            joints = _joints(s.get("select"))
            errors += [f"contract: {w}: {j!r} is not a joint of robot {robot.name}'s controllers"
                       for j in joints if j not in robot_joints]
            if ch["topic"] == "/joint_states" and w.startswith("observations."):
                state += joints
            name = ch["topic"].strip("/").split("/")[0]
            if ch["topic"].endswith("/joint_trajectory") and w.startswith("actions."):
                if controllers.get(name, (None,))[0] != JTC:
                    errors.append(f"contract: {w}: {ch['topic']}: {name} is not a joint trajectory controller")
                    continue
                if joints != controllers[name][1]:
                    errors.append(f"contract: {w}: selects {joints}, controller {name} has {controllers[name][1]}")
                action += joints
                acted.add(name)
        missing = sorted(n for n, (t, _) in controllers.items() if t == JTC and n not in acted)
        if missing:
            errors.append(f"contract: no action on the joint_trajectory of {missing}")
        if state != action:
            errors.append(f"contract: /joint_states state joints {state} != the action joints {action}")
    except (KeyError, TypeError, AttributeError) as e:
        errors.append(f"contract: malformed ({type(e).__name__}: {e}); Rosetta's loader gives the details")
    return errors


# ------------------------------------------------------------------ task
class Phase(Strict):
    """Ordered: all `done` conditions true -> the next phase; a `while` condition false, or timeout_s passed in this
    phase -> the task fails in this phase."""
    model_config = ConfigDict(populate_by_name=True)
    name: str
    done: list[Condition] = Field(min_length=1)
    while_: list[Condition] = Field(default=[], alias="while")
    timeout_s: float | None = Field(default=None, gt=0)


class Task(Strict):
    name: str
    scene: str
    prompt: str                      # the language instruction
    targets: dict[str, str] = {}     # alias -> a scene entity (<model>::<link|joint>[::<collision>])
    phases: list[Phase] = Field(min_length=1)
    budget_s: float = Field(gt=0)    # the whole episode
    sanity: dict[str, SanityCheck | None] = {}   # overrides/additions to tasks/sanity.yaml by key; null removes

    @model_validator(mode="after")
    def _check(self):
        names = [p.name for p in self.phases]
        if len(set(names)) != len(names):
            raise ValueError(f"phases: duplicate names {names}")
        if GLOBAL_PHASE in names:
            raise ValueError(f"phases: {GLOBAL_PHASE!r} is reserved (untimed sanity failures)")
        for k, v in self.targets.items():
            _name(k)
            e = parse_entity(v)
            if e.robot or e.any or e.name is None:
                raise ValueError(f"targets.{k}: {v!r} is not a scene entity <model>::<link|joint>[::<collision>]")
        return self

    def conditions(self):
        """[(location, condition)] of every phase condition."""
        return [(f"phases.{p.name}.{k}[{i}]", c) for p in self.phases
                for k, cs in (("done", p.done), ("while", p.while_)) for i, c in enumerate(cs)]


def merge_sanity(base, overrides):
    """base (tasks/sanity.yaml) <- a task's overrides: a check replaces or adds by key, null removes the key."""
    out = dict(base)
    for k, v in overrides.items():
        if v is None:
            if k not in out:
                raise SpecError(f"sanity.{k}: null removes a base check, but tasks/{SANITY_FILE} has no {k!r} "
                                f"(checks: {', '.join(base)})")
            del out[k]
        else:
            out[k] = v
    return out


def check_task(task, models, sanity, robot=None):
    """Errors: every entity of the task's conditions and merged sanity checks against the scene's models (and the
    robot, when given)."""
    errors = []
    for k, v in task.targets.items():
        errors += [f"targets.{k}: {m}" for m in check_entity(v, POINT | {GROUP, JOINT}, models, {})]
    for k in task.targets:
        if k in models:
            errors.append(f"targets.{k}: also a scene model name")
    checks = [*task.conditions(), *((f"sanity.{k}", c) for k, c in sanity.items())]
    for loc, c in checks:
        for p, text in c.entities().items():
            kinds = METRICS[c.metric].params[p].kinds
            errors += [f"{loc}.{p}: {m}" for m in check_entity(text, kinds, models, task.targets, robot)]
    return errors


def arm_errors(arm, robot):
    """Errors when the arm start (a robot.yaml pose name or joint positions) does not fit this robot."""
    if isinstance(arm, str) and arm not in robot.poses:
        return [f"robot_start.arm {arm!r} is not a pose of robot {robot.name} (poses: {', '.join(robot.poses)})"]
    if isinstance(arm, list) and len(arm) != robot.arm_dof:
        return [f"robot_start.arm has {len(arm)} joints, robot {robot.name} {robot.arm_dof}"]
    return []


def check_robot_start(scene, robot):
    """Errors when the scene's robot_start does not fit this robot."""
    return [f"scene {scene.name}: {e}" for e in arm_errors(scene.robot_start.arm, robot)]


def check_robot_refs(task, scene, models, sanity, robot):
    """Errors when the task + scene run with this robot: robot::<alias> entities, the robot_start arm pose."""
    errors = check_task(task, models, sanity, robot)    # the task passed without a robot: only robot errors left
    return errors + check_robot_start(scene, robot)


# ------------------------------------------------------------------ loader
REPO_SPECS = Path(__file__).resolve().parents[2] / "specs"    # this repo's specs/ (scene instances resolve against it)
KINDS = {"tasks": Task, "robots": Robot, "scenes": Scene}
FILES = {"tasks": "task.yaml", "robots": "robot.yaml", "scenes": "scene.yaml"}


@dataclass
class Resolved:
    """A task ready to run: the task, its scene (+ model names), the merged sanity checks, the robot (if given)."""
    task: Task
    scene: Scene
    models: dict[str, ModelInfo]
    sanity: dict[str, SanityCheck]
    robot: Robot | None = None


class _UniqueKeyLoader(yaml.SafeLoader):
    """yaml.safe_load, but a duplicate key is an error (safe_load keeps the last value without a word).
    Merge keys (<<: *anchor) stay legal, and an explicit key may override a merged one."""

    def construct_mapping(self, node, deep=False):
        self._check_unique(node)
        return super().construct_mapping(node, deep)

    def _check_unique(self, node):
        seen = set()
        for k, v in node.value:
            if k.tag == "tag:yaml.org,2002:merge":       # its mapping(s): their own keys must be unique too
                for m in v.value if isinstance(v, yaml.SequenceNode) else [v]:
                    if isinstance(m, yaml.MappingNode):
                        self._check_unique(m)
                continue
            key = self.construct_object(k, deep=True)
            try:
                if (type(key), key) in seen:              # 1 and true are different YAML keys
                    raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", k.start_mark)
                seen.add((type(key), key))
            except TypeError:        # an unhashable key: SafeLoader's own error below
                pass


def read_yaml(path):
    try:
        data = yaml.load(Path(path).read_text(), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as e:
        raise SpecError(f"{path}: {e}") from None
    if not isinstance(data, dict):
        raise SpecError(f"{path}: not a YAML mapping")
    return data


def find_root(path):
    """The specs/ dir that holds path: the nearest parent named specs that holds tasks/sanity.yaml (a folder named
    specs without it is not a specs root, e.g. one in a run directory)."""
    p = Path(path).resolve()
    for d in [p, *p.parents]:
        if d.name == "specs" and (d / "tasks" / SANITY_FILE).is_file():
            return d
    raise SpecError(f"{path}: not under a specs/ directory (one that holds tasks/{SANITY_FILE})")


def _fail(path, what, lines):
    raise SpecError("\n".join([f"{path}: {what} invalid", *(f"  {x}" for x in lines)]))


def parse(model, path):
    try:
        return model.model_validate(read_yaml(path))
    except ValidationError as e:
        lines = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            lines.append(f"{loc}: {err['msg'].removeprefix('Value error, ')}" if loc else
                         err["msg"].removeprefix("Value error, "))
        _fail(path, f"{model.__name__} spec", lines)


def spec_path(root, kind, name):
    p = Path(root) / kind / name / FILES[kind]
    if not p.is_file():
        raise SpecError(f"{kind[:-1]} {name!r}: no {p}")
    return p


def load_sanity(root):
    """tasks/sanity.yaml: {key: SanityCheck}; any key that is not a check is rejected."""
    path = Path(root) / "tasks" / SANITY_FILE
    data, out, lines = read_yaml(path), {}, []
    for k, v in data.items():
        if not isinstance(v, dict) or "metric" not in v:
            lines.append(f"{k}: not a check ({{metric: ..., <op>: value}}); {SANITY_FILE} holds only checks")
            continue
        try:
            out[k] = SanityCheck.model_validate(v)
        except ValidationError as e:
            lines += [f"{k}: {err['msg'].removeprefix('Value error, ')}" for err in e.errors()]
    if lines:
        _fail(path, "sanity base", lines)
    return out


def _load(path, root, kind, assets, instance=False, scene=None):
    """scene (a task): run it on this Scene (e.g. a scene instance) instead of its own scene file."""
    spec = parse(KINDS[kind], path)
    if not instance and spec.name != Path(path).parent.name:
        raise SpecError(f"{path}: name {spec.name!r} != its folder {Path(path).parent.name!r}")
    if kind == "robots":
        errs = check_robot_files(spec, Path(path).parent)
        if errs:
            _fail(path, "robot", errs)
        return spec, None
    if kind == "scenes":
        scene, where = spec, path
    elif scene is not None:
        if scene.name != spec.scene:
            raise SpecError(f"{path}: task {spec.name} is on scene {spec.scene}, not {scene.name}")
        where = f"scene {scene.name}"           # a given Scene has no file here
    else:
        where = spec_path(root, "scenes", spec.scene)
        scene = parse(Scene, where)
    models, errs = scene_models(scene, root.parent, assets)
    if errs:
        _fail(where, "scene", errs)
    if kind == "scenes":
        return spec, None
    sanity = merge_sanity(load_sanity(root), spec.sanity)
    errs = check_task(spec, models, sanity)
    if errs:
        _fail(path, "task", errs)
    return spec, Resolved(task=spec, scene=scene, models=models, sanity=sanity)


def load(path, assets=True, root=None):
    """Load and validate one spec file -> Scene, Robot, Task, or {key: SanityCheck} for tasks/sanity.yaml.
    A scene's model:// names and joints, and a task's entities, are checked against the model SDFs
    (assets=False: names only, no asset files read). Raises SpecError with every problem found.
    A file named scene.yaml outside any specs/ directory (find_root) is a scene instance (e.g. one a randomizer wrote
    for a seed): a scene like any other, named like the scene it was made from (not after its folder), its `assets`
    relative to the repo of `root` (a specs/ directory; default the repo's own, REPO_SPECS)."""
    path = Path(path)
    try:
        root = find_root(path)
    except SpecError:
        if path.name != FILES["scenes"]:
            raise SpecError(f"{path}: not under a specs/ directory (one that holds tasks/{SANITY_FILE}), and not a "
                            f"scene instance (a scene instance file is named {FILES['scenes']})") from None
        return _load(path, Path(root or REPO_SPECS), "scenes", assets, instance=True)[0]
    if path.resolve() == (root / "tasks" / SANITY_FILE).resolve():
        return load_sanity(root)
    rel = path.resolve().relative_to(root).parts
    if len(rel) != 3 or rel[0] not in KINDS or rel[2] != FILES[rel[0]]:
        raise SpecError(f"{path}: not specs/{{scenes,robots,tasks}}/<name>/<kind>.yaml or specs/tasks/{SANITY_FILE}")
    return _load(path, root, rel[0], assets)[0]


def resolve(root, task, robot=None, assets=True, scene=None):
    """Task `task` (name) with its scene, merged sanity checks and, if `robot` (name) is given, the robot, with
    every robot::<alias> and the robot start checked against it -> Resolved. scene: a loaded Scene to run the task
    on instead of its own scene file (a scene instance: it must carry the task's scene name)."""
    root = Path(root)
    _, r = _load(spec_path(root, "tasks", task), root, "tasks", assets, scene=scene)
    if robot is not None:
        r.robot = _load(spec_path(root, "robots", robot), root, "robots", assets)[0]
        errs = check_robot_refs(r.task, r.scene, r.models, r.sanity, r.robot)
        if errs:
            _fail(spec_path(root, "tasks", task), f"task with robot {robot}", errs)
    return r


def scene_file(scene, root=REPO_SPECS):
    """A scene name -> root/scenes/<name>/scene.yaml; a path ending in .yaml or .yml -> that file (load checks it)."""
    p = Path(scene)
    return p if p.suffix in (".yaml", ".yml") else Path(root) / "scenes" / str(scene) / FILES["scenes"]


def check_run(scene, robot=None, task=None, assets=True, root=REPO_SPECS):
    """What a simulator run of `scene` (a name or a file, scene_file) with `robot` and `task` (names) would refuse
    -> error lines naming the scene file, [] if none: the scene spec, what the simulator supports, robot_start against
    the robot, the task on this scene with the robot."""
    root = Path(root)
    path = scene_file(scene, root)
    try:
        sc = load(path, assets=assets, root=root)
    except SpecError as e:
        return str(e).splitlines()      # names the file
    if not isinstance(sc, Scene):
        return [f"{path}: not a scene file"]
    try:
        errs = check_sim_support(sc)
        if robot is not None:
            errs += check_robot_start(sc, load(spec_path(root, "robots", robot), assets=assets))
        if task is not None and not errs:
            resolve(root, task, robot=robot, assets=assets, scene=sc)
    except SpecError as e:
        errs = str(e).splitlines()
    what = f"{path}: scene" + (f" with robot {robot}" if robot else "") + (f" and task {task}" if task else "")
    return [f"{what} invalid", *(f"  {x}" for x in errs)] if errs else []


def all_specs(root):
    """Every spec file under specs/: scenes, robots, tasks/sanity.yaml, tasks."""
    root = Path(root)
    out = [p for k in ("scenes", "robots") for p in sorted((root / k).glob(f"*/{FILES[k]}"))]
    if (root / "tasks" / SANITY_FILE).is_file():
        out.append(root / "tasks" / SANITY_FILE)
    return out + sorted((root / "tasks").glob("*/task.yaml"))
