"""Spec schemas (pydantic v2), the metric registry (names + parameters) and the loader.

specs/ layout (every name = its folder name):
  scenes/<s>/scene.yaml     one fixed scene: the models (model:// names in the asset search path) and the robot start
  robots/<r>/robot.yaml     the robot: description (xacro), controllers, named poses, frame and link-group aliases
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
Unknown keys are errors everywhere.
"""
from __future__ import annotations

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
    model_config = ConfigDict(extra="forbid")


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

    @field_validator("models")
    @classmethod
    def _names(cls, v):
        for k in v:
            _name(k)
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


# ------------------------------------------------------------------ robot
class Robot(Strict):
    name: str
    description: str                 # the robot's xacro (URDF), relative to the robot's folder
    controllers: str                 # ros2_control controllers yaml, relative to the robot's folder
    poses: dict[str, list[float]] = Field(min_length=1)   # named arm poses [rad], arm joint order
    frames: dict[str, str] = {}      # alias -> link (robot::<alias>, e.g. tcp, hand, base)
    groups: dict[str, list[str]] = {}   # alias -> links (e.g. fingers)

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


# ------------------------------------------------------------------ task
class Phase(Strict):
    """Ordered: all `done` conditions true -> the next phase; a `while` condition false, or timeout_s passed in this
    phase -> the task fails in this phase."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
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


def check_robot_refs(task, scene, models, sanity, robot):
    """Errors when the task + scene run with this robot: robot::<alias> entities, the robot_start arm pose."""
    errors = check_task(task, models, sanity, robot)    # the task passed without a robot: only robot errors left
    arm = scene.robot_start.arm
    if isinstance(arm, str) and arm not in robot.poses:
        errors.append(f"scene {scene.name}: robot_start.arm {arm!r} is not a pose of robot {robot.name} "
                      f"(poses: {', '.join(robot.poses)})")
    if isinstance(arm, list) and len(arm) != robot.arm_dof:
        errors.append(f"scene {scene.name}: robot_start.arm has {len(arm)} joints, robot {robot.name} {robot.arm_dof}")
    return errors


# ------------------------------------------------------------------ loader
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


def read_yaml(path):
    try:
        data = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as e:
        raise SpecError(f"{path}: {e}") from None
    if not isinstance(data, dict):
        raise SpecError(f"{path}: not a YAML mapping")
    return data


def find_root(path):
    """The specs/ dir that holds path."""
    p = Path(path).resolve()
    for d in [p, *p.parents]:
        if d.is_dir() and d.name == "specs":
            return d
    raise SpecError(f"{path}: not under a specs/ directory")


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


def _load(path, root, kind, assets):
    spec = parse(KINDS[kind], path)
    if spec.name != Path(path).parent.name:
        raise SpecError(f"{path}: name {spec.name!r} != its folder {Path(path).parent.name!r}")
    if kind == "robots":
        errs = check_robot_files(spec, Path(path).parent)
        if errs:
            _fail(path, "robot", errs)
        return spec, None
    scene_file = path if kind == "scenes" else spec_path(root, "scenes", spec.scene)
    scene = spec if kind == "scenes" else parse(Scene, scene_file)
    models, errs = scene_models(scene, root.parent, assets)
    if errs:
        _fail(scene_file, "scene", errs)
    if kind == "scenes":
        return spec, None
    sanity = merge_sanity(load_sanity(root), spec.sanity)
    errs = check_task(spec, models, sanity)
    if errs:
        _fail(path, "task", errs)
    return spec, Resolved(task=spec, scene=scene, models=models, sanity=sanity)


def load(path, assets=True):
    """Load and validate one spec file -> Scene, Robot, Task, or {key: SanityCheck} for tasks/sanity.yaml.
    A scene's model:// names and joints, and a task's entities, are checked against the model SDFs
    (assets=False: names only, no asset files read). Raises SpecError with every problem found."""
    path = Path(path)
    root = find_root(path)
    if path.resolve() == (root / "tasks" / SANITY_FILE).resolve():
        return load_sanity(root)
    rel = path.resolve().relative_to(root).parts
    if len(rel) != 3 or rel[0] not in KINDS or rel[2] != FILES[rel[0]]:
        raise SpecError(f"{path}: not specs/{{scenes,robots,tasks}}/<name>/<kind>.yaml or specs/tasks/{SANITY_FILE}")
    return _load(path, root, rel[0], assets)[0]


def resolve(root, task, robot=None, assets=True):
    """Task `task` (name) with its scene, merged sanity checks and, if `robot` (name) is given, the robot, with
    every robot::<alias> and the robot start checked against it -> Resolved."""
    root = Path(root)
    _, r = _load(spec_path(root, "tasks", task), root, "tasks", assets)
    if robot is not None:
        r.robot = _load(spec_path(root, "robots", robot), root, "robots", assets)[0]
        errs = check_robot_refs(r.task, r.scene, r.models, r.sanity, r.robot)
        if errs:
            _fail(spec_path(root, "tasks", task), f"task with robot {robot}", errs)
    return r


def all_specs(root):
    """Every spec file under specs/: scenes, robots, tasks/sanity.yaml, tasks."""
    root = Path(root)
    out = [p for k in ("scenes", "robots") for p in sorted((root / k).glob(f"*/{FILES[k]}"))]
    if (root / "tasks" / SANITY_FILE).is_file():
        out.append(root / "tasks" / SANITY_FILE)
    return out + sorted((root / "tasks").glob("*/task.yaml"))
