"""The specs (specs/tasks/<t>/task.yaml -> robot + scene, loaded and validated by revolve_common) as the
ported planner reads them: the earlier pipeline's plugin dicts (`load(kind, name)`), built
from the spec fields (the spec's `planner:` blocks flattened in, as they sat in plugin.yaml), and the
generator settings from generator.yaml beside this file (the earlier pipeline's planner config).
Paths are relative to the repo root: $REPO (in the containers /repo).
"""
import os
from functools import lru_cache
from pathlib import Path

import yaml

from revolve_common import specs

ROOT = Path(os.environ.get("REPO", Path(__file__).resolve().parents[4]))
SPECS = Path(os.environ.get("REVOLVE_SPECS", ROOT / "specs"))
HERE = Path(__file__).resolve().parent


def path(p):
    return ROOT / p


def model_dir(dirs, uri):
    """<dir>/<name>/ for model://<name>: the first of dirs (repo-relative) that has it, as
    GZ_SIM_RESOURCE_PATH resolves it."""
    name = uri.replace("model://", "")
    for d in dirs:
        if (path(d) / name / "model.sdf").is_file():
            return path(d) / name
    raise FileNotFoundError(f"{uri} not in {dirs}")


@lru_cache(maxsize=None)
def task_spec(name) -> specs.Resolved:
    return specs.load(SPECS / "tasks" / name / "task.yaml")


@lru_cache(maxsize=None)
def generator():
    return yaml.safe_load((HERE / "generator.yaml").read_text())


def round_blocks(task, robot=None, variant=None):
    """The generator's per-task round blocks (starts, timing, pull) -> {KEY: block} for ROUND_<KEY> env, with
    the task's `robots.<robot>` blocks over them (null drops a block), then those of `variants.<variant>.robots.
    <robot>`; robot: default $REVOLVE_ROBOT, variant: default $REVOLVE_DATAGEN_VARIANT (unset: none). A variant
    the task does not define for the robot is an error."""
    robot = robot or os.environ.get(specs.ROBOT_ENV)
    variant = variant or os.environ.get(specs.VARIANT_ENV) or None
    t = dict(generator().get("tasks", {}).get(task, {}))
    variants = t.pop("variants", {})
    over = [t.pop("robots", {}).get(robot, {})]
    if variant is not None:
        if robot not in variants.get(variant, {}).get("robots", {}):
            raise ValueError(f"datagen variant {variant!r}: generator.yaml tasks.{task}.variants has no "
                             f"{variant}.robots.{robot}")
        over.append(variants[variant]["robots"][robot])
    for o in over:
        for k, v in o.items():
            if v is None:
                t.pop(k, None)
            else:
                t[k] = v
    return t


def load(kind, name):
    """The old plugin dict of kind robot / scene / task / generator."""
    if kind == "generator":
        return {k: v for k, v in generator().items() if k != "tasks"}
    if kind == "task":
        t = task_spec(name).task
        o = t.object
        return dict(kind="task", name=t.name, object=o.name, gz_model=o.gz_model, world_pose=list(o.world_pose),
                    joint=o.joint, handle=o.handle, goal_angle=o.goal, joint_topic=o.joint_topic, prompt=t.prompt,
                    card=t.card, **t.planner)
    root = {"robot": "robots", "scene": "scenes"}[kind]
    s = specs.load(SPECS / root / name / f"{kind}.yaml")
    d = s.model_dump()
    return dict(kind=kind, **{k: v for k, v in d.items() if k != "planner"}, **d.get("planner", {}))
