"""The base scene of a spec and what its yaml and the robot's yaml say, read through revolve_common.specs (no assets).

The specs directory is `specs.REPO_SPECS` (read at call time, so a test can point it at a fixture tree). Every spec
problem is raised as datagen's SpecError, so the CLI prints it as one line."""
from __future__ import annotations

from pathlib import Path

from revolve_common import specs

from datagen.core.spec import DatagenSpec, SpecError


def _read(model, path):
    try:
        return specs.parse(model, path)
    except specs.SpecError as e:
        raise SpecError(str(e)) from None


def _path(kind: str, name: str) -> Path:
    try:
        return specs.spec_path(specs.REPO_SPECS, kind, name)
    except specs.SpecError as e:
        raise SpecError(str(e)) from None


def scene_name(spec: DatagenSpec) -> str:
    """spec.scene, else the scene of the spec's task."""
    return spec.scene or _read(specs.Task, _path("tasks", spec.task)).scene


def scene_file(spec: DatagenSpec) -> Path:
    return _path("scenes", scene_name(spec))


def read_scene(spec: DatagenSpec) -> dict:
    """The base scene.yaml as a plain mapping (validated), what an instance is a copy of."""
    path = scene_file(spec)
    _read(specs.Scene, path)
    try:
        return specs.read_yaml(path)
    except specs.SpecError as e:
        raise SpecError(str(e)) from None


def arm_poses(spec: DatagenSpec) -> tuple[str, ...]:
    """The named arm poses of the spec's robot (robot.yaml `poses`)."""
    return tuple(_read(specs.Robot, _path("robots", spec.robot)).poses)
