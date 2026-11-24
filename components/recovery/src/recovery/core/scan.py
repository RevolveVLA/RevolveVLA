"""The scan of an eval run for a spec: the task (read from specs/), the robot's arm, the triggers built against the task, then
`failures.scan` with a bag reader. Shared by the scanners (host: in process; sim image: `recovery.scan_client`)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from datagen.core.spec import SpecError
from revolve_common import specs
from revolve_common.supervision import Supervisor, build

from . import failures
from .registry import REGISTRY
from .robot import Arm, arm_of
from .spec import RecoverySpec, check_task


def load_task(spec: RecoverySpec) -> specs.Resolved:
    """The spec's task with its scene and sanity checks (names only: no asset files read) and robot, from specs/."""
    try:
        return specs.resolve(specs.REPO_SPECS, spec.task, spec.robot, assets=False)
    except specs.SpecError as e:
        raise SpecError(str(e)) from None


def supervisor_of(spec: RecoverySpec, task: specs.Resolved, arm: Arm) -> Supervisor:
    """The spec's triggers over the task (SpecError naming the yaml path when they do not fit)."""
    check_task(spec, [p.name for p in task.task.phases])
    try:
        return build(spec.triggers, task, arm_joints=arm.joints, path="triggers")
    except specs.SpecError as e:
        raise SpecError(str(e)) from None


def checked(spec: RecoverySpec) -> RecoverySpec:
    """The spec once the task, the robot and the registry have checked it (quota phases, triggers, the live recorder)."""
    supervisor_of(spec, load_task(spec), arm_of(spec.robot, specs.REPO_SPECS))
    if spec.live:
        REGISTRY.create("live", spec.live.plugin, spec, spec.live.params).validate(spec)
    return spec


def scan_run(spec: RecoverySpec, run: Path, read_messages: Callable, only=None) -> failures.Scan:
    """`only`: the (set, rollout key)s of the run to scan (default all)."""
    task, arm = load_task(spec), arm_of(spec.robot, specs.REPO_SPECS)
    return failures.scan(spec, supervisor_of(spec, task, arm), arm.joints, run, read_messages, only)
