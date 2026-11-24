"""The datagen generator of a recovery spec. A recovery episode starts mid-task, so a generator that can take the task over
from the robot's current state (`supports_resume`) is created in that mode unless the spec says `resume: false`; one that
cannot (automoma) plans from the restart, which is its scene instance's `robot_start`."""
from __future__ import annotations

from datagen.core import registry
from datagen.core.interfaces import Generator

from .spec import RecoverySpec


def create(spec: RecoverySpec) -> Generator:
    g = spec.phases.generator
    cls = registry.get("generator", g.plugin)
    params = {"resume": True, **g.params} if cls.supports_resume else g.params
    return cls(spec.datagen_spec(), params)
