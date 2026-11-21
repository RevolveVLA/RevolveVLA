"""A config's layout: what the spec's randomisers realise, as the planner takes it (start_design.parse_layout)."""
from __future__ import annotations

from datagen.core import registry
from datagen.core.interfaces import Config
from datagen.core.spec import DatagenSpec

PLAN_KEYS = ("start_pose", "grasp_id", "reach", "drive")        # start_design.LAYOUT_KEYS


class Realiser:
    """The spec's randomisers applied to a config: `realise` = their merged outputs, `layout` = the planner's part."""

    def __init__(self, spec: DatagenSpec):
        self.spec = spec
        self.randomisers = {n: registry.create("randomiser", n) for n in spec.randomisers}
        self._context = None

    @property
    def context(self) -> dict:
        """The generator's data for Randomiser.realise: the task's start-box centre (task.yaml starts, box method)."""
        if self._context is None:
            from datagen.plugins.randomisers.robot_base import box_design
            self._context = ({"box_centre": tuple(box_design(self.spec).centre)} if "robot_base" in self.randomisers
                             else {})
        return self._context

    def realise(self, config: Config) -> dict:
        out = {}
        for name, randomiser in self.randomisers.items():
            out.update(randomiser.realise(config.of(name), self.context))
        return out

    def layout(self, config: Config) -> dict:
        """{"start_pose": [x, y, yaw], "grasp_id": g} (object frame) of a box start, {"reach": {...}, "drive": {...}} of a
        reach / drive start: the keys that are varied; {} when none is."""
        r = self.realise(config)
        return {k: list(r[k]) if k == "start_pose" else r[k] for k in PLAN_KEYS if k in r}
