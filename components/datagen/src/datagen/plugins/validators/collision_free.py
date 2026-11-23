"""validators: [{plugin: collision_free, clearance_m: 0.05, footprint: {x: [-0.25, 0.36], y: [-0.25, 0.25]}}]

Rejects a config whose scene instance puts the robot in collision (reason `collision: ...`): the robot's footprint
(scene/footprint.py: chassis plus folded-arm overhang, per robot; `footprint` overrides x / y / z) at the instance's
robot_start must stay `clearance_m` (default 0.05) away from every collision of the scene's models (SDF boxes, meshes by
their AABB; scene/sdf_boxes.py), at the heights it occupies. Loads the scene on the first call (needs assets/)."""
from __future__ import annotations

import threading

from revolve_common import specs

from datagen.core.interfaces import Config, Result, Validator
from datagen.core.registry import register
from datagen.core.spec import SpecError
from datagen.scene.footprint import Clearance, Footprint, footprint_of
from datagen.scene.instance import Variation
from datagen.scene.sdf_boxes import scene_boxes

CLEARANCE_M = 0.05


@register("validator", "collision_free")
class CollisionFree(Validator):
    def __init__(self, spec, params):
        self.spec, self.params, self.variation = spec, params, Variation(spec)
        self.clearance_m = params.get("clearance_m", CLEARANCE_M)
        if "robot_start" not in self.variation.randomisers:
            raise SpecError("validators.collision_free: the spec does not vary robot_start (randomisers.robot_start)")
        self._clearance, self._lock = None, threading.Lock()

    def footprint(self) -> Footprint:
        fp = footprint_of(self.spec.robot)
        given = {k: tuple(v) for k, v in self.params.get("footprint", {}).items()}
        if bad := sorted(set(given) - {"x", "y", "z"}):
            raise SpecError(f"validators.collision_free.footprint: unknown keys {bad} (x, y, z)")
        return Footprint(**{**{"x": fp.x, "y": fp.y, "z": fp.z}, **given})

    def clearance(self) -> Clearance:
        with self._lock:                                 # the workers' first calls load the scene once
            if self._clearance is None:
                scene = specs.Scene.model_validate(self.variation.base())
                boxes = scene_boxes(scene, specs.REPO_SPECS.parent)
                self._clearance = Clearance(boxes, self.footprint(), self.clearance_m)
            return self._clearance

    def validate(self, config: Config) -> Result:
        x, y, yaw = self.variation.realisation(config)["robot_start"]["base"]
        reason = self.clearance().reject(x, y, yaw)
        return Result.fail(reason) if reason else Result()
