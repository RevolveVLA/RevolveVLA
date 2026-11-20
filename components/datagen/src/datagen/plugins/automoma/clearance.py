"""validators: [{plugin: automoma}]: the pre-plan chassis clearance (layout_check.py) of a config's base start."""
from __future__ import annotations

import sys
from pathlib import Path

from datagen.core.interfaces import Config, Result, Validator
from datagen.core.registry import register

from .layout import Realiser

HERE = Path(__file__).resolve().parent


@register("validator", "automoma")
class ChassisClearance(Validator):
    """Rejects a config whose base start puts the chassis (grown by the per-attempt jitter) closer than the task's
    replay.spawn_clearance_min_m to a static scene box. Loads the scene on the first call (needs assets/)."""

    def __init__(self, spec, params):
        self.spec, self.realiser, self._reject = spec, Realiser(spec), None

    def reject(self):
        if self._reject is None:
            for p in (HERE, HERE.parent.parent):         # layout_check's flat imports (frames, start_design; specs_bridge)
                if str(p) not in sys.path:
                    sys.path.append(str(p))
            import layout_check
            self._reject = layout_check.clearance_reject(self.spec.task, self.spec.robot)
        return self._reject

    def validate(self, config: Config) -> Result:
        reason = self.reject()(self.realiser.layout(config))
        return Result.fail(reason) if reason else Result()
