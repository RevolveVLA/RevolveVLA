"""generate.py (docker SDK, numpy) as a module, imported when first needed: the plugins that run its jobs (automoma's
plan + check, rosetta's camera replay) share it. Sets $REVOLVE_ROBOT as the CLI does and the datagen variant when given."""
from __future__ import annotations

import os
import sys
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[3]          # components/datagen: generate.py, launch.py


def load(robot: str, task: str, variant: str | None = None):
    if str(COMPONENT) not in sys.path:
        sys.path.append(str(COMPONENT))
    from revolve_common import specs
    os.environ[specs.ROBOT_ENV] = robot
    import generate
    if variant:
        generate.set_variant(variant, task)
    return generate
