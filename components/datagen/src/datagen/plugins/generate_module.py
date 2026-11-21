"""generate.py (docker SDK, numpy) as a module, imported when first needed: the plugins that run its jobs (automoma's
plan + check, rosetta's camera replay) share it. Sets $REVOLVE_ROBOT as the CLI does and the datagen variant when given."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .. import hostmounts

COMPONENT = Path(__file__).resolve().parents[3]          # components/datagen: generate.py, launch.py


def use_robot(robot: str):
    """$REVOLVE_ROBOT for the task-level spec loads of the jobs (a step run alone has nothing else that sets it)."""
    from revolve_common import specs
    os.environ[specs.ROBOT_ENV] = robot


def gui_kw(spec) -> dict:
    """generate.py's `gui=True` for a spec with `run.headless: false` (nothing otherwise: a headless run is unchanged)."""
    return {} if spec.run.headless else {"gui": True}


def load(robot: str, task: str, variant: str | None = None):
    if str(COMPONENT) not in sys.path:
        sys.path.append(str(COMPONENT))
    use_robot(robot)
    import generate
    if hostmounts.MOUNTS:                        # an `out` outside the repo: the launchers must know it before a job starts
        import launch  # noqa: F401  (generate.py imports it lazily; the mounts patch the loaded launchers)
        hostmounts.apply()
    if variant:
        generate.set_variant(variant, task)
    return generate
