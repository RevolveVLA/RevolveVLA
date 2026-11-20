"""AutoMoMa + cuRobo generator plugin: `generator: {plugin: automoma, attempts: 50, checks: 3}` (adapter.py) and the
`validators: [{plugin: automoma}]` chassis clearance pre-filter (clearance.py).

The package itself imports neither torch, cuRobo, AutoMoMa nor docker. The rest of this directory (plan.py, start_design.py,
gen_assets.py, frames.py, fk_check.py, layout_check.py) are the planner's scripts and helpers: they run inside the
datagen image (or the host tests) as flat modules with this directory on PYTHONPATH, as the robots' planner.py
import them (`import gen_assets`)."""
from . import adapter, clearance  # noqa: F401  (register the plugins)
