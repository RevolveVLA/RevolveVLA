"""The `automoma` generator plugin (kind generator): AutoMoMa's cuRobo whole-body plan of a scene instance (plan.py, GPU, datagen
image), streamed to the sim as joint trajectory points + base twists (execute.py, client container). Pure helpers: problem.py,
model.py, assets.py (URDF text), series.py, track.py, frames.py; the planner modules import cuRobo / torch lazily and only run
in the datagen image."""
from .generator import AutoMoma  # noqa: F401  (registers the plugin)
