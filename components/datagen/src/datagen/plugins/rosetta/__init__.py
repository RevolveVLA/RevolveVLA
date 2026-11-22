"""Rosetta converter plugin: `converter: {plugin: rosetta, format: lerobot}` (a step the pipeline skips for now: no Rosetta
contract for the new simulator's bags, DATAGEN-v2.md 7).

Rosetta is the ROS 2 <-> LeRobot bridge (ThirdParty/rosetta): components/dataset/launch.py convert -> revolve-dataset image
(Rosetta's colcon install sourced) -> dataset/entry.py convert -> dataset/port.py -> rosetta_ros.offline.port -> LeRobot v3.0.
The package imports neither docker nor numpy: the job is loaded when the plugin first runs."""
from . import converter  # noqa: F401  (register the plugin)
