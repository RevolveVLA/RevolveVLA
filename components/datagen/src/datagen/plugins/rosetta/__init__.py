"""Rosetta recorder + converter plugins: `recorder: {plugin: rosetta}`, `converter: {plugin: rosetta, format: lerobot}`.

Rosetta is the ROS 2 <-> LeRobot bridge (ThirdParty/rosetta). Call chain of an episode:
  recorder   generate.record_attempt -> simulator `replay --cameras <hz>` (revolve-simulator image, GPU) -> sim/bag.py writes the
             episode's MCAP bag (topics + frame grid of the robot's Rosetta contract, task prompt as lerobot.operator_prompt)
  converter  components/dataset/launch.py convert -> revolve-dataset image (Rosetta's colcon install sourced) -> dataset/entry.py
             convert -> dataset/port.py -> rosetta_ros.offline.port (rosetta_port) -> LeRobot v3.0 dataset
The package imports neither docker nor numpy: the jobs are loaded when a plugin first runs."""
from . import converter, recorder  # noqa: F401  (register the plugins)
