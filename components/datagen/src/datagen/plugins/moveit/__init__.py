"""The `moveit` generator plugin (kind generator): the scripted MoveIt 2 expert for open_fridge_door (mobile_panda) and
open_drawer (mobile_so101_xl), ported from main's T4 reference clients. Pure helpers: geometry.py, motion.py; the rest
(robot_io, planner, setup, experts) needs rclpy / moveit_py and only runs inside the client container."""
from .generator import MoveItGenerator  # noqa: F401  (registers the plugin)
