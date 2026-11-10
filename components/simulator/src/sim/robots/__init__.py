"""The robots: one package per robot implementation (robots/<impl>/: robot.yaml `implementation`, default the
spec name of specs/robots/<robot>/robot.yaml; several specs may share one), each with the same files and API
(robots/base.py has the interface and the shared parts):

  robot.py     make(spec) -> the Robot (kinematics, IK, gravity, arm and gripper laws)
  model.py     main(robot, description, out_dir) -> the gz model (model.sdf) from robot.yaml `description`
  scripted.py  smoke_plan, SCENARIOS (skill tests), EPISODES (scripted episodes) for the kitchen
  planner.py   chain, self_collision, BOX_CENTRE_OBJ: the cuRobo planner robot (datagen's gen_assets; cuRobo only)

load(name) builds a robot from its spec (cached); current() is the robot named by $REVOLVE_ROBOT, which the
CLI sets for every job and gz run from the experiment's `robot:`. There is no default robot.
"""
import importlib
import os

_LOADED = {}


def load(name):
    if name not in _LOADED:
        import spec as sim_spec
        from revolve_common import specs
        rs = specs.load(sim_spec.SPECS / "robots" / name / "robot.yaml")
        _LOADED[name] = importlib.import_module(f"robots.{rs.implementation}.robot").make(rs)
    return _LOADED[name]


def current_name():
    name = os.environ.get("REVOLVE_ROBOT")
    if not name:
        raise RuntimeError("REVOLVE_ROBOT is not set: name the robot (the experiment's `robot:`; the revolve CLI "
                           "and the launchers pass it to every job)")
    return name


def current():
    return load(current_name())
