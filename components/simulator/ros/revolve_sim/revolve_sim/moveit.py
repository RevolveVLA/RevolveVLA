"""A robot's MoveIt 2 parameters for moveit_py (MoveItPy(config_dict=...)) or a move_group node, from
specs/robots/<r>/moveit/: <r>.srdf.xacro, moveit_controllers.yaml and moveit_py.yaml. Kinematics, joint limits, the
OMPL / Pilz pipeline configs and the Pilz Cartesian limits come unchanged from the upstream MoveIt config package
(moveit_resources_panda_moveit_config: the Panda's)."""
from revolve_sim.robot import describe, robot_dir

UPSTREAM = 'moveit_resources_panda_moveit_config'


def moveit_params(repo, robot_name, urdf=None, use_sim_time=True):
    """-> the parameter dict (robot_description from the robot's xacro unless `urdf` is given)."""
    from moveit_configs_utils import MoveItConfigsBuilder
    d = robot_dir(repo, robot_name) / 'moveit'
    cfg = (MoveItConfigsBuilder(robot_name, package_name=UPSTREAM)
           .robot_description_semantic(file_path=str(d / f'{robot_name}.srdf.xacro'))
           .trajectory_execution(file_path=str(d / 'moveit_controllers.yaml'))
           .moveit_cpp(file_path=str(d / 'moveit_py.yaml'))
           .planning_pipelines(pipelines=['ompl', 'pilz_industrial_motion_planner'])
           .robot_description_kinematics()
           .joint_limits()
           .pilz_cartesian_limits()
           .to_moveit_configs())
    cfg.robot_description = {'robot_description': urdf if urdf is not None else describe(repo, robot_name)[1]}
    params = cfg.to_dict()
    params['use_sim_time'] = use_sim_time
    return params
