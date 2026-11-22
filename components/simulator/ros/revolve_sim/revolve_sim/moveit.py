"""A robot's MoveIt 2 parameters for moveit_py (MoveItPy(config_dict=...)) or a move_group node, from
specs/robots/<r>/moveit/: <r>.srdf.xacro, moveit_controllers.yaml and moveit_py.yaml, plus kinematics.yaml,
joint_limits.yaml and pilz_cartesian_limits.yaml when the robot has them (the SO-101 robots: links to
specs/robots/common/so101_moveit/; the OMPL and Pilz pipeline configs are then moveit_configs_utils' defaults). Without
them, these and the pipeline configs come unchanged from the upstream MoveIt config package
(moveit_resources_panda_moveit_config: the Panda's)."""
from revolve_sim.robot import describe, robot_dir

UPSTREAM = 'moveit_resources_panda_moveit_config'
OWN = ('kinematics.yaml', 'joint_limits.yaml', 'pilz_cartesian_limits.yaml')


def moveit_params(repo, robot_name, urdf=None, use_sim_time=True):
    """-> the parameter dict (robot_description from the robot's xacro unless `urdf` is given)."""
    from moveit_configs_utils import MoveItConfigsBuilder
    d = robot_dir(repo, robot_name) / 'moveit'
    own = all((d / f).is_file() for f in OWN)
    f = {k: str(d / k) if own else None for k in OWN}
    # a package without a config/ folder: the pipelines fall back to moveit_configs_utils/default_configs
    cfg = (MoveItConfigsBuilder(robot_name, package_name='moveit_configs_utils' if own else UPSTREAM)
           .robot_description_semantic(file_path=str(d / f'{robot_name}.srdf.xacro'))
           .trajectory_execution(file_path=str(d / 'moveit_controllers.yaml'))
           .moveit_cpp(file_path=str(d / 'moveit_py.yaml'))
           .planning_pipelines(pipelines=['ompl', 'pilz_industrial_motion_planner'])
           .robot_description_kinematics(file_path=f['kinematics.yaml'])
           .joint_limits(file_path=f['joint_limits.yaml'])
           .pilz_cartesian_limits(file_path=f['pilz_cartesian_limits.yaml'])
           .to_moveit_configs())
    cfg.robot_description = {'robot_description': urdf if urdf is not None else describe(repo, robot_name)[1]}
    params = cfg.to_dict()
    params['use_sim_time'] = use_sim_time
    return params
