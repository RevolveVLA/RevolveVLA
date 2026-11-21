"""MoveIt smoke, run inside a simulator container started by `revolve sim` (see conftest.py): prints one RESULT JSON
line. MoveIt 2 (moveit_py, the robot's specs/robots/<r>/moveit/ config) plans the arm from the scene start to each
TCP pose target of robot_cases.MOVEIT (OMPL to a pose goal: the kinematics plugin's IK), executes it on the arm's JTC,
and the TCP pose reached (tf, 1 s after the execution) is compared with the target; then a Pilz LIN back along the
approach from the last target. The targets are FK poses of joint configurations (reachable by a 5-DoF arm).

  python3 check_moveit.py --scene S --robot R [--repo /repo]
"""
import argparse
import json
import math
import os
import sys
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, PlanRequestParameters
from rclpy.executors import MultiThreadedExecutor

from check_control import Probe
from check_open_door import pose_msg
from robot_cases import CONTROL, MOVEIT
from revolve_sim.moveit import moveit_params
from revolve_sim.robot import describe
from revolve_sim.world import make_world

LIN_SCALE = 0.1    # Pilz LIN velocity / acceleration scaling (moveit_py.yaml pilz_lin: 0.2 exceeds a joint's limit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--robot', required=True)
    ap.add_argument('--repo', default='/repo')
    a = ap.parse_args()
    _, scene, _, _ = make_world(a.scene, a.repo)
    robot, urdf, _, contact_links = describe(a.repo, a.robot)
    case = MOVEIT[robot.name]
    group, tcp, base = case['group'], robot.frames['tcp'], robot.frames['base']

    rclpy.init()
    n = Probe(robot.name, scene.name, contact_links)
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    mp = MoveItPy(node_name='check_moveit', config_dict=moveit_params(a.repo, robot.name, urdf))
    arm = mp.get_planning_component(group)
    model = mp.get_robot_model()
    out = {'targets': {}}

    def reached():
        t = n.tf.lookup_transform(base, tcp, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=5)).transform
        T = np.eye(4)
        T[:3, 3] = (t.translation.x, t.translation.y, t.translation.z)
        x, y, z, w = t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w
        T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
        return T

    def go(name, T_goal, params, scale=None):
        arm.set_start_state_to_current_state()
        ps = PoseStamped()
        ps.header.frame_id, ps.pose = base, pose_msg(T_goal)
        arm.set_goal_state(pose_stamped_msg=ps, pose_link=tcp)
        t0 = n.now()
        prp = PlanRequestParameters(mp, params)
        if scale is not None:
            prp.max_velocity_scaling_factor = prp.max_acceleration_scaling_factor = scale
        plan = arm.plan(single_plan_parameters=prp)
        r = {'planned': bool(plan), 'target': [round(float(v), 4) for v in T_goal[:3, 3]]}
        if plan:
            r['executed'] = bool(mp.execute(plan.trajectory, controllers=[]))
            n.wait(1.0)
            T = reached()
            c = (np.trace(T_goal[:3, :3].T @ T[:3, :3]) - 1) / 2
            r.update(pos_err_mm=round(1000 * float(np.linalg.norm(T[:3, 3] - T_goal[:3, 3])), 2),
                     rot_err_deg=round(math.degrees(math.acos(max(-1.0, min(1.0, c)))), 3),
                     reached=[round(float(v), 4) for v in T[:3, 3]])
            with n.lock:
                r['q'] = [round(n.joints[j], 4) for j in CONTROL[robot.name]['arm_joints']]
        r['t_s'] = round(n.now() - t0, 2)
        out['targets'][name] = r
        print(json.dumps({name: r}), file=sys.stderr, flush=True)
        return r

    n.wait(2.0)
    try:
        for name, q in case['targets'].items():
            s = RobotState(model)
            s.set_to_default_values()
            s.set_joint_group_positions(group, np.array(q))
            s.update()
            T_goal = s.get_global_link_transform(tcp)
            go(name, T_goal, 'ompl_rrtc')
        T_lin = T_goal.copy()
        T_lin[:3, 3] -= case['lin_m'] * T_goal[:3, 2]
        go(f'{name}_lin_back', T_lin, 'pilz_lin', LIN_SCALE)
    except Exception as e:   # report what was reached
        out['error'] = repr(e)
    print('RESULT ' + json.dumps(out), flush=True)
    sys.stderr.flush()
    os._exit(0)     # moveit_py segfaults in the interpreter's teardown (rc 139): exit after the RESULT line


if __name__ == '__main__':
    main()
