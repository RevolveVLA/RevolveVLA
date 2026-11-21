"""T4 positive episode of open_drawer, run inside a simulator container started by `revolve sim --task open_drawer
--no-autostart` (see conftest.py): prints one RESULT JSON line. MoveIt 2 (moveit_py, the robot's
specs/robots/<r>/moveit/ config) plans the arm; the base gets open-loop, ramped TwistStamped segments (test code
only). The bar, the slide and the planning-scene boxes come from the scene's model SDFs (joints at 0) and the base
pose from the gz ground truth; helpers from check_open_door.py.

  drive:    rotate to face the bar, then translate to REACH m in front of its centre (the arm folded)
  approach: TCP +z into the drawer (handle -x), TCP x along the bar (handle y): with the 5-DoF arm this pose is
            reachable only in the vertical plane of the shoulder pan axis, so the grasp point is the bar point in
            front of the shoulder (measured after the drive). MoveIt OMPL to the pre-grasp (PRE m back along the
            approach, IK seeded with SEED), Pilz LIN to the bar, more LINs until the TCP is on the bar centre
            within CENTRE_MM in depth and LIFT above it (the fixed jaw pressed up against the bar)
  grasp:    close the moving jaw past the bar (the effort JTC squeezes at its clamp) until the evaluator leaves
            `grasp`
  open:     the base backs off along the slide (one ramped segment, PULL m at V_PULL) with the arm holding; hold
            until the result; then release
The RESULT has the episode result, a log of the steps and a 20 Hz trace from the grasp on: [t, slide [m], TCP in the
handle frame [mm] (tf / ground truth / joint state as they come: approximate while moving), finger contacts].

  python3 check_open_drawer.py --scene S --robot R --task T [--repo /repo]
"""
import argparse
import json
import os
import sys
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit_msgs.msg import CollisionObject
from rclpy.executors import MultiThreadedExecutor
from revolve_common import specs
from shape_msgs.msg import SolidPrimitive

from check_open_door import Door, model_geometry, pose_msg
from robot_cases import CONTROL, MOVEIT
from revolve_sim.moveit import moveit_params
from revolve_sim.robot import describe
from revolve_sim.world import make_world

REACH = 0.80                     # base_footprint in front of the bar [m]: the TCP 0.80 m ahead (robot_cases `bar`)
SEED = [0.0, -0.2718, 1.2384, -0.9666, 0.0]   # the IK branch: TCP 0.75 m ahead at the bar height, horizontal
PRE = 0.10                       # pre-grasp: back along the approach [m]
BAR_MARGIN = 0.05                # the grasp point stays this far inside the bar's ends [m]
V_DRIVE, W_DRIVE = 0.3, 0.5
PULL, V_PULL = 0.20, 0.05        # the base's back-off along the slide [m, m/s] (the task: >= 0.15 m for 1 s)
LIN_SCALE = 0.1                  # Pilz LIN velocity / acceleration scaling (joint acceleration limits, see MOVEIT)
CENTRE_MM, CENTRE_TRIES = 1.0, 4   # TCP on the bar within [mm] (handle x, z) before closing
LIFT = 0.003                     # above the bar centre [m]: the fixed jaw's pad (8 mm under the TCP) 2 mm into the
                                 # bar's bottom (pressed against it)
JAW, JAW_OPEN = CONTROL['mobile_so101_xl']['hand_joint'], 0.0   # 0: a 32 mm gap around the 14 mm bar
JAW_GRIP = -0.3                  # past closed: the effort JTC squeezes the bar at its clamp (~12 N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--robot', required=True)
    ap.add_argument('--task', required=True)
    ap.add_argument('--repo', default='/repo')
    a = ap.parse_args()
    _, scene, _, _ = make_world(a.scene, a.repo)
    robot, urdf, _, contact_links = describe(a.repo, a.robot)
    res = specs.resolve(f'{a.repo}/specs', a.task, robot=a.robot)
    target = specs.parse_entity(res.task.targets['handle'])
    _, boxes, joints = model_geometry(res, target.model)
    T_h, bar = next((T, size) for lk, c, T, size in boxes if lk == target.name and c == target.collision)
    drawer_link = target.name
    slide_joint, axis = next((j, ax) for j, (child, _, ax) in joints.items() if child == drawer_link)
    # TCP: +z into the drawer (= -x of the handle frame), x along the bar (handle y), y = z x x (down: the fixed jaw
    # under the bar)
    R_tcp = np.column_stack([T_h[:3, 1], np.cross(-T_h[:3, 0], T_h[:3, 1]), -T_h[:3, 0]])
    bar_half = 0.5 * bar[1]                         # the bar runs along its collision y

    rclpy.init()
    n = Door(robot.name, scene.name, contact_links, target.model, slide_joint)   # n.door: the slide [m]
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    mp = MoveItPy(node_name='check_open_drawer_moveit', config_dict=moveit_params(a.repo, robot.name, urdf))
    group = MOVEIT[robot.name]['group']
    arm = mp.get_planning_component(group)
    psm = mp.get_planning_scene_monitor()
    model = mp.get_robot_model()
    tcp, plan_frame = robot.frames['tcp'], robot.frames['base']
    log, out = [], {}

    def tf_point(frame):
        tr = n.tf.lookup_transform(plan_frame, frame, rclpy.time.Time(),
                                   timeout=rclpy.duration.Duration(seconds=2)).transform.translation
        return n.base()[0] @ np.array([tr.x, tr.y, tr.z, 1.0])

    def tcp_in_handle():
        """The TCP position in the handle frame at the drawer's current opening [mm]."""
        T = T_h.copy()
        T[:3, 3] += (n.door or 0.0) * axis
        return [round(1000 * float(v), 1) for v in (np.linalg.inv(T) @ tf_point(tcp))[:3]]

    trace, tracing = [], threading.Event()

    def sampler():
        last = {}
        while tracing.is_set():
            with n.lock:
                cnt = dict(n.contacts)
            try:
                e = tcp_in_handle()
            except Exception:
                e = None
            fingers = [int(cnt[lk] > last.get(lk, 0)) for lk in robot.groups['fingers']]
            last = cnt
            trace.append([round(n.now() - t0, 3), round(n.door or 0.0, 4), e, fingers])
            n.wait(0.05)

    def mark(what, **kw):
        log.append({'t': round(n.now() - t0, 3), 'what': what, 'phase': n.phase(), **kw})
        print(json.dumps(log[-1]), file=sys.stderr, flush=True)

    def scene_boxes(T_b, skip_link=None):
        T_inv = np.linalg.inv(T_b)
        with psm.read_write() as sc:
            sc.remove_all_collision_objects()
            for lk, c, T, size in boxes:
                if lk == skip_link:
                    continue
                o = CollisionObject()
                o.header.frame_id, o.id, o.operation = plan_frame, f'{lk}::{c}', CollisionObject.ADD
                o.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=size)]
                o.primitive_poses = [pose_msg(T_inv @ T)]
                sc.apply_collision_object(o)
            sc.current_state.update()

    def run(params_name, goal=None, pose=None, scale=None):
        arm.set_start_state_to_current_state()
        if goal is not None:
            arm.set_goal_state(robot_state=goal)
        else:
            ps = PoseStamped()
            ps.header.frame_id, ps.pose = plan_frame, pose_msg(pose)
            arm.set_goal_state(pose_stamped_msg=ps, pose_link=tcp)
        prp = PlanRequestParameters(mp, params_name)
        if scale is not None:
            prp.max_velocity_scaling_factor = prp.max_acceleration_scaling_factor = scale
        plan = arm.plan(single_plan_parameters=prp)
        if not plan:
            return False
        return bool(mp.execute(plan.trajectory, controllers=[]))

    ok = n.wait_for(lambda: n.states and n.states[-1]['ready'], 60)
    out['ready'] = ok
    n.wait(1.0)
    out['start'] = n.start()
    t0 = n.now()
    mark('start')
    try:
        # --- drive: face the bar (base x = -handle x), REACH in front of its centre
        h, hx = T_h[:3, 3], T_h[:3, 0]
        n.drive_to((h[0] + REACH * hx[0], h[1] + REACH * hx[1]), float(np.arctan2(-hx[1], -hx[0])), V_DRIVE, W_DRIVE)
        n.wait(0.3)
        T_b, pb = n.base()
        mark('drive_done', base=[round(v, 4) for v in pb])

        # --- approach: the grasp point in front of the shoulder pan axis, pre-grasp (OMPL), LIN onto the bar
        along = float((tf_point('shoulder_link')[:3] - h) @ T_h[:3, 1])
        along = max(-bar_half + BAR_MARGIN, min(bar_half - BAR_MARGIN, along))
        T_grasp = np.eye(4)
        T_grasp[:3, :3], T_grasp[:3, 3] = R_tcp, h + along * T_h[:3, 1]
        scene_boxes(T_b)
        T_inv = np.linalg.inv(T_b)
        T_pre = T_grasp.copy()
        T_pre[:3, 3] -= PRE * R_tcp[:, 2]
        g = RobotState(model)
        g.set_to_default_values()
        g.set_joint_group_positions(group, np.array(SEED))
        ik = g.set_from_ik(group, pose_msg(T_inv @ T_pre), tcp, 0.2)
        g.update()
        mark('ik_pre', ok=bool(ik), along=round(along, 4),
             q=[round(float(v), 3) for v in g.get_joint_group_positions(group)])
        out['pre_ok'] = ik and run('ompl_rrtc', goal=g)
        # centre the TCP on the bar in depth (the effort JTC leaves a few mm), then up by LIFT: a centred TCP leaves the
        # fixed jaw 1 mm under the bar, and the stiff arm does not give to the moving jaw's squeeze. The error at the
        # pre-grasp corrects the first LIN's target (the bar centre: the fixed jaw passes under the bar), then up to
        # CENTRE_TRIES more LINs to (depth 0, height LIFT).
        n.wait(0.3)
        e = np.array(tcp_in_handle()) - (1000 * PRE, 1000 * along, 0.0)
        mark('pre_grasp', ok=out['pre_ok'], tcp_mm=[round(float(v), 1) for v in e])
        xz = T_h[:3, [0, 2]]
        T_cmd = T_grasp.copy()
        T_cmd[:3, 3] -= xz @ e[[0, 2]] / 1000.0
        scene_boxes(T_b, skip_link=drawer_link)    # the bar goes between the jaws
        out['lin_ok'] = out['pre_ok'] and run('pilz_lin', pose=T_inv @ T_cmd, scale=LIN_SCALE)
        goal = np.array([0.0, 1000 * LIFT])
        for _ in range(CENTRE_TRIES):
            n.wait(0.3)
            e = np.array(tcp_in_handle())
            mark('at_bar', ok=out['lin_ok'], tcp_mm=e.tolist())
            err = e[[0, 2]] - goal
            if not out['lin_ok'] or max(abs(err)) <= CENTRE_MM:
                break
            T_cmd[:3, 3] -= xz @ err / 1000.0
            out['lin_ok'] = run('pilz_lin', pose=T_inv @ T_cmd, scale=LIN_SCALE)

        # --- grasp
        tracing.set()
        threading.Thread(target=sampler, daemon=True).start()
        n.goal(n.hand, [JAW], [JAW_GRIP], 0.5)
        n.wait_for(lambda: n.phase() not in ('approach', 'grasp'), 8)
        with n.lock:
            jaw = n.joints.get(JAW)
        mark('grasped', tcp_mm=tcp_in_handle(), jaw=round(jaw, 4) if jaw is not None else None)

        # --- open: the base backs off along the slide, the arm holds
        T_b, _ = n.base()
        d = (T_b[:3, :3].T @ axis)[:2]
        n.segment((*(d / np.linalg.norm(d)), 0.0), PULL, V_PULL)
        mark('pull_done', base=[round(v, 4) for v in n.base()[1]], slide=round(n.door or 0.0, 4))
        n.wait_for(lambda: n.result is not None, 15)
        mark('result')
        n.goal(n.hand, [JAW], [JAW_OPEN], 1.0)
        mark('released')
    except Exception as e:   # report what was reached
        out['error'] = repr(e)
    n.wait_for(lambda: n.result is not None, 30)
    tracing.clear()
    out['trace'] = trace
    with n.lock:
        out['result'] = n.result
        sts = list(n.states)
    opn = [st['values'] for st in sts if st['phase'] == 'open']
    out['open_trace'] = {'joint_max': max((v.get('open.done[0]') or 0.0 for v in opn), default=None),
                         'contacts_min': min((v.get('open.while[0]') for v in opn if v.get('open.while[0]')
                                              is not None), default=None),
                         'slip_max': max((v.get('open.while[1]') or 0.0 for v in opn), default=None)}
    out['log'] = log
    print('RESULT ' + json.dumps(out), flush=True)
    sys.stderr.flush()
    os._exit(0)     # moveit_py segfaults in the interpreter's teardown (rc 139): exit after the RESULT line


if __name__ == '__main__':
    main()
