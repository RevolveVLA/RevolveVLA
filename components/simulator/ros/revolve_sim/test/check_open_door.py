"""T4 positive episode of open_fridge_door, run inside a simulator container started by `revolve sim --task
open_fridge_door --no-autostart` (see conftest.py): prints one RESULT JSON line. MoveIt 2 (moveit_py, the robot's
specs/robots/<r>/moveit/ config) plans the arm; the base gets open-loop, ramped TwistStamped segments (test code
only). The grasp, the hinge and the planning-scene boxes come from the scene's model SDFs (joints at 0) and the base
pose from the gz ground truth.

  drive:    rotate, then translate to the grasp base pose (DIST / SIDE / YAW_DEG from the handle; the folded arm
            stays clear of the door)
  approach: MoveIt OMPL to the pre-grasp (PRE m back along the approach, IK seeded with SEED), Pilz LIN to the bar,
            more LINs until the TCP is on the bar centre within CENTRE_MM (measured: tf + ground truth)
  grasp:    close the hand (hand JTC goal) until the evaluator leaves `grasp`
  open:     Pilz CIRC about the hinge for ARC_ARM, then the base orbits the hinge (one constant body twist: the
            robot turns rigidly with the door) up to ARC_TOTAL; hold until the result; then release
The RESULT has the episode result, a log of the steps and a 20 Hz trace from the grasp on: [t, door angle, TCP in
the handle frame [mm] (tf / ground truth / joint state as they come: approximate while moving), finger contacts].

  python3 check_open_door.py --scene S --robot R --task T [--repo /repo]
"""
import argparse
import json
import math
import os
import sys
import threading
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit_msgs.msg import CollisionObject, Constraints, PositionConstraint
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from revolve_common import specs
from shape_msgs.msg import SolidPrimitive

from check_evaluator import EvalProbe
from robot_cases import CONTROL
from revolve_sim.metrics import rpy_matrix
from revolve_sim.moveit import moveit_params
from revolve_sim.robot import describe
from revolve_sim.world import make_world

# the grasp base pose relative to the handle centre (world frame: the door faces -y): base_footprint DIST in front
# of the handle, SIDE towards the hinge (+x), yaw YAW_DEG; found by an IK search over base poses (the approach
# along +y with the TCP x up needs the wrist turned: these leave >= 0.3 rad to every joint limit, also along the
# arm's arc); SEED: the IK branch
DIST, SIDE, YAW_DEG = 0.50, 0.40, 30.0
SEED = [0.187, -1.429, 1.452, -2.148, -0.017, 3.017, 2.188]
PRE = 0.10                       # pre-grasp: back along the approach [m]
ARC_ARM, ARC_TOTAL = math.radians(20.0), math.radians(70.0)
V_DRIVE, W_DRIVE, W_ORBIT = 0.3, 0.5, 0.12
CIRC_SCALE = 0.1                # Pilz velocity / acceleration scaling of the arm's arc (moveit_py.yaml: pilz_circ)
CENTRE_MM, CENTRE_TRIES = 1.0, 4   # TCP on the bar centre within [mm] (handle x, y) before closing
FINGER, OPEN = CONTROL['mobile_panda']['hand_joint'], CONTROL['mobile_panda']['open']
FINGER_GRIP = -0.04              # finger target past closed: the effort JTC squeezes the 14 mm bar at its 12 N clamp


def quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return x, y, z, w


def pose_msg(T):
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in T[:3, 3])
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat(T[:3, :3])
    return p


def sdf_pose(el):
    return rpy_matrix(*map(float, el.findtext('pose', '0 0 0 0 0 0').split()))


def model_geometry(resolved, model):
    """World frames of a scene model (joints at 0): {link: T}, [(link, collision, T, box size)],
    {joint: (child link, T, axis)}."""
    x, y, z, yaw = resolved.scene.models[model].pose
    T_m = rpy_matrix(x, y, z, 0, 0, yaw)
    m = ET.parse(resolved.models[model].sdf).getroot().find('model')
    links, boxes, joints = {}, [], {}
    for lk in m.findall('link'):
        links[lk.get('name')] = T_l = T_m @ sdf_pose(lk)
        for c in lk.findall('collision'):
            b = c.find('geometry/box/size')
            if b is not None:
                boxes.append((lk.get('name'), c.get('name'), T_l @ sdf_pose(c), [float(v) for v in b.text.split()]))
    for j in (j for j in m.findall('joint') if j.find('axis/xyz') is not None):
        T_j = links[j.findtext('child')] @ sdf_pose(j)       # an SDF joint pose is in its child link's frame
        axis = T_j[:3, :3] @ np.array([float(v) for v in j.findtext('axis/xyz').split()])
        joints[j.get('name')] = (j.findtext('child'), T_j, axis)
    return links, boxes, joints


class Door(EvalProbe):
    def __init__(self, robot, world, contact_links, door_model, hinge_joint):
        super().__init__(robot, world, contact_links)
        self.door, self.hinge_joint = None, hinge_joint
        self.create_subscription(JointState, f'/world/{world}/model/{door_model}/joint_state', self.on_door, 10)

    def on_door(self, m):
        if self.hinge_joint in m.name:
            self.door = m.position[m.name.index(self.hinge_joint)]

    def phase(self):
        with self.lock:
            return self.states[-1]['phase'] if self.states else None


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
    T_h = next(T for lk, c, T, _ in boxes if lk == target.name and c == target.collision)
    door_link = target.name
    hinge_joint, hinge, axis = next((j, T[:3, 3], ax) for j, (child, T, ax) in joints.items() if child == door_link)
    # TCP at the bar centre: +z into the door (= -x of the handle frame), x up (= the bar, handle z), y = z x x
    R_tcp = np.column_stack([T_h[:3, 2], np.cross(-T_h[:3, 0], T_h[:3, 2]), -T_h[:3, 0]])
    T_grasp = np.eye(4)
    T_grasp[:3, :3], T_grasp[:3, 3] = R_tcp, T_h[:3, 3]

    rclpy.init()
    n = Door(robot.name, scene.name, contact_links, target.model, hinge_joint)
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    mp = MoveItPy(node_name='check_open_door_moveit', config_dict=moveit_params(a.repo, robot.name, urdf))
    arm = mp.get_planning_component('panda_arm')
    psm = mp.get_planning_scene_monitor()
    model = mp.get_robot_model()
    tcp = robot.frames['tcp']
    plan_frame = robot.frames['base']
    log, out = [], {}

    def tcp_in_handle():
        """The TCP position in the handle frame at the door's current angle [mm]."""
        tr = n.tf.lookup_transform(plan_frame, tcp, rclpy.time.Time(),
                                   timeout=rclpy.duration.Duration(seconds=2)).transform.translation
        p = n.base()[0] @ np.array([tr.x, tr.y, tr.z, 1.0])
        return [round(1000 * float(v), 1) for v in (np.linalg.inv(door_rot(n.door or 0.0) @ T_h) @ p)[:3]]

    def door_rot(th):
        """World motion of the door link at hinge angle th."""
        c, s = math.cos(th), math.sin(th)
        R = np.eye(4)
        R[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) if axis[2] > 0 else \
            np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        P = np.eye(4)
        P[:3, 3] = hinge
        return P @ R @ np.linalg.inv(P)

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

    def run(params_name, goal=None, pose=None, constraints=None, scale=None):
        arm.set_start_state_to_current_state()
        if goal is not None:
            arm.set_goal_state(robot_state=goal)
        else:
            ps = PoseStamped()
            ps.header.frame_id, ps.pose = plan_frame, pose_msg(pose)
            arm.set_goal_state(pose_stamped_msg=ps, pose_link=tcp)
        if constraints is not None:
            arm.set_path_constraints(constraints)
        prp = PlanRequestParameters(mp, params_name)
        if scale is not None:
            prp.max_velocity_scaling_factor = prp.max_acceleration_scaling_factor = scale
        plan = arm.plan(single_plan_parameters=prp)
        if constraints is not None:
            arm.set_path_constraints(Constraints())
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
        # --- drive: rotate to the grasp yaw, then straight to the grasp base pose
        h = T_h[:3, 3]
        n.drive_to((h[0] + SIDE, h[1] - DIST), math.radians(YAW_DEG), V_DRIVE, W_DRIVE)
        n.wait(0.3)
        T_b, pb = n.base()
        mark('drive_done', base=[round(v, 4) for v in pb])

        # --- approach: pre-grasp (OMPL, joint goal from IK seeded on the chosen branch), LIN onto the bar
        scene_boxes(T_b)
        T_inv = np.linalg.inv(T_b)
        T_pre = T_grasp.copy()
        T_pre[:3, 3] -= PRE * R_tcp[:, 2]
        g = RobotState(model)
        g.set_to_default_values()
        g.set_joint_group_positions('panda_arm', np.array(SEED))
        ik = g.set_from_ik('panda_arm', pose_msg(T_inv @ T_pre), tcp, 0.2)
        g.update()
        mark('ik_pre', ok=bool(ik), q=[round(float(v), 3) for v in g.get_joint_group_positions('panda_arm')])
        out['pre_ok'] = ik and run('ompl_rrtc', goal=g)
        # centre the TCP on the bar (the effort JTC's steady-state error is a few mm): the bar is radial to the
        # hinge there, so a finger touching first cannot push it aside, and the stiff arm does not give either.
        # The error at the pre-grasp corrects the LIN target, then up to CENTRE_TRIES more LINs on the bar.
        n.wait(0.3)
        e = np.array(tcp_in_handle()) - (1000 * PRE, 0.0, 0.0)
        mark('pre_grasp', ok=out['pre_ok'], tcp_mm=[round(float(v), 1) for v in e])
        T_cmd = T_grasp.copy()
        T_cmd[:3, 3] -= T_h[:3, :2] @ e[:2] / 1000.0
        scene_boxes(T_b, skip_link=door_link)    # the bar goes between the fingers
        out['lin_ok'] = out['pre_ok'] and run('pilz_lin', pose=T_inv @ T_cmd)
        for _ in range(CENTRE_TRIES):
            n.wait(0.3)
            e = tcp_in_handle()
            mark('at_bar', ok=out['lin_ok'], tcp_mm=e)
            if not out['lin_ok'] or max(abs(e[0]), abs(e[1])) <= CENTRE_MM:
                break
            T_cmd[:3, 3] -= T_h[:3, :2] @ np.array(e[:2]) / 1000.0
            out['lin_ok'] = run('pilz_lin', pose=T_inv @ T_cmd)

        # --- grasp
        tracing.set()
        threading.Thread(target=sampler, daemon=True).start()
        n.goal(n.hand, [FINGER], [FINGER_GRIP], 0.5)
        n.wait_for(lambda: n.phase() not in ('approach', 'grasp'), 8)
        mark('grasped', tcp_mm=tcp_in_handle())

        # --- open: the arm's arc about the hinge, then the base orbits the hinge
        center = np.append(np.append(hinge[:2], T_grasp[2, 3]), 1.0)
        c_b = (T_inv @ center)[:3]
        T_arc = door_rot(ARC_ARM) @ door_rot(n.door or 0.0) @ T_grasp
        circ = Constraints(name='center')
        pc = PositionConstraint()
        pc.header.frame_id, pc.link_name = plan_frame, tcp
        cp = Pose()
        cp.position.x, cp.position.y, cp.position.z = (float(v) for v in c_b)
        pc.constraint_region.primitive_poses = [cp]
        circ.position_constraints = [pc]
        out['circ_ok'] = run('pilz_circ', pose=T_inv @ T_arc, constraints=circ, scale=CIRC_SCALE)
        mark('arm_arc', ok=out['circ_ok'])
        T_b, pb = n.base()
        hb = (np.linalg.inv(T_b) @ center)[:2]
        w = 1.0 if axis[2] > 0 else -1.0
        n.segment((w * hb[1], -w * hb[0], w), ARC_TOTAL - ARC_ARM, W_ORBIT)
        mark('orbit_done', base=[round(v, 4) for v in n.base()[1]])
        n.wait_for(lambda: n.result is not None, 15)
        mark('result')
        n.goal(n.hand, [FINGER], [OPEN], 1.0)
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
    out['open_trace'] = {'door_max': max((v.get('open.done[0]') or 0.0 for v in opn), default=None),
                         'contacts_min': min((v.get('open.while[0]') for v in opn if v.get('open.while[0]')
                                              is not None), default=None),
                         'slip_max': max((v.get('open.while[1]') or 0.0 for v in opn), default=None)}
    out['log'] = log
    print('RESULT ' + json.dumps(out), flush=True)
    sys.stderr.flush()
    os._exit(0)     # moveit_py segfaults in the interpreter's teardown (rc 139): exit after the RESULT line


if __name__ == '__main__':
    main()
