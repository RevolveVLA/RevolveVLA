"""T2 control checks, run inside a simulator container started by `revolve sim` (see conftest.py): prints one JSON
line with the measurements. Commands go only through the upstream interfaces: TwistStamped on
/omni_wheel_drive_controller/cmd_vel (ramped over RAMP_S, as a real source would), FollowJointTrajectory goals to
the arm and hand controllers. Ground truth: the gz model pose (/model/<robot>/pose) and /joint_states.

  base: from robot_start, back / forward, strafe right / left, rotate left / right (each back to the start); the
        steady part of each segment -> body speed error [%], travel heading error [deg], yaw drift [deg].
  arm: the steady-state error at the start pose, then 3 joint targets and `ready`: max error while holding [deg].
  hand: open / close fully; then a static box is spawned between the open fingers and the hand closes on it:
        both finger contact topics must report contacts.
The robot's controllers, joints, targets and probe box: robot_cases.CONTROL.

  python3 check_control.py --scene S --robot R [--repo /repo]
"""
import argparse
import json
import math
import subprocess
import sys
import threading
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import TwistStamped
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

from revolve_sim.robot import describe
from revolve_sim.world import make_world
from robot_cases import CONTROL

RAMP_S, SEG_S, RATE_HZ = 0.5, 3.0, 50.0
SEGMENTS = [('back', (-0.3, 0.0, 0.0)), ('forward', (0.3, 0.0, 0.0)), ('strafe_right', (0.0, -0.3, 0.0)),
            ('strafe_left', (0.0, 0.3, 0.0)), ('rotate_left', (0.0, 0.0, 0.6)), ('rotate_right', (0.0, 0.0, -0.6))]


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Probe(Node):
    def __init__(self, robot, world, contact_links, case=None):
        case = case or CONTROL[robot]
        super().__init__('check_control', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.lock = threading.Lock()
        self.poses, self.joints, self.contacts = [], {}, {lk: 0 for lk in contact_links}
        self.joint_log = None
        self.create_subscription(TFMessage, f'/model/{robot}/pose', self.on_pose, 50)
        self.create_subscription(JointState, '/joint_states', self.on_joints, 50)
        for lk in contact_links:
            self.create_subscription(Contacts, f'/world/{world}/model/{robot}/link/{lk}/sensor/contact/contact',
                                     lambda m, lk=lk: self.on_contact(lk, m), 50)
        self.cmd = self.create_publisher(TwistStamped, '/omni_wheel_drive_controller/cmd_vel', 10)
        self.arm = ActionClient(self, FollowJointTrajectory, f'/{case["arm_controller"]}/follow_joint_trajectory')
        self.hand = ActionClient(self, FollowJointTrajectory, f'/{case["hand_controller"]}/follow_joint_trajectory')
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self)

    def on_pose(self, m):
        t = m.transforms[-1]
        with self.lock:
            self.poses.append((t.header.stamp.sec + t.header.stamp.nanosec * 1e-9, t.transform.translation.x,
                               t.transform.translation.y, t.transform.translation.z, yaw_of(t.transform.rotation)))

    def on_joints(self, m):
        with self.lock:
            self.joints = dict(zip(m.name, m.position))
            if self.joint_log is not None:
                self.joint_log.append(dict(self.joints))

    def on_contact(self, lk, m):
        if m.contacts:
            with self.lock:
                self.contacts[lk] += 1

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def wait(self, sim_s, wall_timeout=120):
        w0 = time.time()
        while self.now() == 0.0 and time.time() - w0 < wall_timeout:   # no /clock yet: sim time 0
            time.sleep(0.005)
        t0 = self.now()
        while self.now() - t0 < sim_s and time.time() - w0 < wall_timeout:
            time.sleep(0.005)

    def twist(self, v):
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'base_footprint'
        m.twist.linear.x, m.twist.linear.y, m.twist.angular.z = v
        self.cmd.publish(m)

    def drive(self, v):
        """Ramp up over RAMP_S, hold SEG_S, ramp down; publish at RATE_HZ (sim time). -> (t_steady0, t_steady1)."""
        t0 = self.now()
        total = 2 * RAMP_S + SEG_S
        while (t := self.now() - t0) < total:
            s = min(1.0, t / RAMP_S, (total - t) / RAMP_S)
            self.twist(tuple(s * c for c in v))
            self.wait(1.0 / RATE_HZ)
        self.twist((0.0, 0.0, 0.0))
        return t0 + RAMP_S + 1.0, t0 + RAMP_S + SEG_S - 0.1

    def goal(self, client, joints, q, duration):
        g = FollowJointTrajectory.Goal()
        g.trajectory.joint_names = joints
        p = JointTrajectoryPoint()
        p.positions = [float(x) for x in q]
        p.time_from_start.sec, p.time_from_start.nanosec = int(duration), int((duration % 1) * 1e9)
        g.trajectory.points = [p]
        assert client.wait_for_server(timeout_sec=30)
        h = client.send_goal_async(g)
        while not h.done():
            time.sleep(0.01)
        r = h.result().get_result_async()
        w0 = time.time()
        while not r.done() and time.time() - w0 < 120:
            time.sleep(0.01)
        return r.result().result.error_code if r.done() else None


def base_segment(poses, t0, t1, v):
    w = np.array([p for p in poses if t0 <= p[0] <= t1])
    t, x, y, yaw = w[:, 0], w[:, 1], w[:, 2], np.unwrap(w[:, 4])
    vx, vy, wz = (np.polyfit(t, s, 1)[0] for s in (x, y, yaw))
    c, s = math.cos(yaw.mean()), math.sin(yaw.mean())
    bx, by = c * vx + s * vy, -s * vx + c * vy
    out = {}
    if v[2] == 0.0:
        cmd = math.hypot(v[0], v[1])
        out['speed_err_pct'] = round(100 * (math.hypot(bx, by) - cmd) / cmd, 2)
        out['heading_err_deg'] = round(math.degrees(wrap(math.atan2(by, bx) - math.atan2(v[1], v[0]))), 2)
        out['yaw_drift_deg'] = round(math.degrees(yaw[-1] - yaw[0]), 2)
    else:
        out['wz_err_pct'] = round(100 * (wz - v[2]) / v[2], 2)
        out['drift_m'] = round(float(math.hypot(x[-1] - x[0], y[-1] - y[0])), 4)
    return out


def quat_mul(a, b):
    """Hamilton product of (x, y, z, w) quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def rotate(q, v):
    x, y, z, _ = quat_mul(quat_mul(q, (*v, 0.0)), (-q[0], -q[1], -q[2], q[3]))
    return np.array([x, y, z])


def rpy_of(q):
    x, y, z, w = q
    return (math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
            math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--robot', required=True)
    ap.add_argument('--repo', default='/repo')
    a = ap.parse_args()
    _, scene, _, _ = make_world(a.scene, a.repo)
    robot, _, _, contact_links = describe(a.repo, a.robot)
    case = CONTROL[robot.name]
    arm_joints, finger = case['arm_joints'], case['hand_joint']
    contact_links = [lk for lk in contact_links if lk in robot.groups.get('fingers', contact_links)]
    start_q = robot.poses[scene.robot_start.arm] if isinstance(scene.robot_start.arm, str) else scene.robot_start.arm

    rclpy.init()
    n = Probe(robot.name, scene.name, contact_links, case)
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    out = {}
    n.wait(3.0)
    with n.lock:
        q = dict(n.joints)
    out['arm_start_err_deg'] = [round(math.degrees(q[j] - t), 3) for j, t in zip(arm_joints, start_q)]

    # base
    out['base'] = {}
    for name, v in SEGMENTS:
        t0, t1 = n.drive(v)
        n.wait(1.0)
        with n.lock:
            poses = list(n.poses)
        out['base'][name] = base_segment(poses, t0, t1, v)
    x0, y0, yaw0 = scene.robot_start.base
    p = poses[-1]
    out['base']['end_offset'] = {'xy_m': round(math.hypot(p[1] - x0, p[2] - y0), 4),
                                 'yaw_deg': round(math.degrees(wrap(p[4] - yaw0)), 2)}

    # arm: targets reached in 3 s, then held 2 s
    out['arm'] = {}
    for name, target in case['arm_targets'].items():
        code = n.goal(n.arm, arm_joints, target, 3.0)
        n.wait(1.0)
        with n.lock:
            n.joint_log = []
        n.wait(2.0)
        with n.lock:
            log, n.joint_log = n.joint_log, None
        err = np.array([[s[j] - t for j, t in zip(arm_joints, target)] for s in log])
        out['arm'][name] = {'result': code, 'max_err_deg': round(float(np.degrees(np.abs(err).max())), 3),
                            'steady_err_deg': [round(float(x), 3) for x in np.degrees(err[-1])]}

    # hand: close, open (free), then close on a box spawned between the open fingers
    out['hand'] = {}
    for name in ('close', 'open'):
        code = n.goal(n.hand, [finger], [case['closed' if name == 'close' else 'open']], 1.0)
        n.wait(1.0)
        with n.lock:
            out['hand'][name] = {'result': code, 'joint': round(n.joints[finger], 4)}
    tcp = n.tf.lookup_transform('base_footprint', robot.frames['tcp'], rclpy.time.Time(),
                                timeout=rclpy.duration.Duration(seconds=5))
    with n.lock:
        _, bx, by, bz, byaw = n.poses[-1]
    r = tcp.transform.rotation
    q_tcp = quat_mul((0.0, 0.0, math.sin(byaw / 2), math.cos(byaw / 2)), (r.x, r.y, r.z, r.w))   # in the world
    tr = tcp.transform.translation
    c = (np.array([bx, by, bz]) + rotate((0.0, 0.0, math.sin(byaw / 2), math.cos(byaw / 2)), (tr.x, tr.y, tr.z))
         + rotate(q_tcp, case['box_offset']))
    rpy = rpy_of(q_tcp) if case['box_aligned'] else (0.0, 0.0, 0.0)
    s = case['box']
    box = ('<sdf version="1.11"><model name="contact_probe"><static>true</static><link name="l"><collision name="c">'
           f'<geometry><box><size>{s} {s} {s}</size></box></geometry></collision></link></model></sdf>')
    r = subprocess.run(['ros2', 'run', 'ros_gz_sim', 'create', '-world', scene.name, '-string', box,
                        '-x', str(c[0]), '-y', str(c[1]), '-z', str(c[2]),
                        '-R', str(rpy[0]), '-P', str(rpy[1]), '-Y', str(rpy[2])], capture_output=True, text=True,
                       timeout=60)
    out['probe_spawned'] = r.returncode == 0
    n.wait(0.5)
    with n.lock:
        n.contacts = {lk: 0 for lk in n.contacts}
    code = n.goal(n.hand, [finger], [case['closed']], 1.0)
    n.wait(1.5)
    with n.lock:
        out['hand']['grasp_box'] = {'result': code, 'joint': round(n.joints[finger], 4),
                                    'contact_msgs': dict(n.contacts)}
    n.goal(n.hand, [finger], [case['open']], 1.0)
    print('RESULT ' + json.dumps(out))
    ex.shutdown()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
