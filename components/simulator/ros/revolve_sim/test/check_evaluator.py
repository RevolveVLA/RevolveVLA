"""T3 evaluator checks, run inside a simulator container started by `revolve sim --task T --no-autostart` (see
conftest.py): prints one RESULT JSON line. Uses only the evaluator's interfaces (/evaluator/start, /evaluator/state,
/evaluator/result) and the upstream topics.

  idle: before the episode, the evaluator's TCP-to-handle distance vs the same distance from the scene geometry
        (the scene pose of the handle's model, its SDF link and collision poses at joint 0, the box size) and the
        robot's ground-truth pose + tf; then an episode with no command until the result.
  push: (`--to x,y,yaw`: first, before the episode, the base drives to that world pose: rotate, then translate)
        an episode while the base drives (ramped TwistStamped, `--speed` m/s forward) into what is in front of it,
        until the result, then on for `--after` s, recording the evaluator's sanity values.

  python3 check_evaluator.py --scene S --robot R --task T --mode idle|push [--speed 0.3] [--after 3] [--to x,y,yaw]
"""
import argparse
import json
import math
import sys
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile
from revolve_common import specs
from std_msgs.msg import String
from std_srvs.srv import Trigger

from check_control import RAMP_S, RATE_HZ, Probe
from revolve_sim.metrics import inv, point_distance, pose_matrix, rpy_matrix
from revolve_sim.robot import describe
from revolve_sim.world import make_world


class EvalProbe(Probe):
    def __init__(self, robot, world, contact_links):
        super().__init__(robot, world, contact_links)
        self.states, self.result = [], None
        self.create_subscription(String, '/evaluator/state', self.on_state, 50)
        self.create_subscription(String, '/evaluator/result', self.on_result,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.start_cli = self.create_client(Trigger, '/evaluator/start')

    def on_state(self, m):
        with self.lock:
            self.states.append(json.loads(m.data))

    def on_result(self, m):
        with self.lock:
            self.result = json.loads(m.data)

    def wait_for(self, pred, wall_timeout):
        w0 = time.time()
        while not pred() and time.time() - w0 < wall_timeout:
            time.sleep(0.02)
        return pred()

    def start(self):
        assert self.start_cli.wait_for_service(timeout_sec=30), 'no /evaluator/start'
        f = self.start_cli.call_async(Trigger.Request())
        assert self.wait_for(f.done, 30), '/evaluator/start timed out'
        return f.result().success, f.result().message

    def segment(self, v, amount, rate):
        """Ramped (RAMP_S) constant body twist v (unit direction scaled by rate) covering `amount` (m or rad)."""
        hold = max(0.0, amount / rate - RAMP_S)
        total, t0 = hold + 2 * RAMP_S, self.now()
        while (t := self.now() - t0) < total:
            s = min(1.0, t / RAMP_S, (total - t) / RAMP_S)
            self.twist(tuple(s * rate * c for c in v))
            self.wait(1.0 / RATE_HZ)
        for _ in range(5):
            self.twist((0.0, 0.0, 0.0))
            self.wait(1.0 / RATE_HZ)

    def base(self):
        with self.lock:
            _, x, y, z, yaw = self.poses[-1]
        return rpy_matrix(x, y, z, 0, 0, yaw), (x, y, yaw)

    def drive_to(self, goal_xy, yaw_g, v, w):
        """Open loop to a world pose: rotate to yaw_g (at w rad/s), then translate straight to goal_xy (v m/s)."""
        _, (x, y, yaw) = self.base()
        dyaw = math.atan2(math.sin(yaw_g - yaw), math.cos(yaw_g - yaw))
        self.segment((0.0, 0.0, math.copysign(1.0, dyaw)), abs(dyaw), w)
        _, (x, y, yaw) = self.base()
        d = np.asarray(goal_xy) - (x, y)
        c, s = math.cos(yaw), math.sin(yaw)
        body = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        self.segment((*(body / np.linalg.norm(body)), 0.0), float(np.linalg.norm(body)), v)


def handle_from_scene(resolved, target):
    """World pose (4x4) + geometry of a collision from the scene pose and the model SDF (joints at 0)."""
    e = specs.parse_entity(target)
    x, y, z, yaw = resolved.scene.models[e.model].pose
    T_model = rpy_matrix(x, y, z, 0, 0, yaw)
    m = ET.parse(resolved.models[e.model].sdf).getroot().find('model')
    link = next(lk for lk in m.findall('link') if lk.get('name') == e.name)
    col = next(c for c in link.findall('collision') if c.get('name') == e.collision)
    T = T_model @ rpy_matrix(*map(float, link.findtext('pose').split())) @ \
        rpy_matrix(*map(float, col.findtext('pose').split()))
    size = [float(v) for v in col.find('geometry/box/size').text.split()]
    return T, ('box', {'size': size})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--robot', required=True)
    ap.add_argument('--task', required=True)
    ap.add_argument('--mode', choices=('idle', 'push'), required=True)
    ap.add_argument('--speed', type=float, default=0.3)
    ap.add_argument('--after', type=float, default=3.0)
    ap.add_argument('--to', type=lambda v: [float(x) for x in v.split(',')])
    ap.add_argument('--repo', default='/repo')
    a = ap.parse_args()
    _, scene, _, _ = make_world(a.scene, a.repo)
    robot, _, _, contact_links = describe(a.repo, a.robot)
    res = specs.resolve(f'{a.repo}/specs', a.task, robot=a.robot)

    rclpy.init()
    n = EvalProbe(robot.name, scene.name, contact_links)
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    out = {}
    ok = n.wait_for(lambda: n.states and n.states[-1]['ready'], 60)
    out['ready'] = ok
    n.wait(1.0)
    if a.mode == 'push' and a.to:
        n.drive_to(a.to[:2], a.to[2], 0.3, 0.5)
        n.wait(1.0)
        out['push_from'] = [round(v, 4) for v in n.base()[1]]
    with n.lock:
        out['state_before'] = n.states[-1] if n.states else None
        _, bx, by, bz, byaw = n.poses[-1]
    if a.mode == 'idle':
        # the evaluator's distance vs the scene geometry (tcp: ground truth x tf)
        tr = n.tf.lookup_transform(robot.frames['base'], robot.frames['tcp'], rclpy.time.Time(),
                                   timeout=rclpy.duration.Duration(seconds=5)).transform
        T_base = rpy_matrix(bx, by, bz, 0, 0, byaw)
        p_tcp = (T_base @ pose_matrix((tr.translation.x, tr.translation.y, tr.translation.z),
                                      (tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w)))[:3, 3]
        T_h, geom = handle_from_scene(res, res.task.targets['handle'])
        out['handle_center'] = [round(float(v), 4) for v in T_h[:3, 3]]
        out['tcp'] = [round(float(v), 4) for v in p_tcp]
        out['distance_geometry'] = round(point_distance((inv(T_h) @ np.append(p_tcp, 1.0))[:3], geom), 4)
        key = next(f'{p.name}.done[{i}]' for p in res.task.phases for i, c in enumerate(p.done)
                   if c.metric == 'distance')
        out['distance_key'] = key
        out['distance_evaluator'] = out['state_before']['values'][key]
    out['start'] = n.start()
    with n.lock:
        n.states.clear()
    if a.mode == 'push':
        t0 = n.now()
        t_res = None
        while True:
            t = n.now() - t0
            s = min(1.0, t / RAMP_S)
            n.twist((s * a.speed, 0.0, 0.0))
            n.wait(1.0 / RATE_HZ)
            if t_res is None and n.result is not None:
                t_res = n.now()
            if (t_res is not None and n.now() - t_res > a.after) or t > 30.0:
                break
        for _ in range(10):
            n.twist((0.0, 0.0, 0.0))
            n.wait(1.0 / RATE_HZ)
        with n.lock:
            sts = list(n.states)
        peak = {}
        for st in sts:
            for k, v in st['values'].items():
                if k.startswith('sanity.') and v is not None:
                    peak[k] = max(peak.get(k, v), v)
        out['sanity_peak'] = peak
        out['penetration_trace'] = [(st['t'], st['values'].get('sanity.penetration'),
                                     st['values'].get('sanity.robot_contacts')) for st in sts][::5]
    else:
        n.wait_for(lambda: n.result is not None, 120)
    with n.lock:
        out['result'] = n.result
        sts = list(n.states)
    if len(sts) > 1:
        out['state_rate_hz'] = round((len(sts) - 1) / (sts[-1]['t'] - sts[0]['t']), 2) if sts[-1]['t'] > sts[0]['t'] \
            else None
    print('RESULT ' + json.dumps(out))
    ex.shutdown()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
