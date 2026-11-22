"""Reset checks, run inside a simulator container started by `revolve sim --task T --no-autostart` (see conftest.py):
prints one RESULT JSON line. Uses only the client interfaces: /simulator/reset, /evaluator/{state,start,result},
the command topics / actions and the upstream state topics.

  loop: --resets N episodes back to back, each started on /evaluator/start once /evaluator/state is ready again:
        even episodes drive the base back (ramped TwistStamped) and move the arm to a T2 target (its tracking error
        while holding), then reset during the episode (result reason `reset`); odd ones stay idle until the result
        (the approach timeout), then reset. After each reset: the base vs robot_start, the arm vs robot_start.arm,
        the scene joints vs the scene, the controllers' states, the reset's wall time, and per episode the sim/wall
        rate.
  once: one reset from the state the sim is in (e.g. after the T4 positive episode), with the same checks and the
        scene joints before it.

  python3 check_reset.py --scene S --robot R --task T --mode loop|once [--resets 5]
"""
import argparse
import json
import math
import sys
import threading
import time

import numpy as np
import rclpy
from controller_manager_msgs.srv import ListControllers
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from check_control import SEGMENTS, wrap
from check_evaluator import EvalProbe
from revolve_sim.robot import describe
from revolve_sim.world import make_world
from robot_cases import CONTROL


class ResetProbe(EvalProbe):
    def __init__(self, robot, world, contact_links, articulated):
        super().__init__(robot, world, contact_links)
        self.scene_q = {}
        for m in articulated:
            self.create_subscription(JointState, f'/world/{world}/model/{m}/joint_state',
                                     lambda msg, m=m: self.on_scene(m, msg), 10)
        self.reset_cli = self.create_client(Trigger, '/simulator/reset')
        self.list_cli = self.create_client(ListControllers, '/controller_manager/list_controllers')

    def on_scene(self, m, msg):
        with self.lock:
            self.scene_q.update({(m, j): q for j, q in zip(msg.name, msg.position)})

    def call(self, cli, req, timeout):
        assert cli.wait_for_service(timeout_sec=30), f'no {cli.srv_name}'
        f = cli.call_async(req)
        assert self.wait_for(f.done, timeout), f'{cli.srv_name} timed out'
        return f.result()

    def reset(self):
        """-> (success, message, wall s, sim time before, the evaluator's reset count before)."""
        with self.lock:
            resets = self.states[-1]['resets']
        before, w0 = self.now(), time.time()
        r = self.call(self.reset_cli, Trigger.Request(), 60)
        return r.success, r.message, round(time.time() - w0, 3), before, resets

    def ready_after(self, resets, timeout=60):
        """A ready state published after the reset (the client contract: `resets` above its count before)."""
        return self.wait_for(lambda: any(s['ready'] and s['resets'] > resets for s in list(self.states)), timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--robot', required=True)
    ap.add_argument('--task', required=True)
    ap.add_argument('--mode', choices=('loop', 'once'), required=True)
    ap.add_argument('--resets', type=int, default=5)
    ap.add_argument('--repo', default='/repo')
    a = ap.parse_args()
    _, scene, _, articulated = make_world(a.scene, a.repo)
    robot, _, _, contact_links = describe(a.repo, a.robot)
    case = CONTROL[robot.name]
    arm_joints = case['arm_joints']
    start_q = robot.poses[scene.robot_start.arm] if isinstance(scene.robot_start.arm, str) else scene.robot_start.arm
    x0, y0, yaw0 = scene.robot_start.base

    rclpy.init()
    n = ResetProbe(robot.name, scene.name, contact_links, articulated)
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()

    def scene_err():
        with n.lock:
            q = dict(n.scene_q)
        return {f'{m}::{j}': round(v - scene.models[m].joints.get(j, 0.0), 5) for (m, j), v in q.items()}

    def after_reset(ok, msg, wall, before, resets):
        """The checks right after a reset returned."""
        time.sleep(0.3)                    # a few fresh base pose / joint state messages
        with n.lock:
            _, x, y, _, yaw = n.poses[-1]
            q = dict(n.joints)
        cs = n.call(n.list_cli, ListControllers.Request(), 30).controller
        return {'ok': ok, 'message': msg, 'wall_s': wall, 'sim_before': round(before, 3),
                'base_xy_m': round(math.hypot(x - x0, y - y0), 5),
                'base_yaw_deg': round(math.degrees(abs(wrap(yaw - yaw0))), 3),
                'arm_err_deg': round(max(math.degrees(abs(q[j] - v)) for j, v in zip(arm_joints, start_q)), 3),
                'scene_err': scene_err(),
                'controllers': {c.name: c.state for c in cs},
                'ready': n.ready_after(resets)}

    out = {'resets': [], 'episodes': []}
    assert n.wait_for(lambda: n.states and n.states[-1]['ready'], 60), 'evaluator never ready'
    n.wait(1.0)
    if a.mode == 'once':
        out['scene_before'] = scene_err()
        out['resets'].append(after_reset(*n.reset()))
        print('RESULT ' + json.dumps(out))
        ex.shutdown()
        rclpy.shutdown()
        return 0

    back = dict(SEGMENTS)['back']
    for k in range(a.resets + 1):
        ep = {'k': k, 'kind': 'move' if k % 2 == 0 else 'idle'}
        with n.lock:
            n.result = None
        w0, s0 = time.time(), n.now()
        ep['start'] = n.start()
        if ep['kind'] == 'move':
            n.drive(back)                   # ~1.1 m backwards (the T2 `back` segment)
            n.goal(n.hand, [case['hand_joint']], [case['closed']], 1.0)
            target = case['arm_targets']['A']
            ep['arm_result'] = n.goal(n.arm, arm_joints, target, 3.0)
            n.wait(1.0)
            with n.lock:
                n.joint_log = []
            n.wait(1.0)
            with n.lock:
                log, n.joint_log = n.joint_log, None
            ep['arm_hold_err_deg'] = round(float(np.degrees(np.abs(
                [[s[j] - t for j, t in zip(arm_joints, target)] for s in log]).max())), 3)
            with n.lock:
                _, x, y, _, _ = n.poses[-1]
            ep['base_moved_m'] = round(math.hypot(x - x0, y - y0), 3)
        else:
            n.wait_for(lambda: n.result is not None, 120)
        ep['rtf'] = round((n.now() - s0) / (time.time() - w0), 3)
        if k == a.resets:                   # the last one runs to its result
            n.wait_for(lambda: n.result is not None, 120)
        else:
            out['resets'].append(after_reset(*n.reset()))
            n.wait_for(lambda: n.result is not None, 10)   # a move episode ends with the reset
        res = n.final_result()
        ep['result'] = {k2: res.get(k2) for k2 in ('episode', 'passed', 'failed_phase', 'reason', 'duration_s',
                                                    'rate_hz', 'result_file', 'bag')} if res else None
        out['episodes'].append(ep)
    print('RESULT ' + json.dumps(out))
    ex.shutdown()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
