"""Contract check: the simulator as datagen uses it (components/datagen/DATAGEN-v2.md). Runs in its own client
container on the host network with the sim's ROS_DOMAIN_ID, next to a sim started by `revolve sim --task T
--no-autostart --record images --out DIR` (test_contract.py); prints one RESULT JSON line.

--scene: the scene name or a scene instance file (its robot_start is checked). Client, through the documented
interface only: wait for /evaluator/state `ready`; /evaluator/start; episode 0: a
ramped TwistStamped segment backwards, a FollowJointTrajectory goal to the arm and one to the hand, then single-point
JointTrajectory messages streamed at 30 Hz (as a policy commands); wait for /evaluator/result and the result file
(written once the episode's bag is closed); /simulator/reset (after reading `resets`), wait for `resets` above it and
`ready`; episode 1: a short stream, the result, the result file.
Then the files in DIR: per episode the result JSON and its bag (episode_<n>_bag/): every topic of
config/record.yaml (filled in for this scene and robot) present with messages (contact topics: only while in
contact, commands only once commanded), the cameras JPEG at ~30 Hz, /tf_static and /robot_description, the controllers' controller_state; and the
30 Hz action series (each JTC's controller_state reference, the base cmd_vel) extracted from the bag.

  python3 check_contract.py --scene S --robot R --task T --out DIR
"""
import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import rosbag2_py
import yaml
from rclpy.executors import MultiThreadedExecutor
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from check_evaluator import EvalProbe
from revolve_sim.robot import controller_types, describe, robot_dir
from revolve_sim.world import make_world
from robot_cases import CONTROL

JTC = 'joint_trajectory_controller/JointTrajectoryController'
OMNI = 'omni_wheel_drive_controller/OmniWheelDriveController'
ACTION_HZ = 30.0
CMD_VEL_TIMEOUT_S = 0.5      # the omni controller's cmd_vel_timeout: no command for longer = zero twist
BACK_MPS, BACK_M = 0.2, 0.3
STREAM_AMPL = 0.1            # rad on the first arm joint, 0.5 Hz
ON_DEMAND = ('/sensor/contact/', '/joint_trajectory', '/cmd_vel')   # topics with messages only while in contact /
                                                                     # commanded (in a bag once they had a publisher)


def frozen_topics(repo, scene_name, robot_name, images):
    """config/record.yaml filled in for the scene and robot -> [topic]."""
    rec = yaml.safe_load((Path(repo) / 'components/simulator/ros/revolve_sim/config/record.yaml').read_text())
    _, scene, _, articulated = make_world(scene_name, repo)
    robot, _, cams, contacts = describe(repo, robot_name)
    types = controller_types(robot_dir(repo, robot.name) / robot.controllers)
    lists = {'{camera}': cams, '{contact_link}': contacts, '{model}': articulated, '{scene_model}': list(scene.models),
             '{jtc}': [n for n, t in types.items() if t == JTC], '{omni}': [n for n, t in types.items() if t == OMNI]}
    out = []
    for e in rec['topics'] + (rec['images'] if images else []):
        key = next((k for k in lists if k in e), None)
        for v in (lists[key] if key else [None]):
            out.append((e.replace(key, v) if key else e).replace('{world}', scene.name).replace('{robot}', robot.name))
    return out


def read_bag(path, decode):
    """-> ({topic: type}, {topic: [bag time s]}, {topic: [message]} for the topics in decode)."""
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    times, msgs = {t: [] for t in types}, {t: [] for t in decode}
    while r.has_next():
        topic, data, t = r.read_next()
        times[topic].append(t * 1e-9)
        if topic in msgs:
            msgs[topic].append(deserialize_message(data, get_message(types[topic])))
    return types, times, msgs


def stamp(m):
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def action_series(states, cmds, t0, t1):
    """The 30 Hz action on [t0, t1]: per JTC its controller_state reference positions (the last state at or before
    each tick), the base twist (the last cmd_vel within CMD_VEL_TIMEOUT_S, else zero) -> (grid, {name: array})."""
    grid = np.arange(t0, t1, 1.0 / ACTION_HZ)
    out = {}
    for name, ms in states.items():
        st = np.array([stamp(m) for m in ms])
        ref = np.array([m.reference.positions for m in ms])
        out[name] = ref[np.clip(np.searchsorted(st, grid, side='right') - 1, 0, len(st) - 1)]
    ct = np.array([stamp(m) for m in cmds]) if cmds else np.zeros(0)
    cv = np.array([[m.twist.linear.x, m.twist.linear.y, m.twist.angular.z] for m in cmds]) if cmds else np.zeros((0, 3))
    base = np.zeros((len(grid), 3))
    if len(ct):
        i = np.searchsorted(ct, grid, side='right') - 1
        ok = (i >= 0) & (grid - ct[np.clip(i, 0, None)] <= CMD_VEL_TIMEOUT_S)
        base[ok] = cv[i[ok]]
    out['cmd_vel'] = base
    return grid, out


def at(grid, series, t):
    return series[min(max(int(np.searchsorted(grid, t, side='right')) - 1, 0), len(grid) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--robot', required=True)
    ap.add_argument('--task', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--repo', default='/repo')
    a = ap.parse_args()
    out_dir = Path(a.out)
    _, scene, _, _ = make_world(a.scene, a.repo)
    robot, _, cams, contact_links = describe(a.repo, a.robot)
    case = CONTROL[robot.name]
    arm, hand, arm_joints = case['arm_controller'], case['hand_controller'], case['arm_joints']
    target = case['arm_targets']['A']

    rclpy.init()
    n = EvalProbe(robot.name, scene.name, contact_links)
    stream = n.create_publisher(JointTrajectory, f'/{arm}/joint_trajectory', 10)
    reset_cli = n.create_client(Trigger, '/simulator/reset')
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()

    def send_stream(q0, duration):
        """Single-point JointTrajectory at 30 Hz (time_from_start 0.1 s), joint 1 on a 0.5 Hz sine. -> (t0, t1, last)"""
        t0 = n.now()
        while (t := n.now() - t0) < duration:
            q = list(q0)
            q[0] += STREAM_AMPL * math.sin(math.pi * t)
            m = JointTrajectory()
            m.joint_names = arm_joints
            m.points = [JointTrajectoryPoint(positions=[float(v) for v in q])]
            m.points[0].time_from_start.nanosec = 100_000_000
            stream.publish(m)
            n.wait(1.0 / ACTION_HZ)
        return t0, n.now(), q

    def episode_files(k, wall_timeout=60):
        """The result (topic) -> wait for its result file; at that moment its bag must be closed."""
        assert n.wait_for(lambda: n.result is not None and n.result['episode'] == k, wall_timeout), f'no result {k}'
        res = n.result
        rf = Path(res['result_file'])
        assert n.wait_for(rf.is_file, 30), f'no result file {rf}'
        return res, (Path(res['bag']) / 'metadata.yaml').is_file() if res['bag'] else False

    out = {'episodes': []}
    w_ready = time.time()
    assert n.wait_for(lambda: n.states and n.states[-1]['ready'], 180), 'evaluator never ready'
    out['ready_wall_s'] = round(time.time() - w_ready, 2)
    x0, y0, _ = scene.robot_start.base

    def start_err():
        with n.lock:
            _, x, y, _, _ = n.poses[-1]
        return round(math.hypot(x - x0, y - y0), 4)
    out['start_err_m'] = start_err()       # the scene (instance)'s robot_start
    marks = {}
    # episode 0
    with n.lock:
        n.result = None
    out['start'] = [n.start()]
    k0 = int(out['start'][0][1].split()[-1])        # the reply's message: "episode <n>"
    marks['back'] = (n.now(), None)
    n.segment((-1.0, 0.0, 0.0), BACK_M, BACK_MPS)
    marks['back'] = (marks['back'][0], n.now())
    out['arm_goal'] = n.goal(n.arm, arm_joints, target, 2.0)
    marks['arm_goal_end'] = n.now()
    out['hand_goal'] = n.goal(n.hand, [case['hand_joint']], [case['closed']], 1.0)
    marks['hand_goal_end'] = n.now()
    s0, s1, last_q = send_stream(target, 2.0)
    marks['stream'] = (s0, s1)
    res0, closed0 = episode_files(k0)
    out['episodes'].append({'result': res0, 'bag_closed_at_result_file': closed0, 'marks': marks})
    # reset + episode 1
    with n.lock:
        resets = n.states[-1]['resets']
    assert reset_cli.wait_for_service(timeout_sec=30)
    w0 = time.time()
    f = reset_cli.call_async(Trigger.Request())
    assert n.wait_for(f.done, 60), '/simulator/reset timed out'
    out['reset'] = {'success': f.result().success, 'message': f.result().message, 'wall_s': round(time.time() - w0, 2)}
    assert n.wait_for(lambda: any(s['ready'] and s['resets'] > resets for s in list(n.states)), 60), 'not ready'
    out['reset']['ready_wall_s'] = round(time.time() - w0, 2)
    out['reset']['start_err_m'] = start_err()
    with n.lock:
        n.result = None
    out['start'].append(n.start())
    k1 = int(out['start'][1][1].split()[-1])
    send_stream(robot.poses[scene.robot_start.arm] if isinstance(scene.robot_start.arm, str)
                else scene.robot_start.arm, 1.0)
    res1, closed1 = episode_files(k1)
    out['episodes'].append({'result': res1, 'bag_closed_at_result_file': closed1, 'marks': {}})
    ex.shutdown()
    rclpy.shutdown()

    # the files
    expected = frozen_topics(a.repo, a.scene, a.robot, images=True)
    jtcs = [t.split('/')[1] for t in expected if t.endswith('/controller_state')]
    cmd_topic = next(t for t in expected if t.endswith('/cmd_vel'))
    out['files'] = sorted(p.name for p in out_dir.iterdir())
    for ep in out['episodes']:
        res = ep['result']
        bag = Path(res['bag'])
        ep['in_out_dir'] = bag.parent == out_dir and Path(res['result_file']).parent == out_dir
        ep['bag_bytes'] = sum(p.stat().st_size for p in bag.iterdir())
        decode = [f'/{j}/controller_state' for j in jtcs] + [cmd_topic, '/evaluator/result'] + \
            [f'/{c}/image/compressed' for c in cams]
        types, times, msgs = read_bag(bag, decode)
        on_demand = lambda t: any(k in t for k in ON_DEMAND)   # noqa: E731
        ep['missing'] = [t for t in expected if t not in types and not on_demand(t)]
        ep['extra'] = sorted(set(types) - set(expected))
        ep['empty'] = [t for t in expected if t in types and not times[t] and not on_demand(t)]
        ep['counts'] = {t: len(v) for t, v in times.items()}
        span = lambda ts: ts[-1] - ts[0] if len(ts) > 1 else 0.0   # noqa: E731
        ep['rates_hz'] = {t: round((len(times[t]) - 1) / span(times[t]), 2) for t in
                          [*(f'/{c}/image/compressed' for c in cams), *(f'/{j}/controller_state' for j in jtcs),
                           '/joint_states', '/tf'] if span(times.get(t, []))}
        ep['jpeg'] = all('jpeg' in m.format for c in cams for m in msgs[f'/{c}/image/compressed'][:5])
        ep['result_in_bag'] = [json.loads(m.data)['episode'] for m in msgs['/evaluator/result']]
        states = {j: msgs[f'/{j}/controller_state'] for j in jtcs}
        t0 = max(stamp(ms[0]) for ms in states.values())
        t1 = min(stamp(ms[-1]) for ms in states.values())
        grid, series = action_series(states, msgs[cmd_topic], t0, t1)
        ep['series'] = {'n': len(grid), 'duration_s': round(t1 - t0, 3), 'hz': round((len(grid) - 1) / (grid[-1] - grid[0]), 3),
                        'finite': bool(all(np.isfinite(v).all() for v in series.values())),
                        'dims': {k: list(v.shape) for k, v in series.items()}}
        if ep['marks']:
            m = ep['marks']
            ep['series']['cmd_vx_min'] = round(float(series['cmd_vel'][:, 0].min()), 4)
            ep['series']['arm_ref_err_at_goal_end'] = round(float(np.abs(      # 0.1 s: the next 30 Hz tick
                at(grid, series[arm], m['arm_goal_end'] + 0.1) - target).max()), 5)
            ep['series']['hand_ref_at_goal_end'] = round(float(at(grid, series[hand], m['hand_goal_end'] + 0.1)[0]), 5)
            in_stream = (grid >= m['stream'][0] + 0.2) & (grid <= m['stream'][1])
            ep['series']['stream_joint1_range'] = round(float(np.ptp(series[arm][in_stream, 0])), 4)
            ep['series']['stream_last_err'] = round(float(np.abs(
                at(grid, series[arm], m['stream'][1] + 0.15) - np.array(last_q)).max()), 4)
    print('RESULT ' + json.dumps(out, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
