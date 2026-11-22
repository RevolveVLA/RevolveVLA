"""T1 bring-up checks, run inside a simulator container started by `revolve sim` (see conftest.py): prints one JSON
line with the measurements. Reads only the upstream topics: /controller_manager, the robot's ground-truth pose
(/model/<robot>/pose), the cameras, the contact and joint-state topics, gz's model list.
Cameras: the render rate from /<camera>/camera_info (published with every rendered frame, small) over WINDOW_S of sim
time and its longest gap (a render stall), and the image size and content from the first FRAMES raw images. The raw
images (0.7 MB each, over UDP) can reach a fresh subscriber with gaps of 1-2 s under host load, so their arrival times
say little about the render rate.

  python3 check_bringup.py --scene S --robot R [--repo /repo] [--settle 3]
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
from controller_manager_msgs.srv import ListControllers
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_msgs.msg import TFMessage

from revolve_sim.robot import describe
from revolve_sim.world import make_world

WINDOW_S = 5.0      # sim time over which camera_info is counted
FRAMES = 5          # raw images checked for size and content


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Probe(Node):
    def __init__(self, robot, cams, world, models):
        super().__init__('check_bringup', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.pose = None
        self.images = {c: [] for c in cams}
        self.info = {c: [] for c in cams}
        self.joints = {}
        self.create_subscription(TFMessage, f'/model/{robot}/pose', self.on_pose, 10)
        for c in cams:
            self.create_subscription(Image, f'/{c}/image', lambda m, c=c: self.on_image(c, m), 50)
            self.create_subscription(CameraInfo, f'/{c}/camera_info', lambda m, c=c: self.info[c].append(
                m.header.stamp.sec + m.header.stamp.nanosec * 1e-9), 50)
        for m in models:
            self.create_subscription(JointState, f'/world/{world}/model/{m}/joint_state',
                                     lambda msg, m=m: self.joints.__setitem__(m, dict(zip(msg.name, msg.position))),
                                     10)

    def on_pose(self, m):
        for t in m.transforms:
            self.pose = t.transform

    def on_image(self, c, m):
        if len(self.images[c]) < FRAMES:
            self.images[c].append((m.width, m.height,
                                   float(np.frombuffer(bytes(m.data), np.uint8).std()) if m.data else 0.0))

    def sim_now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def spin_for(self, sim_s, wall_timeout=60):
        """Wait sim_s of sim time (the executor thread runs the callbacks)."""
        t0, w0 = self.sim_now(), time.time()
        while self.sim_now() - t0 < sim_s and time.time() - w0 < wall_timeout:
            time.sleep(0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--robot', required=True)
    ap.add_argument('--repo', default='/repo')
    ap.add_argument('--settle', type=float, default=3.0)
    a = ap.parse_args()
    _, scene, _, articulated = make_world(a.scene, a.repo)
    robot, _, cams, contacts = describe(a.repo, a.robot)

    rclpy.init()
    n = Probe(robot.name, cams, scene.name, articulated)
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    out = {}
    cli = n.create_client(ListControllers, '/controller_manager/list_controllers')
    cli.wait_for_service(timeout_sec=30)
    n.spin_for(a.settle)
    res = cli.call(ListControllers.Request(), timeout_sec=10)
    out['controllers'] = {c.name: c.state for c in res.controller} if res else {}
    for c in n.images:
        n.images[c].clear()
        n.info[c].clear()
    n.spin_for(WINDOW_S)
    info = {c: list(st) for c, st in n.info.items()}    # the window's camera_info stamps, not the wait's below
    w0 = time.time()
    while not all(len(ims) >= FRAMES for ims in n.images.values()) and time.time() - w0 < 30:
        time.sleep(0.05)
    out['camera_window_s'] = WINDOW_S
    out['cameras'] = {}
    for c, ims in n.images.items():
        st = sorted(info[c])
        rate = (len(st) - 1) / (st[-1] - st[0]) if len(st) > 2 and st[-1] > st[0] else 0.0
        out['cameras'][c] = {'rate_hz': round(rate, 2), 'n': len(st),
                             'max_gap_s': round(float(np.diff(st).max()), 4) if len(st) > 1 else None,
                             'frames': len(ims), 'size': list(ims[-1][:2]) if ims else None,
                             'min_std': round(min((s for *_, s in ims), default=0.0), 2)}
    x, y, yaw = scene.robot_start.base
    if n.pose is not None:
        p = n.pose
        dyaw = (yaw_of(p.rotation) - yaw + math.pi) % (2 * math.pi) - math.pi
        out['pose_err'] = {'xy_m': round(math.hypot(p.translation.x - x, p.translation.y - y), 4),
                           'yaw_deg': round(math.degrees(dyaw), 3), 'z_m': round(p.translation.z, 4)}
    out['scene_joints'] = n.joints
    topics = dict(n.get_topic_names_and_types())
    out['contact_topics'] = {lk: f'/world/{scene.name}/model/{robot.name}/link/{lk}/sensor/contact/contact' in topics
                             for lk in contacts}
    gz = subprocess.run(['gz', 'model', '--list'], capture_output=True, text=True, timeout=30).stdout
    out['gz_models'] = sorted(ln.strip()[2:] for ln in gz.splitlines() if ln.strip().startswith('- '))
    print('RESULT ' + json.dumps(out))
    ex.shutdown()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
