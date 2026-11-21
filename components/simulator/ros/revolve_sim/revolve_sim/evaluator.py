"""The task evaluator: a ROS 2 node on sim time that evaluates a task's phases and sanity checks live
(revolve_sim.episode, revolve_sim.metrics), fed only by upstream publishers:
  robot links      the ground-truth base pose (gz PosePublisher, /model/<robot>/pose) x robot_state_publisher tf
  scene links      gz PosePublisher of every scene model (/model/<m>/pose: model in the world, links in the model)
  scene joints     gz JointStatePublisher (/world/<w>/model/<m>/joint_state)
  contacts         gz Contact sensors on the robot links (/world/<w>/model/<robot>/link/<l>/sensor/contact/contact)
  commands         the controllers' command topics (<omni>/cmd_vel, <jtc>/joint_trajectory): non-finite messages
  gz log           the gz server's console log (~/.gz/sim/log/<time>/server_console.log): [error] / [critical] lines
An episode counts the non-finite commands and gz log lines since its start (bring-up errors are not its own).

Episodes: ~/start (std_srvs/Trigger) starts one (autostart: the first one starts as soon as every input is live).
At the end the result (JSON) is published on ~/result (std_msgs/String, transient local) and written to
<run_dir>/episode_<n>.json; a rosbag2 recorder started paused by the launch (rosbag2_recorder) is resumed for the
episode, then paused and split, so episode n is the bag's file n (bag/<n>_bag_<time>.mcap). ~/state (std_msgs/String JSON, 10 Hz): the sim
time, the active phase and every condition's current value (also between episodes).

  ros2 run revolve_sim evaluator --ros-args -p task:=<t> -p scene:=<s> -p robot:=<r> [-p run_dir:=DIR]
"""
import glob
import json
import math
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TwistStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from revolve_common import specs
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory

from revolve_sim.episode import Episode
from revolve_sim.metrics import GROUND, Metrics, Snapshot, collision_geometry, pose_matrix
from revolve_sim.robot import command_topics, describe, robot_dir
from revolve_sim.world import make_world

CONTACT_WINDOW_S = 0.05     # a sensor's contacts count until this long after its last message (it publishes at 100 Hz)
GZ_PROBLEM = re.compile(r'\[(error|critical)\]')
MSG_TYPES = {'geometry_msgs/msg/TwistStamped': TwistStamped, 'trajectory_msgs/msg/JointTrajectory': JointTrajectory}


def stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def tf_matrix(t):
    tr, q = t.translation, t.rotation
    return pose_matrix((tr.x, tr.y, tr.z), (q.x, q.y, q.z, q.w))


def finite(msg):
    if isinstance(msg, TwistStamped):
        v = msg.twist
        vals = [v.linear.x, v.linear.y, v.linear.z, v.angular.x, v.angular.y, v.angular.z]
    else:
        vals = [x for p in msg.points for x in (*p.positions, *p.velocities, *p.accelerations, *p.effort)]
    return all(math.isfinite(x) for x in vals)


def urdf_links(urdf):
    """(root link, [links])."""
    root = ET.fromstring(urdf)
    links = [lk.get('name') for lk in root.findall('link')]
    children = {j.find('child').get('link') for j in root.findall('joint')}
    return next(lk for lk in links if lk not in children), links


class Evaluator(Node):
    def __init__(self):
        super().__init__('evaluator', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=True)])
        p = {k: self.declare_parameter(k, d).value for k, d in (
            ('repo', os.environ.get('REVOLVE_REPO', '/repo')), ('scene', ''), ('robot', ''), ('task', ''),
            ('run_dir', ''), ('autostart', True), ('rate_hz', 100.0), ('gz_log', ''), ('bag', True))}
        repo = Path(p['repo']).resolve()
        _, scene, _, articulated = make_world(p['scene'], repo)
        self.res = specs.resolve(repo / 'specs', p['task'], robot=p['robot'])
        robot, urdf, _, contact_links = describe(repo, p['robot'], arm=scene.robot_start.arm)
        self.robot, self.world = robot.name, scene.name
        self.root, links = urdf_links(urdf)
        collisions = {m: collision_geometry(info.sdf) for m, info in self.res.models.items() if info.sdf}
        self.metrics = Metrics(self.res, links, collisions)
        self.run_dir = Path(p['run_dir'] or repo / 'runs' / 'sim' / f'{time.strftime("%Y%m%d_%H%M%S")}_{p["task"]}')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.autostart, self.bag, self.gz_log_path = p['autostart'], p['bag'], p['gz_log']
        self.conditions = [(k, c) for ph in self.res.task.phases for k, c in Episode.phase_keys(ph)] + \
            [(f'sanity.{k}', c) for k, c in self.res.sanity.items()]

        self.lock = threading.Lock()
        self.base = None                    # world <- robot root link
        self.model_poses, self.link_poses, self.joints = {}, {}, {}
        self.contacts = {}                  # topic -> (stamp, [(c1, c2, depth)])
        self.bad_commands, self.gz_lines, self.gz_offset = 0, [], 0
        self.counted_from = (0, 0)          # (commands, gz lines) before the episode: it counts only its own
        self.episode, self.n_episodes, self.ticks = None, 0, 0
        self.values, self.ready = {}, False
        self.tf_links = []                  # the robot links tf has (not the passive rollers: no joint states)
        self.tick_wall = []                 # wall time of the ticks of the running episode [s]

        subs = MutuallyExclusiveCallbackGroup()
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self, spin_thread=False)
        self.create_subscription(TFMessage, f'/model/{self.robot}/pose', self.on_robot_pose, 10,
                                 callback_group=subs)
        for m in scene.models:
            self.create_subscription(TFMessage, f'/model/{m}/pose', self.on_model_pose, 10, callback_group=subs)
        for m in articulated:
            self.create_subscription(JointState, f'/world/{self.world}/model/{m}/joint_state',
                                     lambda msg, m=m: self.on_joints(m, msg), 10, callback_group=subs)
        for lk in contact_links:
            topic = f'/world/{self.world}/model/{self.robot}/link/{lk}/sensor/contact/contact'
            self.create_subscription(Contacts, topic, lambda msg, t=topic: self.on_contacts(t, msg), 50,
                                     callback_group=subs)
        for topic, typ in command_topics(robot_dir(repo, robot.name) / robot.controllers):
            self.create_subscription(MSG_TYPES[typ], topic, self.on_command, 10, callback_group=subs)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.result_pub = self.create_publisher(String, '~/result', latched)
        self.state_pub = self.create_publisher(String, '~/state', 10)
        self.bag_clients = {}
        if self.bag:
            from rosbag2_interfaces.srv import Pause, Resume, SplitBagfile
            self.bag_clients = {k: (self.create_client(t, f'/rosbag2_recorder/{k}'), t)
                                for k, t in (('resume', Resume), ('pause', Pause), ('split_bagfile', SplitBagfile))}
        tick = MutuallyExclusiveCallbackGroup()     # the ticks, ~/start and the episode state
        self.create_service(Trigger, '~/start', self.on_start, callback_group=tick)
        self.create_timer(1.0 / p['rate_hz'], self.tick, callback_group=tick)
        self.create_timer(0.1, self.publish_state, callback_group=tick)
        self.create_timer(0.2, self.read_gz_log, callback_group=tick)
        self.create_timer(1.0, self.refresh_frames, callback_group=tick)
        self.get_logger().info(f'task {self.res.task.name} on {self.world} with {self.robot}: '
                               f'{len(self.conditions)} conditions, {len(contact_links)} contact sensors, '
                               f'run dir {self.run_dir}')

    # ------------------------------------------------------------ inputs
    def on_robot_pose(self, msg):
        for t in msg.transforms:
            if t.child_frame_id == self.robot:
                self.base = tf_matrix(t.transform)

    def on_model_pose(self, msg):
        for t in msg.transforms:
            if t.header.frame_id == self.world:
                self.model_poses[t.child_frame_id] = tf_matrix(t.transform)
            else:      # ros_gz turns the scoped name model::link into model/link
                link = t.child_frame_id.removeprefix(t.header.frame_id + '/')
                self.link_poses[(t.header.frame_id, link)] = tf_matrix(t.transform)

    def on_joints(self, model, msg):
        for n, q in zip(msg.name, msg.position):
            self.joints[(model, n)] = q

    def on_contacts(self, topic, msg):
        pairs = {}
        for c in msg.contacts:
            k = (c.collision1.name, c.collision2.name)
            pairs[k] = max(pairs.get(k, 0.0), max(c.depths, default=0.0))
        self.contacts[topic] = (stamp_s(msg.header.stamp), [(a, b, d) for (a, b), d in pairs.items()])

    def on_command(self, msg):
        if not finite(msg):
            with self.lock:
                self.bad_commands += 1

    def read_gz_log(self):
        if not self.gz_log_path:
            found = sorted(glob.glob(os.path.expanduser('~/.gz/sim/log/*/server_console.log')), key=os.path.getmtime)
            if not found:
                return
            self.gz_log_path = found[-1]
        try:
            with open(self.gz_log_path, errors='replace') as f:
                f.seek(self.gz_offset)
                text = f.read()
        except OSError:
            return
        lines = text.split('\n')
        complete, rest = lines[:-1], lines[-1]
        self.gz_offset += len(text.encode()) - len(rest.encode())
        new = [ln for ln in complete if GZ_PROBLEM.search(ln)]
        if new:
            with self.lock:
                self.gz_lines += new

    def refresh_frames(self):
        frames = set(yaml.safe_load(self.tf.all_frames_as_yaml()) or {}) | {self.root}
        self.tf_links = [lk for lk in self.metrics.robot_links if lk in frames]

    # ------------------------------------------------------------ evaluation
    def snapshot(self, t):
        poses = {}
        if self.base is not None:
            for lk in self.tf_links:
                try:
                    tr = self.tf.lookup_transform(self.root, lk, rclpy.time.Time())
                except Exception:          # tf2 raises several types
                    continue
                poses[(self.robot, lk)] = self.base @ tf_matrix(tr.transform)
        for (m, lk), T in list(self.link_poses.items()):
            if m in self.model_poses:
                poses[(m, lk)] = self.model_poses[m] @ T
        contacts = [c for st, cs in list(self.contacts.values()) if st >= t - CONTACT_WINDOW_S for c in cs]
        with self.lock:
            c0, g0 = self.counted_from
            return Snapshot(t, poses, dict(self.joints), contacts, self.bad_commands - c0, self.gz_lines[g0:])

    def tick(self):
        t = self.get_clock().now().nanoseconds * 1e-9
        if t <= 0.0:
            return
        w0 = time.perf_counter()
        snap = self.snapshot(t)
        if self.metrics.robot_links and not any(k[0] == self.robot for k in snap.poses):
            return
        self.metrics.record(snap)
        self.snap = snap
        ep = self.episode
        active = {k for k, _ in ep.keys()} if ep is not None else set()
        values = {}
        for k, c in self.conditions:
            if c.metric == 'slip' and k not in active:
                continue                  # slip is relative to the active phase's start
            values[k] = self.metrics.value(c, snap, k)
        self.values = values
        self.ready = all(math.isfinite(values.get(k, 0.0)) for k, c in self.conditions if c.metric != 'peak_body_speed')
        if ep is None:
            if self.autostart and self.ready and self.n_episodes == 0:
                self.start(t)
            return
        phase = ep.i
        result = ep.step(t, values)
        self.ticks += 1
        self.tick_wall.append(time.perf_counter() - w0)
        if ep.i != phase:
            self.metrics.reset_phase()
        if result is not None:
            self.finish(t, result)

    def start(self, t):
        self.metrics.reset_phase()
        self.episode = Episode(self.res.task, self.res.sanity, t)
        self.t_start, self.ticks, self.tick_wall = t, 0, []
        with self.lock:
            self.counted_from = (self.bad_commands, len(self.gz_lines))
        self.bag_call('resume')
        self.get_logger().info(f'episode {self.n_episodes} started at t={t:.3f}')

    def finish(self, t, result):
        n = self.n_episodes
        dur = t - self.t_start
        out = {'task': self.res.task.name, 'scene': self.world, 'robot': self.robot, 'episode': n,
               'sim_time_start': round(self.t_start, 4), **result,
               'rate_hz': round(self.ticks / dur, 2) if dur > 0 else None,
               'tick_ms': {'mean': round(1e3 * float(np.mean(self.tick_wall)), 3),
                           'max': round(1e3 * float(np.max(self.tick_wall)), 3)} if self.tick_wall else None,
               # the contacts at the end (not with the ground), for a contact / penetration failure
               'contacts': sorted({(a, b, round(d, 5)) for a, b, d in self.snap.contacts
                                   if not any(x.startswith(GROUND + '::') for x in (a, b))}),
               'gz_log': self.gz_log_path or None,
               'result_file': str(self.run_dir / f'episode_{n}.json'),
               'bag': next(iter(sorted(glob.glob(str(self.run_dir / 'bag' / f'{n}_*.mcap')))), None) if self.bag
               else None}
        Path(out['result_file']).write_text(json.dumps(out, indent=1))
        self.result_pub.publish(String(data=json.dumps(out)))
        self.bag_call('pause')
        self.bag_call('split_bagfile')
        self.episode = None
        self.n_episodes += 1
        self.get_logger().info(f'episode {n}: {"passed" if result["passed"] else "failed"} '
                               f'{result["failed_phase"] or ""} {result["reason"] or ""} {result["check"] or ""} '
                               f'at t={result["duration_s"]} s -> {out["result_file"]}')

    def on_start(self, req, res):
        if self.episode is not None:
            res.success, res.message = False, f'episode {self.n_episodes} is running'
        elif not self.ready:
            res.success, res.message = False, 'inputs not ready: ' + ', '.join(
                k for k, v in self.values.items() if not math.isfinite(v)) if self.values else 'no robot pose yet'
        else:
            self.start(self.get_clock().now().nanoseconds * 1e-9)
            res.success, res.message = True, f'episode {self.n_episodes}'
        return res

    def bag_call(self, name):
        if name in self.bag_clients:
            cli, typ = self.bag_clients[name]
            if cli.service_is_ready():
                cli.call_async(typ.Request())
            else:
                self.get_logger().warning(f'rosbag2 recorder: no {name} service')

    def publish_state(self):
        ep = self.episode
        state = {'t': round(self.get_clock().now().nanoseconds * 1e-9, 4), 'ready': self.ready,
                 'episode': self.n_episodes if ep is not None else None, 'phase': ep.phase if ep else None,
                 'values': {k: (round(v, 6) if math.isfinite(v) else None) for k, v in self.values.items()}}
        self.state_pub.publish(String(data=json.dumps(state)))


def main():
    rclpy.init()
    node = Evaluator()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
