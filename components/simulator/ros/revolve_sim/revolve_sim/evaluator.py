"""The task evaluator: a ROS 2 node on sim time that evaluates a task's phases and sanity checks live
(revolve_sim.episode, revolve_sim.metrics), fed only by upstream publishers:
  robot links      the ground-truth base pose (gz PosePublisher, /model/<robot>/pose) x robot_state_publisher tf
  scene links      gz PosePublisher of every scene model (/model/<m>/pose: model in the world, links in the model)
  scene joints     gz JointStatePublisher (/world/<w>/model/<m>/joint_state)
  contacts         gz Contact sensors on the robot links (/world/<w>/model/<robot>/link/<l>/sensor/contact/contact)
  commands         the controllers' command topics (<omni>/cmd_vel, <jtc>/joint_trajectory): non-finite messages
  gz log           the gz server's console log (~/.gz/sim/log/<time>/server_console.log in the container): [error] /
                   [critical] lines
An episode counts the non-finite commands and gz log lines since its start (bring-up errors are not its own).
Poses and contacts are taken at one common stamp per tick, the oldest of the newest gz pose and tf data (tf
interpolated, gz poses from a short history, contacts within CONTACT_WINDOW_S before it): the inputs reach this node
with different delays (under load its callback queues fall behind by tens of ms), and a moving door or arm would
otherwise show slip / speed / lost contacts that are only these time offsets.

Episodes: ~/start (std_srvs/Trigger) starts one (autostart: the first one starts as soon as every input is live).
Each episode is recorded into its own bag, <run_dir>/episode_<n>_bag/ (the rosbag2 recorder of the launch,
rosbag2_recorder: its record service with that uri at the start, stop at the end; a fresh recording subscribes anew,
so it gets the transient-local /tf_static and /robot_description). At the end the result (JSON) is published on
~/result (std_msgs/String, transient local), the bag is closed (metadata.yaml written), then the result is written to
<run_dir>/episode_<n>.json: once that file exists, the episode's files are complete (with the gz log copied first to
<run_dir>/episode_<n>_gz.log: the log itself is in the container). Episode numbers continue after the highest n of
any episode_<n> file already in run_dir, a bag without a result too (a relaunch into the same run_dir after a killed
episode overwrites nothing; an existing bag dir is never recorded into). ~/state (std_msgs/String JSON, 10 Hz): the
sim time, the active phase and every condition's current value (also between episodes; null when not computable,
e.g. a slip condition outside its phase).
A world reset (/simulator/reset, revolve_sim/reset.py: sim time jumps back) re-arms it: a running episode ends with
reason `reset`, every input is dropped for SETTLE_S, then it is ready again once its inputs are live (~/start).
~/state `resets` counts the resets it has seen.

  ros2 run revolve_sim evaluator --ros-args -p task:=<t> -p scene:=<s> -p robot:=<r> [-p run_dir:=DIR]
"""
import glob
import json
import math
import os
import re
import shutil
import threading
import time
from collections import deque
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

from revolve_sim.episode import Episode, next_episode
from revolve_sim.metrics import GROUND, Metrics, Snapshot, collision_geometry, pose_matrix
from revolve_sim.robot import command_topics, describe, robot_dir
from revolve_sim.world import make_world

HISTORY = 200               # gz pose messages kept per model (100 Hz: 2 s)
SYNC_MAX_AGE_S = 0.5        # an input older than this (a stalled publisher) does not hold the common stamp back
CONTACT_WINDOW_S = 0.05     # a sensor's contacts count until this long after its message (it publishes at 100 Hz)
SETTLE_S = 0.5             # sim time after a reset during which inputs (also late ones from before it) are dropped
BAG_CALL_S = 10.0           # wall timeout of a rosbag2 recorder service call (stop: flushes and closes the bag)
BAG_RESULT_S = 0.1          # wall time between publishing the result and stopping the recorder
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
        self.res = specs.resolve(repo / 'specs', p['task'], robot=p['robot'], scene=scene)   # on this scene (instance)
        robot, urdf, _, contact_links = describe(repo, p['robot'], arm=scene.robot_start.arm)
        self.robot, self.world = robot.name, scene.name
        self.root, links = urdf_links(urdf)
        collisions = {m: collision_geometry(info.sdf) for m, info in self.res.models.items() if info.sdf}
        self.metrics_args = (self.res, links, collisions)
        self.metrics = Metrics(*self.metrics_args)
        self.run_dir = Path(p['run_dir'] or repo / 'runs' / 'sim' / f'{time.strftime("%Y%m%d_%H%M%S")}_{p["task"]}')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.autostart, self.bag, self.gz_log_path = p['autostart'], p['bag'], p['gz_log']
        self.conditions = [(k, c) for ph in self.res.task.phases for k, c in Episode.phase_keys(ph)] + \
            [(f'sanity.{k}', c) for k, c in self.res.sanity.items()]

        self.lock = threading.Lock()
        self.base = deque(maxlen=HISTORY)   # [(stamp, world <- robot root link)]
        self.scene_poses = {}               # model -> deque [(stamp, {None: world <- model, link: model <- link})]
        self.joints = {}
        self.contacts = {}                  # topic -> (stamp, [(c1, c2, depth)])
        self.bad_commands, self.gz_lines, self.gz_offset = 0, [], 0
        self.counted_from = (0, 0)          # (commands, gz lines) before the episode: it counts only its own
        self.gz_start = 0                   # the gz log's byte offset at the episode's start: its copy starts there
        self.episode, self.n_episodes, self.ticks = None, next_episode(self.run_dir), 0
        self.started, self.bag_path = False, None
        self.values, self.ready = {}, False
        self.tf_links = []                  # the robot links tf has (not the passive rollers: no joint states)
        self.sync_link = robot.frames.get('tcp', links[-1])   # its tf stamp: the newest of the arm's joint states
        self.tick_wall = []                 # wall time of the ticks of the running episode [s]
        self.last_t, self.settle_until, self.resets = 0.0, None, 0   # the reset detection (sim time jumping back)

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
        self.bag_clients, self.bag_idle = {}, False
        if self.bag:
            from rosbag2_interfaces.srv import Record, Resume, Stop
            self.bag_clients = {k: (self.create_client(t, f'/rosbag2_recorder/{k}'), t)
                                for k, t in (('record', Record), ('resume', Resume), ('stop', Stop))}
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
                self.base.append((stamp_s(t.header.stamp), tf_matrix(t.transform)))

    def on_model_pose(self, msg):
        poses, model = {}, None
        for t in msg.transforms:
            if t.header.frame_id == self.world:
                model = t.child_frame_id
                poses[None] = tf_matrix(t.transform)
            else:      # ros_gz turns the scoped name model::link into model/link
                poses[t.child_frame_id.removeprefix(t.header.frame_id + '/')] = tf_matrix(t.transform)
        if model is not None and msg.transforms:
            self.scene_poses.setdefault(model, deque(maxlen=HISTORY)).append(
                (stamp_s(msg.transforms[0].header.stamp), poses))

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
    @staticmethod
    def at(history, stamp):
        """The newest value of a [(stamp, value)] history not after stamp (else the oldest)."""
        h = list(history)
        return next((v for st, v in reversed(h) if st <= stamp), h[0][1])

    def snapshot(self, t):
        poses, stamps = {}, {}
        newest = [h[-1][0] for h in (self.base, *list(self.scene_poses.values())) if h]
        s = min((st for st in newest if st >= t - SYNC_MAX_AGE_S), default=t)
        if self.base and self.tf_links:
            try:
                s = min(s, stamp_s(self.tf.lookup_transform(self.root, self.sync_link, rclpy.time.Time()).header.stamp)
                        or s)
            except Exception:          # tf2 raises several types
                pass
            base = self.at(self.base, s)
            for lk in self.tf_links:
                try:
                    tr = self.tf.lookup_transform(self.root, lk, rclpy.time.Time(nanoseconds=int(s * 1e9)))
                except Exception:
                    try:
                        tr = self.tf.lookup_transform(self.root, lk, rclpy.time.Time())
                    except Exception:
                        continue
                poses[(self.robot, lk)] = base @ tf_matrix(tr.transform)
                stamps[(self.robot, lk)] = stamp_s(tr.header.stamp) or s     # a static chain: stamp 0
        for m, h in list(self.scene_poses.items()):
            ps = self.at(h, s)
            for lk, T in ps.items():
                if lk is not None:
                    poses[(m, lk)] = ps[None] @ T
                    stamps[(m, lk)] = s
        contacts = [c for st, cs in list(self.contacts.values()) if st >= s - CONTACT_WINDOW_S for c in cs]
        with self.lock:
            c0, g0 = self.counted_from
            return Snapshot(t, poses, dict(self.joints), contacts, self.bad_commands - c0, self.gz_lines[g0:],
                            stamps)

    def check_reset(self, t):
        """A world reset (sim time jumped back): end a running episode (reason `reset`), then drop every input until
        SETTLE_S after the reset. -> True while settling."""
        if t < self.last_t:
            if self.episode is not None:
                self.finish(self.last_t, self.episode._end(self.last_t, False, self.episode.phase, 'reset'))
            self.settle_until, self.resets = t + SETTLE_S, self.resets + 1
            self.tfl.unregister()   # tf2 rejects data older than its newest by the cache time: a new buffer after
            self.get_logger().info(f'world reset: sim time {self.last_t:.3f} -> {t:.3f}')
        self.last_t = t
        if self.settle_until is None:
            return False
        self.base.clear()
        self.scene_poses.clear()
        self.joints.clear()
        self.contacts.clear()
        self.values, self.ready = {}, False
        if t >= self.settle_until:
            self.settle_until = None
            self.metrics = Metrics(*self.metrics_args)
            self.tf = Buffer()      # (Buffer.clear() also drops the static transforms, sent once)
            self.tfl = TransformListener(self.tf, self, spin_thread=False)
        return True

    def tick(self):
        t = self.get_clock().now().nanoseconds * 1e-9
        if t <= 0.0 or self.check_reset(t):
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
            if self.autostart and self.ready and not self.started:
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
        self.gz_start = self.gz_offset      # read_gz_log runs in this callback group: not meanwhile
        self.started = True
        self.bag_path = None
        self.n_episodes = max(self.n_episodes, next_episode(self.run_dir))   # files another writer left meanwhile
        if self.bag:
            if not self.bag_idle:    # the launch's recording (paused, a scratch bag): stopped, unpaused for good
                self.bag_idle = self.bag_call('stop') is not None and self.bag_call('resume') is not None
            uri = self.run_dir / f'episode_{self.n_episodes}_bag'
            if uri.exists():    # rosbag2 would record into "<uri>(1)" and still answer success: never ask it to
                r, why = None, f'{uri} exists'
            else:
                r = self.bag_call('record', uri=str(uri))
                why = r.error_string if r else 'no reply'
            if r is not None and r.return_code == 0:
                self.bag_path = str(uri)
            else:
                self.get_logger().error(f'rosbag2 recorder: episode not recorded ({why})')
        self.get_logger().info(f'episode {self.n_episodes} started at t={t:.3f}')

    def gz_log_copy(self, n):
        """The episode's part of the gz server's console log (from its start to now) into the run dir,
        <run_dir>/episode_<n>_gz.log (the log itself is in the container) -> its path, None without a log."""
        if not self.gz_log_path:
            return None
        dst = self.run_dir / f'episode_{n}_gz.log'
        try:
            with open(self.gz_log_path, 'rb') as src, open(dst, 'wb') as out:
                src.seek(self.gz_start)
                shutil.copyfileobj(src, out)
        except OSError as e:
            self.get_logger().warning(f'gz log not copied: {e}')
            return None
        return str(dst)

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
               'gz_log': self.gz_log_copy(n),
               'result_file': str(self.run_dir / f'episode_{n}.json'),
               'bag': self.bag_path}
        self.result_pub.publish(String(data=json.dumps(out)))
        if self.bag_path:
            time.sleep(BAG_RESULT_S)        # the recorder gets the result before it stops
            self.bag_call('stop')
        Path(out['result_file']).write_text(json.dumps(out, indent=1))
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

    def bag_call(self, name, **req):
        """Call a rosbag2 recorder service and wait for its reply (the clients are in the default callback group,
        served by another executor thread) -> the response, None on no service / timeout."""
        cli, typ = self.bag_clients[name]
        if not cli.wait_for_service(timeout_sec=BAG_CALL_S):
            self.get_logger().warning(f'rosbag2 recorder: no {name} service')
            return None
        f = cli.call_async(typ.Request(**req))
        w0 = time.monotonic()
        while not f.done() and time.monotonic() - w0 < BAG_CALL_S:
            time.sleep(0.005)
        if not f.done():
            self.get_logger().warning(f'rosbag2 recorder: {name} timed out')
            return None
        return f.result()

    def publish_state(self):
        t = self.get_clock().now().nanoseconds * 1e-9
        if t > 0.0:
            self.check_reset(t)
        ep = self.episode
        # every condition, null when not computable (e.g. a slip outside its phase, or before the inputs are live)
        values = {k: self.values.get(k) for k, _ in self.conditions}
        state = {'t': round(t, 4), 'ready': self.ready, 'resets': self.resets,
                 'episode': self.n_episodes if ep is not None else None, 'phase': ep.phase if ep else None,
                 'values': {k: (round(v, 6) if v is not None and math.isfinite(v) else None) for k, v in values.items()}}
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
