"""ROS 2 side of the closed loop, beside gz and outside its 1 kHz loop: gz-transport <-> ROS 2 per the
robot's Rosetta contract. Runs in revolve-simulator-ros (components/simulator/ros/Dockerfile) with ROS 2
Jazzy sourced for this process only (cli.py rollout starts it beside gz). Ported from the earlier
pipeline's ros_relay.py; the topics and joint names come from the robot's contract (revolve_common.policy_io
layout: observation.state = Odometry twist + JointState, action = Twist + JointState), the state log's joint
order from <out>/closed_loop.json (state_names).

  ros_relay.py <out dir> <contract.yaml> <camera model dir>

gz -> ROS 2
  /closed_loop/base_twist (closed_loop_system, every 1 ms step, before the state): the base pose in the world and
      the measured twist in base_link (policy_io.measured_twist) -> each Odometry observation source topic (/odom).
  /closed_loop/state (closed_loop_system, every 1 ms step) -> each JointState observation source topic
      (/joint_states: every robot joint, positions and velocities), then /clock (rosgraph_msgs/Clock); all carry
      the step's sim time.
  /<model>/<camera>/image (the camera sensors in <model dir>/model.sdf, raw RGB) ->
      /cameras/<camera without _cam>/image/compressed (sensor_msgs/CompressedImage, Pillow JPEG q90,
      frame_id = the camera, stamp = the gz render time), as bag.py records them. Every
      CompressedImage topic in the contract must have a camera.
ROS 2 -> gz
  the contract's action sources (/cmd_vel geometry_msgs/Twist: the base twist; /action sensor_msgs/JointState:
      the arm targets and the gripper, by joint name, by position when the message has no names) ->
      /closed_loop/action (gz.msgs.Actuators, the sources concatenated in contract order = the policy action).
      Rosetta publishes a frame's sources back to back in contract order, so a frame goes out when its last
      source arrives, with the newest of the others. Messages with missing joints or non-finite values, and a
      last source before every other source has arrived, are dropped and counted.
Touches <out>/relay_ready once closed_loop_system listens on /closed_loop/action; writes
<out>/relay.json (message counts, JPEG encode time) on SIGTERM or SIGINT. With $CL_FRAMES_DIR set, each
published JPEG is also saved as <dir>/<camera>/<stamp ns>.jpg (the eval's video.mp4).
"""
import io
import json
import math
import os
import signal
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rclpy
import yaml
from builtin_interfaces.msg import Time
from gz.msgs.actuators_pb2 import Actuators
from gz.msgs.image_pb2 import Image
from geometry_msgs.msg import Twist
from gz.transport import Node
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from rclpy.qos import QoSProfile, ReliabilityPolicy
from revolve_common import policy_io as P
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CompressedImage, JointState

ACTION_TOPIC, STATE_TOPIC, TWIST_TOPIC = "/closed_loop/action", "/closed_loop/state", "/closed_loop/base_twist"
JPEG_QUALITY = 90
FRAMES_DIR = os.environ.get("CL_FRAMES_DIR")


def contract_io(path):
    """(image topics, observation.state sources, action sources) of a Rosetta contract (policy_io.Source)."""
    c = yaml.safe_load(Path(path).read_text())
    images = [s.topic for k, v in c["observations"].items() if k != "observation.state" for s in P.sources(v)
              if s.type == P.CI]
    return images, P.state_sources(c), P.action_sources(c)


def decode(src, msg):
    """One action source's values in its names' order, or None (missing joints)."""
    if src.type == P.TWIST:
        return [msg.linear.x, msg.linear.y, msg.angular.z]
    if not msg.name:
        return list(msg.position)
    idx = {n: i for i, n in enumerate(msg.name)}
    if not all(n in idx and idx[n] < len(msg.position) for n in src.names):
        return None
    return [msg.position[idx[n]] for n in src.names]


def cameras(model_dir):
    root = ET.parse(Path(model_dir) / "model.sdf").getroot()
    return {s.get("name"): s.findtext("topic") for s in root.iter("sensor") if s.get("type") == "camera"}


def image_topic(camera):
    return f"/cameras/{camera.removesuffix('_cam')}/image/compressed"     # as bag.py


def stamp(sec, nsec):
    return Time(sec=int(sec), nanosec=int(nsec))


def main(out, contract, model_dir):
    out = Path(out)
    img_topics, state_srcs, action_srcs = contract_io(contract)
    state_names = json.loads((out / "closed_loop.json").read_text())["state_names"]   # the state log's joints
    cams = cameras(model_dir)
    by_topic = {image_topic(c): c for c in cams}
    missing = [t for t in img_topics if t not in by_topic]
    if missing:
        raise SystemExit(f"no camera for contract image topics {missing} in {model_dir}/model.sdf")
    if FRAMES_DIR:
        for c in by_topic.values():
            (Path(FRAMES_DIR) / c).mkdir(parents=True, exist_ok=True)
    rclpy.init()
    ros = rclpy.create_node("closed_loop_relay")
    rel = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    js_pubs = [ros.create_publisher(JointState, s.topic, rel) for s in state_srcs if s.type == P.JS]
    odom_pubs = [ros.create_publisher(Odometry, s.topic, rel) for s in state_srcs if s.type == P.ODOM]
    clock_pub = ros.create_publisher(Clock, "/clock", rel)
    img_pub = {t: ros.create_publisher(CompressedImage, t, rel) for t in img_topics}
    gz = Node()
    act_pub = gz.advertise(ACTION_TOPIC, Actuators)
    stats = dict(state=0, twist=0, images={t: 0 for t in img_topics}, encode_s=[0.0, 0.0], actions=0,
                 actions_dropped=0, first_state_ns=None, last_state_ns=None)
    lock = threading.Lock()

    def on_state(msg):
        js = JointState()
        js.header.stamp = stamp(msg.header.stamp.sec, msg.header.stamp.nsec)
        js.name = state_names
        js.position = list(msg.position)
        js.velocity = list(msg.velocity)
        for pub in js_pubs:
            pub.publish(js)
        clock_pub.publish(Clock(clock=js.header.stamp))
        ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nsec
        stats["state"] += 1
        stats["first_state_ns"] = stats["first_state_ns"] if stats["first_state_ns"] is not None else ns
        stats["last_state_ns"] = ns

    def on_twist(msg):
        (x, y, yaw), (vx, vy, wz) = msg.position, msg.velocity
        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = stamp(msg.header.stamp.sec, msg.header.stamp.nsec), \
            "world", "base_link"
        od.pose.pose.position.x, od.pose.pose.position.y = x, y
        od.pose.pose.orientation.z, od.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        od.twist.twist.linear.x, od.twist.twist.linear.y, od.twist.twist.angular.z = vx, vy, wz
        for pub in odom_pubs:
            pub.publish(od)
        stats["twist"] += 1

    pools = {}

    def on_image(topic, camera):
        pub, pool = img_pub[topic], pools.setdefault(topic, ThreadPoolExecutor(1))   # one per camera: in order

        def encode(sec, nsec, w, h, data):
            t0 = time.perf_counter()
            b = io.BytesIO()
            PILImage.frombuffer("RGB", (w, h), data, "raw", "RGB", 0, 1).save(b, "JPEG", quality=JPEG_QUALITY)
            jpg = b.getvalue()
            m = CompressedImage(format="jpeg", data=jpg)
            m.header.stamp, m.header.frame_id = stamp(sec, nsec), camera
            pub.publish(m)
            if FRAMES_DIR:
                (Path(FRAMES_DIR) / camera / f"{int(sec) * 10**9 + int(nsec):013d}.jpg").write_bytes(jpg)
            dt = time.perf_counter() - t0
            with lock:
                stats["images"][topic] += 1
                stats["encode_s"][0] += dt
                stats["encode_s"][1] = max(stats["encode_s"][1], dt)

        def cb(msg):
            if len(msg.data) == msg.width * msg.height * 3:
                pool.submit(encode, msg.header.stamp.sec, msg.header.stamp.nsec, msg.width, msg.height,
                            bytes(msg.data))
        return cb

    latest = [None] * len(action_srcs)

    def on_action(i):
        src = action_srcs[i]

        def cb(msg):
            v = decode(src, msg)
            if v is None or len(v) != len(src.names) or not all(math.isfinite(x) for x in v):
                stats["actions_dropped"] += 1
                return
            latest[i] = v
            if i < len(action_srcs) - 1:
                return
            if any(x is None for x in latest):
                stats["actions_dropped"] += 1
                return
            m = Actuators()
            m.position.extend([x for part in latest for x in part])
            act_pub.publish(m)
            stats["actions"] += 1
        return cb

    if not gz.subscribe(Actuators, STATE_TOPIC, on_state):
        raise SystemExit(f"cannot subscribe to {STATE_TOPIC}")
    if odom_pubs and not gz.subscribe(Actuators, TWIST_TOPIC, on_twist):
        raise SystemExit(f"cannot subscribe to {TWIST_TOPIC}")
    for t in img_topics:
        if not gz.subscribe(Image, cams[by_topic[t]], on_image(t, by_topic[t])):
            raise SystemExit(f"cannot subscribe to {cams[by_topic[t]]}")
    for i, src in enumerate(action_srcs):
        ros.create_subscription(Twist if src.type == P.TWIST else JointState, src.topic, on_action(i),
                                QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    spin = threading.Thread(target=rclpy.spin, args=(ros,), daemon=True)
    spin.start()
    while not act_pub.has_connections() and not stop.wait(0.05):
        pass
    (out / "relay_ready").touch()
    print(f"relay: {[s.topic for s in state_srcs]}, /clock, {len(img_topics)} images, "
          f"{[s.topic for s in action_srcs]} -> {ACTION_TOPIC}; closed_loop_system connected", flush=True)
    stop.wait()
    for p in pools.values():
        p.shutdown(wait=True)
    n = sum(stats["images"].values())
    (out / "relay.json").write_text(json.dumps(dict(
        contract=contract, cameras={t: by_topic[t] for t in img_topics}, state_msgs=stats["state"],
        twist_msgs=stats["twist"],
        state_span_s=((stats["last_state_ns"] - stats["first_state_ns"]) / 1e9) if stats["state"] else None,
        images=stats["images"], jpeg_encode_ms=dict(mean=stats["encode_s"][0] / n * 1000 if n else None,
                                                  max=stats["encode_s"][1] * 1000),
        actions=stats["actions"], actions_dropped=stats["actions_dropped"]), indent=1))
    print(f"relay: {stats['state']} states, images {stats['images']}, {stats['actions']} actions "
          f"({stats['actions_dropped']} dropped)", flush=True)
    os._exit(0)                    # skip gz-transport's and rclpy's thread teardown


if __name__ == "__main__":
    main(*sys.argv[1:])
