"""ROS 2 side of the closed loop, beside gz and outside its 1 kHz loop: gz-transport <-> ROS 2 per the
robot's Rosetta contract. Runs in revolve-simulator-ros (components/simulator/ros/Dockerfile) with ROS 2
Jazzy sourced for this process only (cli.py rollout starts it beside gz). Ported unchanged from
the earlier pipeline's ros_relay.py but for the default paths (specs/, runs/models/).

  ros_relay.py <out dir> [contract.yaml] [camera model dir]

gz -> ROS 2
  /closed_loop/state (closed_loop_system, every 1 ms step) -> the contract's JointState observation
      topic (/joint_states: the 12 robot joints, positions and velocities), then /clock
      (rosgraph_msgs/Clock); both carry the step's sim time.
  /mobile_panda/<camera>/image (the camera sensors in <model dir>/model.sdf, raw RGB) ->
      /cameras/<camera without _cam>/image/compressed (sensor_msgs/CompressedImage, Pillow JPEG q90,
      frame_id = the camera, stamp = the gz render time), as bag.py records them. Every
      CompressedImage topic in the contract must have a camera.
ROS 2 -> gz
  the contract's action topic (/action, sensor_msgs/JointState) -> /closed_loop/action
      (gz.msgs.Actuators, the contract's select order, by joint name; by position when the message
      has no names). Messages with missing joints or non-finite values are dropped and counted.
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
from gz.transport import Node
from PIL import Image as PILImage
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CompressedImage, JointState

ACTION_TOPIC, STATE_TOPIC = "/closed_loop/action", "/closed_loop/state"
# closed_loop_system's state order (mobile_panda.Controller.names)
STATE_NAMES = ["base_x", "base_y", "base_yaw"] + [f"panda_joint{i}" for i in range(1, 8)] + [
    "panda_finger_joint_left", "panda_finger_joint_right"]
JPEG_QUALITY = 90
FRAMES_DIR = os.environ.get("CL_FRAMES_DIR")
CI, JS = "sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/JointState"


def contract_io(path):
    """(image topics, state topic, action topic, action joint names) from a Rosetta contract."""
    c = yaml.safe_load(Path(path).read_text())
    obs = [v["channel"] for v in c["observations"].values()]
    images = [o["topic"] for o in obs if o["type"] == CI]
    state = [o["topic"] for o in obs if o["type"] == JS]
    (act,) = c["actions"].values()
    names = [s.removeprefix("position.") for s in act["select"]]
    return images, state[0], act["channel"]["topic"], names


def cameras(model_dir):
    root = ET.parse(Path(model_dir) / "model.sdf").getroot()
    return {s.get("name"): s.findtext("topic") for s in root.iter("sensor") if s.get("type") == "camera"}


def image_topic(camera):
    return f"/cameras/{camera.removesuffix('_cam')}/image/compressed"     # as bag.py


def stamp(sec, nsec):
    return Time(sec=int(sec), nanosec=int(nsec))


def main(out, contract="specs/robots/mobile_panda/contract.yaml", model_dir="runs/models/mobile_panda"):
    out = Path(out)
    img_topics, state_topic, action_topic, action_names = contract_io(contract)
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
    js_pub = ros.create_publisher(JointState, state_topic, rel)
    clock_pub = ros.create_publisher(Clock, "/clock", rel)
    img_pub = {t: ros.create_publisher(CompressedImage, t, rel) for t in img_topics}
    gz = Node()
    act_pub = gz.advertise(ACTION_TOPIC, Actuators)
    stats = dict(state=0, images={t: 0 for t in img_topics}, encode_s=[0.0, 0.0], actions=0, actions_dropped=0,
                 first_state_ns=None, last_state_ns=None)
    lock = threading.Lock()

    def on_state(msg):
        js = JointState()
        js.header.stamp = stamp(msg.header.stamp.sec, msg.header.stamp.nsec)
        js.name = STATE_NAMES
        js.position = list(msg.position)
        js.velocity = list(msg.velocity)
        js_pub.publish(js)
        clock_pub.publish(Clock(clock=js.header.stamp))
        ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nsec
        stats["state"] += 1
        stats["first_state_ns"] = stats["first_state_ns"] if stats["first_state_ns"] is not None else ns
        stats["last_state_ns"] = ns

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

    def on_action(msg):
        if msg.name:
            idx = {n: i for i, n in enumerate(msg.name)}
            ok = all(n in idx and idx[n] < len(msg.position) for n in action_names)
            v = [msg.position[idx[n]] for n in action_names] if ok else []
        else:
            v = list(msg.position)
        if len(v) != len(action_names) or not all(math.isfinite(x) for x in v):
            stats["actions_dropped"] += 1
            return
        m = Actuators()
        m.position.extend(v)
        act_pub.publish(m)
        stats["actions"] += 1

    if not gz.subscribe(Actuators, STATE_TOPIC, on_state):
        raise SystemExit(f"cannot subscribe to {STATE_TOPIC}")
    for t in img_topics:
        if not gz.subscribe(Image, cams[by_topic[t]], on_image(t, by_topic[t])):
            raise SystemExit(f"cannot subscribe to {cams[by_topic[t]]}")
    ros.create_subscription(JointState, action_topic, on_action,
                            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    spin = threading.Thread(target=rclpy.spin, args=(ros,), daemon=True)
    spin.start()
    while not act_pub.has_connections() and not stop.wait(0.05):
        pass
    (out / "relay_ready").touch()
    print(f"relay: {state_topic}, /clock, {len(img_topics)} images, {action_topic} -> {ACTION_TOPIC}; "
          f"closed_loop_system connected", flush=True)
    stop.wait()
    for p in pools.values():
        p.shutdown(wait=True)
    n = sum(stats["images"].values())
    (out / "relay.json").write_text(json.dumps(dict(
        contract=contract, cameras={t: by_topic[t] for t in img_topics}, state_msgs=stats["state"],
        state_span_s=((stats["last_state_ns"] - stats["first_state_ns"]) / 1e9) if stats["state"] else None,
        images=stats["images"], jpeg_encode_ms=dict(mean=stats["encode_s"][0] / n * 1000 if n else None,
                                                  max=stats["encode_s"][1] * 1000),
        actions=stats["actions"], actions_dropped=stats["actions_dropped"]), indent=1))
    print(f"relay: {stats['state']} states, images {stats['images']}, {stats['actions']} actions "
          f"({stats['actions_dropped']} dropped)", flush=True)
    os._exit(0)                    # skip gz-transport's and rclpy's thread teardown


if __name__ == "__main__":
    main(*sys.argv[1:])
