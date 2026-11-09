"""The oracle policy on ROS 2: publishes a recorded episode's actions on the contract's action topic,
paced on /clock, where Rosetta's policy runner would publish a policy's actions. Runs in the rollout's
client container (revolve-dataset: ROS 2 Jazzy + Rosetta, the one the Rosetta runner uses), so its
actions take the same DDS path across the rollout's internal network to the sim's ros_relay.py.
Ported from the earlier pipeline's dummy_policy.py: the relay's ready file is read from
the sim's dir and the ready file written to the client's dir (two containers); frames() and
contract_io() are copied in (their modules import gz-transport, which this image has not).

  dummy_policy.py <replay dir> <sim out dir> <client out dir> [ready file name] [contract.yaml]

Action k = action_player.frames (the dataset frame k of that replay), sent once as
sensor_msgs/JointState (the contract's action joint names, stamp = the /clock value) on the first
/clock at or after k x int(1e9 / fps) ns. Touches <client>/<ready file> (default runner_ready) once
<sim>/relay_ready exists, /clock has a publisher and the action topic a subscriber; writes
<client>/dummy_policy.json after the last frame or on SIGTERM.
"""
import json
import os
import signal
import sys
import threading
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState

CI, JS = "sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/JointState"


def frames(replay, fps=30):
    """(rows, actions [n, 11]) of a replay's state log on the 1 / fps grid from its start (action_player.py)."""
    s = np.load(Path(replay) / "state_log.npz")
    dt_ns, tick_ns = round(float(s["control_dt"]) * 1e9), int(1e9 / fps)
    n = (len(s["time_s"]) - 1) * dt_ns // tick_ns + 1
    rows = np.arange(n) * tick_ns // dt_ns
    return rows, np.hstack([s["action_base"], s["action_arm"], s["action_grip"][:, None]])[rows]


def contract_io(path):
    """(image topics, state topic, action topic, action joint names) from a Rosetta contract (ros_relay.py)."""
    c = yaml.safe_load(Path(path).read_text())
    obs = [v["channel"] for v in c["observations"].values()]
    images = [o["topic"] for o in obs if o["type"] == CI]
    state = [o["topic"] for o in obs if o["type"] == JS]
    (act,) = c["actions"].values()
    names = [s.removeprefix("position.") for s in act["select"]]
    return images, state[0], act["channel"]["topic"], names


def main(replay, sim_out, out, ready="runner_ready", contract="specs/robots/mobile_panda/contract.yaml"):
    sim_out, out = Path(sim_out), Path(out)
    fps = yaml.safe_load(Path(contract).read_text())["fps"]
    _, _, action_topic, names = contract_io(contract)
    rows, A = frames(replay, fps)
    tick_ns = int(1e9 / fps)
    rclpy.init()
    node = rclpy.create_node("dummy_policy", parameter_overrides=[Parameter("use_sim_time", value=True)])
    pub = node.create_publisher(JointState, action_topic, QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
    sent, k, done = [], [0], threading.Event()

    def on_clock(msg):
        t_ns = msg.clock.sec * 10**9 + msg.clock.nanosec
        while k[0] < len(A) and k[0] * tick_ns <= t_ns:
            js = JointState(name=names, position=[float(x) for x in A[k[0]]])
            js.header.stamp = msg.clock
            pub.publish(js)
            sent.append((k[0], t_ns))
            k[0] += 1
        if k[0] == len(A):
            done.set()

    node.create_subscription(Clock, "/clock", on_clock, QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE))
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: done.set())
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    while not ((sim_out / "relay_ready").exists() and node.count_publishers("/clock") and pub.get_subscription_count()) \
            and not done.wait(0.05):
        pass
    (out / ready).touch()
    print(f"dummy policy: {len(A)} frames at {fps} Hz on {action_topic} from {replay}; relay connected", flush=True)
    done.wait()
    late = [t - kk * tick_ns for kk, t in sent]
    (out / "dummy_policy.json").write_text(json.dumps(dict(
        replay=str(replay), fps=fps, frames=len(A), sent=len(sent), action_topic=action_topic, names=names,
        send_clock_ns=[t for _, t in sent],
        lateness_ms=dict(max=max(late) / 1e6 if late else None, mean=sum(late) / len(late) / 1e6 if late else None))))
    print(f"dummy policy: sent {len(sent)} of {len(A)} frames", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main(*sys.argv[1:])
