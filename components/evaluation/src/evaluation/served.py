"""The policy client of a served checkpoint (`rollout_client --serve JSON`), in the sim's client container: the observation from
the sim's topics (3 cameras, joint states, odometry), the policy service's chunk, and its actions streamed at 30 Hz on sim time
as the oracle's are (the no-blend sync runner: a chunk is asked for when the previous one's `actions_per_chunk` actions have
been sent, the sim keeps running during the inference; the first action of a chunk is the first one sent).

`--serve` = {"url", "prompt", "robot", "actions_per_chunk", "image_size"}. The episode loop is the rollout client's `episode`, so a
`Live` wrapper (a live recorder) halts it exactly as it halts the oracle: `halt()` is polled before every inference and every action."""
from __future__ import annotations

import time

import numpy as np
from revolve_common.sim_client import RESULT_TIMEOUT_S
from revolve_common.sim_session import RESULT_TIMEOUT, SimFault

from .oracle_actions import ACTION_HZ, ActionSeries
from .policy_contract import CONTRACTS, Contract
from .policy_wire import PolicyClient

OBS_TIMEOUT_S = 30.0                    # wall: every observation topic has a message (the sim publishes from the start)


def decode(data: bytes, size: int | None = None) -> np.ndarray:
    """A compressed image (JPEG / PNG bytes) -> RGB uint8 (H, W, 3), resized to size x size when asked."""
    import cv2
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img if not size or img.shape[:2] == (size, size) else cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


class Observer:
    """The latest messages of the observation topics; `observe()` the policy's input at this moment. `feed_*` are the
    subscriptions' callbacks (tests call them); `wire(node)` subscribes."""

    def __init__(self, contract: Contract, image_size: int | None = None, decoder=decode):
        self.contract, self.size, self.decoder = contract, image_size, decoder
        self.frames: dict[str, bytes] = {}
        self.joints: dict[str, float] = {}
        self.twist: tuple[float, float, float] | None = None

    def feed_image(self, camera: str, data: bytes) -> None:
        self.frames[camera] = bytes(data)

    def feed_joints(self, names, positions) -> None:
        self.joints.update(zip(names, positions))

    def feed_twist(self, vx: float, vy: float, wz: float) -> None:
        self.twist = (vx, vy, wz)

    def ready(self) -> bool:
        c = self.contract
        return (all(cam in self.frames for cam in c.cameras.values()) and self.twist is not None
                and all(j in self.joints for j in (*c.arm_joints, c.finger_joints[0])))

    def missing(self) -> list[str]:
        c = self.contract
        return [*(cam for cam in c.cameras.values() if cam not in self.frames), *(["odom"] if self.twist is None else []),
                *(j for j in (*c.arm_joints, c.finger_joints[0]) if j not in self.joints)]

    def observe(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """(camera name -> RGB image, state vector), in the checkpoint's names and order."""
        c = self.contract
        return ({name: self.decoder(self.frames[cam], self.size) for name, cam in c.cameras.items()},
                c.state(self.twist, self.joints))

    def wire(self, node) -> None:
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage, JointState
        for cam in self.contract.cameras.values():
            node.create_subscription(CompressedImage, f"/{cam}/image/compressed",
                                     lambda m, cam=cam: self.feed_image(cam, m.data), qos_profile_sensor_data)
        node.create_subscription(JointState, "/joint_states", lambda m: self.feed_joints(m.name, m.position),
                                 qos_profile_sensor_data)
        node.create_subscription(Odometry, self.contract.odom, lambda m: self.feed_twist(
            m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.angular.z), qos_profile_sensor_data)

    def wait_ready(self, timeout_s: float = OBS_TIMEOUT_S, sleep=time.sleep) -> None:
        end = time.monotonic() + timeout_s
        while not self.ready():
            if time.monotonic() > end:
                raise RuntimeError(f"no observation after {timeout_s:.0f} s: missing {self.missing()}")
            sleep(0.05)


def run_chunks(client, n: int, observer: Observer, policy, prompt: str, per_chunk: int, publish, halt=None,
               timeout_s: float = RESULT_TIMEOUT_S + 60.0, sleep=time.sleep, wall=time.monotonic) -> dict:
    """Ask the policy for a chunk, send its first `per_chunk` actions at ACTION_HZ on sim time (the latest due tick, none twice;
    the ticks the inference took are skipped), repeat until the result of episode n is out or `halt()`. -> {steps, chunks,
    inference_s: [per chunk]}."""
    c = observer.contract
    t0, nxt, end = client.now(), 0, wall() + timeout_s
    stats = {"steps": 0, "chunks": 0, "inference_s": []}

    def due() -> int:
        return int((client.now() - t0) * ACTION_HZ)

    def over() -> bool:
        if wall() > end:
            raise SimFault(RESULT_TIMEOUT, f"no result for episode {n} after {timeout_s:.0f} s")
        return client.has_result(n) or bool(halt and halt())

    def tick_or_over() -> bool:
        """Wait for the tick `nxt`: True when the episode is over (or halted) first."""
        while not over():
            if due() >= nxt:
                return False
            sleep(0.001)
        return True

    while not over():
        images, state = observer.observe()
        a = wall()
        chunk = np.asarray(policy.act(images, state, prompt))
        stats["inference_s"].append(round(wall() - a, 4))
        stats["chunks"] += 1
        nxt = max(nxt, due())
        for action in chunk[:per_chunk]:
            if tick_or_over():
                break
            publish(*c.action(action))
            nxt, stats["steps"] = nxt + 1, stats["steps"] + 1
    return stats


def summary(stats: dict) -> dict:
    """What `episode.json` holds of the policy: steps, chunks and the inference rate (chunks per wall second of inference)."""
    t = stats["inference_s"]
    return {"steps": stats["steps"], "chunks": stats["chunks"],
            "inference": {"n": len(t), "mean_s": float(np.mean(t)) if t else None, "max_s": max(t) if t else None,
                          "hz": 1.0 / float(np.mean(t)) if t and np.mean(t) > 0 else None}}


def play_served(client, rollout_dir, serve: dict, reset: bool, ready_s: float, target, live, lead_s: float,
                make_streamer=None) -> dict:
    """One episode of the served policy (the rollout client's `episode` with `live` around it) -> what `episode.json` holds."""
    from .rollout_client import RosStreamer, episode
    c = CONTRACTS.get(serve["robot"]) or _unsupported(serve["robot"])
    series = ActionSeries(c.joints, {}, np.zeros((0, 3)))
    observer = Observer(c, serve.get("image_size"))
    observer.wire(client.node)
    streamer = (make_streamer or RosStreamer)(client.node, series, lead_s)
    policy = PolicyClient(serve["url"])

    def prepare():
        streamer.wait_matched()
        observer.wait_ready()

    return episode(client, rollout_dir, reset, ready_s,
                   lambda n: summary(run_chunks(client, n, observer, policy, serve["prompt"], int(serve["actions_per_chunk"]),
                                                streamer, halt=live.halt)),
                   prepare, target, live)


def _unsupported(robot: str):
    raise ValueError(f"no observation / action contract for robot {robot!r} (known: {sorted(CONTRACTS)})")
