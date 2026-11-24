"""Client side of one running simulator (DATAGEN-v2.md sections 4.2-4.5), for code that runs in a client container
(ROS 2 Lyrical, `ClientContainer` in sim_session.py): wait for ready, start episodes, reset between them, wait for the
result and its files. rclpy is imported only in `SimClient.connect()` (a MultiThreadedExecutor in a background thread,
`use_sim_time`, /evaluator/result transient local); everything else is plain Python. A fault of the sim raises
`SimFault`; `run_client` turns it into the exit code the host relaunches on.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import deque

from . import warm_reset
from .sim_session import READY_TIMEOUT, READY_TIMEOUT_S, RESET_FAILED, RESULT_TIMEOUT, SimFault, wait_files
from .warm_reset import MISS_EXIT, ResetReport, ResetTarget, Tolerance

RESET_TIMEOUT_S = 60.0          # the service answers in ~2 s; it waits at most 20 s per step
RESET_TO_TIMEOUT_S = 30.0       # sim s for the drive to the target (a 0.3 m move takes ~3)
RESULT_TIMEOUT_S = 120.0        # wall: a task's budget_s is 45 s of sim time at real-time factor 1


class EpisodeRefused(RuntimeError):
    """/evaluator/start said no (inputs not ready, or an episode is running)."""


class WarmResetMissed(RuntimeError):
    """`reset_to` did not reach its target: the host relaunches the sim with that start instead."""

    def __init__(self, report: ResetReport):
        self.report = report
        super().__init__(report.reason)


class SimClient:
    """The evaluator + reset interface of one sim. `connect()` for the real thing; `SimClient(out=...)` alone has no ROS
    (tests feed `on_state` / `on_result`, and override `_trigger`)."""

    def __init__(self, node=None, out=None):
        self.node, self.out = node, out
        self.states, self.results = deque(maxlen=200), {}       # /evaluator/state (10 Hz), results by episode
        self.tf = self._tf_listener = None
        self._lock = threading.Lock()
        self._clients, self._executor, self._plants = {}, None, {}
        if node is not None:
            self._wire()

    @classmethod
    def connect(cls, out, name: str = "sim_client") -> SimClient:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.parameter import Parameter
        rclpy.init()
        node = rclpy.create_node(name, parameter_overrides=[Parameter("use_sim_time", value=True)])
        client = cls(node, out)
        client._executor = MultiThreadedExecutor(num_threads=4)     # not spin_once: it drops frames and replies
        client._executor.add_node(node)
        threading.Thread(target=client._executor.spin, daemon=True).start()
        client.new_tf()
        return client

    def _wire(self) -> None:
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from std_msgs.msg import String
        from std_srvs.srv import Trigger
        self.node.create_subscription(String, "/evaluator/state", lambda m: self.on_state(m.data), 50)
        self.node.create_subscription(String, "/evaluator/result", lambda m: self.on_result(m.data),
                                      QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._clients = {s: self.node.create_client(Trigger, s) for s in ("/evaluator/start", "/simulator/reset")}
        self._trigger_type = Trigger

    # --- messages
    def on_state(self, data: str) -> None:
        with self._lock:
            self.states.append(json.loads(data))

    def on_result(self, data: str) -> None:
        r = json.loads(data)
        with self._lock:
            self.results[r["episode"]] = r

    def state(self) -> dict | None:
        with self._lock:
            return self.states[-1] if self.states else None

    def has_result(self, n: int) -> bool:
        with self._lock:
            return n in self.results

    def resets(self) -> int:
        s = self.state()
        return s["resets"] if s else 0

    def now(self) -> float:
        """Sim time (s); it restarts at 1 s on every reset."""
        return self.node.get_clock().now().nanoseconds * 1e-9

    def wait_for(self, pred, wall_s: float) -> bool:
        end = time.monotonic() + wall_s
        while not pred():
            if time.monotonic() > end:
                return False
            time.sleep(0.02)
        return True

    def _trigger(self, service: str, wall_s: float) -> tuple[bool, str]:
        cli = self._clients[service]
        if not cli.wait_for_service(timeout_sec=wall_s):
            return False, f"no {service}"
        fut = cli.call_async(self._trigger_type.Request())
        if not self.wait_for(fut.done, wall_s):
            return False, f"{service} timed out"
        return fut.result().success, fut.result().message

    # --- the loop of DATAGEN 4.4
    def wait_ready(self, timeout_s: float = READY_TIMEOUT_S) -> None:
        """`/evaluator/state` ready: true (nothing else shows readiness). SimFault(ready_timeout) otherwise."""
        if not self.wait_for(lambda: (s := self.state()) is not None and s["ready"], timeout_s):
            raise SimFault(READY_TIMEOUT, f"not ready after {timeout_s:.0f} s")

    def reset(self, timeout_s: float = RESET_TIMEOUT_S) -> None:
        """/simulator/reset, then wait for a state with resets above the one before and ready: true; the tf Buffer is
        rebuilt (sim time restarted). SimFault(reset_failed) on a false reply or a timeout."""
        before = self.resets()
        ok, msg = self._trigger("/simulator/reset", timeout_s)
        if not ok:
            raise SimFault(RESET_FAILED, msg)

        def done():
            with self._lock:
                return any(s["ready"] and s["resets"] > before for s in self.states)
        if not self.wait_for(done, timeout_s):
            raise SimFault(RESET_FAILED, f"not ready after the reset ({msg})")
        if self.node is not None:
            self.new_tf()

    def reset_to(self, target: ResetTarget, tol: Tolerance = Tolerance(),
                 timeout_s: float = RESET_TO_TIMEOUT_S) -> ResetReport:
        """`reset()`, then drive the robot from the scene's start to `target` (warm_reset.py). Call it before the episode
        starts. Never raises for a miss: `report.ok` / `report.reason` say. SimFault(reset_failed) as `reset()`."""
        w0 = time.monotonic()
        self.reset()
        report = warm_reset.run(self._plant(target), target, tol, timeout_s)
        return report.with_times(time.monotonic() - w0, report.sim_s)

    def _plant(self, target: ResetTarget):
        key = (target.robot, target.arm_controller, target.arm_joints)
        if key not in self._plants:
            self._plants[key] = warm_reset.RosPlant(self, target)
        return self._plants[key]

    def new_tf(self) -> None:
        """A fresh tf Buffer + listener: `Buffer.clear()` would also drop /tf_static."""
        from tf2_ros import Buffer, TransformListener
        if self._tf_listener is not None:
            self._tf_listener.unregister()
        self.tf = Buffer()
        self._tf_listener = TransformListener(self.tf, self.node, spin_thread=False)

    def start_episode(self) -> int:
        """/evaluator/start -> the episode number n; EpisodeRefused when it says no."""
        ok, msg = self._trigger("/evaluator/start", 30.0)
        if not ok:
            raise EpisodeRefused(msg)
        return int(msg.split()[-1])

    def wait_result(self, n: int, timeout_s: float = RESULT_TIMEOUT_S) -> dict:
        """The /evaluator/result of episode n (the schema of DATAGEN 4.3). SimFault(result_timeout) otherwise."""
        if not self.wait_for(lambda: n in self.results, timeout_s):
            raise SimFault(RESULT_TIMEOUT, f"no result for episode {n} after {timeout_s:.0f} s")
        return self.results[n]

    def wait_files(self, n: int, timeout_s: float = 30.0) -> dict:
        """Episode n's `episode_<n>.json` (parsed): its bag is closed and complete once the file exists."""
        return wait_files(self.out, n, timeout_s)

    def close(self) -> None:
        if self.node is None:
            return
        import rclpy
        self._executor.shutdown()
        self.node.destroy_node()
        rclpy.shutdown()


def run_client(fn, out, *args) -> None:
    """Connect, `fn(client, *args)`, then end the process: exit 0, the exit code of a SimFault (the host relaunches
    the sim on those) or MISS_EXIT for a warm reset that missed (the host relaunches with that start). os._exit: moveit_py segfaults at interpreter shutdown, so flush and leave."""
    code = 0
    try:
        client = SimClient.connect(out)
        fn(client, *args)
    except SimFault as e:
        print(f"SimFault {e}", file=sys.stderr)
        code = e.exit_code
    except WarmResetMissed as e:
        print(f"warm reset missed: {e}", file=sys.stderr)
        code = MISS_EXIT
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
