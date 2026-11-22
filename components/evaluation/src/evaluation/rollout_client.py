"""Policy client of the `sim` rollout: `python3 -m evaluation.rollout_client --sim-out DIR --rollout DIR --bag BAG [--reset]`
runs ONE episode against the sim on this container's ROS domain (a client container from the simulator image, ROS 2 Lyrical;
only revolve_common and numpy are imported): wait for ready, reset when the sim already ran an episode, `/evaluator/start`,
stream the oracle's actions as a policy would (single-point `joint_trajectory` per controller + `cmd_vel`, 30 Hz, sim time:
DATAGEN-v2.md 5.2) until the evaluator's result, wait for its files, write `<rollout>/episode.json` = {episode, resets,
steps, result}. Exit code: 0 done, a SimFault's code (75-78: the host relaunches the sim), 1 anything else."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from revolve_common.sim_client import RESULT_TIMEOUT_S, run_client
from revolve_common.sim_session import READY_TIMEOUT_S, RESULT_TIMEOUT, SimFault

from .oracle_actions import ACTION_HZ, CMD_VEL, ActionSeries, read_bag

LEAD_S = 0.1                    # time_from_start of a streamed point (the simulator's contract test uses 0.1 s)
MATCH_TIMEOUT_S = 30.0          # wall: the sim's controllers must subscribe before the episode starts


class RosStreamer:
    """Publishes one action: a single-point JointTrajectory per controller and the TwistStamped base command."""

    def __init__(self, node, series: ActionSeries):
        from geometry_msgs.msg import TwistStamped
        from trajectory_msgs.msg import JointTrajectory
        self.node, self.series = node, series
        self.arms = {c: node.create_publisher(JointTrajectory, f"/{c}/joint_trajectory", 10) for c in series.joints}
        self.base = node.create_publisher(TwistStamped, CMD_VEL, 10)

    def wait_matched(self, timeout_s: float = MATCH_TIMEOUT_S) -> None:
        """Until every command topic has a subscriber (a point published before would be lost)."""
        pubs = [*self.arms.values(), self.base]
        end = time.monotonic() + timeout_s
        while any(p.get_subscription_count() < 1 for p in pubs):
            if time.monotonic() > end:
                raise RuntimeError(f"no subscriber on {[p.topic_name for p in pubs if not p.get_subscription_count()]}")
            time.sleep(0.05)

    def __call__(self, positions, twist) -> None:
        from geometry_msgs.msg import TwistStamped
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        for c, pub in self.arms.items():
            m = JointTrajectory()                           # a zero header stamp: start now
            m.joint_names = list(self.series.joints[c])
            m.points = [JointTrajectoryPoint(positions=[float(v) for v in positions[c]])]
            m.points[0].time_from_start.nanosec = int(LEAD_S * 1e9)
            pub.publish(m)
        t = TwistStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.twist.linear.x, t.twist.linear.y, t.twist.angular.z = (float(v) for v in twist)
        self.base.publish(t)


def stream(client, n: int, series: ActionSeries, publish, timeout_s: float = RESULT_TIMEOUT_S + 60.0,
           sleep=time.sleep) -> int:
    """Publish the series' actions at ACTION_HZ on sim time (the latest due tick, none twice) until the result of
    episode n is out; the last references are held when the series ends. -> the number of actions sent."""
    t0, nxt, sent, end = client.now(), 0, 0, time.monotonic() + timeout_s
    while not client.has_result(n):
        if time.monotonic() > end:
            raise SimFault(RESULT_TIMEOUT, f"no result for episode {n} after {timeout_s:.0f} s")
        due = int((client.now() - t0) * ACTION_HZ)
        if due >= nxt:
            publish(*series.at(due))
            nxt, sent = due + 1, sent + 1
        else:
            sleep(0.001)
    return sent


def play(client, rollout_dir, series: ActionSeries, publish, reset: bool, ready_s: float = READY_TIMEOUT_S) -> dict:
    """One episode (the loop of DATAGEN 4.4 for a sim that may have run one before) -> what `episode.json` holds."""
    client.wait_ready(ready_s)
    if reset:
        client.reset()
    if hasattr(publish, "wait_matched"):
        publish.wait_matched()
    resets = client.resets()
    n = client.start_episode()
    steps = stream(client, n, series, publish)
    client.wait_result(n)
    episode = {"episode": n, "resets": resets, "steps": steps, "result": client.wait_files(n)}
    Path(rollout_dir).mkdir(parents=True, exist_ok=True)
    (Path(rollout_dir) / "episode.json").write_text(json.dumps(episode, indent=1))
    return episode


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="evaluation.rollout_client")
    ap.add_argument("--sim-out", required=True, help="the sim's --out (episode_<n>.json + bags)")
    ap.add_argument("--rollout", required=True, help="the rollout dir: episode.json goes here")
    ap.add_argument("--bag", required=True, help="the datagen episode bag whose actions are replayed (the oracle)")
    ap.add_argument("--reset", action="store_true", help="the sim already ran an episode: reset it first")
    ap.add_argument("--ready-s", type=float, default=READY_TIMEOUT_S)
    a = ap.parse_args(argv)
    series = read_bag(a.bag)
    run_client(lambda client: play(client, a.rollout, series, RosStreamer(client.node, series), a.reset, a.ready_s),
               a.sim_out)


if __name__ == "__main__":
    main()
