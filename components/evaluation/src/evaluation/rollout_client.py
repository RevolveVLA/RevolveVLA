"""Policy client of the `sim` rollout: `python3 -m evaluation.rollout_client --sim-out DIR --rollout DIR (--bag BAG | --spec
JSON --instance SCENE_YAML | --serve JSON) [--reset | --reset-to TARGET]` runs ONE episode against the sim on this container's ROS domain
(a client container from the simulator image, ROS 2 Lyrical): wait for ready, reset when the sim already ran an episode
(`--reset-to`: a warm reset to another start, revolve_common.warm_reset), `/evaluator/start`, then
the policy acts until the evaluator's result: the oracle streams a bag's actions as a policy would (single-point
`joint_trajectory` per controller + `cmd_vel`, 30 Hz, sim time: DATAGEN-v2.md 5.2), the expert runs the training spec's datagen
generator closed-loop (`prepare` before the episode starts, `drive` after). Then wait for the result's files and write
`<rollout>/episode.json` = {episode, resets, result, + reset (the warm reset's report), + the policy's: steps | expert, notes}.
`--serve JSON` = a served checkpoint (evaluation.served: observation from the sim's topics, actions from a policy service).
`--driver MODULE:FUNC [--driver-args JSON]` (the oracle and a served policy) wraps the policy of the episode with a `Live`
(FUNC(client, args, instance, rollout) -> Live): a live recorder that supervises the policy and may take over.
Exit code: 0 done, a SimFault's code (75-78: the host relaunches the sim), 79 a warm reset that missed its target (the host
relaunches with that start; `<rollout>/reset.json` has the report), 1 anything else."""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

from revolve_common.sim_client import RESULT_TIMEOUT_S, WarmResetMissed, run_client
from revolve_common.sim_session import READY_TIMEOUT_S, RESULT_TIMEOUT, SimFault
from revolve_common.warm_reset import ResetReport, ResetTarget

from .oracle_actions import ACTION_HZ, CMD_VEL, ActionSeries, read_bag

LEAD_S = 0.1                    # default time_from_start of a streamed point (the simulator's contract test uses 0.1 s)
MATCH_TIMEOUT_S = 30.0          # wall: the sim's controllers must subscribe before the episode starts


class Live:
    """What wraps the policy of one episode (a live recorder's driver, see recovery.live_client); this default wraps nothing.
    `halt()` is polled before every action (True: the policy stops acting), `series` may alter the actions to play,
    `driver(drive)` is called once the sim is ready and reset, before the episode starts, and returns the drive that runs it,
    `finish(episode)` gets what `episode.json` holds."""

    def halt(self) -> bool:
        return False

    def series(self, series: ActionSeries) -> ActionSeries:
        return series

    def driver(self, drive):
        return drive

    def finish(self, episode: dict) -> None:
        pass


NO_LIVE = Live()


class RosStreamer:
    """Publishes one action: a single-point JointTrajectory per controller and the TwistStamped base command."""

    def __init__(self, node, series: ActionSeries, lead_s: float = LEAD_S):
        from geometry_msgs.msg import TwistStamped
        from trajectory_msgs.msg import JointTrajectory
        self.node, self.series, self.lead_s = node, series, lead_s
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
            m.points[0].time_from_start.nanosec = int(self.lead_s * 1e9)
            pub.publish(m)
        t = TwistStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.twist.linear.x, t.twist.linear.y, t.twist.angular.z = (float(v) for v in twist)
        self.base.publish(t)


def stream(client, n: int, series: ActionSeries, publish, timeout_s: float = RESULT_TIMEOUT_S + 60.0,
           sleep=time.sleep, halt=None) -> int:
    """Publish the series' actions at ACTION_HZ on sim time (the latest due tick, none twice) until the result of
    episode n is out; the last references are held when the series ends. `halt()` is polled before each action: when it
    says True the stream ends there (the rest of the series is dropped). -> the number of actions sent."""
    t0, nxt, sent, end = client.now(), 0, 0, time.monotonic() + timeout_s
    while not client.has_result(n):
        if time.monotonic() > end:
            raise SimFault(RESULT_TIMEOUT, f"no result for episode {n} after {timeout_s:.0f} s")
        if halt and halt():
            break
        due = int((client.now() - t0) * ACTION_HZ)
        if due >= nxt:
            publish(*series.at(due))
            nxt, sent = due + 1, sent + 1
        else:
            sleep(0.001)
    return sent


def episode(client, rollout_dir, reset: bool, ready_s: float, drive, prepare=None, target: ResetTarget | None = None,
            live: Live = NO_LIVE) -> dict:
    """One episode (the loop of DATAGEN 4.4 for a sim that may have run one before) -> what `episode.json` holds.
    `prepare()` runs once the sim is ready (and reset), before the episode starts; `drive(n)` acts until the result of
    episode n is out and returns what the policy adds to the file. `target`: warm reset to that start instead of a plain
    reset (its report is in the file as `reset`; a miss writes `reset.json` and raises WarmResetMissed). `live` wraps the drive (after `prepare`, before the start) and gets the episode file."""
    client.wait_ready(ready_s)
    report = None
    if target is not None:
        report = client.reset_to(target).to_dict()
        if not report["ok"]:
            Path(rollout_dir).mkdir(parents=True, exist_ok=True)
            (Path(rollout_dir) / "reset.json").write_text(json.dumps(report, indent=1))
            raise WarmResetMissed(ResetReport(**report))
    elif reset:
        client.reset()
    if prepare:
        prepare()
    drive = live.driver(drive)
    resets = client.resets()
    n = client.start_episode()
    extra = drive(n)
    client.wait_result(n)
    ep = {"episode": n, "resets": resets, **extra, **({"reset": report} if report else {}), "result": client.wait_files(n)}
    Path(rollout_dir).mkdir(parents=True, exist_ok=True)
    (Path(rollout_dir) / "episode.json").write_text(json.dumps(ep, indent=1))
    live.finish(ep)
    return ep


def play(client, rollout_dir, series: ActionSeries, publish, reset: bool, ready_s: float = READY_TIMEOUT_S,
         target: ResetTarget | None = None, live: Live = NO_LIVE) -> dict:
    """The oracle: stream the series' actions."""
    series = live.series(series)
    return episode(client, rollout_dir, reset, ready_s, lambda n: {"steps": stream(client, n, series, publish, halt=live.halt)},
                   getattr(publish, "wait_matched", None), target, live)


def play_expert(client, rollout_dir, generator, instance, task: str, robot: str, reset: bool,
                ready_s: float = READY_TIMEOUT_S, target: ResetTarget | None = None) -> dict:
    """The expert: a datagen generator drives the episode closed-loop (its planner starts off the episode's clock)."""
    return episode(client, rollout_dir, reset, ready_s, lambda n: dict(generator.drive(client, instance, task, robot)),
                   lambda: generator.prepare(client, instance, task, robot), target)


def load_live(a, client) -> Live:
    """The `Live` of `--driver MODULE:FUNC`: FUNC(client, args, instance, rollout dir)."""
    module, _, func = a.driver.partition(":")
    return getattr(importlib.import_module(module), func)(client, json.loads(a.driver_args),
                                                          Path(a.instance) if a.instance else None, Path(a.rollout))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="evaluation.rollout_client")
    ap.add_argument("--sim-out", required=True, help="the sim's --out (episode_<n>.json + bags)")
    ap.add_argument("--rollout", required=True, help="the rollout dir: episode.json goes here")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--bag", help="the oracle: the datagen episode bag whose actions are replayed")
    who.add_argument("--spec", help="the expert: the training datagen spec as JSON (DatagenSpec.raw), its generator drives")
    who.add_argument("--serve", help="a served checkpoint: {url, prompt, robot, actions_per_chunk, image_size} as JSON")
    ap.add_argument("--instance", help="the expert: the scene instance's scene.yaml")
    ap.add_argument("--reset", action="store_true", help="the sim already ran an episode: reset it first")
    ap.add_argument("--reset-to", help="warm reset to this start (a ResetTarget as JSON) instead of a plain reset")
    ap.add_argument("--driver", help="the oracle: MODULE:FUNC building the episode's `Live` wrapper (a live recorder)")
    ap.add_argument("--driver-args", default="{}", help="the driver's arguments as JSON")
    ap.add_argument("--ready-s", type=float, default=READY_TIMEOUT_S)
    ap.add_argument("--lead-s", type=float, default=LEAD_S, help="time_from_start of each streamed point (the oracle)")
    a = ap.parse_args(argv)
    target = ResetTarget.from_json(a.reset_to) if a.reset_to else None
    if a.spec:
        if not a.instance:
            ap.error("--spec needs --instance")
        if a.driver:
            ap.error("--driver needs the oracle (--bag): an expert's drive cannot be halted")
        from datagen.core import registry
        from datagen.core.spec import parse
        spec = parse(json.loads(a.spec), "<spec>")
        generator = registry.create("generator", spec.generator.plugin, spec, spec.generator.params)
        run = lambda c: play_expert(c, a.rollout, generator, Path(a.instance), spec.task, spec.robot, a.reset, a.ready_s, target)
    elif a.serve:
        from .served import play_served
        run = lambda c: play_served(c, a.rollout, json.loads(a.serve), a.reset, a.ready_s, target,
                                    load_live(a, c) if a.driver else NO_LIVE, a.lead_s)
    else:
        series = read_bag(a.bag)
        run = lambda c: play(c, a.rollout, series, RosStreamer(c.node, series, a.lead_s), a.reset, a.ready_s, target,
                             load_live(a, c) if a.driver else NO_LIVE)
    run_client(run, a.sim_out)


if __name__ == "__main__":
    main()
