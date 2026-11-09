"""One RunPolicy goal to Rosetta's policy_runner_node for one closed-loop policy rollout (rollout.py). Ported
unchanged from the earlier pipeline's rosetta_goal.py. Runs in the rollout's client container (revolve-dataset: ROS 2
Jazzy, Rosetta sourced from ThirdParty/rosetta/install) beside the runner node:

  rosetta_goal.py <out dir> <prompt> <max_duration_s> <sim done file> <contract.yaml>

1. Waits for the /run_policy action server, then touches <out>/runner_ready.
2. Waits for /clock (the sim was unpaused), then sends {prompt, max_duration_s}.
3. Records the goal's feedback (published_actions, queue_depth) and every message on the contract's action topic,
   each with the /clock time it arrived at, and saves the first message of each observation topic stamped at or
   after the goal's acceptance to <out>/obs_first.npz (JPEG bytes, joint positions by name, stamps): what the
   policy's first observation is sampled from.
4. Once <sim done file> exists (the sim's summary.json: its episode ended) or the goal ends, cancels the goal, waits for its
   result and writes <out>/runner.json, then touches <out>/runner_done. max_duration_s is wall clock (Rosetta's):
   the sim owns the episode budget; this is only a safety stop.
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosetta_interfaces.action import RunPolicy
import numpy as np
import yaml
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CompressedImage, JointState

WAIT_SERVER_S, WAIT_CLOCK_S, WAIT_RESULT_S = 180.0, 300.0, 30.0


def say(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def main(out, prompt, max_duration_s, done_file, contract):
    out, done_file = Path(out), Path(done_file)
    c = yaml.safe_load(Path(contract).read_text())
    (act,) = c["actions"].values()
    action_topic = act["channel"]["topic"]
    obs_topics = {k: o["channel"] for k, o in c["observations"].items()}
    rclpy.init()
    node = rclpy.create_node("eval_goal")
    lock = threading.Lock()
    clock = [None]                       # newest /clock [s]
    feedback, actions = [], []           # (wall s, sim s, published, depth) / (wall s, sim s)
    node.create_subscription(Clock, "/clock", lambda m: clock.__setitem__(0, m.clock.sec + m.clock.nanosec * 1e-9),
                             QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

    def on_action(_msg):
        with lock:
            actions.append((time.time(), clock[0]))
    node.create_subscription(JointState, action_topic, on_action,
                             QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT))
    first, accepted_at = {}, [None]          # observation key -> (stamp s, payload) at or after the goal's acceptance

    def keep(key, stamp, payload):
        if accepted_at[0] is not None and key not in first and stamp >= accepted_at[0]:
            first[key] = (stamp, payload)
    for key, ch in obs_topics.items():
        if ch["type"].endswith("CompressedImage"):
            cb = (lambda k: lambda m: keep(k, m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                                           np.frombuffer(bytes(m.data), np.uint8)))(key)
            node.create_subscription(CompressedImage, ch["topic"], cb, QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))
        elif ch["type"].endswith("JointState"):
            cb = (lambda k: lambda m: keep(k, m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                                           dict(zip(m.name, m.position))))(key)
            node.create_subscription(JointState, ch["topic"], cb, QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))
    client = ActionClient(node, RunPolicy, "/run_policy")
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    rec = dict(prompt=prompt, max_duration_s=float(max_duration_s), action_topic=action_topic, accepted=False,
               termination_reason=None, message=None, status=None, error=None)
    t0 = time.time()
    try:
        if not client.wait_for_server(timeout_sec=WAIT_SERVER_S):
            raise RuntimeError(f"no /run_policy action server in {WAIT_SERVER_S} s")
        rec["server_wait_s"] = round(time.time() - t0, 2)
        (out / "runner_ready").touch()
        say(f"/run_policy server up after {rec['server_wait_s']} s: runner_ready")
        t1 = time.time()
        while clock[0] is None:
            if time.time() - t1 > WAIT_CLOCK_S:
                raise RuntimeError(f"no /clock in {WAIT_CLOCK_S} s after runner_ready")
            time.sleep(0.01)
        rec["goal_sent"] = dict(wall=time.time(), sim=clock[0])
        say(f"/clock at {clock[0]:.3f} s: sending the goal {prompt!r}")

        def on_feedback(fb):
            with lock:
                feedback.append((time.time(), clock[0], fb.feedback.published_actions, fb.feedback.queue_depth))
        fut = client.send_goal_async(RunPolicy.Goal(prompt=prompt, max_duration_s=float(max_duration_s)),
                                     feedback_callback=on_feedback)
        while not fut.done():
            time.sleep(0.01)
        gh = fut.result()
        rec["accepted"] = bool(gh.accepted)
        rec["goal_accepted"] = dict(wall=time.time(), sim=clock[0])
        accepted_at[0] = clock[0]
        if not gh.accepted:
            raise RuntimeError("goal rejected")
        say(f"goal accepted at sim {clock[0]:.3f} s")
        res = gh.get_result_async()
        last = time.time()
        while not (res.done() or done_file.exists()):
            time.sleep(0.05)
            if time.time() - last > 10:
                last = time.time()
                with lock:
                    say(f"sim {clock[0]:.2f} s, actions seen {len(actions)}, feedback {feedback[-1][2:] if feedback else None}")
        rec["sim_done_seen"] = dict(wall=time.time(), sim=clock[0], goal_done_first=res.done())
        say(f"{'goal ended' if res.done() else 'sim done'} at sim {clock[0]:.2f} s")
        if not res.done():
            gh.cancel_goal_async()
            t2 = time.time()
            while not res.done() and time.time() - t2 < WAIT_RESULT_S:
                time.sleep(0.05)
        if res.done():
            r = res.result()
            rec.update(termination_reason=r.result.termination_reason, message=r.result.message,
                       status={GoalStatus.STATUS_SUCCEEDED: "succeeded", GoalStatus.STATUS_CANCELED: "canceled",
                               GoalStatus.STATUS_ABORTED: "aborted"}.get(r.status, str(r.status)))
        else:
            rec["error"] = f"no result {WAIT_RESULT_S} s after the cancel"
    except Exception as e:                # noqa: BLE001 (recorded, the episode becomes an error)
        rec["error"] = f"{type(e).__name__}: {e}"
    with lock:
        rec["feedback"] = feedback
        rec["actions_seen"] = actions
    if first:
        arrays = {}
        for key, (stamp, payload) in first.items():
            arrays[f"{key}.stamp"] = np.array(stamp)
            if isinstance(payload, dict):
                arrays[f"{key}.names"] = np.array(list(payload))
                arrays[f"{key}.position"] = np.array(list(payload.values()))
            else:
                arrays[f"{key}.jpeg"] = payload
        np.savez(out / "obs_first.npz", **arrays)
        rec["obs_first"] = {k: round(v[0], 4) for k, v in first.items()}
    tmp = out / "runner.json.tmp"
    tmp.write_text(json.dumps(rec))
    tmp.rename(out / "runner.json")
    (out / "runner_done").touch()
    print(f"goal: accepted {rec['accepted']}, {len(actions)} actions seen, termination {rec['termination_reason']}, "
          f"error {rec['error']}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main(*sys.argv[1:])
