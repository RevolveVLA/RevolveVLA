"""Records test episodes for the dataset component (DATASET-v2.md section 11): a scripted client that commands the
robot as a policy does, next to a sim started by `revolve sim --task T --no-autostart --record images --out DIR`. Runs
in a client container from the simulator image on the sim's ROS_DOMAIN_ID (bags.py), like the simulator's contract
test client (components/simulator/ros/revolve_sim/test/check_contract.py), and uses only the documented interface.

Per episode: wait for /evaluator/state `ready`; /evaluator/start; stream at 30 Hz single-point JointTrajectory
messages to the arm and hand controllers (header stamp zero: "start now") and a TwistStamped to the base, until the
evaluator's result; wait for the result file; /simulator/reset before the next one. The command is a slow sine about
the start pose, different per episode (seeded), so episodes differ; the base only sways around its start. Such
episodes fail the task (the evaluator's `passed: false`). Prints one RESULT JSON line: the episodes' results.

  python3 stream_client.py --scene S --robot R --task T --episodes N [--seed K] [--repo /repo]
"""
import argparse
import json
import math
import random
import sys
import threading
import time
from pathlib import Path

TEST = Path("/repo/components/simulator/ros/revolve_sim/test")
sys.path[:0] = [str(TEST)]

import rclpy  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # noqa: E402

from check_evaluator import EvalProbe  # noqa: E402
from revolve_sim.robot import describe  # noqa: E402
from revolve_sim.world import make_world  # noqa: E402
from robot_cases import CONTROL  # noqa: E402

HZ = 30.0


def point(joints, q):
    m = JointTrajectory()
    m.joint_names = list(joints)
    m.points = [JointTrajectoryPoint(positions=[float(v) for v in q])]
    m.points[0].time_from_start.nanosec = 100_000_000
    return m


def main():
    ap = argparse.ArgumentParser()
    for k in ("--scene", "--robot", "--task"):
        ap.add_argument(k, required=True)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo", default="/repo")
    a = ap.parse_args()
    _, scene, _, _ = make_world(a.scene, a.repo)
    robot, _, _, contact_links = describe(a.repo, a.robot)
    case = CONTROL[robot.name]
    q0 = robot.poses[scene.robot_start.arm] if isinstance(scene.robot_start.arm, str) else scene.robot_start.arm
    rclpy.init()
    n = EvalProbe(robot.name, scene.name, contact_links)
    arm = n.create_publisher(JointTrajectory, f"/{case['arm_controller']}/joint_trajectory", 10)
    hand = n.create_publisher(JointTrajectory, f"/{case['hand_controller']}/joint_trajectory", 10)
    reset = n.create_client(Trigger, "/simulator/reset")
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    out, rng = {"episodes": []}, random.Random(a.seed)
    assert n.wait_for(lambda: n.states and n.states[-1]["ready"], 300), "evaluator never ready"
    for i in range(a.episodes):
        if i:
            with n.lock:
                resets = n.states[-1]["resets"]
            assert reset.wait_for_service(timeout_sec=30)
            f = reset.call_async(Trigger.Request())
            assert n.wait_for(f.done, 120) and f.result().success, "/simulator/reset failed"
            assert n.wait_for(lambda r=resets: any(s["ready"] and s["resets"] > r for s in list(n.states)), 120), \
                "not ready after the reset"
        with n.lock:
            n.result = None
        ok, msg = n.start()
        assert ok, msg
        k = int(msg.split()[-1])            # "episode <n>"
        amp, freq, base = rng.uniform(0.05, 0.15), rng.uniform(0.2, 0.6), rng.uniform(0.0, 0.05)
        t0, w0 = n.now(), time.time()
        while True:
            with n.lock:
                res = n.result
            if res is not None and res["episode"] == k:
                break
            t = n.now() - t0
            q = list(q0)
            q[0] += amp * math.sin(2 * math.pi * freq * t)
            arm.publish(point(case["arm_joints"], q))
            grip = case["open"] if math.sin(math.pi * freq * t) > 0 else case["closed"]
            hand.publish(point([case["hand_joint"]], [grip]))
            n.twist((base * math.sin(2 * math.pi * 0.25 * t), 0.0, 0.0))
            n.wait(1.0 / HZ)
            assert time.time() - w0 < 600, f"episode {k}: no result within 600 s wall"
        assert n.wait_for(Path(res["result_file"]).is_file, 60), f"no result file {res['result_file']}"
        out["episodes"].append(dict(result=res, wall_s=round(time.time() - w0, 1), amp=amp, freq=freq))
        print(f"episode {k}: {res.get('reason')} {res.get('duration_s')} s, {time.time() - w0:.1f} s wall", flush=True)
    ex.shutdown()
    rclpy.shutdown()
    print("RESULT " + json.dumps(out, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
