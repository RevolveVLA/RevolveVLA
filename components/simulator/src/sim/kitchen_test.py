#!/usr/bin/env python3
"""Drive the kitchen scene test: wait for the (paused) world, start the MP4
recording, unpause, relay /kitchen_test/status, stop the recording, then add
the load checks (gz log, MuJoCo log) and print PASS/FAIL per check.

The scripted test itself runs inside gz-sim (scripts/kitchen_test_system.py),
which writes its numbers to $KITCHEN_TEST_OUT (default <out>/kitchen_test.json).

  kitchen_test.py --out DIR [--video out.mp4] [--log gz_sim.log] [--timeout 300]
                  [--status /kitchen_test/status] [--label KITCHEN_TEST]
Also drives the mobile Panda smoke test (worlds/kitchen_partnet_robot.sdf) with
--status /mobile_panda/status --label MOBILE_PANDA_SMOKE.
Writes DIR/summary.json and DIR/summary.txt. Exit code 0 when every check passes.
"""
import argparse
import json
import os
import queue
import re
import sys
import time

from gz.msgs.boolean_pb2 import Boolean
from gz.msgs.stringmsg_pb2 import StringMsg
from gz.msgs.video_record_pb2 import VideoRecord
from gz.msgs.world_control_pb2 import WorldControl
from gz.transport import Node

WORLD = "kitchen"


def call(node, service, req, req_type, rep_type, tries=40):
    for _ in range(tries):
        ok, rep = node.request(service, req, req_type, rep_type, 2000)
        if ok and rep.data:
            return True
        time.sleep(0.5)
    return False


def record(node, start, path):
    req = VideoRecord()
    req.start, req.stop = start, not start
    req.format, req.save_filename = "mp4", path
    return call(node, "/kitchen_cam/record_video", req, VideoRecord, Boolean)


def log_problems(log_path, mj_log, allow=""):
    """[Err] / NaN lines and MUJOCO_LOG.TXT entries, minus lines matching allow."""
    ok = (lambda line: bool(re.search(allow, line))) if allow else (lambda line: False)
    problems = []
    if os.path.exists(log_path):
        with open(log_path, errors="replace") as f:
            for line in f:
                if ("[Err]" in line or "Nan, Inf or huge" in line) and not ok(line):
                    problems.append(line.strip()[:200])
    else:
        problems.append(f"no log {log_path}")
    if os.path.exists(mj_log):
        with open(mj_log, errors="replace") as f:
            rest = [line for line in f.read().splitlines() if line.strip() and not ok(line)]
        if rest:
            problems.append("MUJOCO_LOG: " + "\n".join(rest)[-200:])
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", default="")
    ap.add_argument("--log", default="gz_sim.log")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--status", default="/kitchen_test/status")
    ap.add_argument("--label", default="KITCHEN_TEST")
    ap.add_argument("--allow", default="", help="regex of known-harmless log lines")
    args = ap.parse_args()
    result_json = os.environ.get("KITCHEN_TEST_OUT", os.path.join(args.out, "kitchen_test.json"))
    mj_log = os.path.join(os.path.dirname(os.path.abspath(args.log)), "MUJOCO_LOG.TXT")
    t_wall = time.time()

    node = Node()
    status = queue.Queue()
    node.subscribe(StringMsg, args.status, lambda m: status.put(m.data))

    if args.video:
        if os.path.exists(args.video):
            os.remove(args.video)
        print("Recording:", "started" if record(node, True, args.video) else "FAILED", flush=True)

    play = WorldControl()
    play.pause = False
    if not call(node, f"/world/{WORLD}/control", play, WorldControl, Boolean):
        print("Could not unpause the world")
        return 2

    done, deadline = None, time.time() + args.timeout
    while done is None and time.time() < deadline:
        try:
            line = status.get(timeout=0.5)
        except queue.Empty:
            continue
        print(line, flush=True)
        if line.startswith("DONE"):
            done = line
    if done is None:
        print("Timed out waiting for the scene test", flush=True)

    video_ok = None
    if args.video:
        time.sleep(1.0)
        stopped = record(node, False, args.video)
        for _ in range(300):         # encoder finalises asynchronously
            if os.path.exists(args.video):
                break
            time.sleep(0.1)
        video_ok = stopped and os.path.exists(args.video)
        print("Recording:", "saved to " + args.video if video_ok else "FAILED", flush=True)

    checks = []
    probs = log_problems(args.log, mj_log, args.allow)
    checks.append({"name": "load: no [Err] / NaN lines in the gz log, no MUJOCO_LOG",
                   "pass": not probs, "value": probs[:5], "limit": []})
    data = {}
    if done and os.path.exists(result_json):
        with open(result_json) as f:
            data = json.load(f)
        checks += data["checks"]
    else:
        checks.append({"name": "scene test finished", "pass": False, "value": done, "limit": "DONE"})
    if video_ok is not None:
        checks.append({"name": "video recorded", "pass": video_ok, "value": args.video, "limit": ""})
    ok = all(c["pass"] for c in checks)

    lines = []
    for c in checks:
        lines.append(f"{'PASS' if c['pass'] else 'FAIL'}  {c['name']}: {c['value']} "
                     f"(limit {c['limit']})")
    lines.append(f"{args.label} {'PASS' if ok else 'FAIL'} "
                 f"({sum(c['pass'] for c in checks)}/{len(checks)} checks, "
                 f"{time.time() - t_wall:.0f} s wall)")
    print("\n".join(lines), flush=True)
    os.makedirs(args.out, exist_ok=True)
    summary = dict(data, pass_all=ok, checks=checks, video=args.video,
                   wall_s=round(time.time() - t_wall, 1),
                   time=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
