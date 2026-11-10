#!/usr/bin/env python3
"""Drive a demo run: wait for the (paused) world, start the scene-camera
recording, unpause, relay /pick_place/status, then stop the recording.

The controller itself runs inside gz-sim (can_demo/pick_place_system.py). The Panda-only can demo.

  pick_place.py [--video out.mp4] [--timeout 120]
Exit code 0 when the can ends up on the plate.
"""
import argparse
import os
import queue
import sys
import time

from gz.msgs.boolean_pb2 import Boolean
from gz.msgs.stringmsg_pb2 import StringMsg
from gz.msgs.video_record_pb2 import VideoRecord
from gz.msgs.world_control_pb2 import WorldControl
from gz.transport import Node

WORLD = "can_on_plate"


def call(node, service, req, req_type, rep_type, tries=20):
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
    return call(node, "/scene_cam/record_video", req, VideoRecord, Boolean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    node = Node()
    status = queue.Queue()
    node.subscribe(StringMsg, "/pick_place/status", lambda m: status.put(m.data))

    if args.video:
        # remove a previous recording so we can tell when the new one lands
        if os.path.exists(args.video):
            os.remove(args.video)
        print("Recording:", "started" if record(node, True, args.video) else "FAILED")

    play = WorldControl()
    play.pause = False
    if not call(node, f"/world/{WORLD}/control", play, WorldControl, Boolean):
        print("Could not unpause the world")
        return 2

    verdict, deadline = None, time.time() + args.timeout
    while verdict is None and time.time() < deadline:
        try:
            line = status.get(timeout=0.5)
        except queue.Empty:
            continue
        print(line, flush=True)
        if line.startswith("DONE"):
            verdict = "SUCCESS" in line
    if verdict is None:
        print("Timed out waiting for the controller")

    if args.video:
        time.sleep(1.0)              # a little footage of the final state
        stopped = record(node, False, args.video)
        for _ in range(300):         # encoder finalises asynchronously
            if os.path.exists(args.video):
                break
            time.sleep(0.1)
        print("Recording:", "saved to " + args.video
              if stopped and os.path.exists(args.video) else "FAILED")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
