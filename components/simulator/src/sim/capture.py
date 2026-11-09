"""Save the training cameras' frames while gz replays one attempt (gz_replay.sh, cameras on).

  capture.py <model dir> <out dir> <rate Hz>

Subscribes to every camera sensor in <model dir>/model.sdf and writes
<out dir>/<camera>/<sim time, s>.png for the frames on a 1/rate grid of sim time (rate at or
above the camera's own rate keeps every frame). Runs beside gz until SIGTERM, then finishes the
pending PNGs and writes <out dir>/capture.json. bag.py puts the frames into the bag.
"""
import json
import os
import signal
import struct
import sys
import threading
import xml.etree.ElementTree as ET
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gz.msgs.image_pb2 import Image
from gz.transport import Node


def cameras(model_dir):
    root = ET.parse(Path(model_dir) / "model.sdf").getroot()
    return [dict(name=s.get("name"), topic=s.findtext("topic"), rate=float(s.findtext("update_rate")))
            for s in root.iter("sensor") if s.get("type") == "camera"]


def write_png(path, w, h, rgb):
    raw = b"".join(b"\x00" + rgb[y * w * 3:(y + 1) * w * 3] for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def main(model_dir, out_dir, rate):
    rate = float(rate)
    cams = cameras(model_dir)
    if not cams:
        raise SystemExit(f"no camera sensors in {model_dir}/model.sdf")
    pool = ThreadPoolExecutor(4)                    # zlib releases the GIL: PNGs encode in parallel
    lock = threading.Lock()
    stats = {c["name"]: dict(topic=c["topic"], camera_hz=c["rate"], received=0, saved=0, first_s=None, last_s=None)
             for c in cams}
    seen = {c["name"]: set() for c in cams}
    errors = []
    closing = []

    def on_image(c):
        d = Path(out_dir) / c["name"]
        d.mkdir(parents=True, exist_ok=True)
        tol = 0.5 / c["rate"] + 1e-6

        def cb(msg):
            t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
            n = round(t * rate)
            with lock:
                s = stats[c["name"]]
                s["received"] += 1
                if closing or abs(t - n / rate) > tol or n in seen[c["name"]]:
                    return
                if len(msg.data) != msg.width * msg.height * 3:
                    errors.append(f"{c['name']} {t:.3f}: {len(msg.data)} bytes for {msg.width}x{msg.height} RGB")
                    return
                seen[c["name"]].add(n)
                s["saved"] += 1
                s["first_s"] = t if s["first_s"] is None else min(s["first_s"], t)
                s["last_s"] = t if s["last_s"] is None else max(s["last_s"], t)
                pool.submit(write_png, d / f"{t:08.3f}.png", msg.width, msg.height, bytes(msg.data))
        return cb

    node = Node()
    for c in cams:
        if not node.subscribe(Image, c["topic"], on_image(c)):
            raise SystemExit(f"cannot subscribe to {c['topic']}")
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    while not stop.wait(0.2):
        pass
    with lock:
        closing.append(True)                        # callbacks submit nothing after this
        done = {k: dict(v) for k, v in stats.items()}
    pool.shutdown(wait=True)
    Path(out_dir, "capture.json").write_text(json.dumps(dict(rate_hz=rate, cameras=done, errors=errors), indent=1))
    print("capture: " + ", ".join(f"{k} {v['saved']} PNG of {v['received']} frames" for k, v in done.items()), flush=True)
    os._exit(1 if errors else 0)                    # skip gz-transport's thread teardown, which can abort


if __name__ == "__main__":
    main(*sys.argv[1:])
