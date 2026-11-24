"""A checkpoint served over HTTP, inside the training image (LeRobot, torch, a GPU), for the policy clients of the new simulator:
`python -m evaluation.policy_service --ckpt DIR --port P [--device cuda]` loads the policy and its pre / post processors, warms them
with a dummy observation, then answers `policy_wire`'s requests with the unnormalised chunk. One request at a time (one model).
Binds 127.0.0.1 only; the payload is numpy `.npz` without pickle."""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from .policy_wire import ACT, HEALTH, pack, unpack


class Model(Protocol):
    info: dict

    def __call__(self, images: Mapping[str, np.ndarray], state: np.ndarray, task: str) -> np.ndarray: ...


class LeRobotModel:
    """A LeRobot policy (`predict_action_chunk`) with the checkpoint's processors, as LeRobot's async server runs it."""

    def __init__(self, ckpt: str, device: str = "cuda"):
        import torch
        from lerobot.policies import get_policy_class, make_pre_post_processors
        self.torch, self.device = torch, device
        kind = json.loads((Path(ckpt) / "config.json").read_text())["type"]
        self.policy = get_policy_class(kind).from_pretrained(ckpt)
        self.policy.to(device).eval()
        override = {"device_processor": {"device": device}}
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=ckpt,
            preprocessor_overrides={**override, "rename_observations_processor": {"rename_map": {}}},
            postprocessor_overrides=override)
        cfg = self.policy.config
        self.info = {"ckpt": ckpt, "device": device, "chunk": cfg.chunk_size, "action_dim": cfg.action_feature.shape[0],
                     "cameras": sorted(k.split(".")[-1] for k in cfg.image_features)}

    def __call__(self, images, state, task):
        t = self.torch
        obs = {"observation.state": t.from_numpy(np.asarray(state, np.float32)),
               **{f"observation.images.{k}": t.from_numpy(np.ascontiguousarray(v)).permute(2, 0, 1).float() / 255
                  for k, v in images.items()}, "task": task}
        with t.inference_mode():
            chunk = self.policy.predict_action_chunk(self.pre(obs))
            acts = [self.post(chunk[:, i, :]) for i in range(chunk.shape[1])]
        return t.stack(acts, dim=1).squeeze(0).float().cpu().numpy()


class Service:
    """A model behind a lock, with its request counters."""

    def __init__(self, model: Model):
        self.model, self.lock, self.requests, self.infer_s = model, threading.Lock(), 0, 0.0

    def act(self, images, state, task) -> np.ndarray:
        with self.lock:
            t0 = time.perf_counter()
            out = np.asarray(self.model(images, state, task), dtype=np.float32)
            self.requests, self.infer_s = self.requests + 1, self.infer_s + time.perf_counter() - t0
        return out

    def health(self) -> dict:
        return {"ready": True, **self.model.info, "requests": self.requests,
                "mean_s": self.infer_s / self.requests if self.requests else None}


def handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, kind: str = "application/octet-stream"):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):                                   # noqa: N802
            if self.path == HEALTH:
                self._send(200, json.dumps(service.health()).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):                                  # noqa: N802
            if self.path != ACT:
                return self._send(404, b"not found", "text/plain")
            try:
                req = unpack(self.rfile.read(int(self.headers["Content-Length"])))
                task, state = str(req.pop("task")), req.pop("state")
                self._send(200, pack(actions=service.act(req, state, task)))
            except Exception:
                traceback.print_exc()
                self._send(500, traceback.format_exc().splitlines()[-1].encode(), "text/plain")

        def log_message(self, *args):
            pass
    return Handler


def serve(service: Service, port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), handler(service))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="evaluation.policy_service")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--state-dim", type=int, required=True, help="the state vector's size (config.json's label may differ)")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    model = LeRobotModel(a.ckpt, a.device)
    c = json.loads((Path(a.ckpt) / "config.json").read_text())
    images = {k.split(".")[-1]: np.zeros((480, 480, 3), np.uint8) for k in c["input_features"] if "images" in k}
    model(images, np.zeros(a.state_dim, np.float32), "warm up")
    print(f"policy service: loaded and warm in {time.perf_counter() - t0:.1f} s: {model.info}", flush=True)
    srv = serve(Service(model), a.port)
    print(f"policy service: ready on {a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
