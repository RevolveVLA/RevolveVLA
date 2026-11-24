"""The policy service's wire format and its client: one POST /act per observation, numpy `.npz` both ways (no pickle).

Request: `camera1..3` (H, W, 3) uint8 RGB, `state` (D,) float32, `task` (a string). Reply: `actions` (chunk, A) float32, the
checkpoint's whole chunk, unnormalised. GET /health -> JSON {ready, chunk, state_dim, action_dim, ...}."""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from typing import Mapping

import numpy as np

ACT, HEALTH = "/act", "/health"


def pack(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **{k: np.asarray(v) for k, v in arrays.items()})
    return buf.getvalue()


def unpack(data: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


class PolicyClient:
    """The client side of the service at `url`."""

    def __init__(self, url: str, timeout_s: float = 60.0):
        self.url, self.timeout_s = url.rstrip("/"), timeout_s

    def health(self) -> dict:
        with urllib.request.urlopen(self.url + HEALTH, timeout=self.timeout_s) as r:
            return json.loads(r.read())

    def act(self, images: Mapping[str, np.ndarray], state: np.ndarray, task: str) -> np.ndarray:
        """The chunk for one observation: (chunk, A)."""
        body = pack(**images, state=np.asarray(state, dtype=np.float32), task=np.array(task))
        req = urllib.request.Request(self.url + ACT, body, {"Content-Type": "application/octet-stream"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return unpack(r.read())["actions"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"policy service: {e.code} {e.read().decode(errors='replace')[:500]}") from e
