"""Backend local: lerobot-train in this container (revolve-training with the GPU, started by launch.py)."""
from __future__ import annotations

import os

import train as T


def run(p: dict, settings: dict, job: str, on_line=None, stream=T.stream) -> dict:
    """-> {"rc", "ckpt" (None: the final checkpoint under p["lerobot_out"])}."""
    T.refuse_token()
    T.progress(phase="train", backend="local")
    return dict(rc=stream(T.argv(p), dict(os.environ), on_line), ckpt=None)
