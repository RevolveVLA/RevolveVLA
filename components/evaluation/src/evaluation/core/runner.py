"""The runner's chunk size against the checkpoint: a policy returns `chunk_size` actions per chunk (config.json), so the
runner cannot execute more than that."""
from __future__ import annotations

import json
from pathlib import Path

from .errors import EvalError
from .interfaces import ResolvedCkpt
from .spec import RunnerSpec


def check_chunk_size(runner: RunnerSpec, ckpt: ResolvedCkpt | None) -> int | None:
    """The ckpt's `chunk_size` (None: no ckpt or none stated). EvalError when `runner` asks for more actions per chunk than
    it holds (Rosetta drops the first of the `actions_per_chunk`, so `n_action_steps` are executed)."""
    config = Path(ckpt.path) / "config.json" if ckpt else None
    size = json.loads(config.read_text()).get("chunk_size") if config and config.is_file() else None
    if size is not None and runner.actions_per_chunk > size:
        raise EvalError(f"runner: n_action_steps {runner.n_action_steps} needs {runner.actions_per_chunk} actions per chunk "
                        f"(the first is dropped) but the checkpoint's chunk_size is {size}: use n_action_steps <= {size - 1}")
    return size
