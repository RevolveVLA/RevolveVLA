"""Checkpoint on disk: `ckpt: {source: local, path: runs/ckpts/<name>}` (a dir under runs/, which the containers mount)."""
from __future__ import annotations

import hashlib
from pathlib import Path

from ...core import sets
from ...core.errors import EvalError
from ...core.interfaces import CkptSource, ResolvedCkpt
from ...core.registry import REGISTRY


def origin_of(ckpt: Path, root: Path, **extra) -> dict:
    """What identifies a checkpoint dir in results.json: its path, config.json hash and weights size."""
    weights = ckpt / "model.safetensors"
    return {**extra, "path": str(ckpt.relative_to(root)),
            "config_sha256": hashlib.sha256((ckpt / "config.json").read_bytes()).hexdigest(),
            "model_bytes": weights.stat().st_size if weights.is_file() else None}


@REGISTRY.register("ckpt", "local")
class LocalCkpt(CkptSource):
    def __init__(self, spec, params):
        if "path" not in params:
            raise EvalError("ckpt local: needs `path`")
        self.spec, self.path = spec, Path(params["path"])

    def resolve(self, out):
        root = sets.repo_root(self.spec, out)
        ckpt = (self.path if self.path.is_absolute() else root / self.path).resolve()
        if not ckpt.is_relative_to(root / "runs"):          # the containers see the repo (ro) and runs/ (rw) only
            raise EvalError(f"ckpt {self.path}: not under {root / 'runs'}")
        if not (ckpt / "config.json").is_file():
            raise EvalError(f"ckpt {self.path}: no config.json in {ckpt}")
        return ResolvedCkpt(str(ckpt), origin_of(ckpt, root, source="local"))
