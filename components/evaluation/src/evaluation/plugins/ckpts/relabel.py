"""Serve a checkpoint whose cameras the robot contract names or sizes differently (port of 55dd408).

A checkpoint trained elsewhere can name its cameras differently (lerobot-train --rename_map, recorded in its
train_config.json) and label its image features with shapes the frames never had (e.g. the base model's 256x256);
LeRobot's async server looks each incoming camera up by name and resizes it to that label. If the image features differ
from what the robot contract sends, `<out>/ckpt/` gets a copy (files hard-linked) whose config.json has the contract's
camera names and frame shapes, same order, plus served_ckpt.json. Wraps any source: the pipeline gets the served one."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import yaml
from revolve_common import specs

from ...core import sets
from ...core.errors import EvalError
from ...core.interfaces import CkptSource, ResolvedCkpt
from ...core.registry import REGISTRY


def served_ckpt(ckpt: Path, out: Path, contract: Path):
    """-> (dir to serve, {checkpoint key: {key, shape}} of the changes, or None and `ckpt` itself)."""
    cfg = json.loads((ckpt / "config.json").read_text())
    train = ckpt / "train_config.json"
    back = {v: k for k, v in (json.loads(train.read_text()).get("rename_map") or {}).items()} if train.is_file() else {}
    obs = yaml.safe_load(contract.read_text())["observations"]
    feats, changes = {}, {}
    for k, f in cfg["input_features"].items():
        key, shape = back.get(k, k), f["shape"]
        if f["type"] == "VISUAL":
            if key not in obs:
                raise EvalError(f"{ckpt}: camera {k}" + (f" (trained as {key})" if key != k else "") +
                                f" is not an observation of {contract}")
            size = [a["resize"] for a in obs[key].get("apply") or [] if isinstance(a, dict) and "resize" in a]
            shape = [shape[0], *size[-1]] if size else shape          # resize [h, w]; the last one sets the size
        feats[key] = dict(f, shape=shape)
        if (key, shape) != (k, f["shape"]):
            changes[k] = dict(key=key, shape=shape)
    if not changes:
        return ckpt, None
    out.mkdir(parents=True, exist_ok=True)
    for p in ckpt.iterdir():
        if p.is_file() and p.name not in ("config.json", "served_ckpt.json"):
            (out / p.name).unlink(missing_ok=True)
            try:
                os.link(p, out / p.name)
            except OSError:
                shutil.copy2(p, out / p.name)
    (out / "config.json").write_text(json.dumps(dict(cfg, input_features=feats), indent=1))
    (out / "served_ckpt.json").write_text(json.dumps(dict(source=str(ckpt), changes=changes), indent=1))
    return out, changes


@REGISTRY.register("ckpt", "relabel")
class Relabel(CkptSource):
    """`params["of"]`: the source to serve (the CLI wraps the yaml's source; not selectable as a yaml `source:`)."""
    relabel = False

    def __init__(self, spec, params):
        if "of" not in params:
            raise EvalError("ckpt relabel wraps a source: it is not a yaml `source:`")
        self.spec, self.source = spec, params["of"]

    def resolve(self, out):
        ckpt = self.source.resolve(out)
        root = sets.repo_root(self.spec, out)
        robot = specs.load(root / "specs" / "tasks" / self.spec.train.task / "task.yaml", self.spec.train.robot).robot
        contract = root / "specs" / "robots" / robot.name / robot.contract
        path, changes = served_ckpt(Path(ckpt.path), out / "ckpt", contract)
        return ckpt if changes is None else ResolvedCkpt(str(path), ckpt.origin, changes)
