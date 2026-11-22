"""Training source `legacy_layouts`: a run made by the pre-datagen generator (`layouts.json` + `turn_0/data/manifest.json`,
e.g. runs/fridge_base50), read as a datagen manifest in memory (nothing is written: a fake manifest under runs/datagen would
be trusted by a later datagen resume).

`layouts.json` splits.train.pool -> `sampled` (cells = the bins' row-major index), each seed of the legacy manifest with an
accepted episode -> `used`: realised start from `plans/seed_N` (datagen's `realised_start`), `generate.paths.check` = the
accepted attempt's check dir (what a rollout starts from). The legacy ranges, bins, strategy, target and task must equal the
training spec's, else EvalError."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from datagen.core.interfaces import Config
from datagen.core.spec import Range

from ...core.errors import EvalError
from ...core.interfaces import TrainingRun, TrainingSource
from ...core.registry import REGISTRY

DATA = "turn_0/data"
SPLIT = "train"


@REGISTRY.register("training", "legacy_layouts")
class LegacyLayouts(TrainingSource):
    def __init__(self, spec, params):
        if not isinstance(params.get("run"), str):
            raise EvalError("training legacy_layouts needs `run:` (the legacy run dir, e.g. runs/fridge_base50)")
        self.spec, self.run = spec, Path(params["run"])

    def load(self, root: Path) -> TrainingRun:
        base = self.run if self.run.is_absolute() else root / self.run
        layouts, manifest = _json(base / "layouts.json"), _json(base / DATA / "manifest.json")
        split = layouts.get("splits", {}).get(SPLIT)
        if split is None:
            raise EvalError(f"legacy training run {base}: layouts.json has no splits.{SPLIT}")
        sizes = self._validate(base, split, manifest)
        pool = {p["seed"]: Config(SPLIT, p["seed"], _cell(p["cell"], sizes), p["layout"]) for p in split["pool"]}
        used, missing = [], []
        for seed, entry in sorted(manifest["seeds"].items(), key=lambda kv: int(kv[0])):
            if not (episode := next((e for e in entry.get("episodes", ()) if e.get("accepted")), None)):
                continue
            seed = int(seed)
            if seed not in pool:
                raise EvalError(f"legacy training run {base}: seed {seed} is not in layouts.json splits.{SPLIT}.pool")
            plan, check = base / DATA / entry["plan"]["dir"], base / DATA / episode["check"]
            missing += [d for d in (plan, check) if not d.is_dir()]
            used.append({"config": pool[seed].to_dict(), "record": {"ok": True},
                         "generate": {"ok": True, "realised": _realised(plan, episode["attempt_id"]),
                                      "paths": {"check": str(self.run / DATA / episode["check"]),
                                                "plan": str(self.run / DATA / entry["plan"]["dir"])}}})
        if missing:
            raise EvalError(f"legacy training run {base}: {len(missing)} start dirs are missing, first {missing[0]}")
        data = {"spec_hash": self.spec.train.hash, "task": manifest["task"],
                "splits": {SPLIT: {"sampled": [c.to_dict() for c in pool.values()], "used": used, "rejected": []}}}
        return TrainingRun(base / DATA / "manifest.json", data)

    def _validate(self, base: Path, split: dict, manifest: dict) -> list[int]:
        """Cells per dim (the bins in the legacy space's order); EvalError when the run is not the training spec's."""
        train, want = self.spec.train, self.spec.train.split(SPLIT)
        dims = {k: (d.lo, d.hi) for k, d in train.dims().items() if isinstance(d, Range)}
        space = {k: tuple(v) for k, v in split["space"].items()}
        bins = {k: split["bins"].get(k, 1) for k in space}
        diffs = [f"{what}: legacy {have}, spec {exp}" for what, have, exp in (
            ("task", manifest.get("task"), train.task), ("strategy", split["strategy"], want.sampler.strategy),
            ("target", split["target"], want.target), ("bins", bins, {k: want.sampler.bins.get(k, 1) for k in space}),
            ("space", space, dims)) if not _same(have, exp)]
        if diffs:
            raise EvalError(f"legacy training run {base} is not the training spec's run ({train.out}): " + "; ".join(diffs))
        return list(bins.values())


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return math.isclose(a, b, abs_tol=1e-9) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else a == b


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise EvalError(f"legacy training run: cannot read {path}: {e}") from e


def _cell(index: int, sizes: list[int]) -> tuple[int, ...]:
    """The legacy cell index (row-major over the dims' bins) as one bin per dim."""
    digits = []
    for n in reversed(sizes):
        index, d = divmod(index, n)
        digits.append(d)
    return tuple(reversed(digits))


def _realised(plan: Path, attempt: int) -> dict:
    from datagen.plugins.automoma.adapter import realised_start         # lazy: needs numpy
    try:
        return realised_start(plan, attempt)
    except (OSError, ValueError, KeyError) as e:
        raise EvalError(f"legacy training run: cannot read the realised start of {plan}: {e}") from e
