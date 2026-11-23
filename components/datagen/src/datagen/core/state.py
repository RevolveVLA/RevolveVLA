"""<out>/state.json (the resumable state of a run, plain JSON) and <out>/report.md. (`manifest.json` is DATAGEN-v2.md 8.4's
episode list, core/manifest.py.)

state = {spec_hash, task, splits: {name: {sampled: [config], used: [{config, generate}], rejected: [{seed, step, reason}]}},
checks?: [{name, ok, reason}]}. `generate` is the Result of a scene: its `realised` start, `paths.scene` (the instance)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .errors import DatagenError
from . import stats
from .spec import DatagenSpec


class StateError(DatagenError):
    pass


class State:
    def __init__(self, path: Path, data: dict, persist: bool = True):
        self.path, self.data, self.persist = path, data, persist

    @classmethod
    def open(cls, path: Path, spec: DatagenSpec, persist: bool = True) -> State:
        if not path.exists():
            return cls(path, {"spec_hash": spec.hash, "task": spec.task, "splits": {}}, persist)
        data = json.loads(path.read_text())
        if data["spec_hash"] != spec.hash:
            raise StateError(f"{path}: made by another spec (hash {data['spec_hash']}, this one {spec.hash}); "
                                "use another `out` or delete the run")
        return cls(path, data, persist)

    def split(self, name: str) -> dict:
        return self.data["splits"].setdefault(name, {"sampled": [], "used": [], "rejected": []})

    def save(self):
        if self.persist:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1))
            os.replace(tmp, self.path)


def _num(pair) -> str:
    return "-" if pair is None else f"{pair[0]:.3f}"


def report(spec: DatagenSpec, data: dict[str, Any]) -> str:
    """Short markdown: per split target vs done, the cell balance of stratified splits, the min distances, the checks."""
    lines = [f"# {spec.task} ({spec.robot}), spec {spec.hash}", "", "| split | target | used | rejected |", "|---|---|---|---|"]
    for s in spec.splits:
        p = stats.progress(s, stats.rec(data, s.name))
        lines.append(f"| {s.name} | {s.target} | {p['used']} | {p['rejected']} |")
    for s in spec.splits:
        if cells := stats.cells(spec, s, stats.rec(data, s.name)):
            lines += ["", f"cells of {s.name} (bins {', '.join(f'{k} {n}' for k, n in cells['bins'].items())})", "",
                      "| cell | pool | rejected | used | realised | exhausted |", "|---|---|---|---|---|---|"]
            lines += [f"| {r['cell']} | {r['pool']} | {r['rejected']} | {r['used']} | "
                      f"{'-' if r['realised'] is None else r['realised']} | {'yes' if r['exhausted'] else ''} |"
                      for r in cells["rows"]]
    if any(s.min_distance for s in spec.splits):
        lines += ["", "| split | from | min distance | sampled | realised |", "|---|---|---|---|---|"]
        for s in (s for s in spec.splits if s.min_distance):
            d = stats.distances(spec, s, data)
            lines.append(f"| {s.name} | {d['from']} | {d['value']}{' (realised)' if d['realised_required'] else ''} | "
                         f"{_num(d['sampled'])} | {_num(d['realised'])} |")
    lines += ["", "| check | verdict | detail |", "|---|---|---|"]
    lines += [f"| {c['name']} | {'PASS' if c['ok'] else 'FAIL'} | {c['reason']} |" for c in data.get("checks", [])]
    return "\n".join(lines) + "\n"
