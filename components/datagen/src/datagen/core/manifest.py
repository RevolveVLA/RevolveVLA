"""<out>/manifest.json (the resumable state of a run, plain JSON) and <out>/report.md.

manifest = {spec_hash, task, splits: {name: {sampled: [config], used: [{config, generate, record?}],
rejected: [{seed, step, reason}]}}, convert?: {episodes, result}, checks?: [{name, ok, reason}]}."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .errors import DatagenError
from .spec import DatagenSpec


class ManifestError(DatagenError):
    pass


class Manifest:
    def __init__(self, path: Path, data: dict, persist: bool = True):
        self.path, self.data, self.persist = path, data, persist

    @classmethod
    def open(cls, path: Path, spec: DatagenSpec, persist: bool = True) -> Manifest:
        if not path.exists():
            return cls(path, {"spec_hash": spec.hash, "task": spec.task, "splits": {}}, persist)
        data = json.loads(path.read_text())
        if data["spec_hash"] != spec.hash:
            raise ManifestError(f"{path}: made by another spec (hash {data['spec_hash']}, this one {spec.hash}); "
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


def report(spec: DatagenSpec, data: dict[str, Any]) -> str:
    """Short markdown summary: per split target vs done, then the checks."""
    lines = [f"# {spec.task} ({spec.robot}), spec {spec.hash}", "", "| split | target | used | rejected | recorded |",
             "|---|---|---|---|---|"]
    for s in spec.splits:
        rec = data["splits"].get(s.name, {"used": [], "rejected": []})
        recorded = sum(e.get("record", {}).get("ok", False) for e in rec["used"])
        lines.append(f"| {s.name} | {s.target} | {len(rec['used'])} | {len(rec['rejected'])} | {recorded} |")
    lines += ["", "| check | verdict | detail |", "|---|---|---|"]
    lines += [f"| {c['name']} | {'PASS' if c['ok'] else 'FAIL'} | {c['reason']} |" for c in data.get("checks", [])]
    return "\n".join(lines) + "\n"
