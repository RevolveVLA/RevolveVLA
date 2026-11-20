"""Whole-run checks over the manifest, registered by name (PRD 06 adds split / min-distance / coverage checks)."""
from __future__ import annotations

from typing import Any, Mapping

from .interfaces import Check, Result
from .registry import register


@register("check", "target")
class TargetReached(Check):
    """Every split has its target number of accepted (and, when it records, recorded) episodes."""
    name = "target"

    def __init__(self, spec, params: Mapping[str, Any]):
        self.spec = spec

    def run(self, manifest: Mapping[str, Any]) -> Result:
        short = []
        for s in self.spec.splits:
            used = manifest["splits"].get(s.name, {}).get("used", [])
            done = sum(e.get("record", {}).get("ok", False) for e in used) if s.record else len(used)
            if done < s.target:
                short.append(f"{s.name} {done}/{s.target}")
        return Result(not short, ", ".join(short))
