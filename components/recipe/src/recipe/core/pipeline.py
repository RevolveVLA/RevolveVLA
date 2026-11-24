"""The recipe's stages: `resolve` (what the sources hold, nothing written), `check` (the plan: select, split, guard; report on
stdout, nothing written), `materialise` (check, then report.md and the output plugin's files in `out`). A recipe that is short of
what it asked for warns, or fails when `on_short: error` (its report is written either way, its selection only when whole)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Mapping, TextIO

from . import report
from .interfaces import Output, Source
from .leakage import Exclusion
from .plan import Inventory, Plan
from .spec import RecipeSpec

STAGES = ("resolve", "check", "materialise")


class Pipeline:
    def __init__(self, spec: RecipeSpec, sources: Mapping[str, Source], exclusion: Exclusion, out: Path,
                 output: Callable[[], Output] | None = None, stdout: TextIO | None = None):
        self.spec, self.sources, self.exclusion, self.out, self.output = spec, sources, exclusion, Path(out), output
        self.stdout = stdout or sys.stdout

    def say(self, text: str = ""):
        print(text, file=self.stdout)

    def resolve(self) -> bool:
        inv = Inventory.read(self.sources, self.exclusion)
        for name, items in inv.items.items():
            phases = {p: n for (s, p), n in inv.counts().items() if s == name}
            self.say(f"{name} ({inv.kinds[name]}): {len(items)} usable"
                     + (": " + ", ".join(f"{p or 'any'} {n}" for p, n in sorted(phases.items(), key=str)) if phases else "")
                     + (f"; refused {inv.refused[name]}" if inv.refused[name] else "")
                     + (f"; left out {dict(inv.skipped[name])}" if inv.skipped[name] else ""))
        return True

    def plan(self) -> Plan:
        return Plan.build(self.spec, Inventory.read(self.sources, self.exclusion))

    def check(self, plan: Plan | None = None) -> bool:
        plan = plan or self.plan()
        self.say(report.render(plan))
        problems = plan.problems(self.exclusion)
        for p in problems:
            self.say(f"FAIL {p}")
        if plan.short:
            self.say(f"{'FAIL' if self.spec.on_short == 'error' else 'WARN'}: short of what the recipe asked for")
        return not problems and not (plan.short and self.spec.on_short == "error")

    def materialise(self) -> bool:
        plan = self.plan()
        ok = self.check(plan)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "report.md").write_text(report.render(plan))
        if ok:
            for path in self.output().write(self.out, plan.to_dict()):
                self.say(f"wrote {path}")
        return ok

    def run(self, stage: str) -> bool:
        return {"resolve": self.resolve, "check": self.check, "materialise": self.materialise}[stage]()
