"""report.md of a recovery run: per phase the failures found, the ones that cannot be reproduced (and why), the jobs selected,
the episodes recorded and how much of the quota they fill."""
from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from .failures import Failure
from .mining import LABEL
from .spec import RecoverySpec


def table(spec: RecoverySpec, failures: Sequence[Failure], selected: Sequence[str], entries: Sequence[Mapping]) -> list[dict]:
    """One row per phase of the quota and of the failures: failures, unreproducible, selected, recorded (the manifest's
    entries: `keep` decides whether failed tries are in), passed, quota, fill (recorded / quota; None without a quota)."""
    phases = sorted({*spec.phases.quota, *(f.phase for f in failures)}, key=lambda p: (p not in spec.phases.quota, p))
    rows = []
    for p in phases:
        mine = [f for f in failures if f.phase == p]
        got = [e for e in entries if e["recovery"]["phase"] == p]
        quota = spec.phases.quota.get(p)
        rows.append({"phase": p, "failures": len(mine), "unreproducible": sum(not f.restart.reproducible for f in mine),
                     "selected": sum(f.id in selected for f in mine), "recorded": len(got),
                     "passed": sum(e["passed"] for e in got), "quota": quota,
                     "fill": min(1.0, len(got) / quota) if quota else None})
    return rows


def render(spec: RecoverySpec, failures: Sequence[Failure], selected: Sequence[str], counts: Mapping[str, int],
           entries: Sequence[Mapping], jobs: Mapping[str, Mapping], mining: Mapping | None = None) -> str:
    pct = lambda v: "-" if v is None else f"{100 * v:.0f}%"
    p = spec.phases
    lines = [f"# Recovery {spec.hash}: {spec.task} ({spec.robot})", "",
             f"From {spec.eval_run}: " + ", ".join(f"{n} {LABEL.get(k, k)}" for k, n in sorted(counts.items())) + " tries. "
             f"Restart from {p.restart_from}, {p.per_failure} per failure, keep {spec.keep}, reset {spec.run.reset}.", "",
             "| phase | failures | unreproducible | selected | recorded | passed | quota | fill |", "|---|---|---|---|---|---|---|---|"]
    if mining:
        lines[4:4] = [f"Mined sets: {', '.join(mining['mined']) or 'none'}; kept unseen: {', '.join(mining['held_out']) or 'none'}."
                      + ("" if mining["mined"] else " Nothing is recorded: every set of the run is held-out (name the sets to mine in "
                         "`from.sets`)."), ""]
    for r in table(spec, failures, selected, entries):
        quota = "-" if r["quota"] is None else r["quota"]
        lines.append(f"| {r['phase']} | {r['failures']} | {r['unreproducible']} | {r['selected']} | {r['recorded']} | "
                     f"{r['passed']} | {quota} | {pct(r['fill'])} |")
    why = Counter(f"{f.phase}: {f.restart.reason}" for f in failures if not f.restart.reproducible)
    if why:
        lines += ["", "| unreproducible | failures |", "|---|---|"] + [f"| {k} | {n} |" for k, n in sorted(why.items())]
    status = Counter(j["status"] for j in jobs.values())
    if status:
        lines += ["", "Jobs: " + ", ".join(f"{n} {k}" for k, n in sorted(status.items()))]
    for k, j in sorted(jobs.items()):
        if j["status"] == "error":
            lines.append(f"- {k}: {j['reason']}")
    return "\n".join(lines) + "\n"
