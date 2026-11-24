"""report.md: asked vs got per source x phase x split, what the guard refused, what the sources left out."""
from __future__ import annotations

from .plan import Plan


def render(plan: Plan) -> str:
    s = plan.spec
    names = list(plan.splits)
    lines = [f"# Recipe {s.hash}", "", f"Unit: {s.unit}" + (f" ({s.fps} fps)" if s.unit == "frames" else "") + f". Seed {s.seed}. "
             f"Splits (by scene): " + ", ".join(f"{n} {f:g}" for n, f in s.splits.items()) + ".", "",
             "| source | kind | phase | asked | got | available | " + " | ".join(names) + " | |",
             "|---|---|---|---|---|---|" + "---|" * len(names) + "---|"]
    for r in plan.rows:
        got = plan.got(r)
        lines.append(f"| {r.source} | {r.kind} | {r.phase or '-'} | {r.asked} | {r.got} | {r.available} | "
                     + " | ".join(str(got[n]) for n in names) + f" | {'SHORT' if r.short else ''} |")
    total = {n: sum(plan.got(r)[n] for r in plan.rows) for n in names}
    lines += ["", "Got per split: " + ", ".join(f"{n} {total[n]} ({len(plan.splits[n])} items, "
                                                 f"{len({i.scene for i in plan.splits[n]})} scenes)" for n in names), ""]
    if refused := {n: k for n, k in plan.inventory.refused.items() if k}:
        lines += ["Refused by the leakage guard (`exclude`): " + ", ".join(f"{n} {k}" for n, k in refused.items()), ""]
    for n, why in plan.inventory.skipped.items():
        if why:
            lines.append(f"Left out of {n}: " + ", ".join(f"{k} {v}" for k, v in sorted(why.items())))
    if short := plan.short:
        lines += ["", "Short: " + ", ".join(f"{r.source}/{r.phase or 'any'} ({r.got} of {r.asked})" for r in short)]
    return "\n".join(lines).rstrip() + "\n"
