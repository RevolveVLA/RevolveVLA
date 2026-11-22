"""`<out>/results.json` (the run's state: spec hash, resolved ckpt, per set the configs, check results and rollouts with
their tries, plus the metrics derived from them) and `<out>/report.md`.

A rollout entry is `{seed, passed, tries: [{dir, restored_at, passed, detail, termination, record}]}`: `passed` = the last
try's judge verdict, a try's `record` = the rollout plugin's record (phase verdict), `detail` = the judge's, `termination`
= why the sim run ended (core/termination.py). A set's `rollout_policy` = the limits that applied. The file is
written atomically after every committed rollout; a different spec hash refuses an existing file."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .errors import EvalError
from .termination import REASONS, termination

NO_PHASE = "success_rule"           # the phases passed but the judge failed (C1 / C2 on the state log)
ERROR = "stack_error"


class ResultsError(EvalError):
    pass


def failure(first_try: Mapping) -> str | None:
    """Why a first try failed: the failed phase, NO_PHASE, or ERROR (no verdict); None when it passed."""
    if first_try["passed"]:
        return None
    rec = first_try["record"]
    return ERROR if rec.get("error") else rec.get("failed_phase") or NO_PHASE


def metrics(rollouts: Mapping[str, Mapping]) -> dict[str, Any]:
    """first_try_success / success_with_retries (fractions of the rollouts), failures per first-try reason, and how
    every try ended (`terminations`: count per reason, over all tries)."""
    n, first = len(rollouts), [r["tries"][0] for r in rollouts.values()]
    k1, k = sum(t["passed"] for t in first), sum(r["passed"] for r in rollouts.values())
    return {"rollouts": n, "first_try_passed": k1, "first_try_success": k1 / n if n else None,
            "passed_with_retries": k, "success_with_retries": k / n if n else None,
            "first_try_failures": dict(Counter(f for t in first if (f := failure(t)))),
            "errors": sum(bool(r["tries"][-1]["record"].get("error")) for r in rollouts.values()),
            "terminations": dict(Counter(t.get("termination") or termination(t["record"])
                                         for r in rollouts.values() for t in r["tries"]))}


def pooled(sets: Mapping[str, Mapping], names) -> dict[str, Any]:
    """metrics over the rollouts of several sets together."""
    return metrics({f"{n}/{k}": r for n in names for k, r in sets.get(n, {}).get("rollouts", {}).items()})


def config_rows(block: Mapping) -> list[dict]:
    """Per config: seed, cell, values and its rollouts' first-try / final passes."""
    rolls = block.get("rollouts", {})
    rows = []
    for c in block.get("configs", ()):
        mine = [r for r in rolls.values() if r["seed"] == c["seed"]]
        rows.append({**c, "rollouts": len(mine), "first_try_passed": sum(r["tries"][0]["passed"] for r in mine),
                     "passed": sum(r["passed"] for r in mine)})
    return rows


class Results:
    def __init__(self, path: Path, data: dict):
        self.path, self.data = path, data

    @classmethod
    def open(cls, path: Path, spec_hash: str) -> Results:
        if not path.exists():
            return cls(path, {"spec_hash": spec_hash, "ckpt": None, "sets": {}})
        data = json.loads(path.read_text())
        if data["spec_hash"] != spec_hash:
            raise ResultsError(f"{path.parent} holds the results of another spec (hash {data['spec_hash']}, this one "
                               f"{spec_hash}): use another `out` or delete it")
        return cls(path, data)

    def set(self, name: str) -> dict:
        return self.data["sets"].setdefault(name, {})

    def rollouts(self, name: str) -> dict:
        return self.set(name).setdefault("rollouts", {})

    def commit(self, name: str, key: str, entry: Mapping):
        block = self.set(name)
        block.setdefault("rollouts", {})[key] = entry
        block["metrics"] = metrics(block["rollouts"])
        self.save()

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        os.replace(tmp, self.path)

    def report(self, judge: str, names, kinds: Mapping[str, str] | None = None) -> str:
        """The report.md text: ckpt origin, then per set its summary, checks, first-try failure reasons and configs; with
        several sets the pooled row and, with `kinds` {set: seen|new}, the seen-vs-new gap."""
        d = self.data
        ckpt = d["ckpt"]
        lines = [f"# Evaluation {d['spec_hash']}", "", f"Judge {judge}. Checkpoint: "
                 + (f"{ckpt['path']} ({json.dumps(ckpt['origin'])})" if ckpt else "none (policy brings its own actions)"), ""]
        pct = lambda v: "-" if v is None else f"{100 * v:.0f}%"
        row = lambda m: (f"{m['rollouts']} | {m['first_try_passed']} ({pct(m['first_try_success'])}) | "
                         f"{m['passed_with_retries']} ({pct(m['success_with_retries'])}) | {m['errors']}")
        head = ["| rollouts | first try | with retries | stack errors |", "|---|---|---|---|"]
        for n in names:
            block = d["sets"].get(n, {})
            m = block.get("metrics") or metrics(block.get("rollouts", {}))
            lines += [f"## {n}" + (f" ({kinds[n]})" if kinds and n in kinds else ""), "", *head, f"| {row(m)} |", ""]
            if checks := block.get("checks"):
                lines += ["| check | result |", "|---|---|"]
                lines += [f"| {c} | {'PASS' if r['ok'] else 'FAIL'}{' ' + r['reason'] if r.get('reason') else ''} |"
                          for c, r in checks.items()] + [""]
            if m["first_try_failures"]:
                lines += ["| first-try failure | count |", "|---|---|"]
                lines += [f"| {p} | {k} |" for p, k in sorted(m["first_try_failures"].items())] + [""]
            if m.get("terminations"):
                lines += ["| termination | tries |", "|---|---|"]
                lines += [f"| {r} | {m['terminations'][r]} |" for r in REASONS if r in m["terminations"]] + [""]
            if pol := block.get("rollout_policy"):
                lines += ["Rollout limits: " + ", ".join(f"{k} {v}" for k, v in pol.items()), ""]
            if rows := config_rows(block):
                lines += ["| seed | cell | values | first try | passed |", "|---|---|---|---|---|"]
                lines += [f"| {r['seed']} | {','.join(map(str, r['cell']))} | "
                          f"{', '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}' for k, v in r['values'].items())} | "
                          f"{r['first_try_passed']}/{r['rollouts']} | {r['passed']}/{r['rollouts']} |" for r in rows] + [""]
        if len(names) > 1:
            groups = {"all sets": list(names)} | {k: [n for n in names if kinds[n] == k] for k in ("seen", "new")
                                                  if kinds and any(kinds.get(n) == k for n in names)}
            ms = {g: pooled(d["sets"], ns) for g, ns in groups.items()}
            lines += ["## all sets", "", "| group | " + head[0][1:], "|---" + head[1], *(f"| {g} | {row(m)} |" for g, m in ms.items())]
            if "seen" in ms and "new" in ms:
                gap = lambda key: "-" if None in (ms["seen"][key], ms["new"][key]) else f"{100 * (ms['seen'][key] - ms['new'][key]):+.0f} pts"
                lines.append(f"| gap seen - new | | {gap('first_try_success')} | {gap('success_with_retries')} | |")
            lines.append("")
        return "\n".join(lines)
