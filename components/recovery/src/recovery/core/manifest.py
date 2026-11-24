"""<out>/manifest.json: datagen's DATAGEN-v2.md 8.4 episode list (one entry per try of every job, merged in job order) with
a `recovery` block per entry:

  recovery = {kind: "phase", from_rollout: {run, set, rollout, try, scene, bag}, phase, trigger, reason, t_restart,
              restart_from, restart_state: {base, arm, object_joint}, reset: {mode, ...}, segment: null}

`result` / `bag` are the sim's files (under `<out>/sims/<slot>/`, relative to `out`); `reset`: `{mode: warm, ok: true, ...the
reset report}`, `{mode: relaunch}` (the sim was launched at the restart) or `{mode: relaunch, fallback: <the warm report that
missed>}`; `segment` is for recorders that keep only a span of the episode. `keep: successes` leaves out the tries the evaluator
failed. Derived from the job logs, never read back."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from datagen.core import episodes
from datagen.core import manifest as datagen_manifest

from .failures import Failure
from .jobs import resets
from .spec import RecoverySpec


def reset_of(done: Sequence[dict], attempt: int) -> dict:
    mine = [r for r in done if r["attempt"] == attempt]
    if not mine:
        return {"mode": "relaunch"}
    last = {k: v for k, v in mine[-1].items() if k != "attempt"}
    return last if last["ok"] else {"mode": "relaunch", "fallback": last}


def entries(out: Path, job_dir: Path, failure: Failure, spec: RecoverySpec) -> list[dict]:
    def rel(p: Path) -> str:
        return os.path.relpath(p, out)
    log, done, found = episodes.read(job_dir), resets(job_dir), []
    r = failure.restart
    for t in log["tries"]:
        sim = Path(t["sim"])
        result, bag = sim / f"episode_{t['episode']}.json", sim / f"episode_{t['episode']}_bag"
        if not result.is_file():
            continue
        passed = bool(json.loads(result.read_text())["passed"])
        if spec.keep == "successes" and not passed:
            continue
        found.append({
            "seed": log["seed"], "scene": rel(job_dir / "scene.yaml"), "robot": spec.robot, "task": spec.task,
            "episode": t["episode"], "result": rel(result), "bag": rel(bag) if bag.is_dir() else None, "passed": passed,
            "datagen": {"expert": t["expert"], "attempt": t["attempt"], "wall_s": t["wall_s"]},
            "recovery": {"kind": "phase",
                         "from_rollout": {"run": spec.eval_run, "set": failure.set, "rollout": failure.rollout,
                                          "try": failure.attempt, "scene": failure.scene, "bag": failure.bag},
                         "phase": failure.phase, "trigger": failure.trigger, "reason": failure.reason, "t_restart": r.t,
                         "restart_from": spec.phases.restart_from,
                         "restart_state": {"base": r.base, "arm": r.arm, "object_joint": r.object_joint},
                         "reset": reset_of(done, t["attempt"]), "segment": None}})
    return found


def build(out: Path, spec: RecoverySpec, jobs: Sequence[tuple[Failure, Path]], created: str | None = None) -> dict:
    """The manifest of the jobs `(failure, job dir)`, in this order (the caller makes it deterministic)."""
    found = [e for f, d in jobs for e in entries(out, d, f, spec)]
    return {"task": spec.task, "robot": spec.robot,
            "created": created or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "episodes": found,
            "counts": {"jobs": len(jobs), "attempted": len(found), "passed": sum(e["passed"] for e in found)}}


write, created = datagen_manifest.write, datagen_manifest.created
