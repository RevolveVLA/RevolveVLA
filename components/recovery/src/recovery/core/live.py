"""The standalone live run: the policy of an evaluation spec rolls out its sets under a live recorder (DAgger) and the
takeovers become manifest entries. The rollouts are the evaluation component's (its Pipeline, its `sim` rollout, in `<out>/live/`):
only the policy is wrapped (`LivePolicy`: its client gets the recorder's driver, `recovery.live_client`), so retries, workers,
warm resets and resumes are the evaluation's own.

`<out>/live/` = an evaluation run (results.json, report.md, `<set>/<seed>_sim/` with every episode's result, bag and sidecar,
see core/flags.py) + `summary.json`: {policy_only_success, takeovers, reasons, recovered, expert_frames_s, episodes, wall_s}
(a line of it is logged). The manifest entries (core/manifest.py shape, `recovery.kind` "dagger") are derived from that run,
never read back: one per episode with a takeover (a policy-only success is the evaluation's rollout)."""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

from datagen.core.spec import SpecError

from . import flags, leakage
from .spec import RecoverySpec

LIVE = "live"
DRIVER = "recovery.live_client:make"
RECOVERY_SRC = "/repo/components/recovery/src"          # the client container's view of this package (repo = /repo)
STAGES = ("sample", "rollout", "report")                # of the evaluation pipeline


class LivePolicy:
    """The evaluation policy `inner` under the live recorder: same name and needs, its client args add the driver (the spec,
    the phases the recorder wants no takeover in, the policy's own perturbation for the test policies). Only a policy whose
    client streams actions can be halted: not `expert` (a closed-loop generator)."""

    def __init__(self, inner, spec: RecoverySpec, recorder, root: Path):
        if inner.name == "expert":
            raise SpecError("policy expert drives closed-loop and cannot be halted: a live recorder needs a policy whose "
                            "client streams actions (oracle, frozen, noisy_expert, a served policy)")
        self.inner, self.spec, self.recorder, self.root = inner, spec, recorder, root
        self.name, self.needs_ckpt, self.client_only = inner.name, inner.needs_ckpt, inner.client_only

    def client_args(self, item) -> list[str]:
        payload = {"spec": self.spec.raw, "closed": sorted(self.recorder.closed(flags.tally(self.root))),
                   "perturb": getattr(self.inner, "perturbation", lambda item: None)(item)}
        return [*self.inner.client_args(item), "--instance", str(item.scene), "--driver", DRIVER,
                "--driver-args", json.dumps(payload)]


def eval_plugins(spec: RecoverySpec, eval_spec, recorder, root: Path):
    """The evaluation's plugins for `eval_spec`, its policy under the recorder, its client container able to import this package."""
    importlib.import_module("recovery.plugins.policies")            # frozen, noisy_expert: policy plugins of the evaluation registry
    from evaluation.cli import build_plugins
    from evaluation.core.registry import REGISTRY
    REGISTRY.extend_path()                                          # $EVAL_PLUGIN_PATH
    plugins = build_plugins(eval_spec, ("sample", "rollout"))
    plugins.policy = LivePolicy(plugins.policy, spec, recorder, root)
    if hasattr(plugins.rollout, "client_src"):
        plugins.rollout.client_src = (*plugins.rollout.client_src, RECOVERY_SRC)
    return plugins


def load_eval(spec: RecoverySpec, root: Path):
    """The evaluation spec `from.eval` names (its `train:` path relative to `root`)."""
    if not spec.eval_spec:
        raise SpecError(f"from.eval: a live run rolls out an evaluation spec (a .yaml file), got {spec.eval_run!r}")
    from evaluation.core.spec import load
    return load(spec.eval_run, root)


def live_eval_spec(spec: RecoverySpec, eval_spec, out: Path):
    """`eval_spec` restricted to the sets to roll out, into `<out>/live`, run as the recovery spec says (leakage checked)."""
    run, reserved = leakage.sets_of(eval_spec, spec.eval_sets, spec.heldout)
    leakage.check(eval_spec, run, reserved)
    from evaluation.core.spec import RunSpec
    return dataclasses.replace(eval_spec, out=str(out / LIVE), sets={n: eval_spec.sets[n] for n in run},
                               run=RunSpec(spec.run.workers, spec.run.headless, spec.run.reset))


class LiveRun:
    """Rolls the evaluation spec out under the recorder: `pipeline` is the evaluation's (a callable `stages -> bool`)."""

    def __init__(self, spec: RecoverySpec, pipeline: Callable[[Sequence[str]], bool], out: Path,
                 log: Callable[[str], None] = print):
        self.spec, self.pipeline, self.out, self.log = spec, pipeline, out, log

    def run(self) -> bool:
        t0 = time.monotonic()
        ok = self.pipeline(STAGES)
        line = {**summary(self.out), "wall_s": round(time.monotonic() - t0, 1)}
        (self.out / LIVE).mkdir(parents=True, exist_ok=True)
        (self.out / LIVE / "summary.json").write_text(json.dumps(line, indent=1))
        self.log("live " + json.dumps(line))
        return ok


# ------------------------------------------------------------------ what the run left
def tries(out: Path):
    """(set, rollout key, try index, record, flags) of every episode of the live run that has a sidecar."""
    try:
        results = json.loads((out / LIVE / "results.json").read_text())
    except (OSError, ValueError):
        return
    for set_name, block in sorted(results["sets"].items()):
        for key, entry in sorted(block.get("rollouts", {}).items()):
            for i, t in enumerate(entry["tries"]):
                rec = t["record"]
                if not rec.get("error") and rec.get("result_file") and (f := flags.read(rec["result_file"])) is not None:
                    yield set_name, key, i, rec, f


def summary(out: Path) -> dict:
    """The one-line health check of a live run (wall_s is the run's)."""
    n = taken = recovered = ok = 0
    reasons: Counter = Counter()
    expert_s = 0.0
    for *_, rec, f in tries(out):
        n += 1
        if f["takeover_t"] is None:
            ok += bool(rec["passed"])
            continue
        taken += 1
        recovered += bool(rec["passed"])
        reasons[f["trigger"]] += 1
        expert_s += sum(t1 - t0 for t0, t1, who in f["segments"] if who == "expert")
    return {"policy_only_success": ok, "takeovers": taken, "reasons": dict(reasons), "recovered": recovered,
            "expert_frames_s": round(expert_s, 2), "episodes": n}


def entries(out: Path, spec: RecoverySpec) -> list[dict]:
    """The manifest entries of the live run: one per episode with a takeover (`keep: successes` leaves out the failed ones)."""
    rel = lambda p: os.path.relpath(p, out)
    found = []
    for set_name, key, i, rec, f in tries(out):
        if f["takeover_t"] is None or (spec.keep == "successes" and not rec["passed"]):
            continue
        bag = rec.get("bag")
        found.append({
            "seed": rec["seed"], "scene": str(rec["scene"]), "robot": spec.robot, "task": spec.task, "episode": rec["episode"],
            "result": rel(rec["result_file"]), "bag": rel(bag) if bag and Path(bag).is_dir() else None,
            "passed": bool(rec["passed"]), "datagen": {"expert": f.get("expert"), "attempt": i, "wall_s": None},
            "recovery": {"kind": "dagger",
                         "from_rollout": {"run": spec.eval_run, "set": set_name, "rollout": key, "try": i,
                                          "scene": str(rec["scene"]), "bag": bag},
                         "phase": f["phase"], "trigger": f["trigger"], "reason": f["reason"], "t_restart": f["takeover_t"],
                         "restart_from": "takeover", "restart_state": None, "reset": rec.get("reset", {"mode": "relaunch"}),
                         "segment": f["segment"], "policy": {"plugin": rec["policy"], "ckpt": rec.get("ckpt")},
                         "expert_failure": f["expert_failure"], "max_delta_action": f["max_delta_action"]}})
    return found
