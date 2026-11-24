"""The standalone live run: the policy of an evaluation spec rolls out its sets under a live recorder (DAgger) and the
takeovers become manifest entries. The rollouts are the evaluation component's (its Pipeline, its `sim` rollout, in `<out>/live/`):
only the policy is wrapped (`LivePolicy`: its client gets the recorder's driver, `recovery.live_client`), so retries, workers,
warm resets and resumes are the evaluation's own.

`<out>/live/` = an evaluation run (results.json, report.md, `<set>/<seed>_sim/` with every episode's result, bag and sidecar,
see core/flags.py) + `summary.json`: {policy_only_success, takeovers, reasons, recovered, expert_frames_s, episodes, interventions (expert spans),
handbacks, policy_success_after_handback (episodes the policy finished after a hand-back), wall_s}
(a line of it is logged). The manifest entries (core/manifest.py shape, `recovery.kind` "dagger") are derived from that run,
never read back: one per episode with a takeover (a policy-only success is the evaluation's rollout). A coupled evaluation
(core/coupling.py) is the same run, in the evaluation's own out: `wrap` puts the recorder around its policy, the functions
below read it from its dir."""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Collection, Sequence

from datagen.core.spec import SpecError

from . import flags, leakage
from .manifest import repo_rel
from .spec import RecoverySpec

LIVE = "live"
DRIVER = "recovery.live_client:make"
RECOVERY_SRC = "/repo/components/recovery/src"          # the client container's view of this package (repo = /repo)
STAGES = ("sample", "rollout", "report")                # of the evaluation pipeline


class LivePolicy:
    """The evaluation policy `inner` under the live recorder: same name and needs, its client args add the driver (the spec,
    the phases the recorder wants no takeover in, the policy's own perturbation for the test policies). Only a policy whose
    client streams actions can be halted: not `expert` (a closed-loop generator). `seeds`: the seed ranges (pools) of the eval sets
it records in (default all): the configs of the others run the plain policy."""

    def __init__(self, inner, spec: RecoverySpec, recorder, root: Path, seeds: Collection[tuple[int, int]] | None = None):
        if inner.name == "expert":
            raise SpecError("policy expert drives closed-loop and cannot be halted: a live recorder needs a policy whose "
                            "client streams actions (oracle, frozen, noisy_expert, a served policy)")
        self.inner, self.spec, self.recorder, self.root, self.seeds = inner, spec, recorder, root, seeds
        self.name, self.needs_ckpt, self.client_only = inner.name, inner.needs_ckpt, inner.client_only

    def client_args(self, item) -> list[str]:
        if self.seeds is not None and not any(lo <= item.config.seed <= hi for lo, hi in self.seeds):
            return self.inner.client_args(item)
        payload = {"spec": self.spec.raw, "closed": sorted(self.recorder.closed(flags.tally(self.root))),
                   "perturb": getattr(self.inner, "perturbation", lambda item: None)(item)}
        return [*self.inner.client_args(item), "--instance", str(item.scene), "--driver", DRIVER,
                "--driver-args", json.dumps(payload)]


def eval_plugins(spec: RecoverySpec, eval_spec, recorder, root: Path):
    """The evaluation's plugins for `eval_spec`, its policy under the recorder, its client container able to import this package."""
    register_policies()
    from evaluation.cli import build_plugins
    from evaluation.core.registry import REGISTRY
    REGISTRY.extend_path()                                          # $EVAL_PLUGIN_PATH
    return wrap(build_plugins(eval_spec, ("sample", "rollout")), spec, recorder, root)


def register_policies() -> None:
    """The test policies (frozen, noisy_expert) into the evaluation registry: before a spec that names them is built."""
    importlib.import_module("recovery.plugins.policies")


def wrap(plugins, spec: RecoverySpec, recorder, root: Path, seeds: Collection[tuple[int, int]] | None = None):
    """`plugins` (an evaluation PluginSet) with its policy under the recorder (for the configs of the pools `seeds`), its client
    container able to import this package."""
    plugins.policy = LivePolicy(plugins.policy, spec, recorder, root, seeds)
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
        line = {**summary(self.out / LIVE), "wall_s": round(time.monotonic() - t0, 1)}
        (self.out / LIVE).mkdir(parents=True, exist_ok=True)
        (self.out / LIVE / "summary.json").write_text(json.dumps(line, indent=1))
        self.log("live " + json.dumps(line))
        return ok


# ------------------------------------------------------------------ what the run left
def tries(run: Path):
    """(set, rollout key, try index, record, flags) of every episode of the live run (an evaluation run's dir) that has a sidecar."""
    try:
        results = json.loads((run / "results.json").read_text())
    except (OSError, ValueError):
        return
    for set_name, block in sorted(results["sets"].items()):
        for key, entry in sorted(block.get("rollouts", {}).items()):
            for i, t in enumerate(entry["tries"]):
                rec = t["record"]
                if not rec.get("error") and rec.get("result_file") and (f := flags.read(rec["result_file"])) is not None:
                    yield set_name, key, i, rec, f


def summary(run: Path) -> dict:
    """The one-line health check of a live run (`run`: its dir)."""
    n = taken = recovered = ok = interventions = handbacks = after = 0
    reasons: Counter = Counter()
    expert_s = 0.0
    for *_, rec, f in tries(run):
        n += 1
        if f["takeover_t"] is None:
            ok += bool(rec["passed"])
            continue
        taken += 1
        recovered += bool(rec["passed"])
        reasons[f["trigger"]] += 1
        spans = [s for s in f["segments"] if s[2] == "expert"]
        expert_s += sum(t1 - t0 for t0, t1, _ in spans)
        interventions += len(spans)
        back = sum(i["handback_t"] is not None for i in f.get("interventions", ()))
        handbacks += back
        after += bool(back and rec["passed"] and f["segments"][-1][2] == "policy")      # the policy finished it
    return {"policy_only_success": ok, "takeovers": taken, "reasons": dict(reasons), "recovered": recovered,
            "expert_frames_s": round(expert_s, 2), "episodes": n, "interventions": interventions, "handbacks": handbacks,
            "policy_success_after_handback": after}


def entries(run: Path, out: Path, spec: RecoverySpec) -> list[dict]:
    """The manifest entries (paths relative to `out`) of the live run `run`: one per episode with a takeover (`keep: successes`
    leaves out the failed ones)."""
    rel = lambda p: os.path.relpath(p, out)
    found = []
    for set_name, key, i, rec, f in tries(run):
        if f["takeover_t"] is None or (spec.keep == "successes" and not rec["passed"]):
            continue
        bag = rec.get("bag")
        found.append({
            "seed": rec["seed"], "scene": rel(rec["scene"]), "robot": spec.robot, "task": spec.task, "episode": rec["episode"],
            "result": rel(rec["result_file"]), "bag": rel(bag) if bag and Path(bag).is_dir() else None,
            "passed": bool(rec["passed"]), "datagen": {"expert": f.get("expert"), "attempt": i, "wall_s": None},
            "recovery": {"kind": "dagger",
                         "from_rollout": {"run": spec.eval_run, "set": set_name, "rollout": key, "try": i,
                                          "scene": repo_rel(rec["scene"]), "bag": repo_rel(bag)},
                         "phase": f["phase"], "trigger": f["trigger"], "reason": f["reason"], "t_restart": f["takeover_t"],
                         "restart_from": "takeover", "restart_state": None, "reset": rec.get("reset", {"mode": "relaunch"}),
                         "segment": f["segment"], "policy": {"plugin": rec["policy"], "ckpt": rec.get("ckpt")},
                         "expert_failure": f["expert_failure"], "max_delta_action": f["max_delta_action"],
                         **({"segments": f["kept"]} if f.get("kept") else {}),
                         **({"interventions": f["interventions"]} if "interventions" in f else {})}})
    return found
