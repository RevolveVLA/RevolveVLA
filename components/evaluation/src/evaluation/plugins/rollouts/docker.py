"""One real closed-loop rollout: rollout.py's `run` (policy server + Rosetta runner + simulator on a per-rollout docker
network), loaded by path on first use (it imports docker and the launch modules, which core and the CLI must not).

Params (yaml `rollout: {plugin: docker, ...}`): budget_s, wait_s, after_success_s, wall_safety, setup_s, rtf, video, as
rollout.run's (the task's `rollout:` values unless given). Without `rtf` the measured one of this machine is used
(evaluate.measured_rtf: runs/eval_rtf.json for the parallel count, else the 0.1 prior)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from datagen.plugins.generate_module import use_robot

from ...core.errors import EvalError
from ...core.interfaces import Rollout
from ...core.registry import REGISTRY

COMPONENT = Path(__file__).resolve().parents[4]          # components/evaluation: rollout.py, evaluate.py
PARAMS = ("budget_s", "wait_s", "after_success_s", "wall_safety", "setup_s", "rtf", "video")


def load(name: str = "rollout"):
    """`rollout` (or `evaluate`, for its GPU-memory default) as a module, imported when first needed."""
    if str(COMPONENT) not in sys.path:
        sys.path.append(str(COMPONENT))
    return importlib.import_module(name)


@REGISTRY.register("rollout", "docker")
class DockerRollout(Rollout):
    def __init__(self, spec, params):
        if unknown := sorted(set(params) - set(PARAMS)):
            raise EvalError(f"rollout docker: unknown parameters {unknown} (known: {list(PARAMS)})")
        self.task, self.params = spec.train.task, dict(params)
        use_robot(spec.train.robot)         # rollout.py loads the task: $REVOLVE_ROBOT (no generating set sets it)
        self.spec, self.workers = spec, None

    def run(self, item, out, *, seed, restore=None, runner, policy, ckpt, gui=False):
        return load().run(item.scene, out, seed=seed, policy=policy.name, ckpt=ckpt.path if ckpt else None,
                          restore=restore, runner_params=runner.params(), gui=gui, task=self.task,
                          **{**self.params, "rtf": self.rtf(policy)})

    def default_workers(self, policy):
        return load("evaluate").default_parallel(policy.name)

    def rtf(self, policy):
        """`rtf` of the yaml, else the measured one for the number of rollouts at a time."""
        if self.params.get("rtf"):
            return self.params["rtf"]
        if self.workers is None:             # the run's rollouts at a time: the GUI's one, run.workers, else by GPU memory
            run = self.spec.run
            self.workers = (run.workers or self.default_workers(policy)) if run.headless else 1
        return load("evaluate").measured_rtf(self.workers)
