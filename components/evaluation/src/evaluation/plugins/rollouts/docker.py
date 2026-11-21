"""One real closed-loop rollout: rollout.py's `run` (policy server + Rosetta runner + simulator on a per-rollout docker
network), loaded by path on first use (it imports docker and the launch modules, which core and the CLI must not).

Params (yaml `rollout: {plugin: docker, ...}`): budget_s, wait_s, after_success_s, rtf, video, as rollout.run's."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from ...core.errors import EvalError
from ...core.interfaces import Rollout
from ...core.registry import REGISTRY

COMPONENT = Path(__file__).resolve().parents[4]          # components/evaluation: rollout.py, evaluate.py
PARAMS = ("budget_s", "wait_s", "after_success_s", "rtf", "video")


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

    def run(self, item, out, *, seed, restore=None, runner, policy, ckpt, gui=False):
        runner_params = dict(actions_per_chunk=runner.actions_per_chunk, chunk_size_threshold=runner.chunk_size_threshold)
        return load().run(item.start_dir, out, seed=seed, policy=policy.name, ckpt=ckpt.path if ckpt else None,
                          restore=restore, runner_params=runner_params, gui=gui, task=self.task, **self.params)

    def default_workers(self, policy):
        return load("evaluate").default_parallel(policy.name)
