"""`./revolve evaluate <spec.yaml> [sample|check|rollout|report|all] [--set S] [--dry-run] [--workers N] [--gui|--headless]`.

Resolves only the plugins the requested stages need; `--dry-run` resolves none (it prints the plan from the spec).
Extra plugin modules are found through $EVAL_PLUGIN_PATH (datagen's, which the sets use, through $DATAGEN_PLUGIN_PATH;
os.pathsep separated). `--workers` / `--gui` override the spec's `run:` block. Exit code: 0 ok, 1 failed check or error, 2 invalid spec, 128 + n when stopped by signal n
(results saved: rerun the same command to resume; the run's docker jobs are stopped). Never imports rollout.py / docker: the plugins do, lazily."""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

from datagen.core.errors import DatagenError
from datagen.core.interrupt import Interrupted, interruptible
from datagen.core.spec import SpecError

from .core.pipeline import STAGES, Pipeline, PluginSet
from .core.registry import REGISTRY
from .core.spec import EvalSpec, load


def build_plugins(spec: EvalSpec, stages) -> PluginSet:
    p = PluginSet()
    if {"sample", "check", "rollout"} & set(stages):
        p.sets = {n: REGISTRY.create("set", s.plugin, spec) for n, s in spec.sets.items()}
    if "rollout" in stages:
        p.policy = REGISTRY.create("policy", spec.policy.plugin, spec, spec.policy.params)
        p.rollout = REGISTRY.create("rollout", spec.rollout.plugin, spec, spec.rollout.params)
        p.judge = REGISTRY.create("judge", spec.judge.plugin, spec, spec.judge.params)
        if spec.ckpt:
            p.ckpt = REGISTRY.create("ckpt", spec.ckpt.source, spec, spec.ckpt.params)
            if p.ckpt.relabel:                       # whatever the source, serve it under the robot contract's cameras
                p.ckpt = REGISTRY.create("ckpt", "relabel", spec, {"of": p.ckpt})
    return p


def with_run(spec: EvalSpec, a) -> EvalSpec:
    """The spec with the command line's `--workers` / `--gui` / `--headless` over its `run:` block (not in the hash)."""
    run = dataclasses.replace(spec.run, **{**({"workers": a.workers} if a.workers else {}),
                                           **({"headless": False} if a.gui else {"headless": True} if a.headless else {})})
    return dataclasses.replace(spec, run=run)


def main(argv=None, root: Path | None = None) -> int:
    ap = argparse.ArgumentParser(prog="revolve evaluate", description=__doc__.split("\n\n")[0])
    ap.add_argument("spec")
    ap.add_argument("stage", nargs="?", default="all", choices=(*STAGES, "all"))
    ap.add_argument("--set", action="append", help="only this eval set (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; no plugin call, nothing written")
    ap.add_argument("--workers", type=int, help="rollouts run at a time (overrides run.workers)")
    gui = ap.add_mutually_exclusive_group()
    gui.add_argument("--gui", action="store_true", help="show the gz GUI on $DISPLAY (run.headless false; one worker)")
    gui.add_argument("--headless", action="store_true", help="no GUI (run.headless true)")
    a = ap.parse_args(argv)
    REGISTRY.extend_path()
    sys.path[:0] = [p for p in os.environ.get("DATAGEN_PLUGIN_PATH", "").split(os.pathsep) if p]
    if a.workers is not None and a.workers < 1:
        ap.error("--workers must be >= 1")
    stages = STAGES if a.stage == "all" else (a.stage,)
    root = root or Path.cwd()
    try:
        spec = with_run(load(a.spec, root), a)
        out = Path(spec.out) if Path(spec.out).is_absolute() else root / spec.out
        pipeline = Pipeline(spec, PluginSet() if a.dry_run else build_plugins(spec, stages), out, a.dry_run,
                            abort=lambda: stop_jobs(out))
        with interruptible():
            return 0 if pipeline.run(stages, a.set) else 1
    except Interrupted as e:
        print(f"evaluate: {e}", file=sys.stderr)
        return e.code
    except SpecError as e:
        print(f"evaluate: {e}", file=sys.stderr)
        return 2
    except DatagenError as e:
        print(f"evaluate: {e}", file=sys.stderr)
        return 1


def stop_jobs(out: Path):
    """Kill the docker jobs (generator, policy server, simulator) of the run at `out`: what an interrupt leaves behind."""
    from datagen.containers import stop_jobs as stop
    stop(out, log=lambda m: print(f"evaluate: {m}", file=sys.stderr))
