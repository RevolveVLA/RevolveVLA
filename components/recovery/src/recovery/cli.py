"""`./revolve recover <spec.yaml> [collect|run|check|all] [--dry-run] [--workers N] [--gui|--headless]`.

Phase-wise recovery of a finished evaluation run: `collect` finds its failed phases (the bags are read in the simulator
image), `run` records recovery episodes from the restart states with the spec's datagen generator, `check` judges the run.
Resolves only the plugins the requested stages need; `--dry-run` resolves none (it prints the plan from the spec). Extra
plugin modules: $RECOVERY_PLUGIN_PATH (and datagen's generators: $DATAGEN_PLUGIN_PATH), os.pathsep separated. `--workers` /
`--gui` override the spec's `run:` block. Exit code: 0 ok, 1 failed check or error, 2 invalid spec, 128 + n when stopped by
signal n (progress saved: rerun the same command to resume; the run's containers are removed). Never imports ROS / docker
SDKs: the plugins do, lazily."""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

from datagen.core import registry as datagen_registry
from datagen.core.errors import DatagenError
from datagen.core.interrupt import Interrupted, interruptible
from datagen.core.spec import SpecError
from revolve_common import specs

from .core.pipeline import STAGES, Pipeline, PluginSet
from .core.registry import REGISTRY
from .core.robot import arm_of
from .core.scan import load_task, supervisor_of
from .core.spec import RecoverySpec, load


def build_plugins(spec: RecoverySpec, stages) -> PluginSet:
    p = PluginSet()
    if "collect" in stages or "run" in stages:
        p.scanner = REGISTRY.create("scanner", spec.scanner.plugin, spec, spec.scanner.params)
    if "run" in stages:
        g = spec.phases.generator
        p.generator = datagen_registry.create("generator", g.plugin, spec.datagen_spec(), g.params)
        p.runner = REGISTRY.create("runner", spec.runner.plugin, spec, spec.runner.params)
    return p


def with_run(spec: RecoverySpec, a) -> RecoverySpec:
    """The spec with the command line's `--workers` / `--gui` / `--headless` over its `run:` block (not in the hash)."""
    run = dataclasses.replace(spec.run, **{**({"workers": a.workers} if a.workers else {}),
                                           **({"headless": False} if a.gui else {"headless": True} if a.headless else {})})
    return dataclasses.replace(spec, run=run)


def checked(spec: RecoverySpec) -> RecoverySpec:
    """The spec once the task, the robot and the registry have checked it (quota phases, triggers, the live recorder)."""
    supervisor_of(spec, load_task(spec), arm_of(spec.robot, specs.REPO_SPECS))
    if spec.live:
        REGISTRY.create("live", spec.live.plugin, spec, spec.live.params).validate(spec)
    return spec


def main(argv=None, root: Path | None = None) -> int:
    ap = argparse.ArgumentParser(prog="revolve recover", description=__doc__.split("\n\n")[0])
    ap.add_argument("spec")
    ap.add_argument("stage", nargs="?", default="all", choices=(*STAGES, "all"))
    ap.add_argument("--dry-run", action="store_true", help="print the plan; no plugin call, nothing written")
    ap.add_argument("--workers", type=int, help="recovery jobs run at a time, one sim each (overrides run.workers)")
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
        spec = with_run(load(a.spec), a)
        absolute = lambda p: str(Path(p) if Path(p).is_absolute() else root / p)
        spec = dataclasses.replace(spec, out=absolute(spec.out), eval_run=absolute(spec.eval_run))
        checked(spec)
        plugins = PluginSet() if a.dry_run else build_plugins(spec, stages)
        pipeline = Pipeline(spec, arm_of(spec.robot, specs.REPO_SPECS), plugins, Path(spec.out), a.dry_run)
        with interruptible():
            return 0 if pipeline.run(stages) else 1
    except Interrupted as e:
        print(f"recover: {e}", file=sys.stderr)
        return e.code
    except (SpecError, specs.SpecError) as e:
        print(f"recover: {e}", file=sys.stderr)
        return 2
    except DatagenError as e:
        print(f"recover: {e}", file=sys.stderr)
        return 1
