"""`./revolve recover <spec.yaml> [collect|run|check|live|all] [--dry-run] [--workers N] [--gui|--headless]`.

Phase-wise recovery of a finished evaluation run: `collect` finds its failed phases (the bags are read in the simulator
image), `run` records recovery episodes from the restart states with the spec's datagen generator, `check` judges the run.
`live` (a spec with `live:` and `from.eval` an evaluation spec) rolls that spec's policy out with the live recorder (DAgger).
`all` = the stages the spec has (phases: collect, run, check; live: live). Resolves only the plugins the requested stages
need; `--dry-run` resolves none (it prints the plan from the spec). Extra plugin modules: $RECOVERY_PLUGIN_PATH (and datagen's generators: $DATAGEN_PLUGIN_PATH), os.pathsep
separated. `--workers` / `--gui` override the spec's `run:` block. Exit code: 0 ok, 1 failed check or error, 2 invalid spec, 128 + n when stopped by
signal n (progress saved: rerun the same command to resume; the run's containers are removed). Never imports ROS / docker
SDKs: the plugins do, lazily."""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

from datagen.core.errors import DatagenError
from datagen.core.interrupt import Interrupted, interruptible
from datagen.core.spec import SpecError
from revolve_common import specs

from .core import live as lv
from .core.pipeline import LIVE, STAGES, Pipeline, PluginSet, phase_plugins
from .core.registry import REGISTRY
from .core.robot import arm_of
from .core.scan import checked
from .core.spec import RecoverySpec, load


def build_plugins(spec: RecoverySpec, stages, root: Path | None = None) -> PluginSet:
    p = phase_plugins(spec, stages)
    if LIVE in stages:
        p.live, out = REGISTRY.create("live", spec.live.plugin, spec, spec.live.params), Path(spec.out)
        from evaluation.cli import stop_jobs
        from evaluation.core.pipeline import Pipeline as EvalPipeline
        eval_spec = lv.live_eval_spec(spec, lv.load_eval(spec, root or Path.cwd()), out)
        pipeline = EvalPipeline(eval_spec, lv.eval_plugins(spec, eval_spec, p.live, out / lv.LIVE), out / lv.LIVE,
                                abort=lambda: stop_jobs(eval_spec))
        p.live_run = lv.LiveRun(spec, pipeline.run, out)
    return p


def stages_of(spec: RecoverySpec, stage: str) -> tuple[str, ...]:
    """The stage asked for; `all`: the stages this spec has (phase-wise recovery: collect, run, check; a live recorder: live)."""
    if stage != "all":
        return (stage,)
    return (*(STAGES if spec.phases else ()), *((LIVE,) if spec.live else ()))


def with_run(spec: RecoverySpec, a) -> RecoverySpec:
    """The spec with the command line's `--workers` / `--gui` / `--headless` over its `run:` block (not in the hash)."""
    run = dataclasses.replace(spec.run, **{**({"workers": a.workers} if a.workers else {}),
                                           **({"headless": False} if a.gui else {"headless": True} if a.headless else {})})
    return dataclasses.replace(spec, run=run)


def main(argv=None, root: Path | None = None) -> int:
    ap = argparse.ArgumentParser(prog="revolve recover", description=__doc__.split("\n\n")[0])
    ap.add_argument("spec")
    ap.add_argument("stage", nargs="?", default="all", choices=(*STAGES, LIVE, "all"))
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
    root = root or Path.cwd()
    try:
        spec = with_run(load(a.spec), a)
        stages = stages_of(spec, a.stage)
        if LIVE in stages and not spec.live:
            raise SpecError("stage live: this spec has no `live:` recorder")
        absolute = lambda p: str(Path(p) if Path(p).is_absolute() else root / p)
        spec = dataclasses.replace(spec, out=absolute(spec.out), eval_run=absolute(spec.eval_run))
        checked(spec)
        plugins = PluginSet() if a.dry_run else build_plugins(spec, stages, root)
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
