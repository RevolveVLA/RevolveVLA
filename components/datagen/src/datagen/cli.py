"""`./revolve datagen <spec.yaml | job.yaml> [sample|generate|convert|check|all] [--split S] [--dry-run] [--workers N] [--gui]`.

Resolves only the plugins the requested steps need, so `sample` runs on the host without docker. Extra plugin
modules are found through $DATAGEN_PLUGIN_PATH (os.pathsep separated). `--workers` / `--gui` override the spec's `run:`
block (workers = (sim, generator) pairs). A job yaml (DATAGEN-v2.md 8.2: instances made elsewhere) runs as a spec. Exit
code: 0 ok, 1 failed check or error, 128 + n when stopped by signal n (SIGINT 130, SIGTERM 143): the state is saved and
the run's sims are stopped."""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

from . import job
from .core import checks, registry
from .core.errors import DatagenError
from .core.interrupt import Interrupted, interruptible
from .core.pipeline import STEPS, Pipeline, PluginSet
from .core.spec import DatagenSpec, PluginRef


REMOVED_RANDOMISERS = {      # the old simulator's start streams -> the one that replaced them
    name: "robot_start (x_m, y_m, yaw_rad, arm: the scene instance's robot_start, DATAGEN-v2.md 8.1)"
    for name in ("robot_base", "reach_start", "drive_start", "grasp")}


def build_plugins(spec: DatagenSpec, steps) -> PluginSet:
    def one(kind, ref):
        return registry.create(kind, ref.plugin, spec, ref.params) if ref else None

    def many(kind, refs):
        return [one(kind, r) for r in refs]
    if old := [n for n in spec.randomisers if n in REMOVED_RANDOMISERS]:
        raise DatagenError(f"randomisers.{old[0]}: removed with the old simulator; use {REMOVED_RANDOMISERS[old[0]]}")
    p = PluginSet(randomisers={n: registry.create("randomiser", n) for n in spec.randomisers})
    if {"sample", "generate"} & set(steps) and any(r.scene_field for r in p.randomisers.values()):
        p.instancer = registry.create("instancer", "scene_instance", spec, {})
    if {"sample", "generate"} & set(steps):
        p.samplers = {s.sampler.strategy: registry.create("sampler", s.sampler.strategy) for s in spec.splits}
    if "generate" in steps:
        p.generator, p.validators = one("generator", spec.generator), many("validator", spec.validators)
        p.runner = one("runner", spec.runner or PluginRef("sim"))
    if "convert" in steps:
        p.converter = one("converter", spec.converter)
    if "check" in steps:
        p.checks = many("check", spec.checks or checks.default_checks(spec))
    return p


def list_randomisers(a, ap) -> int:
    from .plugins.randomisers import listing
    if not (a.task and a.robot):
        ap.error("--list-randomisers needs --task and --robot")
    try:
        print(listing.render(a.task, a.robot))
    except DatagenError as e:
        print(f"datagen: {e}", file=sys.stderr)
        return 1
    return 0


def main(argv=None, root: Path | None = None) -> int:
    ap = argparse.ArgumentParser(prog="revolve datagen", description=__doc__.split("\n\n")[0])
    ap.add_argument("spec", nargs="?")
    ap.add_argument("step", nargs="?", default="all", choices=(*STEPS, "all"))
    ap.add_argument("--split", action="append", help="only this split (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="say what would run; no plugin call, nothing written")
    ap.add_argument("--workers", type=int, help="scenes run at a time, one (sim, generator) pair each (overrides run.workers)")
    gui = ap.add_mutually_exclusive_group()
    gui.add_argument("--gui", action="store_true", help="show the gz GUI of the replays on $DISPLAY (run.headless false; "
                                                        "one window, so one worker)")
    gui.add_argument("--headless", action="store_true", help="no GUI (run.headless true)")
    ap.add_argument("--list-randomisers", action="store_true",
                    help="print the randomisers usable for --task and --robot (no spec), their dims and ranges")
    ap.add_argument("--task"), ap.add_argument("--robot")
    a = ap.parse_args(argv)
    sys.path[:0] = [p for p in os.environ.get("DATAGEN_PLUGIN_PATH", "").split(os.pathsep) if p]
    if a.list_randomisers:
        return list_randomisers(a, ap)
    if not a.spec:
        ap.error("a spec is required (or --list-randomisers --task T --robot R)")
    if a.workers is not None and a.workers < 1:
        ap.error("--workers must be >= 1")
    steps = STEPS if a.step == "all" else (a.step,)
    try:
        spec, scenes = job.load(a.spec, root or Path.cwd())
        spec = with_run(spec, a)
        out = (Path(spec.out) if Path(spec.out).is_absolute() else (root or Path.cwd()) / spec.out)
        spec = dataclasses.replace(spec, out=str(out))                  # the plugins see the absolute out
        plugins = build_plugins(spec, steps)
        pipeline = Pipeline(spec, plugins, out, a.dry_run, abort=plugins.runner and plugins.runner.stop, scenes=scenes)
        with interruptible():
            return 0 if pipeline.run(steps, a.split) else 1
    except Interrupted as e:
        print(f"datagen: {e}", file=sys.stderr)
        return e.code
    except DatagenError as e:
        print(f"datagen: {e}", file=sys.stderr)
        return 1


def with_run(spec: DatagenSpec, a) -> DatagenSpec:
    """The spec with the command line's `--workers` / `--gui` / `--headless` over its `run:` block (not in the hash)."""
    run = dataclasses.replace(spec.run, **{**({"workers": a.workers} if a.workers else {}),
                                           **({"headless": False} if a.gui else {"headless": True} if a.headless else {})})
    return dataclasses.replace(spec, run=run)

