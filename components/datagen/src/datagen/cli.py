"""`./revolve datagen <spec.yaml> [sample|generate|record|convert|check|all] [--split S] [--dry-run] [--workers N] [--gui]`.

Resolves only the plugins the requested steps need, so `sample` runs on the host without docker. Extra plugin
modules are found through $DATAGEN_PLUGIN_PATH (os.pathsep separated). `--workers` / `--gui` override the spec's `run:`
block. Exit code: 0 ok, 1 failed check or error."""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

from .core import checks, registry
from .core.errors import DatagenError
from .core.pipeline import STEPS, Pipeline, PluginSet
from .core.spec import DatagenSpec, load


def build_plugins(spec: DatagenSpec, steps) -> PluginSet:
    def one(kind, ref):
        return registry.create(kind, ref.plugin, spec, ref.params) if ref else None

    def many(kind, refs):
        return [one(kind, r) for r in refs]
    p = PluginSet(randomisers={n: registry.create("randomiser", n) for n in spec.randomisers})
    if {"sample", "generate"} & set(steps):
        p.samplers = {s.sampler.strategy: registry.create("sampler", s.sampler.strategy) for s in spec.splits}
    if "generate" in steps:
        p.generator, p.validators = one("generator", spec.generator), many("validator", spec.validators)
    if "record" in steps:
        p.recorder = one("recorder", spec.recorder)
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
        print(listing.render(a.task, a.robot, a.variant))
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
    ap.add_argument("--workers", type=int, help="configs generated / recorded at a time (overrides run.workers)")
    gui = ap.add_mutually_exclusive_group()
    gui.add_argument("--gui", action="store_true", help="show the gz GUI of the replays on $DISPLAY (run.headless false; "
                                                        "one window, so one worker)")
    gui.add_argument("--headless", action="store_true", help="no GUI (run.headless true)")
    ap.add_argument("--list-randomisers", action="store_true",
                    help="print the randomisers usable for --task and --robot (no spec), their dims and ranges")
    ap.add_argument("--task"), ap.add_argument("--robot"), ap.add_argument("--variant", help="the generator's variant")
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
        spec = with_run(load(a.spec), a)
        out = Path(spec.out) if Path(spec.out).is_absolute() else (root or Path.cwd()) / spec.out
        return 0 if Pipeline(spec, build_plugins(spec, steps), out, a.dry_run).run(steps, a.split) else 1
    except DatagenError as e:
        print(f"datagen: {e}", file=sys.stderr)
        return 1


def with_run(spec: DatagenSpec, a) -> DatagenSpec:
    """The spec with the command line's `--workers` / `--gui` / `--headless` over its `run:` block (not in the hash)."""
    run = dataclasses.replace(spec.run, **{**({"workers": a.workers} if a.workers else {}),
                                           **({"headless": False} if a.gui else {"headless": True} if a.headless else {})})
    return dataclasses.replace(spec, run=run)
