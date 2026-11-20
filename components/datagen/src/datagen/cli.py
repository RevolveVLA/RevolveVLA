"""`./revolve datagen <spec.yaml> [sample|generate|record|convert|check|all] [--split S] [--dry-run]`.

Resolves only the plugins the requested steps need, so `sample` runs on the host without docker. Extra plugin
modules are found through $DATAGEN_PLUGIN_PATH (os.pathsep separated). Exit code: 0 ok, 1 failed check or error."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .core import registry
from .core.errors import DatagenError
from .core.pipeline import STEPS, Pipeline, PluginSet
from .core.spec import DatagenSpec, PluginRef, load


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
        p.checks = many("check", spec.checks or [PluginRef("target")])
    return p


def main(argv=None, root: Path | None = None) -> int:
    ap = argparse.ArgumentParser(prog="revolve datagen", description=__doc__.split("\n\n")[0])
    ap.add_argument("spec")
    ap.add_argument("step", nargs="?", default="all", choices=(*STEPS, "all"))
    ap.add_argument("--split", action="append", help="only this split (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="say what would run; no plugin call, nothing written")
    a = ap.parse_args(argv)
    sys.path[:0] = [p for p in os.environ.get("DATAGEN_PLUGIN_PATH", "").split(os.pathsep) if p]
    steps = STEPS if a.step == "all" else (a.step,)
    try:
        spec = load(a.spec)
        out = Path(spec.out) if Path(spec.out).is_absolute() else (root or Path.cwd()) / spec.out
        return 0 if Pipeline(spec, build_plugins(spec, steps), out, a.dry_run).run(steps, a.split) else 1
    except DatagenError as e:
        print(f"datagen: {e}", file=sys.stderr)
        return 1
