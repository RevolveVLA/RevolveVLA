"""`./revolve recipe <recipe.yaml> [resolve|check|materialise] [--dry-run]`.

Picks the data of the next training round from finished runs (a datagen run, an evaluation run, a recovery run): how many episodes
(or frames) of each source and phase, divided into splits by scene, minus anything under `exclude`. `resolve` prints what the sources
hold, `check` the plan (asked vs got, the guard, the splits), `materialise` (default) also writes `<out>/report.md` and the
output's files (`selection.json`). `--dry-run`: the plan from the yaml only, no run read. Extra plugin modules: $RECIPE_PLUGIN_PATH.
Needs no docker, ROS or GPU. Exit code: 0 ok, 1 short with `on_short: error`, a broken invariant or an unreadable run, 2 invalid spec."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datagen.core.errors import DatagenError
from datagen.core.spec import SpecError

from .core.leakage import Exclusion
from .core.pipeline import STAGES, Pipeline
from .core.registry import REGISTRY
from .core.spec import RecipeSpec, load


def build(spec: RecipeSpec, root: Path, stage: str) -> Pipeline:
    sources = {n: REGISTRY.create("source", ref.plugin, n, ref.params, root) for n, ref in spec.sources.items()}
    output = (lambda: REGISTRY.create("output", spec.output.plugin, spec.output.params)) if stage == "materialise" else None
    out = Path(spec.out) if Path(spec.out).is_absolute() else root / spec.out
    return Pipeline(spec, sources, Exclusion.load(spec.exclude, root), out, output)


def describe(spec: RecipeSpec) -> None:
    print(f"recipe {spec.hash}: unit {spec.unit}, seed {spec.seed}, splits {dict(spec.splits)}, out {spec.out}")
    for name, ref in spec.sources.items():
        take = {p or "any": n for p, n in spec.take[name].items()}
        print(f"  {name}: {ref.plugin} {dict(ref.params)}; take {take}")
    print(f"  exclude {list(spec.exclude)}; on_short {spec.on_short}; output {spec.output.plugin}")


def main(argv=None, root: Path | None = None) -> int:
    ap = argparse.ArgumentParser(prog="revolve recipe", description=__doc__.split("\n\n")[0])
    ap.add_argument("spec")
    ap.add_argument("stage", nargs="?", default="materialise", choices=STAGES)
    ap.add_argument("--dry-run", action="store_true", help="print the recipe from the yaml; read no run, write nothing")
    a = ap.parse_args(argv)
    REGISTRY.extend_path()
    root = root or Path.cwd()
    try:
        spec = load(a.spec)
        if a.dry_run:
            describe(spec)
            return 0
        return 0 if build(spec, root, a.stage).run(a.stage) else 1
    except SpecError as e:
        print(f"recipe: {e}", file=sys.stderr)
        return 2
    except DatagenError as e:
        print(f"recipe: {e}", file=sys.stderr)
        return 1
