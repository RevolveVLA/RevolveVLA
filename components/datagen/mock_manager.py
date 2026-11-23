"""Stand-in for the manager (DATAGEN-v2.md 8.2): hands datagen a job and merges the result.

    python components/datagen/mock_manager.py [--workers N] [--dry-run] <spec.yaml | job.yaml>

A spec yaml: the randomiser samples the scene instances, then N (sim, generator) pairs run them. A job yaml (instances
made elsewhere, `episodes: [{seed, scene}]`): N pairs split its scenes. Either way `<out>/manifest.json` holds one entry
per episode attempted. All of it is `datagen.cli` (the same pipeline as `./revolve datagen <spec> all`)."""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "src"), str(HERE.parents[1] / "common")]

from datagen import cli  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mock_manager.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("spec", help="a datagen spec yaml or a job yaml")
    ap.add_argument("--workers", type=int, help="(sim, generator) pairs (default: the spec's run.workers)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return cli.main([a.spec, "all", *(["--workers", str(a.workers)] if a.workers else []),
                     *(["--dry-run"] if a.dry_run else [])])


if __name__ == "__main__":
    sys.exit(main())
