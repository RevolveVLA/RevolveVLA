"""Client-container entry of the `sim` runner: `python3 -m datagen.scene_client --spec <spec json> --seed N --instance
<scene.yaml>` runs the episode loop (core/episodes.py) of one scene instance against the sim on this container's ROS domain,
with the spec's generator plugin. Exit code: 0 done, a SimFault's code (75-78: the host relaunches the sim), 1 anything else.
Runs in ROS 2 Lyrical (Python 3.14) from the simulator image with /repo on the path: only datagen's core and the generator
plugin are imported."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datagen.core import episodes, registry
from datagen.core.spec import parse
from revolve_common.sim_client import run_client


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="datagen.scene_client")
    ap.add_argument("--spec", required=True, help="the datagen spec as JSON (DatagenSpec.raw)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--instance", required=True, help="the scene instance's scene.yaml (its folder is the sim's --out)")
    a = ap.parse_args(argv)
    spec, instance = parse(json.loads(a.spec), "<spec>"), Path(a.instance)
    generator = registry.create("generator", spec.generator.plugin, spec, spec.generator.params)
    run_client(lambda client: episodes.run_episodes(client, generator, spec.task, spec.robot, instance, a.seed,
                                                    spec.episodes), instance.parent)


if __name__ == "__main__":
    main()
