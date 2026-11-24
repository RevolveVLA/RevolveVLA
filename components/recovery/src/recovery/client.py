"""Job entry, run in a client container of the simulator image: `python3 -m recovery.client --spec <spec json> --job <job json>
--sim-out <sim dir> [--warm]` runs the tries of one recovery job (core/jobs.py) against the sim on this container's ROS domain,
with the spec's datagen generator. `--warm`: each try starts with a warm reset to the job's restart (`SimClient.reset_to`).
Exit code: 0 done, a SimFault's code (75-78: the host relaunches the sim), MISS_EXIT (79: the warm reset missed, the host
relaunches the sim at the restart), 1 anything else."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datagen.core import registry
from revolve_common.sim_client import run_client

from .core.jobs import Job, ResetMissed, exit_after_miss, run_job
from .core.spec import parse


def warm_resetter(client):
    """`Target` -> the client's warm reset (revolve_common.warm_reset.ResetTarget has the same fields)."""
    from revolve_common.warm_reset import ResetTarget
    return lambda t: client.reset_to(ResetTarget(t.robot, t.base, t.arm, t.arm_joints, t.arm_controller))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="recovery.client")
    ap.add_argument("--spec", required=True, help="the recovery spec as JSON (RecoverySpec.raw)")
    ap.add_argument("--job", required=True, help="the job as JSON (Job.to_json)")
    ap.add_argument("--sim-out", required=True, help="the sim's --out (episode_<n>.json + bags)")
    ap.add_argument("--warm", action="store_true", help="reach the restart by a warm reset, not by a sim launched at it")
    a = ap.parse_args(argv)
    spec, job = parse(json.loads(a.spec), "<spec>"), Job.from_json(a.job)
    g = spec.phases.generator
    generator = registry.create("generator", g.plugin, spec.datagen_spec(), g.params)

    def run(client):
        try:
            run_job(client, generator, job, spec.task, spec.robot, spec.phases.episodes, Path(a.sim_out),
                    warm_resetter(client) if a.warm else None)
        except ResetMissed as e:
            exit_after_miss(e)
    run_client(run, a.sim_out)


if __name__ == "__main__":
    main()
