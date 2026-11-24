"""runner sim: the real thing. A slot is a sim launched by `./revolve sim` for a job's scene instance (SimSession,
`--no-autostart`, the slot's folder as `--out`, a leased name + ROS domain); the job's episodes run in a client container
beside it (`python3 -m recovery.client`, ROS 2 Lyrical inside the simulator image) on datagen's episode loop and the spec's
generator. A client that exits with MISS_EXIT missed its warm reset: the sim is relaunched at the restart.

Params: `infra_retries` (3), `wall_s_per_attempt` (600): the client's wall budget per try. Logs: <out>/logs/<slot>.{sim,
client}.log (kept across relaunches). Interrupt: `stop()` removes this run's containers and no sim is launched afterwards."""
from __future__ import annotations

import subprocess
from pathlib import Path

from revolve_common.sim_session import SimSession, stop_sims

from ...core.errors import JobError
from ...core.jobs import MISS_EXIT, ResetMissed
from ...core.registry import REGISTRY
from ...core.runner import RunnerStopped, SlotRunner

DATAGEN_SRC, RECOVERY_SRC = "/repo/components/datagen/src", "/repo/components/recovery/src"    # in the client container
SETUP_S = 300                                                                                  # bring-up, teardown


class _Session(SimSession):
    """A SimSession that does not (re)launch once the runner was stopped."""
    stopped = None

    def start(self):
        if self.stopped.is_set():
            raise RunnerStopped("run interrupted")
        super().start()


@REGISTRY.register("runner", "sim")
class SimRunner(SlotRunner):
    def __init__(self, spec, params):
        super().__init__(spec, params)
        self.attempt_s = float(params.get("wall_s_per_attempt", 600))

    def logs(self, slot) -> Path:
        return Path(self.spec.out) / "logs" / slot.name

    def session(self, slot, scene):
        s = _Session(scene, self.spec.robot, self.spec.task, slot.dir, slot.name, slot.domain,
                     self.spec.phases.episodes.record, gui=not self.spec.run.headless, log=f"{self.logs(slot)}.sim.log")
        s.stopped = self.stopped
        return s

    def work(self, slot, job, warm, generator):
        command = ["python3", "-m", "recovery.client", "--spec", _json(self.spec.raw), "--job", job.to_json(),
                   "--sim-out", str(slot.dir), *(["--warm"] if warm else [])]
        log = f"{self.logs(slot)}.client.log"
        client = slot.session.client(command, pythonpath=[DATAGEN_SRC, RECOVERY_SRC], mounts=[job.dir])
        try:
            rc = client.run(log, timeout=self.spec.phases.episodes.attempts * self.attempt_s + SETUP_S,
                            watch=lambda: None if self.stopped.is_set() else slot.session.check())
        except subprocess.TimeoutExpired:
            raise JobError(f"the job's client ran over its wall budget; see {log}") from None
        if self.stopped.is_set():
            raise RunnerStopped("run interrupted")
        if rc == MISS_EXIT:
            raise ResetMissed(None)
        if rc != 0:
            raise JobError(f"the job's client exited with {rc}; see {log}")

    def stop(self) -> None:
        super().stop()
        stop_sims(self.prefix)


def _json(raw) -> str:
    import json
    return json.dumps(raw)
