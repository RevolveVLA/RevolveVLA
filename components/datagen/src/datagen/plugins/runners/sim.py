"""runner sim: the real thing (DATAGEN-v2.md 4): `./revolve sim` for the scene instance (SimSession, `--no-autostart`, its
folder as `--out`, a leased name + ROS domain) and one client container beside it that runs the episode loop and the
generator (`python3 -m datagen.scene_client`, ROS 2 Lyrical inside the simulator image).

A SimFault (clock fault, sim exit, ready / reset / result timeout) relaunches the sim into the same folder and the loop
resumes from the scene log. Params: `infra_retries` (3), `wall_s_per_attempt` (600): the client's wall budget per try.
Logs: <out>/logs/scene_<seed>.{sim,client}.log (kept across relaunches). Interrupt: `stop()` removes this run's
containers (`stop_sims(prefix)`) and no sim is launched afterwards."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from datagen.core.episodes import SceneError
from datagen.core.registry import register
from datagen.scene import base
from revolve_common.sim_session import REPO, SimSession, stop_sims

from .session_runner import RunnerStopped, SessionRunner

DATAGEN_SRC = "/repo/components/datagen/src"      # in the client container (the repo is mounted at /repo)
SETUP_S = 300                                     # bring-up, planner start, teardown: on top of the tries' budget


class _Session(SimSession):
    """A SimSession that does not (re)launch once the runner was stopped."""
    stopped = None

    def start(self):
        if self.stopped.is_set():
            raise RunnerStopped("run interrupted")
        super().start()


@register("runner", "sim")
class SimRunner(SessionRunner):
    def __init__(self, spec, params):
        super().__init__(spec, params)
        self.attempt_s = float(params.get("wall_s_per_attempt", 600))

    def logs(self, seed: int) -> Path:
        return Path(self.spec.out) / "logs" / f"scene_{seed}"

    def session(self, seed, instance, name, domain):
        scene = instance if Path(instance).is_file() else base.scene_name(self.spec)     # no instancer: the plain scene
        s = _Session(scene, self.spec.robot, self.spec.task, Path(instance).parent, name, domain, self.spec.episodes.record,
                     gui=not self.spec.run.headless, log=f"{self.logs(seed)}.sim.log")
        s.stopped = self.stopped
        return s

    def work(self, session, seed, instance, generator):
        command = ["python3", "-m", "datagen.scene_client", "--spec", json.dumps(self.spec.raw), "--seed", str(seed),
                   "--instance", str(Path(instance).resolve())]
        log = f"{self.logs(seed)}.client.log"
        client = session.client(command, pythonpath=[DATAGEN_SRC])
        try:
            rc = client.run(log, timeout=self.spec.episodes.attempts * self.attempt_s + SETUP_S,
                            watch=lambda: None if self.stopped.is_set() else session.check())
        except subprocess.TimeoutExpired:
            raise SceneError(f"the generator client ran over its wall budget; see {log}") from None
        if self.stopped.is_set():
            raise RunnerStopped("run interrupted")
        if rc != 0:
            raise SceneError(f"the generator client exited with {rc}; see {log}")

    def stop(self) -> None:
        super().stop()
        stop_sims(self.prefix)
