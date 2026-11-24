"""A recovery job: the scene instance of one failure whose `robot_start` is its restart state (`<job dir>/scene.yaml`), and
the episodes the datagen generator records on it. The tries run on datagen's episode loop (core/episodes.py), logged in
`<job dir>/datagen.json` with the sim folder that holds each try's files (`sim`: slots share a sim across jobs).

A job reaches its restart either with a sim launched at it (`relaunch`: every try after the first resets to the instance's
start, which is the restart) or on a running sim of the same world by a warm reset (`warm`: every try, the first too, goes
through `SimClient.reset_to(target)`; a miss raises `WarmResetMissed` and the host relaunches the sim at the restart)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from datagen.core import episodes
from datagen.core.interfaces import EpisodeClient, Generator
from datagen.core.spec import EpisodeSpec
from revolve_common.sim_client import WarmResetMissed
from revolve_common.warm_reset import ResetTarget

from .errors import JobError
from .failures import Failure
from .robot import Arm

RESETS = "resets.json"      # <job dir>: [{attempt, mode, ok, ...the reset report}] of the warm resets
DIGITS = 6


@dataclass(frozen=True)
class Job:
    """`index`: the run order (the episode loop's seed). `key`: the failure's id. `dir`: the job's folder."""
    index: int
    key: str
    phase: str
    dir: Path
    target: ResetTarget

    @property
    def scene(self) -> Path:
        return self.dir / "scene.yaml"

    def to_json(self) -> str:
        return json.dumps(asdict(self) | {"dir": str(self.dir)})

    @classmethod
    def from_json(cls, text: str) -> Job:
        d = json.loads(text)
        t = d["target"]
        return cls(d["index"], d["key"], d["phase"], Path(d["dir"]),
                   ResetTarget(t["robot"], tuple(t["base"]), tuple(t["arm"]), tuple(t["arm_joints"]), t["arm_controller"]))


def make(index: int, failure: Failure, out: Path, robot: str, arm: Arm) -> Job:
    """The job of `failure` and its scene instance (written: the failure's scene with the restart as `robot_start`)."""
    restart = failure.restart
    job = Job(index, failure.id, failure.phase, out / failure.phase / failure.id,
              ResetTarget(robot, restart.base, restart.arm, arm.joints, arm.controller))
    try:
        scene = yaml.safe_load(Path(failure.scene).read_text())
        scene["robot_start"] = {"base": [round(v, DIGITS) for v in restart.base],
                                "arm": [round(v, DIGITS) for v in restart.arm]}
    except (OSError, yaml.YAMLError, TypeError) as e:
        raise JobError(f"{failure.id}: cannot read the scene {failure.scene}: {e}") from None
    job.dir.mkdir(parents=True, exist_ok=True)
    job.scene.write_text(f"# recovery restart of {failure.id} (phase {failure.phase}, t {restart.t:.2f} s)\n"
                         + yaml.safe_dump(scene, sort_keys=False, default_flow_style=None))
    return job


class Warm:
    """The episode client of a warm job: `reset()` is `reset_to` the job's restart; the reports are appended to the job's
    RESETS file. A miss raises WarmResetMissed. Everything else is the real client's."""

    def __init__(self, client: EpisodeClient, job: Job):
        self.client, self.job = client, job

    def reset(self, timeout_s: float = 0) -> None:
        report = self.client.reset_to(self.job.target)
        path = self.job.dir / RESETS
        done = json.loads(path.read_text()) if path.is_file() else []
        path.write_text(json.dumps([*done, {"attempt": len(episodes.read(self.job.dir)["tries"]), "mode": "warm",
                                            **report.to_dict()}], indent=1))
        if not report.ok:
            raise WarmResetMissed(report)

    def __getattr__(self, name: str):
        return getattr(self.client, name)


def run_job(client: EpisodeClient, generator: Generator, job: Job, task: str, robot: str, episodes_spec: EpisodeSpec,
            sim_out: Path, warm: bool = False) -> dict:
    """The tries of `job` on the sim of `client` (its files in `sim_out`) until the job is finished; `warm`: reach the restart by `reset_to`."""
    return episodes.run_episodes(Warm(client, job) if warm else client, generator, task, robot, job.scene, job.index,
                                 episodes_spec, reset_first=warm, extra={"sim": str(sim_out)})


def resets(job_dir: Path) -> list[dict]:
    path = Path(job_dir) / RESETS
    return json.loads(path.read_text()) if path.is_file() else []
