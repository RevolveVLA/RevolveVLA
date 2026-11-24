"""A recovery job: the scene instance of one failure whose `robot_start` is its restart state (`<job dir>/scene.yaml`), and
the episodes the datagen generator records on it. The tries run on datagen's episode loop (core/episodes.py), logged in
`<job dir>/datagen.json` with the sim folder that holds each try's files (`sim`: slots share a sim across jobs).

A job reaches its restart either with a sim launched at it (`relaunch`: every try after the first resets to the instance's
start, which is the restart) or on a running sim of the same world by a warm reset (`warm`: every try, the first too, goes
through `reset_to(target)`; a miss raises `ResetMissed` and the host relaunches the sim at the restart). The warm reset
itself is the client's (`SimClient.reset_to`); here it is the injected `Resetter`."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml
from datagen.core import episodes
from datagen.core.interfaces import EpisodeClient, Generator
from datagen.core.spec import EpisodeSpec

from .errors import JobError
from .failures import Failure
from .robot import Arm

MISS_EXIT = 79              # exit code of a client whose warm reset missed (the host relaunches the sim at the restart)
RESETS = "resets.json"      # <job dir>: [{attempt, mode, ok, ...the reset report}] of the warm resets
DIGITS = 6


class ResetMissed(RuntimeError):
    """A warm reset did not reach the restart state."""

    def __init__(self, report):
        self.report = report
        super().__init__(getattr(report, "reason", None) or "missed the restart state")


@dataclass(frozen=True)
class Target:
    """The start a warm reset reaches: the base pose [x, y, yaw] in the world, the arm joints and where the arm is
    commanded (revolve_common.warm_reset.ResetTarget has these fields)."""
    robot: str
    base: tuple[float, float, float]
    arm: tuple[float, ...]
    arm_joints: tuple[str, ...]
    arm_controller: str


Resetter = Callable[[Target], Any]          # the warm reset: a report with `ok` (and `reason`, `to_dict()`)


@dataclass(frozen=True)
class Job:
    """`index`: the run order (the episode loop's seed). `key`: the failure's id. `dir`: the job's folder."""
    index: int
    key: str
    phase: str
    dir: Path
    target: Target

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
                   Target(t["robot"], tuple(t["base"]), tuple(t["arm"]), tuple(t["arm_joints"]), t["arm_controller"]))


def make(index: int, failure: Failure, out: Path, robot: str, arm: Arm) -> Job:
    """The job of `failure` and its scene instance (written: the failure's scene with the restart as `robot_start`)."""
    restart = failure.restart
    job = Job(index, failure.id, failure.phase, out / failure.phase / failure.id,
              Target(robot, restart.base, restart.arm, arm.joints, arm.controller))
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


class Recovering(Generator):
    """The generator as a recovery episode drives it. The seam where a generator's resume-from-the-current-state entry
    replaces its normal `drive`; until a generator has one, it plans from the scene's start like a datagen episode (the
    scene's start is the restart)."""

    def __init__(self, generator: Generator):
        self.generator = generator

    def prepare(self, client, instance, task, robot) -> None:
        self.generator.prepare(client, instance, task, robot)

    def drive(self, client, instance, task, robot) -> Mapping[str, Any]:
        return self.generator.drive(client, instance, task, robot)


class Warm:
    """The episode client of a warm job: `reset()` is a warm reset to the job's restart; the reports are appended to the
    job's RESETS file. A miss raises ResetMissed. Everything else is the real client's."""

    def __init__(self, client: EpisodeClient, resetter: Resetter, job: Job):
        self.client, self.resetter, self.job = client, resetter, job

    def reset(self, timeout_s: float = 0) -> None:
        report = self.resetter(self.job.target)
        path = self.job.dir / RESETS
        done = json.loads(path.read_text()) if path.is_file() else []
        detail = report.to_dict() if hasattr(report, "to_dict") else {"reason": getattr(report, "reason", None)}
        path.write_text(json.dumps([*done, {"attempt": len(episodes.read(self.job.dir)["tries"]), "mode": "warm",
                                            **detail, "ok": bool(report.ok)}], indent=1))
        if not report.ok:
            raise ResetMissed(report)

    def __getattr__(self, name: str):
        return getattr(self.client, name)


def run_job(client: EpisodeClient, generator: Generator, job: Job, task: str, robot: str, episodes_spec: EpisodeSpec,
            sim_out: Path, resetter: Resetter | None = None) -> dict:
    """The tries of `job` on the sim of `client` (its files in `sim_out`) until the job is finished; `resetter`: warm job."""
    warm = Warm(client, resetter, job) if resetter else client
    return episodes.run_episodes(warm, Recovering(generator), task, robot, job.scene, job.index, episodes_spec,
                                 reset_first=resetter is not None, extra={"sim": str(sim_out)})


def resets(job_dir: Path) -> list[dict]:
    path = Path(job_dir) / RESETS
    return json.loads(path.read_text()) if path.is_file() else []


def exit_after_miss(error: ResetMissed) -> None:
    """End a client process whose warm reset missed (the host relaunches the sim at the restart). os._exit like run_client."""
    import sys
    print(f"warm reset missed: {error}", file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(MISS_EXIT)
