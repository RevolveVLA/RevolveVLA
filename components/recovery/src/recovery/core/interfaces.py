"""Plugin interfaces. Constructor convention (REGISTRY.create): Runner, Scanner and LiveRecorder take `(spec: RecoverySpec,
params: Mapping)`, the params being their yaml block's free-form keys. The datagen generator that records a recovery
episode is datagen's own plugin (`phases.generator`)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol

from datagen.core.interfaces import Generator

if TYPE_CHECKING:
    from .failures import Scan
    from .jobs import Job

# a rollout client's `drive(n)`: act until episode n's result is out -> what it adds to episode.json
Drive = Callable[[int], Mapping[str, Any]]


class Scanner(ABC):
    @abstractmethod
    def scan(self, out: Path) -> Scan:
        """The failures of the spec's eval run, read from its bags (the bag reader needs the simulator image); `out`: the
        run's folder (logs, the scan's own files)."""


class Runner(ABC):
    @abstractmethod
    def run(self, job: Job, generator: Generator) -> None:
        """Record the episodes of `job` (<job.dir>/scene.yaml: the failure's scene starting at the restart state) until
        `spec.phases.episodes` is satisfied (datagen's core/episodes.py: finished), appending every try to
        <job.dir>/datagen.json. Tries already there count (a resumed run). Raises JobError when the job cannot be run."""

    def stop(self) -> None:
        """Stop what running `run` calls started (their containers); called when the run is interrupted."""

    def close(self) -> None:
        """The run is over: free what the runner kept (the sims of its slots)."""


@dataclass(frozen=True)
class LiveContext:
    """What a live recorder gets of the episode it supervises (in the rollout's client container)."""
    client: Any                 # revolve_common.sim_client.SimClient
    task: str
    robot: str
    instance: Path              # the episode's scene.yaml
    out: Path                   # where the episode's sidecar files go


class LiveRecorder(Protocol):
    """A recorder that runs INSIDE a rollout (e.g. DAgger: supervise the policy, take over with an expert, flag the
    interventions). Registry kind `live`; the rollout asks for it through this interface only."""

    def validate(self, spec) -> None:
        """Raise a SpecError when this recorder cannot work with `spec` (a RecoverySpec); checked before anything runs."""

    def driver(self, policy_drive: Drive, ctx: LiveContext) -> Drive:
        """The drive of the episode: `policy_drive` supervised by this recorder (it may hand over to an expert)."""

    def finish(self, episode: Mapping[str, Any]) -> Mapping[str, Any]:
        """The flags of the finished episode (`episode`: its episode.json), JSON-able: they go into the sidecar file."""


class Expert(Protocol):
    """Takes the robot over from where it is and finishes the task (a live recorder's expert; registry kind `expert`)."""

    def takeover(self, ctx: LiveContext, why: str) -> Mapping[str, Any]:
        """Drive until the episode's result is out -> {"expert": a name that identifies the expert version, "notes": ...}."""
