"""Plugin interfaces (small, one job each) and the value objects they exchange.

Constructor convention (REGISTRY.create): CkptSource, TrainingSource, Rollout, Policy and Judge take `(spec: EvalSpec, params: Mapping)`,
the params being their yaml block's free-form keys; EvalSet takes `(spec)`. Configs are datagen's `Config` (sampled
dims) and `Result` (ok / reason) value objects."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from datagen.core.interfaces import Config, Result

from .spec import RunnerSpec, SetSpec

if TYPE_CHECKING:
    from revolve_common.specs import Task


@dataclass(frozen=True)
class ResolvedCkpt:
    """A checkpoint ready to serve: a dir under runs/, where it came from (source, repo, commit, hashes, ...) and what
    was changed to serve it."""
    path: str
    origin: Mapping[str, Any] | str
    changes: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class EvalItem:
    """One evaluation config: the scene instance the policy starts from (`scene.yaml`) and, for a training config, the
    episodes datagen recorded on it (its manifest.json entries, paths made absolute: `bag`, `result`, `scene`)."""
    config: Config
    scene: Path
    episodes: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class Built:
    """An eval set ready to roll out: its items and the verdicts of its checks by check name."""
    items: Sequence[EvalItem]
    checks: Mapping[str, Result] = field(default_factory=dict)


@dataclass(frozen=True)
class Verdict:
    passed: bool
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingRun:
    """The run the policy was trained on: its state file (for messages; scene paths in it are relative to its dir), its
    state in datagen's shape (`{spec_hash, splits: {<split>: {sampled, used, rejected}}}`; None = no state: sampled-only)
    and its recorded episodes by seed (datagen manifest.json `episodes`, paths absolute; empty without a manifest)."""
    path: Path
    data: Mapping[str, Any] | None
    episodes: Mapping[int, Sequence[Mapping[str, Any]]] = field(default_factory=dict)


class TrainingSource(ABC):
    @abstractmethod
    def load(self, root: Path) -> TrainingRun:
        """The training run, read in memory (never written); `root` = what the spec's relative paths are relative to."""


class CkptSource(ABC):
    relabel: bool = True            # False: the dir is served as it is (the CLI wraps the others in the `relabel` plugin)

    @abstractmethod
    def resolve(self, out: Path) -> ResolvedCkpt:
        """Make the checkpoint available (download, relabel, ...) under `out` / runs/ and say where it is."""


class EvalSet(ABC):
    @abstractmethod
    def build(self, spec: SetSpec, out: Path, workers: int, abort: Callable[[], None] | None = None) -> Built:
        """The set's configs + scene instances (written under `<out>/<spec.name>/`) and its checks; resumable (state in
        `<out>/<spec.name>/state.json`). `abort()` stops what the build's workers started when it fails or is interrupted."""


class Policy(ABC):
    name: str
    needs_ckpt: bool = True         # False: the policy brings its own actions (the oracle replays the expert's)
    client_only: bool = False       # True: no policy server container, the client side produces the actions

    def client_args(self, item: EvalItem) -> list[str]:
        """The command line arguments of the policy client (rollout_client.py) for the scene `item`."""
        return []


class Rollout(ABC):
    @abstractmethod
    def run(self, item: EvalItem, out: Path, *, seed: int, runner: RunnerSpec, policy: Policy,
            ckpt: ResolvedCkpt | None, gui: bool = False) -> dict:
        """One episode of the policy from the scene instance `item.scene` into `out` (a retry is a new episode from the
        start: no state is restored); the rollout record (a JSON dict: passed, phases, failed_phase, reason, error, ...)."""

    def validate(self, policy: Policy) -> None:
        """Raise a SpecError when this rollout cannot run `policy` (checked before anything runs)."""

    def release(self, item: EvalItem) -> None:
        """The rollouts of the config `item` are over: free what `run` kept for it (the sim of its scene instance)."""

    def stop(self) -> None:
        """The run is being aborted: start nothing more."""

    def default_workers(self, policy: Policy) -> int:
        """Rollouts to run at a time when the spec sets no `run.workers`."""
        return 1


class Judge(ABC):
    @abstractmethod
    def score(self, record: Mapping[str, Any], rollout_dir: Path, task: Task | None) -> Verdict:
        """The success verdict of one rollout: its record, its dir and the training spec's task (None when its
        task.yaml is not found)."""
