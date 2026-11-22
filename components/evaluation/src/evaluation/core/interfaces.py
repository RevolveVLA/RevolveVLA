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
    """One evaluation config: what the policy starts from (a replay-accepted check dir)."""
    config: Config
    start_dir: Path


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
    """The run the policy was trained on: where it lives (for messages) and its datagen-manifest-shaped data
    (`{spec_hash, splits: {<split>: {sampled, used, rejected}}}`; None = no manifest: sampled-only)."""
    path: Path
    data: Mapping[str, Any] | None


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
        """The set's configs + start dirs (written under `<out>/<spec.name>/`) and its checks; resumable. `abort()` stops
        what the build's workers started (generator jobs) when it fails or is interrupted."""


class Policy(ABC):
    name: str
    needs_ckpt: bool = True         # False: the policy brings its own actions (the oracle replays the expert's)
    client_only: bool = False       # True: no policy server container, the client side produces the actions


class Rollout(ABC):
    @abstractmethod
    def run(self, item: EvalItem, out: Path, *, seed: int, restore: tuple[Path, str] | None = None,
            runner: RunnerSpec, policy: Policy, ckpt: ResolvedCkpt | None, gui: bool = False) -> dict:
        """One policy run from `item.start_dir` into `out` (from the saved state `restore` = (rollout dir, phase) when
        given); the rollout record (a JSON dict: passed, phases, failed_phase, ... as rollout.run returns it)."""

    def default_workers(self, policy: Policy) -> int:
        """Rollouts to run at a time when the spec sets no `run.workers`."""
        return 1

    def rtf(self, policy: Policy) -> float | None:
        """The real-time factor the plugin assumes for its wall limit (None: it has none)."""
        return None


class Judge(ABC):
    @abstractmethod
    def score(self, record: Mapping[str, Any], rollout_dir: Path, task: Task | None) -> Verdict:
        """The success verdict of one rollout: its record, its dir and the training spec's task (None when its
        task.yaml is not found). A record with `restore` is a retry."""
