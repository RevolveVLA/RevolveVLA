"""Plugin interfaces (small, one job each) and the value objects they exchange.

Constructor convention (registry.create): Sampler / Randomiser take no arguments; Instancer, Generator, Runner,
Converter, Validator and Check take `(spec: DatagenSpec, params: Mapping)`, the params being their yaml block's free-form keys.
Values are plain JSON types so the manifest can store them as they are."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .spec import DatagenSpec, Dim, SplitSpec


@dataclass(frozen=True)
class Config:
    """One sampled variation: the seed it was drawn for, its stratification cell and the dotted dim values."""
    split: str
    seed: int
    cell: tuple[int, ...] = ()
    values: Mapping[str, Any] = field(default_factory=dict)       # {"robot_base.x_m": 0.05, ...}

    def of(self, randomiser: str) -> dict:
        """The values of one randomiser by plain dim name: {"x_m": 0.05, ...} (what Randomiser.realise takes)."""
        prefix = randomiser + "."
        return {k[len(prefix):]: v for k, v in self.values.items() if k.startswith(prefix)}

    def to_dict(self) -> dict:
        return asdict(self) | {"cell": list(self.cell)}

    @classmethod
    def from_dict(cls, d: Mapping) -> Config:
        return cls(d["split"], d["seed"], tuple(d["cell"]), dict(d["values"]))


@dataclass(frozen=True)
class Result:
    """Outcome of one plugin call: ok or the reason it failed, what it realised, where it wrote."""
    ok: bool = True
    reason: str = ""
    realised: Mapping[str, Any] = field(default_factory=dict)     # e.g. the start pose the generator ended up with
    paths: Mapping[str, str] = field(default_factory=dict)        # artefact name -> path

    @classmethod
    def fail(cls, reason: str) -> Result:
        return cls(False, reason)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> Result:
        return cls(d["ok"], d.get("reason", ""), dict(d.get("realised", {})), dict(d.get("paths", {})))


class Sampler(ABC):
    """A sampling strategy: configs for the given seeds of a split, deterministic in (spec.seed, split, seeds).

    A draw is redrawn (same cell) when `reject(config)` returns a reason or, with the split's `min_distance`, it lies
    closer than that to a config of `avoid` (the source split's) or, for `from: <itself>`, to an earlier config of this
    call. Redrawn draws are listed in `redrawn` ({seed, cell, values, reason}) until the next call."""
    redrawn: Sequence[Mapping[str, Any]] = ()

    @abstractmethod
    def sample(self, split: SplitSpec, dims: Mapping[str, Dim], seeds: Sequence[int], base_seed: int,
               reject: Callable[[Config], str | None] | None = None, avoid: Sequence[Config] = ()) -> list[Config]:
        ...


class Randomiser(ABC):
    """One stream of variation (robot_start, ...): its hard limits and how a generator realises its values.

    `scene_field`: the scene.yaml key this stream replaces in every scene instance (None: it does not touch the scene);
    `realise(values, {"base": <that key in the base scene>})` is then the new value of the key."""
    scene_field: str | None = None

    @abstractmethod
    def limits(self, spec: DatagenSpec) -> Mapping[str, Dim]:
        """{`<randomiser>.<dim>`: Dim}: the widest range each dim may take for the spec's task and robot."""

    @abstractmethod
    def realise(self, values: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        """Dim values (by plain dim name) -> what the generator applies; `context` is the generator's own data."""

    def measure(self, realised: Mapping[str, Any], base: Any) -> Mapping[str, Any]:
        """What `realise` returned (and the base scene's value of the key) -> the plain dim values it amounts to, for the
        realised min_distance; {} if none."""
        return {}


class Instancer(ABC):
    """Writes the scene instance of a config: the base scene with each `scene_field` randomiser's realisation."""

    @abstractmethod
    def write(self, config: Config, out: Path) -> Result:
        """Write `out`/scene.yaml; Result.paths["scene"] = the file, `realised` = `realised(config)`."""

    @abstractmethod
    def realised(self, config: Config) -> Mapping[str, Any]:
        """The start the instance holds, as `<randomiser>.<dim>` values comparable with the sampled ones."""


class EpisodeClient(Protocol):
    """What a generator and the episode loop use of the live sim (revolve_common.sim_client.SimClient in a client
    container; sim_fake.FakeClient on the CPU): DATAGEN-v2.md 4.2-4.4 plus rclpy access (SimClient.node, .tf)."""

    def wait_ready(self, timeout_s: float = ...) -> None: ...
    def reset(self, timeout_s: float = ...) -> None: ...
    def start_episode(self) -> int: ...
    def wait_result(self, n: int, timeout_s: float = ...) -> dict: ...
    def wait_files(self, n: int, timeout_s: float = ...) -> dict: ...
    def now(self) -> float: ...


class Generator(ABC):
    """The expert: commands the robot of a started episode through the sim's contract topics (DATAGEN-v2.md 5). It does
    not randomise anything (the scene instance is the variation), does not record (the sim does) and does not judge
    (the evaluator does)."""

    def prepare(self, client: EpisodeClient, instance: Path, task: str, robot: str) -> None:
        """Called once per sim launch, after the sim is ready and before the first episode starts: start what is slow
        (a planner), so that it does not run on the episode's clock. Default: nothing."""

    @abstractmethod
    def drive(self, client: EpisodeClient, instance: Path, task: str, robot: str) -> Mapping[str, Any]:
        """Called after /evaluator/start of an episode on the sim of `instance` (the scene.yaml): command the robot until
        done or given up (the episode ends on its own at the evaluator's verdict or timeout; the caller waits for the
        result). -> {"expert": a name that identifies the expert version, "notes": free text (optional)}."""


class Runner(ABC):
    """Runs the episodes of scene instances on the sim: one sim per instance, the episode loop of
    core/episodes.py, relaunch on infrastructure faults. Called from the pipeline's worker threads (one call = one
    (sim, generator) pair)."""

    @abstractmethod
    def run(self, seed: int, instance: Path, generator: Generator) -> None:
        """Make the episodes of the scene instance `instance` (<dir>/scene.yaml; <dir> is the sim's --out) until
        `spec.episodes` is satisfied (core/episodes.py: finished), appending each try to <dir>/datagen.json. Episodes
        already there count (a resumed run). Raises SceneError when the scene cannot be run (infrastructure retries used
        up, the generator crashed)."""

    def stop(self) -> None:
        """Stop what running `run` calls started (their containers); called when the run is interrupted."""


class Converter(ABC):
    @abstractmethod
    def convert(self, episodes: Sequence[Path], out: Path) -> Result:
        """Recorded episodes -> the training dataset under `out`. (No converter runs yet: DATAGEN-v2.md 7 has no Rosetta
        contract for the new bags.)"""


class Validator(ABC):
    """Per-config pre-filter between sampling and generation (e.g. chassis clearance)."""

    @abstractmethod
    def validate(self, config: Config) -> Result:
        ...


class Check(ABC):
    """Whole-run check over the run state (a dict, see state.py); the verdict reads as one line of the report."""
    name: str

    @abstractmethod
    def run(self, manifest: Mapping[str, Any]) -> Result:
        ...
