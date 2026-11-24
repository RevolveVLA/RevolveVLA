"""Item: one episode of any kind of data (or, for a live recorder, one span of it) a recipe can pick. Sources produce them, the
recipe selects and splits them, an output writes them. Paths are absolute and normalised, so the output needs no source's root."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .errors import RecipeError


@dataclass(frozen=True)
class Item:
    """`kind`: datagen | rollout | phase | dagger; `source`: the recipe's name for the source; `phase`: the task phase the item
    trains (None: the whole task); `scene`: the scene instance (what splits and the leakage guard group by); `config_seed`: the
    sampling seed of that scene when known; `segment`: [t0, t1] seconds from the episode's start, None = the whole episode;
    `duration_s`: of the episode, as the evaluator timed it; `labels`: free provenance (set, rollout, trigger, ...)."""
    kind: str
    source: str
    phase: str | None
    scene: str
    config_seed: int | None
    episode_dir: str
    result: str
    bag: str
    segment: tuple[float, float] | None
    passed: bool
    task: str | None
    robot: str | None
    duration_s: float | None
    labels: Mapping[str, Any] = field(default_factory=dict)

    @property
    def seconds(self) -> float | None:
        return self.segment[1] - self.segment[0] if self.segment else self.duration_s

    def frames(self, fps: int) -> int:
        if self.seconds is None:
            raise RecipeError(f"{self.result}: no duration_s, so its frames cannot be counted (unit: frames)")
        return round(self.seconds * fps)

    def to_dict(self, fps: int | None = None) -> dict:
        d = asdict(self)
        d["segment"] = list(self.segment) if self.segment else None
        if fps and self.seconds is not None:
            d["frames"] = self.frames(fps)
        return d


@dataclass(frozen=True)
class Read:
    """What a source found: its usable items, and why the others were left out (reason -> count)."""
    items: list[Item]
    skipped: Mapping[str, int] = field(default_factory=dict)
