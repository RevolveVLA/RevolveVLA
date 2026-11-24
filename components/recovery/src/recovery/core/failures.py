"""The failures of a finished evaluation run (`<run>/results.json`): every try the evaluator failed, read from its recorded
bag by the spec's triggers (an episode the evaluator passed is never a failure; a trigger fires at or before the verdict, so
it can only move the failure point earlier or name a different phase), with the restart state the bag has for it.
Runs where the bag can be read (rosbag2_py: the simulator image); `read_messages` is injected."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Collection, Iterable, Iterator, Mapping

from revolve_common.supervision import BagSource, Supervisor

from .errors import RecoveryError
from .restart import Restart, restart_at, restart_time
from .spec import RecoverySpec


@dataclass(frozen=True)
class Failure:
    """A failed try of the eval run. `id`: `<set>_<rollout>_t<try>` (unique in the run; names its job). `scene`: the eval
    config's scene instance. `t_fail`, `phase_t_start`: episode seconds. `trigger` / `reason`: what named the failure."""
    id: str
    set: str
    rollout: str
    attempt: int
    scene: str
    phase: str
    trigger: str
    reason: str
    t_fail: float
    phase_t_start: float
    bag: str
    result: str
    restart: Restart

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Failure:
        return cls(**{**d, "restart": Restart.from_dict(d["restart"])})


@dataclass
class Scan:
    failures: list[Failure] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)        # tries seen: passed, failed, errors, no_result, untriggered

    def to_dict(self) -> dict:
        return {"failures": [f.to_dict() for f in self.failures], "counts": dict(self.counts)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Scan:
        return cls([Failure.from_dict(f) for f in d["failures"]], Counter(d["counts"]))


def tries(run: Path, only: Collection[tuple[str, str]] | None = None) -> Iterator[tuple[str, str, int, Mapping]]:
    """(set, rollout key, try index, try) of every try in `<run>/results.json`, in sorted order; `only`: the (set, rollout key)s wanted."""
    try:
        results = json.loads((Path(run) / "results.json").read_text())
    except (OSError, ValueError) as e:
        raise RecoveryError(f"{run}: not an evaluation run ({e})") from None
    for set_name, block in sorted(results["sets"].items()):
        for key, entry in sorted(block.get("rollouts", {}).items()):
            for i, t in enumerate(entry["tries"] if only is None or (set_name, key) in only else ()):
                yield set_name, key, i, t


def phase_start(result: Mapping, phase: str) -> float:
    return next((p["t_start"] for p in result["phases"] if p["name"] == phase and p["t_start"] is not None), 0.0)


def scan(spec: RecoverySpec, supervisor: Supervisor, arm_joints: Iterable[str], run: Path, read_messages,
         only: Collection[tuple[str, str]] | None = None) -> Scan:
    """The failures of `run` (of the (set, rollout key)s `only`, default all): its failed tries, each run through the supervisor
    (`spec.triggers`) over its bag."""
    arm, found, p = tuple(arm_joints), Scan(), spec.phases
    for set_name, key, i, t in tries(run, only):
        record = t["record"]
        if record.get("error"):
            found.counts["errors"] += 1
            continue
        result_file, bag = record.get("result_file"), record.get("bag")
        if not (result_file and bag and Path(result_file).is_file() and Path(bag).is_dir()):
            found.counts["no_result"] += 1
            continue
        result = json.loads(Path(result_file).read_text())
        if (result.get("task"), result.get("robot")) not in {(spec.task, spec.robot), (None, None)}:
            raise RecoveryError(f"{result_file}: an episode of {result.get('task')} / {result.get('robot')}, the spec is "
                                f"for {spec.task} / {spec.robot}")
        if result["passed"]:
            found.counts["passed"] += 1
            continue
        found.counts["failed"] += 1
        source = BagSource(read_messages, bag, spec.robot, result)
        firing = supervisor.scan(source.ticks())
        if firing is None:
            found.counts["untriggered"] += 1
            continue
        start = phase_start(result, firing.phase)
        at = restart_time(p.restart_from, p.back_s, firing.t, start)
        found.failures.append(Failure(
            f"{set_name}_{key}_t{i}", set_name, key, i, str(record["scene"]), firing.phase, firing.trigger, firing.reason,
            round(firing.t, 3), start, bag, result_file, restart_at(source, at, arm, p.max_object_joint)))
    return found
