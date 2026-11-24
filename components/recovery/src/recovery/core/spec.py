"""RecoverySpec: the recovery yaml as frozen dataclasses, validated here (datagen's spec helpers: every error names the yaml
path of the offending key). `from.eval` is a finished evaluation run (its `out`); `phases` is the built-in phase-wise
recovery; `live` is an optional live recorder plugin; plugin refs are the only free-form blocks. `hash` fingerprints the
settings (not `out`, not `run`)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml
from datagen.core.spec import (DatagenSpec, EpisodeSpec, PluginRef, SpecError, _fail, _generator, _int, _mapping, _num,
                               _plugin, _str)
from datagen.core.spec import parse as parse_datagen

RESTARTS = ("phase_start", "before_failure")
KEEPS = ("successes", "all")
RESETS = ("warm", "relaunch")
GLOBAL = "global"               # the evaluator's name for untimed sanity failures: no phase to restart


@dataclass(frozen=True)
class PhasesSpec:
    """Phase-wise recovery: restart at `restart_from` (the failed phase's start, or `back_s` before the failure, never before
    the phase's start) if no scene joint is beyond `max_object_joint` there (the door must still be closed); the datagen
    `generator` records `episodes.per_scene` accepted episodes (`per_failure`) of each failure in up to `episodes.attempts` tries;
    `quota`: recorded episodes wanted per phase (a phase not listed is not recovered)."""
    generator: PluginRef
    episodes: EpisodeSpec
    quota: Mapping[str, int]
    restart_from: str = "before_failure"
    back_s: float = 1.0
    max_object_joint: float = 0.02

    @property
    def per_failure(self) -> int:
        return self.episodes.per_scene


@dataclass(frozen=True)
class RunSpec:
    """How the run executes, not what it makes (so not in the hash): jobs at a time, how a job reaches its restart (`warm`:
    reset the running sim and drive to it, a relaunch when that misses; `relaunch`: a sim per job), headless."""
    workers: int = 1
    reset: str = "relaunch"
    headless: bool = True


@dataclass(frozen=True)
class RecoverySpec:
    task: str
    robot: str
    eval_run: str                                       # yaml `from: {eval: <dir>}`
    phases: PhasesSpec
    out: str
    triggers: tuple[Mapping[str, Any], ...] = ({"plugin": "verdict"},)
    live: PluginRef | None = None
    keep: str = "successes"
    seed: int = 0
    run: RunSpec = RunSpec()
    runner: PluginRef = PluginRef("sim")                # runs a job's episodes on a sim
    scanner: PluginRef = PluginRef("container")         # reads the eval run's bags (rosbag2_py: the simulator image)
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)    # the parsed yaml (re-parsed in a container)

    @property
    def hash(self) -> str:
        """Stable across yaml formatting / key order; changes with any setting except `out` and `run`."""
        d = asdict(self)
        d.pop("out"), d.pop("run"), d.pop("raw")
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def datagen_spec(self) -> DatagenSpec:
        """The datagen spec the generator plugin is created with (task, robot and its own params): the recovery's jobs are
        its scene instances, so it has one placeholder split."""
        g = self.phases.generator
        e = self.phases.episodes
        return parse_datagen({"task": self.task, "robot": self.robot, "out": self.out,
                              "generator": {"plugin": g.plugin, **g.params, "attempts": e.attempts, "record": e.record,
                                            "episodes_per_scene": e.per_scene},
                              "splits": {"recovery": {"target": 1, "pool": {"initial": [0, 0]}}}}, "<recovery>")


def _phases(v, at) -> PhasesSpec:
    d = _mapping(v, at, required=("generator", "quota"),
                 optional=("restart_from", "back_s", "max_object_joint", "per_failure"))
    gen = _mapping(d["generator"], f"{at}.generator", free=True)
    if "episodes_per_scene" in gen:
        _fail(f"{at}.generator.episodes_per_scene", f"use {at}.per_failure")
    per_failure = _int(d.get("per_failure", 1), f"{at}.per_failure", 1)
    ref, episodes = _generator({**gen, "episodes_per_scene": per_failure}, f"{at}.generator")
    quota = {}
    for phase, n in _mapping(d["quota"], f"{at}.quota", free=True).items():
        quota[phase] = _int(n, f"{at}.quota.{phase}", 0)
        if phase == GLOBAL:
            _fail(f"{at}.quota.{phase}", "a global (sanity) failure has no phase to restart")
    restart = _str(d.get("restart_from", PhasesSpec.restart_from), f"{at}.restart_from")
    if restart not in RESTARTS:
        _fail(f"{at}.restart_from", f"expected one of {RESTARTS}, got {restart!r}")
    back_s, joint = _num(d.get("back_s", PhasesSpec.back_s), f"{at}.back_s"), _num(
        d.get("max_object_joint", PhasesSpec.max_object_joint), f"{at}.max_object_joint")
    for key, value in (("back_s", back_s), ("max_object_joint", joint)):
        if value < 0:
            _fail(f"{at}.{key}", f"must be >= 0, got {value}")
    return PhasesSpec(ref, episodes, quota, restart, float(back_s), float(joint))


def _run(v, at) -> RunSpec:
    d = _mapping(v, at, optional=("workers", "reset", "headless"))
    headless, reset = d.get("headless", True), d.get("reset", RunSpec.reset)
    if not isinstance(headless, bool):
        _fail(f"{at}.headless", f"expected true/false, got {headless!r}")
    if reset not in RESETS:
        _fail(f"{at}.reset", f"expected one of {RESETS}, got {reset!r}")
    return RunSpec(_int(d.get("workers", 1), f"{at}.workers", 1), reset, headless)


def _triggers(v, at) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(v, list) or not v:
        _fail(at, "expected a non-empty list of {plugin: ...}")
    return tuple(dict(_mapping(x, f"{at}[{i}]", required=("plugin",), free=True)) for i, x in enumerate(v))


def parse(raw: Any, source: str = "<spec>") -> RecoverySpec:
    d = _mapping(raw, source, required=("task", "robot", "from", "phases", "out"),
                 optional=("triggers", "live", "keep", "seed", "run", "runner", "scanner"))
    keep = _str(d.get("keep", RecoverySpec.keep), f"{source}.keep")
    if keep not in KEEPS:
        _fail(f"{source}.keep", f"expected one of {KEEPS}, got {keep!r}")
    fields = {k: _plugin(d[k], f"{source}.{k}") for k in ("live", "runner", "scanner") if k in d}
    return RecoverySpec(
        task=_str(d["task"], f"{source}.task"), robot=_str(d["robot"], f"{source}.robot"),
        eval_run=_str(_mapping(d["from"], f"{source}.from", required=("eval",))["eval"], f"{source}.from.eval"),
        phases=_phases(d["phases"], f"{source}.phases"), out=_str(d["out"], f"{source}.out"),
        triggers=_triggers(d["triggers"], f"{source}.triggers") if "triggers" in d else RecoverySpec.triggers,
        keep=keep, seed=_int(d.get("seed", 0), f"{source}.seed"), run=_run(d.get("run", {}), f"{source}.run"),
        raw=raw, **fields)


def load(path: str | Path) -> RecoverySpec:
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text())
    except (OSError, yaml.YAMLError) as e:
        raise SpecError(f"{p}: cannot read: {e}") from e
    return parse(raw, str(p))


def check_task(spec: RecoverySpec, phases) -> None:
    """The quota's phases are phases of the task."""
    if unknown := [p for p in spec.phases.quota if p not in phases]:
        raise SpecError(f"phases.quota: unknown phase {unknown} (phases of {spec.task}: {', '.join(phases)})")
