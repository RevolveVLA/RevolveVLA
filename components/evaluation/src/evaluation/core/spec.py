"""EvalSpec: the eval yaml as frozen dataclasses. `train:` points at the TRAINING datagen spec, which supplies the
task, robot, randomisers (the DR ranges), generator and training manifest; the eval sets are drawn inside those ranges.

Every error names the yaml path of the offending key (datagen's SpecError); plugin params are the only free-form
blocks. `hash` fingerprints the settings and the training spec's hash (not `out`, not `run`)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml
from datagen.core.spec import (DatagenSpec, Dim, MinDistance, PluginRef, PoolSpec, Range, SamplerSpec, SpecError, _dim,
                               _fail, _int, _mapping, _num, _plugin, _sampler, _seed_range, _str, contains)
from datagen.core.spec import load as load_train

from .task import load_task

NEEDS_SUCCESS = ("c1", "c2")                            # the judges that score a task's `success:` rule
# lerobot 0.6.1 (revolve-training image) lerobot.async_inference.configs.AGGREGATE_FUNCTIONS, the registry Rosetta's
# `aggregate_fn_name` is looked up in (the host venv has no lerobot to import it from)
AGGREGATE_FNS = ("weighted_average", "latest_only", "average", "conservative")
DEFAULT_AGGREGATE = "weighted_average"                  # Rosetta's, used when `aggregate` is absent


@dataclass(frozen=True)
class CkptSpec:
    source: str                                         # the ckpt plugin: local | hf | ...
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunnerSpec:
    """Rosetta's policy runner: actions per chunk (the first is dropped, so 21 = `n_action_steps` 20 executed), the queue
    fraction below which the next chunk is requested (0: only when empty, so chunks do not overlap) and, for overlapping
    chunks (threshold > 0), the `aggregate` function of lerobot's registry that blends them (None: Rosetta's default)."""
    actions_per_chunk: int = 21
    chunk_size_threshold: float = 0.0
    aggregate: str | None = None

    @property
    def n_action_steps(self) -> int:
        return self.actions_per_chunk - 1

    @property
    def warning(self) -> str | None:
        if self.aggregate and self.chunk_size_threshold == 0:
            return (f"runner: aggregate {self.aggregate!r} has no effect with chunk_size_threshold 0 "
                    "(chunks do not overlap): set a threshold > 0 to ensemble")
        return None

    def describe(self) -> str:
        agg = self.aggregate or (f"{DEFAULT_AGGREGATE} (default)" if self.chunk_size_threshold else "none (no overlap)")
        return (f"{self.n_action_steps} action steps ({self.actions_per_chunk} per chunk), "
                f"chunk_size_threshold {self.chunk_size_threshold}, aggregate {agg}")

    def params(self) -> dict[str, Any]:
        """The overrides of rollout.py's RUNNER_PARAMS (the client's runner_params.yaml)."""
        return {"actions_per_chunk": self.actions_per_chunk, "chunk_size_threshold": self.chunk_size_threshold,
                **({"aggregate_fn_name": self.aggregate} if self.aggregate else {})}


@dataclass(frozen=True)
class RunSpec:
    """How the run executes, not what it evaluates (so not in the hash): rollouts in parallel (None: by free GPU memory)."""
    workers: int | None = None
    headless: bool = True


@dataclass(frozen=True)
class SetSpec:
    """An eval set. kind `heldout`: `count` new configs from `pool`, drawn by `sampler` inside the training ranges (or the
    narrower `randomisers`), `min_distance` from the training configs. kind `train`: `count` of the training split `source`."""
    name: str
    kind: str
    count: int
    plugin: str = ""                                    # the set plugin (default: the kind)
    source: str | None = None                           # kind train: the training split
    sampler: SamplerSpec = SamplerSpec()
    pool: PoolSpec | None = None
    min_distance: MinDistance | None = None
    randomisers: Mapping[str, Mapping[str, Dim]] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalSpec:
    train: DatagenSpec
    out: str
    sets: Mapping[str, SetSpec]
    train_manifest: str = ""                            # the `datagen` training plugin's manifest (default <train.out>/state.json)
    training: PluginRef = PluginRef("datagen")          # where the training configs come from: datagen | legacy_layouts
    ckpt: CkptSpec | None = None                        # None only for a policy that needs none (the oracle)
    policy: PluginRef = PluginRef("smolvla")
    rollout: PluginRef = PluginRef("docker")
    judge: PluginRef = PluginRef("c0")                  # yaml `judge: {criterion: c0|c1|c2}`: the plugin is the criterion
    rollouts: int = 1                                   # per config
    retries: int = 0
    runner: RunnerSpec = RunnerSpec()
    run: RunSpec = RunSpec()

    @property
    def hash(self) -> str:
        """Stable across yaml formatting / key order; changes with any setting except `out` and `run`."""
        d = asdict(self) | {"train": self.train.hash}
        d.pop("out"), d.pop("run")
        if d["runner"]["aggregate"] is None:                # as before the field existed (an existing out keeps its hash)
            del d["runner"]["aggregate"]
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ parsing
def _ckpt(v, at) -> CkptSpec:
    d = _mapping(v, at, required=("source",), free=True)
    return CkptSpec(_str(d["source"], f"{at}.source"), {k: x for k, x in d.items() if k != "source"})


def _judge(v, task, at) -> PluginRef:
    """The default is c2 when the task has a `success:` rule, else c0; c1/c2 need the rule (task None = not found: unchecked)."""
    if v is None:
        return PluginRef("c2" if getattr(task, "success", None) else "c0")
    d = _mapping(v, at, required=("criterion",), free=True)
    criterion = _str(d["criterion"], f"{at}.criterion")
    if criterion in NEEDS_SUCCESS and task and getattr(task, "success", None) is None:
        _fail(f"{at}.criterion", f"{criterion} needs a `success:` block in task {task.name} (use c0)")
    return PluginRef(criterion, {k: x for k, x in d.items() if k != "criterion"})


def _runner(v, at) -> RunnerSpec:
    d = _mapping(v, at, optional=("n_action_steps", "actions_per_chunk", "chunk_size_threshold", "temporal_ensembling"))
    apc = _int(d.get("actions_per_chunk", RunnerSpec.actions_per_chunk), f"{at}.actions_per_chunk", 1)
    if "n_action_steps" in d:
        n = _int(d["n_action_steps"], f"{at}.n_action_steps", 1)
        if "actions_per_chunk" in d and apc != n + 1:
            _fail(at, f"n_action_steps {n} and actions_per_chunk {apc} disagree (actions_per_chunk = n_action_steps + 1: "
                      "give one of the two keys)")
        apc = n + 1
    thr = _num(d.get("chunk_size_threshold", RunnerSpec.chunk_size_threshold), f"{at}.chunk_size_threshold")
    if not 0 <= thr <= 1:
        _fail(f"{at}.chunk_size_threshold", f"must be in [0, 1], got {thr}")
    agg = None
    if "temporal_ensembling" in d:
        agg = _str(_mapping(d["temporal_ensembling"], f"{at}.temporal_ensembling", required=("aggregate",))["aggregate"],
                   f"{at}.temporal_ensembling.aggregate")
        if agg not in AGGREGATE_FNS:
            _fail(f"{at}.temporal_ensembling.aggregate", f"unknown aggregate {agg!r} (known: {list(AGGREGATE_FNS)})")
    return RunnerSpec(apc, float(thr), agg)


def _run(v, at) -> RunSpec:
    d = _mapping(v, at, optional=("workers", "headless"))
    headless = d.get("headless", True)
    if not isinstance(headless, bool):
        _fail(f"{at}.headless", f"expected true/false, got {headless!r}")
    return RunSpec(_int(d["workers"], f"{at}.workers", 1) if d.get("workers") is not None else None, headless)


def _pool(v, count, at) -> PoolSpec:
    p = _mapping(v, at, required=("initial",), optional=("replacement",))
    pool = PoolSpec(_seed_range(p["initial"], f"{at}.initial"),
                    _seed_range(p["replacement"], f"{at}.replacement") if "replacement" in p else None)
    if len(pool.seeds()) < count:
        _fail(at, f"{len(pool.seeds())} seeds cannot reach count {count}")
    return pool


def _min_distance(v, train, at) -> MinDistance:
    m = _mapping(v, at, required=("value",), optional=("from", "realised"))
    value = _num(m["value"], f"{at}.value")
    if value <= 0:
        _fail(f"{at}.value", f"must be > 0, got {value}")
    realised = m.get("realised", False)
    if not isinstance(realised, bool):
        _fail(f"{at}.realised", f"expected true/false, got {realised!r}")
    source = _str(m.get("from", "train"), f"{at}.from")
    if source not in [s.name for s in train.splits]:
        _fail(f"{at}.from", f"{source!r} is not a split of the training spec (splits: {[s.name for s in train.splits]})")
    return MinDistance(source, value, realised)


def _inside(dim: Dim, train: Dim) -> bool:
    if isinstance(dim, Range):
        return isinstance(train, Range) and train.lo <= dim.lo and dim.hi <= train.hi
    return all(contains(train, v) for v in dim.values)


def _randomisers(v, train, at):
    dims, rand = train.dims(), {}
    for r, sub in _mapping(v, at, free=True).items():
        for k, x in _mapping(sub, f"{at}.{r}", free=True).items():
            if f"{r}.{k}" not in dims:
                _fail(f"{at}.{r}.{k}", f"not a dim of the training spec (known: {sorted(dims)})")
            dim = _dim(x, f"{at}.{r}.{k}")
            if not _inside(dim, dims[f"{r}.{k}"]):
                _fail(f"{at}.{r}.{k}", f"{dim} is outside the training range {dims[f'{r}.{k}']}")
            rand.setdefault(r, {})[k] = dim
    return rand


def _set(name, v, train, at) -> SetSpec:
    if isinstance(v, dict) and "from" in v:                          # exact training starts
        d = _mapping(v, at, required=("from", "count"), optional=("plugin",))
        source = _str(d["from"], f"{at}.from")
        known = {s.name: s for s in train.splits}
        if source not in known:
            _fail(f"{at}.from", f"{source!r} is not a split of the training spec (splits: {list(known)})")
        count = _int(d["count"], f"{at}.count", 1)
        if count > known[source].target:
            _fail(f"{at}.count", f"{count} > the {known[source].target} configs of training split {source!r}")
        return SetSpec(name, "train", count, _str(d.get("plugin", "train"), f"{at}.plugin"), source)
    d = _mapping(v, at, required=("count", "pool"), optional=("sampler", "min_distance", "randomisers", "plugin"))
    count = _int(d["count"], f"{at}.count", 1)
    md = _min_distance(d["min_distance"], train, f"{at}.min_distance") if "min_distance" in d else None
    return SetSpec(name, "heldout", count, _str(d.get("plugin", "heldout"), f"{at}.plugin"),
                   sampler=_sampler(d.get("sampler", {}), f"{at}.sampler", train.dims()),
                   pool=_pool(d["pool"], count, f"{at}.pool"), min_distance=md,
                   randomisers=_randomisers(d.get("randomisers", {}), train, f"{at}.randomisers"))


def _check_pools(train: DatagenSpec, sets, at):
    """Held-out pools overlap neither the pool of a recorded (= trained on) training split nor each other."""
    taken = [(lo, hi, f"training split {s.name!r}") for s in train.splits if s.record for lo, hi in s.pool.ranges()]
    for s in sets:
        for lo, hi in s.pool.ranges() if s.pool else ():
            for olo, ohi, other in taken:
                if lo <= ohi and olo <= hi:
                    _fail(f"{at}.sets.{s.name}.pool", f"seeds [{lo}, {hi}] overlap [{olo}, {ohi}] of {other}")
            taken.append((lo, hi, f"set {s.name!r}"))


def parse(raw: Any, source: str = "<spec>", root: Path | None = None) -> EvalSpec:
    """`root` (default cwd) is what the relative `train:` path is relative to."""
    d = _mapping(raw, source, required=("train", "sets", "out"),
                 optional=("train_manifest", "training", "ckpt", "policy", "rollout", "judge", "rollouts", "retries", "runner", "run"))
    train_path = Path(_str(d["train"], f"{source}.train"))
    train = load_train(train_path if train_path.is_absolute() else (root or Path.cwd()) / train_path)
    raw_sets = _mapping(d["sets"], f"{source}.sets", free=True)
    if not raw_sets:
        _fail(f"{source}.sets", "needs at least one set")
    sets = {n: _set(n, v, train, f"{source}.sets.{n}") for n, v in raw_sets.items()}
    _check_pools(train, sets.values(), source)
    fields = {k: _plugin(d[k], f"{source}.{k}") for k in ("training", "policy", "rollout") if k in d}
    return EvalSpec(
        train=train, out=_str(d["out"], f"{source}.out"), sets=sets,
        train_manifest=_str(d.get("train_manifest", f"{train.out}/state.json"), f"{source}.train_manifest"),
        ckpt=_ckpt(d["ckpt"], f"{source}.ckpt") if "ckpt" in d else None,
        judge=_judge(d.get("judge"), load_task(train.task, root or Path.cwd()), f"{source}.judge"),
        rollouts=_int(d.get("rollouts", 1), f"{source}.rollouts", 1), retries=_int(d.get("retries", 0), f"{source}.retries", 0),
        runner=_runner(d.get("runner", {}), f"{source}.runner"), run=_run(d.get("run", {}), f"{source}.run"), **fields)


def load(path: str | Path, root: Path | None = None) -> EvalSpec:
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text())
    except (OSError, yaml.YAMLError) as e:
        raise SpecError(f"{p}: cannot read: {e}") from e
    return parse(raw, str(p), root)
