"""A datagen job (DATAGEN-v2.md 8.2): scene instances made elsewhere, as a spec the pipeline runs.

    task: open_fridge_door
    robot: mobile_panda
    episodes: [{seed: 0, scene: runs/x/datagen/scene_0/scene.yaml}, ...]     # one entry per scene instance
    episodes_per_scene: 1
    out: runs/x/datagen
    # optional: generator: {plugin: moveit, attempts: 3, record: images}, runner: {plugin: sim}, run: {workers: 2}

`load` accepts a job or a spec yaml; a job becomes a spec with one split `job` (all scenes, no sampling, no randomisers)
plus the instance paths for `Pipeline(scenes=...)`."""
from __future__ import annotations

from pathlib import Path

import yaml

from .core.errors import DatagenError
from .core.spec import DatagenSpec, SpecError, _int, _mapping, _str, load as load_spec, parse

DEFAULT_GENERATOR = "moveit"


def is_job(raw) -> bool:
    return isinstance(raw, dict) and "episodes" in raw and "splits" not in raw


def to_spec(raw: dict, root: Path, source: str = "<job>") -> tuple[DatagenSpec, dict[int, Path]]:
    d = _mapping(raw, source, required=("task", "robot", "episodes", "out"),
                 optional=("episodes_per_scene", "generator", "runner", "run", "seed"))
    items = d["episodes"]
    if not isinstance(items, list) or not items:
        raise SpecError(f"{source}.episodes: expected a non-empty list of {{seed, scene}}")
    scenes = {}
    for i, item in enumerate(items):
        e = _mapping(item, f"{source}.episodes[{i}]", required=("seed", "scene"))
        seed, path = _int(e["seed"], f"{source}.episodes[{i}].seed", 0), Path(_str(e["scene"], f"{source}.episodes[{i}].scene"))
        path = path if path.is_absolute() else root / path
        if seed in scenes:
            raise SpecError(f"{source}.episodes[{i}].seed: {seed} twice")
        if not path.is_file():
            raise DatagenError(f"{source}.episodes[{i}].scene: no such file {path}")
        scenes[seed] = path
    generator = d.get("generator", {})
    generator = dict(generator) if isinstance(generator, dict) else {"plugin": generator}
    generator.setdefault("plugin", DEFAULT_GENERATOR)
    generator.setdefault("episodes_per_scene", d.get("episodes_per_scene", 1))
    spec = {"task": d["task"], "robot": d["robot"], "out": d["out"], "generator": generator, "seed": d.get("seed", 0),
            "splits": {"job": {"target": len(scenes), "pool": {"initial": [min(scenes), max(scenes)]}}},
            **{k: d[k] for k in ("runner", "run") if k in d}}
    return parse(spec, source), scenes


def load(path: str | Path, root: Path) -> tuple[DatagenSpec, dict[int, Path] | None]:
    """(spec, scenes): `scenes` is None for a spec yaml."""
    try:
        raw = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as e:
        raise SpecError(f"{path}: cannot read: {e}") from e
    return to_spec(raw, root, str(path)) if is_job(raw) else (load_spec(path), None)
