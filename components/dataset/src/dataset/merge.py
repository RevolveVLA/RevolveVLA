"""Finished datasets -> one dataset, in the given order, with LeRobot 0.6.1's aggregate_datasets (DATASET-v2.md
section 9). In the dataset image.

  records, problems = inputs(paths)   each input must be a finished dataset of this component (record.py: ok)
  problems = incompatible(paths)      ... with a byte-identical meta/rosetta_contract.yaml and the same fps, features
                                      (keys, shapes, dtypes) and task
  merge(paths, work)                  -> the merged dataset directory under <work>

aggregate_datasets copies each source's meta/episodes rows whole, so the labels/* columns come along; it shifts the
indices and aggregates the stats."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import record as R
from .layout import CONTRACT, REPO_ID


def inputs(paths) -> tuple[list[R.Record], list[str]]:
    """Each input's record, and a `merge:` problem for each input that is not a successful build of this component
    (no readable record, a failed build, another kind). Reads only the records: no heavy import."""
    recs, problems = [], []
    for p in paths:
        rec = R.read(p)
        if rec is None:
            problems.append(f"merge: {p}: not a dataset of this component (no readable {R.SUMMARY})")
        elif not rec.ok:
            problems.append(f"merge: {p}: not a successful build of this component (kind {rec.kind!r}, problems "
                            f"{rec.problems!r})")
        recs.append(rec)
    return recs, problems


def _facts(p: Path):
    import pandas as pd
    info = json.loads((p / "meta" / "info.json").read_text())
    feats = {k: (v.get("dtype"), list(v.get("shape") or [])) for k, v in info["features"].items()}
    tasks = sorted(str(t) for t in pd.read_parquet(p / "meta" / "tasks.parquet").index)
    return (p / CONTRACT).read_bytes(), info.get("fps"), feats, tasks


def incompatible(paths) -> list[str]:
    problems, facts = [], {}
    for p in paths:
        try:
            facts[p] = _facts(p)
        except (OSError, ValueError, KeyError) as e:
            problems.append(f"merge: {p}: its LeRobot metadata is unreadable ({type(e).__name__}: {e})")
    if problems:
        return problems
    first = paths[0]
    for p in paths[1:]:
        for i, what in enumerate(("meta/rosetta_contract.yaml (not byte-identical)", "fps", "features", "task")):
            if facts[p][i] != facts[first][i]:
                problems.append(f"merge: {p}: {what} differs from {first}"
                                + ("" if i == 0 else f": {facts[p][i]} != {facts[first][i]}"))
    return problems


def merge(paths, work: Path) -> Path:
    from lerobot.datasets.aggregate import aggregate_datasets
    dst = work / "lerobot" / REPO_ID
    aggregate_datasets(repo_ids=[f"input/{i}" for i in range(len(paths))], aggr_repo_id=REPO_ID,
                       roots=[Path(p) for p in paths], aggr_root=dst)
    shutil.copy(Path(paths[0]) / CONTRACT, dst / CONTRACT)
    return dst
