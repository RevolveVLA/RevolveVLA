"""LeRobot datasets (e.g. the previous turn's dataset + this turn's new episodes) -> one dataset, in input
order (runs in revolve-dataset; entry.py merge):

  merge(inputs, out) -> the merged dataset dir <out>.partial/<repo id>

Ported from the earlier pipeline's merge.py:
1. LeRobot 0.6.1 aggregate_datasets (its video concat, file rotation, index shifting and stats aggregation).
   It copies each source's meta/episodes rows whole, so the labels/* columns (labels.py) come along.
2. meta/rosetta_contract.yaml from the first input (all inputs must carry the same one).
Dropped from the legacy merge: the re-fold of one-episode shards' stats from pickled in-memory episode stats
(it made the modal runner's sharded port byte-equal to the single-process port; inputs here are whole
datasets, and aggregate_datasets' stats are LeRobot's own)."""
from __future__ import annotations

import shutil
from pathlib import Path

from port import REPO_ID

CONTRACT = Path("meta") / "rosetta_contract.yaml"


def merge(inputs, out) -> Path:
    from lerobot.datasets.aggregate import aggregate_datasets

    inputs = [Path(p) for p in inputs]
    if len(inputs) < 2:
        raise SystemExit("merge: needs >= 2 input datasets")
    contracts = {p: (p / CONTRACT).read_bytes() if (p / CONTRACT).is_file() else None for p in inputs}
    if len(set(contracts.values())) != 1 or contracts[inputs[0]] is None:
        raise SystemExit(f"merge: inputs carry different (or no) {CONTRACT}: {[str(p) for p in inputs]}")
    out = Path(out)
    work = out.parent / f"{out.name}.partial"
    shutil.rmtree(work, ignore_errors=True)
    dst = work / "lerobot" / REPO_ID
    aggregate_datasets(repo_ids=[f"input/{i}" for i in range(len(inputs))], aggr_repo_id=REPO_ID, roots=inputs,
                       aggr_root=dst)
    shutil.copy(inputs[0] / CONTRACT, dst / CONTRACT)
    return dst
