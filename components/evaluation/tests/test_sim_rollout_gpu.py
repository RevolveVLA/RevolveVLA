"""The expert policy on the real sim (GPU + docker; skipped unless EVAL_GPU=1): `revolve evaluate` with `policy: expert` (the
training spec's `moveit` generator, closed loop, in the policy client container) on the base start of open_fridge_door (the only
varied dim is the arm pose, so the robot starts at the base scene's start): 1 held-out config x EVAL_ROLLOUTS (2) rollouts, one
sim, a reset between the rollouts. Every rollout must pass the evaluator (c0); results.json and report.md are written; no
container is left. Self-contained (no datagen run, no checkpoint, no bag); needs `./revolve build simulator` and the scene's
assets. Rollout paths must be inside the repo, so the run dir is `runs/eval_gpu_<ts>/` (deleted afterwards)."""
import json
import os
import shutil
import time
from pathlib import Path

import pytest
import yaml

from evaluation import cli
from revolve_common.sim_session import Docker

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.environ.get("EVAL_GPU") != "1", reason="needs a GPU: EVAL_GPU=1")]

REPO = Path(__file__).resolve().parents[3]
ROLLOUTS = int(os.environ.get("EVAL_ROLLOUTS", "2"))
TRAIN = {"task": "open_fridge_door", "robot": "mobile_panda", "seed": 0,
         "generator": {"plugin": "moveit", "attempts": 1, "episodes_per_scene": 1, "record": "no-images"},
         "randomisers": {"robot_start": {"arm": {"choices": ["folded"]}}}, "validators": [{"plugin": "collision_free"}],
         "splits": {"train": {"target": 1, "pool": {"initial": [0, 2], "replacement": [3, 7]}}}}


def leftovers():
    return [n for n in Docker().names() if n.startswith(("dg-", "ev-"))]


@pytest.fixture
def run_dir():
    assert not leftovers(), f"containers of another run: {leftovers()}"
    d = REPO / "runs" / f"eval_gpu_{time.strftime('%Y%m%d_%H%M%S')}"
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
    assert not leftovers(), f"containers left: {leftovers()}"


def test_the_expert_passes_the_base_start_with_a_reset_between_rollouts(run_dir):
    rel = lambda p: str(p.relative_to(REPO))
    (run_dir / "train.yaml").write_text(yaml.safe_dump({**TRAIN, "out": rel(run_dir / "train")}))
    (run_dir / "eval.yaml").write_text(yaml.safe_dump({
        "train": rel(run_dir / "train.yaml"), "out": rel(run_dir / "eval"), "policy": {"plugin": "expert"},
        "sets": {"heldout": {"count": 1, "sampler": {"strategy": "sobol"}, "pool": {"initial": [500, 504]}}},
        "rollout": {"plugin": "sim", "record": "no-images"}, "rollouts": ROLLOUTS, "retries": 0,
        "run": {"workers": 1, "headless": True}}))
    assert cli.main([str(run_dir / "eval.yaml"), "all", "--headless"], root=REPO) == 0
    block = json.loads((run_dir / "eval" / "results.json").read_text())["sets"]["heldout"]
    rows = block["rollouts"]
    assert len(rows) == ROLLOUTS and all(e["passed"] for e in rows.values()), rows
    recs = [rows[k]["tries"][0]["record"] for k in sorted(rows)]
    assert [r["resets"] for r in recs] == list(range(ROLLOUTS))                     # a reset between the rollouts
    assert [r["episode"] for r in recs] == list(range(ROLLOUTS)) and all(r["error"] is None for r in recs)
    assert all(e["tries"][0]["termination"] == "success" for e in rows.values())
    assert block["metrics"]["first_try_success"] == 1.0 and (run_dir / "eval" / "report.md").is_file()
