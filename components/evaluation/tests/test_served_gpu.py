"""The fridge SmolVLA checkpoint on the real sim (GPU + docker; skipped unless EVAL_GPU=1): `revolve evaluate` with `policy: smolvla`
(the policy service container from the training image, the policy client beside the sim) on 1 held-out config x EVAL_ROLLOUTS (2)
rollouts. Every rollout must reach a verdict (a failure is fine: the checkpoint was trained on the earlier simulator) without a stack
error, with finite commands (the evaluator's `finite_cmds`) and a measured inference rate; no container is left. Needs
`runs/ckpts/modal_smolvla30k_030000`, the training image and `./revolve build simulator`. The phase each rollout reached is printed
(`pytest -s`). Rollout paths must be inside the repo: `runs/eval_gpu_<ts>/` (deleted afterwards)."""
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
CKPT = os.environ.get("EVAL_SMOLVLA_CKPT", "runs/ckpts/modal_smolvla30k_030000")
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


def test_the_served_checkpoint_runs_to_a_verdict_with_finite_commands(run_dir):
    rel = lambda p: str(p.relative_to(REPO))
    (run_dir / "train.yaml").write_text(yaml.safe_dump({**TRAIN, "out": rel(run_dir / "train")}))
    (run_dir / "eval.yaml").write_text(yaml.safe_dump({
        "train": rel(run_dir / "train.yaml"), "out": rel(run_dir / "eval"), "ckpt": {"source": "local", "path": CKPT},
        "policy": {"plugin": "smolvla"}, "runner": {"n_action_steps": 20, "chunk_size_threshold": 0},
        "sets": {"heldout": {"count": 1, "sampler": {"strategy": "sobol"}, "pool": {"initial": [500, 504]}}},
        "rollout": {"plugin": "sim"}, "rollouts": ROLLOUTS, "retries": 0, "run": {"workers": 1, "headless": True}}))
    assert cli.main([str(run_dir / "eval.yaml"), "all", "--headless"], root=REPO) == 0
    block = json.loads((run_dir / "eval" / "results.json").read_text())["sets"]["heldout"]
    rows = block["rollouts"]
    assert len(rows) == ROLLOUTS
    for k in sorted(rows):
        rec = rows[k]["tries"][0]["record"]
        assert rec["error"] is None and rec["episode"] is not None, rec                      # a verdict, no stack error
        assert rec["check"] != "finite_cmds" and rec["steps"] > 0, rec                       # finite commands, the policy acted
        ep = json.loads((Path(rec["dir"]) / "episode.json").read_text())
        inf = ep["inference"]
        assert inf["n"] >= 1 and inf["hz"] > 0
        print(f"rollout {k}: passed={rec['passed']} failed_phase={rec['failed_phase']} reason={rec['reason']} "
              f"phases={rec['phases']} steps={rec['steps']} chunks={ep['chunks']} inference {inf['hz']:.1f} Hz "
              f"(mean {inf['mean_s']:.3f} s, max {inf['max_s']:.3f} s) duration {rec['duration_s']:.1f} s")
    assert (run_dir / "eval" / "report.md").is_file()
