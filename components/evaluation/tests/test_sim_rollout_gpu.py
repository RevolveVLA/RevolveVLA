"""The oracle on the real sim (GPU + docker; skipped unless EVAL_GPU=1): datagen records EVAL_N (2) scene instances of
open_fridge_door with the moveit expert (`--record no-images`), then `revolve evaluate` replays each one's own bag actions
(policy oracle: streamed `joint_trajectory` + `cmd_vel` at 30 Hz from a client container) for 2 rollouts per instance: one sim
per instance, a reset between its rollouts. Every rollout must pass the evaluator; the terminations, the episode numbers and
the reset counters are written to runs/eval_gpu_<ts>/numbers.json (the run dirs are deleted afterwards). Needs
`./revolve build simulator` and the scene's assets."""
import json
import os
import shutil
import time
from pathlib import Path

import pytest
import yaml

from datagen import cli as datagen_cli
from evaluation import cli
from revolve_common.sim_session import Docker

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.environ.get("EVAL_GPU") != "1", reason="needs a GPU: EVAL_GPU=1")]

REPO = Path(__file__).resolve().parents[3]
RUN = REPO / "runs" / f"eval_gpu_{time.strftime('%Y%m%d_%H%M%S')}"
N = int(os.environ.get("EVAL_N", "2"))


def leftovers():
    return [n for n in Docker().names() if n.startswith(("dg-", "ev-"))]


@pytest.fixture(autouse=True)
def no_containers_left():
    assert not leftovers(), f"containers of another run: {leftovers()}"
    yield
    assert not leftovers(), f"containers left: {leftovers()}"


def test_the_oracle_passes_the_evaluator_with_a_reset_between_rollouts():
    dspec = yaml.safe_load((REPO / "specs" / "datagen" / "smoke_v2.yaml").read_text())
    dspec.update(out=str((RUN / "datagen").relative_to(REPO)),
                 generator={**dspec["generator"], "record": "no-images", "attempts": 3})
    dspec["splits"] = {"train": {"target": N, "pool": {"initial": [0, N - 1], "replacement": [N, N + 6]}}}
    RUN.mkdir(parents=True)
    (RUN / "datagen.yaml").write_text(yaml.safe_dump(dspec))
    try:
        assert datagen_cli.main([str(RUN / "datagen.yaml"), "all", "--headless", "--workers", "1"], root=REPO) == 0
        espec = {"train": str((RUN / "datagen.yaml").relative_to(REPO)), "out": str((RUN / "eval").relative_to(REPO)),
                 "policy": {"plugin": "oracle"}, "sets": {"train": {"from": "train", "count": N}},
                 "rollout": {"plugin": "sim", "record": "no-images"}, "rollouts": 2, "retries": 0,
                 "run": {"workers": 1, "headless": True}}
        (RUN / "eval.yaml").write_text(yaml.safe_dump(espec))
        t0 = time.time()
        rc = cli.main([str(RUN / "eval.yaml"), "all", "--headless"], root=REPO)
        wall = round(time.time() - t0, 1)
        results = json.loads((RUN / "eval" / "results.json").read_text())
        block = results["sets"]["train"]
        rows = {k: {"passed": e["passed"], "termination": e["tries"][0]["termination"],
                    "episode": e["tries"][0]["record"].get("episode"), "resets": e["tries"][0]["record"].get("resets"),
                    "failed_phase": e["tries"][0]["record"].get("failed_phase"),
                    "duration_s": e["tries"][0]["record"].get("duration_s"), "error": e["tries"][0]["record"].get("error")}
                for k, e in block["rollouts"].items()}
        (RUN / "numbers.json").write_text(json.dumps({"wall_s": wall, "rc": rc, "metrics": block["metrics"], "rollouts": rows}, indent=1))
        print(json.dumps({"wall_s": wall, "metrics": block["metrics"], "rollouts": rows}))
        assert len(rows) == 2 * N
        for seed in {k.split("_")[0] for k in rows}:                           # reset between a config's rollouts
            r0, r1 = rows[f"{seed}_r0"], rows[f"{seed}_r1"]
            assert (r0["resets"], r1["resets"]) == (0, 1) and r1["episode"] == r0["episode"] + 1, (r0, r1)
        assert all(r["termination"] for r in rows.values())
        assert rc == 0 and all(r["passed"] for r in rows.values()), rows
        assert (RUN / "eval" / "report.md").is_file()
    finally:
        keep = RUN / "numbers.json"
        saved = keep.read_text() if keep.exists() else None
        shutil.rmtree(RUN, ignore_errors=True)
        if saved:
            RUN.mkdir(parents=True)
            keep.write_text(saved)
