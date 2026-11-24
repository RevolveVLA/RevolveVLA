"""Warm reset on the real sim (GPU + docker; skipped unless EVAL_GPU=1; needs the simulator image and the assets). Fridge /
mobile_panda. (1) one sim, a warm reset to three starts around the scene's (base up to 0.2 m / 0.4 rad, arm folded / ready):
each within the tolerance, and each faster than a relaunch (the sim's launch to ready). (2) `revolve evaluate` with the expert
on EVAL_WARM_CONFIGS (4) held-out configs that differ only in `robot_start`, once with `run.reset: warm` and once with
`relaunch`: warm shares one sim, every warm reset is within the tolerance, no rollout errors, and no warm-reset rollout fails
in `approach`. The expert passes about two rollouts in three, so verdicts are printed, not compared one by one. The numbers are
printed (pytest -s). Outputs in runs/warm_reset_gpu_<ts>/ (deleted afterwards)."""
import json
import os
import shutil
import time
from pathlib import Path

import pytest
import yaml

from evaluation import cli
from revolve_common import specs
from revolve_common.sim_session import Docker, DomainPool, SimSession, run_with_relaunch
from revolve_common.warm_reset import ResetTarget, Tolerance

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.environ.get("EVAL_GPU") != "1", reason="needs a GPU: EVAL_GPU=1")]

REPO = Path(__file__).resolve().parents[3]
CLIENT = "/repo/components/evaluation/tests/warm_reset_client.py"
CONFIGS = int(os.environ.get("EVAL_WARM_CONFIGS", "4"))
TRAIN = {"task": "open_fridge_door", "robot": "mobile_panda", "seed": 0,
         "generator": {"plugin": "moveit", "attempts": 1, "episodes_per_scene": 1, "record": "no-images"},
         "randomisers": {"robot_start": {"x_m": [-1.85, -1.65], "y_m": [1.0, 1.2], "yaw_rad": [1.47, 1.67],
                                         "arm": {"choices": ["folded"]}}},
         "validators": [{"plugin": "collision_free"}],
         "splits": {"train": {"target": 1, "pool": {"initial": [0, 2], "replacement": [3, 7]}}}}


def leftovers():
    return [n for n in Docker().names() if n.startswith(("dg-", "ev-", "warm-"))]


@pytest.fixture
def run_dir():
    assert not leftovers(), f"containers of another run: {leftovers()}"
    d = REPO / "runs" / f"warm_reset_gpu_{time.strftime('%Y%m%d_%H%M%S')}"
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
    assert not leftovers(), f"containers left: {leftovers()}"


def target(base, arm):
    robot = specs.read_yaml(REPO / "specs/robots/mobile_panda/robot.yaml")
    joints = tuple(f"panda_joint{i}" for i in range(1, 8))
    return ResetTarget("mobile_panda", base, tuple(robot["poses"][arm]), joints, "panda_arm_controller")


def test_a_warm_reset_reaches_three_starts_within_the_tolerance_and_beats_a_relaunch(run_dir):
    x, y, yaw = specs.load(REPO / "specs/scenes/kitchen_partnet/scene.yaml").robot_start.base
    targets = [target((x + 0.2, y, yaw + 0.1), "folded"), target((x - 0.1, y + 0.15, yaw - 0.1), "ready"),
               target((x, y - 0.15, yaw + 0.4), "folded")]
    out, log = run_dir / "sim", run_dir / "client.log"
    name, domain = DomainPool("warm-probe").lease()
    session = SimSession("kitchen_partnet", "mobile_panda", "open_fridge_door", out, name, domain, "no-images",
                         log=run_dir / "sim.log")

    def work(s):
        t0 = time.time()
        s.client(["python3", CLIENT, str(out), *[t.to_json() for t in targets]]).run(log, timeout=420, watch=s.check)
        line = [ln for ln in log.read_text().splitlines() if ln.startswith("RESULT ")][-1]
        return t0, json.loads(line[len("RESULT "):])
    t0, res = run_with_relaunch(session, work)
    tol, launch_s = Tolerance(), res["ready_ts"] - t0
    for t, r in zip(targets, res["reports"]):
        assert r["ok"], (t, r)
        assert r["base_err_m"] <= tol.base_m and r["yaw_err_rad"] <= tol.yaw_rad and r["arm_err_rad"] <= tol.arm_rad
        assert r["wall_s"] < launch_s
    print(f"\nWARM-RESET launch to ready {launch_s:.1f} s; reports {json.dumps(res['reports'])}")


def evaluate(run_dir, mode):
    rel = lambda p: str(p.relative_to(REPO))
    (run_dir / "train.yaml").write_text(yaml.safe_dump({**TRAIN, "out": rel(run_dir / "train")}))
    eval_yaml = run_dir / f"eval_{mode}.yaml"
    eval_yaml.write_text(yaml.safe_dump({
        "train": rel(run_dir / "train.yaml"), "out": rel(run_dir / f"eval_{mode}"), "policy": {"plugin": "expert"},
        "sets": {"heldout": {"count": CONFIGS, "sampler": {"strategy": "sobol"}, "pool": {"initial": [500, 509]}}},
        "rollout": {"plugin": "sim", "record": "no-images"}, "rollouts": 1, "retries": 0,
        "run": {"workers": 1, "headless": True, "reset": mode}}))
    t0 = time.monotonic()
    assert cli.main([str(eval_yaml), "all", "--headless"], root=REPO) == 0
    wall = time.monotonic() - t0
    rows = json.loads((run_dir / f"eval_{mode}" / "results.json").read_text())["sets"]["heldout"]["rollouts"]
    return wall, {k: e["tries"][0]["record"] for k, e in rows.items()}


def test_the_expert_runs_after_a_warm_reset_as_after_a_relaunch(run_dir):
    wall_warm, warm = evaluate(run_dir, "warm")
    wall_relaunch, relaunch = evaluate(run_dir, "relaunch")
    summary = lambda rs: {k: (r["passed"], r["failed_phase"], r["reason"]) for k, r in rs.items()}
    starts = {k: yaml.safe_load(Path(r["scene"]).read_text())["robot_start"]["base"] for k, r in warm.items()}
    resets = {k: r["reset"] for k, r in warm.items()}
    print(f"\nWARM-RESET starts {starts}\nWARM-RESET verdicts warm {summary(warm)}\nWARM-RESET verdicts relaunch {summary(relaunch)}"
          f"\nWARM-RESET resets {json.dumps(resets)}\nWARM-RESET wall warm {wall_warm:.0f} s, relaunch {wall_relaunch:.0f} s")
    assert len(list((run_dir / "eval_warm" / "heldout").glob("*_sim"))) == 1                    # one sim for all configs
    assert len(warm) == len(relaunch) == CONFIGS and all(r["error"] is None for r in (*warm.values(), *relaunch.values()))
    assert all(r["mode"] == "warm" and "fallback" not in r for r in resets.values()), resets
    reports = [r["report"] for r in resets.values() if r["report"]]
    assert len(reports) == CONFIGS - 1 and all(r["ok"] for r in reports)                       # the first start is the launch's
    # the expert is not deterministic (it passes about two runs in three), so the verdicts are compared as a rate, and the
    # approach phase, the one a wrong start breaks first, must not be where the warm-reset rollouts fail
    assert all(r["failed_phase"] != "approach" for k, r in warm.items() if resets[k]["report"]), summary(warm)
    assert any(r["passed"] for k, r in warm.items() if resets[k]["report"]), summary(warm)
