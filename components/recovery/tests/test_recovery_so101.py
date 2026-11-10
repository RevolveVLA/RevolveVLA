"""Recovery of a second robot with real jobs: the mobile SO-101 on the drawer. Fixtures (git-ignored, made on the
first run into runs/test_cache/recovery/mobile_so101/, reused after it; delete that dir to regenerate): a datagen
episode (components/datagen/generate.py, seed SEED) and a perturbed open start (regen_sim.py perturb: the pull
stopped at PERTURB_CM, the bar held where it was). recover.py then recovers the episode from its open and grasp
starts and the perturbed open start (1 episode each). This session's conftest names the Panda, so every SO-101 job
gets REVOLVE_ROBOT=mobile_so101: generate.py and recover.py run as host subprocesses with it, the perturb job with it
as a container env override. The recovery plans use the robot's regen_base_corner_spheres (robot.yaml planner).
Outputs: runs/recovery_tests/<stamp>/mobile_so101/ (numbers.json there)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from conftest import CACHE, COMP, REPO, RUN, load
from test_recovery import check_suffix

import recover as R
import specs_bridge

ROBOT = "mobile_so101"
TASK = "open_drawer"
SEED = 8
PERTURB_CM = 8.0
SO_CACHE = CACHE / ROBOT
OUT = RUN / ROBOT


def run_host(args, log):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(REVOLVE_ROBOT=ROBOT, PYTHONPATH=f"{REPO / 'common'}:{COMP / 'src' / 'recovery'}:"
                                               f"{COMP.parent / 'datagen' / 'src' / 'datagen'}")
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        return subprocess.run([sys.executable, *map(str, args)], cwd=REPO, env=env, stdout=f,
                              stderr=subprocess.STDOUT, timeout=3600).returncode


def demo():
    out = SO_CACHE / "datagen"
    m = load(out / "manifest.json") if (out / "manifest.json").is_file() else {}
    if not m.get("episodes"):
        run_host([COMP.parent / "datagen" / "generate.py", "--task", TASK, "--seeds", SEED, "--out", out,
                  "--episodes-per-seed", 1, "--workers", 1], SO_CACHE / "generate.log")
        m = load(out / "manifest.json")
    assert m["episodes"], f"{ROBOT} datagen fixture failed: {m['seeds']} (log {SO_CACHE / 'generate.log'})"
    return out / "episodes" / m["episodes"][0]


def perturbed(src):
    out = SO_CACHE / "perturbed_open"
    ro = out / "rollout"
    was = load(ro / "episode.json")["policy_rollout"]["perturbed_from"] if (ro / "episode.json").is_file() else None
    if not ((ro / "phase_states" / "open_start.npz").is_file() and was == R._datagen_launch().cpath(src)):
        res = R.finish(R.start_sim(["perturb", "--source", str(src), "--out", str(out), "--joint", str(PERTURB_CM)],
                                   SO_CACHE / "jobs" / "perturb", timeout=600, env={"REVOLVE_ROBOT": ROBOT}))
        assert res["exit"] == 0, res
    return ro


@pytest.fixture(scope="module")
def so101():
    src = demo()
    ro = perturbed(src)
    cases = {"open": (src, "open"), "grasp": (src, "grasp"), "perturbed_open": (ro, None)}
    rc = run_host([COMP / "recover.py", "--task", TASK, "--rollouts", *(f"{d}:{p}" if p else str(d) for d, p in
                                                                        cases.values()),
                   "--out", OUT, "--episodes-per-failure", 1, "--workers", len(cases)], OUT / "recover.log")
    return dict(src=src, perturbed=ro, cases=cases, rc=rc, manifest=load(OUT / "manifest.json"), numbers={})


def accepted(so, case):
    d, p = so["cases"][case]
    r = so["manifest"]["rollouts"][R.key(d, p)]
    assert r["state"] == "done", r
    return r, next(x for x in r["episodes"] if x["accepted"])


@pytest.mark.parametrize("case", ["open", "grasp", "perturbed_open"])
def test_so101_recovered(so101, case):
    """One accepted suffix episode per start (check_suffix), from a recovery plan with the robot's corner spheres."""
    assert so101["rc"] == 0
    r, e = accepted(so101, case)
    n = so101["numbers"][case] = check_suffix(OUT, r, e, so101["cases"][case][1] or "open")
    plan = OUT / r["plans"][e["plan"]]["dir"]
    rg, rep = load(plan / "handoff.json")["regenerate"], load(plan / "plan_report.json")
    assert rg["base_corner_spheres"] == specs_bridge.load("robot", ROBOT)["regen_base_corner_spheres"], rg
    # a reach always plans with them; a pull drops them when the saved start collides under them (plan_from_state)
    assert rep["corner_spheres"] > 0 or (rep["mode"] == "pull" and not rep["corner_spheres_in_pull"]["used"]), rep
    n.update(corner_spheres=rep["corner_spheres"], corners_in_pull=rep["corner_spheres_in_pull"]["used"])
    (OUT / "numbers.json").write_text(json.dumps(so101["numbers"], indent=1))
    assert n["frames"] > 100


def test_so101_perturbed_start(so101):
    """The perturbed open start regenerates the pull from the grasp it holds, the drawer starting partly open."""
    ro = so101["perturbed"]
    q0 = load(ro / "episode.json")["policy_rollout"]["joint_si"]
    assert abs(q0 * 100 - PERTURB_CM) < 1.5
    r, e = accepted(so101, "perturbed_open")
    assert r["grasp_from_state"] and r["source"]["policy"] and r["source"]["failed_phase"] == "open"
    door = np.load(OUT / e["dir"] / "state_log.npz")["door_pos"]
    assert abs(door[1] - q0) < 0.01 and door.max() * 100 >= 15.0
    so101["numbers"]["perturbed"] = dict(start_cm=round(q0 * 100, 2), drawer_max_cm=round(door.max() * 100, 2))
    (OUT / "numbers.json").write_text(json.dumps(so101["numbers"], indent=1))
