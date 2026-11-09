"""Recovery tests. test_recover_unit.py: the composition with stubbed jobs (no docker). test_recovery.py: real
jobs. Fixtures are NOT committed binaries: a datagen episode per task (seed 0, components/datagen/generate.py)
and a perturbed fridge open start (regen_sim.py perturb: the pull stopped at 25 deg, the handle held where
it was) are generated on the first run into runs/test_cache/recovery/ (git-ignored) and reused after it
(delete that dir to regenerate them, e.g. after a sim or planner change). Then one recover() per task, all
at once: fridge from its approach, grasp and open starts + the perturbed open start, drawer from its open
start (1 episode each); the fridge open item's plan job is polled over the wrapper's HTTP. Outputs:
runs/recovery_tests/<stamp>/ (git-ignored); numbers.json there sums up the key values.

Needs: `./revolve build recovery` (+ datagen, simulator), the generated kitchen assets in assets
(licence-bound, not in git), a free GPU (~3 GB per planner; REVOLVE_GPU_PLANNERS, default 2).
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

COMP = Path(__file__).resolve().parents[1]
REPO = COMP.parents[1]
sys.path[:0] = [str(COMP), str(COMP / "src" / "recovery"), str(COMP.parent / "datagen" / "src" / "datagen")]
CACHE = REPO / "runs" / "test_cache" / "recovery"
RUN = REPO / "runs" / "recovery_tests" / time.strftime("%Y%m%d_%H%M%S")
SEED = 0
PERTURB_DEG = 25.0
# test id -> (task, phase forced on the fixture or None for the perturbed rollout's own verdict)
CASES = {"fridge_approach": ("open_fridge_door", "approach"), "fridge_grasp": ("open_fridge_door", "grasp"),
         "fridge_open": ("open_fridge_door", "open"), "fridge_perturbed_open": ("open_fridge_door", None),
         "drawer_open": ("open_drawer", "open")}
POLLED = "fridge_open"


def load(p):
    return json.loads(Path(p).read_text())


def preflight():
    import recover as R
    dl = R._datagen_launch()
    missing = [img for img in (R.IMAGE, dl.IMAGE, dl.sim_launch.IMAGE)
               if subprocess.run(["docker", "image", "inspect", img], capture_output=True).returncode]
    missing += [str(p) for p in (dl.INSTALL / "BUILD_OK", dl.sim_launch.INSTALL / "setup.bash",
                                 REPO / "assets/generated_partnet/fridge/model.sdf",
                                 REPO / "assets/generated_partnet/kitchen_counter/model.sdf")
                if not p.is_file()]
    if missing:
        pytest.fail(f"not built / missing: {missing} (./revolve build recovery; ./revolve build simulator)", pytrace=False)


def poll_http(job, seen):
    """GET /status (and /result) on the job container's bridge IP until /result answers."""
    import httpx
    for _ in range(1200):
        try:
            base = f"http://{job.ip()}:8000"
        except Exception:
            if job.container.status == "exited":
                return
            time.sleep(0.5)
            continue
        try:
            s = httpx.get(base + "/status", timeout=2).json()
            seen["status"].append(dict(state=s["state"], progress=s["progress"]))
            r = httpx.get(base + "/result", timeout=2)
            if r.status_code == 200:
                seen["result"] = r.json()
                return
            seen["result_code_while_running"] = r.status_code
        except Exception:
            if seen["status"] and job.container.status != "running":
                return
        time.sleep(1.0)


def demo(task):
    """The cached datagen episode of `task` (seed 0), generated on the first run."""
    import generate as G
    out = CACHE / "datagen" / task
    m = G.read(out / "manifest.json")
    if not (m and m.get("episodes")):
        m = G.generate(task, [SEED], out, episodes_per_seed=1, workers=1)
    assert m["episodes"], f"datagen fixture for {task} failed: {m['seeds']}"
    return out / "episodes" / m["episodes"][0]


def perturbed(fridge_demo):
    import recover as R
    out = CACHE / "perturbed_fridge_open"
    ro = out / "rollout"
    src = load(ro / "episode.json")["policy_rollout"]["perturbed_from"] if (ro / "episode.json").is_file() else None
    if not ((ro / "phase_states" / "open_start.npz").is_file() and src == R._datagen_launch().cpath(fridge_demo)):
        res = R.finish(R.start_sim(["perturb", "--source", str(fridge_demo), "--out", str(out), "--joint", str(PERTURB_DEG)],
                                   CACHE / "jobs" / "perturb", timeout=600))
        assert res["exit"] == 0, res
    return ro


@pytest.fixture(scope="session")
def fixtures():
    preflight()
    t0 = time.time()
    demos = {}
    ths = [threading.Thread(target=lambda t=t: demos.__setitem__(t, demo(t))) for t in ("open_fridge_door", "open_drawer")]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert set(demos) == {"open_fridge_door", "open_drawer"}, demos
    return dict(demo=demos, perturbed=perturbed(demos["open_fridge_door"]), wall_s=round(time.time() - t0, 1))


@pytest.fixture(scope="session")
def recovered(fixtures):
    """recover() per task, both at once; returns {case: (out dir, rollout key)}, manifests, HTTP polls."""
    import recover as R
    seen = {"status": [], "result": None}
    polled_key = R.key(fixtures["demo"]["open_fridge_door"], CASES[POLLED][1])

    def plan(plan_dir, seed, attempts, job_dir):
        if plan_dir.parent.parent.name != polled_key or seen["status"]:
            return R.docker_plan(plan_dir, seed, attempts, job_dir)
        with R.PLANNERS:
            job = R.start_plan(["plan", "--dir", str(plan_dir), "--seed", str(seed), "--attempts", str(attempts)], job_dir,
                               timeout=R.PLAN_TIMEOUT_S, linger_s=8.0)
            th = threading.Thread(target=poll_http, args=(job, seen), daemon=True)
            th.start()
            res = R.finish(job)
            th.join(timeout=30)
        return res["exit"]

    items = {t: [] for t in ("open_fridge_door", "open_drawer")}
    where = {}
    for case, (task, phase) in CASES.items():
        d = fixtures["perturbed"] if phase is None else fixtures["demo"][task]
        items[task].append((d, phase))
        where[case] = (RUN / task, R.key(d, phase))
    man, wall = {}, {}

    def one(task):
        t0 = time.time()
        man[task] = R.recover(task, items[task], RUN / task, episodes_per_failure=1, workers=len(items[task]),
                              jobs=R.Jobs(plan, R.docker_candidate))
        wall[task] = round(time.time() - t0, 1)
    RUN.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ths = [threading.Thread(target=one, args=(t,)) for t in items]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return dict(where=where, manifest=man, wall=wall, total_s=round(time.time() - t0, 1), http=seen)


@pytest.fixture(scope="session")
def numbers():
    d = {}
    yield d
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "numbers.json").write_text(json.dumps(d, indent=1, default=str))
