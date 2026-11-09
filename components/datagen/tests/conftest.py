"""Datagen tests. test_generate_unit.py: the composition with stubbed jobs (no docker). test_datagen.py: real
jobs: `generate` for seed 0 of each task (fridge + drawer started at once, so two planners share the GPU),
the fridge's plan job polled over the wrapper's HTTP, then a resume of the fridge run under the wrapper on
the host. Outputs: runs/datagen_tests/<stamp>/ (git-ignored); numbers.json there sums up the key values.

Needs: `./revolve build datagen` and `./revolve build simulator`, the generated kitchen assets in
assets (licence-bound, not in git), a free GPU (~3 GB per planner).
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
sys.path[:0] = [str(COMP), str(COMP / "src" / "datagen")]
RUN = REPO / "runs" / "datagen_tests" / time.strftime("%Y%m%d_%H%M%S")
TASKS = {"open_fridge_door": 0, "open_drawer": 0}       # task -> seed
POLLED = "open_fridge_door"                             # its plan job is polled over HTTP


def load(p):
    return json.loads(Path(p).read_text())


class GpuSampler(threading.Thread):
    """nvidia-smi memory.used every 0.2 s while the jobs run."""

    def __init__(self):
        super().__init__(daemon=True)
        self.mib, self.stop = [], threading.Event()

    def run(self):
        p = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "200"],
                             stdout=subprocess.PIPE, text=True)
        for line in p.stdout:
            if line.strip().isdigit():
                self.mib.append(int(line))
            if self.stop.is_set():
                break
        p.terminate()
        p.wait()


def poll_http(job, seen):
    """GET /status (and /result) on the plan container's bridge IP until /result answers."""
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


def preflight():
    import launch
    missing = [img for img in (launch.IMAGE, launch.sim_launch.IMAGE)
               if subprocess.run(["docker", "image", "inspect", img], capture_output=True).returncode]
    missing += [str(p) for p in (launch.INSTALL / "BUILD_OK", launch.sim_launch.INSTALL / "setup.bash",
                                 REPO / "assets/generated_partnet/fridge/model.sdf",
                                 REPO / "assets/generated_partnet/kitchen_counter/model.sdf")
                if not p.is_file()]
    if missing:
        pytest.fail(f"not built / missing: {missing} (./revolve build datagen; ./revolve build simulator)", pytrace=False)


@pytest.fixture(scope="session")
def gen():
    """generate() for each task's seed, both at once; returns {task: manifest} + timings."""
    preflight()
    import generate as G
    import launch
    seen = {"status": [], "result": None}

    def polled_plan(task, seed, out, job_dir, attempts=None):
        job = launch.start(["plan", "--task", task, "--seed", str(seed), "--out", str(out)], job_dir,
                           timeout=G.PLAN_TIMEOUT_S, linger_s=8.0)
        th = threading.Thread(target=poll_http, args=(job, seen), daemon=True)
        th.start()
        res = launch.finish(job)
        th.join(timeout=30)
        return res["exit"]

    RUN.mkdir(parents=True, exist_ok=True)
    gpu = GpuSampler()
    gpu.start()
    out, wall = {}, {}

    def one(task):
        t0 = time.time()
        jobs = G.Jobs(polled_plan, G.docker_replay) if task == POLLED else G.DOCKER
        out[task] = G.generate(task, [TASKS[task]], RUN / task, episodes_per_seed=1, workers=1, jobs=jobs)
        wall[task] = round(time.time() - t0, 1)

    ths = [threading.Thread(target=one, args=(t,)) for t in TASKS]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    gpu.stop.set()
    return dict(manifest=out, wall=wall, total_s=round(time.time() - t0, 1), gpu_mib=gpu.mib, http=seen)


@pytest.fixture(scope="session")
def numbers():
    d = {}
    yield d
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "numbers.json").write_text(json.dumps(d, indent=1, default=str))
