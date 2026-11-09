"""Simulator tests: every sim job runs in revolve-simulator through launch.py (the wrapper, as the manager
will), all started at once at session start (they are independent; the restore check needs the replay).
Outputs: runs/sim_tests/<stamp>/ (git-ignored), numbers.json there sums up the key values.

Needs: `./revolve build simulator` (image + ThirdParty/gz_kura/install), the generated kitchen assets in
assets (licence-bound, not in git), and a stored AutoMoMa plan (REVOLVE_SIM_PLAN: an
episode dir with plan/<run>/handoff.* and replay/{episode,verdict}.json + state_log.npz; default:
runs/test_cache/simulator/fridge_seed0, git-ignored, a copy of `./revolve test recovery`'s fridge seed 0 datagen plan
(runs/test_cache/recovery/datagen/open_fridge_door/plans/seed_0 -> plan/seed_0) and this suite's own replay of it
(runs/sim_tests/<stamp>/replay -> replay/; datagen's recorded episode is not bit-identical to it); missing -> the
stored-plan tests skip).
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

COMP = Path(__file__).resolve().parents[1]
REPO = COMP.parents[1]
sys.path[:0] = [str(COMP), str(COMP / "src" / "sim")]
import launch  # noqa: E402

EPISODE = Path(os.environ.get("REVOLVE_SIM_PLAN", REPO / "runs/test_cache/simulator/fridge_seed0"))
RUN = REPO / "runs" / "sim_tests" / time.strftime("%Y%m%d_%H%M%S")
ASSETS = [REPO / "assets/generated_partnet/fridge/model.sdf",
          REPO / "assets/generated_partnet/kitchen_counter/model.sdf",
          REPO / "assets/generated/cup/model.sdf"]


def plan_dir():
    runs = sorted((EPISODE / "plan").glob("*/handoff.npz")) if EPISODE.is_dir() else []
    return runs[0].parent if runs else None


def jobs_spec():
    out = lambda n: str(RUN / n)
    j = {"scene_check": ["scene-check", "--out", out("scene_check")],
         "robot_smoke": ["robot-smoke", "--out", out("robot_smoke")],
         "can_demo": ["can-demo", "--out", out("can_demo")],
         "episode_fridge": ["episode", "--name", "fridge", "--out", out("episode_fridge")],
         "episode_drawer": ["episode", "--name", "drawer", "--out", out("episode_drawer")]}
    if plan_dir() is not None:
        attempt = json.loads((EPISODE / "replay" / "episode.json").read_text())["attempt_id"]
        j["replay"] = ["replay", "--plan", str(plan_dir()), "--attempt", str(attempt), "--out", out("replay")]
    return j


class Jobs:
    def __init__(self):
        self.started, self.done, self.http = {}, {}, {"status": [], "result": None}

    def start(self, name, args, **kw):
        self.started[name] = launch.start(args, RUN / f"{name}_job", timeout=900, **kw)

    def result(self, name):
        if name not in self.done:
            self.done[name] = launch.finish(self.started[name])
        return self.done[name]

    def poll_http(self, name):
        """The wrapper's GET /status and /result on the container's bridge IP while the job runs."""
        import httpx
        job = self.started[name]
        for _ in range(600):
            try:
                base = f"http://{job.ip()}:8000"     # reloads the container
            except Exception:
                if job.container.status == "exited":
                    return
                time.sleep(0.5)
                continue
            try:
                s = httpx.get(base + "/status", timeout=2).json()
                self.http["status"].append(dict(state=s["state"], progress=s["progress"]))
                r = httpx.get(base + "/result", timeout=2)
                if r.status_code == 200:
                    self.http["result"] = r.json()
                    return
                self.http["result_code_while_running"] = r.status_code
            except Exception:
                if self.http["status"] and job.container.status != "running":
                    return
            time.sleep(1.0)


def preflight():
    img = subprocess.run(["docker", "image", "inspect", launch.IMAGE], capture_output=True)
    missing = ([] if img.returncode == 0 else [launch.IMAGE]) + \
        [str(p) for p in [launch.INSTALL / "setup.bash", *ASSETS] if not p.is_file()]
    if missing:
        pytest.fail(f"not built / missing: {missing} (./revolve build simulator; assets: components/simulator/README.md)",
                    pytrace=False)


@pytest.fixture(scope="session")
def jobs():
    preflight()
    RUN.mkdir(parents=True, exist_ok=True)
    js = Jobs()
    t0 = time.time()
    for name, args in jobs_spec().items():
        js.start(name, args, linger_s=8.0 if name == "replay" else 0.0)
    poller = threading.Thread(target=js.poll_http, args=("replay",), daemon=True) if "replay" in js.started else None
    if poller:
        poller.start()
    js.poller = poller
    yield js
    for name in list(js.started):
        if name not in js.done:
            js.result(name)
    print(f"\nsimulator jobs: {time.time() - t0:.0f} s wall, outputs in {RUN}")


@pytest.fixture(scope="session")
def numbers():
    d = {}
    yield d
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "numbers.json").write_text(json.dumps(d, indent=1, default=str))


def load(p):
    return json.loads(Path(p).read_text())
