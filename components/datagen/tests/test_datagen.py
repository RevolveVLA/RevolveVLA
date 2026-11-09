"""Datagen with real jobs (see conftest.py): plan (revolve-datagen) -> replay check -> record (revolve-simulator)
-> acceptance for seed 0 of the fridge and the drawer task; wrapper HTTP; resume."""
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import pytest
from conftest import COMP, POLLED, REPO, RUN, TASKS, load

import launch

PHASES = ("approach", "grasp", "open", "release")
# Legacy plan runs of seed 0 (the earlier pipeline's planner image and v2 round start sets,
# 50 attempts): planned / trajopt successes. The trajectories themselves are not repeatable
# run to run on the GPU (cuRobo; legacy too), so only these counts are compared, within TOL (drawer trajopt
# successes were 34 or 35 over five runs).
TOL = 2
LEGACY = {"open_fridge_door": dict(planned=44, trajopt_success=48), "open_drawer": dict(planned=25, trajopt_success=35)}


def test_build():
    """cuRobo built for the image's arch into the git-ignored install; the submodules' trees stay clean."""
    arch = subprocess.run(["docker", "image", "inspect", launch.IMAGE, "--format", "{{json .Config.Env}}"],
                          capture_output=True, text=True, check=True).stdout
    assert f"TORCH_CUDA_ARCH_LIST={(launch.INSTALL / 'BUILD_OK').read_text().strip()}" in arch
    assert any(launch.INSTALL.glob("curobo/curobolib/*_cu*.so"))
    for m in ("curobo", "automoma"):
        st = subprocess.run(["git", "-C", str(REPO / "ThirdParty" / m), "status", "--porcelain", "--untracked-files=no"],
                            capture_output=True, text=True, check=True).stdout
        assert st == "", f"ThirdParty/{m} changed: {st}"


@pytest.mark.parametrize("task", TASKS)
def test_plan(gen, numbers, task):
    """The plan job: planned attempts, FK check passed, same key numbers as the legacy plan run."""
    m = gen["manifest"][task]
    seed = TASKS[task]
    p = m["seeds"][str(seed)]["plan"]
    d = RUN / task / p["dir"]
    rep, fk, h = load(d / "plan_report.json"), load(d / "fk_check.json"), load(d / "handoff.json")
    assert p["ok"] and p["rc"] == 0 and fk["passed"] and fk["max_pos_err_m"] < 0.02
    assert h["plugins"]["task"] == task and h["seed"] == seed and len(p["planned"]) == rep["planned"]
    assert all(abs(rep[k] - v) <= TOL for k, v in LEGACY[task].items()), {k: rep[k] for k in LEGACY[task]}
    sha = hashlib.sha256(np.load(d / "handoff.npz")["traj_robot_world"].tobytes()).hexdigest()
    st = load(RUN / task / "jobs" / f"plan_seed_{seed}" / "status.json")
    assert st["state"] == "succeeded" and st["progress"]["planned"] == rep["planned"] and st["progress"]["fk_passed"]
    numbers[f"plan_{task}"] = dict(planned=f"{rep['planned']}/{rep['attempts']}", trajopt_success=rep["trajopt_success"],
                                   plan_s=round(rep["times"]["total_s"], 1), vram_peak_mib=rep["vram_peak_mib"],
                                   torch_max_reserved_mib=rep["torch_max_reserved_mib"], traj_sha=sha[:12])


def test_plan_wrapper_http(gen, numbers):
    """GET /status during the plan job shows the parsed progress; /result is 409 until the end."""
    http = gen["http"]
    assert http["result"] is not None and http["result"]["state"] == "succeeded" and http["result"]["returncode"] == 0
    assert http.get("result_code_while_running") == 409
    stages = [s["progress"].get("stage") for s in http["status"]]
    assert "plan" in stages or "ik_traj" in stages, stages
    assert abs(http["result"]["progress"]["planned"] - LEGACY[POLLED]["planned"]) <= TOL
    numbers["http_polls"] = len(http["status"])


@pytest.mark.parametrize("task", TASKS)
def test_episode(gen, numbers, task):
    """One accepted episode: check and record replays accepted, every task phase passes, the record reproduces
    the check's state log, the bag is written and its check (read back with rosbags in the sim) passes."""
    m = gen["manifest"][task]
    seed = TASKS[task]
    rec = m["seeds"][str(seed)]
    assert rec["state"] == "done" and len(m["episodes"]) == 1, rec
    e = next(x for x in rec["episodes"] if x["accepted"])
    d = RUN / task / e["dir"]
    chk = rec["checks"][str(e["attempt_id"])]
    assert chk["accepted"] and e["state_log_sha256"] == chk["state_log_sha256"]
    assert load(d / "verdict.json")["verdict"] == "accepted"
    pr = load(d / "phase_report.json")
    assert pr["accepted"] and [p for p in PHASES if pr["phases"][p]["status"] != "pass"] == []
    assert all((d / "phase_states" / f"{p}_start.npz").is_file() for p in PHASES)
    bc = load(d / "bag_check.json")
    assert bc["pass"] and bc["failed"] == []
    mcap = sorted((d / "bag").glob("*.mcap"))
    assert len(mcap) == 1 and mcap[0].read_bytes()[:8] == b"\x89MCAP0\r\n"
    cams = {t: v["count"] for t, v in bc["topics"].items() if t.startswith("/cameras/")}
    assert len(cams) == 3 and min(cams.values()) > 100, cams         # 30 Hz JPEG frames of the 3 cameras
    ep = load(d / "episode.json")
    assert ep["datagen"] == dict(seed=seed, attempt_id=e["attempt_id"], accepted=True, label="demo", check=chk["dir"])
    assert load(d / "plan.json")["attempt_id"] == e["attempt_id"] and (d / "plan.npz").is_file()
    assert not any((d / c).exists() for c in ("wrist_cam", "overhead_left_cam", "overhead_right_cam"))
    jobs = RUN / task / "jobs"
    numbers[f"episode_{task}"] = dict(
        name=e["name"], checks=len(rec["checks"]), camera_frames=cams, bag_mb=round(sum(p.stat().st_size for p in (d / "bag").iterdir()) / 1e6, 1),
        check_wall_s=load(jobs / f"check_seed_{seed}_a{e['attempt_id']}" / "result.json")["finished_at"]
        - load(jobs / f"check_seed_{seed}_a{e['attempt_id']}" / "result.json")["started_at"],
        record_wall_s=load(jobs / f"record_{e['name']}" / "result.json")["finished_at"]
        - load(jobs / f"record_{e['name']}" / "result.json")["started_at"],
        generate_wall_s=gen["wall"][task])
    numbers["gpu_peak_mib"] = max(gen["gpu_mib"], default=None)
    numbers["generate_both_wall_s"] = gen["total_s"]


def test_gpu_fits(gen):
    """Two planners at once (+ the record's EGL rendering) stay inside the laptop's 8 GB."""
    assert gen["gpu_mib"] and max(gen["gpu_mib"]) < 8000


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_resume_under_wrapper(gen, numbers):
    """generate re-run on the finished fridge dir under revolve_common.wrapper (host): nothing is redone (no job
    dir is touched), /status shows the parsed counts, the manifest still lists the episode."""
    import httpx
    task = POLLED
    out = RUN / task
    before = {p: p.stat().st_mtime_ns for p in (out / "jobs").glob("*/result.json")}
    ep_before = (out / "episodes" / gen["manifest"][task]["episodes"][0]).stat().st_mtime_ns
    jd, port = RUN / "resume_job", free_port()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = f"{REPO / 'common'}:{COMP / 'src' / 'datagen'}"
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-m", "revolve_common.wrapper", "--job-dir", str(jd), "--host", "127.0.0.1",
                          "--port", str(port), "--parser", "progress:parse", "--linger-s", "5", "--",
                          sys.executable, str(COMP / "generate.py"), "--task", task, "--seeds", str(TASKS[task]),
                          "--out", str(out), "--workers", "1"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    res = None
    for _ in range(120):
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/result", timeout=2)
            if r.status_code == 200:
                res = r.json()
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    st = httpx.get(f"http://127.0.0.1:{port}/status", timeout=2).json() if res else None
    assert p.wait(timeout=60) == 0
    wall = time.time() - t0
    assert res and res["state"] == "succeeded" and res["progress"]["episodes_accepted"] == 1, res
    assert st["progress"]["seeds_done"] == 1
    assert {q: q.stat().st_mtime_ns for q in (out / "jobs").glob("*/result.json")} == before
    assert (out / "episodes" / gen["manifest"][task]["episodes"][0]).stat().st_mtime_ns == ep_before
    assert load(out / "manifest.json")["episodes"] == gen["manifest"][task]["episodes"]
    numbers["resume_wall_s"] = round(wall, 1)


def test_plan_cancel(gen, tmp_path):
    """POST /cancel during a plan job ends it promptly (SIGTERM reaches plan.py in its own process group)."""
    import httpx
    from conftest import poll_http  # noqa: F401  (same HTTP route)
    jd = RUN / "cancel_job"
    job = launch.start(["plan", "--task", "open_drawer", "--seed", "7", "--out", str(RUN / "cancel_plan")], jd,
                       timeout=300, linger_s=3.0)
    base, t_cancel = None, None
    for _ in range(240):
        try:
            base = f"http://{job.ip()}:8000"
            s = httpx.get(base + "/status", timeout=2).json()
            if s["progress"].get("stage") == "plan":
                time.sleep(5)                     # plan.py (and its nvidia-smi sampler) are running
                assert httpx.post(base + "/cancel", timeout=2).json() == {"cancelled": True}
                t_cancel = time.time()
                break
        except (httpx.HTTPError, StopIteration, KeyError):
            pass
        time.sleep(0.5)
    res = launch.finish(job)
    assert t_cancel is not None, "the plan job never reached the plan stage"
    assert time.time() - t_cancel < 30 and res["result"]["state"] == "cancelled", res
