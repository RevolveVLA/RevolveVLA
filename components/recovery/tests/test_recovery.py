"""Recovery with real jobs (see conftest.py): plan_from_state (revolve-recovery) -> candidate check -> record
(revolve-simulator, restored from the saved phase state) -> acceptance, for the fridge's approach, grasp and
open starts, a perturbed (policy-like) fridge open start and the drawer's open start; wrapper HTTP; resume;
cancel."""
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
import pytest
from conftest import CASES, COMP, PERTURB_DEG, POLLED, REPO, RUN, load

import recover as R

PHASES = ("approach", "grasp", "open", "release")


def test_build():
    """revolve-recovery is the datagen image plus a label: every datagen layer, nothing added on top."""
    def inspect(img, fmt):
        return json.loads(subprocess.run(["docker", "image", "inspect", img, "--format", fmt], capture_output=True,
                                         text=True, check=True).stdout)
    dl = R._datagen_launch()
    rec, dg = inspect(R.IMAGE, "{{json .RootFS.Layers}}"), inspect(dl.IMAGE, "{{json .RootFS.Layers}}")
    assert rec == dg
    assert inspect(R.IMAGE, "{{json .Config.Labels}}").get("org.revolve.component") == "recovery"


def accepted(recovered, case):
    out, key = recovered["where"][case]
    r = recovered["manifest"][CASES[case][0]]["rollouts"][key]
    assert r["state"] == "done", r
    e = next(x for x in r["episodes"] if x["accepted"])
    return out, r, e


def check_suffix(out, r, e, phase):
    """The accepted suffix episode e of manifest rollout r (recovered from `phase`, into out): check and record
    accepted, every task.yaml phase from the restarted one on passes, the record reproduces the check's state log,
    the restore and bag checks pass, the label says recovery + phase + suffix + its rollout. Returns its numbers."""
    restore = R.RESTORE_PHASE[phase]
    assert r["phase"] == phase and r["mode"] == R.REGENERABLE[phase] and r["restore_phase"] == restore
    d = out / e["dir"]
    chk = next(c for c in r["checks"].values() if c["dir"] == e["check"])
    assert chk["accepted"] and e["state_log_sha256"] == chk["state_log_sha256"]
    assert load(d / "verdict.json")["verdict"] == "accepted"
    pr = load(d / "phase_report.json")
    want = list(PHASES[PHASES.index(restore):])
    assert pr["accepted"] and list(pr["phases"]) == want and all(pr["phases"][p]["status"] == "pass" for p in want), pr["result"]
    assert pr["restore_check"]["pass_"] and pr["offline_this_run"]["verdict"] == "accepted"
    assert all((d / "phase_states" / f"{p}_start.npz").is_file() for p in want)
    bc = load(d / "bag_check.json")
    assert bc["pass"] and bc["failed"] == []
    cams = {t: v["count"] for t, v in bc["topics"].items() if t.startswith("/cameras/")}
    assert len(cams) == 3 and min(cams.values()) > 100, cams
    lab = load(d / "episode.json")["recovery"]
    assert (lab["source"], lab["phase"], lab["suffix"], lab["accepted"]) == ("recovery", phase, True, True)
    assert lab["from_rollout"] == r["source"]["dir"] and lab["action_index_offset"] == r["action_index"]
    plan = load(out / r["plans"][e["plan"]]["dir"] / "plan_report.json")
    jobs = out / e["dir"].split("/")[0] / "jobs"
    return dict(
        plans=len(r["plans"]), candidates=len(r["checks"]), accepted=f"{e['plan']} a{e['attempt_id']}",
        planned=f"{plan['planned']}/{plan['attempts']}", plan_s=round(plan["times"]["total_s"], 1),
        restore_mm=round(pr["restore_check"]["vs_source_next_step"]["link_pos_m"][0] * 1000, 4),
        frames=min(cams.values()),
        record_wall_s=round((lambda j: j["finished_at"] - j["started_at"])(
            load(jobs / f"record_ep_0_{e['plan']}_a{e['attempt_id']}" / "result.json")), 1))


@pytest.mark.parametrize("case", CASES)
def test_recovered(recovered, numbers, case):
    """One accepted suffix episode per case (check_suffix)."""
    task, forced = CASES[case]
    out, r, e = accepted(recovered, case)
    numbers[case] = dict(check_suffix(out, r, e, forced or "open"), recover_wall_s=recovered["wall"][task])
    numbers["recover_all_wall_s"] = recovered["total_s"]


def test_perturbed_start(fixtures, recovered, numbers):
    """The policy-like open start (pull stopped at 25 deg, handle held where it was) regenerates from the grasp
    it holds: grasp_from_state, the pull planned on the measured grasp from the partly open door, door >= 60."""
    ro = fixtures["perturbed"]
    q0 = load(ro / "episode.json")["policy_rollout"]["joint_si"]
    assert abs(math.degrees(q0) - PERTURB_DEG) < 3
    out, r, e = accepted(recovered, "fridge_perturbed_open")
    assert r["grasp_from_state"] and r["source"]["policy"] and r["source"]["failed_phase"] == "open"
    h = load(out / r["plans"][e["plan"]]["dir"] / "handoff.json")
    measured = max(int(g) for g in h["grasps"])            # the measured grasp is appended last
    row = next(x for x in h["attempts_detail"] if x["attempt_id"] == e["attempt_id"])
    assert row["grasp_id"] == measured and abs(h["regenerate"]["start_task_joint"] - q0) < 1e-9
    d = out / e["dir"]
    door = np.load(d / "state_log.npz")["door_pos"]
    assert abs(door[1] - q0) < 0.01 and math.degrees(door.max()) >= 60
    numbers["perturbed"] = dict(start_deg=round(math.degrees(q0), 2), door_max_deg=round(math.degrees(door.max()), 2),
                                grasp_id=measured)


def test_plan_wrapper_http(recovered, numbers):
    """GET /status during the recovery plan job shows the parsed progress; /result is 409 until the end."""
    http = recovered["http"]
    assert http["result"] is not None and http["result"]["state"] == "succeeded" and http["result"]["returncode"] == 0
    assert http.get("result_code_while_running") == 409
    stages = {s["progress"].get("stage") for s in http["status"]}
    assert stages & {"models", "plan"}, stages
    _, r, _ = accepted(recovered, POLLED)
    assert http["result"]["progress"]["planned"] == len(r["plans"]["e0_r0_s100"]["planned"])
    numbers["http_polls"] = len(http["status"])


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_resume_under_wrapper(fixtures, recovered, numbers):
    """recover.py re-run on the finished drawer dir under revolve_common.wrapper (host): nothing is redone (no
    job dir is touched), /status shows the parsed counts, the manifest still lists the episode."""
    import httpx
    out, key = recovered["where"]["drawer_open"]
    before = {p: p.stat().st_mtime_ns for p in out.glob("*/jobs/*/result.json")}
    jd, port = RUN / "resume_job", free_port()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = f"{REPO / 'common'}:{COMP / 'src' / 'recovery'}:{COMP.parent / 'datagen' / 'src' / 'datagen'}"
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-m", "revolve_common.wrapper", "--job-dir", str(jd), "--host", "127.0.0.1",
                          "--port", str(port), "--parser", "recovery_progress:parse", "--linger-s", "5", "--",
                          sys.executable, str(COMP / "recover.py"), "--task", "open_drawer", "--rollouts",
                          f"{fixtures['demo']['open_drawer']}:open", "--out", str(out), "--episodes-per-failure", "1"],
                         env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
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
    assert res and res["state"] == "succeeded" and res["progress"]["episodes_accepted"] == 1, res
    assert st["progress"]["rollouts_done"] == 1
    assert {q: q.stat().st_mtime_ns for q in out.glob("*/jobs/*/result.json")} == before
    assert load(out / "manifest.json")["episodes"] == recovered["manifest"]["open_drawer"]["episodes"]
    numbers["resume_wall_s"] = round(time.time() - t0, 1)


def test_plan_cancel(recovered):
    """POST /cancel during a recovery plan job ends it promptly (SIGTERM reaches plan_from_state.py's group)."""
    import httpx
    out, key = recovered["where"]["fridge_approach"]
    src = out / key / "plans" / "e0_r0_s100"
    pdir = RUN / "cancel_plan"
    pdir.mkdir(parents=True)
    for f in ("regen.json", "start_state.npz", "start_state_next.npz"):
        shutil.copy(src / f, pdir / f)
    job = R.start_plan(["plan", "--dir", str(pdir), "--seed", "7", "--attempts", "10"], RUN / "cancel_job",
                       timeout=300, linger_s=3.0)
    t_cancel = None
    for _ in range(240):
        try:
            base = f"http://{job.ip()}:8000"
            s = httpx.get(base + "/status", timeout=2).json()
            if s["progress"].get("stage") == "plan":
                time.sleep(5)                     # plan_from_state.py (and its nvidia-smi sampler) are running
                assert httpx.post(base + "/cancel", timeout=2).json() == {"cancelled": True}
                t_cancel = time.time()
                break
        except (httpx.HTTPError, StopIteration, KeyError):
            pass
        time.sleep(0.5)
    res = R.finish(job)
    assert t_cancel is not None, "the plan job never reached the plan stage"
    assert time.time() - t_cancel < 30 and res["result"]["state"] == "cancelled", res
