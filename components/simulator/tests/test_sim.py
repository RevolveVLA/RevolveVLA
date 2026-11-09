"""Simulator component tests in Gazebo Kura + MuJoCo (see conftest.py)."""
import hashlib
import math

import numpy as np
import pytest
from conftest import EPISODE, RUN, load, plan_dir

from revolve_common import metrics as M
from revolve_common import specs

import spec

PHASES = ("approach", "grasp", "open", "release")


def checks_by_name(summary):
    return {c["name"]: c for c in summary["checks"]}


def done(jobs, name):
    r = jobs.result(name)
    assert r["result"] is not None, f"{name}: no result.json (exit {r['exit']}); see {r['dir']}/container.log"
    return r


def test_scene_check(jobs, numbers):
    """Kitchen scene (no robot): door / drawer open to their limits and close, rest drift and jitter,
    the cup stays on its fridge shelf, physics sanity (was the earlier kitchen_test)."""
    r = done(jobs, "scene_check")
    s = load(RUN / "scene_check" / "summary.json")
    c = checks_by_name(s)
    assert r["exit"] == 0 and s["pass_all"], [n for n, x in c.items() if not x["pass"]]
    for want in ("top_door_hinge: opens to upper limit (deg, end of push - limit)",
                 "middle_slide_top: opens to upper limit (mm, end of push - limit)",
                 "top_door_hinge: rest open drift over 9.5 s (deg)", "middle_slide_top: rest closed drift over 10 s (mm)",
                 "cup: max displacement over 34 s, door closed/opening/open/closing (cm)",
                 "cup: inside the fridge the whole time (samples outside)"):
        assert c[want]["pass"], want
    numbers["scene_check"] = dict(checks=f"{sum(x['pass'] for x in c.values())}/{len(c)}", wall_s=r["wall_s"],
                                  **{k: c[k]["value"] for k in c if "door" in k or "slide" in k or "cup" in k
                                     or "physics" in k})


def test_robot_smoke(jobs, numbers):
    """Mobile Panda spawned in the kitchen: smoke test of base, arm, gripper, contacts (kitchen_robot_test)."""
    r = done(jobs, "robot_smoke")
    s = load(RUN / "robot_smoke" / "summary.json")
    c = checks_by_name(s)
    assert r["exit"] == 0 and s["pass_all"], [n for n, x in c.items() if not x["pass"]]
    assert len(c) >= 20
    numbers["robot_smoke"] = dict(checks=f"{sum(x['pass'] for x in c.values())}/{len(c)}", wall_s=r["wall_s"],
                                  values={k: c[k]["value"] for k in c})


def test_can_demo(jobs, numbers):
    """The Panda can-on-plate demo: DONE SUCCESS, the can within 8 cm of the plate centre (README baseline 6.2)."""
    r = done(jobs, "can_demo")
    res = load(RUN / "can_demo" / "result_sim.json")
    assert r["exit"] == 0 and res["done"].startswith("DONE SUCCESS"), res
    assert res["distance_cm"] < 8.0
    numbers["can_demo"] = dict(done=res["done"], distance_cm=res["distance_cm"], wall_s=r["wall_s"])


def phase_report(name):
    return load(RUN / name / "phase_report.json")


def assert_conditions_from_task(rep, task):
    """The report's gates are the task spec's conditions, verbatim."""
    t = spec.task(task).task
    for p in t.phases:
        assert rep["conditions"][p.name] == [c.model_dump(exclude_none=True) for c in p.pass_.all], p.name
    assert rep["sanity_timed"] == [c.model_dump(exclude_none=True) for c in t.sanity.timed]
    for p, r in rep["phases"].items():
        names = {c["name"] for c in r["checks"]}
        want = {c.metric for c in t.phase(p).pass_.all if M.applies(c, offline=False, policy=False)}
        assert want <= names, (p, want - names)


def reevaluate(rep, task, override=None):
    """Re-judge the saved metric states with the task's conditions (override: metric -> condition kwargs)."""
    t = spec.task(task).task
    out = {}
    for p in t.phases:
        res = []
        for c in p.pass_.all:
            st = rep["metric_states"].get(p.name, {}).get(c.metric)
            if st is None:
                continue
            if override and c.metric in override:
                c = specs.Condition(**{**c.model_dump(exclude_none=True, exclude={"lt", "le", "gt", "ge", "eq"}),
                                       **override[c.metric]})
            res.append(M.evaluate(c, st)["pass_"])
        out[p.name] = all(res)
    return out


def phase_numbers(rep):
    return {p: dict(status=r["status"], **{c["name"]: c["value"] for c in r["checks"]}) for p, r in rep["phases"].items()}


def test_episode_fridge(jobs, numbers):
    """Scripted fridge episode (open the door to 95 deg, take the cup out, put it on the table): its own
    checks as the legacy `episode fridge`, and per-phase verdicts with the open_fridge_door task.yaml
    gates; every phase start state saved."""
    r = done(jobs, "episode_fridge")
    rep = phase_report("episode_fridge")
    assert rep["episode_checks"]["pass_all"], rep["episode_checks"]
    assert {p: x["status"] for p, x in rep["phases"].items()} == dict.fromkeys(PHASES, "pass")
    assert r["exit"] == 0 and rep["accepted"]
    assert_conditions_from_task(rep, "open_fridge_door")
    for p in PHASES:
        assert (RUN / "episode_fridge" / "phase_states" / f"{p}_start.npz").is_file(), p
    # the verdicts follow task.yaml: the saved states re-judge the same, and fail a stricter door gate
    assert reevaluate(rep, "open_fridge_door") == {"approach": True, "grasp": True, "open": True, "release": True}
    assert reevaluate(rep, "open_fridge_door", {"joint_deg": {"ge": 100.0}})["open"] is False
    numbers["episode_fridge"] = dict(episode_checks=rep["episode_checks"], phases=phase_numbers(rep), wall_s=r["wall_s"],
                                     episode=load(RUN / "episode_fridge" / "summary.json")["extra"]["episode"])


def test_episode_drawer(jobs, numbers):
    """Scripted drawer episode (pull the bar 19 cm, release, back off): its own checks pass (legacy `episode
    drawer`); with the open_drawer task.yaml gates the open phase fails only targets_static_s (the scripted
    episode holds 0.5 s before the release, the task's plan rule wants 1.0 s), every other gate passes."""
    r = done(jobs, "episode_drawer")
    rep = phase_report("episode_drawer")
    assert rep["episode_checks"]["pass_all"], rep["episode_checks"]
    st = {p: x["status"] for p, x in rep["phases"].items()}
    assert st == {"approach": "pass", "grasp": "pass", "open": "fail", "release": "pass"}, st
    assert rep["phases"]["open"]["failed_checks"] == ["targets_static_s"]
    assert rep["result"]["failed_phase"] == "open" and r["exit"] == 1
    joint = next(c for c in rep["phases"]["open"]["checks"] if c["name"] == "joint_cm")
    assert joint["pass"] and joint["value"] >= 15.0
    assert_conditions_from_task(rep, "open_drawer")
    # relaxing the task's hold rule to the episode's 0.5 s passes the phase: the verdict is task.yaml's
    assert reevaluate(rep, "open_drawer", {"targets_static_s": {"ge": 0.5}})["open"] is True
    numbers["episode_drawer"] = dict(episode_checks=rep["episode_checks"], phases=phase_numbers(rep), wall_s=r["wall_s"],
                                     episode=load(RUN / "episode_drawer" / "summary.json")["extra"]["episode"])


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def replay(jobs):
    if "replay" not in jobs.started:
        pytest.skip(f"no stored plan at {EPISODE} (REVOLVE_SIM_PLAN); the scripted episodes cover the phase path")
    return done(jobs, "replay")


def test_replay_stored_plan(jobs, replay, numbers):
    """One stored accepted AutoMoMa plan: offline verdict accepted, every phase passes, the MCAP bag is
    written and checks, and the state log is bit-identical to the stored replay of the same plan."""
    out = RUN / "replay"
    rep, v = phase_report("replay"), load(out / "verdict.json")
    assert replay["exit"] == 0 and v["verdict"] == "accepted", v["reasons"]
    assert {p: x["status"] for p, x in rep["phases"].items()} == dict.fromkeys(PHASES, "pass")
    assert rep["agreement"]["agree"]
    task = load(out / "episode.json")["task"]
    assert_conditions_from_task(rep, task)
    bc = load(out / "bag_check.json")
    assert (out / "bag" / "bag.mcap").is_file() and bc["pass"] is True, bc["failed"]
    assert bc["topics"]["/joint_states"]["count"] == load(out / "episode.json")["steps"]
    legacy = load(EPISODE / "replay" / "verdict.json")
    assert v["verdict"] == legacy["verdict"] and v["reasons"] == legacy["reasons"]
    same = sha(out / "state_log.npz") == sha(EPISODE / "replay" / "state_log.npz")
    numbers["replay"] = dict(plan=str(plan_dir()), verdict=v["verdict"], metrics=v["metrics"], phases=phase_numbers(rep),
                             state_log_sha256=sha(out / "state_log.npz"), identical_to_legacy=same,
                             bag=load(out / "bag.json")["size_mb"], wall_s=replay["wall_s"])
    assert same, "state_log.npz differs from the stored replay"


def test_wrapper_http(jobs, replay):
    """The job ran under revolve_common.wrapper: GET /status while running (409 on /result until the end),
    /result after, and status.json / result.json on the run volume."""
    jobs.poller.join(timeout=60)
    states = [s["state"] for s in jobs.http["status"]]
    assert "running" in states, jobs.http
    assert jobs.http.get("result_code_while_running") == 409
    assert jobs.http["result"] and jobs.http["result"]["state"] == "succeeded" and jobs.http["result"]["returncode"] == 0
    assert any(s["progress"].get("stage") for s in jobs.http["status"])
    assert replay["result"]["state"] == "succeeded"
    assert load(RUN / "replay_job" / "status.json")["state"] == "succeeded"


def test_restore_round_trip(jobs, replay, numbers):
    """Restore the replay's mid-episode open_start.npz (handle grasped, door closed) in a fresh gz and run
    on: the state one step later matches the source run's within task.live tolerances, the rest of the run
    follows the source, and the open and release phases pass again."""
    jobs.start("restore", ["restore-check", "--source", str(RUN / "replay"), "--phase", "open",
                           "--out", str(RUN / "restore_open")])
    r = done(jobs, "restore")
    rc = load(RUN / "restore_open" / "restore_check.json")
    live = spec.task(load(RUN / "replay" / "episode.json")["task"]).task.live
    assert rc["pass_"] and r["exit"] == 0, rc
    assert rc["vs_source_next_step"]["link_pos_m"][0] <= live.restore_pos_max_m
    assert math.degrees(rc["vs_source_next_step"]["link_rot_rad"][0]) <= live.restore_rot_max_deg
    d = rc["max_diff_over_run"]
    assert d["door (deg)"] < 0.5 and d["tcp_mm (mm)"] < 2.0, d
    rep = phase_report("restore_open")
    assert {p: x["status"] for p, x in rep["phases"].items()} == {"open": "pass", "release": "pass"}
    snap = dict(np.load(RUN / "replay" / "phase_states" / "open_start.npz"))
    assert int(snap["action_index"]) == rep["boundaries"]["open"]["action_index"]
    numbers["restore_open"] = dict(restore_check={k: rc[k] for k in ("pass_", "vs_source_next_step", "max_diff_over_run",
                                                                     "max_link_rot_deg_vs_next")},
                                   phases={p: x["status"] for p, x in rep["phases"].items()}, wall_s=r["wall_s"])
