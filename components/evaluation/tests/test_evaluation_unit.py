"""Unit tests with stubs (no docker): wall limit, seed -> start resolution, retries, resume, stack errors,
progress parser, GPU parallelism, the client commands."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

COMP = Path(__file__).resolve().parents[1]
REPO = COMP.parents[1]
sys.path[:0] = [str(COMP), str(COMP / "src" / "evaluation")]
import evaluate as E  # noqa: E402
import eval_progress as progress  # noqa: E402
import rollout as R  # noqa: E402
import starts as S  # noqa: E402

TASK = "open_fridge_door"


STAMP = time.strftime("%Y%m%d_%H%M%S")


@pytest.fixture
def tmp(request):
    """A fresh dir under runs/ (rollout paths must be under the repo's runs/: the containers mount it)."""
    d = REPO / "runs" / "evaluation_tests" / f"unit_{STAMP}" / request.node.name
    d.mkdir(parents=True)
    return d


def test_wall_limit_from_budget_and_rtf():
    ro = S.G.specs_bridge.task_spec(TASK).task.rollout
    sim = R.sim_budget_s(ro)
    assert sim == ro.wait_s + ro.budget_s + ro.after_success_s
    assert R.sim_budget_s(ro, budget_s=6) == ro.wait_s + 6 + ro.after_success_s
    # derived, never the legacy fixed 285 s: halving the RTF doubles the limit
    assert R.wall_limit(sim, 0.12) == pytest.approx(sim / 0.12 * R.SAFETY, abs=0.1)
    assert R.wall_limit(sim, 0.06) == pytest.approx(2 * R.wall_limit(sim, 0.12), abs=0.2)
    assert R.wall_limit(sim, 0.0) == pytest.approx(sim / R.RTF_FLOOR * R.SAFETY, abs=0.1)


class FakeJobs:
    """plan: attempts 0-3 planned; replay: attempt 2 accepted (the first accepted one is the start)."""

    def __init__(self):
        self.calls = []

    def plan(self, task, seed, out, job_dir, attempts=None):
        self.calls.append(("plan", seed))
        out.mkdir(parents=True, exist_ok=True)
        (out / "handoff.json").write_text(json.dumps(dict(attempts_detail=[dict(attempt_id=a, success=True) for a in range(4)])))
        return 0

    def replay(self, task, plan_dir, attempt, out, job_dir, cameras=0):
        self.calls.append(("replay", attempt))
        out.mkdir(parents=True, exist_ok=True)
        ok = attempt == 2
        rng = np.random.default_rng(1000 * int(plan_dir.name.split("_")[1]) + attempt)    # depends on seed + attempt
        np.savez(out / "actions.npz", base=rng.normal(size=(5, 3)), arm=rng.normal(size=(5, 7)), grip=np.full(5, 0.04))
        np.savez(out / "state_log.npz", time_s=np.arange(5))
        (out / "episode.json").write_text(json.dumps(dict(plan_run=plan_dir.name, attempt_id=attempt, grasp_id=attempt % 5)))
        (out / "live.json").write_text(json.dumps(dict(grasp_pose_world=np.eye(4).tolist())))
        (out / "verdict.json").write_text(json.dumps(dict(verdict="accepted" if ok else "rejected", reasons=[])))
        (out / "phase_report.json").write_text(json.dumps(dict(accepted=ok, result={})))
        return 0 if ok else 1


def test_same_seed_same_start(tmp):
    jobs = FakeJobs()
    a = S.resolve(TASK, [7, 8], tmp / "starts", jobs=jobs, say=lambda m: None)
    n = len(jobs.calls)
    assert [c for c in jobs.calls if c[0] == "plan"] and a[7]["attempt_id"] == 2 and a[7]["source"] == "planned"
    b = S.resolve(TASK, [8, 7], tmp / "starts", jobs=jobs, say=lambda m: None)
    assert len(jobs.calls) == n                         # cached: no job ran again
    assert a == b and a[7]["spawn"] != a[8]["spawn"]    # same seed -> same start; seeds differ
    sp = np.load(REPO / a[7]["dir"] / "actions.npz")
    assert a[7]["spawn"]["base"] == sp["base"][0].tolist()
    # a start whose files changed under it is refused, not silently replaced
    np.savez(REPO / a[7]["dir"] / "actions.npz", base=np.zeros((5, 3)), arm=np.zeros((5, 7)), grip=np.zeros(5))
    with pytest.raises(ValueError, match="changed"):
        S.resolve(TASK, [7], tmp / "starts", jobs=jobs, say=lambda m: None)


def test_start_from_datagen_source(tmp):
    src = tmp / "data"
    jobs = FakeJobs()
    jobs.replay(TASK, src / "plans" / "seed_3", 1, src / "checks" / "seed_3_a1", None)
    (src / "checks" / "seed_3_a1" / "verdict.json").write_text(json.dumps(dict(verdict="accepted", reasons=[])))
    (src / "manifest.json").write_text(json.dumps(dict(task=TASK, seeds={"3": dict(
        plan=dict(planned=[0, 1]), checks={"0": dict(accepted=False, dir="checks/seed_3_a0"),
                                           "1": dict(accepted=True, dir="checks/seed_3_a1")})})))
    jobs.calls.clear()
    r = S.resolve(TASK, [3], tmp / "starts", sources=[src], jobs=jobs, say=lambda m: None)[3]
    assert jobs.calls == [] and r["source"] == "source" and r["dir"].endswith("checks/seed_3_a1")


def fake_runner(script):
    """A rollout stub: script[(dir name, try)] -> failed phase (None = pass, 'ERR' = stack error)."""
    calls = []

    def run(start, out, *, seed, policy, ckpt, restore, rtf, task, say, **budget):
        out.mkdir(parents=True, exist_ok=True)
        n = sum(1 for c in calls if c["out"] == out)
        calls.append(dict(out=out, restore=restore, rtf=rtf, budget=budget))
        key = out.name if not out.name.startswith("retry_") else f"{out.parent.name}/{out.name}"
        fp = script.get((key, n), None)
        if fp == "ERR":
            rec = dict(dir=str(out.relative_to(REPO)), phases=None, error="boom", timing={}, restore=None)
        else:
            phases = {p: "pass" for p in ("approach", "grasp", "open", "release")}
            if fp:
                phases[fp] = "fail"
            (out / "phase_states").mkdir(exist_ok=True)
            for p in phases:
                (out / "phase_states" / f"{p}_start.npz").write_bytes(b"x")
            rec = dict(dir=str(out.relative_to(REPO)), phases=phases, failed_phase=fp, passed=fp is None, error=None,
                       rtf=0.3, sim_s=10.0, timing=dict(rollout_wall_s=50.0),
                       restore=None if restore is None else dict(source=str(restore[0]), phase=restore[1]))
        (out / "rollout.json").write_text(json.dumps(rec))
        return rec
    return run, calls


def ev(tmp, script, **kw):
    run, calls = fake_runner(script)
    starts = dict(dir=str((tmp / "s").relative_to(REPO)))
    e = E.Evaluation(TASK, None, kw.pop("seeds", [5, 6]), tmp / "eval", "heldout", tmp / "starts", runner=run,
                     parallel=2, rtf=0.2, say=lambda m: None, **kw)
    S_resolve = S.resolve
    S.resolve = lambda *a, **k: {s: starts for s in e.seeds}
    try:
        return e.run(), calls
    finally:
        S.resolve = S_resolve


def test_retry_restores_the_failed_phase(tmp):
    s, calls = ev(tmp, {("5_r0", 0): "grasp", ("5_r0/retry_1", 0): None, ("6_r0", 0): "open", ("6_r0/retry_1", 0): "open",
                        ("6_r0/retry_2", 0): "open"}, retries=2)
    r5, r6 = s["table"]["5"]["r0"], s["table"]["6"]["r0"]
    retry = [c for c in calls if c["restore"] is not None]
    assert {(c["out"].name, c["restore"][1]) for c in retry} == {("retry_1", "grasp"), ("retry_1", "open"), ("retry_2", "open")}
    first5 = next(c for c in retry if c["out"].parent.name == "5_r0")
    assert first5["restore"][0] == REPO / r5["dir"]     # restored from the first try's own phase state
    assert r5["passed_first_try"] is False and r5["passed"] is True and r5["tries"] == 2
    assert r6["passed"] is False and r6["tries"] == 3 and r6["final_failed_phase"] == "open"
    assert s["first_try_passed"] == 0 and s["passed_with_retries"] == 1
    assert s["all_pass"] is False and s["all_pass_with_retries"] is False
    assert json.loads((tmp / "eval" / "heldout" / "eval_summary.json").read_text())["table"] == s["table"]


def test_no_retry_by_default_and_all_pass(tmp):
    s, calls = ev(tmp, {})
    assert len(calls) == 2 and all(c["restore"] is None for c in calls)
    assert s["all_pass"] and s["first_try_success"] == 1.0
    assert all(c["rtf"] == 0.2 for c in calls)


def test_global_failure_is_not_restored(tmp):
    s, calls = ev(tmp, {("5_r0", 0): "global"}, retries=1, seeds=[5])
    assert len(calls) == 1 and s["table"]["5"]["r0"]["tries"] == 1


def test_stack_error_rerun_once_and_resume(tmp):
    s, calls = ev(tmp, {("5_r0", 0): "ERR"}, seeds=[5])
    assert len(calls) == 2 and s["table"]["5"]["r0"]["passed"]
    s2, calls2 = ev(tmp, {("5_r0", 0): "open"}, seeds=[5])     # same dirs: finished rollouts are kept
    assert calls2 == [] and s2["table"]["5"]["r0"]["passed"]
    s3, _ = ev(tmp, {("7_r0", 0): "ERR", ("7_r0", 1): "ERR"}, seeds=[7])
    assert s3["errors"] == 1 and s3["table"]["7"]["r0"]["error"] == "boom"


def test_progress_parser():
    assert progress.parse('PROGRESS {"rollouts": 4, "rollouts_done": 1, "passed": 1}') == dict(rollouts=4, rollouts_done=1,
                                                                                               passed=1)
    assert progress.parse("10:00:00 evaluate heldout: 100_r0: try 0: FAIL approach, try 1: PASS") == \
        {"last_rollout": "100_r0: try 0: FAIL approach, try 1: PASS"}
    assert progress.parse("anything else") is None


def test_default_parallel(monkeypatch):
    monkeypatch.setattr(E, "gpu_mib", lambda: (8151, 13))
    n = E.default_parallel("smolvla")
    assert n == min(4, (8151 - 13 - E.GPU_RESERVE_MIB) // E.GPU_MIB_PER_ROLLOUT["smolvla"]) and n >= 1
    monkeypatch.setattr(E, "gpu_mib", lambda: (8151, 8000))
    assert E.default_parallel("smolvla") == 1
    monkeypatch.setattr(E, "gpu_mib", lambda: None)
    assert E.default_parallel("oracle") == 1


def test_client_commands(tmp):
    c = REPO / "specs" / "robots" / "mobile_panda" / "contract.yaml"
    s = R.client_script("smolvla", tmp, tmp / "start", "open the fridge door", c, REPO / "runs" / "ck", "rollout-x-serve", 900)
    for part in ("ros2 launch rosetta_ros policy_runner_launch.py", "server_address:=rollout-x-serve:8080",
                 "launch_local_server:=false", "use_sim_time:=true", "pretrained_name_or_path:=/repo/runs/ck",
                 "rosetta_goal.py", "'open the fridge door' 900", "/summary.json"):
        assert part in s, part
    o = R.client_script("oracle", tmp, tmp / "start", "p", c)
    assert "dummy_policy.py" in o and "policy_runner" not in o and "runner_ready" in o


def test_gui_forces_parallel_1_and_reaches_the_runner(tmp):
    s, calls = ev(tmp, {}, gui=True)
    assert s["parallel"] == 1 and {c["budget"]["gui"] for c in calls} == {True}
    s, calls = ev(tmp / "headless", {})
    assert s["parallel"] == 2 and {c["budget"]["gui"] for c in calls} == {False}


def test_video_reaches_the_runner(tmp):
    s, calls = ev(tmp, {}, video=True)
    assert {c["budget"]["video"] for c in calls} == {True}
    s, calls = ev(tmp / "plain", {})
    assert {c["budget"]["video"] for c in calls} == {False}


def test_gui_display_env_and_mounts(tmp, monkeypatch):
    L, seen = R.sim_launch, []

    class Docker:
        containers = type("C", (), {"run": staticmethod(lambda *a, **kw: seen.append(kw))})

    (tmp / "install").mkdir()
    (tmp / "install" / "setup.bash").write_text("")
    (tmp / "xauth").write_text("cookie")
    monkeypatch.setattr(L, "INSTALL", tmp / "install")
    monkeypatch.setattr(L.docker, "from_env", lambda: Docker)
    monkeypatch.setenv("DISPLAY", ":7")
    monkeypatch.setenv("XAUTHORITY", str(tmp / "xauth"))
    L.start(["rollout"], tmp / "job_gui", gui=True)
    L.start(["rollout"], tmp / "job")
    g, h = seen
    assert g["environment"]["DISPLAY"] == ":7" and g["environment"]["XAUTHORITY"] == "/tmp/.Xauthority"
    assert g["volumes"]["/tmp/.X11-unix"] == {"bind": "/tmp/.X11-unix", "mode": "ro"}
    assert g["volumes"][str(tmp / "xauth")] == {"bind": "/tmp/.Xauthority", "mode": "ro"}
    assert "DISPLAY" not in h["environment"] and "XAUTHORITY" not in h["environment"]
    assert set(h["volumes"]) == set(g["volumes"]) - {"/tmp/.X11-unix", str(tmp / "xauth")}
    monkeypatch.delenv("DISPLAY")
    with pytest.raises(SystemExit):
        L.start(["rollout"], tmp / "job_nodisplay", gui=True)
