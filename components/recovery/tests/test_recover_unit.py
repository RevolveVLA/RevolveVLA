"""recover.py's composition with stubbed jobs (no docker, no GPU): which failures are recovered and how
(mode, restore phase, grasp_from_state, the start state handed to the planner), the planner seed schedule,
the acceptance rules, the labels, manifest resume, the CLI rollout syntax and the progress parser."""
import json
import threading
from pathlib import Path

import numpy as np
import pytest

import recover as R
import recovery_progress

PHASES = ("approach", "grasp", "open", "release")
STEP = {"approach": 1, "grasp": 2501, "open": 3271, "release": 15000}


def rollout(tmp, name="r0", failed="open", regenerate=True, policy=False, task="open_fridge_door", grasp_id=0, seed=7,
            missing=()):
    d = Path(tmp) / "rollouts" / name
    (d / "phase_states").mkdir(parents=True)
    for p in PHASES:
        if p in missing:
            continue
        for s in ("", "_next"):
            np.savez(d / "phase_states" / f"{p}_start{s}.npz", action_index=STEP[p] + (s == "_next"), phase=p)
    res = dict(failed_phase=failed, regenerate=regenerate, failed_checks=["handle_slip_m"] if failed else [])
    (d / "phase_report.json").write_text(json.dumps(dict(result=res)))
    ep = dict(task=task, plan_run="seed_7", attempt_id=3, grasp_id=grasp_id, seed=seed)
    if policy:
        ep["policy_rollout"] = {"checkpoint": "x"}
        ep.pop("grasp_id")
    (d / "episode.json").write_text(json.dumps(ep))
    return d


class Stub:
    """plan: attempts 0..3, `planned` succeed (per seed: planned_by_seed). candidate: exit 2 for `precheck`
    attempts, 1 for `rejected`, a different state log on record for `bad_record`."""

    def __init__(self, planned=(0, 1), planned_by_seed=None, precheck=(), rejected=(), bad_record=(), plan_rc=0):
        self.planned, self.by_seed, self.precheck, self.rejected = planned, planned_by_seed or {}, precheck, rejected
        self.bad_record, self.plan_rc, self.calls, self.lock = bad_record, plan_rc, [], threading.Lock()

    def plan(self, plan_dir, seed, attempts, job_dir):
        with self.lock:
            self.calls.append(("plan", plan_dir.parent.parent.name, seed))
        rg = json.loads((plan_dir / "regen.json").read_text())
        assert (plan_dir / "start_state.npz").is_file() and (plan_dir / "start_state_next.npz").is_file()
        planned = self.by_seed.get(seed, self.planned)
        if self.plan_rc == 0:
            rows = [dict(attempt_id=a, success=a in planned) for a in range(4)]
            (plan_dir / "handoff.json").write_text(json.dumps(dict(attempts=4, planned=len(planned), attempts_detail=rows,
                                                                   regenerate=rg)))
        return self.plan_rc

    def candidate(self, plan_dir, attempt, out, job_dir, cameras=0):
        kind = "record" if cameras else "check"
        seed = json.loads((plan_dir / "regen.json").read_text())   # noqa: F841 (the plan dir must hold it)
        with self.lock:
            self.calls.append((kind, plan_dir.name, attempt))
        out.mkdir(parents=True, exist_ok=True)
        if attempt in self.precheck:
            (out / "precheck.json").write_text(json.dumps({"pass_": False}))
            return 2
        ok = attempt not in self.rejected
        (out / "verdict.json").write_text(json.dumps(dict(verdict="accepted" if ok else "rejected", reasons=[] if ok else ["x"])))
        (out / "phase_report.json").write_text(json.dumps(dict(accepted=ok, result=None if ok else "open")))
        (out / "state_log.npz").write_bytes(b"log %s %d" % (plan_dir.name.encode(), attempt)
                                            + (b"x" if cameras and attempt in self.bad_record else b""))
        (out / "episode.json").write_text(json.dumps(dict(attempt_id=attempt, regenerated={"phase": "open"})))
        (out / "phase_states").mkdir(exist_ok=True)
        (out / "restore").mkdir(exist_ok=True)
        (out / "restore" / "open_start.npz").write_bytes(b"snap")
        if cameras:
            (out / "wrist_cam").mkdir(exist_ok=True)
            (out / "bag_check.json").write_text(json.dumps({"pass": True, "failed": []}))
        return 0 if ok else 1

    def jobs(self):
        return R.Jobs(self.plan, self.candidate)


def rec(tmp, stub, rollouts, task="open_fridge_door", **kw):
    return R.recover(task, rollouts, Path(tmp) / "out", jobs=stub.jobs(), say=lambda m: None, **kw)


def test_open_failure_one_episode(tmp_path):
    d = rollout(tmp_path)
    s = Stub()
    m = rec(tmp_path, s, [d], episodes_per_failure=1)
    r = m["rollouts"]["r0"]
    assert r["state"] == "done" and r["mode"] == "pull" and r["restore_phase"] == "open" and not r["grasp_from_state"]
    assert r["action_index"] == STEP["open"]
    assert s.calls == [("plan", "r0", 100), ("check", "e0_r0_s100", 0), ("record", "e0_r0_s100", 0)]
    out = tmp_path / "out" / "r0"
    rg = json.loads((out / "plans" / "e0_r0_s100" / "regen.json").read_text())
    assert rg["mode"] == "pull" and rg["source_grasp_id"] == 0 and rg["source_attempt_id"] == 3 and rg["task"] == "open_fridge_door"
    assert "grasp_from_state" not in rg
    assert str(np.load(out / "plans" / "e0_r0_s100" / "start_state.npz")["phase"]) == "open"
    ep = json.loads((out / "episodes" / "ep_0" / "episode.json").read_text())
    lab = ep["recovery"]
    assert lab["source"] == "recovery" and lab["phase"] == "open" and lab["suffix"] is True and lab["accepted"]
    assert lab["from_rollout"] == str(d) and lab["rollout_seed"] == 7 and lab["planner_seed"] == 100
    assert not (out / "episodes" / "ep_0" / "wrist_cam").exists()         # camera PNGs dropped (in the bag)
    assert (out / "episodes" / "ep_0" / "phase_states" / "open_start.npz").is_file()   # the restored start
    assert m["episodes"] == ["r0/episodes/ep_0"] and m["counts"]["episodes_accepted"] == 1


def test_policy_open_plans_from_the_held_grasp(tmp_path):
    d = rollout(tmp_path, policy=True)
    m = rec(tmp_path, Stub(), [d], episodes_per_failure=1)
    assert m["rollouts"]["r0"]["grasp_from_state"]
    rg = json.loads((tmp_path / "out" / "r0" / "plans" / "e0_r0_s100" / "regen.json").read_text())
    assert rg["grasp_from_state"] is True and rg["source_grasp_id"] is None


@pytest.mark.parametrize("phase,restore", [("approach", "approach"), ("grasp", "approach")])
def test_reach_failures(tmp_path, phase, restore):
    """approach / grasp: MotionGen reach mode from that phase's start; a grasp restart is judged from approach."""
    d = rollout(tmp_path, failed=phase)
    m = rec(tmp_path, Stub(), [d], episodes_per_failure=1)
    r = m["rollouts"]["r0"]
    assert r["state"] == "done" and r["mode"] == "reach" and r["restore_phase"] == restore and r["action_index"] == STEP[phase]
    pdir = tmp_path / "out" / "r0" / "plans" / "e0_r0_s100"
    assert str(np.load(pdir / "start_state.npz")["phase"]) == phase
    assert json.loads((pdir / "regen.json").read_text())["restore_phase"] == restore


@pytest.mark.parametrize("failed,regenerate,phases,state", [
    ("release", True, R.REGENERABLE, "unsupported"), ("global", False, R.REGENERABLE, "unsupported"),
    ("open", False, R.REGENERABLE, "unsupported"), (None, False, R.REGENERABLE, "passed"),
    ("grasp", True, ["open"], "skipped")])
def test_not_recovered(tmp_path, failed, regenerate, phases, state):
    s = Stub()
    m = rec(tmp_path, s, [rollout(tmp_path, failed=failed, regenerate=regenerate)], phases=phases)
    assert m["rollouts"]["r0"]["state"] == state and s.calls == [] and m["counts"]["unsupported"] == 1


def test_missing_state_and_wrong_task(tmp_path):
    a = rollout(tmp_path, "a", missing=("open",))
    b = rollout(tmp_path, "b", task="open_drawer")
    m = rec(tmp_path, Stub(), [a, b])
    assert m["rollouts"]["a"]["state"] == "error" and "open_start" in m["rollouts"]["a"]["reason"]
    assert m["rollouts"]["b"]["state"] == "error" and "open_drawer" in m["rollouts"]["b"]["reason"]


def test_forced_phase_names(tmp_path):
    """The same rollout forced to several phases (an accepted demo's states): one entry each."""
    d = rollout(tmp_path, failed=None)
    m = rec(tmp_path, Stub(), [(d, "approach"), (d, "open")], episodes_per_failure=1)
    assert {k: v["state"] for k, v in m["rollouts"].items()} == {"r0_from_approach": "done", "r0_from_open": "done"}
    with pytest.raises(SystemExit):
        rec(tmp_path, Stub(), [d, d])


def test_seed_schedule_and_rejections(tmp_path):
    """Episode i tries seeds seed0 + runs*i + r; precheck and gz rejections move on; a run with nothing
    accepted goes to the next seed; the budget used up leaves the rollout short."""
    d = rollout(tmp_path)
    # e0: seed 100 a0 precheck, a1 rejected -> seed 101 a2 accepted; e1: seed 103..105 nothing planned
    s = Stub(planned_by_seed={100: (0, 1), 101: (2,), 103: (), 104: (), 105: ()}, precheck=(0,), rejected=(1,))
    m = rec(tmp_path, s, [d], episodes_per_failure=2)
    r = m["rollouts"]["r0"]
    assert [c for c in s.calls if c[0] == "plan"] == [("plan", "r0", x) for x in (100, 101, 103, 104, 105)]
    assert r["state"] == "short" and len([e for e in r["episodes"] if e["accepted"]]) == 1
    assert r["checks"]["e0_r0_s100_a0"]["precheck"] is False and not r["checks"]["e0_r0_s100_a1"]["accepted"]
    assert r["episodes"][0]["plan"] == "e0_r1_s101" and r["episodes"][0]["attempt_id"] == 2


def test_record_must_reproduce_the_check(tmp_path):
    d = rollout(tmp_path)
    s = Stub(planned=(0, 1), bad_record=(0,))
    m = rec(tmp_path, s, [d], episodes_per_failure=1)
    eps = m["rollouts"]["r0"]["episodes"]
    assert [e["accepted"] for e in eps] == [False, True] and "differs" in eps[0]["reasons"][0]
    assert eps[1]["attempt_id"] == 1 and m["counts"]["episodes_accepted"] == 1


def test_resume(tmp_path):
    d = rollout(tmp_path)
    rec(tmp_path, Stub(plan_rc=3), [d], episodes_per_failure=1)          # every plan run crashes
    s = Stub()
    m = rec(tmp_path, s, [d], episodes_per_failure=1)                    # crashed plans are redone
    assert m["rollouts"]["r0"]["state"] == "done" and s.calls[0] == ("plan", "r0", 100)

    class Boom(Stub):
        def plan(self, *a):
            raise AssertionError("re-planned")

        def candidate(self, *a, **k):
            raise AssertionError("re-ran")
    m2 = rec(tmp_path, Boom(), [d], episodes_per_failure=1)              # finished: nothing runs again
    assert m2["rollouts"]["r0"]["state"] == "done" and m2["episodes"] == m["episodes"]


def test_workers_and_planner_cap(tmp_path, monkeypatch):
    """Rollouts run `workers` at a time; docker plan jobs share one process-wide GPU semaphore."""
    import time
    active, peak, lock = [0], [0], threading.Lock()

    class Slow(Stub):
        def plan(self, *a):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.2)
            with lock:
                active[0] -= 1
            return super().plan(*a)
    ds = [rollout(tmp_path, f"r{i}") for i in range(4)]
    m = rec(tmp_path, Slow(), ds, episodes_per_failure=1, workers=3)
    assert peak[0] == 3 and m["counts"]["episodes_accepted"] == 4
    assert R.PLANNERS._value == int(R.os.environ.get("REVOLVE_GPU_PLANNERS", "2"))


def test_cli_rollout_syntax():
    assert R.parse_rollout("runs/x/r0") == "runs/x/r0"
    assert R.parse_rollout("runs/x/r0:grasp") == ("runs/x/r0", "grasp")


def test_progress_parser():
    assert recovery_progress.parse('PROGRESS {"episodes_accepted": 2}') == {"episodes_accepted": 2}
    assert recovery_progress.parse("attempt 3: reach ok 2.10 s") == {"stage": "reach", "reach_attempt": 3, "reach_ok": True}
    assert recovery_progress.parse(' "planned": 4,') == {"planned": 4}
    assert recovery_progress.parse(' "reach_success": 5,') == {"reach_success": 5}
    assert recovery_progress.parse("random line") is None
