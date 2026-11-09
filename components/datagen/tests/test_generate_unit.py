"""generate.py's composition with stubbed jobs (no docker, no GPU): plan -> check -> record order, the
acceptance rules, `workers` bound, manifest resume and the progress parser."""
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import generate as G
import progress


class Stub:
    """plan: attempts 0..4, `planned` succeed. replay: verdicts from `rejected` (check), `bad_record`
    (record with another state log), `bad_bag` (bag check fails)."""

    def __init__(self, planned=(0, 1, 2), rejected=(), bad_record=(), bad_bag=(), plan_rc=0, delay=0.0):
        self.planned, self.rejected, self.bad_record, self.bad_bag = planned, rejected, bad_record, bad_bag
        self.plan_rc, self.delay, self.calls, self.lock, self.active, self.peak = plan_rc, delay, [], threading.Lock(), 0, 0

    def plan(self, task, seed, out, job_dir, attempts=None):
        with self.lock:
            self.calls.append(("plan", seed))
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.delay)
        out.mkdir(parents=True, exist_ok=True)
        rows = [dict(attempt_id=a, success=a in self.planned) for a in range(5)]
        (out / "handoff.json").write_text(json.dumps(dict(attempts=5, planned=len(self.planned), attempts_detail=rows)))
        np.savez(out / "handoff.npz", joint_names=np.array(["j"]), attempt_id=np.arange(5),
                 traj_robot_world=np.zeros((5, 36, 12)), door_angle=np.zeros((5, 36)), tcp_world=np.zeros((5, 36, 4, 4)))
        (out / "fk_check.json").write_text(json.dumps(dict(passed=self.plan_rc == 0)))
        with self.lock:
            self.active -= 1
        return self.plan_rc

    def replay(self, task, plan_dir, attempt, out, job_dir, cameras=0):
        kind = "record" if cameras else "check"
        with self.lock:
            self.calls.append((kind, int(plan_dir.name.split("_")[1]), attempt))
        out.mkdir(parents=True, exist_ok=True)
        ok = attempt not in self.rejected
        (out / "verdict.json").write_text(json.dumps(dict(verdict="accepted" if ok else "rejected",
                                                          reasons=[] if ok else ["door_open_held: 40 (needs >= 60)"])))
        (out / "phase_report.json").write_text(json.dumps(dict(accepted=ok, result=None if ok else "open")))
        (out / "state_log.npz").write_bytes(b"log %d" % attempt + (b"x" if cameras and attempt in self.bad_record else b""))
        (out / "episode.json").write_text(json.dumps(dict(attempt_id=attempt)))
        if cameras:
            (out / "wrist_cam").mkdir(exist_ok=True)
            (out / "bag_check.json").write_text(json.dumps({"pass": attempt not in self.bad_bag, "failed": []}))
        return 0 if ok else 1

    def jobs(self):
        return G.Jobs(self.plan, self.replay)


def gen(tmp, stub, seeds=(0,), **kw):
    return G.generate("open_fridge_door", seeds, tmp, jobs=stub.jobs(), say=lambda m: None, **kw)


def test_one_episode_per_seed(tmp_path):
    st = Stub(planned=(1, 3), rejected=(1,))
    m = gen(tmp_path, st)
    assert st.calls == [("plan", 0), ("check", 0, 1), ("check", 0, 3), ("record", 0, 3)]
    assert m["episodes"] == ["seed_0_a3"] and m["seeds"]["0"]["state"] == "done"
    d = tmp_path / "episodes" / "seed_0_a3"
    assert not (d / "wrist_cam").exists(), "camera PNGs are removed after the bag check"
    assert json.loads((d / "episode.json").read_text())["datagen"] == dict(seed=0, attempt_id=3, accepted=True,
                                                                         label="demo", check="checks/seed_0_a3")
    assert json.loads((d / "plan.json").read_text())["attempt_id"] == 3
    assert np.load(d / "plan.npz")["traj_robot_world"].shape == (36, 12)
    assert m["counts"] == dict(seeds=1, seeds_done=1, episodes_planned=2, episodes_checked=2, episodes_accepted=1,
                               episodes_target=1)


def test_record_must_reproduce_the_check_and_pass_its_bag_check(tmp_path):
    st = Stub(planned=(0, 1, 2), bad_record=(0,), bad_bag=(1,))
    m = gen(tmp_path, st)
    eps = {e["name"]: e for e in m["seeds"]["0"]["episodes"]}
    assert eps["seed_0_a0"]["reasons"] == ["state_log.npz differs from the check's"]
    assert eps["seed_0_a1"]["reasons"][0].startswith("bag check failed")
    assert m["episodes"] == ["seed_0_a2"]


def test_short_and_failed_seeds(tmp_path):
    m = gen(tmp_path / "short", Stub(planned=(0,), rejected=(0,)), episodes_per_seed=1)
    assert m["seeds"]["0"]["state"] == "short" and m["episodes"] == []
    m = gen(tmp_path / "failed", Stub(plan_rc=1))
    assert m["seeds"]["0"]["state"] == "failed" and m["seeds"]["0"]["plan"]["rc"] == 1


def test_resume_skips_done_work(tmp_path):
    st = Stub(planned=(0, 1, 2))
    gen(tmp_path, st, seeds=(0, 1))
    n = len(st.calls)
    st2 = Stub(planned=(0, 1, 2))
    m = gen(tmp_path, st2, seeds=(0, 1))
    assert st2.calls == [] and m["episodes"] == ["seed_0_a0", "seed_1_a0"]
    m = gen(tmp_path, st2, seeds=(0, 1), episodes_per_seed=2)      # more per seed: only the new work runs
    assert sorted(st2.calls) == [("check", 0, 1), ("check", 1, 1), ("record", 0, 1), ("record", 1, 1)]
    assert len(m["episodes"]) == 4 and n == 6


def test_resume_redoes_an_unrecorded_item(tmp_path):
    st = Stub(planned=(0, 1))
    gen(tmp_path, st)
    man = json.loads((tmp_path / "manifest.json").read_text())
    man["seeds"]["0"]["episodes"] = []            # as if killed during the record
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    st2 = Stub(planned=(0, 1))
    m = gen(tmp_path, st2)
    assert st2.calls == [("record", 0, 0)] and m["episodes"] == ["seed_0_a0"]


def test_workers_bound_the_planners(tmp_path):
    st = Stub(delay=0.2)
    gen(tmp_path, st, seeds=range(6), workers=2)
    assert st.peak == 2 and len([c for c in st.calls if c[0] == "plan"]) == 6


def test_task_change_is_refused(tmp_path):
    gen(tmp_path, Stub())
    with pytest.raises(SystemExit):
        G.generate("open_drawer", [0], tmp_path, jobs=Stub().jobs(), say=lambda m: None)


def test_progress_parser():
    assert progress.parse('PROGRESS {"episodes_accepted": 2}') == {"episodes_accepted": 2}
    assert progress.parse("grasp 3: start IK 4 of 5 starts, seeded goal IK 4, clustered goal IK None") == \
        {"stage": "ik_traj", "grasps_done": 4}
    assert progress.parse(' "planned": 12,') == {"planned": 12}
    assert progress.parse(' "passed": true,') == {"fk_passed": True}
    assert progress.parse("unrelated") is None
    assert G.parse_seeds("0-3,8") == [0, 1, 2, 3, 8]


def test_crashed_plan_is_retried_on_resume(tmp_path):
    st = Stub(plan_rc=3)                         # entry.py: plan.py failed (crash, CUDA OOM)
    m = gen(tmp_path, st)
    assert m["seeds"]["0"]["state"] == "failed"
    st2 = Stub()
    m = gen(tmp_path, st2)
    assert st2.calls[0] == ("plan", 0) and m["episodes"] == ["seed_0_a0"]
