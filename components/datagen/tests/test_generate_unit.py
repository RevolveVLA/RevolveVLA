"""generate.py's composition with stubbed jobs (no docker, no GPU): plan -> check -> record order, the
acceptance rules, `workers` bound, manifest resume, the progress parser, the N-of-N repeat check, layouts and
record=False."""
import json
import sys
import types
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import generate as G
import progress


class Stub:
    """plan: attempts 0..4, `planned` succeed. replay: verdicts from `rejected` (check), `bad_record`
    (record with another state log), `bad_bag` (bag check fails); `dirs` {out dir name: reject | differ
    (another state log) | crash}; check out dir names in `checked`; the layouts of the plan calls in `layouts`."""

    def __init__(self, planned=(0, 1, 2), rejected=(), bad_record=(), bad_bag=(), plan_rc=0, delay=0.0, dirs=None):
        self.planned, self.rejected, self.bad_record, self.bad_bag = planned, rejected, bad_record, bad_bag
        self.dirs, self.checked, self.layouts = dirs or {}, [], []
        self.plan_rc, self.delay, self.calls, self.lock, self.active, self.peak = plan_rc, delay, [], threading.Lock(), 0, 0

    def plan(self, task, seed, out, job_dir, attempts=None, layout=None):
        with self.lock:
            self.calls.append(("plan", seed))
            self.layouts.append(layout)
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
            self.checked += [] if cameras else [out.name]
        if self.dirs.get(out.name) == "crash":
            raise RuntimeError("docker died")
        out.mkdir(parents=True, exist_ok=True)
        ok = attempt not in self.rejected and self.dirs.get(out.name) != "reject"
        (out / "verdict.json").write_text(json.dumps(dict(verdict="accepted" if ok else "rejected",
                                                          reasons=[] if ok else ["door_open_held: 40 (needs >= 60)"])))
        (out / "phase_report.json").write_text(json.dumps(dict(accepted=ok, result=None if ok else "open")))
        (out / "state_log.npz").write_bytes(b"log %d" % attempt + (b"x" if cameras and attempt in self.bad_record else b"")
                                            + (b"y" if self.dirs.get(out.name) == "differ" else b""))
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


# ------------------------------------------------------------------ --checks: N-of-N repeat check
def test_checks_1_is_today(tmp_path, monkeypatch):
    a, b = Stub(planned=(1, 3), rejected=(1,)), Stub(planned=(1, 3), rejected=(1,))
    ma, mb = gen(tmp_path / "a", a), gen(tmp_path / "b", b, checks=1)
    assert a.calls == b.calls and a.checked == ["seed_0_a1", "seed_0_a3"]
    strip = lambda m: {k: v for k, v in m.items() if k not in ("created_at", "updated_at")}  # noqa: E731
    assert strip(ma) == strip(mb) and "checks" not in mb and set(mb["seeds"]["0"]["checks"]["3"]) == {
        "rc", "accepted", "verdict", "reasons", "failed_phase", "state_log_sha256", "dir"}
    got = []
    monkeypatch.setattr(G, "generate", lambda *a, **kw: got.append(kw) or dict(seeds={}))
    G.main(["--task", "t", "--seeds", "0", "--out", str(tmp_path / "c")])
    assert got == [dict(checks=1)]


def test_checks_all_pass(tmp_path):
    st = Stub(planned=(0, 1))
    m = gen(tmp_path, st, checks=3)
    assert st.calls == [("plan", 0), ("check", 0, 0), ("check", 0, 0), ("check", 0, 0), ("record", 0, 0)]
    assert st.checked == ["seed_0_a0", "seed_0_a0_r1", "seed_0_a0_r2"] and m["checks"] == 3
    chk = m["seeds"]["0"]["checks"]["0"]
    sha = G.sha256(tmp_path / "checks" / "seed_0_a0" / "state_log.npz")
    assert chk["accepted"] and chk["dir"] == "checks/seed_0_a0" and chk["state_log_sha256"] == sha
    assert chk["repeat_check"] == dict(n=3, accepted=[True] * 3, state_log_sha256=[sha] * 3, identical=True, reason=None)
    assert [r["dir"] for r in chk["repeats"]] == ["checks/seed_0_a0_r1", "checks/seed_0_a0_r2"]
    assert m["episodes"] == ["seed_0_a0"] and m["seeds"]["0"]["episodes"][0]["check"] == "checks/seed_0_a0"


def test_checks_one_repeat_fails_next_attempt(tmp_path):
    st = Stub(planned=(0, 1, 2), dirs={"seed_0_a0_r1": "reject", "seed_0_a1_r2": "differ"})
    m = gen(tmp_path, st, checks=3)
    assert st.checked == ["seed_0_a0", "seed_0_a0_r1", "seed_0_a1", "seed_0_a1_r1", "seed_0_a1_r2"]
    c0, c1 = m["seeds"]["0"]["checks"]["0"], m["seeds"]["0"]["checks"]["1"]
    assert not c0["accepted"] and c0["repeat_check"]["accepted"] == [True, False]
    assert c0["repeat_check"]["reason"] == "repeat check 2/3 failed" and len(c0["repeats"]) == 1
    assert c1["accepted"] and not c1["repeat_check"]["identical"] and c1["repeat_check"]["reason"] is None
    assert len(set(c1["repeat_check"]["state_log_sha256"])) == 2
    assert m["episodes"] == ["seed_0_a1"] and st.calls[-1] == ("record", 0, 1)   # record matches repeat 0
    m = gen(tmp_path / "r0", Stub(planned=(0,), rejected=(0,)), checks=3)
    c = m["seeds"]["0"]["checks"]["0"]
    assert c["repeat_check"]["reason"] == "repeat check 1/3 failed" and c["repeats"] == []


def test_checks_resume_reuses_partial_repeats(tmp_path):
    m = gen(tmp_path, Stub(planned=(0, 1), dirs={"seed_0_a0_r2": "crash"}), checks=3)
    c = m["seeds"]["0"]["checks"]["0"]
    assert m["seeds"]["0"]["state"] == "error" and not c["accepted"]
    assert c["repeat_check"] == dict(n=3, accepted=[True, True], state_log_sha256=[c["state_log_sha256"]] * 2)
    st = Stub(planned=(0, 1))
    m = gen(tmp_path, st, checks=3)
    assert st.checked == ["seed_0_a0_r2"] and st.calls[-1] == ("record", 0, 0)
    assert m["seeds"]["0"]["checks"]["0"]["repeat_check"]["accepted"] == [True] * 3 and m["episodes"] == ["seed_0_a0"]
    st = Stub(planned=(0, 1))
    gen(tmp_path, st, checks=3)
    assert st.calls == []


def test_repeat_check_helper():
    chk = dict(accepted=True, state_log_sha256="a", dir="c")
    assert G.repeat_check(1, chk, None) is chk
    saved, runs = [], iter([dict(accepted=True, state_log_sha256="a"), dict(accepted=False, state_log_sha256=None)])
    out = G.repeat_check(4, chk, lambda k: next(runs), saved.append)
    assert chk == dict(accepted=True, state_log_sha256="a", dir="c")          # not mutated
    assert out["repeat_check"] == dict(n=4, accepted=[True, True, False], state_log_sha256=["a", "a", None],
                                       identical=False, reason="repeat check 3/4 failed")
    assert [s["repeat_check"]["accepted"] for s in saved] == [[True], [True, True], [True, True, False]]
    assert not any(s["accepted"] for s in saved) and G.repeat_check(4, out, None) is out


# ------------------------------------------------------------------ layouts, record=False, failure reasons
def test_layout_goes_to_the_plan_job_and_into_episode_json(tmp_path):
    st = Stub(planned=(0,))
    layout = {"start_pose": [0.1, 0.2, 3.0], "grasp_id": 2}
    gen(tmp_path, st, seeds=(0, 1), layouts={0: layout})
    assert st.layouts == [layout, None]                       # seed 1 has none: the call carries no layout
    ep = lambda s: json.loads((tmp_path / "episodes" / f"seed_{s}_a0" / "episode.json").read_text())["datagen"]  # noqa: E731
    assert ep(0)["layout"] == layout and "layout" not in ep(1)


def test_record_false_stops_at_the_accepted_check(tmp_path):
    st = Stub(planned=(0, 1, 2), rejected=(0,))
    m = gen(tmp_path, st, record=False)
    assert st.calls == [("plan", 0), ("check", 0, 0), ("check", 0, 1)] and not (tmp_path / "episodes").exists()
    assert m["episodes"] == ["seed_0_a1"] and m["seeds"]["0"]["state"] == "done"
    e = m["seeds"]["0"]["episodes"][0]
    assert (e["attempt_id"], e["check"], e["label"], e["accepted"]) == (1, "checks/seed_0_a1", "check", True)


def test_plan_failure_reasons():
    h = lambda **row: lambda: dict(attempts_detail=[row])  # noqa: E731
    assert G.plan_failure(3, [], h(), False) == "plan failed" and G.plan_failure(2, [0], h(), False) == "plan failed"
    assert G.plan_failure(1, [0], h(), False) == "FK check failed" and G.plan_failure(0, [0], h(), True) is None
    assert G.plan_failure(0, [], h(goal_ik="none"), False) == "no IK"
    assert G.plan_failure(0, [], h(goal_ik="seeded"), False) == "plan failed"       # IK found, trajopt failed
    assert G.plan_failure(0, [], h(goal_ik_cluster=2), False) == "plan failed"


def test_docker_plan_args(monkeypatch):
    runs = []
    monkeypatch.setitem(sys.modules, "launch", types.SimpleNamespace(run=lambda args, job_dir, **kw: runs.append(args) or {"exit": 0}))
    G.docker_plan("t", 3, "/o", "/j", 5)
    G.docker_plan("t", 3, "/o", "/j", None, layout={"grasp_id": 1})
    assert runs == [["plan", "--task", "t", "--seed", "3", "--out", "/o", "--attempts", "5"],
                    ["plan", "--task", "t", "--seed", "3", "--out", "/o", "--layout", '{"grasp_id": 1}']]


def test_record_attempt_records_the_accepted_check_of_a_record_false_run(tmp_path):
    st = Stub(planned=(1, 3), rejected=(1,))
    gen(tmp_path, st, record=False, layouts={0: {"grasp_id": 2}}, checks=1)
    assert st.calls == [("plan", 0), ("check", 0, 1), ("check", 0, 3)]
    e = G.record_attempt("open_fridge_door", 0, 3, tmp_path, jobs=st.jobs(), say=lambda m: None)
    assert st.calls[-1] == ("record", 0, 3) and e["accepted"] and e["label"] == "demo" and e["dir"] == "episodes/seed_0_a3"
    ep = json.loads((tmp_path / "episodes" / "seed_0_a3" / "episode.json").read_text())
    assert ep["datagen"]["layout"] == {"grasp_id": 2} and ep["datagen"]["label"] == "demo"
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert [x["label"] for x in m["seeds"]["0"]["episodes"]] == ["demo"] and m["episodes"] == ["seed_0_a3"]
    assert G.record_attempt("open_fridge_door", 0, 3, tmp_path, jobs=st.jobs(), say=lambda m: None) == e     # resume: no replay
    assert [c for c in st.calls if c[0] == "record"] == [("record", 0, 3)]


def test_record_attempt_needs_a_finished_run(tmp_path):
    with pytest.raises(SystemExit):
        G.record_attempt("open_fridge_door", 0, 0, tmp_path, jobs=Stub().jobs())
    gen(tmp_path, Stub(planned=(0,)), record=False)
    with pytest.raises(SystemExit):
        G.record_attempt("open_fridge_door", 0, 4, tmp_path, jobs=Stub().jobs())
