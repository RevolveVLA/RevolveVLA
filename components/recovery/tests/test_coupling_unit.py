"""core/coupling.py + the pipeline's coupled seams (absorb, todo, record) on fake runs (CPU)."""
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import fake_recovery as F
import pytest
from datagen.core.spec import SpecError
from recovery.core import coupling, pipeline as P
from recovery.core import spec as S
from recovery.core.failures import Failure, Scan
from recovery.core.restart import Restart
from recovery.core.robot import arm_of
from recovery.plugins.runners.fake import FakeRunner
from revolve_common import specs

RAW = {"task": "open_fridge_door", "robot": "mobile_panda", "from": {"eval": "unused"}, "out": "unused",
       "phases": {"generator": {"plugin": "fake_recovery"}, "quota": {"approach": 2}, "per_failure": 1},
       "scanner": {"plugin": "local", "reader": "fake_recovery:read_messages"}, "runner": {"plugin": "fake"}}
EVAL = SimpleNamespace(train=SimpleNamespace(task="open_fridge_door", robot="mobile_panda"), run=SimpleNamespace(workers=None, reset="warm", headless=True))


def failure(n, phase="approach", ok=True):
    return Failure(f"s_{n}_r0_t0", "s", f"{n}_r0", 0, str(Path("scene.yaml")), phase, "verdict", "", 10.0, 0.0, "bag", "res",
                   Restart(9.0, (0.0, 0.0, 0.0), (0.1,) * 7, 0.0, None if ok else "door"))


def pipeline(tmp_path):
    spec = coupling.coupled_spec(S.parse(RAW, "r.yaml"), EVAL, tmp_path / "eval")
    return P.Pipeline(spec, arm_of(spec.robot, specs.REPO_SPECS), P.PluginSet(), tmp_path / "eval" / "recovery", log=lambda m: None)


def test_the_coupled_spec_reads_the_evaluations_out_and_runs_as_it_does(tmp_path):
    spec = coupling.coupled_spec(S.parse(RAW, "r.yaml"), EVAL, tmp_path / "eval")
    assert (spec.eval_run, spec.out) == (str(tmp_path / "eval"), str(tmp_path / "eval" / "recovery"))
    assert dataclasses.astuple(spec.run) == (1, "warm", True) and not spec.eval_spec
    gui = coupling.coupled_spec(S.parse(RAW, "r.yaml"), SimpleNamespace(**{**vars(EVAL), "run": SimpleNamespace(workers=4, reset="relaunch", headless=False)}),
                                tmp_path)
    assert dataclasses.astuple(gui.run) == (1, "relaunch", False)
    with pytest.raises(SpecError, match="the spec is for open_fridge_door / mobile_panda, the evaluation for open_drawer / x"):
        coupling.coupled_spec(S.parse(RAW, "r.yaml"), SimpleNamespace(train=SimpleNamespace(task="open_drawer", robot="x")), tmp_path)


def test_absorb_keeps_the_failures_picks_jobs_as_they_come_and_numbers_them_on(tmp_path):
    p = pipeline(tmp_path)
    first = p.absorb(Scan([failure(1), failure(2, "grasp"), failure(3, ok=False)], {"failed": 3}))
    assert [(i, f.id) for i, f in first] == [(0, "s_1_r0_t0")]                  # grasp is not in the quota, 3 cannot be reproduced
    second = p.absorb(Scan([failure(4), failure(5)], {"failed": 2, "passed": 1}))
    assert [(i, f.id) for i, f in second] == [(1, "s_4_r0_t0")]                 # the quota's 2 jobs are taken
    d = p.state.data
    assert d["selection"] == ["s_1_r0_t0", "s_4_r0_t0"] and len(d["failures"]) == 5 and d["scan"] == {"failed": 5, "passed": 1}
    assert [(i, f.id) for i, f in p.todo()] == [(0, "s_1_r0_t0"), (1, "s_4_r0_t0")]
    p.record(failure(1), {"status": "done", "tries": 1, "accepted": 1})
    p.record(failure(4), {"status": "error", "reason": "x"})
    assert [(i, f.id) for i, f in p.todo()] == [(1, "s_4_r0_t0")]               # an error is run again, a finished job is not
    assert json.loads((tmp_path / "eval" / "recovery" / "state.json").read_text())["selection"] == d["selection"]


def test_a_later_run_continues_the_selection_it_found(tmp_path):
    p = pipeline(tmp_path)
    p.absorb(Scan([failure(1)], {"failed": 1}))
    again = pipeline(tmp_path)                                                   # a new process over the same state
    assert [f.id for _, f in again.todo()] == ["s_1_r0_t0"]
    assert [(i, f.id) for i, f in again.absorb(Scan([failure(2), failure(3)], {"failed": 2}))] == [(1, "s_2_r0_t0")]


def test_after_scans_only_the_new_rollouts_and_hands_each_job_out_once(tmp_path):
    p = pipeline(tmp_path)
    scans = []

    class Scanner:
        def scan(self, out, only):
            scans.append(sorted(only))
            return Scan([failure(int(k.split("_")[0])) for _, k in sorted(only)], {"failed": len(only)})
    p.plugins = P.PluginSet(scanner=Scanner(), runner=FakeRunner(p.spec, {}))
    c = coupling.Coupled(p, tmp_path / "eval", log=lambda m: None)
    tasks = c.after("s", {"1_r0": {}, "2_r0": {}, "3_r0": {}})
    assert scans == [[("s", "1_r0"), ("s", "2_r0"), ("s", "3_r0")]] and len(tasks) == 2            # the quota takes the first 2
    assert c.after("s", {"1_r0": {}, "2_r0": {}, "4_r0": {}}) == [] and scans[-1] == [("s", "4_r0")]       # 1 and 2 were scanned; no job left
    assert pipeline(tmp_path).state.data["scanned"] == ["s/1_r0", "s/2_r0", "s/3_r0", "s/4_r0"]
    assert c.after("s", {"1_r0": {}}) == [] and len(scans) == 2                                    # nothing new: no scan
    fresh = coupling.Coupled(pipeline(tmp_path), tmp_path / "eval", log=lambda m: None)             # a resumed process: unfinished jobs again
    fresh.pipeline.plugins = p.plugins
    assert len(fresh.after("s", {"1_r0": {}})) == 2 and len(scans) == 2


def test_a_spec_without_phases_has_no_follow_up_tasks(tmp_path):
    p = pipeline(tmp_path)
    p.spec = dataclasses.replace(p.spec, phases=None)
    assert coupling.Coupled(p, tmp_path, log=lambda m: None).after("s", {"1_r0": {}}) == []
    assert coupling.Coupled(p, tmp_path, log=lambda m: None).check() is True
