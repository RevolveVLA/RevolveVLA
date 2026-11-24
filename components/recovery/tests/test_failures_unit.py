"""core/failures.py: the failures of an eval run: passed tries, errors, missing files, triggers over the bag, restart states."""
import json
from pathlib import Path

import fake_recovery as F
import pytest
from datagen.core.spec import SpecError
from recovery.core import failures as FL
from recovery.core import spec as S
from recovery.core.errors import RecoveryError
from recovery.core.robot import arm_of
from recovery.core.scan import scan_run, supervisor_of
from revolve_common import specs


def spec(triggers=None, **phases):
    raw = {"task": "open_fridge_door", "robot": "mobile_panda", "from": {"eval": "x"}, "out": "o",
           "phases": {"generator": {"plugin": "fake_recovery"}, "quota": {"approach": 1}, **phases}}
    return S.parse({**raw, **({"triggers": triggers} if triggers else {})}, "r.yaml")


def scan(tmp_path, tries, **kw):
    run = F.write_eval_run(tmp_path, tries)
    return scan_run(spec(**kw), run, F.read_messages)


def test_a_failed_try_is_a_failure_at_the_verdict_and_a_passed_one_is_not(tmp_path):
    s = scan(tmp_path, [F.Try(seed=100), F.Try(seed=101, failed="grasp")])
    assert dict(s.counts) == {"passed": 1, "failed": 1}
    (f,) = s.failures
    assert (f.id, f.set, f.rollout, f.attempt, f.phase, f.trigger) == ("heldout_101_r0_t0", "heldout", "101_r0", 0, "grasp", "verdict")
    assert f.reason == "grasp:timeout_s" and f.t_fail == 11.0 and f.phase_t_start == 6.0
    assert Path(f.scene).name == "scene.yaml" and Path(f.bag).is_dir() and Path(f.result).is_file()
    assert f.restart.reproducible and f.restart.t == 10.0                            # before_failure, back_s 1


def test_restart_from_phase_start(tmp_path):
    (f,) = scan(tmp_path, [F.Try(failed="grasp")], restart_from="phase_start").failures
    assert f.restart.t == 6.0 and f.phase_t_start == 6.0


def test_every_try_of_a_rollout_is_scanned(tmp_path):
    s = scan(tmp_path, [F.Try(failed="approach"), F.Try(failed="open"), F.Try()])        # retries of one rollout
    assert [(f.id, f.phase) for f in s.failures] == [("heldout_100_r0_t0", "approach"), ("heldout_100_r0_t1", "open")]
    assert not s.failures[1].restart.reproducible and "fridge::top_door_hinge" in s.failures[1].restart.reason


def test_stall_trigger_moves_the_failure_earlier(tmp_path):
    triggers = [{"plugin": "stall", "phases": ["approach"], "window_s": 3, "min_motion": 0.01}, {"plugin": "verdict"}]
    s = scan(tmp_path, [F.Try(failed="approach", stall=True), F.Try(seed=101, failed="approach", t_fail=8.0)], triggers=triggers)
    stalled, moving = s.failures
    assert (stalled.trigger, stalled.t_fail) == ("stall", 3.0) and stalled.restart.t == 2.0       # a stall at 3 s, not the timeout at 12 s
    assert (moving.trigger, moving.t_fail) == ("verdict", 8.0)


def test_failures_no_trigger_names_are_counted_not_dropped(tmp_path):
    s = scan(tmp_path, [F.Try(failed="grasp")], triggers=[{"plugin": "stall", "phases": ["approach"], "window_s": 3, "min_motion": 0.01}])
    assert not s.failures and dict(s.counts) == {"failed": 1, "untriggered": 1}


def test_tries_without_a_verdict_or_files_are_counted(tmp_path):
    run = F.write_eval_run(tmp_path, [F.Try(failed="grasp"), F.Try(seed=101, failed="grasp"), F.Try(seed=102, failed="grasp")])
    results = json.loads((run / "results.json").read_text())
    rolls = results["sets"]["heldout"]["rollouts"]
    rolls["101_r0"]["tries"][0]["record"]["error"] = "wall limit 600 s"
    Path(rolls["102_r0"]["tries"][0]["record"]["result_file"]).unlink()
    (run / "results.json").write_text(json.dumps(results))
    s = scan_run(spec(), run, F.read_messages)
    assert dict(s.counts) == {"failed": 1, "errors": 1, "no_result": 1} and len(s.failures) == 1


def test_results_of_another_task_are_refused(tmp_path):
    run = F.write_eval_run(tmp_path, [F.Try(failed="grasp")])
    result = next(run.glob("heldout/*_sim/episode_0.json"))
    result.write_text(json.dumps({**json.loads(result.read_text()), "task": "open_drawer"}))
    with pytest.raises(RecoveryError, match="an episode of open_drawer"):
        scan_run(spec(), run, F.read_messages)


def test_not_an_eval_run(tmp_path):
    with pytest.raises(RecoveryError, match="not an evaluation run"):
        scan_run(spec(), tmp_path, F.read_messages)


def test_scan_json_round_trip(tmp_path):
    s = scan(tmp_path, [F.Try(failed="grasp"), F.Try(seed=101, failed="open"), F.Try(seed=102)])
    again = FL.Scan.from_dict(json.loads(json.dumps(s.to_dict())))
    assert again.failures == s.failures and again.counts == s.counts


def test_the_supervisor_is_built_against_the_task_and_the_robot_arm():
    task, arm = specs.resolve(specs.REPO_SPECS, "open_fridge_door", "mobile_panda", assets=False), arm_of("mobile_panda", specs.REPO_SPECS)
    supervisor_of(spec(), task, arm)
    with pytest.raises(SpecError, match=r"triggers\[0\].phases: unknown phase \['nope'\]"):
        supervisor_of(spec(triggers=[{"plugin": "verdict", "phases": ["nope"]}]), task, arm)
    with pytest.raises(SpecError, match=r"quota: unknown phase \['nope'\]"):
        supervisor_of(spec(quota={"nope": 1}), task, arm)
