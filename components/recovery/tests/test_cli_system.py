"""System tests: `recovery.cli.main` (what `./revolve recover` calls) in this process on a fake eval run, with the fake runner (the
real slot logic, a fake sim), the local scanner over fake bags and a fake generator: collect -> run -> check, the quota, the
unreproducible count, warm resets and their fallback, resume, workers, --dry-run and spec errors."""
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import fake_recovery as F
import pytest
import yaml
from datagen.core.spec import SpecError
from recovery import cli
from recovery.core import pipeline as P
from recovery.core.registry import REGISTRY
from recovery.core.robot import arm_of
from recovery.core.spec import load
from revolve_common import specs

REPO = Path(__file__).resolve().parents[3]
TRIES = [F.Try(seed=100 + i, failed="approach") for i in range(3)] + [F.Try(seed=110 + i, failed="grasp") for i in range(2)] + [
    F.Try(seed=120, failed="open"), F.Try(seed=130)]


@pytest.fixture
def run(tmp_path):
    return F.write_eval_run(tmp_path, TRIES)


def write_spec(tmp_path, eval_run, quota=None, runner=None, name="spec.yaml", **edits):
    raw = {"task": "open_fridge_door", "robot": "mobile_panda", "from": {"eval": str(eval_run)}, "out": "runs/recovery/t",
           "phases": {"generator": {"plugin": "fake_recovery", "attempts": 2}, "quota": quota or {"approach": 2, "grasp": 2}},
           "scanner": {"plugin": "local", "reader": "fake_recovery:read_messages"}, "runner": {"plugin": "fake", **(runner or {})}}
    path = tmp_path / name
    path.write_text(yaml.safe_dump({**raw, **edits}))
    return path


def recover(tmp_path, spec, *args, capsys=None):
    code = cli.main([str(spec), *args], root=tmp_path)
    return (code, *(capsys.readouterr() if capsys else ()))


def out_of(tmp_path):
    return tmp_path / "runs" / "recovery" / "t"


def state(tmp_path):
    return json.loads((out_of(tmp_path) / "state.json").read_text())


def manifest(tmp_path):
    return json.loads((out_of(tmp_path) / "manifest.json").read_text())


def test_all_stages(tmp_path, run, capsys):
    code, out, err = recover(tmp_path, write_spec(tmp_path, run), capsys=capsys)
    assert code == 0, err
    assert "collect: 6 failures (1 unreproducible) in 6 failed, 1 passed tries" in out
    assert out.count("run approach") == 2 and out.count("run grasp") == 2 and "check jobs: PASS" in out and "check quota: PASS" in out
    m = manifest(tmp_path)
    assert m["counts"] == {"jobs": 4, "scenes": 4, "attempted": 4, "passed": 4}
    assert sorted(e["recovery"]["phase"] for e in m["episodes"]) == ["approach", "approach", "grasp", "grasp"]
    assert {e["recovery"]["kind"] for e in m["episodes"]} == {"phase"} and all(e["passed"] and e["bag"] for e in m["episodes"])
    assert all((out_of(tmp_path) / e["result"]).is_file() and (out_of(tmp_path) / e["scene"]).is_file() for e in m["episodes"])
    report = (out_of(tmp_path) / "report.md").read_text()
    assert "| approach | 3 | 0 | 2 | 2 | 2 | 2 | 100% |" in report and "| grasp | 2 | 0 | 2 | 2 | 2 | 2 | 100% |" in report
    assert "| open | 1 | 1 | 0 | 0 | 0 | - | - |" in report and "fridge::top_door_hinge" in report


HELD = [F.Try(set="held", seed=500 + i, failed="approach") for i in range(2)]


def test_failures_of_held_out_sets_are_counted_and_never_mined(tmp_path, capsys):
    run = F.write_eval_run(tmp_path, [*HELD, *TRIES], heldout=("held",))
    code, out, err = recover(tmp_path, write_spec(tmp_path, run, quota={"approach": 3}), capsys=capsys)
    assert code == 0, err
    assert "collect: 6 failures (1 unreproducible) in 6 failed, 2 held-out, not mined, 1 passed tries" in out
    entries = manifest(tmp_path)["episodes"]
    assert len(entries) == 3 and {e["recovery"]["from_rollout"]["set"] for e in entries} == {"mined"}
    assert not any("held_" in f["id"] for f in state(tmp_path)["failures"])
    report = (out_of(tmp_path) / "report.md").read_text()
    assert "2 held-out, not mined" in report and "Mined sets: mined; kept unseen: held." in report
    assert "| approach | 3 | 0 | 3 | 3 | 3 | 3 | 100% |" in report


def test_from_sets_and_from_heldout_choose_the_sets_as_for_the_live_recorder(tmp_path, capsys):
    run = F.write_eval_run(tmp_path, [*HELD, *TRIES], heldout=("held",))
    code, out, err = recover(tmp_path, write_spec(tmp_path, run, quota={"approach": 5}, **{"from": {"eval": str(run), "sets": ["held"]}}),
                             capsys=capsys)                                      # the user names the held-out set to mine: theirs to choose
    assert code == 1 and "collect: 2 failures" in out and "held-out" not in out.split("collect:")[1].splitlines()[0]
    assert {f["set"] for f in state(tmp_path)["failures"]} == {"held"}
    both = write_spec(tmp_path, run, name="both.yaml", out="runs/recovery/both",
                      **{"from": {"eval": str(run), "sets": ["mined"], "heldout": ["mined"]}})
    code, _, err = recover(tmp_path, both, "collect", capsys=capsys)
    assert code == 2 and "also in from.heldout" in err


def test_a_run_with_only_held_out_sets_records_nothing_and_says_why(tmp_path, capsys):
    run = F.write_eval_run(tmp_path, HELD, heldout=("held",))
    code, out, err = recover(tmp_path, write_spec(tmp_path, run, quota={"approach": 1}), capsys=capsys)
    assert code == 1 and "collect: 0 failures (0 unreproducible) in 2 held-out, not mined tries" in out
    assert manifest(tmp_path)["episodes"] == [] and "check quota: FAIL" in out
    assert "Nothing is recorded: every set of the run is held-out" in (out_of(tmp_path) / "report.md").read_text()


def test_a_run_that_does_not_say_which_sets_are_held_out_is_refused(tmp_path, capsys):
    run = F.write_eval_run(tmp_path, TRIES)
    results = json.loads((run / "results.json").read_text())
    del results["leakage"]
    (run / "results.json").write_text(json.dumps(results))
    code, _, err = recover(tmp_path, write_spec(tmp_path, run), capsys=capsys)
    assert code == 2 and "will not guess which sets are held-out" in err


def test_a_mined_set_that_overlaps_a_held_out_pool_is_refused(tmp_path, capsys):
    run = F.write_eval_run(tmp_path, [*HELD, F.Try(seed=501, failed="approach")], heldout=("held",))      # mined seeds 501..501 inside held 500..501
    code, _, err = recover(tmp_path, write_spec(tmp_path, run), capsys=capsys)
    assert code == 2 and "set 'mined' (seeds [501, 501]) overlaps held-out set 'held'" in err


def test_the_jobs_scenes_start_at_the_restart_states(tmp_path, run, capsys):
    recover(tmp_path, write_spec(tmp_path, run), capsys=capsys)
    for e in manifest(tmp_path)["episodes"]:
        scene = yaml.safe_load((out_of(tmp_path) / e["scene"]).read_text())
        assert scene["robot_start"]["base"] == pytest.approx(list(e["recovery"]["restart_state"]["base"]), abs=1e-5)
        assert e["recovery"]["t_restart"] == {"approach": 11.0, "grasp": 10.0}[e["recovery"]["phase"]]      # a second before the failure
        assert specs.load(out_of(tmp_path) / e["scene"], assets=False).robot_start.arm == pytest.approx(
            list(e["recovery"]["restart_state"]["arm"]), abs=1e-5)


def test_quota_short_of_failures_fails_the_check_and_counts_unreproducible(tmp_path, run, capsys):
    spec = write_spec(tmp_path, run, quota={"approach": 5, "open": 1})
    code, out, _ = recover(tmp_path, spec, capsys=capsys)
    assert code == 1
    assert "check quota: FAIL episodes short of the quota: approach 3/5, open 0/1" in out
    report = (out_of(tmp_path) / "report.md").read_text()
    assert "| approach | 3 | 0 | 3 | 3 | 3 | 5 | 60% |" in report and "| open | 1 | 1 | 0 | 0 | 0 | 1 | 0% |" in report
    assert "| open: fridge::top_door_hinge = 1.400 is beyond 0.02 | 1 |" in report


def test_per_failure_records_several_episodes_per_restart(tmp_path, run, capsys):
    spec = write_spec(tmp_path, run, quota={"approach": 4}, phases={"generator": {"plugin": "fake_recovery", "attempts": 3},
                                                                    "quota": {"approach": 4}, "per_failure": 2})
    code, out, err = recover(tmp_path, spec, capsys=capsys)
    assert code == 0, err
    entries = manifest(tmp_path)["episodes"]
    assert len(entries) == 4 and len({e["scene"] for e in entries}) == 2                  # 2 failures x 2 episodes


ALL = ["mined_100_r0_t0", "mined_101_r0_t0", "mined_102_r0_t0"]


@pytest.mark.parametrize("keep, passed", [("all", [False, True]), ("successes", [True])])
def test_keep_decides_whether_the_failed_tries_stay_in_the_manifest(tmp_path, run, capsys, keep, passed):
    spec = write_spec(tmp_path, run, quota={"approach": 1}, keep=keep, runner={"pass_after": {k: 1 for k in ALL}})
    code, out, _ = recover(tmp_path, spec, capsys=capsys)
    assert code == 0 and "done (1/1 accepted in 2 tries)" in out
    assert [e["passed"] for e in manifest(tmp_path)["episodes"]] == passed


def test_a_job_that_never_passes_stays_short_and_fails_the_check(tmp_path, run, capsys):
    spec = write_spec(tmp_path, run, quota={"approach": 1}, runner={"fail_jobs": ALL})
    code, out, _ = recover(tmp_path, spec, capsys=capsys)
    assert code == 1 and "short (0/1 accepted in 2 tries)" in out and "check quota: FAIL episodes short of the quota: approach 0/1" in out
    assert manifest(tmp_path)["episodes"] == [] and "1 short" in (out_of(tmp_path) / "report.md").read_text()


def test_warm_reset_after_the_first_job_and_the_fallback_in_the_manifest(tmp_path, run, capsys):
    spec = write_spec(tmp_path, run, quota={"approach": 3}, run={"reset": "warm"}, runner={"miss_jobs": ["mined_101_r0_t0"]})
    code, out, err = recover(tmp_path, spec, capsys=capsys)
    assert code == 0, err
    modes = {e["scene"].split("/")[1]: e["recovery"]["reset"] for e in manifest(tmp_path)["episodes"]}
    assert modes["mined_100_r0_t0"] == {"mode": "relaunch"}                           # the first job launched the sim at its restart
    assert modes["mined_101_r0_t0"]["mode"] == "relaunch" and modes["mined_101_r0_t0"]["fallback"]["ok"] is False
    assert modes["mined_102_r0_t0"]["mode"] == "warm" and modes["mined_102_r0_t0"]["ok"] is True


def test_relaunch_mode_never_resets_warm(tmp_path, run, capsys):
    recover(tmp_path, write_spec(tmp_path, run, run={"reset": "relaunch"}), capsys=capsys)
    assert {json.dumps(e["recovery"]["reset"]) for e in manifest(tmp_path)["episodes"]} == {'{"mode": "relaunch"}'}


def test_rerun_resumes_and_runs_nothing_again(tmp_path, run, capsys):
    spec = write_spec(tmp_path, run)
    recover(tmp_path, spec, capsys=capsys)
    before, calls = manifest(tmp_path), len(F.CALLS)
    code, out, _ = recover(tmp_path, spec, capsys=capsys)
    assert code == 0 and "0 to run" in out and len(F.CALLS) == calls and manifest(tmp_path)["episodes"] == before["episodes"]


def test_a_changed_spec_refuses_the_existing_out(tmp_path, run, capsys):
    recover(tmp_path, write_spec(tmp_path, run), capsys=capsys)
    code, _, err = recover(tmp_path, write_spec(tmp_path, run, quota={"approach": 1}), capsys=capsys)
    assert code == 1 and "made by another spec" in err


def test_a_killed_run_resumes_from_the_committed_jobs(tmp_path, run):
    spec = load(write_spec(tmp_path, run))
    spec = dataclasses.replace(spec, out=str(tmp_path / "out"), eval_run=str(run))
    plugins = cli.build_plugins(spec, P.STAGES)

    class Killed(type(plugins.runner)):
        calls = 0

        def run(self, job, generator):
            Killed.calls += 1
            if Killed.calls == 3:
                raise KeyboardInterrupt("killed")
            super().run(job, generator)
    plugins.runner = Killed(spec, {})
    arm = arm_of(spec.robot, specs.REPO_SPECS)
    with pytest.raises(KeyboardInterrupt):
        P.Pipeline(spec, arm, plugins, Path(spec.out), log=lambda m: None).run()
    done = json.loads((Path(spec.out) / "state.json").read_text())["jobs"]
    assert len(done) == 2 and all(j["status"] == "done" for j in done.values())
    assert len(json.loads((Path(spec.out) / "manifest.json").read_text())["episodes"]) == 2          # the manifest is written on a kill too
    F.CALLS.clear()
    assert P.Pipeline(spec, arm, cli.build_plugins(spec, P.STAGES), Path(spec.out), log=lambda m: None).run()
    assert [c[0] for c in F.CALLS].count("drive") == 2 and len(json.loads((Path(spec.out) / "manifest.json").read_text())["episodes"]) == 4


def test_the_generator_plans_each_job_on_the_host_first_and_a_failed_plan_is_a_job_error(tmp_path, run, capsys):
    spec = write_spec(tmp_path, run, quota={"approach": 3}, phases={"generator": {"plugin": "fake_recovery", "bad_plans": [ALL[1]]},
                                                                    "quota": {"approach": 3}})
    F.CALLS.clear()
    code, out, _ = recover(tmp_path, spec, capsys=capsys)
    scenes = [c[1].parent.name for c in F.CALLS if c[0] == "plan"]
    assert code == 1 and sorted(scenes) == ALL and "run approach mined_101_r0_t0: error (plan: no plan)" in out
    assert [c[1].parent.name for c in F.CALLS if c[0] == "drive"].count(ALL[1]) == 0 and state(tmp_path)["jobs"][ALL[1]]["status"] == "error"
    F.CALLS.clear()
    recover(tmp_path, spec, capsys=capsys)
    assert [c[1].parent.name for c in F.CALLS if c[0] == "plan"] == [ALL[1]]         # the others are finished: not planned again


def test_workers_do_not_change_what_is_recorded(tmp_path, run, capsys):
    got = []
    for workers in (1, 3):
        spec = write_spec(tmp_path, run, name=f"w{workers}.yaml", out=f"runs/recovery/w{workers}", run={"workers": workers})
        assert recover(tmp_path, spec, capsys=capsys)[0] == 0
        m = json.loads((tmp_path / "runs" / "recovery" / f"w{workers}" / "manifest.json").read_text())
        got.append([(e["seed"], e["scene"], e["recovery"]["from_rollout"]["rollout"], e["passed"]) for e in m["episodes"]])
    assert got[0] == got[1] and len(got[0]) == 4


def test_collect_and_check_stages_alone(tmp_path, run, capsys):
    spec = write_spec(tmp_path, run)
    code, out, _ = recover(tmp_path, spec, "collect", capsys=capsys)
    assert code == 0 and "collect: 6 failures" in out and not (out_of(tmp_path) / "approach").exists()
    assert "approach" in state(tmp_path)["failures"][0]["phase"] and "selection" not in state(tmp_path)
    code, out, _ = recover(tmp_path, spec, "check", capsys=capsys)
    assert code == 1 and "check quota: FAIL" in out and "check jobs: PASS" in out
    assert recover(tmp_path, spec, "run", capsys=capsys)[0] == 0 and recover(tmp_path, spec, "check", capsys=capsys)[0] == 0


def test_gui_runs_one_worker(tmp_path, run, capsys):
    code, out, _ = recover(tmp_path, write_spec(tmp_path, run), "--gui", "--workers", "3", capsys=capsys)
    assert code == 0 and "run.headless false: one GUI window, so workers 3 -> 1" in out


def test_dry_run_prints_the_plan_and_writes_nothing(tmp_path, run, capsys):
    code, out, err = recover(tmp_path, write_spec(tmp_path, run, triggers=[{"plugin": "verdict"}]), "--dry-run", "--workers", "3", capsys=capsys)
    assert code == 0 and not (tmp_path / "runs").exists()
    assert out.count("[dry-run]") >= 5 and "quota (episodes): approach 2, grasp 2" in out and "workers: 3, reset relaunch" in out
    assert "restart: before_failure, 1.0 s before the failure" in out and "generator fake_recovery, runner fake" in out


@pytest.mark.parametrize("edit, message", [
    ({"phases": {"generator": {"plugin": "fake_recovery"}, "quota": {"nope": 1}}}, "phases.quota: unknown phase ['nope']"),
    ({"triggers": [{"plugin": "nope"}]}, "triggers[0].plugin: no trigger 'nope'"),
    ({"triggers": [{"plugin": "verdict", "phases": ["nope"]}]}, "unknown phase ['nope']"),
    ({"task": "nope"}, "nope"),
    ({"robot": "nope"}, "nope"),
    ({"keep": "some"}, "keep: expected one of"),
])
def test_invalid_specs_exit_2_with_the_yaml_path(tmp_path, run, capsys, edit, message):
    code, _, err = recover(tmp_path, write_spec(tmp_path, run, **edit), capsys=capsys)
    assert code == 2 and message in err and not (tmp_path / "runs").exists()


def test_a_missing_live_plugin_or_generator_is_an_error(tmp_path, run, capsys):
    code, _, err = recover(tmp_path, write_spec(tmp_path, run, live={"plugin": "nope"}), "--dry-run", capsys=capsys)
    assert code == 1 and "no live plugin 'nope'" in err
    code, _, err = recover(tmp_path, write_spec(tmp_path, run, phases={"generator": {"plugin": "nope"}, "quota": {"approach": 1}}),
                           capsys=capsys)
    assert code == 1 and "no generator plugin 'nope'" in err


def test_a_run_that_is_not_an_eval_run_is_an_error(tmp_path, capsys):
    code, _, err = recover(tmp_path, write_spec(tmp_path, tmp_path / "nothing"), capsys=capsys)
    assert code == 2 and "not an evaluation run" in err


@REGISTRY.register("live", "fake_live")
class FakeLive:
    """A live recorder for the registry test: `validate` rejects a spec whose live block lacks `needs`."""

    def __init__(self, spec, params):
        self.params = params

    def validate(self, spec):
        if "needs" not in self.params:
            raise SpecError("live.needs: missing")

    def driver(self, policy_drive, ctx):
        return policy_drive

    def finish(self, episode):
        return {}


def test_the_live_recorder_is_validated_through_the_registry(tmp_path, run, capsys):
    code, _, err = recover(tmp_path, write_spec(tmp_path, run, live={"plugin": "fake_live"}), "--dry-run", capsys=capsys)
    assert code == 2 and "live.needs: missing" in err
    assert recover(tmp_path, write_spec(tmp_path, run, live={"plugin": "fake_live", "needs": 1}), "--dry-run", capsys=capsys)[0] == 0


def test_the_repo_specs_validate_and_dry_run(tmp_path, capsys):
    for name in ("fridge", "smoke"):
        code = cli.main([str(REPO / "specs" / "recovery" / f"{name}.yaml"), "--dry-run"], root=tmp_path)
        assert code == 0, capsys.readouterr().err
    assert "[dry-run]" in capsys.readouterr().out


def test_revolve_has_a_recover_subcommand():
    code = (REPO / "revolve").read_text()
    assert "def cmd_recover" in code and '["recover"]' in code and '"recovery": REPO / "components" / "recovery" / "tests"' in code
    helps = subprocess.run([sys.executable, str(REPO / "revolve"), "--help"], capture_output=True, text=True)
    assert "revolve recover <spec.yaml>" in helps.stdout
