"""core/runner.py with the fake runner: slots, warm reuse of a sim, the relaunch fallback, faults, interrupts."""
import threading

import fake_recovery as F
import pytest
import yaml
from datagen.core import episodes
from recovery.core import jobs as J
from recovery.core import spec as S
from recovery.core.errors import JobError
from recovery.core.failures import Failure
from recovery.core.restart import Restart
from recovery.core.robot import Arm
from recovery.core.runner import RunnerStopped
from recovery.plugins.runners.fake import FakeRunner
from revolve_common import specs

ARM = Arm("panda_arm_controller", tuple(F.ARM))


def spec(tmp_path, reset="warm", attempts=2, per_failure=1):
    return S.parse({"task": "open_fridge_door", "robot": "mobile_panda", "from": {"eval": "x"}, "out": str(tmp_path / "out"),
                    "phases": {"generator": {"plugin": "fake_recovery", "attempts": attempts}, "quota": {"approach": 1},
                               "per_failure": per_failure}, "run": {"reset": reset}}, "r.yaml")


@pytest.fixture
def jobs(tmp_path):
    base = yaml.safe_load((specs.REPO_SPECS / "scenes" / "kitchen_partnet" / "scene.yaml").read_text())
    made = []
    for i in range(3):
        scene = tmp_path / "eval" / f"scene_{i}.yaml"
        scene.parent.mkdir(exist_ok=True)
        scene.write_text(yaml.safe_dump(base | {"robot_start": {"base": [-1.75, 1.1 + 0.1 * i, 1.5], "arm": "folded"}}))
        f = Failure(f"f{i}", "mined", f"{i}_r0", 0, str(scene), "approach", "verdict", "", 5.0, 0.0, "bag", "res",
                    Restart(4.0, (-1.7, 1.5 + 0.1 * i, 1.5), tuple([0.1] * 7), 0.0))
        made.append(J.make(i, f, tmp_path / "out", "mobile_panda", ARM))
    return made


def runner(tmp_path, **params):
    s = spec(tmp_path, params.pop("reset", "warm"), params.pop("attempts", 2), params.pop("per_failure", 1))
    return s, FakeRunner(s, params)


def run_all(r, s, jobs, **kw):
    g = F.FakeGenerator(None, {})
    for job in jobs:
        r.run(job, g)


def test_warm_jobs_share_one_sim_and_reset_to_each_restart(tmp_path, jobs):
    s, r = runner(tmp_path)
    run_all(r, s, jobs)
    (sim,) = r.sessions
    assert sim.launches == 1 and sim.scene == str(jobs[0].scene)                       # one launch, at the first job's start
    assert [t.base for t in sim.warm_targets] == [j.target.base for j in jobs[1:]]    # a warm reset to each later restart
    assert [len(episodes.read(j.dir)["tries"]) for j in jobs] == [1, 1, 1]
    assert [t["episode"] for j in jobs for t in episodes.read(j.dir)["tries"]] == [0, 1, 2]                # numbered on by the one sim
    assert len(r.slots) == 1 and r.slots[0].dir == tmp_path / "out" / "sims" / r.slots[0].name


def test_relaunch_mode_launches_a_sim_per_job_and_never_resets_to(tmp_path, jobs):
    s, r = runner(tmp_path, reset="relaunch")
    run_all(r, s, jobs)
    assert [x.scene for x in r.sessions] == [str(j.scene) for j in jobs] and all(x.warm_targets == [] for x in r.sessions)
    assert sum(x.closed for x in r.sessions) == 2                                         # each launch closed the one before it


def test_a_job_that_is_a_different_world_relaunches(tmp_path, jobs):
    s, r = runner(tmp_path)
    scene = yaml.safe_load(jobs[1].scene.read_text())
    scene["models"]["table_01"]["pose"][0] += 1.0
    jobs[1].scene.write_text(yaml.safe_dump(scene))
    run_all(r, s, jobs)
    assert [x.scene for x in r.sessions] == [str(j.scene) for j in jobs]                    # job 1 relaunches, and job 2 differs from it
    assert all(x.warm_targets == [] for x in r.sessions)


def test_a_missed_warm_reset_relaunches_at_the_restart_and_goes_on(tmp_path, jobs):
    s, r = runner(tmp_path, miss_jobs=["f1"])
    run_all(r, s, jobs)
    first, second = r.sessions
    assert first.scene == str(jobs[0].scene) and second.scene == str(jobs[1].scene)        # the miss relaunched the sim at job 1's restart
    assert [(x["attempt"], x["ok"]) for x in J.resets(jobs[1].dir)] == [(0, False)]
    assert [t.base for t in first.warm_targets] == [jobs[1].target.base] and [t.base for t in second.warm_targets] == [jobs[2].target.base]
    assert all(episodes.accepted(episodes.read(j.dir)) == 1 for j in jobs)                  # no job lost its episode
    assert J.resets(jobs[2].dir)[0]["ok"]                                                   # the next job reset warm again


def test_every_try_of_a_warm_job_resets_warm(tmp_path, jobs):
    s, r = runner(tmp_path, pass_after={"f0": 1}, attempts=3)
    r.run(jobs[0], F.FakeGenerator(None, {}))
    (sim,) = r.sessions
    assert [t["passed"] for t in episodes.read(jobs[0].dir)["tries"]] == [False, True] and len(sim.warm_targets) == 0   # the first job launched the sim: no reset
    r.run(jobs[1], F.FakeGenerator(None, {}))
    assert len(sim.warm_targets) == 1


def test_a_sim_fault_relaunches_and_is_logged_not_counted_as_a_try(tmp_path, jobs):
    s, r = runner(tmp_path, faults={"f0": {"wait_ready": ["ready_timeout"]}})
    r.run(jobs[0], F.FakeGenerator(None, {}))
    log = episodes.read(jobs[0].dir)
    assert len(log["tries"]) == 1 and log["faults"] == ["ready_timeout"] and len(r.sessions) == 2


def test_infrastructure_retries_run_out_as_a_job_error(tmp_path, jobs):
    s, r = runner(tmp_path, infra_retries=1, faults={"f0": {"wait_ready": ["ready_timeout", "ready_timeout"]}})
    with pytest.raises(JobError, match="infrastructure"):
        r.run(jobs[0], F.FakeGenerator(None, {}))
    assert episodes.read(jobs[0].dir)["faults"] == ["ready_timeout"]


def test_a_finished_job_needs_no_sim(tmp_path, jobs):
    s, r = runner(tmp_path)
    r.run(jobs[0], F.FakeGenerator(None, {}))
    r.close()
    r.run(jobs[0], F.FakeGenerator(None, {}))
    assert len(r.sessions) == 1


def test_a_failing_job_uses_its_attempts_and_stays_short(tmp_path, jobs):
    s, r = runner(tmp_path, fail_jobs=["f0"], attempts=3)
    r.run(jobs[0], F.FakeGenerator(None, {}))
    log = episodes.read(jobs[0].dir)
    assert [t["passed"] for t in log["tries"]] == [False] * 3 and not episodes.finished({"tries": []}, s.phases.episodes)


def test_stopped_runner_launches_nothing_and_close_closes_the_sims(tmp_path, jobs):
    s, r = runner(tmp_path)
    r.run(jobs[0], F.FakeGenerator(None, {}))
    (sim,) = r.sessions
    r.close()
    assert not sim.started and r.slots == []
    r.stop()
    with pytest.raises(RunnerStopped):
        r.run(jobs[1], F.FakeGenerator(None, {}))


def test_parallel_jobs_get_their_own_slots(tmp_path, jobs):
    s, r = runner(tmp_path)
    gate, errors = threading.Barrier(3, timeout=5), []

    class Gated(F.FakeGenerator):
        def drive(self, *a):
            gate.wait()                                   # all three jobs are driving at once: three sims
            return super().drive(*a)

    def go(job):
        try:
            r.run(job, Gated(None, {}))
        except Exception as e:                            # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=go, args=(j,)) for j in jobs]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len({sl.name for sl in r.slots}) == 3 and len(r.sessions) == 3
