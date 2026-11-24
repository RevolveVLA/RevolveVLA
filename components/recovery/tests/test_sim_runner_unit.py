"""plugins/runners/sim.py (the real runner) and client.py (its container entry) with docker and ROS replaced: the launch and client
commands, the exit codes (a warm reset that missed, faults, timeouts), and the entry's own main() on the fake sim."""
import json
import subprocess

import fake_recovery as F
import pytest
from datagen.core import episodes
from recovery import client as CL
from recovery.core import jobs as J
from recovery.core.errors import JobError
from recovery.core.runner import RunnerStopped
from recovery.plugins.runners import fake as FK
from recovery.plugins.runners import sim as SIM
from revolve_common.sim_session import DomainPool, SimFault
from test_runner_unit import jobs, spec  # noqa: F401  (fixtures + the spec helper)


class Container:
    """Stands for ClientContainer: scripted exit codes, else runs `recovery.client` in this process on the session's fake sim."""

    def __init__(self, session, command, kw):
        self.s, self.command, self.kw = session, command, kw

    def run(self, log, timeout=None, watch=None, **_):
        self.s.calls.append({"command": self.command, "kw": self.kw, "log": log, "timeout": timeout, "watch": watch})
        if self.s.script:
            step = self.s.script.pop(0)
            if isinstance(step, Exception):
                raise step
            return step
        return self.s.run_client(self.command)


class Sess(FK.WarmFakeSession):
    calls, script, stopped, patch = [], [], None, None

    def __init__(self, *a, **kw):
        kw.pop("log", None), kw.pop("gui", None)
        super().__init__(*a, **kw)

    def start(self):
        if self.stopped.is_set():
            raise RunnerStopped("run interrupted")
        super().start()

    def client(self, command, **kw):
        return Container(self, command, kw)

    def run_client(self, command):
        self.patch.setattr(CL, "run_client", lambda fn, out: fn(self.connect()))
        self.patch.setattr(CL, "warm_resetter", lambda client: client.reset_to)       # the real one needs revolve_common.warm_reset
        CL.main(command[3:])
        return 0


@pytest.fixture
def runner(tmp_path, monkeypatch):
    Sess.calls, Sess.script, Sess.patch = [], [], monkeypatch
    s = spec(tmp_path, attempts=2)
    r = SIM.SimRunner(s, {})
    r.domain_pool = lambda: DomainPool(r.prefix, docker=FK._NoDocker())
    r.pool = r.domain_pool()
    monkeypatch.setattr(r, "session", lambda slot, scene: _session(r, slot, scene))
    return r


def _session(r, slot, scene):
    s = Sess(scene, r.spec.robot, r.spec.task, slot.dir, slot.name, slot.domain, "no-images")
    s.stopped = r.stopped
    return s


def run(r, job):
    r.run(job, None)


def test_the_client_command_and_its_container(runner, jobs):
    run(runner, jobs[0])
    (call,) = Sess.calls
    c = call["command"]
    assert c[:3] == ["python3", "-m", "recovery.client"] and json.loads(c[c.index("--spec") + 1]) == runner.spec.raw
    assert J.Job.from_json(c[c.index("--job") + 1]) == jobs[0] and c[c.index("--sim-out") + 1] == str(runner.slots[0].dir)
    assert "--warm" not in c                                              # the first job launched the sim at its start
    assert call["kw"] == {"pythonpath": [SIM.DATAGEN_SRC, SIM.RECOVERY_SRC], "mounts": [jobs[0].dir]}
    assert call["timeout"] == 2 * 600 + SIM.SETUP_S and str(call["log"]).endswith(".client.log") and callable(call["watch"])
    assert episodes.accepted(episodes.read(jobs[0].dir)) == 1


def test_the_next_job_on_the_sim_is_warm(runner, jobs):
    run(runner, jobs[0])
    run(runner, jobs[1])
    assert "--warm" not in Sess.calls[0]["command"] and "--warm" in Sess.calls[1]["command"]
    assert [r["mode"] for r in J.resets(jobs[1].dir)] == ["warm"] and len(runner.slots) == 1


def test_exit_79_relaunches_the_sim_at_the_restart(runner, jobs):
    run(runner, jobs[0])
    Sess.script[:] = [J.MISS_EXIT]
    run(runner, jobs[1])
    warm = ["--warm" in c["command"] for c in Sess.calls]
    assert warm == [False, True, False] and len(runner.slots) == 1       # the miss, then the same job again on a relaunched sim
    assert episodes.accepted(episodes.read(jobs[1].dir)) == 1


def test_a_fault_exit_relaunches(runner, jobs):
    Sess.script[:] = [SimFault("reset_failed")]
    run(runner, jobs[0])
    assert len(Sess.calls) == 2 and episodes.read(jobs[0].dir)["faults"] == ["reset_failed"]


@pytest.mark.parametrize("step, message", [(1, "exited with 1"), (subprocess.TimeoutExpired("c", 1), "ran over its wall budget")])
def test_a_client_that_fails_or_hangs_is_a_job_error(runner, jobs, step, message):
    Sess.script[:] = [step]
    with pytest.raises(JobError, match=message):
        run(runner, jobs[0])


def test_stopped_before_the_client_returns(runner, jobs):
    Sess.script[:] = [0]
    runner.stop()
    with pytest.raises(RunnerStopped):
        run(runner, jobs[0])


def test_stop_removes_the_runs_containers(runner, monkeypatch):
    removed = []
    monkeypatch.setattr(SIM, "stop_sims", lambda prefix: removed.append(prefix))
    runner.stop()
    assert removed == [runner.prefix] and runner.prefix.startswith("rc-")


def test_the_entry_runs_the_job_with_the_spec_generator(runner, jobs):
    F.CALLS.clear()
    run(runner, jobs[0])
    assert F.CALLS == [("drive", jobs[0].scene)]
