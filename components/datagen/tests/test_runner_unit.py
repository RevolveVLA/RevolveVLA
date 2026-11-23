"""plugins/runners/sim.py (the real runner) with the docker and ROS parts replaced: the launch argv, the client container, the
in-container entry (datagen.scene_client), relaunch, give-up and interrupt behaviour; plugins/runners/fake.py's knobs."""
import json
import subprocess
import threading

import pytest

import fake_datagen as F          # noqa: F401  (registers the fake generator the client container would import)
from datagen import scene_client
from datagen.core import episodes, registry, spec as S
from datagen.core.episodes import SceneError
from datagen.plugins.runners import sim as SIM
from datagen.plugins.runners.session_runner import RunnerStopped
from revolve_common.sim_fake import FakeSession
from revolve_common.sim_session import SimFault

RAW = {"task": "t", "robot": "r", "out": "unused", "generator": {"plugin": "fake_datagen", "attempts": 2},
       "splits": {"train": {"target": 1, "pool": {"initial": [0, 3]}}}}


class FakeClient:
    """Stands for ClientContainer: runs `datagen.scene_client` in this process on the session's FakeSession."""

    def __init__(self, session, command, kw):
        self.s, self.command, self.kw, self.runs = session, command, kw, []

    def run(self, log, timeout=None, watch=None, **_):
        self.s.calls.append({"command": self.command, "kw": self.kw, "log": log, "timeout": timeout})
        if self.s.client_script:
            return self.s.client_script.pop(0)(self.s)
        return self.s.run_client(self.command)


class Sess(FakeSession):
    """A FakeSession that also answers `client()` like SimSession, and runs scene_client's real main()."""
    calls, client_script = [], []
    stopped = None

    def __init__(self, *a, **kw):
        kw.pop("log", None)
        super().__init__(*a, **kw)

    def start(self):
        if self.stopped.is_set():
            raise RunnerStopped("run interrupted")
        super().start()

    def client(self, command, **kw):
        return FakeClient(self, command, kw)

    def run_client(self, command):
        scene_client.run_client = lambda fn, out, *args: fn(self.connect(), *args)
        scene_client.main(command[3:])
        return 0


@pytest.fixture
def runner(tmp_path, monkeypatch):
    Sess.calls, Sess.client_script = [], []
    monkeypatch.setattr(SIM, "_Session", Sess)
    monkeypatch.setattr(SIM, "stop_sims", lambda prefix: stopped.append(prefix))
    monkeypatch.setattr("revolve_common.sim_session.DomainPool.lease", lambda self: (f"{self.prefix}-0", 110))
    stopped = []
    spec = S.parse(RAW | {"out": str(tmp_path / "run")})
    r = SIM.SimRunner(spec, {})
    r.stopped_calls = stopped
    return r


def instance(tmp_path, seed=0):
    d = tmp_path / "run" / "train" / f"scene_{seed}"
    d.mkdir(parents=True)
    (d / "scene.yaml").write_text("name: x\n")
    return d / "scene.yaml"


def gen(runner):
    return registry.create("generator", "fake_datagen", runner.spec, {})


def test_the_real_session_is_launched_for_the_instance_with_the_contract_flags(tmp_path):
    spec = S.parse(RAW | {"out": str(tmp_path / "run"), "generator": {"plugin": "g", "record": "no-images"}})
    r = SIM.SimRunner(spec, {})
    path = instance(tmp_path, 7)
    argv = r.session(7, path, "dg-0", 120).argv()
    assert argv[1:] == ["sim", "--scene", str(path), "--robot", "r", "--task", "t", "--no-autostart", "--record", "no-images",
                        "--out", str(path.parent), "--name", "dg-0", "--domain-id", "120"]
    assert str(r.session(7, path, "dg-0", 120).log) == str(tmp_path / "run" / "logs" / "scene_7.sim.log")
    assert r.prefix.startswith("dg-") and len(r.prefix) == 9


def test_the_client_container_runs_the_episode_loop_of_the_scene_with_the_specs_generator(tmp_path, runner):
    path = instance(tmp_path)
    runner.run(0, path, gen(runner))
    call = Sess.calls[0]
    assert call["command"][:3] == ["python3", "-m", "datagen.scene_client"] and call["kw"] == {"pythonpath": [SIM.DATAGEN_SRC]}
    assert json.loads(call["command"][call["command"].index("--spec") + 1])["task"] == "t"
    assert call["log"] == f"{tmp_path}/run/logs/scene_0.client.log" and call["timeout"] == 2 * 600 + SIM.SETUP_S
    log = episodes.read(path.parent)
    assert [t["passed"] for t in log["tries"]] == [True] and log["seed"] == 0 and log["faults"] == []
    runner.run(0, path, gen(runner))                      # finished: no second sim
    assert len(Sess.calls) == 1


def test_a_sim_fault_relaunches_and_is_counted(tmp_path, runner):
    path = instance(tmp_path)

    def fault(s):
        raise SimFault("clock_fault", "x")
    Sess.client_script = [fault]
    runner.run(0, path, gen(runner))
    assert len(Sess.calls) == 2 and episodes.read(path.parent)["faults"] == ["clock_fault"]
    assert len(episodes.read(path.parent)["tries"]) == 1


def test_a_generator_crash_a_timeout_and_used_up_retries_are_scene_errors(tmp_path, runner):
    path = instance(tmp_path)
    Sess.client_script = [lambda s: 1]
    with pytest.raises(SceneError, match="exited with 1; see .*scene_0.client.log"):
        runner.run(0, path, gen(runner))

    def slow(s):
        raise subprocess.TimeoutExpired("x", 1)
    Sess.client_script = [slow]
    with pytest.raises(SceneError, match="wall budget"):
        runner.run(0, path, gen(runner))

    def fault(s):
        raise SimFault("reset_failed", "x")
    Sess.client_script = [fault] * 4
    with pytest.raises(SceneError, match="infrastructure: .*3 infrastructure retries used"):
        runner.run(0, path, gen(runner))
    assert episodes.read(path.parent)["faults"] == ["reset_failed"] * 3


def test_stop_removes_the_runs_containers_and_no_sim_is_launched_afterwards(tmp_path, runner):
    path = instance(tmp_path)
    runner.stop()
    assert runner.stopped_calls == [runner.prefix] and runner.stopped.is_set()
    with pytest.raises(RunnerStopped):
        runner.run(0, path, gen(runner))
    assert Sess.calls == []


def test_the_real_session_refuses_to_start_once_stopped(tmp_path):
    s = SIM._Session("x", "r", "t", tmp_path / "o", "n", 1, log=tmp_path / "l.log")
    s.stopped = threading.Event()
    s.stopped.set()
    with pytest.raises(RunnerStopped):
        s.start()
    assert not (tmp_path / "l.log").exists()


def test_the_container_entry_rejects_what_the_spec_cannot_build(tmp_path, monkeypatch):
    monkeypatch.setattr(scene_client, "run_client", lambda fn, out, *a: fn(None, *a))
    with pytest.raises(S.SpecError):
        scene_client.main(["--spec", json.dumps({"task": "t"}), "--seed", "0", "--instance", str(tmp_path / "scene.yaml")])
