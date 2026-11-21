"""An `out` outside the repo (hostmounts), stopping a run's docker jobs (containers), SIGINT/SIGTERM as an exception
(core/interrupt): host side, with docker and the launchers faked."""
import os
import signal
import sys
import threading
import types

import pytest

from datagen import containers, hostmounts
from datagen.core.interrupt import Interrupted, interruptible

REPO = hostmounts.REPO


@pytest.fixture(autouse=True)
def clean():
    hostmounts.clear()
    yield
    hostmounts.clear()


def test_only_paths_outside_runs_are_recorded(tmp_path):
    hostmounts.add(REPO / "runs" / "datagen" / "x")
    assert hostmounts.MOUNTS == []
    hostmounts.add(tmp_path / "ssd" / "x")
    hostmounts.add(tmp_path / "ssd" / "x")
    assert hostmounts.MOUNTS == [(tmp_path / "ssd" / "x").resolve()]


def test_a_symlinked_out_is_recorded_as_its_target(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    hostmounts.add(link)
    assert hostmounts.MOUNTS == [real.resolve()]


def test_nothing_is_patched_without_a_recorded_path(monkeypatch):
    from docker.models.containers import ContainerCollection
    run = ContainerCollection.run
    launcher = types.SimpleNamespace(cpath=lambda p: "orig")
    monkeypatch.setitem(sys.modules, "launch", launcher)
    hostmounts.add(REPO / "runs" / "datagen" / "x")
    hostmounts.apply()
    assert ContainerCollection.run is run and launcher.cpath("/a") == "orig"


def test_a_recorded_path_is_mounted_into_every_job_and_kept_by_cpath(tmp_path, monkeypatch):
    from docker.models.containers import ContainerCollection
    started = []
    monkeypatch.setattr(ContainerCollection, "run", lambda self, image, command=None, **kw: started.append(kw) or "c")

    def orig(p):
        from pathlib import Path
        return "/repo/" + str(Path(p).resolve().relative_to(REPO))
    launchers = {n: types.SimpleNamespace(cpath=orig) for n in ("launch", "sim_launch", "dataset_launch")}
    for n, m in launchers.items():
        monkeypatch.setitem(sys.modules, n, m)
    out = tmp_path / "ssd" / "run"                      # does not exist yet: docker would create it as root
    hostmounts.add(out)
    hostmounts.apply()
    hostmounts.apply()                                  # idempotent
    assert out.is_dir()
    inside, outside = REPO / "runs" / "a" / "job", out / "jobs" / "plan_seed_1"
    for m in launchers.values():
        assert m.cpath(inside) == "/repo/runs/a/job" and m.cpath(outside) == str(outside)
        with pytest.raises(ValueError):
            m.cpath(tmp_path / "other")                 # not recorded: the launcher's own error
    assert ContainerCollection(None).run("img", ["x"], volumes={"/a": {"bind": "/b", "mode": "ro"}}) == "c"
    assert started == [{"volumes": {"/a": {"bind": "/b", "mode": "ro"}, str(out): {"bind": str(out), "mode": "rw"}}}]
    ContainerCollection(None).run("img", ["x"])         # a call without volumes is left alone
    assert started[1] == {}


def test_load_applies_the_mounts_before_the_first_job(tmp_path, monkeypatch):
    from datagen.plugins import generate_module
    launch = types.SimpleNamespace(cpath=lambda p: "orig")
    monkeypatch.setitem(sys.modules, "launch", launch)
    monkeypatch.setenv("REVOLVE_ROBOT", "mobile_panda")
    monkeypatch.setattr(hostmounts, "apply", lambda: seen.append(1))
    seen = []
    generate_module.load("mobile_panda", "open_fridge_door")
    assert seen == []                                   # nothing recorded: untouched
    hostmounts.add(tmp_path / "ssd")
    generate_module.load("mobile_panda", "open_fridge_door")
    assert seen == [1]


class FakeContainer:
    def __init__(self, cmd):
        self.attrs, self.killed = {"Config": {"Cmd": cmd}}, False

    def kill(self):
        self.killed = True


def fake_docker(monkeypatch, containers_list, error=None):
    def from_env():
        if error:
            raise error
        return types.SimpleNamespace(containers=types.SimpleNamespace(
            list=lambda filters: containers_list if filters == {"label": "revolve.component"} else []))
    monkeypatch.setitem(sys.modules, "docker", types.SimpleNamespace(from_env=from_env))


def test_stop_jobs_kills_only_the_containers_of_that_run(tmp_path, monkeypatch):
    out, inrepo = tmp_path / "ssd" / "run", REPO / "runs" / "datagen" / "mine"
    mine = [FakeContainer(["python", "-m", "w", "--job-dir", str(out / "jobs" / "plan_seed_1")]),
            FakeContainer(["bash", "-c", "s", "bash", "--job-dir", str(out / "jobs" / "dataset_convert")]),
            FakeContainer(["python", "--job-dir", "/repo/runs/datagen/mine/jobs/x"])]
    other = [FakeContainer(["python", "--job-dir", str(tmp_path / "ssd" / "run2" / "jobs" / "x")]),
             FakeContainer(["python", "--job-dir", "/repo/runs/datagen/mine2/jobs/x"]), FakeContainer(["sleep"]),
             FakeContainer(None)]
    fake_docker(monkeypatch, mine + other)
    assert containers.stop_jobs(out) == 2 and [c.killed for c in mine] == [True, True, False]
    assert containers.stop_jobs(inrepo) == 1 and mine[2].killed
    assert not any(c.killed for c in other)


def test_stop_jobs_without_a_docker_daemon_stops_nothing_and_says_so(monkeypatch):
    fake_docker(monkeypatch, [], error=RuntimeError("no daemon"))
    logs = []
    assert containers.stop_jobs(REPO / "runs" / "x", log=logs.append) == 0 and "no daemon" in logs[0]


# ------------------------------------------------------------------ interrupt
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_a_signal_becomes_interrupted_once_and_the_handlers_come_back(sig):
    before = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    with pytest.raises(Interrupted) as e:
        with interruptible():
            os.kill(os.getpid(), sig)
            threading.Event().wait(2)                       # the handler runs in this (main) thread
    assert e.value.code == 128 + sig and signal.Signals(sig).name in str(e.value) and "resume" in str(e.value)
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before


def test_a_second_signal_while_stopping_is_ignored():
    with interruptible():
        with pytest.raises(Interrupted):
            os.kill(os.getpid(), signal.SIGINT)
            threading.Event().wait(2)
        os.kill(os.getpid(), signal.SIGINT)                 # shutting down already: no second exception
        threading.Event().wait(0.1)
