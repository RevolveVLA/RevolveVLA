"""Stopping a run's docker jobs (containers), SIGINT/SIGTERM as an exception
(core/interrupt): host side, with docker and the launchers faked."""
import os
import signal
import sys
import threading
import types

import pytest

from datagen import containers
from datagen.core.interrupt import Interrupted, interruptible

REPO = containers.REPO


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
