"""SimSession under the datagen interrupt pattern: a SIGINT / SIGTERM becomes `Interrupted` and the session is closed on
the way out (context manager, run_with_relaunch), the run's leftovers are removed by `stop_sims(prefix)`."""
import os
import signal
import threading

import pytest

from datagen.core.interrupt import Interrupted, interruptible
from revolve_common.sim_fake import FakeSession
from revolve_common.sim_session import run_with_relaunch, stop_sims


def fake(tmp_path):
    return FakeSession("kitchen_partnet", "mobile_panda", "open_fridge_door", tmp_path / "s", "r-0", 110)


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_close_runs_when_a_signal_interrupts_the_with_block(tmp_path, sig):
    s = fake(tmp_path)
    with pytest.raises(Interrupted):
        with interruptible(), s:
            s.start()
            os.kill(os.getpid(), sig)
            threading.Event().wait(2)
    assert not s.started and s.closed == 1


def test_close_runs_when_a_signal_interrupts_the_work(tmp_path):
    s = fake(tmp_path)

    def work(_):
        os.kill(os.getpid(), signal.SIGINT)
        threading.Event().wait(2)
    with pytest.raises(Interrupted):
        with interruptible():
            run_with_relaunch(s, work)
    assert not s.started and s.closed == 1


def test_stop_sims_of_the_run_after_an_interrupt():
    class D:
        removed = []
        names = lambda self: ["r-0", "r-1-client", "q-0"]           # noqa: E731
        rm = lambda self, n: self.removed.append(n)                  # noqa: E731
    d = D()
    assert stop_sims("r", d) == 2 and d.removed == ["r-0", "r-1-client"]
