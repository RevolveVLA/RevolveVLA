"""SIGINT/SIGTERM as an exception (core/interrupt)."""
import os
import signal
import threading

import pytest

from datagen.core.interrupt import Interrupted, interruptible


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
