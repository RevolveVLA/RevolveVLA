"""SIGINT / SIGTERM as an exception, so a run stops through its normal error path (cancel, stop the jobs, save the
manifest) instead of a traceback or a hard kill."""
from __future__ import annotations

import signal
import threading
from contextlib import contextmanager

from .errors import DatagenError


class Interrupted(DatagenError):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"interrupted by {signal.Signals(signum).name}: progress saved, rerun the same command to resume")

    @property
    def code(self) -> int:
        return 128 + self.signum


@contextmanager
def interruptible(signums=(signal.SIGINT, signal.SIGTERM)):
    """While active, the first of `signums` raises Interrupted in the main thread; later ones are ignored (the
    shutdown is under way). A no-op outside the main thread."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    fired = []

    def handler(signum, frame):
        if not fired:
            fired.append(signum)
            raise Interrupted(signum)
    old = {s: signal.signal(s, handler) for s in signums}
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)
