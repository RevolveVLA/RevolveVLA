"""The worker slots of the rollout walk: `workers` threads, one slot each, take the items in order and, when no item is pending, the
follow-up tasks the committed results gave (a recovery of their failures). `commit` runs on the calling thread in item order, so
results (and what they queue) are the same for any worker count. A slot that has no item left calls `idle()` on its own thread (the
rollout plugin frees the sim it kept for that slot), and again when it ends.

Failure (any exception, an interrupt too): nothing queued starts, `abort()` stops what the running ones started, they are awaited."""
from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Sequence

Task = Callable[[], None]


class Slots:
    def __init__(self, workers: int, idle: Callable[[], None] = lambda: None, abort: Callable[[], None] | None = None):
        self.workers, self.idle, self.abort = workers, idle, abort

    def run(self, items: Sequence, call: Callable, commit: Callable[[object, object], Sequence[Task]], tasks: Sequence[Task] = ()):
        """`call(item)` on the slots; `commit(item, result)` in `items` order on this thread -> the tasks to queue; `tasks` are
        queued at the start."""
        self.cv, self.todo, self.tasks = threading.Condition(), deque(enumerate(items)), deque(tasks)
        self.results, self.closed, self.failed = {}, False, None
        threads = [threading.Thread(target=self._slot, args=(call,), name=f"slot-{i}") for i in range(self.workers)]
        for t in threads:
            t.start()
        try:
            for i, item in enumerate(items):
                with self.cv:
                    self.cv.wait_for(lambda: i in self.results or self.failed)
                    if self.failed:
                        raise self.failed
                    result = self.results.pop(i)
                if isinstance(result, _Raised):
                    raise result.error
                queued = commit(item, result)
                with self.cv:
                    self.tasks.extend(queued)
                    self.cv.notify_all()
            with self.cv:
                self.closed = True
                self.cv.notify_all()
            for t in threads:
                t.join()
            if self.failed:
                raise self.failed
        except BaseException:
            with self.cv:
                self.todo.clear(), self.tasks.clear()
                self.failed = self.failed or True
                self.cv.notify_all()
            if self.abort:
                self.abort()
            for t in threads:
                t.join()
            raise

    def _next(self, last: str | None):
        """(kind, index, work) of the slot's next job, waiting for one; None when there is none to come."""
        with self.cv:
            while True:
                if self.failed:
                    return None
                if self.todo:
                    i, item = self.todo.popleft()
                    return "item", i, item
                if self.tasks:
                    return "task", None, self.tasks.popleft()
                if self.closed:
                    return None
                if last == "item":
                    break
                self.cv.wait()
        self.idle()                                     # no item left for this slot: free what it kept for the items
        return self._next(None)

    def _slot(self, call: Callable):
        last = None
        try:
            while (job := self._next(last)) is not None:
                kind, i, work = job
                last = kind
                try:
                    out = call(work) if kind == "item" else work()
                except BaseException as e:
                    with self.cv:
                        if kind == "item":
                            self.results[i] = _Raised(e)
                        else:
                            self.failed = self.failed or e
                        self.cv.notify_all()
                    continue
                with self.cv:
                    if kind == "item":
                        self.results[i] = out
                    self.cv.notify_all()
        finally:
            self.idle()


class _Raised:
    def __init__(self, error: BaseException):
        self.error = error
