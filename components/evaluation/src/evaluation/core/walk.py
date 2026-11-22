"""The rollout walk of one eval set: every config x `rollouts`, config-major, `workers` at a time (datagen's `pool.ordered`:
rollouts run in parallel, results are committed in order on the calling thread, so the results are the same for any worker
count). Each item is a chain of tries in `<out>/<set>/<seed>_r<i>/` (retries in its `retry_<j>/`), judged one by one.

`RetryPolicy` is the only second-attempt mechanism (the seam a later recovery component replaces): after a try the judge
fails whose phase verdict names a restorable failed phase with a saved start state, the policy runs again from that state.
A try without a verdict (a stack error, not the policy's failure) is run once more. A try whose dir already holds a
verdict (rollout.json) is kept, so a killed run resumes inside an item."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from datagen.core import pool

from .errors import EvalError
from .interfaces import Built, EvalItem, Judge, Policy, ResolvedCkpt, Rollout, Verdict
from .spec import RunnerSpec

RESTORABLE = ("approach", "grasp", "open", "release")      # phases whose start state a retry can restore
STACK_ATTEMPTS = 2


class Stopped(EvalError):
    """A worker saw the run being aborted: it starts nothing more."""


@dataclass(frozen=True)
class RetryPolicy:
    retries: int
    restorable: Sequence[str] = RESTORABLE

    def restore(self, tries: int, record: Mapping, passed: bool, rollout_dir: Path) -> tuple[Path, str] | None:
        """(dir, phase) to retry from after `tries` tries, the last one at `rollout_dir`; None = stop here."""
        phase = record.get("failed_phase")
        if passed or tries > self.retries or phase not in self.restorable:
            return None
        return (rollout_dir, phase) if (rollout_dir / "phase_states" / f"{phase}_start.npz").is_file() else None


class Progress:
    """`PROGRESS {json}` lines of one set (what eval_progress.parse reads): rollouts, rollouts_done, passed,
    first_try_passed, running."""
    def __init__(self, name: str, keys: Sequence[str], log: Callable[[str], None]):
        self.name, self.log, self.lock = name, log, threading.Lock()
        self.state, self.first = {k: "pending" for k in keys}, {}

    def update(self, key: str, state: str, first: bool | None = None):
        with self.lock:
            self.state[key] = state
            if first is not None:
                self.first[key] = first
            st = list(self.state.values())
            self.log("PROGRESS " + json.dumps(dict(
                set=self.name, rollouts=len(st), rollouts_done=sum(x in ("pass", "fail", "error") for x in st),
                passed=sum(x == "pass" for x in st), first_try_passed=sum(self.first.values()),
                running=[k for k, x in self.state.items() if x == "running"])))


def key(item: EvalItem, i: int) -> str:
    return f"{item.config.seed}_r{i}"


class Walk:
    def __init__(self, rollout: Rollout, policy: Policy, judge: Judge, retry: RetryPolicy, runner: RunnerSpec,
                 ckpt: ResolvedCkpt | None, task, out: Path, workers: int, gui: bool, log: Callable[[str], None],
                 abort: Callable[[], None] | None = None):
        self.rollout, self.policy, self.judge, self.retry, self.runner = rollout, policy, judge, retry, runner
        self.ckpt, self.task = ckpt, task
        self.out, self.workers, self.gui, self.log, self.hook = out, workers, gui, log, abort
        self.stopping = threading.Event()

    def run(self, name: str, built: Built, rollouts: int, done: Mapping[str, Mapping],
            commit: Callable[[str, dict], None]):
        """Every item of `built` x `rollouts` not in `done` (entries by key); `commit(key, entry)` in order."""
        todo = [(it, i) for it in built.items for i in range(rollouts) if key(it, i) not in done]
        progress = Progress(name, [key(it, i) for it in built.items for i in range(rollouts)], self.log)
        for k, e in done.items():
            progress.state[k], progress.first[k] = "pass" if e["passed"] else "fail", e["tries"][0]["passed"]
        pool.ordered(todo, lambda x: self._item(name, *x, progress), lambda x, e: commit(key(*x), e), self.workers,
                     self._abort)

    def _abort(self):
        self.stopping.set()                 # running workers start no retry / second attempt
        if self.hook:
            self.hook()

    def _item(self, name: str, item: EvalItem, i: int, progress: Progress) -> dict:
        k, here, tries, restore = key(item, i), self.out / name / key(item, i), [], None
        progress.update(k, "running")
        while True:
            record, verdict = self._try(item, here, restore)
            tries.append({"dir": str(here), "restored_at": restore and restore[1], "passed": verdict.passed,
                          "detail": dict(verdict.detail), "record": record})
            if tries[0] is tries[-1]:
                progress.update(k, "running", first=verdict.passed)
            if (restore := self.retry.restore(len(tries), record, verdict.passed, here)) is None:
                break
            here = self.out / name / k / f"retry_{len(tries)}"
        state = "pass" if verdict.passed else "error" if record.get("error") else "fail"
        progress.update(k, state)
        self.log(f"{time.strftime('%H:%M:%S')} evaluate {name}: {k}: " + ", ".join(
            f"try {j}: " + ("PASS" if t["passed"] else f"FAIL {t['record'].get('failed_phase')}" if not t["record"].get("error")
                            else f"ERROR {t['record']['error']}") for j, t in enumerate(tries)))
        return {"seed": item.config.seed, "passed": verdict.passed, "tries": tries}

    def _try(self, item: EvalItem, here: Path, restore) -> tuple[dict, Verdict]:
        record = self._kept(here, restore)
        for attempt in range(1, STACK_ATTEMPTS + 1) if record is None else ():
            if self.stopping.is_set():
                raise Stopped("aborted")
            record = self.rollout.run(item, here, seed=item.config.seed, restore=restore, runner=self.runner,
                                        policy=self.policy, ckpt=self.ckpt, gui=self.gui)
            if not record.get("error"):
                break
            self.log(f"{here.name}: stack error ({record['error']}), attempt {attempt} of {STACK_ATTEMPTS}")
        if record.get("error"):             # no verdict to judge
            return record, Verdict(False, {"error": record["error"]})
        return record, self.judge.score(record, here, self.task)

    @staticmethod
    def _kept(here: Path, restore) -> dict | None:
        """The record of a finished try in `here` (a verdict, and for a retry the restore), else None."""
        try:
            record = json.loads((here / "rollout.json").read_text())
        except (OSError, ValueError):
            return None
        return record if record.get("phases") is not None and (restore is None or record.get("restore")) else None
