"""The runner template: sims kept in slots, one job at a time per slot (`run.workers` jobs in parallel = that many slots).

A slot is one sim (its own folder `<out>/sims/<slot>/`, a leased container name + ROS domain). With `run.reset: warm` a job
that finds a slot whose sim holds its world runs on it after a warm reset to its restart; a miss relaunches the sim at the
restart (the job's own instance) and the job goes on with plain resets. With `relaunch` every job launches its sim. A
SimFault relaunches the sim too, counted in the job's infrastructure retries (not a failed episode).

A subclass says how to make a session and how the job's episode loop reaches it (`work`)."""
from __future__ import annotations

import sys
import threading
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

from datagen.core import episodes
from datagen.core.interfaces import Generator
from revolve_common.sim_client import WarmResetMissed
from revolve_common.sim_session import DomainPool, RelaunchPolicy, SimFault, SimInfraError, same_world

from .errors import JobError
from .interfaces import Runner
from .jobs import Job
from .naming import prefix
from .spec import RecoverySpec


class RunnerStopped(Exception):
    """The run was interrupted: no more sims."""


def say(message: str) -> None:
    print(f"recover: {message}", file=sys.stderr)


@dataclass
class Slot:
    name: str
    domain: int
    dir: Path
    session: object = None
    scene: Path | None = None           # the instance the running sim was launched with

    @property
    def running(self) -> bool:
        return self.session is not None


class SlotRunner(Runner):
    def __init__(self, spec: RecoverySpec, params):
        """params: `infra_retries` (3): sim relaunches per job before it is given up."""
        self.spec, self.retries = spec, int(params.get("infra_retries", 3))
        self.prefix = prefix(spec)
        self.stopped, self.lock = threading.Event(), threading.Lock()
        self.pool, self.slots, self.idle = self.domain_pool(), [], []

    def domain_pool(self) -> DomainPool:
        return DomainPool(self.prefix)

    @abstractmethod
    def session(self, slot: Slot, scene: Path):
        """A not yet started SimSession-like for `scene` (out = the slot's folder)."""

    @abstractmethod
    def work(self, slot: Slot, job: Job, warm: bool, generator: Generator) -> None:
        """The job's episode loop on the slot's started sim (`core.jobs.run_job`); `warm`: reach the restart by a warm reset.
        A SimFault out of it relaunches the sim, a WarmResetMissed relaunches it at the restart."""

    # ---- slots
    def _take(self, job: Job) -> Slot:
        with self.lock:
            fits = [s for s in self.idle if s.running and same_world(s.scene, job.scene)]
            if slot := (fits or self.idle or [None])[0]:
                self.idle.remove(slot)
                return slot
            name, domain = self.pool.lease()
            slot = Slot(name, domain, Path(self.spec.out) / "sims" / name)
            self.slots.append(slot)
            return slot

    def _launch(self, slot: Slot, scene: Path) -> None:
        if self.stopped.is_set():
            raise RunnerStopped("run interrupted")
        self._stop(slot)
        episodes.clean_partial(slot.dir)
        slot.session, slot.scene = self.session(slot, scene), scene
        slot.session.start()

    @staticmethod
    def _stop(slot: Slot) -> None:
        if slot.session is not None:
            slot.session.close()
        slot.session = slot.scene = None

    # ---- Runner
    def run(self, job: Job, generator: Generator) -> None:
        if episodes.finished(episodes.read(job.dir), self.spec.phases.episodes):
            return
        policy = RelaunchPolicy(self.retries)
        try:
            slot = self._take(job)
        except SimInfraError as e:
            raise JobError(f"infrastructure: {e}") from e
        try:
            self._serve(slot, job, generator, policy)
        except SimInfraError as e:
            raise JobError(f"infrastructure: {e}") from e
        finally:
            with self.lock:
                self.idle.append(slot)
            episodes.add_faults(job.dir, policy.faults)

    def _serve(self, slot: Slot, job: Job, generator: Generator, policy: RelaunchPolicy) -> None:
        allowed = self.spec.run.reset == "warm"
        while True:
            warm = allowed and slot.running and same_world(slot.scene, job.scene)
            if not warm:
                self._launch(slot, job.scene)               # a sim at the restart: plain resets reach it again
            try:
                return self.work(slot, job, warm, generator)
            except WarmResetMissed as e:
                say(f"job {job.key}: warm reset missed ({e}): relaunch at the restart")
                allowed = False
            except SimFault as fault:
                policy.record(fault)
                say(f"sim {slot.name}: {fault}: relaunch {len(policy.faults)}/{policy.max_retries}")
                self._stop(slot)

    def stop(self) -> None:
        self.stopped.set()

    def close(self) -> None:
        for slot in self.slots:
            self._stop(slot)
            self.pool.release(slot.name)
        self.slots, self.idle = [], []
