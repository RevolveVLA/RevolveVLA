"""The runner template: one scene instance = one sim session, relaunched on infrastructure faults (DATAGEN-v2.md 4.4, 12).

A subclass says how to make its session and how the episode loop reaches it (`work`); the lease of a (name, ROS domain)
pair, the relaunch budget, the fault bookkeeping in the scene log and the SceneError mapping are the same for all."""
from __future__ import annotations

import sys
import threading
from abc import abstractmethod
from pathlib import Path

from datagen.core import episodes
from datagen.core.interfaces import Generator, Runner
from datagen.core.naming import container_prefix
from datagen.core.spec import DatagenSpec
from revolve_common.sim_session import DomainPool, RelaunchPolicy, SimInfraError, run_with_relaunch


class RunnerStopped(Exception):
    """The run was interrupted: no more sims."""


def say(message: str) -> None:
    print(f"datagen: {message}", file=sys.stderr)


class SessionRunner(Runner):
    def __init__(self, spec: DatagenSpec, params):
        """params: `infra_retries` (3): sim relaunches per scene before it is given up."""
        self.spec, self.retries = spec, int(params.get("infra_retries", 3))
        self.prefix = container_prefix(spec.out)
        self.stopped = threading.Event()
        self.pool = self.domain_pool()

    def domain_pool(self) -> DomainPool:
        return DomainPool(self.prefix)

    @abstractmethod
    def session(self, seed: int, instance: Path, name: str, domain: int):
        """A not yet started SimSession-like for the scene (out = the instance's folder)."""

    @abstractmethod
    def work(self, session, seed: int, instance: Path, generator: Generator) -> None:
        """The episode loop on the started `session`; a SimFault raised out of it relaunches the sim."""

    def run(self, seed: int, instance: Path, generator: Generator) -> None:
        folder, policy = Path(instance).parent, RelaunchPolicy(self.retries)
        if episodes.finished(episodes.read(folder), self.spec.episodes):
            return
        name, domain = self.pool.lease()
        try:
            run_with_relaunch(self.session(seed, instance, name, domain),
                              lambda s: self.work(s, seed, instance, generator), policy, say)
        except SimInfraError as e:
            raise episodes.SceneError(f"infrastructure: {e}") from e
        finally:
            self.pool.release(name)
            episodes.add_faults(folder, policy.faults)

    def stop(self) -> None:
        self.stopped.set()
