"""runner fake: the CPU stand-in of `sim` (revolve_common.sim_fake.FakeSession, in this process, no docker, no ROS).

Params: `fail_seeds` [seed]: the scene never passes; `pass_after` {seed: k}: its first k tries fail; `faults` {seed: {op:
[kind]}}: infrastructure faults injected into the session of that scene (ops: wait_ready, reset, wait_result, check;
kinds: ready_timeout, reset_failed, result_timeout, clock_fault, ...); `infra_retries` as in the template."""
from __future__ import annotations

from pathlib import Path

from datagen.core import episodes
from datagen.core.interfaces import Generator
from datagen.core.registry import register
from revolve_common.sim_fake import FakeSession
from revolve_common.sim_session import DomainPool

from .session_runner import SessionRunner


class _NoDocker:
    def names(self):
        return []

    def domains(self):
        return set()

    def rm(self, name):
        pass


@register("runner", "fake")
class FakeRunner(SessionRunner):
    def __init__(self, spec, params):
        self.fail = {int(s) for s in params.get("fail_seeds", [])}
        self.pass_after = {int(k): v for k, v in params.get("pass_after", {}).items()}
        self.faults = {int(k): v for k, v in params.get("faults", {}).items()}
        super().__init__(spec, params)

    def domain_pool(self):
        return DomainPool(self.prefix, docker=_NoDocker())

    def session(self, seed, instance, name, domain):
        passes = lambda n: seed not in self.fail and n >= self.pass_after.get(seed, 0)         # noqa: E731
        return FakeSession(instance, self.spec.robot, self.spec.task, Path(instance).parent, name, domain,
                           self.spec.episodes.record, passes=passes, faults=self.faults.get(seed))

    def work(self, session, seed, instance, generator: Generator):
        episodes.run_episodes(session.connect(), generator, self.spec.task, self.spec.robot, instance, seed,
                              self.spec.episodes)
