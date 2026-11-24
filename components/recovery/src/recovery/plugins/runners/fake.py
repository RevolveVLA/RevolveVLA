"""runner fake: the CPU stand-in of `sim` (revolve_common.sim_fake.FakeSession, in this process, no docker, no ROS), with the
real slot logic and the real job loop.

Params: `fail_jobs` [key]: the job's episodes never pass; `pass_after` {key: k}: its first k episodes fail; `miss_jobs` [key]:
its warm reset never reaches the restart; `faults` {key: {op: [kind]}}: infrastructure faults injected into the sim the job
runs on (ops: wait_ready, reset, wait_result, check); `infra_retries` as in `sim`. Keys are failure ids. The sessions
(launches, warm targets) are kept in `sessions`."""
from __future__ import annotations

from datagen.core import episodes
from revolve_common.sim_fake import FakeClient, FakeSession
from revolve_common.sim_session import DomainPool

from ...core.jobs import run_job
from ...core.registry import REGISTRY
from ...core.runner import SlotRunner


class _NoDocker:
    def names(self):
        return []

    def domains(self):
        return set()

    def rm(self, name):
        pass


class Report:
    def __init__(self, ok: bool, reason: str | None = None):
        self.ok, self.reason = ok, reason

    def to_dict(self):
        return {"ok": self.ok, "reason": self.reason, "base_err_m": 0.0 if self.ok else 0.5}


class WarmFakeClient(FakeClient):
    def reset_to(self, target):
        self.reset()
        self.s.warm_targets.append(target)
        return Report(True) if self.s.reaches(target) else Report(False, "missed the target (fake)")


class WarmFakeSession(FakeSession):
    def __init__(self, *a, reaches=lambda target: True, **kw):
        super().__init__(*a, **kw)
        self.reaches, self.warm_targets = reaches, []

    def connect(self):
        return WarmFakeClient(self)


@REGISTRY.register("runner", "fake")
class FakeRunner(SlotRunner):
    def __init__(self, spec, params):
        self.fail, self.miss = set(params.get("fail_jobs", [])), set(params.get("miss_jobs", []))
        self.pass_after = dict(params.get("pass_after", {}))
        self.faults = {k: {op: list(v) for op, v in f.items()}          # used up as they are raised
                       for k, f in params.get("faults", {}).items()}
        self.sessions = []
        super().__init__(spec, params)

    def domain_pool(self):
        return DomainPool(self.prefix, docker=_NoDocker())

    def session(self, slot, scene):
        s = WarmFakeSession(scene, self.spec.robot, self.spec.task, slot.dir, slot.name, slot.domain,
                            self.spec.phases.episodes.record)
        self.sessions.append(s)
        return s

    def work(self, slot, job, warm, generator):
        s = slot.session
        s.passes = lambda n: (job.key not in self.fail
                              and len(episodes.read(job.dir)["tries"]) >= self.pass_after.get(job.key, 0))
        s.reaches = lambda target: job.key not in self.miss
        s.faults = self.faults.setdefault(job.key, {})
        client = s.connect()
        run_job(client, generator, job, self.spec.task, self.spec.robot, self.spec.phases.episodes, slot.dir,
                client.reset_to if warm else None)
