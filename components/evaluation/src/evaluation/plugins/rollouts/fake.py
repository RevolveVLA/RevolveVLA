"""A rollout without a sim, for CPU tests: writes rollout.json and returns the record (the shape of the `sim` plugin's),
as scripted by its params.

Params: phases (default approach, grasp, open), fail_seeds (default none) fail at `failed_phase` (default: the last phase)
unless the try is a retry (a dir named `retry_<j>`), `always_fail` (default false) makes retries fail too, `delay_s` (default 0)
sleeps before returning, `end_reasons` {seed: reason} sets the record's termination (success, timeout_s, budget_s, while,
sanity, reset; or the stack errors wall_limit / interrupted / stack_error: the record has an `error`; default: success when
the rollout passed, else timeout_s). `released` lists the configs whose rollouts are over."""
from __future__ import annotations

import json
import time

from ...core.interfaces import Rollout
from ...core.registry import REGISTRY

PHASES = ("approach", "grasp", "open")
ERRORS = {"wall_limit": "wall limit 1 s", "interrupted": "interrupted: fake", "stack_error": "RuntimeError: fake stack error"}


@REGISTRY.register("rollout", "fake")
class FakeRollout(Rollout):
    def __init__(self, spec, params):
        self.phases = tuple(params.get("phases", PHASES))
        self.fail_seeds = set(params.get("fail_seeds", ()))
        self.failed_phase = params.get("failed_phase", self.phases[-1])
        self.always_fail, self.delay_s = params.get("always_fail", False), params.get("delay_s", 0)
        self.end_reasons = {int(k): v for k, v in params.get("end_reasons", {}).items()}
        self.released, self.stopped = [], False

    def run(self, item, out, *, seed, runner, policy, ckpt, gui=False):
        time.sleep(self.delay_s)
        out.mkdir(parents=True, exist_ok=True)
        end = self.end_reasons.get(seed)
        fails = seed in self.fail_seeds and (self.always_fail or not out.name.startswith("retry_"))
        failed = self.failed_phase if fails or end not in (None, "success", *ERRORS) else None
        status, reached = {}, True
        for p in self.phases:
            status[p] = "not_reached" if not reached else "failed" if p == failed else "passed"
            reached = reached and p != failed
        end = end or ("success" if failed is None else "timeout_s")
        error = ERRORS.get(end)
        record = dict(seed=seed, dir=str(out), policy=policy.name, ckpt=ckpt.path if ckpt else None, scene=str(item.scene),
                      episode=None, phases=None if error else status, failed_phase=failed,
                      reason=None if error or end == "success" else end, passed=failed is None and not error, error=error)
        (out / "rollout.json").write_text(json.dumps(record))
        return record

    def release(self, item):
        self.released.append(item.config.seed)

    def stop(self):
        self.stopped = True
