"""A rollout without docker, for CPU tests: writes the rollout dir contract (rollout.json, phase_report.json,
phase_states/<phase>_start.npz) and returns the record, as scripted by its params.

Params: phases (default approach, grasp, open, release), fail_seeds (default none) fail at `failed_phase` (default: the
last phase) unless restored from a saved phase state (a retry passes), `always_fail` (default false) makes retries fail
too, `delay_s` (default 0) sleeps before returning, `end_reasons` {seed: reason} sets the record's termination (success,
budget, no_action, abort, steps; or the stack errors wall_limit / stack_error: the record has an `error`; default: success
when the rollout passed, else budget), `rtf` (default none) is the real-time factor it assumes."""
from __future__ import annotations

import json
import time

import numpy as np

from ...core.interfaces import Rollout
from ...core.registry import REGISTRY

PHASES = ("approach", "grasp", "open", "release")


@REGISTRY.register("rollout", "fake")
class FakeRollout(Rollout):
    def __init__(self, spec, params):
        self.phases = tuple(params.get("phases", PHASES))
        self.fail_seeds = set(params.get("fail_seeds", ()))
        self.failed_phase = params.get("failed_phase", self.phases[-1])
        self.always_fail, self.delay_s = params.get("always_fail", False), params.get("delay_s", 0)
        self.end_reasons, self._rtf = {int(k): v for k, v in params.get("end_reasons", {}).items()}, params.get("rtf")

    def run(self, item, out, *, seed, restore=None, runner, policy, ckpt, gui=False):
        time.sleep(self.delay_s)
        out.mkdir(parents=True, exist_ok=True)
        fails = seed in self.fail_seeds and (restore is None or self.always_fail)
        failed = self.failed_phase if fails else None
        (out / "phase_states").mkdir(exist_ok=True)
        status, reached = {}, True
        for p in self.phases:
            status[p] = "not_reached" if not reached else "fail" if p == failed else "pass"
            reached = reached and p != failed
            if status[p] != "not_reached":
                np.savez(out / "phase_states" / f"{p}_start.npz", qpos=np.zeros(1))
        (out / "phase_report.json").write_text(json.dumps(
            {"phases": {p: {"status": s} for p, s in status.items()},
             "result": {"failed_phase": failed, "regenerate": None, "failed_checks": [failed] if failed else []}}))
        end = self.end_reasons.get(seed) or ("success" if failed is None else "budget")
        error = {"wall_limit": "sim job over its wall limit 1 s", "stack_error": "RuntimeError: fake stack error"}.get(end)
        record = dict(seed=seed, dir=str(out), policy=policy.name, ckpt=ckpt.path if ckpt else None, scene=str(item.scene),
                      restore=None if restore is None else dict(source=str(restore[0]), phase=restore[1]),
                      phases=None if error else status, failed_phase=failed, regenerate=None,
                      failed_checks=[failed] if failed else [], passed=failed is None and not error, error=error, end_reason=None if error else end)
        (out / "rollout.json").write_text(json.dumps(record))
        return record

    def rtf(self, policy):
        return self._rtf
