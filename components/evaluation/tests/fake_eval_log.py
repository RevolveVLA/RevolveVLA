"""The src `fake` rollout plus a call log: every finished call is appended to $FAKE_EVAL_LOG as `rollout <seed> <dir name>`.
Registered as rollout plugin `fake_eval_log` (put tests/ on $EVAL_PLUGIN_PATH). Params as the `fake` rollout."""
import os

from evaluation.core.registry import REGISTRY
from evaluation.plugins.rollouts.fake import FakeRollout


@REGISTRY.register("rollout", "fake_eval_log")
class FakeLogRollout(FakeRollout):
    def run(self, item, out, *, seed, **kw):
        record = super().run(item, out, seed=seed, **kw)
        if log := os.environ.get("FAKE_EVAL_LOG"):
            with open(log, "a") as f:
                f.write(f"rollout {seed} {out.name}\n")
        return record
