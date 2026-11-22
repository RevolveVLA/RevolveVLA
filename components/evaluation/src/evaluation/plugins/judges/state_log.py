"""Judges that score a rollout from its state log with the task's `success:` rule (revolve_common.success).

A retry starts from a restored mid-task state, so its log lacks the earlier travel: a restored rollout gets the phase
verdict of its record (criterion `c0_from_restore`)."""
from __future__ import annotations

import json

from ...core.errors import EvalError
from ...core.interfaces import Judge, Verdict


class StateLogJudge(Judge):
    criterion: str
    key: str                        # the success.score() result that decides the verdict

    def __init__(self, spec, params):
        if params:
            raise EvalError(f"judge {self.criterion}: takes no parameters, got {sorted(params)}")

    def score(self, record, rollout_dir, task):
        if record.get("restore"):
            return Verdict(bool(record["passed"]), {"criterion": "c0_from_restore"})
        if task is None or task.success is None:
            raise EvalError(f"judge {self.criterion}: the task has no `success:` block")
        try:
            import numpy as np
            from revolve_common import success
            live = json.loads((rollout_dir / "live.json").read_text())
            with np.load(rollout_dir / "state_log.npz") as log:
                r = success.score(log, live["grasp_pose_world"], task.success, task.boundaries.policy.grip_closed_max_m)
        except (KeyError, ValueError, OSError) as e:
            return Verdict(False, {"criterion": self.criterion, "error": f"{type(e).__name__}: {e}"})
        detail = {k: r[k] for k in ("opened", "held_fraction", "door_max_deg")}
        return Verdict(bool(r[self.key]), {"criterion": self.criterion, **detail})
