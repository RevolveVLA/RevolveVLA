"""Why a rollout try ended, and the limits that applied: both derived in one place from what the rollout already records.

`termination(record)` reads the rollout record (rollout.py's `end_reason` from the sim, `error` / `sim_exit` from the
stack). `rollout_policy(rollout, params, rtf)` = the task's `rollout:` block with the eval yaml's overrides on top."""
from __future__ import annotations

from typing import Any, Mapping

SUCCESS, BUDGET, NO_ACTION, ABORT = "success", "budget", "no_action", "abort"
WALL_LIMIT, STACK_ERROR, INTERRUPTED, UNKNOWN = "wall_limit", "stack_error", "interrupted", "unknown"
REASONS = (SUCCESS, BUDGET, NO_ACTION, ABORT, WALL_LIMIT, STACK_ERROR, INTERRUPTED, UNKNOWN)

SIM_DRIVER_TIMEOUT = 3                  # simulator cli.py rollout: the scene test timed out (the wall limit)
KILLED = (130, 137, 143)                # SIGINT / SIGKILL / SIGTERM of the sim job (abort = stop_jobs)
RTF_FLOOR = 0.03                        # as rollout.py: a lower measured RTF gives no longer wall limit
LIMITS = ("budget_s", "wait_s", "after_success_s", "wall_safety", "setup_s")   # the task's `rollout:`, eval yaml overrides
NO_TASK_LIMITS = dict(budget_s=60.0, wait_s=0.0, after_success_s=None, wall_safety=2.0, setup_s=240.0)  # main's Task has no `rollout:`; PRD 06


def termination(record: Mapping[str, Any]) -> str:
    """success (after the run-on) | budget | no_action | abort (physics) | wall_limit | stack_error | interrupted |
    unknown (a record without an end reason). `steps` (the sim's fallback end, no ending ever set) counts as budget."""
    if error := record.get("error"):
        exit_ = record.get("sim_exit")
        if "wall limit" in error or (error.startswith("no phase_report") and exit_ == SIM_DRIVER_TIMEOUT):
            return WALL_LIMIT
        return INTERRUPTED if error.startswith("no phase_report") and exit_ in KILLED else STACK_ERROR
    end = record.get("end_reason")
    return BUDGET if end == "steps" else end if end in (SUCCESS, BUDGET, NO_ACTION, ABORT) else UNKNOWN


def rollout_policy(rollout, params: Mapping[str, Any], rtf: float | None) -> dict[str, Any]:
    """The effective limits of a set's rollouts: the task's `rollout:` (budget_s, wait_s, after_success_s, wall_safety,
    setup_s) with `params` over it, the real-time factor `rtf` assumed (None: the plugin has none) and the wall limit
    [s] from the unpause = (wait_s + budget_s + after_success_s) / rtf x wall_safety (the sim job gets setup_s more)."""
    p = {k: params[k] if params.get(k) is not None else getattr(rollout, k, NO_TASK_LIMITS[k]) for k in LIMITS}
    rtf = params["rtf"] if params.get("rtf") is not None else rtf
    sim_s = p["wait_s"] + p["budget_s"] + (p["after_success_s"] or 0.0)
    wall = None if rtf is None else round(sim_s / max(float(rtf), RTF_FLOOR) * p["wall_safety"], 1)
    return {**p, "rtf": rtf, "wall_limit_s": wall}
