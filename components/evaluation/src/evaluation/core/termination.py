"""Why a rollout try ended, and the limits that applied: both derived in one place from what the rollout already records.

`termination(record)` reads the rollout record: the evaluator's `reason` (DATAGEN-v2.md 4.3: timeout_s, budget_s, while,
sanity, reset; `success` when it passed) or, for a stack error, what the plugin says in `error`. `rollout_policy(task,
params)` = the task's limits (task.yaml: the budget and the phases' timeouts) with the eval yaml's wall / ready limits."""
from __future__ import annotations

from typing import Any, Mapping

SUCCESS, TIMEOUT_S, BUDGET_S, WHILE, SANITY, RESET = "success", "timeout_s", "budget_s", "while", "sanity", "reset"
WALL_LIMIT, STACK_ERROR, INTERRUPTED, UNKNOWN = "wall_limit", "stack_error", "interrupted", "unknown"
REASONS = (SUCCESS, TIMEOUT_S, BUDGET_S, WHILE, SANITY, RESET, WALL_LIMIT, STACK_ERROR, INTERRUPTED, UNKNOWN)

WALL_S = 600.0                  # the yaml's `wall_s`: wall seconds one episode's client may take (bring-up of the client to the result)
READY_S = 180.0                 # the yaml's `ready_s`: wall seconds the sim may take to be ready (DATAGEN 4.2)
LIMITS = {"wall_s": WALL_S, "ready_s": READY_S}


def termination(record: Mapping[str, Any]) -> str:
    """success | the evaluator's reason (timeout_s, budget_s, while, sanity, reset) | wall_limit | interrupted |
    stack_error | unknown (a record without a reason). The plugin words a stack error as `wall limit ...` or
    `interrupted ...`; any other `error` is a stack_error."""
    if error := record.get("error"):
        return WALL_LIMIT if error.startswith("wall limit") else INTERRUPTED if error.startswith("interrupted") else STACK_ERROR
    if record.get("passed"):
        return SUCCESS
    reason = record.get("reason")
    return reason if reason in (TIMEOUT_S, BUDGET_S, WHILE, SANITY, RESET) else UNKNOWN


def rollout_policy(task, params: Mapping[str, Any]) -> dict[str, Any]:
    """The limits of a set's rollouts: the task's `budget_s` and its phases' `timeout_s` (None / {} without a task.yaml)
    and the rollout plugin's `wall_s` / `ready_s` params over their defaults."""
    return {"budget_s": getattr(task, "budget_s", None),
            "phase_timeouts_s": {p.name: p.timeout_s for p in getattr(task, "phases", ()) if p.timeout_s},
            **{k: params[k] if params.get(k) is not None else v for k, v in LIMITS.items()}}
