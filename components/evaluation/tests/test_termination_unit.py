"""core/termination.py (why a try ended, the effective rollout limits) and the report's termination table (CPU: synthetic records)."""
from types import SimpleNamespace

import pytest
import yaml

from evaluation.core import results
from evaluation.core.task import load_task
from evaluation.core.termination import REASONS, rollout_policy, termination

REPO = __import__("pathlib").Path(__file__).resolve().parents[3]
TASK = SimpleNamespace(budget_s=45.0, phases=[SimpleNamespace(name="approach", timeout_s=20.0),
                                              SimpleNamespace(name="grasp", timeout_s=None)])


@pytest.mark.parametrize("record, want", [
    ({"passed": True, "reason": None}, "success"),
    ({"passed": False, "reason": "timeout_s"}, "timeout_s"),
    ({"passed": False, "reason": "budget_s"}, "budget_s"),
    ({"passed": False, "reason": "while"}, "while"),
    ({"passed": False, "reason": "sanity"}, "sanity"),
    ({"passed": False, "reason": "reset"}, "reset"),
    ({"passed": False, "reason": None}, "unknown"),
    ({"passed": False, "reason": "something new"}, "unknown"),
    ({}, "unknown"),
    ({"error": "wall limit 600 s: the policy client did not finish; see x"}, "wall_limit"),
    ({"error": "interrupted: the run was stopped"}, "interrupted"),
    ({"error": "infrastructure: 3 infrastructure retries used"}, "stack_error"),
    ({"error": "the policy client exited with 1; see x"}, "stack_error"),
    ({"error": "RuntimeError: boom", "passed": True}, "stack_error"),                   # the error wins
])
def test_termination_reason(record, want):
    assert termination(record) == want and want in REASONS


def test_rollout_policy_is_the_tasks_limits_with_the_yamls_wall_and_ready_over_them():
    assert rollout_policy(TASK, {}) == {"budget_s": 45.0, "phase_timeouts_s": {"approach": 20.0}, "wall_s": 600.0, "ready_s": 180.0}
    p = rollout_policy(TASK, {"wall_s": 90, "record": "no-images", "plugin": "x"})
    assert (p["wall_s"], p["ready_s"]) == (90, 180.0) and "record" not in p
    assert rollout_policy(None, {}) == {"budget_s": None, "phase_timeouts_s": {}, "wall_s": 600.0, "ready_s": 180.0}


def test_the_shipped_tasks_limits_are_read_from_task_yaml():
    for name in ("open_fridge_door", "open_drawer"):
        raw = yaml.safe_load((REPO / "specs" / "tasks" / name / "task.yaml").read_text())
        p = rollout_policy(load_task(name, REPO), {})
        assert p["budget_s"] == raw["budget_s"] and p["phase_timeouts_s"] == {
            ph["name"]: ph["timeout_s"] for ph in raw["phases"] if ph.get("timeout_s")}
    assert load_task("no_such_task", REPO) is None


def test_the_report_has_a_termination_table_and_the_failed_phase_table_per_set():
    tries = lambda *reasons: [{"passed": False, "termination": r, "record": {"failed_phase": "grasp"}} for r in reasons]
    sets = {"heldout": {"rollouts": {"1_r0": {"seed": 1, "passed": False, "tries": tries("timeout_s", "while")},
                                      "2_r0": {"seed": 2, "passed": False, "tries": tries("timeout_s")}},
                        "rollout_policy": rollout_policy(TASK, {})}}
    r = results.Results(None, {"spec_hash": "h", "ckpt": None, "sets": sets})
    assert results.metrics(sets["heldout"]["rollouts"])["terminations"] == {"timeout_s": 2, "while": 1}
    text = r.report("c0", ["heldout"])
    assert "| termination | tries |\n|---|---|\n| timeout_s | 2 |\n| while | 1 |" in text
    assert "| failed phase (first try) | count |\n|---|---|\n| grasp | 2 |" in text
    assert "Rollout limits: budget_s 45.0, phase_timeouts_s {'approach': 20.0}, wall_s 600.0, ready_s 180.0" in text
    old = {"1_r0": {"seed": 1, "passed": True, "tries": [{"passed": True, "record": {"passed": True}}]}}
    assert results.metrics(old)["terminations"] == {"success": 1}              # a try stored without the key
