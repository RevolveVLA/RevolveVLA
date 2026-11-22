"""core/termination.py (why a try ended, the effective rollout limits), the report's termination table, and the docker plugin's
rtf / wall limit params (CPU: records are synthetic, rollout.run is monkeypatched)."""
from types import SimpleNamespace

import pytest

from evaluation.core import results
from evaluation.core.registry import REGISTRY
from evaluation.core.termination import REASONS, rollout_policy, termination
from evaluation.plugins.rollouts import docker as D

TASK = SimpleNamespace(budget_s=45.0, wait_s=30.0, after_success_s=2.0, wall_safety=2.0, setup_s=240.0)
spec = SimpleNamespace(train=SimpleNamespace(task="open_fridge_door", robot="mobile_panda"), run=SimpleNamespace(workers=None, headless=True))
oracle = REGISTRY.create("policy", "oracle", spec, {})


@pytest.mark.parametrize("record, want", [
    ({"end_reason": "success"}, "success"),
    ({"end_reason": "budget"}, "budget"),
    ({"end_reason": "no_action"}, "no_action"),
    ({"end_reason": "abort"}, "abort"),
    ({"end_reason": "steps"}, "budget"),                                       # the sim's fallback end
    ({"end_reason": None}, "unknown"),
    ({}, "unknown"),
    ({"error": "sim job over its wall limit 1780 s", "sim_exit": None}, "wall_limit"),
    ({"error": "no phase_report.json (sim exit 3)", "sim_exit": 3}, "wall_limit"),
    ({"error": "no phase_report.json (sim exit 1)", "sim_exit": 1}, "stack_error"),
    ({"error": "no phase_report.json (sim exit 137)", "sim_exit": 137}, "interrupted"),
    ({"error": "RuntimeError: policy server exited before it was ready"}, "stack_error"),
    ({"error": "RuntimeError: boom", "end_reason": "success", "sim_exit": 137}, "stack_error"),     # the error wins
])
def test_termination_reason(record, want):
    assert termination(record) == want and want in REASONS


def test_rollout_policy_is_the_task_block_with_the_yaml_over_it():
    base = rollout_policy(TASK, {}, None)
    assert base == {"budget_s": 45.0, "wait_s": 30.0, "after_success_s": 2.0, "wall_safety": 2.0, "setup_s": 240.0,
                    "rtf": None, "wall_limit_s": None}
    p = rollout_policy(TASK, {"budget_s": 20, "wall_safety": 3.0, "rtf": 0.3, "video": True, "plugin": "x"}, None)
    assert (p["budget_s"], p["wait_s"], p["wall_safety"], p["rtf"]) == (20, 30.0, 3.0, 0.3)
    assert p["wall_limit_s"] == round((30 + 20 + 2) / 0.3 * 3.0, 1) and "video" not in p
    assert rollout_policy(TASK, {}, 0.5)["wall_limit_s"] == round(77 / 0.5 * 2.0, 1)           # the plugin's rtf
    assert rollout_policy(TASK, {"rtf": 0.2}, 0.5)["rtf"] == 0.2                                 # the yaml's wins
    assert rollout_policy(SimpleNamespace(**{**TASK.__dict__, "after_success_s": None}), {}, 1.0)["wall_limit_s"] == 150.0


def test_rollout_policy_agrees_with_rollout_py_and_the_shipped_task():
    import rollout as R
    from revolve_common import specs
    ro = specs.load(R.REPO / "specs" / "tasks" / "open_fridge_door" / "task.yaml", robot="mobile_panda").task.rollout
    for rtf in (0.1, 0.3, 0.0, 1.0):
        want = R.wall_limit(R.sim_budget_s(ro, budget_s=20), rtf, ro.wall_safety)
        assert rollout_policy(ro, {"budget_s": 20, "rtf": rtf}, None)["wall_limit_s"] == want
    assert (ro.wall_safety, ro.setup_s) == (R.SAFETY, R.SETUP_S)             # the task states the old constants
    assert R.wall_limit(77, 0.3) == R.wall_limit(77, 0.3, R.SAFETY) and R.wall_limit(77, 0.3, 4.0) == pytest.approx(2 * R.wall_limit(77, 0.3), abs=0.2)


def test_the_report_has_a_termination_table_per_set():
    tries = lambda *reasons: [{"passed": False, "termination": r, "record": {"failed_phase": "grasp"}} for r in reasons]
    sets = {"heldout": {"rollouts": {"1_r0": {"seed": 1, "passed": False, "tries": tries("budget", "no_action")},
                                      "2_r0": {"seed": 2, "passed": False, "tries": tries("budget")}},
                        "rollout_policy": rollout_policy(TASK, {"rtf": 0.3}, None)}}
    r = results.Results(None, {"spec_hash": "h", "ckpt": None, "sets": sets})
    assert results.metrics(sets["heldout"]["rollouts"])["terminations"] == {"budget": 2, "no_action": 1}
    text = r.report("c0", ["heldout"])
    assert "| termination | tries |\n|---|---|\n| budget | 2 |\n| no_action | 1 |" in text
    assert "Rollout limits: budget_s 45.0, wait_s 30.0, after_success_s 2.0, wall_safety 2.0, setup_s 240.0, rtf 0.3" in text
    old = {"1_r0": {"seed": 1, "passed": True, "tries": [{"passed": True, "record": {"end_reason": "success"}}]}}
    assert results.metrics(old)["terminations"] == {"success": 1}              # a try stored before the key existed


@pytest.fixture
def calls(monkeypatch):
    calls = []
    monkeypatch.setattr(D, "load", lambda name="rollout": SimpleNamespace(
        run=lambda *a, **k: calls.append(k) or {"passed": True}, default_parallel=lambda p: 3,
        measured_rtf=lambda n: {1: 0.3, 2: 0.31, 3: 0.4}[n]))
    return calls


def run(plugin, tmp_path, workers=None):
    from datagen.core.interfaces import Config
    from evaluation.core.interfaces import EvalItem
    from evaluation.core.spec import RunnerSpec
    plugin.run(EvalItem(Config("heldout", 1), tmp_path), tmp_path / "o", seed=1, runner=RunnerSpec(), policy=oracle, ckpt=None)


def test_docker_uses_the_measured_rtf_unless_the_yaml_gives_one(calls, tmp_path):
    run(REGISTRY.create("rollout", "docker", spec, {}), tmp_path)
    assert calls[-1]["rtf"] == 0.4                                            # 3 workers by free GPU memory
    two = SimpleNamespace(**{**spec.__dict__, "run": SimpleNamespace(workers=2, headless=True)})
    run(REGISTRY.create("rollout", "docker", two, {}), tmp_path)
    assert calls[-1]["rtf"] == 0.31
    gui = SimpleNamespace(**{**spec.__dict__, "run": SimpleNamespace(workers=2, headless=False)})
    assert REGISTRY.create("rollout", "docker", gui, {}).rtf(oracle) == 0.3     # a GUI run has one worker
    run(REGISTRY.create("rollout", "docker", spec, {"rtf": 0.2}), tmp_path)
    assert calls[-1]["rtf"] == 0.2


def test_docker_passes_the_wall_limit_params_and_names_them(calls, tmp_path):
    run(REGISTRY.create("rollout", "docker", spec, {"wall_safety": 3.0, "setup_s": 60}), tmp_path)
    assert (calls[-1]["wall_safety"], calls[-1]["setup_s"]) == (3.0, 60)
    run(REGISTRY.create("rollout", "docker", spec, {}), tmp_path)
    assert "wall_safety" not in calls[-1] and "setup_s" not in calls[-1]       # absent: the task's values (rollout.run)
    assert "wall_safety" in D.PARAMS and "setup_s" in D.PARAMS
