"""T3 evaluator, base pushed into the fridge (see conftest.py and check_evaluator.py): driving the base forward
(ramped TwistStamped) from robot_start into the fridge fails the active phase `approach` on a timed sanity check:
robot_contacts (the chassis touches the fridge) or penetration."""
import json

from conftest import problem_lines, result

ROBOTS = ["mobile_panda"]
TASK = "open_fridge_door"
SIM_ARGS = ["--task", TASK, "--no-autostart"]


def test_push_into_fridge(sim):
    r = result(sim, "check_evaluator.py", f"--task {TASK} --mode push --speed 0.3 --after 3",
               240, sim["logs"] / "check_push.log")
    (sim["logs"] / "evaluator_push.json").write_text(json.dumps(r, indent=1))
    assert r["ready"] and r["start"][0], r["start"]
    res = r["result"]
    assert res and not res["passed"] and res["failed_phase"] == "approach" and res["reason"] == "sanity", res
    assert res["check"] in ("sanity.robot_contacts", "sanity.penetration"), res
    # ~1 m to the fridge at 0.3 m/s: well before the approach timeout
    assert 2.0 <= res["t_fail"] <= 8.0, res
    assert res["rate_hz"] >= 95.0, res["rate_hz"]
    assert not problem_lines(sim["launch_log"]), problem_lines(sim["launch_log"])[:5]
