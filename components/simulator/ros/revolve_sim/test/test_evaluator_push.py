"""T3 evaluator, base pushed into the scene (see conftest.py and check_evaluator.py), per robot of
robot_cases.TASKS: driving the base forward (ramped TwistStamped; mobile_panda from robot_start into the fridge,
mobile_so101_xl into the kitchen counter after a drive to face it) fails the active phase `approach` on a timed
sanity check: robot_contacts (a robot link touches the scene) or penetration."""
import json

import robot_cases
from conftest import problem_lines, result

ROBOTS = list(robot_cases.TASKS)
SIM_ARGS = {r: ["--task", c["task"], "--no-autostart"] for r, c in robot_cases.TASKS.items()}


def test_push_into_scene(sim):
    case = robot_cases.TASKS[sim["robot"]]
    push = case["push"]
    to = f" --to {','.join(str(v) for v in push['to'])}" if push["to"] else ""
    r = result(sim, "check_evaluator.py", f"--task {case['task']} --mode push --speed {push['speed']} --after 3{to}",
               240, sim["logs"] / "check_push.log")
    (sim["logs"] / "evaluator_push.json").write_text(json.dumps(r, indent=1))
    assert r["ready"] and r["start"][0], r["start"]
    res = r["result"]
    assert res and not res["passed"] and res["failed_phase"] == "approach" and res["reason"] == "sanity", res
    assert res["check"] in ("sanity.robot_contacts", "sanity.penetration"), res
    # well before the approach timeout
    assert push["t_fail"][0] <= res["t_fail"] <= push["t_fail"][1], res
    assert res["rate_hz"] >= 95.0, res["rate_hz"]
    assert not problem_lines(sim["launch_log"]), problem_lines(sim["launch_log"])[:5]
