"""T2 control, for every robot of robot_cases.ROBOTS: see conftest.py and check_control.py."""
import json

import robot_cases
from conftest import problem_lines, result

ROBOTS = robot_cases.ROBOTS
SPEED_PCT, HEADING_DEG, ARM_DEG = 5.0, 3.0, 1.0


def test_control(sim):
    case = robot_cases.CONTROL[sim["robot"]]
    r = result(sim, "check_control.py", "", 400, sim["logs"] / "check_control.log")
    (sim["logs"] / "control.json").write_text(json.dumps(r, indent=1))
    for seg, m in r["base"].items():
        if "speed_err_pct" in m:
            assert abs(m["speed_err_pct"]) <= SPEED_PCT and abs(m["heading_err_deg"]) <= HEADING_DEG, (seg, m)
        if "wz_err_pct" in m:
            assert abs(m["wz_err_pct"]) <= SPEED_PCT, (seg, m)
    for name, m in r["arm"].items():
        assert m["result"] == 0 and m["max_err_deg"] <= ARM_DEG, (name, m)
    assert max(map(abs, r["arm_start_err_deg"])) <= ARM_DEG, r["arm_start_err_deg"]
    h = r["hand"]
    assert abs(h["close"]["joint"] - case["closed"]) <= case["hand_tol"], h
    assert abs(h["open"]["joint"] - case["open"]) <= case["hand_tol"], h
    assert r["probe_spawned"], r
    g = h["grasp_box"]
    assert case["grasp"][0] <= g["joint"] <= case["grasp"][1] and all(v > 0 for v in g["contact_msgs"].values()), g
    assert len(g["contact_msgs"]) == 2, g
    assert not problem_lines(sim["launch_log"]), problem_lines(sim["launch_log"])[:5]
