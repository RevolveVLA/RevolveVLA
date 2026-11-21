"""MoveIt smoke (see conftest.py and check_moveit.py), for the robots with a robot_cases.MOVEIT entry: MoveIt 2 plans
the arm to TCP pose targets (OMPL, the robot's kinematics plugin) and a Pilz LIN; each is executed on the arm's JTC
and the TCP ends within POS_MM of its target."""
import json

import robot_cases
from conftest import problem_lines, result

ROBOTS = list(robot_cases.MOVEIT)
POS_MM = 5.0


def test_moveit(sim):
    r = result(sim, "check_moveit.py", "", 300, sim["logs"] / "check_moveit.log")
    (sim["logs"] / "moveit.json").write_text(json.dumps(r, indent=1))
    assert "error" not in r, r["error"]
    assert len(r["targets"]) == len(robot_cases.MOVEIT[sim["robot"]]["targets"]) + 1, r["targets"]
    for name, m in r["targets"].items():
        assert m["planned"] and m["executed"] and m["pos_err_mm"] <= POS_MM, (name, m)
    assert not problem_lines(sim["launch_log"]), problem_lines(sim["launch_log"])[:5]
