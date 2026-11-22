"""T4 positive episode (see conftest.py), per robot of robot_cases.TASKS with its check script (check_open_door.py,
check_open_drawer.py): MoveIt 2 plans the arm (the robot's moveit/ config), the base gets open-loop ramped
TwistStamped segments; the task passes approach, grasp and open (the door / drawer held open for 1 s with both
fingers on the bar and the TCP slip within its bound). Then one /simulator/reset (check_reset.py --mode once): the
opened door / drawer back at its scene value, the robot at robot_start, the controllers active. Slow tier (~2 min each): `./revolve test simulator -- -m "not
slow"` skips it."""
import json

import pytest
import robot_cases
from conftest import host_path, problem_lines, result

ROBOTS = list(robot_cases.TASKS)
SIM_ARGS = {r: ["--task", c["task"], "--no-autostart"] for r, c in robot_cases.TASKS.items()}


@pytest.mark.slow
def test_positive(sim):
    case = robot_cases.TASKS[sim["robot"]]
    script = case["positive"]
    r = result(sim, script, f"--task {case['task']}", 300, sim["logs"] / script.replace(".py", ".log"))
    (sim["logs"] / script.replace(".py", ".json").replace("check_", "")).write_text(json.dumps(r, indent=1))
    assert r["ready"] and r["start"][0], r["start"]
    assert "error" not in r, r["error"]
    res = r["result"]
    assert res, "no episode result"
    assert res["passed"], {k: res.get(k) for k in ("failed_phase", "reason", "check", "value", "t_fail")}
    assert [p["status"] for p in res["phases"]] == ["passed", "passed", "passed"], res["phases"]
    assert r["open_trace"]["joint_max"] >= case["open_ge"] and r["open_trace"]["contacts_min"] == 2, r["open_trace"]
    assert res["rate_hz"] >= 95.0, res["rate_hz"]
    assert json.loads(host_path(res["result_file"]).read_text())["passed"]
    # the reset from the end of the positive episode: the opened joint (the robot holding it) back to the scene's
    rr = result(sim, "check_reset.py", f"--task {case['task']} --mode once", 120, sim["logs"] / "check_reset.log")
    (sim["logs"] / "reset_after_positive.json").write_text(json.dumps(rr, indent=1))
    assert max(abs(v) for v in rr["scene_before"].values()) >= case["open_ge"] * 0.9, rr["scene_before"]
    rs = rr["resets"][0]
    assert rs["ok"] and all(abs(v) <= 0.01 for v in rs["scene_err"].values()), rs
    assert rs["base_xy_m"] <= 0.01 and rs["base_yaw_deg"] <= 1.0 and rs["arm_err_deg"] <= 1.0, rs
    assert set(rs["controllers"].values()) == {"active"} and rs["ready"], rs
    assert not problem_lines(sim["launch_log"]), problem_lines(sim["launch_log"])[:5]
