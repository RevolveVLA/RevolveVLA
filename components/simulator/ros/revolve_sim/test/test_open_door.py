"""T4 positive episode (see conftest.py and check_open_door.py): MoveIt 2 plans the arm (the robot's moveit/ config),
the base gets open-loop ramped TwistStamped segments; open_fridge_door passes approach, grasp and open (the door held
>= 60 deg for 1 s with both fingers on the bar and the TCP slip within its bound). Slow tier (~2 min):
`./revolve test simulator -- -m "not slow"` skips it."""
import json

import pytest
from conftest import host_path, problem_lines, result

TASK = "open_fridge_door"
SIM_ARGS = ["--task", TASK, "--no-autostart"]


@pytest.mark.slow
def test_open_door(sim):
    r = result(sim["name"], "check_open_door.py", f"--task {TASK}", 300, sim["logs"] / "check_open_door.log")
    (sim["logs"] / "open_door.json").write_text(json.dumps(r, indent=1))
    assert r["ready"] and r["start"][0], r["start"]
    assert "error" not in r, r["error"]
    res = r["result"]
    assert res, "no episode result"
    assert res["passed"], {k: res.get(k) for k in ("failed_phase", "reason", "check", "value", "t_fail")}
    assert [p["status"] for p in res["phases"]] == ["passed", "passed", "passed"], res["phases"]
    assert r["open_trace"]["door_max"] >= 1.047 and r["open_trace"]["contacts_min"] == 2, r["open_trace"]
    assert res["rate_hz"] >= 95.0, res["rate_hz"]
    assert json.loads(host_path(res["result_file"]).read_text())["passed"]
    assert not problem_lines(sim["launch_log"]), problem_lines(sim["launch_log"])[:5]
