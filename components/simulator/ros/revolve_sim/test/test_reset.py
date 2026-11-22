"""Reset (see conftest.py and check_reset.py), per robot of robot_cases.TASKS, in one sim container: RESETS resets
between back-to-back episodes (moving ones reset during the episode, idle ones after their result). After each
reset: the base within 1 cm / 1 deg of robot_start, the arm within 1 deg of robot_start.arm, the scene joints at
their scene values, every controller active, the evaluator ready again; the next episode's arm move tracks its
target within 1 deg (T2) and every episode has its own result JSON and bag. No leaks or hangs: each reset within
RESET_S, sim time at ~real time, the container's memory and process count stable, no error lines."""
import json
import subprocess

import robot_cases
from conftest import host_path, problem_lines, result, sh

ROBOTS = list(robot_cases.TASKS)
SIM_ARGS = {r: ["--task", c["task"], "--no-autostart"] for r, c in robot_cases.TASKS.items()}
RESETS = 5
RESET_S = 30.0


def usage(name):
    """(memory MiB, process count) of the container."""
    mem = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", name], capture_output=True,
                         text=True).stdout.split("/")[0].strip()
    mib = float(mem[:-3]) * {"KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0}[mem[-3:]]
    return mib, int(sh(name, "ls -d /proc/[0-9]* | wc -l", 30).stdout)


def test_reset_loop(sim):
    task = robot_cases.TASKS[sim["robot"]]["task"]
    before = usage(sim["name"])
    r = result(sim, "check_reset.py", f"--task {task} --mode loop --resets {RESETS}", 900,
               sim["logs"] / "check_reset.log")
    r["usage_before"], r["usage_after"] = before, usage(sim["name"])
    (sim["logs"] / "reset.json").write_text(json.dumps(r, indent=1))
    assert len(r["resets"]) == RESETS and len(r["episodes"]) == RESETS + 1, r
    for rs in r["resets"]:
        assert rs["ok"] and rs["wall_s"] <= RESET_S, rs
        assert rs["base_xy_m"] <= 0.01 and rs["base_yaw_deg"] <= 1.0, rs
        assert rs["arm_err_deg"] <= 1.0, rs
        assert rs["scene_err"] and all(abs(v) <= 0.01 for v in rs["scene_err"].values()), rs["scene_err"]
        assert set(rs["controllers"].values()) == {"active"}, rs["controllers"]
        assert rs["ready"], rs
    bags = []
    for ep in r["episodes"]:
        assert ep["start"][0], ep
        res = ep["result"]
        assert res and res["episode"] == ep["k"] and not res["passed"], ep
        assert res["reason"] == ("reset" if ep["kind"] == "move" else "timeout_s"), res
        assert ep["rtf"] >= 0.9 and res["rate_hz"] >= 95.0, ep
        if ep["kind"] == "move":
            assert ep["arm_result"] == 0 and ep["arm_hold_err_deg"] <= 1.0 and ep["base_moved_m"] >= 0.5, ep
        assert json.loads(host_path(res["result_file"]).read_text())["episode"] == ep["k"]
        assert res["bag"] and host_path(res["bag"]).stat().st_size > 0, res
        info = sh(sim["name"], f"ros2 bag info {res['bag']}", 60).stdout
        assert "Messages:" in info and int(info.split("Messages:")[1].split()[0]) > 0, info
        bags.append(res["bag"])
    assert len(set(bags)) == len(bags), bags
    (mem0, procs0), (mem1, procs1) = r["usage_before"], r["usage_after"]
    assert procs1 <= procs0 + 2 and mem1 <= mem0 * 1.25 + 200, (r["usage_before"], r["usage_after"])
    assert not problem_lines(sim["launch_log"]), problem_lines(sim["launch_log"])[:5]
