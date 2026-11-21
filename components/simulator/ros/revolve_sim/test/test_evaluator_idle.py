"""T3 evaluator, idle episode (see conftest.py and check_evaluator.py): with no command, open_fridge_door fails phase
`approach` on its timeout_s; before it, the evaluator's TCP-to-handle distance matches the scene geometry."""
import json

from conftest import host_path, problem_lines, result

ROBOTS = ["mobile_panda"]
TASK = "open_fridge_door"
SIM_ARGS = ["--task", TASK, "--no-autostart"]
TIMEOUT_S = 20.0                   # the task's approach timeout_s


def test_idle_episode(sim):
    r = result(sim, "check_evaluator.py", f"--task {TASK} --mode idle", 240, sim["logs"] / "check_idle.log")
    (sim["logs"] / "evaluator_idle.json").write_text(json.dumps(r, indent=1))
    assert r["ready"] and r["start"][0], r["start"]
    # the plumbing: the evaluator's distance (gz PosePublisher, tf) vs the static scene geometry, within 1 cm
    assert abs(r["distance_evaluator"] - r["distance_geometry"]) <= 0.01, r
    v = r["state_before"]["values"]
    assert v["grasp.done[0]"] == 0 and abs(v["open.done[0]"]) < 0.01, v          # no finger contacts, door closed
    assert all(v[f"sanity.{k}"] == 0 for k in ("penetration", "robot_contacts", "finite_cmds", "gz_log")), v
    assert v["sanity.body_speed"] is not None and v["sanity.body_speed"] < 0.1, v
    res = r["result"]
    assert res and not res["passed"] and res["failed_phase"] == "approach" and res["reason"] == "timeout_s", res
    assert abs(res["t_fail"] - TIMEOUT_S) <= 0.05, res
    assert [p["status"] for p in res["phases"]] == ["failed", "not_reached", "not_reached"], res["phases"]
    assert res["rate_hz"] >= 95.0, res["rate_hz"]
    saved = json.loads(host_path(res["result_file"]).read_text())
    assert saved["reason"] == "timeout_s" and saved["episode"] == res["episode"]
    assert res["bag"] and host_path(res["bag"]).stat().st_size > 0, res["bag"]
    assert not problem_lines(sim["launch_log"]), problem_lines(sim["launch_log"])[:5]
