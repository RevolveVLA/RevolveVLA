"""T1 bring-up, for every robot of robot_cases.ROBOTS: see conftest.py."""
import json

import robot_cases
from conftest import problem_lines, result

ROBOTS = robot_cases.ROBOTS
EXPECTED_RATE_HZ = 30.0


def test_bringup(sim):
    r = result(sim, "check_bringup.py", "", 180, sim["logs"] / "check_bringup.log")
    (sim["logs"] / "bringup.json").write_text(json.dumps(r, indent=1))
    assert r["controllers"] and all(s == "active" for s in r["controllers"].values()), r["controllers"]
    assert {"walls", "kitchen_counter", "fridge", "table_01", "cup", sim["robot"]} <= set(r["gz_models"]), r
    assert all(abs(q) < 0.01 for js in r["scene_joints"].values() for q in js.values()), r["scene_joints"]
    assert set(r["scene_joints"]) == {"fridge", "kitchen_counter"}, r["scene_joints"]
    assert r["pose_err"]["xy_m"] <= 0.01 and abs(r["pose_err"]["yaw_deg"]) <= 1.0, r["pose_err"]
    for cam, c in r["cameras"].items():       # the render rate from camera_info, the content from a few raw frames
        assert abs(c["rate_hz"] - EXPECTED_RATE_HZ) <= 0.05 * EXPECTED_RATE_HZ, (cam, c)
        assert c["max_gap_s"] <= 3.0 / EXPECTED_RATE_HZ, (cam, c)          # no render stall inside the window
        assert c["n"] >= 2.0 * EXPECTED_RATE_HZ, (cam, c)                  # at least 2 s of frames measured
        assert c["frames"] >= 3 and c["size"] == [480, 480] and c["min_std"] > 1.0, (cam, c)
    assert len(r["cameras"]) == 3
    assert r["contact_topics"] and all(r["contact_topics"].values()), r["contact_topics"]
    assert not problem_lines(sim["launch_log"]), problem_lines(sim["launch_log"])[:5]
