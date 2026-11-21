"""start_design's layout path (ROUND_LAYOUT parsing, layout_starts) and entry.py's --layout (no GPU)."""
import json
import math

import pytest

import entry
import start_design as S
import specs_bridge
from datagen.plugins.randomisers.robot_base import RobotBase

SPEC = S.parse_starts(dict(clusters=10, per_cluster=10, box_m=0.40, yaw_half_deg=30, offset_m=0.03, offset_deg=3,
                           nominal_arm="ready"))
BOX = tuple(specs_bridge.starts("open_fridge_door", "mobile_panda").centre)
POSE = (BOX[0] + 0.1, BOX[1] - 0.05, S.FACING_YAW_OBJ + math.radians(12.0))
LAYOUT = {"start_pose": list(POSE), "grasp_id": 3}


def test_layout_from_env():
    assert S.layout_from_env({}) is None
    assert S.layout_from_env({"ROUND_LAYOUT": json.dumps(LAYOUT)}) == LAYOUT
    assert S.layout_from_env({"ROUND_LAYOUT": "{}"}) == {}
    for bad in ("[1]", "nope", '{"robot_base.x_m": 0}', '{"grasp_id": 1.5}', '{"grasp_id": 5}', '{"grasp_id": true}',
                '{"start_pose": [0, 0]}', '{"start_pose": [0, 0, NaN]}', '{"start_pose": [0, 0, "1"]}',
                '{"start_pose": [0, 0, true]}', '{"start_pose": 1}'):
        with pytest.raises(ValueError):
            S.layout_from_env({"ROUND_LAYOUT": bad})


def test_layout_spot_defaults_to_the_box_centre_facing_the_object():
    assert S.layout_spot({}, BOX) == (*BOX, S.FACING_YAW_OBJ)
    assert S.layout_spot(LAYOUT, BOX) == pytest.approx(POSE)


def test_layout_spot_is_what_the_robot_base_randomiser_realises():
    values = {"x_m": 0.1, "y_m": -0.05, "yaw_deg": 12.0}
    pose = RobotBase().realise(values, {"box_centre": BOX})["start_pose"]
    assert S.layout_spot({"start_pose": pose}, BOX) == pytest.approx(POSE)


def test_layout_starts_at_the_spot_with_the_jitter():
    st = S.layout_starts(LAYOUT, 7, 50, SPEC, BOX)
    x, y, yaw = POSE
    assert len(st) == 50 and [s.attempt_id for s in st] == list(range(50))
    for s in st:
        assert s.cluster == 0 and s.grasp_id == 3
        assert 0.01 - 1e-9 <= math.hypot(s.x - x, s.y - y) <= SPEC.offset_m + 1e-9
        assert 1 - 1e-9 <= abs(math.degrees(s.yaw - yaw)) <= SPEC.offset_deg + 1e-9
        assert (s.dx, s.dy) == pytest.approx((s.x - x, s.y - y))
    assert st == S.layout_starts(LAYOUT, 7, 50, SPEC, BOX)                # deterministic per plan seed
    assert [(s.dx, s.dy, s.dyaw) for s in st] == S.start_offsets(7, 50, SPEC)


def test_layout_starts_grasp_cycles_without_a_grasp():
    st = S.layout_starts({}, 0, 12, SPEC, BOX)
    assert [s.grasp_id for s in st] == [a % S.GRASPS for a in range(12)]
    assert {round(s.x - s.dx, 9) for s in st} == {round(BOX[0], 9)}


@pytest.mark.parametrize("dx, dy, dyaw", [(0.18, 0, 0), (0, -0.171, 0), (0, 0, 27.5)])
def test_layout_starts_outside_the_window_is_an_error(dx, dy, dyaw):
    def pose(dx, dy, dyaw):
        return {"start_pose": [BOX[0] + dx, BOX[1] + dy, S.FACING_YAW_OBJ + math.radians(dyaw)]}
    with pytest.raises(ValueError):
        S.layout_starts(pose(dx, dy, dyaw), 0, 5, SPEC, BOX)
    S.layout_starts(pose(0.17, 0, -27.0), 0, 5, SPEC, BOX)                # the edge fits


def test_entry_layout_env():
    assert entry.layout_env(None) == {}
    assert json.loads(entry.layout_env(json.dumps(LAYOUT))["ROUND_LAYOUT"]) == LAYOUT
    with pytest.raises(ValueError):
        entry.layout_env('{"grasp_id": 9}')
    assert entry.layout_env('{"grasp": {"along_m": 1.0}}', "open_fridge_door")      # a grasp by value
    with pytest.raises(ValueError, match="0..4"):
        entry.layout_env('{"grasp_id": 5}', "open_fridge_door")                      # the task has 5 points


def test_entry_plan_refuses_a_bad_layout_before_any_job(capsys, tmp_path):
    assert entry.main(["plan", "--task", "open_fridge_door", "--seed", "0", "--out", str(tmp_path / "p"),
                       "--layout", "[1]"]) == 2
    assert "not a JSON object" in capsys.readouterr().out and not (tmp_path / "p").exists()


def test_entry_finds_the_planner_scripts_and_the_simulator_cli():
    assert (entry.AUTOMOMA / "plan.py").is_file() and (entry.AUTOMOMA / "fk_check.py").is_file()
    assert entry.SIM_CLI.is_file()
