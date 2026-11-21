"""layout_check: the pre-plan chassis clearance on a small fake world (no assets needed), and on the
real fridge scene when assets/ is present."""
import json
import math

import numpy as np
import pytest

import layout_check as L
import specs_bridge
import start_design as S

WORLD = """<?xml version="1.0"?>
<sdf version="1.9"><world name="w">
  <include><name>robot</name><uri>model://robot</uri><pose>0 0 0 0 0 0</pose></include>
  <include><name>bin</name><uri>model://bin</uri><pose>{bin_x} 0 0 0 0 0</pose></include>
  <model name="wall"><static>true</static><pose>-2 0 0 0 0 0</pose>
    <link name="l"><collision name="c"><pose>0 0 1 0 0 0</pose><geometry><box><size>0.1 4 2</size></box></geometry>
    </collision></link></model>
</world></sdf>
"""


def fake_boxes(tmp_path, bin_x=1.0):
    """A wall (inline, x -2), a 0.4 m bin (manifest AABB at x bin_x) and a skipped robot."""
    (tmp_path / "bin").mkdir(exist_ok=True)
    (tmp_path / "bin" / "manifest.json").write_text(json.dumps({"aabb_model_frame": [[-0.2, -0.2, 0], [0.2, 0.2, 0.4]]}))
    (tmp_path / "world.sdf").write_text(WORLD.format(bin_x=bin_x))
    return L.scene_boxes(tmp_path / "world.sdf", lambda uri: tmp_path / uri.replace("model://", ""), skip=["robot"])


PANDA = (0.50, 0.50, 0.60)                 # mobile_panda chassis size_xy, mount z


def at(x=0.0, y=0.0, yaw_deg=0.0, box=(0.0, 0.0)):
    """A layout at the offsets from the box centre, facing the object (what the robot_base randomiser realises)."""
    return {"start_pose": (box[0] + x, box[1] + y, S.FACING_YAW_OBJ + math.radians(yaw_deg))}


def check(boxes, min_m=0.0):
    # object at the origin facing +x: the planner frame is the world, the base faces -x (FACING_YAW_OBJ)
    return L.Clearance(boxes, [0.0, 0.0, 0.0, 0.0], (0.0, 0.0), min_m, 0.03, 3, PANDA)


def test_scene_boxes(tmp_path):
    boxes = {n: (T, s) for n, T, s in fake_boxes(tmp_path)}
    assert set(boxes) == {"wall::c", "bin::aabb"}
    assert boxes["bin::aabb"][0][:3, 3] == pytest.approx([1.0, 0, 0.2])
    assert boxes["bin::aabb"][1] == pytest.approx([0.4, 0.4, 0.4])
    assert boxes["wall::c"][0][:3, 3] == pytest.approx([-2, 0, 1])


def test_clearance_distance_and_reject(tmp_path):
    c = check(fake_boxes(tmp_path))
    half = c.size[0] / 2                       # grown chassis: 0.25 m + 3 deg turn + 3 cm
    assert half == pytest.approx(0.25 * (math.cos(math.radians(3)) + math.sin(math.radians(3))) + 0.03)
    d = check(fake_boxes(tmp_path), min_m=1.0).distances({})                # boxes within 1 m
    assert d["bin::aabb"] == pytest.approx(1.0 - 0.2 - half, abs=c.err + 1e-9)
    assert d["bin::aabb"] <= 1.0 - 0.2 - half                                # a lower bound
    assert d["wall::c"] == pytest.approx(2.0 - 0.05 - half, abs=c.err + 1e-9)
    assert "bin::aabb" not in c.distances({})                                # beyond min_m: skipped
    assert c({}) is None
    assert c(at(0.6)).startswith("collision: chassis")      # into the bin
    assert c(at(0.2)) is None
    gap = 1.0 - 0.2 - half - 0.2
    assert check(fake_boxes(tmp_path), min_m=gap + 0.01)(at(0.2)).endswith(f"(min {gap + 0.01} m)")


def test_clearance_follows_the_object_pose(tmp_path):
    c = L.Clearance(fake_boxes(tmp_path, bin_x=0.0), [0.0, -1.0, 0.0, math.pi / 2], (0.0, 0.0), 0.0, 0.03, 3, PANDA)
    assert c.pose({})[:3, 3] == pytest.approx([0.0, -1.0, 0.3025])   # planner origin -> world (0, -1)
    assert c.pose(at(yaw_deg=10))[:2, 0] == pytest.approx(
        [math.cos(math.pi / 2 + math.pi + math.radians(10)), math.sin(math.pi / 2 + math.pi + math.radians(10))])
    assert c({}) is None
    assert c(at(0.6)) is not None                     # planner +x = world +y: into the bin
    assert c(at(-0.6)) is None


def test_a_box_wholly_inside_the_chassis_is_a_collision():
    T = np.eye(4)
    T[:3, 3] = [0, 0, 0.3]
    assert check([("cup::c", T, np.array([0.01, 0.01, 0.01]))])({}) is not None


ASSETS = L.specs_bridge.ROOT / "assets" / "generated_partnet" / "fridge" / "model.sdf"


@pytest.mark.skipif(not ASSETS.is_file(), reason="scene assets (assets/, not in git) are absent")
def test_real_fridge_scene():
    c = L.clearance_reject("open_fridge_door", "mobile_panda")
    assert c({}) is None
    assert c({**at(-0.17, 0.17, 27, c.box_centre), "grasp_id": 0}) is None
    assert c(at(-0.6, box=c.box_centre)).startswith("collision: chassis")     # pushed into the fridge
    assert tuple(specs_bridge.starts("open_fridge_door", "mobile_panda").centre) == c.box_centre
    assert c.size[2] == pytest.approx(0.595) and c.z == pytest.approx(0.3025)
