"""PRD 11: a task without a grasp randomiser plans exactly as before the grasp modes. data/grasp_before.json was written
from the code before (task.yaml `y|z: [...]` keys): the task's grasp block per (task, robot), the planner's grasp poses
(gen_assets.task_object, a stub robot: the poses depend on the task and the object only), the grasp narrowed to a
reach sample's value, and the box starts (cluster / layout grasps). Needs the generated kitchen assets for the poses."""
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

import pytest

import grasp_design
import specs_bridge
import start_design as D

REPO = Path(__file__).resolve().parents[3]
BEFORE = json.loads((Path(__file__).parent / "data" / "grasp_before.json").read_text())
ASSETS = pytest.mark.skipif(not (REPO / "assets" / "generated_partnet" / "kitchen_counter").is_dir(),
                            reason="generated kitchen assets missing")


def stub_robot():
    """The robot part task_object takes (links, joints, kinematics cfg): empty, the grasp poses do not read it."""
    n = [1] * 12
    return dict(links=[], joints=[], kinematics=dict(ee_link="ee", collision_link_names=[], collision_spheres={},
                self_collision_ignore={}, self_collision_buffer={},
                cspace=dict(joint_names=["a"] * 12, retract_config=[0] * 12, null_space_weight=n, cspace_distance_weight=n)))


def poses(robot, task, monkeypatch, narrow=None):
    monkeypatch.setenv("REVOLVE_ROBOT", robot)
    if str(REPO / "components/simulator/src/sim") not in sys.path:
        sys.path.append(str(REPO / "components/simulator/src/sim"))
    import gen_assets                  # imports the simulator's sdf / robots
    t = specs_bridge.load("task", task)
    if narrow is not None:
        t["grasp"] = grasp_design.narrowed(t["grasp"], grasp_design.fixed(t["grasp"], narrow))
    o = gen_assets.task_object(specs_bridge.load("robot", robot), t, stub_robot(), Path(tempfile.mkdtemp()))
    return json.loads(json.dumps({"axis": o["grasp_axis"], "grasps": {
        str(g): {"pose_obj": v["pose_obj"], "T_handle_grasp": v["T_handle_grasp"], "at": v[o["grasp_axis"]]}
        for g, v in o["grasps"].items()}}))


@pytest.mark.parametrize("key", sorted(BEFORE["task_grasp"]))
def test_the_tasks_grasp_block_is_that_of_before(key):
    task, robot = key.split("/")
    g, want = specs_bridge.grasp(task, robot), BEFORE["task_grasp"][key]
    assert (g.axis, g.points, g.depth, g.roll_deg) == (want["axis"], want["values"], want["depth"], want["roll_deg"])


@ASSETS
@pytest.mark.parametrize("key", sorted(BEFORE["poses"]))
def test_the_planner_grasp_poses_are_those_of_before(key, monkeypatch):
    task, robot = key.split("/")[0], key.split("/")[1]
    assert poses(robot, task, monkeypatch) == BEFORE["poses"][key]


@ASSETS
@pytest.mark.parametrize("seed", sorted(BEFORE["narrowed"]))
def test_the_grasp_narrowed_to_a_reach_sample_is_that_of_before(seed, monkeypatch):
    want = BEFORE["narrowed"][seed]
    monkeypatch.setenv("REVOLVE_ROBOT", "mobile_so101_xl")
    block = specs_bridge.starts("open_drawer", "mobile_so101_xl").block()
    value = D.reach_sample(int(seed), D.parse_starts(block)).grasp
    assert value == want["value"]
    assert poses("mobile_so101_xl", "open_drawer", monkeypatch, narrow=value) == {k: want[k] for k in ("axis", "grasps")}


@pytest.mark.parametrize("task", sorted(BEFORE["starts"]))
def test_the_box_starts_and_their_grasps_are_those_of_before(task):
    want = BEFORE["starts"][task]
    design = specs_bridge.starts(task, "mobile_panda")
    spec, box = D.parse_starts(design.block()), tuple(design.centre)
    asdict = lambda starts: json.loads(json.dumps([dataclasses.asdict(s) for s in starts]))     # noqa: E731
    assert [D.grasp_of_cluster(c) for c in range(23)] == want["grasp_of_cluster"]
    for seed in (0, 1, 7, 49):
        assert asdict(D.plan_starts(3, seed, 20, spec, box)) == want[f"plan_starts/{seed}"]
        assert asdict(D.request_starts([0, 1, 2, 7, 5, 6, 9, 3, 4, 8, 1, 2], list(range(10, 20)), seed, spec, box)) \
            == want[f"request_starts/{seed}"]
        for name, layout in (("empty", {}), ("pose", {"start_pose": [box[0] + 0.05, box[1] - 0.04, 3.2]}),
                             ("grasp3", {"grasp_id": 3}), ("both", {"start_pose": [box[0], box[1], 3.1], "grasp_id": 4})):
            assert asdict(D.layout_starts(layout, seed, 12, spec, box)) == want[f"layout_starts/{seed}/{name}"]
            assert asdict(D.layout_starts(layout, seed, 12, spec, box, grasps=5)) == want[f"layout_starts/{seed}/{name}"]
