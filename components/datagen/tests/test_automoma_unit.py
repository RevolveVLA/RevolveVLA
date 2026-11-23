"""The `automoma` generator plugin (plugins/automoma/) without cuRobo, GPU or ROS: the plan inputs built from the v2 specs, the
planner's URDF text, the plan as a 30 Hz command series, the base tracker, the host side of the plan run (container command,
failure mapping with fakes) and the pipeline's plan hook. The planner itself (plan.py) and the streaming (execute.py) run on
the real thing: test_automoma_gpu.py."""
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml

import fake_datagen as F
from datagen import cli
from datagen.core import pipeline, registry, spec as S
from datagen.core.errors import DatagenError
from datagen.core.interfaces import Result
from datagen.core.naming import container_prefix
from datagen.plugins.automoma import assets, frames, generator, launch, problem, series, track
from datagen.plugins.automoma.model import Model
from datagen.plugins.runners.sim import SimRunner

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "specs" / "scenes" / "kitchen_partnet" / "scene.yaml"
needs_assets = pytest.mark.skipif(not (REPO / "assets/generated_partnet/fridge/model.sdf").is_file(),
                                  reason="needs assets/ (./revolve assets)")
TIMING = yaml.safe_load(problem.PLANNER.read_text())["timing"]


# ------------------------------------------------------------------ registration + support
def test_the_plugin_registers_as_a_generator():
    cls = registry.get("generator", "automoma")
    assert cls is generator.AutoMoma
    assert isinstance(cls(None, {}), generator.AutoMoma)


def test_unknown_params_and_unsupported_robots_or_tasks_are_refused():
    with pytest.raises(DatagenError, match="unknown params"):
        generator.AutoMoma(None, {"attempts": 3})
    spec = S.parse({"task": "open_drawer", "robot": "mobile_so101_xl", "out": "x", "generator": {"plugin": "automoma"},
                    "splits": {"train": {"target": 1, "pool": {"initial": [0, 1]}}}})
    with pytest.raises(DatagenError, match="not for robot 'mobile_so101_xl' \\+ task 'open_drawer'"):
        generator.AutoMoma(spec, {})


# ------------------------------------------------------------------ plan inputs from the v2 specs
@needs_assets
def test_the_problem_is_built_from_the_scene_instance_task_and_robot():
    p = problem.build(BASE, "open_fridge_door", "mobile_panda", REPO)
    assert (p.joint, p.object, p.handle.collision) == ("top_door_hinge", "fridge", "top_door_handle")
    assert p.goal == pytest.approx(1.047 + TIMING_GOAL_MARGIN)             # the task's `open` threshold + the margin
    assert p.start_base == [-1.75, 1.1, math.pi / 2] and p.tcp == "panda_tcp" and p.arm_dof == 7
    assert p.start_arm == p.folded                                           # the scene start: arm `folded`
    # the planner frame is the fridge's: its start is the world start seen from the fridge (x out of its front)
    assert frames.base_to_world(p.start_base_obj, p.world_pose) == pytest.approx(p.start_base)
    assert not any(n.startswith("fridge::top_door::") for n, _, _ in p.boxes)      # the moving door is not an obstacle
    assert any(n.startswith("fridge::body::") for n, _, _ in p.boxes)


TIMING_GOAL_MARGIN = yaml.safe_load(problem.PLANNER.read_text())["tasks"]["open_fridge_door"]["goal_margin"]


@needs_assets
def test_an_instance_start_moves_the_plan_start(tmp_path):
    scene = yaml.safe_load(BASE.read_text())
    scene["robot_start"] = {"base": [-1.7, 1.2, 1.6], "arm": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]}
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(scene))
    p = problem.build(path, "open_fridge_door", "mobile_panda", REPO)
    assert p.start_base == [-1.7, 1.2, 1.6] and p.start_arm == p.ready


def test_a_task_without_a_joint_goal_is_not_supported():
    task = type("T", (), {"name": "t", "phases": []})
    with pytest.raises(problem.NotSupported, match="no phase condition"):
        problem.task_joint(task)


def test_the_planner_yaml_describes_the_supported_robot_and_task():
    data = yaml.safe_load(problem.PLANNER.read_text())
    assert set(data["robots"]) == {"mobile_panda"} and set(data["tasks"]) == {"open_fridge_door"}
    problem.check_supported("open_fridge_door", "mobile_panda")
    with pytest.raises(problem.NotSupported):
        problem.check_supported("open_drawer", "mobile_panda")


# ------------------------------------------------------------------ frames + URDF text
def test_planner_frame_round_trip_and_the_object_yaw():
    obj = [-1.5, 2.68, 0.0, -math.pi / 2]
    b = [-1.75, 1.1, math.pi / 2]
    assert frames.base_to_world(frames.base_from_world(b, obj), obj) == pytest.approx(b)
    assert frames.base_from_world([obj[0], obj[1], obj[3]], obj) == pytest.approx([0, 0, 0], abs=1e-12)


def test_poses_and_joints_become_urdf_text():
    T = assets.tf((0.1, 0.2, 0.3), (0.4, -0.5, 0.6))
    assert assets.rpy(T) == pytest.approx((0.4, -0.5, 0.6))
    x, y, z, qw, qx, qy, qz = assets.pose7(T)
    assert (x, y, z) == pytest.approx((0.1, 0.2, 0.3)) and qw**2 + qx**2 + qy**2 + qz**2 == pytest.approx(1.0)
    text = assets.urdf("r", [assets.link("a"), assets.link("b", [("c", T, [1, 2, 3])])],
                       [assets.joint("j", "revolute", "a", "b", T, (0, 0, 1), (-1, 1, 2))])
    root = ET.fromstring(text)
    assert [lk.get("name") for lk in root.findall("link")] == ["a", "b"]
    assert root.find("joint/limit").get("upper") == "1" and root.find("link[@name='b']/collision/geometry/box").get("size") == "1 2 3"


@needs_assets
def test_the_object_assets_hold_the_door_chain_and_one_akr_per_grasp(tmp_path):
    p = problem.build(BASE, "open_fridge_door", "mobile_panda", REPO)
    m = p.object_model()
    rob = {"links": [assets.link("world"), assets.link("panda_tcp")], "joints": [],
           "kinematics": {"ee_link": "panda_tcp", "collision_link_names": ["base_link"], "collision_spheres": {},
                          "self_collision_ignore": {}, "self_collision_buffer": {},
                          "cspace": {"joint_names": [f"j{i}" for i in range(12)], "retract_config": [0.0] * 12,
                                     "null_space_weight": [1] * 12, "cspace_distance_weight": [1] * 12}}}
    out = assets.task_object(p, m, rob, tmp_path)
    obj = ET.parse(out["urdf"]).getroot()
    assert {j.get("name") for j in obj.findall("joint")} >= {"top_door_hinge", "top_door_handle_frame"}
    assert len(out["grasps"]) == len(p.grasp_along)
    akr = ET.parse(Path(out["grasps"][0]["akr_yml"]).with_suffix(".urdf")).getroot()
    assert {"akr_grasp", "akr_handle", "top_door_hinge_akr", "akr_anchor"} <= {j.get("name") for j in akr.findall("joint")}
    k = yaml.safe_load(Path(out["grasps"][0]["akr_yml"]).read_text())["robot_cfg"]["kinematics"]
    assert k["ee_link"] == "fridge_akr_anchor" and "top_door_hinge_akr" in k["cspace"]["joint_names"]
    assert out["door_spheres"] <= 216 and k["cspace"]["joint_names"].index("top_door_hinge_akr") == 3 + p.arm_dof


@needs_assets
def test_the_door_spheres_stay_off_the_rest_of_the_object():
    p = problem.build(BASE, "open_fridge_door", "mobile_panda", REPO)
    m = p.object_model()
    spheres = assets.door_spheres(m, p.handle.collision, p.door_spheres)
    rest = [(m.links[ln] @ T, s) for ln, boxes in m.boxes.items() if ln != m.joint.child for _, T, s in boxes]
    panel = spheres[:-6]                                                 # the handle row is last
    gap = min(assets.box_distance((m.links[m.joint.child] @ [*s["center"], 1])[:3], T, size) - s["radius"]
              for s in panel for T, size in rest)
    assert 20 < len(spheres) <= 216 and gap >= p.door_spheres["clearance"] - 1e-9


@needs_assets
def test_the_door_pose_follows_the_joint():
    m = Model.read(REPO / "assets/generated_partnet/fridge/model.sdf", "top_door_hinge")
    closed, open_ = m.door_pose(0.0), m.door_pose(1.0)
    assert closed == pytest.approx(m.links["top_door"])
    R = np.linalg.inv(closed)[:3, :3] @ open_[:3, :3]                    # a turn of 1 rad about the hinge axis
    assert math.acos((np.trace(R) - 1) / 2) == pytest.approx(1.0)


# ------------------------------------------------------------------ the plan as a 30 Hz series
def rows(n, start, end):
    return np.linspace(start, end, n)[:, None] * np.ones(10)


def test_smooth_interpolation_passes_through_its_knots_and_rests_at_the_ends():
    t = np.array([0.0, 1.0, 3.0, 4.0])
    y = np.array([[0.0], [1.0], [2.0], [5.0]])
    assert series.smooth(t, t, y).ravel() == pytest.approx(y.ravel())
    fine = np.linspace(0, 4, 4001)
    v = np.gradient(series.smooth(fine, t, y).ravel(), fine)
    assert abs(v[0]) < 1e-2 and abs(v[-1]) < 1e-2 and np.abs(np.diff(v)).max() < 0.05      # no jumps in velocity


def test_pull_duration_scales_with_the_joint_limits_and_has_a_floor():
    pull = rows(32, 0.0, 0.3)                                            # 0.3 rad in every joint, 0.3 m in x and y
    vel = np.full(10, 0.5)
    assert series.pull_duration(pull, vel, 3.0, 1.0) == pytest.approx(3.0 * 0.3 * np.sqrt(2) / 0.5)    # the base's xy rate dominates
    assert series.pull_duration(pull, vel, 3.0, 100.0) == 100.0


def test_the_series_runs_drive_approach_close_pull_at_the_action_rate():
    drive, grasp = rows(50, 0.0, 1.0), rows(1, 1.2, 1.2)[0]
    pull = np.vstack([grasp + np.linspace(0, 1, 32)[:, None] * 0.5])
    s = series.build(drive, 0.05, pull, (0.04, -0.04), TIMING, 8.0)
    ev, n = s["events"], len(s["base"])
    assert s["rate_hz"] == 30 and n == int(round(ev["t_end"] * 30)) + 1
    assert ev["t_stage"] == pytest.approx(49 * 0.05 * TIMING["drive_slowdown"])
    assert ev["t_pull"] == pytest.approx(ev["t_close"] + TIMING["close_s"] + TIMING["hold_s"])
    assert s["base"].shape == (n, 3) and s["arm"].shape == (n, 7) and s["hand"].shape == (n,)
    at = lambda t: int(round(t * 30))                                                          # noqa: E731
    assert s["base"][0] == pytest.approx(drive[0, :3]) and s["arm"][0] == pytest.approx(drive[0, 3:])
    assert s["arm"][at(ev["t_stage"])] == pytest.approx(drive[-1, 3:], abs=2e-3)                # staging pose reached (to the sample)
    assert s["arm"][at(ev["t_close"])] == pytest.approx(pull[0, 3:], abs=1e-6)                  # on the grasp pose to close
    assert s["arm"][-1] == pytest.approx(pull[-1, 3:]) and s["base"][-1] == pytest.approx(pull[-1, :3])
    assert s["hand"][at(ev["t_close"])] == pytest.approx(0.04) and s["hand"][at(ev["t_pull"])] == pytest.approx(-0.04)
    assert np.isfinite(s["base"]).all() and np.isfinite(s["arm"]).all() and np.isfinite(s["hand"]).all()
    jump = np.abs(np.diff(np.hstack([s["base"], s["arm"]]), axis=0)).max()
    assert jump < 0.1                                                                           # smooth references at 30 Hz


def test_the_series_is_json_able_after_rounding():
    s = series.build(rows(5, 0.0, 1.0), 0.05, rows(4, 1.0, 2.0), (0.04, -0.04), TIMING, 2.0)
    text = json.dumps({"base": np.round(s["base"], 5).tolist(), "events": s["events"]})
    assert json.loads(text)["events"]["t_end"] == pytest.approx(s["events"]["t_end"])


# ------------------------------------------------------------------ the base tracker
def test_the_twist_follows_the_path_in_the_body_frame_and_corrects_the_pose():
    ref, nxt = (0.0, 0.0, math.pi / 2), (0.0, 0.3 / 30, math.pi / 2)           # 0.3 m/s along +y, base facing +y
    assert track.twist(ref, nxt, ref, 1 / 30, 2.0, 0.4, 0.6) == pytest.approx([0.3, 0.0, 0.0])      # forward = body x
    off = track.twist(ref, nxt, (0.1, 0.0, math.pi / 2), 1 / 30, 2.0, 0.4, 0.6)                    # 0.1 m to the right
    assert off[1] == pytest.approx(0.2) and off[0] == pytest.approx(0.3)      # strafes left
    fast = track.twist(ref, (0.0, 1.0, math.pi / 2), ref, 1 / 30, 2.0, 0.4, 0.6)
    assert np.linalg.norm(fast[:2]) == pytest.approx(0.4)                      # clamped


def test_the_yaw_error_takes_the_short_way_and_is_clamped():
    w = track.twist((0, 0, math.radians(179)), (0, 0, math.radians(179)), (0, 0, math.radians(-179)), 1 / 30, 2.0, 0.4, 0.6)[2]
    assert w == pytest.approx(-2.0 * math.radians(2))                 # +179 deg ref vs -179 deg pose: turn -2 deg
    assert track.twist((0, 0, 3.0), (0, 0, 3.0), (0, 0, 0.0), 1 / 30, 2.0, 0.4, 0.6)[2] == 0.6


def test_the_slew_limit_bounds_the_twist_change_per_tick():
    out = track.slew([0, 0, 0], [0.3, -0.3, 0.0], 0.5, 1 / 30)
    assert out == pytest.approx([0.5 / 30, -0.5 / 30, 0.0])
    assert track.slew([0.29, 0, 0], [0.3, 0, 0], 0.5, 1 / 30) == pytest.approx([0.3, 0, 0])


# ------------------------------------------------------------------ host side: container command, plan results
def test_the_plan_container_mounts_the_repo_read_only_and_the_scene_folder_writable(tmp_path):
    repo = tmp_path / "repo"
    folder = repo / "runs" / "x" / "scene_3"
    folder.mkdir(parents=True)
    argv = launch.argv("dg-abc123-plan-1", folder / "scene.yaml", "open_fridge_door", "mobile_panda", folder, repo)
    assert argv[argv.index("--name") + 1] == "dg-abc123-plan-1" and "--gpus" in argv
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert f"{repo}:/repo:ro" in mounts and f"{folder}:/repo/runs/x/scene_3" in mounts
    assert argv[argv.index("--instance") + 1] == "/repo/runs/x/scene_3/scene.yaml"
    assert argv[argv.index("--out") + 1] == "/repo/runs/x/scene_3" and argv[argv.index("-m") + 1] == "datagen.plugins.automoma.plan"


def test_the_plan_container_is_named_by_the_runs_prefix():
    spec = S.parse({"task": "t", "robot": "r", "out": "runs/x", "generator": {"plugin": "automoma"},
                    "splits": {"train": {"target": 1, "pool": {"initial": [0, 1]}}}})
    assert SimRunner(spec, {}).prefix == container_prefix(spec.out)      # the runner's interrupt removes `<prefix>-*`


@pytest.fixture
def gen(tmp_path):
    spec = S.parse({"task": "open_fridge_door", "robot": "mobile_panda", "out": str(tmp_path / "out"),
                    "generator": {"plugin": "automoma"}, "splits": {"train": {"target": 1, "pool": {"initial": [0, 1]}}}})
    return generator.AutoMoma(spec, {})


def scene(tmp_path):
    folder = tmp_path / "out" / "train" / "scene_4"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "scene.yaml"


def test_a_plan_the_container_wrote_is_ok(gen, tmp_path, monkeypatch):
    calls = []

    def run(prefix, instance, task, robot, folder, log, timeout):
        calls.append((prefix, task, robot, log.name, timeout))
        (folder / "plan.json").write_text(json.dumps({"ok": True}))
        return 0
    monkeypatch.setattr(launch, "run", run)
    assert gen.plan(scene(tmp_path), "open_fridge_door", "mobile_panda").ok
    assert calls == [(container_prefix(gen.spec.out), "open_fridge_door", "mobile_panda", "scene_4.plan.log", 900.0)]


def test_a_failed_plan_is_the_scenes_reason_and_is_not_retried(gen, tmp_path, monkeypatch):
    n = []

    def run(prefix, instance, task, robot, folder, log, timeout):
        n.append(1)
        (folder / "plan.json").write_text(json.dumps({"ok": False, "reason": "grasp 0: no collision-free whole-body IK"}))
        return 3
    monkeypatch.setattr(launch, "run", run)
    inst = scene(tmp_path)
    res = gen.plan(inst, "open_fridge_door", "mobile_panda")
    assert not res.ok and res.reason == "grasp 0: no collision-free whole-body IK"
    assert gen.plan(inst, "open_fridge_door", "mobile_panda") == res and len(n) == 1       # deterministic: the stored failure stands


def test_an_existing_ok_plan_is_reused(gen, tmp_path, monkeypatch):
    inst = scene(tmp_path)
    (inst.parent / "plan.json").write_text(json.dumps({"ok": True}))
    monkeypatch.setattr(launch, "run", lambda *a: pytest.fail("the planner ran again"))
    assert gen.plan(inst, "open_fridge_door", "mobile_panda").ok


def test_a_container_that_cannot_run_fails_the_scene_with_its_log(gen, tmp_path, monkeypatch):
    def run(*a):
        raise launch.PlanError("the plan container exited with 125; see x.log")
    monkeypatch.setattr(launch, "run", run)
    res = gen.plan(scene(tmp_path), "open_fridge_door", "mobile_panda")
    assert not res.ok and "exited with 125" in res.reason


def test_drive_without_a_plan_ends_the_try_with_a_note(gen, tmp_path):
    gen.io = object()
    out = gen.drive(None, scene(tmp_path), "open_fridge_door", "mobile_panda")
    assert out == {"expert": generator.EXPERT, "notes": "error: no plan"}


# ------------------------------------------------------------------ the pipeline's plan hook
RAW = {"task": "t", "robot": "r", "out": "unused", "generator": {"plugin": "fake_datagen"}, "runner": {"plugin": "fake"},
       "splits": {"train": {"target": 2, "pool": {"initial": [0, 1], "replacement": [2, 5]}}}}


def test_a_failed_plan_rejects_the_scene_without_a_sim_and_the_pool_replaces_it(tmp_path, monkeypatch):
    F.CALLS.clear()
    seen = []

    def plan(self, instance, task, robot):
        seen.append((F.seed_of(instance), task, robot))
        return Result.fail("no IK") if F.seed_of(instance) == 1 else Result()
    monkeypatch.setattr(F.FakeGenerator, "plan", plan, raising=False)
    spec = S.parse(RAW)
    assert pipeline.Pipeline(spec, cli.build_plugins(spec, pipeline.STEPS), tmp_path, log=lambda m: None).run(["sample", "generate"])
    rec = json.loads((tmp_path / "state.json").read_text())["splits"]["train"]
    assert [e["config"]["seed"] for e in rec["used"]] == [0, 2]
    assert rec["rejected"] == [{"seed": 1, "step": "generate", "reason": "plan: no IK"}]
    assert sorted(s for k, s in F.CALLS if k == "drive") == [0, 2]                 # the failed plan never reached the sim
    assert (1, "t", "r") in seen and len(seen) == 3
