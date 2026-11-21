from pathlib import Path

import pytest
import yaml

from revolve_common import specs
from conftest import SPECS

ALL = sorted(p for p in SPECS.rglob("*.yaml") if p.name != "contract.yaml" and p.relative_to(SPECS).parts[0] != "datagen")


def test_found_every_kind():
    kinds = {p.relative_to(SPECS).parts[0] for p in ALL}
    assert kinds == {"experiments", "tasks", "robots", "scenes"}


ROBOTS = sorted(d.name for d in (SPECS / "robots").iterdir() if d.is_dir())
CASES = [(p, rb) for p in ALL for rb in (ROBOTS if p.relative_to(SPECS).parts[0] == "tasks" else [None])]


@pytest.mark.parametrize("path, robot", CASES, ids=lambda x: str(x.relative_to(SPECS)) if isinstance(x, Path) else str(x))
def test_every_spec_loads(path, robot):
    """Every spec loads; a robot-free task for every robot."""
    assert specs.load(path, robot=robot) is not None


def test_task_is_robot_free_and_takes_per_robot_settings(monkeypatch):
    t = SPECS / "tasks" / "open_drawer" / "task.yaml"
    monkeypatch.delenv(specs.ROBOT_ENV, raising=False)
    with pytest.raises(specs.SpecError, match="robot-free"):
        specs.load(t)
    panda, so101 = specs.load(t, robot="mobile_panda"), specs.load(t, robot="mobile_so101")
    assert panda.task.planner["base_retract"][0] == 1.0 and so101.task.planner["base_retract"][0] == 1.02
    assert panda.task.planner["grasp"] == so101.task.planner["grasp"] and panda.task.phases == so101.task.phases
    monkeypatch.setenv(specs.ROBOT_ENV, "mobile_so101")
    assert specs.load(t).robot.name == "mobile_so101"
    r = specs.load_experiment(SPECS / "experiments" / "drawer_so101_datagen.yaml")
    assert (r.robot.name, r.task.name, r.task.planner["base_retract"][0]) == ("mobile_so101", "open_drawer", 1.02)
    assert specs.load_experiment(SPECS / "experiments" / "drawer_mini.yaml").robot.name == "mobile_panda"


def test_datagen_variant_is_optional_and_forwarded(monkeypatch):
    assert specs.load_experiment(SPECS / "experiments" / "fridge_mini.yaml").experiment.datagen.variant is None
    r = specs.load_experiment(SPECS / "experiments" / "fridge_panda_f1.yaml")
    assert (r.robot.name, r.experiment.datagen.variant) == ("mobile_panda", "drive_seq")
    monkeypatch.setenv(specs.ROBOT_ENV, "mobile_panda")
    monkeypatch.delenv(specs.VARIANT_ENV, raising=False)
    assert specs.robot_env() == {specs.ROBOT_ENV: "mobile_panda"}
    monkeypatch.setenv(specs.VARIANT_ENV, "drive_seq")
    assert specs.robot_env() == {specs.ROBOT_ENV: "mobile_panda", specs.VARIANT_ENV: "drive_seq"}


def test_robot_joints_derive_state_and_action():
    for rb in ROBOTS:
        r = specs.load(SPECS / "robots" / rb / "robot.yaml")
        assert r.joints.state == [*r.joints.base, *r.joints.arm, *r.joints.gripper]
        assert r.joints.action == [*r.joints.base, *r.joints.arm, "gripper"]
        # the policy (contract): base twist in the robot frame, never the world-frame base pose
        assert r.joints.policy_state == ["base_vx", "base_vy", "base_wz", *r.joints.arm, *r.joints.gripper]
        assert r.joints.policy_action == ["base_vx", "base_vy", "base_wz", *r.joints.arm, "gripper"]


def test_experiment_resolves_task_robot_scene():
    r = specs.load_experiment(SPECS / "experiments" / "fridge_mini.yaml")
    assert (r.task.name, r.robot.name, r.scene.name) == ("open_fridge_door", "mobile_panda", "kitchen_partnet")
    assert [p.name for p in r.task.phases] == ["approach", "grasp", "open", "release"]
    assert r.experiment.seeds.heldout.seeds() == [100, 101, 102, 103]
    assert len(r.robot.joints.state) == 12 and r.robot.joints.action[-1] == "gripper"


def edit(path, fn):
    d = yaml.safe_load(path.read_text())
    fn(d)
    path.write_text(yaml.safe_dump(d))


def test_overlapping_seeds_rejected(specs_copy):
    p = specs_copy / "experiments" / "fridge_mini.yaml"
    edit(p, lambda d: d["seeds"].update(heldout=[5, 8]))
    with pytest.raises(specs.SpecError, match=r"seeds\.train and seeds\.heldout overlap: \[5, 6, 7, 8\]"):
        specs.load(p)


def test_reversed_seed_range_rejected(specs_copy):
    p = specs_copy / "experiments" / "fridge_mini.yaml"
    edit(p, lambda d: d["seeds"].update(probe=[203, 200]))
    with pytest.raises(specs.SpecError, match="seeds.probe"):
        specs.load(p)


@pytest.mark.parametrize("change, field", [
    (lambda d: d.update(max_turns=0), "max_turns"),
    (lambda d: d["training"].update(mode="sometimes"), "training.mode"),
    (lambda d: d.update(typo_key=1), "typo_key"),
    (lambda d: d["datagen"].pop("workers"), "datagen.workers"),
])
def test_bad_experiment_names_the_field(specs_copy, change, field):
    p = specs_copy / "experiments" / "fridge_mini.yaml"
    edit(p, change)
    with pytest.raises(specs.SpecError) as e:
        specs.load(p)
    assert f"  {field}" in str(e.value)


@pytest.mark.parametrize("change, needle", [
    (lambda d: d["phases"][2]["pass"]["all"][0].update(metric="door_angle"), "unknown metric 'door_angle'"),
    (lambda d: d["phases"][0]["pass"]["all"][0].update(lt=0.02), "exactly one of"),
    (lambda d: d["phases"][2]["pass"]["all"][0].pop("joint"), "needs joint"),
    (lambda d: d["phases"][1].update(name="approach"), "duplicate names"),
    (lambda d: d["phases"][3].update(end="whenever"), "phases.3.end"),
])
def test_bad_task_names_the_problem(specs_copy, change, needle):
    p = specs_copy / "tasks" / "open_fridge_door" / "task.yaml"
    edit(p, change)
    with pytest.raises(specs.SpecError) as e:
        specs.load(p)
    assert needle in str(e.value)


def test_recovery_phase_must_exist(specs_copy):
    p = specs_copy / "experiments" / "fridge_mini.yaml"
    edit(p, lambda d: d["recovery"].update(phases=["approach", "wiggle"]))
    with pytest.raises(specs.SpecError, match="wiggle"):
        specs.load(p)


def test_unknown_task_or_robot(specs_copy):
    p = specs_copy / "experiments" / "fridge_mini.yaml"
    edit(p, lambda d: d.update(task="open_oven"))
    with pytest.raises(specs.SpecError, match="task 'open_oven'"):
        specs.load(p)
    edit(p, lambda d: d.update(task="open_fridge_door", robot="spot"))
    with pytest.raises(specs.SpecError, match="robot 'spot'"):
        specs.load(p)
    t = specs_copy / "tasks" / "open_drawer" / "task.yaml"
    edit(t, lambda d: d.update(robots={"spot": {}}))
    with pytest.raises(specs.SpecError, match="no robot 'spot'"):
        specs.load(t, robot="mobile_panda")


# ---- task.yaml `starts:` (the start design of a robot)
def starts_of(path, robot, variant=None):
    return specs.load(path, robot=robot).task.starts_for(robot, variant)


def test_starts_accessor_per_method_and_variant():
    fridge, drawer = (SPECS / "tasks" / t / "task.yaml" for t in ("open_fridge_door", "open_drawer"))
    box = starts_of(drawer, "mobile_panda")
    assert (box.method, box.centre) == ("box", [0.98, 0.0])
    assert box.base_limits() == {"x_m": 0.17, "y_m": 0.17, "yaw_deg": 27}
    assert list(box.block()) == ["clusters", "per_cluster", "box_m", "yaw_half_deg", "offset_m", "offset_deg",
                                 "nominal_arm"]
    drive = starts_of(drawer, "mobile_so101_xl")
    assert drive.method == "drive" and drive.block()["mode"] == "drive" and "sequence" not in drive.block()
    assert starts_of(drawer, "mobile_so101") is None
    seq = starts_of(fridge, "mobile_panda", "drive_seq")
    assert seq.method == "drive" and seq.block()["sequence"] == "drive_then_arm" and seq.block()["bearing_deg"] == 40.0
    assert starts_of(fridge, "mobile_panda").method == "box"
    for robot, variant in (("mobile_so101", "drive_seq"), ("mobile_panda", "nope")):
        with pytest.raises(specs.SpecError, match="datagen variant"):
            starts_of(fridge, robot, variant)


def test_reach_method_block():
    t = specs.parse(specs.Task, SPECS / "tasks" / "open_drawer" / "task.yaml")
    reach = dict(method="reach", grasp_m=[-0.1, 0.1], d_m=[0.68, 0.76], yaw_half_deg=8, jitter_m=0.01, jitter_deg=1)
    t = specs.Task.model_validate({**t.model_dump(by_alias=True), "starts": {"mobile_so101": reach}})
    s = t.starts_for("mobile_so101")
    assert s.block() == {"mode": "reach", **{k: v for k, v in reach.items() if k != "method"}}


BOX = dict(method="box", centre=[0.95, -0.05], clusters=10, per_cluster=10, box_m=0.4, yaw_half_deg=30, offset_m=0.03,
           offset_deg=3)
DRIVE = dict(method="drive", grasp_m=[-0.1, 0.1], d_m=[0.68, 0.76], yaw_half_deg=8, jitter_m=0.01, jitter_deg=1,
             back_m=[1.0, 1.6], side_m=[-0.3, 0.3], heading_half_deg=20, slowdown=2)


@pytest.mark.parametrize("block, needle", [
    (dict(BOX, method="jump"), "starts.mobile_panda"),
    ({k: v for k, v in BOX.items() if k != "centre"}, "centre"),
    (dict(BOX, extra=1), "extra"),
    (dict(BOX, box_m=0.5), "box_m"),
    (dict(BOX, clusters=True), "clusters"),
    (dict(BOX, clusters=1.5), "clusters"),
    (dict(BOX, offset_m=0.01, box_m=0.01), "no start fits"),
    (dict(BOX, centre=[0.95]), "centre"),
    (dict(BOX, nominal_arm="folded"), "nominal_arm"),
    (dict(BOX, variants={"v": dict(BOX, variants={})}), "variants"),
    (dict(DRIVE, d_m=[0.8, 0.7]), "d_m"),
    (dict(DRIVE, d_m=[0.0, 0.7]), "d_m"),
    (dict(DRIVE, back_m=[-1.0, 1.6]), "back_m"),
    (dict(DRIVE, slowdown=0.5), "slowdown"),
    (dict(DRIVE, sequence="sideways"), "sequence"),
    ({k: v for k, v in DRIVE.items() if k != "back_m"}, "back_m"),
])
def test_bad_starts_block_names_the_problem(specs_copy, block, needle):
    p = specs_copy / "tasks" / "open_fridge_door" / "task.yaml"
    edit(p, lambda d: d.update(starts={"mobile_panda": block}))
    with pytest.raises(specs.SpecError) as e:
        specs.load(p, robot="mobile_panda")
    assert "starts" in str(e.value) and needle in str(e.value)


def test_starts_robot_must_exist_and_a_drive_needs_a_folded_arm(specs_copy):
    p = specs_copy / "tasks" / "open_fridge_door" / "task.yaml"
    edit(p, lambda d: d.update(starts={"spot": BOX}))
    with pytest.raises(specs.SpecError, match="starts.spot: no robot 'spot'"):
        specs.load(p, robot="mobile_panda")
    edit(p, lambda d: d.update(starts={"mobile_panda": DRIVE}))
    edit(specs_copy / "robots" / "mobile_panda" / "robot.yaml", lambda d: d["arm"].pop("folded", None))
    with pytest.raises(specs.SpecError, match="no arm.folded"):
        specs.load(p, robot="mobile_panda")
