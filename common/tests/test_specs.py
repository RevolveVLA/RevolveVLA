import pytest
import yaml

from revolve_common import specs
from conftest import SPECS

ALL = sorted(p for p in SPECS.rglob("*.yaml") if p.name != "contract.yaml")


def test_found_every_kind():
    kinds = {p.relative_to(SPECS).parts[0] for p in ALL}
    assert kinds == {"experiments", "tasks", "robots", "scenes"}


@pytest.mark.parametrize("path", ALL, ids=lambda p: str(p.relative_to(SPECS)))
def test_every_spec_loads(path):
    assert specs.load(path) is not None


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
    t = specs_copy / "tasks" / "open_drawer" / "task.yaml"
    edit(t, lambda d: d.update(robot="spot"))
    with pytest.raises(specs.SpecError, match="robot 'spot'"):
        specs.load(t)
