"""plugins/randomisers: robot_base + grasp (limits, realisation, registry discovery, spec rejection), the hard limits
against what the planner realises, and a test-only randomiser added without touching core."""
import math
import subprocess
import sys

import pytest

import specs_bridge
import start_design
from datagen import cli
from datagen.core import registry, spec as S
from datagen.core.interfaces import Config, Randomiser
from datagen.core.pipeline import Pipeline
from datagen.plugins.randomisers import robot_base as RB
from datagen.plugins.randomisers.design import StartDesignError

BOX = (0.95, -0.05)


def raw(randomisers):
    return {"task": "open_fridge_door", "robot": "mobile_panda", "out": "unused",
            "generator": {"plugin": "fake_datagen"}, "randomisers": randomisers,
            "splits": {"train": {"target": 4, "sampler": {"strategy": "stratified"}, "pool": {"initial": [0, 9]}}}}


SPEC = S.parse(raw({}))


def limits(name, spec=SPEC):
    return registry.create("randomiser", name).limits(spec)


def test_robot_base_limits():
    assert limits("robot_base") == {"robot_base.x_m": S.Range(-0.17, 0.17), "robot_base.y_m": S.Range(-0.17, 0.17),
                                    "robot_base.yaw_deg": S.Range(-27.0, 27.0)}


def test_grasp_limits():
    assert limits("grasp") == {"grasp.id": S.Choices((0, 1, 2, 3, 4))}


def test_hard_limits_come_from_the_tasks_start_design():
    """task.yaml starts (box and yaw window) shrunk by the attempt jitter; the planner's grasp count."""
    lim = limits("robot_base")
    s = specs_bridge.starts("open_fridge_door", "mobile_panda").block()
    assert math.isclose(s["box_m"] / 2 - s["offset_m"], lim["robot_base.x_m"].hi)
    assert math.isclose(s["yaw_half_deg"] - s["offset_deg"], lim["robot_base.yaw_deg"].hi)
    drawer = S.parse(dict(raw({}), task="open_drawer"))
    assert limits("robot_base", drawer) == limits("robot_base")
    assert limits("grasp")["grasp.id"].values == tuple(range(start_design.GRASPS))


@pytest.mark.parametrize("task, robot, needle", [
    ("open_drawer", "mobile_so101", "no start design"),
    ("open_drawer", "mobile_so101_xl", "mobile_so101_xl on open_drawer uses drive: use drive_start, not robot_base"),
])
def test_robot_base_needs_a_box_start_design(task, robot, needle):
    with pytest.raises(StartDesignError, match=needle):
        limits("robot_base", S.parse(dict(raw({}), task=task, robot=robot)))


def test_robot_base_realises_a_start_pose_from_the_box_centre():
    r = registry.create("randomiser", "robot_base").realise({"x_m": 0.05, "y_m": -0.1, "yaw_deg": 18.0},
                                                            {"box_centre": BOX})
    assert r["offset"] == (0.05, -0.1, pytest.approx(math.radians(18.0)))
    assert r["start_pose"] == (pytest.approx(1.0), pytest.approx(-0.15), pytest.approx(math.pi + math.radians(18.0)))


def test_robot_base_dims_not_varied_are_zero_and_facing_is_overridable():
    r = registry.create("randomiser", "robot_base").realise({"x_m": 0.1}, {"box_centre": BOX, "facing": 0.0})
    assert r == {"offset": (0.1, 0.0, 0.0), "start_pose": (pytest.approx(1.05), -0.05, 0.0)}


def test_grasp_realises_its_id():
    g = registry.create("randomiser", "grasp")
    assert g.realise({"id": 3}, {}) == {"grasp_id": 3}
    assert g.realise({}, {}) == {}


def test_config_of_splits_values_by_randomiser():
    c = Config("train", 7, (), {"robot_base.x_m": 0.1, "robot_base.yaw_deg": 5, "grasp.id": 2, "grasp_x.id": 9})
    assert c.of("robot_base") == {"x_m": 0.1, "yaw_deg": 5} and c.of("grasp") == {"id": 2} and c.of("none") == {}


@pytest.mark.parametrize("randomisers, needle", [
    ({"robot_base": {"x_m": [-0.18, 0.1]}}, "randomisers.robot_base.x_m"),
    ({"robot_base": {"yaw_deg": [-30, 30]}}, "randomisers.robot_base.yaw_deg"),
    ({"robot_base": {"z_m": [0, 0.1]}}, "unknown dim"),
    ({"grasp": {"id": {"choices": [0, 5]}}}, "randomisers.grasp.id"),
])
def test_spec_outside_the_hard_limits_is_rejected(randomisers, needle):
    s = S.parse(raw(randomisers))
    with pytest.raises(S.SpecError, match=needle):
        S.check_limits(s, {k: d for n in s.randomisers for k, d in limits(n, s).items()})


def test_spec_at_the_hard_limits_is_accepted():
    s = S.parse(raw({"robot_base": {"x_m": [-0.17, 0.17], "yaw_deg": [-27, 27]}, "grasp": {"id": {"choices": [0, 4]}}}))
    S.check_limits(s, {k: d for n in s.randomisers for k, d in limits(n, s).items()})


def test_sampling_and_realising_are_torch_free():
    code = ("import sys; from datagen.core import registry; "
            "[registry.create('randomiser', n).realise({}, {'box_centre': (0, 0)}) for n in ('robot_base', 'grasp', 'reach_start', 'drive_start')]; "
            "assert not {'torch', 'numpy'} & set(sys.modules), sorted(sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True, env={"PYTHONPATH": ":".join(p for p in sys.path if p)})


def test_a_new_randomiser_needs_no_core_edit(tmp_path):
    """Defined and registered here only; the spec, the pipeline's limit check and the sampler pick it up as is."""
    @registry.register("randomiser", "lighting")
    class Lighting(Randomiser):
        def limits(self, spec):
            return {"lighting.lux": S.Range(100.0, 1000.0)}

        def realise(self, values, context):
            return {"intensity": values["lux"] / 1000.0}
    try:
        spec = S.parse(raw({"lighting": {"lux": [300, 800]}, "grasp": {"id": {"choices": [1, 2]}}}))
        pipe = Pipeline(spec, cli.build_plugins(spec, ("sample",)), tmp_path, log=lambda m: None)
        assert pipe.run(["sample"])
        sampled = pipe.manifest.data["splits"]["train"]["sampled"]
        assert len(sampled) == 10
        c = Config.from_dict(sampled[0])
        assert 300 <= c.of("lighting")["lux"] <= 800
        assert registry.create("randomiser", "lighting").realise(c.of("lighting"), {}) == {
            "intensity": c.of("lighting")["lux"] / 1000.0}
        with pytest.raises(S.SpecError, match="outside the hard limits"):
            Pipeline(S.parse(raw({"lighting": {"lux": [50, 800]}})),
                     cli.build_plugins(S.parse(raw({"lighting": {"lux": [50, 800]}})), ("sample",)), tmp_path)
    finally:
        registry._PLUGINS.pop(("randomiser", "lighting"))
