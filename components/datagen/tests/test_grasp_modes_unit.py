"""PRD 11: the grasp is the task's defined points OR a continuous range: the `grasp` randomiser's two modes (limits from
the task's planner.grasp, mode errors, realised layout), the layout the planner takes (parse, starts, one grasp by value
for every start method), the pipeline's sample / validation for a box and a drive robot."""
import dataclasses
import json
import math

import pytest

import grasp_design as G
import specs_bridge
import start_design as D
from datagen import cli
from datagen.core import registry, spec as S
from datagen.core.interfaces import Config
from datagen.core.pipeline import Pipeline
from datagen.plugins.automoma.layout import Realiser
from datagen.plugins.randomisers import listing
from datagen.plugins.randomisers.design import StartDesignError
from revolve_common import specs

DRAWER = dict(axis="y", points=[-0.2, -0.1, 0.0, 0.1, 0.2], range=[-0.2, 0.2], depth=0.0, roll_deg=-90.0)
RICH = dict(DRAWER, depth_range=[0.0, 0.03], roll_range_deg=[-100.0, -80.0])
BOX = specs_bridge.starts("open_fridge_door", "mobile_panda")
REACH = specs_bridge.starts("open_drawer", "mobile_so101_xl")


def make_spec(randomisers, task="open_drawer", robot="mobile_panda", **gen):
    return S.parse({"task": task, "robot": robot, "out": "unused", "generator": {"plugin": "automoma", **gen},
                    "randomisers": randomisers,
                    "splits": {"train": {"target": 4, "sampler": {"strategy": "sobol"}, "pool": {"initial": [0, 9]}}}})


def limits(randomisers, **kw):
    return registry.create("randomiser", "grasp").limits(make_spec(randomisers, **kw))


@pytest.fixture
def rich(monkeypatch):
    """A task whose grasp has depth and roll ranges too (no shipped task does)."""
    monkeypatch.setattr("datagen.specs_bridge.grasp", lambda task, robot=None: specs.Grasp(**RICH))


# ---- grasp_design: the grasp by value
def test_fixed_takes_the_tasks_depth_and_roll():
    assert G.fixed(DRAWER, 0.05) == {"along": 0.05, "depth": 0.0, "roll_deg": -90.0}


def test_a_range_value_keeps_the_fixed_depth_and_roll_it_does_not_name():
    assert G.given({"grasp": {"along_m": 0.13}}, DRAWER) == {"along": 0.13, "depth": 0.0, "roll_deg": -90.0}
    assert G.given({"grasp": {"along_m": -0.2}}, DRAWER)["along"] == -0.2           # the edge
    assert G.given({"grasp": {"along_m": 0.1, "depth_m": 0.02, "roll_deg": -85.0}}, RICH) == {
        "along": 0.1, "depth": 0.02, "roll_deg": -85.0}


@pytest.mark.parametrize("value, needle", [({"along_m": 0.21}, "along_m"), ({"along_m": -0.3}, "along_m"),
                                           ({"along_m": 0.0, "depth_m": 0.01}, "no depth_m range"),
                                           ({"along_m": 0.0, "roll_deg": -80.0}, "no roll_deg range")])
def test_a_value_outside_the_tasks_range_is_an_error(value, needle):
    with pytest.raises(ValueError, match=needle):
        G.given({"grasp": value}, DRAWER)
    with pytest.raises(ValueError, match="outside"):
        G.given({"grasp": {"along_m": 0.0, "depth_m": 0.05}}, RICH)


def test_a_task_without_a_range_takes_no_value():
    with pytest.raises(ValueError, match="no along_m range"):
        G.given({"grasp": {"along_m": 0.0}}, {k: v for k, v in DRAWER.items() if k != "range"})


def test_an_id_is_a_value_for_reach_and_drive_only():
    assert G.given({"grasp_id": 3}, DRAWER) is None                                  # a box design plans the id
    assert G.given({"grasp_id": 3}, DRAWER, by_id=True) == G.fixed(DRAWER, 0.1)
    assert G.given({}, DRAWER, by_id=True) is None and G.given(None, DRAWER) is None


def test_reach_takes_the_layouts_grasp_else_the_seeds_sample():
    assert G.for_reach({}, DRAWER, 0.07) == G.fixed(DRAWER, 0.07)
    assert G.for_reach({"reach": {"d_m": 0.7}}, DRAWER, 0.07) == G.fixed(DRAWER, 0.07)
    assert G.for_reach({"grasp": {"along_m": -0.15}}, DRAWER, 0.07)["along"] == -0.15
    assert G.for_reach({"grasp_id": 0}, DRAWER, 0.07)["along"] == -0.2


def test_narrowed_makes_the_value_the_tasks_only_grasp():
    n = G.narrowed(RICH, {"along": 0.04, "depth": 0.01, "roll_deg": -88.0})
    assert n == dict(RICH, points=[0.04], depth=0.01, roll_deg=-88.0) and RICH["points"] == DRAWER["points"]


# ---- the layout
def test_layout_grasp_by_value_is_parsed():
    ok = {"grasp": {"along_m": 0.1, "depth_m": 0.01, "roll_deg": -90}}
    assert D.parse_layout(ok) == ok
    for bad in ({"grasp": 1}, {"grasp": {}}, {"grasp": {"depth_m": 0.1}}, {"grasp": {"along_m": "0"}},
                {"grasp": {"along_m": float("nan")}}, {"grasp": {"along_m": True}}, {"grasp": {"along_m": 0, "x": 1}}):
        with pytest.raises(ValueError, match="grasp"):
            D.parse_layout(bad)


@pytest.mark.parametrize("layout", [{"grasp_id": 1, "grasp": {"along_m": 0.0}},
                                    {"grasp_id": 1, "reach": {"grasp_m": 0.0}},
                                    {"grasp": {"along_m": 0.0}, "reach": {"grasp_m": 0.0}}])
def test_two_grasp_sources_in_a_layout_are_an_error(layout):
    with pytest.raises(ValueError, match="each give the grasp"):
        D.parse_layout(layout)


def test_the_grasp_id_bound_is_the_tasks_grasp_count():
    assert D.parse_layout({"grasp_id": 6}, grasps=7) == {"grasp_id": 6}
    with pytest.raises(ValueError, match="0..2"):
        D.parse_layout({"grasp_id": 3}, grasps=3)
    with pytest.raises(ValueError):
        D.layout_from_env({"ROUND_LAYOUT": '{"grasp_id": 5}'})                       # the default: 5 points
    assert D.layout_from_env({"ROUND_LAYOUT": '{"grasp_id": 5}'}, grasps=6) == {"grasp_id": 5}


def test_every_block_takes_a_grasp_and_only_its_own_start_keys():
    box, reach = D.parse_starts(BOX.block()), D.parse_starts(REACH.block())
    for layout in ({"grasp_id": 1}, {"grasp": {"along_m": 0.0}}):
        for spec in (box, reach):
            D.check_layout_mode(layout, spec)
    with pytest.raises(ValueError, match="start_pose"):
        D.check_layout_mode({"start_pose": [0, 0, 0], "grasp": {"along_m": 0.0}}, reach)


def test_a_grasp_by_value_is_the_one_grasp_of_every_box_attempt():
    spec, box = D.parse_starts(BOX.block()), tuple(BOX.centre)
    for layout in ({"grasp": {"along_m": 1.0}}, {"grasp": {"along_m": 1.0}, "start_pose": [*box, math.pi]}):
        assert {s.grasp_id for s in D.layout_starts(layout, 4, 12, spec, box)} == {0}
    assert [s.grasp_id for s in D.layout_starts({"grasp_id": 3}, 4, 12, spec, box)] == [3] * 12


def test_the_clusters_cycle_the_tasks_grasp_count():
    spec, box = D.parse_starts(BOX.block()), tuple(BOX.centre)
    assert [D.grasp_of_cluster(c, 3) for c in range(7)] == [0, 1, 2, 0, 1, 2, 0]
    assert [s.grasp_id for s in D.layout_starts({}, 0, 7, spec, box, grasps=3)] == [0, 1, 2, 0, 1, 2, 0]
    assert {s.grasp_id for s in D.plan_starts(0, 0, 40, spec, box, grasps=3)} == {0, 1, 2}


def test_the_reach_sample_keeps_every_other_draw_when_the_grasp_comes_from_the_layout():
    spec = D.parse_starts(REACH.block())
    for seed in (0, 5, 31):
        base = D.reach_sample(seed, spec)
        value = G.for_reach({"grasp": {"along_m": 0.15}}, DRAWER, base.grasp)          # outside the block's grasp_m
        got = dataclasses.replace(base, grasp=value["along"])
        assert (got.d, got.dyaw, got.grasp) == (base.d, base.dyaw, 0.15)
        assert D.reach_attempts(seed, 6, spec, {"grasp": {"along_m": 0.15}}) == D.reach_attempts(seed, 6, spec)


# ---- the randomiser: limits
def test_both_modes_show_the_tasks_values():
    for task, robot, ax in (("open_drawer", "mobile_panda", (-0.2, 0.2)), ("open_fridge_door", "mobile_panda", (0.955, 1.095)),
                            ("open_drawer", "mobile_so101_xl", (-0.2, 0.2))):
        assert limits({}, task=task, robot=robot) == {"grasp.id": S.Choices((0, 1, 2, 3, 4)), "grasp.along_m": S.Range(*ax)}


def test_depth_and_roll_dims_exist_where_the_task_allows_them(rich):
    assert limits({}) == {"grasp.id": S.Choices((0, 1, 2, 3, 4)), "grasp.along_m": S.Range(-0.2, 0.2),
                          "grasp.depth_m": S.Range(0.0, 0.03), "grasp.roll_deg": S.Range(-100.0, -80.0)}


def test_the_grasp_needs_a_start_design():
    with pytest.raises(StartDesignError, match="no start design"):
        limits({}, robot="mobile_so101")


def test_ids_are_a_subset_of_the_tasks_points_in_points_mode(tmp_path):
    ok = make_spec({"grasp": {"id": {"choices": [0, 2, 4]}}})
    Pipeline(ok, cli.build_plugins(ok, ("sample",)), tmp_path, log=lambda m: None)
    bad = make_spec({"grasp": {"id": {"choices": [0, 5]}}})
    with pytest.raises(S.SpecError, match="grasp.id"):
        Pipeline(bad, cli.build_plugins(bad, ("sample",)), tmp_path, log=lambda m: None)


def test_a_range_is_inside_the_tasks_range(tmp_path):
    ok = make_spec({"grasp": {"along_m": [-0.1, 0.1]}})
    Pipeline(ok, cli.build_plugins(ok, ("sample",)), tmp_path, log=lambda m: None)
    bad = make_spec({"grasp": {"along_m": [-0.1, 0.25]}})
    with pytest.raises(S.SpecError, match="grasp.along_m"):
        Pipeline(bad, cli.build_plugins(bad, ("sample",)), tmp_path, log=lambda m: None)


@pytest.mark.parametrize("randomisers, needle", [
    ({"grasp": {"id": {"choices": [0, 1]}, "along_m": [0.0, 0.1]}}, "two modes"),
    ({"grasp": {"depth_m": [0.0, 0.01]}}, "needs along_m"),
])
def test_mixing_the_modes_is_a_clear_error(randomisers, needle):
    with pytest.raises(S.SpecError, match=needle):
        limits(randomisers)


@pytest.mark.parametrize("grasp", [{"along_m": [-0.05, 0.05]}, {"id": {"choices": [0, 1]}}])
def test_a_grasp_next_to_the_start_randomisers_grasp_m_is_a_clear_error(grasp):
    spec = make_spec({"grasp": grasp, "drive_start": {"grasp_m": [-0.1, 0.1]}}, robot="mobile_so101_xl")
    with pytest.raises(S.SpecError, match="both give the grasp"):
        registry.create("randomiser", "grasp").limits(spec)
    spec = make_spec({"grasp": grasp, "drive_start": {"d_m": [0.7, 0.72]}}, robot="mobile_so101_xl")
    registry.create("randomiser", "grasp").limits(spec)                              # another dim is fine


# ---- the randomiser: realised layout
def test_realise_points_mode_is_the_grasp_id_and_range_mode_the_value():
    g = registry.create("randomiser", "grasp")
    assert g.realise({"id": 3}, {}) == {"grasp_id": 3}
    assert g.realise({"along_m": 0.04}, {}) == {"grasp": {"along_m": 0.04}}
    assert g.realise({"along_m": 0.04, "depth_m": 0.01, "roll_deg": -85.0}, {}) == {
        "grasp": {"along_m": 0.04, "depth_m": 0.01, "roll_deg": -85.0}}
    assert g.realise({}, {}) == {}


@pytest.mark.parametrize("robot", ["mobile_panda", "mobile_so101_xl"])
def test_the_layout_carries_the_grasp_for_a_box_and_a_drive_robot(robot):
    def layout(randomisers, values):
        return Realiser(make_spec(randomisers, robot=robot)).layout(Config("train", 3, (), values))
    assert layout({"grasp": {"id": {"choices": [1, 2]}}}, {"grasp.id": 2}) == {"grasp_id": 2}
    got = layout({"grasp": {"along_m": [-0.1, 0.1]}}, {"grasp.along_m": 0.04})
    assert got == {"grasp": {"along_m": 0.04}} and D.parse_layout(got) == got
    if robot == "mobile_so101_xl":
        got = layout({"grasp": {"along_m": [-0.1, 0.1]}, "drive_start": {"d_m": [0.7, 0.72]}},
                     {"grasp.along_m": 0.04, "drive_start.d_m": 0.71})
        assert got == {"grasp": {"along_m": 0.04}, "reach": {"d_m": 0.71}} and D.parse_layout(got) == got


# ---- the pipeline
@pytest.mark.parametrize("task, robot, randomisers, dim", [
    ("open_drawer", "mobile_panda", {"grasp": {"id": {"choices": [0, 2, 4]}}}, "grasp.id"),
    ("open_drawer", "mobile_panda", {"grasp": {"along_m": [-0.1, 0.1]}}, "grasp.along_m"),
    ("open_fridge_door", "mobile_panda", {"grasp": {"along_m": [0.97, 1.08]}}, "grasp.along_m"),
    ("open_drawer", "mobile_so101_xl", {"grasp": {"id": {"choices": [0, 4]}}, "drive_start": {"d_m": [0.68, 0.76]}}, "grasp.id"),
    ("open_drawer", "mobile_so101_xl", {"grasp": {"along_m": [-0.2, 0.2]}, "drive_start": {"back_m": [1.0, 1.6]}}, "grasp.along_m"),
])
def test_sample_draws_the_grasp_dim(tmp_path, task, robot, randomisers, dim):
    spec = make_spec(randomisers, task=task, robot=robot)
    Pipeline(spec, cli.build_plugins(spec, ("sample",)), tmp_path, log=lambda m: None).run(["sample"])
    sampled = json.loads((tmp_path / "manifest.json").read_text())["splits"]["train"]["sampled"]
    got = [c["values"][dim] for c in sampled]
    given = spec.randomisers["grasp"]
    dim_spec = next(iter(given.values()))
    assert len(got) == 10 and all(S.contains(dim_spec, v) for v in got)
    if isinstance(dim_spec, S.Range):
        assert len(set(got)) == 10                                                   # a continuous draw, not a point


def test_a_range_grasp_counts_as_a_continuous_distance_and_an_id_as_a_choice():
    from datagen.core import samplers
    dims = {"grasp.along_m": S.Range(-0.2, 0.2), "grasp.id": S.Choices((0, 1, 2))}
    assert samplers.distance({"grasp.along_m": -0.2}, {"grasp.along_m": 0.2}, dims) == pytest.approx(1.0)
    assert samplers.distance({"grasp.along_m": 0.0}, {"grasp.along_m": 0.1}, dims) == pytest.approx(0.25)
    assert samplers.distance({"grasp.id": 0}, {"grasp.id": 2}, dims) == 1.0 and samplers.distance(
        {"grasp.id": 1}, {"grasp.id": 1}, dims) == 0.0


def test_a_range_grasp_only_spec_with_min_distance_keeps_the_distance(tmp_path):
    raw = {"task": "open_drawer", "robot": "mobile_panda", "out": str(tmp_path), "generator": {"plugin": "automoma"},
           "randomisers": {"grasp": {"along_m": [-0.2, 0.2]}},
           "splits": {"train": {"target": 2, "sampler": {"overrides": {"grasp.along_m": 0.0}}, "pool": {"initial": [0, 5]}},
                      "heldout": {"target": 3, "pool": {"initial": [500, 540]}, "record": False,
                                  "min_distance": {"from": "train", "value": 0.2}}}}
    spec = S.parse(raw)
    Pipeline(spec, cli.build_plugins(spec, ("sample",)), tmp_path, log=lambda m: None).run(["sample"])
    sampled = json.loads((tmp_path / "manifest.json").read_text())["splits"]["heldout"]["sampled"]
    assert len(sampled) == 41 and all(abs(c["values"]["grasp.along_m"]) >= 0.2 * 0.4 - 1e-9 for c in sampled)


# ---- listing
def test_the_listing_shows_both_modes_with_the_tasks_values():
    out = listing.render("open_fridge_door", "mobile_panda")
    assert "grasp.id" in out and "{0, 1, 2, 3, 4}" in out and "grasp.along_m" in out and "[0.955, 1.095]" in out


@pytest.mark.parametrize("name, dim", [("v20_grasp_points", "grasp.id"), ("v21_grasp_range", "grasp.along_m")])
def test_the_matrix_specs_of_both_modes_sample(tmp_path, name, dim):
    spec = dataclasses.replace(S.load(specs_bridge.SPECS / "datagen" / "matrix" / f"{name}.yaml"), out=str(tmp_path))
    Pipeline(spec, cli.build_plugins(spec, ("sample",)), tmp_path, log=lambda m: None).run(["sample"])
    sampled = json.loads((tmp_path / "manifest.json").read_text())["splits"]["train"]["sampled"]
    assert len(sampled) == 12 and {dim} <= set(sampled[0]["values"])
    assert len({c["values"][dim] for c in sampled}) > (3 if dim == "grasp.along_m" else 1)
