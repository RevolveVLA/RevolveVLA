"""PRD 09: reach_start / drive_start (limits from the task's start design, variants, method mismatch), the layout the
planner takes for reach and drive blocks, --list-randomisers, and that a spec with no start randomiser plans as before
(data/start_draws_before.json, written from the planner code before the hooks)."""
import dataclasses
import json
import math
from pathlib import Path

import pytest

import entry
import specs_bridge
import start_design as D
from datagen import cli
from datagen.core import registry, spec as S
from datagen.core.interfaces import Config
from datagen.core.pipeline import Pipeline
from datagen.plugins.automoma import adapter
from datagen.plugins.automoma.layout import Realiser
from datagen.plugins.randomisers import listing
from datagen.plugins.randomisers.design import StartDesignError
from revolve_common import specs

BEFORE = json.loads((Path(__file__).parent / "data" / "start_draws_before.json").read_text())
REACH = dict(method="reach", grasp_m=[-0.1, 0.1], d_m=[0.68, 0.76], yaw_half_deg=8.0, jitter_m=0.01, jitter_deg=1.0)
DRIVE = dict(REACH, method="drive", back_m=[1.0, 1.6], side_m=[-0.3, 0.3], heading_half_deg=20.0, slowdown=2.0)
REACH_DIMS = {"grasp_m": S.Range(-0.1, 0.1), "d_m": S.Range(0.68, 0.76), "yaw_deg": S.Range(-8.0, 8.0)}
DRIVE_DIMS = REACH_DIMS | {"back_m": S.Range(1.0, 1.6), "side_m": S.Range(-0.3, 0.3),
                           "heading_deg": S.Range(-20.0, 20.0)}


def make_spec(randomisers=None, task="open_drawer", robot="mobile_so101_xl", generator=None):
    return S.parse({"task": task, "robot": robot, "out": "unused", "generator": {"plugin": "automoma", **(generator or {})},
                    "randomisers": randomisers or {},
                    "splits": {"train": {"target": 4, "sampler": {"strategy": "sobol"}, "pool": {"initial": [0, 9]}}}})


def limits(name, spec):
    return registry.create("randomiser", name).limits(spec)


@pytest.fixture
def reach_design(monkeypatch):
    """A `method: reach` design for open_drawer / mobile_so101_xl (no shipped task has one), a drive one as variant."""
    designs = {None: specs.ReachStarts(**REACH), "far": specs.DriveStarts(**DRIVE)}
    monkeypatch.setattr("datagen.specs_bridge.starts", lambda task, robot=None, variant=None: designs[variant])
    return designs


# ---- limits
def test_reach_start_limits_come_from_the_reach_block(reach_design):
    assert limits("reach_start", make_spec()) == {f"reach_start.{d}": r for d, r in REACH_DIMS.items()}


def test_drive_start_limits_come_from_the_drive_block():
    assert limits("drive_start", make_spec()) == {f"drive_start.{d}": r for d, r in DRIVE_DIMS.items()}
    block = specs_bridge.starts("open_drawer", "mobile_so101_xl")
    assert S.Range(*block.back_m) == DRIVE_DIMS["back_m"] and block.heading_half_deg == 20.0


def test_limits_follow_the_generators_variant(reach_design):
    assert limits("reach_start", make_spec()).keys() == {f"reach_start.{d}" for d in REACH_DIMS}      # no variant
    assert limits("drive_start", make_spec(generator={"variant": "far"})) == {
        f"drive_start.{d}": r for d, r in DRIVE_DIMS.items()}
    with pytest.raises(StartDesignError, match="uses drive: use drive_start, not reach_start"):
        limits("reach_start", make_spec(generator={"variant": "far"}))


# ---- method mismatch
@pytest.mark.parametrize("randomiser, robot, needle", [
    ("robot_base", "mobile_so101_xl", "mobile_so101_xl on open_drawer uses drive: use drive_start, not robot_base"),
    ("reach_start", "mobile_so101_xl", "mobile_so101_xl on open_drawer uses drive: use drive_start, not reach_start"),
    ("drive_start", "mobile_panda", "mobile_panda on open_drawer uses box: use robot_base, not drive_start"),
    ("reach_start", "mobile_panda", "mobile_panda on open_drawer uses box: use robot_base, not reach_start"),
    ("drive_start", "mobile_so101", "no start design"),
])
def test_a_randomiser_of_another_method_is_a_clear_error(randomiser, robot, needle):
    with pytest.raises(StartDesignError, match=needle):
        limits(randomiser, make_spec(robot=robot))


def test_the_pipeline_rejects_the_mismatch_at_load(tmp_path):
    spec = make_spec({"robot_base": {"x_m": [-0.1, 0.1]}})
    with pytest.raises(StartDesignError, match="use drive_start, not robot_base"):
        Pipeline(spec, cli.build_plugins(spec, ("sample",)), tmp_path, log=lambda m: None)


def test_a_dim_outside_the_blocks_range_is_rejected(tmp_path):
    spec = make_spec({"drive_start": {"back_m": [0.5, 1.2]}})
    with pytest.raises(S.SpecError, match="drive_start.back_m"):
        Pipeline(spec, cli.build_plugins(spec, ("sample",)), tmp_path, log=lambda m: None)


def test_sample_draws_every_dim_of_the_drive_start(tmp_path):
    spec = make_spec({"drive_start": {d: [r.lo, r.hi] for d, r in DRIVE_DIMS.items()}})
    pipe = Pipeline(spec, cli.build_plugins(spec, ("sample",)), tmp_path, log=lambda m: None)
    assert pipe.run(["sample"])
    sampled = [Config.from_dict(c) for c in pipe.manifest.data["splits"]["train"]["sampled"]]
    assert len(sampled) == 10        # the initial pool
    for c in sampled:
        assert set(c.of("drive_start")) == set(DRIVE_DIMS)
        assert all(DRIVE_DIMS[d].lo <= v <= DRIVE_DIMS[d].hi for d, v in c.of("drive_start").items())


# ---- realise
def test_reach_start_realises_the_varied_dims_only():
    r = registry.create("randomiser", "reach_start")
    assert r.realise({"d_m": 0.7, "yaw_deg": 3.0}, {}) == {"reach": {"d_m": 0.7, "yaw_deg": 3.0}}
    assert r.realise({}, {}) == {}


def test_drive_start_realises_reach_and_drive_groups():
    r = registry.create("randomiser", "drive_start")
    assert r.realise({"d_m": 0.7, "back_m": 1.2, "heading_deg": -4}, {}) == {
        "reach": {"d_m": 0.7}, "drive": {"back_m": 1.2, "heading_deg": -4}}
    assert r.realise({"back_m": 1.2}, {}) == {"drive": {"back_m": 1.2}}
    assert r.realise({}, {}) == {}


def test_the_layout_of_a_drive_config_is_the_planners():
    spec = make_spec({"drive_start": {"d_m": [0.68, 0.76], "side_m": [-0.3, 0.3]}})
    layout = Realiser(spec).layout(Config("train", 3, (), {"drive_start.d_m": 0.7, "drive_start.side_m": -0.1}))
    assert layout == {"reach": {"d_m": 0.7}, "drive": {"side_m": -0.1}}
    assert D.parse_layout(layout) == layout
    assert Realiser(make_spec()).layout(Config("train", 3, (), {})) == {}


# ---- planner: layout parsing and consumption (no GPU)
def spec_of(block):
    return D.parse_starts({"mode": block["method"], **{k: v for k, v in block.items() if k != "method"}})


RSPEC, DSPEC = spec_of(REACH), spec_of(DRIVE)


@pytest.mark.parametrize("bad", [
    '{"reach": 1}', '{"reach": {"x": 1}}', '{"reach": {"d_m": "0.7"}}', '{"reach": {"d_m": NaN}}',
    '{"drive": {"back_m": true}}', '{"drive": {"d_m": 1}}', '{"drive": [1]}',
])
def test_a_bad_reach_or_drive_layout_is_rejected(bad):
    with pytest.raises(ValueError):
        D.layout_from_env({"ROUND_LAYOUT": bad})


def test_layout_keys_must_fit_the_blocks_method():
    D.check_layout_mode({"reach": {"d_m": 0.7}}, RSPEC)
    D.check_layout_mode({"reach": {"d_m": 0.7}, "drive": {"back_m": 1.2}}, DSPEC)
    D.check_layout_mode({"grasp_id": 1}, DSPEC)                         # every block takes a grasp
    D.check_layout_mode({"grasp": {"along_m": 0.0}}, RSPEC)
    D.check_layout_mode({"start_pose": [0, 0, 0], "grasp_id": 1}, D.parse_starts(
        dict(clusters=10, per_cluster=10, box_m=0.4, yaw_half_deg=30, offset_m=0.03, offset_deg=3, nominal_arm="ready")))
    with pytest.raises(ValueError, match="drive"):
        D.check_layout_mode({"drive": {"back_m": 1.2}}, RSPEC)
    with pytest.raises(ValueError, match="start_pose"):
        D.check_layout_mode({"start_pose": [0, 0, 0]}, DSPEC)
    with pytest.raises(ValueError, match="reach"):
        D.check_layout_mode({"reach": {"d_m": 0.7}}, D.parse_starts(
            dict(clusters=1, per_cluster=1, box_m=0.4, yaw_half_deg=30, offset_m=0.03, offset_deg=3, nominal_arm="ready")))


def test_a_layout_value_replaces_its_draw_and_the_others_keep_theirs():
    own = D.reach_sample(5, RSPEC)
    s = D.reach_sample(5, RSPEC, {"reach": {"d_m": 0.7, "yaw_deg": -3.0}})
    assert (s.d, s.dyaw, s.grasp) == (0.7, math.radians(-3.0), own.grasp)
    assert D.reach_sample(5, RSPEC, {"reach": {"grasp_m": 0.05}}) == dataclasses.replace(own, grasp=0.05)
    d_own = D.drive_sample(5, DSPEC)
    d = D.drive_sample(5, DSPEC, {"drive": {"back_m": 1.1, "heading_deg": 7.0}})
    assert (d.back, d.heading, d.side) == (1.1, math.radians(7.0), d_own.side)


def test_the_attempts_jitter_the_layouts_sample():
    layout = {"reach": {"d_m": 0.7, "yaw_deg": 2.0}, "drive": {"back_m": 1.3, "side_m": 0.1, "heading_deg": -5.0}}
    (d0, y0), *rest = D.reach_attempts(9, 6, DSPEC, layout)
    assert (d0, y0) == (0.7, math.radians(2.0))
    assert all(abs(d - 0.7) <= DSPEC.jitter_m and abs(y - y0) <= math.radians(DSPEC.jitter_deg) + 1e-12 for d, y in rest)
    dr = D.drive_attempts(9, 6, DSPEC, layout)
    assert (dr[0].back, dr[0].side, dr[0].heading) == (1.3, 0.1, math.radians(-5.0))
    assert all(abs(x.back - 1.3) <= DSPEC.jitter_m for x in dr)
    starts = D.reach_starts(9, 6, DSPEC, lambda d: (1.0 - d, 0.0, D.FACING_YAW_OBJ), (0.1, 0.0), layout)
    assert starts[0].dx == 0.7 and starts[0].dyaw == math.radians(2.0)


@pytest.mark.parametrize("layout", [{"reach": {"d_m": 0.9}}, {"reach": {"yaw_deg": 9.0}}, {"reach": {"grasp_m": 0.2}}])
def test_a_reach_value_outside_the_block_is_rejected(layout):
    with pytest.raises(ValueError, match="outside the block"):
        D.reach_sample(0, RSPEC, layout)


@pytest.mark.parametrize("layout", [{"drive": {"back_m": 0.5}}, {"drive": {"side_m": 0.4}}, {"drive": {"heading_deg": 21}}])
def test_a_drive_value_outside_the_block_is_rejected(layout):
    with pytest.raises(ValueError, match="outside the block"):
        D.drive_sample(0, DSPEC, layout)


def test_entry_takes_a_reach_layout():
    layout = json.dumps({"reach": {"d_m": 0.7}, "drive": {"back_m": 1.2}})
    assert entry.layout_env(layout) == {"ROUND_LAYOUT": layout}


# ---- the equivalence: no start randomiser plans as before
def draws(name):
    spec = D.parse_starts(BEFORE[name]["block"])
    for seed, want in BEFORE[name]["seeds"].items():
        seed = int(seed)
        got = {"reach_sample": dataclasses.asdict(D.reach_sample(seed, spec)),
               "reach_attempts": D.reach_attempts(seed, 12, spec),
               "reach_starts": [dataclasses.asdict(s) for s in D.reach_starts(
                   seed, 12, spec, lambda d: (1.0 - d, 0.05 * d, D.FACING_YAW_OBJ), (0.1, -0.02))]}
        if name == "drive":
            got["drive_sample"] = dataclasses.asdict(D.drive_sample(seed, spec))
            got["drive_attempts"] = [dataclasses.asdict(x) for x in D.drive_attempts(seed, 12, spec)]
        yield seed, json.loads(json.dumps(got)), want


@pytest.mark.parametrize("name", sorted(BEFORE))
def test_without_a_start_randomiser_the_planner_draws_exactly_as_before(name):
    for seed, got, want in draws(name):
        assert got == want, f"seed {seed}"


@pytest.mark.parametrize("name", sorted(BEFORE))
def test_an_empty_layout_draws_as_before_too(name):
    spec = D.parse_starts(BEFORE[name]["block"])
    for seed in map(int, BEFORE[name]["seeds"]):
        assert D.reach_attempts(seed, 12, spec, {}) == D.reach_attempts(seed, 12, spec)


def test_a_spec_without_start_randomisers_has_no_layout():
    assert Realiser(make_spec()).layout(Config("train", 3, (), {})) == {}
    assert Realiser(make_spec(task="open_fridge_door", robot="mobile_panda")).layout(Config("train", 3, (), {})) == {}


# ---- the accepted attempt's realised start
def test_realised_start_of_a_drive_plan(tmp_path):
    starts = {"mode": "drive", "sample": {"grasp_y": 0.03, "d_m": 0.7, "dyaw_deg": 1.0},
              "attempts": [{"attempt_id": 4, "d_m": 0.705, "dyaw_deg": 1.5, "base_obj": [0.3, 0.0, 3.0],
                            "back_m": 1.2, "side_m": -0.1, "heading_deg": 4.0, "start_obj": [-0.9, 0.1, 3.2]}]}
    (tmp_path / "handoff.json").write_text(json.dumps({"starts": starts}))
    assert adapter.realised_start(tmp_path, 4) == {
        "start_pose": [-0.9, 0.1, 3.2], "drive_start.d_m": 0.705, "drive_start.yaw_deg": 1.5,
        "drive_start.grasp_m": 0.03, "drive_start.back_m": 1.2, "drive_start.side_m": -0.1,
        "drive_start.heading_deg": 4.0}


# ---- discoverability
def test_list_randomisers_of_a_drive_pair(capsys):
    assert cli.main(["--list-randomisers", "--task", "open_drawer", "--robot", "mobile_so101_xl"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "open_drawer on mobile_so101_xl: start design drive"
    assert [l.split()[0] for l in out[1:]] == [f"drive_start.{d}" for d in DRIVE_DIMS] + ["grasp.id", "grasp.along_m"]
    assert out[1].split(None, 1)[1].strip() == "[-0.1, 0.1]" and "[-20, 20]" in out[6]
    assert out[7].split(None, 1)[1].strip() == "{0, 1, 2, 3, 4}" and out[8].split(None, 1)[1].strip() == "[-0.2, 0.2]"


def test_list_randomisers_of_a_box_pair(capsys):
    assert cli.main(["--list-randomisers", "--task", "open_fridge_door", "--robot", "mobile_panda"]) == 0
    out = capsys.readouterr().out
    assert "start design box" in out and "grasp.id  " in out and "{0, 1, 2, 3, 4}" in out
    assert "robot_base.yaw_deg  [-27, 27]" in out and "drive_start" not in out


def test_list_randomisers_of_a_pair_without_a_start_design_and_with_a_variant(capsys):
    assert cli.main(["--list-randomisers", "--task", "open_drawer", "--robot", "mobile_so101"]) == 0
    assert capsys.readouterr().out.strip() == "open_drawer on mobile_so101: start design none"
    assert cli.main(["--list-randomisers", "--task", "open_fridge_door", "--robot", "mobile_panda",
                     "--variant", "drive_seq"]) == 0
    out = capsys.readouterr().out
    assert "(variant drive_seq): start design drive" in out and "drive_start.d_m" in out and "robot_base" not in out


@pytest.mark.parametrize("argv", [["--task", "nope", "--robot", "x"],
                                  ["--task", "open_drawer", "--robot", "mobile_panda", "--variant", "zz"]])
def test_list_randomisers_of_an_unknown_pair_or_variant_is_an_error(argv, capsys):
    assert cli.main(["--list-randomisers", *argv]) == 1
    assert "datagen: --list-randomisers" in capsys.readouterr().err


def test_list_randomisers_needs_a_task_and_a_robot():
    with pytest.raises(SystemExit):
        cli.main(["--list-randomisers", "--task", "open_drawer"])


def test_usable_lists_the_randomisers_of_the_package():
    assert sorted(listing.usable("open_fridge_door", "mobile_panda")) == ["grasp", "robot_base"]
    assert sorted(listing.usable("open_drawer", "mobile_so101_xl")) == ["drive_start", "grasp"]


def test_a_grasp_only_spec_with_min_distance_samples_and_keeps_the_distance(tmp_path):
    """The matrix' v06 had to drop heldout min_distance (a choices dim had no distance): grasp ids now differ by 1."""
    raw = {"task": "open_fridge_door", "robot": "mobile_panda", "out": str(tmp_path), "generator": {"plugin": "automoma"},
           "randomisers": {"grasp": {"id": {"choices": [0, 1, 2, 3, 4]}}},
           "splits": {"train": {"target": 2, "sampler": {"overrides": {"grasp.id": 1}}, "pool": {"initial": [0, 5]}},
                      "heldout": {"target": 3, "pool": {"initial": [500, 520]}, "record": False,
                                  "min_distance": {"from": "train", "value": 0.5}}}}
    spec = S.parse(raw)
    Pipeline(spec, cli.build_plugins(spec, ("sample",)), tmp_path, log=lambda m: None).run(["sample"])
    sampled = json.loads((tmp_path / "manifest.json").read_text())["splits"]["heldout"]["sampled"]
    assert len(sampled) == 21 and all(c["values"]["grasp.id"] != 1 for c in sampled)
