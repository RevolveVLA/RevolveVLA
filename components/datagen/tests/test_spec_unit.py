"""core/spec.py: loading, validation (every bad case names its yaml path), hash stability, hard limits."""
import copy
from pathlib import Path

import pytest

from datagen.core import spec as S

REPO = Path(__file__).resolve().parents[3]

GOOD = {
    "task": "open_fridge_door", "robot": "mobile_panda", "out": "runs/datagen/x",
    "generator": {"plugin": "automoma", "attempts": 50, "checks": 3},
    "recorder": {"plugin": "rosetta"}, "converter": {"plugin": "rosetta", "format": "lerobot"},
    "randomisers": {"robot_base": {"x_m": [-0.15, 0.15], "y_m": [-0.15, 0.15], "yaw_deg": [-27, 27]},
                    "grasp": {"id": {"choices": [0, 1, 2]}}},
    "splits": {
        "train": {"target": 50, "sampler": {"strategy": "stratified", "bins": {"robot_base.x_m": 3}},
                  "pool": {"initial": [0, 99], "replacement": [100, 199]}},
        "heldout": {"target": 10, "sampler": {"strategy": "sobol", "overrides": {"grasp.id": 1}},
                    "pool": {"initial": [500, 529]}, "record": False,
                    "min_distance": {"from": "train", "value": 0.1, "realised": True}}},
    "seed": 0,
}


def bad(**edits):
    """GOOD with dotted-path edits applied (None deletes a key)."""
    d = copy.deepcopy(GOOD)
    for path, value in edits.items():
        *parents, last = path.split("__")
        node = d
        for p in parents:
            node = node[p]
        if value is None:
            del node[last]
        else:
            node[last] = value
    return d


def test_parse_full_spec():
    s = S.parse(GOOD)
    assert (s.task, s.robot, s.seed) == ("open_fridge_door", "mobile_panda", 0)
    assert s.generator == S.PluginRef("automoma", {"attempts": 50, "checks": 3})
    assert s.randomisers["robot_base"]["yaw_deg"] == S.Range(-27, 27)
    assert s.randomisers["grasp"]["id"] == S.Choices((0, 1, 2))
    train, held = s.splits
    assert train.sampler == S.SamplerSpec("stratified", {"robot_base.x_m": 3}) and train.record
    assert held.min_distance == S.MinDistance("train", 0.1, True) and not held.record
    assert held.sampler.overrides == {"grasp.id": 1}
    assert len(train.pool.seeds()) == 200 and train.pool.seeds()[100] == 100
    assert set(s.dims()) == {"robot_base.x_m", "robot_base.y_m", "robot_base.yaw_deg", "grasp.id"}


def test_minimal_spec_defaults():
    s = S.parse({"task": "t", "robot": "r", "out": "o", "generator": {"plugin": "g"},
                 "splits": {"a": {"target": 1, "pool": {"initial": [0, 0]}}}})
    assert s.recorder is None and s.converter is None and s.randomisers == {} and s.validators == () and s.seed == 0
    assert s.splits[0].sampler == S.SamplerSpec("random") and s.splits[0].record


@pytest.mark.parametrize("edits, where, msg", [
    ({"task": None}, "<spec>", "missing key 'task'"),
    ({"extra": 1}, "<spec>.extra", "unknown key"),
    ({"generator": {"attempts": 3}}, "<spec>.generator", "`plugin`"),
    ({"splits": {}}, "<spec>.splits", "at least one"),
    ({"seed": "a"}, "<spec>.seed", "integer"),
    ({"randomisers__robot_base__x_m": [0.2, -0.2]}, "randomisers.robot_base.x_m", "lo"),
    ({"randomisers__robot_base__x_m": [0.1]}, "randomisers.robot_base.x_m", "[lo, hi]"),
    ({"randomisers__robot_base__x_m": [True, 1]}, "randomisers.robot_base.x_m[0]", "number"),
    ({"randomisers__grasp__id": {"choices": []}}, "randomisers.grasp.id.choices", "non-empty"),
    ({"splits__train__target": 0}, "splits.train.target", ">= 1"),
    ({"splits__train__target": 1.5}, "splits.train.target", "integer"),
    ({"splits__train__bogus": 1}, "splits.train.bogus", "unknown key"),
    ({"splits__train__pool__initial": [5, 1]}, "splits.train.pool.initial", "first seed"),
    ({"splits__train__pool__replacement": [10, 20]}, "splits.train.pool", "overlap"),
    ({"splits__heldout__pool__initial": [150, 179]}, "splits.heldout.pool", "overlap"),
    ({"splits__heldout__pool__initial": [500, 505]}, "splits.heldout.pool", "cannot reach target 10"),
    ({"splits__heldout__min_distance__from": "nope"}, "min_distance.from", "not this split or an earlier one"),
    ({"splits__train__min_distance": {"from": "heldout", "value": 0.1}}, "splits.train.min_distance.from", "earlier"),
    ({"splits__heldout__min_distance__value": 0}, "min_distance.value", "> 0"),
    ({"splits__heldout__min_distance__realised": "yes"}, "min_distance.realised", "true/false"),
    ({"splits__train__sampler__bins": {"robot_base.nope": 3}}, "sampler.bins.robot_base.nope", "not a continuous dim"),
    ({"splits__train__sampler__bins": {"grasp.id": 2}}, "sampler.bins.grasp.id", "not a continuous dim"),
    ({"splits__train__sampler__bins": {"robot_base.x_m": 0}}, "sampler.bins.robot_base.x_m", ">= 1"),
    ({"splits__heldout__sampler__overrides": {"grasp.id": 9}}, "sampler.overrides.grasp.id", "outside"),
    ({"splits__heldout__sampler__overrides": {"nope.x": 9}}, "sampler.overrides.nope.x", "not a dim"),
])
def test_bad_specs_name_the_yaml_path(edits, where, msg):
    with pytest.raises(S.SpecError) as e:
        S.parse(bad(**edits))
    assert where in str(e.value) and msg in str(e.value), str(e.value)


def test_hash_is_stable_and_tracks_settings():
    h = S.parse(GOOD).hash
    assert h == S.parse(copy.deepcopy(GOOD)).hash
    shuffled = dict(reversed(list(GOOD.items())))
    assert S.parse(shuffled).hash == h
    assert S.parse(bad(out="elsewhere")).hash == h                       # `out` is not a setting
    assert S.parse(bad(run={"workers": 4, "headless": False})).hash == h        # nor how the run executes
    assert S.parse(bad(seed=1)).hash != h
    assert S.parse(bad(splits__train__target=51)).hash != h
    assert S.parse(bad(generator={"plugin": "automoma", "attempts": 51, "checks": 3})).hash != h


def test_load_reports_the_file(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("task: t\n")
    with pytest.raises(S.SpecError, match=r"s\.yaml: missing key 'robot'"):
        S.load(p)
    with pytest.raises(S.SpecError, match="cannot read"):
        S.load(tmp_path / "missing.yaml")
    p.write_text("a: [1\n")
    with pytest.raises(S.SpecError, match="cannot read"):
        S.load(p)


def test_check_limits():
    s = S.parse(GOOD)
    limits = {"robot_base.x_m": S.Range(-0.17, 0.17), "robot_base.y_m": S.Range(-0.17, 0.17),
              "robot_base.yaw_deg": S.Range(-27, 27), "grasp.id": S.Choices((0, 1, 2, 3, 4))}
    S.check_limits(s, limits)
    for key, lim in (("robot_base.x_m", S.Range(-0.1, 0.1)), ("grasp.id", S.Choices((0, 1))),
                     ("robot_base.x_m", S.Choices((0.0,)))):
        with pytest.raises(S.SpecError, match="outside the hard limits"):
            S.check_limits(s, limits | {key: lim})
    with pytest.raises(S.SpecError, match="unknown dim"):
        S.check_limits(s, {k: v for k, v in limits.items() if k != "grasp.id"})


@pytest.mark.parametrize("path", sorted((REPO / "specs" / "datagen").glob("*.yaml")), ids=lambda p: p.name)
def test_the_datagen_specs_of_the_repo_load(path):
    assert S.load(path).splits


def test_run_block_defaults_values_and_errors():
    assert S.parse(GOOD).run == S.RunSpec(1, True)
    assert S.parse(bad(run={"workers": 3})).run == S.RunSpec(3, True)
    assert S.parse(bad(run={"workers": 2, "headless": False})).run == S.RunSpec(2, False)
    for edit, path in (({"workers": 0}, "run.workers"), ({"workers": "3"}, "run.workers"), ({"workers": True}, "run.workers"),
                       ({"headless": "no"}, "run.headless"), ({"threads": 2}, "run.threads")):
        with pytest.raises(S.SpecError, match=path):
            S.parse(bad(run=edit))
