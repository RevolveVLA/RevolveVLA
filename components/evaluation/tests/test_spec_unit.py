"""core/spec.py: loading, defaults, hash, validation errors naming the yaml path."""
import copy
import dataclasses
from pathlib import Path

import pytest
from datagen.core.spec import Range, SpecError

from evaluation.core import spec as S

REPO = Path(__file__).resolve().parents[3]
TRAIN = {
    "task": "t", "robot": "r", "out": "runs/train_run", "generator": {"plugin": "fake_datagen"},
    "randomisers": {"fake_datagen": {"x": [-0.5, 0.5], "y": [1, 2]}},
    "splits": {"train": {"target": 3, "pool": {"initial": [0, 9]}},
               "heldout": {"target": 2, "pool": {"initial": [20, 24]}, "record": False}},
}
RAW = {
    "train": "train.yaml", "out": "runs/eval_run", "ckpt": {"source": "local", "path": "runs/ckpts/x"},
    "sets": {"heldout": {"count": 4, "sampler": {"strategy": "sobol"}, "pool": {"initial": [100, 109]},
                         "min_distance": {"value": 0.1, "realised": True}},
             "train": {"from": "train", "count": 2}},
    "rollouts": 3, "retries": 1,
}


@pytest.fixture
def root(tmp_path):
    import yaml
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(TRAIN))
    return tmp_path


def parse(root, **edits):
    raw = copy.deepcopy(RAW)
    for k, v in edits.items():
        raw[k] = v
    return S.parse(raw, "eval.yaml", root)


def test_the_repo_example_loads():
    spec = S.load(REPO / "specs" / "eval" / "fridge_base50_30k.yaml", REPO)
    assert spec.train.task == "open_fridge_door" and spec.train_manifest == "runs/datagen/fridge_base50/manifest.json"
    assert spec.judge.plugin == "c2" and spec.ckpt.source == "local" and (spec.rollouts, spec.retries) == (5, 1)
    assert spec.sets["heldout"].min_distance.value == 0.1 and spec.sets["train"].source == "train"


def test_values_and_defaults(root):
    spec = parse(root)
    assert spec.train.out == "runs/train_run" and spec.ckpt == S.CkptSpec("local", {"path": "runs/ckpts/x"})
    h, t = spec.sets["heldout"], spec.sets["train"]
    assert (h.kind, h.count, h.sampler.strategy, h.pool.ranges(), h.plugin) == ("heldout", 4, "sobol", [(100, 109)], "heldout")
    assert h.min_distance.source == "train" and h.min_distance.realised
    assert (t.kind, t.source, t.count, t.plugin) == ("train", "train", 2, "train")
    assert spec.runner == S.RunnerSpec(21, 0.0) and spec.run == S.RunSpec(None, True)
    bare = S.parse({"train": "train.yaml", "out": "o", "sets": {"h": {"count": 1, "pool": {"initial": [100, 100]}}}}, "e", root)
    assert (bare.rollouts, bare.retries, bare.ckpt, bare.judge.plugin, bare.policy.plugin) == (1, 0, None, "c0", "smolvla")


def test_judge_criterion_and_params(root):
    j = parse(root, judge={"criterion": "c2", "grip": 1}).judge
    assert (j.plugin, dict(j.params)) == ("c2", {"grip": 1})


def test_hash_ignores_out_and_run_but_not_settings(root):
    base = parse(root)
    assert parse(root, out="elsewhere", run={"workers": 4, "headless": False}).hash == base.hash
    assert parse(root, rollouts=4).hash != base.hash
    assert parse(root, runner={"actions_per_chunk": 30}).hash != base.hash
    other = dataclasses.replace(base, train=dataclasses.replace(base.train, seed=1))
    assert other.hash != base.hash


def test_train_path_is_relative_to_root_or_absolute(root):
    assert parse(root).train.task == "t"
    assert parse(root, train=str(root / "train.yaml")).train.task == "t"
    with pytest.raises(SpecError, match="cannot read"):
        S.parse(RAW | {"train": "missing.yaml"}, "e", root)


@pytest.mark.parametrize("edits, match", [
    ({"nope": 1}, r"eval.yaml.nope: unknown key"),
    ({"rollouts": 0}, r"eval.yaml.rollouts: must be >= 1"),
    ({"retries": -1}, r"eval.yaml.retries: must be >= 0"),
    ({"runner": {"chunk_size_threshold": 1.5}}, r"runner.chunk_size_threshold: must be in \[0, 1\]"),
    ({"runner": {"actions_per_chunk": 0}}, r"runner.actions_per_chunk: must be >= 1"),
    ({"runner": {"chunks": 1}}, r"runner.chunks: unknown key"),
    ({"run": {"workers": 0}}, r"run.workers: must be >= 1"),
    ({"ckpt": {"path": "x"}}, r"ckpt: missing key 'source'"),
    ({"judge": {"plugin": "c2"}}, r"judge: missing key 'criterion'"),
    ({"sets": {}}, r"eval.yaml.sets: needs at least one set"),
    ({"sets": {"h": {"count": 1}}}, r"sets.h: missing key 'pool'"),
    ({"sets": {"h": {"count": 1, "pool": {"initial": [100, 101]}, "poool": 1}}}, r"sets.h.poool: unknown key"),
    ({"sets": {"h": {"count": 5, "pool": {"initial": [100, 102]}}}}, r"sets.h.pool: 3 seeds cannot reach count 5"),
    ({"sets": {"h": {"count": 1, "pool": {"initial": [8, 12]}}}}, r"sets.h.pool: seeds \[8, 12\] overlap \[0, 9\] of training split 'train'"),
    ({"sets": {"a": {"count": 1, "pool": {"initial": [100, 105]}}, "b": {"count": 1, "pool": {"initial": [105, 110]}}}},
     r"sets.b.pool: .*overlap .* of set 'a'"),
    ({"sets": {"h": {"count": 1, "pool": {"initial": [100, 101]}, "min_distance": {"value": 0}}}}, r"min_distance.value: must be > 0"),
    ({"sets": {"h": {"count": 1, "pool": {"initial": [100, 101]}, "min_distance": {"value": 1, "from": "nope"}}}},
     r"min_distance.from: 'nope' is not a split of the training spec"),
    ({"sets": {"h": {"count": 1, "pool": {"initial": [100, 101]}, "sampler": {"bins": {"fake_datagen.nope": 2}}}}},
     r"sets.h.sampler.bins.fake_datagen.nope"),
    ({"sets": {"h": {"count": 1, "pool": {"initial": [100, 101]}, "randomisers": {"fake_datagen": {"z": [0, 1]}}}}},
     r"sets.h.randomisers.fake_datagen.z: not a dim of the training spec"),
    ({"sets": {"h": {"count": 1, "pool": {"initial": [100, 101]}, "randomisers": {"fake_datagen": {"x": [-0.1, 0.7]}}}}},
     r"sets.h.randomisers.fake_datagen.x: Range\(lo=-0.1, hi=0.7\) is outside the training range Range\(lo=-0.5, hi=0.5\)"),
    ({"sets": {"t": {"from": "train", "count": 9}}}, r"sets.t.count: 9 > the 3 configs"),
    ({"sets": {"t": {"from": "nope", "count": 1}}}, r"sets.t.from: 'nope' is not a split"),
    ({"sets": {"t": {"from": "train", "count": 1, "pool": {"initial": [1, 2]}}}}, r"sets.t.pool: unknown key"),
])
def test_errors_name_the_yaml_path(root, edits, match):
    with pytest.raises(SpecError, match=match):
        parse(root, **edits)


def test_narrower_randomisers_are_kept_and_the_unrecorded_training_pool_may_be_reused(root):
    spec = parse(root, sets={"h": {"count": 1, "pool": {"initial": [20, 24]},
                                   "randomisers": {"fake_datagen": {"x": [-0.1, 0.1]}}}})
    assert spec.sets["h"].randomisers["fake_datagen"]["x"] == Range(-0.1, 0.1)      # training's heldout split is not trained on


def test_unreadable_yaml(tmp_path):
    (tmp_path / "bad.yaml").write_text("a: [")
    with pytest.raises(SpecError, match="cannot read"):
        S.load(tmp_path / "bad.yaml")
