"""plugins/sets (heldout, train) + core/sets.py + core/checks.py on datagen's fake generator (CPU, tiny hand-written
training manifests)."""
import copy
import json
from pathlib import Path

import fake_datagen
import pytest
import yaml
from datagen.core.interfaces import Result
from datagen.core.state import State as Manifest
from datagen.core.pool import PoolExhausted
from datagen.core.samplers import distance
from datagen.core.spec import Choices, Range

from evaluation import cli
from evaluation.core import checks, sets, spec as S
from evaluation.core.errors import EvalError
from evaluation.plugins.sets.heldout import HeldoutSet
from evaluation.plugins.sets.train import TrainSet

TRAIN = {
    "task": "t", "robot": "r", "out": "train_run", "generator": {"plugin": "fake_datagen"},
    "randomisers": {"fake_datagen": {"x": [-0.5, 0.5], "y": [1, 2]}},
    "splits": {"train": {"target": 3, "pool": {"initial": [0, 9]},
                         "sampler": {"strategy": "stratified", "bins": {"fake_datagen.x": 3}}},
               "heldout": {"target": 2, "pool": {"initial": [20, 24]}, "record": False}},
}
HELDOUT = {"count": 4, "pool": {"initial": [100, 129]}, "sampler": {"strategy": "stratified", "bins": {"fake_datagen.x": 2}}}
SPEC = {"train": "train.yaml", "out": "run", "sets": {"heldout": HELDOUT}}


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_DATAGEN_LOG", str(tmp_path / "gen.log"))
    fake_datagen.CALLS.clear()
    return tmp_path


def make(root, sets_=None, train=TRAIN, **edits) -> S.EvalSpec:
    (root / "train.yaml").write_text(yaml.safe_dump(train))
    raw = copy.deepcopy(SPEC) | edits
    raw["sets"] = copy.deepcopy(sets_ or {"heldout": HELDOUT})
    return S.parse(raw, "eval.yaml", root)


def training_manifest(spec, root, entries, **extra):
    """runs of `train_run/state.json`: entries = [(seed, x, realised x | None, check dir | None)] of split train, used."""
    m = Manifest.open(root / "train_run" / "state.json", spec.train)
    rec = m.split("train")
    for seed, x, real, check in entries:
        cell = (min(int((x + 0.5) * 3), 2),)
        rec["sampled"].append({"split": "train", "seed": seed, "cell": list(cell), "values": {"fake_datagen.x": x, "fake_datagen.y": 1.5}})
        gen = {"ok": True, "realised": {} if real is None else {"fake_datagen.x": real}, "paths": {"check": str(check)} if check else {}}
        rec["used"].append({"config": rec["sampled"][-1], "generate": gen, "record": {"ok": True}})
    m.data["splits"].update(extra)
    m.save()


def build(spec, root, name="heldout", workers=1, plugin=HeldoutSet):
    return plugin(spec).build(spec.sets[name], root / "run", workers)


# ------------------------------------------------------------------ held-out
def test_heldout_draws_inside_the_training_ranges_and_passes_the_checks(root):
    spec = make(root)
    built = build(spec, root)
    assert [i.config.seed for i in built.items] == sorted({i.config.seed for i in built.items}) and len(built.items) == 4
    assert all(100 <= i.config.seed <= 129 and i.config.split == "heldout" for i in built.items)
    assert all(-0.5 <= i.config.values["fake_datagen.x"] <= 0.5 for i in built.items)
    assert {n: r.ok for n, r in built.checks.items()} == {"in_range": True, "seeds_disjoint": True, "training_configs": True}
    assert "sampled-only" in built.checks["training_configs"].reason
    assert all(i.start_dir == root / "run" / "heldout" / "starts" / f"seed_{i.config.seed}" and i.start_dir.is_dir()
               for i in built.items)                                  # the fake generator reports no check dir: its own dir


def test_a_narrower_subrange_is_drawn_from_but_distance_is_measured_in_the_training_dims(root):
    narrow = {"fake_datagen": {"x": [-0.1, 0.1]}}
    spec = make(root, {"heldout": HELDOUT | {"count": 6, "randomisers": narrow, "min_distance": {"value": 0.3}}})
    training_manifest(spec, root, [(0, 0.0, None, None), (1, 0.2, None, None), (2, -0.3, None, None)])
    built = build(spec, root)
    train = sets.Training.load(spec, root).configs("train")
    dims = spec.train.dims()
    assert all(-0.1 <= i.config.values["fake_datagen.x"] <= 0.1 for i in built.items)
    # x spans only 0.2 of its training range of 1.0, so the 0.3 apart is reached through y (range 1.0): the draw, in
    # the training dims, keeps the distance although the narrower dims alone could not (R2)
    assert min(distance(i.config.values, t.values, dims) for i in built.items for t in train) >= 0.3
    assert built.checks["min_distance"].ok and "training_configs" not in built.checks
    rec = json.loads((root / "run" / "heldout" / "manifest.json").read_text())["splits"]["heldout"]
    assert rec["redrawn"], "the draws that were too close were redrawn"


def test_a_realised_start_too_close_is_rejected_and_replaced_from_the_same_cell(root):
    train = TRAIN | {"generator": {"plugin": "fake_datagen", "realised": {100: 0.01, 101: 0.0} | {s: 0.2 for s in range(102, 130)}}}
    spec = make(root, {"heldout": HELDOUT | {"count": 2, "min_distance": {"value": 0.1, "realised": True}}}, train=train)
    training_manifest(spec, root, [(0, 0.0, 0.0, None), (1, 0.4, 0.4, None)])
    built = build(spec, root)
    rec = json.loads((root / "run" / "heldout" / "manifest.json").read_text())["splits"]["heldout"]
    rejected = {e["seed"]: e for e in rec["rejected"]}
    assert set(rejected) == {100, 101} and all(e["step"] == "realised" for e in rejected.values())
    cells = {c["seed"]: tuple(c["cell"]) for c in rec["sampled"]}
    for r in rejected:                                                # each pick's replacement is in its own cell
        assert any(cells[i.config.seed] == cells[r] and i.config.seed not in (100, 101) for i in built.items)
    assert len(built.items) == 2 and all(r.ok for r in built.checks.values())
    assert "realised_min_distance" in built.checks


def test_pool_exhaustion_is_an_error(root):
    train = TRAIN | {"generator": {"plugin": "fake_datagen", "realised": {s: 0.0 for s in range(100, 130)}}}
    spec = make(root, {"heldout": HELDOUT | {"count": 2, "min_distance": {"value": 0.1, "realised": True}}}, train=train)
    training_manifest(spec, root, [(0, 0.0, 0.0, None)])
    with pytest.raises(PoolExhausted):
        build(spec, root)


def test_a_seed_the_training_run_touched_fails_the_disjoint_check(root):
    spec = make(root, {"heldout": HELDOUT | {"pool": {"initial": [100, 129]}}})
    training_manifest(spec, root, [(105, 0.0, None, None)])           # trained on a seed outside its own pool
    built = build(spec, root)
    assert not built.checks["seeds_disjoint"].ok and "training [105]" in built.checks["seeds_disjoint"].reason
    assert built.checks["in_range"].ok


def test_the_trainings_unrecorded_heldout_split_does_not_count_as_touched(root):
    spec = make(root, {"heldout": HELDOUT | {"pool": {"initial": [20, 49]}}})      # overlaps training's own record:false heldout
    training_manifest(spec, root, [(0, 0.0, None, None)],
                      heldout={"sampled": [{"split": "heldout", "seed": 20, "cell": [], "values": {}}], "used": [], "rejected": []})
    assert build(spec, root).checks["seeds_disjoint"].ok


def test_no_manifest_measures_sampled_distances_to_the_resampled_training_configs(root):
    spec = make(root, {"heldout": HELDOUT | {"min_distance": {"value": 0.1}}})
    built = build(spec, root)
    assert built.checks["min_distance"].ok and "sampled-only" in built.checks["training_configs"].reason
    assert "realised_min_distance" not in built.checks


def test_a_realised_distance_without_the_manifest_fails_before_generating(root):
    spec = make(root, {"heldout": HELDOUT | {"min_distance": {"value": 0.1, "realised": True}}})
    with pytest.raises(EvalError, match=r"sets.heldout.min_distance.realised needs the training manifest .*train_run/state.json"):
        build(spec, root)
    assert not (root / "run").exists() and fake_datagen.CALLS == []


def test_a_manifest_of_another_training_spec_is_refused(root):
    spec = make(root, {"heldout": HELDOUT | {"min_distance": {"value": 0.1}}})
    training_manifest(spec, root, [(0, 0.0, None, None)])
    other = make(root, {"heldout": HELDOUT | {"min_distance": {"value": 0.1}}}, train=TRAIN | {"seed": 5})
    with pytest.raises(Exception, match="made by another spec"):
        build(other, root)


def test_a_set_named_like_its_training_split_is_an_error(root):
    spec = make(root, {"train": HELDOUT | {"min_distance": {"value": 0.1}}})
    with pytest.raises(EvalError, match="has the name of the training split"):
        build(spec, root, "train")


def test_resume_generates_no_seed_twice_and_a_killed_build_continues(root, monkeypatch):
    spec = make(root, {"heldout": HELDOUT | {"count": 3}})
    monkeypatch.setattr(fake_datagen, "KILL_AFTER", 2)
    with pytest.raises(KeyboardInterrupt):
        build(spec, root)
    done = len(json.loads((root / "run" / "heldout" / "manifest.json").read_text())["splits"]["heldout"]["used"])
    assert done == 2
    monkeypatch.setattr(fake_datagen, "KILL_AFTER", None)
    fake_datagen.CALLS.clear()
    first = build(spec, root)
    assert len(fake_datagen.CALLS) == 1 and len(first.items) == 3      # only the missing one
    fake_datagen.CALLS.clear()
    assert [i.config.seed for i in build(spec, root).items] == [i.config.seed for i in first.items]
    assert fake_datagen.CALLS == []


def test_workers_do_not_change_the_result(root):
    train = TRAIN | {"generator": {"plugin": "fake_datagen", "delays": {100: 0.2, 102: 0.1}, "realised": {101: 0.0}}}
    spec = make(root, {"heldout": HELDOUT | {"count": 4, "min_distance": {"value": 0.1, "realised": True}}}, train=train)
    training_manifest(spec, root, [(0, 0.0, 0.0, None), (1, 0.4, 0.4, None)])

    def run(out, workers):
        built = HeldoutSet(spec).build(spec.sets["heldout"], root / out, workers)
        rec = json.loads((root / out / "heldout" / "manifest.json").read_text())["splits"]["heldout"]
        return [i.config for i in built.items], [(e["seed"], e["step"], e["reason"]) for e in rec["rejected"]], \
            {n: r.to_dict() for n, r in built.checks.items()}
    one, three = run("run", 1), run("run3", 3)
    assert one == three and fake_datagen.ACTIVE["peak"] > 1


def test_a_changed_set_refuses_the_existing_state(root):
    build(make(root), root)
    with pytest.raises(Exception, match="made by another spec"):
        build(make(root, {"heldout": HELDOUT | {"count": 3}}), root)


# ------------------------------------------------------------------ train
def test_the_train_set_spreads_over_the_cells_and_starts_from_the_training_check_dirs(root):
    spec = make(root, {"train": {"from": "train", "count": 3}})
    dirs = []
    for s in range(3):
        d = root / "train_run" / f"seed_{s}"
        d.mkdir(parents=True)
        dirs.append(d)
    xs = [-0.4, -0.3, 0.0, 0.4]                                       # cells 0, 0, 1, 2
    d3 = root / "train_run" / "seed_3"
    d3.mkdir()
    training_manifest(spec, root, [(0, xs[0], None, dirs[0]), (1, xs[1], None, dirs[1]), (2, xs[2], None, dirs[2]),
                                   (3, xs[3], None, d3)])
    built = build(spec, root, "train", plugin=TrainSet)
    assert [i.config.seed for i in built.items] == [0, 2, 3]          # one per cell before a second of cell 0
    assert [i.start_dir for i in built.items] == [dirs[0], dirs[2], d3] and built.checks == {}


def test_the_train_set_needs_the_manifest_and_its_start_dirs(root):
    spec = make(root, {"train": {"from": "train", "count": 1}})
    with pytest.raises(EvalError, match="needs the training manifest .*train_run/state.json"):
        build(spec, root, "train", plugin=TrainSet)
    training_manifest(spec, root, [(0, 0.0, None, root / "missing")])
    with pytest.raises(EvalError, match="no start dir for training seed 0"):
        build(spec, root, "train", plugin=TrainSet)
    spec = make(root, {"train": {"from": "train", "count": 3}})
    with pytest.raises(EvalError, match="1 usable configs"):
        build(spec, root, "train", plugin=TrainSet)


# ------------------------------------------------------------------ checks
def view(values, name="heldout"):
    used = [{"config": {"split": name, "seed": s, "cell": [], "values": v}, "generate": {"ok": True}} for s, v in values]
    return {"splits": {name: {"sampled": [], "used": used, "rejected": []}}}


def test_in_range_checks_every_value_against_the_training_dims():
    dims = {"a.x": Range(-1, 1), "a.c": Choices((1, 2))}
    check = checks.InRange("heldout", dims)
    assert check.run(view([(1, {"a.x": 1.0, "a.c": 2})])).ok
    r = check.run(view([(1, {"a.x": 1.01, "a.c": 2}), (2, {"a.x": 0.0, "a.c": 3})]))
    assert not r.ok and "seed 1 a.x=1.01" in r.reason and "seed 2 a.c=3" in r.reason


def test_seeds_disjoint_names_the_owner_of_a_shared_seed():
    v = view([(5, {})])
    v["splits"]["heldout"]["sampled"] = [{"split": "heldout", "seed": 6, "cell": [], "values": {}}]
    check = checks.SeedsDisjoint("heldout", {"training": {1, 2}, "set 'other'": {6, 7}})
    r = check.run(v)
    assert not r.ok and r.reason == "seeds also used by set 'other' [6]"
    assert checks.SeedsDisjoint("heldout", {"training": {1, 2}}).run(v) == Result()


# ------------------------------------------------------------------ cli on the real plugins
def evaluate(root, raw, *args):
    (root / "spec.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    return cli.main([str(root / "spec.yaml"), *args], root=root)


def test_the_cli_samples_and_checks_the_sets(root, capsys):
    spec = make(root, {"heldout": HELDOUT | {"min_distance": {"value": 0.1}}, "train": {"from": "train", "count": 1}})
    d = root / "train_run" / "seed_0"
    d.mkdir(parents=True)
    training_manifest(spec, root, [(0, 0.0, None, d)])
    raw = SPEC | {"sets": {"heldout": HELDOUT | {"min_distance": {"value": 0.1}}, "train": {"from": "train", "count": 1}}}
    assert evaluate(root, raw, "check") == 0
    out = capsys.readouterr().out
    assert "sample heldout: 4 configs" in out and "sample train: 1 configs" in out
    for c in ("in_range", "seeds_disjoint", "min_distance"):
        assert f"check heldout {c}: PASS" in out
    saved = json.loads((root / "run" / "results.json").read_text())["sets"]["heldout"]["checks"]
    assert saved["in_range"]["ok"] and set(saved) == {"in_range", "seeds_disjoint", "min_distance"}
    assert evaluate(root, raw, "check") == 0 and len(fake_datagen.CALLS) == 4          # resumed: nothing generated again


def test_the_cli_exits_1_on_a_failed_check(root, capsys):
    spec = make(root)
    training_manifest(spec, root, [(105, 0.0, None, None)])
    assert evaluate(root, SPEC, "check") == 1
    assert "check heldout seeds_disjoint: FAIL (seeds also used by training [105])" in capsys.readouterr().out


def test_the_cli_reports_a_missing_manifest_for_a_realised_distance(root, capsys):
    make(root)
    raw = SPEC | {"sets": {"heldout": HELDOUT | {"min_distance": {"value": 0.1, "realised": True}}}}
    assert evaluate(root, raw, "sample") == 1
    assert "needs the training manifest" in capsys.readouterr().err
