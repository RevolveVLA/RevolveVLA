"""plugins/sets (heldout, train) + core/sets.py + core/checks.py on scene instances (CPU: a fake instancer that writes a
scene.yaml per seed, datagen's fake validator, tiny hand-written training states)."""
import copy
import json
import time
from pathlib import Path

import fake_datagen
import pytest
import yaml
from datagen.core.interfaces import Result
from datagen.core.state import State
from datagen.core.pool import PoolExhausted
from datagen.core.samplers import distance
from datagen.core.spec import Choices, Range

from evaluation import cli
from evaluation.core import checks, sets, spec as S
from evaluation.core.errors import EvalError
from evaluation.plugins.sets import heldout
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
CALLS = []              # seeds whose instance was written, in order
ACTIVE = {"now": 0, "peak": 0}
KILL_AFTER = [None]     # raise KeyboardInterrupt on the write after this many


def rejecting(*seeds) -> dict:
    """The training spec's validator turning `seeds` down (a collision)."""
    return TRAIN | {"validators": [{"plugin": "fake_datagen", "reject_seeds": list(seeds)}]}


class SceneWriter(fake_datagen.FakeInstancer):
    """What datagen's scene_instance does, without a scene: <out>/scene.yaml naming the seed; `delay_s` per seed."""

    def __init__(self, delays=None):
        super().__init__({})
        self.delays = delays or {}

    def write(self, config, out):
        if KILL_AFTER[0] is not None and len(CALLS) >= KILL_AFTER[0]:
            raise KeyboardInterrupt("simulated kill")
        ACTIVE["now"] += 1
        ACTIVE["peak"] = max(ACTIVE["peak"], ACTIVE["now"])
        time.sleep(self.delays.get(config.seed, 0))
        out.mkdir(parents=True, exist_ok=True)
        (out / "scene.yaml").write_text(yaml.safe_dump({"seed": config.seed}))
        CALLS.append(config.seed)
        ACTIVE["now"] -= 1
        return Result(paths={"scene": str(out / "scene.yaml")})


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_DATAGEN_LOG", str(tmp_path / "gen.log"))
    fake_datagen.CALLS.clear(), CALLS.clear()
    KILL_AFTER[0] = None
    ACTIVE.update(now=0, peak=0)
    monkeypatch.setattr(heldout, "instancer_for", lambda spec: SceneWriter())
    return tmp_path


def make(root, sets_=None, train=TRAIN, **edits) -> S.EvalSpec:
    (root / "train.yaml").write_text(yaml.safe_dump(train))
    raw = copy.deepcopy(SPEC) | edits
    raw["sets"] = copy.deepcopy(sets_ or {"heldout": HELDOUT})
    return S.parse(raw, "eval.yaml", root)


def training_state(spec, root, entries, manifest=None, **extra):
    """`train_run/state.json`: entries = [(seed, x, scene file made under train_run | None)] of split train, used; `manifest`
    = episodes written as train_run/manifest.json."""
    m = State.open(root / "train_run" / "state.json", spec.train)
    rec = m.split("train")
    for seed, x, scene in entries:
        cell = (min(int((x + 0.5) * 3), 2),)
        rec["sampled"].append({"split": "train", "seed": seed, "cell": list(cell), "values": {"fake_datagen.x": x, "fake_datagen.y": 1.5}})
        paths = {}
        if scene:
            (root / "train_run" / scene).parent.mkdir(parents=True, exist_ok=True)
            (root / "train_run" / scene).write_text("scene: {}")
            paths = {"scene": scene}
        rec["used"].append({"config": rec["sampled"][-1], "generate": {"ok": True, "realised": {}, "paths": paths}})
    m.data["splits"].update(extra)
    m.save()
    if manifest:
        (root / "train_run" / "manifest.json").write_text(json.dumps({"task": "t", "robot": "r", "episodes": manifest}))


def build(spec, root, name="heldout", workers=1, plugin=HeldoutSet):
    return plugin(spec).build(spec.sets[name], root / "run", workers)


def state_of(root, out="run", name="heldout"):
    return json.loads((root / out / name / "state.json").read_text())["splits"][name]


# ------------------------------------------------------------------ held-out
def test_heldout_draws_instances_inside_the_training_ranges_and_passes_the_checks(root):
    spec = make(root)
    built = build(spec, root)
    assert [i.config.seed for i in built.items] == sorted({i.config.seed for i in built.items}) and len(built.items) == 4
    assert all(100 <= i.config.seed <= 129 and i.config.split == "heldout" for i in built.items)
    assert all(-0.5 <= i.config.values["fake_datagen.x"] <= 0.5 for i in built.items)
    assert {n: r.ok for n, r in built.checks.items()} == {"in_range": True, "seeds_disjoint": True, "training_configs": True}
    assert "sampled-only" in built.checks["training_configs"].reason
    assert all(i.scene == root / "run" / "heldout" / f"scene_{i.config.seed}" / "scene.yaml" and i.scene.is_file()
               and i.episodes == () for i in built.items)
    assert (root / "run" / "heldout" / "state.json").is_file() and not (root / "run" / "heldout" / "manifest.json").exists()


def test_a_held_out_set_needs_no_generator(root):
    spec = make(root, train=TRAIN | {"generator": {"plugin": "no_such_generator"}})
    assert len(build(spec, root).items) == 4 and fake_datagen.CALLS == []


def test_a_narrower_subrange_is_drawn_from_but_distance_is_measured_in_the_training_dims(root):
    narrow = {"fake_datagen": {"x": [-0.1, 0.1]}}
    spec = make(root, {"heldout": HELDOUT | {"count": 6, "randomisers": narrow, "min_distance": {"value": 0.3}}})
    training_state(spec, root, [(0, 0.0, None), (1, 0.2, None), (2, -0.3, None)])
    built = build(spec, root)
    train = sets.Training.load(spec, root).configs("train")
    dims = spec.train.dims()
    assert all(-0.1 <= i.config.values["fake_datagen.x"] <= 0.1 for i in built.items)
    # x spans only 0.2 of its training range of 1.0, so the 0.3 apart is reached through y (range 1.0): the draw, in
    # the training dims, keeps the distance although the narrower dims alone could not (R2)
    assert min(distance(i.config.values, t.values, dims) for i in built.items for t in train) >= 0.3
    assert built.checks["min_distance"].ok and "training_configs" not in built.checks
    assert state_of(root)["redrawn"], "the draws that were too close were redrawn"


def test_a_pick_the_validator_rejects_is_replaced_from_the_same_cell(root):
    spec = make(root, {"heldout": HELDOUT | {"count": 2}}, train=rejecting(100, 101))
    built = build(spec, root)
    rec = state_of(root)
    rejected = {e["seed"]: e for e in rec["rejected"]}
    assert set(rejected) == {100, 101} and all(e["step"] == "validate" and e["reason"] == "too close to the wall" for e in rejected.values())
    cells = {c["seed"]: tuple(c["cell"]) for c in rec["sampled"]}
    for r in rejected:                                                # each pick's replacement is in its own cell
        assert any(cells[i.config.seed] == cells[r] and i.config.seed not in (100, 101) for i in built.items)
    assert len(built.items) == 2 and all(r.ok for r in built.checks.values())
    assert not (root / "run" / "heldout" / "scene_100").exists()           # a rejected pick has no instance


def test_pool_exhaustion_is_an_error(root):
    spec = make(root, {"heldout": HELDOUT | {"count": 2}}, train=rejecting(*range(100, 130)))
    with pytest.raises(PoolExhausted):
        build(spec, root)


def test_realised_is_an_alias_the_instance_is_the_realised_start(root):
    spec = make(root, {"heldout": HELDOUT | {"min_distance": {"value": 0.1, "realised": True}}})
    built = build(spec, root)                                         # no training state needed, no error
    assert built.checks["min_distance"].ok and "sampled-only" in built.checks["training_configs"].reason
    assert "realised_min_distance" not in built.checks
    assert not sets.derive(spec, spec.sets["heldout"]).split("heldout").min_distance.realised


def test_a_seed_the_training_run_touched_fails_the_disjoint_check(root):
    spec = make(root, {"heldout": HELDOUT | {"pool": {"initial": [100, 129]}}})
    training_state(spec, root, [(105, 0.0, None)])                    # trained on a seed outside its own pool
    built = build(spec, root)
    assert not built.checks["seeds_disjoint"].ok and "training [105]" in built.checks["seeds_disjoint"].reason
    assert built.checks["in_range"].ok


def test_the_trainings_unrecorded_heldout_split_does_not_count_as_touched(root):
    spec = make(root, {"heldout": HELDOUT | {"pool": {"initial": [20, 49]}}})      # overlaps training's own record:false heldout
    training_state(spec, root, [(0, 0.0, None)],
                   heldout={"sampled": [{"split": "heldout", "seed": 20, "cell": [], "values": {}}], "used": [], "rejected": []})
    assert build(spec, root).checks["seeds_disjoint"].ok


def test_no_state_measures_sampled_distances_to_the_resampled_training_configs(root):
    spec = make(root, {"heldout": HELDOUT | {"min_distance": {"value": 0.1}}})
    built = build(spec, root)
    assert built.checks["min_distance"].ok and "sampled-only" in built.checks["training_configs"].reason


def test_a_state_of_another_training_spec_is_refused(root):
    spec = make(root, {"heldout": HELDOUT | {"min_distance": {"value": 0.1}}})
    training_state(spec, root, [(0, 0.0, None)])
    other = make(root, {"heldout": HELDOUT | {"min_distance": {"value": 0.1}}}, train=TRAIN | {"seed": 5})
    with pytest.raises(Exception, match="made by another spec"):
        build(other, root)


def test_a_set_named_like_its_training_split_is_an_error(root):
    spec = make(root, {"train": HELDOUT | {"min_distance": {"value": 0.1}}})
    with pytest.raises(EvalError, match="has the name of the training split"):
        build(spec, root, "train")


def test_resume_writes_no_instance_twice_and_a_killed_build_continues(root):
    spec = make(root, {"heldout": HELDOUT | {"count": 3}})
    KILL_AFTER[0] = 2
    with pytest.raises(KeyboardInterrupt):
        build(spec, root)
    assert len(state_of(root)["used"]) == 2
    KILL_AFTER[0] = None
    CALLS.clear()
    first = build(spec, root)
    assert len(CALLS) == 1 and len(first.items) == 3                   # only the missing one
    CALLS.clear()
    assert [i.config.seed for i in build(spec, root).items] == [i.config.seed for i in first.items]
    assert CALLS == []


def test_workers_do_not_change_the_result(root, monkeypatch):
    monkeypatch.setattr(heldout, "instancer_for", lambda spec: SceneWriter({100: 0.2, 102: 0.1}))
    spec = make(root, {"heldout": HELDOUT | {"count": 4, "min_distance": {"value": 0.1}}}, train=rejecting(101))
    training_state(spec, root, [(0, 0.0, None), (1, 0.4, None)])

    def run(out, workers):
        built = HeldoutSet(spec).build(spec.sets["heldout"], root / out, workers)
        rec = state_of(root, out)
        return [i.config for i in built.items], [(e["seed"], e["step"], e["reason"]) for e in rec["rejected"]], \
            {n: r.to_dict() for n, r in built.checks.items()}
    one, three = run("run", 1), run("run3", 3)
    assert one == three and ACTIVE["peak"] > 1


def test_a_changed_set_refuses_the_existing_state(root):
    build(make(root), root)
    with pytest.raises(Exception, match="made by another spec"):
        build(make(root, {"heldout": HELDOUT | {"count": 3}}), root)


# ------------------------------------------------------------------ train
def test_the_train_set_spreads_over_the_cells_and_starts_from_the_training_scenes(root):
    spec = make(root, {"train": {"from": "train", "count": 3}})
    xs = [-0.4, -0.3, 0.0, 0.4]                                       # cells 0, 0, 1, 2
    training_state(spec, root, [(s, x, f"train/scene_{s}/scene.yaml") for s, x in enumerate(xs)],
                   manifest=[{"seed": 2, "scene": "train/scene_2/scene.yaml", "episode": 1, "result": "train/scene_2/episode_1.json",
                              "bag": "train/scene_2/episode_1_bag", "passed": True},
                             {"seed": 0, "scene": "train/scene_0/scene.yaml", "episode": 1, "result": "train/scene_0/episode_1.json",
                              "bag": None, "passed": False}])
    built = build(spec, root, "train", plugin=TrainSet)
    train = root / "train_run" / "train"
    assert [i.config.seed for i in built.items] == [0, 2, 3]          # one per cell before a second of cell 0
    assert [i.scene for i in built.items] == [train / f"scene_{s}" / "scene.yaml" for s in (0, 2, 3)] and built.checks == {}
    assert [len(i.episodes) for i in built.items] == [1, 1, 0]        # the manifest's episodes by seed, paths made absolute
    assert built.items[1].episodes[0]["bag"] == str(train / "scene_2" / "episode_1_bag")
    assert built.items[0].episodes[0]["bag"] is None and built.items[0].episodes[0]["passed"] is False


def test_the_train_set_needs_the_state_and_its_scene_files(root):
    spec = make(root, {"train": {"from": "train", "count": 1}})
    with pytest.raises(EvalError, match="needs the training state .*train_run/state.json"):
        build(spec, root, "train", plugin=TrainSet)
    training_state(spec, root, [(0, 0.0, None)])
    with pytest.raises(EvalError, match=r"no scene instance for training seed 0 \(generate.paths.scene missing\)"):
        build(spec, root, "train", plugin=TrainSet)
    (root / "train_run" / "state.json").unlink()
    training_state(spec, root, [(0, 0.0, "train/scene_0/scene.yaml")])
    (root / "train_run" / "train" / "scene_0" / "scene.yaml").unlink()
    with pytest.raises(EvalError, match=r"no scene instance for training seed 0 \(.*scene_0/scene.yaml\)"):
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
    sets_ = {"heldout": HELDOUT | {"min_distance": {"value": 0.1}}, "train": {"from": "train", "count": 1}}
    spec = make(root, sets_)
    training_state(spec, root, [(0, 0.0, "train/scene_0/scene.yaml")])
    raw = SPEC | {"sets": sets_}
    assert evaluate(root, raw, "check") == 0
    out = capsys.readouterr().out
    assert "sample heldout: 4 configs" in out and "sample train: 1 configs" in out
    for c in ("in_range", "seeds_disjoint", "min_distance"):
        assert f"check heldout {c}: PASS" in out
    saved = json.loads((root / "run" / "results.json").read_text())["sets"]["heldout"]["checks"]
    assert saved["in_range"]["ok"] and set(saved) == {"in_range", "seeds_disjoint", "min_distance"}
    assert evaluate(root, raw, "check") == 0 and len(CALLS) == 4          # resumed: no instance written again
    assert (root / "run" / "heldout" / "state.json").is_file()


def test_the_cli_exits_1_on_a_failed_check(root, capsys):
    spec = make(root)
    training_state(spec, root, [(105, 0.0, None)])
    assert evaluate(root, SPEC, "check") == 1
    assert "check heldout seeds_disjoint: FAIL (seeds also used by training [105])" in capsys.readouterr().out


def test_the_cli_refuses_a_legacy_training_run_and_the_state_log_judges(root, capsys):
    make(root)
    assert evaluate(root, SPEC | {"training": {"plugin": "legacy_layouts", "run": "runs/fridge_base50"}}, "sample") == 2
    assert "training.plugin: a run of the old simulator" in capsys.readouterr().err
    for criterion in ("c1", "c2"):
        assert evaluate(root, SPEC | {"judge": {"criterion": criterion}}, "sample") == 2
        assert f"judge.criterion: {criterion} is not available on the new simulator" in capsys.readouterr().err
