"""plugins/trainings: `datagen` (state.json + manifest.json of a v2 run), and `legacy_layouts` on a tiny synthetic legacy run
dir (LEGACY in conftest: the spec refuses it now, PRD 07 deletes those tests); the report's seen / new labels."""
import json
from pathlib import Path

import numpy as np
import pytest
from datagen.core.spec import SpecError

from evaluation.core import results, sets, spec as S
from evaluation.core.errors import EvalError
from evaluation.plugins.sets.heldout import HeldoutSet
from evaluation.plugins.sets.train import TrainSet
from evaluation.plugins.trainings import legacy_layouts

import test_sets_unit as T

REPO = Path(__file__).resolve().parents[3]
X, Y = "fake_datagen.x", "fake_datagen.y"
LEGACY = {"plugin": "legacy_layouts", "run": "legacy"}
ACCEPTED = {0: 0, 1: 1, 3: 0}                       # legacy seed -> accepted attempt; seed 2 has no accepted episode


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_DATAGEN_LOG", str(tmp_path / "gen.log"))
    T.fake_datagen.CALLS.clear()
    return tmp_path


def legacy_run(root, **layouts):
    """root/legacy: layouts.json (4 seeds in the pool, cells over x's 3 bins) + turn_0/data with plans and check dirs."""
    xs = {0: -0.4, 1: -0.3, 2: 0.0, 3: 0.4}
    pool = [{"seed": s, "cell": min(int((x + 0.5) * 3), 2), "layout": {X: x, Y: 1.5}} for s, x in xs.items()]
    split = {"strategy": "stratified", "target": 3, "space": {X: [-0.5, 0.5], Y: [1, 2]}, "bins": {X: 3}, "pool": pool,
             "rejected": []} | layouts
    (root / "legacy").mkdir()
    (root / "legacy" / "layouts.json").write_text(json.dumps({"splits": {"train": split}}))
    data, seeds = root / "legacy" / "turn_0" / "data", {}
    for s, attempt in ACCEPTED.items():
        plan = data / "plans" / f"seed_{s}"
        plan.mkdir(parents=True)
        (plan / "handoff.json").write_text(json.dumps({"starts": {"box_centre_obj": [1.0, 0.0], "facing_yaw_obj": 0.0}}))
        traj = np.zeros((attempt + 1, 2, 12))
        traj[attempt, 0, :3] = (1.0 + 0.1 * s, 0.0, 0.0)                  # x offset 0.1 * seed
        np.savez(plan / "handoff.npz", traj_robot_obj=traj, attempt_id=np.arange(attempt + 1))
        (data / "checks" / f"seed_{s}_a{attempt}").mkdir(parents=True)
        seeds[str(s)] = {"plan": {"dir": f"plans/seed_{s}"},
                         "episodes": [{"seed": s, "attempt_id": attempt, "accepted": True, "check": f"checks/seed_{s}_a{attempt}"}]}
    seeds["2"] = {"plan": {"dir": "plans/seed_2"}, "episodes": [{"seed": 2, "attempt_id": 0, "accepted": False}]}
    (data / "manifest.json").write_text(json.dumps({"task": "t", "seeds": seeds}))


def make(root, **edits):
    return T.make(root, {"heldout": T.HELDOUT}, training=LEGACY, **edits)


# ------------------------------------------------------------------ the plugin
def test_the_legacy_run_becomes_a_manifest_shaped_training_run(root):
    legacy_run(root)
    training = sets.Training.load(make(root), root)
    rec = training.rec("train")
    assert not training.sampled_only and len(rec["sampled"]) == 4 and rec["rejected"] == []
    assert [e["config"]["seed"] for e in rec["used"]] == [0, 1, 3]                  # the accepted ones only
    e = rec["used"][2]
    assert e["config"] == {"split": "train", "seed": 3, "cell": [2, 0], "values": {X: 0.4, Y: 1.5}}
    assert e["generate"]["ok"] and e["record"]["ok"]
    assert e["generate"]["realised"]["robot_base.x_m"] == pytest.approx(0.3)           # x - box centre
    assert e["generate"]["paths"]["check"] == "legacy/turn_0/data/checks/seed_3_a0"
    assert rec["used"][1]["generate"]["paths"]["check"] == "legacy/turn_0/data/checks/seed_1_a1"      # the accepted attempt
    assert [x["generate"]["realised"]["robot_base.x_m"] for x in rec["used"]][:2] == pytest.approx([-0.0, 0.1])
    assert not (root / "train_run").exists()                                          # nothing written
    assert training.touched() >= {0, 1, 2, 3, 9}


@pytest.mark.parametrize("edit, why", [
    ({"space": {X: [-0.5, 0.4], Y: [1, 2]}}, "space"), ({"target": 5}, "target"), ({"bins": {X: 2}}, "bins"),
    ({"strategy": "random"}, "strategy")])
def test_legacy_ranges_that_differ_from_the_training_spec_are_an_error(root, edit, why):
    legacy_run(root, **edit)
    with pytest.raises(EvalError, match=f"not the training spec's run.*{why}: legacy"):
        sets.Training.load(make(root), root)


def test_a_legacy_run_of_another_task_is_an_error(root):
    legacy_run(root)
    m = root / "legacy" / "turn_0" / "data" / "manifest.json"
    m.write_text(json.dumps(json.loads(m.read_text()) | {"task": "other"}))
    with pytest.raises(EvalError, match="task: legacy other, spec t"):
        sets.Training.load(make(root), root)


def test_missing_files_are_errors(root):
    with pytest.raises(EvalError, match="cannot read .*layouts.json"):
        sets.Training.load(make(root), root)
    legacy_run(root)
    (root / "legacy" / "turn_0" / "data" / "checks" / "seed_1_a1").rmdir()
    with pytest.raises(EvalError, match="1 start dirs are missing, first .*seed_1_a1"):
        sets.Training.load(make(root), root)
    with pytest.raises(EvalError, match="needs `run:`"):
        legacy_layouts.LegacyLayouts(make(root), {})


def test_the_default_plugin_reads_the_datagen_state(root):
    spec = T.make(root)
    assert spec.training.plugin == "datagen" and sets.Training.load(spec, root).sampled_only
    T.training_state(spec, root, [(0, 0.0, "train/scene_0/scene.yaml")])
    training = sets.Training.load(spec, root)
    assert [c.seed for c in training.configs("train")] == [0] and training.episodes == {}      # no manifest.json: no episodes
    assert training.base == root / "train_run" and training.path == root / "train_run" / "state.json"


def test_the_manifest_episodes_are_grouped_by_seed_with_absolute_paths(root):
    spec = T.make(root)
    episode = lambda seed, n, bag: {"seed": seed, "scene": f"train/scene_{seed}/scene.yaml", "episode": n,
                                    "result": f"train/scene_{seed}/episode_{n}.json", "bag": bag, "passed": bag is not None}
    T.training_state(spec, root, [(0, 0.0, None)], manifest=[episode(0, 1, None), episode(0, 2, "train/scene_0/episode_2_bag"),
                                                             episode(3, 1, "train/scene_3/episode_1_bag")])
    eps = sets.Training.load(spec, root).episodes
    base = root / "train_run"
    assert sorted(eps) == [0, 3] and [e["episode"] for e in eps[0]] == [1, 2] and eps[0][0]["bag"] is None
    assert eps[0][1]["bag"] == str(base / "train/scene_0/episode_2_bag") and eps[3][0]["scene"] == str(base / "train/scene_3/scene.yaml")
    (base / "manifest.json").write_text("{not json")
    with pytest.raises(EvalError, match="cannot read its episodes"):
        sets.Training.load(spec, root)


def test_legacy_layouts_is_a_spec_error_on_the_new_simulator(root):
    with pytest.raises(SpecError, match=r"training.plugin: a run of the old simulator .*training: \{plugin: datagen\}"):
        make(root)


def test_the_training_source_is_in_the_spec_hash(root):
    assert make(root).hash != T.make(root).hash
    assert make(root).hash != T.make(root, training={"plugin": "legacy_layouts", "run": "other"}).hash


# ------------------------------------------------------------------ the sets on top of it
def test_heldout_realised_distance_is_measured_to_the_legacy_realised_starts(root):
    legacy_run(root)                                     # realised x offsets of the training seeds: 0.0, 0.1, 0.3
    train = T.TRAIN | {"generator": {"plugin": "fake_datagen", "realised": {100: 0.05, 101: 0.0} | {s: 0.7 for s in range(102, 130)}}}
    real = lambda plan, attempt: {X: 0.1 * int(plan.name[5:])}              # the dim the fake generator realises
    spec = T.make(root, {"heldout": T.HELDOUT | {"count": 2, "min_distance": {"value": 0.1, "realised": True}}},
                  train=train, training=LEGACY)
    legacy_layouts._realised, saved = real, legacy_layouts._realised
    try:
        built = T.build(spec, root)
    finally:
        legacy_layouts._realised = saved
    manifest = json.loads((root / "run" / "heldout" / "manifest.json").read_text())["splits"]["heldout"]
    assert {e["seed"] for e in manifest["rejected"]} == {100, 101}               # 0.05 and 0.0 are within 0.1 of a training start
    assert all(e["step"] == "realised" for e in manifest["rejected"])
    assert "sampled-only" not in "".join(r.reason for r in built.checks.values()) and "realised_min_distance" in built.checks
    assert all(r.ok for r in built.checks.values())


def test_the_train_set_starts_from_the_legacy_check_dirs(root):
    legacy_run(root)
    spec = T.make(root, {"train": {"from": "train", "count": 3}}, training=LEGACY)
    built = T.build(spec, root, "train", plugin=TrainSet)
    data = root / "legacy" / "turn_0" / "data" / "checks"
    assert [i.config.seed for i in built.items] == [0, 3, 1]                    # one per cell first
    assert [i.start_dir for i in built.items] == [data / "seed_0_a0", data / "seed_3_a0", data / "seed_1_a1"]


# ------------------------------------------------------------------ the report
def entry(seed, first, final):
    return {"seed": seed, "passed": final, "tries": [{"passed": first, "record": {"failed_phase": "grasp"}}] + (
        [{"passed": final, "record": {}}] if not first else [])}


def test_the_report_labels_the_sets_and_pools_them_with_the_seen_vs_new_gap():
    r = results.Results(Path("x"), {"spec_hash": "h", "ckpt": None, "sets": {
        "heldout": {"rollouts": {f"{s}_r0": entry(s, s < 1, s < 2) for s in range(4)}},
        "train": {"rollouts": {f"{s}_r0": entry(s, s < 3, True) for s in range(4)}}}})
    text = r.report("c2", ["heldout", "train"], {"heldout": "new", "train": "seen"})
    assert "## heldout (new)" in text and "## train (seen)" in text and "## all sets" in text
    assert "| all sets | 8 | 4 (50%) | 6 (75%) | 0 |" in text and "| seen | 4 | 3 (75%) | 4 (100%) | 0 |" in text
    assert "| new | 4 | 1 (25%) | 2 (50%) | 0 |" in text
    assert "| gap seen - new | | +50 pts | +50 pts | |" in text
    one = r.report("c2", ["train"], {"train": "seen"})
    assert "## all sets" not in one
    assert "(seen)" not in r.report("c2", ["train"])


# ------------------------------------------------------------------ the real legacy run (read only)
LEGACY_RUN = REPO / "runs" / "fridge_base50"


@pytest.mark.skipif(not (LEGACY_RUN / "layouts.json").exists(), reason="runs/fridge_base50 is not on this machine")
def test_the_real_fridge_base50_run_loads():
    spec = S.load(REPO / "specs" / "eval" / "fridge_base50_30k.yaml", REPO)
    assert spec.training.plugin == "legacy_layouts"
    training = sets.Training.load(spec, REPO)
    used = training.rec("train")["used"]
    assert sorted(e["config"]["seed"] for e in used) == list(range(50)) and not training.sampled_only
    assert all(e["generate"]["realised"].get("robot_base.x_m") is not None for e in used)
    assert all((REPO / e["generate"]["paths"]["check"]).is_dir() for e in used)
    assert len({tuple(e["config"]["cell"]) for e in used}) == 27
