"""plugins/trainings: `datagen` (state.json + manifest.json of a v2 run); the report's seen / new labels."""
from pathlib import Path

import pytest

from evaluation.core import results, sets
from evaluation.core.errors import EvalError

import test_sets_unit as T


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_DATAGEN_LOG", str(tmp_path / "gen.log"))
    T.fake_datagen.CALLS.clear()
    return tmp_path


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
