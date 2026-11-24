"""plugins/sources/*: the datagen manifest, the evaluation run and the recovery run read into items (tests/data)."""
import json
import os
from pathlib import Path

import pytest

from datagen.core.spec import SpecError
from recipe.core.errors import RecipeError
from recipe.core.registry import REGISTRY


def source(root, plugin, name="s", **params):
    return REGISTRY.create("source", plugin, name, params, root)


def rel(root, path):
    return os.path.relpath(path, root)


def test_datagen_reads_the_accepted_episodes_with_a_bag(root):
    read = source(root, "datagen", manifest="datagen/manifest.json").read()
    assert [(i.config_seed, i.passed, i.duration_s) for i in read.items] == [(1, True, 20.0), (2, True, 30.0), (3, True, 25.0)]
    assert dict(read.skipped) == {"not_wanted": 1, "no_bag": 1}
    i = read.items[0]
    assert (i.kind, i.source, i.phase, i.segment, i.task, i.robot) == ("datagen", "s", None, None, "open_fridge_door", "mobile_panda")
    assert rel(root, i.scene) == "datagen/train/scene_1/scene.yaml" and rel(root, i.bag) == "datagen/train/scene_1/episode_0_bag"
    assert i.episode_dir == str(root / "datagen/train/scene_1") and i.labels["expert"] == "moveit"


def test_datagen_passed_any_keeps_the_failed_try(root):
    items = source(root, "datagen", manifest="datagen/manifest.json", passed="any").read().items
    assert [i.passed for i in items] == [True, True, False, True]


def test_rollouts_read_one_item_per_try(root):
    read = source(root, "rollouts", run="eval").read()
    assert sorted((i.labels["set"], i.config_seed, i.labels["try"]) for i in read.items) == [
        ("heldout", 500, 0), ("train", 1, 0), ("train", 2, 1)]
    assert dict(read.skipped) == {"not_wanted": 2, "error": 1}
    retry = next(i for i in read.items if i.config_seed == 2)
    assert rel(root, retry.scene) == "datagen/train/scene_2/scene.yaml" and retry.duration_s == 31.0 and retry.passed
    assert retry.labels["policy"] == "oracle" and (retry.kind, retry.phase) == ("rollout", None)


def test_rollouts_filter_by_set_and_verdict(root):
    train = source(root, "rollouts", run="eval", set="train", passed=False).read().items
    assert [(i.config_seed, i.labels["failed_phase"]) for i in train] == [(2, "grasp")]
    both = source(root, "rollouts", run="eval", set=["train", "heldout"], passed="any").read().items
    assert len(both) == 5                                                    # the stack error is left out


def test_phase_recovery_items_take_the_scene_of_the_failed_rollout(root):
    src = source(root, "recovery", run="recovery", kind="phase")
    read = src.read()
    assert src.kind == "phase" and dict(read.skipped) == {"not_wanted": 4}
    assert sorted((i.phase, i.config_seed) for i in read.items) == [("approach", 2), ("approach", 2), ("approach", 501), ("grasp", 2)]
    i = read.items[0]
    assert rel(root, i.scene) in ("datagen/train/scene_2/scene.yaml", "eval/heldout/scene_501/scene.yaml")
    assert (i.kind, i.segment, i.labels["trigger"], i.labels["rollout"]) == ("phase", None, "verdict", "2_r0")


def test_dagger_whole_episodes(root):
    items = source(root, "recovery", run="recovery", kind="dagger").read().items
    assert sorted((i.phase, i.segment) for i in items) == [("approach", None), ("grasp", None), ("open", None)]


def test_dagger_interventions_use_the_manifest_segment_or_the_sidecar(root):
    read = source(root, "recovery", run="recovery", kind="dagger", segment="interventions").read()
    assert sorted((i.phase, i.segment) for i in read.items) == [("grasp", (8.0, 30.0)), ("open", (12.0, 40.0))]
    assert dict(read.skipped)["no_segment"] == 1
    assert sorted(i.frames(30) for i in read.items) == [660, 840] and all(i.seconds == i.segment[1] - i.segment[0] for i in read.items)


def test_dagger_sidecar_with_several_expert_spans_makes_an_item_each(root):
    side = root / "recovery/sims/0/d0/episode_0.interventions.json"
    side.write_text(json.dumps({"segments": [[0, 5, "policy"], [5, 9, "expert"], [9, 12, "policy"], [12, 40, "expert"]]}))
    items = source(root, "recovery", run="recovery", kind="dagger", segment="interventions").read().items
    assert sorted(i.segment for i in items if i.phase == "open") == [(5.0, 9.0), (12.0, 40.0)]


def test_the_manifests_segments_list_every_expert_span_and_wins_over_segment_and_sidecar(tmp_path):
    from recipe.plugins.sources.recovery import expert_spans
    (tmp_path / "episode_0.interventions.json").write_text(json.dumps({"segments": [[0, 4, "policy"], [4, 9, "expert"]]}))
    entry = {"recovery": {"segments": [[2, 4], [6, 9.5]], "segment": [2, 4]}}
    assert expert_spans(entry, str(tmp_path / "episode_0.json")) == [(2.0, 4.0), (6.0, 9.5)]
    assert expert_spans({"recovery": {"segment": [2, 4]}}, str(tmp_path / "episode_0.json")) == [(2.0, 4.0)]


def test_each_episode_of_a_shared_sim_folder_has_its_own_sidecar(tmp_path):
    from recipe.plugins.sources.recovery import expert_spans
    (tmp_path / "episode_0.interventions.json").write_text(json.dumps({"segments": [[0, 4, "policy"], [4, 9, "expert"]]}))
    (tmp_path / "episode_1.interventions.json").write_text(json.dumps({"segments": [[0, 7, "policy"], [7, 12, "expert"]]}))
    entry = {"recovery": {}}
    assert expert_spans(entry, str(tmp_path / "episode_1.json")) == [(7.0, 12.0)]
    assert expert_spans(entry, str(tmp_path / "episode_0.json")) == [(4.0, 9.0)]
    assert expert_spans(entry, str(tmp_path / "episode_2.json")) == []


def test_a_broken_sidecar_is_an_error(root):
    (root / "recovery/sims/0/d0/episode_0.interventions.json").write_text('{"segments": 3}')
    with pytest.raises(RecipeError, match="expected segments"):
        source(root, "recovery", run="recovery", kind="dagger", segment="interventions").read()


def test_phase_episodes_stay_whole_with_segment_interventions(root):
    items = source(root, "recovery", run="recovery", kind="phase", segment="interventions").read().items
    assert len(items) == 4 and all(i.segment is None for i in items)


def test_a_missing_run_is_a_recipe_error(root):
    for plugin, params in (("datagen", {"manifest": "nope/manifest.json"}), ("rollouts", {"run": "nope"}), ("recovery", {"run": "nope"})):
        with pytest.raises(RecipeError, match="nope"):
            source(root, plugin, **params).read()


@pytest.mark.parametrize("plugin, params, at", [
    ("datagen", {}, "sources.s: missing key 'manifest'"), ("datagen", {"manifest": "m", "colour": 1}, "sources.s.colour"),
    ("datagen", {"manifest": "m", "passed": "yes"}, "sources.s.passed"), ("rollouts", {"run": "r", "set": 3}, "sources.s.set"),
    ("recovery", {"run": "r", "kind": "human"}, "sources.s.kind"), ("recovery", {"run": "r", "segment": "half"}, "sources.s.segment")])
def test_params_are_checked_with_their_path(root, plugin, params, at):
    with pytest.raises(SpecError, match=at):
        source(root, plugin, **params)


def test_frames_need_a_duration():
    from recipe.core.items import Item
    item = Item("datagen", "s", None, "/s", None, "/e", "/e/r.json", "/e/b", None, True, None, None, None)
    with pytest.raises(RecipeError, match="no duration_s"):
        item.frames(30)
    assert item.to_dict(30)["segment"] is None and "frames" not in item.to_dict(30)
