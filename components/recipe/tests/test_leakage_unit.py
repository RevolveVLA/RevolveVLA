"""core/leakage.py: the guard on the fake eval run's held-out scenes."""
import pytest

from recipe.core.errors import RecipeError
from recipe.core.leakage import Exclusion
from recipe.core.registry import REGISTRY


def items(root, plugin, **params):
    return REGISTRY.create("source", plugin, "s", params, root).read().items


def test_a_folder_names_its_scenes_and_seeds(root):
    ex = Exclusion.load(["eval/heldout"], root)
    assert ex.seeds == {500, 501} and ex.paths == (str(root / "eval/heldout"),)


def test_rollouts_of_the_excluded_scenes_are_refused(root):
    ex = Exclusion.load(["eval/heldout"], root)
    refused = [i.config_seed for i in items(root, "rollouts", run="eval", passed="any") if ex.refuses(i)]
    assert sorted(refused) == [500, 501]


def test_recovery_of_an_excluded_scene_is_refused(root):
    ex = Exclusion.load(["eval/heldout"], root)
    got = items(root, "recovery", run="recovery", kind="phase")
    assert [i.config_seed for i in got if ex.refuses(i)] == [501]


def test_a_seed_alone_refuses(root):
    ex = Exclusion.load(["eval/heldout/scene_500/scene.yaml"], root)
    assert ex.seeds == {500}
    base = items(root, "datagen", manifest="datagen/manifest.json")
    assert not any(ex.refuses(i) for i in base)
    assert Exclusion([], {2}).refuses(base[1]) and not Exclusion([], {2}).refuses(base[0])


def test_a_folder_of_episodes_refuses_what_is_under_it(root):
    ex = Exclusion([str(root / "datagen/train/scene_3")])
    assert [i.config_seed for i in items(root, "datagen", manifest="datagen/manifest.json") if ex.refuses(i)] == [3]
    assert not Exclusion([str(root / "datagen/train/scene_")]).refuses(items(root, "datagen", manifest="datagen/manifest.json")[2])


@pytest.mark.parametrize("entry", ["eval/nope", "datagen/train/scene_1/episode_0_bag"])
def test_an_exclude_that_excludes_nothing_is_refused(root, entry):
    with pytest.raises(RecipeError, match="no scene.yaml"):
        Exclusion.load([entry], root)
