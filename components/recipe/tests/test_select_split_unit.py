"""core/select.py and core/split.py on synthetic items."""
import random

import pytest

from recipe.core.errors import RecipeError
from recipe.core.items import Item
from recipe.core.select import pick, select, spread
from recipe.core.split import assign
from recipe.core.spec import ALL


def item(scene, n=0, source="s", phase=None, dur=10.0, segment=None):
    return Item("phase", source, phase, f"/scene_{scene}/scene.yaml", scene, f"/e/{scene}", f"/e/{scene}/r{n}.json", f"/e/{scene}/b{n}",
                segment, True, None, None, dur)


def test_spread_is_balanced_over_scenes():
    items = [item(1, n) for n in range(6)] + [item(2, 0), item(3, 0), item(3, 1)]
    ordered = spread(items, random.Random(0))
    assert len(ordered) == 9 and len({i.scene for i in ordered[:3]}) == 3          # one of each scene first
    assert {i.config_seed for i in ordered[3:5]} == {1, 3} and {i.config_seed for i in ordered[5:]} == {1}


def test_spread_is_deterministic_and_independent_of_the_input_order():
    items = [item(s, n) for s in range(5) for n in range(3)]
    a = spread(items, random.Random("x"))
    assert a == spread(list(reversed(items)), random.Random("x")) and a != spread(items, random.Random("y"))


def test_pick_amounts():
    items = [item(s, n) for s in range(3) for n in range(2)]
    assert len(pick(items, ALL, "episodes", 30, random.Random(0))) == 6
    assert len(pick(items, 4, "episodes", 30, random.Random(0))) == 4 and len(pick(items, 99, "episodes", 30, random.Random(0))) == 6
    assert pick(items, 0, "episodes", 30, random.Random(0)) == []


def test_frames_are_taken_until_reached():
    items = [item(s, dur=2.0) for s in range(5)] + [item(9, dur=10.0, segment=(1.0, 3.0))]           # 60 frames each, the last 60 too
    got = pick(items, 100, "frames", 30, random.Random(0))
    assert len(got) == 2 and sum(i.frames(30) for i in got) == 120
    assert len(pick(items, 120, "frames", 30, random.Random(0))) == 2 and len(pick(items, 121, "frames", 30, random.Random(0))) == 3


def test_select_makes_a_row_per_take_entry():
    items = {"a": [item(1, 0, "a", "approach"), item(2, 0, "a", "approach"), item(3, 0, "a", "grasp")],
             "b": [item(4, n, "b") for n in range(3)]}
    taken, rows = select(items, {"a": "phase", "b": "datagen"}, {"a": {"approach": 5, "grasp": 1, "open": 2}, "b": {None: ALL}},
                         "episodes", 30, 0)
    assert [(r.source, r.phase, r.asked, r.got, r.available, r.short) for r in rows] == [
        ("a", "approach", 5, 2, 2, True), ("a", "grasp", 1, 1, 1, False), ("a", "open", 2, 0, 0, True), ("b", None, ALL, 3, 3, False)]
    assert len(taken) == 6 and {r.kind for r in rows} == {"phase", "datagen"}


def test_select_is_seeded():
    items = {"a": [item(s, n, "a") for s in range(6) for n in range(2)]}
    pick_ = lambda seed: [i.result for i in select(items, {"a": "k"}, {"a": {None: 4}}, "episodes", 30, seed)[0]]
    assert pick_(0) == pick_(0) and pick_(0) != pick_(1)


def test_split_groups_by_scene_and_follows_the_fractions():
    items = [item(s, n) for s in range(10) for n in range(3)] + [item(s, 0, source="t") for s in range(10)]
    parts = assign(items, {"train": 0.8, "val": 0.2}, "episodes", 30, 0)
    assert not {i.scene for i in parts["train"]} & {i.scene for i in parts["val"]}
    assert sum(map(len, parts.values())) == 40 and len(parts["val"]) == 8 and len(parts["train"]) == 32
    assert parts == assign(list(reversed(items)), {"train": 0.8, "val": 0.2}, "episodes", 30, 0) != assign(
        items, {"train": 0.8, "val": 0.2}, "episodes", 30, 1)


def test_split_weighs_frames():
    items = [item(0, dur=30.0)] + [item(s, dur=1.0) for s in range(1, 31)]                          # 900 frames vs 30 x 30
    parts = assign(items, {"train": 0.5, "val": 0.5}, "frames", 30, 3)
    frames = [sum(i.frames(30) for i in p) for p in parts.values()]
    assert sum(frames) == 1800 and min(frames) >= 600          # a whole scene (900) is the only granularity: never 31 items vs none


def test_split_with_one_split_keeps_everything():
    items = [item(s) for s in range(3)]
    assert assign(items, {"train": 1.0}, "episodes", 30, 0) == {"train": sorted(items, key=lambda i: i.scene)}


def test_frames_of_an_item_without_duration_fail():
    with pytest.raises(RecipeError):
        pick([item(1, dur=None)], 10, "frames", 30, random.Random(0))
