"""dataset.labels (DATASET-v2.md section 6 "Labels"): the label set and types, the split derivation, demo vs recovery,
fixed types (a bool is not an int, None never), columns / of_row / select."""
import copy

import pytest

from dataset import labels as L

NAMES = ("source", "seed", "split", "scene", "episode", "expert", "attempt", "turn", "manifest")
TYPES = {"source": str, "seed": int, "split": str, "scene": str, "episode": int, "expert": str, "attempt": int,
         "turn": int, "manifest": str}
MANIFEST = "runs/fridge/datagen/manifest.json"


def demo(**over):
    e = {"seed": 7, "scene": "train/scene_7/scene.yaml", "robot": "mobile_panda", "task": "open_fridge_door",
         "episode": 1, "result": "train/scene_7/episode_1.json", "bag": "train/scene_7/episode_1_bag", "passed": True,
         "datagen": {"expert": "moveit_ompl_pilz_v1", "attempt": 1, "wall_s": 38.7}}
    e.update(over)
    return e


def recovery(**block):
    e = demo()
    del e["datagen"]
    e["recovery"] = dict(block)
    return e


# ------------------------------------------------------------------ constants

def test_prefix_names_types():
    # labels: PREFIX, NAMES (in order) and TYPES exactly as in DATASET-v2 6's label table
    assert L.PREFIX == "labels/"
    assert tuple(L.NAMES) == NAMES
    assert dict(L.TYPES) == TYPES


def test_label_error_is_value_error():
    # labels: class LabelError(ValueError)
    assert issubclass(L.LabelError, ValueError)


# ------------------------------------------------------------------ split_of

@pytest.mark.parametrize("scene, split", [("train/scene_7/scene.yaml", "train"), ("val/scene_1/scene.yaml", "val"),
                                          ("heldout/scene_2/scene.yaml", "heldout"),
                                          ("test/a/scene_3/scene.yaml", "test")])
def test_split_from_scene(scene, split):
    # DATASET-v2 6 --split: no `split` field -> the first component of a relative <split>/<scene dir>/<file> path
    assert L.split_of({"scene": scene}) == split


def test_split_field_wins_over_scene():
    # DATASET-v2 6 --split: the entry's `split` field if the producer writes one
    assert L.split_of({"split": "val", "scene": "train/scene_7/scene.yaml"}) == "val"
    assert L.split_of({"split": "test"}) == "test"            # no scene needed when the field is there
    assert L.split_of({"split": "train", "scene": "scene_7/scene.yaml"}) == "train"     # datagen's job layout


@pytest.mark.parametrize("entry", [
    {"scene": "/abs/train/scene_7/scene.yaml"},               # absolute scene
    {"scene": "scene.yaml"},                                   # one component
    {"scene": "scene_7/scene.yaml"},                           # two components: datagen's job layout, no split dir
    {"scene": "heldout/scene.yaml"},                           # two components
    {},                                                        # no scene
    {"scene": None},                                           # None is never accepted
    {"scene": 7},                                              # not a path string
    {"split": 1, "scene": "train/scene_7/scene.yaml"},         # split field not a str
    {"split": None, "scene": "train/scene_7/scene.yaml"},      # split None
    {"split": True, "scene": "train/scene_7/scene.yaml"},      # split bool
], ids=["absolute", "one-component", "two-components-job", "two-components",
        "missing", "scene-none", "scene-int", "split-int", "split-none", "split-bool"])
def test_split_failures(entry):
    # labels.split_of: absolute / one- or two-component / missing scene, or a non-str split -> LabelError
    with pytest.raises(L.LabelError):
        L.split_of(entry)


# ------------------------------------------------------------------ of_entry

def test_demo_entry_labels_exact_and_ordered():
    # DATASET-v2 6 label table: a datagen block -> source demo, expert/attempt from it; keys exactly NAMES, in order
    got = L.of_entry(demo(), turn=3, manifest=MANIFEST)
    assert list(got) == list(NAMES)
    assert got == {"source": "demo", "seed": 7, "split": "train", "scene": "train/scene_7/scene.yaml", "episode": 1,
                   "expert": "moveit_ompl_pilz_v1", "attempt": 1, "turn": 3, "manifest": MANIFEST}
    for k, v in got.items():
        assert type(v) is TYPES[k], (k, v)


def test_of_entry_uses_split_field():
    # DATASET-v2 6: split label = the entry's split field when present
    assert L.of_entry(demo(split="val"), turn=0, manifest=MANIFEST)["split"] == "val"


def test_of_entry_does_not_modify_entry():
    # DATASET-v2 3 rule 5: never modify inputs (the entry dict read from the manifest)
    e = demo()
    before = copy.deepcopy(e)
    L.of_entry(e, turn=0, manifest=MANIFEST)
    assert e == before


def test_recovery_entry_with_expert_and_attempt():
    # labels.of_entry: a recovery block -> source recovery, its expert / attempt when present
    got = L.of_entry(recovery(expert="recovery_v1", attempt=4), turn=1, manifest=MANIFEST)
    assert got["source"] == "recovery" and got["expert"] == "recovery_v1" and got["attempt"] == 4
    assert list(got) == list(NAMES)


def test_recovery_entry_defaults():
    # labels.of_entry: recovery without expert / attempt -> "" / -1
    got = L.of_entry(recovery(), turn=1, manifest=MANIFEST)
    assert got["source"] == "recovery" and got["expert"] == "" and got["attempt"] == -1


def test_neither_block_fails():
    # DATASET-v2 6 source: anything other than a datagen or recovery block fails closed
    e = demo()
    del e["datagen"]
    with pytest.raises(L.LabelError):
        L.of_entry(e, turn=0, manifest=MANIFEST)


def test_unknown_producer_block_fails():
    # DATASET-v2 6 source: an evaluation-style block (neither datagen nor recovery) -> fail closed
    e = demo()
    e["evaluation"] = e.pop("datagen")
    with pytest.raises(L.LabelError):
        L.of_entry(e, turn=0, manifest=MANIFEST)


def test_both_blocks_fail():
    # labels.of_entry: neither or BOTH blocks -> LabelError
    e = demo()
    e["recovery"] = {"expert": "r", "attempt": 0}
    with pytest.raises(L.LabelError):
        L.of_entry(e, turn=0, manifest=MANIFEST)


@pytest.mark.parametrize("missing", ["expert", "attempt"])
def test_datagen_block_requires_expert_and_attempt(missing):
    # labels.of_entry: expert / attempt are required in a datagen block
    e = demo()
    del e["datagen"][missing]
    with pytest.raises(L.LabelError):
        L.of_entry(e, turn=0, manifest=MANIFEST)


@pytest.mark.parametrize("key, value", [
    ("seed", True), ("seed", False), ("episode", True),       # a bool is NOT an int
    ("seed", None), ("episode", None), ("scene", None),       # None is never accepted
    ("seed", "7"), ("episode", 1.0), ("seed", 7.0),           # wrong types
    ("scene", 3),
])
def test_entry_value_types(key, value):
    # labels.of_entry: every value must have its TYPES type; bool is not int; None never
    with pytest.raises(L.LabelError):
        L.of_entry(demo(**{key: value}), turn=0, manifest=MANIFEST)


@pytest.mark.parametrize("key", ["seed", "episode"])
def test_entry_missing_value(key):
    # DATASET-v2 6: labels are never null -> a missing seed / episode cannot be labelled
    e = demo()
    del e[key]
    with pytest.raises(L.LabelError):
        L.of_entry(e, turn=0, manifest=MANIFEST)


@pytest.mark.parametrize("key, value", [("expert", None), ("expert", 5), ("attempt", True), ("attempt", None),
                                        ("attempt", "1"), ("attempt", 1.0)])
def test_datagen_block_value_types(key, value):
    # labels.of_entry: expert str / attempt int from the datagen block; bool not int; None never
    e = demo()
    e["datagen"][key] = value
    with pytest.raises(L.LabelError):
        L.of_entry(e, turn=0, manifest=MANIFEST)


@pytest.mark.parametrize("key, value", [("expert", None), ("attempt", None), ("attempt", True), ("expert", 0)])
def test_recovery_block_value_types(key, value):
    # labels.of_entry: a recovery block's expert / attempt, when present, must have the label's type
    with pytest.raises(L.LabelError):
        L.of_entry(recovery(**{key: value}), turn=0, manifest=MANIFEST)


@pytest.mark.parametrize("turn, manifest", [(True, MANIFEST), (None, MANIFEST), ("0", MANIFEST), (0, None),
                                            (0, 5)])
def test_turn_and_manifest_types(turn, manifest):
    # labels.of_entry: turn int (not bool), manifest str; None never
    with pytest.raises(L.LabelError):
        L.of_entry(demo(), turn=turn, manifest=manifest)


@pytest.mark.parametrize("scene", ["/abs/scene_7/scene.yaml", "scene.yaml", "scene_7/scene.yaml"])
def test_of_entry_split_failure_propagates(scene):
    # labels: a scene the split cannot be derived from (and no split field) -> LabelError
    with pytest.raises(L.LabelError):
        L.of_entry(demo(scene=scene), turn=0, manifest=MANIFEST)


def test_absolute_scene_ok_with_split_field():
    # labels.split_of: the split field wins, so an absolute scene with an explicit split is labelled;
    # scene label = entry.scene as written
    got = L.of_entry(demo(scene="/abs/scene_7/scene.yaml", split="train"), turn=0, manifest=MANIFEST)
    assert got["split"] == "train" and got["scene"] == "/abs/scene_7/scene.yaml"


# ------------------------------------------------------------------ columns / of_row / select

def test_columns_and_of_row_round_trip():
    # labels.columns -> {"labels/<name>": value}; of_row(row) -> the labels dict
    lab = L.of_entry(demo(), turn=2, manifest=MANIFEST)
    cols = L.columns(lab)
    assert cols == {f"labels/{k}": v for k, v in lab.items()}
    row = {"episode_index": 5, "length": 601, "tasks": ["open the fridge door"], **cols}
    back = L.of_row(row)
    assert back == lab and list(back) == list(NAMES)


def rows():
    out = []
    for idx, (src, split, turn) in zip((10, 11, 12, 13), (("demo", "train", 0), ("recovery", "train", 0),
                                                          ("recovery", "val", 1), ("demo", "val", 1)), strict=True):
        lab = L.of_entry(demo(split=split) if src == "demo" else recovery(expert="r", attempt=0),
                         turn=turn, manifest=MANIFEST)
        lab["split"] = split
        out.append({"episode_index": idx, "length": 100, **L.columns(lab)})
    return out


def test_select_one_label():
    # labels.select: episode indices (row["episode_index"], not row positions) whose labels match
    assert list(L.select(rows(), source="recovery")) == [11, 12]
    assert list(L.select(rows(), source="demo")) == [10, 13]


def test_select_several_labels_all_must_match():
    # labels.select: labels equal EVERY want
    assert list(L.select(rows(), source="recovery", split="val")) == [12]
    assert list(L.select(rows(), source="demo", turn=1)) == [13]
    assert list(L.select(rows(), source="demo", turn=5)) == []


def test_select_no_want_is_all():
    # labels.select: no constraint -> every row
    assert list(L.select(rows())) == [10, 11, 12, 13]


def test_select_unknown_label_is_value_error():
    # labels.select: an unknown label name -> ValueError
    with pytest.raises(ValueError):
        L.select(rows(), colour="red")
    with pytest.raises(ValueError):
        L.select(rows(), **{"labels/source": "demo"})     # the column name is not a label name
