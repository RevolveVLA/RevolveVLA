"""dataset.manifest.read and bag_message_counts (DATASET-v2.md section 6): passed entries only, in manifest order
across manifests, relative and absolute paths, --split (including datagen's job layout), every listed problem, and
failed entries never checked or labelled."""
import hashlib
import json
from pathlib import Path

import pytest

from conftest import DROP, contract_topics, write_bag_metadata
from dataset import manifest as M


def resolved(p):
    return Path(p).resolve()


def edit(entry, key, value):
    """Set (or with DROP remove) a key of an entry already listed (the Producer keeps the same dict)."""
    if value is DROP:
        entry.pop(key, None)
    else:
        entry[key] = value
    return entry


def inputs_problems(plan):
    assert plan.problems, "expected problems"
    for p in plan.problems:
        assert isinstance(p, str) and p.startswith("inputs:"), p       # problems start with the check name
    return plan.problems


# ------------------------------------------------------------------ happy paths

def test_passed_only_in_order(producer):
    # DATASET-v2 6: only passed: true entries, in manifest order
    p = producer()
    e0 = p.episode(passed=False)
    e1 = p.episode()
    e2 = p.episode(passed=False)
    e3 = p.episode()
    m = p.manifest()
    plan = M.read([str(m)])
    assert plan.problems == []
    assert plan.task == "open_fridge_door" and plan.robot == "mobile_panda"
    assert [resolved(e.bag) for e in plan.episodes] == [resolved(m.parent / e["bag"]) for e in (e1, e3)]
    assert [resolved(e.result) for e in plan.episodes] == [resolved(m.parent / e["result"]) for e in (e1, e3)]
    assert [e.labels for e in plan.episodes] == [p.labels(e, m) for e in (e1, e3)]
    assert e0 and e2


def test_episode_fields(producer):
    # manifest.Episode: where (str), bag / result paths, labels (keys = labels.NAMES)
    p = producer()
    p.episode()
    plan = M.read([str(p.manifest())])
    ep = plan.episodes[0]
    assert isinstance(ep.where, str) and ep.where
    assert isinstance(ep.bag, Path) and isinstance(ep.result, Path)
    assert list(ep.labels) == ["source", "seed", "split", "scene", "episode", "expert", "attempt", "turn",
                               "manifest"]


def test_several_manifests_in_given_order(producer):
    # DATASET-v2 6: several manifests in the order given, each in its own order; label manifest = the path as given
    a, b = producer(), producer()
    a0, a1 = a.episode(seed=1), a.episode(seed=1)
    b0, b1 = b.episode(seed=2), b.episode(seed=2)
    ma, mb = a.manifest(), b.manifest()
    plan = M.read([str(mb), str(ma)], turn=4)
    assert plan.problems == []
    want = [(mb, b0), (mb, b1), (ma, a0), (ma, a1)]
    assert [resolved(e.bag) for e in plan.episodes] == [resolved(m.parent / e["bag"]) for m, e in want]
    assert [e.labels for e in plan.episodes] == [b.labels(e, m, turn=4) for m, e in want]


def test_digests_are_sha256_of_each_manifest(producer):
    # manifest.Plan.digests: sha256 hex of each manifest file read, in order
    a, b = producer(), producer()
    a.episode(), b.episode()
    ma, mb = a.manifest(), b.manifest()
    plan = M.read([str(ma), str(mb)])
    assert plan.digests == [hashlib.sha256(m.read_bytes()).hexdigest() for m in (ma, mb)]


def test_absolute_entry_paths(producer):
    # DATASET-v2 6: absolute entry paths are accepted
    p = producer()
    e = p.episode(absolute=True)
    m = p.manifest()
    plan = M.read([str(m)])
    assert plan.problems == []
    assert resolved(plan.episodes[0].bag) == resolved(e["bag"])
    assert resolved(plan.episodes[0].result) == resolved(e["result"])


def test_relative_manifest_path_as_given(producer, tmp_path, monkeypatch):
    # DATASET-v2 6: paths relative to the MANIFEST's directory (not the cwd); label manifest = the path as given
    p = producer(root=tmp_path / "prod" / "datagen")
    e = p.episode()
    p.manifest()
    monkeypatch.chdir(tmp_path)
    plan = M.read(["prod/datagen/manifest.json"])
    assert plan.problems == []
    assert resolved(plan.episodes[0].bag) == resolved(tmp_path / "prod" / "datagen" / e["bag"])
    assert plan.episodes[0].labels["manifest"] == "prod/datagen/manifest.json"


def test_turn_label(producer):
    # DATASET-v2 6 labels: turn = --turn K
    p = producer()
    p.episode()
    plan = M.read([str(p.manifest())], turn=7)
    assert plan.episodes[0].labels["turn"] == 7


def test_split_filter_derived(producer):
    # DATASET-v2 6 --split S: keep only the entries of split S (derived from entry.scene), order kept
    p = producer()
    t0 = p.episode(split="train", seed=0)
    v0 = p.episode(split="val", seed=1)
    t1 = p.episode(split="train", seed=0)
    v1 = p.episode(split="val", seed=1)
    m = p.manifest()
    for split, want in (("train", [t0, t1]), ("val", [v0, v1])):
        plan = M.read([str(m)], split=split)
        assert plan.problems == []
        assert [resolved(e.bag) for e in plan.episodes] == [resolved(m.parent / e["bag"]) for e in want]
        assert all(e.labels["split"] == split for e in plan.episodes)
    assert len(M.read([str(m)]).episodes) == 4                         # no --split: everything passed


def test_split_filter_explicit_field(producer):
    # DATASET-v2 6 --split: an explicit `split` field wins over the scene path
    p = producer()
    p.episode(split="train", seed=0)
    e = edit(p.episode(split="train", seed=0), "split", "val")
    plan = M.read([str(p.manifest())], split="val")
    assert plan.problems == [] and len(plan.episodes) == 1
    assert plan.episodes[0].labels["split"] == "val"
    assert resolved(plan.episodes[0].bag).name == Path(e["bag"]).name


def test_job_manifest_needs_split_fields(producer):
    # DATASET-v2 6 --split: a datagen job manifest's scenes are scene_<seed>/scene.yaml (no split directory), so its
    # passed entries need a `split` field: without one each is a problem naming it (never a split "scene_0"), a failed
    # entry is not labelled at all, and with the fields --split filters on them
    p = producer()
    e0 = p.episode(scene="scene_0/scene.yaml")
    p.episode(passed=False, scene="scene_1/scene.yaml")
    m = p.manifest()
    probs = inputs_problems(M.read([str(m)]))
    assert any(f"{m} entry 0" in x for x in probs) and not any(f"{m} entry 1" in x for x in probs), probs
    e0["split"] = "train"
    p.episode(scene="scene_2/scene.yaml")["split"] = "val"
    m = p.manifest()
    plan = M.read([str(m)])
    assert plan.problems == [] and [e.labels["split"] for e in plan.episodes] == ["train", "val"]
    plan = M.read([str(m)], split="val")
    assert plan.problems == [] and [e.labels["scene"] for e in plan.episodes] == ["scene_2/scene.yaml"]


def test_failed_entries_never_checked_or_labelled(producer):
    # manifest.read: failed entries are never checked further and never labelled, even malformed
    p = producer()
    p.episode(passed=False, block=None)                                # unlabellable
    p.episode(passed=False, block="both")                              # both blocks
    p.episode(passed=False, seed=True)                                 # bool seed
    edit(edit(p.episode(passed=False, bag=False, result=False), "bag", DROP), "result", DROP)
    p.episode(passed=False, task="other_task", robot="other_robot")    # differs from the manifest
    p.episode(passed=False, scene="/abs/scene.yaml")                   # no derivable split
    good = p.episode()
    m = p.manifest()
    plan = M.read([str(m)])
    assert plan.problems == []
    assert [resolved(e.bag) for e in plan.episodes] == [resolved(m.parent / good["bag"])]


def test_read_does_not_modify_manifest(producer):
    # DATASET-v2 3 rule 5: never modify inputs
    p = producer()
    p.episode()
    m = p.manifest()
    before = m.read_bytes()
    M.read([str(m)])
    assert m.read_bytes() == before


def test_read_does_not_scan_folders(producer):
    # DATASET-v2 3 rule 2: no folder scanning: a bag on disk that the manifest does not list is not converted
    p = producer()
    p.episode()
    m = p.manifest()
    p.episode()                       # written to disk after the manifest: not listed
    plan = M.read([str(m)])
    assert len(plan.episodes) == 1


# ------------------------------------------------------------------ problems

def test_missing_manifest(tmp_path):
    # manifest.read problems: an unreadable manifest
    m = tmp_path / "nope" / "manifest.json"
    probs = inputs_problems(M.read([str(m)]))
    assert any(str(m) in x for x in probs)                             # names the manifest


def test_manifest_not_json(tmp_path):
    # manifest.read problems: a non-JSON manifest
    m = tmp_path / "manifest.json"
    m.write_text("{not json")
    probs = inputs_problems(M.read([str(m)]))
    assert any(str(m) in x for x in probs)


@pytest.mark.parametrize("doc", [[], "manifest", 3, None], ids=["list", "str", "int", "null"])
def test_manifest_not_an_object(tmp_path, doc):
    # manifest.read problems: a JSON document that is no manifest (no `episodes` list)
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps(doc))
    inputs_problems(M.read([str(m)]))


@pytest.mark.parametrize("episodes", [DROP, None, {}, "x"], ids=["missing", "null", "dict", "str"])
def test_no_episodes_list(producer, episodes):
    # manifest.read problems: no `episodes` list
    p = producer()
    p.episode()
    inputs_problems(M.read([str(p.manifest(episodes=episodes))]))


@pytest.mark.parametrize("key", ["task", "robot"])
def test_manifest_without_task_or_robot(producer, key):
    # manifest.read problems: a manifest without task or robot
    p = producer()
    p.episode()
    m = p.manifest(**{key: DROP})
    probs = inputs_problems(M.read([str(m)]))
    assert any(str(m) in x for x in probs)


@pytest.mark.parametrize("key, value", [("task", "open_drawer"), ("robot", "mobile_so101")])
def test_manifests_differ(producer, key, value):
    # DATASET-v2 6: every manifest must have the same task and robot
    a, b = producer(), producer()
    a.episode()
    b.episode(**{key: value})
    ma, mb = a.manifest(), b.manifest(**{key: value})
    inputs_problems(M.read([str(ma), str(mb)]))


@pytest.mark.parametrize("key, value", [("task", "open_drawer"), ("robot", "mobile_so101")])
def test_passed_entry_differs_from_manifest(producer, key, value):
    # DATASET-v2 6: every (passed) entry has the manifest's task / robot; the problem names manifest and entry
    p = producer()
    p.episode()
    p.episode(**{key: value})
    m = p.manifest()
    probs = inputs_problems(M.read([str(m)]))
    assert any(f"{m} entry 1" in x for x in probs), probs


@pytest.mark.parametrize("passed", ["true", 1, 0, None, DROP], ids=["str", "one", "zero", "null", "missing"])
def test_passed_not_a_bool(producer, passed):
    # manifest.read problems: an entry whose `passed` is not a bool (1 / 0 are not bools either)
    p = producer()
    p.episode()
    edit(p.episode(), "passed", passed)
    m = p.manifest()
    probs = inputs_problems(M.read([str(m)]))
    assert any(f"{m} entry 1" in x for x in probs), probs


@pytest.mark.parametrize("kw", [dict(block=None), dict(block="both"), dict(seed=True), dict(seed=None),
                                dict(episode="0"), dict(scene="/abs/scene_0/scene.yaml"), dict(scene="scene.yaml"),
                                dict(expert=None)],
                         ids=["no-block", "both-blocks", "bool-seed", "null-seed", "str-episode", "abs-scene",
                              "one-component-scene", "null-expert"])
def test_passed_entry_cannot_be_labelled(producer, kw):
    # manifest.read problems: a passed entry that cannot be labelled (LabelError) -> problem, not a skip
    p = producer()
    p.episode()
    p.episode(**kw)
    m = p.manifest()
    probs = inputs_problems(M.read([str(m)]))
    assert any(f"{m} entry 1" in x for x in probs), probs


@pytest.mark.parametrize("key, value", [("bag", DROP), ("result", DROP), ("bag", None), ("result", 5),
                                        ("bag", ["a"])])
def test_passed_entry_without_bag_or_result_string(producer, key, value):
    # manifest.read problems: a passed entry without a `bag` or `result` string
    p = producer()
    p.episode()
    edit(p.episode(), key, value)
    m = p.manifest()
    probs = inputs_problems(M.read([str(m)]))
    assert any(f"{m} entry 1" in x for x in probs), probs


def test_zero_passed(producer):
    # manifest.read problems: zero passed entries in total
    p = producer()
    p.episode(passed=False)
    p.episode(passed=False)
    inputs_problems(M.read([str(p.manifest())]))


def test_empty_episodes_list(producer):
    # manifest.read problems: zero passed entries (an empty list)
    p = producer()
    inputs_problems(M.read([str(p.manifest(entries=[]))]))


def test_zero_passed_after_split_filter(producer):
    # manifest.read problems: zero passed entries AFTER the split filter
    p = producer()
    p.episode(split="train")
    p.episode(split="val", passed=False)
    inputs_problems(M.read([str(p.manifest())], split="val"))
    inputs_problems(M.read([str(p.manifest())], split="test"))


def test_zero_passed_over_several_manifests_is_a_total(producer):
    # manifest.read: "zero passed entries in total": one manifest with none passed is fine if another has some
    a, b = producer(), producer()
    a.episode(passed=False)
    b.episode()
    plan = M.read([str(a.manifest()), str(b.manifest())])
    assert plan.problems == [] and len(plan.episodes) == 1


def test_problem_in_second_manifest_names_it(producer):
    # an input problem names the manifest path and the entry index
    a, b = producer(), producer()
    a.episode()
    b.episode(block=None)
    ma, mb = a.manifest(), b.manifest()
    probs = inputs_problems(M.read([str(ma), str(mb)]))
    assert any(f"{mb} entry 0" in x for x in probs), probs
    assert not any(f"{ma} entry" in x for x in probs), probs


def test_entry_index_counts_failed_entries(producer):
    # "<manifest> entry <i>": i read as the index in the manifest's `episodes` list (failed entries count)
    p = producer()
    p.episode(passed=False)
    p.episode(passed=False)
    p.episode(block=None)
    m = p.manifest()
    probs = inputs_problems(M.read([str(m)]))
    assert any(f"{m} entry 2" in x for x in probs), probs


def test_every_problem_reported(producer):
    # DATASET-v2 3 rule 4 / 8.4: a problem list, one string per problem (not just the first)
    p = producer()
    p.episode(block=None)
    edit(p.episode(), "bag", DROP)
    m = p.manifest()
    probs = inputs_problems(M.read([str(m)]))
    assert any(f"{m} entry 0" in x for x in probs) and any(f"{m} entry 1" in x for x in probs), probs


# ------------------------------------------------------------------ bag_message_counts

def test_bag_message_counts(tmp_path):
    # manifest.bag_message_counts: {topic: count} from metadata.yaml's topics_with_message_count
    counts = {t: i for i, t in enumerate(contract_topics("mobile_panda"))}
    counts["/clock"] = 20000
    write_bag_metadata(tmp_path / "bag", counts)
    assert M.bag_message_counts(tmp_path / "bag") == counts
