"""dataset.record: the meta/revolve_dataset.json record of a dataset directory, host only."""
import json
from pathlib import Path

import pytest

from dataset import record as R


def full_record(**over):
    kw = dict(kind="convert", inputs={"manifests": ["/runs/a/manifest.json"]}, contract="mobile_panda",
              prompt="open the fridge door", episodes=3, frames=1800,
              features={"observation.state": {"dtype": "float32", "shape": [9]}},
              per_episode=[{"episode": 0, "frames": 600}], checks={"frames": "ok"}, warnings=["w"], problems=[],
              frame_tolerance=2, action_late_fails=True)
    kw.update(over)
    return R.Record(**kw)


def write_doc(root, doc):
    (root / "meta").mkdir(parents=True, exist_ok=True)
    path = root / "meta" / "revolve_dataset.json"
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    return path


# the record lives at meta/revolve_dataset.json and its kinds are convert and merge
def test_constants():
    assert R.SUMMARY == Path("meta") / "revolve_dataset.json"
    assert R.KINDS == ("convert", "merge")


# a record built from kind and inputs alone gets the documented defaults
def test_defaults():
    r = R.Record("merge", {"datasets": ["a"]})
    assert (r.contract, r.prompt, r.episodes, r.frames) == (None, None, 0, 0)
    assert (r.features, r.per_episode, r.checks, r.warnings, r.problems) == ({}, [], {}, [], [])
    assert (r.frame_tolerance, r.action_late_fails) == (None, None)


# ok is true only for a known kind with no problems
@pytest.mark.parametrize("kind,problems,ok", [("convert", [], True), ("merge", [], True),
                                              ("convert", ["port: x"], False), ("review", [], False),
                                              ("", [], False)])
def test_ok(kind, problems, ok):
    assert R.Record(kind, {}, problems=problems).ok is ok


# write puts every field in the JSON file, the two optional ones only when set
def test_write_all_fields(tmp_path):
    (tmp_path / "meta").mkdir()
    full_record().write(tmp_path)
    doc = json.loads((tmp_path / R.SUMMARY).read_text())
    assert doc["kind"] == "convert" and doc["frame_tolerance"] == 2 and doc["action_late_fails"] is True
    assert set(doc) == {"kind", "inputs", "contract", "prompt", "episodes", "frames", "features", "per_episode",
                        "checks", "warnings", "problems", "frame_tolerance", "action_late_fails"}


# write leaves out frame_tolerance and action_late_fails when None, but writes contract and prompt as null
def test_write_omits_none_optionals(tmp_path):
    (tmp_path / "meta").mkdir()
    R.Record("merge", {"datasets": []}).write(tmp_path)
    doc = json.loads((tmp_path / R.SUMMARY).read_text())
    assert "frame_tolerance" not in doc and "action_late_fails" not in doc
    assert doc["contract"] is None and doc["prompt"] is None
    assert doc["episodes"] == 0 and doc["problems"] == [] and doc["features"] == {}


# a written record reads back equal, with and without the optional fields
@pytest.mark.parametrize("rec", [full_record(), full_record(frame_tolerance=None, action_late_fails=None),
                                 R.Record("merge", {"datasets": ["x"]}, problems=["merge: bad"]),
                                 full_record(action_late_fails=False, frame_tolerance=0)])
def test_round_trip(tmp_path, rec):
    (tmp_path / "meta").mkdir()
    rec.write(tmp_path)
    assert R.read(tmp_path) == rec


# read gives None for a missing, unparsable or malformed record file
@pytest.mark.parametrize("doc", [None, "{not json", "[]", '"convert"', "null",
                                 {"inputs": {}}, {"kind": 1, "inputs": {}}, {"kind": None, "inputs": {}},
                                 {"kind": "convert"}, {"kind": "convert", "inputs": []},
                                 {"kind": "convert", "inputs": "x"},
                                 {"kind": "convert", "inputs": {}, "problems": "bad"},
                                 {"kind": "convert", "inputs": {}, "problems": {}},
                                 {"kind": "convert", "inputs": {}, "problems": None}])
def test_read_rejects(tmp_path, doc):
    if doc is not None:
        write_doc(tmp_path, doc)
    assert R.read(tmp_path) is None


# read ignores unknown keys and accepts an unknown kind string (which is then not ok)
def test_read_ignores_unknown_keys(tmp_path):
    write_doc(tmp_path, {"kind": "convert", "inputs": {"manifests": ["m"]}, "problems": [], "episodes": 2,
                         "frames": 5, "surprise": {"a": 1}})
    r = R.read(tmp_path)
    assert r is not None and r.kind == "convert" and r.inputs == {"manifests": ["m"]}
    assert (r.episodes, r.frames, r.ok) == (2, 5, True)
    assert not hasattr(r, "surprise")
    write_doc(tmp_path, {"kind": "review", "inputs": {}})
    r = R.read(tmp_path)
    assert r is not None and r.kind == "review" and r.ok is False
