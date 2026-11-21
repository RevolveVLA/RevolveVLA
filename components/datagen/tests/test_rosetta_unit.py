"""plugins/rosetta with the docker jobs stubbed (no docker, no GPU): the recorder's camera replay of the accepted attempt (argv of
the replay, episode paths, resume, failures), the converter's call of components/dataset (argv, output dir), and a CPU pipeline
automoma -> rosetta recorder -> rosetta converter that records only the `record: true` splits."""
import json
import os
import subprocess
import sys

import pytest

import generate as G
from datagen.core import registry, spec as S
from datagen.core.interfaces import Config, Converter, Recorder, Result
from datagen.core.pipeline import Pipeline, PluginSet
from datagen.plugins import generate_module
from datagen.plugins.automoma import adapter
from datagen.plugins.rosetta import converter, recorder
from test_automoma_unit import PlanStub

TASK = "open_fridge_door"
RAW = {"task": TASK, "robot": "mobile_panda", "out": "unused", "generator": {"plugin": "automoma"},
       "recorder": {"plugin": "rosetta"}, "converter": {"plugin": "rosetta", "format": "lerobot", "turn": 2},
       "randomisers": {"robot_base": {"x_m": [-0.1, 0.1]}},
       "splits": {"train": {"target": 2, "sampler": {"strategy": "random"}, "pool": {"initial": [0, 9]}},
                  "heldout": {"target": 1, "sampler": {"strategy": "random"}, "pool": {"initial": [20, 29]}, "record": False}}}


class FakeLaunch:
    """components/dataset/launch.py: convert(task, episodes, out, turn) -> the launcher's result dict."""

    def __init__(self, exit=0):
        self.exit, self.calls = exit, []

    def convert(self, task, episodes, out, turn):
        self.calls.append((task, episodes, out, turn))
        return dict(exit=self.exit, wall_s=1.0, result=None, dir=out + "/../jobs/dataset_convert_dataset")


def generated(tmp_path, stub, seed=0, **kw):
    """A finished record=False generate of `seed` in tmp_path + the generator's Result for it."""
    G.generate(TASK, [seed], tmp_path, jobs=stub.jobs(), say=lambda m: None, record=False, **kw)
    return Result(True, "", {"attempt_id": stub.planned[0]}, {})


def test_registered_and_import_needs_no_heavy_stack():
    assert issubclass(registry.get("recorder", "rosetta"), Recorder)
    assert issubclass(registry.get("converter", "rosetta"), Converter)
    code = ("import sys; sys.path.insert(0, 'components/datagen/src'); import datagen.plugins.rosetta\n"
            "bad = [m for m in ('torch', 'numpy', 'docker', 'generate') if m in sys.modules]\n"
            "assert not bad, bad")
    r = subprocess.run([sys.executable, "-c", code], cwd=generate_module.COMPONENT.parents[1], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_params_are_checked():
    spec = S.parse(RAW)
    for bad in ({"fps": 20}, {"format": "hdf5"}):
        with pytest.raises(ValueError):
            converter.RosettaConverter(spec, bad)
    with pytest.raises(ValueError):
        recorder.RosettaRecorder(spec, {"cameras": 3})


def test_the_recorder_replays_the_accepted_attempt_with_cameras(tmp_path):
    stub = PlanStub(planned=(1, 2))
    gen = generated(tmp_path, stub)
    res = recorder.RosettaRecorder(S.parse(RAW), {}, jobs=stub.jobs()).record(Config("train", 0), gen, tmp_path)
    assert stub.calls == [("plan", 0), ("check", 0, 1), ("record", 0, 1)]       # the camera replay of the accepted attempt
    ep = tmp_path / "episodes" / "seed_0_a1"
    assert res.ok and res.paths == {"episode": str(ep), "bag": str(ep / "bag")}
    assert json.loads((ep / "episode.json").read_text())["datagen"]["label"] == "demo"
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert [e["label"] for e in m["seeds"]["0"]["episodes"]] == ["demo"] and m["episodes"] == ["seed_0_a1"]


def test_the_replay_argv_has_the_camera_rate_and_the_task(tmp_path, monkeypatch):
    runs = []
    monkeypatch.setattr(G.specs_bridge, "round_blocks", lambda task: {})
    launch = types_launch(runs)
    monkeypatch.setitem(sys.modules, "launch", launch)
    G.docker_replay(TASK, tmp_path / "plans" / "seed_0", 1, tmp_path / "episodes" / "seed_0_a1", tmp_path / "job", 20)
    args, kw = runs[0]
    assert args == ["replay", "--plan", str(tmp_path / "plans" / "seed_0"), "--attempt", "1", "--task", TASK, "--out",
                    str(tmp_path / "episodes" / "seed_0_a1"), "--cameras", "20"] and kw["gpu"] is True


def types_launch(runs):
    import types
    sim = types.SimpleNamespace(run=lambda args, job_dir, **kw: runs.append((args, kw)) or {"exit": 0})
    return types.SimpleNamespace(sim_launch=sim)


def test_resume_skips_an_episode_already_recorded(tmp_path):
    stub = PlanStub(planned=(0,))
    gen = generated(tmp_path, stub)
    rec = recorder.RosettaRecorder(S.parse(RAW), {}, jobs=stub.jobs())
    first = rec.record(Config("train", 0), gen, tmp_path)
    again = rec.record(Config("train", 0), gen, tmp_path)
    assert [c for c in stub.calls if c[0] == "record"] == [("record", 0, 0)] and again == first


def test_a_rejected_recording_fails_with_its_reasons_and_is_retried(tmp_path):
    stub = PlanStub(planned=(0,), bad_bag=(0,))
    gen = generated(tmp_path, stub)
    rec = recorder.RosettaRecorder(S.parse(RAW), {}, jobs=stub.jobs())
    res = rec.record(Config("train", 0), gen, tmp_path)
    assert not res.ok and "bag check failed" in res.reason
    rec.record(Config("train", 0), gen, tmp_path)
    assert [c for c in stub.calls if c[0] == "record"] == [("record", 0, 0)] * 2


def test_an_infrastructure_error_fails_the_episode(tmp_path):
    stub = PlanStub(planned=(0,))
    gen = generated(tmp_path, stub)
    stub.dirs = {"seed_0_a0": "crash"}
    res = recorder.RosettaRecorder(S.parse(RAW), {}, jobs=stub.jobs()).record(Config("train", 0), gen, tmp_path)
    assert not res.ok and "docker died" in res.reason
    assert not recorder.RosettaRecorder(S.parse(RAW), {}).record(Config("train", 0), Result(), tmp_path).ok


def test_the_converter_calls_the_dataset_launcher(tmp_path):
    launch = FakeLaunch()
    eps = [tmp_path / "train" / "episodes" / "seed_0" / "episodes" / "seed_0_a1"]
    res = converter.RosettaConverter(S.parse(RAW), {"format": "lerobot", "turn": 2}, launch).convert(eps, tmp_path / "dataset")
    assert launch.calls == [(TASK, [str(eps[0])], str(tmp_path / "dataset"), 2)]
    assert res.ok and res.paths["dataset"] == str(tmp_path / "dataset") and res.realised == {"episodes": 1}


def test_the_converter_names_the_robot_for_the_dataset_job(tmp_path, monkeypatch):
    """`convert` run alone (no generator loaded) still gives the dataset job $REVOLVE_ROBOT: a task spec is robot-free."""
    monkeypatch.delenv("REVOLVE_ROBOT", raising=False)
    converter.RosettaConverter(S.parse(RAW), {}, FakeLaunch()).convert([tmp_path], tmp_path / "dataset")
    assert os.environ["REVOLVE_ROBOT"] == S.parse(RAW).robot


def test_the_converter_reports_a_failed_job_and_an_empty_list(tmp_path):
    spec = S.parse(RAW)
    res = converter.RosettaConverter(spec, {}, FakeLaunch(exit=1)).convert([tmp_path], tmp_path / "dataset")
    assert not res.ok and "exit 1" in res.reason
    assert not converter.RosettaConverter(spec, {}, FakeLaunch()).convert([], tmp_path / "dataset").ok


def test_the_dataset_launcher_is_the_components():
    assert converter.LAUNCH.parts[-3:] == ("components", "dataset", "launch.py") and converter.LAUNCH.is_file()
    assert "def convert(task, episodes, out, turn" in converter.LAUNCH.read_text()


def test_pipeline_records_and_converts_only_the_record_true_split(tmp_path):
    spec, stub, launch = S.parse(RAW), PlanStub(planned=(0, 1)), FakeLaunch()
    plugins = PluginSet(samplers={"random": registry.create("sampler", "random")},
                        randomisers={"robot_base": registry.create("randomiser", "robot_base")},
                        generator=adapter.AutoMoma(spec, {}, jobs=stub.jobs()),
                        recorder=recorder.RosettaRecorder(spec, {}, jobs=stub.jobs()),
                        converter=converter.RosettaConverter(spec, {"turn": 2}, launch))
    pipe = Pipeline(spec, plugins, tmp_path, log=lambda m: None)
    assert pipe.run(["sample", "generate", "record", "convert"])
    assert sorted(c[1] for c in stub.calls if c[0] == "record") == sorted(
        e["config"]["seed"] for e in json.loads((tmp_path / "manifest.json").read_text())["splits"]["train"]["used"])
    assert all(c[1] < 20 for c in stub.calls if c[0] == "record")               # no heldout (record: false) recording
    [(task, episodes, out, turn)] = launch.calls
    assert out == str(tmp_path / "dataset") and turn == 2 and len(episodes) == 2
    assert all(e.startswith(str(tmp_path / "train" / "episodes")) and e.split("/")[-2] == "episodes" for e in episodes)
    stub.calls.clear()
    assert Pipeline(spec, plugins, tmp_path, log=lambda m: None).run(["record", "convert"])
    assert stub.calls == [] and len(launch.calls) == 1                          # resume: nothing replayed or converted again


def test_the_recorder_asks_for_the_gui_of_a_gui_spec(tmp_path):
    stub = PlanStub(planned=(1,))
    gen = generated(tmp_path, stub)
    stub.replay_kw.clear()
    recorder.RosettaRecorder(S.parse(RAW | {"run": {"headless": False}}), {}, jobs=stub.jobs()).record(
        Config("train", 0), gen, tmp_path)
    assert stub.replay_kw == [{"gui": True}]
