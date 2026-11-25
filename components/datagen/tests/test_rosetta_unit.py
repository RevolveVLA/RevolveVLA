"""plugins/rosetta with the dataset launcher stubbed (no docker): registered and importable without the heavy stack, params
checked, the call of components/dataset (the run's manifest, output dir, turn), failures, and the pipeline's skip of the step."""
import subprocess
import sys

import pytest

from datagen.core import registry, spec as S
from datagen.core.interfaces import Converter
from datagen.core.pipeline import NO_CONVERT, Pipeline, PluginSet
from datagen.plugins.rosetta import converter

TASK = "open_fridge_door"
RAW = {"task": TASK, "robot": "mobile_panda", "out": "unused", "generator": {"plugin": "moveit"},
       "converter": {"plugin": "rosetta", "format": "lerobot", "turn": 2},
       "splits": {"train": {"target": 1, "pool": {"initial": [0, 2]}}}}


class FakeLaunch:
    """components/dataset/launch.py: convert(manifests, out, turn, split=None) -> the launcher's result dict. result:
    the job's result.json (default: a finished job whose progress holds `episodes`)."""

    def __init__(self, exit=0, episodes=1, result=None):
        self.exit, self.calls = exit, []
        self.result = result if result is not None else (
            {"progress": {"stage": "done", "episodes": episodes}} if exit == 0 else None)

    def convert(self, manifests, out, turn, split=None):
        self.calls.append((manifests, out, turn, split))
        return dict(exit=self.exit, wall_s=1.0, result=self.result, dir=out + "/../jobs/dataset_convert_dataset")


def test_registered_and_import_needs_no_heavy_stack():
    assert issubclass(registry.get("converter", "rosetta"), Converter)
    code = ("import sys; sys.path[:0] = ['components/datagen/src', 'common']; import datagen.plugins.rosetta\n"
            "bad = [m for m in ('torch', 'numpy', 'docker') if m in sys.modules]\n"
            "assert not bad, bad")
    r = subprocess.run([sys.executable, "-c", code], cwd=converter.LAUNCH.parents[2], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_spec_accepts_the_converter_and_the_plugin_is_built_from_it():
    spec = S.parse(RAW)
    c = registry.create("converter", spec.converter.plugin, spec, spec.converter.params)
    assert isinstance(c, converter.RosettaConverter) and c.turn == 2


def test_params_are_checked():
    spec = S.parse(RAW)
    for bad in ({"fps": 20}, {"format": "hdf5"}):
        with pytest.raises(ValueError):
            converter.RosettaConverter(spec, bad)


def test_the_converter_calls_the_dataset_launcher(tmp_path):
    launch = FakeLaunch()
    eps = [tmp_path / "train" / "scene_0" / "episode_0_bag"]
    res = converter.RosettaConverter(S.parse(RAW), {"format": "lerobot", "turn": 2}, launch).convert(eps, tmp_path / "dataset")
    # every split of the run's manifest: the plugin interface has no split
    assert launch.calls == [([str(converter.REPO / "unused" / "manifest.json")], str(tmp_path / "dataset"), 2, None)]
    assert res.ok and res.paths["dataset"] == str(tmp_path / "dataset") and res.realised == {"episodes": 1}


def test_the_converter_reports_a_failed_job_and_an_empty_list(tmp_path):
    spec = S.parse(RAW)
    res = converter.RosettaConverter(spec, {}, FakeLaunch(exit=1)).convert([tmp_path], tmp_path / "dataset")
    assert not res.ok and "exit 1" in res.reason
    assert not converter.RosettaConverter(spec, {}, FakeLaunch()).convert([], tmp_path / "dataset").ok


def test_the_dataset_launcher_is_the_components():
    assert converter.LAUNCH.parts[-3:] == ("components", "dataset", "launch.py") and converter.LAUNCH.is_file()
    assert "def convert(manifests, out, turn" in converter.LAUNCH.read_text()


def test_the_pipeline_skips_the_step_with_one_message(tmp_path):
    spec, launch, logs = S.parse(RAW), FakeLaunch(), []
    plugins = PluginSet(converter=converter.RosettaConverter(spec, {}, launch))
    assert Pipeline(spec, plugins, tmp_path, log=logs.append).run(["convert"])
    assert logs.count(NO_CONVERT) == 1 and launch.calls == []


@pytest.mark.parametrize("n_eps, count", [(1, 5), (3, 2)])
def test_realised_is_the_datasets_own_count(tmp_path, n_eps, count):
    # the episodes the dataset holds (the job's result.json progress), not the number of episodes handed in
    eps = [tmp_path / f"episode_{i}_bag" for i in range(n_eps)]
    res = converter.RosettaConverter(S.parse(RAW), {}, FakeLaunch(episodes=count)).convert(eps, tmp_path / "dataset")
    assert res.ok, res.reason
    assert res.realised == {"episodes": count}


@pytest.mark.parametrize("result", [{}, {"progress": {}}, {"progress": {"stage": "done"}},
                                    {"progress": {"stage": "done", "episodes": True}}],
                         ids=["empty", "progress-empty", "no-episodes", "bool"])
def test_a_finished_job_without_a_count_fails(tmp_path, result):
    res = converter.RosettaConverter(S.parse(RAW), {}, FakeLaunch(result=result)).convert([tmp_path / "e"],
                                                                                          tmp_path / "dataset")
    assert not res.ok and "no episode count" in res.reason


def test_a_failed_job_fails_even_with_a_count(tmp_path):
    launch = FakeLaunch(exit=1, result={"progress": {"stage": "failed", "episodes": 3, "problems": ["x"]}})
    res = converter.RosettaConverter(S.parse(RAW), {}, launch).convert([tmp_path / "e"], tmp_path / "dataset")
    assert not res.ok and "exit 1" in res.reason
