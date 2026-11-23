"""core/episodes.py (the scene log + the episode loop), core/manifest.py and job.py on revolve_common's FakeSession: the loop of
DATAGEN-v2.md 4.4, resume from the log, the manifest entries, a DATAGEN 8.2 job turned into a spec."""
import json

import pytest

import fake_datagen as F
from datagen import job
from datagen.core import episodes, manifest
from datagen.core.spec import DatagenSpec, EpisodeSpec, SpecError
from revolve_common.sim_fake import FakeSession
from revolve_common.sim_session import SimFault

GEN = F.FakeGenerator(None, {})


def session(tmp_path, seed=0, passes=lambda n: True, faults=None):
    d = tmp_path / f"scene_{seed}"
    return FakeSession(d / "scene.yaml", "r", "t", d, "n", 100, passes=passes, faults=faults), d / "scene.yaml"


def loop(s, instance, ep, seed=0):
    return episodes.run_episodes(s.connect(), GEN, "t", "r", instance, seed, ep)


def test_the_loop_runs_tries_until_enough_are_accepted_and_logs_each(tmp_path):
    s, instance = session(tmp_path, passes=lambda n: n == 1)
    s.start()
    log = loop(s, instance, EpisodeSpec(attempts=3, per_scene=1))
    assert [(t["attempt"], t["episode"], t["passed"], t["reason"]) for t in log["tries"]] == [
        (0, 0, False, "timeout_s"), (1, 1, True, None)]
    assert log["tries"][0]["expert"] == "fake_v1" and log["seed"] == 0 and s.client_state["resets"] == 1   # reset between
    assert episodes.read(instance.parent) == log
    assert episodes.verdict(log, EpisodeSpec(3, 1)).ok and not episodes.verdict(log, EpisodeSpec(3, 2)).ok


def test_the_loop_gives_up_after_attempts_and_says_why(tmp_path):
    s, instance = session(tmp_path, passes=lambda n: False)
    s.start()
    log = loop(s, instance, EpisodeSpec(2, 1))
    assert len(log["tries"]) == 2
    res = episodes.verdict(log, EpisodeSpec(2, 1))
    assert not res.ok and res.reason == "0/1 accepted in 2 tries (last: timeout_s)"


def test_a_finished_scene_needs_no_ready_and_a_resumed_one_continues_numbering(tmp_path):
    s, instance = session(tmp_path, faults={"wait_ready": ["ready_timeout"]})
    s.start()
    ep = EpisodeSpec(3, 2)
    with pytest.raises(SimFault):                      # the sim never gets ready: no try was made
        loop(s, instance, ep)
    s.start()
    log = loop(s, instance, EpisodeSpec(3, 1))
    assert [t["episode"] for t in log["tries"]] == [0]
    s2 = FakeSession(instance, "r", "t", instance.parent, "n", 100, faults={"wait_ready": ["ready_timeout"]})
    s2.start()
    assert loop(s2, instance, EpisodeSpec(3, 1)) == log        # finished: the failing wait_ready is never reached
    s3 = FakeSession(instance, "r", "t", instance.parent, "n", 100)
    s3.start()
    assert [t["episode"] for t in loop(s3, instance, ep)["tries"]] == [0, 1]      # one more wanted: numbering continues


def test_the_generator_is_prepared_once_per_launch_after_ready_and_before_the_first_episode(tmp_path):
    calls = []

    class Gen(F.FakeGenerator):
        def prepare(self, client, instance, task, robot):
            calls.append(("prepare", client.s.client_state["next"], task, robot))      # next episode number: none started

        def drive(self, client, instance, task, robot):
            calls.append(("drive",))
            return super().drive(client, instance, task, robot)

    s, instance = session(tmp_path, passes=lambda n: n == 1)
    s.start()
    episodes.run_episodes(s.connect(), Gen(None, {}), "t", "r", instance, 0, EpisodeSpec(3, 1))
    assert calls == [("prepare", 0, "t", "r"), ("drive",), ("drive",)]
    assert GEN.prepare(s.connect(), instance, "t", "r") is None                        # the default does nothing


def test_partial_bags_without_a_result_file_are_removed_before_the_loop(tmp_path):
    s, instance = session(tmp_path)
    d = instance.parent
    (d / "episode_0_bag").mkdir(parents=True)
    (d / "episode_0.json").write_text("{}")
    (d / "episode_1_bag").mkdir()                                  # a killed sim's: no episode_1.json
    (d / "keep_bag").mkdir()
    s.start()
    episodes.clean_partial(d)
    assert (d / "episode_0_bag").is_dir() and not (d / "episode_1_bag").exists() and (d / "keep_bag").is_dir()


def test_faults_are_appended_to_the_log(tmp_path):
    episodes.add_faults(tmp_path, [])
    assert not (tmp_path / "datagen.json").exists()
    episodes.add_faults(tmp_path, ["clock_fault"])
    episodes.add_faults(tmp_path, ["sim_exit"])
    assert episodes.read(tmp_path)["faults"] == ["clock_fault", "sim_exit"]


# ------------------------------------------------------------------ manifest.json entries
def test_manifest_entries_come_from_the_log_and_the_result_files(tmp_path):
    s, instance = session(tmp_path, seed=7, passes=lambda n: n == 1)
    s.start()
    loop(s, instance, EpisodeSpec(3, 1), seed=7)
    (instance.parent / "episode_1_bag").rmdir()                    # a run without a bag: bag null
    found = manifest.entries(tmp_path, instance.parent, "t", "r")
    assert [(e["episode"], e["passed"], e["bag"]) for e in found] == [(0, False, "scene_7/episode_0_bag"), (1, True, None)]
    assert found[0]["result"] == "scene_7/episode_0.json" and found[0]["scene"] == "scene_7/scene.yaml"
    assert found[0]["datagen"] == {"expert": "fake_v1", "attempt": 0, "wall_s": found[0]["datagen"]["wall_s"]}
    (instance.parent / "episode_0.json").unlink()                  # a try whose result file is gone is left out
    assert [e["episode"] for e in manifest.entries(tmp_path, instance.parent, "t", "r")] == [1]
    m = manifest.build(tmp_path, "t", "r", [instance.parent, tmp_path / "no_scene"], "2026-01-01T00:00:00Z")
    assert m["counts"] == {"scenes": 1, "attempted": 1, "passed": 1} and m["created"] == "2026-01-01T00:00:00Z"
    manifest.write(tmp_path / "manifest.json", m)
    assert manifest.created(tmp_path / "manifest.json") == "2026-01-01T00:00:00Z" and manifest.created(tmp_path / "x") is None


# ------------------------------------------------------------------ job.py (DATAGEN-v2.md 8.2)
def job_yaml(tmp_path, **edits):
    for i in (0, 4):
        (tmp_path / f"s{i}").mkdir(exist_ok=True)
        (tmp_path / f"s{i}" / "scene.yaml").write_text("name: x\n")
    return {"task": "t", "robot": "r", "out": "o", "episodes_per_scene": 2,
            "episodes": [{"seed": 0, "scene": "s0/scene.yaml"}, {"seed": 4, "scene": "s4/scene.yaml"}]} | edits


def test_a_job_becomes_a_spec_with_one_split_over_its_scenes(tmp_path):
    spec, scenes = job.to_spec(job_yaml(tmp_path), tmp_path)
    assert isinstance(spec, DatagenSpec) and spec.generator.plugin == "moveit" and spec.episodes == EpisodeSpec(3, 2, "images")
    assert spec.splits[0].name == "job" and spec.splits[0].target == 2 and spec.splits[0].pool.initial == (0, 4)
    assert scenes == {0: tmp_path / "s0" / "scene.yaml", 4: tmp_path / "s4" / "scene.yaml"}
    spec, _ = job.to_spec(job_yaml(tmp_path, generator={"plugin": "g", "attempts": 5}, runner={"plugin": "fake"},
                                   run={"workers": 2}), tmp_path)
    assert (spec.generator.plugin, spec.episodes.attempts, spec.runner.plugin, spec.run.workers) == ("g", 5, "fake", 2)


@pytest.mark.parametrize("edit, msg", [
    ({"episodes": []}, "episodes: expected a non-empty list"),
    ({"episodes": [{"seed": 0, "scene": "s0/scene.yaml"}] * 2}, "seed: 0 twice"),
    ({"episodes": [{"seed": 0, "scene": "nope.yaml"}]}, "no such file"),
    ({"episodes": [{"seed": 0}]}, "missing key 'scene'"),
    ({"x": 1}, "unknown key"),
])
def test_a_bad_job_is_one_line(tmp_path, edit, msg):
    with pytest.raises(Exception, match=msg):
        job.to_spec(job_yaml(tmp_path, **edit), tmp_path)


def test_a_spec_yaml_is_not_a_job(tmp_path):
    assert not job.is_job({"task": "t", "episodes": [], "splits": {}}) and job.is_job(job_yaml(tmp_path))
    with pytest.raises(SpecError, match="cannot read"):
        job.load(tmp_path / "missing.yaml", tmp_path)
