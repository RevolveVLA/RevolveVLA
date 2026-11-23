"""core/pipeline.py + pool.py + state.py + manifest.py with the fake generator and the `fake` runner (revolve_common's FakeSession,
no sim): step order, rejections, resume after a kill, exhaustion, dry run, changed spec refused, split filter, workers, the
manifest.json derived from the scene logs."""
import json

import pytest

import fake_datagen as F
from datagen import cli
from datagen.core import episodes, pipeline, spec as S, state
from datagen.core.pool import PoolExhausted

RAW = {
    "task": "t", "robot": "r", "out": "unused", "generator": {"plugin": "fake_datagen"}, "runner": {"plugin": "fake"},
    "converter": {"plugin": "fake_datagen"},
    "randomisers": {"fake_datagen": {"x": [-0.5, 0.5], "y": [1, 2]}},
    "splits": {"train": {"target": 3, "pool": {"initial": [0, 3], "replacement": [4, 9]}},
               "heldout": {"target": 2, "pool": {"initial": [20, 24]}, "record": False}},
}


@pytest.fixture(autouse=True)
def calls():
    F.CALLS.clear()
    return F.CALLS


def make(tmp_path, steps=pipeline.STEPS, dry_run=False, log=lambda m: None, **edits):
    spec = S.parse(RAW | edits)
    return pipeline.Pipeline(spec, cli.build_plugins(spec, steps), tmp_path, dry_run, log=log)


def load(tmp_path, name="state.json"):
    return json.loads((tmp_path / name).read_text())


def drives(calls):
    return [seed for kind, seed in calls if kind == "drive"]


def test_all_steps_in_order_and_outputs(tmp_path, calls):
    logs = []
    assert make(tmp_path, log=logs.append).run()
    assert calls == [("drive", s) for s in (0, 1, 2)]               # train 3 scenes; heldout (record: false) runs no sim
    m = load(tmp_path)
    assert [e["config"]["seed"] for e in m["splits"]["train"]["used"]] == [0, 1, 2]
    assert m["splits"]["train"]["used"][1]["generate"]["paths"] == {"scene": "train/scene_1/scene.yaml"}
    assert [e["config"]["seed"] for e in m["splits"]["heldout"]["used"]] == [20, 21]
    assert [(c["name"], c["ok"]) for c in m["checks"]] == [("target", True), ("split_disjoint", True)]
    assert any(l.startswith("convert: skipped, no Rosetta contract") for l in logs)
    assert not (tmp_path / "dataset").exists()
    assert "| train | 3 | 3 | 0 |" in (tmp_path / "report.md").read_text()
    man = load(tmp_path, "manifest.json")
    assert man["counts"] == {"scenes": 3, "attempted": 3, "passed": 3}
    assert [e["datagen"]["expert"] for e in man["episodes"]] == ["fake_v1"] * 3


def test_rejections_are_recorded_and_replaced_from_the_pool(tmp_path, calls):
    validators = [{"plugin": "fake_datagen", "reject_seeds": [0]}]
    make(tmp_path, runner={"plugin": "fake", "fail_seeds": [1]}, validators=validators).run(["sample", "generate"])
    rec = load(tmp_path)["splits"]["train"]
    assert [e["config"]["seed"] for e in rec["used"]] == [2, 3, 4]
    assert rec["rejected"] == [{"seed": 0, "step": "validate", "reason": "too close to the wall"},
                               {"seed": 1, "step": "generate", "reason": "0/1 accepted in 3 tries (last: timeout_s)"}]
    assert 0 not in drives(calls)                              # a validator rejection never reaches the sim
    man = load(tmp_path, "manifest.json")                      # the failed tries stay in the manifest
    assert [(e["seed"], e["passed"]) for e in man["episodes"]] == [(1, False)] * 3 + [(2, True), (3, True), (4, True)]
    assert man["counts"] == {"scenes": 4, "attempted": 6, "passed": 3}


def test_episodes_per_scene_and_attempts(tmp_path, calls):
    gen = {"plugin": "fake_datagen", "attempts": 3, "episodes_per_scene": 2}
    runner = {"plugin": "fake", "pass_after": {1: 1}, "fail_seeds": [2]}
    make(tmp_path, generator=gen, runner=runner).run(["generate"], splits=["train"])
    man = load(tmp_path, "manifest.json")["episodes"]
    by_seed = {s: [(e["episode"], e["passed"], e["datagen"]["attempt"]) for e in man if e["seed"] == s] for s in range(5)}
    assert by_seed[0] == [(0, True, 0), (1, True, 1)]                       # two accepted, no third try
    assert by_seed[1] == [(0, False, 0), (1, True, 1), (2, True, 2)]        # the failed try stays, numbering continues
    assert by_seed[2] == [(0, False, 0), (1, False, 1), (2, False, 2)]      # out of tries
    assert [e["config"]["seed"] for e in load(tmp_path)["splits"]["train"]["used"]] == [0, 1, 3]


def test_an_infrastructure_relaunch_is_counted_and_is_not_a_try(tmp_path, calls):
    runner = {"plugin": "fake", "faults": {"1": {"wait_result": ["result_timeout"], "reset": ["reset_failed"]}}}
    make(tmp_path, runner=runner, generator={"plugin": "fake_datagen", "episodes_per_scene": 2}).run(["sample", "generate"])
    assert episodes.read(tmp_path / "train" / "scene_1")["faults"] == ["result_timeout", "reset_failed"]
    assert [t["episode"] for t in episodes.read(tmp_path / "train" / "scene_1")["tries"]] == [0, 1]
    assert [e["episode"] for e in load(tmp_path, "manifest.json")["episodes"] if e["seed"] == 1] == [0, 1]
    assert "2 infrastructure relaunches" in (tmp_path / "report.md").read_text()


def test_too_many_infrastructure_faults_reject_the_scene(tmp_path):
    runner = {"plugin": "fake", "infra_retries": 1, "faults": {"0": {"wait_ready": ["ready_timeout", "clock_fault"]}}}
    make(tmp_path, runner=runner).run(["sample", "generate"])
    rec = load(tmp_path)["splits"]["train"]
    assert rec["rejected"][0]["seed"] == 0 and rec["rejected"][0]["reason"].startswith("infrastructure:")
    assert [e["config"]["seed"] for e in rec["used"]] == [1, 2, 3]


def test_resume_after_a_kill_redoes_nothing_done(tmp_path, calls, monkeypatch):
    monkeypatch.setattr(F, "KILL_AFTER", 2)
    with pytest.raises(KeyboardInterrupt):
        make(tmp_path).run()
    assert [e["config"]["seed"] for e in load(tmp_path)["splits"]["train"]["used"]] == [0, 1]
    calls.clear()
    monkeypatch.setattr(F, "KILL_AFTER", None)
    assert make(tmp_path).run()
    assert drives(calls) == [2]
    assert [e["config"]["seed"] for e in load(tmp_path)["splits"]["train"]["used"]] == [0, 1, 2]
    assert [(e["seed"], e["episode"]) for e in load(tmp_path, "manifest.json")["episodes"]] == [(0, 0), (1, 0), (2, 0)]
    calls.clear()
    assert make(tmp_path).run()                                   # a finished run is a no-op
    assert calls == []


def test_a_scene_resumes_from_its_own_log(tmp_path, calls):
    """The decision was not saved (killed after the scene's episodes): the rerun launches no sim for a finished scene."""
    make(tmp_path).run(["sample", "generate"], splits=["train"])
    (tmp_path / "state.json").unlink()
    calls.clear()
    make(tmp_path).run(["sample", "generate"], splits=["train"])
    assert calls == []
    assert len(load(tmp_path, "manifest.json")["episodes"]) == 3


def test_exhausted_pool_saves_progress_and_raises(tmp_path):
    runner = {"plugin": "fake", "fail_seeds": [s for s in range(10) if s != 5]}
    with pytest.raises(PoolExhausted, match="1/3 accepted, 9 rejected"):
        make(tmp_path, runner=runner, generator={"plugin": "fake_datagen", "attempts": 1}).run()
    assert len(load(tmp_path)["splits"]["train"]["rejected"]) == 9
    assert (tmp_path / "report.md").exists() and load(tmp_path, "manifest.json")["counts"]["attempted"] == 10


def test_changed_spec_is_refused(tmp_path):
    make(tmp_path).run(["sample"])
    with pytest.raises(state.StateError, match="another spec"):
        make(tmp_path, seed=1)


def test_dry_run_calls_no_plugin_and_writes_nothing(tmp_path, calls):
    logs = []
    spec = S.parse(RAW)
    p = pipeline.Pipeline(spec, cli.build_plugins(spec, pipeline.STEPS), tmp_path / "o", dry_run=True, log=logs.append)
    assert p.run()                                                # no verdict on checks: nothing ran
    assert calls == [] and not (tmp_path / "o").exists()
    assert "[dry-run] generate train seed 0" in logs
    assert any(l.startswith("[dry-run] check target") for l in logs)


def test_split_filter(tmp_path, calls):
    make(tmp_path).run(["sample", "generate"], splits=["heldout"])
    assert set(load(tmp_path)["splits"]) == {"heldout"} and calls == []       # a record: false split runs no sim
    with pytest.raises(pipeline.PipelineError, match="unknown split"):
        make(tmp_path).run(splits=["nope"])


def test_a_missing_converter_skips_the_step_and_a_missing_runner_is_the_sim(tmp_path, calls):
    logs = []
    raw = {k: v for k, v in RAW.items() if k not in ("converter", "runner")}
    assert S.parse(raw).runner is None
    assert cli.build_plugins(S.parse(raw), ["generate"]).runner.__class__.__name__ == "SimRunner"
    spec = S.parse({k: v for k, v in RAW.items() if k != "converter"})
    assert pipeline.Pipeline(spec, cli.build_plugins(spec, pipeline.STEPS), tmp_path, log=logs.append).run()
    assert "convert: no converter in the spec, skipped" in logs and ("convert", 3) not in calls


def test_sampling_is_deterministic_and_within_ranges(tmp_path):
    make(tmp_path / "a").run(["sample"])
    make(tmp_path / "b").run(["sample"])
    a, b = (json.loads((tmp_path / n / "state.json").read_text())["splits"]["train"]["sampled"] for n in "ab")
    assert a == b and len(a) == 10
    assert all(-0.5 <= c["values"]["fake_datagen.x"] <= 0.5 and 1 <= c["values"]["fake_datagen.y"] <= 2 for c in a)
    assert len({c["values"]["fake_datagen.x"] for c in a}) == 10


def test_limits_come_from_the_randomisers(tmp_path):
    with pytest.raises(S.SpecError, match="outside the hard limits"):
        make(tmp_path, randomisers={"fake_datagen": {"x": [-2, 0.5], "y": [1, 2]}})


def test_min_distance_to_a_source_split_and_redraws_are_recorded(tmp_path):
    splits = {"train": RAW["splits"]["train"],
              "heldout": {"target": 2, "pool": {"initial": [20, 29]}, "record": False,
                          "min_distance": {"from": "train", "value": 0.25}}}
    make(tmp_path, splits=splits).run(["sample"], splits=["heldout"])      # samples train first
    m = load(tmp_path)["splits"]
    train = [c["values"] for c in m["train"]["sampled"]]
    dims = S.parse(RAW).dims()
    from datagen.core.samplers import distance
    assert all(distance(c["values"], t, dims) >= 0.25 for c in m["heldout"]["sampled"] for t in train)
    assert m["heldout"]["redrawn"] and {r["reason"] for r in m["heldout"]["redrawn"]} == {"min_distance"}
    assert "redrawn" not in m["train"]


def test_stratified_replacement_stays_in_the_cell(tmp_path):
    strat = {"strategy": "stratified", "bins": {"fake_datagen.x": 2}}
    splits = {"train": RAW["splits"]["train"] | {"sampler": strat, "pool": {"initial": [0, 5], "replacement": [6, 11]}}}
    make(tmp_path, splits=splits, runner={"plugin": "fake", "fail_seeds": [0]}).run(["generate"])
    rec = load(tmp_path)["splits"]["train"]
    cell = {c["seed"]: c["cell"] for c in rec["sampled"]}
    new = [e["config"]["seed"] for e in rec["used"] if e["config"]["seed"] >= 3]
    assert len(new) == 1 and cell[new[0]] == cell[0] and rec["rejected"][0]["seed"] == 0


# ------------------------------------------------------------------ runtime: workers, log lines
def text(path):
    return (path / "state.json").read_text().replace(str(path), "OUT")


def manifest_of(path):
    """manifest.json without the wall times and the creation time (the only things that differ between runs)."""
    m = json.loads((path / "manifest.json").read_text())
    m.pop("created")
    for e in m["episodes"]:
        e["datagen"].pop("wall_s")
    return m


def slow_spec(workers, **edits):
    """12 seeds, train target 5 over a pool with rejections, heldout with a realised min_distance; slower seeds first."""
    delays = {s: 0.06 - 0.01 * (s % 6) for s in list(range(0, 12)) + list(range(20, 26))}
    splits = {"train": {"target": 5, "pool": {"initial": [0, 5], "replacement": [6, 11]}},
              "heldout": {"target": 2, "pool": {"initial": [20, 25]}, "record": False,
                          "min_distance": {"from": "train", "value": 0.1, "realised": True}}}
    return RAW | dict(generator={"plugin": "fake_datagen", "delays": delays},
                      runner={"plugin": "fake", "fail_seeds": [1, 3, 7]}, splits=splits, run={"workers": workers}) | edits


REALISED = {0: -0.4, 2: -0.2, 4: 0.0, 5: 0.4, 6: 0.45, 20: -0.38, 21: 0.01, 22: 0.2, 23: -0.1}


def pipe(spec, path, log=lambda m: None):
    plugins = cli.build_plugins(spec, pipeline.STEPS)
    plugins.instancer = F.FakeInstancer({s: {"fake_datagen.x": x} for s, x in REALISED.items()})
    return pipeline.Pipeline(spec, plugins, path, log=log)


def test_the_state_and_manifest_are_the_same_for_one_and_three_workers(tmp_path, calls):
    logs = {}
    for n in (1, 3):
        F.ACTIVE.update(now=0, peak=0)
        logs[n] = []
        assert pipe(S.parse(slow_spec(n)), tmp_path / str(n), logs[n].append).run()
        assert F.ACTIVE["peak"] == (1 if n == 1 else 3)           # n at a time, never more, and really in parallel
    assert text(tmp_path / "1") == text(tmp_path / "3") and manifest_of(tmp_path / "1") == manifest_of(tmp_path / "3")
    m = load(tmp_path / "3")["splits"]
    assert [e["config"]["seed"] for e in m["train"]["used"]] == [0, 2, 4, 5, 6] and len(m["train"]["rejected"]) == 2
    assert [(e["seed"], e["step"]) for e in m["heldout"]["rejected"]] == [(20, "realised"), (21, "realised")]
    assert [e["config"]["seed"] for e in m["heldout"]["used"]] == [22, 23]                       # decided in order
    assert (tmp_path / "1" / "report.md").read_text() == (tmp_path / "3" / "report.md").read_text()
    assert logs[1] == logs[3]


def test_a_killed_parallel_run_resumes_like_a_sequential_one(tmp_path, calls, monkeypatch):
    monkeypatch.setattr(F, "KILL_AFTER", 3)
    spec = S.parse(slow_spec(3))
    with pytest.raises(KeyboardInterrupt):
        pipe(spec, tmp_path / "a").run()
    monkeypatch.setattr(F, "KILL_AFTER", None)
    assert pipe(spec, tmp_path / "a").run()
    assert pipe(S.parse(slow_spec(1)), tmp_path / "b").run()
    assert text(tmp_path / "a") == text(tmp_path / "b") and manifest_of(tmp_path / "a") == manifest_of(tmp_path / "b")


def test_an_error_in_a_worker_calls_abort_and_keeps_what_was_decided(tmp_path, calls, monkeypatch):
    monkeypatch.setattr(F, "KILL_AFTER", 1)
    aborted = []
    spec = S.parse(RAW | dict(generator={"plugin": "fake_datagen", "delays": {0: 0.0, 1: 0.3, 2: 0.3}}, run={"workers": 3}))
    p = pipeline.Pipeline(spec, cli.build_plugins(spec, ["generate"]), tmp_path, log=lambda m: None,
                          abort=lambda: aborted.append(1))
    with pytest.raises(KeyboardInterrupt):
        p.run(["generate"])
    assert aborted == [1]
    assert len(load(tmp_path)["splits"]["train"]["used"]) <= 1                  # only what was decided before the error


def test_gui_runs_have_one_worker_and_say_so(tmp_path):
    logs = []
    spec = S.parse(RAW | {"run": {"workers": 3, "headless": False}})
    p = pipeline.Pipeline(spec, cli.build_plugins(spec, ["sample"]), tmp_path, log=logs.append)
    assert p.workers == 1 and logs == ["run.headless false: one GUI window, so workers 3 -> 1"]
    assert pipeline.Pipeline(S.parse(RAW | {"run": {"workers": 3}}), cli.build_plugins(spec, ["sample"]), tmp_path / "x",
                             log=lambda m: None).workers == 3


def test_every_log_line_is_said_once(tmp_path):
    logs = []
    spec = S.parse(RAW | {"splits": RAW["splits"] | {"heldout": RAW["splits"]["heldout"] | {
        "min_distance": {"from": "train", "value": 0.01}}}})
    pipeline.Pipeline(spec, cli.build_plugins(spec, pipeline.STEPS), tmp_path, log=logs.append).run()
    assert [l for l in logs if l.startswith("sample")] == ["sample train: 10 configs", "sample heldout: 5 configs"]
    assert len(logs) == len(set(logs))


def test_a_dry_run_says_what_it_would_do_and_claims_no_result(tmp_path):
    logs = []
    spec = S.parse(RAW)
    pipeline.Pipeline(spec, cli.build_plugins(spec, pipeline.STEPS), tmp_path / "o", dry_run=True, log=logs.append).run()
    assert not [l for l in logs if "accepted" in l]
    assert len([l for l in logs if l.startswith("sample")]) == 2 and len(logs) == len(set(logs))
