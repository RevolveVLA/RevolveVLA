"""core/pipeline.py + pool.py + manifest.py with the fake plugins: step order, rejections, resume after a kill,
exhaustion, dry run, changed spec refused, split filter, the target check."""
import json

import pytest

import fake_datagen as F
from datagen import cli
from datagen.core import manifest, pipeline, spec as S
from datagen.core.pool import PoolExhausted

RAW = {
    "task": "t", "robot": "r", "out": "unused", "generator": {"plugin": "fake_datagen"},
    "recorder": {"plugin": "fake_datagen"}, "converter": {"plugin": "fake_datagen"},
    "randomisers": {"fake_datagen": {"x": [-0.5, 0.5], "y": [1, 2]}},
    "splits": {"train": {"target": 3, "pool": {"initial": [0, 3], "replacement": [4, 9]}},
               "heldout": {"target": 2, "pool": {"initial": [20, 24]}, "record": False}},
}


@pytest.fixture(autouse=True)
def calls():
    F.CALLS.clear()
    return F.CALLS


def make(tmp_path, steps=pipeline.STEPS, dry_run=False, **edits):
    spec = S.parse(RAW | edits)
    return pipeline.Pipeline(spec, cli.build_plugins(spec, steps), tmp_path, dry_run, log=lambda m: None)


def load(tmp_path):
    return json.loads((tmp_path / "manifest.json").read_text())


def test_all_steps_in_order_and_outputs(tmp_path, calls):
    assert make(tmp_path).run()
    kinds = [c[0] for c in calls]
    assert kinds == ["generate"] * 5 + ["record"] * 3 + ["convert"]      # train 3 + heldout 2; train recorded only
    m = load(tmp_path)
    assert [e["config"]["seed"] for e in m["splits"]["train"]["used"]] == [0, 1, 2]
    assert m["splits"]["heldout"]["used"][0]["generate"]["realised"]["x"] is not None
    assert all("record" not in e for e in m["splits"]["heldout"]["used"])
    assert m["convert"]["result"]["ok"] and len(m["convert"]["episodes"]) == 3
    assert json.loads((tmp_path / "dataset" / "dataset.json").read_text()) == ["seed_0", "seed_1", "seed_2"]
    assert (tmp_path / "train" / "episodes" / "seed_1" / "bag.txt").exists()
    assert [(c["name"], c["ok"]) for c in m["checks"]] == [("target", True), ("split_disjoint", True)]
    assert "| train | 3 | 3 | 0 | 3 |" in (tmp_path / "report.md").read_text()


def test_rejections_are_recorded_and_replaced_from_the_pool(tmp_path, calls):
    gen = {"plugin": "fake_datagen", "fail_seeds": [1]}
    validators = [{"plugin": "fake_datagen", "reject_seeds": [0]}]
    make(tmp_path, generator=gen, validators=validators).run(["sample", "generate"])
    rec = load(tmp_path)["splits"]["train"]
    assert [e["config"]["seed"] for e in rec["used"]] == [2, 3, 4]
    assert rec["rejected"] == [{"seed": 0, "step": "validate", "reason": "too close to the wall"},
                               {"seed": 1, "step": "generate", "reason": "planner found no path"}]
    assert ("generate", 0) not in calls                      # a validator rejection never reaches the generator


def test_resume_after_a_kill_redoes_nothing_done(tmp_path, calls, monkeypatch):
    monkeypatch.setattr(F, "KILL_AFTER", 2)
    with pytest.raises(KeyboardInterrupt):
        make(tmp_path).run()
    assert [e["config"]["seed"] for e in load(tmp_path)["splits"]["train"]["used"]] == [0, 1]
    calls.clear()
    monkeypatch.setattr(F, "KILL_AFTER", None)
    assert make(tmp_path).run()
    assert [c for c in calls if c[0] == "generate"] == [("generate", s) for s in (2, 20, 21)]
    assert [e["config"]["seed"] for e in load(tmp_path)["splits"]["train"]["used"]] == [0, 1, 2]
    calls.clear()
    assert make(tmp_path).run()                                   # a finished run is a no-op
    assert calls == []


def test_record_retries_failures_only(tmp_path, calls):
    rec = {"plugin": "fake_datagen", "fail_seeds": [1]}
    assert not make(tmp_path, recorder=rec).run()                      # the target check fails: seed 1 not recorded
    assert [e["record"]["ok"] for e in load(tmp_path)["splits"]["train"]["used"]] == [True, False, True]
    calls.clear()
    make(tmp_path, recorder=rec).run(["record"])
    assert calls == [("record", 1)]


def test_exhausted_pool_saves_progress_and_raises(tmp_path):
    with pytest.raises(PoolExhausted, match="1/3 accepted, 9 rejected"):
        make(tmp_path, generator={"plugin": "fake_datagen", "fail_seeds": [s for s in range(10) if s != 5]}).run()
    assert len(load(tmp_path)["splits"]["train"]["rejected"]) == 9
    assert (tmp_path / "report.md").exists()                      # the report of the aborted run


def test_changed_spec_is_refused(tmp_path):
    make(tmp_path).run(["sample"])
    with pytest.raises(manifest.ManifestError, match="another spec"):
        make(tmp_path, seed=1)


def test_dry_run_calls_no_plugin_and_writes_nothing(tmp_path, calls):
    logs = []
    spec = S.parse(RAW)
    p = pipeline.Pipeline(spec, cli.build_plugins(spec, pipeline.STEPS), tmp_path / "o", dry_run=True, log=logs.append)
    assert p.run()                                                # no verdict on checks: nothing ran
    assert calls == [] and not (tmp_path / "o").exists()
    assert "[dry-run] generate train seed 0" in logs and "[dry-run] convert 3 episodes" in logs
    assert any(l.startswith("[dry-run] check target") for l in logs)


def test_split_filter(tmp_path, calls):
    make(tmp_path).run(["sample", "generate"], splits=["heldout"])
    assert set(load(tmp_path)["splits"]) == {"heldout"} and len(calls) == 2
    with pytest.raises(pipeline.PipelineError, match="unknown split"):
        make(tmp_path).run(splits=["nope"])


def test_missing_optional_plugins_skip_their_steps(tmp_path, calls):
    raw = {k: v for k, v in RAW.items() if k not in ("recorder", "converter")}
    spec = S.parse(raw)
    ok = pipeline.Pipeline(spec, cli.build_plugins(spec, pipeline.STEPS), tmp_path, log=lambda m: None).run()
    assert not ok                                                  # target check: train is not recorded
    assert {c[0] for c in calls} == {"generate"}


def test_sampling_is_deterministic_and_within_ranges(tmp_path):
    make(tmp_path / "a").run(["sample"])
    make(tmp_path / "b").run(["sample"])
    a, b = (json.loads((tmp_path / n / "manifest.json").read_text())["splits"]["train"]["sampled"] for n in "ab")
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
    make(tmp_path, splits=splits, generator={"plugin": "fake_datagen", "fail_seeds": [0]}).run(["generate"])
    rec = load(tmp_path)["splits"]["train"]
    cell = {c["seed"]: c["cell"] for c in rec["sampled"]}
    new = [e["config"]["seed"] for e in rec["used"] if e["config"]["seed"] >= 3]
    assert len(new) == 1 and cell[new[0]] == cell[0] and rec["rejected"][0]["seed"] == 0


# ------------------------------------------------------------------ runtime: workers, log lines
def manifest_text(path):
    return (path / "manifest.json").read_text().replace(str(path), "OUT")


def slow_spec(workers, **edits):
    """12 seeds, train target 5 over a pool with rejections, heldout with a realised min_distance; slower seeds first."""
    delays = {s: 0.06 - 0.01 * (s % 6) for s in list(range(0, 12)) + list(range(20, 26))}
    gen = {"plugin": "fake_datagen", "fail_seeds": [1, 3, 7], "delays": delays,
           "realised": {0: -0.4, 2: -0.2, 4: 0.0, 5: 0.4, 6: 0.45, 20: -0.38, 21: 0.01, 22: 0.2, 23: -0.1}}
    splits = {"train": {"target": 5, "pool": {"initial": [0, 5], "replacement": [6, 11]}},
              "heldout": {"target": 2, "pool": {"initial": [20, 25]}, "record": False,
                          "min_distance": {"from": "train", "value": 0.1, "realised": True}}}
    return RAW | dict(generator=gen, splits=splits, run={"workers": workers}) | edits


def test_the_manifest_is_the_same_for_one_and_three_workers(tmp_path, calls):
    logs = {}
    for n in (1, 3):
        F.ACTIVE.update(now=0, peak=0)
        spec = S.parse(slow_spec(n, recorder={"plugin": "fake_datagen", "delays": {0: 0.1, 2: 0.05, 4: 0.0}}))
        logs[n] = []
        p = pipeline.Pipeline(spec, cli.build_plugins(spec, pipeline.STEPS), tmp_path / str(n), log=logs[n].append)
        assert p.run()
        assert F.ACTIVE["peak"] == (1 if n == 1 else 3)           # n at a time, never more, and really in parallel
    assert manifest_text(tmp_path / "1") == manifest_text(tmp_path / "3")
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
        pipeline.Pipeline(spec, cli.build_plugins(spec, pipeline.STEPS), tmp_path / "a", log=lambda m: None).run()
    monkeypatch.setattr(F, "KILL_AFTER", None)
    assert pipeline.Pipeline(spec, cli.build_plugins(spec, pipeline.STEPS), tmp_path / "a", log=lambda m: None).run()
    spec1 = S.parse(slow_spec(1))
    assert pipeline.Pipeline(spec1, cli.build_plugins(spec1, pipeline.STEPS), tmp_path / "b", log=lambda m: None).run()
    assert manifest_text(tmp_path / "a") == manifest_text(tmp_path / "b")


def test_an_error_in_a_worker_discards_the_unfinished_episodes_and_calls_abort(tmp_path, calls, monkeypatch):
    monkeypatch.setattr(F, "KILL_AFTER", 1)
    aborted = []
    spec = S.parse(RAW | dict(generator={"plugin": "fake_datagen", "delays": {0: 0.0, 1: 0.3, 2: 0.3}},
                              run={"workers": 3}))
    p = pipeline.Pipeline(spec, cli.build_plugins(spec, ["generate"]), tmp_path, log=lambda m: None,
                          abort=lambda: aborted.append(1))
    with pytest.raises(KeyboardInterrupt):
        p.run(["generate"])
    assert aborted == [1]
    assert not (tmp_path / "train" / "episodes" / "seed_1").exists()      # started, not finished: not resumable from
    assert not (tmp_path / "train" / "episodes" / "seed_2").exists()


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
    assert not [l for l in logs if "accepted" in l or "recorded" in l or "convert: ok" in l]
    assert len([l for l in logs if l.startswith("sample")]) == 2 and len(logs) == len(set(logs))
