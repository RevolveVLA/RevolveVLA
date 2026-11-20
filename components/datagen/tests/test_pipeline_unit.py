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
    p.run()
    assert calls == [] and not (tmp_path / "o").exists()
    assert "[dry-run] generate train seed 0" in logs and "[dry-run] convert 3 episodes" in logs


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
