"""core/checks.py + stats.py + separation.py + the report on synthetic manifests (no plugins, no docker): each check's
PASS and FAIL cases (incl. the realised 0.041 < 0.1 start), the default selection, the report snapshot, the exit code
of `check`, and the per-seed realised rejection of the pool walk."""
import json

import pytest
import yaml

from datagen import cli
from datagen.core import checks, pipeline, separation, spec as S, state
from datagen.core.interfaces import Result
from datagen.core.spec import Choices

DIM = "fake_datagen.x"           # range [-0.5, 0.5]: distances are plain differences
SPLITS = {"train": {"target": 4, "sampler": {"strategy": "stratified", "bins": {DIM: 2}}, "pool": {"initial": [0, 9]}},
          "heldout": {"target": 2, "pool": {"initial": [20, 29]}, "record": False,
                      "min_distance": {"from": "train", "value": 0.1, "realised": True}}}
RAW = {"task": "t", "robot": "r", "out": "unused", "generator": {"plugin": "fake_datagen", "checks": 3}, "runner": {"plugin": "fake"},
       "randomisers": {"fake_datagen": {"x": [-0.5, 0.5]}}, "splits": SPLITS}


def spec(**edits):
    return S.parse(RAW | edits)


def used(split, seed, x, realised=None, cell=()):
    return {"config": {"split": split, "seed": seed, "cell": list(cell), "values": {DIM: x}},
            "generate": {"ok": True, "reason": "", "paths": {}, "realised": {} if realised is None else {DIM: realised}}}


def sampled(split, seed, x, cell=()):
    return {"split": split, "seed": seed, "cell": list(cell), "values": {DIM: x}}


def split(*used_, sampled_=(), rejected=()):
    return {"sampled": list(sampled_), "used": list(used_), "rejected": list(rejected)}


def good():
    """train: 2 per cell (x < 0 cell 0, x >= 0 cell 1), realised = sampled; heldout 0.2 / 0.3 away from train."""
    train = [used("train", i, x, x, (int(x >= 0),)) for i, x in enumerate([-0.4, -0.2, 0.2, 0.4])]
    pool = [sampled("train", u["config"]["seed"], u["config"]["values"][DIM], tuple(u["config"]["cell"])) for u in train]
    held = [used("heldout", 20, 0.0, 0.0), used("heldout", 21, -0.1, -0.1)]
    return {"splits": {"train": split(*train, sampled_=pool), "heldout": split(*held)}}


def run(name, m, sp=None, **params):
    return checks.BUILTIN[name](sp or spec(), params).run(m)


def test_a_good_run_passes_every_check():
    m = good()
    for name in checks.BUILTIN:
        res = run(name, m)
        assert res.ok, (name, res.reason)


def test_target_counts_the_accepted_scenes():
    m = good()
    assert run("target", m).ok
    del m["splits"]["train"]["used"][1]
    assert run("target", m).reason == "train 3/4"


def test_split_disjoint_names_the_shared_seeds():
    m = good()
    m["splits"]["heldout"]["rejected"].append({"seed": 2, "step": "generate", "reason": "x"})
    res = run("split_disjoint", m)
    assert not res.ok and "train/heldout [2]" in res.reason
    m = good()
    m["splits"]["heldout"]["sampled"] = [sampled("heldout", 3, 0.0)]
    assert not run("split_disjoint", m).ok


def test_sampled_min_distance():
    m = good()
    m["splits"]["heldout"]["used"][0]["config"]["values"][DIM] = 0.45               # 0.05 from train seed 3 (0.4)
    res = run("min_distance", m)
    assert not res.ok and "heldout seed 20 - train seed 3: 0.050 < 0.1" in res.reason


def test_realised_min_distance_known_failing_case():
    m = good()
    m["splits"]["heldout"]["used"][0]["generate"]["realised"][DIM] = 0.241           # train realised 0.2 -> 0.041
    res = run("realised_min_distance", m)
    assert not res.ok and res.reason == "heldout seed 20 - train seed 2: 0.041 < 0.1"
    assert run("min_distance", m).ok                                                # sampled values are fine
    m["splits"]["heldout"]["used"][0]["generate"]["realised"][DIM] = 0.31            # 0.09 from 0.4 and 0.11 from 0.2
    assert "0.090 < 0.1" in run("realised_min_distance", m).reason
    m["splits"]["heldout"]["used"][0]["generate"]["realised"][DIM] = 0.3             # exactly 0.1 from 0.2 and 0.4: fine
    assert run("realised_min_distance", m).ok


def test_realised_min_distance_needs_the_realised_starts():
    m = good()
    m["splits"]["heldout"]["used"][1]["generate"]["realised"] = {}
    res = run("realised_min_distance", m)
    assert not res.ok and res.reason == "heldout: no realised start for seeds [21]"


def test_min_distance_inside_one_split():
    sp = spec(splits={"train": SPLITS["train"] | {"min_distance": {"from": "train", "value": 0.3}}})
    m = {"splits": {"train": good()["splits"]["train"]}}
    res = run("min_distance", m, sp)                                  # 0.4 - 0.2 = 0.2 < 0.3 inside train
    assert not res.ok and "train seed 0 - train seed 1" in res.reason


def test_coverage_cells_sampled_and_realised():
    m = good()
    assert run("coverage", m).ok
    m["splits"]["train"]["used"] = m["splits"]["train"]["used"][2:]                  # cell 0 unfilled, spares left
    m["splits"]["train"]["sampled"] += [sampled("train", s, -0.3, (0,)) for s in (8, 9)]
    res = run("coverage", m)
    assert not res.ok and res.reason == "train cell 0 empty"
    m = good()
    for u in m["splits"]["train"]["used"][:2]:
        u["generate"]["realised"][DIM] = 0.1                                         # jitter moved both into cell 1
    assert run("coverage", m).reason == "train cell 0 empty when realised"


def test_coverage_names_exhausted_cells():
    m = good()
    train = m["splits"]["train"]
    train["used"] = train["used"][2:]                                                # cell 0: both picks lost
    train["rejected"] = [{"seed": s, "step": "generate", "reason": "no path"} for s in (0, 1)]
    res = run("coverage", m)
    assert not res.ok and "train cell 0 exhausted" in res.reason
    train["sampled"].append(sampled("train", 9, -0.3, (0,)))                         # a spare is left: not exhausted
    assert stats_rows(m)["0"]["exhausted"] is False


def stats_rows(m):
    from datagen.core import stats
    sp = spec()
    return {r["cell"]: r for r in stats.cells(sp, sp.split("train"), m["splits"]["train"])["rows"]}


def test_repeat_stats_rate():
    m = good()
    flaky = [{"seed": 50 + i, "step": "generate", "reason": "replay rejected (repeat check 2/3 failed)"} for i in range(5)]
    m["splits"]["train"]["rejected"] = flaky
    res = run("repeat_stats", m)                                                      # 5 of 9 tried
    assert not res.ok and res.reason == "train 5/9 seeds failed a repeat check (> 0.5)"
    assert run("repeat_stats", m, max_rate=0.6).ok
    m["splits"]["train"]["rejected"] = [{"seed": 50, "step": "validate", "reason": "repeat check"}]
    assert run("repeat_stats", m).ok                                                  # validator rejections are not tries


def test_default_checks_are_the_applicable_ones():
    names = lambda sp: [r.plugin for r in checks.default_checks(sp)]
    assert names(spec()) == ["target", "split_disjoint", "min_distance", "realised_min_distance", "coverage", "repeat_stats"]
    plain = spec(generator={"plugin": "fake_datagen"}, splits={"train": {"target": 1, "pool": {"initial": [0, 3]}}})
    assert names(plain) == ["target"]
    assert names(spec(splits={"train": SPLITS["train"], "heldout": SPLITS["heldout"] | {
        "min_distance": {"from": "train", "value": 0.1}}})) == [
        "target", "split_disjoint", "min_distance", "coverage", "repeat_stats"]


def test_the_cli_builds_the_checks_a_spec_names_or_all_applicable():
    assert [c.name for c in cli.build_plugins(spec(), ("check",)).checks] == list(checks.BUILTIN)
    named = spec(checks=[{"plugin": "coverage"}, {"plugin": "repeat_stats", "max_rate": 0.1}])
    built = cli.build_plugins(named, ("check",)).checks
    assert [c.name for c in built] == ["coverage", "repeat_stats"] and built[1].params == {"max_rate": 0.1}


# ------------------------------------------------------------------ report
def test_report_snapshot():
    sp, m = spec(), good()
    m["splits"]["train"]["rejected"] = [{"seed": 9, "step": "generate", "reason": "no path"}]
    m["splits"]["heldout"]["used"][0]["generate"]["realised"][DIM] = 0.241
    m["checks"] = [{"name": n, **Result(*r).to_dict()} for n, r in (("target", (True,)),
                                                                    ("realised_min_distance", (False, "x: 0.041 < 0.1")))]
    assert state.report(sp, m) == f"""# t (r), spec {sp.hash}

| split | target | used | rejected |
|---|---|---|---|
| train | 4 | 4 | 1 |
| heldout | 2 | 2 | 0 |

cells of train (bins {DIM} 2)

| cell | pool | rejected | used | realised | exhausted |
|---|---|---|---|---|---|
| 0 | 2 | 0 | 2 | 2 |  |
| 1 | 2 | 0 | 2 | 2 |  |

| split | from | min distance | sampled | realised |
|---|---|---|---|---|
| heldout | train | 0.1 (realised) | 0.100 | 0.041 |

| check | verdict | detail |
|---|---|---|
| target | PASS |  |
| realised_min_distance | FAIL | x: 0.041 < 0.1 |
"""


# ------------------------------------------------------------------ `check` exit code
@pytest.mark.parametrize("x, code", [(0.0, 0), (0.241, 1)])
def test_check_exits_non_zero_on_a_failed_check(tmp_path, capsys, x, code):
    raw = RAW | {"out": str(tmp_path / "run"), "checks": [{"plugin": "realised_min_distance"}]}
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    m = good() | {"spec_hash": S.parse(raw).hash, "task": "t"}
    m["splits"]["heldout"]["used"][0]["generate"]["realised"][DIM] = x
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "state.json").write_text(json.dumps(m))
    assert cli.main([str(tmp_path / "spec.yaml"), "check"]) == code
    assert ("FAIL" in (tmp_path / "run" / "report.md").read_text()) == bool(code)
    assert json.loads((tmp_path / "run" / "state.json").read_text())["checks"][0]["ok"] == (code == 0)


# ------------------------------------------------------------------ the walk rejects a too-close realised start
def test_too_close_is_per_seed_and_needs_a_comparable_start():
    sp, m = spec(), good()
    held = sp.split("heldout")
    dims = sp.dims()
    assert separation.too_close(held, dims, Result(realised={DIM: 0.241}), m) == "min_distance (realised)"
    assert separation.too_close(held, dims, Result(realised={DIM: 0.0}), m) is None
    assert separation.too_close(held, dims, Result(realised={}), m) is None               # nothing reported
    assert separation.too_close(held, dims, Result.fail("x"), m) is None
    assert separation.too_close(sp.split("train"), dims, Result(realised={DIM: 0.2}), m) is None   # no min_distance


def test_the_walk_replaces_a_realised_rejection_in_its_cell(tmp_path):
    import fake_datagen as F
    F.CALLS.clear()
    realised = {0: -0.4, 1: 0.4, 20: -0.38, 21: 0.0, 22: 0.1}
    raw = RAW | {"checks": [{"plugin": "realised_min_distance"}],
                 "splits": {"train": {"target": 2, "pool": {"initial": [0, 3]}},
                            "heldout": SPLITS["heldout"] | {"target": 2, "pool": {"initial": [20, 29]}}}}
    sp = S.parse(raw)
    plugins = cli.build_plugins(sp, ("sample", "generate", "check"))
    plugins.instancer = F.FakeInstancer({s: {DIM: x} for s, x in realised.items()})        # the start of each scene instance
    p = pipeline.Pipeline(sp, plugins, tmp_path, log=lambda m: None)
    assert p.run(["sample", "generate", "check"])
    rec = json.loads((tmp_path / "state.json").read_text())["splits"]["heldout"]
    assert rec["rejected"] == [{"seed": 20, "step": "realised", "reason": "min_distance (realised)"}]
    assert [e["config"]["seed"] for e in rec["used"]] == [21, 22]


def test_realised_choices_count_in_the_distance_too():
    """grasp only: a seed whose realised grasp equals a train one is `too close`, a different grasp is not."""
    dims = {"grasp.id": Choices((0, 1, 2))}
    sp = S.parse(RAW | {"randomisers": {"grasp": {"id": {"choices": [0, 1, 2]}}},
                        "splits": {"train": {"target": 1, "pool": {"initial": [0, 9]}}, "heldout": SPLITS["heldout"]}})
    held = sp.split("heldout")
    m = {"splits": {"train": {"used": [{"config": {"seed": 0, "values": {"grasp.id": 1}},
                                        "generate": {"realised": {"grasp.id": 1}}}]}}}
    assert separation.too_close(held, sp.dims(), Result(realised={"grasp.id": 1}), m) == "min_distance (realised)"
    assert separation.too_close(held, sp.dims(), Result(realised={"grasp.id": 2}), m) is None
    assert separation.varied(dims, {"grasp.id": 2}) == dims and separation.varied({"g": Choices((4,))}, {"g": 4}) == {}
