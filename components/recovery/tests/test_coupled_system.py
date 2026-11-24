"""System tests of the recovery coupled to an evaluation (`recovery:` in the eval spec): `evaluation.cli.main` in this process on fake
sets, scripted rollouts with fake bags, the real recovery core (local scanner over the fake bags, fake runner and generator).
Covered: the jobs run on the evaluation's worker slots behind the pending rollouts, never more than `workers` things at a time, the
quota, the manifest / report, the `recover` stage and resume, the live recorder in the evaluation's own out, and the default (no
`recovery:`, or a disabled one) leaving the evaluation as it was."""
import json

import fake_coupled as C
import fake_live as L
import fake_recovery as F
import pytest
import yaml
from evaluation import cli as ecli
from fake_live import StateStub
from recovery.plugins.live import dagger

TRAIN = {"task": "open_fridge_door", "robot": "mobile_panda", "generator": {"plugin": "fake_datagen"},
         "randomisers": {"fake_datagen": {"x": [-0.5, 0.5], "y": [1, 2]}},
         "splits": {"train": {"target": 3, "pool": {"initial": [0, 9]}, "sampler": {"strategy": "stratified", "bins": {"fake_datagen.x": 3}}},
                    "aside": {"target": 2, "pool": {"initial": [20, 24]}, "record": False}}}
SCRIPT = [F.Try(seed=100 + i, failed="approach") for i in range(3)] + [F.Try(seed=103 + i, failed="grasp") for i in range(2)] + [F.Try(seed=105)]


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    C.reset()
    L.LOG.clear()
    L.SCRIPTS.clear()
    monkeypatch.setattr(dagger, "RosState", StateStub)
    C.TRIES.update({t.seed: t for t in SCRIPT})


def write_recovery(root, name="rec.yaml", **edits):
    raw = {"task": "open_fridge_door", "robot": "mobile_panda", "from": {"eval": "unused", "sets": ["scored"]}, "out": "unused",
           "phases": {"generator": {"plugin": "probe_recovery", "attempts": 2}, "quota": {"approach": 2, "grasp": 1}},
           "scanner": {"plugin": "local", "reader": "fake_recovery:read_messages"}, "runner": {"plugin": "fake"}}
    (root / name).write_text(yaml.safe_dump({**raw, **edits}))
    return root / name


def write_eval(root, recovery="rec.yaml", workers=1, name="eval.yaml", out="runs/eval/e", **edits):
    (root / "train.yaml").write_text(yaml.safe_dump(TRAIN | {"out": str(root / "train_run")}))
    raw = {"train": str(root / "train.yaml"), "out": out, "policy": {"plugin": "fake_policy"}, "rollout": {"plugin": "fake_bags"},
           "sets": {"scored": {"count": 6, "pool": {"initial": [100, 105]}, "plugin": "fake_scenes"}}, "run": {"workers": workers}}
    if recovery:
        raw["recovery"] = {"spec": str(root / recovery)}
    (root / name).write_text(yaml.safe_dump({**raw, **edits}))
    return root / name


def evaluate(root, *args, spec=None, capsys=None):
    code = ecli.main([str(spec or root / "eval.yaml"), *args], root=root)
    return (code, *(capsys.readouterr() if capsys else ()))


def out_of(root):
    return root / "runs" / "eval" / "e"


def manifest(root):
    return json.loads((out_of(root) / "recovery" / "manifest.json").read_text())


def test_failures_of_the_rollouts_are_recovered_on_the_evaluations_slots_into_its_recovery_dir(tmp_path, capsys):
    write_recovery(tmp_path)
    write_eval(tmp_path)
    code, out, err = evaluate(tmp_path, capsys=capsys)
    assert code == 0, err
    m = manifest(tmp_path)
    assert sorted((e["recovery"]["phase"], e["seed"]) for e in m["episodes"]) == [("approach", 0), ("approach", 1), ("grasp", 2)]
    assert {e["recovery"]["kind"] for e in m["episodes"]} == {"phase"} and all(e["passed"] and e["bag"] for e in m["episodes"])
    assert {e["recovery"]["from_rollout"]["rollout"] for e in m["episodes"]} == {"100_r0", "101_r0", "103_r0"}     # first come
    assert all(e["recovery"]["from_rollout"]["run"] == str(out_of(tmp_path)) for e in m["episodes"])
    state = json.loads((out_of(tmp_path) / "recovery" / "state.json").read_text())
    assert state["scanned"] == [f"scored/{s}_r0" for s in range(100, 106)] and len(state["failures"]) == 5
    assert state["scan"] == {"failed": 5, "passed": 1} and len(state["selection"]) == 3
    assert "check jobs: PASS" in out and "check quota: PASS" in out and out.count("recover scored:") == 6
    report = (out_of(tmp_path) / "report.md").read_text()
    assert "## Recovery" in report and "phase-wise: 3 jobs selected, 3 done" in report
    assert "| approach | 3 | 0 | 2 | 2 | 2 | 2 | 100% |" in (out_of(tmp_path) / "recovery" / "report.md").read_text()
    for e in m["episodes"]:
        assert (out_of(tmp_path) / "recovery" / e["scene"]).is_file() and (out_of(tmp_path) / "recovery" / e["result"]).is_file()


BLIND = {"count": 3, "pool": {"initial": [300, 304]}, "plugin": "fake_scenes"}


def test_the_failures_of_a_held_out_set_are_counted_not_scanned_and_never_recovered(tmp_path):
    C.TRIES.update({300 + i: F.Try(seed=300 + i, failed="approach") for i in range(3)})
    write_recovery(tmp_path)                                                    # mines `scored`; `blind` (held-out kind) stays unseen
    write_eval(tmp_path, sets={"scored": {"count": 6, "pool": {"initial": [100, 105]}, "plugin": "fake_scenes"}, "blind": BLIND})
    assert evaluate(tmp_path)[0] == 0
    m = manifest(tmp_path)
    assert {e["recovery"]["from_rollout"]["set"] for e in m["episodes"]} == {"scored"} and len(m["episodes"]) == 3
    state = json.loads((out_of(tmp_path) / "recovery" / "state.json").read_text())
    assert state["scan"] == {"failed": 5, "passed": 1, "held_out": 3}         # the blind set's 3 failures were counted, not read
    assert state["mining"] == {"mined": ["scored"], "held_out": ["blind"]}
    assert all(f["set"] == "scored" for f in state["failures"])
    assert "3 held-out, not mined" in (out_of(tmp_path) / "recovery" / "report.md").read_text()


def test_a_set_inside_the_pool_the_training_keeps_aside_is_refused_before_anything_runs(tmp_path, capsys):
    write_recovery(tmp_path)
    write_eval(tmp_path, sets={"scored": {"count": 3, "pool": {"initial": [22, 29]}, "plugin": "fake_scenes"}})
    code, _, err = evaluate(tmp_path, capsys=capsys)
    assert code == 2 and "overlaps training split 'aside'" in err and not out_of(tmp_path).exists()


def test_with_only_held_out_sets_nothing_is_recovered_and_the_report_says_why(tmp_path, capsys):
    write_recovery(tmp_path, **{"from": {"eval": "unused"}})                    # default: the sets that are not held-out; there are none
    write_eval(tmp_path)
    code, out, _ = evaluate(tmp_path, capsys=capsys)
    assert code == 1 and "check quota: FAIL" in out and manifest(tmp_path)["episodes"] == []
    state = json.loads((out_of(tmp_path) / "recovery" / "state.json").read_text())
    assert state["failures"] == [] and state["scan"] == {"held_out": 5} and "scanned" in state
    assert "Nothing is recorded: every set of the run is held-out" in (out_of(tmp_path) / "recovery" / "report.md").read_text()


def test_no_job_starts_while_a_rollout_is_pending_and_never_more_than_the_workers_run(tmp_path):
    write_recovery(tmp_path)
    write_eval(tmp_path, workers=2, rollout={"plugin": "fake_bags", "delay_s": 0.25})              # long enough for both slots to be busy together
    assert evaluate(tmp_path)[0] == 0
    kinds = [e.split()[1] for e in C.EVENTS if e.startswith("start")]
    assert kinds[:6] == ["rollout"] * 6 and kinds[6:] == ["job"] * 3                    # the rollouts are all taken before a job runs
    assert C.PEAK[0] == 2                                                              # rollouts and jobs share the two slots


@pytest.mark.parametrize("workers", [1, 3])
def test_the_workers_do_not_change_what_is_recorded(tmp_path, workers):
    write_recovery(tmp_path)
    write_eval(tmp_path, workers=workers)
    assert evaluate(tmp_path)[0] == 0
    assert [(e["seed"], e["scene"], e["recovery"]["from_rollout"]["rollout"]) for e in manifest(tmp_path)["episodes"]] == [
        (0, "approach/scored_100_r0_t0/scene.yaml", "100_r0"), (1, "approach/scored_101_r0_t0/scene.yaml", "101_r0"),
        (2, "grasp/scored_103_r0_t0/scene.yaml", "103_r0")]
    assert C.PEAK[0] <= workers


def test_a_rerun_resumes_and_runs_nothing_again(tmp_path):
    write_recovery(tmp_path)
    write_eval(tmp_path)
    evaluate(tmp_path)
    first, events = manifest(tmp_path), list(C.EVENTS)
    assert evaluate(tmp_path)[0] == 0 and C.EVENTS == events and manifest(tmp_path)["episodes"] == first["episodes"]


def test_the_recover_stage_takes_what_the_rollouts_left_and_judges_it(tmp_path, capsys):
    write_recovery(tmp_path)
    write_eval(tmp_path)
    evaluate(tmp_path, "sample")
    assert evaluate(tmp_path, "rollout")[0] == 0
    first = manifest(tmp_path)
    import shutil
    shutil.rmtree(out_of(tmp_path) / "recovery")                                       # as if the jobs had never been taken
    C.EVENTS.clear()
    code, out, err = evaluate(tmp_path, "recover", capsys=capsys)
    assert code == 0, err
    assert [e.split()[1] for e in C.EVENTS if e.startswith("start")] == ["job"] * 3     # no rollout runs in this stage
    assert [e["recovery"]["from_rollout"]["rollout"] for e in manifest(tmp_path)["episodes"]] == [
        e["recovery"]["from_rollout"]["rollout"] for e in first["episodes"]] and "check quota: PASS" in out


def test_a_quota_the_failures_cannot_fill_fails_the_recover_stage(tmp_path, capsys):
    write_recovery(tmp_path, phases={"generator": {"plugin": "probe_recovery", "attempts": 2}, "quota": {"approach": 5}})
    write_eval(tmp_path)
    code, out, _ = evaluate(tmp_path, capsys=capsys)
    assert code == 1 and "check quota: FAIL episodes short of the quota: approach 3/5" in out


def test_warm_resets_and_the_runner_slots_follow_the_evaluations_run_block(tmp_path):
    write_recovery(tmp_path, run={"workers": 7, "reset": "relaunch"})                  # the recovery spec's own run block is not used
    write_eval(tmp_path, run={"workers": 1, "reset": "warm"})
    assert evaluate(tmp_path)[0] == 0
    assert [e["recovery"]["reset"]["mode"] for e in manifest(tmp_path)["episodes"]] == ["relaunch", "warm", "warm"]


def test_the_default_leaves_the_evaluation_as_it_was(tmp_path):
    write_eval(tmp_path, recovery=None)
    assert evaluate(tmp_path)[0] == 0
    assert not (out_of(tmp_path) / "recovery").exists() and "## Recovery" not in (out_of(tmp_path) / "report.md").read_text()
    assert [e for e in C.EVENTS if e.startswith("start")] == [f"start rollout {s}" for s in range(100, 106)]
    write_recovery(tmp_path)
    write_eval(tmp_path, name="off.yaml", out="runs/eval/off", recovery="rec.yaml")
    raw = yaml.safe_load((tmp_path / "off.yaml").read_text())
    raw["recovery"]["enabled"] = False
    (tmp_path / "off.yaml").write_text(yaml.safe_dump(raw))
    assert evaluate(tmp_path, spec=tmp_path / "off.yaml")[0] == 0 and not (tmp_path / "runs/eval/off/recovery").exists()


def test_a_recovery_spec_for_another_task_or_with_unknown_phases_is_a_spec_error(tmp_path, capsys):
    write_recovery(tmp_path, task="open_drawer", robot="mobile_so101_xl")
    write_eval(tmp_path)
    code, _, err = evaluate(tmp_path, capsys=capsys)
    assert code == 2 and "the spec is for open_drawer / mobile_so101_xl" in err and not out_of(tmp_path).exists()
    write_recovery(tmp_path, phases={"generator": {"plugin": "probe_recovery"}, "quota": {"nope": 1}})
    code, _, err = evaluate(tmp_path, capsys=capsys)
    assert code == 2 and "unknown phase ['nope']" in err
    write_eval(tmp_path, recovery="missing.yaml")
    assert evaluate(tmp_path, capsys=capsys)[0] == 2


# ------------------------------------------------------------------ the live recorder in the evaluation's own run
PHASE_TIME = {"plugin": "phase_time", "phase": "approach", "after_s": 1.5}


def live_eval(tmp_path, **edits):
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(TRAIN | {"out": str(tmp_path / "train_run")}))
    raw = {"train": str(tmp_path / "train.yaml"), "out": "runs/eval/e", "policy": {"plugin": "frozen"}, "rollout": {"plugin": "fake_client"},
           "sets": {"dagger": {"count": 3, "pool": {"initial": [200, 209]}, "plugin": "fake_set"},
                    "scored": {"count": 2, "pool": {"initial": [100, 104]}, "plugin": "fake_set"}},
           "recovery": {"spec": str(tmp_path / "live.yaml")}}
    (tmp_path / "eval.yaml").write_text(yaml.safe_dump({**raw, **edits}))


def live_spec(tmp_path, **edits):
    raw = {"task": "open_fridge_door", "robot": "mobile_panda", "from": {"eval": "unused", "sets": ["dagger"]}, "out": "unused",
           "live": {"plugin": "dagger", "expert": {"plugin": "fake_takeover"}, "triggers": [PHASE_TIME]}}
    (tmp_path / "live.yaml").write_text(yaml.safe_dump({**raw, **edits}))


def test_the_policy_runs_under_the_recorder_in_the_live_sets_only_and_the_report_separates_the_takeovers(tmp_path, capsys):
    live_spec(tmp_path)
    live_eval(tmp_path)
    code, out, err = evaluate(tmp_path, capsys=capsys)
    assert code == 0, err
    results = json.loads((out_of(tmp_path) / "results.json").read_text())["sets"]
    assert all(r["tries"][0]["record"]["passed"] for r in results["dagger"]["rollouts"].values())
    sidecars = sorted(p.parent.parent.name + "/" + p.parent.name for p in out_of(tmp_path).rglob("*.interventions.json"))
    assert sidecars == ["dagger/200_sim", "dagger/201_sim", "dagger/202_sim"]               # `scored` ran the plain policy
    assert [e for e in L.LOG if e[0] == "takeover"] and len([e for e in L.LOG if e[0] == "takeover"]) == 3
    m = manifest(tmp_path)
    assert sorted(e["seed"] for e in m["episodes"]) == [200, 201, 202] and {e["recovery"]["kind"] for e in m["episodes"]} == {"dagger"}
    assert all(e["recovery"]["from_rollout"]["run"] == str(out_of(tmp_path)) and (out_of(tmp_path) / "recovery" / e["result"]).is_file()
               for e in m["episodes"])
    report = (out_of(tmp_path) / "report.md").read_text()
    assert "the policy alone passed 0 of 3 episodes in the live sets; 3 taken over (phase_time 3), 3 of them passed with the expert" in report
    summary = json.loads((out_of(tmp_path) / "recovery" / "live_summary.json").read_text())
    assert (summary["takeovers"], summary["recovered"]) == (3, 3) and "live {" in out


def test_one_spec_may_have_phases_and_live_when_coupled(tmp_path, capsys):
    live_spec(tmp_path, phases={"generator": {"plugin": "probe_recovery"}, "quota": {"approach": 1}},
              scanner={"plugin": "local", "reader": "fake_recovery:read_messages"}, runner={"plugin": "fake"})
    live_eval(tmp_path, sets={"dagger": {"count": 3, "pool": {"initial": [200, 209]}, "plugin": "fake_set"}})
    code, out, err = evaluate(tmp_path, capsys=capsys)
    assert code == 1 and "check quota: FAIL episodes short of the quota: approach 0/1" in out        # every episode passed (the expert): no failure
    state = json.loads((out_of(tmp_path) / "recovery" / "state.json").read_text())
    assert state["scan"] == {"passed": 3} and state["failures"] == [] and state["scanned"] == [f"dagger/{s}_r0" for s in (200, 201, 202)]
    assert sorted(e["seed"] for e in manifest(tmp_path)["episodes"]) == [200, 201, 202]              # the takeovers, in the same manifest


def test_the_leakage_guard_applies_to_the_coupled_live_sets(tmp_path, capsys):
    live_spec(tmp_path, **{"from": {"eval": "unused", "sets": ["dagger", "scored"], "heldout": ["scored"]}})
    live_eval(tmp_path)
    code, _, err = evaluate(tmp_path, capsys=capsys)
    assert code == 2 and "also in from.heldout" in err
    live_spec(tmp_path)
    live_eval(tmp_path, sets={"dagger": {"count": 2, "pool": {"initial": [22, 29]}, "plugin": "fake_set"}})        # inside the aside pool
    code, _, err = evaluate(tmp_path, capsys=capsys)
    assert code == 2 and "overlaps training split 'aside'" in err
