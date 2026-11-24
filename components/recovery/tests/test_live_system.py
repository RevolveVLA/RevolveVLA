"""System tests of the live path: `recovery.cli.main` (what `./revolve recover <spec> live` calls) rolls an evaluation spec's policy
out (the evaluation's Pipeline, its real policy client against a scripted client, fake sets) under the dagger recorder with a fake
expert: the takeovers, the sidecars, the manifest, the summary, the quota, the leakage guard, resuming and the spec errors."""
import json
import os

import fake_live as L
import pytest
import yaml
from fake_live import SCRIPTS, Script
from recovery import cli
from recovery.core import flags
from recovery.plugins.live import dagger

TRAIN = {"task": "open_fridge_door", "robot": "mobile_panda", "generator": {"plugin": "fake_datagen"},
         "randomisers": {"fake_datagen": {"x": [-0.5, 0.5], "y": [1, 2]}},
         "splits": {"train": {"target": 3, "pool": {"initial": [0, 9]}, "sampler": {"strategy": "stratified", "bins": {"fake_datagen.x": 3}}},
                    "aside": {"target": 2, "pool": {"initial": [20, 24]}, "record": False}}}
PHASE_TIME = {"plugin": "phase_time", "phase": "approach", "after_s": 1.5}


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    L.LOG.clear()
    SCRIPTS.clear()
    monkeypatch.setattr(dagger, "RosState", L.StateStub)


def write_eval(root, sets=None, policy=None):
    (root / "train.yaml").write_text(yaml.safe_dump(TRAIN | {"out": str(root / "train_run")}))
    raw = {"train": str(root / "train.yaml"), "out": str(root / "runs/eval/e"), "policy": policy or {"plugin": "frozen"},
           "rollout": {"plugin": "fake_client"}, "rollouts": 1,
           "sets": sets or {"dagger": {"count": 3, "pool": {"initial": [200, 209]}, "plugin": "fake_set"},
                            "scored": {"count": 2, "pool": {"initial": [100, 104]}, "plugin": "fake_set"}}}
    (root / "eval.yaml").write_text(yaml.safe_dump(raw))
    return root / "eval.yaml"


def write_spec(root, eval_spec=None, live=None, **edits):
    raw = {"task": "open_fridge_door", "robot": "mobile_panda", "from": {"eval": str(eval_spec or write_eval(root)), "sets": ["dagger"]},
           "out": "runs/recovery/live",
           "live": {"plugin": "dagger", "expert": {"plugin": "fake_takeover"}, "triggers": [PHASE_TIME], **(live or {})}}
    (root / "spec.yaml").write_text(yaml.safe_dump({**raw, **edits}))
    return root / "spec.yaml"


def recover(root, spec, *args, capsys):
    code = cli.main([str(spec), *args], root=root)
    return (code, *capsys.readouterr())


def out_of(root):
    return root / "runs" / "recovery" / "live"


def manifest(root):
    return json.loads((out_of(root) / "manifest.json").read_text())


def test_the_policy_is_taken_over_and_every_takeover_is_a_manifest_entry(tmp_path, capsys):
    code, out, err = recover(tmp_path, write_spec(tmp_path), "live", capsys=capsys)
    assert code == 0, err
    summary = json.loads((out_of(tmp_path) / "live" / "summary.json").read_text())
    assert {k: summary[k] for k in ("policy_only_success", "takeovers", "reasons", "recovered", "episodes")} == {
        "policy_only_success": 0, "takeovers": 3, "reasons": {"phase_time": 3}, "recovered": 3, "episodes": 3}
    assert summary["expert_frames_s"] == pytest.approx(3 * 1.5, abs=0.6) and "wall_s" in summary and "live {" in out
    m = manifest(tmp_path)
    assert m["counts"] == {"jobs": 0, "scenes": 3, "attempted": 3, "passed": 3}
    for e in m["episodes"]:
        r = e["recovery"]
        assert (r["kind"], r["phase"], r["trigger"], r["restart_from"], r["segment"], r["policy"]["plugin"]) == (
            "dagger", "approach", "phase_time", "takeover", None, "frozen")
        assert e["passed"] and r["from_rollout"]["set"] == "dagger" and r["expert_failure"] is False and r["max_delta_action"] == 0.02
        result = out_of(tmp_path) / e["result"]
        assert result.is_file() and (out_of(tmp_path) / e["bag"]).is_dir()
        side = json.loads(flags.sidecar_of(result).read_text())
        assert side["trigger"] == "phase_time" and side["segments"][-1][2] == "expert" and r["t_restart"] == side["takeover_t"]
    assert sorted(e["seed"] for e in m["episodes"]) == [200, 201, 202]
    assert "dagger" in (out_of(tmp_path) / "live" / "report.md").read_text()


def test_dagger_entries_store_relative_paths(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("recovery.core.manifest.REPO", tmp_path)           # the test's runs stand in for the repo's
    recover(tmp_path, write_spec(tmp_path), "live", capsys=capsys)
    entries = manifest(tmp_path)["episodes"]
    assert len(entries) == 3
    for e in entries:
        r = e["recovery"]["from_rollout"]
        assert not any(os.path.isabs(p) for p in (e["scene"], e["result"], e["bag"], r["scene"], r["bag"]))
        assert (out_of(tmp_path) / e["scene"]).is_file() and (tmp_path / r["scene"]).is_file()    # out-relative / repo-relative
        assert (tmp_path / r["bag"]).is_dir()


def test_a_policy_that_passes_alone_is_no_takeover_and_no_manifest_entry(tmp_path, capsys):
    SCRIPTS[201] = Script(only_phase="grasp", passes_alone=True, end_s=5.0)
    recover(tmp_path, write_spec(tmp_path), "live", capsys=capsys)
    summary = json.loads((out_of(tmp_path) / "live" / "summary.json").read_text())
    assert (summary["policy_only_success"], summary["takeovers"], summary["episodes"]) == (1, 2, 3)
    assert sorted(e["seed"] for e in manifest(tmp_path)["episodes"]) == [200, 202]
    side = flags.read(next((out_of(tmp_path) / "live").rglob("201_sim/episode_0.json")))
    assert side["takeover_t"] is None and side["segments"] == [[0.0, 3.0, "policy"]]


def test_each_episode_of_a_sim_has_its_own_sidecar_next_to_its_result(tmp_path, capsys):
    eval_spec = write_eval(tmp_path)
    raw = yaml.safe_load(eval_spec.read_text())
    raw["rollouts"] = 2                                                    # two episodes of one config share a sim folder
    eval_spec.write_text(yaml.safe_dump(raw))
    code, _, err = recover(tmp_path, write_spec(tmp_path, eval_spec), "live", capsys=capsys)
    assert code == 0, err
    sides = sorted(p.name for p in (out_of(tmp_path) / "live").rglob("*.interventions.json"))
    assert len(sides) == 6 and len(set(sides)) == 2 and sides[0] == "episode_0.interventions.json"
    assert len({e["result"] for e in manifest(tmp_path)["episodes"]}) == 6


def test_the_quota_stops_the_takeovers_of_a_full_phase_and_the_verdict_stands(tmp_path, capsys):
    code, out, err = recover(tmp_path, write_spec(tmp_path, live={"quota": {"approach": 2}}), "live", capsys=capsys)
    assert code == 0, err
    summary = json.loads((out_of(tmp_path) / "live" / "summary.json").read_text())
    assert summary["takeovers"] == 2 and summary["policy_only_success"] == 0 and summary["episodes"] == 3
    assert len(manifest(tmp_path)["episodes"]) == 2


def test_record_interventions_puts_the_expert_span_in_the_manifest(tmp_path, capsys):
    recover(tmp_path, write_spec(tmp_path, live={"record": "interventions"}), "live", capsys=capsys)
    for e in manifest(tmp_path)["episodes"]:
        t0 = e["recovery"]["t_restart"]
        assert e["recovery"]["segment"] == [t0, 3.0]


def test_keep_successes_leaves_out_a_failed_expert_and_keep_all_keeps_it(tmp_path, capsys):
    SCRIPTS[201] = Script(expert_works=False)
    recover(tmp_path, write_spec(tmp_path), "live", capsys=capsys)
    m = manifest(tmp_path)
    assert sorted(e["seed"] for e in m["episodes"]) == [200, 202]
    out_of(tmp_path).joinpath("state.json").unlink()                       # another spec, the same rollouts (resumed: none re-run)
    recover(tmp_path, write_spec(tmp_path, keep="all"), "live", capsys=capsys)
    both = {e["seed"]: e for e in manifest(tmp_path)["episodes"]}
    assert sorted(both) == [200, 201, 202] and both[201]["passed"] is False and both[201]["recovery"]["expert_failure"] is True


def test_a_rerun_resumes_and_adds_nothing(tmp_path, capsys):
    spec = write_spec(tmp_path)
    recover(tmp_path, spec, "live", capsys=capsys)
    first, starts = manifest(tmp_path), [x for x in L.LOG if x[0] == "start"]
    assert recover(tmp_path, spec, "live", capsys=capsys)[0] == 0
    assert manifest(tmp_path)["episodes"] == first["episodes"] and [x for x in L.LOG if x[0] == "start"] == starts


def test_all_runs_the_stages_a_live_only_spec_has(tmp_path, capsys):
    assert recover(tmp_path, write_spec(tmp_path), capsys=capsys)[0] == 0
    assert len(manifest(tmp_path)["episodes"]) == 3


def test_noisy_expert_is_a_policy_the_recorder_can_supervise(tmp_path, capsys):
    eval_spec = write_eval(tmp_path, policy={"plugin": "noisy_expert", "sigma": 0.02})
    assert recover(tmp_path, write_spec(tmp_path, eval_spec), "live", capsys=capsys)[0] == 0
    assert {e["recovery"]["policy"]["plugin"] for e in manifest(tmp_path)["episodes"]} == {"noisy_expert"}


def test_the_leakage_guard_refuses_scenes_the_policy_is_scored_on(tmp_path, capsys):
    overlapping = {"dagger": {"count": 2, "pool": {"initial": [22, 29]}, "plugin": "fake_set"}}        # inside the training spec's aside pool
    code, _, err = recover(tmp_path, write_spec(tmp_path, write_eval(tmp_path, sets=overlapping)), "live", capsys=capsys)
    assert code == 2 and "overlaps training split 'aside'" in err and not out_of(tmp_path).exists()
    both = write_spec(tmp_path, write_eval(tmp_path), **{"from": {"eval": str(tmp_path / "eval.yaml"), "sets": ["dagger", "scored"],
                                                                  "heldout": ["scored"]}})
    code, _, err = recover(tmp_path, both, "live", capsys=capsys)
    assert code == 2 and "also in from.heldout" in err


def test_a_live_run_needs_an_evaluation_spec_a_streaming_policy_and_known_sets(tmp_path, capsys):
    run_dir = tmp_path / "some_run"
    code, _, err = recover(tmp_path, write_spec(tmp_path, **{"from": {"eval": str(run_dir)}}), "live", capsys=capsys)
    assert code == 2 and "an evaluation spec (a .yaml file)" in err
    code, _, err = recover(tmp_path, write_spec(tmp_path, write_eval(tmp_path, policy={"plugin": "expert"})), "live", capsys=capsys)
    assert code == 2 and "cannot be halted" in err
    code, _, err = recover(tmp_path, write_spec(tmp_path, write_eval(tmp_path), **{"from": {"eval": str(tmp_path / "eval.yaml"),
                                                                                         "sets": ["nope"]}}), "live", capsys=capsys)
    assert code == 2 and "unknown eval set ['nope']" in err
    code, _, err = recover(tmp_path, write_spec(tmp_path), "collect", capsys=capsys)
    assert code == 1 and "no `phases`" in err


def test_dry_run_prints_the_live_plan_and_touches_nothing(tmp_path, capsys):
    code, out, err = recover(tmp_path, write_spec(tmp_path), "--dry-run", capsys=capsys)
    assert code == 0, err
    assert "live: sets dagger of" in out and "under dagger" in out and "unseen: its held-out sets" in out and not out_of(tmp_path).exists()
