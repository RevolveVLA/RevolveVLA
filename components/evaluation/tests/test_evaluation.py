"""Real rollouts (see conftest.py): oracle on seed 0, the test checkpoint on seeds 0 and 1 under the wrapper."""
import json
import sys
from pathlib import Path

import docker
import numpy as np

from conftest import REPO, SEEDS, TASK, load

PHASES = ["approach", "grasp", "open", "release"]


def recovery():
    sys.path.insert(0, str(REPO / "components" / "recovery"))
    import recover
    return recover


def first_tries(runs):
    return {int(s): REPO / t["r0"]["dir"] for s, t in runs["heldout"]["table"].items()}


def test_oracle_passes_every_phase(runs):
    o = runs["oracle"]
    row = o["table"]["0"]["r0"]
    assert o["all_pass"] and row["passed_first_try"] and row["first_try"] == {p: "pass" for p in PHASES}, row
    d = REPO / row["dir"]
    rec = load(d / "rollout.json")
    sent = load(d / "client" / "dummy_policy.json")
    relay = load(d / "relay.json")
    # the actions crossed the ROS path: client container -> DDS -> relay -> gz, none dropped
    assert rec["actions_accepted"] > 100 and relay["actions_dropped"] == 0
    assert 0 <= relay["actions"] - rec["actions_accepted"] <= 2          # in flight when the run ended
    assert sent["sent"] >= relay["actions"] and rec["end_reason"] == "success"
    assert all(len(v) == 3 for v in [relay["images"]]) and min(relay["images"].values()) > 100   # 3 cameras rendered
    rep = load(d / "phase_report.json")
    assert rep["result"]["failed_phase"] is None and rep["accepted"]
    for p in PHASES:
        assert (d / "phase_states" / f"{p}_start.npz").is_file()


def test_policy_rollouts_verdicts_and_phase_states(runs):
    h = runs["heldout"]
    assert sorted(h["table"]) == [str(s) for s in SEEDS] and h["rollouts"] == len(SEEDS) and h["errors"] == 0
    for seed, d in first_tries(runs).items():
        rec, rep, ep = load(d / "rollout.json"), load(d / "phase_report.json"), load(d / "episode.json")
        assert set(rec["phases"]) == set(PHASES) and all(v in ("pass", "fail", "not_reached") for v in rec["phases"].values())
        assert rec["runner"]["accepted"] and rec["server"]["inferences"] > 0 and rec["actions_accepted"] > 0, rec
        assert rec["sim_s"] <= rec["sim_budget_s"] + 0.1
        assert rep["result"]["failed_phase"] == rec["failed_phase"]
        assert ep["task"] == TASK and ep["seed"] == seed and "policy_rollout" in ep and "grasp_id" in ep
        assert (ep["plan_run"], ep["attempt_id"]) == (runs["starts"][seed]["plan_run"], runs["starts"][seed]["attempt_id"])
        reached = [p for p in PHASES if rec["phases"][p] != "not_reached"]
        for p in reached:
            for f in (f"{p}_start.npz", f"{p}_start_next.npz"):
                assert (d / "phase_states" / f).is_file(), (d, f)


def test_failed_rollout_is_recovery_input(runs):
    R = recovery()
    failed = {s: d for s, d in first_tries(runs).items() if load(d / "rollout.json")["failed_phase"]}
    assert failed, "an untrained checkpoint with a 6 s budget should fail a phase"
    for seed, d in failed.items():
        rec = load(d / "rollout.json")
        src = R.source(d, TASK)                   # recovery's own loader of a rollout dir
        assert src["failed_phase"] == rec["failed_phase"] and src["phase"] == rec["failed_phase"]
        assert src["policy"] and src["seed"] == seed and src["regenerate"] is True
        assert (src["plan_run"], src["attempt_id"], src["grasp_id"]) == (
            runs["starts"][seed]["plan_run"], runs["starts"][seed]["attempt_id"], runs["starts"][seed]["grasp_id"])
        bad = R.unsupported(src, list(R.REGENERABLE))     # recovery's precheck
        if rec["failed_phase"] in R.REGENERABLE:
            assert bad is None, bad
        else:
            assert bad[0] == "unsupported"
        snap = np.load(d / "phase_states" / f"{src['phase']}_start.npz")
        assert int(snap["action_index"]) >= 0 and "joint_pos" in snap.files


def test_teardown_clean(runs):
    dk = docker.from_env()
    assert dk.containers.list(all=True, filters={"label": "revolve.rollout"}) == []
    assert [n.name for n in dk.networks.list() if n.name.startswith("rollout-")] == []
    for d in [REPO / runs["oracle"]["table"]["0"]["r0"]["dir"], *first_tries(runs).values()]:
        assert load(d / "jobs" / "sim" / "result.json")["state"] in ("succeeded", "failed")


def test_same_seed_same_start(runs):
    import starts as S
    again = S.resolve(TASK, SEEDS, runs["ckpt"].parent / "starts", sources=[])
    assert again == runs["starts"]                           # cached record, sha-checked
    o = REPO / runs["oracle"]["table"]["0"]["r0"]["dir"]
    p = first_tries(runs)[0]
    for d in (o, p):                                         # both rollouts spawned at the seed's start
        cl = load(d / "closed_loop.json")
        assert cl["spawn"] == runs["starts"][0]["spawn"]
    a, b = (np.load(d / "phase_states" / "approach_start.npz") for d in (o, p))
    # the state saved at the first step (before any policy action acts) is the same for both policies
    assert np.array_equal(a["joint_pos"], b["joint_pos"]) and np.array_equal(a["link_pos"], b["link_pos"])
    assert runs["starts"][0]["spawn"] != runs["starts"][1]["spawn"]


def test_retries_restore_the_phase_start(runs):
    rows = [(s, t["r0"]) for s, t in runs["heldout"]["table"].items()]
    retried = [(s, r) for s, r in rows if not r["passed_first_try"]]
    assert retried
    for s, r in retried:
        assert r["tries"] == 2 and r["retries"][0]["restored_at"] == r["failed_phase"]
        d0, d1 = REPO / r["dir"], REPO / r["retries"][0]["dir"]
        ph = r["failed_phase"]
        assert (d1 / "restore" / f"{ph}_start.npz").read_bytes() == (d0 / "phase_states" / f"{ph}_start.npz").read_bytes()
        rep = load(d1 / "phase_report.json")
        assert rep["restore_check"]["pass_"], rep["restore_check"]["vs_boundary_snapshot"]
        assert rep["boundaries"][ph]["step"] == 0
        assert rep["boundaries"][ph]["action_index"] == int(np.load(d0 / "phase_states" / f"{ph}_start.npz")["action_index"])
        assert list(rep["phases"]) == PHASES[PHASES.index(ph):]
        assert load(d1 / "rollout.json")["restore"]["phase"] == ph


def test_wrapper_status_progress(runs):
    assert runs["wrapper_rc"] == 0 and runs["wrapper_result"]["state"] == "succeeded"
    prog = runs["wrapper_result"]["progress"]
    assert prog["rollouts"] == len(SEEDS) and prog["rollouts_done"] == len(SEEDS)
    assert prog["passed"] == runs["heldout"]["passed_with_retries"] and "last_rollout" in prog
    live = [s["progress"] for s in runs["wrapper_status"] if s["state"] == "running"]
    assert live and any(p.get("running") for p in live)          # /status answered while rollouts ran
    print(json.dumps(dict(evals_wall_s=runs["evals_wall_s"], heldout_rtf=runs["heldout"]["rtf"],
                          gpu=runs["heldout"]["gpu_memory"]), indent=None))
