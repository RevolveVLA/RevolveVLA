"""Oracle rollout of a second robot: the mobile SO-101 on the drawer. evaluate.py --policy oracle on the seed of
the recovery tests' cached SO-101 datagen episode (runs/test_cache/recovery/mobile_so101/datagen: run
`./revolve test recovery` first): the episode's own actions through the ROS 2 path (client container -> DDS ->
ros_relay -> closed_loop_system, the robot's 9-wide contract) and the live phase validator; every phase must pass
and the rollout's containers and network must be gone. This session's conftest names the Panda, so evaluate.py runs
as a host subprocess with REVOLVE_ROBOT=mobile_so101. Starts cache: runs/test_cache/evaluation/mobile_so101/starts;
outputs: runs/evaluation_tests/<stamp>/mobile_so101/."""
import json
import os
import subprocess
import sys

import docker
import pytest
import yaml
from conftest import CACHE, COMP, REPO, RUN, load

ROBOT = "mobile_so101"
TASK = "open_drawer"
DATAGEN = REPO / "runs" / "test_cache" / "recovery" / ROBOT / "datagen"
OUT = RUN / ROBOT
PHASES = ["approach", "grasp", "open", "release"]


@pytest.fixture(scope="module")
def oracle():
    if not (DATAGEN / "manifest.json").is_file() or not load(DATAGEN / "manifest.json").get("episodes"):
        pytest.fail(f"no {DATAGEN}: run ./revolve test recovery first", pytrace=False)
    m = load(DATAGEN / "manifest.json")
    seed = load(DATAGEN / "episodes" / m["episodes"][0] / "episode.json")["datagen"]["seed"]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(REVOLVE_ROBOT=ROBOT, PYTHONPATH=f"{REPO / 'common'}:{COMP / 'src' / 'evaluation'}")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "evaluate.log", "w") as f:
        rc = subprocess.run([sys.executable, str(COMP / "evaluate.py"), "--task", TASK, "--seeds", str(seed),
                             "--set", "oracle", "--out", str(OUT), "--starts", str(CACHE / ROBOT / "starts"),
                             "--source", str(DATAGEN), "--policy", "oracle", "--parallel", "1"],
                            cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=1800).returncode
    return dict(rc=rc, seed=seed, summary=load(OUT / "oracle" / "eval_summary.json"))


def test_so101_oracle_passes_every_phase(oracle):
    o = oracle["summary"]
    row = o["table"][str(oracle["seed"])]["r0"]
    assert oracle["rc"] == 0 and o["all_pass"] and row["first_try"] == {p: "pass" for p in PHASES}, row
    d = REPO / row["dir"]
    rec, relay = load(d / "rollout.json"), load(d / "relay.json")
    sent = load(d / "client" / "dummy_policy.json")
    contract = yaml.safe_load((REPO / "specs" / "robots" / ROBOT / "contract.yaml").read_text())
    assert len(contract["actions"]["action"]["select"]) == 9
    assert rec["actions_accepted"] > 100 and relay["actions_dropped"] == 0 and rec["end_reason"] == "success"
    assert 0 <= relay["actions"] - rec["actions_accepted"] <= 2 and sent["sent"] >= relay["actions"]
    assert len(relay["images"]) == 3 and min(relay["images"].values()) > 100
    rep = load(d / "phase_report.json")
    assert rep["result"]["failed_phase"] is None and rep["accepted"]
    assert load(d / "episode.json")["task"] == TASK
    (OUT / "numbers.json").write_text(json.dumps(dict(
        seed=oracle["seed"], actions=rec["actions_accepted"], images=relay["images"], sim_s=rec["sim_s"],
        **{k: o[k] for k in ("all_pass", "rtf", "rollout_wall_s")}), indent=1, default=str))


def test_so101_teardown_clean(oracle):
    dk = docker.from_env()
    assert dk.containers.list(all=True, filters={"label": "revolve.rollout"}) == []
    assert [n.name for n in dk.networks.list() if n.name.startswith("rollout-")] == []
    d = REPO / oracle["summary"]["table"][str(oracle["seed"])]["r0"]["dir"]
    assert load(d / "jobs" / "sim" / "result.json")["state"] in ("succeeded", "failed")
